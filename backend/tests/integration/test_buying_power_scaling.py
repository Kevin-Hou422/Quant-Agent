"""
资金不足时买单按比例缩量（BUYING_POWER_RULE = scale_buys_pro_rata，取代"整张拒绝"）。

来由：隔离评估（C:\\QuantAgentEnv\\buying-power-eval）显示整张拒绝时平均只投入 76–87% 的权益、几乎每个调仓日都有买单被拒，
而且按代码顺序累计 —— 排在后面的票系统性地买不到。前向证据记录的是这个打了折、有顺序偏差的组合，不是规格里的目标组合。

规则：全部买单要的钱（数量 × 限价）超过可用资金 → 每张 × 同一比例、整股向下取整；剩下的钱按"舍掉的零头从大到小"
逐张补一股。可用资金的口径不变（切片权益 − 已持仓市值 − 在途买单；卖单的钱不预支）—— 不加杠杆。
    · 规则本身：不超资金、不超原数量、比例一致、钱基本用完、与代码顺序无关
    · 下单门：先缩量再过风控（满仓换仓时不再被总敞口上限整张拦下）；缩到 0 才算拒单；在途买单先占资金
    · 券商计划与模拟冻结（逐单版、数组版）在同一份输入上给出同一份委托
    · 每轮计划记录写明原数量 → 实发数量；下一轮记录没买够的那部分怎么重新决策
    · 规则名进可执行规格的执行合同：规则一变，规格哈希与验收合同都变
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.core.execution import order_intent as oi
from app.core.execution.order_builder import build_rebalance_orders
from app.core.execution.pretrade_gate import GateContext, PreTradeLimits
from tests.unit.execution.test_order_manager import D1, D2, _cycle, _env, _placed
from tests.unit.execution.test_pretrade_gate import BIG_ADV, LIM, _ctx, _o, check

WIDE = PreTradeLimits(max_name_weight=1.0, max_gross=2.0, max_participation_pct=.10, max_daily_loss=.5,
                      max_price_deviation=.15)


# ── 规则本身 ─────────────────────────────────────────────

def test_float_noise_does_not_cost_a_share():
    """
    2 股 @0.7、14 股 @0.2（要 4.2），手里 3.3：B 的 14 × 3.3 / 4.2 = 11，浮点里算成 10.999999999999998。
    floor 成 10 也不吃亏：零头最大，第一个补回来 → B 11 股，A 1 股（再补 A 要 3.6 > 3.3）。
    """
    assert 14 * (3.3 / 4.2) < 11
    assert oi.scale_buys_to_power([2, 14], [0.7, 0.2], 3.3, True, ["A", "B"]).tolist() == [1.0, 11.0]


def test_enough_money_changes_nothing():
    q = oi.scale_buys_to_power([50, 30], [100.0, 20.0], 5_600.0, True, ["A", "B"])
    assert q.tolist() == [50.0, 30.0]                                # 恰好够（5000 + 600）也不缩


def test_a_hand_worked_example():
    """
    三张买单各 100 股 @ 10 / 20 / 30（要 6000），手里 3000：比例 0.5 → 各 50 股，花 3000，一分不剩。
    手里 3015：比例 0.5025 → 50.25 股 → 各 50 股（3000），剩 15 → 零头相同按代码补：A 补一股（10），剩 5，B / C 买不起。
    """
    assert oi.scale_buys_to_power([100, 100, 100], [10.0, 20.0, 30.0], 3_000.0, True, list("ABC")).tolist() == [50, 50, 50]
    assert oi.scale_buys_to_power([100, 100, 100], [10.0, 20.0, 30.0], 3_015.0, True, list("ABC")).tolist() == [51, 50, 50]
    exact = oi.scale_buys_to_power([100, 100, 100], [10.0, 20.0, 30.0], 3_015.0, False, list("ABC"))
    assert exact == pytest.approx([50.25] * 3)


@pytest.mark.parametrize("power", [0.0, -5.0, float("nan"), float("inf"), None])
def test_no_usable_money_buys_nothing(power):
    assert oi.scale_buys_to_power([10, 20], [5.0, 5.0], power, True, ["A", "B"]).tolist() == [0.0, 0.0]


@pytest.mark.parametrize("seed", range(60))
def test_the_rule_never_overspends_and_treats_every_buy_alike(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 12))
    q = rng.integers(1, 400, n).astype(float)
    lim = np.round(rng.uniform(0.5, 400.0, n), 2)
    names = [f"T{i:02d}" for i in rng.permutation(n)]
    need = float(np.sum(q * lim))
    power = need * float(rng.uniform(0.02, 0.99))
    got = oi.scale_buys_to_power(q, lim, power, True, names)
    spent = float(np.sum(got * lim))
    ratio = power / need
    assert spent <= power + 1e-6 and (got <= q).all() and (got == np.floor(got)).all()
    # 同一比例：每张都落在 floor(原数量 × 比例) 与它 + 1 之间 —— 没有哪一张因为排在后面而整张买不到
    assert ((got >= np.floor(q * ratio)) & (got <= np.floor(q * ratio) + 1)).all()
    # 钱用完了：没补到那一股的，都是因为再买一股就超资金（或已经是原数量）
    base = np.floor(q * ratio)
    for i in range(n):
        assert got[i] > base[i] or got[i] == q[i] or spent + lim[i] > power + 1e-6
    # 与输入顺序无关（只与代码有关）
    perm = rng.permutation(n)
    again = oi.scale_buys_to_power(q[perm], lim[perm], power, True, [names[j] for j in perm])
    assert again.tolist() == got[perm].tolist()


# ── 下单门 ───────────────────────────────────────────────

def _qty(dec):
    return {o.ticker: o.qty for o in dec.approved}


def test_the_gate_shrinks_every_buy_by_the_same_ratio_instead_of_dropping_the_last():
    """整张拒绝时：A、B 买满、C 一股都没有。现在三张各缩到 2/3；卖单不动；卖出所得仍然不计入。"""
    orders = [_o("S", "SELL", 50, cur=50), _o("A", "BUY", 60, limit=100.0), _o("B", "BUY", 60, limit=100.0),
              _o("C", "BUY", 30, limit=100.0)]                       # 买单要 15000，手里 10000
    dec = check(orders, _ctx(cur={"S": 50}, adv={**BIG_ADV, "S": 1e9}, power=10_000.0), LIM)
    assert _qty(dec) == {"S": 50, "A": 40, "B": 40, "C": 20} and dec.rejected == []
    assert sorted((o.ticker, planned, o.qty) for o, planned in dec.resized) == [("A", 60, 40), ("B", 60, 40), ("C", 30, 20)]
    assert [(r["ticker"], r["planned_qty"], r["qty"]) for r in dec.to_dict()["resized"]] == [
        ("A", 60, 40), ("B", 60, 40), ("C", 30, 20)]


def test_a_buy_shrunk_to_nothing_is_a_rejection_with_its_planned_quantity():
    orders = [_o("A", "BUY", 90, limit=100.0), _o("B", "BUY", 1, limit=100.0)]     # 要 9100，手里 150
    dec = check(orders, _ctx(adv=BIG_ADV, power=150.0), LIM)
    assert _qty(dec) == {"A": 1}
    assert [(o.ticker, o.qty, why) for o, why in dec.rejected] == [("B", 1, "insufficient_buying_power")]


def test_without_a_buying_power_figure_no_buy_is_sent_but_sells_are():
    orders = [_o("S", "SELL", 5, cur=10), _o("A", "BUY", 5)]
    dec = check(orders, _ctx(cur={"S": 10}, adv={**BIG_ADV, "S": 1e9}, power=None), LIM)
    assert _qty(dec) == {"S": 5} and [(o.ticker, why) for o, why in dec.rejected] == [("A", "insufficient_buying_power")]


def test_open_buy_orders_come_out_of_the_money_first():
    from app.core.execution.pretrade_gate import OpenOrder
    ctx = GateContext(equity=100_000.0, buying_power=10_000.0, current_qty={}, broker_prices={}, adv_usd=BIG_ADV,
                      day_return=0.0, open_orders=[OpenOrder("K", "BUY", 60, 100.0)])       # 在途 6000
    dec = check([_o("A", "BUY", 40, limit=100.0), _o("B", "BUY", 40, limit=100.0)], ctx, LIM)
    assert _qty(dec) == {"A": 20, "B": 20}                           # 剩 4000，两张各一半


def test_a_fully_invested_rotation_now_buys_what_the_cash_allows():
    """
    满仓换仓：持有 H（9%），目标换成 A（9%）。手里有 1000 现金。旧规则下买 A 的单按原数量算最坏敞口 —— 缩量发生在
    总敞口检查之后就没有意义；现在先缩到 1000 买得起的股数，再过风控。
    """
    lim = PreTradeLimits(**{**LIM.__dict__, "max_gross": 0.10})
    orders = [_o("H", "SELL", 90, cur=90), _o("A", "BUY", 90, limit=100.0)]
    dec = check(orders, _ctx(cur={"H": 90}, adv={**BIG_ADV, "H": 1e9}, power=1_000.0), lim)
    assert _qty(dec) == {"H": 90, "A": 10} and dec.rejected == []


def test_orders_stopped_by_the_kill_switch_take_no_share_of_the_money():
    from app.core.execution.order_builder import PURPOSE_FLATTEN
    lim = PreTradeLimits(**{**LIM.__dict__, "allow_short": True})
    orders = [_o("A", "BUY", 50, limit=100.0), _o("S", "BUY", 10, cur=-10, limit=100.0, purpose=PURPOSE_FLATTEN)]
    dec = check(orders, _ctx(cur={"S": -10}, adv={**BIG_ADV, "S": 1e9}, power=1_000.0, kill=True), lim)
    assert _qty(dec) == {"S": 10} and [(o.ticker, why) for o, why in dec.rejected] == [("A", "kill_switch_engaged")]
    dec = check(orders, _ctx(cur={"S": -10}, adv={**BIG_ADV, "S": 1e9}, power=500.0, kill=True), lim)
    assert _qty(dec) == {"S": 5}, "全平的买单照样受资金约束（500 / 限价 100 → 5 股）；被熔断拦下的 A 不分钱"


# ── 券商计划 = 模拟冻结 ──────────────────────────────────

@pytest.mark.parametrize("seed", range(40))
def test_the_broker_gate_and_both_simulation_freezes_agree_order_by_order(seed):
    from datetime import datetime, timezone

    from app.core.execution.broker_gateway import Quote
    rng = np.random.default_rng(1000 + seed)
    n = 8
    names = [f"T{i}" for i in rng.permutation(n)]
    px = np.round(rng.uniform(5, 400, n), 2)
    w = rng.dirichlet(np.ones(n)) * rng.uniform(0.6, 1.0)
    cur = np.trunc(np.abs(rng.normal(0, 20, n)))
    eq = float(rng.uniform(2e4, 2e5))
    power = float(eq * rng.uniform(0.02, 0.5))
    tw, ref, held = pd.Series(w, index=names), pd.Series(px, index=names), dict(zip(names, cur))

    built = build_rebalance_orders(tw, held, ref, eq, date(2025, 3, 4), book_id=1, band_bps=50.0)
    ctx = GateContext(equity=eq, buying_power=power, current_qty=held, broker_prices=dict(ref), adv_usd={k: 1e12 for k in names},
                      day_return=0.0, now=datetime(2025, 3, 4, 22, 0, tzinfo=timezone.utc),
                      quotes={k: Quote(float(ref[k]), datetime(2025, 3, 4, 16, 0)) for k in names})
    from app.core.execution.pretrade_gate import check as gate
    dec = gate(built.orders, ctx, PreTradeLimits(max_name_weight=10.0, max_gross=10.0, max_participation_pct=1.0,
                                                 max_daily_loss=.5, max_price_deviation=.15))
    broker = {o.ticker: o.signed_qty for o in dec.approved}

    plan = oi.freeze_rebalance(tw, held, ref, eq, date(2025, 3, 4), band_bps=50.0, book_id=1, buying_power=power)
    one_by_one = {i.ticker: i.signed_qty for i in plan.intents}
    qty, _lim, rej = oi.freeze_arrays(w, cur, px, eq, 0.005, True, power, names)
    arrays = {names[j]: qty[j] for j in range(n) if qty[j] != 0}

    assert broker == one_by_one == arrays
    assert sorted(o.ticker for o, _ in dec.rejected) == sorted(r["ticker"] for r in plan.rejected) \
        == sorted(names[j] for j in np.flatnonzero(rej))
    assert sorted(o.ticker for o, _ in dec.resized) == sorted(r["ticker"] for r in plan.resized)
    spent = sum(o.qty * o.limit_price for o in dec.approved if o.side == "BUY")
    assert spent <= power + 1e-6


def test_the_paper_ledger_records_the_part_it_could_not_afford(tmp_path):
    """纸账本：缩了量的买单，没委托的那部分记成 buying_power_resized（不是悄悄少买）。"""
    from app.core.execution.paper_broker import PaperBroker
    from app.db.position_store import PositionStore
    from tests.integration.test_frozen_order_intent_ledger import _bars, _params
    rows = [[(100, 100, 100, 100), (50, 50, 50, 50)]] * 4
    bars = _bars(rows, names=("A", "B"))
    idx = bars["close"].index
    w = pd.DataFrame({"A": [0.95, 0.95, 0.0, 0.0], "B": [0.0, 0.0, 0.95, 0.95]}, index=idx)
    vol = pd.DataFrame(1e12, index=idx, columns=["A", "B"])
    store = PositionStore(f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    PaperBroker(store=store, cost_params=_params(), initial_capital=100_000.0, limit_band_bps=50.0).replay(
        1, w, bars["close"], vol, ohlc=bars)
    (b,) = [f for f in store.fills_on(1, idx[3].date()) if f.ticker == "B"]
    assert b.reject_reason == "buying_power_resized" and 0 < b.traded_weight < 0.06
    assert b.unfilled_weight == pytest.approx(0.95 - b.traded_weight, abs=0.005)
    (a,) = [f for f in store.fills_on(1, idx[1].date()) if f.ticker == "A"]
    assert a.reject_reason == "" and a.unfilled_weight == 0.0, "钱够的买单照常成交，不标缩量"


# ── 订单管理：实发数量、计划记录、下一轮 ─────────────────────

PX = {"AAA": 50.0, "BBB": 20.0}


def _plans(e):
    return [json.loads(ev.payload) for ev in reversed(e.es.events_of("cycle_plan", 50))]


def test_the_cycle_sends_the_shrunk_quantities_and_records_planned_versus_sent(tmp_path):
    e = _env(tmp_path)
    e.om.limits = WIDE
    rep = _cycle(e, D1, w={"AAA": .6, "BBB": .5}, px=PX)             # 要 60300 + 50250 = 110550，切片 100000
    sent = {tk: qty for tk, _side, qty, *_ in _placed(e)}
    assert sent == {"US.AAA": 1085, "US.BBB": 2262}                 # 1200 × 0.9046 → 1085；2500 × 0.9046 → 2261 + 补 1 股
    (p,) = _plans(e)
    assert {r["ticker"]: (r["planned_qty"], r["qty"]) for r in p["resized"]} == {"AAA": (1200, 1085), "BBB": (2500, 2262)}
    assert p["rejected"] == [] and rep.n_gate_rejected == 0
    c = p["capital"]
    assert 0 <= c["remaining_cash"] < 20.10                          # 剩下的钱连一股 BBB 都买不起
    assert c["utilization_after_buys"] == pytest.approx(1.0, abs=3e-4)


def test_the_next_cycle_records_what_became_of_the_part_not_bought(tmp_path):
    e = _env(tmp_path)
    e.om.limits = WIDE
    _cycle(e, D1, w={"AAA": .6, "BBB": .5}, px=PX)
    e.ctx.now = "2025-03-04 09:30:01.000"
    e.ctx.open_session({"AAA": 50.0, "BBB": 20.0})
    e.ctx.mark(PX)
    e.ctx.now = "2025-03-04 16:30:00.000"
    _cycle(e, D2, w={"AAA": .6, "BBB": .5}, px=PX)
    prev = {r["ticker"]: r for r in _plans(e)[-1]["previous_rejections"]}
    assert set(prev) == {"AAA", "BBB"} and prev["AAA"]["rejected_qty"] == 1200 - 1085
    assert all(r["rejected_on"] == str(D1) for r in prev.values())


# ── 规则进执行合同 ───────────────────────────────────────

def test_the_rule_is_part_of_the_execution_contract(monkeypatch):
    from app.config import settings
    from app.core.execution import acceptance_evidence as ae
    from app.core.portfolio_manager import executable_spec as es
    assert es._execution_section(settings)["buying_power_rule"] == oi.BUYING_POWER_RULE == "scale_buys_pro_rata"
    before = ae.contract_of_settings(settings)
    monkeypatch.setattr(oi, "BUYING_POWER_RULE", "reject_whole_order")
    assert ae.contract_of_settings(settings) != before, "规则变了，旧规则下的现场验收证据不能继承"


# ── 等价变异 ─────────────────────────────────────────────

PROVEN_EQUIVALENT = {
    "app/core/execution/order_intent.py ×1 — scale_buys_to_power 里 `if round(need, 6) <= round(power, 6)` -> `<`":
        "两式只在要的钱恰好等于资金时不同：原式原样返回；变异式往下算，比例 power / need 恰好是 1.0，"
        "exact = 数量 × 1.0 = 原数量（整数），floor 不变，补股时 spent + 限价 > power，一股也不补 —— 结果相同。"
        "见 test_a_mutated_scaling_gives_identical_quantities（同一变异施加到函数副本，随机输入含恰好够钱的情形）。",
}


def test_a_mutated_scaling_gives_identical_quantities():
    import inspect
    import textwrap
    code = textwrap.dedent(inspect.getsource(oi.scale_buys_to_power))
    old, new = "if round(need, 6) <= round(power, 6):", "if round(need, 6) < round(power, 6):"
    a, b = (old, new) if old in code else (new, old)            # 双向：变异工具运行时源码已是变异式
    assert code.count(a) == 1 and b not in code
    ns = dict(vars(oi))
    exec(compile(code.replace(a, b), "<mutant>", "exec"), ns)      # noqa: S102
    other = ns["scale_buys_to_power"]
    rng = np.random.default_rng(3)
    for _ in range(300):
        n = int(rng.integers(1, 6))
        q = rng.integers(1, 50, n).astype(float)
        lim = np.round(rng.uniform(1.0, 50.0, n), 2)
        need = float(np.sum(q * lim))
        for power in (need, need * float(rng.uniform(0.3, 1.0)), need + 1.0):
            for whole in (True, False):
                names = [f"T{i}" for i in range(n)]
                np.testing.assert_array_equal(oi.scale_buys_to_power(q, lim, power, whole, names),
                                              other(q, lim, power, whole, names))


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 1
