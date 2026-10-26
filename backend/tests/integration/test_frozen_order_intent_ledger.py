"""
盘后冻结的订单数量：券商计划、纸账本、研究回测是同一份数量（2026-10-25 审计 R1）。

审计原话："券商计划按 T 日参考价和当时权益冻结整数股数；模拟引擎将昨日目标权重按 T+1 开盘市值重新解释为
开盘目标持仓……现金 10,000，T 收盘 100，目标权重 50%，券商冻结 50 股、买入限价 100.5；次日开盘 90、收盘 100……
冻结订单的净值为 1.05，PaperBroker 却模拟约 55.5556 股，净值 1.0555556。"
验收原话："独立手算账本覆盖向上/向下跳空、原有持仓、现金不足、整数股和价格精度、限价未触及、盘中触及、
部分成交、成本、撤单及剩余数量；包括上述 50 股案例。不能仅比较两个共享实现的引擎。"

这里的手算账本**不调用任何引擎 / 建单代码**：股数、现金、限价的最小价位、买入力规则都在本文件里另写一遍。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.core.backtest_engine.transaction_cost import CostParams

BAND_BPS = 50.0


def _params(fixed_bps=0.0, mpp=1.0):
    return CostParams(fixed_bps=fixed_bps, min_ticket_fee=0.0, spread_bps=0.0, impact_coef=0.0,
                      short_borrow_annual_bps=0.0, max_participation_pct=mpp)


def _tick_limit(ref, side):
    """限价 = 参考价 × (1 ± 50bp)，按最小价位取整：≥$1 为 0.01；买向下、卖向上（手写，不用产品代码）。"""
    raw = ref * (1 + BAND_BPS * 1e-4) if side > 0 else ref * (1 - BAND_BPS * 1e-4)
    tick = 0.01 if raw >= 1.0 else 0.0001
    n = raw / tick
    n = math.floor(n + 1e-9) if side > 0 else math.ceil(n - 1e-9)
    return round(n * tick, 4)


def _fill(side, lim, o, h, lo):
    if side > 0:
        return o if o <= lim else (lim if lo <= lim else None)
    return o if o >= lim else (lim if h >= lim else None)


def _hand_ledger(bars, w, capital, vol=None, mpp=1.0, fee_bps=0.0, long_only_cash=True):
    """
    股数 + 现金。T 收盘：数量 = trunc(w × T 收盘权益 / T 收盘价) − 现有股数（整股）；纯多头时买单按
    "卖单在前、买单按代码顺序"累计 数量 × 限价，超出（T 收盘权益 − T 收盘持仓市值）的买单不发。
    T+1：按限价规则成交；成交量 ≤ floor(截至 T 的 ADV × 参与率 / 开盘价)；DAY 单不成交就作废（剩余量不留到下一天）。
    返回每日收盘净值 / 初始资金，以及逐日成交 {(日期, 代码): (股数, 价格)}。
    """
    close, open_, high, low = bars["close"], bars["open"], bars["high"], bars["low"]
    names = list(close.columns)
    sh = {k: 0.0 for k in names}
    cash = float(capital)
    if vol is not None:
        from app.core.backtest_engine.transaction_cost import LiquidityConstraint
        adv = LiquidityConstraint(_params(mpp=mpp)).compute_adv(vol, close)       # 市场数据：与引擎同口径
    out, fills = [], {}
    for t, d in enumerate(close.index):
        if t > 0:
            c = close.iloc[t - 1]
            eq = cash + sum(sh[k] * c[k] for k in names)
            orders = []
            for k in names:
                tgt = math.trunc(float(w.iloc[t - 1][k]) * eq / c[k])
                delta = tgt - sh[k]
                q = math.floor(abs(delta) + 1e-9)
                if q:
                    side = 1 if delta > 0 else -1
                    orders.append((k, side, q, _tick_limit(c[k], side)))
            orders = [o for o in orders if o[1] < 0] + [o for o in orders if o[1] > 0]
            if long_only_cash:
                power, spent, kept = eq - sum(abs(sh[k] * c[k]) for k in names), 0.0, []
                for o in orders:
                    if o[1] > 0:
                        if round(spent + o[2] * o[3], 6) > round(power, 6):
                            continue
                        spent += o[2] * o[3]
                    kept.append(o)
                orders = kept
            for k, side, q, lim in orders:
                p = _fill(side, lim, open_.loc[d, k], high.loc[d, k], low.loc[d, k])
                if p is None:
                    continue
                if vol is not None:
                    q = min(q, math.floor(adv.iloc[t - 1][k] * mpp / open_.loc[d, k] + 1e-9))
                if q <= 0:
                    continue
                sh[k] += side * q
                cash -= side * q * p + q * p * fee_bps * 1e-4
                fills[(d, k)] = (side * q, p)
        out.append((cash + sum(sh[k] * close.loc[d, k] for k in names)) / capital)
    return pd.Series(out, index=close.index), fills


def _bars(rows, names=("A",)):
    """rows: [(open, high, low, close) per name …] per day。"""
    idx = pd.bdate_range("2025-03-03", periods=len(rows))
    out = {k: pd.DataFrame(index=idx, columns=list(names), dtype=float) for k in ("open", "high", "low", "close")}
    for i, day in enumerate(rows):
        for j, k in enumerate(names):
            o, h, lo, c = day[j]
            out["open"].iloc[i, j], out["high"].iloc[i, j], out["low"].iloc[i, j], out["close"].iloc[i, j] = o, h, lo, c
    return out


def _engines(bars, w, capital, tmp_path, vol=None, mpp=1.0, fee_bps=0.0):
    from app.core.backtest_engine.backtest_engine import BacktestEngine
    from app.core.execution.paper_broker import PaperBroker
    from app.db.position_store import PositionStore
    vol = vol if vol is not None else pd.DataFrame(1e12, index=bars["close"].index, columns=bars["close"].columns)
    p = _params(fixed_bps=fee_bps, mpp=mpp)
    bt = BacktestEngine(cost_params=p, initial_capital=capital, limit_band_bps=BAND_BPS).run(
        w, bars["close"], vol, w, ohlc=bars).equity_curve
    pb = PaperBroker(store=PositionStore(f"sqlite:///{(tmp_path / 'p.db').as_posix()}"), cost_params=p,
                     initial_capital=capital, limit_band_bps=BAND_BPS).replay(1, w, bars["close"], vol, ohlc=bars)
    return np.asarray(bt, dtype=float), np.asarray(pb, dtype=float)


def test_the_fifty_share_case(tmp_path):
    """审计的案例：冻结 50 股 → 开盘 90 买到 → 收盘 100 → 净值 1.05（不是按开盘重算的 55.56 股 / 1.0556）。"""
    bars = _bars([[(100, 100, 100, 100)], [(90, 100, 89, 100)]])
    w = pd.DataFrame({"A": [0.5, 0.5]}, index=bars["close"].index)
    hand, fills = _hand_ledger(bars, w, 10_000.0)
    assert fills[(bars["close"].index[1], "A")] == (50, 90)
    assert hand.iloc[1] == pytest.approx(1.05, abs=1e-12)
    bt, pb = _engines(bars, w, 10_000.0, tmp_path)
    assert bt[1] == pytest.approx(1.05, abs=1e-12) and pb[1] == pytest.approx(1.05, abs=1e-12)


SCENARIOS = {
    # 名字: (bars, 权重, 资金)
    "gap_up_fill_at_limit": (  # 开盘 101 > 限价 100.5，盘中回落触及 → 按限价
        [[(100, 100, 100, 100)], [(101, 102, 100.2, 101.5)], [(101.5, 102, 101, 101.8)]], {"A": 0.5}, 10_000.0),
    "gap_up_not_reached": (    # 开盘 102、全天不回到 100.5 → 不成交；DAY 作废，下一天按新状态重新冻结
        [[(100, 100, 100, 100)], [(102, 103, 101, 102.5)], [(102.5, 103, 102, 102.8)]], {"A": 0.5}, 10_000.0),
    "gap_down_fill_at_open": (
        [[(100, 100, 100, 100)], [(95, 99, 94, 98)], [(98, 99, 97, 97.5)]], {"A": 0.5}, 10_000.0),
    "price_precision": (       # 33.33 × 1.005 = 33.49665 → 买限价 33.49；trunc(5000 / 33.33) = 150 股
        [[(33.33, 33.33, 33.33, 33.33)], [(33.49, 33.6, 33.4, 33.5)], [(33.5, 33.6, 33.4, 33.45)]], {"A": 0.5}, 10_000.0),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_single_name_scenarios_match_the_hand_ledger(name, tmp_path):
    rows, wd, cap = SCENARIOS[name]
    bars = _bars(rows)
    w = pd.DataFrame({k: [v] * len(rows) for k, v in wd.items()}, index=bars["close"].index)
    hand, fills = _hand_ledger(bars, w, cap)
    bt, pb = _engines(bars, w, cap, tmp_path)
    assert bt == pytest.approx(np.asarray(hand), abs=1e-12) and pb == pytest.approx(np.asarray(hand), abs=1e-12)
    d1 = bars["close"].index[1]
    assert fills.get((d1, "A")) == DAY1_FILL[name]


#: 每个场景第 1 天（T+1）应有的成交（股数, 价）；None = 不成交。50 股 = trunc(5000 / 100)；限价 100.5
DAY1_FILL = {
    "gap_up_fill_at_limit": (50, 100.5),
    "gap_up_not_reached": None,
    "gap_down_fill_at_open": (50, 95),
    "price_precision": (150, 33.49),
}


def test_a_day_order_that_missed_is_refrozen_the_next_day(tmp_path):
    """gap_up_not_reached：T+1 没成交，DAY 作废；T+1 收盘按新状态重新冻结，T+2 成交。"""
    rows, wd, cap = SCENARIOS["gap_up_not_reached"]
    bars = _bars(rows)
    w = pd.DataFrame({k: [v] * len(rows) for k, v in wd.items()}, index=bars["close"].index)
    _, fills = _hand_ledger(bars, w, cap)
    assert (bars["close"].index[2], "A") in fills


def _two_names():
    """A、B 两只；第 2 天从 A 全换到 B —— 卖 A 的钱当天不能拿来买 B（买入力不预支未成交卖单）。"""
    rows = [[(100, 100, 100, 100), (50, 50, 50, 50)],
            [(100, 101, 99, 100), (50, 51, 49, 50)],
            [(100, 101, 99, 101), (50, 51, 49, 51)],
            [(101, 102, 100, 102), (51, 52, 50, 52)],
            [(102, 103, 101, 102), (52, 53, 51, 52)]]
    bars = _bars(rows, names=("A", "B"))
    idx = bars["close"].index
    w = pd.DataFrame({"A": [0.95, 0.95, 0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.95, 0.95, 0.95]}, index=idx)
    return bars, w


def test_a_rotation_waits_for_the_sale_cash_like_the_broker(tmp_path):
    bars, w = _two_names()
    hand, fills = _hand_ledger(bars, w, 100_000.0)
    idx = bars["close"].index
    assert fills[(idx[3], "A")][0] < 0 and (idx[3], "B") not in fills, "卖 A 的同一天不能用这笔钱买 B"
    assert fills[(idx[4], "B")][0] > 0
    bt, pb = _engines(bars, w, 100_000.0, tmp_path)
    assert bt == pytest.approx(np.asarray(hand), abs=1e-12) and pb == pytest.approx(np.asarray(hand), abs=1e-12)


def test_partial_fills_costs_and_existing_positions(tmp_path):
    """成交量上限只够每天成交一部分；有佣金；之后在已有持仓上加减仓。"""
    rows = [[(100, 100, 100, 100)], [(100, 101, 99, 100.5)], [(100.5, 101, 100, 101)],
            [(101, 102, 100, 101.5)], [(101.5, 102, 100.5, 100.8)], [(100.8, 101, 100, 100.2)]]
    bars = _bars(rows)
    idx = bars["close"].index
    w = pd.DataFrame({"A": [0.6, 0.6, 0.6, 0.3, 0.3, 0.3]}, index=idx)
    vol = pd.DataFrame({"A": [2_000.0] * len(rows)}, index=idx)          # ADV 很小 → 部分成交
    hand, fills = _hand_ledger(bars, w, 100_000.0, vol=vol, mpp=0.10, fee_bps=10.0)
    assert any(abs(q) < 500 for q, _ in fills.values())                  # 确实发生了部分成交
    assert any(q < 0 for q, _ in fills.values())                         # 确实在已有持仓上减仓
    bt, pb = _engines(bars, w, 100_000.0, tmp_path, vol=vol, mpp=0.10, fee_bps=10.0)
    assert bt == pytest.approx(np.asarray(hand), rel=1e-12) and pb == pytest.approx(np.asarray(hand), rel=1e-12)


def test_the_broker_plan_and_the_simulation_freeze_the_same_order():
    """同一份输入：券商建单与模拟冻结出的委托逐字段相同（数量、方向、限价、参考价）。"""
    from datetime import date

    from app.core.execution.order_builder import build_rebalance_orders
    from app.core.execution.order_intent import freeze_rebalance
    tw = pd.Series({"A": 0.5, "B": 0.2, "C": 0.0})
    cur = {"A": 10.0, "C": 7.0}
    ref = pd.Series({"A": 100.0, "B": 33.33, "C": 12.5})
    po = build_rebalance_orders(tw, cur, ref, 10_000.0, date(2025, 3, 4), book_id=1, band_bps=BAND_BPS).orders
    it = freeze_rebalance(tw, cur, ref, 10_000.0, date(2025, 3, 4), band_bps=BAND_BPS, book_id=1).intents
    assert [(o.ticker, o.side, o.qty, o.limit_price, o.ref_price, o.client_id) for o in po] == \
        [(i.ticker, i.side, i.qty, i.limit_price, i.ref_price, i.client_id) for i in it]


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("whole", [True, False])
def test_the_engines_array_freeze_matches_the_broker_order_by_order(seed, whole):
    """
    两个模拟引擎逐日用数组版 `freeze_arrays`（逐单对象版太慢）；它必须与券商同一函数的逐单版逐股、逐限价一致，
    包括已有持仓、做空、碎价（<$1 的最小价位）、资金规则拒单的顺序。
    """
    from app.core.execution.order_intent import freeze_arrays, freeze_rebalance
    rng = np.random.default_rng(seed)
    n = 8
    names = [f"T{i}" for i in rng.permutation(n)]
    px = np.round(np.where(rng.random(n) < 0.2, rng.uniform(0.3, 0.99, n), rng.uniform(5, 400, n)), 4)
    long_only = seed % 2 == 0
    w = rng.dirichlet(np.ones(n)) * rng.uniform(0.5, 1.0) if long_only else rng.normal(0, 0.15, n)
    cur = np.trunc(rng.normal(0, 30, n)) if not long_only else np.trunc(np.abs(rng.normal(0, 30, n)))
    if not whole:
        cur = cur + rng.uniform(0, 0.9, n) * (cur != 0)
    eq = float(rng.uniform(5e3, 2e5))
    power = float(eq * rng.uniform(0.05, 0.6))
    qty, lim, rej = freeze_arrays(w, cur, px, eq, 0.005, whole, power, names)
    plan = freeze_rebalance(pd.Series(w, index=names), dict(zip(names, cur)), pd.Series(px, index=names), eq,
                            pd.Timestamp("2025-03-04").date(), band_bps=50.0, whole_shares=whole,
                            buying_power=power, allow_short=True)
    want_q = {i.ticker: i.signed_qty for i in plan.intents}
    want_l = {i.ticker: i.limit_price for i in plan.intents}
    got_q = {names[j]: qty[j] for j in range(n) if qty[j] != 0}
    assert set(got_q) == set(want_q)
    for tk in want_q:
        j = names.index(tk)
        assert got_q[tk] == pytest.approx(want_q[tk], abs=1e-9) and lim[j] == pytest.approx(want_l[tk], abs=1e-12)
    assert sorted(names[j] for j in np.flatnonzero(rej)) == sorted(r["ticker"] for r in plan.rejected)
