"""
冻结订单意图的边界（复审 21cd90b 之后的关键路径变异复核：app/core/execution/order_intent.py）。

首轮存活的点集中在三处，原有用例都离边界很远：
    · 股数的取整容差：反推 / 撮合出来的股数离整数在 SHARE_TOL × max(1, |整数|) 之内按整数，之外撮合向零取整、反推保留原值
    · 权益 / 参考价的守卫：0、负数、NaN、无穷大都不冻结任何意图（不是"按 0 算目标、把持仓卖光"）
    · 不按整股的逐单版本（研究对照用）：先卖后买、空仓且不在目标里的名字不出现、恰好 1e-12 的差额不下单
另有资金规则：买单累计金额恰好等于可用资金时照发（超过才拒）。
剩下的等价变异写在 PROVEN_EQUIVALENT（各配一条机械验证的用例）。
"""

from __future__ import annotations

import inspect
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.core.execution import order_intent as oi

D = date(2025, 3, 3)


def _fr(w, cur, px, equity=10_000.0, **kw):
    return oi.freeze_rebalance(pd.Series(w, dtype=float), cur, pd.Series(px, dtype=float), equity, D,
                               band_bps=50.0, **kw)


# ── 取整容差 ─────────────────────────────────────────────

def test_back_solved_shares_snap_only_within_the_relative_tolerance():
    """权益 1、价格 1：股数 = 权重。离整数 ≤ 1e-6 × max(1, |整数|) 的按整数，否则原样（不是四舍五入）。"""
    w = np.array([49.9999999, 50.0000001, 49.6, -1000.0005, 1000.0005, 1e-6, -49.9999999])
    got = oi.shares_from_weights(w, 1.0, np.ones(len(w)), True)
    assert got.tolist() == [50.0, 50.0, 49.6, -1000.0, 1000.0, 0.0, -50.0]
    assert oi.shares_from_weights(w, 1.0, np.ones(len(w)), False).tolist() == w.tolist()


def test_a_zero_or_missing_price_gives_no_shares():
    got = oi.shares_from_weights(np.array([0.5, 0.5, 0.5]), 100.0, np.array([0.0, np.nan, 10.0]), True)
    assert got.tolist() == [0.0, 0.0, 5.0]


def test_filled_shares_snap_float_noise_and_otherwise_truncate_toward_zero():
    """撮合给出的成交股数：浮点噪声按整数，其余向零取整（不能四舍五入成多成交一股）。499999.5 恰好压在容差上。"""
    filled = np.array([999.9995, 999.6, -999.6, -999.9995, 499999.5])
    intent = np.array([2000.0, 2000.0, -2000.0, -2000.0, 1e6])
    assert oi.clamp_filled_shares(filled, intent, True).tolist() == [1000.0, 999.0, -999.0, -1000.0, 500000.0]


# ── 权益与参考价的守卫 ───────────────────────────────────

@pytest.mark.parametrize("equity", [0.0, -1.0, float("nan"), float("inf"), None])
def test_no_positive_finite_equity_freezes_nothing(equity):
    plan = _fr({"AAA": 0.5}, {"AAA": 10.0}, {"AAA": 100.0}, equity=equity)
    assert (plan.intents, plan.rejected, plan.skipped) == ([], [], [])
    qty, limit, rej = oi.freeze_arrays(np.array([0.5]), np.array([10.0]), np.array([100.0]), equity, 0.005, True,
                                       None, ["AAA"])
    assert qty.tolist() == [0.0] and np.isnan(limit).all() and not rej.any()


@pytest.mark.parametrize("px", [0.0, -1.0, float("inf"), float("nan")])
def test_a_name_without_a_usable_reference_price_is_skipped_not_traded(px):
    plan = _fr({"AAA": 0.5}, {}, {"AAA": px}, whole_shares=False)
    assert plan.intents == [] and [s["reason"] for s in plan.skipped] == ["no_reference_price"]


def test_shorts_are_off_unless_asked_for():
    for whole in (True, False):
        assert _fr({"AAA": -0.5}, {}, {"AAA": 100.0}, whole_shares=whole).intents == []
    assert [i.side for i in _fr({"AAA": -0.5}, {}, {"AAA": 100.0}, whole_shares=False, allow_short=True).intents] \
        == ["SELL"]


def test_the_session_is_carried_on_every_intent():
    plan = _fr({"AAA": 0.5}, {}, {"AAA": 100.0}, session="2025-03-04")
    assert [i.session for i in plan.intents] == [date(2025, 3, 4)]


# ── 不按整股的逐单版本 ───────────────────────────────────

def test_fractional_freeze_sells_first_and_ignores_empty_names_outside_the_target():
    plan = _fr({"AAA": 0.5, "BBB": 0.0}, {"BBB": 10.0, "ZZZ": 0.0}, {"AAA": 100.0, "BBB": 50.0}, whole_shares=False)
    assert [(i.ticker, i.side) for i in plan.intents] == [("BBB", "SELL"), ("AAA", "BUY")]
    assert plan.skipped == []                                   # ZZZ 空仓、不在目标里、也没有参考价：不出现


def test_a_difference_of_exactly_1e_12_shares_is_not_an_order():
    assert _fr({"AAA": 1e-12}, {}, {"AAA": 1.0}, equity=1.0, whole_shares=False).intents == []
    qty, limit, _ = oi.freeze_arrays(np.array([1e-12]), np.array([0.0]), np.array([1.0]), 1.0, 0.005, False, None,
                                     ["AAA"])
    assert qty.tolist() == [0.0] and np.isnan(limit).all()


def test_a_leftover_below_one_share_is_not_a_whole_share_order():
    """整股账本：现有 10.5 股、目标 10 股 → 差额不足一股，不下单（委托量 0、没有限价）。"""
    qty, limit, _ = oi.freeze_arrays(np.array([0.1]), np.array([10.5]), np.array([100.0]), 10_000.0, 0.005, True,
                                     None, ["AAA"])
    assert qty.tolist() == [0.0] and np.isnan(limit).all()


# ── 资金规则 ─────────────────────────────────────────────

def test_a_buy_that_uses_exactly_the_buying_power_is_sent():
    """50 股 × 限价 100.50 = 5025.00：可用资金恰好 5025 → 照发；少一分 → 缩到买得起的 49 股；连一股都买不起 → 不发。"""
    assert [i.qty for i in _fr({"AAA": 0.5}, {}, {"AAA": 100.0}, buying_power=5025.0).intents] == [50.0]
    plan = _fr({"AAA": 0.5}, {}, {"AAA": 100.0}, buying_power=5024.99)
    assert [i.qty for i in plan.intents] == [49.0] and plan.rejected == []
    assert plan.resized == [{"ticker": "AAA", "qty": 50.0, "resized_qty": 49.0, "reason": "insufficient_buying_power"}]
    plan = _fr({"AAA": 0.5}, {}, {"AAA": 100.0}, buying_power=100.49)
    assert plan.intents == [] and [r["reason"] for r in plan.rejected] == ["insufficient_buying_power"]
    qty, _, rej = oi.freeze_arrays(np.array([0.5]), np.array([0.0]), np.array([100.0]), 10_000.0, 0.005, True,
                                   5025.0, ["AAA"])
    assert qty.tolist() == [50.0] and not rej.any()


# ── 撮合与开盘一步 ───────────────────────────────────────

def _fill(qty, limit, o, h, lo):
    return oi.limit_fill_array(np.array([float(qty)]), np.array([float(limit)]), np.array([float(o)]),
                               np.array([float(h)]), np.array([float(lo)]))[0]


def test_a_zero_limit_or_a_zero_price_in_the_bar_never_fills():
    assert math.isnan(_fill(-5, 0.0, 10, 11, 9))           # 卖单限价 0
    assert math.isnan(_fill(5, 10.0, 0, 11, 9))            # 开盘价 0
    assert math.isnan(_fill(5, 10.0, 5, 0, 4))             # 最高价 0
    assert math.isnan(_fill(5, 10.0, 5, 6, 0))             # 最低价 0


def test_touching_the_limit_exactly_fills_at_the_limit():
    assert _fill(5, 10.0, 11, 12, 10) == 10.0              # 买：开在限价上方、最低恰好触及
    assert _fill(-5, 10.0, 9, 10, 8) == 10.0               # 卖：开在限价下方、最高恰好触及


def test_opening_exactly_at_the_limit_fills_at_the_open_whatever_the_range_says():
    """开盘价恰好等于限价就按开盘价成交，不再看最高 / 最低（数据里最高 / 最低与开盘矛盾时也一样）。"""
    assert _fill(5, 10.0, 10, 12, 10.5) == 10.0
    assert _fill(-5, 10.0, 10, 9.5, 8) == 10.0


def _open(prev_w, target_w, ref_px, o_eff, equity=10_000.0, v_open=10_000.0, tickers=("AAA", "BBB")):
    n = len(tickers)
    o = np.asarray(o_eff, dtype=float)
    return oi.open_targets(np.asarray(target_w, dtype=float), np.asarray(prev_w, dtype=float),
                           np.asarray(prev_w, dtype=float), equity, v_open, np.asarray(ref_px, dtype=float), o,
                           o, o * 1.01, o * 0.99, list(tickers), D, date(2025, 3, 4), 0.005, True,
                           tradable=np.ones(n, dtype=bool))


def test_a_zero_reference_or_open_price_freezes_no_order_for_that_name():
    for ref, opn in (([0.0, 50.0], [100.0, 50.0]), ([100.0, 50.0], [0.0, 50.0])):
        _, _, _, plan = _open([0.0, 0.0], [0.4, 0.4], ref, opn)
        assert plan.signed_qty[0] == 0.0 and np.isnan(plan.limit[0]) and plan.signed_qty[1] == 80.0


def test_a_short_position_uses_up_buying_power_too():
    """可用资金 = 权益 − 已投入的总敞口（空头按绝对值）：空 20% 时只剩 80%，买 90% 的单缩到 8000 买得起的 79 股。"""
    _, _, _, plan = _open([0.0, -0.2], [0.9, -0.2], [100.0, 50.0], [100.0, 50.0])
    assert plan.signed_qty[0] == 79.0 and plan.rejected == [] and [r["ticker"] for r in plan.resized] == ["AAA"]
    _, _, _, plan = _open([0.0, -0.2], [0.7, -0.2], [100.0, 50.0], [100.0, 50.0])
    assert plan.signed_qty[0] == 70.0 and plan.rejected == []


def test_an_open_value_of_zero_leaves_the_weights_as_given():
    pre, after, _, _ = _open([0.5, 0.0], [0.5, 0.3], [100.0, 50.0], [100.0, 50.0], v_open=0.0)
    assert pre.tolist() == after.tolist() == [0.5, 0.0]


class _Fills:
    def __init__(self, filled_w):
        self.filled_w = np.asarray(filled_w, dtype=float)


def test_fills_are_capped_at_the_intended_shares_at_the_real_open_price():
    """开盘价 50、开盘市值 10000：意图 0.5 → 100 股；撮合报 0.6（120 股）→ 截到 100 股 = 0.5。开盘价 0 的名字不成交。"""
    with np.errstate(all="ignore"):
        fr = oi.whole_share_fills(_Fills([0.6, 0.3]), np.zeros(2), np.array([0.5, 0.3]), 10_000.0,
                                  np.array([50.0, 0.0]), True)
    assert fr.filled_d.tolist() == [0.5, 0.0] and fr.unfilled_d.tolist() == [0.0, 0.3]


# ── 等价变异 ─────────────────────────────────────────────
#
# 每条证明都配同一种机械验证：把同一个变异施加到函数源码的副本上（不改模块本身），
# 与原函数在随机输入上逐元素比对 —— 输入专门覆盖两式唯一可能不同的那个值（0、-0.0、恰好 1.0）。

PROVEN_EQUIVALENT = {
    "app/core/execution/order_intent.py ×1 — L122 `if w < 0 and not allow_short` -> `w <= 0`":
        "两式只在 w == 0（含 -0.0）时不同，而那时分支体 `w = 0.0` 赋的就是同一个值：目标股数、差额、"
        "纯多头判断（-0.0 >= 0 为真）都不变。见 test_mutated_freeze_rebalance_gives_identical_plans。",
    "app/core/execution/order_intent.py ×1 — L135 `side = 'BUY' if delta > 0` -> `>=`":
        "两式只在 delta == 0 时不同；而上一行 `if abs(delta) <= 1e-12: continue` 已把 |delta| ≤ 1e-12 的全部跳过，"
        "走到这一行的 delta 不可能为 0。见 test_mutated_freeze_rebalance_gives_identical_plans。",
    "app/core/execution/order_intent.py ×1 — L167 `tick = np.where(raw >= 1.0, 0.01, 0.0001)` -> `>`":
        "两式只在 raw == 1.0 时选不同的价位；1.0 在两种价位网格上都恰好是格点，买向下、卖向上取整都回到 1.0"
        "（取整带 ±1e-9 的护栏）。与 order_builder.round_to_tick 的同型证明一致。"
        "见 test_mutated_array_functions_give_identical_results。",
    "app/core/execution/order_intent.py ×1 — L200 `buy = delta > 0` -> `>=`（freeze_arrays）":
        "两式只在 delta == 0 时不同。delta == 0 的名字在两个分支里都不 active（整股：`delta != 0`；"
        "不按整股：q = |delta| = 0 不大于 1e-12），而 buy 只在 active 的掩码下被使用：限价 "
        "`np.where(active, lim, nan)`、资金循环 `active & buy`、委托量 `np.where(active, sign·q, 0)`。"
        "见 test_mutated_array_functions_give_identical_results。",
    "app/core/execution/order_intent.py ×1 — L224 `buy = signed_qty > 0` -> `>=`（limit_fill_array）":
        "两式只在委托量为 0 时不同，而 good 掩码含 `signed_qty != 0`，委托量 0 的名字结果一律 NaN。"
        "见 test_mutated_array_functions_give_identical_results。",
    "app/core/execution/order_intent.py ×1 — L272 `o = np.where(o_eff > 0, o_eff, nan)` -> `>=`（whole_share_fills）":
        "两式只在开盘价恰好为 0 时不同：原式得 NaN、成交权重经 nan_to_num 为 0；变异式得 0，成交权重 = 股数 × 0 / 开盘市值，"
        "仍是 0（股数为 ±inf 时 inf × 0 = NaN，再经 nan_to_num 为 0）。见 test_mutated_array_functions_give_identical_results。",
}


def _mutated(fn, old, new):
    """
    把同一变异施加到函数源码的副本上、在原模块的命名空间里重新定义，返回变异后的函数（不改模块本身）。
    **双向**：变异工具运行时模块本身已经是变异后的写法，这时副本换回原式 —— 比对的仍是"原式 vs 变异式"。
    只认单向的话，证明用例会因为找不到原式而红，把一个等价变异记成"杀死"（击杀率虚高）。
    """
    import textwrap
    code = textwrap.dedent(inspect.getsource(fn))
    a, b = (old, new) if old in code else (new, old)
    assert code.count(a) == 1 and b not in code, (old, new)
    ns = dict(vars(oi))
    exec(compile(code.replace(a, b), "<mutant>", "exec"), ns)             # noqa: S102
    return ns[fn.__name__]


def test_mutated_array_functions_give_identical_results():
    rng = np.random.default_rng(7)
    fill = _mutated(oi.limit_fill_array, "buy = signed_qty > 0", "buy = signed_qty >= 0")
    fills = _mutated(oi.whole_share_fills, "np.where(o_eff > 0, o_eff, np.nan)", "np.where(o_eff >= 0, o_eff, np.nan)")
    tick = _mutated(oi._tick_round, "raw >= 1.0", "raw > 1.0")
    frz = _mutated(oi.freeze_arrays, "buy = delta > 0", "buy = delta >= 0")
    for _ in range(200):
        n = 12
        qty = rng.choice([-3.0, 0.0, 0.0, 5.0], n)
        lim, o = rng.choice([0.0, 9.0, 10.0, 11.0], n), rng.choice([0.0, 9.0, 10.0, 11.0], n)
        h, lo = o + rng.choice([0.0, 1.0], n), o - rng.choice([0.0, 1.0], n)
        np.testing.assert_array_equal(oi.limit_fill_array(qty, lim, o, h, lo), fill(qty, lim, o, h, lo))
        o_eff = rng.choice([0.0, 0.0, 25.0, 50.0], n)
        pre, after, fw = (rng.uniform(-0.2, 0.6, n) for _ in range(3))
        raw, buy = rng.choice([1.0, 1.0, 0.99995, 1.00004, 37.123], n), rng.random(n) < 0.5
        np.testing.assert_array_equal(oi._tick_round(raw, buy), tick(raw, buy))
        px = rng.choice([1.0, 20.0, 50.0], n)
        cur = rng.choice([0.0, 0.0, 10.0, 40.0], n)
        w = np.where(rng.random(n) < 0.5, cur * px / 10_000.0, rng.choice([0.0, 0.02, 0.1], n))   # 一半名字差额恰好为 0
        for whole in (True, False):
            a = oi.whole_share_fills(_Fills(fw), pre, after, 10_000.0, o_eff, whole)
            b = fills(_Fills(fw), pre, after, 10_000.0, o_eff, whole)
            np.testing.assert_array_equal(a.filled_d, b.filled_d)
            np.testing.assert_array_equal(a.unfilled_d, b.unfilled_d)
            names = [f"N{i:02d}" for i in range(n)]
            for x, y in zip(oi.freeze_arrays(w, cur, px, 10_000.0, 0.005, whole, 6_000.0, names),
                            frz(w, cur, px, 10_000.0, 0.005, whole, 6_000.0, names)):
                np.testing.assert_array_equal(x, y)


def test_mutated_freeze_rebalance_gives_identical_plans():
    rng = np.random.default_rng(11)
    variants = [_mutated(oi.freeze_rebalance, "if w < 0 and not allow_short", "if w <= 0 and not allow_short"),
                _mutated(oi.freeze_rebalance, 'side = "BUY" if delta > 0', 'side = "BUY" if delta >= 0')]
    names = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    for _ in range(100):
        px = dict(zip(names, rng.choice([1.0, 20.0, 50.0], 5)))
        cur = dict(zip(names, rng.choice([0.0, 10.0, 40.0], 5)))
        w = {k: float(rng.choice([0.0, -0.0, 0.1, -0.1, cur[k] * px[k] / 10_000.0])) for k in names}
        for whole in (True, False):
            args = (pd.Series(w, dtype=float), cur, pd.Series(px, dtype=float), 10_000.0, D)
            kw = dict(band_bps=50.0, whole_shares=whole, buying_power=4_000.0)
            base = oi.freeze_rebalance(*args, **kw)
            for f in variants:
                got = f(*args, **kw)
                assert (got.intents, got.rejected, got.skipped) == (base.intents, base.rejected, base.skipped)


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 6
