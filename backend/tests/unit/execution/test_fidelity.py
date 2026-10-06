"""
test_fidelity.py — Phase 12.3 内部模拟 vs moomoo 纸交易 保真度报告

数字全部手算：总滑点 ≈ 隔夜缺口 + 开盘执行，且分别对得上。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.core.execution.fidelity import (
    PERMANENT_IMPACT_NOTE, build_fidelity_report, fidelity_from_stores,
)
from app.tasks.cost_calibration import SCALE_BOUNDS

D1, D2 = date(2025, 3, 3), date(2025, 3, 4)


def _orders(rows):
    cols = ["client_id", "decision_date", "ticker", "side", "qty", "dealt_qty", "status",
            "purpose", "ref_price"]
    return pd.DataFrame(rows, columns=cols)


def _fills(rows):
    cols = ["client_id", "decision_date", "fill_date", "ticker", "qty", "price", "ref_price"]
    return pd.DataFrame(rows, columns=cols)


def _sim(rows):
    return pd.DataFrame(rows, columns=["date", "ticker", "traded_weight", "unfilled_weight",
                                       "fill_price"])


OPEN = pd.DataFrame({"AAA": [50.0, 50.5], "BBB": [20.0, 19.8]},
                    index=pd.DatetimeIndex(["2025-03-03", "2025-03-04"]))


def _report(orders, fills, sim, **kw):
    return build_fidelity_report(orders, fills, sim, OPEN, spread_bps=kw.pop("spread_bps", 2.0),
                                 impact_coef=kw.pop("impact_coef", 0.1),
                                 period_start="2025-03-03", period_end="2025-03-04", **kw)


def test_three_way_decomposition_is_exact():
    orders = _orders([("a", D1, "AAA", "BUY", 100, 100, "FILLED", "rebalance", 50.0),
                      ("b", D1, "BBB", "SELL", 50, 50, "FILLED", "rebalance", 20.0)])
    fills = _fills([("a", D1, D2, "AAA", 100.0, 50.6, 50.0),
                    ("b", D1, D2, "BBB", -50.0, 19.7, 20.0)])
    rep = _report(orders, fills, _sim([]), min_fills=1)
    # AAA 买：总 (50.6−50)/50 = 120bps；缺口 (50.5−50)/50 = 100bps；执行 (50.6−50.5)/50.5 = 19.80bps
    # BBB 卖：总 −(19.7−20)/20 = 150bps；缺口 −(19.8−20)/20 = 100bps；执行 −(19.7−19.8)/19.8 = 50.51bps
    assert rep.total_slippage_bps["median"] == pytest.approx(135.0)
    assert rep.overnight_gap_bps["median"] == pytest.approx(100.0)
    exe = sorted([1e4 * 0.1 / 50.5, 1e4 * 0.1 / 19.8])
    assert rep.at_open_exec_bps["median"] == pytest.approx(np.median(exe))
    # 按成交额加权：AAA 5060，BBB 985
    w = np.array([100 * 50.6, 50 * 19.7])
    assert rep.total_slippage_bps["mean"] == pytest.approx(np.average([120.0, 150.0], weights=w))
    assert rep.n_live_fills == 2 and rep.n_live_orders == 2


def test_permanent_impact_is_declared_unidentifiable_not_estimated():
    rep = _report(_orders([]), _fills([]), _sim([]))
    assert rep.permanent_impact_bps is None
    assert rep.permanent_impact_note == PERMANENT_IMPACT_NOTE and "实盘" in rep.permanent_impact_note
    assert "永久冲击：不可识别" in rep.to_markdown()


def _many(n, exec_bps):
    """n 笔买单：决策日收盘 100、次日开盘 100、成交价 = 100×(1+exec_bps/1e4)。"""
    opn = pd.DataFrame({f"T{i}": [100.0, 100.0] for i in range(n)},
                       index=pd.DatetimeIndex(["2025-03-03", "2025-03-04"]))
    orders = _orders([(f"c{i}", D1, f"T{i}", "BUY", 10, 10, "FILLED", "rebalance", 100.0)
                      for i in range(n)])
    fills = _fills([(f"c{i}", D1, D2, f"T{i}", 10.0, 100.0 * (1 + exec_bps / 1e4), 100.0)
                    for i in range(n)])
    return orders, fills, opn


@pytest.mark.parametrize("exec_bps,scale", [(1.5, 1.5), (0.6, 0.6), (10.0, SCALE_BOUNDS[1]),
                                            (0.1, SCALE_BOUNDS[0]), (-3.0, SCALE_BOUNDS[0])])
def test_impact_recommendation_is_bounded_and_relative_to_half_spread(exec_bps, scale):
    o, f, opn = _many(25, exec_bps)
    rep = build_fidelity_report(o, f, _sim([]), opn, spread_bps=2.0, impact_coef=0.1,
                                period_start="a", period_end="b")
    # 半价差基线 = 1bp；建议缩放 = 中位开盘执行 / 1bp，截到 [0.5, 2]
    assert rep.recommended_scale == pytest.approx(scale)
    assert rep.recommended_impact_coef == pytest.approx(0.1 * scale)


def test_too_few_fills_means_no_recommendation():
    o, f, opn = _many(5, 10.0)
    rep = build_fidelity_report(o, f, _sim([]), opn, spread_bps=2.0, impact_coef=0.1,
                                period_start="a", period_end="b")
    assert rep.recommended_scale == 1.0 and rep.recommended_impact_coef == 0.1
    assert any("不足以校准" in w for w in rep.warnings)


def test_flatten_orders_are_excluded_from_the_comparison():
    orders = _orders([("a", D1, "AAA", "BUY", 100, 100, "FILLED", "rebalance", 50.0),
                      ("f", D1, "BBB", "SELL", 50, 50, "FILLED", "flatten", 20.0)])
    fills = _fills([("a", D1, D2, "AAA", 100.0, 50.6, 50.0),
                    ("f", D1, D2, "BBB", -50.0, 15.0, 20.0)])
    rep = _report(orders, fills, _sim([]), min_fills=1)
    assert rep.n_live_orders == 1 and rep.n_live_fills == 1
    assert rep.total_slippage_bps["median"] == pytest.approx(120.0)


def test_fill_ratios_and_sim_matching():
    orders = _orders([("a", D1, "AAA", "BUY", 100, 100, "FILLED", "rebalance", 50.0),
                      ("b", D1, "BBB", "BUY", 100, 40, "PARTIAL", "rebalance", 20.0),
                      ("c", D1, "CCC", "BUY", 100, 0, "CANCELLED", "rebalance", 10.0),
                      ("d", D1, "DDD", "SELL", 100, 0, "SUBMITTED", "rebalance", 5.0)])
    sim = _sim([(D1, "AAA", 0.05, 0.0, 50.0),          # 同向、全成交
                (D1, "BBB", 0.02, 0.02, 20.0),         # 同向、一半
                (D1, "CCC", -0.01, 0.0, 10.0),         # 反向 → 不算匹配
                (D1, "DDD", -0.01, 0.0, 5.001)])       # 同向，但参考价不一致
    rep = _report(orders, _fills([]), sim)
    assert rep.live_fill_ratio_median == pytest.approx(0.4)     # 在途的 DDD 不计：1, 0.4, 0
    assert rep.n_live_unfilled == 1
    assert rep.n_matched_sim == 3
    assert rep.sim_fill_ratio_median == pytest.approx(1.0)      # 1, 0.5, 1
    assert rep.n_ref_mismatch == 1
    assert any("同一份收盘价" in w for w in rep.warnings)


def test_missing_open_price_is_counted_not_guessed():
    orders = _orders([("a", D1, "ZZZ", "BUY", 10, 10, "FILLED", "rebalance", 50.0)])
    fills = _fills([("a", D1, D2, "ZZZ", 10.0, 50.5, 50.0)])
    rep = _report(orders, fills, _sim([]), min_fills=1)
    assert rep.total_slippage_bps["n"] == 1 and rep.overnight_gap_bps["n"] == 0
    assert any("找不到成交日开盘价" in w for w in rep.warnings)


def test_markdown_carries_the_never_auto_apply_rule():
    md = _report(_orders([]), _fills([]), _sim([])).to_markdown()
    assert "切勿自动改" in md and "纸交易引擎撮合口径" in md


def test_stats_weighting_edge_cases():
    from app.core.execution.fidelity import _stats
    assert _stats([1.0, 3.0])["mean"] == 2.0
    assert _stats([1.0, 3.0], [0.0, 0.0])["mean"] == 2.0          # 权重全 0 → 退回等权
    assert _stats([1.0, 3.0], [1.0, 3.0])["mean"] == 2.5


def test_markdown_shows_numbers_when_there_are_samples():
    orders = _orders([("a", D1, "AAA", "BUY", 100, 100, "FILLED", "rebalance", 50.0)])
    fills = _fills([("a", D1, D2, "AAA", 100.0, 50.6, 50.0)])
    md = _report(orders, fills, _sim([]), min_fills=1).to_markdown()
    assert "中位 120.00" in md


def test_degenerate_orders_and_fills_are_skipped_not_divided_by():
    from app.core.execution.broker_gateway import DUST_QTY
    orders = _orders([("z", D1, "AAA", "BUY", 0, 0, "CANCELLED", "rebalance", 50.0),
                      ("d", D1, "BBB", "BUY", 10, DUST_QTY, "CANCELLED", "rebalance", 20.0),
                      ("r", D1, "AAA", "BUY", 10, 10, "FILLED", "rebalance", 0.0)])
    fills = _fills([("r", D1, D2, "AAA", 10.0, 50.0, 0.0),            # 补登单：无参考价
                    ("d", D1, D2, "BBB", DUST_QTY, 20.0, 20.0),       # 零股
                    ("d", D1, D2, "BBB", 5.0, 0.0, 20.0)])            # 价格 0
    sim = _sim([(D1, "AAA", 0.05, 0.0, 50.0)])
    rep = _report(orders, fills, sim, min_fills=1)
    assert rep.n_live_fills == 0
    # qty=0 的单不参与（否则除零）；剩下 d（零股/10）与 r（1.0）两张取中位
    assert rep.live_fill_ratio_median == pytest.approx((DUST_QTY / 10 + 1.0) / 2)
    assert rep.n_live_unfilled == 2                                  # 0 与零股都算没成交
    assert rep.n_ref_mismatch == 0                                   # 参考价 0 不比较


def test_sim_matching_ignores_dust_intentions():
    from app.core.execution.fidelity import WEIGHT_DUST
    orders = _orders([("a", D1, "AAA", "BUY", 10, 10, "FILLED", "rebalance", 50.0),
                      ("b", D1, "BBB", "BUY", 10, 10, "FILLED", "rebalance", 20.0)])
    sim = _sim([(D1, "AAA", WEIGHT_DUST, 0.0, 50.0), (D1, "BBB", 2 * WEIGHT_DUST, 0.0, 20.0)])
    assert _report(orders, _fills([]), sim).n_matched_sim == 1


def test_reference_price_tolerance_boundary_and_scale():
    from app.core.execution.fidelity import REF_PRICE_TOL
    orders = _orders([("a", D1, "A", "BUY", 1, 1, "FILLED", "rebalance", 1.0),
                      ("b", D1, "B", "BUY", 1, 1, "FILLED", "rebalance", 1.0),
                      ("c", D1, "C", "BUY", 1, 1, "FILLED", "rebalance", 100.0)])
    sim = _sim([(D1, "A", 0.01, 0.0, 1.0 + REF_PRICE_TOL),           # 恰在容差上 → 同一份价
                (D1, "B", 0.01, 0.0, 1.0 + 2 * REF_PRICE_TOL),       # 超出 → 不一致
                (D1, "C", 0.01, 0.0, 100.00005)])                    # 相对 5e-7 → 同一份价
    assert _report(orders, _fills([]), sim).n_ref_mismatch == 1


def test_invalid_open_prices_are_counted_as_missing():
    opn = pd.DataFrame({"AAA": [50.0, 0.0], "BBB": [20.0, -1.0]},
                       index=pd.DatetimeIndex(["2025-03-03", "2025-03-04"]))
    orders = _orders([("a", D1, "AAA", "BUY", 10, 10, "FILLED", "rebalance", 50.0),
                      ("b", D1, "BBB", "BUY", 10, 10, "FILLED", "rebalance", 20.0)])
    fills = _fills([("a", D1, D2, "AAA", 10.0, 50.5, 50.0), ("b", D1, D2, "BBB", 10.0, 20.1, 20.0)])
    rep = build_fidelity_report(orders, fills, _sim([]), opn, spread_bps=2.0, impact_coef=0.1,
                                period_start="a", period_end="b", min_fills=1)
    assert rep.total_slippage_bps["n"] == 2 and rep.overnight_gap_bps["n"] == 0


def test_recommendation_at_exactly_min_fills_and_relative_to_half_spread():
    o, f, opn = _many(20, 1.5)
    rep = build_fidelity_report(o, f, _sim([]), opn, spread_bps=4.0, impact_coef=0.1,
                                period_start="a", period_end="b", min_fills=20)
    assert rep.recommended_scale == pytest.approx(0.75)               # 1.5 / (4/2)


def test_zero_median_execution_gets_the_conservative_note():
    o, f, opn = _many(25, 0.0)
    rep = build_fidelity_report(o, f, _sim([]), opn, spread_bps=2.0, impact_coef=0.1,
                                period_start="a", period_end="b")
    assert rep.recommended_scale == SCALE_BOUNDS[0]
    assert any("≤ 0" in n for n in rep.notes)


def test_report_from_stores_reads_both_books(tmp_path):
    """端到端取数：执行账本（book −1）与模拟组合账本（book 0）各取各的，不串账。"""
    from app.db.execution_store import ExecutionStore
    from app.db.position_store import DailyPnL, PositionStore
    from app.core.execution.order_builder import PlannedOrder
    from app.core.execution.broker_gateway import BrokerOrder

    es = ExecutionStore(db_url=f"sqlite:///{tmp_path / 'e.db'}")
    ps = PositionStore(db_url=f"sqlite:///{tmp_path / 'p.db'}")
    o = PlannedOrder("qaR-1-20250303-AAA-B", "AAA", "BUY", 100, 50.0, 50.25, 0, 100, 0.05)
    es.add_intent(o, -1, D1)
    es.mark_submitted(o.client_id, "1")
    es.apply_broker_state(o.client_id, BrokerOrder(
        "1", o.client_id, "AAA", "BUY", 100, 50.25, "FILLED_ALL", 100, 50.6, "", "", ""), D2)
    ps.record_day(0, D1, {"AAA": 0.05}, [{"ticker": "AAA", "target_weight": 0.05,
                                         "filled_weight": 0.05, "fill_price": 50.0,
                                         "traded_weight": 0.05, "unfilled_weight": 0.0}],
                  DailyPnL(0, "2025-03-03", 0.0, 0.0, 0.0, 1.0))
    rep = fidelity_from_stores("2025-03-01", "2025-03-05", OPEN, exec_store=es, position_store=ps)
    assert rep.n_live_orders == 1 and rep.n_matched_sim == 1 and rep.n_ref_mismatch == 0
    assert rep.total_slippage_bps["median"] == pytest.approx(120.0)

    # 别的账本、区间之后的决策日都不能混进来
    other = PlannedOrder("qaR7-20250303-BBB-B", "BBB", "BUY", 5, 20.0, 20.1, 0, 5, 0.01)
    es.add_intent(other, 7, D1)
    late = PlannedOrder("qaR-1-20250310-CCC-B", "CCC", "BUY", 5, 10.0, 10.05, 0, 5, 0.01)
    es.add_intent(late, -1, date(2025, 3, 10))
    rep2 = fidelity_from_stores("2025-03-01", "2025-03-05", OPEN, exec_store=es, position_store=ps)
    assert rep2.n_live_orders == 1
