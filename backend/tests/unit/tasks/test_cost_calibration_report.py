"""
tasks/cost_calibration.py —— 成本校准建议的定钉测试（变异测试驱动）

来由：16 个变异点，首测击杀率 50.0%（存活 8）。

它把"内部 close-fill 模拟"与"真实 T+1 开盘缺口"对齐，给出 `impact_coef` 的
**调整建议**（不自动改参数，人工确认）。存活项集中在两处：

  - `np.abs(g)` 被删掉：隔夜缺口有正有负，不取绝对值后中位数≈0，
    于是无论真实成本多高，建议都是"下调"——**方向恰好反了**
  - `base * _SCALE_HI` 的 `*` 写成 `/`、`>` 放宽成 `>=`：判定档位错位，
    结论文本与实际情况不符

既有覆盖（test_phase8_cost_calib）只验证了"能产出报告对象"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.tasks.cost_calibration import _SCALE_HI, _SCALE_LO, calibrate


def _dataset(days: int = 12, tickers=("AAA", "BBB"), open_offset_bps=0.0):
    """构造 open/close：次日开盘相对当日成交价偏离 open_offset_bps。"""
    idx = pd.bdate_range("2024-01-02", periods=days)
    cols = list(tickers)
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    open_px = close.shift(-1) * (1 + open_offset_bps / 1e4)
    open_px = open_px.ffill()
    return {"close": close, "open": open_px,
            "high": close * 1.002, "low": close * 0.998,
            "volume": pd.DataFrame(1e6, index=idx, columns=cols)}


def _fills(ds: dict, gaps_bps, tickers=("AAA",)) -> pd.DataFrame:
    """
    按给定的缺口序列构造成交 + 对应的 open 价：
    成交价恒为 100，次日开盘 = 100 * (1 + gap/1e4)。
    """
    idx = ds["close"].index
    rows, open_px = [], ds["open"].copy()
    for i, gap in enumerate(gaps_bps):
        d = idx[i]
        tkr = tickers[i % len(tickers)]
        rows.append({"date": d, "ticker": tkr, "filled_weight": 0.1,
                     "fill_price": 100.0, "reject_reason": ""})
        open_px.loc[idx[i + 1], tkr] = 100.0 * (1 + gap / 1e4)
    ds["open"] = open_px
    return pd.DataFrame(rows)


# ===========================================================================
# A. 缺口统计必须用绝对值
# ===========================================================================

class TestAbsoluteGap:

    def test_symmetric_gaps_are_not_cancelled_out(self):
        """
        `abs_median = np.median(np.abs(g))` —— 删掉 `np.abs` 后，
        +50bps 与 −50bps 互相抵消，中位数≈0 → 校准结论恒为"成本偏保守，可下调"，
        而真实执行成本其实是 50bps。**这是方向性错误，不是精度问题。**
        """
        ds = _dataset(days=12)
        fills = _fills(ds, [50, -50, 50, -50, 50, -50, 50, -50])
        rep = calibrate(fills, ds, spread_bps=2.0, impact_coef=0.1)
        assert rep is not None
        assert rep.realized_gap_bps_abs_median == pytest.approx(50.0, abs=1e-6), (
            f"|缺口| 中位数为 {rep.realized_gap_bps_abs_median:.4f}，"
            f"正负缺口疑似相互抵消了")
        assert rep.realized_gap_bps_mean == pytest.approx(0.0, abs=1e-6), (
            "均值本来就应当接近 0 —— 这正是必须用绝对值的理由")

    def test_p90_also_uses_absolute_values(self):
        ds = _dataset(days=12)
        fills = _fills(ds, [-80, -60, -40, -20, 20, 40, 60, 80])
        rep = calibrate(fills, ds, spread_bps=2.0, impact_coef=0.1)
        assert rep is not None
        assert rep.realized_gap_bps_abs_p90 >= rep.realized_gap_bps_abs_median > 0
        assert rep.realized_gap_bps_abs_p90 == pytest.approx(
            float(np.percentile(np.abs([-80, -60, -40, -20, 20, 40, 60, 80]), 90)),
            abs=1e-9)


# ===========================================================================
# B. 档位判定
# ===========================================================================

class TestScaleDecision:

    @staticmethod
    def _report(gap: float, spread_bps: float = 2.0, impact: float = 0.1):
        ds = _dataset(days=12)
        fills = _fills(ds, [gap] * 8)
        return calibrate(fills, ds, spread_bps=spread_bps, impact_coef=impact)

    def test_large_gap_recommends_raising_impact(self):
        """
        `if abs_median > base * _SCALE_HI:` —— 缺口远大于基线 → 建议**上调**。
        `*` 写成 `/` 会把门槛从 base×2 变成 base÷2，几乎所有情况都落进"上调"。
        """
        rep = self._report(gap=100.0, spread_bps=2.0)
        assert rep is not None
        assert "上调" in rep.note, f"缺口 100bps vs 基线 2bps 却未建议上调：{rep.note}"
        assert rep.recommended_scale == pytest.approx(_SCALE_HI, abs=1e-9), (
            "放大倍数应被上界截断")

    def test_small_gap_recommends_lowering_impact(self):
        """`elif abs_median < base * _SCALE_LO:` —— 缺口远小于基线 → 建议下调。"""
        rep = self._report(gap=0.1, spread_bps=50.0)
        assert rep is not None
        assert "下调" in rep.note, f"缺口远小于基线却未建议下调：{rep.note}"
        assert rep.recommended_scale == pytest.approx(_SCALE_LO, abs=1e-9)

    def test_comparable_gap_recommends_holding(self):
        """两个门槛之间 → 维持。这条同时排除"永远上调/永远下调"的退化实现。"""
        rep = self._report(gap=2.0, spread_bps=2.0)
        assert rep is not None
        assert "维持" in rep.note, f"缺口与基线量级相当却未建议维持：{rep.note}"
        assert _SCALE_LO <= rep.recommended_scale <= _SCALE_HI

    def test_boundary_at_scale_hi_is_not_an_upgrade(self):
        """
        `>` 的边界：|缺口| **恰好等于** base×_SCALE_HI 时不算"显著大于"，
        应落在"维持"。放宽成 `>=` 会把它判成上调。

        ⚠️ 不能直接令 gap = spread×_SCALE_HI —— 缺口要经过
        `(open/close - 1)×1e4` 的价格往返，浮点上不会精确落回那个值
        （第一版就是这样，变异测试证实它存活）。
        改为**先测出实际的 abs_median，再反解出让它恰好等于门槛的 spread**，
        并断言这个前提确实成立。
        """
        probe = self._report(gap=4.0, spread_bps=1.0)
        assert probe is not None
        measured = probe.realized_gap_bps_abs_median
        spread = measured / _SCALE_HI
        assert spread * _SCALE_HI == measured, (
            f"反解不精确（{spread * _SCALE_HI!r} != {measured!r}），本用例测不到边界")

        rep = self._report(gap=4.0, spread_bps=spread)
        assert rep is not None
        assert rep.realized_gap_bps_abs_median == measured
        assert "上调" not in rep.note, (
            f"|缺口| 恰好等于上界门槛却被判为显著偏大：{rep.note}")
        assert "维持" in rep.note

    def test_boundary_at_scale_lo_is_not_a_downgrade(self):
        spread = 50.0
        rep = self._report(gap=spread * _SCALE_LO, spread_bps=spread)
        assert rep is not None
        assert "下调" not in rep.note, (
            f"|缺口| 恰好等于下界门槛却被判为显著偏小：{rep.note}")

    def test_recommended_coefficient_is_current_times_scale(self):
        rep = self._report(gap=100.0, spread_bps=2.0, impact=0.1)
        assert rep is not None
        assert rep.recommended_impact_coef == pytest.approx(
            rep.current_impact_coef * rep.recommended_scale, abs=1e-12)

    def test_scale_is_clipped_to_the_documented_bounds(self):
        for gap, spread in ((1e6, 1.0), (1e-6, 1e5)):
            rep = self._report(gap=gap, spread_bps=spread)
            assert rep is not None
            assert _SCALE_LO <= rep.recommended_scale <= _SCALE_HI


# ===========================================================================
# C. 无数据时的行为
# ===========================================================================

class TestNoData:

    def test_no_matchable_fills_returns_none_with_a_warning(self):
        """
        没有可匹配 T+1 开盘价的成交 → 返回 None 并留下告警，
        **不得**凭空给出一个建议。
        """
        ds = _dataset(days=6)
        fills = pd.DataFrame([{"date": ds["close"].index[-1], "ticker": "AAA",
                               "filled_weight": 0.1, "fill_price": 100.0,
                               "reject_reason": ""}])
        assert calibrate(fills, ds, spread_bps=2.0, impact_coef=0.1) is None

    def test_unknown_ticker_is_skipped(self):
        ds = _dataset(days=12)
        fills = pd.DataFrame([{"date": ds["close"].index[0], "ticker": "ZZZ",
                               "filled_weight": 0.1, "fill_price": 100.0,
                               "reject_reason": ""}])
        assert calibrate(fills, ds, spread_bps=2.0, impact_coef=0.1) is None

    def test_non_positive_fill_price_is_skipped(self):
        ds = _dataset(days=12)
        fills = pd.DataFrame([{"date": ds["close"].index[0], "ticker": "AAA",
                               "filled_weight": 0.1, "fill_price": 0.0,
                               "reject_reason": ""}])
        assert calibrate(fills, ds, spread_bps=2.0, impact_coef=0.1) is None

    def test_report_records_the_sample_it_used(self):
        ds = _dataset(days=12)
        fills = _fills(ds, [30.0] * 6)
        rep = calibrate(fills, ds, spread_bps=2.0, impact_coef=0.1)
        assert rep is not None
        assert rep.n_fills == 6
        assert rep.n_days == 6
        assert rep.period_start == str(ds["close"].index[0].date())


# ===========================================================================
# D. 月度任务的空成交短路
# ===========================================================================

def test_monthly_calibration_skips_when_there_are_no_fills(monkeypatch, tmp_path):
    """
    `if not fill_rows: return None` —— 删掉 `not` 会在**有**成交时反而跳过校准，
    而在没有成交时继续往下走（对空 DataFrame 取 min/max 抛异常）。
    """
    import app.tasks.cost_calibration as cc
    from app.db.position_store import PositionStore

    monkeypatch.setattr(PositionStore, "fills_in_range", lambda self, *a, **k: [])
    called = {"n": 0}

    def _should_not_run(*a, **k):
        called["n"] += 1
        raise AssertionError("无成交时不该去加载数据集")

    monkeypatch.setattr("app.core.data_engine.dataset_registry.load_registry_dataset",
                        _should_not_run)
    assert cc.run_monthly_calibration("px", "2024-01-01", "2024-01-31") is None
    assert called["n"] == 0


def test_monthly_calibration_loads_without_the_registry_health_gate(monkeypatch):
    """
    `load_registry_dataset(..., health_check=False)` —— 校准是**事后复盘**，
    读的是已经落过库的历史数据，不该再被注册表的健康门二次拦截
    （改成 True 后，任何一个月只要数据质量分偏低，整月校准就直接抛异常中止，
    而不是给出一份带 warnings 的报告）。
    """
    import app.tasks.cost_calibration as cc
    from app.db.position_store import PositionStore

    class _Row:
        date, ticker = pd.Timestamp("2024-01-02"), "AAA"
        filled_weight, fill_price, reject_reason = 0.1, 100.0, ""

    monkeypatch.setattr(PositionStore, "fills_in_range", lambda self, *a, **k: [_Row()])
    seen = {}

    def _load(name, start=None, end=None, health_check=False):
        seen["health_check"] = health_check
        return type("DS", (), {"data": _dataset(days=12)})()

    monkeypatch.setattr("app.core.data_engine.dataset_registry.load_registry_dataset",
                        _load)
    cc.run_monthly_calibration("px", "2024-01-01", "2024-01-31")
    assert seen["health_check"] is False, (
        "月度校准加载历史数据时打开了注册表自带的健康门")
