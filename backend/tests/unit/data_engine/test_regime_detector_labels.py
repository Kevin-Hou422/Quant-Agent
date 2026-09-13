"""
data_engine/regime_detector.py —— Regime 标注边界的定钉测试（变异测试驱动）

来由：17 个变异点，首测击杀率 **52.9%**（存活 8）—— 存活的全是**判定边界**。

RegimeDetector 的输出一路往下走：MarketObserver 用它挑今晚往哪个方向挖因子，
`regime_to_alpha_weights` 用它给 Alpha Pool 里的候选做家族倾斜加权。
标错不会报错、不会让任何东西变红，只会让系统长期押错方向。

存活项：
  - `trend > threshold` / `trend < -threshold` / `vol > vol_cut` 三个判定
    —— 放宽一格，**恰好踩在阈值上**的日子会被标成牛/熊/高波动
  - `trend_window < 5 or vol_window < 5` / `not 0.5 <= vol_quantile < 1.0`
    —— 构造参数的合法区间两端
  - `len(ret) < trend_window + vol_window` —— 数据量刚好够时该不该拒
  - `.apply(np.prod, raw=True)` 的 raw —— 改成 False 会让每个窗口都构造一个
    Series 再传进去，慢几十倍，且 np.prod 拿到的是 Series（结果相同但性能塌陷）
  - `base = max(sharpe, 0.0) + 0.1` 的 `+` —— 改成 `-` 会让 Sharpe < 0.1 的
    候选拿到**负权重**，归一化之后出现负的配比

既有覆盖（test_phase4 / test_phase6）只验证"fit 完能出标签、标签在 REGIMES 里"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.regime_detector import REGIMES, RegimeDetector


IDX = pd.bdate_range("2022-01-03", periods=400)


def _ret(values) -> pd.Series:
    arr = np.asarray(values, dtype=float)
    return pd.Series(arr, index=IDX[:len(arr)])


# ===========================================================================
# A. 构造参数的合法区间
# ===========================================================================

class TestConstructorBounds:

    def test_minimum_windows_are_accepted(self):
        """`< 5` 放宽成 `<= 5` 会把最小合法窗口 5 判成非法。"""
        d = RegimeDetector(trend_window=5, vol_window=5)
        assert d.trend_window == 5 and d.vol_window == 5

    @pytest.mark.parametrize("tw,vw", [(4, 20), (60, 4), (4, 4)])
    def test_windows_below_five_are_rejected(self, tw, vw):
        with pytest.raises(ValueError, match="至少为 5"):
            RegimeDetector(trend_window=tw, vol_window=vw)

    def test_vol_quantile_lower_bound_is_inclusive(self):
        """`0.5 <= q`：0.5（中位数）必须**被接受**。"""
        assert RegimeDetector(vol_quantile=0.5).vol_quantile == 0.5

    def test_vol_quantile_upper_bound_is_exclusive(self):
        """
        `q < 1.0`：1.0 必须**被拒**。放宽成 `<=` 会让分位取到 1.0 ——
        阈值等于历史最大波动，`vol > vol_cut` 永远为假，
        **high_vol 这个状态从此再也不会出现**，而没有任何报错。
        """
        with pytest.raises(ValueError, match="vol_quantile"):
            RegimeDetector(vol_quantile=1.0)
        assert RegimeDetector(vol_quantile=0.999).vol_quantile == 0.999

    def test_vol_quantile_below_half_is_rejected(self):
        with pytest.raises(ValueError, match="vol_quantile"):
            RegimeDetector(vol_quantile=0.4999)

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="仅支持 'trend'"):
            RegimeDetector().fit(_ret(np.zeros(200)), method="ema")


# ===========================================================================
# B. 样本量下限
# ===========================================================================

class TestSampleSize:

    def test_exactly_enough_data_is_accepted(self):
        """
        `if len(ret) < trend_window + vol_window: raise` —— **严格小于**。
        放宽成 `<=` 会让样本量**恰好够**时也抛错，回测起点那一折白白报废。
        """
        d = RegimeDetector(trend_window=10, vol_window=5)
        d.fit(_ret(np.full(15, 0.001)))          # 恰好 15 = 10 + 5
        assert d.predict() is not None

    def test_one_row_short_is_rejected(self):
        d = RegimeDetector(trend_window=10, vol_window=5)
        with pytest.raises(ValueError, match="数据不足"):
            d.fit(_ret(np.full(14, 0.001)))

    def test_nan_rows_do_not_count_towards_the_requirement(self):
        """`market_returns.dropna()` —— NaN 不该被算进样本量。"""
        vals = np.full(20, 0.001)
        vals[:8] = np.nan
        d = RegimeDetector(trend_window=10, vol_window=5)
        with pytest.raises(ValueError, match="数据不足"):
            d.fit(_ret(vals))                    # 有效只有 12 < 15


# ===========================================================================
# C. 三个判定阈值
# ===========================================================================

class TestLabelThresholds:

    @staticmethod
    def _fit(trend_per_day: float, n: int = 120, tw: int = 20, vw: int = 10,
             q: float = 0.99) -> RegimeDetector:
        """
        常数日收益 → 窗口累计收益恒定、滚动波动率恒为 0。
        vol_quantile 取 0.99 让高波动几乎不触发，好把趋势阈值单独隔离出来。
        """
        d = RegimeDetector(trend_window=tw, vol_window=vw,
                           trend_threshold=0.05, vol_quantile=q)
        return d.fit(_ret(np.full(n, trend_per_day)))

    @staticmethod
    def _spike(spike: float, n: int = 120, tw: int = 20, vw: int = 10,
               thr: float = 0.5):
        """
        全零收益里插一个 `spike`：含它的 tw 个窗口的累计收益**恰好等于** spike
        （1×1×…×(1+spike) 在浮点上是精确的，`- 1.0` 也精确），
        其余窗口为 0。阈值取 |spike| → 那 tw 天正好踩在边界上。

        用常数日收益反推 r 使 (1+r)^20 = 1.05 是行不通的：
        乘方再减一在浮点上落在 0.05000000000000204，**边界值不可构造** ——
        这正是 A 档总结过的"浮点边界需要反解"那一类，第一版就栽在这里。
        """
        v = np.zeros(n)
        v[n // 2] = spike
        d = RegimeDetector(trend_window=tw, vol_window=vw,
                           trend_threshold=thr, vol_quantile=0.99)
        d.fit(_ret(v))
        return d.predict().iloc[n // 2: n // 2 + tw].dropna()

    def test_trend_exactly_at_the_threshold_is_not_bull(self):
        """
        `labels[trend > threshold] = "bull"` —— **严格大于**。
        放宽成 `>=` 会让恰好踩线的窗口被标成牛市：阈值的语义是"超过才算"，
        踩线算牛市等于**每个阈值都往下挪了一格**。
        （踩线那几天可能因为波动尖峰被标成 high_vol —— 那是另一条规则，
        这里只断言"不是 bull"。）
        """
        seg = self._spike(0.5)
        assert "bull" not in set(seg), (
            f"窗口累计收益恰好等于阈值却被标成了 {sorted(set(seg))} —— "
            f"`trend > threshold` 被放宽成了 `>=`")

    def test_trend_above_the_threshold_is_bull(self):
        tw = 20
        r = 1.20 ** (1 / tw) - 1.0
        labels = self._fit(r, n=120, tw=tw).predict().dropna()
        assert (labels == "bull").all(), f"明显上涨没有被标成牛市：{set(labels)}"

    def test_trend_exactly_at_the_negative_threshold_is_not_bear(self):
        """`labels[trend < -threshold]` —— 对称的另一侧，同样是严格不等号。"""
        seg = self._spike(-0.5)
        assert "bear" not in set(seg), (
            f"窗口累计收益恰好等于 -阈值却被标成了 {sorted(set(seg))}")

    def test_trend_below_the_negative_threshold_is_bear(self):
        tw = 20
        r = 0.80 ** (1 / tw) - 1.0
        labels = self._fit(r, n=120, tw=tw).predict().dropna()
        assert (labels == "bear").all(), f"明显下跌没有被标成熊市：{set(labels)}"

    def test_flat_volatility_never_triggers_high_vol(self):
        """
        `labels[vol > vol_cut] = "high_vol"` —— **严格大于**。
        常数收益序列的滚动波动率恒为 0，扩展分位数也是 0：
        放宽成 `>=` 会让 `0 >= 0` 成立，**整段历史全被标成高波动**，
        牛熊标签被彻底覆盖掉。
        """
        labels = self._fit(0.001, n=120).predict().dropna()
        assert "high_vol" not in set(labels), (
            f"零波动序列被标成了高波动：{set(labels)} —— `vol > vol_cut` 被放宽")

    def test_a_volatility_spike_does_trigger_high_vol(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(0, 0.002, 200)
        vals[150:] = rng.normal(0, 0.05, 50)      # 后段波动放大 25 倍
        d = RegimeDetector(trend_window=20, vol_window=10, vol_quantile=0.8)
        labels = d.fit(_ret(vals)).predict().dropna()
        assert "high_vol" in set(labels), "波动明显放大却没有出现 high_vol"

    def test_high_vol_overrides_the_trend_label(self):
        """高波动是最后一条赋值，必须覆盖牛熊。"""
        tw = 20
        r = 1.50 ** (1 / tw) - 1.0
        vals = np.full(200, r)
        vals[150:] += np.random.default_rng(1).normal(0, 0.05, 50)
        d = RegimeDetector(trend_window=tw, vol_window=10, vol_quantile=0.8)
        labels = d.fit(_ret(vals)).predict().dropna()
        assert set(labels) >= {"bull", "high_vol"}, (
            f"高波动没有覆盖趋势标签：{set(labels)}")

    def test_warmup_rows_carry_no_label(self):
        d = self._fit(0.001, n=120, tw=20, vw=10)
        labels = d.predict()
        warmup = max(20, 10)
        assert labels.iloc[: warmup - 1].isna().all(), (
            "预热期内出现了标签 —— 窗口未满就下了结论")
        assert labels.iloc[warmup - 1:].notna().all()

    def test_every_label_is_a_known_regime(self):
        rng = np.random.default_rng(2)
        d = RegimeDetector(trend_window=20, vol_window=10)
        labels = d.fit(_ret(rng.normal(0.0005, 0.01, 300))).predict().dropna()
        assert set(labels) <= set(REGIMES), f"出现了未知 regime：{set(labels) - set(REGIMES)}"


# ===========================================================================
# D. 滚动累计收益的算法
# ===========================================================================

class TestTrendComputation:

    def test_trend_is_a_compounded_return_not_a_sum(self):
        """
        `(1 + ret).rolling(w).apply(np.prod) - 1` 是**复利**累计。
        若被改成求和，1% × 20 天会算成 20% 而不是 22.02%，
        阈值判定整体偏松；窗口越长偏得越多。
        """
        tw = 20
        r = 0.01
        d = RegimeDetector(trend_window=tw, vol_window=5, trend_threshold=0.21)
        labels = d.fit(_ret(np.full(80, r))).predict().dropna()
        compounded = (1 + r) ** tw - 1        # 0.2202 > 0.21 → bull
        simple_sum = r * tw                   # 0.2000 < 0.21 → sideways
        assert compounded > 0.21 > simple_sum, "构造的阈值没有夹在两种算法之间"
        assert (labels == "bull").all(), (
            f"复利累计收益 {compounded:.4f} 超过阈值 0.21 却没标成牛市："
            f"{set(labels)} —— 疑似退化成了简单求和")

    def test_raw_flag_does_not_change_the_result(self):
        """
        `rolling(w).apply(np.prod, raw=True)` 里的 `raw`。

        这不是一处盲区，是一处**真等价**：`raw=False` 会让 pandas 为每个窗口
        构造一个 Series 再传给 np.prod，而 `np.prod(Series)` 与
        `np.prod(ndarray)` 返回同一个标量 —— 结果逐位相同，只是慢几十倍。
        本条既是等价性证明，也是失效告警：哪天 np.prod 对 Series 的行为变了，
        这里会红。
        """
        s = _ret(np.random.default_rng(9).normal(0.001, 0.01, 120))
        a = (1.0 + s).rolling(20).apply(np.prod, raw=True) - 1.0
        b = (1.0 + s).rolling(20).apply(np.prod, raw=False) - 1.0
        pd.testing.assert_series_equal(
            a, b, obj="raw=True/False 给出了不同的累计收益 —— 该变异不再等价，需补用例")

    def test_rolling_is_right_aligned_with_no_look_ahead(self):
        """
        标签只能用 t 及之前的数据。把一段后期的大涨换成大跌，
        **前面的标签必须一个字都不变**。
        """
        base = np.full(200, 0.001)
        a = base.copy()
        b = base.copy()
        b[150:] = -0.05                        # 只动后 50 天
        d1 = RegimeDetector(trend_window=20, vol_window=10).fit(_ret(a))
        d2 = RegimeDetector(trend_window=20, vol_window=10).fit(_ret(b))
        l1 = d1.predict().iloc[:150]
        l2 = d2.predict().iloc[:150]
        pd.testing.assert_series_equal(l1, l2, obj="改动未来数据改变了历史标签")


# ===========================================================================
# E. regime_to_alpha_weights
# ===========================================================================

class TestRegimeWeights:

    POOL = [
        {"dsl": "ts_mean(close,20)", "sharpe_oos": 1.5},
        {"dsl": "rank(ts_delta(close,5))", "sharpe_oos": 0.0},
        {"dsl": "ts_std(returns,10)", "sharpe_oos": -2.0},
    ]

    def test_weights_are_all_positive_and_sum_to_one(self):
        """
        `base = max(sharpe, 0.0) + 0.1` —— `+` 改成 `-` 会让
        Sharpe <= 0 的候选 base 变成 **-0.1**，归一化之后出现负权重：
        一个表现最差的因子会被**做空配置**，而这里从来没打算做空。
        """
        w = RegimeDetector().regime_to_alpha_weights("bull", self.POOL)
        assert len(w) == 3
        assert all(v > 0 for v in w.values()), f"出现了非正权重：{w}"
        assert sum(w.values()) == pytest.approx(1.0)

    def test_a_zero_sharpe_candidate_still_gets_a_floor_weight(self):
        """`+ 0.1` 的作用就是给零/负 Sharpe 一个下限，不让它权重归零。"""
        w = RegimeDetector().regime_to_alpha_weights("sideways", self.POOL)
        assert w["ts_std(returns,10)"] > 0, "负 Sharpe 的候选权重被归零了"

    def test_higher_sharpe_gets_more_weight_within_a_family(self):
        pool = [{"dsl": "ts_mean(close,20)", "sharpe_oos": 2.0},
                {"dsl": "ts_mean(close,60)", "sharpe_oos": 0.5}]
        w = RegimeDetector().regime_to_alpha_weights("bull", pool)
        assert w["ts_mean(close,20)"] > w["ts_mean(close,60)"]

    def test_regime_actually_tilts_the_weights(self):
        a = RegimeDetector().regime_to_alpha_weights("bull", self.POOL)
        b = RegimeDetector().regime_to_alpha_weights("high_vol", self.POOL)
        assert a != b, "换了 regime 权重完全一样 —— 家族倾斜没有生效"

    def test_unknown_regime_is_rejected(self):
        with pytest.raises(ValueError, match="未知 regime"):
            RegimeDetector().regime_to_alpha_weights("moon", self.POOL)

    def test_empty_pool_returns_empty(self):
        assert RegimeDetector().regime_to_alpha_weights("bull", []) == {}

    def test_entries_without_dsl_are_skipped(self):
        pool = [{"dsl": "", "sharpe_oos": 9.0},
                {"dsl": "rank(close)", "sharpe_oos": 1.0}]
        w = RegimeDetector().regime_to_alpha_weights("bull", pool)
        assert set(w) == {"rank(close)"}, f"空 dsl 的条目进了结果：{w}"


# ===========================================================================
# F. 未拟合时的行为
# ===========================================================================

class TestUnfitted:

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            RegimeDetector().predict()

    def test_current_regime_before_fit_raises(self):
        with pytest.raises(Exception):
            RegimeDetector().current_regime()

    def test_current_regime_falls_back_when_everything_is_warmup(self):
        """全是预热期（无有效标签）时返回 sideways，而不是抛错或 NaN。"""
        d = RegimeDetector(trend_window=10, vol_window=5)
        d.fit(_ret(np.full(15, 0.001)))
        d._labels[:] = np.nan
        assert d.current_regime() == "sideways"

    def test_predict_reindexes_forward_only(self):
        d = RegimeDetector(trend_window=10, vol_window=5)
        d.fit(_ret(np.full(40, 0.001)))
        future = pd.bdate_range(IDX[39], periods=5)
        out = d.predict(future)
        assert out.notna().all(), "前向填充没有覆盖到未来日期"
        assert (out == out.iloc[0]).all(), "未来日期没有沿用最后已知状态"


# ===========================================================================
# G. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "L103 `.rolling(w).apply(np.prod, raw=True)` → `raw=False`":
        "`raw` 只决定 pandas 传给 apply 的是 ndarray 还是 Series。"
        "`np.prod` 对两者返回同一个标量，因此滚动累计收益逐位相同 —— "
        "差别只在性能（raw=False 每个窗口多构造一个 Series）。"
        "机械验证见 test_raw_flag_does_not_change_the_result，"
        "它同时是失效告警：np.prod 对 Series 的行为一旦变化，那条会红。",
}


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 1
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
