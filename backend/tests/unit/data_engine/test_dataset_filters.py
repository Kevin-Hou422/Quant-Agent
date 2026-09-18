"""
data_engine/dataset_filters.py —— 数据集选定之后的动态筛选

**此前零测试**（54 个变异点，D 档里最大的一个）。

这一层决定 **universe 里到底剩哪些票**。它错了不会抛异常、不会让任何
测试变红 —— 只会让回测/实盘跑在一个**与配置不符的标的集合**上：
说好只要高流动性的，结果混进了一堆日成交额 500 万的小票；
说好 bull regime 才开仓，结果 bear 里照样满仓。

八类筛选全部落在**阈值比较**上，所以变异测试打的就是这些比较符：
`>` 翻成 `>=`、`<` 翻成 `<=`，恰好卡在阈值上的那一只票就会易主。
因此本文件的核心手法是**把指标构造得精确等于阈值**
（`adv20 == 100_000_000.0` 而不是"差不多一亿"），
让每一个比较符的两种取值都能被区分出来。

另有三处算术，靠**参考实现逐位比对**（rtol=1e-12）来钉：
  - `adv20 = (close * volume).rolling(20, min_periods=10).mean()`
  - `returns = np.log(close / close.shift(1))`
  - `beta = cov / spy_var`
这些量在下游只参与比较，量纲一换往往仍能通过"有票通过"这种松断言，
必须直接钉数值本身。
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.dataset_filters import (
    _BETA_HIGH,
    _BETA_LOW,
    _CORR_HIGH,
    _CORR_LOW,
    _LIQUIDITY_THRESHOLDS,
    _MARKET_CAP_THRESHOLDS,
    _MOMENTUM_THRESHOLD,
    _SIDEWAYS_PCT,
    _SPY_MA_LONG,
    _SPY_MA_SHORT,
    _VOLATILITY_THRESHOLDS,
    VALID_FILTER_VALUES,
    DatasetFilterEngine,
    FilterConfig,
    FilterResult,
    apply_filters,
    validate_filter_config,
)

E = DatasetFilterEngine


def _s(**kv) -> pd.Series:
    """ticker → 指标值。用 dict 构造是为了让每个断言都能指名道姓。"""
    return pd.Series(kv, dtype=float)


# ===========================================================================
# A. 流动性阈值 —— 四档，每档的每个比较符都钉在阈值上
# ===========================================================================

class TestLiquidityThresholds:

    def test_ultra_high_excludes_the_ticker_sitting_exactly_on_the_threshold(self):
        """
        `adv20 > _LIQUIDITY_THRESHOLDS["ultra_high"]`

        `>` 翻成 `>=` 只会改变**恰好等于一亿**的那一只票的去留。
        构造 ON 精确等于阈值、UNDER 差一个最小浮点步长、OVER 明确在上方，
        三者的归属把 `>` 与 `>=` 完全区分开。
        """
        thr = _LIQUIDITY_THRESHOLDS["ultra_high"]
        adv = _s(UNDER=np.nextafter(thr, 0.0), ON=float(thr), OVER=thr * 2.0)
        assert E._apply_liquidity(adv, "ultra_high") == {"OVER"}, (
            "恰好等于一亿的标的被放进了 ultra_high —— `>` 被翻成了 `>=`")

    def test_high_excludes_the_ticker_sitting_exactly_on_the_threshold(self):
        thr = _LIQUIDITY_THRESHOLDS["high"]
        adv = _s(UNDER=np.nextafter(thr, 0.0), ON=float(thr), OVER=thr * 2.0)
        assert E._apply_liquidity(adv, "high") == {"OVER"}

    def test_medium_is_open_below_and_closed_above(self):
        """
        `(adv20 > medium_lo) & (adv20 <= medium_hi)`

        两个比较符方向相反 —— 下界开、上界闭。任何一个被翻，
        LO（恰好一千万）或 HI（恰好五千万）就会易主。
        """
        lo = _LIQUIDITY_THRESHOLDS["medium_lo"]
        hi = _LIQUIDITY_THRESHOLDS["medium_hi"]
        adv = _s(BELOW_LO=np.nextafter(lo, 0.0),
                 ON_LO=float(lo),
                 MID=(lo + hi) / 2.0,
                 ON_HI=float(hi),
                 ABOVE_HI=np.nextafter(hi, np.inf))
        assert E._apply_liquidity(adv, "medium") == {"MID", "ON_HI"}, (
            "medium 区间的开闭搞反了：下界应当开（不含 1e7）、上界应当闭（含 5e7）")

    def test_low_includes_the_ticker_sitting_exactly_on_the_threshold(self):
        """`adv20 <= low` —— 闭区间。翻成 `<` 会把恰好一千万的票踢掉。"""
        thr = _LIQUIDITY_THRESHOLDS["low"]
        adv = _s(UNDER=np.nextafter(thr, 0.0), ON=float(thr),
                 OVER=np.nextafter(thr, np.inf))
        assert E._apply_liquidity(adv, "low") == {"UNDER", "ON"}, (
            "恰好一千万的标的被排除出 low —— `<=` 被翻成了 `<`")

    def test_the_four_levels_use_the_documented_constants(self):
        """
        阈值常量本身被改（1e8 → 1e9）不会让上面任何一条红 ——
        因为它们都用常量自身构造输入。这一条直接钉死字面量。
        """
        assert _LIQUIDITY_THRESHOLDS == {
            "ultra_high": 100_000_000,
            "high":        50_000_000,
            "medium_lo":   10_000_000,
            "medium_hi":   50_000_000,
            "low":         10_000_000,
        }, f"流动性阈值被改动：{_LIQUIDITY_THRESHOLDS}"

    def test_nan_metrics_are_rejected_not_silently_passed(self):
        """
        `adv20[mask.fillna(False)]` —— `fillna(False)` 去掉会让
        NaN 参与索引，pandas 报错或（更糟）把取不到成交额的票放行。
        """
        adv = _s(GOOD=2e8, MISSING=np.nan)
        assert E._apply_liquidity(adv, "ultra_high") == {"GOOD"}, (
            "adv20 为 NaN 的标的被放行 —— `fillna(False)` 没了")

    def test_a_nullable_dtype_missing_value_is_also_rejected(self):
        """
        **首测存活项（L333）**：`fillna(False)` 翻成 `fillna(True)` 在
        `float64` 上完全观察不到 —— `np.nan > x` 直接给 `False`，
        掩码里根本不会出现缺失值，`fillna` 是空操作。

        只有 **pandas 可空 dtype**（`Float64`，parquet / arrow 回读时的
        自然产物）比较后得到 `boolean` 掩码，缺失位是 `pd.NA`，
        这时 `fillna` 才真正决定该标的的去留：
        `False` = 落选（正确），`True` = **静默放行一只成交额未知的票**。
        """
        adv = pd.Series({"GOOD": 2e8, "MISSING": pd.NA}, dtype="Float64")
        assert (adv > 1e8).isna().any(), "前提失效：可空 dtype 的比较没有产生 NA 掩码"
        assert E._apply_liquidity(adv, "ultra_high") == {"GOOD"}, (
            "成交额缺失（pd.NA）的标的被放行 —— `fillna(False)` 被翻成了 `fillna(True)`")

    def test_an_unknown_level_raises_instead_of_passing_everything(self):
        """
        `raise ValueError` 被删或改成 `return set()` 都很危险：
        前者让拼错的配置静默生效，后者让整个 universe 清零。
        """
        with pytest.raises(ValueError, match="Unknown liquidity level"):
            E._apply_liquidity(_s(A=1.0), "hgih")


# ===========================================================================
# A2. 缺失值掩码 —— 六个 `_apply_*` 共有的 `fillna(False)`
# ===========================================================================

class TestMissingValueMaskAcrossAllFilters:
    """
    **首测的 6 个存活项**（L333/346/356/371/381/391）集中在这里。

    六个筛选函数末尾都是 `set(x[mask.fillna(False)].index)`。
    在 `float64` 上这个 `fillna` 是**死代码** —— NaN 参与比较直接得
    `False`，掩码里不会有缺失。所以 `fillna(True)` 的变异在常规
    输入下完全观察不到，六处一起存活。

    但它不是真的死代码：pandas 的可空 dtype（`Float64`/`Int64`，
    pyarrow 后端读 parquet 时的默认产物）比较后得到 `boolean` 掩码，
    缺失位是 `pd.NA`。这时 `fillna(True)` = **把指标缺失的标的
    当作通过筛选**，universe 里会混进一批数据本身就没拿到的票。

    本项目的 `local_parquet_provider` 正是从 parquet 读面板，
    所以这条路径是可达的，不是假想。
    """

    CASES = [
        ("_apply_liquidity",  "ultra_high",       2e8,  _LIQUIDITY_THRESHOLDS["ultra_high"]),
        ("_apply_volatility", "high_vol",         0.10, _VOLATILITY_THRESHOLDS["high_vol"]),
        ("_apply_momentum",   "strong_uptrend",   0.50, _MOMENTUM_THRESHOLD),
        ("_apply_market_cap", "mega_cap",         3e12, _MARKET_CAP_THRESHOLDS["mega_cap"]),
        ("_apply_beta",       "high_beta",        2.0,  _BETA_HIGH),
        ("_apply_corr",       "high_corr",        0.95, _CORR_HIGH),
    ]

    @pytest.mark.parametrize("fn_name,level,good,_thr", CASES,
                             ids=[c[0] for c in CASES])
    def test_a_nullable_missing_metric_never_passes(self, fn_name, level, good, _thr):
        fn = getattr(E, fn_name)
        metric = pd.Series({"GOOD": good, "MISSING": pd.NA}, dtype="Float64")
        got = fn(metric, level)
        assert got == {"GOOD"}, (
            f"{fn_name} 放行了指标缺失（pd.NA）的标的：{sorted(got)} —— "
            f"`mask.fillna(False)` 被翻成了 `fillna(True)`")

    @pytest.mark.parametrize("fn_name,level,good,_thr", CASES,
                             ids=[c[0] for c in CASES])
    def test_the_nullable_mask_really_carries_a_missing_entry(
            self, fn_name, level, good, _thr):
        """
        上一条的前提：可空 dtype 比较后掩码里**确实**有 NA。
        pandas 将来若改掉这个行为，这条会先红，提醒上一条已失去杀伤力
        （而不是让它悄悄退化成恒真断言）。
        """
        metric = pd.Series({"GOOD": good, "MISSING": pd.NA}, dtype="Float64")
        mask = metric > _thr
        assert str(mask.dtype) == "boolean", f"掩码 dtype 是 {mask.dtype}，不是 boolean"
        assert mask.isna().any(), "可空 dtype 的比较没有产生 NA 掩码，前提已失效"

    def test_plain_float_nan_produces_no_missing_mask_at_all(self):
        """
        对照组：在 `float64` 上 `np.nan > x` 是 `False` 而不是 NaN ——
        这正是六个 `fillna(False)` 在常规输入下杀不掉的原因。
        把这个事实钉下来，免得后来者以为上面那组用例是多余的。
        """
        metric = pd.Series({"A": np.nan}, dtype="float64")
        mask = metric > 1.0
        assert str(mask.dtype) == "bool"
        assert not mask.isna().any()
        assert mask.iloc[0] is np.False_ or mask.iloc[0] == False   # noqa: E712


# ===========================================================================
# B. 波动率阈值
# ===========================================================================

class TestVolatilityThresholds:

    def test_high_vol_excludes_the_value_exactly_on_the_threshold(self):
        thr = _VOLATILITY_THRESHOLDS["high_vol"]
        vol = _s(UNDER=np.nextafter(thr, 0.0), ON=float(thr),
                 OVER=np.nextafter(thr, np.inf))
        assert E._apply_volatility(vol, "high_vol") == {"OVER"}, (
            "波动率恰好 4% 的标的被算作 high_vol —— `>` 被翻成了 `>=`")

    def test_medium_vol_is_open_below_and_closed_above(self):
        lo = _VOLATILITY_THRESHOLDS["medium_lo"]
        hi = _VOLATILITY_THRESHOLDS["medium_hi"]
        vol = _s(BELOW_LO=np.nextafter(lo, 0.0),
                 ON_LO=float(lo),
                 MID=(lo + hi) / 2.0,
                 ON_HI=float(hi),
                 ABOVE_HI=np.nextafter(hi, np.inf))
        assert E._apply_volatility(vol, "medium_vol") == {"MID", "ON_HI"}

    def test_low_vol_includes_the_value_exactly_on_the_threshold(self):
        thr = _VOLATILITY_THRESHOLDS["low_vol"]
        vol = _s(UNDER=np.nextafter(thr, 0.0), ON=float(thr),
                 OVER=np.nextafter(thr, np.inf))
        assert E._apply_volatility(vol, "low_vol") == {"UNDER", "ON"}

    def test_the_volatility_constants_are_two_and_four_percent(self):
        assert _VOLATILITY_THRESHOLDS == {
            "high_vol":  0.04, "medium_lo": 0.02,
            "medium_hi": 0.04, "low_vol":   0.02,
        }, f"波动率阈值被改动：{_VOLATILITY_THRESHOLDS}"

    def test_nan_volatility_is_rejected(self):
        assert E._apply_volatility(_s(GOOD=0.10, MISSING=np.nan), "high_vol") == {"GOOD"}

    def test_an_unknown_level_raises(self):
        with pytest.raises(ValueError, match="Unknown volatility level"):
            E._apply_volatility(_s(A=0.01), "mid_vol")


# ===========================================================================
# C. 动量区间 —— 唯一一处用到负号的阈值
# ===========================================================================

class TestMomentumRegime:

    def test_strong_uptrend_excludes_the_value_exactly_on_the_threshold(self):
        thr = _MOMENTUM_THRESHOLD
        mom = _s(UNDER=np.nextafter(thr, 0.0), ON=float(thr),
                 OVER=np.nextafter(thr, np.inf))
        assert E._apply_momentum(mom, "strong_uptrend") == {"OVER"}, (
            "60 日涨幅恰好 15% 的标的被算作强势上涨 —— `>` 被翻成了 `>=`")

    def test_strong_downtrend_uses_the_negated_threshold(self):
        """
        `mom60 < -_MOMENTUM_THRESHOLD`

        一元负号被去掉会让条件变成 `< 0.15` —— 于是**几乎所有票**
        都被判为"强势下跌"，包括涨了 10% 的。
        """
        mom = _s(CRASH=-0.30, FLAT=0.0, RALLY=0.10)
        assert E._apply_momentum(mom, "strong_downtrend") == {"CRASH"}, (
            "横盘或上涨的标的被判为强势下跌 —— `-_MOMENTUM_THRESHOLD` 的负号没了")

    def test_strong_downtrend_excludes_the_value_exactly_on_the_threshold(self):
        thr = -_MOMENTUM_THRESHOLD
        mom = _s(BELOW=np.nextafter(thr, -np.inf), ON=float(thr),
                 ABOVE=np.nextafter(thr, 0.0))
        assert E._apply_momentum(mom, "strong_downtrend") == {"BELOW"}, (
            "跌幅恰好 15% 的标的被算作强势下跌 —— `<` 被翻成了 `<=`")

    def test_the_momentum_threshold_is_fifteen_percent(self):
        assert _MOMENTUM_THRESHOLD == 0.15

    def test_nan_momentum_is_rejected(self):
        assert E._apply_momentum(_s(GOOD=0.5, MISSING=np.nan), "strong_uptrend") == {"GOOD"}

    def test_an_unknown_level_raises(self):
        with pytest.raises(ValueError, match="Unknown momentum_regime"):
            E._apply_momentum(_s(A=0.5), "uptrend")


# ===========================================================================
# D. 市值阈值
# ===========================================================================

class TestMarketCapThresholds:

    def test_mega_cap_excludes_the_value_exactly_on_the_threshold(self):
        thr = _MARKET_CAP_THRESHOLDS["mega_cap"]
        mc = _s(UNDER=np.nextafter(thr, 0.0), ON=float(thr), OVER=thr * 2.0)
        assert E._apply_market_cap(mc, "mega_cap") == {"OVER"}

    def test_large_cap_excludes_the_value_exactly_on_the_threshold(self):
        thr = _MARKET_CAP_THRESHOLDS["large_cap"]
        mc = _s(UNDER=np.nextafter(thr, 0.0), ON=float(thr), OVER=thr * 2.0)
        assert E._apply_market_cap(mc, "large_cap") == {"OVER"}

    def test_mid_cap_is_open_below_and_closed_above(self):
        lo = _MARKET_CAP_THRESHOLDS["mid_cap_lo"]
        hi = _MARKET_CAP_THRESHOLDS["mid_cap_hi"]
        mc = _s(BELOW_LO=np.nextafter(lo, 0.0),
                ON_LO=float(lo),
                MID=(lo + hi) / 2.0,
                ON_HI=float(hi),
                ABOVE_HI=np.nextafter(hi, np.inf))
        assert E._apply_market_cap(mc, "mid_cap") == {"MID", "ON_HI"}

    def test_small_cap_includes_the_value_exactly_on_the_threshold(self):
        thr = _MARKET_CAP_THRESHOLDS["small_cap"]
        mc = _s(UNDER=np.nextafter(thr, 0.0), ON=float(thr),
                OVER=np.nextafter(thr, np.inf))
        assert E._apply_market_cap(mc, "small_cap") == {"UNDER", "ON"}

    def test_large_cap_and_mid_cap_share_the_same_ten_billion_boundary(self):
        """
        `large_cap` 的下界与 `mid_cap` 的上界是同一个数。
        任一常量被单独改动，这条互补关系就断了 ——
        中间会出现一段既不算中盘也不算大盘的真空。
        """
        assert _MARKET_CAP_THRESHOLDS["large_cap"] == _MARKET_CAP_THRESHOLDS["mid_cap_hi"]
        assert _MARKET_CAP_THRESHOLDS["small_cap"] == _MARKET_CAP_THRESHOLDS["mid_cap_lo"]
        on_boundary = _s(X=float(_MARKET_CAP_THRESHOLDS["large_cap"]))
        assert E._apply_market_cap(on_boundary, "mid_cap") == {"X"}
        assert E._apply_market_cap(on_boundary, "large_cap") == set()

    def test_the_market_cap_constants_are_unchanged(self):
        assert _MARKET_CAP_THRESHOLDS == {
            "mega_cap":  200_000_000_000,
            "large_cap":  10_000_000_000,
            "mid_cap_lo":  2_000_000_000,
            "mid_cap_hi": 10_000_000_000,
            "small_cap":   2_000_000_000,
        }, f"市值阈值被改动：{_MARKET_CAP_THRESHOLDS}"

    def test_a_ticker_with_no_market_cap_is_rejected(self):
        """
        取不到市值的票按 NaN 参与筛选（见源码里那条 warning）。
        NaN 必须**落选**而不是被放行 —— 否则筛"大盘股"会混进未知市值的票。
        """
        assert E._apply_market_cap(_s(AAPL=3e12, UNKNOWN=np.nan), "mega_cap") == {"AAPL"}

    def test_an_unknown_level_raises(self):
        with pytest.raises(ValueError, match="Unknown market_cap level"):
            E._apply_market_cap(_s(A=1e9), "huge_cap")


# ===========================================================================
# E. Beta / 相关性
# ===========================================================================

class TestBetaAndCorrelation:

    def test_high_beta_excludes_the_value_exactly_on_the_threshold(self):
        b = _s(UNDER=np.nextafter(_BETA_HIGH, 0.0), ON=float(_BETA_HIGH),
               OVER=np.nextafter(_BETA_HIGH, np.inf))
        assert E._apply_beta(b, "high_beta") == {"OVER"}

    def test_low_beta_excludes_the_value_exactly_on_the_threshold(self):
        b = _s(UNDER=np.nextafter(_BETA_LOW, 0.0), ON=float(_BETA_LOW),
               OVER=np.nextafter(_BETA_LOW, np.inf))
        assert E._apply_beta(b, "low_beta") == {"UNDER"}, (
            "beta 恰好 0.8 的标的被算作低 beta —— `<` 被翻成了 `<=`")

    def test_the_beta_band_leaves_a_gap_between_low_and_high(self):
        """
        0.8 ≤ beta ≤ 1.2 两边都不属于 —— 这是设计意图（只要极端）。
        任一常量被改会让这个"中性带"移位或消失。
        """
        assert _BETA_LOW == 0.8 and _BETA_HIGH == 1.2
        mid = _s(NEUTRAL=1.0)
        assert E._apply_beta(mid, "high_beta") == set()
        assert E._apply_beta(mid, "low_beta") == set()

    def test_high_corr_excludes_the_value_exactly_on_the_threshold(self):
        c = _s(UNDER=np.nextafter(_CORR_HIGH, 0.0), ON=float(_CORR_HIGH),
               OVER=np.nextafter(_CORR_HIGH, np.inf))
        assert E._apply_corr(c, "high_corr") == {"OVER"}

    def test_low_corr_excludes_the_value_exactly_on_the_threshold(self):
        c = _s(UNDER=np.nextafter(_CORR_LOW, 0.0), ON=float(_CORR_LOW),
               OVER=np.nextafter(_CORR_LOW, np.inf))
        assert E._apply_corr(c, "low_corr") == {"UNDER"}

    def test_the_correlation_band_is_point_three_to_point_seven(self):
        assert _CORR_LOW == 0.3 and _CORR_HIGH == 0.7
        mid = _s(NEUTRAL=0.5)
        assert E._apply_corr(mid, "high_corr") == set()
        assert E._apply_corr(mid, "low_corr") == set()

    def test_nan_beta_and_corr_are_rejected(self):
        assert E._apply_beta(_s(OK=2.0, MISSING=np.nan), "high_beta") == {"OK"}
        assert E._apply_corr(_s(OK=0.9, MISSING=np.nan), "high_corr") == {"OK"}

    def test_unknown_levels_raise(self):
        with pytest.raises(ValueError, match="Unknown beta level"):
            E._apply_beta(_s(A=1.0), "mid_beta")
        with pytest.raises(ValueError, match="Unknown correlation level"):
            E._apply_corr(_s(A=0.5), "mid_corr")


# ===========================================================================
# F. Regime 检测 —— 唯一一个"数据集级"的筛选（不通过就全灭）
# ===========================================================================

def _spy_series(values) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


class TestRegimeDetection:

    def test_a_short_history_falls_back_to_sideways(self):
        """
        `if len(spy_close) < _SPY_MA_LONG: return "sideways"`

        `<` 翻成 `<=` 只会改变**恰好 200 根**的那一种情况。
        构造 199 / 200 两条序列，且让 200 根那条的真实 regime
        **不是** sideways，两种取值才能被区分。
        """
        assert E._detect_regime(_spy_series(np.arange(1, 200) * 1.0)) == "sideways", (
            "只有 199 根 K 线时应当退回 sideways")

        rising = _spy_series(100.0 * (1.02 ** np.arange(_SPY_MA_LONG)))
        assert len(rising) == _SPY_MA_LONG
        assert E._detect_regime(rising) == "bull", (
            "恰好 200 根时应当真正去算均线，而不是退回 sideways —— "
            "`len(...) < 200` 被翻成了 `<=`")

    def test_price_below_the_200d_average_is_bear(self):
        """
        `if last > ma200: ... return "bear"`

        持续下跌 → 现价必然低于 200 日均线。
        """
        falling = _spy_series(100.0 * (0.99 ** np.arange(300)))
        assert E._detect_regime(falling) == "bear"

    def test_price_exactly_on_the_200d_average_is_bear(self):
        """
        `last > ma200` 翻成 `>=` 只改变**现价恰好等于 200 日均线**的情形。
        用常数序列构造：任何窗口的均值都精确等于该常数，
        于是 `last == ma200` 精确成立（浮点上也成立）。

        此时若走 `>=` 分支，`pct_from_50 = |last-ma50|/ma50 = 0 < 0.03`
        → 返回 sideways；走 `>` 分支 → 返回 bear。两者可区分。
        """
        flat = _spy_series(np.full(300, 100.0))
        ma200 = flat.rolling(_SPY_MA_LONG).mean().iloc[-1]
        assert flat.iloc[-1] == ma200, "常数序列上现价与均线应当逐位相等"
        assert E._detect_regime(flat) == "bear", (
            "现价恰好等于 200 日均线时应当算 bear —— `>` 被翻成了 `>=`")

    def test_above_the_200d_but_hugging_the_50d_is_sideways(self):
        """
        `pct_from_50 < _SIDEWAYS_PCT` —— 站上 200 日线、
        但离 50 日线不足 3% → sideways（不是 bull）。

        构造：前 250 根缓慢上涨把 200 日均线压低，最后 50 根走平
        → 现价 > ma200，且现价 == ma50（偏离 0%）。
        """
        vals = list(100.0 * (1.004 ** np.arange(250))) + [100.0 * 1.004 ** 249] * 50
        s = _spy_series(vals)
        ma200 = s.rolling(_SPY_MA_LONG).mean().iloc[-1]
        ma50 = s.rolling(_SPY_MA_SHORT).mean().iloc[-1]
        assert s.iloc[-1] > ma200
        assert abs(s.iloc[-1] - ma50) / ma50 < _SIDEWAYS_PCT
        assert E._detect_regime(s) == "sideways"

    def test_above_the_200d_and_far_from_the_50d_is_bull(self):
        strong = _spy_series(100.0 * (1.01 ** np.arange(300)))
        ma50 = strong.rolling(_SPY_MA_SHORT).mean().iloc[-1]
        assert abs(strong.iloc[-1] - ma50) / ma50 > _SIDEWAYS_PCT
        assert E._detect_regime(strong) == "bull"

    def test_the_sideways_band_boundary_sits_at_three_percent(self):
        """
        `pct_from_50 < _SIDEWAYS_PCT`

        构造一段 249 根走平、最后一根跳涨 `x` 的序列：
        此时 `ma50 = M·(1 + x/50)`、`last = M·(1 + x)`，
        于是 `pct_from_50 ≈ x·49/50` —— 把 x 调到 3% 两侧，
        regime 就在 sideways / bull 之间翻转。
        阈值常量被改成任何别的数，这两条断言必有一条红。
        """
        base = list(100.0 * (1.004 ** np.arange(210)))     # 把 ma200 压低
        flat = float(base[-1])

        def regime_for(jump: float) -> str:
            path = base + [flat] * 49 + [flat * (1.0 + jump)]
            return E._detect_regime(_spy_series(path))

        # x·49/50 < 3%  →  x < 3.061%
        assert regime_for(0.020) == "sideways", "偏离约 1.96% 时不应当算 bull"
        # x·49/50 > 3%  →  x > 3.061%
        assert regime_for(0.040) == "bull", "偏离约 3.92% 时不应当算 sideways"

    def test_the_sideways_bull_frontier_is_a_single_float_wide(self):
        """
        把跳涨幅度二分到相邻浮点，确认 regime 恰好在
        `pct_from_50 == _SIDEWAYS_PCT` 这一点上翻转 ——
        阈值被改动 1e-9 都会让这个分界点整体平移。
        """
        base = list(100.0 * (1.004 ** np.arange(210)))
        flat = float(base[-1])

        def pct_for(jump: float) -> float:
            path = _spy_series(base + [flat] * 49 + [flat * (1.0 + jump)])
            ma50 = path.rolling(_SPY_MA_SHORT).mean().iloc[-1]
            return abs(path.iloc[-1] - ma50) / ma50

        lo, hi = 0.0, 0.10
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if mid <= lo or mid >= hi:
                break
            if pct_for(mid) < _SIDEWAYS_PCT:
                lo = mid
            else:
                hi = mid
        assert pct_for(lo) < _SIDEWAYS_PCT <= pct_for(hi)
        assert E._detect_regime(_spy_series(base + [flat] * 49
                                            + [flat * (1.0 + lo)])) == "sideways"
        assert E._detect_regime(_spy_series(base + [flat] * 49
                                            + [flat * (1.0 + hi)])) == "bull", (
            "偏离刚越过 3% 的那一侧没有翻成 bull —— 阈值常量被改了")

    def test_a_deviation_exactly_on_three_percent_is_bull_not_sideways(self):
        """
        **首测存活项（L407）**：`pct_from_50 < _SIDEWAYS_PCT` 翻成 `<=`
        只在偏离**精确等于** 0.03 时才有差别，靠随便调幅度是撞不上的
        （实测 pct 随最后一根价格每跳 1 ulp 变动约 36 ulp，会直接
        跨过 0.03 这个双精度值）。

        这里改为**让除法本身精确**：把最后 50 根构造成
        48 根 100 + 1 根 97 + 1 根 103 → 和恰好 5000、`ma50` 精确等于
        100.0，`(103.0 - 100.0) / 100.0` 精确等于字面量 0.03。
        前面 210 根压在 50.0，保证 `last > ma200`。

        此时 `<` 为假 → bull；`<=` 为真 → sideways，两者可区分。
        """
        path = [50.0] * 210 + [100.0] * 48 + [97.0] + [103.0]
        s = _spy_series(path)
        ma50 = s.rolling(_SPY_MA_SHORT).mean().iloc[-1]
        ma200 = s.rolling(_SPY_MA_LONG).mean().iloc[-1]
        # 前提：三个量都精确落在构造值上
        assert ma50 == 100.0, f"ma50 不是精确的 100.0，而是 {ma50!r}"
        assert s.iloc[-1] == 103.0
        assert s.iloc[-1] > ma200
        assert abs(s.iloc[-1] - ma50) / ma50 == _SIDEWAYS_PCT, (
            "偏离没有精确落在 3% 上，本用例已失去区分力")

        assert E._detect_regime(s) == "bull", (
            "偏离恰好 3% 时应当算 bull（区间左开右闭于 sideways 之外）—— "
            "`pct_from_50 < 0.03` 被翻成了 `<=`")

    def test_the_regime_constants_are_unchanged(self):
        assert _SPY_MA_LONG == 200 and _SPY_MA_SHORT == 50 and _SIDEWAYS_PCT == 0.03

    def test_the_two_moving_average_windows_are_not_interchangeable(self):
        """
        `rolling(_SPY_MA_LONG)` 与 `rolling(_SPY_MA_SHORT)` 被互换，
        或常量互相赋值 —— 在单调上涨序列上 ma50 > ma200，
        互换之后 sideways 判定的分母与被减数全变。
        直接比对两条均线本身。
        """
        s = _spy_series(100.0 * (1.01 ** np.arange(300)))
        ma200 = s.rolling(_SPY_MA_LONG).mean().iloc[-1]
        ma50 = s.rolling(_SPY_MA_SHORT).mean().iloc[-1]
        assert ma50 > ma200, "上涨序列里 50 日均线必须高于 200 日均线"


# ===========================================================================
# G. 滚动 beta / 相关性 —— 参考实现逐位比对
# ===========================================================================

def _panel(n_days: int = 120, seed: int = 7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n_days, freq="B")
    spy = pd.Series(rng.normal(0.0004, 0.011, n_days), index=idx)
    rets = pd.DataFrame({
        "HIGHB": spy * 1.8 + rng.normal(0, 0.002, n_days),
        "LOWB":  spy * 0.4 + rng.normal(0, 0.002, n_days),
        "INDEP": rng.normal(0, 0.010, n_days),
    }, index=idx)
    return rets, spy


class TestRollingBetaAndCorr:

    def test_beta_matches_the_textbook_cov_over_var(self):
        """
        `result[col] = float(cov) / float(spy_var)`

        `/` 翻成 `*` 会让 beta 变成 cov*var —— 数量级掉到 1e-8，
        于是**所有票都不是 high_beta**。但如果只断言
        "HIGHB 的 beta 比 LOWB 大"，这个变异照样通过
        （乘法保序）。所以必须逐位比对参考实现。
        """
        rets, spy = _panel()
        got = E._compute_beta(rets, spy)

        spy_al = spy.reindex(rets.index).fillna(0)
        var = spy_al.rolling(60, min_periods=20).var().iloc[-1]
        for col in rets.columns:
            cov = rets[col].fillna(0).rolling(60, min_periods=20).cov(spy_al).iloc[-1]
            assert got[col] == pytest.approx(cov / var, rel=1e-12), (
                f"{col} 的 beta 与 cov/var 不符：{got[col]} vs {cov / var} —— "
                f"除法方向或窗口被改了")

    def test_beta_is_near_the_construction_ratio(self):
        """
        构造时 HIGHB = 1.8×SPY + 噪声，LOWB = 0.4×SPY + 噪声。
        回归系数应当还原这两个倍数 —— 这条钉的是"算的确实是 beta"，
        而不是某个恰好保序的别的量。
        """
        rets, spy = _panel()
        got = E._compute_beta(rets, spy)
        assert got["HIGHB"] == pytest.approx(1.8, abs=0.15)
        assert got["LOWB"] == pytest.approx(0.4, abs=0.15)
        assert abs(got["INDEP"]) < 0.3

    def test_a_degenerate_spy_returns_all_nan_instead_of_dividing_by_zero(self):
        """
        `if spy_var < 1e-12: return pd.Series(np.nan, ...)`

        守卫被删会让 beta 变成 inf/nan 混合；`<` 翻成 `>` 会让
        **正常情况**走进这个分支，全市场 beta 归 NaN → 没有票能通过。
        """
        rets, spy = _panel()
        flat_spy = pd.Series(0.0, index=spy.index)
        out = E._compute_beta(rets, flat_spy)
        assert out.isna().all(), "SPY 方差为 0 时 beta 必须全部是 NaN"
        assert list(out.index) == list(rets.columns)

        normal = E._compute_beta(rets, spy)
        assert normal.notna().all(), (
            "正常 SPY 也走进了退化分支 —— `spy_var < 1e-12` 的方向被翻了")

    def test_a_variance_exactly_on_the_epsilon_still_computes_beta(self):
        """
        **首测存活项（L425）**：`if spy_var < 1e-12:` 翻成 `<=`
        只在方差**精确等于** 1e-12 时才有差别。

        构造：25 个交易日，前 24 天 SPY 收益率为 0、最后一天为
        `4.9999999999999996e-06`。`rolling(60, min_periods=20).var()`
        在这组输入上**逐位等于** 1e-12（下面第一条断言就是在守住这个前提，
        pandas 换算法时它会先红）。

        原始 `<` 为假 → 正常算 beta；`<=` 为真 → 整个市场 beta 归 NaN，
        beta 筛选静默变成"一只票都不通过"。
        """
        n = 25
        idx = pd.date_range("2021-01-04", periods=n, freq="B")
        spy = pd.Series(np.zeros(n), index=idx)
        spy.iloc[-1] = 4.9999999999999996e-06
        spy_var = spy.rolling(60, min_periods=20).var().iloc[-1]
        assert spy_var == 1e-12, (
            f"前提失效：构造出的方差是 {spy_var!r}，不再精确等于 1e-12，"
            f"本用例已无法区分 `<` 与 `<=`")

        rets = pd.DataFrame({"A": np.linspace(0.0, 1e-5, n)}, index=idx)
        out = E._compute_beta(rets, spy)
        assert out.notna().all(), (
            "方差恰好等于 1e-12 时 beta 被判成退化 —— "
            "`spy_var < 1e-12` 被翻成了 `<=`")

    def test_the_beta_index_covers_every_asset(self):
        rets, spy = _panel()
        assert set(E._compute_beta(rets, spy).index) == set(rets.columns)

    def test_corr_matches_pandas_rolling_corr(self):
        rets, spy = _panel()
        got = E._compute_corr(rets, spy)
        spy_al = spy.reindex(rets.index).fillna(0)
        for col in rets.columns:
            ref = rets[col].fillna(0).rolling(60, min_periods=20).corr(spy_al).iloc[-1]
            assert got[col] == pytest.approx(ref, rel=1e-12)

    def test_corr_is_bounded_and_ordered_by_construction(self):
        rets, spy = _panel()
        got = E._compute_corr(rets, spy)
        assert ((got >= -1.0) & (got <= 1.0)).all(), f"相关系数越界：{got.to_dict()}"
        assert got["HIGHB"] > got["INDEP"], "与 SPY 同向构造的标的相关性应当更高"

    def test_an_empty_panel_yields_nan_correlations_instead_of_raising(self):
        """
        **首测存活项（L445）**：`corr_series.iloc[-1] if len(corr_series) > 0 else np.nan`
        —— `>` 翻成 `>=` 会让守卫恒真（长度永远 ≥ 0），
        于是空面板上 `.iloc[-1]` 直接 IndexError。

        空面板不是假想：`DataManager` 在日期区间落在停牌/休市段时
        就会返回有列无行的 DataFrame。
        """
        empty = pd.DataFrame({"A": pd.Series(dtype=float),
                              "B": pd.Series(dtype=float)})
        out = E._compute_corr(empty, pd.Series(dtype=float))
        assert list(out.index) == ["A", "B"]
        assert out.isna().all(), (
            "空面板上的相关系数不是 NaN —— `len(corr_series) > 0` 的守卫失效了")

    def test_missing_returns_are_filled_with_zero_not_dropped(self):
        """
        `returns[col].fillna(0)` —— 去掉之后 rolling 的
        `min_periods` 计数会变，最后一格可能直接是 NaN。
        构造：在窗口中间挖 5 个 NaN，结果仍须可算。
        """
        rets, spy = _panel()
        rets.iloc[50:55, 0] = np.nan
        got = E._compute_beta(rets, spy)
        assert np.isfinite(got["HIGHB"]), "收益率里有少量缺失就算不出 beta 了"


# ===========================================================================
# H. apply() 的算术：adv20 与 returns
# ===========================================================================

def _make_data(n_days: int = 60, tickers=("A", "B", "C"), seed: int = 3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n_days, freq="B")
    close = pd.DataFrame(
        {t: 100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, n_days)) for t in tickers},
        index=idx)
    volume = pd.DataFrame(
        {t: rng.uniform(1e5, 5e6, n_days) for t in tickers}, index=idx)
    return {"close": close, "volume": volume}


class TestApplyArithmetic:

    def test_adv20_is_the_rolling_mean_of_price_times_volume(self):
        """
        `adv20 = (close * volume).rolling(20, min_periods=10).mean()`

        `*` 翻成 `/` 会让 adv20 从"日成交额"变成"价量比" ——
        数量级从 1e8 掉到 1e-4，于是 `liquidity="low"` 永远全过、
        `"high"` 永远全不过。但只断言"有票通过"是察觉不到的。
        这里直接比对参考实现。
        """
        data = _make_data()
        cfg = FilterConfig(liquidity="low")
        res = E().apply(data, cfg)

        ref = (data["close"] * data["volume"]).rolling(20, min_periods=10).mean().iloc[-1]
        got = res.filter_metrics["adv20"]
        for t in ref.index:
            assert got[t] == pytest.approx(ref[t], rel=1e-12), (
                f"{t} 的 adv20 与 mean(close*volume) 不符 —— 乘法被改了")

    def test_the_adv20_window_is_twenty_days_with_ten_minimum(self):
        """
        `rolling(20, min_periods=10)` —— 窗口或 min_periods 被改
        会让 adv20 的数值整体偏移。用只有 12 根 K 线的面板来钉
        min_periods：10 能算出来、把它改成 20 就是 NaN。
        """
        data = _make_data(n_days=12)
        res = E().apply(data, FilterConfig(liquidity="low"))
        assert res.filter_metrics["adv20"].notna().all(), (
            "只有 12 根 K 线时 adv20 全是 NaN —— min_periods 被调大了")

        ref = (data["close"] * data["volume"]).rolling(20, min_periods=10).mean().iloc[-1]
        assert res.filter_metrics["adv20"].values == pytest.approx(
            ref.values, rel=1e-12)

    def test_returns_default_to_log_returns_of_close(self):
        """
        `returns = data.get("returns", np.log(close / close.shift(1)))`

        `/` 翻成 `*` 会让"收益率"变成 log(close²) ≈ 9.2 这种量级，
        于是波动率筛选整体失真。逐位比对。
        """
        data = _make_data()
        res = E().apply(data, FilterConfig(volatility="low_vol"))
        ref = np.log(data["close"] / data["close"].shift(1)) \
                .rolling(20, min_periods=10).std().iloc[-1]
        got = res.filter_metrics["vol20"]
        for t in ref.index:
            assert got[t] == pytest.approx(ref[t], rel=1e-12), (
                f"{t} 的 vol20 与 std(log-return) 不符 —— 除法被改了")

    def test_a_supplied_returns_field_takes_precedence_over_the_default(self):
        """
        `data.get("returns", <default>)` —— 数据里带了 returns 就必须用它。
        默认值被写成无条件计算会让上游提供的（可能已复权/已清洗的）
        收益率被悄悄丢弃。
        """
        data = _make_data()
        supplied = pd.DataFrame(0.0, index=data["close"].index,
                                columns=data["close"].columns)
        supplied.iloc[:, :] = 0.05          # 恒定 5% → std 恰好为 0
        data["returns"] = supplied
        res = E().apply(data, FilterConfig(volatility="low_vol"))
        assert res.filter_metrics["vol20"].abs().max() == pytest.approx(0.0, abs=1e-15), (
            "外部提供的 returns 被忽略了 —— `data.get` 退化成了直接计算")

    def test_momentum_uses_a_sixty_day_percentage_change(self):
        """`mom60 = close.pct_change(60)` —— 周期被改会整体改变动量口径。"""
        data = _make_data(n_days=90)
        res = E().apply(data, FilterConfig(momentum_regime="strong_uptrend"))
        ref = data["close"].pct_change(60).iloc[-1]
        got = res.filter_metrics["mom60"]
        for t in ref.index:
            assert got[t] == pytest.approx(ref[t], rel=1e-12)

    def test_volume_defaults_to_nan_when_the_field_is_absent(self):
        """
        `data.get("volume", DataFrame(np.nan, ...))`

        默认值被改成 0 会让 adv20 变成 0 → 所有票都是 "low" 流动性，
        整个 universe 被静默降级。默认 NaN 则是"全部落选"，
        是可见的失败。
        """
        data = _make_data()
        del data["volume"]
        res = E().apply(data, FilterConfig(liquidity="low"))
        assert res.filter_metrics["adv20"].isna().all(), (
            "没有 volume 字段时 adv20 不是 NaN —— 默认值被改了")
        assert res.passed_tickers == [], "没有成交额数据却有票通过了流动性筛选"


# ===========================================================================
# I. apply() 的组合逻辑
# ===========================================================================

class TestApplyComposition:

    def test_an_empty_config_passes_everything_untouched(self):
        data = _make_data()
        res = E().apply(data, FilterConfig())
        assert res.passed_tickers == list(data["close"].columns)
        assert res.rejected_tickers == []
        assert res.filter_metrics == {}
        assert res.regime_label is None

    def test_two_filters_intersect_rather_than_union(self):
        """
        `passing &= passed` —— 交集。写成 `|=` 会让"高流动性**且**高波动"
        变成"高流动性**或**高波动"，universe 反而变大。
        构造三只票：A 两条都满足、B 只满足流动性、C 只满足波动率。
        交集 = {A}；并集 = {A,B,C}。
        """
        idx = pd.date_range("2022-01-03", periods=40, freq="B")
        rng = np.random.default_rng(11)

        def series(vol: float, base: float) -> np.ndarray:
            return base * np.cumprod(1 + rng.normal(0, vol, 40))

        close = pd.DataFrame({"A": series(0.08, 100), "B": series(0.001, 100),
                              "C": series(0.08, 100)}, index=idx)
        volume = pd.DataFrame({"A": np.full(40, 5e6), "B": np.full(40, 5e6),
                               "C": np.full(40, 1.0)}, index=idx)
        data = {"close": close, "volume": volume}

        liq = E().apply(data, FilterConfig(liquidity="high")).passed_tickers
        vol = E().apply(data, FilterConfig(volatility="high_vol")).passed_tickers
        both = E().apply(data, FilterConfig(liquidity="high",
                                            volatility="high_vol")).passed_tickers

        assert set(liq) == {"A", "B"}, f"流动性单跑结果不符预期：{liq}"
        assert set(vol) == {"A", "C"}, f"波动率单跑结果不符预期：{vol}"
        assert set(both) == {"A"}, (
            f"两个筛选组合后得到 {both}，应当是交集 {{'A'}} —— `&=` 被改成了并集")

    def test_passed_and_rejected_preserve_the_original_column_order(self):
        """
        `[t for t in all_tickers if t in passing]` —— 顺序来自原始列序，
        不是 set 的迭代序。写成 `list(passing)` 会让顺序随哈希漂移，
        下游按位置取权重时就会错配。
        """
        data = _make_data(tickers=("ZZZ", "AAA", "MMM"))
        res = E().apply(data, FilterConfig(liquidity="low"))
        merged = res.passed_tickers + res.rejected_tickers
        assert sorted(merged) == sorted(list(data["close"].columns))
        order = list(data["close"].columns)
        assert res.passed_tickers == [t for t in order if t in res.passed_tickers]
        assert res.rejected_tickers == [t for t in order if t in res.rejected_tickers]

    def test_passed_and_rejected_are_disjoint_and_exhaustive(self):
        data = _make_data(tickers=("A", "B", "C", "D"))
        res = E().apply(data, FilterConfig(liquidity="high"))
        assert set(res.passed_tickers) & set(res.rejected_tickers) == set()
        assert set(res.passed_tickers) | set(res.rejected_tickers) == set("ABCD")

    def test_each_active_filter_appends_exactly_one_note(self):
        data = _make_data()
        res = E().apply(data, FilterConfig(liquidity="low", volatility="low_vol",
                                           momentum_regime="strong_uptrend"))
        assert len(res.notes) == 3, f"备注条数与启用的筛选数不符：{res.notes}"
        joined = " ".join(res.notes)
        for frag in ("liquidity=low", "volatility=low_vol",
                     "momentum_regime=strong_uptrend"):
            assert frag in joined, f"备注里缺少 {frag}：{res.notes}"

    def test_the_note_reports_the_count_that_passed_that_single_filter(self):
        """
        `f"...: {len(passed)}/{len(all_tickers)} pass"` ——
        分子是**该筛选单独**通过的数量，分母是全体。
        写成 `len(passing)` 会让备注变成累计交集，与"这一条筛掉了多少"对不上。
        """
        idx = pd.date_range("2022-01-03", periods=40, freq="B")
        close = pd.DataFrame({"A": np.full(40, 100.0), "B": np.full(40, 100.0)},
                             index=idx)
        volume = pd.DataFrame({"A": np.full(40, 1e7), "B": np.full(40, 1.0)},
                              index=idx)
        res = E().apply({"close": close, "volume": volume},
                        FilterConfig(liquidity="high"))
        assert "1/2 pass" in res.notes[0], f"备注计数不对：{res.notes}"


# ===========================================================================
# J. regime 是数据集级筛选 —— 不匹配就全灭
# ===========================================================================

class TestRegimeGate:

    @staticmethod
    def _with_spy(monkeypatch, spy_returns: pd.Series):
        monkeypatch.setattr(E, "_fetch_spy_returns",
                            staticmethod(lambda index: spy_returns.reindex(index)))

    def test_a_mismatched_regime_blocks_every_ticker(self, monkeypatch):
        """
        `passing.clear()` —— regime 不匹配时**清空**。
        这行被删会让"只在牛市交易"的配置在熊市里照样满仓。
        """
        data = _make_data(n_days=260)
        spy = pd.Series(-0.004, index=data["close"].index)   # 持续下跌 → bear
        self._with_spy(monkeypatch, spy)

        res = E().apply(data, FilterConfig(regime="bull"))
        assert res.regime_label == "bear"
        assert res.passed_tickers == [], (
            "检测到 bear 但配置要求 bull，却仍有票通过 —— `passing.clear()` 没生效")
        assert set(res.rejected_tickers) == set(data["close"].columns)
        assert "BLOCKED" in res.notes[0]

    def test_a_matching_regime_lets_everything_through(self, monkeypatch):
        data = _make_data(n_days=260)
        spy = pd.Series(0.004, index=data["close"].index)    # 持续上涨 → bull
        self._with_spy(monkeypatch, spy)

        res = E().apply(data, FilterConfig(regime="bull"))
        assert res.regime_label == "bull"
        assert res.passed_tickers == list(data["close"].columns)
        assert "OK" in res.notes[0] and "BLOCKED" not in res.notes[0]

    def test_the_detected_regime_is_reported_even_when_it_matches(self, monkeypatch):
        data = _make_data(n_days=260)
        self._with_spy(monkeypatch, pd.Series(0.004, index=data["close"].index))
        res = E().apply(data, FilterConfig(regime="bull"))
        assert res.regime_label == "bull", "匹配时 regime_label 被漏填了"

    def test_the_synthetic_spy_price_path_starts_from_one_hundred(self, monkeypatch):
        """
        `spy_close = (1 + spy_returns.fillna(0)).cumprod() * 100`

        `1 +` 的 `+` 被翻成 `-` 会让累乘项变成 (1-r)，
        方向完全反过来：上涨的 SPY 被判成 bear。
        """
        data = _make_data(n_days=260)
        self._with_spy(monkeypatch, pd.Series(0.004, index=data["close"].index))
        up = E().apply(data, FilterConfig(regime="bull")).regime_label

        self._with_spy(monkeypatch, pd.Series(-0.004, index=data["close"].index))
        down = E().apply(data, FilterConfig(regime="bull")).regime_label

        assert (up, down) == ("bull", "bear"), (
            f"SPY 上涨判为 {up}、下跌判为 {down} —— "
            f"`(1 + r).cumprod()` 的符号被改了")

    def test_beta_and_corr_filters_run_only_when_spy_is_needed(self, monkeypatch):
        calls = []

        def fake(index):
            calls.append(index)
            return pd.Series(0.004, index=index)

        monkeypatch.setattr(E, "_fetch_spy_returns", staticmethod(fake))
        data = _make_data(n_days=120)

        E().apply(data, FilterConfig(liquidity="low"))
        assert calls == [], "不需要 SPY 的配置却去拉了 SPY"

        E().apply(data, FilterConfig(beta="high_beta"))
        assert len(calls) == 1, "需要 SPY 的配置没有拉 SPY"

    def test_spy_is_fetched_once_for_three_spy_dependent_filters(self, monkeypatch):
        """
        `spy_returns = self._fetch_spy_returns(...)` 在 `if config.needs_spy():`
        里只调用一次。被挪进各个子分支会让一次配置发出三次网络请求。
        """
        calls = []
        monkeypatch.setattr(E, "_fetch_spy_returns", staticmethod(
            lambda index: (calls.append(1), pd.Series(0.004, index=index))[1]))
        data = _make_data(n_days=260)
        E().apply(data, FilterConfig(regime="bull", beta="high_beta",
                                     correlation="high_corr"))
        assert len(calls) == 1, f"SPY 被重复拉取了 {len(calls)} 次"


# ===========================================================================
# K. 市值与财报窗口 —— 外部数据分支
# ===========================================================================

class TestMarketCapFetching:

    def test_prefetched_market_cap_skips_the_network_entirely(self, monkeypatch):
        """
        `if market_cap_data is None:` —— 传了就不能再去拉。
        守卫被删会在每次筛选时打一轮 yfinance，慢且可能被限流。
        """
        monkeypatch.setattr(E, "_fetch_market_cap", staticmethod(
            lambda t, r: pytest.fail("已提供市值却仍然发起了抓取")))
        data = _make_data(tickers=("BIG", "SMALL"))
        res = E().apply(data, FilterConfig(market_cap="mega_cap"),
                        market_cap_data={"BIG": 3e12, "SMALL": 1e9})
        assert res.passed_tickers == ["BIG"]

    def test_missing_market_cap_is_fetched_when_not_supplied(self, monkeypatch):
        seen = {}

        def fake(tickers, region):
            seen["tickers"], seen["region"] = list(tickers), region
            return {t: 3e12 for t in tickers}

        monkeypatch.setattr(E, "_fetch_market_cap", staticmethod(fake))
        data = _make_data(tickers=("X", "Y"))
        res = E().apply(data, FilterConfig(market_cap="mega_cap"), region="HongKong")
        assert seen["tickers"] == ["X", "Y"]
        assert seen["region"] == "HongKong", "region 没有透传给市值抓取"
        assert res.passed_tickers == ["X", "Y"]

    def test_market_cap_is_reindexed_onto_the_full_ticker_list(self, monkeypatch):
        """
        `pd.Series(market_cap_data).reindex(all_tickers)` ——
        reindex 被删会让缺失的票直接从 Series 里消失，
        于是它们**不会落选**（不在 mask 里），静默通过市值筛选。
        """
        data = _make_data(tickers=("A", "B", "C"))
        res = E().apply(data, FilterConfig(market_cap="mega_cap"),
                        market_cap_data={"A": 3e12})     # B、C 没有市值
        assert list(res.filter_metrics["market_cap"].index) == ["A", "B", "C"], (
            "市值 Series 没有对齐到全体标的 —— reindex 被删了")
        assert res.passed_tickers == ["A"]
        assert set(res.rejected_tickers) == {"B", "C"}

    def test_global_region_returns_all_nan_without_touching_yfinance(self):
        """
        `if region == "Global": return {t: np.nan for t in tickers}`
        —— 加密货币没有市值概念。这个早退被删会掉进 yfinance 分支，
        对 BTC-USD 之类的代码发起无意义的请求。
        """
        out = E._fetch_market_cap(["BTC-USD", "ETH-USD"], "Global")
        assert set(out) == {"BTC-USD", "ETH-USD"}
        assert all(np.isnan(v) for v in out.values())

    def test_a_yfinance_failure_degrades_to_nan_for_that_ticker_only(self, monkeypatch):
        """
        内层 `except` 把单只票的失败圈住 —— 它被移除会让
        一只退市股的异常掀翻整批市值抓取。
        """
        class _FI:
            def __init__(self, mc):
                self.market_cap = mc

        class _Tk:
            def __init__(self, sym):
                if sym == "BAD":
                    raise RuntimeError("delisted")
                self.fast_info = _FI(3e12)

        monkeypatch.setitem(sys.modules, "yfinance",
                            types.SimpleNamespace(Ticker=_Tk))
        out = E._fetch_market_cap(["GOOD", "BAD"], "US")
        assert out["GOOD"] == pytest.approx(3e12)
        assert np.isnan(out["BAD"]), "单票失败没有降级为 NaN"

    def test_a_zero_market_cap_becomes_nan_rather_than_zero(self, monkeypatch):
        """
        `float(getattr(fi, "market_cap", np.nan) or np.nan)`
        —— `or np.nan` 把 0 和 None 都折成 NaN。
        去掉它会让市值 0 的票被当成"真的市值为 0"→ 稳稳落进 small_cap。
        """
        class _Tk:
            def __init__(self, sym):
                self.fast_info = types.SimpleNamespace(market_cap=0)

        monkeypatch.setitem(sys.modules, "yfinance",
                            types.SimpleNamespace(Ticker=_Tk))
        out = E._fetch_market_cap(["Z"], "US")
        assert np.isnan(out["Z"]), "市值 0 没有被折成 NaN"

    def test_china_region_delegates_to_akshare(self, monkeypatch):
        import app.core.data_engine.providers.akshare_provider as ak
        monkeypatch.setattr(ak.AkshareProvider, "fetch_market_cap",
                            staticmethod(lambda tickers: {t: 5e9 for t in tickers}))
        out = E._fetch_market_cap(["600000", "000001"], "China")
        assert out == {"600000": 5e9, "000001": 5e9}

    def test_an_akshare_failure_degrades_to_all_nan(self, monkeypatch):
        import app.core.data_engine.providers.akshare_provider as ak

        def boom(tickers):
            raise RuntimeError("rate limited")

        monkeypatch.setattr(ak.AkshareProvider, "fetch_market_cap", staticmethod(boom))
        out = E._fetch_market_cap(["600000"], "China")
        assert np.isnan(out["600000"])


class TestEarningsWindow:

    @staticmethod
    def _stub_calendar(monkeypatch, mapping):
        """
        mapping: ticker → 距今天数（None 表示没有日历）。

        财报日期额外加 1 小时：被测代码里的
        `delta = (earn_date - pd.Timestamp.today()).days` 用的是
        **它自己那一刻**的时钟，与本 stub 取时刻之间存在毫秒级差。
        `Timedelta.days` 向下取整，纯整数天差会在这点抖动上翻面
        （Windows 时钟分辨率 15.6 ms，实测会偶发）。
        加 1 小时把每个 delta 顶到格子中间，抖动就不再能改变结果，
        同时 `d 天 + 1 小时` 向下取整仍精确等于 `d`（负数也成立）。
        """
        today = pd.Timestamp.today()

        class _Tk:
            def __init__(self, sym):
                self._sym = sym

            @property
            def calendar(self):
                delta = mapping.get(self._sym)
                if delta is None:
                    return None
                stamp = today + pd.Timedelta(days=delta, hours=1)
                return pd.DataFrame({"Earnings Date": [stamp]})

        monkeypatch.setitem(sys.modules, "yfinance",
                            types.SimpleNamespace(Ticker=_Tk))

    def test_pre_earnings_window_is_zero_to_five_days_inclusive(self, monkeypatch):
        """
        `if level == "pre_earnings" and 0 <= delta <= 5`

        两个比较符各自被翻会让 delta==0（今天发财报）或 delta==5
        （第五天）易主。把 -1..6 全测一遍，边界两侧都钉死。
        """
        self._stub_calendar(monkeypatch, {"D%d" % d: d for d in range(-1, 7)})
        got = E._apply_earnings([f"D{d}" for d in range(-1, 7)], "pre_earnings")
        assert got == {"D0", "D1", "D2", "D3", "D4", "D5"}, (
            f"pre_earnings 窗口是 {sorted(got)}，应当是 0..5 天 —— 边界被改了")

    def test_post_earnings_window_is_minus_five_to_minus_one(self, monkeypatch):
        """
        `elif level == "post_earnings" and -5 <= delta < 0`

        上界是**开区间**（不含 0） —— 翻成 `<=` 会让"今天发财报"
        同时算进 pre 和 post 两个窗口。
        """
        self._stub_calendar(monkeypatch, {"D%d" % d: d for d in range(-7, 2)})
        got = E._apply_earnings([f"D{d}" for d in range(-7, 2)], "post_earnings")
        assert got == {"D-5", "D-4", "D-3", "D-2", "D-1"}, (
            f"post_earnings 窗口是 {sorted(got)}，应当是 -5..-1 天")

    def test_the_two_windows_do_not_overlap(self, monkeypatch):
        self._stub_calendar(monkeypatch, {"D%d" % d: d for d in range(-7, 8)})
        names = [f"D{d}" for d in range(-7, 8)]
        pre = E._apply_earnings(names, "pre_earnings")
        post = E._apply_earnings(names, "post_earnings")
        assert pre & post == set(), f"两个财报窗口重叠了：{sorted(pre & post)}"

    def test_a_ticker_without_a_calendar_is_skipped(self, monkeypatch):
        self._stub_calendar(monkeypatch, {"HASCAL": 2, "NOCAL": None})
        assert E._apply_earnings(["HASCAL", "NOCAL"], "pre_earnings") == {"HASCAL"}

    def test_an_unknown_level_selects_nobody_rather_than_everybody(self, monkeypatch):
        """
        `_apply_earnings` 没有 else 分支 —— 未知 level 返回空集。
        它是唯一一个不抛异常的筛选，这条把这个事实固定下来，
        免得将来有人把它改成"未知就全过"。
        """
        self._stub_calendar(monkeypatch, {"A": 1, "B": -1})
        assert E._apply_earnings(["A", "B"], "during_earnings") == set()

    def test_earnings_filter_is_skipped_outside_us_and_hk(self, monkeypatch):
        """
        `if region in ("US", "HongKong")` —— 其他市场跳过并留下说明。
        条件被翻会让 A 股也去问 yfinance 要财报日历，全部取不到 → 全灭。
        """
        monkeypatch.setattr(E, "_apply_earnings", staticmethod(
            lambda t, l: pytest.fail("非美/港市场不应当调用财报日历")))
        data = _make_data(tickers=("600000", "000001"))
        res = E().apply(data, FilterConfig(earnings_window="pre_earnings"),
                        region="China")
        assert res.passed_tickers == ["600000", "000001"]
        assert "skipped" in res.notes[0]

    def test_the_earnings_filter_runs_for_us_and_hongkong(self, monkeypatch):
        seen = []

        def fake(tickers, level):
            seen.append((sorted(tickers), level))
            return {"A"}

        monkeypatch.setattr(E, "_apply_earnings", staticmethod(fake))
        for region in ("US", "HongKong"):
            data = _make_data(tickers=("A", "B"))
            res = E().apply(data, FilterConfig(earnings_window="pre_earnings"),
                            region=region)
            assert res.passed_tickers == ["A"], f"{region} 的财报筛选没有生效"
        assert seen == [(["A", "B"], "pre_earnings")] * 2, (
            f"传给财报筛选的标的或档位不对：{seen}")

    def test_the_earnings_note_counts_against_the_already_surviving_set(self, monkeypatch):
        """
        `before = len(passing)` 取在 `passing &= passed` **之前**，
        所以分母是"进入这一步时还活着的数量"，不是全体。
        两行顺序颠倒会让分母变成筛完之后的数（永远等于分子）。
        """
        idx = pd.date_range("2022-01-03", periods=40, freq="B")
        close = pd.DataFrame({t: np.full(40, 100.0) for t in "ABCD"}, index=idx)
        volume = pd.DataFrame({"A": np.full(40, 1e7), "B": np.full(40, 1e7),
                               "C": np.full(40, 1.0), "D": np.full(40, 1.0)},
                              index=idx)
        monkeypatch.setattr(E, "_apply_earnings", staticmethod(lambda t, l: {"A"}))
        res = E().apply({"close": close, "volume": volume},
                        FilterConfig(liquidity="high", earnings_window="pre_earnings"))
        assert res.passed_tickers == ["A"]
        assert "1/2 pass" in res.notes[-1], (
            f"财报备注的分母应当是进入该步时存活的 2 只，实际：{res.notes[-1]}")


# ===========================================================================
# L. SPY 抓取的降级路径
# ===========================================================================

class TestSpyFetchDegradation:

    def test_an_empty_download_returns_an_empty_series_not_a_crash(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(
            download=lambda *a, **k: pd.DataFrame()))
        idx = pd.date_range("2022-01-03", periods=30, freq="B")
        out = E._fetch_spy_returns(idx)
        assert isinstance(out, pd.Series) and out.empty

    def test_a_download_exception_degrades_to_an_empty_series(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("network down")

        monkeypatch.setitem(sys.modules, "yfinance",
                            types.SimpleNamespace(download=boom))
        out = E._fetch_spy_returns(pd.date_range("2022-01-03", periods=10, freq="B"))
        assert isinstance(out, pd.Series) and out.empty, (
            "SPY 抓取失败没有降级 —— 异常会掀翻整个筛选")

    def test_returns_are_log_returns_reindexed_onto_the_panel(self, monkeypatch):
        idx = pd.date_range("2022-01-03", periods=30, freq="B")
        prices = pd.DataFrame({"Close": 100.0 * (1.001 ** np.arange(30))}, index=idx)
        monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(
            download=lambda *a, **k: prices))
        out = E._fetch_spy_returns(idx)
        ref = np.log(prices["Close"] / prices["Close"].shift(1))
        assert list(out.index) == list(idx)
        assert out.iloc[1:].values == pytest.approx(ref.iloc[1:].values, rel=1e-12), (
            "SPY 收益率不是 log(P_t / P_{t-1}) —— 除法方向被改了")

    def test_a_multiindex_close_column_is_flattened(self, monkeypatch):
        """
        `if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]`
        —— yfinance 新版对单票也返回 MultiIndex 列。
        这一步被删会让后面的 `close.shift(1)` 得到 DataFrame，
        最终 reindex 出一个二维对象，下游 `.fillna(0)` 静默变形。
        """
        idx = pd.date_range("2022-01-03", periods=20, freq="B")
        cols = pd.MultiIndex.from_product([["Close"], ["SPY"]])
        prices = pd.DataFrame(100.0 * (1.001 ** np.arange(20)).reshape(-1, 1),
                              index=idx, columns=cols)
        monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(
            download=lambda *a, **k: prices))
        out = E._fetch_spy_returns(idx)
        assert isinstance(out, pd.Series), (
            f"MultiIndex 列没有被压平，返回了 {type(out).__name__}")

    def test_the_download_window_comes_from_the_panel_index(self, monkeypatch):
        seen = {}

        def fake_download(sym, start=None, end=None, **k):
            seen.update(symbol=sym, start=start, end=end)
            return pd.DataFrame()

        monkeypatch.setitem(sys.modules, "yfinance",
                            types.SimpleNamespace(download=fake_download))
        idx = pd.date_range("2022-01-03", periods=30, freq="B")
        E._fetch_spy_returns(idx)
        assert seen["symbol"] == "SPY"
        assert seen["start"] == str(idx[0].date()), "起始日期不是面板的第一天"
        assert seen["end"] == str(idx[-1].date()), "结束日期不是面板的最后一天"

    def test_spy_is_downloaded_adjusted_and_without_a_progress_bar(self):
        """
        **首测存活项（L460，两处）**：`auto_adjust=True, progress=False`
        两个布尔字面量翻面都活了下来 —— 之前的用例只看返回值，
        而 stub 忽略了 kwargs。

        两个参数各有实际后果：
          - `auto_adjust=False` → 拿到**未复权**的收盘价，
            分红/拆股当天会凭空出现一根巨大的"收益率"，
            regime 判定与 beta 全部受污染，且没有任何报错。
          - `progress=True` → 在无人值守的日循环日志里
            刷一堆进度条控制字符。
        """
        seen = {}

        def fake_download(sym, **k):
            seen.update(k)
            return pd.DataFrame()

        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setitem(sys.modules, "yfinance",
                                types.SimpleNamespace(download=fake_download))
            E._fetch_spy_returns(pd.date_range("2022-01-03", periods=10, freq="B"))
        finally:
            monkeypatch.undo()

        assert seen.get("auto_adjust") is True, (
            f"SPY 不是按复权价下载的（auto_adjust={seen.get('auto_adjust')!r}）—— "
            f"除权日会被当成一根真实的大涨大跌")
        assert seen.get("progress") is False, (
            f"下载进度条没有关闭（progress={seen.get('progress')!r}）")


# ===========================================================================
# M. FilterConfig / FilterResult / slice_data / 校验
# ===========================================================================

class TestFilterConfig:

    def test_is_empty_covers_all_eight_categories(self):
        """
        `is_empty` 的字段清单漏掉任何一类，都会让"只设了那一类"的配置
        被当成空配置 → **该筛选整个不生效**，而且没有任何报错。
        逐个字段单独设值验证。
        """
        assert FilterConfig().is_empty()
        for name in VALID_FILTER_VALUES:
            cfg = FilterConfig(**{name: VALID_FILTER_VALUES[name][0]})
            assert not cfg.is_empty(), (
                f"只设了 {name} 的配置被判为空 —— is_empty 的字段清单漏了它")

    def test_is_empty_enumerates_exactly_the_eight_public_categories(self):
        assert set(VALID_FILTER_VALUES) == {
            "market_cap", "liquidity", "volatility", "regime",
            "beta", "correlation", "momentum_regime", "earnings_window"}

    def test_needs_spy_is_true_for_exactly_regime_beta_and_correlation(self):
        """
        多一个字段会让不需要 SPY 的配置白跑一次下载；
        少一个会让 beta/corr 拿不到 SPY → 该筛选被静默跳过。
        """
        spy_fields = {"regime", "beta", "correlation"}
        for name in VALID_FILTER_VALUES:
            cfg = FilterConfig(**{name: VALID_FILTER_VALUES[name][0]})
            assert cfg.needs_spy() is (name in spy_fields), (
                f"needs_spy() 对 {name} 的判断错了")
        assert FilterConfig().needs_spy() is False

    def test_needs_market_cap_and_needs_earnings(self):
        assert FilterConfig(market_cap="mega_cap").needs_market_cap() is True
        assert FilterConfig(liquidity="high").needs_market_cap() is False
        assert FilterConfig(earnings_window="pre_earnings").needs_earnings() is True
        assert FilterConfig(liquidity="high").needs_earnings() is False

    def test_the_default_config_is_all_none(self):
        cfg = FilterConfig()
        for name in VALID_FILTER_VALUES:
            assert getattr(cfg, name) is None
        assert cfg.lookback_days == 60


class TestFilterResult:

    def test_n_total_is_passed_plus_rejected(self):
        """
        `return self.n_passed + len(self.rejected_tickers)`
        —— `+` 翻成 `-` 会让总数变成差；翻成 `*` 会在一边为 0 时归零。
        用不相等且都非零的两侧构造，三种算符互不相同。
        """
        r = FilterResult(passed_tickers=["A", "B", "C"],
                         rejected_tickers=["D", "E"], filter_metrics={})
        assert r.n_passed == 3
        assert r.n_total == 5, f"n_total 是 {r.n_total}，应当是 3+2=5"

    def test_n_total_of_an_all_rejected_result(self):
        r = FilterResult(passed_tickers=[], rejected_tickers=["A", "B"],
                         filter_metrics={})
        assert (r.n_passed, r.n_total) == (0, 2)

    def test_to_dict_carries_every_documented_key(self):
        r = FilterResult(passed_tickers=["A"], rejected_tickers=["B"],
                         filter_metrics={"adv20": pd.Series({"A": 1.0})},
                         regime_label="bull", notes=["n1"])
        d = r.to_dict()
        assert d == {"passed": ["A"], "rejected": ["B"], "n_passed": 1,
                     "n_total": 2, "regime": "bull", "notes": ["n1"]}

    def test_notes_default_to_a_fresh_list_per_instance(self):
        """
        `field(default_factory=list)` —— 写成 `= []` 会让所有
        FilterResult 共用同一个列表，备注跨结果串台。
        """
        a = FilterResult([], [], {})
        b = FilterResult([], [], {})
        a.notes.append("x")
        assert b.notes == [], "两个结果共用了同一个 notes 列表"


class TestSliceData:

    def test_slice_keeps_only_the_requested_columns_in_the_requested_order(self):
        data = _make_data(tickers=("A", "B", "C"))
        out = E.slice_data(data, ["C", "A"])
        for field in data:
            assert list(out[field].columns) == ["C", "A"], (
                f"{field} 的列序不是请求的顺序")

    def test_slice_tolerates_tickers_missing_from_a_field(self):
        """
        `available = [t for t in tickers if t in df.columns]`
        —— 这个过滤被删会在某个字段缺列时抛 KeyError，
        而字段之间列不齐是真实数据里常见的情况。
        """
        data = _make_data(tickers=("A", "B"))
        data["volume"] = data["volume"][["A"]]
        out = E.slice_data(data, ["A", "B"])
        assert list(out["close"].columns) == ["A", "B"]
        assert list(out["volume"].columns) == ["A"]

    def test_slice_returns_copies_not_views(self):
        """
        `df[available].copy()` —— `.copy()` 被删会让调用方对切片的
        写入回流到原始面板（或触发 SettingWithCopyWarning）。
        """
        data = _make_data(tickers=("A", "B"))
        out = E.slice_data(data, ["A"])
        out["close"].iloc[0, 0] = -999.0
        assert data["close"].iloc[0, 0] != -999.0, (
            "对切片的写入回流到了原始数据 —— `.copy()` 被删了")

    def test_slice_preserves_every_field(self):
        data = _make_data()
        data["open"] = data["close"] * 0.99
        out = E.slice_data(data, ["A"])
        assert set(out) == set(data), "切片之后字段数量变了"

    def test_slicing_to_an_empty_ticker_list_yields_empty_frames(self):
        data = _make_data()
        out = E.slice_data(data, [])
        for field, df in out.items():
            assert df.shape[1] == 0 and len(df.index) == len(data[field].index)


class TestApplyFiltersWrapper:

    def test_the_wrapper_returns_data_already_sliced_to_the_survivors(self):
        idx = pd.date_range("2022-01-03", periods=40, freq="B")
        close = pd.DataFrame({"KEEP": np.full(40, 100.0), "DROP": np.full(40, 100.0)},
                             index=idx)
        volume = pd.DataFrame({"KEEP": np.full(40, 1e7), "DROP": np.full(40, 1.0)},
                              index=idx)
        filtered, res = apply_filters({"close": close, "volume": volume},
                                      FilterConfig(liquidity="high"))
        assert res.passed_tickers == ["KEEP"]
        assert list(filtered["close"].columns) == ["KEEP"], (
            "包装函数返回的数据没有按结果切片")
        assert list(filtered["volume"].columns) == ["KEEP"]

    def test_the_wrapper_forwards_region_and_prefetched_market_cap(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(E, "_fetch_market_cap", staticmethod(
            lambda t, r: seen.setdefault("region", r) or {x: 3e12 for x in t}))
        data = _make_data(tickers=("A",))
        apply_filters(data, FilterConfig(market_cap="mega_cap"), region="HongKong")
        assert seen["region"] == "HongKong", "region 没有透传"

    def test_an_empty_config_through_the_wrapper_returns_everything(self):
        data = _make_data(tickers=("A", "B"))
        filtered, res = apply_filters(data, FilterConfig())
        assert list(filtered["close"].columns) == ["A", "B"]
        assert res.n_total == 2


class TestValidateFilterConfig:

    def test_a_valid_config_produces_no_errors(self):
        assert validate_filter_config({"liquidity": "high", "regime": "bull"}) == []

    def test_an_unknown_category_is_reported(self):
        errs = validate_filter_config({"sector": "tech"})
        assert len(errs) == 1 and "Unknown filter category 'sector'" in errs[0]

    def test_an_invalid_value_is_reported_with_the_valid_list(self):
        errs = validate_filter_config({"liquidity": "very_high"})
        assert len(errs) == 1
        assert "Invalid value 'very_high'" in errs[0]
        assert "ultra_high" in errs[0], "报错里没有列出合法取值，使用者无从修正"

    def test_every_category_gets_its_own_error(self):
        """
        `for ... : errors.append(...)` —— 循环里提前 return
        会让第二个错误被吞掉，使用者改完一个再撞一个。
        """
        errs = validate_filter_config({"liquidity": "bad", "regime": "bad",
                                       "nonsense": "x"})
        assert len(errs) == 3, f"三个问题只报了 {len(errs)} 个：{errs}"

    def test_an_empty_config_is_valid(self):
        assert validate_filter_config({}) == []

    def test_the_declared_valid_values_are_accepted_by_the_filters(self):
        """
        `VALID_FILTER_VALUES` 与各 `_apply_*` 的分支必须一一对应。
        任一边被改（多一个、少一个、拼写不同）都会让"校验通过的配置
        在运行时抛 Unknown level" —— 这是最难查的一类不一致。
        """
        checks = {
            "liquidity":       (E._apply_liquidity, _s(X=1.0)),
            "volatility":      (E._apply_volatility, _s(X=0.03)),
            "momentum_regime": (E._apply_momentum, _s(X=0.0)),
            "market_cap":      (E._apply_market_cap, _s(X=5e9)),
            "beta":            (E._apply_beta, _s(X=1.0)),
            "correlation":     (E._apply_corr, _s(X=0.5)),
        }
        for category, (fn, sample) in checks.items():
            for value in VALID_FILTER_VALUES[category]:
                fn(sample, value)      # 不抛即通过
            assert validate_filter_config({category: VALID_FILTER_VALUES[category][0]}) == []

    def test_the_regime_values_match_what_detect_regime_can_return(self):
        """
        `regime` 是唯一没有 `_apply_*` 函数的类别 ——
        它的合法值必须是 `_detect_regime` 的值域，否则
        配置 `regime="flat"` 永远不可能匹配，静默全灭。
        """
        produced = {
            E._detect_regime(_spy_series(np.full(300, 100.0))),                  # bear
            E._detect_regime(_spy_series(100.0 * (1.01 ** np.arange(300)))),     # bull
            E._detect_regime(_spy_series(np.arange(1, 100) * 1.0)),              # sideways
        }
        assert produced == set(VALID_FILTER_VALUES["regime"]), (
            f"_detect_regime 能产出 {sorted(produced)}，"
            f"但合法值声明为 {VALID_FILTER_VALUES['regime']}")


# ===========================================================================
# N. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "app/core/data_engine/dataset_filters.py ×1 — L243 `spy_close = (1 + spy_returns.fillna(0)).cumprod() * 100` 的 `*` → `/`":
        "`spy_close` 这个局部变量**只**流向 `self._detect_regime(spy_close)`，"
        "再无第二个消费者（它不进 metrics、不进 notes、不返回）。"
        "而 `_detect_regime` 对正数缩放完全不变："
        "`len(s) < 200` 与长度有关、与取值无关；`last > ma200` 两边同乘同一正数；"
        "`abs(last - ma50) / ma50` 分子分母同乘同一正数后约掉。"
        "`(1+r).cumprod()` 恒为正（收益率 r > -1），`*100` 与 `/100` "
        "相差的因子 1e4 是正数，故两者判定逐点相同。"
        "注意 `+` 与 `-` 两个算符变异**不**等价（仿射平移不是缩放），"
        "已由 test_the_synthetic_spy_price_path_starts_from_one_hundred 杀死。"
        "机械验证见 test_regime_detection_is_invariant_under_positive_scaling。",
}


def test_regime_detection_is_invariant_under_positive_scaling():
    """
    L243 等价性的机械验证：对多种形态的 SPY 路径，
    乘以任意正因子（含 `*100` 与 `/100` 之间相差的 1e4）后 regime 不变。

    这条断言一旦因为 `_detect_regime` 里引入了绝对量阈值而变红，
    上面那份等价性证明就同时作废 —— 这正是它存在的意义。
    """
    paths = {
        "bull":     100.0 * (1.01 ** np.arange(300)),
        "bear":     100.0 * (0.99 ** np.arange(300)),
        "flat":     np.full(300, 100.0),
        "short":    np.arange(1.0, 150.0),
        "sideways": np.array(list(100.0 * (1.004 ** np.arange(210)))
                             + [100.0 * 1.004 ** 209] * 50),
    }
    factors = [1e-4, 1e-2, 0.5, 1.0, 2.0, 1e2, 1e4, 1e6]
    for name, arr in paths.items():
        base = E._detect_regime(_spy_series(arr))
        for k in factors:
            scaled = E._detect_regime(_spy_series(arr * k))
            assert scaled == base, (
                f"路径 '{name}' 缩放 {k} 倍之后 regime 从 {base} 变成了 {scaled} —— "
                f"_detect_regime 不再是缩放不变的，L243 的等价性证明作废")


def test_the_synthetic_spy_price_is_consumed_only_by_regime_detection():
    """
    等价性证明的另一半前提：`spy_close` 没有第二个消费者。
    用 AST 检查 `apply` 里所有以 `spy_close` 为实参/被读取的位置，
    确认它只出现在赋值处和 `_detect_regime(...)` 调用里。
    """
    import ast
    import inspect

    src = inspect.getsource(E.apply)
    tree = ast.parse(src.lstrip() if src.startswith(" ") else src,
                     mode="exec") if not src.lstrip().startswith("def") else ast.parse(
        "\n".join(line[4:] if line.startswith("    ") else line
                  for line in src.splitlines()))

    reads = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "spy_close" \
                and isinstance(node.ctx, ast.Load):
            reads.append(node)
    assert reads, "源码里找不到对 spy_close 的读取 —— 变量被改名，证明需重做"

    call_args = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = (node.func.attr if isinstance(node.func, ast.Attribute)
                     else getattr(node.func, "id", None))
            for a in node.args:
                if isinstance(a, ast.Name) and a.id == "spy_close":
                    call_args.add(fname)
    assert call_args == {"_detect_regime"}, (
        f"spy_close 还被传给了 {sorted(call_args - {'_detect_regime'})} —— "
        f"它不再只服务于缩放不变的 regime 判定，L243 的等价性证明作废")
    assert len(reads) == len(
        [n for n in ast.walk(tree)
         if isinstance(n, ast.Call)
         and getattr(n.func, "attr", None) == "_detect_regime"]), (
        "spy_close 的读取次数多于传入 _detect_regime 的次数 —— 存在别的消费者")


def test_every_survivor_has_a_written_proof():
    """
    首测 54 点 / 存活 12：其中 11 处已由新增用例杀死，
    只剩 L243 一处为等价变异。这个数字被改动时本条会红。
    """
    assert len(PROVEN_EQUIVALENT) == 1
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
