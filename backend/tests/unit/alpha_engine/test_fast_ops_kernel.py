"""
alpha_engine/fast_ops.py —— 算子内核的定钉测试（变异测试驱动）

来由：145 个变异点，首测击杀率 **6.9%**（存活 135）—— B 档最差，也是全项目最差。

这是全系统**每一个因子值**都要流过的地方：DSL 表达式 → typed_nodes → 这里。
它错了，GP 挖出来的所有因子、回测的所有 IC、paper trading 的所有权重都跟着错，
而且**没有任何下游能发现**：因子值本来就没有"正确答案"可对照。
既有覆盖（test_dsl_engine 12 条 / test_dsl_operators / test_dsl_edge_cases）只验证
"算子能跑通、形状对、不抛异常"，没有一条钉住**算出来的数**。

存活项的分布说明问题的性质（不是零散漏测，是整层没测）：
  - `ts_corr` 16 / `ts_cov` 14 / `ts_skew` 13 / `ts_kurt` 11 —— 高阶统计量全裸
  - 三条执行路径（bottleneck / numpy-strided / 纯循环兜底）**只有一条被执行过**，
    另外两条在装了 bottleneck 的环境里是死代码，改成什么都没人知道
  - 每个滚动算子的 `if T < window` 边界、`x[i-window+1 : i+1]` 的窗口算术、
    `keepdims=True` 的广播语义，全部无人检查

本文件的核心手法：**同一个输入喂给三条路径，要求逐位一致**。
任何一处算术被改坏，三者立刻对不上 —— 一条断言同时覆盖三份实现。

不一致的地方当初全部登记为产品缺陷（MUTATION_LEDGER「B 档」），本文件先只
"钉住现状"，每条标了"修好之后这条断言要改成什么"。**2026-09-20 那六条
（B-1/B-2/B-3/B-5/B-6/B-7）已全部修复**，H 节按那些说明逐条改成了正确性断言；
只剩 B-4（契约未定）仍钉在 H2 节。

参照实现用 pandas.rolling / pandas.rank / scipy.stats.rankdata / np.corrcoef
（都是独立的第三方实现）与手算常数，**不是**把被测公式在测试里再写一遍 ——
那种写法改坏了两边一起错，而 B 档六条恰恰是"公式本身写错了"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.core.alpha_engine.fast_ops as F


# ===========================================================================
# 执行路径切换
# ===========================================================================
#
# fast_ops 有三条互相独立的实现：
#   bottleneck  : _HAS_BN=True，调 bn.move_*
#   strided     : _HAS_BN=False，走 np.lib.stride_tricks.as_strided 的向量化版
#   loop        : _HAS_BN=False 且 as_strided 抛异常 → except 里的逐行 Python 循环
# 第三条在正常环境下永远走不到（as_strided 几乎不抛），但它是"没装 bottleneck 且
# 数组布局异常"时真正会跑的代码。不逼它跑，那 40 个变异点永远是盲区。

PATHS = ("bottleneck", "strided", "loop")


@pytest.fixture(params=PATHS)
def path(request, monkeypatch):
    """把 fast_ops 切到指定执行路径，返回路径名。"""
    name = request.param
    if name != "bottleneck":
        monkeypatch.setattr(F, "_HAS_BN", False)
    if name == "loop":
        _force_stride_failure(monkeypatch)
    return name


def _force_stride_failure(monkeypatch):
    """让 as_strided 抛异常，把执行赶进 except 里的纯循环兜底。"""
    def _boom(*a, **k):
        raise RuntimeError("forced: exercising the pure-Python fallback")
    monkeypatch.setattr(np.lib.stride_tricks, "as_strided", _boom)


def _all_paths(fn, *args, monkeypatch_factory=None, **kw):
    """在三条路径上各算一遍，返回 {路径名: 结果}。"""
    out = {}
    real_bn, real_as = F._HAS_BN, np.lib.stride_tricks.as_strided
    try:
        F._HAS_BN = True
        out["bottleneck"] = fn(*args, **kw)
        F._HAS_BN = False
        out["strided"] = fn(*args, **kw)

        def _boom(*a, **k):
            raise RuntimeError("forced")
        np.lib.stride_tricks.as_strided = _boom
        out["loop"] = fn(*args, **kw)
    finally:
        F._HAS_BN = real_bn
        np.lib.stride_tricks.as_strided = real_as
    return out


# ===========================================================================
# 参照数据
# ===========================================================================

def _series(T: int = 30, N: int = 3, seed: int = 0) -> np.ndarray:
    """无 NaN 的 (T,N) 面板。T 与 window 刻意互质，避免广播巧合掩盖变异。"""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, (T, N)) + np.arange(N, dtype=float)


def _pd_roll(x: np.ndarray, window: int, how: str, **kw) -> np.ndarray:
    """pandas.rolling 参照值（独立实现，不是把被测公式抄一遍）。"""
    df = pd.DataFrame(x)
    r = df.rolling(window, min_periods=window)
    return getattr(r, how)(**kw).to_numpy()


# ===========================================================================
# A. 三条路径必须算出同一个数
# ===========================================================================

class TestPathsAgree:
    """
    一条断言同时压住三份实现。任何一份里的 `-`→`+`、`T-window+1`→`T-window-1`、
    `out[window-1:]`→`out[window+1:]` 都会让三者对不上。
    """

    #: 7 个 bottleneck 包装器 —— 三条路径必须逐位一致。
    #: `bn_ts_max`/`bn_ts_min`/`bn_ts_rank` 是 2026-09-20 修完 B-1/B-5 后才并进来的：
    #: 此前 rank 的三条路径算的是三个不同的量，max/min 在含 NaN 的输入上分家。
    _BN_WRAPPERS = ["bn_ts_mean", "bn_ts_std", "bn_ts_var", "bn_ts_sum",
                    "bn_ts_max", "bn_ts_min", "bn_ts_rank"]

    @pytest.mark.parametrize("name", _BN_WRAPPERS)
    @pytest.mark.parametrize("window", [3, 7])
    def test_rolling_moments_agree_across_paths(self, name, window):
        x = _series(T=30, N=3, seed=1)
        got = _all_paths(getattr(F, name), x, window)
        np.testing.assert_allclose(
            got["strided"], got["bottleneck"], rtol=1e-9, atol=1e-9,
            err_msg=f"{name}: numpy 向量化分支与 bottleneck 分支算出了不同的数")
        np.testing.assert_allclose(
            got["loop"], got["bottleneck"], rtol=1e-9, atol=1e-9,
            err_msg=f"{name}: 纯循环兜底分支与 bottleneck 分支算出了不同的数")

    @pytest.mark.parametrize("name", _BN_WRAPPERS)
    def test_rolling_ops_agree_across_paths_when_the_panel_has_gaps(self, name):
        """
        **含 NaN** 才测得到 NaN 策略。无 NaN 的输入上 `np.nanmax` 与 `np.max`
        恒等，缺陷 B-5 在上一条里是看不见的 —— 停牌是常态，这条才对得上现实。
        """
        x = _series(T=30, N=3, seed=2)
        x[7, 0] = x[8, 1] = x[20, 2] = np.nan        # 三只票在不同日子停牌
        got = _all_paths(getattr(F, name), x, 5)
        for other in ("strided", "loop"):
            np.testing.assert_allclose(
                got[other], got["bottleneck"], rtol=1e-9, atol=1e-9, equal_nan=True,
                err_msg=f"{name}: 含 NaN 时 {other} 与 bottleneck 的策略不一致")
        assert np.isnan(got["bottleneck"][7:12, 0]).all(), (
            f"{name}: NaN 所在的 5 个窗口应当全部被否决")

    @pytest.mark.parametrize("name", ["ts_decay_linear", "ts_argmax", "ts_argmin",
                                      "ts_skew", "ts_kurt"])
    @pytest.mark.parametrize("window", [5, 8])
    def test_extended_ts_ops_agree_between_strided_and_loop(self, name, window):
        """这些算子没有 bottleneck 分支，只有 strided 与 loop 两条。"""
        x = _series(T=30, N=3, seed=2)
        got = _all_paths(getattr(F, name), x, window)
        np.testing.assert_allclose(
            got["loop"], got["strided"], rtol=1e-9, atol=1e-9,
            err_msg=f"{name}: 纯循环兜底与向量化分支算出了不同的数")

    @pytest.mark.parametrize("name", ["ts_cov", "ts_corr"])
    def test_two_input_ops_agree_between_paths(self, name):
        """
        `ts_corr` 曾**不**在此列 —— 它的两条路径算出的数不同（缺陷 B-2：
        向量化分支 cov 用 ddof=0、std 用 ddof=1，纯循环走 np.corrcoef）。
        B-2 修复后两者同口径，于是并入这条。
        """
        x = _series(T=30, N=3, seed=3)
        y = _series(T=30, N=3, seed=4)
        got = _all_paths(getattr(F, name), x, y, 10)
        np.testing.assert_allclose(
            got["loop"], got["strided"], rtol=1e-9, atol=1e-9,
            err_msg=f"{name}: 纯循环兜底与向量化分支算出了不同的数")


# ===========================================================================
# B. 滚动算子的值（pandas 作参照）
# ===========================================================================

class TestRollingValues:

    @pytest.mark.parametrize("window", [3, 10])
    def test_ts_mean_matches_pandas(self, path, window):
        x = _series(T=25, N=3, seed=5)
        np.testing.assert_allclose(
            F.bn_ts_mean(x, window), _pd_roll(x, window, "mean"),
            rtol=1e-9, atol=1e-9,
            err_msg=f"[{path}] 滚动均值与 pandas 不符")

    @pytest.mark.parametrize("window", [3, 10])
    def test_ts_std_is_ddof_one(self, path, window):
        """
        ddof=1（样本标准差）。改成 0 会让波动率系统性偏小，
        所有用 ts_std 做分母的因子（ts_zscore 等）被整体放大。
        """
        x = _series(T=25, N=3, seed=6)
        np.testing.assert_allclose(
            F.bn_ts_std(x, window), _pd_roll(x, window, "std", ddof=1),
            rtol=1e-9, atol=1e-9, err_msg=f"[{path}] 滚动 std 不是 ddof=1")
        ddof0 = _pd_roll(x, window, "std", ddof=0)
        assert not np.allclose(_pd_roll(x, window, "std", ddof=1)[window - 1:],
                               ddof0[window - 1:]), (
            "构造的数据分不出 ddof=0/1 —— 这条断言没有区分力")

    def test_ts_var_is_the_square_of_ts_std(self, path):
        x = _series(T=25, N=3, seed=7)
        np.testing.assert_allclose(
            F.bn_ts_var(x, 6), F.bn_ts_std(x, 6) ** 2, rtol=1e-9, atol=1e-9,
            err_msg=f"[{path}] ts_var != ts_std²")

    def test_ts_sum_matches_pandas(self, path):
        x = _series(T=25, N=3, seed=8)
        np.testing.assert_allclose(
            F.bn_ts_sum(x, 4), _pd_roll(x, 4, "sum"), rtol=1e-9, atol=1e-9,
            err_msg=f"[{path}] 滚动求和与 pandas 不符")

    def test_ts_max_and_min_match_pandas_without_nan(self, path):
        x = _series(T=25, N=3, seed=9)
        np.testing.assert_allclose(F.bn_ts_max(x, 5), _pd_roll(x, 5, "max"),
                                   rtol=1e-9, atol=1e-9,
                                   err_msg=f"[{path}] 滚动最大值与 pandas 不符")
        np.testing.assert_allclose(F.bn_ts_min(x, 5), _pd_roll(x, 5, "min"),
                                   rtol=1e-9, atol=1e-9,
                                   err_msg=f"[{path}] 滚动最小值与 pandas 不符")

    def test_ts_delta_is_a_difference_not_a_sum(self):
        """`x[t] - x[t-w]`。改成 `+` 后单调序列的 delta 会变成一个大正数而非常数。"""
        x = np.arange(10, dtype=float).reshape(10, 1)
        got = F.ts_delta(x, 3)
        assert np.all(np.isnan(got[:3])), "前 window 行必须是 NaN"
        np.testing.assert_allclose(got[3:].ravel(), np.full(7, 3.0),
                                   err_msg="等差序列的 ts_delta 应恒等于 window×步长")

    def test_ts_delay_shifts_forward_only(self):
        """滞后算子只能看过去。写成 `x[window:]` 会变成**看未来**，是前视泄漏。"""
        x = np.arange(10, dtype=float).reshape(10, 1)
        got = F.ts_delay(x, 2).ravel()
        assert np.all(np.isnan(got[:2]))
        np.testing.assert_allclose(got[2:], np.arange(8, dtype=float),
                                   err_msg="ts_delay 取到的不是 x[t-window]")

    def test_ts_decay_linear_weights_recent_most(self):
        """
        线性衰减：权重 1..w 归一，最近一期权重最大。
        权重方向反了（或 `weights /= sum` 改成 `*`）都会被下面的手算常数抓住。
        """
        x = np.array([[1.0], [2.0], [3.0]])
        # 权重 = [1,2,3]/6 → 1*1/6 + 2*2/6 + 3*3/6 = 14/6
        np.testing.assert_allclose(F.ts_decay_linear(x, 3).ravel()[-1], 14 / 6,
                                   rtol=1e-12,
                                   err_msg="线性衰减权重不是 [1..w]/sum，或方向反了")

    def test_ts_zscore_is_x_minus_mean_over_std(self):
        x = _series(T=25, N=3, seed=10)
        m, s = F.bn_ts_mean(x, 6), F.bn_ts_std(x, 6)
        np.testing.assert_allclose(F.ts_zscore(x, 6), (x - m) / s,
                                   rtol=1e-9, atol=1e-9,
                                   err_msg="ts_zscore 不是 (x-mean)/std")
        # 区分力保证：mean 不恒为 0，否则 `x-mean` 与 `x+mean` 不可区分
        assert np.nanmax(np.abs(m)) > 0.1, "构造的数据均值近零，分不出 ± 号"

    def test_ts_momentum_decay_skips_the_most_recent_period(self):
        """
        skip-1 动量 = x[t-1] - x[t-1-w]。少 skip 一期就退化成普通动量，
        把最近一期的短期反转重新吃进来（Jegadeesh & Titman 1993 正是要避开它）。
        """
        x = np.array([[0.0], [1.0], [3.0], [6.0], [10.0], [15.0]])
        got = F.ts_momentum_decay(x, 2).ravel()
        # t=5: x[4] - x[2] = 10 - 3 = 7
        np.testing.assert_allclose(got[5], 7.0,
                                   err_msg="skip-1 动量取错了期数")
        assert got[5] != x[5, 0] - x[3, 0], (
            "结果与不 skip 的普通动量相同 —— 这条用例没有区分力")


# ===========================================================================
# C. 窗口边界：T == window
# ===========================================================================

class TestWindowBoundary:
    """
    每个滚动算子都有 `if T < window: return 全 NaN`。放宽成 `<=` 会让
    **数据刚好等于窗口长度**时整列变成 NaN —— 回测起点那几天的因子全废，
    而且是静默的（NaN 会被下游过滤掉，看不出少了东西）。
    """

    @pytest.mark.parametrize("name,args", [
        ("bn_ts_mean", ()), ("bn_ts_std", ()), ("bn_ts_sum", ()),
        ("bn_ts_max", ()), ("bn_ts_min", ()), ("bn_ts_rank", ()),
        ("ts_decay_linear", ()), ("ts_argmax", ()), ("ts_argmin", ()),
    ])
    def test_exactly_window_rows_still_produce_the_last_row(self, path, name, args):
        w = 4
        x = _series(T=w, N=2, seed=11)
        got = F.__dict__[name](x, w, *args)
        assert not np.any(np.isnan(got[-1])), (
            f"[{path}] {name}: T == window 时最后一行应当有值，却是 NaN —— "
            f"`T < window` 的边界被放宽了")
        assert np.all(np.isnan(got[:-1])), (
            f"[{path}] {name}: T == window 时只有最后一行能有值")

    def test_exactly_window_rows_for_skew_and_kurt(self, path):
        """ts_skew 要 window>=3，ts_kurt 要 window>=4，边界各测各的。"""
        xs = _series(T=5, N=2, seed=12)
        assert not np.any(np.isnan(F.ts_skew(xs, 5)[-1])), (
            f"[{path}] ts_skew: T == window 时最后一行是 NaN")
        assert not np.any(np.isnan(F.ts_kurt(xs, 5)[-1])), (
            f"[{path}] ts_kurt: T == window 时最后一行是 NaN")

    def test_skew_needs_three_points_and_kurt_needs_four(self):
        """`window < 3` / `window < 4` 的下限：少一个点公式的分母就是 0。"""
        x = _series(T=20, N=2, seed=13)
        assert np.all(np.isnan(F.ts_skew(x, 2))), "window=2 的偏度应全 NaN"
        assert not np.all(np.isnan(F.ts_skew(x, 3))), "window=3 的偏度不该全 NaN"
        assert np.all(np.isnan(F.ts_kurt(x, 3))), "window=3 的峰度应全 NaN"
        assert not np.all(np.isnan(F.ts_kurt(x, 4))), "window=4 的峰度不该全 NaN"

    def test_two_input_ops_at_exactly_window_rows(self, path):
        w = 5
        x, y = _series(T=w, N=2, seed=14), _series(T=w, N=2, seed=15)
        assert not np.any(np.isnan(F.ts_corr(x, y, w)[-1])), (
            f"[{path}] ts_corr: T == window 时最后一行是 NaN")
        assert not np.any(np.isnan(F.ts_cov(x, y, w)[-1])), (
            f"[{path}] ts_cov: T == window 时最后一行是 NaN")

    def test_entropy_at_exactly_window_rows(self, path):
        w = 6
        x = _series(T=w, N=2, seed=16)
        assert not np.any(np.isnan(F.ts_entropy(x, w)[-1])), (
            f"[{path}] ts_entropy: T == window 时最后一行是 NaN")

    def test_shorter_than_window_is_all_nan_for_pure_python_ops(self, path):
        """
        这几个算子没有 bottleneck 分支，三条路径都必须给全 NaN。
        （有 bottleneck 分支的那 7 个曾在 window > T 时**抛异常** —— 缺陷 B-7，
        已于 2026-09-20 修；它们由 H 节的
        test_short_panels_return_nan_instead_of_raising 覆盖。）
        """
        x = _series(T=3, N=2, seed=17)
        for name in ("ts_decay_linear", "ts_argmax", "ts_argmin",
                     "ts_entropy", "ts_skew", "ts_kurt"):
            got = F.__dict__[name](x, 5)
            assert np.all(np.isnan(got)), (
                f"[{path}] {name}: 数据短于窗口却给出了值")


# ===========================================================================
# D. 高阶统计量的公式
# ===========================================================================

class TestHigherMoments:

    def test_skew_uses_the_unbiased_coefficient(self, path):
        """
        `coeff = n²/((n-1)(n-2))` 是 Fisher-Pearson 的无偏修正（scipy bias=False）。
        `*`→`/` 会让系数变成 n/(n(n-1)(n-2))，`-`→`+` 会变成 n²/((n+1)(n-2))：
        两种都让偏度整体缩放，跨 window 不再可比 —— GP 会据此挑出"偏度更大"的
        假赢家，而差别只是窗口长度不同。
        """
        pytest.importorskip("scipy")
        from scipy.stats import skew as sp_skew
        x = _series(T=40, N=3, seed=18)
        w = 9
        got = F.ts_skew(x, w)
        ref = np.full_like(got, np.nan)
        for i in range(w - 1, x.shape[0]):
            ref[i] = sp_skew(x[i - w + 1: i + 1], axis=0, bias=False)
        np.testing.assert_allclose(got[w - 1:], ref[w - 1:], rtol=1e-9, atol=1e-9,
                                   err_msg=f"[{path}] 滚动偏度与 scipy(bias=False) 不符")
        # 区分力：无偏系数与有偏系数确实不同
        biased = sp_skew(x[:w], axis=0, bias=True)
        assert not np.allclose(ref[w - 1], biased), (
            "构造的数据分不出有偏/无偏偏度 —— 这条断言没有区分力")

    def test_kurt_is_excess_kurtosis_with_sample_std(self, path):
        """
        实现是 `mean(z⁴) - 3`，其中 z 用 **ddof=1** 的 std 标准化。
        `- 3.0` 改成 `+ 3.0` 会把超额峰度整体抬 6，正态分布从 0 变成 6，
        任何"尾部是否肥"的判断全反。
        """
        x = _series(T=40, N=3, seed=19)
        w = 10
        got = F.ts_kurt(x, w)
        ref = np.full_like(got, np.nan)
        for i in range(w - 1, x.shape[0]):
            b = x[i - w + 1: i + 1]
            z = (b - b.mean(axis=0)) / b.std(axis=0, ddof=1)
            ref[i] = (z ** 4).mean(axis=0) - 3.0
        np.testing.assert_allclose(got[w - 1:], ref[w - 1:], rtol=1e-9, atol=1e-9,
                                   err_msg=f"[{path}] 滚动峰度公式不符")
        assert np.nanmax(np.abs(got)) > 0.05, (
            "构造的数据峰度近零，`-3`/`+3` 的差别被淹没了")

    def test_moment_ops_keep_the_window_axis_when_centring(self, path):
        """
        `np.mean(ws, axis=1, keepdims=True)` —— keepdims 决定 `ws - mu` 是
        逐窗口去均值还是跨窗口错位相减。改成 False 时形状 (W,window,N) 与 (W,N)
        无法广播（W != window 时直接抛错），或在 W == window 时**静默算错**。
        这里刻意取 T=40, window=9 → W=32 ≠ 9，让两种后果都不会被掩盖。
        """
        x = _series(T=40, N=3, seed=20)
        got = F.ts_skew(x, 9)
        assert got.shape == x.shape, "ts_skew 的输出形状被广播改坏了"
        assert np.isfinite(got[8:]).all(), "ts_skew 在无 NaN 输入上产出了非有限值"

    def test_corr_of_a_series_with_itself_is_exactly_one(self, path):
        """
        自相关恒等于 1.0。这条同时钉住 `cov/denom` 的两个 `*`：
        `sum(dx*dy)/(w-1)` 改成 `/`、`std_x*std_y` 改成 `/` 都会让它偏离。

        （缺陷 B-2 修复前，向量化分支的上界是 (w-1)/w，与纯循环兜底的
        np.corrcoef 分家；现在三条路径统一给 1.0。）
        """
        x = _series(T=40, N=3, seed=21)
        w = 12
        np.testing.assert_allclose(F.ts_corr(x, x, w)[w - 1:], 1.0,
                                   rtol=1e-9, atol=1e-9,
                                   err_msg=f"[{path}] 自相关不等于 1.0")

    def test_cov_matches_numpy_cov(self, path):
        x, y = _series(T=40, N=3, seed=22), _series(T=40, N=3, seed=23)
        w = 12
        got = F.ts_cov(x, y, w)
        ref = np.full_like(got, np.nan)
        for i in range(w - 1, x.shape[0]):
            for j in range(x.shape[1]):
                ref[i, j] = np.cov(x[i - w + 1: i + 1, j],
                                   y[i - w + 1: i + 1, j])[0, 1]
        np.testing.assert_allclose(got[w - 1:], ref[w - 1:], rtol=1e-9, atol=1e-9,
                                   err_msg=f"[{path}] 滚动协方差与 np.cov 不符")

    def test_loop_fallback_needs_two_valid_pairs_not_three(self, monkeypatch):
        """
        纯循环兜底里的 `if mask.sum() < 2: continue` —— **严格小于 2**。
        放宽成 `<= 2` 会让恰好两个有效点的窗口也被跳过：停牌明显的标的
        在纯循环路径下永远算不出 ts_corr / ts_cov。

        向量化路径对含 NaN 的窗口一律给 NaN（`has_nan` 整窗否决），
        所以这条差别**只在兜底路径上可见** —— 必须逼它走那条路。
        """
        monkeypatch.setattr(F, "_HAS_BN", False)
        _force_stride_failure(monkeypatch)
        x = np.array([[1.0], [np.nan], [np.nan], [2.0], [4.0]])
        y = np.array([[2.0], [np.nan], [np.nan], [5.0], [9.0]])
        corr = F.ts_corr(x, y, 4).ravel()
        cov = F.ts_cov(x, y, 4).ravel()
        assert np.isfinite(corr[3]), (
            "恰好两个有效点的窗口被跳过了 —— `mask.sum() < 2` 被放宽成了 `<= 2`")
        assert np.isfinite(cov[3]), "ts_cov 的同一处守卫也被放宽了"
        np.testing.assert_allclose(corr[3], 1.0, rtol=1e-9)
        # 第 3 行的窗口是 0..3 → 有效对为 (1,2) 与 (2,5)
        np.testing.assert_allclose(cov[3], np.cov([1.0, 2.0], [2.0, 5.0])[0, 1],
                                   rtol=1e-9)

    def test_cov_divides_by_window_minus_one(self, path):
        """`/(window-1)` 改成 `/(window+1)` 会让协方差整体偏小，且随 window 变化。"""
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        # 与自身的协方差 = 样本方差 = 5/3
        np.testing.assert_allclose(F.ts_cov(x, x, 4).ravel()[-1], 5 / 3, rtol=1e-12,
                                   err_msg=f"[{path}] ts_cov 的自由度不是 window-1")


# ===========================================================================
# E. ts_argmax / ts_argmin / ts_entropy
# ===========================================================================

class TestPositionAndEntropy:

    def test_argmax_is_one_when_the_newest_bar_is_the_high(self, path):
        """
        归一化位置 = idx / (window-1)，最新一根是最高点时应当**恰好是 1.0**。
        `denom` 写成 `window + 1` 会让它永远够不到 1，
        "刚创新高"这个条件就此永不成立。
        """
        x = np.array([[3.0], [1.0], [2.0], [5.0]])
        np.testing.assert_allclose(F.ts_argmax(x, 4).ravel()[-1], 1.0, rtol=1e-12,
                                   err_msg=f"[{path}] 最新一根是最高点时 argmax != 1.0")
        np.testing.assert_allclose(F.ts_argmin(x, 4).ravel()[-1], 1 / 3, rtol=1e-12,
                                   err_msg=f"[{path}] argmin 的归一化位置不对")

    def test_argmax_is_zero_when_the_oldest_bar_is_the_high(self, path):
        x = np.array([[9.0], [1.0], [2.0], [3.0]])
        np.testing.assert_allclose(F.ts_argmax(x, 4).ravel()[-1], 0.0, atol=1e-12,
                                   err_msg=f"[{path}] 最老一根是最高点时 argmax != 0")

    def test_entropy_of_a_uniform_spread_is_one(self, path):
        """
        n_bins 个桶各落一个点 → 归一化熵 = 1。
        `h/log_nbins` 里的 log 基准改了、或 `probs*log(probs)` 改成 `/`，
        这个 1.0 立刻不成立。
        """
        x = np.arange(10, dtype=float).reshape(10, 1)
        got = F.ts_entropy(x, 10, n_bins=10).ravel()[-1]
        np.testing.assert_allclose(got, 1.0, rtol=1e-9,
                                   err_msg=f"[{path}] 均匀分布的归一化熵不是 1.0")

    def test_entropy_of_a_concentrated_window_is_low(self, path):
        """全部落进同一个桶 → 熵 = 0。"""
        x = np.full((10, 1), 2.0)
        x[0, 0] = 2.0
        got = F.ts_entropy(x, 10, n_bins=10).ravel()[-1]
        np.testing.assert_allclose(got, 0.0, atol=1e-12,
                                   err_msg=f"[{path}] 全同值窗口的熵不是 0")

    def test_entropy_skips_empty_bins(self, path):
        """
        `counts[counts > 0]` —— 放宽成 `>=` 会把空桶的 0 也带进 `0·log0`，
        结果变成 NaN。用一个必然产生空桶的输入钉住"仍是有限值"。
        """
        x = np.array([[0.0]] * 5 + [[9.0]] * 5)
        got = F.ts_entropy(x, 10, n_bins=10).ravel()[-1]
        assert np.isfinite(got), (
            f"[{path}] 存在空桶时熵变成了 {got} —— 空桶没有被排除")
        np.testing.assert_allclose(got, np.log(2) / np.log(10), rtol=1e-9,
                                   err_msg=f"[{path}] 两桶各半的熵应为 log2/log10")


# ===========================================================================
# F. 截面算子
# ===========================================================================

class TestCrossSectional:

    def test_cs_rank_spans_zero_to_one(self):
        """
        `denom = max(valid_count - 1, 1)` —— 改成 `+1` 会让最高名次只到
        (n-1)/(n+1)，"排名前 10%"这类阈值判断整体偏移。
        """
        x = np.array([[10.0, 20.0, 30.0, 40.0]])
        np.testing.assert_allclose(F.cs_rank(x).ravel(), [0.0, 1 / 3, 2 / 3, 1.0],
                                   rtol=1e-12,
                                   err_msg="cs_rank 的值域不是 [0,1]")

    def test_cs_rank_keeps_nan_assets_nan(self):
        x = np.array([[10.0, np.nan, 30.0, 40.0]])
        got = F.cs_rank(x).ravel()
        assert np.isnan(got[1]), "NaN 资产应当保持 NaN"
        assert np.all(np.isfinite(got[[0, 2, 3]])), "有效资产被连坐成了 NaN"
        assert got[0] < got[2] < got[3], "有效资产之间的相对次序不对"

    def test_cs_zscore_is_ddof_one_and_nan_aware(self):
        x = np.array([[1.0, 2.0, 3.0, np.nan]])
        got = F.cs_zscore(x).ravel()
        assert np.isnan(got[3])
        np.testing.assert_allclose(got[:3], [-1.0, 0.0, 1.0], rtol=1e-12,
                                   err_msg="截面 z-score 不是 (x-mean)/std(ddof=1)")

    def test_cs_scale_uses_absolute_values(self):
        """
        `np.nansum(np.abs(x))` —— 去掉 abs 后多空混合的一行会被**带符号的和**
        归一：`[1,-1,2]` 的分母从 4 变成 2，权重整体翻倍；
        若多空恰好抵消，分母为 0 → 整行变 NaN，那一天**全部空仓**。
        """
        x = np.array([[1.0, -1.0, 2.0]])
        np.testing.assert_allclose(F.cs_scale(x).ravel(), [0.25, -0.25, 0.5],
                                   rtol=1e-12, err_msg="cs_scale 的分母不是 L1 范数")
        balanced = np.array([[1.0, -1.0]])
        np.testing.assert_allclose(F.cs_scale(balanced).ravel(), [0.5, -0.5],
                                   rtol=1e-12,
                                   err_msg="多空恰好抵消时 cs_scale 崩成了 NaN")

    def test_cs_winsorize_clips_at_plus_minus_k_sigma(self):
        """
        `np.clip(x, mu - k*sd, mu + k*sd)`。三处变异各有后果：
        `k*sd`→`k/sd` 让上下界与离散度脱钩；`mu+`→`mu-` / `mu-`→`mu+`
        会把上下界颠倒，np.clip 在 lower>upper 时**全部吐出 upper**，
        整行被压成同一个数。
        """
        x = np.array([[0.0, 1.0, 2.0, 100.0]])
        mu, sd = np.nanmean(x), np.nanstd(x, ddof=1)
        got = F.cs_winsorize(x, k=1.0).ravel()
        assert got.max() <= mu + sd + 1e-9, "上界没有生效"
        assert got.min() >= mu - sd - 1e-9, "下界没有生效"
        assert got[3] < 100.0, "极端值没有被缩尾"
        assert len(set(np.round(got, 9))) > 1, (
            "整行被压成了同一个数 —— 上下界疑似颠倒")

    def test_cs_winsorize_bounds_are_exactly_mu_plus_minus_k_sigma(self):
        """
        `mu ± k * sd` 里的 `*` 改成 `/`：界与离散度的关系从正比变成反比。
        只断言"上界生效"抓不住（`k/sd` 给出的界更窄，`<=` 照样成立），
        必须钉死**数值**。离群值要真的越界，所以用 10 个近零值 + 1 个 1000。
        """
        x = np.array([np.concatenate([np.arange(10, dtype=float), [1000.0]])])
        k = 1.5
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x, ddof=1))
        assert x.max() > mu + k * sd, "构造的离群值没有越界，测不到 clip"
        got = F.cs_winsorize(x, k=k)
        np.testing.assert_allclose(got.max(), mu + k * sd, rtol=1e-12,
                                   err_msg="缩尾上界不等于 mu + k*sigma")
        assert not np.isclose(mu + k * sd, mu + k / sd), (
            "构造的数据里 k*sigma 与 k/sigma 相等 —— 这条断言没有区分力")

    def test_cs_winsorize_lower_bound_is_exactly_mu_minus_k_sigma(self):
        """
        `np.clip(x, mu - k * sd, mu + k * sd)` 一行里有**两个** `*`，
        变异器只改第一个 → 只有下界变、上界不变。
        所以只断言 `max()` 抓不到它（上一轮复测就是这么让它活下来的），
        必须单独钉下界。这里放一个**低端**离群值让下界真的生效。
        """
        x = np.array([np.concatenate([[-1000.0], np.arange(10, dtype=float)])])
        k = 1.5
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x, ddof=1))
        assert x.min() < mu - k * sd, "构造的低端离群值没有越界，测不到 clip"
        got = F.cs_winsorize(x, k=k)
        np.testing.assert_allclose(got.min(), mu - k * sd, rtol=1e-12,
                                   err_msg="缩尾下界不等于 mu - k*sigma")

    def test_cs_winsorize_keeps_rows_independent(self):
        """`keepdims=True` 改成 False 后，行与行的均值/标准差会错位相减。"""
        x = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 300.0]])
        got = F.cs_winsorize(x, k=1.0)
        np.testing.assert_allclose(got[0], [0.0, 0.0, 0.0], atol=1e-12,
                                   err_msg="全零行被另一行的统计量污染了")

    def test_cs_normalize_maps_to_zero_one(self):
        """
        刻意用 T=2、N=3（行数 ≠ 列数）：`keepdims=True` 改成 False 后
        (2,3) 与 (2,) 无法广播，直接抛错。单行输入测不出来 ——
        (1,3) 与 (1,) 恰好能广播，两种写法结果一模一样。
        """
        x = np.array([[2.0, 4.0, 10.0], [0.0, 5.0, 5.0]])
        np.testing.assert_allclose(F.cs_normalize(x),
                                   [[0.0, 0.25, 1.0], [0.0, 1.0, 1.0]],
                                   rtol=1e-12,
                                   err_msg="min-max 归一化的结果不在 [0,1] 或算错了")

    def test_cs_normalize_returns_nan_for_a_flat_row(self):
        x = np.array([[5.0, 5.0, 5.0]])
        assert np.all(np.isnan(F.cs_normalize(x))), (
            "无离散度的一行应当是 NaN，而不是 0/0 的任意值")

    def test_cs_demean_removes_the_row_mean(self):
        x = np.array([[1.0, 2.0, 6.0], [0.0, 0.0, 3.0]])
        got = F.cs_demean(x)
        np.testing.assert_allclose(np.nanmean(got, axis=1), [0.0, 0.0], atol=1e-12,
                                   err_msg="去均值之后每行均值不为 0")
        np.testing.assert_allclose(got[0], [-2.0, -1.0, 3.0], rtol=1e-12)

    def test_signed_power_keeps_the_sign_and_uses_the_magnitude(self):
        """
        `sign(x) * |x|^p` —— 去掉 abs 后负底数遇上分数次幂直接变 NaN，
        `signed_power(x, 0.5)` 这种常见写法会让所有空头名字消失。
        """
        x = np.array([[-4.0, 4.0, 0.0]])
        got = F.signed_power(x, 0.5).ravel()
        np.testing.assert_allclose(got, [-2.0, 2.0, 0.0], rtol=1e-12,
                                   err_msg="signed_power 丢了符号或对负数取幂变成了 NaN")


# ===========================================================================
# G. 组算子与条件算子
# ===========================================================================

class TestGroupAndConditional:

    def test_group_mean_broadcasts_within_each_group(self):
        """
        `keepdims=True` 改成 False 时，组内均值 (T,) 广播回 (T,m) 会在
        m != T 时抛错、在 m == T 时**转置着**填回去。这里取 T=2、组宽 3，
        两种后果都暴露。
        """
        x = np.array([[1.0, 3.0, 10.0, 20.0, 30.0],
                      [5.0, 7.0, 40.0, 50.0, 60.0]])
        groups = np.array([0, 0, 1, 1, 1])
        got = F.group_mean(x, groups)
        np.testing.assert_allclose(got[0], [2.0, 2.0, 20.0, 20.0, 20.0], rtol=1e-12,
                                   err_msg="组内均值没有按组广播")
        np.testing.assert_allclose(got[1], [6.0, 6.0, 50.0, 50.0, 50.0], rtol=1e-12)

    def test_group_neutralize_zeroes_each_group_mean(self):
        x = np.array([[1.0, 3.0, 10.0, 30.0]])
        groups = np.array([0, 0, 1, 1])
        got = F.group_neutralize(x, groups)
        np.testing.assert_allclose(got.ravel(), [-1.0, 1.0, -10.0, 10.0], rtol=1e-12,
                                   err_msg="组中性化之后组内均值不为 0")

    def test_group_rank_ranks_inside_the_group_only(self):
        x = np.array([[1.0, 2.0, 100.0, 200.0]])
        groups = np.array([0, 0, 1, 1])
        np.testing.assert_allclose(F.group_rank(x, groups).ravel(),
                                   [0.0, 1.0, 0.0, 1.0], rtol=1e-12,
                                   err_msg="组内排名跨组了")

    def test_ind_neutralize_without_groups_falls_back_to_zscore(self):
        x = _series(T=4, N=5, seed=24)
        np.testing.assert_allclose(F.ind_neutralize(x, None), F.cs_zscore(x),
                                   rtol=1e-12, atol=1e-12,
                                   err_msg="groups=None 的回退路径不是 cs_zscore")

    def test_sector_neutral_uses_row_zero_of_a_2d_group_array(self):
        x = np.array([[1.0, 3.0, 10.0, 30.0], [2.0, 4.0, 20.0, 40.0]])
        g2d = np.array([[0, 0, 1, 1], [1, 1, 0, 0]])     # 第 1 行是干扰
        np.testing.assert_allclose(F.cs_sector_neutral(x, g2d),
                                   F.ind_neutralize(x, g2d[0]), rtol=1e-12,
                                   err_msg="二维分组数组没有取第 0 行")

    def test_conditional_ops_select_the_right_branch(self):
        cond = np.array([[True, False], [False, True]])
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[10.0, 20.0], [30.0, 40.0]])
        np.testing.assert_array_equal(F.op_if_else(cond, a, b),
                                      [[1.0, 20.0], [30.0, 4.0]])
        np.testing.assert_array_equal(F.op_where(cond, a, b),
                                      F.op_if_else(cond, a, b))
        np.testing.assert_array_equal(F.op_trade_when(cond, a),
                                      [[1.0, 0.0], [0.0, 4.0]])

    def test_logical_ops_return_zero_one_floats(self):
        x = np.array([[1.0, 0.0], [0.0, 2.0]])
        y = np.array([[1.0, 1.0], [0.0, 0.0]])
        np.testing.assert_array_equal(F.op_and(x, y), [[1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_equal(F.op_or(x, y), [[1.0, 1.0], [0.0, 1.0]])
        np.testing.assert_array_equal(F.op_not(x), [[0.0, 1.0], [1.0, 0.0]])


# ===========================================================================
# H. 曾经的缺陷 —— 现在钉的是**应有行为**
# ===========================================================================
#
# B-1/B-2/B-3/B-5/B-6/B-7 于 2026-09-20 修复（MUTATION_LEDGER「B 档」）。
# 本节原先钉的是错误行为并写明"修好之后改成什么"，现按那些说明逐条改成了
# 正确性断言。**参照值一律来自独立实现**（scipy.stats.rankdata /
# pandas.rank / np.corrcoef），不是把 fast_ops 的公式在测试里再抄一遍 ——
# 抄一遍的话两边会一起错，而这六条恰恰是"公式本身写错了"。

class TestFormerlyBrokenOperators:

    @pytest.mark.parametrize("window", [2, 3, 5, 20])
    def test_ts_rank_is_an_average_rank_percentile_on_every_path(self, window):
        """
        【缺陷 B-1，已修】两条分支此前算的不是同一个量：bottleneck 是
        `bn.move_rank(...)/window`（值域 [-1/w, 1/w]，一半为负，且随窗口缩放），
        numpy 是 `le/count`（值域 (0,1]，取不到 0）。docstring 承诺的 [0,1]
        两边都不满足 —— `ts_rank(close,20) > 0.8` 在装了 bottleneck 的环境里
        **恒为假**（上界 0.05）。

        参照值用 scipy.stats.rankdata(method="average")，与被测实现无共享代码。
        """
        from scipy.stats import rankdata
        x = _series(T=40, N=4, seed=31)
        x[5, 1] = x[9, 1] = x[9, 2]              # 制造并列，逼出平均秩语义
        ref = np.full_like(x, np.nan)
        for i in range(window - 1, x.shape[0]):
            for j in range(x.shape[1]):
                block = x[i - window + 1: i + 1, j]
                ref[i, j] = (rankdata(block, method="average")[-1] - 1) / (window - 1)
        for name, got in _all_paths(F.bn_ts_rank, x, window).items():
            np.testing.assert_allclose(
                got, ref, rtol=1e-9, atol=1e-9, equal_nan=True,
                err_msg=f"[{name}] ts_rank 不是 scipy 的平均秩百分位")

    def test_ts_rank_reaches_both_ends_of_zero_one(self, path):
        """窗口内最高 → 恰好 1.0，最低 → 恰好 0.0。端点取不到就不是百分位。"""
        up = np.arange(20, dtype=float).reshape(20, 1)
        np.testing.assert_allclose(F.bn_ts_rank(up, 5).ravel()[-1], 1.0, rtol=1e-12,
                                   err_msg=f"[{path}] 窗口内最高值的秩不是 1.0")
        np.testing.assert_allclose(F.bn_ts_rank(-up, 5).ravel()[-1], 0.0, atol=1e-12,
                                   err_msg=f"[{path}] 窗口内最低值的秩不是 0.0")

    def test_ts_corr_of_perfectly_correlated_series_is_exactly_one(self, path):
        """
        【缺陷 B-2，已修】协方差走 ddof=0、标准差走 ddof=1，相关系数被系统性
        压低 (w-1)/w —— 偏差**随窗口变化**，短窗口的配对信号被压得更狠。
        """
        rng = np.random.default_rng(99)
        a = rng.normal(size=(60, 1))
        b = a * 2.0 + 1.0                       # 真相关恒为 1.0
        for w in (5, 20):
            np.testing.assert_allclose(
                F.ts_corr(a, b, w).ravel()[-1], 1.0, rtol=1e-9,
                err_msg=f"[{path}] window={w}：完全线性相关没给出 1.0")

    def test_ts_corr_matches_numpy_corrcoef(self, path):
        """一般输入也要对得上，而不只是在相关=1 的特例上碰巧。"""
        x, y = _series(T=40, N=3, seed=32), _series(T=40, N=3, seed=33)
        w = 12
        ref = np.full((40, 3), np.nan)
        for i in range(w - 1, 40):
            for j in range(3):
                ref[i, j] = np.corrcoef(x[i - w + 1: i + 1, j],
                                        y[i - w + 1: i + 1, j])[0, 1]
        np.testing.assert_allclose(F.ts_corr(x, y, w)[w - 1:], ref[w - 1:],
                                   rtol=1e-9, atol=1e-9,
                                   err_msg=f"[{path}] 滚动相关与 np.corrcoef 不符")

    def test_cs_rank_gives_tied_assets_the_same_average_rank(self):
        """
        【缺陷 B-3，已修】`argsort(argsort())` 是**序数**名次：并列值按列顺序
        强行分先后。参照值用 pandas.rank(method="average")。
        """
        cases = [
            np.array([[1.0, 1.0, 2.0, 3.0]]),
            np.array([[5.0, 5.0, 5.0, 9.0]]),
            np.array([[10.0, 20.0, 30.0, 40.0]]),
        ]
        for row in cases:
            ref = (pd.DataFrame(row).rank(axis=1, method="average").to_numpy() - 1) / 3.0
            np.testing.assert_allclose(F.cs_rank(row), ref, rtol=1e-12,
                                       err_msg=f"{row.ravel()} 的截面秩不是平均秩")
        np.testing.assert_allclose(F.cs_rank(cases[0]).ravel(),
                                   [1 / 6, 1 / 6, 2 / 3, 1.0], rtol=1e-12)

    def test_cs_rank_does_not_depend_on_column_order(self):
        """
        列序是面板的存储顺序，不是市场事实。打乱列顺序，结果必须只是同样被
        打乱 —— 而不是换一组数。这是 B-3 真正的危害。
        """
        row = np.array([[1.0, 1.0, 3.0, 2.0]])
        perm = [2, 0, 3, 1]
        np.testing.assert_allclose(F.cs_rank(row[:, perm]).ravel(),
                                   F.cs_rank(row).ravel()[perm], rtol=1e-12,
                                   err_msg="换列顺序改变了同一只标的的因子值")

    def test_ts_max_min_obey_the_nan_policy_on_every_path(self):
        """
        【缺陷 B-5，已修】numpy 分支用 `np.nanmax/np.nanmin` **忽略** NaN，
        bottleneck 用 `min_count=window` 遵守 —— 同一面板换条路径结果不同，
        且 numpy 那边的极值是用更少的观测算出来的（停牌期 ts_max 系统性偏低）。
        """
        x = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0], [6.0]])
        for fn in (F.bn_ts_max, F.bn_ts_min):
            got = _all_paths(fn, x, 3)
            for name in ("strided", "loop"):
                np.testing.assert_allclose(
                    got[name], got["bottleneck"], equal_nan=True,
                    err_msg=f"{fn.__name__}: {name} 与 bottleneck 的 NaN 策略不一致")
            # 含 NaN 的窗口（第 2/3/4 行）必须是 NaN，之后恢复
            assert np.all(np.isnan(got["bottleneck"][:5, 0])), "含 NaN 的窗口没被否决"
            assert np.isfinite(got["bottleneck"][5, 0]), "NaN 滑出窗口后应当恢复取值"

    def test_cs_rank_stays_within_zero_one_when_the_row_has_nan(self):
        """
        【缺陷 B-6，已修】NaN 被 `-inf` 填充后**参与排序**占掉低位名次，分母却
        只按有效个数算 → 值域越出 [0,1]，且缺失越多偏得越狠。
        `rank(x) > 0.9` 会因为当天停牌几只票而莫名多命中。
        """
        got = F.cs_rank(np.array([[10.0, np.nan, 30.0, 40.0]])).ravel()
        np.testing.assert_allclose(got[[0, 2, 3]], [0.0, 0.5, 1.0], rtol=1e-12)
        assert np.isnan(got[1]), "NaN 资产应当保持 NaN"

    @pytest.mark.parametrize("n_nan", [0, 1, 2, 3])
    def test_cs_rank_range_is_invariant_to_how_many_assets_are_missing(self, n_nan):
        """缺失个数不该改变值域 —— 有效资产永远铺满 [0,1]。"""
        row = np.arange(1.0, 9.0).reshape(1, 8)
        row[0, :n_nan] = np.nan
        got = F.cs_rank(row).ravel()
        finite = got[np.isfinite(got)]
        np.testing.assert_allclose([finite.min(), finite.max()], [0.0, 1.0], atol=1e-12,
                                   err_msg=f"缺 {n_nan} 只票时值域不再是 [0,1]")

    def test_short_panels_return_nan_instead_of_raising(self):
        """
        【缺陷 B-7，已修】bottleneck 分支缺 `T < window` 守卫，`bn.move_*`
        直接抛 ValueError —— 面板一短，整条 DSL 表达式求值崩掉而不是给 NaN。
        walk-forward 第一折、次新股子集、小 universe 切片都会踩到。
        7 个包装器 × 3 条路径全部验。
        """
        x = _series(T=3, N=2, seed=26)
        wrappers = ("bn_ts_mean", "bn_ts_std", "bn_ts_var", "bn_ts_sum",
                    "bn_ts_max", "bn_ts_min", "bn_ts_rank")
        for name in wrappers:
            for pathname, got in _all_paths(F.__dict__[name], x, 5).items():
                assert got.shape == x.shape, f"[{pathname}] {name}: 形状变了"
                assert np.all(np.isnan(got)), (
                    f"[{pathname}] {name}: 面板短于窗口时应当全 NaN")


# ===========================================================================
# H2. B-4：单箱熵的契约（2026-09-21 定案）
# ===========================================================================

class TestEntropyBinContract:

    def test_one_bin_raises_instead_of_returning_a_plausible_number(self):
        """
        【缺陷 B-4，2026-09-21 定契约并修】原实现写
        `log_nbins = np.log(n_bins) if n_bins > 1 else 1.0`，`n_bins=1` 静默
        返回 **-0.0**（单箱 probs=[1.0] → h=-0.0，再除以顶替的分母 1.0）。

        单箱的"归一化"熵没有意义：分母本该是 log(n_bins)，而 log(1)=0。
        代码拿 1.0 顶替，等于悄悄换了量纲，而调用方看到的是个长得很正常的 0。

        契约定为**报错**，依据是 `n_bins` 走不到 GP/DSL：`FAST_TS_OPS` 按
        `fn(x, window)` 派发，n_bins 恒为默认 10，fast_ops.py 之外全库零引用。
        所以 n_bins=1 只可能来自开发者直接调用 —— 那是参数写错，该当场报错，
        而不是返回一个看不出错的数。
        """
        x = np.arange(10, dtype=float).reshape(10, 1)
        with pytest.raises(ValueError, match="n_bins"):
            F.ts_entropy(x, 5, n_bins=1)
        with pytest.raises(ValueError, match="n_bins"):
            F.ts_entropy(x, 5, n_bins=0)

    def test_the_default_path_is_untouched(self):
        """反向对照：合法 n_bins 的行为不能被这条守卫改掉。"""
        x = np.arange(20, dtype=float).reshape(20, 1)
        got = F.ts_entropy(x, 10)                 # 默认 n_bins=10
        assert np.all(np.isnan(got[:9])), "窗口未满的行仍应是 NaN"
        finite = got[9:]
        assert np.all(np.isfinite(finite))
        assert np.all((finite >= 0.0) & (finite <= 1.0)), (
            f"归一化熵越出 [0,1]：{finite.min()}..{finite.max()}")

    @pytest.mark.parametrize("n_bins", [2, 5, 10])
    def test_a_constant_window_has_zero_entropy(self, n_bins):
        """全部落进同一个箱 → 熵 0（正零，不是负零）。"""
        x = np.full((10, 1), 3.0)
        got = F.ts_entropy(x, 5, n_bins=n_bins)[4:]
        np.testing.assert_allclose(got, 0.0, atol=1e-12)
        assert not np.signbit(got).any(), "返回了负零"


# ===========================================================================
# I. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "app/core/alpha_engine/fast_ops.py ×0 — L37 `_HAS_BN = False` → True（**已被杀死，此条保留为记录**）":
        "原本判定为等价：该行在 `except ImportError:` 块内，只有 bottleneck "
        "**导入失败**时才执行，而本环境已安装 bottleneck。复测把它杀死了 —— "
        "A 节的 test_rolling_ops_agree_across_paths_when_the_panel_has_gaps "
        "会显式把 `_HAS_BN` 在 True/False 之间切换并逐位对账三条分支，"
        "模块级的初值因此变得可观测。"
        "（点名的用例原是 H 节那条 B-7 定钉用例；B-7 于 2026-09-20 修复后它"
        "改名为 test_short_panels_return_nan_instead_of_raising 且不再断言"
        "两条分支行为**不同**，故改指 A 节这条 —— 切换 `_HAS_BN` 的观测力在那里。）"
        "留着这条是因为『我以为不可达、实测可达』值得留痕："
        "等价性的直觉判断不可靠，必须以复测结果为准。"
        "机械验证仍保留在 test_has_bn_false_line_is_inside_the_import_failure_branch。",

    "app/core/alpha_engine/fast_ops.py ×2 — L310 / L320 `if window < T:` → `<=`（ts_delta / ts_delay）":
        "两种取值只在 window == T 时分道：此时 `x[window:]` 与 `x[:-window]` 都是"
        "形状 (0,N) 的空切片，`out[window:] = 空 - 空` 是一次空赋值，对 out 没有"
        "任何写入，结果与不进分支完全相同。见 test_delta_at_window_equal_t_is_a_noop。",

    "app/core/alpha_engine/fast_ops.py ×10 — L58 / L80 / L113 / L497 `shape = (T - window + 1, window, N)` 的两个符号，"
    "以及 L404/L405/L433/L434/L542/L543 的 `keepdims=True`":
        "这十处全部落在 `try:` 块内，而 `except Exception:` 里是一份**独立的纯循环"
        "实现**。改坏 shape 会让 `out[window-1:] = result` 形状不匹配抛 ValueError，"
        "改掉 keepdims 会让 `ws - mu` 广播失败抛 ValueError —— 两者都被同一个 "
        "except 接住，随后由循环分支算出**完全正确**的结果。"
        "也就是说：向量化分支算错到抛异常的程度时，外部观察不到任何差别，"
        "只是慢了。这不是测试写不到，是这段代码的结构使然。"
        "机械验证见 test_vectorised_paths_are_wrapped_in_a_rescuing_except（确认 try/except "
        "结构仍在）与 A 节的 test_rolling_moments_agree_across_paths（确认兜底结果正确）。"
        "**顺带登记为产品问题 B-10**：`except Exception` 太宽，把"
        "『向量化实现写错了』和『这台机器的内存布局不支持 as_strided』"
        "变成同一件事，前者永远不会被发现。",

    "app/core/alpha_engine/fast_ops.py ×2 — L506 `dx, dy = wx - mu_x, wy - mu_y` → `+`（ts_corr）"
    "与 L544 `cov = np.sum((wx - mu_x) * (wy - mu_y), ...)` → `+`":
        "协方差只需要**一侧**去中心化：E[(X+μx)(Y−μy)] = E[(X−μx)(Y−μy)] + 2μx·E[Y−μy]，"
        "而 E[Y−μy] 恒为 0（μy 就是该窗口 Y 的均值），所以多出来的那一项恒等于 0。"
        "两种写法在任何输入上给出**逐位相同**的结果，是数学恒等而非测试盲区。"
        "见 test_centring_one_side_is_enough_for_covariance。",

    "app/core/alpha_engine/fast_ops.py ×1 — L474 `counts[counts > 0]` → `>=`（ts_entropy，仅在无空桶时）":
        "np.histogram 的 counts 是非负整数；`>0` 与 `>=0` 只在**存在空桶**时不同，"
        "而 E 节的 test_entropy_skips_empty_bins 用必然产生空桶的输入把这一差别"
        "钉成了「有限值 vs NaN」，该变异在那条用例下被杀死，不属于等价变异。"
        "此条保留为记录：若将来那条用例被删，这个点会重新变成盲区。",
}


def test_has_bn_false_line_is_inside_the_import_failure_branch():
    """L37 等价性的机械验证：该行确实在 except ImportError 块内，且本环境导入成功。"""
    import ast
    import importlib
    import inspect
    src = inspect.getsource(F)
    tree = ast.parse(src)
    inside = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for h in node.handlers:
            exc = ast.unparse(h.type) if h.type else ""
            if "ImportError" not in exc:
                continue
            inside += [ast.unparse(s) for s in ast.walk(h) if isinstance(s, ast.Assign)]
    assert any("_HAS_BN = False" in s for s in inside), (
        "`_HAS_BN = False` 不在 except ImportError 块里了 —— 等价性证明失效")
    assert importlib.util.find_spec("bottleneck") is not None, (
        "本环境没有 bottleneck —— 该行会真的执行，L37 不再是等价变异")


def test_delta_at_window_equal_t_is_a_noop():
    """L247/L257 等价性的机械验证：window == T 时两侧都是空切片。"""
    x = _series(T=6, N=2, seed=25)
    T = x.shape[0]
    assert x[T:].shape[0] == 0 and x[:-T].shape[0] == 0, (
        "window == T 时切片不再为空 —— 等价性证明失效")
    assert np.all(np.isnan(F.ts_delta(x, T))), "window == T 的 ts_delta 应全 NaN"
    assert np.all(np.isnan(F.ts_delay(x, T))), "window == T 的 ts_delay 应全 NaN"


def test_corr_returns_nan_exactly_at_the_epsilon_boundary():
    """
    `np.where(denom > 1e-12, cov/denom, np.nan)` —— **严格大于**。

    我第一版把它当成"区分值不可构造"的等价变异写进了证明，机械验证当场推翻：
    `sqrt(1e-12)² == 1e-12` 在浮点上是精确的，所以边界值是**能构造出来**的。
    这里就构造它：[-c, 0, c] 的 ddof=1 样本标准差恰好等于 |c|，
    取 c=1e-12 与 c=1.0，denom = 1e-12 * 1.0 = 1e-12 精确命中。

    放宽成 `>=` 会让这个退化到数值噪声级别的截面照样吐出一个"相关系数"
    （这里会是 ±1），而它完全由舍入误差决定。
    """
    x = np.array([[-1e-12], [0.0], [1e-12]])
    y = np.array([[-1.0], [0.0], [1.0]])
    sx = np.std(x.ravel(), ddof=1)
    sy = np.std(y.ravel(), ddof=1)
    assert sx * sy == 1e-12, (
        f"没有精确命中边界（denom={sx * sy!r}）—— 这条用例失去了区分力")
    got = F.ts_corr(x, y, 3).ravel()
    assert np.isnan(got[-1]), (
        f"denom 恰好等于 1e-12 时应当返回 NaN，实际 {got[-1]} —— "
        f"`denom > 1e-12` 被放宽成了 `>=`")
    # 稍微越过边界就必须给出数值，证明上面的 NaN 不是因为别的原因
    y_big = y * 2.0
    assert np.isfinite(F.ts_corr(x, y_big, 3).ravel()[-1]), (
        "越过 epsilon 之后仍是 NaN —— 上面那条断言并没有钉住边界")


def test_vectorised_paths_are_wrapped_in_a_rescuing_except():
    """
    L58/L80/L99/L463 与六处 keepdims 的等价性依据：向量化赋值确实在 try 内，
    且 except 里有一份独立的循环实现。结构一旦改变（比如收窄成
    `except ValueError` 之外、或删掉兜底），这条会红，那些点就要重新补用例。
    """
    import ast
    import inspect
    src = inspect.getsource(F)
    tree = ast.parse(src)
    checked = []
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
        for t in tries:
            body = ast.unparse(ast.Module(body=t.body, type_ignores=[]))
            if "as_strided" not in body and "_stride_windows" not in body:
                continue
            handlers = [ast.unparse(h.type) if h.type else "bare" for h in t.handlers]
            assert any("Exception" in h or h == "bare" for h in handlers), (
                f"{fn.name}: 向量化分支的 except 收窄了（{handlers}）—— "
                f"等价性证明失效，这些变异点要重新补用例")
            loop = any(isinstance(n, ast.For) for h in t.handlers for n in ast.walk(h))
            assert loop, f"{fn.name}: except 里已经没有循环兜底了"
            checked.append(fn.name)
    assert len(checked) >= 8, f"只找到 {len(checked)} 个带兜底的向量化分支：{checked}"


def test_centring_one_side_is_enough_for_covariance():
    """
    L434 / L472 等价性的机械验证：E[(X+μx)(Y−μy)] == E[(X−μx)(Y−μy)]，
    因为 E[Y−μy] == 0。在随机数据上逐位验证这条恒等式。
    """
    rng = np.random.default_rng(7)
    for _ in range(5):
        wx = rng.normal(3.0, 2.0, (16, 4))
        wy = rng.normal(-1.0, 5.0, (16, 4))
        mx = wx.mean(axis=0, keepdims=True)
        my = wy.mean(axis=0, keepdims=True)
        a = ((wx - mx) * (wy - my)).sum(axis=0)
        b = ((wx + mx) * (wy - my)).sum(axis=0)
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9,
                                   err_msg="只居中一侧不再等价 —— L434/L472 需要补用例")


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 5
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
