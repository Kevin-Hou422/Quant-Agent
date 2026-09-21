"""
High-Performance Operator Library.

All operators accept and return 2-D NumPy arrays of shape (T, N):
  - axis=0 is the time axis
  - axis=1 is the asset axis

Bottleneck is used for rolling operations (falls back to NumPy with a
warning if not installed). All rolling operators enforce strict NaN
policy: fewer than `window` valid observations → NaN output.

Cross-sectional operators use fully-vectorised NumPy operations (no
Python-level loops). NaN assets are excluded from aggregation but
their positions remain NaN in the output.
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Bottleneck availability
# ---------------------------------------------------------------------------

try:
    import bottleneck as bn
    _HAS_BN = True
except ImportError:
    warnings.warn(
        "bottleneck is not installed. Rolling operations will fall back to "
        "pure NumPy (slower). Install with: pip install bottleneck",
        ImportWarning,
        stacklevel=2,
    )
    _HAS_BN = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 0:
        return x  # scalar — let broadcasting handle it
    if x.ndim == 1:
        return x[:, np.newaxis]
    return x


def _numpy_move_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Pure-NumPy fallback for rolling mean."""
    out = np.full_like(x, np.nan, dtype=float)
    T, N = x.shape
    if T < window:
        return out
    shape   = (T - window + 1, window, N)
    strides = (x.strides[0], x.strides[0], x.strides[1])
    try:
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        valid = (~np.isnan(windows)).sum(axis=1)
        sums  = np.nansum(windows, axis=1)
        out[window - 1:] = np.where(valid >= window, sums / valid, np.nan)
    except Exception:
        for i in range(window - 1, T):
            block = x[i - window + 1: i + 1]
            valid_counts = (~np.isnan(block)).sum(axis=0)
            sums = np.nansum(block, axis=0)
            out[i] = np.where(valid_counts >= window, sums / valid_counts, np.nan)
    return out


def _numpy_move_std(x: np.ndarray, window: int) -> np.ndarray:
    """Pure-NumPy fallback for rolling std."""
    out = np.full_like(x, np.nan, dtype=float)
    T, N = x.shape
    if T < window:
        return out
    shape   = (T - window + 1, window, N)
    strides = (x.strides[0], x.strides[0], x.strides[1])
    try:
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        valid = (~np.isnan(windows)).sum(axis=1)
        var   = np.nanvar(windows, axis=1, ddof=1)
        out[window - 1:] = np.where(valid >= window, np.sqrt(var), np.nan)
    except Exception:
        for i in range(window - 1, T):
            block = x[i - window + 1: i + 1].astype(float)
            valid = np.sum(~np.isnan(block), axis=0)
            var = np.nanvar(block, axis=0, ddof=1)
            out[i] = np.where(valid >= window, np.sqrt(var), np.nan)
    return out


def _too_short(x: np.ndarray, window: int) -> bool:
    """
    面板行数不足一个窗口。

    【缺陷 B-7，2026-09-20 修】模块 docstring 承诺"不足 window 个有效观测 → NaN"。
    numpy 分支一直照做（`if T < window: return 全 NaN`），bottleneck 分支却没有
    这个守卫 —— `bn.move_*` 直接抛 `ValueError: Moving window (=5) must between
    1 and 3`。于是面板一短，整条 DSL 表达式求值**崩掉**而不是给 NaN：
    walk-forward 第一折、次新股子集、小 universe 切片都会踩到。
    7 个 bn_* 包装器全部受影响（已逐个实测），所以守卫提到分支**之前**。
    """
    # `x.ndim` 取真值而不写 `x.ndim >= 1`：`_ensure_2d` 之后 ndim 只可能是 0 或 ≥2，
    # 于是 `>= 1` 与 `> 1` 在任何输入上同值 —— 那是个**任何用例都杀不死的等价变异点**
    # （已用 verify_mutant 实测存活）。写成真值判断就没有这个可变异的比较符。
    return bool(x.ndim and x.shape[0] < window)


def _stride_windows(x: np.ndarray, window: int):
    """Return stride_tricks view of shape (T-w+1, window, N)."""
    T, N = x.shape
    shape   = (T - window + 1, window, N)
    strides = (x.strides[0], x.strides[0], x.strides[1])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


# ---------------------------------------------------------------------------
# Time-Series Operators — Standard
# ---------------------------------------------------------------------------

def bn_ts_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean; NaN for fewer than `window` valid observations."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    if _too_short(x, window):                      # B-7：两条分支统一给全 NaN
        return np.full(x.shape, np.nan)
    if _HAS_BN:
        return bn.move_mean(x, window=window, min_count=window, axis=0)
    return _numpy_move_mean(x, window)


def bn_ts_std(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling std (ddof=1); NaN for fewer than `window` valid obs."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    if _too_short(x, window):                      # B-7：两条分支统一给全 NaN
        return np.full(x.shape, np.nan)
    if _HAS_BN:
        return bn.move_std(x, window=window, min_count=window, axis=0, ddof=1)
    return _numpy_move_std(x, window)


def bn_ts_var(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling variance (ddof=1)."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    if _too_short(x, window):                      # B-7：两条分支统一给全 NaN
        return np.full(x.shape, np.nan)
    if _HAS_BN:
        return bn.move_var(x, window=window, min_count=window, axis=0, ddof=1)
    std = _numpy_move_std(x, window)
    return std ** 2


def bn_ts_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum; NaN for fewer than `window` valid obs."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    if _too_short(x, window):                      # B-7：两条分支统一给全 NaN
        return np.full(x.shape, np.nan)
    if _HAS_BN:
        return bn.move_sum(x, window=window, min_count=window, axis=0)
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T < window:
        return out
    try:
        windows = _stride_windows(x, window)
        valid = (~np.isnan(windows)).sum(axis=1)
        sums  = np.nansum(windows, axis=1)
        out[window - 1:] = np.where(valid >= window, sums, np.nan)
    except Exception:
        for i in range(window - 1, T):
            block = x[i - window + 1: i + 1]
            vc = (~np.isnan(block)).sum(axis=0)
            s  = np.nansum(block, axis=0)
            out[i] = np.where(vc >= window, s, np.nan)
    return out


def bn_ts_max(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling max; NaN for fewer than `window` valid observations."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    if _too_short(x, window):                      # B-7：两条分支统一给全 NaN
        return np.full(x.shape, np.nan)
    if _HAS_BN:
        return bn.move_max(x, window=window, min_count=window, axis=0)
    T, N = x.shape
    out = np.full((T, N), np.nan)
    try:
        windows = _stride_windows(x, window)
        # B-5：`np.max` **传播** NaN，`np.nanmax` 忽略它 —— 见下方注释。
        out[window - 1:] = np.max(windows, axis=1)
    except Exception:
        for i in range(window - 1, T):
            out[i] = np.max(x[i - window + 1: i + 1], axis=0)
    return out


def bn_ts_min(x: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling min; NaN for fewer than `window` valid observations.

    【缺陷 B-5，2026-09-20 修】模块 docstring 承诺 strict NaN policy，
    bottleneck 分支用 `min_count=window` 遵守了，numpy 两条分支却用
    `np.nanmin/np.nanmax` **直接忽略** NaN —— 同一个面板换条执行路径，
    缺了一根 bar 的标的在 bottleneck 下是 NaN、在 numpy 下照样吐出极值。
    那个极值还是**用更少的观测**算出来的：停牌期间的 ts_max 会系统性偏低、
    ts_min 偏高，而调用方无从知道这一行的样本数不足。
    改用 `np.max/np.min`（NaN 自然传播），三条路径逐位一致。
    """
    x = _ensure_2d(np.asarray(x, dtype=float))
    if _too_short(x, window):                      # B-7：两条分支统一给全 NaN
        return np.full(x.shape, np.nan)
    if _HAS_BN:
        return bn.move_min(x, window=window, min_count=window, axis=0)
    T, N = x.shape
    out = np.full((T, N), np.nan)
    try:
        windows = _stride_windows(x, window)
        out[window - 1:] = np.min(windows, axis=1)
    except Exception:
        for i in range(window - 1, T):
            out[i] = np.min(x[i - window + 1: i + 1], axis=0)
    return out


def _avg_rank_fraction(windows: np.ndarray) -> np.ndarray:
    """
    窗口内**最后一个**值的平均秩，归一到 [0,1]。

    平均秩（0-based）r = #{更小} + (#{并列} - 1)/2，归一分母 count-1。
    并列取中点 —— 与 `cs_rank`、`average_ranks_1d` 同一个约定。
    """
    last  = windows[:, -1:, :]
    valid = ~np.isnan(windows)
    count = valid.sum(axis=1)
    less  = ((windows < last) & valid).sum(axis=1)
    ties  = ((windows == last) & valid).sum(axis=1)     # 含最后一个值自己
    avg_rank = less + (ties - 1.0) / 2.0
    denom = np.maximum(count - 1, 1)
    return np.where(count >= windows.shape[1], avg_rank / denom, np.nan)


def bn_ts_rank(x: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling rank of the most recent value, as a percentile in [0, 1].

    0 = 窗口内最低，1 = 最高，并列取平均秩。

    【缺陷 B-1，2026-09-20 修】两条分支此前算的**根本不是同一个量**：
      - bottleneck：`bn.move_rank` 的值域是 **[-1, 1]**（实测），再 `/window`
        → 实际值域 [-1/w, 1/w]。w=5 时单调上升序列的最新一根给 **0.2**，
        w=20 时给 0.05 —— 同一个"排在窗口最高位"的事实，**因窗口长度而异**，
        而且一半取值是负的。docstring 承诺的 [0,1] 从来没成立过。
      - numpy：`le/count`（含并列的"小于等于"计数），值域 (0,1]，取不到 0。

    后果不只是"数不好看"：`ts_rank(close, 20) > 0.8` 这类阈值在 bottleneck
    环境下**恒为假**（上界才 0.05），装没装 bottleneck 决定了信号有没有。
    GP 进化出来的表达式因此依赖于运行环境。

    统一到平均秩 /(count-1)：bottleneck 的 [-1,1] 线性映射回 [0,1] 就是
    `(raw+1)/2` —— 这不是凑出来的，`bn.move_rank` 的定义
    `(#less - #greater)/(count-1)` 恒等于 `2·r/(count-1) - 1`（r 为 0-based
    平均秩），两边代数相等，已用暴力参照逐位校验。
    """
    x = _ensure_2d(np.asarray(x, dtype=float))
    if _too_short(x, window):                      # B-7：两条分支统一给全 NaN
        return np.full(x.shape, np.nan)
    if _HAS_BN:
        raw = bn.move_rank(x, window=window, min_count=window, axis=0)
        return (raw + 1.0) / 2.0
    T, N = x.shape
    out = np.full((T, N), np.nan)
    try:
        out[window - 1:] = _avg_rank_fraction(_stride_windows(x, window))
    except Exception:
        for i in range(window - 1, T):
            block = x[i - window + 1: i + 1][np.newaxis]      # (1, window, N)
            out[i] = _avg_rank_fraction(block)[0]
    return out


def ts_decay_linear(x: np.ndarray, window: int) -> np.ndarray:
    """Linearly-weighted moving average (most-recent = highest weight)."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    T, N = x.shape
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    out = np.full((T, N), np.nan)
    if T < window:
        return out
    try:
        windows = _stride_windows(x, window)
        has_nan = np.any(np.isnan(windows), axis=1)
        result  = np.einsum("twn,w->tn", windows, weights)
        result[has_nan] = np.nan
        out[window - 1:] = result
    except Exception:
        for i in range(window - 1, T):
            block = x[i - window + 1: i + 1]
            if np.any(np.isnan(block)):
                continue
            out[i] = weights @ block
    return out


def ts_delta(x: np.ndarray, window: int) -> np.ndarray:
    """x[t] - x[t - window]; first `window` rows → NaN."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if window < T:
        out[window:] = x[window:] - x[:-window]
    return out


def ts_delay(x: np.ndarray, window: int) -> np.ndarray:
    """Lag operator: x shifted by `window` rows; first `window` rows → NaN."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if window < T:
        out[window:] = x[:-window]
    return out


# ---------------------------------------------------------------------------
# Time-Series Operators — Extended
# ---------------------------------------------------------------------------

def ts_argmax(x: np.ndarray, window: int) -> np.ndarray:
    """
    Normalized position of rolling argmax in [0,1].
    0 = maximum occurred at oldest position, 1 = most recent.
    """
    x = _ensure_2d(np.asarray(x, dtype=float))
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T < window:
        return out
    denom = max(window - 1, 1)
    try:
        windows = _stride_windows(x, window)
        has_nan = np.any(np.isnan(windows), axis=1)
        with np.errstate(all="ignore"):
            idx = np.nanargmax(windows, axis=1).astype(float)
        idx[has_nan] = np.nan
        out[window - 1:] = idx / denom
    except Exception:
        for i in range(window - 1, T):
            block = x[i - window + 1: i + 1]
            if np.any(np.isnan(block)):
                continue
            out[i] = np.nanargmax(block, axis=0) / denom
    return out


def ts_argmin(x: np.ndarray, window: int) -> np.ndarray:
    """
    Normalized position of rolling argmin in [0,1].
    0 = minimum occurred at oldest position, 1 = most recent.
    """
    x = _ensure_2d(np.asarray(x, dtype=float))
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T < window:
        return out
    denom = max(window - 1, 1)
    try:
        windows = _stride_windows(x, window)
        has_nan = np.any(np.isnan(windows), axis=1)
        with np.errstate(all="ignore"):
            idx = np.nanargmin(windows, axis=1).astype(float)
        idx[has_nan] = np.nan
        out[window - 1:] = idx / denom
    except Exception:
        for i in range(window - 1, T):
            block = x[i - window + 1: i + 1]
            if np.any(np.isnan(block)):
                continue
            out[i] = np.nanargmin(block, axis=0) / denom
    return out


def ts_zscore(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score: (x[t] - rolling_mean) / rolling_std."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    mean = bn_ts_mean(x, window)
    std  = bn_ts_std(x, window)
    safe_std = np.where(std == 0, np.nan, std)
    return (x - mean) / safe_std


def ts_skew(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling skewness (Fisher-Pearson unbiased estimator)."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T < window or window < 3:
        return out
    n = window
    coeff = (n * n) / ((n - 1) * (n - 2))
    try:
        ws = _stride_windows(x, window)
        has_nan = np.any(np.isnan(ws), axis=1)
        mu  = np.mean(ws, axis=1, keepdims=True)
        std = np.std(ws, axis=1, keepdims=True, ddof=1)
        safe = np.where(std == 0, np.nan, std)
        z    = (ws - mu) / safe
        skew = np.mean(z ** 3, axis=1) * coeff
        skew[has_nan] = np.nan
        out[window - 1:] = skew
    except Exception:
        for i in range(window - 1, T):
            b = x[i - window + 1: i + 1].copy()
            if np.any(np.isnan(b)):
                continue
            mu  = np.mean(b, axis=0)
            std = np.std(b, axis=0, ddof=1)
            z   = (b - mu) / np.where(std == 0, np.nan, std)
            out[i] = np.mean(z ** 3, axis=0) * coeff
    return out


def ts_kurt(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling excess kurtosis (kurtosis - 3)."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T < window or window < 4:
        return out
    try:
        ws = _stride_windows(x, window)
        has_nan = np.any(np.isnan(ws), axis=1)
        mu  = np.mean(ws, axis=1, keepdims=True)
        std = np.std(ws, axis=1, keepdims=True, ddof=1)
        safe = np.where(std == 0, np.nan, std)
        z    = (ws - mu) / safe
        kurt = np.mean(z ** 4, axis=1) - 3.0
        kurt[has_nan] = np.nan
        out[window - 1:] = kurt
    except Exception:
        for i in range(window - 1, T):
            b = x[i - window + 1: i + 1].copy()
            if np.any(np.isnan(b)):
                continue
            mu  = np.mean(b, axis=0)
            std = np.std(b, axis=0, ddof=1)
            z   = (b - mu) / np.where(std == 0, np.nan, std)
            out[i] = np.mean(z ** 4, axis=0) - 3.0
    return out


def ts_entropy(x: np.ndarray, window: int, n_bins: int = 10) -> np.ndarray:
    """
    Rolling normalized Shannon entropy in [0, 1].
    1 = uniform distribution, 0 = all mass on one bin.
    Uses a fixed-bin histogram approximation.

    `n_bins` must be >= 2.

    【缺陷 B-4，2026-09-21 定契约并修】原实现写
    `log_nbins = np.log(n_bins) if n_bins > 1 else 1.0`，于是 `n_bins=1` 静默
    返回 **-0.0**（单箱 probs=[1.0] → h=-0.0，再除以 1.0）。

    单箱的"归一化"熵没有意义：归一化的分母本该是 `log(n_bins)`，而 log(1)=0，
    真要归一就是 0/0。代码用 1.0 顶替分母，等于悄悄换了一个量纲，
    而调用方看不出参数给错了 —— 返回的还是个长得很正常的 0。

    **已核实 `n_bins` 走不到 GP/DSL**：`FAST_TS_OPS` 按 `fn(x, window)` 派发，
    n_bins 恒为默认 10；fast_ops.py 之外全库零引用。所以 `n_bins=1` 只可能来自
    开发者直接调用 —— 那是调用方写错参数，应当**当场报错**，
    而不是返回一个看不出错的数。
    """
    if n_bins < 2:
        raise ValueError(
            f"ts_entropy 需要 n_bins >= 2，收到 {n_bins}。"
            f"单箱分布没有不确定性可言，归一化分母 log(n_bins) 会是 0 —— "
            f"这是调用方的参数错误，不是数据问题。")
    x = _ensure_2d(np.asarray(x, dtype=float))
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T < window:
        return out
    log_nbins = np.log(n_bins)
    for n_idx in range(N):
        col = x[:, n_idx]
        for i in range(window - 1, T):
            block = col[i - window + 1: i + 1]
            if np.any(np.isnan(block)):
                continue
            counts, _ = np.histogram(block, bins=n_bins)
            total = counts.sum()
            if total == 0:
                continue
            probs = counts[counts > 0] / total
            h = -np.sum(probs * np.log(probs))
            # `+ 0.0` 消除**负零**：常数窗口（某只票当天没动）只有一个非空箱，
            # probs=[1.0] → log(1)=0 → `-np.sum(...)` 得到 **-0.0**。
            # 这是登记表 B-4 列的第二个问题，与 n_bins 无关 —— 合法 n_bins
            # 在完全正常的数据上照样触发。负零会往下传播（`1/-0.0 = -inf`），
            # 报告里也显示成 "-0.0"，看着像故障。
            out[i, n_idx] = h / log_nbins + 0.0
    return out


def ts_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling Pearson correlation between x and y.

    【缺陷 B-2，2026-09-20 修】分子分母的自由度不配套：协方差走
    `np.mean(dx*dy)`（ddof=0，除以 w），标准差走 `np.std(..., ddof=1)`
    （除以 w-1）。相关系数于是被系统性压低 **(w-1)/w** —— 完全线性相关的
    两条序列 w=5 时只给 0.8、w=20 时给 0.95。偏差随窗口变化，所以
    `|ts_corr| > 0.7` 这类阈值在短窗口上更难触发，短窗口的配对信号被压制。
    分子改成同样的 ddof=1。（`ts_cov` 一直是对的，两者本该一致。）
    """
    x = _ensure_2d(np.asarray(x, dtype=float))
    y = _ensure_2d(np.asarray(y, dtype=float))
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T < window:
        return out
    shape = (T - window + 1, window, N)
    sx = (x.strides[0], x.strides[0], x.strides[1])
    sy = (y.strides[0], y.strides[0], y.strides[1])
    try:
        wx = np.lib.stride_tricks.as_strided(x, shape=shape, strides=sx)
        wy = np.lib.stride_tricks.as_strided(y, shape=shape, strides=sy)
        has_nan = np.any(np.isnan(wx) | np.isnan(wy), axis=1)
        mu_x = np.mean(wx, axis=1, keepdims=True)
        mu_y = np.mean(wy, axis=1, keepdims=True)
        dx, dy = wx - mu_x, wy - mu_y
        cov_xy = np.sum(dx * dy, axis=1) / (window - 1)    # B-2：与 std 的 ddof=1 配套
        std_x  = np.std(wx, axis=1, ddof=1)
        std_y  = np.std(wy, axis=1, ddof=1)
        denom  = std_x * std_y
        corr   = np.where(denom > 1e-12, cov_xy / denom, np.nan)
        corr[has_nan] = np.nan
        out[window - 1:] = corr
    except Exception:
        for i in range(window - 1, T):
            bx = x[i - window + 1: i + 1]
            by = y[i - window + 1: i + 1]
            for n_idx in range(N):
                mask = ~(np.isnan(bx[:, n_idx]) | np.isnan(by[:, n_idx]))
                if mask.sum() < 2:
                    continue
                c = np.corrcoef(bx[mask, n_idx], by[mask, n_idx])
                out[i, n_idx] = c[0, 1]
    return out


def ts_cov(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Rolling covariance (ddof=1) between x and y."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    y = _ensure_2d(np.asarray(y, dtype=float))
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T < window:
        return out
    shape = (T - window + 1, window, N)
    sx = (x.strides[0], x.strides[0], x.strides[1])
    sy = (y.strides[0], y.strides[0], y.strides[1])
    try:
        wx = np.lib.stride_tricks.as_strided(x, shape=shape, strides=sx)
        wy = np.lib.stride_tricks.as_strided(y, shape=shape, strides=sy)
        has_nan = np.any(np.isnan(wx) | np.isnan(wy), axis=1)
        mu_x = np.mean(wx, axis=1, keepdims=True)
        mu_y = np.mean(wy, axis=1, keepdims=True)
        cov  = np.sum((wx - mu_x) * (wy - mu_y), axis=1) / (window - 1)
        cov[has_nan] = np.nan
        out[window - 1:] = cov
    except Exception:
        for i in range(window - 1, T):
            bx = x[i - window + 1: i + 1]
            by = y[i - window + 1: i + 1]
            for n_idx in range(N):
                mask = ~(np.isnan(bx[:, n_idx]) | np.isnan(by[:, n_idx]))
                if mask.sum() < 2:
                    continue
                c = np.cov(bx[mask, n_idx], by[mask, n_idx])
                out[i, n_idx] = c[0, 1]
    return out


# ---------------------------------------------------------------------------
# Cross-Sectional Operators — Standard
# ---------------------------------------------------------------------------

def cs_rank(x: np.ndarray) -> np.ndarray:
    """
    Cross-sectional percentile rank [0, 1] per row.
    NaN assets excluded from ranking; ties resolved by average rank.

    【缺陷 B-3 + B-6，2026-09-20 修】旧实现 `argsort(argsort(where(nan, -inf, x)))`
    一行里错了两件事：

    B-3 —— 那是**序数**名次，不是 docstring 承诺的平均名次。并列值按**列顺序**
    被强行排出先后，于是同一个截面把标的换个顺序，因子值就变了。而列序不是
    市场事实，是面板的存储顺序。（与 C-1 是同一个错误的截面版本。）

    B-6 —— NaN 被填成 `-inf` 后**参与了排序**，占掉最低的几个名次，分母却只按
    有效个数算 `valid_count-1`。于是有效资产的名次从 n_nan/(n_valid-1) 起跳，
    最高到 (n-1)/(n_valid-1) > 1 —— 值域越出 [0,1]，缺失越多偏得越狠。
    `rank(x) > 0.9` 这类条件会因为当天缺了几只票而莫名多命中，
    而且命中的多少取决于**停牌数量**，不取决于信号。

    新实现：NaN 不填充（`np.argsort` 本就把 NaN 排到末尾，不占有效名次），
    在排序后的序列上按并列区间取中点，再散射回原位。全向量化，无 Python 循环。
    """
    x = _ensure_2d(np.asarray(x, dtype=float))
    T, N = x.shape
    nan_mask = np.isnan(x)

    order = np.argsort(x, axis=1, kind="mergesort")       # NaN 自然落到末尾
    xs    = np.take_along_axis(x, order, axis=1)          # 每行升序后的值

    # 并列区间的起点/终点（NaN != NaN，所以每个 NaN 自成一组，稍后被屏蔽）
    idx = np.arange(N)
    is_start = np.ones((T, N), dtype=bool)
    is_start[:, 1:] = xs[:, 1:] != xs[:, :-1]
    is_end = np.ones((T, N), dtype=bool)
    is_end[:, :-1] = is_start[:, 1:]

    # 非边界位置填的是"输不掉的哨兵"：起点向右取最大故填 0，终点自右向左取最小故填 N。
    # 第 0 位必是起点、第 N-1 位必是终点，所以哨兵永远赢不了真实下标。
    # 终点这侧写 `N` 而不是 `N - 1`：两者都正确（任何 ≥ N-1 的值都行），
    # 但 `N - 1` 会多出一个**杀不死的等价变异点**（已实测 `N-1 → N+1` 存活）。
    start = np.maximum.accumulate(np.where(is_start, idx, 0), axis=1)
    end   = np.minimum.accumulate(np.where(is_end, idx, N)[:, ::-1], axis=1)[:, ::-1]

    ranks_sorted = 0.5 * (start + end)                    # 并列取中点 = 平均秩
    ranks = np.empty((T, N), dtype=float)
    np.put_along_axis(ranks, order, ranks_sorted, axis=1)

    valid_count = (~nan_mask).sum(axis=1, keepdims=True).astype(float)
    ranks = ranks / np.maximum(valid_count - 1.0, 1.0)
    ranks[nan_mask] = np.nan
    return ranks


def cs_zscore(x: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score per row (NaN-aware)."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    mu    = np.nanmean(x, axis=1, keepdims=True)
    sigma = np.nanstd(x, axis=1, ddof=1, keepdims=True)
    safe  = np.where(sigma == 0, np.nan, sigma)
    return (x - mu) / safe


def cs_scale(x: np.ndarray) -> np.ndarray:
    """L1-norm scaling per row: x / sum(|x|)."""
    x   = _ensure_2d(np.asarray(x, dtype=float))
    l1  = np.nansum(np.abs(x), axis=1, keepdims=True)
    safe = np.where(l1 == 0, np.nan, l1)
    return x / safe


def ind_neutralize(x: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Subtract group mean per row (industry neutralization).
    Falls back to cs_zscore when groups=None.
    """
    x = _ensure_2d(np.asarray(x, dtype=float))
    if groups is None:
        return cs_zscore(x)
    groups = np.asarray(groups, dtype=int)
    out = x.copy()
    for g in np.unique(groups):
        mask  = (groups == g)
        block = x[:, mask]
        gmean = np.nanmean(block, axis=1, keepdims=True)
        out[:, mask] = block - gmean
    return out


# ---------------------------------------------------------------------------
# Cross-Sectional Operators — Extended
# ---------------------------------------------------------------------------

def cs_winsorize(x: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Cross-sectional winsorize at ±k std dev per row."""
    x  = _ensure_2d(np.asarray(x, dtype=float))
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, ddof=1, keepdims=True)
    return np.clip(x, mu - k * sd, mu + k * sd)


def cs_normalize(x: np.ndarray) -> np.ndarray:
    """Cross-sectional min-max normalization to [0, 1] per row."""
    x  = _ensure_2d(np.asarray(x, dtype=float))
    mn = np.nanmin(x, axis=1, keepdims=True)
    mx = np.nanmax(x, axis=1, keepdims=True)
    denom = mx - mn
    safe  = np.where(denom == 0, np.nan, denom)
    return (x - mn) / safe


# ---------------------------------------------------------------------------
# Group Operators  (cross-sectional within user-defined groups)
# ---------------------------------------------------------------------------

def group_rank(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Cross-sectional rank within each group."""
    x      = _ensure_2d(np.asarray(x, dtype=float))
    groups = np.asarray(groups, dtype=int)
    out    = np.full_like(x, np.nan)
    for g in np.unique(groups):
        mask = (groups == g)
        out[:, mask] = cs_rank(x[:, mask])
    return out


def group_zscore(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score within each group."""
    x      = _ensure_2d(np.asarray(x, dtype=float))
    groups = np.asarray(groups, dtype=int)
    out    = np.full_like(x, np.nan)
    for g in np.unique(groups):
        mask = (groups == g)
        out[:, mask] = cs_zscore(x[:, mask])
    return out


def group_mean(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Broadcast within-group mean to each asset per row."""
    x      = _ensure_2d(np.asarray(x, dtype=float))
    groups = np.asarray(groups, dtype=int)
    out    = np.full_like(x, np.nan)
    for g in np.unique(groups):
        mask  = (groups == g)
        gmean = np.nanmean(x[:, mask], axis=1, keepdims=True)
        out[:, mask] = np.broadcast_to(gmean, x[:, mask].shape).copy()
    return out


def group_neutralize(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract within-group mean (group-level demean)."""
    return ind_neutralize(x, groups)


# ---------------------------------------------------------------------------
# New operators (Task 3.4)
# ---------------------------------------------------------------------------

def ts_momentum_decay(x: np.ndarray, window: int) -> np.ndarray:
    """
    Skip-1 momentum: return from t-window-1 to t-1, excluding the most
    recent period.  Eliminates the short-term reversal that contaminates
    raw momentum signals.

    Equivalent to: ts_delay(ts_delta(x, window), 1) = x[t-1] - x[t-1-window]

    Academic reference: Jegadeesh & Titman (1993) standard 12-1 momentum uses
    the prior 12 months' return while skipping the most recent month to avoid
    short-term reversal contamination.
    """
    delayed = ts_delay(x, 1)        # x[t-1]
    return ts_delta(delayed, window)  # x[t-1] - x[t-1-window]


def cs_demean(x: np.ndarray) -> np.ndarray:
    """Subtract cross-sectional mean per row (pure demean, no scaling)."""
    mu = np.nanmean(x, axis=1, keepdims=True)
    return x - mu


def cs_sector_neutral(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """
    Sector-neutralized cross-sectional signal.

    Subtracts the within-sector (GICS) mean from each asset's signal,
    producing a signal with zero net exposure to each sector.

    Uses the dataset['groups'] field which carries real GICS L1 integer
    codes after Phase 0 real-data activation.

    Functionally equivalent to ind_neutralize(x, groups); named separately
    to express explicit financial intent in DSL expressions.

    Parameters
    ----------
    x      : (T, N) signal array
    groups : (T, N) or (N,) integer sector-code array.
             GICS codes are static so if 2-D, row 0 is used as the grouping
             vector (same assignment for every time step).
    """
    g = np.asarray(groups)
    # ind_neutralize expects a 1-D (N,) group vector
    groups_1d = g[0] if g.ndim == 2 else g
    return ind_neutralize(x, groups_1d)


# ---------------------------------------------------------------------------
# Advanced / Conditional Operators
# ---------------------------------------------------------------------------

def signed_power(x: np.ndarray, p: float | np.ndarray) -> np.ndarray:
    """sign(x) * |x|^p"""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.abs(x) ** p


def op_if_else(cond: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """np.where(cond, x, y) — vectorized conditional."""
    return np.where(np.asarray(cond).astype(bool), x, y)


def op_where(cond: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Alias for op_if_else."""
    return op_if_else(cond, x, y)


def op_trade_when(cond: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return x where cond is True, 0 elsewhere."""
    return np.where(np.asarray(cond).astype(bool), np.asarray(x, dtype=float), 0.0)


def op_and(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Element-wise logical AND → float (0/1)."""
    return (np.asarray(x).astype(bool) & np.asarray(y).astype(bool)).astype(float)


def op_or(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Element-wise logical OR → float (0/1)."""
    return (np.asarray(x).astype(bool) | np.asarray(y).astype(bool)).astype(float)


def op_not(x: np.ndarray) -> np.ndarray:
    """Element-wise logical NOT → float (0/1)."""
    return (~np.asarray(x).astype(bool)).astype(float)


# ---------------------------------------------------------------------------
# Dispatch tables (used by typed_nodes)
# ---------------------------------------------------------------------------

# Single-input TS ops: fn(x, window)
FAST_TS_OPS = {
    "ts_mean":            bn_ts_mean,
    "ts_std":             bn_ts_std,
    "ts_var":             bn_ts_var,
    "ts_sum":             bn_ts_sum,
    "ts_max":             bn_ts_max,
    "ts_min":             bn_ts_min,
    "ts_rank":            bn_ts_rank,
    "ts_decay_linear":    ts_decay_linear,
    "ts_delta":           ts_delta,
    "ts_delay":           ts_delay,
    "ts_argmax":          ts_argmax,
    "ts_argmin":          ts_argmin,
    "ts_zscore":          ts_zscore,
    "ts_skew":            ts_skew,
    "ts_kurt":            ts_kurt,
    "ts_entropy":         ts_entropy,
    "ts_momentum_decay":  ts_momentum_decay,   # Task 3.4
    # Two-input ops — called as fn(x, y, window) via TimeSeriesNode._second_child
    "ts_corr":            ts_corr,
    "ts_cov":             ts_cov,
}

# Two-input TS ops requiring a second child series
_TWO_INPUT_TS_OPS = frozenset({"ts_corr", "ts_cov"})

FAST_CS_OPS = {
    "rank":             cs_rank,
    "zscore":           cs_zscore,
    "scale":            cs_scale,
    "ind_neutralize":   ind_neutralize,
    "sector_neutral":   cs_sector_neutral,     # Task 3.4
    "winsorize":        cs_winsorize,
    "normalize":        cs_normalize,
}

FAST_GROUP_OPS = {
    "group_rank":       group_rank,
    "group_zscore":     group_zscore,
    "group_mean":       group_mean,
    "group_neutralize": group_neutralize,
}

def average_ranks_1d(x: np.ndarray) -> np.ndarray:
    """
    一维**平均秩**（并列取均值）—— Spearman 的定义要求的做法。

    【缺陷 C-1，2026-09-20 修】此前 GP 适应度、alpha_combiner、evaluation_utils、
    alpha_workflows **四处**各自写了 `np.argsort(np.argsort(x))`。那是**序数**名次：
    对并列值按出现顺序强行排先后，于是一个截面恒定、**零信息**的信号会拿到
    `[0,1,2,...]` 的假秩 —— 算出来的 IC 不是 0，而是
    "ticker 在面板里的位置 vs 未来收益"的伪相关。
    把列顺序打乱，同一个信号的 fitness 就变了；而列序不是市场事实。

    `daily_trading_loop._average_ranks` 早就有一份正确实现并写明了理由，
    但没有被复用 —— 同一个错误复制了四份。现在统一到这里。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)      # 并列区间取平均秩
        i = j + 1
    return ranks

