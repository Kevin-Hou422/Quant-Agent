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
    if _HAS_BN:
        return bn.move_mean(x, window=window, min_count=window, axis=0)
    return _numpy_move_mean(x, window)


def bn_ts_std(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling std (ddof=1); NaN for fewer than `window` valid obs."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    if _HAS_BN:
        return bn.move_std(x, window=window, min_count=window, axis=0, ddof=1)
    return _numpy_move_std(x, window)


def bn_ts_var(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling variance (ddof=1)."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    if _HAS_BN:
        return bn.move_var(x, window=window, min_count=window, axis=0, ddof=1)
    std = _numpy_move_std(x, window)
    return std ** 2


def bn_ts_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum; NaN for fewer than `window` valid obs."""
    x = _ensure_2d(np.asarray(x, dtype=float))
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
    x = _ensure_2d(np.asarray(x, dtype=float))
    if _HAS_BN:
        return bn.move_max(x, window=window, min_count=window, axis=0)
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T < window:
        return out
    try:
        windows = _stride_windows(x, window)
        out[window - 1:] = np.nanmax(windows, axis=1)
    except Exception:
        for i in range(window - 1, T):
            out[i] = np.nanmax(x[i - window + 1: i + 1], axis=0)
    return out


def bn_ts_min(x: np.ndarray, window: int) -> np.ndarray:
    x = _ensure_2d(np.asarray(x, dtype=float))
    if _HAS_BN:
        return bn.move_min(x, window=window, min_count=window, axis=0)
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T < window:
        return out
    try:
        windows = _stride_windows(x, window)
        out[window - 1:] = np.nanmin(windows, axis=1)
    except Exception:
        for i in range(window - 1, T):
            out[i] = np.nanmin(x[i - window + 1: i + 1], axis=0)
    return out


def bn_ts_rank(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling rank (percentile in [0,1]) of the most recent value."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    if _HAS_BN:
        raw = bn.move_rank(x, window=window, min_count=window, axis=0)
        return raw / window
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T < window:
        return out
    try:
        windows = _stride_windows(x, window)
        last     = windows[:, -1:, :]
        count    = (~np.isnan(windows)).sum(axis=1)
        le       = (windows <= last).sum(axis=1)
        last_nan = np.isnan(windows[:, -1, :])
        result   = np.where((count >= window) & ~last_nan, le / count, np.nan)
        out[window - 1:] = result
    except Exception:
        for i in range(window - 1, T):
            block = x[i - window + 1: i + 1]
            last  = block[-1]
            count = np.sum(~np.isnan(block), axis=0)
            le    = np.sum(block <= last, axis=0)
            out[i] = np.where(count >= window, le / count, np.nan)
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
    """
    x = _ensure_2d(np.asarray(x, dtype=float))
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T < window:
        return out
    log_nbins = np.log(n_bins) if n_bins > 1 else 1.0
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
            out[i, n_idx] = h / log_nbins
    return out


def ts_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Rolling Pearson correlation between x and y."""
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
        cov_xy = np.mean(dx * dy, axis=1)
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
    """
    x = _ensure_2d(np.asarray(x, dtype=float))
    nan_mask  = np.isnan(x)
    x_filled  = np.where(nan_mask, -np.inf, x)
    order     = np.argsort(np.argsort(x_filled, axis=1), axis=1).astype(float)
    valid_count = (~nan_mask).sum(axis=1, keepdims=True).astype(float)
    order[nan_mask] = np.nan
    denom = np.maximum(valid_count - 1, 1)
    order = order / denom
    order[nan_mask] = np.nan
    return order


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
    "ts_mean":         bn_ts_mean,
    "ts_std":          bn_ts_std,
    "ts_var":          bn_ts_var,
    "ts_sum":          bn_ts_sum,
    "ts_max":          bn_ts_max,
    "ts_min":          bn_ts_min,
    "ts_rank":         bn_ts_rank,
    "ts_decay_linear": ts_decay_linear,
    "ts_delta":        ts_delta,
    "ts_delay":        ts_delay,
    "ts_argmax":       ts_argmax,
    "ts_argmin":       ts_argmin,
    "ts_zscore":       ts_zscore,
    "ts_skew":         ts_skew,
    "ts_kurt":         ts_kurt,
    "ts_entropy":      ts_entropy,
    # Two-input ops — called as fn(x, y, window) via TimeSeriesNode._second_child
    "ts_corr":         ts_corr,
    "ts_cov":          ts_cov,
}

# Two-input TS ops requiring a second child series
_TWO_INPUT_TS_OPS = frozenset({"ts_corr", "ts_cov"})

FAST_CS_OPS = {
    "rank":            cs_rank,
    "zscore":          cs_zscore,
    "scale":           cs_scale,
    "ind_neutralize":  ind_neutralize,
    "winsorize":       cs_winsorize,
    "normalize":       cs_normalize,
}

FAST_GROUP_OPS = {
    "group_rank":       group_rank,
    "group_zscore":     group_zscore,
    "group_mean":       group_mean,
    "group_neutralize": group_neutralize,
}
