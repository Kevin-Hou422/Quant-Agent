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
    """Pure-NumPy fallback for rolling mean (used only when bottleneck absent)."""
    out = np.full_like(x, np.nan, dtype=float)
    # stride_tricks gives a zero-copy view: shape (T-w+1, window, N)
    T, N = x.shape
    if T < window:
        return out
    shape   = (T - window + 1, window, N)
    strides = (x.strides[0], x.strides[0], x.strides[1])
    try:
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        valid = (~np.isnan(windows)).sum(axis=1)          # (T-w+1, N)
        sums  = np.nansum(windows, axis=1)                # (T-w+1, N)
        out[window - 1:] = np.where(valid >= window, sums / valid, np.nan)
    except Exception:
        for i in range(window - 1, T):
            block = x[i - window + 1: i + 1]
            valid_counts = (~np.isnan(block)).sum(axis=0)
            sums = np.nansum(block, axis=0)
            out[i] = np.where(valid_counts >= window, sums / valid_counts, np.nan)
    return out


def _numpy_move_std(x: np.ndarray, window: int) -> np.ndarray:
    """Pure-NumPy fallback for rolling std (used only when bottleneck absent)."""
    out = np.full_like(x, np.nan, dtype=float)
    T, N = x.shape
    if T < window:
        return out
    shape   = (T - window + 1, window, N)
    strides = (x.strides[0], x.strides[0], x.strides[1])
    try:
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        valid = (~np.isnan(windows)).sum(axis=1)
        mu    = np.nanmean(windows, axis=1)
        var   = np.nanvar(windows, axis=1, ddof=1)
        out[window - 1:] = np.where(valid >= window, np.sqrt(var), np.nan)
    except Exception:
        for i in range(window - 1, T):
            block = x[i - window + 1: i + 1].astype(float)
            valid = np.sum(~np.isnan(block), axis=0)
            var = np.nanvar(block, axis=0, ddof=1)
            out[i] = np.where(valid >= window, np.sqrt(var), np.nan)
    return out


# ---------------------------------------------------------------------------
# Time-Series Operators (bottleneck preferred)
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


def bn_ts_max(x: np.ndarray, window: int) -> np.ndarray:
    x = _ensure_2d(np.asarray(x, dtype=float))
    if _HAS_BN:
        return bn.move_max(x, window=window, min_count=window, axis=0)
    # Vectorised fallback via stride_tricks
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T < window:
        return out
    shape   = (T - window + 1, window, N)
    strides = (x.strides[0], x.strides[0], x.strides[1])
    try:
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
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
    shape   = (T - window + 1, window, N)
    strides = (x.strides[0], x.strides[0], x.strides[1])
    try:
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        out[window - 1:] = np.nanmin(windows, axis=1)
    except Exception:
        for i in range(window - 1, T):
            out[i] = np.nanmin(x[i - window + 1: i + 1], axis=0)
    return out


def bn_ts_rank(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling rank (percentile in [0,1]) of the most recent value."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    if _HAS_BN:
        # move_rank returns rank in [1..window]; scale to [0,1]
        raw = bn.move_rank(x, window=window, min_count=window, axis=0)
        return raw / window
    # Vectorised NumPy fallback via stride_tricks
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T < window:
        return out
    shape   = (T - window + 1, window, N)
    strides = (x.strides[0], x.strides[0], x.strides[1])
    try:
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        last    = windows[:, -1:, :]          # (T-w+1, 1, N) — most recent
        count   = (~np.isnan(windows)).sum(axis=1)   # (T-w+1, N)
        le      = (windows <= last).sum(axis=1)       # (T-w+1, N) — including NaN-safe?
        # Mask windows that have NaN in the last position
        last_nan = np.isnan(windows[:, -1, :])        # (T-w+1, N)
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
    """
    Linearly-weighted moving average.
    Weights: [1, 2, …, window] / sum, most-recent = highest weight.
    NaN for fewer than `window` rows.
    """
    x = _ensure_2d(np.asarray(x, dtype=float))
    T, N = x.shape
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()   # shape (window,)

    out = np.full((T, N), np.nan)
    if T < window:
        return out

    # Use stride_tricks for zero-copy sliding windows
    shape   = (T - window + 1, window, N)
    strides = (x.strides[0], x.strides[0], x.strides[1])
    try:
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        # windows: (T-w+1, window, N)
        has_nan = np.any(np.isnan(windows), axis=1)   # (T-w+1, N)
        result  = np.einsum("twn,w->tn", windows, weights)  # vectorized dot
        result[has_nan] = np.nan
        out[window - 1:] = result
    except Exception:
        # Safe fallback (only triggered by stride_tricks memory layout error)
        for i in range(window - 1, T):
            block = x[i - window + 1: i + 1]
            if np.any(np.isnan(block)):
                continue
            out[i] = weights @ block  # (N,)
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
# Cross-Sectional Operators  (fully vectorised — zero Python-level T loops)
# ---------------------------------------------------------------------------

def cs_rank(x: np.ndarray) -> np.ndarray:
    """
    Cross-sectional percentile rank [0, 1] per row.

    Fully vectorised via double argsort — no Python loop over T.
    NaN assets are excluded from ranking but remain NaN in output.

    Ties are resolved by average rank (matching scipy.stats.rankdata
    'average' method behaviour).
    """
    x = _ensure_2d(np.asarray(x, dtype=float))
    nan_mask = np.isnan(x)

    # Replace NaN with -inf so argsort puts them at the end (lowest rank)
    x_filled = np.where(nan_mask, -np.inf, x)

    # Double argsort gives ordinal ranks (0-based) per row — O(N log N) vectorised
    order = np.argsort(np.argsort(x_filled, axis=1), axis=1).astype(float)

    # How many valid (non-NaN) assets per row
    valid_count = (~nan_mask).sum(axis=1, keepdims=True).astype(float)  # (T, 1)

    # Positions occupied by NaN values were assigned ranks at the TOP of argsort
    # (since -inf < everything else) — we must zero them out before normalising.
    order[nan_mask] = np.nan

    # Normalise to [0, 1]: rank 0 → 0.0, rank (valid-1) → 1.0
    # Avoid div-by-zero when an entire row is NaN
    denom = np.maximum(valid_count - 1, 1)
    order = order / denom
    order[nan_mask] = np.nan
    return order


def cs_zscore(x: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score per row (NaN-aware, fully vectorised)."""
    x = _ensure_2d(np.asarray(x, dtype=float))
    mu    = np.nanmean(x, axis=1, keepdims=True)
    sigma = np.nanstd(x, axis=1, ddof=1, keepdims=True)
    # Where sigma == 0, return 0 (not NaN) to avoid division issues
    safe_sigma = np.where(sigma == 0, np.nan, sigma)
    return (x - mu) / safe_sigma


def cs_scale(x: np.ndarray) -> np.ndarray:
    """
    L1-norm scaling per row: x / sum(|x|).
    NaN assets are excluded from the norm denominator.
    Fully vectorised.
    """
    x = _ensure_2d(np.asarray(x, dtype=float))
    l1 = np.nansum(np.abs(x), axis=1, keepdims=True)
    safe_l1 = np.where(l1 == 0, np.nan, l1)
    return x / safe_l1


def ind_neutralize(x: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Subtract the group mean from each asset per row (industry neutralization).

    Vectorisation strategy: outer loop over unique group IDs (G groups,
    typically < 30 industries). Each iteration applies nanmean across
    the full time axis in one NumPy call — no Python loop over T.

    Parameters
    ----------
    x      : (T, N) array.
    groups : (N,) integer array mapping each asset to a group index.
             If None, falls back to cs_zscore.
    """
    x = _ensure_2d(np.asarray(x, dtype=float))
    if groups is None:
        return cs_zscore(x)

    groups = np.asarray(groups, dtype=int)
    out = x.copy()

    for g in np.unique(groups):              # outer: G iterations (G << T)
        mask  = (groups == g)                # (N,) bool — static per group
        block = x[:, mask]                   # (T, Ng) — view, no copy
        gmean = np.nanmean(block, axis=1, keepdims=True)  # (T, 1) fully vectorised
        out[:, mask] = block - gmean         # broadcast (T, Ng)

    return out


# ---------------------------------------------------------------------------
# Advanced Operators
# ---------------------------------------------------------------------------

def signed_power(x: np.ndarray, p: float | np.ndarray) -> np.ndarray:
    """sign(x) * |x|^p"""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.abs(x) ** p


def op_if_else(
    cond: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """np.where(cond, x, y) — vectorized conditional."""
    return np.where(cond.astype(bool), x, y)


# ---------------------------------------------------------------------------
# Dispatch tables (used by typed_nodes)
# ---------------------------------------------------------------------------

FAST_TS_OPS = {
    "ts_mean":         bn_ts_mean,
    "ts_std":          bn_ts_std,
    "ts_max":          bn_ts_max,
    "ts_min":          bn_ts_min,
    "ts_rank":         bn_ts_rank,
    "ts_decay_linear": ts_decay_linear,
    "ts_delta":        ts_delta,
    "ts_delay":        ts_delay,
}

FAST_CS_OPS = {
    "rank":            cs_rank,
    "zscore":          cs_zscore,
    "scale":           cs_scale,
    "ind_neutralize":  ind_neutralize,
}
