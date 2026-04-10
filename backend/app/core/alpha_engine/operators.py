"""
Vectorized operator implementations.

Each function receives one or more ``pd.DataFrame`` objects (time × assets)
and returns a ``pd.DataFrame`` of the same shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align(*dfs: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Reindex all DataFrames to the union of their indices/columns."""
    if len(dfs) == 1:
        return dfs
    idx = dfs[0].index
    cols = dfs[0].columns
    for df in dfs[1:]:
        idx = idx.union(df.index)
        cols = cols.union(df.columns)
    return tuple(df.reindex(index=idx, columns=cols) for df in dfs)


def _validate(df: pd.DataFrame, name: str = "input") -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame for '{name}', got {type(df)}")


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------

def op_add(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    a, b = _align(a, b)
    return a.add(b)


def op_sub(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    a, b = _align(a, b)
    return a.sub(b)


def op_mul(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    a, b = _align(a, b)
    return a.mul(b)


def op_div(a: pd.DataFrame, b: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """Safe division — replace near-zero denominators with NaN."""
    a, b = _align(a, b)
    safe_b = b.where(b.abs() > eps, other=np.nan)
    return a.div(safe_b)


# ---------------------------------------------------------------------------
# Unary operators
# ---------------------------------------------------------------------------

def op_log(x: pd.DataFrame) -> pd.DataFrame:
    """Natural log. Values <= 0 become NaN."""
    return np.log(x.where(x > 0))


def op_abs(x: pd.DataFrame) -> pd.DataFrame:
    return x.abs()


def op_neg(x: pd.DataFrame) -> pd.DataFrame:
    return -x


def op_sqrt(x: pd.DataFrame) -> pd.DataFrame:
    return np.sqrt(x.where(x >= 0))


def op_sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)


# ---------------------------------------------------------------------------
# Time-series operators
# ---------------------------------------------------------------------------

def ts_mean(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return x.rolling(window=window, min_periods=max(1, window // 2)).mean()


def ts_std(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return x.rolling(window=window, min_periods=max(2, window // 2)).std()


def ts_delta(x: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """x - lag(x, window)"""
    return x.diff(window)


def ts_delay(x: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """Lag operator."""
    return x.shift(window)


def ts_max(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return x.rolling(window=window, min_periods=max(1, window // 2)).max()


def ts_min(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return x.rolling(window=window, min_periods=max(1, window // 2)).min()


def ts_decay_linear(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Linearly-weighted moving average.
    Weights: 1, 2, ..., window  (most recent has highest weight).
    """
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()

    def _weighted(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        return float(np.dot(arr, weights))

    return x.rolling(window=window, min_periods=window).apply(_weighted, raw=True)


def ts_rank(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Rolling rank of the most-recent value within the past ``window`` observations,
    scaled to [0, 1].
    """
    def _rank_last(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        last = valid[-1]
        return float((valid <= last).sum()) / len(valid)

    return x.rolling(window=window, min_periods=max(1, window // 2)).apply(
        _rank_last, raw=True
    )


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling column-wise Pearson correlation between x and y."""
    x, y = _align(x, y)
    result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for col in x.columns:
        result[col] = x[col].rolling(window=window, min_periods=max(2, window // 2)).corr(y[col])
    return result


def ts_cov(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling column-wise covariance."""
    x, y = _align(x, y)
    result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for col in x.columns:
        result[col] = x[col].rolling(window=window, min_periods=max(2, window // 2)).cov(y[col])
    return result


# ---------------------------------------------------------------------------
# Cross-sectional operators
# ---------------------------------------------------------------------------

def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank in [0, 1] per row."""
    return x.rank(axis=1, pct=True, na_option="keep")


def cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score."""
    mu = x.mean(axis=1)
    sigma = x.std(axis=1).replace(0, np.nan)
    return x.sub(mu, axis=0).div(sigma, axis=0)


def cs_demean(x: pd.DataFrame) -> pd.DataFrame:
    """Subtract cross-sectional mean per row."""
    return x.sub(x.mean(axis=1), axis=0)


def cs_group_rank(x: pd.DataFrame, groups: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Rank within groups.  If ``groups`` is None, falls back to cs_rank.
    ``groups`` should be a Series mapping column name -> group label.
    """
    if groups is None:
        return cs_rank(x)
    result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for grp in groups.unique():
        cols = groups[groups == grp].index.intersection(x.columns).tolist()
        if cols:
            result[cols] = x[cols].rank(axis=1, pct=True, na_option="keep")
    return result


def cs_group_zscore(x: pd.DataFrame, groups: Optional[pd.Series] = None) -> pd.DataFrame:
    """Z-score within groups.  If ``groups`` is None, falls back to cs_zscore."""
    if groups is None:
        return cs_zscore(x)
    result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for grp in groups.unique():
        cols = groups[groups == grp].index.intersection(x.columns).tolist()
        if cols:
            sub = x[cols]
            mu = sub.mean(axis=1)
            sigma = sub.std(axis=1).replace(0, np.nan)
            result[cols] = sub.sub(mu, axis=0).div(sigma, axis=0)
    return result


def cs_winsorize(x: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """
    Clip each row to [mu - k*sigma, mu + k*sigma] cross-sectionally.
    """
    mu = x.mean(axis=1)
    sigma = x.std(axis=1)
    lower = mu - k * sigma
    upper = mu + k * sigma
    return x.clip(lower=lower, upper=upper, axis=0)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

OPERATOR_MAP = {
    # arithmetic
    "add":  op_add,
    "sub":  op_sub,
    "mul":  op_mul,
    "div":  op_div,
    # unary
    "log":  op_log,
    "abs":  op_abs,
    "neg":  op_neg,
    "sqrt": op_sqrt,
    "sign": op_sign,
    # time-series
    "ts_mean":         ts_mean,
    "ts_std":          ts_std,
    "ts_delta":        ts_delta,
    "ts_delay":        ts_delay,
    "ts_max":          ts_max,
    "ts_min":          ts_min,
    "ts_decay_linear": ts_decay_linear,
    "ts_rank":         ts_rank,
    "ts_corr":         ts_corr,
    "ts_cov":          ts_cov,
    # cross-sectional
    "rank":         cs_rank,
    "zscore":       cs_zscore,
    "demean":       cs_demean,
    "group_rank":   cs_group_rank,
    "group_zscore": cs_group_zscore,
    "winsorize":    cs_winsorize,
}
