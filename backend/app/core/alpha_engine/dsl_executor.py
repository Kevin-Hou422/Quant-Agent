"""
Memoized Alpha DSL Executor.

Converts a dict[str, pd.DataFrame] OHLCV dataset into aligned NumPy arrays,
evaluates a typed AST (or DSL string), and returns a pd.DataFrame signal.

Sub-expression memoization ensures that identical sub-trees (same repr)
are computed at most once per run() call.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .typed_nodes import Node, DataNode, Cache, Dataset
from .validator import AlphaValidator, ValidationError
from .parser import Parser, ParseError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal dataset conversion
# ---------------------------------------------------------------------------

def _align_dataset(
    raw: Dict[str, pd.DataFrame],
) -> tuple[pd.DatetimeIndex, pd.Index, Dict[str, pd.DataFrame]]:
    """
    Align all DataFrames in ``raw`` to a shared (DatetimeIndex × asset_columns).

    Returns
    -------
    (shared_index, shared_columns, aligned_dict)
    """
    if not raw:
        raise ValueError("Dataset is empty.")

    # Build union of all indices and columns
    idx  = None
    cols = None
    for df in raw.values():
        df_idx  = pd.DatetimeIndex(df.index)
        df_cols = df.columns
        idx  = df_idx  if idx  is None else idx.union(df_idx)
        cols = df_cols if cols is None else cols.union(df_cols)

    aligned = {
        field: df.reindex(index=idx, columns=cols)
        for field, df in raw.items()
    }
    return idx, cols, aligned


def _to_arrays(aligned: Dict[str, pd.DataFrame]) -> Dataset:
    """Convert aligned DataFrames to float64 NumPy arrays."""
    return {field: df.to_numpy(dtype=float) for field, df in aligned.items()}


def _add_derived(
    aligned: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Auto-generate derived fields if not already present."""
    if "returns" not in aligned and "close" in aligned:
        close = aligned["close"]
        aligned["returns"] = np.log(close / close.shift(1))
    if "vwap" not in aligned and all(f in aligned for f in ("high", "low", "close")):
        aligned["vwap"] = (aligned["high"] + aligned["low"] + aligned["close"]) / 3.0
    return aligned


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """
    High-performance, memoized executor for Alpha DSL expressions.

    Parameters
    ----------
    validate : bool
        If True (default), run AlphaValidator before execution.
    neutralize : bool
        If True, apply cross-sectional demeaning to the final signal.
    winsorize_k : float | None
        If set, winsorize the final signal at ±k cross-sectional std dev.

    Usage
    -----
    ::

        executor = Executor()
        signal   = executor.run_expr(
            "rank(ts_delta(log(close),5))/ts_std(close,20)",
            dataset,          # dict[str, pd.DataFrame]
        )
    """

    def __init__(
        self,
        validate: bool = True,
        neutralize: bool = False,
        winsorize_k: Optional[float] = None,
    ) -> None:
        self.validate     = validate
        self.neutralize   = neutralize
        self.winsorize_k  = winsorize_k
        self._validator   = AlphaValidator() if validate else None
        self._parser      = Parser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_expr(
        self,
        expr: str,
        dataset: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Parse ``expr``, validate, and execute against ``dataset``.

        Returns
        -------
        pd.DataFrame (T × N) signal matrix.
        """
        node = self._parser.parse(expr)
        return self.run(node, dataset)

    def run(
        self,
        node: Node,
        dataset: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Evaluate a pre-built typed AST ``node`` against ``dataset``.

        Returns
        -------
        pd.DataFrame (T × N) signal matrix.
        """
        # 1. Validate
        if self._validator is not None:
            self._validator.validate(node)

        # 2. Align + derive fields
        idx, cols, aligned = _align_dataset(dataset)
        aligned = _add_derived(aligned)

        # 3. Convert to NumPy arrays
        arrays: Dataset = _to_arrays(aligned)

        # 4. Execute with fresh per-call cache (memoization scope = one run)
        cache: Cache = {}
        result = node.evaluate(arrays, cache)

        logger.debug(
            "Executed '%s' — cache hits: %d sub-expressions memoized.",
            repr(node)[:80],
            len(cache),
        )

        # 5. Broadcast scalar result
        T, N = len(idx), len(cols)
        if result.ndim == 0:
            result = np.full((T, N), float(result))
        elif result.ndim == 1 and result.shape[0] == T:
            result = np.tile(result[:, np.newaxis], (1, N))

        # 6. Post-processing
        result = self._postprocess(result)

        return pd.DataFrame(result, index=idx, columns=cols)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _postprocess(self, x: np.ndarray) -> np.ndarray:
        if self.winsorize_k is not None:
            mu    = np.nanmean(x, axis=1, keepdims=True)
            sigma = np.nanstd(x, axis=1, ddof=1, keepdims=True)
            lo = mu - self.winsorize_k * sigma
            hi = mu + self.winsorize_k * sigma
            x = np.clip(x, lo, hi)

        if self.neutralize:
            mu = np.nanmean(x, axis=1, keepdims=True)
            x  = x - mu

        return x

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_cache_keys(
        self,
        node: Node,
        dataset: Dict[str, pd.DataFrame],
    ) -> list[str]:
        """
        Dry-run: return the list of cache keys that would be populated
        during a real execution (useful for debugging memoization).
        """
        idx, cols, aligned = _align_dataset(dataset)
        aligned = _add_derived(aligned)
        arrays  = _to_arrays(aligned)
        cache: Cache = {}
        node.evaluate(arrays, cache)
        return list(cache.keys())
