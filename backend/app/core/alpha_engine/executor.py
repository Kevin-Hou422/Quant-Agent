"""
Alpha Executor

Traverses an alpha AST and evaluates it against a dataset,
returning a signal matrix (pd.DataFrame, time × assets).

Dataset format
--------------
A dict[str, pd.DataFrame] where each key is a data field name
('close', 'open', 'high', 'low', 'volume', 'vwap', 'returns')
and each value is a DataFrame with shape (T, N):
    - index  : DatetimeIndex (time axis)
    - columns: asset tickers (asset axis)
"""

from __future__ import annotations

from typing import Dict, Optional
import pandas as pd
import numpy as np

from .ast import Node
from .operators import OPERATOR_MAP


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Dataset = Dict[str, pd.DataFrame]   # field -> (time × assets) DataFrame


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class AlphaExecutor:
    """
    Evaluates alpha AST nodes against a multi-asset dataset.

    Parameters
    ----------
    neutralize : bool
        If True, apply cross-sectional demeaning to the final signal.
    winsorize_k : float | None
        If set, winsorize the final signal at ±k standard deviations.
    """

    def __init__(
        self,
        neutralize: bool = False,
        winsorize_k: Optional[float] = None,
    ) -> None:
        self.neutralize = neutralize
        self.winsorize_k = winsorize_k

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, node: Node, dataset: Dataset) -> pd.DataFrame:
        """
        Evaluate ``node`` against ``dataset``.

        Parameters
        ----------
        node    : root Node of the alpha expression.
        dataset : dict mapping field names to (time × assets) DataFrames.

        Returns
        -------
        pd.DataFrame  (time × assets) signal matrix.
        """
        self._validate_dataset(dataset)
        signal = self._eval(node, dataset)
        signal = self._postprocess(signal)
        return signal

    # ------------------------------------------------------------------
    # Internal recursive evaluator
    # ------------------------------------------------------------------

    def _eval(self, node: Node, dataset: Dataset) -> pd.DataFrame:
        """Recursively evaluate a node."""
        op = node.op

        # --- Leaf: raw data field ---
        if op == "data":
            field = node.params.get("field", "close")
            if field not in dataset:
                raise KeyError(
                    f"Data field '{field}' not found in dataset. "
                    f"Available fields: {list(dataset.keys())}"
                )
            return dataset[field].copy()

        # --- Evaluate children ---
        children = [self._eval(child, dataset) for child in node.children]

        # --- Dispatch ---
        fn = OPERATOR_MAP.get(op)
        if fn is None:
            raise NotImplementedError(f"No implementation for operator '{op}'.")

        params = {k: v for k, v in node.params.items()}
        result = self._dispatch(fn, op, children, params)
        return result

    def _dispatch(
        self,
        fn,
        op: str,
        children: list[pd.DataFrame],
        params: dict,
    ) -> pd.DataFrame:
        """Call the operator function with the right signature."""
        from .ast import ARITHMETIC_OPS, UNARY_OPS, TIME_SERIES_OPS, CROSS_SECTIONAL_OPS

        if op in ARITHMETIC_OPS:
            # binary: (a, b)
            return fn(children[0], children[1])

        if op in UNARY_OPS:
            # unary: (x,)
            return fn(children[0])

        if op in {"ts_corr", "ts_cov"}:
            # two-child time-series
            return fn(children[0], children[1], window=params.get("window", 10))

        if op in TIME_SERIES_OPS:
            # single-child time-series
            return fn(children[0], window=params.get("window", 10))

        if op in CROSS_SECTIONAL_OPS:
            # cross-sectional; pass through extra params
            extra = {k: v for k, v in params.items() if k != "field"}
            return fn(children[0], **extra)

        # Fallback: try calling with just the children
        return fn(*children, **params)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _postprocess(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Optional neutralization and winsorization of the final signal."""
        if self.winsorize_k is not None:
            from .operators import cs_winsorize
            signal = cs_winsorize(signal, k=self.winsorize_k)

        if self.neutralize:
            from .operators import cs_demean
            signal = cs_demean(signal)

        return signal

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_dataset(dataset: Dataset) -> None:
        if not dataset:
            raise ValueError("Dataset is empty.")
        shapes = {k: df.shape for k, df in dataset.items()}
        shape_set = set(shapes.values())
        if len(shape_set) > 1:
            raise ValueError(
                f"All dataset fields must have the same shape. Got: {shapes}"
            )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def execute_alpha(
    node: Node,
    dataset: Dataset,
    neutralize: bool = False,
    winsorize_k: Optional[float] = None,
) -> pd.DataFrame:
    """
    Shortcut: evaluate an alpha expression and return the signal matrix.

    Parameters
    ----------
    node        : root Node of the alpha AST.
    dataset     : dict[field_name, DataFrame(time × assets)].
    neutralize  : subtract cross-sectional mean from the signal.
    winsorize_k : clip signal at ±k cross-sectional std deviations.

    Returns
    -------
    pd.DataFrame  (time × assets)
    """
    executor = AlphaExecutor(neutralize=neutralize, winsorize_k=winsorize_k)
    return executor.execute(node, dataset)


def batch_execute(
    nodes: list[Node],
    dataset: Dataset,
    neutralize: bool = False,
    winsorize_k: Optional[float] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Evaluate multiple alpha expressions and return a dict of signals.

    Returns
    -------
    dict mapping alpha_index (e.g. 'alpha_0') -> signal DataFrame.
    """
    executor = AlphaExecutor(neutralize=neutralize, winsorize_k=winsorize_k)
    results: Dict[str, pd.DataFrame] = {}
    for i, node in enumerate(nodes):
        try:
            results[f"alpha_{i}"] = executor.execute(node, dataset)
        except Exception as exc:
            import warnings
            warnings.warn(f"alpha_{i} failed: {exc}")
            results[f"alpha_{i}"] = pd.DataFrame()
    return results
