"""
multi_dataset_backtester.py — Run IS+OOS backtest across multiple universes.

Eliminates single-dataset overfitting: an alpha must prove positive Sharpe on
ALL (or a configurable subset of) datasets to be considered robust.

Usage::

    backtester = MultiDatasetBacktester(
        is_split  = 0.7,
        aggregation = "mean",    # or "min"
    )
    result = backtester.run(
        dsl = "rank(ts_delta(log(close), 5))",
        datasets = {
            "us_equity": us_equity_dataset.data,
            "crypto":    crypto_dataset.data,
        }
    )
    print(result.aggregated_sharpe)   # mean OOS sharpe across datasets
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DatasetBacktestResult:
    """Per-dataset backtest output."""
    dataset_name:    str
    sharpe_is:       float
    sharpe_oos:      float
    max_drawdown:    float
    ann_turnover:    float
    ann_return:      float
    mean_ic:         float
    overfitting_score: float
    error:           Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "dataset":           self.dataset_name,
            "sharpe_is":         round(self.sharpe_is,         4),
            "sharpe_oos":        round(self.sharpe_oos,        4),
            "max_drawdown":      round(self.max_drawdown,      4),
            "ann_turnover":      round(self.ann_turnover,      4),
            "ann_return":        round(self.ann_return,        4),
            "mean_ic":           round(self.mean_ic,           4),
            "overfitting_score": round(self.overfitting_score, 4),
            "error":             self.error,
        }


@dataclass
class MultiDatasetResult:
    """
    Aggregated result across all datasets.

    Attributes
    ----------
    per_dataset       : Detailed result for each dataset
    aggregated_sharpe : mean or min OOS Sharpe depending on aggregation_mode
    aggregation_mode  : "mean" or "min"
    datasets_passed   : Number of datasets with sharpe_oos > 0
    """
    per_dataset:       Dict[str, DatasetBacktestResult]
    aggregated_sharpe: float
    aggregation_mode:  str
    datasets_passed:   int
    datasets_total:    int
    errors:            List[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.datasets_total == 0:
            return 0.0
        return self.datasets_passed / self.datasets_total

    def to_dict(self) -> dict:
        return {
            "aggregated_sharpe": round(self.aggregated_sharpe, 4),
            "aggregation_mode":  self.aggregation_mode,
            "datasets_passed":   self.datasets_passed,
            "datasets_total":    self.datasets_total,
            "pass_rate":         round(self.pass_rate, 4),
            "per_dataset":       {k: v.to_dict() for k, v in self.per_dataset.items()},
            "errors":            self.errors,
        }

    def summary(self) -> str:
        lines = [
            f"MultiDataset Backtest ({self.aggregation_mode} aggregation)",
            f"  Aggregated Sharpe : {self.aggregated_sharpe:.4f}",
            f"  Pass rate         : {self.datasets_passed}/{self.datasets_total}",
        ]
        for name, r in self.per_dataset.items():
            if r.error:
                lines.append(f"  [{name}] ERROR: {r.error}")
            else:
                lines.append(
                    f"  [{name}] IS={r.sharpe_is:.3f}  OOS={r.sharpe_oos:.3f}"
                    f"  DD={r.max_drawdown:.3f}  TO={r.ann_turnover:.3f}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MultiDatasetBacktester
# ---------------------------------------------------------------------------

class MultiDatasetBacktester:
    """
    Run IS+OOS backtests across multiple market datasets and return an
    aggregated generalization score.

    Parameters
    ----------
    config        : SimulationConfig — signal processing parameters
    cost_params   : CostParams — transaction cost model (optional)
    aggregation   : "mean" (lenient) or "min" (strict — all datasets must pass)
    is_split      : Fraction of data used for IS training (default 0.7)
    """

    def __init__(
        self,
        config        = None,
        cost_params   = None,
        aggregation:  str   = "mean",
        is_split:     float = 0.7,
    ) -> None:
        self._config      = config
        self._cost_params = cost_params
        self._aggregation = aggregation
        self._is_split    = is_split

        if aggregation not in ("mean", "min"):
            raise ValueError(f"aggregation must be 'mean' or 'min', got '{aggregation}'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        dsl:      str,
        datasets: Dict[str, Dict[str, Any]],
    ) -> MultiDatasetResult:
        """
        Run IS+OOS backtest on ``dsl`` for each dataset in ``datasets``.

        Parameters
        ----------
        dsl      : Alpha DSL expression string
        datasets : dict mapping dataset_name → raw_data_dict
                   (each value is a dict[field → pd.DataFrame(T×N)])

        Returns
        -------
        MultiDatasetResult
        """
        if not datasets:
            raise ValueError("datasets dict is empty.")

        per_dataset: Dict[str, DatasetBacktestResult] = {}
        errors: List[str] = []

        for name, raw_data in datasets.items():
            logger.info("MultiDataset: evaluating '%s' on dataset '%s'", dsl[:60], name)
            result = self._run_one(name, dsl, raw_data)
            per_dataset[name] = result
            if result.error:
                errors.append(f"[{name}] {result.error}")

        # Aggregate sharpe across successful datasets
        oos_sharpes = [
            r.sharpe_oos for r in per_dataset.values()
            if r.error is None and not np.isnan(r.sharpe_oos)
        ]

        if not oos_sharpes:
            agg_sharpe = 0.0
        elif self._aggregation == "min":
            agg_sharpe = float(min(oos_sharpes))
        else:
            agg_sharpe = float(np.mean(oos_sharpes))

        datasets_passed = sum(1 for r in per_dataset.values() if r.sharpe_oos > 0)

        return MultiDatasetResult(
            per_dataset       = per_dataset,
            aggregated_sharpe = agg_sharpe,
            aggregation_mode  = self._aggregation,
            datasets_passed   = datasets_passed,
            datasets_total    = len(datasets),
            errors            = errors,
        )

    def run_with_datasets_obj(
        self,
        dsl:      str,
        datasets: list,  # List[Dataset]
    ) -> MultiDatasetResult:
        """Convenience wrapper accepting Dataset objects from multi_dataset.py."""
        raw_dict = {ds.name: ds.data for ds in datasets}
        return self.run(dsl, raw_dict)

    # ------------------------------------------------------------------
    # Single-dataset evaluation
    # ------------------------------------------------------------------

    def _run_one(
        self,
        name:     str,
        dsl:      str,
        raw_data: Dict[str, Any],
    ) -> DatasetBacktestResult:
        """Partition raw_data IS/OOS and run RealisticBacktester."""
        try:
            from app.core.backtest_engine.realistic_backtester import (
                RealisticBacktester,
            )
            from app.core.alpha_engine.signal_processor import SimulationConfig

            config = self._default_config()
            is_data, oos_data = _split_dataset(raw_data, self._is_split)

            bt     = RealisticBacktester(config=config, cost_params=self._cost_params)
            result = bt.run(dsl, is_data, oos_dataset=oos_data)

            def _f(v: Any) -> float:
                try:
                    fv = float(v)
                    return fv if not np.isnan(fv) else 0.0
                except (TypeError, ValueError):
                    return 0.0

            is_r  = result.is_report
            oos_r = result.oos_report

            sharpe_is  = _f(is_r.sharpe_ratio)
            sharpe_oos = _f(oos_r.sharpe_ratio)  if oos_r else 0.0
            max_dd     = _f(oos_r.max_drawdown)   if oos_r else 0.0
            turnover   = _f(is_r.ann_turnover)
            ann_ret    = _f(is_r.annualized_return)
            mean_ic    = _f(is_r.mean_ic)

            overfit = 0.0
            if abs(sharpe_is) > 1e-9 and oos_r:
                overfit = float(np.clip((sharpe_is - sharpe_oos) / abs(sharpe_is), 0.0, 1.0))

            return DatasetBacktestResult(
                dataset_name      = name,
                sharpe_is         = sharpe_is,
                sharpe_oos        = sharpe_oos,
                max_drawdown      = max_dd,
                ann_turnover      = turnover,
                ann_return        = ann_ret,
                mean_ic           = mean_ic,
                overfitting_score = overfit,
            )

        except Exception as exc:
            logger.warning("MultiDataset eval failed for dataset '%s': %s", name, exc)
            return DatasetBacktestResult(
                dataset_name      = name,
                sharpe_is         = 0.0,
                sharpe_oos        = 0.0,
                max_drawdown      = 0.0,
                ann_turnover      = 0.0,
                ann_return        = 0.0,
                mean_ic           = 0.0,
                overfitting_score = 0.0,
                error             = str(exc),
            )

    def _default_config(self):
        if self._config is not None:
            return self._config
        from app.core.alpha_engine.signal_processor import SimulationConfig
        return SimulationConfig(
            delay            = 1,
            decay_window     = 0,
            truncation_min_q = 0.05,
            truncation_max_q = 0.95,
            portfolio_mode   = "long_short",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_dataset(
    raw_data: Dict[str, Any],
    is_split: float,
) -> tuple:
    """
    Temporally split raw_data dict into IS and OOS portions.

    Returns
    -------
    (is_data, oos_data) — both in dict[field → DataFrame] format
    """
    import pandas as pd

    if not raw_data:
        return {}, {}

    # Get shared index from first field
    first_df = next(iter(raw_data.values()))
    if not isinstance(first_df, pd.DataFrame) or len(first_df) == 0:
        return raw_data, {}

    n = len(first_df)
    split_idx = max(1, int(n * is_split))

    is_data  = {f: df.iloc[:split_idx]  for f, df in raw_data.items()}
    oos_data = {f: df.iloc[split_idx:]  for f, df in raw_data.items()}
    return is_data, oos_data


def compute_multi_dataset_fitness(
    result: MultiDatasetResult,
    turnover: float = 0.0,
    max_drawdown: float = 0.0,
    sharpe_is: float = 0.0,
) -> float:
    """
    Compute GP fitness from MultiDatasetResult using the PROMPT-2 formula
    but replacing sharpe_oos with the aggregated cross-dataset Sharpe.

    fitness = aggregated_sharpe
            - 0.2 * turnover
            - 0.3 * abs(max_drawdown)
            - 0.5 * max(0, sharpe_is - aggregated_sharpe)
    """
    from app.core.gp_engine.fitness import compute_fitness
    return compute_fitness(
        sharpe_is    = sharpe_is,
        sharpe_oos   = result.aggregated_sharpe,
        turnover     = turnover,
        max_drawdown = max_drawdown,
    )
