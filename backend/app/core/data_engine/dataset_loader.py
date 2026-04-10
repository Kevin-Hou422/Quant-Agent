"""
Dataset Loader

High-level helper that:
  1. Pulls raw data from a DataProvider.
  2. Cleans / aligns / forward-fills the data.
  3. Returns a ready-to-use RawDataset for the alpha / backtest engines.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

import pandas as pd
import numpy as np

from .base import DataProvider, RawDataset
from .yahoo_provider import YahooFinanceProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: Dict[str, Type[DataProvider]] = {
    "yahoo": YahooFinanceProvider,
}


def register_provider(name: str, cls: Type[DataProvider]) -> None:
    """Register a custom DataProvider under ``name``."""
    _PROVIDER_REGISTRY[name.lower()] = cls


def get_provider(name: str, **kwargs) -> DataProvider:
    """Instantiate a registered provider by name."""
    key = name.lower()
    if key not in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider '{name}'. Registered: {list(_PROVIDER_REGISTRY.keys())}"
        )
    return _PROVIDER_REGISTRY[key](**kwargs)


# ---------------------------------------------------------------------------
# DatasetLoader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """
    Orchestrates data fetching, cleaning, and packaging.

    Parameters
    ----------
    provider : DataProvider
        Data source. Defaults to YahooFinanceProvider.
    ffill_limit : int
        Maximum number of consecutive NaN days to forward-fill (default 5).
    min_coverage : float
        Drop assets with less than this fraction of non-NaN close values (default 0.5).
    """

    def __init__(
        self,
        provider: Optional[DataProvider] = None,
        ffill_limit: int = 5,
        min_coverage: float = 0.5,
    ) -> None:
        self.provider = provider or YahooFinanceProvider()
        self.ffill_limit = ffill_limit
        self.min_coverage = min_coverage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        tickers: List[str],
        start: str,
        end: str,
        fields: Optional[List[str]] = None,
    ) -> RawDataset:
        """
        Fetch and clean data for ``tickers`` over [start, end].

        Parameters
        ----------
        tickers : list of ticker symbols.
        start   : start date string ('YYYY-MM-DD').
        end     : end date string ('YYYY-MM-DD').
        fields  : fields to include; None → all available.

        Returns
        -------
        RawDataset  – dict[field, DataFrame(time × assets)]
        """
        logger.info("Loading dataset: %d tickers, %s to %s", len(tickers), start, end)

        raw = self.provider.fetch(tickers=tickers, start=start, end=end, fields=fields)

        cleaned = self._clean(raw)
        logger.info("Dataset ready: fields=%s, shape=%s",
                    list(cleaned.keys()),
                    next(iter(cleaned.values())).shape if cleaned else None)
        return cleaned

    # ------------------------------------------------------------------
    # Cleaning pipeline
    # ------------------------------------------------------------------

    def _clean(self, dataset: RawDataset) -> RawDataset:
        """Apply alignment, coverage filter, and forward-fill."""
        if not dataset:
            return dataset

        # 1. Ensure all fields share the same index & columns
        dataset = self._align_shapes(dataset)

        # 2. Drop low-coverage assets
        if "close" in dataset:
            dataset = self._filter_coverage(dataset, ref_field="close")

        # 3. Forward-fill NaN values (limit to ffill_limit days)
        dataset = self._ffill(dataset)

        return dataset

    def _align_shapes(self, dataset: RawDataset) -> RawDataset:
        """Ensure all DataFrames share the same DatetimeIndex and columns."""
        ref = next(iter(dataset.values()))
        idx = ref.index
        cols = ref.columns
        for field, df in dataset.items():
            idx = idx.union(df.index)
            cols = cols.union(df.columns)
        return {
            field: df.reindex(index=idx, columns=cols)
            for field, df in dataset.items()
        }

    def _filter_coverage(self, dataset: RawDataset, ref_field: str) -> RawDataset:
        """Drop columns (assets) with insufficient non-NaN coverage in ref_field."""
        ref = dataset[ref_field]
        n_rows = len(ref)
        if n_rows == 0:
            return dataset
        coverage = ref.notna().sum() / n_rows
        keep = coverage[coverage >= self.min_coverage].index.tolist()
        dropped = set(ref.columns) - set(keep)
        if dropped:
            logger.warning("Dropping low-coverage assets: %s", sorted(dropped))
        return {field: df[keep] for field, df in dataset.items()}

    def _ffill(self, dataset: RawDataset) -> RawDataset:
        """Forward-fill NaN values up to ffill_limit consecutive days."""
        return {
            field: df.ffill(limit=self.ffill_limit)
            for field, df in dataset.items()
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def describe(self, dataset: RawDataset) -> pd.DataFrame:
        """Return a summary DataFrame with per-field coverage statistics."""
        rows = []
        for field, df in dataset.items():
            rows.append({
                "field":    field,
                "shape":    df.shape,
                "n_assets": df.shape[1],
                "n_dates":  df.shape[0],
                "nan_pct":  float(df.isna().mean().mean()) * 100,
                "start":    df.index[0] if len(df) else None,
                "end":      df.index[-1] if len(df) else None,
            })
        return pd.DataFrame(rows).set_index("field")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def load_dataset(
    tickers: List[str],
    start: str,
    end: str,
    fields: Optional[List[str]] = None,
    provider: str = "yahoo",
    **provider_kwargs,
) -> RawDataset:
    """
    One-liner to load a clean dataset.

    Parameters
    ----------
    tickers  : list of ticker symbols.
    start    : start date ('YYYY-MM-DD').
    end      : end date   ('YYYY-MM-DD').
    fields   : fields to return; None → all.
    provider : name of the registered data provider (default 'yahoo').
    **provider_kwargs : passed to the provider constructor.

    Returns
    -------
    RawDataset
    """
    data_provider = get_provider(provider, **provider_kwargs)
    loader = DatasetLoader(provider=data_provider)
    return loader.load(tickers=tickers, start=start, end=end, fields=fields)
