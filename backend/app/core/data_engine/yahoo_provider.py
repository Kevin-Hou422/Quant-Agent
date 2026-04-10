"""
Yahoo Finance DataProvider using ``yfinance``.

Downloads OHLCV data for a multi-asset universe and constructs the
standard RawDataset format used by the alpha / backtest engines.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import DataProvider, RawDataset

logger = logging.getLogger(__name__)

_SUPPORTED_FIELDS = ["open", "high", "low", "close", "volume", "vwap", "returns"]


class YahooFinanceProvider(DataProvider):
    """
    Fetches OHLCV data from Yahoo Finance via the ``yfinance`` library.

    Extra derived fields
    --------------------
    vwap    : approximated as (high + low + close) / 3
    returns : daily log return of the adjusted close

    Parameters
    ----------
    auto_adjust : bool
        If True (default), use split/dividend-adjusted prices.
    progress    : bool
        Show yfinance download progress bar (default False).
    """

    def __init__(
        self,
        auto_adjust: bool = True,
        progress: bool = False,
    ) -> None:
        self.auto_adjust = auto_adjust
        self.progress = progress

    # ------------------------------------------------------------------
    # DataProvider interface
    # ------------------------------------------------------------------

    def available_fields(self) -> List[str]:
        return list(_SUPPORTED_FIELDS)

    def fetch(
        self,
        tickers: List[str],
        start: str,
        end: str,
        fields: Optional[List[str]] = None,
    ) -> RawDataset:
        """
        Download data for ``tickers`` from Yahoo Finance.

        Returns
        -------
        RawDataset  – dict[field, DataFrame(time × assets)]
        """
        import yfinance as yf

        if fields is None:
            fields = _SUPPORTED_FIELDS
        else:
            self.validate_fields(fields)

        tickers = [t.upper() for t in tickers]
        logger.info("Downloading %d tickers from Yahoo Finance [%s → %s]", len(tickers), start, end)

        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=self.auto_adjust,
            progress=self.progress,
            group_by="column",
        )

        if raw.empty:
            raise ValueError("yfinance returned empty data. Check tickers and date range.")

        dataset = self._parse(raw, tickers, fields)
        logger.info("Fetched dataset: shape=%s, fields=%s", next(iter(dataset.values())).shape, list(dataset.keys()))
        return dataset

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse(
        self,
        raw: pd.DataFrame,
        tickers: List[str],
        fields: List[str],
    ) -> RawDataset:
        """
        Convert a yfinance multi-level DataFrame into RawDataset.

        yfinance with group_by='column' returns MultiIndex columns:
            (field, ticker)   e.g. ('Close', 'AAPL')
        """
        # Normalize column levels
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_tuples(
                [(lvl0.lower(), lvl1.upper()) for lvl0, lvl1 in raw.columns]
            )
        else:
            # Single ticker: add a fake MultiIndex level
            single_ticker = tickers[0] if tickers else "UNKNOWN"
            raw.columns = pd.MultiIndex.from_tuples(
                [(col.lower(), single_ticker) for col in raw.columns]
            )

        # Ensure DatetimeIndex
        raw.index = pd.to_datetime(raw.index)

        dataset: RawDataset = {}

        def _extract(field_key: str) -> pd.DataFrame:
            """Extract a field across all tickers into a (time × assets) DataFrame."""
            try:
                df = raw[field_key]
            except KeyError:
                return pd.DataFrame(index=raw.index, columns=tickers, dtype=float)
            if isinstance(df, pd.Series):
                df = df.to_frame(name=tickers[0])
            # Reindex to requested tickers (NaN for missing)
            df = df.reindex(columns=tickers)
            return df.astype(float)

        # Core OHLCV
        ohlcv_map = {
            "open":   "open",
            "high":   "high",
            "low":    "low",
            "close":  "close",
            "volume": "volume",
        }
        raw_frames: Dict[str, pd.DataFrame] = {}
        for field, yf_key in ohlcv_map.items():
            if field in fields or "vwap" in fields or "returns" in fields:
                raw_frames[field] = _extract(yf_key)

        # Derived: vwap ≈ (H + L + C) / 3
        if "vwap" in fields:
            h = raw_frames.get("high", _extract("high"))
            l = raw_frames.get("low",  _extract("low"))
            c = raw_frames.get("close", _extract("close"))
            raw_frames["vwap"] = (h + l + c) / 3.0

        # Derived: log returns
        if "returns" in fields:
            c = raw_frames.get("close", _extract("close"))
            raw_frames["returns"] = np.log(c / c.shift(1))

        # Package only requested fields
        for f in fields:
            if f in raw_frames:
                dataset[f] = raw_frames[f]

        return dataset
