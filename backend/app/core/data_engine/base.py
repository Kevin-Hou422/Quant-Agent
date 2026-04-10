"""
Abstract DataProvider base class for the Data Engine.

All data providers must implement this interface so that the rest of the
system can consume market data in a uniform way.

v2 扩展（向后兼容）：
- fetch_panel()  返回标准 long-format DataFrame
- metadata()     返回 provider 元信息
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Return type aliases
# ---------------------------------------------------------------------------

# A "RawDataset" is a dict mapping field name -> DataFrame(time × assets)
RawDataset = dict[str, pd.DataFrame]

# Long-format panel DataFrame: columns include timestamp, ticker, + OHLCV fields
PanelFrame = pd.DataFrame


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DataProvider(ABC):
    """
    Abstract market data provider.

    Concrete subclasses must implement ``fetch`` and ``available_fields``.
    ``fetch_panel`` and ``metadata`` have default implementations and are
    optional to override.
    """

    # ------------------------------------------------------------------
    # Abstract interface (must implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch(
        self,
        tickers: List[str],
        start: str,
        end: str,
        fields: Optional[List[str]] = None,
    ) -> RawDataset:
        """
        Download / load market data for ``tickers`` over [start, end].

        Parameters
        ----------
        tickers : list of ticker symbols (e.g. ['AAPL', 'MSFT']).
        start   : start date string, e.g. '2020-01-01'.
        end     : end date string,   e.g. '2023-12-31'.
        fields  : list of fields to return; if None, return all available.

        Returns
        -------
        RawDataset
            dict mapping field name -> pd.DataFrame(time × assets).
            DataFrame index is a DatetimeIndex; columns are ticker symbols.
        """
        ...

    @abstractmethod
    def available_fields(self) -> List[str]:
        """Return the list of field names this provider supports."""
        ...

    # ------------------------------------------------------------------
    # Extended interface (default implementations, safe to override)
    # ------------------------------------------------------------------

    def fetch_panel(
        self,
        tickers: List[str],
        start: str,
        end: str,
        fields: Optional[List[str]] = None,
    ) -> PanelFrame:
        """
        返回 long-format DataFrame，列包含 [timestamp, ticker, ...fields]。

        默认实现：调用 fetch() 并将 wide-format 转为 long-format。
        子类可覆盖此方法直接返回 long-format（更高效）。

        Returns
        -------
        pd.DataFrame  长表，每行 = 一个 ticker 在一个时间点的全量字段。
        """
        from .schema import wide_to_long, SchemaEnforcer

        t0 = time.monotonic()
        raw = self.fetch(tickers=tickers, start=start, end=end, fields=fields)
        if not raw:
            return pd.DataFrame()

        long_df = wide_to_long(raw)
        enforcer = SchemaEnforcer(allow_extra=True)
        panel = enforcer.enforce(long_df)

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._last_latency_ms = elapsed_ms
        return panel

    def metadata(self) -> Dict:
        """
        返回 provider 的元信息字典。
        子类可覆盖以提供实际值。

        Returns
        -------
        dict  包含 name, latency_ms, rate_limit, available_fields 等键
        """
        return {
            "name":             self.__class__.__name__,
            "latency_ms":       getattr(self, "_last_latency_ms", None),
            "rate_limit":       None,
            "available_fields": self.available_fields(),
        }

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def validate_fields(self, fields: List[str]) -> None:
        """Raise ValueError for any field not supported by this provider."""
        supported = set(self.available_fields())
        unknown = set(fields) - supported
        if unknown:
            raise ValueError(
                f"Unknown fields {unknown}. Supported: {sorted(supported)}"
            )

    def universe_size(self, tickers: List[str]) -> int:
        return len(tickers)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fields={self.available_fields()})"
