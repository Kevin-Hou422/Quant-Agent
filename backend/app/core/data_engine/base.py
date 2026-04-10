"""
Abstract DataProvider base class for the Data Engine.

All data providers must implement this interface so that the rest of the
system can consume market data in a uniform way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


# ---------------------------------------------------------------------------
# Return type alias
# ---------------------------------------------------------------------------

# A "RawDataset" is a dict mapping field name -> DataFrame(time × assets)
RawDataset = dict[str, pd.DataFrame]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DataProvider(ABC):
    """
    Abstract market data provider.

    Concrete subclasses must implement ``fetch`` and ``available_fields``.
    """

    # ------------------------------------------------------------------
    # Abstract interface
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
    # Optional helpers (may be overridden)
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
