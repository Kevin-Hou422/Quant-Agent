"""
multi_dataset.py — Dataset abstraction and registry for multi-universe backtesting.

Provides:
    Dataset          — immutable container (name, frequency, universe, data)
    DatasetRegistry  — loads and caches datasets by name
    load_dataset()   — public convenience API

Supported dataset names:
    "us_equity"  — large-cap US equities via yfinance
    "china_a"    — CSI 300 A-share subset via yfinance (.SS/.SZ)
    "crypto"     — top crypto assets via yfinance (-USD) or ccxt
    "etf"        — sector/factor ETFs via yfinance

All datasets output exactly 7 standard fields:
    open, high, low, close, volume, vwap, returns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STANDARD_FIELDS: List[str] = ["open", "high", "low", "close", "volume", "vwap", "returns"]

# ---------------------------------------------------------------------------
# Default universes
# ---------------------------------------------------------------------------

US_EQUITY_UNIVERSE: List[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    "UNH",  "JNJ",  "JPM",  "V",    "PG",   "HD",   "MA",
    "XOM",  "BAC",  "PFE",  "ABBV", "KO",   "AVGO", "PEP",
    "COST", "TMO",  "MRK",  "WMT",  "CSCO", "MCD",  "DIS",
    "ACN",  "ABT",  "NFLX", "ADBE", "VZ",   "CRM",  "NKE",
    "INTC", "T",    "PM",   "ORCL", "CVX",  "LLY",  "AMGN",
    "MDT",  "TXN",
]

CHINA_A_UNIVERSE: List[str] = [
    "600519.SS",  # Kweichow Moutai
    "601318.SS",  # Ping An Insurance
    "600036.SS",  # China Merchants Bank
    "601166.SS",  # Industrial Bank
    "000858.SZ",  # Wuliangye Yibin
    "000333.SZ",  # Midea Group
    "002415.SZ",  # Hikvision
    "600276.SS",  # Jiangsu Hengrui Medicine
    "601398.SS",  # ICBC
    "600900.SS",  # China Yangtze Power
    "000001.SZ",  # Ping An Bank
    "600000.SS",  # SPD Bank
    "601988.SS",  # Bank of China
    "601288.SS",  # Agricultural Bank
    "000002.SZ",  # Vanke A
    "600309.SS",  # Wanhua Chemical
    "601012.SS",  # LONGi Green Energy
    "000568.SZ",  # Luzhou Laojiao
    "600887.SS",  # Inner Mongolia Yili
    "002594.SZ",  # BYD
]

CRYPTO_UNIVERSE: List[str] = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD",
    "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD",
    "LINK-USD", "UNI-USD", "ATOM-USD", "LTC-USD", "BCH-USD",
]

ETF_UNIVERSE: List[str] = [
    "SPY", "QQQ", "IWM", "GLD", "TLT", "EFA", "EEM",
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU",
    "VNQ", "IAU", "SHY", "HYG", "LQD",
]

# ---------------------------------------------------------------------------
# Dataset dataclass
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """
    Immutable container for a market dataset.

    Attributes
    ----------
    name      : Logical dataset name ("us_equity", "china_a", "crypto", "etf")
    frequency : Data frequency ("daily", "hourly")
    universe  : List of asset identifiers in the dataset
    data      : dict[field, DataFrame(T × N)] — standard 7-field panel
    """
    name:      str
    frequency: str
    universe:  List[str]
    data:      Dict[str, pd.DataFrame] = field(repr=False)

    def __post_init__(self) -> None:
        missing = [f for f in STANDARD_FIELDS if f not in self.data]
        if missing:
            raise ValueError(
                f"Dataset '{self.name}' is missing standard fields: {missing}"
            )

    @property
    def n_assets(self) -> int:
        return len(self.universe)

    @property
    def n_dates(self) -> int:
        return len(next(iter(self.data.values())))

    def __repr__(self) -> str:
        return (
            f"Dataset(name='{self.name}', freq='{self.frequency}', "
            f"assets={self.n_assets}, dates={self.n_dates})"
        )


# ---------------------------------------------------------------------------
# Alignment & standardisation helper
# ---------------------------------------------------------------------------

def _align_and_standardize(
    raw: Dict[str, pd.DataFrame],
    universe: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Align raw field DataFrames to (shared_index × universe) and ensure all
    7 STANDARD_FIELDS are present.  Missing assets → NaN.  Delisted assets
    remain as NaN after last available date (forward-fill before that).

    Parameters
    ----------
    raw      : dict field → DataFrame (may have different indices / columns)
    universe : Canonical asset list (columns of output panels)

    Returns
    -------
    dict with exactly STANDARD_FIELDS, each DataFrame (T × len(universe))
    """
    if not raw:
        raise ValueError("Empty raw dataset.")

    # Build union DatetimeIndex across all fields
    idx: Optional[pd.DatetimeIndex] = None
    for df in raw.values():
        i = pd.DatetimeIndex(df.index)
        idx = i if idx is None else idx.union(i)
    assert idx is not None

    # Reindex each field to master timeline × universe, ffill gaps ≤ 5 days
    aligned: Dict[str, pd.DataFrame] = {}
    for f, df in raw.items():
        df2 = df.reindex(index=idx, columns=universe).ffill(limit=5)
        aligned[f] = df2.astype(float)

    # Derive vwap if not present
    if "vwap" not in aligned:
        h = aligned.get("high")
        l = aligned.get("low")
        c = aligned.get("close")
        if h is not None and l is not None and c is not None:
            aligned["vwap"] = (h + l + c) / 3.0
        else:
            # Fallback to close
            base = aligned.get("close", next(iter(aligned.values())))
            aligned["vwap"] = base.copy()

    # Derive returns if not present
    if "returns" not in aligned:
        c = aligned.get("close", next(iter(aligned.values())))
        aligned["returns"] = np.log(c / c.shift(1))

    # Ensure all standard fields exist (fill missing with NaN)
    for f in STANDARD_FIELDS:
        if f not in aligned:
            base = next(iter(aligned.values()))
            aligned[f] = pd.DataFrame(
                np.nan, index=base.index, columns=base.columns
            )

    return {f: aligned[f] for f in STANDARD_FIELDS}


# ---------------------------------------------------------------------------
# Per-source loaders
# ---------------------------------------------------------------------------

def _load_via_yfinance(
    tickers: List[str],
    start: str,
    end: str,
) -> Dict[str, pd.DataFrame]:
    """Download OHLCV from Yahoo Finance and return aligned raw dict."""
    from .yahoo_provider import YahooFinanceProvider
    provider = YahooFinanceProvider(auto_adjust=True, progress=False)
    try:
        raw = provider.fetch(tickers, start=start, end=end)
    except Exception as exc:
        logger.warning("yfinance fetch failed for %d tickers: %s", len(tickers), exc)
        raw = {}
    return raw


def _load_via_ccxt(
    tickers: List[str],
    start: str,
    end: str,
    exchange_id: str = "binance",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV from a ccxt exchange for crypto tickers.

    Ticker format expected: "BTC-USD" → converts to "BTC/USDT" for ccxt.
    Falls back to yfinance on any failure.
    """
    try:
        import ccxt
    except ImportError:
        logger.debug("ccxt not installed; falling back to yfinance for crypto")
        return _load_via_yfinance(tickers, start, end)

    try:
        exchange_cls = getattr(ccxt, exchange_id)
        exchange = exchange_cls({"enableRateLimit": True})

        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms   = int(pd.Timestamp(end).timestamp() * 1000)

        open_d:   Dict[str, List] = {}
        high_d:   Dict[str, List] = {}
        low_d:    Dict[str, List] = {}
        close_d:  Dict[str, List] = {}
        volume_d: Dict[str, List] = {}
        index_set: set = set()

        for ticker in tickers:
            # Convert "BTC-USD" → "BTC/USDT"
            ccxt_symbol = ticker.replace("-USD", "/USDT").replace("-USDT", "/USDT")
            try:
                ohlcv = exchange.fetch_ohlcv(
                    ccxt_symbol, timeframe="1d",
                    since=start_ms, limit=1000,
                )
                if not ohlcv:
                    continue
                ts   = [pd.Timestamp(row[0], unit="ms") for row in ohlcv]
                # Filter to [start, end]
                rows = [
                    (t, row) for t, row in zip(ts, ohlcv)
                    if start_ms <= row[0] <= end_ms
                ]
                if not rows:
                    continue
                ts_filtered = [r[0] for r in rows]
                ohlcv_f     = [r[1] for r in rows]
                index_set.update(ts_filtered)
                open_d[ticker]   = dict(zip(ts_filtered, [r[1] for r in ohlcv_f]))
                high_d[ticker]   = dict(zip(ts_filtered, [r[2] for r in ohlcv_f]))
                low_d[ticker]    = dict(zip(ts_filtered, [r[3] for r in ohlcv_f]))
                close_d[ticker]  = dict(zip(ts_filtered, [r[4] for r in ohlcv_f]))
                volume_d[ticker] = dict(zip(ts_filtered, [r[5] for r in ohlcv_f]))
            except Exception as e:
                logger.debug("ccxt failed for %s: %s", ticker, e)

        if not index_set:
            logger.warning("ccxt returned no data; falling back to yfinance")
            return _load_via_yfinance(tickers, start, end)

        idx = pd.DatetimeIndex(sorted(index_set))

        def _build(d: Dict[str, Dict]) -> pd.DataFrame:
            return pd.DataFrame(
                {t: pd.Series(d.get(t, {})) for t in tickers},
                index=idx,
            ).reindex(idx)

        return {
            "open":   _build(open_d),
            "high":   _build(high_d),
            "low":    _build(low_d),
            "close":  _build(close_d),
            "volume": _build(volume_d),
        }

    except Exception as exc:
        logger.warning("ccxt pipeline failed: %s — falling back to yfinance", exc)
        return _load_via_yfinance(tickers, start, end)


def _load_from_local(
    data_dir: str,
    tickers: List[str],
    start: str,
    end: str,
) -> Dict[str, pd.DataFrame]:
    """Load from local Parquet files via LocalParquetProvider."""
    from .local_parquet_provider import LocalParquetProvider
    provider = LocalParquetProvider(root_dir=data_dir)
    try:
        return provider.fetch(tickers, start=start, end=end)
    except Exception as exc:
        logger.warning("LocalParquetProvider failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Public loaders per dataset type
# ---------------------------------------------------------------------------

def load_us_equity(
    start: str = "2018-01-01",
    end:   str = "2024-01-01",
    tickers: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
) -> Dataset:
    """Load US equity dataset (default universe: 44 large-cap stocks)."""
    universe = tickers or US_EQUITY_UNIVERSE
    raw = (
        _load_from_local(data_dir, universe, start, end)
        if data_dir else
        _load_via_yfinance(universe, start, end)
    )
    data = _align_and_standardize(raw, universe)
    return Dataset(
        name="us_equity",
        frequency="daily",
        universe=universe,
        data=data,
    )


def load_china_a(
    start: str = "2018-01-01",
    end:   str = "2024-01-01",
    tickers: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
) -> Dataset:
    """Load China A-share dataset (default: CSI 300 representative stocks)."""
    universe = tickers or CHINA_A_UNIVERSE
    raw = (
        _load_from_local(data_dir, universe, start, end)
        if data_dir else
        _load_via_yfinance(universe, start, end)
    )
    data = _align_and_standardize(raw, universe)
    return Dataset(
        name="china_a",
        frequency="daily",
        universe=universe,
        data=data,
    )


def load_crypto(
    start: str = "2018-01-01",
    end:   str = "2024-01-01",
    tickers: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
    use_ccxt: bool = False,
    exchange_id: str = "binance",
) -> Dataset:
    """Load crypto dataset (default: top 15 by market cap, -USD pairs)."""
    universe = tickers or CRYPTO_UNIVERSE
    if data_dir:
        raw = _load_from_local(data_dir, universe, start, end)
    elif use_ccxt:
        raw = _load_via_ccxt(universe, start, end, exchange_id)
    else:
        raw = _load_via_yfinance(universe, start, end)
    data = _align_and_standardize(raw, universe)
    return Dataset(
        name="crypto",
        frequency="daily",
        universe=universe,
        data=data,
    )


def load_etf(
    start: str = "2018-01-01",
    end:   str = "2024-01-01",
    tickers: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
) -> Dataset:
    """Load ETF dataset (default: 20 sector/factor/bond ETFs)."""
    universe = tickers or ETF_UNIVERSE
    raw = (
        _load_from_local(data_dir, universe, start, end)
        if data_dir else
        _load_via_yfinance(universe, start, end)
    )
    data = _align_and_standardize(raw, universe)
    return Dataset(
        name="etf",
        frequency="daily",
        universe=universe,
        data=data,
    )


# ---------------------------------------------------------------------------
# DatasetRegistry
# ---------------------------------------------------------------------------

class DatasetRegistry:
    """
    Registry of named dataset loaders.

    Built-in datasets: "us_equity", "china_a", "crypto", "etf"
    Custom loaders can be registered via register().
    """

    _BUILTIN_LOADERS: Dict[str, Callable] = {
        "us_equity": load_us_equity,
        "china_a":   load_china_a,
        "crypto":    load_crypto,
        "etf":       load_etf,
    }

    def __init__(self) -> None:
        self._loaders: Dict[str, Callable] = dict(self._BUILTIN_LOADERS)
        self._cache:   Dict[str, Dataset]  = {}

    def register(self, name: str, loader_fn: Callable) -> None:
        """Register a custom loader function: loader_fn(**kwargs) → Dataset."""
        self._loaders[name] = loader_fn

    def available(self) -> List[str]:
        return sorted(self._loaders.keys())

    def load(
        self,
        name: str,
        start: str = "2018-01-01",
        end:   str = "2024-01-01",
        use_cache: bool = True,
        **kwargs,
    ) -> Dataset:
        """
        Load a dataset by name.

        Parameters
        ----------
        name      : Dataset identifier (e.g. "us_equity")
        start/end : Date range strings (ISO format)
        use_cache : If True, return cached version if already loaded
        **kwargs  : Extra args passed to the loader function
        """
        cache_key = f"{name}|{start}|{end}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        if name not in self._loaders:
            raise KeyError(
                f"Unknown dataset '{name}'. Available: {self.available()}"
            )

        logger.info("Loading dataset '%s' [%s → %s]", name, start, end)
        dataset = self._loaders[name](start=start, end=end, **kwargs)

        if use_cache:
            self._cache[cache_key] = dataset

        return dataset

    def clear_cache(self) -> None:
        self._cache.clear()


# Module-level default registry
_default_registry = DatasetRegistry()


def load_dataset(
    name: str,
    start: str = "2018-01-01",
    end:   str = "2024-01-01",
    use_cache: bool = True,
    **kwargs,
) -> Dataset:
    """
    Convenience function: load a named dataset from the default registry.

    Usage::

        ds = load_dataset("us_equity", start="2020-01-01", end="2024-01-01")
        signal = executor.run_expr("rank(ts_delta(close,5))", ds.data)
    """
    return _default_registry.load(name, start=start, end=end, use_cache=use_cache, **kwargs)


def get_registry() -> DatasetRegistry:
    """Return the module-level default registry."""
    return _default_registry
