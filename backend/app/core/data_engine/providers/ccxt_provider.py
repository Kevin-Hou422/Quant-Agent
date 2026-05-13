"""
ccxt_provider.py — CCXT Binance provider for crypto OHLCV data.

Uses ccxt.binance.fetch_ohlcv() for daily candles.
Symbol format: "BTC/USDT", "ETH/USDT" (ccxt native format).
Falls back to yfinance ("-USD" suffix) if ccxt is not installed.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MS_PER_DAY = 86_400_000


class CcxtBinanceProvider:
    """
    Fetch daily OHLCV from Binance via ccxt.

    Parameters
    ----------
    exchange_id : ccxt exchange identifier (default "binance")
    limit       : Max candles per request (Binance daily limit = 1000)
    delay_s     : Sleep between requests
    """

    def __init__(
        self,
        exchange_id: str  = "binance",
        limit:       int  = 1000,
        delay_s:     float = 0.2,
    ) -> None:
        self.exchange_id = exchange_id
        self.limit       = limit
        self.delay_s     = delay_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        symbols: List[str],
        start:   str,
        end:     str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily OHLCV for all symbols from Binance.

        Parameters
        ----------
        symbols : list of ccxt symbols ("BTC/USDT", "ETH/USDT", …)
        start   : ISO date string "YYYY-MM-DD"
        end     : ISO date string "YYYY-MM-DD"

        Returns
        -------
        dict[field → DataFrame(T × N)]
        """
        try:
            import ccxt
        except ImportError:
            logger.warning(
                "ccxt not installed — falling back to yfinance for crypto. "
                "Install with: pip install ccxt"
            )
            return self._yfinance_fallback(symbols, start, end)

        try:
            exchange_cls = getattr(ccxt, self.exchange_id)
        except AttributeError as exc:
            raise ValueError(f"Unknown ccxt exchange: '{self.exchange_id}'") from exc

        exchange = exchange_cls({"enableRateLimit": True})
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms   = int(pd.Timestamp(end).timestamp()   * 1000)

        frames: Dict[str, pd.DataFrame] = {}

        for sym in symbols:
            candles = self._fetch_all_candles(exchange, sym, start_ms, end_ms)
            if candles:
                df = pd.DataFrame(
                    candles,
                    columns=["ts", "open", "high", "low", "close", "volume"]
                )
                df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
                df = df.set_index("date").drop(columns=["ts"]).sort_index()
                df = df[df.index >= pd.Timestamp(start)]
                df = df[df.index <= pd.Timestamp(end)]
                frames[sym] = df.astype(float)
            else:
                logger.warning("No CCXT data returned for %s", sym)
            time.sleep(self.delay_s)

        if not frames:
            logger.warning("CCXT returned no data — trying yfinance fallback")
            return self._yfinance_fallback(symbols, start, end)

        return self._to_panel(frames, symbols)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_all_candles(
        self,
        exchange,
        symbol: str,
        start_ms: int,
        end_ms:   int,
    ) -> list:
        """Paginate through ccxt to collect all daily candles in [start, end]."""
        all_candles = []
        since = start_ms
        for _ in range(30):  # max 30 pages × 1000 = 30k days
            try:
                batch = exchange.fetch_ohlcv(
                    symbol,
                    timeframe = "1d",
                    since     = since,
                    limit     = self.limit,
                )
            except Exception as exc:
                logger.warning("CCXT error fetching %s: %s", symbol, exc)
                break
            if not batch:
                break
            all_candles.extend(batch)
            last_ts = batch[-1][0]
            if last_ts >= end_ms or len(batch) < self.limit:
                break
            since = last_ts + _MS_PER_DAY
            time.sleep(self.delay_s)
        # Filter to [start_ms, end_ms]
        return [c for c in all_candles if start_ms <= c[0] <= end_ms]

    def _to_panel(
        self,
        frames: Dict[str, pd.DataFrame],
        symbols: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """Build dict[field → T×N DataFrame] from per-symbol DataFrames."""
        idx = None
        for df in frames.values():
            idx = df.index if idx is None else idx.union(df.index)

        def _field(key: str) -> pd.DataFrame:
            cols = {}
            for sym in symbols:
                if sym in frames and key in frames[sym].columns:
                    cols[sym] = frames[sym][key].reindex(idx).ffill(limit=3)
                else:
                    cols[sym] = pd.Series(np.nan, index=idx)
            return pd.DataFrame(cols, index=idx).astype(float)

        close  = _field("close")
        high   = _field("high")
        low    = _field("low")
        open_  = _field("open")
        vol    = _field("volume")
        vwap   = (high + low + close) / 3.0
        rets   = np.log(close / close.shift(1))

        return {
            "open":    open_,
            "high":    high,
            "low":     low,
            "close":   close,
            "volume":  vol,
            "vwap":    vwap,
            "returns": rets,
        }

    # ------------------------------------------------------------------
    # yfinance fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _yfinance_fallback(
        symbols: List[str],
        start:   str,
        end:     str,
    ) -> Dict[str, pd.DataFrame]:
        """Convert BTC/USDT → BTC-USD and fetch from yfinance."""
        import yfinance as yf
        import numpy as np

        yf_tickers = [s.replace("/USDT", "-USD").replace("/BTC", "-BTC") for s in symbols]
        ticker_map  = dict(zip(yf_tickers, symbols))

        raw = yf.download(
            tickers=yf_tickers, start=start, end=end,
            auto_adjust=True, progress=False, group_by="column",
        )
        if raw.empty:
            return {}

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_tuples(
                [(a.lower(), b.upper()) for a, b in raw.columns]
            )
        else:
            t = yf_tickers[0].upper()
            raw.columns = pd.MultiIndex.from_tuples([(c.lower(), t) for c in raw.columns])

        def _extract(field: str) -> pd.DataFrame:
            try:
                df = raw[field]
            except KeyError:
                return pd.DataFrame(index=raw.index, columns=symbols, dtype=float)
            if isinstance(df, pd.Series):
                df = df.to_frame(name=yf_tickers[0].upper())
            df.columns = [ticker_map.get(c, c) for c in df.columns]
            return df.reindex(columns=symbols).astype(float)

        close  = _extract("close")
        high   = _extract("high")
        low    = _extract("low")
        open_  = _extract("open")
        vol    = _extract("volume")
        vwap   = (high + low + close) / 3.0
        rets   = np.log(close / close.shift(1))

        return {
            "open": open_, "high": high, "low": low,
            "close": close, "volume": vol, "vwap": vwap, "returns": rets,
        }
