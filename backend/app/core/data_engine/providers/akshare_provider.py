"""
akshare_provider.py — AkShare data provider for China A-share equities.

Uses akshare.stock_zh_a_hist() for daily OHLCV (forward-adjusted).
Symbol format accepted: "600519.SH", "300750.SZ" — exchange suffix is
stripped automatically (akshare uses bare 6-digit codes).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# akshare column → standard field
_COLUMN_MAP = {
    "日期":   "date",
    "开盘":   "open",
    "收盘":   "close",
    "最高":   "high",
    "最低":   "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_chg",
}


def _strip_suffix(symbol: str) -> str:
    """'600519.SH' → '600519', '300750.SZ' → '300750'."""
    return symbol.split(".")[0]


class AkshareProvider:
    """
    Fetch daily OHLCV data for China A-share stocks via AkShare.

    Parameters
    ----------
    adjust  : "qfq" (forward-adjusted, default) | "hfq" | "" (unadjusted)
    delay_s : Sleep between requests to avoid rate limiting
    """

    def __init__(self, adjust: str = "qfq", delay_s: float = 0.3) -> None:
        self.adjust  = adjust
        self.delay_s = delay_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        symbols: List[str],
        start: str,
        end:   str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for all symbols.

        Parameters
        ----------
        symbols : list of ticker codes ("600519.SH", "300750.SZ", …)
        start   : ISO date string "YYYY-MM-DD"
        end     : ISO date string "YYYY-MM-DD"

        Returns
        -------
        dict[field → DataFrame(T × N)] — keys: open,high,low,close,volume,vwap,returns
        """
        try:
            import akshare as ak
        except ImportError as exc:
            raise ImportError(
                "akshare is required for China A-share data. "
                "Install with: pip install akshare"
            ) from exc

        start_ak = start.replace("-", "")
        end_ak   = end.replace("-", "")

        frames: Dict[str, pd.DataFrame] = {}  # symbol → per-ticker df

        for sym in symbols:
            code = _strip_suffix(sym)
            for attempt in range(3):
                try:
                    df = ak.stock_zh_a_hist(
                        symbol     = code,
                        period     = "daily",
                        start_date = start_ak,
                        end_date   = end_ak,
                        adjust     = self.adjust,
                    )
                    if df is not None and not df.empty:
                        df = df.rename(columns=_COLUMN_MAP)
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.set_index("date").sort_index()
                        frames[sym] = df
                    break
                except Exception as exc:
                    logger.warning(
                        "akshare fetch failed for %s (attempt %d): %s",
                        sym, attempt + 1, exc
                    )
                    if attempt < 2:
                        time.sleep(self.delay_s * 2)
            time.sleep(self.delay_s)

        if not frames:
            raise ValueError(f"AkshareProvider: no data returned for {symbols}")

        return self._to_panel(frames, symbols)

    # ------------------------------------------------------------------
    # Panel construction
    # ------------------------------------------------------------------

    def _to_panel(
        self,
        frames: Dict[str, pd.DataFrame],
        symbols: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """Convert per-ticker DataFrames to dict[field → T×N DataFrame]."""
        # Build union index
        idx = None
        for df in frames.values():
            idx = df.index if idx is None else idx.union(df.index)

        def _field(key: str) -> pd.DataFrame:
            cols = {}
            for sym in symbols:
                if sym in frames and key in frames[sym].columns:
                    cols[sym] = frames[sym][key].reindex(idx)
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
    # Market cap helper (batch)
    # ------------------------------------------------------------------

    @staticmethod
    def fetch_market_cap(symbols: List[str]) -> Dict[str, float]:
        """
        Fetch approximate market cap (CNY) for each symbol via akshare.
        Returns dict[symbol → market_cap_cny].  Missing symbols → NaN.
        """
        try:
            import akshare as ak
        except ImportError:
            return {}

        result: Dict[str, float] = {}
        for sym in symbols:
            code = _strip_suffix(sym)
            try:
                info = ak.stock_individual_info_em(symbol=code)
                # info is a 2-column DataFrame: item / value
                row = info[info["item"] == "总市值"]
                if not row.empty:
                    val = str(row["value"].iloc[0]).replace("亿", "")
                    result[sym] = float(val) * 1e8   # convert 亿 to CNY
            except Exception:
                result[sym] = float("nan")
            time.sleep(0.1)
        return result
