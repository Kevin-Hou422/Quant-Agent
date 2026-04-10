"""
Alpha Vantage 数据源提供者。

调用 TIME_SERIES_DAILY_ADJUSTED 端点，返回标准 Schema long-format DataFrame。

限速策略（免费 API Key）：每分钟最多 5 次请求，内置指数退避重试。
批量下载失败的 ticker 返回空行（不中断整批下载）。
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from .base import DataProvider, RawDataset
from .schema import SchemaEnforcer, STANDARD_COLUMNS

logger = logging.getLogger(__name__)

# Alpha Vantage API 端点
_BASE_URL = "https://www.alphavantage.co/query"

# AV 原始字段 → 标准 Schema 字段
_FIELD_MAP: Dict[str, str] = {
    "1. open":             "open",
    "2. high":             "high",
    "3. low":              "low",
    "4. close":            "close",
    "5. adjusted close":   "adj_close_raw",  # 用于计算 adj_factor
    "6. volume":           "volume",
    "7. dividend amount":  "dividend",
    "8. split coefficient": "split_coef",
}

_SUPPORTED_FIELDS = ["open", "high", "low", "close", "volume", "vwap", "adj_factor", "returns"]


class AlphaVantageProvider(DataProvider):
    """
    通过 Alpha Vantage REST API 获取每日 OHLCV 数据（含复权因子）。

    Parameters
    ----------
    api_key     : Alpha Vantage API Key
    output_size : 'full'（全量历史）或 'compact'（最近 100 条）
    requests_per_minute : API 速率限制（免费账户为 5）
    max_retries : 单个 ticker 最大重试次数
    timeout_s   : HTTP 请求超时秒数
    """

    def __init__(
        self,
        api_key: str,
        output_size: str = "full",
        requests_per_minute: int = 5,
        max_retries: int = 3,
        timeout_s: int = 30,
    ) -> None:
        if not api_key:
            raise ValueError("Alpha Vantage api_key 不能为空")
        self.api_key = api_key
        self.output_size = output_size
        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._min_interval_s = 60.0 / requests_per_minute
        self._last_request_ts = 0.0
        self._enforcer = SchemaEnforcer(allow_extra=True)

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
        批量获取多 ticker 数据，返回 wide-format RawDataset。
        单个 ticker 失败时记录 warning，继续处理其他 ticker。
        """
        tickers = [t.upper() for t in tickers]
        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end)

        all_long: list[pd.DataFrame] = []

        for ticker in tickers:
            df = self._fetch_one(ticker, start_dt, end_dt)
            if df is not None and not df.empty:
                all_long.append(df)

        if not all_long:
            logger.warning("AlphaVantageProvider: 所有 ticker 均获取失败或无数据")
            return {}

        combined = pd.concat(all_long, ignore_index=True)
        return self._to_raw_dataset(combined, fields)

    def fetch_panel(
        self,
        tickers: List[str],
        start: str,
        end: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """直接返回 long-format 面板（覆盖父类默认实现以避免二次转换）。"""
        tickers = [t.upper() for t in tickers]
        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end)

        all_long: list[pd.DataFrame] = []
        for ticker in tickers:
            df = self._fetch_one(ticker, start_dt, end_dt)
            if df is not None and not df.empty:
                all_long.append(df)

        if not all_long:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        combined = pd.concat(all_long, ignore_index=True)
        return self._enforcer.enforce(combined)

    def metadata(self) -> dict:
        return {
            "name":             "AlphaVantageProvider",
            "latency_ms":       getattr(self, "_last_latency_ms", None),
            "rate_limit":       self.requests_per_minute,
            "available_fields": self.available_fields(),
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _fetch_one(
        self,
        ticker: str,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        """获取单个 ticker 数据，带重试和速率限制。"""
        for attempt in range(self.max_retries):
            self._rate_limit_wait()
            t0 = time.monotonic()
            try:
                resp = requests.get(
                    _BASE_URL,
                    params={
                        "function":   "TIME_SERIES_DAILY_ADJUSTED",
                        "symbol":     ticker,
                        "outputsize": self.output_size,
                        "apikey":     self.api_key,
                        "datatype":   "json",
                    },
                    timeout=self.timeout_s,
                )
                self._last_latency_ms = (time.monotonic() - t0) * 1000
                resp.raise_for_status()
                data = resp.json()

                if "Note" in data or "Information" in data:
                    msg = data.get("Note") or data.get("Information", "")
                    warnings.warn(
                        f"AlphaVantage 速率限制 ({ticker}): {msg[:120]}",
                        stacklevel=3,
                    )
                    time.sleep(60)
                    continue

                ts_key = "Time Series (Daily)"
                if ts_key not in data:
                    warnings.warn(
                        f"AlphaVantageProvider: ticker '{ticker}' 无数据 "
                        f"(response keys: {list(data.keys())})",
                        stacklevel=3,
                    )
                    return None

                return self._parse_response(data[ts_key], ticker, start_dt, end_dt)

            except requests.RequestException as exc:
                wait = 2 ** attempt
                logger.warning(
                    "AlphaVantage 请求失败 ticker=%s, attempt=%d/%d, "
                    "等待 %ds: %s",
                    ticker, attempt + 1, self.max_retries, wait, exc,
                )
                time.sleep(wait)

        warnings.warn(
            f"AlphaVantageProvider: ticker '{ticker}' 在 {self.max_retries} 次"
            "重试后仍失败，跳过。",
            stacklevel=2,
        )
        return None

    def _rate_limit_wait(self) -> None:
        """确保请求间隔 ≥ min_interval_s 秒。"""
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)
        self._last_request_ts = time.monotonic()

    @staticmethod
    def _parse_response(
        ts_data: dict,
        ticker: str,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
    ) -> pd.DataFrame:
        """将 AV JSON 时间序列解析为 long-format DataFrame。"""
        rows = []
        for date_str, vals in ts_data.items():
            ts = pd.Timestamp(date_str)
            if not (start_dt <= ts <= end_dt):
                continue
            row: dict = {"timestamp": ts, "ticker": ticker}
            for av_key, std_key in _FIELD_MAP.items():
                row[std_key] = float(vals.get(av_key, np.nan))

            # adj_factor = adjusted_close / close（防止除零）
            close = row.get("close", np.nan)
            adj_c = row.pop("adj_close_raw", np.nan)
            if close and close != 0 and not np.isnan(close) and not np.isnan(adj_c):
                row["adj_factor"] = adj_c / close
            else:
                row["adj_factor"] = 1.0

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _to_raw_dataset(
        self,
        long_df: pd.DataFrame,
        fields: Optional[List[str]],
    ) -> RawDataset:
        """将 long-format 转为 wide RawDataset（兼容旧接口）。"""
        result: RawDataset = {}
        target_fields = fields or ["open", "high", "low", "close", "volume", "adj_factor"]

        for field in target_fields:
            if field not in long_df.columns:
                continue
            wide = long_df.pivot_table(
                index="timestamp", columns="ticker", values=field, aggfunc="last"
            )
            wide.index = pd.DatetimeIndex(wide.index)
            result[field] = wide

        return result
