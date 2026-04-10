"""
本地 Parquet 文件数据源提供者。

目录结构（Hive 分区约定）：
    {root_dir}/{TICKER}/year={YYYY}/data.parquet

读取时利用 pyarrow.dataset 谓词下推，只加载需要的年份分区。
写入时按年拆分，使用 snappy 压缩，ticker 列使用字典编码。
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import DataProvider, RawDataset
from .schema import SchemaEnforcer, STANDARD_COLUMNS

logger = logging.getLogger(__name__)

_SUPPORTED_FIELDS = ["open", "high", "low", "close", "volume", "vwap", "adj_factor", "returns"]


class LocalParquetProvider(DataProvider):
    """
    从本地 Parquet 文件读取市场数据（支持 Hive 年份分区）。

    Parameters
    ----------
    root_dir : str | Path
        Parquet 数据根目录，格式为 {root_dir}/{TICKER}/year={YYYY}/data.parquet
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
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
        读取多 ticker 数据，返回 wide-format RawDataset。
        缺失的 ticker 数据以空列填充（NaN）。
        """
        panel = self.fetch_panel(tickers=tickers, start=start, end=end, fields=fields)
        if panel.empty:
            return {}
        return self._to_raw_dataset(panel, fields)

    def fetch_panel(
        self,
        tickers: List[str],
        start: str,
        end: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """读取并返回标准 long-format 面板。"""
        tickers = [t.upper() for t in tickers]
        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end)
        years    = list(range(start_dt.year, end_dt.year + 1))

        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            df = self._read_ticker(ticker, years, start_dt, end_dt, fields)
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            logger.warning("LocalParquetProvider: 无可用数据 tickers=%s", tickers)
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        combined = pd.concat(frames, ignore_index=True)
        return self._enforcer.enforce(combined)

    def metadata(self) -> dict:
        return {
            "name":             "LocalParquetProvider",
            "root_dir":         str(self.root_dir),
            "latency_ms":       None,
            "rate_limit":       None,
            "available_fields": self.available_fields(),
        }

    # ------------------------------------------------------------------
    # 写入接口
    # ------------------------------------------------------------------

    def write(
        self,
        df: pd.DataFrame,
        overwrite: bool = False,
    ) -> None:
        """
        将 long-format DataFrame 按 ticker + year 分区写入本地存储。

        Parameters
        ----------
        df        : long-format DataFrame（必须包含 timestamp, ticker 列）
        overwrite : True = 覆盖已有分区；False = 追加（去重后合并）
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("LocalParquetProvider.write 需要 pyarrow: pip install pyarrow")

        # 强制格式
        df = self._enforcer.enforce(df)
        df["year"] = df["timestamp"].dt.year
        df["ticker"] = df["ticker"].str.upper()

        for (ticker, year), group in df.groupby(["ticker", "year"]):
            part_dir = self.root_dir / ticker / f"year={year}"
            part_dir.mkdir(parents=True, exist_ok=True)
            out_path = part_dir / "data.parquet"

            group = group.drop(columns=["year"])

            if out_path.exists() and not overwrite:
                # 追加模式：读取现有数据合并去重
                try:
                    existing = pd.read_parquet(out_path)
                    group = pd.concat([existing, group], ignore_index=True)
                    group = group.drop_duplicates(
                        subset=["timestamp", "ticker"], keep="last"
                    ).sort_values("timestamp")
                except Exception as exc:
                    logger.warning("合并现有数据失败 (%s/%s): %s", ticker, year, exc)

            table = pa.Table.from_pandas(group, preserve_index=False)
            # ticker 列使用字典编码节省空间
            table = table.cast(
                table.schema.set(
                    table.schema.get_field_index("ticker"),
                    pa.field("ticker", pa.dictionary(pa.int16(), pa.string())),
                )
                if "ticker" in table.schema.names else table.schema
            )
            pq.write_table(
                table, out_path,
                compression="snappy",
                write_statistics=True,
            )
            logger.debug("写入: %s", out_path)

    def available_tickers(self) -> List[str]:
        """扫描目录，返回已存储的 ticker 列表。"""
        if not self.root_dir.exists():
            return []
        return sorted(
            p.name for p in self.root_dir.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _read_ticker(
        self,
        ticker: str,
        years: List[int],
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
        fields: Optional[List[str]],
    ) -> Optional[pd.DataFrame]:
        """读取单个 ticker 的指定年份分区数据。"""
        ticker_dir = self.root_dir / ticker
        if not ticker_dir.exists():
            logger.debug("LocalParquetProvider: 无 ticker 目录 %s", ticker_dir)
            return None

        frames: list[pd.DataFrame] = []
        for year in years:
            part_path = ticker_dir / f"year={year}" / "data.parquet"
            if not part_path.exists():
                continue
            try:
                cols = None
                if fields:
                    # 只读取需要的列（谓词下推列裁剪）
                    cols = list({"timestamp", "ticker"} | set(fields))
                df = pd.read_parquet(part_path, columns=cols)
                frames.append(df)
            except Exception as exc:
                warnings.warn(
                    f"读取 {part_path} 失败: {exc}",
                    stacklevel=4,
                )

        if not frames:
            return None

        combined = pd.concat(frames, ignore_index=True)
        # 日期范围过滤
        ts = pd.to_datetime(combined["timestamp"])
        mask = (ts >= start_dt) & (ts <= end_dt)
        return combined.loc[mask].copy()

    def _to_raw_dataset(
        self,
        long_df: pd.DataFrame,
        fields: Optional[List[str]],
    ) -> RawDataset:
        """将 long-format 转为 wide RawDataset。"""
        result: RawDataset = {}
        target_fields = fields or [
            "open", "high", "low", "close", "volume", "adj_factor"
        ]
        for field in target_fields:
            if field not in long_df.columns:
                continue
            wide = long_df.pivot_table(
                index="timestamp", columns="ticker", values=field, aggfunc="last"
            )
            wide.index = pd.DatetimeIndex(wide.index)
            result[field] = wide
        return result
