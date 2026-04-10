"""
Parquet 特征库（Feature Store）与分块加载器（DataChunker）。

存储约定（Hive 分区）：
    {store_dir}/{name}/year={YYYY}/data.parquet

元数据文件：
    {store_dir}/{name}/_metadata.json
    记录 schema、ticker 列表、日期范围，用于缓存命中判断。

读取采用 pyarrow.dataset 谓词下推：
    - 按年份分区过滤（只扫描需要的分区目录）
    - 按 ticker 列过滤（减少内存占用）

DataChunker 将大 Universe 按 chunk_size 分批懒加载，避免 OOM。
"""

from __future__ import annotations

import json
import logging
from itertools import islice
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _batched(iterable, n: int):
    """将可迭代对象按 n 个一批分割（Python 3.12+ 有 itertools.batched，此处自实现）。"""
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


# ---------------------------------------------------------------------------
# ParquetFeatureStore
# ---------------------------------------------------------------------------

class ParquetFeatureStore:
    """
    基于 Apache Parquet + 年份分区的本地特征库。

    Parameters
    ----------
    store_dir : str | Path  特征库根目录
    """

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def save(
        self,
        panel: pd.DataFrame,
        name: str,
        date_col: str = "timestamp",
        ticker_col: str = "ticker",
        overwrite: bool = False,
    ) -> None:
        """
        将 long-format 面板写入特征库，按年份分区存储。

        Parameters
        ----------
        panel    : long-format DataFrame
        name     : 数据集名称（如 "us_equity_daily"）
        overwrite: True = 完全覆盖；False = 追加（去重合并）
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("feature_store 需要 pyarrow: pip install pyarrow")

        if panel.empty:
            logger.warning("FeatureStore.save: 输入面板为空，跳过写入")
            return

        dataset_dir = self.store_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        panel = panel.copy()
        panel[date_col] = pd.to_datetime(panel[date_col])
        panel["_year"]  = panel[date_col].dt.year

        for year, group in panel.groupby("_year"):
            part_dir = dataset_dir / f"year={year}"
            part_dir.mkdir(parents=True, exist_ok=True)
            out_path = part_dir / "data.parquet"

            group = group.drop(columns=["_year"])

            if out_path.exists() and not overwrite:
                try:
                    existing = pd.read_parquet(out_path)
                    group = pd.concat([existing, group], ignore_index=True)
                    group = (
                        group
                        .drop_duplicates(subset=[date_col, ticker_col], keep="last")
                        .sort_values([date_col, ticker_col])
                    )
                except Exception as exc:
                    logger.warning("合并已有分区数据失败 (%s/%s): %s", name, year, exc)

            # 构建 PyArrow Table，ticker 列字典编码
            table = pa.Table.from_pandas(group, preserve_index=False)
            if ticker_col in table.schema.names:
                idx = table.schema.get_field_index(ticker_col)
                new_field = pa.field(ticker_col, pa.dictionary(pa.int16(), pa.string()))
                new_schema = table.schema.set(idx, new_field)
                table = table.cast(new_schema)

            import pyarrow.parquet as pq
            pq.write_table(
                table, out_path,
                compression="snappy",
                write_statistics=True,
            )
            logger.debug("FeatureStore: 写入 %s", out_path)

        # 更新元数据
        self._update_metadata(name, panel, date_col, ticker_col)

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def load(
        self,
        name: str,
        tickers: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        fields: Optional[List[str]] = None,
        date_col: str = "timestamp",
        ticker_col: str = "ticker",
    ) -> pd.DataFrame:
        """
        从特征库加载数据，支持按年份、ticker、字段裁剪。

        Returns
        -------
        pd.DataFrame (long-format)，若数据集不存在则返回空 DataFrame。
        """
        dataset_dir = self.store_dir / name
        if not dataset_dir.exists():
            logger.debug("FeatureStore: 数据集 '%s' 不存在", name)
            return pd.DataFrame()

        start_dt = pd.Timestamp(start) if start else None
        end_dt   = pd.Timestamp(end)   if end   else None

        # 确定需要加载的年份分区
        years = self._get_years(dataset_dir, start_dt, end_dt)
        if not years:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        read_cols = None
        if fields:
            read_cols = list({date_col, ticker_col} | set(fields))

        for year in years:
            part_path = dataset_dir / f"year={year}" / "data.parquet"
            if not part_path.exists():
                continue
            try:
                df = pd.read_parquet(part_path, columns=read_cols)
                frames.append(df)
            except Exception as exc:
                logger.warning("读取分区失败 %s: %s", part_path, exc)

        if not frames:
            return pd.DataFrame()

        panel = pd.concat(frames, ignore_index=True)
        panel[date_col] = pd.to_datetime(panel[date_col])
        panel[ticker_col] = panel[ticker_col].astype(str)

        # 日期过滤
        if start_dt:
            panel = panel[panel[date_col] >= start_dt]
        if end_dt:
            panel = panel[panel[date_col] <= end_dt]

        # Ticker 过滤
        if tickers:
            upper_tickers = [t.upper() for t in tickers]
            panel = panel[panel[ticker_col].str.upper().isin(upper_tickers)]

        return panel.sort_values([date_col, ticker_col]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 缓存命中检查
    # ------------------------------------------------------------------

    def has_data(
        self,
        name: str,
        tickers: List[str],
        start: str,
        end: str,
    ) -> bool:
        """
        检查特征库中是否已有覆盖所有 tickers 且满足日期范围的数据。
        用于 DataManager 的缓存命中判断。
        """
        meta = self._load_metadata(name)
        if not meta:
            return False

        meta_start = pd.Timestamp(meta.get("start", "1900-01-01"))
        meta_end   = pd.Timestamp(meta.get("end",   "1900-01-01"))
        meta_tickers = set(meta.get("tickers", []))

        req_start = pd.Timestamp(start)
        req_end   = pd.Timestamp(end)
        req_tickers = {t.upper() for t in tickers}

        return (
            meta_start <= req_start
            and meta_end >= req_end
            and req_tickers.issubset(meta_tickers)
        )

    # ------------------------------------------------------------------
    # 元数据
    # ------------------------------------------------------------------

    def _update_metadata(
        self,
        name: str,
        panel: pd.DataFrame,
        date_col: str,
        ticker_col: str,
    ) -> None:
        meta_path = self.store_dir / name / "_metadata.json"
        existing  = self._load_metadata(name) or {}

        new_start   = panel[date_col].min()
        new_end     = panel[date_col].max()
        new_tickers = set(panel[ticker_col].str.upper().unique())

        # 与已有元数据合并（取并集）
        if existing:
            old_start = pd.Timestamp(existing.get("start", new_start))
            old_end   = pd.Timestamp(existing.get("end",   new_end))
            combined_tickers = set(existing.get("tickers", [])) | new_tickers
            start_ts = min(old_start, new_start)
            end_ts   = max(old_end,   new_end)
        else:
            combined_tickers = new_tickers
            start_ts = new_start
            end_ts   = new_end

        meta = {
            "name":     name,
            "start":    str(start_ts.date()),
            "end":      str(end_ts.date()),
            "tickers":  sorted(combined_tickers),
            "fields":   [c for c in panel.columns if c not in (date_col, ticker_col)],
            "n_rows":   len(panel),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _load_metadata(self, name: str) -> Optional[Dict]:
        meta_path = self.store_dir / name / "_metadata.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _get_years(
        dataset_dir: Path,
        start_dt: Optional[pd.Timestamp],
        end_dt: Optional[pd.Timestamp],
    ) -> List[int]:
        """扫描分区目录，返回需要加载的年份列表。"""
        years = []
        for p in dataset_dir.iterdir():
            if p.is_dir() and p.name.startswith("year="):
                try:
                    y = int(p.name.split("=")[1])
                    if start_dt and y < start_dt.year:
                        continue
                    if end_dt and y > end_dt.year:
                        continue
                    years.append(y)
                except ValueError:
                    pass
        return sorted(years)


# ---------------------------------------------------------------------------
# DataChunker
# ---------------------------------------------------------------------------

class DataChunker:
    """
    按 ticker 分批懒加载特征库数据，避免大 Universe 爆内存。

    Parameters
    ----------
    store      : ParquetFeatureStore 实例
    chunk_size : 每批加载的 ticker 数量（默认 50）
    """

    def __init__(self, store: ParquetFeatureStore, chunk_size: int = 50) -> None:
        self.store      = store
        self.chunk_size = chunk_size

    def iter_chunks(
        self,
        name: str,
        tickers: List[str],
        start: str,
        end: str,
        fields: Optional[List[str]] = None,
        date_col: str = "timestamp",
        ticker_col: str = "ticker",
    ) -> Iterator[pd.DataFrame]:
        """
        按批次逐步加载数据，每次 yield 一个 chunk DataFrame。

        Example
        -------
        ::

            chunker = DataChunker(store, chunk_size=50)
            for chunk in chunker.iter_chunks("us_daily", tickers, "2020-01-01", "2023-12-31"):
                process(chunk)   # 逐批处理，无需全量加载
        """
        tickers = [t.upper() for t in tickers]
        n_total = len(tickers)
        n_batches = (n_total + self.chunk_size - 1) // self.chunk_size

        logger.info(
            "DataChunker: %d tickers，分 %d 批（chunk_size=%d）",
            n_total, n_batches, self.chunk_size,
        )

        for i, batch in enumerate(_batched(tickers, self.chunk_size)):
            logger.debug("DataChunker: 加载第 %d/%d 批 (%d tickers)", i + 1, n_batches, len(batch))
            chunk = self.store.load(
                name=name,
                tickers=batch,
                start=start,
                end=end,
                fields=fields,
                date_col=date_col,
                ticker_col=ticker_col,
            )
            if not chunk.empty:
                yield chunk

    def collect(
        self,
        name: str,
        tickers: List[str],
        start: str,
        end: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        分批加载并拼接为完整 DataFrame（适合内存允许时使用）。
        大 Universe 建议直接使用 iter_chunks。
        """
        chunks = list(self.iter_chunks(name, tickers, start, end, fields))
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)
