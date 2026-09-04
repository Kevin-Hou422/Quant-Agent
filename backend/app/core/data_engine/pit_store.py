"""
pit_store.py — Point-in-Time（时点）数据存储（Phase 8.1，2026-08-19）

目的
----
每日摄取的数据按 `(field, date, as_of)` **追加**存储，历史分区**只追加不修改**。
`as_of` = 该观测**被看到/被摄取**的时点（数据 vintage）；`date`(timestamp) = 观测**所属**
的市场交易日。二者分离 = 双时点（bitemporal）语义，从根本上防止"用今天修订过的数据
去回测昨天"这类前视泄漏，并从启动日起自然积累**无幸存者偏差**的自有数据（F3 的长期解法）。

核心不变式
----------
- **不可变历史**：任一 `(timestamp, ticker, as_of)` 一旦写入，其值不再改变；对同一交易日的
  数据修订，是以**更晚的 as_of** 追加**新一行**，而非覆盖旧行。
- **as_of 查询**：`load_pit(..., as_of=T)` 对每个 `(timestamp, ticker)` 返回满足
  `as_of ≤ T` 中 **as_of 最大**的那一行（即"在 T 时点能看到的最新数据视角"）。
  `as_of=None` → 返回全局最新 vintage（回测默认）。

存储布局（Hive 年份分区，复用 feature_store 约定）
    {store_dir}/{name}/year={YYYY}/data.parquet
    long-format 列：timestamp, ticker, as_of, <field...>
    幂等键：(timestamp, ticker, as_of) —— 同一 vintage 重跑覆盖同一行，不产生重复。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# 摄取数据的标准 wide 形态：field -> DataFrame(index=DatetimeIndex, columns=ticker)
WidePanel = Dict[str, pd.DataFrame]

_KEY_COLS = ["timestamp", "ticker", "as_of"]


class PITStore:
    """
    时点数据存储。

    Parameters
    ----------
    store_dir : 存储根目录（每个 `name` 一个子目录）。
    """

    def __init__(self, store_dir: Union[str, Path]) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 写入（追加，不可变）
    # ------------------------------------------------------------------

    def append(
        self,
        data: WidePanel,
        as_of: Union[str, pd.Timestamp],
        name: str = "pit",
    ) -> int:
        """
        将一次摄取的 wide 面板按给定 `as_of` vintage **追加**写入。

        Parameters
        ----------
        data  : {field: DataFrame(index=date, columns=ticker)}，即 Dataset.data /
                IngestResult.dataset 的形态。
        as_of : 本次观测的时点（vintage）。字符串（ISO）或 Timestamp 均可。
        name  : 数据集名（子目录）。

        Returns
        -------
        写入的行数（long-format）。同一 `(timestamp,ticker,as_of)` 重复写入不新增行。
        """
        as_of_ts = pd.Timestamp(as_of)
        long_df = self._to_long(data, as_of_ts)
        if long_df.empty:
            logger.warning("[pit_store] append: 面板为空，跳过（name=%s as_of=%s）", name, as_of_ts)
            return 0
        self._append_long(long_df, name)
        logger.info(
            "[pit_store] append '%s' | as_of=%s | %d 行 | %d 交易日 × %d 标的",
            name, as_of_ts.date(), len(long_df),
            long_df["timestamp"].nunique(), long_df["ticker"].nunique(),
        )
        return len(long_df)

    # ------------------------------------------------------------------
    # 读取（时点查询）
    # ------------------------------------------------------------------

    def load_pit(
        self,
        fields: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        as_of: Optional[Union[str, pd.Timestamp]] = None,
        tickers: Optional[List[str]] = None,
        name: str = "pit",
    ) -> WidePanel:
        """
        时点查询，返回 wide 面板 {field: DataFrame(index=date, columns=ticker)}。

        对每个 `(timestamp, ticker)`，返回满足 `as_of ≤ 参数 as_of` 中 as_of 最大的一行；
        `as_of=None` → 全局最新 vintage。数据集不存在时返回 {}。
        """
        long_df = self._load_long(name, start=start, end=end, tickers=tickers)
        if long_df.empty:
            return {}

        if as_of is not None:
            as_of_ts = pd.Timestamp(as_of)
            long_df = long_df[long_df["as_of"] <= as_of_ts]
            if long_df.empty:
                return {}

        # 每个 (timestamp, ticker) 取 as_of 最大的一行（最新可见 vintage）
        latest = (
            long_df.sort_values("as_of")
            .drop_duplicates(subset=["timestamp", "ticker"], keep="last")
            .sort_values(["timestamp", "ticker"])
        )

        avail_fields = [c for c in latest.columns if c not in _KEY_COLS]
        want = [f for f in (fields or avail_fields) if f in avail_fields]

        out: WidePanel = {}
        for f in want:
            wide = latest.pivot(index="timestamp", columns="ticker", values=f)
            wide.index = pd.DatetimeIndex(wide.index)
            wide.columns.name = None
            out[f] = wide.sort_index()
        return out

    def latest_timestamp(self, name: str = "pit") -> Optional[pd.Timestamp]:
        """
        库中该数据集**最新一根 bar 的日期**（Phase 11 增量摄取用：只拉这天之后的数据）。
        只读最后一个 year 分区，不加载全量。空库返回 None。
        """
        dataset_dir = self.store_dir / name
        if not dataset_dir.exists():
            return None
        parts = sorted(dataset_dir.glob("year=*/data.parquet"))
        if not parts:
            return None
        try:
            df = pd.read_parquet(parts[-1], columns=["timestamp"])
            if df.empty:
                return None
            return pd.to_datetime(df["timestamp"]).max().normalize()
        except Exception as exc:
            logger.warning("[pit_store] 读取最新 bar 日期失败 %s: %s", parts[-1], exc)
            return None

    def available_as_of(self, name: str = "pit") -> List[pd.Timestamp]:
        """返回该数据集中出现过的所有 as_of vintage（升序去重）。"""
        long_df = self._load_long(name)
        if long_df.empty:
            return []
        return sorted(pd.to_datetime(long_df["as_of"]).unique().tolist())

    # ------------------------------------------------------------------
    # 内部：long ↔ wide 与分区 IO
    # ------------------------------------------------------------------

    @staticmethod
    def _to_long(data: WidePanel, as_of_ts: pd.Timestamp) -> pd.DataFrame:
        """wide dict → long-format(timestamp, ticker, as_of, <field...>)。"""
        frames: List[pd.DataFrame] = []
        for field, wide in data.items():
            if wide is None or wide.empty:
                continue
            # future_stack=True：pandas 2.2+ 新实现，保留 NA 行且不排序（旧 dropna 参数已废弃）
            s = wide.stack(future_stack=True)     # MultiIndex (date, ticker) -> value
            df = s.rename(field).reset_index()
            df.columns = ["timestamp", "ticker", field]
            frames.append(df.set_index(["timestamp", "ticker"]))
        if not frames:
            return pd.DataFrame(columns=_KEY_COLS)

        merged = pd.concat(frames, axis=1).reset_index()
        merged["timestamp"] = pd.to_datetime(merged["timestamp"])
        merged["ticker"] = merged["ticker"].astype(str)
        merged["as_of"] = as_of_ts
        # 丢弃所有 field 都为 NaN 的行（无信息）
        field_cols = [c for c in merged.columns if c not in _KEY_COLS]
        merged = merged.dropna(subset=field_cols, how="all")
        return merged

    def _append_long(self, long_df: pd.DataFrame, name: str) -> None:
        try:
            import pyarrow as pa  # noqa: F401
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError:
            raise ImportError("pit_store 需要 pyarrow: pip install pyarrow")

        dataset_dir = self.store_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        long_df = long_df.copy()
        long_df["_year"] = long_df["timestamp"].dt.year

        for year, group in long_df.groupby("_year"):
            part_dir = dataset_dir / f"year={year}"
            part_dir.mkdir(parents=True, exist_ok=True)
            out_path = part_dir / "data.parquet"
            group = group.drop(columns=["_year"])

            if out_path.exists():
                try:
                    existing = pd.read_parquet(out_path)
                    existing["timestamp"] = pd.to_datetime(existing["timestamp"])
                    existing["as_of"] = pd.to_datetime(existing["as_of"])
                    group = pd.concat([existing, group], ignore_index=True)
                    # 幂等：同一 (timestamp,ticker,as_of) 只保留最后一次（同 vintage 重跑覆盖）
                    group = group.drop_duplicates(
                        subset=["timestamp", "ticker", "as_of"], keep="last"
                    )
                except Exception as exc:  # 坏分区不静默吞掉，抛出让上层拒绝
                    logger.error("[pit_store] 合并分区失败 %s: %s", out_path, exc)
                    raise

            group = group.sort_values(["timestamp", "ticker", "as_of"]).reset_index(drop=True)
            group.to_parquet(out_path, compression="snappy", index=False)
            logger.debug("[pit_store] 写入 %s (%d 行)", out_path, len(group))

    def _load_long(
        self,
        name: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        dataset_dir = self.store_dir / name
        if not dataset_dir.exists():
            return pd.DataFrame()

        start_dt = pd.Timestamp(start) if start else None
        end_dt = pd.Timestamp(end) if end else None

        frames: List[pd.DataFrame] = []
        for part_dir in sorted(dataset_dir.glob("year=*")):
            try:
                y = int(part_dir.name.split("=")[1])
            except (IndexError, ValueError):
                continue
            if start_dt is not None and y < start_dt.year:
                continue
            if end_dt is not None and y > end_dt.year:
                continue
            part_path = part_dir / "data.parquet"
            if not part_path.exists():
                continue
            try:
                frames.append(pd.read_parquet(part_path))
            except Exception as exc:
                logger.warning("[pit_store] 读取分区失败 %s: %s", part_path, exc)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["as_of"] = pd.to_datetime(df["as_of"])
        df["ticker"] = df["ticker"].astype(str)

        if start_dt is not None:
            df = df[df["timestamp"] >= start_dt]
        if end_dt is not None:
            df = df[df["timestamp"] <= end_dt]
        if tickers:
            up = {t.upper() for t in tickers}
            df = df[df["ticker"].str.upper().isin(up)]
        return df
