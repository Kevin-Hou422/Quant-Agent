"""
面板工厂（Panel Factory）

职责：
1. reindex_to_master —— 将多个 long-format DataFrame 对齐到全集
   (date × ticker) 笛卡尔积主索引，退市/停牌的行以 NaN 保留。
2. UniverseFilter —— 基于 Point-in-Time 的活跃 ticker 过滤，
   彻底消除生存偏差（survivorship bias）。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PanelFactory
# ---------------------------------------------------------------------------

class PanelFactory:
    """
    将来自不同数据源的 long-format DataFrame 列表合并为统一面板。

    所有 ticker 在全集日期轴上都有行（缺失时为 NaN），确保：
    - 横截面运算（rank, zscore 等）不会因 index 不对齐而报错
    - 退市/停牌不会从面板中消失，历史数据完整保留

    Parameters
    ----------
    date_col   : 时间列名（默认 "timestamp"）
    ticker_col : ticker 列名（默认 "ticker"）
    freq       : 日期频率，用于生成主索引（默认 "B" = 工作日）
    """

    def __init__(
        self,
        date_col: str = "timestamp",
        ticker_col: str = "ticker",
        freq: str = "B",
    ) -> None:
        self.date_col   = date_col
        self.ticker_col = ticker_col
        self.freq       = freq

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def reindex_to_master(
        self,
        dfs: List[pd.DataFrame],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        将多个 long-format DataFrame 对齐到全集 (date × ticker) 主索引。

        Parameters
        ----------
        dfs   : long-format DataFrame 列表（每个含 timestamp + ticker 列）
        start : 主索引起始日期（字符串），None 则取数据最小日期
        end   : 主索引终止日期（字符串），None 则取数据最大日期

        Returns
        -------
        pd.DataFrame  long-format，行数 = |全集日期| × |全集 ticker|
                      缺失组合以 NaN 填充（ticker 列和 timestamp 列有值，其余 NaN）
        """
        if not dfs:
            raise ValueError("dfs 列表不能为空")

        # 1. 合并所有数据
        combined = pd.concat(dfs, ignore_index=True)
        combined[self.date_col] = pd.to_datetime(combined[self.date_col])
        combined[self.ticker_col] = combined[self.ticker_col].str.upper()

        # 2. 确定全集日期轴
        data_start = combined[self.date_col].min()
        data_end   = combined[self.date_col].max()
        master_start = pd.Timestamp(start) if start else data_start
        master_end   = pd.Timestamp(end)   if end   else data_end

        master_dates   = pd.date_range(master_start, master_end, freq=self.freq)
        master_tickers = sorted(combined[self.ticker_col].unique())

        logger.info(
            "PanelFactory: 全集日期 %d 个，全集 ticker %d 个，"
            "预期行数 %d",
            len(master_dates), len(master_tickers),
            len(master_dates) * len(master_tickers),
        )

        # 3. 建立笛卡尔积主索引
        master_idx = pd.MultiIndex.from_product(
            [master_dates, master_tickers],
            names=[self.date_col, self.ticker_col],
        )
        master_df = pd.DataFrame(index=master_idx).reset_index()

        # 4. 左连接（保留全集行，缺失字段为 NaN）
        combined = combined.drop_duplicates(
            subset=[self.date_col, self.ticker_col], keep="last"
        )
        panel = master_df.merge(
            combined,
            on=[self.date_col, self.ticker_col],
            how="left",
        )

        panel = panel.sort_values(
            [self.date_col, self.ticker_col]
        ).reset_index(drop=True)

        nan_pct = panel.drop(
            columns=[self.date_col, self.ticker_col]
        ).isna().mean().mean() * 100
        logger.info("PanelFactory: 面板 NaN 率 %.2f%%", nan_pct)

        return panel

    def wide_to_long(
        self,
        wide_dict: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        将 wide-format RawDataset（field → time×asset DataFrame）
        转为 long-format 面板后调用 reindex_to_master。
        """
        from .schema import wide_to_long as _w2l
        long_df = _w2l(wide_dict)
        return self.reindex_to_master([long_df])


# ---------------------------------------------------------------------------
# UniverseFilter  ——  PIT 活跃 ticker 过滤
# ---------------------------------------------------------------------------

class UniverseFilter:
    """
    基于 Point-in-Time（时间点）的 Universe 过滤器。

    universe_df 格式（long-format）：
        date   : datetime64   某日期
        ticker : str          资产代码
        active : bool         该日期该 ticker 是否处于活跃状态

    设计原则
    --------
    - 只查询"截止 as_of 日期"的活跃状态，不使用未来信息
    - 从未出现在 universe_df 中的 ticker → 视为始终活跃（宽松模式）
      或始终不活跃（严格模式 strict=True）
    """

    def __init__(
        self,
        universe_df: Optional[pd.DataFrame] = None,
        date_col: str = "date",
        ticker_col: str = "ticker",
        active_col: str = "active",
        strict: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        universe_df : PIT Universe DataFrame（可选，None = 无过滤）
        strict      : True = 未出现在 universe_df 的 ticker 视为不活跃
        """
        self.date_col   = date_col
        self.ticker_col = ticker_col
        self.active_col = active_col
        self.strict     = strict
        self._universe: Optional[pd.DataFrame] = None

        if universe_df is not None:
            self._load(universe_df)

    def _load(self, df: pd.DataFrame) -> None:
        df = df.copy()
        df[self.date_col]   = pd.to_datetime(df[self.date_col])
        df[self.ticker_col] = df[self.ticker_col].str.upper()
        df[self.active_col] = df[self.active_col].astype(bool)
        self._universe = df.sort_values(self.date_col)

    def filter(
        self,
        panel: pd.DataFrame,
        as_of: str,
        date_col: str = "timestamp",
        ticker_col: str = "ticker",
    ) -> pd.DataFrame:
        """
        保留在 as_of 日期仍处于活跃状态的 ticker 的所有行。

        Parameters
        ----------
        panel     : long-format 面板
        as_of     : 时间截面（字符串），用于查询活跃状态
        date_col  : 面板中的时间列名
        ticker_col: 面板中的 ticker 列名

        Returns
        -------
        pd.DataFrame  过滤后的面板（保留历史行但只含活跃 ticker）
        """
        if self._universe is None:
            # 无 Universe 定义 → 返回原始面板
            return panel

        as_of_ts = pd.Timestamp(as_of)

        # 找到 ≤ as_of 最新的 active 状态
        hist = self._universe[self._universe[self.date_col] <= as_of_ts]
        if hist.empty:
            if self.strict:
                return panel.head(0)  # 无历史 → 严格模式返回空
            return panel

        # 每个 ticker 取最新一条记录
        latest = (
            hist.sort_values(self.date_col)
            .groupby(self.ticker_col, as_index=False)
            .last()
        )
        active_tickers = set(
            latest.loc[latest[self.active_col], self.ticker_col].str.upper()
        )

        if not self.strict:
            # 宽松模式：universe_df 未覆盖的 ticker 视为活跃
            all_panel_tickers = set(panel[ticker_col].str.upper().unique())
            universe_tickers  = set(self._universe[self.ticker_col].str.upper().unique())
            not_in_universe   = all_panel_tickers - universe_tickers
            active_tickers   |= not_in_universe

        mask = panel[ticker_col].str.upper().isin(active_tickers)
        filtered = panel.loc[mask].copy()

        removed = len(set(panel[ticker_col].unique())) - len(active_tickers)
        if removed > 0:
            logger.info(
                "UniverseFilter: as_of=%s 过滤掉 %d 个非活跃 ticker",
                as_of, removed,
            )
        return filtered

    @classmethod
    def from_date_ranges(
        cls,
        ranges: List[dict],
        **kwargs,
    ) -> "UniverseFilter":
        """
        从日期区间列表构建 UniverseFilter。

        Parameters
        ----------
        ranges : list of dict，每个 dict 含：
                 - ticker   : str
                 - ipo_date : str  上市/纳入日期
                 - delist_date : str | None  退市日期（None = 仍活跃）
        """
        rows = []
        for item in ranges:
            ticker     = item["ticker"].upper()
            ipo        = pd.Timestamp(item["ipo_date"])
            delist     = pd.Timestamp(item["delist_date"]) if item.get("delist_date") else None
            end_date   = delist or pd.Timestamp.today()

            for date in pd.date_range(ipo, end_date, freq="B"):
                rows.append({
                    "date":   date,
                    "ticker": ticker,
                    "active": delist is None or date <= delist,
                })

        universe_df = pd.DataFrame(rows)
        return cls(universe_df=universe_df, **kwargs)
