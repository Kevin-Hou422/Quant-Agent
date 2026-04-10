"""
数据预处理管道。

三个独立组件，可单独使用，也可通过 Preprocessor 组合运行：

1. CorporateActionAdjuster  —— 用 adj_factor 对 OHLC 复权
2. SyntheticFieldBuilder    —— 合成 vwap / returns / log_returns
3. MissingValueStrategy     —— ffill 并统计剩余 NaN

面板格式约定（long-format）：
    每行 = 一个 ticker 在一个时间点，列含 timestamp / ticker / OHLCV / ...
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. CorporateActionAdjuster
# ---------------------------------------------------------------------------

class CorporateActionAdjuster:
    """
    前复权：将 OHLC 价格乘以 adj_factor。

    adj_factor 保留原始值不变，额外新增 adj_close 列（= close × adj_factor）。
    Volume 不做复权（股数不受复权影响）。
    """

    PRICE_COLS = ["open", "high", "low", "close"]

    def adjust(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        panel : long-format DataFrame，必须含 adj_factor 列

        Returns
        -------
        pd.DataFrame  OHLC 已乘以 adj_factor，新增 adj_close 列
        """
        if panel.empty:
            return panel

        if "adj_factor" not in panel.columns:
            warnings.warn(
                "CorporateActionAdjuster: 面板缺少 adj_factor 列，跳过复权",
                stacklevel=2,
            )
            return panel

        panel = panel.copy()
        factor = panel["adj_factor"].fillna(1.0)

        for col in self.PRICE_COLS:
            if col in panel.columns:
                panel[col] = panel[col] * factor

        # 记录前复权后的收盘价
        if "close" in panel.columns:
            panel["adj_close"] = panel["close"]  # 已经是复权后的值

        n_adjusted = (panel["adj_factor"] != 1.0).sum()
        logger.debug("CorporateActionAdjuster: %d 行执行了复权", n_adjusted)
        return panel


# ---------------------------------------------------------------------------
# 2. SyntheticFieldBuilder
# ---------------------------------------------------------------------------

class SyntheticFieldBuilder:
    """
    计算合成字段：

    - vwap        : (high + low + close) / 3  当 vwap 列全为 NaN 时
    - returns     : 每个 ticker 内 close 的日收益率（pct_change）
    - log_returns : log(close_t / close_{t-1})，每个 ticker 独立计算
    """

    def build(
        self,
        panel: pd.DataFrame,
        date_col: str = "timestamp",
        ticker_col: str = "ticker",
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        panel : long-format DataFrame

        Returns
        -------
        pd.DataFrame  新增 vwap（如需）、returns、log_returns 列
        """
        if panel.empty:
            return panel

        panel = panel.copy()
        panel = panel.sort_values([ticker_col, date_col])

        # ---- vwap ----
        if "vwap" not in panel.columns or panel["vwap"].isna().all():
            req = {"high", "low", "close"}
            if req.issubset(panel.columns):
                panel["vwap"] = (
                    panel["high"] + panel["low"] + panel["close"]
                ) / 3.0
                logger.debug("SyntheticFieldBuilder: 合成 vwap")
            else:
                panel["vwap"] = np.nan

        # ---- returns & log_returns（按 ticker 分组计算）----
        close = panel["close"] if "close" in panel.columns else None

        if close is not None:
            grp_close = panel.groupby(ticker_col, sort=False)["close"]

            panel["returns"] = grp_close.transform(
                lambda s: s.pct_change(fill_method=None)
            )
            panel["log_returns"] = grp_close.transform(
                lambda s: np.log(s / s.shift(1))
            )
        else:
            panel["returns"]     = np.nan
            panel["log_returns"] = np.nan

        return panel


# ---------------------------------------------------------------------------
# 3. MissingValueStrategy
# ---------------------------------------------------------------------------

class MissingValueStrategy:
    """
    缺失值处理策略：

    1. 按 ticker 内时间顺序 ffill（上限 ffill_limit 个连续交易日）
    2. 统计剩余 NaN 情况，返回报告字典
    """

    def apply(
        self,
        panel: pd.DataFrame,
        ffill_limit: int = 5,
        date_col: str = "timestamp",
        ticker_col: str = "ticker",
        report: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Parameters
        ----------
        panel       : long-format DataFrame
        ffill_limit : 最大前向填充天数
        report      : 是否计算 NaN 报告

        Returns
        -------
        (cleaned_panel, nan_report)
        nan_report = {field: {ticker: nan_count, ...}, ...}
        """
        if panel.empty:
            return panel, {}

        panel = panel.copy().sort_values([ticker_col, date_col])

        numeric_cols = panel.select_dtypes(include="number").columns.tolist()

        # 按 ticker 分组 ffill
        def _ffill_group(grp: pd.DataFrame) -> pd.DataFrame:
            grp[numeric_cols] = grp[numeric_cols].ffill(limit=ffill_limit)
            return grp

        panel = (
            panel
            .groupby(ticker_col, sort=False, group_keys=False)
            .apply(_ffill_group)
            .reset_index(drop=True)
        )

        # 构建 NaN 报告
        nan_report: Dict = {}
        if report:
            for col in numeric_cols:
                nan_counts = (
                    panel.groupby(ticker_col)[col]
                    .apply(lambda s: s.isna().sum())
                    .to_dict()
                )
                if any(v > 0 for v in nan_counts.values()):
                    nan_report[col] = nan_counts

            total_nan  = panel[numeric_cols].isna().sum().sum()
            total_cells = len(panel) * len(numeric_cols)
            nan_report["_summary"] = {
                "total_nan":   int(total_nan),
                "total_cells": int(total_cells),
                "nan_pct":     round(total_nan / total_cells * 100, 4) if total_cells else 0,
                "ffill_limit": ffill_limit,
            }
            logger.info(
                "MissingValueStrategy: ffill_limit=%d, 剩余 NaN %.4f%%",
                ffill_limit, nan_report["_summary"]["nan_pct"],
            )

        return panel, nan_report


# ---------------------------------------------------------------------------
# 4. Preprocessor（组合三步）
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    完整预处理管道：复权 → 合成字段 → 缺失值填充。

    可通过参数灵活启停各步骤。
    """

    def __init__(
        self,
        date_col: str = "timestamp",
        ticker_col: str = "ticker",
    ) -> None:
        self.date_col   = date_col
        self.ticker_col = ticker_col
        self._adjuster  = CorporateActionAdjuster()
        self._synth     = SyntheticFieldBuilder()
        self._mv        = MissingValueStrategy()

    def run(
        self,
        panel: pd.DataFrame,
        apply_adj: bool = True,
        build_synthetic: bool = True,
        ffill_limit: int = 5,
        report_nan: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        依次执行预处理步骤。

        Parameters
        ----------
        panel           : long-format 原始面板
        apply_adj       : 是否执行复权
        build_synthetic : 是否合成 vwap / returns / log_returns
        ffill_limit     : ffill 最大天数（0 = 不 ffill）
        report_nan      : 是否生成 NaN 报告

        Returns
        -------
        (processed_panel, nan_report)
        """
        if panel.empty:
            return panel, {}

        if apply_adj:
            panel = self._adjuster.adjust(panel)

        if build_synthetic:
            panel = self._synth.build(
                panel,
                date_col=self.date_col,
                ticker_col=self.ticker_col,
            )

        panel, nan_report = self._mv.apply(
            panel,
            ffill_limit=ffill_limit,
            date_col=self.date_col,
            ticker_col=self.ticker_col,
            report=report_nan,
        )

        return panel, nan_report
