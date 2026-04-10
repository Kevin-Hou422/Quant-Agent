"""
数据质量健康检查模块。

三个独立检测器：
- GapDetector       : 找出连续缺失行超过阈值的 ticker
- SpikeDetector     : 找出单日价格涨跌幅超过阈值的异常点
- ZeroVolumeDetector: 找出成交量为零或 NaN 的交易日

DataHealthChecker 组合运行并输出 HealthReport。
overall_score ∈ [0,1]，越高越健康。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HealthReport 数据类
# ---------------------------------------------------------------------------

@dataclass
class HealthReport:
    """数据质量检测结果汇总。"""

    gaps: pd.DataFrame        # columns: ticker, gap_start, gap_end, gap_days
    spikes: pd.DataFrame      # columns: ticker, date, field, pct_change, threshold
    zero_volume: pd.DataFrame # columns: ticker, date, volume
    nan_summary: pd.DataFrame # columns: ticker, field, nan_count, nan_pct
    overall_score: float      # [0, 1]

    # 元信息
    n_tickers: int = 0
    n_dates:   int = 0
    notes:     list = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"HealthReport(score={self.overall_score:.3f}, "
            f"gaps={len(self.gaps)}, spikes={len(self.spikes)}, "
            f"zero_vol={len(self.zero_volume)}, tickers={self.n_tickers})"
        )

    def is_healthy(self, min_score: float = 0.8) -> bool:
        return self.overall_score >= min_score


# ---------------------------------------------------------------------------
# GapDetector
# ---------------------------------------------------------------------------

class GapDetector:
    """
    检测每个 ticker 内连续缺失行超过 min_gap_days 的数据空洞。

    "缺失"定义：该 ticker 在某个日期（全集主索引中存在）的 close 为 NaN。
    """

    def __init__(self, min_gap_days: int = 1) -> None:
        self.min_gap_days = min_gap_days

    def detect(
        self,
        panel: pd.DataFrame,
        date_col: str = "timestamp",
        ticker_col: str = "ticker",
        value_col: str = "close",
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame  columns: [ticker, gap_start, gap_end, gap_days]
        """
        if panel.empty or value_col not in panel.columns:
            return pd.DataFrame(columns=["ticker", "gap_start", "gap_end", "gap_days"])

        rows = []
        for ticker, grp in panel.groupby(ticker_col, sort=False):
            grp = grp.sort_values(date_col)
            is_nan = grp[value_col].isna()
            dates  = grp[date_col].values

            # 找连续 NaN 块
            in_gap     = False
            gap_start  = None
            gap_len    = 0

            for i, (nan_flag, dt) in enumerate(zip(is_nan, dates)):
                if nan_flag:
                    if not in_gap:
                        in_gap    = True
                        gap_start = dt
                        gap_len   = 1
                    else:
                        gap_len += 1
                else:
                    if in_gap and gap_len >= self.min_gap_days:
                        rows.append({
                            "ticker":    ticker,
                            "gap_start": pd.Timestamp(gap_start),
                            "gap_end":   pd.Timestamp(dates[i - 1]),
                            "gap_days":  gap_len,
                        })
                    in_gap  = False
                    gap_len = 0

            # 末尾的空洞
            if in_gap and gap_len >= self.min_gap_days:
                rows.append({
                    "ticker":    ticker,
                    "gap_start": pd.Timestamp(gap_start),
                    "gap_end":   pd.Timestamp(dates[-1]),
                    "gap_days":  gap_len,
                })

        return pd.DataFrame(rows, columns=["ticker", "gap_start", "gap_end", "gap_days"])


# ---------------------------------------------------------------------------
# SpikeDetector
# ---------------------------------------------------------------------------

class SpikeDetector:
    """
    检测单日价格涨跌幅超过 threshold 的异常点。
    逐 ticker 计算 pct_change，绝对值 > threshold 则标记。
    """

    def __init__(
        self,
        threshold: float = 0.50,
        price_fields: list[str] | None = None,
    ) -> None:
        self.threshold    = threshold
        self.price_fields = price_fields or ["close"]

    def detect(
        self,
        panel: pd.DataFrame,
        date_col: str = "timestamp",
        ticker_col: str = "ticker",
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame  columns: [ticker, date, field, pct_change, threshold]
        """
        rows = []
        for field in self.price_fields:
            if field not in panel.columns:
                continue
            for ticker, grp in panel.groupby(ticker_col, sort=False):
                grp = grp.sort_values(date_col)
                pct = grp[field].pct_change(fill_method=None)
                spikes = grp.loc[pct.abs() > self.threshold, date_col]
                for dt in spikes:
                    idx = grp[grp[date_col] == dt].index[0]
                    rows.append({
                        "ticker":     ticker,
                        "date":       pd.Timestamp(dt),
                        "field":      field,
                        "pct_change": round(float(pct.loc[idx]), 6),
                        "threshold":  self.threshold,
                    })

        return pd.DataFrame(
            rows,
            columns=["ticker", "date", "field", "pct_change", "threshold"],
        )


# ---------------------------------------------------------------------------
# ZeroVolumeDetector
# ---------------------------------------------------------------------------

class ZeroVolumeDetector:
    """
    检测成交量为零或 NaN 的交易日。
    零成交量可能意味着停牌、数据错误或非交易日。
    """

    def detect(
        self,
        panel: pd.DataFrame,
        date_col: str = "timestamp",
        ticker_col: str = "ticker",
        volume_col: str = "volume",
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame  columns: [ticker, date, volume]
        """
        if volume_col not in panel.columns:
            return pd.DataFrame(columns=["ticker", "date", "volume"])

        mask = panel[volume_col].isna() | (panel[volume_col] == 0)
        flagged = panel.loc[mask, [ticker_col, date_col, volume_col]].copy()
        flagged.columns = ["ticker", "date", "volume"]
        return flagged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# DataHealthChecker（组合检测器）
# ---------------------------------------------------------------------------

class DataHealthChecker:
    """
    组合运行所有检测器并计算 HealthReport。

    overall_score 计算逻辑：
        gap_score        = max(0, 1 - n_gap_tickers / n_tickers)
        spike_score      = max(0, 1 - n_spike_dates / total_rows)
        zero_vol_score   = max(0, 1 - n_zero_vol_rows / total_rows)
        nan_score        = max(0, 1 - global_nan_pct)
        overall_score    = mean(以上四项)
    """

    def __init__(
        self,
        gap_min_days: int = 1,
        spike_threshold: float = 0.50,
        spike_fields: list[str] | None = None,
    ) -> None:
        self._gap_det  = GapDetector(min_gap_days=gap_min_days)
        self._spk_det  = SpikeDetector(
            threshold=spike_threshold,
            price_fields=spike_fields or ["close"],
        )
        self._zvol_det = ZeroVolumeDetector()

    def check(
        self,
        panel: pd.DataFrame,
        date_col: str = "timestamp",
        ticker_col: str = "ticker",
    ) -> HealthReport:
        """
        对 long-format 面板执行全量数据质量检测。

        Returns
        -------
        HealthReport
        """
        if panel.empty:
            return HealthReport(
                gaps=pd.DataFrame(), spikes=pd.DataFrame(),
                zero_volume=pd.DataFrame(), nan_summary=pd.DataFrame(),
                overall_score=0.0,
            )

        n_tickers = panel[ticker_col].nunique()
        n_dates   = panel[date_col].nunique()
        total_rows = len(panel)

        # 1. Gap 检测
        gaps = self._gap_det.detect(panel, date_col=date_col, ticker_col=ticker_col)

        # 2. Spike 检测
        spikes = self._spk_det.detect(panel, date_col=date_col, ticker_col=ticker_col)

        # 3. Zero Volume 检测
        zero_vol = self._zvol_det.detect(panel, date_col=date_col, ticker_col=ticker_col)

        # 4. NaN 汇总
        nan_summary = self._nan_summary(panel, date_col, ticker_col)

        # 5. overall_score
        numeric_cols = panel.select_dtypes(include="number").columns
        global_nan_pct = (
            panel[numeric_cols].isna().sum().sum()
            / max(total_rows * len(numeric_cols), 1)
        )

        gap_tickers   = gaps["ticker"].nunique() if not gaps.empty else 0
        spike_rows    = len(spikes)
        zero_vol_rows = len(zero_vol)

        gap_score     = max(0.0, 1.0 - gap_tickers   / max(n_tickers, 1))
        spike_score   = max(0.0, 1.0 - spike_rows    / max(total_rows, 1))
        zvol_score    = max(0.0, 1.0 - zero_vol_rows / max(total_rows, 1))
        nan_score     = max(0.0, 1.0 - global_nan_pct)

        overall = float(np.mean([gap_score, spike_score, zvol_score, nan_score]))

        report = HealthReport(
            gaps=gaps,
            spikes=spikes,
            zero_volume=zero_vol,
            nan_summary=nan_summary,
            overall_score=round(overall, 4),
            n_tickers=n_tickers,
            n_dates=n_dates,
        )

        logger.info(
            "HealthReport: score=%.3f | gaps=%d | spikes=%d | "
            "zero_vol=%d | nan_pct=%.2f%%",
            overall, len(gaps), spike_rows, zero_vol_rows,
            global_nan_pct * 100,
        )
        return report

    def to_html(self, report: HealthReport) -> str:
        """输出 HTML 格式的健康报告（用于 Jupyter / Web 展示）。"""
        parts = [
            "<h2>数据健康报告</h2>",
            f"<p><b>综合评分：{report.overall_score:.3f}</b> "
            f"({'健康' if report.is_healthy() else '需关注'})</p>",
            f"<p>覆盖 {report.n_tickers} 个 ticker，{report.n_dates} 个交易日</p>",
            "<h3>空洞检测（Gaps）</h3>",
            report.gaps.to_html(index=False) if not report.gaps.empty
            else "<p>无异常</p>",
            "<h3>价格异常（Spikes > {:.0%}）</h3>".format(
                self._spk_det.threshold
            ),
            report.spikes.to_html(index=False) if not report.spikes.empty
            else "<p>无异常</p>",
            "<h3>零成交量</h3>",
            report.zero_volume.to_html(index=False) if not report.zero_volume.empty
            else "<p>无异常</p>",
            "<h3>NaN 汇总</h3>",
            report.nan_summary.to_html(index=False) if not report.nan_summary.empty
            else "<p>无缺失</p>",
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------------

    @staticmethod
    def _nan_summary(
        panel: pd.DataFrame,
        date_col: str,
        ticker_col: str,
    ) -> pd.DataFrame:
        """按 ticker × field 统计 NaN 数量与比例。"""
        numeric_cols = [
            c for c in panel.select_dtypes(include="number").columns
            if c not in (date_col,)
        ]
        rows = []
        for ticker, grp in panel.groupby(ticker_col, sort=False):
            n = len(grp)
            for col in numeric_cols:
                nan_cnt = int(grp[col].isna().sum())
                if nan_cnt > 0:
                    rows.append({
                        "ticker":    ticker,
                        "field":     col,
                        "nan_count": nan_cnt,
                        "nan_pct":   round(nan_cnt / n * 100, 2),
                    })
        return pd.DataFrame(rows, columns=["ticker", "field", "nan_count", "nan_pct"])
