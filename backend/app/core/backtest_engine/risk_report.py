"""
RiskReport — 回测绩效汇总数据类

聚合 PerformanceAnalyzer 的所有输出，提供：
  - 结构化字段（可序列化）
  - summary() 文本摘要（适合打印/日志）
  - to_dict() JSON 序列化
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

from .backtest_engine import BacktestResult
from .performance_analyzer import PerformanceAnalyzer


# ---------------------------------------------------------------------------
# RiskReport
# ---------------------------------------------------------------------------

@dataclass
class RiskReport:
    """完整绩效风险报告。"""

    # ---- 收益 ----
    annualized_return:  float = np.nan
    annualized_vol:     float = np.nan
    sharpe_ratio:       float = np.nan
    calmar_ratio:       float = np.nan

    # ---- 风险 ----
    max_drawdown:       float = np.nan
    max_dd_start:       Optional[object] = None
    max_dd_end:         Optional[object] = None
    max_dd_duration:    int   = 0
    sortino_ratio:      float = np.nan
    var_95:             float = np.nan
    cvar_95:            float = np.nan

    # ---- Alpha 质量 ----
    mean_ic:            float = np.nan
    ic_ir:              float = np.nan
    ann_turnover:       float = np.nan
    cost_drag_bps:      float = np.nan

    # ---- 分档（可选）----
    decile_returns:     Optional[pd.Series] = field(default=None, repr=False)

    # ---- 供可视化的原始序列 ----
    equity_curve:       Optional[pd.Series] = field(default=None, repr=False)
    gross_returns:      Optional[pd.Series] = field(default=None, repr=False)
    net_returns:        Optional[pd.Series] = field(default=None, repr=False)
    rolling_sharpe:     Optional[pd.Series] = field(default=None, repr=False)
    rolling_ic:         Optional[pd.Series] = field(default=None, repr=False)
    drawdown_series:    Optional[pd.Series] = field(default=None, repr=False)

    # ---- 元信息 ----
    n_days:             int   = 0
    n_assets:           int   = 0
    initial_capital:    float = 1_000_000.0

    # ------------------------------------------------------------------
    # 工厂方法（从 BacktestResult 直接构建）
    # ------------------------------------------------------------------

    @classmethod
    def from_result(
        cls,
        result: BacktestResult,
        prices: Optional[pd.DataFrame] = None,
        rf: float = 0.0,
        rolling_sharpe_window: int = 60,
        rolling_ic_window: int = 20,
    ) -> "RiskReport":
        """
        Parameters
        ----------
        result : BacktestResult
        prices : 资产价格矩阵（传入后 IC 和 Decile 分析精确计算）
        rf     : 无风险日收益率
        """
        pa = PerformanceAnalyzer(result, rf=rf)
        metrics = pa.summarize(prices=prices)

        # 精确 IC（若有价格）
        if prices is not None:
            ic_series = pa.rolling_rank_ic_from_prices(result.signal, prices, rolling_ic_window)
        else:
            ic_series = pa.rolling_rank_ic(rolling_ic_window)

        metrics["mean_ic"] = float(ic_series.mean()) if len(ic_series) else np.nan
        metrics["ic_ir"]   = pa.ic_ir(ic_series)

        # 分档分析
        decile_ret = pa.decile_analysis(prices=prices)

        # 滚动序列
        rs = pa.rolling_sharpe(rolling_sharpe_window)
        dd = pa.drawdown_series()

        return cls(
            annualized_return  = metrics["annualized_return"],
            annualized_vol     = metrics["annualized_vol"],
            sharpe_ratio       = metrics["sharpe_ratio"],
            calmar_ratio       = metrics["calmar_ratio"],
            max_drawdown       = metrics["max_drawdown"],
            max_dd_start       = metrics["max_dd_start"],
            max_dd_end         = metrics["max_dd_end"],
            max_dd_duration    = int(metrics["max_dd_duration"]),
            sortino_ratio      = metrics["sortino_ratio"],
            var_95             = metrics["var_95"],
            cvar_95            = metrics["cvar_95"],
            mean_ic            = metrics["mean_ic"],
            ic_ir              = metrics["ic_ir"],
            ann_turnover       = metrics["ann_turnover"],
            cost_drag_bps      = metrics["cost_drag_bps"],
            decile_returns     = decile_ret,
            equity_curve       = result.equity_curve,
            gross_returns      = result.gross_returns,
            net_returns        = result.net_returns,
            rolling_sharpe     = rs,
            rolling_ic         = ic_series,
            drawdown_series    = dd,
            n_days             = len(result.equity_curve),
            n_assets           = result.positions.shape[1],
        )

    # ------------------------------------------------------------------
    # 文本摘要
    # ------------------------------------------------------------------

    def summary(self) -> str:
        def _fmt(v, fmt=".4f", pct=False) -> str:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            if pct:
                return f"{v * 100:{fmt}}%"
            return f"{v:{fmt}}"

        lines = [
            "=" * 52,
            "  回测绩效摘要 (Backtest Performance Summary)",
            "=" * 52,
            f"  样本天数        : {self.n_days}  |  资产数 : {self.n_assets}",
            "-" * 52,
            "  【收益指标】",
            f"  年化收益率      : {_fmt(self.annualized_return, pct=True)}",
            f"  年化波动率      : {_fmt(self.annualized_vol,     pct=True)}",
            f"  夏普比率        : {_fmt(self.sharpe_ratio)}",
            f"  卡玛比率        : {_fmt(self.calmar_ratio)}",
            "-" * 52,
            "  【风险指标】",
            f"  最大回撤        : {_fmt(self.max_drawdown, pct=True)}",
            f"  最大回撤持续    : {self.max_dd_duration} 天",
            f"  Sortino 比率    : {_fmt(self.sortino_ratio)}",
            f"  VaR (95%)       : {_fmt(self.var_95, pct=True)}",
            f"  CVaR (95%)      : {_fmt(self.cvar_95, pct=True)}",
            "-" * 52,
            "  【Alpha 质量】",
            f"  平均 Rank IC    : {_fmt(self.mean_ic)}",
            f"  IC IR           : {_fmt(self.ic_ir)}",
            f"  年化换手率      : {_fmt(self.ann_turnover, pct=True)}",
            f"  年化成本拖累    : {_fmt(self.cost_drag_bps, '.2f')} bps",
            "=" * 52,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """返回 JSON 可序列化字典（排除 pd.Series 字段）。"""
        excluded = {
            "decile_returns", "equity_curve", "gross_returns",
            "net_returns", "rolling_sharpe", "rolling_ic", "drawdown_series",
        }
        d = {}
        for f_name, f_val in self.__dataclass_fields__.items():
            if f_name in excluded:
                continue
            val = getattr(self, f_name)
            if isinstance(val, (pd.Timestamp, pd.Period)):
                val = str(val)
            elif isinstance(val, float) and np.isnan(val):
                val = None
            d[f_name] = val
        return d
