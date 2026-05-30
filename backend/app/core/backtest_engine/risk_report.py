"""
RiskReport — 回测绩效汇总数据类

聚合 PerformanceAnalyzer 的所有输出，提供：
  - 结构化字段（可序列化）
  - summary() 文本摘要（适合打印/日志）
  - to_dict() JSON 序列化
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

    # ---- 统计检验 ----
    sharpe_tstat:       float = np.nan   # Lo (2002) t 统计量；> 1.96 为 5% 显著
    ic_method:          str   = ""       # "exact_price" | "approx_position" | "provided"

    # ---- O4: 多空腿分离 ----
    long_ann_return:    float = np.nan
    long_sharpe:        float = np.nan
    short_ann_return:   float = np.nan
    short_sharpe:       float = np.nan

    # ---- F8: 基准分解 ----
    benchmark_beta:     float = np.nan
    benchmark_alpha:    float = np.nan
    benchmark_ann_ret:  float = np.nan
    tracking_error:     float = np.nan
    information_ratio:  float = np.nan

    # ---- O2: 压力测试 ----
    stress_test:        Optional[dict] = field(default=None, repr=False)

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
        result:            BacktestResult,
        prices:            Optional[pd.DataFrame] = None,
        rf:                float = 0.0,
        rf_annual:         float = 0.0,
        rolling_sharpe_window: int = 60,
        rolling_ic_window: int = 20,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> "RiskReport":
        """
        Parameters
        ----------
        result            : BacktestResult
        prices            : 资产价格矩阵（传入后 IC 和 Decile 分析精确计算）
        rf                : 无风险日收益率（优先使用 rf_annual）
        rf_annual         : 年化无风险利率（如 0.05 = 5%），推荐使用此参数
        rolling_ic_window : 仅用于 rolling_ic 序列的平滑窗口（不影响 mean_ic/ic_ir）
        benchmark_returns : 基准日收益率序列（F8：alpha/beta 分解用）
        """
        pa = PerformanceAnalyzer(result, rf=rf, rf_annual=rf_annual)
        # summarize() 内部已根据 prices 选择正确 IC 方法并返回 ic_method
        metrics = pa.summarize(prices=prices)

        # rolling_ic：对逐日 IC 应用滚动均值（用于可视化），不影响 mean_ic/ic_ir
        ic_method = metrics.get("ic_method", "approx_position")
        if prices is not None:
            raw_ic = pa.rolling_rank_ic_from_prices(result.signal, prices)
        else:
            raw_ic = pa.rolling_rank_ic()
        rolling_ic = (
            raw_ic.rolling(rolling_ic_window, min_periods=max(2, rolling_ic_window // 2)).mean()
            if len(raw_ic) >= 2 else raw_ic
        )

        # 分档分析
        decile_ret = pa.decile_analysis(prices=prices)

        # F8: 基准 alpha/beta 分解（仅当 benchmark_returns 传入时计算）
        bench_metrics: dict = {}
        if benchmark_returns is not None:
            bench_metrics = pa.benchmark_analysis(benchmark_returns)

        # O2: 压力测试子区间分析
        stress = pa.stress_test()

        # 滚动序列
        rs = pa.rolling_sharpe(rolling_sharpe_window)
        dd = pa.drawdown_series()

        return cls(
            annualized_return  = metrics["annualized_return"],
            annualized_vol     = metrics["annualized_vol"],
            sharpe_ratio       = metrics["sharpe_ratio"],
            sharpe_tstat       = metrics["sharpe_tstat"],
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
            ic_method          = ic_method,
            long_ann_return    = metrics.get("long_ann_return",  np.nan),
            long_sharpe        = metrics.get("long_sharpe",      np.nan),
            short_ann_return   = metrics.get("short_ann_return", np.nan),
            short_sharpe       = metrics.get("short_sharpe",     np.nan),
            benchmark_beta     = bench_metrics.get("benchmark_beta",    np.nan),
            benchmark_alpha    = bench_metrics.get("benchmark_alpha",   np.nan),
            benchmark_ann_ret  = bench_metrics.get("benchmark_ann_ret", np.nan),
            tracking_error     = bench_metrics.get("tracking_error",    np.nan),
            information_ratio  = bench_metrics.get("information_ratio", np.nan),
            stress_test        = stress,
            decile_returns     = decile_ret,
            equity_curve       = result.equity_curve,
            gross_returns      = result.gross_returns,
            net_returns        = result.net_returns,
            rolling_sharpe     = rs,
            rolling_ic         = rolling_ic,
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

        sig_flag = (
            " ✓显著" if (not np.isnan(self.sharpe_tstat) and self.sharpe_tstat > 1.96)
            else " ✗不显著" if not np.isnan(self.sharpe_tstat)
            else ""
        )
        ruin_line = (
            [f"  熔断日期        : {self.max_dd_end} (净值归零)"]
            if not np.isnan(self.max_drawdown) and self.max_drawdown <= -0.999
            else []
        )
        lines = [
            "=" * 52,
            "  回测绩效摘要 (Backtest Performance Summary)",
            "=" * 52,
            f"  样本天数        : {self.n_days}  |  资产数 : {self.n_assets}",
            "-" * 52,
            "  【收益指标】",
            f"  年化收益率      : {_fmt(self.annualized_return, pct=True)}",
            f"  年化波动率      : {_fmt(self.annualized_vol,     pct=True)}",
            f"  夏普比率        : {_fmt(self.sharpe_ratio)}  (t={_fmt(self.sharpe_tstat, '.2f')}{sig_flag})",
            f"  卡玛比率        : {_fmt(self.calmar_ratio)}",
            "-" * 52,
            "  【风险指标】",
            f"  最大回撤        : {_fmt(self.max_drawdown, pct=True)}",
            f"  最大回撤持续    : {self.max_dd_duration} 天",
            f"  Sortino 比率    : {_fmt(self.sortino_ratio)}",
            f"  VaR (95%)       : {_fmt(self.var_95, pct=True)}",
            f"  CVaR (95%)      : {_fmt(self.cvar_95, pct=True)}",
            *ruin_line,
            "-" * 52,
            "  【Alpha 质量】",
            f"  平均 Rank IC    : {_fmt(self.mean_ic)}  [{self.ic_method or 'unknown'}]",
            f"  IC IR           : {_fmt(self.ic_ir)}",
            f"  年化换手率      : {_fmt(self.ann_turnover, pct=True)}",
            f"  年化成本拖累    : {_fmt(self.cost_drag_bps, '.2f')} bps",
            "-" * 52,
            "  【多空腿分解】",
            f"  多头年化收益    : {_fmt(self.long_ann_return,  pct=True)}  "
            f"Sharpe={_fmt(self.long_sharpe)}",
            f"  空头年化收益    : {_fmt(self.short_ann_return, pct=True)}  "
            f"Sharpe={_fmt(self.short_sharpe)}",
        ]

        # F8: 基准分解（仅当有基准数据时显示）
        if not np.isnan(self.benchmark_beta):
            lines += [
                "-" * 52,
                "  【基准 Alpha/Beta 分解】",
                f"  Beta            : {_fmt(self.benchmark_beta)}",
                f"  Alpha（年化）   : {_fmt(self.benchmark_alpha, pct=True)}",
                f"  基准收益（年化）: {_fmt(self.benchmark_ann_ret, pct=True)}",
                f"  跟踪误差        : {_fmt(self.tracking_error,  pct=True)}",
                f"  信息比率        : {_fmt(self.information_ratio)}",
            ]

        # O2: 压力测试（仅显示核心数字，不显示危机明细）
        if self.stress_test:
            st = self.stress_test
            lines += [
                "-" * 52,
                "  【压力测试】",
                f"  最差月度收益    : {_fmt(st.get('worst_month'), pct=True)}  ({st.get('worst_month_date', 'N/A')})",
                f"  最差季度收益    : {_fmt(st.get('worst_quarter'), pct=True)}",
                f"  最差年度收益    : {_fmt(st.get('worst_year'),    pct=True)}",
                f"  最长连续亏损    : {st.get('max_consecutive_loss_days', 0)} 天",
            ]
            crises = st.get("crisis_period_returns", {})
            if crises:
                lines.append("  危机区间表现:")
                for name, ret in crises.items():
                    lines.append(f"    {name:<16}: {_fmt(ret, pct=True)}")

        lines.append("=" * 52)
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
        for f_name in self.__dataclass_fields__:
            if f_name in excluded:
                continue
            val = getattr(self, f_name)
            if isinstance(val, (pd.Timestamp, pd.Period)):
                val = str(val)
            elif isinstance(val, float) and np.isnan(val):
                val = None
            elif isinstance(val, dict):
                val = val  # stress_test dict is already JSON-serializable
            d[f_name] = val
        return d
