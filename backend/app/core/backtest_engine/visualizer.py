"""
BacktestVisualizer — Plotly 4 面板交互式可视化

plot(report)        → 4 面板主图（净值 / 回撤 / 滚动夏普 / 滚动 IC）
plot_decile_bar(report) → 分档收益柱状图

均返回 plotly.graph_objects.Figure，支持 .show() / .write_html()。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

from .risk_report import RiskReport


def _require_plotly() -> None:
    if not _HAS_PLOTLY:
        raise ImportError("visualizer 需要 plotly：pip install plotly")


# ---------------------------------------------------------------------------
# BacktestVisualizer
# ---------------------------------------------------------------------------

class BacktestVisualizer:
    """
    生成回测结果的 Plotly 可视化图表。

    Parameters
    ----------
    theme : 'plotly_dark' / 'plotly_white' 等 Plotly 主题
    """

    def __init__(self, theme: str = "plotly_white") -> None:
        self.theme = theme

    # ------------------------------------------------------------------
    # 主图：4 面板
    # ------------------------------------------------------------------

    def plot(
        self,
        report: RiskReport,
        title: str = "回测绩效报告",
        show: bool = False,
    ) -> "go.Figure":
        """
        4 面板图：
          Row 1: 累计净值（毛 vs 净）
          Row 2: 水下回撤
          Row 3: 滚动夏普比率（60日）
          Row 4: 滚动 Rank IC（20日）

        Parameters
        ----------
        report : RiskReport
        title  : 图表标题
        show   : True 则直接调用 fig.show()

        Returns
        -------
        plotly.graph_objects.Figure
        """
        _require_plotly()

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "累计净值 (Gross vs Net)",
                "水下回撤 (Drawdown Under Water)",
                "滚动夏普比率 (60日)",
                "滚动 Rank IC (20日)",
            ],
            vertical_spacing=0.06,
            row_heights=[0.35, 0.20, 0.22, 0.23],
        )

        dates = (
            report.equity_curve.index
            if report.equity_curve is not None
            else pd.date_range("2020-01-01", periods=10)
        )

        # ---- Row 1: 净值曲线 ----
        self._add_equity(fig, report, row=1)

        # ---- Row 2: 回撤 ----
        self._add_drawdown(fig, report, row=2)

        # ---- Row 3: 滚动夏普 ----
        self._add_rolling_sharpe(fig, report, row=3)

        # ---- Row 4: 滚动 IC ----
        self._add_rolling_ic(fig, report, row=4)

        # ---- 注释框（关键指标） ----
        ann_txt = self._metrics_annotation(report)
        fig.add_annotation(
            x=0.01, y=0.99,
            xref="paper", yref="paper",
            text=ann_txt,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#888",
            borderwidth=1,
            font=dict(size=10, family="monospace"),
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            template=self.theme,
            height=900,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
            margin=dict(l=60, r=30, t=80, b=40),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#eee")
        fig.update_yaxes(showgrid=True, gridcolor="#eee")

        if show:
            fig.show()
        return fig

    # ------------------------------------------------------------------
    # 分档收益柱状图
    # ------------------------------------------------------------------

    def plot_decile_bar(
        self,
        report: RiskReport,
        show: bool = False,
    ) -> "go.Figure":
        """分 10 档的平均前瞻收益柱状图（验证单调性）。"""
        _require_plotly()

        if report.decile_returns is None or report.decile_returns.empty:
            raise ValueError("RiskReport.decile_returns 为空，无法绘制分档图")

        dr  = report.decile_returns
        clr = [
            "#d62728" if v < 0 else "#2ca02c"
            for v in dr.values
        ]

        fig = go.Figure(go.Bar(
            x=[f"D{i}" for i in dr.index],
            y=dr.values * 100,
            marker_color=clr,
            name="平均前瞻收益 (%)",
            hovertemplate="第 %{x} 档<br>均值: %{y:.4f}%<extra></extra>",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
        fig.update_layout(
            title="分档收益分析 (Decile Return Analysis)",
            xaxis_title="信号分档",
            yaxis_title="平均前瞻收益 (%)",
            template=self.theme,
            height=400,
        )
        if show:
            fig.show()
        return fig

    # ------------------------------------------------------------------
    # 辅助：各 Row 添加 traces
    # ------------------------------------------------------------------

    def _add_equity(self, fig, report: RiskReport, row: int) -> None:
        idx = (
            report.equity_curve.index
            if report.equity_curve is not None else []
        )
        # 毛收益
        if report.gross_returns is not None:
            gross_eq = (1 + report.gross_returns.fillna(0)).cumprod()
            fig.add_trace(go.Scatter(
                x=idx, y=gross_eq.values,
                name="毛净值 (Gross)",
                line=dict(color="lightgray", width=1.5, dash="dot"),
                hovertemplate="%{x|%Y-%m-%d}<br>毛净值: %{y:.4f}<extra></extra>",
            ), row=row, col=1)

        # 净收益
        if report.equity_curve is not None:
            fig.add_trace(go.Scatter(
                x=idx, y=report.equity_curve.values,
                name="净净值 (Net)",
                line=dict(color="#1f77b4", width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>净净值: %{y:.4f}<extra></extra>",
            ), row=row, col=1)

        # 基准线
        fig.add_hline(y=1.0, line_dash="dash", line_color="black",
                      line_width=1, row=row, col=1)

    def _add_drawdown(self, fig, report: RiskReport, row: int) -> None:
        if report.drawdown_series is None:
            return
        dd = report.drawdown_series.fillna(0)
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100,
            name="回撤 (%)",
            fill="tozeroy",
            line=dict(color="#d62728", width=1),
            fillcolor="rgba(214,39,40,0.25)",
            hovertemplate="%{x|%Y-%m-%d}<br>回撤: %{y:.2f}%<extra></extra>",
        ), row=row, col=1)
        fig.update_yaxes(ticksuffix="%", row=row, col=1)

    def _add_rolling_sharpe(self, fig, report: RiskReport, row: int) -> None:
        if report.rolling_sharpe is None:
            return
        rs = report.rolling_sharpe.dropna()
        pos = rs.clip(lower=0)
        neg = rs.clip(upper=0)

        fig.add_trace(go.Scatter(
            x=rs.index, y=pos.values,
            name="滚动夏普 (正)",
            fill="tozeroy",
            line=dict(color="#2ca02c", width=1),
            fillcolor="rgba(44,160,44,0.25)",
            hovertemplate="%{x|%Y-%m-%d}<br>夏普: %{y:.2f}<extra></extra>",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=rs.index, y=neg.values,
            name="滚动夏普 (负)",
            fill="tozeroy",
            line=dict(color="#d62728", width=1),
            fillcolor="rgba(214,39,40,0.25)",
            hovertemplate="%{x|%Y-%m-%d}<br>夏普: %{y:.2f}<extra></extra>",
        ), row=row, col=1)
        fig.add_hline(y=0, line_color="black", line_width=0.8, row=row, col=1)

    def _add_rolling_ic(self, fig, report: RiskReport, row: int) -> None:
        if report.rolling_ic is None:
            return
        ic = report.rolling_ic.dropna()
        fig.add_trace(go.Scatter(
            x=ic.index, y=ic.values,
            name="滚动 Rank IC",
            line=dict(color="#1f77b4", width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>IC: %{y:.4f}<extra></extra>",
        ), row=row, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="black",
                      line_width=0.8, row=row, col=1)

    @staticmethod
    def _metrics_annotation(report: RiskReport) -> str:
        def f(v, pct=False):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return f"{v*100:.2f}%" if pct else f"{v:.3f}"

        return (
            f"年化收益: {f(report.annualized_return, True)}  "
            f"夏普: {f(report.sharpe_ratio)}  "
            f"最大回撤: {f(report.max_drawdown, True)}<br>"
            f"Sortino: {f(report.sortino_ratio)}  "
            f"IC: {f(report.mean_ic)}  "
            f"IC-IR: {f(report.ic_ir)}  "
            f"成本: {report.cost_drag_bps:.1f}bps"
        )
