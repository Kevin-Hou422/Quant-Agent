"""
test_backtest_plot_fidelity.py — 回测曲线保留且真实（研究用，非伪动画）

保证:
- `plot_backtest(result, prices)` 便捷入口可用
- 图里的净值曲线**逐点等于真实回测 equity_curve**（防"伪动画/装饰曲线"回归）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.backtest_engine.backtest_engine import BacktestEngine
from app.core.backtest_engine import plot_backtest


def _real_backtest(seed=0, T=250, N=8):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=T)
    cols = [f"S{i}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0006, 0.015, (T, N)), 0), idx, cols)
    vol = pd.DataFrame(1e6, idx, cols)
    sig = close.pct_change().rank(axis=1)
    w = sig.sub(sig.mean(1), axis=0)
    w = w.div(w.abs().sum(1), axis=0).fillna(0.0)
    return BacktestEngine().run(w, close, vol, sig), close


def test_backtest_curve_matches_real_equity_not_fake():
    result, prices = _real_backtest()
    fig = plot_backtest(result, prices)
    real = np.asarray(result.equity_curve.values, dtype=float)
    # 图里必须有一条曲线逐点等于真实回测净值（否则就是伪造/装饰曲线）
    ys = [np.asarray(tr.y, dtype=float) for tr in fig.data
          if getattr(tr, "y", None) is not None and len(tr.y) > 10]
    assert any(len(y) == len(real) and np.allclose(y, real) for y in ys), \
        "净值曲线与真实回测数据不匹配——可能是伪动画/装饰曲线"


def test_different_backtests_yield_different_curves():
    # 不同回测 → 不同曲线（伪动画会给出雷同/固定形状）
    r1, p1 = _real_backtest(seed=1)
    r2, p2 = _real_backtest(seed=2)
    assert not np.allclose(r1.equity_curve.values, r2.equity_curve.values)
    plot_backtest(r1, p1); plot_backtest(r2, p2)   # 均能出图


# ===========================================================================
# 逐面板保真度（D 档补强：visualizer.py 此前只有上面两条端到端用例）
# ===========================================================================
#
# 上面两条证明"图里有一条曲线等于真实净值"。但 visualizer.py 有 19 个
# 变异点，绝大多数落在**另外三个面板**和**注释框**上，那里改坏了：
#   - 回撤忘了 ×100 → 图上显示 -0.35% 而不是 -35%，风险被少看两个数量级；
#   - 滚动夏普的 clip(lower=0) / clip(upper=0) 互换 → 正负区域填反，
#     亏损期被画成绿色；
#   - 毛净值 (1 + r).cumprod() 的符号 → 成本前后的对比整个反过来；
#   - 分档柱子的颜色阈值 v < 0 翻面 → 负收益档被染成绿色。
# 这些都不会抛异常，图照样出得来。以下逐个面板钉住。
#
# 【安全前提】本文件里每条用例都会真的构造 plotly Figure。
# `visualizer.plot()` 的 `show` 参数一旦为真就会**打开浏览器标签**，
# 而变异测试会把这个布尔字面量翻成 True —— 于是几十条用例一起弹窗。
# 所以 `tests/conftest.py` 里有一个 session 级 autouse 的
# `_never_open_a_browser`，把 `plotly.graph_objects.Figure.show`
# 全局换成记账替身；需要断言"show 有没有被调用"的用例用
# `figure_show_calls` fixture 读那份记录，**不要**自己去 monkeypatch。

import copy
import functools

import pytest

from app.core.backtest_engine.risk_report import RiskReport
from app.core.backtest_engine.visualizer import BacktestVisualizer


@functools.lru_cache(maxsize=4)
def _base_report(seed=0, T=250, N=8) -> RiskReport:
    """
    跑一次真实回测并缓存 —— 下面三十多条用例共用同一份报告。

    不缓存的话每条用例都要重跑一次回测（整个文件 30s），
    而变异测试要把这个文件跑几十遍。
    """
    result, prices = _real_backtest(seed=seed, T=T, N=N)
    return RiskReport.from_result(result, prices)


def _report(seed=0, T=250, N=8) -> RiskReport:
    """每条用例拿到一份**独立副本** —— 有些用例会就地改字段。"""
    return copy.copy(_base_report(seed, T, N))


def _traces_named(fig, fragment):
    return [t for t in fig.data if fragment in (getattr(t, "name", "") or "")]


class TestEquityPanel:

    def test_the_net_curve_is_the_report_equity_curve_point_by_point(self):
        rep = _report()
        fig = BacktestVisualizer().plot(rep)
        net = _traces_named(fig, "净净值")
        assert len(net) == 1, f"净值曲线有 {len(net)} 条"
        assert np.allclose(np.asarray(net[0].y, dtype=float),
                           rep.equity_curve.values, equal_nan=True)

    def test_the_gross_curve_is_the_cumulative_product_of_gross_returns(self):
        """
        gross_eq = (1 + report.gross_returns.fillna(0)).cumprod()

        `1 +` 翻成 `-` 会让"成本前"的曲线整体倒过来 ——
        于是图上看起来"扣了成本反而更赚"，而这正是这张图要回答的问题。
        逐点比对参考实现。
        """
        rep = _report()
        assert rep.gross_returns is not None, "前提：报告里有毛收益序列"
        fig = BacktestVisualizer().plot(rep)
        gross = _traces_named(fig, "毛净值")
        assert len(gross) == 1
        ref = (1 + rep.gross_returns.fillna(0)).cumprod().values
        assert np.allclose(np.asarray(gross[0].y, dtype=float), ref,
                           rtol=1e-12, equal_nan=True), (
            "毛净值曲线不等于 (1+毛收益).cumprod() —— 符号或累乘被改了")

    def test_the_gross_curve_sits_above_the_net_one(self):
        """成本只会拖累收益 —— 毛净值终点必须不低于净净值终点。"""
        rep = _report()
        fig = BacktestVisualizer().plot(rep)
        g = np.asarray(_traces_named(fig, "毛净值")[0].y, dtype=float)
        n = np.asarray(_traces_named(fig, "净净值")[0].y, dtype=float)
        assert g[-1] >= n[-1] - 1e-9, (
            f"毛净值终点 {g[-1]:.4f} 低于净净值 {n[-1]:.4f} —— 成本方向反了")

    def test_both_equity_traces_share_the_report_index(self):
        rep = _report()
        fig = BacktestVisualizer().plot(rep)
        for frag in ("毛净值", "净净值"):
            tr = _traces_named(fig, frag)[0]
            assert list(pd.DatetimeIndex(tr.x)) == list(rep.equity_curve.index), (
                f"{frag} 的横轴不是报告的日期索引")

    def test_a_report_without_curves_still_plots(self):
        """
        两处 `is not None` 守卫被删会让一份只有标量指标的报告
        在出图时 AttributeError。
        """
        fig = BacktestVisualizer().plot(RiskReport())
        assert fig is not None
        assert _traces_named(fig, "净净值") == []


class TestDrawdownPanel:

    def test_the_drawdown_is_plotted_in_percent(self):
        """
        y=dd.values * 100

        `* 100` 被改成 `/ 100` 会让 -35% 的回撤在图上显示成 -0.0035，
        而 y 轴后缀仍然是 %% —— 读图的人会以为风险微不足道。
        """
        rep = _report()
        assert rep.drawdown_series is not None
        fig = BacktestVisualizer().plot(rep)
        dd = _traces_named(fig, "回撤")
        assert len(dd) == 1
        ref = rep.drawdown_series.fillna(0).values * 100
        assert np.allclose(np.asarray(dd[0].y, dtype=float), ref, rtol=1e-12), (
            "回撤曲线不等于 回撤序列×100 —— 百分比换算被改了")

    def test_the_drawdown_is_non_positive(self):
        rep = _report()
        fig = BacktestVisualizer().plot(rep)
        y = np.asarray(_traces_named(fig, "回撤")[0].y, dtype=float)
        assert np.nanmax(y) <= 1e-9, f"回撤出现了正值：{np.nanmax(y)}"

    def test_a_missing_drawdown_series_skips_the_panel(self):
        """守卫被删会 AttributeError。"""
        rep = _report()
        rep.drawdown_series = None
        fig = BacktestVisualizer().plot(rep)
        assert _traces_named(fig, "回撤") == []


class TestRollingSharpePanel:

    def test_the_positive_and_negative_areas_are_not_swapped(self):
        """
        pos = rs.clip(lower=0) / neg = rs.clip(upper=0)

        两者互换会让**亏损期被填成绿色、盈利期填成红色** ——
        图形状一模一样，只有颜色与符号对不上，极难用肉眼发现。
        这里直接比对两条 trace 的数值。
        """
        rep = _report()
        assert rep.rolling_sharpe is not None
        fig = BacktestVisualizer().plot(rep)
        rs = rep.rolling_sharpe.dropna()

        pos = _traces_named(fig, "滚动夏普 (正)")
        neg = _traces_named(fig, "滚动夏普 (负)")
        assert len(pos) == 1 and len(neg) == 1

        py = np.asarray(pos[0].y, dtype=float)
        ny = np.asarray(neg[0].y, dtype=float)
        assert np.allclose(py, rs.clip(lower=0).values, rtol=1e-12), (
            "「正」区域不等于 clip(lower=0) —— 正负两条被互换了")
        assert np.allclose(ny, rs.clip(upper=0).values, rtol=1e-12), (
            "「负」区域不等于 clip(upper=0) —— 正负两条被互换了")
        assert np.nanmin(py) >= -1e-12, "「正」区域里出现了负值"
        assert np.nanmax(ny) <= 1e-12, "「负」区域里出现了正值"

    def test_the_two_halves_sum_back_to_the_original_series(self):
        rep = _report()
        fig = BacktestVisualizer().plot(rep)
        rs = rep.rolling_sharpe.dropna()
        py = np.asarray(_traces_named(fig, "滚动夏普 (正)")[0].y, dtype=float)
        ny = np.asarray(_traces_named(fig, "滚动夏普 (负)")[0].y, dtype=float)
        assert np.allclose(py + ny, rs.values, rtol=1e-12), (
            "正负两段相加不等于原始滚动夏普序列")

    def test_nan_warmup_rows_are_dropped(self):
        """
        rs = report.rolling_sharpe.dropna() —— 去掉会让 60 日窗口
        起始那一段 NaN 进入填充区域，plotly 会画出断裂的色块。
        """
        rep = _report()
        fig = BacktestVisualizer().plot(rep)
        y = np.asarray(_traces_named(fig, "滚动夏普 (正)")[0].y, dtype=float)
        assert not np.isnan(y).any(), "滚动夏普里残留了预热期 NaN"
        assert len(y) == len(rep.rolling_sharpe.dropna())

    def test_a_missing_rolling_sharpe_skips_the_panel(self):
        rep = _report()
        rep.rolling_sharpe = None
        fig = BacktestVisualizer().plot(rep)
        assert _traces_named(fig, "滚动夏普") == []


class TestRollingIcPanel:

    def test_the_ic_trace_matches_the_report_series(self):
        rep = _report()
        assert rep.rolling_ic is not None
        fig = BacktestVisualizer().plot(rep)
        ic = _traces_named(fig, "滚动 Rank IC")
        assert len(ic) == 1
        ref = rep.rolling_ic.dropna()
        assert np.allclose(np.asarray(ic[0].y, dtype=float), ref.values,
                           rtol=1e-12)
        assert list(pd.DatetimeIndex(ic[0].x)) == list(ref.index)

    def test_the_ic_is_not_rescaled(self):
        """
        IC 与回撤不同 —— 它**不**乘 100。这里正面钉住量纲，
        免得有人照抄回撤那一行把 IC 也放大 100 倍。
        """
        rep = _report()
        fig = BacktestVisualizer().plot(rep)
        y = np.asarray(_traces_named(fig, "滚动 Rank IC")[0].y, dtype=float)
        assert np.nanmax(np.abs(y)) <= 1.0 + 1e-9, (
            f"滚动 IC 的绝对值超过 1（{np.nanmax(np.abs(y))}）—— 被乘了百分比系数")

    def test_a_missing_rolling_ic_skips_the_panel(self):
        rep = _report()
        rep.rolling_ic = None
        fig = BacktestVisualizer().plot(rep)
        assert _traces_named(fig, "滚动 Rank IC") == []


class TestFigureLayout:

    def test_the_figure_has_four_stacked_panels_in_the_documented_order(self):
        """
        四个子图的标题顺序就是模块 docstring 里的 Row 1..4。
        行号被改（比如回撤画到第 3 行）会让标题与内容错位。
        """
        fig = BacktestVisualizer().plot(_report())
        titles = [a.text for a in fig.layout.annotations if a.text]
        for frag in ("累计净值", "水下回撤", "滚动夏普", "滚动 Rank IC"):
            assert any(frag in t for t in titles), f"缺少子图标题：{frag}"

    def test_each_trace_lands_on_its_documented_row(self):
        """
        row=1..4 —— 行号错位会把回撤画进净值面板，
        两条量纲差 100 倍的线叠在一起，净值曲线被压成一条直线。
        用 plotly 的 y 轴编号反查每条 trace 落在哪一行。
        """
        fig = BacktestVisualizer().plot(_report())
        axis_of = {}
        for tr in fig.data:
            ax = getattr(tr, "yaxis", None) or "y"
            axis_of.setdefault(ax, []).append(tr.name)

        def row_of(fragment):
            for ax, names in axis_of.items():
                if any(fragment in (n or "") for n in names):
                    return 1 if ax == "y" else int(ax[1:])
            return None

        assert row_of("净净值") == 1
        assert row_of("回撤") == 2
        assert row_of("滚动夏普") == 3
        assert row_of("滚动 Rank IC") == 4

    def test_the_theme_is_applied_and_configurable(self):
        assert BacktestVisualizer().theme == "plotly_white"
        fig = BacktestVisualizer(theme="plotly_dark").plot(_report())
        assert fig.layout.template is not None

    def test_the_title_is_used(self):
        fig = BacktestVisualizer().plot(_report(), title="我的策略")
        assert fig.layout.title.text == "我的策略"

    def test_show_is_off_by_default(self, figure_show_calls):
        """
        `show: bool = False` + `if show: fig.show()`

        默认值翻成 True 会让**每一次出图都弹一个浏览器标签**。
        在无人值守的日循环里这会堆积上百个页面；
        在变异测试里它一次性在使用者屏幕上弹了三十多个 —— 实测事故。

        注意这里用的是 conftest 里**全局**的 `figure_show_calls`，
        而不是自己 monkeypatch `Figure.show`：
        自己打补丁只保护自己这一条，别的用例照样能弹窗
        （事故就是这么发生的）。
        """
        BacktestVisualizer().plot(_report())
        assert figure_show_calls == [], (
            "show 默认是 False，却调用了 fig.show() —— 默认值被翻成了 True")
        BacktestVisualizer().plot(_report(), show=True)
        assert len(figure_show_calls) == 1, (
            "显式 show=True 时没有调用 fig.show()")


class TestMetricsAnnotation:

    def _text(self, rep):
        fig = BacktestVisualizer().plot(rep)
        boxes = [a.text for a in fig.layout.annotations
                 if a.text and "年化收益" in a.text]
        assert len(boxes) == 1, "关键指标注释框不存在或不唯一"
        return boxes[0]

    def test_rate_metrics_are_rendered_as_percentages(self):
        """
        f"{v*100:.2f}%" —— `* 100` 被改会让"年化收益 12.34%"
        显示成 0.12%，读图的人会认为策略几乎没赚钱。
        """
        rep = _report()
        rep.annualized_return = 0.1234
        rep.max_drawdown = -0.3567
        txt = self._text(rep)
        assert "12.34%" in txt, f"年化收益没有按百分比渲染：{txt}"
        assert "-35.67%" in txt, f"最大回撤没有按百分比渲染：{txt}"

    def test_ratio_metrics_are_rendered_with_three_decimals(self):
        rep = _report()
        rep.sharpe_ratio = 1.23456
        rep.sortino_ratio = 2.34567
        rep.mean_ic = 0.03456
        rep.ic_ir = 0.45678
        txt = self._text(rep)
        for frag in ("1.235", "2.346", "0.035", "0.457"):
            assert frag in txt, f"注释框里缺少 {frag}：{txt}"

    def test_a_missing_metric_shows_na_rather_than_nan(self):
        """
        if v is None or isnan(v): return "N/A"
        —— 守卫被删会在图上打出 nan，看起来像数据出错而不是"未计算"。
        """
        rep = _report()
        rep.sharpe_ratio = float("nan")
        rep.sortino_ratio = None
        txt = self._text(rep)
        assert "nan" not in txt.lower(), f"注释框里出现了 nan：{txt}"
        assert txt.count("N/A") >= 2

    def test_the_cost_drag_is_shown_in_basis_points(self):
        rep = _report()
        rep.cost_drag_bps = 42.678
        assert "42.7bps" in self._text(rep)

    def test_every_documented_metric_appears(self):
        txt = self._text(_report())
        for label in ("年化收益", "夏普", "最大回撤", "Sortino", "IC", "IC-IR", "成本"):
            assert label in txt, f"注释框里缺少 {label}"


class TestDecileBar:

    def _rep_with_deciles(self, values):
        rep = _report()
        rep.decile_returns = pd.Series(values, index=range(1, len(values) + 1))
        return rep

    def test_bar_heights_are_the_decile_returns_in_percent(self):
        """y=dr.values * 100 —— 与回撤同一处坑。"""
        vals = [-0.004, -0.001, 0.0, 0.002, 0.006]
        fig = BacktestVisualizer().plot_decile_bar(self._rep_with_deciles(vals))
        assert np.allclose(np.asarray(fig.data[0].y, dtype=float),
                           np.array(vals) * 100, rtol=1e-12)

    def test_negative_buckets_are_red_and_the_rest_green(self):
        """
        "#d62728" if v < 0 else "#2ca02c"

        `<` 翻成 `>` 会让整张图的颜色**完全反过来** —— 亏钱的档变绿。
        `<` 翻成 `<=` 只改变恰好为 0 的那一档（当前契约：0 算绿）。
        """
        vals = [-0.004, 0.0, 0.006]
        fig = BacktestVisualizer().plot_decile_bar(self._rep_with_deciles(vals))
        colors = list(fig.data[0].marker.color)
        assert colors == ["#d62728", "#2ca02c", "#2ca02c"], (
            f"分档颜色是 {colors} —— 负收益档应当红色，0 与正收益为绿色")

    def test_bucket_labels_follow_the_decile_index(self):
        fig = BacktestVisualizer().plot_decile_bar(
            self._rep_with_deciles([0.001, 0.002, 0.003]))
        assert list(fig.data[0].x) == ["D1", "D2", "D3"]

    def test_an_empty_decile_series_raises_instead_of_drawing_nothing(self):
        """
        raise ValueError("RiskReport.decile_returns 为空...")
        —— 被删会产出一张空柱状图，看起来像"所有档收益都是 0"。
        """
        rep = _report()
        rep.decile_returns = None
        with pytest.raises(ValueError, match="decile_returns 为空"):
            BacktestVisualizer().plot_decile_bar(rep)

        rep.decile_returns = pd.Series(dtype=float)
        with pytest.raises(ValueError, match="decile_returns 为空"):
            BacktestVisualizer().plot_decile_bar(rep)

    def test_a_zero_reference_line_is_drawn(self):
        fig = BacktestVisualizer().plot_decile_bar(
            self._rep_with_deciles([-0.001, 0.002]))
        lines = [s for s in fig.layout.shapes if getattr(s, "y0", None) == 0]
        assert lines, "分档图缺少 y=0 基准线，正负档无从对比"

    def test_show_is_off_by_default(self, figure_show_calls):
        """分档图同样有一个 `show` 默认值 —— 与主图各算一个变异点。"""
        rep = self._rep_with_deciles([0.001, 0.002])
        BacktestVisualizer().plot_decile_bar(rep)
        assert figure_show_calls == [], (
            "分档图的 show 默认是 False，却调用了 fig.show()")
        BacktestVisualizer().plot_decile_bar(rep, show=True)
        assert len(figure_show_calls) == 1


class TestPlotlyGuard:

    def test_a_missing_plotly_raises_an_actionable_error(self, monkeypatch):
        """
        _require_plotly —— plotly 是可选依赖。守卫被删会让调用方看到
        NameError: name 'make_subplots' is not defined，看不出该装什么。
        """
        import app.core.backtest_engine.visualizer as V
        monkeypatch.setattr(V, "_HAS_PLOTLY", False)
        with pytest.raises(ImportError, match="pip install plotly"):
            V.BacktestVisualizer().plot(_report())
        with pytest.raises(ImportError, match="pip install plotly"):
            V.BacktestVisualizer().plot_decile_bar(_report())

    def test_the_guard_passes_when_plotly_is_present(self):
        import app.core.backtest_engine.visualizer as V
        assert V._HAS_PLOTLY is True
        V._require_plotly()      # 不抛即通过


def _backend_root():
    """向上找到含 app/ 的目录 = backend/（子进程用例要用它当 cwd）。"""
    from pathlib import Path
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "app").is_dir():
            return parent
    raise RuntimeError(f"从 {p} 向上找不到含 app/ 的 backend 根目录")


# ===========================================================================
# 首测存活项收口：五个 plotly 布局开关
# ===========================================================================
#
# 首测 19 点 / 击杀 73.7% / 存活 5，全是布尔字面量翻面。
# 它们一个都不改变曲线数值，只改变**图长什么样** ——
# 而这张图的用途就是给人看，所以"看起来对不对"就是它的正确性。

class TestLayoutSwitches:

    def test_the_four_panels_share_one_x_axis(self):
        """
        **首测存活项（L79）**：`make_subplots(shared_xaxes=True)` 翻成 False。

        四个面板叠在一起看的前提就是**横轴对齐**：净值的回撤段、
        滚动夏普的低谷、IC 的失效期必须能竖着连起来。
        各自独立缩放之后，图形状还在、数字也对，
        但"某段回撤对应哪段 IC"完全读不出来 —— 这张图就白画了。
        """
        lay = BacktestVisualizer().plot(_report()).layout
        linked = [getattr(lay, k).matches for k in
                  ("xaxis", "xaxis2", "xaxis3") if getattr(lay, k, None)]
        assert linked and all(m is not None for m in linked), (
            f"上面三个面板的横轴没有绑定到第四个（matches={linked}）—— "
            f"`shared_xaxes=True` 被翻成了 False")
        assert len(set(linked)) == 1, f"横轴绑定不一致：{linked}"

    def test_the_metrics_box_is_a_plain_label_not_an_arrow_callout(self):
        """
        **首测存活项（L114）**：`fig.add_annotation(showarrow=False)` 翻成 True。

        关键指标框是钉在图左上角（`xref="paper"`）的说明块，
        不是指向某个数据点的标注。`showarrow=True` 会让 plotly
        从 (0.01, 0.99) 这个**纸面坐标**拉一根箭头指向数据区，
        把净值曲线盖掉一角。
        """
        lay = BacktestVisualizer().plot(_report()).layout
        boxes = [a for a in lay.annotations if a.text and "年化收益" in a.text]
        assert len(boxes) == 1
        assert boxes[0].showarrow is False, (
            "关键指标框带上了指示箭头 —— `showarrow=False` 被翻成了 True")
        assert boxes[0].xref == "paper" and boxes[0].yref == "paper", (
            "指标框不是钉在纸面坐标上，会随数据缩放漂移")

    def test_both_axes_keep_their_grid(self):
        """
        **首测存活项（L129 / L130）**：
        `fig.update_xaxes(showgrid=True, ...)` / `update_yaxes(...)` 翻成 False。

        网格线是从图上读数的唯一依据 —— 关掉之后，
        "这段回撤大概多深""夏普在哪一段跌破 0" 全靠目测。
        两条轴各是一个变异点，所以两条都要断言。
        """
        lay = BacktestVisualizer().plot(_report()).layout
        xs = [getattr(lay, k) for k in ("xaxis", "xaxis2", "xaxis3", "xaxis4")
              if getattr(lay, k, None) is not None]
        ys = [getattr(lay, k) for k in ("yaxis", "yaxis2", "yaxis3", "yaxis4")
              if getattr(lay, k, None) is not None]
        assert xs and ys, "取不到坐标轴对象 —— 本用例需要重写"
        assert all(a.showgrid for a in xs), (
            f"横轴网格被关掉了：{[a.showgrid for a in xs]}")
        assert all(a.showgrid for a in ys), (
            f"纵轴网格被关掉了：{[a.showgrid for a in ys]}")
        assert all(a.gridcolor == "#eee" for a in xs + ys), (
            "网格颜色被改了 —— 过深会盖住曲线，过浅等于没有")


class TestPlotlyAbsentEnvironment:
    """
    `_HAS_PLOTLY` 的 except 分支在装了 plotly 的环境里**永远不执行**，
    所以 `_HAS_PLOTLY = False` 那一行的变异在进程内怎么测都是等价的。

    唯一诚实的办法是**在子进程里把 plotly 屏蔽掉再导入模块** ——
    既真正走到那条分支，又不会用 `importlib.reload` 污染当前会话
    （台账里"发现 3"记过：reload 会让 session 级 fixture 抓着旧对象）。
    """

    def test_the_module_degrades_when_plotly_is_missing(self):
        import subprocess
        import sys
        import textwrap

        code = textwrap.dedent(
            """
            import sys

            class _Blocker:
                def find_module(self, name, path=None):
                    return self if name.split(".")[0] == "plotly" else None

                def load_module(self, name):
                    raise ImportError("plotly blocked for this test")

                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] == "plotly":
                        raise ImportError("plotly blocked for this test")
                    return None

            for mod in [m for m in sys.modules if m.split(".")[0] == "plotly"]:
                del sys.modules[mod]
            sys.meta_path.insert(0, _Blocker())

            import app.core.backtest_engine.visualizer as V
            assert V._HAS_PLOTLY is False, "没装 plotly，_HAS_PLOTLY 却是 True"
            try:
                V._require_plotly()
            except ImportError as exc:
                assert "pip install plotly" in str(exc), exc
            else:
                raise AssertionError("没装 plotly 时 _require_plotly 没有抛")
            print("OK")
            """
        )
        proc = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, timeout=180,
                              cwd=str(_backend_root()))
        assert proc.returncode == 0, (
            f"屏蔽 plotly 之后模块行为不对：\\n"
            f"stdout={proc.stdout}\\nstderr={proc.stderr[-2000:]}")
        assert "OK" in proc.stdout

    def test_the_flag_is_true_in_this_environment(self):
        """对照组：本机装了 plotly，标志位必须是 True。"""
        import app.core.backtest_engine.visualizer as V
        assert V._HAS_PLOTLY is True
