"""
risk_report.py —— 契约与边界定钉测试（变异测试驱动）

来由：32 个变异点首测击杀率 **3.1%** —— 全项目最差，**改坏 32 处只有 1 处会被发现**。

而 `RiskReport` 是所有对外结论的载体：策略门读它的 `sharpe_ratio` / `deflated_sharpe`，
晋级门读它的 `insufficient_sample`，前端与台账读它的 `to_dict()`。
既有覆盖（test_phase6 / test_phase_pm_wiring）只断言"字段存在"，
对**样本不足置空**、**显著性标记**、**序列字段不进 repr/to_dict** 这些关键契约零断言。

其中最危险的一条：`min_obs` 那两行判定一旦改坏，
**样本不足的策略会带着一个看起来正常的年化 Sharpe 通过晋级门**。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.backtest_engine.backtest_engine import BacktestResult
from app.core.backtest_engine.risk_report import (
    MIN_OBS_FOR_SHARPE,
    RiskReport,
    sharpe_standard_error,
)

RATIO_FIELDS = ("sharpe_ratio", "sharpe_tstat", "calmar_ratio", "sortino_ratio",
                "ic_ir", "deflated_sharpe", "information_ratio",
                "long_sharpe", "short_sharpe", "annualized_return", "annualized_vol")


def _result(n_days: int = 120, n_tickers: int = 6, seed: int = 0) -> BacktestResult:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n_days)
    cols = [f"T{i}" for i in range(n_tickers)]
    ret = pd.Series(rng.normal(0.0006, 0.01, n_days), index=idx)
    sig = pd.DataFrame(rng.normal(0, 1, (n_days, n_tickers)), index=idx, columns=cols)
    pos = pd.DataFrame(rng.normal(0, 0.1, (n_days, n_tickers)), index=idx, columns=cols)
    return BacktestResult(
        equity_curve=(1 + ret).cumprod(), gross_returns=ret, net_returns=ret,
        positions=pos, trade_log=pd.DataFrame(),
        turnover=pd.Series(0.2, index=idx), signal=sig,
        daily_cost_bps=pd.Series(3.0, index=idx),
        long_returns=ret * 1.2, short_returns=-ret * 0.4)


def _prices(res: BacktestResult) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    cols = res.signal.columns
    return pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (len(res.signal), len(cols))), axis=0),
        index=res.signal.index, columns=cols)


# ===========================================================================
# A. Sharpe 标准误（Lo 2002）
# ===========================================================================

class TestSharpeStandardError:
    """`SE = √((1 + SR²/2) / T年)`。三处变异全在这一行。"""

    def test_formula_exact(self):
        sr, n, tdays = 1.5, 252, 252.0
        expected = float(np.sqrt((1.0 + 0.5 * sr * sr) / (n / tdays)))
        assert sharpe_standard_error(sr, n, tdays) == pytest.approx(expected, abs=1e-12)
        assert sharpe_standard_error(sr, n, tdays) == pytest.approx(1.4577379737113252,
                                                                   abs=1e-9)

    def test_numerator_is_one_plus_half_sr_squared(self):
        """
        `1.0 + 0.5*sr*sr` 写成 `1.0 - 0.5*sr*sr`：SR=1.5 时分子由 2.125 变成 -0.125
        → 开负数的平方根 → NaN。`*` 写成 `/` 则分子变成 1 + 0.5/(sr*sr)。
        """
        se = sharpe_standard_error(1.5, 252, 252.0)
        assert np.isfinite(se), "分子疑似被写成减法，开出了负数的平方根"
        wrong_div = float(np.sqrt((1.0 + 0.5 / (1.5 * 1.5)) / 1.0))
        assert se != pytest.approx(wrong_div, abs=1e-6)

    def test_se_shrinks_with_more_observations(self):
        """样本越长标准误越小 —— 这是它存在的全部意义。"""
        assert sharpe_standard_error(1.0, 1008) < sharpe_standard_error(1.0, 252)

    def test_se_grows_with_sharpe(self):
        assert sharpe_standard_error(3.0, 252) > sharpe_standard_error(0.5, 252)

    def test_degenerate_inputs_give_nan(self):
        """`if not np.isfinite(sr) or n_obs <= 1: return nan` —— 删掉 not 会反过来。"""
        assert np.isnan(sharpe_standard_error(float("nan"), 252))
        assert np.isnan(sharpe_standard_error(float("inf"), 252))
        assert np.isnan(sharpe_standard_error(1.0, 1))
        assert np.isfinite(sharpe_standard_error(1.0, 2)), "n_obs=2 是合法的（边界）"


# ===========================================================================
# B. 样本不足 —— 最危险的一处
# ===========================================================================

class TestInsufficientSample:
    """
    ```
    if n < min_obs or (min_obs == 0 and n < MIN_OBS_FOR_SHARPE):
        self.insufficient_sample = True
    if min_obs > 0 and n < min_obs:
        <把比率类指标全部置 NaN>
    ```
    这两行决定"样本不够时要不要把数字抹掉"。首测时**每一个变异都存活**，
    意味着当时把它改成"永不置空"也没有任何测试会红。
    """

    def test_short_sample_blanks_the_ratio_metrics(self):
        r = RiskReport.from_result(_result(n_days=30), min_obs=60)
        assert r.insufficient_sample is True
        for f in RATIO_FIELDS:
            assert np.isnan(getattr(r, f)), f"{f} 在样本不足时仍给出了数字"

    def test_sufficient_sample_keeps_the_numbers(self):
        """对照组：样本足够时不得置空，否则上一条可以靠"永远置空"作弊通过。"""
        r = RiskReport.from_result(_result(n_days=120), min_obs=60)
        assert r.insufficient_sample is False
        assert np.isfinite(r.sharpe_ratio)
        assert np.isfinite(r.annualized_return)

    def test_exactly_min_obs_is_sufficient(self):
        """
        `if n < min_obs` 的边界：**恰好等于** min_obs 就算够。
        放宽成 `<=` 会把刚好攒够观测的策略多卡一次，且理由是"样本不足"。
        """
        r = RiskReport.from_result(_result(n_days=60), min_obs=60)
        assert r.insufficient_sample is False, "恰好 60 个观测被判成样本不足"
        assert np.isfinite(r.sharpe_ratio)

    def test_one_short_of_min_obs_is_insufficient(self):
        r = RiskReport.from_result(_result(n_days=59), min_obs=60)
        assert r.insufficient_sample is True
        assert np.isnan(r.sharpe_ratio)

    def test_min_obs_zero_flags_but_does_not_blank(self):
        """
        `min_obs=0` 是**内部搜索**专用：只标注不置空（否则 GP 的 fitness 全 NaN，
        选择退化为掷硬币）。两个判定必须同时成立：
          - `insufficient_sample` 仍为 True（对外展示据此拒绝）
          - 比率类指标**保留数字**
        `min_obs > 0 and n < min_obs` 里的 `and` 改成 `or` 会把这条路径也置空。
        """
        r = RiskReport.from_result(_result(n_days=30), min_obs=0)
        assert r.insufficient_sample is True, "min_obs=0 时仍须如实标注样本不足"
        assert np.isfinite(r.sharpe_ratio), (
            "min_obs=0 只应标注不置空 —— 置空会让 GP 的 fitness 全部变成 NaN")

    def test_min_obs_zero_with_long_sample_is_not_flagged(self):
        """`min_obs == 0 and n < MIN_OBS_FOR_SHARPE` 的另一侧。"""
        r = RiskReport.from_result(_result(n_days=120), min_obs=0)
        assert r.insufficient_sample is False

    def test_min_obs_zero_boundary_at_module_constant(self):
        """
        `n < MIN_OBS_FOR_SHARPE` 的边界：恰好 60 不算不足。
        这条同时钉住模块常量本身没被悄悄改动。
        """
        assert MIN_OBS_FOR_SHARPE == 60
        assert RiskReport.from_result(
            _result(n_days=60), min_obs=0).insufficient_sample is False
        assert RiskReport.from_result(
            _result(n_days=59), min_obs=0).insufficient_sample is True

    def test_insufficient_sample_defaults_to_false(self):
        """`insufficient_sample: bool = False` —— 默认值改成 True 会让每份报告都被拒。"""
        r = RiskReport.from_result(_result(n_days=200), min_obs=60)
        assert r.insufficient_sample is False

    def test_raw_series_survive_the_blanking(self):
        """置空的是**比率类**指标，原始序列必须保留 —— 数据本身没错。"""
        r = RiskReport.from_result(_result(n_days=30), min_obs=60)
        assert r.net_returns is not None and len(r.net_returns) == 30
        assert r.equity_curve is not None and len(r.equity_curve) == 30

    def test_min_obs_required_is_recorded(self):
        r = RiskReport.from_result(_result(n_days=30), min_obs=60)
        assert r.min_obs_required == 60


PROVEN_EQUIVALENT = {
    "app/core/backtest_engine/risk_report.py ×1 — L270 `if min_obs > 0 and n < min_obs:` → `>=`":
        "`min_obs >= 0` 只在 min_obs == 0 时与原式不同，而此时右侧 `n < 0` 恒为假"
        "（n 是长度，非负），`and` 短路后整体仍为假。两侧对**所有** min_obs 取值"
        "行为一致。见 test_min_obs_zero_never_blanks_regardless_of_the_comparison。",

    "app/core/backtest_engine/risk_report.py ×1 — L192 `if var_m > 1e-12:` → `>=`":
        "唯一的区分点是 var_m **恰好等于** 1e-12。我一度以为常数基准（var_m=0）"
        "能区分 —— **判错了**：`0.0 >= 1e-12` 同样为假，两侧都跳过除法。"
        "var_m = float(np.var(x)) 是平方和除以 n 的浮点结果，无法构造成精确的 1e-12。"
        "见 test_variance_guard_boundary_is_unreachable。",
}


def test_variance_guard_boundary_is_unreachable():
    """
    L192 等价性的机械验证：两种比较符在**所有可达的** var_m 上判定一致，
    且 1e-12 这个精确值无法由 np.var 构造出来。
    """
    tol = 1e-12
    for var_m in (0.0, 1e-20, 1e-13, 1e-12 / 2, 1e-6, 1.0):
        assert (var_m > tol) == (var_m >= tol), f"var_m={var_m} 上两种比较符不一致"
    rng = np.random.default_rng(3)
    for scale in (1e-6, 1e-7, 1e-8):
        for _ in range(200):
            assert float(np.var(rng.normal(0.0, scale, 120))) != tol
    for base in (1.0, 1e-4, 1e-8):
        assert (base + tol) - base != tol


def test_min_obs_zero_never_blanks_regardless_of_the_comparison():
    """L270 等价性的机械验证：min_obs=0 时两种比较符都不会触发置空。"""
    for n in (0, 1, 30, 60, 1000):
        assert not (0 > 0 and n < 0)
        assert not (0 >= 0 and n < 0)


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 2
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"


# ===========================================================================
# C. 基准 beta 的条件分支
# ===========================================================================

class TestBenchmarkBranches:

    def test_no_benchmark_leaves_beta_nan(self):
        """`if benchmark_returns is not None:` 删掉 not → 没传基准反而去算。"""
        r = RiskReport.from_result(_result(n_days=120))
        assert np.isnan(r.benchmark_beta)
        assert np.isnan(r.portfolio_beta)

    def test_benchmark_populates_beta(self):
        res = _result(n_days=120)
        bench = res.net_returns * 0.5 + 0.0001
        r = RiskReport.from_result(res, benchmark_returns=bench)
        assert np.isfinite(r.benchmark_beta), "传了基准却没算出 beta"
        assert np.isfinite(r.portfolio_beta)

    def test_zero_variance_benchmark_leaves_portfolio_beta_nan(self):
        """
        `if var_m > 1e-12:` —— 常数基准方差为 0，不做除法，beta 保持 NaN。
        （`>` 与 `>=` 在 var_m == 0 上取值相同，该变异的等价性见
        PROVEN_EQUIVALENT / test_variance_guard_boundary_is_unreachable。）
        """
        import warnings
        res = _result(n_days=120)
        flat = pd.Series(0.0, index=res.net_returns.index)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = RiskReport.from_result(res, benchmark_returns=flat)
        bad = [w for w in caught
               if "invalid value" in str(w.message) or "divide" in str(w.message)]
        assert not bad, f"零方差基准触发了除零警告：{[str(w.message) for w in bad]}"
        assert np.isnan(r.portfolio_beta)

    def test_observation_count_comes_from_the_return_series(self):
        """
        `n = len(self.net_returns.dropna()) if self.net_returns is not None else self.n_days`
        —— 删掉 `not` 后会改用 `n_days`（**未剔除 NaN** 的总天数）。

        构造：120 天里只有 40 天有有效净收益。按 dropna 计 n=40 → 样本不足；
        按 n_days 计 n=120 → 会被判为样本充足并给出年化 Sharpe。
        """
        res = _result(n_days=120)
        nr = res.net_returns.copy()
        nr.iloc[40:] = np.nan
        object.__setattr__(res, "net_returns", nr)
        r = RiskReport.from_result(res, min_obs=60)
        assert r.insufficient_sample is True, (
            "有效观测只有 40 天却被判样本充足 —— 观测数疑似取的是总天数")
        assert np.isnan(r.sharpe_ratio)

    def test_prices_switch_the_ic_method(self):
        """`if prices is not None:` 删掉 not → 有价格反而用近似 IC。"""
        res = _result(n_days=120)
        with_px = RiskReport.from_result(res, prices=_prices(res))
        without = RiskReport.from_result(res)
        assert with_px.ic_method == "exact_price"
        assert without.ic_method == "approx_position"


# ===========================================================================
# D. 文本摘要的标记
# ===========================================================================

class TestSummaryFlags:

    def test_significant_flag_needs_tstat_above_196(self):
        """
        `" ✓显著" if (not isnan(t) and t > 1.96) else " ✗不显著" if not isnan(t) else ""`
        —— 三个分支两两可改坏：删 `not`、`and` 改 `or`、`>` 改 `>=`。
        直接构造三个 t 值核对输出。
        """
        r = RiskReport.from_result(_result(n_days=120))
        object.__setattr__(r, "sharpe_tstat", 2.5)
        assert "✓显著" in r.summary()
        object.__setattr__(r, "sharpe_tstat", 1.0)
        assert "✗不显著" in r.summary()
        object.__setattr__(r, "sharpe_tstat", float("nan"))
        s = r.summary()
        assert "✓显著" not in s and "✗不显著" not in s, (
            "t 统计量为 NaN 时不得给出任何显著性结论")

    def test_significance_boundary_is_strictly_above_196(self):
        """t 恰好 1.96 不算显著（`>` 而非 `>=`）——这是常规约定，且区分值可精确构造。"""
        r = RiskReport.from_result(_result(n_days=120))
        object.__setattr__(r, "sharpe_tstat", 1.96)
        assert "✗不显著" in r.summary()

    def test_ruin_line_only_when_equity_hits_zero(self):
        """`if not isnan(dd) and dd <= -0.999` —— 删 not / 改 or 都会让熔断行乱出。"""
        r = RiskReport.from_result(_result(n_days=120))
        object.__setattr__(r, "max_drawdown", -0.5)
        assert "净值归零" not in r.summary()
        object.__setattr__(r, "max_drawdown", -0.9995)
        assert "净值归零" in r.summary()
        object.__setattr__(r, "max_drawdown", float("nan"))
        assert "净值归零" not in r.summary()

    def test_benchmark_block_only_when_beta_is_known(self):
        """`if not np.isnan(self.benchmark_beta):` —— 没基准时不得显示分解块。"""
        r = RiskReport.from_result(_result(n_days=120))
        assert "基准 Alpha/Beta 分解" not in r.summary()
        res = _result(n_days=120)
        r2 = RiskReport.from_result(res, benchmark_returns=res.net_returns * 0.5 + 1e-4)
        assert "基准 Alpha/Beta 分解" in r2.summary()

    def test_nan_metrics_render_as_na_not_as_numbers(self):
        """`_fmt` 的 `v is None or (isinstance(v,float) and isnan(v))`：
        `and` 改成 `or` 会让**所有** float 都显示成 N/A。"""
        r = RiskReport.from_result(_result(n_days=120))
        s = r.summary()
        assert "N/A" not in s.split("【基准")[0], f"正常数值被显示成 N/A：\n{s}"


# ===========================================================================
# E. to_dict —— 落库/前端读的就是它
# ===========================================================================

class TestToDict:

    def test_series_fields_are_excluded(self):
        """
        `equity_curve` / `net_returns` / `rolling_ic` 等序列字段不得进 to_dict，
        否则每次落库都会把整段序列塞进 JSON。
        """
        d = RiskReport.from_result(_result(n_days=120)).to_dict()
        for f in ("equity_curve", "gross_returns", "net_returns", "rolling_sharpe",
                  "rolling_ic", "drawdown_series", "decile_returns"):
            assert f not in d, f"{f} 出现在 to_dict 里"

    def test_nan_becomes_none_for_json(self):
        """
        `elif isinstance(val, float) and np.isnan(val): val = None`
        —— `and` 改成 `or` 后**任何 float** 都会变成 None，整份报告只剩空值。
        """
        d = RiskReport.from_result(_result(n_days=30), min_obs=60).to_dict()
        assert d["sharpe_ratio"] is None, "NaN 未被转成 JSON 可序列化的 None"
        assert d["insufficient_sample"] is True
        d2 = RiskReport.from_result(_result(n_days=120), min_obs=60).to_dict()
        assert isinstance(d2["sharpe_ratio"], float), (
            "正常数值被一并转成了 None —— NaN 判定疑似被放宽成 or")

    def test_timestamps_are_stringified(self):
        """
        `isinstance(val, (pd.Timestamp, pd.Period)) → str(val)`：
        回撤起止是 Timestamp，不转成字符串落库时会抛 JSON 序列化错误。

        ⚠️ 原来写成 `if d.get(f) is not None: assert ...` —— 字段一旦变成 None
        就什么都没查（§A）。120 天的回测必然有回撤区间，直接断言非空。
        """
        d = RiskReport.from_result(_result(n_days=120)).to_dict()
        for f in ("max_dd_start", "max_dd_end"):
            assert d.get(f) is not None, f"{f} 缺失 —— 120 天回测必然有回撤区间"
            assert isinstance(d[f], str), f"{f} 不是字符串，JSON 无法序列化"

    def test_repr_does_not_include_the_series(self):
        """
        八个 `field(repr=False)`：一次 `logger.info(report)` 不该打印整段序列。
        ⚠️ 必须**逐个**列全 —— 第一版漏了 `gross_returns` 与 `rolling_sharpe`，
        那两处的变异因此存活（DEV_LESSONS §S：审计单位错了）。
        """
        text = repr(RiskReport.from_result(_result(n_days=120)))
        for f in ("stress_test", "decile_returns", "equity_curve", "gross_returns",
                  "net_returns", "rolling_sharpe", "rolling_ic", "drawdown_series"):
            assert f not in text, f"{f} 出现在 repr 里 —— repr=False 疑似被改掉"
        assert len(text) < 3000, f"repr 长度 {len(text)}，序列疑似被打进了 repr"
