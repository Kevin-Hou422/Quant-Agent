"""
performance_analyzer.py —— 逐公式定钉测试（变异测试驱动）

来由：这是**所有绩效结论的出处**（晋级门看它的 Sharpe、报告看它的回撤、
策略门看它的 DSR），却是 A 档里变异点最多、保护最薄的模块 —— 92 个变异点。

既有覆盖只有两块：
  - `tests/test_phase4.py::TestDeflatedSharpe` —— 只测模块级 `deflated_sharpe_from_returns`
  - `tests/golden/test_golden_backtest.py::TestGoldenDrawdown` —— 只测 `max_drawdown` 一个值

也就是说 `annualized_return` / `annualized_volatility` / `sharpe_ratio` /
`sharpe_tstat` / `sortino_ratio` / `calmar_ratio` / `var_cvar` / `rolling_sharpe` /
`turnover_analysis` / `benchmark_analysis` / `stress_test` / `decile_analysis` /
Rank IC 全家 —— **一个都没有数值断言**。年化因子写成除法、收益公式的减号写成加号，
测试照样全绿。

本文件用一条**完全确定的**收益序列（无随机数）把这些公式逐个钉死。
参考值由该序列一次性算出并写死在下面的常量里，推导过程见每条用例的 docstring。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.backtest_engine.backtest_engine import BacktestResult
from app.core.backtest_engine.performance_analyzer import PerformanceAnalyzer

# ---------------------------------------------------------------------------
# 基准夹具：120 个交易日，4 日循环，**没有随机数**
#   cycle = [+1.2%, -0.8%, +0.8%, -1.0%]  → 日均 +0.05%
#   2023-01-02 起 120 个工作日 → 跨度 165 自然日
#   tdays = 120 / (165 / 365.25) = 265.6363636363636   ← 动态年化系数，非 252
# ---------------------------------------------------------------------------

N = 120
IDX = pd.bdate_range("2023-01-02", periods=N)
CYCLE = [0.012, -0.008, 0.008, -0.010]
RET = pd.Series((CYCLE * (N // len(CYCLE)))[:N], index=IDX)

TDAYS = 265.6363636363636          # = N / ((IDX[-1] - IDX[0]).days / 365.25)
MEAN_D = 0.0005                    # = (0.012 - 0.008 + 0.008 - 0.010) / 4
STD_D = 0.00967106052947172        # ret.std(ddof=1)

ANN_RETURN = 0.142004427660126     # (1 + 0.0005) ** TDAYS - 1
ANN_VOL = 0.15762237415489141      # STD_D * sqrt(TDAYS)
SHARPE = 0.9009154215668769        # ANN_RETURN / ANN_VOL   (rf = 0)
# ↓ **钉住当前（错误的）实现**，不是正确答案：年化 SR 配日频 √T，频率不一致。
#   已登记为缺陷 N-4；修好后这个常量要一起改。见 TestSharpeTStat 的类注释。
SHARPE_T = 8.32356013267212        # SHARPE * sqrt(120) / sqrt(1 + 0.5 * SHARPE**2)
SORTINO = 8.639895768446092
CALMAR = 14.111035246689319
MAX_DD = -0.010063360000000183
MAX_DD_DURATION = 5


def _result(ret: pd.Series,
            positions: pd.DataFrame | None = None,
            signal: pd.DataFrame | None = None,
            turnover: float = 0.2,
            cost_bps: float = 3.0) -> BacktestResult:
    cols = ["A", "B"]
    return BacktestResult(
        equity_curve=(1 + ret).cumprod(),
        gross_returns=ret,
        net_returns=ret,
        positions=positions if positions is not None
        else pd.DataFrame(0.1, index=ret.index, columns=cols),
        trade_log=pd.DataFrame(),
        turnover=pd.Series(turnover, index=ret.index),
        signal=signal if signal is not None
        else pd.DataFrame(0.0, index=ret.index, columns=cols),
        daily_cost_bps=pd.Series(cost_bps, index=ret.index),
    )


@pytest.fixture
def pa() -> PerformanceAnalyzer:
    return PerformanceAnalyzer(_result(RET), rf_annual=0.0)


# ===========================================================================
# A. 动态年化系数 —— 所有年化指标的公共因子
# ===========================================================================

class TestDynamicTradingDays:
    """
    `_tdays = len(idx) / ((idx[-1] - idx[0]).days / 365.25)`，
    数据点 < 2 时回退 252。这个系数乘进了几乎每一个年化指标，
    它错了所有绩效数字一起错，却此前没有任何断言。
    """

    def test_tdays_is_derived_from_the_actual_date_span(self, pa):
        span_days = (IDX[-1] - IDX[0]).days
        assert span_days == 165
        assert pa._tdays == pytest.approx(N / (span_days / 365.25), abs=1e-9)
        assert pa._tdays == pytest.approx(TDAYS, abs=1e-9)

    def test_tdays_is_not_the_hardcoded_252(self, pa):
        """若有人把动态计算改回写死 252，本条立刻红。"""
        assert abs(pa._tdays - 252.0) > 1.0

    def test_single_point_series_falls_back_to_252(self):
        """`if len(idx) < 2:` 的边界：1 个点回退，2 个点必须走动态分支。"""
        one = pd.Series([0.01], index=IDX[:1])
        assert PerformanceAnalyzer(_result(one))._tdays == 252.0

    def test_two_point_series_uses_dynamic_branch(self):
        """把 `< 2` 放宽成 `<= 2` 时，2 个点会被误判为"数据不足"退回 252。"""
        two = pd.Series([0.01, -0.01], index=IDX[:2])
        t = PerformanceAnalyzer(_result(two))._tdays
        assert t != 252.0, "2 个数据点应当走动态计算，而不是回退常量"
        assert t == pytest.approx(2 / ((IDX[1] - IDX[0]).days / 365.25), abs=1e-9)


# ===========================================================================
# B. 收益 / 波动 / 夏普 —— 公式本身
# ===========================================================================

class TestReturnVolSharpe:

    def test_annualized_return_exact(self, pa):
        """
        `(1 + mean_d) ** tdays - 1`。两个符号都能改坏而不被发现：
          `1 + mean_d` → `1 - mean_d`：正收益变成负收益
          `- 1`        → `+ 1`：结果整体偏移 2（0.142 → 2.142）
        """
        assert pa.annualized_return() == pytest.approx(ANN_RETURN, abs=1e-12)
        assert pa.annualized_return() == pytest.approx(
            (1 + MEAN_D) ** TDAYS - 1, abs=1e-12)

    def test_annualized_return_sign_follows_mean(self):
        """日均为负 → 年化必须为负。`1 + mean` 写成 `1 - mean` 会让它变正。"""
        neg = PerformanceAnalyzer(_result(-RET))
        assert neg.annualized_return() < 0

    def test_annualized_volatility_exact(self, pa):
        """`std(ddof=1) * sqrt(tdays)`；写成 `/` 会小 265 倍。"""
        assert pa.annualized_volatility() == pytest.approx(ANN_VOL, abs=1e-12)
        assert pa.annualized_volatility() == pytest.approx(
            STD_D * np.sqrt(TDAYS), abs=1e-12)

    def test_annualized_volatility_uses_sample_std(self, pa):
        """ddof=1（样本）而非 ddof=0（总体）—— 两者在 N=120 时差 0.4%。"""
        assert pa.annualized_volatility() != pytest.approx(
            RET.std(ddof=0) * np.sqrt(TDAYS), abs=1e-9)

    def test_sharpe_exact_without_risk_free(self, pa):
        assert pa.sharpe_ratio() == pytest.approx(SHARPE, abs=1e-12)

    def test_sharpe_subtracts_risk_free_not_adds(self):
        """
        `(ann_return - rf_annual) / vol`。把 `-` 写成 `+` 时，
        rf=5% 会让夏普**上升**而不是下降 —— 方向完全反了。
        """
        with_rf = PerformanceAnalyzer(_result(RET), rf_annual=0.05)
        assert with_rf.sharpe_ratio() == pytest.approx(0.5837015725300244, abs=1e-12)
        assert with_rf.sharpe_ratio() < SHARPE, "扣掉无风险利率后夏普反而变高"

    def test_zero_volatility_gives_nan_not_division_error(self):
        """
        `if vol > 0 else nan` 的边界。放宽成 `>=` 后 vol==0 会真的去做除法
        （Python float 除零直接抛 ZeroDivisionError，整份报告崩掉）。

        ⚠️ 构造要用**全 0** 收益：`pd.Series(0.001)` 的 std(ddof=1) 是
        1.06e-17 而不是 0（浮点求和残差），vol > 0 依然成立，测不到这个边界。
        """
        flat = PerformanceAnalyzer(_result(pd.Series(0.0, index=IDX)))
        assert flat.annualized_volatility() == 0.0
        assert np.isnan(flat.sharpe_ratio())

    def test_rf_annual_and_rf_daily_are_consistent(self):
        """
        `rf_annual = rf * tdays`（当只给了日频 rf 时）。写成 `/` 会让年化无风险
        利率变成日频的 1/265，扣减项几乎消失。
        """
        pa_d = PerformanceAnalyzer(_result(RET), rf=0.0002)
        assert pa_d.rf_annual == pytest.approx(0.0002 * TDAYS, abs=1e-12)
        assert pa_d.rf == pytest.approx(0.0002, abs=1e-15)

    def test_rf_annual_takes_priority_over_rf_daily(self):
        """两个都给时以年化为准，并反算出日频。"""
        pa_a = PerformanceAnalyzer(_result(RET), rf=0.09, rf_annual=0.05)
        assert pa_a.rf_annual == pytest.approx(0.05, abs=1e-15)
        assert pa_a.rf == pytest.approx(0.05 / TDAYS, abs=1e-15)


class TestSharpeTStat:
    """
    t = SR × √T / √(1 + 0.5 × SR²)（Lo 2002）。

    **本组断言钉住的是当前实现，不是正确答案。** 外部审计 2026-09-15（N-4）
    指出：这里的 `SR` 取的是**年化** Sharpe（`ANN_RETURN / ANN_VOL`，
    TDAYS≈265.6），`√T` 取的却是**日频**观测数 √120 —— 两个频率不一致。
    同一组收益，按同频日 Sharpe 代入同一分母得 0.5660，单样本 t 参考 0.5664，
    而产品给出 8.3236；`risk_report.py` 按 1.96 判显著，于是这组收益被显示成
    "✓显著"，正确口径下是"✗不显著"。

    `SHARPE_T` 这个常量是**照着实现算出来的**（见其行内注释的公式），
    所以它检测得了"公式被改动"，检测不了"公式本来就错"。
    应有行为由 `test_known_defects.py::TestSharpeTStatFrequency` 以 xfail 断言，
    修好之后本组常量必须同步改掉。
    """

    def test_sharpe_tstat_exact(self, pa):
        assert pa.sharpe_tstat() == pytest.approx(SHARPE_T, abs=1e-10)
        assert pa.sharpe_tstat() == pytest.approx(
            SHARPE * np.sqrt(N) / np.sqrt(1.0 + 0.5 * SHARPE ** 2), abs=1e-10)

    def test_sharpe_tstat_denominator_sign(self, pa):
        """
        分母 `1.0 + 0.5*SR²` 写成 `1.0 - 0.5*SR²`：本例 SR≈0.9 时
        分母从 1.406 变成 0.594，t 从 8.32 跳到 12.8 —— **虚假显著**。
        SR > √2 时更会开负数的根号变成 NaN。
        """
        wrong = SHARPE * np.sqrt(N) / np.sqrt(1.0 - 0.5 * SHARPE ** 2)
        assert pa.sharpe_tstat() != pytest.approx(wrong, abs=1e-6)
        assert pa.sharpe_tstat() < wrong

    def test_sharpe_tstat_scales_with_sqrt_T(self):
        """
        `SR × √T` 写成 `SR / √T` 时 t 会随样本增大而**变小**。
        同一个 SR、样本翻倍 → t 必须按 √2 增大。
        """
        long_idx = pd.bdate_range("2023-01-02", periods=240)
        long_ret = pd.Series((CYCLE * 60)[:240], index=long_idx)
        t_long = PerformanceAnalyzer(_result(long_ret)).sharpe_tstat()
        assert t_long > pa_t(RET), "样本翻倍后 t 反而没变大 —— √T 的方向疑似反了"

    def test_sharpe_tstat_is_nan_when_sharpe_is_nan(self):
        flat = PerformanceAnalyzer(_result(pd.Series(0.0, index=IDX)))
        assert np.isnan(flat.sharpe_tstat())


def pa_t(ret: pd.Series) -> float:
    return PerformanceAnalyzer(_result(ret)).sharpe_tstat()


# ===========================================================================
# C. 回撤
# ===========================================================================

class TestDrawdown:

    def test_max_drawdown_exact(self, pa):
        dd, peak, trough, dur = pa.max_drawdown()
        assert dd == pytest.approx(MAX_DD, abs=1e-12)
        assert dur == MAX_DD_DURATION
        assert peak < trough, "峰值日期必须早于谷底日期"

    def test_drawdown_series_is_non_positive_and_starts_at_zero(self, pa):
        """
        `(ec - ec.cummax()) / ec.cummax()`：把 `-` 写成 `+` 后
        回撤序列会变成 ≈ +2 的正数 —— "回撤"变成了两倍净值比。
        """
        s = pa.drawdown_series()
        assert (s <= 1e-15).all(), f"回撤序列出现正值：max={s.max()}"
        assert s.iloc[0] == pytest.approx(0.0, abs=1e-15)
        assert s.min() == pytest.approx(MAX_DD, abs=1e-12)

    def test_non_datetime_index_is_rejected_before_reaching_the_fallback(self):
        """
        `max_drawdown` 里有一条 else 分支 `int(trough_idx - peak_idx)`，
        用意是"索引不是日期时按序号相减"。**这条分支实际不可达**：
        `__init__` 里就会取 `self._tdays`，而它无条件做 `(idx[-1] - idx[0]).days`，
        整数索引在那里先抛 AttributeError。

        这条用例把"不可达"钉成事实——L151 的 `-` → `+` 因此是等价变异。
        同时它也是一份缺陷记录：非日期索引下报错信息指向 `_tdays` 的内部实现，
        而不是"索引必须是 DatetimeIndex"。（产品问题，登记在 MUTATION_LEDGER。）
        """
        ret = pd.Series([0.10, -0.20, 0.05, 0.02], index=[0, 1, 2, 3])
        with pytest.raises(AttributeError, match="days"):
            PerformanceAnalyzer(_result(ret))

    def test_calmar_is_annual_return_over_abs_drawdown(self, pa):
        assert pa.calmar_ratio() == pytest.approx(CALMAR, abs=1e-9)
        assert pa.calmar_ratio() == pytest.approx(ANN_RETURN / abs(MAX_DD), abs=1e-9)

    def test_calmar_is_nan_without_drawdown(self):
        """净值单调上升 → 回撤为 0 → Calmar 无定义，不得返回 inf。"""
        up = PerformanceAnalyzer(_result(pd.Series(0.001, index=IDX)))
        assert np.isnan(up.calmar_ratio())


# ===========================================================================
# D. 下行风险
# ===========================================================================

class TestSortinoAndTail:

    def test_sortino_exact(self, pa):
        assert pa.sortino_ratio() == pytest.approx(SORTINO, abs=1e-9)

    def test_zero_return_day_is_not_downside(self):
        """
        `neg = ret[ret < rf]`：rf=0 时**恰好为 0** 的那天不算下行。
        放宽成 `<=` 会把它计入，下行标准差被拉低 → Sortino 虚高。
        区分值就是 0.0 本身，可精确构造。
        """
        ret = pd.Series(([0.012, -0.008, 0.0, -0.010] * 30)[:N], index=IDX)
        pa0 = PerformanceAnalyzer(_result(ret))
        neg = ret[ret < 0.0]
        assert len(neg) == 60, "构造有误：应恰好 60 个负收益日"
        expected = pa0.annualized_return() / (neg.std(ddof=1) * np.sqrt(TDAYS))
        assert pa0.sortino_ratio() == pytest.approx(expected, abs=1e-12)

    def test_sortino_needs_more_than_one_downside_day(self):
        """
        `if len(neg) < 2: return nan` 的边界：**恰好 2 个**下行日必须能算出来。
        放宽成 `<= 2` 会把 2 个下行日误判为样本不足。
        """
        vals = [0.001] * (N - 2) + [-0.02, -0.01]
        ret = pd.Series(vals, index=IDX)
        s = PerformanceAnalyzer(_result(ret)).sortino_ratio()
        assert not np.isnan(s), "恰好 2 个下行日却返回 NaN —— 边界判错"

    def test_sortino_is_nan_with_single_downside_day(self):
        vals = [0.001] * (N - 1) + [-0.02]
        assert np.isnan(PerformanceAnalyzer(
            _result(pd.Series(vals, index=IDX))).sortino_ratio())

    def test_sortino_is_nan_when_downside_std_is_zero(self):
        """
        `if downside_std > 0 else nan` 的边界：所有下行日同值 → std=0，
        放宽成 `>=` 会真的去除以 0。
        """
        vals = [0.001] * (N - 4) + [-0.02] * 4
        s = PerformanceAnalyzer(_result(pd.Series(vals, index=IDX))).sortino_ratio()
        assert np.isnan(s)

    def test_var_cvar_uses_alpha_as_percent(self):
        """
        `np.percentile(ret, alpha * 100)`。写成 `alpha / 100` 会把 5% 分位
        变成 0.0005% 分位 —— 实际取到的是**最差单日**，VaR 被严重高估。
        构造：只有一天是 -5%，其余不低于 -1% → 两者取值不同。
        """
        vals = [-0.05] + [0.01, -0.01] * ((N - 1) // 2) + [0.01]
        ret = pd.Series(vals[:N], index=IDX)
        var, cvar = PerformanceAnalyzer(_result(ret)).var_cvar(0.05)
        assert var == pytest.approx(-np.percentile(ret, 5.0), abs=1e-12)
        assert var != pytest.approx(0.05, abs=1e-6), "VaR 取到了最差单日 —— alpha 口径疑似写反"
        assert cvar >= var, "CVaR（尾部均值）不可能小于 VaR"

    def test_cvar_is_mean_of_the_tail(self, pa):
        var, cvar = pa.var_cvar(0.05)
        assert var == pytest.approx(0.01, abs=1e-12)
        assert cvar == pytest.approx(0.01, abs=1e-12)


# ===========================================================================
# E. 滚动指标与换手
# ===========================================================================

class TestRollingAndTurnover:

    def test_rolling_sharpe_is_annualised(self, pa):
        """`(mean/std) * sqrt(tdays)`；写成 `/` 会小 265 倍。"""
        rs = pa.rolling_sharpe(20).dropna()
        assert len(rs) > 0
        assert rs.iloc[-1] == pytest.approx(0.8247429485265863, abs=1e-9)
        # 与手算的最后一个窗口一致
        win = RET.iloc[-20:]
        assert rs.iloc[-1] == pytest.approx(
            win.mean() / win.std(ddof=1) * np.sqrt(TDAYS), abs=1e-9)

    def test_turnover_annualisation_exact(self, pa):
        d = pa.turnover_analysis()
        assert d["mean_daily_turnover"] == pytest.approx(0.2, abs=1e-12)
        assert d["ann_turnover"] == pytest.approx(0.2 * TDAYS, abs=1e-9)
        assert d["cost_drag_bps"] == pytest.approx(3.0 * TDAYS, abs=1e-9)

    def test_turnover_scales_with_daily_value(self):
        """换手翻倍 → 年化换手必须翻倍（乘法写成除法会反向）。"""
        a = PerformanceAnalyzer(_result(RET, turnover=0.2)).turnover_analysis()
        b = PerformanceAnalyzer(_result(RET, turnover=0.4)).turnover_analysis()
        assert b["ann_turnover"] == pytest.approx(2 * a["ann_turnover"], abs=1e-9)


# ===========================================================================
# F. Rank IC 家族 —— 前视方向与最小样本
# ===========================================================================

class TestRankIC:

    @staticmethod
    def _panel(n_tickers: int = 6, n_days: int = 40):
        idx = pd.bdate_range("2023-01-02", periods=n_days)
        cols = [f"T{i}" for i in range(n_tickers)]
        # 信号 = 每日固定的横截面排序；价格让排名靠前的涨得多 → IC 应为正
        sig = pd.DataFrame(
            np.tile(np.arange(n_tickers, dtype=float), (n_days, 1)),
            index=idx, columns=cols)
        step = 1 + 0.001 * np.arange(n_tickers, dtype=float)
        px = pd.DataFrame(
            100 * np.cumprod(np.tile(step, (n_days, 1)), axis=0),
            index=idx, columns=cols)
        return sig, px

    def test_exact_rank_ic_uses_forward_returns(self):
        """
        `fwd_ret = prices.pct_change().shift(-1)` —— 信号对齐的是**下一期**收益。
        构造成"信号排名越高、次日涨得越多"，IC 必须恒为 +1。
        """
        sig, px = self._panel()
        res = _result(RET.iloc[:len(sig)].set_axis(sig.index), signal=sig)
        ic = PerformanceAnalyzer(res).rolling_rank_ic_from_prices(sig, px)
        assert len(ic) > 0, "一条 IC 都没算出来"
        assert ic.mean() == pytest.approx(1.0, abs=1e-9)

    def test_rank_ic_needs_at_least_five_valid_names(self):
        """
        `if mask.sum() < 5: continue` 的边界：**恰好 5 个**有效名必须参与计算。
        放宽成 `<= 5` 会把 5 名的截面丢掉，窄票池的 IC 整段消失。
        """
        sig, px = self._panel(n_tickers=5)
        res = _result(RET.iloc[:len(sig)].set_axis(sig.index), signal=sig)
        ic = PerformanceAnalyzer(res).rolling_rank_ic_from_prices(sig, px)
        assert len(ic) > 0, "恰好 5 个标的时 IC 被整段丢弃 —— 最小样本边界判错"

    def test_four_names_are_rejected(self):
        """对照组：4 个名确实不够，否则上一条无法区分。"""
        sig, px = self._panel(n_tickers=4)
        res = _result(RET.iloc[:len(sig)].set_axis(sig.index), signal=sig)
        ic = PerformanceAnalyzer(res).rolling_rank_ic_from_prices(sig, px)
        assert len(ic) == 0

    def test_approx_rank_ic_uses_next_day_positions(self):
        """
        近似版用 `pos_arr[t + 1]`（**下期**持仓）代理收益。写成 `t - 1` 就成了上期。

        ⚠️ 持仓若逐日不变，`t+1` 与 `t-1` 取到的是同一行，**测不出差别**
        （第一版就是这样，变异测试证实它存活）。这里让持仓**逐日循环右移**：
        对齐到 t+1 时 IC 恒为 +1，对齐到 t-1 时会得到完全不同的排序。
        """
        n_days, n_tickers = 30, 6
        idx = pd.bdate_range("2023-01-02", periods=n_days)
        cols = [f"T{i}" for i in range(n_tickers)]
        base = np.arange(n_tickers, dtype=float)
        # 持仓第 t 天 = base 右移 t 位；信号第 t 天 = 下一天的持仓（即右移 t+1 位）
        pos = np.stack([np.roll(base, t) for t in range(n_days)])
        sig = np.stack([np.roll(base, t + 1) for t in range(n_days)])
        pos_df = pd.DataFrame(pos, index=idx, columns=cols)
        sig_df = pd.DataFrame(sig, index=idx, columns=cols)
        ret = pd.Series(0.001, index=idx)
        ic = PerformanceAnalyzer(
            _result(ret, positions=pos_df, signal=sig_df)).rolling_rank_ic()
        assert len(ic) > 0
        assert ic.mean() == pytest.approx(1.0, abs=1e-9), (
            "信号与**下一期**持仓完全同序，IC 应恒为 +1 —— 对齐方向疑似取成了上一期")

    def test_approx_rank_ic_needs_five_valid_names(self):
        """近似版 IC 的同一个最小样本边界（L210），与精确版是两份独立代码。"""
        n_days = 20
        idx = pd.bdate_range("2023-01-02", periods=n_days)
        for n_tickers, expect_any in ((5, True), (4, False)):
            cols = [f"T{i}" for i in range(n_tickers)]
            base = np.arange(n_tickers, dtype=float)
            frame = pd.DataFrame(np.tile(base, (n_days, 1)), index=idx, columns=cols)
            res = _result(pd.Series(0.001, index=idx), positions=frame, signal=frame)
            ic = PerformanceAnalyzer(res).rolling_rank_ic()
            assert (len(ic) > 0) is expect_any, (
                f"{n_tickers} 个标的时 IC 是否产出与预期不符（最小样本应为 5）")

    def test_ic_ir_is_mean_over_sample_std(self, pa):
        ic = pd.Series([0.02, 0.04, -0.01, 0.03, 0.00])
        assert pa.ic_ir(ic) == pytest.approx(ic.mean() / ic.std(ddof=1), abs=1e-12)

    def test_ic_ir_uses_the_provided_series_not_the_approximation(self, pa):
        """
        `ic = ic_series if ic_series is not None else self.rolling_rank_ic()`
        删掉 `not` 后，**明确传进来的** IC 序列会被忽略、改用近似版重算。
        本夹具的信号是常数 → 近似版算不出 IC → 结果会是 NaN 而不是下面这个值。
        """
        ic = pd.Series([0.05, 0.03])
        assert pa.ic_ir(ic) == pytest.approx(ic.mean() / ic.std(ddof=1), abs=1e-12)

    def test_ic_ir_needs_at_least_two_points(self, pa):
        """`if len(ic) < 2 ...` 的边界：**恰好 2 个** IC 必须能算出 IR。"""
        assert not np.isnan(pa.ic_ir(pd.Series([0.05, 0.03])))
        assert np.isnan(pa.ic_ir(pd.Series([0.05])))

    def test_ic_ir_is_nan_for_constant_ic(self, pa):
        assert np.isnan(pa.ic_ir(pd.Series([0.02, 0.02, 0.02])))

    def test_ic_decay_needs_five_valid_names(self):
        """
        衰减曲线里的第三份最小样本判定（L282）。**两侧都要测**：
        恰好 5 个必须算得出来，4 个必须整段为 NaN —— 只测 4 个的那一侧
        无法区分 `< 5` 与 `<= 5`（第一版就是这样）。
        """
        sig5, px5 = self._panel(n_tickers=5, n_days=60)
        res5 = _result(pd.Series(0.001, index=sig5.index), signal=sig5)
        curve5 = PerformanceAnalyzer(res5).ic_decay_curve(sig5, px5, horizons=[1, 5])
        assert curve5.notna().all(), "恰好 5 个标的时衰减曲线整段为空 —— 最小样本边界判错"

        sig4, px4 = self._panel(n_tickers=4, n_days=60)
        res4 = _result(pd.Series(0.001, index=sig4.index), signal=sig4)
        curve4 = PerformanceAnalyzer(res4).ic_decay_curve(sig4, px4, horizons=[1, 5])
        assert curve4.isna().all(), "4 个标的不足以算 IC，却给出了数值"

    def test_ic_decay_curve_covers_all_horizons(self):
        """
        `for t in range(n - h)` —— 写成 `n + h` 会越界取到 NaN 行甚至 IndexError。
        同时确认每个 horizon 都真的产出了数值（不是整列 NaN）。
        """
        sig, px = self._panel(n_days=90)
        res = _result(pd.Series(0.001, index=sig.index), signal=sig)
        curve = PerformanceAnalyzer(res).ic_decay_curve(sig, px, horizons=[1, 5, 20])
        assert list(curve.index) == [1, 5, 20]
        assert curve.notna().all(), f"有 horizon 没算出 IC：{curve.to_dict()}"
        assert (curve > 0.9).all(), "构造成单调正相关，各 horizon 的 IC 都应接近 +1"


# ===========================================================================
# F2. 分档分析与多空腿 —— 此前零断言
# ===========================================================================

class TestDecileAndLegs:

    @staticmethod
    def _monotone_panel(n_days: int = 60, n_t: int = 10):
        idx = pd.bdate_range("2023-01-02", periods=n_days)
        cols = [f"T{i}" for i in range(n_t)]
        sig = pd.DataFrame(np.tile(np.arange(n_t, dtype=float), (n_days, 1)),
                           index=idx, columns=cols)
        step = 1 + 0.001 * np.arange(n_t, dtype=float)
        px = pd.DataFrame(100 * np.cumprod(np.tile(step, (n_days, 1)), axis=0),
                          index=idx, columns=cols)
        return idx, cols, sig, px

    def test_decile_analysis_is_monotone_for_a_monotone_signal(self):
        """
        构造"信号越高、次日收益越高"，10 档的平均下期收益必须**严格单调递增**，
        且第 d 档恰好等于第 d 只标的的日收益 0.001×(d-1)。

        这一条同时钉住四处：分档循环的上界 `n_deciles + 1`、
        分位切点、`prices is not None` 的分支方向、以及"用下一期收益"。
        """
        idx, cols, sig, px = self._monotone_panel()
        res = _result(pd.Series(0.001, index=idx),
                      positions=pd.DataFrame(0.1, index=idx, columns=cols), signal=sig)
        d = PerformanceAnalyzer(res).decile_analysis(prices=px)
        assert list(d.index) == list(range(1, 11))
        assert d.notna().all(), f"有档位没有样本：{d.to_dict()}"
        assert (d.diff().dropna() > 0).all(), f"分档收益非单调递增：{d.to_dict()}"
        for k in range(1, 11):
            assert d.loc[k] == pytest.approx(0.001 * (k - 1), abs=1e-9)

    def test_decile_falls_back_to_position_diff_without_prices(self):
        """
        `if prices is not None:` 删掉 `not` 后，**给了价格**反而走持仓差分的降级路径。
        两条路径的数值完全不同：给价格时第 10 档是 0.009，
        持仓恒定时差分全为 0 → 所有档位都是 0。
        """
        idx, cols, sig, px = self._monotone_panel()
        res = _result(pd.Series(0.001, index=idx),
                      positions=pd.DataFrame(0.1, index=idx, columns=cols), signal=sig)
        pa_ = PerformanceAnalyzer(res)
        with_px = pa_.decile_analysis(prices=px)
        without = pa_.decile_analysis()
        assert with_px.loc[10] == pytest.approx(0.009, abs=1e-9)
        assert without.loc[10] == pytest.approx(0.0, abs=1e-12)
        assert not np.allclose(with_px.to_numpy(), without.to_numpy())

    def test_decile_cuts_are_half_open_except_the_last(self):
        """
        `dmask = (s >= lo) & (s <= hi if d == n_deciles else s < hi)` ——
        除最后一档外区间是**左闭右开**，否则切点上的标的会被两个档位重复计入。

        ⚠️ 10 个等距值配 10 档时切点落在**值与值之间**（0.9、1.8…），
        `<` 与 `<=` 没有区别，测不出来（第一版就是这样）。
        改用 **11 个等距值**：分位切点恰好落在 0,1,2,…,10 这些**数据点本身**上，
        `<=` 会让第 d 档多吃一个标的。
        """
        n_days, n_t = 40, 11
        idx = pd.bdate_range("2023-01-02", periods=n_days)
        cols = [f"T{i}" for i in range(n_t)]
        sig = pd.DataFrame(np.tile(np.arange(n_t, dtype=float), (n_days, 1)),
                           index=idx, columns=cols)
        step = 1 + 0.001 * np.arange(n_t, dtype=float)
        px = pd.DataFrame(100 * np.cumprod(np.tile(step, (n_days, 1)), axis=0),
                          index=idx, columns=cols)
        res = _result(pd.Series(0.001, index=idx),
                      positions=pd.DataFrame(0.1, index=idx, columns=cols), signal=sig)
        d = PerformanceAnalyzer(res).decile_analysis(prices=px)
        cuts = np.percentile(np.arange(n_t, dtype=float), np.linspace(0, 100, 11))
        assert np.allclose(cuts, np.arange(11.0)), "构造前提：切点必须落在数据点上"
        # 第 1 档只应含标的 0（收益 0.000），若右端改成闭区间会把标的 1 也算进来
        assert d.loc[1] == pytest.approx(0.0, abs=1e-12), (
            f"第 1 档混入了切点上的标的：{d.loc[1]}")
        # 最后一档是闭区间，含标的 9 与 10
        assert d.loc[10] == pytest.approx((0.009 + 0.010) / 2, abs=1e-12)

    def test_empty_decile_bucket_does_not_poison_the_average(self):
        """
        `if dmask.sum() > 0:` —— 空档位当日**不记录**。放宽成 `>=` 会把
        `np.mean(空数组)` 的 NaN 塞进去，该档位的均值从此永远是 NaN
        （并且每天抛一个 "Mean of empty slice" 警告）。

        构造：前半段信号取值密集在两端（多个档位当日为空），后半段取值分散。
        """
        import warnings
        n_days, n_t = 40, 10
        idx = pd.bdate_range("2023-01-02", periods=n_days)
        cols = [f"T{i}" for i in range(n_t)]
        dense = np.array([0.0] * 5 + [9.0] * 5)              # 只有两端有值 → 中间档位空
        spread = np.arange(n_t, dtype=float)
        rows = [dense if t < n_days // 2 else spread for t in range(n_days)]
        sig = pd.DataFrame(np.stack(rows), index=idx, columns=cols)
        step = 1 + 0.001 * np.arange(n_t, dtype=float)
        px = pd.DataFrame(100 * np.cumprod(np.tile(step, (n_days, 1)), axis=0),
                          index=idx, columns=cols)
        res = _result(pd.Series(0.001, index=idx),
                      positions=pd.DataFrame(0.1, index=idx, columns=cols), signal=sig)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            d = PerformanceAnalyzer(res).decile_analysis(prices=px)
        assert d.notna().sum() >= 5, (
            f"多数档位变成 NaN —— 空档位的 NaN 疑似被记进了均值：{d.to_dict()}")

    def test_decile_skips_cross_sections_with_too_few_names(self):
        """`if mask.sum() < n_deciles: continue` —— 标的数少于档数时该日整段跳过。"""
        idx, cols, sig, px = self._monotone_panel(n_t=6)
        res = _result(pd.Series(0.001, index=idx),
                      positions=pd.DataFrame(0.1, index=idx, columns=cols), signal=sig)
        d = PerformanceAnalyzer(res).decile_analysis(prices=px, n_deciles=10)
        assert d.isna().all(), "6 个标的不足以分 10 档，却给出了数值"

    def test_leg_analysis_exact(self):
        """
        多空腿各自年化与夏普。此前**完全没有断言** —— 两条腿的公式与主口径
        是各自独立的一份代码（`_ann_sharpe` 内联函数），主口径测到了不代表它们对。
        """
        n_days = 60
        idx = pd.bdate_range("2023-01-02", periods=n_days)
        long_r = pd.Series(([0.01, -0.002] * (n_days // 2))[:n_days], index=idx)
        short_r = pd.Series(([-0.004, 0.003] * (n_days // 2))[:n_days], index=idx)
        base = _result(pd.Series(0.001, index=idx))
        res = BacktestResult(
            equity_curve=base.equity_curve, gross_returns=base.gross_returns,
            net_returns=base.net_returns, positions=base.positions,
            trade_log=base.trade_log, turnover=base.turnover, signal=base.signal,
            daily_cost_bps=base.daily_cost_bps,
            long_returns=long_r, short_returns=short_r)
        legs = PerformanceAnalyzer(res).leg_analysis()
        assert legs["long_ann_return"] == pytest.approx(1.9448666394772562, abs=1e-9)
        assert legs["long_sharpe"] == pytest.approx(19.54162743300988, abs=1e-9)
        assert legs["short_ann_return"] == pytest.approx(-0.12655630320183053, abs=1e-9)
        assert legs["short_sharpe"] == pytest.approx(-2.179906611239585, abs=1e-9)
        assert legs["long_ann_return"] > 0 > legs["short_ann_return"], "两条腿的方向反了"

        # ⚠️ rf_annual=0 时 `ann_ret - rf` 与 `ann_ret + rf` 完全相同，
        #    单腿夏普里那个减号测不出来（第一版就是这样）。加一组带无风险利率的。
        pa_rf = PerformanceAnalyzer(res, rf_annual=0.30)
        legs_rf = pa_rf.leg_analysis()
        vol_l = float(long_r.std(ddof=1) * np.sqrt(pa_rf._tdays))
        assert legs_rf["long_sharpe"] == pytest.approx(
            (legs["long_ann_return"] - 0.30) / vol_l, abs=1e-9)
        assert legs_rf["long_sharpe"] < legs["long_sharpe"], (
            "扣掉无风险利率后多头腿夏普反而更高 —— 减号疑似写成了加号")

    def test_leg_sharpe_is_nan_when_a_leg_has_no_volatility(self):
        """单腿 `if vol > 0 else nan` 的边界：放宽成 `>=` 会对 0 波动做除法。"""
        n_days = 60
        idx = pd.bdate_range("2023-01-02", periods=n_days)
        flat = pd.Series(0.0, index=idx)
        base = _result(pd.Series(0.001, index=idx))
        res = BacktestResult(
            equity_curve=base.equity_curve, gross_returns=base.gross_returns,
            net_returns=base.net_returns, positions=base.positions,
            trade_log=base.trade_log, turnover=base.turnover, signal=base.signal,
            daily_cost_bps=base.daily_cost_bps,
            long_returns=flat, short_returns=flat)
        legs = PerformanceAnalyzer(res).leg_analysis()
        assert np.isnan(legs["long_sharpe"]) and np.isnan(legs["short_sharpe"])
        assert legs["long_ann_return"] == pytest.approx(0.0, abs=1e-15)

    def test_leg_analysis_is_nan_when_legs_are_absent(self, pa):
        """
        `_ann_sharpe(x) if x is not None else (nan, nan)` —— 删掉 `not` 后
        会对 None 调用 `.fillna()` 抛异常。缺腿数据时必须安静地给 NaN。
        """
        legs = pa.leg_analysis()
        assert all(np.isnan(v) for v in legs.values())


# ===========================================================================
# G. 基准分解与压力测试
# ===========================================================================

class TestBenchmarkAndStress:

    BENCH = pd.Series(([0.006, -0.004] * (N // 2))[:N], index=IDX)

    def test_benchmark_beta_exact(self, pa):
        d = pa.benchmark_analysis(self.BENCH)
        assert d["benchmark_beta"] == pytest.approx(1.9, abs=1e-4)

    def test_benchmark_alpha_is_net_of_beta_exposure(self, pa):
        """
        `alpha_daily = r.mean() - beta * b.mean()`。把 `-` 写成 `+` 会把
        基准收益**加**进 alpha —— 一个纯 beta 策略会显示出巨大 alpha。
        本例基准年化 30.4%、策略 14.2%、beta 1.9 → alpha 必须是负的。
        """
        d = pa.benchmark_analysis(self.BENCH)
        assert d["benchmark_alpha"] < 0, f"beta 敞口未被扣除：{d}"
        assert d["benchmark_alpha"] == pytest.approx(-0.3108, abs=1e-4)

    def test_benchmark_tracking_error_and_ir(self, pa):
        d = pa.benchmark_analysis(self.BENCH)
        assert d["tracking_error"] == pytest.approx(0.0259, abs=1e-4)
        # IR 用**未取整**的 alpha 与 te 相除后再取整，不能拿输出里已取整的两个数反推
        assert d["information_ratio"] == pytest.approx(-12.0083, abs=1e-4)
        assert d["information_ratio"] < 0, "alpha 为负时 IR 不可能为正"

    def test_benchmark_ann_return_exact(self, pa):
        """基准自身的年化收益也是一份独立公式（L431），与策略口径分开算。"""
        d = pa.benchmark_analysis(self.BENCH)
        assert d["benchmark_ann_ret"] == pytest.approx(0.3041, abs=1e-4)
        assert d["benchmark_ann_ret"] > 0, "基准日均为正，年化不可能为负"

    def test_benchmark_too_short_returns_all_nan(self):
        """`if len(common) < 20` —— 不足 20 个共同交易日时不得给出任何数字。"""
        short = pd.Series(0.001, index=IDX[:10])
        pa_s = PerformanceAnalyzer(_result(RET.iloc[:10]))
        d = pa_s.benchmark_analysis(short)
        assert all(np.isnan(v) for v in d.values())

    def test_benchmark_exactly_twenty_days_is_enough(self):
        """
        `< 20` 的边界：**恰好 20 天**必须算，放宽成 `<= 20` 会多卡一天。
        区分值就是 20 本身，可精确构造。
        """
        idx20 = IDX[:20]
        ret20 = RET.iloc[:20]
        bench20 = pd.Series(([0.006, -0.004] * 10), index=idx20)
        d = PerformanceAnalyzer(_result(ret20)).benchmark_analysis(bench20)
        assert not np.isnan(d["benchmark_beta"]), "恰好 20 个共同交易日却判为样本不足"

    def test_zero_variance_benchmark_gives_zero_beta_not_division(self):
        """
        `beta = cov01 / var_bench if var_bench > 1e-12 else 0.0`。
        放宽成 `>=` 后，方差为 0 的基准（常数序列）会走进除零。
        """
        flat_bench = pd.Series(0.0, index=IDX)
        d = PerformanceAnalyzer(_result(RET)).benchmark_analysis(flat_bench)
        assert d["benchmark_beta"] == 0.0
        assert np.isfinite(d["benchmark_alpha"])

    def test_zero_tracking_error_gives_nan_information_ratio(self):
        """
        `ir = alpha / te if te > 1e-9 else nan`。跟踪误差为 0 时 IR 无定义，
        放宽成 `>=` 依然会除以一个极小数得到荒谬的 IR。
        构造：策略与基准都是常数 → beta=0、主动收益恒定 → te=0。
        """
        const = pd.Series(0.001, index=IDX)
        d = PerformanceAnalyzer(_result(const)).benchmark_analysis(const)
        assert d["tracking_error"] == pytest.approx(0.0, abs=1e-12)
        assert np.isnan(d["information_ratio"])

    def test_max_consecutive_loss_days_counts_the_longest_run(self):
        """
        连亏天数的计数器：`if cur > max_consec` 与 `cur = 0` 的复位。
        构造一段明确的 4 连亏，且它不是最后一段，确保"复位后仍保留历史最大值"。
        """
        vals = ([0.01] * 3 + [-0.01] * 4 + [0.01] * 3 + [-0.01] * 2)
        vals = (vals * 10)[:N]
        d = PerformanceAnalyzer(_result(pd.Series(vals, index=IDX))).stress_test()
        assert d["max_consecutive_loss_days"] == 4

    def test_worst_month_is_the_minimum_compounded_month(self, pa):
        d = pa.stress_test()
        monthly = RET.resample("ME").apply(lambda x: float((1 + x).prod() - 1))
        assert d["worst_month"] == pytest.approx(round(float(monthly.min()), 4), abs=1e-9)
        assert d["worst_month_date"] == str(monthly.idxmin())

    def test_zero_return_day_breaks_a_losing_streak(self):
        """
        `losses = (ret < 0)`：**收益恰好为 0 的一天不是亏损日**，它会打断连亏。
        放宽成 `<=` 时下面这段的最长连亏会从 2 变成 4。区分值是 0.0 本身。
        """
        pattern = [-0.01, -0.01, 0.0, -0.01, 0.01]
        vals = (pattern * 24)[:N]
        d = PerformanceAnalyzer(_result(pd.Series(vals, index=IDX))).stress_test()
        assert d["max_consecutive_loss_days"] == 2, (
            "收益为 0 的一天被算成了亏损日，连亏计数被拉长")

    def test_worst_quarter_and_year_are_reported(self, pa):
        """
        月/季/年三个 `round(x, 4) if not np.isnan(x) else nan` 是**三份独立代码**，
        只测月度不代表季度年度也对（删掉任一处的 `not` 都会让该项恒为 NaN）。
        """
        d = pa.stress_test()
        for key in ("worst_month", "worst_quarter", "worst_year"):
            assert not np.isnan(d[key]), f"{key} 恒为 NaN —— 该项的 not 分支疑似被改反"
        quarterly = RET.resample("QE").apply(lambda x: float((1 + x).prod() - 1))
        yearly = RET.resample("YE").apply(lambda x: float((1 + x).prod() - 1))
        assert d["worst_quarter"] == pytest.approx(round(float(quarterly.min()), 4), abs=1e-9)
        assert d["worst_year"] == pytest.approx(round(float(yearly.min()), 4), abs=1e-9)

    def test_crisis_windows_are_empty_outside_their_range(self, pa):
        """2023 年的回测不覆盖任何已知危机区间 → 该字段必须是空的，不能凭空造值。"""
        assert pa.stress_test()["crisis_period_returns"] == {}

    def test_crisis_window_return_is_compounded(self):
        """
        覆盖到危机区间时必须给出**复利**累计收益 `(1 + r).prod() - 1`。
        两个符号各能改坏一次：`1 + r` → `1 - r` 让亏损变盈利；
        `- 1` → `+ 1` 让结果偏移 2。
        """
        cidx = pd.bdate_range("2020-02-03", periods=60)
        cret = pd.Series(-0.001, index=cidx)
        d = PerformanceAnalyzer(_result(cret)).stress_test()
        crisis = d["crisis_period_returns"]
        assert "COVID_2020" in crisis, f"回测覆盖 2020 年 2-4 月，危机区间却是空的：{crisis}"
        assert crisis["COVID_2020"] < 0, "每日 -0.1% 的区间累计收益不可能为正"
        assert crisis["COVID_2020"] == pytest.approx(-0.0583, abs=1e-4)


# ===========================================================================
# H. summarize —— 汇总口径必须与逐项一致
# ===========================================================================

def test_summarize_matches_individual_metrics(pa):
    """
    汇总是外部（报告/前端/晋级门）真正读的东西。
    它与逐项方法必须逐字段一致，否则"报告里的夏普"和"门看的夏普"会是两个数。
    """
    s = pa.summarize()
    assert s["annualized_return"] == pytest.approx(pa.annualized_return(), abs=1e-15)
    assert s["annualized_vol"] == pytest.approx(pa.annualized_volatility(), abs=1e-15)
    assert s["sharpe_ratio"] == pytest.approx(pa.sharpe_ratio(), abs=1e-15)
    assert s["sharpe_tstat"] == pytest.approx(pa.sharpe_tstat(), abs=1e-15)
    assert s["calmar_ratio"] == pytest.approx(pa.calmar_ratio(), abs=1e-15)
    assert s["max_drawdown"] == pytest.approx(pa.max_drawdown()[0], abs=1e-15)
    assert s["sortino_ratio"] == pytest.approx(pa.sortino_ratio(), abs=1e-15)
    assert s["var_95"] == pytest.approx(pa.var_cvar(0.05)[0], abs=1e-15)
    assert s["ic_method"] == "approx_position"


def test_summarize_records_which_ic_method_was_used(pa):
    """ic_method 是**口径声明**：拿近似 IC 当精确 IC 用会高估信号质量。"""
    sig, px = TestRankIC._panel(n_days=40)
    res = _result(pd.Series(0.001, index=sig.index), signal=sig)
    p = PerformanceAnalyzer(res)
    assert p.summarize(prices=px)["ic_method"] == "exact_price"
    assert p.summarize(ic_series=pd.Series([0.1, 0.2]))["ic_method"] == "provided"


# ===========================================================================
# I. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "app/core/backtest_engine/performance_analyzer.py ×1 — L151 `int(trough_idx - peak_idx)` → `+`（max_drawdown 的非日期索引分支）":
        "该分支不可达：`__init__` 里先取 `self._tdays`，而 `_tdays` 无条件执行 "
        "`(idx[-1] - idx[0]).days`，非日期索引在那一步就抛 AttributeError，"
        "根本走不到 max_drawdown。见 "
        "test_non_datetime_index_is_rejected_before_reaching_the_fallback。",

    "app/core/backtest_engine/performance_analyzer.py ×1 — L495 `if cur > max_consec:` → `>=`（最长连亏计数）":
        "两侧结果恒等：`>` 只在 cur 严格更大时赋值，`>=` 在相等时也赋值，"
        "而相等时赋的是同一个值。max_consec 的最终取值与比较符无关。"
        "见 test_consecutive_loss_counter_is_insensitive_to_the_comparison。",

    "app/core/backtest_engine/performance_analyzer.py ×1 — L428 `beta = ... if var_bench > 1e-12 else 0.0` → `>=`":
        "唯一的区分点是 var_bench **恰好等于** 1e-12。我一度以为常数基准"
        "（var=0）能区分 —— 判错了：`0.0 >= 1e-12` 同样为假，两侧都走 else。"
        "var_bench 来自 np.cov 的浮点结果，无法构造成精确的 1e-12。",

    "app/core/backtest_engine/performance_analyzer.py ×1 — L437 `ir = alpha/te if te > 1e-9 else nan` → `>=`":
        "同上：区分点是 te **恰好等于** 1e-9。te = std(active_ret)·√tdays，"
        "是多步浮点运算的结果；实际取到的特殊值是 0.0，而 `0.0 >= 1e-9` 为假，"
        "两侧都返回 NaN。",
}


def test_epsilon_guards_in_benchmark_analysis_are_unreachable():
    """L428 / L437 等价性的机械验证：两种比较符在所有可达取值上判定一致。"""
    for tol in (1e-12, 1e-9):
        for x in (0.0, -0.0, 1e-20, tol / 2, tol * 2, 1.0):
            assert (x > tol) == (x >= tol), f"x={x} tol={tol} 上两种比较符不一致"
    rng = np.random.default_rng(21)
    for _ in range(300):
        r = rng.normal(0, 0.01, 120)
        b = rng.normal(0, 0.01, 120)
        assert float(np.cov(r, b)[1, 1]) != 1e-12
        assert float(np.std(r - b, ddof=1) * np.sqrt(252.0)) != 1e-9


def test_consecutive_loss_counter_is_insensitive_to_the_comparison():
    """L495 等价性的机械验证：把两种比较符都跑一遍，逐例比对结果。"""
    def _longest(losses, inclusive):
        max_consec = cur = 0
        for v in losses:
            if v:
                cur += 1
                if (cur >= max_consec) if inclusive else (cur > max_consec):
                    max_consec = cur
            else:
                cur = 0
        return max_consec

    rng = np.random.default_rng(5)
    cases = [[], [True], [False], [True, True, False, True],
             [False] * 5, [True] * 7]
    cases += [list(rng.random(n) < 0.5) for n in (3, 10, 40)]
    for c in cases:
        assert _longest(c, False) == _longest(c, True), f"在 {c} 上两种比较符结果不同"


def test_every_survivor_has_a_written_proof():
    """存活项要么被用例杀死，要么在此有书面证明；不许有第三种状态。"""
    assert len(PROVEN_EQUIVALENT) == 4
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
