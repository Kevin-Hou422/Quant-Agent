"""
PerformanceAnalyzer — 全量绩效指标计算

输入  : BacktestResult
输出  : 各指标数值 / pd.Series / pd.DataFrame

所有收益序列统一为日频。
E3 修复：TRADING_DAYS 不再硬编码为 252，改为从实际数据日期范围动态计算。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .backtest_engine import BacktestResult

# 模块级回退常量（仅用于极端情况 < 2 个交易日）
_FALLBACK_TDAYS = 252.0


# ---------------------------------------------------------------------------
# PerformanceAnalyzer
# ---------------------------------------------------------------------------

class PerformanceAnalyzer:
    """
    对 BacktestResult 进行全量绩效分析。

    Parameters
    ----------
    result    : BacktestResult
    rf        : 无风险日收益率（默认 0）。优先级低于 rf_annual。
    rf_annual : 无风险年化利率（如 0.05 = 5%）。设置后自动转换为日频率。
                推荐使用此参数而非 rf，以避免混淆日/年频率。
    """

    def __init__(
        self,
        result:    BacktestResult,
        rf:        float = 0.0,
        rf_annual: float = 0.0,
    ) -> None:
        self.r    = result
        self._ret = result.net_returns.fillna(0.0)
        # rf_annual 优先：年化率转日频（用动态 tdays 保持一致）
        _tdays = self._tdays
        self.rf        = rf_annual / _tdays if rf_annual != 0.0 else rf
        self.rf_annual = rf_annual if rf_annual != 0.0 else rf * _tdays

    # ------------------------------------------------------------------
    # E3: 动态年化系数
    # ------------------------------------------------------------------

    @property
    def _tdays(self) -> float:
        """
        从实际净值曲线日期范围动态计算年化交易日系数。

        替代硬编码 252，适配不同市场（A 股 ~244、加密货币 365 等）。
        当数据点不足 2 时回退到 252。
        """
        idx = self._ret.index
        if len(idx) < 2:
            return _FALLBACK_TDAYS
        calendar_years = max((idx[-1] - idx[0]).days / 365.25, 1.0 / 365.25)
        return float(len(idx)) / calendar_years

    # ================================================================
    # 收益指标
    # ================================================================

    def annualized_return(self) -> float:
        """(1 + mean_daily_ret)^tdays - 1（使用动态交易日系数）"""
        mean_d = self._ret.mean()
        return float((1 + mean_d) ** self._tdays - 1)

    def annualized_volatility(self) -> float:
        return float(self._ret.std(ddof=1) * np.sqrt(self._tdays))

    def sharpe_ratio(self) -> float:
        vol = self.annualized_volatility()
        return float((self.annualized_return() - self.rf_annual) / vol) if vol > 0 else np.nan

    def sharpe_tstat(self) -> float:
        """
        Sharpe 比率的 t 统计量（Lo 2002 公式）。

        t = SR × √T / √(1 + 0.5 × SR²)

        T = 实际净收益观测天数。
        t > 1.96 → 在 5% 显著性水平下统计显著。
        """
        sr = self.sharpe_ratio()
        if np.isnan(sr):
            return np.nan
        T = float(max(len(self._ret.dropna()), 1))
        return float(sr * np.sqrt(T) / np.sqrt(1.0 + 0.5 * sr ** 2))

    def deflated_sharpe_ratio(
        self,
        n_trials:      int = 1,
        trial_sharpes: Optional[list] = None,
    ) -> float:
        """
        Deflated Sharpe Ratio（Bailey & López de Prado 2014，Task 4.3）。
        委托给模块级 ``deflated_sharpe_from_returns()``，详见其文档。
        """
        return deflated_sharpe_from_returns(
            self._ret,
            n_trials      = n_trials,
            trial_sharpes = trial_sharpes,
            tdays         = self._tdays,
        )

    def calmar_ratio(self) -> float:
        dd_val, *_ = self.max_drawdown()
        if dd_val == 0:
            return np.nan
        return float(self.annualized_return() / abs(dd_val))

    # ================================================================
    # 风险指标
    # ================================================================

    def max_drawdown(self) -> Tuple[float, pd.Timestamp, pd.Timestamp, int]:
        """
        Returns
        -------
        (max_drawdown_value, peak_date, trough_date, duration_days)
        max_drawdown_value < 0
        """
        ec           = self.r.equity_curve
        rolling_peak = ec.cummax()
        drawdown     = (ec - rolling_peak) / rolling_peak

        trough_idx = drawdown.idxmin()
        max_dd     = float(drawdown.min())

        peak_idx = rolling_peak[:trough_idx].idxmax()
        duration = (
            int((trough_idx - peak_idx).days)
            if hasattr(trough_idx - peak_idx, "days")
            else int(trough_idx - peak_idx)
        )

        return max_dd, peak_idx, trough_idx, duration

    def drawdown_series(self) -> pd.Series:
        """每日回撤序列（相对历史最高净值）。"""
        ec = self.r.equity_curve
        return (ec - ec.cummax()) / ec.cummax()

    def sortino_ratio(self) -> float:
        """ann_return / (downside_std × √tdays)"""
        neg = self._ret[self._ret < self.rf]
        if len(neg) < 2:
            return np.nan
        downside_std = float(neg.std(ddof=1))
        ann_ret      = self.annualized_return()
        return float(ann_ret / (downside_std * np.sqrt(self._tdays))) if downside_std > 0 else np.nan

    def var_cvar(self, alpha: float = 0.05) -> Tuple[float, float]:
        """
        历史法 VaR / CVaR（损失取正号方便阅读）。

        Returns
        -------
        (VaR_alpha, CVaR_alpha)  均为正数
        """
        ret  = self._ret.dropna()
        var  = float(-np.percentile(ret, alpha * 100))
        cvar = float(-ret[ret <= -var].mean()) if (ret <= -var).any() else var
        return var, cvar

    def rolling_sharpe(self, window: int = 60) -> pd.Series:
        """滚动夏普比率（使用动态 tdays）。"""
        mean = self._ret.rolling(window, min_periods=max(2, window // 2)).mean()
        std  = self._ret.rolling(window, min_periods=max(2, window // 2)).std(ddof=1)
        return (mean / std.replace(0, np.nan)) * np.sqrt(self._tdays)

    # ================================================================
    # Alpha 质量
    # ================================================================

    def rolling_rank_ic(self) -> pd.Series:
        """
        近似版 Rank IC（无需价格数据）。

        用下期持仓权重代理逐资产收益，会系统性低估真实 IC。
        仅在无价格矩阵时作为降级选项，优先使用 rolling_rank_ic_from_prices()。
        """
        sig     = self.r.signal
        pos_arr = self.r.positions.to_numpy(dtype=float)
        sig_arr = sig.to_numpy(dtype=float)
        n       = len(sig)
        ic_vals = np.full(n, np.nan)

        for t in range(n - 1):
            s     = sig_arr[t]
            fwd_w = pos_arr[t + 1]
            mask  = ~(np.isnan(s) | np.isnan(fwd_w))
            if mask.sum() < 5:
                continue
            rho, _ = scipy_stats.spearmanr(s[mask], fwd_w[mask])
            ic_vals[t] = rho

        return pd.Series(ic_vals, index=sig.index, name="rank_ic_approx").dropna()

    def rolling_rank_ic_from_prices(
        self,
        signal: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.Series:
        """
        精确版 Rank IC：逐日 spearmanr(signal_t, fwd_price_ret_{t+1})。
        需要外部传入价格矩阵。推荐在有价格数据时始终使用此方法。
        """
        fwd_ret = prices.pct_change().shift(-1)
        n       = len(signal)
        ic_vals = np.full(n, np.nan)

        sig_arr = signal.to_numpy(dtype=float)
        fwd_arr = fwd_ret.to_numpy(dtype=float)

        for t in range(n - 1):
            s    = sig_arr[t]
            f    = fwd_arr[t]
            mask = ~(np.isnan(s) | np.isnan(f))
            if mask.sum() < 5:
                continue
            rho, _ = scipy_stats.spearmanr(s[mask], f[mask])
            ic_vals[t] = rho

        return pd.Series(ic_vals, index=signal.index, name="rank_ic").dropna()

    def ic_decay_curve(
        self,
        signal:   pd.DataFrame,
        prices:   pd.DataFrame,
        horizons: Optional[List[int]] = None,
    ) -> pd.Series:
        """
        O1 修复：多时间跨度 IC 衰减曲线。

        对每个 horizon h 计算 spearmanr(signal_t, h日后累计收益_t)，
        返回 mean IC Series，揭示信号的持仓期适配性。

        Parameters
        ----------
        signal   : (T×N) 信号矩阵
        prices   : (T×N) 价格矩阵（用于计算 h 日前瞻收益）
        horizons : 待测时间跨度列表（天数），默认 [1, 5, 10, 20, 60]

        Returns
        -------
        pd.Series  index=horizon, value=mean_IC（可为 NaN 当数据不足）
        """
        if horizons is None:
            horizons = [1, 5, 10, 20, 60]

        sig_arr = signal.to_numpy(dtype=float)
        n       = len(signal)
        results: dict = {}

        for h in horizons:
            fwd_ret = prices.pct_change(h).shift(-h)   # h 日累计收益
            fwd_arr = fwd_ret.to_numpy(dtype=float)
            ic_vals: list = []

            for t in range(n - h):
                s    = sig_arr[t]
                f    = fwd_arr[t]
                mask = ~(np.isnan(s) | np.isnan(f))
                if mask.sum() < 5:
                    continue
                rho, _ = scipy_stats.spearmanr(s[mask], f[mask])
                if not np.isnan(rho):
                    ic_vals.append(float(rho))

            results[h] = float(np.mean(ic_vals)) if ic_vals else np.nan

        return pd.Series(results, name="mean_ic_by_horizon")

    def ic_ir(self, ic_series: pd.Series | None = None) -> float:
        """IC / std(IC)"""
        ic = ic_series if ic_series is not None else self.rolling_rank_ic()
        ic = ic.dropna()
        if len(ic) < 2 or ic.std() == 0:
            return np.nan
        return float(ic.mean() / ic.std(ddof=1))

    def turnover_analysis(self) -> dict:
        """
        Returns
        -------
        dict with keys:
          mean_daily_turnover  : float   平均日换手率
          ann_turnover         : float   年化换手率（使用动态 tdays）
          cost_drag_bps        : float   年化成本拖累（bps）
        """
        to         = self.r.turnover.fillna(0.0)
        mean_daily = float(to.mean())
        ann_to     = mean_daily * self._tdays
        cost_drag  = float(self.r.daily_cost_bps.fillna(0.0).mean() * self._tdays)

        return {
            "mean_daily_turnover": mean_daily,
            "ann_turnover":        ann_to,
            "cost_drag_bps":       cost_drag,
        }

    def decile_analysis(
        self,
        prices:   pd.DataFrame | None = None,
        n_deciles: int = 10,
    ) -> pd.Series:
        """
        将信号按截面分 10 档，统计每档平均下期收益（验证单调性）。

        若不传入 prices，则用持仓权重的绝对大小近似（相对比较）。

        Returns
        -------
        pd.Series  index=1..n_deciles，值=平均下期收益
        """
        sig = self.r.signal.to_numpy(dtype=float)

        if prices is not None:
            fwd = prices.pct_change().shift(-1).to_numpy(dtype=float)
        else:
            pos = self.r.positions.to_numpy(dtype=float)
            fwd = np.diff(pos, axis=0, prepend=pos[:1])

        T, _           = sig.shape
        bucket_returns = {d: [] for d in range(1, n_deciles + 1)}

        for t in range(T - 1):
            s    = sig[t]
            f    = fwd[t]
            mask = ~(np.isnan(s) | np.isnan(f))
            if mask.sum() < n_deciles:
                continue
            s_valid = s[mask]
            f_valid = f[mask]
            cuts    = np.percentile(s_valid, np.linspace(0, 100, n_deciles + 1))
            for d in range(1, n_deciles + 1):
                lo    = cuts[d - 1]
                hi    = cuts[d]
                dmask = (s_valid >= lo) & (s_valid <= hi if d == n_deciles else s_valid < hi)
                if dmask.sum() > 0:
                    bucket_returns[d].append(float(np.mean(f_valid[dmask])))

        means = {d: float(np.mean(v)) if v else np.nan for d, v in bucket_returns.items()}
        return pd.Series(means, name="mean_fwd_return")

    def leg_analysis(self) -> dict:
        """
        O4 修复：计算多头腿和空头腿各自的年化收益与 Sharpe。

        依赖 BacktestResult.long_returns / short_returns（由 BacktestEngine 填充）。
        若这两个字段不存在则返回 NaN。
        """
        def _ann_sharpe(ret_series: pd.Series) -> Tuple[float, float]:
            ret = ret_series.fillna(0.0)
            tdays = self._tdays
            ann_ret = float((1 + ret.mean()) ** tdays - 1)
            vol     = float(ret.std(ddof=1) * np.sqrt(tdays))
            sharpe  = float((ann_ret - self.rf_annual) / vol) if vol > 0 else np.nan
            return ann_ret, sharpe

        long_ret  = getattr(self.r, "long_returns",  None)
        short_ret = getattr(self.r, "short_returns", None)

        long_ann, long_sharpe   = _ann_sharpe(long_ret)  if long_ret  is not None else (np.nan, np.nan)
        short_ann, short_sharpe = _ann_sharpe(short_ret) if short_ret is not None else (np.nan, np.nan)

        return {
            "long_ann_return":  long_ann,
            "long_sharpe":      long_sharpe,
            "short_ann_return": short_ann,
            "short_sharpe":     short_sharpe,
        }

    def benchmark_analysis(self, benchmark_returns: pd.Series) -> dict:
        """
        F8 修复：基准 alpha/beta 分解。

        计算策略净收益与基准收益之间的 OLS beta，分离出真实 alpha
        和相对风险调整收益（Information Ratio）。

        Parameters
        ----------
        benchmark_returns : 与策略同频的基准日收益率（如 SPY 日收益）

        Returns
        -------
        dict with keys:
          benchmark_beta    : 策略对基准的暴露（1.0 = 同步基准）
          benchmark_alpha   : 年化 alpha（超额收益，扣除 beta×基准收益后）
          benchmark_ann_ret : 基准年化收益
          tracking_error    : 主动收益（ret - beta×bench）年化标准差
          information_ratio : alpha / tracking_error
        """
        ret   = self._ret
        bench = benchmark_returns.reindex(ret.index).fillna(0.0)

        common = ret.index.intersection(bench.index)
        if len(common) < 20:
            nan4 = dict.fromkeys(
                ["benchmark_beta", "benchmark_alpha", "benchmark_ann_ret",
                 "tracking_error", "information_ratio"], np.nan
            )
            return nan4

        r = ret.loc[common].to_numpy(dtype=float)
        b = bench.loc[common].to_numpy(dtype=float)

        cov_mat   = np.cov(r, b)
        var_bench = cov_mat[1, 1]
        beta      = float(cov_mat[0, 1] / var_bench) if var_bench > 1e-12 else 0.0

        tdays          = self._tdays
        benchmark_ann  = float((1 + b.mean()) ** tdays - 1)
        alpha_daily    = r.mean() - beta * b.mean()
        alpha_annual   = float((1 + alpha_daily) ** tdays - 1)

        active_ret = r - beta * b
        te         = float(active_ret.std(ddof=1) * np.sqrt(tdays))
        ir         = float(alpha_annual / te) if te > 1e-9 else np.nan

        return {
            "benchmark_beta":    round(beta,         4),
            "benchmark_alpha":   round(alpha_annual, 4),
            "benchmark_ann_ret": round(benchmark_ann, 4),
            "tracking_error":    round(te,           4),
            "information_ratio": round(ir,           4) if not np.isnan(ir) else np.nan,
        }

    def stress_test(self) -> dict:
        """
        O2 修复：压力测试子区间分析。

        无需外部基准数据，仅用策略净收益序列计算：
          - 最差月度收益 / 季度收益 / 年度收益及其发生时间
          - 最大连续亏损天数
          - 已知危机区间内的策略累计收益（若回测期覆盖该区间）

        Returns
        -------
        dict with keys:
          worst_month, worst_month_date, worst_quarter, worst_quarter_date,
          worst_year, worst_year_period, max_consecutive_loss_days,
          crisis_period_returns
        """
        ret = self._ret

        def _safe_resample(freq: str) -> pd.Series:
            try:
                return ret.resample(freq).apply(lambda x: float((1 + x).prod() - 1))
            except Exception:
                return pd.Series(dtype=float)

        # 月度 / 季度 / 年度最差
        monthly   = _safe_resample("ME")
        quarterly = _safe_resample("QE")
        yearly    = _safe_resample("YE")

        def _worst(s: pd.Series):
            if s.empty or s.isna().all():
                return np.nan, None
            idx = s.idxmin()
            return float(s.min()), str(idx) if not pd.isnull(idx) else None

        w_month,  w_month_date   = _worst(monthly)
        w_quarter, w_quarter_date = _worst(quarterly)
        w_year,   w_year_date    = _worst(yearly)

        # 最大连续亏损天数
        losses     = (ret.fillna(0.0) < 0).to_numpy()
        max_consec = 0
        cur        = 0
        for v in losses:
            if v:
                cur += 1
                if cur > max_consec:
                    max_consec = cur
            else:
                cur = 0

        # 已知危机区间（数据覆盖时才计算）
        _CRISIS = {
            "GFC_2008":    ("2008-09-01", "2009-03-31"),
            "EU_Debt_2011": ("2011-07-01", "2011-12-31"),
            "China_2015":  ("2015-06-01", "2015-09-30"),
            "COVID_2020":  ("2020-02-01", "2020-04-30"),
            "Bear_2022":   ("2022-01-01", "2022-10-31"),
        }
        crisis_returns = {}
        for name, (start, end) in _CRISIS.items():
            try:
                period = ret.loc[start:end]
                if len(period) >= 5:
                    crisis_returns[name] = round(float((1 + period).prod() - 1), 4)
            except Exception:
                pass

        return {
            "worst_month":                round(w_month,   4) if not np.isnan(w_month)   else np.nan,
            "worst_month_date":           w_month_date,
            "worst_quarter":              round(w_quarter, 4) if not np.isnan(w_quarter) else np.nan,
            "worst_quarter_date":         w_quarter_date,
            "worst_year":                 round(w_year,    4) if not np.isnan(w_year)    else np.nan,
            "worst_year_date":            w_year_date,
            "max_consecutive_loss_days":  int(max_consec),
            "crisis_period_returns":      crisis_returns,
        }

    # ================================================================
    # 综合指标汇总
    # ================================================================

    def summarize(
        self,
        prices:    pd.DataFrame | None = None,
        ic_series: pd.Series   | None = None,
    ) -> dict:
        """
        一次性计算所有指标，返回 dict。

        若传入 prices，使用精确 Rank IC（spearman(signal, fwd_price_ret)）；
        否则降级为近似 IC（持仓权重代理）。
        ic_method 字段记录实际使用的计算方式。
        """
        dd_val, dd_start, dd_end, dd_dur = self.max_drawdown()
        var95, cvar95 = self.var_cvar(0.05)
        to_dict       = self.turnover_analysis()
        leg           = self.leg_analysis()

        if ic_series is not None:
            ic        = ic_series
            ic_method = "provided"
        elif prices is not None:
            ic        = self.rolling_rank_ic_from_prices(self.r.signal, prices)
            ic_method = "exact_price"
        else:
            ic        = self.rolling_rank_ic()
            ic_method = "approx_position"

        return {
            "annualized_return":   self.annualized_return(),
            "annualized_vol":      self.annualized_volatility(),
            "sharpe_ratio":        self.sharpe_ratio(),
            "sharpe_tstat":        self.sharpe_tstat(),
            "calmar_ratio":        self.calmar_ratio(),
            "max_drawdown":        dd_val,
            "max_dd_start":        dd_start,
            "max_dd_end":          dd_end,
            "max_dd_duration":     dd_dur,
            "sortino_ratio":       self.sortino_ratio(),
            "var_95":              var95,
            "cvar_95":             cvar95,
            "mean_ic":             float(ic.mean()) if len(ic) else np.nan,
            "ic_ir":               self.ic_ir(ic),
            "ann_turnover":        to_dict["ann_turnover"],
            "cost_drag_bps":       to_dict["cost_drag_bps"],
            "ic_method":           ic_method,
            **leg,
        }


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio（Task 4.3，模块级 — 可脱离 BacktestResult 单独调用）
# ---------------------------------------------------------------------------

def deflated_sharpe_from_returns(
    returns:       pd.Series,
    n_trials:      int = 1,
    trial_sharpes: Optional[list] = None,
    tdays:         float = 252.0,
) -> float:
    """
    Deflated Sharpe Ratio（Bailey & López de Prado 2014）。

    返回「真实 Sharpe > 0」的概率 ∈ [0, 1]，对以下两个效应做了校正：
      1. 多重检验：从 n_trials 个候选中选出最优者，期望最大 SR 天然为正
      2. 收益非正态：偏度 / 峰度对 SR 估计量方差的影响

    计算（全部使用日频 SR，不年化）：
      SR0  = √V[SR] × [(1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))]   （期望最大 SR）
      DSR  = Φ( (SR - SR0)·√(T-1) / √(1 - γ₃·SR + (γ₄-1)/4·SR²) )
      其中 γ = 0.5772（Euler-Mascheroni），γ₃ = 偏度，γ₄ = 峰度（非超额）

    Parameters
    ----------
    returns       : 策略日净收益序列
    n_trials      : 候选策略总数（GP: Σ 每代 population_size；Optuna: n_trials）
    trial_sharpes : 各候选的**年化** Sharpe 列表（可选）。提供时用其横截面
                    方差估计 V[SR]；缺省时退化为单一策略 SR 估计量的抽样方差
                    （视各候选统计性质相似的保守近似）
    tdays         : 年化交易日数（trial_sharpes 反年化用）

    Returns
    -------
    float ∈ [0,1]；> 0.95 表示在校正多重检验后 Sharpe 仍显著为正。
    n_trials <= 1 且无 trial_sharpes 时等价于单策略 PSR（SR0=0）。
    观测数 < 20 或收益退化时返回 NaN。
    """
    _st = scipy_stats

    ret = returns.dropna() if returns is not None else pd.Series(dtype=float)
    T   = len(ret)
    if T < 20:
        return np.nan

    std = ret.std(ddof=1)
    if std <= 0 or np.isnan(std):
        return np.nan
    sr_d = float(ret.mean() / std)

    skew = float(_st.skew(ret))
    kurt = float(_st.kurtosis(ret, fisher=False))   # 非超额峰度（正态=3）

    # ---- SR 估计量的抽样方差（含高阶矩修正）----
    denom_adj = 1.0 - skew * sr_d + (kurt - 1.0) / 4.0 * sr_d ** 2
    if denom_adj <= 0:
        denom_adj = 1e-9
    var_sr_hat = denom_adj / max(T - 1, 1)

    # ---- 期望最大 SR（SR0）----
    n = max(int(n_trials), 1)
    if n <= 1 and not trial_sharpes:
        sr0 = 0.0                                     # 单策略 → PSR
    else:
        if trial_sharpes and len(trial_sharpes) >= 2:
            arr    = np.asarray(trial_sharpes, dtype=float) / np.sqrt(max(tdays, 1.0))
            arr    = arr[~np.isnan(arr)]
            var_tr = float(np.var(arr, ddof=1)) if len(arr) >= 2 else var_sr_hat
            n      = max(n, len(arr))
        else:
            var_tr = var_sr_hat
        gamma = 0.5772156649015329
        z1 = _st.norm.ppf(1.0 - 1.0 / n)
        z2 = _st.norm.ppf(1.0 - 1.0 / (n * np.e))
        sr0 = float(np.sqrt(max(var_tr, 0.0)) * ((1 - gamma) * z1 + gamma * z2))

    # ---- DSR ----
    z = (sr_d - sr0) * np.sqrt(max(T - 1, 1)) / np.sqrt(denom_adj)
    return float(_st.norm.cdf(z))
