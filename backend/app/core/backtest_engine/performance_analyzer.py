"""
PerformanceAnalyzer — 全量绩效指标计算

输入  : BacktestResult
输出  : 各指标数值 / pd.Series / pd.DataFrame

所有收益序列统一为日频。
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .backtest_engine import BacktestResult

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# PerformanceAnalyzer
# ---------------------------------------------------------------------------

class PerformanceAnalyzer:
    """
    对 BacktestResult 进行全量绩效分析。

    Parameters
    ----------
    result : BacktestResult
    rf     : 无风险日收益率（默认 0）
    """

    def __init__(self, result: BacktestResult, rf: float = 0.0) -> None:
        self.r  = result
        self.rf = rf
        self._ret = result.net_returns.fillna(0.0)

    # ================================================================
    # 收益指标
    # ================================================================

    def annualized_return(self) -> float:
        """(1 + mean_daily_ret)^252 - 1"""
        mean_d = self._ret.mean()
        return float((1 + mean_d) ** TRADING_DAYS - 1)

    def annualized_volatility(self) -> float:
        return float(self._ret.std(ddof=1) * np.sqrt(TRADING_DAYS))

    def sharpe_ratio(self) -> float:
        vol = self.annualized_volatility()
        return float((self.annualized_return() - self.rf * TRADING_DAYS) / vol) if vol > 0 else np.nan

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
        ec = self.r.equity_curve
        rolling_peak = ec.cummax()
        drawdown     = (ec - rolling_peak) / rolling_peak

        trough_idx = drawdown.idxmin()
        max_dd     = float(drawdown.min())

        # 找最近一次高峰（trough 之前）
        peak_idx  = rolling_peak[:trough_idx].idxmax()
        duration  = int((trough_idx - peak_idx).days) if hasattr(trough_idx - peak_idx, "days") else int(trough_idx - peak_idx)

        return max_dd, peak_idx, trough_idx, duration

    def drawdown_series(self) -> pd.Series:
        """每日回撤序列（相对历史最高净值）。"""
        ec = self.r.equity_curve
        return (ec - ec.cummax()) / ec.cummax()

    def sortino_ratio(self) -> float:
        """ann_return / (downside_std * √252)"""
        neg = self._ret[self._ret < self.rf]
        if len(neg) < 2:
            return np.nan
        downside_std = float(neg.std(ddof=1))
        ann_ret      = self.annualized_return()
        return float(ann_ret / (downside_std * np.sqrt(TRADING_DAYS))) if downside_std > 0 else np.nan

    def var_cvar(self, alpha: float = 0.05) -> Tuple[float, float]:
        """
        历史法 VaR / CVaR（损失取正号方便阅读）。

        Returns
        -------
        (VaR_alpha, CVaR_alpha)  均为正数
        """
        ret = self._ret.dropna()
        var  = float(-np.percentile(ret, alpha * 100))
        cvar = float(-ret[ret <= -var].mean()) if (ret <= -var).any() else var
        return var, cvar

    def rolling_sharpe(self, window: int = 60) -> pd.Series:
        """滚动夏普比率。"""
        mean = self._ret.rolling(window, min_periods=max(2, window // 2)).mean()
        std  = self._ret.rolling(window, min_periods=max(2, window // 2)).std(ddof=1)
        return (mean / std.replace(0, np.nan)) * np.sqrt(TRADING_DAYS)

    # ================================================================
    # Alpha 质量
    # ================================================================

    def rolling_rank_ic(self, window: int = 20) -> pd.Series:
        """
        逐日计算 Rank IC：
            rank_ic[t] = spearmanr(signal[t], fwd_return[t+1])
        返回 IC 序列（长度 = T-1，最后一天无前瞻收益）。
        """
        sig    = self.r.signal
        prices = self.r.positions   # 用 positions 提取持仓日收益（近似）
        # 用净收益（long-only 视角）近似逐资产收益
        # 更准确：需要逐资产价格 → 此处用信号×组合收益的替代
        # 标准实现：fwd_ret[t] = (price[t+1] - price[t]) / price[t]
        # 由于 BacktestResult 未直接暴露价格，用净收益的截面分布近似
        # —— 正确做法需要用户传入价格；此处提供基于持仓的 IC 近似

        n = len(sig)
        ic_vals = np.full(n, np.nan)

        sig_arr = sig.to_numpy(dtype=float)

        # 用 positions 差分近似逐资产贡献收益（粗糙但无依赖）
        pos_arr = self.r.positions.to_numpy(dtype=float)

        for t in range(n - 1):
            s = sig_arr[t]
            # 下期各资产收益的截面排名近似 = 用权重差分作为信号
            # 若有价格数据可替换为精确计算
            fwd_w = pos_arr[t + 1]
            mask = ~(np.isnan(s) | np.isnan(fwd_w))
            if mask.sum() < 5:
                continue
            rho, _ = scipy_stats.spearmanr(s[mask], fwd_w[mask])
            ic_vals[t] = rho

        ic_series = pd.Series(ic_vals, index=sig.index, name="rank_ic")
        return ic_series.dropna()

    def rolling_rank_ic_from_prices(
        self,
        signal: pd.DataFrame,
        prices: pd.DataFrame,
        window: int = 20,
    ) -> pd.Series:
        """
        精确版 Rank IC：逐日 spearmanr(signal_t, fwd_price_ret_{t+1})。
        需要外部传入价格矩阵。
        """
        fwd_ret = prices.pct_change().shift(-1)   # 前瞻一期收益
        n = len(signal)
        ic_vals = np.full(n, np.nan)

        sig_arr = signal.to_numpy(dtype=float)
        fwd_arr = fwd_ret.to_numpy(dtype=float)

        for t in range(n - 1):
            s = sig_arr[t]
            f = fwd_arr[t]
            mask = ~(np.isnan(s) | np.isnan(f))
            if mask.sum() < 5:
                continue
            rho, _ = scipy_stats.spearmanr(s[mask], f[mask])
            ic_vals[t] = rho

        return pd.Series(ic_vals, index=signal.index, name="rank_ic").dropna()

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
          ann_turnover         : float   年化换手率
          cost_drag_bps        : float   年化成本拖累（bps）
        """
        to = self.r.turnover.fillna(0.0)
        mean_daily = float(to.mean())
        ann_to     = mean_daily * TRADING_DAYS

        cost_drag  = float(self.r.daily_cost_bps.fillna(0.0).mean() * TRADING_DAYS)

        return {
            "mean_daily_turnover": mean_daily,
            "ann_turnover":        ann_to,
            "cost_drag_bps":       cost_drag,
        }

    def decile_analysis(
        self,
        prices: pd.DataFrame | None = None,
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
            # 近似：用下一期持仓权重绝对值（不精确，仅供参考）
            pos  = self.r.positions.to_numpy(dtype=float)
            fwd  = np.diff(pos, axis=0, prepend=pos[:1])

        T, N  = sig.shape
        bucket_returns = {d: [] for d in range(1, n_deciles + 1)}

        for t in range(T - 1):
            s = sig[t]
            f = fwd[t]
            mask = ~(np.isnan(s) | np.isnan(f))
            if mask.sum() < n_deciles:
                continue
            s_valid = s[mask]
            f_valid = f[mask]
            cuts = np.percentile(s_valid, np.linspace(0, 100, n_deciles + 1))
            for d in range(1, n_deciles + 1):
                lo = cuts[d - 1]
                hi = cuts[d]
                dmask = (s_valid >= lo) & (s_valid <= hi if d == n_deciles else s_valid < hi)
                if dmask.sum() > 0:
                    bucket_returns[d].append(float(np.mean(f_valid[dmask])))

        means = {d: float(np.mean(v)) if v else np.nan for d, v in bucket_returns.items()}
        return pd.Series(means, name="mean_fwd_return")

    # ================================================================
    # 综合指标汇总
    # ================================================================

    def summarize(
        self,
        prices: pd.DataFrame | None = None,
        ic_series: pd.Series | None = None,
    ) -> dict:
        """一次性计算所有指标，返回 dict。"""
        dd_val, dd_start, dd_end, dd_dur = self.max_drawdown()
        var95, cvar95 = self.var_cvar(0.05)
        to_dict       = self.turnover_analysis()
        ic = ic_series if ic_series is not None else self.rolling_rank_ic()

        return {
            "annualized_return":   self.annualized_return(),
            "annualized_vol":      self.annualized_volatility(),
            "sharpe_ratio":        self.sharpe_ratio(),
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
        }
