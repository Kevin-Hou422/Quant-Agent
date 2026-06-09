"""
Portfolio Constructor — 信号矩阵 → 权重矩阵

组件：
  DecilePortfolio        — Top-10% Long / Bottom-10% Short，等权
  SignalWeightedPortfolio — 权重 ∝ 截面 z-score，归一化
  NeutralizationLayer    — 市场中性 / 行业中性 + L1 归一化
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class PortfolioConstructor(ABC):
    """信号矩阵 (T×N) → 权重矩阵 (T×N)"""

    @abstractmethod
    def construct(self, signal: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ...

    @staticmethod
    def _safe_divide_row(arr: np.ndarray, denom: np.ndarray) -> np.ndarray:
        """逐行安全除法；分母为零时该行返回 0。"""
        safe = np.where(denom == 0, np.nan, denom)
        return np.where(np.isnan(safe[:, None]), 0.0, arr / safe[:, None])


# ---------------------------------------------------------------------------
# DecilePortfolio
# ---------------------------------------------------------------------------

class DecilePortfolio(PortfolioConstructor):
    """
    每日截面：
      Top ``top_pct`` 分位 → 做多，等权  (+1 / n_long)
      Bottom ``bottom_pct`` 分位 → 做空，等权  (-1 / n_short)
      其余资产权重 = 0

    Parameters
    ----------
    top_pct    : 多头分位上限（默认 0.10 = Top 10%）
    bottom_pct : 空头分位下限（默认 0.10 = Bottom 10%）
    """

    def __init__(self, top_pct: float = 0.10, bottom_pct: float = 0.10) -> None:
        self.top_pct    = top_pct
        self.bottom_pct = bottom_pct

    def construct(self, signal: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sig = signal.to_numpy(dtype=float)          # (T, N)
        T, N = sig.shape
        weights = np.zeros_like(sig)

        for t in range(T):
            row = sig[t]
            valid = ~np.isnan(row)
            if valid.sum() < 2:
                continue
            vals = row[valid]

            lo_cut = np.nanquantile(vals, self.bottom_pct)
            hi_cut = np.nanquantile(vals, 1.0 - self.top_pct)

            long_mask  = valid & (row >= hi_cut)
            short_mask = valid & (row <= lo_cut)

            n_long  = long_mask.sum()
            n_short = short_mask.sum()

            if n_long  > 0: weights[t, long_mask]  =  1.0 / n_long
            if n_short > 0: weights[t, short_mask] = -1.0 / n_short

        return pd.DataFrame(weights, index=signal.index, columns=signal.columns)


# ---------------------------------------------------------------------------
# SignalWeightedPortfolio
# ---------------------------------------------------------------------------

class SignalWeightedPortfolio(PortfolioConstructor):
    """
    权重 ∝ 截面 z-score(信号)，再做 L1 归一化使 |w|.sum()=1。
    NaN 信号视为 0。

    Parameters
    ----------
    clip_z : float  将 z-score 裁剪到 [-clip_z, +clip_z]（默认 3.0）
    """

    def __init__(self, clip_z: float = 3.0) -> None:
        self.clip_z = clip_z

    def construct(self, signal: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sig = signal.to_numpy(dtype=float)
        T, N = sig.shape

        # Zero out rows that are entirely NaN (burn-in period).
        # Without this guard, np.nanmean / np.nanstd emit "All-NaN slice"
        # RuntimeWarnings and return NaN for those rows, propagating NaN
        # through the z-score and ultimately into the PnL series.
        # By setting them to 0.0 upfront, those rows produce 0-weight
        # (no position) cleanly and without any warnings.
        all_nan = np.all(np.isnan(sig), axis=1)   # (T,) bool
        if all_nan.any():
            sig = sig.copy()
            sig[all_nan] = 0.0

        # 截面 z-score（忽略局部 NaN，全零行会得到 mu=0 sigma=nan → z=0）
        mu    = np.nanmean(sig, axis=1, keepdims=True)
        sigma = np.nanstd(sig,  axis=1, keepdims=True, ddof=1)
        sigma = np.where(sigma == 0, np.nan, sigma)

        z = (sig - mu) / sigma
        z = np.nan_to_num(z, nan=0.0)
        z = np.clip(z, -self.clip_z, self.clip_z)

        # L1 归一化（逐行）
        l1 = np.abs(z).sum(axis=1, keepdims=True)
        l1 = np.where(l1 == 0, 1.0, l1)
        w  = z / l1

        return pd.DataFrame(w, index=signal.index, columns=signal.columns)


# ---------------------------------------------------------------------------
# NeutralizationLayer
# ---------------------------------------------------------------------------

class NeutralizationLayer:
    """
    对权重矩阵施加中性化约束，并重新 L1 归一化。

    用法：
        layer = NeutralizationLayer()
        w_neutral = layer.market_neutral(weights)
        w_ind     = layer.industry_neutral(weights, industry_map)
    """

    @staticmethod
    def market_neutral(weights: pd.DataFrame) -> pd.DataFrame:
        """
        每行减去截面均值 → sum(w_t) ≈ 0（市场中性）。
        NaN 保持 NaN，忽略 NaN 计算均值。
        """
        w = weights.to_numpy(dtype=float)
        row_mean = np.nanmean(w, axis=1, keepdims=True)
        w = w - row_mean
        w = NeutralizationLayer._l1_normalize(w)
        return pd.DataFrame(w, index=weights.index, columns=weights.columns)

    @staticmethod
    def industry_neutral(
        weights: pd.DataFrame,
        industry_map: Dict[str, str],
    ) -> pd.DataFrame:
        """
        每行内按行业分组，各组减去组均值 → 每行业 sum(w)≈0。
        未映射的 ticker 单独归为 '__other__' 组。

        Parameters
        ----------
        industry_map : {ticker: industry_label}
        """
        w   = weights.to_numpy(dtype=float).copy()
        cols = list(weights.columns)

        # 构建行业分组索引
        groups: Dict[str, list[int]] = {}
        for i, col in enumerate(cols):
            grp = industry_map.get(col, "__other__")
            groups.setdefault(grp, []).append(i)

        T = w.shape[0]
        for t in range(T):
            for idxs in groups.values():
                row_slice = w[t, idxs]
                valid = ~np.isnan(row_slice)
                if valid.sum() == 0:
                    continue
                grp_mean = np.nanmean(row_slice)
                w[t, idxs] = row_slice - grp_mean

        w = NeutralizationLayer._l1_normalize(w)
        return pd.DataFrame(w, index=weights.index, columns=weights.columns)

    @staticmethod
    def beta_neutral(
        weights:        pd.DataFrame,
        asset_returns:  pd.DataFrame,
        market_returns: pd.Series,
        window:         int = 60,
    ) -> pd.DataFrame:
        """
        Beta-neutralize the weight matrix (Task 3.3).

        Adjusts weights so that the portfolio's net market-beta exposure ≈ 0.
        Uses rolling OLS Beta estimates (window trading days).

        Algorithm:
          1. Estimate beta_i(t) = Cov(r_i, r_m) / Var(r_m) over [t-window, t]
          2. Portfolio beta = sum_i w_i(t) * beta_i(t)
          3. Adjust by hedging out the market: w_adj = w - port_beta * (1/N)
          4. Re-normalize to L1 = 1

        Parameters
        ----------
        weights        : (T×N) target weight matrix
        asset_returns  : (T×N) asset daily log returns
        market_returns : (T,)  market index daily returns
        window         : rolling window for beta estimation

        Returns
        -------
        pd.DataFrame (T×N) beta-adjusted weights
        """
        dates   = weights.index
        tickers = list(weights.columns)
        T, N    = len(dates), len(tickers)

        ret_arr = asset_returns.reindex(index=dates, columns=tickers).to_numpy(dtype=float)
        mkt_arr = market_returns.reindex(dates).to_numpy(dtype=float)
        w       = weights.to_numpy(dtype=float).copy()

        # Rolling Beta: (T, N)
        beta = np.full((T, N), 1.0)   # default beta = 1 (fully correlated with market)
        for t in range(window, T):
            r_m   = mkt_arr[t - window : t]
            mask  = ~np.isnan(r_m)
            if mask.sum() < 10:
                continue
            var_m = np.nanvar(r_m)
            if var_m < 1e-12:
                continue
            for j in range(N):
                r_i = ret_arr[t - window : t, j]
                both = mask & ~np.isnan(r_i)
                if both.sum() < 10:
                    continue
                beta[t, j] = float(np.cov(r_i[both], r_m[both])[0, 1] / var_m)

        # Portfolio beta per day
        port_beta = np.nansum(w * beta, axis=1, keepdims=True)   # (T, 1)

        # Subtract beta exposure: w_adj = w - port_beta * (1/N) for each asset
        w_adj = w - port_beta / N

        w_adj = NeutralizationLayer._l1_normalize(w_adj)
        return pd.DataFrame(w_adj, index=dates, columns=tickers)

    @staticmethod
    def _l1_normalize(w: np.ndarray) -> np.ndarray:
        """逐行 L1 归一化；全零行保持全零。"""
        l1 = np.nansum(np.abs(w), axis=1, keepdims=True)
        l1 = np.where(l1 == 0, 1.0, l1)
        return w / l1
