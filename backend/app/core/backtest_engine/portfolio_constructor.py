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

        # 截面 z-score（忽略 NaN）
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
    def _l1_normalize(w: np.ndarray) -> np.ndarray:
        """逐行 L1 归一化；全零行保持全零。"""
        l1 = np.nansum(np.abs(w), axis=1, keepdims=True)
        l1 = np.where(l1 == 0, 1.0, l1)
        return w / l1
