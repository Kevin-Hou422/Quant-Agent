"""
spread.py — 从免费日频数据估计真实买卖价差（Phase TR）

散户最大真实成本是**买卖价差**，而它能只用每日 High/Low **免费**估出来：
  - Corwin-Schultz (2012)：两日 H/L 比值 → 有效价差比例。
  - Abdi-Ranaldi (2017) CHL：Close+High+Low，常更稳。

这替代硬编码的 `spread_bps=2`：**不流动/小盘股自动得到更宽的估计价差**，且随数据时变、每次重算。
注意：这是仿真用的**估计（T2）**；真正交易时应换成盘口实时价差（T3，走 provider）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_CS_DENOM = 3.0 - 2.0 * np.sqrt(2.0)     # ≈ 0.17157，Corwin-Schultz 常数


def corwin_schultz_spread(
    high: pd.DataFrame,
    low: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Corwin-Schultz (2012) 有效价差估计。

    Returns
    -------
    每只股票的价差**比例**（fraction，×1e4 = bps），取最近 `window` 日均值。负估计截 0。
    """
    H = np.asarray(high, dtype=float)
    L = np.asarray(low, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(np.where((H > 0) & (L > 0), H / L, np.nan))          # (T,N) ln(H/L)
        # β_t：相邻两日 ln(H/L)² 之和；γ_t：两日合并 H/L
        beta = hl[:-1] ** 2 + hl[1:] ** 2                                # (T-1,N)
        Hmax = np.maximum(H[:-1], H[1:])
        Lmin = np.minimum(L[:-1], L[1:])
        gamma = np.log(np.where((Hmax > 0) & (Lmin > 0), Hmax / Lmin, np.nan)) ** 2
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_DENOM - np.sqrt(gamma / _CS_DENOM)
        S = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))          # (T-1,N) 价差比例
    S = np.where(np.isfinite(S), S, np.nan)
    S = np.where(S < 0.0, 0.0, S)                                        # 负值截 0（CS 惯例）
    Sdf = pd.DataFrame(S, columns=high.columns)
    est = Sdf.tail(max(1, window)).mean(axis=0, skipna=True)
    return est.fillna(0.0)


def corwin_schultz_spread_bps(high: pd.DataFrame, low: pd.DataFrame, window: int = 20) -> pd.Series:
    """同上，单位 bps。"""
    return corwin_schultz_spread(high, low, window) * 1e4


def abdi_ranaldi_spread(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Abdi-Ranaldi (2017) CHL 有效价差估计（常比 CS 更稳）。

    s² = 4·E[(c_t − η_t)(c_{t+1} − η_t)]，η_t = (h_t + l_t)/2（对数中价）；
    spread = √max(s²,0)（比例）。取最近 window 日均值。
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.log(high.where(high > 0)); l = np.log(low.where(low > 0)); c = np.log(close.where(close > 0))
    eta = (h + l) / 2.0
    x = (c - eta) * (c.shift(-1) - eta)                                  # (T,N)
    s2 = 4.0 * x
    s2 = s2.clip(lower=0.0)
    spread = np.sqrt(s2)
    est = spread.tail(max(1, window)).mean(axis=0, skipna=True)
    return est.fillna(0.0)
