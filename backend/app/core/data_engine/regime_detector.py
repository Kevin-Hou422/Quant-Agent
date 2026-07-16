"""
regime_detector.py — 市场状态识别（Task 4.1）

基于趋势强度 + 已实现波动率的规则式 Regime 分类：

  high_vol : 滚动波动率超过全样本滚动波动率的 vol_quantile 分位
  bull     : 非高波动，且过去 trend_window 日累计收益 > +trend_threshold
  bear     : 非高波动，且过去 trend_window 日累计收益 < -trend_threshold
  sideways : 其余（趋势不明显的震荡市）

设计说明
--------
- 纯规则式（无 HMM/聚类），确定性、可测试、无前视：t 日标签只使用
  t 日及之前的收益数据（rolling 窗口右对齐）。
- `regime_to_alpha_weights()` 按因子家族对 Alpha Pool 候选做倾斜加权：
  牛市偏动量、熊市/震荡偏均值回归、高波动偏低波动/防御因子。
  家族分类复用 FinancialInterpreter；分类失败的候选保持基准权重。

Usage
-----
    det = RegimeDetector().fit(market_returns)          # pd.Series 日收益
    labels  = det.predict()                             # pd.Series[str]
    current = det.current_regime()                      # "bull" | ...
    weights = det.regime_to_alpha_weights(current, pool_top5)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIMES = ("bull", "bear", "sideways", "high_vol")

# 因子家族 → 各 Regime 下的权重倾斜系数
# 家族名与 FinancialInterpreter.FACTOR_FAMILIES 对齐
_FAMILY_TILT: Dict[str, Dict[str, float]] = {
    "bull":     {"momentum": 1.5, "reversion": 0.75, "volatility": 0.75},
    "bear":     {"momentum": 0.5, "reversion": 1.25, "volatility": 1.25},
    "sideways": {"momentum": 0.75, "reversion": 1.5},
    "high_vol": {"momentum": 0.5, "reversion": 1.0, "volatility": 1.5},
}


class RegimeDetector:
    """
    趋势强度 + 波动率分位的市场状态识别器。

    Parameters
    ----------
    trend_window    : 趋势判定窗口（交易日，默认 60）
    vol_window      : 波动率计算窗口（交易日，默认 20）
    trend_threshold : 牛/熊判定的窗口累计收益阈值（默认 0.05 = ±5%）
    vol_quantile    : 高波动判定分位（默认 0.80，即滚动波动率进入
                      全样本前 20% 时标记 high_vol）
    """

    def __init__(
        self,
        trend_window:    int   = 60,
        vol_window:      int   = 20,
        trend_threshold: float = 0.05,
        vol_quantile:    float = 0.80,
    ) -> None:
        if trend_window < 5 or vol_window < 5:
            raise ValueError("trend_window / vol_window 至少为 5 个交易日")
        if not 0.5 <= vol_quantile < 1.0:
            raise ValueError(f"vol_quantile 应在 [0.5, 1.0) 内，当前={vol_quantile}")
        self.trend_window    = trend_window
        self.vol_window      = vol_window
        self.trend_threshold = trend_threshold
        self.vol_quantile    = vol_quantile
        self._labels: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, market_returns: pd.Series, method: str = "trend") -> "RegimeDetector":
        """
        对市场日收益序列做全样本 Regime 标注。

        Parameters
        ----------
        market_returns : 市场日收益（如指数或等权组合的 pct_change）
        method         : 目前仅支持 "trend"（趋势强度规则）
        """
        if method != "trend":
            raise ValueError(f"未知 method '{method}'，目前仅支持 'trend'")

        ret = market_returns.dropna().astype(float)
        if len(ret) < self.trend_window + self.vol_window:
            raise ValueError(
                f"数据不足：需要至少 {self.trend_window + self.vol_window} 个"
                f"交易日，当前={len(ret)}"
            )

        # 滚动窗口累计收益（右对齐，无前视）
        trend = (1.0 + ret).rolling(self.trend_window).apply(np.prod, raw=True) - 1.0
        # 滚动已实现波动率
        vol = ret.rolling(self.vol_window).std(ddof=1)
        vol_cut = float(vol.quantile(self.vol_quantile))

        labels = pd.Series("sideways", index=ret.index, dtype=object)
        labels[trend >  self.trend_threshold] = "bull"
        labels[trend < -self.trend_threshold] = "bear"
        labels[vol > vol_cut] = "high_vol"          # 高波动覆盖趋势标签
        # 预热期（趋势/波动窗口未满）无标签
        warmup = max(self.trend_window, self.vol_window)
        labels.iloc[: warmup - 1] = np.nan

        self._labels = labels
        counts = labels.value_counts()
        logger.info(
            "RegimeDetector fit 完成 | %d 天 | %s",
            len(ret),
            ", ".join(f"{k}={v}" for k, v in counts.items()),
        )
        return self

    def predict(self, dates: Optional[pd.DatetimeIndex] = None) -> pd.Series:
        """
        返回逐日 Regime 标签。``dates`` 为 None 时返回 fit 全样本标签，
        否则前向填充对齐到给定日期（未来日期沿用最后已知状态）。
        """
        self._check_fitted()
        if dates is None:
            return self._labels.copy()
        return self._labels.reindex(dates, method="ffill")

    def current_regime(self) -> str:
        """最近一个有效交易日的 Regime。"""
        self._check_fitted()
        valid = self._labels.dropna()
        if valid.empty:
            return "sideways"
        return str(valid.iloc[-1])

    def regime_counts(self) -> Dict[str, int]:
        """各 Regime 的天数统计（诊断用）。"""
        self._check_fitted()
        return {str(k): int(v) for k, v in self._labels.value_counts().items()}

    def regime_to_alpha_weights(
        self,
        regime:    str,
        pool_top5: List[dict],
    ) -> Dict[str, float]:
        """
        按当前 Regime 对 Alpha Pool 候选做因子家族倾斜加权。

        基准权重 = max(sharpe_oos, 0) + 0.1（避免全零），
        乘以该家族在当前 Regime 下的倾斜系数后归一化（和为 1）。

        Parameters
        ----------
        regime    : REGIMES 之一
        pool_top5 : [{"dsl": str, "sharpe_oos": float, ...}, ...]

        Returns
        -------
        dict[dsl → weight]，权重和为 1；pool 为空时返回 {}
        """
        if regime not in REGIMES:
            raise ValueError(f"未知 regime '{regime}'，应为 {REGIMES} 之一")
        if not pool_top5:
            return {}

        tilts = _FAMILY_TILT.get(regime, {})
        raw: Dict[str, float] = {}
        for entry in pool_top5:
            dsl = entry.get("dsl", "") if isinstance(entry, dict) else getattr(entry, "dsl", "")
            if not dsl:
                continue
            sharpe = float(entry.get("sharpe_oos", 0.0) or 0.0)
            base   = max(sharpe, 0.0) + 0.1
            family = self._classify_family(dsl)
            raw[dsl] = base * tilts.get(family, 1.0)

        total = sum(raw.values())
        if total <= 0:
            n = len(raw)
            return {d: 1.0 / n for d in raw} if n else {}
        return {d: v / total for d, v in raw.items()}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_family(dsl: str) -> str:
        """复用 FinancialInterpreter 的因子家族分类；失败时返回 'unknown'。"""
        try:
            from ..alpha_engine.financial_interpreter import FinancialInterpreter
            return FinancialInterpreter().interpret(dsl).factor_family
        except Exception:
            return "unknown"

    def _check_fitted(self) -> None:
        if self._labels is None:
            raise RuntimeError("RegimeDetector 未 fit()，请先调用 fit(market_returns)")
