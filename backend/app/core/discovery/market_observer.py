"""
market_observer.py — 市场观察引擎（Phase 9.1）

角色
----
把**对市场的观察**（纯价量统计）转成**结构化的"假设方向"**——即"当前市场下，哪些因子家族
最值得挖"。这是让 agent **自主发现**（不再依赖用户给假设文本）的输入端：观察引擎输出一个
按优先级排序的家族列表，交给发现编排器（9.2）去跑 GP。

**不含 LLM、不读用户文本**：全部来自 OHLCV 的确定性统计，可复现、可审计。

观测指标（近窗 vs 历史）
    regime              市场状态（复用 RegimeDetector：bull/bear/high_vol/sideways）
    dispersion          截面收益离散度（越大 → 截面 alpha 机会越多）
    momentum_breadth    趋势一致性（多少比例资产同向趋势）
    short_term_reversal 短期反转强度（近窗滞后-1 自相关，负 = 均值回复）
    vol_level           近期波动相对历史（高 → 波动率因子更相关）

家族评分 → 排序（全部为有效 GP 种子家族）
    momentum / trend_following  —— 趋势市 + 高一致性时更高
    reversion                   —— 震荡/高波动 + 短期反转强时更高
    volatility                  —— 高波动时更高
    liquidity / price_volume_corr —— 离散度高时更高（量价类分散器）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WidePanel = Dict[str, pd.DataFrame]

# 观察引擎产出的家族（均为 gp_engine._SEED_DSLS_BY_FAMILY 中的有效家族）
_FAMILIES = ["momentum", "trend_following", "reversion",
             "volatility", "liquidity", "price_volume_corr"]


@dataclass
class HypothesisDirection:
    family:    str
    score:     float          # [0,1] 优先级（已归一化）
    rationale: str

    def to_dict(self) -> dict:
        return {"family": self.family, "score": round(self.score, 4), "rationale": self.rationale}


@dataclass
class MarketObservation:
    regime:              str
    dispersion:          float
    momentum_breadth:    float
    short_term_reversal: float
    vol_level:           float
    hypotheses:          List[HypothesisDirection] = field(default_factory=list)

    def top_families(self, n: int = 3) -> List[str]:
        return [h.family for h in self.hypotheses[:n]]

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "dispersion": round(self.dispersion, 6),
            "momentum_breadth": round(self.momentum_breadth, 4),
            "short_term_reversal": round(self.short_term_reversal, 4),
            "vol_level": round(self.vol_level, 4),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
        }


class MarketObserver:
    """
    Parameters
    ----------
    lookback : 近窗长度（交易日，默认 90）——离散度/广度/反转的观察窗。
    mom_window : 计算趋势广度的动量窗（默认 20）。
    """

    def __init__(self, lookback: int = 90, mom_window: int = 20) -> None:
        self.lookback = lookback
        self.mom_window = mom_window

    def observe(self, dataset: WidePanel) -> MarketObservation:
        rets = self._returns(dataset)
        regime = self._regime(rets)

        recent = rets.tail(self.lookback)
        # 截面离散度：近窗每日截面 std 的均值，除以全样本基线 → 归一化
        disp_recent = float(recent.std(axis=1).mean())
        disp_hist = float(rets.std(axis=1).mean()) or 1e-9
        dispersion = disp_recent
        dispersion_norm = float(np.clip(disp_recent / disp_hist / 2.0, 0.0, 1.0))

        # 趋势广度：末日 mom_window 累计收益 > 0 的资产比例
        mom = rets.tail(self.mom_window).sum(axis=0)
        breadth = float((mom > 0).mean()) if len(mom) else 0.5
        trend_strength = float(abs(breadth - 0.5) * 2.0)     # [0,1] 方向一致性

        # 短期反转：近窗各资产滞后-1 自相关的均值（负 = 均值回复）
        reversal = self._mean_lag1_autocorr(recent)
        reversal_strength = float(np.clip(-reversal * 3.0, 0.0, 1.0))

        # 波动水平：近期市场波动 / 历史市场波动
        mkt = rets.mean(axis=1)
        vol_recent = float(mkt.tail(self.lookback).std()) or 0.0
        vol_hist = float(mkt.std()) or 1e-9
        vol_level = float(vol_recent / vol_hist)
        vol_norm = float(np.clip(vol_level - 0.5, 0.0, 1.0))

        trending = 1.0 if regime in ("bull", "bear") else 0.0
        choppy = 1.0 if regime in ("sideways", "high_vol") else 0.0
        is_highvol = 1.0 if regime == "high_vol" else 0.0

        raw: Dict[str, float] = {
            "momentum":         0.20 + 0.50 * trend_strength + 0.30 * trending,
            "trend_following":  0.15 + 0.45 * trend_strength + 0.25 * trending,
            "reversion":        0.20 + 0.50 * reversal_strength + 0.30 * choppy,
            "volatility":       0.15 + 0.55 * vol_norm + 0.20 * is_highvol,
            "liquidity":        0.20 + 0.40 * dispersion_norm,
            "price_volume_corr":0.20 + 0.40 * dispersion_norm,
        }
        top = max(raw.values()) or 1.0
        hyps = [
            HypothesisDirection(
                family=f, score=raw[f] / top,
                rationale=self._rationale(f, regime, trend_strength, reversal_strength,
                                          vol_norm, dispersion_norm),
            )
            for f in _FAMILIES
        ]
        hyps.sort(key=lambda h: h.score, reverse=True)

        obs = MarketObservation(
            regime=regime, dispersion=dispersion, momentum_breadth=breadth,
            short_term_reversal=reversal, vol_level=vol_level, hypotheses=hyps,
        )
        logger.info("[market_observer] regime=%s | top families=%s",
                    regime, obs.top_families(3))
        return obs

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _returns(dataset: WidePanel) -> pd.DataFrame:
        r = dataset.get("returns")
        if r is None or r.empty:
            close = dataset["close"]
            r = np.log(close / close.shift(1))
        return r.replace([np.inf, -np.inf], np.nan)

    def _regime(self, rets: pd.DataFrame) -> str:
        try:
            from app.core.data_engine.regime_detector import RegimeDetector
            mkt = rets.mean(axis=1).dropna()
            det = RegimeDetector().fit(mkt, method="trend")
            return det.current_regime()
        except Exception as exc:  # 数据不足等 → 中性
            logger.debug("[market_observer] regime 判定失败，用 sideways: %s", exc)
            return "sideways"

    @staticmethod
    def _mean_lag1_autocorr(recent: pd.DataFrame) -> float:
        vals = []
        for col in recent.columns:
            s = recent[col].dropna()
            if len(s) > 5 and s.std() > 0:
                a = s.autocorr(lag=1)
                if a is not None and np.isfinite(a):
                    vals.append(a)
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def _rationale(family, regime, trend_s, rev_s, vol_norm, disp_norm) -> str:
        if family in ("momentum", "trend_following"):
            return f"regime={regime}，趋势一致性={trend_s:.2f}（趋势市 + 高一致性利好动量/趋势）"
        if family == "reversion":
            return f"regime={regime}，短期反转强度={rev_s:.2f}（震荡/高波动 + 均值回复利好反转）"
        if family == "volatility":
            return f"regime={regime}，相对波动={vol_norm:.2f}（高波动利好波动率因子）"
        return f"截面离散度={disp_norm:.2f}（离散度高利好量价/流动性类分散因子）"
