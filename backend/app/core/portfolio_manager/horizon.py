"""
horizon.py — horizon 感知的调仓与因子快慢分类（Phase PM.6）

两件事
    ① **无交易带调仓**（合并 TR.1 的 rebalance_band）：每名目标权重相对**已持仓**漂移不足 band 就
       不动 → 减少无谓换手 → 省成本（$10k 散户下成本≈半价差主导，少调仓直接提升净收益）。
       这是 PM.6 里**真正生效**的部分（改交易的权重）。
    ② **因子快慢分类**：按各因子构造出的权重的**年化换手**把因子分 fast/slow，供组合层差异化对待
       与前端展示（报告用）。快因子高换手→成本更敏感、容量占用更贵；慢因子→更适合放大配额。

纪律：只做减法/分类，不新造金融逻辑；band 由 TradingContext（T2，数据推导）给或显式配置，不写死。
LLM 不参与。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


def annualized_turnover(weights: pd.DataFrame, tdays_per_year: float = 252.0) -> float:
    """单边年化换手 = mean_t(0.5·Σ|w_t − w_{t-1}|) × 交易日/年。"""
    if len(weights) < 2:
        return 0.0
    daily = (weights.diff().abs().sum(axis=1) / 2.0).iloc[1:].mean()
    return float(daily) * float(tdays_per_year)


def classify_horizon(turnover_ann: float, fast_threshold: float = 4.0) -> str:
    """年化换手 > 阈值(默认 4×/年 ≈ 季度以内全换) → fast，否则 slow。"""
    return "fast" if turnover_ann > fast_threshold else "slow"


def apply_no_trade_band(weights: pd.DataFrame, band: float) -> pd.DataFrame:
    """
    无交易带：逐日对每名，若**目标相对已持仓**的漂移 < band 则保持已持仓（不调），
    ≥ band 才调到目标。band ≤ 0 时原样返回（关闭）。返回"黏性"权重面板。
    """
    if band <= 0 or weights.empty or len(weights) < 2:
        return weights
    arr = np.array(weights.to_numpy(dtype=float), copy=True)
    held = arr[0].copy()
    for t in range(1, len(arr)):
        target = arr[t]
        move = np.abs(target - held) >= band
        held = np.where(move, target, held)
        arr[t] = held
    return pd.DataFrame(arr, index=weights.index, columns=weights.columns)


@dataclass
class FactorHorizon:
    factor:       str
    turnover_ann: float
    horizon:      str          # fast | slow

    def to_dict(self) -> dict:
        return {"factor": self.factor, "turnover_ann": round(self.turnover_ann, 3),
                "horizon": self.horizon}


def horizon_profile(factor_signals: Dict[str, pd.DataFrame],
                    clip_z: float = 3.0, fast_threshold: float = 4.0,
                    tdays_per_year: float = 252.0) -> List[FactorHorizon]:
    """对每个因子构造权重 → 年化换手 → fast/slow 分类（报告用，不改交易）。"""
    from app.core.backtest_engine.portfolio_constructor import SignalWeightedPortfolio
    out: List[FactorHorizon] = []
    for name, sig in factor_signals.items():
        try:
            w = SignalWeightedPortfolio(clip_z=clip_z).construct(sig)
            to = annualized_turnover(w, tdays_per_year)
            out.append(FactorHorizon(name, to, classify_horizon(to, fast_threshold)))
        except Exception:
            continue
    return out
