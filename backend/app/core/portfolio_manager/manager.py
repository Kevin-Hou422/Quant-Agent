"""
manager.py — 组合与资金管理层 PortfolioManager（Phase PM 第一批）

把 N 个孤立的 paper 因子，变成**一个真实、美元计价、容量感知**的组合账本——
即"像真人 PM 一样决定：这个账本里每只股票具体投多少钱"。

覆盖（PM 第一批）
    PM.1 多因子合成 + **跨因子净持仓**：多个因子信号 IC-IR 加权合成一个组合信号 → 一个净持仓
         权重向量（A 多 AAPL、B 空 AAPL 自动对冲）。复用 `AlphaCombiner`。
    PM.2 容量：按真实 AUM 给每只股票施加 **ADV 容量上限**（复用 water-filling `project_to_capped_l1`）；
         AUM 越大、容量越binding，容量不足时组合 gross<1（真实——装不下就不硬装）。
    PM.3 资本配置：把容量后的权重 × AUM → **每只股票具体美元/股数**。

不含执行侧（怎么下单/滑点/借券）——那是 Phase 12（Alpaca）。LLM 不参与。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WidePanel = Dict[str, pd.DataFrame]


@dataclass
class PortfolioResult:
    weights:      pd.DataFrame                 # 容量后净持仓权重面板 (T×N)
    combo_weights: Dict[str, float]            # 各因子在合成中的权重
    aum:          float
    prices:       pd.DataFrame = field(repr=False, default=None)  # 用于 book_view
    composite:    pd.DataFrame = field(repr=False, default=None)  # 合成信号（组合级 realized IC 用）

    def gross_series(self) -> pd.Series:
        return self.weights.abs().sum(axis=1)

    def net_series(self) -> pd.Series:
        return self.weights.sum(axis=1)

    def book_on(self, date) -> Dict[str, dict]:
        """某日的具体持仓账本：{ticker: {weight, dollars, shares}}。"""
        w = self.weights.loc[date]
        px = self.prices.loc[date] if self.prices is not None else None
        out: Dict[str, dict] = {}
        for tk in w.index:
            wi = float(w[tk])
            if abs(wi) < 1e-9:
                continue
            dollars = wi * self.aum
            shares = (dollars / float(px[tk])) if (px is not None and float(px[tk]) > 0) else float("nan")
            out[tk] = {"weight": round(wi, 6), "dollars": round(dollars, 2), "shares": shares}
        return out


class PortfolioManager:
    """
    Parameters
    ----------
    aum        : 真实资金量（美元）。容量约束依赖它——这是"孤立归一化因子"和"真实账本"的分水岭。
    method     : 多因子合成方法（`ic_weighted` | `equal_weight` | `min_variance`）。
    clip_z     : 合成信号 → 权重时的 z-score 截断。
    cost_params: CostParams（ADV 窗口/上限等）；None 用默认。
    """

    def __init__(
        self,
        aum: float = 1_000_000.0,
        method: str = "ic_weighted",
        clip_z: float = 3.0,
        cost_params=None,
    ) -> None:
        from app.core.backtest_engine.transaction_cost import CostParams, LiquidityConstraint
        self.aum = float(aum)
        self.method = method
        self.clip_z = clip_z
        self.params = cost_params or CostParams()
        self._liq = LiquidityConstraint(self.params)

    # ------------------------------------------------------------------ PM.1
    def combined_weights(self, factor_signals: Dict[str, pd.DataFrame],
                         prices: pd.DataFrame):
        """合成多因子信号 → 净持仓权重面板 (T×N) + 因子组合权重。"""
        from app.core.backtest_engine.alpha_combiner import AlphaCombiner
        from app.core.backtest_engine.portfolio_constructor import SignalWeightedPortfolio

        if not factor_signals:
            raise ValueError("factor_signals 为空")
        combiner = AlphaCombiner()
        returns = prices.pct_change()
        combo_w = combiner.optimize_weights(factor_signals, returns=returns, method=self.method)
        composite = combiner.combine(factor_signals, weights=combo_w)      # 合成信号（含跨因子净化）
        weights = SignalWeightedPortfolio(clip_z=self.clip_z).construct(composite)  # 净持仓，L1≈1
        return weights, combo_w, composite

    # ------------------------------------------------------------------ PM.2
    def apply_capacity(self, weights: pd.DataFrame, prices: pd.DataFrame,
                       volume: pd.DataFrame) -> pd.DataFrame:
        """按 AUM 施加每名 ADV 容量上限（water-filling）。容量不足时 gross<1。"""
        from app.core.backtest_engine.transaction_cost import project_to_capped_l1

        cols = weights.columns
        px = prices.reindex(index=weights.index, columns=cols).ffill(limit=5)
        vol = volume.reindex(index=weights.index, columns=cols).ffill(limit=5).fillna(0.0)
        adv_usd = self._liq.compute_adv(vol, px)                # (T×N) $ADV
        cap_usd = adv_usd * self.params.adv_cap_pct             # 每名每日 $ 上限
        cap_w = (cap_usd / self.aum).to_numpy(dtype=float)
        cap_w = np.where(np.isfinite(cap_w) & (cap_w > 0.0), cap_w, 0.0)
        capped = project_to_capped_l1(weights.to_numpy(dtype=float), cap_w, target=1.0)
        return pd.DataFrame(capped, index=weights.index, columns=cols)

    # ------------------------------------------------------------------ 编排
    def build_book(self, factor_signals: Dict[str, pd.DataFrame],
                   prices: pd.DataFrame, volume: pd.DataFrame) -> PortfolioResult:
        """PM.1→PM.2→(PM.3 view)：产出一个真实 AUM 下、容量约束后的组合账本。"""
        weights, combo_w, composite = self.combined_weights(factor_signals, prices)
        capped = self.apply_capacity(weights, prices, volume)
        logger.info(
            "[portfolio_manager] AUM=%.0f | 因子=%d | 末日 gross=%.3f net=%.3f",
            self.aum, len(factor_signals),
            float(capped.abs().sum(axis=1).iloc[-1]),
            float(capped.sum(axis=1).iloc[-1]),
        )
        return PortfolioResult(weights=capped, combo_weights=combo_w, aum=self.aum,
                               prices=prices.reindex(columns=capped.columns),
                               composite=composite.reindex(columns=capped.columns))
