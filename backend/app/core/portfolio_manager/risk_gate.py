"""
risk_gate.py — 组合级风控门（Phase PM.5）

在"合成+容量"之后、"下单"之前，对**组合账本**施加真人 PM 的风控纪律：
    - **敞口上限**：gross（总敞口）/ net（净敞口）不超限；
    - **集中度上限**：单票权重 ≤ 上限；单行业总权重 ≤ 上限（避免押注单一名/单一板块）；
    - **目标波动缩放**：按估计的组合年化波动把整体仓位缩放到目标波动（vol targeting）；
    - **回撤熔断**：组合回撤超阈值 → 触发降/清仓信号（`should_halt`）。

两种用法
    - `apply(weights)`：产出**满足所有硬约束**的调整后权重（真正交易的账本）。
    - `check(weights)`：只返回**违规清单**（不改权重），供 PM.7「审批一份组合配置」时做门。

纪律
    - 只做**减法/缩放**，绝不无中生有加敞口；缩放优先降风险。
    - 当前 long-only（`allow_short=False`）：net=gross，做空相关的 **beta 中性**留待开启做空后
      （见 Phase R.2 / B6），本门先落敞口/集中度/波动/回撤这四项对 long-only 就生效的约束。
LLM 不参与。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    max_gross:         float = 1.0     # 总敞口上限（Σ|w|）
    max_net:           float = 1.0     # 净敞口上限（|Σw|）；long-only 时 = max_gross
    max_name_weight:   float = 0.10    # 单票权重上限（|w_i|）
    max_sector_weight: float = 0.30    # 单行业总权重上限（占 gross 的比例）
    target_vol_ann:    Optional[float] = None   # 目标年化波动（None=不做 vol targeting）
    max_drawdown:      float = 0.20    # 回撤熔断阈值（0.20 = 回撤 20% 触发）
    long_only:         bool  = True

    def __post_init__(self):
        if self.long_only:
            self.max_net = min(self.max_net, self.max_gross)


@dataclass
class RiskReport:
    n_name_clipped:   int = 0          # 被单票上限削过的（名·日）次数
    n_sector_scaled:  int = 0          # 被行业上限缩过的（行业·日）次数
    n_gross_scaled:   int = 0          # 被 gross 上限缩过的日数
    vol_scalar:       float = 1.0      # 目标波动缩放系数（1=未缩放）
    violations:       List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_name_clipped": self.n_name_clipped,
            "n_sector_scaled": self.n_sector_scaled,
            "n_gross_scaled": self.n_gross_scaled,
            "vol_scalar": round(self.vol_scalar, 4),
            "violations": self.violations,
        }


class PortfolioRiskGate:
    def __init__(self, limits: Optional[RiskLimits] = None) -> None:
        self.limits = limits or RiskLimits()

    # ------------------------------------------------------------------
    # 施加（产出合规账本）
    # ------------------------------------------------------------------

    def apply(self, weights: pd.DataFrame, sectors: Optional[pd.Series] = None,
              port_vol_ann: Optional[float] = None) -> Tuple[pd.DataFrame, RiskReport]:
        """
        对权重面板逐日施加风控约束，返回 (调整后权重, RiskReport)。

        sectors      : Series(index=ticker → 行业标签/代码)；None 时自动按 GICS 推导。
        port_vol_ann : 组合当前估计年化波动；配合 limits.target_vol_ann 做 vol targeting。
        """
        lim = self.limits
        cols = list(weights.columns)
        sec = self._resolve_sectors(sectors, cols)
        rep = RiskReport()

        # ---- 目标波动缩放（先算一个全局标量，再逐日 clip 保证硬约束）----
        vol_scalar = 1.0
        if lim.target_vol_ann and port_vol_ann and port_vol_ann > 1e-9:
            vol_scalar = float(lim.target_vol_ann / port_vol_ann)
            vol_scalar = float(np.clip(vol_scalar, 0.0, 3.0))    # 防止极端放大
        rep.vol_scalar = vol_scalar

        out = weights.copy().astype(float) * vol_scalar
        for dt in out.index:
            v = out.loc[dt]
            v_adj, nclip, nsec, gscaled = self._apply_vector(v, sec, lim)
            out.loc[dt] = v_adj
            rep.n_name_clipped += nclip
            rep.n_sector_scaled += nsec
            rep.n_gross_scaled += int(gscaled)
        return out, rep

    # ------------------------------------------------------------------
    # 检查（只报违规，不改）——供 PM.7 审批一份配置
    # ------------------------------------------------------------------

    def check(self, weights: pd.DataFrame, sectors: Optional[pd.Series] = None,
              tol: float = 1e-6) -> List[str]:
        lim = self.limits
        cols = list(weights.columns)
        sec = self._resolve_sectors(sectors, cols)
        viol: List[str] = []

        gross = weights.abs().sum(axis=1)
        net = weights.sum(axis=1)
        if (gross > lim.max_gross + tol).any():
            viol.append(f"gross 超限（max={gross.max():.3f} > {lim.max_gross}）")
        if (net.abs() > lim.max_net + tol).any():
            viol.append(f"net 超限（max|net|={net.abs().max():.3f} > {lim.max_net}）")
        maxname = weights.abs().max().max()
        if maxname > lim.max_name_weight * lim.max_gross + tol:
            viol.append(f"单票超限（max={maxname:.3f} > {lim.max_name_weight * lim.max_gross}）")
        if lim.long_only and (weights < -tol).any().any():
            viol.append("long-only 下存在负权重（做空）")
        # 行业集中度（末日快照，NAV 比例绝对上限）
        if sec is not None and len(weights) > 0:
            last = weights.iloc[-1]
            sec_abs = last.abs().groupby(sec).sum()
            sec_cap = lim.max_sector_weight * lim.max_gross
            if len(sec_abs) and sec_abs.max() > sec_cap + tol:
                viol.append(f"行业集中度超限（{sec_abs.idxmax()}={sec_abs.max():.3f} > {sec_cap}）")
        return viol

    # ------------------------------------------------------------------
    # 回撤熔断
    # ------------------------------------------------------------------

    def should_halt(self, equity: pd.Series) -> Tuple[bool, float]:
        """给定净值序列，返回 (是否触发熔断, 当前回撤)。回撤 = 1 - equity/历史峰值。"""
        e = pd.Series(equity, dtype=float).dropna()
        if len(e) < 2:
            return False, 0.0
        peak = e.cummax()
        cur_dd = float(1.0 - e.iloc[-1] / peak.iloc[-1])
        halt = cur_dd >= self.limits.max_drawdown
        if halt:
            logger.warning("[risk_gate] 回撤熔断触发：当前回撤=%.3f ≥ 阈值 %.3f", cur_dd, self.limits.max_drawdown)
        return halt, cur_dd

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_sectors(sectors: Optional[pd.Series], cols: List[str]) -> Optional[pd.Series]:
        if sectors is not None:
            return sectors.reindex(cols)
        try:
            from app.core.data_engine.sector_mapper import get_sector_code
            return pd.Series({c: get_sector_code(c) for c in cols})
        except Exception:
            return None

    @staticmethod
    def _apply_vector(w: pd.Series, sectors: Optional[pd.Series],
                      lim: RiskLimits) -> Tuple[pd.Series, int, int, bool]:
        idx = w.index
        a = np.array(w.to_numpy(dtype=float), copy=True)   # 可写副本
        # 1. long-only：去负
        if lim.long_only:
            a = np.where(a < 0.0, 0.0, a)
        # 2. 单票上限（NAV 比例：cap = max_name_weight × max_gross 的绝对权重）
        cap = lim.max_name_weight * lim.max_gross
        nclip = int(np.sum(np.abs(a) > cap + 1e-12))
        a = np.clip(a, -cap, cap)
        # 3. 行业集中度（NAV 比例上限，绝对权重 = max_sector_weight × max_gross）。
        #    只把超限行业缩到上限、不再把 gross 拉回——因为多行业同时 ≤ 上限时 gross 可能达不到满仓
        #    （如 2 个行业各 ≤30%），真人 PM 也会宁可欠配也不违反集中度。
        nsec = 0
        if sectors is not None:
            sec_vals = sectors.reindex(idx).to_numpy()
            sec_cap = lim.max_sector_weight * lim.max_gross
            for sec_id in pd.unique(sectors.dropna().values):
                mask = (sec_vals == sec_id)
                sec_gross = float(np.abs(a[mask]).sum())
                if sec_gross > sec_cap + 1e-12 and sec_gross > 0:
                    a[mask] *= (sec_cap / sec_gross)
                    nsec += 1
        # 4. gross 上限（均匀缩放，不破坏单票/行业约束）
        g = float(np.abs(a).sum())
        gscaled = False
        if g > lim.max_gross + 1e-12:
            a *= (lim.max_gross / g)
            gscaled = True
        return pd.Series(a, index=idx), nclip, nsec, gscaled
