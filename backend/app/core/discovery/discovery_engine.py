"""
discovery_engine.py — 自主发现编排器（Phase 9.2）

角色
----
把"市场观察 → 假设 → GP → 候选"串成一条**无需用户提供思路**的自动流水线：
  1. `MarketObserver` 观察市场 → 排序的家族方向（数据驱动，非用户文本）
  2. 对每个 top 家族跑 `GenerationWorkflow`（复用；家族名作为其"假设"驱动种子生成——
     该家族来自观察引擎而非用户，故仍是自主的）
  3. 去重（本轮内按 DSL 精确去重；跨轮/相关性去重见备注）
  4. 赢家一律存为 **CANDIDATE**（经 `AlphaResult` 默认；再由验证门/审批逐级晋级）

**LLM 不参与**：可无 key 运行（GenerationWorkflow 有确定性回退）。供 `nightly_discovery_job`
每晚调用，实现"收盘后自动发现"。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

WidePanel = Dict[str, Any]


@dataclass
class DiscoveredCandidate:
    alpha_id: Optional[int]
    family:   str
    dsl:      str
    metrics:  Dict[str, Any] = field(default_factory=dict)
    validated: bool = False                      # 验证门是否通过（→ VALIDATED）
    gate:      Optional[dict] = None             # ValidationResult.to_dict()

    def to_dict(self) -> dict:
        return {"alpha_id": self.alpha_id, "family": self.family, "dsl": self.dsl,
                "metrics": self.metrics, "validated": self.validated, "gate": self.gate}


@dataclass
class DiscoveryReport:
    regime:      str
    families:    List[str]
    candidates:  List[DiscoveredCandidate] = field(default_factory=list)
    observation: Optional[dict] = None

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    def to_dict(self) -> dict:
        return {
            "regime": self.regime, "families": self.families,
            "n_candidates": self.n_candidates,
            "candidates": [c.to_dict() for c in self.candidates],
            "observation": self.observation,
        }


class DiscoveryEngine:
    """
    Parameters
    ----------
    n_families    : 每轮探索的 top 家族数（默认 2）。
    pop_size / n_generations / n_optuna_trials / oos_ratio / seed : GP 参数（默认偏小，夜间可调大）。
    """

    def __init__(
        self,
        n_families:      int = 2,
        pop_size:        int = 12,
        n_generations:   int = 4,
        n_optuna_trials: int = 5,
        oos_ratio:       float = 0.30,
        seed:            int = 42,
    ) -> None:
        self.n_families = n_families
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.n_optuna_trials = n_optuna_trials
        self.oos_ratio = oos_ratio
        self.seed = seed

    def run(self, dataset: WidePanel, store=None, save: bool = True,
            auto_validate: bool = True) -> DiscoveryReport:
        """
        auto_validate : True（默认）时，对每个新候选自动跑验证门；通过者升 VALIDATED
                        并进入待批准队列——实现"除人工批准外全程无人值守"。
        """
        from app.core.discovery.market_observer import MarketObserver
        from app.core.workflows.alpha_workflows import GenerationWorkflow

        obs = MarketObserver().observe(dataset)
        families = obs.top_families(self.n_families)
        logger.info("[discovery] regime=%s → 探索家族=%s", obs.regime, families)

        if store is None and save:
            from app.db.alpha_store import AlphaStore
            from app.config import settings
            store = AlphaStore(db_url=settings.database_url)

        seen: set[str] = set()
        candidates: List[DiscoveredCandidate] = []
        for fam in families:
            try:
                wf = GenerationWorkflow(
                    pop_size=self.pop_size, n_generations=self.n_generations,
                    n_optuna_trials=self.n_optuna_trials, oos_ratio=self.oos_ratio,
                    seed=self.seed,
                ).run(hypothesis=fam, dataset=dataset)   # 家族名=观察引擎产出，非用户文本
            except Exception as exc:                     # 单家族失败不拖垮整轮
                logger.warning("[discovery] 家族 '%s' GP 失败，跳过: %s", fam, exc)
                continue

            # S.3：把本轮 GP 试过的策略数累加进全局 trial 计数器（喂给验证门的 DSR 去膨胀）
            try:
                from app.db.trial_ledger import TrialLedger
                TrialLedger().add(self.pop_size * self.n_generations + self.n_optuna_trials)
            except Exception as exc:  # 计数失败不阻断发现
                logger.debug("[discovery] trial 计数失败: %s", exc)

            dsl = (wf.best_dsl or "").strip()
            if not dsl or dsl in seen:                   # 本轮去重
                continue
            seen.add(dsl)

            m = wf.metrics or {}
            aid = None
            if save:
                aid = self._save_candidate(store, dsl, fam, obs.regime, m, wf.explanation)

            cand = DiscoveredCandidate(alpha_id=aid, family=fam, dsl=dsl, metrics=m)
            # 自动验证门：通过者升 VALIDATED → 进入待批准队列（无人值守到审批点为止）
            if auto_validate and save and aid is not None:
                cand.validated, cand.gate = self._run_gate(store, aid, dsl, dataset)
            candidates.append(cand)

        n_val = sum(1 for c in candidates if c.validated)
        logger.info("[discovery] 产出 %d 个 CANDIDATE，其中 %d 个通过验证门→VALIDATED（regime=%s）",
                    len(candidates), n_val, obs.regime)
        return DiscoveryReport(
            regime=obs.regime, families=families,
            candidates=candidates, observation=obs.to_dict(),
        )

    def _run_gate(self, store, aid, dsl, dataset) -> "tuple[bool, Optional[dict]]":
        """
        因子入池门：**默认低门槛泄漏过滤**（PM.7 修正错配——严门在策略层 StrategyGate，不加单因子）。
        通过则 CANDIDATE→VALIDATED（= 进池）。配置 `factor_gate_mode="strict"` 可切回旧的因子级严门。
        fail-closed：出错视为不通过。
        """
        try:
            from app.config import settings
            mode = getattr(settings, "factor_gate_mode", "leak")
        except Exception:
            mode = "leak"
        try:
            if mode == "strict":
                from app.core.lifecycle.validation_gate import ValidationGate
                res = ValidationGate().evaluate(dsl, dataset)
                passed, detail = res.passed, res.to_dict()
            else:
                from app.core.lifecycle.leak_filter import leak_filter
                passed, detail = leak_filter(dsl, dataset)
            if passed:
                store.update_status(aid, "validated")
            return passed, detail
        except Exception as exc:
            logger.warning("[discovery] 候选 %s 入池门出错（视为不通过）: %s", aid, exc)
            return False, None

    @staticmethod
    def _save_candidate(store, dsl, family, regime, m, explanation) -> Optional[int]:
        from app.db.alpha_store import AlphaResult
        try:
            return store.save(AlphaResult(
                dsl=dsl,
                hypothesis=f"auto-discovery | family={family} | regime={regime}",
                sharpe=float(m.get("is_sharpe") or 0.0),
                ann_return=float(m.get("is_return") or 0.0),
                ic_ir=float(m.get("is_ic") or 0.0),
                ann_turnover=float(m.get("is_turnover") or 0.0),
                reasoning=explanation or "",
                # status 默认 candidate（Phase 9.3）
            ))
        except Exception as exc:
            logger.warning("[discovery] 保存候选失败: %s", exc)
            return None
