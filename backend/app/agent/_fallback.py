"""
_fallback.py — FallbackOrchestrator: LLM-free workflow orchestration.

Runs Workflow A (hypothesis → alpha) and Workflow B (optimize existing DSL)
using only QuantTools + OverfitCritic.  When overfitting is detected it calls
tool_mutate_ast for structural mutation rather than simple regeneration.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Dict, Tuple

from app.agent._constants import _MAX_CORRECTION_ROUNDS
from app.agent._critic import OverfitCritic
from app.agent._tools import QuantTools

logger = logging.getLogger(__name__)


class FallbackOrchestrator:
    """
    Orchestrator for the no-LLM execution path.

    Workflow A and B both follow the same anti-overfitting loop:
      generate/run → backtest → critique → mutate (up to _MAX_CORRECTION_ROUNDS) → save
    """

    def __init__(self, tools: QuantTools) -> None:
        self._tools  = tools
        self._critic = OverfitCritic()

    # ------------------------------------------------------------------
    # Workflow A — hypothesis → DSL → optimise → validate → save
    # ------------------------------------------------------------------

    def run_workflow_a(self, hypothesis: str) -> Tuple[str, Dict]:
        dsl: str         = ""
        final_metrics    = {}
        for attempt in range(_MAX_CORRECTION_ROUNDS + 1):
            dsl_result    = json.loads(self._tools.tool_generate_alpha_dsl(hypothesis))
            dsl           = dsl_result["dsl"]
            opt_result    = json.loads(self._tools.tool_run_optuna(dsl))
            best_config   = json.dumps(opt_result.get("best_config", {}))
            final_metrics = json.loads(self._tools.tool_run_backtest(dsl, best_config))

            passed, reason = self._critic.check(final_metrics)
            if passed:
                break

            logger.info("Attempt %d 过拟合: %s", attempt + 1, reason)
            if attempt < _MAX_CORRECTION_ROUNDS:
                mut_result = json.loads(self._tools.tool_mutate_ast(dsl, reason))
                dsl        = mut_result.get("mutated_dsl", dsl)
                logger.info(
                    "结构变异 → 新 DSL: %s (%s)",
                    dsl, mut_result.get("mutation_type", ""),
                )
                hypothesis = dsl   # feed mutated DSL as next-round hypothesis

        self._tools.tool_save_alpha(
            name         = f"auto_{int(time.time())}",
            dsl          = dsl,
            metrics_json = json.dumps(final_metrics),
        )
        return dsl, final_metrics

    # ------------------------------------------------------------------
    # Workflow B — existing DSL → optimise → validate → save
    # ------------------------------------------------------------------

    def run_workflow_b(self, user_dsl: str) -> Tuple[str, Dict]:
        dsl: str      = user_dsl
        final_metrics = {}
        for attempt in range(_MAX_CORRECTION_ROUNDS + 1):
            opt_result    = json.loads(self._tools.tool_run_optuna(dsl))
            best_config   = json.dumps(opt_result.get("best_config", {}))
            final_metrics = json.loads(self._tools.tool_run_backtest(dsl, best_config))

            passed, reason = self._critic.check(final_metrics)
            if passed:
                break

            logger.info("Workflow B attempt %d 过拟合: %s", attempt + 1, reason)
            if attempt < _MAX_CORRECTION_ROUNDS:
                mut_result = json.loads(self._tools.tool_mutate_ast(dsl, reason))
                dsl        = mut_result.get("mutated_dsl", dsl)
                logger.info(
                    "结构变异 → 新 DSL: %s (%s)",
                    dsl, mut_result.get("mutation_type", ""),
                )

        self._tools.tool_save_alpha(
            name         = f"optimized_{int(time.time())}",
            dsl          = dsl,
            metrics_json = json.dumps(final_metrics),
        )
        return dsl, final_metrics
