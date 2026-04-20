"""
_fallback.py — FallbackOrchestrator: LLM-free GP-driven workflow.

Phase 2/4/7 compliance:
  OLD: Optuna → Backtest → Critic → string-mutate → repeat
  NEW: GP structural search (AST mutation + crossover + selection + Optuna fine-tune)

Both Workflow A and B are now GP-first:
  1. Generate seed DSL (hypothesis → DSL or use provided DSL)
  2. tool_run_gp_optimization:
       - Initialize population (seed + GP random variants)
       - Evolve for N generations with full IS+OOS backtest per individual
       - Multi-objective fitness with overfitting penalty
       - Diversity-filtered AlphaPool
       - Optuna fine-tunes the best GP-selected structure
  3. OverfitCritic final check
  4. If still overfitting: tool_mutate_ast (real AST mutation, NOT string templates)
  5. tool_save_alpha

Optuna is NOT the main optimizer here.  It is called ONLY inside GP's
fine-tuning step (_optuna_fine_tune in population_evolver.py).
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
    LLM-free GP-driven workflow orchestrator.

    Replaced Optuna-dominated loops with PopulationEvolver (GP engine).
    tool_mutate_ast is used only for single-step post-GP correction
    when the winner still fails the OverfitCritic.
    """

    def __init__(self, tools: QuantTools) -> None:
        self._tools  = tools
        self._critic = OverfitCritic()

    # ------------------------------------------------------------------
    # Workflow A — hypothesis → GP evolution → validate → save
    # ------------------------------------------------------------------

    def run_workflow_a(self, hypothesis: str) -> Tuple[str, Dict]:
        """
        Full GP-driven alpha generation from a natural language hypothesis.

        Steps:
          1. tool_generate_alpha_dsl  → seed DSL
          2. tool_run_gp_optimization → GP evolves structure + Optuna fine-tunes winner
          3. OverfitCritic check
          4. If FAIL: up to _MAX_CORRECTION_ROUNDS of tool_mutate_ast (real AST ops)
          5. tool_save_alpha
        """
        # Step 1: seed DSL from hypothesis
        dsl_result = json.loads(self._tools.tool_generate_alpha_dsl(hypothesis))
        seed_dsl   = dsl_result.get("dsl", "rank(ts_delta(log(close), 5))")
        logger.info("Workflow A seed DSL: %s", seed_dsl)

        # Step 2: GP structural search (PRIMARY optimizer)
        gp_result     = json.loads(self._tools.tool_run_gp_optimization(seed_dsl=seed_dsl))
        best_dsl      = gp_result.get("best_dsl", seed_dsl)
        final_metrics = gp_result.get("metrics") or {}

        _log_evolution(gp_result)

        # Step 3: OverfitCritic check
        passed, reason = self._critic.check(final_metrics)

        # Step 4: AST-level correction if still overfitting
        for attempt in range(_MAX_CORRECTION_ROUNDS):
            if passed:
                break
            logger.info(
                "Workflow A: GP winner still overfitting (attempt %d): %s",
                attempt + 1, reason,
            )
            mut_result    = json.loads(self._tools.tool_mutate_ast(best_dsl, reason))
            best_dsl      = mut_result.get("mutated_dsl", best_dsl)
            mutation_type = mut_result.get("mutation_type", "unknown")
            logger.info("AST mutation (%s) → new DSL: %s", mutation_type, best_dsl)

            # Re-validate mutated DSL
            bt_result = json.loads(self._tools.tool_run_backtest(best_dsl))
            final_metrics.update(bt_result)
            passed, reason = self._critic.check(final_metrics)

        # Step 5: Save
        self._tools.tool_save_alpha(
            name         = f"gp_auto_{int(time.time())}",
            dsl          = best_dsl,
            metrics_json = json.dumps(final_metrics),
        )
        return best_dsl, final_metrics

    # ------------------------------------------------------------------
    # Workflow B — existing DSL → GP evolution → validate → save
    # ------------------------------------------------------------------

    def run_workflow_b(self, user_dsl: str) -> Tuple[str, Dict]:
        """
        GP-driven optimization of a user-provided DSL.

        The user's DSL seeds the GP population; GP evolves the structure
        and explores AST variants via mutation + crossover.

        Steps:
          1. tool_run_gp_optimization(user_dsl) → GP evolves + Optuna fine-tunes
          2. OverfitCritic check
          3. If FAIL: up to _MAX_CORRECTION_ROUNDS of tool_mutate_ast
          4. tool_save_alpha
        """
        logger.info("Workflow B starting with seed: %s", user_dsl[:80])

        # Step 1: GP structural search seeded with user's DSL
        gp_result     = json.loads(self._tools.tool_run_gp_optimization(seed_dsl=user_dsl))
        best_dsl      = gp_result.get("best_dsl", user_dsl)
        final_metrics = gp_result.get("metrics") or {}

        _log_evolution(gp_result)

        # Step 2: OverfitCritic check
        passed, reason = self._critic.check(final_metrics)

        # Step 3: AST-level correction
        for attempt in range(_MAX_CORRECTION_ROUNDS):
            if passed:
                break
            logger.info(
                "Workflow B: GP winner still overfitting (attempt %d): %s",
                attempt + 1, reason,
            )
            mut_result    = json.loads(self._tools.tool_mutate_ast(best_dsl, reason))
            best_dsl      = mut_result.get("mutated_dsl", best_dsl)
            mutation_type = mut_result.get("mutation_type", "unknown")
            logger.info("AST mutation (%s) → new DSL: %s", mutation_type, best_dsl)

            bt_result = json.loads(self._tools.tool_run_backtest(best_dsl))
            final_metrics.update(bt_result)
            passed, reason = self._critic.check(final_metrics)

        # Step 4: Save
        self._tools.tool_save_alpha(
            name         = f"gp_optimized_{int(time.time())}",
            dsl          = best_dsl,
            metrics_json = json.dumps(final_metrics),
        )
        return best_dsl, final_metrics


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_evolution(gp_result: Dict) -> None:
    """Log GP evolution summary at INFO level."""
    gens = gp_result.get("evolution_log") or []
    n    = gp_result.get("generations_run", 0)
    best = gp_result.get("best_dsl", "")
    m    = gp_result.get("metrics") or {}

    logger.info(
        "GP complete: %d generations | best_dsl=%s | oos_sharpe=%.4f | overfit=%.4f",
        n, best[:60],
        float(m.get("oos_sharpe") or 0.0),
        float(m.get("overfitting_score") or 0.0),
    )
    for g in gens:
        logger.info(
            "  Gen %d/%d | pop=%d | best_fitness=%.4f | best_oos=%.4f | %s",
            g.get("generation", 0), n,
            g.get("population_size", 0),
            g.get("best_fitness", 0.0),
            g.get("best_oos_sharpe", 0.0),
            g.get("best_dsl", "")[:60],
        )

    pool = gp_result.get("pool_top5") or []
    if pool:
        logger.info("AlphaPool top-5:")
        for i, e in enumerate(pool):
            logger.info(
                "  #%d fitness=%.4f oos=%.4f overfit=%.2f | %s",
                i + 1,
                e.get("fitness", 0.0),
                e.get("sharpe_oos", 0.0),
                e.get("overfitting_score", 0.0),
                e.get("dsl", "")[:60],
            )
