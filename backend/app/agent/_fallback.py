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
from app.agent._tools import QuantTools, _detect_factor_family

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
          1. _generate_diverse_seeds → ≥10 hypothesis-specific diverse DSLs
          2. tool_run_gp_optimization(seed_dsls_json=...) — whole diverse pool as seeds
          3. OverfitCritic check
          4. If FAIL: up to _MAX_CORRECTION_ROUNDS of tool_mutate_ast (real AST ops)
          5. tool_save_alpha
        """
        from app.core.workflows.alpha_workflows import _generate_diverse_seeds

        # Step 1: Generate ≥10 diverse seed DSLs from hypothesis
        seed_dsls = _generate_diverse_seeds(hypothesis, n_target=12)
        if not seed_dsls:
            dsl_result = json.loads(self._tools.tool_generate_alpha_dsl(hypothesis))
            seed_dsls  = [dsl_result.get("dsl", "rank(ts_delta(log(close), 5))")]

        # Detect factor family from first seed — biases GP mutation weights
        factor_family = _detect_factor_family(seed_dsls[0]) if seed_dsls else ""

        logger.info(
            "Workflow A: %d diverse seed DSLs for hypothesis='%.60s' | family='%s'",
            len(seed_dsls), hypothesis, factor_family or "unknown",
        )

        # Step 2: GP structural search with full diverse population
        gp_result     = json.loads(self._tools.tool_run_gp_optimization(
            seed_dsls_json = json.dumps(seed_dsls),
            factor_family  = factor_family,
        ))
        best_dsl      = gp_result.get("best_dsl", seed_dsls[0])
        final_metrics = gp_result.get("metrics") or {}

        _log_evolution(gp_result)

        # Step 3: OverfitCritic — structured diagnosis
        critic_result = self._critic.check(final_metrics)

        # Step 4: Targeted AST-level correction loop
        for attempt in range(_MAX_CORRECTION_ROUNDS):
            if critic_result.passed:
                break
            logger.info(
                "Workflow A: critic FAIL attempt=%d mode=%s severity=%s target=%s | %s",
                attempt + 1,
                critic_result.failure_mode,
                critic_result.severity,
                critic_result.recommended_mutation,
                critic_result.reason,
            )
            mut_result = json.loads(self._tools.tool_mutate_ast(
                best_dsl,
                critic_result.reason,
                mutation_target = critic_result.recommended_mutation,
            ))
            best_dsl      = mut_result.get("mutated_dsl", best_dsl)
            mutation_type = mut_result.get("mutation_type", "unknown")
            logger.info("Targeted AST mutation (%s) → %s", mutation_type, best_dsl)

            bt_result = json.loads(self._tools.tool_run_backtest(best_dsl))
            final_metrics.update(bt_result)
            critic_result = self._critic.check(final_metrics)

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

        Steps:
          1. Quick-eval → diagnose initial quality for targeted mutation
          2. _expand_for_optimization + _targeted_mutations → full seed population
          3. tool_run_gp_optimization(seed_dsls_json=...) → GP + Optuna fine-tune
          4. OverfitCritic check
          5. If FAIL: up to _MAX_CORRECTION_ROUNDS of tool_mutate_ast
          6. tool_save_alpha
        """
        from app.core.workflows.alpha_workflows import (
            _expand_for_optimization, _targeted_mutations, _quick_metrics,
        )

        logger.info("Workflow B starting with seed: %s", user_dsl[:80])

        # Step 1: Quick diagnosis for targeted mutation strategy
        init_metrics = _quick_metrics(
            user_dsl, self._tools._is_data, self._tools._oos_data,
        )
        logger.info(
            "Workflow B init: is_sharpe=%.4f oos_sharpe=%.4f turnover=%.2f overfit=%.4f",
            init_metrics["is_sharpe"], init_metrics["oos_sharpe"],
            init_metrics["turnover"],  init_metrics["overfitting_score"],
        )

        # Step 2: Expand population (original + mutations + targeted variants)
        seed_dsls = _expand_for_optimization(user_dsl, n_mutations=8)
        targeted  = _targeted_mutations(user_dsl, init_metrics)
        seen_set  = set(seed_dsls)
        for td in targeted:
            if td not in seen_set:
                seed_dsls.append(td)
                seen_set.add(td)

        # Detect factor family from user DSL — biases GP mutation weights
        factor_family = _detect_factor_family(user_dsl)

        logger.info(
            "Workflow B: %d seeds (%d targeted) from DSL='%.60s' | family='%s'",
            len(seed_dsls), len(targeted), user_dsl, factor_family or "unknown",
        )

        # Step 3: GP structural search with expanded population
        gp_result     = json.loads(self._tools.tool_run_gp_optimization(
            seed_dsls_json = json.dumps(seed_dsls),
            factor_family  = factor_family,
        ))
        best_dsl      = gp_result.get("best_dsl", user_dsl)
        final_metrics = gp_result.get("metrics") or {}

        _log_evolution(gp_result)

        # Step 2: OverfitCritic — structured diagnosis
        critic_result = self._critic.check(final_metrics)

        # Step 3: Targeted AST-level correction loop
        for attempt in range(_MAX_CORRECTION_ROUNDS):
            if critic_result.passed:
                break
            logger.info(
                "Workflow B: critic FAIL attempt=%d mode=%s severity=%s target=%s | %s",
                attempt + 1,
                critic_result.failure_mode,
                critic_result.severity,
                critic_result.recommended_mutation,
                critic_result.reason,
            )
            mut_result = json.loads(self._tools.tool_mutate_ast(
                best_dsl,
                critic_result.reason,
                mutation_target = critic_result.recommended_mutation,
            ))
            best_dsl      = mut_result.get("mutated_dsl", best_dsl)
            mutation_type = mut_result.get("mutation_type", "unknown")
            logger.info("Targeted AST mutation (%s) → %s", mutation_type, best_dsl)

            bt_result = json.loads(self._tools.tool_run_backtest(best_dsl))
            final_metrics.update(bt_result)
            critic_result = self._critic.check(final_metrics)

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
