"""
alpha_workflows.py — Production-grade alpha generation and optimization pipelines.

WORKFLOW A — GenerationWorkflow
    Input  : natural language hypothesis
    Steps  : LLM → ≥10 diverse DSLs → AlphaPool init → GP (5-10 gen) → Optuna tune
    Output : best DSL + IS/OOS metrics + evolution log + explanation

WORKFLOW B — OptimizationWorkflow
    Input  : existing DSL
    Steps  : parse → expand (original + mutations + targeted + random) →
             GP (same core) → Optuna tune
    Output : best DSL + IS/OOS metrics + evolution log

Both workflows delegate ALL structural evolution to PopulationEvolver (shared GP core).
Optuna is called ONLY for parameter fine-tuning AFTER the GP selects the best structure.

STRICT RULES (enforced):
    - No string-based mutation  (all ops work on typed AST nodes)
    - Optuna does NOT drive structure search  (structure fixed before Optuna)
    - Population evolves across generations  (tracked in evolution_log)
    - OOS never touches Optuna objective  (AlphaOptimizer uses IS only)
    - All DSLs validated before entering population  (AlphaValidator)
    - Duplicate / high-correlation alphas rejected  (AlphaPool)
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.alpha_engine.parser import Parser, ParseError
from app.core.alpha_engine.validator import AlphaValidator, ValidationError
from app.core.gp_engine.population_evolver import PopulationEvolver, GPEvolutionResult
from app.core.gp_engine.mutations import (
    hoist_mutation, param_mutation, point_mutation,
)
from app.core.gp_engine.gp_engine import _SEED_DSLS, generate_random_alpha

logger = logging.getLogger(__name__)

_parser    = Parser()
_validator = AlphaValidator()


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class WorkflowResult:
    """Unified result for Workflow A and Workflow B."""
    workflow:        str
    best_dsl:        str
    metrics:         Dict[str, Any]
    evolution_log:   List[Dict]      = field(default_factory=list)
    pool_top5:       List[Dict]      = field(default_factory=list)
    best_config:     Optional[Dict]  = None
    seed_dsls:       List[str]       = field(default_factory=list)
    generations_run: int             = 0
    explanation:     str             = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow":        self.workflow,
            "best_dsl":        self.best_dsl,
            "metrics":         self.metrics,
            "evolution_log":   self.evolution_log,
            "pool_top5":       self.pool_top5,
            "best_config":     self.best_config,
            "seed_dsls":       self.seed_dsls,
            "generations_run": self.generations_run,
            "explanation":     self.explanation,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_valid(dsl: str) -> Optional[Any]:
    """Parse + validate a DSL string; return typed Node or None on failure."""
    try:
        node = _parser.parse(dsl)
        _validator.validate(node)
        return node
    except Exception:
        return None


def _partition(
    dataset:   Dict[str, Any],
    oos_ratio: float,
) -> Tuple[Dict, Dict]:
    """Split dataset into IS / OOS using DataPartitioner.  Returns (is_data, oos_data)."""
    from app.core.data_engine.data_partitioner import DataPartitioner

    if oos_ratio <= 0:
        return dataset, {}

    dates = next(iter(dataset.values())).index
    dp    = DataPartitioner(
        start     = str(dates[0].date()),
        end       = str(dates[-1].date()),
        oos_ratio = oos_ratio,
    )
    part = dp.partition(dataset)
    return part.train(), part.test()


def _quick_metrics(
    dsl:      str,
    is_data:  Dict,
    oos_data: Optional[Dict],
) -> Dict[str, Any]:
    """
    Fast IS+OOS backtest with default config.
    Used to diagnose initial alpha quality for targeted mutation decisions.
    """
    from app.core.backtest_engine.realistic_backtester import RealisticBacktester
    from app.core.alpha_engine.signal_processor import SimulationConfig
    from app.core.gp_engine.fitness import compute_fitness

    default_cfg = SimulationConfig(
        delay            = 1,
        decay_window     = 0,
        truncation_min_q = 0.05,
        truncation_max_q = 0.95,
        portfolio_mode   = "long_short",
    )

    def _f(v: Any) -> float:
        try:
            fv = float(v)
            return fv if not np.isnan(fv) else 0.0
        except Exception:
            return 0.0

    try:
        bt     = RealisticBacktester(config=default_cfg)
        result = bt.run(dsl, is_data, oos_dataset=oos_data or None)
        is_r   = result.is_report
        oos_r  = result.oos_report

        s_is  = _f(is_r.sharpe_ratio)
        s_oos = _f(oos_r.sharpe_ratio) if oos_r else 0.0
        turn  = _f(is_r.ann_turnover)
        fit   = compute_fitness(s_is, s_oos, turn)
        overfit = (
            float(np.clip((s_is - s_oos) / abs(s_is), 0.0, 1.0))
            if abs(s_is) > 1e-9 else 0.0
        )
        return {
            "is_sharpe":         s_is,
            "oos_sharpe":        s_oos,
            "turnover":          turn,
            "fitness":           fit,
            "overfitting_score": overfit,
        }
    except Exception as exc:
        logger.debug("_quick_metrics failed for '%s': %s", dsl[:60], exc)
        return {
            "is_sharpe": 0.0, "oos_sharpe": 0.0,
            "turnover": 0.0, "fitness": 0.0, "overfitting_score": 0.0,
        }


# ---------------------------------------------------------------------------
# Workflow A helpers — diverse seed generation
# ---------------------------------------------------------------------------

# Keyword → fallback DSL templates (used when no LLM key is set)
_HYPOTHESIS_TEMPLATES: Dict[str, List[str]] = {
    "momentum":   [
        "rank(ts_delta(close, 5))",
        "zscore(ts_mean(returns, 10))",
        "rank(ts_delta(log(close), 20))",
        "rank(ts_mean(returns, 5))",
        "zscore(ts_delta(close, 10))",
    ],
    "reversion":  [
        "rank(ts_mean(close, 20)) * -1",
        "zscore(ts_delta(close, 5)) * -1",
        "rank(ts_delta(returns, 3)) * -1",
        "zscore(ts_mean(returns, 15)) * -1",
        "rank(ts_std(close, 10)) * -1",
    ],
    "volume":     [
        "rank(ts_mean(volume, 20))",
        "zscore(ts_delta(log(volume), 5))",
        "rank(ts_corr(close, volume, 10))",
        "rank(ts_std(volume, 20))",
        "zscore(ts_mean(volume, 10))",
    ],
    "volatility": [
        "rank(ts_std(returns, 20))",
        "zscore(ts_std(close, 10))",
        "rank(ts_max(high, 20) - ts_min(low, 20))",
        "rank(ts_std(returns, 5))",
        "zscore(ts_std(returns, 10))",
    ],
    "default":    [
        "rank(ts_mean(close, 20))",
        "zscore(ts_delta(close, 5))",
        "rank(ts_std(close, 10))",
        "rank(ts_corr(close, volume, 10))",
        "zscore(ts_mean(returns, 20))",
    ],
}


def _hypothesis_templates(hypothesis: str) -> List[str]:
    """Select template DSLs based on keywords in the hypothesis."""
    h = hypothesis.lower()
    for kw, dsls in _HYPOTHESIS_TEMPLATES.items():
        if kw in h:
            return dsls
    return _HYPOTHESIS_TEMPLATES["default"]


def _generate_diverse_seeds(hypothesis: str, n_target: int = 12) -> List[str]:
    """
    Generate ≥n_target diverse DSL seeds for Workflow A.

    Strategy (layered fallback):
      1. LLM via AlphaAgent._generate_dsls() — hypothesis-specific, up to 5
      2. Keyword-matched templates from _HYPOTHESIS_TEMPLATES
      3. AST mutations of valid seeds (point / hoist / param)
      4. Random alphas via generate_random_alpha() + _SEED_DSLS
    """
    valid_nodes: List[Any] = []
    valid_dsls:  List[str] = []
    seen:        set       = set()

    def _try_add(dsl: str) -> bool:
        node = _parse_valid(dsl)
        if node is None:
            return False
        key = repr(node)
        if key in seen:
            return False
        valid_nodes.append(node)
        valid_dsls.append(key)
        seen.add(key)
        return True

    # ── Layer 1: LLM generation ───────────────────────────────────────
    try:
        from app.agent.alpha_agent import AlphaAgent
        agent    = AlphaAgent()
        llm_dsls = agent._generate_dsls(hypothesis)
        for d in llm_dsls:
            _try_add(d)
        logger.info("[GenWorkflow] LLM produced %d valid seeds", len(valid_dsls))
    except Exception as exc:
        logger.warning("[GenWorkflow] LLM DSL generation failed: %s", exc)

    # ── Layer 2: Keyword templates ────────────────────────────────────
    for d in _hypothesis_templates(hypothesis):
        _try_add(d)

    # ── Layer 3: AST mutations of existing seeds ──────────────────────
    mutation_ops = [point_mutation, hoist_mutation, param_mutation]
    attempts = 0
    while len(valid_dsls) < n_target and attempts < n_target * 12 and valid_nodes:
        attempts += 1
        parent = random.choice(valid_nodes)
        op     = random.choice(mutation_ops)
        try:
            mutant = op(parent)
            _validator.validate(mutant)
            key = repr(mutant)
            if key not in seen:
                valid_nodes.append(mutant)
                valid_dsls.append(key)
                seen.add(key)
        except Exception:
            pass

    # ── Layer 4: Random alphas ────────────────────────────────────────
    attempts = 0
    while len(valid_dsls) < n_target and attempts < n_target * 20:
        attempts += 1
        try:
            if random.random() < 0.4 and _SEED_DSLS:
                node = _parse_valid(random.choice(_SEED_DSLS))
            else:
                node = generate_random_alpha()
            if node is not None:
                key = repr(node)
                if key not in seen:
                    valid_dsls.append(key)
                    seen.add(key)
        except Exception:
            pass

    logger.info(
        "[GenWorkflow] Final seed pool: %d DSLs for hypothesis='%.60s'",
        len(valid_dsls), hypothesis,
    )
    return valid_dsls


# ---------------------------------------------------------------------------
# Workflow B helpers — population expansion + targeted mutation
# ---------------------------------------------------------------------------

def _expand_for_optimization(dsl: str, n_mutations: int = 8) -> List[str]:
    """
    Workflow B Step 2: original DSL → population candidates.
    Returns: [canonical, ...mutations, ...random_fill]
    """
    seed_node = _parse_valid(dsl)
    if seed_node is None:
        raise ValueError(f"Cannot parse DSL for optimization: {dsl}")

    canonical = repr(seed_node)
    results:  List[str] = [canonical]
    seen:     set       = {canonical}

    mutation_ops = [point_mutation, hoist_mutation, param_mutation]
    attempts = 0
    while len(results) - 1 < n_mutations and attempts < n_mutations * 15:
        attempts += 1
        op = random.choice(mutation_ops)
        try:
            mutant = op(seed_node)
            _validator.validate(mutant)
            key = repr(mutant)
            if key not in seen:
                results.append(key)
                seen.add(key)
        except Exception:
            pass

    # Random fill
    attempts = 0
    while len(results) < n_mutations + 3 and attempts < 60:
        attempts += 1
        try:
            node = generate_random_alpha()
            key  = repr(node)
            if key not in seen:
                results.append(key)
                seen.add(key)
        except Exception:
            pass

    logger.info("[OptWorkflow] Expanded '%s' → %d candidates", dsl[:60], len(results))
    return results


def _targeted_mutations(dsl: str, metrics: Dict[str, Any]) -> List[str]:
    """
    Workflow B Step 4: inject targeted structural variants based on metric diagnosis.

    High turnover (>3.0)  → ts_decay_linear wrappers  (smooth signal)
    Low OOS Sharpe (<0.3) → signal combination / rank wrapping  (boost predictivity)
    Unstable (overfit>0.5) → ts_mean smoothing  (reduce noise)
    """
    extra: List[str] = []
    seen:  set       = set()

    turnover = float(metrics.get("turnover", 0.0) or 0.0)
    oos_s    = float(metrics.get("oos_sharpe", 0.0) or 0.0)
    overfit  = float(metrics.get("overfitting_score", 0.0) or 0.0)

    def _try(candidate: str) -> None:
        node = _parse_valid(candidate)
        if node:
            key = repr(node)
            if key not in seen:
                extra.append(key)
                seen.add(key)

    if turnover > 3.0:
        for w in [3, 5, 8]:
            _try(f"ts_decay_linear({dsl}, {w})")

    if oos_s < 0.3:
        _try(f"rank({dsl})")
        _try(f"scale({dsl})")
        _try(f"({dsl} + zscore(ts_delta(close, 5))) / 2")
        _try(f"({dsl} + rank(ts_mean(volume, 10))) / 2")

    if overfit > 0.5:
        for w in [3, 5, 10]:
            _try(f"ts_mean({dsl}, {w})")

    logger.info(
        "[OptWorkflow] Targeted mutations: turnover=%.2f oos_sharpe=%.2f overfit=%.2f → %d variants",
        turnover, oos_s, overfit, len(extra),
    )
    return extra


# ---------------------------------------------------------------------------
# Workflow A: GenerationWorkflow
# ---------------------------------------------------------------------------

class GenerationWorkflow:
    """
    Workflow A — hypothesis → GP-optimized alpha.

    Full pipeline:
      1. Generate ≥n_seed_dsls diverse DSL seeds (LLM → templates → mutations → random)
      2. Initialize PopulationEvolver with ALL seeds as the starting population
      3. GP evolution for n_generations (fitness = OOS Sharpe - overfitting penalty)
      4. Optuna fine-tunes execution params of the GP-selected best structure
      5. Auto-save best alpha + return WorkflowResult with full evolution log
    """

    def __init__(
        self,
        pop_size:        int   = 20,
        n_generations:   int   = 7,
        n_optuna_trials: int   = 10,
        n_seed_dsls:     int   = 12,
        oos_ratio:       float = 0.30,
        seed:            int   = 42,
    ) -> None:
        self._pop_size    = pop_size
        self._n_gen       = n_generations
        self._n_optuna    = n_optuna_trials
        self._n_seeds     = n_seed_dsls
        self._oos_ratio   = oos_ratio
        self._seed        = seed

    def run(
        self,
        hypothesis:  str,
        dataset:     Dict[str, Any],
        on_progress: Optional[Any] = None,
    ) -> WorkflowResult:
        logger.info("[Workflow A] START  hypothesis='%.80s'", hypothesis)

        def _emit(text: str) -> None:
            if on_progress is not None:
                try:
                    on_progress(text)
                except Exception:
                    pass

        _emit(f'[Workflow A] Hypothesis: "{hypothesis[:80]}"')
        _emit(f"[Workflow A] Generating {self._n_seeds} diverse seed DSLs...")

        # Step 1: IS/OOS partition
        is_data, oos_data = _partition(dataset, self._oos_ratio)

        # Step 2: Generate ≥10 diverse seed DSLs
        seed_dsls = _generate_diverse_seeds(hypothesis, n_target=self._n_seeds)
        if not seed_dsls:
            seed_dsls = [repr(generate_random_alpha())]

        effective_pop = max(self._pop_size, len(seed_dsls) + 4)
        _emit(f"[Workflow A] {len(seed_dsls)} valid seed DSLs generated")
        _emit(
            f"[GP] Launching evolution: pop={effective_pop} | "
            f"gen={self._n_gen} | optuna_trials={self._n_optuna}"
        )
        logger.info(
            "[Workflow A] Seeding GP with %d diverse DSLs | pop_size=%d | gen=%d",
            len(seed_dsls), effective_pop, self._n_gen,
        )

        # Step 3: GP evolution — seed_dsls become the entire initial population
        def _on_gen_end(gen_log: dict) -> None:
            _emit(
                f"[GP] Gen {gen_log['generation']}/{self._n_gen} | "
                f"pop={gen_log['population_size']} | "
                f"fitness={gen_log['best_fitness']:.4f} | "
                f"oos_sharpe={gen_log['best_oos_sharpe']:.4f}\n"
                f"     → {gen_log['best_dsl'][:70]}"
            )

        evolver = PopulationEvolver(
            is_data        = is_data,
            oos_data       = oos_data,
            pop_size       = effective_pop,
            n_generations  = self._n_gen,
            seed           = self._seed,
        )
        gp_result: GPEvolutionResult = evolver.run(
            seed_dsls         = seed_dsls,
            n_optuna_trials   = self._n_optuna,
            on_generation_end = _on_gen_end,
        )

        m = gp_result.metrics
        oos_s = m.get("oos_sharpe")
        _emit(
            f"[Optuna] Fine-tuning complete | "
            f"config: {gp_result.best_config or 'default'}"
        )
        _emit(
            f"[Result] IS Sharpe={m.get('is_sharpe', 0):.4f} | "
            f"OOS Sharpe={f'{oos_s:.4f}' if oos_s is not None else 'N/A'}"
        )
        _emit(
            f"[Result] "
            + ("\u26a0 Overfitting detected" if m.get("is_overfit") else "\u2713 Anti-overfitting check passed")
        )
        _emit(f"[GP] Best DSL: {gp_result.best_dsl}")

        explanation = self._explain(hypothesis, gp_result, seed_dsls)
        logger.info("[Workflow A] DONE  best_dsl='%.80s'", gp_result.best_dsl)

        return WorkflowResult(
            workflow        = "generation",
            best_dsl        = gp_result.best_dsl,
            metrics         = gp_result.metrics,
            evolution_log   = gp_result.evolution_log,
            pool_top5       = gp_result.pool_top5,
            best_config     = gp_result.best_config,
            seed_dsls       = seed_dsls,
            generations_run = gp_result.generations_run,
            explanation     = explanation,
        )

    @staticmethod
    def _explain(
        hypothesis: str,
        gp:         GPEvolutionResult,
        seeds:      List[str],
    ) -> str:
        m   = gp.metrics
        log = gp.evolution_log
        parts = [
            f"Hypothesis: {hypothesis}",
            f"Seeded GP with {len(seeds)} diverse DSLs.",
            f"Evolved for {gp.generations_run} generation(s).",
        ]
        if len(log) >= 2:
            parts.append(
                f"Best fitness: {log[0].get('best_fitness', '?')} → "
                f"{log[-1].get('best_fitness', '?')}"
            )
        oos_s = m.get("oos_sharpe")
        if oos_s is not None:
            parts.append(f"OOS Sharpe: {oos_s:.4f}")
        overfit = m.get("overfitting_score", 0.0) or 0.0
        parts.append(
            f"Overfitting: {overfit:.4f} "
            f"({'WARNING: overfit' if overfit > 0.5 else 'healthy'})"
        )
        parts.append(f"Best DSL: {gp.best_dsl}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Workflow B: OptimizationWorkflow
# ---------------------------------------------------------------------------

class OptimizationWorkflow:
    """
    Workflow B — existing DSL → GP-optimized alpha.

    Full pipeline:
      1. Parse DSL → validated AST
      2. Quick-evaluate to diagnose initial quality
      3. Expand: original + structural mutations (n_mutations) + targeted variants
      4. GP evolution with expanded population (same core as Workflow A)
      5. Optuna fine-tunes best structure
      6. Return WorkflowResult with evolution log and explanation
    """

    def __init__(
        self,
        pop_size:        int   = 20,
        n_generations:   int   = 7,
        n_optuna_trials: int   = 10,
        n_mutations:     int   = 8,
        oos_ratio:       float = 0.30,
        seed:            int   = 42,
    ) -> None:
        self._pop_size    = pop_size
        self._n_gen       = n_generations
        self._n_optuna    = n_optuna_trials
        self._n_mutations = n_mutations
        self._oos_ratio   = oos_ratio
        self._seed        = seed

    def run(
        self,
        dsl:         str,
        dataset:     Dict[str, Any],
        on_progress: Optional[Any] = None,
    ) -> WorkflowResult:
        logger.info("[Workflow B] START  dsl='%.80s'", dsl)

        def _emit(text: str) -> None:
            if on_progress is not None:
                try:
                    on_progress(text)
                except Exception:
                    pass

        _emit(f"[Workflow B] Input DSL: {dsl[:80]}")
        _emit("[Workflow B] Parsing and diagnosing initial quality...")

        # Step 1: Parse + validate; get canonical form
        node = _parse_valid(dsl)
        if node is None:
            raise ValueError(f"Invalid or unparseable DSL: {dsl}")
        canonical = repr(node)

        # Step 2: IS/OOS partition + quick initial evaluation
        is_data, oos_data = _partition(dataset, self._oos_ratio)
        init_metrics = _quick_metrics(canonical, is_data, oos_data)
        logger.info(
            "[Workflow B] Initial metrics: is_sharpe=%.4f oos_sharpe=%.4f "
            "turnover=%.2f overfit=%.4f",
            init_metrics["is_sharpe"], init_metrics["oos_sharpe"],
            init_metrics["turnover"],  init_metrics["overfitting_score"],
        )
        _emit(
            f"[Diagnose] IS Sharpe={init_metrics['is_sharpe']:.4f} | "
            f"OOS Sharpe={init_metrics['oos_sharpe']:.4f} | "
            f"Turnover={init_metrics['turnover']:.2f} | "
            f"Overfit={init_metrics['overfitting_score']:.4f}"
        )

        # Step 3: Expand population
        _emit("[Workflow B] Expanding population with mutations and targeted variants...")
        seed_dsls = _expand_for_optimization(canonical, n_mutations=self._n_mutations)

        # Step 4: Targeted structural mutations based on metric diagnosis
        targeted = _targeted_mutations(canonical, init_metrics)
        seen_set = set(seed_dsls)
        for td in targeted:
            if td not in seen_set:
                seed_dsls.append(td)
                seen_set.add(td)

        effective_pop = max(self._pop_size, len(seed_dsls) + 4)
        _emit(
            f"[Workflow B] Population: {len(seed_dsls)} seeds "
            f"({len(targeted)} targeted) | pop={effective_pop}"
        )
        _emit(
            f"[GP] Launching evolution: pop={effective_pop} | "
            f"gen={self._n_gen} | optuna_trials={self._n_optuna}"
        )
        logger.info(
            "[Workflow B] Population: %d seeds (%d targeted) | pop_size=%d | gen=%d",
            len(seed_dsls), len(targeted), effective_pop, self._n_gen,
        )

        # Step 5: GP evolution with expanded population
        def _on_gen_end(gen_log: dict) -> None:
            _emit(
                f"[GP] Gen {gen_log['generation']}/{self._n_gen} | "
                f"pop={gen_log['population_size']} | "
                f"fitness={gen_log['best_fitness']:.4f} | "
                f"oos_sharpe={gen_log['best_oos_sharpe']:.4f}\n"
                f"     → {gen_log['best_dsl'][:70]}"
            )

        evolver = PopulationEvolver(
            is_data        = is_data,
            oos_data       = oos_data,
            pop_size       = effective_pop,
            n_generations  = self._n_gen,
            seed           = self._seed,
        )
        gp_result: GPEvolutionResult = evolver.run(
            seed_dsls         = seed_dsls,
            n_optuna_trials   = self._n_optuna,
            on_generation_end = _on_gen_end,
        )

        m = gp_result.metrics
        oos_s = m.get("oos_sharpe")
        init_oos = init_metrics.get("oos_sharpe", 0.0) or 0.0
        _emit(
            f"[Optuna] Fine-tuning complete | "
            f"config: {gp_result.best_config or 'default'}"
        )
        _emit(
            f"[Result] IS Sharpe={m.get('is_sharpe', 0):.4f} | "
            f"OOS Sharpe={f'{oos_s:.4f}' if oos_s is not None else 'N/A'} "
            + (f"({'↑' if (oos_s or 0) > init_oos else '↓'}{abs((oos_s or 0) - init_oos):.4f} vs input)"
               if oos_s is not None else "")
        )
        _emit(
            f"[Result] "
            + ("\u26a0 Overfitting detected" if m.get("is_overfit") else "\u2713 Anti-overfitting check passed")
        )
        _emit(f"[GP] Best DSL: {gp_result.best_dsl}")

        explanation = self._explain(dsl, gp_result, init_metrics, seed_dsls, targeted)
        logger.info("[Workflow B] DONE  best_dsl='%.80s'", gp_result.best_dsl)

        return WorkflowResult(
            workflow        = "optimization",
            best_dsl        = gp_result.best_dsl,
            metrics         = gp_result.metrics,
            evolution_log   = gp_result.evolution_log,
            pool_top5       = gp_result.pool_top5,
            best_config     = gp_result.best_config,
            seed_dsls       = seed_dsls,
            generations_run = gp_result.generations_run,
            explanation     = explanation,
        )

    @staticmethod
    def _explain(
        original_dsl:  str,
        gp:            GPEvolutionResult,
        init_metrics:  Dict[str, Any],
        seed_dsls:     List[str],
        targeted:      List[str],
    ) -> str:
        m   = gp.metrics
        log = gp.evolution_log
        parts = [
            f"Input DSL: {original_dsl}",
            f"Expanded to {len(seed_dsls)} candidates ({len(targeted)} targeted).",
            f"Evolved for {gp.generations_run} generation(s).",
        ]
        if len(log) >= 2:
            parts.append(
                f"Fitness: {log[0].get('best_fitness', '?')} → "
                f"{log[-1].get('best_fitness', '?')}"
            )
        init_oos = init_metrics.get("oos_sharpe", 0.0) or 0.0
        final_oos = m.get("oos_sharpe")
        if final_oos is not None:
            delta = final_oos - init_oos
            parts.append(
                f"OOS Sharpe: {init_oos:.4f} → {final_oos:.4f} "
                f"({'↑' if delta > 0 else '↓'}{abs(delta):.4f})"
            )
        overfit = m.get("overfitting_score", 0.0) or 0.0
        parts.append(
            f"Overfitting: {overfit:.4f} "
            f"({'WARNING' if overfit > 0.5 else 'OK'})"
        )
        changed = gp.best_dsl != repr(_parse_valid(original_dsl) or original_dsl)
        parts.append(f"Structure changed: {'YES → ' + gp.best_dsl if changed else 'NO (original was best)'}")
        return " | ".join(parts)
