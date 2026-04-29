"""
population_evolver.py — PopulationEvolver: TRUE GP-driven Alpha Optimization Engine.

This is a REAL structural search system. No string manipulation. No fake mutation.
All evolution operates on typed AST nodes from mutations.py.

Pipeline (per the spec):
─────────────────────────────────────────────────────────────────────────────
Initialize population P of size N
    - seed_dsl parsed to AST node
    - remaining slots filled with random nodes from _SEED_DSLS

FOR generation in 1..G:

    1. Evaluate ALL alphas:
         - RealisticBacktester IS+OOS backtest for each individual
         - compute: sharpe_is, sharpe_oos, turnover, overfitting_score

    2. Compute fitness:
         fitness = sharpe_oos
                 - 0.2 * turnover
                 - 0.3 * abs(max_drawdown)
                 - 0.5 * max(0, sharpe_is - sharpe_oos)

    3. Selection:
         - keep top K (elitism, K = elite_ratio × pop_size)
         - diversity filter via AlphaPool (signal correlation < 0.9)

    4. Generate new population (11 operators, adaptive weights):
         - crossover         — structural combination via subtree swap
         - point             — operator swap within same type class
         - hoist             — tree simplification (remove a level)
         - param             — TS window parameter ±20 %
         - wrap_rank         — add rank/zscore layer
         - add_ts_smoothing  — add TS smoothing layer
         - add_condition     — add momentum/trend condition
         - add_volume_filter — add volume gate
         - combine_signals   — arithmetic combination of two signals
         - replace_subtree   — swap subtree with generated one
         - add_operator      — wrap with unary/arithmetic layer
         Adaptive weights via fitness.mutation_weights_from_metrics()

    5. Repeat

    6. Logging per generation:
         Gen N/G | pop=K | best_oos_sharpe=X.XXXX | best_dsl=<expr>

After all generations:
    - Optuna fine-tunes the best structure's execution parameters
      (delay, decay, truncation) — ONLY parameter search, structure fixed.
    - Final IS+OOS backtest with tuned config.
    - Return GPEvolutionResult

─────────────────────────────────────────────────────────────────────────────
Architecture note:
    AlphaPool (alpha_pool.py) stores all evaluated alphas with deduplication
    and correlation filtering, acting as a cross-generation memory.
    The pool is NOT the population — the population evolves each generation
    while the pool accumulates the best unique discoveries.
"""
from __future__ import annotations

import copy
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.gp_engine.mutations import (
    hoist_mutation,
    param_mutation,
    point_mutation,
    subtree_crossover,
    wrap_rank,
    add_ts_smoothing,
    add_condition,
    add_volume_filter,
    combine_signals,
    replace_subtree,
    add_operator,
)
from app.core.gp_engine.fitness import compute_fitness, mutation_weights_from_metrics
from app.core.gp_engine.alpha_pool import AlphaPool, PoolEntry
from app.core.gp_engine.gp_engine import _SEED_DSLS, generate_random_alpha
from app.core.alpha_engine.parser import Parser
from app.core.alpha_engine.validator import AlphaValidator
from app.core.alpha_engine.typed_nodes import Node

logger = logging.getLogger(__name__)

_parser    = Parser()
_validator = AlphaValidator()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Evaluation outcome for one DSL candidate."""
    dsl:               str
    fitness:           float
    sharpe_is:         float
    sharpe_oos:        float
    turnover:          float
    max_drawdown:      float
    overfitting_score: float
    node:              Optional[Node] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "dsl":               self.dsl,
            "fitness":           round(self.fitness,           4),
            "sharpe_is":         round(self.sharpe_is,         4),
            "sharpe_oos":        round(self.sharpe_oos,        4),
            "turnover":          round(self.turnover,          4),
            "max_drawdown":      round(self.max_drawdown,      4),
            "overfitting_score": round(self.overfitting_score, 4),
        }


@dataclass
class GPEvolutionResult:
    """Full result returned by PopulationEvolver.run()."""
    best_dsl:        str
    metrics:         Dict[str, Any]
    generations_run: int
    pool_top5:       List[Dict]
    evolution_log:   List[Dict]       = field(default_factory=list)
    best_config:     Optional[Dict]   = field(default=None)


# ---------------------------------------------------------------------------
# PopulationEvolver
# ---------------------------------------------------------------------------

class PopulationEvolver:
    """
    True GP-driven alpha evolution engine.

    Parameters
    ----------
    is_data        : In-sample dataset (dict of pd.DataFrame)
    oos_data       : Out-of-sample dataset (dict of pd.DataFrame)
    pop_size       : Individuals per generation
    n_generations  : Number of evolution generations
    elite_ratio    : Fraction of pop kept as elite parents
    corr_threshold : Signal correlation threshold for diversity filter
    seed           : Random seed
    """

    def __init__(
        self,
        is_data:        Dict[str, pd.DataFrame],
        oos_data:       Dict[str, pd.DataFrame],
        pop_size:       int   = 12,
        n_generations:  int   = 4,
        elite_ratio:    float = 0.25,
        corr_threshold: float = 0.90,
        seed:           int   = 42,
    ) -> None:
        self._is_data        = is_data
        self._oos_data       = oos_data
        self._pop_size       = pop_size
        self._n_gen          = n_generations
        self._elite_ratio    = elite_ratio
        self._corr_threshold = corr_threshold
        self._seed           = seed
        random.seed(seed)
        np.random.seed(seed)

        self._pool = AlphaPool(max_size=200, corr_threshold=corr_threshold)

        # Default SimulationConfig for GP evaluation (no parameter tuning per-individual)
        from app.core.alpha_engine.signal_processor import SimulationConfig
        self._default_cfg = SimulationConfig(
            delay            = 1,
            decay_window     = 0,
            truncation_min_q = 0.05,
            truncation_max_q = 0.95,
            portfolio_mode   = "long_short",
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        seed_dsl:           Optional[str]       = None,
        seed_dsls:          Optional[List[str]] = None,
        n_optuna_trials:    int                 = 8,
        on_generation_end:  Optional[Any]       = None,
    ) -> GPEvolutionResult:
        """
        Run the full GP optimization pipeline.

        Steps:
          1. Initialize population
          2. Evolve for n_generations
          3. Optuna fine-tunes the best structural winner
          4. Return GPEvolutionResult

        Parameters
        ----------
        seed_dsl        : Single starting DSL (backward compat)
        seed_dsls       : List of starting DSLs; if provided, all are used as
                          initial population seeds (Workflow A / B multi-seed init)
        n_optuna_trials : Optuna trials for parameter fine-tuning (0 = skip)
        """
        evolution_log: List[Dict] = []

        # ── Step 1: Initialize population ─────────────────────────────
        population = self._init_population(seed_dsl, seed_dsls)
        n_seeds = len(seed_dsls or []) + (1 if seed_dsl else 0)
        logger.info(
            "GP start | pop=%d | gen=%d | n_seeds=%d | seed0='%s'",
            len(population), self._n_gen, n_seeds,
            (seed_dsl or (seed_dsls[0] if seed_dsls else ""))[:60],
        )

        # ── Step 2: Evolution loop ─────────────────────────────────────
        for gen in range(self._n_gen):
            # a. Evaluate all
            results = self._evaluate_population(population)
            if not results:
                logger.warning("Gen %d: all evaluations failed", gen + 1)
                break

            # b. Update pool (diversity-filtered)
            for r in results:
                sig_vec = self._compute_signal_vec(r.dsl)
                self._pool.add(PoolEntry(
                    dsl               = r.dsl,
                    fitness           = r.fitness,
                    sharpe_is         = r.sharpe_is,
                    sharpe_oos        = r.sharpe_oos,
                    turnover          = r.turnover,
                    overfitting_score = r.overfitting_score,
                    generation        = gen + 1,
                    signal_vec        = sig_vec,
                ))

            # c. Log this generation
            best_r = max(results, key=lambda x: x.fitness)
            gen_log = {
                "generation":       gen + 1,
                "population_size":  len(results),
                "best_fitness":     round(best_r.fitness,    4),
                "best_oos_sharpe":  round(best_r.sharpe_oos, 4),
                "best_dsl":         best_r.dsl,
                "mean_fitness":     round(float(np.mean([r.fitness for r in results])), 4),
            }
            evolution_log.append(gen_log)
            logger.info(
                "Gen %d/%d | pop=%d | best_fitness=%.4f | best_oos=%.4f | dsl=%s",
                gen + 1, self._n_gen,
                len(results),
                best_r.fitness,
                best_r.sharpe_oos,
                best_r.dsl[:80],
            )
            if on_generation_end is not None:
                try:
                    on_generation_end(gen_log)
                except Exception:
                    pass

            # d. Phase 8: adaptive mutation weights from population diagnostics
            diag    = self._pool.population_diagnostics()
            weights = mutation_weights_from_metrics(
                sharpe_oos    = diag["mean_sharpe_oos"],
                turnover      = diag["mean_turnover"],
                overfit_score = diag["mean_overfit"],
            )
            logger.debug("Gen %d mutation weights: %s", gen + 1, weights)

            # e. Generate next generation (skip on last gen)
            if gen < self._n_gen - 1:
                population = self._generate_next_population(
                    population, results, weights
                )

        # ── Step 3: Optuna fine-tunes best structure ───────────────────
        pool_best = self._pool.best()
        if pool_best is None:
            # Fallback: use seed or default DSL
            best_dsl = seed_dsl or "rank(ts_delta(log(close), 5))"
            metrics  = self._make_fallback_metrics(best_dsl)
            best_cfg = {}
        else:
            best_dsl = pool_best.dsl
            best_cfg, metrics = self._optuna_fine_tune(best_dsl, n_optuna_trials)

        pool_top5 = [e.to_dict() for e in self._pool.top_k(5)]

        return GPEvolutionResult(
            best_dsl        = best_dsl,
            metrics         = metrics,
            generations_run = len(evolution_log),
            pool_top5       = pool_top5,
            evolution_log   = evolution_log,
            best_config     = best_cfg,
        )

    # ------------------------------------------------------------------
    # Population initialisation
    # ------------------------------------------------------------------

    def _init_population(
        self,
        seed_dsl:  Optional[str],
        seed_dsls: Optional[List[str]] = None,
    ) -> List[Node]:
        """
        Build the initial population from seed DSL(s) + random fill.

        If seed_dsls is provided (Workflow A / B multi-seed mode), all valid
        DSLs in the list are parsed and inserted first.  seed_dsl (single) is
        also included for backward compatibility.  Remaining pop_size slots are
        filled with mutations of existing seeds and random alphas.
        """
        pop:  List[Node] = []
        seen: set        = set()

        # 1. Collect all seeds (deduplicated, preserving order)
        all_seeds: List[str] = []
        if seed_dsl:
            all_seeds.append(seed_dsl)
        for d in (seed_dsls or []):
            if d not in all_seeds:
                all_seeds.append(d)

        for dsl in all_seeds:
            try:
                node = _parser.parse(dsl)
                _validator.validate(node)
                key = repr(node)
                if key not in seen:
                    pop.append(node)
                    seen.add(key)
            except Exception as exc:
                logger.warning("Failed to parse seed DSL '%s': %s", dsl[:60], exc)

        # 2. Fill remainder from crossover of seeds + mutations + random
        attempts = 0
        while len(pop) < self._pop_size and attempts < self._pop_size * 20:
            attempts += 1
            try:
                roll = random.random()
                if len(pop) >= 2 and roll < 0.30:
                    # Crossover between two seeds for structured diversity
                    p1, p2 = random.sample(pop[:min(len(pop), 8)], 2)
                    c1, c2 = subtree_crossover(p1, p2)
                    cand   = random.choice([c1, c2])
                elif pop and roll < 0.65:
                    # Mutate an existing individual for more directed exploration
                    parent = random.choice(pop)
                    cand   = random.choice([
                        point_mutation(parent),
                        hoist_mutation(parent),
                        param_mutation(parent),
                    ])
                else:
                    cand = generate_random_alpha()

                _validator.validate(cand)
                key = repr(cand)
                if key not in seen:
                    pop.append(cand)
                    seen.add(key)
            except Exception:
                pass

        return pop

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_population(self, population: List[Node]) -> List[EvalResult]:
        results = []
        for node in population:
            dsl = repr(node)
            r   = self._evaluate_one(dsl, node)
            if r is not None:
                results.append(r)
        return results

    def _evaluate_one(self, dsl: str, node: Optional[Node] = None) -> Optional[EvalResult]:
        """
        Full IS+OOS backtest for one DSL using the default SimulationConfig.
        Returns None on any failure.
        """
        from app.core.backtest_engine.realistic_backtester import RealisticBacktester

        try:
            bt     = RealisticBacktester(config=self._default_cfg)
            result = bt.run(dsl, self._is_data, oos_dataset=self._oos_data)
            is_r   = result.is_report
            oos_r  = result.oos_report

            def _f(v: Any) -> float:
                try:
                    fv = float(v)
                    return fv if not np.isnan(fv) else 0.0
                except (TypeError, ValueError):
                    return 0.0

            sharpe_is    = _f(is_r.sharpe_ratio)
            sharpe_oos   = _f(oos_r.sharpe_ratio) if oos_r else 0.0
            turnover     = _f(is_r.ann_turnover)
            max_drawdown = _f(oos_r.max_drawdown) if oos_r else 0.0

            if abs(sharpe_is) > 1e-9 and oos_r:
                deg = (sharpe_is - sharpe_oos) / abs(sharpe_is)
                overfit_score = float(np.clip(deg, 0.0, 1.0))
            else:
                overfit_score = 0.0

            fitness = compute_fitness(sharpe_is, sharpe_oos, turnover, max_drawdown)

            return EvalResult(
                dsl               = dsl,
                fitness           = fitness,
                sharpe_is         = sharpe_is,
                sharpe_oos        = sharpe_oos,
                turnover          = turnover,
                max_drawdown      = max_drawdown,
                overfitting_score = overfit_score,
                node              = node,
            )

        except Exception as exc:
            logger.debug("Eval failed '%s': %s", dsl[:60], exc)
            return None

    def _compute_signal_vec(self, dsl: str) -> Optional[np.ndarray]:
        """
        Compute 1-D signal fingerprint for AlphaPool correlation filtering.
        Returns the cross-sectional mean of the rank signal across IS dates.
        """
        from app.core.alpha_engine.dsl_executor import Executor
        try:
            sig = Executor().run_expr(dsl, self._is_data)   # pd.DataFrame (T, N)
            return np.nanmean(sig.to_numpy(dtype=float), axis=1)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Selection & next generation
    # ------------------------------------------------------------------

    def _generate_next_population(
        self,
        population:  List[Node],
        results:     List[EvalResult],
        mut_weights: Dict[str, float],
    ) -> List[Node]:
        """
        Tournament selection + adaptive GP operators → next generation.

        Elite fraction is carried forward unchanged (elitism).
        Remaining slots filled by crossover / mutation / exploration.
        """
        if not results:
            return [generate_random_alpha() for _ in range(self._pop_size)]

        # Build map DSL → Node
        dsl_to_node = {r.dsl: r.node or _try_parse(r.dsl) for r in results}
        dsl_to_node = {k: v for k, v in dsl_to_node.items() if v is not None}

        sorted_res = sorted(results, key=lambda x: x.fitness, reverse=True)

        # ── Elitism ──
        n_elite  = max(1, int(self._elite_ratio * self._pop_size))
        elite    = [dsl_to_node[r.dsl] for r in sorted_res[:n_elite] if r.dsl in dsl_to_node]
        next_gen = list(elite)
        seen     = {repr(n) for n in next_gen}

        # ── Tournament selector ──
        tourn_size = min(3, len(sorted_res))

        def tournament() -> Optional[Node]:
            contestants = random.sample(sorted_res, tourn_size)
            winner      = max(contestants, key=lambda x: x.fitness)
            return dsl_to_node.get(winner.dsl)

        # ── Fill remaining slots ──
        attempts = 0
        while len(next_gen) < self._pop_size and attempts < self._pop_size * 15:
            attempts += 1
            roll = random.random()

            try:
                # Choose operator based on adaptive weights
                op = _weighted_choice(mut_weights)

                if op == "crossover" and len(sorted_res) >= 2:
                    p1, p2 = tournament(), tournament()
                    if p1 is None or p2 is None:
                        continue
                    c1, c2 = subtree_crossover(p1, p2)
                    candidates = [c1, c2]

                elif op == "point":
                    parent = tournament()
                    if parent is None:
                        continue
                    candidates = [point_mutation(parent)]

                elif op == "hoist":
                    parent = tournament()
                    if parent is None:
                        continue
                    candidates = [hoist_mutation(parent)]

                elif op == "param":
                    parent = tournament()
                    if parent is None:
                        continue
                    candidates = [param_mutation(parent)]

                elif op == "wrap_rank":
                    parent = tournament()
                    if parent is None:
                        continue
                    candidates = [wrap_rank(parent)]

                elif op == "add_ts_smoothing":
                    parent = tournament()
                    if parent is None:
                        continue
                    candidates = [add_ts_smoothing(parent)]

                elif op == "add_condition":
                    parent = tournament()
                    if parent is None:
                        continue
                    candidates = [add_condition(parent)]

                elif op == "add_volume_filter":
                    parent = tournament()
                    if parent is None:
                        continue
                    candidates = [add_volume_filter(parent)]

                elif op == "combine_signals":
                    p1, p2 = tournament(), tournament()
                    if p1 is None or p2 is None:
                        continue
                    candidates = [combine_signals(p1, p2)]

                elif op == "replace_subtree":
                    parent = tournament()
                    if parent is None:
                        continue
                    candidates = [replace_subtree(parent)]

                elif op == "add_operator":
                    parent = tournament()
                    if parent is None:
                        continue
                    candidates = [add_operator(parent)]

                else:
                    # Exploration: random new individual
                    candidates = [generate_random_alpha()]

                for cand in candidates:
                    key = repr(cand)
                    if key in seen:
                        continue
                    try:
                        _validator.validate(cand)
                        next_gen.append(cand)
                        seen.add(key)
                    except Exception:
                        pass

            except Exception:
                pass

        # Pad with random individuals if still short
        while len(next_gen) < self._pop_size:
            try:
                node = generate_random_alpha()
                key  = repr(node)
                if key not in seen:
                    _validator.validate(node)
                    next_gen.append(node)
                    seen.add(key)
            except Exception:
                pass

        return next_gen[: self._pop_size]

    # ------------------------------------------------------------------
    # Optuna fine-tuning (Phase 4 — parameter search AFTER structure selected)
    # ------------------------------------------------------------------

    def _optuna_fine_tune(
        self,
        dsl:      str,
        n_trials: int,
    ) -> Tuple[Dict, Dict]:
        """
        Fine-tune execution parameters of the GP-selected best structure.
        Optuna searches: delay, decay, truncation, portfolio_mode.
        Returns (best_config_dict, final_metrics_dict).
        """
        if n_trials <= 0:
            return {}, self._quick_metrics(dsl)

        try:
            from app.core.ml_engine.alpha_optimizer import AlphaOptimizer, SearchSpace

            optimizer = AlphaOptimizer(
                dsl          = dsl,
                is_dataset   = self._is_data,
                search_space = SearchSpace(
                    delay_range     = (0, 3),
                    decay_range     = (0, 8),
                    portfolio_modes = ("long_short",),
                ),
                n_trials = n_trials,
                seed     = self._seed,
            )
            best_cfg, _ = optimizer.optimize()
            cfg_dict = {
                "delay":            best_cfg.delay,
                "decay_window":     best_cfg.decay_window,
                "truncation_min_q": best_cfg.truncation_min_q,
                "truncation_max_q": best_cfg.truncation_max_q,
                "portfolio_mode":   best_cfg.portfolio_mode,
            }

            # Final IS+OOS backtest with tuned config
            from app.core.backtest_engine.realistic_backtester import RealisticBacktester
            bt     = RealisticBacktester(config=best_cfg)
            result = bt.run(dsl, self._is_data, oos_dataset=self._oos_data)
            metrics = self._extract_metrics(result)
            return cfg_dict, metrics

        except Exception as exc:
            logger.warning("Optuna fine-tuning failed for '%s': %s", dsl[:60], exc)
            return {}, self._quick_metrics(dsl)

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def _quick_metrics(self, dsl: str) -> Dict[str, Any]:
        """Run backtest with default config, return metrics dict."""
        r = self._evaluate_one(dsl)
        if r is None:
            return {"is_sharpe": None, "oos_sharpe": None,
                    "overfitting_score": 0.0, "is_overfit": False}
        return {
            "is_sharpe":         r.sharpe_is,
            "oos_sharpe":        r.sharpe_oos,
            "is_turnover":       r.turnover,
            "overfitting_score": r.overfitting_score,
            "is_overfit":        r.overfitting_score > 0.5,
        }

    @staticmethod
    def _extract_metrics(result: Any) -> Dict[str, Any]:
        """Extract standard metrics dict from RealisticBacktestResult."""
        is_r  = result.is_report
        oos_r = result.oos_report

        def _f(v: Any) -> Optional[float]:
            try:
                fv = float(v)
                return None if np.isnan(fv) else fv
            except (TypeError, ValueError):
                return None

        is_s   = _f(is_r.sharpe_ratio) or 0.0
        oos_s  = _f(oos_r.sharpe_ratio) if oos_r else None
        turn   = _f(is_r.ann_turnover)  or 0.0

        overfit = 0.0
        if oos_s is not None and abs(is_s) > 1e-9:
            overfit = float(np.clip((is_s - oos_s) / abs(is_s), 0.0, 1.0))

        return {
            "is_sharpe":         is_s,
            "oos_sharpe":        oos_s,
            "is_return":         _f(is_r.annualized_return),
            "is_turnover":       turn,
            "is_ic":             _f(is_r.mean_ic),
            "overfitting_score": overfit,
            "is_overfit":        overfit > 0.5,
            "summary":           result.summary(),
        }

    @staticmethod
    def _make_fallback_metrics(dsl: str) -> Dict:
        return {
            "is_sharpe": None, "oos_sharpe": None,
            "overfitting_score": 0.0, "is_overfit": False,
            "summary": f"GP fallback — no valid evolution for '{dsl}'",
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _weighted_choice(weights: Dict[str, float]) -> str:
    """Sample a key from a probability dict."""
    keys  = list(weights.keys())
    probs = [weights[k] for k in keys]
    return random.choices(keys, weights=probs, k=1)[0]


def _try_parse(dsl: str) -> Optional[Node]:
    try:
        return _parser.parse(dsl)
    except Exception:
        return None
