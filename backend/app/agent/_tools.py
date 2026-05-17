"""
_tools.py — QuantTools: 6 agent tool implementations.

Each tool is callable directly (Fallback path) or wrapped as a LangChain @tool
(LangChain path) inside _lc_agent.py.

  Tool 1: tool_generate_alpha_dsl  — hypothesis → DSL (with field constraints)
  Tool 2: tool_run_gp_optimization — GP structural search + Optuna fine-tuning  ← PRIMARY
  Tool 3: tool_run_backtest        — IS+OOS validation backtest
  Tool 4: tool_mutate_ast          — single AST-level structural mutation (GP ops)
  Tool 5: tool_run_optuna          — IS-only Optuna parameter fine-tuning (secondary)
  Tool 6: tool_save_alpha          — persist alpha to SQLite AlphaStore

Phase 4 compliance:
  - tool_run_gp_optimization is the MAIN optimizer (structural search via GP)
  - tool_run_optuna is called ONLY inside GP's fine-tuning step (parameter polishing)
  - tool_mutate_ast now uses real AST mutations from gp_engine/mutations.py
    (NO string templates, NO fake mutation)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from app.agent._constants import (
    _DEFAULT_N_DAYS,
    _DEFAULT_N_TICKERS,
    _DEFAULT_N_TRIALS,
    _DEFAULT_OOS_RATIO,
    _FALLBACK_DSL_MAP,
    _VALID_FIELDS,
    _VALID_OPS,
)
from app.agent._data_utils import _make_synthetic_dataset, _partition, _run_backtest_core, load_real_dataset
from app.agent._helpers import _safe_json_loads

logger = logging.getLogger(__name__)


def _detect_factor_family(dsl: str) -> str:
    """
    Parse ``dsl`` and return its factor family via FinancialInterpreter.
    Returns empty string on any failure (safe fallback — GP runs with generic weights).
    """
    try:
        from app.core.alpha_engine.financial_interpreter import FinancialInterpreter
        return FinancialInterpreter().interpret(dsl).factor_family
    except Exception:
        return ""


class QuantTools:
    """
    Stateless (per-dataset) tool implementations shared across all sessions.

    The synthetic dataset is generated once at construction time with a fixed
    seed, making all backtest results deterministic and cross-session safe.
    """

    def __init__(
        self,
        n_tickers:     int   = _DEFAULT_N_TICKERS,
        n_days:        int   = _DEFAULT_N_DAYS,
        oos_ratio:     float = _DEFAULT_OOS_RATIO,
        n_trials:      int   = _DEFAULT_N_TRIALS,
        seed:          int   = 42,
        llm:           Any   = None,
        dataset_name:  str   = "",
        dataset_start: str   = "2021-01-01",
        dataset_end:   str   = "2024-01-01",
    ) -> None:
        self._n_tickers = n_tickers
        self._n_days    = n_days
        self._oos_ratio = oos_ratio
        self._n_trials  = n_trials
        self._seed      = seed
        self._llm       = llm

        if dataset_name:
            try:
                self._is_data, self._oos_data = load_real_dataset(
                    dataset_name, dataset_start, dataset_end, oos_ratio
                )
                self._full_dataset = {**self._is_data, **self._oos_data}
                logger.info("QuantTools: real dataset '%s' [%s→%s] loaded", dataset_name, dataset_start, dataset_end)
            except Exception as exc:
                logger.warning("Real dataset '%s' load failed: %s — falling back to synthetic", dataset_name, exc)
                ds = _make_synthetic_dataset(n_tickers, n_days, seed)
                self._full_dataset = ds
                self._is_data, self._oos_data = _partition(ds, oos_ratio)
        else:
            ds = _make_synthetic_dataset(n_tickers, n_days, seed)
            self._full_dataset = ds
            self._is_data, self._oos_data = _partition(ds, oos_ratio)

    # ------------------------------------------------------------------
    # Tool 1 — hypothesis → seed DSL
    # ------------------------------------------------------------------

    def tool_generate_alpha_dsl(self, hypothesis: str) -> str:
        """Translate a market hypothesis into an Alpha DSL seed expression.
        Returns JSON: {"dsl": str, "explanation": str}"""
        if self._llm is not None:
            return self._llm_generate_dsl(hypothesis)
        return self._fallback_generate_dsl(hypothesis)

    def _fallback_generate_dsl(self, hypothesis: str) -> str:
        h_lower = hypothesis.lower()
        for kw, dsl in _FALLBACK_DSL_MAP.items():
            if kw in h_lower:
                return json.dumps({
                    "dsl":         dsl,
                    "explanation": f"[Fallback] Mapped '{kw}' to seed DSL.",
                })
        return json.dumps({
            "dsl":         _FALLBACK_DSL_MAP["default"],
            "explanation": "[Fallback] No keyword match, using default seed.",
        })

    def _llm_generate_dsl(self, hypothesis: str) -> str:
        system = (
            "You are a DSL compiler for quantitative alpha strategies.\n"
            "Translate the user's market hypothesis into ONE Alpha DSL expression.\n\n"
            f"**CRITICAL: If the hypothesis explicitly mentions a data field "
            f"(e.g., 'vwap', 'volume', 'high', 'low'), you MUST include that exact "
            f"field in your DSL formula. Never substitute a different field.**\n\n"
            f"Available data fields: {_VALID_FIELDS}\n"
            f"Available operators: {_VALID_OPS}\n\n"
            "Output ONLY valid JSON (no markdown):\n"
            '{"dsl": "<expression>", "explanation": "<one sentence>"}'
        )
        few_shot = (
            "Examples:\n"
            "- hypothesis='vwap momentum' "
            '→ {"dsl": "rank(ts_delta(vwap, 5))", "explanation": "5-day VWAP change rank."}\n'
            "- hypothesis='volume spike' "
            '→ {"dsl": "rank(ts_delta(log(volume), 5))", "explanation": "Volume surge rank."}\n'
            "- hypothesis='price reversal' "
            '→ {"dsl": "rank(-ts_delta(close, 1))", "explanation": "1-day price reversal."}'
        )
        prompt = f"{system}\n\n{few_shot}\n\nHypothesis: {hypothesis}"
        try:
            response = self._llm.invoke(prompt)
            text     = response.content if hasattr(response, "content") else str(response)
            parsed   = _safe_json_loads(text)
            if parsed.get("dsl"):
                return json.dumps(parsed)
        except Exception as exc:
            logger.warning("LLM DSL 生成失败，降级: %s", exc)
        return self._fallback_generate_dsl(hypothesis)

    # ------------------------------------------------------------------
    # Tool 2 — GP structural search (PRIMARY optimizer)
    # ------------------------------------------------------------------

    def tool_run_gp_optimization(
        self,
        seed_dsl:         str = "",
        seed_dsls_json:   str = "",   # JSON list of seed DSLs for multi-seed init
        n_generations:    int = 4,
        pop_size:         int = 12,
        n_optuna_trials:  int = 8,
        factor_family:    str = "",   # e.g. "momentum","reversion","volatility"
        dataset_name:     str = "",   # real dataset override for this GP run
        dataset_start:    str = "2021-01-01",
        dataset_end:      str = "2024-01-01",
    ) -> str:
        """
        Run GP-driven alpha optimization.

        Structure space is searched by GP (AST mutation + crossover + selection).
        Optuna fine-tunes execution parameters of the GP-selected best structure ONLY.

        seed_dsls_json : optional JSON-encoded list of DSL strings (multi-seed init).
        factor_family  : factor family of the seed DSL(s).  When empty, auto-detected
                         from the first seed via FinancialInterpreter.  The family
                         biases mutation operator selection toward financially relevant
                         transformations for that factor type.

        Returns JSON: {best_dsl, generations_run, population_size, metrics, pool_top5,
                       factor_family}
        """
        from app.core.gp_engine.population_evolver import PopulationEvolver

        # Parse seed_dsls from JSON when provided (multi-seed population init)
        seed_dsls_list = None
        if seed_dsls_json:
            try:
                seed_dsls_list = json.loads(seed_dsls_json)
                if not isinstance(seed_dsls_list, list):
                    seed_dsls_list = None
            except Exception:
                pass

        # Auto-detect factor family from seed DSL if not provided
        if not factor_family:
            first_dsl = seed_dsl or (seed_dsls_list[0] if seed_dsls_list else "")
            if first_dsl:
                factor_family = _detect_factor_family(first_dsl)

        if factor_family:
            logger.info(
                "GP optimization: factor_family='%s' — mutation weights biased accordingly",
                factor_family,
            )

        # Resolve dataset: per-run override > session default
        if dataset_name:
            try:
                gp_is, gp_oos = load_real_dataset(
                    dataset_name, dataset_start, dataset_end, self._oos_ratio
                )
                logger.info("GP: using real dataset '%s' [%s→%s]", dataset_name, dataset_start, dataset_end)
            except Exception as exc:
                logger.warning("GP real dataset '%s' failed: %s — using session data", dataset_name, exc)
                gp_is, gp_oos = self._is_data, self._oos_data
        else:
            gp_is, gp_oos = self._is_data, self._oos_data

        effective_pop = max(pop_size, len(seed_dsls_list or []) + 4)
        evolver = PopulationEvolver(
            is_data        = gp_is,
            oos_data       = gp_oos,
            factor_family  = factor_family,
            pop_size       = effective_pop,
            n_generations  = n_generations,
            seed           = self._seed,
        )
        try:
            result = evolver.run(
                seed_dsl        = seed_dsl or None,
                seed_dsls       = seed_dsls_list,
                n_optuna_trials = n_optuna_trials,
            )
        except Exception as exc:
            logger.warning("GP 优化失败: %s", exc)
            fallback_dsl = seed_dsl or (seed_dsls_list[0] if seed_dsls_list else "rank(ts_delta(log(close), 5))")
            return json.dumps({
                "best_dsl":        fallback_dsl,
                "generations_run": 0,
                "population_size": effective_pop,
                "metrics":         {},
                "pool_top5":       [],
                "error":           str(exc),
            })

        return json.dumps({
            "best_dsl":        result.best_dsl,
            "generations_run": result.generations_run,
            "population_size": pop_size,
            "metrics":         result.metrics,
            "pool_top5":       result.pool_top5,
            "evolution_log":   result.evolution_log,
            "best_config":     result.best_config or {},
        }, default=str)

    # ------------------------------------------------------------------
    # Tool 3 — IS+OOS validation backtest
    # ------------------------------------------------------------------

    def tool_run_backtest(self, dsl: str, config_json: str = "{}") -> str:
        """Run full IS+OOS validation backtest with the given config.
        Returns JSON: {is_sharpe, oos_sharpe, overfitting_score, is_overfit, summary}"""
        from app.core.alpha_engine.signal_processor import SimulationConfig

        cfg_dict = _safe_json_loads(config_json) if config_json.strip() else {}
        cfg = SimulationConfig(
            delay            = cfg_dict.get("delay", 1),
            decay_window     = cfg_dict.get("decay_window", 0),
            truncation_min_q = cfg_dict.get("truncation_min_q", 0.05),
            truncation_max_q = cfg_dict.get("truncation_max_q", 0.95),
            portfolio_mode   = cfg_dict.get("portfolio_mode", "long_short"),
        )
        try:
            metrics = _run_backtest_core(dsl, cfg, self._is_data, self._oos_data)
        except Exception as exc:
            logger.warning("回测失败: %s", exc)
            metrics = {
                "is_sharpe": None, "oos_sharpe": None,
                "overfitting_score": 0.0, "is_overfit": False,
                "summary": f"Backtest failed: {exc}",
            }
        return json.dumps(metrics, default=str)

    # ------------------------------------------------------------------
    # Tool 4 — Single AST-level structural mutation  (REAL GP operations)
    # ------------------------------------------------------------------

    def tool_mutate_ast(
        self,
        current_dsl:     str  = "",
        overfit_reason:  str  = "",
        mutation_target: str  = "",
    ) -> str:
        """
        Apply ONE real AST-level structural mutation to the given DSL.

        Parameters
        ----------
        current_dsl     : DSL expression to mutate.
        overfit_reason  : Human-readable failure description (for weight bias
                          and LLM guidance when mutation_target is not given).
        mutation_target : Direct mutation key from CriticResult.recommended_mutation.
                          When set, bypasses all weight-based randomness and applies
                          the specified operator immediately (with retries).
                          Valid keys: hoist, wrap_rank, add_ts_smoothing, add_condition,
                          add_volume_filter, replace_subtree, add_operator, point, param.

        Returns
        -------
        JSON: {"mutated_dsl": str, "mutation_type": str, "explanation": str}
        """
        from app.core.alpha_engine.parser import Parser
        from app.core.alpha_engine.validator import AlphaValidator
        from app.core.gp_engine.mutations import (
            point_mutation, hoist_mutation, param_mutation,
            wrap_rank, add_ts_smoothing, add_condition, add_volume_filter,
            replace_subtree, add_operator,
        )
        from app.core.gp_engine.fitness import mutation_weights_from_metrics

        parser    = Parser()
        validator = AlphaValidator()

        # Full map of named mutations (including all structural operators)
        _ALL_OPS = {
            "hoist":             (hoist_mutation,    "hoist_mutation"),
            "wrap_rank":         (wrap_rank,          "wrap_rank"),
            "add_ts_smoothing":  (add_ts_smoothing,   "add_ts_smoothing"),
            "add_condition":     (add_condition,      "add_condition"),
            "add_volume_filter": (add_volume_filter,  "add_volume_filter"),
            "replace_subtree":   (replace_subtree,    "replace_subtree"),
            "add_operator":      (add_operator,       "add_operator"),
            "point":             (point_mutation,     "point_mutation"),
            "param":             (param_mutation,     "param_mutation"),
        }

        # Parse current DSL to AST
        try:
            node = parser.parse(current_dsl)
        except Exception as exc:
            logger.warning("tool_mutate_ast: parse failed '%s': %s", current_dsl[:60], exc)
            return json.dumps({
                "mutated_dsl":   current_dsl,
                "mutation_type": "parse_failed",
                "explanation":   f"Could not parse DSL: {exc}",
            })

        # ── Path A: Direct targeted mutation (from CriticResult) ─────────
        if mutation_target and mutation_target in _ALL_OPS:
            fn, label = _ALL_OPS[mutation_target]
            for attempt in range(10):
                try:
                    mutated = fn(node)
                    validator.validate(mutated)
                    logger.info(
                        "tool_mutate_ast [targeted=%s attempt=%d]: %s → %s",
                        mutation_target, attempt + 1,
                        current_dsl[:50], repr(mutated)[:50],
                    )
                    return json.dumps({
                        "mutated_dsl":   repr(mutated),
                        "mutation_type": label,
                        "explanation":   (
                            f"Targeted {label} applied (failure={mutation_target}): "
                            f"'{current_dsl[:40]}...'"
                        ),
                    })
                except Exception:
                    continue
            # Targeted mutation failed — fall through to weight-based

        # ── Path B: Weight-based selection (legacy / LLM-guided) ─────────
        overfit_score = 0.6 if "过拟合" in overfit_reason or "overfit" in overfit_reason.lower() else 0.1
        sharpe_low    = 0.1 if ("low" in overfit_reason.lower() or "低" in overfit_reason) else 0.5
        high_turn     = 2.5 if ("turnover" in overfit_reason.lower() or "换手" in overfit_reason) else 1.0

        mutation_type_hint = self._llm_guide_mutation(current_dsl, overfit_reason)

        weights = mutation_weights_from_metrics(
            sharpe_oos    = sharpe_low,
            turnover      = high_turn,
            overfit_score = overfit_score,
        )
        if mutation_type_hint and mutation_type_hint in weights:
            weights = {k: 0.05 for k in weights}
            weights[mutation_type_hint] = 0.85

        # Restrict to ops that don't need a second parent
        classic_ops = {
            k: v for k, v in _ALL_OPS.items()
            if k in ("point", "hoist", "param",
                     "wrap_rank", "add_ts_smoothing", "add_condition",
                     "replace_subtree", "add_operator")
        }
        pop_weights = {k: weights.get(k, 0.05) for k in classic_ops}
        total = sum(pop_weights.values()) or 1.0
        pop_weights = {k: v / total for k, v in pop_weights.items()}

        import random
        keys  = list(pop_weights.keys())
        probs = [pop_weights[k] for k in keys]

        for _ in range(8):
            chosen_key  = random.choices(keys, weights=probs, k=1)[0]
            fn, label   = classic_ops[chosen_key]
            try:
                mutated     = fn(node)
                mutated_dsl = repr(mutated)
                validator.validate(mutated)
                return json.dumps({
                    "mutated_dsl":   mutated_dsl,
                    "mutation_type": label,
                    "explanation":   f"AST {label} applied to '{current_dsl[:40]}...'",
                })
            except Exception:
                continue

        return json.dumps({
            "mutated_dsl":   current_dsl,
            "mutation_type": "no_valid_mutation",
            "explanation":   "All AST mutation attempts failed validation; returning original.",
        })

    def _llm_guide_mutation(self, dsl: str, reason: str) -> Optional[str]:
        """Ask LLM which mutation type to prefer. Returns key or None."""
        if self._llm is None or not reason:
            return None
        prompt = (
            f"Given this overfitting alpha: {dsl}\n"
            f"Reason: {reason}\n"
            "Which AST mutation should fix it?\n"
            'Reply with ONE word: "point", "hoist", or "param"'
        )
        try:
            resp = self._llm.invoke(prompt)
            text = (resp.content if hasattr(resp, "content") else str(resp)).strip().lower()
            for key in ("point", "hoist", "param"):
                if key in text:
                    return key
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Tool 5 — Optuna IS-only parameter fine-tuning (secondary / internal)
    # ------------------------------------------------------------------

    def tool_run_optuna(self, dsl: str, n_trials: int = 0) -> str:
        """Fine-tune execution parameters for a FIXED DSL structure via Optuna.
        Called after GP selects the best structure. OOS is never seen.
        Returns JSON: {"best_config": {...}, "best_fitness": float, "n_trials": int}"""
        from app.core.ml_engine.alpha_optimizer import AlphaOptimizer, SearchSpace

        n         = n_trials if n_trials > 0 else self._n_trials
        optimizer = AlphaOptimizer(
            dsl          = dsl,
            is_dataset   = self._is_data,
            search_space = SearchSpace(
                delay_range     = (0, 3),
                decay_range     = (0, 8),
                portfolio_modes = ("long_short",),
            ),
            n_trials = n,
            seed     = self._seed,
        )
        try:
            best_cfg, summary = optimizer.optimize()
        except Exception as exc:
            logger.warning("Optuna 优化失败: %s", exc)
            return json.dumps({
                "best_config":  {"delay": 1, "decay_window": 0, "portfolio_mode": "long_short"},
                "best_fitness": -999.0,
                "n_trials":     0,
                "error":        str(exc),
            })

        return json.dumps({
            "best_config": {
                "delay":            best_cfg.delay,
                "decay_window":     best_cfg.decay_window,
                "truncation_min_q": best_cfg.truncation_min_q,
                "truncation_max_q": best_cfg.truncation_max_q,
                "portfolio_mode":   best_cfg.portfolio_mode,
            },
            "best_fitness": summary.best_value,
            "n_trials":     summary.n_trials,
        })

    # ------------------------------------------------------------------
    # Tool 6 — Financial interpretation + diagnosis
    # ------------------------------------------------------------------

    def tool_interpret_factor(
        self,
        dsl:          str,
        metrics_json: str = "{}",
    ) -> str:
        """
        Interpret a DSL expression in financial terms and diagnose its weaknesses.

        Produces:
          - Natural language description (what the factor actually measures)
          - Factor family classification (momentum / reversion / volatility / etc.)
          - Design issues (missing normalization, no smoothing, etc.)
          - Improvement suggestions with concrete DSL patches
          - Financial theory behind each suggestion

        Returns JSON with keys:
          description, factor_family, data_fields, is_normalized, complexity,
          issues, suggestions, diagnosis, severity, regime_insight
        """
        from app.core.alpha_engine.financial_interpreter import FinancialInterpreter
        from app.core.alpha_engine.financial_diagnostics import FinancialDiagnostics

        interp  = FinancialInterpreter()
        diag_en = FinancialDiagnostics()

        try:
            result = interp.interpret(dsl)
        except Exception as exc:
            return json.dumps({"error": f"Interpretation failed: {exc}", "dsl": dsl})

        metrics = _safe_json_loads(metrics_json) if metrics_json.strip() else {}

        diagnosis = None
        if metrics:
            try:
                diagnosis = diag_en.diagnose(dsl, metrics, interpreter_result=result)
            except Exception as exc:
                logger.warning("Diagnosis failed: %s", exc)

        output = {
            "dsl":           dsl,
            "description":   result.description,
            "factor_family": result.factor_family,
            "family_desc":   result.family_desc,
            "data_fields":   result.data_fields,
            "max_window":    result.max_window,
            "is_normalized": result.is_normalized,
            "has_volume":    result.has_volume,
            "complexity":    result.complexity,
            "design_issues": result.issues,
            "design_suggestions": result.suggestions,
        }

        if diagnosis is not None:
            output["diagnosis"] = {
                "primary_issue":  diagnosis.primary_issue,
                "explanation":    diagnosis.diagnosis,
                "severity":       diagnosis.severity,
                "regime_insight": diagnosis.regime_insight,
                "top_suggestions": [
                    {
                        "action":      s.action,
                        "dsl_patch":   s.dsl_patch,
                        "reason":      s.reason,
                        "finance_why": s.finance_why,
                    }
                    for s in diagnosis.suggestions[:4]
                ],
            }

        return json.dumps(output, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Tool 7 — Persist to AlphaStore
    # ------------------------------------------------------------------

    def tool_save_alpha(
        self,
        name:         str,
        dsl:          str,
        metrics_json: str = "{}",
    ) -> str:
        """Persist a validated alpha to the SQLite AlphaStore.
        Returns JSON: {"id": int, "status": "saved", "dsl": str}"""
        from app.db.alpha_store import AlphaStore, AlphaResult

        metrics = _safe_json_loads(metrics_json) if metrics_json.strip() else {}
        result  = AlphaResult(
            dsl          = dsl,
            hypothesis   = name,
            sharpe       = float(metrics.get("is_sharpe")   or 0.0),
            ann_return   = float(metrics.get("is_return")   or 0.0),
            ann_turnover = float(metrics.get("is_turnover") or 0.0),
            ic_ir        = float(metrics.get("is_ic")       or 0.0),
        )
        try:
            store    = AlphaStore()
            alpha_id = store.save(result)
            return json.dumps({"id": alpha_id, "status": "saved", "dsl": dsl})
        except Exception as exc:
            logger.warning("AlphaStore.save 失败: %s", exc)
            return json.dumps({"id": -1, "status": "error", "detail": str(exc)})
