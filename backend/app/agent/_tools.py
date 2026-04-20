"""
_tools.py — QuantTools: 5 agent tool implementations.

Each tool is callable directly (Fallback path) or wrapped as a LangChain @tool
(LangChain path) inside _lc_agent.py.

  Tool 1: tool_generate_alpha_dsl  — hypothesis → DSL (with field constraints)
  Tool 2: tool_run_optuna          — IS-only Optuna hyperparameter search
  Tool 3: tool_run_backtest        — IS+OOS backtest via RealisticBacktester
  Tool 4: tool_mutate_ast          — structural DSL mutation (Genetic Programming)
  Tool 5: tool_save_alpha          — persist alpha to SQLite AlphaStore
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

from app.agent._constants import (
    _DEFAULT_N_DAYS,
    _DEFAULT_N_TICKERS,
    _DEFAULT_N_TRIALS,
    _DEFAULT_OOS_RATIO,
    _FALLBACK_DSL_MAP,
    _FALLBACK_MUTATION_TEMPLATES,
    _VALID_FIELDS,
    _VALID_OPS,
)
from app.agent._data_utils import _make_synthetic_dataset, _partition, _run_backtest_core
from app.agent._helpers import _safe_json_loads

logger = logging.getLogger(__name__)


class QuantTools:
    """
    Stateless (per-dataset) tool implementations shared across all sessions.

    The synthetic dataset is generated once at construction time with a fixed
    seed, making all backtest results deterministic and cross-session safe.
    """

    def __init__(
        self,
        n_tickers: int   = _DEFAULT_N_TICKERS,
        n_days:    int   = _DEFAULT_N_DAYS,
        oos_ratio: float = _DEFAULT_OOS_RATIO,
        n_trials:  int   = _DEFAULT_N_TRIALS,
        seed:      int   = 42,
        llm:       Any   = None,
    ) -> None:
        self._n_tickers = n_tickers
        self._n_days    = n_days
        self._oos_ratio = oos_ratio
        self._n_trials  = n_trials
        self._seed      = seed
        self._llm       = llm

        ds = _make_synthetic_dataset(n_tickers, n_days, seed)
        self._is_data, self._oos_data = _partition(ds, oos_ratio)

    # ------------------------------------------------------------------
    # Tool 1 — hypothesis → DSL
    # ------------------------------------------------------------------

    def tool_generate_alpha_dsl(self, hypothesis: str) -> str:
        """Translate a market hypothesis into an Alpha DSL expression.
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
                    "explanation": f"[Fallback] Mapped '{kw}' to DSL.",
                })
        return json.dumps({
            "dsl":         _FALLBACK_DSL_MAP["default"],
            "explanation": "[Fallback] No keyword match, using default.",
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
    # Tool 2 — Optuna IS-only hyperparameter search
    # ------------------------------------------------------------------

    def tool_run_optuna(self, dsl: str, n_trials: int = 0) -> str:
        """Run Optuna on IS dataset (OOS is never touched).
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
    # Tool 3 — IS+OOS full backtest
    # ------------------------------------------------------------------

    def tool_run_backtest(self, dsl: str, config_json: str = "{}") -> str:
        """Run full IS+OOS backtest.
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
                "summary": f"回测失败: {exc}",
            }
        return json.dumps(metrics, default=str)

    # ------------------------------------------------------------------
    # Tool 4 — Structural mutation (Genetic Programming)
    # ------------------------------------------------------------------

    def tool_mutate_ast(self, current_dsl: str, overfit_reason: str = "") -> str:
        """Structurally mutate an overfitting Alpha DSL.
        Returns JSON: {"mutated_dsl": str, "mutation_type": str, "explanation": str}"""
        if self._llm is not None:
            return self._llm_mutate_ast(current_dsl, overfit_reason)
        return self._fallback_mutate_ast(current_dsl)

    def _llm_mutate_ast(self, current_dsl: str, overfit_reason: str) -> str:
        system = (
            "You are a Genetic Programmer for quantitative alpha strategies.\n"
            "Your task: apply ONE structural mutation to fix overfitting.\n\n"
            "MUTATION RULES:\n"
            "1. You MUST change the MATHEMATICAL STRUCTURE, not just window numbers.\n"
            "2. Valid structural mutations include:\n"
            "   - Wrap the expression in rank() to reduce noise\n"
            "   - Multiply by a volume filter: * rank(ts_mean(volume, 20))\n"
            "   - Add ts_decay_linear() smoothing as an outer wrapper\n"
            "   - Replace ts_delta with ts_std (magnitude vs. direction)\n"
            "   - Combine with a complementary signal: (A + B) * 0.5\n"
            "   - Use zscore() instead of rank() for normalization\n"
            "   - Substitute data field: close → vwap, returns → ts_delta(close,1)\n"
            f"3. Available operators: {_VALID_OPS}\n"
            f"4. Available data fields: {_VALID_FIELDS}\n"
            "5. Do NOT produce nested cross-sectional ops (e.g. rank(rank(...))).\n\n"
            "Output ONLY JSON (no markdown):\n"
            '{"mutated_dsl": "<expression>", "mutation_type": "<type>", '
            '"explanation": "<one sentence>"}'
        )
        examples = (
            "Examples:\n"
            '- Input: rank(ts_delta(close,5)) → '
            '{"mutated_dsl": "rank(ts_delta(close,5)) * rank(-ts_std(returns,20))", '
            '"mutation_type": "vol_filter", "explanation": "Volume-volatility filter added."}\n'
            '- Input: rank(ts_mean(returns,10)) → '
            '{"mutated_dsl": "rank(ts_decay_linear(ts_mean(returns,10),5))", '
            '"mutation_type": "decay_smooth", "explanation": "Linear decay smoothing added."}\n'
            '- Input: rank(ts_delta(vwap,5)) → '
            '{"mutated_dsl": "(rank(ts_delta(vwap,5)) + rank(-ts_delta(close,1))) * 0.5", '
            '"mutation_type": "signal_combine", '
            '"explanation": "Combined VWAP momentum with price reversal."}'
        )
        prompt = (
            f"{system}\n\n{examples}\n\n"
            f"Current DSL (overfitting): {current_dsl}\n"
            f"Overfit reason: {overfit_reason or 'OOS Sharpe degradation > 50%'}\n\n"
            "Apply ONE structural mutation:"
        )
        try:
            response = self._llm.invoke(prompt)
            text     = response.content if hasattr(response, "content") else str(response)
            parsed   = _safe_json_loads(text)
            if parsed.get("mutated_dsl"):
                try:
                    from app.core.alpha_engine.parser import Parser
                    from app.core.alpha_engine.validator import AlphaValidator
                    node = Parser().parse(parsed["mutated_dsl"])
                    AlphaValidator().validate(node)
                    return json.dumps(parsed)
                except Exception as val_exc:
                    logger.warning(
                        "LLM 变异后 DSL 验证失败 '%s': %s",
                        parsed["mutated_dsl"], val_exc,
                    )
        except Exception as exc:
            logger.warning("LLM AST 变异失败，降级: %s", exc)
        return self._fallback_mutate_ast(current_dsl)

    def _fallback_mutate_ast(self, current_dsl: str) -> str:
        """Pure-Python structural mutation using hash-selected templates."""
        idx = int(hashlib.md5(current_dsl.encode()).hexdigest(), 16)
        for offset in range(len(_FALLBACK_MUTATION_TEMPLATES)):
            mutation_type, template = _FALLBACK_MUTATION_TEMPLATES[
                (idx + offset) % len(_FALLBACK_MUTATION_TEMPLATES)
            ]
            try:
                mutated = template.format(dsl=current_dsl)
                from app.core.alpha_engine.parser import Parser
                from app.core.alpha_engine.validator import AlphaValidator
                node = Parser().parse(mutated)
                AlphaValidator().validate(node)
                return json.dumps({
                    "mutated_dsl":   mutated,
                    "mutation_type": mutation_type,
                    "explanation":   f"[Fallback] Applied {mutation_type} structural mutation.",
                })
            except Exception:
                continue

        safe_dsl = "rank(ts_delta(log(close), 5))"
        return json.dumps({
            "mutated_dsl":   safe_dsl,
            "mutation_type": "safe_default",
            "explanation":   "[Fallback] All templates failed; using safe default DSL.",
        })

    # ------------------------------------------------------------------
    # Tool 5 — Persist to AlphaStore
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
