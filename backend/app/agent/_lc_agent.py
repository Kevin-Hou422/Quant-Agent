"""
_lc_agent.py — _build_langchain_agent.

Wires QuantTools into a LangChain AgentExecutor and wraps it with
RunnableWithMessageHistory for per-session DB-backed memory.

Tool ordering (weaknesses 1-7 compliant):
  tool_generate_alpha_dsl  — seed DSL from hypothesis
  tool_interpret_factor    — DSL → factor_family + design diagnosis  [NEW Tool 7]
  tool_run_gp_optimization — PRIMARY optimizer; accepts factor_family + dataset_name
  tool_run_backtest        — IS+OOS validation
  tool_mutate_ast          — single AST mutation; accepts mutation_target for direct dispatch
  tool_run_optuna          — SECONDARY (parameter fine-tuning only)
  tool_save_alpha          — persist to AlphaStore

Key parameter links (weakness-7 fix):
  tool_interpret_factor(seed_dsl) → {factor_family, ...}
  tool_run_gp_optimization(seed_dsl, factor_family=<above>)
    → GP mutation weights biased toward financially appropriate operators

  tool_run_backtest → {overfitting_score, is_turnover, ...}
  tool_mutate_ast(dsl, reason, mutation_target=<from CriticResult>)
    → targeted AST correction instead of random mutation
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.agent._chat_history import _make_history_factory
from app.agent._prompts import _SYSTEM_PROMPT
from app.agent._tools import QuantTools

logger = logging.getLogger(__name__)


def _build_langchain_agent(
    llm:        Any,
    tools_obj:  QuantTools,
    chat_store: Optional[Any] = None,
) -> Any:
    """
    Build a LangChain AgentExecutor (6 tools) optionally wrapped with
    RunnableWithMessageHistory for per-session SQLite-backed memory.
    """
    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain.tools import tool as lc_tool
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    except ImportError as exc:
        raise ImportError(
            "需要安装 langchain 和 langchain-openai: "
            "pip install langchain langchain-openai langchain-core"
        ) from exc

    # ── Tool 1: hypothesis → seed DSL ─────────────────────────────────

    @lc_tool
    def tool_generate_alpha_dsl(hypothesis: str) -> str:
        """Translate a natural language market hypothesis into a seed Alpha DSL expression.
        CRITICAL: if the hypothesis mentions a specific data field (vwap, volume, etc.),
        the generated DSL MUST include that field. This seed is then evolved by GP."""
        return tools_obj.tool_generate_alpha_dsl(hypothesis)

    # ── Tool 2: GP structural search (PRIMARY optimizer) ──────────────

    @lc_tool
    def tool_run_gp_optimization(
        seed_dsl:        str = "",
        n_generations:   int = 4,
        pop_size:        int = 12,
        n_optuna_trials: int = 8,
        factor_family:   str = "",
        dataset_name:    str = "",
        dataset_start:   str = "2021-01-01",
        dataset_end:     str = "2024-01-01",
    ) -> str:
        """
        PRIMARY OPTIMIZER. Run GP-driven alpha evolution.

        1. Initialises population with seed_dsl + random AST variants.
        2. Evolves for n_generations using AST mutation + subtree crossover + selection.
        3. Multi-objective fitness: sharpe_oos - 0.2*turnover - 0.3*overfit_penalty.
        4. Diversity filter: rejects alphas with signal correlation > 0.9.
        5. After evolution, Optuna fine-tunes ONLY the best structure's parameters.

        factor_family: pass the value from tool_interpret_factor to bias GP mutations
          toward financially appropriate operators for this factor type.
          (e.g. "momentum", "reversion", "volatility", "liquidity", "composite")
        dataset_name: optional real market dataset (e.g. "us_sectors", "cn_ashares").
          When omitted, uses the session-default dataset.

        Returns: best_dsl, metrics, generations_run, pool_top5, evolution_log.
        Use this as the MAIN optimizer — NOT tool_run_optuna standalone.
        """
        return tools_obj.tool_run_gp_optimization(
            seed_dsl        = seed_dsl,
            n_generations   = n_generations,
            pop_size        = pop_size,
            n_optuna_trials = n_optuna_trials,
            factor_family   = factor_family,
            dataset_name    = dataset_name,
            dataset_start   = dataset_start,
            dataset_end     = dataset_end,
        )

    # ── Tool 3: IS+OOS validation backtest ────────────────────────────

    @lc_tool
    def tool_run_backtest(dsl: str, config_json: str = "{}") -> str:
        """Run full IS+OOS validation backtest with a given config.
        Returns IS/OOS Sharpe, overfitting_score, is_overfit flag.
        Use after GP to validate the final selected alpha."""
        return tools_obj.tool_run_backtest(dsl, config_json)

    # ── Tool 4: Single AST structural mutation ────────────────────────

    @lc_tool
    def tool_mutate_ast(
        current_dsl:     str,
        overfit_reason:  str = "",
        mutation_target: str = "",
    ) -> str:
        """
        Apply ONE real AST-level structural mutation (NOT string templates).
        Uses GP engine operations: point_mutation, hoist_mutation, param_mutation.
        Adaptive weights based on overfit_reason (turnover / sharpe / overfitting).
        Call when GP result still shows overfitting after tool_run_gp_optimization.

        mutation_target: directly dispatch to a specific GP operator. Accepted values:
          replace_subtree, hoist, add_ts_smoothing, add_condition, wrap_rank, point
          (pass the recommended_mutation from tool_interpret_factor diagnosis).
        """
        return tools_obj.tool_mutate_ast(current_dsl, overfit_reason, mutation_target)

    # ── Tool 5: Optuna parameter fine-tuning (SECONDARY) ──────────────

    @lc_tool
    def tool_run_optuna(dsl: str, n_trials: int = 10) -> str:
        """SECONDARY optimizer — fine-tunes execution parameters (delay, decay, truncation)
        for a FIXED DSL structure. OOS is never seen during optimization.
        NOTE: GP already calls this internally. Only call manually if GP was skipped."""
        return tools_obj.tool_run_optuna(dsl, n_trials)

    # ── Tool 6: Persist to AlphaStore ─────────────────────────────────

    @lc_tool
    def tool_save_alpha(name: str, dsl: str, metrics_json: str = "{}") -> str:
        """Save the validated alpha strategy to the AlphaStore SQLite ledger."""
        return tools_obj.tool_save_alpha(name, dsl, metrics_json)

    # ── Tool 7: Financial interpretation + diagnosis ───────────────────

    @lc_tool
    def tool_interpret_factor(dsl: str, metrics_json: str = "{}") -> str:
        """
        Interpret a DSL alpha expression in financial terms and diagnose weaknesses.

        ALWAYS call this BEFORE tool_run_gp_optimization.
        The returned factor_family MUST be passed to tool_run_gp_optimization so that
        GP mutation weights are biased toward financially appropriate operators.

        Returns JSON with:
          factor_family   : use as factor_family param in tool_run_gp_optimization
          description     : what the factor measures in plain language
          design_issues   : list of detected design flaws (missing normalization, etc.)
          design_suggestions : DSL-level fix patches for each issue
          diagnosis       : (if metrics_json provided) metric-driven financial diagnosis
            recommended_mutation : pass as mutation_target to tool_mutate_ast
        """
        return tools_obj.tool_interpret_factor(dsl, metrics_json)

    lc_tools = [
        tool_generate_alpha_dsl,
        tool_interpret_factor,
        tool_run_gp_optimization,
        tool_run_backtest,
        tool_mutate_ast,
        tool_run_optuna,
        tool_save_alpha,
    ]

    # ── Prompt with chat_history placeholder ──────────────────────────

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent    = create_tool_calling_agent(llm, lc_tools, prompt)
    executor = AgentExecutor(
        agent                 = agent,
        tools                 = lc_tools,
        verbose               = False,
        max_iterations        = 15,
        handle_parsing_errors = True,
    )

    # ── Wrap with DB-backed session history if chat_store provided ─────

    if chat_store is not None:
        try:
            from langchain_core.runnables.history import RunnableWithMessageHistory
            chain = RunnableWithMessageHistory(
                executor,
                get_session_history  = _make_history_factory(chat_store),
                input_messages_key   = "input",
                history_messages_key = "chat_history",
            )
            logger.info("RunnableWithMessageHistory 构建成功（DB 持久化记忆）")
            return chain
        except ImportError:
            logger.warning("RunnableWithMessageHistory 不可用，退回裸 AgentExecutor")

    return executor
