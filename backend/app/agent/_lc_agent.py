"""
_lc_agent.py — _build_langchain_agent.

Wires QuantTools into a LangChain AgentExecutor and wraps it with
RunnableWithMessageHistory so every session gets its own DB-backed history.
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
    Build a LangChain AgentExecutor (5 tools) optionally wrapped with
    RunnableWithMessageHistory for per-session SQLite-backed memory.

    Parameters
    ----------
    llm        : initialised ChatOpenAI instance
    tools_obj  : shared QuantTools instance
    chat_store : ChatStore; if provided, wraps executor with history; else returns bare executor
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

    # ── Wrap 5 tools as LangChain @tool callables ─────────────────────

    @lc_tool
    def tool_generate_alpha_dsl(hypothesis: str) -> str:
        """Translate a natural language market hypothesis into an Alpha DSL expression.
        CRITICAL: if the hypothesis mentions a specific data field (vwap, volume, etc.),
        the generated DSL MUST include that field."""
        return tools_obj.tool_generate_alpha_dsl(hypothesis)

    @lc_tool
    def tool_run_optuna(dsl: str, n_trials: int = 15) -> str:
        """Run Optuna hyperparameter search on IS dataset only (OOS is locked).
        Returns best config (delay, decay, truncation) and IS fitness score."""
        return tools_obj.tool_run_optuna(dsl, n_trials)

    @lc_tool
    def tool_run_backtest(dsl: str, config_json: str = "{}") -> str:
        """Run full IS+OOS backtest.
        Returns IS/OOS Sharpe, overfitting_score, is_overfit flag."""
        return tools_obj.tool_run_backtest(dsl, config_json)

    @lc_tool
    def tool_mutate_ast(current_dsl: str, overfit_reason: str = "") -> str:
        """Structurally mutate an overfitting Alpha DSL using Genetic Programming.
        Changes mathematical structure (operators, fields, signal combinations),
        not just Optuna parameters.  Call when is_overfit=true BEFORE re-running Optuna."""
        return tools_obj.tool_mutate_ast(current_dsl, overfit_reason)

    @lc_tool
    def tool_save_alpha(name: str, dsl: str, metrics_json: str = "{}") -> str:
        """Save the validated alpha strategy to the AlphaStore SQLite ledger."""
        return tools_obj.tool_save_alpha(name, dsl, metrics_json)

    lc_tools = [
        tool_generate_alpha_dsl,
        tool_run_optuna,
        tool_run_backtest,
        tool_mutate_ast,
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
        max_iterations        = 12,
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
