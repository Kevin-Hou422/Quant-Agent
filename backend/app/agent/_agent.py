"""
_agent.py — QuantAgent: the main conversational entry point.

Single-instance design: one QuantAgent serves all chat sessions.
All per-session state is externalized:
  - LangChain path  → SQLite via RunnableWithMessageHistory
  - Fallback path   → in-process _session_memories dict (ConversationMemory)
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

from app.agent._constants import (
    _DEFAULT_N_DAYS,
    _DEFAULT_N_TICKERS,
    _DEFAULT_N_TRIALS,
    _DEFAULT_OOS_RATIO,
)
from app.agent._fallback import FallbackOrchestrator
from app.agent._helpers import _extract_balanced
from app.agent._lc_agent import _build_langchain_agent
from app.agent._memory import ConversationMemory
from app.agent._tools import QuantTools

logger = logging.getLogger(__name__)


class QuantAgent:
    """
    Conversational quantitative research agent (Phase 4).

    Parameters
    ----------
    n_tickers  : tickers in synthetic dataset
    n_days     : trading days in synthetic dataset
    oos_ratio  : IS/OOS split ratio
    n_trials   : Optuna trial count
    seed       : random seed (kept stable for reproducibility)
    api_key    : OpenAI API key (falls back to OPENAI_API_KEY env var)
    model      : OpenAI model name
    chat_store : ChatStore instance for DB-backed history and Fallback persistence
    """

    def __init__(
        self,
        n_tickers:  int            = _DEFAULT_N_TICKERS,
        n_days:     int            = _DEFAULT_N_DAYS,
        oos_ratio:  float          = _DEFAULT_OOS_RATIO,
        n_trials:   int            = _DEFAULT_N_TRIALS,
        seed:       int            = 42,
        api_key:    Optional[str]  = None,
        model:      str            = "gpt-4o-mini",
        chat_store: Optional[Any]  = None,
    ) -> None:
        self._chat_store = chat_store
        self._llm        = None

        # Per-session state (both paths need DSL tracking; Fallback also needs memory)
        self._session_memories: Dict[str, ConversationMemory] = {}
        self._session_last_dsl: Dict[str, Optional[str]]      = {}

        # Initialise LLM if key available
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if key:
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model       = model,
                    api_key     = key,
                    temperature = 0.1,
                    max_tokens  = 800,
                )
                logger.info("LLM 初始化成功: %s", model)
            except Exception as exc:
                logger.warning("LLM 初始化失败，降级 Fallback: %s", exc)

        # Shared tool instance (dataset is deterministic → safe across sessions)
        self._tools = QuantTools(
            n_tickers = n_tickers,
            n_days    = n_days,
            oos_ratio = oos_ratio,
            n_trials  = n_trials,
            seed      = seed,
            llm       = self._llm,
        )

        # Orchestrators
        self._chain:    Optional[Any]        = None
        self._fallback: FallbackOrchestrator = FallbackOrchestrator(self._tools)

        if self._llm is not None:
            try:
                self._chain = _build_langchain_agent(self._llm, self._tools, chat_store)
            except Exception as exc:
                logger.warning("LangChain Agent 构建失败，降级: %s", exc)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Process one user message and return a structured response.

        Returns
        -------
        {"reply": str, "dsl": Optional[str], "metrics": Optional[dict]}
        """
        enriched          = self._enrich_message(message, session_id)
        intent, dsl_hint  = self._detect_intent(enriched, session_id)

        try:
            if self._chain is not None:
                result = self._lc_chat(enriched, session_id)
            else:
                result = self._fallback_chat(intent, enriched, dsl_hint, session_id)
        except Exception as exc:
            logger.exception("QuantAgent.chat 失败 session=%s", session_id)
            result = {"reply": f"处理失败: {exc}", "dsl": None, "metrics": None}

        # Track last DSL per session (used for context enrichment)
        if result.get("dsl"):
            self._session_last_dsl[session_id] = result["dsl"]

        # Fallback path: update in-process memory
        if self._chain is None:
            mem = self._get_or_create_memory(session_id)
            mem.add_user(message)
            mem.add_assistant(
                result.get("reply", ""),
                dsl     = result.get("dsl"),
                metrics = result.get("metrics"),
            )

        return result

    @property
    def tools(self) -> QuantTools:
        return self._tools

    # ------------------------------------------------------------------
    # LangChain path
    # ------------------------------------------------------------------

    def _lc_chat(self, message: str, session_id: str) -> Dict[str, Any]:
        """Invoke AgentExecutor via RunnableWithMessageHistory (session-scoped history)."""
        has_history = (
            hasattr(self._chain, "get_session_history")
            or type(self._chain).__name__ == "RunnableWithMessageHistory"
        )

        if has_history:
            resp = self._chain.invoke(
                {"input": message},
                config={"configurable": {"session_id": session_id}},
            )
        else:
            # Bare AgentExecutor (no DB history available)
            history = (
                self._get_or_create_memory(session_id).to_langchain_messages()
                if hasattr(self._get_or_create_memory(session_id), "to_langchain_messages")
                else []
            )
            resp = self._chain.invoke({
                "input":        message,
                "chat_history": history,
            })

        reply   = resp.get("output", "")
        dsl     = self._extract_dsl_from_text(reply) or self._session_last_dsl.get(session_id)
        return {"reply": reply[:600], "dsl": dsl, "metrics": None}

    # ------------------------------------------------------------------
    # Fallback path
    # ------------------------------------------------------------------

    def _fallback_chat(
        self,
        intent:     str,
        message:    str,
        dsl_hint:   Optional[str],
        session_id: str,
    ) -> Dict[str, Any]:
        if intent == "workflow_b" and dsl_hint:
            dsl, metrics = self._fallback.run_workflow_b(dsl_hint)
        else:
            dsl, metrics = self._fallback.run_workflow_a(message)

        # Persist via chat_store (LangChain path handles its own persistence)
        if self._chat_store is not None:
            try:
                self._chat_store.ensure_session(session_id)
                self._chat_store.save_message(session_id, "user", message)
            except Exception:
                pass

        oos_s   = metrics.get("oos_sharpe")
        is_s    = metrics.get("is_sharpe")
        overfit = metrics.get("overfitting_score", 0.0)

        reply = (
            f"DSL: {dsl} | "
            f"IS Sharpe={is_s:.3f}  OOS Sharpe={oos_s:.3f}  "
            f"过拟合={overfit:.2f}  "
            f"{'⚠ 过拟合' if metrics.get('is_overfit') else '✓ 通过'}"
        ) if is_s is not None else f"生成 DSL: {dsl}（回测数据不足）"

        if self._chat_store is not None:
            try:
                self._chat_store.save_message(session_id, "assistant", reply)
            except Exception:
                pass

        return {"reply": reply, "dsl": dsl, "metrics": metrics}

    # ------------------------------------------------------------------
    # Per-session memory helpers
    # ------------------------------------------------------------------

    def _get_or_create_memory(self, session_id: str) -> ConversationMemory:
        if session_id not in self._session_memories:
            mem = ConversationMemory(max_turns=10)
            # Warm up from DB so resumed sessions have context
            if self._chat_store is not None:
                try:
                    history = self._chat_store.get_history(session_id)
                    for msg in history[-20:]:
                        if msg.role == "user":
                            mem.add_user(msg.content)
                        elif msg.role == "assistant":
                            mem.add_assistant(msg.content)
                except Exception:
                    pass
            self._session_memories[session_id] = mem
        return self._session_memories[session_id]

    # ------------------------------------------------------------------
    # Message enrichment & intent detection
    # ------------------------------------------------------------------

    def _enrich_message(self, message: str, session_id: str) -> str:
        """Inject last-DSL context (Fallback only; LangChain injects via MessagesPlaceholder)."""
        if self._chain is not None:
            return message
        last_dsl = self._session_last_dsl.get(session_id)
        if last_dsl and any(
            kw in message.lower()
            for kw in ["last", "previous", "上一个", "刚才", "that one", "那个"]
        ):
            return f"{message} [Context: last DSL was '{last_dsl}']"
        return message

    def _detect_intent(
        self,
        message:    str,
        session_id: str,
    ) -> Tuple[str, Optional[str]]:
        """
        Fast 0-token intent classification → ("workflow_a" | "workflow_b", dsl_hint).

        workflow_b is triggered when the message contains an explicit DSL expression
        or when the user asks to optimise/mutate the last known DSL.
        """
        dsl_match = re.search(
            r"(-?\s*(?:rank|ts_mean|ts_std|ts_delta|ts_decay_linear|log|abs|sign|zscore|scale)"
            r"\s*\()",
            message,
        )
        if dsl_match:
            dsl_hint = _extract_balanced(message, dsl_match.start())
            if dsl_hint:
                return "workflow_b", dsl_hint

        last_dsl = self._session_last_dsl.get(session_id)
        if last_dsl and any(
            kw in message.lower()
            for kw in ["optimize", "improve", "refine", "mutate",
                       "优化", "改进", "变异", "结构"]
        ):
            return "workflow_b", last_dsl

        return "workflow_a", None

    # ------------------------------------------------------------------
    # Streaming chat (SSE path)
    # ------------------------------------------------------------------

    def stream_chat(
        self,
        message:    str,
        session_id: str            = "default",
        on_event:   Optional[Any]  = None,
    ) -> None:
        """
        Streaming version of chat.  Calls on_event with:
          {"type": "text",  "text": str}          — incremental progress line
          {"type": "done",  "result": {...}}       — final {reply, dsl, metrics}
          {"type": "error", "message": str}        — terminal error
        Runs synchronously — call inside a background thread for SSE.
        """
        def _emit_text(text: str) -> None:
            if on_event:
                try: on_event({"type": "text", "text": text})
                except Exception: pass

        def _emit_done(result: dict) -> None:
            if on_event:
                try: on_event({"type": "done", "result": result})
                except Exception: pass

        def _emit_error(msg: str) -> None:
            if on_event:
                try: on_event({"type": "error", "message": msg})
                except Exception: pass

        enriched          = self._enrich_message(message, session_id)
        intent, dsl_hint  = self._detect_intent(enriched, session_id)
        dataset           = self._tools._full_dataset
        oos_ratio         = self._tools._oos_ratio
        seed              = self._tools._seed

        try:
            from app.core.workflows.alpha_workflows import (
                GenerationWorkflow, OptimizationWorkflow,
            )

            if intent == "workflow_b" and dsl_hint:
                _emit_text(f"[Diagnose] 检测到 DSL 表达式，启动结构优化...")
                _emit_text(f"[Workflow B] 输入: {dsl_hint[:70]}")
                wf = OptimizationWorkflow(
                    pop_size=20, n_generations=7, n_optuna_trials=10,
                    n_mutations=8, oos_ratio=oos_ratio, seed=seed,
                )
                result = wf.run(dsl_hint, dataset, on_progress=_emit_text)
            else:
                _emit_text(f"[Diagnose] 解析策略假设: {message[:80]}")
                _emit_text("[Workflow A] 启动 GP 进化生成流程...")
                wf = GenerationWorkflow(
                    pop_size=20, n_generations=7, n_optuna_trials=10,
                    n_seed_dsls=12, oos_ratio=oos_ratio, seed=seed,
                )
                result = wf.run(message, dataset, on_progress=_emit_text)

            # Persist to chat store
            if self._chat_store is not None:
                try:
                    self._chat_store.ensure_session(session_id)
                    self._chat_store.save_message(session_id, "user", message)
                    self._chat_store.save_message(
                        session_id, "assistant",
                        result.explanation or result.best_dsl,
                    )
                except Exception:
                    pass

            # Track last DSL per session
            self._session_last_dsl[session_id] = result.best_dsl

            # Update in-process memory for Fallback path
            if self._chain is None:
                mem = self._get_or_create_memory(session_id)
                mem.add_user(message)
                mem.add_assistant(result.explanation or "", dsl=result.best_dsl)

            m = result.metrics
            _emit_done({
                "reply":   result.explanation or f"生成完成: {result.best_dsl}",
                "dsl":     result.best_dsl,
                "metrics": {
                    "sharpe_ratio":      m.get("is_sharpe"),
                    "annualized_return": m.get("is_return"),
                    "ic_ir":             m.get("is_ic"),
                },
            })

        except Exception as exc:
            logger.exception("stream_chat failed session=%s", session_id)
            _emit_error(str(exc))

    def _extract_dsl_from_text(self, text: str) -> Optional[str]:
        """Try to extract a DSL expression embedded in the agent's natural-language reply."""
        m = re.search(
            r"(-?\s*(?:rank|ts_mean|ts_std|ts_delta|ts_decay_linear)"
            r"\s*\([^)]*(?:\([^)]*\)[^)]*)*\))",
            text,
        )
        if m:
            return m.group(0).strip()
        return None
