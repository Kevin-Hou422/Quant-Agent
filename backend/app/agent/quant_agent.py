"""
quant_agent.py — public entry point for the QuantAgent package.

Implementation is split across private modules for maintainability:

  _constants.py     — thresholds, DSL maps, mutation templates
  _helpers.py       — _extract_balanced, _safe_json_loads
  _chat_history.py  — SQLAlchemyChatMessageHistory, _make_history_factory
  _memory.py        — TurnRecord, ConversationMemory
  _data_utils.py    — synthetic dataset, IS/OOS partition, backtest core
  _tools.py         — QuantTools (5 agent tools)
  _critic.py        — OverfitCritic (anti-overfitting gate)
  _prompts.py       — LangChain system prompt
  _lc_agent.py      — _build_langchain_agent (RunnableWithMessageHistory)
  _fallback.py      — FallbackOrchestrator (no-LLM workflow)
  _agent.py         — QuantAgent (main class)

This module re-exports the public API so existing imports remain unchanged.
"""
from app.agent._agent        import QuantAgent
from app.agent._chat_history import SQLAlchemyChatMessageHistory
from app.agent._fallback     import FallbackOrchestrator
from app.agent._memory       import ConversationMemory
from app.agent._tools        import QuantTools

__all__ = [
    "QuantAgent",
    "ConversationMemory",
    "FallbackOrchestrator",
    "QuantTools",
    "SQLAlchemyChatMessageHistory",
]
