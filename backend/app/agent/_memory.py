"""
_memory.py — lightweight in-process conversation memory for the Fallback path.

When no LLM is configured, ConversationMemory replaces LangChain's history system.
QuantAgent maintains one ConversationMemory instance per session_id in a plain dict,
giving the same isolation guarantee as SQLAlchemyChatMessageHistory but purely in RAM.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional


@dataclass
class TurnRecord:
    role:    str
    content: str
    dsl:     Optional[str]             = None
    metrics: Optional[Dict[str, Any]]  = None


class ConversationMemory:
    """
    Circular-buffer conversation memory for the no-LLM Fallback path.

    Parameters
    ----------
    max_turns : retain at most this many (user + assistant) turn pairs
    """

    def __init__(self, max_turns: int = 10) -> None:
        self._buffer:       Deque[TurnRecord]        = deque(maxlen=max_turns * 2)
        self._last_dsl:     Optional[str]            = None
        self._last_metrics: Optional[Dict[str, Any]] = None

    def add_user(self, content: str) -> None:
        self._buffer.append(TurnRecord(role="user", content=content))

    def add_assistant(
        self,
        content: str,
        dsl:     Optional[str]            = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        rec = TurnRecord(role="assistant", content=content, dsl=dsl, metrics=metrics)
        self._buffer.append(rec)
        if dsl:
            self._last_dsl = dsl
        if metrics:
            self._last_metrics = metrics

    @property
    def last_dsl(self) -> Optional[str]:
        return self._last_dsl

    @property
    def last_metrics(self) -> Optional[Dict[str, Any]]:
        return self._last_metrics
