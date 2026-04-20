"""
_chat_history.py — SQLAlchemyChatMessageHistory + _make_history_factory.

Implements the LangChain BaseChatMessageHistory interface backed by ChatStore (SQLite).
All reads/writes are scoped by WHERE session_id = :sid — session A cannot read any
records belonging to session B.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


class SQLAlchemyChatMessageHistory:
    """
    LangChain BaseChatMessageHistory backed by SQLAlchemy ChatStore.

    Parameters
    ----------
    session_id  : UUID matching chat_sessions.id
    chat_store  : ChatStore instance (injected via dependencies.get_chat_store)
    max_history : maximum number of recent messages loaded per request (token guard)
    """

    def __init__(
        self,
        session_id:  str,
        chat_store:  Any,
        max_history: int = 20,
    ) -> None:
        self._session_id  = session_id
        self._store       = chat_store
        self._max_history = max_history
        self._store.ensure_session(session_id)   # idempotent FK guard

    # ------------------------------------------------------------------
    # BaseChatMessageHistory interface
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list:
        """Load this session's history from DB → LangChain message objects."""
        try:
            from langchain_core.messages import HumanMessage, AIMessage
        except ImportError:
            return []

        rows = self._store.get_history(self._session_id)
        rows = rows[-self._max_history:]   # prevent context overflow
        result = []
        for row in rows:
            if row.role == "user":
                result.append(HumanMessage(content=row.content))
            else:
                result.append(AIMessage(content=row.content))
        return result

    def add_message(self, message: Any) -> None:
        """Persist a LangChain BaseMessage to DB (truncated to 2 000 chars)."""
        try:
            from langchain_core.messages import HumanMessage
            role = "user" if isinstance(message, HumanMessage) else "assistant"
        except ImportError:
            role = "assistant"

        content: str = (
            message.content
            if hasattr(message, "content") and isinstance(message.content, str)
            else str(message)
        )
        if len(content) > 2000:
            content = content[:2000] + "…[truncated]"
        self._store.save_message(self._session_id, role, content)

    def add_messages(self, messages: Any) -> None:
        for m in messages:
            self.add_message(m)

    def clear(self) -> None:
        """Interface stub — history is retained in the DB."""
        pass


def _make_history_factory(
    chat_store: Any,
) -> Callable[[str], SQLAlchemyChatMessageHistory]:
    """
    Return a session_id → SQLAlchemyChatMessageHistory factory.
    Passed to RunnableWithMessageHistory as get_session_history.
    """
    def get_session_history(session_id: str) -> SQLAlchemyChatMessageHistory:
        return SQLAlchemyChatMessageHistory(session_id, chat_store)
    return get_session_history
