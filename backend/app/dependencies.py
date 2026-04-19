"""
dependencies.py — FastAPI 依赖注入

通过 Depends() 注入常用服务实例，保证单例复用。
"""

from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.db.alpha_store import AlphaStore
from app.db.chat_store import ChatStore


@lru_cache(maxsize=1)
def _get_store_singleton() -> AlphaStore:
    return AlphaStore(db_url=settings.database_url)


def get_store() -> AlphaStore:
    """FastAPI Depends 注入 AlphaStore 单例。"""
    return _get_store_singleton()


@lru_cache(maxsize=1)
def _get_chat_store_singleton() -> ChatStore:
    return ChatStore(db_url=settings.database_url)


def get_chat_store() -> ChatStore:
    """FastAPI Depends 注入 ChatStore 单例（与 AlphaStore 共用同一 SQLite 文件）。"""
    return _get_chat_store_singleton()
