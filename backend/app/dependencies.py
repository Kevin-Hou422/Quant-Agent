"""
dependencies.py — FastAPI 依赖注入

通过 Depends() 注入常用服务实例，保证单例复用。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Generator

from app.config import settings
from app.core.ml_engine.alpha_store import AlphaStore


@lru_cache(maxsize=1)
def _get_store_singleton() -> AlphaStore:
    return AlphaStore(db_url=settings.database_url)


def get_store() -> AlphaStore:
    """FastAPI Depends 注入 AlphaStore 单例。"""
    return _get_store_singleton()
