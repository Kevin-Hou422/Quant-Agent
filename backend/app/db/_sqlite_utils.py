"""
_sqlite_utils.py — SQLite 并发加固（Task 6.2）

统一为所有 SQLite 引擎开启：
  - WAL 日志模式：读写并发（读不阻塞写、写不阻塞读），避免调度线程与 API
    线程池并发写同一文件时的 "database is locked"（体检 E-N2）
  - busy_timeout：写锁竞争时最多等待 N 毫秒再报错，而非立即失败
  - synchronous=NORMAL：WAL 下的推荐值，兼顾安全与吞吐

通过 SQLAlchemy 的 connect 事件对**每个新连接**应用 PRAGMA（连接池会复用连接，
但新连接也需重新设置，故挂在事件上而非一次性执行）。
"""

from __future__ import annotations

import logging

from sqlalchemy import event
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

_BUSY_TIMEOUT_MS = 5000


def harden_sqlite_engine(engine: Engine) -> Engine:
    """对 SQLite 引擎应用 WAL + busy_timeout。非 SQLite 引擎原样返回。"""
    if engine.dialect.name != "sqlite":
        return engine

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _connection_record):  # noqa: ANN001
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()

    return engine
