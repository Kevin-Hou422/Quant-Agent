"""
chat_store.py — Chat Session 持久化层（SQLAlchemy ORM）

表结构：
  chat_sessions
    id         : UUID (str primary key)
    title      : 会话标题
    created_at : 创建时间

  chat_messages
    id         : 自增整数主键
    session_id : ForeignKey → chat_sessions.id (CASCADE DELETE)
    role       : "user" | "assistant"
    content    : 消息正文
    created_at : 发送时间

DATABASE_URL 环境变量配置，默认 sqlite:///alphas.db（与 AlphaStore 共用同一文件）。

注意：使用独立的 DeclarativeBase，与 alpha_store.py 的 _Base 隔离，
两者在 create_all() 时互不干扰。
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column, DateTime, ForeignKey, Integer, String, Text,
    create_engine, select,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker


# ---------------------------------------------------------------------------
# ORM Base（独立，不与 alpha_store._Base 混用）
# ---------------------------------------------------------------------------

class _ChatBase(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ChatSession(_ChatBase):
    """一个研究会话（对应前端的一个独立对话窗口）。"""
    __tablename__      = "chat_sessions"
    __allow_unmapped__ = True   # 允许旧式 Column() 声明与 DeclarativeBase 共存

    id         = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title      = Column(String(256), nullable=False, default="New Session")
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


class ChatMessage(_ChatBase):
    """会话内的单条消息（user 或 assistant）。"""
    __tablename__      = "chat_messages"
    __allow_unmapped__ = True

    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(36),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role       = Column(String(16), nullable=False)   # "user" | "assistant"
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


# ---------------------------------------------------------------------------
# ChatStore — CRUD 接口
# ---------------------------------------------------------------------------

class ChatStore:
    """
    Chat Session 持久化服务。

    Parameters
    ----------
    db_url : SQLAlchemy 数据库 URL。
             默认读取环境变量 DATABASE_URL，否则 sqlite:///alphas.db。
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        url = db_url or os.getenv("DATABASE_URL", "sqlite:///alphas.db")
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self._engine  = create_engine(url, connect_args=connect_args, echo=False)
        _ChatBase.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def create_session(self, title: str = "New Session") -> ChatSession:
        """创建新会话，返回持久化后的 ChatSession 对象。"""
        sess = ChatSession(
            id         = str(uuid.uuid4()),
            title      = title,
            created_at = datetime.utcnow(),
        )
        with self._Session() as db:
            db.add(sess)
            db.commit()
        return sess

    def list_sessions(self, limit: int = 100) -> List[ChatSession]:
        """按创建时间倒序列出最近 ``limit`` 个会话（不含消息详情）。"""
        with self._Session() as db:
            stmt = (
                select(ChatSession)
                .order_by(ChatSession.created_at.desc())
                .limit(limit)
            )
            return list(db.scalars(stmt))

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """按 id 查询会话，不存在返回 None。"""
        with self._Session() as db:
            return db.get(ChatSession, session_id)

    def ensure_session(self, session_id: str, title: str = "New Session") -> ChatSession:
        """
        若 ``session_id`` 对应的会话不存在则自动创建（幂等）。
        用于 /api/chat 端点：前端可传入任意 UUID，后端自动初始化。
        """
        with self._Session() as db:
            sess = db.get(ChatSession, session_id)
            if sess is None:
                sess = ChatSession(
                    id         = session_id,
                    title      = title,
                    created_at = datetime.utcnow(),
                )
                db.add(sess)
                db.commit()
            return sess

    # ------------------------------------------------------------------
    # Message CRUD
    # ------------------------------------------------------------------

    def save_message(
        self,
        session_id: str,
        role:       str,
        content:    str,
    ) -> ChatMessage:
        """
        向指定会话追加一条消息。

        Parameters
        ----------
        session_id : 会话 id（必须已存在，否则外键约束报错）
        role       : "user" | "assistant"
        content    : 消息正文
        """
        msg = ChatMessage(
            session_id = session_id,
            role       = role,
            content    = content,
            created_at = datetime.utcnow(),
        )
        with self._Session() as db:
            db.add(msg)
            db.commit()
        return msg

    def get_history(self, session_id: str) -> List[ChatMessage]:
        """按时间正序返回指定会话的全部消息。"""
        with self._Session() as db:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
            )
            return list(db.scalars(stmt))
