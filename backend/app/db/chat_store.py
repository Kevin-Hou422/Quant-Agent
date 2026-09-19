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
    create_engine, func, select,
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
    #: 【缺陷 B-12，2026-09-20 修】单调递增的**插入序**。
    #:
    #: `created_at` 的分辨率是秒，同一秒创建的多个会话在 `ORDER BY created_at DESC`
    #: 下顺序不确定，实测会把最老的排最前（与"最近的在前"这个契约相反）。
    #: 主键 `id` 是 **UUID 字符串**，按它排序是随机序、不是插入序，当不了第二排序键
    #: —— 所以另开一列自增序号。
    seq        = Column(Integer, index=True, default=0)

    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        # 【缺陷 B-12，2026-09-20 修】只按 created_at 排序**没有稳定的第二排序键**。
        # `created_at` 的分辨率是秒（datetime.utcnow()），同一秒内插入的多条消息
        # 顺序由数据库返回顺序决定 —— 实测会出现**倒序**：一问一答被显示成
        # 先答后问。自增主键 `id` 是天然的插入序，作第二键。
        order_by="ChatMessage.created_at, ChatMessage.id",
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
        # Task 6.2：统一默认库路径来源（与 AlphaStore 一致）
        if db_url is None:
            db_url = os.getenv("DATABASE_URL", "")
            if not db_url:
                try:
                    from app.config import settings
                    db_url = settings.database_url
                except Exception:
                    db_url = "sqlite:///alphas.db"
        url = db_url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self._engine  = create_engine(url, connect_args=connect_args, echo=False)
        from ._sqlite_utils import harden_sqlite_engine
        harden_sqlite_engine(self._engine)          # Task 6.2：WAL + busy_timeout
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
            # `seq` 在**同一个事务里**取 max+1（缺陷 B-12）。
            # `autoincrement=True` 只对整型主键生效，这里主键是 UUID 字符串，
            # 所以自己算。SQLite 写事务是串行的，同一事务内 max+1 不会撞号；
            # 换成并发写的后端时应改为数据库序列（Sequence/IDENTITY）。
            nxt = db.execute(select(func.coalesce(func.max(ChatSession.seq), 0))).scalar()
            sess.seq = int(nxt or 0) + 1
            db.add(sess)
            db.commit()
        return sess

    def list_sessions(self, limit: int = 100) -> List[ChatSession]:
        """按创建时间倒序列出最近 ``limit`` 个会话（不含消息详情）。"""
        with self._Session() as db:
            stmt = (
                select(ChatSession)
                # 第二排序键：同一秒创建的会话必须有确定顺序（缺陷 B-12）。
                # 倒序列表里 id 也要倒序，才与 created_at 的方向一致。
                .order_by(ChatSession.created_at.desc(), ChatSession.seq.desc())
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
                # 第二排序键：同一秒内的消息按插入序（缺陷 B-12）。
                .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
            )
            return list(db.scalars(stmt))

    def update_session_title(self, session_id: str, title: str) -> bool:
        """更新会话标题，返回 True 表示成功，False 表示会话不存在。"""
        with self._Session() as db:
            sess = db.get(ChatSession, session_id)
            if sess is None:
                return False
            sess.title = title[:256]
            db.commit()
            return True

    def delete_session(self, session_id: str) -> bool:
        """删除会话及其所有消息（CASCADE），返回 True 表示成功。"""
        with self._Session() as db:
            sess = db.get(ChatSession, session_id)
            if sess is None:
                return False
            db.delete(sess)
            db.commit()
            return True
