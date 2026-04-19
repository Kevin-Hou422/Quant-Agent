"""
chat_router.py — 对话式 Quant Agent HTTP API (Phase 3)

端点：
  POST /api/chat                          — 发送消息，返回 Agent 回复 + DSL + 绩效指标
  POST /api/chat/sessions                 — 创建新 ChatSession
  GET  /api/chat/sessions                 — 列出所有会话（摘要，不含消息体）
  GET  /api/chat/sessions/{session_id}    — 获取指定会话及其完整消息历史

设计：
  - 每个 session_id 映射到独立的 QuantAgent 实例（含 ConversationMemory）
  - 内存层：最多保留 MAX_SESSIONS 个活跃 QuantAgent，LRU 淘汰（纯内存，重启清空）
  - 持久层：每轮 user/assistant 消息写入 SQLite chat_messages 表，跨重启可读取历史
  - 新 session 通过 POST /api/chat/sessions 创建，或在 POST /api/chat 时自动幂等创建
  - 无 OPENAI_API_KEY 时自动降级（不崩溃）
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_chat_store
from app.db.chat_store import ChatStore

logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/api/chat", tags=["Chat Agent"])


# ---------------------------------------------------------------------------
# QuantAgent 内存 Session 管理（LRU）
# ---------------------------------------------------------------------------

MAX_SESSIONS = 50


class _AgentStore:
    """LRU 内存缓存：session_id → QuantAgent 实例。"""

    def __init__(self, max_size: int = MAX_SESSIONS) -> None:
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._atime: Dict[str, float] = {}
        self._max = max_size

    def get(self, session_id: str) -> Optional[Any]:
        if session_id in self._store:
            self._store.move_to_end(session_id)
            self._atime[session_id] = time.time()
            return self._store[session_id]
        return None

    def set(self, session_id: str, agent: Any) -> None:
        if session_id in self._store:
            self._store.move_to_end(session_id)
        else:
            if len(self._store) >= self._max:
                oldest = next(iter(self._store))
                self._store.pop(oldest)
                self._atime.pop(oldest, None)
            self._store[session_id] = agent
        self._atime[session_id] = time.time()

    def list_ids(self) -> List[str]:
        return list(self._store.keys())


_agents = _AgentStore()


def _get_or_create_agent(session_id: str) -> Any:
    agent = _agents.get(session_id)
    if agent is None:
        from app.agent.quant_agent import QuantAgent
        agent = QuantAgent(
            n_tickers=20,
            n_days=252,
            oos_ratio=0.30,
            n_trials=10,
        )
        _agents.set(session_id, agent)
        logger.info("新建 QuantAgent session: %s", session_id)
    return agent


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    title: str = Field("New Session", max_length=256, description="会话标题")


class SessionSummary(BaseModel):
    session_id: str
    title:      str
    created_at: str


class MessageOut(BaseModel):
    id:         int
    role:       str
    content:    str
    created_at: str


class SessionDetailResponse(BaseModel):
    session_id: str
    title:      str
    created_at: str
    messages:   List[MessageOut]


class SessionListResponse(BaseModel):
    sessions: List[SessionSummary]
    count:    int


class ChatRequest(BaseModel):
    message:    str = Field(..., min_length=1, max_length=2000, description="用户消息")
    session_id: str = Field("default", description="会话 ID（用于跨轮记忆和消息持久化）")


class ChatResponse(BaseModel):
    session_id: str
    reply:      str
    dsl:        Optional[str]
    metrics:    Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Helper: datetime → ISO string
# ---------------------------------------------------------------------------

def _dt_str(dt) -> str:
    return str(dt) if dt else ""


# ---------------------------------------------------------------------------
# POST /api/chat/sessions — 创建新会话
# ---------------------------------------------------------------------------

@chat_router.post("/sessions", response_model=SessionSummary, status_code=201)
def create_session(
    req:   CreateSessionRequest,
    store: ChatStore = Depends(get_chat_store),
) -> SessionSummary:
    """
    创建一个新的 ChatSession。
    返回 session_id（UUID）、标题和创建时间。
    """
    sess = store.create_session(title=req.title)
    return SessionSummary(
        session_id = sess.id,
        title      = sess.title,
        created_at = _dt_str(sess.created_at),
    )


# ---------------------------------------------------------------------------
# GET /api/chat/sessions — 列出所有会话
# ---------------------------------------------------------------------------

@chat_router.get("/sessions", response_model=SessionListResponse)
def list_sessions(
    store: ChatStore = Depends(get_chat_store),
) -> SessionListResponse:
    """
    列出所有 ChatSession（按创建时间倒序），不含消息体。
    """
    sessions = store.list_sessions(limit=200)
    items = [
        SessionSummary(
            session_id = s.id,
            title      = s.title,
            created_at = _dt_str(s.created_at),
        )
        for s in sessions
    ]
    return SessionListResponse(sessions=items, count=len(items))


# ---------------------------------------------------------------------------
# GET /api/chat/sessions/{session_id} — 获取会话 + 完整历史
# ---------------------------------------------------------------------------

@chat_router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(
    session_id: str,
    store:      ChatStore = Depends(get_chat_store),
) -> SessionDetailResponse:
    """
    返回指定会话的元信息及其按时间排序的完整消息历史。
    """
    sess = store.get_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' 不存在")

    messages = store.get_history(session_id)
    msg_items = [
        MessageOut(
            id         = m.id,
            role       = m.role,
            content    = m.content,
            created_at = _dt_str(m.created_at),
        )
        for m in messages
    ]
    return SessionDetailResponse(
        session_id = sess.id,
        title      = sess.title,
        created_at = _dt_str(sess.created_at),
        messages   = msg_items,
    )


# ---------------------------------------------------------------------------
# POST /api/chat — 发送消息
# ---------------------------------------------------------------------------

@chat_router.post("", response_model=ChatResponse)
def chat(
    req:   ChatRequest,
    store: ChatStore = Depends(get_chat_store),
) -> ChatResponse:
    """
    向 Quant Agent 发送一条消息。

    - 若 session_id 对应的 ChatSession 不存在，自动创建（幂等）。
    - user 消息和 assistant 回复均持久化到 chat_messages 表。
    - 有 OPENAI_API_KEY：LangChain AgentExecutor（Workflow A/B 全功能）
    - 无 API Key：FallbackOrchestrator（纯 Python，关键词映射）
    - 跨请求记忆：session_id 相同则保持对话上下文
    """
    # 1. 确保 ChatSession 存在（幂等）
    store.ensure_session(req.session_id)

    # 2. 持久化 user 消息
    store.save_message(
        session_id = req.session_id,
        role       = "user",
        content    = req.message,
    )

    # 3. 运行 Agent
    agent = _get_or_create_agent(req.session_id)
    try:
        result = agent.chat(req.message)
    except Exception as exc:
        logger.exception("agent.chat 异常 session=%s", req.session_id)
        raise HTTPException(status_code=500, detail=str(exc))

    reply = result.get("reply", "")

    # 4. 持久化 assistant 回复
    store.save_message(
        session_id = req.session_id,
        role       = "assistant",
        content    = reply,
    )

    return ChatResponse(
        session_id = req.session_id,
        reply      = reply,
        dsl        = result.get("dsl"),
        metrics    = result.get("metrics"),
    )
