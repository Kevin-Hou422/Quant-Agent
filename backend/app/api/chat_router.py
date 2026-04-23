"""
chat_router.py — 对话式 Quant Agent HTTP API (Phase 4)

端点：
  POST /api/chat                          — 发送消息，返回 Agent 回复 + DSL + 绩效指标
  POST /api/chat/sessions                 — 创建新 ChatSession
  GET  /api/chat/sessions                 — 列出所有会话（摘要，不含消息体）
  GET  /api/chat/sessions/{session_id}    — 获取指定会话及其完整消息历史

设计（Phase 4 升级）：
  - 全局单例 QuantAgent：构造时注入 ChatStore，跨 session 共享
  - 会话隔离：agent.chat(message, session_id=...) 路由到正确的历史记录
  - LangChain 路径：RunnableWithMessageHistory 通过 SQLAlchemyChatMessageHistory 自动持久化
  - Fallback 路径：QuantAgent._fallback_chat 内部直接调用 chat_store.save_message
  - 路由层不再重复调用 save_message（防止双写）
  - 无 OPENAI_API_KEY 时自动降级（不崩溃）
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue as _queue
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.dependencies import get_chat_store
from app.db.chat_store import ChatStore

logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/api/chat", tags=["Chat Agent"])


# ---------------------------------------------------------------------------
# 全局单例 QuantAgent（Phase 4：注入 ChatStore，跨 session 共享）
# ---------------------------------------------------------------------------

_agent: Optional[Any] = None


def _get_agent(store: ChatStore) -> Any:
    """
    惰性初始化全局 QuantAgent 单例。
    首次调用时将 ChatStore 注入，后续直接复用同一实例。
    """
    global _agent
    if _agent is None:
        from app.agent.quant_agent import QuantAgent
        _agent = QuantAgent(
            n_tickers  = 20,
            n_days     = 252,
            oos_ratio  = 0.30,
            n_trials   = 10,
            chat_store = store,
        )
        logger.info("全局 QuantAgent 单例已创建（ChatStore 已注入）")
    return _agent


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


class UpdateSessionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=256, description="新会话标题")


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
# PATCH /api/chat/sessions/{session_id} — 重命名会话
# ---------------------------------------------------------------------------

@chat_router.patch("/sessions/{session_id}", response_model=SessionSummary)
def rename_session(
    session_id: str,
    req:        UpdateSessionRequest,
    store:      ChatStore = Depends(get_chat_store),
) -> SessionSummary:
    """重命名指定会话标题。"""
    ok = store.update_session_title(session_id, req.title)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' 不存在")
    sess = store.get_session(session_id)
    return SessionSummary(
        session_id = sess.id,
        title      = sess.title,
        created_at = _dt_str(sess.created_at),
    )


# ---------------------------------------------------------------------------
# DELETE /api/chat/sessions/{session_id} — 删除会话
# ---------------------------------------------------------------------------

@chat_router.delete("/sessions/{session_id}", status_code=204)
def delete_session(
    session_id: str,
    store:      ChatStore = Depends(get_chat_store),
) -> None:
    """删除指定会话及其所有消息。"""
    ok = store.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' 不存在")


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

    - LangChain 路径：RunnableWithMessageHistory 通过 SQLAlchemyChatMessageHistory
      自动持久化 user + assistant 消息（不重复写入）
    - Fallback 路径：QuantAgent._fallback_chat 内部调用 chat_store.save_message
    - 路由层不再手动调用 save_message，防止双写
    - 有 OPENAI_API_KEY：LangChain AgentExecutor（Workflow A/B 全功能）
    - 无 API Key：FallbackOrchestrator（纯 Python，关键词映射 + 结构变异）
    - 跨请求记忆：session_id 相同则保持对话上下文（DB 持久化，重启不丢失）
    """
    agent = _get_agent(store)
    try:
        result = agent.chat(req.message, session_id=req.session_id)
    except Exception as exc:
        logger.exception("agent.chat 异常 session=%s", req.session_id)
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        session_id = req.session_id,
        reply      = result.get("reply", ""),
        dsl        = result.get("dsl"),
        metrics    = result.get("metrics"),
    )


# ---------------------------------------------------------------------------
# POST /api/chat/stream — 流式对话（SSE）
# ---------------------------------------------------------------------------

@chat_router.post("/stream")
async def chat_stream(
    req:   ChatRequest,
    store: ChatStore = Depends(get_chat_store),
) -> StreamingResponse:
    """
    Streaming SSE version of /api/chat.

    Events (NDJSON over text/event-stream):
      {"type": "text",  "text": "..."}      — incremental progress / thinking line
      {"type": "ping"}                       — keep-alive
      {"type": "done",  "result": {...}}     — final {reply, dsl, metrics}
      {"type": "error", "message": "..."}    — terminal error
    """
    agent        = _get_agent(store)
    event_queue: _queue.Queue = _queue.Queue()

    def _on_event(event: dict) -> None:
        event_queue.put(event)

    def _run() -> None:
        try:
            agent.stream_chat(req.message, session_id=req.session_id, on_event=_on_event)
        except Exception as exc:
            logger.exception("chat_stream worker failed session=%s", req.session_id)
            event_queue.put({"type": "error", "message": str(exc)})

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    loop = asyncio.get_event_loop()

    async def _generate():
        while True:
            try:
                event = await loop.run_in_executor(
                    None, lambda: event_queue.get(timeout=300)
                )
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
            except _queue.Empty:
                yield 'data: {"type":"ping"}\n\n'

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )
