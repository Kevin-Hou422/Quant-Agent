"""
chat_router.py — 对话式 Quant Agent HTTP API (Phase 3)

端点：
  POST /api/chat   — 发送消息，返回 Agent 回复 + DSL + 绩效指标
  GET  /api/chat/sessions — 列出活跃 session（仅 session_id 列表）

设计：
  - 每个 session_id 映射到独立的 QuantAgent 实例（含 ConversationMemory）
  - 最多保留 MAX_SESSIONS 个活跃 session，LRU 淘汰
  - 无 OPENAI_API_KEY 时自动降级（不崩溃）
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/api/chat", tags=["Chat Agent"])

# ---------------------------------------------------------------------------
# Session 管理（LRU）
# ---------------------------------------------------------------------------

MAX_SESSIONS = 50   # 最多并发 session 数

class _SessionStore:
    """简单 LRU Session 存储（无外部依赖）。"""

    def __init__(self, max_size: int = MAX_SESSIONS) -> None:
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._max   = max_size
        self._atime: Dict[str, float] = {}

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


_sessions = _SessionStore()


def _get_or_create_agent(session_id: str):
    agent = _sessions.get(session_id)
    if agent is None:
        from app.agent.quant_agent import QuantAgent
        agent = QuantAgent(
            n_tickers = 20,
            n_days    = 252,
            oos_ratio = 0.30,
            n_trials  = 10,   # API 模式少跑一点 trial，速度优先
        )
        _sessions.set(session_id, agent)
        logger.info("新建 QuantAgent session: %s", session_id)
    return agent


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message:    str  = Field(..., min_length=1, max_length=2000,
                              description="用户消息")
    session_id: str  = Field("default", description="会话 ID（用于跨轮记忆）")


class ChatResponse(BaseModel):
    session_id: str
    reply:      str
    dsl:        Optional[str]
    metrics:    Optional[Dict[str, Any]]


class SessionListResponse(BaseModel):
    sessions: List[str]
    count:    int


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------

@chat_router.post("", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    发送一条消息到 Quant Agent。

    - 有 OPENAI_API_KEY：LangChain AgentExecutor（Workflow A/B 全功能）
    - 无 API Key：FallbackOrchestrator（纯 Python，关键词映射）
    - 跨请求记忆：session_id 相同则保持对话上下文
    """
    agent = _get_or_create_agent(req.session_id)
    try:
        result = agent.chat(req.message)
    except Exception as exc:
        logger.exception("agent.chat 异常 session=%s", req.session_id)
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        session_id = req.session_id,
        reply      = result.get("reply", ""),
        dsl        = result.get("dsl"),
        metrics    = result.get("metrics"),
    )


@chat_router.get("/sessions", response_model=SessionListResponse)
def list_sessions() -> SessionListResponse:
    """列出当前活跃的 session ID 列表。"""
    ids = _sessions.list_ids()
    return SessionListResponse(sessions=ids, count=len(ids))
