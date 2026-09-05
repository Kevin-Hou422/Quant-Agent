"""
test_api_chat.py — Chat API 端点集成测试（无 LLM，Fallback 模式）

覆盖：POST /api/chat, POST /api/chat/sessions,
      GET /api/chat/sessions, GET /api/chat/sessions/{id},
      PATCH /api/chat/sessions/{id}, DELETE /api/chat/sessions/{id}
"""
from __future__ import annotations

import uuid
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _ok(resp, expect: int = 200):
    """DEV_LESSONS §S：断言具体状态码，不容忍 5xx，不用 if 包住断言。"""
    assert resp.status_code == expect, (
        f"期望 {expect} 实际 {resp.status_code}｜body={resp.text[:600]}"
    )
    return resp.json()


class TestChatEndpoint:

    def test_chat_basic_returns_200(self, client):
        resp = client.post("/api/chat", json={
            "message": "Generate a momentum alpha",
            "session_id": "test-session-001",
        })
        assert resp.status_code == 200

    def test_chat_response_has_reply(self, client):
        resp = client.post("/api/chat", json={
            "message": "Generate momentum alpha",
            "session_id": "test-session-reply",
        })
        body = _ok(resp)
        assert isinstance(body.get("reply"), str) and body["reply"], f"reply 为空：{body!r}"

    def test_chat_response_has_session_id(self, client):
        sid = f"test-{uuid.uuid4()}"
        resp = client.post("/api/chat", json={"message": "hello", "session_id": sid})
        body = _ok(resp)
        assert body.get("session_id") == sid, f"回显 session_id 不符：{body.get('session_id')!r}"

    def test_chat_dsl_field_present(self, client):
        """reply 中 dsl 字段应存在（可为 None 或字符串）。"""
        resp = client.post("/api/chat", json={
            "message": "backtest rank(close)",
            "session_id": "test-dsl-001",
        })
        body = _ok(resp)
        assert "dsl" in body, f"缺 dsl 字段：{sorted(body)}"
        # 数据契约（Phase A）：任何带指标的回复都必须自报来源
        assert "data_source" in body, f"缺 data_source：{sorted(body)}"

    def test_chat_no_api_key_fallback_200(self, client):
        """无 OPENAI_API_KEY 时应降级到 fallback 模式，仍返回 200。"""
        import os
        original = os.environ.get("OPENAI_API_KEY")
        try:
            os.environ["OPENAI_API_KEY"] = ""
            resp = client.post("/api/chat", json={
                "message": "test fallback mode",
                "session_id": "fallback-test",
            })
            assert resp.status_code == 200
        finally:
            if original:
                os.environ["OPENAI_API_KEY"] = original
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

    def test_chat_empty_message_rejected(self, client):
        resp = client.post("/api/chat", json={"message": "", "session_id": "test"})
        assert resp.status_code == 422, (
            f"空消息必须被 Pydantic min_length 拒绝，实际 {resp.status_code}：{resp.text[:300]}"
        )

    def test_multi_turn_same_session(self, client):
        """多轮对话使用相同 session_id，应维持上下文。"""
        sid = f"multiturn-{uuid.uuid4()}"
        r1 = client.post("/api/chat", json={"message": "First message", "session_id": sid})
        r2 = client.post("/api/chat", json={"message": "Second message", "session_id": sid})
        _ok(r1); _ok(r2)
        # 上下文必须真的落库：两轮消息都应出现在该会话历史里
        hist = _ok(client.get(f"/api/chat/sessions/{sid}"))
        contents = [m["content"] for m in hist.get("messages", [])]
        assert "First message" in contents and "Second message" in contents, (
            f"多轮消息未持久化到同一 session：{contents}"
        )


class TestSessionCRUD:

    def test_create_session_returns_200(self, client):
        resp = client.post("/api/chat/sessions", json={"title": "Test Session"})
        assert resp.status_code == 201, f"创建会话应返回 201，实际 {resp.status_code}"

    def test_create_session_has_session_id(self, client):
        body = _ok(client.post("/api/chat/sessions", json={"title": "My Session"}), 201)
        assert body.get("session_id"), f"session_id 为空：{body!r}"

    def test_list_sessions_returns_200(self, client):
        resp = client.get("/api/chat/sessions")
        assert resp.status_code == 200

    def test_list_sessions_has_sessions_field(self, client):
        resp = client.get("/api/chat/sessions")
        body = resp.json()
        assert "sessions" in body
        assert isinstance(body["sessions"], list)

    def test_get_session_detail(self, client):
        # 先创建一个会话
        sid = _ok(client.post("/api/chat/sessions", json={"title": "Detail Test"}), 201)["session_id"]
        body = _ok(client.get(f"/api/chat/sessions/{sid}"))
        assert body["session_id"] == sid
        assert isinstance(body.get("messages"), list)

    def test_delete_session(self, client):
        sid = _ok(client.post("/api/chat/sessions", json={"title": "To Delete"}), 201)["session_id"]
        del_resp = client.delete(f"/api/chat/sessions/{sid}")
        assert del_resp.status_code == 204, (
            f"删除应返回 204，实际 {del_resp.status_code}（404 说明删除根本没生效，"
            f"旧断言把 404 也算通过）"
        )
        # 删除必须真的生效
        assert client.get(f"/api/chat/sessions/{sid}").status_code == 404, "删除后仍能读到该会话"

    def test_rename_session(self, client):
        sid = _ok(client.post("/api/chat/sessions", json={"title": "Old"}), 201)["session_id"]
        body = _ok(client.patch(f"/api/chat/sessions/{sid}", json={"title": "New Title"}))
        assert body["title"] == "New Title", f"重命名未生效：{body}"
        assert _ok(client.get(f"/api/chat/sessions/{sid}"))["title"] == "New Title", "重命名未落库"

    def test_get_nonexistent_session_404(self, client):
        resp = client.get("/api/chat/sessions/totally-nonexistent-uuid-abc123")
        assert resp.status_code == 404, f"不存在的会话必须 404，实际 {resp.status_code}"

    def test_created_session_appears_in_list(self, client):
        title = f"session-{uuid.uuid4()}"
        sid = _ok(client.post("/api/chat/sessions", json={"title": title}), 201)["session_id"]
        sessions = _ok(client.get("/api/chat/sessions"))["sessions"]
        assert sid in [s["session_id"] for s in sessions], "新建会话未出现在列表中"
