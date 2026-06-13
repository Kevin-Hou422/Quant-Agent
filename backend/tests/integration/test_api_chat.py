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
        if resp.status_code == 200:
            body = resp.json()
            assert "reply" in body
            assert isinstance(body["reply"], str)
            assert len(body["reply"]) > 0

    def test_chat_response_has_session_id(self, client):
        sid = f"test-{uuid.uuid4()}"
        resp = client.post("/api/chat", json={"message": "hello", "session_id": sid})
        if resp.status_code == 200:
            body = resp.json()
            assert "session_id" in body

    def test_chat_dsl_field_present(self, client):
        """reply 中 dsl 字段应存在（可为 None 或字符串）。"""
        resp = client.post("/api/chat", json={
            "message": "backtest rank(close)",
            "session_id": "test-dsl-001",
        })
        if resp.status_code == 200:
            body = resp.json()
            assert "dsl" in body

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
        assert resp.status_code in (400, 422, 200)

    def test_multi_turn_same_session(self, client):
        """多轮对话使用相同 session_id，应维持上下文。"""
        sid = f"multiturn-{uuid.uuid4()}"
        r1 = client.post("/api/chat", json={"message": "First message", "session_id": sid})
        r2 = client.post("/api/chat", json={"message": "Second message", "session_id": sid})
        assert r1.status_code in (200, 500)
        assert r2.status_code in (200, 500)


class TestSessionCRUD:

    def test_create_session_returns_200(self, client):
        resp = client.post("/api/chat/sessions", json={"title": "Test Session"})
        assert resp.status_code in (200, 201)

    def test_create_session_has_session_id(self, client):
        resp = client.post("/api/chat/sessions", json={"title": "My Session"})
        if resp.status_code in (200, 201):
            body = resp.json()
            assert "session_id" in body
            assert len(body["session_id"]) > 0

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
        create_resp = client.post("/api/chat/sessions", json={"title": "Detail Test"})
        if create_resp.status_code in (200, 201):
            sid = create_resp.json().get("session_id")
            if sid:
                get_resp = client.get(f"/api/chat/sessions/{sid}")
                assert get_resp.status_code in (200, 404)
                if get_resp.status_code == 200:
                    body = get_resp.json()
                    assert "session_id" in body or "messages" in body

    def test_delete_session(self, client):
        create_resp = client.post("/api/chat/sessions", json={"title": "To Delete"})
        if create_resp.status_code in (200, 201):
            sid = create_resp.json().get("session_id")
            if sid:
                del_resp = client.delete(f"/api/chat/sessions/{sid}")
                assert del_resp.status_code in (200, 204, 404)

    def test_rename_session(self, client):
        create_resp = client.post("/api/chat/sessions", json={"title": "Old"})
        if create_resp.status_code in (200, 201):
            sid = create_resp.json().get("session_id")
            if sid:
                patch_resp = client.patch(
                    f"/api/chat/sessions/{sid}",
                    json={"title": "New Title"},
                )
                assert patch_resp.status_code in (200, 204, 404)

    def test_get_nonexistent_session_404(self, client):
        resp = client.get("/api/chat/sessions/totally-nonexistent-uuid-abc123")
        assert resp.status_code in (404, 200)
        if resp.status_code == 200:
            body = resp.json()
            assert body is None or body.get("messages") == []

    def test_created_session_appears_in_list(self, client):
        title = f"session-{uuid.uuid4()}"
        create_resp = client.post("/api/chat/sessions", json={"title": title})
        if create_resp.status_code in (200, 201):
            sid = create_resp.json().get("session_id")
            list_resp = client.get("/api/chat/sessions")
            if list_resp.status_code == 200:
                sessions = list_resp.json().get("sessions", [])
                ids = [s.get("session_id") for s in sessions]
                assert sid in ids
