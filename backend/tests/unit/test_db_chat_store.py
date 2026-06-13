"""
test_db_chat_store.py — ChatStore 会话管理测试（内存 SQLite）
"""
from __future__ import annotations

import pytest

from app.db.chat_store import ChatStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_chat.db"
    return ChatStore(db_url=f"sqlite:///{db_path}")


class TestSessionCRUD:

    def test_create_session_returns_session(self, store):
        sess = store.create_session("Test Session")
        assert sess is not None
        assert sess.id is not None and len(sess.id) > 0
        assert sess.title == "Test Session"

    def test_create_session_default_title(self, store):
        sess = store.create_session()
        assert sess.title == "New Session"

    def test_list_sessions_returns_created(self, store):
        store.create_session("Session A")
        store.create_session("Session B")
        sessions = store.list_sessions()
        assert len(sessions) == 2

    def test_list_sessions_sorted_by_created_desc(self, store):
        s1 = store.create_session("First")
        s2 = store.create_session("Second")
        sessions = store.list_sessions()
        # 最新创建的排在前面
        assert sessions[0].id == s2.id

    def test_get_session_found(self, store):
        sess = store.create_session("Find Me")
        fetched = store.get_session(sess.id)
        assert fetched is not None
        assert fetched.title == "Find Me"

    def test_get_session_not_found_returns_none(self, store):
        result = store.get_session("nonexistent-uuid-12345")
        assert result is None

    def test_update_title_succeeds(self, store):
        sess = store.create_session("Old Title")
        success = store.update_session_title(sess.id, "New Title")
        assert success is True
        updated = store.get_session(sess.id)
        assert updated.title == "New Title"

    def test_update_title_nonexistent_returns_false(self, store):
        result = store.update_session_title("bad-id", "title")
        assert result is False

    def test_delete_session_returns_true(self, store):
        sess = store.create_session("To Delete")
        result = store.delete_session(sess.id)
        assert result is True

    def test_delete_session_removed_from_list(self, store):
        sess = store.create_session("Gone")
        store.delete_session(sess.id)
        sessions = store.list_sessions()
        ids = [s.id for s in sessions]
        assert sess.id not in ids

    def test_delete_nonexistent_returns_false(self, store):
        result = store.delete_session("does-not-exist")
        assert result is False


class TestMessageCRUD:

    def test_save_message_persisted(self, store):
        sess = store.create_session()
        msg = store.save_message(sess.id, "user", "Hello")
        assert msg.id is not None
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_get_history_returns_messages_in_order(self, store):
        sess = store.create_session()
        store.save_message(sess.id, "user",      "First message")
        store.save_message(sess.id, "assistant", "Second reply")
        store.save_message(sess.id, "user",      "Third")
        history = store.get_history(sess.id)
        assert len(history) == 3
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    def test_get_history_empty_session(self, store):
        sess = store.create_session()
        history = store.get_history(sess.id)
        assert history == []

    def test_delete_session_removes_messages_cascade(self, store):
        """删除会话后，消息应被级联删除。"""
        sess = store.create_session()
        store.save_message(sess.id, "user", "Will be deleted")
        store.delete_session(sess.id)
        history = store.get_history(sess.id)
        assert history == []


class TestEnsureSession:

    def test_ensure_creates_new_if_not_exists(self, store):
        new_id = "fixed-uuid-for-test-123"
        sess = store.ensure_session(new_id, "Auto Created")
        assert sess.id == new_id

    def test_ensure_returns_existing_if_exists(self, store):
        sess1 = store.create_session("Existing")
        sess2 = store.ensure_session(sess1.id, "Should Not Change")
        assert sess2.id == sess1.id
        # 不应修改标题
        assert sess2.title == "Existing"
