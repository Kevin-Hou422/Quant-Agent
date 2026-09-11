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


@pytest.fixture
def tick(monkeypatch):
    """
    把 `chat_store` 里的时钟换成"每次读取前进一秒"的确定时钟。

    为什么必须注入：`create_session` / `save_message` 都用
    `datetime.utcnow()` 记 `created_at`，而 **Windows 上 `utcnow()` 的分辨率
    是约 15.6 ms**（实测连续两次调用返回完全相同的值）。连续创建的两条记录
    很可能落在同一个 tick 上 → `created_at` 相同 → `ORDER BY created_at`
    的结果由 SQLite 自行决定，**与插入顺序无关**。

    这正是 `test_list_sessions_sorted_by_created_desc` 在全量套件里
    随机变红的原因：单独跑时两次创建之间的 DB flush 恰好跨过一个 tick，
    机器忙的时候就跨不过去。断言本身没错，错在它依赖了一个不受控的量。

    顺带登记为产品问题（见 MUTATION_LEDGER「B-12」）：
    `list_sessions()` / `get_history()` 的 ORDER BY **没有第二排序键**，
    同一毫秒内创建的会话在界面上的顺序是不稳定的。
    """
    import app.db.chat_store as mod
    from datetime import datetime as _dt, timedelta

    state = {"n": 0}
    base = _dt(2026, 1, 1, 0, 0, 0)

    class _Clock(_dt):
        @classmethod
        def utcnow(cls):
            state["n"] += 1
            return base + timedelta(seconds=state["n"])

    monkeypatch.setattr(mod, "datetime", _Clock)
    return state


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

    def test_list_sessions_sorted_by_created_desc(self, store, tick):
        """
        最新创建的排在最前。用注入时钟保证三条记录的 `created_at`
        确实互不相同 —— 否则这条断言测的是 SQLite 对并列值的任意裁决
        （旧版就是这么在全量套件里随机变红的，见 `tick` fixture 的说明）。
        """
        s1 = store.create_session("First")
        s2 = store.create_session("Second")
        s3 = store.create_session("Third")
        sessions = store.list_sessions()
        assert [s.id for s in sessions] == [s3.id, s2.id, s1.id], (
            f"会话没有按创建时间倒序排列："
            f"{[s.title for s in sessions]}")
        assert sessions[0].created_at > sessions[-1].created_at, (
            "构造的时间戳没有区分开 —— 这条断言失去了区分力")

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

    def test_get_history_returns_messages_in_order(self, store, tick):
        """
        历史按 `created_at` **升序**。同样需要注入时钟：三条消息连着存，
        在 Windows 的 15.6 ms 时钟分辨率下极可能同时刻，
        那时这条断言测的是 SQLite 对并列值的任意裁决，而不是排序契约。
        """
        sess = store.create_session()
        store.save_message(sess.id, "user",      "First message")
        store.save_message(sess.id, "assistant", "Second reply")
        store.save_message(sess.id, "user",      "Third")
        history = store.get_history(sess.id)
        assert [m.content for m in history] == [
            "First message", "Second reply", "Third"], (
            f"历史消息顺序不对：{[m.content for m in history]}")
        assert [m.role for m in history] == ["user", "assistant", "user"]
        assert history[0].created_at < history[-1].created_at, (
            "构造的时间戳没有区分开 —— 这条断言失去了区分力")

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
