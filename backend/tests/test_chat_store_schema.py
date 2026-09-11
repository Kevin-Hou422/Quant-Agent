"""
db/chat_store.py —— schema 与引擎配置的定钉测试（变异测试驱动）

来由：17 个变异点，首测击杀率 58.8%（存活 7）。存活项全在**声明层**：
`__allow_unmapped__`、三处 `nullable=False`、`index=True`、`echo=False`、
以及默认库路径的 `if not db_url`。

既有覆盖（`unit/test_db_chat_store.py`）只测 CRUD 行为，
对"表长什么样、引擎怎么建的"零断言 —— 而这两者决定：
  - 会话能不能存进没有标题、没有角色、没有内容的消息
  - 调度线程与 API 线程能不能共用同一个 sqlite 连接
  - 每一条 SQL 会不会被打进日志
"""
from __future__ import annotations

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from app.db.chat_store import ChatMessage, ChatSession, ChatStore


@pytest.fixture
def store(tmp_path) -> ChatStore:
    return ChatStore(db_url=f"sqlite:///{tmp_path/'chat.db'}")


# ===========================================================================
# A. 表结构
# ===========================================================================

class TestSchema:

    def test_required_columns_are_not_nullable(self, store):
        """
        `title` / `session_id` / `role` / `content` 允许为空后，
        会话列表会出现无标题项、消息会出现**没有角色或没有内容**的条目，
        而这些条目会被原样喂回 LLM 的对话历史。
        """
        insp = inspect(store._engine)
        expected = {
            "chat_sessions": ("title",),
            "chat_messages": ("session_id", "role", "content"),
        }
        for table, cols in expected.items():
            actual = {c["name"]: c for c in insp.get_columns(table)}
            for col in cols:
                assert actual[col]["nullable"] is False, f"{table}.{col} 允许为空"

    def test_session_id_is_indexed(self, store):
        insp = inspect(store._engine)
        indexed = {c for ix in insp.get_indexes("chat_messages")
                   for c in ix["column_names"]}
        assert "session_id" in indexed, "按会话拉取消息是最频繁的查询，却没有索引"

    def test_null_role_is_rejected_by_the_database(self, store):
        sid = store.create_session("t").id
        with store._Session() as s:
            s.add(ChatMessage(session_id=sid, role=None, content="x"))
            with pytest.raises(IntegrityError):
                s.commit()

    def test_null_content_is_rejected_by_the_database(self, store):
        sid = store.create_session("t").id
        with store._Session() as s:
            s.add(ChatMessage(session_id=sid, role="user", content=None))
            with pytest.raises(IntegrityError):
                s.commit()

    def test_legacy_column_style_is_allowed(self):
        """
        `__allow_unmapped__ = True` —— 本模块用旧式 `Column()` 声明配
        `DeclarativeBase`。改成 False 后**模块导入即失败**
        （SQLAlchemy 2.0 会对未标注的类属性报错）。
        两个模型都要检查：它们是两处独立的声明。
        """
        assert ChatSession.__allow_unmapped__ is True
        assert ChatMessage.__allow_unmapped__ is True

    def test_title_has_a_default(self, store):
        sess = store.create_session()
        assert sess.title == "New Session"
        rows = store.list_sessions()
        assert any(r.id == sess.id and r.title for r in rows), "新会话没有默认标题"

    def test_deleting_a_session_removes_its_messages(self, store):
        """`cascade="all, delete-orphan"`：删会话不得留下孤儿消息。"""
        sid = store.create_session("t").id
        store.save_message(sid, "user", "hello")
        store.delete_session(sid)
        with store._Session() as s:
            left = s.query(ChatMessage).filter(ChatMessage.session_id == sid).count()
        assert left == 0, f"删除会话后仍残留 {left} 条消息"


# ===========================================================================
# B. 引擎配置
# ===========================================================================

class TestEngineConfiguration:

    def test_engine_does_not_echo_sql(self, store):
        """`echo=False`：改成 True 会把每条 SQL 打进日志。"""
        assert store._engine.echo is False

    def test_sqlite_allows_cross_thread_use(self, tmp_path):
        """`check_same_thread: False` —— 调度线程写、API 线程读。"""
        import threading
        st = ChatStore(db_url=f"sqlite:///{tmp_path/'t.db'}")
        sid = st.create_session("t").id
        st.save_message(sid, "user", "hi")
        box = {}

        def _read():
            try:
                box["v"] = st.get_history(sid)
            except Exception as exc:      # noqa: BLE001
                box["err"] = exc

        th = threading.Thread(target=_read)
        th.start()
        th.join()
        assert "err" not in box, f"跨线程读取失败：{box.get('err')}"
        assert len(box["v"]) == 1

    def test_objects_stay_usable_after_commit(self, store):
        """`expire_on_commit=False`：返回的 ORM 对象在 session 关闭后仍可读。"""
        sid = store.create_session("title-x").id
        store.save_message(sid, "assistant", "answer")
        msgs = store.get_history(sid)
        assert msgs[0].role == "assistant"
        assert msgs[0].content == "answer"

    def test_env_database_url_is_used_when_no_arg(self, tmp_path, monkeypatch):
        """`if not db_url:` —— 删掉 `not` 会让环境变量与 settings 两个来源互换。"""
        target = f"sqlite:///{tmp_path/'env.db'}"
        monkeypatch.setenv("DATABASE_URL", target)
        assert str(ChatStore()._engine.url) == target

    def test_settings_url_is_used_when_env_is_empty(self, tmp_path, monkeypatch):
        import app.config
        monkeypatch.setenv("DATABASE_URL", "")
        target = f"sqlite:///{tmp_path/'cfg.db'}"
        monkeypatch.setattr(app.config.settings, "database_url", target, raising=False)
        assert str(ChatStore()._engine.url) == target


# ===========================================================================
# C. 消息顺序
# ===========================================================================

def test_messages_come_back_in_creation_order(store):
    """`order_by="ChatMessage.created_at"` —— 对话历史的顺序不能乱。"""
    sid = store.create_session("t").id
    for i in range(5):
        store.save_message(sid, "user" if i % 2 == 0 else "assistant", f"m{i}")
    assert [m.content for m in store.get_history(sid)] == [f"m{i}" for i in range(5)]
