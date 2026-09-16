"""
test_sqlite_hardening.py — `_sqlite_utils.harden_sqlite_engine` 的行为保护

来由：补齐变异器（工具缺陷 #11）之后，`app/db/_sqlite_utils.py` 才第一次
**进入**变异测试范围 —— 旧算子集里 `!=` 根本没有对应变异器，于是这个模块
一直显示"无变异点"，实际是**一条测试都没有**。首测 1 个点、0 杀死、
击杀率 0.0%。

被测的那一个变异点：

    L28  if engine.dialect.name != "sqlite":      →      == "sqlite"

翻过来的后果是彻底反转：SQLite 引擎**拿不到** WAL 与 busy_timeout
（调度线程与 API 线程池并发写就会 "database is locked"，正是 Task 6.2
要消灭的那个故障），而非 SQLite 引擎反倒被挂上 SQLite 专用 PRAGMA。
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from app.db._sqlite_utils import _BUSY_TIMEOUT_MS, harden_sqlite_engine


@pytest.fixture
def db_url(tmp_path):
    # **必须落到文件**：内存库不支持 WAL（`PRAGMA journal_mode` 返回 "memory"），
    # 用内存库会让这条断言测不到真实行为。
    return f"sqlite:///{tmp_path / 'harden.db'}"


class TestSqliteEnginesGetHardened:

    def test_journal_mode_becomes_wal(self, db_url):
        """
        WAL 是这个模块存在的理由。`!=` 翻成 `==` 会让 SQLite 引擎直接 return，
        日志模式停在默认的 `delete` —— 这条断言就是为它准备的。
        """
        eng = harden_sqlite_engine(create_engine(db_url))
        with eng.connect() as c:
            mode = c.execute(text("PRAGMA journal_mode")).scalar()
        assert str(mode).lower() == "wal", (
            f"journal_mode={mode!r}，不是 wal —— SQLite 引擎没被加固，"
            f"并发写会退回 『database is locked』")

    def test_busy_timeout_matches_the_declared_constant(self, db_url):
        """
        钉住的是**模块自己声明的常量**，不是抄一个 5000 下来：
        常量改了这条要跟着走，而 PRAGMA 没生效时它会读到 0。
        """
        eng = harden_sqlite_engine(create_engine(db_url))
        with eng.connect() as c:
            got = c.execute(text("PRAGMA busy_timeout")).scalar()
        assert int(got) == _BUSY_TIMEOUT_MS, (
            f"busy_timeout={got}，应为 {_BUSY_TIMEOUT_MS} —— "
            f"写锁竞争时会立刻失败而不是等待")

    def test_synchronous_is_normal(self, db_url):
        """WAL 下 synchronous=NORMAL（值 1）是推荐配置；未生效时是 FULL（2）。"""
        eng = harden_sqlite_engine(create_engine(db_url))
        with eng.connect() as c:
            got = c.execute(text("PRAGMA synchronous")).scalar()
        assert int(got) == 1, f"synchronous={got}，应为 1(NORMAL)"

    def test_pragmas_apply_to_every_new_connection_not_just_the_first(self, db_url):
        """
        挂的是 connect 事件而不是一次性执行 —— 连接池会开新连接，
        新连接同样必须带上 PRAGMA。改成"只在第一条连接上设"这里会红。
        """
        eng = harden_sqlite_engine(create_engine(db_url))
        seen = []
        for _ in range(3):
            with eng.connect() as c:
                seen.append(int(c.execute(text("PRAGMA busy_timeout")).scalar()))
            eng.dispose()          # 丢掉连接池，强制下一次开新连接
        assert seen == [_BUSY_TIMEOUT_MS] * 3, (
            f"三次新连接读到的 busy_timeout 是 {seen} —— PRAGMA 没有应用到每个新连接")

    def test_the_same_engine_object_is_returned(self, db_url):
        """返回的必须是**同一个**引擎对象，不是副本 —— 调用方拿的是它的引用。"""
        eng = create_engine(db_url)
        assert harden_sqlite_engine(eng) is eng


class TestNonSqliteEnginesAreLeftAlone:

    def test_a_non_sqlite_dialect_gets_no_listener_and_no_pragmas(self, db_url):
        """
        另一侧：非 SQLite 引擎必须原样返回、**不挂**任何 connect 监听器。

        不引入 psycopg2（它不在 requirements 里，也不该为一条测试而加）：
        用同一个 SQLite 引擎，只把 `dialect.name` 改掉 —— 被测函数读的就是这个字段。
        于是两个用例的差别**只有分支条件本身**，正是 `!=` / `==` 要区分的那一格。
        """
        eng = create_engine(db_url)
        object.__setattr__(eng.dialect, "name", "postgresql")
        assert not _has_connect_listener(eng), "引擎在加固之前就带着 connect 监听器"

        out = harden_sqlite_engine(eng)
        assert out is eng
        assert not _has_connect_listener(eng), (
            "非 SQLite 引擎被挂上了 connect 监听器 —— "
            "SQLite 专用 PRAGMA 会发到别的数据库上")

        # 行为面：既然没挂监听器，连上去也不该有 WAL。
        with eng.connect() as c:
            mode = c.execute(text("PRAGMA journal_mode")).scalar()
        assert str(mode).lower() != "wal", (
            f"dialect.name={eng.dialect.name!r} 却仍然应用了 WAL（journal_mode={mode!r}）")


def _has_connect_listener(eng) -> bool:
    """引擎上是否挂了 connect 事件监听器。"""
    disp = getattr(eng, "dispatch", None)
    listeners = list(getattr(disp, "connect", []) or [])
    return bool(listeners)
