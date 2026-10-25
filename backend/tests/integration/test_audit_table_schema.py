"""
证据 / 正式前向记录 / 试验账本 / 冻结边界这几张审计表的结构契约，以及它们的引擎配置
（D5 关键路径变异复核：evidence_store、forward_record、trial_ledger）。

复核时这几个模块约 70 个存活点是表结构：把 `nullable=False` 改成可空、把可空的旧列改成非空、去掉索引 ——
没有任何用例会发现。审计表的列能不能为空是**契约**：信号存档没有内容、证据没有日期、试验运行没有状态，
都会让"这天算不算前向 / 试验史全不全"无从判断；而 clock_version 等后加的列必须可空（旧行为 NULL = c1）。

这里把期望**直接写出来**，对照真实建出来的 SQLite 表（PRAGMA table_info / index_list），
并验证三个引擎配置的行为：跨线程使用（调度线程与 API 线程共用一个 Store）、不回显 SQL、提交后返回的对象仍可读。
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

PROVEN_EQUIVALENT = {
    "app/db/evidence_store.py ×2 — L47 / L74 `book_key` / `strategy_id` 的 `index=True` -> `False`":
        "signal_archive 有唯一约束 (book_key, asof)、strategy_evidence 有 (strategy_id, date)，SQLite 为它们建的复合索引"
        "前导列就是这两列；本模块的查询都以它们为条件，去掉单列索引后查询计划照样走唯一索引，结果与性能都不变。"
        "见 test_lookups_are_served_by_the_unique_index_alone。",
    "app/db/forward_record.py ×2 — L82 / L84 forward_days 的 `strategy_id` / `decision_date` 的 `index=True` -> `False`":
        "forward_days 有唯一约束 (strategy_id, decision_date)；本模块对它的每个按日期的查询都同时带 strategy_id，"
        "复合索引覆盖；去掉两个单列索引后查询计划照样走唯一索引。见 test_lookups_are_served_by_the_unique_index_alone。",
}


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 2


#: 表 → (必须非空的列, 必须可空的列, 必须有索引的列)
EXPECTED = {
    "signal_archive": ({"id", "book_key", "asof", "spec_hash", "payload", "recorded_at"},
                       {"price_source", "data_digest", "universe_digest", "universe_n", "vintage", "code_version",
                        "clock_version"},
                       {"book_key"}),
    "strategy_evidence": ({"id", "strategy_id", "spec_hash", "date", "ic", "signal_asof", "decision_ts", "recorded_at"},
                          {"price_source", "data_digest", "universe_digest", "universe_n", "vintage", "code_version",
                           "signal_digest", "clock_version", "evidence_class"},
                          {"strategy_id"}),
    "evidence_revisions": ({"id", "kind", "key", "date", "at"}, {"old_value", "new_value", "reason"}, set()),
    "forward_runs": ({"id", "strategy_id", "spec_hash", "started_at", "actor", "reason", "code_version",
                      "data_identity", "environment"}, {"benchmark"}, set()),
    "forward_days": ({"id", "strategy_id", "spec_hash", "decision_date", "formal", "formal_reason", "recorded_at",
                      "payload"}, set(), {"strategy_id", "decision_date"}),
    "forward_day_events": ({"id", "strategy_id", "decision_date", "kind", "recorded_at", "payload"}, set(),
                           {"strategy_id", "decision_date"}),
    "trial_ledger": ({"id", "total"}, {"updated_at"}, set()),
    "trial_runs": ({"run_id", "n", "source", "created_at"}, set(), set()),
    "trial_run_state": ({"run_id", "source", "status", "n_recorded", "started_at", "note"},
                        {"finished_at", "context"}, set()),
    "holdout_usages": ({"id", "dataset_key", "test_key", "purpose", "used_at"},
                       {"experiment", "universe_digest", "universe_n"}, {"dataset_key", "test_key"}),
    "frozen_boundaries": ({"dataset_key", "test_start", "frozen_by", "created_at"}, {"actor", "reason"}, set()),
}


def _stores(url):
    from app.db.evidence_store import EvidenceStore
    from app.db.forward_record import ForwardRecordStore
    from app.db.trial_ledger import BoundaryStore, HoldoutLedger, TrialLedger
    return [EvidenceStore(db_url=url), ForwardRecordStore(db_url=url), TrialLedger(db_url=url),
            HoldoutLedger(db_url=url), BoundaryStore(db_url=url)]


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "audit.db"
    stores = _stores(f"sqlite:///{p.as_posix()}")
    yield p, stores
    for s in stores:
        s._engine.dispose()                                       # noqa: SLF001


@pytest.mark.parametrize("table", sorted(EXPECTED))
def test_not_null_nullable_and_indexes(db, table):
    path, _ = db
    con = sqlite3.connect(path)
    try:
        info = {r[1]: r for r in con.execute(f'PRAGMA table_info("{table}")')}
        indexed = set()
        for _seq, name, *_rest in con.execute(f'PRAGMA index_list("{table}")'):
            indexed |= {r[2] for r in con.execute(f'PRAGMA index_info("{name}")')}
    finally:
        con.close()
    not_null, nullable, want_ix = EXPECTED[table]
    assert set(info) == not_null | nullable, f"{table} 的列变了：{sorted(set(info) ^ (not_null | nullable))}"
    pk = {c for c, r in info.items() if r[5]}
    assert {c for c, r in info.items() if r[3] or c in pk} == not_null, table
    assert want_ix <= indexed, f"{table} 缺索引：{sorted(want_ix - indexed)}"


def test_each_store_can_be_used_from_another_thread(db):
    """调度线程与 API 线程共用一个 Store：连接池里的连接会被别的线程拿到（check_same_thread=False）。"""
    _, stores = db
    for s in stores:
        with s._engine.connect() as c:                             # noqa: SLF001 —— 先在主线程建一条池化连接
            c.exec_driver_sql("SELECT 1")
    errors = []

    def work():
        for s in stores:
            try:
                with s._engine.connect() as c:                     # noqa: SLF001
                    c.exec_driver_sql("SELECT 1")
            except Exception as exc:
                errors.append(f"{type(s).__name__}: {exc}")
    t = threading.Thread(target=work)
    t.start()
    t.join(30)
    assert errors == []


def test_no_store_echoes_sql(db):
    """echo=True 会把每条 SQL 打到标准输出（调度器日志里就是满屏 SQL）。
    SQLAlchemy 的输出处理器在建 Store 时就绑在当时的 stdout 上，测试的输出捕获看不到它 —— 所以直接看引擎的回显开关。"""
    _, stores = db
    assert [type(s).__name__ for s in stores if s._engine.echo] == []   # noqa: SLF001


@pytest.mark.parametrize("sql,index", [
    ("SELECT * FROM signal_archive WHERE book_key = 'a' AND asof = '2025-03-04'", "sqlite_autoindex_signal_archive"),
    ("SELECT asof, recorded_at FROM signal_archive WHERE book_key = 'a'", "sqlite_autoindex_signal_archive"),
    ("SELECT * FROM strategy_evidence WHERE strategy_id = 1 ORDER BY date", "sqlite_autoindex_strategy_evidence"),
    ("SELECT * FROM forward_days WHERE strategy_id = 1 AND decision_date = '2025-03-04'", "sqlite_autoindex_forward_days"),
    ("SELECT * FROM forward_days WHERE strategy_id = 1 AND decision_date < '2025-03-04' "
     "ORDER BY decision_date DESC LIMIT 1", "sqlite_autoindex_forward_days"),
])
def test_lookups_are_served_by_the_unique_index_alone(db, sql, index):
    """单列索引的等价验证：去掉单列索引后，这些表上实际的查询仍走唯一约束的复合索引（SQLite 命名为 sqlite_autoindex_<表>_N，前导列就是查询条件）。"""
    path, _ = db
    con = sqlite3.connect(path)
    try:
        for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'ix_%'").fetchall():
            con.execute(f'DROP INDEX "{name}"')
        plan = " ".join(str(r[-1]) for r in con.execute(f"EXPLAIN QUERY PLAN {sql}"))
    finally:
        con.close()
    assert index in plan, plan


def test_objects_stay_readable_after_the_session_commits(db):
    """返回给调用方的行在会话提交、关闭之后还要能读（expire_on_commit=False）。"""
    import datetime as dt

    from app.db.evidence_store import EvidenceRevision
    from app.db.forward_record import ForwardDayEvent
    from app.db.trial_ledger import FrozenBoundary, HoldoutUse, TrialRunState
    _, stores = db
    now = dt.datetime(2026, 10, 24, 12, 0)
    rows = {
        0: EvidenceRevision(kind="k", key="x", date=dt.date(2026, 10, 1), at=now),
        1: ForwardDayEvent(strategy_id=1, decision_date=dt.date(2026, 10, 1), kind="revision", recorded_at=now, payload="{}"),
        2: TrialRunState(run_id="r", source="gp", status="running", n_recorded=0, started_at=now, note=""),
        3: HoldoutUse(dataset_key="d", test_key="t", purpose="p", used_at=now),
        4: FrozenBoundary(dataset_key="d", test_start="2024-10-24", frozen_by="registered", created_at=now),
    }
    for i, row in rows.items():
        with stores[i]._Session() as s:                            # noqa: SLF001
            s.add(row)
            s.commit()
        assert row.__dict__.get("_sa_instance_state") is not None
        assert getattr(row, list(row.__table__.columns.keys())[1]) is not None
