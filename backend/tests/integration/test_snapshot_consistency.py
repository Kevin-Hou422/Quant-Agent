"""
只读快照与恢复核验在并发 WAL 写入下的证明（2026-10-25 审计 R2）。

审计复现："临时 WAL 库开始有 1 行，已有空 WAL/SHM。预检查后另一个连接提交第 2 行，快照仍只有 1 行，
工具却返回 live_unchanged=true、restore.ok=true。真实连接可见 2 行。"
要求："明确选择完整停写窗口并验证……恢复核验除 integrity_check、表和行数外，核对关键主键、内容摘要、
订单认领/成交/熔断/正式证据等业务不变量。只对真正冻结的副本使用 immutable。"

    · 停写窗口内有提交（只追加 WAL / 提交后立即 checkpoint 回主库 / 复制完成之后才提交）→ 副本作废、报错，不再报"未变"
    · 窗口内没有写入 → 副本、清单、恢复核验通过；活库目录不多任何文件
    · 恢复核验：行数相同但内容 / 主键 / 熔断状态不同 → 不通过；运行槽不唯一 → 列为违反
    · 产品备份写清单；恢复演练对照快照文件本身
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.tasks import snapshot_manifest as sm
from tools.research import factor_metrics_dry_run as fm


def _wal_db(tmp_path):
    """审计的构造：WAL 库，1 行，已有空的 -wal / -shm（写连接一直开着）。"""
    d = tmp_path / "QuantAgentData"
    d.mkdir()
    db = d / "alphas.db"
    w = sqlite3.connect(db)
    w.execute("PRAGMA journal_mode=WAL")
    w.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    w.execute("INSERT INTO t (v) VALUES ('a')")
    w.commit()
    w.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert Path(str(db) + "-wal").stat().st_size == 0 and Path(str(db) + "-shm").exists()
    return db, w


def _rows(db):
    con = sqlite3.connect(db)
    try:
        return con.execute("SELECT count(*) FROM t").fetchone()[0]
    finally:
        con.close()


@pytest.fixture
def writer(tmp_path):
    db, w = _wal_db(tmp_path)
    yield db, w
    w.close()


def _commit_during_copy(monkeypatch, w, when="before", checkpoint=False):
    real = sm._copy_bytes

    def copy(src, dest):
        if when == "after":
            real(src, dest)
        w.execute("INSERT INTO t (v) VALUES ('b')")
        w.commit()
        if checkpoint:
            w.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if when == "before":
            real(src, dest)
    monkeypatch.setattr(sm, "_copy_bytes", copy)


@pytest.mark.parametrize("when,checkpoint", [("before", False), ("before", True), ("after", False)],
                         ids=["commit_into_wal", "commit_then_checkpoint", "commit_after_copy"])
def test_a_commit_inside_the_window_voids_the_copy(writer, tmp_path, monkeypatch, when, checkpoint):
    db, w = writer
    _commit_during_copy(monkeypatch, w, when, checkpoint)
    dest = tmp_path / "frozen" / db.name
    with pytest.raises(sm.SourceChanged):
        sm.copy_in_quiet_window(db, dest)
    assert not dest.exists(), "作废的副本必须删掉"
    assert _rows(db) == 2, "真实连接可见 2 行"


def test_the_audit_reproduction_no_longer_reports_unchanged(writer, tmp_path, monkeypatch):
    """原工具在这里返回 live_unchanged=true、restore.ok=true（快照只有 1 行）。"""
    db, w = writer
    _commit_during_copy(monkeypatch, w)
    with pytest.raises(fm.SourceChanged):
        fm.snapshot_live(db, tmp_path / "preview")
    assert not (tmp_path / "preview" / "preview.json").exists()


def test_a_quiet_window_gives_a_verified_copy_and_leaves_the_live_directory_alone(writer, tmp_path):
    db, _w = writer
    before = sm.db_files_state(db)
    rep = fm.snapshot_live(db, tmp_path / "preview")
    assert sm.db_files_state(db) == before
    assert rep["live_unchanged"] is True and rep["restore"]["ok"] is True
    assert rep["window"]["copy_sha256"] == before["db"]["sha256"]
    assert rep["manifest"]["tables"]["t"]["rows"] == 1 and rep["restore"]["content_mismatches"] == []
    on_disk = json.loads((tmp_path / "preview" / "preview.json").read_text(encoding="utf-8"))
    assert on_disk["window"]["window_opened_at"] <= on_disk["window"]["window_closed_at"]


def test_a_non_empty_wal_is_still_refused_before_reading(writer, tmp_path):
    db, w = writer
    w.execute("INSERT INTO t (v) VALUES ('b')")
    w.commit()
    with pytest.raises(fm.LiveDbBusy):
        fm.snapshot_live(db, tmp_path / "preview")
    assert Path(str(db) + "-wal").stat().st_size > 0, "不靠删除 WAL 实现一致"


# ── 恢复核验：内容与业务不变量，不只是行数 ─────────────────────────────

def _business_db(path: Path, running=(1,), kill="off"):
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE strategy_configs (id INTEGER PRIMARY KEY, status TEXT);
        CREATE TABLE exec_orders (id INTEGER PRIMARY KEY, client_id TEXT UNIQUE, status TEXT, dealt_qty REAL, qty INTEGER);
        CREATE TABLE exec_fills (id INTEGER PRIMARY KEY, client_id TEXT, cum_qty REAL, qty REAL, price REAL);
        CREATE TABLE exec_state (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE forward_days (id INTEGER PRIMARY KEY, strategy_id INTEGER, spec_hash TEXT, decision_date TEXT,
                                   formal INTEGER);
    """)
    for i in (1, 2, 3):
        con.execute("INSERT INTO strategy_configs VALUES (?, ?)", (i, "paper_running" if i in running else "retired"))
    con.execute("INSERT INTO exec_orders VALUES (1, 'c-1', 'FILLED', 50, 50)")
    con.execute("INSERT INTO exec_orders VALUES (2, 'c-2', 'SUBMITTED', 0, 10)")
    con.execute("INSERT INTO exec_fills VALUES (1, 'c-1', 50, 50, 90.0)")
    con.execute("INSERT INTO exec_state VALUES ('kill_switch', ?)", (json.dumps({"engaged": kill == "on"}),))
    con.execute("INSERT INTO forward_days VALUES (1, 1, 'h', '2026-10-01', 1)")
    con.commit()
    con.close()
    return path


def test_same_row_counts_with_different_content_fail_verification(tmp_path):
    from app.tasks.restore import verify_restore
    a = _business_db(tmp_path / "a.db")
    b = _business_db(tmp_path / "b.db")
    con = sqlite3.connect(b)
    con.execute("UPDATE exec_orders SET status = 'NOT_SUBMITTED' WHERE client_id = 'c-2'")   # 在途认领被"恢复"成未提交
    con.commit()
    con.close()
    v = verify_restore(a, b)
    assert v["row_count_mismatches"] == {} and v["ok"] is False
    assert "exec_orders" in v["content_mismatches"] and "orders" in v["invariant_mismatches"]


def test_a_lost_kill_switch_fails_verification(tmp_path):
    from app.tasks.restore import verify_restore
    v = verify_restore(_business_db(tmp_path / "a.db", kill="on"), _business_db(tmp_path / "b.db", kill="off"))
    assert v["ok"] is False and "kill_switch" in v["invariant_mismatches"]


def test_two_running_strategies_are_reported_as_a_violation(tmp_path):
    from app.tasks.restore import verify_restore
    a = _business_db(tmp_path / "a.db", running=(1, 2))
    v = verify_restore(a, a)
    assert v["ok"] is True, "恢复忠实于源：两边一致"
    assert any("运行槽不唯一" in x for x in v["invariant_violations"])
    assert sm.describe_db(a)["invariants"]["facts"]["orders_in_flight"] == ["c-2"]


def test_the_product_backup_writes_a_manifest_and_the_drill_checks_against_the_snapshot(tmp_path):
    from tools.ops.recovery_drill import run_synthetic_drill
    rep = run_synthetic_drill(tmp_path / "drill")
    assert rep["ok"] is True
    snap = Path(rep["paths"]["snapshot"])
    man = sm.read_manifest(snap)
    assert man is not None and man["ok"] is True and man["cross_item_consistent"] is False
    item = man["items"]["alphas.db"]
    assert item["method"] == "vacuum_into" and item["description"]["tables"]["alpha_records"]["rows"] == 3
    assert rep["verification"]["frozen_source"] == str(snap / "alphas.db")
    assert man["items"]["pit_store.zip"]["files"]


def test_the_boundary_report_refuses_a_write_inside_its_window(writer, monkeypatch):
    from tools.research.research_boundary_report import SourceChanged, build_report
    db, w = writer
    _commit_during_copy(monkeypatch, w)
    with pytest.raises(SourceChanged):
        build_report(db)
