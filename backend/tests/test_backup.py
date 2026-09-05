"""
test_backup.py — 每日一致性快照备份

- VACUUM INTO 产出的是**可用**副本(能查、integrity_check=ok),不是坏文件
- **活库在写入中**时快照仍然有效(这正是 copy 文件做不到的)
- PIT 目录被打包
- 保留 N 份,多余的被清理
- 失败不静默:坏源 → ok=False 且 errors 有内容
"""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

import pytest

from app.tasks.backup import (
    snapshot_sqlite, snapshot_tree, prune_snapshots, run_daily_backup,
)


def _make_db(path: Path, rows=100):
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    con.executemany("INSERT INTO t (v) VALUES (?)", [(f"row{i}",) for i in range(rows)])
    con.commit(); con.close()


def test_snapshot_is_valid_and_queryable(tmp_path):
    src = tmp_path / "src.db"; _make_db(src, 250)
    dst = tmp_path / "out" / "snap.db"
    n = snapshot_sqlite(src, dst)
    assert n > 0 and dst.exists()
    con = sqlite3.connect(str(dst))
    assert con.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 250   # 数据完整
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    con.close()


def test_snapshot_consistent_while_db_in_wal_with_open_writer(tmp_path):
    """
    活库处于 WAL 且有未 checkpoint 的写入时,快照仍应是**一致可用**的副本。
    直接 copy .db 文件在这种情况下会丢掉 WAL 里的数据。
    """
    src = tmp_path / "live.db"; _make_db(src, 10)
    con = sqlite3.connect(str(src))
    con.execute("PRAGMA journal_mode=WAL")
    con.executemany("INSERT INTO t (v) VALUES (?)", [(f"wal{i}",) for i in range(90)])
    con.commit()                                   # 数据在 WAL 里,可能还没并回主文件
    dst = tmp_path / "snap.db"
    snapshot_sqlite(src, dst)                      # 快照期间写连接仍开着
    con.close()

    chk = sqlite3.connect(str(dst))
    assert chk.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 100   # WAL 中的写入也在
    assert chk.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    chk.close()


def test_snapshot_tree_zips_pit(tmp_path):
    pit = tmp_path / "pit" / "ds" / "year=2026"
    pit.mkdir(parents=True)
    (pit / "data.parquet").write_bytes(b"PAR1fake")
    out = snapshot_tree(tmp_path / "pit", tmp_path / "snap" / "pit_store")
    assert out > 0
    z = zipfile.ZipFile(tmp_path / "snap" / "pit_store.zip")
    assert any("data.parquet" in n for n in z.namelist())


def test_prune_keeps_latest_n(tmp_path):
    root = tmp_path / "backups"
    for name in ["snap-20260101T000000Z", "snap-20260102T000000Z",
                 "snap-20260103T000000Z", "snap-20260104T000000Z"]:
        (root / name).mkdir(parents=True)
    pruned = prune_snapshots(root, keep=2)
    left = sorted(d.name for d in root.iterdir())
    assert len(left) == 2 and left == ["snap-20260103T000000Z", "snap-20260104T000000Z"]
    assert len(pruned) == 2


def test_run_daily_backup_end_to_end(tmp_path, monkeypatch):
    from app.config import settings
    main = tmp_path / "alphas.db"; _make_db(main, 42)
    sch = tmp_path / "scheduler_jobs.db"; _make_db(sch, 3)
    pit = tmp_path / "pit" / "ds" / "year=2026"; pit.mkdir(parents=True)
    (pit / "data.parquet").write_bytes(b"PAR1fake")

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{main}")
    monkeypatch.setattr(settings, "scheduler_db_url", f"sqlite:///{sch}")
    monkeypatch.setattr(settings, "pit_store_dir", str(tmp_path / "pit"))
    monkeypatch.setattr(settings, "backup_dir", str(tmp_path / "backups"))
    monkeypatch.setattr(settings, "backup_keep_n", 3)

    res = run_daily_backup()
    assert res.ok and not res.errors
    assert "alphas.db" in res.items and "pit_store.zip" in res.items
    snap = Path(res.snapshot_dir)
    con = sqlite3.connect(str(snap / "alphas.db"))
    assert con.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 42     # 备份真的可用
    con.close()


def test_failure_is_not_silent(tmp_path, monkeypatch):
    """源库不存在 → ok=False 且 errors 非空(坏备份比没备份更危险)。"""
    from app.config import settings
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path/'missing.db'}")
    monkeypatch.setattr(settings, "scheduler_db_url", "sqlite:///nope.db")
    monkeypatch.setattr(settings, "pit_store_dir", str(tmp_path / "nopit"))
    monkeypatch.setattr(settings, "backup_dir", str(tmp_path / "b"))
    res = run_daily_backup()
    assert res.ok is False and res.errors
