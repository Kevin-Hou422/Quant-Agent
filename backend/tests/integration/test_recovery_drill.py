"""
隔离恢复演练（2026-10-22 用户决定第 21 条）。

用户原话："做隔离恢复演练、告警接口和文档核对……合成恢复通过不等于活库恢复已验证；直接修正文档原结论，避免追加矛盾脚注。"

    · 恢复只能落到一个空的、隔离的目录：活库、活 PIT 目录不能是目标，已有内容的目录不覆盖
    · 恢复后核对：完整性检查、表一致、逐表行数与源一致、PIT 文件逐字节一致
    · 演练报告写明范围：合成数据的演练通过 ≠ 活库恢复已验证
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _drill(tmp_path):
    from tools.ops.recovery_drill import run_synthetic_drill
    return run_synthetic_drill(tmp_path / "drill")


class TestSyntheticDrill:

    def test_backup_restore_and_verify_round_trip(self, tmp_path):
        rep = _drill(tmp_path)
        assert rep["ok"] is True, rep
        v = rep["verification"]
        assert v["integrity"] == {"main": "ok"} and v["missing_tables"] == [] and v["row_count_mismatches"] == {}
        assert v["tables_checked"] >= 5 and v["pit_files_checked"] >= 1 and v["pit_mismatches"] == []

    def test_the_report_says_a_synthetic_pass_is_not_a_verified_live_restore(self, tmp_path):
        rep = _drill(tmp_path)
        assert rep["scope"] == "synthetic" and rep["live_restore_verified"] is False
        assert "不等于活库恢复已验证" in rep["note"]

    def test_the_drill_never_touches_the_configured_live_paths(self, tmp_path):
        from app.config import settings
        live_db = Path(settings.database_url.split("///", 1)[-1])
        before = live_db.stat().st_mtime_ns if live_db.exists() else None
        rep = _drill(tmp_path)
        assert all(str(tmp_path) in p for p in rep["paths"].values()), rep["paths"]
        assert (live_db.stat().st_mtime_ns if live_db.exists() else None) == before


class TestRestoreSafety:

    def test_restoring_onto_the_live_database_or_pit_is_refused(self, tmp_path, monkeypatch):
        from app.config import settings
        from app.tasks.restore import RestoreRefused, restore_snapshot
        rep = _drill(tmp_path)
        snap = Path(rep["paths"]["snapshot"])
        live_dir = tmp_path / "live"
        live_dir.mkdir()
        monkeypatch.setattr(settings, "database_url", f"sqlite:///{(live_dir / 'alphas.db').as_posix()}")
        with pytest.raises(RestoreRefused, match="活"):
            restore_snapshot(snap, live_dir)
        monkeypatch.setattr(settings, "pit_store_dir", str(tmp_path / "live_pit"))
        with pytest.raises(RestoreRefused, match="活"):
            restore_snapshot(snap, tmp_path / "live_pit" / "x")

    def test_a_non_empty_target_is_not_overwritten(self, tmp_path):
        from app.tasks.restore import RestoreRefused, restore_snapshot
        rep = _drill(tmp_path)
        target = tmp_path / "busy"
        target.mkdir()
        (target / "keep.txt").write_text("x")
        with pytest.raises(RestoreRefused, match="不是空的"):
            restore_snapshot(Path(rep["paths"]["snapshot"]), target)
        assert (target / "keep.txt").read_text() == "x"

    def test_verification_catches_lost_rows_and_damaged_pit_files(self, tmp_path):
        from app.tasks.restore import verify_restore
        rep = _drill(tmp_path)
        restored = Path(rep["paths"]["restored"])
        db = next(restored.glob("*.db"))
        con = sqlite3.connect(db)
        table = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alpha_records'").fetchone()[0]
        con.execute(f"DELETE FROM {table} WHERE id = (SELECT max(id) FROM {table})")
        con.commit(); con.close()
        pit_file = next((restored / "pit_store").rglob("*.parquet"))
        pit_file.write_bytes(pit_file.read_bytes()[:-8])
        v = verify_restore(Path(rep["paths"]["source_db"]), db, Path(rep["paths"]["source_pit"]), restored / "pit_store")
        assert v["ok"] is False
        assert "alpha_records" in v["row_count_mismatches"]
        assert v["pit_mismatches"] == [str(pit_file.relative_to(restored / "pit_store")).replace("\\", "/")]


def test_verifying_against_a_wal_mode_source_leaves_no_files_next_to_it(tmp_path):
    """源库可能是活库：比对时不能在它旁边建 -wal / -shm（普通 mode=ro 打开 WAL 库就会建）。"""
    from app.tasks.restore import verify_restore
    rep = _drill(tmp_path)
    src = Path(rep["paths"]["source_db"])
    con = sqlite3.connect(src)
    con.execute("PRAGMA journal_mode=WAL")
    con.close()
    before = sorted(p.name for p in src.parent.iterdir())
    verify_restore(src, next(Path(rep["paths"]["restored"]).glob("*.db")))
    assert sorted(p.name for p in src.parent.iterdir()) == before


class TestRestoreTargets:

    def test_an_existing_empty_directory_is_a_valid_target(self, tmp_path):
        from app.tasks.restore import restore_snapshot
        rep = _drill(tmp_path)
        target = tmp_path / "empty"
        target.mkdir()
        out = restore_snapshot(Path(rep["paths"]["snapshot"]), target)
        assert out["ok"] is True and out["pit_dir"] == str(target / "pit_store")

    def test_a_nested_new_directory_is_created(self, tmp_path):
        from app.tasks.restore import restore_snapshot
        rep = _drill(tmp_path)
        target = tmp_path / "a" / "b" / "c"
        assert restore_snapshot(Path(rep["paths"]["snapshot"]), target)["ok"] is True

    def test_a_file_as_target_is_refused(self, tmp_path):
        from app.tasks.restore import RestoreRefused, restore_snapshot
        rep = _drill(tmp_path)
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(RestoreRefused, match="不是空的"):
            restore_snapshot(Path(rep["paths"]["snapshot"]), f)

    def test_a_missing_table_alone_fails_the_check(self, tmp_path):
        from app.tasks.restore import verify_restore
        rep = _drill(tmp_path)
        restored = next(Path(rep["paths"]["restored"]).glob("*.db"))
        con = sqlite3.connect(restored)
        con.execute("DROP TABLE alpha_decisions")
        con.commit()
        con.close()
        v = verify_restore(Path(rep["paths"]["source_db"]), restored)
        assert v["missing_tables"] == ["alpha_decisions"] and v["row_count_mismatches"] == {} and v["ok"] is False


def test_the_cli_prints_readable_paths(tmp_path, capsys):
    from app.tasks.restore import main
    rep = _drill(tmp_path)
    target = tmp_path / "恢复演练"
    assert main([rep["paths"]["snapshot"], str(target)]) == 0
    assert "恢复演练" in capsys.readouterr().out
