"""
研究边界工具：活库只读报告（只读连接、不建表、拒绝未合并的 WAL、读前读后文件不变）与登记命令（缺省预演）。

来源（原测试文件的说明，保留出处）：

2026-10-22 用户决定 · 证据 / 研究纪律。

D12 原话："同意先只读查询，不授权写边界。使用明确的活库路径和真正只读连接，不调用会建表或迁移的 Store 构造函数。
列出已有边界、最早曝光日期、次数、股票池和实验身份，给出日期建议后由我确认。正式入口不要继续依赖首次调用偶然冻结；
改实验名不能刷新重叠数据的曝光额度。"
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# D12-c：只读查询（明确路径、真正只读连接、不调用会建表 / 迁移的 Store 构造函数）
# ---------------------------------------------------------------------------

def _make_live_like_db(path: Path, *, with_boundaries: bool) -> None:
    """按真实活库的形状造一个 WAL 库：有 holdout_usages / trial_run_state / alpha_records，可选 frozen_boundaries。"""
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE holdout_usages (id INTEGER PRIMARY KEY, dataset_key TEXT, test_key TEXT, "
                "purpose TEXT, used_at TEXT)")
    con.execute("CREATE TABLE trial_run_state (run_id TEXT PRIMARY KEY, source TEXT, status TEXT, n_recorded INT, "
                "started_at TEXT, finished_at TEXT, note TEXT, context TEXT)")
    con.execute("CREATE TABLE alpha_records (id INTEGER PRIMARY KEY, dsl TEXT, created_at TEXT, status TEXT)")
    con.execute("CREATE TABLE trial_ledger (id INTEGER PRIMARY KEY, total INT, updated_at TEXT)")
    con.executemany("INSERT INTO holdout_usages (dataset_key, test_key, purpose, used_at) VALUES (?,?,?,?)", [
        ("us_tech_large", "2024-10-01..2026-09-30", "validation_gate", "2026-10-01 10:00:00"),
        ("us_tech_large", "2024-09-15..2026-10-09", "api.backtest", "2026-10-10 10:00:00"),
    ])
    con.execute("INSERT INTO trial_run_state VALUES ('r1','gp','done',120,'2026-10-11 09:00:00',"
                "'2026-10-11 09:10:00','', ?)",
                ('{"dataset_key": "us_broad_large", "experiment": "default", "universe_digest": "abc", '
                 '"universe_n": 98, "test_start": null, "selection_start": "2021-01-04", '
                 '"selection_end": "2026-10-09"}',))
    con.execute("INSERT INTO alpha_records (dsl, created_at, status) VALUES ('rank(close)', '2026-10-11 09:07:40', "
                "'validated')")
    con.execute("INSERT INTO trial_ledger VALUES (1, 258, '2026-10-11 09:15:27')")
    if with_boundaries:
        con.execute("CREATE TABLE frozen_boundaries (dataset_key TEXT PRIMARY KEY, test_start TEXT, frozen_by TEXT, "
                    "created_at TEXT)")
        con.execute("INSERT INTO frozen_boundaries VALUES ('us_tech_large', '2024-10-01', 'years', "
                    "'2026-10-01 09:00:00')")
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    for suf in ("-wal", "-shm"):
        p = Path(str(path) + suf)
        if p.exists():
            p.unlink()


def _snapshot(path: Path):
    return (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns,
            sorted(os.listdir(path.parent)))


class TestReadOnlyReport:

    def test_the_report_lists_boundaries_exposure_counts_universe_and_experiment(self, tmp_path):
        from tools.research.research_boundary_report import build_report
        db = tmp_path / "live.db"
        _make_live_like_db(db, with_boundaries=True)
        before = _snapshot(db)
        rep = build_report(db)
        assert _snapshot(db) == before, "只读查询改动了活库文件或在旁边建了文件"
        tech = rep["datasets"]["us_tech_large"]
        assert tech["boundary"] == {"test_start": "2024-10-01", "frozen_by": "years", "registered": False}
        assert tech["earliest_exposed_start"] == "2024-09-15" and tech["exposures"] == 2
        assert set(tech["purposes"]) == {"validation_gate", "api.backtest"}
        broad = rep["datasets"]["us_broad_large"]
        assert broad["experiments"] == ["default"] and broad["universes"] == [{"digest": "abc", "n": 98}]
        assert broad["research_windows"] == [{"start": "2021-01-04", "end": "2026-10-09", "test_start": None}]
        assert rep["trial_total"] == 258

    def test_research_without_any_boundary_is_flagged_as_unrecorded_exposure(self, tmp_path):
        from tools.research.research_boundary_report import build_report
        db = tmp_path / "live.db"
        _make_live_like_db(db, with_boundaries=False)
        rep = build_report(db)
        assert rep["tables_missing"] == ["frozen_boundaries"], "表不存在应如实报告，不能替它建表"
        broad = rep["datasets"]["us_broad_large"]
        assert broad["boundary"] is None
        assert broad["unrecorded_exposure"]["through"] == "2026-10-09"
        sugg = {s["option"]: s for s in broad["suggestions"]}
        assert sugg["clean"]["test_start"] > "2026-10-09", "干净方案的 Test 起点必须晚于所有被看过的日期"
        assert sugg["clean"]["needs_user_confirmation"] is True
        assert sqlite3.connect(db).execute(
            "select count(*) from sqlite_master where name='frozen_boundaries'").fetchone()[0] == 0

    def test_a_pending_wal_file_is_refused_rather_than_read_partially(self, tmp_path):
        from tools.research.research_boundary_report import LiveDbBusy, build_report
        db = tmp_path / "live.db"
        _make_live_like_db(db, with_boundaries=True)
        Path(str(db) + "-wal").write_bytes(b"\0" * 32)
        with pytest.raises(LiveDbBusy):
            build_report(db)

    def test_the_report_module_never_touches_store_constructors(self):
        """D12："不调用会建表或迁移的 Store 构造函数" —— 报告模块只许用 sqlite3 的只读连接。"""
        import ast
        src = Path(__file__).resolve().parents[2] / "tools" / "research" / "research_boundary_report.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any(m.startswith(("app.db", "sqlalchemy")) for m in mods), mods
        calls = {getattr(n.func, "attr", getattr(n.func, "id", "")) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assert "create_all" not in calls and "create_engine" not in calls


def test_an_old_schema_db_reports_inferred_exposure_from_research_activity(tmp_path):
    """
    活库是旧结构：没有 frozen_boundaries / trial_run_state，研究记录里没有数据集与窗口，只有研究发生的时刻。
    报告不能把"没记录"说成"没看过"：按用户给定的数据起点，推定研究那天之前的全部数据都已曝光，并标明是推定。
    """
    from tools.research.research_boundary_report import build_report
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE alpha_records (id INTEGER PRIMARY KEY, dsl TEXT, created_at TEXT, status TEXT)")
    con.execute("CREATE TABLE trial_ledger (id INTEGER PRIMARY KEY, total INT, updated_at TEXT)")
    con.execute("CREATE TABLE holdout_usages (id INTEGER PRIMARY KEY, dataset_key TEXT, test_key TEXT, purpose TEXT, "
                "used_at TEXT)")
    con.execute("INSERT INTO alpha_records (dsl, created_at, status) VALUES ('x', '2026-10-11 09:07:40', 'validated')")
    con.execute("INSERT INTO trial_ledger VALUES (1, 258, '2026-10-11 09:15:27')")
    con.commit(); con.close()
    before = _snapshot(db)
    rep = build_report(db, assume_data_start={"us_broad_large": "2021-01-01"})
    assert _snapshot(db) == before
    assert set(rep["tables_missing"]) == {"frozen_boundaries", "trial_run_state"}
    assert rep["research_activity"] == {"first": "2026-10-11 09:07:40", "last": "2026-10-11 09:15:27",
                                        "alpha_records": 1, "trial_total": 258}
    broad = rep["datasets"]["us_broad_large"]
    ue = broad["unrecorded_exposure"]
    assert ue["inferred"] is True and ue["from"] == "2021-01-01" and ue["through"] == "2026-10-09"
    assert {s["option"] for s in broad["suggestions"]} >= {"clean", "conventional"}
    conv = {s["option"]: s for s in broad["suggestions"]}["conventional"]
    assert conv["already_exposed"] is True


def test_registration_is_a_dry_run_unless_confirmed(tmp_path):
    from tools.research.register_research_boundary import main

    from app.db.trial_ledger import BoundaryStore
    db = tmp_path / "live.db"
    _make_live_like_db(db, with_boundaries=False)
    before = _snapshot(db)
    assert main(["--db", str(db), "--dataset", "us_broad_large", "--test-start", "2026-10-12",
                 "--actor", "Kevin-Hou422", "--reason", "D12 clean"]) == 0
    assert _snapshot(db) == before, "预演改动了库"
    assert main(["--db", str(db), "--dataset", "us_broad_large", "--test-start", "2026-10-12",
                 "--actor", "Kevin-Hou422", "--reason", "D12 clean", "--confirm"]) == 0
    st = BoundaryStore(db_url=f"sqlite:///{db.as_posix()}").status("us_broad_large")
    assert st == (pd.Timestamp("2026-10-12"), "registered")
    assert main(["--db", str(db), "--dataset", "us_broad_large", "--test-start", "2025-01-02",
                 "--actor", "x", "--reason", "改", "--confirm"]) == 2
