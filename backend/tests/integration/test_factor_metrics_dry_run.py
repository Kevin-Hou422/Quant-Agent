"""
活库因子指标的备份 + 恢复验证 + 版本化 dry-run（2026-10-22 用户决定 D2）。

用户原话："先备份、验证恢复并做版本化 dry-run。展示实际受影响记录、旧新值、算法/数据/时钟版本及排序影响。
保留旧值和曝光史。展示差异后，我另行确认写活库；不要直接覆盖。"

    · 快照：活库只读（mode=ro&immutable=1）—— 库文件、修改时间、目录里的文件一个都不变；旁边有未合并的 -wal 时拒绝
    · 恢复验证：快照恢复到隔离目录，逐表比行数
    · dry-run 只在恢复出的副本上做：每条记录是哪条路径写的、那一列存的到底是什么、旧值 / 新值 / 排序变化、
      算法 / 数据 / 时钟版本、曝光史；**不写任何库**（活库与副本都不写）
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import _make_dataset
from tools.research import factor_metrics_dry_run as fm


def _live(tmp_path, wal=True):
    """一个"活库"：WAL 模式、有因子记录（发现路径写的、基准种子写的），放在单独目录里。"""
    from app.db.alpha_store import AlphaRecord, AlphaStore
    d = tmp_path / "QuantAgentData"
    d.mkdir()
    db = d / "alphas.db"
    url = f"sqlite:///{db.as_posix()}"
    st = AlphaStore(db_url=url)
    with st._Session() as s:                                          # noqa: SLF001 —— 造存量记录，不走晋级门
        s.add_all([
            AlphaRecord(dsl="rank(ts_mean(close, 5))", hypothesis="auto-discovery | family=reversion | regime=x",
                        reasoning="Hypothesis: reversion | Seeded GP", status="validated", sharpe=0.2, ic_ir=-0.02),
            AlphaRecord(dsl="rank(-ts_std(returns, 20))", hypothesis="auto-discovery | family=volatility | regime=x",
                        reasoning="", status="validated", sharpe=0.4, ic_ir=0.01),
            AlphaRecord(dsl="rank(ts_delta(log(close), 20))", hypothesis="[baseline:momentum] 动量", reasoning="",
                        status="paper", sharpe=0.0, ic_ir=0.0),
        ])
        s.commit()
    st._engine.dispose()                                              # noqa: SLF001
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=WAL" if wal else "PRAGMA journal_mode=DELETE")
    con.close()
    return db


def _fingerprint(db: Path):
    return (hashlib.sha256(db.read_bytes()).hexdigest(), db.stat().st_mtime_ns, sorted(p.name for p in db.parent.iterdir()))


class TestSnapshot:

    def test_the_live_database_and_its_directory_are_left_exactly_as_they_were(self, tmp_path):
        db = _live(tmp_path)
        before = _fingerprint(db)
        rep = fm.snapshot_live(db, tmp_path / "preview")
        assert _fingerprint(db) == before
        assert rep["live_unchanged"] is True and rep["restore"]["ok"] is True
        assert rep["restore"]["row_count_mismatches"] == {} and rep["restore"]["tables_checked"] >= 3

    def test_an_unmerged_wal_is_refused(self, tmp_path):
        db = _live(tmp_path)
        Path(str(db) + "-wal").write_bytes(b"\x00" * 64)
        with pytest.raises(fm.LiveDbBusy):
            fm.snapshot_live(db, tmp_path / "preview")

    def test_the_output_must_be_a_new_empty_directory_away_from_the_live_one(self, tmp_path):
        db = _live(tmp_path)
        with pytest.raises(fm.PreviewRefused, match="活库所在目录"):
            fm.snapshot_live(db, db.parent / "preview")
        busy = tmp_path / "busy"
        busy.mkdir()
        (busy / "x").write_text("x")
        with pytest.raises(fm.PreviewRefused, match="不是空的"):
            fm.snapshot_live(db, busy)


class TestInventory:

    def test_each_record_says_who_wrote_it_and_what_the_column_holds(self, tmp_path):
        rep = fm.snapshot_live(_live(tmp_path), tmp_path / "preview")
        inv = {r["id"]: r for r in fm.inventory(Path(rep["copy"]))}
        assert inv[1]["writer"] == "discovery" and inv[1]["ic_ir_column_holds"] == "样本内 IC 均值（is_ic）"
        assert inv[3]["writer"] == "baseline_seed" and inv[3]["evaluated"] is False
        assert inv[1]["stored"]["ic_ir"] == pytest.approx(-0.02)


def _loader(name, start=None, end=None, **kw):
    return SimpleNamespace(data=_make_dataset(n_days=400, n_tickers=12, seed=9))


class TestDryRun:

    def test_it_shows_old_and_new_values_versions_and_rank_changes_without_writing(self, tmp_path, monkeypatch):
        from app.config import settings
        live = _live(tmp_path)
        rep = fm.snapshot_live(live, tmp_path / "preview")
        copy = Path(rep["copy"])
        before_copy, before_live = _fingerprint(copy), _fingerprint(live)
        url_before = settings.database_url
        out = fm.dry_run(copy, "synthetic_test", None, None, loader=_loader)
        assert settings.database_url == url_before, "跑完要把配置还原"
        assert _fingerprint(live) == before_live
        assert _fingerprint(copy)[0] == before_copy[0], "dry-run 不写副本"
        rows = {r["id"]: r for r in out["records"]}
        assert set(rows) == {1, 2, 3}
        for r in rows.values():
            assert {"mean_ic", "ic_ir", "sharpe"} <= set(r["recomputed"])
            assert "rank_by_stored" in r and "rank_by_recomputed_mean_ic" in r
        assert rows[3]["stored"]["ic_ir"] == 0.0 and rows[3]["evaluated_before"] is False
        v = out["versions"]
        assert v["clock_version"] and v["code_version"] and v["data"]["data_digest"] and v["dataset"] == "synthetic_test"
        assert v["boundary"] and v["segment"] == "selection" and v["data"]["price_source"]
        assert "exposure" in out and "holdout_usages" in out["exposure"]
        assert out["writes"] == "none"
        assert Path(out["report_file"]).exists() and json.loads(Path(out["report_file"]).read_text(encoding="utf-8"))

    def test_the_dry_run_refuses_to_point_at_the_configured_live_database(self, tmp_path, monkeypatch):
        from app.config import settings
        live = _live(tmp_path)
        monkeypatch.setattr(settings, "database_url", f"sqlite:///{live.as_posix()}")
        with pytest.raises(fm.PreviewRefused, match="副本"):
            fm.dry_run(live, "synthetic_test", None, None, loader=_loader)
