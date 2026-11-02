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
        # 审计决定 2：不再按"指标是否非零"推断评估过；旧行除基准种子外一律"不知道"
        assert inv[1]["ic_status"] == "legacy_unknown" and inv[1]["evaluated"] is None


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
        # 审计决定 2 / 3：新值分列（ic_mean / ic_ir），列语义纠错与新版本重评估是分开的修订理由
        import json as _json
        pf = rows[1]["proposed_fields"]
        assert set(pf) == {"ic_mean", "ic_ir", "ic_metrics"} and _json.loads(pf["ic_metrics"])["segment"] == "selection"
        assert any("列语义纠错" in x for x in rows[1]["revision_reasons"])
        assert any("未评估" in x for x in rows[3]["revision_reasons"])
        assert all(any("新版本重评估" in x for x in r["revision_reasons"]) for r in rows.values())
        assert Path(out["report_file"]).exists() and json.loads(Path(out["report_file"]).read_text(encoding="utf-8"))

    def test_the_dry_run_refuses_to_point_at_the_configured_live_database(self, tmp_path, monkeypatch):
        from app.config import settings
        live = _live(tmp_path)
        monkeypatch.setattr(settings, "database_url", f"sqlite:///{live.as_posix()}")
        with pytest.raises(fm.PreviewRefused, match="副本"):
            fm.dry_run(live, "synthetic_test", None, None, loader=_loader)


class TestWritebackPlan:
    """
    2026-10-30 用户决定 4："请先在一致快照副本上完成预演，列出：每条记录旧值、新值、来源、样本、算法和时钟版本；旧算法纠错
    与使用 moomoo/c2/新边界重新评估的不同修订理由；受影响的排序、准入和历史结论；修订记录、回滚步骤和恢复验证。完成差异后
    交我确认；不要直接写活库。"
    """

    def _copy(self, tmp_path, old_schema=False):
        db = _live(tmp_path)
        if old_schema:                                         # 活库现状：还没有 ic_mean / ic_metrics 两列
            con = sqlite3.connect(db)
            con.execute("ALTER TABLE alpha_records DROP COLUMN ic_mean")
            con.execute("ALTER TABLE alpha_records DROP COLUMN ic_metrics")
            con.commit()
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            con.close()
        return db, Path(fm.snapshot_live(db, tmp_path / "preview")["copy"])

    @pytest.mark.parametrize("old_schema", [False, True])
    def test_the_fix_is_rehearsed_with_revisions_rollback_and_restore_on_copies_only(self, tmp_path, old_schema):
        live, copy = self._copy(tmp_path, old_schema)
        before = _fingerprint(live), _fingerprint(copy)
        plan = fm.writeback_plan(copy, tmp_path / "plan")
        assert (_fingerprint(live), _fingerprint(copy)) == before, "活库与快照副本都不能被写"
        assert plan["migration"]["add_columns"] == (["ic_mean", "ic_metrics"] if old_schema else [])
        fix = {c["id"]: c for c in plan["fix"]}
        assert set(fix) == {1, 2, 3} and plan["fix_unresolved"] == []
        assert fix[1]["new"]["ic_mean"] == pytest.approx(-0.02) and fix[1]["new"]["ic_ir"] is None
        assert json.loads(fix[1]["new"]["ic_metrics"])["status"] == "values_without_provenance"
        assert json.loads(fix[3]["new"]["ic_metrics"])["status"] == "never_evaluated"
        assert {c["reason_kind"] for c in plan["fix"]} == {fm.FIX}
        r = plan["rehearsal"]
        assert r["after_apply"]["ok"] is True and r["after_apply"]["revision_rows_added"] == 3
        assert r["logical_rollback"] == {**r["logical_rollback"], "original_columns_restored": True,
                                         "revision_rows_removed": True}
        assert r["physical_rollback"]["ok"] is True
        ranks = plan["affected"]["ranks"]
        assert ranks[2]["by_ic_ir_before"] == 1 and (ranks[2]["by_ic_mean_after"], ranks[1]["by_ic_mean_after"]) == (1, 2)
        assert plan["affected"]["admission"][1]["status_changes"].startswith("无")

    def test_a_value_that_changed_since_planning_aborts_the_whole_transaction(self, tmp_path):
        _live_db, copy = self._copy(tmp_path)
        plan = fm.writeback_plan(copy, tmp_path / "plan")
        work = tmp_path / "w.db"
        import shutil
        shutil.copy2(copy, work)
        con = sqlite3.connect(work)
        con.execute("UPDATE alpha_records SET ic_ir = 0.5 WHERE id = 2")         # 计划之后有人改了这一行
        con.commit()
        con.close()
        before = _fingerprint(work)[0]
        with pytest.raises(RuntimeError, match="#2"):
            fm._apply_fix(work, plan["fix"], "tester", "2026-10-30T00:00:00+00:00")   # noqa: SLF001
        con = sqlite3.connect(work)
        assert con.execute("SELECT count(*) FROM alpha_decisions WHERE decision = 'metric_fix'").fetchone()[0] == 0
        assert con.execute("SELECT ic_ir FROM alpha_records WHERE id = 1").fetchone()[0] == pytest.approx(-0.02)
        con.close()
        assert _fingerprint(work)[0] == before

    @pytest.mark.parametrize("src,boundary,executable", [("yahoo", "provisional", False), ("moomoo", "provisional", False),
                                                         ("moomoo", "registered", True)])
    def test_reevaluation_is_a_separate_set_and_only_moomoo_on_a_registered_boundary_is_executable(
            self, tmp_path, src, boundary, executable):
        _live_db, copy = self._copy(tmp_path)
        from app.core.alpha_engine.ic_metrics import ic_fields
        rep = {"versions": {"algorithm": "RealisticBacktester", "code_version": "abc", "boundary": boundary,
                            "data": {"price_source": src}},
               "records": [{"id": 1, "proposed_fields": ic_fields(0.01, 0.2, 300, source="dry", segment="selection",
                                                                  window=["2021-01-04", "2026-10-02"])}]}
        p = tmp_path / "dry.json"
        p.write_text(json.dumps(rep), encoding="utf-8")
        plan = fm.writeback_plan(copy, tmp_path / "plan", reevaluation=p)
        re = plan["reevaluation"]
        assert re["executable"] is executable and re["changes"][0]["reason_kind"] == fm.REEVAL
        c = re["changes"][0]
        assert (c["n_obs"], c["price_source"], c["window"]) == (300, src, ["2021-01-04", "2026-10-02"])
        assert {x["reason_kind"] for x in plan["fix"]} == {fm.FIX}             # 两种理由不合并

    def test_the_command_line_writes_only_the_plan(self, tmp_path, capsys):
        _live_db, copy = self._copy(tmp_path)
        assert fm.main(["writeback-plan", "--copy", str(copy), "--out", str(tmp_path / "plan")]) == 0
        assert (tmp_path / "plan" / "writeback_plan.json").exists()
        assert '"writes": "rehearsal copies only' in capsys.readouterr().out
