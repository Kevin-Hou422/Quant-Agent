"""
现役策略重新提案的隔离预演（2026-10-22 用户决定 D13）。

用户原话："同意重提案，但先修正确性、D14 和资格状态。先隔离预演，再由我批准活环境操作。
不手工给旧策略补 hash，不继承旧时钟证据，不要先激活旧时钟再重做。"

    · 只在活库快照的副本上做（再复制一份预演库）：活库、恢复副本都不写
    · 两种提案：按提案接口原样（当前全部 PAPER/ACTIVE 因子）/ 按旧策略的成分
    · 模拟"批准并开始纸交易"（与接口同一组迁移）：旧的 active 被退役（留谱系），**它的 verdict 一个字都不改**（不补哈希）
    · 新规格没有任何前向证据（不继承旧时钟的证据），研究边界没登记时提案的正式结论不通过
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
from tools.research import reproposal_rehearsal as rr


def _legacy_world(tmp_path):
    """一个"活库"：8 个存量 paper 因子 + 一份旧版 active 策略（没有规格、c1 时期批准）。"""
    from app.db.alpha_store import AlphaRecord, AlphaStore
    from app.db.strategy_store import StrategyConfig, StrategyStore
    d = tmp_path / "QuantAgentData"
    d.mkdir()
    db = d / "alphas.db"
    url = f"sqlite:///{db.as_posix()}"
    st = AlphaStore(db_url=url)
    dsls = ["rank(ts_delta(log(close), 60))", "rank(-ts_std(returns, 20))", "rank(-ts_sum(returns, 5))",
            "rank(close / ts_max(close, 60))"]
    with st._Session() as s:                                         # noqa: SLF001
        s.add_all([AlphaRecord(dsl=x, status="paper", hypothesis="[baseline:test]") for x in dsls])
        s.commit()
    st._engine.dispose()                                             # noqa: SLF001
    ss = StrategyStore(db_url=url)
    sid = ss.save(StrategyConfig(factors=["1"], combo_weights={"1": 1.0}, aum=10_000.0, method="ic_weighted",
                                 passed=False, verdict={"passed": False, "reasons": ["旧版"]}, name="legacy-momentum",
                                 status="proposed"))
    with ss._Session() as s:                                         # noqa: SLF001 —— 造一条旧版 active（c1 时期的批准）
        from app.db.strategy_store import StrategyConfigRecord
        s.get(StrategyConfigRecord, sid).status = "active"
        s.commit()
    ss._engine.dispose()                                             # noqa: SLF001
    return db, sid


def _loader(name, start=None, end=None, **kw):
    return SimpleNamespace(data=_make_dataset(n_days=1600, n_tickers=10, seed=3))


def _fp(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def preview(tmp_path):
    live, sid = _legacy_world(tmp_path)
    rep = fm.snapshot_live(live, tmp_path / "preview")
    return live, Path(rep["copy"]), sid


def test_the_rehearsal_writes_neither_the_live_database_nor_the_restored_copy(preview, tmp_path):
    live, copy, _ = preview
    before = (_fp(live), _fp(copy))
    rr.rehearse(copy, dataset="synthetic_test", loader=_loader, actor="rehearsal")
    assert (_fp(live), _fp(copy)) == before


def test_the_legacy_strategy_is_retired_without_a_patched_hash(preview):
    _, copy, sid = preview
    out = rr.rehearse(copy, dataset="synthetic_test", loader=_loader, actor="rehearsal")
    old = out["legacy"][str(sid)]
    assert old["status_before"] == "active" and old["status_after"] == "retired"
    assert old["verdict_unchanged"] is True and old["has_spec_after"] is False


def test_both_proposals_carry_a_spec_and_no_inherited_evidence(preview):
    _, copy, _ = preview
    out = rr.rehearse(copy, dataset="synthetic_test", loader=_loader, actor="rehearsal")
    assert set(out["proposals"]) == {"as_api", "same_composition"}
    for p in out["proposals"].values():
        assert p["spec_hash"] and p["clock_version"] == "c2" and p["forward_evidence_rows"] == 0
        assert p["gates"]["research_boundary"]["passed"] is False, "研究边界没登记：提案的正式结论不能通过"
        assert p["passed"] is False and any("研究边界" in r for r in p["reasons"])
    same = out["proposals"]["same_composition"]
    assert same["factors"] == ["1"] and same["aum"] == 10_000.0


def test_starting_the_new_version_goes_to_paper_running_not_verified(preview):
    _, copy, _ = preview
    out = rr.rehearse(copy, dataset="synthetic_test", loader=_loader, actor="rehearsal")
    started = out["started"]
    assert started["status"] == "paper_running" and started["permissions"]["verified"] is False
    assert started["runtime_spec_mismatches"] == []
    work = sqlite3.connect(out["work_db"])
    try:
        n = work.execute("SELECT count(*) FROM strategy_decisions WHERE decision = 'retire'").fetchone()[0]
    finally:
        work.close()
    assert n == 1


def test_it_refuses_anything_but_a_snapshot_copy(preview):
    live, _, _ = preview
    with pytest.raises(fm.PreviewRefused):
        rr.rehearse(live, dataset="synthetic_test", loader=_loader, actor="rehearsal")


def test_the_report_is_written_next_to_the_preview(preview):
    _, copy, _ = preview
    out = rr.rehearse(copy, dataset="synthetic_test", loader=_loader, actor="rehearsal")
    rep = json.loads(Path(out["report_file"]).read_text(encoding="utf-8"))
    assert rep["writes"] == "rehearsal copy only" and rep["versions"]["clock_version"] == "c2"
