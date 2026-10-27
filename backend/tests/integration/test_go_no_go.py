"""
Go/No-Go 核对单（2026-10-25 审计工作包 D："生成最终 Go/No-Go，不代替用户启动"）。

    · 逐项列证据；缺证据 = 未覆盖（不是通过）
    · 有阻断项 → No-Go；全部满足也只给"可提交你做最后决定" —— 工具永远不给 Go
    · 只读：库按停写窗口复制后读副本
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.ops import go_no_go as g

DATASET = "us_broad_large"


def _settings():
    from app.config import settings
    return settings.model_copy(update={"price_source": "moomoo", "risk_target_vol_ann": 0.0})


@pytest.fixture
def world(tmp_path, monkeypatch):
    from app.core.execution import run_permit as rp
    from app.core.portfolio_manager.executable_spec import build_spec, spec_hash
    from app.db.execution_store import ExecutionStore
    from app.db.strategy_store import StrategyConfig, StrategyStore
    from app.db.trial_ledger import BoundaryStore
    db = tmp_path / "alphas.db"
    url = f"sqlite:///{db.as_posix()}"
    spec = build_spec({"1": "rank(ts_delta(close, 20))"}, {"1": 1.0}, "equal_weight", 1e4, 0.0, settings=_settings())
    ss = StrategyStore(db_url=url)
    sid = ss.save(StrategyConfig(factors=["1"], combo_weights={"1": 1.0}, aum=1e4, status="approved",
                                 verdict={"executable_spec": spec, "spec_hash": spec_hash(spec)}))
    ss.start_running(sid, "paper_running", "r", "kevin")
    bs = BoundaryStore(db_url=url)
    bs.register(DATASET, "2026-10-28", actor="kevin", reason="未来窗口")
    ev = tmp_path / "simulate_acceptance.jsonl"
    ev.write_text("\n".join(json.dumps({"at": "2026-10-30T21:00:00+00:00", "scenario": s, "step": "verdict",
                                        "code_version": "abc", "data": {"passed": True, "checks": []}})
                            for s in g.SCENARIOS) + "\n", encoding="utf-8")
    edb = tmp_path / "exec.db"
    es = ExecutionStore(db_url=f"sqlite:///{edb.as_posix()}")
    rp.issue(es, mode="unattended", actor="kevin", reason="go", account=2001, hours=48, acceptance_evidence=str(ev),
             settings=_settings(), code="abc123abc123", spec_hash=spec_hash(spec), health=lambda s, now: [])
    bk = tmp_path / "bk"
    bk.mkdir()
    (bk / "last_backup.json").write_text(json.dumps({"ok": True, "at": datetime.now(timezone.utc).isoformat()}))
    from app.core.alerting.dispatcher import AlertDispatcher
    events = tmp_path / "events.jsonl"
    d = AlertDispatcher(event_file=events, notifier=lambda t, b: None)
    d.confirm_visible(d.send_test_notification(), actor="kevin")
    monkeypatch.setattr(g, "code_items", lambda: [g._item("固定代码提交", True, "HEAD abc（测试）")])   # noqa: SLF001
    cfg = {"price_source": "moomoo", "execution_mode": "moomoo_paper", "intraday_watchdog_enabled": True,
           "risk_target_vol_ann": 0.0}
    args = dict(db=db, dataset=DATASET, settings=cfg, exec_db=edb, evidence=ev, backup_dir=str(bk),
                alert_events=str(events))
    for s in (ss, es, bs):                                              # 关连接：WAL 合并回主文件
        s._engine.dispose()                                             # noqa: SLF001
    return args, tmp_path


def test_everything_in_place_is_handed_to_you_not_declared_go(world):
    args, _ = world
    rep = g.build(**args)
    assert rep["blockers"] == [], rep["blockers"]
    assert rep["verdict"].startswith("可提交你做最后决定") and rep["verdict"] != "Go"
    assert len(rep["manual_confirmations"]) == len(g.MANUAL) and all(m["confirmed_by"] is None
                                                                    for m in rep["manual_confirmations"])


def test_a_scenario_without_a_verdict_is_not_covered_and_blocks(world):
    args, tmp = world
    ev = Path(args["evidence"])
    lines = [x for x in ev.read_text(encoding="utf-8").splitlines() if "partial_fill_then_cancel" not in x]
    ev.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rep = g.build(**args)
    assert rep["verdict"] == "No-Go" and any("partial_fill_then_cancel" in b and "未覆盖" in b for b in rep["blockers"])


def test_an_unregistered_boundary_blocks(world):
    args, _ = world
    rep = g.build(**{**args, "dataset": "us_tech_large"})
    assert rep["verdict"] == "No-Go" and any("研究边界已登记" in b for b in rep["blockers"])


def test_a_yahoo_price_source_blocks(world):
    args, _ = world
    rep = g.build(**{**args, "settings": {**args["settings"], "price_source": "yahoo"}})
    assert any("价格源" in b for b in rep["blockers"])


def test_no_permit_blocks(world, tmp_path):
    from app.db.execution_store import ExecutionStore
    args, _ = world
    empty = tmp_path / "empty_exec.db"
    ExecutionStore(db_url=f"sqlite:///{empty.as_posix()}")._engine.dispose()     # noqa: SLF001
    rep = g.build(**{**args, "exec_db": empty})
    assert any("无人值守许可" in b for b in rep["blockers"])


def test_the_live_database_is_read_through_a_quiet_window_copy(world):
    from app.tasks.snapshot_manifest import db_files_state
    args, _ = world
    before = db_files_state(args["db"])
    rep = g.build(**args)
    assert db_files_state(args["db"]) == before
    assert any(i["name"] == "读库方式" and "停写窗口" in i["detail"] for i in rep["items"])


def test_the_scenarios_are_the_acceptance_scenarios():
    from tools.ops import simulate_acceptance as sa
    assert set(g.SCENARIOS) == set(sa.SCENARIOS)


def test_the_cli_exits_non_zero_on_no_go(world, capsys):
    args, tmp = world
    cfg = tmp / "settings.json"
    cfg.write_text(json.dumps({**args["settings"], "price_source": "yahoo"}), encoding="utf-8")
    code = g.main(["--db", str(args["db"]), "--dataset", DATASET, "--exec-db", str(args["exec_db"]),
                   "--evidence", str(args["evidence"]), "--backup-dir", args["backup_dir"],
                   "--alert-events", args["alert_events"], "--settings-json", str(cfg)])
    assert code == 2 and "No-Go" in capsys.readouterr().out
