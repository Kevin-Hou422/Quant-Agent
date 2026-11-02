"""
Go/No-Go 核对单（2026-10-25 审计工作包 D："生成最终 Go/No-Go，不代替用户启动"）。

    · 逐项列证据；缺证据 = 未覆盖（不是通过）
    · 有阻断项 → No-Go；全部满足也只给"可提交你做最后决定" —— 工具永远不给 Go
    · 只读：库按停写窗口复制后读副本
    · 复审 21cd90b："分别给出四个结论……不得用其中任何一项代替其他项"；现场验收只认许可绑定的那一份证据
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.ops import go_no_go as g

DATASET = "us_broad_large"
CODE = "abc123abc123"
HEAD = CODE + "0" * 28


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    from tests.integration.test_simulate_acceptance import record_all_scenarios
    return record_all_scenarios(tmp_path_factory.mktemp("acceptance"), code=CODE)


def _settings():
    from app.config import settings
    return settings.model_copy(update={"price_source": "moomoo", "risk_target_vol_ann": 0.0})


@pytest.fixture
def world(tmp_path, monkeypatch, recorded):
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
    from app.core.execution.acceptance_evidence import contract_of_spec
    ev = tmp_path / "acceptance" / "simulate_acceptance.jsonl"
    ev.parent.mkdir()
    shutil.copy(recorded, ev)
    edb = tmp_path / "exec.db"
    es = ExecutionStore(db_url=f"sqlite:///{edb.as_posix()}")
    rp.issue(es, mode="unattended", actor="kevin", reason="go", account=2001, hours=48, acceptance_evidence=str(ev),
             settings=_settings(), code=CODE, spec_hash=spec_hash(spec), health=lambda s, now: [],
             contract=contract_of_spec(spec))
    adb = tmp_path / "acceptance" / "acceptance_exec.db"
    ExecutionStore(db_url=f"sqlite:///{adb.as_posix()}")._engine.dispose()       # noqa: SLF001
    bk = tmp_path / "bk"
    bk.mkdir()
    (bk / "last_backup.json").write_text(json.dumps({"ok": True, "at": datetime.now(timezone.utc).isoformat()}))
    from app.core.alerting.dispatcher import AlertDispatcher
    events = tmp_path / "events.jsonl"
    d = AlertDispatcher(event_file=events, notifier=lambda t, b: None)
    d.confirm_visible(d.send_test_notification(), actor="kevin")
    monkeypatch.setattr(g, "code_items", lambda: [g._item("固定代码提交", True, f"HEAD {HEAD}")])   # noqa: SLF001
    cfg = {"price_source": "moomoo", "execution_mode": "moomoo_paper", "intraday_watchdog_enabled": True,
           "risk_target_vol_ann": 0.0}
    args = dict(db=db, dataset=DATASET, settings=cfg, exec_db=edb, evidence=ev, backup_dir=str(bk),
                alert_events=str(events), acceptance_exec_db=adb)
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


def test_the_four_conclusions_are_separate(world):
    args, _ = world
    rep = g.build(**args)
    c = rep["conclusions"]
    assert set(c) == {"supervised", "paper", "unattended", "verified"}
    assert c["supervised"]["verdict"].startswith("可提交") and c["paper"]["verdict"].startswith("可提交")
    assert c["unattended"]["verdict"].startswith("可提交") and "人工确认" in c["unattended"]["verdict"]
    assert c["verified"]["verdict"] == "不满足"                                  # 刚开始收证据，没有前向天数
    assert any("verified_active" in b for b in c["verified"]["blockers"])


def test_a_scenario_edited_after_issuing_blocks_unattended(world):
    """证据文件在签发之后被改（删掉一个场景）：许可绑定的 sha256 对不上，重算也是未覆盖。"""
    args, tmp = world
    ev = Path(args["evidence"])
    lines = [x for x in ev.read_text(encoding="utf-8").splitlines() if "partial_fill_then_cancel" not in x]
    ev.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rep = g.build(**args)
    assert rep["verdict"] == "No-Go"
    assert any("partial_fill_then_cancel" in b and "未覆盖" in b for b in rep["blockers"])
    assert any("许可绑定的证据文件没变" in b for b in rep["blockers"])
    assert rep["conclusions"]["paper"]["verdict"].startswith("可提交")          # 人工值守的纸交易：列为限制，不阻断


def test_a_report_pointing_at_another_file_than_the_permit_blocks(world, tmp_path):
    """复审："许可与报告引用不同文件"：内容一模一样的另一份文件也不行。"""
    args, _ = world
    other = tmp_path / "copy_of_evidence.jsonl"
    shutil.copy(args["evidence"], other)
    rep = g.build(**{**args, "evidence": other})
    assert rep["verdict"] == "No-Go" and any("就是许可绑定的那一份" in b for b in rep["blockers"])


def test_the_acceptance_db_must_not_be_the_formal_one(world):
    args, _ = world
    rep = g.build(**{**args, "acceptance_exec_db": args["exec_db"]})
    for k in ("supervised", "paper", "unattended"):
        assert any("分开" in b for b in rep["conclusions"][k]["blockers"]), k


def test_a_blind_watchdog_or_an_unrecorded_alert_blocks_unattended_only(world):
    """巡检最近一拍失明、告警事件没写进去（记在执行库）→ 无人值守 No-Go；人工值守的纸交易不受影响。"""
    from app.db.execution_store import ExecutionStore
    args, _ = world
    es = ExecutionStore(db_url=f"sqlite:///{Path(args['exec_db']).as_posix()}")
    es.set_state("intraday_watchdog", json.dumps({"at": "2026-10-30T15:00:00+00:00", "state": "partial",
                                                  "names": {"AAA": "fresh", "BBB": "missing"}}))
    es.set_state("alert_channel_fault", json.dumps({"first_at": "2099-01-01T00:00:00+00:00",
                                                    "last_at": "2099-01-01T00:00:00+00:00", "count": 1,
                                                    "title": "盘中巡检拉起熔断", "error": "PermissionError"}))
    es._engine.dispose()                                                    # noqa: SLF001
    rep = g.build(**args)
    ub = rep["conclusions"]["unattended"]["blockers"]
    assert any("盘中巡检最近一拍" in b and "BBB" in b for b in ub)
    assert any("通知通道" in b and "没能写进事件文件" in b for b in ub)
    assert rep["conclusions"]["paper"]["verdict"].startswith("可提交")


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


def test_a_setting_missing_from_the_export_is_unknown_not_off(world):
    """配置导出里没有 RISK_TARGET_VOL_ANN：不能当作"目标波动已关闭"（以前 None 被判成通过）。"""
    args, _ = world
    cfg = {k: v for k, v in args["settings"].items() if k != "risk_target_vol_ann"}
    rep = g.build(**{**args, "settings": cfg})
    item = next(i for i in rep["items"] if i["name"] == "目标波动首轮关闭")
    assert item["status"] == "unknown" and any("目标波动" in b for b in rep["conclusions"]["paper"]["blockers"])
