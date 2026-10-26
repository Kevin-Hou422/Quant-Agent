"""
运行结果的传播：异常拒单、正式记录写入失败、备份失败（2026-10-25 审计 R4）。

审计复现："计划 2 张买单，均因缺报价时刻或独立报价被 gate 拒绝，实际发送 0 张，得到退出码 0 和 run_completed/info。"
要求："按结构化原因区分正常无交易、已存在的重复订单、计划内风险缩减、数据/报价/对账异常和提交结果未知。
异常导致的全部拒绝必须有明确失败/受阻结果和 warning/critical 通知；部分拒绝也要可见。不要反过来将所有正常风控拒绝
都粗暴判成事故。……`daily_backup_job` 在备份失败时只记 error 日志而正常返回；正式证据/归档写入失败不能只留日志
却继续报告'前向成功'。……传播至健康门、退出码及通知。"
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from app.core.alerting.events import (
    GATE_REASON_CLASS,
    events_for_daily_run,
    execution_outcome,
)
from app.tasks.forward import EXIT_OK, EXIT_RUN_DEGRADED, EXIT_RUN_FAILED, classify_run

BACKEND = Path(__file__).resolve().parents[2]


def _out(ex=None, **pf):
    return {"ingest_accepted": True, "portfolio": {"execution": {"mode": "moomoo_paper", "blocked": "", **(ex or {})},
                                                  **pf}}


def _ex(planned, sent, reasons, dup=0, **kw):
    return {"n_planned": planned, "n_submitted": sent, "n_duplicate": dup, "n_gate_rejected": len(reasons),
            "n_submit_errors": 0, "gate": {"rejected": [{"ticker": f"T{i}", "reason": r} for i, r in enumerate(reasons)]},
            **kw}


def _first(out):
    ev = events_for_daily_run(out, run_date="2026-10-26")[0]
    return ev.kind, ev.severity


def test_the_audit_case_all_buys_rejected_for_missing_quotes_is_a_failure():
    out = _out(_ex(2, 0, ["quote_without_time", "no_independent_quote"]))
    code, verdict = classify_run(out)
    assert code == EXIT_RUN_FAILED and "data 2" in verdict
    assert _first(out) == ("orders_rejected_abnormal", "critical")


def test_all_orders_trimmed_by_planned_risk_limits_is_visible_but_not_an_incident():
    out = _out(_ex(2, 0, ["exceeds_name_limit", "insufficient_buying_power"]))
    assert classify_run(out)[0] == EXIT_OK
    assert _first(out) == ("orders_rejected_by_limits", "warning")


def test_a_partial_rejection_for_bad_data_is_degraded_and_warned():
    out = _out(_ex(3, 2, ["stale_quote"]))
    assert classify_run(out)[0] == EXIT_RUN_DEGRADED
    assert _first(out) == ("orders_partially_rejected", "warning")


def test_a_partial_rejection_by_limits_completes_and_says_so():
    out = _out(_ex(3, 2, ["exceeds_gross_limit"]))
    code, verdict = classify_run(out)
    assert code == EXIT_OK and "limits 1" in verdict
    kind, sev = _first(out)
    assert (kind, sev) == ("run_completed", "info")
    assert "limits 1" in events_for_daily_run(out, run_date="2026-10-26")[0].detail


def test_orders_already_sent_earlier_are_a_normal_rerun():
    out = _out({"n_planned": 3, "n_submitted": 0, "n_duplicate": 3, "n_gate_rejected": 0})
    assert classify_run(out)[0] == EXIT_OK and _first(out) == ("orders_already_submitted", "info")


def test_rejections_without_listed_reasons_are_treated_as_abnormal():
    out = _out({"n_planned": 2, "n_submitted": 0, "n_duplicate": 0, "n_gate_rejected": 2})
    assert classify_run(out)[0] == EXIT_RUN_FAILED
    assert execution_outcome(out["portfolio"]["execution"])["rejections"] == {"unknown": 2}


def test_a_kill_switch_rejection_is_abnormal():
    assert execution_outcome(_ex(1, 0, ["kill_switch_engaged"]))["status"] == "failed"


def test_every_reason_the_gate_can_give_is_classified():
    src = (BACKEND / "app" / "core" / "execution" / "pretrade_gate.py").read_text(encoding="utf-8")
    found = set(re.findall(r'undo\("(\w+)"\)', src)) | set(re.findall(r'rejected\.append\(\(o, "(\w+)"\)\)', src))
    found |= set(re.findall(r'return "(\w+)"', src)) | set(re.findall(r'halted = "(\w+)"', src))
    assert len(found) >= 20
    assert sorted(found - set(GATE_REASON_CLASS)) == []


# ── 正式记录写入失败 ─────────────────────────────────────────

def test_a_forward_record_write_failure_fails_the_run_and_raises_critical(monkeypatch, fresh_db):
    import app.db.forward_record as fr
    import app.db.strategy_store as ss
    from app.tasks.daily_trading_loop import DailyTradingLoop
    loop = DailyTradingLoop()

    class _Strategies:
        def get(self, sid):
            return SimpleNamespace(id=sid, verdict="{}", status="paper_running")

    def _broken(*a, **k):
        raise OSError("磁盘已满")
    monkeypatch.setattr(ss, "StrategyStore", lambda *a, **k: _Strategies())
    monkeypatch.setattr(fr, "ForwardRecordStore", _broken)
    monkeypatch.setattr(loop, "_run_portfolio", lambda *a, **k: {
        "active_config": 7, "spec_hash": "h", "execution": _ex(2, 2, [])})
    import pandas as pd
    ds = {"close": pd.DataFrame({"A": [1.0]}, index=pd.bdate_range("2026-10-23", periods=1))}
    result = loop.run_portfolio(ds)
    assert result["record_failures"] == [{"stage": "forward_record", "error": "OSError: 磁盘已满"}]
    out = {"ingest_accepted": True, "portfolio": result}
    code, verdict = classify_run(out)
    assert code == EXIT_RUN_FAILED and "forward_record" in verdict
    kinds = [(e.kind, e.severity) for e in events_for_daily_run(out, run_date="2026-10-26")]
    assert ("record_write_failed", "critical") in kinds


def test_failures_do_not_leak_into_the_next_run(monkeypatch, fresh_db):
    from app.tasks.daily_trading_loop import DailyTradingLoop
    loop = DailyTradingLoop()
    calls = {"n": 0}

    def run(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            loop._note_failure("signal_archive_read", RuntimeError("x"))     # noqa: SLF001
        return {}
    monkeypatch.setattr(loop, "_run_portfolio", run)
    monkeypatch.setattr(loop, "_record_forward_day", lambda *a, **k: None)
    assert len(loop.run_portfolio({})["record_failures"]) == 1
    assert loop.run_portfolio({})["record_failures"] == []


# ── 备份 ─────────────────────────────────────────────────

def test_a_failed_backup_is_recorded_and_blocks_the_health_check(tmp_path):
    from app.tasks.backup import backup_health, run_daily_backup
    res = run_daily_backup(backup_dir=str(tmp_path / "bk"), database_url="sqlite:///" + (tmp_path / "missing.db").as_posix(),
                           scheduler_db_url="sqlite:///" + (tmp_path / "s.db").as_posix(), pit_store_dir=str(tmp_path / "pit"))
    assert res.ok is False
    rec = json.loads((tmp_path / "bk" / "last_backup.json").read_text(encoding="utf-8"))
    assert rec["ok"] is False and rec["errors"]
    ok, detail, blocking = backup_health(str(tmp_path / "bk"))
    assert (ok, blocking) == (False, True) and "失败" in detail


def test_backup_health_distinguishes_never_backed_up_stale_and_fresh(tmp_path):
    from app.tasks.backup import LAST_BACKUP, backup_health
    root = tmp_path / "bk"
    assert backup_health(str(root))[::2] == (False, False)
    root.mkdir()
    now = datetime(2026, 10, 26, 12, tzinfo=timezone.utc)
    (root / LAST_BACKUP).write_text(json.dumps({"ok": True, "at": (now - timedelta(hours=3)).isoformat()}))
    assert backup_health(str(root), now)[0] is True
    assert backup_health(str(root), now + timedelta(days=3))[::2] == (False, True)


def test_the_scheduled_backup_job_raises_a_critical_alert_on_failure(monkeypatch):
    import app.core.alerting.dispatcher as disp
    import app.tasks.backup as bk
    from app.tasks.scheduler import daily_backup_job
    sent = []
    monkeypatch.setattr(bk, "run_daily_backup", lambda: bk.BackupResult(ok=False, snapshot_dir="s", errors=["main_db: x"]))
    monkeypatch.setattr(disp, "emit_safely", lambda ev, d=None: sent.append(ev))
    daily_backup_job()
    assert [(e.kind, e.severity) for e in sent] == [("backup_failed", "critical")]


def test_the_backup_command_exits_non_zero_on_failure(monkeypatch, capsys):
    import app.tasks.backup as bk
    monkeypatch.setattr(bk, "run_daily_backup", lambda: bk.BackupResult(ok=False, errors=["main_db: x"]))
    assert bk.main([]) == 1
    monkeypatch.setattr(bk, "run_daily_backup", lambda: bk.BackupResult(ok=True))
    assert bk.main([]) == 0


def test_the_preflight_blocks_on_a_failed_backup_and_only_warns_before_the_first(tmp_path):
    """健康门：预检读 last_backup.json —— 从没备份过是提醒；最近一次失败是阻塞项。"""
    from app.core.execution.golive import run_preflight
    from tests.unit.execution.test_golive import JOBS, _settings
    s = _settings(tmp_path, backup_dir=str(tmp_path / "bk"))

    def check():
        rep = run_preflight(s, probe=lambda h, p: False, now=datetime.now(timezone.utc), cwd=tmp_path,
                            data_probe=lambda st, t: (True, ""), jobs=lambda: JOBS,
                            strategy_check=lambda st: (True, ""), fundamentals_check=lambda st: (True, "", True))
        return {x.name: x for x in rep.checks}["backup"]
    c = check()
    assert (c.ok, c.blocking) == (False, False)
    (tmp_path / "bk").mkdir()
    (tmp_path / "bk" / "last_backup.json").write_text(json.dumps(
        {"ok": False, "at": datetime.now(timezone.utc).isoformat(), "errors": ["main_db: x"]}))
    c = check()
    assert (c.ok, c.blocking) == (False, True) and "main_db" in c.detail
