"""
test_forward_cli.py — `python -m app.tasks.forward preflight | run-now`

run-now 会真的下纸交易单，所以：预检有阻塞项就**一定不跑**管线，退出码非 0；
跑的必须是调度器用的同一个 `daily_trading_job`，不是另一条路径。
"""

from __future__ import annotations

import json

import pytest

from app.core.execution.golive import PreflightReport
from app.tasks import forward


def _rep(*checks):
    r = PreflightReport(checked_at="2025-03-04T22:00:00+00:00")
    for name, ok, blocking in checks:
        r.add(name, ok, f"{name} detail", blocking)
    return r


READY = _rep(("opend", True, True), ("advice", False, False))
NOT_READY = _rep(("opend", True, True), ("account_funds", False, True))


def test_preflight_exit_codes_follow_blocking_checks(capsys):
    assert forward.preflight(run=lambda: READY) == forward.EXIT_OK
    out = capsys.readouterr().out
    assert "[OK  ] opend" in out and "[WARN] advice" in out and "可以前向交易" in out
    assert forward.preflight(run=lambda: NOT_READY) == forward.EXIT_NOT_READY
    out = capsys.readouterr().out
    assert "[FAIL] account_funds" in out and "未就绪" in out


def test_run_now_refuses_without_a_clean_preflight(capsys):
    called = []
    code = forward.run_now(run=lambda: NOT_READY, job=lambda: called.append(1) or {})
    assert code == forward.EXIT_NOT_READY and called == []
    assert "拒绝执行 run-now" in capsys.readouterr().out


def test_run_now_runs_the_daily_job_and_prints_the_execution_report(capsys):
    out = {"ingest_accepted": True, "mode": "incremental", "forward_from": "2025-03-04",
           "n_new_bars": 1, "portfolio": {"execution": {"n_submitted": 3, "blocked": ""}}}
    assert forward.run_now(run=lambda: READY, job=lambda: out) == forward.EXIT_OK
    text = capsys.readouterr().out
    body = json.loads(text[text.index("{"):])
    assert body["execution"] == {"n_submitted": 3, "blocked": ""}
    assert body["ingest_accepted"] is True and body["forward_from"] == "2025-03-04"


def test_run_now_prints_reasons_readably(capsys):
    """拒单原因是中文；转义成 \\uXXXX 的话，人在终端里看不懂当天为什么没下单。"""
    out = {"portfolio": {"execution": {"blocked": "券商不可用"}}}
    forward.run_now(run=lambda: READY, job=lambda: out)
    assert '"blocked": "券商不可用"' in capsys.readouterr().out


def test_run_now_reports_a_skipped_day(capsys):
    assert forward.run_now(run=lambda: READY,
                           job=lambda: {"skipped": "not_a_trading_day"}) == forward.EXIT_OK
    text = capsys.readouterr().out
    body = json.loads(text[text.index("{"):])
    assert body["skipped"] == "not_a_trading_day" and body["execution"] is None


def test_run_now_defaults_to_the_scheduler_job(monkeypatch):
    import app.tasks.scheduler as sched
    called = []
    monkeypatch.setattr(sched, "daily_trading_job", lambda: called.append(1) or {})
    assert forward.run_now(run=lambda: READY) == forward.EXIT_OK
    assert called == [1]


def test_main_dispatches_and_rejects_unknown_commands(monkeypatch):
    monkeypatch.setattr(forward, "preflight", lambda: 7)
    monkeypatch.setattr(forward, "run_now", lambda: 9)
    assert forward.main(["preflight"]) == 7
    assert forward.main(["run-now"]) == 9
    with pytest.raises(SystemExit):
        forward.main(["trade-real-money"])
