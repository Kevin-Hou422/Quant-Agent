"""
告警：桌面通知 + 持久化事件文件（2026-10-22 用户决定 D15）。

用户原话："首版选桌面通知＋持久化事件文件。实现分类、严重度、去重、重试和送达状态，区分正常无订单与异常跳过。
桌面通知须现场确认可见；文件落盘不等于通知成功。不可达时不能称为无人值守就绪。"

送达状态：written（只进了文件）→ handed_to_os（通知交给了操作系统；**不等于有人看见**）/ failed（重试用尽）；
"现场确认可见"只能由人记（confirmed_visible）。无人值守就绪要求：最近有一次现场确认、通知通道最近没有失败。
测试里不弹真实的桌面通知（通知器是注入的）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

T0 = datetime(2026, 10, 26, 22, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


class Notifier:
    """可编排的桌面通知器：fail_first 次失败后成功；记下每次调用。"""

    def __init__(self, fail_first=0, always_fail=False):
        self.fail_first, self.always_fail, self.calls = fail_first, always_fail, []

    def __call__(self, title, body):
        self.calls.append((title, body))
        if self.always_fail or len(self.calls) <= self.fail_first:
            raise OSError("toast unavailable")


def _dispatcher(tmp_path, notifier=None, clock=None, **kw):
    from app.core.alerting.dispatcher import AlertDispatcher
    return AlertDispatcher(event_file=tmp_path / "alerts" / "events.jsonl", notifier=notifier or Notifier(),
                           now=clock or Clock(), sleep=lambda s: None, **kw)


def _ev(kind="ingest_rejected", severity="critical", key="ingest:us_tech_large", title="摄取被拒", detail="健康门"):
    from app.core.alerting.events import AlertEvent
    return AlertEvent(kind=kind, severity=severity, title=title, detail=detail, dedup_key=key)


class TestClassificationAndPersistence:

    def test_an_event_is_persisted_with_category_severity_and_key(self, tmp_path):
        d = _dispatcher(tmp_path)
        rep = d.emit(_ev())
        rows = _dispatcher(tmp_path).history()                      # 新实例从文件读回来
        assert [(r["kind"], r["severity"], r["dedup_key"]) for r in rows] == [
            ("ingest_rejected", "critical", "ingest:us_tech_large")]
        assert rows[0]["event_id"] == rep.event_id and rows[0]["status"] == "handed_to_os"

    def test_unknown_categories_and_severities_are_refused(self):
        from app.core.alerting.events import AlertEvent
        with pytest.raises(ValueError):
            AlertEvent(kind="whatever", severity="critical", title="x", detail="", dedup_key="k")
        with pytest.raises(ValueError):
            AlertEvent(kind="ingest_rejected", severity="loud", title="x", detail="", dedup_key="k")

    def test_info_events_go_to_the_file_only(self, tmp_path):
        n = Notifier()
        rep = _dispatcher(tmp_path, n).emit(_ev(kind="run_completed", severity="info", key="run:2026-10-26"))
        assert rep.status == "file_only" and n.calls == []


class TestDedupAndRetry:

    def test_a_repeat_within_the_window_is_suppressed_not_resent(self, tmp_path):
        n, clock = Notifier(), Clock()
        d = _dispatcher(tmp_path, n, clock, dedup_minutes=60)
        assert d.emit(_ev()).status == "handed_to_os"
        clock.t = T0 + timedelta(minutes=30)
        rep = d.emit(_ev())
        assert rep.status == "suppressed" and len(n.calls) == 1
        clock.t = T0 + timedelta(minutes=61)
        assert d.emit(_ev()).status == "handed_to_os" and len(n.calls) == 2
        assert [r["status"] for r in d.history()] == ["handed_to_os", "suppressed", "handed_to_os"]

    def test_a_failed_delivery_is_retried_and_every_attempt_is_recorded(self, tmp_path):
        n = Notifier(fail_first=2)
        d = _dispatcher(tmp_path, n, retry_attempts=3)
        rep = d.emit(_ev())
        assert rep.status == "handed_to_os" and rep.attempts == 3
        assert [a["result"] for a in d.attempts(rep.event_id)] == ["failed", "failed", "handed_to_os"]

    def test_exhausted_retries_leave_a_failed_status_and_a_later_retry_can_deliver(self, tmp_path):
        bad = Notifier(always_fail=True)
        d = _dispatcher(tmp_path, bad, retry_attempts=2)
        rep = d.emit(_ev())
        assert rep.status == "failed" and len(bad.calls) == 2
        good = Notifier()
        d2 = _dispatcher(tmp_path, good)
        out = d2.retry_failed()
        assert out == {rep.event_id: "handed_to_os"} and len(good.calls) == 1
        assert {r["event_id"]: r["status"] for r in d2.history()}[rep.event_id] == "handed_to_os"

    def test_a_written_file_is_not_a_delivered_notification(self, tmp_path):
        """文件落盘不等于通知成功：通知失败的事件在文件里，但状态是 failed，不是 delivered / handed_to_os。"""
        d = _dispatcher(tmp_path, Notifier(always_fail=True), retry_attempts=1)
        rep = d.emit(_ev())
        row = d.history()[0]
        assert row["event_id"] == rep.event_id and row["status"] == "failed"


class TestUnattendedReadiness:

    def test_without_an_on_site_confirmation_it_is_not_ready(self, tmp_path):
        d = _dispatcher(tmp_path)
        d.send_test_notification()
        ok, why = d.unattended_readiness()
        assert ok is False and "现场确认" in why

    def test_a_confirmed_test_notification_makes_it_ready(self, tmp_path):
        clock = Clock()
        d = _dispatcher(tmp_path, clock=clock)
        tid = d.send_test_notification()
        d.confirm_visible(tid, actor="Kevin-Hou422")
        ok, why = d.unattended_readiness()
        assert ok is True, why
        clock.t = T0 + timedelta(days=15)
        ok, why = d.unattended_readiness()
        assert ok is False and "过期" in why, "很久以前的一次确认不能代表现在还看得见"

    def test_confirming_needs_a_person_and_a_test_that_was_handed_to_the_os(self, tmp_path):
        d = _dispatcher(tmp_path, Notifier(always_fail=True), retry_attempts=1)
        tid = d.send_test_notification()
        with pytest.raises(ValueError):
            d.confirm_visible(tid, actor="Kevin-Hou422")             # 根本没交到系统，谈不上看见
        d2 = _dispatcher(tmp_path)
        tid2 = d2.send_test_notification()
        with pytest.raises(ValueError):
            d2.confirm_visible(tid2, actor="")

    def test_a_recent_delivery_failure_makes_it_not_ready_again(self, tmp_path):
        clock = Clock()
        d = _dispatcher(tmp_path, clock=clock)
        d.confirm_visible(d.send_test_notification(), actor="Kevin-Hou422")
        clock.t = T0 + timedelta(hours=1)
        _dispatcher(tmp_path, Notifier(always_fail=True), clock, retry_attempts=1).emit(_ev(key="other"))
        ok, why = _dispatcher(tmp_path, clock=clock).unattended_readiness()
        assert ok is False and "失败" in why

    def test_the_preflight_reports_the_alert_channel(self, tmp_path, monkeypatch):
        from app.core.execution.golive import run_preflight
        from tests.unit.execution.test_golive import JOBS, _settings
        s = _settings(tmp_path, alert_event_file=str(tmp_path / "alerts" / "events.jsonl"))
        rep = run_preflight(s, probe=lambda h, p: False, now=T0, cwd=tmp_path, data_probe=lambda st, t: (True, ""),
                            jobs=lambda: JOBS, strategy_check=lambda st: (True, ""),
                            fundamentals_check=lambda st: (True, "", True))
        c = {x.name: x for x in rep.checks}["alert_channel"]
        assert c.ok is False and c.blocking is True and "无人值守" in c.detail
        d = _dispatcher(tmp_path)                        # 与 s.alert_event_file 同一个文件
        d.confirm_visible(d.send_test_notification(), actor="Kevin-Hou422")
        rep = run_preflight(s, probe=lambda h, p: False, now=T0, cwd=tmp_path, data_probe=lambda st, t: (True, ""),
                            jobs=lambda: JOBS, strategy_check=lambda st: (True, ""),
                            fundamentals_check=lambda st: (True, "", True))
        assert {x.name: x for x in rep.checks}["alert_channel"].ok is True, "预检读的是传入配置里的事件文件"


class TestDailyRunOutcomes:
    """区分正常无订单与异常跳过。"""

    @pytest.mark.parametrize("out,kind,severity", [
        ({"skipped": "not_a_trading_day", "date": "2026-10-25"}, "no_trading_day", "info"),
        ({"skipped": "calendar_unavailable", "error": "x"}, "run_skipped", "critical"),
        ({"ingest_accepted": False, "reject_reason": "health_gate"}, "ingest_rejected", "critical"),
        ({"ingest_accepted": False, "reason": "no_new_bar"}, "no_new_bar", "warning"),
        ({"ingest_accepted": True, "portfolio": {"error": "boom"}}, "portfolio_failed", "critical"),
        ({"ingest_accepted": True, "portfolio": {"execution": {"mode": "moomoo_paper", "blocked": "kill_switch_engaged"}}},
         "execution_blocked", "critical"),
        ({"ingest_accepted": True, "portfolio": {"execution": {"mode": "moomoo_paper", "n_planned": 3, "n_submitted": 1,
                                                               "n_submit_errors": 2}}}, "order_submit_errors", "critical"),
        ({"ingest_accepted": True, "portfolio": {"execution": {"mode": "moomoo_paper", "n_planned": 0, "n_submitted": 0}}},
         "no_orders_normal", "info"),
        ({"ingest_accepted": True, "portfolio": {"execution": {"mode": "moomoo_paper", "n_planned": 4, "n_submitted": 4}}},
         "run_completed", "info"),
    ])
    def test_each_outcome_has_its_own_category(self, out, kind, severity):
        from app.core.alerting.events import events_for_daily_run
        evs = events_for_daily_run(out, run_date="2026-10-26")
        assert (evs[0].kind, evs[0].severity) == (kind, severity)

    def test_a_normal_no_order_day_and_an_abnormal_skip_never_share_a_category(self):
        from app.core.alerting.events import events_for_daily_run
        normal = events_for_daily_run({"ingest_accepted": True, "portfolio": {"execution": {
            "mode": "moomoo_paper", "n_planned": 0, "n_submitted": 0}}}, run_date="2026-10-26")[0]
        skipped = events_for_daily_run({"ingest_accepted": True, "portfolio": {"execution": {
            "mode": "moomoo_paper", "blocked": "reconcile_discrepancy"}}}, run_date="2026-10-26")[0]
        assert normal.kind != skipped.kind and normal.severity == "info" and skipped.severity == "critical"

    def test_new_corporate_action_flags_raise_their_own_warning(self):
        from app.core.alerting.events import events_for_daily_run
        evs = events_for_daily_run({"ingest_accepted": True, "portfolio": {
            "ca_flags": {"AAA": "2026-10-26"},
            "execution": {"mode": "moomoo_paper", "n_planned": 0, "n_submitted": 0}}}, run_date="2026-10-26")
        assert ("corporate_action_quarantine", "warning") in {(e.kind, e.severity) for e in evs}

    def test_the_daily_job_emits_its_outcome(self, monkeypatch, tmp_path):
        import app.tasks.scheduler as sch
        from app.config import settings
        monkeypatch.setattr(settings, "alert_event_file", str(tmp_path / "alerts" / "events.jsonl"))
        seen = []
        monkeypatch.setattr("app.core.alerting.dispatcher.AlertDispatcher.emit",
                            lambda self, ev: seen.append(ev) or None)
        monkeypatch.setattr("app.core.data_engine.market_calendar.is_trading_day", lambda d: False)
        sch.daily_trading_job()
        assert [e.kind for e in seen] == ["no_trading_day"]


class TestSchedulerFailures:

    def test_a_failed_or_missed_job_is_a_critical_event(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED

        import app.tasks.scheduler as sch
        seen = []
        monkeypatch.setattr("app.core.alerting.dispatcher.AlertDispatcher.emit",
                            lambda self, ev: seen.append(ev) or None)
        sch._on_job_event(SimpleNamespace(code=EVENT_JOB_ERROR, job_id="daily_trading", exception=RuntimeError("x"),
                                          scheduled_run_time=T0))
        sch._on_job_event(SimpleNamespace(code=EVENT_JOB_MISSED, job_id="daily_backup", exception=None,
                                          scheduled_run_time=T0))
        assert [(e.kind, e.severity) for e in seen] == [("scheduled_job_failed", "critical"),
                                                         ("scheduled_job_missed", "critical")]


class TestDesktopNotifier:

    def test_the_message_never_becomes_part_of_the_script(self, monkeypatch):
        """标题 / 正文经环境变量传给 PowerShell，不拼进脚本 —— 带引号与分号的文本不能变成命令。"""
        from app.core.alerting import desktop_notifier as dn
        seen = {}

        def fake_run(args, env=None, **kw):
            seen.update(args=args, env=env)
            from types import SimpleNamespace
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        monkeypatch.setattr(dn.subprocess, "run", fake_run)
        evil = 'x"; Remove-Item C:\\ -Recurse; "'
        dn.show_toast(evil, "body")
        assert all(evil not in a for a in seen["args"])
        assert seen["env"]["QA_TOAST_TITLE"] == evil

    def test_a_non_zero_exit_is_a_failure(self, monkeypatch):
        from types import SimpleNamespace

        from app.core.alerting import desktop_notifier as dn
        monkeypatch.setattr(dn.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="no"))
        with pytest.raises(OSError):
            dn.show_toast("t", "b")


def test_the_suite_never_shows_a_real_desktop_notification():
    """conftest 把真实通知整体替换掉了：分发器缺省的通知器走到的是记录器，不是 PowerShell。"""
    from app.core.alerting import desktop_notifier
    sent = desktop_notifier.notify.sent
    n = len(sent)
    desktop_notifier.notify("t", "b")
    assert len(sent) == n + 1


def test_the_operator_command_runs_the_on_site_confirmation_flow(tmp_path, capsys):
    from app.tasks.alerts import main
    d = _dispatcher(tmp_path)
    assert main(["status"], dispatcher=d) == 1                       # 还没确认过 → 未就绪
    assert main(["test"], dispatcher=d) == 0
    eid = next(e["event_id"] for e in d.history() if e["kind"] == "notification_test")
    assert main(["confirm", eid, "--actor", "Kevin-Hou422"], dispatcher=d) == 0
    assert main(["status"], dispatcher=d) == 0
    assert "就绪" in capsys.readouterr().out


def test_engaging_the_kill_switch_raises_a_critical_alert(fresh_db, test_client, monkeypatch):
    seen = []
    monkeypatch.setattr("app.core.alerting.dispatcher.AlertDispatcher.emit", lambda self, ev: seen.append(ev) or None)
    import app.dependencies as deps
    from app.core.execution.broker_gateway import BrokerError

    def no_broker():
        raise BrokerError("OpenD 没开")
    monkeypatch.setattr(deps, "get_gateway_factory", lambda: no_broker, raising=False)
    tok = test_client.post("/api/execution/kill_switch/arm?action=engage").json()["token"]
    test_client.post("/api/execution/kill_switch/confirm", json={"token": tok, "actor": "Kevin-Hou422", "reason": "测试"})
    assert [(e.kind, e.severity) for e in seen] == [("kill_switch", "critical")]
