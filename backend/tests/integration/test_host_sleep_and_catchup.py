"""
跑调度器的机器会睡着（2026-11-14）。

来由：调度器跑在笔记本上；每日交易排在 21:30 UTC（北京时间 05:30），盘中巡检在美股时段（北京时间夜里）。系统睡眠记录显示
这台机器在一次长任务中途待机了 3 个多小时。睡着时调度器、盘中巡检、日亏熔断都不运行；醒来超过 1 小时补跑宽限的每日任务
被直接跳过 —— 当天不下单，前向记录缺一天，而且没人知道。另外按 UTC 日期判断"今天是不是交易日"，周五的运行拖到 UTC 周六
才补就会被当成"非交易日"跳过。

    · 上线预检 host_power：没接电源 / 接电源时空闲会睡 / 接电源时合盖会睡 / 读不到 → 不通过
    · 调度器运行期间请求系统别因空闲待机（请求跟着线程走：专用线程一直拿着，停的时候撤销）
    · 量出机器实际睡了多久 → 严重告警 + 执行库事件；负载造成的几十秒延迟不算
    · 按时没跑成的每日交易：两次定时运行都过点、下一个交易时段开盘前 15 分钟之前 → 补跑；开盘之后不补，严重告警；
      一个决策日最多跑 CATCHUP_MAX_ATTEMPTS 次
    · 补跑的决策日 = 最近一个已收盘交易日（与摄取 / 执行同一定义），不是 UTC 日期
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from app.core import host_power as hp
from app.core.execution.golive import host_power_check
from app.tasks import scheduler as sch

AC_OK = {"ac": True, "battery_pct": 100, "saver": False}
SET_OK = {"lid_action": 0, "standby_idle_s": 0, "hibernate_idle_s": 0}


# ── 上线预检 ─────────────────────────────────────────────

@pytest.mark.skipif(os.name != "nt", reason="Windows power policy is only enforced on Windows")
@pytest.mark.parametrize("status,power,needle", [
    ({"ac": False, "battery_pct": 40, "saver": False}, SET_OK, "没有接电源（在用电池，剩 40%）"),
    (AC_OK, {**SET_OK, "standby_idle_s": 1800}, "空闲 30 分钟会睡眠"),
    (AC_OK, {**SET_OK, "hibernate_idle_s": 3600}, "空闲 60 分钟会休眠"),
    (AC_OK, {**SET_OK, "lid_action": 1}, "合上盖子会睡眠"),
    ({"ac": None, "battery_pct": None, "saver": None}, SET_OK, "读不到是否接电源"),
    (AC_OK, {**SET_OK, "lid_action": None}, "读不到合盖动作"),
    (AC_OK, {**SET_OK, "hibernate_idle_s": None}, "读不到空闲休眠"),
])
def test_power_settings_that_would_put_the_machine_to_sleep_fail_the_preflight(status, power, needle):
    ok, detail = host_power_check(status, power)
    assert ok is False and needle in detail and "powercfg" in detail


@pytest.mark.skipif(os.name != "nt", reason="Windows power policy is only enforced on Windows")
def test_a_machine_that_stays_awake_passes():
    ok, detail = host_power_check(AC_OK, SET_OK)
    assert ok is True and "合盖不采取任何操作" in detail


@pytest.mark.skipif(os.name != "nt", reason="Windows power policy is only enforced on Windows")
def test_the_preflight_includes_the_power_check_as_blocking(tmp_path):
    from tests.unit.execution.test_golive import _env
    _ctx, _es, _ps, _calls, run = _env(tmp_path)
    from app.core.execution import golive
    rep = run()
    (c,) = [c for c in rep.checks if c.name == "host_power"]
    assert c.ok and c.blocking
    real = golive._read_host_power
    try:
        golive._read_host_power = lambda: (AC_OK, {**SET_OK, "lid_action": 2})
        rep = run()
    finally:
        golive._read_host_power = real
    (c,) = [c for c in rep.checks if c.name == "host_power"]
    assert not c.ok and rep.ready is False


@pytest.mark.skipif(os.name == "nt", reason="Non-Windows behavior")
def test_other_platforms_skip_windows_power_policy():
    ok, detail = host_power_check({}, {})
    assert ok is True and "不是 Windows" in detail


def test_reading_the_real_machine_never_guesses():
    st, ps = hp.power_status(), hp.power_settings()
    assert set(st) == {"ac", "battery_pct", "saver"} and st["ac"] in (True, False, None)
    assert set(ps) == {"lid_action", "standby_idle_s", "hibernate_idle_s"}
    assert all(v is None or (isinstance(v, int) and v >= 0) for v in ps.values())


# ── 防待机与量待机 ───────────────────────────────────────

def test_the_stay_awake_request_is_held_by_one_thread_and_withdrawn_by_it():
    import threading
    calls = []
    ka = hp.KeepAwake(set_state=lambda flags: calls.append((flags, threading.get_ident())) or True).start()
    assert ka.granted and len(calls) == 1 and calls[0][0] == 0x80000001
    ka.stop()
    assert [f for f, _ in calls] == [0x80000001, 0x80000000]
    assert calls[0][1] == calls[1][1] != threading.get_ident(), "请求跟着线程走：同一个专用线程发出与撤销"


def test_sleep_is_measured_and_load_jitter_is_not():
    seen = []
    ticks = iter([0.0, 15.0, 70.0, 85.0, 85.0 + 3 * 3600, 3 * 3600 + 100.0])
    w = hp.SuspendWatch(lambda a, b, s: seen.append(s), interval=15.0, slack=60.0, clock=lambda: next(ticks))
    for _ in range(5):
        w.tick()
    assert seen == [3 * 3600 - 15.0], "晚到 40 秒是负载，不是待机；隔了 3 小时才是"


def test_a_failing_report_does_not_stop_the_watch():
    ticks = iter([0.0, 1000.0, 2000.0])
    attempts = []

    def boom(*a):
        attempts.append(a)
        raise OSError("告警写不进去")
    w = hp.SuspendWatch(boom, interval=15.0, slack=60.0, clock=lambda: next(ticks))
    w.tick()
    w.tick()                                           # 没有抛出
    assert len(attempts) == 2                         # 第二次异常间隔也尝试报告


def test_a_detected_sleep_is_alerted_and_recorded(fresh_db, monkeypatch):
    from app.core.alerting import dispatcher
    from app.db.execution_store import ExecutionStore
    sent = []
    monkeypatch.setattr(dispatcher, "emit_safely", lambda ev, d=None: sent.append(ev))
    a = datetime(2026, 10, 30, 9, 33, tzinfo=timezone.utc)
    sch._on_host_suspended(a, a + timedelta(hours=3, minutes=27), 3 * 3600 + 27 * 60)
    assert [(e.kind, e.severity) for e in sent] == [("host_suspended", "critical")]
    assert "207 分钟" in sent[0].title and "盘中巡检" in sent[0].detail
    (ev,) = ExecutionStore().events_of("host_suspended", 5)
    assert json.loads(ev.payload)["seconds"] == 12420


def test_the_scheduler_holds_the_guards_while_running_and_releases_them(tmp_path, monkeypatch):
    held = []
    monkeypatch.setattr(hp, "_set_execution_state", lambda flags: held.append(flags) or True)
    sch.start_scheduler(db_url=f"sqlite:///{(tmp_path / 'j.db').as_posix()}")
    try:
        kinds = sorted(type(g).__name__ for g in sch._host_guards)
        assert kinds == ["KeepAwake", "SuspendWatch"] and held == [0x80000001]
    finally:
        sch.shutdown_scheduler()
    assert held == [0x80000001, 0x80000000] and sch._host_guards == []


def test_regular_run_and_catchup_cannot_run_concurrently(tmp_path, monkeypatch):
    import threading
    from app.core.execution import order_manager
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(order_manager, "_account_lock_dir", lambda: tmp_path)
    def pipeline(now):
        entered.set()
        assert release.wait(5)
        return {"skipped": "not_a_trading_day"}
    monkeypatch.setattr(sch, "_run_daily_trading", pipeline)
    monkeypatch.setattr(sch, "_record_run", lambda *a: None)
    monkeypatch.setattr(sch, "_emit_daily_outcome", lambda *a: None)
    thread = threading.Thread(target=sch.daily_trading_job)
    thread.start()
    try:
        assert entered.wait(5)
        assert sch.daily_trading_catchup_job() == "busy"
        assert sch.daily_trading_job()["skipped"] == "daily_run_in_progress"
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()


def test_pipeline_lock_failure_is_not_misreported_as_an_overlapping_daily_run(tmp_path, monkeypatch):
    from app.core.execution import order_manager
    from app.db.process_lock import ProcessLockTimeout
    monkeypatch.setattr(order_manager, "_account_lock_dir", lambda: tmp_path)
    def failed_pipeline(now):
        raise ProcessLockTimeout("PIT lock timed out")
    monkeypatch.setattr(sch, "_run_daily_trading", failed_pipeline)
    with pytest.raises(ProcessLockTimeout, match="PIT lock"):
        sch.daily_trading_job()


# ── 按时没跑成的每日交易：补跑 ───────────────────────────

FRI = date(2025, 3, 7)                                   # 周五；下周一 2025-03-10 开盘 13:30 UTC（夏令时已开始）
SAT_01 = datetime(2025, 3, 8, 1, 0, tzinfo=timezone.utc)
MON_DEADLINE = datetime(2025, 3, 10, 13, 15, tzinfo=timezone.utc)


def test_the_decision_window_is_the_last_closed_session_until_just_before_the_next_open():
    assert sch.decision_window(SAT_01) == (FRI, MON_DEADLINE)
    # 周五收盘前（美东 15:59）：决策日还是周四，截止是周五开盘前
    s, dl = sch.decision_window(datetime(2025, 3, 7, 20, 59, tzinfo=timezone.utc))
    assert s == date(2025, 3, 6) and dl == datetime(2025, 3, 7, 14, 15, tzinfo=timezone.utc)


@pytest.fixture
def world(fresh_db, monkeypatch):
    """补跑的周边：告警记下来、每日任务记下调用；运行记录在本用例自己的执行库里。"""
    from app.core.alerting import dispatcher
    alerts, runs = [], []
    monkeypatch.setattr(dispatcher, "emit_safely", lambda ev, d=None: alerts.append(ev))
    monkeypatch.setattr(sch, "daily_trading_job", lambda now=None: runs.append(now) or {})
    return alerts, runs


def _record(s, **kw):
    from app.db.execution_store import ExecutionStore
    ExecutionStore().set_state(sch.RUNS_KEY, json.dumps({str(s): kw}))


def test_nothing_is_done_before_both_regular_runs_had_their_turn(world):
    alerts, runs = world
    assert sch.daily_trading_catchup_job(datetime(2025, 3, 7, 22, 54, tzinfo=timezone.utc)) == "regular_runs_pending"
    assert runs == [] and alerts == []


def test_a_run_missed_while_asleep_is_caught_up_before_the_next_open(world):
    alerts, runs = world
    assert sch.daily_trading_catchup_job(SAT_01) == "ran" and runs == [SAT_01]
    assert [(a.kind, a.severity) for a in alerts] == [("daily_run_catchup", "warning")]
    assert str(MON_DEADLINE.date()) in alerts[0].detail


def test_a_session_already_done_is_not_run_again(world):
    alerts, runs = world
    _record(FRI, done=True, attempts=1)
    assert sch.daily_trading_catchup_job(SAT_01) == "done" and runs == [] and alerts == []


def test_after_the_deadline_it_is_not_run_but_alerted(world):
    alerts, runs = world
    _record(FRI, done=False, attempts=2, outcome="no_new_bar")
    assert sch.daily_trading_catchup_job(MON_DEADLINE) == "missed" and runs == []
    assert [(a.kind, a.severity) for a in alerts] == [("daily_run_missed", "critical")]
    assert "no_new_bar" in alerts[0].detail and "尝试 2 次" in alerts[0].detail
    assert sch.daily_trading_catchup_job(MON_DEADLINE - timedelta(seconds=1)) == "ran", "截止前一秒还补"


def test_it_gives_up_after_the_maximum_attempts(world):
    alerts, runs = world
    _record(FRI, done=False, attempts=sch.CATCHUP_MAX_ATTEMPTS)
    assert sch.daily_trading_catchup_job(SAT_01) == "gave_up" and runs == [] and alerts == []
    _record(FRI, done=False, attempts=sch.CATCHUP_MAX_ATTEMPTS - 1)
    assert sch.daily_trading_catchup_job(SAT_01) == "ran"


def test_a_run_is_recorded_as_done_only_when_the_data_reached_the_session(fresh_db, monkeypatch):
    from app.core.data_engine import pit_store
    from app.db.execution_store import ExecutionStore
    last = {"v": pd.Timestamp("2025-03-06")}
    monkeypatch.setattr(pit_store.PITStore, "latest_timestamp", lambda self, name="pit": last["v"])
    sch._record_run({"ingest_accepted": False, "reason": "no_new_bar"}, SAT_01)       # 数据还停在周四
    rec = sch._runs(ExecutionStore())[str(FRI)]
    assert rec["done"] is False and rec["attempts"] == 1 and rec["data_through"] == "2025-03-06"
    last["v"] = pd.Timestamp("2025-03-07")
    sch._record_run({"skipped": "calendar_unavailable"}, SAT_01)                      # 数据到了，但这次没跑成
    assert sch._runs(ExecutionStore())[str(FRI)]["done"] is False
    sch._record_run({"ingest_accepted": True, "portfolio": {"execution": {"mode": "moomoo_paper"}}}, SAT_01)
    rec = sch._runs(ExecutionStore())[str(FRI)]
    assert rec["done"] is True and rec["attempts"] == 3
    sch._record_run({"ingest_accepted": False, "reason": "no_new_bar"}, SAT_01)       # 跑成之后不会被改回没跑成
    assert sch._runs(ExecutionStore())[str(FRI)]["done"] is True


@pytest.mark.parametrize("out", [
    {"ingest_accepted": False, "reject_reason": "health_failed"},
    {"ingest_accepted": True, "portfolio": {"execution": {"mode": "moomoo_paper", "blocked": "run_permit"}}},
    {"ingest_accepted": True, "portfolio": {"execution": {"mode": "moomoo_paper", "n_submit_errors": 1}}},
    {"ingest_accepted": False, "reason": "no_new_bar"},
    {"ingest_accepted": True, "portfolio": {"execution": {"mode": "moomoo_paper"}, "record_failures": [{"stage": "archive", "error": "failed"}]}},
])
def test_reached_pit_does_not_make_failed_run_done(fresh_db, monkeypatch, out):
    from app.core.data_engine import pit_store
    monkeypatch.setattr(pit_store.PITStore, "latest_timestamp", lambda self, name="pit": pd.Timestamp("2025-03-07"))
    sch._record_run(out, SAT_01)
    assert sch._runs()[str(FRI)]["done"] is False


def test_no_new_bar_with_successful_execution_is_done(fresh_db, monkeypatch):
    from app.core.data_engine import pit_store
    monkeypatch.setattr(pit_store.PITStore, "latest_timestamp", lambda self, name="pit": pd.Timestamp("2025-03-07"))
    sch._record_run({"ingest_accepted": False, "reason": "no_new_bar",
                     "portfolio": {"execution": {"mode": "moomoo_paper"}}}, SAT_01)
    assert sch._runs()[str(FRI)]["done"] is True


def test_the_regular_trigger_on_a_utc_weekend_runs_the_owed_friday_instead_of_skipping(fresh_db, monkeypatch):
    """周五那次拖到 UTC 周六才跑（机器睡过了）：按 UTC 日期是非交易日，但周五还欠着 → 照跑，决策时刻传下去。"""
    import app.tasks.daily_ingest as di
    seen = []
    monkeypatch.setattr(di, "run_daily_pipeline", lambda *a, **kw: seen.append(kw["now"]) or {"ingest_accepted": True})
    out = sch._run_daily_trading(SAT_01)
    assert seen == [SAT_01] and out == {"ingest_accepted": True}
    _record(FRI, done=True)
    assert sch._run_daily_trading(SAT_01)["skipped"] == "not_a_trading_day" and len(seen) == 1
