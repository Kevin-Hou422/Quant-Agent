"""
盘中巡检（2026-10-22 用户决定第 17 条）。

用户原话："实现，无人值守前验收。按交易所日历和 DST 调度，频率先用可配置的 60 秒，沿用已批准风险阈值。
共用账户锁和熔断状态，覆盖锁竞争、断线、陈旧快照及恢复，避免并发重复平仓。"

每一拍（tick）：
    不在常规交易时段（交易所日历，含夏令时切换）→ 不连券商
    拿不到账户锁（另一个进程在动这个账户）→ 本拍什么都不做，下一拍再来
    连不上券商 → 记一次；连续几次 → 严重告警；恢复后告警恢复
    快照指纹（总资产 + 各持仓价）长时间不变 → 陈旧快照：不据此做任何决定，告警
    切片相对上一个可信收盘快照的跌幅 ≥ 已批准的日亏阈值（max_daily_loss）→ 拉起共用的熔断开关并全平
    熔断已开（不论谁开的）→ 只续做全平；已有在途全平单时不再下（不重复平仓）
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from tests.unit.execution.test_order_manager import _day2, _env, _placed

OPEN_D3 = datetime(2025, 3, 5, 14, 30, tzinfo=timezone.utc)          # 2025-03-05 美东 09:30（冬令时）


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def _watchdog(e, clock, factory=None, alerts=None, **kw):
    from app.core.execution.intraday_watchdog import IntradayWatchdog
    alerts = alerts if alerts is not None else []
    e.om._clock = clock
    return IntradayWatchdog(make_order_manager=factory or (lambda: e.om), now=clock,
                            alert=alerts.append, **kw), alerts


def _world(tmp_path):
    e = _env(tmp_path)
    _day2(e)                                       # 持仓 AAA 200 / BBB 400 / CCC 500，D2 收盘有可信快照
    e.ctx.now = "2025-03-05 09:31:00.000"
    e.ctx.quote_time = None                        # 盘中行情带当前时刻（新鲜度只认行情时间，审计 R6）
    return e


def _drop(e, pct):
    e.ctx.mark({"AAA": 51.0 * (1 - pct), "BBB": 20.0 * (1 - pct), "CCC": 10.0 * (1 - pct)})


def _sells(e):
    return [p for p in _placed(e) if p[1] == "SELL"]


class TestSchedule:

    @pytest.mark.parametrize("t", [datetime(2025, 3, 5, 14, 0, tzinfo=timezone.utc),       # 开盘前
                                   datetime(2025, 3, 5, 21, 30, tzinfo=timezone.utc),      # 收盘后
                                   datetime(2025, 3, 8, 16, 0, tzinfo=timezone.utc)])      # 周六
    def test_outside_the_regular_session_the_broker_is_not_touched(self, tmp_path, t):
        e = _world(tmp_path)
        called = []
        wd, _ = _watchdog(e, Clock(t), factory=lambda: called.append(1) or e.om)
        assert wd.tick()["status"] == "outside_session" and called == []

    def test_daylight_saving_moves_the_session_in_utc(self, tmp_path):
        """2025-03-10 起夏令时：同样 14:00 UTC，03-07 还没开盘（开盘 14:30），03-10 已开盘（开盘 13:30）。"""
        from app.core.execution.intraday_watchdog import in_regular_session
        assert in_regular_session(datetime(2025, 3, 7, 14, 0, tzinfo=timezone.utc)) is False
        assert in_regular_session(datetime(2025, 3, 10, 14, 0, tzinfo=timezone.utc)) is True

    def test_the_interval_is_configurable_and_registered_only_when_enabled(self, monkeypatch, tmp_path):
        from app.config import settings
        from app.tasks.scheduler import create_scheduler
        monkeypatch.setattr(settings, "intraday_watchdog_enabled", False)
        s = create_scheduler(db_url=f"sqlite:///{(tmp_path / 's1.db').as_posix()}")
        assert s.get_job("intraday_watchdog") is None
        monkeypatch.setattr(settings, "intraday_watchdog_enabled", True)
        monkeypatch.setattr(settings, "intraday_watchdog_interval_s", 45)
        s = create_scheduler(db_url=f"sqlite:///{(tmp_path / 's2.db').as_posix()}")
        job = s.get_job("intraday_watchdog")
        assert job is not None and job.trigger.interval.total_seconds() == 45

    def test_the_calendar_being_unavailable_is_an_alert_not_a_guess(self, tmp_path, monkeypatch):
        import app.core.execution.intraday_watchdog as iw
        from app.core.data_engine.market_calendar import CalendarUnavailable
        e = _world(tmp_path)

        def boom(*a, **k):
            raise CalendarUnavailable("no calendar")
        monkeypatch.setattr(iw, "in_regular_session", boom)
        called = []
        wd, alerts = _watchdog(e, Clock(OPEN_D3 + timedelta(minutes=5)), factory=lambda: called.append(1) or e.om)
        assert wd.tick()["status"] == "calendar_unavailable" and called == []
        assert alerts and alerts[0].severity == "critical"


class TestBreaker:

    def test_a_small_intraday_loss_does_nothing(self, tmp_path):
        e = _world(tmp_path)
        _drop(e, 0.01)
        wd, alerts = _watchdog(e, Clock(OPEN_D3 + timedelta(minutes=5)))
        out = wd.tick()
        assert out["status"] == "ok" and e.es.kill_switch().get("engaged") is False
        assert -0.05 < out["slice_return"] < 0, out                     # 真的算出了一个小幅亏损，没到阈值

    def test_a_loss_beyond_the_approved_threshold_engages_the_shared_switch_and_flattens(self, tmp_path):
        e = _world(tmp_path)
        _drop(e, 0.40)                                   # 持仓约 2.0 万 → 切片亏约 8% > 5%
        wd, alerts = _watchdog(e, Clock(OPEN_D3 + timedelta(minutes=5)))
        out = wd.tick()
        assert out["status"] == "breaker_engaged" and out["slice_return"] <= -0.05
        ks = e.es.kill_switch()
        assert ks["engaged"] is True and ks["actor"] == "intraday_watchdog"
        assert {p[0] for p in _sells(e)} == {"US.AAA", "US.BBB", "US.CCC"}
        assert [(a.kind, a.severity) for a in alerts] == [("intraday_watchdog", "critical")]

    def test_the_threshold_is_the_approved_one_used_by_the_order_gate(self, tmp_path):
        e = _world(tmp_path)
        wd, _ = _watchdog(e, Clock(OPEN_D3 + timedelta(minutes=5)))
        assert wd.threshold() == e.om.limits.max_daily_loss == 0.05

    def test_a_second_tick_does_not_flatten_again(self, tmp_path):
        e = _world(tmp_path)
        _drop(e, 0.40)
        clock = Clock(OPEN_D3 + timedelta(minutes=5))
        wd, _ = _watchdog(e, clock)
        wd.tick()
        n = len(_sells(e))
        clock.t += timedelta(minutes=1)
        out = wd.tick()
        assert out["status"] == "breaker_active" and len(_sells(e)) == n, "重复下了全平单"

    def test_a_switch_engaged_elsewhere_is_continued_not_re_engaged(self, tmp_path):
        e = _world(tmp_path)
        e.es.set_kill_switch(True, "Kevin-Hou422", "手动")
        wd, _ = _watchdog(e, Clock(OPEN_D3 + timedelta(minutes=5)))
        out = wd.tick()
        assert out["status"] == "breaker_active"
        assert e.es.kill_switch()["actor"] == "Kevin-Hou422"
        assert {p[0] for p in _sells(e)} == {"US.AAA", "US.BBB", "US.CCC"}   # 续做全平


class TestFailureModes:

    def test_a_busy_account_lock_means_this_tick_does_nothing(self, tmp_path):
        e = _world(tmp_path)
        _drop(e, 0.40)
        n0 = len(_sells(e))                              # D2 调仓里已有的卖单
        held, release = threading.Event(), threading.Event()

        def hold():
            with e.om._account_lock():
                held.set()
                release.wait(10)
        t = threading.Thread(target=hold)
        t.start()
        try:
            held.wait(10)
            wd, _ = _watchdog(e, Clock(OPEN_D3 + timedelta(minutes=5)))
            out = wd.tick()
            assert out["status"] == "lock_busy"
            assert e.es.kill_switch().get("engaged") is False and len(_sells(e)) == n0
        finally:
            release.set()
            t.join(10)
        assert wd.tick()["status"] == "breaker_engaged", "锁放开之后下一拍要接着处理"

    def test_disconnects_are_counted_alerted_and_recovery_is_reported(self, tmp_path):
        from app.core.execution.broker_gateway import BrokerError
        e = _world(tmp_path)
        up = {"ok": False}

        def factory():
            if not up["ok"]:
                raise BrokerError("OpenD 断线")
            return e.om
        clock = Clock(OPEN_D3 + timedelta(minutes=5))
        wd, alerts = _watchdog(e, clock, factory=factory, unreachable_alert_after=3)
        for i in range(3):
            assert wd.tick()["status"] == "broker_unreachable"
            clock.t += timedelta(minutes=1)
        assert [(a.kind, a.severity) for a in alerts] == [("intraday_watchdog", "critical")]
        up["ok"] = True
        assert wd.tick()["status"] == "ok"
        assert alerts[-1].severity == "info" and "恢复" in alerts[-1].title

    def test_a_frozen_snapshot_is_reported_and_not_acted_on(self, tmp_path):
        """陈旧由**行情时间**判定（审计 R6）：券商那边的行情时间 11 分钟没前进 → 巡检失明，不据此做决定。"""
        e = _world(tmp_path)
        clock = Clock(OPEN_D3 + timedelta(minutes=5))
        wd, alerts = _watchdog(e, clock, stale_after_s=600)
        assert wd.tick()["status"] == "ok"
        clock.t += timedelta(minutes=11)                 # 行情时间停在 09:31
        out = wd.tick()
        assert out["status"] == "stale_snapshot"
        assert alerts and alerts[-1].kind == "intraday_watchdog" and "陈旧" in alerts[-1].title
        _drop(e, 0.40)
        clock.t += timedelta(minutes=1)
        e.ctx.now = "2025-03-05 09:47:00.000"            # 行情恢复更新
        assert wd.tick()["status"] == "breaker_engaged", "行情恢复更新后照常判断"

    def test_no_trusted_close_snapshot_means_no_verdict(self, tmp_path):
        e = _env(tmp_path)                               # 没跑过任何周期：没有可信快照
        e.ctx.now = "2025-03-05 09:31:00.000"
        wd, alerts = _watchdog(e, Clock(OPEN_D3 + timedelta(minutes=5)))
        assert wd.tick()["status"] == "no_baseline"


def test_the_preflight_requires_the_watchdog_for_unattended_running(tmp_path):
    from app.core.execution.golive import run_preflight
    from tests.unit.execution.test_golive import JOBS, _settings
    s = _settings(tmp_path, intraday_watchdog_enabled=False)
    rep = run_preflight(s, probe=lambda h, p: False, now=datetime(2025, 3, 4, 22, 0, tzinfo=timezone.utc),
                        cwd=tmp_path, data_probe=lambda st, t: (True, ""), jobs=lambda: JOBS,
                        strategy_check=lambda st: (True, ""), fundamentals_check=lambda st: (True, "", True))
    c = {x.name: x for x in rep.checks}["intraday_watchdog"]
    assert c.ok is False and c.blocking is True and "盘中巡检" in c.detail


class TestFreshnessIsAboutTimeNotValues:
    """
    2026-10-25 审计 R6："API 每次查询都成功、账户无持仓且总资产稳定，11 分钟后仍被判为 stale_snapshot 并发 critical。
    纯现金、价格短暂不动都不能证明数据源陈旧。"新鲜度只认行情时间；数值不变最多是辅助信号。
    """

    @staticmethod
    def _cash_world(tmp_path):
        from tests.unit.execution.test_order_manager import D2, _cycle
        e = _env(tmp_path)
        _cycle(e, D2, w={"AAA": 0.0})                    # 只建立可信收盘基准，不持仓
        e.ctx.now = "2025-03-05 09:31:00.000"
        e.ctx.quote_time = None
        return e

    def test_a_healthy_cash_only_account_is_never_stale(self, tmp_path):
        e = self._cash_world(tmp_path)
        clock = Clock(OPEN_D3 + timedelta(minutes=5))
        wd, alerts = _watchdog(e, clock, stale_after_s=600)
        for _ in range(3):
            out = wd.tick()
            assert out["status"] == "ok" and out["freshness"] == "no_positions", out
            clock.t += timedelta(minutes=11)
        assert [a for a in alerts if a.severity == "critical"] == []

    def test_static_prices_with_updating_quote_times_are_fresh(self, tmp_path):
        e = _world(tmp_path)
        clock = Clock(OPEN_D3 + timedelta(minutes=5))
        wd, alerts = _watchdog(e, clock, stale_after_s=600)
        for minute in (5, 16, 27):
            clock.t = OPEN_D3 + timedelta(minutes=minute)
            e.ctx.now = f"2025-03-05 09:{30 + minute - 1:02d}:00.000"   # 价格没动，但行情时间在走
            out = wd.tick()
            assert out["status"] == "ok" and out["freshness"] == "fresh", out
        assert alerts == []

    def test_quote_times_that_really_stopped_are_stale(self, tmp_path):
        e = _world(tmp_path)
        e.ctx.quote_time = "2025-03-05 09:31:00.000"
        clock = Clock(OPEN_D3 + timedelta(minutes=25))
        wd, alerts = _watchdog(e, clock, stale_after_s=600)
        _drop(e, 0.40)                                   # 数值变了，但行情时间是 20 分钟前的 —— 不据此拉熔断
        out = wd.tick()
        assert out["status"] == "stale_snapshot" and e.es.kill_switch().get("engaged") is False
        assert [(a.severity, "陈旧" in a.title) for a in alerts] == [("critical", True)]

    def test_without_quote_times_unchanged_values_only_warn_and_the_check_still_runs(self, tmp_path, monkeypatch):
        e = _world(tmp_path)
        from app.core.execution.broker_gateway import Quote
        monkeypatch.setattr(e.gw, "quotes", lambda tks: {tk: Quote(price=1.0, asof=None) for tk in tks})
        clock = Clock(OPEN_D3 + timedelta(minutes=5))
        wd, alerts = _watchdog(e, clock, stale_after_s=600)
        assert wd.tick()["freshness"] == "unproven"
        clock.t += timedelta(minutes=11)
        out = wd.tick()
        assert out["status"] == "ok" and out["freshness"] == "unproven"
        assert [(a.severity, "无法证明" in a.title) for a in alerts] == [("warning", True)]
        _drop(e, 0.40)
        clock.t += timedelta(minutes=1)
        assert wd.tick()["status"] == "breaker_engaged", "证明不了新鲜不等于失明：亏损照常判断"

    def test_one_halted_name_is_a_warning_not_blindness(self, tmp_path, monkeypatch):
        e = _world(tmp_path)
        from datetime import datetime as _dt

        from app.core.execution.broker_gateway import Quote
        def quotes(tks):                                 # AAA 当前；BBB / CCC 停在昨天下午（停牌之类）
            return {tk: Quote(price=1.0, asof=_dt(2025, 3, 5, 9, 31) if tk == "AAA" else _dt(2025, 3, 4, 15, 0))
                    for tk in tks}
        monkeypatch.setattr(e.gw, "quotes", quotes)
        wd, alerts = _watchdog(e, Clock(OPEN_D3 + timedelta(minutes=5)), stale_after_s=600)
        out = wd.tick()
        assert out["status"] == "ok" and out["freshness"] == "fresh"
        assert sorted(a.title for a in alerts) == ["盘中巡检：BBB 的行情时间偏旧", "盘中巡检：CCC 的行情时间偏旧"]
        assert {a.severity for a in alerts} == {"warning"}
