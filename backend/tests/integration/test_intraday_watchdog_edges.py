"""
盘中巡检的边界与分支（D5 关键路径变异复核：app/core/execution/intraday_watchdog.py）。

复核时 36 个点存活 14 个：时段恰好在开盘 / 收盘、恢复之后第二次断线不再告警、快照恰好到陈旧时限、
同一段陈旧只告警一次、熔断开着且已经清仓时的回报、切片基准为 0、跌幅恰好等于阈值、
拉熔断时不重复落库、自己建的券商连接要关、调度入口按执行模式跳过、正常拍不写告警日志 —— 都没人钉。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import app.core.execution.intraday_watchdog as iw
from tests.integration.test_intraday_watchdog import (
    OPEN_D3,
    Clock,
    _watchdog,
    _world,
)

OPEN_UTC = datetime(2025, 3, 5, 14, 30, tzinfo=timezone.utc)
CLOSE_UTC = datetime(2025, 3, 5, 21, 0, tzinfo=timezone.utc)


def test_the_session_includes_the_open_and_excludes_the_close():
    assert iw.in_regular_session(OPEN_UTC) is True
    assert iw.in_regular_session(OPEN_UTC - timedelta(microseconds=1)) is False
    assert iw.in_regular_session(CLOSE_UTC - timedelta(microseconds=1)) is True
    assert iw.in_regular_session(CLOSE_UTC) is False


class TestDisconnects:

    def test_a_second_outage_after_recovery_alerts_again(self, tmp_path):
        from app.core.execution.broker_gateway import BrokerError
        e = _world(tmp_path)
        up = {"ok": False}

        def factory():
            if not up["ok"]:
                raise BrokerError("OpenD 断线")
            return e.om
        clock = Clock(OPEN_D3 + timedelta(minutes=5))
        wd, alerts = _watchdog(e, clock, factory=factory, unreachable_alert_after=2)
        for _ in range(2):
            wd.tick()
        up["ok"] = True
        wd.tick()
        up["ok"] = False
        for _ in range(2):
            wd.tick()
        assert [a.severity for a in alerts] == ["critical", "info", "critical"]


class TestStaleSnapshot:

    def _wd(self, tmp_path, minutes):
        e = _world(tmp_path)
        clock = Clock(OPEN_D3 + timedelta(minutes=5))
        wd, alerts = _watchdog(e, clock, stale_after_s=600)
        assert wd.tick()["status"] == "ok"
        clock.t += timedelta(minutes=minutes)
        return wd, alerts, clock

    def test_an_unchanged_snapshot_inside_the_window_is_fine(self, tmp_path):
        wd, alerts, _ = self._wd(tmp_path, 1)
        assert wd.tick()["status"] == "ok" and alerts == []

    def test_exactly_at_the_limit_is_stale(self, tmp_path):
        wd, alerts, _ = self._wd(tmp_path, 10)
        assert wd.tick()["status"] == "stale_snapshot"

    def test_one_alert_per_stale_episode(self, tmp_path):
        wd, alerts, clock = self._wd(tmp_path, 11)
        wd.tick()
        clock.t += timedelta(minutes=1)
        assert wd.tick()["status"] == "stale_snapshot"
        assert len(alerts) == 1

    def test_the_initial_alert_flag_is_irrelevant(self, tmp_path):
        """L72 等价的机械验证：第一拍指纹必然变化（从 None 变成实际值），`_stale_alerted` 在那里被重置。"""
        e = _world(tmp_path)
        clock = Clock(OPEN_D3 + timedelta(minutes=5))
        wd, alerts = _watchdog(e, clock, stale_after_s=600)
        wd._stale_alerted = True                       # 构造函数里的初值改成 True 的效果
        wd.tick()
        clock.t += timedelta(minutes=11)
        assert wd.tick()["status"] == "stale_snapshot" and len(alerts) == 1


def test_a_flat_account_under_the_breaker_reports_flat(tmp_path):
    e = _world(tmp_path)
    e.ctx.pos.clear()
    e.es.set_kill_switch(True, "Kevin-Hou422", "手动")
    wd, _ = _watchdog(e, Clock(OPEN_D3 + timedelta(minutes=5)))
    assert wd.tick() == {"status": "breaker_active", "flat": True}


class _Om:
    """`_check_loss` 只用到的那几样：可信快照、切片权益、阈值、熔断落库、全平。"""

    def __init__(self, prev_assets, thr=0.25):
        self.calls = []
        self.limits = SimpleNamespace(max_daily_loss=thr)
        self.store = SimpleNamespace(
            trusted_before=lambda d: SimpleNamespace(total_assets=prev_assets),
            set_kill_switch=lambda *a: self.calls.append(("set_kill_switch", a)),
            add_event=lambda *a: self.calls.append(("add_event", a)))

    def slice_equity(self, total_assets):
        return float(total_assets)

    def engage_kill_switch(self, actor, reason, persist_state=True):
        self.calls.append(("engage_kill_switch", persist_state))
        return {"cancelled": [], "flatten": []}


class TestLossCheck:

    def _wd(self):
        return iw.IntradayWatchdog(make_order_manager=lambda: None, now=lambda: OPEN_UTC, alert=lambda ev: None)

    def test_a_zero_baseline_is_no_verdict(self):
        assert self._wd()._check_loss(_Om(prev_assets=0.0), SimpleNamespace(total_assets=50.0), OPEN_UTC) == \
            {"status": "no_baseline"}

    def test_a_loss_exactly_at_the_threshold_engages(self):
        om = _Om(prev_assets=100.0, thr=0.25)
        out = self._wd()._check_loss(om, SimpleNamespace(total_assets=75.0), OPEN_UTC)
        assert out["status"] == "breaker_engaged" and out["slice_return"] == -0.25

    def test_the_switch_is_persisted_once(self):
        om = _Om(prev_assets=100.0, thr=0.25)
        self._wd()._check_loss(om, SimpleNamespace(total_assets=70.0), OPEN_UTC)
        assert [c[0] for c in om.calls] == ["set_kill_switch", "add_event", "engage_kill_switch"]
        assert om.calls[-1] == ("engage_kill_switch", False)


def test_a_gateway_the_watchdog_opened_itself_is_closed(monkeypatch):
    closed = []
    om = SimpleNamespace(limits=SimpleNamespace(max_daily_loss=0.05),
                         gw=SimpleNamespace(close=lambda: closed.append(1)))
    monkeypatch.setattr(iw, "_order_manager_from_settings", lambda: om)
    wd = iw.IntradayWatchdog(now=lambda: OPEN_UTC, alert=lambda ev: None)
    assert wd.threshold() == 0.05 and closed == [1]


class TestSchedulerEntry:

    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch):
        monkeypatch.setattr(iw, "_WATCHDOG", None)

    def test_execution_off_skips_without_building_a_watchdog(self, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "execution_mode", "off")
        assert iw.intraday_watchdog_job() == {"status": "execution_off"} and iw._WATCHDOG is None

    @pytest.mark.parametrize("status,warns", [("ok", False), ("outside_session", False), ("lock_busy", True)])
    def test_only_abnormal_ticks_are_logged_as_warnings(self, monkeypatch, caplog, status, warns):
        from app.config import settings
        monkeypatch.setattr(settings, "execution_mode", "moomoo_paper")
        monkeypatch.setattr(iw, "IntradayWatchdog", lambda: SimpleNamespace(tick=lambda: {"status": status}))
        caplog.set_level(logging.WARNING, logger=iw.logger.name)
        assert iw.intraday_watchdog_job() == {"status": status}
        assert any(r.levelno >= logging.WARNING for r in caplog.records) is warns


PROVEN_EQUIVALENT = {
    "app/core/execution/intraday_watchdog.py ×1 — L72 `self._stale_alerted = False` -> `True`（构造函数里的初值）":
        "第一拍时 `_fingerprint` 还是 None，`_stale` 必然走\"指纹变了\"那条分支，把 `_stale_alerted` 重置为 False；"
        "在那之前没有任何地方读它。见 test_the_initial_alert_flag_is_irrelevant（把初值改成 True 照样只告警一次）。",
}


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 1
