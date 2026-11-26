"""日亏基线需要可验证的前一交易日收盘后采集时间。"""

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.db.execution_store import close_snapshot_eligible
from tests.unit.execution.test_order_manager import D1, D2, _cycle, _env


CLOSE = datetime(2025, 3, 3, 21, 0, tzinfo=timezone.utc)
OPEN = datetime(2025, 3, 4, 14, 30, tzinfo=timezone.utc)
NOW = OPEN + timedelta(hours=1)


@pytest.mark.parametrize("stamp,now,expected", [
    (CLOSE - timedelta(microseconds=1), NOW, False),
    (CLOSE, NOW, True),
    (CLOSE, CLOSE, True),
    (OPEN - timedelta(microseconds=1), NOW, True),
    (OPEN, NOW, False),
    (OPEN - timedelta(minutes=1), OPEN - timedelta(minutes=2), False),
    (None, NOW, False),
])
def test_close_snapshot_boundaries(stamp, now, expected):
    snap = SimpleNamespace(market_date=D1, taken_at=stamp)
    assert close_snapshot_eligible(snap, D2, now) is expected


def test_weekend_and_timezone_aware_snapshot():
    friday = date(2025, 3, 7)
    monday = date(2025, 3, 10)
    snap = SimpleNamespace(market_date=friday,
                           taken_at=datetime(2025, 3, 8, 5, 30, tzinfo=timezone(timedelta(hours=8))))
    assert close_snapshot_eligible(snap, monday, datetime(2025, 3, 10, 14, 0, tzinfo=timezone.utc))


def test_missing_wrong_date_and_naive_timestamps():
    assert not close_snapshot_eligible(None, D2, NOW)
    assert not close_snapshot_eligible(SimpleNamespace(market_date=D2, taken_at=CLOSE), D2, NOW)
    assert close_snapshot_eligible(SimpleNamespace(market_date=D1, taken_at=CLOSE.replace(tzinfo=None)),
                                   D2, NOW.replace(tzinfo=None))


@pytest.mark.parametrize("missing", ["session_close_utc", "session_open_utc"])
def test_missing_session_boundary_is_ineligible(monkeypatch, missing):
    from app.core.data_engine import market_calendar
    monkeypatch.setattr(market_calendar, missing, lambda _: None)
    assert not close_snapshot_eligible(SimpleNamespace(market_date=D1, taken_at=CLOSE), D2, NOW)


def test_snapshot_timestamp_is_recorded_in_utc(tmp_path):
    e = _env(tmp_path)
    e.clock.t = datetime(2025, 3, 4, 6, 30, tzinfo=timezone(timedelta(hours=8)))
    result = e.om.reconcile()
    assert result.return_basis["current_taken_at"] == "2025-03-03T22:30:00"


def test_intraday_prior_snapshot_cannot_authorize_new_risk(tmp_path):
    e = _env(tmp_path)
    e.clock.t = CLOSE - timedelta(minutes=30)
    e.om.reconcile()
    result = _cycle(e, D2)
    assert result.reconcile["day_return"] is None
    assert result.gate["halted"] == "daily_loss_baseline_unavailable"
    assert result.n_submitted == 0


def test_watchdog_marks_an_intraday_prior_snapshot_unproven(tmp_path):
    from tests.integration.test_intraday_watchdog import Clock, OPEN_D3, _watchdog
    e = _env(tmp_path)
    e.clock.after_close(D2, hours=-0.5)
    e.om.reconcile()
    e.ctx.now = "2025-03-05 09:31:00.000"
    watchdog, alerts = _watchdog(e, Clock(OPEN_D3 + timedelta(minutes=5)))
    result = watchdog.tick()
    assert result["status"] == "no_baseline"
    assert any(a.severity == "critical" and "日亏基线" in a.title for a in alerts)
