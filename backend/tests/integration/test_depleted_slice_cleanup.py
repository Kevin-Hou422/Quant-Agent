"""切片耗尽后的在途单清理：隔离券商替身和临时账本。"""

from __future__ import annotations

from app.db.execution_store import ST_CANCELLED, ST_SUBMITTED
from tests.unit.execution.test_order_manager import D1, D2, _cycle, _env, _placed


def _deplete(e):
    e.ctx.account_override = {"total_assets": 900_000.0}


def test_depletion_cancels_rebalance_orders_before_checking_permit(tmp_path):
    e = _env(tmp_path, cash=1_000_000.0)
    _cycle(e, D1)
    original = len(_placed(e))
    _deplete(e)
    def permit_should_not_run(_om):
        raise AssertionError("风险清理应先于运行许可")
    e.om.permit_check = permit_should_not_run
    report = _cycle(e, D2)
    assert report.blocked == "slice_depleted"
    assert len(report.risk_cleanup["requested"]) == 3
    assert all(not report.risk_cleanup[k] for k in ("errors", "unconfirmed", "unknown_submission"))
    assert {r.status for r in e.es.orders()} == {ST_CANCELLED}
    assert len(_placed(e)) == original


def test_cancel_failure_is_recorded_for_each_order_and_does_not_send_new_orders(tmp_path):
    from app.core.alerting.events import events_for_daily_run
    e = _env(tmp_path, cash=1_000_000.0)
    _cycle(e, D1)
    _deplete(e)
    e.ctx.fail["modify_order"] = "cancel refused"
    report = _cycle(e, D2)
    assert report.blocked == "slice_depleted_cleanup_incomplete"
    assert len(report.risk_cleanup["errors"]) == 3 and len(e.ctx.calls_of("modify_order")) == 3
    assert {r.status for r in e.es.orders()} == {ST_SUBMITTED}
    assert len(_placed(e)) == 3
    event = events_for_daily_run({"ingest_accepted": True, "portfolio": {"execution": report.to_dict()}}, "2025-03-04")[0]
    assert event.severity == "critical" and "cancel refused" in event.detail


def test_depletion_cleans_orders_even_when_position_reconciliation_has_a_discrepancy(tmp_path):
    e = _env(tmp_path, cash=1_000_000.0)
    _cycle(e, D1)
    _deplete(e)
    e.ctx.set_position("AAA", 1, 50.0)
    report = _cycle(e, D2)
    assert report.blocked == "slice_depleted_cleanup_incomplete"
    assert len(report.risk_cleanup["requested"]) == 3
    assert report.reconcile["status"] == "discrepancy"
    assert {r.status for r in e.es.orders()} == {ST_CANCELLED}


def test_unconfirmed_cancel_is_not_reported_as_complete(tmp_path):
    e = _env(tmp_path, cash=1_000_000.0)
    _cycle(e, D1)
    _deplete(e)
    e.ctx.async_cancel = True
    e.om.cancel_wait_polls = 1
    report = _cycle(e, D2)
    assert report.blocked == "slice_depleted_cleanup_incomplete"
    assert len(report.risk_cleanup["requested"]) == len(report.risk_cleanup["unconfirmed"]) == 3
    assert len(_placed(e)) == 3


def test_broker_error_while_confirming_cancels_keeps_the_run_blocked(tmp_path, monkeypatch):
    from app.core.execution.broker_gateway import BrokerError
    e = _env(tmp_path, cash=1_000_000.0)
    _cycle(e, D1)
    _deplete(e)
    def fail(ids):
        raise BrokerError("order lookup failed")
    monkeypatch.setattr(e.om, "_await_terminal", fail)
    report = _cycle(e, D2)
    assert report.blocked == "slice_depleted_cleanup_incomplete"
    assert "order lookup failed" in report.risk_cleanup["errors"][0]["error"]
    assert len(_placed(e)) == 3


def test_fills_during_cancel_are_reconciled(tmp_path):
    e = _env(tmp_path, cash=1_000_000.0)
    _cycle(e, D1)
    _deplete(e)
    e.ctx.async_cancel = True
    def fill_then_settle(ctx):
        if ctx.cancel_requested:
            ctx.open_session({"AAA": 50.0}, ratio=0.5)
            ctx.settle_cancels()
    e.ctx.on_order_read = fill_then_settle
    report = _cycle(e, D2)
    assert report.blocked == "slice_depleted"
    assert any(f.ticker == "AAA" and f.qty == 100 for f in e.es.fills())
    assert len(_placed(e)) == 3


def test_slice_can_be_depleted_while_waiting_for_a_normal_cancel(tmp_path):
    e = _env(tmp_path, cash=1_000_000.0)
    _cycle(e, D1)
    e.ctx.async_cancel = True
    def deplete_then_settle(ctx):
        if ctx.cancel_requested:
            _deplete(e)
            ctx.settle_cancels()
    e.ctx.on_order_read = deplete_then_settle
    report = _cycle(e, D2)
    assert report.blocked == "slice_depleted"
    assert len(report.cancelled_stale) == 3 and len(_placed(e)) == 3
    assert report.reconcile["status"] == "clean"


def test_unknown_slice_equity_after_cancel_stops_the_run(tmp_path, monkeypatch):
    e = _env(tmp_path, cash=1_000_000.0)
    _cycle(e, D1)
    e.ctx.async_cancel = True
    original = e.om.slice_equity
    unavailable = {"value": False}
    monkeypatch.setattr(e.om, "slice_equity", lambda total: None if unavailable["value"] else original(total))
    def lose_basis_then_settle(ctx):
        if ctx.cancel_requested:
            unavailable["value"] = True
            ctx.settle_cancels()
    e.ctx.on_order_read = lose_basis_then_settle
    report = _cycle(e, D2)
    assert report.blocked == "slice_equity_unavailable" and len(_placed(e)) == 3


def test_kill_switch_activated_during_cleanup_still_runs_flatten_maintenance(tmp_path):
    e = _env(tmp_path, cash=1_000_000.0)
    _cycle(e, D1)
    _deplete(e)
    e.ctx.async_cancel = True
    def engage_then_settle(ctx):
        if ctx.cancel_requested:
            e.es.set_kill_switch(True, "test", "risk")
            ctx.settle_cancels()
    e.ctx.on_order_read = engage_then_settle
    report = _cycle(e, D2)
    assert report.blocked == "kill_switch_engaged"
    assert report.flatten["status"] == "flat" and e.es.kill_switch()["engaged"]


def test_unconfirmed_submission_is_not_treated_as_cancelled(tmp_path):
    e = _env(tmp_path, cash=1_000_000.0)
    e.ctx.fail["place_order"] = "submit timeout"
    _cycle(e, D1)
    _deplete(e)
    report = _cycle(e, D1)
    assert report.blocked == "slice_depleted_cleanup_incomplete"
    assert len(report.risk_cleanup["unknown_submission"]) == 3
    assert not report.risk_cleanup["requested"]


def test_existing_flatten_order_is_preserved(tmp_path):
    from types import SimpleNamespace
    from app.core.execution.order_builder import build_flatten_orders
    from app.core.execution.order_manager import ExecutionReport, LIVE_BOOK_ID
    e = _env(tmp_path, cash=1_000_000.0)
    e.ctx.set_position("AAA", 100, 50.0)
    _cycle(e, D1)
    order = build_flatten_orders({"AAA": SimpleNamespace(qty=100, nominal_price=50.0)}, "20250303220000",
                                 book_id=LIVE_BOOK_ID, band_bps=500.0).orders[0]
    assert e.om._submit(order, D1, ExecutionReport())["result"] == "submitted"  # noqa: SLF001
    _deplete(e)
    report = _cycle(e, D2)
    assert report.blocked == "slice_depleted"
    assert order.client_id not in report.risk_cleanup["requested"]
    assert e.es.get(order.client_id).status == ST_SUBMITTED
