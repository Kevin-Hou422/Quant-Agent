"""券商镜像账本的日收益与账户快照必须使用同一交易日区间。"""

from datetime import date

import pytest

from app.core.execution.order_manager import LIVE_BOOK_ID
from app.db.position_store import DailyPnL, PositionStore
from tests.unit.execution.test_order_manager import D1, D2, D3, _env


def test_missing_session_is_unknown_and_the_following_session_can_resume(tmp_path):
    e = _env(tmp_path)
    e.clock.after_close(D1)
    e.om.reconcile()
    e.ctx.account_override = {"total_assets": 90_000.0}
    e.clock.after_close(D3)
    report = e.om.reconcile()
    pnl = e.ps.pnl_history(LIVE_BOOK_ID)[-1]
    assert report.day_return is None and pnl.equity == pytest.approx(0.9)
    assert pnl.net_ret is None and pnl.gross_ret is None and pnl.cost_bps is None

    e.ctx.account_override = {"total_assets": 95_000.0}
    e.clock.after_close(date(2025, 3, 6))
    report = e.om.reconcile()
    pnl = e.ps.pnl_history(LIVE_BOOK_ID)[-1]
    assert pnl.net_ret == pytest.approx(95_000 / 90_000 - 1)
    assert pnl.net_ret == pytest.approx(report.day_return) and pnl.equity == pytest.approx(0.95)


def test_missing_first_baseline_is_unknown_and_sql_null_is_preserved(tmp_path):
    e = _env(tmp_path, cash=0.0)
    e.clock.after_close(D1)
    e.om.reconcile()
    e.clock.after_close(D2)
    e.om.reconcile()
    pnl = e.ps.pnl_history(LIVE_BOOK_ID)[-1]
    assert (pnl.equity, pnl.net_ret, pnl.gross_ret, pnl.cost_bps) == (None, None, None, None)


def test_missing_broker_history_keeps_daily_shortfall_unknown(tmp_path):
    e = _env(tmp_path, history_supported=False)
    e.clock.after_close(D1)
    e.om.reconcile()
    e.clock.after_close(D2)
    report = e.om.reconcile()
    pnl = e.ps.pnl_history(LIVE_BOOK_ID)[-1]
    assert report.history_ok is False and pnl.net_ret == pytest.approx(0.0)
    assert pnl.cost_bps is None and pnl.gross_ret is None


def test_negative_slice_equity_is_not_clamped_to_zero(tmp_path):
    e = _env(tmp_path)
    e.clock.after_close(D1)
    e.om.reconcile()
    e.ctx.account_override = {"total_assets": -1_000.0}
    e.clock.after_close(D2)
    report = e.om.reconcile()
    pnl = e.ps.pnl_history(LIVE_BOOK_ID)[-1]
    assert pnl.equity == pytest.approx(-0.01)
    assert pnl.net_ret == pytest.approx(-1.01) and pnl.net_ret == pytest.approx(report.day_return)


def test_unknown_pnl_fields_survive_store_round_trip_and_rerun(tmp_path):
    store = PositionStore(db_url=f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    record = DailyPnL(alpha_id=-1, date=str(D1), gross_ret=None, net_ret=None, cost_bps=None, equity=0.9)
    fill = {"ticker": "AAA", "target_weight": None, "filled_weight": None,
            "fill_price": None, "cost_usd": None, "traded_weight": None, "unfilled_weight": None}
    store.record_day(-1, D1, {"AAA": 0.1}, [fill], record)
    store.record_day(-1, D1, {"AAA": 0.1}, [fill], record)
    pnl = store.pnl_history(-1)
    assert len(pnl) == 1 and pnl[0].equity == 0.9
    assert (pnl[0].net_ret, pnl[0].gross_ret, pnl[0].cost_bps) == (None, None, None)
    assert store.latest_positions(-1) == {"AAA": 0.1}
    stored = store.fills_on(-1, D1)[0]
    assert (stored.target_weight, stored.filled_weight, stored.fill_price,
            stored.cost_usd, stored.traded_weight, stored.unfilled_weight) == (None, None, None, None, None, None)


def test_unknown_weight_still_requires_a_price_for_the_held_security(monkeypatch):
    import pandas as pd
    from app.db import position_store
    from app.tasks.daily_ingest import _held_without_close
    class FakeStore:
        def latest_positions(self, book_id):
            return {"AAA": None}
    monkeypatch.setattr(position_store, "PositionStore", FakeStore)
    data = {"close": pd.DataFrame({"BBB": [10.0]}, index=pd.to_datetime(["2025-03-04"]))}
    assert "AAA" in _held_without_close(data)


def test_dust_weight_at_threshold_does_not_count_as_held(monkeypatch):
    import pandas as pd
    from app.db import position_store
    from app.tasks.daily_ingest import _held_without_close
    class FakeStore:
        def latest_positions(self, book_id):
            return {"AAA": 1e-12}
    monkeypatch.setattr(position_store, "PositionStore", FakeStore)
    data = {"close": pd.DataFrame({"BBB": [10.0]}, index=pd.to_datetime(["2025-03-04"]))}
    assert _held_without_close(data) == ""


def test_zero_previous_equity_does_not_produce_shortfall_rate(tmp_path):
    from app.core.execution.broker_gateway import AccountSnapshot
    e = _env(tmp_path)
    e.clock.after_close(D1)
    e.om.reconcile()
    e.om._record_book(D2, AccountSnapshot(100_000, 100_000, 100_000, 0, "USD"), {},  # noqa: SLF001
                      day_ret=0.0, previous_equity=0.0, history_ok=True)
    assert e.ps.pnl_history(LIVE_BOOK_ID)[-1].cost_bps is None


def test_filled_order_without_current_position_has_unknown_weight_at_zero_equity(tmp_path):
    from app.core.execution.broker_gateway import AccountSnapshot, BrokerOrder
    from app.core.execution.order_builder import PlannedOrder
    e = _env(tmp_path)
    e.clock.after_close(D1)
    e.om.reconcile()
    order = PlannedOrder("qaR-1-20250303-AAA-B", "AAA", "BUY", 1, 50.0, 50.25, 0, 1, 0.01)
    e.es.add_intent(order, LIVE_BOOK_ID, D1)
    e.es.apply_broker_state(order.client_id, BrokerOrder(
        "1", order.client_id, "AAA", "BUY", 1, 50.25, "FILLED_ALL", 1, 50, "", "", ""), D2)
    e.om._record_book(D2, AccountSnapshot(0, 0, 0, 0, "USD"), {},  # noqa: SLF001
                      day_ret=-1.0, previous_equity=100_000.0, history_ok=True)
    fill = e.ps.fills_on(LIVE_BOOK_ID, D2)[0]
    assert fill.filled_weight is None and fill.traded_weight is None


def test_opposing_same_day_fills_do_not_produce_a_misleading_average_price(tmp_path):
    from app.core.execution.broker_gateway import AccountSnapshot, BrokerOrder
    from app.core.execution.order_builder import PlannedOrder
    e = _env(tmp_path)
    e.clock.after_close(D1)
    e.om.reconcile()
    for side, qty, price in (("BUY", 2, 100.0), ("SELL", 1, 110.0)):
        cid = f"qaR-1-20250303-AAA-{side[0]}"
        order = PlannedOrder(cid, "AAA", side, qty, 100.0, 110.0, 0, 0, 0.0)
        e.es.add_intent(order, LIVE_BOOK_ID, D1)
        e.es.apply_broker_state(cid, BrokerOrder(
            side, cid, "AAA", side, qty, 110, "FILLED_ALL", qty, price, "", "", ""), D2)
    e.om._record_book(D2, AccountSnapshot(100_000, 100_000, 100_000, 0, "USD"), {},  # noqa: SLF001
                      day_ret=0.0, previous_equity=100_000.0, history_ok=True)
    fill = e.ps.fills_on(LIVE_BOOK_ID, D2)[0]
    assert fill.fill_price is None and fill.traded_weight == pytest.approx(0.0009)


@pytest.mark.parametrize("quantities", [[1.0], [-1.0], [1.0, 0.0], [-1.0, 0.0]])
def test_one_sided_fills_keep_their_price_and_ignore_zero_quantity(tmp_path, monkeypatch, quantities):
    from types import SimpleNamespace
    from app.core.execution.broker_gateway import AccountSnapshot
    e = _env(tmp_path)
    e.clock.after_close(D1)
    e.om.reconcile()
    fills = [SimpleNamespace(ticker="AAA", qty=q, price=100.0, ref_price=100.0, fill_date=D2) for q in quantities]
    monkeypatch.setattr(e.es, "fills", lambda **kwargs: fills)
    e.om._record_book(D2, AccountSnapshot(100_000, 100_000, 100_000, 0, "USD"), {},  # noqa: SLF001
                      day_ret=0.0, previous_equity=100_000.0, history_ok=True)
    assert e.ps.fills_on(LIVE_BOOK_ID, D2)[0].fill_price == pytest.approx(100.0)


def test_dust_quantity_at_the_threshold_has_no_average_price(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from app.core.execution.broker_gateway import AccountSnapshot, DUST_QTY
    e = _env(tmp_path)
    e.clock.after_close(D1)
    e.om.reconcile()
    fill = SimpleNamespace(ticker="AAA", qty=DUST_QTY, price=100.0, ref_price=100.0, fill_date=D2)
    monkeypatch.setattr(e.es, "fills", lambda **kwargs: [fill])
    e.om._record_book(D2, AccountSnapshot(100_000, 100_000, 100_000, 0, "USD"), {},  # noqa: SLF001
                      day_ret=0.0, previous_equity=100_000.0, history_ok=True)
    assert e.ps.fills_on(LIVE_BOOK_ID, D2)[0].fill_price is None


def test_overflowed_equity_and_position_weight_are_unknown(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from app.core.execution.broker_gateway import AccountSnapshot
    e = _env(tmp_path)
    e.om.paper_aum = 1e-308
    monkeypatch.setattr(e.om, "slice_equity", lambda total: 1e308)
    e.om._record_book(D1, AccountSnapshot(1, 1, 1, 0, "USD"), {},  # noqa: SLF001
                      day_ret=None, previous_equity=None, history_ok=True)
    assert e.ps.latest_equity(LIVE_BOOK_ID) is None
    monkeypatch.setattr(e.om, "slice_equity", lambda total: 1e-308)
    e.om._record_book(D2, AccountSnapshot(1, 1, 1, 0, "USD"), {"AAA": SimpleNamespace(market_val=1e308)},  # noqa: SLF001
                      day_ret=None, previous_equity=None, history_ok=True)
    assert e.ps.latest_positions(LIVE_BOOK_ID) == {"AAA": None}
