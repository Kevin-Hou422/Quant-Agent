"""
execution_store.py —— 执行账本的 schema 与查询契约（Phase 12）

这张账本决定"一张单发没发过"（幂等）、"一笔成交记没记过"（不重不漏）、"熔断开没开"。
列约束写错不会报错，只会让账本悄悄接受坏数据：
  · client_id 不唯一 → 同一决策日同一标的可以下两张单；
  · broker_order_id 不可空 → 写前日志（下单之前先记意图）根本写不进去；
  · (client_id, cum_qty) 不唯一 → 重复对账会把同一笔成交记两次。
本文件把这些写成断言。表结构用**显式期望表**逐列核对，而不是"看起来对"。
"""
from __future__ import annotations

import json
import threading
from datetime import date

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from app.core.execution.broker_gateway import AccountSnapshot, BrokerOrder, TERMINAL_STATUSES
from app.core.execution.order_builder import PlannedOrder
from app.db.execution_store import (
    ST_CANCELLED, ST_FAILED, ST_FILLED, ST_GATE, ST_NOT_SUBMITTED, ST_PARTIAL, ST_PENDING,
    ST_SUBMITTED, ExecFill, ExecOrder, ExecutionStore, local_status_from_broker,
)

D1, D2, D3 = date(2025, 3, 3), date(2025, 3, 4), date(2025, 3, 5)


@pytest.fixture
def store(tmp_path) -> ExecutionStore:
    return ExecutionStore(db_url=f"sqlite:///{tmp_path / 'e.db'}")


def _po(tk="AAA", side="BUY", qty=10, cid=None, purpose="rebalance"):
    return PlannedOrder(cid or f"qaR-1-20250303-{tk}-{side[0]}", tk, side, qty, 50.0,
                        50.25 if side == "BUY" else 49.75, 0.0, float(qty), 0.05, purpose)


def _bo(cid, status="FILLED_ALL", dealt=10.0, avg=50.1, oid="9001", side="BUY", tk="AAA"):
    return BrokerOrder(oid, cid, tk, side, 10.0, 50.25, status, dealt, avg, "", "", "")


def _acct(total=100_000.0):
    return AccountSnapshot(total_assets=total, cash=total, power=total, market_val=0.0,
                           currency="USD")


# ===========================================================================
# A. 表结构：逐列的非空、主键、唯一、索引
# ===========================================================================

#: 列名 → 是否允许为空。**全表全列**，新增/改动列都必须改这张表。
NULLABLE = {
    "exec_orders": {
        "id": False, "client_id": False, "book_id": False, "decision_date": False,
        "purpose": False, "ticker": False, "side": False, "qty": False, "ref_price": False,
        "limit_price": False, "target_weight": True, "status": False,
        "broker_order_id": True, "broker_status": True, "dealt_qty": True,
        "dealt_avg_price": True, "reject_reason": True, "last_err_msg": True,
        "created_at": True, "updated_at": True},
    "exec_fills": {
        "id": False, "client_id": False, "broker_order_id": False, "book_id": False,
        "decision_date": False, "fill_date": False, "ticker": False, "side": False,
        "qty": False, "price": False, "ref_price": False, "cum_qty": False,
        "snapshot_seq": False, "recorded_at": True},
    "exec_snapshots": {
        "id": False, "taken_at": True, "market_date": False, "total_assets": False,
        "cash": False, "power": False, "market_val": True, "positions_json": True,
        "discrepancies_json": True, "status": False, "note": True},
    "exec_events": {"id": False, "at": True, "kind": False, "actor": True, "reason": True,
                    "payload": True},
    "exec_state": {"key": False, "value": True, "updated_at": True},
}
PRIMARY_KEY = {"exec_orders": ["id"], "exec_fills": ["id"], "exec_snapshots": ["id"],
               "exec_events": ["id"], "exec_state": ["key"]}
INDEXED = {
    "exec_orders": {"book_id", "decision_date", "status", "broker_order_id"},
    "exec_fills": {"client_id", "book_id", "decision_date", "fill_date"},
    "exec_snapshots": {"taken_at", "market_date"},
    "exec_events": {"at"},
}


class TestSchema:

    def test_every_column_has_the_documented_nullability(self, store):
        insp = inspect(store._engine)
        for table, cols in NULLABLE.items():
            actual = {c["name"]: c["nullable"] for c in insp.get_columns(table)}
            assert set(actual) == set(cols), f"{table} 的列与期望表不一致：{set(actual) ^ set(cols)}"
            wrong = {k: v for k, v in actual.items() if v is not cols[k]}
            assert not wrong, f"{table} 可空性与期望不符：{wrong}"

    def test_primary_keys(self, store):
        insp = inspect(store._engine)
        for table, cols in PRIMARY_KEY.items():
            assert insp.get_pk_constraint(table)["constrained_columns"] == cols

    def test_lookup_columns_are_indexed(self, store):
        insp = inspect(store._engine)
        for table, cols in INDEXED.items():
            got = {c for ix in insp.get_indexes(table) for c in ix["column_names"]}
            assert cols <= got, f"{table} 缺索引：{cols - got}"

    def test_uniqueness_that_idempotency_depends_on(self, store):
        insp = inspect(store._engine)
        uq_orders = [u["column_names"] for u in insp.get_unique_constraints("exec_orders")]
        uq_orders += [ix["column_names"] for ix in insp.get_indexes("exec_orders") if ix["unique"]]
        assert ["client_id"] in uq_orders, "client_id 不唯一 —— 同一张单可以记两次"
        uq_fills = {u["name"]: u["column_names"] for u in insp.get_unique_constraints("exec_fills")}
        assert uq_fills.get("uq_exec_fill") == ["client_id", "cum_qty"]

    def test_duplicate_client_id_is_rejected_by_the_database(self, store):
        with store._Session() as s:
            for _ in range(2):
                s.add(ExecOrder(client_id="dup", book_id=-1, decision_date=D1,
                                purpose="rebalance", ticker="A", side="BUY", qty=1,
                                ref_price=1.0, limit_price=1.0, status=ST_PENDING))
            with pytest.raises(IntegrityError):
                s.commit()

    def test_duplicate_fill_increment_is_rejected_by_the_database(self, store):
        row = dict(client_id="c", broker_order_id="1", book_id=-1, decision_date=D1,
                   fill_date=D2, ticker="A", side="BUY", qty=5.0, price=1.0, ref_price=1.0,
                   cum_qty=5.0, snapshot_seq=0)
        with store._Session() as s:
            s.add(ExecFill(**row))
            s.add(ExecFill(**row))
            with pytest.raises(IntegrityError):
                s.commit()

    def test_null_identity_is_rejected_by_the_database(self, store):
        with store._Session() as s:
            s.add(ExecOrder(client_id="x", book_id=None, decision_date=D1, purpose="rebalance",
                            ticker="A", side="BUY", qty=1, ref_price=1.0, limit_price=1.0,
                            status=ST_PENDING))
            with pytest.raises(IntegrityError):
                s.commit()


class TestEngineConfiguration:

    def test_cross_thread_use(self, tmp_path):
        st = ExecutionStore(db_url=f"sqlite:///{tmp_path / 't.db'}")
        st.add_intent(_po(), -1, D1)
        box = {}

        def _read():
            try:
                box["v"] = st.get(_po().client_id).status
            except Exception as exc:          # noqa: BLE001 —— 带回主线程断言
                box["err"] = exc
        t = threading.Thread(target=_read)
        t.start()
        t.join()
        assert "err" not in box, f"跨线程读取失败：{box.get('err')}"
        assert box["v"] == ST_PENDING

    def test_sql_is_not_echoed(self, store):
        assert store._engine.echo is False

    def test_returned_objects_stay_readable_after_commit(self, store):
        snap = store.add_snapshot(market_date=D1, account=_acct(), positions={},
                                  discrepancies=[], status="baseline")
        assert snap.id >= 1 and snap.status == "baseline"     # session 已关闭，仍须可读

    def test_env_url_wins_over_settings(self, tmp_path, monkeypatch):
        target = f"sqlite:///{tmp_path / 'env.db'}"
        monkeypatch.setenv("DATABASE_URL", target)
        assert str(ExecutionStore()._engine.url) == target

    def test_settings_url_used_when_env_empty(self, tmp_path, monkeypatch):
        from app.config import settings
        monkeypatch.setenv("DATABASE_URL", "")
        target = f"sqlite:///{tmp_path / 'cfg.db'}"
        monkeypatch.setattr(settings, "database_url", target)
        assert str(ExecutionStore()._engine.url) == target


# ===========================================================================
# B. 订单：写意图 / 幂等 / 可重试
# ===========================================================================

class TestOrders:

    def test_intent_is_written_before_any_broker_id_exists(self, store):
        assert store.add_intent(_po(), -1, D1) is True
        row = store.get(_po().client_id)
        assert (row.status, row.broker_order_id, row.book_id, row.decision_date) == (
            ST_PENDING, None, -1, D1)
        assert (row.qty, row.ref_price, row.limit_price, row.target_weight) == (10, 50.0, 50.25, 0.05)

    @pytest.mark.parametrize("status", [ST_PENDING, ST_SUBMITTED, ST_FILLED, ST_PARTIAL,
                                        ST_CANCELLED, ST_FAILED])
    def test_orders_that_may_have_reached_the_broker_are_final(self, store, status):
        store.add_intent(_po(), -1, D1)
        with store._Session() as s:
            s.query(ExecOrder).update({"status": status})
            s.commit()
        assert store.add_intent(_po(qty=99), -1, D1) is False
        assert store.record_gate_rejection(_po(qty=99), -1, D1, "x") is False
        assert store.get(_po().client_id).qty == 10, "已发出的单被覆盖了"
        assert store.is_final_for_day(_po().client_id) is True

    @pytest.mark.parametrize("status", [ST_GATE, ST_NOT_SUBMITTED])
    def test_orders_that_never_reached_the_broker_can_be_retried(self, store, status):
        store.add_intent(_po(), -1, D1)
        with store._Session() as s:
            s.query(ExecOrder).update({"status": status, "broker_order_id": "old",
                                       "dealt_qty": 3.0, "reject_reason": "no_adv",
                                       "last_err_msg": "boom"})
            s.commit()
        assert store.is_final_for_day(_po().client_id) is False
        assert store.add_intent(_po(qty=12), -1, D2) is True
        row = store.get(_po().client_id)
        assert (row.status, row.qty, row.decision_date, row.broker_order_id, row.dealt_qty,
                row.reject_reason, row.last_err_msg) == (ST_PENDING, 12, D2, None, 0.0, "", "")

    def test_unknown_client_id_is_not_final(self, store):
        assert store.is_final_for_day("nope") is False and store.get("nope") is None

    def test_gate_rejection_is_recorded_with_reason(self, store):
        assert store.record_gate_rejection(_po(), -1, D1, "no_adv") is True
        row = store.get(_po().client_id)
        assert (row.status, row.reject_reason) == (ST_GATE, "no_adv")

    def test_submit_lifecycle(self, store):
        cid = _po().client_id
        store.add_intent(_po(), -1, D1)
        store.note_submit_error(cid, "timeout")
        r = store.get(cid)
        assert (r.status, r.last_err_msg) == (ST_PENDING, "timeout")
        store.mark_submitted(cid, 9001)
        r = store.get(cid)
        assert (r.status, r.broker_order_id) == (ST_SUBMITTED, "9001")
        store.mark_not_submitted(cid, "absent")
        assert store.get(cid).status == ST_NOT_SUBMITTED

    def test_open_orders_filters_status_and_book(self, store):
        store.add_intent(_po("AAA"), -1, D1)                        # PENDING
        store.add_intent(_po("BBB"), -1, D1)
        store.mark_submitted(_po("BBB").client_id, "1")             # SUBMITTED
        store.record_gate_rejection(_po("CCC"), -1, D1, "x")        # 不在途
        store.add_intent(_po("DDD", cid="qaR7-20250303-DDD-B"), 7, D1)   # 别的账本
        assert [r.ticker for r in store.open_orders(-1)] == ["AAA", "BBB"]
        assert [r.ticker for r in store.open_orders()] == ["AAA", "BBB", "DDD"]

    def test_orders_since_is_inclusive_and_newest_first(self, store):
        store.add_intent(_po("AAA", cid="a"), -1, D1)
        store.add_intent(_po("BBB", cid="b"), -1, D2)
        store.add_intent(_po("CCC", cid="c"), -1, D3)
        assert [r.client_id for r in store.orders(since=D2)] == ["c", "b"]
        assert [r.client_id for r in store.orders(limit=2)] == ["c", "b"]

    def test_adopt_creates_a_row_once(self, store):
        bo = _bo("qaR-1-20250303-AAA-B", status="SUBMITTED", dealt=0.0, avg=0.0)
        store.adopt_broker_order(bo, -1, D1, "rebalance", ref_price=0.0)
        store.adopt_broker_order(bo, -1, D2, "flatten", ref_price=9.9)   # 已存在 → 不覆盖
        r = store.get(bo.client_id)
        assert (r.status, r.broker_order_id, r.decision_date, r.purpose, r.last_err_msg) == (
            ST_SUBMITTED, "9001", D1, "rebalance", "adopted_from_broker")


# ===========================================================================
# C. 成交增量
# ===========================================================================

class TestFills:

    def _submitted(self, store, side="BUY"):
        po = _po(side=side)
        store.add_intent(po, -1, D1)
        store.mark_submitted(po.client_id, "9001")
        return po.client_id

    def test_dust_level_changes_are_not_fills(self, store):
        from app.core.execution.broker_gateway import DUST_QTY
        cid = self._submitted(store)
        assert store.apply_broker_state(cid, _bo(cid, "SUBMITTED", DUST_QTY, 50.0), D2) is None
        assert store.fills() == []
        inc = store.apply_broker_state(cid, _bo(cid, "SUBMITTED", 1 + DUST_QTY, 50.0), D2)
        assert inc is not None and inc.qty == pytest.approx(1.0)

    def test_no_change_means_no_increment(self, store):
        cid = self._submitted(store)
        assert store.apply_broker_state(cid, _bo(cid, "SUBMITTED", 0.0, 0.0), D2) is None
        assert store.fills() == [] and store.get(cid).status == ST_SUBMITTED

    def test_increments_are_differenced_from_cumulative_state(self, store):
        cid = self._submitted(store)
        a = store.apply_broker_state(cid, _bo(cid, "FILLED_PART", 4.0, 50.0), D2)
        b = store.apply_broker_state(cid, _bo(cid, "FILLED_ALL", 10.0, 50.3), D2)
        assert (a.qty, a.price) == (4.0, pytest.approx(50.0))
        # (50.3×10 − 50×4) / 6 = 50.5
        assert (b.qty, b.price, b.ref_price, b.fill_date, b.decision_date) == (
            6.0, pytest.approx(50.5), 50.0, D2, D1)
        rows = store.fills()
        assert [(f.qty, f.cum_qty) for f in rows] == [(4.0, 4.0), (6.0, 10.0)]
        r = store.get(cid)
        assert (r.status, r.dealt_qty, r.dealt_avg_price, r.broker_status) == (
            ST_FILLED, 10.0, 50.3, "FILLED_ALL")

    def test_sell_increments_are_negative(self, store):
        cid = self._submitted(store, side="SELL")
        inc = store.apply_broker_state(cid, _bo(cid, "FILLED_ALL", 10.0, 49.9, side="SELL"), D2)
        assert inc.qty == -10.0 and store.fills()[0].qty == -10.0

    def test_fill_reversal_is_a_negative_increment_at_the_old_average(self, store):
        cid = self._submitted(store)
        store.apply_broker_state(cid, _bo(cid, "FILLED_ALL", 10.0, 50.2), D2)
        inc = store.apply_broker_state(cid, _bo(cid, "FILL_CANCELLED", 7.0, 50.2), D3)
        assert (inc.qty, inc.price) == (-3.0, pytest.approx(50.2))

    def test_broker_error_message_is_kept(self, store):
        cid = self._submitted(store)
        bo = BrokerOrder("9001", cid, "AAA", "BUY", 10, 50.25, "FAILED", 0.0, 0.0, "", "", "rejected: halt")
        store.apply_broker_state(cid, bo, D2)
        r = store.get(cid)
        assert (r.status, r.last_err_msg) == (ST_FAILED, "rejected: halt")

    def test_fills_filters(self, store):
        cid = self._submitted(store)
        store.apply_broker_state(cid, _bo(cid, "FILLED_ALL", 10.0, 50.0), D2)
        po = _po("BBB", cid="qaR7-20250304-BBB-B")
        store.add_intent(po, 7, D2)
        store.apply_broker_state(po.client_id, _bo(po.client_id, "FILLED_ALL", 10.0, 1.0,
                                                   tk="BBB"), D3)
        assert [f.ticker for f in store.fills(book_id=-1)] == ["AAA"]
        assert [f.ticker for f in store.fills(start=D2)] == ["BBB"]      # 按决策日
        assert [f.ticker for f in store.fills(end=D1)] == ["AAA"]
        assert [f.ticker for f in store.fills(start=D1, end=D1)] == ["AAA"]

    def test_fills_after_snapshot_uses_the_snapshot_sequence(self, store):
        cid = self._submitted(store)
        store.apply_broker_state(cid, _bo(cid, "FILLED_PART", 4.0, 50.0), D2)    # seq 0
        s1 = store.add_snapshot(market_date=D2, account=_acct(), positions={},
                                discrepancies=[], status="clean")
        store.apply_broker_state(cid, _bo(cid, "FILLED_ALL", 10.0, 50.0), D3)    # seq s1.id
        assert [f.qty for f in store.fills_after_snapshot(s1.id, -1)] == [6.0]
        assert [f.qty for f in store.fills_after_snapshot(0, -1)] == [4.0, 6.0]
        assert store.fills_after_snapshot(s1.id, 7) == []


@pytest.mark.parametrize("status,dealt,expected", [
    ("SUBMITTED", 0.0, ST_SUBMITTED), ("FILLED_PART", 3.0, ST_SUBMITTED),
    ("N/A", 0.0, ST_SUBMITTED), ("FILLED_ALL", 10.0, ST_FILLED),
    ("CANCELLED_PART", 3.0, ST_PARTIAL), ("CANCELLED_ALL", 0.0, ST_CANCELLED),
    ("FILL_CANCELLED", 0.0, ST_CANCELLED), ("FILL_CANCELLED", 2.0, ST_PARTIAL),
    ("FAILED", 0.0, ST_FAILED), ("SUBMIT_FAILED", 0.0, ST_FAILED),
    ("TIMEOUT", 0.0, ST_FAILED), ("DELETED", 0.0, ST_FAILED), ("DISABLED", 0.0, ST_FAILED),
])
def test_local_status_mapping(status, dealt, expected):
    assert local_status_from_broker(status, dealt) == expected


def test_dust_dealt_quantity_counts_as_unfilled():
    from app.core.execution.broker_gateway import DUST_QTY
    assert local_status_from_broker("CANCELLED_PART", DUST_QTY) == ST_CANCELLED
    assert local_status_from_broker("CANCELLED_PART", 2 * DUST_QTY) == ST_PARTIAL


def test_every_terminal_broker_status_maps_to_a_terminal_local_status():
    for s in TERMINAL_STATUSES:
        assert local_status_from_broker(s, 0.0) != ST_SUBMITTED, s


# ===========================================================================
# D. 快照 / 状态 / 事件
# ===========================================================================

class TestSnapshotsAndState:

    def test_trusted_lookups(self, store):
        for d, st in ((D1, "baseline"), (D2, "discrepancy"), (D2, "accepted"), (D3, "unstable")):
            store.add_snapshot(market_date=d, account=_acct(), positions={},
                               discrepancies=[], status=st)
        assert store.latest_snapshot().status == "unstable"
        assert store.latest_snapshot(trusted_only=True).status == "accepted"
        assert store.trusted_before(D3).status == "accepted"
        assert store.trusted_before(D2).status == "baseline", "必须严格早于该日"
        assert store.trusted_before(D1) is None
        assert [s.status for s in store.snapshots(limit=2)] == ["unstable", "accepted"]

    def test_snapshot_text_is_stable_and_human_readable(self, store):
        a = store.add_snapshot(market_date=D1, account=_acct(),
                               positions={"BBB": {"qty": 1}, "AAA": {"qty": 2}},
                               discrepancies=[{"ticker": "AAA", "原因": "拆股"}], status="discrepancy")
        b = store.add_snapshot(market_date=D1, account=_acct(),
                               positions={"AAA": {"qty": 2}, "BBB": {"qty": 1}},
                               discrepancies=[], status="clean")
        assert a.positions_json == b.positions_json, "同一持仓写出了两种文本（审计比对失效）"
        assert a.positions_json.index("AAA") < a.positions_json.index("BBB")
        with store._engine.connect() as c:
            raw = c.execute(text("SELECT discrepancies_json FROM exec_snapshots WHERE id=:i"),
                            {"i": a.id}).scalar()
        assert "拆股" in raw, "审计字段被转义成 \\uXXXX，人读不了"
        assert json.loads(raw) == [{"ticker": "AAA", "原因": "拆股"}]

    def test_kill_switch_state(self, store):
        assert store.kill_switch() == {"engaged": False}
        st = store.set_kill_switch(True, "kevin", "演练")
        assert store.kill_switch() == st and st["engaged"] is True and st["actor"] == "kevin"
        with store._engine.connect() as c:
            raw = c.execute(text("SELECT value FROM exec_state WHERE key='kill_switch'")).scalar()
        assert "演练" in raw
        store.set_kill_switch(False, "kevin", "结束")
        assert store.kill_switch()["engaged"] is False

    def test_state_upsert(self, store):
        assert store.get_state("k") is None
        store.set_state("k", "1")
        store.set_state("k", "2")
        assert store.get_state("k") == "2"

    def test_events_newest_first_and_readable(self, store):
        store.add_event("a", "kevin", "第一", {"n": 1})
        store.add_event("b", "kevin", "第二")
        ev = store.events(limit=1)
        assert [e.kind for e in ev] == ["b"]
        assert [e.kind for e in store.events()] == ["b", "a"]
        with store._engine.connect() as c:
            raw = c.execute(text("SELECT payload FROM exec_events WHERE kind='a'")).scalar()
        assert json.loads(raw) == {"n": 1}
        store.add_event("c", payload={"why": "熔断"})
        with store._engine.connect() as c:
            raw = c.execute(text("SELECT payload FROM exec_events WHERE kind='c'")).scalar()
        assert "熔断" in raw
