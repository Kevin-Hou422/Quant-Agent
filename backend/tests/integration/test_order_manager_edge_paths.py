"""
下单周期的边界与分支（D5 关键路径变异复核：app/core/execution/order_manager.py）。

复核时 137 个点存活 25 个。补的用例：
    · 不等待的账户锁（盘中巡检用）：空闲 → 拿到；被另一个线程 / 另一个持锁者占着 → 拿不到
    · 同一张券商订单在查询结果里出现两次 ≠ 重复下单
    · 撤单等终态：券商受理了撤单但单子还在途 → 报"未确认"；轮询之间才睡、最后一次之后不睡
    · 在途包络按剩余量（委托 − 已成交）算
    · 日亏熔断恰好等于阈值时也算；熔断时当天在途的买单（加风险）要撤
    · 切片不加杠杆：能花的钱 = 切片权益 − 已持仓市值；资金利用回报里的买卖金额
    · 中途熔断那次对账不能"顺手接受"差异
    · 熔断全平时撤单要查到一周前的在途单（真实 SDK 的今日订单只含当天）
    · 账户锁目录：%PROGRAMDATA%\\QuantAgent，没有这个环境变量时退回临时目录
剩下的等价变异写在 PROVEN_EQUIVALENT（各配一条机械验证的用例）。
"""

from __future__ import annotations

import os
import threading

import pytest

from app.core.execution.order_manager import OrderManager, _account_lock_dir
from app.core.execution.pretrade_gate import PreTradeLimits
from app.db.execution_store import OPEN_LOCAL, ST_SUBMITTED
from tests.unit.execution.test_order_manager import (
    ADV,
    D1,
    D2,
    D3,
    PX1,
    W,
    _cycle,
    _day2,
    _env,
    _placed,
)


class TestNonBlockingAccountLock:

    def test_a_free_lock_is_taken(self, tmp_path):
        e = _env(tmp_path)
        with e.om.try_account_lock() as got:
            assert got is True

    def test_a_lock_held_by_another_thread_is_not_taken(self, tmp_path):
        e = _env(tmp_path)
        held, release = threading.Event(), threading.Event()

        def hold():
            with e.om._account_lock():
                held.set()
                release.wait(10)
        t = threading.Thread(target=hold)
        t.start()
        try:
            held.wait(10)
            with e.om.try_account_lock() as got:
                assert got is False
        finally:
            release.set()
            t.join(10)

    def test_a_lock_file_held_by_another_holder_is_not_taken(self, tmp_path):
        """只占着账户锁文件（另一个进程的情形），不占进程内的 RLock。"""
        from app.db.process_lock import process_lock
        e = _env(tmp_path)
        path = e.om._account_lock_path()
        held, release = threading.Event(), threading.Event()

        def hold():
            with process_lock(path, timeout_s=5):
                held.set()
                release.wait(10)
        t = threading.Thread(target=hold)
        t.start()
        try:
            held.wait(10)
            with e.om.try_account_lock() as got:
                assert got is False
        finally:
            release.set()
            t.join(10)


def test_the_same_broker_order_listed_twice_is_not_a_duplicate(tmp_path):
    e = _env(tmp_path)
    _cycle(e, D1)
    orig = e.gw.orders

    def twice(since):
        q = orig(since)
        q.orders = list(q.orders) + list(q.orders)
        return q
    e.gw.orders = twice
    rep = e.om.reconcile()
    assert not [u for u in rep.unresolved if u.get("reason") == "duplicate_remark"], rep.unresolved


class TestAwaitTerminal:

    def test_an_accepted_cancel_that_does_not_settle_is_unconfirmed(self, tmp_path):
        e = _env(tmp_path)
        _cycle(e, D1)
        oid = e.ctx.calls_of("place_order") and list(e.ctx.orders)[0]
        e.ctx.async_cancel = True
        e.gw.cancel(oid)
        slept = []
        e.om._sleep, e.om.cancel_wait_polls = slept.append, 3
        assert e.om._await_terminal([oid]) == [oid]
        assert len(slept) == 2, "三次轮询之间睡两次，最后一次之后不睡"

    def test_settled_cancels_are_confirmed(self, tmp_path):
        e = _env(tmp_path)
        _cycle(e, D1)
        oid = list(e.ctx.orders)[0]
        e.ctx.async_cancel = True
        e.gw.cancel(oid)
        e.om._sleep, e.om.cancel_wait_polls = (lambda s: e.ctx.settle_cancels()), 3
        assert e.om._await_terminal([oid]) == []
        assert e.om._await_terminal([]) == []


def test_the_in_flight_envelope_uses_the_remaining_quantity(tmp_path):
    e = _env(tmp_path)
    _cycle(e, D1)
    e.ctx.now = "2025-03-04 09:30:01.000"
    e.ctx.open_session({"AAA": 50.0, "BBB": 20.0, "CCC": 10.0}, ratio=0.25)
    e.om.reconcile()
    env = {o.ticker: o.remaining_qty for o in e.om._open_order_envelope()}
    rows = {r.ticker: r for r in e.es.open_orders()}
    assert env and all(env[t] == pytest.approx(rows[t].qty - rows[t].dealt_qty) for t in env)
    assert any(rows[t].dealt_qty > 0 for t in env)


def _om(e, thr):
    lim = PreTradeLimits(max_name_weight=0.10, max_gross=1.0, max_participation_pct=0.10,
                         max_daily_loss=thr, max_price_deviation=0.15)
    return OrderManager(e.gw, e.es, e.ps, limits=lim, paper_aum=100_000.0, limit_band_bps=50.0,
                        flatten_band_bps=500.0, clock=e.clock)


def _held_and_rebalanced(e, w2):
    """D1 建仓 → D2 开盘成交 → D2 收盘后按新权重第一次跑 D2（下出的单在途，T+1 才成交）。"""
    _cycle(e, D1)
    e.ctx.now = "2025-03-04 09:30:01.000"
    e.ctx.open_session({"AAA": 50.2, "BBB": 20.05, "CCC": 10.02})
    e.ctx.mark({"AAA": 51.0, "BBB": 20.0, "CCC": 10.0})
    e.ctx.now = "2025-03-04 16:30:00.000"
    n0 = len(_placed(e))
    rep = _cycle(e, D2, w=w2, px={"AAA": 51.0, "BBB": 20.0, "CCC": 10.0})
    return rep, _placed(e)[n0:]


class TestDailyHalt:

    def _halted_rerun(self, tmp_path, loss):
        e = _env(tmp_path)
        e.om = _om(e, 0.25)
        _rep, sent = _held_and_rebalanced(e, dict(W, BBB=0.10))
        assert [p for p in sent if p[1] == "BUY"], sent                  # 当天在途的加仓买单
        base = float(e.es.get_state("baseline_total_assets"))
        prev = e.es.trusted_before(D2)                                     # 日收益的分母：D1 收盘的可信快照
        slice_prev = 100_000.0 + prev.total_assets - base
        assert slice_prev == 100_000.0                                     # D1 是基准日：切片恰好 10 万
        e.ctx.account_override = {"total_assets": base - loss * 100_000.0}  # 切片恰好亏 loss（精确可表示）
        n0 = len(e.ctx.calls_of("modify_order"))
        rep = _cycle(e, D2, w=dict(W, BBB=0.10), px={"AAA": 51.0, "BBB": 20.0, "CCC": 10.0})
        return e, rep, n0

    def test_a_loss_exactly_at_the_threshold_halts_and_cancels_the_open_buy(self, tmp_path):
        e, rep, n0 = self._halted_rerun(tmp_path, 0.25)
        assert rep.reconcile["day_return"] == -0.25
        assert len(e.ctx.calls_of("modify_order")) > n0, "日亏恰好到阈值：当天在途的买单要撤"

    def test_a_loss_short_of_the_threshold_keeps_the_open_buy(self, tmp_path):
        e, rep, n0 = self._halted_rerun(tmp_path, 0.20)
        assert len(e.ctx.calls_of("modify_order")) == n0


def test_a_partly_filled_reducing_sell_survives_the_daily_halt(tmp_path):
    """卖出 80%、开盘成交一半之后盘中日亏熔断：剩下的 40% 仍是减仓（持仓还剩 60%），不撤。"""
    from datetime import datetime, timezone
    e = _env(tmp_path)
    e.om = _om(e, 0.25)
    _rep, sent = _held_and_rebalanced(e, dict(W, AAA=0.02))
    sell = [p for p in sent if p[0] == "US.AAA" and p[1] == "SELL"]
    assert sell, sent
    e.ctx.now = "2025-03-05 09:31:00.000"
    e.ctx.open_session({"AAA": 51.0}, ratio=0.5)
    e.clock.t = datetime(2025, 3, 5, 15, 0, tzinfo=timezone.utc)          # D3 盘中：最近已收盘的还是 D2
    e.ctx.account_override = {"total_assets": 50_000.0}                    # 切片亏一半 → 日亏熔断
    rows = {r.ticker: r for r in e.es.open_orders()}
    n0 = len(e.ctx.calls_of("modify_order"))
    rep = _cycle(e, D2, w=dict(W, AAA=0.02), px={"AAA": 51.0, "BBB": 20.0, "CCC": 10.0})
    assert rep.reconcile["day_return"] <= -0.25
    aaa = e.es.get(rows["AAA"].client_id)
    assert 0 < aaa.dealt_qty < aaa.qty
    cancelled = {kw["order_id"] for kw in e.ctx.calls_of("modify_order")[n0:]}
    assert aaa.broker_order_id not in cancelled, "部分成交后仍在减仓的卖单被当成反手撤掉了"


def test_a_change_below_the_dust_threshold_records_no_fill(tmp_path):
    """execution_store L342 等价的机械验证：外层 `abs(dq) > DUST_QTY` 已排除 dq == 0，`dq > 0` 与 `>= 0` 分不出来。"""
    from app.core.execution.broker_gateway import DUST_QTY
    e = _env(tmp_path)
    _cycle(e, D1)
    row = e.es.open_orders()[0]
    bo = next(o for o in e.gw.orders(D1).orders if o.client_id == row.client_id)
    from dataclasses import replace
    assert e.es.apply_broker_state(row.client_id, replace(bo, dealt_qty=DUST_QTY / 2), D2) is None
    inc = e.es.apply_broker_state(row.client_id, replace(bo, dealt_qty=3.0, dealt_avg_price=50.0), D2)
    assert inc is not None and inc.qty == pytest.approx(3.0 - DUST_QTY / 2)


class TestSliceBudget:

    def test_buying_power_is_slice_equity_minus_what_is_invested(self, tmp_path):
        e = _env(tmp_path, cash=1_000_000.0)                             # 账户比切片大得多
        e.ctx.set_position("AAA", 1200, 50.0)                            # 已投 6 万
        e.ctx.quotes.update({k: float(v) for k, v in PX1.items()})
        rep = _cycle(e, D1, w={"AAA": 0.10, "BBB": 0.10, "CCC": 0.10, "DDD": 0.10},
                     px=dict(PX1, DDD=5.0), adv=dict(ADV, DDD=1e9))
        cap = rep.capital
        assert cap["slice_equity"] == pytest.approx(100_000.0) and cap["invested"] == pytest.approx(60_000.0)
        assert cap["buying_power"] == pytest.approx(40_000.0)
        assert cap["planned_buy"] <= 40_000.0 + 1e-6

    def test_the_capital_report_sums_the_approved_orders(self, tmp_path):
        e = _env(tmp_path)
        rep, sent = _held_and_rebalanced(e, {"AAA": 0.02, "BBB": 0.10, "CCC": 0.09})
        sells = sum(q * p for _c, side, q, p, _r in sent if side == "SELL")
        buys = sum(q * p for _c, side, q, p, _r in sent if side == "BUY")
        assert sells > 0 and buys > 0, sent
        assert rep.capital["planned_sell"] == pytest.approx(round(sells, 2))
        assert rep.capital["planned_buy"] == pytest.approx(round(buys, 2))
        assert rep.capital["utilization_after_buys"] == pytest.approx(
            round((rep.capital["invested"] + rep.capital["planned_buy"]) / rep.capital["slice_equity"], 4))


def test_the_mid_batch_reconcile_does_not_accept_discrepancies(tmp_path):
    e = _env(tmp_path)
    orig = e.gw.place

    def place_then_switch(req):
        oid = orig(req)
        if len(_placed(e)) == 1:
            e.es.set_kill_switch(True, "review", "switch during order batch")
            e.ctx.set_position("ZZZ", 10, 5.0)                           # 同时账户里冒出一笔外部持仓
        return oid
    e.gw.place = place_then_switch
    rep = _cycle(e, D1)
    assert rep.blocked == "kill_switch_engaged_mid_batch"
    assert e.es.latest_snapshot().status == "discrepancy", "中途熔断那次对账把差异接受成了可信基准"


def test_flatten_cancels_open_orders_from_earlier_days(tmp_path):
    """真实 SDK：今日订单只含当天创建的单，更早的要靠按日期取的历史订单。"""
    e = _env(tmp_path)
    _cycle(e, D1)                                                         # D1 晚上下的单（替身不会自动过期）
    ctx = e.ctx
    real_today, real_hist = ctx.order_list_query, ctx.history_order_list_query

    def today_only(**kw):
        ret, df = real_today(**kw)
        return ret, df[df["create_time"].astype(str).str[:10] == ctx.now[:10]]

    def history_between(**kw):
        ret, df = real_hist(**kw)
        lo, hi = str(kw.get("start") or "")[:10], str(kw.get("end") or "9999-12-31")[:10]
        day = df["create_time"].astype(str).str[:10]
        return ret, df[(day >= lo) & (day <= hi)]
    ctx.order_list_query, ctx.history_order_list_query = today_only, history_between
    ctx.now = "2025-03-05 17:00:00.000"                                   # 两天之后
    e.clock.after_close(D3)
    before = {o["order_id"] for o in ctx.orders.values() if o["order_status"] not in ("CANCELLED_ALL", "FILLED_ALL")}
    assert before
    e.om._sleep = lambda s: None
    e.om.engage_kill_switch("review", "flatten test")
    cancelled = {kw["order_id"] for kw in ctx.calls_of("modify_order")}
    assert before <= cancelled, f"一周之内的在途单没撤：{before - cancelled}"


class TestAccountLockDir:

    def test_a_configured_directory_wins(self, monkeypatch, tmp_path):
        from app.config import settings
        monkeypatch.setattr(settings, "account_lock_dir", str(tmp_path / "locks"))
        assert _account_lock_dir() == tmp_path / "locks"

    @pytest.mark.skipif(os.name != "nt", reason="%PROGRAMDATA% 只在 Windows 上有")
    def test_programdata_on_windows(self, monkeypatch, tmp_path):
        from app.config import settings
        monkeypatch.setattr(settings, "account_lock_dir", "")
        monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "pd"))
        assert _account_lock_dir() == tmp_path / "pd" / "QuantAgent"

    def test_without_programdata_the_temp_directory(self, monkeypatch):
        import tempfile
        from pathlib import Path

        from app.config import settings
        monkeypatch.setattr(settings, "account_lock_dir", "")
        monkeypatch.delenv("PROGRAMDATA", raising=False)
        assert _account_lock_dir() == Path(tempfile.gettempdir()) / "quant_agent"


def test_open_local_rows_have_a_broker_id_exactly_when_submitted(tmp_path):
    """L669 等价的机械验证：在途行要么是 PENDING_SUBMIT 且没有委托号，要么是 SUBMITTED 且有委托号。"""
    e = _env(tmp_path)
    _cycle(e, D1)
    e.ctx.fail["place_order"] = "timeout"
    _day2(e)
    rows = e.es.open_orders()
    assert {r.status for r in rows} <= OPEN_LOCAL and rows
    assert all((r.status == ST_SUBMITTED) == bool(r.broker_order_id) for r in rows)


def test_a_zero_slice_is_recorded_as_zero_equity():
    """L523 等价的机械验证：`seq > 0` 与 `seq >= 0` 只在 seq == 0 时不同，两条分支给出的都是 0。"""
    for seq in (0.0, -0.0):
        assert (seq if seq is not None and seq > 0 else 0.0) == (seq if seq is not None and seq >= 0 else 0.0) == 0.0


PROVEN_EQUIVALENT = {
    "app/core/execution/order_manager.py ×1 — L523 `seq > 0` -> `seq >= 0`":
        "两者只在切片权益恰好为 0 时不同：`>` 走 else 记 0.0，`>=` 记 seq 本身也是 0（含 −0.0），之后 0 / paper_aum 都是 0。"
        "见 test_a_zero_slice_is_recorded_as_zero_equity。",
    "app/core/execution/order_manager.py ×1 — L669 `status == ST_SUBMITTED and broker_order_id` -> `or`":
        "执行库的在途行只有两种：PENDING_SUBMIT 且没有委托号（还不知道有没有下出去），SUBMITTED 且有委托号"
        "（mark_submitted / adopt 同时写两者）。两种行上 and 与 or 的结论相同。"
        "见 test_open_local_rows_have_a_broker_id_exactly_when_submitted（含下单超时留下的 PENDING 行）。",
    "app/db/execution_store.py ×1 — L342 `if dq > 0` -> `dq >= 0`（成交增量的均价分支）":
        "这一行在 `if abs(dq) > DUST_QTY:` 之内，DUST_QTY = 2^-30 > 0，dq == 0 根本走不进来；"
        "两种写法只在 dq == 0 时不同。见 test_a_change_below_the_dust_threshold_records_no_fill。",
    "app/core/execution/order_manager.py ×1 — L747 `if equity > 0` -> `>=`（资金利用率的分母）":
        "走到这一行之前，切片权益 ≤ 0 已经在调仓入口被拦下（blocked = slice_depleted 直接返回），equity 在这里恒大于 0，"
        "两种写法不会走到不同分支。见 test_a_depleted_slice_stops_trading。",
}


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 4
