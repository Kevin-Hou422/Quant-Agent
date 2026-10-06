"""
test_order_manager.py — Phase 12.1 / 12.2 / 12.4 确定性执行工作流

用 moomoo 交易上下文替身（参数名与返回列都与已安装 SDK 机械对齐）+ 真实美股日历 +
注入时钟，走完整日程：收盘后下单 → 次日开盘成交 → 对账回 PositionStore。
数字断言全部手算可复核（见各用例注释）。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from app.core.data_engine.market_calendar import session_close_utc
from app.core.execution.broker_gateway import OrderRequest
from app.core.execution.order_builder import build_rebalance_orders
from app.core.execution.order_manager import (
    LIVE_BOOK_ID, OrderManager, execution_status, freshness_problem, last_closed_session,
    recover_on_startup,
)
from app.core.execution.pretrade_gate import PreTradeLimits
from app.db.execution_store import (
    ST_FILLED, ST_GATE, ST_NOT_SUBMITTED, ST_PARTIAL, ST_PENDING, ST_SUBMITTED, ExecutionStore,
)
from app.db.position_store import PositionStore
from tests.unit.execution.moomoo_fake import FakeTradeContext, make_gateway

D1, D2, D3 = date(2025, 3, 3), date(2025, 3, 4), date(2025, 3, 5)     # 周一至周三
LIM = PreTradeLimits(max_name_weight=0.10, max_gross=1.0, max_participation_pct=0.10,
                     max_daily_loss=0.05, max_price_deviation=0.15)
W = {"AAA": 0.10, "BBB": 0.08, "CCC": 0.05}
PX1 = {"AAA": 50.0, "BBB": 20.0, "CCC": 10.0}
ADV = {"AAA": 1e9, "BBB": 1e9, "CCC": 1e9}


class Clock:
    def __init__(self):
        self.t = datetime(2025, 3, 3, 22, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t

    def after_close(self, d, hours=1.0):
        self.t = session_close_utc(d) + timedelta(hours=hours)


def _env(tmp_path, cash=100_000.0, **ctx_kw):
    ctx = FakeTradeContext(cash=cash, **ctx_kw)
    gw = make_gateway(ctx)
    es = ExecutionStore(db_url=f"sqlite:///{tmp_path / 'e.db'}")
    ps = PositionStore(db_url=f"sqlite:///{tmp_path / 'p.db'}")
    clock = Clock()
    om = OrderManager(gw, es, ps, limits=LIM, paper_aum=100_000.0, limit_band_bps=50.0,
                      flatten_band_bps=500.0, max_aum_mismatch=0.5, clock=clock)
    return SimpleNamespace(ctx=ctx, gw=gw, es=es, ps=ps, clock=clock, om=om)


def _cycle(e, d, w=W, px=PX1, adv=ADV):
    e.clock.after_close(d)
    return e.om.run_cycle(pd.Series(w, dtype=float), pd.Series(px, dtype=float),
                          pd.Series(adv, dtype=float), d)


def _placed(e):
    return [(kw["code"], kw["trd_side"], kw["qty"], kw["price"], kw["remark"])
            for kw in e.ctx.calls_of("place_order")]


def _day2(e):
    """D1 下单 → D2 开盘成交（AAA 50.2 / BBB 20.05 / CCC 10.02）→ D2 收盘价 51 / 20 / 10。"""
    _cycle(e, D1)
    e.ctx.now = "2025-03-04 09:30:01.000"
    e.ctx.open_session({"AAA": 50.2, "BBB": 20.05, "CCC": 10.02})
    e.ctx.mark({"AAA": 51.0, "BBB": 20.0, "CCC": 10.0})
    e.ctx.now = "2025-03-04 16:30:00.000"
    return _cycle(e, D2, px={"AAA": 51.0, "BBB": 20.0, "CCC": 10.0})


# ---------------------------------------------------------------------------
# 主流程：下单 → 成交 → 对账回 PositionStore
# ---------------------------------------------------------------------------

def test_first_cycle_establishes_baseline_and_submits_exact_orders(tmp_path):
    e = _env(tmp_path)
    rep = _cycle(e, D1)
    assert rep.blocked == "" and rep.reconcile["status"] == "baseline"
    assert rep.reconcile["day_return"] is None
    assert sorted(_placed(e)) == [
        ("US.AAA", "BUY", 200, 50.25, "qaR-1-20250303-AAA-B"),     # trunc(10000/50)，50×1.005
        ("US.BBB", "BUY", 400, 20.10, "qaR-1-20250303-BBB-B"),
        ("US.CCC", "BUY", 500, 10.05, "qaR-1-20250303-CCC-B"),
    ]
    assert rep.n_submitted == 3 and rep.n_gate_rejected == 0 and rep.n_submit_errors == 0
    rows = {r.ticker: r for r in e.es.orders()}
    assert {r.status for r in rows.values()} == {ST_SUBMITTED}
    assert {r.broker_order_id for r in rows.values()} == set(e.ctx.orders)
    assert e.ps.last_pnl_date(LIVE_BOOK_ID) == D1
    assert e.ps.latest_equity(LIVE_BOOK_ID) == pytest.approx(1.0)


def test_next_day_fills_are_reconciled_into_position_store(tmp_path):
    e = _env(tmp_path)
    rep = _day2(e)
    rec = rep.reconcile
    assert rec["status"] == "clean" and rec["discrepancies"] == []
    got = {f["ticker"]: (f["qty"], f["price"], f["ref_price"]) for f in rec["fills"]}
    assert got == {"AAA": pytest.approx((200.0, 50.2, 50.0)),
                   "BBB": pytest.approx((400.0, 20.05, 20.0)),
                   "CCC": pytest.approx((500.0, 10.02, 10.0))}
    # 现金 100000 − (200×50.2 + 400×20.05 + 500×10.02) = 76930；+ 10200 + 8000 + 5000 = 100130
    assert rec["account"]["total_assets"] == pytest.approx(100_130.0)
    assert rec["day_return"] == pytest.approx(0.0013)

    pnl = e.ps.pnl_history(LIVE_BOOK_ID)[-1]
    assert str(pnl.date) == "2025-03-04"
    assert pnl.equity == pytest.approx(1.0013) and pnl.net_ret == pytest.approx(0.0013)
    # 执行缺口 200×0.2 + 400×0.05 + 500×0.02 = $70 → 70 / 100000 = 7 bps
    assert pnl.cost_bps == pytest.approx(7.0)
    assert pnl.gross_ret == pytest.approx(0.0020)
    pos = e.ps.latest_positions(LIVE_BOOK_ID)
    assert pos["AAA"] == pytest.approx(10_200 / 100_130)
    fills = {f.ticker: f for f in e.ps.fills_on(LIVE_BOOK_ID, D2)}
    assert fills["AAA"].fill_price == pytest.approx(50.2) and fills["AAA"].cost_usd == pytest.approx(40.0)
    assert {r.status for r in e.es.orders() if r.decision_date == D1} == {ST_FILLED}

    # D2 再平衡：AAA 目标 trunc(0.10×100130/51)=196 → 卖 4，限价 51×0.995=50.745 → 50.75
    new = [p for p in _placed(e) if "20250304" in p[4]]
    assert new == [("US.AAA", "SELL", 4, 50.75, "qaR-1-20250304-AAA-S")]


def test_rerunning_the_same_decision_date_places_nothing_new(tmp_path):
    e = _env(tmp_path)
    _day2(e)
    n = len(_placed(e))
    rep = _cycle(e, D2, px={"AAA": 51.0, "BBB": 20.0, "CCC": 10.0})
    assert rep.blocked == "" and rep.reconcile["status"] == "clean"
    assert len(_placed(e)) == n
    assert rep.n_planned == 1 and rep.n_duplicate == 1 and rep.n_submitted == 0
    assert e.ctx.calls_of("modify_order") == [] and rep.cancelled_stale == [], \
        "同一决策日重跑撤掉了自己刚下的单"


# ---------------------------------------------------------------------------
# 12.4 崩溃恢复
# ---------------------------------------------------------------------------

def _d1_orders():
    return build_rebalance_orders(pd.Series(W), {}, pd.Series(PX1), 100_000.0, D1,
                                  book_id=LIVE_BOOK_ID, band_bps=50.0).orders


def test_crash_after_place_before_persist_is_adopted_not_resubmitted(tmp_path):
    e = _env(tmp_path)
    e.clock.after_close(D1)
    e.om.reconcile()                                           # 基准
    aaa = next(o for o in _d1_orders() if o.ticker == "AAA")
    e.es.add_intent(aaa, LIVE_BOOK_ID, D1)                     # 写了意图
    e.gw.place(OrderRequest(aaa.client_id, "AAA", "BUY", aaa.qty, aaa.limit_price))
    # ……进程在 mark_submitted 之前死掉。新进程：
    om2 = OrderManager(e.gw, e.es, e.ps, limits=LIM, paper_aum=100_000.0, limit_band_bps=50.0,
                       flatten_band_bps=500.0, max_aum_mismatch=0.5, clock=e.clock)
    rep = om2.run_cycle(pd.Series(W), pd.Series(PX1), pd.Series(ADV), D1)
    assert rep.blocked == ""
    codes = [p[0] for p in _placed(e)]
    assert codes.count("US.AAA") == 1, "崩溃前已到达券商的单被重下了一张"
    assert sorted(codes) == ["US.AAA", "US.BBB", "US.CCC"]
    row = e.es.get(aaa.client_id)
    assert row.status == ST_SUBMITTED and row.broker_order_id in e.ctx.orders


def test_crash_before_place_is_confirmed_absent_then_retried_once(tmp_path):
    e = _env(tmp_path)
    e.clock.after_close(D1)
    e.om.reconcile()
    aaa = next(o for o in _d1_orders() if o.ticker == "AAA")
    e.es.add_intent(aaa, LIVE_BOOK_ID, D1)                     # 意图写了，单没发出去
    rep = _cycle(e, D1)
    assert rep.blocked == ""
    assert e.es.get(aaa.client_id).status == ST_SUBMITTED
    assert [p[0] for p in _placed(e)].count("US.AAA") == 1


def test_unconfirmable_pending_order_blocks_until_a_human_accepts(tmp_path):
    e = _env(tmp_path, history_supported=False)
    e.clock.after_close(D1)
    e.om.reconcile()
    aaa = next(o for o in _d1_orders() if o.ticker == "AAA")
    e.es.add_intent(aaa, LIVE_BOOK_ID, D1)
    rep = _cycle(e, D1)
    assert rep.blocked == "reconcile_unresolved"
    assert rep.reconcile["unresolved"][0]["client_id"] == aaa.client_id
    assert _placed(e) == []
    with pytest.raises(ValueError):
        e.om.accept_discrepancies("", "")
    acc = e.om.accept_discrepancies("kevin", "在 moomoo App 里核对过，该单不存在")
    assert acc.status == "accepted"
    assert e.es.get(aaa.client_id).status == ST_NOT_SUBMITTED
    assert any(ev.kind == "reconcile_accepted" and ev.actor == "kevin" for ev in e.es.events())
    rep2 = _cycle(e, D1)
    assert rep2.blocked == "" and [p[0] for p in _placed(e)].count("US.AAA") == 1


def test_broker_order_unknown_locally_is_adopted(tmp_path):
    e = _env(tmp_path)
    e.clock.after_close(D1)
    e.om.reconcile()
    e.gw.place(OrderRequest("qaR-1-20250303-AAA-B", "AAA", "BUY", 10, 50.25))   # 本地库丢了
    rec = e.om.reconcile()
    assert rec.adopted == ["qaR-1-20250303-AAA-B"]
    row = e.es.get("qaR-1-20250303-AAA-B")
    assert row.status == ST_SUBMITTED and row.decision_date == D1 and row.purpose == "rebalance"
    assert row.last_err_msg == "adopted_from_broker"


# ---------------------------------------------------------------------------
# 对账差异 / 读数不稳定
# ---------------------------------------------------------------------------

def test_external_change_blocks_trading_until_accepted(tmp_path):
    e = _env(tmp_path)
    _day2(e)
    code = "US.AAA"
    e.ctx.pos[code]["qty"] *= 2                                # 拆股：持仓翻倍、价格减半
    e.ctx.pos[code]["nominal_price"] /= 2
    n = len(_placed(e))
    rep = _cycle(e, D3, px={"AAA": 25.5, "BBB": 20.0, "CCC": 10.0})
    assert rep.blocked == "reconcile_discrepancy"
    assert rep.reconcile["discrepancies"] == [{"ticker": "AAA", "expected": 200.0, "broker": 400.0}]
    assert len(_placed(e)) == n, "持仓对不上时下了单"
    e.om.accept_discrepancies("kevin", "AAA 2:1 拆股")
    rep2 = _cycle(e, D3, px={"AAA": 25.5, "BBB": 20.0, "CCC": 10.0})
    assert rep2.reconcile["status"] == "clean" and rep2.blocked == ""


def test_unstable_reads_do_not_produce_a_verdict(tmp_path):
    e = _env(tmp_path)
    _cycle(e, D1)

    def _trickle(ctx):                                           # 每读一次就又成交 1 股
        for o in ctx.orders.values():
            if o["order_status"] in ("SUBMITTED", "FILLED_PART"):
                ctx._fill(o, o["price"], 1.0)
                return
    e.ctx.on_order_read = _trickle
    rep = _cycle(e, D2)
    assert rep.blocked == "reconcile_unstable"
    assert e.es.latest_snapshot().status == "unstable"
    assert e.es.latest_snapshot(trusted_only=True).status == "baseline"


def test_broker_failure_during_reconcile_blocks_the_cycle(tmp_path):
    e = _env(tmp_path)
    e.ctx.fail["position_list_query"] = "disconnected"
    rep = _cycle(e, D1)
    assert rep.blocked.startswith("reconcile_failed") and "disconnected" in rep.blocked
    assert _placed(e) == []


# ---------------------------------------------------------------------------
# 成交增量
# ---------------------------------------------------------------------------

def test_partial_fills_are_differenced_into_increments(tmp_path):
    e = _env(tmp_path)
    _cycle(e, D1)
    e.ctx.now = "2025-03-04 09:30:01.000"
    e.ctx.open_session({"AAA": 50.2}, ratio=0.5)               # 100 股 @ 50.2
    r1 = e.om.reconcile()
    assert [(f["qty"], f["price"]) for f in r1.fills] == [pytest.approx((100.0, 50.2))]
    o = next(o for o in e.ctx.orders.values() if o["code"] == "US.AAA")
    e.ctx._fill(o, 50.1, 100.0)                                # 再 100 股 @ 50.1 → 累计均价 50.15
    r2 = e.om.reconcile()
    assert len(r2.fills) == 1
    assert r2.fills[0]["qty"] == pytest.approx(100.0) and r2.fills[0]["price"] == pytest.approx(50.1)
    assert e.es.get("qaR-1-20250303-AAA-B").status == ST_FILLED
    assert r2.status == "clean"


def test_partial_then_expired_is_partial_status(tmp_path):
    e = _env(tmp_path)
    _cycle(e, D1)
    e.ctx.open_session({"AAA": 50.2}, ratio=0.5)
    e.ctx.expire_day()
    e.om.reconcile()
    row = e.es.get("qaR-1-20250303-AAA-B")
    assert row.status == ST_PARTIAL and row.dealt_qty == 100.0
    assert e.es.get("qaR-1-20250303-BBB-B").status == "CANCELLED"


def test_fill_reversal_produces_a_negative_increment(tmp_path):
    e = _env(tmp_path)
    _cycle(e, D1)
    e.ctx.open_session({"AAA": 50.2})
    e.om.reconcile()
    o = next(o for o in e.ctx.orders.values() if o["code"] == "US.AAA")
    o["dealt_qty"], o["order_status"] = 150.0, "FILL_CANCELLED"   # 券商撤销 50 股成交
    e.ctx.pos["US.AAA"]["qty"] -= 50
    rec = e.om.reconcile()
    assert [(f["qty"], f["price"]) for f in rec.fills] == [pytest.approx((-50.0, 50.2))]
    assert rec.status == "clean", "券商撤销的成交已经通过订单告诉了我们，不该被当成外部变动"


# ---------------------------------------------------------------------------
# 新鲜度 / 资金口径 / 风控拒单 / 熔断
# ---------------------------------------------------------------------------

def test_stale_or_unclosed_decision_dates_place_nothing(tmp_path):
    e = _env(tmp_path)
    e.clock.after_close(D2)
    rep = e.om.run_cycle(pd.Series(W), pd.Series(PX1), pd.Series(ADV), D1)
    assert rep.blocked == "stale_decision_date"
    e.clock.t = session_close_utc(D2) - timedelta(hours=1)
    rep2 = e.om.run_cycle(pd.Series(W), pd.Series(PX1), pd.Series(ADV), D2)
    assert rep2.blocked == "session_not_closed"
    rep3 = e.om.run_cycle(pd.Series(W), pd.Series(PX1), pd.Series(ADV), date(2025, 3, 8))
    assert rep3.blocked == "not_a_trading_day"
    assert _placed(e) == []


def test_freshness_rules_on_the_real_calendar():
    close_d1 = session_close_utc(D1)
    assert freshness_problem(D1, close_d1 + timedelta(minutes=1)) == ""
    assert freshness_problem(D1, close_d1 - timedelta(minutes=1)) == "session_not_closed"
    # 周五收盘后下单、周末与周一开盘前仍可交 —— 周一盘中之后就是陈旧决策
    fri = date(2025, 3, 7)
    sat = datetime(2025, 3, 8, 15, 0, tzinfo=timezone.utc)
    assert last_closed_session(sat) == fri and freshness_problem(fri, sat) == ""
    mon_after = session_close_utc(date(2025, 3, 10)) + timedelta(minutes=5)
    assert freshness_problem(fri, mon_after) == "stale_decision_date"


@pytest.mark.parametrize("cash,blocked", [(300_000.0, "aum_mismatch"), (150_000.0, ""),
                                          (149_000.0, ""), (51_000.0, ""), (50_000.0, ""),
                                          (49_000.0, "aum_mismatch")])
def test_account_size_must_match_paper_aum(tmp_path, cash, blocked):
    e = _env(tmp_path, cash=cash)
    rep = _cycle(e, D1)
    assert rep.blocked == blocked
    assert (len(_placed(e)) == 0) == bool(blocked)


def test_gate_rejections_are_recorded_and_retryable(tmp_path):
    e = _env(tmp_path)
    rep = _cycle(e, D1, adv={"AAA": 1e9, "BBB": 1e9, "CCC": 0.0})
    assert rep.n_gate_rejected == 1 and rep.n_submitted == 2
    row = e.es.get("qaR-1-20250303-CCC-B")
    assert row.status == ST_GATE and row.reject_reason == "no_adv"
    rep2 = _cycle(e, D1)                                       # ADV 修好后同日重跑
    assert rep2.n_submitted == 1 and rep2.n_duplicate == 2
    assert [p[0] for p in _placed(e)].count("US.CCC") == 1


def test_daily_loss_halt_lets_only_reductions_through(tmp_path):
    e = _env(tmp_path)
    _cycle(e, D1)
    e.ctx.open_session({"AAA": 50.2, "BBB": 20.05, "CCC": 10.02})
    low = {"AAA": 30.0, "BBB": 12.0, "CCC": 6.0}
    e.ctx.mark(low)                                            # 总资产 90730 → 日收益 −9.27%
    rep = _cycle(e, D2, w={"AAA": 0.0, "BBB": 0.08, "CCC": 0.05}, px=low)
    assert rep.reconcile["day_return"] == pytest.approx(-0.0927)
    assert rep.gate["halted"] == "daily_loss_halt"
    assert {r["ticker"]: r["reason"] for r in rep.gate["rejected"]} == {
        "BBB": "daily_loss_halt", "CCC": "daily_loss_halt"}
    assert [p[:3] for p in _placed(e) if "20250304" in p[4]] == [("US.AAA", "SELL", 200)]


def test_submit_errors_wait_a_day_before_being_declared_absent(tmp_path):
    e = _env(tmp_path)
    e.ctx.fail["place_order"] = "request timeout"
    rep = _cycle(e, D1)
    assert rep.n_submit_errors == 3 and rep.n_submitted == 0
    rows = e.es.orders()
    assert {r.status for r in rows} == {ST_PENDING}
    assert all("request timeout" in r.last_err_msg for r in rows)
    del e.ctx.fail["place_order"]
    # 同一天：券商可能已收单只是列表里还没有 —— 不判"未提交"、不重试、不阻断
    rep2 = _cycle(e, D1)
    assert rep2.blocked == "" and rep2.n_submitted == 0 and rep2.n_duplicate == 3
    assert len(rep2.reconcile["awaiting_confirmation"]) == 3
    assert len(_placed(e)) == 3                                # 只有第一次（报错的）调用
    # 之后的交易日仍查不到 → 确认未提交；当日按新决策日正常下单
    rep3 = _cycle(e, D2)
    assert {r.status for r in e.es.orders() if r.decision_date == D1} == {ST_NOT_SUBMITTED}
    assert rep3.blocked == "" and rep3.n_submitted == 3


def test_timeout_after_the_broker_accepted_never_duplicates(tmp_path):
    """下单调用报超时、但券商其实收下了：同日重跑必须认领这张单，而不是再下一张。"""
    e = _env(tmp_path)
    e.ctx.accept_then_error = "timeout waiting for response"
    rep = _cycle(e, D1)
    assert rep.n_submit_errors == 3 and len(e.ctx.orders) == 3
    e.ctx.accept_then_error = None
    rep2 = _cycle(e, D1)
    assert rep2.blocked == "" and rep2.n_submitted == 0
    assert len(e.ctx.orders) == 3, "券商侧出现了重复订单"
    assert {r.status for r in e.es.orders()} == {ST_SUBMITTED}
    assert {r.broker_order_id for r in e.es.orders()} == set(e.ctx.orders)


def test_accepting_during_an_unstable_read_is_refused(tmp_path):
    e = _env(tmp_path)
    _cycle(e, D1)

    def _trickle(ctx):
        for o in ctx.orders.values():
            if o["order_status"] in ("SUBMITTED", "FILLED_PART"):
                ctx._fill(o, o["price"], 1.0)
                return
    e.ctx.on_order_read = _trickle
    e.clock.after_close(D2)
    with pytest.raises(RuntimeError, match="不稳定"):
        e.om.accept_discrepancies("kevin", "x")
    assert not any(ev.kind == "reconcile_accepted" for ev in e.es.events())


def test_stale_orders_are_cancelled_before_new_ones(tmp_path):
    e = _env(tmp_path)
    _day2(e)                                                   # D2 挂了 AAA 卖 4（未成交）
    rep = _cycle(e, D3, px={"AAA": 51.0, "BBB": 20.0, "CCC": 10.0})
    assert rep.cancelled_stale == ["qaR-1-20250304-AAA-S"]
    stale_id = e.es.get("qaR-1-20250304-AAA-S").broker_order_id
    assert [c["order_id"] for c in e.ctx.calls_of("modify_order")] == [stale_id]
    assert ("US.AAA", "SELL", 4, 50.75, "qaR-1-20250305-AAA-S") in _placed(e)


def test_failed_stale_cancel_blocks_new_orders(tmp_path):
    e = _env(tmp_path)
    _day2(e)
    n = len(_placed(e))
    e.ctx.fail["modify_order"] = "cancel rejected"
    rep = _cycle(e, D3, px={"AAA": 51.0, "BBB": 20.0, "CCC": 10.0})
    assert rep.blocked == "stale_cancel_failed" and len(_placed(e)) == n


# ---------------------------------------------------------------------------
# 一键全平
# ---------------------------------------------------------------------------

def test_kill_switch_cancels_flattens_and_stops_rebalancing(tmp_path):
    e = _env(tmp_path)
    _day2(e)
    with pytest.raises(ValueError):
        e.om.engage_kill_switch("", "x")
    out = e.om.engage_kill_switch("kevin", "drill")
    assert out["state"]["engaged"] is True
    stale = e.es.get("qaR-1-20250304-AAA-S").broker_order_id
    assert out["cancelled"] == [stale]
    flat = sorted((p[0], p[1], p[2], p[3]) for p in _placed(e) if p[4].startswith("qaF-1-"))
    # 券商现价 51 / 20 / 10，全平带 5%：卖单限价 48.45 / 19.0 / 9.5
    assert flat == [("US.AAA", "SELL", 200, 48.45), ("US.BBB", "SELL", 400, 19.0),
                    ("US.CCC", "SELL", 500, 9.5)]
    assert out["flatten"]["gate"]["halted"] == "kill_switch_engaged"

    n = len(_placed(e))
    rep = _cycle(e, D3, px={"AAA": 51.0, "BBB": 20.0, "CCC": 10.0})
    assert rep.blocked == "kill_switch_engaged" and rep.flatten["status"] == "waiting"
    assert len(_placed(e)) == n, "熔断开启后仍在调仓"

    e.ctx.now = "2025-03-06 09:30:01.000"
    e.ctx.open_session({"AAA": 50.0, "BBB": 19.5, "CCC": 9.8})
    rep2 = _cycle(e, date(2025, 3, 6), px={"AAA": 50.0, "BBB": 19.5, "CCC": 9.8})
    assert rep2.reconcile["status"] == "clean" and rep2.flatten == {"status": "flat"}
    assert e.ctx.pos == {}

    e.om.reset_kill_switch("kevin", "drill over")
    rep3 = _cycle(e, date(2025, 3, 6), px={"AAA": 50.0, "BBB": 19.5, "CCC": 9.8})
    assert rep3.blocked == "" and rep3.n_submitted == 3
    kinds = [ev.kind for ev in e.es.events()]
    assert "kill_switch_engaged" in kinds and "kill_switch_reset" in kinds


def test_kill_switch_stays_engaged_when_the_broker_fails(tmp_path):
    e = _env(tmp_path)
    _day2(e)
    e.ctx.fail["position_list_query"] = "disconnected"
    out = e.om.engage_kill_switch("kevin", "panic")
    assert "disconnected" in out["error"]
    assert e.es.kill_switch()["engaged"] is True
    del e.ctx.fail["position_list_query"]
    rep = _cycle(e, D3, px={"AAA": 51.0, "BBB": 20.0, "CCC": 10.0})
    assert rep.blocked == "kill_switch_engaged" and rep.flatten["status"] == "resubmitted"
    assert sum(1 for p in _placed(e) if p[4].startswith("qaF-1-")) == 3


# ---------------------------------------------------------------------------
# 只读状态 / 启动恢复
# ---------------------------------------------------------------------------

def test_execution_status_reports_store_contents(tmp_path):
    e = _env(tmp_path)
    _day2(e)
    st = execution_status(e.es)
    assert st["last_snapshot"]["status"] == "clean"
    assert st["last_snapshot"]["positions"]["AAA"]["qty"] == 200.0
    assert [o["client_id"] for o in st["open_orders"]] == ["qaR-1-20250304-AAA-S"]
    assert len(st["recent_fills"]) == 3 and st["kill_switch"] == {"engaged": False}


def test_session_boundaries_are_inclusive_of_the_close():
    close = session_close_utc(D2)
    assert last_closed_session(close) == D2, "收盘那一刻起这一天就算已收盘"
    assert freshness_problem(D2, close) == ""
    assert last_closed_session(close - timedelta(hours=2)) == D1, "盘中时最近已收盘的是前一交易日"


@pytest.mark.parametrize("aum", [0.0, -1.0, float("inf"), float("nan")])
def test_paper_aum_must_be_a_positive_finite_number(tmp_path, aum):
    e = _env(tmp_path)
    with pytest.raises(ValueError, match="paper_aum"):
        OrderManager(e.gw, e.es, e.ps, limits=LIM, paper_aum=aum, limit_band_bps=50.0,
                     flatten_band_bps=500.0, max_aum_mismatch=0.5, clock=e.clock)


def test_history_window_starts_a_week_before_the_oldest_relevant_date(tmp_path):
    e = _env(tmp_path)
    _cycle(e, D1)
    e.clock.after_close(D2)
    e.om.reconcile()
    assert e.ctx.calls_of("history_order_list_query")[-1]["start"] == "2025-02-24 00:00:00"
    e.om.engage_kill_switch("kevin", "drill")                      # 撤单查询按今天回看 7 天
    assert e.ctx.calls_of("history_order_list_query")[-1]["start"] == "2025-02-25 00:00:00"


def test_failed_submission_without_history_stays_unresolved_next_day(tmp_path):
    e = _env(tmp_path, history_supported=False)
    e.ctx.fail["place_order"] = "timeout"
    _cycle(e, D1)
    del e.ctx.fail["place_order"]
    rep = _cycle(e, D2)
    assert rep.blocked == "reconcile_unresolved", "查不到历史就不能断定没下出去"
    assert {u["status"] for u in rep.reconcile["unresolved"]} == {ST_PENDING}


def test_discrepancy_persists_until_a_human_accepts(tmp_path):
    e = _env(tmp_path)
    _day2(e)
    e.ctx.pos["US.AAA"]["qty"] = 400
    for _ in range(2):                                              # 不接受 → 每次都拦
        rep = _cycle(e, D3, px={"AAA": 51.0, "BBB": 20.0, "CCC": 10.0})
        assert rep.blocked == "reconcile_discrepancy"


def test_dust_level_position_differences_are_noise(tmp_path):
    from app.core.execution.broker_gateway import DUST_QTY
    e = _env(tmp_path)
    _day2(e)
    e.clock.after_close(D3)
    # 先测超出阈值：差异快照不可信，基准仍是 D2；若先测"干净"，基准会随之移动
    e.ctx.pos["US.AAA"]["qty"] = 200 + 2 * DUST_QTY
    assert e.om.reconcile().status == "discrepancy"
    e.ctx.pos["US.AAA"]["qty"] = 200 + DUST_QTY
    assert e.om.reconcile().status == "clean"


def test_net_return_compounds_on_the_previous_day(tmp_path):
    e = _env(tmp_path)
    _day2(e)                                                        # D2 净值 1.0013
    e.ctx.mark({"AAA": 52.0})
    e.clock.after_close(D3)
    e.om.reconcile()                                               # 76930 + 10400 + 8000 + 5000
    pnl = e.ps.pnl_history(LIVE_BOOK_ID)[-1]
    assert pnl.equity == pytest.approx(1.0033)
    assert pnl.net_ret == pytest.approx(1.0033 / 1.0013 - 1.0)
    d2 = {f.ticker: f for f in e.ps.fills_on(LIVE_BOOK_ID, D2)}
    assert d2["AAA"].traded_weight == pytest.approx(200 * 50.2 / 100_130)


def test_baseline_survives_acceptance(tmp_path):
    e = _env(tmp_path)
    _day2(e)
    e.om.accept_discrepancies("kevin", "例行核对")
    assert float(e.es.get_state("baseline_total_assets")) == 100_000.0
    assert e.ps.latest_equity(LIVE_BOOK_ID) == pytest.approx(1.0013)


def test_empty_account_does_not_divide_by_zero_and_baseline_waits_for_money(tmp_path):
    e = _env(tmp_path, cash=0.0)
    e.clock.after_close(D1)
    e.om.reconcile()
    e.clock.after_close(D2)
    rec = e.om.reconcile()
    assert rec.day_return is None, "上一日资产为 0 时日收益无定义"
    pnl = e.ps.pnl_history(LIVE_BOOK_ID)[-1]
    assert (pnl.equity, pnl.net_ret, pnl.cost_bps) == (0.0, 0.0, 0.0)
    assert e.es.get_state("baseline_total_assets") is None
    e.ctx.cash = 100_000.0                                          # 入金之后建立基线
    e.clock.after_close(D3)
    e.om.reconcile()
    assert float(e.es.get_state("baseline_total_assets")) == 100_000.0
    assert e.ps.latest_equity(LIVE_BOOK_ID) == pytest.approx(1.0)


def test_zero_total_assets_with_positions_and_fills_does_not_divide_by_zero(tmp_path):
    """现金 −1000 + 持仓市值 1000 = 总资产恰为 0，且当天有成交：权重/成交权重都不能 x/0。"""
    from app.core.execution.order_builder import PlannedOrder
    e = _env(tmp_path, cash=0.0)
    e.clock.after_close(D1)
    e.om.reconcile()
    po = PlannedOrder("qaR-1-20250303-AAA-B", "AAA", "BUY", 10, 100.0, 100.5, 0, 10, 0.01)
    e.es.add_intent(po, LIVE_BOOK_ID, D1)
    e.es.mark_submitted(po.client_id, e.gw.place(OrderRequest(po.client_id, "AAA", "BUY", 10, 100.5)))
    e.ctx.now = "2025-03-04 09:30:01.000"
    e.ctx.open_session({"AAA": 100.0})
    e.clock.after_close(D2)
    rec = e.om.reconcile()
    assert rec.account["total_assets"] == 0.0 and rec.positions["AAA"]["qty"] == 10.0
    assert e.ps.latest_positions(LIVE_BOOK_ID) == {}
    assert e.ps.fills_on(LIVE_BOOK_ID, D2)[0].traded_weight == 0.0


def test_adopted_orders_are_excluded_from_execution_shortfall(tmp_path):
    e = _env(tmp_path)
    e.clock.after_close(D1)
    e.om.reconcile()
    e.gw.place(OrderRequest("qaR-1-20250303-DDD-B", "DDD", "BUY", 10, 60.0))   # 本地无记录
    e.ctx.now = "2025-03-04 09:30:01.000"
    e.ctx.open_session({"DDD": 50.0})
    e.clock.after_close(D2)
    e.om.reconcile()
    assert e.ps.pnl_history(LIVE_BOOK_ID)[-1].cost_bps == 0.0, "没有决策日参考价的成交被算进了执行缺口"


def test_net_zero_daily_fills_report_no_fill_price(tmp_path):
    from app.core.execution.broker_gateway import AccountSnapshot, BrokerOrder, DUST_QTY
    from app.core.execution.order_builder import PlannedOrder
    e = _env(tmp_path)
    po = PlannedOrder("qaR-1-20250303-AAA-B", "AAA", "BUY", 1, 50.0, 50.25, 0, 1, 0.01)
    e.es.add_intent(po, LIVE_BOOK_ID, D1)
    e.es.apply_broker_state(po.client_id, BrokerOrder(
        "1", po.client_id, "AAA", "BUY", 1, 50.25, "FILLED_ALL", 1.0, 50.0, "", "", ""), D2)
    e.es.apply_broker_state(po.client_id, BrokerOrder(       # 撤销到只剩 1 个零股阈值
        "1", po.client_id, "AAA", "BUY", 1, 50.25, "FILL_CANCELLED", DUST_QTY, 50.0, "", "", ""), D2)
    e.om._record_book(D2, AccountSnapshot(100_000.0, 100_000.0, 100_000.0, 0.0, "USD"), {})
    assert e.ps.fills_on(LIVE_BOOK_ID, D2)[0].fill_price == 0.0


def test_leftover_flatten_orders_are_cancelled_after_reset(tmp_path):
    e = _env(tmp_path)
    _day2(e)
    e.om.engage_kill_switch("kevin", "drill")
    flat = {r.client_id for r in e.es.open_orders() if r.purpose == "flatten"}
    assert len(flat) == 3
    e.om.reset_kill_switch("kevin", "drill over")
    rep = _cycle(e, D2, px={"AAA": 51.0, "BBB": 20.0, "CCC": 10.0})
    assert set(rep.cancelled_stale) == flat, "熔断解除后残留的全平卖单会和新调仓打架"


def test_accepting_a_missing_order_closes_it_with_the_right_status(tmp_path):
    e = _env(tmp_path, history_supported=False)
    _cycle(e, D1)
    e.ctx.open_session({"BBB": 20.05}, ratio=0.5)                  # BBB 成交一半
    e.om.reconcile()
    for code in ("US.AAA", "US.BBB"):                              # 券商侧把这两张单"弄丢"了
        oid = next(k for k, o in e.ctx.orders.items() if o["code"] == code)
        del e.ctx.orders[oid]
    assert _cycle(e, D2).blocked == "reconcile_unresolved"
    e.om.accept_discrepancies("kevin", "券商记录缺失，按 App 核对")
    a, b = e.es.get("qaR-1-20250303-AAA-B"), e.es.get("qaR-1-20250303-BBB-B")
    assert (a.status, a.broker_status, a.last_err_msg) == ("CANCELLED", "CANCELLED_ALL",
                                                           "accepted_missing_at_broker")
    assert (b.status, b.broker_status, b.dealt_qty) == (ST_PARTIAL, "CANCELLED_PART", 200.0)


def test_maintenance_reconciles_and_continues_flattening_without_rebalancing(tmp_path):
    e = _env(tmp_path)
    _day2(e)
    n = len(_placed(e))
    e.clock.after_close(D3)
    rep = e.om.maintain("no valid signals")
    assert rep.blocked == "maintenance_only: no valid signals" and rep.flatten is None
    assert rep.reconcile["status"] == "clean" and len(_placed(e)) == n
    e.es.set_kill_switch(True, "kevin", "drill")
    rep2 = e.om.maintain("strategy_gate_failed")
    assert rep2.flatten["status"] == "resubmitted"
    assert sum(1 for p in _placed(e) if p[4].startswith("qaF-1-")) == 3
    e.ctx.fail["position_list_query"] = "down"
    assert e.om.maintain("x").blocked.startswith("reconcile_failed")


def test_engage_without_persisting_requires_the_state_to_be_set_first(tmp_path):
    e = _env(tmp_path)
    with pytest.raises(RuntimeError, match="未落库"):
        e.om.engage_kill_switch("kevin", "x", persist_state=False)
    assert e.ctx.calls_of("place_order") == []
    e.es.set_kill_switch(True, "kevin", "x")
    out = e.om.engage_kill_switch("kevin", "x", persist_state=False)
    assert out["state"]["engaged"] is True
    assert [ev.kind for ev in e.es.events()].count("kill_switch_engaged") == 0


def test_recover_on_startup_reports_reconcile_failures(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'r.db'}")
    ctx = FakeTradeContext()
    ctx.fail["position_list_query"] = "disconnected"
    out = recover_on_startup(lambda: make_gateway(ctx))
    assert out["ok"] is False and "disconnected" in out["error"] and ctx.closed
    assert ExecutionStore().events()[0].kind == "startup_recovery_failed"


def test_recover_on_startup_reconciles_and_logs(tmp_path, monkeypatch):
    from app.config import settings
    from app.core.execution.broker_gateway import BrokerError
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'live.db'}")
    ctx = FakeTradeContext(cash=float(settings.paper_aum))
    ctx.set_position("AAA", 10, 50.0)
    out = recover_on_startup(lambda: make_gateway(ctx))
    assert out["ok"] is True and out["report"]["status"] == "baseline"
    assert out["report"]["positions"]["AAA"]["qty"] == 10.0
    assert ctx.closed is True
    es = ExecutionStore()
    assert [ev.kind for ev in es.events()][0] == "startup_recovery"

    def _down():
        raise BrokerError("OpenD not running")
    out2 = recover_on_startup(_down)
    assert out2 == {"ok": False, "error": "OpenD not running"}
    assert es.events()[0].kind == "startup_recovery_failed"
