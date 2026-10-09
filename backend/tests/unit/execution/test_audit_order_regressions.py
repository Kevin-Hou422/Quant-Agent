"""
test_audit_order_regressions.py — 外部审计（2026-10-07）订单层缺陷的**正确行为**回归测试

审计的复现脚本断言的是"缺陷存在"；这里把同样的场景改成断言"缺陷不再发生"。
券商替身打开**悲观契约**：新单延迟可见（F03）、撤单异步生效（F04）—— 这两条以前的替身
都是乐观的（下单即可见、撤单即终态），缺陷正是藏在那两个乐观假设后面。
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.core.execution.broker_gateway import OrderRequest
from app.core.execution.order_builder import build_rebalance_orders
from app.core.execution.pretrade_gate import GateContext, OpenOrder, PreTradeLimits, check
from app.db.execution_store import ST_PENDING, ST_SUBMITTED
from tests.unit.execution.test_order_manager import (
    D1, D2, PX1, _cycle, _d1_orders, _env, _placed,
)


def _no_wait(e):
    e.om._sleep = lambda s: None
    return e


def _count(e, code):
    return [p[0] for p in _placed(e)].count(code)


# ---------------------------------------------------------------------------
# F03：下单后落库前崩溃 + 券商侧暂不可见 → 绝不同日重下
# ---------------------------------------------------------------------------

def test_crash_after_the_broker_accepted_with_delayed_visibility_never_resubmits(tmp_path):
    e = _env(tmp_path)
    e.clock.after_close(D1)
    e.om.reconcile()
    o = next(o for o in _d1_orders() if o.ticker == "AAA")
    # 与 _submit 完全相同的落库顺序：写意图 → 标记尝试 → 下单；然后进程在 mark_submitted 之前死掉
    e.es.add_intent(o, -1, D1)
    e.es.mark_attempting(o.client_id)
    e.ctx.hide_new_orders = True                       # 刚收的单还查不到
    e.gw.place(OrderRequest(o.client_id, o.ticker, o.side, o.qty, o.limit_price))
    rep = _cycle(e, D1)                                # 同日重启重跑
    assert _count(e, "US.AAA") == 1, "券商已收单、只是暂不可见 —— 却被判成未提交又下了一张"
    assert o.client_id in {a["client_id"] for a in rep.reconcile["awaiting_confirmation"]}
    assert e.es.get(o.client_id).status == ST_PENDING

    e.ctx.reveal_orders()                              # 券商侧终于可见
    _cycle(e, D1)
    assert _count(e, "US.AAA") == 1
    assert e.es.get(o.client_id).status == ST_SUBMITTED


def test_the_attempt_marker_is_persisted_before_the_broker_call(tmp_path):
    """进程死在 place 调用里面（没有返回、也没有异常被我们接住）：标记必须已经落盘。"""
    e = _env(tmp_path)
    seen = {}
    real_place = e.gw.place

    def killed(req):
        seen["attempted"] = e.es.get(req.client_id).submit_attempted_at
        real_place(req)
        raise KeyboardInterrupt("process killed")      # 不被 _submit 的 except 接住

    e.gw.place = killed
    with pytest.raises(KeyboardInterrupt):
        _cycle(e, D1)
    assert seen["attempted"] is not None, "调用券商之前没有落 submit_attempted_at"


def test_a_crash_before_the_attempt_is_still_retried_once(tmp_path):
    """没标记过尝试的意图（崩在调用之前）仍可确认未提交并重下 —— 修 F03 不能把它也锁死。"""
    e = _env(tmp_path)
    e.clock.after_close(D1)
    e.om.reconcile()
    o = next(o for o in _d1_orders() if o.ticker == "AAA")
    e.es.add_intent(o, -1, D1)                         # 只写了意图
    _cycle(e, D1)
    assert _count(e, "US.AAA") == 1


def test_duplicate_remarks_at_the_broker_block_trading(tmp_path):
    """remark 不是券商保证唯一的 id：同一 remark 出现在两张单上 = 重复下单已发生，必须停下等人看。"""
    e = _env(tmp_path)
    _cycle(e, D1)
    o = next(o for o in _d1_orders() if o.ticker == "AAA")
    e.gw.place(OrderRequest(o.client_id, o.ticker, o.side, o.qty, o.limit_price))
    rep = _cycle(e, D1)
    assert rep.reconcile["status"] == "unresolved" and rep.blocked == "reconcile_unresolved"
    dup = [u for u in rep.reconcile["unresolved"] if u.get("reason") == "duplicate_remark"]
    assert dup and dup[0]["client_id"] == o.client_id and len(dup[0]["broker_order_ids"]) == 2


# ---------------------------------------------------------------------------
# F04：撤单受理 ≠ 订单结束
# ---------------------------------------------------------------------------

def test_new_orders_wait_until_stale_cancels_are_confirmed(tmp_path):
    e = _no_wait(_env(tmp_path))
    _cycle(e, D1)
    assert len(_placed(e)) == 3
    e.ctx.async_cancel = True
    rep = _cycle(e, D2)
    assert rep.blocked == "cancel_unconfirmed"
    assert len(rep.cancelled_stale) == 3 and len(_placed(e)) == 3, "撤单未确认就下了新单"
    assert sum(o["order_status"] == "SUBMITTED" for o in e.ctx.orders.values()) == 3

    e.ctx.settle_cancels()
    rep = _cycle(e, D2)
    assert rep.blocked == "" and rep.n_submitted == 3
    assert sum(o["order_status"] == "SUBMITTED" for o in e.ctx.orders.values()) == 3


def test_orders_are_rebuilt_from_positions_after_the_cancel(tmp_path):
    """撤单生效前原单部分成交了：新单必须按撤单**之后**的持仓算，而不是撤单前读到的。"""
    e = _no_wait(_env(tmp_path))
    _cycle(e, D1)
    e.ctx.async_cancel = True

    def fill_then_settle(ctx):
        if ctx.cancel_requested:
            ctx.open_session({"AAA": 50.0}, ratio=0.5)     # 撤单生效前成交了一半
            ctx.settle_cancels()
    e.ctx.on_order_read = fill_then_settle
    rep = _cycle(e, D2)
    assert rep.blocked == "", rep.blocked
    aaa = [p for p in _placed(e)[3:] if p[0] == "US.AAA"]
    assert aaa and aaa[0][2] == 100, f"AAA 应按成交后持仓 100 股补到 200，实际下了 {aaa}"


def test_unconfirmed_cancels_time_out_after_the_configured_polls(tmp_path):
    e = _env(tmp_path)
    naps = []
    e.om._sleep = naps.append
    e.om.cancel_wait_polls, e.om.cancel_wait_s = 4, 0.25
    _cycle(e, D1)
    e.ctx.async_cancel = True
    assert _cycle(e, D2).blocked == "cancel_unconfirmed"
    assert naps == [0.25, 0.25, 0.25], "轮询次数 / 间隔与配置不符（最后一次轮询后不应再睡）"


def test_kill_switch_waits_for_cancels_before_flattening(tmp_path):
    e = _no_wait(_env(tmp_path))
    _cycle(e, D1)
    e.ctx.now = "2025-03-04 09:30:01.000"
    e.ctx.open_session({"AAA": 50.2, "BBB": 20.05, "CCC": 10.02}, ratio=0.5)
    e.ctx.async_cancel = True
    out = e.om.engage_kill_switch("kevin", "drill")
    assert out["flatten"]["status"] == "waiting_cancel" and out["flatten"]["unconfirmed"]
    n_before = len(_placed(e))
    assert all(p[1] != "SELL" for p in _placed(e)), "撤单未确认就下了全平单"

    e.ctx.open_session({"AAA": 50.2, "BBB": 20.05, "CCC": 10.02}, ratio=1.0)  # 撤单生效前成交完
    e.ctx.settle_cancels()
    rep = e.om.maintain("daily")
    sells = {p[0]: p[2] for p in _placed(e)[n_before:] if p[1] == "SELL"}
    assert rep.flatten["status"] == "resubmitted"
    assert sells == {"US.AAA": 200, "US.BBB": 400, "US.CCC": 500}, (
        f"全平没有按撤单之后的真实持仓下：{sells}")


# ---------------------------------------------------------------------------
# F05：挂单与未成交的卖单都计入风险预算
# ---------------------------------------------------------------------------

def test_same_day_risk_check_counts_previous_open_orders(tmp_path):
    e = _env(tmp_path)
    e.om.limits = PreTradeLimits(max_name_weight=.1, max_gross=.1, max_participation_pct=.1,
                                 max_daily_loss=.05, max_price_deviation=.15)
    one = _cycle(e, D1, w={"AAA": .1}, px={"AAA": 50.}, adv={"AAA": 1e9})
    two = _cycle(e, D1, w={"BBB": .1}, px={"BBB": 20.}, adv={"BBB": 1e9})
    assert one.n_submitted == 1 and two.n_submitted == 0
    assert [r["reason"] for r in two.gate["rejected"]] == ["exceeds_gross_limit"]


LIM = PreTradeLimits(max_name_weight=.1, max_gross=.1, max_participation_pct=.1,
                     max_daily_loss=.05, max_price_deviation=.15)


def test_an_unfilled_sell_does_not_free_budget_for_a_buy():
    build = build_rebalance_orders(pd.Series({"AAA": 0., "BBB": .1}), {"AAA": 100.},
                                   pd.Series({"AAA": 100., "BBB": 100.}), 100000., date(2025, 3, 3),
                                   book_id=-1, band_bps=50.)
    dec = check(build.orders, GateContext(equity=100000., buying_power=100000.,
                                          current_qty={"AAA": 100.}, broker_prices={"AAA": 100.},
                                          adv_usd={"AAA": 1e9, "BBB": 1e9}), LIM)
    assert [o.ticker for o in dec.approved] == ["AAA"]                  # 卖单照放
    assert [(o.ticker, r) for o, r in dec.rejected] == [("BBB", "exceeds_gross_limit")]


def test_two_sells_cannot_together_exceed_the_position():
    o = build_rebalance_orders(pd.Series({"AAA": 0.04}), {"AAA": 100.}, pd.Series({"AAA": 100.}),
                               100000., date(2025, 3, 3), book_id=-1, band_bps=50.).orders
    assert o[0].side == "SELL" and o[0].qty == 60
    ctx = GateContext(equity=100000., buying_power=100000., current_qty={"AAA": 100.},
                      broker_prices={"AAA": 100.}, adv_usd={"AAA": 1e9},
                      open_orders=[OpenOrder("AAA", "SELL", 60.0, 99.5)])
    dec = check(o, ctx, LIM)
    assert [(x.ticker, r) for x, r in dec.rejected] == [("AAA", "would_short")]


def test_open_buys_consume_buying_power():
    o = build_rebalance_orders(pd.Series({"BBB": .05}), {}, pd.Series({"BBB": 100.}), 100000.,
                               date(2025, 3, 3), book_id=-1, band_bps=0.).orders
    lim = PreTradeLimits(max_name_weight=1., max_gross=1., max_participation_pct=1.,
                         max_daily_loss=.05, max_price_deviation=.15)
    ctx = GateContext(equity=100000., buying_power=8000., current_qty={}, broker_prices={},
                      adv_usd={"BBB": 1e9}, open_orders=[OpenOrder("CCC", "BUY", 40.0, 100.0)])
    dec = check(o, ctx, lim)       # 在途 4000 + 本单 5000 > 8000
    assert [r for _, r in dec.rejected] == ["insufficient_buying_power"]


def test_net_and_sector_limits_are_enforced_at_the_order_layer():
    o = build_rebalance_orders(pd.Series({"AAA": .08, "BBB": .08}), {},
                               pd.Series({"AAA": 100., "BBB": 100.}), 100000., date(2025, 3, 3),
                               book_id=-1, band_bps=0.).orders
    net = PreTradeLimits(max_name_weight=1., max_gross=1., max_participation_pct=1.,
                         max_daily_loss=.05, max_price_deviation=.15, max_net=.10)
    ctx = GateContext(equity=100000., buying_power=1e6, current_qty={}, broker_prices={},
                      adv_usd={"AAA": 1e9, "BBB": 1e9})
    dec = check(o, ctx, net)
    assert [(x.ticker, r) for x, r in dec.rejected] == [("BBB", "exceeds_net_limit")]

    sec = PreTradeLimits(max_name_weight=1., max_gross=1., max_participation_pct=1.,
                         max_daily_loss=.05, max_price_deviation=.15, max_sector_weight=.10)
    ctx.sectors = {"AAA": 45, "BBB": 45}
    dec = check(o, ctx, sec)
    assert [(x.ticker, r) for x, r in dec.rejected] == [("BBB", "exceeds_sector_limit")]
    ctx.sectors = {"AAA": 45, "BBB": 40}
    assert len(check(o, ctx, sec).approved) == 2


def test_daily_loss_halt_cancels_open_risk_increasing_orders(tmp_path):
    """日亏熔断只拒新单不撤旧单的话，旧的加仓单照样在开盘成交（审计 F05 末段）。"""
    e = _no_wait(_env(tmp_path))
    _cycle(e, D1)
    e.ctx.now = "2025-03-04 09:30:01.000"
    e.ctx.open_session({"AAA": 50.2, "BBB": 20.05, "CCC": 10.02})
    e.ctx.now = "2025-03-04 16:30:00.000"
    first = _cycle(e, D2, w={"AAA": .10, "BBB": .08, "CCC": .05, "DDD": .02},
                   px={"AAA": 50.0, "BBB": 20.0, "CCC": 10.0, "DDD": 5.0},
                   adv={"AAA": 1e9, "BBB": 1e9, "CCC": 1e9, "DDD": 1e9})
    ddd = next(r for r in e.es.orders() if r.ticker == "DDD")
    assert first.n_submitted >= 1 and ddd.status == ST_SUBMITTED
    e.ctx.account_override = {"total_assets": 90_000.0}          # 同日重跑时已跌 10%
    rep = _cycle(e, D2, w={"AAA": .10, "BBB": .08, "CCC": .05, "DDD": .02},
                 px={"AAA": 50.0, "BBB": 20.0, "CCC": 10.0, "DDD": 5.0},
                 adv={"AAA": 1e9, "BBB": 1e9, "CCC": 1e9, "DDD": 1e9})
    assert ddd.client_id in rep.cancelled_stale, "日亏熔断后在途的加仓单没有被撤"
    assert e.ctx.orders[ddd.broker_order_id]["order_status"] == "CANCELLED_ALL"


# ---------------------------------------------------------------------------
# 最坏情况包络的边界（逐点变异复核补的用例）
# ---------------------------------------------------------------------------

def _lim(**kw):
    base = dict(max_name_weight=1., max_gross=1., max_participation_pct=1., max_daily_loss=.05,
                max_price_deviation=.15)
    base.update(kw)
    return PreTradeLimits(**base)


def _buys(weights, px=100., equity=100000.):
    tw = pd.Series(weights, dtype=float)
    return build_rebalance_orders(tw, {}, pd.Series(px, index=tw.index), equity, date(2025, 3, 3),
                                  book_id=-1, band_bps=0.).orders


@pytest.mark.parametrize("field", ["max_net", "max_sector_weight"])
@pytest.mark.parametrize("bad", [0.0, -0.1, float("inf"), float("nan"), "0.1"])
def test_optional_limits_must_be_positive_finite_or_none(field, bad):
    with pytest.raises(ValueError, match=field):
        _lim(**{field: bad})
    assert getattr(_lim(**{field: None}), field) is None
    assert getattr(_lim(**{field: 0.2}), field) == 0.2


def test_a_dust_position_without_a_price_does_not_block_the_sector_check():
    ctx = GateContext(equity=100000., buying_power=1e6, current_qty={"DST": 2.0 ** -30},
                      broker_prices={}, adv_usd={"AAA": 1e9}, sectors={"AAA": 45, "DST": 45})
    dec = check(_buys({"AAA": .05}), ctx, _lim(max_sector_weight=.5))
    assert len(dec.approved) == 1, dec.rejected


def test_an_open_order_without_a_usable_price_makes_gross_unpriceable():
    for lp in (0.0, float("nan")):
        ctx = GateContext(equity=100000., buying_power=1e6, current_qty={}, broker_prices={},
                          adv_usd={"AAA": 1e9},
                          open_orders=[OpenOrder("ZZZ", "BUY", 10.0, lp)])
        dec = check(_buys({"AAA": .05}), ctx, _lim())
        assert [r for _, r in dec.rejected] == ["gross_unpriceable"], lp


def test_the_broker_price_wins_over_an_open_orders_limit_price():
    """在途买单的限价只给**没有券商现价**的标的定价；持仓标的仍按券商现价算敞口。"""
    ctx = GateContext(equity=100000., buying_power=1e6, current_qty={"AAA": 50.},
                      broker_prices={"AAA": 100.}, adv_usd={"BBB": 1e9},
                      open_orders=[OpenOrder("AAA", "BUY", 50.0, 1.0)])
    dec = check(_buys({"BBB": .06}), ctx, _lim(max_gross=.15))
    # 按券商价：AAA 最坏 100 股 × 100 = 10%，+ BBB 6% = 16% > 15%
    assert [r for _, r in dec.rejected] == ["exceeds_gross_limit"]


def test_float_sector_codes_are_grouped():
    """行业代码从 DataFrame 行里取出来是 float（45.0）—— 不能被当成"未知行业"放过。"""
    ctx = GateContext(equity=100000., buying_power=1e6, current_qty={}, broker_prices={},
                      adv_usd={"AAA": 1e9, "BBB": 1e9}, sectors={"AAA": 45.0, "BBB": 45.0})
    dec = check(_buys({"AAA": .08, "BBB": .08}), ctx, _lim(max_sector_weight=.1))
    assert [(o.ticker, r) for o, r in dec.rejected] == [("BBB", "exceeds_sector_limit")]


def test_sector_cap_is_a_share_of_max_gross():
    """行业上限 = max_sector_weight × max_gross（与组合层 RiskLimits 同口径）。"""
    ctx = GateContext(equity=100000., buying_power=1e6, current_qty={}, broker_prices={},
                      adv_usd={"AAA": 1e9, "BBB": 1e9}, sectors={"AAA": 45, "BBB": 45})
    lim = _lim(max_gross=.5, max_sector_weight=.3)               # 上限 0.15；写成除法则是 0.6
    dec = check(_buys({"AAA": .08, "BBB": .08}), ctx, lim)
    assert [(o.ticker, r) for o, r in dec.rejected] == [("BBB", "exceeds_sector_limit")]


def test_exactly_at_the_sector_cap_is_allowed():
    ctx = GateContext(equity=100000., buying_power=1e6, current_qty={}, broker_prices={},
                      adv_usd={"AAA": 1e9, "BBB": 1e9}, sectors={"AAA": 45, "BBB": 45})
    dec = check(_buys({"AAA": .05, "BBB": .05}), ctx, _lim(max_sector_weight=.1))
    assert len(dec.approved) == 2 and not dec.rejected


def test_reductions_pass_even_when_the_book_is_over_every_limit():
    """账户已经超 gross / net 上限时，减仓单必须放行 —— 卡住它们等于把超限锁死。"""
    o = build_rebalance_orders(pd.Series({"AAA": .05}), {"AAA": 150.}, pd.Series({"AAA": 100.}),
                               100000., date(2025, 3, 3), book_id=-1, band_bps=0.).orders
    assert o[0].side == "SELL" and o[0].qty == 100
    ctx = GateContext(equity=100000., buying_power=1e6, current_qty={"AAA": 150.},
                      broker_prices={"AAA": 100.}, adv_usd={"AAA": 1e9})
    dec = check(o, ctx, _lim(max_name_weight=.1, max_gross=.1, max_net=.1))
    assert len(dec.approved) == 1 and not dec.rejected, dec.rejected


# ---------------------------------------------------------------------------
# 订单层无交易带（审计 F07）：相对**券商实际持仓**判断；超限时减仓不被吞
# 数值用 2 的幂（价格 1、权益 1024），权重比较在浮点下精确，边界可以被精确测试。
# ---------------------------------------------------------------------------

EQ, PX = 1024.0, 1.0


def _band(target, held, band, name_cap=None, gross_cap=None):
    tw = pd.Series(target, dtype=float)
    names = sorted(set(tw.index) | set(held))
    return build_rebalance_orders(tw, held, pd.Series(PX, index=names), EQ, date(2025, 3, 3),
                                  book_id=-1, band_bps=0., no_trade_band=band,
                                  name_cap=name_cap, gross_cap=gross_cap)


def _did(res, tk):
    return [(o.side, o.qty) for o in res.orders if o.ticker == tk]


def _skipped(res, tk):
    return any(s["ticker"] == tk and s["reason"] == "no_trade_band" for s in res.skipped)


@pytest.mark.parametrize("bad", [float("nan"), -0.125, float("inf")])
def test_no_trade_band_must_be_finite_and_non_negative(bad):
    with pytest.raises(ValueError, match="无交易带"):
        _band({"A": .25}, {}, bad)


def test_band_is_measured_against_actual_holdings():
    # 持仓 128 股 = 0.125，目标 0.25：差 0.125 < 带宽 0.2 → 不调（持仓权重按 股数×价格÷权益 算）
    res = _band({"A": .25}, {"A": 128.0}, .2)
    assert _skipped(res, "A") and _did(res, "A") == []
    s = next(x for x in res.skipped if x["ticker"] == "A")
    assert s["current_weight"] == .125 and s["target_weight"] == .25


def test_a_move_exactly_at_the_band_is_traded():
    res = _band({"A": .25}, {"A": 128.0}, .125)          # 差恰好 = 带宽：严格小于才不调
    assert not _skipped(res, "A") and _did(res, "A") == [("BUY", 128)]


def test_band_zero_disables_it():
    res = _band({"A": .25}, {"A": 255.0}, 0.0)           # 只差 1 股也照调
    assert _did(res, "A") == [("BUY", 1)]


def test_small_reductions_inside_the_band_are_skipped_without_limits():
    res = _band({"A": .0625}, {"A": 128.0}, .1)           # 0.125 → 0.0625，差 0.0625 < 0.1
    assert _skipped(res, "A")


def test_reductions_bypass_the_band_when_the_name_is_over_its_cap():
    res = _band({"A": .125}, {"A": 192.0}, .1, name_cap=.125)   # 持仓 0.1875 > 单票 0.125
    assert _did(res, "A") == [("SELL", 64)]


def test_a_name_exactly_at_its_cap_is_not_over_it():
    res = _band({"A": .0625}, {"A": 128.0}, .1, name_cap=.125)  # 持仓恰好 = 上限
    assert _skipped(res, "A")


def test_reductions_bypass_the_band_when_the_book_is_over_gross():
    held = {"A": 128.0, "B": 128.0, "C": 128.0}                  # gross 0.375 > 0.25
    res = _band({"A": .0625, "B": .125, "C": .125}, held, .1, gross_cap=.25)
    assert _did(res, "A") == [("SELL", 64)]


def test_a_book_exactly_at_gross_is_not_over_it():
    held = {"A": 128.0, "B": 128.0}                              # gross 0.25 = 上限
    res = _band({"A": .0625, "B": .125}, held, .1, gross_cap=.25)
    assert _skipped(res, "A")


def test_increases_inside_the_band_stay_skipped_even_when_over_a_cap():
    res = _band({"A": .25}, {"A": 192.0}, .1, name_cap=.125, gross_cap=.125)
    assert _skipped(res, "A")


def test_a_reversal_is_not_a_reduction_for_the_band():
    """+128 股 → −128 股：|目标| = |持仓|，不是减仓，超限也不能借"减仓优先"绕过带宽。"""
    tw = pd.Series({"A": -.125})
    res = build_rebalance_orders(tw, {"A": 128.0}, pd.Series({"A": PX}), EQ, date(2025, 3, 3),
                                 book_id=-1, band_bps=0., allow_short=True, no_trade_band=.3,
                                 name_cap=.0625)
    assert _skipped(res, "A") and res.orders == []


def test_partially_filled_open_orders_count_only_their_remainder(tmp_path):
    """在途包络按**剩余**股数算：已成交的部分已经在持仓里了，再算一遍就是重复计入。"""
    e = _env(tmp_path)
    _cycle(e, D1)
    e.ctx.now = "2025-03-04 09:30:01.000"
    e.ctx.open_session({"AAA": 50.2, "BBB": 20.05, "CCC": 10.02}, ratio=0.5)
    e.om.reconcile()
    env = {o.ticker: o for o in e.om._open_order_envelope()}
    assert env["AAA"].remaining_qty == 100.0 and env["AAA"].side == "BUY"   # 200 − 100
    assert env["BBB"].remaining_qty == 200.0 and env["CCC"].remaining_qty == 250.0
    assert env["AAA"].limit_price == 50.25


@pytest.mark.parametrize("assets,cancelled", [(75_000.0, True), (75_001.0, False)])
def test_daily_loss_exactly_at_the_limit_halts(tmp_path, assets, cancelled):
    """日收益恰好 = −阈值（−25%，浮点精确）也算触发，与下单前风控门同口径（≤）。"""
    e = _no_wait(_env(tmp_path))
    e.om.limits = PreTradeLimits(max_name_weight=.10, max_gross=1.0, max_participation_pct=.10,
                                 max_daily_loss=.25, max_price_deviation=.15)
    _cycle(e, D1)
    e.ctx.now = "2025-03-04 09:30:01.000"
    e.ctx.open_session({"AAA": 50.2, "BBB": 20.05, "CCC": 10.02})
    e.ctx.now = "2025-03-04 16:30:00.000"
    w = {"AAA": .10, "BBB": .08, "CCC": .05, "DDD": .02}
    px = {"AAA": 50.0, "BBB": 20.0, "CCC": 10.0, "DDD": 5.0}
    adv = {k: 1e9 for k in w}
    _cycle(e, D2, w=w, px=px, adv=adv)
    ddd = next(r for r in e.es.orders() if r.ticker == "DDD")
    trusted = e.es.trusted_before(D2)
    e.ctx.account_override = {"total_assets": trusted.total_assets * assets / 100_000.0}
    rep = _cycle(e, D2, w=w, px=px, adv=adv)
    assert (ddd.client_id in rep.cancelled_stale) is cancelled, rep.reconcile["day_return"]


def test_daily_loss_halt_keeps_a_partially_filled_reduction(tmp_path):
    """
    日亏熔断只撤**加风险**的在途单。部分成交的减仓卖单按**剩余**股数判断：
    持仓 120、卖单剩 80 → 卖完还剩 40，是减仓，必须留着；剩余量算错（例如算成 240）
    就会被判成"卖穿反手"而撤掉 —— 熔断时反倒取消了降风险的单。
    """
    e = _no_wait(_env(tmp_path))
    _cycle(e, D1)
    e.ctx.now = "2025-03-04 09:30:01.000"
    e.ctx.open_session({"AAA": 50.2, "BBB": 20.05, "CCC": 10.02})
    e.ctx.now = "2025-03-04 16:30:00.000"
    w = {"AAA": .02, "BBB": .08, "CCC": .05}
    _cycle(e, D2, w=w, px={"AAA": 50.0, "BBB": 20.0, "CCC": 10.0})
    sell = next(r for r in e.es.orders() if r.ticker == "AAA" and r.side == "SELL")
    assert sell.qty == 160
    e.ctx.open_session({"AAA": 50.0}, ratio=0.5)                  # 卖出 80，持仓 120
    e.ctx.account_override = {"total_assets": 70_000.0}           # 日亏 30%
    rep = _cycle(e, D2, w=w, px={"AAA": 50.0, "BBB": 20.0, "CCC": 10.0})
    assert rep.reconcile["day_return"] < -0.05
    assert sell.client_id not in rep.cancelled_stale, "熔断撤掉了部分成交的减仓卖单"
    assert e.ctx.orders[sell.broker_order_id]["order_status"] == "FILLED_PART"


def test_an_engaged_kill_switch_flattens_held_positions_with_no_open_orders(tmp_path):
    e = _no_wait(_env(tmp_path))
    _cycle(e, D1)
    e.ctx.now = "2025-03-04 09:30:01.000"
    e.ctx.open_session({"AAA": 50.2, "BBB": 20.05, "CCC": 10.02})       # 全部成交，无在途单
    e.es.set_kill_switch(True, "kevin", "drill")
    rep = e.om.maintain("daily")
    sells = {p[0]: p[2] for p in _placed(e) if p[1] == "SELL"}
    assert rep.flatten["status"] == "resubmitted"
    assert sells == {"US.AAA": 200, "US.BBB": 400, "US.CCC": 500}


def test_an_engaged_kill_switch_on_a_flat_account_reports_flat(tmp_path):
    e = _no_wait(_env(tmp_path))
    e.clock.after_close(D1)
    e.es.set_kill_switch(True, "kevin", "drill")
    rep = e.om.maintain("daily")
    assert rep.flatten == {"status": "flat"} and _placed(e) == []


def test_exactly_at_the_net_cap_is_allowed():
    ctx = GateContext(equity=100000., buying_power=1e6, current_qty={}, broker_prices={},
                      adv_usd={"AAA": 1e9, "BBB": 1e9})
    dec = check(_buys({"AAA": .05, "BBB": .05}), ctx, _lim(max_net=.1))
    assert len(dec.approved) == 2 and not dec.rejected, dec.rejected


def test_band_weights_use_price_times_shares_over_equity():
    """价格 ≠ 1 时持仓权重 = 股数 × 价格 ÷ 权益：64 股 × $2 ÷ 1024 = 0.125。"""
    tw = pd.Series({"A": .25})
    res = build_rebalance_orders(tw, {"A": 64.0}, pd.Series({"A": 2.0}), EQ, date(2025, 3, 3),
                                 book_id=-1, band_bps=0., no_trade_band=.2)
    s = [x for x in res.skipped if x["reason"] == "no_trade_band"]
    assert s and s[0]["current_weight"] == .125 and res.orders == []
