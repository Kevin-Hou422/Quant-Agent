"""
test_pretrade_gate.py — Phase 12.2 下单前风控门

每条规则都测两侧：越界的单被拦（reason 精确），边界内的单放行。
只测"坏单被拒"的门，在被改成恒拒时照样绿；只测"好单通过"的门，在被改成恒放行时照样绿（§B）。
"""

from __future__ import annotations

import pytest

from app.core.execution.order_builder import PURPOSE_FLATTEN, PURPOSE_REBALANCE, PlannedOrder
from app.core.execution.pretrade_gate import (
    GateContext, PreTradeLimits, check, increases_risk,
)

LIM = PreTradeLimits(max_name_weight=0.10, max_gross=1.0, max_participation_pct=0.10,
                     max_daily_loss=0.05, max_price_deviation=0.15)


def _o(tk, side, qty, px=100.0, cur=0.0, purpose=PURPOSE_REBALANCE, limit=None):
    if limit is None:
        limit = px * (1.005 if side == "BUY" else 0.995)
    return PlannedOrder(client_id=f"qaR-1-20250303-{tk}-{side[0]}", ticker=tk, side=side, qty=qty,
                        ref_price=px, limit_price=limit, current_qty=cur,
                        target_qty=cur + (qty if side == "BUY" else -qty), target_weight=0.0,
                        purpose=purpose)


def _ctx(cur=None, prices=None, adv=None, equity=100_000.0, power=100_000.0,
         day_return=0.0, kill=False):
    cur = cur or {}
    return GateContext(equity=equity, buying_power=power, current_qty=cur,
                       broker_prices=prices if prices is not None else {k: 100.0 for k in cur},
                       adv_usd=adv if adv is not None else {}, day_return=day_return,
                       kill_switch=kill)


BIG_ADV = {t: 1e9 for t in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K")}


def _reasons(dec):
    return {o.ticker: r for o, r in dec.rejected}


def _ok(dec):
    return {o.ticker for o in dec.approved}


# ---------------------------------------------------------------------------

def test_kill_switch_blocks_rebalance_but_lets_flatten_through():
    orders = [_o("A", "SELL", 10, cur=50, purpose=PURPOSE_FLATTEN), _o("B", "BUY", 10)]
    dec = check(orders, _ctx(cur={"A": 50}, adv=BIG_ADV, kill=True), LIM)
    assert _ok(dec) == {"A"} and _reasons(dec) == {"B": "kill_switch_engaged"}
    assert dec.halted == "kill_switch_engaged"


def test_daily_loss_halt_blocks_only_risk_increasing_orders():
    orders = [_o("A", "SELL", 10, cur=50), _o("B", "BUY", 10)]
    dec = check(orders, _ctx(cur={"A": 50}, adv=BIG_ADV, day_return=-0.06), LIM)
    assert _ok(dec) == {"A"} and _reasons(dec) == {"B": "daily_loss_halt"}
    assert dec.halted == "daily_loss_halt"
    dec2 = check(orders, _ctx(cur={"A": 50}, adv=BIG_ADV, day_return=-0.049), LIM)
    assert _ok(dec2) == {"A", "B"} and dec2.halted == ""


def test_daily_loss_at_exactly_the_limit_halts():
    dec = check([_o("B", "BUY", 10)], _ctx(adv=BIG_ADV, day_return=-0.05), LIM)
    assert _reasons(dec) == {"B": "daily_loss_halt"}


def test_no_previous_snapshot_is_noted_not_assumed():
    dec = check([_o("B", "BUY", 10)], _ctx(adv=BIG_ADV, day_return=None), LIM)
    assert _ok(dec) == {"B"}
    assert any("无上一交易日快照" in n for n in dec.notes)


def test_reference_price_must_be_valid():
    dec = check([_o("A", "BUY", 1, px=float("nan"), limit=1.0),
                 _o("B", "BUY", 1, px=0.0, limit=1.0)], _ctx(adv=BIG_ADV), LIM)
    assert _reasons(dec) == {"A": "no_reference_price", "B": "no_reference_price"}


def test_price_mismatch_against_broker_quote():
    orders = [_o("A", "SELL", 5, px=100.0, cur=10), _o("B", "SELL", 5, px=100.0, cur=10)]
    dec = check(orders, _ctx(cur={"A": 10, "B": 10}, prices={"A": 120.0, "B": 110.0},
                             adv=BIG_ADV), LIM)
    # |100−120|/120 = 16.7% > 15% → 拒；|100−110|/110 = 9.1% → 放行
    assert _reasons(dec) == {"A": "price_mismatch"} and _ok(dec) == {"B"}


def test_long_only_cannot_sell_more_than_held():
    dec = check([_o("A", "SELL", 11, cur=10), _o("B", "SELL", 10, cur=10)],
                _ctx(cur={"A": 10, "B": 10}, adv=BIG_ADV), LIM)
    assert _reasons(dec) == {"A": "would_short"} and _ok(dec) == {"B"}
    lim = PreTradeLimits(**{**LIM.__dict__, "allow_short": True})
    dec2 = check([_o("A", "SELL", 11, cur=10)], _ctx(cur={"A": 10}, adv=BIG_ADV), lim)
    assert _ok(dec2) == {"A"}


def test_fat_finger_vs_adv_and_missing_adv_fails_closed():
    orders = [_o("A", "BUY", 11, px=100.0), _o("B", "BUY", 10, px=100.0), _o("C", "BUY", 1)]
    # 10% × ADV 10_000 = $1000：A 是 $1100（拒），B 恰好 $1000（放行），C 没有 ADV（拒）
    dec = check(orders, _ctx(adv={"A": 10_000, "B": 10_000}), LIM)
    assert _reasons(dec) == {"A": "exceeds_adv_participation", "C": "no_adv"}
    assert _ok(dec) == {"B"}


def test_flatten_is_exempt_from_adv_limit():
    o = _o("A", "SELL", 500, cur=500, purpose=PURPOSE_FLATTEN)
    dec = check([o], _ctx(cur={"A": 500}, adv={}, kill=True), LIM)
    assert _ok(dec) == {"A"}


def test_name_limit_only_blocks_increases():
    # 100 股 × $100 / $100k = 10%：恰好上限；101 股 = 10.1%
    dec = check([_o("A", "BUY", 101), _o("B", "BUY", 100)], _ctx(adv=BIG_ADV), LIM)
    assert _reasons(dec) == {"A": "exceeds_name_limit"} and _ok(dec) == {"B"}
    # 已经超限的持仓减一点仍超限 —— 但减仓必须放行
    dec2 = check([_o("A", "SELL", 10, cur=150)], _ctx(cur={"A": 150}, adv=BIG_ADV), LIM)
    assert _ok(dec2) == {"A"}


def test_gross_limit_is_cumulative_across_orders():
    lim = PreTradeLimits(**{**LIM.__dict__, "max_gross": 0.25})
    held = {"H": 100}                                       # 已有 10%
    orders = [_o("A", "BUY", 100), _o("B", "BUY", 50), _o("C", "BUY", 1)]
    dec = check(orders, _ctx(cur=held, adv=BIG_ADV), lim)
    # 10% + A 10% = 20% 放行；+ B 5% = 25% 恰好放行；+ C 0.1% = 25.1% 拒
    assert _ok(dec) == {"A", "B"} and _reasons(dec) == {"C": "exceeds_gross_limit"}


def test_held_position_without_any_price_makes_gross_unpriceable():
    dec = check([_o("A", "BUY", 1)], _ctx(cur={"H": 10}, prices={}, adv=BIG_ADV), LIM)
    assert _reasons(dec) == {"A": "gross_unpriceable"}


def test_buying_power_is_cumulative_at_limit_price_and_ignores_sell_proceeds():
    orders = [_o("S", "SELL", 50, cur=50),                   # 卖出所得不计入
              _o("A", "BUY", 50, limit=100.0),               # $5000
              _o("B", "BUY", 50, limit=100.0),               # 累计 $10000（恰好）
              _o("C", "BUY", 1, limit=100.0)]                # 累计 $10100 → 拒
    dec = check(orders, _ctx(cur={"S": 50}, adv={**BIG_ADV, "S": 1e9}, power=10_000.0), LIM)
    assert _ok(dec) == {"S", "A", "B"} and _reasons(dec) == {"C": "insufficient_buying_power"}


def test_rejected_orders_do_not_consume_budget():
    lim = PreTradeLimits(**{**LIM.__dict__, "max_gross": 0.15})
    orders = [_o("A", "BUY", 101), _o("B", "BUY", 100), _o("C", "BUY", 50)]
    dec = check(orders, _ctx(adv=BIG_ADV), lim)
    # A 超单票上限被拒，不应占用总敞口：B 10% + C 5% = 15% 恰好放行
    assert _ok(dec) == {"B", "C"} and _reasons(dec) == {"A": "exceeds_name_limit"}


def test_kill_switch_defaults_to_off():
    ctx = GateContext(equity=100_000.0, buying_power=100_000.0, current_qty={},
                      broker_prices={}, adv_usd=BIG_ADV, day_return=0.0)
    dec = check([_o("B", "BUY", 10)], ctx, LIM)
    assert _ok(dec) == {"B"} and dec.halted == ""


def test_price_deviation_exactly_at_the_limit_passes():
    # |115 − 100| / 100 = 0.15 恰好等于上限：规则是"超过"才拒
    dec = check([_o("A", "SELL", 5, px=115.0, cur=10)],
                _ctx(cur={"A": 10}, prices={"A": 100.0}, adv=BIG_ADV), LIM)
    assert _ok(dec) == {"A"}


def test_unusable_broker_prices_are_ignored_not_divided_by():
    dec = check([_o("A", "SELL", 5, cur=10), _o("B", "SELL", 5, cur=10)],
                _ctx(cur={"A": 10, "B": 10}, prices={"A": 0.0, "B": None}, adv=BIG_ADV), LIM)
    assert _ok(dec) == {"A", "B"}


def test_ending_exactly_one_dust_short_is_still_flat():
    from app.core.execution.broker_gateway import DUST_QTY
    cur = 1.0 - DUST_QTY                                   # 精确可表示
    dec = check([_o("A", "SELL", 1, cur=cur)], _ctx(cur={"A": cur}, adv=BIG_ADV), LIM)
    assert _ok(dec) == {"A"}, "卖完只剩 −1 个零股阈值，不算卖成空头"


def test_dust_holdings_do_not_make_gross_unpriceable():
    from app.core.execution.broker_gateway import DUST_QTY
    dec = check([_o("A", "BUY", 1)], _ctx(cur={"H": DUST_QTY}, prices={}, adv=BIG_ADV), LIM)
    assert _ok(dec) == {"A"}


def test_buying_back_a_short_needs_buying_power_too():
    lim = PreTradeLimits(**{**LIM.__dict__, "allow_short": True})
    dec = check([_o("S", "BUY", 10, cur=-10)],
                _ctx(cur={"S": -10}, adv={"S": 1e9}, power=500.0), lim)
    assert _reasons(dec) == {"S": "insufficient_buying_power"}, "回补空头同样要花现金"


@pytest.mark.parametrize("equity", [0.0, -1.0, float("nan"), float("inf"), None])
def test_unknown_equity_rejects_everything(equity):
    dec = check([_o("A", "SELL", 1, cur=5), _o("B", "BUY", 1)],
                _ctx(cur={"A": 5}, adv=BIG_ADV, equity=equity), LIM)
    assert dec.approved == [] and dec.halted == "equity_unavailable"
    assert set(_reasons(dec).values()) == {"equity_unavailable"}


def test_increases_risk_classification():
    from app.core.execution.broker_gateway import DUST_QTY
    assert increases_risk(0, 10) and increases_risk(10, 5) and increases_risk(-10, -5)
    assert not increases_risk(10, -5) and not increases_risk(-10, 5) and not increases_risk(10, -10)
    assert increases_risk(10, -15), "反手（多翻空）是增加风险"
    assert not increases_risk(10, 0), "零变动不是增加风险"
    assert not increases_risk(0, DUST_QTY), "零股阈值以内的变动不算"
    assert increases_risk(0, 2 * DUST_QTY)


@pytest.mark.parametrize("field", ["max_name_weight", "max_gross", "max_participation_pct",
                                   "max_daily_loss", "max_price_deviation"])
@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan")])
def test_limits_must_be_positive(field, bad):
    with pytest.raises(ValueError, match=field):
        PreTradeLimits(**{**LIM.__dict__, field: bad})


def test_limits_come_from_the_same_settings_as_the_portfolio_gate(monkeypatch):
    """§J 单一来源：改组合层风控 / 成本模型的参数，下单前风控门跟着变。"""
    from app.config import settings
    from app.core.backtest_engine.transaction_cost import CostParams
    monkeypatch.setattr(settings, "risk_max_name_weight", 0.07)
    monkeypatch.setattr(settings, "risk_max_gross", 0.6)
    monkeypatch.setattr(settings, "exec_max_daily_loss", 0.03)
    monkeypatch.setattr(settings, "exec_max_price_deviation", 0.2)
    monkeypatch.setattr(settings, "trading_allow_short", True)
    lim = PreTradeLimits.from_settings(settings, CostParams(max_participation_pct=0.04))
    assert (lim.max_name_weight, lim.max_gross, lim.max_participation_pct, lim.max_daily_loss,
            lim.max_price_deviation, lim.allow_short) == (0.07, 0.6, 0.04, 0.03, 0.2, True)
    assert PreTradeLimits.from_settings(settings).max_participation_pct == \
        CostParams().max_participation_pct


def test_decision_to_dict_lists_every_rejection():
    dec = check([_o("A", "BUY", 101), _o("B", "BUY", 1)], _ctx(adv={"A": 1e9}), LIM)
    d = dec.to_dict()
    assert d["n_rejected"] == 2 and d["n_approved"] == 0
    assert {r["ticker"]: r["reason"] for r in d["rejected"]} == {
        "A": "exceeds_name_limit", "B": "no_adv"}
