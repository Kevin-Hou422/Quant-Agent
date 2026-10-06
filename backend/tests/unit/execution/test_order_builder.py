"""
test_order_builder.py — Phase 12.1 目标权重 → 股数订单

断言用具体数字（股数、限价），不用"有订单 / 方向对"这类退化实现也能满足的弱断言（§T）。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.core.execution.order_builder import (
    PURPOSE_FLATTEN, PURPOSE_REBALANCE, build_flatten_orders, build_rebalance_orders,
    is_own_client_id, make_client_id, round_to_tick,
)
from app.core.execution.broker_gateway import BrokerPosition
from app.core.execution.order_manager import parse_client_id

D = date(2025, 3, 3)


def _build(w, cur=None, px=None, equity=100_000.0, band=50.0, allow_short=False):
    return build_rebalance_orders(
        pd.Series(w, dtype=float), cur or {}, pd.Series(px, dtype=float), equity, D,
        book_id=-1, band_bps=band, allow_short=allow_short)


def _by_ticker(res):
    return {o.ticker: o for o in res.orders}


def test_share_counts_and_limits_are_exact():
    res = _build({"AAA": 0.10, "BBB": 0.08, "CCC": 0.05},
                 px={"AAA": 33.0, "BBB": 20.0, "CCC": 0.85})
    o = _by_ticker(res)
    assert o["AAA"].qty == 303 and o["AAA"].side == "BUY"       # trunc(10000/33)=303
    assert o["AAA"].limit_price == 33.16                          # 33×1.005=33.165 → 买单向下取整
    assert o["BBB"].qty == 400 and o["BBB"].limit_price == 20.10
    assert o["CCC"].qty == 5882                                   # trunc(5000/0.85)=5882
    assert o["CCC"].limit_price == 0.8542                         # <$1 用 0.0001 价位：0.85425→0.8542
    assert o["AAA"].client_id == "qaR-1-20250303-AAA-B"


def test_delta_against_current_position_and_sells_first():
    res = _build({"AAA": 0.10, "BBB": 0.02}, cur={"AAA": 100, "BBB": 300, "OLD": 50},
                 px={"AAA": 50.0, "BBB": 20.0, "OLD": 10.0})
    o = _by_ticker(res)
    assert (o["AAA"].side, o["AAA"].qty) == ("BUY", 100)          # 目标 200，已有 100
    assert (o["BBB"].side, o["BBB"].qty) == ("SELL", 200)         # 目标 100，已有 300
    assert o["BBB"].limit_price == 19.90                          # 20×0.995，卖单向上取整
    assert (o["OLD"].side, o["OLD"].qty) == ("SELL", 50), "目标里没有的持仓必须卖掉"
    assert [x.side for x in res.orders] == ["SELL", "SELL", "BUY"], "必须先卖后买"


def test_no_order_when_already_at_target():
    res = _build({"AAA": 0.10}, cur={"AAA": 200}, px={"AAA": 50.0})
    assert res.orders == [] and res.skipped == [], "已在目标上的名字不该出现在任何清单里"


def test_rounding_drift_value_for_a_nonzero_target():
    res = _build({"AAA": 0.10}, px={"AAA": 33.0})
    # 303 股 × 33 / 100000 − 0.10 = −0.00001
    assert res.rounding["AAA"] == pytest.approx(303 * 33 / 100_000 - 0.10, abs=1e-15)


def test_dust_holdings_are_treated_as_zero():
    from app.core.execution.broker_gateway import DUST_QTY
    res = _build({}, cur={"DUST": DUST_QTY}, px={"DUST": 10.0})
    assert res.orders == [] and res.skipped == [] and res.rounding == {}


def test_zero_target_is_not_reported_as_a_short():
    res = _build({"AAA": 0.0}, cur={"AAA": 10}, px={"AAA": 10.0})
    assert res.notes == []
    assert [(o.side, o.qty) for o in res.orders] == [("SELL", 10)]


def test_rounding_never_buys_more_than_target():
    rng = np.random.default_rng(7)
    for _ in range(200):
        w, px = float(rng.uniform(0, 0.1)), float(rng.uniform(0.5, 900))
        o = _by_ticker(_build({"X": w}, px={"X": px})).get("X")
        qty = o.qty if o else 0
        assert qty * px <= w * 100_000 + 1e-6
        assert (qty + 1) * px > w * 100_000 - 1e-6, "取整多丢了一整股"


def test_rounding_drift_is_reported():
    res = _build({"PRICY": 0.05}, px={"PRICY": 7000.0})          # $5000 买不起一股 $7000
    assert res.orders == []
    assert res.rounding["PRICY"] == pytest.approx(-0.05)


def test_short_target_clipped_when_not_allowed_and_kept_when_allowed():
    res = _build({"AAA": -0.05}, cur={"AAA": 10}, px={"AAA": 50.0})
    o = _by_ticker(res)["AAA"]
    assert (o.side, o.qty, o.target_qty) == ("SELL", 10, 0.0)
    assert any("不允许做空" in n for n in res.notes)
    res2 = _build({"AAA": -0.05}, cur={"AAA": 10}, px={"AAA": 50.0}, allow_short=True)
    assert _by_ticker(res2)["AAA"].qty == 110                     # 10 → −100


def test_missing_price_skips_the_ticker_and_says_why():
    res = _build({"AAA": 0.1, "BBB": 0.1}, cur={"BBB": 5},
                 px={"AAA": float("nan"), "BBB": 0.0})
    assert res.orders == []
    assert {s["ticker"]: s["reason"] for s in res.skipped} == {
        "AAA": "no_reference_price", "BBB": "no_reference_price"}


def test_nan_weight_means_zero_target():
    o = _by_ticker(_build({"AAA": float("nan")}, cur={"AAA": 7}, px={"AAA": 10.0}))["AAA"]
    assert (o.side, o.qty) == ("SELL", 7)


def test_fractional_holdings_leave_only_whole_share_orders():
    res = _build({"AAA": 0.0}, cur={"AAA": 0.4}, px={"AAA": 10.0})
    assert res.orders == [] and res.skipped[0]["reason"] == "fractional_remainder"


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_equity_is_refused(bad):
    with pytest.raises(ValueError):
        _build({"AAA": 0.1}, px={"AAA": 10.0}, equity=bad)


def test_invalid_band_is_refused():
    with pytest.raises(ValueError):
        _build({"AAA": 0.1}, px={"AAA": 10.0}, band=-1.0)


# ---------------------------------------------------------------------------
# 价位取整与 client_id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("price,side,expected", [
    (33.165, "BUY", 33.16), (33.165, "SELL", 33.17),
    (10.0, "BUY", 10.0), (10.0, "SELL", 10.0),
    (0.85425, "BUY", 0.8542), (0.85425, "SELL", 0.8543),
    (0.99999, "BUY", 0.9999), (1.004, "SELL", 1.01),
])
def test_round_to_tick(price, side, expected):
    assert round_to_tick(price, side) == expected


@pytest.mark.parametrize("bad", [0.0, -3.0, float("nan")])
def test_round_to_tick_rejects_bad_prices(bad):
    with pytest.raises(ValueError):
        round_to_tick(bad, "BUY")


def test_client_ids_are_deterministic_distinct_and_parseable():
    a = make_client_id(PURPOSE_REBALANCE, -1, "20250303", "BRK-B", "BUY")
    assert a == make_client_id(PURPOSE_REBALANCE, -1, "20250303", "BRK-B", "BUY")
    variants = {a,
                make_client_id(PURPOSE_REBALANCE, -1, "20250303", "BRK-B", "SELL"),
                make_client_id(PURPOSE_REBALANCE, -1, "20250304", "BRK-B", "BUY"),
                make_client_id(PURPOSE_FLATTEN, -1, "20250303", "BRK-B", "BUY")}
    assert len(variants) == 4
    assert parse_client_id(a) == {"purpose": "rebalance", "book_id": -1,
                                  "decision_date": date(2025, 3, 3), "ticker": "BRK-B",
                                  "side": "BUY"}
    f = make_client_id(PURPOSE_FLATTEN, -1, "20250303153000", "AAPL", "SELL")
    assert parse_client_id(f)["purpose"] == "flatten"
    assert parse_client_id(f)["decision_date"] == date(2025, 3, 3)
    assert is_own_client_id(a) and is_own_client_id(f)
    assert not is_own_client_id("manual order") and not is_own_client_id("")
    assert parse_client_id("manual order") is None


def test_client_id_length_limit():
    with pytest.raises(ValueError, match="64"):
        make_client_id(PURPOSE_REBALANCE, -1, "20250303", "X" * 60, "BUY")


def test_client_id_of_exactly_64_bytes_is_accepted():
    """SDK 的上限是 ≤ 64 字节：恰好 64 必须能用，65 才拒。"""
    cid = make_client_id(PURPOSE_REBALANCE, -1, "20250303", "X" * 47, "BUY")
    assert len(cid.encode("utf-8")) == 64
    with pytest.raises(ValueError):
        make_client_id(PURPOSE_REBALANCE, -1, "20250303", "X" * 48, "BUY")


def test_default_is_long_only():
    """不传 allow_short 时必须按只做多处理（§U：兜底朝保守一侧）。"""
    res = build_rebalance_orders(pd.Series({"AAA": -0.05}), {"AAA": 10}, pd.Series({"AAA": 50.0}),
                                 100_000.0, D, book_id=-1, band_bps=50.0)
    o = res.orders[0]
    assert (o.side, o.qty, o.target_qty) == ("SELL", 10, 0.0)


def test_zero_band_is_allowed_and_means_limit_at_reference():
    o = _by_ticker(_build({"AAA": 0.1}, px={"AAA": 50.0}, band=0.0))["AAA"]
    assert o.limit_price == 50.0 and o.qty == 200


# ---------------------------------------------------------------------------
# 一键全平
# ---------------------------------------------------------------------------

def _pos(tk, qty, px):
    return BrokerPosition(tk, qty, max(qty, 0), px, qty * px)


def test_flatten_orders_reverse_every_position():
    res = build_flatten_orders(
        {"AAA": _pos("AAA", 120, 50.0), "SHT": _pos("SHT", -30, 10.0), "ZERO": _pos("ZERO", 0, 5.0)},
        "20250303153000", book_id=-1, band_bps=500.0)
    o = _by_ticker(res)
    assert set(o) == {"AAA", "SHT"}
    assert (o["AAA"].side, o["AAA"].qty, o["AAA"].limit_price) == ("SELL", 120, 47.5)
    assert (o["SHT"].side, o["SHT"].qty, o["SHT"].limit_price) == ("BUY", 30, 10.5)
    assert all(x.purpose == PURPOSE_FLATTEN and x.target_qty == 0.0 for x in res.orders)
    assert res.orders[0].side == "SELL"


def test_flatten_ignores_dust_and_skips_fractional_remainders():
    from app.core.execution.broker_gateway import DUST_QTY
    res = build_flatten_orders({"D": _pos("D", DUST_QTY, 10.0), "F": _pos("F", 0.5, 10.0)},
                               "20250303153000", book_id=-1, band_bps=500.0)
    assert res.orders == []
    assert res.skipped == [{"ticker": "F", "reason": "fractional_remainder", "current_qty": 0.5}]


def test_flatten_skips_positions_without_a_price():
    res = build_flatten_orders({"AAA": _pos("AAA", 5, float("nan"))}, "20250303153000",
                               book_id=-1, band_bps=500.0)
    assert res.orders == [] and res.skipped[0]["reason"] == "no_broker_price"
