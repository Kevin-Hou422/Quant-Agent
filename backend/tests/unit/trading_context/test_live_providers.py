"""
test_live_providers.py — Phase 12 兑现 TR.3 的 live 模式：账户走券商网关、盘口走行情快照。

纪律（§J T3 / §U）：实时数据拿不到就抛错，**绝不**退回估计值、绝不用 {} 冒充空仓。
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.trading_context import get_trade_providers
from app.core.trading_context.providers import (
    LiveAccountProvider, LiveBorrowProvider, LiveQuoteProvider,
)
from tests.unit.execution.moomoo_fake import FakeTradeContext, make_gateway


def _snap(rows):
    def _fn(tickers):
        return pd.DataFrame(rows)
    return _fn


def test_live_account_reports_broker_state_as_weights():
    ctx = FakeTradeContext(cash=60_000.0)
    ctx.set_position("AAPL", 100, 200.0)                        # 20000
    ctx.set_position("BRK-B", 50, 400.0)                        # 20000
    acc = LiveAccountProvider(make_gateway(ctx))
    assert acc.cash() == 60_000.0 and acc.buying_power() == 60_000.0
    assert acc.positions() == {"AAPL": pytest.approx(0.2), "BRK-B": pytest.approx(0.2)}


def test_live_account_failure_raises_instead_of_empty_book():
    ctx = FakeTradeContext()
    acc = LiveAccountProvider(make_gateway(ctx))
    ctx.fail["position_list_query"] = "disconnected"
    with pytest.raises(Exception, match="disconnected"):
        acc.positions()


def test_live_account_with_non_positive_assets_refuses():
    ctx = FakeTradeContext(cash=0.0)
    with pytest.raises(RuntimeError, match="总资产"):
        LiveAccountProvider(make_gateway(ctx)).positions()


def test_live_quote_spread_and_mid_from_bid_ask():
    q = LiveQuoteProvider(_snap([{"code": "US.AAPL", "bid_price": 199.98, "ask_price": 200.02}]))
    assert q.mid_price("AAPL") == pytest.approx(200.0)
    assert q.spread_bps("AAPL") == pytest.approx(2.0)           # 0.04 / 200 × 1e4


@pytest.mark.parametrize("bid,ask", [(0.0, 0.0), (200.1, 200.0), (float("nan"), 200.0)])
def test_live_quote_refuses_invalid_quotes(bid, ask):
    q = LiveQuoteProvider(_snap([{"code": "US.AAPL", "bid_price": bid, "ask_price": ask}]))
    with pytest.raises(RuntimeError, match="无有效买卖报价"):
        q.spread_bps("AAPL")


def test_live_quote_missing_ticker_raises():
    q = LiveQuoteProvider(_snap([{"code": "US.MSFT", "bid_price": 1.0, "ask_price": 1.01}]))
    with pytest.raises(RuntimeError, match="没有行情快照"):
        q.mid_price("AAPL")


def test_live_borrow_is_long_only_by_config_and_never_guesses_fees():
    b = LiveBorrowProvider(allow_short=False)
    assert b.is_shortable("AAPL") is False
    with pytest.raises(NotImplementedError, match="单位未经核实"):
        b.borrow_fee_bps("AAPL")
    b2 = LiveBorrowProvider(allow_short=True, snapshot_fn=_snap(
        [{"code": "US.AAPL", "enable_short_sell": True}, {"code": "US.GME", "enable_short_sell": False}]))
    assert b2.is_shortable("AAPL") is True and b2.is_shortable("GME") is False
    with pytest.raises(RuntimeError):
        LiveBorrowProvider(allow_short=True).is_shortable("AAPL")


def test_locked_market_is_a_valid_quote():
    q = LiveQuoteProvider(_snap([{"code": "US.AAPL", "bid_price": 200.0, "ask_price": 200.0}]))
    assert q.spread_bps("AAPL") == 0.0 and q.mid_price("AAPL") == 200.0


def test_factory_defaults_to_long_only_even_if_the_broker_allows_shorting():
    ctx = FakeTradeContext()
    tp = get_trade_providers("live", gateway=make_gateway(ctx), snapshot_fn=_snap(
        [{"code": "US.AAPL", "enable_short_sell": True}]))
    assert tp.borrow.is_shortable("AAPL") is False


class _QuoteCtx:
    instances = []

    def __init__(self, host, port, ret=0, data=None):
        self.host, self.port, self.ret, self.data = host, port, ret, data
        self.closed, self.requested = False, None
        _QuoteCtx.instances.append(self)

    def get_market_snapshot(self, code_list):
        self.requested = list(code_list)
        return self.ret, self.data

    def close(self):
        self.closed = True


@pytest.mark.parametrize("ret,data,ok", [
    (0, pd.DataFrame([{"code": "US.BRK.B", "bid_price": 1.0, "ask_price": 1.1}]), True),
    (-1, "no quote permission", False),
])
def test_snapshot_fn_checks_return_code_and_always_closes(monkeypatch, ret, data, ok):
    import moomoo
    import app.core.execution.broker_gateway as bg
    from app.core.trading_context.providers import moomoo_snapshot_fn
    _QuoteCtx.instances.clear()
    monkeypatch.setattr(bg, "opend_reachable", lambda *a, **k: True)
    monkeypatch.setattr(moomoo, "OpenQuoteContext",
                        lambda host, port: _QuoteCtx(host, port, ret, data))
    fn = moomoo_snapshot_fn("127.0.0.1", 11111)
    if ok:
        out = fn(["BRK-B"])
        assert out is data
    else:
        with pytest.raises(RuntimeError, match="no quote permission"):
            fn(["BRK-B"])
    c = _QuoteCtx.instances[-1]
    assert c.requested == ["US.BRK.B"] and c.closed is True


def test_snapshot_fn_never_constructs_when_opend_is_down(monkeypatch):
    import moomoo
    import app.core.execution.broker_gateway as bg
    from app.core.trading_context.providers import moomoo_snapshot_fn
    monkeypatch.setattr(bg, "opend_reachable", lambda *a, **k: False)
    monkeypatch.setattr(moomoo, "OpenQuoteContext",
                        lambda **k: pytest.fail("OpenD 不在时构造了行情上下文（会永久阻塞）"))
    with pytest.raises(RuntimeError, match="未在监听"):
        moomoo_snapshot_fn("127.0.0.1", 11111)(["AAPL"])


def test_factory_live_mode_wires_all_three():
    ctx = FakeTradeContext(cash=100_000.0)
    tp = get_trade_providers("live", gateway=make_gateway(ctx),
                             snapshot_fn=_snap([{"code": "US.AAPL", "bid_price": 1.0, "ask_price": 1.0}]))
    assert tp.mode == "live"
    assert tp.to_dict() == {"mode": "live", "buying_power": 100_000.0, "n_positions": 0}
