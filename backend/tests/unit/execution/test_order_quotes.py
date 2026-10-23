"""
加风险订单的报价：独立报价源、报价合同（有限正价、有时刻、不在未来、与参考价对得上）、按常规盘 / 盘前 / 盘后的报价时段。

来源（原测试文件的说明，保留出处）：

外部审查 2026-10-18 F08 / F09。

F08（R11）两个独立数据库会话同时认领同一条被拒的意图，两边都返回 True —— 随后两个进程都可能调券商。
     进程内 RLock 挡不住另一个进程（调度器 / run-now / API / 恢复任务）。
F09（R5）新开仓的票不在持仓里、没有券商价，下单门跳过价格偏差检查，拿自己的参考价给自己估值 ——
     空账户第一笔 BUY 没有任何独立报价就放行了。

---

复审 2026-10-19（`review-v6/audit-review/复审报告_2026-10-19.md`）N01–N09 的回归。

每条都从复审独立脚本（`independent_checks.py`）里对应的 residual_* 场景改写：原脚本断言"问题仍在"，
这里断言修复后应有的行为。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from app.core.execution.broker_gateway import BrokerError, Quote
from app.db.execution_store import ST_GATE
from tests.unit.execution.test_order_manager import (
    D1,
    PX1,
    _cycle,
    _env,
    _placed,
)

# ---------------------------------------------------------------------------
# N06：加风险订单的报价合同
# ---------------------------------------------------------------------------

def _buy(qty_held=0.0):
    from app.core.execution.order_builder import build_rebalance_orders
    held = {"AAA": qty_held} if qty_held else {}
    return build_rebalance_orders(pd.Series({"AAA": .05}), held, pd.Series({"AAA": 100.}), 10000,
                                  pd.Timestamp("2026-01-05").date(), book_id=-1, band_bps=50).orders


def _check(orders, quotes, held=0.0, now=datetime(2026, 1, 5, 22, tzinfo=timezone.utc)):
    from app.core.execution.pretrade_gate import GateContext, PreTradeLimits, check
    ctx = GateContext(10000, 10000, {"AAA": held} if held else {}, {"AAA": 100.} if held else {},
                      {"AAA": 1e9}, quotes=quotes, now=now)
    return check(orders, ctx, PreTradeLimits(.1, 1., .1, .05, .15))


# ---------------------------------------------------------------------------
# F09：独立报价
# ---------------------------------------------------------------------------

class TestIndependentQuotes:

    def test_a_first_buy_without_a_quote_is_not_placed(self, tmp_path):
        e = _env(tmp_path)
        rep = _cycle(e, D1, quote=False)
        assert _placed(e) == [] and rep.n_submitted == 0
        assert {r["reason"] for r in rep.gate["rejected"]} == {"no_independent_quote"}
        assert {r.status for r in e.es.orders()} == {ST_GATE}       # 可重试：报价回来后照常下

    def test_a_quote_far_from_the_reference_price_blocks_the_order(self, tmp_path):
        e = _env(tmp_path)
        e.ctx.quotes.update({"AAA": 50.0 * 1.3})
        e.ctx.quotes.update({"BBB": 20.0, "CCC": 10.0})
        e.clock.after_close(D1)
        rep = e.om.run_cycle(pd.Series({"AAA": .1, "BBB": .08, "CCC": .05}), pd.Series(PX1),
                             pd.Series({k: 1e9 for k in PX1}), D1)
        rej = {r["ticker"]: r["reason"] for r in rep.gate["rejected"]}
        assert rej == {"AAA": "quote_mismatch"}
        assert sorted(p[0] for p in _placed(e)) == ["US.BBB", "US.CCC"]

    def test_a_stale_quote_counts_as_none(self, tmp_path):
        e = _env(tmp_path)
        e.ctx.quotes.update(PX1)
        e.ctx.quote_time = "2025-02-28 16:00:00.000"                  # 上周五；决策日 2025-03-03 周一已收盘
        rep = _cycle(e, D1, quote=False)
        assert {r["reason"] for r in rep.gate["rejected"]} == {"stale_quote"}

    def test_a_quote_outage_blocks_new_positions_but_reports_why(self, tmp_path):
        e = _env(tmp_path)
        e.ctx.quote_fail = "OpenD quote down"
        rep = _cycle(e, D1)
        assert "OpenD quote down" in rep.quote_error and _placed(e) == []
        assert {r["reason"] for r in rep.gate["rejected"]} == {"no_independent_quote"}

    def test_reducing_an_existing_position_needs_no_quote(self, tmp_path):
        e = _env(tmp_path)
        e.ctx.set_position("AAA", 400, 50.0)
        e.ctx.quotes.clear()
        e.ctx.quote_fail = "down"
        rep = _cycle(e, D1, w={"AAA": 0.10}, px={"AAA": 50.0}, quote=False)   # 400 股 → 200 股
        assert [p[:3] for p in _placed(e)] == [("US.AAA", "SELL", 200)], rep.gate

    def test_flatten_is_never_blocked_by_quotes(self, tmp_path):
        e = _env(tmp_path)
        e.ctx.set_position("AAA", 100, 50.0)
        e.ctx.quote_fail = "down"
        e.clock.after_close(D1)
        e.om.reconcile()
        out = e.om.engage_kill_switch("kevin", "test")
        assert out["flatten"] and [p[:3] for p in _placed(e)] == [("US.AAA", "SELL", 100)]


def test_gateway_quotes_parse_the_snapshot_and_skip_unpriced_rows():
    from tests.unit.execution.moomoo_fake import FakeTradeContext, make_gateway
    rows = pd.DataFrame([
        {"code": "US.AAA", "update_time": "2025-03-03 16:00:00", "last_price": 50.5, "bid_price": 1, "ask_price": 1},
        {"code": "US.BBB", "update_time": "2025-03-03 16:00:00", "last_price": 0.0, "bid_price": 19.9, "ask_price": 20.1},
        {"code": "US.CCC", "update_time": "N/A", "last_price": "N/A", "bid_price": "N/A", "ask_price": "N/A"},
    ])
    gw = make_gateway(FakeTradeContext(), quote_fn=lambda tks: rows)
    q = gw.quotes(["AAA", "BBB", "CCC"])
    assert q["AAA"] == Quote(50.5, pd.Timestamp("2025-03-03 16:00:00").to_pydatetime())
    assert q["BBB"].price == pytest.approx(20.0)
    assert "CCC" not in q
    assert gw.quotes([]) == {}


def test_a_gateway_without_a_quote_source_refuses():
    from app.core.execution.broker_gateway import BrokerGateway

    class Bare(BrokerGateway):
        account = positions = place = orders = cancel = describe = lambda *a, **k: None
    with pytest.raises(BrokerError, match="不提供独立报价"):
        Bare().quotes(["AAA"])


class TestQuoteContract:

    def test_a_quote_without_a_time_does_not_open_a_position(self):
        from app.core.execution.broker_gateway import Quote
        dec = _check(_buy(), {"AAA": Quote(100., None)})
        assert [r for _, r in dec.rejected] == ["quote_without_time"]

    def test_adding_to_a_held_name_still_needs_a_quote(self):
        dec = _check(_buy(qty_held=1.0), {}, held=1.0)
        assert [r for _, r in dec.rejected] == ["no_independent_quote"], "已有持仓价不能替代独立报价去加仓"

    def test_a_quote_from_the_future_is_not_trusted(self):
        from app.core.execution.broker_gateway import Quote
        dec = _check(_buy(), {"AAA": Quote(100., datetime(2026, 1, 6, 9, 0))})   # 纽约次日上午
        assert [r for _, r in dec.rejected] == ["quote_in_future"]

    def test_a_quote_from_before_the_last_closed_session_is_stale(self):
        from app.core.execution.broker_gateway import Quote
        dec = _check(_buy(), {"AAA": Quote(100., datetime(2026, 1, 2, 16, 0))})   # 上周五，今天周一已收盘
        assert [r for _, r in dec.rejected] == ["stale_quote"]

    def test_a_good_quote_from_the_closed_session_opens_the_position(self):
        from app.core.execution.broker_gateway import Quote
        dec = _check(_buy(), {"AAA": Quote(100.2, datetime(2026, 1, 5, 16, 0))})
        assert len(dec.approved) == 1 and dec.rejected == []

    def test_a_reduction_needs_no_quote(self):
        from app.core.execution.order_builder import build_rebalance_orders
        orders = build_rebalance_orders(pd.Series({"AAA": .0}), {"AAA": 10.}, pd.Series({"AAA": 100.}), 10000,
                                        pd.Timestamp("2026-01-05").date(), book_id=-1, band_bps=50).orders
        assert [o.side for o in orders] == ["SELL"]
        dec = _check(orders, {}, held=10.0)
        assert len(dec.approved) == 1


# ---------------------------------------------------------------------------
# N06-d：按常规盘 / 盘前 / 盘后定义报价合同（复审："避免简单 4 个自然日被当成任意时段的可交易报价标准"）
# 2026-01-05（周一）收盘后决策；参考价 = 当日收盘。
# ---------------------------------------------------------------------------

class TestQuoteSessions:

    @staticmethod
    def _reason(quote_ny, now_utc=datetime(2026, 1, 5, 22, tzinfo=timezone.utc)):
        from app.core.execution.broker_gateway import Quote
        dec = _check(_buy(), {"AAA": Quote(100., quote_ny)}, now=now_utc)
        return [r for _, r in dec.rejected]

    def test_a_regular_session_quote_hours_before_the_close_is_stale(self):
        """11:00 之后就没更新过 → 这只票下午大概停牌 / 没成交，不是收盘时的市场价。"""
        assert self._reason(datetime(2026, 1, 5, 11, 0)) == ["quote_stale_in_session"]

    def test_a_quote_near_the_close_is_accepted(self):
        assert self._reason(datetime(2026, 1, 5, 15, 45)) == []

    def test_an_after_hours_quote_of_the_closed_session_is_accepted(self):
        assert self._reason(datetime(2026, 1, 5, 16, 45)) == []           # 决策时刻 = 纽约 17:00

    def test_a_pre_market_quote_of_the_next_session_is_accepted(self):
        now = datetime(2026, 1, 6, 13, 30, tzinfo=timezone.utc)          # 纽约 08:30，次日盘前
        assert self._reason(datetime(2026, 1, 6, 8, 0), now_utc=now) == []

    def test_a_pre_market_quote_of_the_closed_day_is_stale(self):
        assert self._reason(datetime(2026, 1, 5, 8, 0)) == ["stale_quote"]

    def test_a_quote_stamped_on_a_non_trading_day_is_rejected(self):
        now = datetime(2026, 1, 5, 22, tzinfo=timezone.utc)
        assert self._reason(datetime(2026, 1, 3, 12, 0), now_utc=now) == ["quote_outside_session"]   # 周六
