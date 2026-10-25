"""
加风险订单报价合同的边界（D5 关键路径变异复核：app/core/execution/pretrade_gate.py::quote_problem）。

复核时存活 7 个，全是边界：报价为 0、恰好晚 5 分钟、恰好 4 天没更新（耶稣受难日长周末）、恰好偏离 15%、
恰好开盘时刻、恰好收盘前 30 分钟。逐个钉住；剩下 1 个是等价变异（证明在 PROVEN_EQUIVALENT）。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.core.execution.broker_gateway import Quote
from app.core.execution.pretrade_gate import PreTradeLimits, quote_problem

LIM = PreTradeLimits(max_name_weight=0.10, max_gross=1.0, max_participation_pct=0.10,
                     max_daily_loss=0.05, max_price_deviation=0.15)
#: 2025-03-05（周三，冬令时）收盘 16:00 ET = 21:00 UTC；决策在收盘后一小时
NOW = datetime(2025, 3, 5, 22, 0, tzinfo=timezone.utc)


def _q(asof, price=100.0):
    return Quote(price=price, asof=datetime.fromisoformat(asof))


def test_a_quote_at_the_close_is_good():
    assert quote_problem(_q("2025-03-05 16:00:00"), 100.0, NOW, LIM) == ""


def test_a_zero_price_is_no_quote():
    assert quote_problem(_q("2025-03-05 16:00:00", price=0.0), 100.0, NOW, LIM) == "no_independent_quote"


def test_exactly_five_minutes_ahead_is_tolerated():
    assert quote_problem(_q("2025-03-05 17:05:00"), 100.0, NOW, LIM) == ""
    assert quote_problem(_q("2025-03-05 17:05:01"), 100.0, NOW, LIM) == "quote_in_future"


def test_exactly_the_maximum_age_is_still_fresh():
    """耶稣受难日长周末：周四收盘前的报价，到周一收盘前恰好 4 天 —— 最近一个已收盘交易日仍是周四。"""
    now = datetime(2025, 4, 21, 19, 50, tzinfo=timezone.utc)          # 周一 15:50 ET（夏令时）
    assert quote_problem(_q("2025-04-17 15:50:00"), 100.0, now, LIM) == ""
    assert quote_problem(_q("2025-04-17 15:49:59"), 100.0, now, LIM) == "stale_quote"


def test_exactly_the_maximum_deviation_is_accepted():
    assert quote_problem(_q("2025-03-05 16:00:00"), 115.0, NOW, LIM) == ""
    assert quote_problem(_q("2025-03-05 16:00:00"), 115.01, NOW, LIM) == "quote_mismatch"


def test_a_quote_exactly_at_the_open_is_an_early_in_session_quote_not_a_pre_market_one():
    assert quote_problem(_q("2025-03-05 09:30:00"), 100.0, NOW, LIM) == "quote_stale_in_session"
    assert quote_problem(_q("2025-03-05 09:29:59"), 100.0, NOW, LIM) == "stale_quote"


def test_exactly_thirty_minutes_before_the_close_is_late_enough():
    assert quote_problem(_q("2025-03-05 15:30:00"), 100.0, NOW, LIM) == ""
    assert quote_problem(_q("2025-03-05 15:29:59"), 100.0, NOW, LIM) == "quote_stale_in_session"


@pytest.mark.parametrize("asof", ["2025-03-05 16:00:00", "2025-03-05 15:59:59", "2025-03-05 16:00:01"])
def test_at_and_around_the_close_the_two_comparisons_agree(asof):
    """L218 `t <= c_` 等价的机械验证：恰好在收盘时刻，c_ − t = 0 不大于容差，"<=" 与 "<" 都不拦。"""
    assert quote_problem(_q(asof), 100.0, NOW, LIM) == ""


PROVEN_EQUIVALENT = {
    "app/core/execution/pretrade_gate.py ×1 — L218 `t <= c_` -> `t < c_`":
        "两者只在报价时刻恰好等于收盘时刻时不同；那时同一行的另一半 `c_ - t > REGULAR_CLOSE_TOL` 是 0 > 30 分钟，"
        "整个条件无论哪种写法都为假，结论都是不拦。见 test_at_and_around_the_close_the_two_comparisons_agree。",
}


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 1
