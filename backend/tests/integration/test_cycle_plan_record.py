"""
满仓买入力：每一轮都要留下可审计的计划记录（2026-10-30 用户决定），资金不足时买单按比例缩量（见 test_buying_power_scaling.py）。

用户原话："请补充：每日计划订单、资金占用、被拒原因、剩余现金和后续周期重新决策的记录……不要声称被拒订单次日必然成交。"

exec_events 的 `cycle_plan`：每个决策日一条（同日重跑再记一条）。资金不足缩了量的买单记原数量与实发数量；没买够的那部分、
以及一股都买不起被拒的买单，下一轮怎么重新决策写在 previous_rejections —— 记的是决策（又批准 / 又被拒 / 不再需要），不是成交。
"""

from __future__ import annotations

import json

import pytest

from app.core.execution.pretrade_gate import PreTradeLimits
from tests.unit.execution.test_order_manager import D1, D2, _cycle, _env

WIDE = PreTradeLimits(max_name_weight=1.0, max_gross=2.0, max_participation_pct=.10, max_daily_loss=.5,
                      max_price_deviation=.15)
PX = {"AAA": 50.0, "BBB": 20.0}


def _plans(e):
    return [json.loads(ev.payload) for ev in reversed(e.es.events_of("cycle_plan", 50))]


def _world(tmp_path):
    e = _env(tmp_path)
    e.om.limits = WIDE
    rep = _cycle(e, D1, w={"AAA": .6, "BBB": .5}, px=PX)                 # 要 60300 + 50250 > 100000：两张都缩量
    e.ctx.now = "2025-03-04 09:30:01.000"
    e.ctx.open_session({"AAA": 50.0, "BBB": 20.0})
    e.ctx.mark(PX)
    e.ctx.now = "2025-03-04 16:30:00.000"
    return e, rep


def test_a_shrunk_buy_is_recorded_with_capital_planned_and_sent_quantities(tmp_path):
    e, rep = _world(tmp_path)
    (p,) = _plans(e)
    assert p["decision_date"] == str(D1)
    assert [(a["ticker"], a["side"], a["qty"]) for a in p["approved"]] == [("AAA", "BUY", 1085), ("BBB", "BUY", 2262)]
    assert [(r["ticker"], r["planned_qty"], r["qty"], r["reason"]) for r in p["resized"]] == [
        ("AAA", 1200, 1085, "insufficient_buying_power"), ("BBB", 2500, 2262, "insufficient_buying_power")]
    assert p["rejected"] == []
    assert p["approved"][0]["notional"] == pytest.approx(1085 * 50.25, abs=0.01)
    c = p["capital"]
    assert c["remaining_cash"] == pytest.approx(c["buying_power"] - c["planned_buy"], abs=0.01)
    assert 0 <= c["remaining_cash"] < 20.10                                 # 剩的钱连最便宜的一股都买不起
    assert rep.capital == c and p["previous_rejections"] == []


@pytest.mark.parametrize("w_bbb,outcome", [(.5, "approved"), (0.0, "not_planned")])
def test_the_next_cycle_records_how_the_part_not_bought_was_decided_again(tmp_path, w_bbb, outcome):
    """下一轮按当时的目标与资金重新算：还要买 → 批准（可能又是缩量）；目标没了 → 不再需要。不保证成交。"""
    e, _ = _world(tmp_path)
    _cycle(e, D2, w={"AAA": .6, "BBB": w_bbb}, px=PX)
    last = _plans(e)[-1]
    assert last["decision_date"] == str(D2)
    prev = {r["ticker"]: r for r in last["previous_rejections"]}
    assert (prev["BBB"]["rejected_qty"], prev["BBB"]["rejected_on"], prev["BBB"]["this_cycle"]) == (
        2500 - 2262, str(D1), outcome)
    assert (prev["BBB"]["qty_now"] is None) == (outcome == "not_planned")
    assert prev["AAA"]["this_cycle"] == "approved" and prev["AAA"]["qty_now"] > 0


def test_a_buy_the_cash_could_not_afford_at_all_is_recorded_as_rejected_and_followed_up(tmp_path):
    """D2 手里只剩约 500：再加一只 600 一股的票，缩到 0 股 → 记为被拒；D3 目标还在、钱还是不够 → 记"又被拒"。"""
    from tests.unit.execution.test_order_manager import D3
    e, _ = _world(tmp_path)
    px = {**PX, "CCC": 600.0}
    adv = {"AAA": 1e9, "BBB": 1e9, "CCC": 1e9}
    _cycle(e, D2, w={"AAA": .54, "BBB": .45, "CCC": .006}, px=px, adv=adv)
    last = _plans(e)[-1]
    assert [(r["ticker"], r["reason"]) for r in last["rejected"]] == [("CCC", "insufficient_buying_power")]
    e.ctx.now = "2025-03-05 16:30:00.000"
    _cycle(e, D3, w={"AAA": .54, "BBB": .45, "CCC": .006}, px=px, adv=adv)
    (r,) = [x for x in _plans(e)[-1]["previous_rejections"] if x["ticker"] == "CCC"]
    assert (r["rejected_on"], r["this_cycle"]) == (str(D2), "rejected:insufficient_buying_power")


def test_a_rerun_of_the_same_day_compares_with_the_previous_day_not_itself(tmp_path):
    e, _ = _world(tmp_path)
    _cycle(e, D2, w={"AAA": .6, "BBB": .5}, px=PX)
    _cycle(e, D2, w={"AAA": .6, "BBB": .5}, px=PX)                          # 同日重跑（重启）
    a, b = _plans(e)[-2:]
    assert a["previous_rejections"][0]["rejected_on"] == b["previous_rejections"][0]["rejected_on"] == str(D1)
