"""
满仓买入力：首轮保留"剩余资金不足时整张拒绝"，但每一轮都要留下可审计的计划记录（2026-10-30 用户决定）。

用户原话："首轮保留当前'剩余资金不足时整张拒绝'的保守规则。请补充：每日计划订单、资金占用、被拒原因、剩余现金和后续周期
重新决策的记录……不要声称被拒订单次日必然成交。"

exec_events 的 `cycle_plan`：每个决策日一条（同日重跑再记一条），被拒的买单下一轮怎么重新决策写在 previous_rejections ——
记的是决策（又批准 / 又被拒 / 不再需要），不是成交。
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
    rep = _cycle(e, D1, w={"AAA": .6, "BBB": .5}, px=PX)                 # 60k + 50k > 100k：BBB 整张被拒
    e.ctx.now = "2025-03-04 09:30:01.000"
    e.ctx.open_session({"AAA": 50.0, "BBB": 20.0})
    e.ctx.mark(PX)
    e.ctx.now = "2025-03-04 16:30:00.000"
    return e, rep


def test_a_rejected_buy_is_recorded_with_capital_and_reason(tmp_path):
    e, rep = _world(tmp_path)
    (p,) = _plans(e)
    assert p["decision_date"] == str(D1)
    assert [(a["ticker"], a["side"]) for a in p["approved"]] == [("AAA", "BUY")]
    assert [(r["ticker"], r["reason"]) for r in p["rejected"]] == [("BBB", "insufficient_buying_power")]
    assert p["rejected"][0]["notional"] == pytest.approx(p["rejected"][0]["qty"] * p["rejected"][0]["limit"], abs=0.01)
    c = p["capital"]
    assert c["remaining_cash"] == pytest.approx(c["buying_power"] - c["planned_buy"], abs=0.01)
    assert 0 < c["remaining_cash"] < p["rejected"][0]["notional"]           # 剩的钱不够那一整张
    assert rep.capital == c and p["previous_rejections"] == []


@pytest.mark.parametrize("w_bbb,outcome", [(.35, "approved"), (.45, "rejected:insufficient_buying_power"),
                                           (0.0, "not_planned")])
def test_the_next_cycle_records_how_the_rejected_buy_was_decided_again(tmp_path, w_bbb, outcome):
    """下一轮按当时的目标与资金重新算：能买得下 → 批准；还买不下 → 又被拒；目标没了 → 不再需要。不保证成交。"""
    e, _ = _world(tmp_path)
    _cycle(e, D2, w={"AAA": .6, "BBB": w_bbb}, px=PX)
    last = _plans(e)[-1]
    assert last["decision_date"] == str(D2)
    (r,) = last["previous_rejections"]
    assert (r["ticker"], r["rejected_on"], r["this_cycle"]) == ("BBB", str(D1), outcome)
    assert (r["qty_now"] is None) == (outcome == "not_planned")


def test_a_rerun_of_the_same_day_compares_with_the_previous_day_not_itself(tmp_path):
    e, _ = _world(tmp_path)
    _cycle(e, D2, w={"AAA": .6, "BBB": .45}, px=PX)
    _cycle(e, D2, w={"AAA": .6, "BBB": .45}, px=PX)                         # 同日重跑（重启）
    a, b = _plans(e)[-2:]
    assert a["previous_rejections"][0]["rejected_on"] == b["previous_rejections"][0]["rejected_on"] == str(D1)
