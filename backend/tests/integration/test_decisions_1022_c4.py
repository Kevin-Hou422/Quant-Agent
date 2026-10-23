"""
2026-10-22 用户决定 · C4：正式日报的不可变与启动资格。

  "正式日报仍可被另一条正式日报覆盖：当前只防 formal→nonformal，不防 formal→formal。
   决策和信号应不可变；后续成交、费用、更正采用追加事件或完整修订链，不能无痕替换。
   启动资格必须比较完整 UTC 时间戳，不能只比较日期；启动前已经发生的收益不得计入正式成绩。"

多日端到端（假券商按真实契约成交），策略置于 paper_running（C6）。
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from app.core.data_engine.market_calendar import session_close_utc, session_open_utc
from tests.integration.test_forward_daily_e2e import D1, D2, D3, world  # noqa: F401


def _running():
    import json
    from app.db.strategy_store import StrategyStore
    a = StrategyStore().latest_running()
    return a.id, json.loads(a.verdict)["spec_hash"]


def _store():
    from app.db.forward_record import ForwardRecordStore
    return ForwardRecordStore()


def _day(sid, d):
    rows = [r for r in _store().days(sid) if r["decision_date"] == str(pd.Timestamp(d).date())]
    assert len(rows) == 1
    return rows[0]


def _next_day(world, d):
    world["open_session"](d)
    world["yahoo"].published_through = d
    world["clear"]()
    return world["run_day"](d)


def test_a_second_formal_run_appends_instead_of_replacing(world):
    world["run_day"](D1)
    sid, h = _running()
    _store().start(sid, h, "kevin", "go", benchmark_universe=list(world["yahoo"].close.columns), started_at=session_close_utc(D1) + timedelta(minutes=30),
                   strategy_status="paper_running")
    _next_day(world, D2)
    first = _day(sid, D2)
    assert first["formal"] is True, first["formal_reason"]
    world["clear"]()
    world["run_day"](D2, minutes_after_close=150)                # 同一决策日、仍在决策窗口里重跑
    again = _day(sid, D2)
    assert again["decision"] == first["decision"], "正式日报的决策 / 信号被另一条正式日报替换了"
    chain = _store().history(sid, D2)
    assert len(chain) >= 2 and chain[0]["kind"] == "decision" and chain[-1]["kind"] == "revision"
    assert chain[-1]["payload"]["targets_generated_at"] != first["decision"]["targets_generated_at"]


def test_the_start_is_compared_as_a_full_utc_timestamp(world):
    """启动时刻在 D2 当天、但晚于 D2 的决策时刻 → D2 不是正式（此前只比日期，会算成正式）。"""
    world["run_day"](D1)
    sid, h = _running()
    world["open_session"](D2)
    world["yahoo"].published_through = D2
    world["clear"]()
    decision_at = session_close_utc(D2) + timedelta(minutes=90)
    _store().start(sid, h, "kevin", "go", benchmark_universe=list(world["yahoo"].close.columns), started_at=decision_at + timedelta(minutes=5),
                   strategy_status="paper_running")
    world["run_day"](D2)
    rec = _day(sid, D2)
    assert rec["formal"] is False and rec["formal_reason"] == "decided_before_formal_start"


def test_returns_earned_before_the_start_are_not_formal_performance(world):
    world["run_day"](D1)                                         # D1 的决策在启动之前
    sid, h = _running()
    _store().start(sid, h, "kevin", "go", benchmark_universe=list(world["yahoo"].close.columns), started_at=session_close_utc(D1) + timedelta(minutes=120),
                   strategy_status="paper_running")
    _next_day(world, D2)                                         # D2 的结果来自 D1 的决策（启动前）
    _next_day(world, D3)                                         # D3 的结果来自 D2 的决策（正式）
    perf = _store().formal_performance(sid)
    assert [p["outcome_date"] for p in perf] == [str(D3.date())], perf
    assert all(p["decision_date"] == str(D2.date()) for p in perf)
