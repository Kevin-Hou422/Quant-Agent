"""
正式前向每日记录：必填字段、次日结算前一日订单、迟到重跑不能改写正式日、回放永不正式；决策不可变、修订链追加、启动按完整时间戳比较。

来源（原测试文件的说明，保留出处）：

复审 2026-10-19 §7.3：正式前向的启动记录与每日记录。

  "记录唯一启动时间、规格哈希、代码版本、数据身份和环境版本。联调和历史回放不得补记成正式天数。"
  "每天至少记录：信号/目标生成与存档时间、预期与实际 session、目标权重/股数、实际成交、未成交原因、
   持仓现金、成本、净收益、基准超额、回撤、gross/net、资金利用率、对账状态及异常。"

从每日管线入口连跑两个交易日（假券商按真实契约成交），检查每条记录逐项都有、能算的都算出来了、
算不出的写明原因；没启动 → 不算正式；启动后在决策窗口里跑 → 正式；事后补跑不能把正式日覆盖成非正式。

---

2026-10-22 用户决定 · C4：正式日报的不可变与启动资格。

  "正式日报仍可被另一条正式日报覆盖：当前只防 formal→nonformal，不防 formal→formal。
   决策和信号应不可变；后续成交、费用、更正采用追加事件或完整修订链，不能无痕替换。
   启动资格必须比较完整 UTC 时间戳，不能只比较日期；启动前已经发生的收益不得计入正式成绩。"

多日端到端（假券商按真实契约成交），策略置于 paper_running（C6）。
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from app.core.data_engine.market_calendar import session_close_utc, session_open_utc
from tests.integration.test_forward_daily_end_to_end import D1, D2, D3, world  # noqa: F401


def _active():
    import json

    from app.db.strategy_store import StrategyStore
    a = StrategyStore().latest_active()
    return a.id, json.loads(a.verdict)["spec_hash"]


def _day(sid, d):
    from app.db.forward_record import ForwardRecordStore
    rows = [r for r in ForwardRecordStore().days(sid) if r["decision_date"] == str(pd.Timestamp(d).date())]
    assert len(rows) == 1, f"{d} 没有（或不止一条）每日记录"
    return rows[0]


def _running():
    import json

    from app.db.strategy_store import StrategyStore
    a = StrategyStore().latest_running()
    return a.id, json.loads(a.verdict)["spec_hash"]


def _store():
    from app.db.forward_record import ForwardRecordStore
    return ForwardRecordStore()


def _next_day(world, d):
    world["open_session"](d)
    world["yahoo"].published_through = d
    world["clear"]()
    return world["run_day"](d)


def test_every_day_is_recorded_with_all_required_fields(world):
    from app.db.forward_record import REQUIRED_FIELDS
    world["run_day"](D1)
    sid, _ = _active()
    rec = _day(sid, D1)
    assert not [k for k in REQUIRED_FIELDS if k not in rec], "缺 §7.3 要求的字段"
    assert rec["formal"] is False and rec["formal_reason"] == "no_formal_start", "没启动却算成了正式前向"
    assert rec["expected_session"] == str(D2.date())
    assert rec["signal_archived_at"] and rec["targets_generated_at"]
    assert rec["target_weights"] and rec["target_shares"], "目标权重 / 股数没记"
    assert rec["benchmark"]["name"] == "ew_universe" and rec["benchmark"]["return"] is not None
    assert rec["costs"]["broker_fees"] is None and "costs.broker_fees" in rec["missing"], \
        "算不出的项要写明原因，不能当 0"


def test_the_second_day_settles_the_first_days_orders(world):
    from app.db.forward_record import ForwardRecordStore
    world["run_day"](D1)
    sid, h = _active()
    # 人启动正式前向（D13：重新提案并批准之后）；启动时刻在 D2 开盘之前
    run = ForwardRecordStore().start(sid, h, "kevin", "re-proposed and approved", benchmark_universe=list(world["yahoo"].close.columns),
                                     started_at=session_open_utc(D2) - timedelta(hours=1))
    assert run.code_version and run.environment and "python" in run.environment
    with pytest.raises(ValueError):
        ForwardRecordStore().start(sid, h, "kevin", "again", benchmark_universe=list(world["yahoo"].close.columns))       # 启动时间唯一

    world["open_session"](D2)
    world["yahoo"].published_through = D2
    world["clear"]()
    world["run_day"](D2)
    rec = _day(sid, D2)
    assert rec["formal"] is True, rec["formal_reason"]
    assert rec["fills"], "D1 的单在 D2 开盘成交了，记录里没有"
    assert {f["decision_date"] for f in rec["fills"]} == {str(D1.date())}
    assert rec["actual_sessions"] == [str(D2.date())], "实际成交 session 不对"
    assert rec["expected_session"] == "2025-03-05"
    assert rec["cash"] is not None and rec["positions"], "持仓现金没记"
    assert rec["costs"]["paper_cost_bps"] is not None and rec["costs"]["broker_slippage_bps"] is not None
    assert rec["net_return"]["paper"] is not None and rec["net_return"]["broker"] is not None
    # 2026-10-22 D16：相对满仓基准的差额（不叫 alpha）+ 按净敞口折算的差额；基准成分在启动时冻结
    ex = rec["excess_return"]
    assert None not in (ex["paper_vs_full_benchmark"], ex["broker_vs_full_benchmark"],
                        ex["paper_exposure_matched"], ex["broker_exposure_matched"]), ex
    assert rec["benchmark"]["universe_frozen"] is True and rec["benchmark"]["return"] is not None
    assert rec["drawdown"]["paper"] is not None and rec["drawdown"]["broker"] is not None
    e = rec["exposure"]
    assert None not in (e["target_gross"], e["target_net"], e["actual_gross"], e["actual_net"])
    assert rec["capital_utilization"] is not None
    assert rec["reconcile_status"] == "clean"
    assert isinstance(rec["anomalies"], list) and isinstance(rec["unfilled"], list)
    assert rec["data_identity"] and rec["code_version"] and rec["spec_hash"] == h


def test_a_late_rerun_cannot_turn_a_formal_day_into_a_replay_or_overwrite_it(world):
    from app.db.forward_record import ForwardRecordStore
    world["run_day"](D1)
    sid, h = _active()
    ForwardRecordStore().start(sid, h, "kevin", "go", benchmark_universe=list(world["yahoo"].close.columns), started_at=session_open_utc(D2) - timedelta(hours=1))
    world["open_session"](D2)
    world["yahoo"].published_through = D2
    world["clear"]()
    world["run_day"](D2)
    before = _day(sid, D2)
    assert before["formal"] is True
    # 第二天开盘之后再跑同一个决策日：不在决策窗口 → 不是正式；也不能覆盖已经是正式的那条
    world["clear"]()
    late = (session_open_utc(pd.Timestamp("2025-03-05")) - session_close_utc(D2)).total_seconds() / 60 + 30
    world["run_day"](D2, minutes_after_close=int(late))
    after = _day(sid, D2)
    # 2026-10-22 C4：决策不可变；补跑的内容追加进修订链（不是覆盖，也不是悄悄丢掉）
    assert after["decision"] == before["decision"] and after["formal"] is True, "事后补跑覆盖了正式那天的决策"
    assert after["revisions"] == before["revisions"] + 1
    from app.db.forward_record import ForwardRecordStore
    late_rev = ForwardRecordStore().history(sid, D2)[-1]
    assert late_rev["kind"] == "revision" and late_rev["payload"]["formal"] is False


def test_a_replayed_day_is_never_formal(world):
    """启动之后，用事后的时钟补跑一个决策日 → 记录写明 not_run_in_decision_window。"""
    from app.db.forward_record import ForwardRecordStore
    world["run_day"](D1)
    sid, h = _active()
    ForwardRecordStore().start(sid, h, "kevin", "go", benchmark_universe=list(world["yahoo"].close.columns), started_at=session_close_utc(D1) - timedelta(hours=1))
    world["clear"]()
    late = (session_open_utc(D2) - session_close_utc(D1)).total_seconds() / 60 + 30
    world["run_day"](D1, minutes_after_close=int(late))
    rec = _day(sid, D1)
    assert rec["formal"] is False                                   # 当天的决策（首跑）不是正式
    from app.db.forward_record import ForwardRecordStore
    late = ForwardRecordStore().history(sid, D1)[-1]["payload"]     # 补跑那次进修订链，也不是正式
    assert late["formal"] is False and late["formal_reason"] in ("not_run_in_decision_window", "execution_not_live")


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
