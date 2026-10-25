"""
专用 SIMULATE 账户验收的动作脚本与判定（2026-10-22 用户决定 D10：先准备脚本和证据模板）。

用户原话："同意人工监督、专用 SIMULATE 账户验收。先准备脚本和证据模板，连接账户、订单和时间由我安排。
覆盖提交成交、DAY 跨 session、部分成交撤单、未知订单、重启去重、熔断续平、输入异常和恢复。"

这里在替身券商（按真实 SDK 契约的 FakeTradeContext）上把八个场景各走一遍，验的是**脚本本身**：
    · 安全约束：非 SIMULATE / 账户不是写明的那个 / 不带 --confirm / 超单笔上限 → 不下单
    · 每一步都落证据；verify 只读证据判定 —— 走完的场景判过，缺步骤或结果不对的判不过
真券商上的行为要在真 OpenD 上由人验收（这些用例不能代替）。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from tests.unit.execution.test_order_manager import D1, D2, D3, _env
from tools.ops import simulate_acceptance as sa

ACC = 2001                                   # 替身里唯一的 SIMULATE 账户


class Crash(BaseException):
    """替代 os._exit：让用例能接住"进程在下单途中退出"。"""


def _session(e, tmp_path, **kw):
    kw.setdefault("crash", lambda: (_ for _ in ()).throw(Crash()))
    return sa.AcceptanceSession(e.gw, e.es, account=ACC, evidence_file=tmp_path / sa.EVIDENCE_NAME, actor="tester",
                                now=e.clock, om_factory=lambda gw, st, now: e.om, **kw)


def _quote(e, d, px):
    e.ctx.quotes.update(px)
    e.ctx.quote_time = f"{d:%Y-%m-%d} 16:00:00.000"
    e.ctx.mark(px)


def _after_close(e, d, px={"AAA": 50.0}):
    e.clock.after_close(d)
    e.ctx.now = f"{d:%Y-%m-%d} 17:00:00.000"
    _quote(e, d, px)


def _next_open(e, d, opens, ratio=1.0):
    from datetime import timedelta

    from app.core.data_engine.market_calendar import session_open_utc
    e.clock.t = session_open_utc(d) + timedelta(minutes=15)
    e.ctx.now = f"{d:%Y-%m-%d} 09:45:00.000"
    e.ctx.open_session(opens, ratio=ratio)


def _verdict(s, scenario):
    out = s.verify(scenario)["data"]
    return out["passed"], {c["name"]: c["ok"] for c in out["checks"]}


class TestSafety:

    def test_the_account_must_be_the_one_written_down(self, tmp_path):
        e = _env(tmp_path)
        with pytest.raises(sa.AcceptanceRefused, match="专用账户"):
            sa.AcceptanceSession(e.gw, e.es, account=9999, evidence_file=tmp_path / "x.jsonl", now=e.clock)

    def test_only_a_simulate_gateway_is_accepted(self, tmp_path):
        e = _env(tmp_path)
        e.gw.describe = lambda: {"trd_env": "REAL", "acc_id": ACC}
        with pytest.raises(sa.AcceptanceRefused, match="SIMULATE"):
            sa.AcceptanceSession(e.gw, e.es, account=ACC, evidence_file=tmp_path / "x.jsonl", now=e.clock)

    def test_without_confirm_nothing_is_sent_or_recorded(self, tmp_path):
        e = _env(tmp_path)
        s = _session(e, tmp_path)
        _after_close(e, D1)
        assert s.cycle("submit_and_fill", "AAA", 0.01) is None
        assert s.external_order("unknown_order", "AAA", "BUY", 1, 50.0) is None
        assert s.kill_switch("kill_switch_flatten", True, "测试") is None
        assert e.ctx.calls_of("place_order") == [] and not (tmp_path / sa.EVIDENCE_NAME).exists()
        assert e.es.kill_switch().get("engaged") is False

    def test_order_size_limits(self, tmp_path):
        e = _env(tmp_path)
        s = _session(e, tmp_path)
        _after_close(e, D1)
        with pytest.raises(sa.AcceptanceRefused, match="单笔上限"):
            s.cycle("submit_and_fill", "AAA", 0.05, confirm=True)          # 5% × 10 万 = 5000 > 2000
        with pytest.raises(sa.AcceptanceRefused, match="1–5"):
            s.external_order("unknown_order", "AAA", "BUY", 6, 1.0, confirm=True)
        with pytest.raises(sa.AcceptanceRefused, match="单笔上限"):
            s.external_order("unknown_order", "AAA", "BUY", 5, 500.0, confirm=True)
        assert e.ctx.calls_of("place_order") == []

    def test_every_record_names_the_account_and_code_version(self, tmp_path):
        e = _env(tmp_path)
        s = _session(e, tmp_path)
        s.snapshot("submit_and_fill", "开始前")
        row = json.loads((tmp_path / sa.EVIDENCE_NAME).read_text(encoding="utf-8").splitlines()[0])
        assert row["account"]["acc_id"] == ACC and row["account"]["trd_env"] == "SIMULATE"
        assert row["code_version"] and row["actor"] == "tester" and row["data"]["note"] == "开始前"

    def test_the_plan_and_the_checks_cover_the_same_scenarios(self):
        assert set(sa.SCENARIOS) == set(sa.CHECKS) == {
            "submit_and_fill", "day_order_across_sessions", "partial_fill_then_cancel", "unknown_order",
            "restart_dedup", "kill_switch_flatten", "input_anomaly", "crash_recovery"}


class TestScenarios:

    def test_submit_and_fill(self, tmp_path):
        e = _env(tmp_path)
        s = _session(e, tmp_path)
        _after_close(e, D1)
        assert _verdict(s, "submit_and_fill")[0] is False                     # 还什么都没做
        s.cycle("submit_and_fill", "AAA", 0.01, confirm=True)
        _next_open(e, D2, {"AAA": 50.1})
        _after_close(e, D2, {"AAA": 50.3})
        s.snapshot("submit_and_fill")
        s.reconcile("submit_and_fill")
        ok, checks = _verdict(s, "submit_and_fill")
        assert ok, checks

    def test_day_order_across_sessions(self, tmp_path):
        e = _env(tmp_path)
        s = _session(e, tmp_path)
        _after_close(e, D1)
        s.cycle("day_order_across_sessions", "AAA", 0.01, confirm=True)
        _next_open(e, D2, {"AAA": 60.0})                                     # 开在限价之上：不成交
        s.snapshot("day_order_across_sessions", "次日开盘后")
        assert _verdict(s, "day_order_across_sessions")[1] == {
            "下单时间在 T 收盘之后": True, "次一交易日的订单列表里有这张单": True,
            "次一交易日收盘后这张单已结束（成交或过期）": False}
        e.ctx.expire_day()
        _after_close(e, D2, {"AAA": 60.0})
        s.snapshot("day_order_across_sessions", "次日收盘后")
        assert _verdict(s, "day_order_across_sessions")[0] is True

    def test_partial_fill_then_cancel(self, tmp_path):
        e = _env(tmp_path)
        s = _session(e, tmp_path)
        _after_close(e, D1)
        rec = s.cycle("partial_fill_then_cancel", "AAA", 0.01, confirm=True)
        oid = rec["data"]["submitted"][0]["broker_order_id"]
        _next_open(e, D2, {"AAA": 50.0}, ratio=0.5)
        s.cancel("partial_fill_then_cancel", oid, confirm=True)
        _after_close(e, D2, {"AAA": 50.0})
        s.reconcile("partial_fill_then_cancel")
        ok, checks = _verdict(s, "partial_fill_then_cancel")
        assert ok, checks

    def test_unknown_order(self, tmp_path):
        e = _env(tmp_path)
        s = _session(e, tmp_path)
        _after_close(e, D1)
        s.reconcile("unknown_order")                                          # 先建立对账基准
        s.external_order("unknown_order", "AAA", "BUY", 3, 51.0, confirm=True)
        _next_open(e, D2, {"AAA": 50.0})
        _after_close(e, D2, {"AAA": 50.0})
        s.reconcile("unknown_order")
        s.cycle("unknown_order", "AAA", 0.01, confirm=True)
        ok, checks = _verdict(s, "unknown_order")
        assert ok, checks
        with pytest.raises(sa.AcceptanceRefused, match="reason"):
            s.accept("unknown_order", "  ", confirm=True)
        rec = s.accept("unknown_order", "核对过：3 股是验收时手工下的外部单", confirm=True)
        assert rec["data"]["report"]["status"] == "accepted"
        assert s.cycle("unknown_order", "AAA", 0.01, confirm=True)["data"]["report"]["blocked"] == ""

    def test_restart_dedup(self, tmp_path):
        e = _env(tmp_path)
        _after_close(e, D1)
        _session(e, tmp_path).cycle("restart_dedup", "AAA", 0.01, confirm=True)
        s2 = _session(e, tmp_path)                                            # "重启"：新会话、同一个执行库
        s2.cycle("restart_dedup", "AAA", 0.01, confirm=True)
        ok, checks = _verdict(s2, "restart_dedup")
        assert ok, checks

    def test_kill_switch_flatten(self, tmp_path):
        e = _env(tmp_path)
        s = _session(e, tmp_path)
        _after_close(e, D1)
        s.cycle("kill_switch_flatten", "AAA", 0.01, confirm=True)
        _next_open(e, D2, {"AAA": 50.0})
        _after_close(e, D2, {"AAA": 50.0})
        s.kill_switch("kill_switch_flatten", True, "验收：熔断续平", confirm=True)
        assert _verdict(s, "kill_switch_flatten")[1]["持仓最终清空"] is False
        _next_open(e, D3, {"AAA": 50.0})
        _after_close(e, D3, {"AAA": 50.0})
        s.maintain("kill_switch_flatten", confirm=True)
        ok, checks = _verdict(s, "kill_switch_flatten")
        assert ok, checks

    def test_input_anomaly(self, tmp_path):
        e = _env(tmp_path)
        s = _session(e, tmp_path)
        _after_close(e, D1)
        s.cycle("input_anomaly", "AAA", 0.01, confirm=True, ref_skew=0.3)
        ok, checks = _verdict(s, "input_anomaly")
        assert ok, checks
        assert e.ctx.calls_of("place_order") == []

    def test_crash_recovery(self, tmp_path):
        e = _env(tmp_path)
        _after_close(e, D1)
        with pytest.raises(Crash):
            _session(e, tmp_path).cycle("crash_recovery", "AAA", 0.01, confirm=True, crash_after_submit=True)
        s = _session(e, tmp_path)
        s.reconcile("crash_recovery")
        s.cycle("crash_recovery", "AAA", 0.01, confirm=True)
        ok, checks = _verdict(s, "crash_recovery")
        assert ok, checks
        assert len(e.ctx.calls_of("place_order")) == 1


def test_a_scenario_that_did_not_happen_does_not_pass(tmp_path):
    e = _env(tmp_path)
    s = _session(e, tmp_path)
    s.snapshot("crash_recovery")
    for name in sa.SCENARIOS:
        assert s.verify(name)["data"]["passed"] is False, name


def test_the_timestamp_of_records_is_aware_utc(tmp_path):
    e = _env(tmp_path)
    e.clock.t = datetime(2025, 3, 3, 22, 0, tzinfo=timezone.utc)
    row = _session(e, tmp_path).snapshot("submit_and_fill")
    assert row["at"].endswith("+00:00")


def test_the_evidence_template_walks_through_every_scenario_and_check():
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[2] / "docs" / "SIMULATE_ACCEPTANCE.md").read_text(encoding="utf-8")
    sections = {line.split()[1] for line in doc.splitlines() if line.startswith("### ")}
    assert sections == set(sa.SCENARIOS)
    for key, s in sa.SCENARIOS.items():
        body = doc.split(f"### {key} ", 1)[1].split("\n### ", 1)[0]
        missing = [p for p in s["pass"] if p not in body]
        assert not missing, (key, missing)
