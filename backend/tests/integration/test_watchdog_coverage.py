"""
盘中巡检：逐只持仓的行情覆盖（复审固定提交 21cd90b，"放行边界 2"）。

复审原话："当前实现看起来只收集返回的有效 asof，再取其中最新时间。如果一只持仓缺少行情时间，另一只新鲜，整体可能仍被判为
fresh；未来时间也需要明确处理。"要求："逐一列出每只持仓的行情覆盖与状态：新鲜、缺失、过期、未来时间、时区不可判定或其他
异常；不把'至少一只新鲜'等同于'全部持仓新鲜'；明确缺失和过期时对盘中亏损判断、告警及无人值守许可的影响；纯现金账户和
价格不变但时间正常更新的账户不能误报陈旧；补多持仓部分缺失、部分过期、未来时间、停牌、时区异常、纯现金、断连恢复及熔断
续平测试。"

修复前（HEAD 21cd90b，反例 C:\\QuantAgentEnv\\review-repro\\repro_at_21cd90b.json）：一只新鲜、一只没有行情时间 → fresh / ok、
没有告警；一只时间在明天、一只 3 小时没更新 → fresh / ok。
"""

from __future__ import annotations

import json
from datetime import datetime as _dt
from datetime import timedelta

import pytest

from app.core.execution import intraday_watchdog as iw
from tests.integration.test_intraday_watchdog import OPEN_D3, Clock, _drop, _sells, _watchdog, _world

NOW = OPEN_D3 + timedelta(minutes=5)                     # 2025-03-05 美东 09:35
FRESH = _dt(2025, 3, 5, 9, 34)                           # 美东本地时间（券商行情时间的口径）
HALTED = _dt(2025, 3, 4, 15, 0)


def _quotes(e, monkeypatch, asof: dict):
    """asof: {票: 时间 / None（没有时间戳）/ "drop"（这只没返回行情）}"""
    from app.core.execution.broker_gateway import Quote

    def quotes(tks):
        return {tk: Quote(price=1.0, asof=asof.get(tk, FRESH)) for tk in tks if asof.get(tk, FRESH) != "drop"}
    monkeypatch.setattr(e.gw, "quotes", quotes)


def _tick(e, monkeypatch, asof, clock=None, **kw):
    _quotes(e, monkeypatch, asof)
    wd, alerts = _watchdog(e, clock or Clock(NOW), stale_after_s=600, **kw)
    return wd, alerts, wd.tick()


def _recorded(e):
    return json.loads(e.es.get_state(iw.STATE_KEY))


class TestEachHoldingIsJudged:

    def test_one_missing_quote_time_among_fresh_ones_is_partial_not_fresh(self, tmp_path, monkeypatch):
        e = _world(tmp_path)
        _wd, alerts, out = _tick(e, monkeypatch, {"BBB": None})
        assert out["freshness"] == "partial" and out["status"] == "partial_ok"
        assert out["names"] == {"AAA": "fresh", "BBB": "missing", "CCC": "fresh"}
        assert [(a.severity, a.title) for a in alerts] == [("critical", "盘中巡检：BBB 的行情缺失")]
        assert _recorded(e)["names"]["BBB"] == "missing"

    def test_a_holding_without_any_quote_is_missing(self, tmp_path, monkeypatch):
        e = _world(tmp_path)
        _wd, _a, out = _tick(e, monkeypatch, {"CCC": "drop"})
        assert out["names"]["CCC"] == "missing" and out["freshness"] == "partial"

    def test_part_stale_is_partial_with_a_warning_per_name(self, tmp_path, monkeypatch):
        e = _world(tmp_path)
        _wd, alerts, out = _tick(e, monkeypatch, {"BBB": HALTED})
        assert out["names"] == {"AAA": "fresh", "BBB": "stale", "CCC": "fresh"}
        assert [(a.severity, a.title) for a in alerts] == [("warning", "盘中巡检：BBB 的行情时间偏旧")]

    def test_a_future_quote_time_is_not_fresh(self, tmp_path, monkeypatch):
        e = _world(tmp_path)
        _wd, alerts, out = _tick(e, monkeypatch, {"AAA": _dt(2025, 3, 6, 9, 34)})
        assert out["names"]["AAA"] == "future" and out["freshness"] == "partial"
        assert ("critical", "盘中巡检：AAA 的行情时间在未来") in [(a.severity, a.title) for a in alerts]

    def test_a_small_clock_skew_is_not_the_future(self, tmp_path, monkeypatch):
        e = _world(tmp_path)
        _wd, _a, out = _tick(e, monkeypatch, {"AAA": _dt(2025, 3, 5, 9, 36)})      # 比此刻晚 1 分钟
        assert out["names"]["AAA"] == "fresh"

    @pytest.mark.parametrize("raw", ["2025-11-02 01:30:00", "不是时间"])
    def test_an_undeterminable_time_or_zone_is_bad_time(self, tmp_path, monkeypatch, raw):
        """2025-11-02 01:30 美东出现两次（夏令时结束）：不带时区的这个时刻判定不了是哪一个。"""
        e = _world(tmp_path)
        _wd, alerts, out = _tick(e, monkeypatch, {"BBB": raw})
        assert out["names"]["BBB"] == "bad_time" and out["freshness"] == "partial"
        assert any(a.severity == "critical" and "BBB" in a.title for a in alerts)

    def test_when_no_holding_is_fresh_the_watchdog_is_blind_and_decides_nothing(self, tmp_path, monkeypatch):
        e = _world(tmp_path)
        _drop(e, 0.40)
        _wd, alerts, out = _tick(e, monkeypatch, {"AAA": _dt(2025, 3, 6, 9, 34), "BBB": HALTED, "CCC": HALTED})
        assert out["status"] == "stale_snapshot" and e.es.kill_switch().get("engaged") is False
        assert [a.severity for a in alerts] == ["critical"]

    def test_a_breach_with_partial_coverage_still_engages_the_breaker(self, tmp_path, monkeypatch):
        """部分持仓的估值不可信时仍然判断亏损：跌穿阈值就拉熔断（减风险方向）。"""
        e = _world(tmp_path)
        _drop(e, 0.40)
        _wd, _a, out = _tick(e, monkeypatch, {"BBB": None})
        assert out["status"] == "breaker_engaged" and out["freshness"] == "partial"
        assert e.es.kill_switch().get("engaged") is True and _sells(e)

    def test_under_the_breaker_partial_coverage_keeps_flattening(self, tmp_path, monkeypatch):
        e = _world(tmp_path)
        e.om.engage_kill_switch("tester", "演练")
        n = len(_sells(e))
        _wd, _a, out = _tick(e, monkeypatch, {"BBB": None, "CCC": HALTED})
        assert out["status"] == "breaker_active" and "maintain" in out
        assert len(_sells(e)) >= n                                                   # 续做，不重复下全平单


class TestNoFalseStaleness:

    def test_a_cash_only_account_is_never_stale_and_never_blocks(self, tmp_path):
        from tests.unit.execution.test_order_manager import _env
        e = _env(tmp_path)
        wd, alerts = _watchdog(e, Clock(NOW), stale_after_s=600)
        out = wd.tick()
        assert out["freshness"] == "no_positions" and alerts == []
        assert iw.blocking_reasons(e.es) == []

    def test_unchanged_prices_with_moving_quote_times_stay_fresh(self, tmp_path, monkeypatch):
        e = _world(tmp_path)
        clock = Clock(NOW)
        wd, alerts = _watchdog(e, clock, stale_after_s=600)
        from app.core.execution.broker_gateway import Quote
        for k in range(4):                                                          # 20 分钟里价格一动不动
            t = FRESH + timedelta(minutes=5 * k)
            monkeypatch.setattr(e.gw, "quotes", lambda tks, t=t: {tk: Quote(price=1.0, asof=t) for tk in tks})
            out = wd.tick()
            assert out["freshness"] == "fresh" and out["status"] == "ok"
            clock.t += timedelta(minutes=5)
        assert alerts == [] and iw.blocking_reasons(e.es) == []


class TestEffectOnUnattended:

    @pytest.mark.parametrize("asof,blocks", [
        ({"BBB": None}, True), ({"AAA": _dt(2025, 3, 6, 9, 34)}, True), ({"BBB": "不是时间"}, True),
        ({"BBB": HALTED}, False),                                                     # 只是停牌：列出、不阻断
        ({"AAA": HALTED, "BBB": HALTED, "CCC": HALTED}, True), ({}, False)])
    def test_the_last_tick_decides_whether_new_risk_may_be_added(self, tmp_path, monkeypatch, asof, blocks):
        e = _world(tmp_path)
        _tick(e, monkeypatch, asof)
        assert bool(iw.blocking_reasons(e.es)) is blocks, iw.blocking_reasons(e.es)

    def test_a_disconnect_blocks_and_the_recovery_tick_releases(self, tmp_path, monkeypatch):
        from app.core.execution.broker_gateway import BrokerError
        e = _world(tmp_path)
        _quotes(e, monkeypatch, {})
        state = {"down": True}

        def factory():
            if state["down"]:
                raise BrokerError("OpenD 断线")
            return e.om
        clock = Clock(NOW)
        wd, alerts = _watchdog(e, clock, factory=factory, unreachable_alert_after=2, stale_after_s=600)
        for _ in range(2):
            assert wd.tick()["status"] == "broker_unreachable"
        assert _recorded(e)["state"] == "unreachable" and iw.blocking_reasons(e.es)
        assert [a.severity for a in alerts] == ["critical"]
        state["down"] = False
        clock.t += timedelta(minutes=1)
        out = wd.tick()
        assert out["status"] == "ok" and iw.blocking_reasons(e.es) == []
        assert alerts[-1].severity == "info" and "恢复" in alerts[-1].title

    def test_a_restarted_process_sees_the_last_tick(self, tmp_path, monkeypatch):
        """巡检停了而最后一拍是坏的：新进程（新的执行库对象）读到的仍是那一拍 → 一直不新增风险。"""
        from app.db.execution_store import ExecutionStore
        e = _world(tmp_path)
        _tick(e, monkeypatch, {"BBB": None})
        fresh_store = ExecutionStore(db_url=str(e.es._engine.url))                    # noqa: SLF001
        assert iw.blocking_reasons(fresh_store)

    def test_a_corrupt_record_blocks(self, tmp_path):
        from tests.unit.execution.test_order_manager import _env
        e = _env(tmp_path)
        e.es.set_state(iw.STATE_KEY, "{坏的")
        assert iw.blocking_reasons(e.es)

    def test_the_unattended_permit_reads_it_every_cycle(self, tmp_path, monkeypatch):
        from app.core.execution import run_permit as rp
        e = _world(tmp_path)
        _tick(e, monkeypatch, {"BBB": None})
        assert any("盘中巡检" in r for r in rp._store_health(e.es))                    # noqa: SLF001
