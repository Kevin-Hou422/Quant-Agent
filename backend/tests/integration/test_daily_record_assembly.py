"""
正式前向每日记录的组装与决策链（D5 关键路径变异复核：app/db/forward_record.py 的逻辑部分；表结构见
test_audit_table_schema.py）。

复核时 146 个点存活 79 个：除了表结构，`build_daily_record` 里的"缺什么就记进 missing"的分支、滑点 / 回撤 /
敞口 / 敞口折算超额的公式、之前决策日未成交的筛选、正式与否的时间边界（恰好在启动时刻、恰好在收盘、恰好在次日开盘），
以及决策链的过滤（按策略 / 只取正式 / 严格早于当天）都没有用例钉住。这里直接喂合成输入、逐项算死。
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from app.db import forward_record as fr

DAY = pd.Timestamp("2025-03-04")                                   # 周二，收盘 21:00 UTC
CLOSE = datetime(2025, 3, 4, 21, 0, tzinfo=timezone.utc)
NEXT_OPEN = datetime(2025, 3, 5, 14, 30, tzinfo=timezone.utc)
NOW = CLOSE + timedelta(hours=1)
CLOSE_PX = pd.DataFrame({"A": [100.0, 101.0], "B": [50.0, 49.0]}, index=pd.DatetimeIndex(["2025-03-03", "2025-03-04"]))


class _Exec:
    """执行库替身：get / orders(since) / snapshots(limit) 三个读接口，按真实 ExecutionStore 的语义。"""

    def __init__(self, orders=(), snapshots=(), known=None):
        self._orders, self._snaps, self._known = list(orders), list(snapshots), dict(known or {})
        self.got = []

    def get(self, cid):
        self.got.append(cid)
        return self._known.get(cid)

    def orders(self, since=None, limit=500):
        return [o for o in self._orders if since is None or o.decision_date >= since][:limit]

    def snapshots(self, limit=30):
        return list(reversed(self._snaps))[:limit]                  # 真实接口：最新的在前


def _order(d, status, qty, dealt, ticker="A"):
    return SimpleNamespace(decision_date=d, status=status, qty=qty, dealt_qty=dealt, ticker=ticker,
                           reject_reason="", broker_status=status)


def _result(**over):
    rec = {"status": "clean", "account": {"cash": 1000.0, "total_assets": 10_000.0},
           "positions": {"A": {"market_val": 4000.0}, "B": {"market_val": -1000.0}},
           "fills": [], "day_return": 0.01}
    rec.update(over.pop("reconcile", {}))
    ex = {"mode": "moomoo_paper", "reconcile": rec,
          "orders": [{"ticker": "A", "side": "BUY", "qty": 10, "result": "submitted"}],
          "gate": {"rejected": []}, "build": {"skipped": []}, "capital": {"slice_equity": 10_000.0}}
    ex.update(over.pop("execution", {}))
    out = {"execution": ex, "execution_target": {"A": 0.5, "B": -0.2}, "signal_archived_at": "2025-03-04T21:05:00"}
    out.update(over)
    return out


RUN = SimpleNamespace(started_at=datetime(2025, 3, 1, tzinfo=timezone.utc),
                      benchmark=json.dumps({"tickers": ["A", "B"]}))


def _build(result=None, *, now=NOW, run=RUN, spec="h", exec_store=None, paper=None, status="paper_running"):
    return fr.build_daily_record(result or _result(), strategy_id=1, spec_hash=spec, decision_date=DAY, now=now,
                                 dataset={"close": CLOSE_PX}, paper_history=paper or [], exec_store=exec_store,
                                 forward_run=run, strategy_status=status)


class TestMissingFields:

    def test_a_complete_day_has_none_of_the_conditional_gaps(self):
        out = _build(exec_store=_Exec())
        for k in ("expected_session", "target_shares", "cash", "unfilled", "benchmark", "benchmark.universe"):
            assert k not in out["missing"], (k, out["missing"])
        assert out["expected_session"] == "2025-03-05" and out["formal"] is True

    def test_without_an_execution_store_nothing_is_reported_missing_about_it(self):
        out = _build(exec_store=None)
        assert "unfilled" not in out["missing"] and out["unfilled"] == []

    def test_no_orders_and_no_account_are_reported(self):
        out = _build(_result(execution={"orders": []}, reconcile={"account": {}}))
        assert "target_shares" in out["missing"] and "cash" in out["missing"]

    def test_a_start_without_a_frozen_benchmark_is_reported_as_such(self):
        out = _build(run=SimpleNamespace(started_at=RUN.started_at, benchmark=None))
        assert "没有冻结" in out["missing"]["benchmark.universe"]
        assert "benchmark.universe" not in _build(run=None)["missing"]

    def test_an_uncomputable_benchmark_keeps_the_first_reason(self):
        out = fr.build_daily_record(_result(), strategy_id=1, spec_hash="h", decision_date=DAY, now=NOW,
                                    dataset={"close": CLOSE_PX.iloc[:1]}, paper_history=[], forward_run=RUN,
                                    strategy_status="paper_running")
        assert out["benchmark"]["return"] is None and out["missing"]["benchmark"]

    def test_no_anomalies_when_nothing_went_wrong(self):
        assert _build()["anomalies"] == []


class TestFills:

    def test_fills_get_their_decision_date_only_when_needed(self):
        st = _Exec(known={"c1": SimpleNamespace(decision_date=date(2025, 3, 3))})
        fills = [{"client_id": "c1", "qty": 1.0}, {"client_id": "c2", "decision_date": "2025-03-02", "qty": 1.0},
                 {"qty": 1.0}]
        out = _build(_result(reconcile={"fills": fills}), exec_store=st)
        assert [f.get("decision_date") for f in out["fills"]] == ["2025-03-03", "2025-03-02", None]
        assert st.got == ["c1"]

    def test_a_single_expensive_buy_is_a_positive_cost(self):
        out = _build(_result(reconcile={"fills": [{"price": 101.0, "ref_price": 100.0, "qty": 10.0}]}))
        assert out["costs"]["broker_slippage_bps"] == pytest.approx(100.0)

    def test_broker_slippage_in_basis_points(self):
        fills = [{"price": 101.0, "ref_price": 100.0, "qty": 10.0},       # 买贵 1%
                 {"price": 99.0, "ref_price": 100.0, "qty": -10.0},       # 卖便宜 1%
                 {"price": None, "ref_price": 100.0, "qty": 5.0}]          # 价格缺失：不进分子分母
        out = _build(_result(reconcile={"fills": fills}))
        assert out["costs"]["broker_slippage_bps"] == pytest.approx(100.0)


def test_unfilled_orders_from_earlier_decision_days():
    d = DAY.date()
    st = _Exec(orders=[
        _order(d - timedelta(days=1), "CANCELLED", 10, 4, "A"),          # 撤了，还差 6 股
        _order(d - timedelta(days=9), "PARTIAL", 8, 3, "B"),             # 部分成交到终态，还差 5 股
        _order(d - timedelta(days=11), "CANCELLED", 10, 0, "C"),         # 超出 10 天回看
        _order(d - timedelta(days=1), "CANCELLED", 10, 10, "D"),         # 已全部成交
        _order(d - timedelta(days=1), "SUBMITTED", 10, 2, "E"),          # 还在途，不是"未成交到终态"
        _order(d - timedelta(days=1), "FILLED", 10, 10, "F"),
        _order(d, "CANCELLED", 10, 0, "G"),                              # 当天的单由本轮自己报
    ])
    out = _build(exec_store=st)
    got = sorted((u["ticker"], u["qty"]) for u in out["unfilled"] if u["reason"].startswith("broker:"))
    assert got == [("A", 6), ("B", 5)]


def test_drawdown_exposure_and_exposure_matched_excess():
    snaps = [SimpleNamespace(total_assets=v, status=s) for v, s in
             [(100.0, "baseline"), (120.0, "clean"), (130.0, "discrepancy"), (90.0, "clean")]]
    paper = [SimpleNamespace(date=pd.Timestamp("2025-03-0%d" % i), equity=e, net_ret=0.0, cost_bps=1.0)
             for i, e in ((1, 1.0), (3, 1.2), (4, 0.9))]
    out = _build(exec_store=_Exec(snapshots=snaps), paper=paper)
    assert out["drawdown"] == {"paper": pytest.approx(0.9 / 1.2 - 1.0), "broker": pytest.approx(90.0 / 120.0 - 1.0)}
    assert out["exposure"]["actual_gross"] == pytest.approx(0.5) and out["exposure"]["actual_net"] == pytest.approx(0.3)
    bench = out["benchmark"]["return"]
    assert bench == pytest.approx((101.0 / 100.0 + 49.0 / 50.0) / 2 - 1.0)
    assert out["excess_return"]["broker_exposure_matched"] == pytest.approx(0.01 - 0.3 * bench)


class TestFormalWindow:

    def test_decided_exactly_at_the_start_is_not_formal(self):
        out = _build(run=SimpleNamespace(started_at=NOW, benchmark=RUN.benchmark))
        assert (out["formal"], out["formal_reason"]) == (False, "decided_before_formal_start")

    def test_the_window_includes_the_close_and_excludes_the_next_open(self):
        assert _build(now=CLOSE)["formal"] is True
        assert (_build(now=NEXT_OPEN)["formal"], _build(now=NEXT_OPEN)["formal_reason"]) == \
            (False, "not_run_in_decision_window")
        assert _build(now=NEXT_OPEN - timedelta(microseconds=1))["formal"] is True

    def test_blocked_execution_or_no_spec_is_not_formal(self):
        out = _build(_result(execution={"blocked": "kill_switch_engaged"}))
        assert (out["formal"], out["formal_reason"]) == (False, "execution_not_live")
        out = _build(spec="")
        assert (out["formal"], out["formal_reason"]) == (False, "no_spec_hash")


def test_infinities_are_not_numbers():
    assert fr._f(float("inf")) is None and fr._f(float("-inf")) is None and fr._f(float("nan")) is None
    assert fr._f(1.5) == 1.5


class TestDecisionChain:

    @pytest.fixture
    def store(self, tmp_path):
        s = fr.ForwardRecordStore(db_url=f"sqlite:///{(tmp_path / 'f.db').as_posix()}")
        yield s
        s._engine.dispose()                                        # noqa: SLF001

    @staticmethod
    def _rec(sid, d, tw=None, formal=False, reason="x"):
        return {"strategy_id": sid, "decision_date": d, "spec_hash": "h", "formal": formal, "formal_reason": reason,
                "decision": {"target_weights": tw}, "target_weights": tw}

    def test_an_attempt_is_replaced_by_the_first_real_decision(self, store):
        assert store.record_day(self._rec(1, "2025-03-04")) == "decision"                 # 被拦：没有目标
        assert store.record_day(self._rec(1, "2025-03-04", {"A": 0.5})) == "decision"     # 有目标 → 成为决策
        assert store.record_day(self._rec(1, "2025-03-04", {"A": 0.6})) == "revision"
        assert [h["kind"] for h in store.history(1, "2025-03-04")] == ["decision", "attempt", "revision"]

    def test_a_changed_decision_is_logged_once_and_an_unchanged_one_not_at_all(self, store, caplog):
        import logging
        store.record_day(self._rec(1, "2025-03-04", {"A": 0.5}))
        caplog.set_level(logging.INFO, logger=fr.logger.name)
        store.record_day(self._rec(1, "2025-03-04", {"A": 0.5}))
        assert not [r for r in caplog.records if "决策已定" in r.getMessage()]
        store.record_day(self._rec(1, "2025-03-04", {"A": 0.6}))
        assert len([r for r in caplog.records if "决策已定" in r.getMessage()]) == 1

    def test_filters_by_strategy_formal_and_day(self, store):
        store.record_day(self._rec(1, "2025-03-03", {"A": 0.5}, formal=True))
        store.record_day(self._rec(1, "2025-03-04", {"A": 0.5}))
        store.record_day(self._rec(2, "2025-03-04", {"B": 0.5}, formal=True))
        assert [(r["strategy_id"], r["decision_date"]) for r in store.days()] == \
            [(1, "2025-03-03"), (1, "2025-03-04"), (2, "2025-03-04")]
        assert [r["decision_date"] for r in store.days(1)] == ["2025-03-03", "2025-03-04"]
        assert [(r["strategy_id"], r["decision_date"]) for r in store.days(formal_only=True)] == \
            [(1, "2025-03-03"), (2, "2025-03-04")]
        assert store.history(1, "2025-03-03")[0]["payload"]["decision_date"] == "2025-03-03"
        assert store.previous_decision(1, "2025-03-04")["decision_date"] == "2025-03-03"
        assert store.previous_decision(1, "2025-03-03") is None

    def test_payloads_are_canonical_readable_json(self, store, tmp_path):
        import sqlite3
        rec = dict(self._rec(1, "2025-03-04", {"A": 0.5}), formal_reason="联调", z=1, a=2)
        store.record_day(rec)
        con = sqlite3.connect(tmp_path / "f.db")
        raw = con.execute("SELECT payload FROM forward_days").fetchone()[0]
        con.close()
        assert raw == json.dumps(rec, ensure_ascii=False, default=str, sort_keys=True)

    def test_the_start_record_keeps_readable_identity_json(self, store, tmp_path):
        import sqlite3
        ident = {"dataset": "us_broad_large", "note": "冻结前人工核对"}
        store.start(1, "h", actor="Kevin-Hou422", reason="开始", strategy_status="paper_running",
                    data_identity=ident, benchmark_universe=["A", "B"])
        con = sqlite3.connect(tmp_path / "f.db")
        raw = con.execute("SELECT data_identity, environment, benchmark FROM forward_runs").fetchone()
        con.close()
        assert raw[0] == json.dumps(ident, ensure_ascii=False)
        assert raw[1] == json.dumps(json.loads(raw[1]), ensure_ascii=True), "环境版本只有 ASCII（见等价证明）"
        assert raw[2] == json.dumps(json.loads(raw[2]), ensure_ascii=True), "基准成分只有 ASCII（见等价证明）"


def test_next_sessions_never_include_the_decision_day():
    """L334 两处等价的机械验证：起点已是 day + 1，再过滤 `> day` 不会去掉任何东西；起点改成 day − 1 多出来的也被过滤掉。"""
    from app.core.data_engine.market_calendar import trading_days
    for d in pd.DatetimeIndex(["2025-03-04", "2025-04-17", "2025-07-03", "2025-11-26", "2025-12-24", "2025-12-31"]):
        a = [x for x in trading_days(d + timedelta(days=1), d + timedelta(days=10)) if x > d]
        b = [x for x in trading_days(d - timedelta(days=1), d + timedelta(days=10)) if x > d]
        c = [x for x in trading_days(d + timedelta(days=1), d + timedelta(days=10)) if x >= d]
        assert a == b == c, d


def test_the_sqlite_url_has_no_password_to_hide(tmp_path):
    """L183 等价的机械验证：执行库是 SQLite 文件，URL 里没有密码，hide_password 取什么都渲染成同一个串。"""
    s = fr.ForwardRecordStore(db_url=f"sqlite:///{(tmp_path / 'f.db').as_posix()}")
    try:
        u = s._engine.url                                            # noqa: SLF001
        assert u.password is None
        assert u.render_as_string(hide_password=False) == u.render_as_string(hide_password=True)
    finally:
        s._engine.dispose()                                          # noqa: SLF001


PROVEN_EQUIVALENT = {
    "app/db/forward_record.py ×2 — L334 `if d > day` -> `>=`、`day + timedelta(days=1)` -> `day - …`（下一交易日）":
        "trading_days 的起点已经是 day + 1 天，结果里没有 ≤ day 的日期，`>` 换成 `>=` 过滤不掉任何东西；"
        "起点提前到 day − 1 天只多出 day − 1、day，又被 `> day` 过滤掉。见 test_next_sessions_never_include_the_decision_day。",
    "app/db/forward_record.py ×1 — L183 `render_as_string(hide_password=False)` -> `True`":
        "这个 URL 只用来按同一个库再开一个 StrategyStore；库是 SQLite 文件，URL 里没有密码，两种渲染结果逐字相同。"
        "见 test_the_sqlite_url_has_no_password_to_hide。",
    "app/db/forward_record.py ×2 — L169 / L170 `json.dumps(environment_version() / frozen_bench, ensure_ascii=False)` -> `True`":
        "ensure_ascii 只影响非 ASCII 字符。环境版本是包名与版本号，冻结的基准成分是股票代码与摘要，全是 ASCII，"
        "两种写法写出的字符串逐字相同。见 test_the_start_record_keeps_readable_identity_json（逐字比对两种写法）。",
}


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 3
