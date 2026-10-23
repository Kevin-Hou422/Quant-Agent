"""
2026-10-22 用户决定 · 批 3：证据 / 研究纪律。

D12 原话："同意先只读查询，不授权写边界。使用明确的活库路径和真正只读连接，不调用会建表或迁移的 Store 构造函数。
列出已有边界、最早曝光日期、次数、股票池和实验身份，给出日期建议后由我确认。正式入口不要继续依赖首次调用偶然冻结；
改实验名不能刷新重叠数据的曝光额度。"
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REAL = "real_ds_d12"


def _panel(T=1300, N=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=T)
    cols = [f"S{i:02d}" for i in range(N)]
    q = np.linspace(-1, 1, N)
    c = pd.DataFrame(100 * np.cumprod(1 + 0.0008 * q[None, :] + rng.normal(0, 0.01, (T, N)), 0), idx, cols)
    o = c.shift(1).fillna(c)
    return {"open": o, "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * 0.998, "close": c,
            "vwap": c, "volume": pd.DataFrame(1e7, idx, cols), "returns": np.log(c / c.shift(1))}


# ---------------------------------------------------------------------------
# D12-a：正式入口不依赖首次调用偶然冻结
# ---------------------------------------------------------------------------

class TestNoAccidentalFreeze:

    def test_the_first_research_call_does_not_pin_a_boundary(self, fresh_db):
        from app.core.data_engine.research_context import open_research
        from app.db.trial_ledger import BoundaryStore
        ctx = open_research(_panel(), REAL, "test.first_call")
        assert BoundaryStore().get(REAL) is None, "第一次研究调用把边界写死了（偶然冻结）"
        assert ctx.boundary == "provisional" and ctx.split is not None
        assert ctx.describe()["boundary"] == "provisional"

    def test_a_formal_validation_verdict_needs_a_registered_boundary(self, fresh_db):
        from app.core.lifecycle.validation_gate import ValidationGate
        res = ValidationGate(n_splits=3, embargo_days=5, dsr_threshold=0.0, min_tstat=0.0).evaluate(
            "rank(ts_mean(returns,20))", _panel(), n_trials=1, dataset_key=REAL)
        assert res.passed is False
        assert any("边界未登记" in r for r in res.reasons), res.reasons
        assert res.research_boundary == "provisional"
        # 对照：登记之后这条理由消失（上面的不通过不是信号本身的原因）
        from app.db.trial_ledger import BoundaryStore
        BoundaryStore().register(REAL, pd.Timestamp(res.partition["test_start"]), actor="Kevin-Hou422", reason="D12")
        res2 = ValidationGate(n_splits=3, embargo_days=5, dsr_threshold=0.0, min_tstat=0.0).evaluate(
            "rank(ts_mean(returns,20))", _panel(), n_trials=1, dataset_key=REAL)
        assert res2.research_boundary == "registered"
        assert not any("边界未登记" in r for r in res2.reasons), res2.reasons

    def test_a_registered_boundary_is_used_and_is_write_once(self, fresh_db):
        from app.core.data_engine.research_context import open_research
        from app.db.trial_ledger import BoundaryStore, BoundaryAlreadyRegistered
        bs = BoundaryStore()
        bs.register(REAL, pd.Timestamp("2024-06-03"), actor="Kevin-Hou422", reason="D12 确认的日期")
        ctx = open_research(_panel(), REAL, "test.registered")
        assert ctx.boundary == "registered" and ctx.test_start == "2024-06-03"
        with pytest.raises(BoundaryAlreadyRegistered):
            bs.register(REAL, pd.Timestamp("2025-01-02"), actor="x", reason="改日期")
        assert bs.get(REAL) == pd.Timestamp("2024-06-03")

    @pytest.mark.parametrize("actor,reason", [("", "理由"), ("someone", ""), ("  ", "  ")])
    def test_registration_needs_an_actor_and_a_reason(self, fresh_db, actor, reason):
        from app.db.trial_ledger import BoundaryStore
        with pytest.raises(ValueError):
            BoundaryStore().register(REAL, pd.Timestamp("2024-06-03"), actor=actor, reason=reason)
        assert BoundaryStore().get(REAL) is None

    def test_a_legacy_auto_pinned_boundary_still_freezes_but_is_not_formal(self, fresh_db):
        """旧代码首次调用写下的边界：Test 段照样冻结（不能解冻），但不算登记过 —— 正式结论仍要人确认。"""
        from app.core.data_engine.research_context import open_research
        from app.core.lifecycle.validation_gate import ValidationGate
        from app.db.trial_ledger import BoundaryStore
        BoundaryStore().pin(REAL, pd.Timestamp("2024-06-03"), frozen_by="years")
        ctx = open_research(_panel(), REAL, "test.legacy")
        assert ctx.boundary == "unconfirmed" and ctx.test_start == "2024-06-03"
        res = ValidationGate(n_splits=3, embargo_days=5, dsr_threshold=0.0, min_tstat=0.0).evaluate(
            "rank(ts_mean(returns,20))", _panel(), n_trials=1, dataset_key=REAL)
        assert res.passed is False and any("边界未登记" in r for r in res.reasons)

    def test_a_proposal_on_an_unregistered_boundary_is_not_passed_and_not_grade_a(self, fresh_db):
        from app.core.data_engine.research_context import open_research
        from app.core.portfolio_manager.strategy_builder import build_strategy_config
        ds = _panel(T=700)
        with open_research(ds, REAL, "proposal").active() as ctx:
            sig = {"f": ctx.selection["close"].pct_change(20).rank(axis=1)}
            cfg = build_strategy_config(sig, ctx.selection, aum=10_000.0, factor_dsl={"f": "rank(close)"})
        g = cfg.verdict["gates"]["research_boundary"]
        assert g["evaluated"] is True and g["passed"] is False and g["boundary"] == "provisional"
        assert cfg.passed is False and cfg.verdict["paper_grade"] != "A"

    def test_a_proposal_outside_any_research_context_has_an_unevaluated_boundary_gate(self, fresh_db):
        from app.core.portfolio_manager.strategy_builder import build_strategy_config
        ds = _panel(T=700)
        cfg = build_strategy_config({"f": ds["close"].pct_change(20).rank(axis=1)}, ds, aum=10_000.0,
                                    factor_dsl={"f": "rank(close)"})
        assert cfg.verdict["gates"]["research_boundary"]["evaluated"] is False
        assert cfg.passed is False and cfg.verdict["paper_grade"] != "A"

    def test_a_proposal_on_a_registered_boundary_passes_the_boundary_gate(self, fresh_db):
        from app.core.data_engine.research_context import open_research
        from app.core.portfolio_manager.strategy_builder import build_strategy_config
        from app.db.trial_ledger import BoundaryStore
        ds = _panel(T=700)
        BoundaryStore().register(REAL, ds["close"].index[-200], actor="Kevin-Hou422", reason="D12")
        with open_research(ds, REAL, "proposal").active() as ctx:
            cfg = build_strategy_config({"f": ctx.selection["close"].pct_change(20).rank(axis=1)},
                                        ctx.selection, aum=10_000.0, factor_dsl={"f": "rank(close)"})
        g = cfg.verdict["gates"]["research_boundary"]
        assert g["evaluated"] is True and g["passed"] is True and g["boundary"] == "registered"


# ---------------------------------------------------------------------------
# D12-b：改实验名不能刷新重叠数据的曝光额度
# ---------------------------------------------------------------------------

def test_renaming_the_experiment_does_not_refresh_the_exposure_quota(fresh_db):
    from app.core.data_engine.data_partitioner import account_holdout_use
    from app.core.data_engine.research_context import open_research
    from app.db.trial_ledger import BoundaryStore, HoldoutLedger
    ds = _panel()
    BoundaryStore().register(REAL, ds["close"].index[-300], actor="Kevin-Hou422", reason="D12")
    ctx_a = open_research(ds, REAL, "validation_gate", experiment="exp-A")
    with ctx_a.active():
        first = account_holdout_use(ctx_a.split, REAL, purpose="validation_gate", budget=1)
    ctx_b = open_research(_panel(T=1310), REAL, "validation_gate", experiment="exp-B-renamed")
    with ctx_b.active():
        second = account_holdout_use(ctx_b.split, REAL, purpose="validation_gate", budget=1)
    assert first["uses"] == 1 and second["uses"] == 2 and second["over_budget"] is True
    rows = HoldoutLedger().usages(REAL)
    assert [r["experiment"] for r in rows] == ["exp-A", "exp-B-renamed"]
    assert all(r["universe_digest"] and r["universe_n"] == 12 for r in rows)


# ---------------------------------------------------------------------------
# D12-c：只读查询（明确路径、真正只读连接、不调用会建表 / 迁移的 Store 构造函数）
# ---------------------------------------------------------------------------

def _make_live_like_db(path: Path, *, with_boundaries: bool) -> None:
    """按真实活库的形状造一个 WAL 库：有 holdout_usages / trial_run_state / alpha_records，可选 frozen_boundaries。"""
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE holdout_usages (id INTEGER PRIMARY KEY, dataset_key TEXT, test_key TEXT, "
                "purpose TEXT, used_at TEXT)")
    con.execute("CREATE TABLE trial_run_state (run_id TEXT PRIMARY KEY, source TEXT, status TEXT, n_recorded INT, "
                "started_at TEXT, finished_at TEXT, note TEXT, context TEXT)")
    con.execute("CREATE TABLE alpha_records (id INTEGER PRIMARY KEY, dsl TEXT, created_at TEXT, status TEXT)")
    con.execute("CREATE TABLE trial_ledger (id INTEGER PRIMARY KEY, total INT, updated_at TEXT)")
    con.executemany("INSERT INTO holdout_usages (dataset_key, test_key, purpose, used_at) VALUES (?,?,?,?)", [
        ("us_tech_large", "2024-10-01..2026-09-30", "validation_gate", "2026-10-01 10:00:00"),
        ("us_tech_large", "2024-09-15..2026-10-09", "api.backtest", "2026-10-10 10:00:00"),
    ])
    con.execute("INSERT INTO trial_run_state VALUES ('r1','gp','done',120,'2026-10-11 09:00:00',"
                "'2026-10-11 09:10:00','', ?)",
                ('{"dataset_key": "us_broad_large", "experiment": "default", "universe_digest": "abc", '
                 '"universe_n": 98, "test_start": null, "selection_start": "2021-01-04", '
                 '"selection_end": "2026-10-09"}',))
    con.execute("INSERT INTO alpha_records (dsl, created_at, status) VALUES ('rank(close)', '2026-10-11 09:07:40', "
                "'validated')")
    con.execute("INSERT INTO trial_ledger VALUES (1, 258, '2026-10-11 09:15:27')")
    if with_boundaries:
        con.execute("CREATE TABLE frozen_boundaries (dataset_key TEXT PRIMARY KEY, test_start TEXT, frozen_by TEXT, "
                    "created_at TEXT)")
        con.execute("INSERT INTO frozen_boundaries VALUES ('us_tech_large', '2024-10-01', 'years', "
                    "'2026-10-01 09:00:00')")
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    for suf in ("-wal", "-shm"):
        p = Path(str(path) + suf)
        if p.exists():
            p.unlink()


def _snapshot(path: Path):
    return (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns,
            sorted(os.listdir(path.parent)))


class TestReadOnlyReport:

    def test_the_report_lists_boundaries_exposure_counts_universe_and_experiment(self, tmp_path):
        from tools.research.d12_boundary_report import build_report
        db = tmp_path / "live.db"
        _make_live_like_db(db, with_boundaries=True)
        before = _snapshot(db)
        rep = build_report(db)
        assert _snapshot(db) == before, "只读查询改动了活库文件或在旁边建了文件"
        tech = rep["datasets"]["us_tech_large"]
        assert tech["boundary"] == {"test_start": "2024-10-01", "frozen_by": "years", "registered": False}
        assert tech["earliest_exposed_start"] == "2024-09-15" and tech["exposures"] == 2
        assert set(tech["purposes"]) == {"validation_gate", "api.backtest"}
        broad = rep["datasets"]["us_broad_large"]
        assert broad["experiments"] == ["default"] and broad["universes"] == [{"digest": "abc", "n": 98}]
        assert broad["research_windows"] == [{"start": "2021-01-04", "end": "2026-10-09", "test_start": None}]
        assert rep["trial_total"] == 258

    def test_research_without_any_boundary_is_flagged_as_unrecorded_exposure(self, tmp_path):
        from tools.research.d12_boundary_report import build_report
        db = tmp_path / "live.db"
        _make_live_like_db(db, with_boundaries=False)
        rep = build_report(db)
        assert rep["tables_missing"] == ["frozen_boundaries"], "表不存在应如实报告，不能替它建表"
        broad = rep["datasets"]["us_broad_large"]
        assert broad["boundary"] is None
        assert broad["unrecorded_exposure"]["through"] == "2026-10-09"
        sugg = {s["option"]: s for s in broad["suggestions"]}
        assert sugg["clean"]["test_start"] > "2026-10-09", "干净方案的 Test 起点必须晚于所有被看过的日期"
        assert sugg["clean"]["needs_user_confirmation"] is True
        assert sqlite3.connect(db).execute(
            "select count(*) from sqlite_master where name='frozen_boundaries'").fetchone()[0] == 0

    def test_a_pending_wal_file_is_refused_rather_than_read_partially(self, tmp_path):
        from tools.research.d12_boundary_report import build_report, LiveDbBusy
        db = tmp_path / "live.db"
        _make_live_like_db(db, with_boundaries=True)
        Path(str(db) + "-wal").write_bytes(b"\0" * 32)
        with pytest.raises(LiveDbBusy):
            build_report(db)

    def test_the_report_module_never_touches_store_constructors(self):
        """D12："不调用会建表或迁移的 Store 构造函数" —— 报告模块只许用 sqlite3 的只读连接。"""
        import ast
        src = Path(__file__).resolve().parents[2] / "tools" / "research" / "d12_boundary_report.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any(m.startswith(("app.db", "sqlalchemy")) for m in mods), mods
        calls = {getattr(n.func, "attr", getattr(n.func, "id", "")) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assert "create_all" not in calls and "create_engine" not in calls


def test_an_old_schema_db_reports_inferred_exposure_from_research_activity(tmp_path):
    """
    活库是旧结构：没有 frozen_boundaries / trial_run_state，研究记录里没有数据集与窗口，只有研究发生的时刻。
    报告不能把"没记录"说成"没看过"：按用户给定的数据起点，推定研究那天之前的全部数据都已曝光，并标明是推定。
    """
    from tools.research.d12_boundary_report import build_report
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE alpha_records (id INTEGER PRIMARY KEY, dsl TEXT, created_at TEXT, status TEXT)")
    con.execute("CREATE TABLE trial_ledger (id INTEGER PRIMARY KEY, total INT, updated_at TEXT)")
    con.execute("CREATE TABLE holdout_usages (id INTEGER PRIMARY KEY, dataset_key TEXT, test_key TEXT, purpose TEXT, "
                "used_at TEXT)")
    con.execute("INSERT INTO alpha_records (dsl, created_at, status) VALUES ('x', '2026-10-11 09:07:40', 'validated')")
    con.execute("INSERT INTO trial_ledger VALUES (1, 258, '2026-10-11 09:15:27')")
    con.commit(); con.close()
    before = _snapshot(db)
    rep = build_report(db, assume_data_start={"us_broad_large": "2021-01-01"})
    assert _snapshot(db) == before
    assert set(rep["tables_missing"]) == {"frozen_boundaries", "trial_run_state"}
    assert rep["research_activity"] == {"first": "2026-10-11 09:07:40", "last": "2026-10-11 09:15:27",
                                        "alpha_records": 1, "trial_total": 258}
    broad = rep["datasets"]["us_broad_large"]
    ue = broad["unrecorded_exposure"]
    assert ue["inferred"] is True and ue["from"] == "2021-01-01" and ue["through"] == "2026-10-09"
    assert {s["option"] for s in broad["suggestions"]} >= {"clean", "conventional"}
    conv = {s["option"]: s for s in broad["suggestions"]}["conventional"]
    assert conv["already_exposed"] is True


def test_registration_is_a_dry_run_unless_confirmed(tmp_path):
    from tools.research.d12_register_boundary import main
    from app.db.trial_ledger import BoundaryStore
    db = tmp_path / "live.db"
    _make_live_like_db(db, with_boundaries=False)
    before = _snapshot(db)
    assert main(["--db", str(db), "--dataset", "us_broad_large", "--test-start", "2026-10-12",
                 "--actor", "Kevin-Hou422", "--reason", "D12 clean"]) == 0
    assert _snapshot(db) == before, "预演改动了库"
    assert main(["--db", str(db), "--dataset", "us_broad_large", "--test-start", "2026-10-12",
                 "--actor", "Kevin-Hou422", "--reason", "D12 clean", "--confirm"]) == 0
    st = BoundaryStore(db_url=f"sqlite:///{db.as_posix()}").status("us_broad_large")
    assert st == (pd.Timestamp("2026-10-12"), "registered")
    assert main(["--db", str(db), "--dataset", "us_broad_large", "--test-start", "2025-01-02",
                 "--actor", "x", "--reason", "改", "--confirm"]) == 2


# ---------------------------------------------------------------------------
# D16：首轮等权宇宙基准 —— 预先固定股票池时点、再平衡、收益窗口、缺价和公司行动处理
# 原话："不能事后删除缺价股票再平均。SPY 第二基准后排；低仓位策略对满仓基准的差额不要直接称为 alpha。"
# ---------------------------------------------------------------------------

_IDX = pd.bdate_range("2025-03-03", periods=2)
_D = "2025-03-04"


def _rec(close: pd.DataFrame, *, forward_run=None, result=None, now="2025-03-04 22:30"):
    from types import SimpleNamespace
    from app.db.forward_record import build_daily_record
    r = {"execution": {"mode": "moomoo_paper", "reconcile": {"status": "clean"}}}
    r.update(result or {})
    return build_daily_record(r, strategy_id=1, spec_hash="h", decision_date=_D,
                              now=pd.Timestamp(now, tz="UTC"), dataset={"close": close},
                              paper_history=[SimpleNamespace(date=_D, net_ret=0.01, cost_bps=1.0, equity=1.01)],
                              forward_run=forward_run, strategy_status="paper_running")


def _run(universe):
    from types import SimpleNamespace
    from app.core.benchmark import freeze_universe
    import json
    return SimpleNamespace(started_at=pd.Timestamp("2025-01-01"), benchmark=json.dumps(freeze_universe(universe)))


class TestEqualWeightUniverseBenchmark:

    def test_a_missing_price_stays_in_the_denominator(self):
        close = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 50.0], "C": [20.0, np.nan], "D": [10.0, 9.8]},
                             index=_IDX)
        b = _rec(close, forward_run=_run(["A", "B", "C", "D"]))["benchmark"]
        assert b["return"] == pytest.approx((0.10 + 0.0 + 0.0 - 0.02) / 4), "缺价的票被删掉后再平均了"
        assert b["missing"] == ["C"] and b["missing_share"] == pytest.approx(0.25)

    def test_the_universe_is_the_one_frozen_at_the_formal_start(self):
        """启动之后新出现的票（E）不进基准；启动时有、现在没有列的票（D）留在分母、记缺价。"""
        close = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 50.0], "E": [5.0, 10.0]}, index=_IDX)
        b = _rec(close, forward_run=_run(["A", "B", "D"]))["benchmark"]
        assert b["universe_frozen"] is True and b["n"] == 3
        assert b["return"] == pytest.approx(0.10 / 3)
        assert b["missing"] == ["D"]

    def test_before_a_formal_start_the_universe_is_marked_as_not_frozen(self):
        close = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 50.0]}, index=_IDX)
        b = _rec(close, forward_run=None)["benchmark"]
        assert b["universe_frozen"] is False and b["return"] == pytest.approx(0.05)

    def test_the_rules_are_fixed_in_advance_and_recorded(self):
        from app.core.benchmark import BENCHMARK_RULES
        for k in ("universe", "rebalance", "return_window", "price", "missing_price", "corporate_actions"):
            assert BENCHMARK_RULES.get(k), f"基准规则没有预先写明 {k}"
        close = pd.DataFrame({"A": [100.0, 110.0]}, index=_IDX)
        b = _rec(close, forward_run=_run(["A"]))["benchmark"]
        assert b["name"] == "ew_universe" and b["rules_version"] == BENCHMARK_RULES["version"]
        assert b["secondary"] is None and "SPY" in b["secondary_note"]

    def test_corporate_action_quarantined_members_are_held_at_zero_not_dropped(self):
        close = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 25.0]}, index=_IDX)     # B 疑似未知拆股
        rec = _rec(close, forward_run=_run(["A", "B"]), result={"ca_quarantined": ["B"]})
        b = rec["benchmark"]
        assert b["return"] == pytest.approx(0.10 / 2) and b["quarantined"] == ["B"]

    def test_a_low_exposure_difference_is_not_called_alpha(self):
        close = pd.DataFrame({"A": [100.0, 102.0], "B": [50.0, 51.0]}, index=_IDX)     # 基准 +2%
        rec = _rec(close, forward_run=_run(["A", "B"]),
                   result={"execution_target": {"A": 0.15, "B": 0.15}})               # 只有 30% 净敞口
        ex = rec["excess_return"]
        assert ex["paper_vs_full_benchmark"] == pytest.approx(0.01 - 0.02)
        assert ex["paper_exposure_matched"] == pytest.approx(0.01 - 0.30 * 0.02)
        assert "alpha" not in " ".join(ex.keys()).lower()
        assert "不是 alpha" in ex["label"]

    def test_a_formal_start_must_freeze_the_benchmark_universe(self, fresh_db):
        from app.db.forward_record import ForwardRecordStore
        from app.db.strategy_store import StrategyConfig, StrategyStore
        ss = StrategyStore()
        sid = ss.save(StrategyConfig(factors=["1"], combo_weights={"1": 1.0}, aum=1e4, status="proposed",
                                     verdict={"spec_hash": "h"}))
        ss.update_status(sid, "approved"); ss.update_status(sid, "paper_running")
        with pytest.raises(ValueError, match="基准"):
            ForwardRecordStore().start(sid, "h", "kevin", "go")
        run = ForwardRecordStore().start(sid, "h", "kevin", "go", benchmark_universe=["B", "A", "A"])
        import json
        assert json.loads(run.benchmark)["tickers"] == ["A", "B"]


# ---------------------------------------------------------------------------
# F02′（用户决定第 18 条）："价量策略同样依赖拆股、分红、复权及券商股数 / 现金正确。
# 未知公司行动必须隔离、停止新增风险并对账，不能以'没基本面因子'豁免。"
# ---------------------------------------------------------------------------

def _ca_panel():
    idx = pd.bdate_range("2025-03-03", periods=6)
    close = pd.DataFrame({"A": [100.0, 101, 102, 103, 104, 52.5],      # 末日约 1/2：疑似未复权 2:1 拆股
                          "B": [50.0, 50.5, 51, 51.5, 52, 53],         # 正常
                          "C": [20.0, 20, 20, 20, 20, 8.0],            # −60%：极端波动，不像任何拆股比例
                          "D": [10.0, 10, 10, 10, 10, 8.0]},           # −20%：不算
                         index=idx)
    vol = pd.DataFrame(1e6, index=idx, columns=close.columns)
    vol.iloc[-1, 0] = 2e6
    return close, vol


class TestUnknownCorporateActions:

    def test_split_like_and_extreme_moves_are_flagged_on_recent_bars(self):
        from app.core.data_engine.corporate_actions import detect_suspects
        close, vol = _ca_panel()
        got = {s["ticker"]: s for s in detect_suspects(close, vol)}
        assert set(got) == {"A", "C"}
        assert got["A"]["nearest_split"] == "2:1" and got["A"]["bar_date"] == str(close.index[-1].date())
        assert got["C"]["nearest_split"] is None

    def test_flags_persist_until_a_person_resolves_them(self, fresh_db):
        from app.db.corporate_actions import CorporateActionStore
        st = CorporateActionStore()
        assert st.flag("A", "2025-03-10", kind="suspected", ratio=0.505, nearest_split="1:2") is True
        assert st.flag("A", "2025-03-10", kind="suspected", ratio=0.505, nearest_split="1:2") is False   # 去重
        assert st.open_tickers() == {"A": "2025-03-10"}
        with pytest.raises(ValueError):
            st.resolve("A", actor="", resolution="x")
        st.resolve("A", actor="Kevin-Hou422", resolution="券商股数 ×2、现金未变；数据源已复权")
        assert st.open_tickers() == {}
        assert st.history("A")[0]["resolved_by"] == "Kevin-Hou422"

    def test_quarantined_names_cannot_gain_risk_in_the_paper_targets(self):
        from app.core.data_engine.corporate_actions import no_new_risk
        idx = pd.bdate_range("2025-03-03", periods=4)
        w = pd.DataFrame({"A": [0.10, 0.10, 0.20, -0.05], "B": [0.1, 0.2, 0.3, 0.4]}, index=idx)
        out = no_new_risk(w, {"A": str(idx[2].date())})
        assert out["A"].tolist() == [0.10, 0.10, 0.10, 0.0], "隔离之后加了仓 / 反手了"
        pd.testing.assert_series_equal(out["B"], w["B"])

    def test_the_broker_gate_rejects_risk_increasing_orders_on_quarantined_names(self):
        from app.core.execution.order_builder import PlannedOrder
        from app.core.execution.pretrade_gate import GateContext, PreTradeLimits, check
        mk = lambda tk, side, q: PlannedOrder(client_id=f"c-{tk}-{side}", ticker=tk, side=side, qty=q,   # noqa: E731
                                              ref_price=100.0, limit_price=100.5, current_qty=10.0,
                                              target_qty=10.0 + (q if side == "BUY" else -q), target_weight=0.01)
        ctx = GateContext(equity=1e5, buying_power=1e5, current_qty={"A": 10.0}, broker_prices={"A": 100.0},
                          adv_usd={"A": 1e9}, quotes={}, quarantined=frozenset({"A"}))
        dec = check([mk("A", "BUY", 5), mk("A", "SELL", 4)], ctx, PreTradeLimits(.1, 1., .1, .05, .15))
        rej = {o.side: r for o, r in dec.rejected}
        assert rej.get("BUY") == "corporate_action_quarantine"
        assert [o.side for o in dec.approved] == ["SELL"], "隔离不应挡住减仓"

    def test_a_broker_share_jump_is_annotated_for_reconciliation(self):
        from app.core.data_engine.corporate_actions import annotate_discrepancies
        out = annotate_discrepancies([{"ticker": "A", "expected": 10.0, "broker": 20.0},
                                      {"ticker": "B", "expected": 10.0, "broker": 9.0}])
        assert out[0]["suspected_corporate_action"] == "2:1"
        assert "suspected_corporate_action" not in out[1]


def test_the_daily_loop_quarantines_a_suspected_split_and_adds_no_risk(tmp_path, monkeypatch):
    """端到端：末日某只票价格翻倍（疑似合股）→ 记隔离标记、结果里列出、券商目标不超过前一行、隔离名单送到下单层。"""
    from types import SimpleNamespace
    from tests.unit.portfolio_manager.test_audit_portfolio_regressions import _loop
    from app.db.corporate_actions import CorporateActionStore

    def ok_gate(self, *a, **k):
        return SimpleNamespace(passed=True, sharpe=1.0, deflated_sharpe=.99, t_stat=3.0,
                               reasons=[], to_dict=lambda: {"passed": True})
    loop, ds, _ = _loop(tmp_path, monkeypatch, gate_block=False, evaluate=ok_gate)
    tk = ds["close"].columns[0]
    for f in ("open", "high", "low", "close", "vwap"):
        if f in ds:
            ds[f] = ds[f].copy()
            ds[f].iloc[-1, 0] = ds[f].iloc[-2, 0] * 2.0          # 疑似 1:2 合股（价格翻倍）
    clipped = []
    import app.core.data_engine.corporate_actions as ca
    real = ca.no_new_risk

    def spy(w, q):
        # 收缩结果的最后一行强制为 0（合规：平仓不加风险）—— 下面看券商目标是否真的用了收缩后的这份
        res = real(w, q)
        if tk in q:
            res.loc[res.index[-1], tk] = 0.0
        clipped.append((w, res))
        return res
    monkeypatch.setattr(ca, "no_new_risk", spy)

    out = loop.run_portfolio(ds)
    assert tk in out["ca_quarantined"] and out["ca_flags"][tk] == str(ds["close"].index[-1].date())
    assert tk in CorporateActionStore().open_tickers()
    # 券商目标（execution_target）= 收缩后的最后一行；它不比标记前一行更大、也不反手
    w_in, w_out = clipped[-1]
    assert abs(float(w_in[tk].iloc[-1])) > 1e-9, "前提：不收缩时这只票有目标"
    assert float(out["execution_target"].get(tk, 0.0)) == pytest.approx(float(w_out[tk].iloc[-1])) == 0.0, (
        "券商目标没有用收缩后的权重")


def test_the_quarantine_reaches_the_order_gate(monkeypatch):
    """_execute_live 把隔离名单交给 OrderManager.run_cycle（下单层按实际持仓拒绝加风险单）。"""
    from types import SimpleNamespace
    import app.core.execution.order_manager as om
    from app.tasks.daily_trading_loop import DailyTradingLoop
    seen = {}

    class FakeOM:
        def run_cycle(self, **k):
            seen.update(k)
            return SimpleNamespace(to_dict=lambda: {"ok": True})
    monkeypatch.setattr(om.OrderManager, "from_settings", classmethod(lambda cls, *a, **k: FakeOM()))
    loop = DailyTradingLoop(store=SimpleNamespace(), broker=SimpleNamespace(store=None))
    monkeypatch.setattr(loop, "_open_live_gateway", lambda: (SimpleNamespace(close=lambda: None), None))
    idx = pd.bdate_range("2025-03-03", periods=2)
    px = pd.DataFrame({"A": [1.0, 1.0]}, index=idx)
    loop._execute_live(pd.Series({"A": 0.1}), idx[-1], px, px, quarantined={"A"})
    assert seen["quarantined"] == {"A"}


# ---------------------------------------------------------------------------
# D7："保持 NaN。不拿今天快照回填历史，不把缺失变成零市值或便宜估值。"
# ---------------------------------------------------------------------------

class TestMultiClassSharesStayNaN:

    @staticmethod
    def _panels():
        from app.core.data_engine.fundamentals.panels import build_panels
        from tests.unit.data_engine import fundamentals_fixtures as fx
        aapl = fx.facts("AAPL")
        no_shares = aapl[~aapl["concept"].str.startswith("shares")].assign(ticker="BRK-B")   # 像 BRK-B / V：没有可用股本
        idx = pd.bdate_range("2023-03-01", "2023-06-30")
        raw = pd.DataFrame({"AAPL": 160.0, "BRK-B": 300.0}, index=idx)
        p, rep = build_panels(idx, ["AAPL", "BRK-B"], {"AAPL": aapl, "BRK-B": no_shares}, {}, raw)
        return p, rep

    def test_valuation_fields_are_nan_not_zero_or_cheap(self):
        p, _ = self._panels()
        for f in ("market_cap", "earnings_yield", "book_to_price"):
            assert p[f]["BRK-B"].isna().all(), f"{f}：没有股本却算出了值（零市值 / 便宜估值）"
            assert p[f]["AAPL"].notna().any(), f"前提：{f} 对正常公司有值"
        assert p["roe"]["BRK-B"].notna().any(), "不需要股本的字段不该受影响"

    def test_a_cross_sectional_rank_keeps_the_gap_as_a_gap(self):
        from app.core.alpha_engine.fast_ops import cs_fund_rank
        p, _ = self._panels()
        ey = p["earnings_yield"][["AAPL", "BRK-B"]].to_numpy(dtype=float)
        r = cs_fund_rank(np.column_stack([ey, ey[:, :1] * 0.9, ey[:, :1] * 1.1, ey[:, :1] * 1.2]),
                         min_coverage=0.5)
        assert np.isnan(r[:, 1]).all(), "缺失的估值在截面排名里被当成了某个名次"

    def test_no_snapshot_share_source_exists_in_the_fundamentals_path(self):
        """不拿"今天"的券商快照股本回填历史：基本面路径里没有任何快照股本的来源。"""
        import re
        root = Path(__file__).resolve().parents[2] / "app" / "core" / "data_engine" / "fundamentals"
        hits = [f.name for f in root.glob("*.py")
                if re.search(r"issued_shares|total_shares|outstanding_shares|market_snapshot",
                             f.read_text(encoding="utf-8"))]
        assert hits == [], hits


# ---------------------------------------------------------------------------
# D1："基本面暂不正式晋级。允许选择段内预登记诊断和试验记账，不自动消耗最终 Test。"
# ---------------------------------------------------------------------------

def _fund_panel():
    ds = _panel(T=1300)
    rng = np.random.default_rng(1)
    ds["roe"] = pd.DataFrame(rng.normal(0.1, 0.05, ds["close"].shape), ds["close"].index, ds["close"].columns)
    ds["earnings_yield"] = ds["roe"] * 0.5
    return ds


class TestFundamentalsAreNotPromoted:

    @pytest.mark.parametrize("to", ["validated"])
    def test_a_fundamental_factor_cannot_be_promoted_by_any_path(self, fresh_db, to):
        from app.db.alpha_store import AlphaResult, AlphaStore
        s = AlphaStore()
        fund = s.save(AlphaResult(dsl="rank(earnings_yield)", status="candidate"))
        price = s.save(AlphaResult(dsl="rank(ts_delta(close, 5))", status="candidate"))
        with pytest.raises(ValueError, match="D1"):
            s.update_status(fund, to)
        assert s.get_by_id(fund).status == "candidate"
        assert s.update_status(price, to).status == to, "价量因子不受影响"

    def test_the_status_api_refuses_it_too(self, fresh_db, test_client):
        from app.db.alpha_store import AlphaResult, AlphaStore
        aid = AlphaStore().save(AlphaResult(dsl="rank(roe)", status="candidate"))
        r = test_client.patch(f"/api/alphas/{aid}/status", json={"status": "validated"})
        assert r.status_code == 422 and "D1" in r.text

    def test_the_validation_gate_gives_a_selection_only_diagnostic_never_a_pass(self, fresh_db):
        from app.core.lifecycle.validation_gate import ValidationGate
        from app.db.trial_ledger import BoundaryStore, HoldoutLedger
        ds = _fund_panel()
        BoundaryStore().register(REAL, ds["close"].index[-300], actor="Kevin-Hou422", reason="D12")
        res = ValidationGate(n_splits=3, embargo_days=5, dsr_threshold=0.0, min_tstat=-99).evaluate(
            "rank(roe)", ds, n_trials=1, dataset_key=REAL)
        assert res.passed is False and res.diagnostic_only is True
        assert any("D1" in r for r in res.reasons)
        assert res.n_folds >= 3, "诊断指标照样要在选择段上算出来"
        assert res.held_out_test is None and HoldoutLedger().usages(REAL) == []

    def test_pre_registered_diagnostics_are_counted_as_trials_and_never_touch_test(self, fresh_db):
        from app.core.lifecycle.fundamentals_policy import run_fundamental_diagnostics
        from app.db.trial_ledger import BoundaryStore, HoldoutLedger, TrialLedger
        ds = _fund_panel()
        BoundaryStore().register(REAL, ds["close"].index[-300], actor="Kevin-Hou422", reason="D12")
        before = TrialLedger().total()
        out = run_fundamental_diagnostics(["rank(roe)", "rank(earnings_yield)"], ds, REAL,
                                          registered_by="Kevin-Hou422", note="D1 预登记诊断")
        assert TrialLedger().total() == before + 2, "诊断没有记进试验史"
        assert out["registration"]["n"] == 2 and len(out["registration"]["digest"]) == 64
        assert all(r["diagnostic_only"] and r["promotable"] is False for r in out["results"])
        assert HoldoutLedger().usages(REAL) == [], "诊断动用了冻结 Test 段"
        runs = [r for r in TrialLedger().run_states() if r.source == "fund_diag"]
        assert len(runs) == 1 and runs[0].status == "done" and runs[0].n_recorded == 2

    def test_registration_needs_a_person_and_a_purpose(self, fresh_db):
        from app.core.lifecycle.fundamentals_policy import run_fundamental_diagnostics
        with pytest.raises(ValueError):
            run_fundamental_diagnostics(["rank(roe)"], _fund_panel(), REAL, registered_by="", note="x")

    def test_a_strategy_using_fundamentals_is_not_passed(self, fresh_db):
        from app.core.data_engine.research_context import open_research
        from app.core.portfolio_manager.strategy_builder import build_strategy_config
        ds = _panel(T=700)
        with open_research(ds, "", "proposal").active() as ctx:
            cfg = build_strategy_config({"f": ctx.selection["close"].rank(axis=1)}, ctx.selection, aum=1e4,
                                        factor_dsl={"f": "rank(earnings_yield)"})
        g = cfg.verdict["gates"]["fundamentals_policy"]
        assert g["evaluated"] is True and g["passed"] is False and "D1" in g["detail"]
        assert cfg.passed is False


# ---------------------------------------------------------------------------
# N05-c："首轮关闭目标波动。由我手工调整活 .env；你提供预检和配置说明，开发不必因此停下。
#         该决定须体现在新规格中，不能偷偷改变已批准策略。"
# ---------------------------------------------------------------------------

class TestTargetVolOff:

    @staticmethod
    def _pre(tmp_path, **kw):
        from datetime import datetime, timezone
        from app.core.execution.golive import run_preflight
        from tests.unit.execution.test_golive import JOBS, _settings
        rep = run_preflight(_settings(tmp_path, **kw), probe=lambda h, p: False,
                            now=datetime(2025, 3, 4, 22, 0, tzinfo=timezone.utc), cwd=tmp_path,
                            data_probe=lambda st, t: (True, ""), jobs=lambda: JOBS,
                            strategy_check=lambda st: (True, ""), fundamentals_check=lambda st: (True, "", True))
        return {c.name: c for c in rep.checks}["target_vol_off"]

    def test_the_preflight_is_red_while_target_vol_is_still_on(self, tmp_path):
        c = self._pre(tmp_path, risk_target_vol_ann=0.12)
        assert c.ok is False and c.blocking is True and "RISK_TARGET_VOL_ANN" in c.detail

    def test_the_preflight_is_green_once_it_is_off_and_no_running_spec_says_otherwise(self, tmp_path):
        assert self._pre(tmp_path, risk_target_vol_ann=0.0).ok is True

    def test_a_running_strategy_whose_spec_still_targets_vol_must_be_re_proposed(self, tmp_path):
        """改了 .env 不会悄悄改变已批准策略：规格里还开着目标波动 → 预检红，要求重新提案。"""
        from app.core.portfolio_manager.executable_spec import build_spec, spec_hash
        from app.core.portfolio_manager.risk_gate import RiskLimits
        from app.db.strategy_store import StrategyConfig, StrategyStore
        from tests.unit.execution.test_golive import _settings
        s_on = _settings(tmp_path, risk_target_vol_ann=0.12)
        spec = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e6, 0.0, settings=s_on)
        assert spec["risk_limits"]["target_vol_ann"] == pytest.approx(0.12)
        ss = StrategyStore(db_url=s_on.database_url)
        sid = ss.save(StrategyConfig(factors=["1"], combo_weights={"1": 1.0}, aum=1e6, status="proposed",
                                     verdict={"executable_spec": spec, "spec_hash": spec_hash(spec)}))
        ss.update_status(sid, "approved"); ss.update_status(sid, "paper_running")
        c = self._pre(tmp_path, risk_target_vol_ann=0.0)
        assert c.ok is False and "重新提案" in c.detail and f"#{sid}" in c.detail

    def test_the_new_spec_records_the_decision_and_old_specs_stop_matching(self, tmp_path):
        from app.core.portfolio_manager.executable_spec import build_spec, verify_against_runtime
        from tests.unit.execution.test_golive import _settings
        s_on, s_off = _settings(tmp_path, risk_target_vol_ann=0.12), _settings(tmp_path, risk_target_vol_ann=0.0)
        old = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e6, 0.0, settings=s_on)
        new = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e6, 0.0, settings=s_off)
        assert new["risk_limits"]["target_vol_ann"] is None, "新规格里看不出目标波动已关闭"
        diff = verify_against_runtime(old, {"1": "rank(close)"}, 1e6, settings=s_off)
        assert any("target_vol" in d for d in diff), f"旧规格在关闭目标波动后仍被当成一致：{diff}"


def test_the_quarantine_cli_lists_read_only_and_resolves_with_a_reason(tmp_path):
    from app.db.corporate_actions import CorporateActionStore
    from tools.ops.ca_flags import list_flags, main
    db = tmp_path / "x.db"
    st = CorporateActionStore(db_url=f"sqlite:///{db.as_posix()}")
    st.flag("A", "2025-03-10", kind="suspected_price_jump", ratio=0.5, nearest_split="2:1")
    st._engine.dispose()
    assert [r["status"] for r in list_flags(db)] == ["open"]
    with pytest.raises(SystemExit):
        main(["--db", str(db), "resolve", "A", "--actor", "Kevin-Hou422"])            # 缺结论
    assert main(["--db", str(db), "resolve", "A", "--actor", "Kevin-Hou422", "--resolution", "股数 ×2，已复权"]) == 0
    assert [r["status"] for r in list_flags(db)] == ["resolved"]


def test_saving_a_fundamental_factor_directly_in_a_promoted_status_is_refused(fresh_db):
    """D1：直接以 validated / paper 入库也是一条晋级路径，同样拦住（候选状态照常入库）。"""
    from app.db.alpha_store import AlphaResult, AlphaStore
    s = AlphaStore()
    for st in ("validated", "paper", "active"):
        with pytest.raises(ValueError, match="D1"):
            s.save(AlphaResult(dsl="rank(roe)", status=st))
    assert s.save(AlphaResult(dsl="rank(roe)", status="candidate")) > 0
