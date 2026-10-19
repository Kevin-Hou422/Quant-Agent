"""
外部审查 2026-10-18 F03 / F04 / F05：研究隔离与试验记账。

F04  Test 起点此前 = 本次数据末日 − 2 年，数据每多一天就滑动一天，test_key 又是起止日期，
     于是每次都是"第 1 次使用"（R1：两个窗口重叠 521 天，uses 都是 1）。
     现在：真实数据集第一次切分时把 Test 起点（绝对日期）存进库；holdout 按区间重叠计次。
F03  提案、工作流、自主发现的观察引擎此前都看全量数据（R2、R3）；工作流每次自动汇报 Test 且不记账。
     现在：所有选择流程只看选择段；Test 只经显式、记账的出口。
F05  试验数此前只有自主发现按配置估算一次。现在 GP / Optuna 底层按实际评估数、按 run 幂等记账；
     台账读不到 → 验证门判不通过（证据不完整）。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from tests.conftest import _make_dataset

from app.core.data_engine.data_partitioner import FrozenBoundaryError, split_from_settings
from app.db.trial_ledger import BoundaryStore, HoldoutLedger, TrialLedger

BACKEND = Path(__file__).resolve().parents[2]


@pytest.fixture
def ds():
    return _make_dataset(n_days=1600, n_tickers=10)


def _cut(d, start=None, end=None):
    return {k: v.loc[start:end] for k, v in d.items()}


def _holdout_rows(url) -> int:
    from sqlalchemy import create_engine, text
    with create_engine(url).connect() as c:
        return int(c.execute(text("SELECT count(*) FROM holdout_usages")).scalar_one())


# ===========================================================================
# F04：冻结边界
# ===========================================================================

class TestFrozenBoundary:

    def test_the_test_start_no_longer_slides_with_the_data_end(self, ds, fresh_db):
        a = split_from_settings(ds, dataset_key="us_x")
        b = split_from_settings({k: v.iloc[:-1] for k, v in ds.items()}, dataset_key="us_x")
        c = split_from_settings({k: v.iloc[:-200] for k, v in ds.items()}, dataset_key="us_x")
        assert a.frozen_by == "years" and b.frozen_by == c.frozen_by == "pinned"
        assert a.test_start == b.test_start == c.test_start
        assert a.selection["close"].index.equals(b.selection["close"].index)
        assert BoundaryStore().get("us_x") == a.test_start

    def test_the_unpinned_contrast_does_slide(self, ds, fresh_db):
        a = split_from_settings(ds)
        b = split_from_settings({k: v.iloc[:-200] for k, v in ds.items()})
        assert a.test_start != b.test_start

    def test_new_data_lands_on_the_test_side(self, ds, fresh_db):
        first = split_from_settings({k: v.iloc[:-100] for k, v in ds.items()}, dataset_key="us_x")
        later = split_from_settings(ds, dataset_key="us_x")
        assert later.test_start == first.test_start
        assert later.n_test == first.n_test + 100
        assert later.selection["close"].index.equals(first.selection["close"].index)

    def test_data_entirely_before_the_boundary_has_no_test(self, ds, fresh_db):
        pinned = split_from_settings(ds, dataset_key="us_x").test_start
        early = split_from_settings(_cut(ds, end=pinned - pd.Timedelta(days=200)), dataset_key="us_x")
        assert early.n_test == 0 and early.test_key == "empty"
        assert all(v.empty for v in early.test.values())

    def test_data_that_starts_inside_the_frozen_segment_is_refused_not_widened(self, ds, fresh_db):
        pinned = split_from_settings(ds, dataset_key="us_x").test_start
        with pytest.raises(FrozenBoundaryError, match="冻结"):
            split_from_settings(_cut(ds, start=pinned - pd.Timedelta(days=30)), dataset_key="us_x")
        assert not issubclass(FrozenBoundaryError, ValueError)     # 各入口的"太短→全量"兜底接不住它

    @pytest.mark.parametrize("key", [None, "", "synthetic", "unknown"])
    def test_synthetic_keys_are_never_pinned(self, ds, fresh_db, key):
        split_from_settings(ds, dataset_key=key)
        assert BoundaryStore().get(key or "") is None

    def test_a_boundary_is_written_once(self, fresh_db):
        bs = BoundaryStore()
        assert bs.pin("k", pd.Timestamp("2024-01-02")) == pd.Timestamp("2024-01-02")
        assert bs.pin("k", pd.Timestamp("2025-06-02")) == pd.Timestamp("2024-01-02")

    def test_an_unreadable_boundary_store_refuses_instead_of_sliding(self, ds, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("db down")
        monkeypatch.setattr("app.db.trial_ledger.BoundaryStore", boom)
        with pytest.raises(FrozenBoundaryError, match="db down"):
            split_from_settings(ds, dataset_key="us_x")

    def test_backtest_entrypoints_refuse_rather_than_fall_back_to_the_full_panel(self, ds, fresh_db):
        from fastapi import HTTPException
        import app.api.router as router
        pinned = split_from_settings(ds, dataset_key="us_x").test_start
        with pytest.raises(HTTPException) as e:
            router._three_way(_cut(ds, start=pinned - pd.Timedelta(days=30)), "us_x", "backtest/run")
        assert e.value.status_code == 422

    def test_validation_gate_fails_closed_on_the_frozen_segment(self, ds, fresh_db):
        from app.core.lifecycle.validation_gate import ValidationGate
        pinned = split_from_settings(ds, dataset_key="us_x").test_start
        res = ValidationGate().evaluate("rank(close)", _cut(ds, start=pinned - pd.Timedelta(days=30)),
                                        n_trials=1, dataset_key="us_x")
        assert res.passed is False and res.reasons[0].startswith("冻结边界")
        assert res.n_folds == 0                                   # 没有退回全量去跑 WalkForward


class TestHoldoutExposureFollowsTheData:

    def test_a_sliding_window_still_counts_as_the_same_exposure(self, ds, fresh_db):
        from app.core.data_engine.data_partitioner import account_holdout_use
        a = split_from_settings(ds, dataset_key="us_x")
        b = split_from_settings({k: v.iloc[:-1] for k, v in ds.items()}, dataset_key="us_x")
        assert account_holdout_use(a, "us_x")["uses"] == 1
        second = account_holdout_use(b, "us_x")
        assert (second["uses"], second["over_budget"]) == (2, True)


# ===========================================================================
# F03：选择流程只看选择段
# ===========================================================================

class TestSelectionOnly:

    def test_strategy_proposal_sees_only_the_selection_segment(self, ds, fresh_db):
        """审查 R2 的反面：此前完整面板进了提案路径。"""
        import app.api.router as router

        class Captured(Exception):
            pass
        seen = {}

        def capture(_store, panel, **kw):
            seen["last"] = panel["close"].index[-1]
            raise Captured()
        with patch("app.core.data_engine.dataset_registry.load_registry_dataset",
                   return_value=SimpleNamespace(data=ds)), \
             patch("app.core.portfolio_manager.propose_from_paper_factors", side_effect=capture):
            with pytest.raises(Captured):
                router.strategy_propose(store=object(), sstore=object())
        from app.config import settings
        split = split_from_settings(ds, dataset_key=settings.paper_dataset)
        assert seen["last"] == split.selection["close"].index[-1] < split.test_start

    def _run_generation(self, ds, **kw):
        import app.core.workflows.alpha_workflows as wf
        from app.core.gp_engine.population_evolver import GPEvolutionResult
        fake = GPEvolutionResult("rank(close)", {}, 1, [], [], None)
        segs = []
        with patch.object(wf, "_generate_diverse_seeds", return_value=["rank(close)"]), \
             patch.object(wf.PopulationEvolver, "run", return_value=fake), \
             patch.object(wf, "_quick_metrics", return_value={"is_sharpe": 1.2}), \
             patch.object(wf, "_combine_pool_alphas",
                          side_effect=lambda pool, data, emit, is_data=None:
                          segs.append(data["close"].index[-1]) or {"sharpe": 0.1}):
            out = wf.GenerationWorkflow(pop_size=2, n_generations=1, n_optuna_trials=0).run(
                "audit", ds, **kw)
        return out, segs

    def test_the_workflow_does_not_look_at_test_by_default(self, ds, fresh_db):
        """审查 R3 的反面：此前 held_out_test=True 而台账新增 0 条。"""
        out, segs = self._run_generation(ds, dataset_key="us_x")
        split = split_from_settings(ds, dataset_key="us_x")
        assert out.metrics["held_out_test"] is False and out.metrics["test_withheld"] is True
        assert "test_sharpe" not in out.metrics
        assert segs == [split.validate["close"].index[-1]]       # 组合也只在 Validate 上评估
        assert out.combined_metrics["eval_segment"] == "validate"
        assert _holdout_rows(fresh_db) == 0

    def test_asking_for_test_is_metered(self, ds, fresh_db):
        out, segs = self._run_generation(ds, dataset_key="us_x", report_test=True)
        assert out.metrics["held_out_test"] is True
        assert out.metrics["holdout_usage"]["uses"] == 1
        assert segs == [ds["close"].index[-1]]
        assert out.combined_metrics["eval_segment"] == "test"
        assert _holdout_rows(fresh_db) == 1

    def test_the_optimization_workflow_never_hands_frozen_dates_to_gp(self, ds, fresh_db):
        """此前 Workflow B 对全量两段切，GP 按最后 30% 择优 —— 正是冻结 Test 所在的位置。"""
        import app.core.workflows.alpha_workflows as wf
        from app.core.gp_engine.population_evolver import GPEvolutionResult
        seen = {}
        real_init = wf.PopulationEvolver.__init__

        def spy(self, is_data, oos_data, **kw):
            seen["last"] = oos_data["close"].index[-1]
            real_init(self, is_data, oos_data, **kw)
        fake = GPEvolutionResult("rank(close)", {}, 1, [], [], None)
        with patch.object(wf.PopulationEvolver, "__init__", spy), \
             patch.object(wf.PopulationEvolver, "run", return_value=fake), \
             patch.object(wf, "_quick_metrics", return_value={
                 "is_sharpe": 1.0, "oos_sharpe": 1.0, "turnover": 0.1, "overfitting_score": 0.0}):
            wf.OptimizationWorkflow(pop_size=2, n_generations=1, n_optuna_trials=0, n_mutations=1).run(
                "rank(close)", ds, dataset_key="us_x")
        assert seen["last"] < split_from_settings(ds, dataset_key="us_x").test_start

    def test_discovery_observes_only_the_selection_segment(self, ds, fresh_db):
        from app.core.discovery import discovery_engine as de
        from app.core.discovery.market_observer import MarketObserver
        seen = {}
        real = MarketObserver.observe

        def spy(self, data):
            seen["last"] = data["close"].index[-1]
            return real(self, data)
        with patch.object(MarketObserver, "observe", spy), \
             patch("app.core.workflows.alpha_workflows.GenerationWorkflow.run",
                   side_effect=RuntimeError("skip gp")):
            de.DiscoveryEngine(n_families=1).run(ds, save=False, dataset_key="us_x")
        assert seen["last"] < split_from_settings(ds, dataset_key="us_x").test_start

    def test_agent_tools_research_inside_the_window_and_meter_the_test(self, ds, fresh_db):
        import json
        from app.agent._tools import QuantTools
        with patch("app.core.data_engine.dataset_registry.load_registry_dataset",
                   return_value=SimpleNamespace(data=ds)):
            tools = QuantTools(dataset_name="us_x")
        split = split_from_settings(ds, dataset_key="us_x")
        assert tools._full_dataset["close"].index[-1] == split.selection["close"].index[-1]
        assert tools._oos_data["close"].index[-1] < split.test_start
        out = json.loads(tools.tool_run_backtest("rank(close)", use_test_set=True))
        assert out["is_true_holdout"] is True and out["holdout_usage"]["uses"] == 1
        assert _holdout_rows(fresh_db) == 1


# ===========================================================================
# F05：试验记账
# ===========================================================================

class TestTrialAccounting:

    def test_the_same_run_is_counted_once(self, fresh_db):
        led = TrialLedger()
        assert led.add(5, run_id="gp:abc", source="gp") == 5
        assert led.add(5, run_id="gp:abc", source="gp") == 5
        assert led.add(3, run_id="gp:def", source="gp") == 8
        assert [r.run_id for r in led.runs("gp")] == ["gp:abc", "gp:def"]

    def test_two_processes_adding_at_once_lose_nothing(self, fresh_db):
        """
        回归护栏，**不是**对旧写法的反证：实测旧的"读 total → 加 → 写回"在 SQLite 上也不丢
        （写事务被串行化，冲突的一方报错重试而不是覆盖）。这条钉的是新写法（原子 UPDATE +
        trial_runs 主键）在两个进程并发时同样不丢、不撞键。
        """
        code = textwrap.dedent("""
            import sys
            sys.path.insert(0, %r)
            from app.db.trial_ledger import TrialLedger
            led = TrialLedger(%r)
            for i in range(40):
                led.add(1, run_id=f"{sys.argv[1]}:{i}", source="t")
        """) % (str(BACKEND), fresh_db)
        TrialLedger(fresh_db)          # 建表在前（两个进程同时对一个全新文件 create_all 是另一回事）
        procs = [subprocess.Popen([sys.executable, "-c", code, tag]) for tag in ("a", "b")]
        for p in procs:
            assert p.wait(timeout=180) == 0
        assert TrialLedger(fresh_db).total() == 80

    def test_gp_records_what_it_actually_evaluated(self, fresh_db):
        from app.core.gp_engine.population_evolver import PopulationEvolver
        d = _make_dataset(n_days=200, n_tickers=8)
        is_d = {k: v.iloc[:140] for k, v in d.items()}
        oos_d = {k: v.iloc[140:] for k, v in d.items()}
        res = PopulationEvolver(is_d, oos_d, pop_size=6, n_generations=2, seed=1).run(
            seed_dsl="rank(close)", n_optuna_trials=0)
        expected = sum(g["population_size"] for g in res.evolution_log)
        assert expected > 0 and res.n_structures_evaluated == expected
        assert res.trials_recorded is True
        # 复审 2026-10-19 N07：先登记、逐代累加、正常结束
        assert [(r.n_recorded, r.source, r.status) for r in TrialLedger().run_states()] == [(expected, "gp", "done")]
        assert TrialLedger().total() == expected

    def test_optuna_records_the_trials_it_ran(self, fresh_db):
        from app.core.ml_engine.alpha_optimizer import AlphaOptimizer
        d = _make_dataset(n_days=200, n_tickers=8)
        opt = AlphaOptimizer(dsl="rank(close)", is_dataset=d, n_trials=3, seed=1)
        _, summary = opt.optimize()
        assert summary.n_trials == 3 and opt.trials_recorded is True
        assert [(r.n_recorded, r.source, r.status) for r in TrialLedger().run_states()] == [(3, "optuna", "done")]

    def test_an_unwritable_ledger_refuses_to_start_a_search(self, monkeypatch):
        """复审 2026-10-19 N07：登记不进去就不搜 —— 没登记的尝试会从研究史里消失。"""
        from app.core.gp_engine.population_evolver import PopulationEvolver
        from app.db.trial_ledger import TrialLedgerUnavailable

        def boom(*a, **k):
            raise RuntimeError("ledger down")
        monkeypatch.setattr("app.db.trial_ledger.TrialLedger", boom)
        d = _make_dataset(n_days=120, n_tickers=6)
        with pytest.raises(TrialLedgerUnavailable, match="拒绝开始 gp 搜索"):
            PopulationEvolver({k: v.iloc[:80] for k, v in d.items()}, {k: v.iloc[80:] for k, v in d.items()},
                              pop_size=4, n_generations=1, seed=1).run(seed_dsl="rank(close)", n_optuna_trials=0)

    def test_validate_api_defaults_to_the_global_count(self, test_client, fresh_db):
        from app.db.alpha_store import AlphaResult, AlphaStore
        aid = AlphaStore(db_url=fresh_db).save(AlphaResult(dsl="rank(close)", status="candidate"))
        seen = {}

        def fake_eval(self, dsl, data, n_trials=None, dataset_key="", report_test=False):
            seen.update(n_trials=n_trials, dataset_key=dataset_key)
            from app.core.lifecycle.validation_gate import ValidationResult
            return ValidationResult(passed=False, dsl=dsl, reasons=["x"])
        with patch("app.core.data_engine.dataset_registry.load_registry_dataset",
                   return_value=SimpleNamespace(data=_make_dataset(n_days=100))), \
             patch("app.core.lifecycle.validation_gate.ValidationGate.evaluate", fake_eval):
            r = test_client.post(f"/api/alphas/{aid}/validate?dataset_name=us_x")
        assert r.status_code == 200, r.text
        assert seen == {"n_trials": None, "dataset_key": "us_x"}
