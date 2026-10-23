"""
选择流程只看选择段：提案、工作流、发现、聊天工具、命令行都不碰冻结 Test 段；Test 只经显式且记账的出口。

来源（原测试文件的说明，保留出处）：

外部审查 2026-10-18 F03 / F04 / F05：研究隔离与试验记账。

F04  Test 起点此前 = 本次数据末日 − 2 年，数据每多一天就滑动一天，test_key 又是起止日期，
     于是每次都是"第 1 次使用"（R1：两个窗口重叠 521 天，uses 都是 1）。
     现在：真实数据集第一次切分时把 Test 起点（绝对日期）存进库；holdout 按区间重叠计次。
F03  提案、工作流、自主发现的观察引擎此前都看全量数据（R2、R3）；工作流每次自动汇报 Test 且不记账。
     现在：所有选择流程只看选择段；Test 只经显式、记账的出口。
F05  试验数此前只有自主发现按配置估算一次。现在 GP / Optuna 底层按实际评估数、按 run 幂等记账；
     台账读不到 → 验证门判不通过（证据不完整）。

---

复审 2026-10-19（`review-v6/audit-review/复审报告_2026-10-19.md`）N01–N09 的回归。

每条都从复审独立脚本（`independent_checks.py`）里对应的 residual_* 场景改写：原脚本断言"问题仍在"，
这里断言修复后应有的行为。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from app.core.data_engine.data_partitioner import (
    split_from_settings,
)
from tests.conftest import _make_dataset


@pytest.fixture
def ds():
    return _make_dataset(n_days=1600, n_tickers=10)


def _holdout_rows(url) -> int:
    from sqlalchemy import create_engine, text
    with create_engine(url).connect() as c:
        return int(c.execute(text("SELECT count(*) FROM holdout_usages")).scalar_one())


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


# ---------------------------------------------------------------------------
# N09：CLI 也只看研究窗口
# ---------------------------------------------------------------------------

def test_cli_gp_never_hands_frozen_dates_to_gp(fresh_db):
    import app.main as main
    from app.core.data_engine.data_partitioner import split_from_settings
    from app.core.gp_engine.population_evolver import PopulationEvolver
    ds = _make_dataset(1600, 10)
    frozen = split_from_settings(ds, dataset_key="audit_us")
    seen = {}

    class Stop(Exception):
        pass

    def capture(self, *a, **kw):
        seen["overlap"] = int((kw["oos_data"]["close"].index >= frozen.test_start).sum())
        raise Stop()
    with patch.object(main, "_load_dataset", return_value=ds), patch.object(PopulationEvolver, "__init__", capture):
        with pytest.raises(Stop):
            main.run_gp(SimpleNamespace(pop_size=2, generations=1, optuna_trials=0, dataset="audit_us"))
    assert seen["overlap"] == 0


def test_cli_refuses_data_that_starts_inside_the_frozen_segment(fresh_db):
    import app.main as main
    from app.core.data_engine.data_partitioner import split_from_settings
    from app.db.trial_ledger import BoundaryStore
    ds = _make_dataset(1600, 10)
    frozen = split_from_settings(ds, dataset_key="audit_us")
    BoundaryStore().register("audit_us", frozen.test_start, actor="tester", reason="N09 回归")   # D12：人登记
    late = {k: v.loc[frozen.test_start - pd.Timedelta(days=20):] for k, v in ds.items()}
    with patch.object(main, "_load_dataset", return_value=late):
        with pytest.raises(SystemExit):
            main.run_gp(SimpleNamespace(pop_size=2, generations=1, optuna_trials=0, dataset="audit_us"))


def test_workflow_pnl_split_stays_inside_the_research_window(fresh_db):
    import app.api.router as router
    from app.core.data_engine.data_partitioner import split_from_settings
    ds = _make_dataset(1600, 10)
    with patch("app.core.data_engine.dataset_registry.load_registry_dataset",
               return_value=SimpleNamespace(data=ds)):
        full, is_, oos = router._resolve_dataset("audit_us", "", "", 10, 100, 1, 0.3)
    frozen = split_from_settings(ds, dataset_key="audit_us")
    assert len(full["close"]) == 1600
    assert oos["close"].index[-1] < frozen.test_start
