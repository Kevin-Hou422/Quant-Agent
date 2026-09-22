"""
Phase S 各处"读不到配置"时的兜底方向（DEV_LESSONS §U）

§U 的规矩：兜底要朝**保守**一侧倒。Phase S 里"保守"的含义很具体 ——
读不到配置时应当**照切三段、照用 CPCV**，而不是"那就不切了/用更松的口径"。
后者不会让任何功能测试变红：函数都还对，只是从此不再有样本外。

这些分支平时永远走不到，所以必须专门构造：把 `app.config` 模块打成不可导入，
让那几处 `except` 真的被执行一次。
"""
from __future__ import annotations

import sys

import pytest


@pytest.fixture
def config_unreadable(monkeypatch):
    """让 `from app.config import settings` 抛错（模拟配置读不出来）。"""
    monkeypatch.setitem(sys.modules, "app.config", None)
    yield


class TestGatesStaySctrictWhenConfigIsUnreadable:

    def test_the_validation_gate_still_splits_three_ways(self, config_unreadable):
        from app.core.lifecycle.validation_gate import ValidationGate
        assert ValidationGate().three_way is True, (
            "读不到 s_three_way_enabled 就不切三段 —— 门会看到本该冻结的数据，"
            "这是朝**更松**的一侧兜底")

    def test_the_strategy_gate_still_uses_cpcv(self, config_unreadable):
        from app.core.portfolio_manager.strategy_gate import StrategyGate
        gate = StrategyGate(use_global_trials=False)
        assert gate.use_cpcv is True, (
            "读不到 s_use_cpcv 就退回 CSCV —— CSCV 块间无 purge、PBO 偏低，"
            "等于门悄悄变松")
        assert gate.embargo_days == 20, f"embargo 兜底成了 {gate.embargo_days}"

    # GP 适应度口径的同型兜底（读不到 s_fitness_mode 时仍走 purged_cv）放在
    # tests/integration/test_phase_s_fitness_wiring.py —— 那条要构造
    # PopulationEvolver，按 DEV_LESSONS §Z 属于集成层。

    def test_the_partitioner_still_freezes_two_years(self, config_unreadable):
        import numpy as np
        import pandas as pd

        from app.core.data_engine.data_partitioner import split_from_settings

        idx = pd.bdate_range("2018-01-01", periods=1300)
        panel = {"close": pd.DataFrame(np.random.default_rng(1).normal(100, 1, (1300, 2)),
                                       index=idx, columns=["A", "B"]),
                 "volume": pd.DataFrame(1e6, index=idx, columns=["A", "B"])}
        s = split_from_settings(panel)
        assert s.frozen_by == "years" and s.embargo_days == 20, (
            f"读不到配置时切分退化成了 {s.frozen_by} / embargo={s.embargo_days}")
        assert s.n_test > 0, "兜底路径切出了空的 Test 段"

    def test_the_holdout_budget_falls_back_to_one(self, config_unreadable, tmp_path,
                                                  monkeypatch):
        """
        预算读不到时必须按**最严**的 1 次计。兜底成一个大数字等于
        "随便看几次都不算超" —— 台账还在，但那条线没了。
        """
        import numpy as np
        import pandas as pd

        from app.core.data_engine import data_partitioner as dp

        seen = {}

        class _Ledger:
            def __init__(self, db_url=None, budget=1):
                seen["budget"] = budget

            def record_use(self, dataset_key, test_key, purpose=""):
                from app.db.trial_ledger import HoldoutUsage
                return HoldoutUsage(dataset_key=dataset_key, test_key=test_key,
                                    uses=1, budget=seen["budget"], over_budget=False)

        monkeypatch.setattr("app.db.trial_ledger.HoldoutLedger", _Ledger)

        idx = pd.bdate_range("2018-01-01", periods=1300)
        panel = {"close": pd.DataFrame(np.random.default_rng(2).normal(100, 1, (1300, 2)),
                                       index=idx, columns=["A", "B"]),
                 "volume": pd.DataFrame(1e6, index=idx, columns=["A", "B"])}
        # split_from_settings 自身也会走兜底，这正是被测场景
        split = dp.split_from_settings(panel)
        dp.account_holdout_use(split, dataset_key="ds", purpose="t")
        assert seen["budget"] == 1, f"预算兜底成了 {seen['budget']}，不是最严的 1"
