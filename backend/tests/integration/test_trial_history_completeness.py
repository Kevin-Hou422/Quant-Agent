"""
试验史完整性：进行中 / 中断 / 漏记的搜索运行一律不给正式结论（不设宽限期）。

来源（原测试文件的说明，保留出处）：

2026-10-22 用户决定 · 批 1：正确性与状态（`docs/DECISION_IMPLEMENTATION_TRACKER.md` C1 / C2 / C4 / C5 / C6）。

每条断言都对着用户的原话写；先跑、先看它们在旧代码上失败，再改。

---

复审 2026-10-19（`review-v6/audit-review/复审报告_2026-10-19.md`）N01–N09 的回归。

每条都从复审独立脚本（`independent_checks.py`）里对应的 residual_* 场景改写：原脚本断言"问题仍在"，
这里断言修复后应有的行为。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tests.conftest import _make_dataset

# ---------------------------------------------------------------------------
# C5：试验史 —— 进行中 / 未结案的运行一律不给正式资格
# ---------------------------------------------------------------------------

class TestOpenRuns:

    def test_a_fresh_running_run_already_blocks_formal_verdicts(self, fresh_db):
        from app.db.trial_ledger import TrialLedger, trial_history_problem
        TrialLedger().begin_run("gp")                        # 刚开始、远未到 6 小时
        assert trial_history_problem(), "进行中的运行在 6 小时内被当成了完整的试验史"

    def test_a_closed_history_is_complete(self, fresh_db):
        from app.db.trial_ledger import TrialLedger, trial_history_problem
        led = TrialLedger()
        rid = led.begin_run("gp")
        led.add_progress(rid, 3)
        led.finish_run(rid, "done")
        assert trial_history_problem() == ""


# ---------------------------------------------------------------------------
# N07：试验没记全，门不给正式结论
# ---------------------------------------------------------------------------

class TestIncompleteTrialHistory:

    def test_a_run_that_never_finished_blocks_the_strategy_gate(self, fresh_db):
        import app.core.portfolio_manager.strategy_gate as sg
        from app.db.trial_ledger import TrialLedger
        led = TrialLedger()
        led.add(1)
        run = led.begin_run("gp")                                       # 开始了、没结束（进程被杀）
        led.add_progress(run, 3)
        ds = _make_dataset(300, 10)
        rets = pd.Series(np.random.default_rng(2).normal(.01, .001, 300), index=ds["close"].index)
        later = datetime.utcnow() + timedelta(hours=7)
        with patch.object(sg, "strategy_net_returns", return_value=(rets, ds["close"])), \
             patch("app.db.trial_ledger.datetime") as dt:
            dt.utcnow.return_value = later
            verdict = sg.StrategyGate().evaluate({"f": ds["close"]}, ds)
        assert verdict.passed is False and "试验史不完整" in verdict.reasons[0]

    def test_resolving_the_run_adds_the_estimate_and_unblocks(self, fresh_db):
        from app.db.trial_ledger import TrialLedger, trial_history_problem
        led = TrialLedger()
        run = led.begin_run("gp")
        with patch("app.db.trial_ledger.datetime") as dt:
            dt.utcnow.return_value = datetime.utcnow() + timedelta(hours=7)
            assert trial_history_problem() != ""
        assert led.resolve_run(run, 50, "kevin", "进程被杀，按配置的种群 × 代数估计") == 50
        assert trial_history_problem() == ""
        (st,) = led.run_states()
        assert st.status == "resolved" and "kevin" in st.note

    def test_a_progress_write_failure_marks_the_run_incomplete(self, fresh_db, monkeypatch):
        from app.db.trial_ledger import (
            TrialLedger,
            TrialRecorder,
            trial_history_problem,
        )
        rec = TrialRecorder("gp")
        monkeypatch.setattr(TrialLedger, "add_progress", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        rec.add(7)
        rec.finish("done")
        (st,) = TrialLedger().run_states()
        assert rec.ok is False and st.status == "incomplete" and "7 次" in st.note
        assert trial_history_problem() != ""                             # 立即生效，不等过期

    def test_if_even_the_finish_marker_fails_the_run_stays_running(self, fresh_db, monkeypatch):
        from app.db.trial_ledger import TrialLedger, TrialRecorder
        rec = TrialRecorder("gp")
        boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))   # noqa: E731
        monkeypatch.setattr(TrialLedger, "add_progress", boom)
        monkeypatch.setattr(TrialLedger, "finish_run", boom)
        rec.add(7)
        rec.finish("done")
        (st,) = TrialLedger().run_states()
        assert st.status == "running"                                    # 过了 STALE_AFTER 被门看见

    def test_a_crashing_search_keeps_what_it_already_recorded(self, fresh_db, monkeypatch):
        from app.core.gp_engine.population_evolver import PopulationEvolver
        from app.db.trial_ledger import TrialLedger
        d = _make_dataset(160, 8)
        ev = PopulationEvolver({k: v.iloc[:110] for k, v in d.items()}, {k: v.iloc[110:] for k, v in d.items()},
                               pop_size=4, n_generations=3, seed=1)
        real = ev._generate_next_population
        monkeypatch.setattr(ev, "_generate_next_population",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("killed")))
        with pytest.raises(RuntimeError, match="killed"):
            ev.run(seed_dsl="rank(close)", n_optuna_trials=0)
        (st,) = TrialLedger().run_states()
        assert st.status == "aborted" and st.n_recorded > 0 and TrialLedger().total() == st.n_recorded
        assert callable(real)


# ---------------------------------------------------------------------------
# N07-c：审查方脚本 residual_lost_trials_not_persisted 的等价场景
# （原脚本用的 record_trials 已删除，那一项是 ImportError，不算验证）
# ---------------------------------------------------------------------------

def test_a_lost_trial_write_blocks_the_gate_right_away(fresh_db, monkeypatch):
    """先登记 1 次；一次 7 个尝试的搜索记账写入暂时失败；**恢复之后马上**评估策略门 —— 不得通过。"""
    import app.core.portfolio_manager.strategy_gate as sg
    from app.db.trial_ledger import TrialLedger, TrialRecorder
    TrialLedger().add(1)
    rec = TrialRecorder("gp")
    real = TrialLedger.add_progress
    monkeypatch.setattr(TrialLedger, "add_progress",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("temporary write failure")))
    rec.add(7)
    monkeypatch.setattr(TrialLedger, "add_progress", real)            # 写入恢复
    rec.finish("done")
    ds = _make_dataset(300, 10)
    rets = pd.Series(np.random.default_rng(2).normal(.01, .001, 300), index=ds["close"].index)
    with patch.object(sg, "strategy_net_returns", return_value=(rets, ds["close"])):
        verdict = sg.StrategyGate().evaluate({"f": ds["close"]}, ds)
    assert verdict.passed is False and "试验史不完整" in verdict.reasons[0]
    (st,) = TrialLedger().run_states()
    assert st.status == "incomplete" and "7" in st.note
