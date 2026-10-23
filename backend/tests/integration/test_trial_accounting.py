"""
试验记账：GP / Optuna 在搜索底层按实际评估数、按运行幂等记进全局试验数。

来源（原测试文件的说明，保留出处）：

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

import pytest

from app.db.trial_ledger import TrialLedger
from tests.conftest import _make_dataset

BACKEND = Path(__file__).resolve().parents[2]


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
