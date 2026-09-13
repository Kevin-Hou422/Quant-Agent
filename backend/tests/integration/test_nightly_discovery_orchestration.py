"""
test_phase9_discovery.py — Phase 9.2 自主发现编排器 + 夜间任务验收

覆盖：
  - DiscoveryEngine.run 只吃 dataset（无用户假设）→ 观察→GP→存 CANDIDATE
  - 产出的因子状态为 candidate（分级地基）
  - 调度器在 ENABLE_DISCOVERY 时注册 nightly_discovery
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.discovery.discovery_engine import DiscoveryEngine, DiscoveryReport


def _trending_dataset(T=240, N=12, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    ret = 0.0015 + rng.normal(0, 0.004, (T, N))
    idx = pd.bdate_range("2021-01-01", periods=T)
    cols = [f"A{i:02d}" for i in range(N)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=idx, columns=cols)
    r = pd.DataFrame(ret, index=idx, columns=cols)
    vol = pd.DataFrame(1e6, index=idx, columns=cols)
    return {"open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": vol, "vwap": close, "returns": r}


def test_discovery_runs_without_user_hypothesis_and_saves_candidates(tmp_path):
    from app.db.alpha_store import AlphaStore
    store = AlphaStore(db_url=f"sqlite:///{tmp_path/'disc.db'}")

    # 小 GP 参数，1 个家族，跑得快
    eng = DiscoveryEngine(n_families=1, pop_size=8, n_generations=1,
                          n_optuna_trials=1, oos_ratio=0.3, seed=42)
    report = eng.run(_trending_dataset(), store=store, save=True, auto_validate=False)

    assert isinstance(report, DiscoveryReport)
    assert report.regime in ("bull", "bear", "high_vol", "sideways")
    assert len(report.families) == 1                    # 家族来自观察引擎
    assert report.n_candidates >= 1, report.to_dict()

    c = report.candidates[0]
    assert c.alpha_id is not None
    assert c.dsl
    # 落库状态必须是 candidate（不再直接 active）
    rec = store.get_by_id(c.alpha_id)
    assert rec is not None and rec.status == "candidate"
    assert "auto-discovery" in rec.hypothesis


def test_discovery_dedupes_within_run(tmp_path):
    from app.db.alpha_store import AlphaStore
    store = AlphaStore(db_url=f"sqlite:///{tmp_path/'disc2.db'}")
    eng = DiscoveryEngine(n_families=2, pop_size=8, n_generations=1,
                          n_optuna_trials=1, oos_ratio=0.3, seed=1)
    report = eng.run(_trending_dataset(seed=3), store=store, save=True, auto_validate=False)
    dsls = [c.dsl for c in report.candidates]
    assert len(dsls) == len(set(dsls))                  # 本轮无重复 DSL


def test_discovery_auto_validate_runs_gate(tmp_path):
    """auto_validate=True：每个候选自动跑验证门，落 gate 结果；通过者升 VALIDATED。"""
    from app.db.alpha_store import AlphaStore
    store = AlphaStore(db_url=f"sqlite:///{tmp_path/'disc3.db'}")
    eng = DiscoveryEngine(n_families=1, pop_size=8, n_generations=1,
                          n_optuna_trials=1, oos_ratio=0.3, seed=42)
    report = eng.run(_trending_dataset(), store=store, save=True, auto_validate=True)
    assert report.n_candidates >= 1
    for c in report.candidates:
        assert c.gate is not None                       # 验证门确实跑过
        assert isinstance(c.validated, bool)
        # 状态与门结论一致
        st = store.get_by_id(c.alpha_id).status
        assert st == ("validated" if c.validated else "candidate")


def test_scheduler_registers_discovery_job(monkeypatch):
    monkeypatch.setattr("app.config.settings.enable_discovery", True)
    from app.tasks.scheduler import create_scheduler
    sched = create_scheduler(db_url="sqlite:///:memory:")
    assert "nightly_discovery" in {j.id for j in sched.get_jobs()}


def test_scheduler_omits_discovery_when_disabled(monkeypatch):
    monkeypatch.setattr("app.config.settings.enable_discovery", False)
    from app.tasks.scheduler import create_scheduler
    sched = create_scheduler(db_url="sqlite:///:memory:")
    assert "nightly_discovery" not in {j.id for j in sched.get_jobs()}
