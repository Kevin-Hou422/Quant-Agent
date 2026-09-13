"""
test_phase_tr4_promotion.py — Phase TR.4 分级晋级门 + 阈值配置化 + 实验模式

- grade_paper_entry：A/B/C 分级；实验模式放行但如实标注等级；关掉实验模式则只放 A
- check_active_promotion：观测不足 / IC 均值≤0 / t≤阈值 → 不过；强证据 → 过
- 阈值全来自配置（改 settings 即改门）
- 接线：strategy approve(activate) 会跑 →ACTIVE 门并把结论写进谱系
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.lifecycle.promotion_gate import (
    PromotionThresholds, grade_paper_entry, check_active_promotion,
)


# ── 第 4 步：进 PAPER 分级 ────────────────────────────────────────────────

def test_paper_entry_grades():
    th = PromotionThresholds(experiment_mode=True)
    okA, dA = grade_paper_entry({"passed": True, "sharpe": 1.2}, th)
    okB, dB = grade_paper_entry({"passed": False, "sharpe": 0.4}, th)
    okC, dC = grade_paper_entry({"passed": False, "sharpe": -0.3}, th)
    assert (dA["grade"], dB["grade"], dC["grade"]) == ("A", "B", "C")
    assert okA and okB and okC                 # 实验模式：都放行收前向证据
    assert dC["gate_passed"] is False          # 但如实标注未过严门


def test_paper_entry_strict_when_experiment_off():
    th = PromotionThresholds(experiment_mode=False)
    okA, _ = grade_paper_entry({"passed": True, "sharpe": 1.0}, th)
    okB, _ = grade_paper_entry({"passed": False, "sharpe": 0.5}, th)
    assert okA is True and okB is False        # 非实验模式：只放 A


# ── 第 5 步：→ACTIVE 最严门 ──────────────────────────────────────────────

def test_active_promotion_rejects_insufficient_days():
    ok, d = check_active_promotion([0.05] * 10, PromotionThresholds(min_forward_days=60))
    assert ok is False and "前向观测不足" in d["reasons"][0]


def test_active_promotion_rejects_negative_ic():
    rng = np.random.default_rng(0)
    ics = list(rng.normal(-0.02, 0.05, 100))   # 均值为负
    ok, d = check_active_promotion(ics, PromotionThresholds(min_forward_days=60))
    assert ok is False and any("均值" in r for r in d["reasons"])


def test_active_promotion_passes_with_strong_forward_evidence():
    rng = np.random.default_rng(1)
    ics = list(rng.normal(0.05, 0.05, 120))    # 强正 IC，t 很大
    ok, d = check_active_promotion(ics, PromotionThresholds(min_forward_days=60, min_ic_tstat=2.0))
    assert ok is True and d["ic_tstat"] > 2.0


def test_thresholds_come_from_config():
    from app.config import settings
    old = settings.tr_min_ic_tstat
    try:
        settings.tr_min_ic_tstat = 99.0        # 把门调到不可能
        th = PromotionThresholds.from_settings()
        assert th.min_ic_tstat == 99.0
        rng = np.random.default_rng(1)
        ok, _ = check_active_promotion(list(rng.normal(0.05, 0.05, 120)), th)
        assert ok is False                      # 阈值确实生效（配置化）
    finally:
        settings.tr_min_ic_tstat = old


# ── 接线：approve(activate) 跑 →ACTIVE 门并写谱系 ────────────────────────

def test_paper_grade_is_wired_into_strategy_config():
    """§K：分级必须真的接进主线 —— build_strategy_config 产出的 verdict 带 paper_grade。"""
    import pandas as pd
    from app.core.portfolio_manager import build_strategy_config
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-03", periods=200); cols = [f"S{i}" for i in range(8)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0004, 0.015, (200, 8)), 0), idx, cols)
    ds = {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
          "vwap": close, "volume": pd.DataFrame(1e6, idx, cols),
          "returns": close.pct_change().fillna(0.0)}
    cfg = build_strategy_config({"f1": close.pct_change().rank(axis=1)}, ds, aum=10_000)
    assert cfg.verdict.get("paper_grade") in ("A", "B", "C")
    assert "paper_entry" in cfg.verdict


def test_paper_grade_is_wired_into_run_portfolio(tmp_path):
    """§K：run_portfolio 的 strategy_verdict 必须带 paper_grade（分级真正生效）。"""
    import pandas as pd
    from app.db.alpha_store import AlphaStore, AlphaResult
    from app.db.position_store import PositionStore
    from app.core.execution.paper_broker import PaperBroker
    from app.tasks.daily_trading_loop import DailyTradingLoop
    store = AlphaStore(db_url=f"sqlite:///{tmp_path/'a.db'}")
    aid = store.save(AlphaResult(dsl="rank(ts_delta(close,5))", status="candidate"))
    store.update_status(aid, "validated"); store.update_status(aid, "paper")
    broker = PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp_path/'p.db'}"),
                         initial_capital=10_000.0)
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-03", periods=150); cols = [f"S{i}" for i in range(8)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.015, (150, 8)), 0), idx, cols)
    ds = {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
          "vwap": close, "volume": pd.DataFrame(1e6, idx, cols),
          "returns": close.pct_change().fillna(0.0)}
    out = DailyTradingLoop(store=store, broker=broker).run_portfolio(ds, aum=10_000.0)
    assert (out["strategy_verdict"] or {}).get("paper_grade") in ("A", "B", "C")


def test_activate_records_tr4_gate_in_lineage(test_client):
    from app.dependencies import get_strategy_store
    from app.db.strategy_store import StrategyConfig
    ss = get_strategy_store()
    sid = ss.save(StrategyConfig(factors=["1"], combo_weights={"1": 1.0}, aum=10_000,
                                 passed=True, name="tr4"))
    r = test_client.post(f"/api/strategies/{sid}/approve", json={"activate": True, "reason": "go"})
    assert r.status_code == 200 and r.json()["status"] == "active"
    det = test_client.get(f"/api/strategies/{sid}").json()
    act = [d for d in det["decisions"] if d["decision"] == "activate"]
    assert act and "TR.4 门" in act[0]["reason"]      # 门结论写进了谱系
