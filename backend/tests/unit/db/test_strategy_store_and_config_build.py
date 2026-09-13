"""
test_phase_pm7_strategy.py — Phase PM.7 策略配置一等实体 验收（核心，端点另测）

- StrategyStore：save/query/状态机/审批谱系；非法流转报错
- build_strategy_config：因子集合 → 一份策略配置（成分/配额/verdict/风控/换手）
- propose_from_paper_factors：从 AlphaStore 的 PAPER 因子构建配置
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.db.strategy_store import StrategyStore, StrategyConfig, IllegalStrategyTransition


def _ds(T=200, N=8, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=T)
    cols = [f"S{i}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0004, 0.015, (T, N)), 0), idx, cols)
    return {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
            "vwap": close, "volume": pd.DataFrame(1e6, idx, cols),
            "returns": close.pct_change().fillna(0.0)}


def test_strategy_store_crud_and_lifecycle(tmp_path):
    store = StrategyStore(db_url=f"sqlite:///{tmp_path/'s.db'}")
    sid = store.save(StrategyConfig(factors=["1", "2"], combo_weights={"1": 0.6, "2": 0.4},
                                    aum=10_000, passed=True, name="e2e"))
    rec = store.get(sid)
    assert rec.status == "proposed" and rec.aum == 10_000
    # proposed → approved → active → retired
    store.update_status(sid, "approved"); store.record_decision(sid, "approve", "proposed", "approved")
    store.update_status(sid, "active")
    assert store.latest_active().id == sid
    store.update_status(sid, "retired")
    # 非法流转
    with pytest.raises(IllegalStrategyTransition):
        store.update_status(sid, "approved")
    assert len(store.get_decisions(sid)) == 1


def test_build_strategy_config(tmp_path):
    from app.core.portfolio_manager import build_strategy_config
    ds = _ds()
    sig1 = ds["close"].pct_change().rank(axis=1)
    sig2 = (-ds["returns"].rolling(20).std()).rank(axis=1)
    cfg = build_strategy_config({"f1": sig1, "f2": sig2}, ds, aum=10_000)
    assert isinstance(cfg, StrategyConfig)
    assert len(cfg.factors) >= 1 and cfg.aum == 10_000
    assert "passed" in cfg.verdict and cfg.no_trade_band >= 0.0


def test_propose_from_paper_factors(tmp_path):
    from app.db.alpha_store import AlphaStore, AlphaResult
    from app.core.portfolio_manager import propose_from_paper_factors
    astore = AlphaStore(db_url=f"sqlite:///{tmp_path/'a.db'}")
    for dsl in ["rank(ts_delta(close,5))", "rank((-ts_std(returns,20)))"]:
        aid = astore.save(AlphaResult(dsl=dsl, status="candidate"))
        astore.update_status(aid, "validated"); astore.update_status(aid, "paper")
    cfg = propose_from_paper_factors(astore, _ds(), aum=10_000)
    assert cfg is not None and len(cfg.factors) >= 1

    # 空库 → None
    empty = AlphaStore(db_url=f"sqlite:///{tmp_path/'empty.db'}")
    assert propose_from_paper_factors(empty, _ds(), aum=10_000) is None
