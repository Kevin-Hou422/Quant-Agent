"""
test_phase_fetr_endpoints.py — FE-TR 数据源验收（诊断持久化 + 两个只读端点）

- run_portfolio 的诊断被**持久化**（此前只进日志），且含 trading_context / t3 / 门分级
- GET /api/portfolio/diagnostics 返回最近若干轮
- GET /api/trading/status 返回配置+OpenD 连通+门开关（轻量，不加载数据集）
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ds(T=150, N=8, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=T)
    cols = [f"S{i}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.015, (T, N)), 0), idx, cols)
    return {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
            "vwap": close, "volume": pd.DataFrame(1e6, idx, cols),
            "returns": close.pct_change().fillna(0.0)}


def test_run_portfolio_result_has_trading_context(tmp_path):
    from app.db.alpha_store import AlphaStore, AlphaResult
    from app.db.position_store import PositionStore
    from app.core.execution.paper_broker import PaperBroker
    from app.tasks.daily_trading_loop import DailyTradingLoop

    store = AlphaStore(db_url=f"sqlite:///{tmp_path/'a.db'}")
    aid = store.save(AlphaResult(dsl="rank(ts_delta(close,5))", status="candidate"))
    store.update_status(aid, "validated"); store.update_status(aid, "paper")
    broker = PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp_path/'p.db'}"),
                         initial_capital=10_000.0)
    out = DailyTradingLoop(store=store, broker=broker).run_portfolio(_ds(), aum=10_000.0)
    tc = out["trading_context"]
    assert tc is not None
    for k in ("median_spread_bps", "n_tradable", "n_shortable", "rebalance_band", "allow_short"):
        assert k in tc
    assert tc["allow_short"] is False              # long-only 现状


def test_diagnostics_persisted_and_endpoint(test_client, tmp_path):
    from app.db.diagnostics_store import DiagnosticsStore
    ds = DiagnosticsStore()                        # conftest 已把 database_url 指向临时库
    before = len(ds.recent(limit=100))
    ds.save({"n_factors": 2, "equity": 1.03, "t3": {"mode": "sim"},
             "trading_context": {"median_spread_bps": 3.2}})
    assert len(ds.recent(limit=100)) == before + 1

    r = test_client.get("/api/portfolio/diagnostics", params={"limit": 5})
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list) and len(rows) >= 1
    assert "run_at" in rows[0] and rows[0].get("t3", {}).get("mode") == "sim"


def test_trading_status_endpoint(test_client):
    r = test_client.get("/api/trading/status")
    assert r.status_code == 200
    d = r.json()
    for k in ("price_source", "same_source", "moomoo", "broker", "account_type",
              "allow_short", "gates"):
        assert k in d
    assert "opend_reachable" in d["moomoo"]
    # 门开关必须可见（让人一眼看出门是否在真拦）
    for k in ("experiment_mode", "enforce_active_gate", "min_forward_days",
              "min_ic_tstat", "factor_gate_mode"):
        assert k in d["gates"]
