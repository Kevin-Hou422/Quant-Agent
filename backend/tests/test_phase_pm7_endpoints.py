"""
test_phase_pm7_endpoints.py — Phase PM.7 策略配置端点 + active 配置交易接线

- /strategies/pending、/approve(activate)、/reject、/{id}(+谱系) 端点闭环
- run_portfolio 在有 active 配置时**只交易该配置的成分**
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _save_cfg(status="proposed", factors=("1", "2")):
    from app.dependencies import get_strategy_store
    from app.db.strategy_store import StrategyConfig
    s = get_strategy_store()
    return s, s.save(StrategyConfig(factors=list(factors), combo_weights={f: 0.5 for f in factors},
                                    aum=10_000, passed=True, status=status, name="ep-test"))


def test_strategy_endpoints_lifecycle(test_client):
    s, sid = _save_cfg()
    # pending 含它
    r = test_client.get("/api/strategies/pending")
    assert r.status_code == 200 and any(x["id"] == sid for x in r.json())
    # approve + activate
    r = test_client.post(f"/api/strategies/{sid}/approve", json={"activate": True, "reason": "ok"})
    assert r.status_code == 200 and r.json()["status"] == "active"
    # 详情含谱系
    r = test_client.get(f"/api/strategies/{sid}")
    assert r.status_code == 200 and len(r.json()["decisions"]) >= 2   # approve + activate
    # 已 active 不能再 approve
    assert test_client.post(f"/api/strategies/{sid}/approve").status_code == 409


def test_strategy_reject_endpoint(test_client):
    s, sid = _save_cfg()
    r = test_client.post(f"/api/strategies/{sid}/reject", json={"reason": "no"})
    assert r.status_code == 200 and r.json()["status"] == "rejected"


def test_run_portfolio_trades_active_config(tmp_path):
    from app.db.alpha_store import AlphaStore, AlphaResult
    from app.db.position_store import PositionStore
    from app.db.strategy_store import StrategyStore, StrategyConfig
    from app.core.execution.paper_broker import PaperBroker
    from app.tasks.daily_trading_loop import DailyTradingLoop

    db = f"sqlite:///{tmp_path/'a.db'}"
    store = AlphaStore(db_url=db)
    ids = []
    for dsl in ["rank(ts_delta(close,5))", "rank((-ts_std(returns,20)))", "rank(ts_delta(log(close),60))"]:
        aid = store.save(AlphaResult(dsl=dsl, status="candidate"))
        store.update_status(aid, "validated"); store.update_status(aid, "paper")
        ids.append(str(aid))
    # active 配置只含前 1 个因子
    sstore = StrategyStore(db_url=db)
    sid = sstore.save(StrategyConfig(factors=[ids[0]], combo_weights={ids[0]: 1.0},
                                     aum=10_000, passed=True, status="proposed"))
    sstore.update_status(sid, "approved"); sstore.update_status(sid, "active")

    broker = PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp_path/'p.db'}"), initial_capital=10_000.0)
    loop = DailyTradingLoop(store=store, broker=broker)
    # 让 run_portfolio 的 StrategyStore() 指向同一库
    import app.config as cfgmod
    old = cfgmod.settings.database_url
    cfgmod.settings.database_url = db
    try:
        rng = np.random.default_rng(0); idx = pd.bdate_range("2022-01-03", periods=160); cols = [f"S{i}" for i in range(10)]
        close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.015, (160, 10)), 0), idx, cols)
        ds = {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
              "vwap": close, "volume": pd.DataFrame(1e6, idx, cols), "returns": close.pct_change().fillna(0.0)}
        out = loop.run_portfolio(ds, aum=10_000.0)
    finally:
        cfgmod.settings.database_url = old
    assert out["active_config"] == sid        # 按 active 配置交易
    assert out["n_factors"] == 1              # 只交易配置里的 1 个因子（不是全部 3 个）
