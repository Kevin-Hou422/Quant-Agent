"""
test_phase_pm_wiring.py — 验证 PM.S1/S2/PM.5 已接进主线 run_portfolio（不是"库但没生效"）

- run_portfolio 返回 selection / strategy_verdict / risk_report / drawdown 四个字段
- 风控**真的改了权本**（集中持仓 → 单票被削），即 PM.5 生效而非旁路
- 策略门对组合产出 verdict（PM.S1 生效）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ds(T=200, N=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=T)
    cols = [f"S{i}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.015, (T, N)), 0), idx, cols)
    vol = pd.DataFrame(rng.uniform(1e6, 5e6, (T, N)), idx, cols)
    return {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
            "vwap": close, "volume": vol, "returns": close.pct_change().fillna(0.0)}


def _loop_with_paper_factors(tmp_path, dsls):
    from app.db.alpha_store import AlphaStore, AlphaResult
    from app.db.position_store import PositionStore
    from app.core.execution.paper_broker import PaperBroker
    from app.tasks.daily_trading_loop import DailyTradingLoop

    store = AlphaStore(db_url=f"sqlite:///{tmp_path/'a.db'}")
    for dsl in dsls:
        aid = store.save(AlphaResult(dsl=dsl, status="candidate"))
        store.update_status(aid, "validated"); store.update_status(aid, "paper")
    broker = PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp_path/'p.db'}"),
                         initial_capital=10_000.0)
    return DailyTradingLoop(store=store, broker=broker)


def test_run_portfolio_exposes_pm_gate_and_risk_fields(tmp_path):
    loop = _loop_with_paper_factors(tmp_path, [
        "rank(ts_delta(close,5))", "rank((-ts_std(returns,20)))", "rank(ts_delta(log(close),60))",
    ])
    out = loop.run_portfolio(_ds(), aum=10_000.0)
    # 四个新字段都在 → 三个门都接进了主线
    for k in ("selection", "strategy_verdict", "risk_report", "drawdown"):
        assert k in out, f"缺字段 {k}"
    assert out["days_processed"] > 0


def test_risk_gate_actually_clips_positions(tmp_path):
    # 单只波动因子 → 会产生集中持仓 → 单票上限应真的削到 ≤10%
    from app.config import settings
    settings.risk_max_name_weight = 0.10
    loop = _loop_with_paper_factors(tmp_path, ["rank(ts_delta(close,5))",
                                               "rank((-ts_std(returns,20)))"])
    out = loop.run_portfolio(_ds(seed=1), aum=10_000.0)
    rr = out["risk_report"]
    # 风控生效的证据：有过单票削或行业缩（不是旁路）
    assert rr["n_name_clipped"] > 0 or rr["n_sector_scaled"] > 0


def test_strategy_gate_produces_verdict(tmp_path):
    loop = _loop_with_paper_factors(tmp_path, ["rank(ts_delta(close,5))",
                                               "rank(ts_delta(log(close),60))"])
    out = loop.run_portfolio(_ds(seed=2), aum=10_000.0)
    sv = out["strategy_verdict"]
    assert sv is not None and "passed" in sv and "deflated_sharpe" in sv
