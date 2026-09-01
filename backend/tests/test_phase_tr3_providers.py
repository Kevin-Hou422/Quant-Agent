"""
test_phase_tr3_providers.py — Phase TR.3 T3 providers（盘口/借券/账户）验收

- SimQuoteProvider：价差来自 Corwin-Schultz 估计（不流动的更宽）、中间价=最近收盘
- SimBorrowProvider：long-only 时全不可做空；margin+allow_short 时仅流动大盘可空
- SimAccountProvider：买入力/持仓来自纸账户真实记账
- **live 模式必须显式抛错**（绝不用估计冒充实时——T3 纪律）
- 接线：run_portfolio 返回 t3 状态
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.trading_context import (
    SimQuoteProvider, SimBorrowProvider, SimAccountProvider, get_trade_providers,
)


def _ds(T=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=T)
    close = pd.DataFrame({
        "BIG":   200 * np.cumprod(1 + rng.normal(0, 0.008, T)),
        "SMALL":   3 * np.cumprod(1 + rng.normal(0, 0.03, T)),
    }, index=idx)
    amp = pd.DataFrame({"BIG": np.abs(rng.normal(0, 0.0015, T)),
                        "SMALL": np.abs(rng.normal(0, 0.02, T))}, index=idx)
    vol = pd.DataFrame({"BIG": rng.uniform(5e6, 1e7, T),
                        "SMALL": rng.uniform(1e4, 5e4, T)}, index=idx)
    return {"open": close, "high": close * (1 + amp), "low": close * (1 - amp),
            "close": close, "vwap": close, "volume": vol,
            "returns": close.pct_change().fillna(0.0)}


def test_sim_quote_provider_spread_and_mid():
    ds = _ds()
    q = SimQuoteProvider(ds)
    assert q.spread_bps("SMALL") > q.spread_bps("BIG")      # 不流动的价差更宽
    assert q.spread_bps("BIG") >= 0.0
    assert abs(q.mid_price("BIG") - ds["close"]["BIG"].iloc[-1]) < 1e-9
    assert np.isfinite(q.spread_bps("UNKNOWN"))             # 未知票退回中位估计


def test_sim_borrow_provider_long_only_and_margin():
    ds = _ds()
    lo = SimBorrowProvider(ds, aum=10_000, allow_short=False)
    assert lo.is_shortable("BIG") is False                   # long-only 全不可空
    assert lo.borrow_fee_bps("BIG") == 0.0
    ms = SimBorrowProvider(ds, aum=1_000_000, account_type="margin", allow_short=True)
    assert ms.is_shortable("BIG") and not ms.is_shortable("SMALL")


def test_sim_account_provider_reads_paper_book(tmp_path):
    from app.db.position_store import PositionStore
    from app.core.execution.paper_broker import PaperBroker
    broker = PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp_path/'p.db'}"),
                         initial_capital=10_000.0)
    a = SimAccountProvider(broker, book_id=0)
    assert a.buying_power() == pytest.approx(10_000.0)       # 空账户 equity=1.0 → $10k
    assert a.positions() == {}
    assert a.cash() == pytest.approx(10_000.0)


def test_live_mode_refuses_rather_than_faking():
    # T3 纪律：live 未接入时必须显式抛错，不能静默退回估计
    with pytest.raises(NotImplementedError, match="Phase 12"):
        get_trade_providers("live")


def test_factory_sim_requires_dataset_and_broker():
    with pytest.raises(ValueError):
        get_trade_providers("sim")


def test_run_portfolio_exposes_t3(tmp_path):
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
    assert out["t3"] is not None and out["t3"]["mode"] == "sim"
    assert out["t3"]["buying_power"] > 0
