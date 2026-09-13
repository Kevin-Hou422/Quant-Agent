"""
test_phase_pm6_horizon.py — Phase PM.6 horizon 感知 验收

- annualized_turnover / classify_horizon 正确
- apply_no_trade_band 真的**减少换手**（小漂移被吸收），band=0 不变
- horizon_profile 给出每因子快慢
- 接线:run_portfolio 返回 no_trade_band / turnover_ann / horizon，且无交易带生效
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.portfolio_manager import (
    annualized_turnover, classify_horizon, apply_no_trade_band, horizon_profile,
)


def _panel(rows):
    idx = pd.bdate_range("2024-01-02", periods=len(rows))
    return pd.DataFrame(rows, index=idx, columns=["A", "B"])


def test_turnover_and_classify():
    const = _panel([[0.5, 0.5]] * 10)
    assert annualized_turnover(const) == 0.0
    assert classify_horizon(0.0) == "slow"
    assert classify_horizon(10.0) == "fast"


def test_no_trade_band_reduces_turnover():
    # A 每日在 0.5/0.52 间小幅震荡（漂移 0.02）
    rows = [[0.5, 0.5], [0.52, 0.5], [0.5, 0.5], [0.52, 0.5], [0.5, 0.5], [0.52, 0.5]]
    w = _panel(rows)
    to_before = annualized_turnover(w)
    banded = apply_no_trade_band(w, band=0.05)     # 0.02 < 0.05 → 不动
    to_after = annualized_turnover(banded)
    assert to_after < to_before                     # 换手被压低
    assert (banded["A"] == 0.5).all()               # 小漂移全被吸收


def test_no_trade_band_zero_is_noop():
    w = _panel([[0.5, 0.5], [0.9, 0.1]])
    pd.testing.assert_frame_equal(apply_no_trade_band(w, band=0.0), w)


def test_no_trade_band_large_move_passes():
    w = _panel([[0.5, 0.5], [0.9, 0.1]])            # 漂移 0.4 ≥ band
    banded = apply_no_trade_band(w, band=0.05)
    assert banded.iloc[-1]["A"] == 0.9              # 大调仓照常执行


def test_horizon_profile_classifies():
    idx = pd.bdate_range("2022-01-03", periods=120)
    cols = [f"S{i}" for i in range(6)]
    rng = np.random.default_rng(0)
    slow_sig = pd.DataFrame(np.repeat(rng.normal(0, 1, (1, 6)), 120, axis=0), idx, cols)  # 不变→慢
    fast_sig = pd.DataFrame(rng.normal(0, 1, (120, 6)), idx, cols)                        # 每日新→快
    prof = {p.factor: p for p in horizon_profile({"slow": slow_sig, "fast": fast_sig})}
    assert prof["slow"].turnover_ann <= prof["fast"].turnover_ann


# -------------------------------------------------------------------------
# 接线
# -------------------------------------------------------------------------

def test_run_portfolio_exposes_horizon_and_band(tmp_path):
    from app.db.alpha_store import AlphaStore, AlphaResult
    from app.db.position_store import PositionStore
    from app.core.execution.paper_broker import PaperBroker
    from app.tasks.daily_trading_loop import DailyTradingLoop

    store = AlphaStore(db_url=f"sqlite:///{tmp_path/'a.db'}")
    for dsl in ["rank(ts_delta(close,5))", "rank((-ts_std(returns,20)))"]:
        aid = store.save(AlphaResult(dsl=dsl, status="candidate"))
        store.update_status(aid, "validated"); store.update_status(aid, "paper")
    broker = PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp_path/'p.db'}"),
                         initial_capital=10_000.0)

    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-03", periods=180); cols = [f"S{i}" for i in range(10)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.015, (180, 10)), 0), idx, cols)
    ds = {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
          "vwap": close, "volume": pd.DataFrame(1e6, idx, cols),
          "returns": close.pct_change().fillna(0.0)}
    out = DailyTradingLoop(store=store, broker=broker).run_portfolio(ds, aum=10_000.0)
    assert "no_trade_band" in out and "turnover_ann" in out and "horizon" in out
    assert out["no_trade_band"] >= 0.0
    assert isinstance(out["horizon"], list) and len(out["horizon"]) >= 1
