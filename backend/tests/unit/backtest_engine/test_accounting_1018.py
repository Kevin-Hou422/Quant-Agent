"""
外部审查 2026-10-18 F10 / F11。

F10（R6）持仓按权重保存，调仓时直接拿昨日权重与今日目标做差，没有先按价格变动漂移。
     50/50 两只票、A 涨 10%：调仓前应为 52.38 / 47.62，系统仍当 50 / 50，回到 50 / 50 的成本记成 0。
     两套引擎逐位一致证明的是实现一致，不是账算对了 —— 这里用**手算 oracle**（现金 + 股数账本）对。
F11（R13）成本缓存键只看形状、端点、首末行与前 4 列：改中间的 high，键不变，价差从 12.85 → 1584.34 bps。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.backtest_engine.transaction_cost import CostParams, drift_weights


def test_drift_matches_the_review_r6_numbers():
    w = drift_weights(np.array([0.5, 0.5]), np.array([0.10, 0.0]))
    assert w == pytest.approx([0.55 / 1.05, 0.5 / 1.05])
    assert w[0] == pytest.approx(0.5238095, abs=1e-7)


def test_drift_handles_cash_shorts_and_missing_prices():
    w = drift_weights(np.array([0.3, -0.2, 0.4]), np.array([0.10, 0.05, np.nan]))
    gross = 0.3 * 0.10 + (-0.2) * 0.05
    assert w == pytest.approx(np.array([0.33, -0.21, 0.4]) / (1 + gross))
    assert drift_weights(np.zeros(2), np.array([0.5, -0.5])) == pytest.approx([0.0, 0.0])


def _no_cost():
    return CostParams(fixed_bps=0.0, min_ticket_fee=0.0, spread_bps=0.0, impact_coef=0.0,
                      short_borrow_annual_bps=0.0)


def _panel():
    idx = pd.bdate_range("2025-01-06", periods=3)
    close = pd.DataFrame({"A": [100.0, 110.0, 110.0], "B": [50.0, 50.0, 50.0]}, index=idx)
    weights = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=idx)
    vol = pd.DataFrame(1e12, index=idx, columns=close.columns)
    return idx, close, weights, vol


def _oracle_trades(close, weights, capital):
    """现金 + 股数账本：每天按收盘价把股数调到目标权重，记下成交名义额。"""
    shares = {c: 0.0 for c in close.columns}
    cash = capital
    out = []
    for d in close.index:
        px = close.loc[d]
        eq = cash + sum(shares[c] * px[c] for c in shares)
        traded = 0.0
        for c in shares:
            tgt = weights.loc[d, c] * eq / px[c]
            traded += abs(tgt - shares[c]) * px[c]
            cash -= (tgt - shares[c]) * px[c]
            shares[c] = tgt
        out.append(traded / eq)
    return out


def test_the_backtest_engine_trades_what_a_share_ledger_would_trade():
    from app.core.backtest_engine.backtest_engine import BacktestEngine
    idx, close, weights, vol = _panel()
    res = BacktestEngine(cost_params=_no_cost(), initial_capital=1e6).run(weights, close, vol, weights)
    oracle = _oracle_trades(close, weights, 1e6)
    assert oracle[1] == pytest.approx(2 * (0.55 / 1.05 - 0.5), abs=1e-12)   # 第二天：漂移后调回 50/50
    got = np.asarray(res.turnover, dtype=float) * 2.0               # 单边换手 × 2 = 成交名义额 / 权益
    assert got[1] == pytest.approx(oracle[1], abs=1e-9), "回测把调回 50/50 当成了零成交（权重没漂移）"
    assert got[2] == pytest.approx(oracle[2], abs=1e-9)                    # 价格不动的一天：零成交


def test_the_paper_broker_trades_from_the_drifted_weights(tmp_path):
    from app.core.execution.paper_broker import PaperBroker
    from app.db.position_store import PositionStore
    idx, close, weights, vol = _panel()
    br = PaperBroker(store=PositionStore(f"sqlite:///{(tmp_path / 'p.db').as_posix()}"),
                     cost_params=_no_cost(), initial_capital=1e6)
    adv = pd.Series(1e15, index=close.columns)
    dv = pd.Series(0.02, index=close.columns)
    br.step(1, idx[0], target_w=weights.iloc[0], prices_t=close.iloc[0], prices_prev=close.iloc[0],
            adv_usd=adv, daily_vol=dv, tdays_per_year=252)
    br.step(1, idx[1], target_w=weights.iloc[1], prices_t=close.iloc[1], prices_prev=close.iloc[0],
            adv_usd=adv, daily_vol=dv, tdays_per_year=252)
    traded = sum(abs(f.traded_weight) for f in br.store.fills_on(1, idx[1]))
    assert traded == pytest.approx(_oracle_trades(close, weights, 1e6)[1], abs=1e-9)


def test_paper_broker_and_backtest_still_agree_after_the_fix(tmp_path):
    """两套引擎仍然逐日一致（共用 drift_weights），而且现在一致的是**对的**那份账。"""
    from app.core.backtest_engine.backtest_engine import BacktestEngine
    from app.core.execution.paper_broker import PaperBroker
    from app.db.position_store import PositionStore
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2025-01-06", periods=40)
    cols = list("ABCD")
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (40, 4)), 0), idx, cols)
    w = pd.DataFrame(rng.dirichlet(np.ones(4), 40) * 0.9, idx, cols)
    vol = pd.DataFrame(1e9, idx, cols)
    bt = BacktestEngine(initial_capital=1e6).run(w, close, vol, w)
    br = PaperBroker(store=PositionStore(f"sqlite:///{(tmp_path / 'p.db').as_posix()}"), initial_capital=1e6)
    eq = br.replay(1, w, close, vol)
    assert np.asarray(eq, dtype=float) == pytest.approx(np.asarray(bt.equity_curve, dtype=float), rel=1e-9)


# ---------------------------------------------------------------------------
# F11
# ---------------------------------------------------------------------------

def test_the_review_r13_interior_change_now_changes_the_key():
    import app.core.portfolio_manager.strategy_gate as sg
    from tests.conftest import _make_dataset
    orig = _make_dataset(n_days=80, n_tickers=10)
    changed = {k: v.copy() for k, v in orig.items()}
    changed["high"].iloc[1:-1] = changed["close"].iloc[1:-1] * 1.2
    assert sg._cache_key("cost", orig, 1e4) != sg._cache_key("cost", changed, 1e4)
    assert sg._cache_key("cost", orig, 1e4) == sg._cache_key("cost", {k: v.copy() for k, v in orig.items()}, 1e4)


@pytest.mark.parametrize("field,value", [("trading_broker", "ibkr_us"), ("trading_account_type", "cash"),
                                         ("trading_allow_short", True)])
def test_the_key_follows_the_broker_fields_the_derivation_reads(monkeypatch, field, value):
    import app.core.portfolio_manager.strategy_gate as sg
    from app.config import settings
    from tests.conftest import _make_dataset
    ds = _make_dataset(n_days=40, n_tickers=5)
    before = sg._cache_key("cost", ds, 1e4)
    monkeypatch.setattr(settings, field, value)
    assert sg._cache_key("cost", ds, 1e4) != before
