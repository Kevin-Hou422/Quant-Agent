"""
复审 2026-10-19 N08-b：用**独立的股数 + 现金参考账本**验算两套引擎。

"回测和 paper 一起改过且彼此相等，仍不能证明经济记账正确" —— 两套引擎互相对账只证明实现一致。
这里的参考账本不用权重：持有股数、现金；每天按收盘价估值、按目标调股数，成交受日成交量上限约束（部分成交），
费用按成交额收，空头按昨收市值收借券费。覆盖：涨跌、现金、空头、费用、借券、部分成交。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.backtest_engine.transaction_cost import CostParams, LiquidityConstraint

FIXED_BPS, BORROW_BPS, MPP = 10.0, 100.0, 0.10


def _params():
    return CostParams(fixed_bps=FIXED_BPS, min_ticket_fee=0.0, spread_bps=0.0, impact_coef=0.0,
                      short_borrow_annual_bps=BORROW_BPS, max_participation_pct=MPP)


def _market(seed=5, T=30):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-02-03", periods=T)
    cols = ["A", "B", "C", "D"]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.03, (T, 4)), 0), idx, cols)
    # D 的成交量很小 → 每天只能成交一小部分（部分成交）；其余充足
    vol = pd.DataFrame(1e9, idx, cols)
    vol["D"] = 300.0
    w = pd.DataFrame({"A": 0.4, "B": 0.3, "C": -0.25, "D": 0.2}, index=idx)
    w.iloc[10:, :] = [[0.1, 0.5, -0.4, -0.15]] * (T - 10)        # 中途换仓：加空头、D 由多翻空
    return close, vol, w


def _oracle(close, vol, w, capital=1e6):
    """股数 + 现金账本。返回每日收盘后的净值 / 初始资金。"""
    dates = close.index
    adv = LiquidityConstraint(_params()).compute_adv(vol, close)      # 市场数据（与引擎同口径），不是账务
    cal_years = max((dates[-1] - dates[0]).days / 365.25, 1 / 365.25)
    daily_borrow = BORROW_BPS * 1e-4 / (len(dates) / cal_years)
    shares = pd.Series(0.0, index=close.columns)
    cash = capital
    out = []
    for t, d in enumerate(dates):
        p = close.loc[d]
        prev_p = close.iloc[t - 1] if t > 0 else p
        borrow = float((np.maximum(-shares, 0.0) * prev_p).sum()) * daily_borrow
        v_pre = cash + float((shares * p).sum())                       # 调仓前市值
        target = w.loc[d] * v_pre / p
        want = target - shares
        cap_sh = (adv.loc[d] * MPP) / p
        trade = np.sign(want) * np.minimum(np.abs(want), cap_sh)
        fee = float((np.abs(trade) * p).sum()) * FIXED_BPS * 1e-4
        shares = shares + trade
        cash = cash - float((trade * p).sum()) - fee - borrow
        out.append((cash + float((shares * p).sum())) / capital)
    return pd.Series(out, index=dates)


def test_the_paper_broker_matches_the_share_and_cash_ledger(tmp_path):
    from app.core.execution.paper_broker import PaperBroker
    from app.db.position_store import PositionStore
    close, vol, w = _market()
    br = PaperBroker(store=PositionStore(f"sqlite:///{(tmp_path / 'p.db').as_posix()}"),
                     cost_params=_params(), initial_capital=1e6)
    got = br.replay(1, w, close, vol)
    ref = _oracle(close, vol, w)
    assert np.asarray(got) == pytest.approx(np.asarray(ref), rel=1e-9)


def test_the_backtest_engine_matches_the_share_and_cash_ledger():
    from app.core.backtest_engine.backtest_engine import BacktestEngine
    close, vol, w = _market()
    res = BacktestEngine(cost_params=_params(), initial_capital=1e6).run(w, close, vol, w)
    ref = _oracle(close, vol, w)
    assert np.asarray(res.equity_curve) == pytest.approx(np.asarray(ref), rel=1e-9)


def test_the_scenario_actually_exercises_partial_fills_shorts_and_borrow():
    """上面两条的区分力前提：D 确实只部分成交、空头确实产生借券费。"""
    close, vol, w = _market()
    adv = LiquidityConstraint(_params()).compute_adv(vol, close)
    first_target_d = 0.2 * 1e6 / close["D"].iloc[0]
    assert (adv["D"].iloc[0] * MPP) / close["D"].iloc[0] < first_target_d     # 一天成交不完
    assert (w < 0).any().any()
    assert _oracle(close, vol, w).iloc[-1] != _oracle(close, vol.assign(D=1e9), w).iloc[-1]
