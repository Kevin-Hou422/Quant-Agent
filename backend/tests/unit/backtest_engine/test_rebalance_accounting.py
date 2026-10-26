"""
调仓记账：先按价格漂移再调仓（与份额 + 现金账本对照）、两套引擎一致、费用按调仓前市值计。

来源（原测试文件的说明，保留出处）：

外部审查 2026-10-18 F10 / F11。

F10（R6）持仓按权重保存，调仓时直接拿昨日权重与今日目标做差，没有先按价格变动漂移。
     50/50 两只票、A 涨 10%：调仓前应为 52.38 / 47.62，系统仍当 50 / 50，回到 50 / 50 的成本记成 0。
     两套引擎逐位一致证明的是实现一致，不是账算对了 —— 这里用**手算 oracle**（现金 + 股数账本）对。
F11（R13）成本缓存键只看形状、端点、首末行与前 4 列：改中间的 high，键不变，价差从 12.85 → 1584.34 bps。

---

复审 2026-10-19（`review-v6/audit-review/复审报告_2026-10-19.md`）N01–N09 的回归。

每条都从复审独立脚本（`independent_checks.py`）里对应的 residual_* 场景改写：原脚本断言"问题仍在"，
这里断言修复后应有的行为。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.backtest_engine.transaction_cost import CostParams, drift_weights
from tests.unit.backtest_engine.prev_close_bars import (
    prev_close_bars,  # 时钟 c2：只有收盘价的用例配这根 K 线（见模块说明）
)

#: 2026-10-25 审计 R1 之后两套引擎按券商同一合同冻结委托（整股、买单按限价占用资金）。这里的用例验的是
#: 漂移与调仓前市值的记账，把整股取整和限价占资关掉（限价带 0、按分数股）；那两条规则由
#: tests/integration/test_frozen_order_intent_ledger.py 的独立账本验
ISOLATE = dict(limit_band_bps=0.0, whole_shares=False)


def _no_cost():
    return CostParams(fixed_bps=0.0, min_ticket_fee=0.0, spread_bps=0.0, impact_coef=0.0,
                      short_borrow_annual_bps=0.0)


def _panel():
    idx = pd.bdate_range("2025-01-06", periods=3)
    close = pd.DataFrame({"A": [100.0, 110.0, 110.0], "B": [50.0, 50.0, 50.0]}, index=idx)
    # 45% / 45%，留 10% 现金：满仓多头时卖 A 的钱当天不能买 B（券商买入力不预支未成交卖单，审计 R1 之后模拟也一样），
    # 调仓会拆到两天 —— 这里验的是漂移记账，留出现金让两条腿当天都能成交
    weights = pd.DataFrame({"A": [0.45, 0.45, 0.45], "B": [0.45, 0.45, 0.45]}, index=idx)
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


def test_drift_matches_the_hand_computed_numbers():
    w = drift_weights(np.array([0.5, 0.5]), np.array([0.10, 0.0]))
    assert w == pytest.approx([0.55 / 1.05, 0.5 / 1.05])
    assert w[0] == pytest.approx(0.5238095, abs=1e-7)


def test_drift_handles_cash_shorts_and_missing_prices():
    w = drift_weights(np.array([0.3, -0.2, 0.4]), np.array([0.10, 0.05, np.nan]))
    gross = 0.3 * 0.10 + (-0.2) * 0.05
    assert w == pytest.approx(np.array([0.33, -0.21, 0.4]) / (1 + gross))
    assert drift_weights(np.zeros(2), np.array([0.5, -0.5])) == pytest.approx([0.0, 0.0])


def test_the_backtest_engine_trades_what_a_share_ledger_would_trade():
    from app.core.backtest_engine.backtest_engine import BacktestEngine
    idx, close, weights, vol = _panel()
    res = BacktestEngine(cost_params=_no_cost(), initial_capital=1e6, **ISOLATE).run(weights, close, vol, weights,
                                                                         ohlc=prev_close_bars(close))
    oracle = _oracle_trades(close, weights, 1e6)
    # 第二天：A 涨 10% 后漂移成 0.495/1.045、B 0.45/1.045，调回 45/45
    assert oracle[1] == pytest.approx(abs(0.495 / 1.045 - 0.45) + abs(0.45 / 1.045 - 0.45), abs=1e-12)
    got = np.asarray(res.turnover, dtype=float) * 2.0               # 单边换手 × 2 = 成交名义额 / 权益
    # 时钟 c2：第 t 天的决策在 t+1 开盘执行；这根 K 线的开盘 = 前收，成交价与 oracle 在 t 收盘成交相同，
    # 所以 回测第 t+1 天的成交 == oracle 第 t 天的成交
    assert got[0] == pytest.approx(0.0, abs=1e-12), "决策当天就成交了"
    assert got[1] == pytest.approx(oracle[0], abs=1e-9)
    assert got[2] == pytest.approx(oracle[1], abs=1e-9), "回测把调回 50/50 当成了零成交（权重没漂移）"


def test_the_paper_broker_trades_from_the_drifted_weights(tmp_path):
    from app.core.execution.paper_broker import PaperBroker
    from app.db.position_store import PositionStore
    idx, close, weights, vol = _panel()
    br = PaperBroker(store=PositionStore(f"sqlite:///{(tmp_path / 'p.db').as_posix()}"),
                     cost_params=_no_cost(), initial_capital=1e6, **ISOLATE)
    adv = pd.Series(1e15, index=close.columns)
    dv = pd.Series(0.02, index=close.columns)
    bars = prev_close_bars(close)
    # 时钟 c2：第 t 天开盘撮合第 t−1 天收盘冻结的目标；第 0 天只决策
    for t in range(3):
        br.step(1, idx[t], target_w=weights.iloc[t - 1] if t > 0 else None, prices_t=close.iloc[t],
                prices_prev=close.iloc[max(t - 1, 0)], adv_usd=adv, daily_vol=dv, tdays_per_year=252,
                open_t=bars["open"].iloc[t], high_t=bars["high"].iloc[t], low_t=bars["low"].iloc[t])
    traded = sum(abs(f.traded_weight) for f in br.store.fills_on(1, idx[2]))
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
    bt = BacktestEngine(initial_capital=1e6).run(w, close, vol, w, ohlc=prev_close_bars(close))
    br = PaperBroker(store=PositionStore(f"sqlite:///{(tmp_path / 'p.db').as_posix()}"), initial_capital=1e6)
    eq = br.replay(1, w, close, vol, ohlc=prev_close_bars(close))
    assert np.asarray(eq, dtype=float) == pytest.approx(np.asarray(bt.equity_curve, dtype=float), rel=1e-9)


# ---------------------------------------------------------------------------
# N08：费用按调仓前市值算
# ---------------------------------------------------------------------------

def test_an_overnight_gap_after_the_decision_does_not_resize_the_frozen_order(tmp_path):
    """
    此前这条叫 test_the_rebalance_fee_uses_the_pre_trade_value，预期是"开盘按开盘市值调回 50/50、卖 A 买 B、费用 5.00"
    —— 那正是 2026-10-25 审计 R1 指出的错误：券商在 T 收盘按 T 收盘价冻结股数，隔夜跳空不会让委托变大变小。
    T 收盘 A、B 都是 100、各 50 股 = 目标 50%：不需要交易；次日 A 开在 110，冻结的委托（没有）不变，费用 0。
    调仓前市值计费的那一层由下面的回测用例（漂移发生在决策之前）验。
    """
    from app.core.backtest_engine.transaction_cost import CostParams
    from app.core.execution.paper_broker import PaperBroker
    from app.db.position_store import DailyPnL, PositionStore
    ps = PositionStore(f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    ps.record_day(1, "2026-01-05", {"AAA": .5, "BBB": .5}, [], DailyPnL(1, "2026-01-05", 0., 0., 0., 1.))
    broker = PaperBroker(ps, CostParams(fixed_bps=100, min_ticket_fee=0, spread_bps=0, impact_coef=0,
                                        short_borrow_annual_bps=0), 10000)
    bar = pd.Series({"AAA": 110., "BBB": 100.})
    pnl = broker.step(1, "2026-01-06", pd.Series({"AAA": .5, "BBB": .5}), bar,
                      pd.Series({"AAA": 100., "BBB": 100.}), pd.Series({"AAA": 1e9, "BBB": 1e9}),
                      pd.Series({"AAA": .02, "BBB": .02}), open_t=bar, high_t=bar, low_t=bar)
    assert pnl.cost_bps == pytest.approx(0.0)
    assert pnl.equity == pytest.approx((5500 + 5000) / 10000)
    assert [f.traded_weight for f in ps.fills_on(1, "2026-01-06") if abs(f.traded_weight) > 0] == []


def test_the_backtest_engine_charges_the_same_fee(tmp_path):
    from app.core.backtest_engine.backtest_engine import BacktestEngine
    from app.core.backtest_engine.transaction_cost import CostParams
    idx = pd.bdate_range("2026-01-05", periods=3)
    close = pd.DataFrame({"AAA": [100., 110., 110.], "BBB": [100., 100., 100.]}, index=idx)
    w = pd.DataFrame(.45, index=idx, columns=close.columns)
    res = BacktestEngine(cost_params=CostParams(fixed_bps=100, min_ticket_fee=0, spread_bps=0, impact_coef=0,
                                                short_borrow_annual_bps=0),
                         initial_capital=10000, **ISOLATE).run(w, close, pd.DataFrame(1e12, index=idx, columns=close.columns), w,
                                                    ohlc=prev_close_bars(close))
    # 时钟 c2（开盘 = 前收）：第 0 天的决策第 1 天开盘按 100 成交，费用从**现金**扣（N08-b：此前把建仓费用摊进了持仓）
    day2_cost = (res.gross_returns.iloc[2] - res.net_returns.iloc[2]) * res.equity_curve.iloc[1] * 10000
    # 目标 45% / 45%（留现金，理由见 _panel）：第 1 天买入 9,000 × 100 bps = 90 → 现金 910；盘中 A 到 4,950，
    # 净值 10,360。第 2 天开盘调仓前市值 10,360，目标各 4,662 → 卖 A 288、买 B 162，共 450 × 100 bps = 4.50
    assert res.equity_curve.iloc[1] == pytest.approx(1.036)
    assert day2_cost == pytest.approx(4.50)
