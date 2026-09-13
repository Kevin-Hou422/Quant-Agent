"""
test_phase_pm.py — Phase PM 第一批验收

PM.1 多因子合成 + 跨因子净持仓
PM.2 容量：AUM 越大越 binding（gross 下降）
PM.3 资本配置：具体美元/股数账本
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.portfolio_manager.manager import PortfolioManager, PortfolioResult


def _panel(T=120, N=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-01", periods=T)
    cols = [f"S{i}" for i in range(N)]
    ret = rng.normal(0, 0.01, (T, N))
    close = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=idx, columns=cols)
    volume = pd.DataFrame(rng.uniform(1e6, 5e6, (T, N)), index=idx, columns=cols)
    return close, volume, idx, cols


def _signal(idx, cols, vals):
    return pd.DataFrame(vals, index=idx, columns=cols)


def test_combine_and_net_positions():
    close, volume, idx, cols = _panel()
    # 因子A 看多 S0/空 S5；因子B 看空 S0（与 A 在 S0 上对冲）
    sigA = _signal(idx, cols, np.tile([2, 1, 0, 0, -1, -2], (len(idx), 1)).astype(float))
    sigB = _signal(idx, cols, np.tile([-2, 0, 1, 1, 0, 0], (len(idx), 1)).astype(float))
    pm = PortfolioManager(aum=1_000_000, method="equal_weight")
    res = pm.build_book({"A": sigA, "B": sigB}, close, volume)
    assert isinstance(res, PortfolioResult)
    assert list(res.weights.columns) == cols
    # 组合是一个净持仓向量（不是两个孤立账本）
    w_last = res.weights.iloc[-1]
    # S0 被 A(+) 和 B(-) 对冲 → 合成后 |S0 权重| 应小于单看 A 时
    assert abs(w_last["S0"]) < abs(w_last["S4"]) + abs(w_last["S5"]) + 1e-9
    assert res.combo_weights  # 有因子组合权重


def test_capacity_binds_at_large_aum():
    close, volume, idx, cols = _panel(seed=1)
    sig = _signal(idx, cols, np.tile([2, 1, 0, 0, -1, -2], (len(idx), 1)).astype(float))
    small = PortfolioManager(aum=1_000_000, method="equal_weight").build_book({"A": sig}, close, volume)
    large = PortfolioManager(aum=5_000_000_000, method="equal_weight").build_book({"A": sig}, close, volume)
    g_small = float(small.gross_series().iloc[-1])
    g_large = float(large.gross_series().iloc[-1])
    # 小 AUM 基本满仓(gross≈1)，大 AUM 容量 binding → gross 显著下降
    assert g_small > 0.9
    assert g_large < g_small - 0.05, (g_small, g_large)


def test_book_on_gives_concrete_dollar_positions():
    close, volume, idx, cols = _panel(seed=2)
    sig = _signal(idx, cols, np.tile([2, 1, 0, 0, -1, -2], (len(idx), 1)).astype(float))
    pm = PortfolioManager(aum=2_000_000, method="equal_weight")
    res = pm.build_book({"A": sig}, close, volume)
    book = res.book_on(idx[-1])
    assert len(book) > 0
    for tk, pos in book.items():
        assert set(pos) == {"weight", "dollars", "shares"}
        # dollars = weight × AUM
        assert abs(pos["dollars"] - pos["weight"] * 2_000_000) < 1.0
        # shares = dollars / price（符号一致）
        assert np.sign(pos["shares"]) == np.sign(pos["weight"])
    # 总多头 $ ≈ gross/2 × AUM 量级（美元中性 → 多空各半）
    longs = sum(p["dollars"] for p in book.values() if p["dollars"] > 0)
    assert longs > 0


def test_weights_are_net_not_two_books():
    """跨因子净持仓：同一 ticker 只有一个净权重，不是每因子一份。"""
    close, volume, idx, cols = _panel(seed=3)
    sigA = _signal(idx, cols, np.tile([3, 0, 0, 0, 0, -3], (len(idx), 1)).astype(float))
    sigB = _signal(idx, cols, np.tile([-3, 0, 0, 0, 0, 3], (len(idx), 1)).astype(float))
    # A 与 B 完全相反 → 合成后接近全对冲、gross 很小
    res = PortfolioManager(aum=1_000_000, method="equal_weight").build_book(
        {"A": sigA, "B": sigB}, close, volume)
    assert float(res.gross_series().iloc[-1]) < 0.5   # 大幅对冲


# --------------------------------------------------------------------------
# PM.4：组合账本接入每日循环
# --------------------------------------------------------------------------

def test_run_portfolio_trades_one_combined_book(tmp_path):
    from app.db.alpha_store import AlphaStore, AlphaResult
    from app.db.position_store import PositionStore
    from app.core.execution.paper_broker import PaperBroker
    from app.core.monitor.alpha_monitor import AlphaMonitor
    from app.tasks.daily_trading_loop import DailyTradingLoop, PORTFOLIO_BOOK_ID

    url = f"sqlite:///{tmp_path/'pm4.db'}"
    astore = AlphaStore(db_url=url)
    astore.save(AlphaResult(dsl="rank(ts_mean(close,10))", status="paper"))
    astore.save(AlphaResult(dsl="rank((-ts_delta(close,5)))", status="paper"))
    pstore = PositionStore(db_url=url)
    loop = DailyTradingLoop(store=astore, broker=PaperBroker(store=pstore),
                            monitor=AlphaMonitor(astore))

    close, volume, idx, cols = _panel(T=160, N=8, seed=5)
    # H/L 必须用**真实感的随机日内振幅**：固定 ±1% 会让 Corwin-Schultz 估出 ~54bps 的荒谬价差
    # （真实大盘 2~10bps），成本把任何策略碾死 → 边际准入误拒因子。见 DEV_LESSONS。
    _amp = np.abs(np.random.default_rng(5).normal(0, 0.006, close.shape))
    ds = {"close": close, "volume": volume, "open": close, "high": close * (1 + _amp),
          "low": close * (1 - _amp), "vwap": close, "returns": close.pct_change().fillna(0.0)}

    out = loop.run_portfolio(ds, aum=2_000_000)
    # PM.S2 边际准入接线后，合并账本**合法地只含通过边际贡献的子集**（不再必然等于全部 paper 因子）。
    # 本用例的不变量是"交易的是一个合并账本"，不是"用满 2 个因子"。
    assert 1 <= out["n_factors"] <= 2
    assert out["days_processed"] > 50

    # 组合账本记在保留 book id 下（一个账本，不是每因子一份）
    pnl = pstore.pnl_history(PORTFOLIO_BOOK_ID)
    assert len(pnl) > 50
    assert all(np.isfinite(p.equity) and p.equity > 0 for p in pnl)

    # 幂等续跑：再跑一次不新增记账
    n1 = len(pnl)
    loop.run_portfolio(ds, aum=2_000_000)
    assert len(pstore.pnl_history(PORTFOLIO_BOOK_ID)) == n1
