"""
test_phase_visualizer.py — 回测曲线保留且真实（研究用，非伪动画）

保证:
- `plot_backtest(result, prices)` 便捷入口可用
- 图里的净值曲线**逐点等于真实回测 equity_curve**（防"伪动画/装饰曲线"回归）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.backtest_engine.backtest_engine import BacktestEngine
from app.core.backtest_engine import plot_backtest


def _real_backtest(seed=0, T=250, N=8):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=T)
    cols = [f"S{i}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0006, 0.015, (T, N)), 0), idx, cols)
    vol = pd.DataFrame(1e6, idx, cols)
    sig = close.pct_change().rank(axis=1)
    w = sig.sub(sig.mean(1), axis=0)
    w = w.div(w.abs().sum(1), axis=0).fillna(0.0)
    return BacktestEngine().run(w, close, vol, sig), close


def test_backtest_curve_matches_real_equity_not_fake():
    result, prices = _real_backtest()
    fig = plot_backtest(result, prices)
    real = np.asarray(result.equity_curve.values, dtype=float)
    # 图里必须有一条曲线逐点等于真实回测净值（否则就是伪造/装饰曲线）
    ys = [np.asarray(tr.y, dtype=float) for tr in fig.data
          if getattr(tr, "y", None) is not None and len(tr.y) > 10]
    assert any(len(y) == len(real) and np.allclose(y, real) for y in ys), \
        "净值曲线与真实回测数据不匹配——可能是伪动画/装饰曲线"


def test_different_backtests_yield_different_curves():
    # 不同回测 → 不同曲线（伪动画会给出雷同/固定形状）
    r1, p1 = _real_backtest(seed=1)
    r2, p2 = _real_backtest(seed=2)
    assert not np.allclose(r1.equity_curve.values, r2.equity_curve.values)
    plot_backtest(r1, p1); plot_backtest(r2, p2)   # 均能出图
