"""
test_phase_pm_s.py — Phase PM.S1/S2 策略级门 + 边际贡献准入 验收

- StrategyGate：严门加在**组合策略**（分段 OOS + DSR + t），空策略/噪声策略不通过，fail-closed
- marginal_factor_selection：与已选**高相关的冗余因子被拒**（边际≈0）、**能抬升策略 OOS 的因子被纳入**
- 复用同一回测/成本引擎，与单因子门口径一致
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.portfolio_manager import (
    StrategyGate,
    strategy_net_returns,
    marginal_factor_selection,
)


def _predictive_dataset(T=400, N=8, seed=0, edge=0.06):
    """构造一个"信号 f 预测次日收益"的数据集：returns[t] ≈ edge·f[t-1] + 噪声。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=T)
    cols = [f"S{i}" for i in range(N)]
    f = pd.DataFrame(rng.normal(0, 1, (T, N)), index=idx, columns=cols)      # 潜在预测因子
    noise = rng.normal(0, 0.015, (T, N))
    ret = edge * f.shift(1).fillna(0.0).to_numpy() * 0.02 + noise            # 次日收益含 f 的边际
    ret_df = pd.DataFrame(ret, index=idx, columns=cols)
    close = 100 * (1 + ret_df).cumprod()
    vol = pd.DataFrame(rng.uniform(1e6, 5e6, (T, N)), index=idx, columns=cols)
    # H/L 用**真实感随机振幅**（~0.6%）：固定 ±1% 会让 Corwin-Schultz 估出 ~54bps 荒谬价差，
    # 成本足以把有真实 alpha 的策略压成负 Sharpe → 边际准入误拒。见 DEV_LESSONS。
    amp = np.abs(rng.normal(0, 0.006, (T, N)))
    ds = {"open": close, "high": close * (1 + amp), "low": close * (1 - amp),
          "close": close, "vwap": close, "volume": vol, "returns": ret_df}
    return ds, f


# --------------------------------------------------------------------------
# strategy_net_returns
# --------------------------------------------------------------------------

def test_strategy_net_returns_shape_and_engine():
    ds, f = _predictive_dataset()
    rets, composite = strategy_net_returns({"f": f}, ds, aum=10_000.0)
    assert isinstance(rets, pd.Series) and len(rets) > 100
    assert composite.shape[1] == ds["close"].shape[1]


# --------------------------------------------------------------------------
# PM.S1 StrategyGate
# --------------------------------------------------------------------------

def test_strategy_gate_empty_fails():
    ds, _ = _predictive_dataset()
    res = StrategyGate().evaluate({}, ds)
    assert res.passed is False
    assert any("空" in r for r in res.reasons)


def test_strategy_gate_result_is_consistent_and_serializable():
    ds, f = _predictive_dataset()
    res = StrategyGate(aum=10_000.0).evaluate({"f": f}, ds)
    # 一致性不变量：passed 当且仅当无 reasons
    assert res.passed == (len(res.reasons) == 0)
    assert res.n_factors == 1 and res.n_trials >= 1
    d = res.to_dict()
    assert set(d) >= {"passed", "sharpe", "deflated_sharpe", "t_stat", "pct_seg_positive"}


def test_strategy_gate_computes_pbo_for_multi_factor():
    # ≥2 因子 → PBO(CSCV,S.3)被算出且落在 [0,1]；接进策略门
    ds, f = _predictive_dataset()
    _, g = _predictive_dataset(seed=321)
    res = StrategyGate(aum=10_000.0).evaluate({"f": f, "g": g.reindex_like(f)}, ds)
    assert res.pbo is not None and 0.0 <= res.pbo <= 1.0
    assert "pbo" in res.to_dict()


def test_strategy_gate_pbo_none_for_single_factor():
    ds, f = _predictive_dataset()
    res = StrategyGate(aum=10_000.0).evaluate({"f": f}, ds)
    assert res.pbo is None            # 单因子无法算 PBO → 不作为门


def test_strategy_gate_noise_strategy_rejected():
    # 纯噪声因子（无边际预测力）→ 分段 OOS/DSR 应判不通过（fail-closed 语义）
    rng = np.random.default_rng(7)
    T, N = 400, 8
    idx = pd.bdate_range("2021-01-01", periods=T)
    cols = [f"S{i}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.015, (T, N)), axis=0),
                         index=idx, columns=cols)
    vol = pd.DataFrame(rng.uniform(1e6, 5e6, (T, N)), index=idx, columns=cols)
    ds = {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
          "vwap": close, "volume": vol, "returns": close.pct_change().fillna(0.0)}
    noise = pd.DataFrame(rng.normal(0, 1, (T, N)), index=idx, columns=cols)
    res = StrategyGate(aum=10_000.0).evaluate({"noise": noise}, ds)
    assert res.passed is False and len(res.reasons) > 0


# --------------------------------------------------------------------------
# PM.S2 边际贡献准入
# --------------------------------------------------------------------------

def test_marginal_admits_predictive_factor_from_empty():
    ds, f = _predictive_dataset(edge=0.10)
    out = marginal_factor_selection({"f": f}, ds, aum=10_000.0, min_improve=0.01)
    assert "f" in out.selected                       # 有边际的因子从空集被纳入
    assert out.steps and out.steps[0].admitted


def test_marginal_rejects_redundant_duplicate():
    ds, f = _predictive_dataset(edge=0.10)
    fdup = f.copy()                                   # 与已选完全相同 → 边际≈0
    out = marginal_factor_selection(
        {"fdup": fdup}, ds, aum=10_000.0, min_improve=0.05, seed_signals={"f": f},
    )
    assert "fdup" not in out.selected                 # 冗余因子被拒
    assert any(s.factor == "fdup" and not s.admitted for s in out.steps)


def test_marginal_prefers_diversifier_over_redundant():
    ds, f = _predictive_dataset(edge=0.10, seed=1)
    _, g = _predictive_dataset(edge=0.10, seed=999)   # 独立的另一个预测因子（低相关）
    fdup = f.copy()
    out = marginal_factor_selection(
        {"fdup": fdup, "g": g.reindex_like(f)}, ds, aum=10_000.0,
        min_improve=0.01, seed_signals={"f": f},
    )
    # 低相关的 g 更可能被纳入，冗余 fdup 不被纳入
    assert "fdup" not in out.selected
    d = out.to_dict()
    assert "selected" in d and "steps" in d
