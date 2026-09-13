"""
test_phase_s3.py — Phase S.3 验收：全局 trial 计数 + PBO + t≥3.0 门槛

- TrialLedger 跨实例持久累加
- PBO：噪声 ≈0.5、真持续信号 → 低
- 验证门用全局 trial 数去膨胀；t 门槛拦截低显著性
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.db.trial_ledger import TrialLedger
from app.core.backtest_engine.overfit_stats import probability_of_backtest_overfitting
from app.core.lifecycle.validation_gate import ValidationGate


# ---------------------------------------------------------------- TrialLedger
def test_trial_ledger_accumulates_and_persists(tmp_path):
    url = f"sqlite:///{tmp_path/'t.db'}"
    tl = TrialLedger(db_url=url)
    assert tl.total() == 0
    assert tl.add(100) == 100
    assert tl.add(50) == 150
    assert TrialLedger(db_url=url).total() == 150       # 跨实例持久
    tl.reset()
    assert tl.total() == 0
    assert tl.add(-5) == 0                              # 负数不减


# ---------------------------------------------------------------- PBO
def test_pbo_near_half_for_pure_noise():
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (240, 20))
    pbo = probability_of_backtest_overfitting(R, n_splits=8)
    assert 0.25 <= pbo <= 0.75                          # 纯噪声 ≈ 0.5


def test_pbo_low_for_persistent_signal():
    rng = np.random.default_rng(1)
    R = rng.normal(0, 0.01, (240, 20))
    R[:, 0] += 0.003                                    # 策略0 持续真 alpha
    pbo = probability_of_backtest_overfitting(R, n_splits=8)
    assert pbo < 0.2                                    # 有真信号 → 选择不过拟合


def test_pbo_raises_on_single_strategy():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((100, 1)))


# ---------------------------------------------------------------- gate 集成
def _signal_dataset(T=520, N=20, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    q = np.linspace(-1, 1, N)
    ret = 0.0015 * q[None, :] + rng.normal(0, 0.008, (T, N))
    close = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0),
                         index=pd.bdate_range("2021-01-01", periods=T),
                         columns=[f"A{i:02d}" for i in range(N)])
    r = np.log(close / close.shift(1))
    vol = pd.DataFrame(1e6, index=close.index, columns=close.columns)
    return {"open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": vol, "vwap": close, "returns": r}


def test_gate_uses_global_trial_count():
    """不传 n_trials → 门用全局累计（命中 hermetic 临时库）。"""
    TrialLedger().reset()
    TrialLedger().add(777)
    try:
        gate = ValidationGate(n_splits=3, embargo_days=5)
        res = gate.evaluate("rank(nonexistent_op(close))", _signal_dataset())  # 快速失败
        assert res.n_trials == 777                      # 用了全局累计数
    finally:
        TrialLedger().reset()


def test_gate_t_threshold_blocks_low_significance():
    """t 门槛 3.0 拦截低显著性；隔离 DSR。"""
    TrialLedger().reset()
    gate = ValidationGate(n_splits=3, embargo_days=5, dsr_threshold=0.0, min_tstat=3.0)
    # 噪声数据 → t 很低 → 应被 t 门槛拦下
    rng = np.random.default_rng(2)
    T, N = 400, 15
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (T, N)), axis=0),
                         index=pd.bdate_range("2021-01-01", periods=T),
                         columns=[f"A{i:02d}" for i in range(N)])
    r = np.log(close / close.shift(1))
    ds = {"open": close, "high": close, "low": close, "close": close,
          "volume": pd.DataFrame(1e6, index=close.index, columns=close.columns),
          "vwap": close, "returns": r}
    res = gate.evaluate("rank(ts_mean(returns,10))", ds, n_trials=1)
    assert res.passed is False
    assert any("t=" in r for r in res.reasons)
