"""
目标波动的历史回放不能用未来信息（2026-10-25 审计 §4.2 第 2 条，非首轮路径；首轮 RISK_TARGET_VOL_ANN=0）。

审计原话："executable_spec.py:201 使用整个传入样本末尾 lookback 估一个波动标量，再由 risk_gate.py:94 乘到整段历史权重。
对截至当天的最后一个决策可以是当时信息，但对过去各天的回测权重会使用后来的样本。……将来启用要逐决策日只用当时已实现数据，
补'延长未来样本不改变过去目标'的不变性测试。"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.portfolio_manager.risk_gate import RiskLimits
from app.core.portfolio_manager.strategy_contract import generate_targets
from tests.conftest import _make_dataset


def _limits():
    return RiskLimits(max_name_weight=1.0, max_gross=5.0, target_vol_ann=0.10)


def _signal(dataset):
    from app.core.alpha_engine.dsl_executor import Executor
    from app.core.alpha_engine.signal_processor import SignalProcessor
    from app.core.portfolio_manager.executable_spec import signal_config
    return {"1": SignalProcessor(signal_config()).process(Executor().run_expr("rank(ts_delta(close, 5))", dataset))}


def _cut(dataset, n):
    return {k: (v.iloc[:n] if isinstance(v, pd.DataFrame) else v) for k, v in dataset.items()}


def _targets(dataset):
    return generate_targets(_signal(dataset), dataset, aum=1e6, method="equal_weight", limits=_limits(), band=0.0,
                            fixed_combo_weights={"1": 1.0}).gated


def test_extending_the_sample_with_a_wild_future_does_not_change_past_targets():
    full = _make_dataset(n_days=320, n_tickers=12, seed=11)
    # 未来 70 天：价格剧烈波动（波动是之前的 6 倍）—— 只有用了未来信息，过去的目标才会被它改变
    c = full["close"].copy()
    r = c.pct_change().fillna(0.0)
    r.iloc[250:] = r.iloc[250:] * 6.0
    full = {**full, "close": c.iloc[0] * (1 + r).cumprod()}
    past = _targets(_cut(full, 250))
    later = _targets(full).loc[past.index]
    assert np.allclose(past.to_numpy(), later.to_numpy(), atol=1e-12, equal_nan=True), (
        "延长样本改变了过去各天的目标：目标波动缩放用了未来的已实现收益")


def test_the_last_decision_uses_the_same_estimate_as_before():
    """逐日估计只改历史回放：最后一个决策日的缩放仍是"截至当天最近 lookback 个已实现收益"。"""
    from app.core.portfolio_manager.executable_spec import estimate_port_vol, estimate_port_vol_series
    d = _make_dataset(n_days=200, n_tickers=10, seed=3)
    w = _signal(d)["1"].fillna(0.0)
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    s = estimate_port_vol_series(w, d["close"], 60)
    assert s.iloc[-1] == pytest.approx(estimate_port_vol(w, d["close"], 60), rel=1e-12)
    from app.core.portfolio_manager.executable_spec import MIN_VOL_SAMPLE
    # 第 t 行有 t+1 个样本（首日没有持仓记 0，与此前同一口径）：不足 MIN_VOL_SAMPLE 的早期日子不缩放，不借后面的样本补
    assert s.iloc[:MIN_VOL_SAMPLE - 1].isna().all() and np.isfinite(s.iloc[MIN_VOL_SAMPLE - 1])
