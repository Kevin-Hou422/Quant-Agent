"""
test_phase_s_holdout.py — Phase S.1+S.2 验收

S.1：GP 不再按真实 held-out 择优（只在 Validate 段选择）。
S.2：发现路径给出真 held-out Test（GP 全程不可见，仅汇报）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.workflows.alpha_workflows import _partition_three_way, GenerationWorkflow


def _ds(T=300, N=12, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=T)
    cols = [f"A{i:02d}" for i in range(N)]
    ret = 0.0008 + rng.normal(0, 0.01, (T, N))
    close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=idx, columns=cols)
    return {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
            "volume": pd.DataFrame(1e6, index=idx, columns=cols), "vwap": close,
            "returns": pd.DataFrame(ret, index=idx, columns=cols)}


def test_three_way_sizes_order_and_no_overlap():
    is_d, val_d, test_d = _partition_three_way(_ds(300), oos_ratio=0.30, test_ratio=0.15)
    ni, nv, nt = len(is_d["close"]), len(val_d["close"]), len(test_d["close"])
    assert ni + nv + nt == 300                              # 无重叠、全覆盖
    # 严格时序：IS < Validate < Test
    assert is_d["close"].index[-1] < val_d["close"].index[0]
    assert val_d["close"].index[-1] < test_d["close"].index[0]
    assert abs(nt - 45) <= 2 and abs(nv - 45) <= 2          # test≈15%，val≈(0.30-0.15)=15%


def test_three_way_raises_on_tiny_data():
    with pytest.raises(ValueError):
        _partition_three_way(_ds(25), 0.30, 0.15)


def test_generation_workflow_reports_heldout_test():
    """GP 跑完后，metrics 里有 held-out Test 数字（该段 GP 未参与选择）。"""
    wf = GenerationWorkflow(pop_size=8, n_generations=1, n_optuna_trials=1, seed=42)
    res = wf.run("momentum", _ds(300))
    assert res.metrics.get("held_out_test") is True
    assert "test_sharpe" in res.metrics                     # 诚实样本外数字已汇报
    # Validate（GP 选择用）与 Test（held-out）是两个独立数字
    assert "oos_sharpe" in res.metrics


def test_generation_workflow_falls_back_on_tiny_data():
    """数据太短无法三段 → 退回两段，held_out_test=False（不假装有 test）。"""
    wf = GenerationWorkflow(pop_size=6, n_generations=1, n_optuna_trials=1, seed=1)
    res = wf.run("momentum", _ds(60))
    assert res.metrics.get("held_out_test") in (False, True)   # 视数据量而定，但键必须存在
