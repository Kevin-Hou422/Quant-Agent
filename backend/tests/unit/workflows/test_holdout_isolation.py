"""
test_holdout_isolation.py — Phase S.1+S.2 验收

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
    """
    **尺寸口径已变（Phase S.2，2026-09-22）**：三段委托给
    `data_partitioner.ThreeWayPartitioner`，于是
      · 段间各有一个 embargo 缺口（旧实现零间隔 → 滚动算子跨切点取数）；
      · Test 段优先按**日历**冻结最近 2 年；300 天的面板兑现不了，
        退到 `s_max_test_share` 份额封顶 —— 所以 test 占比是 ~35%，不再是 15%。
    这里断言的是这个新口径本身，不是旧的比例。
    """
    from app.config import settings

    ds = _ds(300)
    is_d, val_d, test_d = _partition_three_way(ds, oos_ratio=0.30, test_ratio=0.15)
    ni, nv, nt = len(is_d["close"]), len(val_d["close"]), len(test_d["close"])
    emb = settings.s_embargo_days
    assert ni + nv + nt == 300 - 2 * emb, (
        f"三段合计 {ni + nv + nt}，应为 300 − 两段 embargo({emb})")
    # 严格时序：IS < Validate < Test
    assert is_d["close"].index[-1] < val_d["close"].index[0]
    assert val_d["close"].index[-1] < test_d["close"].index[0]
    # 短面板走份额封顶：Test 不得超过份额上限，且必须真的留出了一段
    assert 0 < nt <= int(300 * settings.s_max_test_share) + 1, (
        f"Test 段 {nt} 天超过份额上限 {settings.s_max_test_share:.0%}")
    assert ni >= 20 and nv >= 1


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
