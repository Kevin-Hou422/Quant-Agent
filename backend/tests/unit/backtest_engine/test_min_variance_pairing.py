"""
min_variance 的因子组合收益按信号日期配下一期的已实现标签（2026-10-25 审计 §4.2 第 1 条，非首轮路径）。

审计原话："alpha_combiner.py:234 附近用 signal[t] 直接乘 returns[t] 估因子组合收益；但当前 `clock.paired_returns` 的合同
明确为 signal[t] 应配 returns[t+1]。IC 加权的 helper 已做后一配对，最小方差路径没有同样处理。……统一为按信号日期对齐的已实现
标签/因子收益，并清除训练末尾尚不可知标签；继续保持 min_variance 非缺省。……如要启用，须补独立数据构造，先证明错误再修复。"

独立参照：不调用产品函数，手写"信号 t 的美元中性因子组合 × returns[t+1]"，按同一闭式（收缩协方差 → 逆 × 1 → 只做多 → 归一）求权重。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.backtest_engine.alpha_combiner import AlphaCombiner


def _panel(seed=0, T=200, N=12):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=T)
    cols = [f"S{i}" for i in range(N)]
    ret = pd.DataFrame(rng.normal(0, 0.02, (T, N)), idx, cols)
    # 因子 A 在 t 看到 t+1 的收益（"预测"得好）→ 配对正确时它的因子组合收益又大又稳；
    # 同日配对时它只是噪声。B、C 是噪声因子，C 的量级更大
    sig = {"A": ret.shift(-1).fillna(0.0) + rng.normal(0, 0.01, (T, N)),
           "B": pd.DataFrame(rng.normal(0, 1, (T, N)), idx, cols),
           "C": pd.DataFrame(rng.normal(0, 1, (T, N)), idx, cols) * 3.0}
    return sig, ret


def _oracle(signals, returns):
    vecs = []
    for s in signals.values():
        a = s.to_numpy(dtype=float)
        r = returns.reindex(index=s.index, columns=s.columns).to_numpy(dtype=float)
        w = a - a.mean(axis=1, keepdims=True)
        w = w / np.abs(w).sum(axis=1, keepdims=True)
        vecs.append((w[:-1] * r[1:]).sum(axis=1))           # 信号 t × returns[t+1]；最后一个信号日没有标签
    mat = np.stack(vecs, axis=1)
    k = AlphaCombiner.MIN_VAR_SHRINK
    sample = np.cov(mat.T)
    cov = (1 - k) * sample + k * np.diag(np.diag(sample))
    raw = np.linalg.inv(cov + 1e-6 * np.eye(mat.shape[1])) @ np.ones(mat.shape[1])
    w = np.maximum(raw, 0.0)
    return dict(zip(signals, w / w.sum()))


def test_min_variance_pairs_each_signal_day_with_the_next_realized_return():
    sig, ret = _panel()
    got = AlphaCombiner()._min_variance_weights(sig, ret)          # noqa: SLF001
    want = _oracle(sig, ret)
    assert got == pytest.approx(want, abs=1e-9)


def test_the_last_signal_day_without_a_label_does_not_enter_the_estimate():
    """末尾那个信号日的标签还不可知：往 returns 末行写什么都不能改变权重。"""
    sig, ret = _panel(seed=1)
    a = AlphaCombiner()._min_variance_weights(sig, ret)            # noqa: SLF001
    ret2 = ret.copy()
    ret2.iloc[0] = 5.0                                              # 第 0 行收益只配"信号 −1"，不存在
    b = AlphaCombiner()._min_variance_weights(sig, ret2)           # noqa: SLF001
    assert a == pytest.approx(b, abs=1e-12)


def test_min_variance_is_still_not_the_default_method():
    import inspect
    sig = inspect.signature(AlphaCombiner.optimize_weights)
    assert sig.parameters["method"].default != "min_variance"
