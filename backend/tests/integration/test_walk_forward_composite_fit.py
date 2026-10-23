"""
滚动前推合成权重的拟合不读未来标签；需要拟合却没给配对收益时拒绝。

来源（原测试文件的说明，保留出处）：

2026-10-22 用户决定 · 批 2：时钟（D14 (a) + C3）与 Validate 预热（D9）。

D14 原话："T 收盘信息完成并过健康门后冻结信号与目标，T+1 按明确交易规则执行，更新时钟版本。同步修改信号、
回测、内部模拟、IC、归档、基准及调度。'计划开盘执行'不等于限价单必然按开盘价成交；预测 IC 与实际持仓收益
分别记录。旧历史保留旧版本，先隔离重算。"
C3 原话："目前虽然要求开盘前存档，评分仍是前收盘到今收盘，包含已经发生的隔夜收益。必须与 D14 一起明确标签
起止、信息截止和归档截止，不能只提前 deadline。"
D9 原话："只用当时已知历史计算特征，只评分 Validate 日期；拟合与标准化不能读取未来评分信息。覆盖 252 日因子、
嵌套窗口、跨界标签、初始持仓及历史不足。"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# D9：Validate 段预热
# ---------------------------------------------------------------------------

def _long_panel(T=800, N=12, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=T)
    cols = [f"S{i}" for i in range(N)]
    c = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.015, (T, N)), 0), idx, cols)
    o = c.shift(1).fillna(c) * (1 + rng.normal(0, 0.003, (T, N)))
    amp = np.abs(rng.normal(0, 0.006, (T, N)))
    return {"open": o, "high": np.maximum(o, c) * (1 + amp), "low": np.minimum(o, c) * (1 - amp),
            "close": c, "volume": pd.DataFrame(1e7, idx, cols), "vwap": c,
            "returns": c.pct_change().fillna(0.0)}


# ---------------------------------------------------------------------------
# 合成权重的拟合（D9 "拟合不能读取未来"；D14 "同步 IC"）
# ---------------------------------------------------------------------------

def _two_factor_signals(ds):
    c = ds["close"]
    return {"mom": c.pct_change(20), "rev": -c.pct_change(5)}


def test_walk_forward_fit_uses_only_labels_realized_before_the_fold():
    """
    第 k 段的合成权重只用 start 收盘时已经实现的标签。配对收益第 start 行 = 开盘(start)→开盘(start+1)，
    start 收盘时还没实现 —— 从这一行起换掉，第 2 段的合成信号必须一点不变。
    两个因子都真有预测力（IC 为正），否则 ic_weighted 退回等权，看不出拟合读没读未来。
    """
    from app.core.portfolio_manager.manager import PortfolioManager
    rng = np.random.default_rng(5)
    T, N = 240, 20
    idx = pd.bdate_range("2022-01-03", periods=T)
    cols = [f"S{i}" for i in range(N)]
    a = pd.DataFrame(rng.normal(size=(T, N)), idx, cols)
    b = pd.DataFrame(rng.normal(size=(T, N)), idx, cols)
    ret = pd.DataFrame(rng.normal(0, 0.01, (T, N)), idx, cols)
    ret.iloc[1:] += 0.004 * a.iloc[:-1].to_numpy() + 0.002 * b.iloc[:-1].to_numpy()   # 第 t+1 行配信号 t
    sig = {"a": a, "b": b}
    pm = PortfolioManager(aum=1e6, method="ic_weighted")
    folds = np.array_split(np.arange(T), 3)
    start2, start3 = idx[folds[1][0]], idx[folds[2][0]]
    _, _, comp_a = pm.walk_forward_composite(sig, a, 3, returns=ret)
    swapped = ret.copy()
    swapped.loc[swapped.index >= start2] = rng.normal(0, 0.01, swapped.loc[swapped.index >= start2].shape)
    swapped.loc[swapped.index >= start2, "S0"] -= 0.05 * a.loc[a.index >= start2, "S0"].shift(1).fillna(0)
    swapped.loc[swapped.index >= start2] -= 0.02 * a.loc[a.index >= start2].shift(1).fillna(0).to_numpy()
    _, _, comp_b = pm.walk_forward_composite(sig, a, 3, returns=swapped)
    fold2 = (comp_a.index >= start2) & (comp_a.index < start3)
    pd.testing.assert_frame_equal(comp_a[fold2], comp_b[fold2])
    # 第 3 段确实用到了换掉的标签（否则上面的相等说明不了什么）
    assert not comp_a[comp_a.index >= start3].equals(comp_b[comp_b.index >= start3])


def test_fitting_combo_weights_without_paired_returns_is_refused():
    """需要拟合却没给 c2 配对收益 → 报错，不拿收盘到收盘悄悄凑。"""
    from app.core.portfolio_manager.manager import PortfolioManager
    ds = _long_panel(T=120)
    with pytest.raises(ValueError, match="配对收益"):
        PortfolioManager(aum=1e6, method="ic_weighted").combined_weights(_two_factor_signals(ds), ds["close"])
