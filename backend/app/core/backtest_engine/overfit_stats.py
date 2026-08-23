"""
overfit_stats.py — 回测过拟合概率 PBO（Phase S.3）

PBO（Probability of Backtest Overfitting, Bailey et al. 2015）用 CSCV（组合对称交叉验证）
量化**"筛选流程本身是否导致过拟合"**——与 DSR 互补：DSR 校正**单个**夏普的膨胀，PBO 检验
**从 N 个候选里挑最优**这个动作是否会挑出"样本内好、样本外烂"的策略。

方法（CSCV）
    1. 把 T 期收益切成 S 个不相交连续块（S 偶）。
    2. 枚举 C(S, S/2) 种"一半块作 IS、另一半作 OOS"的组合。
    3. 每种组合：取 IS 夏普最高的策略 n*，看它在 OOS 上的排名 → 相对排名 ω∈(0,1)、logit λ=ln(ω/(1-ω))。
    4. PBO = λ<0 的组合占比（即 IS-最优在 OOS 上低于中位数的频率）。

PBO 越高越糟：**PBO > 0.5** 意味着"挑 IS 最优"在样本外反而更可能低于中位数 → 选择流程在过拟合。
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _blockwise_sharpe(block: np.ndarray) -> np.ndarray:
    """block: (t, N) → 每列（策略）的夏普 (N,)。std=0 记 0。"""
    mu = np.nanmean(block, axis=0)
    sd = np.nanstd(block, axis=0)
    return np.where(sd > 1e-12, mu / sd, 0.0)


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_splits: int = 10,
) -> float:
    """
    Parameters
    ----------
    returns_matrix : (T, N) —— T 期 × N 个候选策略的**每期收益**。
    n_splits       : 切块数 S（自动取偶数，默认 10 → C(10,5)=252 种组合）。

    Returns
    -------
    PBO ∈ [0, 1]（越高越过拟合；>0.5 为警示）。N<2 或 T 过短时抛 ValueError。
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns_matrix 必须是 (T, N) 二维")
    T, N = R.shape
    if N < 2:
        raise ValueError("PBO 需要至少 2 个候选策略")

    S = n_splits - (n_splits % 2)          # 取偶
    blocks = [b for b in np.array_split(np.arange(T), S) if len(b) > 0]
    S = len(blocks)
    if S < 2:
        raise ValueError(f"数据太短，无法切成 {n_splits} 块")

    lam_neg = 0
    total = 0
    all_idx = range(S)
    for is_combo in combinations(all_idx, S // 2):
        is_set = set(is_combo)
        is_rows  = np.concatenate([blocks[i] for i in all_idx if i in is_set])
        oos_rows = np.concatenate([blocks[i] for i in all_idx if i not in is_set])

        is_perf  = _blockwise_sharpe(R[is_rows])
        oos_perf = _blockwise_sharpe(R[oos_rows])

        n_star = int(np.argmax(is_perf))            # IS 最优策略
        rank = int(np.sum(oos_perf <= oos_perf[n_star]))   # 1..N，越大越好
        omega = rank / (N + 1.0)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        lam = np.log(omega / (1.0 - omega))
        total += 1
        if lam < 0:
            lam_neg += 1

    return lam_neg / total if total else float("nan")
