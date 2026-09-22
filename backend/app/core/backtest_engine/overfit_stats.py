"""
overfit_stats.py — 回测过拟合概率 PBO / CPCV（Phase S.3）

PBO（Probability of Backtest Overfitting, Bailey et al. 2015）用 CSCV（组合对称交叉验证）
量化**"筛选流程本身是否导致过拟合"**——与 DSR 互补：DSR 校正**单个**夏普的膨胀，PBO 检验
**从 N 个候选里挑最优**这个动作是否会挑出"样本内好、样本外烂"的策略。

方法（CSCV）
    1. 把 T 期收益切成 S 个不相交连续块（S 偶）。
    2. 枚举 C(S, S/2) 种"一半块作 IS、另一半作 OOS"的组合。
    3. 每种组合：取 IS 夏普最高的策略 n*，看它在 OOS 上的排名 → 相对排名 ω∈(0,1)、logit λ=ln(ω/(1-ω))。
    4. PBO = λ<0 的组合占比（即 IS-最优在 OOS 上低于中位数的频率）。

PBO 越高越糟：**PBO > 0.5** 意味着"挑 IS 最优"在样本外反而更可能低于中位数 → 选择流程在过拟合。

CPCV（组合式 purged 交叉验证，López de Prado AFML §12）
-------------------------------------------------------
CSCV 的块是**直接相邻**的：IS 块的最后一天和 OOS 块的第一天之间没有任何间隔。
日频面板上，滚动窗口算子（ts_mean(20)…）会让这两天共享原始 bar，于是"样本外"
块的头部其实被样本内摸到过 —— **PBO 被系统性低估**（看起来比真实更不过拟合）。

CPCV 在每个 OOS 块的**两侧**各 purge 掉 `embargo` 个样本再组 IS，并由
C(N, k) 种组合生成 φ = C(N,k)·k/N 条**互不相同的回测路径** ——
既修掉边界泄漏，又给出"同一策略在多条历史路径上的表现分布"，
而不是单条路径上的一个点估计。
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


# ---------------------------------------------------------------------------
# CPCV — 组合式 purged 交叉验证（Phase S.3 收尾）
# ---------------------------------------------------------------------------

def combinatorial_purged_splits(
    n_obs: int,
    n_groups: int = 6,
    k: int = 2,
    embargo: int = 20,
):
    """
    生成 CPCV 的 (train_pos, test_pos) 位置索引对。

    Parameters
    ----------
    n_obs    : 样本数 T。
    n_groups : 连续分组数 N（AFML 推荐 6）。
    k        : 每次留出的组数（推荐 2 → C(6,2)=15 组，φ=5 条路径）。
    embargo  : 每个留出组两侧各 purge 掉的样本数。

    Returns
    -------
    list[(np.ndarray, np.ndarray)] —— 训练位置、测试位置。
    训练集被 purge 到空的组合会被**跳过**（不是返回空训练集）。
    """
    if n_groups < 2:
        raise ValueError(f"n_groups 至少为 2，当前={n_groups}")
    if not (1 <= k < n_groups):
        raise ValueError(f"k 必须在 [1, n_groups)，当前 k={k}, n_groups={n_groups}")
    if embargo < 0:
        raise ValueError(f"embargo 不能为负，当前={embargo}")
    if n_obs < n_groups:
        raise ValueError(f"样本数 {n_obs} 少于分组数 {n_groups}")

    groups = [g for g in np.array_split(np.arange(n_obs), n_groups) if len(g) > 0]
    n_groups = len(groups)
    if k >= n_groups:
        raise ValueError(f"有效分组只有 {n_groups} 组，无法留出 {k} 组")

    out = []
    for combo in combinations(range(n_groups), k):
        test_pos = np.concatenate([groups[i] for i in combo])
        blocked = np.zeros(n_obs, dtype=bool)
        for i in combo:
            lo, hi = int(groups[i][0]), int(groups[i][-1])
            blocked[max(0, lo - embargo): min(n_obs, hi + embargo + 1)] = True
        train_pos = np.flatnonzero(~blocked)
        if len(train_pos) == 0:
            continue
        out.append((train_pos, np.sort(test_pos)))
    if not out:
        raise ValueError("CPCV 未能生成任何有效组合（embargo 过大？）")
    return out


def cpcv_pbo(
    returns_matrix: np.ndarray,
    n_groups: int = 6,
    k: int = 2,
    embargo: int = 20,
) -> float:
    """
    用 CPCV 的 purged 组合算 PBO —— 与 `probability_of_backtest_overfitting`
    同一个统计量，区别只在 IS 块**被 purge 过**，不再隔着切点摸到 OOS 的头部。

    因此两者可以直接比较：CPCV-PBO 通常 **≥** CSCV-PBO，差额就是边界泄漏
    此前替选择流程掩盖掉的那部分过拟合。

    Returns
    -------
    PBO ∈ [0, 1]。N<2 或组合不足时抛 ValueError（**不返回 0 冒充"不过拟合"**）。
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns_matrix 必须是 (T, N) 二维")
    T, N = R.shape
    if N < 2:
        raise ValueError("PBO 需要至少 2 个候选策略")

    lam_neg = 0
    total = 0
    for train_pos, test_pos in combinatorial_purged_splits(T, n_groups, k, embargo):
        is_perf = _blockwise_sharpe(R[train_pos])
        oos_perf = _blockwise_sharpe(R[test_pos])
        n_star = int(np.argmax(is_perf))
        rank = int(np.sum(oos_perf <= oos_perf[n_star]))
        omega = min(max(rank / (N + 1.0), 1e-6), 1 - 1e-6)
        total += 1
        if np.log(omega / (1.0 - omega)) < 0:
            lam_neg += 1

    if total == 0:
        raise ValueError("CPCV 没有可用组合")
    return lam_neg / total


def cpcv_path_sharpes(
    returns: np.ndarray,
    n_groups: int = 6,
    k: int = 2,
    embargo: int = 20,
) -> np.ndarray:
    """
    单个策略在 CPCV 各留出组合上的**样本外夏普分布**（每期口径，未年化）。

    用途：把"一个回测 Sharpe"换成"φ 条路径的分布"。分布的下尾比点估计
    诚实得多 —— 一条路径上的 1.5 可能来自另外几条路径上的 -0.3。
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    splits = combinatorial_purged_splits(len(r), n_groups, k, embargo)
    return np.array([_blockwise_sharpe(r[test_pos].reshape(-1, 1))[0]
                     for _, test_pos in splits], dtype=float)
