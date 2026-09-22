"""
evaluation_utils.py — Shared fast IC-IR evaluation utilities.

Provides a single entry point for the quick (no-backtest-engine) IC-IR
calculation that was previously duplicated in:
  - app/agent/alpha_agent._quick_eval()
  - inline in test helpers

This is distinct from the FULL IS+OOS backtest path in:
  - alpha_workflows._quick_metrics()      → uses RealisticBacktester
  - population_evolver._quick_metrics()   → wraps _evaluate_one()

Use quick_ic_eval() only for fast initial screening (no transaction cost,
no signal processing pipeline). Use RealisticBacktester for final evaluation.
"""

from __future__ import annotations

import logging

from typing import Dict

import numpy as np
from ..alpha_engine.fast_ops import average_ranks_1d
import pandas as pd

logger = logging.getLogger(__name__)


def quick_ic_eval(
    dsl: str,
    dataset: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """
    Fast cross-sectional IC-IR evaluation for a DSL expression.

    Does NOT run BacktestEngine or SignalProcessor — purely measures the
    Spearman rank correlation between the signal and 1-day-forward returns.

    Parameters
    ----------
    dsl     : Alpha DSL expression string
    dataset : dict[field → (T×N) pd.DataFrame]

    Returns
    -------
    dict with keys: ic_ir, ann_turnover, sharpe (= ic_ir)
    On failure returns: {"ic_ir": 0.0, "ann_turnover": 99.0, "sharpe": -1.0}
    """
    from ..alpha_engine.dsl_executor import Executor

    try:
        signal_df = Executor().run_expr(dsl, dataset)
    except Exception as exc:
        # 这是**惩罚哨兵**（让求值失败的候选排到最后），不是真实指标。
        # 但如果不留痕，"这个因子很差"与"这个因子根本没跑起来"就无法区分。
        logger.debug("[quick_eval] 求值失败，返回惩罚哨兵指标: %s | %s", dsl[:60], exc)
        return {"ic_ir": 0.0, "ann_turnover": 99.0, "sharpe": -1.0}

    close = dataset.get("close")
    if close is None:
        return {"ic_ir": 0.0, "ann_turnover": 99.0, "sharpe": -1.0}

    sig = signal_df.to_numpy(dtype=float)
    cls = close.to_numpy(dtype=float)

    # 1-day forward returns
    fwd = np.full_like(cls, np.nan)
    fwd[:-1] = (cls[1:] - cls[:-1]) / np.where(cls[:-1] == 0, np.nan, cls[:-1])

    # Cross-sectional Spearman Rank IC via vectorised double-argsort
    T = min(sig.shape[0], fwd.shape[0])
    ics: list[float] = []
    for t in range(T - 1):
        s, r = sig[t], fwd[t]
        mask = ~(np.isnan(s) | np.isnan(r))
        n_valid = int(mask.sum())
        if n_valid < 5:
            continue
        # Vectorised rank correlation (avoids scipy import per call)
        # 缺陷 C-1：必须用**平均秩**（并列取均值），不是 argsort 两次的序数名次 ——
        # 后者把截面恒定的零信息信号排成 [0,1,2,...]，算出按列顺序的伪 IC。
        rs = average_ranks_1d(s[mask])
        rr = average_ranks_1d(r[mask])
        rs -= rs.mean()
        rr -= rr.mean()
        denom = np.sqrt((rs ** 2).sum() * (rr ** 2).sum())
        if denom > 0:
            ics.append(float(np.dot(rs, rr) / denom))

    if not ics:
        return {"ic_ir": 0.0, "ann_turnover": 99.0, "sharpe": -1.0}

    ic_arr = np.array(ics)
    ic_ir  = float(np.mean(ic_arr) / (np.std(ic_arr) + 1e-9))

    # Annualised turnover proxy: mean daily L1 change of pct-ranked signal
    ranks = pd.DataFrame(sig).rank(axis=1, pct=True).to_numpy()
    turn  = float(np.nanmean(np.abs(np.diff(ranks, axis=0)))) * 252

    return {"ic_ir": ic_ir, "ann_turnover": turn, "sharpe": ic_ir}


# ---------------------------------------------------------------------------
# Purged K-fold CV 的适应度口径（Phase S.1 收尾）
# ---------------------------------------------------------------------------

def purged_cv_sharpe(
    net_returns: "pd.Series",
    n_splits: int = 5,
    embargo_days: int = 20,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    把一条**已经算好**的日净收益序列，按 purged K 折切成 K 个留出块，
    返回各块年化夏普的均值/标准差/最差值（Phase S.1）。

    这买到了什么
    ------------
    适应度不再取决于"最后那一段恰好是什么行情"：单段 Validate 的结论会随切点
    大幅漂移，K 折让**每个样本都当过一次留出**，方差与段位偏倚都降下来。
    块两侧各 purge `embargo_days` 个样本，避免滚动算子跨块借数。

    这**没有**买到什么（别把它当更强的保证）
    -------------------------------------------
    GP 的"训练"是搜索本身，它看得见整个 IS —— 这里没有逐折重新拟合参数，
    所以这不是"样本外"，而是**样本内的稳健性度量**。真正的样本外仍然只有
    三段切割里那段 GP 全程看不见的 Test（S.2）。

    Returns
    -------
    {"mean", "std", "min", "n_folds"}；折数不足时 n_folds=0 且其余为 0.0
    （**不抛异常**：适应度路径上抛异常会让候选被静默丢弃，见审计 #9）。
    """
    from ..data_engine.data_partitioner import PurgedKFold

    empty = {"mean": 0.0, "std": 0.0, "min": 0.0, "n_folds": 0.0}
    s = pd.Series(net_returns).dropna()
    if len(s) < 30:
        return empty
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.RangeIndex(len(s))
        dates = pd.DatetimeIndex(pd.date_range("2000-01-03", periods=len(s), freq="B"))
    else:
        dates = pd.DatetimeIndex(s.index)

    try:
        folds = PurgedKFold(n_splits=n_splits, embargo_days=embargo_days).split(dates)
    except ValueError as exc:
        logger.debug("purged CV 折数不足，退回单段口径: %s", exc)
        return empty

    vals = s.to_numpy(dtype=float)
    pos = {d: i for i, d in enumerate(dates)}
    sharpes: list[float] = []
    for f in folds:
        idx = [pos[d] for d in f.test_idx if d in pos]
        if len(idx) < 5:
            continue
        block = vals[idx]
        sd = float(np.nanstd(block, ddof=1)) if len(block) > 1 else 0.0
        if sd <= 1e-12:
            continue
        sharpes.append(float(np.nanmean(block) / sd) * float(np.sqrt(periods_per_year)))

    if not sharpes:
        return empty
    arr = np.array(sharpes, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min":  float(np.min(arr)),
        "n_folds": float(len(arr)),
    }
