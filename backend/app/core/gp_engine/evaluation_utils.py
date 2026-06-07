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

from typing import Dict

import numpy as np
import pandas as pd


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
    except Exception:
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
        rs = np.argsort(np.argsort(s[mask])).astype(float)
        rr = np.argsort(np.argsort(r[mask])).astype(float)
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
