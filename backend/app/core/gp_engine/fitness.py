"""
fitness.py — Multi-objective fitness function for GP alpha evolution.

Spec-mandated formula (PROMPT 2):
    fitness = sharpe_oos
            - 0.2 * turnover
            - 0.3 * abs(max_drawdown)
            - 0.5 * max(0, sharpe_is - sharpe_oos)

The overfitting penalty makes fitness self-regularising: an alpha that
fits IS well but degrades OOS is penalised structurally.
The drawdown penalty discourages high-risk strategies regardless of Sharpe.

Phase 8 (analysis feedback):
    mutation_weights_from_metrics() adaptively biases operator selection
    based on the current population's dominant failure mode.
"""
from __future__ import annotations


def compute_fitness(
    sharpe_is:    float,
    sharpe_oos:   float,
    turnover:     float,
    max_drawdown: float = 0.0,
) -> float:
    """
    Multi-objective fitness combining OOS quality, cost, drawdown, and overfitting.

    Parameters
    ----------
    sharpe_is    : In-sample annualised Sharpe ratio
    sharpe_oos   : Out-of-sample annualised Sharpe ratio
    turnover     : Annualised portfolio turnover (lower is better)
    max_drawdown : Maximum drawdown value (negative or zero; lower is worse)

    Returns
    -------
    Scalar fitness value — higher is better.
    """
    overfit_penalty = max(0.0, float(sharpe_is) - float(sharpe_oos))
    fitness = (
        float(sharpe_oos)
        - 0.2 * float(turnover)
        - 0.3 * abs(float(max_drawdown))
        - 0.5 * overfit_penalty
    )
    return fitness


def mutation_weights_from_metrics(
    sharpe_oos:    float,
    turnover:      float,
    overfit_score: float,
) -> dict:
    """
    Phase 8: Return adaptive mutation operator weights based on population diagnostics.

    Base distribution (11 operators):
        crossover         : 0.20  (structural combination)
        point             : 0.15  (operator swap)
        hoist             : 0.08  (tree simplification)
        param             : 0.08  (window tweak)
        wrap_rank         : 0.10  (add rank/zscore layer)
        add_ts_smoothing  : 0.10  (add TS smoothing layer)
        add_condition     : 0.08  (add momentum/trend condition)
        add_volume_filter : 0.06  (add volume gate)
        combine_signals   : 0.07  (arithmetic combination of two signals)
        replace_subtree   : 0.05  (swap subtree with generated one)
        add_operator      : 0.03  (wrap with unary/arithmetic layer)

    Adjustments:
        high turnover  → boost param + hoist (smaller windows, simpler trees)
        low OOS sharpe → boost structural mutations (explore new structures)
        overfitting    → boost hoist + structural (simplification + diversity)

    Returns
    -------
    dict with keys for all 11 operators.
    """
    w = {
        "crossover":         0.20,
        "point":             0.15,
        "hoist":             0.08,
        "param":             0.08,
        "wrap_rank":         0.10,
        "add_ts_smoothing":  0.10,
        "add_condition":     0.08,
        "add_volume_filter": 0.06,
        "combine_signals":   0.07,
        "replace_subtree":   0.05,
        "add_operator":      0.03,
    }

    if turnover > 2.0:
        # High turnover: prefer parameter tuning and simplification
        w["param"]            += 0.08
        w["hoist"]            += 0.05
        w["crossover"]        -= 0.06
        w["combine_signals"]  -= 0.04
        w["add_ts_smoothing"] -= 0.03

    if sharpe_oos < 0.2:
        # Low OOS performance: explore new structures aggressively
        w["wrap_rank"]         += 0.06
        w["add_ts_smoothing"]  += 0.06
        w["add_condition"]     += 0.05
        w["combine_signals"]   += 0.05
        w["replace_subtree"]   += 0.04
        w["param"]             -= 0.08
        w["hoist"]             -= 0.06
        w["add_volume_filter"] -= 0.02

    if overfit_score > 0.5:
        # Overfitting: structural simplification + diversification
        w["hoist"]            += 0.08
        w["replace_subtree"]  += 0.06
        w["add_operator"]     += 0.04
        w["crossover"]        -= 0.06
        w["combine_signals"]  -= 0.06
        w["param"]            -= 0.06

    # Clip negatives, re-normalise
    w = {k: max(0.01, v) for k, v in w.items()}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}
