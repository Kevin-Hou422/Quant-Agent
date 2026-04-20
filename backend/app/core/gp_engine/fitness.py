"""
fitness.py — Multi-objective fitness function for GP alpha evolution.

Spec-mandated formula:
    fitness = sharpe_oos
            - 0.2 * turnover
            - 0.3 * max(0, sharpe_is - sharpe_oos)

The overfitting penalty term makes the fitness self-regularising:
an alpha that fits IS well but degrades on OOS is penalised structurally,
not just post-hoc by an external critic.

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
    Multi-objective fitness combining OOS quality, cost, and overfitting penalty.

    Parameters
    ----------
    sharpe_is    : In-sample annualised Sharpe ratio
    sharpe_oos   : Out-of-sample annualised Sharpe ratio
    turnover     : Annualised portfolio turnover (lower is better)
    max_drawdown : Reserved for future drawdown penalty (not yet applied)

    Returns
    -------
    Scalar fitness value — higher is better.
    """
    overfit_penalty = max(0.0, float(sharpe_is) - float(sharpe_oos))
    fitness = float(sharpe_oos) - 0.2 * float(turnover) - 0.3 * overfit_penalty
    return fitness


def mutation_weights_from_metrics(
    sharpe_oos:    float,
    turnover:      float,
    overfit_score: float,
) -> dict:
    """
    Phase 8: Return adaptive mutation operator weights based on population diagnostics.

    Base distribution:
        crossover : 0.40  (structural combination — most exploratory)
        point     : 0.30  (operator swap — moderate exploration)
        hoist     : 0.15  (tree simplification — anti-overfitting)
        param     : 0.15  (window tweak — exploitation)

    Adjustments:
        high turnover  → boost param (smaller windows) + hoist
        low OOS sharpe → boost point + hoist (explore new operators)
        overfitting    → boost hoist + point (structural simplification)

    Returns
    -------
    dict with keys: "crossover", "point", "hoist", "param"
    """
    w = {"crossover": 0.40, "point": 0.30, "hoist": 0.15, "param": 0.15}

    if turnover > 2.0:
        # High turnover: prefer smaller windows and simpler structures
        w["param"]     += 0.15
        w["hoist"]     += 0.05
        w["crossover"] -= 0.12
        w["point"]     -= 0.08

    if sharpe_oos < 0.2:
        # Low performance: explore new operator/field combinations
        w["point"]     += 0.15
        w["hoist"]     += 0.10
        w["param"]     -= 0.10
        w["crossover"] -= 0.15

    if overfit_score > 0.5:
        # Overfitting: structural simplification over parameter tuning
        w["hoist"]     += 0.15
        w["point"]     += 0.10
        w["param"]     -= 0.15
        w["crossover"] -= 0.10

    # Clip negatives, re-normalise
    w = {k: max(0.01, v) for k, v in w.items()}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}
