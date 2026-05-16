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



# ---------------------------------------------------------------------------
# Factor-family-specific operator biases
# Applied AFTER metric-driven adjustments as a second layer of guidance.
# Each entry: positive = boost this operator, negative = suppress.
# ---------------------------------------------------------------------------
_FAMILY_WEIGHT_BIASES: dict = {
    # Momentum: needs regime filter to prevent bear-market crashes,
    # smoothing to reduce turnover, volume confirmation to filter noise.
    "momentum": {
        "add_condition":     +0.08,   # regime gate is the #1 improvement
        "add_ts_smoothing":  +0.07,   # reduce flip frequency
        "add_volume_filter": +0.05,   # price + volume = more signal
        "replace_subtree":   -0.05,   # protect momentum core
    },

    # Reversion: needs short windows, volume confirmation.
    # Smoothing must be suppressed — it kills the reversion signal.
    "reversion": {
        "param":             +0.08,   # tune to shorter windows (1–5d)
        "add_volume_filter": +0.07,   # volume confirms price reversal
        "add_condition":     +0.04,   # regime filter (avoid trending markets)
        "add_ts_smoothing":  -0.05,   # smoothing destroys reversion
        "combine_signals":   -0.03,
    },

    # Volatility: cross-sectional normalization is critical;
    # combine with momentum for regime balance.
    "volatility": {
        "wrap_rank":         +0.08,   # must have CS normalization
        "combine_signals":   +0.07,   # pair with momentum for balance
        "add_condition":     +0.05,   # regime conditioning
        "replace_subtree":   -0.05,   # protect volatility core
        "add_ts_smoothing":  -0.03,
    },

    # Liquidity: volume signals are noisy — smooth aggressively.
    "liquidity": {
        "add_ts_smoothing":  +0.07,   # smooth noisy volume
        "wrap_rank":         +0.06,   # normalize cross-sectionally
        "add_condition":     +0.05,   # regime filter
        "add_volume_filter": +0.04,   # volume confirmation on volume signal
        "replace_subtree":   -0.04,
    },

    # Price-volume correlation: tune correlation window, normalize.
    "price_volume_corr": {
        "wrap_rank":         +0.08,   # normalize the corr output
        "param":             +0.06,   # tune correlation window
        "add_condition":     +0.05,   # regime filter
        "combine_signals":   -0.05,   # keep the correlation structure
        "replace_subtree":   -0.04,
    },

    # Trend following: regime conditioning is the single most important fix.
    "trend_following": {
        "add_condition":     +0.10,   # regime filter is essential
        "param":             +0.05,   # tune long windows
        "add_ts_smoothing":  +0.03,
        "replace_subtree":   -0.06,   # protect trend structure
        "combine_signals":   +0.03,
    },

    # Composite: already complex — prioritize simplification.
    "composite": {
        "hoist":             +0.08,   # simplify the complex tree
        "wrap_rank":         +0.05,   # ensure CS normalization
        "combine_signals":   -0.06,   # already composite, don't add more
        "replace_subtree":   +0.04,
    },

    # Quality (entropy/skew/kurtosis): normalize and pair with momentum.
    "quality": {
        "wrap_rank":         +0.07,
        "combine_signals":   +0.05,   # pair with momentum
        "add_condition":     +0.04,
        "param":             +0.04,
    },
}


def mutation_weights_from_metrics(
    sharpe_oos:    float,
    turnover:      float,
    overfit_score: float,
    factor_family: str = "",
) -> dict:
    """
    Return adaptive mutation operator weights based on population diagnostics
    AND the factor family of the alpha being evolved.

    Two-layer adaptation:
      Layer 1 (metric-driven): responds to high turnover / low sharpe / overfitting
      Layer 2 (family-driven): biases toward operators that make financial sense
                               for the specific factor type (momentum, reversion, etc.)

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

    Returns
    -------
    dict with keys for all 11 operators, normalised to sum to 1.
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

    # ── Layer 1: metric-driven adjustments ─────────────────────────────
    if turnover > 2.0:
        w["param"]            += 0.08
        w["hoist"]            += 0.05
        w["crossover"]        -= 0.06
        w["combine_signals"]  -= 0.04
        w["add_ts_smoothing"] -= 0.03

    if sharpe_oos < 0.2:
        w["wrap_rank"]         += 0.06
        w["add_ts_smoothing"]  += 0.06
        w["add_condition"]     += 0.05
        w["combine_signals"]   += 0.05
        w["replace_subtree"]   += 0.04
        w["param"]             -= 0.08
        w["hoist"]             -= 0.06
        w["add_volume_filter"] -= 0.02

    if overfit_score > 0.5:
        w["hoist"]            += 0.08
        w["replace_subtree"]  += 0.06
        w["add_operator"]     += 0.04
        w["crossover"]        -= 0.06
        w["combine_signals"]  -= 0.06
        w["param"]            -= 0.06

    # ── Layer 2: factor-family biases ──────────────────────────────────
    if factor_family and factor_family in _FAMILY_WEIGHT_BIASES:
        for k, delta in _FAMILY_WEIGHT_BIASES[factor_family].items():
            if k in w:
                w[k] += delta

    # Clip negatives, re-normalise
    w = {k: max(0.01, v) for k, v in w.items()}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}
