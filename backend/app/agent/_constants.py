"""
_constants.py — shared constants for the QuantAgent module.

Centralised here so every sub-module imports from one place; avoids
scattered magic-number duplication.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Thresholds & defaults
# ---------------------------------------------------------------------------

_MAX_CORRECTION_ROUNDS = 2
_OVERFIT_THRESHOLD     = 0.50
_MIN_OOS_SHARPE        = 0.20
_DEFAULT_N_TICKERS     = 20
_DEFAULT_N_DAYS        = 252
_DEFAULT_OOS_RATIO     = 0.30
_DEFAULT_N_TRIALS      = 15

# ---------------------------------------------------------------------------
# Valid field / operator lists (used in LLM prompts to prevent hallucination)
# ---------------------------------------------------------------------------

_VALID_FIELDS = "close, open, high, low, volume, vwap, returns"

_VALID_OPS = (
    "rank, zscore, scale, "
    "ts_mean, ts_std, ts_delta, ts_delay, ts_max, ts_min, ts_rank, "
    "ts_decay_linear, ts_corr, "
    "log, abs, sqrt, sign, signed_power, if_else"
)

# ---------------------------------------------------------------------------
# Fallback DSL keyword→expression mapping (no-LLM path)
# ---------------------------------------------------------------------------

_FALLBACK_DSL_MAP: Dict[str, str] = {
    "volume":     "rank(ts_delta(log(volume), 5))",
    "vwap":       "rank(ts_delta(vwap, 5))",
    "momentum":   "rank(ts_mean(returns, 10))",
    "reversal":   "rank(-ts_delta(close, 1))",
    "volatility": "rank(-ts_std(returns, 20))",
    "price":      "rank(ts_delta(log(close), 5))",
    "default":    "rank(ts_delta(log(close), 5))",
}

# ---------------------------------------------------------------------------
# Structural mutation templates (no-LLM path; {dsl} is the current formula)
# ---------------------------------------------------------------------------

_FALLBACK_MUTATION_TEMPLATES: List[Tuple[str, str]] = [
    ("rank_wrap",    "rank({dsl})"),
    ("vol_filter",   "rank({dsl}) * rank(-ts_std(returns, 20))"),
    ("decay_smooth", "rank(ts_decay_linear(ts_mean(returns, 10), 5))"),
    ("zscore_wrap",  "zscore(ts_delta(log(close), 5))"),
    ("reversal_mix", "rank(-ts_delta(close, 3))"),
    ("momentum_mix", "rank(ts_mean(returns, 5))"),
]
