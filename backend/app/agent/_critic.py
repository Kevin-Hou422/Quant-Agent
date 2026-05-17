"""
_critic.py — OverfitCritic: structured anti-overfitting gate.

Returns a CriticResult dataclass that contains:
  - passed              : bool gate
  - failure_mode        : canonical failure label
  - severity            : "ok" | "minor" | "moderate" | "critical"
  - recommended_mutation: GP mutation key to apply in the correction loop
  - reason              : human-readable explanation
  - metrics_snapshot    : echo of key metric values for logging

Failure mode → recommended mutation mapping (Weakness-6 fix):

  no_signal          → replace_subtree  (complete structural replacement)
  severe_overfitting → hoist            (simplify tree depth to remove IS noise)
  high_turnover      → add_ts_smoothing (add smoothing layer to reduce signal flips)
  high_drawdown      → add_condition    (add regime filter to cut tail risk)
  mild_overfitting   → wrap_rank        (add CS normalization to remove IS bias)
  weak_signal        → point            (swap operator to explore alternatives)

Passes:
  oos_sharpe >= _MIN_OOS_SHARPE  AND  overfitting_score <= _OVERFIT_THRESHOLD
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from app.agent._constants import _MIN_OOS_SHARPE, _OVERFIT_THRESHOLD


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_SEVERE_OVERFIT_THRESHOLD = 0.60   # IS Sharpe degrades > 60 %
_HIGH_TURNOVER_THRESHOLD  = 3.0    # annualized turnover
_HIGH_DRAWDOWN_THRESHOLD  = 0.30   # |max drawdown|  (absolute value)
_WEAK_SIGNAL_THRESHOLD    = 0.35   # OOS Sharpe below which signal is "weak"
_NO_SIGNAL_THRESHOLD      = 0.05   # OOS Sharpe near zero

# Failure mode → GP mutation operator key
_FAILURE_MUTATIONS: Dict[str, str] = {
    "no_signal":          "replace_subtree",
    "severe_overfitting": "hoist",
    "high_turnover":      "add_ts_smoothing",
    "high_drawdown":      "add_condition",
    "mild_overfitting":   "wrap_rank",
    "weak_signal":        "point",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CriticResult:
    """
    Structured output of OverfitCritic.check().

    Attributes
    ----------
    passed               : True iff the alpha passes all quality gates.
    failure_mode         : Canonical failure label (empty string when passed).
    severity             : "ok" | "minor" | "moderate" | "critical"
    recommended_mutation : GP mutation key — pass directly to tool_mutate_ast
                           as ``mutation_target`` for targeted correction.
    reason               : Human-readable explanation for logs / UI.
    metrics_snapshot     : Key metric values echoed for traceability.
    """
    passed:               bool
    failure_mode:         str
    severity:             str
    recommended_mutation: str
    reason:               str
    metrics_snapshot:     Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        """Backward-compat: allows ``passed, reason = critic.check(m)``."""
        yield self.passed
        yield self.reason


# ---------------------------------------------------------------------------
# OverfitCritic
# ---------------------------------------------------------------------------

class OverfitCritic:
    """
    Stateless validator that detects six failure modes (priority-ordered)
    and returns a structured CriticResult with a targeted correction action.

    Priority order (checked top-to-bottom; first match wins):
      1. no_signal          — OOS Sharpe near zero
      2. severe_overfitting — IS/OOS degradation > 60 %
      3. high_turnover      — annualized turnover > 3×
      4. high_drawdown      — |max_drawdown| > 30 %
      5. mild_overfitting   — IS/OOS degradation > threshold OR OOS below min
      6. weak_signal        — OOS Sharpe below moderate threshold
    """

    @staticmethod
    def check(metrics: Dict[str, Any]) -> CriticResult:
        """
        Evaluate ``metrics`` and return a CriticResult.

        Parameters
        ----------
        metrics : dict with any subset of keys:
            oos_sharpe, is_sharpe, overfitting_score, is_overfit,
            is_turnover, turnover, max_drawdown

        Returns
        -------
        CriticResult
        """
        def _f(key: str, default: float = 0.0) -> float:
            v = metrics.get(key, default)
            try:
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        oos_sharpe    = _f("oos_sharpe",        0.0)
        is_sharpe     = _f("is_sharpe",         0.0)
        overfit_score = _f("overfitting_score", 0.0)
        is_overfit    = bool(metrics.get("is_overfit", False))
        # Support both "is_turnover" (from backtest) and "turnover" (from GP)
        turnover      = _f("is_turnover", _f("turnover", 0.0))
        max_dd        = abs(_f("max_drawdown", 0.0))

        snap = {
            "oos_sharpe":    round(oos_sharpe,    3),
            "is_sharpe":     round(is_sharpe,     3),
            "overfit_score": round(overfit_score, 3),
            "turnover":      round(turnover,      2),
            "max_drawdown":  round(-max_dd,       3),
        }

        # ── 1. No signal ────────────────────────────────────────────────
        if oos_sharpe < _NO_SIGNAL_THRESHOLD:
            return CriticResult(
                passed               = False,
                failure_mode         = "no_signal",
                severity             = "critical",
                recommended_mutation = _FAILURE_MUTATIONS["no_signal"],
                reason               = (
                    f"OOS Sharpe {oos_sharpe:.3f} is near zero — "
                    "the signal has no predictive power. "
                    "Replace the core signal structure (replace_subtree)."
                ),
                metrics_snapshot     = snap,
            )

        # ── 2. Severe overfitting ───────────────────────────────────────
        if (is_overfit or overfit_score > _SEVERE_OVERFIT_THRESHOLD) and is_sharpe > 0.5:
            return CriticResult(
                passed               = False,
                failure_mode         = "severe_overfitting",
                severity             = "critical",
                recommended_mutation = _FAILURE_MUTATIONS["severe_overfitting"],
                reason               = (
                    f"Severe overfitting — IS Sharpe {is_sharpe:.3f} degrades "
                    f"{overfit_score * 100:.1f}% OOS. "
                    "Simplify tree with hoist_mutation."
                ),
                metrics_snapshot     = snap,
            )

        # ── 3. High turnover ────────────────────────────────────────────
        if turnover > _HIGH_TURNOVER_THRESHOLD:
            return CriticResult(
                passed               = False,
                failure_mode         = "high_turnover",
                severity             = "critical" if turnover > 5.0 else "moderate",
                recommended_mutation = _FAILURE_MUTATIONS["high_turnover"],
                reason               = (
                    f"Annualized turnover {turnover:.1f}× is unsustainably high. "
                    "At 30bps/trade this erases gross alpha. "
                    "Add ts_mean/ts_decay_linear smoothing (add_ts_smoothing)."
                ),
                metrics_snapshot     = snap,
            )

        # ── 4. High drawdown ────────────────────────────────────────────
        if max_dd > _HIGH_DRAWDOWN_THRESHOLD:
            return CriticResult(
                passed               = False,
                failure_mode         = "high_drawdown",
                severity             = "moderate",
                recommended_mutation = _FAILURE_MUTATIONS["high_drawdown"],
                reason               = (
                    f"Max drawdown {-max_dd:.1%} is too severe. "
                    "Add a market-state condition gate (add_condition) "
                    "to avoid bear-market periods."
                ),
                metrics_snapshot     = snap,
            )

        # ── 5. Mild overfitting / low OOS Sharpe ───────────────────────
        if is_overfit or overfit_score > _OVERFIT_THRESHOLD or oos_sharpe < _MIN_OOS_SHARPE:
            return CriticResult(
                passed               = False,
                failure_mode         = "mild_overfitting",
                severity             = "moderate",
                recommended_mutation = _FAILURE_MUTATIONS["mild_overfitting"],
                reason               = (
                    f"IS/OOS gap (overfit={overfit_score:.2f}, "
                    f"OOS Sharpe={oos_sharpe:.3f}) indicates mild overfitting. "
                    "Add cross-sectional normalization (wrap_rank) to remove IS bias."
                ),
                metrics_snapshot     = snap,
            )

        # ── 6. Weak signal ──────────────────────────────────────────────
        if oos_sharpe < _WEAK_SIGNAL_THRESHOLD:
            return CriticResult(
                passed               = False,
                failure_mode         = "weak_signal",
                severity             = "minor",
                recommended_mutation = _FAILURE_MUTATIONS["weak_signal"],
                reason               = (
                    f"OOS Sharpe {oos_sharpe:.3f} is marginal. "
                    "Swap to a different operator class (point_mutation) "
                    "to explore alternative signal formulations."
                ),
                metrics_snapshot     = snap,
            )

        # ── Pass ─────────────────────────────────────────────────────────
        return CriticResult(
            passed               = True,
            failure_mode         = "",
            severity             = "ok",
            recommended_mutation = "",
            reason               = (
                f"Passed — OOS Sharpe {oos_sharpe:.3f}, "
                f"overfit {overfit_score:.2f}, turnover {turnover:.1f}×"
            ),
            metrics_snapshot     = snap,
        )
