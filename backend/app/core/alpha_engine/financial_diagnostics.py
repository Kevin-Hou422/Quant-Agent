"""
financial_diagnostics.py — Translate backtest metrics into financial diagnoses.

Given quantitative metrics (Sharpe, IC, turnover, overfitting), produces:
  - Primary failure diagnosis with financial reasoning
  - Ranked list of concrete DSL-level improvement suggestions
  - Regime/market-environment insight

Usage::

    from app.core.alpha_engine.financial_diagnostics import FinancialDiagnostics
    diag = FinancialDiagnostics().diagnose(
        dsl     = "rank(ts_delta(close, 5))",
        metrics = {"oos_sharpe": 0.3, "is_sharpe": 1.2, "turnover": 3.5,
                   "mean_ic": 0.025, "ic_ir": 0.25, "max_drawdown": -0.18},
    )
    print(diag.primary_issue)       # "high_turnover"
    print(diag.diagnosis)           # "Signal changes sign too frequently..."
    for s in diag.suggestions[:2]:
        print(s["action"], s["dsl_patch"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Suggestion:
    """A single actionable DSL improvement."""
    action:      str           # short action label
    priority:    int           # 1 = highest
    dsl_patch:   str           # concrete DSL transformation to try
    reason:      str           # why this should help
    finance_why: str           # financial theory behind the suggestion

    def to_dict(self) -> dict:
        return {
            "action":      self.action,
            "priority":    self.priority,
            "dsl_patch":   self.dsl_patch,
            "reason":      self.reason,
            "finance_why": self.finance_why,
        }


@dataclass
class FactorDiagnosis:
    """Full financial diagnosis for one alpha factor."""
    primary_issue:   str               # key failure mode
    diagnosis:       str               # explanation in financial terms
    severity:        str               # "critical" | "moderate" | "minor" | "healthy"
    suggestions:     List[Suggestion]  # ranked improvement actions
    regime_insight:  Optional[str]     # market condition note
    metrics_summary: Dict[str, Any]    # key metrics echoed back

    def to_dict(self) -> dict:
        return {
            "primary_issue":   self.primary_issue,
            "diagnosis":       self.diagnosis,
            "severity":        self.severity,
            "suggestions":     [s.to_dict() for s in self.suggestions],
            "regime_insight":  self.regime_insight,
            "metrics_summary": self.metrics_summary,
        }

    def summary(self) -> str:
        lines = [
            f"Primary Issue : {self.primary_issue} [{self.severity.upper()}]",
            f"Diagnosis     : {self.diagnosis}",
        ]
        if self.regime_insight:
            lines.append(f"Regime Note   : {self.regime_insight}")
        for i, s in enumerate(self.suggestions[:3], 1):
            lines.append(f"Suggestion {i}  : [{s.action}] {s.reason}")
            lines.append(f"  DSL patch   : {s.dsl_patch}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Diagnostics engine
# ---------------------------------------------------------------------------

class FinancialDiagnostics:
    """
    Analyze backtest metrics and return actionable financial improvement suggestions.

    Parameters
    ----------
    None (stateless)
    """

    def diagnose(
        self,
        dsl:     str,
        metrics: Dict[str, Any],
        interpreter_result=None,   # optional InterpretResult from FinancialInterpreter
    ) -> FactorDiagnosis:
        """
        Diagnose an alpha factor from its DSL and backtest metrics.

        Parameters
        ----------
        dsl              : Alpha DSL expression string
        metrics          : dict with keys: oos_sharpe, is_sharpe, turnover,
                           mean_ic, ic_ir, max_drawdown, overfitting_score
        interpreter_result : Optional pre-computed InterpretResult

        Returns
        -------
        FactorDiagnosis
        """
        def _f(key: str, default: float = 0.0) -> float:
            v = metrics.get(key, default)
            try:
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        sharpe_is    = _f("is_sharpe",        0.0)
        sharpe_oos   = _f("oos_sharpe",        0.0)
        turnover     = _f("turnover",          0.0)
        mean_ic      = _f("mean_ic",           0.0)
        ic_ir        = _f("ic_ir",             0.0)
        max_dd       = _f("max_drawdown",      0.0)
        overfit      = _f("overfitting_score", 0.0)

        metrics_summary = {
            "is_sharpe":    round(sharpe_is,  3),
            "oos_sharpe":   round(sharpe_oos, 3),
            "turnover":     round(turnover,   2),
            "mean_ic":      round(mean_ic,    4),
            "ic_ir":        round(ic_ir,      3),
            "max_drawdown": round(max_dd,     3),
            "overfit_score":round(overfit,    3),
        }

        # Get factor family if interpreter result provided
        family    = interpreter_result.factor_family if interpreter_result else "unknown"
        raw_descr = interpreter_result.description   if interpreter_result else dsl

        # Detect primary issue
        issue, diagnosis, severity = self._detect_issue(
            sharpe_is, sharpe_oos, turnover, mean_ic, ic_ir, max_dd, overfit, family
        )

        # Build suggestions
        suggestions = self._build_suggestions(
            dsl, family, sharpe_is, sharpe_oos, turnover,
            mean_ic, ic_ir, max_dd, overfit
        )

        # Regime insight
        regime_note = self._regime_insight(
            sharpe_oos, turnover, mean_ic, family
        )

        return FactorDiagnosis(
            primary_issue   = issue,
            diagnosis       = diagnosis,
            severity        = severity,
            suggestions     = suggestions,
            regime_insight  = regime_note,
            metrics_summary = metrics_summary,
        )

    # ------------------------------------------------------------------
    # Issue detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_issue(
        sharpe_is, sharpe_oos, turnover, mean_ic, ic_ir, max_dd, overfit, family
    ):
        # Severity-ordered issue detection
        if overfit > 0.6 and sharpe_is > 0.8:
            return (
                "severe_overfitting",
                (
                    f"The factor achieves IS Sharpe {sharpe_is:.2f} but only {sharpe_oos:.2f} OOS. "
                    f"Overfitting score {overfit:.2f} indicates the alpha memorizes in-sample noise. "
                    f"Complex DSL structures tend to curve-fit to historical data without predictive power."
                ),
                "critical",
            )

        if turnover > 3.0:
            return (
                "high_turnover",
                (
                    f"Annualized turnover {turnover:.1f}× is unsustainably high. "
                    f"At typical transaction costs (10-30bps/trade), this erodes most of the "
                    f"gross alpha. Short-window signals or raw un-smoothed deltas are "
                    f"the usual culprit."
                ),
                "critical" if turnover > 5.0 else "moderate",
            )

        if sharpe_oos < 0.1 and mean_ic < 0.01:
            return (
                "no_signal",
                (
                    f"OOS Sharpe {sharpe_oos:.2f} and mean IC {mean_ic:.4f} indicate the factor "
                    f"has near-zero predictive power. The signal may be random noise, or the "
                    f"factor hypothesis itself does not hold in the data."
                ),
                "critical",
            )

        if mean_ic > 0.02 and ic_ir < 0.3:
            return (
                "noisy_signal",
                (
                    f"Mean IC {mean_ic:.4f} suggests directional skill, but IC IR {ic_ir:.2f} "
                    f"is too low for consistent alpha extraction. The signal is correct on "
                    f"average but unreliable day-to-day — classic sign of high signal noise. "
                    f"Smoothing or cross-sectional normalization typically helps."
                ),
                "moderate",
            )

        if abs(max_dd) > 0.25:
            return (
                "high_drawdown",
                (
                    f"Maximum drawdown {max_dd:.1%} is severe. The strategy has significant "
                    f"tail risk, often caused by factor crowding (too correlated with market beta), "
                    f"or absence of regime conditioning (trading in both bull and bear markets "
                    f"without adjustment)."
                ),
                "moderate",
            )

        if 0.1 < overfit <= 0.6 and sharpe_is > sharpe_oos + 0.3:
            return (
                "mild_overfitting",
                (
                    f"IS/OOS Sharpe gap ({sharpe_is:.2f} vs {sharpe_oos:.2f}) suggests mild "
                    f"overfitting. The factor picks up some genuine signal (OOS > 0) but "
                    f"also captures IS-specific patterns. Simplifying the DSL or adding "
                    f"cross-sectional neutralization typically closes this gap."
                ),
                "moderate",
            )

        if sharpe_oos > 0.5 and overfit < 0.3:
            return (
                "healthy",
                (
                    f"OOS Sharpe {sharpe_oos:.2f} with low overfitting ({overfit:.2f}) indicates "
                    f"a robust, generalizable alpha. Signal shows genuine predictive power "
                    f"confirmed on held-out data."
                ),
                "minor",
            )

        return (
            "weak_alpha",
            (
                f"OOS Sharpe {sharpe_oos:.2f} with turnover {turnover:.1f}× is marginal. "
                f"The signal likely captures a real but weak effect. Improving signal "
                f"construction or adding complementary factors may help."
            ),
            "moderate",
        )

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------

    @staticmethod
    def _build_suggestions(
        dsl, family, sharpe_is, sharpe_oos, turnover,
        mean_ic, ic_ir, max_dd, overfit,
    ) -> List[Suggestion]:
        suggestions = []
        priority    = 1

        # ── High turnover → smoothing ─────────────────────────────────
        if turnover > 2.0:
            suggestions.append(Suggestion(
                action      = "add_smoothing",
                priority    = priority,
                dsl_patch   = f"ts_mean({dsl}, 3)",
                reason      = f"Wrap signal with 3-day moving average to reduce flip frequency",
                finance_why = (
                    "Short-window signals change direction too often. A moving average "
                    "decay filter retains the direction of the signal while smoothing out "
                    "1-2 day noise. Every avoided roundtrip saves 20-40bps in costs."
                ),
            ))
            priority += 1
            suggestions.append(Suggestion(
                action      = "apply_decay",
                priority    = priority,
                dsl_patch   = f"ts_decay_linear({dsl}, 5)",
                reason      = "Linear decay weights recent signal more but avoids abrupt changes",
                finance_why = (
                    "Exponential/linear decay is the standard industry technique for "
                    "managing signal turnover. WorldQuant and Two Sigma use it extensively."
                ),
            ))
            priority += 1

        # ── Overfitting → simplify + neutralize ──────────────────────
        if overfit > 0.5:
            suggestions.append(Suggestion(
                action      = "neutralize",
                priority    = priority,
                dsl_patch   = f"zscore({dsl})",
                reason      = "Cross-sectional z-score removes market/sector level bias",
                finance_why = (
                    "Overfitting often occurs because the signal captures systematic "
                    "(market or sector) variation that is specific to the IS period. "
                    "Cross-sectional neutralization forces the signal to be long/short "
                    "within each period, removing directional beta."
                ),
            ))
            priority += 1
            suggestions.append(Suggestion(
                action      = "simplify_structure",
                priority    = priority,
                dsl_patch   = f"rank(ts_delta(log(close), 10))",
                reason      = "Replace complex tree with simple, robust momentum baseline",
                finance_why = (
                    "Occam's razor applies to alpha: simpler expressions generalize better "
                    "because they have fewer parameters to overfit. A log-return rank is "
                    "the canonical momentum factor that consistently works out-of-sample."
                ),
            ))
            priority += 1

        # ── Noisy IC → normalization and volume filter ─────────────────
        if mean_ic > 0.01 and ic_ir < 0.3:
            suggestions.append(Suggestion(
                action      = "add_zscore_ts",
                priority    = priority,
                dsl_patch   = f"rank(ts_zscore({dsl}, 20))",
                reason      = "Time-series z-score before ranking stabilizes signal level",
                finance_why = (
                    "Low IC IR with positive mean IC = signal has the right direction but "
                    "unstable magnitude. ts_zscore normalizes the signal relative to its "
                    "own recent history, removing non-stationarity in the signal level."
                ),
            ))
            priority += 1
            suggestions.append(Suggestion(
                action      = "volume_confirmation",
                priority    = priority,
                dsl_patch   = f"rank(ts_delta(log(close), 5) * sign(ts_delta(log(volume), 5)))",
                reason      = "Require volume confirmation for price signals",
                finance_why = (
                    "Price moves on high volume are more informationally meaningful than "
                    "low-volume moves. Multiplying by the sign of volume change filters "
                    "out noise trades and improves IC consistency."
                ),
            ))
            priority += 1

        # ── Low overall signal → try complementary factor ──────────────
        if sharpe_oos < 0.2 and mean_ic < 0.015:
            if family in ("momentum", "trend_following"):
                suggestions.append(Suggestion(
                    action      = "try_reversion",
                    priority    = priority,
                    dsl_patch   = "rank(neg(ts_delta(close, 3)))",
                    reason      = "Test short-term mean reversion as alternative to weak momentum",
                    finance_why = (
                        "When price momentum is weak, short-term reversion often works. "
                        "Institutional market makers create temporary price pressure that "
                        "reverses within 1-5 days — the classic liquidity provision premium."
                    ),
                ))
                priority += 1
                suggestions.append(Suggestion(
                    action      = "add_volume_alpha",
                    priority    = priority,
                    dsl_patch   = "rank(ts_delta(log(volume), 5) - ts_mean(ts_delta(log(volume), 5), 20))",
                    reason      = "Volume surprise signal — relative volume change vs. baseline",
                    finance_why = (
                        "Abnormal volume often predicts future returns independently of price. "
                        "Volume breakouts precede institutional accumulation/distribution. "
                        "This is a distinct alpha source from price momentum."
                    ),
                ))
                priority += 1
            elif family in ("reversion",):
                suggestions.append(Suggestion(
                    action      = "try_momentum",
                    priority    = priority,
                    dsl_patch   = "rank(ts_delta(log(close), 20))",
                    reason      = "Test medium-term momentum as the reversion signal may be too short",
                    finance_why = (
                        "Mean reversion works at 1-5 day horizons; momentum works at "
                        "1-12 month horizons. If your reversion horizon is too long, "
                        "you may be fighting the momentum effect."
                    ),
                ))
                priority += 1

        # ── High drawdown → regime filter ──────────────────────────────
        if abs(max_dd) > 0.20:
            suggestions.append(Suggestion(
                action      = "add_regime_filter",
                priority    = priority,
                dsl_patch   = f"trade_when(close > ts_mean(close, 200), {dsl})",
                reason      = "Only trade signal when above 200-day MA (bull regime)",
                finance_why = (
                    "Most equity long-short factors have significantly worse performance "
                    "in bear markets due to correlation breakdown and factor crowding. "
                    "The 200-day MA rule is a simple, effective market regime filter "
                    "that cuts tail losses substantially."
                ),
            ))
            priority += 1

        # ── Volatility adjustment ────────────────────────────────────
        if family in ("momentum", "composite") and turnover <= 2.0:
            suggestions.append(Suggestion(
                action      = "risk_adjust",
                priority    = priority,
                dsl_patch   = f"rank(ts_delta(log(close), 10) / ts_std(returns, 20))",
                reason      = "Divide momentum by volatility for risk-adjusted signal",
                finance_why = (
                    "Raw momentum is stronger on high-volatility stocks but carries "
                    "more risk. Dividing by realized volatility creates a risk-adjusted "
                    "momentum signal (similar to Sharpe ratio of returns) that has "
                    "better OOS performance and lower drawdowns."
                ),
            ))
            priority += 1

        return sorted(suggestions, key=lambda s: s.priority)

    # ------------------------------------------------------------------
    # Regime insight
    # ------------------------------------------------------------------

    @staticmethod
    def _regime_insight(sharpe_oos, turnover, mean_ic, family) -> Optional[str]:
        if family == "momentum" and sharpe_oos < 0.3:
            return (
                "Momentum underperforms in choppy / low-trend markets and during "
                "sharp risk-off reversals. Consider testing performance specifically "
                "in bull vs. bear regimes before deployment."
            )
        if family == "reversion" and turnover > 2.5:
            return (
                "Short-term reversion factors work best in high-liquidity, range-bound "
                "markets. In trending markets, this strategy fights the price trend, "
                "causing sustained drawdowns."
            )
        if family == "volatility" and sharpe_oos > 0.5:
            return (
                "Low-volatility factors tend to be defensive — they outperform in "
                "bear markets but lag in strong bull markets. Consider pairing with a "
                "momentum factor for regime balance."
            )
        if mean_ic > 0.03 and sharpe_oos > 0.5:
            return (
                "Strong IC suggests genuine predictive power. "
                "Ensure the factor remains valid across different market regimes "
                "(bull/bear/sideways) before scaling capital."
            )
        return None
