"""
financial_interpreter.py — Semantic interpretation of Alpha DSL expressions.

Converts a typed AST into human-readable financial descriptions and
classifies the factor into a canonical financial family.

Usage::

    from app.core.alpha_engine.financial_interpreter import FinancialInterpreter
    result = FinancialInterpreter().interpret("rank(ts_delta(log(close), 5))")
    print(result.description)
    # "Cross-sectional rank of 5-day log-return momentum (closing price)"
    print(result.factor_family)
    # "momentum"
    print(result.issues)
    # ["No smoothing applied — signal may be noisy", "No volume component"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Field → human name
# ---------------------------------------------------------------------------

_FIELD_NAMES: Dict[str, str] = {
    "close":   "closing price",
    "open":    "opening price",
    "high":    "daily high",
    "low":     "daily low",
    "volume":  "trading volume",
    "vwap":    "volume-weighted average price (VWAP)",
    "returns": "daily log returns",
}

# ---------------------------------------------------------------------------
# Factor family taxonomy
# ---------------------------------------------------------------------------

FACTOR_FAMILIES = {
    "momentum":           "Price Momentum — buys recent winners, sells recent losers",
    "reversion":          "Mean Reversion — fades short-term price extremes",
    "volatility":         "Volatility / Risk — trades on realized risk levels",
    "liquidity":          "Liquidity — exploits volume or price-volume relationships",
    "quality":            "Quality / Stability — favors low-risk, stable assets",
    "trend_following":    "Trend Following — aligns with medium/long-term price direction",
    "price_volume_corr":  "Price-Volume Correlation — exploits co-movement of price and flow",
    "carry":              "Carry / Yield — favors high-return, low-cost assets",
    "composite":          "Composite — combines multiple factor families",
}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InterpretResult:
    """Full financial interpretation of a DSL expression."""
    dsl:           str
    description:   str                  # natural language description
    factor_family: str                  # canonical family name
    family_desc:   str                  # one-line family explanation
    data_fields:   List[str]            # data fields used
    operators_ts:  List[str]            # time-series operators used
    operators_cs:  List[str]            # cross-sectional operators used
    max_window:    int                  # longest time window
    min_window:    int                  # shortest time window
    is_normalized: bool                 # has rank/zscore at root level
    has_volume:    bool                 # uses volume data
    has_condition: bool                 # has if_else / trade_when
    complexity:    int                  # 1 (simple) – 5 (complex)
    issues:        List[str]            # potential design issues
    suggestions:   List[str]            # improvement suggestions

    def to_dict(self) -> dict:
        return {
            "description":   self.description,
            "factor_family": self.factor_family,
            "family_desc":   self.family_desc,
            "data_fields":   self.data_fields,
            "max_window":    self.max_window,
            "min_window":    self.min_window,
            "is_normalized": self.is_normalized,
            "has_volume":    self.has_volume,
            "complexity":    self.complexity,
            "issues":        self.issues,
            "suggestions":   self.suggestions,
        }

    def summary(self) -> str:
        lines = [
            f"Factor: {self.factor_family.upper()} | Complexity: {self.complexity}/5",
            f"Description: {self.description}",
            f"Fields used: {', '.join(self.data_fields)}",
            f"Windows: {self.min_window}–{self.max_window} days",
            f"Cross-sectional normalized: {'Yes' if self.is_normalized else 'No'}",
        ]
        if self.issues:
            lines.append("Issues: " + "; ".join(self.issues))
        if self.suggestions:
            lines.append("Suggestions: " + "; ".join(self.suggestions))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main interpreter
# ---------------------------------------------------------------------------

class FinancialInterpreter:
    """
    Interprets an Alpha DSL expression or typed AST node into a structured
    financial description.
    """

    def interpret(self, dsl: str) -> InterpretResult:
        """
        Parse ``dsl`` and return a full InterpretResult.

        Parameters
        ----------
        dsl : Alpha DSL expression string

        Returns
        -------
        InterpretResult
        """
        from app.core.alpha_engine.parser import Parser
        node = Parser().parse(dsl)
        return self.interpret_node(node, dsl)

    def interpret_node(self, node, dsl: str = "") -> InterpretResult:
        """Interpret a pre-parsed typed AST node."""
        description   = _describe(node)
        fields        = sorted(_collect_fields(node))
        ts_ops        = sorted(set(_collect_ts_ops(node)))
        cs_ops        = sorted(set(_collect_cs_ops(node)))
        all_ops       = _collect_all_ops(node)
        windows       = _collect_windows(node)
        max_w         = max(windows) if windows else 0
        min_w         = min(windows) if windows else 0
        complexity    = _compute_complexity(node)
        is_normalized = _has_cs_at_root(node)
        has_volume    = "volume" in fields
        has_cond      = bool({"if_else", "trade_when", "where"} & set(all_ops))

        family   = _classify_family(node, fields, all_ops)
        fam_desc = FACTOR_FAMILIES.get(family, family)

        issues, suggestions = _analyze_design(
            node, family, is_normalized, has_volume,
            has_cond, max_w, ts_ops, all_ops, complexity
        )

        return InterpretResult(
            dsl           = dsl or repr(node),
            description   = description,
            factor_family = family,
            family_desc   = fam_desc,
            data_fields   = fields,
            operators_ts  = ts_ops,
            operators_cs  = cs_ops,
            max_window    = max_w,
            min_window    = min_w,
            is_normalized = is_normalized,
            has_volume    = has_volume,
            has_condition = has_cond,
            complexity    = complexity,
            issues        = issues,
            suggestions   = suggestions,
        )


# ---------------------------------------------------------------------------
# Tree-walking description (bottom-up)
# ---------------------------------------------------------------------------

def _describe(node) -> str:
    """Recursively generate a human-readable description of the AST."""
    from app.core.alpha_engine.typed_nodes import (
        DataNode, ScalarNode, StringLiteralNode,
        TimeSeriesNode, CrossSectionalNode, GroupNode, ArithmeticNode,
    )

    if isinstance(node, DataNode):
        return _FIELD_NAMES.get(node.field, node.field)

    if isinstance(node, (ScalarNode,)):
        v = node.value
        return str(int(v)) if v == int(v) else str(round(v, 3))

    if isinstance(node, StringLiteralNode):
        return f"'{node.value}'"

    if isinstance(node, TimeSeriesNode):
        return _describe_ts(node)

    if isinstance(node, CrossSectionalNode):
        return _describe_cs(node)

    if isinstance(node, GroupNode):
        child = _describe(node.child)
        op = node.op
        gf = node.group_field
        mapping = {
            "group_rank":       f"within-group rank of [{child}] by {gf}",
            "group_zscore":     f"within-group z-score of [{child}] by {gf}",
            "group_mean":       f"within-group mean of [{child}] by {gf}",
            "group_neutralize": f"[{child}] neutralized within {gf} groups",
        }
        return mapping.get(op, f"{op}([{child}], {gf})")

    if isinstance(node, ArithmeticNode):
        return _describe_arith(node)

    return repr(node)


def _describe_ts(node) -> str:
    from app.core.alpha_engine.typed_nodes import TimeSeriesNode
    child = _describe(node.child)
    w     = node.window
    op    = node.op

    _period = lambda n: (
        f"{n}-day" if n != 1 else "daily"
    )

    if op == "ts_delta":
        if w == 1:   return f"1-day price change in {child}"
        elif w <= 5: return f"{w}-day short-term momentum of {child}"
        elif w <= 20: return f"{w}-day medium-term momentum of {child}"
        else:        return f"{w}-day long-term trend of {child}"

    if op == "ts_mean":
        if w <= 5:   return f"{w}-day fast moving average of {child}"
        elif w <= 20: return f"{w}-day medium moving average of {child}"
        else:        return f"{w}-day long-term trend line of {child}"

    if op == "ts_std":
        return f"{w}-day realized volatility of {child}"

    if op == "ts_var":
        return f"{w}-day variance of {child}"

    if op == "ts_delay":
        return f"{child} lagged {w} day{'s' if w > 1 else ''}"

    if op == "ts_rank":
        return f"historical percentile rank of {child} over {w} days"

    if op == "ts_zscore":
        return f"time-series z-score of {child} over {w} days (mean-reverting normalization)"

    if op == "ts_max":
        return f"{w}-day peak level of {child}"

    if op == "ts_min":
        return f"{w}-day trough level of {child}"

    if op == "ts_sum":
        return f"{w}-day cumulative sum of {child}"

    if op == "ts_decay_linear":
        return f"linearly time-decayed {child} over {w} days (recent days weighted more)"

    if op == "ts_argmax":
        return f"days since {w}-day peak of {child}"

    if op == "ts_argmin":
        return f"days since {w}-day trough of {child}"

    if op == "ts_corr":
        child2 = _describe(node.second_child) if node.second_child else "?"
        return f"{w}-day rolling correlation between {child} and {child2}"

    if op == "ts_cov":
        child2 = _describe(node.second_child) if node.second_child else "?"
        return f"{w}-day rolling covariance between {child} and {child2}"

    if op == "ts_skew":
        return f"{w}-day return skewness of {child} (distribution asymmetry)"

    if op == "ts_kurt":
        return f"{w}-day return kurtosis of {child} (tail risk measure)"

    if op == "ts_entropy":
        return f"{w}-day information entropy of {child} (complexity measure)"

    return f"{op}({child}, {w})"


def _describe_cs(node) -> str:
    child = _describe(node.child)
    op    = node.op
    if op == "rank":
        return f"cross-sectional percentile rank of [{child}] (0=worst, 1=best)"
    if op == "zscore":
        return f"cross-sectional z-score of [{child}] (deviations from universe mean)"
    if op == "scale":
        return f"cross-sectionally unit-scaled [{child}]"
    if op == "winsorize":
        k = getattr(node, "k", 3.0)
        return f"[{child}] with outliers clipped at ±{k}σ"
    if op == "normalize":
        return f"cross-sectionally min-max normalized [{child}]"
    if op == "ind_neutralize":
        return f"[{child}] with industry effects removed"
    return f"{op}([{child}])"


def _describe_arith(node) -> str:
    from app.core.alpha_engine.typed_nodes import ArithmeticNode
    ch = node.children()
    op = node.op

    UNARY = {
        "neg":         lambda c: f"inverted signal of [{c[0]}]",
        "log":         lambda c: f"log-transformed {c[0]}",
        "abs":         lambda c: f"absolute value of [{c[0]}]",
        "sqrt":        lambda c: f"square root of [{c[0]}]",
        "sign":        lambda c: f"directional sign of [{c[0]}] (+1 or -1)",
        "logical_not": lambda c: f"NOT [{c[0]}]",
    }
    BINARY = {
        "add":         lambda a, b: f"({a} + {b})",
        "sub":         lambda a, b: f"({a} minus {b})",
        "mul":         lambda a, b: f"({a} × {b})",
        "div":         lambda a, b: f"ratio of ({a}) to ({b})",
        "pow":         lambda a, b: f"({a}) raised to power {b}",
        "max2":        lambda a, b: f"maximum of [{a}] and [{b}]",
        "min2":        lambda a, b: f"minimum of [{a}] and [{b}]",
        "signed_power":lambda a, b: f"sign-preserving power {b} of [{a}]",
        "logical_and": lambda a, b: f"({a} AND {b})",
        "logical_or":  lambda a, b: f"({a} OR {b})",
        "ne":          lambda a, b: f"({a} ≠ {b})",
        "gt":          lambda a, b: f"({a} > {b})",
        "lt":          lambda a, b: f"({a} < {b})",
        "gte":         lambda a, b: f"({a} ≥ {b})",
        "lte":         lambda a, b: f"({a} ≤ {b})",
        "eq":          lambda a, b: f"({a} = {b})",
    }

    descs = [_describe(c) for c in ch]

    if op in UNARY:
        return UNARY[op](descs)
    if op in BINARY and len(descs) >= 2:
        return BINARY[op](descs[0], descs[1])

    if op == "if_else" and len(descs) >= 3:
        return (
            f"conditional signal: [{descs[1]}] when {descs[0]}, "
            f"otherwise [{descs[2]}]"
        )
    if op == "trade_when" and len(descs) >= 2:
        return (
            f"trade signal [{descs[1]}] only when {descs[0]} is true; "
            f"flat otherwise"
        )
    if op == "where" and len(descs) >= 3:
        return (
            f"[{descs[1]}] where {descs[0]}, else [{descs[2]}]"
        )
    if op == "weighted_sum":
        n  = len(descs) // 2
        pairs = [f"{descs[i]} × {descs[i+n]}" for i in range(n)]
        return f"weighted combination: {' + '.join(pairs)}"

    return f"{op}({', '.join(descs)})"


# ---------------------------------------------------------------------------
# Tree-walking helpers
# ---------------------------------------------------------------------------

def _collect_fields(node) -> Set[str]:
    from app.core.alpha_engine.typed_nodes import DataNode
    fields = set()
    if isinstance(node, DataNode):
        fields.add(node.field)
    for child in node.children():
        fields |= _collect_fields(child)
    return fields


def _collect_ts_ops(node) -> List[str]:
    from app.core.alpha_engine.typed_nodes import TimeSeriesNode
    ops = []
    if isinstance(node, TimeSeriesNode):
        ops.append(node.op)
    for child in node.children():
        ops.extend(_collect_ts_ops(child))
    return ops


def _collect_cs_ops(node) -> List[str]:
    from app.core.alpha_engine.typed_nodes import CrossSectionalNode, GroupNode
    ops = []
    if isinstance(node, (CrossSectionalNode, GroupNode)):
        ops.append(node.op)
    for child in node.children():
        ops.extend(_collect_cs_ops(child))
    return ops


def _collect_all_ops(node) -> List[str]:
    ops = []
    if hasattr(node, "op"):
        ops.append(node.op)
    for child in node.children():
        ops.extend(_collect_all_ops(child))
    return ops


def _collect_windows(node) -> List[int]:
    from app.core.alpha_engine.typed_nodes import TimeSeriesNode
    windows = []
    if isinstance(node, TimeSeriesNode):
        windows.append(node.window)
    for child in node.children():
        windows.extend(_collect_windows(child))
    return windows


def _has_cs_at_root(node) -> bool:
    from app.core.alpha_engine.typed_nodes import CrossSectionalNode
    if isinstance(node, CrossSectionalNode) and node.op in ("rank", "zscore", "scale", "normalize"):
        return True
    # Check if root is arithmetic and direct child is CS
    from app.core.alpha_engine.typed_nodes import ArithmeticNode
    if isinstance(node, ArithmeticNode):
        return any(
            isinstance(c, CrossSectionalNode) and c.op in ("rank", "zscore")
            for c in node.children()
        )
    return False


def _compute_complexity(node) -> int:
    """Return complexity score 1-5 based on node count and nesting depth."""
    n_nodes = node.node_count() if hasattr(node, "node_count") else _count_nodes(node)
    depth   = node.depth()       if hasattr(node, "depth")      else _tree_depth(node)
    if n_nodes <= 3:   return 1
    if n_nodes <= 6:   return 2
    if n_nodes <= 10:  return 3
    if n_nodes <= 16:  return 4
    return 5


def _count_nodes(node) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children())


def _tree_depth(node) -> int:
    ch = node.children()
    if not ch:
        return 0
    return 1 + max(_tree_depth(c) for c in ch)


# ---------------------------------------------------------------------------
# Factor family classification
# ---------------------------------------------------------------------------

def _classify_family(node, fields: Set[str], all_ops: List[str]) -> str:
    ops = set(all_ops)

    # Price-volume correlation → specific liquidity subfamily
    if "ts_corr" in ops and "volume" in fields:
        return "price_volume_corr"

    # Pure volume / liquidity
    if "volume" in fields and not ({"ts_delta", "ts_zscore"} & ops):
        return "liquidity"

    # Volatility signal (ts_std / ts_var as primary component)
    if ("ts_std" in ops or "ts_var" in ops) and "ts_delta" not in ops:
        return "volatility"

    # Entropy / skew / kurtosis → risk/quality proxy
    if ops & {"ts_entropy", "ts_skew", "ts_kurt"}:
        return "quality"

    # Check if momentum is inverted (mean reversion)
    if "ts_delta" in ops and _is_inverted_momentum(node):
        return "reversion"

    # ts_zscore on price → mean reversion normalization
    if "ts_zscore" in ops and "ts_delta" not in ops:
        return "reversion"

    # Clear momentum signals
    if ops & {"ts_delta", "ts_rank", "ts_argmax", "ts_argmin"}:
        # Long-window momentum → trend following
        windows = _collect_windows(node)
        if windows and max(windows) >= 60:
            return "trend_following"
        return "momentum"

    # Time-series smoothing only → trend following
    if "ts_mean" in ops and not (ops & {"ts_delta", "ts_std", "ts_corr"}):
        return "trend_following"

    # Multiple components
    families_found = []
    if ops & {"ts_delta"}:              families_found.append("momentum")
    if "volume" in fields:              families_found.append("liquidity")
    if ops & {"ts_std", "ts_var"}:      families_found.append("volatility")

    if len(families_found) >= 2:
        return "composite"

    return "momentum"  # default


def _is_inverted_momentum(node) -> bool:
    """Return True if the dominant momentum signal is negated."""
    from app.core.alpha_engine.typed_nodes import ArithmeticNode, TimeSeriesNode
    if isinstance(node, ArithmeticNode) and node.op == "neg":
        ch = node.children()
        return len(ch) > 0 and _has_op(ch[0], {"ts_delta", "ts_rank"})
    for child in node.children():
        if _is_inverted_momentum(child):
            return True
    return False


def _has_op(node, ops: Set[str]) -> bool:
    if hasattr(node, "op") and node.op in ops:
        return True
    return any(_has_op(c, ops) for c in node.children())


# ---------------------------------------------------------------------------
# Design issue analysis
# ---------------------------------------------------------------------------

def _analyze_design(
    node, family: str, is_normalized: bool, has_volume: bool,
    has_cond: bool, max_window: int, ts_ops: List[str],
    all_ops: List[str], complexity: int,
) -> tuple:
    issues:      List[str] = []
    suggestions: List[str] = []
    ops = set(all_ops)

    # No cross-sectional normalization
    if not is_normalized:
        issues.append("No cross-sectional normalization (rank/zscore missing at output)")
        suggestions.append(
            "Wrap the final signal with rank() or zscore() to remove universe-level bias"
        )

    # Momentum without smoothing
    if family in ("momentum", "trend_following") and "ts_mean" not in ops and "ts_decay_linear" not in ops:
        issues.append("Raw momentum signal without smoothing — likely high turnover")
        suggestions.append(
            "Apply ts_mean(signal, 3-5) or ts_decay_linear(signal, 5) to reduce signal noise"
        )

    # Short window momentum without volume confirmation
    if family == "momentum" and max_window <= 5 and not has_volume:
        issues.append("Short-term momentum without volume confirmation — prone to false signals")
        suggestions.append(
            "Add volume filter: trade_when(volume > ts_mean(volume, 20), signal)"
        )

    # Reversion signal needs normalization to avoid mean drift
    if family == "reversion" and "ts_zscore" not in ops and not is_normalized:
        issues.append("Mean reversion signal without z-score normalization — unstable signal levels")
        suggestions.append(
            "Replace negated delta with ts_zscore(close, N) for cleaner reversion signal"
        )

    # Very long or very short windows
    if max_window > 120:
        issues.append(f"Very long window ({max_window}d) — possible look-back overfitting")
        suggestions.append("Consider reducing the longest window to ≤ 60 days for generalizability")

    if max_window > 0 and max_window < 5 and family != "reversion":
        issues.append(f"Very short window ({max_window}d) — high signal noise and turnover")
        suggestions.append("Increase window to at least 5-10 days to reduce noise")

    # High complexity
    if complexity >= 4:
        issues.append(f"High complexity (score {complexity}/5) — overfitting risk")
        suggestions.append(
            "Simplify by removing nested TS operations; keep the strongest signal component"
        )

    # No regime conditioning for momentum
    if family in ("momentum", "trend_following") and not has_cond:
        suggestions.append(
            "Consider adding regime filter: trade_when(close > ts_mean(close, 200), signal) "
            "to trade momentum only in uptrends"
        )

    # Volatility signal without normalization
    if family == "volatility" and not is_normalized:
        issues.append("Volatility signal not cross-sectionally normalized")
        suggestions.append(
            "Use rank(neg(ts_std(returns, 20))) for a low-volatility long-short factor"
        )

    return issues, suggestions
