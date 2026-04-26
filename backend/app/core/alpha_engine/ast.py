"""
Alpha Expression AST (Abstract Syntax Tree)
Defines the node structure for alpha factor expressions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional
import copy


# ---------------------------------------------------------------------------
# Supported operator registry
# ---------------------------------------------------------------------------

ARITHMETIC_OPS = {"add", "sub", "mul", "div", "ne", "pow", "max2", "min2"}

UNARY_OPS = {"log", "abs", "neg", "sqrt", "sign", "logical_not"}

LOGICAL_OPS = {"logical_and", "logical_or"}

CONDITIONAL_OPS = {"if_else", "where", "trade_when", "signed_power", "weighted_sum"}

TIME_SERIES_OPS = {
    # Standard
    "ts_mean", "ts_std", "ts_var", "ts_sum",
    "ts_max", "ts_min", "ts_rank",
    "ts_decay_linear", "ts_delta", "ts_delay",
    # Extended single-input
    "ts_argmax", "ts_argmin", "ts_zscore",
    "ts_skew", "ts_kurt", "ts_entropy",
    # Two-input
    "ts_corr", "ts_cov",
}

CROSS_SECTIONAL_OPS = {
    "rank", "zscore", "scale", "ind_neutralize",
    "winsorize", "normalize",
}

GROUP_OPS = {
    "group_rank", "group_zscore", "group_mean", "group_neutralize",
}

LEAF_OPS = {"data"}  # raw data field reference

ALL_OPS = (
    ARITHMETIC_OPS | UNARY_OPS | LOGICAL_OPS | CONDITIONAL_OPS
    | TIME_SERIES_OPS | CROSS_SECTIONAL_OPS | GROUP_OPS | LEAF_OPS
)

# Map op -> expected number of children (None = variadic)
OP_ARITY: dict[str, int | None] = {
    # arithmetic
    "add": 2, "sub": 2, "mul": 2, "div": 2,
    "ne": 2, "pow": 2, "max2": 2, "min2": 2,
    # unary
    "log": 1, "abs": 1, "neg": 1, "sqrt": 1, "sign": 1, "logical_not": 1,
    # logical binary
    "logical_and": 2, "logical_or": 2,
    # conditional
    "if_else": 3, "where": 3, "trade_when": 2,
    "signed_power": 2, "weighted_sum": None,
    # time-series (single input + window param)
    "ts_mean": 1, "ts_std": 1, "ts_var": 1, "ts_sum": 1,
    "ts_delta": 1, "ts_delay": 1,
    "ts_max": 1, "ts_min": 1, "ts_decay_linear": 1, "ts_rank": 1,
    "ts_argmax": 1, "ts_argmin": 1, "ts_zscore": 1,
    "ts_skew": 1, "ts_kurt": 1, "ts_entropy": 1,
    # time-series (two inputs + window param)
    "ts_corr": 2, "ts_cov": 2,
    # cross-sectional
    "rank": 1, "zscore": 1, "scale": 1, "ind_neutralize": 1,
    "winsorize": 1, "normalize": 1,
    # group
    "group_rank": 1, "group_zscore": 1, "group_mean": 1, "group_neutralize": 1,
    # leaf
    "data": 0,
}

# Default params per op
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "ts_mean":         {"window": 10},
    "ts_std":          {"window": 10},
    "ts_var":          {"window": 10},
    "ts_sum":          {"window": 10},
    "ts_delta":        {"window": 1},
    "ts_delay":        {"window": 1},
    "ts_max":          {"window": 10},
    "ts_min":          {"window": 10},
    "ts_decay_linear": {"window": 10},
    "ts_rank":         {"window": 10},
    "ts_argmax":       {"window": 10},
    "ts_argmin":       {"window": 10},
    "ts_zscore":       {"window": 20},
    "ts_skew":         {"window": 20},
    "ts_kurt":         {"window": 20},
    "ts_entropy":      {"window": 20},
    "ts_corr":         {"window": 10},
    "ts_cov":          {"window": 10},
    "winsorize":       {"k": 3.0},
    "data":            {"field": "close"},
}

VALID_DATA_FIELDS = {"open", "high", "low", "close", "volume", "vwap", "returns"}


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """
    A single node in the alpha expression AST.

    Attributes
    ----------
    op : str
        Operator name (must be in ALL_OPS).
    children : List[Node]
        Child nodes (operands).
    params : dict
        Operator-specific parameters (e.g. window sizes, field names).
    """
    op: str
    children: List["Node"] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.op not in ALL_OPS:
            raise ValueError(f"Unknown operator: '{self.op}'. Valid ops: {sorted(ALL_OPS)}")
        expected = OP_ARITY.get(self.op)
        if expected is not None and len(self.children) != expected:
            raise ValueError(
                f"Operator '{self.op}' expects {expected} children, "
                f"got {len(self.children)}."
            )
        # Merge defaults without overwriting explicit params
        defaults = DEFAULT_PARAMS.get(self.op, {})
        for k, v in defaults.items():
            self.params.setdefault(k, v)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def data(cls, field: str = "close") -> "Node":
        if field not in VALID_DATA_FIELDS:
            raise ValueError(f"Unknown data field: '{field}'. Valid: {VALID_DATA_FIELDS}")
        return cls(op="data", params={"field": field})

    @classmethod
    def binary(cls, op: str, left: "Node", right: "Node") -> "Node":
        return cls(op=op, children=[left, right])

    @classmethod
    def unary(cls, op: str, child: "Node", **params) -> "Node":
        return cls(op=op, children=[child], params=params)

    @classmethod
    def ts(cls, op: str, child: "Node", window: int = 10, **extra) -> "Node":
        return cls(op=op, children=[child], params={"window": window, **extra})

    @classmethod
    def ts2(cls, op: str, left: "Node", right: "Node", window: int = 10) -> "Node":
        return cls(op=op, children=[left, right], params={"window": window})

    @classmethod
    def cs(cls, op: str, child: "Node", **params) -> "Node":
        return cls(op=op, children=[child], params=params)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self.children)

    def is_leaf(self) -> bool:
        return self.op in LEAF_OPS

    def clone(self) -> "Node":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        if self.op == "data":
            return f"data({self.params.get('field', 'close')})"
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        children_str = ", ".join(repr(c) for c in self.children)
        parts = [s for s in [children_str, params_str] if s]
        return f"{self.op}({', '.join(parts)})"

    def to_dict(self) -> dict:
        """Serialize to a JSON-serializable dict."""
        return {
            "op": self.op,
            "params": self.params,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        """Deserialize from a dict produced by ``to_dict``."""
        children = [cls.from_dict(c) for c in d.get("children", [])]
        return cls(op=d["op"], children=children, params=dict(d.get("params", {})))
