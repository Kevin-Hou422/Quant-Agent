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

ARITHMETIC_OPS = {"add", "sub", "mul", "div"}

UNARY_OPS = {"log", "abs", "neg", "sqrt", "sign"}

TIME_SERIES_OPS = {
    "ts_mean",      # rolling mean over window
    "ts_std",       # rolling std over window
    "ts_delta",     # x - lag(x, window)
    "ts_delay",     # lag(x, window)
    "ts_max",       # rolling max over window
    "ts_min",       # rolling min over window
    "ts_decay_linear",  # linearly-weighted decay over window
    "ts_rank",      # rolling rank of latest value within window
    "ts_corr",      # rolling cross-asset correlation (2 inputs)
    "ts_cov",       # rolling covariance (2 inputs)
}

CROSS_SECTIONAL_OPS = {
    "rank",         # cross-sectional rank (0..1)
    "zscore",       # cross-sectional z-score
    "demean",       # subtract cross-sectional mean
    "group_rank",   # rank within a group
    "group_zscore", # z-score within a group
    "winsorize",    # clip to [-k*std, +k*std] cross-sectionally
}

LEAF_OPS = {"data"}  # raw data field reference

ALL_OPS = ARITHMETIC_OPS | UNARY_OPS | TIME_SERIES_OPS | CROSS_SECTIONAL_OPS | LEAF_OPS

# Map op -> expected number of children (None = variadic)
OP_ARITY: dict[str, int | None] = {
    # arithmetic
    "add": 2, "sub": 2, "mul": 2, "div": 2,
    # unary
    "log": 1, "abs": 1, "neg": 1, "sqrt": 1, "sign": 1,
    # time-series (single input + window param)
    "ts_mean": 1, "ts_std": 1, "ts_delta": 1, "ts_delay": 1,
    "ts_max": 1, "ts_min": 1, "ts_decay_linear": 1, "ts_rank": 1,
    # time-series (two inputs + window param)
    "ts_corr": 2, "ts_cov": 2,
    # cross-sectional
    "rank": 1, "zscore": 1, "demean": 1,
    "group_rank": 1, "group_zscore": 1, "winsorize": 1,
    # leaf
    "data": 0,
}

# Default params per op
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "ts_mean":         {"window": 10},
    "ts_std":          {"window": 10},
    "ts_delta":        {"window": 1},
    "ts_delay":        {"window": 1},
    "ts_max":          {"window": 10},
    "ts_min":          {"window": 10},
    "ts_decay_linear": {"window": 10},
    "ts_rank":         {"window": 10},
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
