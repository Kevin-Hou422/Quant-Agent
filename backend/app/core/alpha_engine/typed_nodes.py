"""
Typed AST Node Hierarchy for the Alpha DSL Engine.

Each node carries a strict NodeType, a deterministic __repr__ that
reconstructs the original DSL string, and an evaluate() method that
returns a (T, N) NumPy array.

Memoization is handled externally by the Executor via the shared
``cache: dict[str, np.ndarray]`` argument passed into evaluate().
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# NodeType enum
# ---------------------------------------------------------------------------

class NodeType(Enum):
    DATA         = auto()   # raw field leaf
    SCALAR       = auto()   # numeric literal
    TIME_SERIES  = auto()   # rolling / time-axis operator
    CROSS_SECTIONAL = auto() # cross-asset operator
    ARITHMETIC   = auto()   # +, -, *, /, etc. — resolved from children


# Shorthand
_TS = NodeType.TIME_SERIES
_CS = NodeType.CROSS_SECTIONAL
_DA = NodeType.DATA
_SC = NodeType.SCALAR
_AR = NodeType.ARITHMETIC


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

# Type alias used throughout
Dataset = Dict[str, np.ndarray]   # field -> (T, N) array
Cache   = Dict[str, np.ndarray]   # repr(node) -> (T, N) array


class Node(ABC):
    """Abstract base for all DSL AST nodes."""

    node_type: NodeType  # must be set by each concrete class

    @abstractmethod
    def __repr__(self) -> str:
        """Reconstruct the exact DSL expression string."""
        ...

    @abstractmethod
    def _compute(self, dataset: Dataset, cache: Cache) -> np.ndarray:
        """Core computation (no caching logic here)."""
        ...

    def evaluate(self, dataset: Dataset, cache: Cache) -> np.ndarray:
        """
        Evaluate this node, consulting and populating the shared cache.
        Cache key is the deterministic repr() of the node.
        """
        key = repr(self)
        if key in cache:
            return cache[key]
        result = self._compute(dataset, cache)
        cache[key] = result
        return result

    def depth(self) -> int:
        return 0

    def children(self) -> List["Node"]:
        return []


# ---------------------------------------------------------------------------
# Scalar  (numeric literal)
# ---------------------------------------------------------------------------

class ScalarNode(Node):
    """A numeric constant — broadcasts to the shape of context data."""

    node_type = _SC

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __repr__(self) -> str:
        # Preserve integer appearance when possible
        if self.value == int(self.value):
            return str(int(self.value))
        return repr(self.value)

    def _compute(self, dataset: Dataset, cache: Cache) -> np.ndarray:
        # Lazy broadcast: return a 0-d array; Executor widens as needed.
        return np.array(self.value)

    def depth(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Data  (raw field reference)
# ---------------------------------------------------------------------------

class DataNode(Node):
    """Leaf node referencing a named OHLCV / derived field."""

    node_type = _DA

    def __init__(self, field: str) -> None:
        self.field = field

    def __repr__(self) -> str:
        return self.field

    def _compute(self, dataset: Dataset, cache: Cache) -> np.ndarray:
        if self.field not in dataset:
            raise KeyError(
                f"Field '{self.field}' not found in dataset. "
                f"Available: {sorted(dataset.keys())}"
            )
        return dataset[self.field].astype(float)

    def depth(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# TimeSeriesNode
# ---------------------------------------------------------------------------

_TS_OPS = frozenset({
    "ts_mean", "ts_std", "ts_max", "ts_min", "ts_rank",
    "ts_decay_linear", "ts_delta", "ts_delay",
    "ts_corr", "ts_cov",
})


class TimeSeriesNode(Node):
    """Rolling / time-axis operator applied to a single child."""

    node_type = _TS

    def __init__(self, op: str, child: Node, window: int, **extra_params) -> None:
        if op not in _TS_OPS:
            raise ValueError(f"Unknown TS operator: '{op}'")
        self.op = op
        self.child = child
        self.window = int(window)
        self.extra_params = extra_params  # e.g. second child for ts_corr

    def __repr__(self) -> str:
        base = f"{self.op}({repr(self.child)},{self.window})"
        return base

    def _compute(self, dataset: Dataset, cache: Cache) -> np.ndarray:
        from .fast_ops import FAST_TS_OPS
        x = self.child.evaluate(dataset, cache)
        fn = FAST_TS_OPS[self.op]
        return fn(x, self.window)

    def depth(self) -> int:
        return 1 + self.child.depth()

    def children(self) -> List[Node]:
        return [self.child]


# ---------------------------------------------------------------------------
# CrossSectionalNode
# ---------------------------------------------------------------------------

_CS_OPS = frozenset({"rank", "zscore", "scale", "ind_neutralize"})


class CrossSectionalNode(Node):
    """Cross-asset operator; child must be DATA or TIME_SERIES."""

    node_type = _CS

    def __init__(self, op: str, child: Node, **params) -> None:
        if op not in _CS_OPS:
            raise ValueError(f"Unknown CS operator: '{op}'")
        if child.node_type is _CS:
            raise TypeError(
                f"CrossSectionalNode '{op}': child must be DATA or TIME_SERIES, "
                f"got CROSS_SECTIONAL ({repr(child)}). "
                "Chaining CS ops is not allowed."
            )
        self.op = op
        self.child = child
        self.params = params  # e.g. groups= for ind_neutralize

    def __repr__(self) -> str:
        return f"{self.op}({repr(self.child)})"

    def _compute(self, dataset: Dataset, cache: Cache) -> np.ndarray:
        from .fast_ops import FAST_CS_OPS
        x = self.child.evaluate(dataset, cache)
        fn = FAST_CS_OPS[self.op]
        if self.op == "ind_neutralize":
            groups = self.params.get("groups")
            return fn(x, groups)
        return fn(x)

    def depth(self) -> int:
        return 1 + self.child.depth()

    def children(self) -> List[Node]:
        return [self.child]


# ---------------------------------------------------------------------------
# ArithmeticNode  (+, -, *, /, log, abs, signed_power, if_else, comparison)
# ---------------------------------------------------------------------------

_BINOP_SYMBOLS = {
    "add": "+", "sub": "-", "mul": "*", "div": "/",
    "gt": ">", "lt": "<", "gte": ">=", "lte": "<=", "eq": "==",
}
_BINOP_OPS = set(_BINOP_SYMBOLS.keys())
_UNARY_OPS = frozenset({"log", "abs", "neg", "sqrt", "sign"})
_ADVANCED_OPS = frozenset({"signed_power", "if_else"})


def _resolve_type(*nodes: Node) -> NodeType:
    """Propagate node type: CS > TS > DATA > SCALAR."""
    types = {n.node_type for n in nodes}
    if _CS in types:
        return _CS
    if _TS in types:
        return _TS
    if _DA in types:
        return _DA
    return _SC


class ArithmeticNode(Node):
    """
    Handles binary arithmetic, unary math, signed_power, if_else.
    node_type is resolved from children.
    """

    def __init__(
        self,
        op: str,
        children: List[Node],
    ) -> None:
        valid = _BINOP_OPS | _UNARY_OPS | _ADVANCED_OPS
        if op not in valid:
            raise ValueError(f"Unknown ArithmeticNode op: '{op}'")
        self.op = op
        self._children = children
        self.node_type = _resolve_type(*children)

    def __repr__(self) -> str:
        if self.op in _BINOP_SYMBOLS:
            sym = _BINOP_SYMBOLS[self.op]
            return f"({repr(self._children[0])}{sym}{repr(self._children[1])})"
        # Unary minus: emit  -(expr)  so the repr round-trips through the Lark parser
        # (the grammar rule  ?factor: _MINUS factor -> neg  expects a literal "-", not
        # the word "neg").  All other unary ops (log, abs, sqrt, sign) are real function
        # calls and keep their name-prefixed form.
        if self.op == "neg":
            return f"(-{repr(self._children[0])})"
        if self.op in _UNARY_OPS:
            return f"{self.op}({repr(self._children[0])})"
        if self.op == "signed_power":
            return f"signed_power({repr(self._children[0])},{repr(self._children[1])})"
        if self.op == "if_else":
            a, b, c = self._children
            return f"if_else({repr(a)},{repr(b)},{repr(c)})"
        return f"{self.op}({','.join(repr(c) for c in self._children)})"

    def _compute(self, dataset: Dataset, cache: Cache) -> np.ndarray:
        evaled = [c.evaluate(dataset, cache) for c in self._children]
        op = self.op

        if op == "add":  return evaled[0] + evaled[1]
        if op == "sub":  return evaled[0] - evaled[1]
        if op == "mul":  return evaled[0] * evaled[1]
        if op == "div":
            denom = evaled[1]
            safe = np.where(np.abs(denom) < 1e-8, np.nan, denom)
            return evaled[0] / safe
        if op == "gt":   return (evaled[0] > evaled[1]).astype(float)
        if op == "lt":   return (evaled[0] < evaled[1]).astype(float)
        if op == "gte":  return (evaled[0] >= evaled[1]).astype(float)
        if op == "lte":  return (evaled[0] <= evaled[1]).astype(float)
        if op == "eq":   return (evaled[0] == evaled[1]).astype(float)
        if op == "log":  return np.log(np.where(evaled[0] > 0, evaled[0], np.nan))
        if op == "abs":  return np.abs(evaled[0])
        if op == "neg":  return -evaled[0]
        if op == "sqrt": return np.sqrt(np.where(evaled[0] >= 0, evaled[0], np.nan))
        if op == "sign": return np.sign(evaled[0])
        if op == "signed_power":
            x, p = evaled[0], evaled[1]
            return np.sign(x) * np.abs(x) ** p
        if op == "if_else":
            cond = evaled[0].astype(bool)
            return np.where(cond, evaled[1], evaled[2])
        raise NotImplementedError(f"ArithmeticNode._compute: unknown op '{op}'")

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self._children), default=0)

    def children(self) -> List[Node]:
        return list(self._children)
