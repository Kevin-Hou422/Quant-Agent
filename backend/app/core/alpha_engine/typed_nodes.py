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
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# NodeType enum
# ---------------------------------------------------------------------------

class NodeType(Enum):
    DATA            = auto()   # raw field leaf
    SCALAR          = auto()   # numeric literal
    STRING_LITERAL  = auto()   # string literal (for group field names)
    TIME_SERIES     = auto()   # rolling / time-axis operator
    CROSS_SECTIONAL = auto()   # cross-asset operator
    GROUP           = auto()   # within-group cross-sectional operator
    ARITHMETIC      = auto()   # +, -, *, /, etc. — resolved from children


# Shorthand
_TS = NodeType.TIME_SERIES
_CS = NodeType.CROSS_SECTIONAL
_GR = NodeType.GROUP
_DA = NodeType.DATA
_SC = NodeType.SCALAR
_ST = NodeType.STRING_LITERAL


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

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
# ScalarNode  (numeric literal)
# ---------------------------------------------------------------------------

class ScalarNode(Node):
    """A numeric constant — broadcasts to the shape of context data."""

    node_type = _SC

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __repr__(self) -> str:
        if self.value == int(self.value):
            return str(int(self.value))
        return repr(self.value)

    def _compute(self, dataset: Dataset, cache: Cache) -> np.ndarray:
        return np.array(self.value)

    def depth(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# StringLiteralNode  (string literal — for group field names)
# ---------------------------------------------------------------------------

class StringLiteralNode(Node):
    """
    A string literal used as a function argument (e.g., group field name).
    Cannot be evaluated as an array directly; consumed by GroupNode.
    """

    node_type = _ST

    def __init__(self, value: str) -> None:
        self.value = str(value)

    def __repr__(self) -> str:
        return f"'{self.value}'"

    def _compute(self, dataset: Dataset, cache: Cache) -> np.ndarray:
        raise RuntimeError(
            f"StringLiteralNode('{self.value}') cannot be evaluated as an array. "
            "It should be consumed by a GroupNode."
        )  # dataset/cache intentionally unused — node raises before any access

    def depth(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# DataNode  (raw field reference)
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
    # Standard
    "ts_mean", "ts_std", "ts_var", "ts_sum",
    "ts_max", "ts_min", "ts_rank",
    "ts_decay_linear", "ts_delta", "ts_delay",
    # Extended single-input
    "ts_argmax", "ts_argmin", "ts_zscore",
    "ts_skew", "ts_kurt", "ts_entropy",
    # Two-input (require second_child)
    "ts_corr", "ts_cov",
})

# Two-input TS ops that take (x, y, window)
_TWO_INPUT_TS_OPS = frozenset({"ts_corr", "ts_cov"})


class TimeSeriesNode(Node):
    """Rolling / time-axis operator applied to one or two children."""

    node_type = _TS

    def __init__(
        self,
        op: str,
        child: Node,
        window: int,
        second_child: Optional[Node] = None,
    ) -> None:
        if op not in _TS_OPS:
            raise ValueError(f"Unknown TS operator: '{op}'")
        self.op           = op
        self.child        = child
        self.window       = int(window)
        self.second_child = second_child

    def __repr__(self) -> str:
        if self.second_child is not None:
            return (
                f"{self.op}({repr(self.child)},"
                f"{repr(self.second_child)},{self.window})"
            )
        return f"{self.op}({repr(self.child)},{self.window})"

    def _compute(self, dataset: Dataset, cache: Cache) -> np.ndarray:
        from .fast_ops import FAST_TS_OPS
        x  = self.child.evaluate(dataset, cache)
        fn = FAST_TS_OPS[self.op]
        if self.op in _TWO_INPUT_TS_OPS:
            if self.second_child is None:
                raise ValueError(
                    f"'{self.op}' requires a second child series "
                    f"(e.g. ts_corr(x, y, window))"
                )
            y = self.second_child.evaluate(dataset, cache)
            return fn(x, y, self.window)
        return fn(x, self.window)

    def depth(self) -> int:
        base = 1 + self.child.depth()
        if self.second_child is not None:
            return max(base, 1 + self.second_child.depth())
        return base

    def children(self) -> List[Node]:
        if self.second_child is not None:
            return [self.child, self.second_child]
        return [self.child]


# ---------------------------------------------------------------------------
# CrossSectionalNode
# ---------------------------------------------------------------------------

_CS_OPS = frozenset({
    "rank", "zscore", "scale", "ind_neutralize",
    "winsorize", "normalize",
})


class CrossSectionalNode(Node):
    """Cross-asset operator per row."""

    node_type = _CS

    def __init__(self, op: str, child: Node, **params) -> None:
        if op not in _CS_OPS:
            raise ValueError(f"Unknown CS operator: '{op}'")
        self.op     = op
        self.child  = child
        self.params = params

    def __repr__(self) -> str:
        if self.op == "winsorize":
            k = self.params.get("k", 3.0)
            if k != 3.0:
                return f"{self.op}({repr(self.child)},{k})"
        if self.op == "ind_neutralize":
            groups_node = self.params.get("groups_node")
            if groups_node is not None:
                return f"{self.op}({repr(self.child)},{repr(groups_node)})"
        return f"{self.op}({repr(self.child)})"

    def _compute(self, dataset: Dataset, cache: Cache) -> np.ndarray:
        from .fast_ops import FAST_CS_OPS
        x  = self.child.evaluate(dataset, cache)
        fn = FAST_CS_OPS[self.op]
        if self.op == "ind_neutralize":
            groups = self.params.get("groups")
            return fn(x, groups)
        if self.op == "winsorize":
            k = self.params.get("k", 3.0)
            return fn(x, k)
        return fn(x)

    def depth(self) -> int:
        return 1 + self.child.depth()

    def children(self) -> List[Node]:
        return [self.child]


# ---------------------------------------------------------------------------
# GroupNode  (within-group cross-sectional operations)
# ---------------------------------------------------------------------------

_GROUP_OPS = frozenset({
    "group_rank", "group_zscore", "group_mean", "group_neutralize",
})


class GroupNode(Node):
    """
    Cross-sectional operation within user-defined asset groups.

    Parameters
    ----------
    op          : one of _GROUP_OPS
    child       : expression to apply the op to
    group_field : name of the dataset field holding (N,) integer group labels
                  (auto-generated as 10 equal groups when not in dataset)
    """

    node_type = _GR

    def __init__(self, op: str, child: Node, group_field: str = "groups") -> None:
        if op not in _GROUP_OPS:
            raise ValueError(f"Unknown group operator: '{op}'")
        self.op          = op
        self.child       = child
        self.group_field = group_field

    def __repr__(self) -> str:
        return f"{self.op}({repr(self.child)},'{self.group_field}')"

    def _compute(self, dataset: Dataset, cache: Cache) -> np.ndarray:
        from .fast_ops import FAST_GROUP_OPS
        x  = self.child.evaluate(dataset, cache)
        fn = FAST_GROUP_OPS[self.op]

        raw = dataset.get(self.group_field)
        if raw is not None:
            arr = np.asarray(raw)
            # If stored as (T, N), take first row (groups are time-invariant)
            groups = arr[0] if arr.ndim == 2 else arr
        else:
            # Auto-generate: divide assets into 10 groups
            N = x.shape[1] if x.ndim == 2 else 1
            groups = np.arange(N) % 10

        return fn(x, groups.astype(int))

    def depth(self) -> int:
        return 1 + self.child.depth()

    def children(self) -> List[Node]:
        return [self.child]


# ---------------------------------------------------------------------------
# ArithmeticNode  — arithmetic, unary, logical, conditional
# ---------------------------------------------------------------------------

_BINOP_SYMBOLS = {
    "add": "+", "sub": "-", "mul": "*", "div": "/",
    "gt": ">", "lt": "<", "gte": ">=", "lte": "<=", "eq": "==", "ne": "!=",
}
_BINOP_OPS = set(_BINOP_SYMBOLS.keys())

_UNARY_OPS = frozenset({"log", "abs", "neg", "sqrt", "sign", "logical_not"})

_ADVANCED_OPS = frozenset({
    "signed_power",
    "if_else",          # if_else(cond, x, y)
    "where",            # alias for if_else
    "trade_when",       # trade_when(cond, x) → x if cond else 0
    "logical_and",      # logical_and(x, y)
    "logical_or",       # logical_or(x, y)
    "pow",              # pow(x, n)
    "max2",             # max(x, y)
    "min2",             # min(x, y)
    "weighted_sum",     # weighted_sum(x1, x2, ..., w1, w2, ...)
})


def _resolve_type(*nodes: Node) -> NodeType:
    """Propagate node type: GROUP > CS > TS > DATA > SCALAR."""
    types = {n.node_type for n in nodes if n.node_type not in (_ST,)}
    if _GR in types:
        return _GR
    if _CS in types:
        return _CS
    if _TS in types:
        return _TS
    if _DA in types:
        return _DA
    return _SC


class ArithmeticNode(Node):
    """
    Handles binary arithmetic, comparison, unary math, logical,
    signed_power, conditional, and weighted_sum operations.
    node_type is resolved from children.
    """

    def __init__(self, op: str, children: List[Node]) -> None:
        valid = _BINOP_OPS | _UNARY_OPS | _ADVANCED_OPS
        if op not in valid:
            raise ValueError(f"Unknown ArithmeticNode op: '{op}'")
        self.op       = op
        self._children = children
        self.node_type = _resolve_type(*children)

    def __repr__(self) -> str:
        op = self.op
        if op in _BINOP_SYMBOLS:
            sym = _BINOP_SYMBOLS[op]
            return f"({repr(self._children[0])}{sym}{repr(self._children[1])})"
        if op == "neg":
            return f"(-{repr(self._children[0])})"
        if op == "logical_not":
            return f"not({repr(self._children[0])})"
        if op in _UNARY_OPS:
            return f"{op}({repr(self._children[0])})"
        if op == "signed_power":
            return f"signed_power({repr(self._children[0])},{repr(self._children[1])})"
        if op == "if_else" or op == "where":
            a, b, c = self._children
            return f"if_else({repr(a)},{repr(b)},{repr(c)})"
        if op == "trade_when":
            a, b = self._children
            return f"trade_when({repr(a)},{repr(b)})"
        if op == "logical_and":
            return f"and({repr(self._children[0])},{repr(self._children[1])})"
        if op == "logical_or":
            return f"or({repr(self._children[0])},{repr(self._children[1])})"
        if op == "pow":
            return f"pow({repr(self._children[0])},{repr(self._children[1])})"
        if op == "max2":
            return f"max({repr(self._children[0])},{repr(self._children[1])})"
        if op == "min2":
            return f"min({repr(self._children[0])},{repr(self._children[1])})"
        if op == "weighted_sum":
            n    = len(self._children) // 2
            vals = self._children[:n]
            wgts = self._children[n:]
            pairs = ",".join(
                f"{repr(v)},{repr(w)}" for v, w in zip(vals, wgts)
            )
            return f"weighted_sum({pairs})"
        return f"{op}({','.join(repr(c) for c in self._children)})"

    def _compute(self, dataset: Dataset, cache: Cache) -> np.ndarray:
        evaled = [c.evaluate(dataset, cache) for c in self._children]
        op = self.op

        if op == "add":   return evaled[0] + evaled[1]
        if op == "sub":   return evaled[0] - evaled[1]
        if op == "mul":   return evaled[0] * evaled[1]
        if op == "div":
            denom = evaled[1]
            safe  = np.where(np.abs(denom) < 1e-8, np.nan, denom)
            return evaled[0] / safe
        if op == "gt":    return (evaled[0] >  evaled[1]).astype(float)
        if op == "lt":    return (evaled[0] <  evaled[1]).astype(float)
        if op == "gte":   return (evaled[0] >= evaled[1]).astype(float)
        if op == "lte":   return (evaled[0] <= evaled[1]).astype(float)
        if op == "eq":    return (evaled[0] == evaled[1]).astype(float)
        if op == "ne":    return (evaled[0] != evaled[1]).astype(float)
        if op == "log":   return np.log(np.where(evaled[0] > 0, evaled[0], np.nan))
        if op == "abs":   return np.abs(evaled[0])
        if op == "neg":   return -evaled[0]
        if op == "sqrt":  return np.sqrt(np.where(evaled[0] >= 0, evaled[0], np.nan))
        if op == "sign":  return np.sign(evaled[0])
        if op == "signed_power":
            x, p = evaled[0], evaled[1]
            return np.sign(x) * np.abs(x) ** p
        if op in ("if_else", "where"):
            cond = evaled[0].astype(bool)
            return np.where(cond, evaled[1], evaled[2])
        if op == "trade_when":
            cond = evaled[0].astype(bool)
            return np.where(cond, evaled[1], 0.0)
        if op == "logical_and":
            return (evaled[0].astype(bool) & evaled[1].astype(bool)).astype(float)
        if op == "logical_or":
            return (evaled[0].astype(bool) | evaled[1].astype(bool)).astype(float)
        if op == "logical_not":
            return (~evaled[0].astype(bool)).astype(float)
        if op == "pow":
            return evaled[0] ** evaled[1]
        if op == "max2":
            return np.maximum(evaled[0], evaled[1])
        if op == "min2":
            return np.minimum(evaled[0], evaled[1])
        if op == "weighted_sum":
            n    = len(evaled) // 2
            vals = evaled[:n]
            wgts = evaled[n:]
            return sum(v * w for v, w in zip(vals, wgts))

        raise NotImplementedError(f"ArithmeticNode._compute: unknown op '{op}'")

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self._children), default=0)

    def children(self) -> List[Node]:
        return list(self._children)
