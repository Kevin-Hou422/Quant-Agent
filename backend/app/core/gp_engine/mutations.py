"""
mutations.py — GP Structural Mutation & Crossover Operators.

Eleven operations — 4 classic + 7 structural (PROMPT 2):

  Classic (parameter/topology):
    point_mutation      : swap one operator for another of the same type
    hoist_mutation      : replace a subtree with one of its children (simplify)
    param_mutation      : adjust TS window parameter ±20 %
    subtree_crossover   : swap type-compatible subtrees between two trees

  Structural (NEW — discover new alpha structures, not just tweak windows):
    wrap_rank           : x → rank(x) or zscore(x)
    add_ts_smoothing    : x → ts_mean(x, w) / ts_decay_linear(x, w) / ts_zscore(x, w)
    add_condition       : wrap with a momentum / volume / trend condition
    add_volume_filter   : gate signal on volume-above-ADV condition
    combine_signals     : signal = alpha1 OP alpha2  (arithmetic combination)
    replace_subtree     : swap an internal subtree with a freshly generated one
    add_operator        : wrap root or subtree with unary / arithmetic layer

All operations:
  - Preserve NodeType constraints (no illegal cross-type rewiring)
  - Return NEW trees (deep-copy semantics — originals untouched)
  - Validate output; fall back to deepcopy of original on failure
"""

from __future__ import annotations

import copy
import random
from typing import List, Optional, Tuple

from ..alpha_engine.typed_nodes import (
    Node, NodeType,
    DataNode, ScalarNode, StringLiteralNode,
    TimeSeriesNode, CrossSectionalNode, ArithmeticNode,
    _TS_OPS, _CS_OPS,
)
from ..alpha_engine.validator import AlphaValidator

_validator = AlphaValidator()

# ---------------------------------------------------------------------------
# TS / CS operator menus
# ---------------------------------------------------------------------------

# Single-input TS ops (exclude two-input ops and entropy which is slow)
_TS_LIST = sorted(
    _TS_OPS
    - {"ts_corr", "ts_cov", "ts_entropy"}
)

# CS ops that are safe to use as wrappers
_CS_LIST = sorted(_CS_OPS - {"ind_neutralize"})

# TS windows to sample from
_TS_WINDOWS = [3, 5, 10, 20, 40, 60]

# Field names available as data leaves
_DATA_FIELDS = ["close", "open", "high", "low", "volume", "returns", "vwap"]

# Smoothing ops for add_ts_smoothing (subset of _TS_LIST)
_SMOOTH_OPS = ["ts_mean", "ts_decay_linear", "ts_zscore", "ts_rank", "ts_std"]

# Wrapping CS ops for wrap_rank
_WRAP_CS_OPS = ["rank", "zscore", "scale"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_children(node: Node) -> list:
    """Handle both ast.Node (children is a list field) and typed_nodes.Node."""
    ch = node.children
    return ch if isinstance(ch, list) else ch()


def _collect_nodes(root: Node) -> List[Node]:
    """BFS — collect all nodes in tree (including root)."""
    result, queue = [], [root]
    while queue:
        n = queue.pop(0)
        result.append(n)
        queue.extend(_get_children(n))
    return result


def _replace_node(root: Node, target: Node, replacement: Node) -> Node:
    """
    Deep-copy root and replace the first occurrence of target (by object id)
    with a deep-copy of replacement.  If target IS root, return a copy of
    replacement directly.
    """
    if root is target:
        return copy.deepcopy(replacement)
    root_copy = copy.deepcopy(root)
    _replace_inplace(root_copy, id(target), copy.deepcopy(replacement))
    return root_copy


def _replace_inplace(node: Node, target_id: int, replacement: Node) -> bool:
    """In-place replacement helper (run AFTER deep-copy)."""
    if isinstance(node, TimeSeriesNode):
        if id(node.child) == target_id:
            node.child = replacement
            return True
        if node.second_child is not None:
            if id(node.second_child) == target_id:
                node.second_child = replacement
                return True
            if _replace_inplace(node.second_child, target_id, replacement):
                return True
        return _replace_inplace(node.child, target_id, replacement)
    if isinstance(node, CrossSectionalNode):
        if id(node.child) == target_id:
            node.child = replacement
            return True
        return _replace_inplace(node.child, target_id, replacement)
    if isinstance(node, ArithmeticNode):
        for i, ch in enumerate(node._children):
            if id(ch) == target_id:
                node._children[i] = replacement
                return True
        for ch in node._children:
            if _replace_inplace(ch, target_id, replacement):
                return True
    # Fallback for ast.Node (children is a list field)
    children_attr = getattr(node, "children", None)
    if isinstance(children_attr, list):
        for i, ch in enumerate(children_attr):
            if id(ch) == target_id:
                node.children[i] = replacement
                return True
        for ch in children_attr:
            if _replace_inplace(ch, target_id, replacement):
                return True
    return False


def _node_type(node: Node):
    """Return NodeType (typed_nodes) or a string category (ast.Node fallback)."""
    nt = getattr(node, "node_type", None)
    if nt is not None:
        return nt
    op = getattr(node, "op", "data")
    if op in _TS_OPS:
        return "ts"
    if op in _CS_OPS:
        return "cs"
    return "other"


def _try_validate(node: Node) -> bool:
    try:
        _validator.validate(node)
        return True
    except Exception:
        return False


def _tree_depth(node: Node) -> int:
    """Return the depth of a typed_nodes or ast.Node tree."""
    if hasattr(node, "depth") and callable(node.depth):
        return node.depth()
    ch = _get_children(node)
    if not ch:
        return 0
    return 1 + max(_tree_depth(c) for c in ch)


# ---------------------------------------------------------------------------
# Random typed node generator (for structural mutations)
# ---------------------------------------------------------------------------

def _make_data_node() -> DataNode:
    return DataNode(random.choice(_DATA_FIELDS))


def _generate_typed_node(max_depth: int = 2) -> Node:
    """
    Generate a small random typed_nodes expression tree.
    Used by replace_subtree and combine_signals when no second parent is available.
    """
    if max_depth <= 0 or random.random() < 0.35:
        return _make_data_node()

    roll = random.random()

    if roll < 0.40:
        # Time-series wrap
        op     = random.choice(_TS_LIST)
        child  = _generate_typed_node(max_depth - 1)
        window = random.choice(_TS_WINDOWS)
        try:
            return TimeSeriesNode(op, child, window)
        except Exception:
            return _make_data_node()

    if roll < 0.60:
        # Cross-sectional wrap
        op    = random.choice(_WRAP_CS_OPS)
        child = _generate_typed_node(max_depth - 1)
        return CrossSectionalNode(op, child)

    if roll < 0.80:
        # Binary arithmetic
        op  = random.choice(["add", "sub", "mul"])
        l   = _generate_typed_node(max_depth - 1)
        r   = _generate_typed_node(max_depth - 1)
        return ArithmeticNode(op, [l, r])

    return _make_data_node()


def _make_momentum_condition() -> ArithmeticNode:
    """Build a simple trend/momentum boolean condition."""
    choices = [
        # positive recent delta
        lambda: ArithmeticNode("gt", [
            TimeSeriesNode("ts_delta", DataNode("close"), random.choice([1, 5, 10])),
            ScalarNode(0),
        ]),
        # price above rolling mean
        lambda: ArithmeticNode("gt", [
            DataNode("close"),
            TimeSeriesNode("ts_mean", DataNode("close"), random.choice([10, 20])),
        ]),
        # positive returns
        lambda: ArithmeticNode("gt", [
            DataNode("returns"),
            ScalarNode(0),
        ]),
        # recent mean above longer mean (golden cross)
        lambda: ArithmeticNode("gt", [
            TimeSeriesNode("ts_mean", DataNode("close"), random.choice([5, 10])),
            TimeSeriesNode("ts_mean", DataNode("close"), random.choice([20, 40])),
        ]),
        # high > previous high (breakout)
        lambda: ArithmeticNode("gt", [
            DataNode("high"),
            TimeSeriesNode("ts_max", DataNode("high"), random.choice([10, 20])),
        ]),
    ]
    return random.choice(choices)()


def _make_volume_condition() -> ArithmeticNode:
    """Build a volume-above-ADV condition."""
    adv_window = random.choice([10, 20])
    return ArithmeticNode("gt", [
        DataNode("volume"),
        TimeSeriesNode("ts_mean", DataNode("volume"), adv_window),
    ])


# ---------------------------------------------------------------------------
# 1. point_mutation  (classic — operator swap, same type)
# ---------------------------------------------------------------------------

def point_mutation(root: Node) -> Node:
    """Replace one TS or CS operator with a different one of the same type."""
    nodes    = _collect_nodes(root)
    ts_nodes = [n for n in nodes
                if isinstance(n, TimeSeriesNode) or getattr(n, "op", "") in _TS_OPS]
    cs_nodes = [n for n in nodes
                if isinstance(n, CrossSectionalNode) or getattr(n, "op", "") in _CS_OPS]

    candidates: List[Tuple[str, Node]] = (
        [("ts", n) for n in ts_nodes] + [("cs", n) for n in cs_nodes]
    )
    if not candidates:
        return copy.deepcopy(root)

    kind, target = random.choice(candidates)
    new_root     = copy.deepcopy(root)
    all_new      = _collect_nodes(new_root)
    target_repr  = repr(target)
    matched      = [n for n in all_new if repr(n) == target_repr]
    if not matched:
        return new_root

    node_to_mutate = matched[0]
    if kind == "ts":
        ts_op = getattr(node_to_mutate, "op", None)
        if ts_op in _TS_OPS:
            ops = [op for op in _TS_LIST if op != ts_op]
            if ops:
                node_to_mutate.op = random.choice(ops)
    elif kind == "cs":
        cs_op = getattr(node_to_mutate, "op", None)
        if cs_op in _CS_OPS:
            ops = [op for op in _CS_LIST if op != cs_op]
            if ops:
                node_to_mutate.op = random.choice(ops)
    return new_root


# ---------------------------------------------------------------------------
# 2. hoist_mutation  (classic — simplify by hoisting a child)
# ---------------------------------------------------------------------------

def hoist_mutation(root: Node) -> Node:
    """Replace a subtree with one of its children (reduces depth)."""
    nodes       = _collect_nodes(root)
    has_children = [n for n in nodes if _get_children(n)]
    if not has_children:
        return copy.deepcopy(root)

    target   = random.choice(has_children)
    children = _get_children(target)
    if not children:
        return copy.deepcopy(root)

    hoisted  = random.choice(children)
    new_root = _replace_node(root, target, hoisted)

    if _try_validate(new_root):
        return new_root
    return copy.deepcopy(root)


# ---------------------------------------------------------------------------
# 3. param_mutation  (classic — adjust TS window ±20 %)
# ---------------------------------------------------------------------------

def param_mutation(root: Node) -> Node:
    """Adjust one TS window parameter by ±20 %, clamped to [2, 60]."""
    new_root = copy.deepcopy(root)
    nodes    = _collect_nodes(new_root)
    ts_nodes = [n for n in nodes
                if isinstance(n, TimeSeriesNode) or getattr(n, "op", "") in _TS_OPS]
    if not ts_nodes:
        return new_root

    target = random.choice(ts_nodes)
    if hasattr(target, "window"):
        old_w         = target.window
        delta         = random.uniform(-0.2, 0.2)
        target.window = max(2, min(60, int(old_w * (1 + delta))))
    elif hasattr(target, "params") and "window" in target.params:
        old_w                   = target.params["window"]
        delta                   = random.uniform(-0.2, 0.2)
        target.params["window"] = max(2, min(60, int(old_w * (1 + delta))))
    return new_root


# ---------------------------------------------------------------------------
# 4. subtree_crossover  (classic — swap compatible subtrees between two trees)
# ---------------------------------------------------------------------------

def subtree_crossover(root1: Node, root2: Node) -> Tuple[Node, Node]:
    """
    Swap type-compatible random subtrees between two parent trees.
    Returns two children; falls back to copies of parents on failure.
    """
    nodes1, nodes2 = _collect_nodes(root1), _collect_nodes(root2)

    def by_type(nodes):
        d: dict = {}
        for n in nodes:
            d.setdefault(_node_type(n), []).append(n)
        return d

    g1, g2        = by_type(nodes1), by_type(nodes2)
    common_types  = set(g1.keys()) & set(g2.keys())
    if not common_types:
        return copy.deepcopy(root1), copy.deepcopy(root2)

    chosen_type = random.choice(list(common_types))
    swap1       = random.choice(g1[chosen_type])
    swap2       = random.choice(g2[chosen_type])

    new_root1 = _replace_node(root1, swap1, swap2)
    new_root2 = _replace_node(root2, swap2, swap1)

    if _try_validate(new_root1) and _try_validate(new_root2):
        return new_root1, new_root2
    return copy.deepcopy(root1), copy.deepcopy(root2)


# ===========================================================================
# STRUCTURAL MUTATIONS (PROMPT 2)
# ===========================================================================


# ---------------------------------------------------------------------------
# 5. wrap_rank  — x → rank(x) or zscore(x)
# ---------------------------------------------------------------------------

def wrap_rank(root: Node) -> Node:
    """
    Wrap a random non-leaf node (or the whole root) with rank() or zscore().

    Before:  ts_delta(close, 5)
    After:   rank(ts_delta(close, 5))
    """
    op = random.choice(_WRAP_CS_OPS)

    # Try wrapping the whole root first (most impactful)
    if random.random() < 0.6:
        try:
            wrapped = CrossSectionalNode(op, copy.deepcopy(root))
            if _try_validate(wrapped):
                return wrapped
        except Exception:
            pass

    # Otherwise wrap a random internal node
    nodes      = _collect_nodes(root)
    candidates = [
        n for n in nodes
        if not isinstance(n, (DataNode, ScalarNode, StringLiteralNode))
        and _node_type(n) is not NodeType.CROSS_SECTIONAL
    ]
    if not candidates:
        candidates = [n for n in nodes if isinstance(n, (DataNode, TimeSeriesNode))]
    if not candidates:
        return copy.deepcopy(root)

    target = random.choice(candidates)
    try:
        wrapped_target = CrossSectionalNode(op, copy.deepcopy(target))
        new_root       = _replace_node(root, target, wrapped_target)
        if _try_validate(new_root):
            return new_root
    except Exception:
        pass

    return copy.deepcopy(root)


# ---------------------------------------------------------------------------
# 6. add_ts_smoothing  — x → ts_mean(x, w) / ts_decay_linear(x, w)
# ---------------------------------------------------------------------------

def add_ts_smoothing(root: Node) -> Node:
    """
    Wrap a DATA or TS node with a rolling smoothing operator.

    Before:  rank(close)
    After:   rank(ts_mean(close, 10))
    """
    nodes      = _collect_nodes(root)
    # Prefer DATA or simple TS nodes as targets (avoid re-wrapping deeply)
    candidates = [
        n for n in nodes
        if isinstance(n, DataNode)
        or (isinstance(n, TimeSeriesNode) and n.window <= 20)
    ]
    if not candidates:
        candidates = [n for n in nodes if not isinstance(n, (ScalarNode, StringLiteralNode))]
    if not candidates:
        return copy.deepcopy(root)

    target = random.choice(candidates)
    op     = random.choice(_SMOOTH_OPS)
    window = random.choice(_TS_WINDOWS)

    try:
        wrapped_target = TimeSeriesNode(op, copy.deepcopy(target), window)
        new_root       = _replace_node(root, target, wrapped_target)
        if _try_validate(new_root):
            return new_root
    except Exception:
        pass

    return copy.deepcopy(root)


# ---------------------------------------------------------------------------
# 7. add_condition  — wrap with a momentum / trend conditional
# ---------------------------------------------------------------------------

def add_condition(root: Node) -> Node:
    """
    Add a conditional filter to the signal.

    Examples:
        x → trade_when(close > ts_mean(close, 20), x)
        x → trade_when(ts_delta(close, 5) > 0, x)
        x → if_else(returns > 0, x, -x)   (amplify directional bet)
    """
    cond = _make_momentum_condition()
    root_copy = copy.deepcopy(root)

    variant = random.choice(["trade_when", "if_else_zero", "if_else_flip"])

    try:
        if variant == "trade_when":
            result = ArithmeticNode("trade_when", [cond, root_copy])
        elif variant == "if_else_zero":
            result = ArithmeticNode("if_else", [cond, root_copy, ScalarNode(0)])
        else:  # if_else_flip: go long if condition, short otherwise
            result = ArithmeticNode("if_else", [cond, root_copy,
                                                ArithmeticNode("neg", [copy.deepcopy(root_copy)])])

        if _try_validate(result):
            return result
    except Exception:
        pass

    return copy.deepcopy(root)


# ---------------------------------------------------------------------------
# 8. add_volume_filter  — gate signal on volume > ADV
# ---------------------------------------------------------------------------

def add_volume_filter(root: Node) -> Node:
    """
    Multiply/gate the signal by a volume-above-ADV condition.

    Before:  rank(ts_delta(close, 5))
    After:   trade_when(volume > ts_mean(volume, 20), rank(ts_delta(close, 5)))

    Alternative:
    After:   rank(ts_delta(close, 5)) * (volume / ts_mean(volume, 20))
    """
    vol_cond  = _make_volume_condition()
    root_copy = copy.deepcopy(root)

    variant = random.choice(["gate", "scale_by_vol"])

    try:
        if variant == "gate":
            result = ArithmeticNode("trade_when", [vol_cond, root_copy])
        else:
            # Scale by relative volume (volume / ADV)
            adv_window = random.choice([10, 20])
            rel_vol    = ArithmeticNode("div", [
                DataNode("volume"),
                TimeSeriesNode("ts_mean", DataNode("volume"), adv_window),
            ])
            result = ArithmeticNode("mul", [root_copy, CrossSectionalNode("rank", rel_vol)])

        if _try_validate(result):
            return result
    except Exception:
        pass

    return copy.deepcopy(root)


# ---------------------------------------------------------------------------
# 9. combine_signals  — alpha = root OP other_signal
# ---------------------------------------------------------------------------

def combine_signals(
    root: Node,
    other_root: Optional[Node] = None,
) -> Node:
    """
    Combine the current signal with a second signal via arithmetic.

    child = root  OP  other_signal
    where OP ∈ {+, -, *, /}

    If other_root is None, a small random signal is generated.
    Depth guard: skip if combined depth would exceed 8.
    """
    max_combined_depth = 8

    root_depth = _tree_depth(root)
    if root_depth >= max_combined_depth - 1:
        # Tree is already very deep — fall back to a simpler combination
        other = _make_data_node()
    elif other_root is not None:
        other_depth = _tree_depth(other_root)
        # If other tree also deep, use its shallow representative (hoist a leaf)
        if other_depth >= 4:
            leaves = [n for n in _collect_nodes(other_root)
                      if isinstance(n, DataNode)]
            other = copy.deepcopy(random.choice(leaves)) if leaves else _make_data_node()
        else:
            other = copy.deepcopy(other_root)
    else:
        other = _generate_typed_node(max_depth=2)

    op = random.choice(["add", "sub", "mul"])

    try:
        if op == "mul":
            # Prefer rank-scaling for multiplication (bounds the result)
            if not isinstance(other, (ScalarNode, DataNode)):
                other = CrossSectionalNode("rank", other)
            combined = ArithmeticNode("mul", [copy.deepcopy(root), other])
        else:
            combined = ArithmeticNode(op, [copy.deepcopy(root), other])

        if _try_validate(combined):
            return combined
    except Exception:
        pass

    return copy.deepcopy(root)


# ---------------------------------------------------------------------------
# 10. replace_subtree  — swap an internal subtree with a generated one
# ---------------------------------------------------------------------------

def replace_subtree(root: Node) -> Node:
    """
    Replace a randomly chosen internal node with a freshly generated
    expression of similar complexity.
    """
    nodes      = _collect_nodes(root)
    internals  = [n for n in nodes
                  if not isinstance(n, (DataNode, ScalarNode, StringLiteralNode))
                  and _get_children(n)]
    if not internals:
        # If no internals, replace whole root
        new_tree = _generate_typed_node(max_depth=3)
        if _try_validate(new_tree):
            return new_tree
        return copy.deepcopy(root)

    target     = random.choice(internals)
    target_d   = _tree_depth(target)
    new_subtree = _generate_typed_node(max_depth=max(1, target_d))

    try:
        new_root = _replace_node(root, target, new_subtree)
        if _try_validate(new_root):
            return new_root
    except Exception:
        pass

    return copy.deepcopy(root)


# ---------------------------------------------------------------------------
# 11. add_operator  — wrap root/subtree with a unary or arithmetic layer
# ---------------------------------------------------------------------------

def add_operator(root: Node) -> Node:
    """
    Add an operator layer to the root or a random subtree.

    Examples:
        x → sign(x)
        x → abs(x)
        x → signed_power(x, 2)
        x → x * rank(x)         (self-rank interaction)
        x → rank(x) - ts_mean(rank(x), 10)  (rank deviation)
    """
    variant = random.choice([
        "unary_sign",
        "unary_abs",
        "signed_power",
        "self_rank",
        "rank_deviation",
        "scaled",
    ])
    root_copy = copy.deepcopy(root)

    try:
        if variant == "unary_sign":
            result = ArithmeticNode("sign", [root_copy])

        elif variant == "unary_abs":
            result = ArithmeticNode("abs", [root_copy])

        elif variant == "signed_power":
            p      = ScalarNode(random.choice([0.5, 1.5, 2.0, 3.0]))
            result = ArithmeticNode("signed_power", [root_copy, p])

        elif variant == "self_rank":
            # x * rank(x) — amplify high-ranked assets
            ranked = CrossSectionalNode("rank", copy.deepcopy(root_copy))
            result = ArithmeticNode("mul", [root_copy, ranked])

        elif variant == "rank_deviation":
            # rank(x) - ts_mean(rank(x), w) — de-trend the rank signal
            w        = random.choice([5, 10, 20])
            ranked   = CrossSectionalNode("rank", root_copy)
            ma_rank  = TimeSeriesNode("ts_mean", CrossSectionalNode("rank", copy.deepcopy(root_copy)), w)
            result   = ArithmeticNode("sub", [ranked, ma_rank])

        else:  # scaled
            # scale(x) — L1-normalise for portfolio neutrality
            result = CrossSectionalNode("scale", root_copy)

        if _try_validate(result):
            return result
    except Exception:
        pass

    return copy.deepcopy(root)
