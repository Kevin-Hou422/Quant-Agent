"""
Random Alpha Generator.

Generates syntactically valid alpha expression trees using a
probabilistic grammar over the operator set defined in ``ast.py``.
"""

from __future__ import annotations

import random
from typing import List, Optional

from .ast import (
    Node,
    ARITHMETIC_OPS,
    UNARY_OPS,
    TIME_SERIES_OPS,
    CROSS_SECTIONAL_OPS,
    VALID_DATA_FIELDS,
)


# ---------------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------------

# Windows to sample from for time-series ops
TS_WINDOWS = [3, 5, 10, 20, 40, 60]

# Fields available as leaves
DATA_FIELDS = list(VALID_DATA_FIELDS - {"vwap"}) + ["vwap"]  # vwap weighted slightly

# Operator categories and their sampling weights at each depth level.
# Higher depth -> more likely to pick a leaf; lower depth -> more complex ops.
_BINARY_OPS  = sorted(ARITHMETIC_OPS)
_UNARY_OPS   = sorted(UNARY_OPS)
_TS_OPS_1    = sorted(TIME_SERIES_OPS - {"ts_corr", "ts_cov"})  # single-child
_TS_OPS_2    = sorted({"ts_corr", "ts_cov"})                    # two-child
_CS_OPS      = sorted(CROSS_SECTIONAL_OPS)

# Safe unary ops (never produce invalid intermediate values when chained)
_SAFE_UNARY = {"abs", "neg", "sign"}
# Ops that require positive input
_POSITIVE_REQUIRED = {"log", "sqrt"}
# Safe child ops for log/sqrt (guaranteed non-negative output)
_NON_NEG_OPS = {"abs", "ts_max", "ts_min", "rank", "zscore"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _random_window() -> int:
    return random.choice(TS_WINDOWS)


def _random_field() -> str:
    return random.choice(DATA_FIELDS)


def _can_be_denominator(node: Node) -> bool:
    """Heuristic: is this node unlikely to be constantly zero?"""
    return node.op not in {"neg", "demean", "zscore"}


def _is_non_negative(node: Node) -> bool:
    """Heuristic: does this node produce non-negative values?"""
    return node.op in _NON_NEG_OPS or (
        node.op == "data" and node.params.get("field") in {"volume", "vwap"}
    )


# ---------------------------------------------------------------------------
# Core recursive generator
# ---------------------------------------------------------------------------

def _generate(
    depth: int,
    max_depth: int,
    force_positive: bool = False,
) -> Node:
    """
    Recursively generate a random alpha AST node.

    Parameters
    ----------
    depth       : current depth (0 = root).
    max_depth   : maximum allowed depth.
    force_positive : if True, the generated sub-tree must be non-negative.
    """
    remaining = max_depth - depth

    # At leaf level (or forced early termination) → return a data node
    if remaining <= 0 or (remaining == 1 and random.random() < 0.6):
        field = _random_field()
        if force_positive and field not in {"volume", "vwap"}:
            field = random.choice(["volume", "vwap"])
        return Node.data(field)

    # Probability distribution over op categories
    # As we get deeper, shift toward leaves / simpler ops
    depth_factor = depth / max_depth  # 0 at root, 1 at max
    weights = {
        "binary": max(0.05, 0.35 - 0.2 * depth_factor),
        "unary":  0.15,
        "ts1":    max(0.05, 0.25 - 0.1 * depth_factor),
        "ts2":    max(0.0,  0.05 - 0.05 * depth_factor),
        "cs":     0.15,
        "leaf":   min(0.9, 0.05 + 0.5 * depth_factor),
    }
    total = sum(weights.values())
    categories = list(weights.keys())
    probs = [weights[c] / total for c in categories]

    category = random.choices(categories, weights=probs, k=1)[0]

    # --- Leaf ---
    if category == "leaf":
        field = _random_field()
        if force_positive and field not in {"volume", "vwap"}:
            field = random.choice(["volume", "vwap"])
        return Node.data(field)

    # --- Binary arithmetic ---
    if category == "binary":
        op = random.choice(_BINARY_OPS)
        left = _generate(depth + 1, max_depth)
        if op == "div":
            # Ensure denominator avoids zero
            right = _generate(depth + 1, max_depth, force_positive=True)
            # Wrap denominator in abs to be safe
            if not _is_non_negative(right):
                right = Node.unary("abs", right)
        else:
            right = _generate(depth + 1, max_depth)
        return Node.binary(op, left, right)

    # --- Unary ---
    if category == "unary":
        op = random.choice(_UNARY_OPS)
        if op in _POSITIVE_REQUIRED:
            child = _generate(depth + 1, max_depth, force_positive=True)
            if not _is_non_negative(child):
                child = Node.unary("abs", child)
        else:
            child = _generate(depth + 1, max_depth)
        return Node.unary(op, child)

    # --- Time-series (1 child) ---
    if category == "ts1":
        op = random.choice(_TS_OPS_1)
        child = _generate(depth + 1, max_depth)
        return Node.ts(op, child, window=_random_window())

    # --- Time-series (2 children) ---
    if category == "ts2":
        op = random.choice(_TS_OPS_2)
        left = _generate(depth + 1, max_depth)
        right = _generate(depth + 1, max_depth)
        return Node.ts2(op, left, right, window=_random_window())

    # --- Cross-sectional ---
    if category == "cs":
        op = random.choice(_CS_OPS)
        child = _generate(depth + 1, max_depth)
        params: dict = {}
        if op == "winsorize":
            params["k"] = random.choice([2.0, 2.5, 3.0])
        return Node.cs(op, child, **params)

    # Fallback
    return Node.data(_random_field())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_random_alpha(
    depth: int = 4,
    seed: Optional[int] = None,
) -> Node:
    """
    Generate a single random alpha expression tree.

    Parameters
    ----------
    depth : int
        Maximum depth of the expression tree (default 4).
    seed : int | None
        Optional random seed for reproducibility.

    Returns
    -------
    Node
        Root node of the generated alpha expression.
    """
    if seed is not None:
        random.seed(seed)
    return _generate(depth=0, max_depth=depth)


def generate_n_alphas(
    n: int,
    depth: int = 4,
    seed: Optional[int] = None,
    deduplicate: bool = True,
) -> List[Node]:
    """
    Generate ``n`` random alpha expressions.

    Parameters
    ----------
    n           : number of alphas to generate.
    depth       : maximum depth of each tree.
    seed        : optional base random seed.
    deduplicate : if True, discard duplicate expressions (by repr).

    Returns
    -------
    List[Node]
    """
    if seed is not None:
        random.seed(seed)

    alphas: List[Node] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = n * 20  # cap to avoid infinite loops on tiny search spaces

    while len(alphas) < n and attempts < max_attempts:
        attempts += 1
        node = _generate(depth=0, max_depth=depth)
        key = repr(node)
        if deduplicate and key in seen:
            continue
        seen.add(key)
        alphas.append(node)

    if len(alphas) < n:
        import warnings
        warnings.warn(
            f"Could only generate {len(alphas)}/{n} unique alphas after "
            f"{max_attempts} attempts."
        )

    return alphas
