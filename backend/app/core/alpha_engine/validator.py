"""
Static AST Validators for the Alpha DSL Engine.

Four composable validators run before any computation:

  WindowValidator   — window size in [1, 252]
  LookAheadValidator — no future-data access (window ≥ 1 for delay/delta)
  DepthValidator    — nesting depth ≤ 10
  SparseFieldValidator — no short-window dispersion ops on step (quarterly/event) fields

AlphaValidator is the composite that runs all of them and collects every
error before raising a single ValidationError with a full report.
"""

from __future__ import annotations

import warnings
from typing import List

from .fields import STEP_FIELDS
from .typed_nodes import Node, TimeSeriesNode, DataNode


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when an alpha expression fails static validation."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        bullet = "\n  • ".join(errors)
        super().__init__(f"Alpha validation failed:\n  • {bullet}")


# ---------------------------------------------------------------------------
# Base validator
# ---------------------------------------------------------------------------

class _BaseValidator:
    """Walk the AST recursively, collect errors into self._errors."""

    def validate(self, node: Node) -> None:
        self._errors: List[str] = []
        self._walk(node)
        if self._errors:
            raise ValidationError(self._errors)

    def collect(self, node: Node) -> List[str]:
        """Return errors without raising."""
        self._errors = []
        self._walk(node)
        return list(self._errors)

    def _walk(self, node: Node) -> None:
        self._check(node)
        for child in node.children():
            self._walk(child)

    def _check(self, node: Node) -> None:  # noqa: B027
        pass  # override in subclasses


# ---------------------------------------------------------------------------
# WindowValidator
# ---------------------------------------------------------------------------

class WindowValidator(_BaseValidator):
    """
    Ensures time-series window sizes are in the valid range [1, 252].
    Also warns when a child TS window > parent TS window (redundant computation).
    """

    MIN_WINDOW = 1
    MAX_WINDOW = 252

    def _check(self, node: Node) -> None:
        if not isinstance(node, TimeSeriesNode):
            return
        w = node.window
        if w < self.MIN_WINDOW:
            self._errors.append(
                f"'{node.op}': window={w} is invalid (min={self.MIN_WINDOW}). "
                "Negative or zero window sizes cause look-ahead bias."
            )
        elif w > self.MAX_WINDOW:
            self._errors.append(
                f"'{node.op}': window={w} exceeds maximum allowed "
                f"window of {self.MAX_WINDOW} trading days."
            )
        else:
            # Warn if child TS window > this node's window (potential redundancy)
            child = node.child
            if isinstance(child, TimeSeriesNode) and child.window > w:
                warnings.warn(
                    f"'{node.op}(window={w})' wraps '{child.op}(window={child.window})'. "
                    "Child window is larger than parent window — this may produce "
                    "redundant rolling computation.",
                    stacklevel=4,
                )


# ---------------------------------------------------------------------------
# LookAheadValidator
# ---------------------------------------------------------------------------

class LookAheadValidator(_BaseValidator):
    """
    Prevents future-data access:
    - ts_delay / ts_delta must have window ≥ 1.
    - DataNode field names starting with 'future_' are forbidden.
    """

    _DELAY_OPS = {"ts_delay", "ts_delta"}

    def _check(self, node: Node) -> None:
        if isinstance(node, TimeSeriesNode):
            if node.op in self._DELAY_OPS and node.window < 1:
                self._errors.append(
                    f"'{node.op}': window={node.window} — a window < 1 accesses "
                    "future data (look-ahead bias). Use window ≥ 1."
                )
        elif isinstance(node, DataNode):
            if node.field.startswith("future_"):
                self._errors.append(
                    f"DataNode field '{node.field}' starts with 'future_' — "
                    "this naming convention is reserved for look-ahead data "
                    "and is forbidden in live alphas."
                )


# ---------------------------------------------------------------------------
# DepthValidator
# ---------------------------------------------------------------------------

class DepthValidator(_BaseValidator):
    """Raises if the AST nesting depth exceeds MAX_DEPTH."""

    MAX_DEPTH = 10

    def validate(self, node: Node) -> None:
        d = node.depth()
        if d > self.MAX_DEPTH:
            raise ValidationError([
                f"Expression depth {d} exceeds maximum allowed depth of "
                f"{self.MAX_DEPTH}. Simplify the alpha expression."
            ])

    def collect(self, node: Node) -> List[str]:
        d = node.depth()
        if d > self.MAX_DEPTH:
            return [
                f"Expression depth {d} exceeds maximum allowed depth of "
                f"{self.MAX_DEPTH}."
            ]
        return []


# ---------------------------------------------------------------------------
# SparseFieldValidator（Phase 10.1）
# ---------------------------------------------------------------------------

class SparseFieldValidator(_BaseValidator):
    """
    拒绝在**阶梯字段**上做短窗口的离散度类时序运算。

    季频字段（roe / gross_margin / asset_growth / sue）在两次财报之间是常数，
    `days_since_earnings` 是机械递增的计数器。窗口短于 MIN_WINDOW（≈ 两个季度）时，
    窗口里通常一次更新都没有：ts_std / ts_var 处处为 0（做分母 → inf），
    ts_zscore 是 0/0，ts_corr / ts_cov 没有定义，ts_rank 全是并列。

    只看**叶子全是阶梯字段**的子树：`ts_std(roe * returns, 20)` 每天都在变，不拦。
    估值字段（earnings_yield / book_to_price）分母是市值，每天都变，也不拦。
    水平类运算（ts_mean / ts_delta / ts_delay …）在阶梯上有明确含义，不拦。
    """

    MIN_WINDOW = 126
    _DISPERSION_OPS = frozenset({"ts_std", "ts_var", "ts_zscore", "ts_skew", "ts_kurt",
                                 "ts_entropy", "ts_rank", "ts_corr", "ts_cov"})

    def _check(self, node: Node) -> None:
        if not isinstance(node, TimeSeriesNode) or node.op not in self._DISPERSION_OPS:
            return
        if node.window >= self.MIN_WINDOW:
            return
        offending: set = set()
        for operand in node.children():          # 双输入算子：任一侧是纯阶梯就不成立
            leaves = _leaf_fields(operand)
            if leaves and leaves <= STEP_FIELDS:
                offending |= leaves
        if offending:
            self._errors.append(
                f"'{node.op}': window={node.window} on {sorted(offending)} — these fields only "
                f"change at filings (about every 63 trading days), so a window under "
                f"{self.MIN_WINDOW} is constant almost everywhere. Use window ≥ "
                f"{self.MIN_WINDOW}, or a level operator (ts_mean / ts_delta).")


def _leaf_fields(node: Node) -> set:
    if isinstance(node, DataNode):
        return {node.field}
    out: set = set()
    for child in node.children():
        out |= _leaf_fields(child)
    return out


# ---------------------------------------------------------------------------
# Composite AlphaValidator
# ---------------------------------------------------------------------------

class AlphaValidator:
    """
    Runs all validators, collects every error, and raises a single
    ValidationError with a complete report if any errors are found.
    """

    def __init__(self) -> None:
        self._validators = [
            DepthValidator(),
            WindowValidator(),
            LookAheadValidator(),
            SparseFieldValidator(),
        ]

    def validate(self, node: Node) -> None:
        """
        Validate the AST rooted at ``node``.
        Raises ValidationError listing all detected problems.
        """
        all_errors: List[str] = []
        for v in self._validators:
            all_errors.extend(v.collect(node))
        if all_errors:
            raise ValidationError(all_errors)

    def is_valid(self, node: Node) -> bool:
        """Return True iff the node passes all validators."""
        try:
            self.validate(node)
            return True
        except ValidationError:
            return False
