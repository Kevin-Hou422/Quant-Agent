"""
Static AST Validators for the Alpha DSL Engine.

Three composable validators run before any computation:

  WindowValidator   — window size in [1, 252]
  LookAheadValidator — no future-data access (window ≥ 1 for delay/delta)
  DepthValidator    — nesting depth ≤ 10

AlphaValidator is the composite that runs all three and collects every
error before raising a single ValidationError with a full report.
"""

from __future__ import annotations

import warnings
from typing import List

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
# Composite AlphaValidator
# ---------------------------------------------------------------------------

class AlphaValidator:
    """
    Runs all three validators, collects every error, and raises a single
    ValidationError with a complete report if any errors are found.
    """

    def __init__(self) -> None:
        self._validators = [
            DepthValidator(),
            WindowValidator(),
            LookAheadValidator(),
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
