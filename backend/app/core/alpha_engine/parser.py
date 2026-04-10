"""
DSL Parser — Lark grammar → Typed AST.

Grammar supports:
  - Binary arithmetic: +  -  *  /
  - Comparison operators: >  <  >=  <=  ==
  - Function calls: ts_mean(close, 20), rank(x), if_else(cond, x, y), ...
  - Numeric literals (int / float)
  - Field identifiers (close, open, high, low, volume, vwap, returns, ...)
  - Parenthesised sub-expressions

The Lark transformer maps each parse-tree rule to the appropriate
typed_nodes class.  Operator dispatch is table-driven so adding a new
operator only requires updating the lookup tables.
"""

from __future__ import annotations

from typing import Any, List

from lark import Lark, Transformer, Token, Tree
from lark.exceptions import UnexpectedInput, UnexpectedCharacters

from .typed_nodes import (
    Node, ScalarNode, DataNode,
    TimeSeriesNode, CrossSectionalNode, ArithmeticNode,
    _TS_OPS, _CS_OPS,
)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Raised when the DSL string cannot be parsed."""
    pass


# ---------------------------------------------------------------------------
# Lark Grammar
# ---------------------------------------------------------------------------

_GRAMMAR = r"""
    ?start : expr

    ?expr  : expr _PLUS    term   -> add
           | expr _MINUS   term   -> sub
           | expr _GTE     term   -> gte
           | expr _LTE     term   -> lte
           | expr _GT      term   -> gt
           | expr _LT      term   -> lt
           | expr _EQ      term   -> eq
           | term

    ?term  : term _STAR    factor -> mul
           | term _SLASH   factor -> div
           | factor

    ?factor: _MINUS factor        -> neg
           | atom

    ?atom  : func_call
           | NUMBER              -> number
           | FIELD               -> field_ref
           | "(" expr ")"

    func_call : FUNC "(" arglist ")"

    arglist : expr ("," expr)*

    _PLUS  : "+"
    _MINUS : "-"
    _STAR  : "*"
    _SLASH : "/"
    _GT    : ">"
    _LT    : "<"
    _GTE   : ">="
    _LTE   : "<="
    _EQ    : "=="

    FUNC  : /[a-z][a-z0-9_]*/
    FIELD : /[a-z][a-z0-9_]*/
    NUMBER: /[0-9]+(\.[0-9]+)?/

    %ignore /\s+/
"""

# Operator name → binary ArithmeticNode op key
_BINOP_MAP = {
    "add": "add", "sub": "sub", "mul": "mul", "div": "div",
    "gt":  "gt",  "lt":  "lt",  "gte": "gte", "lte": "lte", "eq": "eq",
}

# Function name → TS / CS / Arithmetic category
_UNARY_MATH = {"log", "abs", "sqrt", "sign"}
_ADVANCED   = {"signed_power", "if_else"}


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class _AlphaTransformer(Transformer):
    """Transform a Lark parse tree into a typed Node tree."""

    # ------ Binary ops -------------------------------------------------------

    def add(self, args):  return ArithmeticNode("add", [args[0], args[1]])
    def sub(self, args):  return ArithmeticNode("sub", [args[0], args[1]])
    def mul(self, args):  return ArithmeticNode("mul", [args[0], args[1]])
    def div(self, args):  return ArithmeticNode("div", [args[0], args[1]])
    def gt(self, args):   return ArithmeticNode("gt",  [args[0], args[1]])
    def lt(self, args):   return ArithmeticNode("lt",  [args[0], args[1]])
    def gte(self, args):  return ArithmeticNode("gte", [args[0], args[1]])
    def lte(self, args):  return ArithmeticNode("lte", [args[0], args[1]])
    def eq(self, args):   return ArithmeticNode("eq",  [args[0], args[1]])
    def neg(self, args):  return ArithmeticNode("neg", [args[0]])

    # ------ Literals / fields ------------------------------------------------

    def number(self, args) -> ScalarNode:
        return ScalarNode(float(str(args[0])))

    def field_ref(self, args) -> DataNode:
        return DataNode(str(args[0]))

    # ------ Function calls ---------------------------------------------------

    def arglist(self, args) -> list:
        return list(args)

    def func_call(self, args) -> Node:
        name: str = str(args[0])
        arglist: list = args[1] if len(args) > 1 else []

        # Time-series operators
        if name in _TS_OPS:
            return self._build_ts(name, arglist)

        # Cross-sectional operators
        if name in _CS_OPS:
            return self._build_cs(name, arglist)

        # Unary math
        if name in _UNARY_MATH:
            if len(arglist) != 1:
                raise ParseError(f"'{name}' expects 1 argument, got {len(arglist)}")
            return ArithmeticNode(name, [arglist[0]])

        # Advanced
        if name == "signed_power":
            if len(arglist) != 2:
                raise ParseError(f"'signed_power' expects 2 arguments, got {len(arglist)}")
            return ArithmeticNode("signed_power", [arglist[0], arglist[1]])

        if name == "if_else":
            if len(arglist) != 3:
                raise ParseError(f"'if_else' expects 3 arguments, got {len(arglist)}")
            return ArithmeticNode("if_else", [arglist[0], arglist[1], arglist[2]])

        # Unknown — treat as a data field (e.g. a custom signal name)
        # This allows extensibility without hard-coding every field name.
        raise ParseError(
            f"Unknown function or field: '{name}'. "
            f"Known TS ops: {sorted(_TS_OPS)}. "
            f"Known CS ops: {sorted(_CS_OPS)}."
        )

    # ------ Helpers ----------------------------------------------------------

    def _build_ts(self, name: str, arglist: list) -> TimeSeriesNode:
        """
        Most TS ops: func(expr, window)
        Two-input ops (ts_corr, ts_cov): func(expr, expr, window)
        """
        if name in {"ts_corr", "ts_cov"}:
            if len(arglist) < 3:
                raise ParseError(f"'{name}' expects 3 arguments (x, y, window)")
            child1, child2 = arglist[0], arglist[1]
            window = self._extract_window(arglist[2], name)
            # Store second child as extra_param; evaluation handled in fast_ops
            node = TimeSeriesNode(name, child1, window)
            node._second_child = child2   # type: ignore[attr-defined]
            return node
        else:
            if len(arglist) < 2:
                raise ParseError(
                    f"'{name}' expects at least 2 arguments (expr, window), "
                    f"got {len(arglist)}: {arglist}"
                )
            child  = arglist[0]
            window = self._extract_window(arglist[1], name)
            return TimeSeriesNode(name, child, window)

    def _build_cs(self, name: str, arglist: list) -> CrossSectionalNode:
        if name == "ind_neutralize":
            if len(arglist) < 1:
                raise ParseError("'ind_neutralize' expects at least 1 argument")
            child = arglist[0]
            # groups passed as second arg (ScalarNode treated as group array later)
            groups_node = arglist[1] if len(arglist) > 1 else None
            return CrossSectionalNode(name, child, groups_node=groups_node)
        if len(arglist) != 1:
            raise ParseError(f"'{name}' expects exactly 1 argument, got {len(arglist)}")
        return CrossSectionalNode(name, arglist[0])

    @staticmethod
    def _extract_window(arg: Any, op_name: str) -> int:
        if isinstance(arg, ScalarNode):
            return int(arg.value)
        raise ParseError(
            f"'{op_name}': second argument (window) must be an integer literal, "
            f"got {repr(arg)}"
        )


# ---------------------------------------------------------------------------
# Parser class (public API)
# ---------------------------------------------------------------------------

class Parser:
    """
    Parse an Alpha DSL expression string into a typed AST.

    Usage::

        parser = Parser()
        node   = parser.parse("rank(ts_delta(log(close),5))/ts_std(close,20)")
    """

    def __init__(self) -> None:
        self._lark = Lark(
            _GRAMMAR,
            parser="earley",
            ambiguity="resolve",
        )
        self._transformer = _AlphaTransformer()

    def parse(self, expr: str) -> Node:
        """
        Parse ``expr`` and return the root Node of the typed AST.

        Raises
        ------
        ParseError
            If the expression has a syntax error or uses an unknown operator.
        """
        expr = expr.strip()
        if not expr:
            raise ParseError("Empty expression string.")
        try:
            tree = self._lark.parse(expr)
            node = self._transformer.transform(tree)
        except (UnexpectedInput, UnexpectedCharacters) as exc:
            raise ParseError(
                f"Syntax error in expression '{expr}': {exc}"
            ) from exc
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(f"Failed to parse '{expr}': {exc}") from exc
        return node
