"""
DSL Parser — Lark grammar → Typed AST.

Grammar supports:
  - Binary arithmetic:       +  -  *  /
  - Comparison operators:    >  <  >=  <=  ==  !=
  - Logical infix operators: &&  ||
  - Unary:                   -  !
  - Function calls:          ts_mean(close, 20), rank(x), group_rank(x,'ind'), ...
  - Numeric literals (int / float)
  - String literals ('name' or "name") — used as group field arguments
  - Field identifiers (close, open, high, low, volume, vwap, returns, ...)
  - Parenthesised sub-expressions

Operator precedence (low → high):
  ||  →  &&  →  ==  !=  >  <  >=  <=  →  +  -  →  *  /  →  unary - !  →  atom

The Lark transformer maps each parse-tree rule to the appropriate
typed_nodes class.
"""

from __future__ import annotations

from lark import Lark, Transformer
from lark.exceptions import UnexpectedInput, UnexpectedCharacters

from .typed_nodes import (
    Node, ScalarNode, DataNode, StringLiteralNode,
    TimeSeriesNode, CrossSectionalNode, ArithmeticNode,
    GroupNode,
    _TS_OPS, _CS_OPS, _GROUP_OPS,
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

    // Precedence (low → high): || → && → cmp → add → mul → unary → atom
    ?expr  : expr _OR     and_expr  -> logical_or
           | and_expr

    ?and_expr : and_expr _AND  cmp_expr -> logical_and
              | cmp_expr

    ?cmp_expr : cmp_expr _GTE  add_expr -> gte
              | cmp_expr _LTE  add_expr -> lte
              | cmp_expr _GT   add_expr -> gt
              | cmp_expr _LT   add_expr -> lt
              | cmp_expr _EQ   add_expr -> eq
              | cmp_expr _NEQ  add_expr -> ne
              | add_expr

    ?add_expr : add_expr _PLUS  mul_expr -> add
              | add_expr _MINUS mul_expr -> sub
              | mul_expr

    ?mul_expr : mul_expr _STAR  unary -> mul
              | mul_expr _SLASH unary -> div
              | unary

    ?unary : _MINUS unary -> neg
           | _NOT   unary -> logical_not
           | atom

    ?atom  : func_call
           | NUMBER              -> number
           | STRLIT              -> string_lit
           | IDENT               -> ident_ref
           | "(" expr ")"

    func_call : IDENT "(" arglist ")"

    arglist : arg ("," arg)*
    ?arg    : expr | STRLIT -> string_lit

    _PLUS  : "+"
    _MINUS : "-"
    _STAR  : "*"
    _SLASH : "/"
    _GT    : ">"
    _LT    : "<"
    _GTE   : ">="
    _LTE   : "<="
    _EQ    : "=="
    _NEQ   : "!="
    _AND   : "&&"
    _OR    : "||"
    _NOT   : "!"

    IDENT  : /[a-z][a-z0-9_]*/
    NUMBER : /[0-9]+(\.[0-9]+)?/
    STRLIT : "'" /[^']*/ "'" | "\"" /[^\"]*/ "\""

    %ignore /\s+/
"""

# Function names that map to unary ArithmeticNode ops
_UNARY_MATH = {"log", "abs", "sqrt", "sign"}


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class _AlphaTransformer(Transformer):
    """Transform a Lark parse tree into a typed Node tree."""

    # ------ Infix binary ops -----------------------------------------------

    def add(self, args):  return ArithmeticNode("add", [args[0], args[1]])
    def sub(self, args):  return ArithmeticNode("sub", [args[0], args[1]])
    def mul(self, args):  return ArithmeticNode("mul", [args[0], args[1]])
    def div(self, args):  return ArithmeticNode("div", [args[0], args[1]])
    def gt(self, args):   return ArithmeticNode("gt",  [args[0], args[1]])
    def lt(self, args):   return ArithmeticNode("lt",  [args[0], args[1]])
    def gte(self, args):  return ArithmeticNode("gte", [args[0], args[1]])
    def lte(self, args):  return ArithmeticNode("lte", [args[0], args[1]])
    def eq(self, args):   return ArithmeticNode("eq",  [args[0], args[1]])
    def ne(self, args):   return ArithmeticNode("ne",  [args[0], args[1]])

    def logical_and(self, args):
        return ArithmeticNode("logical_and", [args[0], args[1]])

    def logical_or(self, args):
        return ArithmeticNode("logical_or", [args[0], args[1]])

    def logical_not(self, args):
        return ArithmeticNode("logical_not", [args[0]])

    def neg(self, args):
        return ArithmeticNode("neg", [args[0]])

    # ------ Literals -------------------------------------------------------

    def number(self, args) -> ScalarNode:
        return ScalarNode(float(str(args[0])))

    def string_lit(self, args) -> StringLiteralNode:
        s = str(args[0])
        # Strip surrounding quotes (' or ")
        return StringLiteralNode(s[1:-1])

    def ident_ref(self, args) -> DataNode:
        return DataNode(str(args[0]))

    # ------ Function calls -------------------------------------------------

    def arglist(self, args) -> list:
        return list(args)

    def func_call(self, args) -> Node:
        name: str    = str(args[0])
        arglist: list = args[1] if len(args) > 1 else []

        # ---- Time-series operators ----
        if name in _TS_OPS:
            return self._build_ts(name, arglist)

        # ---- Cross-sectional operators ----
        if name in _CS_OPS:
            return self._build_cs(name, arglist)

        # ---- Group operators ----
        if name in _GROUP_OPS:
            return self._build_group(name, arglist)

        # ---- Unary math ----
        if name in _UNARY_MATH:
            if len(arglist) != 1:
                raise ParseError(f"'{name}' expects 1 argument, got {len(arglist)}")
            return ArithmeticNode(name, [arglist[0]])

        # ---- signed_power ----
        if name == "signed_power":
            if len(arglist) != 2:
                raise ParseError(f"'signed_power' expects 2 arguments")
            return ArithmeticNode("signed_power", [arglist[0], arglist[1]])

        # ---- if_else / where ----
        if name in ("if_else", "where"):
            if len(arglist) != 3:
                raise ParseError(f"'{name}' expects 3 arguments (cond, x, y)")
            return ArithmeticNode("if_else", arglist)

        # ---- trade_when ----
        if name == "trade_when":
            if len(arglist) != 2:
                raise ParseError(f"'trade_when' expects 2 arguments (cond, x)")
            return ArithmeticNode("trade_when", arglist)

        # ---- and / or / not (function-call form) ----
        if name == "and":
            if len(arglist) != 2:
                raise ParseError(f"'and' expects 2 arguments")
            return ArithmeticNode("logical_and", arglist)

        if name == "or":
            if len(arglist) != 2:
                raise ParseError(f"'or' expects 2 arguments")
            return ArithmeticNode("logical_or", arglist)

        if name == "not":
            if len(arglist) != 1:
                raise ParseError(f"'not' expects 1 argument")
            return ArithmeticNode("logical_not", arglist)

        # ---- pow ----
        if name == "pow":
            if len(arglist) != 2:
                raise ParseError(f"'pow' expects 2 arguments (base, exp)")
            return ArithmeticNode("pow", arglist)

        # ---- max / min (two-arg form) ----
        if name == "max":
            if len(arglist) != 2:
                raise ParseError(f"'max' expects 2 arguments")
            return ArithmeticNode("max2", arglist)

        if name == "min":
            if len(arglist) != 2:
                raise ParseError(f"'min' expects 2 arguments")
            return ArithmeticNode("min2", arglist)

        # ---- weighted_sum ----
        if name == "weighted_sum":
            if len(arglist) < 2 or len(arglist) % 2 != 0:
                raise ParseError(
                    f"'weighted_sum' expects an even number of arguments "
                    f"(value1, weight1, value2, weight2, ...), got {len(arglist)}"
                )
            vals = arglist[0::2]   # v1, v2, ...
            wgts = arglist[1::2]   # w1, w2, ...
            return ArithmeticNode("weighted_sum", vals + wgts)

        raise ParseError(
            f"Unknown function: '{name}'. "
            f"Known TS ops: {sorted(_TS_OPS)}. "
            f"Known CS ops: {sorted(_CS_OPS)}. "
            f"Known group ops: {sorted(_GROUP_OPS)}."
        )

    # ------ Builder helpers ------------------------------------------------

    def _build_ts(self, name: str, arglist: list) -> TimeSeriesNode:
        """
        Standard TS:  func(expr, window)
        Two-input TS: func(expr, expr, window)  — ts_corr, ts_cov
        """
        from .fast_ops import _TWO_INPUT_TS_OPS
        if name in _TWO_INPUT_TS_OPS:
            if len(arglist) < 3:
                raise ParseError(
                    f"'{name}' expects 3 arguments (x, y, window), "
                    f"got {len(arglist)}"
                )
            child1 = arglist[0]
            child2 = arglist[1]
            window = self._extract_window(arglist[2], name)
            return TimeSeriesNode(name, child1, window, second_child=child2)
        else:
            if len(arglist) < 2:
                raise ParseError(
                    f"'{name}' expects at least 2 arguments (expr, window), "
                    f"got {len(arglist)}"
                )
            child  = arglist[0]
            window = self._extract_window(arglist[1], name)
            return TimeSeriesNode(name, child, window)

    def _build_cs(self, name: str, arglist: list) -> CrossSectionalNode:
        if name == "ind_neutralize":
            if len(arglist) < 1:
                raise ParseError("'ind_neutralize' expects at least 1 argument")
            child       = arglist[0]
            groups_node = arglist[1] if len(arglist) > 1 else None
            return CrossSectionalNode(name, child, groups_node=groups_node)
        if name == "winsorize":
            if len(arglist) < 1:
                raise ParseError("'winsorize' expects 1 or 2 arguments (x[, k])")
            child = arglist[0]
            k     = 3.0
            if len(arglist) > 1 and isinstance(arglist[1], ScalarNode):
                k = float(arglist[1].value)
            return CrossSectionalNode(name, child, k=k)
        if len(arglist) != 1:
            raise ParseError(
                f"'{name}' expects exactly 1 argument, got {len(arglist)}"
            )
        return CrossSectionalNode(name, arglist[0])

    def _build_group(self, name: str, arglist: list) -> GroupNode:
        if len(arglist) < 1:
            raise ParseError(f"'{name}' expects at least 1 argument")
        child       = arglist[0]
        group_field = "groups"  # default
        if len(arglist) > 1:
            second = arglist[1]
            if isinstance(second, StringLiteralNode):
                group_field = second.value
            elif isinstance(second, DataNode):
                group_field = second.field
        return GroupNode(name, child, group_field)

    @staticmethod
    def _extract_window(arg, op_name: str) -> int:
        if isinstance(arg, ScalarNode):
            return int(arg.value)
        raise ParseError(
            f"'{op_name}': window argument must be an integer literal, "
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
        node   = parser.parse("group_rank(close,'industry') - group_mean(close,'industry')")
        node   = parser.parse("ts_corr(close,volume,20) * (close > ts_mean(close,10))")
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
