"""
Alpha Engine package.

Exports both the original procedural API and the new typed DSL engine.
"""

# --- Original API (preserved) ---
from .ast import Node as ASTNode
from .generator import generate_random_alpha, generate_n_alphas
from .executor import AlphaExecutor, execute_alpha, batch_execute

# --- DSL Engine ---
from .typed_nodes import (
    Node,
    NodeType,
    ScalarNode,
    DataNode,
    TimeSeriesNode,
    CrossSectionalNode,
    ArithmeticNode,
)
from .validator import AlphaValidator, ValidationError
from .parser import Parser, ParseError
from .dsl_executor import Executor

__all__ = [
    # original
    "ASTNode",
    "generate_random_alpha",
    "generate_n_alphas",
    "AlphaExecutor",
    "execute_alpha",
    "batch_execute",
    # DSL typed nodes
    "Node",
    "NodeType",
    "ScalarNode",
    "DataNode",
    "TimeSeriesNode",
    "CrossSectionalNode",
    "ArithmeticNode",
    # validation
    "AlphaValidator",
    "ValidationError",
    # parser
    "Parser",
    "ParseError",
    # executor
    "Executor",
]
