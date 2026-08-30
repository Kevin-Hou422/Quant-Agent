"""
Alpha Engine package.

Current (typed DSL) API — use these:
  from app.core.alpha_engine.typed_nodes import Node, ...
  from app.core.alpha_engine.parser     import Parser, ParseError
  from app.core.alpha_engine.validator  import AlphaValidator, ValidationError
  from app.core.alpha_engine.dsl_executor import Executor
  from app.core.alpha_engine.signal_processor import SimulationConfig, SignalProcessor

旧 ast.Node 引擎（ast/executor/generator/operators）已于 2026-08-30 删除——
被 typed_nodes + parser + dsl_executor + gp_engine 完全取代。
"""

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
from .signal_processor import SimulationConfig, SignalProcessor

__all__ = [
    # typed DSL nodes
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
    # signal processor
    "SimulationConfig",
    "SignalProcessor",
]
