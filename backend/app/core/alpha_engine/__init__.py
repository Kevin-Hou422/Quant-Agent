"""
Alpha Engine package.

Current (typed DSL) API — use these:
  from app.core.alpha_engine.typed_nodes import Node, ...
  from app.core.alpha_engine.parser     import Parser, ParseError
  from app.core.alpha_engine.validator  import AlphaValidator, ValidationError
  from app.core.alpha_engine.dsl_executor import Executor
  from app.core.alpha_engine.signal_processor import SimulationConfig, SignalProcessor

Legacy API (ast.Node-based) — deprecated, kept for backward compatibility only:
  generator.generate_random_alpha  → superseded by gp_engine.gp_engine.generate_random_alpha
  executor.AlphaExecutor           → superseded by dsl_executor.Executor
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

# --- Legacy API (ast.Node system) — retained for test backward-compat ---
# These use the OLD ast.Node, not typed_nodes.Node.
# New code should use gp_engine.gp_engine.generate_random_alpha() instead.
from .ast import Node as ASTNode                                   # noqa: F401
from .generator import generate_random_alpha, generate_n_alphas    # noqa: F401
from .executor import AlphaExecutor, execute_alpha, batch_execute  # noqa: F401

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
    # --- legacy (deprecated) ---
    # S3 修复（2026-07-24）：AlphaExecutor/execute_alpha/batch_execute 已从
    # __all__ 移除 —— 它们只接受旧 ast.Node，喂入 Parser 产出的 typed AST 会以
    # 无提示的 TypeError 崩溃。符号仍可显式 import（向后兼容），但实例化时
    # 触发 DeprecationWarning（见 executor.py）。
    "ASTNode",
    "generate_random_alpha",
    "generate_n_alphas",
]
