"""app.core.portfolio_manager — 组合与资金管理层（Phase PM）。"""

from .manager import PortfolioManager, PortfolioResult
from .strategy_gate import (
    StrategyGate,
    StrategyValidationResult,
    strategy_net_returns,
    marginal_factor_selection,
    MarginalSelectionResult,
)

__all__ = [
    "PortfolioManager", "PortfolioResult",
    "StrategyGate", "StrategyValidationResult", "strategy_net_returns",
    "marginal_factor_selection", "MarginalSelectionResult",
]
