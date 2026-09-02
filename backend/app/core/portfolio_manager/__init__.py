"""app.core.portfolio_manager — 组合与资金管理层（Phase PM）。"""

from .manager import PortfolioManager, PortfolioResult
from .strategy_gate import (
    StrategyGate,
    StrategyValidationResult,
    strategy_net_returns,
    resolve_cost_params,
    marginal_factor_selection,
    MarginalSelectionResult,
)
from .risk_gate import PortfolioRiskGate, RiskLimits, RiskReport
from .horizon import (
    apply_no_trade_band, annualized_turnover, classify_horizon,
    horizon_profile, FactorHorizon,
)
from .strategy_builder import build_strategy_config, propose_from_paper_factors

__all__ = [
    "PortfolioManager", "PortfolioResult",
    "StrategyGate", "StrategyValidationResult", "strategy_net_returns", "resolve_cost_params",
    "marginal_factor_selection", "MarginalSelectionResult",
    "PortfolioRiskGate", "RiskLimits", "RiskReport",
    "apply_no_trade_band", "annualized_turnover", "classify_horizon",
    "horizon_profile", "FactorHorizon",
    "build_strategy_config", "propose_from_paper_factors",
]
