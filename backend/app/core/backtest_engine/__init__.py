"""
Backtest Engine 包入口

统一导出所有公共 API。
"""

from .portfolio_constructor import (
    PortfolioConstructor,
    DecilePortfolio,
    SignalWeightedPortfolio,
    NeutralizationLayer,
)
from .transaction_cost import (
    CostParams,
    SlippageModel,
    LiquidityConstraint,
    TransactionCostEngine,
    TradeRecord,
)
from .backtest_engine import BacktestEngine, BacktestResult
from .performance_analyzer import PerformanceAnalyzer
from .risk_report import RiskReport
from .visualizer import BacktestVisualizer

__all__ = [
    # Portfolio Construction
    "PortfolioConstructor",
    "DecilePortfolio",
    "SignalWeightedPortfolio",
    "NeutralizationLayer",
    # Transaction Cost
    "CostParams",
    "SlippageModel",
    "LiquidityConstraint",
    "TransactionCostEngine",
    "TradeRecord",
    # Engine
    "BacktestEngine",
    "BacktestResult",
    # Analysis
    "PerformanceAnalyzer",
    "RiskReport",
    # Visualization
    "BacktestVisualizer",
    # Realistic Backtester
    "RealisticBacktester",
    "RealisticBacktestResult",
]

from .realistic_backtester import RealisticBacktester, RealisticBacktestResult
