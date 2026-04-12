"""
portfolio_engine — 信号处理、组合构建与现实主义回测

集中管理与组合逻辑相关的组件：
  - SimulationConfig / SignalProcessor : 4步信号管道（Phase 1）
  - DecilePortfolio / SignalWeightedPortfolio / NeutralizationLayer : 组合构建
  - RealisticBacktester / RealisticBacktestResult : IS+OOS 双段回测（Phase 1）
"""

from .signal_processor import SimulationConfig, SignalProcessor
from .portfolio_constructor import (
    DecilePortfolio,
    SignalWeightedPortfolio,
    NeutralizationLayer,
)
from .realistic_backtester import RealisticBacktester, RealisticBacktestResult

__all__ = [
    "SimulationConfig", "SignalProcessor",
    "DecilePortfolio", "SignalWeightedPortfolio", "NeutralizationLayer",
    "RealisticBacktester", "RealisticBacktestResult",
]
