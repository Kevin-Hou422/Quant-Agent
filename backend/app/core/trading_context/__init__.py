"""app.core.trading_context — 交易现实引擎（Phase TR）。

- TradingContext / BrokerProfile：从免费数据推导真实成本、可做空性、调仓带（T1/T2）。
- providers：T3（只有交易当时才知道的）盘口/借券/账户接口——仿真背估计、实盘背 moomoo，同接口切换。
"""

from .context import (
    TradingContext, TradingContextResult, BrokerProfile,
    MOOMOO_US, get_broker_profile, grounded_cost_params,
)
from .providers import (
    QuoteProvider, BorrowProvider, AccountProvider,
    SimQuoteProvider, SimBorrowProvider, SimAccountProvider,
    TradeProviders, get_trade_providers,
)

__all__ = [
    "TradingContext", "TradingContextResult", "BrokerProfile",
    "MOOMOO_US", "get_broker_profile", "grounded_cost_params",
    "QuoteProvider", "BorrowProvider", "AccountProvider",
    "SimQuoteProvider", "SimBorrowProvider", "SimAccountProvider",
    "TradeProviders", "get_trade_providers",
]
