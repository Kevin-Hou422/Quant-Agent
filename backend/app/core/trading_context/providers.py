"""
providers.py — T3（只有交易当时才知道的）数据接口（Phase TR.3 补齐）

三层纪律（DEV_LESSONS §J）里的 **T3**：真实盘口价差、可借券/借券费、买入力/持仓/PDT——
这些**只有交易那一刻才知道**，**绝不能写字面值**。本模块给它们一个**统一接口**：
    - **仿真**（paper）→ 背 TR.1 的数据推导估计（Corwin-Schultz 价差、可做空启发式、纸账户状态）
    - **实盘/纸交易接券商**（Phase 12）→ 背 moomoo 实时 API
**同一接口切换**，上层代码不必改，也不会退化成硬编码。

接口
    QuoteProvider   : spread_bps / mid_price       —— 盘口
    BorrowProvider  : is_shortable / borrow_fee_bps —— 借券
    AccountProvider : buying_power / cash / positions —— 账户
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WidePanel = Dict[str, pd.DataFrame]


# ---------------------------------------------------------------------------
# 抽象接口
# ---------------------------------------------------------------------------

class QuoteProvider(ABC):
    """盘口：交易当时的价差与中间价。"""

    @abstractmethod
    def spread_bps(self, ticker: str) -> float: ...

    @abstractmethod
    def mid_price(self, ticker: str) -> float: ...


class BorrowProvider(ABC):
    """借券：能否做空、借券费。"""

    @abstractmethod
    def is_shortable(self, ticker: str) -> bool: ...

    @abstractmethod
    def borrow_fee_bps(self, ticker: str) -> float: ...


class AccountProvider(ABC):
    """账户：买入力、现金、持仓。"""

    @abstractmethod
    def buying_power(self) -> float: ...

    @abstractmethod
    def cash(self) -> float: ...

    @abstractmethod
    def positions(self) -> Dict[str, float]: ...


# ---------------------------------------------------------------------------
# 仿真实现：背 TR.1 的数据推导估计 / 纸账户状态（**估计，不是实时**）
# ---------------------------------------------------------------------------

class SimQuoteProvider(QuoteProvider):
    """价差 = Corwin-Schultz 从免费 H/L 估（TR.1）；中间价 = 最近收盘。均为**估计**。"""

    def __init__(self, dataset: WidePanel) -> None:
        from app.core.trading_context.spread import corwin_schultz_spread_bps
        self._spread = corwin_schultz_spread_bps(dataset["high"], dataset["low"])
        self._median = float(np.nanmedian(self._spread.values)) if len(self._spread) else 0.0
        self._px = dataset["close"].ffill().iloc[-1]

    def spread_bps(self, ticker: str) -> float:
        v = self._spread.get(ticker, np.nan)
        return float(v) if np.isfinite(v) else self._median

    def mid_price(self, ticker: str) -> float:
        v = self._px.get(ticker, np.nan)
        return float(v) if np.isfinite(v) else float("nan")


class SimBorrowProvider(BorrowProvider):
    """可做空性 = TradingContext 的推导（long-only/账户类型/流动性启发式）；仿真借券费记 0。"""

    def __init__(self, dataset: WidePanel, aum: float,
                 account_type: str = "margin", allow_short: bool = False) -> None:
        from app.core.trading_context.context import TradingContext
        res = TradingContext(aum=aum, account_type=account_type,
                             allow_short=allow_short).analyze(dataset)
        self._shortable = res.shortable

    def is_shortable(self, ticker: str) -> bool:
        return bool(self._shortable.get(ticker, False))

    def borrow_fee_bps(self, ticker: str) -> float:
        return 0.0          # 仿真：不建模借券费（long-only 阶段不适用）


class SimAccountProvider(AccountProvider):
    """账户状态 = 纸账户（PaperBroker + PositionStore）的真实记账。"""

    def __init__(self, broker, book_id: int = 0) -> None:
        self._broker = broker
        self._book = int(book_id)

    def _equity_dollars(self) -> float:
        eq = float(self._broker.store.latest_equity(self._book))     # 归一化净值
        return eq * float(self._broker.initial_capital)

    def buying_power(self) -> float:
        return self._equity_dollars()

    def cash(self) -> float:
        pos = self.positions()
        invested = sum(abs(w) for w in pos.values())                 # 权重口径
        return max(0.0, (1.0 - invested)) * self._equity_dollars()

    def positions(self) -> Dict[str, float]:
        try:
            return dict(self._broker.store.latest_positions(self._book))
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# 工厂：同接口切换 sim ↔ live
# ---------------------------------------------------------------------------

@dataclass
class TradeProviders:
    quote:   QuoteProvider
    borrow:  BorrowProvider
    account: AccountProvider
    mode:    str

    def to_dict(self) -> dict:
        return {"mode": self.mode,
                "buying_power": round(self.account.buying_power(), 2),
                "n_positions": len(self.account.positions())}


def get_trade_providers(mode: str = "sim", *, dataset: Optional[WidePanel] = None,
                        aum: float = 0.0, broker=None, book_id: int = 0,
                        account_type: str = "margin",
                        allow_short: bool = False) -> TradeProviders:
    """
    `mode="sim"` → 仿真三件套（背 TR.1 估计 + 纸账户）。
    `mode="live"` → moomoo 实时三件套（**Phase 12 实现**；此处显式抛错，绝不静默退回估计，
                    否则会把"估计"当成"实时"用——那正是 T3 纪律要防的事）。
    """
    if mode == "live":
        raise NotImplementedError(
            "live T3 providers（moomoo 实时盘口/借券/账户）待 Phase 12 接入 OpenD 交易上下文；"
            "在此之前不要以 live 模式运行——绝不用估计冒充实时。")
    if dataset is None or broker is None:
        raise ValueError("sim 模式需要 dataset 与 broker")
    return TradeProviders(
        quote=SimQuoteProvider(dataset),
        borrow=SimBorrowProvider(dataset, aum=aum, account_type=account_type,
                                 allow_short=allow_short),
        account=SimAccountProvider(broker, book_id=book_id),
        mode="sim",
    )
