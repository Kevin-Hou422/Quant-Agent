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
        """
        当前持仓。**读失败绝不能返回 {}**：空字典 = "我什么都没持有"，
        下游据此把全部权益当成可用买入力，会按满额重新建仓（等于凭空加杠杆）。
        读不到就抛错，让调用方决定是跳过本轮还是告警。
        """
        try:
            return dict(self._broker.store.latest_positions(self._book))
        except Exception as exc:
            logger.error("[SimAccountProvider] 持仓读取失败 —— 拒绝返回空仓假象: %s", exc)
            raise RuntimeError(f"无法读取当前持仓，本轮不可交易: {exc}") from exc


# ---------------------------------------------------------------------------
# 实时实现（Phase 12）：背 moomoo 券商网关 / 行情快照（**实时，不是估计**）
# ---------------------------------------------------------------------------

class LiveAccountProvider(AccountProvider):
    """账户 = moomoo 纸交易账户的实时状态（经 BrokerGateway）。失败一律抛错，不给空仓假象。"""

    def __init__(self, gateway) -> None:
        self._gw = gateway

    def buying_power(self) -> float:
        return float(self._gw.account().power)

    def cash(self) -> float:
        return float(self._gw.account().cash)

    def positions(self) -> Dict[str, float]:
        """与仿真同口径：权重 = 市值 / 账户总资产。"""
        acct = self._gw.account()
        if not (acct.total_assets > 0):
            raise RuntimeError(f"账户总资产非正（{acct.total_assets}），无法折算持仓权重")
        return {tk: p.market_val / acct.total_assets for tk, p in self._gw.positions().items()}


class LiveQuoteProvider(QuoteProvider):
    """
    盘口 = moomoo 行情快照的买一/卖一。`snapshot_fn(tickers) -> DataFrame`
    （列至少含 code / bid_price / ask_price，即 OpenQuoteContext.get_market_snapshot 的返回）。
    买一或卖一不是正数（休市、停牌、无报价）→ 抛错，绝不回退到估计价差。
    """

    def __init__(self, snapshot_fn) -> None:
        self._snap = snapshot_fn

    def _quote(self, ticker: str):
        from app.core.data_engine.providers.moomoo_provider import _from_moomoo_code
        df = self._snap([ticker])
        rows = df[df["code"].map(lambda c: _from_moomoo_code(str(c))) == ticker]
        if rows.empty:
            raise RuntimeError(f"{ticker} 没有行情快照")
        r = rows.iloc[0]
        bid, ask = float(r["bid_price"]), float(r["ask_price"])
        if not (np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask >= bid):
            raise RuntimeError(f"{ticker} 当前无有效买卖报价（bid={bid}, ask={ask}）")
        return bid, ask

    def spread_bps(self, ticker: str) -> float:
        bid, ask = self._quote(ticker)
        return (ask - bid) / ((ask + bid) / 2.0) * 1e4

    def mid_price(self, ticker: str) -> float:
        bid, ask = self._quote(ticker)
        return (ask + bid) / 2.0


class LiveBorrowProvider(BorrowProvider):
    """
    long-only 阶段（trading_allow_short=False）恒为不可做空 —— 与配置一致，不是估计。
    借券费：快照里的 short_sell_rate **单位未经核实**，猜错单位会让成本差两个数量级，
    所以这里拒绝给数，等开启做空时再按券商文档核实后接入。
    """

    def __init__(self, allow_short: bool = False, snapshot_fn=None) -> None:
        self._allow = bool(allow_short)
        self._snap = snapshot_fn

    def is_shortable(self, ticker: str) -> bool:
        if not self._allow:
            return False
        if self._snap is None:
            raise RuntimeError("允许做空时必须提供行情快照来源以查询可卖空性")
        from app.core.data_engine.providers.moomoo_provider import _from_moomoo_code
        df = self._snap([ticker])
        rows = df[df["code"].map(lambda c: _from_moomoo_code(str(c))) == ticker]
        if rows.empty:
            raise RuntimeError(f"{ticker} 没有行情快照")
        return bool(rows.iloc[0]["enable_short_sell"])

    def borrow_fee_bps(self, ticker: str) -> float:
        raise NotImplementedError(
            "实时借券费未接入：short_sell_rate 的单位未经核实，拒绝猜测（long-only 阶段不需要）")


def moomoo_snapshot_fn(host: str, port: int):
    """
    生产用的行情快照来源：每次调用开一个 OpenQuoteContext、取完即关。
    建连前先探测端口 —— OpenD 不在时 SDK 构造会无限阻塞（见 broker_gateway.opend_reachable）。
    """
    def _fn(tickers):
        from app.core.data_engine.providers.moomoo_provider import _to_moomoo_code
        from app.core.execution.broker_gateway import opend_reachable
        import moomoo
        if not opend_reachable(host, port):
            raise RuntimeError(f"OpenD {host}:{port} 未在监听 —— 不建行情连接")
        ctx = moomoo.OpenQuoteContext(host=host, port=port)
        try:
            ret, df = ctx.get_market_snapshot([_to_moomoo_code(t) for t in tickers])
            if ret != moomoo.RET_OK:
                raise RuntimeError(f"get_market_snapshot 失败: {df}")
            return df
        finally:
            ctx.close()
    return _fn


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
                        allow_short: bool = False,
                        gateway=None, snapshot_fn=None) -> TradeProviders:
    """
    `mode="sim"`  → 仿真三件套（背 TR.1 估计 + 纸账户）。
    `mode="live"` → moomoo 实时三件套（Phase 12）：账户走 `gateway`（BrokerGateway），
                    盘口/可卖空走 `snapshot_fn`。缺任何一个就抛错 —— **绝不静默退回估计**，
                    否则会把"估计"当成"实时"用，那正是 T3 纪律要防的事。
    """
    if mode == "live":
        if gateway is None or snapshot_fn is None:
            raise ValueError("live 模式需要 gateway（账户）与 snapshot_fn（盘口）；不会退回估计")
        return TradeProviders(
            quote=LiveQuoteProvider(snapshot_fn),
            borrow=LiveBorrowProvider(allow_short=allow_short, snapshot_fn=snapshot_fn),
            account=LiveAccountProvider(gateway),
            mode="live",
        )
    if mode != "sim":
        raise ValueError(f"未知的 providers 模式 {mode!r}（sim | live）")
    if dataset is None or broker is None:
        raise ValueError("sim 模式需要 dataset 与 broker")
    return TradeProviders(
        quote=SimQuoteProvider(dataset),
        borrow=SimBorrowProvider(dataset, aum=aum, account_type=account_type,
                                 allow_short=allow_short),
        account=SimAccountProvider(broker, book_id=book_id),
        mode="sim",
    )
