"""
context.py — 交易现实引擎 TradingContext（Phase TR）

系统**自己**从"真实情景 + 免费数据"推导财务关键参数，取代硬编码默认：
  - 成本：价差用 Corwin-Schultz 从免费 H/L 估（每名、时变）；佣金/规费来自**券商档位**（现实事实）；
    冲击来自参与率（$10k 下≈0）。→ 取代 `spread_bps=2 / fixed_bps=5 / min_ticket=$1` 这些机构默认。
  - 可做空性：由 `allow_short`（用户声明）+ 账户类型 + 流动性启发式推导；long-only 时全不可做空。
  - 调仓带：无交易带宽由成本推导（价差越宽、带越宽），取代隐含"每日全额调仓"。
  - 可交易池：流动性/价格过滤。

三层纪律（见 DEV_LESSONS §J）：
  T1 用户声明的现实事实 → 显式配置（AUM/账户类型/allow_short/券商）。
  T2 数据推导的估计（本模块）→ 每次随当前数据重算、明确是"估计"。
  T3 只有交易当时才知道的（真实盘口价差/借券/成交价）→ 不在此写死，交易时走 provider（后续）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from app.core.trading_context.spread import corwin_schultz_spread_bps

WidePanel = Dict[str, pd.DataFrame]


@dataclass
class BrokerProfile:
    """券商真实费率（现实事实，非默认）。moomoo 美股：佣金免费，仅规费。"""
    name:                  str
    commission_bps:        float = 0.0     # 佣金（notional bps）
    commission_per_share:  float = 0.0     # 佣金（$/股）
    commission_min:        float = 0.0     # 每单最低佣金（$）
    reg_fee_bps:           float = 0.15    # SEC+FINRA 规费近似（仅卖出，均摊到单边 ~0.15bps）


# moomoo 美股：佣金免费（推广）；仅规费。用户券商 = moomoo。
MOOMOO_US = BrokerProfile(name="moomoo_us", commission_bps=0.0, commission_per_share=0.0,
                          commission_min=0.0, reg_fee_bps=0.15)


@dataclass
class TradingContextResult:
    spread_bps:          pd.Series            # 每名估计价差（bps）
    tradable:            pd.Series            # 每名是否可交易（bool）
    shortable:           pd.Series            # 每名是否可做空（bool）
    est_cost_bps_oneway: pd.Series            # 每名单边成本估计（bps）= 半价差 + 佣金 + 规费
    rebalance_band:      float                # 无交易带宽（权重漂移阈值）
    aum:                 float
    allow_short:         bool
    notes:               List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "aum": self.aum, "allow_short": self.allow_short,
            "rebalance_band": round(self.rebalance_band, 5),
            "n_tradable": int(self.tradable.sum()),
            "n_shortable": int(self.shortable.sum()),
            "median_spread_bps": round(float(self.spread_bps.median()), 3),
            "median_cost_oneway_bps": round(float(self.est_cost_bps_oneway.median()), 3),
            "notes": self.notes,
        }


class TradingContext:
    """
    Parameters（T1：用户声明的现实事实，显式，不是隐藏默认）
    ----------
    aum          : 真实资金（美元）。
    account_type : "cash" | "margin"（决定能否做空）。
    allow_short  : 是否允许做空（用户可暂关；关 → long-only）。
    broker       : 券商费率档（现实事实）。
    min_price    : 可交易最低价（避开 penny stock 的巨宽价差）。
    min_adv_usd  : 可交易最低 20 日 $ADV。
    """

    def __init__(
        self,
        aum: float,
        account_type: str = "margin",
        allow_short: bool = False,
        broker: BrokerProfile = MOOMOO_US,
        min_price: float = 5.0,
        min_adv_usd: float = 1_000_000.0,
    ) -> None:
        self.aum = float(aum)
        self.account_type = account_type
        self.allow_short = bool(allow_short)
        self.broker = broker
        self.min_price = float(min_price)
        self.min_adv_usd = float(min_adv_usd)

    def analyze(self, dataset: WidePanel) -> TradingContextResult:
        high, low = dataset["high"], dataset["low"]
        close = dataset["close"]
        volume = dataset.get("volume")
        if volume is None:
            volume = pd.DataFrame(1e6, index=close.index, columns=close.columns)

        # ---- T2：价差（Corwin-Schultz，免费 H/L）----
        spread_bps = corwin_schultz_spread_bps(high, low)

        px = close.ffill().iloc[-1]
        adv_usd = (close * volume).tail(20).mean(axis=0)

        # ---- 可交易池：价格 + 流动性 ----
        tradable = (px > self.min_price) & (adv_usd > self.min_adv_usd)
        tradable = tradable.reindex(close.columns).fillna(False)

        # ---- 可做空性：long-only / 现金账户 → 全不可做空；否则易借启发式 ----
        notes: List[str] = []
        if not self.allow_short or self.account_type != "margin":
            shortable = pd.Series(False, index=close.columns)
            notes.append(f"long-only（allow_short={self.allow_short}, account={self.account_type}）：不做空")
        else:
            shortable = tradable & (adv_usd > 5 * self.min_adv_usd) & (px > 10.0)
            notes.append("long-short：仅在流动大盘（易借启发式）子集里做空")

        # ---- 单边成本估计（bps）= 半价差 + 佣金 + 规费 ----
        est_cost = spread_bps / 2.0 + self.broker.commission_bps + self.broker.reg_fee_bps
        est_cost = est_cost.reindex(close.columns).fillna(est_cost.median())
        notes.append(f"券商={self.broker.name}：佣金={self.broker.commission_bps}bps，规费≈{self.broker.reg_fee_bps}bps")

        # ---- 调仓带（无交易带）：与成本挂钩（成本越高、带越宽）----
        med_cost_frac = float(np.nanmedian(est_cost.values)) / 1e4
        rebalance_band = float(np.clip(2.0 * med_cost_frac, 0.001, 0.05))
        notes.append(f"无交易带={rebalance_band:.4f}（≈2×中位单边成本；成本越高越少调仓）")

        # ---- $10k 洞察：小资金下容量/冲击几乎不 binding ----
        if self.aum <= 100_000:
            notes.append(f"AUM=${self.aum:,.0f} 极小：市场冲击≈0、容量几乎不 binding；成本≈半价差主导")

        return TradingContextResult(
            spread_bps=spread_bps.reindex(close.columns).fillna(0.0),
            tradable=tradable, shortable=shortable,
            est_cost_bps_oneway=est_cost, rebalance_band=rebalance_band,
            aum=self.aum, allow_short=self.allow_short, notes=notes,
        )
