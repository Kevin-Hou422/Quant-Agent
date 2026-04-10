"""
Transaction Cost & Liquidity Model

组件：
  CostParams          — 成本参数数据类
  SlippageModel       — 平方根冲击法则 / 简化线性模型
  LiquidityConstraint — ADV 流动性上限截断
  TransactionCostEngine — 组合计算，返回滑点 + 净成本
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 成本参数
# ---------------------------------------------------------------------------

@dataclass
class CostParams:
    """
    回测成本参数配置。

    fixed_bps       : 单边固定手续费（bps，基点）
    min_ticket_fee  : 最小票面费（USD）
    spread_bps      : 买卖价差（bps）
    impact_coef     : 市场冲击系数（平方根法则）
    adv_window      : ADV 计算窗口（交易日）
    adv_cap_pct     : 单标的持仓上限 = adv_cap_pct × 20日ADV（USD）
    slippage_model  : 'sqrt'（平方根法则）或 'linear'（简化线性）
    """
    fixed_bps:      float = 5.0
    min_ticket_fee: float = 1.0
    spread_bps:     float = 2.0
    impact_coef:    float = 0.1
    adv_window:     int   = 20
    adv_cap_pct:    float = 0.10
    slippage_model: Literal["sqrt", "linear"] = "sqrt"


# ---------------------------------------------------------------------------
# 滑点模型
# ---------------------------------------------------------------------------

class SlippageModel:
    """
    计算每笔交易的滑点（单位：bps）。

    sqrt 模式（平方根冲击法则）：
        slippage_bps = spread/2 + impact * σ * sqrt(trade_usd / adv_usd)

    linear 模式（简化线性）：
        slippage_bps = 0.5 * spread + 0.1 * vol_impact_proxy
    """

    def __init__(self, params: CostParams) -> None:
        self.p = params

    def compute(
        self,
        trade_usd:  np.ndarray,   # (N,) 每个资产的交易金额（USD，取绝对值）
        adv_usd:    np.ndarray,   # (N,) 20日 ADV（USD）
        daily_vol:  np.ndarray,   # (N,) 日收益率波动率
    ) -> np.ndarray:              # (N,) 滑点 bps
        """向量化计算，零交易量位置返回 0。"""
        trade_abs = np.abs(trade_usd)
        adv_safe  = np.where(adv_usd <= 0, np.nan, adv_usd)

        if self.p.slippage_model == "sqrt":
            participation = trade_abs / adv_safe          # 参与率
            slippage = (
                self.p.spread_bps / 2.0
                + self.p.impact_coef * daily_vol * 10_000   # vol → bps scale
                * np.sqrt(np.nan_to_num(participation, nan=0.0))
            )
        else:  # linear
            slippage = (
                0.5 * self.p.spread_bps
                + 0.1 * daily_vol * 10_000
            ) * np.ones_like(trade_abs)

        # 无交易时滑点为 0
        slippage = np.where(trade_abs == 0, 0.0, slippage)
        return np.nan_to_num(slippage, nan=0.0)


# ---------------------------------------------------------------------------
# ADV 流动性约束
# ---------------------------------------------------------------------------

class LiquidityConstraint:
    """
    将权重矩阵中超过 ADV 上限的持仓截断，并重新归一化。

    上限规则：
        max_position_usd[t, i] = adv_cap_pct × adv_usd[t, i]
        weight_cap[t, i]        = max_position_usd[t, i] / portfolio_value
    """

    def __init__(self, params: CostParams) -> None:
        self.p = params

    def compute_adv(
        self,
        volume: pd.DataFrame,   # (T×N) 成交量（股数）
        prices: pd.DataFrame,   # (T×N) 收盘价
    ) -> pd.DataFrame:
        """计算 20日 ADV（USD），shape=(T×N)。"""
        dollar_vol = volume * prices                                   # USD 成交额
        adv = dollar_vol.rolling(
            window=self.p.adv_window,
            min_periods=max(1, self.p.adv_window // 2),
        ).mean()
        return adv.bfill().fillna(0.0)                                  # 初期用后向填充

    def apply(
        self,
        weights: pd.DataFrame,  # (T×N) 目标权重
        adv_usd: pd.DataFrame,  # (T×N) ADV（USD）
        portfolio_value: float,
    ) -> pd.DataFrame:
        """
        截断超 ADV 上限的权重，并重新做 L1 归一化。
        返回调整后的权重矩阵。
        """
        w   = weights.to_numpy(dtype=float).copy()
        adv = adv_usd.to_numpy(dtype=float)

        # 上限（以权重表示）
        max_usd = adv * self.p.adv_cap_pct                      # (T, N)
        cap_w   = np.where(
            portfolio_value > 0,
            max_usd / portfolio_value,
            np.inf,
        )

        # 截断（方向保留，幅度缩减）
        clipped = np.sign(w) * np.minimum(np.abs(w), cap_w)

        # 重新 L1 归一化
        l1 = np.nansum(np.abs(clipped), axis=1, keepdims=True)
        l1 = np.where(l1 == 0, 1.0, l1)
        clipped = clipped / l1

        return pd.DataFrame(clipped, index=weights.index, columns=weights.columns)


# ---------------------------------------------------------------------------
# TransactionCostEngine（组合入口）
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    date:         pd.Timestamp
    ticker:       str
    direction:    str      # 'BUY' / 'SELL'
    shares:       float
    price:        float
    slippage_bps: float
    cost_usd:     float
    net_price:    float


class TransactionCostEngine:
    """
    给定 delta_weights（换手量）、价格、ADV、日波动率，
    计算每笔交易的滑点和总成本，并生成 trade_log。
    """

    def __init__(self, params: CostParams) -> None:
        self.p        = params
        self._slippage = SlippageModel(params)

    def compute(
        self,
        date:          pd.Timestamp,
        delta_w:       np.ndarray,   # (N,) 权重变化（有符号）
        prices:        np.ndarray,   # (N,) 当日收盘价
        adv_usd:       np.ndarray,   # (N,) 20日 ADV（USD）
        daily_vol:     np.ndarray,   # (N,) 日收益率波动率
        portfolio_val: float,
        tickers:       list[str],
    ) -> Tuple[np.ndarray, float, List[TradeRecord]]:
        """
        Returns
        -------
        net_cost_weight : (N,) 每个资产的成本（以权重单位，从净值扣除）
        total_cost_usd  : float 当日总成本（USD）
        records         : List[TradeRecord]
        """
        trade_usd = delta_w * portfolio_val          # (N,) 交易金额

        # 滑点
        slip_bps  = self._slippage.compute(trade_usd, adv_usd, daily_vol)

        # 固定手续费（单边）
        notional   = np.abs(trade_usd)
        fixed_cost = notional * self.p.fixed_bps * 1e-4
        slip_cost  = notional * slip_bps          * 1e-4

        # 最小票面费
        ticket_fee = np.where(
            notional > 0,
            np.maximum(fixed_cost + slip_cost, self.p.min_ticket_fee),
            0.0,
        )

        total_cost_usd = float(ticket_fee.sum())

        # 构建 trade_log 记录
        records: List[TradeRecord] = []
        for i, (dw, p, s_bps, cost) in enumerate(
            zip(delta_w, prices, slip_bps, ticket_fee)
        ):
            if abs(dw) < 1e-10:
                continue
            direction = "BUY" if dw > 0 else "SELL"
            shares    = abs(dw) * portfolio_val / p if p > 0 else 0.0
            net_price = p * (1 + (s_bps * 1e-4) * (1 if dw > 0 else -1))
            records.append(TradeRecord(
                date=date, ticker=tickers[i],
                direction=direction, shares=shares,
                price=p, slippage_bps=float(s_bps),
                cost_usd=float(cost), net_price=float(net_price),
            ))

        # 成本归一化为权重单位（从净值中扣除）
        net_cost_w = ticket_fee / max(portfolio_val, 1.0)
        return net_cost_w, total_cost_usd, records

    @staticmethod
    def records_to_df(records: List[TradeRecord]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(columns=[
                "date", "ticker", "direction", "shares",
                "price", "slippage_bps", "cost_usd", "net_price",
            ])
        return pd.DataFrame([r.__dict__ for r in records])
