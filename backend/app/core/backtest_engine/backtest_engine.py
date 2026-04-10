"""
BacktestEngine — 逐日回测主引擎

接收目标权重矩阵 (T×N)，逐日计算：
  持仓变化 → ADV 截断 → 滑点/成本 → 净/毛 PnL → 净值曲线

输出 BacktestResult，包含：
  equity_curve / gross_returns / net_returns
  positions / trade_log / turnover / signal
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .transaction_cost import (
    CostParams,
    LiquidityConstraint,
    TradeRecord,
    TransactionCostEngine,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """完整回测结果容器。"""

    equity_curve:   pd.Series       # 每日净值（初始=1.0）
    gross_returns:  pd.Series       # 毛收益率序列
    net_returns:    pd.Series       # 净收益率序列（扣除所有成本）
    positions:      pd.DataFrame    # (T×N) 每日实际持仓权重
    trade_log:      pd.DataFrame    # 完整交易日志
    turnover:       pd.Series       # 每日单边换手率
    signal:         pd.DataFrame    # 原始信号矩阵（IC 计算用）
    daily_cost_bps: pd.Series       # 每日成本（bps）


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    向量化回测引擎。

    Parameters
    ----------
    cost_params     : CostParams 实例（成本、滑点参数）
    initial_capital : 初始资金（USD）
    vol_window      : 计算日波动率所用的滚动窗口（交易日）
    """

    def __init__(
        self,
        cost_params:     Optional[CostParams] = None,
        initial_capital: float = 1_000_000.0,
        vol_window:      int   = 20,
    ) -> None:
        self.params          = cost_params or CostParams()
        self.initial_capital = initial_capital
        self.vol_window      = vol_window
        self._liq            = LiquidityConstraint(self.params)
        self._tc             = TransactionCostEngine(self.params)

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def run(
        self,
        weights: pd.DataFrame,
        prices:  pd.DataFrame,
        volume:  pd.DataFrame,
        signal:  pd.DataFrame,
    ) -> BacktestResult:
        """
        执行完整回测。

        Parameters
        ----------
        weights : (T×N) 目标权重矩阵（由 PortfolioConstructor 生成）
        prices  : (T×N) 收盘价
        volume  : (T×N) 成交量（股数）
        signal  : (T×N) 原始信号（用于 IC 计算）

        Returns
        -------
        BacktestResult
        """
        # --- 对齐 ---
        dates  = weights.index
        tickers = list(weights.columns)
        T, N   = len(dates), len(tickers)

        prices  = prices.reindex(index=dates, columns=tickers).ffill().fillna(0.0)
        volume  = volume.reindex(index=dates, columns=tickers).ffill().fillna(0.0)
        weights = weights.reindex(index=dates, columns=tickers).fillna(0.0)
        signal  = signal.reindex(index=dates, columns=tickers)

        prices_arr  = prices.to_numpy(dtype=float)
        volume_arr  = volume.to_numpy(dtype=float)
        weights_arr = weights.to_numpy(dtype=float)

        # --- 预计算 ADV & 日波动率 ---
        adv_usd_df = self._liq.compute_adv(volume, prices)         # (T×N)
        adv_usd    = adv_usd_df.to_numpy(dtype=float)

        price_ret     = pd.DataFrame(prices_arr, index=dates, columns=tickers).pct_change()
        daily_vol_df  = price_ret.rolling(self.vol_window, min_periods=2).std().fillna(0.02)
        daily_vol_arr = daily_vol_df.to_numpy(dtype=float)

        # --- ADV 流动性截断（整体） ---
        adj_weights = self._liq.apply(
            weights, adv_usd_df, self.initial_capital
        ).to_numpy(dtype=float)

        # --- 逐日迭代 ---
        equity          = self.initial_capital
        prev_w          = np.zeros(N)
        all_records:    List[TradeRecord] = []
        equity_series   = np.empty(T)
        gross_ret_arr   = np.empty(T)
        net_ret_arr     = np.empty(T)
        turnover_arr    = np.empty(T)
        cost_bps_arr    = np.empty(T)
        realized_pos    = np.empty((T, N))

        for t in range(T):
            target_w   = adj_weights[t]
            price_t    = prices_arr[t]
            adv_t      = adv_usd[t]
            vol_t      = daily_vol_arr[t]

            # 换手（单边）
            delta_w    = target_w - prev_w
            turnover_t = float(np.abs(delta_w).sum()) / 2.0

            # 计算成本
            cost_w, cost_usd, records = self._tc.compute(
                date         = dates[t],
                delta_w      = delta_w,
                prices       = price_t,
                adv_usd      = adv_t,
                daily_vol    = vol_t,
                portfolio_val = equity,
                tickers      = tickers,
            )
            all_records.extend(records)

            # 毛收益（价格涨跌）
            if t == 0:
                gross_ret = 0.0
            else:
                prev_price = prices_arr[t - 1]
                safe_prev  = np.where(prev_price == 0, np.nan, prev_price)
                price_chg  = (price_t - prev_price) / safe_prev   # (N,)
                gross_ret  = float(np.nansum(prev_w * price_chg))

            # 净收益
            cost_ret = float(cost_w.sum())
            net_ret  = gross_ret - cost_ret

            # 更新净值
            equity   = equity * (1 + net_ret)
            equity_series[t] = equity / self.initial_capital

            gross_ret_arr[t] = gross_ret
            net_ret_arr[t]   = net_ret
            turnover_arr[t]  = turnover_t
            cost_bps_arr[t]  = cost_ret * 10_000
            realized_pos[t]  = target_w
            prev_w           = target_w.copy()

            if t % 50 == 0:
                logger.debug(
                    "Backtest t=%d/%d date=%s equity=%.4f",
                    t, T, dates[t].date(), equity / self.initial_capital,
                )

        logger.info(
            "回测完成: %d 天 × %d 资产 | 最终净值=%.4f | 总交易=%d",
            T, N, equity_series[-1], len(all_records),
        )

        trade_log = TransactionCostEngine.records_to_df(all_records)

        return BacktestResult(
            equity_curve   = pd.Series(equity_series,  index=dates, name="equity"),
            gross_returns  = pd.Series(gross_ret_arr,  index=dates, name="gross_ret"),
            net_returns    = pd.Series(net_ret_arr,    index=dates, name="net_ret"),
            positions      = pd.DataFrame(realized_pos, index=dates, columns=tickers),
            trade_log      = trade_log,
            turnover       = pd.Series(turnover_arr,   index=dates, name="turnover"),
            signal         = signal,
            daily_cost_bps = pd.Series(cost_bps_arr,  index=dates, name="cost_bps"),
        )
