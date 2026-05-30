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

    equity_curve:   pd.Series            # 每日净值（初始=1.0）
    gross_returns:  pd.Series            # 毛收益率序列
    net_returns:    pd.Series            # 净收益率序列（扣除所有成本）
    positions:      pd.DataFrame         # (T×N) 每日实际持仓权重
    trade_log:      pd.DataFrame         # 完整交易日志
    turnover:       pd.Series            # 每日单边换手率
    signal:         pd.DataFrame         # 原始信号矩阵（IC 计算用）
    daily_cost_bps: pd.Series            # 每日成本（bps）
    long_returns:   pd.Series  = field(default=None)  # 多头腿每日毛收益（O4）
    short_returns:  pd.Series  = field(default=None)  # 空头腿每日毛收益（O4）
    ruin_date:      Optional[pd.Timestamp] = None     # 净值穿零熔断日期（正常为 None）


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

        # E7 修复: ffill 限制最多 5 天，防止停牌/退市股票价格被无限前向填充
        prices  = prices.reindex(index=dates, columns=tickers).ffill(limit=5).fillna(0.0)
        volume  = volume.reindex(index=dates, columns=tickers).ffill(limit=5).fillna(0.0)
        weights = weights.reindex(index=dates, columns=tickers).fillna(0.0)
        signal  = signal.reindex(index=dates, columns=tickers)

        prices_arr  = prices.to_numpy(dtype=float)
        # volume/weights used as DataFrames by liq helpers; no separate numpy arrays needed

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

        # --- E3: 从实际日期范围动态计算年化系数 ---
        if T >= 2:
            calendar_years = max((dates[-1] - dates[0]).days / 365.25, 1.0 / 365.25)
            tdays_per_year = float(T) / calendar_years
        else:
            tdays_per_year = 252.0

        # --- F5: 日化借券成本（年化 bps → 日化率）---
        daily_borrow_rate = self.params.short_borrow_annual_bps * 1e-4 / tdays_per_year

        # --- 逐日迭代 ---
        equity          = self.initial_capital
        prev_w          = np.zeros(N)
        all_records:    List[TradeRecord] = []
        equity_series   = np.zeros(T)      # 预填 0 — 熔断后剩余天数保持 0
        gross_ret_arr   = np.zeros(T)
        net_ret_arr     = np.zeros(T)
        long_ret_arr    = np.zeros(T)      # O4: 多头腿毛收益
        short_ret_arr   = np.zeros(T)      # O4: 空头腿毛收益
        turnover_arr    = np.zeros(T)
        cost_bps_arr    = np.zeros(T)
        realized_pos    = np.zeros((T, N))
        ruin_date: Optional[pd.Timestamp] = None

        for t in range(T):
            target_w   = adj_weights[t]
            price_t    = prices_arr[t]
            adv_t      = adv_usd[t]
            vol_t      = daily_vol_arr[t]

            # 换手（单边）
            delta_w    = target_w - prev_w
            turnover_t = float(np.abs(delta_w).sum()) / 2.0

            # 计算交易成本
            cost_w, _, records = self._tc.compute(
                date          = dates[t],
                delta_w       = delta_w,
                prices        = price_t,
                adv_usd       = adv_t,
                daily_vol     = vol_t,
                portfolio_val = equity,
                tickers       = tickers,
            )
            all_records.extend(records)

            # 毛收益（价格涨跌）+ 多空腿分离
            if t == 0:
                gross_ret = long_ret = short_ret = 0.0
            else:
                prev_price  = prices_arr[t - 1]
                safe_prev   = np.where(prev_price == 0, np.nan, prev_price)
                price_chg   = (price_t - prev_price) / safe_prev         # (N,)
                long_w_prev  = np.maximum(prev_w, 0.0)                   # O4
                short_w_prev = np.minimum(prev_w, 0.0)                   # O4
                long_ret    = float(np.nansum(long_w_prev  * price_chg))
                short_ret   = float(np.nansum(short_w_prev * price_chg))
                gross_ret   = long_ret + short_ret

            # F5: 做空借券成本（空头净敞口每日扣除）
            short_exposure = float(np.sum(np.maximum(-prev_w, 0.0)))
            borrow_cost    = short_exposure * daily_borrow_rate

            # 净收益 = 毛收益 - 交易成本 - 借券成本
            cost_ret = float(cost_w.sum())
            net_ret  = gross_ret - cost_ret - borrow_cost

            # 更新净值
            equity   = equity * (1 + net_ret)
            equity_series[t] = equity / self.initial_capital

            gross_ret_arr[t] = gross_ret
            net_ret_arr[t]   = net_ret
            long_ret_arr[t]  = long_ret
            short_ret_arr[t] = short_ret
            turnover_arr[t]  = turnover_t
            cost_bps_arr[t]  = (cost_ret + borrow_cost) * 10_000
            realized_pos[t]  = target_w
            prev_w           = target_w.copy()

            # ── E6 熔断：净值归零后停止模拟 ─────────────────────────────
            if equity <= 0:
                ruin_date = dates[t]
                logger.warning(
                    "净值归零熔断 | date=%s | day=%d/%d | 剩余 %d 天填充为 0",
                    dates[t].date() if hasattr(dates[t], "date") else dates[t],
                    t + 1, T, T - t - 1,
                )
                break
            # ─────────────────────────────────────────────────────────────

            if t % 50 == 0:
                logger.debug(
                    "Backtest t=%d/%d date=%s equity=%.4f",
                    t, T, dates[t].date(), equity / self.initial_capital,
                )

        logger.info(
            "回测完成: %d 天 × %d 资产 | tdays/yr=%.1f | 最终净值=%.4f | 总交易=%d%s",
            T, N, tdays_per_year,
            equity_series[equity_series != 0][-1] if (equity_series != 0).any() else 0.0,
            len(all_records),
            f" | 熔断={ruin_date.date()}" if ruin_date is not None else "",
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
            long_returns   = pd.Series(long_ret_arr,  index=dates, name="long_ret"),
            short_returns  = pd.Series(short_ret_arr, index=dates, name="short_ret"),
            ruin_date      = ruin_date,
        )
