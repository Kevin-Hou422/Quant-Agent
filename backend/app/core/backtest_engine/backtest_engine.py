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

        # --- E1: 预计算不依赖 equity 的数组（减少 Python 循环体积）---
        # prev_w_mat[t] = adj_weights[t-1]，t=0 时为零向量
        prev_w_mat  = np.vstack([np.zeros((1, N)), adj_weights[:-1]])    # (T, N)
        delta_w_mat = adj_weights - prev_w_mat                           # (T, N)

        # 换手（单边）— 全量向量化
        turnover_arr_pre = np.sum(np.abs(delta_w_mat), axis=1) / 2.0    # (T,)

        # 价格涨跌矩阵 (T, N)
        price_chg_mat             = np.zeros((T, N))
        safe_prev_mat             = np.where(prices_arr[:-1] == 0, np.nan, prices_arr[:-1])
        price_chg_mat[1:]         = (prices_arr[1:] - prices_arr[:-1]) / safe_prev_mat

        # 毛收益与多空腿分离 — 全量向量化
        long_w_mat  = np.maximum(prev_w_mat, 0.0)
        short_w_mat = np.minimum(prev_w_mat, 0.0)
        gross_ret_pre = np.nansum(prev_w_mat  * price_chg_mat, axis=1)   # (T,)
        long_ret_pre  = np.nansum(long_w_mat  * price_chg_mat, axis=1)   # (T,)
        short_ret_pre = np.nansum(short_w_mat * price_chg_mat, axis=1)   # (T,)
        gross_ret_pre[0] = long_ret_pre[0] = short_ret_pre[0] = 0.0      # 首日无价格变化

        # 借券成本 — 全量向量化（不依赖 equity，只依赖权重）
        short_exp_arr  = np.sum(np.maximum(-prev_w_mat, 0.0), axis=1)    # (T,)
        borrow_arr_pre = short_exp_arr * daily_borrow_rate                # (T,)

        # --- 逐日迭代（仅剩 equity 递推 + 交易成本归一化）---
        equity          = self.initial_capital
        all_records:    List[TradeRecord] = []
        equity_series   = np.zeros(T)
        gross_ret_arr   = np.zeros(T)
        net_ret_arr     = np.zeros(T)
        long_ret_arr    = np.zeros(T)
        short_ret_arr   = np.zeros(T)
        cost_bps_arr    = np.zeros(T)
        realized_pos    = adj_weights.copy()   # positions = target weights
        ruin_date: Optional[pd.Timestamp] = None

        for t in range(T):
            # 交易成本（仍需 equity 做归一化，无法向量化）
            cost_w, _, records = self._tc.compute(
                date          = dates[t],
                delta_w       = delta_w_mat[t],
                prices        = prices_arr[t],
                adv_usd       = adv_usd[t],
                daily_vol     = daily_vol_arr[t],
                portfolio_val = equity,
                tickers       = tickers,
            )
            all_records.extend(records)

            cost_ret = float(cost_w.sum())
            net_ret  = gross_ret_pre[t] - cost_ret - borrow_arr_pre[t]

            # 更新净值（递推，必须串行）
            equity               = equity * (1 + net_ret)
            equity_series[t]     = equity / self.initial_capital
            gross_ret_arr[t]     = gross_ret_pre[t]
            net_ret_arr[t]       = net_ret
            long_ret_arr[t]      = long_ret_pre[t]
            short_ret_arr[t]     = short_ret_pre[t]
            cost_bps_arr[t]      = (cost_ret + borrow_arr_pre[t]) * 10_000

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

        # turnover_arr 已向量化，直接使用
        turnover_arr = turnover_arr_pre

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
