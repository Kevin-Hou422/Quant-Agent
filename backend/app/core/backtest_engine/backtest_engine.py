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
    simulate_partial_fills,
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
    max_net         : 组合净敞口上限 |Σw|。**None = 明确选择不限制**，
                      与 `PaperBroker` 的默认一致 —— 两个引擎必须同款，
                      否则"统一语义"只统一了一半。
                      需要限制的调用方显式传入（生产链路由 RiskLimits.max_net 给）。
    """

    def __init__(
        self,
        cost_params:     Optional[CostParams] = None,
        initial_capital: float = 1_000_000.0,
        vol_window:      int   = 20,
        max_net:         Optional[float] = None,
    ) -> None:
        self.max_net         = max_net
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

        # --- 流动性：逐日按**交易差额**部分成交（执行层语义）---
        #
        # 【缺陷 A-6 后半，2026-09-21 修】原实现是
        #     adj_weights = self._liq.apply(weights, adv_usd_df, capital)
        # 即对**整个 T×N 持仓矩阵**做 water-filling。两处与执行层分家：
        #
        #   ① 口径错：它裁的是**目标持仓**，而流动性约束的是**当天能成交多少**。
        #      持仓 0.5 且昨天已经持有 0.5 的名字今天根本不需要交易，
        #      却照样被 ADV 上限削掉。
        #   ② 会**再分配**：water-filling 把 A 被削掉的额度摊给 B
        #      （[0.9,-0.1] → [0.01,-0.99]，10% 的对冲腿变成 99% 的方向性空头）。
        #      券商不会因为 A 买不到就多买 B。
        #
        # 执行层（PaperBroker）已于 2026-09-18/19 改为 `simulate_partial_fills`，
        # 于是两个引擎在限流场景下语义分家 —— `test_replay_matches_backtest_engine`
        # 的 1e-9 对账用的是上限不绑定的数据，**看不见**这个分叉。
        #
        # 现在回测走同一份原语、同一个 cap 公式，逐日推进（**路径依赖**：
        # 今天成不了的量不会凭空消失，也不会被摊给别人，而是留到以后接着成交）。
        #
        # 用户决定 #4：接受由此产生的历史回测收益变化，前后对比见
        # docs/A6_ENGINE_UNIFICATION.md。
        tgt_arr = weights.to_numpy(dtype=float)
        if self.initial_capital > 0:
            cap_trade_mat = (
                adv_usd * self.params.max_participation_pct / self.initial_capital)
        else:
            cap_trade_mat = np.full_like(adv_usd, np.inf)

        adj_weights = np.zeros((T, N), dtype=float)
        prev_filled = np.zeros(N, dtype=float)
        for t in range(T):
            fr = simulate_partial_fills(
                prev_filled, tgt_arr[t], cap_trade_mat[t], max_net=self.max_net)
            adj_weights[t] = fr.filled_w
            prev_filled = fr.filled_w

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
