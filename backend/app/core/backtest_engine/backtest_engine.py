"""
BacktestEngine — 逐日回测主引擎（时钟 c2，见 app/core/clock.py）

接收目标权重矩阵 (T×N)：第 T 行 = T 收盘冻结的目标，**T+1** 按限价规则执行。逐日：
  旧仓隔夜收益（昨收→今开）→ 开盘撮合昨天的目标（截至昨天的 ADV 上限、限价规则）→ 成本 →
  新仓盘中收益（今开→今收）→ 收盘净值

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
    drift_weights,
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
        limit_band_bps:  Optional[float] = None,
    ) -> None:
        from app.core.clock import default_band
        self.max_net         = max_net
        #: T+1 限价单的限价带（小数）。缺省 = settings.exec_limit_band_bps（与券商下单同一参数）
        self.band            = default_band() if limit_band_bps is None else float(limit_band_bps) * 1e-4
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
        ohlc:    Optional[dict] = None,
    ) -> BacktestResult:
        """
        执行完整回测。

        Parameters
        ----------
        weights : (T×N) 目标权重矩阵：第 T 行 = T 收盘冻结的目标（T+1 执行）
        prices  : (T×N) 收盘价
        volume  : (T×N) 成交量（股数）
        signal  : (T×N) 原始信号（用于 IC 计算）
        ohlc    : {"open", "high", "low"}（时钟 c2 必需：T+1 按限价规则成交）

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
        from app.core.clock import limit_fill_prices, require_ohlc
        ohlc_arr = require_ohlc(ohlc, dates, tickers)
        open_raw = ohlc_arr["open"]
        high_arr, low_arr = ohlc_arr["high"], ohlc_arr["low"]

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

        # F10：每天先把昨仓按今日价格漂移成**调仓前权重**，再撮合到目标；成交差额 = 目标 − 漂移后。
        # 复审 2026-10-19 N08：ADV 成交上限按**调仓前市值**换成权重 —— 它依赖当天的权益，
        # 所以撮合与净值递推放在同一个逐日循环里（下面），不再预先算好。
        adj_weights = np.zeros((T, N), dtype=float)

        # --- E3: 从实际日期范围动态计算年化系数 ---
        if T >= 2:
            calendar_years = max((dates[-1] - dates[0]).days / 365.25, 1.0 / 365.25)
            tdays_per_year = float(T) / calendar_years
        else:
            tdays_per_year = 252.0

        # --- F5: 日化借券成本（年化 bps → 日化率）---
        daily_borrow_rate = self.params.short_borrow_annual_bps * 1e-4 / tdays_per_year

        # --- 逐日迭代（时钟 c2）：隔夜 → 开盘撮合昨天的目标 → 成本 → 盘中 → 收盘净值 ---
        equity          = self.initial_capital
        prev_filled     = np.zeros(N, dtype=float)       # 相对**昨收净值**的持仓权重
        delta_w_mat     = np.zeros((T, N), dtype=float)
        all_records:    List[TradeRecord] = []
        equity_series   = np.zeros(T)
        gross_ret_arr   = np.zeros(T)
        net_ret_arr     = np.zeros(T)
        long_ret_arr    = np.zeros(T)
        short_ret_arr   = np.zeros(T)
        cost_bps_arr    = np.zeros(T)
        realized_pos    = adj_weights          # 成交那一刻的持仓权重（相对开盘市值）
        ruin_date: Optional[pd.Timestamp] = None

        if T:
            equity_series[0] = 1.0 if self.initial_capital else 0.0     # 第一天只决策，不成交
        for t in range(1, T):
            c_prev, c_now = prices_arr[t - 1], prices_arr[t]
            # 缺开盘价 → 当天按"开盘 = 昨收"估值（没有隔夜收益），且下面的限价规则判不成交
            o = np.where(np.isfinite(open_raw[t]) & (open_raw[t] > 0), open_raw[t], c_prev)
            with np.errstate(divide="ignore", invalid="ignore"):
                on_chg = np.where((c_prev > 0) & np.isfinite(c_prev), o / c_prev - 1.0, 0.0)
                in_chg = np.where((o > 0) & np.isfinite(c_now) & (c_now > 0), c_now / o - 1.0, 0.0)
            # 1) 隔夜：旧仓（相对昨收净值）× 昨收→今开；借券按昨收市值
            r_on = float(np.nansum(prev_filled * on_chg))
            borrow = float(np.sum(np.maximum(-prev_filled, 0.0))) * daily_borrow_rate
            v_open = equity * (1.0 + r_on)
            pre = drift_weights(prev_filled, on_chg)              # 相对开盘市值
            # 2) 开盘撮合**昨天收盘**定下的目标：截至昨天的 ADV 上限；限价规则下碰不到的名字上限置 0
            target = tgt_arr[t - 1]
            if v_open > 0:
                cap = adv_usd[t - 1] * self.params.max_participation_pct / v_open
            else:
                cap = np.full(N, np.inf)
            px = limit_fill_prices(np.sign(target - pre), c_prev, open_raw[t], high_arr[t], low_arr[t],
                                   self.band)
            cap = np.where(np.isfinite(px), cap, 0.0)
            fr = simulate_partial_fills(pre, target, cap, max_net=self.max_net)
            delta = fr.filled_w - pre
            delta_w_mat[t] = delta
            exec_px = np.where(np.isfinite(px), px, o)
            with np.errstate(divide="ignore", invalid="ignore"):
                px_ratio = np.where(o > 0, exec_px / o, 1.0)
            # 3) 成本：按**成交价**的成交额收费；成交价相对开盘价的差额是执行成本（买高 / 卖低为正）
            _, cost_usd, records = self._tc.compute(
                date          = dates[t],
                delta_w       = delta * px_ratio,
                prices        = exec_px,
                adv_usd       = adv_usd[t - 1],
                daily_vol     = daily_vol_arr[t - 1],
                portfolio_val = max(v_open, 0.0),
                tickers       = tickers,
            )
            all_records.extend(records)
            exec_cost = float(np.nansum(delta * (px_ratio - 1.0))) * max(v_open, 0.0)
            # 4) 盘中：新仓（相对开盘市值）× 今开→今收
            r_in = float(np.nansum(fr.filled_w * in_chg))
            close_val = v_open * (1.0 + r_in)
            new_equity = close_val - float(cost_usd) - exec_cost - borrow * equity
            scale_open = (v_open / equity) if equity > 0 else 0.0
            gross_ret = close_val / equity - 1.0 if equity > 0 else 0.0
            net_ret = new_equity / equity - 1.0 if equity > 0 else 0.0
            cost_ret = (float(cost_usd) + exec_cost) / equity if equity > 0 else 0.0
            long_ret = (float(np.nansum(np.maximum(prev_filled, 0.0) * on_chg))
                        + float(np.nansum(np.maximum(fr.filled_w, 0.0) * in_chg)) * scale_open)
            short_ret = (float(np.nansum(np.minimum(prev_filled, 0.0) * on_chg))
                         + float(np.nansum(np.minimum(fr.filled_w, 0.0) * in_chg)) * scale_open)
            # 收盘持仓（相对收盘净值）：成交后股数按今收估值，费用 / 借券从现金扣（复审 N08-b 的归一）
            if new_equity > 0:
                prev_filled = fr.filled_w * (1.0 + in_chg) * v_open / new_equity
            else:
                prev_filled = np.zeros(N)
            adj_weights[t] = fr.filled_w
            equity = new_equity
            equity_series[t]     = equity / self.initial_capital if self.initial_capital else 0.0
            gross_ret_arr[t]     = gross_ret
            net_ret_arr[t]       = net_ret
            long_ret_arr[t]      = long_ret
            short_ret_arr[t]     = short_ret
            cost_bps_arr[t]      = (cost_ret + borrow) * 10_000

            # ── E6 熔断：净值归零后停止模拟 ─────────────────────────────
            # 初始资金为 0 的回测（只看权重、不设容量上限）没有"净值归零"一说
            if self.initial_capital > 0 and equity <= 0:
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
                    t, T, dates[t].date(), equity_series[t],
                )

        # 换手（单边）= 真实成交量（漂移后 → 成交后）
        turnover_arr = np.sum(np.abs(delta_w_mat), axis=1) / 2.0

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
