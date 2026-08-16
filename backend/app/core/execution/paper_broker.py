"""
paper_broker.py — Paper Trading 模拟经纪（Task 7.2，2026-08-02）

逐日、有状态的模拟成交器，**严格复刻 BacktestEngine 的记账口径**（复用同一
`TransactionCostEngine` 与 `LiquidityConstraint`，禁止另立成本模型），因此对同一
权重序列，PaperBroker 逐日累计净值与 RealisticBacktester 同区间回测**逐位一致**。
（未来接入 T+1 开盘价成交时，会引入 roadmap 记载的 <1bp/日 差异；replay 用收盘价
对账，差异为浮点级。）

核心 API：
  step(alpha_id, date, target_w, prices_t, prices_prev, adv_usd, daily_vol, ...)
      → 单日原子记账（幂等持久化到 PositionStore），返回 DailyPnL
  replay(alpha_id, weights_df, prices_df, volume_df)
      → 便捷：内部按 BacktestEngine 口径算 adv/vol，逐日 step，返回净值 Series
        （用于对账、历史补跑）
  get_positions / latest_equity → 崩溃恢复读取

记账（与 BacktestEngine 一致）：
  持有昨仓 prev_w 度过今日 → gross = Σ prev_w·(P_t/P_{t-1}−1)
  借券 = Σ max(−prev_w,0)·日化借券率
  今日按 target 调仓（先 ADV 上限投影）→ delta = filled − prev_w → 成本
  net = gross − cost − borrow ; equity_t = equity_{t-1}·(1+net)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from ..backtest_engine.transaction_cost import (
    CostParams, LiquidityConstraint, TransactionCostEngine, project_to_capped_l1,
)
from ...db.position_store import DailyPnL, PositionStore

logger = logging.getLogger(__name__)


class PaperBroker:
    def __init__(
        self,
        store:           Optional[PositionStore] = None,
        cost_params:     Optional[CostParams] = None,
        initial_capital: float = 1_000_000.0,
    ) -> None:
        self.store           = store or PositionStore()
        self.params          = cost_params or CostParams()
        self.initial_capital = initial_capital
        self._liq            = LiquidityConstraint(self.params)
        self._tc             = TransactionCostEngine(self.params)

    # ------------------------------------------------------------------
    # 单日原子记账
    # ------------------------------------------------------------------

    def step(
        self,
        alpha_id:       int,
        date,
        target_w:       pd.Series,      # (N,) 今日目标权重（索引=ticker）
        prices_t:       pd.Series,      # (N,) 今日成交/估值价
        prices_prev:    pd.Series,      # (N,) 昨日价
        adv_usd:        pd.Series,      # (N,) 今日 20日 ADV（USD）
        daily_vol:      pd.Series,      # (N,) 今日日波动率
        tdays_per_year: float = 252.0,
    ) -> DailyPnL:
        """
        执行 alpha 在 date 的单日模拟成交并原子持久化（幂等：同日重跑覆盖）。
        昨仓与净值从 PositionStore 读取（崩溃恢复安全）。
        """
        tickers = list(target_w.index)
        # 幂等/续跑：状态取**严格早于 date** 的最近一日（非全局最近），
        # 使重跑第 t 日基于 t-1 状态，全量重放逐位可复现。
        equity, prev_pos = self.store.state_before(alpha_id, date)

        prev_w = np.array([prev_pos.get(tk, 0.0) for tk in tickers], dtype=float)
        tgt    = target_w.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
        p_t    = prices_t.reindex(tickers).to_numpy(dtype=float)
        p_prev = prices_prev.reindex(tickers).to_numpy(dtype=float)
        adv    = np.nan_to_num(adv_usd.reindex(tickers).to_numpy(dtype=float), nan=0.0)
        vol    = np.nan_to_num(daily_vol.reindex(tickers).to_numpy(dtype=float), nan=0.02)

        # 1) 持有昨仓度过今日：毛收益
        with np.errstate(divide="ignore", invalid="ignore"):
            price_chg = np.where(p_prev == 0, 0.0, (p_t - p_prev) / p_prev)
        gross_ret = float(np.nansum(prev_w * price_chg))

        # 2) 借券成本（昨仓空头）
        daily_borrow = self.params.short_borrow_annual_bps * 1e-4 / max(tdays_per_year, 1.0)
        borrow_ret = float(np.sum(np.maximum(-prev_w, 0.0)) * daily_borrow)

        # 3) ADV 上限投影（与 BacktestEngine 一致：以 initial_capital 计）
        if self.initial_capital > 0:
            cap_w = adv * self.params.adv_cap_pct / self.initial_capital
        else:
            cap_w = np.full_like(adv, np.inf)
        filled = project_to_capped_l1(tgt[np.newaxis, :], cap_w[np.newaxis, :], target=1.0)[0]

        # 4) 调仓成本
        delta = filled - prev_w
        cost_w, _cost_usd_total, _ = self._tc.compute(
            date=date, delta_w=delta, prices=p_t, adv_usd=adv, daily_vol=vol,
            portfolio_val=equity * self.initial_capital, tickers=tickers,
        )
        cost_ret = float(np.sum(cost_w))

        # 5) 净收益 + 净值
        net_ret   = gross_ret - cost_ret - borrow_ret
        equity_new = equity * (1.0 + net_ret)

        # 6) 组装 fills（仅有交易的名）+ 新持仓
        fills: List[dict] = []
        # 逐名成本（USD）——按 |delta| 占比分摊 total（TransactionCostEngine 已逐名，
        # 这里用 cost_w×组合市值近似逐名 USD，仅供审计展示）
        pv = equity * self.initial_capital
        for i, tk in enumerate(tickers):
            if abs(delta[i]) < 1e-12 and abs(filled[i]) < 1e-12:
                continue
            reject = "adv_cap" if abs(filled[i]) < abs(tgt[i]) - 1e-9 else ""
            fills.append({
                "ticker": tk, "target_weight": float(tgt[i]),
                "filled_weight": float(filled[i]), "fill_price": float(p_t[i]),
                "cost_usd": float(cost_w[i] * pv), "reject_reason": reject,
            })
        positions = {tk: float(filled[i]) for i, tk in enumerate(tickers) if abs(filled[i]) > 1e-12}

        pnl = DailyPnL(
            alpha_id=alpha_id, date=str(_as_date(date)),
            gross_ret=gross_ret, net_ret=net_ret,
            cost_bps=(cost_ret + borrow_ret) * 1e4, equity=equity_new,
        )
        self.store.record_day(alpha_id, date, positions, fills, pnl)
        return pnl

    # ------------------------------------------------------------------
    # 便捷：整段回放（对账 / 历史补跑）
    # ------------------------------------------------------------------

    def replay(
        self,
        alpha_id:   int,
        weights_df: pd.DataFrame,   # (T×N) 每日目标权重
        prices_df:  pd.DataFrame,   # (T×N) 收盘价
        volume_df:  pd.DataFrame,   # (T×N) 成交量
    ) -> pd.Series:
        """
        从头逐日回放（按 BacktestEngine 口径算 adv/vol），返回净值 Series。
        用于：① 对账 RealisticBacktester；② 新因子入 PAPER 时的历史补跑。
        """
        dates   = weights_df.index
        tickers = list(weights_df.columns)
        prices  = prices_df.reindex(index=dates, columns=tickers).ffill(limit=5).fillna(0.0)
        volume  = volume_df.reindex(index=dates, columns=tickers).ffill(limit=5).fillna(0.0)
        weights = weights_df.reindex(index=dates, columns=tickers).fillna(0.0)

        adv_df = self._liq.compute_adv(volume, prices)
        vol_df = prices.pct_change().rolling(20, min_periods=2).std().fillna(0.02)

        if len(dates) >= 2:
            cal_years = max((dates[-1] - dates[0]).days / 365.25, 1.0 / 365.25)
            tdays = float(len(dates)) / cal_years
        else:
            tdays = 252.0

        equity_curve = []
        for t in range(len(dates)):
            prev_prices = prices.iloc[t - 1] if t > 0 else prices.iloc[t]
            pnl = self.step(
                alpha_id, dates[t],
                target_w=weights.iloc[t], prices_t=prices.iloc[t], prices_prev=prev_prices,
                adv_usd=adv_df.iloc[t], daily_vol=vol_df.iloc[t], tdays_per_year=tdays,
            )
            equity_curve.append(pnl.equity)
        return pd.Series(equity_curve, index=dates, name="paper_equity")

    def get_positions(self, alpha_id: int) -> pd.Series:
        d = self.store.latest_positions(alpha_id)
        return pd.Series(d, dtype=float)

    def latest_equity(self, alpha_id: int) -> float:
        return self.store.latest_equity(alpha_id)


def _as_date(d):
    from datetime import date as _date, datetime
    if isinstance(d, str):
        return _date.fromisoformat(d)
    if isinstance(d, datetime):
        return d.date()
    if hasattr(d, "date") and not isinstance(d, _date):
        return d.date()
    return d
