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
    CostParams, LiquidityConstraint, TransactionCostEngine, simulate_partial_fills,
)
from ...db.position_store import DailyPnL, PositionStore

logger = logging.getLogger(__name__)


class PaperBroker:
    def __init__(
        self,
        store:           Optional[PositionStore] = None,
        cost_params:     Optional[CostParams] = None,
        initial_capital: float = 1_000_000.0,
        max_net:         Optional[float] = None,
        limit_band_bps:  Optional[float] = None,
    ) -> None:
        from app.core.clock import default_band
        self.store           = store or PositionStore()
        #: T+1 限价单的限价带（小数）。缺省 = settings.exec_limit_band_bps（与券商下单同一参数；时钟 c2）
        self.band            = default_band() if limit_band_bps is None else float(limit_band_bps) * 1e-4
        self.params          = cost_params or CostParams()
        self.initial_capital = initial_capital
        # 部分成交会破坏对冲比例 → 成交后净敞口要回查（用户决策 #2）。
        # None = 不约束；显式给值时，超限就整组缩减新增订单，缩不动就暂缓。
        self.max_net         = max_net
        self._liq            = LiquidityConstraint(self.params)
        self._tc             = TransactionCostEngine(self.params)

    # ------------------------------------------------------------------
    # 单日原子记账
    # ------------------------------------------------------------------

    def step(
        self,
        alpha_id:       int,
        date,
        target_w:       Optional[pd.Series],   # (N,) **昨天收盘**冻结的目标（None = 今天不交易）
        prices_t:       pd.Series,      # (N,) 今日收盘价（估值）
        prices_prev:    pd.Series,      # (N,) 昨日收盘价（限价单参考价、隔夜收益的起点）
        adv_usd:        pd.Series,      # (N,) **截至昨天**的 20 日 ADV（USD）—— 开盘时当天的量还不知道
        daily_vol:      pd.Series,      # (N,) 截至昨天的日波动率
        *,
        open_t:         pd.Series,      # (N,) 今日开盘价
        high_t:         pd.Series,      # (N,) 今日最高价
        low_t:          pd.Series,      # (N,) 今日最低价
        tdays_per_year: float = 252.0,
    ) -> DailyPnL:
        """
        执行 alpha 在 date 的单日模拟成交并原子持久化（幂等：同日重跑覆盖）。时钟 c2（app/core/clock.py）：

          1) 旧仓吃隔夜收益（昨收 → 今开）
          2) 开盘撮合**昨天收盘**冻结的目标：DAY 限价单，参考价 = 昨收、限价带 = exec_limit_band_bps；
             开盘在限价内 → 开盘价；否则当日触及限价 → 限价；否则不成交（未成交原因 limit_not_reached）
          3) 费用按成交价的成交额；成交价相对开盘价的差额记为执行成本
          4) 新仓吃盘中收益（今开 → 今收），收盘估值

        开 / 高 / 低价是**必填**的：没有它们就不能按规则成交，也不能悄悄假设"按开盘价成交"。
        昨仓与净值从 PositionStore 读取（崩溃恢复安全）。
        """
        from app.core.backtest_engine.transaction_cost import drift_weights
        from app.core.clock import limit_fill_prices
        # 幂等/续跑：状态取**严格早于 date** 的最近一日（非全局最近），
        # 使重跑第 t 日基于 t-1 状态，全量重放逐位可复现。
        equity, prev_pos = self.store.state_before(alpha_id, date)

        # 【缺陷 N-5】资产集合 = **昨仓与新目标的并集**：不在新目标里的旧持仓也要估值、并按"目标 0"去平仓
        held = [tk for tk, w in prev_pos.items() if abs(float(w)) > 1e-12]
        tgt_names = list(target_w.index) if target_w is not None else []
        tickers = tgt_names + [tk for tk in held if tk not in tgt_names]

        prev_w = np.array([prev_pos.get(tk, 0.0) for tk in tickers], dtype=float)
        p_t    = prices_t.reindex(tickers).to_numpy(dtype=float)
        p_prev = prices_prev.reindex(tickers).to_numpy(dtype=float)
        o_raw  = open_t.reindex(tickers).to_numpy(dtype=float)
        h_raw  = high_t.reindex(tickers).to_numpy(dtype=float)
        l_raw  = low_t.reindex(tickers).to_numpy(dtype=float)
        adv    = np.nan_to_num(adv_usd.reindex(tickers).to_numpy(dtype=float), nan=0.0)
        vol    = np.nan_to_num(daily_vol.reindex(tickers).to_numpy(dtype=float), nan=0.02)

        # 旧持仓今天可能拿不到行情（标的退出了数据集）：不估值变动、也不交易（上限置 0），并留痕
        stale = ~np.isfinite(p_t) | ~np.isfinite(p_prev)
        if np.any(stale & (np.abs(prev_w) > 1e-12)):
            logger.warning(
                "[paper_broker] alpha=%s date=%s 有旧持仓拿不到今日行情，"
                "本日不估值也不交易（按真实状态挂账）：%s",
                alpha_id, date, [tickers[i] for i in np.flatnonzero(stale)
                                 if abs(prev_w[i]) > 1e-12])

        # 1) 隔夜：缺开盘价 → 开盘 = 昨收（没有隔夜收益；下面的限价规则也判不成交）
        o_eff = np.where(np.isfinite(o_raw) & (o_raw > 0), o_raw, p_prev)
        with np.errstate(divide="ignore", invalid="ignore"):
            on_chg = np.where(~stale & (p_prev > 0), o_eff / p_prev - 1.0, 0.0)
            in_chg = np.where(~stale & (o_eff > 0), p_t / o_eff - 1.0, 0.0)
        on_chg = np.nan_to_num(on_chg, nan=0.0)
        in_chg = np.nan_to_num(in_chg, nan=0.0)
        gross_on = float(np.nansum(prev_w * on_chg))

        # 2) 借券成本（昨仓空头，按昨收市值）
        daily_borrow = self.params.short_borrow_annual_bps * 1e-4 / max(tdays_per_year, 1.0)
        borrow_ret = float(np.sum(np.maximum(-prev_w, 0.0)) * daily_borrow)

        # 3) 开盘撮合：从**漂移到开盘**的权重出发（F10），按单日可成交量部分成交、不做任何再分配（A-6）；
        #    三个量分开（N08）：期初权益 v_prev（昨收）、开盘市值 v_open、收盘后净值
        v_prev = equity * self.initial_capital
        v_open = v_prev * (1.0 + gross_on)
        pre_w = drift_weights(prev_w, on_chg)
        tgt = (target_w.reindex(tickers).fillna(0.0).to_numpy(dtype=float) if target_w is not None
               else pre_w.copy())
        if v_open > 0:
            cap_trade_w = adv * self.params.max_participation_pct / v_open
        else:
            cap_trade_w = np.full_like(adv, np.inf)
        cap_trade_w = np.where(stale, 0.0, cap_trade_w)
        px = limit_fill_prices(np.sign(tgt - pre_w), p_prev, o_raw, h_raw, l_raw, self.band)
        limit_blocked = ~np.isfinite(px) & (np.abs(tgt - pre_w) > 1e-12) & ~stale
        cap_trade_w = np.where(np.isfinite(px), cap_trade_w, 0.0)
        fr = simulate_partial_fills(pre_w, tgt, cap_trade_w, max_net=self.max_net)
        filled, delta = fr.filled_w, fr.filled_d
        exec_px = np.where(np.isfinite(px), px, o_eff)
        with np.errstate(divide="ignore", invalid="ignore"):
            px_ratio = np.where(o_eff > 0, exec_px / o_eff, 1.0)

        # 4) 成本：按**成交价**的成交额；成交价相对开盘价的差额是执行成本（买高 / 卖低为正）
        cost_w, _cost_usd_total, _ = self._tc.compute(
            date=date, delta_w=delta * px_ratio, prices=exec_px, adv_usd=adv, daily_vol=vol,
            portfolio_val=max(v_open, 0.0), tickers=tickers,
        )
        exec_cost = float(np.nansum(delta * (px_ratio - 1.0))) * max(v_open, 0.0)

        # 5) 盘中 + 收盘净值
        close_val = v_open * (1.0 + float(np.nansum(filled * in_chg)))
        new_val = close_val - float(_cost_usd_total) - exec_cost - borrow_ret * v_prev
        gross_ret = close_val / v_prev - 1.0 if v_prev > 0 else 0.0
        cost_ret = (float(_cost_usd_total) + exec_cost) / v_prev if v_prev > 0 else 0.0
        net_ret = new_val / v_prev - 1.0 if v_prev > 0 else 0.0
        equity_new = equity * (1.0 + net_ret)

        # 6) 成交记录（成交那一刻的权重，相对开盘市值）+ 收盘持仓（相对收盘净值）
        fills: List[dict] = []
        pv = max(v_open, 0.0)
        for i, tk in enumerate(tickers):
            # 没交易、没持仓、也没有未成交的意图 → 跳过；一笔**完全没成交**的订单（c2 限价没碰到）必须留记录
            if abs(delta[i]) < 1e-12 and abs(filled[i]) < 1e-12 and abs(float(fr.unfilled_d[i])) <= 1e-12:
                continue
            # 未成交的原因分得开（用户决策 #2）：limit_not_reached（限价没碰到）/ participation（成交量打满）/
            # net_exposure_*（整组新增订单被缩减 / 暂缓，否则部分成交会把净敞口推出限额）
            unfilled = float(fr.unfilled_d[i])
            if abs(unfilled) <= 1e-12:
                reject = ""
            elif limit_blocked[i]:
                reject = "limit_not_reached"
            elif abs(delta[i]) >= abs(cap_trade_w[i]) - 1e-12:
                reject = "participation"
            else:
                reject = "net_exposure_defer" if fr.deferred else "net_exposure_scaled"
            fills.append({
                "ticker": tk, "target_weight": float(tgt[i]),
                "filled_weight": float(filled[i]), "fill_price": float(exec_px[i]),
                "cost_usd": float(cost_w[i] * pv), "reject_reason": reject,
                "traded_weight": float(delta[i]), "unfilled_weight": unfilled,
            })
        # 收盘持仓：成交后的股数按今收估值，费用 / 借券从现金扣 → 相对**收盘净值**重新归一（复审 N08-b）
        if new_val > 0 and v_open > 0:
            post = filled * (1.0 + in_chg) * v_open / new_val
        else:
            post = filled
        positions = {tk: float(post[i]) for i, tk in enumerate(tickers) if abs(post[i]) > 1e-12}

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
        weights_df: pd.DataFrame,   # (T×N) 第 T 行 = T 收盘冻结的目标
        prices_df:  pd.DataFrame,   # (T×N) 收盘价
        volume_df:  pd.DataFrame,   # (T×N) 成交量
        ohlc:       Optional[dict] = None,   # {"open","high","low"}（时钟 c2 必需）
    ) -> pd.Series:
        """
        从头逐日回放（按 BacktestEngine 口径算 adv/vol），返回净值 Series。第一天只决策、不成交。
        用于：① 对账 RealisticBacktester；② 新因子入 PAPER 时的历史补跑。
        """
        from app.core.clock import require_ohlc
        dates   = weights_df.index
        tickers = list(weights_df.columns)
        prices  = prices_df.reindex(index=dates, columns=tickers).ffill(limit=5).fillna(0.0)
        volume  = volume_df.reindex(index=dates, columns=tickers).ffill(limit=5).fillna(0.0)
        weights = weights_df.reindex(index=dates, columns=tickers).fillna(0.0)
        require_ohlc(ohlc, dates, tickers)
        bars = {k: ohlc[k].reindex(index=dates, columns=tickers) for k in ("open", "high", "low")}

        adv_df = self._liq.compute_adv(volume, prices)
        vol_df = prices.pct_change().rolling(20, min_periods=2).std().fillna(0.02)

        if len(dates) >= 2:
            cal_years = max((dates[-1] - dates[0]).days / 365.25, 1.0 / 365.25)
            tdays = float(len(dates)) / cal_years
        else:
            tdays = 252.0

        equity_curve = []
        for t in range(len(dates)):
            prev = t - 1 if t > 0 else t
            pnl = self.step(
                alpha_id, dates[t],
                target_w=weights.iloc[t - 1] if t > 0 else None, prices_t=prices.iloc[t],
                prices_prev=prices.iloc[prev], adv_usd=adv_df.iloc[prev], daily_vol=vol_df.iloc[prev],
                open_t=bars["open"].iloc[t], high_t=bars["high"].iloc[t], low_t=bars["low"].iloc[t],
                tdays_per_year=tdays,
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
