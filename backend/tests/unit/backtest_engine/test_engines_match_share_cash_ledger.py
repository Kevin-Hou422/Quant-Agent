"""
独立的**股数 + 现金**参考账本验算两套引擎（复审 2026-10-19 N08-b；2026-10-22 D14 起按时钟 c2）。

"回测和 paper 一起改过且彼此相等，仍不能证明经济记账正确" —— 两套引擎互相对账只证明实现一致。
参考账本不用权重：持有股数、现金。时钟 c2（2026-10-22 用户决定 D14 (a)）：

  · T 收盘决策（目标 w[T]），**委托股数在 T 收盘冻结**（2026-10-25 审计 R1）：
      trunc(w × T 收盘净值 / T 收盘价) − 现有股数 —— 次日开盘价只决定成交价，不重新解释目标
  · T+1 按明确规则执行：DAY 限价单，参考价 = T 收盘，限价 = 参考价 × (1 ± b) 按 0.01 取整（买向下、卖向上）；
      买：开盘 ≤ 限价 → 按开盘价成交；否则当日最低 ≤ 限价 → 按限价成交；否则不成交（卖对称）
  · 计划在开盘执行 ≠ 必按开盘价成交 —— 上面第二、三种情形都覆盖
  · 成交量上限按**截至前一天**的 ADV（开盘时当天的量还不知道），整股向零取整，不超过委托量
  · 隔夜（昨收 → 今开）算旧仓，盘中（今开 → 今收）算新仓；费用按成交额从现金扣；空头按昨收市值收借券费
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.core.backtest_engine.transaction_cost import CostParams, LiquidityConstraint

FIXED_BPS, BORROW_BPS, MPP, BAND_BPS = 10.0, 100.0, 0.10, 50.0


def _params():
    return CostParams(fixed_bps=FIXED_BPS, min_ticket_fee=0.0, spread_bps=0.0, impact_coef=0.0,
                      short_borrow_annual_bps=BORROW_BPS, max_participation_pct=MPP)


def _market(seed=5, T=30):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-02-03", periods=T)
    cols = ["A", "B", "C", "D"]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.03, (T, 4)), 0), idx, cols)
    # 开盘相对昨收跳空（含大跳空 → 越过限价带），日内高低包住开收
    gap = rng.normal(0, 0.012, (T, 4))
    gap[7, 0], gap[12, 1], gap[18, 2] = 0.03, -0.03, 0.025           # 跳空超过 50bp 限价带
    open_ = close.shift(1).fillna(close) * (1 + gap)
    amp = np.abs(rng.normal(0, 0.01, (T, 4)))
    high = np.maximum(open_, close) * (1 + amp)
    low = np.minimum(open_, close) * (1 - amp)
    low.iloc[7, 0] = open_.iloc[7, 0] * 0.999                        # A 第 7 天：越过限价、盘中也没回来 → 不成交
    high.iloc[12, 1] = open_.iloc[12, 1] * 1.001
    # D 的成交量很小 → 每天只能成交一小部分（部分成交）；其余充足
    vol = pd.DataFrame(1e9, idx, cols)
    vol["D"] = 300.0
    w = pd.DataFrame({"A": 0.4, "B": 0.3, "C": -0.25, "D": 0.2}, index=idx)
    w.iloc[10:, :] = [[0.1, 0.5, -0.4, -0.15]] * (T - 10)        # 中途换仓：加空头、D 由多翻空
    w.iloc[6:, 0] = 0.6                                           # A 第 7 天要加仓（会撞上跳空）
    return {"open": open_, "high": high, "low": low, "close": close}, vol, w


def _limit(ref, side, band):
    """限价 = 参考价 × (1 ± b)，按 0.01（<$1 为 0.0001）取整：买向下、卖向上（与引擎无关地写一遍）。"""
    raw = ref * (1 + band) if side > 0 else ref * (1 - band)
    tick = 0.01 if raw >= 1.0 else 0.0001
    n = math.floor(raw / tick + 1e-9) if side > 0 else math.ceil(raw / tick - 1e-9)
    return round(n * tick, 4)


def _fill_price(side, ref, o, h, lo, band):
    """限价规则（与引擎无关地写一遍）：返回成交价或 None。"""
    lim = _limit(ref, side, band)
    if side > 0:
        if o <= lim:
            return o
        return lim if lo <= lim else None
    if o >= lim:
        return o
    return lim if h >= lim else None


def _oracle(px, vol, w, capital=1e6):
    """股数 + 现金账本。返回每日收盘后的净值 / 初始资金。"""
    close, open_, high, low = px["close"], px["open"], px["high"], px["low"]
    dates = close.index
    adv = LiquidityConstraint(_params()).compute_adv(vol, close)     # 市场数据（与引擎同口径）
    cal_years = max((dates[-1] - dates[0]).days / 365.25, 1 / 365.25)
    daily_borrow = BORROW_BPS * 1e-4 / (len(dates) / cal_years)
    band = BAND_BPS * 1e-4
    shares = pd.Series(0.0, index=close.columns)
    cash = capital
    out = []
    for t, d in enumerate(dates):
        if t > 0:
            prev_c = close.iloc[t - 1]
            borrow = float((np.maximum(-shares, 0.0) * prev_c).sum()) * daily_borrow
            o = open_.loc[d]
            eq_decision = cash + float((shares * prev_c).sum())      # 昨天收盘的净值
            target = np.trunc(w.iloc[t - 1] * eq_decision / prev_c)  # 昨天收盘冻结的目标股数
            want = target - shares
            cap_sh = (adv.iloc[t - 1] * MPP) / o                     # 截至前一天的 ADV
            for tk in close.columns:
                q = float(np.sign(want[tk]) * math.floor(min(abs(want[tk]), cap_sh[tk])))
                if abs(q) < 1e-12:
                    continue
                p = _fill_price(np.sign(q), prev_c[tk], o[tk], high.loc[d, tk], low.loc[d, tk], band)
                if p is None:
                    continue
                fee = abs(q) * p * FIXED_BPS * 1e-4
                shares[tk] += q
                cash -= q * p + fee
            cash -= borrow
        out.append((cash + float((shares * close.loc[d]).sum())) / capital)
    return pd.Series(out, index=dates)


def test_the_paper_broker_matches_the_share_and_cash_ledger(tmp_path):
    from app.core.execution.paper_broker import PaperBroker
    from app.db.position_store import PositionStore
    px, vol, w = _market()
    br = PaperBroker(store=PositionStore(f"sqlite:///{(tmp_path / 'p.db').as_posix()}"),
                     cost_params=_params(), initial_capital=1e6, limit_band_bps=BAND_BPS)
    got = br.replay(1, w, px["close"], vol, ohlc=px)
    ref = _oracle(px, vol, w)
    assert np.asarray(got) == pytest.approx(np.asarray(ref), rel=1e-9)


def test_the_backtest_engine_matches_the_share_and_cash_ledger():
    from app.core.backtest_engine.backtest_engine import BacktestEngine
    px, vol, w = _market()
    res = BacktestEngine(cost_params=_params(), initial_capital=1e6, limit_band_bps=BAND_BPS).run(
        w, px["close"], vol, w, ohlc=px)
    ref = _oracle(px, vol, w)
    assert np.asarray(res.equity_curve) == pytest.approx(np.asarray(ref), rel=1e-9)


def test_the_scenario_actually_exercises_every_branch():
    """区分力前提：部分成交、空头借券、按开盘价成交、按限价成交、跳空后不成交都确实发生了。"""
    px, vol, w = _market()
    adv = LiquidityConstraint(_params()).compute_adv(vol, px["close"])
    assert (adv["D"].iloc[0] * MPP) / px["open"]["D"].iloc[1] < 0.2 * 1e6 / px["open"]["D"].iloc[1]
    assert (w < 0).any().any()
    band = BAND_BPS * 1e-4
    c, o, h, lo = px["close"], px["open"], px["high"], px["low"]
    kinds = set()
    for t in range(1, len(c)):
        for tk in c.columns:
            for side in (1, -1):
                p = _fill_price(side, c.iloc[t - 1][tk], o.iloc[t][tk], h.iloc[t][tk], lo.iloc[t][tk], band)
                kinds.add("none" if p is None else ("open" if p == o.iloc[t][tk] else "limit"))
    assert kinds == {"open", "limit", "none"}
    # A 第 7 天要加仓，开盘越过限价、盘中没回来 → 参考账本里没成交
    assert _fill_price(1, c.iloc[6]["A"], o.iloc[7]["A"], h.iloc[7]["A"], lo.iloc[7]["A"], band) is None
    assert _oracle(px, vol, w).iloc[-1] != _oracle(px, vol.assign(D=1e9), w).iloc[-1]


def test_a_decision_is_never_traded_on_its_own_day():
    """c2：T 收盘的目标在 T 当天不成交（第一天之后的第一笔成交发生在 T+1）。"""
    from app.core.backtest_engine.backtest_engine import BacktestEngine
    px, vol, w = _market()
    res = BacktestEngine(cost_params=_params(), initial_capital=1e6, limit_band_bps=BAND_BPS).run(
        w, px["close"], vol, w, ohlc=px)
    assert float(res.positions.iloc[0].abs().sum()) == 0.0, "第一天的目标当天就成交了"
    assert float(res.positions.iloc[1].abs().sum()) > 0.0


def test_the_engines_refuse_to_run_without_open_high_low():
    from app.core.backtest_engine.backtest_engine import BacktestEngine
    px, vol, w = _market()
    with pytest.raises(ValueError, match="open"):
        BacktestEngine(cost_params=_params(), initial_capital=1e6).run(w, px["close"], vol, w)
