"""
test_phase_tr.py — Phase TR（TradingContext）验收

- Corwin-Schultz 价差：宽 H/L 股 > 窄 H/L 股，量级合理
- long-only（allow_short=False）→ 全不可做空
- margin + allow_short → 只在流动大盘子集可做空
- 成本 = 半价差 + 券商费（moomoo 佣金免费 → ≈半价差）
- 调仓带随成本，$10k 有小资金洞察
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.trading_context.spread import corwin_schultz_spread_bps
from app.core.trading_context.context import TradingContext, MOOMOO_US


def _dataset(T=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-01", periods=T)
    cols = ["BIG", "MID", "SMALL"]
    # 价格
    close = pd.DataFrame({
        "BIG":   200 * np.cumprod(1 + rng.normal(0, 0.008, T)),
        "MID":    50 * np.cumprod(1 + rng.normal(0, 0.015, T)),
        "SMALL":   3 * np.cumprod(1 + rng.normal(0, 0.03,  T)),   # penny，价<5
    }, index=idx)
    # 日内振幅：BIG 窄、SMALL 宽
    amp = pd.DataFrame({
        "BIG":   np.abs(rng.normal(0, 0.0015, T)),
        "MID":   np.abs(rng.normal(0, 0.006,  T)),
        "SMALL": np.abs(rng.normal(0, 0.02,   T)),
    }, index=idx)
    high = close * (1 + amp)
    low  = close * (1 - amp)
    volume = pd.DataFrame({
        "BIG":   rng.uniform(5e6, 1e7, T),      # 高流动
        "MID":   rng.uniform(2e5, 8e5, T),
        "SMALL": rng.uniform(1e4, 5e4, T),      # 低流动
    }, index=idx)
    return {"open": close, "high": high, "low": low, "close": close, "vwap": close,
            "volume": volume, "returns": close.pct_change().fillna(0.0)}


def test_spread_estimate_orders_by_liquidity():
    ds = _dataset()
    s = corwin_schultz_spread_bps(ds["high"], ds["low"])
    assert s["SMALL"] > s["MID"] > s["BIG"]     # 越不流动价差越宽
    assert s["BIG"] < 5.0                        # 大盘 <5bps，合理


def test_long_only_makes_nothing_shortable():
    ctx = TradingContext(aum=10_000, account_type="margin", allow_short=False, broker=MOOMOO_US)
    res = ctx.analyze(_dataset())
    assert not res.shortable.any()               # long-only → 全不可做空
    assert any("long-only" in n for n in res.notes)


def test_margin_allow_short_only_liquid_large():
    ctx = TradingContext(aum=1_000_000, account_type="margin", allow_short=True,
                         min_adv_usd=1e6, broker=MOOMOO_US)
    res = ctx.analyze(_dataset())
    # 流动大盘 BIG 可做空；penny SMALL 不可（价<5 且不流动）
    assert res.shortable["BIG"] and not res.shortable["SMALL"]


def test_cost_is_half_spread_for_commission_free_broker():
    ctx = TradingContext(aum=10_000, broker=MOOMOO_US)
    res = ctx.analyze(_dataset())
    s = res.spread_bps
    # moomoo 佣金=0 → 单边成本 ≈ 半价差 + 规费(0.15)
    for c in ["BIG", "MID"]:
        assert abs(res.est_cost_bps_oneway[c] - (s[c] / 2 + 0.15)) < 1e-6


def test_tradable_filters_penny_and_illiquid():
    res = TradingContext(aum=10_000, min_price=5.0, min_adv_usd=1e6).analyze(_dataset())
    assert res.tradable["BIG"]                    # 大盘可交易
    assert not res.tradable["SMALL"]              # penny(价<5)+不流动 → 不可交易


def test_small_aum_insight_and_band():
    res = TradingContext(aum=10_000).analyze(_dataset())
    assert 0.001 <= res.rebalance_band <= 0.05
    assert any("极小" in n or "AUM" in n for n in res.notes)
    d = res.to_dict()
    assert d["aum"] == 10_000 and d["allow_short"] is False


# --------------------------------------------------------------------------
# TR.3：grounded 成本（真实/推导，取代硬编码机构默认）
# --------------------------------------------------------------------------

def test_grounded_cost_params_commission_free_and_data_spread():
    from app.core.trading_context.context import grounded_cost_params, MOOMOO_US
    cp = grounded_cost_params(_dataset(), broker=MOOMOO_US, aum=10_000)
    assert cp.fixed_bps == 0.0            # moomoo 佣金免费（原硬编码 5）
    assert cp.min_ticket_fee == 0.0       # 原硬编码 $1
    assert cp.spread_bps > 0.0            # 来自 Corwin-Schultz 数据估计
    # 与硬编码 spread=2 相比，数据估计应不同（反映真实 universe）
    assert abs(cp.spread_bps - 2.0) > 1e-9
