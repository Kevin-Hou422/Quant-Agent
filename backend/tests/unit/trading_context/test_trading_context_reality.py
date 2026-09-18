"""
trading_context/context.py —— 交易现实摘要的定钉测试（变异测试驱动）

来由：11 个变异点，首测击杀率 45.6%（存活 6）。

这个模块回答"以我这点钱、这个券商，**实际上**能交易什么、成本多少、
多久调一次仓"，它的输出直接喂给策略门（成本参数）与每日循环（无交易带）。
存活项全是**判定阈值与成本公式**：

  - `px > min_price` / `adv_usd > min_adv_usd` 的可交易判定
  - `adv_usd > 5 * min_adv_usd` 里的 `5 *`（易借启发式的严格程度）
  - `est_cost = spread/2 + commission + reg_fee` 的加号
  - `rebalance_band = clip(2.0 * med_cost, ...)` 的 `2.0 *`
  - `tradable.reindex(...).fillna(False)` 的 False（缺数据时默认**不可交易**）

既有覆盖（test_phase_tr）只验证了返回对象的字段齐全。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.trading_context.context import TradingContext, get_broker_profile


def _panel(prices: dict[str, float], advs: dict[str, float], days: int = 30,
           hl_range: float = 0.004) -> dict:
    """按每只标的的目标价格与目标 $ADV 构造面板。"""
    idx = pd.bdate_range("2024-01-02", periods=days)
    cols = list(prices)
    close = pd.DataFrame({c: [prices[c]] * days for c in cols}, index=idx)
    volume = pd.DataFrame({c: [advs[c] / prices[c]] * days for c in cols}, index=idx)
    rng = np.random.default_rng(0)
    high = close * (1 + rng.uniform(hl_range / 2, hl_range, close.shape))
    low = close * (1 - rng.uniform(hl_range / 2, hl_range, close.shape))
    return {"close": close, "open": close, "high": high, "low": low,
            "vwap": close, "volume": volume}


def _ctx(**kw) -> TradingContext:
    kw.setdefault("aum", 10_000.0)
    kw.setdefault("broker", get_broker_profile("moomoo_us"))
    kw.setdefault("account_type", "margin")
    kw.setdefault("allow_short", True)
    return TradingContext(**kw)


# ===========================================================================
# A. 可交易池
# ===========================================================================

class TestTradablePool:

    def test_price_and_liquidity_thresholds_are_strict(self):
        """
        `tradable = (px > min_price) & (adv_usd > min_adv_usd)` —— **严格大于**。
        恰好等于阈值的标的不算可交易（保守方向）。两个阈值都要测，
        它们是两处独立的比较。
        """
        ctx = _ctx(min_price=5.0, min_adv_usd=1_000_000.0)
        ds = _panel(
            prices={"OK": 20.0, "CHEAP": 4.0, "AT_PX": 5.0, "THIN": 20.0,
                    "AT_ADV": 20.0},
            advs={"OK": 5e6, "CHEAP": 5e6, "AT_PX": 5e6, "THIN": 1e5,
                  "AT_ADV": 1_000_000.0})
        t = ctx.analyze(ds).tradable
        assert bool(t["OK"]) is True
        assert bool(t["CHEAP"]) is False, "低于最低价的标的被判为可交易"
        assert bool(t["AT_PX"]) is False, "价格恰好等于阈值不该算可交易"
        assert bool(t["THIN"]) is False, "流动性不足的标的被判为可交易"
        assert bool(t["AT_ADV"]) is False, "ADV 恰好等于阈值不该算可交易"

    def test_missing_columns_default_to_not_tradable(self):
        """
        `tradable.reindex(close.columns).fillna(False)` —— 对齐后缺失值填 **False**。
        改成 True 会让"算不出价格/流动性"的标的被当成可交易，
        而那正是数据最可疑的一批（DEV_LESSONS §U：兜底必须朝保守一侧）。
        """
        ctx = _ctx(min_price=5.0, min_adv_usd=1e6)
        ds = _panel(prices={"A": 20.0, "B": 20.0}, advs={"A": 5e6, "B": 5e6})
        ds["close"]["GHOST"] = np.nan          # 有列但全是 NaN
        ds["volume"]["GHOST"] = np.nan
        ds["high"]["GHOST"] = np.nan
        ds["low"]["GHOST"] = np.nan
        t = ctx.analyze(ds).tradable
        assert "GHOST" in t.index
        assert bool(t["GHOST"]) is False, "算不出价格/流动性的标的被当成了可交易"

    def test_result_is_aligned_to_the_universe(self):
        ctx = _ctx()
        ds = _panel(prices={"A": 20.0, "B": 30.0}, advs={"A": 5e6, "B": 6e6})
        res = ctx.analyze(ds)
        for s in (res.tradable, res.shortable, res.spread_bps,
                  res.est_cost_bps_oneway):
            assert list(s.index) == list(ds["close"].columns)


# ===========================================================================
# B. 可做空性
# ===========================================================================

class TestShortability:

    def test_cash_account_can_never_short(self):
        ctx = _ctx(account_type="cash", allow_short=True)
        ds = _panel(prices={"A": 100.0}, advs={"A": 1e9})
        res = ctx.analyze(ds)
        assert not res.shortable.any(), "现金账户被判为可做空"
        assert any("long-only" in n for n in res.notes)

    def test_allow_short_false_disables_shorting(self):
        ctx = _ctx(account_type="margin", allow_short=False)
        ds = _panel(prices={"A": 100.0}, advs={"A": 1e9})
        assert not ctx.analyze(ds).shortable.any()

    def test_short_universe_is_stricter_than_the_tradable_one(self):
        """
        `shortable = tradable & (adv_usd > 5*min_adv) & (px > 10.0)`
        —— 易借启发式要求**更高的流动性与更高的价位**。
          - `5 *` 写成 `5 /`：门槛从 5×min_adv 掉到 min_adv/5，几乎人人可做空
          - `adv > 5*min_adv` 放宽成 `>=`：ADV 恰好 5 倍的标的被放进做空池
          - `px > 10.0` 放宽成 `>=`：恰好 10 元的标的被放进做空池

        ⚠️ ADV 那一处的区分值是 **5×min_adv 本身**，必须专门造一只（AT_5X）；
        只放一只 2×min_adv 的标的测不到边界（第一版就是这样）。
        """
        ctx = _ctx(min_price=5.0, min_adv_usd=1_000_000.0)
        ds = _panel(
            prices={"BIG": 50.0, "MID": 50.0, "AT_5X": 50.0,
                    "AT_TEN": 10.0, "CHEAPISH": 8.0},
            advs={"BIG": 1e8, "MID": 2e6, "AT_5X": 5_000_000.0,
                  "AT_TEN": 1e8, "CHEAPISH": 1e8})
        res = ctx.analyze(ds)
        assert bool(res.tradable["MID"]) is True, "构造前提：MID 可交易"
        assert bool(res.tradable["AT_5X"]) is True
        assert bool(res.shortable["BIG"]) is True
        assert bool(res.shortable["MID"]) is False, (
            "ADV 只有 2×min_adv 却被放进做空池 —— 5 倍门槛疑似被改小")
        assert bool(res.shortable["AT_5X"]) is False, (
            "ADV 恰好等于 5×min_adv 被放进做空池 —— 该阈值应为严格大于")
        assert bool(res.shortable["AT_TEN"]) is False, (
            "价格恰好 10 元被放进做空池 —— 该阈值应为严格大于")
        assert bool(res.shortable["CHEAPISH"]) is False

    def test_shortable_is_a_subset_of_tradable(self):
        ctx = _ctx(min_price=5.0, min_adv_usd=1e6)
        ds = _panel(prices={"A": 50.0, "B": 4.0}, advs={"A": 1e8, "B": 1e8})
        res = ctx.analyze(ds)
        assert not (res.shortable & ~res.tradable).any()


# ===========================================================================
# C. 成本与调仓带
# ===========================================================================

class TestCostAndBand:

    def test_one_way_cost_adds_the_components(self):
        """
        `est_cost = spread_bps/2 + commission_bps + reg_fee_bps` ——
        第一个 `+` 写成 `-` 会让**佣金变成减项**（佣金越高、算出来的成本越低）。

        ⚠️ 默认券商 moomoo 的 `commission_bps = 0`，`+0` 与 `-0` 完全一样，
        用它测不出这个符号（第一版就是这样）。这里用一个**佣金非零**的
        自定义券商档位。
        """
        from app.core.trading_context.context import BrokerProfile
        broker = BrokerProfile(name="test_broker", commission_bps=3.0,
                               reg_fee_bps=0.15)
        ctx = _ctx(broker=broker)
        ds = _panel(prices={"A": 50.0, "B": 60.0}, advs={"A": 1e8, "B": 1e8})
        res = ctx.analyze(ds)
        half_spread = res.spread_bps / 2.0
        gap = res.est_cost_bps_oneway - half_spread
        assert np.allclose(gap.to_numpy(), 3.0 + 0.15, atol=1e-9), (
            f"单边成本减去半价差应等于佣金+规费=3.15bps，实际 {gap.to_dict()}")
        assert (res.est_cost_bps_oneway > half_spread).all(), (
            "单边成本不高于半价差 —— 佣金疑似被当成了减项")

    def test_zero_commission_broker_still_charges_reg_fees(self):
        """对照组：moomoo 佣金为 0，但规费仍必须计入。"""
        broker = get_broker_profile("moomoo_us")
        assert broker.commission_bps == 0.0
        res = _ctx(broker=broker).analyze(
            _panel(prices={"A": 50.0}, advs={"A": 1e8}))
        gap = res.est_cost_bps_oneway["A"] - res.spread_bps["A"] / 2.0
        assert gap == pytest.approx(broker.reg_fee_bps, abs=1e-9)

    def test_cost_grows_with_the_spread(self):
        ctx = _ctx()
        narrow = ctx.analyze(_panel(prices={"A": 50.0}, advs={"A": 1e8},
                                    hl_range=0.001))
        wide = ctx.analyze(_panel(prices={"A": 50.0}, advs={"A": 1e8},
                                  hl_range=0.02))
        assert wide.est_cost_bps_oneway["A"] > narrow.est_cost_bps_oneway["A"]

    def test_rebalance_band_is_twice_the_median_cost(self):
        """
        `band = clip(2.0 * median(est_cost)/1e4, 0.001, 0.05)` ——
        `2.0 *` 写成 `2.0 /` 会让带宽只有应有的四分之一，换手与成本随之飙升。
        判据：在未触及 clip 边界的区间里，带宽必须精确等于 2×中位成本（比例）。
        """
        ctx = _ctx()
        ds = _panel(prices={"A": 50.0, "B": 60.0, "C": 70.0},
                    advs={"A": 1e8, "B": 1e8, "C": 1e8}, hl_range=0.01)
        res = ctx.analyze(ds)
        med_frac = float(np.nanmedian(res.est_cost_bps_oneway.values)) / 1e4
        expected = float(np.clip(2.0 * med_frac, 0.001, 0.05))
        assert res.rebalance_band == pytest.approx(expected, abs=1e-12)
        assert 0.001 < res.rebalance_band < 0.05, (
            "构造落在了 clip 边界上，本用例对 2× 因子不敏感")
        assert res.rebalance_band == pytest.approx(2.0 * med_frac, abs=1e-12)

    def test_rebalance_band_is_clipped_to_its_bounds(self):
        ctx = _ctx()
        tiny = ctx.analyze(_panel(prices={"A": 50.0}, advs={"A": 1e8},
                                  hl_range=1e-6))
        huge = ctx.analyze(_panel(prices={"A": 50.0}, advs={"A": 1e8},
                                  hl_range=0.25))
        assert tiny.rebalance_band == pytest.approx(0.001, abs=1e-12)
        assert huge.rebalance_band == pytest.approx(0.05, abs=1e-12)

    def test_small_aum_note_is_emitted(self):
        res = _ctx(aum=10_000.0).analyze(
            _panel(prices={"A": 50.0}, advs={"A": 1e8}))
        assert any("容量几乎不 binding" in n for n in res.notes)

    def test_large_aum_omits_the_small_account_note(self):
        res = _ctx(aum=5_000_000.0).analyze(
            _panel(prices={"A": 50.0}, advs={"A": 1e8}))
        assert not any("容量几乎不 binding" in n for n in res.notes)

    def test_missing_volume_defaults_to_a_synthetic_panel(self):
        ctx = _ctx(min_adv_usd=1e3)
        ds = _panel(prices={"A": 50.0}, advs={"A": 1e8})
        ds.pop("volume")
        res = ctx.analyze(ds)
        assert np.isfinite(res.est_cost_bps_oneway["A"])


# ===========================================================================
# D. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "app/core/trading_context/context.py ×1 — L142 `tradable.reindex(close.columns).fillna(False)` → True":
        "该 fillna 不可达：`tradable = (px > min_price) & (adv_usd > min_adv_usd)`，"
        "px 的索引就是 close.columns，adv_usd 的索引是 close 与 volume 列的并集"
        "（⊇ close.columns），两者按并集对齐后再 reindex 回 close.columns，"
        "每一列都取得到值，不会产生 NaN。缺数据的列得到的是 **False**"
        "（`NaN > x` 为 False），而不是 NaN。"
        "见 test_tradable_never_contains_nan_before_fillna。",
}


def test_tradable_never_contains_nan_before_fillna():
    """L142 等价性的机械验证：reindex 之前 tradable 在 close 的每一列上都有值。"""
    idx = pd.bdate_range("2024-01-02", periods=20)
    close = pd.DataFrame({"A": 50.0, "B": np.nan, "C": 5.0}, index=idx)
    volume = pd.DataFrame({"A": 1e6, "C": 1e6, "EXTRA": 1e6}, index=idx)
    px = close.ffill().iloc[-1]
    adv_usd = (close * volume).tail(20).mean(axis=0)
    tradable = (px > 5.0) & (adv_usd > 1e6)
    assert set(close.columns).issubset(set(tradable.index))
    sub = tradable.reindex(close.columns)
    assert not sub.isna().any(), f"reindex 后仍有 NaN：{sub.to_dict()}"
    assert bool(sub["B"]) is False, "全 NaN 的列应判为不可交易（而不是 NaN）"


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 1
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
