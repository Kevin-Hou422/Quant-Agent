"""
时钟 c2 的成交规则与交易日边界（D5 关键路径变异复核：app/core/clock.py）。

复核时 33 个变异点存活 21 个：成交规则的每一条边界（开盘恰好等于限价、最低 / 最高恰好触及限价、
价格为 0、side = 0）、缺省限价带的换算、"后 n 个交易日"恰好够数、require_ohlc 的正常路径都没人钉。
这里逐条钉死；剩下 4 个是等价变异，证明在 PROVEN_EQUIVALENT（每条都有一条机械验证的用例）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core import clock

REF, BAND = 100.0, 0.5          # 限价恰好可表示：买 150、卖 50
LIM_B, LIM_S = 150.0, 50.0


def _fill(side, o, h, lo, ref=REF, band=BAND):
    return clock.limit_fill_prices(np.array([side]), np.array([ref]), np.array([o]), np.array([h]),
                                   np.array([lo]), band)[0]


class TestLimitFillBoundaries:

    def test_a_buy_opening_exactly_at_the_limit_fills_at_the_open(self):
        assert _fill(+1, o=LIM_B, h=160.0, lo=140.0) == LIM_B

    def test_a_buy_whose_low_exactly_touches_the_limit_fills_at_the_limit(self):
        assert _fill(+1, o=155.0, h=160.0, lo=LIM_B) == LIM_B

    def test_a_buy_that_never_reaches_the_limit_does_not_fill(self):
        assert np.isnan(_fill(+1, o=155.0, h=160.0, lo=150.01))

    def test_a_sell_opening_exactly_at_the_limit_fills_at_the_open(self):
        assert _fill(-1, o=LIM_S, h=60.0, lo=40.0) == LIM_S

    def test_a_sell_whose_high_exactly_touches_the_limit_fills_at_the_limit(self):
        assert _fill(-1, o=45.0, h=LIM_S, lo=40.0) == LIM_S

    def test_the_sell_limit_is_below_the_reference(self):
        """卖的限价 = ref·(1 − band)：开盘 60 在卖限价 50 之上 → 按开盘价卖；高点 70 也够不到 ref·(1+band)。"""
        assert _fill(-1, o=60.0, h=70.0, lo=55.0) == 60.0
        assert np.isnan(_fill(-1, o=45.0, h=49.0, lo=40.0))

    def test_zero_side_is_no_trade(self):
        assert np.isnan(_fill(0, o=100.0, h=101.0, lo=99.0))

    @pytest.mark.parametrize("ref,o", [(0.0, 100.0), (100.0, 0.0)])
    def test_a_zero_reference_or_open_is_no_fill(self, ref, o):
        assert np.isnan(_fill(+1, o=o, h=101.0, lo=0.0, ref=ref))
        assert np.isnan(_fill(-1, o=o, h=101.0, lo=0.0, ref=ref))


class TestDefaultBand:

    def test_the_band_is_the_configured_basis_points(self, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "exec_limit_band_bps", 20.0)
        assert clock.default_band() == pytest.approx(0.002, abs=1e-15)

    def test_an_unreadable_setting_falls_back_to_fifty_basis_points(self, monkeypatch):
        import app.config

        class _Broken:
            def __getattr__(self, name):
                raise RuntimeError("配置坏了")
        monkeypatch.setattr(app.config, "settings", _Broken())
        assert clock.default_band() == pytest.approx(0.005, abs=1e-15)


class TestNextSessions:

    def test_asking_for_exactly_as_many_sessions_as_the_window_holds_works(self):
        from app.core.data_engine.market_calendar import trading_days
        d = pd.Timestamp("2025-03-04")
        window = [x for x in trading_days(d + pd.Timedelta(days=1), d + pd.Timedelta(days=20)) if x > d]
        assert clock._next_sessions(d, len(window)) == window
        with pytest.raises(ValueError):
            clock._next_sessions(d, len(window) + 1)

    def test_next_sessions_never_include_the_day_itself_or_earlier(self):
        """L62 两处等价的机械验证：把起点改成 d − 1 天 / 过滤改成 ≥ d，结果不变（节假日前后、周五、周末、年末）。"""
        from app.core.data_engine.market_calendar import trading_days
        days = ["2024-12-31", "2025-01-03", "2025-01-17", "2025-04-17", "2025-04-19", "2025-05-23",
                "2025-07-03", "2025-08-29", "2025-11-26", "2025-11-28", "2025-12-24", "2025-12-31"]
        for d in pd.DatetimeIndex(days):
            got = clock._next_sessions(d, 2)
            earlier = [x for x in trading_days(d - pd.Timedelta(days=1), d + pd.Timedelta(days=20)) if x > d][:2]
            ge = [x for x in trading_days(d + pd.Timedelta(days=1), d + pd.Timedelta(days=20)) if x >= d][:2]
            assert got == earlier == ge, d


class TestRequireOhlc:

    def test_valid_ohlc_is_aligned_to_the_requested_grid(self):
        idx, cols = pd.bdate_range("2025-01-02", periods=3), ["A", "B"]
        frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], index=idx, columns=cols)
        out = clock.require_ohlc({"open": frame, "high": frame + 1, "low": frame - 1}, idx, ["B", "A"])
        assert out["open"].tolist() == [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]]
        assert out["high"][0, 0] == 3.0 and out["low"][2, 1] == 4.0

    def test_missing_ohlc_is_refused(self):
        with pytest.raises(ValueError, match="ohlc"):
            clock.require_ohlc(None, [], [])
        with pytest.raises(ValueError, match="low"):
            clock.require_ohlc({"open": pd.DataFrame(), "high": pd.DataFrame()}, [], [])


def test_the_tie_at_the_limit_gives_the_same_price_on_either_branch():
    """L130 / L132 等价的机械验证：开盘恰好等于限价时，"按开盘价"与"按限价"是同一个数。"""
    grid = np.linspace(40.0, 160.0, 25)
    for side, lim in ((+1, LIM_B), (-1, LIM_S)):
        for lo in grid:
            for h in grid:
                if h < lo:
                    continue
                got = _fill(side, o=lim, h=max(h, lim), lo=min(lo, lim))
                assert got == lim


PROVEN_EQUIVALENT = {
    "app/core/clock.py ×1 — L62 `if x > d` -> `>=`":
        "trading_days 的起点已经是 d + 1 天，结果里不可能出现 ≤ d 的日期，过滤条件取 > 还是 ≥ 都过滤不掉任何东西。"
        "见 test_next_sessions_never_include_the_day_itself_or_earlier（节假日前后、周末、年末逐个比对）。",
    "app/core/clock.py ×1 — L62 `d + pd.Timedelta(days=1)` -> `d - pd.Timedelta(days=1)`":
        "起点提前到 d − 1 天只会多出 d − 1、d 两天，而它们被紧跟着的 `x > d` 过滤掉，结果逐项相同。"
        "见 test_next_sessions_never_include_the_day_itself_or_earlier（节假日前后、周末、年末逐个比对）。",
    "app/core/clock.py ×1 — L130 `buy & (o > lim_b)` -> `>=`":
        "两者只在开盘恰好等于买限价时不同：这时上一行已按开盘价成交，本行改写成限价 —— 而限价就等于开盘价，"
        "数组里的值不变。见 test_the_tie_at_the_limit_gives_the_same_price_on_either_branch。",
    "app/core/clock.py ×1 — L132 `sell & (o < lim_s)` -> `<=`":
        "与买侧对称：只在开盘恰好等于卖限价时不同，此时按开盘价与按限价是同一个数。"
        "见 test_the_tie_at_the_limit_gives_the_same_price_on_either_branch。",
}


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 4
