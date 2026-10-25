"""
交易所日程按年缓存（market_calendar._year_schedule）。

pandas_market_calendars 每次 `schedule()` 都重算节假日表（约 0.5 秒）；下单周期一轮调几十次。
改成按 (交易所, 年) 缓存之后只切片 —— 结果必须与直接调用逐项相同，缺库仍 fail-closed。
"""

from __future__ import annotations

import pandas as pd
import pandas_market_calendars as mcal
import pytest

from app.core.data_engine import market_calendar as mc

NYSE = mcal.get_calendar("NYSE")

RANGES = [("2024-12-20", "2025-01-10"),            # 跨年 + 圣诞 / 元旦
          ("2024-11-25", "2024-11-29"),            # 感恩节 + 次日半日市
          ("2025-03-07", "2025-03-11"),            # 夏令时切换
          ("2025-10-31", "2025-11-04"),            # 冬令时切换
          ("2023-01-01", "2026-12-31"),            # 跨四年
          ("2025-07-04", "2025-07-04"),            # 节假日单日
          ("2025-07-06", "2025-07-05")]            # 起止颠倒


@pytest.mark.parametrize("start,end", RANGES)
def test_cached_slices_equal_the_direct_schedule(start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    direct = NYSE.schedule(start_date=s, end_date=e) if e >= s else NYSE.schedule(start_date=s, end_date=s).iloc[0:0]
    got = mc._schedule("NYSE", s, e)
    if len(direct) == 0:                                  # 空结果的索引精度不同（us / s），没有行可比
        assert len(got) == 0
    else:
        pd.testing.assert_frame_equal(got[["market_open", "market_close"]], direct[["market_open", "market_close"]],
                                      check_freq=False)
    assert list(mc.trading_days(s, e)) == [pd.Timestamp(d).normalize() for d in direct.index]


@pytest.mark.parametrize("day", ["2024-11-29", "2025-03-10", "2025-11-03", "2025-12-24", "2025-07-04", "2025-07-05"])
def test_session_times_equal_the_direct_schedule(day):
    direct = NYSE.schedule(start_date=day, end_date=day)
    if len(direct) == 0:
        assert mc.session_open_utc(day) is None and mc.session_close_utc(day) is None
        return
    assert mc.session_open_utc(day) == pd.Timestamp(direct.iloc[0]["market_open"]).to_pydatetime()
    assert mc.session_close_utc(day) == pd.Timestamp(direct.iloc[0]["market_close"]).to_pydatetime()


def test_a_year_is_computed_once(monkeypatch):
    calls = []
    real = mcal.get_calendar

    def counting(name):
        cal = real(name)
        orig = cal.schedule
        cal.schedule = lambda *a, **k: calls.append(1) or orig(*a, **k)
        return cal
    mc._year_schedule.cache_clear()
    monkeypatch.setattr(mcal, "get_calendar", counting)
    try:
        for d in pd.bdate_range("2025-02-03", "2025-02-28"):
            mc.session_close_utc(d)
            mc.is_trading_day(d)
        assert len(calls) == 1
    finally:
        mc._year_schedule.cache_clear()


def test_a_missing_library_still_fails_closed_after_the_cache_is_warm(monkeypatch):
    import sys
    mc.session_close_utc("2025-03-05")                    # 缓存里已有 2025
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    with pytest.raises(mc.CalendarUnavailable):
        mc.session_close_utc("2025-03-05")


# ---------------------------------------------------------------------------
# 重测 market_calendar 时存活的两个点（都在缓存之外的原有代码里）
# ---------------------------------------------------------------------------

def test_next_trading_day_falls_back_to_tomorrow_when_the_calendar_gives_nothing(monkeypatch):
    monkeypatch.setattr(mc, "trading_days", lambda *a, **k: pd.DatetimeIndex([]))
    assert mc.next_trading_day("2025-03-05") == pd.Timestamp("2025-03-06")


def test_missing_library_fails_closed_even_when_settings_cannot_be_read(monkeypatch):
    """L56（等价，证明在 test_market_calendar_fail_closed.py）配置读不到的那条路径：照样 fail-closed。"""
    import sys
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    monkeypatch.setitem(sys.modules, "app.config", None)            # 配置也读不到
    with pytest.raises(mc.CalendarUnavailable):
        mc.is_trading_day("2025-07-04")

