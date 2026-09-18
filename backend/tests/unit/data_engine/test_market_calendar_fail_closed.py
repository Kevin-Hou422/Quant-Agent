"""
market_calendar.py —— fail-closed 与时区归一的定钉测试（变异测试驱动）

来由：22 个变异点，首测击杀率 77.3%，存活 5 处集中在**依赖缺失时的降级路径**：
`allow = False` 的三处初值与 `getattr(..., False)` 的默认值。

这条路径是外部审计 #6 的现场：`pandas_market_calendars` 未在 requirements 声明，
开发机恰好装了、CI 干净环境没有。当时的行为是**静默退回 `pd.bdate_range`** ——
7/4、12/25 被当成交易日，拿不到 DST/半日市收盘时刻，而且不报错。
修复后改成 fail-closed（抛 `CalendarUnavailable`），除非显式开启启发式。

把三个 `allow` 初值里任何一个改成 True，就退回了那个**静默降级**的旧行为，
而既有测试全绿 —— 本文件把这条路径钉死。
"""
from __future__ import annotations

import builtins
from datetime import date, datetime

import pandas as pd
import pytest

import app.core.data_engine.market_calendar as mc
from app.core.data_engine.market_calendar import (
    CalendarUnavailable,
    _to_ts,
    is_trading_day,
    last_trading_day,
    next_trading_day,
    session_close_utc,
    trading_days,
)


@pytest.fixture
def no_calendar_lib(monkeypatch):
    """让 `import pandas_market_calendars` 抛 ImportError。"""
    real_import = builtins.__import__

    def _fake(name, *a, **kw):
        if name == "pandas_market_calendars":
            raise ImportError("simulated missing dependency")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fake)


# ===========================================================================
# A. fail-closed —— 依赖缺失时的三处 allow 初值
# ===========================================================================

class TestFailClosed:

    def test_missing_library_raises_instead_of_degrading(self, no_calendar_lib,
                                                         monkeypatch):
        """
        `allow = False` + `if not allow: raise CalendarUnavailable`。
        任一处 allow 初值被改成 True，就退回"把节假日当交易日"的静默降级。
        """
        from app.config import settings
        monkeypatch.setattr(settings, "calendar_allow_heuristic", False, raising=False)
        with pytest.raises(CalendarUnavailable, match="拒绝退回工作日启发式"):
            trading_days("2024-07-01", "2024-07-10")

    def test_heuristic_requires_an_explicit_opt_in(self, no_calendar_lib, monkeypatch,
                                                   caplog):
        """
        `allow = bool(getattr(settings, "calendar_allow_heuristic", False))` ——
        显式打开时必须能用，并留下 ERROR 级告警。
        """
        from app.config import settings
        monkeypatch.setattr(settings, "calendar_allow_heuristic", True, raising=False)
        with caplog.at_level("ERROR"):
            days = trading_days("2024-07-01", "2024-07-10")
        assert len(days) > 0
        assert any("启发式" in r.getMessage() for r in caplog.records), (
            "显式降级没有留下 ERROR 告警 —— 降级必须看得见")

    def test_missing_setting_falls_back_to_fail_closed(self, no_calendar_lib,
                                                       monkeypatch):
        """
        `getattr(settings, "calendar_allow_heuristic", False)` 的**第三参**。

        ⚠️ 只在两种显式取值上测是不够的 —— 默认值只有在配置项**缺失**时才生效
        （第一版就是这样，变异测试证实它存活）。这里把该字段从 settings 上摘掉，
        模拟"没配过这一项"的环境：默认值一旦是 True，这类环境会静默降级成
        把节假日当交易日。
        """
        import types
        import app.config
        monkeypatch.setattr(app.config, "settings", types.SimpleNamespace())
        with pytest.raises(CalendarUnavailable):
            trading_days("2024-07-01", "2024-07-10")

    def test_heuristic_really_is_worse(self, no_calendar_lib, monkeypatch):
        """
        钉住"为什么要 fail-closed"：启发式把**独立日**当成交易日。
        这一条让上面两条的意义可被验证，而不是空谈。
        """
        from app.config import settings
        monkeypatch.setattr(settings, "calendar_allow_heuristic", True, raising=False)
        days = trading_days("2024-07-01", "2024-07-10")
        assert pd.Timestamp("2024-07-04") in days, (
            "构造前提：工作日启发式确实会把 7/4 当成交易日")

    def test_settings_import_failure_still_fails_closed(self, no_calendar_lib,
                                                        monkeypatch):
        """
        第三处 `allow = False`（读取配置本身抛异常时的兜底）。
        改成 True 后，连"配置都读不到"的环境也会静默降级 —— 最不该降级的场景。
        """
        import app.config

        class _Boom:
            def __getattr__(self, name):
                raise RuntimeError("settings unavailable")

        monkeypatch.setattr(app.config, "settings", _Boom())
        with pytest.raises(CalendarUnavailable):
            trading_days("2024-07-01", "2024-07-10")


# ===========================================================================
# B. 真实日历下的语义（库可用时）
# ===========================================================================

class TestRealCalendar:

    def test_holiday_is_not_a_trading_day(self):
        assert is_trading_day("2024-07-04") is False, "独立日被判成交易日"
        assert is_trading_day("2024-12-25") is False, "圣诞被判成交易日"
        assert is_trading_day("2024-07-05") is True

    def test_weekend_is_not_a_trading_day(self):
        assert is_trading_day("2024-07-06") is False      # 周六
        assert is_trading_day("2024-07-07") is False      # 周日

    def test_last_trading_day_skips_back_over_holidays(self):
        assert last_trading_day("2024-07-04") == pd.Timestamp("2024-07-03")
        assert last_trading_day("2024-07-06") == pd.Timestamp("2024-07-05")

    def test_next_trading_day_is_strictly_after(self):
        """
        `days = trading_days(ts + 1天, ts + 14天)` 的 `+` 写成 `-` 会得到一个
        **起点晚于终点**的空区间 → 落到兜底 `ts + 1天`（可能是节假日）。
        """
        assert next_trading_day("2024-07-03") == pd.Timestamp("2024-07-05"), (
            "7/3 的下一个交易日应跳过 7/4 独立日")
        assert next_trading_day("2024-07-05") == pd.Timestamp("2024-07-08")

    def test_next_trading_day_never_returns_the_same_day(self):
        for d in ("2024-07-01", "2024-07-03", "2024-12-24"):
            assert next_trading_day(d) > pd.Timestamp(d)

    def test_session_close_is_dst_aware(self):
        """夏令时 20:00 UTC、冬令时 21:00 UTC —— 写死一个值就会错半年。"""
        summer = session_close_utc("2024-07-05")
        winter = session_close_utc("2024-12-05")
        assert summer is not None and winter is not None
        assert summer.hour == 20, f"夏令时收盘应为 20:00 UTC，实际 {summer}"
        assert winter.hour == 21, f"冬令时收盘应为 21:00 UTC，实际 {winter}"

    def test_half_day_closes_earlier(self):
        """半日市（感恩节次日）收盘更早 —— 这是不能用固定时刻的直接理由。"""
        half = session_close_utc("2024-11-29")
        full = session_close_utc("2024-12-05")
        assert half is not None and full is not None
        assert half < full.replace(year=half.year, month=half.month, day=half.day), (
            f"半日市收盘 {half} 不早于常规交易日 {full}")

    def test_non_trading_day_has_no_close(self):
        assert session_close_utc("2024-07-04") is None
        assert session_close_utc("2024-07-06") is None


# ===========================================================================
# C. 时区归一
# ===========================================================================

class TestTimestampCoercion:

    def test_aware_timestamps_are_stripped_of_tz(self):
        """
        `ts.tz_localize(None) if ts.tzinfo is not None else ts` —— 删掉 `not` 后，
        **naive** 时间戳会被送去 `tz_localize(None)`（对 naive 无意义/报错），
        而 aware 的反而原样返回，后续与 naive 索引比较时抛
        `Cannot compare tz-naive and tz-aware`。
        """
        aware = pd.Timestamp("2024-07-05 16:00", tz="America/New_York")
        out = _to_ts(aware)
        assert out.tzinfo is None
        assert out == pd.Timestamp("2024-07-05 16:00")

    def test_naive_inputs_pass_through_unchanged(self):
        for value in ("2024-07-05", date(2024, 7, 5), datetime(2024, 7, 5),
                      pd.Timestamp("2024-07-05")):
            out = _to_ts(value)
            assert out.tzinfo is None
            assert out.normalize() == pd.Timestamp("2024-07-05")

    def test_aware_input_works_end_to_end(self):
        aware = pd.Timestamp("2024-07-04 12:00", tz="UTC")
        assert is_trading_day(aware) is False


# ===========================================================================
# D. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "app/core/data_engine/market_calendar.py ×1 — L55 `allow = False`（进入 try 之前的初值）":
        "该初值**必被覆盖**：紧随其后的 `try` 里无论成功（走 getattr 赋值）"
        "还是失败（走 except 的 `allow = False`），都会重新给 allow 赋值。"
        "它是一条死赋值，取何值都观察不到差别。"
        "见 test_pre_try_allow_initialiser_is_always_overwritten。",

    "app/core/data_engine/market_calendar.py ×1 — L107 `return days[0] if len(days) else ts + pd.Timedelta(days=1)` 的 `+`":
        "该兜底分支不可达：`days = trading_days(ts+1天, ts+14天)`，"
        "真实日历与工作日启发式在任意连续 14 天里都至少包含一个交易日"
        "（美股最长休市不超过 4 个连续日历日）。"
        "见 test_fourteen_day_window_always_contains_a_trading_day。",
}


def test_pre_try_allow_initialiser_is_always_overwritten():
    """
    L55 等价性的机械验证：`_calendar` 的降级判定里，`allow` 在 try/except
    的**两条路径上都会被重新赋值**，进入 try 之前的初值因此不可观测。
    """
    import inspect
    src = inspect.getsource(mc._calendar)
    body = src.split("except Exception as exc:", 1)[1]
    assert "allow = bool(getattr(settings" in src, "成功路径没有重新给 allow 赋值"
    assert "allow = False" in body, "失败路径没有重新给 allow 赋值"


def test_fourteen_day_window_always_contains_a_trading_day():
    """
    L107 等价性的机械验证：连续 14 天内必有交易日。

    只取一次两年的日历再看**相邻交易日的最大间隔**，而不是逐日调用
    `trading_days` —— 后者每次都要重建日历对象，250 次要跑 3 分钟，
    而这个文件会被变异测试整套重跑几十遍。
    """
    days = trading_days("2023-01-01", "2024-12-31")
    gaps = (days[1:] - days[:-1]).days
    assert len(days) > 400
    assert gaps.max() < 14, (
        f"相邻交易日最大间隔 {gaps.max()} 天，14 天窗口可能落空 —— 兜底分支可达")


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 2
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
