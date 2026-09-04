"""
market_calendar.py — 美股交易日历与收盘时点（Phase 11.2）

**双权威设计**（按用户决策"两者都要"）：
    - **数据是权威**：某天到底有没有行情，以 provider 实际返回的 bar 为准
      （moomoo 只在交易日返回 bar）——前向摄取的"是否有新数据"永远由数据说了算。
    - **日历库做交叉校验与提前调度**：`pandas_market_calendars` 的 NYSE 日历用来
      ① 提前知道今天是否开市、几点收盘（**DST 感知**，夏令时 20:00 UTC / 冬令时 21:00 UTC）；
      ② **交叉校验**：日历说是交易日却没拿到 bar → 数据异常告警；反之亦然。
    两者不一致时**不静默**，记 warning 让人看见（数据/日历任一出问题都能被发现）。

注：原调度写死 21:00 UTC，只在冬令时才是"收盘后"；夏令时会偏 1 小时。本模块提供 DST 感知的收盘时点。
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

DateLike = Union[str, date, datetime, pd.Timestamp]
_DEFAULT_EXCHANGE = "NYSE"


def _to_ts(d: DateLike) -> pd.Timestamp:
    ts = pd.Timestamp(d)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _calendar(exchange: str = _DEFAULT_EXCHANGE):
    """返回日历对象；库缺失时返回 None（调用方退回工作日启发式并告警）。"""
    try:
        import pandas_market_calendars as mcal
        return mcal.get_calendar(exchange)
    except Exception as exc:      # pragma: no cover - 依赖缺失路径
        logger.warning("[market_calendar] 日历库不可用（退回工作日启发式）: %s", exc)
        return None


def trading_days(start: DateLike, end: DateLike,
                 exchange: str = _DEFAULT_EXCHANGE) -> pd.DatetimeIndex:
    """[start, end] 内的交易日（不含节假日/周末）。库缺失时退回工作日。"""
    cal = _calendar(exchange)
    s, e = _to_ts(start).normalize(), _to_ts(end).normalize()
    if cal is None:
        return pd.bdate_range(s, e)
    sched = cal.schedule(start_date=s, end_date=e)
    return pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in sched.index])


def is_trading_day(d: DateLike, exchange: str = _DEFAULT_EXCHANGE) -> bool:
    ts = _to_ts(d).normalize()
    return len(trading_days(ts, ts, exchange)) > 0


def last_trading_day(asof: Optional[DateLike] = None,
                     exchange: str = _DEFAULT_EXCHANGE) -> pd.Timestamp:
    """<= asof 的最近一个交易日（asof 默认今天 UTC）。"""
    ts = _to_ts(asof) if asof is not None else pd.Timestamp.utcnow().tz_localize(None)
    ts = ts.normalize()
    days = trading_days(ts - pd.Timedelta(days=14), ts, exchange)
    if len(days) == 0:
        return ts
    return days[-1]


def next_trading_day(asof: Optional[DateLike] = None,
                     exchange: str = _DEFAULT_EXCHANGE) -> pd.Timestamp:
    """> asof 的下一个交易日。"""
    ts = _to_ts(asof) if asof is not None else pd.Timestamp.utcnow().tz_localize(None)
    ts = ts.normalize()
    days = trading_days(ts + pd.Timedelta(days=1), ts + pd.Timedelta(days=14), exchange)
    return days[0] if len(days) else ts + pd.Timedelta(days=1)


def session_close_utc(d: Optional[DateLike] = None,
                      exchange: str = _DEFAULT_EXCHANGE) -> Optional[datetime]:
    """
    某交易日的**收盘 UTC 时刻**（DST 感知：夏令时 20:00 UTC，冬令时 21:00 UTC；半日市自动更早）。
    非交易日返回 None。
    """
    cal = _calendar(exchange)
    ts = (_to_ts(d) if d is not None else pd.Timestamp.utcnow().tz_localize(None)).normalize()
    if cal is None:
        return None
    sched = cal.schedule(start_date=ts, end_date=ts)
    if len(sched) == 0:
        return None
    close = pd.Timestamp(sched.iloc[0]["market_close"])
    return close.to_pydatetime().astimezone(timezone.utc)


def minutes_after_close_utc(minutes: int = 30, d: Optional[DateLike] = None,
                            exchange: str = _DEFAULT_EXCHANGE) -> Optional[datetime]:
    """收盘后 N 分钟的 UTC 时刻（用于安排"收盘后摄取"）。非交易日 None。"""
    c = session_close_utc(d, exchange)
    return (c + timedelta(minutes=minutes)) if c is not None else None


def cross_check(day: DateLike, has_bar: bool,
                exchange: str = _DEFAULT_EXCHANGE) -> Tuple[bool, str]:
    """
    **数据 ↔ 日历 交叉校验**（不静默）：
        日历=交易日 且 无 bar  → 数据异常（漏数据/源故障）
        日历=非交易日 且 有 bar → 日历或数据口径异常
    返回 (一致?, 说明)。数据始终是权威，本函数只负责**让不一致被看见**。
    """
    ts = _to_ts(day).normalize()
    expect = is_trading_day(ts, exchange)
    if expect and not has_bar:
        msg = f"{ts.date()} 日历为交易日但未取到 bar —— 疑似数据源缺数据"
        logger.warning("[market_calendar] %s", msg)
        return False, msg
    if (not expect) and has_bar:
        msg = f"{ts.date()} 日历为非交易日却有 bar —— 日历/数据口径不一致"
        logger.warning("[market_calendar] %s", msg)
        return False, msg
    return True, "一致"


def calendar_available() -> bool:
    """日历库是否可用（供状态端点/前端显示）。"""
    return _calendar() is not None
