"""
intraday_watchdog.py — 盘中巡检（2026-10-22 用户决定第 17 条）

用户原话："实现，无人值守前验收。按交易所日历和 DST 调度，频率先用可配置的 60 秒，沿用已批准风险阈值。
共用账户锁和熔断状态，覆盖锁竞争、断线、陈旧快照及恢复，避免并发重复平仓。"

此前熔断只在收盘后的日频任务里检查 —— 盘中跌穿日亏阈值要等到收盘后才处理。巡检每一拍：

    时段      交易所日历判断"现在是不是常规交易时段"（session_open_utc / session_close_utc，夏令时由日历处理）；
              不在 → 不连券商。日历不可用 → 严重告警，不猜
    账户锁    与调仓 / 全平 / 对账同一把锁，但**不等待**：拿不到 = 另一个进程正在动这个账户 → 本拍什么都不做
    断线      连不上券商 → 计数；连续 `unreachable_alert_after` 次 → 严重告警；恢复后发恢复通知
    陈旧快照  总资产 + 各持仓价组成的指纹 `stale_after_s` 秒没变 → 不据此做任何决定，严重告警（巡检失明）
    熔断      共用执行库里的熔断开关。已开（不论是谁开的）→ 只续做全平（在途全平单存在时不再下，不重复平仓）；
              没开且切片相对上一个可信收盘快照跌幅 ≥ 已批准的日亏阈值（与下单风控门同一个 max_daily_loss）
              → 先把熔断状态落库，再撤单、全平
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

EXCHANGE_TZ = "America/New_York"


def exchange_date(now: datetime):
    return pd.Timestamp(now).tz_convert(EXCHANGE_TZ).date() if pd.Timestamp(now).tzinfo else \
        pd.Timestamp(now).tz_localize("UTC").tz_convert(EXCHANGE_TZ).date()


def in_regular_session(now: datetime) -> bool:
    """`now` 是否落在交易所的常规交易时段里（日历不可用 → CalendarUnavailable）。"""
    from app.core.data_engine.market_calendar import (
        is_trading_day,
        session_close_utc,
        session_open_utc,
    )
    d = pd.Timestamp(exchange_date(now))
    if not is_trading_day(d):
        return False
    t = pd.Timestamp(now)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return pd.Timestamp(session_open_utc(d)) <= t < pd.Timestamp(session_close_utc(d))


class IntradayWatchdog:
    def __init__(self, make_order_manager: Optional[Callable] = None, now: Optional[Callable[[], datetime]] = None,
                 alert: Optional[Callable] = None, unreachable_alert_after: Optional[int] = None,
                 stale_after_s: Optional[float] = None) -> None:
        from app.config import settings
        self._owns_gateway = make_order_manager is None
        self.make_om = make_order_manager or _order_manager_from_settings
        self.now = now or (lambda: datetime.now(timezone.utc))
        if alert is None:
            from app.core.alerting.dispatcher import emit_safely
            alert = emit_safely
        self.alert = alert
        self.unreachable_alert_after = int(unreachable_alert_after if unreachable_alert_after is not None
                                           else getattr(settings, "intraday_watchdog_unreachable_alert_after", 3))
        self.stale_after = timedelta(seconds=float(stale_after_s if stale_after_s is not None
                                                   else getattr(settings, "intraday_watchdog_stale_after_s", 600)))
        self._fails = 0
        self._unreachable_alerted = False
        self._fingerprint = None
        self._fp_since: Optional[datetime] = None
        self._stale_alerted = False

    # ------------------------------------------------------------------
    def threshold(self) -> float:
        """已批准的日亏阈值：与下单风控门同一个 `max_daily_loss`。"""
        om = self.make_om()
        try:
            return float(om.limits.max_daily_loss)
        finally:
            self._close(om)

    def _event(self, severity: str, title: str, detail: str, key: str) -> None:
        from app.core.alerting.events import AlertEvent
        self.alert(AlertEvent(kind="intraday_watchdog", severity=severity, title=title, detail=detail,
                              dedup_key=f"intraday_watchdog:{key}"))

    def _close(self, om) -> None:
        if self._owns_gateway:
            close = getattr(getattr(om, "gw", None), "close", None)
            if close is not None:
                try:
                    close()
                except Exception as exc:
                    logger.warning("[watchdog] 关闭券商连接失败: %s", exc)

    def _unreachable(self, exc, now) -> dict:
        """调用方已把 `_fails` 加一并记了日志。"""
        if self._fails >= self.unreachable_alert_after and not self._unreachable_alerted:
            self._unreachable_alerted = True
            self._event("critical", "盘中巡检连不上券商",
                        f"连续 {self._fails} 次连不上（{exc}）—— 盘中跌穿日亏阈值也处理不了", f"unreachable:{now.date()}")
        return {"status": "broker_unreachable", "consecutive_failures": self._fails, "error": str(exc)}

    def _reachable_again(self, now) -> None:
        if self._unreachable_alerted:
            self._event("info", "盘中巡检恢复：券商重新连上", f"此前连续 {self._fails} 次连不上", f"recovered:{now.isoformat()}")
        self._fails, self._unreachable_alerted = 0, False

    def _stale(self, fp, now) -> bool:
        if fp != self._fingerprint:
            self._fingerprint, self._fp_since, self._stale_alerted = fp, now, False
            return False
        if now - self._fp_since < self.stale_after:
            return False
        if not self._stale_alerted:
            self._stale_alerted = True
            self._event("critical", "盘中巡检：券商快照陈旧",
                        f"总资产与持仓价自 {self._fp_since.isoformat()} 起没有变化 —— 巡检失明，不据此做决定",
                        f"stale:{self._fp_since.isoformat()}")
        return True

    # ------------------------------------------------------------------
    def tick(self) -> dict:
        from app.core.data_engine.market_calendar import CalendarUnavailable
        from app.core.execution.broker_gateway import BrokerError
        now = self.now()
        try:
            if not in_regular_session(now):
                return {"status": "outside_session", "at": now.isoformat()}
        except CalendarUnavailable as exc:
            logger.error("[watchdog] 交易日历不可用，本拍不判断: %s", exc)
            self._event("critical", "盘中巡检：交易日历不可用", str(exc), f"calendar:{now.date()}")
            return {"status": "calendar_unavailable", "error": str(exc)}
        try:
            om = self.make_om()
        except BrokerError as exc:
            self._fails += 1
            logger.error("[watchdog] 连不上券商（连续第 %d 次）: %s", self._fails, exc)
            return self._unreachable(exc, now)
        try:
            with om.try_account_lock() as got:
                if not got:
                    logger.info("[watchdog] 账户正被另一个进程操作 → 本拍跳过")
                    return {"status": "lock_busy"}
                try:
                    acct, positions = om.gw.account(), om.gw.positions()
                except BrokerError as exc:
                    self._fails += 1
                    logger.error("[watchdog] 读券商账户失败（连续第 %d 次）: %s", self._fails, exc)
                    return self._unreachable(exc, now)
                self._reachable_again(now)
                fp = (round(float(acct.total_assets), 6),
                      tuple(sorted((tk, round(float(p.nominal_price), 6)) for tk, p in positions.items())))
                stale = self._stale(fp, now)
                if om.store.kill_switch().get("engaged"):
                    if not positions:
                        return {"status": "breaker_active", "flat": True}
                    rep = om.maintain("intraday_watchdog: 熔断已开，续做全平")
                    return {"status": "breaker_active", "maintain": rep.to_dict()}
                if stale:
                    return {"status": "stale_snapshot", "since": self._fp_since.isoformat()}
                return self._check_loss(om, acct, now)
        finally:
            self._close(om)

    def _check_loss(self, om, acct, now) -> dict:
        prev = om.store.trusted_before(exchange_date(now))
        s_now = om.slice_equity(acct.total_assets)
        s_prev = om.slice_equity(prev.total_assets) if prev is not None else None
        if s_now is None or s_prev is None or s_prev <= 0:
            return {"status": "no_baseline"}
        ret = s_now / s_prev - 1.0
        thr = float(om.limits.max_daily_loss)
        if ret > -thr:
            return {"status": "ok", "slice_return": ret}
        reason = f"盘中切片较上一可信收盘跌 {ret:.2%}，达到已批准的日亏阈值 −{thr:.2%}"
        # 先落熔断状态（之后任何调仓周期都只会全平），再撤单、全平 —— 与手动熔断同一条路径
        om.store.set_kill_switch(True, "intraday_watchdog", reason)
        om.store.add_event("kill_switch_engaged", "intraday_watchdog", reason)
        self._event("critical", "盘中巡检拉起熔断（全平并停止调仓）", reason, f"engaged:{exchange_date(now)}")
        out = om.engage_kill_switch("intraday_watchdog", reason, persist_state=False)
        return {"status": "breaker_engaged", "slice_return": ret, "flatten": out}


def _order_manager_from_settings():
    from app.core.execution.broker_gateway import make_gateway_from_settings
    from app.core.execution.order_manager import OrderManager
    return OrderManager.from_settings(make_gateway_from_settings())


_WATCHDOG: Optional[IntradayWatchdog] = None


def intraday_watchdog_job() -> dict:
    """调度器每 `intraday_watchdog_interval_s` 秒调一次（同一个实例跨拍保留断线计数与快照指纹）。"""
    global _WATCHDOG
    from app.config import settings
    if str(getattr(settings, "execution_mode", "off")) != "moomoo_paper":
        return {"status": "execution_off"}
    if _WATCHDOG is None:
        _WATCHDOG = IntradayWatchdog()
    out = _WATCHDOG.tick()
    if out.get("status") not in ("ok", "outside_session"):
        logger.warning("[watchdog] %s", out)
    return out
