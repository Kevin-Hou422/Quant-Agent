"""
intraday_watchdog.py — 盘中巡检（2026-10-22 用户决定第 17 条）

用户原话："实现，无人值守前验收。按交易所日历和 DST 调度，频率先用可配置的 60 秒，沿用已批准风险阈值。
共用账户锁和熔断状态，覆盖锁竞争、断线、陈旧快照及恢复，避免并发重复平仓。"

此前熔断只在收盘后的日频任务里检查 —— 盘中跌穿日亏阈值要等到收盘后才处理。巡检每一拍：

    时段      交易所日历判断"现在是不是常规交易时段"（session_open_utc / session_close_utc，夏令时由日历处理）；
              不在 → 不连券商。日历不可用 → 严重告警，不猜
    账户锁    与调仓 / 全平 / 对账同一把锁，但**不等待**：拿不到 = 另一个进程正在动这个账户 → 本拍什么都不做
    断线      连不上券商 → 计数；连续 `unreachable_alert_after` 次 → 严重告警；恢复后发恢复通知
    新鲜度    （2026-10-25 审计 R6："用账户数值不变判断数据陈旧，会误报健康的纯现金账户"）只认**权威时间**：持仓标的的
              独立行情时间戳。（复审 21cd90b："不把'至少一只新鲜'等同于'全部持仓新鲜'"）**逐只**判定：
                  fresh 新鲜 / missing 没有行情或没有时间戳 / stale 超过 `stale_after_s` 没更新（停牌常见）/
                  future 比此刻晚 `FUTURE_TOLERANCE` 以上（时钟或时区错了）/ bad_time 时间解析不了或时区不可判定
              再合成整体结论（纯现金账户没有需要估值的持仓 → no_positions，不存在陈旧；价格不变但时间在走 → fresh）：

    | 整体        | 条件                                   | 盘中亏损判断                       | 告警                         | 无人值守许可          |
    |-------------|----------------------------------------|------------------------------------|------------------------------|-----------------------|
    | fresh       | 每只都新鲜                             | 判断                               | —                            | 不影响                |
    | partial     | 有新鲜的，也有不新鲜的                 | 照常判断：跌穿阈值仍拉熔断（减风险方向）；没跌穿报 partial_ok，不报 ok | 每只异常持仓一条：missing / future / bad_time = critical，stale = warning（每天每只每种一次） | 有 missing / future / bad_time → 不新增风险；只有 stale（停牌）→ 不阻断 |
    | unproven    | 没有一只新鲜，且全部是 missing         | 照常判断；数值长时间不变 → warning | warning                      | 不新增风险            |
    | stale       | 没有一只新鲜，且有时间可判（过期 / 未来 / 不可判定） | **不判断**（巡检失明）   | critical                     | 不新增风险            |
    | unreachable | 连不上券商                             | 不判断                             | 连续 N 次后 critical；恢复后 info | 不新增风险，恢复后的下一拍自动解除 |

              每一拍的结论落到执行库（exec_state `intraday_watchdog`：时刻、整体、逐只状态），无人值守许可每轮核对时读最近一拍
              （`blocking_reasons`）—— 恢复后的下一拍覆盖它；巡检停了而最后一拍是坏的 → 一直不新增风险
    熔断      共用执行库里的熔断开关。已开（不论是谁开的）→ 只续做全平（在途全平单存在时不再下，不重复平仓）；
              没开且切片相对上一个可信收盘快照跌幅 ≥ 已批准的日亏阈值（与下单风控门同一个 max_daily_loss）
              → 先把熔断状态落库，再撤单、全平
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

EXCHANGE_TZ = "America/New_York"
#: 行情时间比此刻晚这么多以上 = 未来时间（几十秒的时钟偏差不算）
FUTURE_TOLERANCE = timedelta(seconds=120)
STATE_KEY = "intraday_watchdog"
#: 这几种持仓状态说明行情链路本身有问题（不是停牌）：无人值守不新增风险
DEFECT_STATES = ("missing", "future", "bad_time")
BLOCKING_OVERALL = ("stale", "unproven", "unreachable")
QUOTE_LABEL = {"missing": "缺失", "future": "时间在未来", "bad_time": "时间不可判定"}


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
                 stale_after_s: Optional[float] = None, health_store=None) -> None:
        from app.config import settings
        #: 每拍结论落在哪个执行库（缺省 = 与 OrderManager.from_settings 同一个）；连不上券商时也要能落
        self._health_store = health_store
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
        self._unproven_alerted = False
        self._stale_names_alerted: set = set()

    def _record(self, store, now, overall: str, names: Optional[dict] = None, detail: str = "") -> None:
        """这一拍的覆盖结论落库（无人值守许可读它）。落不了 → 记 error，不打断巡检。"""
        try:
            if store is None:
                if self._health_store is None:
                    from app.db.execution_store import ExecutionStore
                    self._health_store = ExecutionStore()
                store = self._health_store
            store.set_state(STATE_KEY, json.dumps({"at": now.isoformat(), "state": overall, "names": names or {},
                                                   "detail": detail}, ensure_ascii=False))
        except Exception as exc:
            logger.error("[watchdog] 巡检结论没能落库（无人值守许可看不到这一拍）: %s", exc)

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
        self._record(None, now, "unreachable", detail=f"连续 {self._fails} 次连不上券商：{exc}")
        return {"status": "broker_unreachable", "consecutive_failures": self._fails, "error": str(exc)}

    def _reachable_again(self, now) -> None:
        if self._unreachable_alerted:
            self._event("info", "盘中巡检恢复：券商重新连上", f"此前连续 {self._fails} 次连不上", f"recovered:{now.isoformat()}")
        self._fails, self._unreachable_alerted = 0, False

    def _name_state(self, q, n) -> dict:
        from app.core.execution.pretrade_gate import QUOTE_TZ
        raw = None if q is None else getattr(q, "asof", None)
        if q is None or raw is None or str(raw).strip() == "":
            return {"state": "missing", "detail": "没有行情" if q is None else "行情没有时间戳"}
        try:
            t = pd.Timestamp(raw)
            if pd.isna(t):
                raise ValueError("空时间")
            t = t.tz_localize(QUOTE_TZ) if t.tzinfo is None else t      # 夏令时切换的那一小时：不可判定 → 抛错
        except Exception as exc:
            logger.warning("[watchdog] 行情时间 %r 判定不了: %s", raw, exc)
            return {"state": "bad_time", "detail": f"行情时间 {raw!r} 解析不了 / 时区不可判定：{exc}"}
        age = n - t
        if age < -FUTURE_TOLERANCE:
            return {"state": "future", "as_of": t.isoformat(), "detail": f"比此刻晚 {-age}"}
        if age > self.stale_after:
            return {"state": "stale", "as_of": t.isoformat(), "detail": f"{age} 没更新"}
        return {"state": "fresh", "as_of": t.isoformat()}

    def _freshness(self, om, positions, now) -> dict:
        """逐只持仓的行情覆盖 → 整体结论（见模块说明的表）。"""
        from app.core.execution.broker_gateway import BrokerError
        if not positions:
            return {"state": "no_positions", "names": {}}
        n = pd.Timestamp(now)
        n = n.tz_localize("UTC") if n.tzinfo is None else n
        try:
            quotes = om.gw.quotes(sorted(positions)) or {}
        except BrokerError as exc:
            logger.warning("[watchdog] 取不到持仓行情时间: %s", exc)
            quotes = {}
        names = {tk: self._name_state(quotes.get(tk), n) for tk in sorted(positions)}
        fresh = [tk for tk, v in names.items() if v["state"] == "fresh"]
        times = [pd.Timestamp(v["as_of"]) for v in names.values() if v["state"] == "fresh"]
        out = {"names": names, "as_of": max(times).isoformat() if times else None,
               "old_names": sorted(tk for tk, v in names.items() if v["state"] == "stale")}
        if len(fresh) == len(names):
            return {**out, "state": "fresh"}
        if fresh:
            return {**out, "state": "partial"}
        if all(v["state"] == "missing" for v in names.values()):
            return {**out, "state": "unproven", "detail": "持仓行情都没有时间戳"}
        return {**out, "state": "stale"}

    def _on_freshness(self, f, fp, now) -> bool:
        """处理新鲜度结论；返回"这一拍不能据此做决定"。"""
        if fp != self._fingerprint:
            self._fingerprint, self._fp_since, self._unproven_alerted = fp, now, False
        if f["state"] != "stale":
            self._stale_alerted = False
        if f["state"] == "stale":
            if not self._stale_alerted:
                self._stale_alerted = True
                bad = {tk: v["state"] for tk, v in f["names"].items()}
                self._event("critical", "盘中巡检：行情时间陈旧（没有一只持仓的行情是新鲜的）",
                            f"逐只：{bad}（过期超过 {self.stale_after} / 未来时间 / 时区不可判定）—— 巡检失明，不据此做决定",
                            f"stale:{exchange_date(now)}:{sorted(bad.items())}")
            return True
        if f["state"] == "partial":
            day = exchange_date(now)
            for tk, v in f["names"].items():
                if v["state"] == "fresh" or (tk, v["state"], day) in self._stale_names_alerted:
                    continue
                self._stale_names_alerted.add((tk, v["state"], day))
                if v["state"] == "stale":
                    self._event("warning", f"盘中巡检：{tk} 的行情时间偏旧",
                                f"{v['detail']}；其余持仓有新鲜行情（停牌 / 没有成交？）—— 这只票的估值可能不是现价",
                                f"old_quote:{tk}:{day}")
                else:
                    self._event("critical", f"盘中巡检：{tk} 的行情{QUOTE_LABEL[v['state']]}",
                                f"{v['detail']} —— 这只持仓的估值不可信；盘中亏损照常判断（跌穿仍拉熔断），"
                                "无人值守不新增风险，直到它恢复", f"quote_{v['state']}:{tk}:{day}")
        if f["state"] == "unproven" and now - self._fp_since >= self.stale_after and not self._unproven_alerted:
            self._unproven_alerted = True
            self._event("warning", "盘中巡检：无法证明快照新鲜",
                        f"{f.get('detail', '')}；总资产与持仓价自 {self._fp_since.isoformat()} 起没变 —— "
                        "数值不变本身不等于陈旧，但也证明不了新鲜", f"unproven:{self._fp_since.isoformat()}")
        return False

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
                fresh = self._freshness(om, positions, now)
                stale = self._on_freshness(fresh, fp, now)
                self._record(om.store, now, fresh["state"],
                             {tk: v["state"] for tk, v in fresh["names"].items()}, fresh.get("detail", ""))
                if om.store.kill_switch().get("engaged"):
                    if not positions:
                        return {"status": "breaker_active", "flat": True}
                    rep = om.maintain("intraday_watchdog: 熔断已开，续做全平")
                    return {"status": "breaker_active", "maintain": rep.to_dict()}
                if stale:
                    return {"status": "stale_snapshot", "as_of": fresh.get("as_of"), "freshness": fresh}
                res = self._check_loss(om, acct, now)
                if res["status"] == "ok" and fresh["state"] not in ("fresh", "no_positions"):
                    res["status"] = f"{fresh['state']}_ok"            # 没跌穿，但不是每只持仓都新鲜：不报 ok
                return {**res, "freshness": fresh["state"],
                        "names": {tk: v["state"] for tk, v in fresh["names"].items()}}
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


def blocking_reasons(store) -> list:
    """无人值守许可每轮核对用：最近一拍巡检的结论里不允许新增风险的部分（没有记录 → 空）。"""
    raw = store.get_state(STATE_KEY)
    if not raw:
        return []
    try:
        rec = json.loads(raw)
    except (TypeError, ValueError) as exc:
        logger.error("[watchdog] 巡检结论记录损坏: %s", exc)
        return ["盘中巡检的结论记录损坏 —— 证明不了巡检在看着持仓"]
    names = rec.get("names") or {}
    if rec.get("state") in BLOCKING_OVERALL:
        return [f"盘中巡检最近一拍（{rec.get('at')}）是 {rec['state']}：{rec.get('detail') or names}"]
    bad = {tk: st for tk, st in names.items() if st in DEFECT_STATES}
    if bad:
        return [f"盘中巡检最近一拍（{rec.get('at')}）有持仓的行情缺失 / 时间在未来 / 时区不可判定：{bad}"]
    return []
