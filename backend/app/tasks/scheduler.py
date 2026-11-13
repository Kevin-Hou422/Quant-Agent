"""
scheduler.py — APScheduler 定时任务框架（Task 5.3）

设计说明
--------
- BackgroundScheduler + SQLAlchemyJobStore（SQLite 持久化）：进程重启后
  任务定义与 next_run_time 从数据库恢复，不丢失。
- misfire_grace_time=3600：停机期间错过的任务在重启后 1 小时内补跑一次，
  超过则跳过并记日志（coalesce=True 合并多次错过为一次）。
- 机器睡着（2026-11-14）：调度器跑在笔记本上，每日交易排在 21:30 UTC（北京时间 05:30）。启动时请求系统别因空闲待机
  （app/core/host_power.KeepAwake），并量出实际睡了多久（SuspendWatch → 严重告警）；按时没跑成的每日交易由
  `daily_trading_catchup` 在**下一个交易时段开盘之前**补上（决策日 = 最近一个已收盘的交易日，与摄取 / 执行同一定义），
  开盘之后就不补了（委托是给那个时段的），只告警。
- 时区显式配置（settings.scheduler_timezone，默认 UTC），避免 Windows
  本地时区歧义。
- 默认注册的任务：
    daily_monitor_job — 每个交易日收盘后（默认 UTC 21:00，可配）对全部
    非终态因子做一次衰减检查，告警写日志。Phase 7 的 daily_trading_loop
    将挂载到同一调度器。
- FastAPI 集成：main.py 中 ``settings.enable_scheduler=True`` 时随应用
  startup/shutdown 启停；测试与 CLI 模式默认不启动。
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_scheduler = None                     # 模块级单例
_host_guards: list = []               # 调度器运行期间：防待机 + 量待机


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def daily_monitor_job() -> None:
    """每日衰减巡检：对全部非终态因子跑 check_decay，告警写日志。"""
    from app.db.alpha_store import AlphaStore
    from app.core.monitor.alpha_monitor import AlphaMonitor
    from app.config import settings

    # Task 6.2：显式传 settings.database_url，确保调度线程与 API 命中同一物理库
    store   = AlphaStore(db_url=settings.database_url)
    monitor = AlphaMonitor(store)
    rows    = monitor.get_dashboard()
    alerts  = [r for r in rows if r.has_alert]
    logger.info(
        "[daily_monitor_job] 巡检 %d 个因子 | %d 个衰减告警",
        len(rows), len(alerts),
    )
    for r in alerts:
        logger.warning(
            "[daily_monitor_job] ALERT alpha_id=%d status=%s 连续负IC=%d 滚动均值IC=%.4f",
            r.alpha_id, r.status, r.consecutive_neg, r.rolling_mean_ic,
        )


def daily_trading_job(now=None) -> dict:
    """
    每日管线 + 结论进告警（D15：每一种结论 —— 包括非交易日、日历不可用这些提前返回 —— 都走同一个出口）。
    跑完记下这一次对应的决策日、数据有没有到那一天（补跑据此判断还欠不欠）。
    """
    import pandas as _pd
    from datetime import datetime, timezone
    from contextlib import ExitStack
    from app.db.process_lock import ProcessLockTimeout
    now = now or datetime.now(timezone.utc)
    with ExitStack() as stack:
        try:
            stack.enter_context(_daily_run_lock())
        except ProcessLockTimeout:
            logger.warning("[scheduler] 另一个每日任务正在运行，本次不重复运行")
            return {"skipped": "daily_run_in_progress"}
        out = _run_daily_trading(now)
        _record_run(out, now)
        _emit_daily_outcome(out, str(_pd.Timestamp(now).tz_convert("UTC").tz_localize(None).normalize().date()))
        return out


def _daily_run_lock():
    """普通、重试和补跑共用跨进程锁；与券商账户锁分开，覆盖摄取到运行记录。"""
    from app.core.execution.order_manager import _account_lock_dir
    from app.db.process_lock import lock_path_for, process_lock
    return process_lock(lock_path_for("scheduler", "daily-trading", _account_lock_dir()), timeout_s=0.0)


# ---------------------------------------------------------------------------
# 按时没跑成的每日交易：在下一个交易时段开盘之前补上
# ---------------------------------------------------------------------------

#: 补跑检查的间隔；开盘前留出的余量（委托要在开盘前到券商）；每个决策日最多补跑几次（数据源一直没发布时别空转一夜）
CATCHUP_EVERY_MIN = 10
CATCHUP_OPEN_MARGIN_MIN = 15
CATCHUP_MAX_ATTEMPTS = 6
RUNS_KEY = "daily_trading_runs"


def decision_window(now):
    """(决策日, 补跑截止时刻)：决策日 = 最近一个已收盘的交易日；截止 = 它之后那个交易日开盘前 CATCHUP_OPEN_MARGIN_MIN 分钟。"""
    from datetime import timedelta
    from app.core.data_engine.market_calendar import last_closed_session, next_trading_day, session_open_utc
    s = last_closed_session(now)
    nxt_open = session_open_utc(next_trading_day(s))
    deadline = None if nxt_open is None else nxt_open - timedelta(minutes=CATCHUP_OPEN_MARGIN_MIN)
    return s.date(), deadline


def _runs(store=None) -> dict:
    import json
    from app.db.execution_store import ExecutionStore
    raw = (store or ExecutionStore()).get_state(RUNS_KEY)
    return json.loads(raw) if raw else {}


def _record_run(out: dict, now, store=None) -> None:
    """仅在数据到齐且交易管线确实成功时标记 done；失败可重试，未知提交由执行层对账。"""
    import json
    try:
        from app.config import settings
        from app.core.data_engine.pit_store import PITStore
        from app.db.execution_store import ExecutionStore
        store = store or ExecutionStore()
        s, _ = decision_window(now)
        last = PITStore(settings.pit_store_dir).latest_timestamp(settings.paper_dataset)
        reached = last is not None and last.date() >= s
        from app.tasks.forward import EXIT_OK, classify_run
        # 无新 bar 时也可能通过 PIT 完成交易；必须有成功的执行报告，不能仅凭数据日期。
        done = (reached and not out.get("skipped")
                and (out.get("ingest_accepted") is True or out.get("reason") == "no_new_bar")
                and classify_run(out)[0] == EXIT_OK)
        runs = _runs(store)
        prev = runs.get(str(s), {})
        runs[str(s)] = {"done": bool(done or prev.get("done")), "attempts": int(prev.get("attempts", 0)) + 1,
                        "last_at": now.isoformat(timespec="seconds"),
                        "data_through": str(last.date()) if last is not None else None,
                        "outcome": out.get("skipped") or out.get("reason") or out.get("reject_reason") or "ran"}
        store.set_state(RUNS_KEY, json.dumps(dict(sorted(runs.items())[-30:]), ensure_ascii=False))
    except Exception as exc:
        logger.error("[scheduler] 每日运行记录没写进执行库（补跑判断会再跑一次）: %s", exc)


def daily_trading_catchup_job(now=None) -> Optional[str]:
    """
    每 CATCHUP_EVERY_MIN 分钟一次：最近一个已收盘交易日的每日交易还没跑成（机器睡着 / 关机 / 进程没起 / 数据晚到），
    且两次定时运行（DAILY_TRADING_AT、DAILY_TRADING_RETRY_AT）都已过点、下一个时段还没开盘 → 现在补跑。
    开盘之后就不补了，发严重告警。返回做了什么（给测试与日志看）。
    """
    from contextlib import ExitStack
    from app.db.process_lock import ProcessLockTimeout
    with ExitStack() as stack:
        try:
            stack.enter_context(_daily_run_lock())
        except ProcessLockTimeout:
            logger.info("[scheduler] 每日任务正在运行，补跑检查等下一次")
            return "busy"
        return _daily_trading_catchup_locked(now)


def _daily_trading_catchup_locked(now=None) -> Optional[str]:
    from datetime import datetime, timedelta, timezone
    import pandas as _pd
    from app.config import settings
    from app.core.alerting.dispatcher import emit_safely
    from app.core.alerting.events import AlertEvent
    from app.core.data_engine.market_calendar import CalendarUnavailable
    now = now or datetime.now(timezone.utc)
    try:
        s, deadline = decision_window(now)
    except CalendarUnavailable as exc:
        logger.error("[scheduler] 补跑检查：交易日历不可用（不补跑）: %s", exc)
        return "calendar_unavailable"
    tz = str(getattr(settings, "scheduler_timezone", "UTC") or "UTC")
    retry_at = _pd.Timestamp(datetime(s.year, s.month, s.day, *DAILY_TRADING_RETRY_AT)).tz_localize(tz)
    if now < retry_at.to_pydatetime() + timedelta(minutes=CATCHUP_EVERY_MIN):
        return "regular_runs_pending"
    rec = _runs().get(str(s), {})
    if rec.get("done"):
        return "done"
    if deadline is None or now >= deadline:
        emit_safely(AlertEvent(kind="daily_run_missed", severity="critical",
                               title=f"{s} 的每日交易没有跑成，已过补跑时限",
                               detail=f"下一个交易时段开盘前（{deadline}）没能补上（尝试 {rec.get('attempts', 0)} 次；"
                                      f"最后一次 {rec.get('outcome', '没有运行')}）—— 这一天没有调仓",
                               dedup_key=f"daily_run_missed:{s}"))
        return "missed"
    if int(rec.get("attempts", 0)) >= CATCHUP_MAX_ATTEMPTS:
        return "gave_up"
    emit_safely(AlertEvent(kind="daily_run_catchup", severity="warning", title=f"补跑 {s} 的每日交易",
                           detail=f"按时没跑成（{rec.get('outcome', '没有运行记录：进程没在跑或机器睡着')}）；"
                                  f"在 {deadline} 之前补上",
                           dedup_key=f"daily_run_catchup:{s}:{int(rec.get('attempts', 0))}"))
    daily_trading_job(now)
    return "ran"


def _on_host_suspended(start, end, seconds) -> None:
    """机器睡了一段（SuspendWatch）：严重告警 + 执行库事件。睡着期间调度器、盘中巡检、日亏熔断都没有运行。"""
    from app.core.alerting.dispatcher import emit_safely
    from app.core.alerting.events import AlertEvent
    mins = seconds / 60.0
    logger.error("[scheduler] 机器睡了 %.0f 分钟（%s → %s）：期间调度器与盘中巡检没有运行", mins, start, end)
    emit_safely(AlertEvent(kind="host_suspended", severity="critical", title=f"运行调度器的机器睡了 {mins:.0f} 分钟",
                           detail=f"{start.isoformat(timespec='seconds')} → {end.isoformat(timespec='seconds')}："
                                  "这段时间调度任务、盘中巡检、日亏熔断都没有运行；错过的每日交易会在下一个时段开盘前补跑",
                           dedup_key=f"host_suspended:{start.isoformat(timespec='seconds')}"))
    try:
        from app.db.execution_store import ExecutionStore
        ExecutionStore().add_event("host_suspended", actor="scheduler", reason=f"{mins:.0f} 分钟",
                                   payload={"from": start.isoformat(), "to": end.isoformat(), "seconds": round(seconds)})
    except Exception as exc:
        logger.error("[scheduler] 待机事件没写进执行库: %s", exc)


def _run_daily_trading(now=None) -> dict:
    """
    Task 7.1/7.3 + Phase 11：每日**前向增量**摄取（健康门）→ 交易循环。

    - 非交易日直接跳过（美股日历，节假日/周末）——不空转、不产生噪声日志。
    - 摄取走增量：PIT 空则回填，之后每天只拉新 bar；无新 bar 就不交易。

    返回管线结果（调度器忽略；`python -m app.tasks.forward run-now` 打印给人看）。
    """
    from app.tasks.daily_ingest import run_daily_pipeline
    from app.core.data_engine.market_calendar import is_trading_day, CalendarUnavailable
    from app.config import settings
    import pandas as _pd

    from datetime import datetime, timezone
    now = now or datetime.now(timezone.utc)
    today = _pd.Timestamp(now).tz_convert("UTC").tz_localize(None).normalize()
    try:
        trading = is_trading_day(today)
        if not trading:
            # 非交易日的定时运行：最近一个已收盘交易日还欠着（机器睡过了那天的运行）→ 这次就是补跑
            s, deadline = decision_window(now)
            trading = not _runs().get(str(s), {}).get("done") and deadline is not None and now < deadline
    except CalendarUnavailable as exc:
        # **判不出今天是不是交易日 → 不交易**（fail-closed）。
        # 让异常直接冒出去会变成一条不透明的 APScheduler job error；
        # 而退回工作日启发式又会在休市日下单。两者都不可接受。
        logger.error("[scheduler] 交易日历不可用 → 本日**不交易**（fail-closed）: %s", exc)
        return {"skipped": "calendar_unavailable", "error": str(exc)}
    if not trading:
        logger.info("[scheduler] %s 非美股交易日 → 跳过每日摄取/交易", today.date())
        return {"skipped": "not_a_trading_day", "date": str(today.date())}
    out = run_daily_pipeline(settings.paper_dataset, settings.paper_start, incremental=True, now=now)
    if out.get("ingest_accepted"):
        logger.info(
            "[daily_trading_job] 完成 | 健康=%.3f | %d 因子 | %d 告警 | %d 错误",
            out.get("health_score", 0.0), out.get("n_alphas", 0),
            out.get("n_alerts", 0), out.get("n_errors", 0),
        )
    else:
        logger.warning("[daily_trading_job] 摄取未产生新 bar（%s）",
                       out.get("reject_reason") or out.get("reason"))
    if str(getattr(settings, "execution_mode", "off")) != "off":
        from app.tasks.forward import EXIT_OK, classify_run
        code, verdict = classify_run(out)
        (logger.info if code == EXIT_OK else logger.error)("[daily_trading_job] 执行结论：%s", verdict)
    return out


def _emit_daily_outcome(out: dict, run_date: str) -> None:
    """D15：每日运行的结论进告警（正常无订单与异常跳过分属不同分类）。告警出错不影响交易结果。"""
    try:
        from app.core.alerting.dispatcher import AlertDispatcher, emit_safely
        from app.core.alerting.events import events_for_daily_run
        d = AlertDispatcher()
        for ev in events_for_daily_run(out, run_date):
            emit_safely(ev, d)
    except Exception as exc:
        logger.error("[daily_trading_job] 运行结论没有进告警: %s", exc)


def _on_job_event(event) -> None:
    """D15：调度任务抛错 / 错过执行时间 → 严重告警（此前只在日志里）。"""
    from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED
    from app.core.alerting.dispatcher import emit_safely
    from app.core.alerting.events import AlertEvent
    when = getattr(event, "scheduled_run_time", None)
    if event.code == EVENT_JOB_ERROR:
        ev = AlertEvent(kind="scheduled_job_failed", severity="critical", title=f"调度任务 {event.job_id} 失败",
                        detail=str(getattr(event, "exception", "") or ""), dedup_key=f"job_failed:{event.job_id}:{when}")
    elif event.code == EVENT_JOB_MISSED:
        ev = AlertEvent(kind="scheduled_job_missed", severity="critical", title=f"调度任务 {event.job_id} 错过执行",
                        detail=f"计划时间 {when}（进程没在运行或被阻塞）", dedup_key=f"job_missed:{event.job_id}:{when}")
    else:
        return
    emit_safely(ev)


def alerts_retry_job() -> None:
    """D15：重新通知送达失败的告警。"""
    from app.core.alerting.dispatcher import AlertDispatcher
    out = AlertDispatcher().retry_failed()
    if out:
        logger.warning("[alerts] 重试 %d 条送达失败的告警：%s", len(out), out)


#: 每日交易任务与补跑的触发时刻（scheduler_timezone 下的 时, 分）。进可执行规格（复审 2026-10-19 §5.2）：
#: 改了提交时段就是另一个策略版本
DAILY_TRADING_AT = (21, 30)
DAILY_TRADING_RETRY_AT = (22, 45)


def nightly_discovery_job() -> None:
    """Phase 9.2：收盘后自主因子发现。观察市场 → GP → 赢家存为 CANDIDATE（不依赖用户假设）。"""
    from app.core.data_engine.dataset_registry import load_registry_dataset
    from app.core.discovery.discovery_engine import DiscoveryEngine
    from app.config import settings

    try:
        ds = load_registry_dataset(
            settings.discovery_dataset,
            start=settings.discovery_start, end=settings.discovery_end,
            health_check=False,
        )
    except Exception as exc:
        logger.error("[nightly_discovery_job] 数据加载失败，跳过本轮: %s", exc)
        return

    report = DiscoveryEngine().run(ds.data, save=True, dataset_key=settings.discovery_dataset)
    logger.info(
        "[nightly_discovery_job] regime=%s | 家族=%s | 新增 %d 个 CANDIDATE",
        report.regime, report.families, report.n_candidates,
    )


def daily_backup_job() -> None:
    """每日一致性快照备份：主库 + 调度库 + PIT（前向数据不可再生，丢了买不回来）。"""
    from app.tasks.backup import run_daily_backup
    res = run_daily_backup()
    if res.ok:
        logger.info("[scheduler] 备份完成 %s | %s | %.1f KB",
                    res.snapshot_dir, res.items, res.bytes_written / 1024)
    else:
        # 2026-10-25 审计 R4：不能只记日志 —— 发 critical 告警；结果已写进 last_backup.json，预检据此阻塞
        logger.error("[scheduler] 备份**失败** %s: %s", res.snapshot_dir, res.errors)
        from app.core.alerting.dispatcher import emit_safely
        from app.core.alerting.events import AlertEvent
        emit_safely(AlertEvent(kind="backup_failed", severity="critical", title="每日备份失败",
                               detail="; ".join(res.errors), dedup_key=f"backup_failed:{res.snapshot_dir}"))


def monthly_cost_calibration_job() -> None:
    """Task 8.2：每月对上月成交做成本模型校准，产出建议报告（**不自动改 CostParams**）。"""
    from datetime import date, timedelta
    from app.tasks.cost_calibration import run_monthly_calibration
    from app.config import settings

    today = date.today()
    first_this = today.replace(day=1)
    last_prev  = first_this - timedelta(days=1)      # 上月最后一天
    first_prev = last_prev.replace(day=1)            # 上月第一天
    report = run_monthly_calibration(
        settings.paper_dataset, first_prev.isoformat(), last_prev.isoformat(),
        write_path=f"cost_calibration_{first_prev:%Y%m}.md",
    )
    if report is None:
        logger.info("[monthly_cost_calibration_job] 上月无成交或无法校准，跳过")
    else:
        logger.info(
            "[monthly_cost_calibration_job] impact_coef 建议 %.4f→%.4f（×%.3f）— 仅建议，人工确认",
            report.current_impact_coef, report.recommended_impact_coef, report.recommended_scale,
        )

    # Phase 12.3：执行层开着时，同一月度任务再出一份"内部模拟 vs moomoo 纸交易"保真度报告
    if getattr(settings, "execution_mode", "off") != "off":
        from app.tasks.cost_calibration import run_monthly_fidelity
        fid = run_monthly_fidelity(
            settings.paper_dataset, first_prev.isoformat(), last_prev.isoformat(),
            write_path=f"exec_fidelity_{first_prev:%Y%m}.md")
        if fid is None:
            logger.info("[monthly_cost_calibration_job] 上月无纸交易成交，跳过保真度报告")
        else:
            logger.info("[monthly_cost_calibration_job] 保真度：%d 笔纸交易成交，总滑点中位 %s bps",
                        fid.n_live_fills, fid.total_slippage_bps.get("median"))


# ---------------------------------------------------------------------------
# Scheduler lifecycle
# ---------------------------------------------------------------------------

def create_scheduler(
    db_url:   Optional[str] = None,
    timezone: str = "UTC",
):
    """
    构建（不启动）BackgroundScheduler，任务持久化到 SQLite。

    独立函数便于测试：可构建后检查 job 注册情况而不真正运行。
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    from apscheduler.triggers.cron import CronTrigger

    url = db_url or "sqlite:///scheduler_jobs.db"
    sched = BackgroundScheduler(
        jobstores    = {"default": SQLAlchemyJobStore(url=url)},
        job_defaults = {
            "coalesce":           True,     # 多次错过合并为一次
            "misfire_grace_time": 3600,     # 错过 1 小时内补跑，超过跳过
            "max_instances":      1,        # 同一任务不并发
        },
        timezone = timezone,
    )
    # D15：任务抛错 / 错过 → 告警；送达失败的告警定期重试
    from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED
    from apscheduler.triggers.interval import IntervalTrigger
    sched.add_listener(_on_job_event, EVENT_JOB_ERROR | EVENT_JOB_MISSED)
    sched.add_job(alerts_retry_job, trigger=IntervalTrigger(minutes=30, timezone=timezone),
                  id="alerts_retry", name="重试送达失败的告警", replace_existing=True)
    # 第 17 条：盘中巡检（开启后才注册；每拍自己按交易所日历判断是否在常规时段）
    try:
        from app.config import settings
        if getattr(settings, "intraday_watchdog_enabled", False):
            from app.core.execution.intraday_watchdog import intraday_watchdog_job
            sched.add_job(intraday_watchdog_job,
                          trigger=IntervalTrigger(seconds=int(settings.intraday_watchdog_interval_s), timezone=timezone),
                          id="intraday_watchdog", name="盘中巡检（日亏熔断）", replace_existing=True,
                          max_instances=1, coalesce=True, misfire_grace_time=30)
    except Exception as exc:
        logger.error("[scheduler] 注册盘中巡检失败: %s", exc)
    # replace_existing=True：重启时以代码定义为准，避免 jobstore 中的旧定义漂移
    sched.add_job(
        daily_monitor_job,
        # 显式传 timezone：CronTrigger 默认用构造时的本地时区，会无视 scheduler 的 timezone
        trigger = CronTrigger(hour=21, minute=0, timezone=timezone),   # 21:00 (默认 UTC) ≈ 美股收盘后
        id      = "daily_monitor",
        name    = "每日因子衰减巡检",
        replace_existing = True,
    )
    # Task 7.1/7.3：Paper Trading 每日管线（默认关闭，ENABLE_PAPER_TRADING=true 时注册）
    try:
        from app.config import settings
        if settings.enable_paper_trading:
            sched.add_job(
                daily_trading_job,
                trigger = CronTrigger(hour=DAILY_TRADING_AT[0], minute=DAILY_TRADING_AT[1],
                                      timezone=timezone),                  # 巡检之后
                id      = "daily_trading",
                name    = "每日数据摄取 + 交易循环",
                replace_existing = True,
            )
            # 补跑：数据源收盘后更新有延迟时，21:30 那次拿不到当日 bar（no_new_bar → 不调仓）。
            # 同一个任务再跑一次：已摄取则复用 PIT 面板、执行层按 client id 去重，重跑安全；
            # 收盘后挂的 DAY 单仍在次日开盘前（审计设计问题 #4："每天只触发一次"）。
            sched.add_job(
                daily_trading_job,
                trigger = CronTrigger(hour=DAILY_TRADING_RETRY_AT[0], minute=DAILY_TRADING_RETRY_AT[1],
                                      timezone=timezone),
                id      = "daily_trading_retry",
                name    = "每日交易循环补跑（数据延迟兜底）",
                replace_existing = True,
            )
            sched.add_job(
                daily_trading_catchup_job,
                trigger = IntervalTrigger(minutes=CATCHUP_EVERY_MIN, timezone=timezone),
                id      = "daily_trading_catchup",
                name    = "每日交易补跑（按时没跑成时，下一个时段开盘前补上）",
                replace_existing = True, max_instances = 1, coalesce = True, misfire_grace_time = 60,
            )
            # Task 8.2：每月 1 号成本模型校准（产出建议报告，不自动改 CostParams）
            sched.add_job(
                monthly_cost_calibration_job,
                trigger = CronTrigger(day=1, hour=22, minute=0, timezone=timezone),
                id      = "monthly_cost_calibration",
                name    = "每月成本模型校准（建议报告）",
                replace_existing = True,
            )
    except Exception as exc:
        logger.warning("[scheduler] 注册 paper trading 任务失败: %s", exc)

    # Phase 9.2：自主因子发现（默认关闭，ENABLE_DISCOVERY=true 时注册）
    try:
        from app.config import settings
        if settings.enable_discovery:
            sched.add_job(
                nightly_discovery_job,
                trigger = CronTrigger(hour=23, minute=0, timezone=timezone),   # 收盘 + 交易循环之后
                id      = "nightly_discovery",
                name    = "每晚自主因子发现",
                replace_existing = True,
            )
    except Exception as exc:
        logger.warning("[scheduler] 注册 nightly_discovery 失败: %s", exc)

    # 每日一致性快照备份（前向数据不可再生 —— 丢了买不回来）
    try:
        from app.config import settings
        if getattr(settings, "enable_backup", True):
            sched.add_job(
                daily_backup_job,
                trigger = CronTrigger(hour=23, minute=45, timezone=timezone),  # 当日全部任务之后
                id      = "daily_backup",
                name    = "每日一致性快照备份",
                replace_existing = True,
            )
    except Exception as exc:
        logger.warning("[scheduler] 注册 daily_backup 失败: %s", exc)
    return sched


def start_scheduler(db_url: Optional[str] = None, timezone: str = "UTC"):
    """启动模块级单例调度器（幂等：已启动则直接返回）。"""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        return _scheduler
    _scheduler = create_scheduler(db_url=db_url, timezone=timezone)
    _scheduler.start()
    from app.core.host_power import KeepAwake, SuspendWatch
    awake = KeepAwake().start()
    _host_guards[:] = [awake, SuspendWatch(_on_host_suspended).start()]
    (logger.info if awake.granted else logger.warning)(
        "[scheduler] 防待机请求%s（合盖 / 电量耗尽仍会睡着 —— 睡着会被量出来并告警）",
        "已生效" if awake.granted else "没有生效（不是 Windows 或系统拒绝）")
    logger.info(
        "[scheduler] 已启动 | jobs=%s",
        [f"{j.id}(next={j.next_run_time})" for j in _scheduler.get_jobs()],
    )
    return _scheduler


def shutdown_scheduler() -> None:
    """停止调度器（幂等）。"""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("[scheduler] 已停止")
    _scheduler = None
    for g in _host_guards:
        g.stop()
    _host_guards.clear()


def get_scheduler_status() -> dict:
    """调度器运行状态与任务列表（API / 前端 FE-5.3 用）。"""
    if _scheduler is None or not _scheduler.running:
        return {"running": False, "jobs": []}
    return {
        "running": True,
        "jobs": [
            {
                "id":       j.id,
                "name":     j.name,
                "next_run": str(j.next_run_time) if j.next_run_time else None,
                "trigger":  str(j.trigger),
            }
            for j in _scheduler.get_jobs()
        ],
    }
