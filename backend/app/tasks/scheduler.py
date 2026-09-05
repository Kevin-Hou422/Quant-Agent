"""
scheduler.py — APScheduler 定时任务框架（Task 5.3）

设计说明
--------
- BackgroundScheduler + SQLAlchemyJobStore（SQLite 持久化）：进程重启后
  任务定义与 next_run_time 从数据库恢复，不丢失。
- misfire_grace_time=3600：停机期间错过的任务在重启后 1 小时内补跑一次，
  超过则跳过并记日志（coalesce=True 合并多次错过为一次）。
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


def daily_trading_job() -> None:
    """
    Task 7.1/7.3 + Phase 11：每日**前向增量**摄取（健康门）→ 交易循环。

    - 非交易日直接跳过（美股日历，节假日/周末）——不空转、不产生噪声日志。
    - 摄取走增量：PIT 空则回填，之后每天只拉新 bar；无新 bar 就不交易。
    """
    from app.tasks.daily_ingest import run_daily_pipeline
    from app.core.data_engine.market_calendar import is_trading_day
    from app.config import settings
    import pandas as _pd

    today = _pd.Timestamp.utcnow().tz_localize(None).normalize()
    if not is_trading_day(today):
        logger.info("[scheduler] %s 非美股交易日 → 跳过每日摄取/交易", today.date())
        return
    out = run_daily_pipeline(settings.paper_dataset, settings.paper_start, incremental=True)
    if out.get("ingest_accepted"):
        logger.info(
            "[daily_trading_job] 完成 | 健康=%.3f | %d 因子 | %d 告警 | %d 错误",
            out.get("health_score", 0.0), out.get("n_alphas", 0),
            out.get("n_alerts", 0), out.get("n_errors", 0),
        )
    else:
        logger.warning("[daily_trading_job] 摄取被拒（%s）→ 已跳过交易循环", out.get("reject_reason"))


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

    report = DiscoveryEngine().run(ds.data, save=True)
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
        logger.error("[scheduler] 备份**失败** %s: %s", res.snapshot_dir, res.errors)


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
                trigger = CronTrigger(hour=21, minute=30, timezone=timezone),  # 巡检之后
                id      = "daily_trading",
                name    = "每日数据摄取 + 交易循环",
                replace_existing = True,
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
