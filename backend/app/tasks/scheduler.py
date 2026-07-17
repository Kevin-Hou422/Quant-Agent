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

    store   = AlphaStore()
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
        trigger = CronTrigger(hour=21, minute=0),   # UTC 21:00 ≈ 美股收盘后
        id      = "daily_monitor",
        name    = "每日因子衰减巡检",
        replace_existing = True,
    )
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
