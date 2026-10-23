"""
events.py — 告警事件：分类、严重度、去重键（2026-10-22 用户决定 D15）

分类是一张固定的表（不认识的分类 / 严重度直接报错，不让拼错的名字悄悄成为一类）。每日运行的结果按
`events_for_daily_run` 归类，**正常无订单**（no_orders_normal / no_trading_day，info）与**异常跳过**
（run_skipped / execution_blocked / ingest_rejected …，critical）永远不会落在同一个分类里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

SEVERITIES = ("info", "warning", "critical")

#: 分类 → 一句话说明（也是允许出现的全部分类）
KINDS = {
    # 每日运行
    "no_trading_day":        "非交易日，按计划不运行",
    "run_completed":         "运行完成并下了单",
    "no_orders_normal":      "运行完成，按计划没有订单（目标在无交易带内等）",
    "run_skipped":           "任务没有执行（日历不可用等）",
    "ingest_rejected":       "数据摄取被健康门拒绝",
    "no_new_bar":            "没有新 bar（数据晚到；重试任务会再取）",
    "portfolio_failed":      "组合账本 / 目标生成失败",
    "execution_off":         "执行层关闭（EXECUTION_MODE=off），没有下单",
    "execution_blocked":     "执行被拦（熔断 / 对账差异 / 规格不一致 / 无运行中策略等）",
    "order_submit_errors":   "有订单下单异常（待对账确认）",
    "corporate_action_quarantine": "疑似未知公司行动，已隔离（不加风险），待人对账",
    "post_band_violation":   "无交易带 + 二次风控后仍违规",
    # 调度
    "scheduled_job_failed":  "调度任务抛错",
    "scheduled_job_missed":  "调度任务错过了执行时间",
    # 风险
    "kill_switch":           "熔断开关被拉起 / 确认",
    "intraday_watchdog":     "盘中巡检发现问题或采取了动作",
    # 通道
    "notification_test":     "通知通道的测试通知（现场确认是否可见）",
}


@dataclass
class AlertEvent:
    kind:      str
    severity:  str
    title:     str
    detail:    str
    dedup_key: str
    extra:     dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"未知的告警分类 {self.kind!r}（允许：{sorted(KINDS)}）")
        if self.severity not in SEVERITIES:
            raise ValueError(f"未知的严重度 {self.severity!r}（允许：{SEVERITIES}）")
        if not str(self.dedup_key or "").strip():
            raise ValueError("告警必须有去重键")


def events_for_daily_run(out: dict, run_date: str) -> List[AlertEvent]:
    """每日管线的结果 → 告警事件（第一条是本次运行的结论；其后是附带的问题）。"""
    out = out or {}
    pf = out.get("portfolio") or {}
    ex = pf.get("execution") if isinstance(pf, dict) else None
    key = f"daily_run:{run_date}"
    evs: List[AlertEvent] = []

    def add(kind, severity, title, detail="", k=None):
        evs.append(AlertEvent(kind=kind, severity=severity, title=title, detail=str(detail or ""),
                              dedup_key=k or f"{key}:{kind}"))

    if out.get("skipped") == "not_a_trading_day":
        add("no_trading_day", "info", f"{run_date} 非交易日，不运行")
    elif out.get("skipped"):
        add("run_skipped", "critical", "每日任务没有执行", f"{out.get('skipped')} {out.get('error', '')}".strip())
    elif out.get("ingest_accepted") is False and out.get("reason") == "no_new_bar":
        add("no_new_bar", "warning", "没有新 bar", "数据源还没有发布最近一个交易日的数据；重试任务会再取")
    elif out.get("ingest_accepted") is False:
        add("ingest_rejected", "critical", "数据摄取被拒", out.get("reject_reason") or out.get("reason"))
    elif isinstance(pf, dict) and pf.get("error"):
        add("portfolio_failed", "critical", "组合账本失败", pf.get("error"))
    elif not ex:
        add("execution_blocked", "critical", "执行层没有运行", pf.get("reason") if isinstance(pf, dict) else "")
    elif ex.get("mode") == "off":
        add("execution_off", "warning", "执行层关闭，没有下单", "EXECUTION_MODE=off")
    elif ex.get("blocked"):
        add("execution_blocked", "critical", "执行被拦", ex.get("blocked"))
    elif ex.get("n_submit_errors"):
        add("order_submit_errors", "critical", f"{ex['n_submit_errors']} 张单下单异常",
            f"计划 {ex.get('n_planned', 0)} / 已下 {ex.get('n_submitted', 0)}")
    elif not ex.get("n_planned") and not ex.get("n_submitted"):
        add("no_orders_normal", "info", "运行完成，按计划没有订单")
    else:
        add("run_completed", "info", "运行完成",
            f"计划 {ex.get('n_planned', 0)} / 已下 {ex.get('n_submitted', 0)} / 风控拒 {ex.get('n_gate_rejected', 0)}")

    if isinstance(pf, dict):
        for tk, since in (pf.get("ca_flags") or {}).items():
            add("corporate_action_quarantine", "warning", f"{tk} 疑似未知公司行动，已隔离",
                f"自 {since} 起不加风险；核对券商股数 / 现金与数据源复权后结案", k=f"corporate_action:{tk}:{since}")
        for v in pf.get("post_band_violations") or []:
            add("post_band_violation", "warning", "带后风控仍违规", v, k=f"{key}:post_band:{v}")
    return evs
