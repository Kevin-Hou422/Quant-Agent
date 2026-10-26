"""
events.py — 告警事件：分类、严重度、去重键（2026-10-22 用户决定 D15）

分类是一张固定的表（不认识的分类 / 严重度直接报错，不让拼错的名字悄悄成为一类）。每日运行的结果按
`events_for_daily_run` 归类，**正常无订单**（no_orders_normal / no_trading_day，info）与**异常跳过**
（run_skipped / execution_blocked / ingest_rejected …，critical）永远不会落在同一个分类里。

下单门的拒单按原因分类（2026-10-25 审计 R4：计划 2 张买单都因缺报价时刻 / 独立报价被拒、实际 0 张，
却得到退出码 0 和 run_completed/info）—— `execution_outcome` 是每日结论与 `forward.classify_run` 的**同一个**出口：
    数据 / 报价 / 估值异常、熔断类  → 异常；本轮新单全被它们拒 → orders_rejected_abnormal（critical，运行失败）
    计划内风控（单票 / 总敞口 / 行业 / 净敞口 / 参与率 / 不做空 / 买入力）→ 正常的风险缩减；全被拒 → warning，不算失败
    部分被异常拒 → orders_partially_rejected（warning，运行"受损"：退出码单列）
    计划的单都已在之前发过 → orders_already_submitted（info）
正式证据 / 信号存档写入失败（record_write_failed）与备份失败（backup_failed）各有分类，不能只留日志。
"""

from __future__ import annotations

from collections import Counter
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
    "orders_already_submitted": "计划的单都已在之前发过（同日重跑去重）",
    "orders_rejected_abnormal": "本轮新单全部被数据 / 报价 / 估值异常或熔断拦下，实际没有下单",
    "orders_rejected_by_limits": "本轮新单全部被计划内风控缩减（限额 / 买入力），实际没有下单",
    "orders_partially_rejected": "部分新单因数据 / 报价 / 估值异常被拦下",
    "record_write_failed":   "正式证据 / 每日记录 / 信号存档写入失败",
    "backup_failed":         "每日备份失败",
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


#: 下单门拒单原因 → 类别（app/core/execution/pretrade_gate.py 的全部原因；不认识的按异常）
GATE_REASON_CLASS = {
    **{r: "data" for r in ("no_reference_price", "price_mismatch", "no_independent_quote", "quote_without_time",
                           "no_decision_clock", "quote_in_future", "stale_quote", "quote_mismatch",
                           "quote_outside_session", "quote_stale_in_session", "no_adv", "gross_unpriceable",
                           "sector_unpriceable", "net_unpriceable", "equity_unavailable")},
    **{r: "halt" for r in ("kill_switch_engaged", "daily_loss_halt")},
    "corporate_action_quarantine": "quarantine",
    **{r: "limits" for r in ("would_short", "exceeds_adv_participation", "exceeds_name_limit",
                             "exceeds_gross_limit", "exceeds_sector_limit", "exceeds_net_limit",
                             "insufficient_buying_power", "permit_notional_limit")},
}
ABNORMAL_CLASSES = ("data", "halt", "unknown")


def gate_reason_class(reason: str) -> str:
    return GATE_REASON_CLASS.get(str(reason or ""), "unknown")


def execution_outcome(ex: dict) -> dict:
    """
    执行报告 → {kind, severity, status, detail, rejections}。status：ok / degraded / failed。
    只看"计划了、没在之前发过的新单"：它们被谁拦下、发出去多少。
    """
    ex = ex or {}
    planned, dup = int(ex.get("n_planned") or 0), int(ex.get("n_duplicate") or 0)
    sent, rej = int(ex.get("n_submitted") or 0), int(ex.get("n_gate_rejected") or 0)
    listed = list((ex.get("gate") or {}).get("rejected") or [])
    by = Counter(gate_reason_class(r.get("reason")) for r in listed)
    if rej > len(listed):                                  # 报告里没列出原因的拒单：按原因不明（异常）
        by["unknown"] += rej - len(listed)
    abnormal = sum(by[c] for c in ABNORMAL_CLASSES)
    counts = {k: v for k, v in sorted(by.items()) if v}
    detail = (f"计划 {planned} / 之前已发 {dup} / 已下 {sent} / 风控拒 {rej}"
              + (f"（{', '.join(f'{k} {v}' for k, v in counts.items())}）" if counts else "")
              + (f"；独立报价取不到：{ex['quote_error']}" if ex.get("quote_error") else ""))
    fresh = planned - dup
    out = {"rejections": counts, "detail": detail}
    if planned == 0 and sent == 0:
        return {**out, "kind": "no_orders_normal", "severity": "info", "status": "ok"}
    if fresh <= 0 and sent == 0:
        return {**out, "kind": "orders_already_submitted", "severity": "info", "status": "ok"}
    if sent == 0 and rej >= fresh:
        if abnormal:
            return {**out, "kind": "orders_rejected_abnormal", "severity": "critical", "status": "failed"}
        return {**out, "kind": "orders_rejected_by_limits", "severity": "warning", "status": "ok"}
    if abnormal:
        return {**out, "kind": "orders_partially_rejected", "severity": "warning", "status": "degraded"}
    return {**out, "kind": "run_completed", "severity": "info", "status": "ok"}


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
    else:
        o = execution_outcome(ex)
        add(o["kind"], o["severity"], KINDS[o["kind"]] if o["kind"] != "run_completed" else "运行完成", o["detail"])

    for f in (pf.get("record_failures") or []) if isinstance(pf, dict) else []:
        add("record_write_failed", "critical", f"{f.get('stage')} 写入失败", f.get("error"),
            k=f"{key}:record:{f.get('stage')}")

    if isinstance(pf, dict):
        for tk, since in (pf.get("ca_flags") or {}).items():
            add("corporate_action_quarantine", "warning", f"{tk} 疑似未知公司行动，已隔离",
                f"自 {since} 起不加风险；核对券商股数 / 现金与数据源复权后结案", k=f"corporate_action:{tk}:{since}")
        for v in pf.get("post_band_violations") or []:
            add("post_band_violation", "warning", "带后风控仍违规", v, k=f"{key}:post_band:{v}")
    return evs
