"""
dispatcher.py — 告警的落盘、去重、重试与送达状态（2026-10-22 用户决定 D15）

用户原话："首版选桌面通知＋持久化事件文件。实现分类、严重度、去重、重试和送达状态，区分正常无订单与异常跳过。
桌面通知须现场确认可见；文件落盘不等于通知成功。不可达时不能称为无人值守就绪。"

事件文件（JSON Lines，只追加）里有四种行：
    event         事件本身（分类、严重度、标题、详情、去重键、时刻）
    delivery      每一次通知尝试（结果 failed / handed_to_os 与错误）
    status        事件的送达结论：file_only（info 只进文件）/ suppressed（去重窗口内重复）/
                  handed_to_os（交给了操作系统 —— **不等于有人看见**）/ failed（重试用尽）
    confirmation  人现场确认某条测试通知确实看见了

无人值守就绪 = 最近 `confirm_valid_days` 天内有一次现场确认，且确认之后：
    · 没有送达失败的事件
    · 没有结果未知的事件（写了事件行、却没有送达结论 —— 进程在通知途中退出、结论行没写进去）
    · 事件文件没有读不出来的行（损坏的那一行可能正是一条失败）
    · 没有"事件根本没写进去"的告警通道故障（复审 21cd90b）
文件写进去了不算通知成功；通知交给系统了也不算有人看见。

告警通道故障（复审 21cd90b："事件无法持久化、通知结果未知或通道失败时，应有可靠的健康状态使无人值守新增风险受阻；
不以一条日志代替可供许可判断的状态"）：`emit_safely` 写事件文件失败时，把故障记到**另一份存储** —— 执行库的
exec_state `alert_channel_fault`（首次 / 最近时刻、次数、错误、哪条告警），并记 exec_events；执行库也写不进去时，
至少留在本进程里（`_PROCESS_FAULTS`）。就绪判断读它：故障晚于最近一次现场确认 → 不就绪。重启不清除；修好通道后
重新 `alerts test` + `confirm`（确认晚于故障）才恢复 —— 没有自动清除。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from app.core.alerting.events import AlertEvent, SEVERITIES

logger = logging.getLogger(__name__)

NOTIFY_SEVERITIES = ("warning", "critical")
FAULT_KEY = "alert_channel_fault"
#: 执行库也写不进去时的最后一层：本进程里记得（就绪判断同样读）
_PROCESS_FAULTS: List[dict] = []
_UNSET = object()


@dataclass
class DeliveryReport:
    event_id: str
    status:   str
    attempts: int = 0
    error:    str = ""


def default_event_file(settings=None) -> Path:
    """缺省事件文件：配置的 alert_event_file；没配 → 活库同目录下的 alerts/events.jsonl。"""
    if settings is None:
        from app.config import settings
    raw = str(getattr(settings, "alert_event_file", "") or "")
    if raw:
        return Path(raw).expanduser()
    url = str(getattr(settings, "database_url", "") or "")
    if url.startswith("sqlite"):
        return Path(url.split("///", 1)[-1]).expanduser().resolve().parent / "alerts" / "events.jsonl"
    return Path("alerts") / "events.jsonl"


class AlertDispatcher:
    def __init__(self, event_file: Optional[Path] = None, notifier: Optional[Callable[[str, str], None]] = None,
                 now: Optional[Callable[[], datetime]] = None, sleep: Callable[[float], None] = time.sleep,
                 dedup_minutes: Optional[float] = None, retry_attempts: Optional[int] = None,
                 confirm_valid_days: Optional[float] = None) -> None:
        from app.config import settings
        self.path = Path(event_file) if event_file is not None else default_event_file()
        if notifier is None:
            from app.core.alerting import desktop_notifier
            notifier = lambda title, body: desktop_notifier.notify(title, body)   # noqa: E731 —— 调用时再取（可被替换）
        self.notifier = notifier
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.sleep = sleep
        self.dedup = timedelta(minutes=float(dedup_minutes if dedup_minutes is not None
                                             else getattr(settings, "alert_dedup_minutes", 60)))
        self.retry_attempts = max(1, int(retry_attempts if retry_attempts is not None
                                         else getattr(settings, "alert_retry_attempts", 3)))
        self.confirm_valid = timedelta(days=float(confirm_valid_days if confirm_valid_days is not None
                                                  else getattr(settings, "alert_confirm_valid_days", 14)))

    # ------------------------------------------------------------------ 文件
    def _append(self, row: dict) -> None:
        from app.db.process_lock import process_lock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        with process_lock(self.path.with_suffix(".lock"), timeout_s=10.0):
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()

    def _rows(self) -> List[dict]:
        self.corrupt_lines: List[int] = []
        if not self.path.exists():
            return []
        out = []
        with open(self.path, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError as exc:
                    self.corrupt_lines.append(i + 1)
                    logger.error("[alerts] 事件文件第 %d 行读不出来（跳过）: %s", i + 1, exc)
        return out

    def history(self) -> List[dict]:
        """每个事件一行：事件内容 + 最后的送达结论（时间顺序）。"""
        events: Dict[str, dict] = {}
        for r in self._rows():
            if r.get("type") == "event":
                events[r["event_id"]] = {**r, "status": "written"}
            elif r.get("type") == "status" and r.get("event_id") in events:
                events[r["event_id"]]["status"] = r["status"]
                events[r["event_id"]]["status_at"] = r["at"]
            elif r.get("type") == "confirmation" and r.get("event_id") in events:
                events[r["event_id"]]["confirmed_by"] = r["actor"]
                events[r["event_id"]]["confirmed_at"] = r["at"]
        return list(events.values())

    def attempts(self, event_id: str) -> List[dict]:
        return [r for r in self._rows() if r.get("type") == "delivery" and r.get("event_id") == event_id]

    # ------------------------------------------------------------------ 发出
    def emit(self, ev: AlertEvent) -> DeliveryReport:
        now = self.now()
        eid = uuid.uuid4().hex[:16]
        self._append({"type": "event", "event_id": eid, "kind": ev.kind, "severity": ev.severity,
                      "title": ev.title, "detail": ev.detail, "dedup_key": ev.dedup_key, "extra": ev.extra,
                      "at": now.isoformat()})
        if self._recently_sent(ev.dedup_key, now, exclude=eid, severity=ev.severity):
            self._status(eid, "suppressed", now)
            return DeliveryReport(eid, "suppressed")
        if ev.severity not in NOTIFY_SEVERITIES:
            self._status(eid, "file_only", now)
            return DeliveryReport(eid, "file_only")
        return self._deliver(eid, ev.title, ev.detail, ev.severity)

    def _recently_sent(self, key: str, now: datetime, exclude: str, severity: str) -> bool:
        """只有同等级或更严重的先前事件才能去重；需通知的事件必须真的交给过系统。"""
        for e in self.history():
            if e["event_id"] == exclude or e.get("dedup_key") != key:
                continue
            if e["status"] not in ("handed_to_os", "file_only"):
                continue                                         # 失败的不算发过：重复事件应当再试
            previous_severity = e.get("severity")
            if previous_severity not in SEVERITIES or SEVERITIES.index(previous_severity) < SEVERITIES.index(severity):
                continue
            if severity in NOTIFY_SEVERITIES and e["status"] != "handed_to_os":
                continue                                         # 仅写文件不等于发过通知
            if now - datetime.fromisoformat(e["at"]) < self.dedup:
                return True
        return False

    def _deliver(self, eid: str, title: str, detail: str, severity: str) -> DeliveryReport:
        err = ""
        label = {"critical": "【严重】", "warning": "【注意】"}.get(severity, "")
        for i in range(1, self.retry_attempts + 1):
            try:
                self.notifier(f"Quant Agent {label}{title}", detail or title)
                self._append({"type": "delivery", "event_id": eid, "attempt": i, "result": "handed_to_os",
                              "at": self.now().isoformat()})
                self._status(eid, "handed_to_os", self.now())
                return DeliveryReport(eid, "handed_to_os", attempts=i)
            except Exception as exc:                             # 通知失败不能打断调用方（交易循环 / 调度器）
                err = str(exc)
                self._append({"type": "delivery", "event_id": eid, "attempt": i, "result": "failed",
                              "error": err, "at": self.now().isoformat()})
                logger.error("[alerts] 事件 %s 第 %d 次通知失败: %s", eid, i, exc)
                if i < self.retry_attempts:
                    self.sleep(min(2.0 ** (i - 1), 30.0))
        self._status(eid, "failed", self.now())
        return DeliveryReport(eid, "failed", attempts=self.retry_attempts, error=err)

    def _status(self, eid: str, status: str, at: datetime) -> None:
        self._append({"type": "status", "event_id": eid, "status": status, "at": at.isoformat()})

    def retry_failed(self) -> Dict[str, str]:
        """重新通知送达失败的事件（调度器定期调用）。返回 {事件 id: 新结论}。"""
        out = {}
        for e in self.history():
            if e["status"] == "failed":
                out[e["event_id"]] = self._deliver(e["event_id"], e["title"], e["detail"], e["severity"]).status
        return out

    # ------------------------------------------------------------------ 现场确认 / 就绪
    def send_test_notification(self) -> str:
        """发一条测试通知（需要人现场看一眼，然后 confirm_visible）。返回事件 id。"""
        ev = AlertEvent(kind="notification_test", severity="warning", title="通知通道测试",
                        detail="看到这条通知后，请在终端里确认（python -m app.tasks.alerts confirm <事件 id> --actor 你）",
                        dedup_key=f"notification_test:{uuid.uuid4().hex}")
        return self.emit(ev).event_id

    def confirm_visible(self, event_id: str, actor: str) -> None:
        actor = (actor or "").strip()
        if not actor:
            raise ValueError("现场确认必须写明是谁看见的")
        ev = next((e for e in self.history() if e["event_id"] == event_id), None)
        if ev is None or ev.get("kind") != "notification_test":
            raise ValueError(f"{event_id} 不是一条测试通知")
        if ev["status"] != "handed_to_os":
            raise ValueError(f"测试通知 {event_id} 没有交给系统（{ev['status']}），谈不上看见 —— 先排查通知通道")
        self._append({"type": "confirmation", "event_id": event_id, "actor": actor, "at": self.now().isoformat()})
        logger.warning("[alerts] %s 现场确认测试通知 %s 可见", actor, event_id)

    def unattended_readiness(self, channel_fault=_UNSET) -> Tuple[bool, str]:
        """`channel_fault`：执行库里记的告警通道故障（缺省读缺省执行库；只读工具从副本读出来传进来，None = 没有）。"""
        now = self.now()
        try:
            hist = self.history()
        except OSError as exc:
            logger.error("[alerts] 事件文件读不了: %s", exc)
            return False, f"事件文件 {self.path} 读不了（{exc}）—— 证明不了通知通道可用"
        if channel_fault is _UNSET:
            try:
                channel_fault = read_channel_fault()
            except Exception as exc:
                logger.error("[alerts] 读不到告警通道故障记录: %s", exc)
                return False, f"读不到告警通道故障记录（{exc}）—— 证明不了告警都写进去了"
        confirmed = [e for e in hist if e.get("confirmed_at")]
        if not confirmed:
            return False, ("桌面通知没有现场确认可见过 —— 不能称为无人值守就绪。"
                           "先 `python -m app.tasks.alerts test`，看见后 `confirm`")
        last = max(confirmed, key=lambda e: e["confirmed_at"])
        at = datetime.fromisoformat(last["confirmed_at"])
        if now - at > self.confirm_valid:
            return False, (f"最近一次现场确认在 {at.date()}，已过期（超过 {self.confirm_valid.days} 天）—— "
                           f"重新发测试通知并确认")
        failed = [e for e in hist if e["status"] == "failed" and datetime.fromisoformat(e["at"]) >= at]
        if failed:
            return False, (f"确认之后有 {len(failed)} 条告警通知失败（最近：{failed[-1]['title']}）—— "
                           f"通知通道不可达时不能称为无人值守就绪")
        unknown = [e for e in hist if e["status"] == "written" and e.get("severity") in NOTIFY_SEVERITIES
                   and datetime.fromisoformat(e["at"]) >= at]
        if unknown:
            return False, (f"确认之后有 {len(unknown)} 条告警没有送达结论（最近：{unknown[-1]['title']}）—— "
                           f"通知结果未知，不能当作送达")
        if self.corrupt_lines:
            return False, (f"事件文件有 {len(self.corrupt_lines)} 行读不出来（第 {self.corrupt_lines[:5]} 行…）—— "
                           f"证明不了其中没有失败")
        faults = [f for f in ([channel_fault] if channel_fault else []) + _PROCESS_FAULTS
                  if datetime.fromisoformat(f["last_at"]) >= at]
        if faults:
            f = faults[0]
            return False, (f"确认之后告警事件没能写进事件文件 {f.get('count', 1)} 次（最近 {f['last_at']}：{f.get('title')}，"
                           f"{f.get('error')}）—— 修好后重新 `alerts test` 并 `confirm`")
        return True, f"{last['confirmed_by']} 于 {at.date()} 现场确认可见；此后没有通知失败"


def read_channel_fault(store=None) -> Optional[dict]:
    """执行库里记的告警通道故障（没有 → None）。"""
    if store is None:
        from app.db.execution_store import ExecutionStore
        store = ExecutionStore()
    raw = store.get_state(FAULT_KEY)
    return json.loads(raw) if raw else None


def record_channel_fault(ev: AlertEvent, exc: BaseException, now: Optional[datetime] = None, store=None) -> bool:
    """告警没写进事件文件：把故障记到执行库（另一份存储），供无人值守就绪判断。返回是否落库。"""
    at = (now or datetime.now(timezone.utc)).isoformat()
    fault = {"first_at": at, "last_at": at, "count": 1, "kind": ev.kind, "title": ev.title,
             "severity": ev.severity, "error": f"{type(exc).__name__}: {exc}"}
    _PROCESS_FAULTS.append(fault)
    try:
        prev = read_channel_fault(store)
        if store is None:
            from app.db.execution_store import ExecutionStore
            store = ExecutionStore()
        if prev:
            fault = {**fault, "first_at": prev.get("first_at", at), "count": int(prev.get("count", 0)) + 1}
        store.set_state(FAULT_KEY, json.dumps(fault, ensure_ascii=False))
        store.add_event("alert_channel_fault", actor="alerting", reason=fault["error"], payload=fault)
        return True
    except Exception as exc2:
        logger.critical("[alerts] 告警通道故障也没能记进执行库（只留在本进程里）: %s", exc2)
        return False


def emit_safely(ev: AlertEvent, dispatcher: Optional[AlertDispatcher] = None) -> Optional[DeliveryReport]:
    """给交易循环 / 调度器用：告警本身出错不打断调用方 —— 但故障要落成无人值守能读到的状态（不只是一条日志）。"""
    try:
        return (dispatcher or AlertDispatcher()).emit(ev)
    except Exception as exc:
        logger.error("[alerts] 告警没有写进事件文件（%s：%s）: %s", ev.kind, ev.title, exc)
        record_channel_fault(ev, exc, now=dispatcher.now() if dispatcher is not None else None)
        return None
