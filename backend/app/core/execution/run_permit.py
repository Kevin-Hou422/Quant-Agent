"""
run_permit.py — 运行许可：每一个会新增风险的入口都消费同一份、每次重新核对的许可（2026-10-25 审计 R3）

审计原话："无人值守就绪检查主要在启动包装器，调度入口没有统一执行……将通知就绪设为 false，直接调用 daily_trading_job，
仍会进入交易任务主体。"要求："将运行模式和授权形成统一、可验证的运行许可，由所有新增风险入口强制消费，包括 API、调度、
CLI、重试和进程重启。无人值守许可绑定代码/规格、账户、数据源、验收证据和通知状态；失效时停止新增风险并报警，允许对账、
撤单、熔断维护和必要减仓。人工联调使用单独、明确、限额的许可……就绪结论不能只在部署当时计算一次。"

在哪里强制：所有调仓下单都经过 `OrderManager.run_cycle`（调度任务、重试任务、`forward run-now`、验收脚本的 cycle 都是它；
API 只有对账 / 接受差异 / 熔断，不下调仓单）。`OrderManager.from_settings` 给它装上本模块的核对；**每一轮**下单之前
重新核对（不是部署时算一次）。不通过 → 本轮不发任何调仓单（报告 blocked = "run_permit: …"，每日结论判失败、发 critical），
对账、撤销过期单、熔断续做全平照常。

许可存在执行库的 `exec_state`（键 run_permit；验收用的执行库各有各的许可），签发 / 撤销都记 exec_events：

    supervised   人工联调：写明操作人、理由、账户、有效期（≤ 24 小时）、单笔金额上限；不要求通知 / 验收证据
    unattended   无人值守：另外绑定固定代码版本（不能是 +dirty / unknown）、运行中策略的规格哈希、价格源、
                 **经过判定的**现场验收证据（`acceptance_evidence.assess`：八个场景逐个重算，证据的账户 / 代码 /
                 执行合同要与许可一致；未覆盖的场景只能由签发人写明理由豁免，失败的不能）及其 sha256；
                 签发时与每次核对时都要求通知通道就绪、最近一次备份成功

每次核对：有许可、没过期、账户 = 当前连上的账户、代码版本 = 当前、价格源 = 当前配置、（绑定了的）规格哈希 = 运行中策略的；
无人值守再加：通知就绪（含"告警没写进事件文件"的通道故障）、备份健康、验收证据文件没变（sha256）且按此刻的代码 / 规格
重新判定仍然通过、盘中巡检最近一拍没有失明 / 行情缺失（复审 21cd90b）。
任何一项不满足 → 不新增风险，原因逐条写出。

    python -m app.tasks.run_permit issue --mode supervised --account <acc_id> --hours 8 --max-order-notional 2000 \\
        --actor <你> --reason "…"
    python -m app.tasks.run_permit status | revoke --actor <你> --reason "…"
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

MODES = ("supervised", "unattended")
STATE_KEY = "run_permit"
MAX_SUPERVISED = timedelta(hours=24)
MAX_UNATTENDED = timedelta(days=7)


class PermitRefused(ValueError):
    """签发条件不满足。"""


@dataclass
class PermitDecision:
    allowed: bool
    mode: str = ""
    reasons: List[str] = field(default_factory=list)
    max_order_notional: Optional[float] = None
    expires_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _utc(now: Optional[datetime]) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def _running_spec(settings) -> dict:
    """运行中策略的 {spec_hash, executable_spec}（没有 → {}）。"""
    from app.db.strategy_store import StrategyStore
    rec = StrategyStore(db_url=str(settings.database_url)).latest_running()
    return json.loads(rec.verdict or "{}") if rec is not None else {}


def _running_spec_hash(settings) -> str:
    return _running_spec(settings).get("spec_hash", "")


def _running_contract(settings) -> Optional[str]:
    from app.core.execution.acceptance_evidence import contract_of_spec
    return contract_of_spec(_running_spec(settings).get("executable_spec"))


def _unattended_health(settings, now: datetime) -> List[str]:
    """无人值守的两项现场状态：通知通道就绪、最近一次备份成功。"""
    from app.core.alerting.dispatcher import AlertDispatcher, default_event_file
    from app.tasks.backup import backup_health
    out = []
    ok, why = AlertDispatcher(event_file=default_event_file(settings), now=lambda: now).unattended_readiness()
    if not ok:
        out.append(f"通知未就绪：{why}")
    ok, why, _ = backup_health(getattr(settings, "backup_dir", None), now)
    if not ok:
        out.append(f"备份不健康：{why}")
    return out


def _store_health(store) -> List[str]:
    """读执行库里的现场状态（不可注入：测试替换 health 也绕不过）。"""
    from app.core.execution.intraday_watchdog import blocking_reasons
    return blocking_reasons(store)


def issue(store, *, mode: str, actor: str, reason: str, account: int, hours: float,
          max_order_notional: Optional[float] = None, acceptance_evidence: Optional[str] = None,
          settings=None, now: Optional[datetime] = None, code: Optional[str] = None,
          spec_hash: Optional[str] = None, health: Optional[Callable] = None, contract: Optional[str] = None,
          waive_uncovered: Optional[dict] = None) -> dict:
    """签发一份许可（覆盖旧的）。条件不满足 → PermitRefused，什么都不写。

    `waive_uncovered`：{场景: 理由}，只对"未覆盖"的场景有效（例如当天没出现部分成交），写进许可、Go/No-Go 单列。
    """
    from app.core.data_engine.input_identity import code_version
    if settings is None:
        from app.config import settings
    now = _utc(now)
    if mode not in MODES:
        raise PermitRefused(f"模式只能是 {MODES}")
    if not str(actor or "").strip() or not str(reason or "").strip():
        raise PermitRefused("签发许可要写明操作人和理由")
    if not int(account or 0):
        raise PermitRefused("要写明账户 id（不接受 0 = 自动取第一个）")
    life = timedelta(hours=float(hours))
    cap = MAX_SUPERVISED if mode == "supervised" else MAX_UNATTENDED
    if not (timedelta(0) < life <= cap):
        raise PermitRefused(f"{mode} 许可的有效期要在 (0, {cap}] 之内")
    code = code if code is not None else code_version()
    permit = {"mode": mode, "actor": actor, "reason": reason, "account": int(account),
              "issued_at": now.isoformat(), "expires_at": (now + life).isoformat(), "code_version": code,
              "price_source": str(getattr(settings, "price_source", "")),
              "max_order_notional": None, "spec_hash": "", "acceptance_evidence": None}
    if mode == "supervised":
        if not float(max_order_notional or 0) > 0:
            raise PermitRefused("人工联调许可要写明单笔金额上限")
        permit["max_order_notional"] = float(max_order_notional)
    else:
        if code in ("", "unknown") or code.endswith("+dirty"):
            raise PermitRefused(f"无人值守许可要绑定固定的代码版本（当前 {code!r}）")
        h = spec_hash if spec_hash is not None else _running_spec_hash(settings)
        if not h:
            raise PermitRefused("没有带规格哈希的运行中策略 —— 无人值守许可要绑定规格")
        if not acceptance_evidence or not Path(acceptance_evidence).is_file():
            raise PermitRefused("无人值守许可要给现场验收证据文件（simulate_acceptance.jsonl）")
        from app.core.execution import acceptance_evidence as ae
        c = contract if contract is not None else _running_contract(settings)
        judged = ae.assess(acceptance_evidence, account=int(account), code=code, contract=c, waived=waive_uncovered)
        if not judged["ok"]:
            raise PermitRefused("现场验收证据不支持无人值守：" + "；".join(ae.reasons(judged)))
        problems = (health or _unattended_health)(settings, now) + _store_health(store)
        if problems:
            raise PermitRefused("；".join(problems))
        permit.update(spec_hash=h, acceptance_evidence={
            "path": judged["path"], "sha256": judged["sha256"], "contract": c, "waived": judged["waived"],
            "scenarios": {s: v["status"] for s, v in judged["scenarios"].items()}})
    store.set_state(STATE_KEY, json.dumps(permit, ensure_ascii=False))
    store.add_event("run_permit_issue", actor=actor, reason=reason, payload=permit)
    return permit


def revoke(store, *, actor: str, reason: str) -> None:
    if not str(actor or "").strip() or not str(reason or "").strip():
        raise PermitRefused("撤销许可要写明操作人和理由")
    old = current(store)
    store.set_state(STATE_KEY, "")
    store.add_event("run_permit_revoke", actor=actor, reason=reason, payload=old or {})


def current(store) -> Optional[dict]:
    raw = store.get_state(STATE_KEY)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        logger.error("[permit] 运行许可记录解析失败 → 按损坏处理（不新增风险）: %s", exc)
        return {"mode": "", "corrupt": True}


def evaluate(store, *, account: Optional[int], settings=None, now: Optional[datetime] = None,
             code: Optional[str] = None, spec_hash: Optional[str] = None,
             health: Optional[Callable] = None, contract: Optional[str] = None) -> PermitDecision:
    """此刻能不能新增风险。每一轮下单之前调用（许可的状态会变：过期、账户 / 源 / 代码 / 规格 / 通知 / 备份）。"""
    from app.core.data_engine.input_identity import code_version
    if settings is None:
        from app.config import settings
    now = _utc(now)
    p = current(store)
    if not p:
        return PermitDecision(False, reasons=["没有运行许可 —— `python -m app.tasks.run_permit issue …` 签发"])
    if p.get("corrupt") or p.get("mode") not in MODES:
        return PermitDecision(False, reasons=["运行许可记录损坏 / 模式不认识"])
    why: List[str] = []
    if now >= datetime.fromisoformat(p["expires_at"]):
        why.append(f"许可已于 {p['expires_at']} 过期")
    if account is None or int(account) != int(p["account"]):
        why.append(f"账户变了：许可给的是 {p['account']}，现在连的是 {account}")
    cur_code = code if code is not None else code_version()
    if cur_code != p["code_version"]:
        why.append(f"代码版本变了：许可绑定 {p['code_version']}，现在是 {cur_code}")
    src = str(getattr(settings, "price_source", ""))
    if src != p["price_source"]:
        why.append(f"价格源变了：许可绑定 {p['price_source']!r}，现在是 {src!r}")
    if p.get("spec_hash"):
        h = spec_hash if spec_hash is not None else _running_spec_hash(settings)
        if h != p["spec_hash"]:
            why.append(f"运行中策略的规格变了：许可绑定 {p['spec_hash'][:12]}，现在是 {str(h)[:12] or '无'}")
    if p["mode"] == "unattended":
        ev = p.get("acceptance_evidence") or {}
        if not ev.get("path") or not Path(ev["path"]).is_file() or _sha(Path(ev["path"])) != ev.get("sha256"):
            why.append("现场验收证据文件不在了或被改过")
        else:
            from app.core.execution import acceptance_evidence as ae
            c = contract if contract is not None else _running_contract(settings)
            judged = ae.assess(ev["path"], account=int(p["account"]), code=cur_code, contract=c,
                               waived=ev.get("waived"))
            if not judged["ok"]:
                why.append("现场验收证据按此刻的代码 / 规格判定不通过：" + "；".join(ae.reasons(judged)))
        why += (health or _unattended_health)(settings, now) + _store_health(store)
    return PermitDecision(not why, mode=p["mode"], reasons=why, max_order_notional=p.get("max_order_notional"),
                          expires_at=p["expires_at"])


def settings_check(settings=None) -> Callable:
    """给 `OrderManager.from_settings` 用：按执行库里的许可、当前连上的账户核对。"""
    def check(om) -> PermitDecision:
        acc = (om.gw.describe() or {}).get("acc_id")
        return evaluate(om.store, account=int(acc) if acc is not None else None, settings=settings,
                        now=om._clock())                                       # noqa: SLF001
    return check
