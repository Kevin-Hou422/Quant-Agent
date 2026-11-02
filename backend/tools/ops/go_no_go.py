"""
go_no_go.py — 启动前的 Go/No-Go 核对单（2026-10-25 审计"工作包 D：生成最终 Go/No-Go，不代替用户启动"；复审 21cd90b）

    python tools/ops/go_no_go.py --db C:\\QuantAgentData\\alphas.db --dataset us_broad_large \\
        --exec-db <正式执行库> --acceptance-exec-db <验收执行库> [--evidence <simulate_acceptance.jsonl>] \\
        --backup-dir <备份目录> --alert-events <事件文件> --settings-json <去掉密钥的配置导出> [--out go_no_go.json]

只读：库按停写窗口复制后读副本（`research_boundary_report.read_frozen`，不对活库建连接、不建表）；其余都是读文件。
不读 .env：配置项从 `--settings-json`（你导出的、去掉密钥的那份）读。

四个结论分开给，任何一个都不代替另一个（复审"七"）：

    supervised    能否开始人工监督的工程联调（验收执行库、人工联调许可由你在现场签）
    paper         能否开始正式、**未验证**的纸交易证据收集（paper_running）
    unattended    能否无人值守
    verified      是否满足这份策略预登记的 verified_active 条件

每个核对项标明它挡哪几个结论。前三个结论只有"No-Go（列出阻断项）"或"可提交你做最后决定"两种 —— **工具永远不给 Go**；
第四个只说满足 / 不满足。人工确认项列出来，不替你打勾。

现场验收（复审 21cd90b"许可与验收证据绑定"）：只认**正式执行库里无人值守许可绑定的那一份**证据文件，用
`acceptance_evidence.assess`（与许可签发、每轮核对同一份判定）按许可的账户 / 代码版本 / 豁免、运行中策略规格的执行合同重算；
`--evidence` 给了就必须是同一份文件、同一个 sha256，否则阻断。没有无人值守许可时，按 `--evidence` 判定（只用于前两个结论）。
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.execution.acceptance_evidence import SCENARIOS as _SCENARIOS  # noqa: E402

BACKEND = Path(__file__).resolve().parents[2]

SUPERVISED, PAPER, UNATTENDED, VERIFIED = "supervised", "paper", "unattended", "verified"
CONCLUSIONS = {SUPERVISED: "能否开始人工监督工程联调", PAPER: "能否开始正式但未验证的纸交易证据收集",
               UNATTENDED: "能否无人值守", VERIFIED: "是否满足预登记的 verified_active 条件"}
#: 正式运行（纸交易与无人值守）都要满足的项
FORMAL = (PAPER, UNATTENDED)
ALL3 = (SUPERVISED, PAPER, UNATTENDED)

#: 审计要求的现场验收场景（就是 acceptance_evidence.SCENARIOS）
SCENARIOS = tuple(_SCENARIOS)

#: 工具看不到、必须现场确认的项（审计决定 8 / 9 与工作包 C；复审"六""七"）
MANUAL = (
    "桌面通知在实际启动方式、同一 Windows 用户、锁屏 / 会话断开、重启后都能看到，并确认了准确的事件 id",
    "专用 SIMULATE 账户的初始持仓 / 现金 / 出入金基线已核对，账户里没有不明现金流",
    "OpenD / SDK 版本、部署路径、运行用户已记录",
    "盘后提交的 DAY 单在实际账户上 T+1 有效（day_order_across_sessions 现场通过）",
    "R1 跳空日：券商成交股数与模拟账本一致（现场跳空日核对）",
    "R6 盘中巡检：每只持仓的行情时间逐拍前进（现场看巡检记录）",
    "停止 / 回滚操作与负责人已写明（熔断、撤销许可、停调度的命令与顺序）",
    "未覆盖事项（例如当天没出现的部分成交）已列明并接受",
)


def _item(name: str, ok: Optional[bool], detail: str, blocking: bool = True, gates=ALL3) -> dict:
    return {"name": name, "status": "pass" if ok else ("unknown" if ok is None else "fail"),
            "blocking": blocking, "gates": list(gates) if blocking else [], "detail": detail}


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=BACKEND, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"error: {exc}"


def code_items() -> List[dict]:
    sha = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    lock = BACKEND / "requirements.lock"
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None
    return [_item("固定代码提交", bool(sha) and not dirty and not sha.startswith("error"),
                  f"HEAD {sha}" + ("；工作区有未提交改动（许可要绑定固定提交）" if dirty else "")),
            _item("依赖锁", lock_sha is not None, f"requirements.lock sha256 {lock_sha}"),
            _item("Python", True, sys.version.split()[0], blocking=False),
            _item("部署路径与运行用户", True, f"{BACKEND}；{getpass.getuser()}", blocking=False)]


def _rows(con, sql, *args):
    try:
        return [dict(r) for r in con.execute(sql, args)]
    except Exception as exc:                            # 旧库缺表：如实报告
        return [{"_error": str(exc)}]


def _verified_item(rec: dict, forward: List[dict]) -> dict:
    from app.core.lifecycle.validation_design import check_registered_activation
    rows = [SimpleNamespace(date=r["date"], ic=r["ic"], evidence_class=r.get("evidence_class")) for r in forward]
    ok, d = check_registered_activation(SimpleNamespace(id=rec["id"], verdict=rec["verdict"]), rows)
    return _item("verified_active 预登记条件", ok and rec["status"] in ("paper_running", "verified_active"),
                 f"#{rec['id']} {rec['status']}；登记后的正式前向 {d.get('n_days')} 天"
                 f"（登记前 {d.get('n_before_registration')} 天不算）；" + "; ".join(d.get("reasons") or []),
                 gates=(VERIFIED,))


def db_items(db: Path, dataset: str, settings: dict, ctx: Optional[dict] = None) -> List[dict]:
    from app.core.alpha_engine.ic_metrics import ICM_VERSION
    from app.core.alpha_engine.validation_policy import check_frozen
    from app.core.portfolio_manager.executable_spec import spec_hash
    from tools.research.research_boundary_report import read_frozen
    ctx = ctx if ctx is not None else {}
    out: List[dict] = []
    with read_frozen(db) as (con, window):
        running = _rows(con, "SELECT id, status, verdict FROM strategy_configs WHERE status IN "
                             "('integration','paper_running','verified_active','active') ORDER BY id")
        boundary = _rows(con, "SELECT * FROM frozen_boundaries WHERE dataset_key = ?", dataset)
        running = [r for r in running if "_error" not in r]
        forward = (_rows(con, "SELECT date, ic, evidence_class FROM strategy_evidence WHERE strategy_id = ? "
                              "AND spec_hash = ? AND (evidence_class = 'forward' OR evidence_class IS NULL) "
                              "ORDER BY date", running[0]["id"],
                         json.loads(running[0]["verdict"] or "{}").get("spec_hash", ""))
                   if len(running) == 1 else [])
    forward = [r for r in forward if "_error" not in r]
    out.append(_item("唯一运行策略", len(running) == 1,
                     f"运行状态的策略：{[(r['id'], r['status']) for r in running]}", gates=FORMAL))
    if len(running) == 1:
        r = running[0]
        out.append(_item("运行状态", r["status"] in ("paper_running", "verified_active"),
                         f"#{r['id']} {r['status']}（integration 是工程联调、不计证据；active 是没有规格的旧字样）",
                         gates=FORMAL))
        v = json.loads(r["verdict"] or "{}")
        spec = v.get("executable_spec")
        ok_hash = isinstance(spec, dict) and spec_hash(spec) == v.get("spec_hash")
        out.append(_item("规格哈希", ok_hash, f"spec_hash {v.get('spec_hash')}", gates=FORMAL))
        if ok_hash:
            ctx["spec"] = spec
            probs = check_frozen(spec)
            out.append(_item("表达式校验政策", not probs, "; ".join(probs) or
                             f"政策 {spec['validation']['policy_version']} 核对通过", gates=FORMAL))
            out.append(_item("时钟", spec.get("clock") == "c2", f"规格时钟 {spec.get('clock')}", gates=FORMAL))
            src = (spec.get("data") or {}).get("price_source")
            out.append(_item("价格源", src == "moomoo" and settings.get("price_source") == "moomoo",
                             f"规格 {src} / 配置 {settings.get('price_source')}（审计决定 4：正式输入必须是 moomoo）",
                             gates=FORMAL))
            out.append(_item("宇宙与执行规则", True,
                             f"数据集 {(spec.get('data') or {}).get('dataset')}、宇宙 {spec.get('universe')}、"
                             f"AUM {spec.get('aum')}、执行 {spec.get('execution')}、提交 {spec.get('submission')}",
                             blocking=False))
        pe = v.get("paper_entry") or {}
        out.append(_item("研究门的真实结论", True,
                         f"等级 {pe.get('grade')}、严门通过 {pe.get('gate_passed')}、失败的门 {pe.get('gates_failed')}、"
                         f"未评估 {pe.get('gates_unevaluated')} —— paper_running 收集的是未验证证据，不改写这些结论",
                         blocking=False))
        out.append(_verified_item(r, forward))
    else:
        out.append(_item("verified_active 预登记条件", False, "没有唯一的运行策略", gates=(VERIFIED,)))
    b = [x for x in boundary if "_error" not in x]
    reg = bool(b) and bool(str(b[0].get("actor") or "").strip() and str(b[0].get("reason") or "").strip())
    out.append(_item("研究边界已登记", reg, f"{dataset}：{b[0] if b else '没有冻结边界'}"
                     + ("" if reg else "（审计决定 5/6：先登记边界，再正式重提案）"), gates=FORMAL))
    out.append(_item("IC 指标口径", True, f"{ICM_VERSION}（ic_mean 与 ic_ir 分列；未评估 = NULL / 明确状态）",
                     blocking=False))
    out.append(_item("读库方式", True, f"停写窗口副本 {window['window_opened_at']} → {window['window_closed_at']}",
                     blocking=False))
    return out


def _exec_state(exec_db: Path) -> dict:
    """执行库副本里的 exec_state（键 → 原始字符串）。"""
    from tools.research.research_boundary_report import read_frozen
    with read_frozen(exec_db) as (con, _w):
        rows = _rows(con, "SELECT key, value FROM exec_state")
    return {r["key"]: r["value"] for r in rows if "_error" not in r}


def _json(raw):
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return {"mode": "", "corrupt": True}


def _permit(exec_db: Path) -> Optional[dict]:
    return _json(_exec_state(exec_db).get("run_permit"))


def _alive(p: dict) -> bool:
    try:
        return datetime.fromisoformat(p["expires_at"]) > datetime.now(timezone.utc)
    except (KeyError, TypeError, ValueError):
        return False


def permit_items(exec_db: Optional[Path], settings: dict, head: str, acceptance_exec_db: Optional[Path],
                 ctx_fault: Optional[list] = None) -> tuple:
    """(核对项, 正式执行库里的许可)。`ctx_fault[0]` 带回执行库里的告警通道故障（通知就绪判断要用）。"""
    ctx_fault = ctx_fault if ctx_fault is not None else [None]
    out: List[dict] = []
    if acceptance_exec_db is None:
        out.append(_item("验收执行库与正式执行库分开", None, "没有给 --acceptance-exec-db", gates=ALL3))
    elif exec_db is not None and Path(acceptance_exec_db).resolve() == Path(exec_db).resolve():
        out.append(_item("验收执行库与正式执行库分开", False, "是同一个文件 —— 联调证据会混进正式前向", gates=ALL3))
    else:
        out.append(_item("验收执行库与正式执行库分开", True, f"验收 {acceptance_exec_db} / 正式 {exec_db}",
                         gates=ALL3))
        ap = _permit(Path(acceptance_exec_db)) if Path(acceptance_exec_db).exists() else None
        out.append(_item("人工联调许可（验收执行库）", True,
                         (f"{ap.get('mode')}，账户 {ap.get('account')}，到期 {ap.get('expires_at')}，"
                          f"单笔上限 {ap.get('max_order_notional')}，{'有效' if _alive(ap) else '已过期'}") if ap
                         else "还没签 —— 现场第一步：run_permit issue --mode supervised --exec-db <验收执行库> …",
                         blocking=False))
    if exec_db is None:
        out.append(_item("正式执行库的运行许可", None, "没有给 --exec-db", gates=FORMAL))
        return out, None
    state = _exec_state(Path(exec_db))
    p = _json(state.get("run_permit"))
    from app.core.execution.intraday_watchdog import blocking_reasons
    wd = blocking_reasons(SimpleNamespace(get_state=state.get))
    out.append(_item("盘中巡检最近一拍", not wd, "; ".join(wd) or
                     (f"{_json(state.get('intraday_watchdog'))}" if state.get("intraday_watchdog") else
                      "还没有巡检记录（巡检在交易时段里才跑）"), gates=(UNATTENDED,)))
    out.append(_item("告警通道故障记录", True, state.get("alert_channel_fault") or "没有", blocking=False))
    ctx_fault[0] = _json(state.get("alert_channel_fault"))
    if not p or p.get("corrupt"):
        out.append(_item("正式执行库的运行许可", False, "正式执行库里没有可读的运行许可", gates=FORMAL))
        out.append(_item("无人值守许可", False, "没有", gates=(UNATTENDED,)))
        return out, None
    live, code_ok = _alive(p), bool(head) and head.startswith(str(p.get("code_version") or "-"))
    src_ok = p.get("price_source") == settings.get("price_source")
    desc = (f"模式 {p.get('mode')}，账户 {p.get('account')}，到期 {p.get('expires_at')}，"
            f"代码 {p.get('code_version')}（HEAD {head[:12]}），价格源 {p.get('price_source')}")
    out.append(_item("正式执行库的运行许可", live and code_ok and src_ok, desc, gates=(PAPER,)))
    out.append(_item("无人值守许可", p.get("mode") == "unattended" and live and code_ok and src_ok, desc,
                     gates=(UNATTENDED,)))
    return out, p


def acceptance_items(evidence: Optional[Path], permit: Optional[dict], contract: Optional[str],
                     head: str) -> List[dict]:
    from app.core.execution import acceptance_evidence as ae
    bound = (permit or {}).get("acceptance_evidence") if (permit or {}).get("mode") == "unattended" else None
    out: List[dict] = []
    if bound:
        path = Path(bound["path"])
        if evidence is not None and Path(evidence).resolve() != path.resolve():   # 内容一样的另一份文件也不算
            out.append(_item("证据文件就是许可绑定的那一份", False,
                             f"核对单给的 {evidence}（sha256 {ae.file_sha256(evidence)}）≠ 许可绑定的 {path}"
                             f"（sha256 {bound.get('sha256')}）", gates=ALL3))
        now_sha = ae.file_sha256(path)
        out.append(_item("许可绑定的证据文件没变", now_sha == bound.get("sha256"),
                         f"{path}：签发时 sha256 {bound.get('sha256')}，现在 {now_sha}", gates=(UNATTENDED,)))
        account, code, waived = int(permit["account"]), str(permit.get("code_version")), bound.get("waived") or {}
    elif evidence is not None:
        path = Path(evidence)
        rows = ae.read_evidence(path) if path.is_file() else []
        accs = sorted({str((r.get("account") or {}).get("acc_id")) for r in rows})
        account = int(accs[0]) if len(accs) == 1 and accs[0].isdigit() else -1     # 多个 / 没有账户 → 必然不一致
        code, waived = head[:12], {}
        out.append(_item("证据文件就是许可绑定的那一份", None, "没有无人值守许可 —— 按 --evidence 判定，只用于前两个结论",
                         gates=(UNATTENDED,)))
    else:
        return [_item(f"现场验收 {s}", None, "没有证据文件（没有无人值守许可，也没给 --evidence）", gates=FORMAL)
                for s in SCENARIOS]
    j = ae.assess(path, account=account, code=code, contract=contract, waived=waived)
    if j["problems"]:
        out.append(_item("现场验收证据整体", False, "；".join(j["problems"]), gates=FORMAL))
    for s, v in j["scenarios"].items():
        st = v["status"]
        if st == ae.PASSED:
            out.append(_item(f"现场验收 {s}", True, v["detail"], gates=FORMAL))
        elif st == ae.WAIVED:
            out.append(_item(f"现场验收 {s}", None, v["detail"], blocking=False))
        elif st == ae.NOT_COVERED:                     # 人工值守的纸交易可以带着"未覆盖"开始（列为限制）；无人值守不行
            out.append(_item(f"现场验收 {s}", None, "未覆盖（不是通过）：" + v["detail"], gates=(UNATTENDED,)))
        else:
            out.append(_item(f"现场验收 {s}", False, v["detail"], gates=FORMAL))
    return out


def runtime_items(backup_dir: Optional[str], alert_events: Optional[str], settings: dict,
                  channel_fault: Optional[dict] = None) -> List[dict]:
    from app.core.alerting.dispatcher import AlertDispatcher
    from app.tasks.backup import backup_health
    out = []
    ok, why, _ = backup_health(backup_dir)
    out.append(_item("最近一次备份", ok, why, gates=FORMAL))
    if alert_events:
        ok, why = AlertDispatcher(event_file=Path(alert_events)).unattended_readiness(channel_fault=channel_fault)
        out.append(_item("通知通道", ok, why, gates=(UNATTENDED,)))
    else:
        out.append(_item("通知通道", None, "没有给 --alert-events", gates=(UNATTENDED,)))
    out += [_item("执行模式", settings.get("execution_mode") == "moomoo_paper",
                  f"EXECUTION_MODE={settings.get('execution_mode')}"),
            _item("盘中巡检", bool(settings.get("intraday_watchdog_enabled")),
                  f"INTRADAY_WATCHDOG_ENABLED={settings.get('intraday_watchdog_enabled')}", gates=(UNATTENDED,)),
            # 配置导出里没有这一项 → 未知（不当作"关着"：缺证据不是通过）
            _item("目标波动首轮关闭", (None if "risk_target_vol_ann" not in settings
                                      else not settings["risk_target_vol_ann"]),
                  f"RISK_TARGET_VOL_ANN={settings.get('risk_target_vol_ann', '（配置导出里没有）')}", gates=FORMAL)]
    return out


def build(db: Path, dataset: str, settings: dict, exec_db: Optional[Path] = None, evidence: Optional[Path] = None,
          backup_dir: Optional[str] = None, alert_events: Optional[str] = None,
          acceptance_exec_db: Optional[Path] = None) -> dict:
    from app.core.execution.acceptance_evidence import contract_of_spec
    items = code_items()
    head = next((i["detail"].split()[1] for i in items if i["name"] == "固定代码提交"), "")
    ctx: dict = {}
    items += db_items(db, dataset, settings, ctx)
    fault: list = [None]
    p_items, permit = permit_items(exec_db, settings, head, acceptance_exec_db, fault)
    items += p_items
    items += acceptance_items(evidence, permit, contract_of_spec(ctx.get("spec")), head)
    items += runtime_items(backup_dir, alert_events, settings, fault[0])
    conclusions = {}
    for key, title in CONCLUSIONS.items():
        blockers = [i for i in items if i["blocking"] and key in i["gates"] and i["status"] != "pass"]
        if key == VERIFIED:
            verdict = "满足" if not blockers else "不满足"
        else:
            verdict = "No-Go" if blockers else "可提交你做最后决定（工具不给 Go）"
        conclusions[key] = {"question": title, "verdict": verdict,
                            "blockers": [f"{i['name']}：{i['detail']}" for i in blockers]}
    if conclusions[UNATTENDED]["verdict"] != "No-Go":
        conclusions[UNATTENDED]["verdict"] += "；人工确认项全部由你填写之后"
    limits = [f"{i['name']}：{i['detail']}" for i in items if i["status"] != "pass" and not i["blocking"]]
    return {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "dataset": dataset, "code": head,
            "conclusions": conclusions,
            # 兼容旧字段：verdict / blockers = 无人值守那一个结论
            "verdict": conclusions[UNATTENDED]["verdict"], "blockers": conclusions[UNATTENDED]["blockers"],
            "limitations": limits, "items": items,
            "manual_confirmations": [{"item": m, "confirmed_by": None, "confirmed_at": None} for m in MANUAL],
            "note": "人工确认项由你在现场填写；工具不替你打勾。启动由你决定。"}


def render(rep: dict) -> str:
    mark = {"pass": "OK  ", "fail": "FAIL", "unknown": "??  "}
    lines = [f"Go/No-Go 核对（{rep['at']}，{rep['dataset']}，HEAD {rep['code']}）"]
    for c in rep["conclusions"].values():
        lines.append(f"  {c['question']}：{c['verdict']}")
    lines.append("")
    lines += [f"  [{mark[i['status']]}] {i['name']}（挡：{'/'.join(i['gates']) or '-'}）：{i['detail']}"
              for i in rep["items"]]
    lines += ["", "必须你现场确认："] + [f"  [  ] {m['item']}" for m in rep["manual_confirmations"]]
    for c in rep["conclusions"].values():
        if c["blockers"]:
            lines += ["", f"{c['question']} —— 阻断 {len(c['blockers'])} 项："] + [f"  - {b}" for b in c["blockers"]]
    if rep["limitations"]:
        lines += ["", "可接受的限制（不阻断，但要你知道）："] + [f"  - {x}" for x in rep["limitations"]]
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="启动前的 Go/No-Go 核对单（只读，不代替你启动）")
    ap.add_argument("--db", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--exec-db", help="正式执行库")
    ap.add_argument("--acceptance-exec-db", help="现场验收专用的执行库")
    ap.add_argument("--evidence", help="现场验收证据；有无人值守许可时必须就是许可绑定的那一份")
    ap.add_argument("--backup-dir")
    ap.add_argument("--alert-events")
    ap.add_argument("--settings-json", help="去掉密钥的配置导出（price_source / execution_mode / …）；不读 .env")
    ap.add_argument("--out")
    a = ap.parse_args(list(argv) if argv is not None else None)
    settings = json.loads(Path(a.settings_json).read_text(encoding="utf-8")) if a.settings_json else {}
    rep = build(Path(a.db), a.dataset, settings, Path(a.exec_db) if a.exec_db else None,
                Path(a.evidence) if a.evidence else None, a.backup_dir, a.alert_events,
                Path(a.acceptance_exec_db) if a.acceptance_exec_db else None)
    print(render(rep))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0 if not rep["blockers"] else 2


if __name__ == "__main__":
    sys.exit(main())
