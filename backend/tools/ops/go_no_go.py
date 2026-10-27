"""
go_no_go.py — 无人值守前向的 Go/No-Go 核对单（2026-10-25 审计"工作包 D：生成最终 Go/No-Go，不代替用户启动"）

    python tools/ops/go_no_go.py --db C:\\QuantAgentData\\alphas.db --exec-db <执行库> \\
        --evidence <simulate_acceptance.jsonl> --backup-dir <备份目录> --alert-events <事件文件> \\
        --dataset us_broad_large [--out go_no_go.json]

只读：库按停写窗口复制后读副本（`research_boundary_report.read_frozen`，不对活库建连接、不建表）；其余都是读文件。
不读 .env：配置项要显式给，或从 `--settings-json`（你导出的、去掉密钥的那份）读。

审计列出的 Go/No-Go 内容逐项对应：
    代码与环境      固定提交（不能 +dirty）、依赖锁哈希、Python 版本
    账户与资金      执行模式、专用 SIMULATE 账户（许可里写明的那个）、资金切片
    策略与规格      唯一运行策略、状态（paper_running / verified_active）、规格哈希、校验政策核对、时钟 c2、价格源
    研究边界        数据集的冻结边界是否**登记**（provisional / unconfirmed 不算）
    运行许可        无人值守许可是否在、此刻核对是否通过
    现场验收        八个场景最近一次 verify 的结论（没有 = 未覆盖，不是通过）
    运行保障        通知通道就绪、最近一次备份成功
    人工确认        工具看不到、必须你在现场确认的项（列出来，不替你打勾）

结论只有两种：No-Go（列出阻断项）或"可提交你做最后决定"。**工具永远不给 Go** —— 启动由你决定。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BACKEND = Path(__file__).resolve().parents[2]

#: 审计要求的现场验收场景（与 simulate_acceptance.SCENARIOS 一致，由测试核对）
SCENARIOS = ("submit_and_fill", "day_order_across_sessions", "partial_fill_then_cancel", "unknown_order",
             "restart_dedup", "kill_switch_flatten", "input_anomaly", "crash_recovery")

#: 工具看不到、必须现场确认的项（审计决定 8 / 9 与工作包 C）
MANUAL = (
    "桌面通知在实际启动方式、同一 Windows 用户、锁屏 / 会话断开、重启后都能看到，并确认了准确的事件 id",
    "专用 SIMULATE 账户的初始持仓 / 现金 / 出入金基线已核对，账户里没有不明现金流",
    "OpenD / SDK 版本、部署路径、运行用户已记录",
    "盘后提交的 DAY 单在实际账户上 T+1 有效（day_order_across_sessions 现场通过）",
    "R1 跳空日：券商成交股数与模拟账本一致（现场跳空日核对）",
    "停止 / 回滚操作与负责人已写明（熔断、撤销许可、停调度的命令与顺序）",
    "未覆盖事项（例如当天没出现的部分成交）已列明并接受",
)


def _item(name: str, ok: Optional[bool], detail: str, blocking: bool = True) -> dict:
    return {"name": name, "status": "pass" if ok else ("unknown" if ok is None else "fail"),
            "blocking": blocking, "detail": detail}


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
                  f"HEAD {sha}" + ("；工作区有未提交改动（无人值守许可要绑定固定提交）" if dirty else "")),
            _item("依赖锁", lock_sha is not None, f"requirements.lock sha256 {lock_sha}"),
            _item("Python", True, sys.version.split()[0], blocking=False)]


def _rows(con, sql, *args):
    try:
        return [dict(r) for r in con.execute(sql, args)]
    except Exception as exc:                            # 旧库缺表：如实报告
        return [{"_error": str(exc)}]


def db_items(db: Path, dataset: str, settings: dict) -> List[dict]:
    from app.core.alpha_engine.validation_policy import check_frozen
    from app.core.portfolio_manager.executable_spec import spec_hash
    from tools.research.research_boundary_report import read_frozen
    out: List[dict] = []
    with read_frozen(db) as (con, window):
        running = _rows(con, "SELECT id, status, verdict FROM strategy_configs WHERE status IN "
                             "('integration','paper_running','verified_active','active') ORDER BY id")
        boundary = _rows(con, "SELECT * FROM frozen_boundaries WHERE dataset_key = ?", dataset)
    running = [r for r in running if "_error" not in r]
    out.append(_item("唯一运行策略", len(running) == 1,
                     f"运行状态的策略：{[(r['id'], r['status']) for r in running]}"))
    if len(running) == 1:
        r = running[0]
        out.append(_item("运行状态", r["status"] in ("paper_running", "verified_active"),
                         f"#{r['id']} {r['status']}（integration 是工程联调，active 是没有规格的旧字样）"))
        v = json.loads(r["verdict"] or "{}")
        spec = v.get("executable_spec")
        ok_hash = isinstance(spec, dict) and spec_hash(spec) == v.get("spec_hash")
        out.append(_item("规格哈希", ok_hash, f"spec_hash {str(v.get('spec_hash'))[:16]}"))
        if ok_hash:
            probs = check_frozen(spec)
            out.append(_item("表达式校验政策", not probs, "; ".join(probs) or
                             f"政策 {spec['validation']['policy_version']} 核对通过"))
            out.append(_item("时钟", spec.get("clock") == "c2", f"规格时钟 {spec.get('clock')}"))
            src = (spec.get("data") or {}).get("price_source")
            out.append(_item("价格源", src == "moomoo" and settings.get("price_source") == "moomoo",
                             f"规格 {src} / 配置 {settings.get('price_source')}（审计决定 4：正式输入必须是 moomoo）"))
    b = [x for x in boundary if "_error" not in x]
    reg = bool(b) and bool(str(b[0].get("actor") or "").strip() and str(b[0].get("reason") or "").strip())
    out.append(_item("研究边界已登记", reg, f"{dataset}：{b[0] if b else '没有冻结边界'}"
                     + ("" if reg else "（审计决定 5/6：先登记边界，再正式重提案）")))
    out.append(_item("读库方式", True, f"停写窗口副本 {window['window_opened_at']} → {window['window_closed_at']}",
                     blocking=False))
    return out


def permit_items(exec_db: Optional[Path], settings: dict) -> List[dict]:
    if exec_db is None:
        return [_item("运行许可", None, "没有给 --exec-db")]
    from tools.research.research_boundary_report import read_frozen
    with read_frozen(exec_db) as (con, _w):
        row = _rows(con, "SELECT value FROM exec_state WHERE key = 'run_permit'")
    raw = (row[0].get("value") if row and "_error" not in row[0] else None) or ""
    if not raw:
        return [_item("无人值守许可", False, "执行库里没有运行许可")]
    p = json.loads(raw)
    exp = datetime.fromisoformat(p["expires_at"])
    live = exp > datetime.now(timezone.utc)
    return [_item("无人值守许可", p.get("mode") == "unattended" and live,
                  f"模式 {p.get('mode')}，账户 {p.get('account')}，到期 {p['expires_at']}，代码 {p.get('code_version')}"),
            _item("许可绑定的价格源", p.get("price_source") == settings.get("price_source"),
                  f"许可 {p.get('price_source')} / 配置 {settings.get('price_source')}")]


def acceptance_items(evidence: Optional[Path]) -> List[dict]:
    if evidence is None or not Path(evidence).exists():
        return [_item(f"现场验收 {s}", None, "没有证据文件") for s in SCENARIOS]
    rows = [json.loads(line) for line in Path(evidence).read_text(encoding="utf-8").splitlines() if line.strip()]
    out = []
    for s in SCENARIOS:
        v = [r for r in rows if r.get("scenario") == s and r.get("step") == "verdict"]
        if not v:
            out.append(_item(f"现场验收 {s}", None, "没有 verify 结论 = 未覆盖（不是通过）"))
            continue
        last = v[-1]
        fails = [c["name"] for c in last["data"]["checks"] if not c["ok"]]
        out.append(_item(f"现场验收 {s}", bool(last["data"]["passed"]),
                         f"{last['at']} 代码 {last.get('code_version')}" + (f"；未过：{fails}" if fails else "")))
    return out


def runtime_items(backup_dir: Optional[str], alert_events: Optional[str]) -> List[dict]:
    from app.core.alerting.dispatcher import AlertDispatcher
    from app.tasks.backup import backup_health
    out = []
    ok, why, _ = backup_health(backup_dir)
    out.append(_item("最近一次备份", ok, why))
    if alert_events:
        ok, why = AlertDispatcher(event_file=Path(alert_events)).unattended_readiness()
        out.append(_item("通知通道", ok, why))
    else:
        out.append(_item("通知通道", None, "没有给 --alert-events"))
    return out


def build(db: Path, dataset: str, settings: dict, exec_db: Optional[Path] = None, evidence: Optional[Path] = None,
          backup_dir: Optional[str] = None, alert_events: Optional[str] = None) -> dict:
    items = (code_items() + db_items(db, dataset, settings) + permit_items(exec_db, settings)
             + acceptance_items(evidence) + runtime_items(backup_dir, alert_events)
             + [_item("执行模式", settings.get("execution_mode") == "moomoo_paper",
                      f"EXECUTION_MODE={settings.get('execution_mode')}"),
                _item("盘中巡检", bool(settings.get("intraday_watchdog_enabled")),
                      f"INTRADAY_WATCHDOG_ENABLED={settings.get('intraday_watchdog_enabled')}"),
                _item("目标波动首轮关闭", not settings.get("risk_target_vol_ann"),
                      f"RISK_TARGET_VOL_ANN={settings.get('risk_target_vol_ann')}")])
    blockers = [i for i in items if i["blocking"] and i["status"] != "pass"]
    return {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "dataset": dataset,
            "verdict": ("No-Go" if blockers else "可提交你做最后决定（工具不给 Go）"),
            "blockers": [f"{i['name']}：{i['detail']}" for i in blockers], "items": items,
            "manual_confirmations": [{"item": m, "confirmed_by": None, "confirmed_at": None} for m in MANUAL],
            "note": "人工确认项由你在现场填写；工具不替你打勾。启动由你决定。"}


def render(rep: dict) -> str:
    mark = {"pass": "OK  ", "fail": "FAIL", "unknown": "??  "}
    lines = [f"Go/No-Go 核对（{rep['at']}，{rep['dataset']}）：{rep['verdict']}"]
    lines += [f"  [{mark[i['status']]}] {i['name']}：{i['detail']}" for i in rep["items"]]
    lines += ["", "必须你现场确认："] + [f"  [  ] {m['item']}" for m in rep["manual_confirmations"]]
    if rep["blockers"]:
        lines += ["", f"阻断项 {len(rep['blockers'])} 个："] + [f"  - {b}" for b in rep["blockers"]]
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="无人值守前向的 Go/No-Go 核对单（只读，不代替你启动）")
    ap.add_argument("--db", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--exec-db")
    ap.add_argument("--evidence")
    ap.add_argument("--backup-dir")
    ap.add_argument("--alert-events")
    ap.add_argument("--settings-json", help="去掉密钥的配置导出（price_source / execution_mode / …）；不读 .env")
    ap.add_argument("--out")
    a = ap.parse_args(list(argv) if argv is not None else None)
    settings = json.loads(Path(a.settings_json).read_text(encoding="utf-8")) if a.settings_json else {}
    rep = build(Path(a.db), a.dataset, settings, Path(a.exec_db) if a.exec_db else None,
                Path(a.evidence) if a.evidence else None, a.backup_dir, a.alert_events)
    print(render(rep))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0 if not rep["blockers"] else 2


if __name__ == "__main__":
    sys.exit(main())
