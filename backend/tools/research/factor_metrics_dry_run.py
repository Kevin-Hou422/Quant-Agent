"""
factor_metrics_dry_run.py — 活库因子指标的备份 + 恢复验证 + 版本化 dry-run（2026-10-22 用户决定 D2）

用户原话："先备份、验证恢复并做版本化 dry-run。展示实际受影响记录、旧新值、算法/数据/时钟版本及排序影响。
保留旧值和曝光史。展示差异后，我另行确认写活库；不要直接覆盖。"

    python tools/research/factor_metrics_dry_run.py snapshot --db C:\\QuantAgentData\\alphas.db --out <新的空目录>
    python tools/research/factor_metrics_dry_run.py inventory --copy <out>/restored/alphas.db
    python tools/research/factor_metrics_dry_run.py dry-run  --copy <out>/restored/alphas.db --dataset us_tech_large [--start … --end …]

snapshot   活库不建连接：在**验证过的停写窗口**里逐字节复制主库到 `<out>/frozen`（窗口两端主库、-wal、-shm、-journal 的
           哈希 / 大小 / 修改时间必须完全相同，副本哈希等于窗口内主库哈希；否则副本作废、报错 —— 2026-10-25 审计 R2），
           之后只在这份冻结副本上 immutable 读：`VACUUM INTO` 出快照、恢复到 `<out>/restored`，逐表比行数、主键、内容摘要
           与业务不变量（`app.tasks.snapshot_manifest`），清单写进 `preview.json`。旁边有未合并的 -wal → 拒绝
           （先停掉用库的进程；不要手动删除 -wal / -shm）。输出目录必须是新的空目录、不在活库目录里
inventory  逐条列出：是哪条路径写的、`ic_ir` 这一列存的到底是什么、算没算过
writeback-plan  写回活库之前的完整预演（2026-10-30 用户决定 4："列出每条记录旧值、新值、来源、样本、算法和时钟版本；
           旧算法纠错与使用 moomoo/c2/新边界重新评估的不同修订理由；受影响的排序、准入和历史结论；修订记录、回滚步骤和
           恢复验证。完成差异后交我确认；不要直接写活库"）。只在副本的拷贝上执行，见 `writeback_plan`
dry-run    只在快照副本上（再复制一份工作库，配置指向它 —— 研究上下文构造 Store 会建表，恢复副本保持原样）：
           按**当前代码**在研究选择段上重算（不碰 Test；边界没登记时是临时切分，报告里写明），
           列出旧值 / 新值（IC 均值与 IC-IR 都算）/ 排序变化 / 算法·数据·时钟版本 / 曝光史。**不写任何库**；
           写回活库是另一步，要你看过差异后另行确认（本工具不提供）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PREVIEW_MANIFEST = "preview.json"

#: 哪条路径写的这条记录 → `ic_ir` 列里存的是什么（2026-10-24 查明：多数写入路径存的是样本内 IC 均值）
WRITERS = [
    ("auto-discovery", "discovery", "样本内 IC 均值（is_ic）"),
    ("[baseline:", "baseline_seed", "从未评估（0）"),
]


from app.tasks.snapshot_manifest import SourceBusy as LiveDbBusy  # noqa: E402  活库旁边有未合并的 -wal
from app.tasks.snapshot_manifest import SourceChanged  # noqa: E402,F401  停写窗口内活库变了


class PreviewRefused(RuntimeError):
    """输出位置 / 输入库不符合只读预览的约束。"""


def _ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro&immutable=1", uri=True)


def fingerprint(db: Path) -> dict:
    db = Path(db)
    return {"sha256": hashlib.sha256(db.read_bytes()).hexdigest(), "size": db.stat().st_size,
            "mtime_ns": db.stat().st_mtime_ns, "directory": sorted(p.name for p in db.parent.iterdir())}


def _inside(a: Path, b: Path) -> bool:
    a, b = Path(a).resolve(), Path(b).resolve()
    return a == b or b in a.parents


def snapshot_live(db: Path, out: Path) -> dict:
    from app.tasks.restore import restore_snapshot, verify_restore
    from app.tasks.snapshot_manifest import copy_in_quiet_window, describe_db
    db, out = Path(db), Path(out)
    if _inside(out, db.parent):
        raise PreviewRefused(f"输出目录 {out} 在活库所在目录 {db.parent} 里 —— 放到别处（例如 C:\\QuantAgentEnv\\…）")
    if out.exists() and any(out.iterdir()):
        raise PreviewRefused(f"输出目录 {out} 不是空的")
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "frozen" / db.name
    try:
        window = copy_in_quiet_window(db, frozen)
    except Exception:
        shutil.rmtree(out / "frozen", ignore_errors=True)
        raise
    snap = out / "snapshot"
    snap.mkdir(parents=True)
    dest = snap / db.name
    con = _ro(frozen)                                         # 冻结副本：immutable 只用在这里
    try:
        con.execute("VACUUM INTO ?", (str(dest),))
    finally:
        con.close()
    restored = out / "restored"
    rs = restore_snapshot(snap, restored)
    ver = verify_restore(frozen, restored / db.name)
    rep = {"at": datetime.now(timezone.utc).isoformat(), "live": str(db), "frozen_copy": str(frozen),
           "snapshot": str(dest), "copy": str(restored / db.name),
           "window": window, "manifest": describe_db(frozen),
           "restore": {**ver, "integrity_after_restore": rs["integrity"]},
           "live_before": window["before"], "live_after": window["after"],
           "live_unchanged": window["before"] == window["after"]}
    (out / PREVIEW_MANIFEST).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


def _revision_reasons(r: dict) -> List[str]:
    """
    审计决定 3："#1/#2 的旧算法纠错，与使用 moomoo/c2/新边界重新评估，是不同修订理由，不能合并成'只修两条错误值'。"
    逐条写出这条记录若要改，各是哪种理由（写回时分开记进修订链）。
    """
    out = []
    if r.get("writer") == "discovery":
        out.append("列语义纠错：旧 ic_ir 列存的是样本内 IC 均值 → 移到 ic_mean，ic_ir 按统一口径重算或置空")
    if r.get("evaluated_before") is False:
        out.append("未评估：旧值 0 不是评估结果 → 标 never_evaluated，不补造旧验证历史")
    out.append("新版本重评估：当前代码 / 时钟 c2 / 研究选择段 / 本次价格源（与纠错是不同的修订理由）")
    return out


def _writer(hypothesis: str, reasoning: str) -> tuple:
    text = f"{hypothesis or ''} {reasoning or ''}"
    for marker, name, meaning in WRITERS:
        if marker in text:
            return name, meaning
    return "unknown", "未知（写入路径认不出 —— 可能是 IC 均值，也可能是 IC-IR）"


def inventory(copy: Path) -> List[dict]:
    con = _ro(copy)
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(alpha_records)")]
        want = [c for c in ("id", "dsl", "status", "created_at", "hypothesis", "reasoning", "sharpe", "ann_return",
                            "ic_ir", "ic_mean", "ic_metrics", "ann_turnover", "clock_version") if c in cols]
        rows = [dict(zip(want, r)) for r in con.execute(f"SELECT {', '.join(want)} FROM alpha_records ORDER BY id")]
    finally:
        con.close()
    from types import SimpleNamespace

    from app.core.alpha_engine.ic_metrics import EVALUATED, LEGACY_UNKNOWN, status_of
    out = []
    for r in rows:
        writer, meaning = _writer(r.get("hypothesis"), r.get("reasoning"))
        stored = {k: r.get(k) for k in ("sharpe", "ann_return", "ic_ir", "ic_mean", "ann_turnover")}
        st = status_of(SimpleNamespace(ic_metrics=r.get("ic_metrics")))
        # 审计决定 2："不再靠所有指标是否为 0 推断 evaluated"：有 ic_metrics 就照它；旧行只有一种写入路径能确定
        # （基准种子入库时没有任何评估），其余记 None = 不知道
        if st["status"] != LEGACY_UNKNOWN:
            evaluated = st["status"] == EVALUATED
        else:
            evaluated = False if writer == "baseline_seed" else None
        out.append({"id": r["id"], "dsl": r["dsl"], "status": r["status"], "created_at": str(r.get("created_at")),
                    "clock_version": r.get("clock_version") or "c1（列不存在或为空 = 2026-10-20 之前）",
                    "writer": writer, "ic_ir_column_holds": meaning, "ic_status": st["status"],
                    "evaluated": evaluated, "stored": stored})
    return out


def _exposure(copy: Path) -> dict:
    con = _ro(copy)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        out = {}
        for t in ("holdout_usages", "trial_ledger", "trial_runs", "frozen_boundaries", "alpha_decisions"):
            out[t] = int(con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]) if t in tables else None
        return out
    finally:
        con.close()


@contextmanager
def _settings_pointing_at(db: Path):
    from app.config import settings
    old = settings.database_url
    settings.database_url = f"sqlite:///{Path(db).as_posix()}"
    try:
        yield
    finally:
        settings.database_url = old


def _ranks(values: Dict[int, float]) -> Dict[int, int]:
    order = sorted(values, key=lambda k: (-(values[k] if values[k] == values[k] else float("-inf")), k))
    return {k: i + 1 for i, k in enumerate(order)}


def dry_run(copy: Path, dataset: str, start: Optional[str], end: Optional[str],
            loader: Optional[Callable] = None, report_dir: Optional[Path] = None) -> dict:
    import numpy as np

    from app.config import settings
    from app.core.alpha_engine.signal_processor import SimulationConfig
    from app.core.backtest_engine.realistic_backtester import RealisticBacktester
    from app.core.data_engine.input_identity import (
        clock_version,
        code_version,
        identify,
    )
    from app.core.data_engine.research_context import open_research
    from app.tasks.backup import _sqlite_path_from_url
    copy = Path(copy)
    live = _sqlite_path_from_url(getattr(settings, "database_url", "") or "")
    manifest = copy.parent.parent / PREVIEW_MANIFEST
    if (live is not None and live == copy.resolve()) or not manifest.exists():
        raise PreviewRefused(f"{copy} 不是 snapshot 产出的副本（找不到 {manifest.name}，或它就是配置里的活库）—— dry-run 只在副本上做")
    inv = inventory(copy)
    work = copy.parent.parent / "work" / copy.name
    work.parent.mkdir(exist_ok=True)
    shutil.copy2(copy, work)
    if loader is None:
        from app.core.data_engine.dataset_registry import (
            load_registry_dataset as loader,
        )
    from app.core.alpha_engine.ic_metrics import ic_fields
    with _settings_pointing_at(work):
        panel = loader(dataset, start=start, end=end).data
        ctx = open_research(panel, dataset, "factor_metrics_dry_run")
        sel = ctx.selection
        bt = RealisticBacktester(config=SimulationConfig(), min_obs=0)
        for r in inv:
            try:
                rep = bt.run(r["dsl"], sel).is_report
                f = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)   # noqa: E731
                r["recomputed"] = {"mean_ic": f(rep.mean_ic), "ic_ir": f(rep.ic_ir), "sharpe": f(rep.sharpe_ratio),
                                   "ann_turnover": f(rep.ann_turnover), "n_obs": int(rep.ic_n_obs)}
                # 按统一口径（审计决定 2）将要写成什么 —— 只预演，不写
                r["proposed_fields"] = ic_fields(rep.mean_ic, rep.ic_ir, rep.ic_n_obs,
                                                 source="factor_metrics_dry_run", segment="selection",
                                                 window=[str(ctx.selection_start), str(ctx.selection_end)])
            except Exception as exc:
                r["recomputed"] = {"mean_ic": None, "ic_ir": None, "sharpe": None, "ann_turnover": None,
                                   "error": f"{type(exc).__name__}: {exc}"}
    nan = float("nan")
    old_rank = _ranks({r["id"]: float(r["stored"]["ic_ir"] or 0.0) for r in inv})
    new_rank = _ranks({r["id"]: (r["recomputed"]["mean_ic"] if r["recomputed"]["mean_ic"] is not None else nan)
                       for r in inv})
    for r in inv:
        r["evaluated_before"] = r.pop("evaluated")
        r["revision_reasons"] = _revision_reasons(r)
        r["rank_by_stored"] = old_rank[r["id"]]
        r["rank_by_recomputed_mean_ic"] = new_rank[r["id"]]
    ident = identify(sel)
    out = {
        "at": datetime.now(timezone.utc).isoformat(),
        "copy": str(copy), "writes": "none",
        "versions": {"code_version": code_version(), "clock_version": clock_version(),
                     "algorithm": "RealisticBacktester.run(dsl, selection).is_report（mean_ic / ic_ir / sharpe）",
                     "dataset": dataset, "window": {"start": start, "end": end},
                     "segment": "selection", "selection": [ctx.selection_start, ctx.selection_end],
                     "boundary": ctx.boundary or "none", "test_start": ctx.test_start,
                     "data": {"data_digest": ident.data_digest, "last_day": ident.last_day,
                              "universe_digest": ident.universe_digest,
                              "price_source": str(getattr(settings, "price_source", ""))}},
        "records": inv,
        "exposure": _exposure(copy),
        "note": ("旧值原样保留（本工具不写库）。旧行的 `ic_ir` 列在多数写入路径里存的是样本内 IC 均值；按审计决定 2，"
                 "新值分成 ic_mean 与 ic_ir 两列（proposed_fields），每条的修订理由分开列（revision_reasons）。"
                 "写回活库要你看过差异后另行确认。"),
    }
    report_dir = Path(report_dir or copy.parent.parent)
    path = report_dir / "factor_metrics_dry_run.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    out["report_file"] = str(path)
    return out


#: 写回的两种修订理由（写进 alpha_decisions.decision，≤ 16 字）
FIX, REEVAL = "metric_fix", "metric_reeval"
IC_COLS = ("ic_ir", "ic_mean", "ic_metrics")


def _changes_fix(inv: List[dict]) -> tuple:
    """
    A 纠错：不依赖价格源 / 边界 / 重算，只把旧行的列语义说清楚（审计决定 2、3）。
        discovery 写的旧行：ic_ir 列存的是样本内 IC 均值 → ic_mean = 旧值，ic_ir = NULL，状态 values_without_provenance
        基准种子：从未评估 → ic_mean / ic_ir = NULL，状态 never_evaluated（旧值 0 不是评估结果）
        已有 ic_metrics 的行不动；认不出写入路径的不动（列出来，要你判断）
    """
    from app.core.alpha_engine import ic_metrics as icm
    changes, unresolved = [], []
    for r in inv:
        if r["ic_status"] != icm.LEGACY_UNKNOWN:
            continue
        old = {"ic_ir": r["stored"].get("ic_ir"), "ic_mean": r["stored"].get("ic_mean"), "ic_metrics": None}
        if r["writer"] == "discovery":
            new = {"ic_mean": old["ic_ir"], "ic_ir": None, **icm.no_provenance("discovery（旧行，列语义纠错）")}
            why = "列语义纠错：旧 ic_ir 列存的是样本内 IC 均值 → 移到 ic_mean；ic_ir 置空（这条记录没有算过真正的 IC-IR）"
        elif r["writer"] == "baseline_seed":
            new = icm.never_evaluated("baseline_seed（旧行）")
            why = "未评估：旧值 0 不是评估结果 → never_evaluated；不补造旧的验证历史"
        else:
            unresolved.append({"id": r["id"], "dsl": r["dsl"], "stored": r["stored"],
                               "why": "写入路径认不出：ic_ir 列可能是 IC 均值也可能是 IC-IR —— 不改，要你判断"})
            continue
        changes.append({"id": r["id"], "dsl": r["dsl"], "status": r["status"], "reason_kind": FIX, "reason": why,
                        "old": old, "new": new})
    return changes, unresolved


def _changes_reeval(report: Optional[dict]) -> dict:
    """
    B 重评估：dry-run 按当前代码 / 时钟 / 研究选择段 / 当次价格源算出的新值。用户决定 5："最终权威重算和重提案使用确认后的
    moomoo 数据；Yahoo 结果仅用于工程预演"—— 价格源不是 moomoo、或边界没有登记 → 只列出（preview_only），不进可执行变更集。
    """
    if not report:
        return {"executable": False, "why": "没有给 --reevaluation（dry-run 报告）", "changes": []}
    v = report["versions"]
    src, boundary = v["data"].get("price_source"), v.get("boundary")
    blockers = ([] if src == "moomoo" else [f"价格源是 {src!r}，不是 moomoo（只算工程预演）"]) +                ([] if boundary == "registered" else [f"研究边界是 {boundary!r}，不是已登记的"])
    changes = []
    for r in report["records"]:
        pf = r.get("proposed_fields")
        if not pf:
            continue
        meta = json.loads(pf["ic_metrics"])
        changes.append({"id": r["id"], "reason_kind": REEVAL,
                        "reason": "新版本重评估：当前代码 / 时钟 / 研究选择段 / 价格源（与纠错是不同的修订理由）",
                        "new": {"ic_mean": pf["ic_mean"], "ic_ir": pf["ic_ir"]},
                        "source": meta.get("source"), "n_obs": meta.get("n_obs"), "window": meta.get("window"),
                        "algorithm": v.get("algorithm"), "ic_version": meta.get("version"), "clock": meta.get("clock"),
                        "code_version": v.get("code_version"), "price_source": src, "boundary": boundary})
    return {"executable": not blockers, "why": "; ".join(blockers) or "可执行", "changes": changes}


def _affected(copy: Path, fix: List[dict], inv: List[dict]) -> dict:
    """受影响的排序（按 ic_ir、按 ic_mean）、准入（状态与审批谱系）与引用这些因子的策略的历史结论（不改写）。"""
    changed = {c["id"]: c for c in fix}
    before = {r["id"]: r["stored"].get("ic_ir") for r in inv}
    after = {i: (changed[i]["new"]["ic_ir"] if i in changed else v) for i, v in before.items()}
    mean_after = {r["id"]: (changed[r["id"]]["new"]["ic_mean"] if r["id"] in changed else r["stored"].get("ic_mean"))
                  for r in inv}
    nan = float("nan")
    rank = lambda d: _ranks({k: (v if v is not None else nan) for k, v in d.items()})   # noqa: E731
    rb, ra, rm = rank(before), rank(after), rank(mean_after)
    con = _ro(copy)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        decisions = ({i: [dict(zip(("decision", "from_status", "to_status", "reason", "actor", "decided_at"), row))
                          for row in con.execute("SELECT decision, from_status, to_status, reason, actor, decided_at "
                                                 "FROM alpha_decisions WHERE alpha_id = ? ORDER BY id", (i,))]
                      for i in changed} if "alpha_decisions" in tables else {})
        strategies = ([dict(zip(("id", "status", "factors", "verdict"), row))
                       for row in con.execute("SELECT id, status, factors, verdict FROM strategy_configs ORDER BY id")]
                      if "strategy_configs" in tables else [])
    finally:
        con.close()
    using = []
    for st in strategies:
        ids = {int(x) for x in json.loads(st["factors"] or "[]") if str(x).isdigit()}
        hit = sorted(ids & set(changed))
        if hit:
            pe = (json.loads(st["verdict"] or "{}").get("paper_entry") or {})
            using.append({"strategy": st["id"], "status": st["status"], "factors_changed": hit,
                          "paper_entry_grade": pe.get("grade"), "gates_failed": pe.get("gates_failed"),
                          "note": "策略的研究门结论是策略级回测算出的，不读因子的 ic_ir 列 —— 纠错不改写这些结论"})
    return {"ranks": {i: {"by_ic_ir_before": rb[i], "by_ic_ir_after": ra[i], "by_ic_mean_after": rm[i]} for i in rb},
            "admission": {i: {"status": changed[i]["status"], "decisions": decisions.get(i, []),
                              "status_changes": "无（写回只改三列指标，不动状态）"} for i in changed},
            "strategies": using}


def _apply_fix(db: Path, fix: List[dict], actor: str, at: str) -> List[int]:
    """在给定的库上执行 A（一个事务）：补列（旧库缺 ic_mean / ic_metrics）→ 逐行更新（带旧值守卫）→ 逐行写修订记录。"""
    con = sqlite3.connect(db)
    try:
        con.execute("BEGIN IMMEDIATE")
        cols = {r[1] for r in con.execute("PRAGMA table_info(alpha_records)")}
        if "ic_mean" not in cols:
            con.execute("ALTER TABLE alpha_records ADD COLUMN ic_mean FLOAT")
        if "ic_metrics" not in cols:
            con.execute("ALTER TABLE alpha_records ADD COLUMN ic_metrics TEXT")
        ids = []
        for c in fix:
            cur = con.execute("UPDATE alpha_records SET ic_mean = ?, ic_ir = ?, ic_metrics = ? "
                              "WHERE id = ? AND ic_ir IS ? AND ic_metrics IS NULL",
                              (c["new"]["ic_mean"], c["new"]["ic_ir"], c["new"]["ic_metrics"], c["id"], c["old"]["ic_ir"]))
            if cur.rowcount != 1:
                raise RuntimeError(f"#{c['id']} 的旧值与预演时不同（或已被改过）—— 整个事务回滚，什么都不写")
            reason = json.dumps({"kind": c["reason_kind"], "why": c["reason"], "old": c["old"], "new": c["new"]},
                                ensure_ascii=False)
            ids.append(con.execute("INSERT INTO alpha_decisions (alpha_id, decision, from_status, to_status, reason, "
                                   "actor, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                   (c["id"], c["reason_kind"], c["status"], c["status"], reason, actor, at)).lastrowid)
        con.commit()
        return ids
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def _rows(db: Path, table: str, cols) -> Dict[int, dict]:
    """{id: {列: 值}}（表里没有的列不取）。"""
    con = _ro(db)
    try:
        have = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}
        sel = [c for c in cols if c in have]
        return {r[0]: dict(zip(sel, r[1:])) for r in con.execute(
            f'SELECT id, {", ".join(sel)} FROM "{table}" ORDER BY id')}
    finally:
        con.close()


def writeback_plan(copy: Path, out: Path, reevaluation: Optional[Path] = None, actor: str = "<写回时由你填写>") -> dict:
    """
    写回活库之前的完整预演 —— **不写活库**，只在 `copy`（停写窗口快照恢复出的副本）的拷贝上：

        0 迁移        旧库缺 ic_mean / ic_metrics 两列 → ALTER TABLE 补上（新列为空，旧值一个不动）
        A 纠错        `_changes_fix`；每条一行 alpha_decisions（decision = metric_fix，from = to = 原状态，
                      reason = 旧值 / 新值 / 理由的 JSON），与更新在同一个事务里；任何一行旧值对不上 → 整个事务回滚
        B 重评估      `_changes_reeval`；价格源不是 moomoo 或边界没登记 → 只列出、不执行
        核对          只有目标行的三列变了；其余表的内容摘要不变；alpha_decisions 恰好多了 A 的条数
        回滚          ① 逻辑回滚：逐行恢复旧值、删除本次写入的修订行 → 原有列与修订表回到写回前；
                      ② 整库回滚：把写回前的快照恢复回去 → `verify_restore` 与写回前逐表内容、业务不变量完全相同
    """
    from app.tasks.restore import verify_restore
    from app.tasks.snapshot_manifest import describe_db
    copy, out = Path(copy), Path(out)
    if out.exists() and any(out.iterdir()):
        raise PreviewRefused(f"输出目录 {out} 不是空的")
    out.mkdir(parents=True, exist_ok=True)
    inv = inventory(copy)
    fix, unresolved = _changes_fix(inv)
    reeval = _changes_reeval(json.loads(Path(reevaluation).read_text(encoding="utf-8")) if reevaluation else None)
    pre = out / "before_writeback.db"
    shutil.copy2(copy, pre)
    work = out / "rehearsal.db"
    shutil.copy2(copy, work)
    d0 = describe_db(pre)
    con = _ro(pre)
    try:
        have = {r[1] for r in con.execute("PRAGMA table_info(alpha_records)")}
    finally:
        con.close()
    missing = [c for c in ("ic_mean", "ic_metrics") if c not in have]
    at = datetime.now(timezone.utc).isoformat()
    revision_ids = _apply_fix(work, fix, actor, at)
    d1 = describe_db(work)
    other = sorted(t for t in d0["tables"] if t not in ("alpha_records", "alpha_decisions")
                   and d0["tables"][t]["content_sha256"] != d1["tables"].get(t, {}).get("content_sha256"))
    base_cols = ["dsl", "status", "sharpe", "ann_return", "ann_turnover", "hypothesis", "reasoning"]
    r0, r1 = (_rows(pre, "alpha_records", base_cols + list(IC_COLS)),
              _rows(work, "alpha_records", base_cols + list(IC_COLS)))
    target = {c["id"] for c in fix}
    # 非目标行：写回前有的列一个不变，补上的列（写回前没有 → None）仍为空
    untouched_changed = sorted(i for i in r0 if i not in target
                               and any(r0[i].get(k) != r1[i].get(k) for k in base_cols + list(IC_COLS)))
    landed = all({k: r1[c["id"]].get(k) for k in IC_COLS} == {k: c["new"][k] for k in IC_COLS} for c in fix)
    n_dec = (d1["tables"].get("alpha_decisions", {}).get("rows", 0) - d0["tables"].get("alpha_decisions", {}).get("rows", 0))
    check_after = {"only_target_rows_changed": not untouched_changed, "other_tables_unchanged": not other,
                   "new_values_landed": landed, "revision_rows_added": n_dec, "expected_revision_rows": len(fix),
                   "integrity": d1["integrity"], "ok": (not untouched_changed and not other and landed
                                                        and n_dec == len(fix) and d1["integrity"] == "ok")}
    # ① 逻辑回滚（在演练库上）
    con = sqlite3.connect(work)
    try:
        con.execute("BEGIN IMMEDIATE")
        for c in fix:
            con.execute("UPDATE alpha_records SET ic_ir = ?, ic_mean = ?, ic_metrics = ? WHERE id = ?",
                        (c["old"]["ic_ir"], c["old"]["ic_mean"], c["old"]["ic_metrics"], c["id"]))
        con.executemany("DELETE FROM alpha_decisions WHERE id = ?", [(i,) for i in revision_ids])
        con.commit()
    finally:
        con.close()
    r2 = _rows(work, "alpha_records", base_cols + list(IC_COLS))
    dec0, dec2 = (_rows(pre, "alpha_decisions", ["alpha_id", "decision", "reason"]),
                  _rows(work, "alpha_decisions", ["alpha_id", "decision", "reason"]))
    restored_rows = set(r2) == set(r0) and all(r2[i].get(k) == r0[i].get(k) for i in r0
                                               for k in base_cols + list(IC_COLS))
    logical = {"original_columns_restored": restored_rows, "revision_rows_removed": dec2 == dec0,
               "note": "补上的两列（ic_mean / ic_metrics）留在表里、值为空 —— 与下次启动时程序自己补列的结果相同"}
    # ② 整库回滚：写回前的快照恢复回去
    restored = out / "restored_after_rollback.db"
    shutil.copy2(pre, restored)
    physical = verify_restore(pre, restored)
    plan = {"at": at, "copy": str(copy), "writes": "rehearsal copies only（活库没有任何连接）",
            "migration": {"add_columns": missing, "values_changed": 0},
            "fix": fix, "fix_unresolved": unresolved, "reevaluation": reeval,
            "affected": _affected(copy, fix, inv),
            "rehearsal": {"after_apply": check_after, "logical_rollback": logical,
                          "physical_rollback": {"ok": physical["ok"], "content_mismatches": physical["content_mismatches"],
                                                "invariant_mismatches": physical["invariant_mismatches"]}},
            "live_steps": [
                "1. 你停掉服务与调度；确认活库旁的 -wal 是 0 字节（不要删 -wal / -shm）",
                "2. python tools/research/factor_metrics_dry_run.py snapshot --db <活库> --out <新的空目录>（停写窗口快照 + 恢复核验）",
                "3. 在那份新副本上重跑 writeback-plan，确认与本预演的变更集相同（旧值守卫保证：不同就整笔不写）",
                "4. 你确认后才执行写回（本工具不提供写活库的命令；写回时用同一个 _apply_fix、一个事务）",
                "5. 写回后再取一次停写窗口快照，核对：只有目标行三列变化、修订行条数相符、其余表内容摘要不变",
                "回滚：整库换回第 2 步的快照（verify_restore 核对与写回前完全相同）；或逐行恢复旧值并删除本次的修订行"]}
    (out / "writeback_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    plan["report_file"] = str(out / "writeback_plan.json")
    return plan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="活库因子指标：备份 + 恢复验证 + 版本化 dry-run（不写库）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--db", required=True)
    s.add_argument("--out", required=True)
    i = sub.add_parser("inventory")
    i.add_argument("--copy", required=True)
    d = sub.add_parser("dry-run")
    d.add_argument("--copy", required=True)
    d.add_argument("--dataset", required=True)
    d.add_argument("--start")
    d.add_argument("--end")
    w = sub.add_parser("writeback-plan")
    w.add_argument("--copy", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--reevaluation", help="dry-run 产出的 factor_metrics_dry_run.json（B 重评估的新值）")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "snapshot":
            out = snapshot_live(Path(a.db), Path(a.out))
        elif a.cmd == "inventory":
            out = inventory(Path(a.copy))
        elif a.cmd == "writeback-plan":
            out = writeback_plan(Path(a.copy), Path(a.out), Path(a.reevaluation) if a.reevaluation else None)
        else:
            out = dry_run(Path(a.copy), a.dataset, a.start, a.end)
    except (LiveDbBusy, PreviewRefused) as exc:
        print(f"拒绝：{exc}")
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
