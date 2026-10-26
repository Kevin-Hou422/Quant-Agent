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
                            "ic_ir", "ann_turnover", "clock_version") if c in cols]
        rows = [dict(zip(want, r)) for r in con.execute(f"SELECT {', '.join(want)} FROM alpha_records ORDER BY id")]
    finally:
        con.close()
    out = []
    for r in rows:
        writer, meaning = _writer(r.get("hypothesis"), r.get("reasoning"))
        stored = {k: r.get(k) for k in ("sharpe", "ann_return", "ic_ir", "ann_turnover")}
        out.append({"id": r["id"], "dsl": r["dsl"], "status": r["status"], "created_at": str(r.get("created_at")),
                    "clock_version": r.get("clock_version") or "c1（列不存在或为空 = 2026-10-20 之前）",
                    "writer": writer, "ic_ir_column_holds": meaning,
                    "evaluated": any(abs(float(v or 0.0)) > 0 for v in stored.values()), "stored": stored})
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
                                   "ann_turnover": f(rep.ann_turnover)}
            except Exception as exc:
                r["recomputed"] = {"mean_ic": None, "ic_ir": None, "sharpe": None, "ann_turnover": None,
                                   "error": f"{type(exc).__name__}: {exc}"}
    nan = float("nan")
    old_rank = _ranks({r["id"]: float(r["stored"]["ic_ir"] or 0.0) for r in inv})
    new_rank = _ranks({r["id"]: (r["recomputed"]["mean_ic"] if r["recomputed"]["mean_ic"] is not None else nan)
                       for r in inv})
    for r in inv:
        r["evaluated_before"] = r.pop("evaluated")
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
        "note": ("旧值原样保留（本工具不写库）。`ic_ir` 列在多数写入路径里存的是样本内 IC 均值 —— 新值两种口径都给；"
                 "该列以后存什么、要不要写回，等你决定。"),
    }
    report_dir = Path(report_dir or copy.parent.parent)
    path = report_dir / "factor_metrics_dry_run.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    out["report_file"] = str(path)
    return out


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
    a = ap.parse_args(argv)
    try:
        if a.cmd == "snapshot":
            out = snapshot_live(Path(a.db), Path(a.out))
        elif a.cmd == "inventory":
            out = inventory(Path(a.copy))
        else:
            out = dry_run(Path(a.copy), a.dataset, a.start, a.end)
    except (LiveDbBusy, PreviewRefused) as exc:
        print(f"拒绝：{exc}")
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
