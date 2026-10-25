"""
reproposal_rehearsal.py — 现役策略重新提案的隔离预演（2026-10-22 用户决定 D13）

用户原话："同意重提案，但先修正确性、D14 和资格状态。先隔离预演，再由我批准活环境操作。
不手工给旧策略补 hash，不继承旧时钟证据，不要先激活旧时钟再重做。"

    python tools/research/reproposal_rehearsal.py --copy <preview>/restored/alphas.db [--dataset …]

前提：先用 `factor_metrics_dry_run.py snapshot` 拿到活库快照副本。本工具把副本**再复制一份**到 `<preview>/rehearsal/`，
配置指向这份预演库，活库与恢复副本都不写：

    1. 记下现有策略配置（状态、有没有规格、verdict 原文的哈希）
    2. 在研究上下文里提两份：按提案接口原样（当前全部 PAPER/ACTIVE 因子，切片 = PAPER_AUM）/ 按旧策略的成分与切片
    3. 按旧策略成分的那份模拟"批准并开始纸交易"—— 与 `POST /strategies/{id}/approve?activate=true` 同一组迁移
       （旧版本退役并留谱系）
    4. 核对：旧配置的 verdict 一字未改（不补哈希）、新规格没有前向证据（不继承旧时钟的）、
       状态是 paper_running（不是已验证）、规格与此刻的运行时配置一致
报告写到 `<preview>/reproposal_rehearsal.json`。真的在活库上做，要你看过报告后另行批准。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.research.factor_metrics_dry_run import (  # noqa: E402
    PREVIEW_MANIFEST,
    PreviewRefused,
    _settings_pointing_at,
)


def _sha(text) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _legacy_view(sstore) -> dict:
    from app.core.portfolio_manager.executable_spec import approved_spec
    return {str(r.id): {"name": r.name, "status": str(r.status), "verdict_sha256": _sha(r.verdict),
                        "has_spec": approved_spec(r) is not None} for r in sstore.query(limit=500)}


def _summary(rec, evidence) -> dict:
    v = json.loads(rec.verdict or "{}")
    gates = {k: {"evaluated": g.get("evaluated"), "passed": g.get("passed"),
                 "detail": g.get("detail") or "; ".join(g.get("reasons", []) or []) or g.get("error", "")}
             for k, g in (v.get("gates") or {}).items()}
    return {"id": rec.id, "name": rec.name, "status": str(rec.status), "factors": list(json.loads(rec.factors)),
            "aum": float(rec.aum), "method": rec.method, "passed": bool(v.get("passed")), "grade": v.get("paper_grade"),
            "gates": gates, "reasons": v.get("reasons", []), "spec_hash": v.get("spec_hash", ""),
            "clock_version": v.get("clock_version"), "approved_code_version": v.get("approved_code_version"),
            "forward_evidence_rows": len(evidence.strategy_forward_ic(rec.id, v.get("spec_hash") or None)),
            "replay_evidence_rows": len(evidence.strategy_replay_ic(rec.id, v.get("spec_hash") or None))}


def rehearse(copy: Path, dataset: str, start: Optional[str] = None, end: Optional[str] = None,
             loader: Optional[Callable] = None, actor: str = "reproposal_rehearsal") -> dict:
    from app.config import settings
    from app.core.alpha_engine.dsl_executor import Executor
    from app.core.alpha_engine.signal_processor import SignalProcessor
    from app.core.data_engine.input_identity import clock_version, code_version
    from app.core.data_engine.research_context import open_research
    from app.core.lifecycle.strategy_states import LEGACY_ACTIVE, permissions
    from app.core.portfolio_manager import propose_from_paper_factors
    from app.core.portfolio_manager.executable_spec import (
        approved_spec,
        signal_config,
        verify_against_runtime,
    )
    from app.core.portfolio_manager.strategy_builder import build_strategy_config
    from app.db.alpha_store import AlphaStore
    from app.db.evidence_store import EvidenceStore
    from app.db.strategy_store import StrategyStore
    from app.tasks.backup import _sqlite_path_from_url

    copy = Path(copy)
    preview = copy.parent.parent
    live = _sqlite_path_from_url(getattr(settings, "database_url", "") or "")
    if not (preview / PREVIEW_MANIFEST).exists() or (live is not None and live == copy.resolve()):
        raise PreviewRefused(f"{copy} 不是快照产出的副本 —— 预演只在副本上做")
    work = preview / "rehearsal" / copy.name
    if work.parent.exists():
        shutil.rmtree(work.parent)                              # 预演库是本工具自己的产物：每次重来
    work.parent.mkdir(parents=True)
    shutil.copy2(copy, work)
    if loader is None:
        from app.core.data_engine.dataset_registry import (
            load_registry_dataset as loader,
        )

    url = f"sqlite:///{work.as_posix()}"
    with _settings_pointing_at(work):
        astore, sstore, evidence = AlphaStore(db_url=url), StrategyStore(db_url=url), EvidenceStore(db_url=url)
        try:
            before = _legacy_view(sstore)
            legacy = [r for r in sstore.query(limit=500) if str(r.status) == LEGACY_ACTIVE]
            panel = loader(dataset, start=start, end=end).data
            ctx = open_research(panel, dataset, "reproposal_rehearsal")
            proposals = {}
            with ctx.active():
                cfg_api = propose_from_paper_factors(astore, ctx.selection, aum=float(settings.paper_aum),
                                                     name="rehearsal-as-api")
                if cfg_api is not None:
                    proposals["as_api"] = sstore.save(cfg_api)
                if legacy:
                    old = legacy[0]
                    ids = [str(x) for x in json.loads(old.factors)]
                    dsl = {i: astore.get_by_id(int(i)).dsl for i in ids}
                    cfg = signal_config()
                    sig = {i: SignalProcessor(cfg).process(Executor(validate=False).run_expr(d, ctx.selection))
                           for i, d in dsl.items()}
                    proposals["same_composition"] = sstore.save(build_strategy_config(
                        sig, ctx.selection, aum=float(old.aum), method=old.method, factor_dsl=dsl,
                        name=f"{old.name}-c2"))
            started = None
            if "same_composition" in proposals:
                from app.api.router import _start_steps
                sid = proposals["same_composition"]
                req = SimpleNamespace(reason="隔离预演：按旧策略成分重新提案", actor=actor)
                sstore.apply_transitions([(sid, "approved", "approve", req.reason, actor)]
                                         + _start_steps(sstore, sid, req, "paper_running"))
                rec = sstore.get(sid)
                spec = approved_spec(rec)
                dsl_now = {k: astore.get_by_id(int(k)).dsl for k in (spec or {}).get("factors", {})}
                started = {"id": sid, "status": str(rec.status), "permissions": permissions(str(rec.status)),
                           "runtime_spec_mismatches": verify_against_runtime(spec, dsl_now, float(rec.aum))
                           if spec else ["没有规格"]}
            after = _legacy_view(sstore)
            out = {
                "at": datetime.now(timezone.utc).isoformat(), "writes": "rehearsal copy only",
                "work_db": str(work), "copy": str(copy),
                "versions": {"code_version": code_version(), "clock_version": clock_version(), "dataset": dataset,
                             "price_source": str(getattr(settings, "price_source", "")),
                             "research_context": ctx.describe()},
                "legacy": {k: {"name": b["name"], "status_before": b["status"],
                               "status_after": after.get(k, {}).get("status"),
                               "verdict_unchanged": after.get(k, {}).get("verdict_sha256") == b["verdict_sha256"],
                               "has_spec_after": after.get(k, {}).get("has_spec")}
                           for k, b in before.items()},
                "proposals": {k: _summary(sstore.get(v), evidence) for k, v in proposals.items()},
                "started": started,
            }
        finally:
            for st in (astore, sstore, evidence):
                st._engine.dispose()                             # noqa: SLF001
    path = preview / "reproposal_rehearsal.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    out["report_file"] = str(path)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="现役策略重新提案的隔离预演（只写预演库）")
    ap.add_argument("--copy", required=True)
    ap.add_argument("--dataset", default=None, help="缺省 = 配置里的 PAPER_DATASET（与提案接口同一个）")
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args(argv)
    from app.config import settings
    try:
        out = rehearse(Path(a.copy), a.dataset or str(settings.paper_dataset), a.start or settings.paper_start,
                       a.end or settings.paper_end)
    except PreviewRefused as exc:
        print(f"拒绝：{exc}")
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
