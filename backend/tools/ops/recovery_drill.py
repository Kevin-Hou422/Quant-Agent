"""
recovery_drill.py — 合成数据上的隔离恢复演练（2026-10-22 用户决定第 21 条）

    python tools/ops/recovery_drill.py --work <一个新的空目录>

在 `--work` 里造一个合成主库（因子 / 审批谱系 / IC 历史 / 执行与策略表）和一份合成 PIT，
用产品备份路径（`run_daily_backup`）做快照，再用 `restore_snapshot` 恢复到隔离目录，`verify_restore` 逐表、逐文件核对。

**范围**：只证明"备份 → 恢复 → 核对"这条路径在合成数据上走得通。它不读、不写活库，
所以 **合成恢复通过不等于活库恢复已验证** —— 活库的恢复要拿活库的真实快照另做（D2 / D13，须人工确认）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

NOTE = ("合成数据演练：只证明备份 → 恢复 → 核对这条路径走得通；没有读写活库，"
        "合成恢复通过不等于活库恢复已验证（活库恢复须用活库的真实快照另做，并经人工确认）。")


def _build_source(src: Path) -> tuple:
    import numpy as np
    import pandas as pd

    from app.core.data_engine.pit_store import PITStore
    from app.db.alpha_store import AlphaResult, AlphaStore
    from app.db.execution_store import ExecutionStore
    from app.db.strategy_store import StrategyStore

    src.mkdir(parents=True, exist_ok=True)
    db = src / "alphas.db"
    url = f"sqlite:///{db.as_posix()}"
    alphas = AlphaStore(db_url=url)
    try:
        for i in range(3):
            aid = alphas.save(AlphaResult(dsl=f"rank(ts_mean(close, {5 + i}))", sharpe=0.5 + i / 10))
            alphas.record_decision(aid, "reject", "candidate", "rejected", reason="合成", actor="recovery_drill")
            alphas.record_ic(aid, "2025-03-03", 0.01 * (i + 1))
    finally:
        alphas._engine.dispose()                                   # noqa: SLF001
    for store in (ExecutionStore(db_url=url), StrategyStore(db_url=url)):
        store._engine.dispose()                                    # noqa: SLF001

    pit = src / "pit_store"
    idx = pd.bdate_range("2025-03-03", periods=5)
    rng = np.random.default_rng(0)
    panel = {f: pd.DataFrame(rng.random((5, 3)) + 10, index=idx, columns=["AAA", "BBB", "CCC"])
             for f in ("close", "volume")}
    PITStore(pit).append(panel, as_of="2025-03-07")
    return db, pit


def run_synthetic_drill(work_dir: Path) -> dict:
    from app.tasks.backup import run_daily_backup
    from app.tasks.restore import restore_snapshot, verify_restore

    work = Path(work_dir)
    if work.exists() and any(work.iterdir()):
        raise ValueError(f"演练目录 {work} 不是空的")
    db, pit = _build_source(work / "source")
    backups = work / "backups"
    bk = run_daily_backup(backup_dir=str(backups), keep=3, database_url=f"sqlite:///{db.as_posix()}",
                          scheduler_db_url=f"sqlite:///{(work / 'source' / 'scheduler.db').as_posix()}",
                          pit_store_dir=str(pit))
    restored = work / "restored"
    rs = restore_snapshot(Path(bk.snapshot_dir), restored) if bk.ok else None
    # 对照的是快照文件本身（冻结的）；源库在备份之后可能又被写过，不能拿它当恢复的基准（2026-10-25 审计 R2）
    ver = verify_restore(Path(bk.snapshot_dir) / db.name, restored / db.name, pit, restored / "pit_store") if rs else None
    ok = bool(bk.ok and rs and rs["ok"] and ver and ver["ok"])
    return {"ok": ok, "scope": "synthetic", "live_restore_verified": False, "note": NOTE,
            "paths": {"source_db": str(db), "source_pit": str(pit), "backups": str(backups),
                      "snapshot": bk.snapshot_dir, "restored": str(restored)},
            "backup": bk.to_dict(), "restore": rs, "verification": ver}


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="合成数据上的隔离恢复演练（不碰活库）")
    ap.add_argument("--work", required=True, help="一个新的空目录")
    a = ap.parse_args(list(argv) if argv is not None else None)
    rep = run_synthetic_drill(Path(a.work))
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
