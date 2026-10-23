"""
register_research_boundary.py — 登记一个数据集的冻结 Test 起点（2026-10-22 D12：日期建议后由用户确认）

缺省是**预演**：只读查一遍、打印将要写什么，不碰库（不建表、不补列）。加 `--confirm` 才写，
写的是 `frozen_boundaries` 的一行（操作人 + 理由，只写一次；已有登记 / 已冻结在别的日期 → 拒绝）。
写活库前请先备份（D2）。

    python tools/research/register_research_boundary.py --db C:\\QuantAgentData\\alphas.db \\
        --dataset us_tech_large --test-start 2026-10-12 --actor <你> --reason "<为什么选这一天>" [--confirm]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def plan(db: Path, dataset: str, test_start: str) -> dict:
    """预演：只读看现状，返回将要发生什么。"""
    from tools.research.research_boundary_report import _connect_readonly, _columns, _tables
    con = _connect_readonly(db)
    try:
        existing = None
        if "frozen_boundaries" in _tables(con):
            cols = _columns(con, "frozen_boundaries")
            row = con.execute("SELECT * FROM frozen_boundaries WHERE dataset_key = ?", (dataset,)).fetchone()
            if row is not None:
                r = dict(row)
                existing = {"test_start": r["test_start"],
                            "registered": bool({"actor", "reason"} <= cols and (r.get("actor") or "").strip())}
    finally:
        con.close()
    day = str(pd.Timestamp(test_start).date())
    if existing is None:
        action = "insert"
    elif existing["registered"]:
        action = "refuse: 已登记"
    elif existing["test_start"] != day:
        action = f"refuse: 已冻结在 {existing['test_start']}"
    else:
        action = "confirm_existing"
    return {"dataset": dataset, "test_start": day, "existing": existing, "action": action}


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="D12：登记冻结 Test 起点（缺省只预演）")
    ap.add_argument("--db", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--test-start", required=True)
    ap.add_argument("--actor", required=True)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--confirm", action="store_true", help="真的写库（缺省只预演）")
    a = ap.parse_args(list(argv) if argv is not None else None)
    db = Path(a.db)
    p = plan(db, a.dataset, a.test_start)
    print(f"预演：{p}")
    if not a.confirm:
        print("未加 --confirm：没有写库")
        return 0
    if p["action"].startswith("refuse"):
        print(f"拒绝：{p['action']}")
        return 2
    from app.db.trial_ledger import BoundaryStore
    store = BoundaryStore(db_url=f"sqlite:///{db.as_posix()}")
    try:
        got = store.register(a.dataset, pd.Timestamp(a.test_start), actor=a.actor, reason=a.reason)
    finally:
        store._engine.dispose()                  # noqa: SLF001 —— 关掉连接，WAL 随之合并回主文件
    print(f"已登记：{a.dataset} 的冻结 Test 起点 = {got.date()}（{a.actor}：{a.reason}）")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
