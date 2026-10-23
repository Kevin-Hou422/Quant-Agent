"""
ca_flags.py — 疑似未知公司行动的隔离标记：列出 / 对账后结案（2026-10-22 用户决定第 18 条）

    python tools/ops/ca_flags.py --db C:\\QuantAgentData\\alphas.db list
    python tools/ops/ca_flags.py --db C:\\QuantAgentData\\alphas.db resolve AAPL \\
        --actor <你> --resolution "券商股数 ×4、现金未变；数据源已复权"

`list` 只读（sqlite3 只读连接，不建表）。`resolve` 写库：结案后这只票恢复可以加风险。
结案前请核对：券商持仓股数 / 现金是否与拆股 / 分红一致、数据源的复权价是否已重算（必要时整段重取）。
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, Optional


def list_flags(db: Path) -> list:
    con = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    try:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE name='corporate_action_flags'").fetchone():
            return []
        cols = ("ticker", "bar_date", "kind", "ratio", "nearest_split", "status", "resolved_by", "resolution")
        rows = con.execute(f"SELECT {', '.join(cols)} FROM corporate_action_flags ORDER BY status, bar_date").fetchall()
        return [dict(zip(cols, r)) for r in rows]
    finally:
        con.close()


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="疑似未知公司行动的隔离标记")
    ap.add_argument("--db", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("resolve")
    r.add_argument("ticker")
    r.add_argument("--actor", required=True)
    r.add_argument("--resolution", required=True)
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.cmd == "list":
        rows = list_flags(Path(a.db))
        for row in rows:
            print(row)
        print(f"共 {len(rows)} 条（open = 仍在隔离：不加风险）")
        return 0
    from app.db.corporate_actions import CorporateActionStore
    store = CorporateActionStore(db_url=f"sqlite:///{Path(a.db).as_posix()}")
    try:
        n = store.resolve(a.ticker, actor=a.actor, resolution=a.resolution)
    finally:
        store._engine.dispose()                  # noqa: SLF001
    print(f"{a.ticker}：{n} 条隔离标记结案")
    return 0 if n else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
