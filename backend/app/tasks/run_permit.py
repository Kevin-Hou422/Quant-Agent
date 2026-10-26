"""
run_permit.py — 运行许可的命令行（签发 / 查看 / 撤销；规则见 app/core/execution/run_permit.py）

在 backend/ 目录下：

  python -m app.tasks.run_permit status [--exec-db <执行库>]
  python -m app.tasks.run_permit issue --mode supervised --account <acc_id> --hours 8 --max-order-notional 2000 \\
      --actor <你> --reason "…" [--exec-db <执行库>]
  python -m app.tasks.run_permit issue --mode unattended --account <acc_id> --hours 72 \\
      --acceptance-evidence <simulate_acceptance.jsonl> --actor <你> --reason "…"
  python -m app.tasks.run_permit revoke --actor <你> --reason "…"

缺省执行库 = settings.database_url。签发不满足条件 → 退出码 2，什么都不写。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

EXIT_OK = 0
EXIT_REFUSED = 2


def _store(exec_db: Optional[str]):
    from app.config import settings
    from app.db.execution_store import ExecutionStore
    return ExecutionStore(db_url=f"sqlite:///{exec_db}" if exec_db else str(settings.database_url))


def main(argv: Optional[List[str]] = None) -> int:
    from app.core.execution import run_permit as rp
    ap = argparse.ArgumentParser(prog="python -m app.tasks.run_permit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["status", "issue", "revoke"])
    ap.add_argument("--exec-db")
    ap.add_argument("--mode", choices=rp.MODES)
    ap.add_argument("--account", type=int)
    ap.add_argument("--hours", type=float)
    ap.add_argument("--max-order-notional", type=float)
    ap.add_argument("--acceptance-evidence")
    ap.add_argument("--actor", default="")
    ap.add_argument("--reason", default="")
    a = ap.parse_args(argv)
    store = _store(a.exec_db)
    try:
        if a.command == "status":
            p = rp.current(store)
            dec = rp.evaluate(store, account=(p or {}).get("account")) if p else None
            print(json.dumps({"permit": p, "now": dec.to_dict() if dec else None}, ensure_ascii=False, indent=1))
            return EXIT_OK
        if a.command == "revoke":
            rp.revoke(store, actor=a.actor, reason=a.reason)
            print("已撤销")
            return EXIT_OK
        if not (a.mode and a.account and a.hours):
            print("签发要给 --mode --account --hours")
            return EXIT_REFUSED
        p = rp.issue(store, mode=a.mode, actor=a.actor, reason=a.reason, account=a.account, hours=a.hours,
                     max_order_notional=a.max_order_notional, acceptance_evidence=a.acceptance_evidence)
        print(json.dumps(p, ensure_ascii=False, indent=1))
        return EXIT_OK
    except rp.PermitRefused as exc:
        print(f"拒绝：{exc}")
        return EXIT_REFUSED
    finally:
        store._engine.dispose()                                     # noqa: SLF001


if __name__ == "__main__":       # pragma: no cover - 命令行入口
    sys.exit(main())
