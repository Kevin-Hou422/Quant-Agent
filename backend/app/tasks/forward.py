"""
forward.py — 前向交易的人工入口（Phase 12 收尾）

在 backend/ 目录下：

  python -m app.tasks.forward preflight     # 逐项检查能不能真跑（会对账一次，不下单）
  python -m app.tasks.forward run-now       # 预检全过 → 立刻跑一次每日管线（会真的下纸交易单）

`run-now` 走的是调度器每天 21:30 UTC 跑的**同一个** `daily_trading_job`，不是另一条路径；
预检有任何阻塞项不过就拒绝执行，退出码 2。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, List, Optional

EXIT_OK = 0
EXIT_NOT_READY = 2


def _print_preflight(rep) -> None:
    for c in rep.checks:
        mark = "OK  " if c.ok else ("FAIL" if c.blocking else "WARN")
        print(f"  [{mark}] {c.name:<15} {c.detail}")
    print("  => " + ("可以前向交易" if rep.ready else "未就绪：先处理 FAIL 项"))


def preflight(run: Optional[Callable] = None) -> int:
    from app.core.execution.golive import run_preflight
    rep = (run or run_preflight)()
    _print_preflight(rep)
    return EXIT_OK if rep.ready else EXIT_NOT_READY


def run_now(run: Optional[Callable] = None, job: Optional[Callable[[], dict]] = None) -> int:
    from app.core.execution.golive import run_preflight
    rep = (run or run_preflight)()
    _print_preflight(rep)
    if not rep.ready:
        print("  拒绝执行 run-now：预检未通过。")
        return EXIT_NOT_READY
    if job is None:
        from app.tasks.scheduler import daily_trading_job as job
    out = job() or {}
    pf = out.get("portfolio") or {}
    summary = {k: out.get(k) for k in ("skipped", "ingest_accepted", "reason", "reject_reason",
                                       "mode", "forward_from", "n_new_bars")}
    summary["execution"] = pf.get("execution")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.tasks.forward", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["preflight", "run-now"])
    args = ap.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    return preflight() if args.command == "preflight" else run_now()


if __name__ == "__main__":       # pragma: no cover - 命令行入口
    sys.exit(main())
