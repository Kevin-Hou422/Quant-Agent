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
#: 跑了，但没有按计划完成（执行被拦 / 下单异常 / 组合报错 / 摄取被拒 / 日历不可用）。
#: 以前这些情况一律返回 0，外部监控会把"一张单都没下"当成成功（审计 F14）。
EXIT_RUN_FAILED = 3


def classify_run(out: dict) -> tuple:
    """(退出码, 一句话结论)。"按计划无交易"（非交易日）算成功，其余未完成一律失败。"""
    out = out or {}
    if out.get("skipped") == "not_a_trading_day":
        return EXIT_OK, "非美股交易日，按计划不交易"
    if out.get("skipped"):
        return EXIT_RUN_FAILED, f"任务未执行：{out.get('skipped')} {out.get('error', '')}".strip()
    pf = out.get("portfolio") or {}
    if out.get("ingest_accepted") is False and out.get("reason") != "no_new_bar":
        return EXIT_RUN_FAILED, f"摄取被拒：{out.get('reject_reason')}"
    if pf.get("error"):
        return EXIT_RUN_FAILED, f"组合账本失败：{pf['error']}"
    ex = pf.get("execution")
    if not ex:
        return EXIT_RUN_FAILED, "执行层没有运行"
    if ex.get("mode") == "off":
        return EXIT_RUN_FAILED, "EXECUTION_MODE=off，没有下单"
    if ex.get("blocked"):
        return EXIT_RUN_FAILED, f"执行被拦：{ex['blocked']}"
    if ex.get("n_submit_errors"):
        return EXIT_RUN_FAILED, f"{ex['n_submit_errors']} 张单下单异常（待对账确认）"
    return EXIT_OK, (f"执行完成：计划 {ex.get('n_planned', 0)} / 已下 {ex.get('n_submitted', 0)} / "
                     f"风控拒 {ex.get('n_gate_rejected', 0)} / 重复 {ex.get('n_duplicate', 0)}")


def _print_preflight(rep) -> None:
    for c in rep.checks:
        mark = "OK  " if c.ok else ("FAIL" if c.blocking else "WARN")
        print(f"  [{mark}] {getattr(c, 'layer', ''):<7} {c.name:<16} {c.detail}")
    print(f"  => 券商侧：{'可用' if getattr(rep, 'connected', rep.ready) else '不可用'}；"
          f"交易就绪：{'是，可以交给调度器' if rep.ready else '否 —— 先处理 FAIL 项'}")


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
    summary["portfolio_error"] = pf.get("error")
    summary["execution"] = pf.get("execution")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    code, verdict = classify_run(out)
    print(("  => " if code == EXIT_OK else "  => 失败：") + verdict)
    return code


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
