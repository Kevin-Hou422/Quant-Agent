"""
alerts.py — 告警通道的人工入口（2026-10-22 用户决定 D15）

在 backend/ 目录下：

  python -m app.tasks.alerts test                          # 发一条测试通知（请在电脑前看一眼）
  python -m app.tasks.alerts confirm <事件 id> --actor 你   # 看见了 → 记一次现场确认（无人值守就绪的前提）
  python -m app.tasks.alerts status                        # 就绪结论 + 最近的告警与送达状态
  python -m app.tasks.alerts retry                         # 重新通知送达失败的告警

"通知交给了系统"不等于"有人看见"：专注模式、通知设置、锁屏都会让它看不见，所以就绪只认现场确认。
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None, dispatcher=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.tasks.alerts", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test")
    c = sub.add_parser("confirm")
    c.add_argument("event_id")
    c.add_argument("--actor", required=True)
    sub.add_parser("status")
    sub.add_parser("retry")
    a = ap.parse_args(argv)
    from app.core.alerting.dispatcher import AlertDispatcher
    d = dispatcher or AlertDispatcher()
    if a.cmd == "test":
        eid = d.send_test_notification()
        st = next(e["status"] for e in d.history() if e["event_id"] == eid)
        print(f"测试通知 {eid}：{st}")
        if st == "handed_to_os":
            print(f"看见了就运行：python -m app.tasks.alerts confirm {eid} --actor <你>")
            return 0
        print("通知没有交给系统 —— 通知通道不可用，不能称为无人值守就绪")
        return 1
    if a.cmd == "confirm":
        d.confirm_visible(a.event_id, a.actor)
        print("已记录现场确认")
        return 0
    if a.cmd == "retry":
        print(d.retry_failed())
        return 0
    ok, why = d.unattended_readiness()
    print(("就绪：" if ok else "未就绪：") + why)
    for e in d.history()[-20:]:
        print(f"  {e['at'][:19]}  {e['severity']:<8} {e['kind']:<28} {e['status']:<13} {e['title']}")
    return 0 if ok else 1


if __name__ == "__main__":       # pragma: no cover - 命令行入口
    sys.exit(main())
