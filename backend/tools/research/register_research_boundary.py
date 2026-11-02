"""
register_research_boundary.py — 登记一个数据集的冻结 Test 起点（2026-10-22 D12：日期建议后由用户确认）

缺省是**预演**：只读查一遍、打印将要写什么，不碰库（不建表、不补列）。加 `--confirm` 才写，
写的是 `frozen_boundaries` 的一行（操作人 + 理由，只写一次；已有登记 / 已冻结在别的日期 → 拒绝）。
写活库前请先备份（D2）。

    python tools/research/register_research_boundary.py --db C:\\QuantAgentData\\alphas.db \\
        --dataset us_tech_large --test-start 2026-10-12 --actor <你> --reason "<为什么选这一天>" [--confirm]

未来窗口（2026-10-30 用户决定："采用真正未见的未来窗口原则……在实际登记准备时，按当天和交易所日历重新计算'登记日之后的
下一个交易日'"）：`--test-start next-session` —— 起点在**登记的那一刻**按交易所日历算（交易所当地日期之后的下一个交易日），
不预先写死某个日期（写死的日期会过期：2026-10-28 就是这样作废的）。预演与 `--confirm` 各算一次，跨了交易日两者不同 →
以写库那一刻为准并打印出来。给固定日期时，计划里写明它在登记时是否已经可见（`unseen_at_registration`）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


NEXT_SESSION = "next-session"


def resolve_test_start(test_start: str, now=None) -> tuple:
    """(Test 起点, 登记日 = 交易所当地日期)。`next-session` → 登记日之后的下一个交易日（交易所日历）。"""
    from app.core.data_engine.market_calendar import trading_days
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    today = now.tz_convert("America/New_York").tz_localize(None).normalize()
    if str(test_start).strip().lower() != NEXT_SESSION:
        return pd.Timestamp(test_start).normalize(), today
    nxt = next(d for d in trading_days(today + pd.Timedelta(days=1), today + pd.Timedelta(days=15)) if d > today)
    return pd.Timestamp(nxt).normalize(), today


def plan(db: Path, dataset: str, test_start: str, now=None) -> dict:
    """预演：只读看现状，返回将要发生什么。"""
    from tools.research.research_boundary_report import _columns, _tables, read_frozen
    with read_frozen(db) as (con, _window):
        existing = None
        if "frozen_boundaries" in _tables(con):
            cols = _columns(con, "frozen_boundaries")
            row = con.execute("SELECT * FROM frozen_boundaries WHERE dataset_key = ?", (dataset,)).fetchone()
            if row is not None:
                r = dict(row)
                existing = {"test_start": r["test_start"],
                            "registered": bool({"actor", "reason"} <= cols and (r.get("actor") or "").strip())}
    start, today = resolve_test_start(test_start, now)
    day = str(start.date())
    if existing is None:
        action = "insert"
    elif existing["registered"]:
        action = "refuse: 已登记"
    elif existing["test_start"] != day:
        action = f"refuse: 已冻结在 {existing['test_start']}"
    else:
        action = "confirm_existing"
    return {"dataset": dataset, "test_start": day, "existing": existing, "action": action,
            "registered_on": str(today.date()), "unseen_at_registration": bool(start > today),
            "rule": ("登记日之后的下一个交易日（交易所日历，登记那一刻计算）" if str(test_start).strip().lower() == NEXT_SESSION
                     else "固定日期" + ("" if start > today else "：登记时已经过去 / 可见 —— 不是未见的未来窗口"))}


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="D12：登记冻结 Test 起点（缺省只预演）")
    ap.add_argument("--db", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--test-start", required=True, help=f"YYYY-MM-DD，或 {NEXT_SESSION}（登记那一刻按交易所日历算）")
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
        start, _today = resolve_test_start(a.test_start)                 # 写库那一刻再算一次（预演之后可能跨了交易日）
        got = store.register(a.dataset, start, actor=a.actor, reason=a.reason)
    finally:
        store._engine.dispose()                  # noqa: SLF001 —— 关掉连接，WAL 随之合并回主文件
    print(f"已登记：{a.dataset} 的冻结 Test 起点 = {got.date()}（{a.actor}：{a.reason}）")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
