"""
d12_boundary_report.py — 研究边界与 Test 段曝光的**只读**报告（2026-10-22 用户决定 D12）

用户原话："同意先只读查询，不授权写边界。使用明确的活库路径和真正只读连接，不调用会建表或迁移的 Store 构造函数。
列出已有边界、最早曝光日期、次数、股票池和实验身份，给出日期建议后由我确认。"

只读的含义（本模块的硬约束，由 tests/integration/test_decisions_1022_batch3.py 检查）：
  · 只用标准库 sqlite3，URI `mode=ro&immutable=1` —— 不加锁、不建 -wal / -shm、不建表、不迁移；
    不 import app.db 的任何 Store（它们的构造函数会 create_all / 补列）
  · 库旁边有未合并的 -wal 文件时拒绝（immutable 读会漏掉其中已提交的数据）：请先停掉用库的进程
  · 读前读后比对文件哈希、修改时间和目录内容，有任何变化就报错

用法：
    python tools/research/d12_boundary_report.py --db C:\\QuantAgentData\\alphas.db \\
        --assume us_broad_large=2021-01-01 --assume us_tech_large=2020-01-01 [--out report.json]

`--assume`：旧库里研究记录没有数据集与窗口，只有发生的时刻。报告**不把"没记录"当成"没看过"**：
按给定的数据起点，推定研究发生那天之前的数据都已曝光（标 inferred）。起点以实际 `.env` 配置为准。

登记边界不在本工具里做（写库要用户另行确认）：见 `tools/research/d12_register_boundary.py`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

#: D12 要看的三张表；缺了就如实报告，不替它建
D12_TABLES = ("frozen_boundaries", "holdout_usages", "trial_run_state")

#: 惯例口径：Test 段 = 最近 2 年（settings.s_test_freeze_years 的缺省值；实际值以 .env 为准）
CONVENTIONAL_TEST_YEARS = 2


class LiveDbBusy(RuntimeError):
    """库旁边有未合并的 -wal：有进程在用（或异常退出没合并），只读快照会漏数据。"""


def _snapshot(path: Path):
    return (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns,
            sorted(os.listdir(path.parent)))


def _connect_readonly(path: Path) -> sqlite3.Connection:
    wal = Path(str(path) + "-wal")
    if wal.exists() and wal.stat().st_size > 0:
        raise LiveDbBusy(f"{path} 旁边有未合并的 {wal.name}（{wal.stat().st_size} 字节）—— 请先停掉用库的进程再查")
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _tables(con) -> set:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(con, table: str) -> set:
    return {r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}


def _window(test_key: str):
    try:
        a, b = str(test_key).split("..")
        return pd.Timestamp(a), pd.Timestamp(b)
    except (ValueError, TypeError):
        return None


def _sessions_between(a: pd.Timestamp, b: pd.Timestamp) -> List[pd.Timestamp]:
    try:
        from app.core.data_engine.market_calendar import trading_days
        return list(trading_days(a, b))
    except Exception:                                   # 日历不可用：按工作日（报告里会写明）
        return list(pd.bdate_range(a, b))


def _last_session_before(ts: pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(ts).normalize()
    days = [d for d in _sessions_between(day - pd.Timedelta(days=10), day) if d < day]
    return days[-1]


def _next_session_after(day: pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(day).normalize()
    return [d for d in _sessions_between(day + pd.Timedelta(days=1), day + pd.Timedelta(days=10)) if d > day][0]


def _ds(datasets: dict, key: str) -> dict:
    return datasets.setdefault(key, {
        "boundary": None, "exposures": 0, "earliest_exposed_start": None, "latest_exposed_end": None,
        "purposes": [], "experiments": [], "universes": [], "research_windows": [],
        "unrecorded_exposure": None, "suggestions": [],
    })


def _add_unique(lst: list, item) -> None:
    if item not in lst:
        lst.append(item)


def build_report(db_path, assume_data_start: Optional[Dict[str, str]] = None) -> dict:
    """读活库（只读），返回 D12 报告。不写任何东西；读前读后文件不变，否则抛错。"""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"没有这个库文件：{path}")
    before = _snapshot(path)
    con = _connect_readonly(path)
    try:
        rep = _collect(con, assume_data_start or {})
    finally:
        con.close()
    if _snapshot(path) != before:
        raise RuntimeError(f"只读查询前后 {path} 或其目录发生了变化 —— 请检查是否有别的进程在写")
    rep["db"] = {"path": str(path), "sha256": before[0], "read_mode": "sqlite3 mode=ro&immutable=1",
                 "unchanged_after_read": True}
    return rep


def _collect(con, assume: Dict[str, str]) -> dict:
    tables = _tables(con)
    rep: dict = {"tables_missing": [t for t in D12_TABLES if t not in tables], "datasets": {},
                 "trial_total": None, "research_activity": None, "notes": []}
    datasets = rep["datasets"]

    if "frozen_boundaries" in tables:
        cols = _columns(con, "frozen_boundaries")
        for r in con.execute("SELECT * FROM frozen_boundaries"):
            r = dict(r)
            registered = bool(str(r.get("actor") or "").strip() and str(r.get("reason") or "").strip()) \
                if {"actor", "reason"} <= cols else False
            _ds(datasets, r["dataset_key"])["boundary"] = {
                "test_start": r["test_start"], "frozen_by": r.get("frozen_by") or "", "registered": registered}

    if "holdout_usages" in tables:
        cols = _columns(con, "holdout_usages")
        for r in con.execute("SELECT * FROM holdout_usages ORDER BY id"):
            r = dict(r)
            d = _ds(datasets, r["dataset_key"])
            d["exposures"] += 1
            _add_unique(d["purposes"], r.get("purpose") or "")
            if "experiment" in cols and r.get("experiment"):
                _add_unique(d["experiments"], r["experiment"])
            if "universe_digest" in cols and r.get("universe_digest"):
                _add_unique(d["universes"], {"digest": r["universe_digest"], "n": r.get("universe_n")})
            w = _window(r["test_key"])
            if w is not None:
                a, b = str(w[0].date()), str(w[1].date())
                d["earliest_exposed_start"] = min(filter(None, [d["earliest_exposed_start"], a]))
                d["latest_exposed_end"] = max(filter(None, [d["latest_exposed_end"], b]))

    if "trial_run_state" in tables and "context" in _columns(con, "trial_run_state"):
        for r in con.execute("SELECT context FROM trial_run_state WHERE context IS NOT NULL AND context != ''"):
            try:
                c = json.loads(r[0])
            except (TypeError, ValueError):
                rep["notes"].append("有一条运行的研究上下文不是合法 JSON，已跳过")
                continue
            key = c.get("dataset_key") or "synthetic"
            d = _ds(datasets, key)
            if c.get("experiment"):
                _add_unique(d["experiments"], c["experiment"])
            if c.get("universe_digest"):
                _add_unique(d["universes"], {"digest": c["universe_digest"], "n": c.get("universe_n")})
            if c.get("selection_start") and c.get("selection_end"):
                _add_unique(d["research_windows"], {"start": c["selection_start"], "end": c["selection_end"],
                                                    "test_start": c.get("test_start")})

    if "trial_ledger" in tables:
        row = con.execute("SELECT total, updated_at FROM trial_ledger ORDER BY id LIMIT 1").fetchone()
        rep["trial_total"] = int(row[0]) if row is not None else None
    stamps: List[str] = []
    n_alpha = 0
    if "alpha_records" in tables and "created_at" in _columns(con, "alpha_records"):
        rows = [r[0] for r in con.execute("SELECT created_at FROM alpha_records WHERE created_at IS NOT NULL")]
        n_alpha = len(rows)
        stamps += rows
    if "trial_ledger" in tables:
        stamps += [r[0] for r in con.execute("SELECT updated_at FROM trial_ledger WHERE updated_at IS NOT NULL")]
    if stamps:
        rep["research_activity"] = {"first": str(min(stamps))[:19], "last": str(max(stamps))[:19],
                                    "alpha_records": n_alpha, "trial_total": rep["trial_total"]}

    # 旧库：研究记录里没有数据集 / 窗口 → 按用户给定的数据起点推定曝光
    for key, start in assume.items():
        d = _ds(datasets, key)
        if rep["research_activity"] is not None and d["boundary"] is None:
            through = _last_session_before(pd.Timestamp(rep["research_activity"]["last"]))
            d["unrecorded_exposure"] = {
                "from": str(pd.Timestamp(start).date()), "through": str(through.date()), "inferred": True,
                "why": (f"{rep['research_activity']['first']} 起有研究活动（{n_alpha} 条因子记录、累计试验 "
                        f"{rep['trial_total']}），当时没有冻结边界、库里也没记数据集与窗口 —— 推定那天之前的数据都看过")}

    for key, d in datasets.items():
        if key == "synthetic":
            continue
        if d["unrecorded_exposure"] is None and not (d["boundary"] or {}).get("registered"):
            unbounded = [w for w in d["research_windows"] if not w.get("test_start")]
            if unbounded:
                d["unrecorded_exposure"] = {
                    "from": min(w["start"] for w in unbounded), "through": max(w["end"] for w in unbounded),
                    "inferred": False,
                    "why": f"{len(unbounded)} 次研究运行没有冻结 Test 段，选择段覆盖到这些日期"}
        d["suggestions"] = _suggest(d)
    return rep


def _suggest(d: dict) -> List[dict]:
    """给出候选日期，**都要用户确认**。不替用户选。"""
    seen_end = max(filter(None, [d["latest_exposed_end"], (d["unrecorded_exposure"] or {}).get("through"),
                                 *[w["end"] for w in d["research_windows"]]]), default=None)
    out: List[dict] = []
    if seen_end is not None:
        clean = _next_session_after(pd.Timestamp(seen_end))
        out.append({
            "option": "clean", "test_start": str(clean.date()), "needs_user_confirmation": True,
            "already_exposed": False,
            "meaning": (f"Test 段从所有被看过的日期（到 {seen_end}）之后开始：Test 里没有任何被研究看过的数据；"
                        f"代价是 Test 段从很短开始、只随新数据增长")})
        conv = _next_session_after(pd.Timestamp(seen_end) - pd.DateOffset(years=CONVENTIONAL_TEST_YEARS))
        out.append({
            "option": "conventional", "test_start": str(conv.date()), "needs_user_confirmation": True,
            "already_exposed": True,
            "meaning": (f"按惯例口径冻结最近 {CONVENTIONAL_TEST_YEARS} 年（以被看过的最后一天为末端）：Test 段够长，"
                        f"但其中的日期**已被研究看过** —— 结论只能按『已曝光』解读，不能称为未见样本外")})
    if d["earliest_exposed_start"]:
        out.append({
            "option": "earliest_exposed", "test_start": d["earliest_exposed_start"], "needs_user_confirmation": True,
            "already_exposed": True,
            "meaning": (f"以台账里最早被看过的 Test 窗口起点为界：之前的使用都算进这段的次数"
                        f"（已记 {d['exposures']} 次）")})
    if d["boundary"] is not None:
        out.append({
            "option": "keep_existing", "test_start": d["boundary"]["test_start"], "needs_user_confirmation": True,
            "already_exposed": bool(d["exposures"] or d["unrecorded_exposure"]),
            "meaning": ("沿用库里已有的冻结日期" + ("" if d["boundary"]["registered"]
                                              else "（旧代码首次调用写下的，确认后才算登记）"))})
    return out


def render(rep: dict) -> str:
    lines = [f"库：{rep['db']['path']}（{rep['db']['read_mode']}；读前后文件未变）",
             f"缺的表：{', '.join(rep['tables_missing']) or '无'}"]
    if rep["research_activity"]:
        a = rep["research_activity"]
        lines.append(f"研究活动：{a['first']} — {a['last']}；因子记录 {a['alpha_records']} 条；累计试验 {a['trial_total']}")
    for key, d in sorted(rep["datasets"].items()):
        lines += ["", f"== {key}",
                  f"  已有边界：{d['boundary'] or '无'}",
                  f"  台账曝光：{d['exposures']} 次；最早曝光起点 {d['earliest_exposed_start']}；"
                  f"最晚 {d['latest_exposed_end']}；用途 {d['purposes']}",
                  f"  实验：{d['experiments'] or '无记录'}；股票池：{d['universes'] or '无记录'}",
                  f"  研究窗口：{d['research_windows'] or '无记录'}"]
        if d["unrecorded_exposure"]:
            u = d["unrecorded_exposure"]
            lines.append(f"  未入账的曝光：{u['from']} — {u['through']}（{'推定' if u['inferred'] else '有运行记录'}）："
                         f"{u['why']}")
        for s in d["suggestions"]:
            lines.append(f"  建议[{s['option']}] Test 起点 {s['test_start']}：{s['meaning']}（需你确认）")
    for n in rep["notes"]:
        lines.append(f"注：{n}")
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="D12：研究边界与 Test 段曝光的只读报告")
    ap.add_argument("--db", required=True, help="活库文件的明确路径（不读 .env、不用 DATABASE_URL）")
    ap.add_argument("--assume", action="append", default=[], metavar="DATASET=YYYY-MM-DD",
                    help="旧库没有窗口记录时，给定数据集的数据起点（推定曝光用）")
    ap.add_argument("--out", help="把 JSON 报告写到这个文件（写的是报告文件，不是库）")
    args = ap.parse_args(list(argv) if argv is not None else None)
    assume = dict(a.split("=", 1) for a in args.assume)
    rep = build_report(args.db, assume)
    print(render(rep))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
