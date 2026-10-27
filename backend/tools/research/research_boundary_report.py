"""
research_boundary_report.py — 研究边界与 Test 段曝光的**只读**报告（2026-10-22 用户决定 D12）

用户原话："同意先只读查询，不授权写边界。使用明确的活库路径和真正只读连接，不调用会建表或迁移的 Store 构造函数。
列出已有边界、最早曝光日期、次数、股票池和实验身份，给出日期建议后由我确认。"

只读的含义（本模块的硬约束，由 tests/integration/test_research_boundary_tools.py 检查）：
  · 活库上不建任何连接：在**验证过的停写窗口**里把主库逐字节复制到临时目录（窗口两端主库、-wal、-shm、-journal 的
    哈希 / 大小 / 修改时间完全相同，副本哈希等于窗口内主库哈希，否则报错 —— 2026-10-25 审计 R2：
    immutable 不看 WAL、也不检测变化，只能用在真正冻结的副本上），再用标准库 sqlite3 `mode=ro&immutable=1` 读这份副本；
    不建表、不迁移，不 import app.db 的任何 Store（它们的构造函数会 create_all / 补列）
  · 库旁边有未合并的 -wal 文件时拒绝（其中已提交的数据不在主库文件里）：请先停掉用库的进程，不要手动删除 -wal / -shm

用法：
    python tools/research/research_boundary_report.py --db C:\\QuantAgentData\\alphas.db \\
        --assume us_broad_large=2021-01-01 --assume us_tech_large=2020-01-01 [--out report.json]

`--assume`：旧库里研究记录没有数据集与窗口，只有发生的时刻。报告**不把"没记录"当成"没看过"**：
按给定的数据起点，推定研究发生那天之前的数据都已曝光（标 inferred）。起点以实际 `.env` 配置为准。

登记边界不在本工具里做（写库要用户另行确认）：见 `tools/research/register_research_boundary.py`。

曝光汇总（2026-10-25 审计决定 5："汇总所有曝光来源、最后曝光区间和不确定项；确认 10 月 12 日以后是否被用于择因子、
择参数、择方案或解释优劣。无法证明时，从新预登记的真正未见未来窗口开始，而不是选一个方便的过去日期。"）：

    --artifacts DIR   扫描本机留下的研究 / 预演产物：*.json 里的研究上下文（选择段起止、Test 起点）与 *.db 副本里的
                      Test 使用台账、运行上下文 —— 都按已曝光计
    exposure_summary  每个数据集：各来源及其最后曝光日、合计的最后曝光日、**写明的不确定项**（外部文件、手工回测、图表等
                      本工具看不到的东西 —— 看不到 ≠ 没看过）、建议
    --data-end / --candidate   候选 Test 起点的**真实代价**：用系统实际在用的三段切分器，算出各候选下 selection / validate /
                      embargo / Test 的起止与天数（纠正"Test 起点后移会缩短选择段"：同一数据起止与规则下，后移缩短的是 Test）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

#: D12 要看的三张表；缺了就如实报告，不替它建
D12_TABLES = ("frozen_boundaries", "holdout_usages", "trial_run_state")

#: 惯例口径：Test 段 = 最近 2 年（settings.s_test_freeze_years 的缺省值；实际值以 .env 为准）
CONVENTIONAL_TEST_YEARS = 2


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.tasks.snapshot_manifest import SourceBusy as LiveDbBusy  # noqa: E402  库旁边有未合并的 -wal
from app.tasks.snapshot_manifest import (  # noqa: E402
    SourceChanged,  # noqa: E402  停写窗口内库变了
    copy_in_quiet_window,
    frozen_ro,
)

__all__ = ["LiveDbBusy", "SourceChanged", "build_report", "render", "scan_artifacts", "split_cost"]

logger = logging.getLogger(__name__)


def _snapshot(path: Path):
    return (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns,
            sorted(os.listdir(path.parent)))


@contextmanager
def read_frozen(path: Path):
    """停写窗口里复制出一份副本，产出 (副本上的只读连接, 窗口记录)；离开时关连接、删副本。"""
    with tempfile.TemporaryDirectory(prefix="boundary-report-") as tmp:
        window = copy_in_quiet_window(Path(path), Path(tmp) / Path(path).name)
        con = frozen_ro(Path(window["copy"]))
        con.row_factory = sqlite3.Row
        try:
            yield con, window
        finally:
            con.close()


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


def build_report(db_path, assume_data_start: Optional[Dict[str, str]] = None,
                 artifacts: Optional[Iterable[str]] = None, data_end: Optional[str] = None,
                 candidates: Optional[Iterable[str]] = None) -> dict:
    """读活库（只读），返回 D12 报告。不写任何东西；读前读后文件不变，否则抛错。"""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"没有这个库文件：{path}")
    before = _snapshot(path)
    with read_frozen(path) as (con, window):
        rep = _collect(con, assume_data_start or {})
    if _snapshot(path) != before:
        raise RuntimeError(f"只读查询前后 {path} 或其目录发生了变化 —— 请检查是否有别的进程在写")
    scanned = scan_artifacts(artifacts) if artifacts else None
    rep["artifacts"] = scanned
    rep["exposure_summary"] = exposure_summary(rep, scanned)
    if candidates and data_end:
        rep["split_cost"] = {k: split_cost(v, data_end, candidates) for k, v in (assume_data_start or {}).items()}
        rep["split_note"] = SPLIT_NOTE
    rep["db"] = {"path": str(path), "sha256": before[0],
                 "read_mode": "停写窗口逐字节副本 + sqlite3 mode=ro&immutable=1（只读副本）",
                 "window_opened_at": window["window_opened_at"], "window_closed_at": window["window_closed_at"],
                 "unchanged_after_read": True}
    return rep


def _contexts_in(obj, found: list) -> None:
    """JSON 里所有长得像研究上下文的字典（有 dataset_key 与 selection_start / selection_end）。"""
    if isinstance(obj, dict):
        if obj.get("dataset_key") and obj.get("selection_start") and obj.get("selection_end"):
            found.append({k: obj.get(k) for k in ("dataset_key", "selection_start", "selection_end", "test_start",
                                                   "boundary", "purpose")})
        elif obj.get("dataset") and isinstance(obj.get("selection"), list) and len(obj["selection"]) == 2:
            # D2 dry-run 报告的 versions 块：dataset + selection [起, 止]
            found.append({"dataset_key": obj["dataset"], "selection_start": obj["selection"][0],
                          "selection_end": obj["selection"][1], "test_start": obj.get("test_start"),
                          "boundary": obj.get("boundary"), "purpose": obj.get("segment")})
        for v in obj.values():
            _contexts_in(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _contexts_in(v, found)


def scan_artifacts(dirs: Iterable[str]) -> dict:
    """本机研究 / 预演产物里的曝光。JSON：研究上下文；DB 副本：Test 使用台账与运行上下文（停写窗口副本，只读）。"""
    out = {"files": [], "contexts": [], "db_copies": {}}
    for d in dirs:
        root = Path(d)
        if not root.exists():
            out["files"].append({"path": str(root), "error": "不存在"})
            continue
        for f in sorted(root.rglob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                out["files"].append({"path": str(f), "error": f"读不出：{exc}"})
                continue
            found: list = []
            _contexts_in(data, found)
            for c in found:
                out["contexts"].append({**c, "file": str(f), "file_mtime": _mtime(f)})
            out["files"].append({"path": str(f), "contexts": len(found)})
        for f in sorted(root.rglob("*.db")):
            try:
                with read_frozen(f) as (con, _w):
                    out["db_copies"][str(f)] = _collect(con, {})
            except Exception as exc:                                   # 副本被占用 / 不是 SQLite：如实列出
                logger.warning("[boundary] 读不了副本库 %s: %s", f, exc)
                out["files"].append({"path": str(f), "error": f"读不了：{exc}"})
    return out


def _mtime(f: Path) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")


#: 本工具看不到、因而**不能**当作"没看过"的曝光来源
UNPROVABLE = (
    "本机以外 / 本工具没扫描到的文件：旧报告、导出的 CSV、截图与图表",
    "手工回测、笔记本、聊天里贴过的结果",
    "前端页面上看过的回测曲线与指标（不落库）",
    "2026-10-22 之后的预演与研究：只覆盖 --artifacts 给到的目录",
)


def exposure_summary(rep: dict, artifacts: Optional[dict]) -> Dict[str, dict]:
    """每个数据集：曝光来源、最后曝光日、不确定项与建议（不替用户选日期）。"""
    out: Dict[str, dict] = {}
    keys = set(rep["datasets"]) - {"synthetic"}
    if artifacts:
        keys |= {c["dataset_key"] for c in artifacts["contexts"]}
        for db in artifacts["db_copies"].values():
            keys |= set(db.get("datasets", {})) - {"synthetic"}
    for key in sorted(keys):
        d = rep["datasets"].get(key) or {}
        src: List[dict] = []
        if d.get("latest_exposed_end"):
            src.append({"source": "Test 使用台账（活库）", "through": d["latest_exposed_end"], "times": d.get("exposures")})
        for w in d.get("research_windows") or []:
            src.append({"source": "研究运行的选择段（活库）", "through": w["end"], "test_start": w.get("test_start")})
        if d.get("unrecorded_exposure"):
            u = d["unrecorded_exposure"]
            src.append({"source": "推定（库里没记数据集与窗口的研究活动）", "through": u["through"], "why": u["why"]})
        for c in (artifacts or {}).get("contexts", []):
            if c["dataset_key"] == key:
                src.append({"source": f"本机产物 {c['file']}", "through": c["selection_end"],
                            "test_start": c.get("test_start"), "purpose": c.get("purpose"), "at": c["file_mtime"]})
        for path, db in (artifacts or {}).get("db_copies", {}).items():
            dd = (db.get("datasets") or {}).get(key) or {}
            if dd.get("latest_exposed_end"):
                src.append({"source": f"副本库的 Test 使用台账 {path}", "through": dd["latest_exposed_end"],
                            "times": dd.get("exposures")})
            for w in dd.get("research_windows") or []:
                src.append({"source": f"副本库的研究运行 {path}", "through": w["end"], "test_start": w.get("test_start")})
        last = max((x["through"] for x in src if x.get("through")), default=None)
        out[key] = {"sources": src, "last_exposed": last, "uncertain": list(UNPROVABLE),
                    "advice": ("库里与扫描到的产物只能给出**下界**：上面列出的看不到的来源无法排除。证明不了时，"
                               "从新预登记的、今天之后的未来窗口开始 Test（--candidate 给出它的代价），不选一个方便的过去日期；"
                               "已曝光的历史段只能作『已曝光的历史诊断』")}
    return out


def split_cost(data_start: str, data_end: str, candidates: Iterable[str]) -> List[dict]:
    """
    候选 Test 起点在**系统实际在用的**三段切分规则下的代价（同一数据起止）。日期按美股交易日历；
    规则参数读当前配置（s_val_ratio / s_embargo_days …），与研究上下文同一套。
    """
    from app.core.data_engine.data_partitioner import ThreeWayPartitioner
    try:
        from app.config import settings
        kw = dict(test_years=settings.s_test_freeze_years, val_ratio=settings.s_val_ratio,
                  embargo_days=settings.s_embargo_days, min_selection_days=settings.s_min_selection_days,
                  max_test_share=settings.s_max_test_share)
    except Exception as exc:
        logger.error("[boundary] 读不到三段切分配置，用保守默认: %s", exc)
        kw = dict(test_years=2.0, val_ratio=0.25, embargo_days=20, min_selection_days=378, max_test_share=0.35)
    dates = pd.DatetimeIndex(_sessions_between(pd.Timestamp(data_start), pd.Timestamp(data_end)))
    frame = pd.DataFrame({"x": 0.0}, index=dates)
    rows = []
    for c in candidates:
        ts = pd.Timestamp(c)
        try:
            sp = ThreeWayPartitioner(**kw).partition({"close": frame}, test_start=ts)
        except ValueError as exc:
            rows.append({"test_start": str(ts.date()), "error": str(exc)})
            continue
        sel = sp.selection["close"].index
        tst = sp.test["close"].index
        rows.append({"test_start": str(ts.date()), "selection": [str(sel[0].date()), str(sel[-1].date())],
                     "n_selection": len(sel), "n_train": sp.n_is, "n_validate": sp.n_val,
                     "embargo_days": sp.embargo_days, "n_test": sp.n_test,
                     "test": [str(tst[0].date()), str(tst[-1].date())] if len(tst) else None})
    return rows


SPLIT_NOTE = ("同一数据起止与切分规则下，把 Test 起点往后移：selection（训练 + embargo + 验证）**变长**，Test **变短**；"
              "选择段历史不会因此变少。新的干净 Test 起初很短，只随新数据增长 —— 工程联调不必为此等两年，"
              "但统计门槛不因 Test 短而降低。预热：因子最长窗口（如 252 日）要在选择段起点之前已有数据，只有评估日计分；"
              "c2 标签是开盘(T+1)→开盘(T+2)，各段之间的 embargo（上面的天数）同时起到清除跨段标签的作用。")


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
        after = _next_session_after(pd.Timestamp(seen_end))
        # 2026-10-25 审计决定 5："不能直接把 2026-10-12 盖章为干净" —— 库里记录到的曝光只是下界
        out.append({
            "option": "after_recorded_exposure", "test_start": str(after.date()), "needs_user_confirmation": True,
            "already_exposed": None,
            "meaning": (f"从**库里记录到的**曝光（到 {seen_end}）之后开始。记录之外的曝光（曝光汇总里的不确定项、"
                        f"之后的预演与研究）无法排除 —— 不能据此称为干净，要先确认这段之后没有被用来择因子 / 参数 / 方案")})
        future = _next_session_after(pd.Timestamp.now().normalize())
        out.append({
            "option": "future_window", "test_start": str(future.date()), "needs_user_confirmation": True,
            "already_exposed": False,
            "meaning": ("从今天之后的下一个交易日开始、先预登记：在登记之前不可能有人看过。代价是 Test 从 0 开始、"
                        "只随新数据增长；统计门槛不因此降低")})
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
    for key, e in (rep.get("exposure_summary") or {}).items():
        lines += ["", f"== 曝光汇总 {key}：已知的最后曝光日 {e['last_exposed']}（下界）"]
        lines += [f"  · {x['source']}：到 {x.get('through')}" for x in e["sources"]]
        lines += [f"  ? 看不到的来源：{u}" for u in e["uncertain"]]
        lines.append(f"  建议：{e['advice']}")
    for key, rows in (rep.get("split_cost") or {}).items():
        lines += ["", f"== 候选 Test 起点的代价 {key}"]
        for r in rows:
            lines.append(f"  {r['test_start']}：" + (r["error"] if "error" in r else
                         f"selection {r['selection'][0]}–{r['selection'][1]}（{r['n_selection']} 天；训练 {r['n_train']} / "
                         f"验证 {r['n_validate']} / embargo {r['embargo_days']}）；Test {r['n_test']} 天"))
        lines.append(f"  说明：{rep.get('split_note', '')}")
    for n in rep["notes"]:
        lines.append(f"注：{n}")
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="D12：研究边界与 Test 段曝光的只读报告")
    ap.add_argument("--db", required=True, help="活库文件的明确路径（不读 .env、不用 DATABASE_URL）")
    ap.add_argument("--assume", action="append", default=[], metavar="DATASET=YYYY-MM-DD",
                    help="旧库没有窗口记录时，给定数据集的数据起点（推定曝光用）")
    ap.add_argument("--out", help="把 JSON 报告写到这个文件（写的是报告文件，不是库）")
    ap.add_argument("--artifacts", action="append", default=[], metavar="DIR",
                    help="扫描这个目录里的研究 / 预演产物（*.json 研究上下文、*.db 副本），按已曝光计")
    ap.add_argument("--data-end", help="候选 Test 起点代价的数据末日（与 --assume 的起点配对）")
    ap.add_argument("--candidate", action="append", default=[], metavar="YYYY-MM-DD", help="候选 Test 起点")
    args = ap.parse_args(list(argv) if argv is not None else None)
    assume = dict(a.split("=", 1) for a in args.assume)
    rep = build_report(args.db, assume, artifacts=args.artifacts, data_end=args.data_end, candidates=args.candidate)
    print(render(rep))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
