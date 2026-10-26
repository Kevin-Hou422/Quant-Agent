"""
snapshot_manifest.py — 一致快照的取得方式、清单与恢复核验（2026-10-25 审计 R2）

审计原话："检查与实际读取之间没有确保源库持续静止：`immutable=1` 忽略 WAL 和变化检测；文件指纹只覆盖主库内容/属性
及目录名称，没有覆盖整个 WAL 提交状态；恢复比对主要是表和行数。"复现：预检查后另一个连接提交第 2 行，快照只有 1 行，
工具却返回 live_unchanged=true、restore.ok=true。
要求："明确选择完整停写窗口并验证，或使用能正确处理 WAL 的 SQLite 一致性读取/backup；记录精确快照时间、数据与文件清单。
……恢复核验除 integrity_check、表和行数外，核对关键主键、内容摘要、订单认领/成交/熔断/正式证据等业务不变量。
只对真正冻结的副本使用 immutable。不要靠手动删除 WAL/SHM 实现'一致'。"

两种取得方式，各自适用的场合写死：

    产品备份（`app.tasks.backup`，应用自己的库）  普通连接上 `VACUUM INTO`：在一个读事务里生成副本，WAL 里已提交的帧
                                                  都在内（SQLite 自己处理 WAL）。
    活库只读预览（不允许在活库旁建 -wal / -shm）  **验证过的停写窗口**（`copy_in_quiet_window`）：窗口开始与结束时
                                                  主库、-wal、-shm、-journal 的内容哈希 / 大小 / 修改时间完全相同，
                                                  且主库副本的哈希等于窗口内的主库哈希；-wal 非空或任何一项变了 → 作废、
                                                  删副本、报错。之后只在这份**冻结的副本**上用 immutable。

清单（`describe_db`）：每张表的行数、按主键的内容摘要（与行顺序无关）、schema 摘要，加业务不变量
（运行槽、订单认领 / 成交 / 熔断、正式证据）。恢复核验（`compare_descriptions`）逐项对照冻结副本与恢复出的库。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

#: 主库旁边会与它一起构成提交状态的文件
SIDE_FILES = ("-wal", "-shm", "-journal")

#: 运行槽：任何时刻同一账户 / book 至多一份（R7）；旧版 active 也算
RUNNING_STATUSES = ("integration", "paper_running", "verified_active", "active")


class SourceBusy(RuntimeError):
    """源库旁边有未合并的 -wal / 回滚日志：有进程在用（或异常退出没合并）。"""


class SourceChanged(RuntimeError):
    """停写窗口内源库或它的 -wal / -shm 变了：副本不能代表任何一个确定的提交状态。"""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_state(path: Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    st = p.stat()
    return {"exists": True, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": _sha(p)}


def db_files_state(db: Path) -> dict:
    """主库与旁边的 -wal / -shm / -journal 的完整状态（内容哈希、大小、修改时间），加目录里的文件名。"""
    db = Path(db)
    return {"db": file_state(db), **{s: file_state(Path(str(db) + s)) for s in SIDE_FILES},
            "directory": sorted(p.name for p in db.parent.iterdir())}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _copy_bytes(src: Path, dest: Path) -> None:
    shutil.copyfile(src, dest)


def copy_in_quiet_window(db: Path, dest: Path) -> dict:
    """
    验证过的停写窗口里逐字节复制主库。返回窗口记录（开始 / 结束时刻与两端的文件状态）。
    -wal 非空或有回滚日志 → SourceBusy（不读）；窗口内任何一项变了 → 删掉副本，SourceChanged。
    """
    db, dest = Path(db), Path(dest)
    before = db_files_state(db)
    if before["-wal"].get("size", 0) > 0 or before["-journal"]["exists"]:
        busy = "-wal" if before["-wal"].get("size", 0) > 0 else "-journal"
        raise SourceBusy(f"{db} 旁边有未合并的 {db.name}{busy}（{before[busy].get('size')} 字节）—— "
                         f"先停掉用库的进程再做快照（不要手动删除这些文件）")
    opened = _now()
    dest.parent.mkdir(parents=True, exist_ok=True)
    _copy_bytes(db, dest)
    copy_sha = _sha(dest)
    after = db_files_state(db)
    closed = _now()
    changed = sorted(k for k in before if before[k] != after[k])
    if changed or copy_sha != before["db"]["sha256"]:
        dest.unlink(missing_ok=True)
        raise SourceChanged(f"停写窗口内 {db} 有变化（{', '.join(changed) or '副本与窗口内主库哈希不同'}）"
                            f"—— 副本作废；先停掉写库的进程再做")
    return {"method": "quiet_window_copy", "window_opened_at": opened, "window_closed_at": closed,
            "source": str(db), "copy": str(dest), "copy_sha256": copy_sha, "before": before, "after": after}


def frozen_ro(db: Path) -> sqlite3.Connection:
    """只给**冻结的副本**用：immutable 不加锁、不看 WAL，正因为如此不能拿来读还可能在变的库。"""
    return sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro&immutable=1", uri=True)


def _tables(con) -> List[str]:
    return [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def _cols(con, table: str) -> List[str]:
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]


def _row_key(row) -> str:
    return json.dumps(list(row), ensure_ascii=False, default=lambda v: v.hex() if isinstance(v, bytes) else repr(v))


def table_digest(con, table: str) -> dict:
    """行数、主键集合摘要与全部内容摘要 —— 都与行的物理顺序无关（VACUUM 会重排没有整数主键的表的 rowid）。"""
    cols = _cols(con, table)
    pk = [r[1] for r in sorted(con.execute(f'PRAGMA table_info("{table}")'), key=lambda r: r[5]) if r[5]]
    rows = [_row_key(r) for r in con.execute(f'SELECT {", ".join(chr(34) + c + chr(34) for c in cols)} FROM "{table}"')]
    keys = ([_row_key(r) for r in con.execute(f'SELECT {", ".join(chr(34) + c + chr(34) for c in pk)} FROM "{table}"')]
            if pk else [])
    return {"rows": len(rows), "primary_key": pk,
            "keys_sha256": hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest() if pk else None,
            "content_sha256": hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()}


def _q(con, sql: str, *args) -> list:
    return [tuple(r) for r in con.execute(sql, args)]


def business_invariants(con) -> dict:
    """
    恢复后必须原样存在的业务事实，与"哪些已经违反了"。表不在（旧库 / 只建了一部分）就不报那一项。
        运行槽       处于运行状态的策略（integration / paper_running / verified_active / 旧版 active）—— 多于一份即违反
        订单认领     每张订单的 client_id、状态、成交量；在途（PENDING_SUBMIT / SUBMITTED）的单
        成交         每笔成交增量；增量合计与订单成交量不同的单（列出）；没有对应订单的成交即违反
        熔断         全平熔断的状态原文
        正式证据     每个策略的正式前向天数与最后一天、证据行数与规格哈希、信号存档的最后一天
    """
    tables = set(_tables(con))
    facts: Dict[str, object] = {}
    violations: List[str] = []
    if "strategy_configs" in tables:
        running = _q(con, "SELECT id, status FROM strategy_configs WHERE status IN (%s) ORDER BY id"
                     % ",".join("?" * len(RUNNING_STATUSES)), *RUNNING_STATUSES)
        facts["running_strategies"] = [list(r) for r in running]
        if len(running) > 1:
            violations.append(f"运行槽不唯一：{[r[0] for r in running]} 同时处于运行状态")
        facts["strategy_status"] = [list(r) for r in _q(con, "SELECT id, status FROM strategy_configs ORDER BY id")]
    if "exec_orders" in tables:
        orders = _q(con, "SELECT client_id, status, dealt_qty, qty FROM exec_orders ORDER BY client_id")
        facts["orders"] = [list(r) for r in orders]
        facts["orders_in_flight"] = [r[0] for r in orders if r[1] in ("PENDING_SUBMIT", "SUBMITTED")]
        if "exec_fills" in tables:
            fills = dict(_q(con, "SELECT client_id, SUM(qty) FROM exec_fills GROUP BY client_id"))
            # 订单成交量与成交增量由同一轮对账写入；不一致如实列出（是否异常要看那一轮对账），不在这里判违反
            facts["fill_totals_differ"] = [[cid, dealt, abs(float(fills.get(cid) or 0.0))]
                                           for cid, _st, dealt, _qty in orders
                                           if abs(abs(float(fills.get(cid) or 0.0)) - float(dealt or 0.0)) > 1e-6]
            orphan = sorted(set(fills) - {r[0] for r in orders})
            if orphan:
                violations.append(f"有成交没有对应订单：{orphan}")
            facts["fills"] = [list(r) for r in _q(con, "SELECT client_id, cum_qty, qty, price FROM exec_fills "
                                                       "ORDER BY client_id, cum_qty")]
    if "exec_state" in tables:
        ks = _q(con, "SELECT value FROM exec_state WHERE key = 'kill_switch'")
        facts["kill_switch"] = ks[0][0] if ks else None
        facts["exec_state_keys"] = [r[0] for r in _q(con, "SELECT key FROM exec_state ORDER BY key")]
    if "forward_days" in tables:
        facts["formal_days"] = [list(r) for r in _q(
            con, "SELECT strategy_id, spec_hash, SUM(formal), COUNT(*), MAX(decision_date) FROM forward_days "
                 "GROUP BY strategy_id, spec_hash ORDER BY strategy_id, spec_hash")]
        dup = _q(con, "SELECT strategy_id, decision_date, COUNT(*) FROM forward_days "
                      "GROUP BY strategy_id, decision_date HAVING COUNT(*) > 1")
        if dup:
            violations.append(f"同一策略同一天有多条正式记录：{dup}")
    if "strategy_evidence" in tables:
        facts["evidence"] = [list(r) for r in _q(
            con, "SELECT strategy_id, spec_hash, COUNT(*), MAX(date) FROM strategy_evidence "
                 "GROUP BY strategy_id, spec_hash ORDER BY strategy_id, spec_hash")]
    if "signal_archive" in tables:
        facts["signal_archive"] = [list(r) for r in _q(
            con, "SELECT book_key, COUNT(*), MAX(asof) FROM signal_archive GROUP BY book_key ORDER BY book_key")]
    if "frozen_boundaries" in tables:
        facts["frozen_boundaries"] = [list(r) for r in _q(
            con, "SELECT dataset_key, test_start, frozen_by FROM frozen_boundaries ORDER BY dataset_key")]
    return {"facts": facts, "violations": violations}


def describe_db(db: Path) -> dict:
    """冻结副本的清单：完整性、页数、schema 摘要、逐表内容摘要、业务不变量。"""
    con = frozen_ro(db)
    try:
        schema = "\n".join(str(r[0]) for r in con.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"))
        out = {"file": str(db), "sha256": _sha(Path(db)), "sqlite_version": sqlite3.sqlite_version,
               "integrity": str(con.execute("PRAGMA integrity_check").fetchone()[0]),
               "foreign_key_violations": len(con.execute("PRAGMA foreign_key_check").fetchall()),
               "page_count": int(con.execute("PRAGMA page_count").fetchone()[0]),
               "schema_sha256": hashlib.sha256(schema.encode()).hexdigest(),
               "tables": {t: table_digest(con, t) for t in _tables(con)},
               "invariants": business_invariants(con)}
    finally:
        con.close()
    return out


def compare_descriptions(frozen: dict, restored: dict) -> dict:
    """冻结副本与恢复出的库逐项对照。ok = 恢复库完整、表齐、行数 / 主键 / 内容 / 业务不变量全部相同。"""
    ft, rt = frozen["tables"], restored["tables"]
    missing = sorted(set(ft) - set(rt))
    rows = {t: {"source": d["rows"], "restored": rt[t]["rows"]} for t, d in ft.items()
            if t in rt and rt[t]["rows"] != d["rows"]}
    keys = sorted(t for t, d in ft.items() if t in rt and rt[t]["keys_sha256"] != d["keys_sha256"])
    content = sorted(t for t, d in ft.items() if t in rt and rt[t]["content_sha256"] != d["content_sha256"])
    inv = sorted(k for k in set(frozen["invariants"]["facts"]) | set(restored["invariants"]["facts"])
                 if frozen["invariants"]["facts"].get(k) != restored["invariants"]["facts"].get(k))
    ok = (restored["integrity"] == "ok" and not missing and not rows and not keys and not content and not inv)
    return {"ok": ok, "integrity": {"main": restored["integrity"]}, "tables_checked": len(ft),
            "missing_tables": missing, "row_count_mismatches": rows, "key_mismatches": keys,
            "content_mismatches": content, "invariant_mismatches": inv,
            "invariant_violations": list(restored["invariants"]["violations"])}


def write_manifest(directory: Path, manifest: dict, name: str = "manifest.json") -> Path:
    p = Path(directory) / name
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return p


def read_manifest(directory: Path, name: str = "manifest.json") -> Optional[dict]:
    p = Path(directory) / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
