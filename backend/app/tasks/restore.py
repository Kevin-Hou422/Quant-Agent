"""
restore.py — 把一份备份快照恢复到隔离目录并核对（2026-10-22 用户决定第 21 条）

用户原话："做隔离恢复演练……合成恢复通过不等于活库恢复已验证。"

    python -m app.tasks.restore <snapshot 目录> <空的目标目录>

只做隔离恢复：
    · 目标不能是活库所在目录、不能在活 PIT 目录里，也不能把活库 / 活 PIT 包在里面 —— 拒绝
    · 目标目录已有内容 —— 拒绝（不覆盖任何东西）
    · 恢复出来的每个库都跑 `PRAGMA integrity_check`
把恢复结果换回活库是另一回事：要人看过 `verify_restore` 的差异后另行确认（D2），本模块不做。

`verify_restore` 对照**冻结的**源副本（快照文件本身，或停写窗口里复制出的副本；immutable 只用在这种不会再变的文件上）
与恢复出的库：完整性、表是否齐、逐表行数、主键与内容摘要、业务不变量（运行槽、订单认领 / 成交 / 熔断、正式证据），
见 `app.tasks.snapshot_manifest`（2026-10-25 审计 R2：只比表和行数证明不了恢复的是同一份数据）；PIT 文件逐字节（sha256）比对。
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

PIT_ARCHIVE = "pit_store.zip"
PIT_DIRNAME = "pit_store"


class RestoreRefused(RuntimeError):
    """恢复目标不安全（活路径 / 非空目录）。"""


def _live_paths() -> List[Path]:
    from app.config import settings
    from app.tasks.backup import _sqlite_path_from_url
    out = []
    for url in (getattr(settings, "database_url", ""), getattr(settings, "scheduler_db_url", "")):
        p = _sqlite_path_from_url(url or "")
        if p is not None:
            out.append(p.parent)
    out.append(Path(getattr(settings, "pit_store_dir", "pit_store")).expanduser().resolve())
    return out


def _overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def check_target(target: Path) -> None:
    t = Path(target).expanduser().resolve()
    for live in _live_paths():
        if _overlaps(t, live):
            raise RestoreRefused(f"恢复目标 {t} 与活数据路径 {live} 重叠 —— 只做隔离恢复，不碰活库 / 活 PIT")
    if t.exists() and (not t.is_dir() or any(t.iterdir())):
        raise RestoreRefused(f"恢复目标 {t} 不是空的 —— 不覆盖任何已有内容，请指定一个新的空目录")


def _ro(db: Path) -> sqlite3.Connection:
    """恢复出的库（本模块自己的产物、没有别的连接）用普通只读连接。"""
    return sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)


def _integrity(db: Path) -> str:
    con = _ro(db)
    try:
        return str(con.execute("PRAGMA integrity_check").fetchone()[0])
    finally:
        con.close()


def restore_snapshot(snapshot_dir: Path, target_dir: Path) -> dict:
    """把快照里的库与 PIT 解到 `target_dir`（必须是空的隔离目录）。返回恢复出的文件与完整性检查结果。"""
    snap = Path(snapshot_dir)
    check_target(target_dir)
    dbs = sorted(snap.glob("*.db"))
    if not dbs:
        raise RestoreRefused(f"快照 {snap} 里没有数据库文件")
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    integrity: Dict[str, str] = {}
    for db in dbs:
        dest = target / db.name
        shutil.copy2(db, dest)
        integrity[db.name] = _integrity(dest)
    pit = None
    if (snap / PIT_ARCHIVE).exists():
        pit = target / PIT_DIRNAME
        shutil.unpack_archive(str(snap / PIT_ARCHIVE), str(pit), "zip")
    return {"ok": all(v == "ok" for v in integrity.values()), "target": str(target),
            "databases": [d.name for d in dbs], "integrity": integrity,
            "pit_dir": str(pit) if pit is not None else None}


def _file_digests(root: Optional[Path]) -> Dict[str, str]:
    if root is None or not Path(root).exists():
        return {}
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def verify_restore(frozen_db: Path, restored_db: Path,
                   source_pit: Optional[Path] = None, restored_pit: Optional[Path] = None) -> dict:
    """
    冻结源副本与恢复库逐项对照（`snapshot_manifest.compare_descriptions`）；PIT 文件逐字节对照。
    `frozen_db` 必须是不会再变的文件（快照文件 / 停写窗口副本）—— 不要传活库。ok = 完整性 ok 且没有任何差异。
    """
    from app.tasks.snapshot_manifest import compare_descriptions, describe_db
    out = compare_descriptions(describe_db(Path(frozen_db)), describe_db(Path(restored_db)))
    s_pit, d_pit = _file_digests(source_pit), _file_digests(restored_pit)
    pit_bad = sorted(k for k, h in s_pit.items() if d_pit.get(k) != h)
    out.update(ok=bool(out["ok"] and not pit_bad), frozen_source=str(frozen_db),
               pit_files_checked=len(s_pit), pit_mismatches=pit_bad)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="把备份快照恢复到一个空的隔离目录（不碰活库）")
    ap.add_argument("snapshot")
    ap.add_argument("target")
    a = ap.parse_args(argv)
    try:
        out = restore_snapshot(Path(a.snapshot), Path(a.target))
    except RestoreRefused as exc:
        print(f"拒绝：{exc}")
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
