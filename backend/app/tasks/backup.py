"""
backup.py — 每日一致性快照备份（长期可行的数据安全方案）

为什么必须有
------------
**前向数据不可再生**：券商只给历史 K 线；"前向"的身份来自你**当时就记下了**。
磁盘一坏，几个月的前向证据归零，TR.4 的 →ACTIVE 门直接回到起点。
交易台账（因子/审批/IC/成交）同样是不可重建的审计资产。

设计要点
--------
1. **一致性快照，不是复制文件**：SQLite 用 `VACUUM INTO`（3.27+，读事务内生成完整单文件副本），
   绝不 `copy` 活库——WAL 模式下 `.db`/`.db-wal`/`.db-shm` 三件套必须互相一致，
   逐文件复制/云同步会产出损坏的副本（这正是"活库不能放在 OneDrive 里"的原因）。
2. **快照后立即校验**：对副本跑 `PRAGMA integrity_check`，坏副本当场判失败，不静默留下坏备份。
3. **快照是静态文件 → 放进云盘是安全的**（与活库不同）。
4. **保留 N 份**：按时间倒序保留，旧的删除。
"""

from __future__ import annotations

import logging
import shutil
import sqlite3

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BackupResult:
    ok:            bool
    snapshot_dir:  str = ""
    items:         List[str] = field(default_factory=list)
    bytes_written: int = 0
    pruned:        List[str] = field(default_factory=list)
    errors:        List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "snapshot_dir": self.snapshot_dir, "items": self.items,
                "bytes": self.bytes_written, "pruned": self.pruned, "errors": self.errors}


def _sqlite_path_from_url(url: str) -> Optional[Path]:
    """sqlite:///./alphas.db → Path。非 sqlite 返回 None。"""
    if not url.startswith("sqlite"):
        return None
    raw = url.split("///", 1)[-1]
    return Path(raw).expanduser().resolve()


def snapshot_sqlite(db_path: Path, dest: Path) -> int:
    """
    用 `VACUUM INTO` 生成**一致性**副本并校验完整性。返回字节数；失败抛异常。
    （相较直接 copy：copy 活库在 WAL 下可能得到不一致/损坏的副本。）
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("VACUUM INTO ?", (str(dest),))
    finally:
        con.close()

    # 校验副本：坏备份比没备份更危险（会让人以为有救）
    chk = sqlite3.connect(str(dest))
    try:
        res = chk.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        chk.close()
    if res != "ok":
        raise RuntimeError(f"快照完整性校验失败 {dest.name}: {res}")
    return dest.stat().st_size


def snapshot_tree(src_dir: Path, dest_base: Path) -> int:
    """把目录（PIT parquet 树）打包成 zip。返回字节数。"""
    dest_base.parent.mkdir(parents=True, exist_ok=True)
    out = shutil.make_archive(str(dest_base), "zip", root_dir=str(src_dir))
    return Path(out).stat().st_size


def prune_snapshots(backup_root: Path, keep: int) -> List[str]:
    """只保留最近 keep 个快照目录（按名字倒序＝时间倒序）。"""
    if keep <= 0 or not backup_root.exists():
        return []
    dirs = sorted([d for d in backup_root.iterdir() if d.is_dir() and d.name.startswith("snap-")],
                  reverse=True)
    pruned = []
    for d in dirs[keep:]:
        try:
            shutil.rmtree(d)
            pruned.append(d.name)
        except Exception as exc:
            logger.warning("[backup] 删除旧快照失败 %s: %s", d, exc)
    return pruned


def run_daily_backup(backup_dir: Optional[str] = None,
                     keep: Optional[int] = None) -> BackupResult:
    """
    每日快照：主库 + 调度库 + PIT 全量。放进 `backup_dir`（建议指向云盘同步目录——
    快照是静态文件，同步是安全的；活库则必须留在非同步盘）。
    任一环节失败都记进 errors 并让 ok=False（不静默"成功"）。
    """
    from app.config import settings

    root = Path(backup_dir or getattr(settings, "backup_dir", "") or "backups").expanduser()
    keep_n = int(keep if keep is not None else getattr(settings, "backup_keep_n", 14))
    stamp = datetime.now(timezone.utc).strftime("snap-%Y%m%dT%H%M%SZ")
    snap = root / stamp
    res = BackupResult(ok=True, snapshot_dir=str(snap))

    # 1) 主库（因子/审批/IC/成交/策略配置 —— 审计资产）
    main_db = _sqlite_path_from_url(getattr(settings, "database_url", ""))
    if main_db and main_db.exists():
        try:
            n = snapshot_sqlite(main_db, snap / f"{main_db.name}")
            res.items.append(main_db.name); res.bytes_written += n
        except Exception as exc:
            res.ok = False; res.errors.append(f"main_db: {exc}")
    else:
        # 主库是核心审计资产：备不了就是**失败**，绝不能报成功让人以为有备份
        res.ok = False
        res.errors.append("main_db: 不存在或非 sqlite")

    # 2) 调度库（可重建，但一并备份便于整机恢复）
    sch_db = _sqlite_path_from_url(getattr(settings, "scheduler_db_url", ""))
    if sch_db and sch_db.exists():
        try:
            n = snapshot_sqlite(sch_db, snap / sch_db.name)
            res.items.append(sch_db.name); res.bytes_written += n
        except Exception as exc:
            logger.warning("[backup] 调度库快照失败（不致命）: %s", exc)
            res.errors.append(f"scheduler_db: {exc}")

    # 3) PIT —— **最不可再生**的部分（前向数据买不回来）
    pit = Path(getattr(settings, "pit_store_dir", "pit_store")).expanduser()
    if pit.exists() and any(pit.rglob("*.parquet")):
        try:
            n = snapshot_tree(pit, snap / "pit_store")
            res.items.append("pit_store.zip"); res.bytes_written += n
        except Exception as exc:
            res.ok = False; res.errors.append(f"pit: {exc}")
    else:
        logger.info("[backup] PIT 为空，跳过（尚未开始前向积累）")

    res.pruned = prune_snapshots(root, keep_n)
    logger.info("[backup] 快照 %s | 项目=%s | %.1f KB | 清理旧快照=%d | ok=%s",
                stamp, res.items, res.bytes_written / 1024, len(res.pruned), res.ok)
    return res
