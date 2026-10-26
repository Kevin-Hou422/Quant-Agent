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
5. **清单**（2026-10-25 审计 R2）：快照目录里写 `manifest.json` —— 每一项的取得时刻、方式、源路径与快照文件哈希、
   逐表行数 / 主键 / 内容摘要与业务不变量（`app.tasks.snapshot_manifest`）。各项**逐个**取得，不在同一个事务里：
   每个库各自一致；跨库 / PIT 的一致只在取快照期间调度停着时成立，清单如实记 `cross_item_consistent`。
6. **结果要传播**（2026-10-25 审计 R4："daily_backup_job 在备份失败时只记 error 日志而正常返回"）：每次备份
   （成功或失败）都写 `<backup_dir>/last_backup.json`；`backup_health` 读它给预检（健康门），调度任务失败时发
   backup_failed 告警，命令行 `python -m app.tasks.backup` 失败退出码 1。
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import sys

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

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


LAST_BACKUP = "last_backup.json"
#: 最近一次成功备份最多多久以前（每日任务 + 一次错过的余量）
BACKUP_MAX_AGE = timedelta(hours=48)


def backup_health(backup_dir: Optional[str] = None, now: Optional[datetime] = None) -> Tuple[bool, str, bool]:
    """(ok, 说明, 是否阻塞)。从没备份过 → 不通过但不阻塞（首次部署）；最近一次失败 / 太久没成功 → 阻塞。"""
    from app.config import settings
    root = Path(backup_dir or getattr(settings, "backup_dir", "") or "backups").expanduser()
    now = now or datetime.now(timezone.utc)
    p = root / LAST_BACKUP
    if not p.exists():
        return False, f"{root} 里还没有备份记录（首次部署：先跑一次 python -m app.tasks.backup）", False
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
        at = datetime.fromisoformat(rec["at"])
    except Exception as exc:
        logger.error("[backup] 备份记录 %s 读不出来: %s", p, exc)
        return False, f"备份记录 {p} 读不出来：{exc}", True
    if not rec.get("ok"):
        return False, f"最近一次备份（{rec['at']}）失败：{'; '.join(rec.get('errors') or [])}", True
    if now - at > BACKUP_MAX_AGE:
        return False, f"最近一次成功备份在 {rec['at']}，超过 {BACKUP_MAX_AGE} 没有新的备份", True
    return True, f"最近一次备份 {rec['at']} 成功：{rec.get('snapshot_dir')}", True


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


def _manifest_item(src: Path, copy: Path) -> dict:
    from app.tasks.snapshot_manifest import describe_db
    return {"source": str(src), "method": "vacuum_into", "captured_at": datetime.now(timezone.utc).isoformat(
        timespec="microseconds"), "description": describe_db(copy)}


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
                     keep: Optional[int] = None, *,
                     database_url: Optional[str] = None, scheduler_db_url: Optional[str] = None,
                     pit_store_dir: Optional[str] = None) -> BackupResult:
    """
    每日快照：主库 + 调度库 + PIT 全量。放进 `backup_dir`（建议指向云盘同步目录——
    快照是静态文件，同步是安全的；活库则必须留在非同步盘）。
    任一环节失败都记进 errors 并让 ok=False（不静默"成功"）。
    database_url / scheduler_db_url / pit_store_dir：缺省取 settings（恢复演练用它们指向隔离的合成数据）。
    """
    from app.config import settings

    root = Path(backup_dir or getattr(settings, "backup_dir", "") or "backups").expanduser()
    keep_n = int(keep if keep is not None else getattr(settings, "backup_keep_n", 14))
    stamp = datetime.now(timezone.utc).strftime("snap-%Y%m%dT%H%M%SZ")
    snap = root / stamp
    res = BackupResult(ok=True, snapshot_dir=str(snap))
    manifest: dict = {"snapshot": stamp, "started_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                      "items": {}, "cross_item_consistent": False,
                      "consistency": "每个 SQLite 库在各自的读事务里一致（VACUUM INTO）；各项依次取得，不是同一时刻"}

    # 1) 主库（因子/审批/IC/成交/策略配置 —— 审计资产）
    main_db = _sqlite_path_from_url(database_url or getattr(settings, "database_url", ""))
    if main_db and main_db.exists():
        try:
            n = snapshot_sqlite(main_db, snap / f"{main_db.name}")
            res.items.append(main_db.name); res.bytes_written += n
            manifest["items"][main_db.name] = _manifest_item(main_db, snap / main_db.name)
        except Exception as exc:
            res.ok = False; res.errors.append(f"main_db: {exc}")
    else:
        # 主库是核心审计资产：备不了就是**失败**，绝不能报成功让人以为有备份
        res.ok = False
        res.errors.append("main_db: 不存在或非 sqlite")

    # 2) 调度库（可重建，但一并备份便于整机恢复）
    sch_db = _sqlite_path_from_url(scheduler_db_url or getattr(settings, "scheduler_db_url", ""))
    if sch_db and sch_db.exists():
        try:
            n = snapshot_sqlite(sch_db, snap / sch_db.name)
            res.items.append(sch_db.name); res.bytes_written += n
            manifest["items"][sch_db.name] = _manifest_item(sch_db, snap / sch_db.name)
        except Exception as exc:
            logger.warning("[backup] 调度库快照失败（不致命）: %s", exc)
            res.errors.append(f"scheduler_db: {exc}")

    # 3) PIT —— **最不可再生**的部分（前向数据买不回来）
    pit = Path(pit_store_dir or getattr(settings, "pit_store_dir", "pit_store")).expanduser()
    if pit.exists() and any(pit.rglob("*.parquet")):
        try:
            n = snapshot_tree(pit, snap / "pit_store")
            res.items.append("pit_store.zip"); res.bytes_written += n
            from app.tasks.restore import _file_digests
            manifest["items"]["pit_store.zip"] = {"source": str(pit), "captured_at": datetime.now(
                timezone.utc).isoformat(timespec="microseconds"), "files": _file_digests(pit)}
        except Exception as exc:
            res.ok = False; res.errors.append(f"pit: {exc}")
    else:
        logger.info("[backup] PIT 为空，跳过（尚未开始前向积累）")

    snap.mkdir(parents=True, exist_ok=True)
    if snap.exists():
        try:
            from app.tasks.snapshot_manifest import write_manifest
            manifest.update(finished_at=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                            ok=res.ok, errors=list(res.errors))
            write_manifest(snap, manifest)
        except Exception as exc:
            logger.error("[backup] 清单写入失败: %s", exc)
            res.ok = False; res.errors.append(f"manifest: {exc}")
    res.pruned = prune_snapshots(root, keep_n)
    try:
        (root / LAST_BACKUP).write_text(json.dumps(
            {**res.to_dict(), "at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as exc:
        logger.error("[backup] last_backup.json 写入失败（预检会看不到这次结果）: %s", exc)
        res.ok = False; res.errors.append(f"last_backup: {exc}")
    logger.info("[backup] 快照 %s | 项目=%s | %.1f KB | 清理旧快照=%d | ok=%s",
                stamp, res.items, res.bytes_written / 1024, len(res.pruned), res.ok)
    return res


def main(argv: Optional[List[str]] = None) -> int:
    res = run_daily_backup()
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=1))
    return 0 if res.ok else 1


if __name__ == "__main__":       # pragma: no cover - 命令行入口
    sys.exit(main())
