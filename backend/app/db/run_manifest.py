"""
run_manifest.py — 可复现性台账（Task 6.5，2026-07-30）

每次回测 / GP / 模拟成交记录一条 RunManifest：
  {run_type, data_sha256, git_commit, seed, config_json, timestamp, summary_json}

配合 R-N1（GP 生成/变异已确定性），达成 A4：任一历史记录可用
「数据哈希 + 代码版本 + 种子 + 配置」完整重放，指标误差为零。

data_sha256：对数据集做规范化哈希（字段名排序 + 每个 (T×N) 面板的字节），
使"同一数据"可被验证——重放前比对 data_sha256 即知数据是否一致。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import (
    Column, DateTime, Integer, String, Text, create_engine, select,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class _Base(DeclarativeBase):
    pass


class RunManifestRecord(_Base):
    __tablename__ = "run_manifests"

    id:          int      = Column(Integer, primary_key=True, autoincrement=True)
    run_type:    str      = Column(String(32), nullable=False, index=True)  # backtest|gp|paper
    data_sha256: str      = Column(String(64), nullable=False, index=True)
    git_commit:  str      = Column(String(64), default="")
    seed:        int      = Column(Integer, default=0)
    config_json: str      = Column(Text, default="{}")
    summary_json:str      = Column(Text, default="{}")
    created_at:  datetime = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# 哈希与版本辅助
# ---------------------------------------------------------------------------

def dataset_sha256(dataset: Dict[str, pd.DataFrame]) -> str:
    """
    对 dict[field -> (T×N) DataFrame] 做确定性 SHA256。

    规范化：字段名升序；每个字段哈希其 (index, columns, values) 的字节，
    使字段顺序、构造顺序不影响结果，"同一数据"→"同一哈希"。
    """
    h = hashlib.sha256()
    for field in sorted(dataset.keys()):
        v = dataset[field]
        h.update(field.encode("utf-8"))
        if isinstance(v, pd.DataFrame):
            h.update(pd.util.hash_pandas_object(v.index, index=False).values.tobytes())
            h.update("|".join(map(str, v.columns)).encode("utf-8"))
            # 值：float64 字节；NaN 规范为固定位模式（np 已保证一致）
            h.update(v.to_numpy(dtype="float64").tobytes())
        else:
            # (N,) 数组等辅助字段
            import numpy as np
            h.update(np.asarray(v).tobytes())
    return h.hexdigest()


def current_git_commit() -> str:
    """当前 HEAD 短哈希；非 git 环境返回 ''。"""
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

@dataclass
class RunManifest:
    run_type:    str
    data_sha256: str
    seed:        int = 0
    git_commit:  str = ""
    config:      Optional[Dict[str, Any]] = None
    summary:     Optional[Dict[str, Any]] = None


class RunManifestStore:
    def __init__(self, db_url: Optional[str] = None) -> None:
        if db_url is None:
            db_url = os.getenv("DATABASE_URL", "")
            if not db_url:
                try:
                    from app.config import settings
                    db_url = settings.database_url
                except Exception:
                    db_url = "sqlite:///alphas.db"
        connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        self._engine = create_engine(db_url, connect_args=connect_args, echo=False)
        from ._sqlite_utils import harden_sqlite_engine
        harden_sqlite_engine(self._engine)
        _Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    def record(
        self,
        run_type:    str,
        dataset:     Dict[str, pd.DataFrame],
        seed:        int = 0,
        config:      Optional[Dict[str, Any]] = None,
        summary:     Optional[Dict[str, Any]] = None,
        git_commit:  Optional[str] = None,
    ) -> int:
        """计算数据哈希 + 记录一条 manifest，返回自增 id。"""
        rec = RunManifestRecord(
            run_type    = run_type,
            data_sha256 = dataset_sha256(dataset),
            git_commit  = git_commit if git_commit is not None else current_git_commit(),
            seed        = int(seed),
            config_json = json.dumps(config or {}, ensure_ascii=False, default=str),
            summary_json= json.dumps(summary or {}, ensure_ascii=False, default=str),
        )
        with self._Session() as s:
            s.add(rec)
            s.commit()
            return rec.id  # type: ignore[return-value]

    def get(self, manifest_id: int) -> Optional[RunManifestRecord]:
        with self._Session() as s:
            return s.get(RunManifestRecord, manifest_id)

    def query(self, run_type: Optional[str] = None, limit: int = 100) -> List[RunManifestRecord]:
        with self._Session() as s:
            stmt = select(RunManifestRecord)
            if run_type:
                stmt = stmt.where(RunManifestRecord.run_type == run_type)
            stmt = stmt.order_by(RunManifestRecord.id.desc()).limit(limit)
            return list(s.scalars(stmt))

    def verify_dataset(self, manifest_id: int, dataset: Dict[str, pd.DataFrame]) -> bool:
        """重放前校验：给定数据集的哈希是否与 manifest 记录一致。"""
        rec = self.get(manifest_id)
        if rec is None:
            raise KeyError(f"RunManifest id={manifest_id} 不存在")
        return dataset_sha256(dataset) == rec.data_sha256
