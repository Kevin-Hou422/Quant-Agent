"""
diagnostics_store.py — 每轮组合运行的诊断持久化（FE-TR 前置）

`run_portfolio` 每天算出一堆**决定"能否赚钱"的运行时事实**——交易现实(TR.1)、T3 账户(TR.3)、
门分级(TR.4)、风控施加(PM.5)、换手/无交易带(PM.6)、策略衰减(PM.7)——此前**只进日志**，
前端看不见、事后无法核对。本表把每轮诊断**只增不改**地存下来，供 FE-TR 展示与事后审计。

（这也是"决策+结果可追溯"的第一块地基：先把每天系统看到什么、做了什么记下来。）
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column, DateTime, Integer, Text, create_engine, select
from sqlalchemy.orm import declarative_base, sessionmaker

_Base = declarative_base()


class PortfolioDiagnostic(_Base):
    __tablename__ = "portfolio_diagnostics"

    id:      int      = Column(Integer, primary_key=True, autoincrement=True)
    run_at:  datetime = Column(DateTime, default=datetime.utcnow, index=True)
    payload: str      = Column(Text, default="{}")     # run_portfolio 返回的完整诊断 JSON


class DiagnosticsStore:
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
        try:
            from ._sqlite_utils import harden_sqlite_engine
            harden_sqlite_engine(self._engine)
        except Exception:
            pass
        _Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    def save(self, payload: dict) -> int:
        """存一轮诊断（JSON 序列化失败的字段用 str 兜底，绝不因诊断失败影响交易）。"""
        rec = PortfolioDiagnostic(
            run_at=datetime.utcnow(),
            payload=json.dumps(payload, ensure_ascii=False, default=str),
        )
        with self._Session() as s:
            s.add(rec); s.commit(); return rec.id  # type: ignore[return-value]

    def recent(self, limit: int = 20) -> List[dict]:
        """最近 N 轮诊断（新→旧）。"""
        with self._Session() as s:
            rows = list(s.scalars(
                select(PortfolioDiagnostic)
                .order_by(PortfolioDiagnostic.run_at.desc()).limit(limit)))
        out = []
        for r in rows:
            try:
                payload = json.loads(r.payload or "{}")
            except Exception:
                payload = {}
            out.append({"id": r.id,
                        "run_at": r.run_at.isoformat() if r.run_at else None,
                        **payload})
        return out
