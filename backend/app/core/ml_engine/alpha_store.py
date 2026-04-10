"""
AlphaStore — SQLite 持久化层（SQLAlchemy ORM）。

表结构：alpha_records
  id, dsl, created_at, hypothesis,
  ann_return, sharpe, max_drawdown, ic_ir, ann_turnover,
  status, reasoning (JSON text)

DATABASE_URL 环境变量配置，默认 sqlite:///alphas.db。
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text,
    create_engine, select,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


# ---------------------------------------------------------------------------
# ORM
# ---------------------------------------------------------------------------

class _Base(DeclarativeBase):
    pass


class AlphaRecord(_Base):
    """每条 Alpha 的持久化记录。"""
    __tablename__ = "alpha_records"

    id:           int      = Column(Integer, primary_key=True, autoincrement=True)
    dsl:          str      = Column(Text,    nullable=False)
    created_at:   datetime = Column(DateTime, default=datetime.utcnow)
    hypothesis:   str      = Column(String(512), default="")
    ann_return:   float    = Column(Float, default=0.0)
    sharpe:       float    = Column(Float, default=0.0)
    max_drawdown: float    = Column(Float, default=0.0)
    ic_ir:        float    = Column(Float, default=0.0)
    ann_turnover: float    = Column(Float, default=0.0)
    status:       str      = Column(String(32), default="active")
    reasoning:    str      = Column(Text, default="")


# ---------------------------------------------------------------------------
# Input dataclass
# ---------------------------------------------------------------------------

@dataclass
class AlphaResult:
    """AlphaEvolver / AlphaAgent 产出，传给 AlphaStore.save()。"""
    dsl:          str
    hypothesis:   str   = ""
    ann_return:   float = 0.0
    sharpe:       float = 0.0
    max_drawdown: float = 0.0
    ic_ir:        float = 0.0
    ann_turnover: float = 0.0
    status:       str   = "active"
    reasoning:    str   = ""   # JSON string (ReasoningLog.to_json())


# ---------------------------------------------------------------------------
# AlphaStore
# ---------------------------------------------------------------------------

class AlphaStore:
    """
    提供 save / query / export_csv 三个核心接口。

    Parameters
    ----------
    db_url : SQLAlchemy 数据库 URL。
             默认读取环境变量 DATABASE_URL，否则 sqlite:///alphas.db。
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        url = db_url or os.getenv("DATABASE_URL", "sqlite:///alphas.db")
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self._engine = create_engine(url, connect_args=connect_args, echo=False)
        _Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    def save(self, result: AlphaResult) -> int:
        """持久化一条 Alpha，返回自增 id。"""
        record = AlphaRecord(
            dsl          = result.dsl,
            created_at   = datetime.utcnow(),
            hypothesis   = result.hypothesis,
            ann_return   = result.ann_return,
            sharpe       = result.sharpe,
            max_drawdown = result.max_drawdown,
            ic_ir        = result.ic_ir,
            ann_turnover = result.ann_turnover,
            status       = result.status,
            reasoning    = result.reasoning,
        )
        with self._Session() as session:
            session.add(record)
            session.commit()
            return record.id  # type: ignore[return-value]

    def query(
        self,
        min_sharpe:   float = -999.0,
        status:       Optional[str] = None,
        limit:        int = 200,
    ) -> List[AlphaRecord]:
        """按 Sharpe 过滤查询。"""
        with self._Session() as session:
            stmt = select(AlphaRecord).where(AlphaRecord.sharpe >= min_sharpe)
            if status:
                stmt = stmt.where(AlphaRecord.status == status)
            stmt = stmt.order_by(AlphaRecord.sharpe.desc()).limit(limit)
            return list(session.scalars(stmt))

    def get_by_id(self, alpha_id: int) -> Optional[AlphaRecord]:
        with self._Session() as session:
            return session.get(AlphaRecord, alpha_id)

    def export_csv(self, path: str) -> None:
        """将所有记录导出为 CSV 文件。"""
        records = self.query(limit=100_000)
        if not records:
            return
        fields = [
            "id", "dsl", "created_at", "hypothesis",
            "ann_return", "sharpe", "max_drawdown",
            "ic_ir", "ann_turnover", "status",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in records:
                writer.writerow({k: getattr(r, k) for k in fields})
