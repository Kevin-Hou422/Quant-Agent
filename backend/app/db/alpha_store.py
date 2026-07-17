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
    Column, Date, DateTime, Float, Integer, String, Text, UniqueConstraint,
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


class AlphaICRecord(_Base):
    """Task 5.1：因子逐日 realized IC 历史（AlphaMonitor 写入）。"""
    __tablename__ = "alpha_ic_history"
    __table_args__ = (
        # 幂等键：同一因子同一天只允许一条记录（重跑覆盖而非追加）
        UniqueConstraint("alpha_id", "date", name="uq_alpha_ic_date"),
    )

    id:              int      = Column(Integer, primary_key=True, autoincrement=True)
    alpha_id:        int      = Column(Integer, nullable=False, index=True)
    date:            object   = Column(Date, nullable=False)
    realized_ic:     float    = Column(Float, default=0.0)
    realized_return: float    = Column(Float, default=0.0)
    recorded_at:     datetime = Column(DateTime, default=datetime.utcnow)


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

    # ------------------------------------------------------------------
    # Task 5.2：状态流转（经状态机校验）
    # ------------------------------------------------------------------

    def update_status(self, alpha_id: int, new_status: str) -> AlphaRecord:
        """
        流转因子状态；非法流转抛 IllegalTransition，不存在抛 KeyError。
        校验规则见 app.db.alpha_lifecycle。
        """
        from .alpha_lifecycle import validate_transition

        with self._Session() as session:
            record = session.get(AlphaRecord, alpha_id)
            if record is None:
                raise KeyError(f"Alpha id={alpha_id} 不存在")
            validate_transition(record.status, new_status)
            record.status = new_status.strip().lower()
            session.commit()
            return record

    # ------------------------------------------------------------------
    # Task 5.1：IC 历史（AlphaMonitor 写入/读取）
    # ------------------------------------------------------------------

    def record_ic(
        self,
        alpha_id:        int,
        date,                          # datetime.date | str "YYYY-MM-DD"
        realized_ic:     float,
        realized_return: float = 0.0,
    ) -> None:
        """
        写入某因子某日的 realized IC。幂等：同 (alpha_id, date) 重复调用
        覆盖旧值而非追加（重跑当日任务不会重复记账）。
        """
        from datetime import date as _date
        if isinstance(date, str):
            date = _date.fromisoformat(date)

        with self._Session() as session:
            existing = session.scalars(
                select(AlphaICRecord).where(
                    AlphaICRecord.alpha_id == alpha_id,
                    AlphaICRecord.date == date,
                )
            ).first()
            if existing is not None:
                existing.realized_ic     = float(realized_ic)
                existing.realized_return = float(realized_return)
                existing.recorded_at     = datetime.utcnow()
            else:
                session.add(AlphaICRecord(
                    alpha_id        = alpha_id,
                    date            = date,
                    realized_ic     = float(realized_ic),
                    realized_return = float(realized_return),
                ))
            session.commit()

    def get_ic_history(self, alpha_id: int, limit: int = 250) -> List[AlphaICRecord]:
        """按日期升序返回某因子最近 limit 条 IC 记录。"""
        with self._Session() as session:
            stmt = (
                select(AlphaICRecord)
                .where(AlphaICRecord.alpha_id == alpha_id)
                .order_by(AlphaICRecord.date.desc())
                .limit(limit)
            )
            rows = list(session.scalars(stmt))
        return list(reversed(rows))

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
