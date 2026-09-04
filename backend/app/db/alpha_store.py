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
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, Integer, String, Text, UniqueConstraint,
    create_engine, select,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


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
    # Phase 11：**回放 vs 真前向**。首次运行会把整段历史逐日"回放"记进来，那不是前向证据；
    # 只有摄取到**新 bar 之后**产生的 IC 才算前向。TR.4 的 →ACTIVE 门只能用 is_forward=True 的样本，
    # 否则会把历史回放当成前向战绩（这正是该门此前只能"仅记录不拦"的原因）。
    is_forward:      bool     = Column(Boolean, default=False, index=True)


class AlphaDecision(_Base):
    """Phase 9.4：审批/拒绝决策谱系（append-only，可审计）。"""
    __tablename__ = "alpha_decisions"

    id:          int      = Column(Integer, primary_key=True, autoincrement=True)
    alpha_id:    int      = Column(Integer, nullable=False, index=True)
    decision:    str      = Column(String(16), nullable=False)   # approve | reject | auto_approve
    from_status: str      = Column(String(32), default="")
    to_status:   str      = Column(String(32), default="")
    reason:      str      = Column(Text, default="")
    actor:       str      = Column(String(64), default="human")  # human | system(auto)
    decided_at:  datetime = Column(DateTime, default=datetime.utcnow)


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
    # Phase 9.3：新因子一律 CANDIDATE 起步，经验证门（ValidationGate）+ 人工审批逐级晋级，
    # 不再直接写 active（修复"发现即激活"绕过分级的缺口）。特殊场景可显式传 status。
    status:       str   = "candidate"
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
        # Task 6.2：默认库路径统一从 settings.database_url 取，避免调度线程
        # （无参 AlphaStore()）与 API（settings.database_url）解析到不同物理文件。
        if db_url is None:
            db_url = os.getenv("DATABASE_URL", "")
            if not db_url:
                try:
                    from app.config import settings
                    db_url = settings.database_url
                except Exception:
                    db_url = "sqlite:///alphas.db"
        url = db_url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self._engine = create_engine(url, connect_args=connect_args, echo=False)
        # Task 6.2：WAL + busy_timeout，支持调度线程与 API 并发写（体检 E-N2）
        from ._sqlite_utils import harden_sqlite_engine
        harden_sqlite_engine(self._engine)
        _Base.metadata.create_all(self._engine)
        self._migrate_add_columns()
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    def _migrate_add_columns(self) -> None:
        """
        轻量迁移：给已存在的表补新列（`create_all` 不会 ALTER 既有表）。
        目前只有 Phase 11 的 `alpha_ic_history.is_forward`。SQLite 的
        ADD COLUMN 是 O(1) 元数据操作，安全且幂等。
        """
        try:
            with self._engine.begin() as conn:
                cols = {r[1] for r in conn.exec_driver_sql(
                    "PRAGMA table_info(alpha_ic_history)").fetchall()}
                if cols and "is_forward" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE alpha_ic_history ADD COLUMN is_forward BOOLEAN DEFAULT 0")
                    logger.info("[alpha_store] 迁移：alpha_ic_history 增列 is_forward")
        except Exception as exc:      # 迁移失败不应让系统起不来，但要看得见
            logger.warning("[alpha_store] 迁移 is_forward 失败（旧库将缺该列）: %s", exc)

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
    # Phase 9.4：审批/拒绝决策谱系（append-only）
    # ------------------------------------------------------------------

    def record_decision(
        self,
        alpha_id:    int,
        decision:    str,               # approve | reject | auto_approve
        from_status: str,
        to_status:   str,
        reason:      str = "",
        actor:       str = "human",
    ) -> None:
        with self._Session() as session:
            session.add(AlphaDecision(
                alpha_id=alpha_id, decision=decision,
                from_status=from_status, to_status=to_status,
                reason=reason, actor=actor,
            ))
            session.commit()

    def get_decisions(self, alpha_id: int) -> List[AlphaDecision]:
        """按时间升序返回某因子的审批谱系。"""
        with self._Session() as session:
            return list(session.scalars(
                select(AlphaDecision)
                .where(AlphaDecision.alpha_id == alpha_id)
                .order_by(AlphaDecision.decided_at)
            ))

    # ------------------------------------------------------------------
    # Task 5.1：IC 历史（AlphaMonitor 写入/读取）
    # ------------------------------------------------------------------

    def record_ic(
        self,
        alpha_id:        int,
        date,                          # datetime.date | str "YYYY-MM-DD"
        realized_ic:     float,
        realized_return: float = 0.0,
        is_forward:      bool = False,
    ) -> None:
        """
        写入某因子某日的 realized IC。幂等：同 (alpha_id, date) 重复调用
        覆盖旧值而非追加（重跑当日任务不会重复记账）。

        is_forward : 该日是否为**真前向**（摄取到新 bar 之后产生），而非历史回放。
                     只有前向样本才可用于 TR.4 的 →ACTIVE 门。**一旦标为前向就不再降级**
                     （重跑回放不会把已积累的前向证据抹掉）。
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
                # 前向标记只升不降：已确认的前向证据不会被后续回放重跑覆盖掉
                if is_forward:
                    existing.is_forward = True
            else:
                session.add(AlphaICRecord(
                    alpha_id        = alpha_id,
                    date            = date,
                    realized_ic     = float(realized_ic),
                    realized_return = float(realized_return),
                    is_forward      = bool(is_forward),
                ))
            session.commit()

    def get_forward_ic(self, alpha_id: int, limit: int = 5000) -> List[AlphaICRecord]:
        """只取**真前向**的 IC 记录（TR.4 →ACTIVE 门专用；回放样本一律排除）。"""
        with self._Session() as session:
            return list(session.scalars(
                select(AlphaICRecord)
                .where(AlphaICRecord.alpha_id == alpha_id,
                       AlphaICRecord.is_forward.is_(True))
                .order_by(AlphaICRecord.date)
                .limit(limit)
            ))

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
