"""
strategy_store.py — 组合**策略配置**的持久化（Phase PM.7）

把"你真正交易的那份策略"升为**一等实体**：组合成分（选中的因子）、每因子配额（combo 权重）、
AUM/方法、策略级门 verdict、风控快照、换手/无交易带，带**状态 + 版本 + 审批谱系**。

这修正了架构错配（见 DEV_LESSONS §K 的延伸）：晋级/审批的对象从"单因子"升为"组合策略配置"——
资金决策单位是策略，不是因子。

状态机（策略级，比因子的 7 态简化）：
    proposed → approved → active → retired
                      ↘ rejected（终态）
    active → retired（衰减/人工退役）
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Boolean, create_engine, select
from sqlalchemy.orm import declarative_base, sessionmaker

_Base = declarative_base()

_ALLOWED = {
    "proposed": {"approved", "rejected"},
    "approved": {"active", "retired"},
    "active":   {"retired"},
    "rejected": set(),
    "retired":  set(),
}


class StrategyConfigRecord(_Base):
    __tablename__ = "strategy_configs"

    id:            int      = Column(Integer, primary_key=True, autoincrement=True)
    name:          str      = Column(String(128), default="")
    status:        str      = Column(String(16), default="proposed", index=True)
    version:       int      = Column(Integer, default=1)
    factors:       str      = Column(Text, default="[]")       # JSON: [factor_id/name,...]
    combo_weights: str      = Column(Text, default="{}")       # JSON: {factor: weight}
    aum:           float    = Column(Float, default=0.0)
    method:        str      = Column(String(32), default="ic_weighted")
    passed:        bool     = Column(Boolean, default=False)   # 策略门是否通过
    verdict:       str      = Column(Text, default="{}")       # JSON: StrategyGate.to_dict()
    risk_report:   str      = Column(Text, default="{}")       # JSON
    turnover_ann:  float    = Column(Float, default=0.0)
    no_trade_band: float    = Column(Float, default=0.0)
    created_at:    datetime = Column(DateTime, default=datetime.utcnow)


class StrategyDecision(_Base):
    __tablename__ = "strategy_decisions"

    id:          int      = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id: int      = Column(Integer, nullable=False, index=True)
    decision:    str      = Column(String(16), nullable=False)   # approve | reject | activate | retire
    from_status: str      = Column(String(16), default="")
    to_status:   str      = Column(String(16), default="")
    reason:      str      = Column(Text, default="")
    actor:       str      = Column(String(64), default="human")
    decided_at:  datetime = Column(DateTime, default=datetime.utcnow)


class IllegalStrategyTransition(Exception):
    pass


@dataclass
class StrategyConfig:
    """传给 StrategyStore.save() 的输入。"""
    factors:       List[str]
    combo_weights: Dict[str, float]
    aum:           float
    method:        str = "ic_weighted"
    passed:        bool = False
    verdict:       dict = field(default_factory=dict)
    risk_report:   dict = field(default_factory=dict)
    turnover_ann:  float = 0.0
    no_trade_band: float = 0.0
    name:          str = ""
    status:        str = "proposed"


class StrategyStore:
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

    def save(self, cfg: StrategyConfig) -> int:
        rec = StrategyConfigRecord(
            name=cfg.name, status=cfg.status,
            factors=json.dumps(cfg.factors), combo_weights=json.dumps(cfg.combo_weights),
            aum=cfg.aum, method=cfg.method, passed=cfg.passed,
            verdict=json.dumps(cfg.verdict), risk_report=json.dumps(cfg.risk_report),
            turnover_ann=cfg.turnover_ann, no_trade_band=cfg.no_trade_band,
            created_at=datetime.utcnow(),
        )
        with self._Session() as s:
            s.add(rec); s.commit(); return rec.id  # type: ignore[return-value]

    def get(self, sid: int) -> Optional[StrategyConfigRecord]:
        with self._Session() as s:
            return s.get(StrategyConfigRecord, sid)

    def query(self, status: Optional[str] = None, limit: int = 200) -> List[StrategyConfigRecord]:
        with self._Session() as s:
            stmt = select(StrategyConfigRecord)
            if status:
                stmt = stmt.where(StrategyConfigRecord.status == status)
            stmt = stmt.order_by(StrategyConfigRecord.created_at.desc()).limit(limit)
            return list(s.scalars(stmt))

    def latest_active(self) -> Optional[StrategyConfigRecord]:
        """最近一个 active（在交易的）策略配置；无则 None。"""
        act = self.query(status="active", limit=1)
        return act[0] if act else None

    def update_status(self, sid: int, new_status: str) -> StrategyConfigRecord:
        with self._Session() as s:
            rec = s.get(StrategyConfigRecord, sid)
            if rec is None:
                raise KeyError(f"Strategy id={sid} 不存在")
            old = str(rec.status)
            if new_status not in _ALLOWED.get(old, set()):
                raise IllegalStrategyTransition(f"非法流转 {old} → {new_status}")
            rec.status = new_status
            s.commit(); return rec

    def record_decision(self, sid: int, decision: str, from_status: str,
                        to_status: str, reason: str = "", actor: str = "human") -> None:
        with self._Session() as s:
            s.add(StrategyDecision(strategy_id=sid, decision=decision,
                                   from_status=from_status, to_status=to_status,
                                   reason=reason, actor=actor))
            s.commit()

    def get_decisions(self, sid: int) -> List[StrategyDecision]:
        with self._Session() as s:
            return list(s.scalars(
                select(StrategyDecision).where(StrategyDecision.strategy_id == sid)
                .order_by(StrategyDecision.decided_at)))

    @staticmethod
    def to_dict(rec: StrategyConfigRecord) -> dict:
        return {
            "id": rec.id, "name": rec.name, "status": rec.status, "version": rec.version,
            "factors": json.loads(rec.factors or "[]"),
            "combo_weights": json.loads(rec.combo_weights or "{}"),
            "aum": rec.aum, "method": rec.method, "passed": rec.passed,
            "verdict": json.loads(rec.verdict or "{}"),
            "risk_report": json.loads(rec.risk_report or "{}"),
            "turnover_ann": rec.turnover_ann, "no_trade_band": rec.no_trade_band,
            "created_at": rec.created_at.isoformat() if rec.created_at else None,
        }
