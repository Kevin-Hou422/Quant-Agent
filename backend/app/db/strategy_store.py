"""
strategy_store.py — 组合**策略配置**的持久化（Phase PM.7）

把"你真正交易的那份策略"升为**一等实体**：组合成分（选中的因子）、每因子配额（combo 权重）、
AUM/方法、策略级门 verdict、风控快照、换手/无交易带，带**状态 + 版本 + 审批谱系**。

这修正了架构错配（见 DEV_LESSONS §K 的延伸）：晋级/审批的对象从"单因子"升为"组合策略配置"——
资金决策单位是策略，不是因子。

状态机（策略级；运行状态与权限见 app/core/lifecycle/strategy_states.py）：
    proposed → approved ─┬→ integration（工程联调）→ paper_running
                         └→ paper_running（纸交易、收集证据）→ verified_active（过了预登记的验证设计）
    proposed → rejected；任何运行状态 → retired；verified_active → paper_running（降级）
    `active` 是 2026-10-22 之前的旧字样：不再能进入，旧行只能退役（未验证、没有规格）。

唯一运行槽（2026-10-25 审计 R7）：首轮只有一个账户 / 资金 book，任何时刻至多一份策略处于运行状态
（integration / paper_running / verified_active / 旧 active）。两道保证：
    · 数据库约束：部分唯一索引 `uq_strategy_running_slot`（运行状态的行上 (status IS NOT NULL) 唯一）——
      任何写入路径（含直接 SQL）都不能留下第二份；
    · `start_running`：在一个 `BEGIN IMMEDIATE` 事务里选出当前运行者、退役、启动新版本、写谱系 —— 并发的第二个请求
      等第一个提交后读到新状态（而不是两边都读到"没有运行策略"）。
旧库里已经有多份运行策略时建不了索引：如实记在 `running_slot_problem`，`latest_running` 拒绝静默挑最新的一份。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Boolean, create_engine, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

_Base = declarative_base()
logger = logging.getLogger(__name__)

_ALLOWED = {
    "proposed":        {"approved", "rejected"},
    "approved":        {"integration", "paper_running", "retired"},
    "integration":     {"paper_running", "retired"},
    "paper_running":   {"verified_active", "retired"},
    "verified_active": {"paper_running", "retired"},
    "active":          {"retired"},          # 旧字样（2026-10-22 之前）：只能退役
    "rejected":        set(),
    "retired":         set(),
}


def _label(status):
    from app.core.lifecycle.strategy_states import label
    return label(status)


def _perm(status):
    from app.core.lifecycle.strategy_states import permissions
    return permissions(status)


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


class RunningSlotConflict(IllegalStrategyTransition):
    """唯一运行槽被占 / 已被破坏（同一 book 出现第二份运行策略）。"""


#: 占用运行槽的状态（与 strategy_states.RUNNING_STATES + LEGACY_ACTIVE 一致，由测试核对）
SLOT_STATUSES = ("integration", "paper_running", "verified_active", "active")
RUNNING_SLOT_INDEX = "uq_strategy_running_slot"


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
        #: 建不了唯一运行槽约束的原因（旧库里已有多份运行策略）；None = 约束在
        self.running_slot_problem: Optional[str] = self._ensure_running_slot_index()

    def _ensure_running_slot_index(self) -> Optional[str]:
        if self._engine.dialect.name != "sqlite":
            return None
        statuses = ", ".join(f"'{s}'" for s in SLOT_STATUSES)
        try:
            with self._engine.begin() as c:
                c.exec_driver_sql(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS {RUNNING_SLOT_INDEX} ON strategy_configs "
                    f"((status IS NOT NULL)) WHERE status IN ({statuses})")
            return None
        except IntegrityError:
            ids = [r.id for r in self.running()]
            logger.error("[strategy_store] 唯一运行槽被破坏：%s 同时处于运行状态 —— 约束建不了，"
                         "每日循环不会按其中任何一份交易", ids)
            return f"唯一运行槽被破坏：策略 {ids} 同时处于运行状态"

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

    def running(self) -> List[StrategyConfigRecord]:
        """占用运行槽的全部策略（正常情况下至多一份）。"""
        with self._Session() as s:
            stmt = (select(StrategyConfigRecord).where(StrategyConfigRecord.status.in_(SLOT_STATUSES))
                    .order_by(StrategyConfigRecord.id))
            return list(s.scalars(stmt))

    def latest_running(self) -> Optional[StrategyConfigRecord]:
        """
        正在运行（在模拟账户交易）的那份策略配置：integration / paper_running / verified_active；
        旧的 `active` 行也返回（让每日循环看见它并拒绝 —— 它没有规格），由调用方按 strategy_states 判权限。
        多于一份 → RunningSlotConflict（2026-10-25 审计 R7：不静默挑最新的一份）。
        """
        rows = self.running()
        if len(rows) > 1:
            raise RunningSlotConflict(f"唯一运行槽被破坏：策略 {[r.id for r in rows]} 同时处于运行状态 —— "
                                      "不按其中任何一份交易；人工退役多余的那份")
        return rows[0] if rows else None

    def latest_active(self) -> Optional[StrategyConfigRecord]:
        """兼容旧调用：同 latest_running（"active" 不再表示已验证，见 strategy_states）。"""
        return self.latest_running()

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

    def apply_transitions(self, steps: List[tuple]) -> None:
        """
        一组状态流转 + 谱系记录**在一个事务里**完成（外部审查 2026-10-18 F14）。
        steps：[(sid, new_status, decision, reason, actor), ...]，按顺序校验；任何一步非法 →
        整组回滚，什么都不改。此前批准、退役旧配置、激活、写 decision 分别提交，中途出错会留下
        "已批准却没有 active"或谱系不全的状态。
        """
        with self._Session() as s:
            try:
                self._apply(s, steps)
                s.commit()
            except IntegrityError as exc:
                s.rollback()
                raise RunningSlotConflict(f"唯一运行槽冲突：进入运行状态要走 start_running（{exc.orig}）") from exc
            except Exception:
                s.rollback()
                raise

    def start_running(self, sid: int, target: str, reason: str, actor: str,
                      before: Optional[List[tuple]] = None) -> List[int]:
        """
        原子地占用唯一运行槽：一个 `BEGIN IMMEDIATE` 事务里 ① 应用 `before`（例如 proposed→approved）
        ② 读出**此刻**的运行者并逐个退役（留谱系）③ 把 `sid` 迁到 `target` 并写谱系。返回被退役的 id。
        非法流转 → IllegalStrategyTransition（整组回滚）；锁等待超时 / 约束冲突 → RunningSlotConflict。
        """
        if target not in SLOT_STATUSES:
            raise ValueError(f"{target} 不是运行状态")
        try:
            with self._engine.connect() as conn:
                if self._engine.dialect.name == "sqlite":
                    conn.exec_driver_sql("BEGIN IMMEDIATE")      # 先拿写锁，再读当前运行者
                with Session(bind=conn, expire_on_commit=False) as s:
                    self._apply(s, list(before or []))
                    others = s.scalars(select(StrategyConfigRecord).where(
                        StrategyConfigRecord.status.in_(SLOT_STATUSES), StrategyConfigRecord.id != sid)).all()
                    retire = [(o.id, "retired", "retire", f"被策略 #{sid} 取代", actor) for o in others]
                    self._apply(s, retire + [(sid, target, "start_" + target, reason, actor)])
                    s.commit()
                conn.commit()
        except IntegrityError as exc:
            raise RunningSlotConflict(f"唯一运行槽冲突：另一份策略同时进入了运行状态（{exc.orig}）") from exc
        except OperationalError as exc:
            raise RunningSlotConflict(f"唯一运行槽正被另一个请求修改，稍后重试（{exc.orig}）") from exc
        return [r[0] for r in retire]

    @staticmethod
    def _apply(s, steps: List[tuple]) -> None:
        for sid, new_status, decision, reason, actor in steps:
            rec = s.get(StrategyConfigRecord, sid)
            if rec is None:
                raise KeyError(f"Strategy id={sid} 不存在")
            old = str(rec.status)
            if new_status not in _ALLOWED.get(old, set()):
                raise IllegalStrategyTransition(f"非法流转 {old} → {new_status}（#{sid}）")
            rec.status = new_status
            s.add(StrategyDecision(strategy_id=sid, decision=decision, from_status=old,
                                   to_status=new_status, reason=reason, actor=actor))
            s.flush()

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
            # 2026-10-22 用户核验 C6：状态的权限写明，前端不再从字样猜"是否已验证"
            "status_label": _label(rec.status), "verified": _perm(rec.status)["verified"],
            "permissions": _perm(rec.status),
            "verdict": json.loads(rec.verdict or "{}"),
            "risk_report": json.loads(rec.risk_report or "{}"),
            "turnover_ann": rec.turnover_ann, "no_trade_band": rec.no_trade_band,
            "created_at": rec.created_at.isoformat() if rec.created_at else None,
        }
