"""
execution_store.py — 执行层持久化（Phase 12.1 / 12.4）

与 PositionStore 同库（settings.database_url）、各自建表：
  exec_orders    : 每张单一行，client_id 唯一。**先写意图再下单**（PENDING_SUBMIT），
                   下单返回后才补券商订单号 —— 进程在两步之间崩溃时，恢复流程按
                   client_id(=remark) 去券商侧找回，而不是重下一张。
  exec_fills     : 成交**增量**（券商只给累计成交量与累计均价，这里差分出每次新增的部分）。
                   UNIQUE(client_id, cum_qty) 兜底：同一累计量重复对账不会记两次。
  exec_snapshots : 每次对账的券商账户/持仓快照 + 与账本推算值的差异。
  exec_events    : 人工动作与熔断的审计日志（全平/复位/接受差异/恢复）。
  exec_state     : 键值状态（全平熔断是否开启、资产基线）。

本地订单状态
------------
PENDING_SUBMIT  已写意图，还不知道券商是否收到（下单调用未返回 / 返回异常）
SUBMITTED       券商已收到、仍在途
FILLED / PARTIAL / CANCELLED / FAILED   券商侧终态（PARTIAL = 部分成交后终止）
NOT_SUBMITTED   已确认券商侧不存在（崩溃发生在下单之前）
REJECTED_BY_GATE 被下单前风控拦下，从未发出

只有 REJECTED_BY_GATE 与 NOT_SUBMITTED 可以在同一决策日重试 —— 它们确定没有到达券商，
重试不会产生重复单。其余状态一律视为"这一天这一张已经发过了"。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date as _date, datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import (
    Column, Date, DateTime, Float, Integer, String, Text, UniqueConstraint,
    create_engine, select,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.execution.broker_gateway import DUST_QTY, TERMINAL_STATUSES, BrokerOrder


class _Base(DeclarativeBase):
    pass


ST_PENDING = "PENDING_SUBMIT"
ST_SUBMITTED = "SUBMITTED"
ST_FILLED = "FILLED"
ST_PARTIAL = "PARTIAL"
ST_CANCELLED = "CANCELLED"
ST_FAILED = "FAILED"
ST_NOT_SUBMITTED = "NOT_SUBMITTED"
ST_GATE = "REJECTED_BY_GATE"

OPEN_LOCAL = frozenset({ST_PENDING, ST_SUBMITTED})
RETRYABLE = frozenset({ST_GATE, ST_NOT_SUBMITTED})


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def local_status_from_broker(broker_status: str, dealt_qty: float) -> str:
    s = str(broker_status)
    if s not in TERMINAL_STATUSES:
        return ST_SUBMITTED
    if s == "FILLED_ALL":
        return ST_FILLED
    if dealt_qty > DUST_QTY:
        return ST_PARTIAL
    if s in ("CANCELLED_ALL", "CANCELLED_PART", "FILL_CANCELLED"):
        return ST_CANCELLED
    return ST_FAILED


class ExecOrder(_Base):
    __tablename__ = "exec_orders"
    id:              int   = Column(Integer, primary_key=True, autoincrement=True)
    client_id:       str   = Column(String(64), nullable=False, unique=True)
    book_id:         int   = Column(Integer, nullable=False, index=True)
    decision_date:   object = Column(Date, nullable=False, index=True)
    purpose:         str   = Column(String(16), nullable=False)
    ticker:          str   = Column(String(32), nullable=False)
    side:            str   = Column(String(8), nullable=False)
    qty:             int   = Column(Integer, nullable=False)
    ref_price:       float = Column(Float, nullable=False)
    limit_price:     float = Column(Float, nullable=False)
    target_weight:   float = Column(Float, default=0.0)
    status:          str   = Column(String(24), nullable=False, index=True)
    broker_order_id: str   = Column(String(64), nullable=True, index=True)
    broker_status:   str   = Column(String(32), default="")
    dealt_qty:       float = Column(Float, default=0.0)
    dealt_avg_price: float = Column(Float, default=0.0)
    reject_reason:   str   = Column(String(64), default="")
    last_err_msg:    str   = Column(Text, default="")
    #: 下单调用**发出之前**落的时间戳（审计 F03）。非空 = 请求可能已到达券商，
    #: 对账时找不到也不得判"未提交"后自动重下 —— 进程可能死在券商收单与 mark_submitted 之间。
    submit_attempted_at: datetime = Column(DateTime, nullable=True)
    created_at:      datetime = Column(DateTime, default=_now)
    updated_at:      datetime = Column(DateTime, default=_now)


class ExecFill(_Base):
    __tablename__ = "exec_fills"
    __table_args__ = (UniqueConstraint("client_id", "cum_qty", name="uq_exec_fill"),)
    id:              int   = Column(Integer, primary_key=True, autoincrement=True)
    client_id:       str   = Column(String(64), nullable=False, index=True)
    broker_order_id: str   = Column(String(64), nullable=False)
    book_id:         int   = Column(Integer, nullable=False, index=True)
    decision_date:   object = Column(Date, nullable=False, index=True)
    fill_date:       object = Column(Date, nullable=False, index=True)
    ticker:          str   = Column(String(32), nullable=False)
    side:            str   = Column(String(8), nullable=False)
    qty:             float = Column(Float, nullable=False)     # 本次增量（有符号：买正卖负）
    price:           float = Column(Float, nullable=False)     # 本次增量的成交均价
    ref_price:       float = Column(Float, nullable=False)     # 决策日参考价（模拟账本的成交价口径）
    cum_qty:         float = Column(Float, nullable=False)     # 本次之后的累计成交量
    #: 入账时最新一张快照的 id。对账用它判断"这笔成交是否已包含在某张快照的持仓里"：
    #: 同一轮对账先入账成交、后写快照，所以 snapshot_seq < 快照 id 的成交都已在快照里。
    #: 用自增 id 定序而不是时间戳 —— Windows 时钟精度下两者可能拿到同一个时间戳。
    snapshot_seq:    int   = Column(Integer, nullable=False, default=0)
    recorded_at:     datetime = Column(DateTime, default=_now)


class ExecSnapshot(_Base):
    __tablename__ = "exec_snapshots"
    id:                 int   = Column(Integer, primary_key=True, autoincrement=True)
    taken_at:           datetime = Column(DateTime, default=_now, index=True)
    market_date:        object = Column(Date, nullable=False, index=True)
    total_assets:       float = Column(Float, nullable=False)
    cash:               float = Column(Float, nullable=False)
    power:              float = Column(Float, nullable=False)
    market_val:         float = Column(Float, default=0.0)
    positions_json:     str   = Column(Text, default="{}")   # {ticker: {qty, price, market_val}}
    discrepancies_json: str   = Column(Text, default="[]")
    status:             str   = Column(String(16), nullable=False)  # clean|discrepancy|accepted|baseline
    note:               str   = Column(Text, default="")


class ExecEvent(_Base):
    __tablename__ = "exec_events"
    id:      int   = Column(Integer, primary_key=True, autoincrement=True)
    at:      datetime = Column(DateTime, default=_now, index=True)
    kind:    str   = Column(String(32), nullable=False)
    actor:   str   = Column(String(64), default="")
    reason:  str   = Column(Text, default="")
    payload: str   = Column(Text, default="{}")


class ExecState(_Base):
    __tablename__ = "exec_state"
    key:        str = Column(String(64), primary_key=True)
    value:      str = Column(Text, default="")
    updated_at: datetime = Column(DateTime, default=_now)


#: 可被信任为对账基准的快照状态
TRUSTED_SNAPSHOT = ("clean", "accepted", "baseline")


@dataclass
class FillIncrement:
    client_id: str
    ticker:    str
    side:      str
    qty:       float          # 有符号
    price:     float
    ref_price: float
    fill_date: _date
    decision_date: _date


class ExecutionStore:
    def __init__(self, db_url: Optional[str] = None) -> None:
        if db_url is None:
            db_url = os.getenv("DATABASE_URL", "")
            if not db_url:
                from app.config import settings
                db_url = settings.database_url
        connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        self._engine = create_engine(db_url, connect_args=connect_args, echo=False)
        from ._sqlite_utils import harden_sqlite_engine
        harden_sqlite_engine(self._engine)
        _Base.metadata.create_all(self._engine)
        self._add_missing_columns()
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    def _add_missing_columns(self) -> None:
        """create_all 不会给已存在的表补列；新增的可空列在这里补（只加、不改、不删）。"""
        from sqlalchemy import inspect, text
        have = {c["name"] for c in inspect(self._engine).get_columns("exec_orders")}
        if "submit_attempted_at" not in have:
            with self._engine.begin() as conn:
                conn.execute(text("ALTER TABLE exec_orders ADD COLUMN submit_attempted_at DATETIME"))

    # ------------------------------------------------------------------
    # 订单：写意图 / 下单结果 / 风控拒单
    # ------------------------------------------------------------------

    def get(self, client_id: str) -> Optional[ExecOrder]:
        with self._Session() as s:
            return s.scalars(select(ExecOrder).where(ExecOrder.client_id == client_id)).first()

    def _upsert_planned(self, order, book_id: int, decision_date, status: str,
                        reject_reason: str = "") -> bool:
        """
        新建或（仅当现有行可重试时）覆盖一行。返回 False = 已有不可重试的行（这张单发过了）。
        """
        with self._Session() as s:
            row = s.scalars(select(ExecOrder).where(ExecOrder.client_id == order.client_id)).first()
            if row is not None and row.status not in RETRYABLE:
                return False
            if row is None:
                row = ExecOrder(client_id=order.client_id)
                s.add(row)
            row.created_at = _now()
            row.book_id = int(book_id)
            row.decision_date = decision_date
            row.purpose = order.purpose
            row.ticker = order.ticker
            row.side = order.side
            row.qty = int(order.qty)
            row.ref_price = float(order.ref_price)
            row.limit_price = float(order.limit_price)
            row.target_weight = float(order.target_weight)
            row.status = status
            row.broker_order_id = None
            row.broker_status = ""
            row.dealt_qty = 0.0
            row.dealt_avg_price = 0.0
            row.reject_reason = reject_reason
            row.last_err_msg = ""
            row.submit_attempted_at = None
            row.updated_at = _now()
            s.commit()
            return True

    def mark_attempting(self, client_id: str) -> None:
        """下单调用发出前调用：之后无论进程在哪一步死掉，这张单都按"可能已到达券商"处理。"""
        with self._Session() as s:
            row = s.scalars(select(ExecOrder).where(ExecOrder.client_id == client_id)).one()
            row.submit_attempted_at = _now()
            row.updated_at = _now()
            s.commit()

    def add_intent(self, order, book_id: int, decision_date) -> bool:
        """写前日志。返回 False = 同一 client_id 已发过（幂等跳过）。"""
        return self._upsert_planned(order, book_id, decision_date, ST_PENDING)

    def record_gate_rejection(self, order, book_id: int, decision_date, reason: str) -> bool:
        return self._upsert_planned(order, book_id, decision_date, ST_GATE, reject_reason=reason)

    def is_final_for_day(self, client_id: str) -> bool:
        row = self.get(client_id)
        return row is not None and row.status not in RETRYABLE

    def mark_submitted(self, client_id: str, broker_order_id: str) -> None:
        with self._Session() as s:
            row = s.scalars(select(ExecOrder).where(ExecOrder.client_id == client_id)).one()
            row.broker_order_id = str(broker_order_id)
            row.status = ST_SUBMITTED
            row.updated_at = _now()
            s.commit()

    def note_submit_error(self, client_id: str, msg: str) -> None:
        """下单调用失败：**保持 PENDING_SUBMIT**（超时时券商可能已经收到），交给对账去确认。"""
        with self._Session() as s:
            row = s.scalars(select(ExecOrder).where(ExecOrder.client_id == client_id)).one()
            row.last_err_msg = str(msg)[:2000]
            row.updated_at = _now()
            s.commit()

    def mark_not_submitted(self, client_id: str, why: str) -> None:
        with self._Session() as s:
            row = s.scalars(select(ExecOrder).where(ExecOrder.client_id == client_id)).one()
            row.status = ST_NOT_SUBMITTED
            row.last_err_msg = str(why)[:2000]
            row.updated_at = _now()
            s.commit()

    def open_orders(self, book_id: Optional[int] = None) -> List[ExecOrder]:
        with self._Session() as s:
            stmt = select(ExecOrder).where(ExecOrder.status.in_(list(OPEN_LOCAL)))
            if book_id is not None:
                stmt = stmt.where(ExecOrder.book_id == book_id)
            return list(s.scalars(stmt.order_by(ExecOrder.id)))

    def orders(self, since=None, limit: int = 500) -> List[ExecOrder]:
        with self._Session() as s:
            stmt = select(ExecOrder)
            if since is not None:
                stmt = stmt.where(ExecOrder.decision_date >= since)
            return list(s.scalars(stmt.order_by(ExecOrder.id.desc()).limit(limit)))

    def adopt_broker_order(self, bo: BrokerOrder, book_id: int, decision_date, purpose: str,
                           ref_price: float) -> None:
        """
        券商侧有、本地没有的**我方**订单（remark 带我方前缀）—— 本地库丢失或崩在写意图之前。
        按券商记录补一行，dealt 从 0 开始，随后的增量对账会把它的成交补记进来。
        """
        with self._Session() as s:
            if s.scalars(select(ExecOrder).where(ExecOrder.client_id == bo.client_id)).first():
                return
            s.add(ExecOrder(
                client_id=bo.client_id, book_id=int(book_id), decision_date=decision_date,
                purpose=purpose, ticker=bo.ticker, side=bo.side, qty=int(round(bo.qty)),
                ref_price=float(ref_price), limit_price=float(bo.price), target_weight=0.0,
                status=ST_SUBMITTED, broker_order_id=bo.broker_order_id,
                broker_status=bo.status, dealt_qty=0.0, dealt_avg_price=0.0,
                last_err_msg="adopted_from_broker", created_at=_now(), updated_at=_now()))
            s.commit()

    @staticmethod
    def _latest_snapshot_id(s) -> int:
        sid = s.scalars(select(ExecSnapshot.id).order_by(ExecSnapshot.id.desc()).limit(1)).first()
        return int(sid or 0)

    def apply_broker_state(self, client_id: str, bo: BrokerOrder,
                           fill_date: _date) -> Optional[FillIncrement]:
        """
        把券商的累计成交状态落到本地，返回本次**新增**成交（无新增返回 None）。
        增量均价 = (新累计额 − 旧累计额) / (新累计量 − 旧累计量)。
        """
        with self._Session() as s:
            row = s.scalars(select(ExecOrder).where(ExecOrder.client_id == client_id)).one()
            old_q, old_p = float(row.dealt_qty or 0.0), float(row.dealt_avg_price or 0.0)
            new_q, new_p = float(bo.dealt_qty), float(bo.dealt_avg_price)
            inc = None
            dq = new_q - old_q
            if abs(dq) > DUST_QTY:
                if dq > 0:
                    price = (new_p * new_q - old_p * old_q) / dq
                else:
                    price = old_p               # 成交被撤销（FILL_CANCELLED）：按原均价冲回
                sign = 1.0 if row.side == "BUY" else -1.0
                inc = FillIncrement(
                    client_id=client_id, ticker=row.ticker, side=row.side,
                    qty=sign * dq, price=price, ref_price=float(row.ref_price),
                    fill_date=fill_date, decision_date=row.decision_date)
                s.add(ExecFill(
                    client_id=client_id, broker_order_id=bo.broker_order_id,
                    book_id=row.book_id, decision_date=row.decision_date, fill_date=fill_date,
                    ticker=row.ticker, side=row.side, qty=inc.qty, price=price,
                    ref_price=float(row.ref_price), cum_qty=new_q,
                    snapshot_seq=self._latest_snapshot_id(s), recorded_at=_now()))
            row.broker_order_id = bo.broker_order_id
            row.broker_status = bo.status
            row.dealt_qty = new_q
            row.dealt_avg_price = new_p
            row.status = local_status_from_broker(bo.status, new_q)
            if bo.last_err_msg:
                row.last_err_msg = bo.last_err_msg[:2000]
            row.updated_at = _now()
            s.commit()
            return inc

    def fills_after_snapshot(self, snapshot_id: int, book_id: int) -> List[ExecFill]:
        """入账于快照 `snapshot_id` 写下之后的成交（尚未包含在该快照持仓里的那些）。"""
        with self._Session() as s:
            return list(s.scalars(
                select(ExecFill).where(ExecFill.snapshot_seq >= snapshot_id,
                                       ExecFill.book_id == book_id)
                .order_by(ExecFill.id)))

    def fills(self, start=None, end=None, book_id: Optional[int] = None) -> List[ExecFill]:
        with self._Session() as s:
            stmt = select(ExecFill)
            if start is not None:
                stmt = stmt.where(ExecFill.decision_date >= start)
            if end is not None:
                stmt = stmt.where(ExecFill.decision_date <= end)
            if book_id is not None:
                stmt = stmt.where(ExecFill.book_id == book_id)
            return list(s.scalars(stmt.order_by(ExecFill.id)))

    # ------------------------------------------------------------------
    # 快照
    # ------------------------------------------------------------------

    def add_snapshot(self, *, market_date, account, positions: Dict[str, dict],
                     discrepancies: List[dict], status: str, note: str = "") -> ExecSnapshot:
        with self._Session() as s:
            snap = ExecSnapshot(
                taken_at=_now(), market_date=market_date,
                total_assets=float(account.total_assets), cash=float(account.cash),
                power=float(account.power), market_val=float(account.market_val),
                positions_json=json.dumps(positions, sort_keys=True),
                discrepancies_json=json.dumps(discrepancies, ensure_ascii=False),
                status=status, note=note)
            s.add(snap)
            s.commit()
            return snap

    def latest_snapshot(self, trusted_only: bool = False) -> Optional[ExecSnapshot]:
        with self._Session() as s:
            stmt = select(ExecSnapshot)
            if trusted_only:
                stmt = stmt.where(ExecSnapshot.status.in_(TRUSTED_SNAPSHOT))
            return s.scalars(stmt.order_by(ExecSnapshot.id.desc()).limit(1)).first()

    def trusted_before(self, market_date) -> Optional[ExecSnapshot]:
        """严格早于 market_date 的最近一个可信快照（日亏熔断的比较基准）。"""
        with self._Session() as s:
            return s.scalars(
                select(ExecSnapshot)
                .where(ExecSnapshot.status.in_(TRUSTED_SNAPSHOT),
                       ExecSnapshot.market_date < market_date)
                .order_by(ExecSnapshot.id.desc()).limit(1)).first()

    def snapshots(self, limit: int = 30) -> List[ExecSnapshot]:
        with self._Session() as s:
            return list(s.scalars(select(ExecSnapshot).order_by(ExecSnapshot.id.desc()).limit(limit)))

    # ------------------------------------------------------------------
    # 状态与事件
    # ------------------------------------------------------------------

    def get_state(self, key: str) -> Optional[str]:
        with self._Session() as s:
            row = s.get(ExecState, key)
            return None if row is None else row.value

    def set_state(self, key: str, value: str) -> None:
        with self._Session() as s:
            row = s.get(ExecState, key)
            if row is None:
                s.add(ExecState(key=key, value=value, updated_at=_now()))
            else:
                row.value = value
                row.updated_at = _now()
            s.commit()

    def kill_switch(self) -> dict:
        raw = self.get_state("kill_switch")
        if not raw:
            return {"engaged": False}
        return json.loads(raw)

    def set_kill_switch(self, engaged: bool, actor: str, reason: str) -> dict:
        st = {"engaged": bool(engaged), "actor": actor, "reason": reason,
              "at": _now().isoformat(timespec="seconds")}
        self.set_state("kill_switch", json.dumps(st, ensure_ascii=False))
        return st

    def add_event(self, kind: str, actor: str = "", reason: str = "", payload: Optional[dict] = None) -> None:
        with self._Session() as s:
            s.add(ExecEvent(at=_now(), kind=kind, actor=actor, reason=reason,
                            payload=json.dumps(payload or {}, ensure_ascii=False, default=str)))
            s.commit()

    def events(self, limit: int = 50) -> List[ExecEvent]:
        with self._Session() as s:
            return list(s.scalars(select(ExecEvent).order_by(ExecEvent.id.desc()).limit(limit)))
