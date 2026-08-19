"""
position_store.py — Paper Trading 持久化（Task 7.2，2026-08-02）

三张表，全部**幂等**（唯一键约束 + upsert），使每日循环重跑或崩溃补跑不重复记账：
  paper_positions  : (alpha_id, date, ticker) → 当日收盘持仓权重   UNIQUE(alpha_id,date,ticker)
  paper_fills      : (alpha_id, date, ticker) → 目标/成交/成本/拒绝  UNIQUE(alpha_id,date,ticker)
  paper_daily_pnl  : (alpha_id, date)         → 毛/净收益/成本/净值   UNIQUE(alpha_id,date)

崩溃恢复：`last_pnl_date(alpha_id)` 返回该 alpha 已记账的最后日期，
`latest_positions(alpha_id)` 返回其当前持仓——循环启动时据此续跑，不回退不重复。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date as _date, datetime
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import (
    Column, Date, DateTime, Float, Integer, String, Text,
    UniqueConstraint, create_engine, delete, select,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class _Base(DeclarativeBase):
    pass


class PaperPosition(_Base):
    __tablename__ = "paper_positions"
    __table_args__ = (UniqueConstraint("alpha_id", "date", "ticker", name="uq_pos"),)
    id:       int    = Column(Integer, primary_key=True, autoincrement=True)
    alpha_id: int    = Column(Integer, nullable=False, index=True)
    date:     object = Column(Date, nullable=False, index=True)
    ticker:   str    = Column(String(32), nullable=False)
    weight:   float  = Column(Float, default=0.0)


class PaperFill(_Base):
    __tablename__ = "paper_fills"
    __table_args__ = (UniqueConstraint("alpha_id", "date", "ticker", name="uq_fill"),)
    id:            int    = Column(Integer, primary_key=True, autoincrement=True)
    alpha_id:      int    = Column(Integer, nullable=False, index=True)
    date:          object = Column(Date, nullable=False, index=True)
    ticker:        str    = Column(String(32), nullable=False)
    target_weight: float  = Column(Float, default=0.0)
    filled_weight: float  = Column(Float, default=0.0)
    fill_price:    float  = Column(Float, default=0.0)
    cost_usd:      float  = Column(Float, default=0.0)
    reject_reason: str    = Column(String(64), default="")


class PaperDailyPnL(_Base):
    __tablename__ = "paper_daily_pnl"
    __table_args__ = (UniqueConstraint("alpha_id", "date", name="uq_pnl"),)
    id:          int    = Column(Integer, primary_key=True, autoincrement=True)
    alpha_id:    int    = Column(Integer, nullable=False, index=True)
    date:        object = Column(Date, nullable=False, index=True)
    gross_ret:   float  = Column(Float, default=0.0)
    net_ret:     float  = Column(Float, default=0.0)
    cost_bps:    float  = Column(Float, default=0.0)
    equity:      float  = Column(Float, default=1.0)      # 归一化净值（初始 1.0）
    recorded_at: datetime = Column(DateTime, default=datetime.utcnow)


@dataclass
class DailyPnL:
    alpha_id:  int
    date:      str
    gross_ret: float
    net_ret:   float
    cost_bps:  float
    equity:    float


def _to_date(d) -> _date:
    return _date.fromisoformat(d) if isinstance(d, str) else (
        d.date() if isinstance(d, datetime) else d
    )


class PositionStore:
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

    # ------------------------------------------------------------------
    # 写入（幂等：同 (alpha_id, date[, ticker]) 覆盖）
    # ------------------------------------------------------------------

    def record_day(
        self,
        alpha_id:  int,
        date,
        positions: Dict[str, float],
        fills:     List[dict],
        pnl:       DailyPnL,
    ) -> None:
        """原子写入某 alpha 某日的持仓 + 成交 + PnL（先删同日旧记录，再插入 → 幂等）。"""
        d = _to_date(date)
        with self._Session() as s:
            # 幂等：清除该 (alpha, date) 的旧行
            for tbl in (PaperPosition, PaperFill, PaperDailyPnL):
                s.execute(delete(tbl).where(tbl.alpha_id == alpha_id, tbl.date == d))
            for tk, w in positions.items():
                s.add(PaperPosition(alpha_id=alpha_id, date=d, ticker=tk, weight=float(w)))
            for f in fills:
                s.add(PaperFill(
                    alpha_id=alpha_id, date=d, ticker=f["ticker"],
                    target_weight=float(f.get("target_weight", 0.0)),
                    filled_weight=float(f.get("filled_weight", 0.0)),
                    fill_price=float(f.get("fill_price", 0.0)),
                    cost_usd=float(f.get("cost_usd", 0.0)),
                    reject_reason=str(f.get("reject_reason", "")),
                ))
            s.add(PaperDailyPnL(
                alpha_id=alpha_id, date=d,
                gross_ret=pnl.gross_ret, net_ret=pnl.net_ret,
                cost_bps=pnl.cost_bps, equity=pnl.equity,
            ))
            s.commit()

    # ------------------------------------------------------------------
    # 读取（崩溃恢复 + 前端）
    # ------------------------------------------------------------------

    def last_pnl_date(self, alpha_id: int) -> Optional[_date]:
        with self._Session() as s:
            stmt = (select(PaperDailyPnL.date)
                    .where(PaperDailyPnL.alpha_id == alpha_id)
                    .order_by(PaperDailyPnL.date.desc()).limit(1))
            row = s.scalars(stmt).first()
            return row

    def latest_positions(self, alpha_id: int) -> Dict[str, float]:
        """返回该 alpha 最近一日的持仓权重 dict（无则空）。"""
        last = self.last_pnl_date(alpha_id)
        return self._positions_on(alpha_id, last) if last is not None else {}

    def latest_equity(self, alpha_id: int) -> float:
        last = self.last_pnl_date(alpha_id)
        return self._equity_on(alpha_id, last) if last is not None else 1.0

    def state_before(self, alpha_id: int, date) -> "tuple[float, Dict[str, float]]":
        """
        返回**严格早于 date** 的最近一日的 (equity, positions)——供 step 幂等/续跑：
        重跑第 t 日或崩溃续跑时，应基于 t-1 的状态，而非全局最近一日。
        无更早记录时返回 (1.0, {})。
        """
        d = _to_date(date)
        with self._Session() as s:
            prev = s.scalars(
                select(PaperDailyPnL.date)
                .where(PaperDailyPnL.alpha_id == alpha_id, PaperDailyPnL.date < d)
                .order_by(PaperDailyPnL.date.desc()).limit(1)
            ).first()
        if prev is None:
            return 1.0, {}
        return self._equity_on(alpha_id, prev), self._positions_on(alpha_id, prev)

    def _positions_on(self, alpha_id: int, d) -> Dict[str, float]:
        with self._Session() as s:
            rows = s.scalars(
                select(PaperPosition).where(
                    PaperPosition.alpha_id == alpha_id, PaperPosition.date == d)
            ).all()
            return {r.ticker: r.weight for r in rows}

    def _equity_on(self, alpha_id: int, d) -> float:
        with self._Session() as s:
            row = s.scalars(
                select(PaperDailyPnL).where(
                    PaperDailyPnL.alpha_id == alpha_id, PaperDailyPnL.date == d)
            ).first()
            return row.equity if row else 1.0

    def pnl_history(self, alpha_id: int, limit: int = 500) -> List[PaperDailyPnL]:
        with self._Session() as s:
            rows = list(s.scalars(
                select(PaperDailyPnL).where(PaperDailyPnL.alpha_id == alpha_id)
                .order_by(PaperDailyPnL.date.desc()).limit(limit)
            ))
        return list(reversed(rows))

    def fills_on(self, alpha_id: int, date) -> List[PaperFill]:
        d = _to_date(date)
        with self._Session() as s:
            return list(s.scalars(
                select(PaperFill).where(
                    PaperFill.alpha_id == alpha_id, PaperFill.date == d)
            ))

    def fills_in_range(self, start, end, alpha_id: Optional[int] = None) -> List[PaperFill]:
        """返回 [start, end] 内的成交记录（可选按 alpha 过滤）——供成本校准（Task 8.2）。"""
        d0, d1 = _to_date(start), _to_date(end)
        with self._Session() as s:
            stmt = select(PaperFill).where(PaperFill.date >= d0, PaperFill.date <= d1)
            if alpha_id is not None:
                stmt = stmt.where(PaperFill.alpha_id == alpha_id)
            return list(s.scalars(stmt.order_by(PaperFill.date, PaperFill.ticker)))
