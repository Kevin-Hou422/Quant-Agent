"""
corporate_actions.py — 疑似未知公司行动的隔离标记（2026-10-22 用户决定第 18 条 / F02′）

识别见 `app/core/data_engine/corporate_actions.py`。一只票被标记后一直处于隔离（不加风险）状态，
直到**人**核对完券商股数 / 现金与数据源复权、写明结论后结案（`resolve`）—— 不会因为价格序列后来被复权
"看起来正常了"就自己消失。同一只票同一天只记一次。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, UniqueConstraint, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)


class _Base(DeclarativeBase):
    pass


class CorporateActionFlag(_Base):
    __tablename__ = "corporate_action_flags"
    id:            int      = Column(Integer, primary_key=True, autoincrement=True)
    ticker:        str      = Column(String(32), nullable=False, index=True)
    bar_date:      str      = Column(String(10), nullable=False)          # 跳变发生的那根 bar
    kind:          str      = Column(String(32), nullable=False)          # suspected / broker_share_ratio …
    ratio:         float    = Column(Float, nullable=True)                # 价格比例（今 / 昨）或股数比例
    nearest_split: str      = Column(String(16), nullable=True)
    detected_at:   datetime = Column(DateTime, nullable=False)
    status:        str      = Column(String(16), nullable=False, default="open")   # open / resolved
    resolved_by:   str      = Column(String(64), nullable=True)
    resolution:    str      = Column(Text, nullable=True)
    resolved_at:   datetime = Column(DateTime, nullable=True)
    __table_args__ = (UniqueConstraint("ticker", "bar_date", name="uq_ca_flag"),)


class CorporateActionStore:
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
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    def flag(self, ticker: str, bar_date: str, *, kind: str, ratio: Optional[float] = None,
             nearest_split: Optional[str] = None) -> bool:
        """记一条隔离标记。已有（同票同日）→ False，不重复记。"""
        row = CorporateActionFlag(ticker=str(ticker), bar_date=str(bar_date), kind=str(kind)[:32],
                                  ratio=None if ratio is None else float(ratio), nearest_split=nearest_split,
                                  detected_at=datetime.now(timezone.utc).replace(tzinfo=None), status="open")
        with self._Session() as s:
            s.add(row)
            try:
                s.commit()
            except IntegrityError:
                s.rollback()
                logger.debug("[corporate_action] %s 在 %s 已有隔离标记，不重复记", ticker, bar_date)
                return False
        logger.error("[corporate_action] %s 在 %s 疑似未知公司行动（%s，比例 %s，像 %s）→ 隔离：不加风险，待人对账",
                     ticker, bar_date, kind, ratio, nearest_split)
        return True

    def open_tickers(self) -> Dict[str, str]:
        """仍在隔离中的票 → 最早一条未结案标记的日期。"""
        with self._Session() as s:
            rows = s.scalars(select(CorporateActionFlag).where(CorporateActionFlag.status == "open")
                             .order_by(CorporateActionFlag.bar_date)).all()
        out: Dict[str, str] = {}
        for r in rows:
            out.setdefault(r.ticker, r.bar_date)
        return out

    def resolve(self, ticker: str, actor: str, resolution: str) -> int:
        """人对账后结案（这只票所有未结案的标记）。必须写明操作人与结论。返回结案条数。"""
        actor, resolution = (actor or "").strip(), (resolution or "").strip()
        if not actor or not resolution:
            raise ValueError("结案必须写明操作人和对账结论（券商股数 / 现金、数据源复权）")
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with self._Session() as s:
            rows = s.scalars(select(CorporateActionFlag).where(CorporateActionFlag.ticker == str(ticker),
                                                               CorporateActionFlag.status == "open")).all()
            for r in rows:
                r.status, r.resolved_by, r.resolution, r.resolved_at = "resolved", actor[:64], resolution, now
            s.commit()
        logger.warning("[corporate_action] %s 的 %d 条隔离标记结案（%s：%s）", ticker, len(rows), actor, resolution)
        return len(rows)

    def history(self, ticker: str) -> List[dict]:
        with self._Session() as s:
            rows = s.scalars(select(CorporateActionFlag).where(CorporateActionFlag.ticker == str(ticker))
                             .order_by(CorporateActionFlag.id)).all()
        return [{"bar_date": r.bar_date, "kind": r.kind, "ratio": r.ratio, "nearest_split": r.nearest_split,
                 "status": r.status, "resolved_by": r.resolved_by, "resolution": r.resolution} for r in rows]
