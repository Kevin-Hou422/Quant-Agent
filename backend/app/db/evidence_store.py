"""
evidence_store.py — 前向证据：先存信号，后结算（外部审查 2026-10-18 F06）

此前的问题
----------
- 前向 IC 写进 `alpha_ic_history` 后可被回放覆盖（R12：先写前向 0.1，回放写 −0.9，读出来 −0.9 且仍标前向）；
- "是不是前向"由 `forward_from` 决定：停机几天后一次补拉的 bar 全部标成前向，可那几天的信号
  从没在当时产生过；摄取落盘后计算失败、重跑时 `forward_from=None`，证据又会丢标；
- 组合 IC 全写在 book 0，策略激活时只按"创建日期之后"筛 —— 那些数可能来自别的策略。

现在
----
- **信号存档**（`signal_archive`）：每次运行把当天的信号向量按 (账本键, 信号日期) 存一份，只写一次；
- **前向 = 结算时用的是在结果发生之前就存好的那份信号**：日期 d 的 IC 用 d 的前一根 bar 的存档信号，
  且存档时刻必须早于 d 的收盘。补拉的、重跑的、离线补算的都对不上这一条，自然只算回放；
- **按策略版本记账**（`strategy_evidence`）：键是 (策略 id, 日期)，并记规格哈希、信号日期、决策时刻；
  只存前向样本，写入后不可改；
- 任何想改已存证据的写入都不生效，记一条 `evidence_revisions`（带原因），人看得见。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date as _date, datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import (Column, Date, DateTime, Float, Integer, String, Text, UniqueConstraint,
                        create_engine, select)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)


class _Base(DeclarativeBase):
    pass


class SignalArchive(_Base):
    __tablename__ = "signal_archive"
    __table_args__ = (UniqueConstraint("book_key", "asof", name="uq_signal_archive"),)
    id:          int      = Column(Integer, primary_key=True, autoincrement=True)
    book_key:    str      = Column(String(64), nullable=False, index=True)   # alpha:12 / portfolio / strategy:5
    asof:        object   = Column(Date, nullable=False)
    spec_hash:   str      = Column(String(64), nullable=False, default="")
    payload:     str      = Column(Text, nullable=False)                     # {ticker: value}
    recorded_at: datetime = Column(DateTime, nullable=False)                 # UTC，naive
    # 复审 2026-10-19 N02-c：这份信号 / 这次结算用的输入（见 data_engine/input_identity.py）
    price_source:    str      = Column(String(16), nullable=True)
    data_digest:     str      = Column(String(64), nullable=True)
    universe_digest: str      = Column(String(16), nullable=True)
    universe_n:      int      = Column(Integer, nullable=True)
    vintage:         str      = Column(String(40), nullable=True)
    code_version:    str      = Column(String(40), nullable=True)


class StrategyEvidence(_Base):
    __tablename__ = "strategy_evidence"
    __table_args__ = (UniqueConstraint("strategy_id", "date", name="uq_strategy_evidence"),)
    id:           int      = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id:  int      = Column(Integer, nullable=False, index=True)
    spec_hash:    str      = Column(String(64), nullable=False)
    date:         object   = Column(Date, nullable=False)        # 结果发生的交易日
    ic:           float    = Column(Float, nullable=False)
    signal_asof:  object   = Column(Date, nullable=False)        # 用的是哪一天的信号
    decision_ts:  datetime = Column(DateTime, nullable=False)    # 那份信号存档的时刻
    recorded_at:  datetime = Column(DateTime, nullable=False)
    # 复审 2026-10-19 N02-c：**结算**时用的输入；决策时的输入摘要在 signal_digest（见 data_engine/input_identity.py）
    price_source:    str      = Column(String(16), nullable=True)
    data_digest:     str      = Column(String(64), nullable=True)
    universe_digest: str      = Column(String(16), nullable=True)
    universe_n:      int      = Column(Integer, nullable=True)
    vintage:         str      = Column(String(40), nullable=True)
    code_version:    str      = Column(String(40), nullable=True)
    signal_digest:   str      = Column(String(64), nullable=True)


class EvidenceRevision(_Base):
    """想改已存证据的写入：不生效，但留痕（谁、改成什么、为什么）。"""
    __tablename__ = "evidence_revisions"
    id:        int      = Column(Integer, primary_key=True, autoincrement=True)
    kind:      str      = Column(String(32), nullable=False)     # alpha_ic / signal_archive / strategy_ic
    key:       str      = Column(String(64), nullable=False)
    date:      object   = Column(Date, nullable=False)
    old_value: str      = Column(Text, default="")
    new_value: str      = Column(Text, default="")
    reason:    str      = Column(Text, default="")
    at:        datetime = Column(DateTime, nullable=False)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _naive_utc(ts: Optional[datetime]) -> datetime:
    if ts is None:
        return _utcnow()
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def _as_date(d) -> _date:
    return pd.Timestamp(d).date()


def session_deadline(d) -> Optional[datetime]:
    """
    结算 d 这一天的结果时，信号最晚的存档时刻 = **d 的开盘**（UTC，naive）。

    复审 2026-10-19 N03：此前截止是 d 的收盘 —— 收盘前 1 分钟才存档的信号也算"前向"，可那一天的
    大部分收益已经发生了。被评分的收益是前一个收盘 → d 收盘；信号只能在前一个收盘之后才算得出来，
    所以最严、又做得到的截止是 d 开盘之前：开盘前存档的信号还来得及在 d 当天交易。
    （前一个收盘到 d 开盘之间的隔夜跳空仍在被评分窗口里 —— 标签口径是 D14 要统一的事。）
    算不出来（日历不可用 / 非交易日）→ None → 不认前向（保守）。
    """
    try:
        from app.core.data_engine.market_calendar import session_open_utc
        t = session_open_utc(pd.Timestamp(d))
    except Exception as exc:
        logger.warning("[evidence] 取 %s 开盘时刻失败，这一天不认前向: %s", d, exc)
        return None
    return None if t is None else _naive_utc(t)


class EvidenceStore:
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
        self._migrate_add_columns()
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    #: 后加的可空列（create_all 不会给已存在的表补列）
    _ADDED_COLUMNS = {
        "signal_archive": {"price_source": "VARCHAR(16)", "data_digest": "VARCHAR(64)",
                           "universe_digest": "VARCHAR(16)", "universe_n": "INTEGER",
                           "vintage": "VARCHAR(40)", "code_version": "VARCHAR(40)"},
        "strategy_evidence": {"price_source": "VARCHAR(16)", "data_digest": "VARCHAR(64)",
                              "universe_digest": "VARCHAR(16)", "universe_n": "INTEGER",
                              "vintage": "VARCHAR(40)", "code_version": "VARCHAR(40)",
                              "signal_digest": "VARCHAR(64)"},
    }

    def _migrate_add_columns(self) -> None:
        """只加、不改、不删（2026-10-18 那版建的表没有输入身份列）。"""
        from sqlalchemy import inspect, text
        insp = inspect(self._engine)
        for table, cols in self._ADDED_COLUMNS.items():
            have = {c["name"] for c in insp.get_columns(table)}
            for name, typ in cols.items():
                if name not in have:
                    with self._engine.begin() as conn:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {typ}"))

    # ------------------------------------------------------------------ 信号存档
    def archive(self, book_key: str, asof, vector: pd.Series, spec_hash: str = "",
                recorded_at: Optional[datetime] = None, identity: Optional[dict] = None) -> bool:
        """
        存一份信号（只写一次）。已有且内容不同 → 不改，记修订事件。返回是否新写入。
        identity：算出这份信号时用的输入（InputIdentity.to_dict()），一并存下。
        """
        ident = _ident_cols(identity)
        payload = json.dumps({str(k): (None if not np.isfinite(v) else float(v))
                              for k, v in vector.astype(float).items()}, sort_keys=True)
        day = _as_date(asof)
        with self._Session() as s:
            s.add(SignalArchive(book_key=book_key, asof=day, spec_hash=spec_hash or "",
                                payload=payload, recorded_at=_naive_utc(recorded_at), **ident))
            try:
                s.commit()
                return True
            except IntegrityError:
                s.rollback()
                logger.debug("[evidence] %s 在 %s 已有存档，保留最早的那份", book_key, day)
            old = s.scalars(select(SignalArchive).where(SignalArchive.book_key == book_key,
                                                        SignalArchive.asof == day)).first()
        if old is not None and (old.payload != payload or old.spec_hash != (spec_hash or "")):
            self.log_revision("signal_archive", book_key, day, old.payload[:200], payload[:200],
                              "同一天的信号已存档，新算出来的不同（数据被重算或规格变了）—— 保留最早的那份")
        return False

    def archived_before(self, book_key: str, asof, deadline: Optional[datetime]) -> Optional[dict]:
        """(asof 那天的存档信号, 存档时刻, 规格哈希)，且存档时刻早于 deadline。否则 None。"""
        if deadline is None:
            return None
        with self._Session() as s:
            row = s.scalars(select(SignalArchive).where(SignalArchive.book_key == book_key,
                                                        SignalArchive.asof == _as_date(asof))).first()
        if row is None or row.recorded_at >= deadline:
            return None
        vec = pd.Series({k: (np.nan if v is None else v) for k, v in json.loads(row.payload).items()},
                        dtype=float)
        return {"signal": vec, "recorded_at": row.recorded_at, "spec_hash": row.spec_hash,
                "price_source": row.price_source or "", "data_digest": row.data_digest or ""}

    def archive_dates(self, book_key: str) -> Dict[_date, datetime]:
        """该账本键所有存档的 {信号日期: 存档时刻}（不读向量本身，结算前先用它筛）。"""
        with self._Session() as s:
            rows = s.execute(select(SignalArchive.asof, SignalArchive.recorded_at)
                             .where(SignalArchive.book_key == book_key)).all()
        return {a: r for a, r in rows}

    def forward_signal(self, book_key: str, prev_day, day) -> Optional[dict]:
        """结算 `day` 的结果时可用的前向信号：前一根 bar 的存档，且存于 `day` 收盘之前。"""
        return self.archived_before(book_key, prev_day, session_deadline(day))

    def book(self, book_key: str) -> "ForwardBook":
        return ForwardBook(self, book_key)

    # ------------------------------------------------------------------ 策略版本证据
    def record_strategy_ic(self, strategy_id: int, spec_hash: str, day, ic: float,
                           signal_asof, decision_ts: datetime, identity: Optional[dict] = None,
                           signal_digest: str = "") -> bool:
        day = _as_date(day)
        with self._Session() as s:
            s.add(StrategyEvidence(strategy_id=int(strategy_id), spec_hash=spec_hash, date=day,
                                   ic=float(ic), signal_asof=_as_date(signal_asof),
                                   decision_ts=_naive_utc(decision_ts), recorded_at=_utcnow(),
                                   signal_digest=signal_digest or None, **_ident_cols(identity)))
            try:
                s.commit()
                return True
            except IntegrityError:
                s.rollback()
                logger.debug("[evidence] 策略 #%s 在 %s 已有前向证据，不改写", strategy_id, day)
            old = s.scalars(select(StrategyEvidence).where(
                StrategyEvidence.strategy_id == int(strategy_id), StrategyEvidence.date == day)).first()
        if old is not None and (abs(old.ic - float(ic)) > 1e-12 or old.spec_hash != spec_hash):
            self.log_revision("strategy_ic", f"strategy:{strategy_id}", day,
                              f"{old.ic}@{old.spec_hash[:12]}", f"{ic}@{spec_hash[:12]}",
                              "前向证据已存，不可改写")
        return False

    def strategy_forward_ic(self, strategy_id: int, spec_hash: Optional[str] = None) -> List[StrategyEvidence]:
        with self._Session() as s:
            q = select(StrategyEvidence).where(StrategyEvidence.strategy_id == int(strategy_id))
            if spec_hash:
                q = q.where(StrategyEvidence.spec_hash == spec_hash)
            return list(s.scalars(q.order_by(StrategyEvidence.date)).all())

    # ------------------------------------------------------------------ 修订事件
    def log_revision(self, kind: str, key: str, day, old: str, new: str, reason: str) -> None:
        logger.warning("[evidence] 拒绝改写已存证据 %s %s@%s：%s → %s（%s）", kind, key, day, old, new, reason)
        with self._Session() as s:
            s.add(EvidenceRevision(kind=kind, key=key, date=_as_date(day), old_value=str(old),
                                   new_value=str(new), reason=reason, at=_utcnow()))
            s.commit()

    def revisions(self, kind: Optional[str] = None) -> List[EvidenceRevision]:
        with self._Session() as s:
            q = select(EvidenceRevision).order_by(EvidenceRevision.id)
            if kind:
                q = q.where(EvidenceRevision.kind == kind)
            return list(s.scalars(q).all())


def _ident_cols(identity: Optional[dict]) -> dict:
    keys = ("price_source", "data_digest", "universe_digest", "universe_n", "vintage", "code_version")
    return {k: (identity or {}).get(k) for k in keys}


class ForwardBook:
    """
    一个账本键的前向结算助手：先一次取出全部存档日期，逐日只在可能是前向时才读向量。
    identity：本轮的输入身份。复审 2026-10-19 N02-d：存档时的价格源与本轮不同 → 不当前向结算
    （不同数据源的历史与前向证据不能混用）。
    """

    def __init__(self, store: Optional[EvidenceStore], book_key: str,
                 identity: Optional[dict] = None) -> None:
        self.store, self.key, self.identity = store, book_key, identity or {}
        try:
            self._idx = store.archive_dates(book_key) if store is not None else {}
        except Exception as exc:
            logger.error("[evidence] 读 %s 的信号存档失败 → 本轮全部按回放记: %s", book_key, exc)
            self._idx = {}

    def signal_for(self, prev_day, day) -> Optional[dict]:
        """结算 `day` 时可用的前向信号（前一根 bar 的存档、存于 `day` 收盘前），没有 → None。"""
        rec_at = self._idx.get(_as_date(prev_day))
        if rec_at is None:
            return None
        deadline = session_deadline(day)
        if deadline is None or rec_at >= deadline:
            return None
        arch = self.store.archived_before(self.key, prev_day, deadline)
        now_src = str(self.identity.get("price_source") or "")
        if arch is not None and arch["price_source"] and now_src and arch["price_source"] != now_src:
            logger.warning("[evidence] %s 在 %s 的存档信号来自 %s，本轮数据来自 %s → 不当前向结算",
                           self.key, _as_date(prev_day), arch["price_source"], now_src)
            return None
        return arch

    def archive(self, asof, vector: pd.Series, now: Optional[datetime], spec_hash: str = "") -> None:
        if self.store is None:
            return
        try:
            self.store.archive(self.key, asof, vector, spec_hash=spec_hash, recorded_at=now,
                               identity=self.identity)
        except Exception as exc:
            # 存不上 → 明天这一天的结果只能算回放（前向证据少一天），不影响今天的交易
            logger.error("[evidence] %s 的 %s 信号存档失败（明天的结算不会算前向）: %s",
                         self.key, pd.Timestamp(asof).date(), exc)
