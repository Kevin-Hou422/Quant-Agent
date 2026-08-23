"""
trial_ledger.py — 全局多重检验计数器（Phase S.3）

角色
----
DSR（去膨胀夏普）的诚实性取决于 `n_trials`——"为选出这个因子，一共试过多少个策略"。
现状默认 1（最宽松），系统性**低估膨胀**。正确的口径是**整个研究史的累计**：跨会话、跨 GP run、
跨 Optuna trial。本模块提供一个**持久化、跨会话**的累计计数器，验证门用它作为 DSR 的 n_trials。

单行表 `trial_ledger(id=1, total, updated_at)`。append-only 语义（只增，reset 仅供测试）。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, create_engine, select
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class _Base(DeclarativeBase):
    pass


class TrialCount(_Base):
    __tablename__ = "trial_ledger"
    id:         int      = Column(Integer, primary_key=True)      # 固定 =1
    total:      int      = Column(Integer, default=0, nullable=False)
    updated_at: datetime = Column(DateTime, default=datetime.utcnow)


class TrialLedger:
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

    def add(self, n: int) -> int:
        """累加 n 个 trial，返回新的累计总数。n<=0 时不变。"""
        n = max(0, int(n))
        with self._Session() as s:
            row = s.get(TrialCount, 1)
            if row is None:
                row = TrialCount(id=1, total=0)
                s.add(row)
            row.total = int(row.total) + n
            row.updated_at = datetime.utcnow()
            s.commit()
            return int(row.total)

    def total(self) -> int:
        with self._Session() as s:
            row = s.scalars(select(TrialCount).where(TrialCount.id == 1)).first()
            return int(row.total) if row else 0

    def reset(self) -> None:
        """仅供测试：清零。"""
        with self._Session() as s:
            row = s.get(TrialCount, 1)
            if row is not None:
                row.total = 0
                s.commit()
