"""
trial_ledger.py — 统计诚实性台账（Phase S.2 + S.3）

本模块存放两本**互补**的账，它们回答的是同一个问题的两半：
"这个数字是从多少次尝试里挑出来的，以及我们已经偷看过样本外几次"。

TrialLedger（S.3）—— 全局多重检验计数器
---------------------------------------
DSR（去膨胀夏普）的诚实性取决于 `n_trials`——"为选出这个因子，一共试过多少个策略"。
现状默认 1（最宽松），系统性**低估膨胀**。正确的口径是**整个研究史的累计**：跨会话、跨 GP run、
跨 Optuna trial。本模块提供一个**持久化、跨会话**的累计计数器，验证门用它作为 DSR 的 n_trials。

单行表 `trial_ledger(id=1, total, updated_at)`。append-only 语义（只增，reset 仅供测试）。

HoldoutLedger（S.2）—— 冻结 Test 段的使用次数
---------------------------------------------
"一次性使用"如果只是文档里的一句话，它**必然**被违反：同一个 test 段跑第 20 次时，
选出来的东西早就是按它挑的了，而屏幕上的数字看起来和第 1 次一模一样。
所以次数必须被**记下来并报出来**：每次有人真的在 Test 段上算了指标，就 +1；
超过预算阈值时返回 `over_budget=True`，调用方把它落进 RunManifest / 响应体。

append-only 表 `holdout_usages(dataset_key, test_key, purpose, used_at)`——
只增不改，`test_key` 是 Test 段的起止日期（`ThreeWaySplit.test_key`），
换了日期就是另一个段、另起一本账。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# HoldoutLedger — 冻结 Test 段的使用次数（Phase S.2）
# ---------------------------------------------------------------------------

class HoldoutUse(_Base):
    __tablename__ = "holdout_usages"
    id:          int      = Column(Integer, primary_key=True, autoincrement=True)
    dataset_key: str      = Column(String(128), nullable=False, index=True)
    test_key:    str      = Column(String(64),  nullable=False, index=True)
    purpose:     str      = Column(String(64),  nullable=False, default="")
    used_at:     datetime = Column(DateTime, default=datetime.utcnow, nullable=False)


@dataclass(frozen=True)
class HoldoutUsage:
    """一次 Test 段使用的记账结果。"""
    dataset_key: str
    test_key:    str
    uses:        int          # 含本次在内的累计使用次数
    budget:      int          # 允许的次数上限
    over_budget: bool         # uses > budget
    recorded:    bool = True  # False = 台账不可写（结论仍可用，但次数已失真）

    def to_dict(self) -> dict:
        return {
            "dataset_key": self.dataset_key, "test_key": self.test_key,
            "uses": self.uses, "budget": self.budget,
            "over_budget": self.over_budget, "recorded": self.recorded,
        }


class HoldoutLedger:
    """
    冻结 Test 段的**一次性使用**台账（append-only）。

    `budget` 默认 1 —— 字面意义的"一次性"。超了不阻断（阻断会让人干脆绕开台账，
    那就彻底看不见了），而是把 `over_budget=True` 一路带到响应体与 RunManifest：
    **让"这个样本外已经被看过 N 次"和数字本身一起出现**。
    """

    def __init__(self, db_url: Optional[str] = None, budget: int = 1) -> None:
        if budget < 1:
            raise ValueError(f"budget 至少为 1，当前={budget}")
        self.budget = int(budget)
        if db_url is None:
            db_url = os.getenv("DATABASE_URL", "")
            if not db_url:
                try:
                    from app.config import settings
                    db_url = settings.database_url
                except Exception as exc:
                    # 落到默认库路径会让这本账**记到另一个文件里** —— 看起来一切正常，
                    # 而"这段 holdout 用过几次"从此对不上号。必须留痕。
                    logger.warning(
                        "[S.2] 读不到 database_url，holdout 台账落到默认 alphas.db：%s", exc)
                    db_url = "sqlite:///alphas.db"
        connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        self._engine = create_engine(db_url, connect_args=connect_args, echo=False)
        from ._sqlite_utils import harden_sqlite_engine
        harden_sqlite_engine(self._engine)
        _Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    def count(self, dataset_key: str, test_key: str) -> int:
        with self._Session() as s:
            return int(s.scalar(
                select(func.count()).select_from(HoldoutUse)
                .where(HoldoutUse.dataset_key == dataset_key)
                .where(HoldoutUse.test_key == test_key)
            ) or 0)

    def record_use(
        self, dataset_key: str, test_key: str, purpose: str = "",
    ) -> HoldoutUsage:
        """记一次 Test 段使用并返回记账结果（含本次的累计次数）。"""
        with self._Session() as s:
            s.add(HoldoutUse(dataset_key=dataset_key, test_key=test_key,
                             purpose=purpose or ""))
            s.commit()
            uses = int(s.scalar(
                select(func.count()).select_from(HoldoutUse)
                .where(HoldoutUse.dataset_key == dataset_key)
                .where(HoldoutUse.test_key == test_key)
            ) or 1)
        usage = HoldoutUsage(
            dataset_key=dataset_key, test_key=test_key,
            uses=uses, budget=self.budget, over_budget=uses > self.budget,
        )
        if usage.over_budget:
            logger.error(
                "[S.2] 冻结 Test 段 %s@%s 已被使用 %d 次（预算 %d）—— "
                "它**不再是样本外**：反复在同一段上汇报，等于把它也变成了选择集。"
                "该段上的结论应按『已挖掘』解读，或换一段真正没看过的数据。",
                test_key, dataset_key, uses, self.budget,
            )
        return usage

    def reset(self) -> None:
        """仅供测试：清空使用记录。"""
        with self._Session() as s:
            for row in s.scalars(select(HoldoutUse)).all():
                s.delete(row)
            s.commit()
