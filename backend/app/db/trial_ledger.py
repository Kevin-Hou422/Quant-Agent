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

**谁来记**（外部审查 2026-10-18 F05）：搜索的**底层**记 —— GP 进化（`PopulationEvolver.run`）
按实际评估过的个体数记，Optuna（`AlphaOptimizer.optimize`）按实际跑完的 trial 数记。
此前只有自主发现编排按配置**估算**一次，直接 GP、工作流、Optuna、聊天工具都不记。
每次运行带一个 `run_id`，记在 `trial_runs` 表里：同一个 run 记两次只算一次（重试不会重复计数）；
累加用一条 `UPDATE total = total + n`，两个进程同时记也不会丢。

HoldoutLedger（S.2）—— 冻结 Test 段的使用次数
---------------------------------------------
"一次性使用"如果只是文档里的一句话，它**必然**被违反：同一个 test 段跑第 20 次时，
选出来的东西早就是按它挑的了，而屏幕上的数字看起来和第 1 次一模一样。
所以次数必须被**记下来并报出来**：每次有人真的在 Test 段上算了指标，就 +1；
超过预算阈值时返回 `over_budget=True`，调用方把它落进 RunManifest / 响应体。

append-only 表 `holdout_usages(dataset_key, test_key, purpose, used_at)`——
只增不改，`test_key` 是 Test 段的起止日期（`ThreeWaySplit.test_key`）。

**按区间重叠计次**（外部审查 2026-10-18 F04）：此前"换了日期就是另一个段、另起一本账"，
于是数据每多一天、Test 窗口滑动一天，就换来一份全新的预算 —— 审查的 R1 里两个窗口重叠 521 天，
两次都显示"第 1 次使用"。现在一次使用的次数 = 同一数据集里与它**有任何重叠**的历史使用数 + 1。
看过的数据不会因为窗口滑动而恢复清白；要一个干净的样本外，只能等新收集的、从没被看过的数据。

FrozenBoundary —— 冻结边界持久化（F04）
---------------------------------------
Test 起点此前由每次的数据末日倒推 2 年得出，数据每天增长它就每天滑动。
现在每个**真实数据集**第一次切分时把 Test 起点（绝对日期）写进 `frozen_boundaries`，
以后不管数据截到哪天，都用这个日期切。选择段从此固定；新数据全部落在 Test 一侧。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from typing import List, Tuple

import pandas as pd
from sqlalchemy import Column, DateTime, Integer, String, create_engine, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)


class _Base(DeclarativeBase):
    pass


class TrialCount(_Base):
    __tablename__ = "trial_ledger"
    id:         int      = Column(Integer, primary_key=True)      # 固定 =1
    total:      int      = Column(Integer, default=0, nullable=False)
    updated_at: datetime = Column(DateTime, default=datetime.utcnow)


class TrialRun(_Base):
    """一次搜索运行记过的试验数（`run_id` 唯一 → 同一次运行重复记账只算一次）。"""
    __tablename__ = "trial_runs"
    run_id:     str      = Column(String(64), primary_key=True)
    n:          int      = Column(Integer, nullable=False)
    source:     str      = Column(String(32), nullable=False, default="")
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)


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

    def add(self, n: int, run_id: Optional[str] = None, source: str = "") -> int:
        """
        累加 n 个 trial，返回新的累计总数。n<=0 时不变。

        `run_id` 给定时幂等：这个 run 已经记过 → 不再累加，直接返回当前总数。
        累加是一条原子 UPDATE（两个进程同时记不会互相覆盖）。
        """
        n = max(0, int(n))
        with self._Session() as s:
            if run_id:
                s.add(TrialRun(run_id=str(run_id)[:64], n=n, source=(source or "")[:32]))
                try:
                    s.flush()
                except IntegrityError:
                    s.rollback()
                    logger.info("[S.3] run %s 已经记过账，不重复累加", run_id)
                    return self.total()
            if s.get(TrialCount, 1) is None:
                s.add(TrialCount(id=1, total=0))
                s.flush()
            s.execute(update(TrialCount).where(TrialCount.id == 1)
                      .values(total=TrialCount.total + n, updated_at=datetime.utcnow()))
            s.commit()
            return int(s.scalar(select(TrialCount.total).where(TrialCount.id == 1)) or 0)

    def total(self) -> int:
        with self._Session() as s:
            row = s.scalars(select(TrialCount).where(TrialCount.id == 1)).first()
            return int(row.total) if row else 0

    def runs(self, source: Optional[str] = None) -> List[TrialRun]:
        with self._Session() as s:
            q = select(TrialRun).order_by(TrialRun.created_at)
            if source:
                q = q.where(TrialRun.source == source)
            return list(s.scalars(q).all())

    def reset(self) -> None:
        """仅供测试：清零。"""
        with self._Session() as s:
            row = s.get(TrialCount, 1)
            if row is not None:
                row.total = 0
            for r in s.scalars(select(TrialRun)).all():
                s.delete(r)
            s.commit()


def record_trials(n: int, source: str, run_id: Optional[str] = None) -> bool:
    """
    搜索底层用的记账入口：记 n 个试验，返回是否记上。

    记不上**不打断搜索**（搜索结果本身没错），但必须让调用方知道 —— 调用方把
    `trials_recorded=False` 带进结果，验证门据此把"全局试验数"标成不完整。
    """
    import uuid
    if int(n) <= 0:
        return True
    try:
        TrialLedger().add(int(n), run_id=run_id or f"{source}:{uuid.uuid4().hex}", source=source)
        return True
    except Exception as exc:
        logger.error("[S.3] 试验数记账失败（%s 的 %d 次尝试没进全局计数，DSR 会少校正）: %s",
                     source, n, exc)
        return False


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
        """该数据集里与 `test_key` 这段区间**有重叠**的使用次数（区间解析不了时按键精确匹配）。"""
        with self._Session() as s:
            keys = list(s.scalars(select(HoldoutUse.test_key)
                                  .where(HoldoutUse.dataset_key == dataset_key)).all())
        return _count_overlapping(keys, test_key)

    def record_use(
        self, dataset_key: str, test_key: str, purpose: str = "",
    ) -> HoldoutUsage:
        """记一次 Test 段使用并返回记账结果（含本次的累计次数）。"""
        with self._Session() as s:
            s.add(HoldoutUse(dataset_key=dataset_key, test_key=test_key,
                             purpose=purpose or ""))
            s.commit()
        uses = max(1, self.count(dataset_key, test_key))
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


def _parse_window(test_key: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    try:
        a, b = str(test_key).split("..")
        return pd.Timestamp(a), pd.Timestamp(b)
    except (ValueError, TypeError) as exc:      # "empty" 或旧格式：只能按键精确匹配
        logger.debug("[S.2] test_key %r 不是日期区间: %s", test_key, exc)
        return None


def _count_overlapping(keys: List[str], test_key: str) -> int:
    me = _parse_window(test_key)
    if me is None:
        return sum(1 for k in keys if k == test_key)
    n = 0
    for k in keys:
        w = _parse_window(k)
        if w is None:
            continue
        if w[0] <= me[1] and me[0] <= w[1]:
            n += 1
    return n


# ---------------------------------------------------------------------------
# FrozenBoundary — 冻结 Test 起点的持久化（外部审查 2026-10-18 F04）
# ---------------------------------------------------------------------------

class FrozenBoundary(_Base):
    __tablename__ = "frozen_boundaries"
    dataset_key: str      = Column(String(128), primary_key=True)
    test_start:  str      = Column(String(10), nullable=False)       # YYYY-MM-DD，含当天
    frozen_by:   str      = Column(String(16), nullable=False, default="")
    created_at:  datetime = Column(DateTime, default=datetime.utcnow, nullable=False)


class BoundaryStore:
    """冻结边界：每个数据集一行，**只写一次**。改它是一个需要人拍板的研究决定，不在代码里提供。"""

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

    def get(self, dataset_key: str) -> Optional[pd.Timestamp]:
        with self._Session() as s:
            row = s.get(FrozenBoundary, dataset_key)
            return pd.Timestamp(row.test_start) if row is not None else None

    def pin(self, dataset_key: str, test_start: pd.Timestamp, frozen_by: str = "") -> pd.Timestamp:
        """第一次调用写入并返回 `test_start`；已有边界 → 不改，返回已有的那个（两个进程抢着写也一样）。"""
        with self._Session() as s:
            s.add(FrozenBoundary(dataset_key=dataset_key,
                                 test_start=str(pd.Timestamp(test_start).date()),
                                 frozen_by=frozen_by[:16]))
            try:
                s.commit()
                logger.warning("[S.2] 数据集 %s 的冻结 Test 起点固定为 %s（%s）—— 以后不随数据末日滑动",
                               dataset_key, pd.Timestamp(test_start).date(), frozen_by)
            except IntegrityError:
                s.rollback()
                logger.info("[S.2] 数据集 %s 已有冻结边界，沿用已有的", dataset_key)
        got = self.get(dataset_key)
        if got is None:
            raise RuntimeError(f"冻结边界写入后读不回来：{dataset_key}")
        return got
