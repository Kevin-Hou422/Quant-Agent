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

FrozenBoundary —— 冻结边界持久化（F04 / 2026-10-22 D12）
--------------------------------------------------------
Test 起点此前由每次的数据末日倒推 2 年得出，数据每天增长它就每天滑动。F04 改成"第一次切分时写死"，
D12 指出这又让边界取决于**哪一次调用碰巧最先发生**。现在：边界只能由人**登记**
（`BoundaryStore.register`：日期 + 操作人 + 理由，只写一次），研究入口在登记之前按"临时切分"跑、不写库；
正式结论（验证门、策略提案）要求已登记的边界。旧代码首次调用写下的行（没有操作人）照样冻结 Test 段
（冻结的数据不能被解冻），但算"未确认"，正式结论同样等人确认。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from typing import List, Tuple

import pandas as pd
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, select, update
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


class TrialRunState(_Base):
    """
    一次搜索运行的生命周期（复审 2026-10-19 N07）：开始前登记 running，逐批累加 n_recorded，
    结束时置 done / aborted。进程被杀、机器重启 → 永远停在 running → 过了 STALE_AFTER 就是
    "试验史不完整"，门据此拒绝给出正式结论，直到有人用 resolve_run 补记并写明理由。
    """
    __tablename__ = "trial_run_state"
    run_id:      str      = Column(String(64), primary_key=True)
    source:      str      = Column(String(32), nullable=False, default="")
    status:      str      = Column(String(16), nullable=False, default="running")
    n_recorded:  int      = Column(Integer, nullable=False, default=0)
    started_at:  datetime = Column(DateTime, nullable=False)
    finished_at: datetime = Column(DateTime, nullable=True)
    note:        str      = Column(String(512), nullable=False, default="")
    #: 这次运行所在的研究上下文（ResearchContext.to_json()：dataset / universe / experiment / boundary）；
    #: 空 = 不在任何上下文里跑的（复审 2026-10-19 N09-c）
    context:     str      = Column(Text, nullable=True)


#: running 状态超过这么久还没结束 → 视为中断（GP / Optuna 单次运行远短于此）
STALE_AFTER = timedelta(hours=6)


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
        self._migrate_add_columns()
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    #: 后加的可空列（create_all 不会给已存在的表补列）
    _ADDED_COLUMNS = {"trial_run_state": {"context": "TEXT"}}

    def _migrate_add_columns(self) -> None:
        """只加、不改、不删。"""
        from sqlalchemy import inspect, text
        insp = inspect(self._engine)
        for table, cols in self._ADDED_COLUMNS.items():
            have = {c["name"] for c in insp.get_columns(table)}
            for name, typ in cols.items():
                if name not in have:
                    with self._engine.begin() as conn:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {typ}"))

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

    # ------------------------------------------------------------------ 运行生命周期（N07）
    def begin_run(self, source: str, context: Optional[str] = None) -> str:
        """搜索开始**之前**登记一次运行（写不进去就抛 —— 调用方据此拒绝开始搜索）。"""
        import uuid
        run_id = f"{source}:{uuid.uuid4().hex}"
        with self._Session() as s:
            s.add(TrialRunState(run_id=run_id, source=(source or "")[:32], status="running",
                                n_recorded=0, started_at=datetime.utcnow(), context=context))
            s.commit()
        return run_id

    def add_progress(self, run_id: str, n: int) -> int:
        """把这一批实际评估过的 n 个尝试记进全局累计与这次运行（一个事务）。返回新的累计总数。"""
        n = max(0, int(n))
        with self._Session() as s:
            if s.get(TrialCount, 1) is None:
                s.add(TrialCount(id=1, total=0))
                s.flush()
            s.execute(update(TrialCount).where(TrialCount.id == 1)
                      .values(total=TrialCount.total + n, updated_at=datetime.utcnow()))
            hit = s.execute(update(TrialRunState).where(TrialRunState.run_id == run_id)
                            .values(n_recorded=TrialRunState.n_recorded + n)).rowcount
            if hit != 1:
                s.rollback()
                raise KeyError(f"没有登记过的运行 {run_id}")
            s.commit()
            return int(s.scalar(select(TrialCount.total).where(TrialCount.id == 1)) or 0)

    def finish_run(self, run_id: str, status: str = "done", note: str = "") -> None:
        with self._Session() as s:
            s.execute(update(TrialRunState).where(TrialRunState.run_id == run_id)
                      .values(status=status, finished_at=datetime.utcnow(), note=(note or "")[:512]))
            s.commit()

    def incomplete_runs(self, now: Optional[datetime] = None) -> List[TrialRunState]:
        """
        研究史里有缺口的运行：
          · status=incomplete —— 运行中有记账写失败、结束时已知漏记（立即生效）；
          · 停在 running 超过 STALE_AFTER —— 被杀 / 崩溃 / 连结束标记都没写进去。
        """
        cutoff = (now or datetime.utcnow()) - STALE_AFTER
        with self._Session() as s:
            return list(s.scalars(select(TrialRunState).where(
                (TrialRunState.status == "incomplete")
                | ((TrialRunState.status == "running") & (TrialRunState.started_at < cutoff)))).all())

    def open_runs(self) -> List[TrialRunState]:
        """
        还没结案的运行：running（不论多久）+ incomplete。2026-10-22 用户核验 C5：正式资格只认**全部结案**的试验史 ——
        进行中的运行、被杀的进程、结束标记写不进去的运行都在这里；STALE_AFTER 只用来把 running 标成"疑似僵尸"。
        """
        with self._Session() as s:
            return list(s.scalars(select(TrialRunState).where(
                TrialRunState.status.in_(("running", "incomplete")))).all())

    def resolve_run(self, run_id: str, extra_trials: int, actor: str, reason: str) -> int:
        """
        人工结案一次中断的运行：把**估计**漏记的尝试数补进全局累计（宁可估多），写明是谁、为什么。
        漏记无法从数据里恢复，这里只能靠人判断 —— 所以必须留痕。
        """
        if not actor or not reason:
            raise ValueError("结案必须写明操作人与原因")
        extra = max(0, int(extra_trials))
        with self._Session() as s:
            row = s.get(TrialRunState, run_id)
            if row is None or row.status not in ("running", "incomplete"):
                raise KeyError(f"{run_id} 不是一个未结束 / 漏记的运行")
            if s.get(TrialCount, 1) is None:
                s.add(TrialCount(id=1, total=0))
                s.flush()
            s.execute(update(TrialCount).where(TrialCount.id == 1)
                      .values(total=TrialCount.total + extra, updated_at=datetime.utcnow()))
            row.status, row.finished_at = "resolved", datetime.utcnow()
            row.n_recorded = int(row.n_recorded) + extra
            row.note = f"{actor}: {reason}（补记 {extra}）"[:512]
            s.commit()
            return int(s.scalar(select(TrialCount.total).where(TrialCount.id == 1)) or 0)

    def run_states(self, source: Optional[str] = None) -> List[TrialRunState]:
        with self._Session() as s:
            q = select(TrialRunState).order_by(TrialRunState.started_at)
            if source:
                q = q.where(TrialRunState.source == source)
            return list(s.scalars(q).all())

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


class TrialRecorder:
    """
    搜索底层用的记账器（复审 2026-10-19 N07）：

        rec = TrialRecorder("gp", panels=[is_data, oos_data])   # 登记不进去 → 抛 TrialLedgerUnavailable
        rec.add(len(results))            # 每一代 / 每个 trial 记一次
        rec.finish("done")               # 或 "aborted"

    逐批写失败不打断搜索，但这次运行会停在 running → 过期后门判"试验史不完整"。

    在 `with ctx.active():`（研究上下文，复审 2026-10-19 N09-c）里：登记时把上下文写进运行记录，
    并检查 `panels` —— 含冻结 Test 段的日期 → FrozenBoundaryError，**不登记、不开始**。
    """

    def __init__(self, source: str, panels=None) -> None:
        self.source, self.ok, self.n, self.lost = source, True, 0, 0
        from app.core.data_engine.research_context import current
        ctx = current()
        if ctx is not None:
            ctx.check_panels(panels or [], what=f"{source} 搜索")
        else:
            logger.info("[S.3] %s 搜索不在研究上下文里运行（运行记录不带 dataset / experiment / boundary）", source)
        try:
            self.ledger = TrialLedger()
            self.run_id = self.ledger.begin_run(source, ctx.to_json() if ctx is not None else None)
        except Exception as exc:
            raise TrialLedgerUnavailable(
                f"试验台账不可写（{exc}）—— 拒绝开始 {source} 搜索：没登记的尝试会从研究史里消失，"
                f"之后的 DSR / 策略门都会按残缺的试验数给结论") from exc

    def add(self, n: int) -> None:
        n = int(n)
        if n <= 0:
            return
        self.n += n
        try:
            self.ledger.add_progress(self.run_id, n)
        except Exception as exc:
            self.ok = False
            self.lost += n
            logger.error("[S.3] %s 的 %d 次尝试没进全局计数（运行 %s 将保持未结束，门会判试验史不完整）: %s",
                         self.source, n, self.run_id, exc)

    def finish(self, status: str = "done") -> None:
        # 有漏记 → 标 incomplete（门**立即**看得见，不必等过期；复审 2026-10-19 N07 的场景：
        # 写入暂时失败、恢复后马上评估）。连这个标记都写不进去 → 停在 running，过期后被看见。
        final = status if self.ok else "incomplete"
        note = "" if self.ok else f"运行中有 {self.lost} 次尝试的记账写失败，没进全局计数"
        try:
            self.ledger.finish_run(self.run_id, final, note=note)
        except Exception as exc:
            logger.error("[S.3] 运行 %s 结束标记写不进去（会被当作中断）: %s", self.run_id, exc)


class TrialLedgerUnavailable(RuntimeError):
    """试验台账写不进去：搜索不得开始。"""


def trial_history_problem(now: Optional[datetime] = None) -> str:
    """
    门用：试验史是否完整。不完整 → 说明原因（门据此不给正式结论）；完整 → 空串。
    2026-10-22 用户核验 C5：此前只把超过 6 小时的 running 算作不完整 —— 刚开始的、被杀的、结束标记写失败的
    运行在 6 小时内都被当成完整史放行。现在任何未结案的运行都让正式结论等待（结束、或 resolve_run 结案）。
    """
    try:
        led = TrialLedger()
        open_ = led.open_runs()
        stale = {r.run_id for r in led.incomplete_runs(now)}
    except Exception as exc:
        logger.error("[S.3] 读不到试验运行状态 → 按试验史不完整处理: %s", exc)
        return f"读不到试验运行状态（{exc}）"
    if not open_:
        return ""
    def _label(r):
        if r.status == "incomplete":
            return f"{r.run_id}（有漏记）"
        return f"{r.run_id}（{'疑似僵尸' if r.run_id in stale else '进行中'}）"
    return (f"{len(open_)} 次搜索运行没有结案（{', '.join(_label(r) for r in open_[:3])}"
            f"{' …' if len(open_) > 3 else ''}）—— 试验总数还会变，DSR 的多重检验校正不完整；"
            f"等它结束，或用 TrialLedger.resolve_run 补记并写明理由后再评估")


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
    #: 2026-10-22 D12：谁、在什么股票池上看的（列出用；**不进计次** —— 次数只按数据集 + 区间重叠算，
    #: 改实验名不会刷新额度）
    experiment:      str = Column(String(128), nullable=True)
    universe_digest: str = Column(String(32), nullable=True)
    universe_n:      int = Column(Integer, nullable=True)


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
        _add_missing_columns(self._engine, {"holdout_usages": {
            "experiment": "VARCHAR(128)", "universe_digest": "VARCHAR(32)", "universe_n": "INTEGER"}})
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    def usages(self, dataset_key: str) -> List[dict]:
        """该数据集的全部使用记录（时间顺序）：区间、用途、实验名、股票池。"""
        with self._Session() as s:
            rows = list(s.scalars(select(HoldoutUse).where(HoldoutUse.dataset_key == dataset_key)
                                  .order_by(HoldoutUse.id)).all())
        return [{"test_key": r.test_key, "purpose": r.purpose, "used_at": r.used_at,
                 "experiment": r.experiment, "universe_digest": r.universe_digest,
                 "universe_n": r.universe_n} for r in rows]

    def count(self, dataset_key: str, test_key: str) -> int:
        """该数据集里与 `test_key` 这段区间**有重叠**的使用次数（区间解析不了时按键精确匹配）。"""
        with self._Session() as s:
            keys = list(s.scalars(select(HoldoutUse.test_key)
                                  .where(HoldoutUse.dataset_key == dataset_key)).all())
        return _count_overlapping(keys, test_key)

    def earliest_exposed_start(self, dataset_key: str) -> Optional[pd.Timestamp]:
        """
        该数据集**看过的**最早一个 Test 窗口的起点（没有记录 → None）。
        复审 2026-10-19 对 D12 的建议：用它作保守的冻结边界（BoundaryStore.pin），而不是让第一次 API 调用
        偶然决定。只读，不替你做决定。
        """
        with self._Session() as s:
            keys = list(s.scalars(select(HoldoutUse.test_key)
                                  .where(HoldoutUse.dataset_key == dataset_key)).all())
        starts = [w[0] for w in (_parse_window(k) for k in keys) if w is not None]
        return min(starts) if starts else None

    def record_use(
        self, dataset_key: str, test_key: str, purpose: str = "", *,
        experiment: Optional[str] = None, universe_digest: Optional[str] = None,
        universe_n: Optional[int] = None,
    ) -> HoldoutUsage:
        """
        记一次 Test 段使用并返回记账结果（含本次的累计次数）。实验名 / 股票池只是记下来，
        **不参与计次**（D12："改实验名不能刷新重叠数据的曝光额度"）。
        """
        with self._Session() as s:
            s.add(HoldoutUse(dataset_key=dataset_key, test_key=test_key,
                             purpose=purpose or "", experiment=experiment,
                             universe_digest=universe_digest, universe_n=universe_n))
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
    #: D12：人登记的边界才有这两项；空 = 旧代码首次调用偶然写下的（"未确认"）
    actor:       str      = Column(String(64), nullable=True)
    reason:      str      = Column(String(512), nullable=True)


class BoundaryAlreadyRegistered(RuntimeError):
    """这个数据集已经有登记过的边界（或与之不同的冻结边界）。改它是另一个研究决定，不在这里做。"""


#: 边界状态（`ThreeWaySplit.boundary` / `ResearchContext.boundary`）
BOUNDARY_REGISTERED = "registered"      # 人登记的：正式结论可用
BOUNDARY_UNCONFIRMED = "unconfirmed"    # 旧代码偶然写下的：照样冻结，正式结论等人确认
BOUNDARY_PROVISIONAL = "provisional"    # 还没有边界：本次临时切分，不写库，正式结论拒绝


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
        _add_missing_columns(self._engine, {"frozen_boundaries": {"actor": "VARCHAR(64)",
                                                                  "reason": "VARCHAR(512)"}})
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    def get(self, dataset_key: str) -> Optional[pd.Timestamp]:
        with self._Session() as s:
            row = s.get(FrozenBoundary, dataset_key)
            return pd.Timestamp(row.test_start) if row is not None else None

    def status(self, dataset_key: str) -> Optional[Tuple[pd.Timestamp, str]]:
        """(Test 起点, registered / unconfirmed)；没有边界 → None。"""
        with self._Session() as s:
            row = s.get(FrozenBoundary, dataset_key)
            if row is None:
                return None
            ok = bool((row.actor or "").strip() and (row.reason or "").strip())
            return pd.Timestamp(row.test_start), (BOUNDARY_REGISTERED if ok else BOUNDARY_UNCONFIRMED)

    def register(self, dataset_key: str, test_start: pd.Timestamp, actor: str, reason: str) -> pd.Timestamp:
        """
        人登记冻结边界（D12：日期建议给出后由用户确认）。只写一次：
          · 没有边界 → 写入（frozen_by = registered，带操作人与理由）
          · 旧代码偶然写下、日期**相同** → 补上操作人与理由（确认）
          · 已登记过，或已有的冻结日期不同 → BoundaryAlreadyRegistered（冻结的段不能被改写成别的日期）
        """
        actor, reason = (actor or "").strip(), (reason or "").strip()
        if not actor or not reason:
            raise ValueError("登记冻结边界必须写明操作人和理由")
        day = str(pd.Timestamp(test_start).date())
        with self._Session() as s:
            row = s.get(FrozenBoundary, dataset_key)
            if row is None:
                s.add(FrozenBoundary(dataset_key=dataset_key, test_start=day, frozen_by="registered",
                                     actor=actor[:64], reason=reason[:512]))
            elif (row.actor or "").strip():
                raise BoundaryAlreadyRegistered(
                    f"数据集 {dataset_key} 的边界已登记为 {row.test_start}（{row.actor}：{row.reason}）")
            elif row.test_start != day:
                raise BoundaryAlreadyRegistered(
                    f"数据集 {dataset_key} 已冻结在 {row.test_start}（旧代码首次调用写下的）；"
                    f"冻结的段不能改到 {day} —— 要换边界是另一个研究决定")
            else:
                row.actor, row.reason = actor[:64], reason[:512]
            try:
                s.commit()
            except IntegrityError as exc:
                s.rollback()
                raise BoundaryAlreadyRegistered(f"数据集 {dataset_key} 同时被别的进程登记了：{exc}") from exc
        logger.warning("[D12] 数据集 %s 的冻结 Test 起点登记为 %s（%s：%s）", dataset_key, day, actor, reason)
        return pd.Timestamp(day)

    def pin(self, dataset_key: str, test_start: pd.Timestamp, frozen_by: str = "") -> pd.Timestamp:
        """
        旧接口（F04 的"首次调用写死"）。D12 之后研究入口**不再调用**它；留着只为测试能构造"旧代码写下的
        未确认边界"。写下的行没有操作人 → 状态是 unconfirmed，正式结论不认。
        """
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


def _add_missing_columns(engine, tables: dict) -> None:
    """后加的可空列：只加、不改、不删（create_all 不会给已存在的表补列）。"""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    for table, cols in tables.items():
        have = {c["name"] for c in insp.get_columns(table)}
        for name, typ in cols.items():
            if name not in have:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {typ}"))
