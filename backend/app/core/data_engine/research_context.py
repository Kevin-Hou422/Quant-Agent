"""
research_context.py — 研究流程的显式上下文（复审 2026-10-19 N09-c）

复审原话："CLI、chat、工作流、发现、回测、验证和提案都取得同一显式 ResearchContext，
携带 dataset/universe/experiment/boundary。"

此前每个入口各自切研究窗口，兜底规则各不相同：有的任何异常都退回全量、有的只在 ValueError 时退回；
`s_three_way_enabled=False` 能让 API 与验证门看到冻结段；chat 的 GP 指定数据集时干脆按全量 70/30 切，
OOS 压在冻结 Test 段上。现在研究窗口**只有** `open_research` 一种切法：

    ctx = open_research(dataset, dataset_key, purpose="cli.gp")
    ctx.selection          # 研究流程唯一被允许看的面板
    ctx.split              # ThreeWaySplit（冻结 Test 段只能经 account_holdout_use 记账后动用）
    with ctx.active():     # 这期间的 GP / Optuna 运行把上下文写进试验运行记录，并拒绝越界数据
        evolver.run(...)

    dataset_key      真实数据集名；"" = 合成 / 未登记（没有持久化的边界）
    universe         选择段里有过收盘价的标的集合（哈希 + 个数）—— 同名数据集换了票就是另一个上下文
    experiment       实验名（缺省 settings.research_experiment）；purpose = 哪个入口
    test_start       冻结 Test 段起点（含）；None = 本次没有 Test 段
    boundary         边界状态（2026-10-22 D12）：registered（人登记的）/ unconfirmed（旧代码偶然写下的）/
                     provisional（还没有边界，临时切分）/ none（真实数据集但本次没有 Test 段）/ synthetic

边界按 dataset_key 持久化，只能由人登记（D12："正式入口不要继续依赖首次调用偶然冻结"）。
正式结论（验证门、策略提案）只认 registered（合成数据没有可登记的边界，按 synthetic 放行）——
`formal_boundary_problem()`。曝光次数按数据集 + 区间重叠计，实验名 / 股票池只记录、不进计次。
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import pandas as pd

from app.core.data_engine.data_partitioner import (
    _UNPINNED_KEYS, FrozenBoundaryError, ThreeWaySplit, split_from_settings,
)

logger = logging.getLogger(__name__)

_CURRENT: ContextVar[Optional["ResearchContext"]] = ContextVar("research_context", default=None)


@dataclass(frozen=True)
class ResearchContext:
    dataset_key:     str
    experiment:      str
    purpose:         str
    universe_digest: str
    universe_n:      int
    test_start:      Optional[str]       # YYYY-MM-DD，含当天；None = 没有 Test 段
    frozen_by:       str
    pinned:          bool                # 边界是持久化的（真实数据集）
    selection_start: str
    selection_end:   str
    note:            str = ""            # 为什么没有 Test 段 / 开关被忽略等
    boundary:        str = ""            # registered / unconfirmed / provisional / none / synthetic（D12）
    selection: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False, compare=False)
    split:     Optional[ThreeWaySplit] = field(default=None, repr=False, compare=False)

    def describe(self) -> dict:
        return {
            "dataset_key": self.dataset_key, "experiment": self.experiment, "purpose": self.purpose,
            "universe_digest": self.universe_digest, "universe_n": self.universe_n,
            "test_start": self.test_start, "frozen_by": self.frozen_by, "pinned": self.pinned,
            "selection_start": self.selection_start, "selection_end": self.selection_end,
            "note": self.note, "boundary": self.boundary,
        }

    def formal_boundary_problem(self) -> str:
        """
        正式结论能不能用这个研究上下文（D12）。能 → ""；不能 → 原因。
        只认人登记过的边界；合成数据（没有可登记的边界）放行。
        """
        if self.boundary in ("registered", "synthetic"):
            return ""
        what = {"provisional": "还没有登记，本次是临时切分",
                "unconfirmed": "是旧代码首次调用偶然写下的，没有人确认",
                "none": "本次没有冻结 Test 段（数据太短或三段切割被关掉）"}.get(self.boundary, self.boundary)
        return (f"研究边界未登记：数据集 {self.dataset_key} 的冻结边界{what} —— "
                f"正式结论要等边界登记（D12：日期建议后由用户确认）")

    def to_json(self) -> str:
        return json.dumps(self.describe(), ensure_ascii=False, sort_keys=True)

    @contextmanager
    def active(self):
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    def check_panels(self, panels: Iterable[Optional[Dict[str, pd.DataFrame]]], what: str = "") -> None:
        """面板里有冻结段（≥ test_start）的日期 → FrozenBoundaryError。没有 Test 段时不检查。"""
        if not self.test_start:
            return
        bound = pd.Timestamp(self.test_start)
        for panel in panels:
            idx = _index_of(panel)
            if idx is not None and len(idx) and pd.Timestamp(idx.max()) >= bound:
                n = int((idx >= bound).sum())
                raise FrozenBoundaryError(
                    f"{what or '搜索'}拿到了 {self.dataset_key} 冻结 Test 段（{self.test_start} 起）的 {n} 个日期 —— "
                    f"研究流程只许看选择段（{self.purpose}）")


def current() -> Optional[ResearchContext]:
    """当前活动的研究上下文（`with ctx.active():` 里）；没有 → None。"""
    return _CURRENT.get()


def open_research(dataset: Dict[str, pd.DataFrame], dataset_key: Optional[str], purpose: str,
                  experiment: Optional[str] = None, three_way: Optional[bool] = None,
                  **split_overrides) -> ResearchContext:
    """
    取得研究上下文。规则（所有入口同一套）：

      · 真实数据集：按持久化的冻结边界切；有边界却切不出选择段 → FrozenBoundaryError（调用方拒绝，绝不退回全量）
      · 数据太短、且还没有冻结边界 → 全量研究、本次没有 Test 段（告警，并写进 note）
      · 三段切割被配置 / 参数关掉：数据集已有冻结边界 → **忽略开关**（已冻结的段不能被开关解冻）；
        没有边界 → 全量并告警
      · 其它异常（读不到边界库等）照常抛出 —— 不退回全量
    """
    key = (dataset_key or "").strip()
    pinned_key = key not in _UNPINNED_KEYS
    exp = experiment if experiment is not None else _default_experiment()
    if three_way is None:
        three_way = _three_way_enabled()
    note = ""
    if not three_way:
        if pinned_key and _has_pinned_boundary(key):
            note = "three_way_disabled_ignored: 数据集已有冻结边界，开关不能解冻"
            logger.error("[research] %s 关掉了三段切割，但 %s 已有冻结边界 → 忽略开关", purpose, key)
        else:
            logger.warning("[research] %s 关掉了三段切割 → 看全部数据，结果按样本内解读", purpose)
            return _build(dataset, key, exp, purpose, None, pinned=False, note="three_way_disabled")
    try:
        split = split_from_settings(dataset, dataset_key=key or None, **split_overrides)
    except FrozenBoundaryError:
        raise
    except ValueError as exc:
        logger.warning("[research] %s：%s 数据不足以三段切割，按全量研究（本次没有冻结 Test 段）: %s",
                       purpose, key or "synthetic", exc)
        return _build(dataset, key, exp, purpose, None, pinned=False,
                      note=(note + "; " if note else "") + f"no_test_segment: {exc}")
    return _build(split.selection, key, exp, purpose, split, pinned=pinned_key, note=note)


# ---------------------------------------------------------------------------

def _build(selection, key, exp, purpose, split, pinned, note) -> ResearchContext:
    idx = _index_of(selection)
    ts = getattr(split, "test_start", None) if split is not None else None
    if key in _UNPINNED_KEYS:
        boundary = "synthetic"
    elif split is None or ts is None:
        boundary = "none"
    else:
        boundary = str(getattr(split, "boundary", "") or "provisional")
    return ResearchContext(
        dataset_key=key, experiment=str(exp), purpose=str(purpose),
        universe_digest=_universe_digest(selection), universe_n=len(_universe(selection)),
        test_start=str(pd.Timestamp(ts).date()) if ts is not None else None,
        frozen_by=str(getattr(split, "frozen_by", "") or "") if split is not None else "",
        pinned=bool(pinned and ts is not None and boundary in ("registered", "unconfirmed")),
        selection_start=str(pd.Timestamp(idx.min()).date()) if idx is not None and len(idx) else "",
        selection_end=str(pd.Timestamp(idx.max()).date()) if idx is not None and len(idx) else "",
        note=note, boundary=boundary, selection=selection, split=split,
    )


def _index_of(panel) -> Optional[pd.Index]:
    if not panel:
        return None
    close = panel.get("close") if isinstance(panel, dict) else None
    if isinstance(close, pd.DataFrame):
        return close.index
    for v in panel.values():
        if isinstance(v, (pd.DataFrame, pd.Series)):
            return v.index
    return None


def _universe(panel) -> list:
    close = panel.get("close") if isinstance(panel, dict) else None
    if not isinstance(close, pd.DataFrame):
        return []
    return sorted(str(c) for c in close.columns[close.notna().any(axis=0)])


def _universe_digest(panel) -> str:
    return hashlib.sha256("\x1f".join(_universe(panel)).encode("utf-8")).hexdigest()[:16]


def _default_experiment() -> str:
    try:
        from app.config import settings
        return str(getattr(settings, "research_experiment", "") or "default")
    except Exception as exc:
        logger.warning("[research] 读不到 research_experiment，实验名记为 default: %s", exc)
        return "default"


def _three_way_enabled() -> bool:
    try:
        from app.config import settings
        return bool(settings.s_three_way_enabled)
    except Exception as exc:
        logger.error("[research] 读不到 s_three_way_enabled，按更严的一侧（切三段）: %s", exc)
        return True


def _has_pinned_boundary(key: str) -> bool:
    try:
        from app.db.trial_ledger import BoundaryStore
        return BoundaryStore().get(key) is not None
    except Exception as exc:
        raise FrozenBoundaryError(f"读不到数据集 {key} 的冻结边界（{exc}）—— 不能确认开关是否可以生效，拒绝") from exc
