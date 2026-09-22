"""
data_partitioner.py — IS/OOS 数据分区引擎（含 Walk-Forward）

核心目标：防止过拟合（Anti-Overfitting Core）。

设计原则：
  1. 物理分区：IS 和 OOS 数据存于独立内存，无共享引用
  2. 严格隔离：优化算法（GP/Agent）只能访问 .train() 分区
  3. 不可篡改：PartitionedDataset 使用 Python __slots__ + 私有属性防止外部写入
  4. 无泄漏：切分点使用严格不等式（IS 末日 < split_date），确保无重叠
  5. Embargo Period：IS/OOS 之间插入空白窗口防止标签泄漏（Task 2.2）

主要类：
  DataPartitioner        — 单次 IS/OOS 固定切分（加 embargo）
  WalkForwardPartitioner — 滚动扩展窗口 Walk-Forward 分区（Task 2.1）
  PartitionedDataset     — 不可变双段数据容器
  WalkForwardFold        — 单轮 WF 分区描述
  ThreeWayPartitioner    — IS / Validate / Test 三段切分（Phase S.1+S.2）
  PurgedKFold            — IS 内部 purged + embargoed K 折（Phase S.1）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PartitionedDataset — 不可变数据容器（防止外部修改）
# ---------------------------------------------------------------------------

class PartitionedDataset:
    """
    持有物理分离的 IS / OOS 数据集，对外只暴露只读访问方法。

    Attributes（只读）
    ------------------
    split_date   : IS/OOS 名义切分日期（OOS 实际首日可能晚于此，因为 embargo）
    oos_ratio    : 名义 OOS 占比（embargo 之前）
    embargo_days : 实际使用的 embargo 天数
    is_days      : IS 实际交易日数
    oos_days     : OOS 实际交易日数
    """

    __slots__ = (
        "_train", "_test",
        "split_date", "oos_ratio", "embargo_days",
        "is_days", "oos_days",
    )

    def __init__(
        self,
        train_data:   Dict[str, pd.DataFrame],
        test_data:    Dict[str, pd.DataFrame],
        split_date:   pd.Timestamp,
        oos_ratio:    float,
        is_days:      int,
        oos_days:     int,
        embargo_days: int = 0,
    ) -> None:
        object.__setattr__(self, "_train",       {k: v.copy() for k, v in train_data.items()})
        object.__setattr__(self, "_test",        {k: v.copy() for k, v in test_data.items()})
        object.__setattr__(self, "split_date",   split_date)
        object.__setattr__(self, "oos_ratio",    oos_ratio)
        object.__setattr__(self, "embargo_days", embargo_days)
        object.__setattr__(self, "is_days",      is_days)
        object.__setattr__(self, "oos_days",     oos_days)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("PartitionedDataset 是只读对象，不允许修改分区数据。")

    def train(self) -> Dict[str, pd.DataFrame]:
        """返回 In-Sample 数据。优化算法应仅使用此分区。"""
        return {k: v.copy() for k, v in self._train.items()}

    def test(self) -> Dict[str, pd.DataFrame]:
        """返回 Out-of-Sample 数据。仅在最终验证阶段使用。"""
        return {k: v.copy() for k, v in self._test.items()}

    def summary(self) -> str:
        any_train = next(iter(self._train.values()))
        any_test  = next(iter(self._test.values()))

        is_start  = any_train.index[0]  if len(any_train) else "N/A"
        is_end    = any_train.index[-1] if len(any_train) else "N/A"
        oos_start = any_test.index[0]   if len(any_test)  else "N/A"
        oos_end   = any_test.index[-1]  if len(any_test)  else "N/A"

        embargo_note = (
            f"  Embargo 天数  : {self.embargo_days} 交易日\n"
            if self.embargo_days > 0 else ""
        )
        lines = [
            "=" * 52,
            "  DataPartitioner — IS/OOS 分区摘要",
            "=" * 52,
            f"  切分日期     : {self.split_date.date()}",
            f"  IS 期间      : {is_start} → {is_end}  ({self.is_days} 交易日)",
            f"  OOS 期间     : {oos_start} → {oos_end}  ({self.oos_days} 交易日)",
            f"  OOS 占比     : {self.oos_ratio * 100:.1f}%",
            f"{embargo_note}  字段数量     : {len(self._train)}",
            "=" * 52,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DataPartitioner — 单次固定切分（+ Embargo Period，Task 2.2）
# ---------------------------------------------------------------------------

class DataPartitioner:
    """
    基于时间轴的严格 In-Sample / Out-of-Sample 物理分区器。

    Parameters
    ----------
    start        : 全局起始日期（"YYYY-MM-DD"）
    end          : 全局终止日期（"YYYY-MM-DD"）
    oos_ratio    : OOS 占**可用**交易日（总交易日扣除 embargo 后）的比例，默认 0.30。
                   注意语义：embargo 先从总跨度里扣掉，再按 ratio 切分剩余部分，
                   因此 oos_ratio=0.3 得到的确实是可用样本的 30%。
                   （旧实现把 IS 按**总天数**取 70%，embargo 再全部从 OOS 里扣 ——
                   于是 60 天 + embargo=20 会得到 OOS **0 行**、80 天得到 4 行，
                   请求的比例从未被兑现，且下游会拿 4 行样本算年化 Sharpe。）
    embargo_days : IS 末日后跳过的交易日数（默认 20）。
                   这些日期不参与 IS 训练，也不参与 OOS 验证，
                   防止时间序列自相关导致的标签泄漏。
                   设为 0 表示不使用 embargo（向后兼容）。
    """

    def __init__(
        self,
        start:        str,
        end:          str,
        oos_ratio:    float = 0.30,
        embargo_days: int   = 20,
    ) -> None:
        if not (0.0 <= oos_ratio < 1.0):
            raise ValueError(f"oos_ratio 必须在 [0, 1) 范围内，当前={oos_ratio}")
        if embargo_days < 0:
            raise ValueError(f"embargo_days 不能为负数，当前={embargo_days}")

        self.start        = pd.Timestamp(start)
        self.end          = pd.Timestamp(end)
        self.oos_ratio    = oos_ratio
        self.embargo_days = embargo_days

        all_bdays  = pd.bdate_range(self.start, self.end)
        total_days = len(all_bdays)

        if total_days < 10:
            raise ValueError(
                f"日期范围 {start}→{end} 交易日不足（{total_days}天）。"
            )

        if oos_ratio > 0:
            # embargo **先从总跨度扣除**，再按 ratio 切分剩余可用样本。
            usable = total_days - embargo_days
            if usable < 2:
                raise ValueError(
                    f"日期范围 {start}→{end} 共 {total_days} 个交易日，"
                    f"扣除 embargo={embargo_days} 后仅剩 {usable} 天，无法切分 IS/OOS。"
                    f"请延长区间或减小 embargo_days。"
                )
            is_count  = int(round(usable * (1.0 - oos_ratio)))
            is_count  = max(1, min(is_count, usable - 1))
            oos_count = usable - is_count
        else:
            is_count  = total_days
            oos_count = 0

        self._all_bdays  = all_bdays
        self._is_count   = is_count

        # IS 末日（包含）= is_count-1 号工作日
        self._is_end    = all_bdays[is_count - 1]
        # 名义切分日期（下一个工作日）
        self._split_date = all_bdays[is_count] if oos_ratio > 0 and is_count < total_days else None
        # OOS 实际首日 = 切分日期 + embargo_days 个工作日
        if self._split_date is not None and embargo_days > 0:
            oos_actual_idx = is_count + embargo_days
            if oos_actual_idx < total_days:
                self._oos_start = all_bdays[oos_actual_idx]
            else:
                self._oos_start = None   # embargo 超出范围
        else:
            self._oos_start = self._split_date

        self._oos_count = oos_count
        # 兑现请求的比例：切完必须真的有 OOS 样本，否则明确报错而非静默返回 0 行。
        # （旧实现在这里静默产出空 OOS，下游 _validate_dataset 才以 500 崩掉。）
        if oos_ratio > 0 and (self._oos_start is None or oos_count < 1):
            raise ValueError(
                f"切分后 OOS 为空：总={total_days} IS={is_count} embargo={embargo_days}。"
                f"请延长日期区间、减小 embargo_days 或提高 oos_ratio。"
            )

        logger.info(
            "DataPartitioner 初始化 | 总=%d | IS=%d | embargo=%d | OOS=%d | split=%s",
            total_days, is_count, embargo_days, self._oos_count,
            self._split_date.date() if self._split_date else "N/A",
        )

    def partition(self, dataset: Dict[str, pd.DataFrame]) -> PartitionedDataset:
        """
        物理切分 dataset 为 IS 和 OOS。IS 末和 OOS 首之间的 embargo 窗口被丢弃。

        Parameters
        ----------
        dataset : dict[field → (T×N) pd.DataFrame]，index=DatetimeIndex

        Returns
        -------
        PartitionedDataset（只读）
        """
        if not dataset:
            raise ValueError("dataset 不能为空")

        train_data: Dict[str, pd.DataFrame] = {}
        test_data:  Dict[str, pd.DataFrame] = {}

        for field_name, df in dataset.items():
            df_idx = pd.DatetimeIndex(df.index)

            if self._split_date is None or self.oos_ratio == 0.0:
                train_data[field_name] = df.copy()
                test_data[field_name]  = df.iloc[0:0].copy()
            else:
                # IS: 严格小于名义切分日期（不含 embargo 区域）
                is_mask = df_idx < self._split_date
                train_data[field_name] = df.loc[is_mask].copy()

                # OOS: 从 oos_start 开始（跳过 embargo）
                if self._oos_start is not None:
                    oos_mask = df_idx >= self._oos_start
                    test_data[field_name] = df.loc[oos_mask].copy()
                else:
                    test_data[field_name] = df.iloc[0:0].copy()

        first_field   = next(iter(train_data))
        actual_is     = len(train_data[first_field])
        actual_oos    = len(test_data[first_field])

        parts = PartitionedDataset(
            train_data   = train_data,
            test_data    = test_data,
            split_date   = self._split_date or self.end,
            oos_ratio    = self.oos_ratio,
            is_days      = actual_is,
            oos_days     = actual_oos,
            embargo_days = self.embargo_days,
        )
        logger.info(
            "分区完成 | IS=%d天 | embargo=%d天 | OOS=%d天 | 字段=%d",
            actual_is, self.embargo_days, actual_oos, len(dataset),
        )
        return parts

    @property
    def split_date(self) -> Optional[pd.Timestamp]:
        return self._split_date

    def summary(self) -> str:
        lines = [
            "=" * 52,
            "  DataPartitioner 配置",
            "=" * 52,
            f"  全局范围     : {self.start.date()} → {self.end.date()}",
            f"  总交易日     : {len(self._all_bdays)}",
            f"  IS 天数      : {self._is_count}",
            f"  Embargo 天数 : {self.embargo_days}",
            f"  OOS 天数     : {self._oos_count}",
            f"  OOS 占比     : {self.oos_ratio * 100:.1f}%",
            f"  切分日期     : {self._split_date.date() if self._split_date else 'N/A'}",
            f"  OOS 首日     : {self._oos_start.date() if self._oos_start else 'N/A'}",
            "=" * 52,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# WalkForwardFold — 单轮 Walk-Forward 元信息
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardFold:
    """描述 Walk-Forward 中单轮的时间范围。"""
    fold_idx:     int
    is_start:     pd.Timestamp
    is_end:       pd.Timestamp
    oos_start:    pd.Timestamp
    oos_end:      pd.Timestamp
    is_days:      int
    oos_days:     int
    embargo_days: int

    def __str__(self) -> str:
        return (
            f"Fold {self.fold_idx+1}: "
            f"IS=[{self.is_start.date()}→{self.is_end.date()}]({self.is_days}d) "
            f"embargo={self.embargo_days}d "
            f"OOS=[{self.oos_start.date()}→{self.oos_end.date()}]({self.oos_days}d)"
        )


# ---------------------------------------------------------------------------
# WalkForwardPartitioner — 滚动扩展窗口分区器（Task 2.1）
# ---------------------------------------------------------------------------

class WalkForwardPartitioner:
    """
    扩展窗口（expanding-window）Walk-Forward 分区器。

    每轮 IS 从全局起始日固定，OOS 窗口向后滚动。
    切分点之间插入 embargo_days 个工作日，防止标签泄漏。

    示例（n_splits=5，总 1200 天，oos_per_fold ≈ 120 天，embargo=20）：
      Fold 1: IS=[T0 → T720]  embargo=[T721→T740]  OOS=[T741 → T860]
      Fold 2: IS=[T0 → T860]  embargo=[T861→T880]  OOS=[T881 → T1000]
      Fold 3: IS=[T0 → T1000] embargo=[T1001→T1020] OOS=[T1021 → T1140]
      ...

    Parameters
    ----------
    n_splits      : 分折数量（推荐 5-10）
    min_train_days: IS 最少天数（确保有足够样本）
    embargo_days  : IS 末日后跳过的工作日数（防标签泄漏）
    """

    def __init__(
        self,
        n_splits:       int = 5,
        min_train_days: int = 120,
        embargo_days:   int = 20,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits 至少为 2，当前={n_splits}")
        if min_train_days < 20:
            raise ValueError(f"min_train_days 至少为 20，当前={min_train_days}")
        if embargo_days < 0:
            raise ValueError(f"embargo_days 不能为负数，当前={embargo_days}")

        self.n_splits       = n_splits
        self.min_train_days = min_train_days
        self.embargo_days   = embargo_days

    def get_folds(
        self,
        dataset: Dict[str, pd.DataFrame],
    ) -> List[WalkForwardFold]:
        """
        计算 Walk-Forward 分折元信息（不复制数据）。

        Returns
        -------
        list[WalkForwardFold]，长度 = n_splits（当日期范围不足时可能更少）
        """
        dates = _extract_dates(dataset)
        return self._compute_folds(dates)

    def split(
        self,
        dataset: Dict[str, pd.DataFrame],
    ) -> List[Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]]:
        """
        将 dataset 切分为 n_splits 轮 IS/OOS 数据对。

        Returns
        -------
        [(is_data_0, oos_data_0), (is_data_1, oos_data_1), ...]
        每个元素均为独立副本（物理隔离）。
        """
        dates = _extract_dates(dataset)
        folds = self._compute_folds(dates)

        if not folds:
            raise ValueError(
                f"日期范围不足，无法生成 Walk-Forward 分折。"
                f" 总天数={len(dates)}，min_train={self.min_train_days}，"
                f" n_splits={self.n_splits}，embargo={self.embargo_days}"
            )

        result: List[Tuple[Dict, Dict]] = []
        for fold in folds:
            is_data  = _slice_dataset(dataset, end=fold.is_end,                   inclusive=True)
            oos_data = _slice_dataset(dataset, start=fold.oos_start, end=fold.oos_end, inclusive=True)
            result.append((is_data, oos_data))
            logger.debug("WF %s", fold)

        logger.info(
            "WalkForward 分区完成 | n_splits=%d | embargo=%d天",
            len(result), self.embargo_days,
        )
        return result

    def _compute_folds(self, dates: pd.DatetimeIndex) -> List[WalkForwardFold]:
        """
        核心逻辑：基于 dates 计算每折的起止索引。

        策略（扩展窗口）：
          - IS 从全局起始固定，末日随折数向后扩展
          - 每折 OOS 窗口大小相等，非重叠，折间有 embargo 间隔
          - 每折数据使用量：每折推进 oos_per_fold 个交易日；embargo 天数不计入 IS 也不计入 OOS

        oos_per_fold 计算：
          末折 OOS 末日索引 = min_train_days + n_splits * oos_per_fold + embargo_days - 1 ≤ n - 1
          ⟹ oos_per_fold ≤ (n - min_train_days - embargo_days) / n_splits
          ⟹ oos_per_fold = (n - min_train_days - embargo_days) // n_splits
        """
        n = len(dates)
        if n < self.min_train_days + self.embargo_days + self.n_splits:
            return []

        # 减去 min_train_days 和一个 embargo 间隔后，剩余天数均分给每折 OOS
        # (embargo 只需扣减一次：每折 OOS 之间没有额外间隔，间隔只在 IS 末 → OOS 首之间)
        available = n - self.min_train_days - self.embargo_days
        oos_per_fold = max(1, available // self.n_splits)
        if oos_per_fold < 5:
            return []

        folds: List[WalkForwardFold] = []
        for i in range(self.n_splits):
            # IS 末日索引（扩展窗口：每折 IS 都包含前面所有历史数据）
            is_end_idx = self.min_train_days + i * oos_per_fold - 1
            if is_end_idx >= n:
                break

            # embargo: 跳过 is_end_idx + 1 到 is_end_idx + embargo_days
            oos_start_idx = is_end_idx + 1 + self.embargo_days
            oos_end_idx   = min(oos_start_idx + oos_per_fold - 1, n - 1)

            if oos_start_idx >= n or oos_start_idx > oos_end_idx:
                break

            fold = WalkForwardFold(
                fold_idx     = i,
                is_start     = dates[0],
                is_end       = dates[is_end_idx],
                oos_start    = dates[oos_start_idx],
                oos_end      = dates[oos_end_idx],
                is_days      = is_end_idx + 1,
                oos_days     = oos_end_idx - oos_start_idx + 1,
                embargo_days = self.embargo_days,
            )
            folds.append(fold)

        return folds

    def summary(self, dataset: Dict[str, pd.DataFrame]) -> str:
        """打印所有折的日期范围摘要。"""
        folds = self.get_folds(dataset)
        lines = [
            "=" * 62,
            f"  WalkForwardPartitioner — {len(folds)} 折",
            f"  n_splits={self.n_splits}  embargo={self.embargo_days}d  "
            f"min_train={self.min_train_days}d",
            "=" * 62,
        ]
        for f in folds:
            lines.append(f"  {f}")
        lines.append("=" * 62)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ThreeWayPartitioner — IS / Validate / Test 三段切分（Phase S.1 + S.2）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThreeWaySplit:
    """
    三段切分结果。**Test 段是冻结 holdout**：选择流程（GP 适应度、参数调优、
    门限调参、人工挑选）都不得看它，只在最终汇报时用一次。

    frozen_by
    ---------
    ``"years"``  — Test 段由**日历**决定（数据末尾最近 N 年），与数据集长度无关。
                   这是 S.2 要求的口径：同一数据集上反复实验时 Test 段**不漂移**，
                   否则"换个起止日期再跑一遍"就等于换了一个新的样本外集。
    ``"ratio"``  — 数据太短，退回按比例切。**这不是等价替代**：比例切的 Test 段
                   会随入参漂移，可被反复挖掘。故 `degraded=True`，调用方必须能看见。
    """
    train:        Dict[str, pd.DataFrame]
    validate:     Dict[str, pd.DataFrame]
    test:         Dict[str, pd.DataFrame]
    #: IS + 段内 embargo + Validate 的**连续**前缀 —— 即"选择流程被允许看见的
    #: 全部数据"。给 WalkForward / DSR 这类需要连续时间轴的评估用：直接把
    #: train 与 validate 拼起来会在切点处留一个洞，滚动算子跨洞取数会出错。
    selection:    Dict[str, pd.DataFrame]
    n_is:         int
    n_val:        int
    n_test:       int
    embargo_days: int
    test_start:   Optional[pd.Timestamp]
    test_end:     Optional[pd.Timestamp]
    frozen_by:    str = "years"
    degraded:     bool = False

    @property
    def test_key(self) -> str:
        """Test 段的稳定标识（用于 holdout 使用次数台账对账）。"""
        if self.test_start is None or self.test_end is None:
            return "empty"
        return f"{self.test_start.date()}..{self.test_end.date()}"

    def to_dict(self) -> dict:
        return {
            "n_is": self.n_is, "n_val": self.n_val, "n_test": self.n_test,
            "embargo_days": self.embargo_days,
            "test_start": str(self.test_start.date()) if self.test_start is not None else None,
            "test_end":   str(self.test_end.date())   if self.test_end   is not None else None,
            "test_key":   self.test_key,
            "frozen_by":  self.frozen_by,
            "degraded":   self.degraded,
        }


class ThreeWayPartitioner:
    """
    IS / Validate / Test 三段时序切分（Phase S.1 + S.2）。

    布局（时间从左到右，embargo 段的样本**两边都不用**）::

        [        IS        ] embargo [ Validate ] embargo [   Test   ]
          结构搜索+参数调优             适应度/门限选择        冻结 holdout

    为什么需要三段
    --------------
    两段切割下 GP 直接按 OOS 择优 → **OOS 退化成第二个样本内**（循环论证）。
    三段把"用来选"和"用来报"分开：Validate 承担选择，Test 从不参与选择。

    为什么 Validate 与 Test 之间也要 embargo
    ----------------------------------------
    旧的 `_partition_three_way` 是纯 iloc 切片、**段间无间隔**。日频面板上任何
    滚动窗口算子（ts_mean(20) 等）都会让 Validate 末尾与 Test 开头共享原始
    bar —— 选择阶段因此能"隔着切点摸到"Test 的前几天。embargo 是防这个。

    Parameters
    ----------
    test_years   : Test 段按日历冻结的年数（默认 2.0）。None → 用 `test_ratio`。
    test_ratio   : 退化口径下 Test 占总样本比例（仅 test_years=None 或数据太短时用）。
    val_ratio    : Validate 占**剔除 Test 之后剩余样本**的比例（默认 0.25）。
    embargo_days : 段间隔离的交易日数（默认 20）。
    min_train_days : IS 的硬下限；不足则抛 ValueError（**绝不静默产出无效切分**）。
    注意：`years` 与 `years_capped` 的交界处，选择段会**变小一次**。
    例：900 天面板按份额封顶 → Test 315、选择段 565；1000 天面板已经能兑现
    完整的两年冻结 → Test 521、选择段 459。这不是 bug：跨过那条线之后，
    我们买到的是一个真正固定、不随入参漂移的样本外窗口，代价是研究样本一次性
    缩小。越过该点后随数据增加仍然单调。

    min_selection_days : selection 段（IS+embargo+Validate）的下限，默认 378
                   （≈1.5 年）。日历冻结若会让选择段小于它，就**按这个下限反推**
                   缩小 Test 并标 degraded —— 一个 2.8 年的面板冻结 2 年会只剩
                   180 天可研究，WalkForward 一折都切不出来，那不是"更严"，
                   是把门变成了恒定失败。
    """

    def __init__(
        self,
        test_years:     Optional[float] = 2.0,
        test_ratio:     float = 0.15,
        val_ratio:      float = 0.25,
        embargo_days:   int   = 20,
        min_train_days: int   = 60,
        min_selection_days: int = 378,
        max_test_share: float = 0.35,
    ) -> None:
        if test_years is not None and test_years <= 0:
            raise ValueError(f"test_years 必须为正，当前={test_years}")
        if not (0.0 < test_ratio < 1.0):
            raise ValueError(f"test_ratio 必须在 (0, 1)，当前={test_ratio}")
        if not (0.0 < val_ratio < 1.0):
            raise ValueError(f"val_ratio 必须在 (0, 1)，当前={val_ratio}")
        if embargo_days < 0:
            raise ValueError(f"embargo_days 不能为负，当前={embargo_days}")
        if min_train_days < 20:
            raise ValueError(f"min_train_days 至少为 20，当前={min_train_days}")
        if min_selection_days < min_train_days + 2 * embargo_days + 1:
            raise ValueError(
                f"min_selection_days={min_selection_days} 小于 IS 下限 + 两段 embargo + 1"
                f"（={min_train_days + 2 * embargo_days + 1}），切出来必然无效。")

        self.test_years     = test_years
        self.test_ratio     = test_ratio
        self.val_ratio      = val_ratio
        self.embargo_days   = embargo_days
        if not (0.0 < max_test_share < 1.0):
            raise ValueError(f"max_test_share 必须在 (0, 1)，当前={max_test_share}")
        self.min_train_days = min_train_days
        self.min_selection_days = min_selection_days
        self.max_test_share = max_test_share

    def partition(self, dataset: Dict[str, pd.DataFrame]) -> ThreeWaySplit:
        dates = _extract_dates(dataset)
        n = len(dates)
        if n == 0:
            raise ValueError("dataset 为空，无法三段切分")

        n_test, frozen_by, degraded = self._test_size(dates)

        # Test 起点（含）索引
        test_lo = n - n_test
        # Validate 与 Test 之间的 embargo
        val_hi = test_lo - self.embargo_days              # Validate 结束（不含）
        if val_hi <= 0:
            raise ValueError(
                f"数据不足以三段切分：总={n}，Test={n_test}，embargo={self.embargo_days}，"
                f"Validate 与 IS 已无位置。请延长区间、减小 test_years 或 embargo_days。"
            )

        remaining = val_hi                                 # IS + embargo + Validate
        n_val = max(1, int(round(remaining * self.val_ratio)))
        n_is  = remaining - self.embargo_days - n_val
        if n_is < self.min_train_days:
            raise ValueError(
                f"数据不足以三段切分：IS 只有 {n_is} 天 < 下限 {self.min_train_days}"
                f"（总={n}，Validate={n_val}，Test={n_test}，embargo={self.embargo_days}×2）。"
            )

        val_lo = n_is + self.embargo_days

        train     = _slice_positional(dataset, 0, n_is)
        validate  = _slice_positional(dataset, val_lo, val_hi)
        test      = _slice_positional(dataset, test_lo, n)
        selection = _slice_positional(dataset, 0, val_hi)

        split = ThreeWaySplit(
            train=train, validate=validate, test=test, selection=selection,
            n_is=n_is, n_val=val_hi - val_lo, n_test=n_test,
            embargo_days=self.embargo_days,
            test_start=dates[test_lo], test_end=dates[-1],
            frozen_by=frozen_by, degraded=degraded,
        )
        logger.info(
            "三段切分 | IS=%d | embargo=%d | Validate=%d | embargo=%d | Test=%d [%s] | 冻结口径=%s%s",
            split.n_is, self.embargo_days, split.n_val, self.embargo_days,
            split.n_test, split.test_key, frozen_by,
            "（退化，Test 段会随入参漂移）" if degraded else "",
        )
        return split

    def _test_size(self, dates: pd.DatetimeIndex) -> Tuple[int, str, bool]:
        """
        返回 (Test 段天数, 冻结口径, 是否退化)。

        日历冻结：从最后一个交易日往前推 test_years 年，落在窗口内的 bar 即 Test。
        **只有在剩余的选择段仍够做研究时才算真冻结**（`min_selection_days`）——
        否则按该下限反推一个更小的 Test 段，口径记为 ``years_capped`` 且
        `degraded=True`：请求的冻结窗口没被兑现这件事必须出现在返回值里，
        而不是变成一个"看起来也是两年"的数字。
        """
        n = len(dates)
        # selection 段 = [0, n - n_test - embargo)：Validate 与 Test 之间的 embargo
        # **也要从选择段里扣**，否则"留够 378 天"会少留一个 embargo 的量。
        keep = self.min_selection_days + self.embargo_days

        if self.test_years is not None:
            cutoff = dates[-1] - pd.DateOffset(days=int(round(self.test_years * 365.25)))
            n_cal = int((dates > cutoff).sum())
            if 0 < n_cal <= n - keep:
                return n_cal, "years", False
            # 兑现不了完整冻结窗口时，按**份额**封顶，而不是把选择段钉在下限上。
            #
            # 钉下限的写法有个隐蔽后果：n=400 与 n=900 会切出同样大的选择段
            # （多出来的数据全进了 Test），于是"给更多数据"在研究侧毫无变化 ——
            # B-2 修的正是"n_days 被静默忽略"，这等于换个地方把它又弄丢了。
            # 按份额封顶则对 n 单调：数据变多，研究样本与冻结样本一起变多。
            capped = min(n_cal, max(1, int(n * self.max_test_share)))
            if capped >= 1 and n - capped - self.embargo_days >= 1:
                logger.warning(
                    "[S.2] 请求冻结最近 %.2f 年（%d 天），但总样本只有 %d 天，"
                    "兑现后选择段只剩 %d 天 < 下限 %d（含段间 embargo）—— "
                    "Test 段**按份额上限 %.0f%% 压到 %d 天**。"
                    "它不再是『最近两年』那个固定窗口，会随数据起止漂移，"
                    "本次样本外结论强度更弱（degraded）。",
                    self.test_years, n_cal, n, n - n_cal, keep,
                    self.max_test_share * 100, capped,
                )
                return capped, "years_capped", True
            logger.warning(
                "[S.2] 样本 %d 天太短，连按份额切一个 Test 段都不够 —— "
                "退回**比例**口径 test_ratio=%.2f。",
                n, self.test_ratio,
            )

        n_ratio = max(1, int(round(n * self.test_ratio)))
        if n_ratio > n - self._min_room_for_selection():
            n_ratio = max(1, n - self._min_room_for_selection())
        return n_ratio, "ratio", True

    def _min_room_for_selection(self) -> int:
        """
        Test 段之外**至少**要留多少行，才能切出一个满足 IS 下限的选择段。

        不是 `min_train + 2*embargo + 1`：那个数假设 Validate 只占 1 行，而
        `n_val` 是按 `val_ratio` 从剩余里分的。第一版就是这么写的，于是
        `test_ratio=0.95` 被"夹住"之后 IS 仍然不够、照样抛错 —— 夹子没有兑现
        自己的承诺（它存在的全部意义就是"宁可少切 Test，也要给 IS 留够"）。

        解：remaining = n − n_test − embargo，要求
            remaining − embargo − remaining·val_ratio ≥ min_train
        ⟹ remaining ≥ (min_train + embargo) / (1 − val_ratio)
        """
        import math

        need_remaining = math.ceil(
            (self.min_train_days + self.embargo_days) / (1.0 - self.val_ratio))
        return self.embargo_days + need_remaining


def partition_three_way(
    dataset:      Dict[str, pd.DataFrame],
    test_years:   Optional[float] = 2.0,
    test_ratio:   float = 0.15,
    val_ratio:    float = 0.25,
    embargo_days: int   = 20,
    min_train_days: int = 60,
    min_selection_days: int = 378,
    max_test_share: float = 0.35,
) -> ThreeWaySplit:
    """`ThreeWayPartitioner` 的函数式入口（见该类的文档）。"""
    return ThreeWayPartitioner(
        test_years=test_years, test_ratio=test_ratio, val_ratio=val_ratio,
        embargo_days=embargo_days, min_train_days=min_train_days,
        min_selection_days=min_selection_days, max_test_share=max_test_share,
    ).partition(dataset)


def split_from_settings(
    dataset: Dict[str, pd.DataFrame], **overrides,
) -> ThreeWaySplit:
    """
    按发布配置（`s_*`）做三段切分。参数缺省时**不写死**，读 `Settings`。

    读不到配置时用本函数签名里的保守默认（2 年冻结 + 20 日 embargo），
    并记 error —— 与 DEV_LESSONS §U 同一条：兜底要朝**更严**的一侧倒，
    而"读不到就不切 Test"恰恰是最松的那一侧。
    """
    kwargs = dict(test_years=2.0, test_ratio=0.15, val_ratio=0.25,
                  embargo_days=20, min_train_days=60, min_selection_days=378,
                  max_test_share=0.35)
    try:
        from app.config import settings
        kwargs.update(
            test_years         = settings.s_test_freeze_years,
            val_ratio          = settings.s_val_ratio,
            embargo_days       = settings.s_embargo_days,
            min_selection_days = settings.s_min_selection_days,
            max_test_share     = settings.s_max_test_share,
        )
    except Exception as exc:
        logger.error(
            "[S.2] 读不到三段切分配置，改用保守默认 %s：%s", kwargs, exc)
    kwargs.update(overrides)
    return partition_three_way(dataset, **kwargs)


def account_holdout_use(
    split:       ThreeWaySplit,
    dataset_key: str,
    purpose:     str = "",
    budget:      Optional[int] = None,
) -> dict:
    """
    在**真的用了** Test 段算指标之后调用：记一次使用并返回可直接塞进响应体 /
    RunManifest 的字典（Phase S.2）。

    返回值总是包含 `uses` / `budget` / `over_budget`；台账不可写时
    `recorded=False` —— **结论照常返回，但"次数已失真"必须可见**，
    不能因为记账失败就假装这是第一次看这段数据。
    """
    payload = dict(split.to_dict())
    try:
        from app.db.trial_ledger import HoldoutLedger
        if budget is None:
            try:
                from app.config import settings
                budget = int(settings.s_holdout_budget)
            except Exception as exc:
                logger.warning("[S.2] 读不到 s_holdout_budget，按最严的 1 次计: %s", exc)
                budget = 1
        usage = HoldoutLedger(budget=budget).record_use(
            dataset_key=dataset_key, test_key=split.test_key, purpose=purpose)
        payload.update(usage.to_dict())
    except Exception as exc:
        logger.error(
            "[S.2] holdout 使用次数无法记账（%s）—— 本次 Test 段结论仍返回，"
            "但『已用几次』这个数字从此刻起是不准的。", exc)
        payload.update({
            "dataset_key": dataset_key, "test_key": split.test_key,
            "uses": -1, "budget": budget if budget is not None else 1,
            "over_budget": False, "recorded": False,
        })
    return payload


# ---------------------------------------------------------------------------
# PurgedKFold — IS 内部 purged + embargoed K 折（Phase S.1）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PurgedFold:
    """单折：`test` 是被留出的连续块，`train` 已剔除其两侧的 purge/embargo 带。"""
    fold_idx:   int
    train_idx:  "pd.Index"
    test_idx:   "pd.Index"

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)


class PurgedKFold:
    """
    时序 K 折交叉验证，带 **purging + embargo**（López de Prado, AFML §7）。

    与普通 KFold 的区别：留出块**两侧**各剔除 `embargo_days` 个样本再作训练集。
    日频面板上滚动算子会让相邻样本共享原始 bar，不剔除就是跨折泄漏 ——
    "K 折平均 Sharpe" 会被这种泄漏系统性抬高。

    与单段 holdout 的区别：**每个样本都轮流当过一次留出**，适应度不再取决于
    "最后那一段恰好是什么行情"，单段验证的方差与段位偏倚一并降低。

    注意：本类只做**索引**切分，不碰数据 —— 调用方用 `fold.test_idx` 自己取片。
    """

    def __init__(self, n_splits: int = 5, embargo_days: int = 20) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits 至少为 2，当前={n_splits}")
        if embargo_days < 0:
            raise ValueError(f"embargo_days 不能为负，当前={embargo_days}")
        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def split(self, dates: pd.DatetimeIndex) -> List[PurgedFold]:
        idx = pd.DatetimeIndex(dates)
        n = len(idx)
        if n < self.n_splits * (2 * self.embargo_days + 2):
            raise ValueError(
                f"样本不足以做 {self.n_splits} 折 purged CV："
                f"n={n}，embargo={self.embargo_days}。"
            )

        bounds = np.array_split(np.arange(n), self.n_splits)
        folds: List[PurgedFold] = []
        for i, block in enumerate(bounds):
            if len(block) == 0:
                continue
            lo, hi = int(block[0]), int(block[-1])
            purge_lo = max(0, lo - self.embargo_days)
            purge_hi = min(n - 1, hi + self.embargo_days)
            train_pos = np.concatenate([
                np.arange(0, purge_lo),
                np.arange(purge_hi + 1, n),
            ])
            if len(train_pos) == 0:
                continue
            folds.append(PurgedFold(
                fold_idx=i,
                train_idx=idx[train_pos],
                test_idx=idx[lo:hi + 1],
            ))
        if not folds:
            raise ValueError("purged CV 切不出任何有效折（embargo 过大？）")
        return folds


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _slice_positional(
    dataset: Dict[str, pd.DataFrame], lo: int, hi: int,
) -> Dict[str, pd.DataFrame]:
    """按**位置**区间 [lo, hi) 切片（三段切分用；各字段 index 对齐由上游保证）。"""
    return {field: df.iloc[lo:hi].copy() for field, df in dataset.items()}


def _extract_dates(dataset: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """从 dataset 中取任意字段的日期索引并排序。"""
    if not dataset:
        raise ValueError("dataset 不能为空")
    raw_idx = next(iter(dataset.values())).index
    return pd.DatetimeIndex(raw_idx).sort_values()


def _slice_dataset(
    dataset: Dict[str, pd.DataFrame],
    *,
    start:     Optional[pd.Timestamp] = None,
    end:       Optional[pd.Timestamp] = None,
    inclusive: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    按日期范围切片 dataset，返回深拷贝。

    Parameters
    ----------
    start/end   : 起止日期（None 表示不限）
    inclusive   : True = 两端包含
    """
    result: Dict[str, pd.DataFrame] = {}
    for field_name, df in dataset.items():
        idx = pd.DatetimeIndex(df.index)
        mask = pd.Series(True, index=range(len(idx)))

        if start is not None:
            mask &= (idx >= start) if inclusive else (idx > start)
        if end is not None:
            mask &= (idx <= end)   if inclusive else (idx < end)

        result[field_name] = df.loc[mask.values].copy()

    return result
