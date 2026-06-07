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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    oos_ratio    : OOS 占全部交易日的比例（默认 0.30）
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

        is_count = int(round(total_days * (1.0 - oos_ratio)))
        is_count = max(1, min(is_count, total_days - 1)) if oos_ratio > 0 else total_days

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

        self._oos_count = max(0, total_days - is_count - embargo_days)

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
            is_data  = _slice_dataset(dataset, end=fold.is_end,    inclusive=True)
            oos_data = _slice_dataset(dataset, start=fold.oos_start, inclusive=True)
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

        策略：
          - 整体 OOS 区域 = 后 (n_splits * oos_per_fold) 个交易日
          - IS 从全局起始到各 OOS 段之前（扩展窗口）
          - 每折 OOS 大小相等（最后一折可能略多）
        """
        n = len(dates)
        if n < self.min_train_days + self.embargo_days + self.n_splits:
            return []

        # 可用于 OOS 的总天数 = 总天数 - 最小 IS 天数
        available_oos = n - self.min_train_days
        oos_per_fold  = max(1, available_oos // (self.n_splits + 1))
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
# 内部辅助函数
# ---------------------------------------------------------------------------

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
