"""
data_partitioner.py — 严格 In-Sample / Out-of-Sample 数据分区引擎

核心目标：防止过拟合（Anti-Overfitting Core）。

设计原则：
  1. 物理分区：IS 和 OOS 数据存于独立内存，无共享引用
  2. 严格隔离：优化算法（GP/Agent）只能访问 .train() 分区
  3. 不可篡改：PartitionedDataset 使用 Python __slots__ + 私有属性防止外部写入
  4. 无泄漏：切分点使用严格不等式（IS 末日 < split_date），确保无重叠

用法：
    partitioner = DataPartitioner(start="2020-01-01", end="2023-12-31", oos_ratio=0.3)
    parts = partitioner.partition(dataset)   # dict[str, pd.DataFrame]

    # 搜索阶段：只用 IS 数据
    is_data = parts.train()
    hof = evolver.evolve(is_data)

    # 评估阶段：用 OOS 数据
    oos_data = parts.test()
    report = backtester.run(best_dsl, oos_data)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

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
    split_date : IS/OOS 切分日期（第一个 OOS 交易日）
    oos_ratio  : OOS 占比
    is_days    : IS 交易日数
    oos_days   : OOS 交易日数
    """

    __slots__ = ("_train", "_test", "split_date", "oos_ratio", "is_days", "oos_days")

    def __init__(
        self,
        train_data:  Dict[str, pd.DataFrame],
        test_data:   Dict[str, pd.DataFrame],
        split_date:  pd.Timestamp,
        oos_ratio:   float,
        is_days:     int,
        oos_days:    int,
    ) -> None:
        # 深拷贝确保物理隔离——外部持有原始 dict 时无法通过修改影响分区
        object.__setattr__(self, "_train", {k: v.copy() for k, v in train_data.items()})
        object.__setattr__(self, "_test",  {k: v.copy() for k, v in test_data.items()})

        object.__setattr__(self, "split_date", split_date)
        object.__setattr__(self, "oos_ratio",  oos_ratio)
        object.__setattr__(self, "is_days",    is_days)
        object.__setattr__(self, "oos_days",   oos_days)

    def __setattr__(self, name: str, value: object) -> None:
        # 禁止外部赋值（除 __init__ 内部通过 object.__setattr__）
        raise AttributeError("PartitionedDataset 是只读对象，不允许修改分区数据。")

    # ------------------------------------------------------------------
    # 公开访问接口
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, pd.DataFrame]:
        """
        返回 In-Sample（训练集）数据。
        优化算法（GP/Agent/参数搜索）应**仅**使用此分区。
        """
        return {k: v.copy() for k, v in self._train.items()}

    def test(self) -> Dict[str, pd.DataFrame]:
        """
        返回 Out-of-Sample（测试集）数据。
        仅在最终验证阶段使用，绝不在参数搜索期间访问。
        """
        return {k: v.copy() for k, v in self._test.items()}

    def summary(self) -> str:
        """打印分区摘要信息。"""
        # 取任意字段的 IS/OOS 日期范围
        any_train = next(iter(self._train.values()))
        any_test  = next(iter(self._test.values()))

        is_start  = any_train.index[0]  if len(any_train) else "N/A"
        is_end    = any_train.index[-1] if len(any_train) else "N/A"
        oos_start = any_test.index[0]   if len(any_test)  else "N/A"
        oos_end   = any_test.index[-1]  if len(any_test)  else "N/A"

        lines = [
            "=" * 52,
            "  DataPartitioner — IS/OOS 分区摘要",
            "=" * 52,
            f"  切分日期     : {self.split_date.date()}",
            f"  IS 期间      : {is_start} → {is_end}  ({self.is_days} 交易日)",
            f"  OOS 期间     : {oos_start} → {oos_end}  ({self.oos_days} 交易日)",
            f"  OOS 占比     : {self.oos_ratio * 100:.1f}%",
            f"  字段数量     : {len(self._train)}",
            "=" * 52,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DataPartitioner — 分区逻辑
# ---------------------------------------------------------------------------

class DataPartitioner:
    """
    基于时间轴的严格 In-Sample / Out-of-Sample 物理分区器。

    Parameters
    ----------
    start     : 全局起始日期（字符串，"YYYY-MM-DD"）
    end       : 全局终止日期（字符串，"YYYY-MM-DD"）
    oos_ratio : OOS 占全部交易日的比例（默认 0.30 = 后 30%）

    Notes
    -----
    - 切分点严格按照实际交易日历计算（使用 pd.bdate_range）
    - IS 末日 < split_date，OOS 首日 = split_date，无重叠
    - oos_ratio = 0 时不做 OOS 分割（test 返回空 DataFrame）
    """

    def __init__(
        self,
        start:     str,
        end:       str,
        oos_ratio: float = 0.30,
    ) -> None:
        if not (0.0 <= oos_ratio < 1.0):
            raise ValueError(
                f"oos_ratio 必须在 [0, 1) 范围内，当前={oos_ratio}"
            )

        self.start     = pd.Timestamp(start)
        self.end       = pd.Timestamp(end)
        self.oos_ratio = oos_ratio

        # 计算切分日期（按工作日历）
        all_bdays  = pd.bdate_range(self.start, self.end)
        total_days = len(all_bdays)

        if total_days < 10:
            raise ValueError(
                f"日期范围 {start}→{end} 交易日不足（{total_days}天），"
                "无法进行有意义的分区。"
            )

        is_count = int(round(total_days * (1.0 - oos_ratio)))
        is_count = max(1, min(is_count, total_days - 1)) if oos_ratio > 0 else total_days

        self._all_bdays   = all_bdays
        self._is_count    = is_count
        self._oos_count   = total_days - is_count
        # IS 末日（包含）
        self._is_end      = all_bdays[is_count - 1]
        # OOS 首日（切分边界）
        self._split_date  = all_bdays[is_count] if oos_ratio > 0 and is_count < total_days else None

        logger.info(
            "DataPartitioner 初始化 | 总交易日=%d | IS=%d | OOS=%d | split=%s",
            total_days, is_count, self._oos_count,
            self._split_date.date() if self._split_date else "N/A",
        )

    # ------------------------------------------------------------------

    def partition(
        self,
        dataset: Dict[str, pd.DataFrame],
    ) -> PartitionedDataset:
        """
        将 dataset（字段 → T×N DataFrame）物理切分为 IS 和 OOS 两份。

        Parameters
        ----------
        dataset : dict，key=字段名，value=(T×N) DataFrame，index=DatetimeIndex

        Returns
        -------
        PartitionedDataset（只读，IS/OOS 物理隔离）
        """
        if not dataset:
            raise ValueError("dataset 不能为空")

        train_data: Dict[str, pd.DataFrame] = {}
        test_data:  Dict[str, pd.DataFrame] = {}

        for field, df in dataset.items():
            df_idx = pd.DatetimeIndex(df.index)

            if self._split_date is None or self.oos_ratio == 0.0:
                # 无 OOS 分割
                train_data[field] = df.copy()
                test_data[field]  = df.iloc[0:0].copy()   # 空 DataFrame，保留列名
            else:
                # 严格切分：IS = [start, split_date)，OOS = [split_date, end]
                # 使用布尔掩码（向量化，无 T 轴循环）
                is_mask  = df_idx < self._split_date     # IS: 严格小于（无重叠）
                oos_mask = df_idx >= self._split_date    # OOS: 大于等于

                train_data[field] = df.loc[is_mask].copy()
                test_data[field]  = df.loc[oos_mask].copy()

        # 计算实际 IS/OOS 天数（取第一个字段）
        first_field = next(iter(train_data))
        actual_is_days  = len(train_data[first_field])
        actual_oos_days = len(test_data[first_field])

        parts = PartitionedDataset(
            train_data = train_data,
            test_data  = test_data,
            split_date = self._split_date or self.end,
            oos_ratio  = self.oos_ratio,
            is_days    = actual_is_days,
            oos_days   = actual_oos_days,
        )

        logger.info(
            "分区完成 | IS=%d天 | OOS=%d天 | 字段=%d个",
            actual_is_days, actual_oos_days, len(dataset),
        )
        return parts

    @property
    def split_date(self) -> Optional[pd.Timestamp]:
        """OOS 起始日期（切分边界），oos_ratio=0 时为 None。"""
        return self._split_date

    def summary(self) -> str:
        """返回分区配置摘要字符串。"""
        lines = [
            "=" * 52,
            "  DataPartitioner 配置",
            "=" * 52,
            f"  全局范围     : {self.start.date()} → {self.end.date()}",
            f"  总交易日     : {len(self._all_bdays)}",
            f"  IS 天数      : {self._is_count}",
            f"  OOS 天数     : {self._oos_count}",
            f"  OOS 占比     : {self.oos_ratio * 100:.1f}%",
            f"  切分日期     : {self._split_date.date() if self._split_date else 'N/A（无OOS）'}",
            "=" * 52,
        ]
        return "\n".join(lines)
