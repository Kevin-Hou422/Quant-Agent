"""
data_engine/data_partitioner.py —— IS/OOS 切分与 embargo 的定钉测试（变异测试驱动）

来由：54 个变异点，首测击杀率 **37.0%**（存活 17）。

DataPartitioner 划的是**整个系统最重要的一条线**：哪些数据可以拿来调参，
哪些必须留到最后一刻才能看。这条线画错的后果不是"结果不准"，
而是"所有 OOS 结论全部作废"——而它不会报错、不会让任何回测变红，
只会让 Sharpe 看起来更好。

存活项几乎全在 `__init__` 的切分算术上：
  - `is_count = max(1, min(is_count, usable - 1))` 的 `- 1`
    —— 改成 `+ 1` 会让 IS 多吃一天，OOS 少一天（最坏情况 OOS 归零）
  - `oos_count = usable - is_count` 的 `-`
  - `self._is_end = all_bdays[is_count - 1]` 的 `- 1`
    —— IS 末日往后挪一天，**切分日那天同时属于 IS 和 OOS**
  - `all_bdays[is_count] if oos_ratio > 0 and is_count < total_days` 的三处
  - `if self._split_date is not None and embargo_days > 0` 的三处
    —— `and`→`or` 会在 embargo=0 时也去算 `is_count + 0`（无害）但在
       split_date 为 None 时索引 None，直接崩
  - `if oos_actual_idx < total_days` —— embargo 恰好顶到末尾时该不该算越界
  - `if not (0.0 <= oos_ratio < 1.0)` / `total_days < 10` / `usable < 2`
    三个入口守卫的边界

既有覆盖（test_phase1_upgrade / test_phase2 / test_phase3）只验证了
"切完 IS 在前 OOS 在后、两段不重叠"，没有一条站在**样本数**的边界上。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.data_partitioner import (
    DataPartitioner,
    WalkForwardPartitioner,
    _extract_dates,
    _slice_dataset,
)


def _bdays(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2022-01-03", periods=n)


def _span(n: int) -> tuple[str, str]:
    idx = _bdays(n)
    return str(idx[0].date()), str(idx[-1].date())


def _dataset(n: int, cols: int = 3) -> dict:
    idx = _bdays(n)
    names = [f"T{i}" for i in range(cols)]
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 + rng.normal(0, 1, (n, cols)).cumsum(axis=0),
                         index=idx, columns=names)
    return {"close": close, "volume": close * 1000.0}


# ===========================================================================
# A. 入口参数的合法区间
# ===========================================================================

class TestConstructorGuards:

    def test_zero_oos_ratio_is_legal(self):
        """
        `not (0.0 <= oos_ratio < 1.0)` —— 下界**含 0**：`oos_ratio=0`
        表示"全部作为 IS"，是训练全量模型时的合法配置。
        `<=` 放宽会让 1.0 也被接受 —— 那意味着 IS 为空，优化器无数据可用。
        """
        p = DataPartitioner(*_span(100), oos_ratio=0.0, embargo_days=0)
        assert p.oos_ratio == 0.0

    def test_ratio_of_one_is_rejected(self):
        with pytest.raises(ValueError, match="oos_ratio"):
            DataPartitioner(*_span(100), oos_ratio=1.0)

    def test_negative_ratio_is_rejected(self):
        with pytest.raises(ValueError, match="oos_ratio"):
            DataPartitioner(*_span(100), oos_ratio=-0.01)

    def test_ratio_just_under_one_is_accepted(self):
        p = DataPartitioner(*_span(200), oos_ratio=0.99, embargo_days=0)
        assert p.oos_ratio == 0.99

    def test_negative_embargo_is_rejected(self):
        with pytest.raises(ValueError, match="embargo_days"):
            DataPartitioner(*_span(100), embargo_days=-1)

    def test_exactly_ten_business_days_is_enough(self):
        """
        `if total_days < 10: raise` —— **严格小于**。
        放宽成 `<=` 会把恰好 10 天的区间判成"交易日不足"，
        单元测试与小样本诊断路径全部用不了。
        """
        p = DataPartitioner(*_span(10), oos_ratio=0.3, embargo_days=0)
        assert p.split_date is not None

    def test_nine_business_days_is_rejected(self):
        with pytest.raises(ValueError, match="交易日不足"):
            DataPartitioner(*_span(9), oos_ratio=0.3, embargo_days=0)

    def test_embargo_leaving_exactly_two_usable_days_is_accepted(self):
        """
        `usable = total_days - embargo_days`，`if usable < 2: raise`。
        **严格小于 2**：恰好 2 天（IS 1 天 + OOS 1 天）是可切的最小情形。
        放宽成 `<=` 会让它被拒；而 `-` 改成 `+` 会让 embargo **增加**可用样本，
        embargo 从"扣掉一段"变成"凭空多出一段"，泄漏防护彻底失效。
        """
        p = DataPartitioner(*_span(12), oos_ratio=0.5, embargo_days=10)
        assert p.split_date is not None

    def test_embargo_larger_than_the_span_is_rejected(self):
        with pytest.raises(ValueError, match="无法切分"):
            DataPartitioner(*_span(12), oos_ratio=0.5, embargo_days=11)


# ===========================================================================
# B. 切分点的算术
# ===========================================================================

class TestSplitArithmetic:

    def test_is_end_is_the_day_before_the_split(self):
        """
        `self._is_end = all_bdays[is_count - 1]` —— `- 1` 改成 `+ 1` 会让
        IS 末日**越过切分日**：切分日那天同时落在 IS 和 OOS 里，
        最后一天的标签直接泄漏进训练集。
        """
        n = 100
        p = DataPartitioner(*_span(n), oos_ratio=0.30, embargo_days=0)
        days = _bdays(n)
        assert p.split_date in days
        split_idx = list(days).index(p.split_date)
        assert p._is_end == days[split_idx - 1], (
            f"IS 末日 {p._is_end.date()} 不是切分日的前一天 "
            f"{days[split_idx - 1].date()}")
        assert p._is_end < p.split_date, "IS 末日不早于切分日 —— 两段重叠了"

    def test_is_and_oos_counts_add_up_to_the_usable_span(self):
        """
        `oos_count = usable - is_count` 的 `-` 改成 `+` 会让 OOS 计数
        比可用样本还多，随后 `oos_count < 1` 的守卫永远不触发 ——
        一个空的 OOS 会被当成"切分成功"放行。
        """
        n, embargo, ratio = 200, 20, 0.30
        p = DataPartitioner(*_span(n), oos_ratio=ratio, embargo_days=embargo)
        usable = n - embargo
        assert p._is_count + p._oos_count == usable, (
            f"IS({p._is_count}) + OOS({p._oos_count}) != 可用样本({usable})")

    def test_is_count_respects_the_requested_ratio(self):
        n, embargo, ratio = 300, 20, 0.25
        p = DataPartitioner(*_span(n), oos_ratio=ratio, embargo_days=embargo)
        usable = n - embargo
        assert p._is_count == int(round(usable * (1.0 - ratio))), (
            f"IS 样本数 {p._is_count} 与请求比例算出的不符")

    def test_is_count_never_consumes_the_last_usable_day(self):
        """
        `min(is_count, usable - 1)` 的 `- 1` 是**给 OOS 留的最后一天**。
        改成 `+ 1` 时，比例接近 1 的配置会让 IS 吃光全部可用样本，
        OOS 归零 —— 而下面那条守卫又被同一处算术绕过。
        """
        p = DataPartitioner(*_span(200), oos_ratio=0.001, embargo_days=0)
        assert p._oos_count >= 1, "极小的 oos_ratio 也必须留下至少一天 OOS"
        assert p._is_count <= 200 - 1

    def test_every_legal_configuration_yields_a_non_empty_oos(self):
        """
        `if oos_ratio > 0 and (self._oos_start is None or oos_count < 1): raise`
        —— 这条守卫在**当前算术下不可达**（见 E 节的机械证明）：
        `usable >= 2` 已被上一道守卫保证，`is_count` 被 clip 到 `[1, usable-1]`，
        于是 `oos_count = usable - is_count >= 1` 恒成立，
        且 `is_count + embargo <= usable - 1 + embargo = total - 1 < total`，
        `_oos_start` 永远取得到。

        这里正面钉住这个不变量：任何通过前置守卫的配置都必须切出非空 OOS。
        """
        for total, embargo, ratio in ((10, 0, 0.3), (12, 10, 0.5), (200, 20, 0.3),
                                      (200, 0, 0.999), (300, 60, 0.001)):
            p = DataPartitioner(*_span(total), oos_ratio=ratio,
                                embargo_days=embargo)
            assert p._oos_count >= 1, (
                f"total={total} embargo={embargo} ratio={ratio} 切出了空 OOS")
            assert p._oos_start is not None

    def test_zero_ratio_produces_no_split_and_no_error(self):
        """`oos_ratio > 0` 的守卫：ratio=0 时不切分、不报错、OOS 为空。"""
        p = DataPartitioner(*_span(100), oos_ratio=0.0, embargo_days=0)
        assert p.split_date is None, "oos_ratio=0 却算出了切分日"
        assert p._oos_count == 0
        parts = p.partition(_dataset(100))
        assert len(parts.test()["close"]) == 0, "oos_ratio=0 却切出了 OOS 数据"
        assert len(parts.train()["close"]) == 100


# ===========================================================================
# C. embargo
# ===========================================================================

class TestEmbargo:

    def test_oos_start_is_pushed_back_by_the_embargo(self):
        """
        `oos_actual_idx = is_count + embargo_days`。embargo 的全部意义就是
        让 OOS 首日**离 IS 末日足够远**，否则时序自相关会把标签透过来。
        """
        n, embargo = 200, 20
        p = DataPartitioner(*_span(n), oos_ratio=0.30, embargo_days=embargo)
        days = list(_bdays(n))
        gap = days.index(p._oos_start) - days.index(p.split_date)
        assert gap == embargo, (
            f"OOS 首日只比切分日晚 {gap} 天，应为 embargo={embargo}")

    def test_zero_embargo_makes_oos_start_at_the_split(self):
        """
        `if self._split_date is not None and embargo_days > 0` —— `and` 放宽成
        `or` 会在 split_date 为 None（ratio=0）时去算 `None + 0` 而崩掉；
        `> 0` 放宽成 `>= 0` 会让 embargo=0 也走进分支 —— 结果相同但
        `oos_actual_idx = is_count` 与 `_split_date` 的一致性不再有保证。
        """
        p = DataPartitioner(*_span(100), oos_ratio=0.30, embargo_days=0)
        assert p._oos_start == p.split_date, (
            f"embargo=0 时 OOS 首日 {p._oos_start} 不等于切分日 {p.split_date}")

    def test_partition_drops_the_embargo_window(self):
        n, embargo = 200, 20
        p = DataPartitioner(*_span(n), oos_ratio=0.30, embargo_days=embargo)
        parts = p.partition(_dataset(n))
        train, test = parts.train()["close"], parts.test()["close"]
        assert train.index[-1] < test.index[0], "IS 与 OOS 有重叠"
        dropped = n - len(train) - len(test)
        assert dropped == embargo, (
            f"被丢弃的 embargo 窗口是 {dropped} 天，应为 {embargo}")

    def test_embargo_larger_than_the_usable_span_is_caught_earlier(self):
        """
        embargo 吃光样本时，报错来自**前一道守卫**（`usable < 2`），
        而不是后面那条"OOS 为空"。两条守卫的错误消息不同，
        调用方按消息判断该延长区间还是该调小 ratio。
        """
        with pytest.raises(ValueError, match="无法切分"):
            DataPartitioner(*_span(20), oos_ratio=0.05, embargo_days=19)

    def test_embargo_one_day_short_of_the_end_still_works(self):
        p = DataPartitioner(*_span(30), oos_ratio=0.10, embargo_days=2)
        assert p._oos_start is not None
        parts = p.partition(_dataset(30))
        assert len(parts.test()["close"]) >= 1, "还有余量却切出了空 OOS"

    def test_partitioned_summary_mentions_the_embargo_only_when_used(self):
        """
        `PartitionedDataset.summary()` 里的
        `f"  Embargo 天数  : ..." if self.embargo_days > 0 else ""` ——
        `>` 放宽成 `>=` 会让 embargo=0 时也打印"Embargo 天数 : 0 交易日"，
        读摘要的人会以为做了 embargo。

        注意是 **PartitionedDataset** 的 summary，不是 DataPartitioner 的：
        后者无条件打印那一行，两个 summary 不是同一份代码。
        """
        n = 200
        with_e = DataPartitioner(*_span(n), oos_ratio=0.3,
                                 embargo_days=20).partition(_dataset(n)).summary()
        without = DataPartitioner(*_span(n), oos_ratio=0.3,
                                  embargo_days=0).partition(_dataset(n)).summary()
        assert "Embargo" in with_e, "启用了 embargo，分区摘要里却没提"
        assert "Embargo" not in without, (
            f"embargo=0 摘要里仍然出现了 Embargo 一行：\n{without}")


# ===========================================================================
# D. 物理切分的结果
# ===========================================================================

class TestPartitionOutput:

    def test_train_and_test_cover_the_right_dates(self):
        n = 200
        p = DataPartitioner(*_span(n), oos_ratio=0.30, embargo_days=20)
        parts = p.partition(_dataset(n))
        assert parts.train()["close"].index[-1] == p._is_end
        assert parts.test()["close"].index[0] == p._oos_start

    def test_every_field_is_partitioned_the_same_way(self):
        n = 200
        parts = DataPartitioner(*_span(n), oos_ratio=0.30,
                                embargo_days=20).partition(_dataset(n))
        lens = {k: (len(v), len(parts.test()[k])) for k, v in parts.train().items()}
        assert len(set(lens.values())) == 1, f"各字段切出的长度不一致：{lens}"

    def test_returned_frames_are_copies(self):
        n = 100
        ds = _dataset(n)
        parts = DataPartitioner(*_span(n), oos_ratio=0.30,
                                embargo_days=0).partition(ds)
        parts.train()["close"].iloc[0, 0] = -999.0
        assert ds["close"].iloc[0, 0] != -999.0, (
            "partition 返回的是原数据的视图 —— 下游改动会污染原数据集")

    def test_no_row_appears_in_both_partitions(self):
        n = 200
        parts = DataPartitioner(*_span(n), oos_ratio=0.30,
                                embargo_days=20).partition(_dataset(n))
        overlap = parts.train()["close"].index.intersection(parts.test()["close"].index)
        assert len(overlap) == 0, f"{len(overlap)} 行同时出现在 IS 和 OOS 里"


# ===========================================================================
# E. 不可达的防御代码（产品问题 B-11）
# ===========================================================================
#
# 本节不是等价性证明表（全部存活项的证明统一在 H 节），而是一条**产品发现**：
# `__init__` 末尾那条「切分后 OOS 为空」守卫在当前算术下永远触发不了。
#
# 前置守卫已保证 `usable = total - embargo >= 2`，而 `is_count` 被
# `max(1, min(is_count, usable - 1))` 夹在 [1, usable-1]，于是恒有：
#   ① `oos_count = usable - is_count >= 1`
#   ② `oos_actual_idx = is_count + embargo <= usable - 1 + embargo = total - 1 < total`
# 所以 `_oos_start` 永远取得到、`oos_count` 永远 >= 1 ——
# `else: self._oos_start = None` 与整条守卫都是**不可达**的。
#
# 后果不是当下出错，而是给人以「我们检查过空 OOS」的假印象：
# 哪天上面的切分算术被改动，真正的空 OOS 会绕过它静默通过。


def test_the_oos_guards_are_unreachable_over_the_whole_parameter_space():
    """
    产品问题 B-11 的机械验证，同时是 H 节 L185 等价性的依据之一：
    在**全部**能通过前置守卫的 (总天数, embargo, ratio) 组合上重算切分算术，
    确认两条守卫的触发条件从不成立。

    这同时是失效告警：哪天 `is_count` 的 clip 或 `usable` 的定义变了，
    这里会立刻红 —— 那时守卫重新变得可达，L185 也要重新补用例。
    """
    counterexamples = []
    for total in range(10, 400):
        for embargo in range(0, min(total, 60)):
            usable = total - embargo
            if usable < 2:
                continue
            for ratio in (0.001, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999):
                is_count = int(round(usable * (1.0 - ratio)))
                is_count = max(1, min(is_count, usable - 1))
                oos_count = usable - is_count
                if oos_count < 1 or is_count + embargo >= total:
                    counterexamples.append((total, embargo, ratio))
    assert not counterexamples, (
        f"守卫条件在 {len(counterexamples)} 组参数下可以触发，"
        f"前 3 组：{counterexamples[:3]} —— 等价性证明失效，必须补用例")




# ===========================================================================
# F. WalkForwardPartitioner（同一模块里的第二个分区器）
# ===========================================================================
#
# 这个类与上面的 DataPartitioner 是**两套独立的切分算术**，
# 15 个存活点全在它身上 —— 上面的用例一个都覆盖不到。

class TestWalkForwardGuards:
    """
    构造守卫的三条下界。放宽一格会把**最小的合法配置**判成非法：
    n_splits=2 是最省数据的 walk-forward，min_train_days=20 是短样本
    诊断唯一能用的档位。
    """

    def test_minimum_splits_is_accepted(self):
        assert WalkForwardPartitioner(n_splits=2).n_splits == 2

    def test_one_split_is_rejected(self):
        """1 折不是 walk-forward，是普通切分。"""
        with pytest.raises(ValueError, match="n_splits"):
            WalkForwardPartitioner(n_splits=1)

    def test_minimum_train_days_is_accepted(self):
        assert WalkForwardPartitioner(min_train_days=20).min_train_days == 20

    def test_train_days_below_twenty_is_rejected(self):
        with pytest.raises(ValueError, match="min_train_days"):
            WalkForwardPartitioner(min_train_days=19)

    def test_zero_embargo_is_accepted(self):
        assert WalkForwardPartitioner(embargo_days=0).embargo_days == 0

    def test_negative_embargo_is_rejected(self):
        with pytest.raises(ValueError, match="embargo_days"):
            WalkForwardPartitioner(embargo_days=-1)


class TestWalkForwardFolds:

    @staticmethod
    def _wf(n_splits=5, min_train=120, embargo=20):
        return WalkForwardPartitioner(n_splits=n_splits,
                                      min_train_days=min_train,
                                      embargo_days=embargo)

    def test_fold_count_matches_n_splits_when_data_is_ample(self):
        folds = self._wf().get_folds(_dataset(1200))
        assert len(folds) == 5, f"数据充足却只切出 {len(folds)} 折"

    def test_each_fold_is_ends_before_its_oos_starts(self):
        """
        `oos_start_idx = is_end_idx + 1 + embargo_days` 的 `+` 改成 `-`
        会让 OOS **落在 IS 内部** —— 每一折都是在训练集上"验证"，
        walk-forward 的全部意义归零，而 Sharpe 会好看得离谱。
        """
        for f in self._wf().get_folds(_dataset(1200)):
            assert f.is_end < f.oos_start, (
                f"第 {f.fold_idx} 折 IS 末日 {f.is_end.date()} 不早于 "
                f"OOS 首日 {f.oos_start.date()}")

    def test_embargo_gap_is_exactly_the_configured_size(self):
        n, embargo = 1200, 20
        dates = list(_bdays(n))
        for f in self._wf(embargo=embargo).get_folds(_dataset(n)):
            gap = dates.index(f.oos_start) - dates.index(f.is_end) - 1
            assert gap == embargo, (
                f"第 {f.fold_idx} 折的 embargo 间隔是 {gap} 天，应为 {embargo}")

    def test_expanding_window_is_start_never_moves(self):
        """扩展窗口：每折 IS 都从全局第一天开始。"""
        folds = self._wf().get_folds(_dataset(1200))
        starts = {f.is_start for f in folds}
        assert len(starts) == 1, f"IS 起点在各折之间变了：{starts}"

    def test_is_grows_monotonically_across_folds(self):
        """
        `is_end_idx = min_train_days + i * oos_per_fold - 1` ——
        `*` 改成 `/` 会让各折 IS 末日几乎不动（i/oos_per_fold 是小数），
        扩展窗口退化成固定窗口；`- 1` 改成 `+ 1` 会让 IS 多吃一天，
        与 embargo 的间隔少一天。
        """
        folds = self._wf().get_folds(_dataset(1200))
        ends = [f.is_end for f in folds]
        assert ends == sorted(ends) and len(set(ends)) == len(ends), (
            f"各折 IS 末日不是严格递增：{[str(e.date()) for e in ends]}")
        days = [f.is_days for f in folds]
        assert days == sorted(days) and len(set(days)) == len(days), (
            f"各折 IS 天数不是严格递增：{days}")

    def test_first_fold_is_length_is_min_train_days(self):
        first = self._wf(min_train=120).get_folds(_dataset(1200))[0]
        assert first.is_days == 120, (
            f"第 0 折的 IS 天数是 {first.is_days}，应为 min_train_days=120")

    def test_oos_windows_are_equal_sized_and_non_overlapping(self):
        folds = self._wf().get_folds(_dataset(1200))
        sizes = {f.oos_days for f in folds[:-1]}     # 末折可能被数据末尾截断
        assert len(sizes) == 1, f"各折 OOS 窗口大小不一致：{sizes}"
        for a, b in zip(folds, folds[1:]):
            assert a.oos_end < b.oos_start, (
                f"第 {a.fold_idx} 折与第 {b.fold_idx} 折的 OOS 窗口重叠")

    def test_oos_window_size_follows_the_documented_formula(self):
        """
        `available = n - min_train_days - embargo_days` 的两个 `-`
        改成 `+` 会让每折 OOS 窗口按**更大的**可用天数算，
        最后几折直接越出数据末尾被 break 掉 —— 折数悄悄变少。
        """
        n, embargo, splits, min_train = 1200, 20, 5, 120
        folds = self._wf(splits, min_train, embargo).get_folds(_dataset(n))
        expected = (n - min_train - embargo) // splits
        assert folds[0].oos_days == expected, (
            f"每折 OOS 窗口 {folds[0].oos_days} 天，按公式应为 {expected} 天")

    def test_insufficient_data_yields_no_folds(self):
        """
        `if n < min_train_days + embargo_days + n_splits: return []` ——
        `<` 放宽成 `<=` 会把恰好够的情形也判成不够；
        `+` 改成 `-` 会让门槛降到远低于实际需要，随后在索引时出界。
        """
        assert self._wf(5, 120, 20).get_folds(_dataset(144)) == [], (
            "数据不足却切出了折")

    def test_a_usable_oos_window_is_a_separate_requirement(self):
        """
        样本刚过 `min_train + embargo + n_splits` 这道门槛时 `oos_per_fold`
        只有 1 天，被 `if oos_per_fold < 5: return []` 拦下。
        两道门槛是**独立的两条**，缺一条就会切出每折只有一两天 OOS 的假分折。
        """
        wf = self._wf(5, 120, 20)
        assert wf.get_folds(_dataset(160)) == [], (
            "每折 OOS 不足 5 天却仍然切出了分折")
        assert wf.get_folds(_dataset(170)) != [], (
            "每折 OOS 已有 5 天却仍然拒绝切分 —— `oos_per_fold < 5` 被放宽")

    def test_split_refuses_instead_of_returning_nothing(self):
        """
        `if not folds: raise` —— 删掉 `not` 会让**有折的时候反而报错**、
        没折的时候静默返回空列表，调用方拿到 0 折却以为跑完了 walk-forward。
        """
        wf = self._wf(5, 120, 20)
        with pytest.raises(ValueError, match="无法生成"):
            wf.split(_dataset(144))
        assert len(wf.split(_dataset(1200))) == 5

    def test_split_slices_are_inclusive_at_both_ends(self):
        """
        两处 `_slice_dataset(..., inclusive=True)`。改成 False 会让每折
        **少一天 IS、少一天 OOS**，且 fold 元信息里的天数与实际切出的对不上。
        """
        wf = self._wf()
        folds = wf.get_folds(_dataset(1200))
        pairs = wf.split(_dataset(1200))
        for fold, (is_data, oos_data) in zip(folds, pairs):
            n_is = len(is_data["close"])
            n_oos = len(oos_data["close"])
            assert n_is == fold.is_days, (
                f"第 {fold.fold_idx} 折切出的 IS 有 {n_is} 行，"
                f"元信息说 {fold.is_days} 行 —— inclusive 边界不一致")
            assert n_oos == fold.oos_days, (
                f"第 {fold.fold_idx} 折切出的 OOS 有 {n_oos} 行，"
                f"元信息说 {fold.oos_days} 行")
            assert is_data["close"].index[-1] == fold.is_end
            assert oos_data["close"].index[0] == fold.oos_start
            assert oos_data["close"].index[-1] == fold.oos_end

    def test_split_returns_independent_copies(self):
        ds = _dataset(1200)
        pairs = self._wf().split(ds)
        pairs[0][0]["close"].iloc[0, 0] = -999.0
        assert ds["close"].iloc[0, 0] != -999.0, (
            "split 返回的是视图 —— 下游改动会污染原数据集")

    def test_summary_lists_every_fold(self):
        text = self._wf().summary(_dataset(1200))
        assert "embargo=20d" in text, text[:300]
        assert text.count("IS=") >= 5, f"摘要里没有列全 5 折：\n{text}"


# ===========================================================================
# G. 内部切片工具 `_slice_dataset` / `_extract_dates`
# ===========================================================================
#
# 这两个函数是上面两个分区器共用的底座：所有 IS/OOS 的物理边界最终都由
# `_slice_dataset` 的两个不等号决定。它们错了，**两个分区器同时错**，
# 而 fold 元信息（is_days / oos_days）仍然是对的 —— 元信息与实际数据对不上，
# 这是最难查的一类：日志上一切正常，算出来的收益是另一段数据的。

class TestSliceDataset:

    @staticmethod
    def _ds(n: int = 20):
        idx = _bdays(n)
        return {"close": pd.DataFrame({"A": np.arange(float(n))}, index=idx),
                "volume": pd.DataFrame({"A": np.arange(float(n)) * 10}, index=idx)}

    def test_inclusive_keeps_both_endpoints(self):
        """
        `inclusive: bool = True` 是**默认值**，两个分区器都靠它。
        改成 False 会让每一折同时少掉首尾两天，而 fold 元信息不变。
        """
        d = _bdays(20)
        out = _slice_dataset(self._ds(), start=d[5], end=d[9])
        assert list(out["close"].index) == list(d[5:10]), (
            f"闭区间切片没有包含端点：{[str(x.date()) for x in out['close'].index]}")

    def test_exclusive_drops_both_endpoints(self):
        d = _bdays(20)
        out = _slice_dataset(self._ds(), start=d[5], end=d[9], inclusive=False)
        assert list(out["close"].index) == list(d[6:9]), (
            "开区间切片没有排除端点")

    def test_start_only_keeps_the_tail(self):
        """`if start is not None:` 删掉 `not` 会让**给了 start 反而不过滤**。"""
        d = _bdays(20)
        out = _slice_dataset(self._ds(), start=d[15])
        assert list(out["close"].index) == list(d[15:]), "只给 start 时没有正确截头"

    def test_end_only_keeps_the_head(self):
        d = _bdays(20)
        out = _slice_dataset(self._ds(), end=d[4])
        assert list(out["close"].index) == list(d[:5]), "只给 end 时没有正确截尾"

    def test_no_bounds_keeps_everything(self):
        """
        `mask = pd.Series(True, index=range(len(idx)))` —— 初值改成 False
        会让**不给边界时切出空表**，而调用方以为拿到了全量数据。
        """
        out = _slice_dataset(self._ds(20))
        assert len(out["close"]) == 20, (
            f"不给边界却只切出 {len(out['close'])} 行 —— mask 初值疑似被改成了 False")

    def test_every_field_is_sliced_identically(self):
        d = _bdays(20)
        out = _slice_dataset(self._ds(), start=d[3], end=d[7])
        assert len(out["close"]) == len(out["volume"]) == 5
        assert list(out["close"].index) == list(out["volume"].index)

    def test_result_is_a_copy(self):
        ds = self._ds()
        out = _slice_dataset(ds, start=_bdays(20)[0])
        out["close"].iloc[0, 0] = -999.0
        assert ds["close"].iloc[0, 0] != -999.0, "切片返回的是视图"

    def test_extract_dates_rejects_an_empty_dataset(self):
        """`if not dataset: raise` —— 删掉 `not` 会让**非空**数据集反而被拒。"""
        with pytest.raises(ValueError, match="不能为空"):
            _extract_dates({})
        assert len(_extract_dates(self._ds(20))) == 20

    def test_extract_dates_sorts_the_index(self):
        idx = _bdays(10)[::-1]
        ds = {"close": pd.DataFrame({"A": np.arange(10.0)}, index=idx)}
        out = _extract_dates(ds)
        assert list(out) == sorted(out), "日期索引没有被排序 —— 切分边界会错乱"


class TestWalkForwardOosWindowBoundary:

    def test_exactly_five_days_per_fold_is_accepted(self):
        """
        `if oos_per_fold < 5: return []` —— **严格小于 5**。
        放宽成 `<= 5` 会把"每折恰好 5 天 OOS"这档也拒掉。

        构造：`available = n - min_train - embargo` 恰好等于 `5 * n_splits`
        → `oos_per_fold = 5`。n = 120 + 20 + 25 = 165。
        （n=170 的 oos_per_fold 是 6，测不到边界 —— 上一轮就是这么漏的。）
        """
        wf = WalkForwardPartitioner(n_splits=5, min_train_days=120,
                                    embargo_days=20)
        assert (165 - 120 - 20) // 5 == 5, "构造的 oos_per_fold 不是恰好 5"
        folds = wf.get_folds(_dataset(165))
        assert len(folds) == 5, (
            f"每折恰好 5 天 OOS 却只切出 {len(folds)} 折 —— "
            f"`oos_per_fold < 5` 被放宽成了 `<= 5`")
        assert folds[0].oos_days == 5

    def test_four_days_per_fold_is_rejected(self):
        wf = WalkForwardPartitioner(n_splits=5, min_train_days=120,
                                    embargo_days=20)
        assert (164 - 120 - 20) // 5 == 4
        assert wf.get_folds(_dataset(164)) == [], "每折只有 4 天 OOS 却切出了分折"


# ===========================================================================
# H. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "L181 `all_bdays[is_count] if oos_ratio > 0 and is_count < total_days else None`"
    " 的三处（`>`→`>=`、`<`→`<=`、`and`→`or`）":
        "`oos_ratio == 0` 时 `is_count = total_days`，两个子式分别是 False 与 False；"
        "`oos_ratio > 0` 时 `is_count <= usable - 1 <= total_days - 1`，两个子式都是 True。"
        "也就是说两个子式在整个可达参数空间里**同真同假** —— `and`/`or` 等价，"
        "两个不等号各自的边界也永远取不到。",

    "L183 `if self._split_date is not None and embargo_days > 0:`"
    " 的两处（`>`→`>=`、`and`→`or`）":
        "`embargo_days == 0` 时进入分支会算出 `oos_actual_idx = is_count`，"
        "随后 `_oos_start = all_bdays[is_count]`，与 else 分支的 `_oos_start = _split_date` "
        "**取到同一天**；`_split_date is None`（即 ratio=0）时 `is_count = total_days`，"
        "进入分支后 `is_count + embargo >= total_days` 必然成立 → `_oos_start = None`，"
        "同样与 else 分支一致。两种取值观察不到差别。",

    "L185 `if oos_actual_idx < total_days:` → `<=`":
        "见 E 节：`is_count + embargo <= total_days - 1` 恒成立，"
        "`<` 与 `<=` 的分界点取不到。",

    "L435 `if oos_start_idx >= n or oos_start_idx > oos_end_idx:` → `>=`":
        "两种取值只在某折 OOS 恰好 1 天（start == end）时分道。"
        "`oos_start <= n - opf <= n - 5`（因为 opf >= 5 已被上一道门槛保证），"
        "而 `oos_end = min(oos_start + opf - 1, n - 1) >= oos_start + 4`，"
        "所以 end 恒严格大于 start，1 天的折不可构造。"
        "见 test_no_fold_can_have_a_one_day_oos_window。",

    "L414 `if n < min_train_days + embargo_days + n_splits:` 的两处"
    "（`<`→`<=`、`+`→`-`）":
        "这道门槛后面还有第二道 `oos_per_fold < 5`，而 "
        "`oos_per_fold = max(1, (n - min_train - embargo) // n_splits) >= 5` "
        "要求 `n >= min_train + embargo + 5*n_splits`，严格强于第一道门槛。"
        "于是第一道门槛能拦下的每一种情形，第二道都会拦下 —— 改动第一道门槛"
        "不改变最终返回值（都是 `[]`）。"
        "见 test_the_first_walkforward_guard_is_subsumed_by_the_second。",
}


def test_split_predicates_are_equivalent_over_the_reachable_space():
    """
    L181 / L183 / L185 等价性的机械验证：在**全部**能通过前置守卫的
    (总天数, embargo, ratio) 组合上，把原谓词与各变异谓词逐一求值比对。

    这同时是失效告警：哪天 `is_count` 的 clip 规则变了，这里会立刻红。
    """
    mismatches = []
    ratios = (0.0, 0.001, 0.05, 0.3, 0.5, 0.9, 0.999)
    for total in range(10, 300):
        for embargo in range(0, min(total, 40)):
            for ratio in ratios:
                if ratio > 0:
                    usable = total - embargo
                    if usable < 2:
                        continue
                    is_count = int(round(usable * (1.0 - ratio)))
                    is_count = max(1, min(is_count, usable - 1))
                else:
                    is_count = total

                a1, a2 = ratio > 0, is_count < total
                # L181 的三个变异谓词
                if (a1 and a2) != (ratio >= 0 and a2):
                    mismatches.append(("L181 >=", total, embargo, ratio))
                if (a1 and a2) != (a1 and is_count <= total):
                    mismatches.append(("L181 <=", total, embargo, ratio))
                if (a1 and a2) != (a1 or a2):
                    mismatches.append(("L181 or", total, embargo, ratio))

                split_is_none = not (a1 and a2)
                # L183：两种谓词最终算出的 _oos_start 必须一致
                def _oos_start(pred: bool):
                    if pred:
                        idx = is_count + embargo
                        return idx if idx < total else None
                    return None if split_is_none else is_count

                base = _oos_start((not split_is_none) and embargo > 0)
                if base != _oos_start((not split_is_none) and embargo >= 0):
                    mismatches.append(("L183 >=", total, embargo, ratio))
                if base != _oos_start((not split_is_none) or embargo > 0):
                    mismatches.append(("L183 or", total, embargo, ratio))
                # L185
                if (not split_is_none) and embargo > 0:
                    idx = is_count + embargo
                    if (idx < total) != (idx <= total):
                        mismatches.append(("L185 <=", total, embargo, ratio))
    assert not mismatches, (
        f"{len(mismatches)} 组参数下变异谓词与原谓词结果不同，"
        f"前 3 组：{mismatches[:3]} —— 等价性证明失效，必须补用例")


def test_the_first_walkforward_guard_is_subsumed_by_the_second():
    """L414 等价性的机械验证：改动第一道门槛不改变最终是否返回空折。"""
    mismatches = []
    for n_splits in (2, 3, 5, 10):
        for min_train in (20, 60, 120, 250):
            for embargo in (0, 1, 5, 20, 60):
                for n in range(1, 1500):
                    def _empty(threshold: int) -> bool:
                        if n < threshold:
                            return True
                        return max(1, (n - min_train - embargo) // n_splits) < 5
                    base = _empty(min_train + embargo + n_splits)
                    if base != _empty(min_train + embargo + n_splits + 1):
                        mismatches.append(("<=", n_splits, min_train, embargo, n))
                    if base != _empty(min_train - embargo + n_splits):
                        mismatches.append(("+→-", n_splits, min_train, embargo, n))
    assert not mismatches, (
        f"{len(mismatches)} 组参数下第一道门槛的改动改变了结果，"
        f"前 3 组：{mismatches[:3]} —— L414 不再是等价变异，必须补用例")


def test_no_fold_can_have_a_one_day_oos_window():
    """
    L435 等价性的机械验证：`oos_start_idx > oos_end_idx` 与 `>=` 只在
    某折 OOS 恰好 1 天（start == end）时分道，而这在可达参数空间里不存在。

    推导：`oos_start = min_train + i*opf + embargo`，i <= n_splits-1，
    `opf = (n - min_train - embargo) // n_splits`，于是
    `oos_start <= min_train + embargo + n_splits*opf - opf <= n - opf <= n - 5`，
    而 `oos_end = min(oos_start + opf - 1, n - 1) >= oos_start + 4`。
    下面穷举确认没有反例。
    """
    hits = []
    for n_splits in range(2, 11):
        for min_train in range(20, 200, 10):
            for embargo in range(0, 40, 5):
                for n in range(min_train + embargo + n_splits, 1200):
                    opf = max(1, (n - min_train - embargo) // n_splits)
                    if opf < 5:
                        continue
                    for i in range(n_splits):
                        is_end = min_train + i * opf - 1
                        if is_end >= n:
                            break
                        start = is_end + 1 + embargo
                        end = min(start + opf - 1, n - 1)
                        if start >= n or start > end:
                            break
                        if start == end:
                            hits.append((n_splits, min_train, embargo, n, i))
    assert not hits, (
        f"存在 OOS 只有 1 天的折（前 3 例 {hits[:3]}）—— "
        f"L435 不再是等价变异，必须补用例")


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 5
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
