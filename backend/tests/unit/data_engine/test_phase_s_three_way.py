"""
test_phase_s_three_way.py — Phase S.1/S.2 三段切割与 purged K 折

这些用例要防的是**具体会发生的坏事**，不是"函数能跑通"：

  · 段之间没有 embargo → 滚动算子隔着切点取数，Validate 摸到 Test 的头部；
  · Test 段随入参漂移 → "换个起止日期再跑一遍"就换了一个新的样本外集，
    于是它可以被无限次重新挖掘，而每次看起来都是"第一次"；
  · 切不动时静默返回空段 → 下游拿 0 行样本算年化 Sharpe（B-5 的老坑）；
  · 冻结窗口兑现不了时把选择段钉死 → "给更多数据"在研究侧毫无变化（B-2 的老坑）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.data_partitioner import (
    PurgedKFold, ThreeWayPartitioner, partition_three_way,
)


def _panel(n: int, start: str = "2018-01-01", n_cols: int = 4) -> dict:
    idx = pd.bdate_range(start, periods=n)
    cols = [f"S{i}" for i in range(n_cols)]
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, n_cols)), axis=0),
                         index=idx, columns=cols)
    return {"close": close, "volume": pd.DataFrame(1e6, index=idx, columns=cols)}


# ===========================================================================
# A. 三段的**物理**性质：有序、不重叠、真的隔开
# ===========================================================================

class TestSegmentsArePhysicallySeparated:

    def test_the_three_segments_are_ordered_and_disjoint(self):
        s = partition_three_way(_panel(1200))
        tr = s.train["close"].index
        va = s.validate["close"].index
        te = s.test["close"].index
        assert tr[-1] < va[0] < va[-1] < te[0], (
            f"段序错乱：train 末={tr[-1]} validate=[{va[0]},{va[-1]}] test 首={te[0]}")
        assert set(tr) & set(va) == set(), "train 与 validate 有重叠样本"
        assert set(va) & set(te) == set(), "validate 与 test 有重叠样本"
        assert set(tr) & set(te) == set(), "train 与 test 有重叠样本"

    @pytest.mark.parametrize("embargo", [5, 20, 40])
    def test_every_boundary_really_skips_embargo_rows(self, embargo):
        """
        段之间必须**真的丢掉** embargo 行。只断言"日期不相等"是空真的 ——
        相邻两天也满足。这里数的是原始面板里被跳过的行数。
        """
        panel = _panel(1600)
        all_dates = list(panel["close"].index)
        pos = {d: i for i, d in enumerate(all_dates)}
        s = partition_three_way(panel, embargo_days=embargo)

        gap_is_val = pos[s.validate["close"].index[0]] - pos[s.train["close"].index[-1]] - 1
        gap_val_te = pos[s.test["close"].index[0]] - pos[s.validate["close"].index[-1]] - 1
        assert gap_is_val == embargo, f"IS→Validate 只隔了 {gap_is_val} 行，应为 {embargo}"
        assert gap_val_te == embargo, f"Validate→Test 只隔了 {gap_val_te} 行，应为 {embargo}"

    def test_the_reported_sizes_match_the_segments_that_were_actually_cut(self):
        """
        `n_is / n_val / n_test` 是**报告字段** —— 调用方（以及 RunManifest、
        前端面板）看的是它们，不是重新去数 DataFrame 的行数。

        变异测试抓到的真盲区：把 `n_val = val_hi - val_lo` 写成 `val_hi + val_lo`，
        切出来的数据完全正确、所有结构断言照样绿，只有**报出去的数字**是错的。
        没有这条用例，那种错法永远不会被发现。
        """
        for n in (450, 900, 1300):
            s = partition_three_way(_panel(n))
            assert s.n_is == len(s.train["close"]), f"n={n}: n_is 与 train 实际行数不符"
            assert s.n_val == len(s.validate["close"]), f"n={n}: n_val 与 validate 不符"
            assert s.n_test == len(s.test["close"]), f"n={n}: n_test 与 test 不符"
            assert s.n_is + s.n_val + s.n_test == n - 2 * s.embargo_days

    def test_the_serialised_form_carries_the_window_and_the_flags(self):
        """
        `to_dict()` 是这份切分唯一会被写进响应体/台账的形态。
        日期字段算错或标志丢失，在上面那些结构断言里都看不出来。
        """
        s = partition_three_way(_panel(1300))
        d = s.to_dict()
        assert d["test_start"] == str(s.test["close"].index[0].date())
        assert d["test_end"] == str(s.test["close"].index[-1].date())
        assert d["test_key"] == f"{d['test_start']}..{d['test_end']}"
        assert d["frozen_by"] == s.frozen_by and d["degraded"] == s.degraded
        assert (d["n_is"], d["n_val"], d["n_test"]) == (s.n_is, s.n_val, s.n_test)

    def test_the_selection_panel_is_a_contiguous_prefix_without_any_test_day(self):
        """
        selection 是"选择流程被允许看见的全部数据"。它必须**连续**
        （否则滚动算子跨洞取数），且**一天 Test 都不含**。
        """
        s = partition_three_way(_panel(1200))
        sel = s.selection["close"].index
        assert list(sel) == list(pd.DatetimeIndex(sel).sort_values()), "selection 未按时间有序"
        assert len(set(sel) & set(s.test["close"].index)) == 0, "selection 里混进了 Test 段的日期"
        # 连续：selection 覆盖 train ∪ embargo ∪ validate，没有洞
        assert len(sel) == len(s.train["close"]) + s.embargo_days + len(s.validate["close"]), (
            "selection 的长度与 train+embargo+validate 对不上 —— 中间有洞或多切了")


# ===========================================================================
# B. 冻结语义：窗口**不随入参漂移**（S.2 的核心承诺）
# ===========================================================================

class TestTheFrozenWindowDoesNotDrift:

    def test_extending_the_history_backwards_does_not_move_the_test_window(self):
        """
        同一个结束日、更长的历史 → Test 段**必须是同一段**。

        这是"一次性使用"能被记账的前提：test_key 变了就等于换了一本账，
        于是同一段数据可以被反复当成"没用过的样本外"。
        """
        long_panel = _panel(1600, start="2015-01-01")
        end = long_panel["close"].index[-1]
        short_panel = {k: v.loc[v.index >= end - pd.Timedelta(days=2000)]
                       for k, v in long_panel.items()}

        a = partition_three_way(long_panel)
        b = partition_three_way(short_panel)
        assert a.frozen_by == b.frozen_by == "years", (
            f"两次都应按日历冻结，实际 {a.frozen_by} / {b.frozen_by}")
        assert a.test_key == b.test_key, (
            f"历史长度一变，冻结窗口就漂了：{a.test_key} vs {b.test_key}")

    def test_an_honoured_freeze_is_not_flagged_as_degraded(self):
        """
        `degraded` 的默认值与"兑现了冻结"这条分支返回的 False 都需要对照：
        若默认或返回值被翻成 True，所有结论都会被读成"这次的样本外强度弱"，
        而只测 degraded=True 那一侧是发现不了的。
        """
        s = partition_three_way(_panel(1300))
        assert s.frozen_by == "years", f"1300 天的面板应能兑现冻结，实际 {s.frozen_by}"
        assert s.degraded is False, "兑现了日历冻结却被标成 degraded"

    def test_the_ratio_fallback_reports_the_ratio_regime_explicitly(self):
        """
        显式关掉日历冻结（test_years=None）→ 必须走比例口径并标 degraded。
        同时钉住比例的**量**：`int(round(n * test_ratio))`，写成除法会得到
        一个被下限夹住的、与 test_ratio 无关的数字。
        """
        n = 1300
        s = partition_three_way(_panel(n), test_years=None, test_ratio=0.2)
        assert s.frozen_by == "ratio" and s.degraded is True
        assert s.n_test == round(n * 0.2), (
            f"test_ratio=0.2 在 {n} 天上应切出 {round(n * 0.2)} 天，实际 {s.n_test}")

    def test_the_ratio_fallback_clamps_the_test_segment_to_leave_room_for_is(self):
        """
        比例口径下若 `n_ratio` 大到挤掉 IS，必须被夹到 `n - floor_left`
        （floor_left = IS 下限 + 两段 embargo + 1）。

        这条钉住的是那个夹子**兑现了承诺**：它存在的全部意义就是"宁可少切
        Test，也要给 IS 留够"。第一版按 `n_val=1` 估算留量，而 n_val 是按
        val_ratio 从剩余里分的 —— 于是夹完 IS 仍然不够、照样抛错。
        """
        import math

        n, emb, floor, vr = 800, 20, 60, 0.25
        s = partition_three_way(_panel(n), test_years=None, test_ratio=0.95,
                                embargo_days=emb, min_train_days=floor, val_ratio=vr)
        need = emb + math.ceil((floor + emb) / (1.0 - vr))
        assert s.n_test == n - need, (
            f"test_ratio=0.95 应被夹到 {n - need} 天，实际 {s.n_test}")
        assert s.n_is >= floor, (
            f"夹住之后 IS 仍只有 {s.n_is} 天 < {floor} —— 夹子没兑现承诺")

    def test_the_calendar_cutoff_is_strict_so_the_boundary_bar_stays_in_selection(self):
        """
        `dates > cutoff` —— **严格大于**。放宽成 `>=` 会把恰好落在 cutoff
        那一天也划进 Test，冻结窗口比声明的多一天。

        用**自然日**索引（不是交易日）才能保证 cutoff 那一天真实存在于面板里，
        否则这一格永远走不到，断言就是空真的。
        """
        n = 1400
        idx = pd.date_range("2018-01-01", periods=n, freq="D")
        rng = np.random.default_rng(0)
        panel = {"close": pd.DataFrame(rng.normal(100, 1, (n, 3)), index=idx,
                                       columns=list("ABC")),
                 "volume": pd.DataFrame(1e6, index=idx, columns=list("ABC"))}
        cutoff = idx[-1] - pd.DateOffset(days=int(round(2.0 * 365.25)))
        assert cutoff in idx, "构造失败：cutoff 那一天不在面板里，本用例是空真的"

        s = partition_three_way(panel, test_years=2.0)
        assert s.n_test == int((idx > cutoff).sum()), (
            f"Test 段 {s.n_test} 天 ≠ 严格晚于 cutoff 的 {int((idx > cutoff).sum())} 天")
        assert cutoff not in pd.DatetimeIndex(s.test["close"].index), (
            "cutoff 当天被划进了 Test —— 冻结窗口比声明的多一天")

    def test_the_freeze_is_honoured_at_the_exact_room_boundary(self):
        """
        `if 0 < n_cal <= n - keep` —— 恰好"刚好放得下"的那一格必须按真冻结处理。
        收紧成 `<` 会让这一格退化成份额封顶（degraded），放宽 `0 <` 则让
        n_cal=0 也走冻结分支。两个方向都只在这一格上有区别。
        """
        emb, keep_days = 20, 378
        keep = keep_days + emb
        # 自然日索引：n_cal 可精确预测
        for n in range(keep + 700, keep + 760):
            idx = pd.date_range("2016-01-01", periods=n, freq="D")
            cutoff = idx[-1] - pd.DateOffset(days=int(round(2.0 * 365.25)))
            if int((idx > cutoff).sum()) == n - keep:
                rng = np.random.default_rng(1)
                panel = {"close": pd.DataFrame(rng.normal(100, 1, (n, 3)), index=idx,
                                               columns=list("ABC")),
                         "volume": pd.DataFrame(1e6, index=idx, columns=list("ABC"))}
                s = partition_three_way(panel, embargo_days=emb,
                                        min_selection_days=keep_days)
                assert s.frozen_by == "years" and s.degraded is False, (
                    f"n={n} 时冻结窗口刚好放得下（n_cal == n − keep），"
                    f"却被判成 {s.frozen_by}/degraded={s.degraded}")
                return
        pytest.skip("这段区间里找不到 n_cal 恰好等于 n−keep 的尺寸")

    def test_the_ratio_fallback_is_flagged_as_degraded(self):
        """
        退化口径切出来的 Test 段会随入参漂移 —— 这件事必须出现在返回值里，
        而不是只写在日志里（调用方读的是返回值）。
        """
        s = partition_three_way(_panel(300))
        assert s.degraded is True, "短面板上没能兑现冻结窗口，却没标 degraded"
        assert s.frozen_by != "years", f"frozen_by 仍自称 years：{s.frozen_by}"

    def test_a_capped_test_window_does_drift_which_is_why_it_is_degraded(self):
        """
        反面对照：被份额封顶时，窗口**确实**会随历史长度变 ——
        证明上一条的 degraded 不是多余的标注。
        """
        a = partition_three_way(_panel(700, start="2016-01-01"))
        b = partition_three_way(_panel(900, start="2015-01-04"))
        assert a.degraded and b.degraded
        assert a.test_key != b.test_key, (
            "封顶口径下两次的 Test 段竟然一样 —— 那 degraded 标注就是错的")


# ===========================================================================
# C. 切不动的时候：报错，不要静默产出无效切分
# ===========================================================================

class TestItRefusesInsteadOfReturningGarbage:

    def test_too_short_a_panel_raises_instead_of_returning_empty_segments(self):
        with pytest.raises(ValueError, match="三段切分"):
            partition_three_way(_panel(80), min_train_days=60)

    def test_an_empty_dataset_raises(self):
        with pytest.raises(ValueError):
            partition_three_way({"close": pd.DataFrame()})

    @pytest.mark.parametrize("kwargs", [
        {"test_years": -1.0},
        # 0 必须也拒：放行的话 `_test_size` 会算出 n_cal=0 然后**静默退回比例口径**，
        # 使用者以为自己关掉了冻结（其实是换了个更弱的口径）。
        {"test_years": 0.0},
        {"test_ratio": 0.0}, {"test_ratio": 1.0},
        {"val_ratio": 0.0}, {"val_ratio": 1.0},
        {"embargo_days": -1}, {"min_train_days": 5},
        {"max_test_share": 0.0}, {"max_test_share": 1.0},
    ])
    def test_nonsense_parameters_are_rejected_at_construction(self, kwargs):
        with pytest.raises(ValueError):
            ThreeWayPartitioner(**kwargs)

    @pytest.mark.parametrize("kwargs", [
        {"embargo_days": 0},                      # 0 是合法的"不隔离"
        {"min_train_days": 20, "min_selection_days": 61},   # 恰好等于下限
        {"test_years": 0.5},                      # 只要为正就合法
        {"val_ratio": 0.5},
        {"max_test_share": 0.5},
    ])
    def test_the_boundary_values_themselves_are_accepted(self, kwargs):
        """
        **反向对照**：上一条只证明了"拒绝非法值"，没有证明"合法值不被误拒"。
        把 `< 0` 收紧成 `<= 0`、把 `< 20` 收紧成 `<= 20` 都只影响恰好踩线的那一格 ——
        只测非法侧的话，这类收紧一个都发现不了，而后果是合法配置突然起不来。
        """
        p = ThreeWayPartitioner(**kwargs)
        # 不能只靠"没抛异常"：那样把构造器整个清空也会通过（DEV_LESSONS §A）。
        # 传进去的值必须真的落到实例上。
        for name, value in kwargs.items():
            assert getattr(p, name) == value, (
                f"{name} 传了 {value!r}，实例上却是 {getattr(p, name)!r}")

    def test_a_panel_whose_is_lands_exactly_on_the_floor_is_accepted(self):
        """
        `if n_is < self.min_train_days` —— **严格小于**。收紧成 `<=` 只影响
        恰好踩线的那一格，全称断言（"被接受的都 ≥ 20"）抓不到它：被误拒的那格
        只是从"接受"挪到"抛错"，两侧断言都还成立。这里必须找到那一格。
        """
        floor, found = 60, None
        for n in range(150, 1500):
            try:
                s = partition_three_way(_panel(n), min_train_days=floor)
            except ValueError as exc:
                # 吞掉异常等于放过"被测行为其实坏了"（§A）。这里只允许
                # 尺寸不足这一种拒绝理由，别的原因必须当场炸出来。
                assert "数据不足" in str(exc), f"n={n} 抛的不是尺寸不足：{exc}"
                continue
            if s.n_is == floor:
                found = n
                break
        assert found is not None, "找不到 IS 恰好等于下限的面板尺寸，本用例无从断言"
        s = partition_three_way(_panel(found), min_train_days=floor)
        assert s.n_is == floor, (
            f"n={found} 时 IS 恰好 {floor} 天却被拒 —— `< min_train_days` "
            f"被收紧成了 `<=`")

    def test_min_selection_days_below_the_structural_floor_is_rejected(self):
        """下限比"IS 下限 + 两段 embargo"还小 → 切出来必然无效，当场拒绝。"""
        with pytest.raises(ValueError, match="min_selection_days"):
            ThreeWayPartitioner(min_train_days=60, embargo_days=20,
                                min_selection_days=100)


# ===========================================================================
# D. 单调性：更多数据 → 研究样本也更多（B-2 的老坑换了个地方复发）
# ===========================================================================

class TestMoreDataMeansMoreResearchData:

    def test_the_selection_panel_grows_with_the_panel_under_the_capped_regime(self):
        small = partition_three_way(_panel(400))
        large = partition_three_way(_panel(900))
        assert small.frozen_by == large.frozen_by == "years_capped", (
            "本用例要测的是封顶口径下的单调性，前提不成立就白测了")
        n_small = len(small.selection["close"])
        n_large = len(large.selection["close"])
        assert n_large > n_small, (
            f"面板从 400 涨到 900，选择段却是 {n_small} → {n_large} —— "
            f"多出来的数据全进了 Test，研究侧拿到的一样多")

    def test_the_test_segment_is_never_empty_whatever_the_panel_size(self):
        """
        这条同时是一份**可机械验证的前提**：下游（端点、验证门）里的
        `split.n_test > 0` 之所以恒真，靠的就是切分器在所有分支上都
        `max(1, ...)`。前提一旦破了，那些守卫就从"冗余"变成"必需"，
        而它们的分支此前从没被执行过。
        """
        for n in range(120, 1500, 37):
            try:
                s = partition_three_way(_panel(n))
            except ValueError as exc:
                assert "数据不足" in str(exc), f"n={n} 抛的不是尺寸不足：{exc}"
                continue
            assert s.n_test >= 1, f"n={n} 切出了空的 Test 段"
            assert s.n_val >= 1 and s.n_is >= 1

    def test_the_test_segment_never_exceeds_the_share_cap_when_capped(self):
        s = partition_three_way(_panel(900), max_test_share=0.35)
        n_total = 900
        assert s.n_test <= int(n_total * 0.35) + 1, (
            f"封顶失效：Test 段 {s.n_test} 超过了 {0.35:.0%} 份额上限")


# ===========================================================================
# E. PurgedKFold
# ===========================================================================

class TestPurgedKFold:

    def test_every_sample_is_held_out_exactly_once(self):
        dates = pd.bdate_range("2019-01-01", periods=600)
        folds = PurgedKFold(n_splits=5, embargo_days=10).split(dates)
        held = [d for f in folds for d in f.test_idx]
        assert len(held) == len(dates), f"留出块合计 {len(held)} 天 ≠ 全样本 {len(dates)} 天"
        assert len(set(held)) == len(held), "有样本被留出了不止一次"

    @pytest.mark.parametrize("embargo", [0, 10, 30])
    def test_training_samples_keep_their_distance_from_the_held_out_block(self, embargo):
        """
        purge 的全部意义：留出块**两侧**各 embargo 个样本不得进训练集。
        不 purge 的话，ts_mean(20) 这类算子会让训练集隔着切点摸到留出块。
        """
        dates = pd.bdate_range("2019-01-01", periods=900)
        pos = {d: i for i, d in enumerate(dates)}
        for f in PurgedKFold(n_splits=6, embargo_days=embargo).split(dates):
            lo = pos[f.test_idx[0]]
            hi = pos[f.test_idx[-1]]
            train_pos = [pos[d] for d in f.train_idx]
            too_close = [p for p in train_pos if lo - embargo <= p <= hi + embargo]
            assert not too_close, (
                f"embargo={embargo} 时仍有 {len(too_close)} 个训练样本落在留出块的"
                f"隔离带 [{lo - embargo}, {hi + embargo}] 内")

    def test_train_and_test_never_overlap(self):
        dates = pd.bdate_range("2019-01-01", periods=600)
        for f in PurgedKFold(n_splits=5, embargo_days=5).split(dates):
            assert set(f.train_idx) & set(f.test_idx) == set(), "同一折里训练集与留出块重叠"

    def test_too_few_samples_raises_rather_than_returning_degenerate_folds(self):
        with pytest.raises(ValueError, match="样本不足"):
            PurgedKFold(n_splits=5, embargo_days=20).split(
                pd.bdate_range("2020-01-01", periods=50))

    @pytest.mark.parametrize("kwargs", [{"n_splits": 1}, {"embargo_days": -1}])
    def test_nonsense_parameters_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            PurgedKFold(**kwargs)

    @pytest.mark.parametrize("kwargs", [{"n_splits": 2}, {"embargo_days": 0}])
    def test_the_boundary_values_themselves_are_accepted(self, kwargs):
        """反向对照：2 折、0 隔离都是合法配置，收紧一格就会把它们误拒。"""
        folds = PurgedKFold(**kwargs).split(pd.bdate_range("2019-01-01", periods=600))
        assert len(folds) >= 2

    def test_the_sample_floor_is_exactly_one_fold_worth_of_room(self):
        """
        `n < n_splits * (2*embargo + 2)` —— 钉住**恰好够**的那一格：
        少一天要拒、正好够要放行。把 `+2` 写成 `-2` 或把 `<` 放宽成 `<=`
        都只在这一格上有区别。
        """
        k, emb = 4, 5
        need = k * (2 * emb + 2)          # = 48
        kf = PurgedKFold(n_splits=k, embargo_days=emb)
        kf.split(pd.bdate_range("2019-01-01", periods=need))       # 恰好够：不抛
        with pytest.raises(ValueError, match="样本不足"):
            kf.split(pd.bdate_range("2019-01-01", periods=need - 1))
