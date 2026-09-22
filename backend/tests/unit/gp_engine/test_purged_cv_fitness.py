"""
test_purged_cv_fitness.py — Phase S.1：GP 适应度改用 IS 内部 purged K 折

旧口径（单段 Validate）的问题不是"错"，是**方差大且带段位偏倚**：分数取决于
最后那一段恰好是什么行情。K 折让每个样本都当过一次留出。

本文件只测 `purged_cv_sharpe` 这个**纯函数**的契约。"口径真的接进了 GP"
那部分会构造 PopulationEvolver 并跑真实回测管线，按 DEV_LESSONS §Z 属于集成层，
放在 tests/integration/test_phase_s_fitness_wiring.py。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.gp_engine.evaluation_utils import purged_cv_sharpe


# ===========================================================================
# A. purged_cv_sharpe 本身
# ===========================================================================

class TestPurgedCvSharpe:

    def test_it_reports_the_number_of_folds_it_actually_used(self):
        r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 900),
                      index=pd.bdate_range("2019-01-02", periods=900))
        out = purged_cv_sharpe(r, n_splits=5, embargo_days=20)
        assert out["n_folds"] == 5, f"5 折只用上了 {out['n_folds']} 折"

    def test_a_constant_positive_drift_gives_a_positive_mean(self):
        r = pd.Series(np.random.default_rng(1).normal(0, 0.001, 900) + 0.002,
                      index=pd.bdate_range("2019-01-02", periods=900))
        out = purged_cv_sharpe(r)
        assert out["mean"] > 0, f"恒正漂移的 CV 均值为 {out['mean']:.3f}"
        assert out["min"] > 0, "恒正漂移下仍有折为负"

    def test_a_short_series_returns_zero_folds_instead_of_raising(self):
        """
        适应度路径上抛异常 = 候选被静默丢弃（审计 #9 的老坑）。
        样本不足时必须给出 n_folds=0 让调用方自己决定怎么退。
        """
        out = purged_cv_sharpe(pd.Series(np.zeros(10)))
        assert out["n_folds"] == 0
        assert out["mean"] == 0.0

    def test_a_flat_series_produces_no_usable_folds(self):
        """零方差块无法定义夏普 —— 必须跳过，而不是记成 0 拉低均值。"""
        r = pd.Series(np.zeros(900), index=pd.bdate_range("2019-01-02", periods=900))
        assert purged_cv_sharpe(r)["n_folds"] == 0

    def test_a_non_datetime_index_is_handled_rather_than_crashing(self):
        out = purged_cv_sharpe(pd.Series(np.random.default_rng(2).normal(0, 0.01, 900)))
        assert out["n_folds"] == 5, "普通 RangeIndex 的序列被拒了"

    def test_the_annualisation_factor_is_multiplied_not_divided(self):
        """
        `mean/sd * sqrt(252)`。写成 `/` 会小 252 倍、写成 `sd * mean` 会量纲反转 ——
        两者都不改变符号，任何"均值为正"的断言都抓不住。必须比数值。
        """
        r = pd.Series(np.random.default_rng(4).normal(0.001, 0.01, 900),
                      index=pd.bdate_range("2019-01-02", periods=900))
        out = purged_cv_sharpe(r, n_splits=5, embargo_days=0)
        # 用同一套切分手算第一折，验证量级与公式
        from app.core.data_engine.data_partitioner import PurgedKFold
        folds = PurgedKFold(n_splits=5, embargo_days=0).split(pd.DatetimeIndex(r.index))
        manual = []
        for f in folds:
            block = r.loc[f.test_idx].to_numpy()
            manual.append(block.mean() / block.std(ddof=1) * np.sqrt(252))
        assert out["mean"] == pytest.approx(float(np.mean(manual)), rel=1e-9)
        assert abs(out["mean"]) > 0.3, "年化后的量级应在个位数，疑似被除了 √252"

    @pytest.mark.parametrize("n,expect_folds", [(29, 0), (30, 5)])
    def test_the_sample_floor_is_exactly_thirty(self, n, expect_folds):
        """
        `if len(s) < 30: return empty` —— 钉住**恰好 30** 这一格。
        放宽/收紧一格的后果是"29 个点也给年化夏普"或"30 个点被判样本不足"，
        两者都不会让别的断言变红。
        """
        r = pd.Series(np.random.default_rng(5).normal(0, 0.01, n),
                      index=pd.bdate_range("2019-01-02", periods=n))
        out = purged_cv_sharpe(r, n_splits=5, embargo_days=0)
        assert out["n_folds"] == expect_folds

    def test_a_fold_with_fewer_than_five_days_is_skipped(self):
        """
        `if len(idx) < 5: continue` —— 不足 5 天的留出块算出来的夏普是噪声。
        构造 5 折 × 6 天，把折数降到 4 天就该被跳过。
        """
        r = pd.Series(np.random.default_rng(6).normal(0.001, 0.01, 30),
                      index=pd.bdate_range("2019-01-02", periods=30))
        assert purged_cv_sharpe(r, n_splits=5, embargo_days=0)["n_folds"] == 5
        # 30 天切 8 折 → 每折 3~4 天 → 全部低于 5 天下限
        assert purged_cv_sharpe(r, n_splits=8, embargo_days=0)["n_folds"] == 0

    def test_a_fold_of_exactly_five_days_is_used(self):
        """
        `len(idx) < 5` 的**恰好 5** 那一格：30 天切 6 折正好每折 5 天，
        必须全部计入。收紧成 `<= 5` 会让这一格全被跳过（n_folds 掉到 0），
        而上一条用例的 0 与 5 两个观察面都看不出这一格。
        """
        r = pd.Series(np.random.default_rng(7).normal(0.001, 0.01, 30),
                      index=pd.bdate_range("2019-01-02", periods=30))
        assert purged_cv_sharpe(r, n_splits=6, embargo_days=0)["n_folds"] == 6

    def test_a_single_usable_fold_reports_zero_spread_not_nan(self):
        """
        `len(arr) > 1` 的另一侧：只有一折可用时，`np.std(ddof=1)` 对单元素
        序列给 **nan**。放宽成 `>=` 会让 std 变成 nan 并一路传到策略卡上。

        构造：把除一折之外的所有留出块做成零方差（被 sd 守卫跳过），
        只留一折有真实波动。
        """
        n = 60
        vals = np.zeros(n)
        vals[50:60] = np.random.default_rng(8).normal(0.001, 0.01, 10)
        r = pd.Series(vals, index=pd.bdate_range("2019-01-02", periods=n))
        out = purged_cv_sharpe(r, n_splits=6, embargo_days=0)
        assert out["n_folds"] == 1, f"构造失败：可用折数是 {out['n_folds']}，本用例需要恰好 1"
        assert out["std"] == 0.0, f"单折的离散度应报 0.0，实际 {out['std']!r}"
        assert not np.isnan(out["std"])

    def test_a_range_indexed_series_gives_the_same_numbers_as_a_dated_one(self):
        """
        **这条是一份等价性证明的验证用例**（对应 `if not isinstance(..., DatetimeIndex)`
        这一格）：两条分支都只把索引用于 purged 切分与位置查找，而合成出来的
        工作日索引与原索引**长度相同、顺序相同**，所以数字必须逐位一致。

        它同时是个护栏：哪天这个函数开始按**日期**做别的事（比如按年份分组），
        这条断言会立刻红，那份等价性证明也就该作废了。
        """
        vals = np.random.default_rng(9).normal(0.001, 0.01, 900)
        dated = pd.Series(vals, index=pd.bdate_range("2019-01-02", periods=900))
        plain = pd.Series(vals)
        a = purged_cv_sharpe(dated, n_splits=5, embargo_days=20)
        b = purged_cv_sharpe(plain, n_splits=5, embargo_days=20)
        assert a == b, f"同一组数值、两种索引给出不同结果：\n{a}\n{b}"

    def test_the_spread_across_folds_is_reported(self):
        """
        只给均值等于把"5 折里有一折 -2.0"藏起来。std/min 必须一起出来。
        """
        r = pd.Series(np.random.default_rng(3).normal(0, 0.01, 900),
                      index=pd.bdate_range("2019-01-02", periods=900))
        out = purged_cv_sharpe(r)
        assert out["std"] > 0, "5 折的夏普完全相同 —— 切分很可能退化了"
        assert out["min"] <= out["mean"], "min 竟然大于 mean"
