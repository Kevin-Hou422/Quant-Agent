"""
backtest_engine/alpha_combiner.py —— 多因子合成

**此前零专属测试**（25 个变异点，D 档）。

把 AlphaPool 里若干条低相关因子压成一条复合信号。它坏掉的方式
全部落在"数值还在、含义变了"这一类：

  - `_ic_ir` 里 `ret[t + 1]` 的 `+1` 被去掉 → **前视泄漏**。
    用当天信号去关联当天已实现收益，IC 会漂亮得离谱，
    而回测流程的其余部分毫无察觉。这是本模块最贵的一处。
  - `max(0.0, ic)` 被去掉 → 负 IC 的因子拿到负权重，
    等于**反向下注**一条已知无效的因子。
  - `w / w_sum` 的归一化用 `np.abs(w).sum()` —— 改成 `w.sum()`
    会在正负权重相抵时把复合信号放大几个数量级。
  - `_align` 用交集 —— 改成并集会引入整片 NaN 行，
    `np.nansum` 把它们当 0，于是"没有数据的日子"变成"信号为 0"。

手法：`_ic_ir` 用**构造好的完美预测信号**（IC=1）与
**滞后一天的信号**（IC≈0）对照，让 `+1` 的有无必然显形；
权重分支逐个用参考公式比对。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.backtest_engine.alpha_combiner import AlphaCombiner, _ic_ir

IDX = pd.bdate_range("2022-01-03", periods=60)
COLS = [f"S{i}" for i in range(8)]


def _df(arr) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(arr, dtype=float), index=IDX, columns=COLS)


def _wide(n_names: int = 30, outlier: float = 1000.0) -> pd.DataFrame:
    """最后一列是离群值、其余为 0 的宽截面信号。"""
    row = np.zeros(n_names)
    row[-1] = outlier
    cols = [f"W{i}" for i in range(n_names)]
    return pd.DataFrame(np.tile(row, (len(IDX), 1)), index=IDX, columns=cols)


def _random_returns(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return _df(rng.normal(0, 0.02, (len(IDX), len(COLS))))


# ===========================================================================
# A. IC-IR —— 前视方向
# ===========================================================================

class TestIcIr:

    def test_a_signal_that_perfectly_predicts_tomorrow_scores_high(self):
        """
        `s, r = sig[t], ret[t + 1]`

        构造 `signal[t] = returns[t+1]` —— 即完美预知明天。
        此时每一天的截面 IC 都是 1，IC-IR 应当极大。
        """
        ret = _random_returns()
        sig = ret.shift(-1)              # signal[t] = ret[t+1]
        got = _ic_ir(sig.fillna(0.0), ret)
        assert got > 50.0, (
            f"完美预测明天的信号 IC-IR 只有 {got:.3f} —— "
            f"`ret[t + 1]` 的前视对齐被改了")

    def test_a_signal_that_only_knows_today_scores_near_zero(self):
        """
        对照组：`signal[t] = returns[t]`（只知道**今天**已经发生的收益）。

        如果 `ret[t + 1]` 的 `+1` 被去掉，这条信号就变成完美预测，
        IC-IR 会飙到几十 —— 与上一条恰好互换。两条一起才封死这个方向。
        """
        ret = _random_returns()
        got = _ic_ir(ret, ret)
        assert abs(got) < 1.0, (
            f"只知道当天收益的信号拿到了 IC-IR={got:.3f} —— "
            f"`ret[t+1]` 退化成了 `ret[t]`，这是前视泄漏")

    def test_a_signal_lagged_by_one_extra_day_scores_near_zero(self):
        """再往前错一天也应当没有预测力 —— 封住 `t + 2` 方向。"""
        ret = _random_returns()
        got = _ic_ir(ret.shift(1).fillna(0.0), ret)
        assert abs(got) < 1.0, f"滞后一天的信号 IC-IR={got:.3f}，不该有预测力"

    def test_a_perfectly_inverted_signal_scores_strongly_negative(self):
        """符号方向：反向完美预测必须是大负数，不能取了绝对值。"""
        ret = _random_returns()
        got = _ic_ir((-ret.shift(-1)).fillna(0.0), ret)
        assert got < -50.0, f"反向完美预测的 IC-IR 是 {got:.3f}，符号丢了"

    def test_a_cross_section_thinner_than_five_names_is_skipped(self):
        """
        `if mask.sum() < 5: continue`

        `<` 翻成 `<=` 会把恰好 5 个有效标的的截面也丢掉；
        守卫被删则会让 2 个标的的截面参与计算（秩相关恒为 ±1，纯噪声）。
        构造：全表只有 5 列有效，其余全 NaN —— 应当仍能算出结果。
        """
        rng = np.random.default_rng(3)
        ret = _df(rng.normal(0, 0.02, (len(IDX), len(COLS))))
        sig = ret.shift(-1).fillna(0.0)
        for c in COLS[5:]:
            ret[c] = np.nan
            sig[c] = np.nan
        got = _ic_ir(sig, ret)
        assert got > 5.0, (
            f"恰好 5 个有效标的的截面被整体跳过了（IC-IR={got}）—— "
            f"`mask.sum() < 5` 被翻成了 `<=`")

    def test_a_four_name_cross_section_is_dropped(self):
        """只有 4 个有效标的时，所有截面被跳过 → 样本不足 → 0.0。"""
        rng = np.random.default_rng(3)
        ret = _df(rng.normal(0, 0.02, (len(IDX), len(COLS))))
        sig = ret.shift(-1).fillna(0.0)
        for c in COLS[4:]:
            ret[c] = np.nan
            sig[c] = np.nan
        assert _ic_ir(sig, ret) == 0.0

    def test_too_few_usable_days_returns_zero(self):
        """
        `if len(ics) < 5: return 0.0`

        样本太少时 IC-IR 毫无意义。守卫被删会让 2 天的数据算出
        一个巨大的 IR（分母是两点的标准差），该因子拿走全部权重。
        """
        short_idx = pd.bdate_range("2022-01-03", periods=4)
        ret = pd.DataFrame(np.random.default_rng(1).normal(0, 0.02, (4, 8)),
                           index=short_idx, columns=COLS)
        assert _ic_ir(ret.shift(-1).fillna(0.0), ret) == 0.0

    def test_a_constant_signal_contributes_no_ic(self):
        """
        `if denom > 0:` —— 截面恒定时秩全相同、去均值后全 0、
        分母为 0。守卫被删会得到 0/0 = NaN，
        之后 `np.mean` 把整条 IC 序列污染成 NaN。
        """
        ret = _random_returns()
        flat = _df(np.ones((len(IDX), len(COLS))))
        got = _ic_ir(flat, ret)
        assert got == 0.0 or np.isfinite(got), f"恒定信号算出了 {got}"
        assert not np.isnan(got), "恒定信号让 IC-IR 变成了 NaN"

    def test_the_ir_denominator_is_epsilon_guarded(self):
        """
        `np.mean(arr) / (np.std(arr) + 1e-9)`

        `+ 1e-9` 被删会在 IC 恒定（完美或完美反向预测）时除以 0 → inf。
        上面那条"完美预测"用例正是这种情形，这里正面断言有限性。
        """
        ret = _random_returns()
        got = _ic_ir(ret.shift(-1).fillna(0.0), ret)
        assert np.isfinite(got), "IC 恒定时 IC-IR 变成了 inf —— epsilon 守卫没了"

    def test_the_row_budget_uses_the_shorter_of_signal_and_returns(self):
        """
        `T = min(sig.shape[0] - 1, ret.shape[0] - 1)`

        `min` 换成 `max` 会在两者长度不同时索引越界；
        两处 `- 1` 任一被删会让 `ret[t+1]` 在最后一行越界。
        构造：returns 比 signal 短。
        """
        ret = _random_returns()
        sig = ret.shift(-1).fillna(0.0)
        got = _ic_ir(sig, ret.iloc[:30])     # 不应当抛
        assert np.isfinite(got)

    def test_a_signal_full_of_nan_yields_zero(self):
        ret = _random_returns()
        assert _ic_ir(_df(np.full((len(IDX), len(COLS)), np.nan)), ret) == 0.0


# ===========================================================================
# B. 权重方法分派
# ===========================================================================

def _two_signals(seed=11):
    rng = np.random.default_rng(seed)
    return {"a": _df(rng.normal(0, 1, (len(IDX), len(COLS)))),
            "b": _df(rng.normal(0, 1, (len(IDX), len(COLS))))}


class TestWeightDispatch:

    def test_no_signals_gives_no_weights(self):
        assert AlphaCombiner().optimize_weights({}) == {}

    def test_equal_weight_splits_evenly_and_sums_to_one(self):
        w = AlphaCombiner().optimize_weights(_two_signals(), method="equal_weight")
        assert w == {"a": 0.5, "b": 0.5}

    def test_equal_weight_across_three_alphas(self):
        sigs = _two_signals()
        sigs["c"] = _df(np.zeros((len(IDX), len(COLS))))
        w = AlphaCombiner().optimize_weights(sigs, method="equal_weight")
        assert sum(w.values()) == pytest.approx(1.0)
        assert all(v == pytest.approx(1 / 3) for v in w.values())

    def test_ic_weighted_without_returns_falls_back_to_equal_weight(self):
        """
        `if returns is None: ... return self.optimize_weights(signals,
                                                              method="equal_weight")`

        守卫被删会让 `_ic_weights(signals, None)` 里的
        `None.reindex(...)` 直接 AttributeError。
        """
        w = AlphaCombiner().optimize_weights(_two_signals(), method="ic_weighted")
        assert w == {"a": 0.5, "b": 0.5}

    def test_the_default_method_is_ic_weighted(self):
        sigs = _two_signals()
        ret = _random_returns()
        assert AlphaCombiner().optimize_weights(sigs, ret) == \
               AlphaCombiner().optimize_weights(sigs, ret, method="ic_weighted")

    def test_an_unknown_method_raises_and_lists_the_valid_ones(self):
        """
        `raise ValueError(f"Unknown method ...")` —— 被删会让未知方法
        静默返回 None，调用方拿到 None 再去 `.get(dsl, 0.0)` 就炸在别处。
        """
        with pytest.raises(ValueError) as ei:
            AlphaCombiner().optimize_weights(_two_signals(), method="max_sharpe")
        msg = str(ei.value)
        assert "max_sharpe" in msg
        for valid in ("ic_weighted", "equal_weight", "min_variance"):
            assert valid in msg, f"报错里没有列出合法方法 {valid}"

    def test_min_variance_is_dispatched(self):
        w = AlphaCombiner().optimize_weights(_two_signals(), method="min_variance")
        assert set(w) == {"a", "b"}
        assert sum(w.values()) == pytest.approx(1.0)


# ===========================================================================
# C. IC 加权
# ===========================================================================

class TestIcWeights:

    def test_a_better_alpha_gets_more_weight(self):
        ret = _random_returns()
        good = ret.shift(-1).fillna(0.0)                       # 完美预测
        noise = _df(np.random.default_rng(9).normal(0, 1, (len(IDX), len(COLS))))
        w = AlphaCombiner().optimize_weights({"good": good, "noise": noise}, ret)
        assert w["good"] > w["noise"], f"好因子拿到的权重更少：{w}"
        assert sum(w.values()) == pytest.approx(1.0)

    def test_a_negative_ic_alpha_gets_zero_weight_not_a_negative_one(self):
        """
        `raw[dsl] = max(0.0, ic)`

        `max` 被删（或改成 `min`）会让反向因子拿到**负权重** ——
        等于反着押一条已知无效的因子；归一化之后另一条的权重
        还会超过 1，复合信号被放大。
        """
        ret = _random_returns()
        good = ret.shift(-1).fillna(0.0)
        bad = (-ret.shift(-1)).fillna(0.0)                     # IC-IR 大负数
        w = AlphaCombiner().optimize_weights({"good": good, "bad": bad}, ret)
        assert w["bad"] == pytest.approx(0.0), (
            f"负 IC 因子拿到了 {w['bad']} 的权重 —— `max(0.0, ic)` 被改了")
        assert w["good"] == pytest.approx(1.0)

    def test_weights_are_proportional_to_the_clipped_ic_ir(self):
        """逐位比对参考实现：归一化必须是 `v / total`。"""
        ret = _random_returns()
        sigs = {"a": ret.shift(-1).fillna(0.0),
                "b": ret.shift(-1).rolling(3).mean().fillna(0.0),
                "c": _df(np.random.default_rng(2).normal(0, 1,
                                                         (len(IDX), len(COLS))))}
        w = AlphaCombiner().optimize_weights(sigs, ret)

        raw = {}
        for dsl, sig in sigs.items():
            raw[dsl] = max(0.0, _ic_ir(sig, ret.reindex(index=sig.index,
                                                        columns=sig.columns)))
        total = sum(raw.values())
        for dsl in sigs:
            assert w[dsl] == pytest.approx(raw[dsl] / total, rel=1e-12), (
                f"{dsl} 的权重与 IC-IR 占比不符")

    def test_all_zero_ic_falls_back_to_equal_weight(self):
        """
        `if total < 1e-12: return {dsl: 1.0 / n ...}`

        守卫被删会让 0/0 变成 NaN 权重，复合信号整片 NaN。
        构造：两条都是反向因子 → 都被 clip 成 0。
        """
        ret = _random_returns()
        bad = (-ret.shift(-1)).fillna(0.0)
        w = AlphaCombiner().optimize_weights({"x": bad, "y": bad.copy()}, ret)
        assert w == {"x": 0.5, "y": 0.5}, f"全零 IC 没有退回等权：{w}"

    def test_returns_are_aligned_to_each_signal_before_scoring(self):
        """
        `returns.reindex(index=sig.index, columns=sig.columns)`

        reindex 被删会在信号与收益的列序不同（或列数不同）时
        把 A 股的信号与 B 股的收益配对 —— IC 变成纯噪声。
        构造：把 returns 的列顺序打乱。
        """
        ret = _random_returns()
        good = ret.shift(-1).fillna(0.0)
        shuffled = ret[list(reversed(COLS))]
        w = AlphaCombiner().optimize_weights(
            {"good": good, "noise": _df(np.zeros((len(IDX), len(COLS))))},
            shuffled)
        assert w["good"] == pytest.approx(1.0), (
            "收益列被打乱后好因子的权重没了 —— reindex 对齐被删了")


# ===========================================================================
# D. 最小方差
# ===========================================================================

class TestMinVarianceWeights:

    def test_a_single_alpha_takes_the_whole_weight(self):
        """`if n < 2: return {dsls[0]: 1.0}` —— `<` 翻成 `<=` 会让两条因子也走这里。"""
        sig = _df(np.random.default_rng(1).normal(0, 1, (len(IDX), len(COLS))))
        assert AlphaCombiner().optimize_weights({"only": sig},
                                                method="min_variance") == {"only": 1.0}

    def test_two_alphas_do_not_take_the_single_alpha_shortcut(self):
        w = AlphaCombiner().optimize_weights(_two_signals(), method="min_variance")
        assert len(w) == 2, (
            f"两条因子走进了单因子分支：{w} —— `n < 2` 被翻成了 `<=`")

    def test_the_low_variance_alpha_gets_more_weight(self):
        """
        最小方差组合的核心性质：波动小的那条拿更多权重。
        `inv_cov @ ones` 写成 `cov @ ones` 会让结论完全反过来。
        """
        rng = np.random.default_rng(4)
        n = len(IDX)
        quiet = _df(np.tile(rng.normal(0, 0.01, (n, 1)), (1, len(COLS))))
        loud = _df(np.tile(rng.normal(0, 5.0, (n, 1)), (1, len(COLS))))
        w = AlphaCombiner().optimize_weights({"quiet": quiet, "loud": loud},
                                             method="min_variance")
        assert w["quiet"] > w["loud"], (
            f"低波动因子拿到的权重更少：{w} —— 最小方差解算反了")

    def test_the_weights_are_long_only_and_sum_to_one(self):
        """`np.maximum(raw_w, 0.0)` —— 做空一条因子不在设计范围内。"""
        rng = np.random.default_rng(17)
        sigs = {f"a{i}": _df(np.tile(rng.normal(0, 1, (len(IDX), 1)),
                                     (1, len(COLS)))) for i in range(4)}
        w = AlphaCombiner().optimize_weights(sigs, method="min_variance")
        assert all(v >= 0.0 for v in w.values()), f"出现了负权重：{w}"
        assert sum(w.values()) == pytest.approx(1.0)

    def test_insufficient_rows_fall_back_to_equal_weight(self):
        """
        `if mask.sum() < n + 1: return equal weight`

        `n + 1` 的 `+1` 被去掉会让样本数恰好等于因子数时也去求协方差 ——
        此时协方差矩阵必然奇异，结果是随机的巨大权重。
        构造：3 条因子、只有 3 行非 NaN。
        """
        short = pd.bdate_range("2022-01-03", periods=3)
        sigs = {f"a{i}": pd.DataFrame(
            np.random.default_rng(i).normal(0, 1, (3, 4)),
            index=short, columns=list("WXYZ")) for i in range(3)}
        w = AlphaCombiner().optimize_weights(sigs, method="min_variance")
        assert all(v == pytest.approx(1 / 3) for v in w.values()), (
            f"样本刚好等于因子数时没有退回等权：{w}")

    def test_identical_alphas_still_produce_usable_weights(self):
        """
        完全相同的两条因子 → 协方差奇异。
        `cov + 1e-6 * np.eye(n)` 的岭正则被删会让 `inv` 抛
        （或给出天文数字的权重）。
        """
        sig = _df(np.tile(np.random.default_rng(6).normal(0, 1, (len(IDX), 1)),
                          (1, len(COLS))))
        w = AlphaCombiner().optimize_weights({"a": sig, "b": sig.copy()},
                                             method="min_variance")
        assert sum(w.values()) == pytest.approx(1.0)
        assert all(np.isfinite(v) for v in w.values()), f"奇异协方差算出了 {w}"

    def test_an_all_negative_solution_falls_back_to_equal_weight(self, monkeypatch):
        """
        `if w_sum < 1e-12: w = ones / n`
        —— 长仓约束把所有权重削成 0 时的兜底。
        守卫被删会让 `w /= 0` 产生 NaN/inf 权重。
        """
        import app.core.backtest_engine.alpha_combiner as AC
        monkeypatch.setattr(AC.np.linalg, "inv",
                            lambda m: -np.ones(m.shape))
        w = AlphaCombiner().optimize_weights(_two_signals(), method="min_variance")
        assert w == {"a": 0.5, "b": 0.5}, f"全负解没有退回等权：{w}"

    def test_a_singular_matrix_error_degrades_to_equal_weight(self, monkeypatch):
        """
        `except np.linalg.LinAlgError: w = np.ones(n) / n`
        —— 源码注释写明"调用方以为拿到的是最小方差，必须说出来"。
        这里钉住兜底值本身（等权），并确认没有把异常抛出去。
        """
        import app.core.backtest_engine.alpha_combiner as AC

        def boom(_m):
            raise np.linalg.LinAlgError("singular")

        monkeypatch.setattr(AC.np.linalg, "inv", boom)
        w = AlphaCombiner().optimize_weights(_two_signals(), method="min_variance")
        assert w == {"a": 0.5, "b": 0.5}

    def test_rows_with_any_nan_are_dropped_before_the_covariance(self):
        """
        `mask = ~np.isnan(mat).any(axis=1)` —— `any` 换成 `all`
        会让"只要不是全 NaN 就保留"，`np.cov` 随后产出 NaN 协方差。
        """
        rng = np.random.default_rng(8)
        a = _df(np.tile(rng.normal(0, 1, (len(IDX), 1)), (1, len(COLS))))
        b = _df(np.tile(rng.normal(0, 2, (len(IDX), 1)), (1, len(COLS))))
        b.iloc[5:10] = np.nan
        w = AlphaCombiner().optimize_weights({"a": a, "b": b},
                                             method="min_variance")
        assert all(np.isfinite(v) for v in w.values()), f"NaN 行污染了协方差：{w}"


# ===========================================================================
# E. 合成
# ===========================================================================

class TestCombine:

    def test_an_empty_signal_dict_raises(self):
        with pytest.raises(ValueError, match="signals dict is empty"):
            AlphaCombiner().combine({})

    def test_the_composite_is_the_weighted_sum_of_the_inputs(self):
        """
        `comp = np.nansum(mat * w[...], axis=2)` —— 逐位比对。
        `*` 翻成 `/`、`nansum` 换成 `nanmean` 都会让数值整体改变，
        而形状与 dtype 一切正常。
        """
        c = AlphaCombiner(clip_extreme=False)
        a = _df(np.full((len(IDX), len(COLS)), 1.0))
        b = _df(np.full((len(IDX), len(COLS)), 3.0))
        out = c.combine({"a": a, "b": b}, weights={"a": 0.25, "b": 0.75})
        assert np.allclose(out.values, 0.25 * 1.0 + 0.75 * 3.0, rtol=1e-12), (
            f"加权合成结果不对：{out.iloc[0, 0]}，应当是 2.5")

    def test_weights_are_normalised_by_their_absolute_sum(self):
        """
        `w_sum = np.abs(w).sum()`

        去掉 `abs` 之后，一正一负相抵会让分母趋近 0，
        复合信号被放大几个数量级（而不是报错）。
        构造：+2 与 -1.9，绝对值和 3.9，裸和只有 0.1（39 倍差别）。
        """
        c = AlphaCombiner(clip_extreme=False)
        a = _df(np.full((len(IDX), len(COLS)), 1.0))
        b = _df(np.full((len(IDX), len(COLS)), 1.0))
        out = c.combine({"a": a, "b": b}, weights={"a": 2.0, "b": -1.9})
        expected = (2.0 - 1.9) / 3.9
        assert out.iloc[0, 0] == pytest.approx(expected, rel=1e-12), (
            f"复合值是 {out.iloc[0, 0]}，应当是 {expected} —— "
            f"归一化分母没有取绝对值")

    def test_all_zero_weights_fall_back_to_equal_weight(self):
        """`if w_sum < 1e-12: w = np.ones(len(w)) / len(w)` —— 否则 0/0。"""
        c = AlphaCombiner(clip_extreme=False)
        a = _df(np.full((len(IDX), len(COLS)), 2.0))
        b = _df(np.full((len(IDX), len(COLS)), 4.0))
        out = c.combine({"a": a, "b": b}, weights={"a": 0.0, "b": 0.0})
        assert out.iloc[0, 0] == pytest.approx(3.0), (
            f"全零权重没有退回等权：{out.iloc[0, 0]}")

    def test_an_alpha_missing_from_the_weight_dict_gets_zero(self):
        """`weights.get(dsl, 0.0)` —— 默认值被改成 1.0 会让漏配的因子满权重。"""
        c = AlphaCombiner(clip_extreme=False)
        a = _df(np.full((len(IDX), len(COLS)), 1.0))
        b = _df(np.full((len(IDX), len(COLS)), 100.0))
        out = c.combine({"a": a, "b": b}, weights={"a": 1.0})
        assert out.iloc[0, 0] == pytest.approx(1.0), (
            f"未配权重的因子参与了合成：{out.iloc[0, 0]}")

    def test_weights_are_computed_when_not_supplied(self):
        """`if weights is None:` —— `not` 被删会忽略调用方给的权重。"""
        ret = _random_returns()
        good = ret.shift(-1).fillna(0.0)
        noise = _df(np.zeros((len(IDX), len(COLS))))
        out = AlphaCombiner(clip_extreme=False).combine(
            {"good": good, "noise": noise}, returns=ret, method="ic_weighted")
        assert np.isfinite(out.to_numpy()).any()

    def test_supplied_weights_are_not_recomputed(self):
        """反向：给了权重就必须用它，不能再算一遍。"""
        c = AlphaCombiner(clip_extreme=False)
        a = _df(np.full((len(IDX), len(COLS)), 1.0))
        b = _df(np.full((len(IDX), len(COLS)), 9.0))
        out = c.combine({"a": a, "b": b}, weights={"a": 1.0, "b": 0.0},
                        returns=_random_returns())
        assert out.iloc[0, 0] == pytest.approx(1.0), "给定的权重被重新算掉了"

    def test_the_result_keeps_the_aligned_index_and_columns(self):
        c = AlphaCombiner(clip_extreme=False)
        a = _df(np.ones((len(IDX), len(COLS))))
        b = _df(np.ones((len(IDX), len(COLS))))
        out = c.combine({"a": a, "b": b}, weights={"a": 0.5, "b": 0.5})
        assert list(out.index) == list(IDX)
        assert list(out.columns) == COLS

    def test_winsorisation_is_applied_when_enabled(self):
        """
        `if self.clip_extreme: aligned = [self._winsorize(s) ...]`

        守卫被删（或翻面）会让一个离群 10σ 的值原样进入复合信号，
        组合构造时那一只票会拿到极端权重。
        """
        # 截面必须够宽：一个离群值如果自己就主宰了标准差，
        # 3σ 的上界会被它自己顶上去，反而削不掉它。
        # 单个离群值时需要 N > 10 才削得动（3·√(N-1) < N-1）。
        sig = _wide(n_names=30, outlier=1000.0)
        clipped = AlphaCombiner(clip_extreme=True).combine(
            {"a": sig}, weights={"a": 1.0})
        raw = AlphaCombiner(clip_extreme=False).combine(
            {"a": sig}, weights={"a": 1.0})
        assert raw.iloc[0, -1] == pytest.approx(1000.0)
        assert clipped.iloc[0, -1] < 1000.0, (
            "开了 clip_extreme，离群值却原样通过了")

    def test_winsorisation_is_off_when_disabled(self):
        out = AlphaCombiner(clip_extreme=False).combine(
            {"a": _wide(n_names=30, outlier=1000.0)}, weights={"a": 1.0})
        assert out.iloc[0, -1] == pytest.approx(1000.0)

    def test_clip_extreme_defaults_to_on(self):
        assert AlphaCombiner().clip_extreme is True


# ===========================================================================
# F. 对齐与去极值
# ===========================================================================

class TestAlignAndWinsorize:

    def test_alignment_uses_the_intersection_of_index_and_columns(self):
        """
        `idx.intersection(s.index)` / `cols.intersection(s.columns)`

        换成 `union` 会引入整片 NaN，而 `np.nansum` 把 NaN 当 0 ——
        于是"这一天某条因子没有数据"变成"这条因子当天信号为 0"，
        复合信号被静默稀释，没有任何报错。
        """
        a = pd.DataFrame(1.0, index=IDX[:40], columns=COLS[:5])
        b = pd.DataFrame(2.0, index=IDX[20:], columns=COLS[3:])
        out = AlphaCombiner._align([a, b])
        assert list(out[0].index) == list(IDX[20:40])
        assert list(out[0].columns) == COLS[3:5]
        assert out[0].shape == out[1].shape

    def test_a_single_signal_is_returned_untouched(self):
        """`if len(signals) == 1: return list(signals)` —— 单条不需要求交集。"""
        a = pd.DataFrame(1.0, index=IDX, columns=COLS)
        out = AlphaCombiner._align([a])
        assert out[0] is a

    def test_alignment_returns_one_frame_per_input(self):
        a = pd.DataFrame(1.0, index=IDX, columns=COLS)
        b = pd.DataFrame(2.0, index=IDX, columns=COLS)
        cc = pd.DataFrame(3.0, index=IDX, columns=COLS)
        assert len(AlphaCombiner._align([a, b, cc])) == 3

    def test_winsorisation_clips_at_three_sigma_per_row(self):
        """
        `np.clip(arr, mu - k * sd, mu + k * sd)`

        两个 `k *` 任一被改（或加减号互换）都会让上下界失衡 ——
        信号被单边削掉，复合结果出现系统性偏移。
        逐位比对参考实现。
        """
        rng = np.random.default_rng(21)
        arr = rng.normal(0, 1, (20, 40))       # 截面够宽，离群值才削得动
        arr[3, 0] = 50.0
        sig = pd.DataFrame(arr, index=pd.bdate_range("2022-01-03", periods=20))
        out = AlphaCombiner._winsorize(sig)

        mu = np.nanmean(arr, axis=1, keepdims=True)
        sd = np.nanstd(arr, axis=1, keepdims=True)
        ref = np.clip(arr, mu - 3.0 * sd, mu + 3.0 * sd)
        assert np.allclose(out.to_numpy(), ref, rtol=1e-12), "去极值结果与参考实现不符"
        assert out.iloc[3, 0] < 50.0, "离群值没有被削"

    def test_the_winsorisation_band_is_symmetric_around_the_row_mean(self):
        rng = np.random.default_rng(22)
        arr = rng.normal(5.0, 1.0, (10, 12))
        sig = pd.DataFrame(arr, index=pd.bdate_range("2022-01-03", periods=10))
        out = AlphaCombiner._winsorize(sig).to_numpy()
        mu = np.nanmean(arr, axis=1)
        sd = np.nanstd(arr, axis=1)
        for i in range(len(mu)):
            assert out[i].max() <= mu[i] + 3.0 * sd[i] + 1e-12
            assert out[i].min() >= mu[i] - 3.0 * sd[i] - 1e-12

    def test_the_clip_factor_is_configurable(self):
        rng = np.random.default_rng(23)
        arr = rng.normal(0, 1, (10, 20))
        arr[0, 0] = 99.0
        sig = pd.DataFrame(arr, index=pd.bdate_range("2022-01-03", periods=10))
        tight = AlphaCombiner._winsorize(sig, k=1.0).iloc[0, 0]
        loose = AlphaCombiner._winsorize(sig, k=3.0).iloc[0, 0]
        assert tight < loose, "k 参数没有生效"

    def test_winsorisation_preserves_shape_index_and_columns(self):
        sig = pd.DataFrame(np.random.default_rng(24).normal(0, 1, (10, 5)),
                           index=pd.bdate_range("2022-01-03", periods=10),
                           columns=list("ABCDE"))
        out = AlphaCombiner._winsorize(sig)
        assert out.shape == sig.shape
        assert list(out.index) == list(sig.index)
        assert list(out.columns) == list(sig.columns)

    def test_winsorisation_ignores_nan_when_computing_the_band(self):
        """`np.nanmean` / `np.nanstd` —— 换成 mean/std 会让含 NaN 的行整行变 NaN。"""
        arr = np.array([[1.0, 2.0, np.nan, 4.0, 100.0]])
        sig = pd.DataFrame(arr, index=pd.bdate_range("2022-01-03", periods=1))
        out = AlphaCombiner._winsorize(sig).to_numpy()
        assert np.isfinite(out[0, 0]), "含 NaN 的行被整行污染了"
        assert np.isnan(out[0, 2]), "原本的 NaN 被填成了数值"


# ===========================================================================
# G. 首测存活项收口
# ===========================================================================
#
# 首测 25 点 / 击杀 64.0% / 存活 9。逐条查完，存活原因归成两类：
#
#   1. **尺度不变性把差别吸收了**。`_ic_ir` 最后算的是 mean/std ——
#      如果每一天的 IC 都被同一个常数缩放，比值原封不动。
#      而我原来的用例里**每天的有效截面宽度都一样**，
#      于是 `denom` 的算符被改（`*` → `/`）只是整体乘了个常数，
#      IC-IR 一点没变。破法：让**每天的有效标的数不同**，
#      缩放因子随 t 变化，比值立刻就变了。
#
#   2. **epsilon 守卫要精确踩线**。`< 1e-12` 翻成 `<=` 只在
#      量恰好等于 1e-12 时才有差别 —— 权重是调用方给的，
#      直接构造得到；协方差那一处则把 `np.linalg.inv` 换成桩来精确控制。


def _ref_ic_ir(sig, ret):
    """
    `_ic_ir` 的独立参考实现（照 docstring 重写一遍，不看源码算符）。

    存在的意义：`_ic_ir` 里四处可变的地方（行数预算、denom 的乘除、
    denom 的零守卫、样本量下限）单独看都能被某个用例绕过，
    合起来逐位比对才封得死。
    """
    s_, r_ = sig.to_numpy(dtype=float), ret.to_numpy(dtype=float)
    T = min(s_.shape[0] - 1, r_.shape[0] - 1)
    ics = []
    for t in range(T):
        s, r = s_[t], r_[t + 1]
        m = ~(np.isnan(s) | np.isnan(r))
        if m.sum() < 5:
            continue
        rs = np.argsort(np.argsort(s[m])).astype(float)
        rr = np.argsort(np.argsort(r[m])).astype(float)
        rs = rs - rs.mean()
        rr = rr - rr.mean()
        denom = np.sqrt((rs ** 2).sum() * (rr ** 2).sum())
        if denom > 0:
            ics.append(float(np.dot(rs, rr) / denom))
    if len(ics) < 5:
        return 0.0
    a = np.array(ics)
    return float(np.mean(a) / (np.std(a) + 1e-9))


def _ragged(seed=5, n_days=60, n_names=8):
    """
    每天缺失的标的数不同（0/1/2 轮换）的收益面板。

    这是杀掉"尺度不变性"那一类变异的关键：截面宽度恒定时，
    `denom` 被改只是乘了个常数，IC-IR 完全不动。
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    cols = [f"S{i}" for i in range(n_names)]
    ret = pd.DataFrame(rng.normal(0, 0.02, (n_days, n_names)),
                       index=idx, columns=cols)
    for t in range(n_days):
        k = t % 3
        if k:
            ret.iloc[t, :k] = np.nan
    sig = pd.DataFrame(rng.normal(0, 1, (n_days, n_names)),
                       index=idx, columns=cols)
    return sig, ret


class TestIcIrAgainstReference:

    def test_it_matches_an_independent_reference_on_a_ragged_panel(self):
        """
        **首测存活项 L64**：`np.sqrt((rs**2).sum() * (rr**2).sum())` 的
        `*` 翻成 `/`。

        截面宽度恒定时这个变异只是把每天的 IC 乘上同一个常数，
        `mean/std` 原样不动 —— 首测就是这么漏掉的。
        换成每天宽度不同的面板，实测 IC-IR 从 1e9 掉到 2.9，
        逐位比对立刻显形。
        """
        sig, ret = _ragged()
        got = _ic_ir(sig, ret)
        assert got == pytest.approx(_ref_ic_ir(sig, ret), rel=1e-12), (
            "IC-IR 与独立参考实现不符 —— denom 的乘除或守卫被改了")

    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_it_matches_the_reference_across_several_panels(self, seed):
        sig, ret = _ragged(seed=seed)
        assert _ic_ir(sig, ret) == pytest.approx(_ref_ic_ir(sig, ret), rel=1e-12)

    def test_a_ragged_panel_really_has_varying_cross_section_widths(self):
        """
        上面那组用例的前提：每天的有效标的数**确实**不同。
        构造被改成等宽之后，L64 会重新变成不可观察 ——
        这条先红，提醒补强已经失效。
        """
        _, ret = _ragged()
        widths = {int((~ret.iloc[t].isna()).sum()) for t in range(len(ret))}
        assert len(widths) >= 3, f"截面宽度只有 {widths} 这几种，区分力不足"

    def test_a_signal_shorter_than_the_returns_does_not_overrun(self):
        """
        **首测存活项 L51**：`T = min(sig.shape[0] - 1, ret.shape[0] - 1)`
        的第一个 `- 1` 翻成 `+ 1`。

        两者等长时 `min` 会挑中另一侧，差别被吃掉。
        只有**信号比收益短**时才暴露：mutant 的 T 会超出信号行数，
        `sig[T-1]` 直接 IndexError。
        """
        _, ret = _ragged(n_days=60)
        sig_short = ret.shift(-1).fillna(0.0).iloc[:30]
        got = _ic_ir(sig_short, ret)          # 不得抛
        assert got == pytest.approx(_ref_ic_ir(sig_short, ret), rel=1e-12)

    def test_a_returns_frame_shorter_than_the_signal_does_not_overrun(self):
        sig, ret = _ragged(n_days=60)
        assert _ic_ir(sig, ret.iloc[:30]) == pytest.approx(
            _ref_ic_ir(sig, ret.iloc[:30]), rel=1e-12)

    def test_exactly_five_usable_days_are_enough(self):
        """
        **首测存活项 L67**：`if len(ics) < 5: return 0.0` 翻成 `<=`。

        只在**恰好 5 天**时有差别。构造一个 6 行的面板 → T = 5 → 5 条 IC。
        原始：5 < 5 为假 → 正常算出一个非零 IC-IR；
        变异：5 <= 5 为真 → 直接返回 0.0，该因子权重归零。
        """
        idx = pd.bdate_range("2022-01-03", periods=6)
        cols = [f"S{i}" for i in range(8)]
        rng = np.random.default_rng(11)
        ret = pd.DataFrame(rng.normal(0, 0.02, (6, 8)), index=idx, columns=cols)
        sig = ret.shift(-1).fillna(0.0)

        got = _ic_ir(sig, ret)
        assert got != 0.0, (
            "恰好 5 天的样本被判为不足 —— `len(ics) < 5` 被翻成了 `<=`")
        assert got == pytest.approx(_ref_ic_ir(sig, ret), rel=1e-12)

    def test_four_usable_days_are_not_enough(self):
        """边界另一侧：5 行面板 → T = 4 → 4 条 IC → 必须返回 0.0。"""
        idx = pd.bdate_range("2022-01-03", periods=5)
        cols = [f"S{i}" for i in range(8)]
        rng = np.random.default_rng(11)
        ret = pd.DataFrame(rng.normal(0, 0.02, (5, 8)), index=idx, columns=cols)
        assert _ic_ir(ret.shift(-1).fillna(0.0), ret) == 0.0


class TestEpsilonGuards:

    def test_a_weight_sum_exactly_on_the_epsilon_is_still_normalised(self):
        """
        **首测存活项 L175**：`if w_sum < 1e-12:` 翻成 `<=`。

        权重是调用方直接给的，所以 `np.abs(w).sum()` 可以**精确**
        构造成 1e-12：`{a: 1e-12, b: 0.0}`。

        原始 `<` 为假 → 归一化成 [1, 0] → 复合值 = a 的值；
        变异 `<=` 为真 → 退回等权 [0.5, 0.5] → 复合值 = 两者均值。
        两者差一倍，用两个差别明显的常数面板区分。
        """
        c = AlphaCombiner(clip_extreme=False)
        a = _df(np.full((len(IDX), len(COLS)), 1.0))
        b = _df(np.full((len(IDX), len(COLS)), 3.0))

        w = np.array([1e-12, 0.0])
        assert np.abs(w).sum() == 1e-12, "前提失效：权重绝对值和不再精确等于 1e-12"

        out = c.combine({"a": a, "b": b}, weights={"a": 1e-12, "b": 0.0})
        assert out.iloc[0, 0] == pytest.approx(1.0), (
            f"权重和恰好等于 1e-12 时被判成全零并退回等权"
            f"（复合值 {out.iloc[0, 0]}，等权应为 2.0）—— "
            f"`w_sum < 1e-12` 被翻成了 `<=`")

    def test_a_weight_sum_just_below_the_epsilon_falls_back_to_equal(self):
        c = AlphaCombiner(clip_extreme=False)
        a = _df(np.full((len(IDX), len(COLS)), 1.0))
        b = _df(np.full((len(IDX), len(COLS)), 3.0))
        tiny = np.nextafter(1e-12, 0.0)
        out = c.combine({"a": a, "b": b}, weights={"a": tiny, "b": 0.0})
        assert out.iloc[0, 0] == pytest.approx(2.0)

    def test_an_ic_total_exactly_on_the_epsilon_is_still_normalised(self, monkeypatch):
        """
        **首测存活项 L213**：`if total < 1e-12:` 翻成 `<=`。

        `total` 是各因子 clip 后 IC-IR 之和，真实数据上撞不到 1e-12，
        所以把 `_ic_ir` 换成桩直接回放 —— 这是**唯一**能精确踩线的办法。

        原始：total=1e-12 不算零 → 权重 [1, 0]（a 独占）；
        变异：算作零 → 退回等权 [0.5, 0.5]。
        """
        import app.core.backtest_engine.alpha_combiner as AC
        seq = {"n": 0}

        def fake(sig, ret):
            seq["n"] += 1
            return 1e-12 if seq["n"] == 1 else 0.0

        monkeypatch.setattr(AC, "_ic_ir", fake)
        w = AlphaCombiner().optimize_weights(_two_signals(), _random_returns())
        assert w == {"a": 1.0, "b": 0.0}, (
            f"IC 总和恰好等于 1e-12 时退回了等权：{w} —— "
            f"`total < 1e-12` 被翻成了 `<=`")

    def test_an_ic_total_just_below_the_epsilon_falls_back_to_equal(self,
                                                                    monkeypatch):
        import app.core.backtest_engine.alpha_combiner as AC
        seq = {"n": 0}

        def fake(sig, ret):
            seq["n"] += 1
            return np.nextafter(1e-12, 0.0) if seq["n"] == 1 else 0.0

        monkeypatch.setattr(AC, "_ic_ir", fake)
        assert AlphaCombiner().optimize_weights(
            _two_signals(), _random_returns()) == {"a": 0.5, "b": 0.5}

    def test_a_min_variance_weight_sum_exactly_on_the_epsilon_is_normalised(
            self, monkeypatch):
        """
        **首测存活项 L256**：最小方差里的 `if w_sum < 1e-12:` 翻成 `<=`。

        `w_sum` 来自 `np.maximum(inv_cov @ ones, 0).sum()`，真实协方差
        撞不到 1e-12 —— 把 `np.linalg.inv` 换成桩，让 `raw_w` 精确等于
        `[5e-13, 5e-13]`，和恰好是 1e-12。

        原始：不算零 → 归一化成 [0.5, 0.5]（碰巧也是等权，所以还要
        用第三条因子把两种路径的结果分开，见下一条）。
        """
        import app.core.backtest_engine.alpha_combiner as AC
        monkeypatch.setattr(
            AC.np.linalg, "inv",
            lambda m: np.diag([5e-13, 1e-12 - 5e-13, 0.0]))

        sigs = _two_signals()
        sigs["c"] = _df(np.random.default_rng(31).normal(0, 1,
                                                         (len(IDX), len(COLS))))
        w = AlphaCombiner().optimize_weights(sigs, method="min_variance")
        assert np.abs(np.array([5e-13, 1e-12 - 5e-13, 0.0])).sum() == 1e-12, (
            "前提失效：桩给出的权重和不再精确等于 1e-12")
        assert w["c"] == pytest.approx(0.0), (
            f"权重和恰好等于 1e-12 时退回了等权：{w} —— "
            f"`w_sum < 1e-12` 被翻成了 `<=`")
        assert w["a"] == pytest.approx(0.5) and w["b"] == pytest.approx(0.5)

    def test_a_min_variance_weight_sum_below_the_epsilon_falls_back_to_equal(
            self, monkeypatch):
        import app.core.backtest_engine.alpha_combiner as AC
        monkeypatch.setattr(
            AC.np.linalg, "inv",
            lambda m: np.diag([1e-13, 1e-13, 0.0]))
        sigs = _two_signals()
        sigs["c"] = _df(np.random.default_rng(31).normal(0, 1,
                                                         (len(IDX), len(COLS))))
        w = AlphaCombiner().optimize_weights(sigs, method="min_variance")
        assert all(v == pytest.approx(1 / 3) for v in w.values()), (
            f"权重和低于 1e-12 却没有退回等权：{w}")


class TestMinVarianceAgainstReference:

    @staticmethod
    def _ref(signals, ridge=1e-6):
        """`_min_variance_weights` 的独立参考实现。"""
        dsls = list(signals)
        n = len(dsls)
        if n < 2:
            return {dsls[0]: 1.0}
        vecs = [np.nanmean(s.to_numpy(dtype=float), axis=1)
                for s in signals.values()]
        mat = np.stack(vecs, axis=1)
        mask = ~np.isnan(mat).any(axis=1)
        if mask.sum() < n + 1:
            return {d: 1.0 / n for d in dsls}
        cov = np.cov(mat[mask].T)
        ones = np.ones(n)
        raw = np.linalg.inv(cov + ridge * np.eye(n)) @ ones
        w = np.maximum(raw, 0.0)
        s = w.sum()
        w = ones / n if s < 1e-12 else w / s
        return {d: float(w[i]) for i, d in enumerate(dsls)}

    def test_it_matches_an_independent_reference(self):
        """
        **首测存活项 L252**：`np.linalg.inv(cov + 1e-6 * np.eye(n))`
        的 `+` 翻成 `-`。

        岭正则项很小，在良态协方差上对结果只有 1e-6 量级的影响 ——
        任何"只看排序""只看和为 1"的断言都察觉不到。
        逐位比对独立参考实现是唯一可靠的手段。
        """
        rng = np.random.default_rng(41)
        sigs = {f"a{i}": _df(np.tile(rng.normal(0, 0.5 + i, (len(IDX), 1)),
                                     (1, len(COLS)))) for i in range(4)}
        got = AlphaCombiner().optimize_weights(sigs, method="min_variance")
        ref = self._ref(sigs)
        for k in sigs:
            assert got[k] == pytest.approx(ref[k], rel=1e-9), (
                f"{k} 的最小方差权重与参考实现不符：{got[k]} vs {ref[k]}")

    def test_the_ridge_sign_actually_changes_the_answer_here(self):
        """
        上一条的区分力前提：把岭正则改成减号**确实**会改变权重。
        协方差如果太良态，加减 1e-6 的差别可能落在 rel=1e-9 之内，
        那条用例就白写了。这里正面确认两者可分。
        """
        rng = np.random.default_rng(41)
        sigs = {f"a{i}": _df(np.tile(rng.normal(0, 0.5 + i, (len(IDX), 1)),
                                     (1, len(COLS)))) for i in range(4)}
        plus = self._ref(sigs, ridge=1e-6)
        minus = self._ref(sigs, ridge=-1e-6)
        assert any(abs(plus[k] - minus[k]) > 1e-9 for k in sigs), (
            "岭正则加减号给出的权重差别小于 1e-9 —— "
            "上一条用例已失去对 L252 的区分力，需要换更病态的构造")

    def test_exactly_n_plus_one_rows_are_enough_for_the_covariance(self):
        """
        **首测存活项 L245**：`if mask.sum() < n + 1:` 翻成 `<=`。

        只在有效行数**恰好等于 n+1** 时有差别。
        构造：2 条因子、恰好 3 行非 NaN。
        原始：3 < 3 为假 → 走协方差求解，给出非等权的解；
        变异：3 <= 3 为真 → 直接等权。两者可分。
        """
        idx = pd.bdate_range("2022-01-03", periods=3)
        cols = list("WXYZ")
        a = pd.DataFrame(np.tile(np.array([[1.0], [2.0], [1.5]]), (1, 4)),
                         index=idx, columns=cols)
        b = pd.DataFrame(np.tile(np.array([[5.0], [1.0], [9.0]]), (1, 4)),
                         index=idx, columns=cols)
        sigs = {"a": a, "b": b}

        mat = np.stack([np.nanmean(s.to_numpy(float), axis=1)
                        for s in sigs.values()], axis=1)
        assert (~np.isnan(mat).any(axis=1)).sum() == 3, "前提失效：有效行数不是 3"

        w = AlphaCombiner().optimize_weights(sigs, method="min_variance")
        assert w != {"a": 0.5, "b": 0.5}, (
            f"有效行数恰好 n+1 时退回了等权：{w} —— "
            f"`mask.sum() < n + 1` 被翻成了 `<=`")
        ref = self._ref(sigs)
        for k in sigs:
            assert w[k] == pytest.approx(ref[k], rel=1e-9)

    def test_n_rows_are_not_enough(self):
        """边界另一侧：2 条因子、只有 2 行有效 → 必须退回等权。"""
        idx = pd.bdate_range("2022-01-03", periods=2)
        cols = list("WXYZ")
        a = pd.DataFrame(np.tile(np.array([[1.0], [2.0]]), (1, 4)),
                         index=idx, columns=cols)
        b = pd.DataFrame(np.tile(np.array([[5.0], [1.0]]), (1, 4)),
                         index=idx, columns=cols)
        assert AlphaCombiner().optimize_weights(
            {"a": a, "b": b}, method="min_variance") == {"a": 0.5, "b": 0.5}


# ===========================================================================
# H. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "L65 `if denom > 0:` 的 `>` → `>=`":
        "`denom = sqrt((rs**2).sum() * (rr**2).sum())`，而 rs / rr 都是 "
        "`argsort(argsort(x))` 的产物 —— 无论输入有没有并列，"
        "它给出的**永远是 0..m-1 的一个排列**（序数名次，不做并列平均；"
        "这正是已登记缺陷 C-1 的成因）。"
        "去均值之后平方和恒等于 m(m²-1)/12，只要 m ≥ 2 就严格为正。"
        "而进入这一行之前已有 `if mask.sum() < 5: continue` 挡着，"
        "所以 m ≥ 5，`denom` 恒 > 0 —— `>` 与 `>=` 对一切可达输入判定相同。"
        "机械验证见 test_the_rank_denominator_is_always_strictly_positive。",
}


def test_the_rank_denominator_is_always_strictly_positive():
    """
    L65 等价性的机械验证：对 m = 5..40 的**任意**输入（含全并列、
    含极端重复），`argsort(argsort(x))` 去均值后的平方和都严格为正，
    且等于闭式 m(m²-1)/12。

    只要 `_ic_ir` 改用真正的平均名次（修掉缺陷 C-1），
    全并列的截面就会让平方和变成 0，这条立刻变红 ——
    上面那份等价性证明也就同时作废。
    """
    rng = np.random.default_rng(2026)
    for m in range(5, 41):
        cases = [
            np.arange(m, dtype=float),            # 严格递增
            np.full(m, 7.0),                      # 全部并列
            np.repeat(np.arange(m, dtype=float), 3)[:m],   # 三三并列
            rng.normal(size=m),                   # 随机
        ]
        for x in cases:
            assert len(x) == m, f"构造出的截面长度 {len(x)} != {m}"
            r = np.argsort(np.argsort(x)).astype(float)
            r = r - r.mean()
            ss = float((r ** 2).sum())
            assert ss > 0.0, (
                f"m={m} 的输入 {x[:6]}... 让秩平方和变成了 0 —— "
                f"L65 的等价性证明作废")
            assert ss == pytest.approx(m * (m * m - 1) / 12.0, rel=1e-12), (
                f"m={m} 的秩平方和 {ss} 不等于闭式 m(m²-1)/12 —— "
                f"argsort(argsort(...)) 的语义变了，证明需要重做")


def test_the_five_name_guard_is_what_makes_the_denominator_safe():
    """
    等价性论证的另一半前提：`denom` 那一行**之前**确实有
    `mask.sum() < 5` 的守卫挡着。守卫被挪走或放宽到 m < 2，
    上面的"恒为正"就不再成立。
    """
    import ast
    import inspect

    import app.core.backtest_engine.alpha_combiner as AC

    src = inspect.getsource(AC._ic_ir)
    tree = ast.parse(src)
    guards = [n for n in ast.walk(tree)
              if isinstance(n, ast.Compare)
              and isinstance(n.comparators[0], ast.Constant)
              and n.comparators[0].value == 5]
    assert guards, (
        "_ic_ir 里找不到 `< 5` 的截面宽度守卫 —— L65 的等价性证明作废")


def test_every_survivor_has_a_written_proof():
    """
    首测 25 点 / 存活 9：其中 8 处已由 G 节的用例杀死，
    剩 L65 一处为等价变异。
    """
    assert len(PROVEN_EQUIVALENT) == 1
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
