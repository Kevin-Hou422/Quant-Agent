"""
ml_engine/alpha_evaluator.py —— 过拟合评分与滚动指标的定钉测试（变异测试驱动）

来由：22 个变异点，首测击杀率 **22.7%**（存活 17）。

AlphaEvaluator 出的是"这个因子是不是过拟合了"的判断，`is_overfit` 会进
评估报告、进谱系、进人工审批时看的那张表。它判错的两个方向：
判不出过拟合 → 过拟合因子一路进 paper；乱判过拟合 → 真因子被毙掉。

存活项：
  - `score > self._threshold` —— 恰好等于阈值算不算过拟合
  - `abs(is_sharpe) < 1e-9` —— IS Sharpe 近零时的除零守卫（分母就是它）
  - `(is_sharpe - oos_sharpe) / abs(is_sharpe)` 的量纲
  - `oos_report is not None and oos_prices is not None and oos_signal is not None`
    的两个 `and` —— 放宽成 `or` 会在只给了一部分 OOS 输入时就去算 OOS 指标
  - `_rolling_sharpe` 的 `len < window`、`* np.sqrt(252)`
  - `_rolling_rank_ic` 的 `len(common_idx) < window + 5`、`px_arr[t + 1]`
    （**前瞻一期**，写成 `t - 1` 就是拿昨天的收益算今天的 IC，IC 会莫名变正）
  - `mask.sum() < 5` 两处最小样本守卫
  - `_isnan` 的两个 `return True`
  - 三个 `field(repr=False)` 的滚动序列

既有覆盖（test_phase2 / test_phase3）只跑通了 evaluate() 并检查返回结构。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.ml_engine.alpha_evaluator import (
    AlphaEvaluator,
    EvalMetrics,
    _cross_section_spearman,
    _isnan,
    _rolling_rank_ic,
    _rolling_sharpe,
)


# ===========================================================================
# A. 过拟合评分
# ===========================================================================

class TestOverfitScore:

    @staticmethod
    def _m(sharpe) -> EvalMetrics:
        return EvalMetrics(sharpe_ratio=sharpe)

    def _score(self, is_s, oos_s, threshold=0.5):
        ev = AlphaEvaluator(overfit_threshold=threshold)
        return ev._overfit_score(self._m(is_s), self._m(oos_s))

    def test_degradation_is_relative_to_the_in_sample_sharpe(self):
        """
        `degradation = (is - oos) / abs(is)`。分母改掉（或 `-` 变 `+`）会让
        同样的衰减幅度在不同 Sharpe 量级上给出完全不同的分数，
        阈值就此失去可比性。IS=2.0 / OOS=1.0 → 恰好衰减一半。
        """
        score, _ = self._score(2.0, 1.0)
        assert score == pytest.approx(0.5), f"衰减一半算出了 {score}"

    def test_score_is_clipped_into_zero_one(self):
        assert self._score(2.0, 3.0)[0] == 0.0, "OOS 比 IS 还好，分数应为 0"
        assert self._score(2.0, -10.0)[0] == 1.0, "极端衰减应当被 clip 到 1"

    def test_score_exactly_at_the_threshold_is_not_flagged(self):
        """
        `return score, score > self._threshold` —— **严格大于**。
        放宽成 `>=` 会让恰好踩线的因子被判过拟合：
        阈值的语义是"超过才算"，踩线判死等于阈值整体下移一格。
        """
        score, flagged = self._score(2.0, 1.0, threshold=0.5)
        assert score == pytest.approx(0.5)
        assert flagged is False, (
            "衰减分恰好等于阈值却被判成过拟合 —— `>` 被放宽成了 `>=`")

    def test_score_just_above_the_threshold_is_flagged(self):
        _score, flagged = self._score(2.0, 0.9, threshold=0.5)
        assert flagged is True, "衰减超过阈值却没有被判过拟合"

    def test_near_zero_in_sample_sharpe_is_not_scored(self):
        """
        `abs(is_sharpe) < 1e-9` —— 这是**除零守卫**：分母就是 abs(is_sharpe)。
        放宽成 `<=` 只差一格，但收紧/删掉会让 IS Sharpe ≈ 0 的因子
        算出 1e9 量级的衰减分，clip 之后恒为 1.0 —— 每一个"没什么表现"的
        因子都被打成"严重过拟合"，报告失去区分度。
        """
        score, flagged = self._score(1e-12, -5.0)
        assert (score, flagged) == (0.0, False), (
            f"IS Sharpe 近零时算出了 {score} —— 除零守卫失效")

    def test_in_sample_sharpe_just_above_the_guard_is_scored(self):
        """`< 1e-9` 是严格小于：恰好 1e-9 必须**被正常评分**。"""
        score, _ = self._score(1e-9, 0.0)
        assert score == pytest.approx(1.0), (
            f"IS Sharpe 恰好等于 1e-9 被守卫误杀，得分 {score}")

    def test_nan_sharpes_are_not_scored(self):
        assert self._score(np.nan, 1.0) == (0.0, False)
        assert self._score(1.0, np.nan) == (0.0, False)

    def test_missing_oos_returns_a_clean_zero(self):
        """`if oos_m is None: return 0.0, False` —— 没有 OOS 就不下结论。"""
        ev = AlphaEvaluator()
        assert ev._overfit_score(self._m(2.0), None) == (0.0, False)

    def test_the_two_branches_are_distinguishable(self):
        """两个 `return 0.0, False` 都改成别的值时这条会红。"""
        ev = AlphaEvaluator(overfit_threshold=0.5)
        no_oos = ev._overfit_score(self._m(2.0), None)
        real = ev._overfit_score(self._m(2.0), self._m(-5.0))
        assert no_oos != real, "有无 OOS 给出了同一个结果"


# ===========================================================================
# B. 滚动 Sharpe
# ===========================================================================

class TestRollingSharpe:

    @staticmethod
    def _ret(n: int, seed: int = 0) -> pd.Series:
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(0.001, 0.01, n),
                         index=pd.bdate_range("2023-01-02", periods=n))

    def test_exactly_window_many_points_produce_a_series(self):
        """
        `if len(net_returns) < window: return None` —— **严格小于**。
        放宽成 `<=` 会让数据量恰好等于窗口时返回 None，
        报告里那一栏永远空着，而看报告的人会以为"这个因子没算滚动 Sharpe"。
        """
        out = _rolling_sharpe(self._ret(60), window=60)
        assert out is not None, "点数恰好等于窗口却返回了 None"
        assert out.notna().sum() == 1, "恰好一个窗口应当只有一个有效值"

    def test_one_point_short_returns_none(self):
        assert _rolling_sharpe(self._ret(59), window=60) is None

    def test_none_input_returns_none(self):
        assert _rolling_sharpe(None, window=60) is None

    def test_sharpe_is_annualised_by_sqrt_252(self):
        """
        `* np.sqrt(252)` 改成 `/` 会让年化 Sharpe 变成日频 Sharpe 的 1/252 ——
        差 252 倍，一个 Sharpe 2.0 的因子会显示成 0.0001。
        用常数收益率 + 已知波动构造可精确核对的值。
        """
        n, w = 80, 60
        rng = np.random.default_rng(1)
        vals = rng.normal(0.002, 0.01, n)
        s = pd.Series(vals, index=pd.bdate_range("2023-01-02", periods=n))
        out = _rolling_sharpe(s, window=w)
        tail = vals[-w:]
        expected = tail.mean() / tail.std(ddof=1) * np.sqrt(252)
        assert out.iloc[-1] == pytest.approx(expected, rel=1e-9), (
            f"年化滚动 Sharpe 是 {out.iloc[-1]}，应为 {expected}")
        assert abs(out.iloc[-1]) > 1.0, (
            "构造的数据年化后量级太小，除以 sqrt(252) 也看不出差别")

    def test_zero_volatility_windows_become_nan_not_infinity(self):
        """`sig.replace(0, np.nan)` —— 常数收益窗口不得算出 inf。"""
        s = pd.Series(np.full(80, 0.001),
                      index=pd.bdate_range("2023-01-02", periods=80))
        out = _rolling_sharpe(s, window=60)
        assert out.iloc[-1] != np.inf and np.isnan(out.iloc[-1]), (
            f"零波动窗口算出了 {out.iloc[-1]}")


# ===========================================================================
# C. 滚动 Rank IC
# ===========================================================================

class TestRollingRankIc:

    @staticmethod
    def _panel(T: int = 60, N: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2023-01-02", periods=T)
        cols = [f"T{i}" for i in range(N)]
        px = pd.DataFrame(100 + rng.normal(0, 1, (T, N)).cumsum(axis=0),
                          index=idx, columns=cols)
        # 信号 = 次日收益本身 → 理论 IC 恒为 +1（用来定向）
        fwd = px.pct_change().shift(-1)
        return fwd, px

    def test_signal_equal_to_next_day_return_gives_ic_one(self):
        """
        `fwd = px_arr[t + 1]` —— **前瞻一期**。写成 `t - 1` 会拿**昨天**的收益
        去和今天的信号算相关：一个真实的动量因子会因此显示出虚高的 IC
        （自相关造成），而一个无效因子也可能"看起来有 IC"。

        构造：信号恰好等于次日收益 → 每日截面 Spearman 必须是 +1。
        """
        sig, px = self._panel()
        out = _rolling_rank_ic(sig, px, window=20)
        assert out is not None
        assert out.dropna().iloc[-1] == pytest.approx(1.0, abs=1e-9), (
            f"信号等于次日收益，滚动 IC 却是 {out.dropna().iloc[-1]} —— "
            f"前瞻期数疑似取错")

    def test_negated_signal_gives_ic_minus_one(self):
        sig, px = self._panel()
        out = _rolling_rank_ic(-sig, px, window=20)
        assert out.dropna().iloc[-1] == pytest.approx(-1.0, abs=1e-9)

    def test_exactly_enough_rows_produce_a_series(self):
        """
        `if len(common_idx) < window + 5: return None` —— 严格小于，
        且 `+ 5` 的余量不能改成 `- 5`（那会让样本更少时也放行，
        滚动均值里全是 NaN，返回一条看着有值实则空的序列）。
        """
        sig, px = self._panel(T=25)
        assert _rolling_rank_ic(sig, px, window=20) is not None, (
            "行数恰好等于 window+5 却返回了 None")
        sig2, px2 = self._panel(T=24)
        assert _rolling_rank_ic(sig2, px2, window=20) is None, (
            "行数少于 window+5 却仍然返回了序列")

    def test_cross_sections_with_too_few_names_are_skipped(self):
        """
        `if mask.sum() < 5: continue` —— **严格小于 5**。
        放宽成 `<= 5` 会让恰好 5 只有效标的的截面被跳过；
        收紧则让 2~4 只标的的截面也去算 Spearman，那里的相关系数纯属噪声
        （两只票的 Spearman 永远是 ±1）。
        """
        T, N = 40, 5
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2023-01-02", periods=T)
        cols = [f"T{i}" for i in range(N)]
        px = pd.DataFrame(100 + rng.normal(0, 1, (T, N)).cumsum(axis=0),
                          index=idx, columns=cols)
        sig = px.pct_change().shift(-1)
        out = _rolling_rank_ic(sig, px, window=20)
        assert out is not None and out.notna().any(), (
            "恰好 5 只标的的截面被整段跳过了 —— `mask.sum() < 5` 被放宽")

    def test_four_names_are_not_enough(self):
        T, N = 40, 4
        rng = np.random.default_rng(4)
        idx = pd.bdate_range("2023-01-02", periods=T)
        cols = [f"T{i}" for i in range(N)]
        px = pd.DataFrame(100 + rng.normal(0, 1, (T, N)).cumsum(axis=0),
                          index=idx, columns=cols)
        sig = px.pct_change().shift(-1)
        out = _rolling_rank_ic(sig, px, window=20)
        assert out is None or out.isna().all(), (
            "只有 4 只标的却算出了截面 IC —— 那是纯噪声")

    def test_ic_decay_helper_needs_five_valid_names_too(self):
        """
        `_cross_section_spearman` 里**另一处** `if mask.sum() < 5: continue`
        —— 与 `_rolling_rank_ic` 里那处是**两份独立的拷贝**（§S 的又一例）。
        只测其中一处，另一处照样是盲区（上一轮复测就是这么活下来的）。

        恰好 5 只有效标的必须算出 IC；只有 4 只必须跳过。
        """
        rng = np.random.default_rng(6)
        sig5 = rng.normal(0, 1, (3, 5))
        fwd5 = sig5 * 2.0 + 1.0               # 完全单调 → Spearman = +1
        out5 = _cross_section_spearman(sig5, fwd5)
        assert np.isfinite(out5).all(), (
            "恰好 5 只有效标的的截面被跳过了 —— `mask.sum() < 5` 被放宽成了 `<= 5`")
        np.testing.assert_allclose(out5, 1.0, rtol=1e-9)

        sig4 = rng.normal(0, 1, (3, 4))
        out4 = _cross_section_spearman(sig4, sig4 * 2.0)
        assert np.isnan(out4).all(), (
            "只有 4 只标的却算出了截面 IC —— 两只票的 Spearman 永远是 ±1，那是噪声")

    def test_ic_decay_helper_skips_rows_with_too_many_nans(self):
        sig = np.full((2, 8), np.nan)
        sig[0, :5] = np.arange(5.0)           # 第 0 行 5 只有效
        sig[1, :4] = np.arange(4.0)           # 第 1 行 4 只有效
        fwd = sig * 3.0
        out = _cross_section_spearman(sig, fwd)
        assert np.isfinite(out[0]), "5 只有效的那一行被跳过了"
        assert np.isnan(out[1]), "4 只有效的那一行没有被跳过"

    def test_non_overlapping_columns_yield_nothing(self):
        sig, px = self._panel()
        sig2 = sig.rename(columns=lambda c: c + "_x")
        out = _rolling_rank_ic(sig2, px, window=20)
        assert out is None or out.isna().all()


# ===========================================================================
# D. OOS 输入的三连守卫
# ===========================================================================

class TestOosGating:
    """
    `if oos_report is not None and oos_prices is not None and oos_signal is not None`
    —— 三个都必须齐。任一 `and` 放宽成 `or` 会让**只给了一部分**时也去算
    OOS 指标，`_compute_metrics(None, ...)` 随即抛属性错误，
    或者更糟：拿 IS 的价格去配 OOS 的报告，算出一份张冠李戴的 OOS 指标。
    """

    @staticmethod
    def _args():
        T, N = 40, 6
        idx = pd.bdate_range("2023-01-02", periods=T)
        cols = [f"T{i}" for i in range(N)]
        rng = np.random.default_rng(5)
        px = pd.DataFrame(100 + rng.normal(0, 1, (T, N)).cumsum(axis=0),
                          index=idx, columns=cols)
        sig = pd.DataFrame(rng.normal(0, 1, (T, N)), index=idx, columns=cols)
        # _compute_metrics 会逐个读 report 的字段，缺一个就 AttributeError。
        report = type("R", (), {
            "sharpe_ratio": 1.0, "annualized_return": 0.1, "annualized_vol": 0.15,
            "max_drawdown": -0.05, "max_dd_duration": 3,
            "mean_ic": 0.02, "ic_ir": 0.5, "ann_turnover": 0.5,
            "drawdown_series": None,
            "net_returns": px.pct_change().mean(axis=1),
        })()
        return report, px, sig

    @pytest.mark.parametrize("missing", ["report", "prices", "signal"])
    def test_partial_oos_input_yields_no_oos_metrics(self, missing):
        rep, px, sig = self._args()
        kwargs = {"oos_report": rep, "oos_prices": px, "oos_signal": sig}
        kwargs[f"oos_{missing}"] = None
        out = AlphaEvaluator().evaluate(rep, px, sig, **kwargs)
        assert out.oos_metrics is None, (
            f"只缺 oos_{missing} 却仍然算出了 OOS 指标 —— 三连守卫被放宽了")

    def test_complete_oos_input_does_produce_metrics(self):
        rep, px, sig = self._args()
        out = AlphaEvaluator().evaluate(rep, px, sig,
                                        oos_report=rep, oos_prices=px,
                                        oos_signal=sig)
        assert out.oos_metrics is not None, "三个输入齐备却没有算 OOS 指标"


# ===========================================================================
# E. _isnan 与格式化
# ===========================================================================

class TestHelpers:

    def test_isnan_recognises_missing_and_accepts_numbers(self):
        for v in (None, np.nan, "abc", object()):
            assert bool(_isnan(v)) is True, f"{v!r} 没有被判成缺失值"
        for v in (0.0, -1.5, 1e9):
            assert bool(_isnan(v)) is False, f"{v!r} 被判成了缺失值"

    def test_isnan_distinguishes_the_two_cases(self):
        assert bool(_isnan(None)) != bool(_isnan(0.0))

    def test_missing_values_are_rendered_as_na(self):
        """
        `if v is None or (isinstance(v, float) and np.isnan(v))` —— `or`/`and`
        被改之后，NaN 会被当成数字格式化成 "nan%"，报告里出现 "+nan%" 这种东西；
        更糟的是 None 走进 `v*100` 直接抛 TypeError，整份报告打不出来。
        """
        from app.core.ml_engine.alpha_evaluator import _fmt_metrics
        m = EvalMetrics(sharpe_ratio=None, annualized_return=np.nan,
                        max_drawdown=-0.1)
        text = _fmt_metrics(m)
        assert "N/A" in text, f"缺失值没有渲染成 N/A：{text}"
        assert "nan" not in text.lower(), f"报告里漏出了 nan：{text}"
        assert "-10.00%" in text, f"有效值没有被正常格式化：{text}"

    def test_rolling_series_stay_out_of_the_repr(self):
        """
        三个 `field(default=None, repr=False)` 的滚动序列。改成 True 会把
        几百上千个点塞进 `repr(EvalMetrics)` —— 而这个 repr 会进日志与报告。
        """
        idx = pd.bdate_range("2023-01-02", periods=500)
        m = EvalMetrics(sharpe_ratio=1.0,
                        rolling_sharpe_60=pd.Series(np.arange(500.0), index=idx),
                        rolling_ic_20=pd.Series(np.arange(500.0), index=idx),
                        drawdown_series=pd.Series(np.arange(500.0), index=idx))
        r = repr(m)
        for field_name in ("rolling_sharpe_60", "rolling_ic_20", "drawdown_series"):
            assert field_name not in r, f"{field_name} 进了 repr：{r[:200]}…"
        assert len(r) < 600, f"EvalMetrics 的 repr 膨胀到了 {len(r)} 字符"
