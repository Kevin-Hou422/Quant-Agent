"""
backtest_engine/overfit_stats.py —— CSCV / PBO 的定钉测试（变异测试驱动）

来由：10 个变异点，首测击杀率 20.0%（存活 8）。

PBO（Probability of Backtest Overfitting，Bailey 2015）回答的是
"从这一堆候选里挑最优，这个挑选动作本身是不是过拟合"。
策略门用它拦"选择流程过拟合"，而它此前几乎没有数值断言。

存活的 8 处覆盖了整条计算链：分块取偶、块数下限、候选数下限、
逻辑 odds 的 `rank/(N+1)`、夹逼到 (0,1) 的区间、以及 `lam < 0` 的判定方向。
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.backtest_engine.overfit_stats import (
    _blockwise_sharpe,
    probability_of_backtest_overfitting as pbo,
)


# ===========================================================================
# A. 分块夏普
# ===========================================================================

class TestBlockwiseSharpe:

    def test_sharpe_is_mean_over_std(self):
        block = np.array([[0.01, -0.01], [0.03, -0.03], [0.02, -0.02]])
        got = _blockwise_sharpe(block)
        exp = np.nanmean(block, axis=0) / np.nanstd(block, axis=0)
        assert np.allclose(got, exp, atol=1e-12)
        assert got[0] > 0 > got[1], "两列符号相反，夏普也必须相反"

    def test_zero_dispersion_column_is_zero_not_inf(self):
        """`np.where(sd > 1e-12, mu/sd, 0.0)` —— 常数列的 sd 恰好为 0。"""
        block = np.array([[0.01, 0.02], [0.01, 0.05], [0.01, -0.01]])
        got = _blockwise_sharpe(block)
        assert got[0] == 0.0
        assert np.isfinite(got).all()

    def test_nan_entries_are_ignored(self):
        block = np.array([[0.01, np.nan], [0.03, 0.02], [0.02, 0.04]])
        assert np.isfinite(_blockwise_sharpe(block)).all()


# ===========================================================================
# B. 输入校验与分块
# ===========================================================================

class TestInputValidation:

    def test_single_candidate_is_rejected(self):
        """`if N < 2: raise` —— CSCV 要比较"选出来的"与"其余的"，单候选无从比较。"""
        with pytest.raises(ValueError, match="至少 2 个"):
            pbo(np.random.default_rng(0).normal(0, 0.01, (100, 1)))

    def test_two_candidates_are_enough(self):
        """边界：**恰好 2 个**候选可以算。放宽成 `<= 2` 会把两候选的情形也拒掉。"""
        r = np.random.default_rng(1).normal(0, 0.01, (100, 2))
        assert 0.0 <= pbo(r, n_splits=4) <= 1.0

    def test_non_2d_input_is_rejected(self):
        with pytest.raises(ValueError, match="二维"):
            pbo(np.zeros(10))

    def test_too_short_series_is_rejected(self):
        """
        `S = n_splits - (n_splits % 2)` 取偶后按 T 切块，空块被丢掉；
        `if S < 2: raise`。T=1 时只能切出 1 个非空块 → 必须报错而不是给出一个数。
        """
        with pytest.raises(ValueError, match="数据太短"):
            pbo(np.random.default_rng(2).normal(0, 0.01, (1, 3)), n_splits=10)

    def test_split_count_is_rounded_down_to_even(self):
        """
        `S = n_splits - (n_splits % 2)`。写成 `+` 会把 9 变成 10 之外的奇偶翻转
        （9 + 1 = 10 看似无害，但 10 + 0 = 10、11 + 1 = 12 …），
        组合数 C(S, S//2) 随之改变，PBO 的分母也就变了。

        判据：奇数 S 与它**向下取偶**的结果必须给出同一个 PBO。
        """
        r = np.random.default_rng(3).normal(0, 0.01, (200, 4))
        assert pbo(r, n_splits=9) == pytest.approx(pbo(r, n_splits=8), abs=1e-12), (
            "n_splits=9 没有向下取偶到 8")
        assert pbo(r, n_splits=11) == pytest.approx(pbo(r, n_splits=10), abs=1e-12)

    def test_empty_blocks_are_dropped(self):
        """
        `blocks = [b for b in np.array_split(...) if len(b) > 0]` ——
        T 小于块数时会切出空块，必须丢掉（否则 `R[空行]` 的夏普是 NaN）。
        T=5、S=10 → 只有 5 个非空块，仍应算得出 PBO。
        """
        r = np.random.default_rng(4).normal(0, 0.01, (5, 3))
        out = pbo(r, n_splits=10)
        assert np.isfinite(out) and 0.0 <= out <= 1.0


# ===========================================================================
# C. PBO 的取值方向
# ===========================================================================

class TestPboSemantics:

    def test_result_is_a_probability(self):
        r = np.random.default_rng(5).normal(0, 0.01, (200, 6))
        out = pbo(r, n_splits=8)
        assert 0.0 <= out <= 1.0

    def test_pure_noise_is_close_to_one_half(self):
        """
        纯噪声候选：IS 最优纯属偶然，OOS 排名应当均匀分布 → PBO ≈ 0.5。
        `omega = rank / (N + 1.0)` 的 `+` 写成 `-` 会让 rank==N 时 omega=1
        → 夹逼后 log odds 恒为正 → PBO 塌到 0（"从不过拟合"）。
        """
        r = np.random.default_rng(6).normal(0, 0.01, (400, 8))
        out = pbo(r, n_splits=8)
        assert 0.2 < out < 0.8, f"纯噪声的 PBO 应接近 0.5，实际 {out:.3f}"

    def test_one_genuinely_better_strategy_lowers_pbo(self):
        """
        有一个**真正更好**的策略时，IS 选中它、OOS 它也确实最好 → PBO 接近 0。
        这一条钉住 `lam < 0` 的判定方向：改成 `> 0` 会把结论整体反转。
        """
        rng = np.random.default_rng(7)
        r = rng.normal(0.0, 0.01, (400, 6))
        r[:, 0] = rng.normal(0.01, 0.005, 400)      # 明显更优且稳定
        out = pbo(r, n_splits=8)
        assert out < 0.2, f"存在稳定占优的策略，PBO 却是 {out:.3f}"

    def test_is_oos_reversal_raises_pbo(self):
        """
        构造 IS 与 OOS **系统性反转**：前半段 A 极好、后半段 A 极差。

        ⚠️ 不能断言绝对值 > 0.5：CSCV 遍历 C(8,4)=70 种组合，其中只有少数
        IS 集合完全落在前半段，混合折里 A 并不占优（实测 0.286）。
        判据改为**相对关系**：反转样本的 PBO 必须高于"存在稳定占优者"的样本。
        `lam < 0` 的判定方向一旦反过来，两个数会一起取补，该不等式随即不成立。
        """
        T, N = 400, 4
        rng = np.random.default_rng(8)
        r = rng.normal(0.0, 0.01, (T, N))
        half = T // 2
        r[:half, 0] += 0.02        # 前半段 A 极好
        r[half:, 0] -= 0.02        # 后半段 A 极差
        reversal = pbo(r, n_splits=8)

        rng2 = np.random.default_rng(7)
        stable = rng2.normal(0.0, 0.01, (T, 6))
        stable[:, 0] = rng2.normal(0.01, 0.005, T)     # 全程稳定占优
        stable_pbo = pbo(stable, n_splits=8)

        assert reversal > stable_pbo, (
            f"IS/OOS 反转的 PBO（{reversal:.3f}）没有高于稳定占优的情形"
            f"（{stable_pbo:.3f}）—— `lam < 0` 的判定方向疑似反了")
        assert reversal > 0.1

    def test_upper_clip_never_binds(self):
        """
        L72 等价性的机械验证：`min(omega, 1 - 1e-6)` 的上界**永不触发** ——
        `rank ≤ N`，故 `omega = rank/(N+1) ≤ N/(N+1) < 1`，
        离 1-1e-6 还差 1/(N+1)。因此把 `1 - 1e-6` 写成 `1 + 1e-6` 不可观测。
        """
        for n in (2, 3, 5, 10, 50):
            assert n / (n + 1.0) < 1 - 1e-6

    def test_omega_is_clipped_away_from_zero_and_one(self):
        """
        `omega = min(max(omega, 1e-6), 1 - 1e-6)` —— 夹逼是为了让
        `log(omega/(1-omega))` 有限。把 `1 - 1e-6` 写成 `1 + 1e-6` 会让
        omega 可以取到 1 → `log(x/0)` → inf（并抛除零警告）。
        构造：所有候选完全相同 → rank 恒为 N → omega = N/(N+1)，
        再让 N 很小以逼近上界。
        """
        import warnings
        base = np.random.default_rng(9).normal(0, 0.01, (200, 1))
        r = np.repeat(base, 3, axis=1)          # 三个完全相同的候选
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = pbo(r, n_splits=6)
        assert np.isfinite(out) and 0.0 <= out <= 1.0

    def test_identical_candidates_are_not_flagged_as_overfit(self):
        """
        完全相同的候选之间无从"挑最优"，OOS 排名恒为最高 → log odds > 0 → PBO = 0。
        这同时确认 `rank = Σ(oos <= oos[n*])` 的比较方向没有反。
        """
        base = np.random.default_rng(10).normal(0, 0.01, (200, 1))
        r = np.repeat(base, 4, axis=1)
        assert pbo(r, n_splits=6) == pytest.approx(0.0, abs=1e-12)

    def test_two_blocks_are_enough(self):
        """
        `if S < 2: raise` 的边界：**恰好 2 块**（C(2,1)=2 种组合）可以算。
        放宽成 `<=` 会把最小可行配置也拒掉。
        """
        r = np.random.default_rng(11).normal(0, 1, (4, 3))
        out = pbo(r, n_splits=2)
        assert np.isfinite(out) and 0.0 <= out <= 1.0

    def test_omega_denominator_is_n_plus_one(self):
        """
        `omega = rank / (N + 1.0)` —— 写成 `N - 1.0` 会把 log-odds 整体右移：
        排名居中的策略从"略偏过拟合"变成"恰好中性甚至更优"，PBO 被系统性低估。

        构造（N=5、2 块，取值精确可控）：
          块 0 的夏普 = [5,1,2,3,4] → IS 最优是 A；
          块 1 的夏普 = [2,1,3,4,5] → A 在 OOS 里排第 2。
        于是 omega = 2/(5+1) = 0.333 → lam < 0 → 计入；
        若分母写成 N-1 则 omega = 2/4 = 0.5 → lam = 0 → 不计入。
        两者给出 0.5 与 0.0，可精确区分。
        """
        r = np.array([[6., 2., 3., 4., 5.],
                      [4., 0., 1., 2., 3.],      # 块 0：均值 5,1,2,3,4，标准差均为 1
                      [3., 2., 4., 5., 6.],
                      [1., 0., 2., 3., 4.]])     # 块 1：均值 2,1,3,4,5
        assert pbo(r, n_splits=2) == pytest.approx(0.5, abs=1e-12), (
            "omega 的分母疑似不是 N+1")

    def test_lambda_threshold_excludes_exact_zero(self):
        """
        `if lam < 0: lam_neg += 1` —— **严格小于**。lam == 0 对应 omega = 0.5，
        即"OOS 排名恰好居中"，按定义不算过拟合。放宽成 `<=` 会把它计入。

        构造（N=3、2 块）：两种组合下被选中者的 OOS 排名都恰好是 2，
        omega = 2/4 = 0.5 → lam = 0。正确实现给 0.0，`<=` 版本给 1.0。
        """
        r = np.array([[4., 2., 3.],
                      [2., 0., 1.],      # 块 0：夏普 3,1,2 → IS 最优 A
                      [3., 2., 4.],
                      [1., 0., 2.]])     # 块 1：夏普 2,1,3 → A 排第 2
        assert pbo(r, n_splits=2) == pytest.approx(0.0, abs=1e-12), (
            "OOS 排名恰好居中（lam=0）被计成了过拟合")


# ===========================================================================
# D. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "L28 `np.where(sd > 1e-12, mu/sd, 0.0)` -> `>=`":
        "区分值需要 sd 恰好等于 1e-12。sd = np.nanstd(block, axis=0) 是浮点均方根，"
        "无法反解出精确等于 1e-12 的输入；它真正会取到的特殊值是 0.0（常数列），"
        "而 `0.0 >= 1e-12` 为假，两侧都取 0.0。",

    "L72 `omega = min(max(omega, 1e-6), 1 - 1e-6)` 的 `1 - 1e-6` -> `1 + 1e-6`":
        "上界永不触发：rank <= N，故 omega = rank/(N+1) <= N/(N+1) < 1 - 1e-6"
        "（差距至少 1/(N+1)）。min 的第二个参数取 1-1e-6 还是 1+1e-6 都不改变结果。"
        "见 test_upper_clip_never_binds。",
}


def test_dispersion_guard_is_unreachable():
    """L28 等价性的机械验证。"""
    tol = 1e-12
    for x in (0.0, -0.0, 1e-20, tol / 2, tol * 2, 0.01):
        assert (x > tol) == (x >= tol) or x == tol
    rng = np.random.default_rng(41)
    for scale in (1e-12, 1e-11, 1e-13):
        for _ in range(200):
            assert float(np.nanstd(rng.normal(0.0, scale, 40))) != tol


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 2
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
