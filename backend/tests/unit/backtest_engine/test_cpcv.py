"""
test_cpcv.py — Phase S.3 收尾：组合式 purged 交叉验证

CSCV（已有）与 CPCV（本轮）算的是同一个统计量，差别只在**块之间有没有 purge**。
这个差别不是学术洁癖：日频面板上滚动算子会让 IS 块的末尾和 OOS 块的开头共享
原始 bar，于是"样本外"块的头部其实被样本内摸到过，PBO 被系统性**低估** ——
也就是让选择流程看起来比实际更不过拟合。

所以这里的断言分两类：
  · 结构类 —— purge 真的发生了（训练集与留出块之间隔着 embargo）；
  · 行为类 —— 对**已知会过拟合**的构造，PBO 必须高；对独立噪声，PBO 在 0.5 附近。
"""
from __future__ import annotations

from math import comb

import numpy as np
import pytest

from app.core.backtest_engine.overfit_stats import (
    combinatorial_purged_splits, cpcv_path_sharpes, cpcv_pbo,
    probability_of_backtest_overfitting,
)


# ===========================================================================
# A. 组合与 purge 的结构性质
# ===========================================================================

class TestCombinatorialSplits:

    @pytest.mark.parametrize("n_groups,k", [(6, 2), (6, 1), (8, 2), (5, 3)])
    def test_the_number_of_combinations_is_n_choose_k(self, n_groups, k):
        splits = combinatorial_purged_splits(1200, n_groups=n_groups, k=k, embargo=5)
        assert len(splits) == comb(n_groups, k), (
            f"C({n_groups},{k}) 应有 {comb(n_groups, k)} 种组合，实得 {len(splits)}")

    @pytest.mark.parametrize("embargo", [0, 10, 25])
    def test_training_positions_keep_their_distance_from_every_test_block(self, embargo):
        """
        每个留出**块**的两侧各 embargo 个样本都不得进训练集。
        k=2 时留出是两段不相邻的块，两段都要各自被隔开 —— 只检查整体
        min/max 会漏掉中间那段的边界。
        """
        n = 1200
        for train_pos, test_pos in combinatorial_purged_splits(
                n, n_groups=6, k=2, embargo=embargo):
            # 把 test_pos 拆成连续块
            blocks, start = [], test_pos[0]
            for a, b in zip(test_pos, test_pos[1:]):
                if b != a + 1:
                    blocks.append((start, a))
                    start = b
            blocks.append((start, test_pos[-1]))

            for lo, hi in blocks:
                bad = train_pos[(train_pos >= lo - embargo) & (train_pos <= hi + embargo)]
                assert len(bad) == 0, (
                    f"embargo={embargo} 时有 {len(bad)} 个训练样本落在留出块 "
                    f"[{lo},{hi}] 的隔离带里")

    def test_train_and_test_never_overlap(self):
        for train_pos, test_pos in combinatorial_purged_splits(600, 6, 2, 10):
            assert not (set(train_pos.tolist()) & set(test_pos.tolist())), \
                "训练位置与留出位置重叠"

    def test_zero_embargo_reduces_to_plain_combinatorial_cv(self):
        """embargo=0 → 训练集恰好是留出块的补集（用来确认 purge 是唯一的差别）。"""
        n = 600
        for train_pos, test_pos in combinatorial_purged_splits(n, 6, 2, 0):
            assert len(train_pos) + len(test_pos) == n, (
                "embargo=0 时训练+留出应覆盖全样本，说明没有多切掉东西")

    #: (非法入参, 期望报错里点名的东西)。**必须校对消息**：两道守卫的判据区间
    #  有重叠（`not (1 <= k < n_groups)` 与 `k >= n_groups`），只断言
    #  "抛 ValueError" 的话，其中一道被放宽后另一道会兜住，测试照样绿。
    @pytest.mark.parametrize("kwargs,msg", [
        ({"n_groups": 1}, "n_groups 至少为 2"),
        ({"k": 0}, r"k 必须在 \[1, n_groups\)"),
        # k == n_groups 必须由**前一道**守卫拦下（消息里说的是区间），
        # 而不是漏到后面那道"有效分组只有 N 组"去 —— 两者的区间一重叠，
        # 只断言"抛错"就分不出是哪道在起作用。
        ({"k": 6}, r"k 必须在 \[1, n_groups\)"),
        ({"embargo": -1}, "embargo 不能为负"),
    ])
    def test_nonsense_parameters_are_rejected(self, kwargs, msg):
        base = dict(n_obs=600, n_groups=6, k=2, embargo=5)
        base.update(kwargs)
        with pytest.raises(ValueError, match=msg):
            combinatorial_purged_splits(**base)

    @pytest.mark.parametrize("kwargs", [
        {"n_groups": 2, "k": 1}, {"k": 1}, {"k": 5}, {"embargo": 0},
    ])
    def test_the_boundary_values_themselves_are_accepted(self, kwargs):
        """
        反向对照：2 组、k=1、k=n_groups−1、embargo=0 都是合法配置。
        只测非法侧的话，把 `< 2` 收紧成 `<= 2`、把 `k < n_groups` 收紧成 `<=`
        这类改动一个都发现不了，而后果是合法配置直接起不来。
        """
        base = dict(n_obs=600, n_groups=6, k=2, embargo=5)
        base.update(kwargs)
        assert combinatorial_purged_splits(**base)

    def test_more_samples_than_groups_is_required(self):
        with pytest.raises(ValueError, match="少于分组数"):
            combinatorial_purged_splits(4, n_groups=6, k=2, embargo=0)

    def test_exactly_as_many_samples_as_groups_is_allowed(self):
        """`n_obs < n_groups` 的**恰好相等**那一格：每组一个样本，仍然可切。"""
        assert len(combinatorial_purged_splits(6, n_groups=6, k=2, embargo=0)) == comb(6, 2)

    def test_two_candidates_are_enough_for_pbo(self):
        """`if N < 2: raise` 的边界：恰好 2 个候选可以算，不该被拒。"""
        rng = np.random.default_rng(9)
        val = cpcv_pbo(rng.normal(0, 0.01, (900, 2)), n_groups=6, k=2, embargo=5)
        assert 0.0 <= val <= 1.0


# ===========================================================================
# B. PBO 的行为：过拟合的构造必须被抓到
# ===========================================================================

def _overfit_matrix(T=600, N=8, seed=0) -> np.ndarray:
    """
    构造"IS 漂亮、OOS 一文不值"的候选集：每个策略只在**自己那一段**里有正收益，
    其余时间是纯噪声。按 IS 最优去挑，挑中的必然是在 OOS 段没有优势的那个。
    """
    rng = np.random.default_rng(seed)
    R = rng.normal(0, 0.01, (T, N))
    seg = T // N
    for j in range(N):
        R[j * seg:(j + 1) * seg, j] += 0.05      # 只在自己的窗口里爆发
    return R


class TestPboBehaviour:

    def test_a_deliberately_overfit_selection_scores_high(self):
        pbo = cpcv_pbo(_overfit_matrix(), n_groups=6, k=2, embargo=5)
        assert pbo > 0.5, (
            f"为『每个策略只在自己那段好』这种教科书式过拟合构造算出 PBO={pbo:.2f} —— "
            f"选择流程过拟合没有被抓到")

    def test_independent_noise_lands_near_one_half(self):
        """
        纯噪声下"挑 IS 最优"没有任何信息 → OOS 排名应随机 → PBO ≈ 0.5。
        偏离太远说明统计量算错了（例如排名方向反了）。
        """
        rng = np.random.default_rng(7)
        pbo = cpcv_pbo(rng.normal(0, 0.01, (900, 6)), n_groups=6, k=2, embargo=5)
        assert 0.2 <= pbo <= 0.8, f"独立噪声的 PBO={pbo:.2f}，离 0.5 太远"

    def test_a_genuinely_persistent_winner_scores_low(self):
        """
        反向对照：一个**始终**更好的策略在 OOS 上也该是最好的 → PBO 低。
        没有这一条，"PBO 恒为高" 也能通过上面那条。
        """
        rng = np.random.default_rng(3)
        R = rng.normal(0, 0.01, (900, 5))
        R[:, 0] += 0.004                       # 全程稳定占优
        pbo = cpcv_pbo(R, n_groups=6, k=2, embargo=5)
        assert pbo < 0.3, f"全程占优的策略被判 PBO={pbo:.2f}（应当很低）"

    def test_a_single_candidate_is_refused_rather_than_scored_zero(self):
        """
        候选只有 1 个时 PBO 无定义。返回 0.0 会被读成"完全不过拟合" ——
        最松的那一侧，正是 DEV_LESSONS §U 禁止的兜底方向。
        """
        with pytest.raises(ValueError, match="至少 2 个"):
            cpcv_pbo(np.random.normal(0, 0.01, (600, 1)))

    def test_a_one_dimensional_input_is_refused(self):
        with pytest.raises(ValueError, match="二维"):
            cpcv_pbo(np.random.normal(0, 0.01, 600))

    def test_cpcv_and_cscv_agree_in_spirit_on_the_overfit_case(self):
        """
        两种口径对同一份"明显过拟合"的数据都应给出高 PBO。
        它们不必相等（purge 会改变块的构成），但结论不该相反 ——
        若相反，说明其中一个的实现有方向性错误。
        """
        R = _overfit_matrix(seed=11)
        cpcv = cpcv_pbo(R, n_groups=6, k=2, embargo=5)
        cscv = probability_of_backtest_overfitting(R, n_splits=8)
        assert cpcv > 0.5 and cscv > 0.5, f"cpcv={cpcv:.2f} cscv={cscv:.2f} 结论相反"


# ===========================================================================
# C. 回测路径分布
# ===========================================================================

class TestPathSharpes:

    def test_it_returns_one_sharpe_per_combination(self):
        out = cpcv_path_sharpes(np.random.normal(0, 0.01, 900), n_groups=6, k=2, embargo=5)
        assert len(out) == comb(6, 2), f"应有 C(6,2)=15 条路径，实得 {len(out)}"
        assert np.isfinite(out).all(), "路径夏普里有 NaN/inf"

    def test_a_strong_drift_makes_every_path_positive(self):
        r = np.random.default_rng(1).normal(0, 0.001, 900) + 0.003
        out = cpcv_path_sharpes(r, n_groups=6, k=2, embargo=5)
        assert (out > 0).all(), f"恒定正漂移下仍有路径为负：{out[out <= 0]}"

    def test_the_distribution_is_not_a_single_point(self):
        """
        这个函数存在的意义就是"把一个点估计换成一个分布"。
        若所有路径都相同，说明切分退化了（比如每次都取到同一块）。
        """
        out = cpcv_path_sharpes(np.random.default_rng(5).normal(0, 0.01, 900),
                                n_groups=6, k=2, embargo=5)
        assert out.std() > 1e-6, f"15 条路径的夏普几乎完全相同（std={out.std():.2e}）"
