"""
test_partial_fill_semantics.py — 执行层撮合原语 `simulate_partial_fills` 的逐项保护

来由（2026-09-20）
------------------
这个原语是修缺陷 A-6 时新写的：执行层**按交易差额部分成交、绝不把未成交额度
分配给别的标的**。写完当时只在一次性脚本里跑了 200 组随机恒等式检查 ——
**那个脚本从没进套件**。

随后按"修完立刻复核新变异点"的规矩跑 `verify_mutant`，结果很难看：

    filled_w = prev_w - filled_d      （符号反了）→ 存活
    unfilled_d = desired + filled_d   （符号反了）→ 存活
    desired = target_w + prev_w       （符号反了）→ 存活
    deferred 的四个变异              → 全部存活
    net_tol 默认值 1e-9 → 1e+9        → 存活

也就是说：**一个决定"实际下多少单"的函数，改坏符号都没人会红。**
临时脚本里验过 ≠ 套件保护着。本文件把它补上。

覆盖（对应用户 2026-09-20 的验收清单 #5）：
  低敞口 / 多空单腿受限 / 零目标 / 已有持仓 / 受限减仓与平仓 /
  未成交量与实际成交成本 / 订单组比例 / 组合级净敞口回查
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.backtest_engine.transaction_cost import (
    _largest_feasible_scale,
    project_to_capped_l1,
    simulate_partial_fills,
)

INF = np.inf


def _f(prev, target, cap, **kw):
    return simulate_partial_fills(np.array(prev, dtype=float),
                                  np.array(target, dtype=float),
                                  np.array(cap, dtype=float), **kw)


# ===========================================================================
# A. 恒等式 —— 三个向量之间的关系不许错，符号尤其
# ===========================================================================

class TestBookkeepingIdentities:
    """
    这三条各自钉死一个**符号**。它们看起来像废话，但正是这三处的符号变异
    在首次复核里**全部存活** —— 因为此前没有任何用例读过 `filled_d` /
    `unfilled_d`，只读了成交后的持仓。
    """

    CASES = [
        ([0.0, 0.0], [0.5, -0.3], [INF, INF]),            # 空仓建仓
        ([0.4, -0.2], [0.1, 0.1], [INF, INF]),            # 有昨仓、双向调
        ([0.4, -0.2], [0.0, 0.0], [INF, INF]),            # 全平
        ([0.3, 0.2], [0.3, 0.2], [INF, INF]),             # 无变动
        ([0.0, 0.0], [0.9, -0.1], [0.01, INF]),           # 单腿受限
        ([0.2], [0.0], [0.05]),                           # 受限平仓
    ]

    @pytest.mark.parametrize("prev,target,cap", CASES)
    def test_filled_book_is_prev_plus_traded(self, prev, target, cap):
        """`filled_w == prev_w + filled_d` —— 写成减号会让账本反向移动。"""
        r = _f(prev, target, cap)
        np.testing.assert_allclose(r.filled_w, np.array(prev) + r.filled_d,
                                   rtol=0, atol=0,
                                   err_msg="成交后持仓 ≠ 昨仓 + 实际成交量")

    @pytest.mark.parametrize("prev,target,cap", CASES)
    def test_desired_splits_into_filled_and_unfilled(self, prev, target, cap):
        """
        `desired_d == filled_d + unfilled_d` —— 未成交量是**差额**。
        写成加号时，未成交量会变成"成交量的两倍"，账本上凭空多出交易意图。
        """
        r = _f(prev, target, cap)
        np.testing.assert_allclose(r.desired_d, r.filled_d + r.unfilled_d,
                                   rtol=0, atol=1e-15,
                                   err_msg="想要的量 ≠ 成交 + 未成交")

    @pytest.mark.parametrize("prev,target,cap", CASES)
    def test_desired_is_target_minus_prev(self, prev, target, cap):
        """
        `desired_d == target_w - prev_w`。写成加号时，有昨仓的情况下
        交易方向直接错 —— 想从 0.4 减到 0.1，会变成"再买 0.5"。
        """
        r = _f(prev, target, cap)
        np.testing.assert_allclose(r.desired_d, np.array(target) - np.array(prev),
                                   rtol=0, atol=0)

    @pytest.mark.parametrize("prev,target,cap", CASES)
    def test_no_trade_ever_exceeds_its_cap(self, prev, target, cap):
        """成交量**逐名**不得超过当日可成交量上限。"""
        r = _f(prev, target, cap)
        assert np.all(np.abs(r.filled_d) <= np.abs(cap) + 1e-15), (
            f"成交量 {r.filled_d} 超过上限 {cap}")

    def test_the_identities_hold_on_random_inputs(self):
        """
        随机批量 —— 单点用例可能恰好绕开某个分支。
        （这批检查原本只存在于一次性脚本里，本条把它固化进套件。）
        """
        rng = np.random.default_rng(20260920)
        for _ in range(300):
            n = int(rng.integers(1, 6))
            prev = rng.normal(0, 0.3, n)
            target = rng.normal(0, 0.3, n)
            cap = np.abs(rng.normal(0, 0.2, n))
            r = _f(prev, target, cap, max_net=float(abs(rng.normal(0, 1))))
            np.testing.assert_allclose(r.filled_w, prev + r.filled_d, rtol=0, atol=0)
            # atol 取 1e-15 而不是 0：`unfilled = desired - filled` 再加回去
            # 有末位浮点残差（实测 5.6e-17）。符号变异会差好几个数量级，仍被抓到。
            np.testing.assert_allclose(r.desired_d, r.filled_d + r.unfilled_d,
                                       rtol=0, atol=1e-15)
            assert np.all(np.abs(r.filled_d) <= cap + 1e-15)


# ===========================================================================
# B. 不再分配：一个标的受限，不得扩大另一个标的
# ===========================================================================

class TestNoRedistribution:

    def test_a_capped_leg_does_not_enlarge_the_other(self):
        """
        缺陷 A-6 的核心：A 想要 90% 但只能成交 1%，B 想要 -10%。
        旧的 water-filling 把 89% 的亏空摊给 B（落账 -0.99）。
        """
        r = _f([0.0, 0.0], [0.9, -0.1], [0.01, INF])
        assert r.filled_w[0] == pytest.approx(0.01, abs=1e-12)
        assert r.filled_w[1] == pytest.approx(-0.10, abs=1e-12), (
            f"B 落账 {r.filled_w[1]} —— A 的未成交额度被摊给了 B")

    def test_the_shortfall_is_reported_not_absorbed(self):
        """未成交的 89% 必须出现在 `unfilled_d` 里，不能消失。"""
        r = _f([0.0, 0.0], [0.9, -0.1], [0.01, INF])
        assert r.unfilled_d[0] == pytest.approx(0.89, abs=1e-12)
        assert r.unfilled_d[1] == pytest.approx(0.0, abs=1e-12)

    def test_zero_target_from_a_position_is_a_full_close(self):
        """零目标 + 有昨仓 = 平仓；上限充足时必须平干净。"""
        r = _f([0.35, -0.15], [0.0, 0.0], [INF, INF])
        np.testing.assert_allclose(r.filled_w, [0.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(r.unfilled_d, [0.0, 0.0], atol=1e-15)

    def test_a_capped_close_leaves_the_remainder_on_the_book(self):
        """
        **受限平仓**（用户验收清单）：想清掉 20%，当日只能成交 5%
        → 账面还剩 15%，未成交 -0.15。不能假装已平。
        """
        r = _f([0.20], [0.0], [0.05])
        assert r.filled_w[0] == pytest.approx(0.15, abs=1e-12)
        assert r.filled_d[0] == pytest.approx(-0.05, abs=1e-12)
        assert r.unfilled_d[0] == pytest.approx(-0.15, abs=1e-12)


# ===========================================================================
# C. 订单组：共同可执行比例从**原始**订单算
# ===========================================================================

class TestGroupRatioIsPreservedFromTheOriginalOrders:

    def test_a_hedged_pair_keeps_its_ratio(self):
        """
        9:1 的对冲对，A 只能成交 1% → 全组按 φ = 0.01/0.90 同比缩。
        "先逐名裁剪再整体缩放"保持的是**裁剪后**的比例，救不回来。
        """
        r = _f([0.0, 0.0], [0.9, -0.1], [0.01, INF], groups=["h", "h"])
        assert r.filled_w[0] / abs(r.filled_w[1]) == pytest.approx(9.0, rel=1e-12)
        assert r.group_frac[0] == pytest.approx(0.01 / 0.9, rel=1e-12)
        assert r.group_frac[1] == pytest.approx(0.01 / 0.9, rel=1e-12)

    def test_ungrouped_names_are_independent(self):
        """不分组时各自独立 —— 比例约束必须**显式声明**，不靠净敞口代劳。"""
        r = _f([0.0, 0.0], [0.9, -0.1], [0.01, INF])
        assert r.filled_w[0] == pytest.approx(0.01, abs=1e-12)
        assert r.filled_w[1] == pytest.approx(-0.10, abs=1e-12)

    def test_the_binding_leg_determines_the_whole_group(self):
        """三腿组里最紧的那条腿决定 φ。"""
        r = _f([0, 0, 0], [0.6, -0.3, 0.1], [INF, 0.03, INF],
               groups=["g", "g", "g"])
        assert r.group_frac[0] == pytest.approx(0.1, rel=1e-12)   # 0.03/0.3
        np.testing.assert_allclose(r.filled_w, [0.06, -0.03, 0.01], atol=1e-15)

    def test_a_group_with_no_orders_is_not_scaled_to_zero(self):
        """组内全是零意图时 φ 取 1（而不是退化成 0 把别的搞乱）。"""
        r = _f([0.2, 0.1], [0.2, 0.1], [INF, INF], groups=["g", "g"])
        np.testing.assert_allclose(r.group_frac, [1.0, 1.0])
        np.testing.assert_allclose(r.filled_d, [0.0, 0.0])

    def test_mismatched_group_length_is_rejected(self):
        with pytest.raises(AssertionError):
            _f([0.0, 0.0], [0.1, 0.1], [INF, INF], groups=["g"])


# ===========================================================================
# D. 组合级净敞口：作用于**全部**交易，减仓不豁免
# ===========================================================================

class TestPortfolioNetConstraint:

    def test_a_per_name_reduction_that_raises_portfolio_net_is_scaled(self):
        """
        **单标的减仓 ≠ 组合降风险。**
        `[+0.30, -0.20]`（net +0.10）平掉空头是逐名"降风险"，
        但组合净敞口会升到 +0.30。把减仓当免检基准会突破 max_net。
        """
        r = _f([0.30, -0.20], [0.30, 0.0], [INF, INF], max_net=0.15)
        assert abs(float(r.filled_w.sum())) <= 0.15 + 1e-9, (
            f"成交后净敞口 {r.filled_w.sum():+.4f} 超出 max_net=0.15")
        assert r.scaled_by == pytest.approx(0.25, rel=1e-9)
        assert r.unfilled_d[1] == pytest.approx(0.15, abs=1e-12)

    def test_a_full_liquidation_is_never_blocked(self):
        """全平会把净敞口降到 0 —— 约束不该拦住它。"""
        r = _f([0.5, -0.1], [0.0, 0.0], [INF, INF], max_net=0.15)
        assert r.scaled_by == pytest.approx(1.0)
        np.testing.assert_allclose(r.filled_w, [0.0, 0.0], atol=1e-15)

    def test_a_target_that_itself_violates_is_flagged(self):
        """
        目标账本自身超限 = **构建层**的问题，执行层补不了。
        必须标出来，而不是靠少成交把它盖住。
        """
        r = _f([0.0, 0.0], [0.5, 0.4], [INF, INF], max_net=0.15)
        assert r.target_violates is True

    def test_a_prior_book_that_violates_is_flagged_separately(self):
        """昨仓已超限时，连"什么都不交易"也救不了 → 另一个标志位。"""
        r = _f([0.5, 0.4], [0.6, 0.5], [INF, INF], max_net=0.15)
        assert r.book_non_compliant is True
        assert r.scaled_by == pytest.approx(0.0)

    def test_no_constraint_means_no_scaling(self):
        """`max_net=None` 是**明确选择不限制**，不得顺手加约束。"""
        r = _f([0.0, 0.0], [0.9, 0.9], [INF, INF], max_net=None)
        assert r.scaled_by == pytest.approx(1.0)
        assert r.target_violates is False
        assert r.book_non_compliant is False


# ===========================================================================
# E. deferred 标志 —— 四个变异全部存活过
# ===========================================================================

class TestDeferredFlag:
    """`deferred = (μ == 0) and 确实有交易意图`。四个部分各自钉一条。"""

    def test_deferred_when_scaled_to_zero_with_real_intent(self):
        r = _f([0.5, 0.4], [0.6, 0.5], [INF, INF], max_net=0.15)
        assert r.scaled_by == 0.0 and r.deferred is True

    def test_not_deferred_when_trading_happens(self):
        """μ > 0 时不许报暂缓 —— `==` 写成 `!=` 会让正常成交也标成暂缓。"""
        r = _f([0.0], [0.2], [INF], max_net=1.0)
        assert r.scaled_by > 0.0 and r.deferred is False

    def test_not_deferred_when_there_was_nothing_to_trade(self):
        """
        没有交易意图时即使 μ==0 也不算"暂缓" ——
        `and` 写成 `or` 会让"本来就没单要下"被报成暂缓。
        """
        r = _f([0.5, 0.4], [0.5, 0.4], [INF, INF], max_net=0.15)
        assert np.all(r.desired_d == 0.0)
        assert r.deferred is False

    def test_intent_is_measured_on_magnitude_not_sign(self):
        """
        判"有没有交易意图"要看 `|desired|`。去掉 `abs` 之后，
        **纯减仓**（desired 全为负）会被判成"没意图"而不报暂缓。
        """
        r = _f([-0.5, -0.4], [-0.6, -0.5], [INF, INF], max_net=0.15)
        assert np.all(r.desired_d < 0.0), "构造的是纯负向意图"
        assert r.scaled_by == 0.0
        assert r.deferred is True, "纯减仓的意图被漏判成了『没有意图』"


# ===========================================================================
# F. `_largest_feasible_scale` —— 边界与容差
# ===========================================================================

class TestLargestFeasibleScale:

    def test_returns_one_when_trade_does_not_move_net(self):
        assert _largest_feasible_scale(0.05, 0.0, 0.10) == pytest.approx(1.0)

    def test_returns_none_when_static_net_already_violates(self):
        assert _largest_feasible_scale(0.5, 0.0, 0.10) is None

    def test_picks_the_upper_end_of_the_feasible_interval(self):
        """0.10 + μ·0.20 ≤ 0.15 → μ ≤ 0.25，取最大可行值。"""
        assert _largest_feasible_scale(0.10, 0.20, 0.15) == pytest.approx(0.25)

    def test_handles_a_negative_step(self):
        """s < 0 时区间端点要对调 —— 不对调会取到错误的一端。"""
        assert _largest_feasible_scale(0.40, -0.40, 0.15) == pytest.approx(1.0)

    def test_is_capped_at_one(self):
        assert _largest_feasible_scale(0.0, 0.01, 10.0) == pytest.approx(1.0)

    def test_the_default_tolerance_is_tiny_not_huge(self):
        """
        `net_tol` 默认 1e-9。写成 1e+9 时**任何**净敞口都会被判为合规 ——
        约束彻底失效却一条用例都不红（该变异首次复核时确实存活）。
        """
        assert _largest_feasible_scale(1.0, 0.0, 0.1) is None, (
            "净敞口 1.0 远超上限 0.1 却判为可行 —— 容差量级错了")
        r = _f([1.0, 0.0], [1.0, 0.0], [INF, INF], max_net=0.1)
        assert r.book_non_compliant is True


# ===========================================================================
# G. 投影的逐行 target（构建层原语，A-6 修复时新增的分支）
# ===========================================================================

class TestPerRowProjectionTarget:

    def test_a_scalar_target_applies_to_every_row(self):
        """
        ⚠️ 上限必须**有限**才谈得上放大：`budget` 对 `cap=inf` 的名取的是
        `|w|` 本身，于是 `row_target = min(target, budget) = |w|.sum()` ——
        全 inf 上限下这个函数只能缩不能放。
        （我第一版用例在这里写错了期望值，被自己的断言抓到。）
        """
        w = np.array([[0.6, -0.4], [0.3, -0.2]])
        cap = np.full_like(w, 10.0)
        out = project_to_capped_l1(w, cap, target=1.0)
        np.testing.assert_allclose(np.abs(out).sum(axis=1), [1.0, 1.0], atol=1e-12)

    def test_a_per_row_target_is_applied_row_by_row(self):
        """
        逐行 target 是 A-6 修复时加的分支（`tgt.ndim == 1`）。
        把条件写反会让 (T,) 形状的 target 当成标量广播 —— 每行都用同一个值。
        两行给**不同**的目标，且都需要**放大**（0.5>0.2 的那一行必须真的被拉到 0.5），
        这样"每行各自的 target"与"共用一个 target"才区分得开。
        """
        w = np.array([[0.06, -0.04], [0.03, -0.02]])      # gross 0.1 / 0.05
        cap = np.full_like(w, 10.0)
        out = project_to_capped_l1(w, cap, target=np.array([0.5, 0.2]))
        np.testing.assert_allclose(np.abs(out).sum(axis=1), [0.5, 0.2], atol=1e-12)

    def test_the_budget_caps_the_target(self):
        """
        `row_target = min(target, budget)` —— 写成 max 会让预算不足时
        把已触顶的权重推回超限（旧缺陷 E-N1 的形态）。
        """
        w = np.array([[0.6, -0.4]])
        cap = np.array([[0.05, 0.05]])
        out = project_to_capped_l1(w, cap, target=1.0)
        assert np.abs(out).sum() == pytest.approx(0.10, abs=1e-12)
        assert np.all(np.abs(out) <= 0.05 + 1e-15)
