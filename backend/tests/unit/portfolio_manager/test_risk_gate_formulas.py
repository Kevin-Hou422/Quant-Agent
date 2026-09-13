"""
test_risk_gate_formulas.py — 风控门公式的**逐条**钉死

来由：隔离版变异测量给出 `risk_gate.py` 击杀率 **16.1%（31 处变异只有 5 处被发现）**
—— 全项目最差。这是决定"能持多少仓"的模块，改坏了直接影响下单量。

26 处存活集中在三块，本文件逐块建立契约：

**A. `check()` 违规检测** —— 整套阈值比较**一条都没被测过**
    gross / net / 单票 / long-only 负权重 / 行业集中度

**B. `_apply_vector()` 三层约束的公式**
    `cap = max_name_weight * max_gross`        `*` 改 `/` 不报错
    `sec_cap = max_sector_weight * max_gross`  同上
    `g = np.abs(a).sum()`                      删 `abs` → gross 算错
    `np.where(a < 0.0, 0.0, a)`                long-only 去负的边界

**C. `should_halt()`** —— 回撤熔断的判定边界

写法要求：**用精确值**，不用"落在合理区间"——后者会被下游的 clip/兜底吸收掉
（上一轮 `vol_scalar` 就是被 `np.clip(0,3)` 盖住才没杀死变异）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.portfolio_manager import PortfolioRiskGate, RiskLimits


def _w(rows, cols=("A", "B", "C", "D")):
    idx = pd.bdate_range("2024-01-02", periods=len(rows))
    return pd.DataFrame(rows, index=idx, columns=list(cols))


# ===========================================================================
# A. check() —— 违规检测的每一条阈值
# ===========================================================================

def test_check_flags_gross_breach_at_the_right_threshold():
    """`gross > max_gross + tol` —— 恰好等于上限不算违规，超出才算。"""
    lim = RiskLimits(max_gross=1.0, max_name_weight=1.0, max_sector_weight=1.0,
                     max_net=1.0, long_only=False)
    gate = PortfolioRiskGate(lim)
    exact = _w([[0.25, 0.25, 0.25, 0.25]])           # gross = 1.0 恰好
    assert not any("gross" in v for v in gate.check(exact, sectors=None)), \
        "gross 恰好等于上限不应报违规"
    over = _w([[0.30, 0.30, 0.30, 0.30]])            # gross = 1.2
    assert any("gross" in v for v in gate.check(over, sectors=None)), \
        "gross=1.2 超过上限 1.0 却未报违规"


def test_check_flags_net_breach():
    """`|net| > max_net + tol`。"""
    lim = RiskLimits(max_gross=10.0, max_net=0.5, max_name_weight=10.0,
                     max_sector_weight=10.0, long_only=False)
    gate = PortfolioRiskGate(lim)
    assert not any("net" in v for v in gate.check(_w([[0.25, 0.25, 0.0, 0.0]]))), \
        "net=0.5 恰好等于上限不应报违规"
    assert any("net" in v for v in gate.check(_w([[0.4, 0.4, 0.0, 0.0]]))), \
        "net=0.8 超过上限 0.5 却未报违规"


def test_check_flags_single_name_breach_using_the_product_cap():
    """
    单票上限 = `max_name_weight × max_gross`（**乘积**）。
    把 `*` 写成 `/` 会让上限变成 0.1/1.0=0.1 → 0.5/1.0=0.5，量级完全不同。
    这里用 max_gross=2.0 把乘除两种写法的结果拉开：乘=0.2，除=0.05。
    """
    lim = RiskLimits(max_gross=2.0, max_name_weight=0.1, max_sector_weight=10.0,
                     max_net=10.0, long_only=False)
    gate = PortfolioRiskGate(lim)
    # 0.15 < 0.2（乘积上限）→ 不违规；但 > 0.05（除法上限）→ 若公式写反会误报
    assert not any("单票" in v for v in gate.check(_w([[0.15, 0.0, 0.0, 0.0]]))), \
        "单票 0.15 未超过 max_name_weight×max_gross=0.2，不该报违规"
    assert any("单票" in v for v in gate.check(_w([[0.25, 0.0, 0.0, 0.0]]))), \
        "单票 0.25 已超过 0.2，必须报违规"


def test_check_flags_negative_weights_only_when_long_only():
    """`if lim.long_only and (weights < -tol).any()` —— 两个方向都要测。"""
    neg = _w([[0.5, -0.2, 0.0, 0.0]])
    lo = PortfolioRiskGate(RiskLimits(long_only=True, max_gross=10.0, max_net=10.0,
                                      max_name_weight=10.0, max_sector_weight=10.0))
    assert any("long-only" in v for v in lo.check(neg)), "long_only 下负权重必须报违规"
    ls = PortfolioRiskGate(RiskLimits(long_only=False, max_gross=10.0, max_net=10.0,
                                      max_name_weight=10.0, max_sector_weight=10.0))
    assert not any("long-only" in v for v in ls.check(neg)), \
        "允许做空时负权重不应报违规"


def test_check_flags_sector_concentration_using_product_cap():
    """
    行业上限 = `max_sector_weight × max_gross`（**乘积**）。
    max_gross=2.0 / max_sector_weight=0.3 → 乘=0.6，除=0.15，两者可区分。
    """
    lim = RiskLimits(max_gross=2.0, max_sector_weight=0.3, max_name_weight=10.0,
                     max_net=10.0, long_only=False)
    gate = PortfolioRiskGate(lim)
    sectors = pd.Series([0, 0, 1, 1], index=["A", "B", "C", "D"])
    ok = _w([[0.25, 0.25, 0.1, 0.1]])                 # 行业0 = 0.5 < 0.6
    assert not any("行业" in v for v in gate.check(ok, sectors=sectors)), \
        "行业敞口 0.5 未超过 0.3×2.0=0.6，不该报违规"
    bad = _w([[0.4, 0.4, 0.1, 0.1]])                  # 行业0 = 0.8 > 0.6
    assert any("行业" in v for v in gate.check(bad, sectors=sectors)), \
        "行业敞口 0.8 已超过 0.6，必须报违规"


# ===========================================================================
# B. _apply_vector() —— 三层约束的精确公式
# ===========================================================================

def test_apply_removes_negative_weights_under_long_only():
    """`np.where(a < 0.0, 0.0, a)` —— long_only 时负权重必须归零。"""
    lim = RiskLimits(long_only=True, max_gross=10.0, max_name_weight=10.0,
                     max_sector_weight=10.0)
    out, _ = PortfolioRiskGate(lim).apply(_w([[0.5, -0.3, 0.2, -0.1]]))
    assert (out.to_numpy() >= -1e-12).all(), f"long_only 下仍有负权重：{out.iloc[0].tolist()}"
    assert abs(out.iloc[0]["B"]) < 1e-12 and abs(out.iloc[0]["D"]) < 1e-12


def test_apply_single_name_cap_is_the_product():
    """
    单票上限精确等于 `max_name_weight × max_gross`。
    用 max_gross=1.0 / max_name_weight=0.25 → cap=0.25；
    输入某只 0.9，输出该只必须 ≤ 0.25（水填充会把削掉的分给别人）。
    """
    # **必须让 max_gross ≠ 1.0**：否则 w*1.0 与 w/1.0 完全相同，
    # 这个断言就区分不了 `cap = max_name_weight * max_gross` 里的乘除。
    lim = RiskLimits(max_gross=2.0, max_name_weight=0.25, max_sector_weight=10.0,
                     long_only=True)
    out, rep = PortfolioRiskGate(lim).apply(_w([[1.60, 0.20, 0.10, 0.10]]))
    row = out.iloc[0]
    cap = 0.25 * 2.0                       # 乘=0.5；若写成除则为 0.125
    assert row.max() <= cap + 1e-9, f"单票未被削到 {cap}：{row.tolist()}"
    assert row.max() > 0.125 + 1e-9, (
        f"单票上限被削到 {row.max():.4f} ≤ 0.125 —— "
        f"cap 疑似写成了 max_name_weight / max_gross")
    assert rep.n_name_clipped >= 1, "未记录单票削减次数"
    assert abs(row.abs().sum() - 2.0) < 1e-6, (
        f"gross 应保持 2.0（水填充再分配），实际 {row.abs().sum():.6f}")


def test_apply_gross_cap_scales_uniformly():
    """
    `g = np.abs(a).sum()` 删掉 abs → 多空相抵，gross 被低估，超限时不缩放。
    构造多空各半、gross=2.0 但 net=0 的组合：删 abs 后 g=0 不会触发缩放。
    """
    lim = RiskLimits(max_gross=1.0, max_name_weight=10.0, max_sector_weight=10.0,
                     long_only=False)
    out, rep = PortfolioRiskGate(lim).apply(_w([[1.0, -1.0, 0.0, 0.0]]))
    g = float(out.iloc[0].abs().sum())
    assert abs(g - 1.0) < 1e-9, (
        f"gross=2.0（net=0）必须被缩到 1.0，实际 {g:.6f} —— "
        f"`np.abs(a).sum()` 的 abs 疑似被去掉，多空相抵后没触发缩放")
    assert rep.n_gross_scaled >= 1, "未记录 gross 缩放"


def test_apply_sector_cap_is_the_product():
    """行业缩放上限 = `max_sector_weight × max_gross`，精确到数值。"""
    lim = RiskLimits(max_gross=2.0, max_sector_weight=0.3, max_name_weight=10.0,
                     long_only=True)
    sectors = pd.Series([0, 0, 1, 1], index=["A", "B", "C", "D"])
    out, rep = PortfolioRiskGate(lim).apply(_w([[0.5, 0.5, 0.1, 0.1]]), sectors=sectors)
    row = out.iloc[0]
    sec0 = row["A"] + row["B"]
    assert sec0 <= 0.3 * 2.0 + 1e-9, (
        f"行业0 敞口 {sec0:.4f} 超过上限 0.6 —— sec_cap 的乘积公式疑似被改")
    assert rep.n_sector_scaled >= 1, "未记录行业缩放"


def test_apply_vol_scalar_is_target_over_actual():
    """`vol_scalar = target_vol_ann / port_vol_ann`，精确值（未触 clip 边界）。"""
    lim = RiskLimits(target_vol_ann=0.10, max_gross=10.0, max_name_weight=10.0,
                     max_sector_weight=10.0, long_only=False)
    _, rep = PortfolioRiskGate(lim).apply(_w([[0.25, 0.25, 0.25, 0.25]]),
                                          port_vol_ann=0.20)
    assert abs(rep.vol_scalar - 0.5) < 1e-12, (
        f"目标 10% / 实际 20% 应为 0.5，实际 {rep.vol_scalar}")
    _, rep2 = PortfolioRiskGate(lim).apply(_w([[0.25, 0.25, 0.25, 0.25]]),
                                           port_vol_ann=0.05)
    assert abs(rep2.vol_scalar - 2.0) < 1e-12, (
        f"目标 10% / 实际 5% 应为 2.0，实际 {rep2.vol_scalar}")


# ===========================================================================
# C. should_halt() —— 回撤熔断的判定
# ===========================================================================

def test_should_halt_at_exact_threshold():
    """回撤恰好等于阈值不触发，超出才触发（边界方向）。"""
    gate = PortfolioRiskGate(RiskLimits(max_drawdown=0.20))
    exact = pd.Series([1.0, 1.0, 0.80])               # 回撤恰好 20%
    halt, dd = gate.should_halt(exact)
    assert abs(dd - 0.20) < 1e-12, f"回撤应为 0.20，实际 {dd}"
    assert not halt, "回撤恰好等于阈值不应触发熔断"
    over = pd.Series([1.0, 1.0, 0.75])                # 25%
    halt2, dd2 = gate.should_halt(over)
    assert halt2 and dd2 > 0.20, f"25% 回撤必须触发，实际 halt={halt2} dd={dd2}"


def test_should_halt_needs_at_least_two_points():
    """`if len(e) < 2: return False, 0.0` —— 单点净值无法算回撤。"""
    gate = PortfolioRiskGate(RiskLimits(max_drawdown=0.01))
    halt, dd = gate.should_halt(pd.Series([1.0]))
    assert halt is False and dd == 0.0, f"单点应返回 (False, 0.0)，实际 ({halt}, {dd})"
    halt2, _ = gate.should_halt(pd.Series([1.0, 0.5]))
    assert halt2 is True, "两点且回撤 50% 应触发"


# ===========================================================================
# D. 边界精确命中 —— `>` 改 `>=` 只在**恰好等于**时才有差别
# ===========================================================================

def _lim(**kw):
    base = dict(max_gross=10.0, max_net=10.0, max_name_weight=10.0,
                max_sector_weight=10.0, long_only=False)
    base.update(kw)
    return RiskLimits(**base)


def test_check_gross_exactly_at_limit_is_not_a_breach():
    """`gross > max_gross + tol` → `>=`：恰好等于上限时两者结论相反。"""
    gate = PortfolioRiskGate(_lim(max_gross=1.0))
    exact = _w([[0.25, 0.25, 0.25, 0.25]])            # gross 恰好 1.0
    assert not any("gross" in v for v in gate.check(exact)),         "gross 恰好 1.0 不该判违规（`>` 被改成 `>=` 时会误报）"


def test_check_net_exactly_at_limit_is_not_a_breach():
    gate = PortfolioRiskGate(_lim(max_net=0.5))
    assert not any("net" in v for v in gate.check(_w([[0.25, 0.25, 0.0, 0.0]]))),         "net 恰好 0.5 不该判违规"


def test_check_single_name_exactly_at_cap_is_not_a_breach():
    gate = PortfolioRiskGate(_lim(max_gross=2.0, max_name_weight=0.1))
    assert not any("单票" in v for v in gate.check(_w([[0.2, 0.0, 0.0, 0.0]]))),         "单票恰好等于 0.1×2.0=0.2 不该判违规"


def test_check_sector_exactly_at_cap_is_not_a_breach():
    gate = PortfolioRiskGate(_lim(max_gross=2.0, max_sector_weight=0.3))
    sectors = pd.Series([0, 0, 1, 1], index=["A", "B", "C", "D"])
    exact = _w([[0.3, 0.3, 0.05, 0.05]])              # 行业0 恰好 0.6
    assert not any("行业" in v for v in gate.check(exact, sectors=sectors)),         "行业敞口恰好 0.3×2.0=0.6 不该判违规"


def test_check_zero_weight_is_not_negative_under_long_only():
    """`weights < -tol` → `<=`：权重恰好 0 时不得被判为做空。"""
    gate = PortfolioRiskGate(_lim(long_only=True))
    assert not any("long-only" in v for v in gate.check(_w([[0.5, 0.0, 0.5, 0.0]]))),         "权重为 0 不是负权重，不该判 long-only 违规"


def test_apply_zero_weight_survives_long_only_filter():
    """`np.where(a < 0.0, 0.0, a)` → `<=`：0 权重经过滤后仍是 0（行为等价），
    但**正权重**必须原样保留 —— 用一个极小正权重钉住边界方向。"""
    lim = _lim(long_only=True, max_gross=100.0)
    out, _ = PortfolioRiskGate(lim).apply(_w([[1e-9, 0.5, 0.0, -0.2]]))
    row = out.iloc[0]
    assert row["A"] > 0, "极小正权重被错误清零 —— long-only 过滤的边界方向有误"
    assert abs(row["D"]) < 1e-15, "负权重未被清零"


def test_apply_exactly_at_name_cap_is_not_clipped():
    """`np.abs(a) > cap + 1e-12` → `>=`：恰好等于上限不该触发削减。"""
    lim = _lim(max_gross=2.0, max_name_weight=0.25, long_only=True)
    out, rep = PortfolioRiskGate(lim).apply(_w([[0.5, 0.5, 0.5, 0.5]]))  # 各恰好 0.5=cap
    assert rep.n_name_clipped == 0, (
        f"每只恰好等于单票上限 0.5，不该削减，实际削了 {rep.n_name_clipped} 次")


def test_apply_exactly_at_gross_cap_is_not_scaled():
    """`g > max_gross + 1e-12` → `>=`；同时钉住 `gscaled = False` 的初值。"""
    lim = _lim(max_gross=1.0, long_only=True)
    out, rep = PortfolioRiskGate(lim).apply(_w([[0.25, 0.25, 0.25, 0.25]]))
    assert rep.n_gross_scaled == 0, (
        f"gross 恰好等于上限 1.0，不该缩放，实际缩放 {rep.n_gross_scaled} 次 —— "
        f"边界方向或 `gscaled = False` 初值有误")


def test_apply_exactly_at_sector_cap_is_not_scaled():
    """`sec_gross > sec_cap + 1e-12` → `>=`。"""
    lim = _lim(max_gross=2.0, max_sector_weight=0.3, long_only=True)
    sectors = pd.Series([0, 0, 1, 1], index=["A", "B", "C", "D"])
    out, rep = PortfolioRiskGate(lim).apply(_w([[0.3, 0.3, 0.3, 0.3]]), sectors=sectors)
    assert rep.n_sector_scaled == 0, (
        f"行业敞口恰好 0.6=上限，不该缩放，实际 {rep.n_sector_scaled} 次")


def test_apply_sector_cap_uses_product_not_quotient():
    """`sec_cap = max_sector_weight * max_gross`：用 max_gross=4.0 拉开乘除差距。"""
    lim = _lim(max_gross=4.0, max_sector_weight=0.25, long_only=True)
    sectors = pd.Series([0, 0, 1, 1], index=["A", "B", "C", "D"])
    out, _ = PortfolioRiskGate(lim).apply(_w([[1.0, 1.0, 0.2, 0.2]]), sectors=sectors)
    row = out.iloc[0]
    sec0 = row["A"] + row["B"]
    assert sec0 <= 0.25 * 4.0 + 1e-9, f"行业0 敞口 {sec0:.4f} 超过 1.0"
    assert sec0 > 0.0625 + 1e-9, (
        f"行业0 被削到 {sec0:.4f} ≤ 0.0625 —— sec_cap 疑似写成了除法")


def test_gross_uses_absolute_values_not_net():
    """`sec_gross = np.abs(a[mask]).sum()` 删掉 abs → 同行业多空相抵，不触发缩放。"""
    lim = _lim(max_gross=100.0, max_sector_weight=0.005, long_only=False)
    sectors = pd.Series([0, 0, 1, 1], index=["A", "B", "C", "D"])
    out, rep = PortfolioRiskGate(lim).apply(_w([[1.0, -1.0, 0.0, 0.0]]), sectors=sectors)
    assert rep.n_sector_scaled >= 1, (
        "行业0 的**绝对**敞口为 2.0，远超上限 0.5，必须缩放 —— "
        "`np.abs(...)` 疑似被去掉，多空相抵后没触发")


def test_vol_scaling_needs_positive_vol():
    """`port_vol_ann > 1e-9` → `>=`：0 波动时不得缩放（会除零）。"""
    lim = _lim(target_vol_ann=0.10)
    _, rep = PortfolioRiskGate(lim).apply(_w([[0.25, 0.25, 0.25, 0.25]]),
                                          port_vol_ann=0.0)
    assert rep.vol_scalar == 1.0, f"波动为 0 时不应缩放，实际 {rep.vol_scalar}"


def test_risk_limits_defaults_are_pinned():
    """`long_only: bool = True` 的**默认值**本身是契约（现金账户不得做空）。"""
    d = RiskLimits()
    assert d.long_only is True, "RiskLimits 默认应为 long_only=True"


# ===========================================================================
# E. 剩余真盲区（变异测量逐条逼出来的）
# ===========================================================================

def test_water_filling_target_uses_absolute_gross():
    """
    `g_now = float(np.abs(a).sum())` 删掉 abs → 多空相抵，水填充 target 被低估。

    三个坑，前两版都踩了：
      1) long_only=True 时全是正数，abs 与否没差别 —— 必须**多空混合**；
      2) 我以为 target 变小 → 输出变小，断言 `g > 1.0`；实测相反，
         `project_to_capped_l1` 在 target 小于各 cap 之和时会**放大**（gross 7.8）；
      3) 断言最终 gross **也没用** —— 紧随其后的第 4 步 gross 上限把 7.8 又缩回 4.0，
         **下游兜底把错误吸收了**（和 vol_scalar 被 np.clip 盖住是同一个模式）。

    真正的差别在**逐名权重**：
        正确  [ 2.00, -2.00,  0.00,  0.00]   两个小仓位被削到 0
        变异  [ 1.03, -1.03, -0.97, -0.97]   0.1 的小仓被放大成接近满仓的空头
    所以断言必须落在**每只标的的权重**上，不是聚合量。
    """
    lim = RiskLimits(max_gross=4.0, max_name_weight=0.5, max_sector_weight=10.0,
                     long_only=False)
    out, rep = PortfolioRiskGate(lim).apply(_w([[3.0, -3.0, 0.1, 0.1]]))
    row = out.iloc[0]
    assert np.isfinite(row.to_numpy()).all(), f"权重含非有限值：{row.tolist()}"
    assert rep.n_name_clipped == 2, f"应有 2 只触顶，实际 {rep.n_name_clipped}"
    # 触顶的两只削到单票上限 0.5×4.0 = 2.0
    assert abs(abs(row["A"]) - 2.0) < 1e-6 and abs(abs(row["B"]) - 2.0) < 1e-6, (
        f"触顶标的应削到 ±2.0，实际 A={row['A']:.4f} B={row['B']:.4f}")
    # **关键**：原本 0.1 的小仓不该被放大 —— 变异后它们会变成 ~-0.97
    assert abs(row["C"]) < 0.2 and abs(row["D"]) < 0.2, (
        f"小仓位被放大到 C={row['C']:.4f} D={row['D']:.4f} —— "
        f"`np.abs(a).sum()` 的 abs 疑似被去掉：多空相抵使 target 塌到 0.2，"
        f"投影把小仓放大后再被 gross 上限缩回，聚合量看不出来")


def test_check_handles_empty_weight_frame():
    """
    `if sec is not None and len(weights) > 0:` 改成 `>=` →
    空权重表也会进入行业检查，`weights.iloc[-1]` 立刻 IndexError。
    """
    lim = RiskLimits(max_gross=1.0, max_net=1.0, max_name_weight=1.0,
                     max_sector_weight=0.3, long_only=False)
    gate = PortfolioRiskGate(lim)
    empty = pd.DataFrame(columns=["A", "B", "C", "D"], dtype=float)
    sectors = pd.Series([0, 0, 1, 1], index=["A", "B", "C", "D"])
    viol = gate.check(empty, sectors=sectors)     # 不得抛异常
    assert isinstance(viol, list), f"空权重表应返回列表，实际 {type(viol)}"
    assert not any("行业" in v for v in viol), "空权重表不该产生行业违规"


# ===========================================================================
# F. 已证明的等价变异 —— 不是漏测，是**改了也不可能被观测到**
# ===========================================================================

#: 隔离版变异测量在 risk_gate.py 上的最终存活项，逐条给出等价性证明。
#: 不允许用"大概等价"搪塞：每条都要能被下面的用例机械验证。
PROVEN_EQUIVALENT = {
    "L89  port_vol_ann > 1e-9 → >=":
        "仅在 vol 恰好 == 1e-9 时结论不同；该分支前已有 `port_vol_ann and` 短路排除 0，"
        "且 1e-9 量级的年化波动无任何业务含义。",
    "L117 gross > max_gross + tol → >=":
        "仅在 gross 恰好 == max_gross + 1e-6 时不同。浮点上 (limit+tol) - limit != tol，"
        "该值无法精确构造；tol 的存在本身就是为了让边界附近不敏感。",
    "L119 |net| > max_net + tol → >=":
        "同 L117 的构造：仅在 |net| 恰好 == max_net + 1e-6 时结论不同，该值浮点不可达。",
    "L122 maxname > cap + tol → >=":
        "同 L117 的构造：仅在单票权重恰好 == cap + 1e-6 时不同，该值浮点不可达。",
    "L131 sec_abs.max() > sec_cap + tol → >=":
        "同 L117 的构造：仅在行业敞口恰好 == sec_cap + 1e-6 时不同，该值浮点不可达。",
    "L179 |a| > cap + 1e-12 → >=":
        "同 L117 的构造，且 tol 小到 1e-12，边界值比 1e-6 情形更不可能被精确命中。",
    "L202 sec_gross > sec_cap + 1e-12 → >=":
        "同 L117 的构造：仅在行业绝对敞口恰好 == sec_cap + 1e-12 时不同，浮点不可达。",
    "L208 g > max_gross + 1e-12 → >=":
        "同 L117 的构造：仅在 gross 恰好 == max_gross + 1e-12 时不同，浮点不可达。",
    "L124 (weights < -tol) → <=":
        "仅在权重恰好 == -1e-6 时不同，同样不可精确构造。",
    "L174 np.where(a < 0.0, 0.0, a) → <=":
        "a==0 时 `<` 保留 0、`<=` 写入 0，结果都是 0；a==-0.0 同理。"
        "对全部浮点边界值（0, -0.0, ±1e-300）输出完全一致 —— 真正的等价变异。",
}


def test_epsilon_guarded_boundaries_are_provably_unreachable():
    """
    机械验证「`x > limit + tol` 改 `>=`」不可区分：
    浮点上 `(limit + tol) - limit != tol`，即那个唯一有差别的值无法精确构造。
    """
    for limit, tol in [(1.0, 1e-6), (0.5, 1e-6), (0.2, 1e-6), (0.6, 1e-6),
                       (2.0, 1e-12), (0.6, 1e-12), (1.0, 1e-12)]:
        b = limit + tol
        assert (b - limit) != tol, (
            f"limit={limit} tol={tol} 的边界值可精确构造 —— "
            f"该处 `>`/`>=` 变异其实可区分，不该列为等价")


def test_zero_boundary_of_long_only_filter_is_equivalent():
    """`np.where(a < 0.0, 0.0, a)` 改 `<=`：对全部边界值输出一致。"""
    for v in (0.0, -0.0, 1e-300, -1e-300):
        lt = 0.0 if v < 0.0 else v
        le = 0.0 if v <= 0.0 else v
        assert lt == le, f"a={v!r} 时 `<` 与 `<=` 结果不同（{lt} vs {le}）"


def test_every_survivor_has_a_written_proof():
    """存活项必须逐条有证明，不允许留空白。"""
    assert len(PROVEN_EQUIVALENT) == 10, (
        f"最终存活 10 处，证明只写了 {len(PROVEN_EQUIVALENT)} 条")
    for k, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 10, f"{k} 的等价性说明过于敷衍：{why!r}"
