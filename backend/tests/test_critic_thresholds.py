"""
agent/_critic.py —— 红队判定阈值的定钉测试（变异测试驱动）

来由：18 个变异点，首测击杀率 **50.0%**（存活 9）—— 存活的**全部是阈值边界**。

`OverfitCritic.check` 是六种失败模式的优先级判定，第一个命中就返回。
它的输出决定两件事：候选**过不过**（`passed`），以及**推荐哪种变异**去修。
六个阈值各差一格的后果不一样，但方向一致：要么放过该拦的，要么拦掉该放的，
而两种都不会让任何测试变红 —— 判定结果本来就没有"正确答案"可对照。

存活项（全部是 `<` / `>` 的边界）：
  `oos_sharpe < 0.05`（无信号）、`overfit_score > 0.60`（严重过拟合）、
  `turnover > 3.0`（高换手）、`turnover > 5.0`（严重度分级）、
  `max_dd > 0.30`（高回撤）、`overfit_score > 0.50` / `oos_sharpe < 0.20`
  （轻度过拟合）、`oos_sharpe < 0.35`（弱信号），以及 `passed=False` 本身。

既有覆盖（unit/test_agent_critic）测的是"六种模式各能触发一次"，
用的都是远离边界的极端值。
"""
from __future__ import annotations

import pytest

from app.agent._constants import _MIN_OOS_SHARPE, _OVERFIT_THRESHOLD
from app.agent._critic import (
    _HIGH_DRAWDOWN_THRESHOLD,
    _HIGH_TURNOVER_THRESHOLD,
    _NO_SIGNAL_THRESHOLD,
    _SEVERE_OVERFIT_THRESHOLD,
    _WEAK_SIGNAL_THRESHOLD,
    OverfitCritic,
)


def _check(**kw):
    """一份"一切正常"的指标，按需覆盖某一项。"""
    m = {"oos_sharpe": 1.0, "is_sharpe": 1.1, "overfitting_score": 0.0,
         "is_overfit": False, "turnover": 0.5, "max_drawdown": -0.05}
    m.update(kw)
    return OverfitCritic.check(m)


# ===========================================================================
# A. 六个阈值的边界
# ===========================================================================

class TestThresholdBoundaries:

    def test_a_healthy_candidate_passes(self):
        """先证明"通过"这条路走得通 —— 否则下面所有断言都可能恒成立。"""
        res = _check()
        assert res.passed is True, f"健康候选被拦下：{res.failure_mode} / {res.reason}"
        assert res.failure_mode in (None, "", "none"), res.failure_mode

    def test_no_signal_threshold_is_strict(self):
        """
        `if oos_sharpe < _NO_SIGNAL_THRESHOLD:` —— **严格小于 0.05**。
        放宽成 `<=` 会把恰好踩在 0.05 上的候选判成"完全无信号"并建议
        `replace_subtree`（整棵树推倒重来）—— 那是最激进的一种修正。
        """
        at = _check(oos_sharpe=_NO_SIGNAL_THRESHOLD)
        assert at.failure_mode != "no_signal", (
            f"OOS Sharpe 恰好等于 {_NO_SIGNAL_THRESHOLD} 被判成无信号 —— "
            f"`<` 被放宽成了 `<=`")
        below = _check(oos_sharpe=_NO_SIGNAL_THRESHOLD - 1e-9)
        assert below.failure_mode == "no_signal", "低于阈值却没有判无信号"
        assert below.severity == "critical"

    def test_severe_overfit_threshold_is_strict(self):
        """
        `if (is_overfit or overfit_score > 0.60) and is_sharpe > 0.5:`
        —— 恰好 0.60 不该触发"严重过拟合"。
        """
        at = _check(oos_sharpe=0.5, is_sharpe=2.0,
                    overfitting_score=_SEVERE_OVERFIT_THRESHOLD)
        assert at.failure_mode != "severe_overfitting", (
            f"过拟合分恰好 {_SEVERE_OVERFIT_THRESHOLD} 被判成严重过拟合")
        over = _check(oos_sharpe=0.5, is_sharpe=2.0,
                      overfitting_score=_SEVERE_OVERFIT_THRESHOLD + 1e-9)
        assert over.failure_mode == "severe_overfitting"

    def test_severe_overfit_needs_a_real_in_sample_sharpe(self):
        """
        `and is_sharpe > 0.5` —— IS 本来就没表现的候选，谈不上"过拟合"，
        应当落到别的模式（无信号/弱信号）。`and` 放宽成 `or` 会让
        任何高过拟合分的候选都被贴上"严重过拟合"，推荐错误的修正方向。
        """
        res = _check(oos_sharpe=0.5, is_sharpe=0.5, overfitting_score=0.9)
        assert res.failure_mode != "severe_overfitting", (
            "IS Sharpe 恰好 0.5（不大于 0.5）却判成了严重过拟合")

    def test_high_turnover_threshold_is_strict(self):
        at = _check(turnover=_HIGH_TURNOVER_THRESHOLD)
        assert at.failure_mode != "high_turnover", (
            f"换手恰好 {_HIGH_TURNOVER_THRESHOLD} 被判成高换手")
        over = _check(turnover=_HIGH_TURNOVER_THRESHOLD + 1e-9)
        assert over.failure_mode == "high_turnover"

    def test_turnover_severity_split_is_strict(self):
        """
        `severity = "critical" if turnover > 5.0 else "moderate"` ——
        恰好 5.0 应当是 moderate。严重度直接决定这个候选是被丢弃还是被修，
        差一格就换了处置方式。
        """
        assert _check(turnover=5.0).severity == "moderate", (
            "换手恰好 5.0 被判成 critical —— `> 5.0` 被放宽成了 `>=`")
        assert _check(turnover=5.0 + 1e-9).severity == "critical"

    def test_high_drawdown_threshold_is_strict(self):
        at = _check(max_drawdown=-_HIGH_DRAWDOWN_THRESHOLD)
        assert at.failure_mode != "high_drawdown", (
            f"回撤恰好 {_HIGH_DRAWDOWN_THRESHOLD} 被判成高回撤")
        over = _check(max_drawdown=-(_HIGH_DRAWDOWN_THRESHOLD + 1e-9))
        assert over.failure_mode == "high_drawdown"

    def test_drawdown_sign_is_normalised(self):
        """`max_dd = abs(...)` —— 传正传负必须判同一个结果。"""
        a = _check(max_drawdown=-0.5).failure_mode
        b = _check(max_drawdown=0.5).failure_mode
        assert a == b == "high_drawdown"

    def test_mild_overfit_thresholds_are_strict(self):
        """
        `if is_overfit or overfit_score > 0.50 or oos_sharpe < 0.20:`
        —— 两个边界分别是 `>` 与 `<`，各自恰好踩线都不该触发。
        """
        at_score = _check(oos_sharpe=0.5, overfitting_score=_OVERFIT_THRESHOLD)
        assert at_score.failure_mode != "mild_overfitting", (
            f"过拟合分恰好 {_OVERFIT_THRESHOLD} 被判成轻度过拟合")
        at_sharpe = _check(oos_sharpe=_MIN_OOS_SHARPE, overfitting_score=0.0)
        assert at_sharpe.failure_mode != "mild_overfitting", (
            f"OOS Sharpe 恰好 {_MIN_OOS_SHARPE} 被判成轻度过拟合")
        # is_sharpe 压到 0.5 以下，避免被 severe 分支（需要 is_sharpe > 0.5）截走
        over = _check(oos_sharpe=0.5, is_sharpe=0.4,
                      overfitting_score=_OVERFIT_THRESHOLD + 1e-9)
        assert over.failure_mode == "mild_overfitting", (
            f"落在 mild 区间却判成了 {over.failure_mode}")
        assert over.passed is False and over.severity == "moderate"

    def test_weak_signal_threshold_is_strict(self):
        at = _check(oos_sharpe=_WEAK_SIGNAL_THRESHOLD)
        assert at.failure_mode != "weak_signal", (
            f"OOS Sharpe 恰好 {_WEAK_SIGNAL_THRESHOLD} 被判成弱信号")
        below = _check(oos_sharpe=_WEAK_SIGNAL_THRESHOLD - 1e-9)
        assert below.failure_mode == "weak_signal"

    def test_the_boundary_values_all_fall_through_to_pass(self):
        """
        把六个阈值**同时**放在各自的边界上：按契约应当一条都不触发。
        任何一处 `<`/`>` 被放宽，这条就会红 —— 一条断言压住六个边界。
        """
        res = _check(oos_sharpe=max(_NO_SIGNAL_THRESHOLD, _WEAK_SIGNAL_THRESHOLD,
                                    _MIN_OOS_SHARPE),
                     is_sharpe=0.5,
                     overfitting_score=min(_SEVERE_OVERFIT_THRESHOLD,
                                           _OVERFIT_THRESHOLD),
                     turnover=_HIGH_TURNOVER_THRESHOLD,
                     max_drawdown=-_HIGH_DRAWDOWN_THRESHOLD)
        assert res.passed is True, (
            f"六个指标全部恰好踩线却被拦下：{res.failure_mode} / {res.reason}")


# ===========================================================================
# B. 优先级与输出结构
# ===========================================================================

class TestPriorityAndShape:

    def test_first_match_wins_in_the_documented_order(self):
        """
        六种模式是优先级判定，第一个命中就返回。同时满足"无信号"和"高换手"时
        必须报无信号（更根本的问题），否则修正方向会错。
        """
        res = _check(oos_sharpe=0.0, turnover=9.0, max_drawdown=-0.9)
        assert res.failure_mode == "no_signal", (
            f"同时命中多种模式时返回了 {res.failure_mode}，应为优先级最高的 no_signal")

    def test_failed_results_are_never_marked_passed(self):
        """
        每个失败分支里的 `passed=False` —— 改成 True 会让红队"发现了问题
        但判定通过"，候选照样往下走，而报告里还写着失败原因。
        """
        for kw in ({"oos_sharpe": 0.0},
                   {"oos_sharpe": 0.5, "is_sharpe": 2.0, "overfitting_score": 0.9},
                   {"turnover": 9.0},
                   {"max_drawdown": -0.9},
                   # mild 分支：过拟合分落在 (0.50, 0.60] 之间 —— 超过 mild 阈值
                   # 但**没到** severe 阈值。上一版用 0.7，被 severe 分支先截走，
                   # mild 那条 `passed=False` 根本没走到（复测时它存活了）。
                   {"oos_sharpe": 0.5, "is_sharpe": 0.4, "overfitting_score": 0.55},
                   {"oos_sharpe": 0.3}):
            res = _check(**kw)
            assert res.passed is False, (
                f"{kw} 命中了 {res.failure_mode} 却仍然 passed=True")
            assert res.failure_mode, f"{kw} 判定失败却没有给出 failure_mode"
            assert res.recommended_mutation, f"{kw} 没有给出推荐的修正变异"
            assert res.reason and len(res.reason) > 20, f"{kw} 的理由过于简略"

    def test_metrics_snapshot_carries_every_input(self):
        res = _check(oos_sharpe=0.0, turnover=7.25, max_drawdown=-0.42)
        snap = res.metrics_snapshot
        assert snap["turnover"] == pytest.approx(7.25, abs=0.01)
        assert snap["max_drawdown"] == pytest.approx(-0.42, abs=0.001), (
            "快照里的回撤没有还原成负号")
        assert set(snap) == {"oos_sharpe", "is_sharpe", "overfit_score",
                             "turnover", "max_drawdown"}

    def test_missing_and_unparseable_metrics_fall_back_to_zero(self):
        """`_f()` 的容错：缺字段/非数字都按 0 处理，不得抛。"""
        res = OverfitCritic.check({})
        assert res.failure_mode == "no_signal", "空指标应当落到无信号"
        res2 = OverfitCritic.check({"oos_sharpe": "abc", "turnover": None})
        assert res2.failure_mode == "no_signal"

    def test_turnover_accepts_both_field_names(self):
        """回测侧叫 `is_turnover`，GP 侧叫 `turnover`，两者都要认。"""
        a = OverfitCritic.check({"oos_sharpe": 1.0, "is_turnover": 9.0})
        b = OverfitCritic.check({"oos_sharpe": 1.0, "turnover": 9.0})
        assert a.failure_mode == b.failure_mode == "high_turnover"

    def test_is_overfit_flag_alone_triggers_overfit_modes(self):
        """`is_overfit or ...` 的短路：布尔标志单独就该触发。"""
        res = _check(oos_sharpe=0.5, is_sharpe=2.0, is_overfit=True,
                     overfitting_score=0.0)
        assert res.failure_mode == "severe_overfitting"
