"""
alpha_engine/financial_diagnostics.py —— 回测指标 → 金融诊断与改进建议

**此前零测试**（397 有效行，D 档首批）。

这个模块决定用户被告知"你的因子哪里坏了、该怎么改"。它的输出还进
`tool_interpret_factor` 的 `diagnosis` 块，是 agent 给出改进方向的依据。
阈值错了不会报错，只会让诊断**系统性地指错方向** ——
一个换手过高的因子被诊断成"没有信号"，用户按建议去换假设，
而真正的问题（没做平滑）原封不动。

`_detect_issue` 是一串**有序**的 if 链，先命中先返回。这意味着：

  1. 每条判据都必须用**恰好只命中它**的指标组合去测 —— 否则前面某条
     会兜住，改坏它的人不会被抓到；
  2. **顺序本身**也是契约：`severe_overfitting` 排在 `high_turnover` 前面，
     意味着"既过拟合又高换手"的因子优先报过拟合。顺序被换掉，
     同一批指标会得到不同的首要问题。

测法与 A/B/C 一致：阈值精确打在边界上，并对"两种取值同解"的参数组合保持警惕。
"""
from __future__ import annotations

import pytest

from app.core.alpha_engine.financial_diagnostics import (
    FactorDiagnosis, FinancialDiagnostics, Suggestion,
)


DSL = "rank(ts_delta(close,5))"


def _diag(**metrics) -> FactorDiagnosis:
    """
    默认指标是一个"健康"的因子；只覆盖要测的那几项。

    默认值必须**不触发任何一条 if** —— 否则每条用例都在和别的判据抢，
    测不出单条判据的边界。
    """
    base = dict(is_sharpe=0.8, oos_sharpe=0.7, turnover=1.0,
                mean_ic=0.015, ic_ir=0.5, max_drawdown=-0.10,
                overfitting_score=0.05)
    base.update(metrics)
    return FinancialDiagnostics().diagnose(DSL, base)


def _issue(**metrics) -> str:
    return _diag(**metrics).primary_issue


# ===========================================================================
# A. 首要问题判定 —— 有序 if 链的每一条
# ===========================================================================

class TestIssueDetectionOrder:

    def test_the_default_metrics_land_on_healthy(self):
        """
        先把基线钉住：默认指标必须命中 `healthy`。
        这是后面每条边界用例的前提 —— 基线一旦漂到别的分支，
        那些用例测的就不是自己声称的那条判据了。
        """
        assert _issue() == "healthy", (
            f"默认指标命中了 {_issue()}，应当是 healthy —— "
            f"本文件所有边界用例的前提被破坏")

    @pytest.mark.parametrize("overfit,is_sharpe,expect", [
        (0.61, 0.81, "severe_overfitting"),   # 两个条件都过
        (0.60, 0.81, None),                   # overfit 恰好 0.6 → 不触发（`> 0.6`）
        (0.61, 0.80, None),                   # is_sharpe 恰好 0.8 → 不触发（`> 0.8`）
    ])
    def test_severe_overfitting_needs_both_conditions_strictly(
            self, overfit, is_sharpe, expect):
        """
        `if overfit > 0.6 and sharpe_is > 0.8:` —— 两个都是**严格大于**，
        而且是 `and`。

        放宽成 `or` 会让"IS Sharpe 高但完全没过拟合"的好因子
        被报成严重过拟合；任一 `>` 放宽成 `>=` 则让边界值误判。
        """
        got = _issue(overfitting_score=overfit, is_sharpe=is_sharpe,
                     oos_sharpe=0.2)
        if expect:
            assert got == expect, f"overfit={overfit}, is={is_sharpe} → {got}"
        else:
            assert got != "severe_overfitting", (
                f"overfit={overfit}, is_sharpe={is_sharpe} 不该判严重过拟合，"
                f"实际 {got} —— 某个 `>` 被放宽成了 `>=`")

    @pytest.mark.parametrize("turnover,expect_issue", [
        (3.0,  False),   # 恰好 3.0 不触发（`> 3.0`）
        (3.01, True),
        (5.0,  True),
    ])
    def test_the_high_turnover_threshold_is_strict(self, turnover, expect_issue):
        got = _issue(turnover=turnover)
        assert (got == "high_turnover") is expect_issue, (
            f"换手 {turnover} 的首要问题是 {got} —— `turnover > 3.0` 的边界被改了")

    @pytest.mark.parametrize("turnover,severity", [
        (3.5, "moderate"),    # 3.0 < t <= 5.0
        (5.0, "moderate"),    # 恰好 5.0 仍是 moderate（`> 5.0` 严格大于）
        (5.01, "critical"),
    ])
    def test_the_turnover_severity_escalation_boundary(self, turnover, severity):
        """
        `"critical" if turnover > 5.0 else "moderate"` —— 严重度分级。
        前端按 severity 决定是否高亮/阻断，判错会让真正危险的因子
        以"中等"的面目出现。
        """
        d = _diag(turnover=turnover)
        assert d.primary_issue == "high_turnover", "用例前提被破坏"
        assert d.severity == severity, (
            f"换手 {turnover} 的严重度是 {d.severity}，应当是 {severity} —— "
            f"`turnover > 5.0` 的边界被改了")

    @pytest.mark.parametrize("oos,ic,expect", [
        (0.05, 0.005, True),    # 两个都低
        (0.10, 0.005, False),   # oos 恰好 0.1 → 不触发（`< 0.1`）
        (0.05, 0.010, False),   # ic 恰好 0.01 → 不触发（`< 0.01`）
    ])
    def test_no_signal_needs_both_conditions_strictly(self, oos, ic, expect):
        """
        `if sharpe_oos < 0.1 and mean_ic < 0.01:` —— `and` 放宽成 `or`
        会把"IC 不高但 Sharpe 不错"的因子也报成"完全没信号"，
        用户会据此丢掉一个能用的因子。
        """
        got = _issue(oos_sharpe=oos, mean_ic=ic)
        assert (got == "no_signal") is expect, (
            f"oos={oos}, ic={ic} → {got} —— 两个严格小于的边界或 and 被改了")

    @pytest.mark.parametrize("ic,ic_ir,expect", [
        (0.021, 0.29, True),
        (0.020, 0.29, False),   # ic 恰好 0.02 → 不触发（`> 0.02`）
        (0.021, 0.30, False),   # ic_ir 恰好 0.3 → 不触发（`< 0.3`）
    ])
    def test_noisy_signal_needs_high_ic_and_low_ir(self, ic, ic_ir, expect):
        """
        `if mean_ic > 0.02 and ic_ir < 0.3:` —— "方向对但不稳定"。
        这条与 `no_signal` 的区别正在于 IC 高不高，两条判据的
        边界一旦重叠，诊断会在两者之间乱跳。
        """
        got = _issue(mean_ic=ic, ic_ir=ic_ir)
        assert (got == "noisy_signal") is expect, (
            f"ic={ic}, ic_ir={ic_ir} → {got}")

    @pytest.mark.parametrize("dd,expect", [
        (-0.26, True),
        (-0.25, False),   # 恰好 25% → 不触发（`abs(dd) > 0.25`）
        (0.26,  True),    # 正号也要认（用的是 abs）
    ])
    def test_the_drawdown_threshold_uses_absolute_value(self, dd, expect):
        """
        `if abs(max_dd) > 0.25:` —— 用绝对值，因为回撤在不同地方
        可能记成 -0.25 或 0.25。去掉 `abs` 会让负号形式（最常见的那种）
        永远不触发，这条诊断彻底失效。
        """
        got = _issue(max_drawdown=dd)
        assert (got == "high_drawdown") is expect, (
            f"回撤 {dd} → {got} —— `abs(max_dd) > 0.25` 被改了")

    @pytest.mark.parametrize("overfit,is_s,oos_s,expect", [
        (0.11, 1.0, 0.6,  True),    # 0.1 < overfit <= 0.6 且 gap > 0.3
        (0.10, 1.0, 0.6,  False),   # overfit 恰好 0.1 → 不触发（`0.1 <` 严格）
        (0.60, 1.0, 0.6,  True),    # 恰好 0.6 → 触发（`<= 0.6` 包含）
        (0.11, 1.0, 0.7,  False),   # gap 恰好 0.3 → 不触发（`>` 严格大于）
    ])
    def test_mild_overfitting_is_a_band_not_a_threshold(
            self, overfit, is_s, oos_s, expect):
        """
        `if 0.1 < overfit <= 0.6 and sharpe_is > sharpe_oos + 0.3:`

        这是一个**区间**（左开右闭）加一个间隙条件。三处边界：
        左端 `0.1 <`、右端 `<= 0.6`、间隙 `> +0.3`。
        区间写错会和 `severe_overfitting` 或 `healthy` 重叠。
        """
        got = _issue(overfitting_score=overfit, is_sharpe=is_s, oos_sharpe=oos_s)
        assert (got == "mild_overfitting") is expect, (
            f"overfit={overfit}, is={is_s}, oos={oos_s} → {got}")

    @pytest.mark.parametrize("oos,overfit,expect", [
        (0.51, 0.29, True),
        (0.50, 0.29, False),   # oos 恰好 0.5 → 不触发（`> 0.5`）
        (0.51, 0.30, False),   # overfit 恰好 0.3 → 不触发（`< 0.3`）
    ])
    def test_healthy_needs_both_good_oos_and_low_overfit(
            self, oos, overfit, expect):
        got = _issue(oos_sharpe=oos, overfitting_score=overfit)
        assert (got == "healthy") is expect, (
            f"oos={oos}, overfit={overfit} → {got}")

    def test_everything_else_falls_through_to_weak_alpha(self):
        """
        末尾的兜底 `return "weak_alpha"`。所有判据都不命中时必须落到这里，
        而不是返回 None 或抛异常 —— 前端按 `primary_issue` 取文案。
        """
        got = _issue(oos_sharpe=0.3, overfitting_score=0.05, mean_ic=0.015,
                     ic_ir=0.5, turnover=1.0, max_drawdown=-0.1)
        assert got == "weak_alpha", f"兜底分支没命中，得到 {got}"


class TestIssuePriorityOrder:
    """
    有序 if 链的**顺序**本身是契约：同时满足多条时，先命中的赢。
    把某两条的顺序对调，同一批指标会得到不同的首要问题，
    而每条判据单独看都还是对的 —— 只有跨判据的用例能抓到。
    """

    def test_severe_overfitting_outranks_high_turnover(self):
        """既严重过拟合又高换手 → 报过拟合（那是更根本的问题）。"""
        assert _issue(overfitting_score=0.8, is_sharpe=1.5, oos_sharpe=0.1,
                      turnover=6.0) == "severe_overfitting"

    def test_high_turnover_outranks_no_signal(self):
        """换手爆表 + 没信号 → 先报换手（成本吃掉了信号是更可能的解释）。"""
        assert _issue(turnover=6.0, oos_sharpe=0.0, mean_ic=0.0) == "high_turnover"

    def test_no_signal_outranks_noisy_signal(self):
        """
        两条的条件在 mean_ic 上互斥（< 0.01 vs > 0.02），构造不出同时满足，
        这里退而验证**它们的区分点**：IC 低就是没信号，IC 高就是噪声大。
        """
        assert _issue(oos_sharpe=0.05, mean_ic=0.005, ic_ir=0.1) == "no_signal"
        assert _issue(oos_sharpe=0.05, mean_ic=0.05, ic_ir=0.1) == "noisy_signal"

    def test_noisy_signal_outranks_high_drawdown(self):
        assert _issue(mean_ic=0.05, ic_ir=0.1, max_drawdown=-0.5) == "noisy_signal"

    def test_high_drawdown_outranks_mild_overfitting(self):
        assert _issue(max_drawdown=-0.5, overfitting_score=0.3,
                      is_sharpe=1.0, oos_sharpe=0.5) == "high_drawdown"

    def test_mild_overfitting_outranks_healthy(self):
        """
        IS/OOS 有明显间隙时，即使 OOS 本身不错也要先报轻度过拟合 ——
        否则用户会把一个被美化过的因子当成健康因子放大资金。
        """
        assert _issue(overfitting_score=0.3, is_sharpe=1.2,
                      oos_sharpe=0.6) == "mild_overfitting"


# ===========================================================================
# B. 改进建议
# ===========================================================================

class TestSuggestions:

    @staticmethod
    def _actions(**metrics) -> list:
        return [s.action for s in _diag(**metrics).suggestions]

    @pytest.mark.parametrize("turnover,expect", [
        (2.0,  False),   # 恰好 2.0 不触发（`> 2.0`）
        (2.01, True),
    ])
    def test_smoothing_suggestions_appear_above_the_turnover_threshold(
            self, turnover, expect):
        """
        `if turnover > 2.0:` 触发 add_smoothing + apply_decay 两条。
        注意这个阈值（2.0）**低于**首要问题的换手阈值（3.0）——
        即"还没到报警程度，但已经该提醒平滑了"。两个阈值任一被改
        都会让这段梯度失效。
        """
        acts = self._actions(turnover=turnover)
        assert ("add_smoothing" in acts) is expect, (
            f"换手 {turnover} 时建议是 {acts} —— `turnover > 2.0` 的边界被改了")

    @pytest.mark.parametrize("overfit,expect", [
        (0.50, False),   # 恰好 0.5 不触发（`> 0.5`）
        (0.51, True),
    ])
    def test_neutralize_suggestion_threshold(self, overfit, expect):
        acts = self._actions(overfitting_score=overfit, is_sharpe=0.5)
        assert ("neutralize" in acts) is expect, (
            f"过拟合 {overfit} 时建议是 {acts}")

    @pytest.mark.parametrize("ic,ic_ir,expect", [
        (0.011, 0.29, True),
        (0.010, 0.29, False),   # ic 恰好 0.01（`> 0.01`）
        (0.011, 0.30, False),   # ic_ir 恰好 0.3（`< 0.3`）
    ])
    def test_zscore_suggestion_needs_both_conditions(self, ic, ic_ir, expect):
        acts = self._actions(mean_ic=ic, ic_ir=ic_ir)
        assert ("add_zscore_ts" in acts) is expect, (
            f"ic={ic}, ic_ir={ic_ir} 时建议是 {acts}")

    def test_a_dead_momentum_factor_is_told_to_try_reversion(self):
        """
        `if sharpe_oos < 0.2 and mean_ic < 0.015:` 里按家族分支：
        动量/趋势 → 建议试反转；反转 → 建议试动量。
        家族判断错了，用户会被建议"把反转改成反转"。
        """
        from app.core.alpha_engine.financial_interpreter import FinancialInterpreter
        interp = FinancialInterpreter().interpret("rank(ts_delta(close,5))")
        d = FinancialDiagnostics().diagnose(
            "rank(ts_delta(close,5))",
            dict(is_sharpe=0.3, oos_sharpe=0.1, turnover=1.0, mean_ic=0.005,
                 ic_ir=0.2, max_drawdown=-0.1, overfitting_score=0.05),
            interpreter_result=interp,
        )
        acts = [s.action for s in d.suggestions]
        assert "try_reversion" in acts, (
            f"动量因子没信号时没有建议试反转：{acts}")
        assert "try_momentum" not in acts, (
            f"动量因子被建议『试动量』：{acts} —— 家族分支反了")

    def test_a_dead_reversion_factor_is_told_to_try_momentum(self):
        from app.core.alpha_engine.financial_interpreter import FinancialInterpreter
        interp = FinancialInterpreter().interpret("rank(ts_zscore(close,5))")
        assert interp.factor_family == "reversion", "用例前提被破坏"
        d = FinancialDiagnostics().diagnose(
            "rank(ts_zscore(close,5))",
            dict(is_sharpe=0.3, oos_sharpe=0.1, turnover=1.0, mean_ic=0.005,
                 ic_ir=0.2, max_drawdown=-0.1, overfitting_score=0.05),
            interpreter_result=interp,
        )
        acts = [s.action for s in d.suggestions]
        assert "try_momentum" in acts, f"反转因子没信号时没有建议试动量：{acts}"

    @pytest.mark.parametrize("dd,expect", [
        (-0.20, False),   # 恰好 20% 不触发（`> 0.20`）
        (-0.21, True),
    ])
    def test_regime_filter_suggestion_threshold(self, dd, expect):
        """
        注意这个阈值（0.20）与首要问题的回撤阈值（0.25）**不同** ——
        建议的门槛更低。两个数字任一被改都会让梯度塌掉。
        """
        acts = self._actions(max_drawdown=dd)
        assert ("add_regime_filter" in acts) is expect, (
            f"回撤 {dd} 时建议是 {acts} —— `abs(max_dd) > 0.20` 的边界被改了")

    def test_the_two_drawdown_thresholds_are_different(self):
        """
        把"建议阈值 0.20 < 首要问题阈值 0.25"这个梯度本身钉住。
        两个数字被改成相同，就没有"先提醒、后报警"的层次了。
        """
        mid = _diag(max_drawdown=-0.22)
        assert mid.primary_issue != "high_drawdown", (
            "22% 回撤就报了首要问题 —— 两个阈值塌成了一个")
        assert "add_regime_filter" in [s.action for s in mid.suggestions], (
            "22% 回撤没有给出 regime 建议")

    def test_priorities_are_consecutive_starting_from_one(self):
        """
        `priority` 从 1 开始、每加一条 `priority += 1`。
        前端按 priority 排序取前 N 条；重复或跳号会让排序不稳定。
        """
        d = _diag(turnover=6.0, overfitting_score=0.8, is_sharpe=1.5,
                  oos_sharpe=0.05, mean_ic=0.05, ic_ir=0.1, max_drawdown=-0.5)
        prios = [s.priority for s in d.suggestions]
        assert prios, "多重问题的因子一条建议都没给"
        assert prios == list(range(1, len(prios) + 1)), (
            f"优先级不是从 1 开始的连续序列：{prios} —— "
            f"某处 `priority += 1` 被删掉或重复了")

    def test_every_suggestion_carries_a_concrete_dsl_patch(self):
        """
        `dsl_patch` 是建议的可执行部分 —— agent 会把它喂回变异流程。
        空的 patch 等于只说"你该改改"而不说改什么。
        """
        d = _diag(turnover=6.0, overfitting_score=0.8, is_sharpe=1.5,
                  oos_sharpe=0.05, mean_ic=0.05, ic_ir=0.1, max_drawdown=-0.5)
        for s in d.suggestions:
            assert s.dsl_patch.strip(), f"建议 {s.action} 没有给出 dsl_patch"
            assert s.reason.strip(), f"建议 {s.action} 没有给出理由"
            assert s.finance_why.strip(), f"建议 {s.action} 没有给出金融依据"

    def test_the_dsl_patch_embeds_the_original_expression(self):
        """
        patch 形如 `ts_mean({dsl}, 3)` —— 必须把**原式**嵌进去。
        丢掉原式会让 patch 变成一条与用户因子无关的新表达式。
        """
        d = _diag(turnover=6.0)
        smoothing = [s for s in d.suggestions if s.action == "add_smoothing"]
        assert smoothing, "高换手时没有平滑建议"
        assert DSL in smoothing[0].dsl_patch, (
            f"平滑建议没有嵌入原式：{smoothing[0].dsl_patch}")

    def test_a_healthy_factor_is_not_buried_in_suggestions(self):
        """
        反向不变式：健康因子不该收到一堆"你该改"的建议。
        所有建议条件同时恒真时这条会红。
        """
        d = _diag()
        assert d.primary_issue == "healthy", "用例前提被破坏"
        assert len(d.suggestions) <= 1, (
            f"健康因子收到了 {len(d.suggestions)} 条改进建议："
            f"{[s.action for s in d.suggestions]}")


# ===========================================================================
# C. 市场环境提示
# ===========================================================================

class TestRegimeInsight:

    @staticmethod
    def _note(family, **metrics):
        from app.core.alpha_engine.financial_diagnostics import FinancialDiagnostics
        base = dict(sharpe_oos=0.7, turnover=1.0, mean_ic=0.015)
        base.update(metrics)
        return FinancialDiagnostics._regime_insight(family=family, **base)

    @pytest.mark.parametrize("oos,expect", [
        (0.29, True),
        (0.30, False),   # 恰好 0.3 不触发（`< 0.3`）
    ])
    def test_weak_momentum_gets_a_choppy_market_note(self, oos, expect):
        note = self._note("momentum", sharpe_oos=oos)
        assert (note is not None and "Momentum underperforms" in note) is expect, (
            f"momentum, oos={oos} → {note!r}")

    @pytest.mark.parametrize("turnover,expect", [
        (2.51, True),
        (2.50, False),   # 恰好 2.5 不触发（`> 2.5`）
    ])
    def test_high_turnover_reversion_gets_a_trending_market_warning(
            self, turnover, expect):
        note = self._note("reversion", turnover=turnover)
        assert (note is not None and "range-bound" in note) is expect, (
            f"reversion, turnover={turnover} → {note!r}")

    def test_family_and_metric_must_both_match(self):
        """
        每条判据都是 `family == X and <指标条件>`。
        `and` 放宽成 `or` 会让任意家族只要指标沾边就拿到别人的提示 ——
        比如给一个流动性因子贴上"低波动因子偏防御"的说明。
        """
        # 换手很高，但家族不是 reversion → 不该拿到 reversion 的提示
        note = self._note("liquidity", turnover=5.0, sharpe_oos=0.7, mean_ic=0.01)
        assert note is None or "range-bound" not in note, (
            f"liquidity 因子拿到了 reversion 的环境提示：{note!r}")

    def test_no_matching_rule_returns_none(self):
        """
        末尾 `return None`。改成返回字符串会让前端永远显示一条
        无关的环境说明；而 `if self.regime_insight:` 的守卫也就永远为真。
        """
        assert self._note("liquidity", sharpe_oos=0.4, turnover=1.0,
                          mean_ic=0.01) is None

    def test_strong_ic_note_is_the_last_resort(self):
        """`if mean_ic > 0.03 and sharpe_oos > 0.5:` 是最后一条兜底规则。"""
        note = self._note("liquidity", mean_ic=0.031, sharpe_oos=0.51)
        assert note is not None and "Strong IC" in note, f"{note!r}"
        # 两个边界任一不到就不给
        assert self._note("liquidity", mean_ic=0.030, sharpe_oos=0.51) is None
        assert self._note("liquidity", mean_ic=0.031, sharpe_oos=0.50) is None


# ===========================================================================
# D. 指标解析与输出契约
# ===========================================================================

class TestMetricParsing:

    def test_missing_metrics_fall_back_to_zero_not_crash(self):
        """
        `_f` 的兜底：缺键、None、非数值都要回到默认值。
        一条回测失败的记录（指标全 None）不该让诊断崩掉。
        """
        d = FinancialDiagnostics().diagnose(DSL, {})
        assert d.primary_issue, "空指标让诊断返回了空的首要问题"
        assert d.metrics_summary["is_sharpe"] == 0.0

    def test_none_valued_metrics_are_coerced(self):
        d = FinancialDiagnostics().diagnose(
            DSL, dict(is_sharpe=None, oos_sharpe=None, turnover=None,
                      mean_ic=None, ic_ir=None, max_drawdown=None,
                      overfitting_score=None))
        assert d.metrics_summary["oos_sharpe"] == 0.0

    def test_unparsable_metrics_are_coerced(self):
        """
        `except (TypeError, ValueError): return default` ——
        字符串指标（例如从 JSON 里读出来的 "N/A"）不能让诊断炸掉。
        """
        d = FinancialDiagnostics().diagnose(
            DSL, dict(is_sharpe="N/A", turnover="high", mean_ic=[1, 2]))
        assert d.metrics_summary["is_sharpe"] == 0.0
        assert d.metrics_summary["turnover"] == 0.0

    def test_a_valid_metric_is_not_replaced_by_the_default(self):
        """
        反向：兜底不能把**正常值**也吃掉。
        `return float(v) if v is not None else default` 里的判定
        若被取反，所有真实指标都会变成 0，诊断永远落在同一个分支。
        """
        d = FinancialDiagnostics().diagnose(DSL, dict(is_sharpe=1.234))
        assert d.metrics_summary["is_sharpe"] == pytest.approx(1.234), (
            "正常的指标值被兜底替换掉了")

    @pytest.mark.parametrize("key,raw,digits", [
        ("is_sharpe", 1.23456, 3),
        ("turnover", 1.23456, 2),
        ("mean_ic", 0.0123456, 4),
        ("ic_ir", 1.23456, 3),
    ])
    def test_each_metric_is_rounded_to_its_own_precision(self, key, raw, digits):
        """
        `metrics_summary` 里每个指标的小数位**各不相同**
        （turnover 2 位、mean_ic 4 位……）。位数被统一或改动，
        前端展示的精度就不对了 —— IC 只保留 2 位等于全变 0.01。
        """
        d = FinancialDiagnostics().diagnose(DSL, {key: raw})
        assert d.metrics_summary[key] == round(raw, digits), (
            f"{key} 被保留成 {d.metrics_summary[key]}，"
            f"应当是 round({raw}, {digits}) = {round(raw, digits)}")

    def test_without_an_interpreter_result_the_family_is_unknown(self):
        """
        `family = interpreter_result.factor_family if interpreter_result else "unknown"`
        —— 判定反了会在没有解释结果时去读 `None.factor_family`。
        """
        d = FinancialDiagnostics().diagnose(DSL, dict(oos_sharpe=0.05, mean_ic=0.005))
        # "unknown" 家族走不到任何家族分支，也不该崩
        assert d.primary_issue == "no_signal"
        acts = [s.action for s in d.suggestions]
        assert "try_reversion" not in acts and "try_momentum" not in acts, (
            f"家族未知时给出了按家族分支的建议：{acts}")


class TestOutputContract:

    def test_to_dict_carries_every_field(self):
        d = _diag(turnover=6.0).to_dict()
        for k in ("primary_issue", "diagnosis", "severity", "suggestions",
                  "regime_insight", "metrics_summary"):
            assert k in d, f"to_dict 缺字段 {k}：{sorted(d)}"
        assert isinstance(d["suggestions"], list)
        # 前提写成断言而不是守卫：turnover=6.0 必然触发换手类建议，
        # 守卫写法会在建议为空时**一条都不检查**。
        assert d["suggestions"], "turnover=6.0 竟然一条建议都没有 —— 用例前提被破坏"
        assert all(isinstance(x, dict) for x in d["suggestions"]), (
            "suggestions 没有被逐条 to_dict —— 前端拿到的是对象不是 JSON")

    def test_summary_shows_at_most_three_suggestions(self):
        """
        `self.suggestions[:3]` —— 摘要只列前三条。切片被改成全量会让
        一条摘要变成几十行，而它是要打进日志和聊天回复的。
        """
        d = _diag(turnover=6.0, overfitting_score=0.8, is_sharpe=1.5,
                  oos_sharpe=0.05, mean_ic=0.05, ic_ir=0.1, max_drawdown=-0.5)
        assert len(d.suggestions) > 3, "用例前提被破坏：建议不足 4 条"
        n_listed = d.summary().count("Suggestion ")
        assert n_listed == 3, f"摘要里列了 {n_listed} 条建议，应当是 3 条"

    def test_summary_omits_the_regime_line_when_there_is_none(self):
        """`if self.regime_insight:` —— 没有提示时不该打出空的 Regime Note 行。"""
        d = _diag(oos_sharpe=0.4, turnover=1.0, mean_ic=0.01,
                  overfitting_score=0.05)
        assert d.regime_insight is None, (
            f"用例前提被破坏：这组指标下不该有 regime 提示，实际是 "
            f"{d.regime_insight!r}")
        assert "Regime Note" not in d.summary(), (
            "没有 regime 提示，摘要里却打出了 Regime Note 行")

    def test_severity_is_always_one_of_the_four_levels(self):
        """
        前端按 severity 决定配色/是否阻断。出现第五种取值会走到
        没有定义的分支（通常表现为灰色/不显示）。
        """
        allowed = {"critical", "moderate", "minor", "healthy"}
        cases = [
            dict(),
            dict(turnover=6.0),
            dict(turnover=3.5),
            dict(overfitting_score=0.8, is_sharpe=1.5, oos_sharpe=0.1),
            dict(oos_sharpe=0.0, mean_ic=0.0),
            dict(mean_ic=0.05, ic_ir=0.1),
            dict(max_drawdown=-0.5),
            dict(overfitting_score=0.3, is_sharpe=1.2, oos_sharpe=0.6),
            dict(oos_sharpe=0.3),
        ]
        for c in cases:
            d = _diag(**c)
            assert d.severity in allowed, (
                f"{c} 的严重度是 {d.severity!r}，不在 {allowed} 里")

    def test_the_diagnosis_text_quotes_the_actual_metrics(self):
        """
        诊断文案里嵌了 f-string 格式化的指标值。数值没被嵌进去
        （比如格式化位数被改成 0 位）会让"IS Sharpe 1.50 但 OOS 只有 0.10"
        这种关键对比变成"1 vs 0"，读的人得不到量级感。
        """
        d = _diag(overfitting_score=0.8, is_sharpe=1.5, oos_sharpe=0.1)
        assert d.primary_issue == "severe_overfitting"
        assert "1.50" in d.diagnosis and "0.10" in d.diagnosis, (
            f"诊断文案里没有嵌入实际指标值：{d.diagnosis}")


# ===========================================================================
# E. 第二轮：首测 0% → 88.6% 之后仍存活的 4 个点
# ===========================================================================


class TestFamilyBranchBoundary:
    """
    `if sharpe_oos < 0.2 and mean_ic < 0.015:` —— 按家族给"换个方向试试"
    的建议之前的准入条件。

    上一版用的是 oos=0.1 / ic=0.005，**离边界很远**，
    `<` 改成 `<=` 完全看不出来。必须精确打在 0.2 与 0.015 上。
    """

    @staticmethod
    def _acts(oos, ic):
        from app.core.alpha_engine.financial_interpreter import FinancialInterpreter
        interp = FinancialInterpreter().interpret("rank(ts_delta(close,5))")
        d = FinancialDiagnostics().diagnose(
            "rank(ts_delta(close,5))",
            dict(is_sharpe=0.3, oos_sharpe=oos, turnover=1.0, mean_ic=ic,
                 ic_ir=0.2, max_drawdown=-0.1, overfitting_score=0.05),
            interpreter_result=interp,
        )
        return [x.action for x in d.suggestions]

    @pytest.mark.parametrize("oos,ic,expect", [
        (0.19,  0.014, True),    # 两个都在门内
        (0.20,  0.014, False),   # oos 恰好 0.2 -> 不触发（`< 0.2`）
        (0.19,  0.015, False),   # ic 恰好 0.015 -> 不触发（`< 0.015`）
    ])
    def test_the_dead_factor_branch_needs_both_conditions_strictly(
            self, oos, ic, expect):
        """
        `and` 放宽成 `or` 时，只要 OOS 低就建议"换个方向"——
        哪怕 IC 显示信号是有的（那时真正该做的是降噪，不是换假设）。
        用户会据此把一个能救的因子整个丢掉。
        """
        acts = self._acts(oos, ic)
        assert ("try_reversion" in acts) is expect, (
            f"oos={oos}, ic={ic} 时建议是 {acts} —— "
            f"`sharpe_oos < 0.2 and mean_ic < 0.015` 的边界或 and 被改了")


class TestRiskAdjustSuggestion:
    """
    `if family in ("momentum", "composite") and turnover <= 2.0:`
    —— 低换手的动量因子建议做风险调整（除以波动）。

    上一版完全没测到这条分支。
    """

    @staticmethod
    def _acts(family_dsl, turnover):
        from app.core.alpha_engine.financial_interpreter import FinancialInterpreter
        interp = FinancialInterpreter().interpret(family_dsl)
        d = FinancialDiagnostics().diagnose(
            family_dsl,
            dict(is_sharpe=0.8, oos_sharpe=0.7, turnover=turnover,
                 mean_ic=0.015, ic_ir=0.5, max_drawdown=-0.10,
                 overfitting_score=0.05),
            interpreter_result=interp,
        )
        return [x.action for x in d.suggestions], interp.factor_family

    @pytest.mark.parametrize("turnover,expect", [
        (1.0,  True),
        (2.0,  True),    # 恰好 2.0 -> 触发（`<= 2.0` 包含）
        (2.01, False),
    ])
    def test_the_risk_adjust_threshold_is_inclusive(self, turnover, expect):
        acts, fam = self._acts("rank(ts_delta(close,5))", turnover)
        assert fam == "momentum", "用例前提被破坏"
        assert ("risk_adjust" in acts) is expect, (
            f"换手 {turnover} 的动量因子建议是 {acts} —— "
            f"`turnover <= 2.0` 的边界被改了")

    def test_a_non_momentum_family_does_not_get_risk_adjust(self):
        """
        `family in ("momentum","composite") and ...` —— `and` 放宽成 `or`
        时，**任何**低换手因子都会被建议做风险调整，
        包括本来就是波动因子的那些（等于建议"把波动除以波动"）。
        """
        acts, fam = self._acts("rank(ts_std(returns,20))", 1.0)
        assert fam == "volatility", "用例前提被破坏"
        assert "risk_adjust" not in acts, (
            f"波动因子被建议做风险调整：{acts} —— "
            f"`family in (...) and turnover <= 2.0` 被放宽成了 `or`")


class TestVolatilityRegimeBoundary:
    """`if family == "volatility" and sharpe_oos > 0.5:` —— 上一版只测了动量与反转。"""

    @staticmethod
    def _note(family, **metrics):
        base = dict(sharpe_oos=0.7, turnover=1.0, mean_ic=0.015)
        base.update(metrics)
        return FinancialDiagnostics._regime_insight(family=family, **base)

    @pytest.mark.parametrize("oos,expect", [
        (0.51, True),
        (0.50, False),   # 恰好 0.5 -> 不触发（`> 0.5`）
    ])
    def test_a_working_volatility_factor_gets_the_defensive_note(self, oos, expect):
        note = self._note("volatility", sharpe_oos=oos, mean_ic=0.01)
        got = note is not None and "defensive" in note
        assert got is expect, (
            f"volatility, oos={oos} -> {note!r} —— `sharpe_oos > 0.5` 的边界被改了")

    def test_the_family_half_of_the_condition_matters(self):
        """
        `family == "volatility" and ...` —— `or` 放宽后，任何
        OOS > 0.5 的因子都会拿到"低波动因子偏防御"的说明，
        包括动量因子（说法完全相反）。
        """
        note = self._note("momentum", sharpe_oos=0.9, mean_ic=0.01)
        assert note is None or "defensive" not in note, (
            f"动量因子拿到了低波动因子的环境说明：{note!r}")
