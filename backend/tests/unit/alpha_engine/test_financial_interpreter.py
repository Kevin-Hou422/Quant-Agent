"""
alpha_engine/financial_interpreter.py —— DSL → 金融语义的翻译与因子家族分类

**此前零测试**（458 有效行，D 档首批）。它不是"锦上添花的说明文字"：

  - `factor_family` 被 **GP 的家族倾斜**消费（`_FAMILY_WEIGHT_BIASES` /
    `get_seeds_for_family` / `_COMPLEMENTARY_FAMILIES`）。分类错了，
    整代进化的算子权重和互补家族选择就全偏了，而适应度曲线看着正常。
  - `issues` / `suggestions` 是用户在前端看到的**唯一**设计诊断。
    阈值错了就变成"该提醒的不提醒、不该提醒的天天刷"。
  - `complexity` 进过拟合风险判定。

测法与 A/B/C 一致：把每个**分类判据**和每个**阈值边界**精确打在边界上，
并且对"两种取值会给出同一结论"的参数组合保持警惕 ——
分类函数里有大量 `and not xxx` 的互斥条件，选错样例就分不开。
"""
from __future__ import annotations

import pytest

from app.core.alpha_engine.financial_interpreter import (
    FACTOR_FAMILIES, FinancialInterpreter, InterpretResult,
)
import app.core.alpha_engine.financial_interpreter as FI


def _it(dsl: str) -> InterpretResult:
    return FinancialInterpreter().interpret(dsl)


def _fam(dsl: str) -> str:
    return _it(dsl).factor_family


# ===========================================================================
# A. 因子家族分类 —— GP 的家族倾斜直接吃这个结果
# ===========================================================================

class TestFamilyClassification:
    """
    `_classify_family` 是一串有序的互斥判定。每条判据都要单独打中，
    而且要**排除掉别的判据也能给出同样答案**的样例 ——
    否则改坏一条判据，另一条会兜住，变异测不出来。
    """

    @pytest.mark.parametrize("dsl,expect", [
        # 纯动量：ts_delta，无 volume / 无 ts_std / 无 ts_corr
        ("rank(ts_delta(close,5))",                 "momentum"),
        ("rank(ts_rank(close,10))",                 "momentum"),
        # 长窗动量 → 趋势跟随（`max(windows) >= 60`）
        ("rank(ts_delta(close,60))",                "trend_following"),
        ("rank(ts_delta(close,120))",               "trend_following"),
        # 纯 ts_mean（无 delta/std/corr）→ 趋势跟随
        ("rank(ts_mean(close,20))",                 "trend_following"),
        # 波动：ts_std / ts_var，无动量
        ("rank(ts_std(returns,20))",                "volatility"),
        ("rank(ts_var(returns,20))",                "volatility"),
        # 流动性：用到 volume，无动量、无 corr
        ("rank(ts_mean(volume,20))",                "liquidity"),
        # 价量相关：ts_corr + volume，无动量
        ("rank(ts_corr(close,volume,20))",          "price_volume_corr"),
        # 质量：熵/偏度/峰度，无动量
        ("rank(ts_skew(returns,20))",               "quality"),
        ("rank(ts_kurt(returns,20))",               "quality"),
        ("rank(ts_entropy(returns,20))",            "quality"),
        # 反转：ts_zscore 且无动量
        ("rank(ts_zscore(close,5))",                "reversion"),
    ])
    def test_each_single_family_signal_is_classified(self, dsl, expect):
        got = _fam(dsl)
        assert got == expect, (
            f"{dsl} 被分到了 {got}，应当是 {expect} —— "
            f"GP 会按错误的家族去做算子倾斜与互补家族选择")

    def test_a_negated_momentum_is_reversion_not_momentum(self):
        """
        `if has_momentum and _is_inverted_momentum(node): return "reversion"`

        取负的动量在金融上是**反转**信号（买跌卖涨），与动量恰好相反。
        分错的后果是 GP 给它配了动量家族的互补因子（波动/流动性），
        而反转真正需要的互补是流动性 —— 组合逻辑整个错位。

        注意用的是 DSL 的**一元负号** `-x`（解析成 `neg` 节点）。
        `(0-x)` 解析成 `sub`，`_is_inverted_momentum` 认不出来 ——
        见缺陷 D-1（两种语义等价的取负写法拿到不同家族）。
        """
        assert _fam("rank(-ts_delta(close,5))") == "reversion", (
            "取负的短期动量没有被识别为反转")
        assert _fam("rank(-ts_rank(close,10))") == "reversion"
        # 对照：不取负就是动量
        assert _fam("rank(ts_delta(close,5))") == "momentum"

    def test_the_negation_detector_only_looks_at_the_dominant_signal(self):
        """
        `_is_inverted_momentum` 要求被取负的子树里含 ts_delta/ts_rank。
        取负一个**波动**因子不是反转，是低波动（quality/volatility）。
        """
        assert _fam("rank(-ts_std(returns,20))") != "reversion", (
            "取负的波动因子被误判为反转")

    def test_two_distinct_family_signals_make_it_composite(self):
        """
        `if len(family_signals) >= 2: return "composite"`

        **严格大于等于 2**。这条在所有单家族判定**之前**，
        所以它一旦失效，复合因子会被误判成其中某一个单一家族，
        GP 会拿单家族的偏好去引导一个本来已经多元的因子。
        """
        # 动量 + 波动
        assert _fam("rank((ts_delta(close,5)/ts_std(returns,20)))") == "composite"
        # 动量 + 质量
        assert _fam("rank((ts_delta(close,5)+ts_skew(returns,20)))") == "composite"

    def test_exactly_one_signal_is_not_composite(self):
        """边界的另一侧：只有一个家族信号时**不能**判成 composite。"""
        for dsl in ("rank(ts_delta(close,5))", "rank(ts_std(returns,20))",
                    "rank(ts_skew(returns,20))"):
            assert _fam(dsl) != "composite", f"{dsl} 被误判为 composite"

    def test_volume_with_momentum_is_not_liquidity(self):
        """
        `elif has_volume_fld and not has_momentum:` —— 那个 `not has_momentum`
        是互斥条件。删掉它，**带成交量确认的动量因子**会被归进流动性家族，
        于是 GP 给它配的互补家族是"动量/反转"，而它自己已经是动量了。
        """
        got = _fam("rank((ts_delta(close,5)*ts_mean(volume,20)))")
        assert got != "liquidity", (
            f"带 volume 的动量因子被归进了 liquidity（实际 {got}）—— "
            f"`and not has_momentum` 的互斥条件失效")

    def test_corr_without_volume_is_not_price_volume_corr(self):
        """
        `if has_corr and has_volume_fld` —— 两个条件缺一不可。
        `ts_corr(close, high)` 里没有成交量，不是价量相关。
        """
        got = _fam("rank(ts_corr(close,high,20))")
        assert got != "price_volume_corr", (
            f"不含成交量的相关性因子被归进了 price_volume_corr（{got}）")

    @pytest.mark.parametrize("window,expect", [
        (40, "momentum"),          # < 60
        (59, "momentum"),          # 边界下
        (60, "trend_following"),   # 恰好 60 —— `>= 60` 严格包含
        (61, "trend_following"),
    ])
    def test_the_trend_following_window_boundary_is_inclusive(self, window, expect):
        """
        `if windows and max(windows) >= 60: return "trend_following"`

        恰好 60 天必须算趋势跟随。这个边界决定同一条 DSL 拿到的是
        动量家族的偏好还是趋势家族的偏好 —— 两者的算子权重不同。
        """
        got = _fam(f"rank(ts_delta(close,{window}))")
        assert got == expect, (
            f"窗口 {window} 天被分到 {got}，应当是 {expect} —— "
            f"`max(windows) >= 60` 的边界被改了")

    def test_every_returned_family_is_a_known_family(self):
        """
        分类结果必须落在 `FACTOR_FAMILIES` 里，否则
        `FACTOR_FAMILIES.get(family, family)` 会把家族名本身当成说明文字，
        而下游 `_FAMILY_WEIGHT_BIASES[family]` 会 KeyError 或静默取不到偏好。
        """
        samples = [
            "rank(ts_delta(close,5))", "rank(ts_std(returns,20))",
            "rank(ts_mean(volume,20))", "rank(ts_corr(close,volume,20))",
            "rank(ts_skew(returns,20))", "rank(ts_zscore(close,5))",
            "rank(ts_delta(close,120))", "rank((0-ts_delta(close,5)))",
            "rank((ts_delta(close,5)/ts_std(returns,20)))",
            "close", "rank(close)", "(close+volume)",
        ]
        for dsl in samples:
            r = _it(dsl)
            assert r.factor_family in FACTOR_FAMILIES, (
                f"{dsl} 的家族 {r.factor_family!r} 不在 FACTOR_FAMILIES 里 —— "
                f"下游按家族取偏好时会取不到")
            assert r.family_desc != r.factor_family, (
                f"{dsl} 的 family_desc 退化成了家族名本身 —— "
                f"`FACTOR_FAMILIES.get(family, family)` 走了兜底分支")


# ===========================================================================
# B. 结构采集
# ===========================================================================

class TestCollectors:

    def test_fields_are_collected_from_the_whole_tree(self):
        r = _it("rank((ts_delta(close,5)*ts_mean(volume,20)))")
        assert r.data_fields == ["close", "volume"], (
            f"字段采集不全：{r.data_fields}")
        assert r.has_volume is True

    def test_has_volume_is_false_without_volume(self):
        """`has_volume = "volume" in fields` —— 它决定"缺量能确认"那条告警。"""
        assert _it("rank(ts_delta(close,5))").has_volume is False

    def test_windows_span_the_whole_tree(self):
        """
        `max_w = max(windows) if windows else 0` / `min_w = min(...)`
        两个都要对：max 进"窗口过长"告警，min 进摘要展示。
        """
        r = _it("rank((ts_delta(close,3)+ts_mean(close,90)))")
        assert (r.min_window, r.max_window) == (3, 90), (
            f"窗口区间算成了 {(r.min_window, r.max_window)}，应当是 (3, 90)")

    def test_a_tree_without_time_series_ops_has_zero_windows(self):
        """
        `max(windows) if windows else 0` —— 空列表时的兜底。
        改成 `max(windows) if not windows else 0` 会在有窗口时返回 0
        （所有窗口告警失效），无窗口时 `max([])` → ValueError。
        """
        r = _it("rank(close)")
        assert (r.min_window, r.max_window) == (0, 0)

    def test_ts_and_cs_operator_lists_are_deduplicated_and_sorted(self):
        r = _it("rank((ts_delta(close,5)+ts_delta(volume,5)))")
        assert r.operators_ts == ["ts_delta"], (
            f"TS 算子列表没有去重/排序：{r.operators_ts}")
        assert r.operators_cs == ["rank"]

    def test_second_child_of_two_input_ops_is_collected(self):
        """`ts_corr(close, w, volume)` 的第二个操作数也要进字段集合。"""
        r = _it("rank(ts_corr(close,volume,20))")
        assert set(r.data_fields) == {"close", "volume"}, (
            f"两输入算子的 second_child 没被采集：{r.data_fields}")


# ===========================================================================
# C. 归一化判定 —— 决定最主要的那条设计告警
# ===========================================================================

class TestNormalizationDetection:
    """
    `_has_cs_at_root` 只认**根节点**或**根为算术时的直接子节点**上的
    rank/zscore/scale/normalize。它决定 `is_normalized`，
    而 `is_normalized` 触发"缺截面归一化"这条最常见的告警。
    """

    @pytest.mark.parametrize("dsl", [
        "rank(ts_delta(close,5))",
        "zscore(ts_delta(close,5))",
        "scale(ts_delta(close,5))",
        "normalize(ts_delta(close,5))",
    ])
    def test_a_cs_op_at_the_root_counts_as_normalized(self, dsl):
        assert _it(dsl).is_normalized is True, f"{dsl} 没被识别为已归一化"

    def test_winsorize_at_the_root_does_not_count(self):
        """
        `node.op in ("rank", "zscore", "scale", "normalize")` ——
        `winsorize` 只裁极值，**不做**截面归一化。
        把它算进来会让"缺归一化"的告警对一批真的缺归一化的因子失效。
        """
        assert _it("winsorize(ts_delta(close,5))").is_normalized is False, (
            "winsorize 被当成了截面归一化")

    def test_a_nested_cs_op_does_not_count_as_root_level(self):
        """
        归一化必须在**输出端**。`ts_mean(rank(close), 5)` 的 rank 在内层，
        输出仍带宇宙级偏置 —— 不能算已归一化。
        """
        assert _it("ts_mean(rank(close),5)").is_normalized is False, (
            "内层的 rank 被当成了输出端归一化")

    def test_an_arithmetic_root_with_a_direct_cs_child_counts(self):
        """
        `if isinstance(node, ArithmeticNode): return any(直接子节点是 rank/zscore)`
        —— 只认 rank/zscore 两个（不含 scale/normalize），且只看**直接**子节点。
        """
        assert _it("(rank(close)+rank(volume))").is_normalized is True
        assert _it("(close+volume)").is_normalized is False

    def test_the_missing_normalization_issue_fires_exactly_when_expected(self):
        """
        `if not is_normalized:` —— 删掉 `not` 会反过来：
        **已经归一化**的因子被告警"缺归一化"，而真的缺归一化的不告警。
        这是前端最常见的那条提示，反了会让人完全不信这套诊断。
        """
        norm = _it("rank(ts_delta(close,20))")
        raw = _it("ts_delta(close,20)")
        msg = "No cross-sectional normalization"
        assert not any(msg in i for i in norm.issues), (
            f"已归一化的因子仍被告警缺归一化：{norm.issues}")
        assert any(msg in i for i in raw.issues), (
            f"未归一化的因子没有被告警：{raw.issues}")


# ===========================================================================
# D. 复杂度分档
# ===========================================================================

class TestComplexity:
    """
    `_compute_complexity` 是四级阈值（3 / 6 / 10 / 16 个节点）。
    complexity ≥ 4 会触发"过拟合风险"告警，所以每个档位的边界都要钉住。
    """

    @staticmethod
    def _nodes(dsl: str) -> int:
        from app.core.alpha_engine.parser import Parser
        return FI._count_nodes(Parser().parse(dsl))

    @pytest.mark.parametrize("dsl,expect_score", [
        ("close",                                  1),   # 1 个节点
        ("rank(close)",                            1),   # 2 个
        ("rank(ts_delta(close,5))",                1),   # 3 个（含 ScalarNode? 见下）
    ])
    def test_small_trees_are_complexity_one(self, dsl, expect_score):
        got = _it(dsl).complexity
        n = self._nodes(dsl)
        assert got == expect_score, (
            f"{dsl}（{n} 个节点）的复杂度是 {got}，应当是 {expect_score}")

    def test_complexity_is_monotone_in_tree_size(self):
        """
        四个阈值组成一条**单调不减**的阶梯。任一阈值被改动（`<=` 改 `<`、
        数字被改）都会在某处打破单调性。
        """
        ladder = [
            "close",
            "rank(ts_delta(close,5))",
            "rank((ts_delta(close,5)+ts_mean(volume,20)))",
            "rank((ts_delta(close,5)+ts_mean(volume,20)+ts_std(returns,10)))",
            "rank((ts_delta(close,5)+ts_mean(volume,20)+ts_std(returns,10)"
            "+ts_corr(close,volume,20)))",
        ]
        scores = [_it(d).complexity for d in ladder]
        sizes = [self._nodes(d) for d in ladder]
        assert scores == sorted(scores), (
            f"复杂度随树规模非单调：规模 {sizes} → 分数 {scores}")
        assert 1 <= min(scores) and max(scores) <= 5, f"复杂度越出 1–5：{scores}"

    def test_the_complexity_thresholds_are_exact(self):
        """
        直接对 `_compute_complexity` 的四个阈值打边界 ——
        用 DSL 很难精确凑出 3/6/10/16 个节点，所以构造**桩节点**。
        """
        class _Stub:
            def __init__(self, n, d=1):
                self._n, self._d = n, d

            def node_count(self):
                return self._n

            def depth(self):
                return self._d

            def children(self):
                return []

        cases = [(3, 1), (4, 2), (6, 2), (7, 3), (10, 3), (11, 4),
                 (16, 4), (17, 5)]
        for n, expect in cases:
            got = FI._compute_complexity(_Stub(n))
            assert got == expect, (
                f"{n} 个节点算出复杂度 {got}，应当是 {expect} —— "
                f"四级阈值（3/6/10/16）被改动了")

    def test_high_complexity_raises_the_overfitting_issue(self):
        """
        `if complexity >= 4:` —— 恰好 4 就该告警（不是 5）。

        【这条原来写成 `if r4.complexity >= 4: assert ...`，而当时选的
        DSL 复杂度只有 **3** —— 守卫从不成立，用例什么都没检查。
        现在把"复杂度确实是 4"写成断言，再断言告警，任一侧变了都会红。】
        """
        r4 = _it("rank((ts_delta(close,5)+ts_mean(volume,20)"
                 "+ts_std(returns,10)+ts_rank(close,15)))")
        assert r4.complexity == 4, (
            f"用例前提被破坏：这条 DSL 的复杂度是 {r4.complexity}，不是 4 —— "
            f"换一条更复杂的种子，否则本用例测不到告警分支")
        assert any("High complexity" in i for i in r4.issues), (
            f"复杂度 4 却没有过拟合告警：{r4.issues}")

        simple = _it("rank(close)")
        assert simple.complexity < 4
        assert not any("High complexity" in i for i in simple.issues), (
            f"简单因子被告警高复杂度：{simple.issues}")


# ===========================================================================
# E. 设计告警的阈值
# ===========================================================================

class TestDesignIssueThresholds:

    @pytest.mark.parametrize("window,should_warn", [
        (120, False),   # 恰好 120 不告警（`> 120` 严格大于）
        (121, True),
        (252, True),
    ])
    def test_the_long_window_warning_boundary_is_strict(self, window, should_warn):
        """
        `if max_window > 120:` —— **严格大于**。放宽成 `>=` 会把
        恰好 120 天（半年，常用）的因子也标成"回看过拟合"。
        """
        r = _it(f"rank(ts_mean(close,{window}))")
        got = any("Very long window" in i for i in r.issues)
        assert got is should_warn, (
            f"窗口 {window} 天 {'应当' if should_warn else '不应'}告警过长，"
            f"实际 {'告警了' if got else '没告警'} —— `> 120` 的边界被改了")

    @pytest.mark.parametrize("window,should_warn", [
        (3, True),      # < 5
        (4, True),
        (5, False),     # 恰好 5 不告警（`< 5` 严格小于）
        (10, False),
    ])
    def test_the_short_window_warning_boundary_is_strict(self, window, should_warn):
        """
        `if max_window > 0 and max_window < 5 and family != "reversion":`
        —— 三个条件缺一不可：
          `> 0` 排除"无窗口"的因子（否则 rank(close) 也被说"窗口过短"）
          `< 5` 是噪声门槛
          `!= "reversion"` 因为反转本来就该用短窗
        """
        r = _it(f"rank(ts_mean(close,{window}))")
        got = any("Very short window" in i for i in r.issues)
        assert got is should_warn, (
            f"窗口 {window} 天 {'应当' if should_warn else '不应'}告警过短，"
            f"实际 {'告警了' if got else '没告警'}")

    def test_a_windowless_factor_is_not_warned_as_short_window(self):
        """`max_window > 0` 这一半：无时序算子的因子 max_window=0，不该告警。"""
        r = _it("rank(close)")
        assert r.max_window == 0
        assert not any("Very short window" in i for i in r.issues), (
            f"无窗口的因子被告警『窗口过短』：{r.issues} —— "
            f"`max_window > 0` 的守卫失效")

    def test_reversion_is_exempt_from_the_short_window_warning(self):
        """`family != "reversion"` 这一半：反转因子用短窗是设计使然。"""
        r = _it("rank(ts_zscore(close,3))")
        assert r.factor_family == "reversion", "用例前提被破坏"
        assert not any("Very short window" in i for i in r.issues), (
            f"反转因子被告警『窗口过短』：{r.issues} —— 家族豁免失效")

    def test_momentum_without_smoothing_is_warned(self):
        """
        `if family in ("momentum","trend_following") and "ts_mean" not in ops
            and "ts_decay_linear" not in ops:`

        两个 `not in` 缺一不可 —— 用 ts_decay_linear 平滑的动量也算平滑过了。
        """
        raw = _it("rank(ts_delta(close,20))")
        assert any("without smoothing" in i for i in raw.issues), (
            f"未平滑的动量没有被告警：{raw.issues}")

        smoothed = _it("rank(ts_mean(ts_delta(close,20),5))")
        assert not any("without smoothing" in i for i in smoothed.issues), (
            f"已用 ts_mean 平滑的动量仍被告警：{smoothed.issues}")

        decayed = _it("rank(ts_decay_linear(ts_delta(close,20),5))")
        assert not any("without smoothing" in i for i in decayed.issues), (
            f"已用 ts_decay_linear 平滑的动量仍被告警：{decayed.issues} —— "
            f"第二个 `not in ops` 被删掉了")

    @pytest.mark.parametrize("window,should_warn", [
        (3, True), (5, True),      # <= 5
        (6, False), (20, False),
    ])
    def test_short_momentum_needs_volume_confirmation(self, window, should_warn):
        """
        `if family == "momentum" and max_window <= 5 and not has_volume:`
        —— `<= 5` **包含** 5。放宽/收紧都会改变"该提醒缺量确认"的范围。
        """
        r = _it(f"rank(ts_delta(close,{window}))")
        got = any("volume confirmation" in i for i in r.issues)
        assert got is should_warn, (
            f"窗口 {window} 的动量 {'应当' if should_warn else '不应'}"
            f"提醒缺量确认，实际 {'提醒了' if got else '没提醒'} —— "
            f"`max_window <= 5` 的边界被改了")

    def test_momentum_with_volume_is_not_warned_about_volume(self):
        """`not has_volume` 这一半。"""
        r = _it("rank((ts_delta(close,3)*ts_mean(volume,20)))")
        assert not any("volume confirmation" in i for i in r.issues), (
            f"已带成交量的因子仍被提醒缺量确认：{r.issues}")

    def test_unnormalized_volatility_is_warned(self):
        """`if family == "volatility" and not is_normalized:`"""
        raw = _it("ts_std(returns,20)")
        assert raw.factor_family == "volatility", "用例前提被破坏"
        assert any("not cross-sectionally normalized" in i for i in raw.issues), (
            f"未归一化的波动因子没有被告警：{raw.issues}")

        norm = _it("rank(ts_std(returns,20))")
        assert not any("not cross-sectionally normalized" in i for i in norm.issues)

    def test_every_issue_comes_with_a_suggestion(self):
        """
        `issues` 与 `suggestions` 是成对 append 的（除了"无条件建议"那一条）。
        只报问题不给改法，对使用者等于没有信息。
        """
        dsls = ("ts_delta(close,3)", "ts_std(returns,20)",
                "rank(ts_mean(close,252))", "rank(ts_delta(close,1))")
        with_issues = 0
        for dsl in dsls:
            r = _it(dsl)
            # 守卫改成计数 + 断言：原来写成 `if r.issues:`，
            # 万一四条全都不报问题，这个用例会静默全过。
            if r.issues:
                with_issues += 1
                assert r.suggestions, (
                    f"{dsl} 报了 {len(r.issues)} 个问题但一条建议都没给")
        assert with_issues == len(dsls), (
            f"{len(dsls)} 条刻意有毛病的 DSL 里只有 {with_issues} 条被挑出问题 —— "
            f"告警条件整体失效了，本用例失去区分力")

    def test_a_clean_factor_has_no_issues(self):
        """
        反向不变式：一条设计良好的因子不该被挑出问题。
        所有告警条件同时失效（恒告警）时这条会红。
        """
        r = _it("rank(ts_mean(ts_delta(close,20),5))")
        assert not r.issues, (
            f"设计良好的因子被挑出问题：{r.issues} —— 某条告警条件恒真")


# ===========================================================================
# F. 描述文案
# ===========================================================================

class TestDescriptions:
    """
    描述文案是用户理解因子的唯一入口。这里不逐字比对全文
    （那会让任何措辞调整都变红），只钉住**语义关键词**与
    **随参数变化的分档**。
    """

    @pytest.mark.parametrize("window,keyword", [
        (1,  "1-day price change"),
        (3,  "short-term momentum"),
        (5,  "short-term momentum"),      # 恰好 5 仍是 short（`<= 5`）
        (6,  "medium-term momentum"),
        (20, "medium-term momentum"),     # 恰好 20 仍是 medium（`<= 20`）
        (21, "long-term trend"),
    ])
    def test_ts_delta_description_tiers(self, window, keyword):
        """
        `ts_delta` 的四档描述（1 / ≤5 / ≤20 / else）。
        两个边界（5 与 20）各是一个变异点，改了会让
        "20 日动量"被说成"长期趋势"，用户据此理解的时间尺度就错了。
        """
        d = _it(f"ts_delta(close,{window})").description
        assert keyword in d, f"窗口 {window} 的描述是 {d!r}，应当含 {keyword!r}"

    @pytest.mark.parametrize("window,keyword", [
        (3,  "fast moving average"),
        (5,  "fast moving average"),      # 恰好 5
        (6,  "medium moving average"),
        (20, "medium moving average"),    # 恰好 20
        (21, "long-term trend line"),
    ])
    def test_ts_mean_description_tiers(self, window, keyword):
        d = _it(f"ts_mean(close,{window})").description
        assert keyword in d, f"窗口 {window} 的描述是 {d!r}，应当含 {keyword!r}"

    def test_field_names_are_humanised(self):
        """`_FIELD_NAMES` 的映射：用户看到的是"closing price"而不是"close"。"""
        assert "closing price" in _it("close").description
        assert "trading volume" in _it("volume").description
        assert "volume-weighted average price" in _it("vwap").description

    def test_an_unknown_field_falls_back_to_its_raw_name(self):
        """`_FIELD_NAMES.get(node.field, node.field)` 的兜底分支。"""
        from app.core.alpha_engine.typed_nodes import DataNode
        assert FI._describe(DataNode("sector")) == "sector"

    def test_the_lagged_description_pluralises_correctly(self):
        """`'s' if w > 1 else ''` —— **严格大于 1**。"""
        assert "lagged 1 day" in _it("ts_delay(close,1)").description
        assert "lagged 1 days" not in _it("ts_delay(close,1)").description
        assert "lagged 5 days" in _it("ts_delay(close,5)").description

    def test_two_input_ops_describe_both_operands(self):
        d = _it("ts_corr(close,volume,20)").description
        assert "closing price" in d and "trading volume" in d, (
            f"两输入算子的描述漏了操作数：{d}")

    def test_an_unknown_ts_op_falls_back_to_a_generic_form(self):
        """
        `return f"{op}({child}, {w})"` 是 `_describe_ts` 的兜底。
        新增算子忘了写描述时，这里给出可读的降级文案而不是崩掉。
        """
        from app.core.alpha_engine.typed_nodes import DataNode, TimeSeriesNode
        node = TimeSeriesNode("ts_momentum_decay", DataNode("close"), 7)
        d = FI._describe(node)
        assert "ts_momentum_decay" in d and "7" in d, (
            f"未知 TS 算子的兜底描述不可读：{d!r}")

    def test_the_scalar_description_drops_a_trailing_zero(self):
        """
        `str(int(v)) if v == int(v) else str(round(v, 3))`
        —— 整数值不显示 `.0`；非整数保留 3 位。
        """
        from app.core.alpha_engine.typed_nodes import ScalarNode
        assert FI._describe(ScalarNode(5.0)) == "5"
        assert FI._describe(ScalarNode(0.5)) == "0.5"
        assert FI._describe(ScalarNode(1.23456)) == "1.235"


# ===========================================================================
# G. 输出契约
# ===========================================================================

class TestOutputContract:

    def test_to_dict_carries_every_field_the_frontend_reads(self):
        d = _it("rank(ts_delta(close,5))").to_dict()
        for k in ("description", "factor_family", "family_desc", "data_fields",
                  "max_window", "min_window", "is_normalized", "has_volume",
                  "complexity", "issues", "suggestions"):
            assert k in d, f"to_dict 缺字段 {k}：{sorted(d)}"

    def test_summary_is_human_readable_and_mentions_the_family(self):
        r = _it("rank(ts_delta(close,5))")
        s = r.summary()
        assert r.factor_family.upper() in s
        assert str(r.complexity) in s
        assert "Windows:" in s and "Cross-sectional normalized:" in s

    def test_summary_omits_empty_sections(self):
        """
        `if self.issues:` / `if self.suggestions:` —— 删掉守卫会打出
        "Issues: " 这样的空行，前端展示出一个空的问题区块。
        """
        clean = _it("rank(ts_mean(ts_delta(close,20),5))")
        assert not clean.issues, "用例前提被破坏：这条因子被挑出了问题"
        assert "Issues:" not in clean.summary()

    def test_interpret_node_works_without_a_dsl_string(self):
        """
        `dsl = dsl or repr(node)` —— 直接传 AST（GP 内部路径）时
        没有原始字符串，要用 repr 兜底，不能留空。
        """
        from app.core.alpha_engine.parser import Parser
        node = Parser().parse("rank(ts_delta(close,5))")
        r = FinancialInterpreter().interpret_node(node)
        assert r.dsl, "interpret_node 不带 dsl 时 dsl 字段为空"
        assert "rank" in r.dsl


# ===========================================================================
# H. 第二轮：首测 0% → 64.6% 之后仍存活的 17 个点
# ===========================================================================
#
# 这 17 个全都是**我自己的断言选错了观察面**，而不是没写用例。
# 共同的毛病：用 `!=`（否定断言）去测互斥条件 ——
# 变异把结果从 A 改成 C 时，`!= B` 照样成立。
# A/B/C 三轮反复踩的就是这个坑，这里逐条改成**确切值**断言。


class TestFamilyExactValues:
    """把上面那些 `assert fam != X` 改写成 `assert fam == 具体值`。"""

    def test_volume_plus_momentum_is_exactly_momentum(self):
        """
        `elif has_volume_fld and not has_momentum:` 删掉 `not` 之后，
        带成交量的动量因子会被 append "liquidity" →
        family_signals 变成 [momentum, liquidity] → **composite**。

        上一版断言的是 `!= "liquidity"`，而 composite 也满足 `!=`，
        所以那个变异活了下来。必须断确切值。
        """
        assert _fam("rank((ts_delta(close,5)*ts_mean(volume,20)))") == "momentum", (
            "带成交量的动量因子不再被判为 momentum —— "
            "`has_volume_fld and not has_momentum` 的互斥条件被改了")

    def test_corr_without_volume_does_not_add_a_family_signal(self):
        """
        `if has_corr and has_volume_fld: family_signals.append("price_volume_corr")`

        `and` 放宽成 `or` 时，只要有 ts_corr（哪怕不含成交量）就会
        多 append 一个家族信号。单独看不出来（len 仍是 1），
        必须让这个多出来的信号**把 len 推到 2** 才分得开 ——
        用 "ts_corr(close,high) + ts_std"：
          正确：family_signals = [volatility]        -> volatility
          `or`：family_signals = [volatility, pvc]   -> composite
        """
        dsl = "rank((ts_corr(close,high,20)+ts_std(returns,20)))"
        assert _fam(dsl) == "volatility", (
            f"{dsl} 不再被判为 volatility —— "
            f"`has_corr and has_volume_fld` 被放宽成了 `or`，"
            f"多出来的家族信号把它推成了 composite")

    def test_a_pure_rolling_op_without_ts_mean_is_not_trend_following(self):
        """
        `if "ts_mean" in ops and not (ops & {"ts_delta","ts_std","ts_corr"}):
             return "trend_following"`

        `and` 放宽成 `or` 时，**任何**不含 delta/std/corr 的因子都会被
        判成趋势跟随 —— 哪怕它根本没有 ts_mean。
        `ts_max` 正好能走到这一行且不含 ts_mean。
        """
        assert _fam("rank(ts_max(close,20))") == "momentum", (
            "不含 ts_mean 的因子被判成了 trend_following —— "
            "`'ts_mean' in ops and not (...)` 被放宽成了 `or`")
        # 对照：真的含 ts_mean 时才是趋势跟随
        assert _fam("rank(ts_mean(close,20))") == "trend_following"

    def test_a_negation_wrapping_a_non_momentum_subtree_is_not_reversion(self):
        """
        `return len(ch) > 0 and _has_op(ch[0], {"ts_delta", "ts_rank"})`

        `and` 放宽成 `or` 时，只要 neg 节点**有子节点**就判成"取负的动量"，
        而不管被取负的到底是不是动量。

        构造：`ts_delta(close,5) + -ts_mean(close,20)` ——
        整棵树有动量（ts_delta），但 neg 包的是 ts_mean。
          正确：neg 的子树里没有 delta/rank -> 不是反转 -> momentum
          `or`：len(ch) > 0 为真 -> 判成反转
        """
        assert _fam("rank((ts_delta(close,5)+-ts_mean(close,20)))") == "momentum", (
            "取负的不是动量子树，却被判成了反转 —— "
            "`len(ch) > 0 and _has_op(...)` 被放宽成了 `or`")

    def test_a_childless_negation_node_does_not_crash(self):
        """
        `len(ch) > 0` 放宽成 `>= 0` 时，零子节点的 neg 节点会走到
        `ch[0]` -> IndexError。

        `ArithmeticNode` 没有元数校验（见缺陷 C-2 的讨论），
        GP 的程序化拼接确实能产出这种节点。
        """
        from app.core.alpha_engine.typed_nodes import ArithmeticNode
        node = ArithmeticNode("neg", [])
        assert FI._is_inverted_momentum(node) is False, (
            "零子节点的 neg 节点没有被 `len(ch) > 0` 挡住")

    def test_has_op_short_circuits_on_nodes_without_an_op_attribute(self):
        """
        `if hasattr(node, "op") and node.op in ops:` —— `and` 放宽成 `or`
        时，没有 `op` 属性的节点（DataNode 就没有）会在第二个子式上
        AttributeError，整条解释链崩掉。
        """
        from app.core.alpha_engine.typed_nodes import DataNode
        leaf = DataNode("close")
        assert not hasattr(leaf, "op"), "用例前提被破坏：DataNode 现在有 op 了"
        assert FI._has_op(leaf, {"ts_delta"}) is False, (
            "没有 op 属性的节点让 _has_op 崩了 —— `hasattr(...) and ...` 被放宽")


class TestNodeCountingHelpers:
    """
    `_count_nodes` / `_tree_depth` 是给**没有** `node_count()` / `depth()`
    的节点用的回退实现。typed_nodes 有 `depth()` 但**没有** `node_count()`，
    所以 `_count_nodes` 是真在用的，`_tree_depth` 只走回退路径。
    """

    def test_count_nodes_adds_one_per_level(self):
        """
        `return 1 + sum(_count_nodes(c) for c in node.children())`
        —— `+` 改成 `-` 会让节点数变成负数，复杂度永远落在最低档，
        "高复杂度过拟合风险"的告警彻底失效。

        上一版的复杂度用例全部用 `_Stub` 直接喂 `node_count()`，
        绕过了这个函数，所以没测到。
        """
        from app.core.alpha_engine.parser import Parser
        p = Parser()
        assert FI._count_nodes(p.parse("close")) == 1
        assert FI._count_nodes(p.parse("rank(close)")) == 2
        sizes = [FI._count_nodes(p.parse(d)) for d in (
            "close",
            "rank(close)",
            "rank(ts_delta(close,5))",
            "rank((ts_delta(close,5)+ts_mean(volume,20)))",
        )]
        assert all(n > 0 for n in sizes), f"节点数出现非正值：{sizes}"
        assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes), (
            f"节点数不随树增大而严格递增：{sizes} —— `1 + sum(...)` 的算符被改了")

    def test_the_complexity_of_a_real_tree_tracks_its_node_count(self):
        """
        端到端把 `_count_nodes` 接回复杂度：`+`->`-` 时节点数变负，
        所有因子的复杂度都会掉到 1。
        """
        big = _it("rank((ts_delta(close,5)+ts_mean(volume,20)+ts_std(returns,10)"
                  "+ts_corr(close,volume,20)))")
        assert big.complexity >= 3, (
            f"一棵大树的复杂度只有 {big.complexity} —— "
            f"`_count_nodes` 的 `1 + sum(...)` 疑似被改成了减法")

    def test_tree_depth_fallback_on_nodes_without_a_depth_method(self):
        """
        `_tree_depth` 的两处变异（`if not ch` 与 `1 + max(...)`）
        只有走**回退路径**才到得了 —— typed_nodes 都有 `depth()`。
        用一个没有 `depth()` 的桩节点直接驱动它。
        """
        class _NoDepth:
            def __init__(self, kids=()):
                self._kids = list(kids)

            def children(self):
                return self._kids

        assert FI._tree_depth(_NoDepth()) == 0, (
            "叶子的深度不是 0 —— `if not ch: return 0` 的 not 被删掉了")
        two = _NoDepth([_NoDepth([_NoDepth()])])
        assert FI._tree_depth(two) == 2, (
            "深度累加的符号反了（`1 + max(...)` -> `1 - max(...)`）")
        three = _NoDepth([_NoDepth([_NoDepth([_NoDepth()])])])
        assert FI._tree_depth(three) == 3


class TestArithmeticDescriptionGuards:
    """
    `_describe_arith` 里四处 `if op == X and len(descs) >= N:` 的元数守卫。
    `and` 放宽成 `or` 时，操作数不够的节点会走进去做 `descs[1]` / `descs[2]`
    -> IndexError，整条解释崩掉。

    这些节点同样能被 GP 程序化拼出来（`ArithmeticNode` 无元数校验）。
    """

    @staticmethod
    def _arith(op, n_children):
        from app.core.alpha_engine.typed_nodes import ArithmeticNode, DataNode
        return ArithmeticNode(op, [DataNode("close")] * n_children)

    @pytest.mark.parametrize("op,short", [
        ("add", 1),
        ("if_else", 2),
        ("trade_when", 1),
        ("where", 2),
    ])
    def test_an_underfilled_arithmetic_node_falls_back_instead_of_crashing(
            self, op, short):
        node = self._arith(op, short)
        try:
            out = FI._describe(node)
        except IndexError as exc:
            pytest.fail(
                f"{op} 只有 {short} 个操作数时下标越界（{exc}）—— "
                f"`op == {op!r} and len(descs) >= N` 的元数守卫被放宽成了 `or`")
        assert isinstance(out, str) and out, f"{op} 的降级描述为空"

    @pytest.mark.parametrize("op,full", [
        ("add", 2),
        ("if_else", 3),
        ("trade_when", 2),
        ("where", 3),
    ])
    def test_a_properly_filled_node_still_uses_the_rich_description(self, op, full):
        """反向：操作数够时必须走到那条专用文案，而不是掉进兜底。"""
        out = FI._describe(self._arith(op, full))
        assert "closing price" in out, (
            f"{op}（{full} 个操作数）的描述里没有操作数：{out!r}")


class TestDesignIssueGuardsRoundTwo:

    def test_an_unnormalized_reversion_without_zscore_is_warned(self):
        """
        `if family == "reversion" and "ts_zscore" not in ops and not is_normalized:`
        —— 三个条件缺一不可。

        构造一个**同时满足三者**的因子：取负的动量（family=reversion）、
        不含 ts_zscore、且输出端没有 rank/zscore。
        """
        r = _it("-ts_delta(close,5)")
        assert r.factor_family == "reversion", "用例前提被破坏"
        assert any("without z-score normalization" in i for i in r.issues), (
            f"未归一化且无 zscore 的反转因子没有被告警：{r.issues}")

    def test_a_zscore_based_reversion_is_not_warned(self):
        """`"ts_zscore" not in ops` 这一半：已经用 zscore 的不该再提醒。"""
        r = _it("ts_zscore(close,5)")
        assert r.factor_family == "reversion", "用例前提被破坏"
        assert not any("without z-score normalization" in i for i in r.issues), (
            f"已用 ts_zscore 的反转因子仍被告警：{r.issues}")

    def test_a_normalized_reversion_is_not_warned(self):
        """`not is_normalized` 这一半。"""
        r = _it("rank(-ts_delta(close,5))")
        assert r.factor_family == "reversion", "用例前提被破坏"
        assert not any("without z-score normalization" in i for i in r.issues), (
            f"已归一化的反转因子仍被告警：{r.issues}")

    def test_momentum_without_a_condition_gets_the_regime_suggestion(self):
        """
        `if family in ("momentum","trend_following") and not has_cond:`
        —— 无条件过滤的动量因子该被建议加 regime filter。
        """
        r = _it("rank(ts_mean(ts_delta(close,20),5))")
        assert r.has_condition is False, "用例前提被破坏"
        assert any("regime filter" in s for s in r.suggestions), (
            f"无条件过滤的动量因子没有收到 regime 建议：{r.suggestions}")

    def test_momentum_that_already_has_a_condition_is_not_nagged(self):
        """
        `not has_cond` 这一半：已经带 trade_when 的因子不该再被建议加过滤。
        删掉 `not` 会让它反过来 —— 已经有条件的天天被提醒，没条件的反而不提醒。
        """
        r = _it("trade_when((close>ts_mean(close,20)),rank(ts_delta(close,20)))")
        assert r.has_condition is True, (
            f"用例前提被破坏：has_condition={r.has_condition}")
        assert not any("regime filter" in s for s in r.suggestions), (
            f"已带条件过滤的因子仍被建议加 regime filter：{r.suggestions}")
