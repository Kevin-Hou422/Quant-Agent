"""
gp_engine/fitness.py —— 适应度合成、量纲稳定性、变异权重的定钉测试（变异测试驱动）

来由：20 个变异点，首测击杀率 **35.0%**（存活 13）。

这个文件管三件事，每一件都直接决定 GP 往哪进化：

1. `compute_fitness` —— 多目标合成
   `fitness = sharpe_oos − 0.2·turnover − 0.3·|maxDD| − 0.5·overfit_penalty`
   三个 `*` 和 `overfit_penalty` 里的 `-` 一旦被改，惩罚项变奖励，
   GP 会专门挑高换手、大回撤、IS/OOS 落差最大的个体。

2. `_is_scale_stable` —— 判断一个表达式的输出量纲是否有界。
   判错的后果是"量纲爆炸的因子拿不到惩罚"或"正常因子被误罚"，
   而它是一整棵 AST 的递归判定，分支多、既有测试一条没覆盖到。

3. `mutation_weights_from_metrics` —— 按当前指标调整变异算子的抽样权重。
   三个阈值（换手 > 2.0、OOS Sharpe < 0.2、过拟合分 > 0.5）决定
   "这一代该往哪个方向变异"，边界差一格就换一套策略。

既有覆盖（test_supplementary_fixes）只用了 `scale_stability_penalty` 的
"稳定 vs 不稳定"两组样例，没有碰合成公式和权重表。
"""
from __future__ import annotations

import pytest

from app.core.alpha_engine.parser import Parser
from app.core.gp_engine.fitness import (
    _is_scale_stable,
    compute_fitness,
    mutation_weights_from_metrics,
    scale_stability_penalty,
)


_parse = Parser().parse


# ===========================================================================
# A. 多目标合成公式
# ===========================================================================

class TestComputeFitness:

    def test_baseline_is_the_oos_sharpe(self):
        """无成本、无回撤、IS==OOS 时，fitness 就是 OOS Sharpe 本身。"""
        assert compute_fitness(sharpe_is=1.5, sharpe_oos=1.5, turnover=0.0) == pytest.approx(1.5)

    def test_overfit_penalty_is_is_minus_oos(self):
        """
        `overfit_penalty = max(0.0, sharpe_is - sharpe_oos)` —— `-` 改成 `+`
        会让 penalty 变成两者之**和**：一个 IS=OOS=2.0 的稳健因子会被
        罚掉 0.5×4.0=2.0，比一个 IS=2/OOS=0 的过拟合因子（罚 0.5×2=1.0）
        还惨 —— 惩罚方向完全颠倒。
        """
        solid = compute_fitness(sharpe_is=2.0, sharpe_oos=2.0, turnover=0.0)
        overfit = compute_fitness(sharpe_is=2.0, sharpe_oos=0.0, turnover=0.0)
        assert solid > overfit, (
            f"IS/OOS 一致的因子得分 {solid} 不高于明显过拟合的 {overfit}")
        assert overfit == pytest.approx(0.0 - 0.5 * 2.0), (
            f"过拟合惩罚不是 0.5×(IS−OOS)，实际 fitness={overfit}")

    def test_overfit_penalty_never_becomes_a_bonus(self):
        """`max(0.0, ...)` 的下限：OOS 比 IS 还好时不得倒贴分。"""
        lucky = compute_fitness(sharpe_is=0.5, sharpe_oos=2.0, turnover=0.0)
        assert lucky == pytest.approx(2.0), (
            f"OOS 优于 IS 时 fitness={lucky}，应当就是 OOS Sharpe（罚项为 0）")

    @pytest.mark.parametrize("kw,coef,label", [
        ({"turnover": 3.0}, 0.2, "换手"),
        ({"max_drawdown": -0.4}, 0.3, "回撤"),
    ])
    def test_cost_terms_use_the_documented_coefficients(self, kw, coef, label):
        """
        `- 0.2*turnover` / `- 0.3*|maxDD|` 的 `*` 改成 `/` 会让系数与指标脱钩：
        换手 3.0 的罚从 0.6 变成 0.067，换手越高罚得越**轻**。
        """
        base = compute_fitness(sharpe_is=1.0, sharpe_oos=1.0, turnover=0.0)
        with_cost = compute_fitness(sharpe_is=1.0, sharpe_oos=1.0, **{'turnover':0.0, **kw})
        magnitude = abs(next(iter(kw.values())))
        assert base - with_cost == pytest.approx(coef * magnitude), (
            f"{label}罚项不是 {coef}×{magnitude}，实测差值 {base - with_cost}")

    def test_drawdown_sign_does_not_matter(self):
        """`abs(max_drawdown)` —— 传正传负都该罚同样多。"""
        a = compute_fitness(sharpe_is=1.0, sharpe_oos=1.0, turnover=0.0, max_drawdown=-0.3)
        b = compute_fitness(sharpe_is=1.0, sharpe_oos=1.0, turnover=0.0, max_drawdown=0.3)
        assert a == pytest.approx(b)

    def test_higher_turnover_and_drawdown_both_lower_the_score(self):
        """把三个罚项的方向一次性钉住 —— 任何一个减号变加号都会红。"""
        best = compute_fitness(sharpe_is=1.0, sharpe_oos=1.0,
                               turnover=0.0, max_drawdown=0.0)
        worse_t = compute_fitness(sharpe_is=1.0, sharpe_oos=1.0, turnover=5.0)
        worse_d = compute_fitness(sharpe_is=1.0, sharpe_oos=1.0, turnover=0.0, max_drawdown=-0.5)
        worse_o = compute_fitness(sharpe_is=3.0, sharpe_oos=1.0, turnover=0.0)
        assert worse_t < best and worse_d < best and worse_o < best, (
            f"罚项没有全部降低得分：base={best} t={worse_t} d={worse_d} o={worse_o}")


# ===========================================================================
# B. 量纲稳定性判定
# ===========================================================================

class TestScaleStability:
    """
    `_is_scale_stable` 是一整棵 AST 的递归判定。既有测试只喂了几条完整 DSL，
    分支覆盖极稀 —— `return True` / `return False` 改掉都没人察觉。
    """

    @pytest.mark.parametrize("dsl", [
        "rank(close)",                    # CS 归一化根
        "zscore(ts_mean(close,20))",
        "scale(close)",
        "normalize(volume)",
        "ts_rank(close,20)",              # TS 天然有界
        "ts_zscore(close,20)",
        "ts_corr(close,volume,20)",
        "group_rank(close)",              # 组内归一化 → 天然有界
        "group_zscore(close)",
        "sign(ts_delta(close,5))",        # ±1 输出
        "(close>volume)",                 # 0/1 输出
        "5",                              # 标量
    ])
    def test_stable_expressions_are_recognised(self, dsl):
        assert _is_scale_stable(_parse(dsl)) is True, f"{dsl} 被误判为量纲不稳定"

    @pytest.mark.parametrize("dsl", [
        "close",                          # 裸字段：量纲随价格
        "group_mean(close)",              # 保留输入量纲 → 看子树
        "ts_mean(close,20)",              # 保留输入量纲
        "(close*volume)",
        "(close/volume)",                 # 分母非标量 → 可无界
        "log(close)",
    ])
    def test_unstable_expressions_are_recognised(self, dsl):
        assert _is_scale_stable(_parse(dsl)) is False, f"{dsl} 被误判为量纲稳定"

    def test_division_by_a_scalar_inherits_the_numerator(self):
        """`div` 的分母是标量时继承分子的稳定性，非标量时一律判不稳定。"""
        assert _is_scale_stable(_parse("(rank(close)/2)")) is True
        assert _is_scale_stable(_parse("(close/2)")) is False
        assert _is_scale_stable(_parse("(rank(close)/volume)")) is False

    def test_two_input_ts_checks_both_children(self):
        """
        `if node.second_child is not None: children.append(...)` —— 删掉 `not`
        会让**有**第二个子树时反而不检查它，`ts_cov(rank(close), close, 20)`
        里那个不稳定的 `close` 就漏掉了。
        （`ts_corr` 走的是上面的白名单，这里要用 `ts_cov`。）
        """
        assert _is_scale_stable(_parse("ts_cov(rank(close),rank(volume),20)")) is True
        assert _is_scale_stable(_parse("ts_cov(rank(close),close,20)")) is False, (
            "第二个子树的量纲没有被检查")

    def test_conditional_ignores_the_condition_branch(self):
        """`if_else` 的 cond 不入量纲，只看两个取值分支。"""
        assert _is_scale_stable(_parse("if_else((close>volume),rank(close),rank(volume))")) is True
        assert _is_scale_stable(_parse("if_else((close>volume),rank(close),close)")) is False

    def test_trade_when_looks_at_the_value_branch_only(self):
        assert _is_scale_stable(_parse("trade_when((close>volume),rank(close))")) is True
        assert _is_scale_stable(_parse("trade_when((close>volume),close)")) is False

    def test_scalar_only_children_are_still_bounded(self):
        assert _is_scale_stable(_parse("(5+3)")) is True   # 纯标量运算确实有界
        assert _is_scale_stable(_parse("(close+5)")) is False

    def test_unknown_node_type_is_conservatively_unstable(self):
        """
        末尾的 `return False`（未知节点类型）改成 True 会让任何将来新增的
        节点类型**默认被当成有界**，量纲惩罚静默失效。
        """
        class _Alien:
            pass
        assert _is_scale_stable(_Alien()) is False, (
            "未知节点类型被判成了量纲稳定 —— 保守兜底失效")

    def test_group_ops_split_into_bounded_and_passthrough(self):
        """
        `if node.op in ("group_rank", "group_zscore"): return True` ——
        这个 `True` 改成 False 会让组内归一化算子被当成量纲不稳定，
        一整类合理因子平白挨罚；而 `group_mean` / `group_neutralize`
        保留输入量纲，必须继续看子树。
        """
        assert _is_scale_stable(_parse("group_rank(close)")) is True
        assert _is_scale_stable(_parse("group_zscore(close)")) is True
        assert _is_scale_stable(_parse("group_mean(close)")) is False
        assert _is_scale_stable(_parse("group_mean(rank(close))")) is True, (
            "group_mean 没有继承子树的量纲稳定性")

    def test_penalty_is_zero_for_stable_and_positive_otherwise(self):
        assert scale_stability_penalty(_parse("rank(close)")) == 0.0
        assert scale_stability_penalty(_parse("close")) > 0.0


# ===========================================================================
# C. 变异权重的三个阈值
# ===========================================================================

class TestMutationWeights:

    @staticmethod
    def _w(**kw):
        base = dict(turnover=0.0, sharpe_oos=1.0, overfit_score=0.0)
        base.update(kw)
        return mutation_weights_from_metrics(**base)

    def test_weights_always_sum_to_one_and_stay_positive(self):
        for kw in ({}, {"turnover": 9.0}, {"sharpe_oos": -1.0},
                   {"overfit_score": 1.0}, {"factor_family": "momentum"}):
            w = self._w(**kw)
            assert sum(w.values()) == pytest.approx(1.0), f"{kw} 权重和不为 1"
            assert min(w.values()) > 0, f"{kw} 出现非正权重"

    def test_turnover_threshold_is_strict(self):
        """
        `if turnover > 2.0:` —— **严格大于**。放宽成 `>=` 会让换手恰好 2.0
        的个体也进入"降换手"模式，与阈值定义不符。
        """
        at = self._w(turnover=2.0)
        just_over = self._w(turnover=2.0001)
        assert at != just_over, (
            "换手恰好 2.0 与略超 2.0 给出了同一套权重 —— 阈值被放宽")
        assert just_over["param"] > at["param"], "超过阈值后没有加大 param 权重"

    def test_low_oos_sharpe_threshold_is_strict(self):
        """`if sharpe_oos < 0.2:` —— 恰好 0.2 不该触发"救 OOS"模式。"""
        at = self._w(sharpe_oos=0.2)
        just_under = self._w(sharpe_oos=0.1999)
        assert at != just_under, (
            "OOS Sharpe 恰好 0.2 与略低于 0.2 给出了同一套权重")
        assert just_under["wrap_rank"] > at["wrap_rank"]

    def test_overfit_threshold_is_strict(self):
        """`if overfit_score > 0.5:` —— 恰好 0.5 不该触发"反过拟合"模式。"""
        at = self._w(overfit_score=0.5)
        just_over = self._w(overfit_score=0.5001)
        assert at != just_over, "过拟合分恰好 0.5 与略超 0.5 给出了同一套权重"
        assert just_over["hoist"] > at["hoist"]

    def test_known_family_bias_is_applied(self):
        """
        `if factor_family and factor_family in _FAMILY_WEIGHT_BIASES:` ——
        `and` 放宽成 `or` 会让**未知族名**也去索引偏置表 → KeyError；
        空族名同理。
        """
        from app.core.gp_engine.fitness import _FAMILY_WEIGHT_BIASES
        fam = next(iter(_FAMILY_WEIGHT_BIASES))
        assert self._w(factor_family=fam) != self._w(), (
            f"已知族 {fam} 的偏置没有生效")

    def test_unknown_and_empty_family_fall_back_to_the_base_weights(self):
        base = self._w()
        assert self._w(factor_family="no_such_family") == base
        assert self._w(factor_family="") == base

    def test_the_three_layers_compose(self):
        """三个条件同时成立时，各自的调整必须叠加，而不是互相覆盖。"""
        only_t = self._w(turnover=9.0)
        only_o = self._w(overfit_score=1.0)
        both = self._w(turnover=9.0, overfit_score=1.0)
        assert both != only_t and both != only_o, (
            "两个条件同时成立时的权重与单独成立时相同 —— 调整没有叠加")


# ===========================================================================
# D. 空子节点兜底
# ===========================================================================

def test_an_arithmetic_node_with_no_children_is_not_scale_stable():
    """
    `return all(...) if children else False` —— 那个 `False` 是保守兜底。

    上一版把它记成"等价变异，因为 parser 产不出零子节点的 ArithmeticNode"。
    那个理由站不住：`ArithmeticNode.__init__` **没有元数校验**，
    `ArithmeticNode("add", [])` 直接就能造出来，而 GP 的变异/交叉是
    程序化拼节点、不走 parser。真让它走到这里时，`all(空)` 是 True ——
    兜底翻成 True 会把一个**形态不明**的节点判成量纲有界，
    fitness 里少扣一档惩罚。

    这条不是证明它不可达，而是直接把它的取值钉死。
    """
    from app.core.alpha_engine.typed_nodes import ArithmeticNode

    node = ArithmeticNode("add", [])
    assert node.children() == [] or list(node.children()) == [], "前提被破坏"
    assert _is_scale_stable(node) is False, (
        "零子节点的 ArithmeticNode 被判成量纲稳定 —— "
        "L133 的兜底从 False 变成了 True，形态不明的节点会少扣量纲惩罚")

    # 对照：同一个 op 有子节点时，判定回到递归结果
    assert _is_scale_stable(_parse("(rank(close)+rank(volume))")) is True
    assert _is_scale_stable(_parse("(close+volume)")) is False


def test_arithmetic_nodes_from_the_parser_always_have_children():
    """
    补充面：parser 这条路径上确实产不出零子节点的算术节点 ——
    所以上面那条兜底只会被程序化构造触发，不会被正常 DSL 触发。
    """
    from app.core.alpha_engine.typed_nodes import ArithmeticNode
    # 注意：`neg` 不是 DSL 函数名（一元负号写成 `(0-x)`），
    # 写进来会直接 ParseError —— 样本清单必须按文法实际支持的来。
    samples = ["(1+2)", "(close-volume)", "log(close)",
               "sign(close)", "abs(close)", "sqrt(close)", "not(close)", "(0-close)",
               "max(close,volume)", "min(close,volume)", "pow(close,2)",
               "signed_power(close,0.5)", "if_else((close>volume),close,volume)",
               "trade_when((close>volume),close)", "and(close,volume)",
               "or(close,volume)", "weighted_sum(close,1,volume,2)"]
    seen = 0
    for dsl in samples:
        node = _parse(dsl)
        for n in [node] + list(node.children()):
            if isinstance(n, ArithmeticNode):
                seen += 1
                assert n.children(), (
                    f"{dsl} 产出了没有子节点的 ArithmeticNode —— "
                    f"L133 的兜底变成可达，需要补用例")
    assert seen >= len(samples), f"只检查到 {seen} 个算术节点，样本不足"


# 本文件已无"存活但被判等价"的变异点：L133 由上面的
# test_an_arithmetic_node_with_no_children_is_not_scale_stable 直接杀死。
PROVEN_EQUIVALENT: dict = {}


def test_every_survivor_has_a_written_proof():
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
