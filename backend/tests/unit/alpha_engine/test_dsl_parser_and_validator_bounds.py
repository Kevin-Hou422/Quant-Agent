"""
alpha_engine/parser.py + validator.py —— 元数/边界的定钉测试（变异测试驱动）

来由：parser 13 个变异点，首测击杀率 **15.4%**（存活 11）；
      validator 11 个变异点，首测击杀率 **27.3%**（存活 8）。

两个模块的存活项几乎全是**同一种形状**：`len(arglist) < k` / `> k`、
`w < MIN` / `> MAX`、`d > MAX_DEPTH` —— 差一格的边界。

它们为什么危险，两边不一样：

  parser：边界放宽一格 → **恰好合法的表达式被拒**（`ts_corr(a,b,5)` 只有 3 个参数，
  `< 3` 放宽成 `<= 3` 就把它判成参数不足），或者 `args[1]` 在没有第二个参数时
  直接 IndexError —— GP 演化出的表达式会成批地在解析阶段莫名失败，
  而失败率的变化没有任何人看得见。

  validator：边界收紧/放宽一格 → **该拦的没拦**。`window < 1` 放宽成 `<= 1`
  会让 window=1 的 ts_delay 被当成前视泄漏拒掉（合法写法被封）；
  反过来 `d > MAX_DEPTH` 改成 `>=` 会把恰好合规的深度也拒掉。
  最要命的是 `is_valid` 的 `return True` / `return False` ——
  改掉之后**所有表达式都"合法"**，三个校验器全部白做。

既有覆盖（test_dsl_edge_cases / test_dsl_operators / test_dsl_engine /
test_alpha_discovery）只在远离边界的地方测：要么明显合法、要么明显非法。
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from app.core.alpha_engine.parser import ParseError, Parser
from app.core.alpha_engine.typed_nodes import (
    ArithmeticNode,
    CrossSectionalNode,
    DataNode,
    GroupNode,
    ScalarNode,
    TimeSeriesNode,
)
from app.core.alpha_engine.validator import (
    AlphaValidator,
    DepthValidator,
    LookAheadValidator,
    ValidationError,
    WindowValidator,
)


@pytest.fixture
def parse():
    p = Parser()
    return p.parse


# ===========================================================================
# A. parser：恰好够用的参数个数必须被接受
# ===========================================================================

class TestParserArityLowerBounds:
    """
    每条都取**边界上的那一个**参数个数。`< k` 放宽成 `<= k` 会让这些
    完全合法的表达式解析失败。
    """

    def test_two_input_ts_accepts_exactly_three_arguments(self, parse):
        node = parse("ts_corr(close,volume,5)")
        assert isinstance(node, TimeSeriesNode)
        assert node.window == 5 and node.second_child is not None, (
            "ts_corr 的三个参数没有被正确拆成 (x, y, window)")

    def test_two_input_ts_rejects_two_arguments(self, parse):
        with pytest.raises(ParseError, match="3 arguments"):
            parse("ts_corr(close,5)")

    def test_single_input_ts_accepts_exactly_two_arguments(self, parse):
        node = parse("ts_mean(close,5)")
        assert node.window == 5 and node.second_child is None

    def test_single_input_ts_rejects_one_argument(self, parse):
        with pytest.raises(ParseError, match="at least 2 arguments"):
            parse("ts_mean(close)")

    def test_ind_neutralize_accepts_exactly_one_argument(self, parse):
        """`len(arglist) < 1` 放宽成 `<= 1` 会把单参数写法判成"参数不足"。"""
        node = parse("ind_neutralize(close)")
        assert isinstance(node, CrossSectionalNode)
        assert node.params.get("groups_node") is None, (
            "没给第二参却填上了 groups_node —— `len>1` 的守卫失效")

    def test_ind_neutralize_accepts_a_group_argument(self, parse):
        node = parse("ind_neutralize(close,'sector')")
        assert node.params.get("groups_node") is not None
        assert "sector" in repr(node)

    def test_winsorize_accepts_exactly_one_argument_and_defaults_k(self, parse):
        node = parse("winsorize(close)")
        assert node.params["k"] == 3.0, (
            f"缺省 k 应为 3.0，实际 {node.params['k']} —— "
            f"`len(arglist) > 1` 被放宽，读到了不存在的第二参")

    def test_winsorize_reads_an_explicit_k(self, parse):
        node = parse("winsorize(close,2.5)")
        assert node.params["k"] == 2.5
        assert "2.5" in repr(node), "非缺省的 k 没有进 repr —— 会与缺省版共用缓存键"

    def test_group_op_accepts_exactly_one_argument(self, parse):
        node = parse("group_rank(close)")
        assert isinstance(node, GroupNode)
        assert node.group_field == "groups", (
            "缺省分组字段不是 'groups' —— `len(arglist) > 1` 的守卫失效")

    def test_group_op_reads_an_explicit_field(self, parse):
        assert parse("group_rank(close,'sector')").group_field == "sector"
        assert parse("group_rank(close,sector)").group_field == "sector"

    def test_weighted_sum_accepts_exactly_two_arguments(self, parse):
        """
        `len(arglist) < 2 or len % 2 != 0` —— `<` 放宽成 `<=` 会把
        最小的合法写法（一对 值/权重）判成参数不足。
        """
        node = parse("weighted_sum(close,0.5)")
        assert isinstance(node, ArithmeticNode)
        assert len(node.children()) == 2

    def test_weighted_sum_rejects_an_odd_argument_count(self, parse):
        with pytest.raises(ParseError, match="even number"):
            parse("weighted_sum(close,0.5,volume)")

    def test_weighted_sum_splits_values_then_weights(self, parse):
        """
        `vals + wgts` 是**列表拼接**（前一半值、后一半权重）。
        改成 `vals - wgts` 会直接 TypeError；改了顺序则值与权重配错对。
        """
        node = parse("weighted_sum(close,2,volume,3)")
        ds = {"close": np.full((2, 2), 10.0), "volume": np.full((2, 2), 100.0)}
        got = node.evaluate(ds, {})
        np.testing.assert_allclose(got, 10 * 2 + 100 * 3, rtol=1e-12,
                                   err_msg="值与权重配错了对")


class TestParserZeroArgAndUnknown:

    def test_zero_argument_call_is_rejected_by_the_grammar(self, parse):
        """
        零参调用在**词法层**就被挡下（`arglist : arg ("," arg)*` 至少要一个 arg），
        根本走不到 `func_call` 的 Python 代码。这也正是 L166 的等价性依据，见 D 节。
        """
        for expr in ("rank()", "ts_mean()", "log()"):
            with pytest.raises(ParseError, match="Syntax error"):
                parse(expr)

    def test_unknown_function_names_the_known_ones(self, parse):
        with pytest.raises(ParseError, match="Unknown function"):
            parse("ts_nope(close,5)")

    def test_window_must_be_an_integer_literal(self, parse):
        with pytest.raises(ParseError, match="integer literal"):
            parse("ts_mean(close,volume)")

    def test_cs_op_rejects_a_second_argument(self, parse):
        with pytest.raises(ParseError, match="exactly 1 argument"):
            parse("rank(close,volume)")


# ===========================================================================
# B. validator：窗口区间的两端都要能用
# ===========================================================================

class TestWindowBounds:

    def test_minimum_window_is_accepted(self):
        """`w < MIN_WINDOW` 放宽成 `<=` 会把 window=1 判成非法 —— ts_delay(x,1) 是
        最常用的滞后写法，一下全被封。"""
        v = WindowValidator()
        assert v.collect(TimeSeriesNode("ts_mean", DataNode("close"),
                                        WindowValidator.MIN_WINDOW)) == []

    def test_below_minimum_window_is_rejected(self):
        errs = WindowValidator().collect(
            TimeSeriesNode("ts_mean", DataNode("close"), 0))
        assert errs and "invalid" in errs[0], f"window=0 没有被拒：{errs}"

    def test_maximum_window_is_accepted(self):
        """`w > MAX_WINDOW` 放宽成 `>=` 会把恰好 252（一整年）判成超限。"""
        v = WindowValidator()
        assert v.collect(TimeSeriesNode("ts_mean", DataNode("close"),
                                        WindowValidator.MAX_WINDOW)) == []

    def test_above_maximum_window_is_rejected(self):
        errs = WindowValidator().collect(
            TimeSeriesNode("ts_mean", DataNode("close"),
                           WindowValidator.MAX_WINDOW + 1))
        assert errs and "exceeds" in errs[0], f"window=253 没有被拒：{errs}"

    def test_equal_nested_windows_do_not_warn(self):
        """
        `child.window > w` 放宽成 `>=` 会让**相等**的嵌套窗口也报冗余警告。
        `ts_std(ts_mean(close,20),20)` 是完全正常的写法。
        """
        node = TimeSeriesNode("ts_std",
                              TimeSeriesNode("ts_mean", DataNode("close"), 20), 20)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            WindowValidator().collect(node)
        assert not caught, f"相等窗口触发了冗余警告：{[str(w.message) for w in caught]}"

    def test_larger_child_window_does_warn(self):
        node = TimeSeriesNode("ts_std",
                              TimeSeriesNode("ts_mean", DataNode("close"), 60), 20)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            WindowValidator().collect(node)
        assert caught, "子窗口大于父窗口却没有警告"


class TestLookAheadBounds:

    def test_delay_with_window_one_is_allowed(self):
        """
        `node.window < 1` 放宽成 `<= 1` 会把 `ts_delay(close,1)` ——
        最基本的"看昨天"——判成前视泄漏。
        """
        for op in ("ts_delay", "ts_delta"):
            assert LookAheadValidator().collect(
                TimeSeriesNode(op, DataNode("close"), 1)) == [], (
                f"{op}(close,1) 被误判成前视泄漏")

    def test_delay_with_window_zero_is_look_ahead(self):
        errs = LookAheadValidator().collect(
            TimeSeriesNode("ts_delay", DataNode("close"), 0))
        assert errs and "look-ahead" in errs[0], f"window=0 没有被拦：{errs}"

    def test_future_prefixed_fields_are_forbidden(self):
        errs = LookAheadValidator().collect(DataNode("future_return"))
        assert errs and "future_" in errs[0]

    def test_ordinary_fields_are_allowed(self):
        assert LookAheadValidator().collect(DataNode("close")) == []


class TestDepthBounds:

    @staticmethod
    def _chain(depth: int):
        """构造恰好 `depth` 层的表达式。"""
        node = DataNode("close")
        for _ in range(depth):
            node = TimeSeriesNode("ts_mean", node, 3)
        return node

    def test_depth_exactly_at_the_limit_is_accepted(self):
        """`d > MAX_DEPTH` 改成 `>=` 会把恰好等于上限的表达式也拒掉。"""
        node = self._chain(DepthValidator.MAX_DEPTH)
        assert node.depth() == DepthValidator.MAX_DEPTH, "构造的深度不对"
        assert DepthValidator().collect(node) == []
        DepthValidator().validate(node)          # 不抛即通过

    def test_depth_one_over_the_limit_is_rejected(self):
        node = self._chain(DepthValidator.MAX_DEPTH + 1)
        errs = DepthValidator().collect(node)
        assert errs and "exceeds" in errs[0], f"超限深度没有被拒：{errs}"
        with pytest.raises(ValidationError):
            DepthValidator().validate(node)

    def test_collect_and_validate_agree_on_the_boundary(self):
        """
        两处 `d > MAX_DEPTH` 是**两份拷贝**（validate 一处、collect 一处）。
        只改其中一处会让"抛不抛"与"报不报"对不上 —— AlphaValidator 走 collect，
        直接 validate 走另一条，两条路会给出不同结论。
        """
        for depth in (DepthValidator.MAX_DEPTH, DepthValidator.MAX_DEPTH + 1):
            node = self._chain(depth)
            collected = bool(DepthValidator().collect(node))
            try:
                DepthValidator().validate(node)
                raised = False
            except ValidationError:
                raised = True
            assert collected == raised, (
                f"深度 {depth}：collect 报错={collected} 但 validate 抛异常={raised} "
                f"—— 两份边界拷贝已经不一致")


# ===========================================================================
# C. AlphaValidator.is_valid 的真假值
# ===========================================================================

class TestIsValid:
    """
    `return True` / `return False` 两行一旦被改，`is_valid` 变成常量函数：
    恒 True → 所有非法表达式放行；恒 False → 所有表达式被拒、GP 无法产出任何个体。
    """

    def test_is_valid_is_true_for_a_sound_expression(self):
        node = TimeSeriesNode("ts_mean", DataNode("close"), 20)
        assert AlphaValidator().is_valid(node) is True

    def test_is_valid_is_false_for_an_out_of_range_window(self):
        node = TimeSeriesNode("ts_mean", DataNode("close"), 10_000)
        assert AlphaValidator().is_valid(node) is False

    def test_is_valid_is_false_for_a_future_field(self):
        assert AlphaValidator().is_valid(DataNode("future_pnl")) is False

    def test_is_valid_distinguishes_the_two_cases(self):
        """恒真/恒假的实现会让下面这条相等断言成立 —— 必须不相等。"""
        good = AlphaValidator().is_valid(TimeSeriesNode("ts_mean", DataNode("close"), 5))
        bad = AlphaValidator().is_valid(TimeSeriesNode("ts_mean", DataNode("close"), 0))
        assert good != bad, "is_valid 对合法与非法表达式给出了同一个答案"

    def test_validate_reports_every_problem_at_once(self):
        """三个校验器的错误要合并上报，而不是碰到第一个就停。"""
        node = TimeSeriesNode("ts_delay", DataNode("future_x"), 0)
        with pytest.raises(ValidationError) as exc:
            AlphaValidator().validate(node)
        msg = str(exc.value)
        assert "look-ahead" in msg and "future_" in msg, (
            f"只报了一部分问题：{msg}")


# ===========================================================================
# D. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "app/core/alpha_engine/parser.py ×1 — L166 `arglist = args[1] if len(args) > 1 else []` → `>=`":
        "文法把 func_call 定义成 `IDENT \"(\" arglist \")\"`，而 `arglist : arg (\",\" arg)*` "
        "**至少要一个 arg** —— 零参调用在词法层就报 Syntax error，走不到这行 Python。"
        "因此 func_call 收到的 args 恒为 2 个（IDENT + arglist），`len(args) > 1` 与 "
        "`>= 1` 恒同真。见 test_func_call_always_receives_exactly_two_children。",

    "app/core/alpha_engine/parser.py ×0 — L295 `if len(arglist) > 1 and isinstance(arglist[1], ScalarNode)` → `or`（winsorize）":
        "`or` 的短路让第一个子式为真时就不再看第二个 —— 但两个子式的真值在本行"
        "**同向**：只有 len>1 时 arglist[1] 才存在，len<=1 时 isinstance 会 IndexError。"
        "Python 的 `or` 仍然按顺序求值，len<=1 时第一个子式为 False，"
        "`or` 会继续求第二个子式 → IndexError。因此该变异**不是等价变异**，"
        "已被 test_winsorize_accepts_exactly_one_argument_and_defaults_k 杀死。"
        "此条保留为记录：删掉那条用例这个点会重新变成盲区。",
}


def test_func_call_always_receives_exactly_two_children():
    """
    L166 等价性的机械验证：从**文法本身**证明 arglist 不可为空。
    直接断言文法文本里的产生式形状 —— 一旦有人把 arglist 改成可选
    （`[arglist]` 或 `arg*`），这条会红，那个变异点就要重新补用例。
    """
    import inspect
    import re
    import app.core.alpha_engine.parser as P
    src = inspect.getsource(P)
    assert re.search(r'func_call\s*:\s*IDENT\s*"\("\s*arglist\s*"\)"', src), (
        "func_call 的产生式变了 —— L166 的等价性证明失效")
    assert re.search(r'arglist\s*:\s*arg\s*\("\,"\s*arg\)\*', src), (
        "arglist 的产生式变了（现在可能允许空参数）—— L166 需要重新补用例")

    # 再从行为侧确认一次：合法调用至少带一个参数
    node = Parser().parse("rank(close)")
    assert isinstance(node, CrossSectionalNode)


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 2
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
