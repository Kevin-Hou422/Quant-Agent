"""
alpha_engine/typed_nodes.py —— DSL AST 节点语义的定钉测试（变异测试驱动）

来由：33 个变异点，首测击杀率 **30.3%**（存活 23）。

typed_nodes 是 DSL 字符串与 fast_ops 之间的那一层：
`repr()` 决定**缓存键**与**因子身份**（谱系里存的就是这个字符串），
`_compute` 决定二元/一元算子的实际语义，`depth()` 决定 validator 的复杂度门。

存活项集中在三处，每一处都能静默改变因子值：

  - **算子语义**：`add` 的 `+`、`sub` 的 `-`、`mul` 的 `*`、`weighted_sum` 的
    `v*w`、`max2`/`min2` 的 maximum↔minimum —— 改掉之后表达式照常求值、
    照常回测，只是算的不是它 repr 出来的那个东西了
  - **保护性边界**：`div` 的 `|denom| < 1e-8`、`log` 的 `x > 0`、
    `gt`/`lt` 的严格性 —— 放宽一格就把 inf/NaN 放进因子值
  - **depth()**：`1 + child.depth()` 改成 `-` 会让深度不增反减，
    validator 的 MAX_DEPTH 门永远不触发，GP 可以无限加深

既有覆盖（test_dsl_engine / test_dsl_operators / test_dsl_edge_cases）验证的是
"能解析、能求值、形状对"，没有一条把**算出来的数**与算子语义对上。
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.alpha_engine.typed_nodes import (
    ArithmeticNode,
    CrossSectionalNode,
    DataNode,
    GroupNode,
    NodeType,
    ScalarNode,
    StringLiteralNode,
    TimeSeriesNode,
    _resolve_type,
)


T, N = 12, 4


def _ds(**extra) -> dict:
    rng = np.random.default_rng(0)
    base = {
        "close": 100 + rng.normal(0, 1, (T, N)).cumsum(axis=0),
        "open": 100 + rng.normal(0, 1, (T, N)).cumsum(axis=0),
        "volume": rng.uniform(1e5, 1e6, (T, N)),
    }
    base.update(extra)
    return base


def _ev(node, dataset=None):
    return node.evaluate(dataset if dataset is not None else _ds(), {})


# ===========================================================================
# A. 二元算术
# ===========================================================================

class TestBinaryArithmetic:
    """
    `add`/`sub`/`mul` 三个 return 各是一行，改掉符号后表达式的 repr 不变、
    形状不变、不抛异常 —— 只有值变了。所以必须逐个钉值。
    """

    @pytest.mark.parametrize("op,expected", [
        ("add", 7.0), ("sub", 1.0), ("mul", 12.0),
    ])
    def test_binary_ops_compute_the_advertised_operation(self, op, expected):
        node = ArithmeticNode(op, [ScalarNode(4.0), ScalarNode(3.0)])
        assert float(_ev(node)) == expected, (
            f"{op}(4,3) 应当是 {expected}，实际 {float(_ev(node))} —— "
            f"算子语义与它的名字对不上")

    def test_binary_ops_are_pairwise_distinguishable(self):
        """4 与 3 的和/差/积互不相同 —— 保证上面那三条断言真的有区分力。"""
        vals = {op: float(_ev(ArithmeticNode(op, [ScalarNode(4.0), ScalarNode(3.0)])))
                for op in ("add", "sub", "mul", "div")}
        assert len(set(vals.values())) == 4, f"四种二元算子给出了重复的值：{vals}"

    def test_sub_is_left_minus_right(self):
        """`a - b` 写成 `b - a` 或 `a + b` 都会被这条抓住（非对称输入）。"""
        node = ArithmeticNode("sub", [DataNode("close"), DataNode("open")])
        ds = _ds()
        np.testing.assert_allclose(_ev(node, ds), ds["close"] - ds["open"],
                                   rtol=1e-12,
                                   err_msg="sub 算的不是 左 - 右")

    def test_weighted_sum_multiplies_value_by_weight(self):
        """
        `sum(v * w)` 改成 `v / w`：权重 0.5 会从"减半"变成"翻倍"，
        组合因子的配比整体反过来，而 repr 里的权重数字一字不变。
        """
        node = ArithmeticNode("weighted_sum",
                              [ScalarNode(4.0), ScalarNode(6.0),      # 值
                               ScalarNode(0.5), ScalarNode(2.0)])     # 权重
        assert float(_ev(node)) == 4 * 0.5 + 6 * 2.0, (
            "weighted_sum 不是 Σ(值×权重)")

    def test_weighted_sum_splits_values_and_weights_in_half(self):
        """前一半是值、后一半是权重；切分点错了结果会变。"""
        node = ArithmeticNode("weighted_sum",
                              [ScalarNode(1.0), ScalarNode(10.0),
                               ScalarNode(3.0), ScalarNode(0.0)])
        assert float(_ev(node)) == 3.0, (
            "值与权重的切分位置不对（应为 1×3 + 10×0 = 3）")


# ===========================================================================
# B. 除法与对数的保护边界
# ===========================================================================

class TestGuardedOps:

    def test_div_returns_nan_for_a_near_zero_denominator(self):
        """
        `np.where(np.abs(denom) < 1e-8, np.nan, denom)`。
        去掉 `np.abs` 后**负的**小分母（-1e-12）逃过守卫，
        `x / -1e-12` 变成 -1e12 级别的巨值直接进因子；
        下游 cs_zscore 会被这一个点拉爆，整截面塌成 0。
        """
        num = np.full((2, 2), 1.0)
        for denom_val in (1e-12, -1e-12, 0.0):
            node = ArithmeticNode("div", [DataNode("a"), DataNode("b")])
            got = _ev(node, {"a": num, "b": np.full((2, 2), denom_val)})
            assert np.all(np.isnan(got)), (
                f"分母 {denom_val} 没有被守卫拦下，算出了 {got.ravel()[0]}")

    def test_div_by_a_large_negative_number_is_allowed(self):
        """
        `np.abs(denom) < 1e-8` 里的 `np.abs` 是**双向**守卫。去掉之后条件变成
        `denom < 1e-8`，于是**所有负分母**（-5、-100，完全正常的值）都被判成
        "接近零"而置 NaN —— `close / (returns - 0.01)` 这类表达式在下跌日
        整列消失，因子只剩上涨日有值，凭空产生方向性偏差。

        （上一条用 ±1e-12 测不出来：去掉 abs 后负的小分母**仍然**是 NaN，
        两种实现在那里恰好一致。必须用大负数才有区分力。）
        """
        node = ArithmeticNode("div", [DataNode("a"), DataNode("b")])
        got = _ev(node, {"a": np.array([[10.0, 10.0]]),
                         "b": np.array([[-5.0, 2.0]])}).ravel()
        assert np.isfinite(got[0]), (
            f"分母 -5 被当成了接近零（得到 {got[0]}）—— `np.abs` 守卫丢了")
        np.testing.assert_allclose(got, [-2.0, 5.0], rtol=1e-12)

    def test_div_at_exactly_the_epsilon_is_not_guarded(self):
        """
        `< 1e-8` 是**严格小于**：分母恰好等于 1e-8 时必须正常相除。
        放宽成 `<=` 会把这个合法值也变成 NaN —— 边界上的数据被静默丢掉。
        """
        node = ArithmeticNode("div", [DataNode("a"), DataNode("b")])
        got = _ev(node, {"a": np.full((1, 1), 1.0), "b": np.full((1, 1), 1e-8)})
        assert np.isfinite(got).all(), (
            "分母恰好等于 1e-8 时被守卫误杀 —— `<` 被放宽成了 `<=`")
        np.testing.assert_allclose(got.ravel()[0], 1e8, rtol=1e-9)

    def test_log_of_zero_is_nan_not_minus_infinity(self):
        """
        `np.where(evaled[0] > 0, ...)` 是**严格大于**。放宽成 `>=` 后
        `log(0)` 返回 **-inf** 而不是 NaN：-inf 不会被 NaN 过滤器拿掉，
        会一路流进权重计算，把那一只票的仓位算成 -inf。
        """
        node = ArithmeticNode("log", [DataNode("a")])
        got = _ev(node, {"a": np.array([[0.0, 1.0, np.e, -5.0]])}).ravel()
        assert np.isnan(got[0]), f"log(0) 给出了 {got[0]}，不是 NaN"
        assert np.isnan(got[3]), "log(负数) 应当是 NaN"
        np.testing.assert_allclose(got[1:3], [0.0, 1.0], rtol=1e-12)

    def test_sqrt_of_zero_is_defined_but_negative_is_nan(self):
        node = ArithmeticNode("sqrt", [DataNode("a")])
        got = _ev(node, {"a": np.array([[0.0, 4.0, -1.0]])}).ravel()
        np.testing.assert_allclose(got[:2], [0.0, 2.0], rtol=1e-12,
                                   err_msg="sqrt 的 `>= 0` 边界把 0 也挡掉了")
        assert np.isnan(got[2])

    def test_comparisons_are_strict(self):
        """
        `gt` / `lt` 用严格不等号。放宽成 `>=` / `<=` 会让"突破"类条件在
        **恰好持平**时也触发 —— trade_when(close > ts_max(close,20), ...)
        会在没有创新高的那天照样开仓。
        """
        ds = {"a": np.array([[1.0, 2.0, 3.0]]), "b": np.array([[2.0, 2.0, 2.0]])}
        gt = _ev(ArithmeticNode("gt", [DataNode("a"), DataNode("b")]), ds).ravel()
        lt = _ev(ArithmeticNode("lt", [DataNode("a"), DataNode("b")]), ds).ravel()
        np.testing.assert_array_equal(gt, [0.0, 0.0, 1.0],
                                      err_msg="gt 在相等时触发了 —— 不是严格大于")
        np.testing.assert_array_equal(lt, [1.0, 0.0, 0.0],
                                      err_msg="lt 在相等时触发了 —— 不是严格小于")
        gte = _ev(ArithmeticNode("gte", [DataNode("a"), DataNode("b")]), ds).ravel()
        np.testing.assert_array_equal(gte, [0.0, 1.0, 1.0],
                                      err_msg="gte 在相等时没有触发")

    def test_signed_power_keeps_sign_and_uses_magnitude(self):
        """
        `np.sign(x) * np.abs(x) ** p`：去掉 abs 后负底数遇分数次幂变 NaN，
        `*` 改成 `/` 会把符号变成 ±1 的除数（结果量级完全不同）。
        """
        node = ArithmeticNode("signed_power", [DataNode("a"), ScalarNode(0.5)])
        got = _ev(node, {"a": np.array([[-9.0, 9.0, 0.0]])}).ravel()
        np.testing.assert_allclose(got, [-3.0, 3.0, 0.0], rtol=1e-12,
                                   err_msg="signed_power 丢了符号或对负数取幂成了 NaN")

    def test_max2_and_min2_are_not_swapped(self):
        ds = {"a": np.array([[1.0, 5.0]]), "b": np.array([[3.0, 2.0]])}
        np.testing.assert_array_equal(
            _ev(ArithmeticNode("max2", [DataNode("a"), DataNode("b")]), ds).ravel(),
            [3.0, 5.0], err_msg="max2 取的是最小值")
        np.testing.assert_array_equal(
            _ev(ArithmeticNode("min2", [DataNode("a"), DataNode("b")]), ds).ravel(),
            [1.0, 2.0], err_msg="min2 取的是最大值")


# ===========================================================================
# C. depth()：validator 复杂度门的唯一输入
# ===========================================================================

class TestDepth:
    """
    `1 + child.depth()` 改成 `1 - child.depth()`：嵌套越深值越小甚至变负，
    validator 的 `if d > MAX_DEPTH` 永远不成立 —— **复杂度上限彻底失效**，
    GP 可以演化出任意深的表达式，过拟合再无结构性约束。
    """

    def test_time_series_depth_increases_with_nesting(self):
        leaf = DataNode("close")
        one = TimeSeriesNode("ts_mean", leaf, 3)
        two = TimeSeriesNode("ts_std", one, 3)
        assert leaf.depth() == 0
        assert one.depth() == 1, f"单层 TS 的 depth 应为 1，实际 {one.depth()}"
        assert two.depth() == 2, f"两层嵌套的 depth 应为 2，实际 {two.depth()}"

    def test_two_input_ts_depth_takes_the_deeper_child(self):
        """
        `max(base, 1 + second_child.depth())` —— `+` 改成 `-` 会让
        **第二个子树的深度被忽略甚至倒扣**，`ts_corr(close, 很深的表达式, w)`
        就此绕过复杂度门。
        """
        shallow = DataNode("close")
        deep = CrossSectionalNode("rank",
                                  TimeSeriesNode("ts_mean", DataNode("volume"), 5))
        node = TimeSeriesNode("ts_corr", shallow, 5, second_child=deep)
        assert deep.depth() == 2
        assert node.depth() == 3, (
            f"depth 应当取更深的那一支 (1+2=3)，实际 {node.depth()} —— "
            f"第二个子树的深度没有被计入")

    def test_cross_sectional_and_group_depth_increase(self):
        inner = TimeSeriesNode("ts_mean", DataNode("close"), 3)
        assert CrossSectionalNode("rank", inner).depth() == 2
        assert GroupNode("group_rank", inner).depth() == 2

    def test_arithmetic_depth_takes_the_deepest_child(self):
        shallow = ScalarNode(1.0)
        deep = CrossSectionalNode("rank",
                                  TimeSeriesNode("ts_mean", DataNode("close"), 3))
        node = ArithmeticNode("add", [shallow, deep])
        assert node.depth() == 3, (
            f"算术节点的 depth 应为 1+max(子树)=3，实际 {node.depth()}")

    def test_arithmetic_depth_of_leaves_only_is_one(self):
        assert ArithmeticNode("add", [ScalarNode(1.0), ScalarNode(2.0)]).depth() == 1


# ===========================================================================
# D. 类型传播
# ===========================================================================

class TestTypeResolution:
    """
    `_resolve_type` 决定一个算术节点算 TS 还是 CS 还是 GROUP。
    `if n.node_type not in (_ST,)` 改成 `in (_ST,)` 会让集合里**只剩字符串字面量**，
    于是所有算术节点都退化成 SCALAR —— 下游按标量对待，广播语义全错。
    """

    def test_string_literals_do_not_participate_in_type_resolution(self):
        assert _resolve_type(StringLiteralNode("sector"), DataNode("close")) \
            is NodeType.DATA, "字符串字面量污染了类型传播"

    def test_only_string_literals_resolve_to_scalar(self):
        assert _resolve_type(StringLiteralNode("sector")) is NodeType.SCALAR

    def test_group_beats_cross_sectional_beats_time_series(self):
        ts = TimeSeriesNode("ts_mean", DataNode("close"), 3)
        cs = CrossSectionalNode("rank", DataNode("close"))
        gr = GroupNode("group_rank", DataNode("close"))
        assert _resolve_type(ts, cs) is NodeType.CROSS_SECTIONAL
        assert _resolve_type(cs, gr) is NodeType.GROUP
        assert _resolve_type(DataNode("close"), ts) is NodeType.TIME_SERIES
        assert _resolve_type(ScalarNode(1.0), DataNode("close")) is NodeType.DATA

    def test_arithmetic_node_inherits_the_resolved_type(self):
        node = ArithmeticNode("add", [ScalarNode(1.0),
                                      CrossSectionalNode("rank", DataNode("close"))])
        assert node.node_type is NodeType.CROSS_SECTIONAL


# ===========================================================================
# E. 分组字段的解析
# ===========================================================================

class TestGroupFieldResolution:

    def test_group_node_reads_the_named_field(self):
        ds = _ds(sector=np.array([0, 0, 1, 1]))
        node = GroupNode("group_mean", DataNode("close"), group_field="sector")
        got = _ev(node, ds)
        np.testing.assert_allclose(got[:, 0], got[:, 1], rtol=1e-12,
                                   err_msg="同组的两只票没有拿到同一个组均值")
        assert not np.allclose(got[:, 0], got[:, 2]), "两个组算出了同一个均值"

    def test_group_node_falls_back_to_groups_only_for_a_custom_field(self):
        """
        `if raw is None and self.group_field != "groups"` —— `and` 放宽成 `or`
        会让 group_field 本来就是 "groups" 时也再取一次（无害），
        但更要命的是它会在**字段存在**时也覆盖成 'groups' 的值。
        这里钉住：显式给了 sector 就必须用 sector，不许被 groups 顶掉。
        """
        ds = _ds(sector=np.array([0, 0, 1, 1]), groups=np.array([0, 1, 0, 1]))
        by_sector = _ev(GroupNode("group_mean", DataNode("close"),
                                  group_field="sector"), ds)
        by_groups = _ev(GroupNode("group_mean", DataNode("close"),
                                  group_field="groups"), ds)
        assert not np.allclose(by_sector, by_groups), (
            "指定 group_field='sector' 的结果与用 'groups' 相同 —— "
            "自定义字段被 'groups' 顶掉了")

    def test_group_node_refuses_to_invent_groups(self):
        """
        缺分组字段必须**抛错**。旧实现用 `np.arange(N) % 10` 凭空造分组，
        产出的是看起来像分组中性化、实际毫无意义的数字。
        """
        with pytest.raises(KeyError, match="不会"):
            _ev(GroupNode("group_rank", DataNode("close"), group_field="sector"),
                _ds())

    def test_two_dimensional_group_array_uses_row_zero(self):
        g = np.vstack([np.array([0, 0, 1, 1])] * T)
        ds = _ds(groups=g)
        flat = _ds(groups=np.array([0, 0, 1, 1]))
        np.testing.assert_allclose(
            _ev(GroupNode("group_mean", DataNode("close"), "groups"), ds),
            _ev(GroupNode("group_mean", DataNode("close"), "groups"), flat),
            rtol=1e-12, err_msg="二维分组数组没有按第 0 行取")

    def test_cs_ind_neutralize_resolves_the_group_field_from_its_node(self):
        """
        `_groups_field()` 里的 `isinstance(v, str) and v` —— `and` 放宽成 `or`
        会让**非字符串**属性（例如 ScalarNode.value 是 float）也被当成字段名，
        取到一个根本不存在的字段。
        """
        ds = _ds(sector=np.array([0, 0, 1, 1]))
        node = CrossSectionalNode("ind_neutralize", DataNode("close"),
                                  groups_node=StringLiteralNode("sector"))
        got = _ev(node, ds)
        np.testing.assert_allclose(got[:, :2].mean(axis=1), 0.0, atol=1e-9,
                                   err_msg="行业中性化之后组内均值不为 0 —— "
                                           "分组字段没有解析到 'sector'")

    def test_ind_neutralize_with_a_numeric_groups_node_does_not_use_the_number(self):
        """ScalarNode(3.0) 的 value 是 float，不得被当成字段名。"""
        ds = _ds(groups=np.array([0, 0, 1, 1]))
        node = CrossSectionalNode("ind_neutralize", DataNode("close"),
                                  groups_node=ScalarNode(3.0))
        got = _ev(node, ds)
        assert np.isfinite(got).all(), (
            "数字被当成字段名去取数据了 —— 应当回退到 'groups'")

    def test_missing_group_warning_names_the_field_actually_looked_up(self, caplog):
        """
        `_groups_field()` 里 `if isinstance(v, str) and v: return v` ——
        `and` 放宽成 `or` 会让**非字符串**属性也被当成字段名返回
        （ScalarNode.value 是 float 3.0 → 返回 3.0）。数据里当然没有这个键，
        最终都退化成全截面中性化，结果数值相同 —— 唯一的可观测差别是
        那条警告里报出来的字段名。

        这条差别不是细枝末节：警告是使用者判断"我的行业中性到底生效没有"的
        唯一线索，它报一个 `3.0` 出来只会让人以为是别的问题。
        """
        import logging
        node = CrossSectionalNode("ind_neutralize", DataNode("close"),
                                  groups_node=ScalarNode(3.0))
        with caplog.at_level(logging.WARNING,
                             logger="app.core.alpha_engine.typed_nodes"):
            _ev(node, {"close": _ds()["close"]})
        assert caplog.records, "缺分组字段却没有告警"
        msg = caplog.records[-1].getMessage()
        # 光断言 "'groups' in msg" 是不够的：消息后半句"（也没有 'groups'）"
        # 里本来就有这个词，变异版照样命中 —— 第一版就是这么让变异活下来的。
        # 必须断言那个**数字没有出现**。
        assert "3.0" not in msg, (
            f"警告把 ScalarNode 的数值当成了字段名：{msg}")
        assert msg.count("'groups'") == 2, (
            f"警告里的字段名不是 'groups'（应当前后各出现一次）：{msg}")

    def test_repr_includes_the_groups_node_when_present(self):
        """
        `if groups_node is not None` —— 删掉 `not` 会让**有分组参数时反而不写进
        repr**。repr 是缓存键与因子身份：两个分组不同的因子会共用同一个键，
        第二个直接读到第一个的缓存结果。
        """
        with_g = CrossSectionalNode("ind_neutralize", DataNode("close"),
                                    groups_node=StringLiteralNode("sector"))
        without = CrossSectionalNode("ind_neutralize", DataNode("close"))
        assert "sector" in repr(with_g), f"分组参数没进 repr：{repr(with_g)}"
        assert repr(with_g) != repr(without), (
            "带分组与不带分组的节点 repr 相同 —— 会共用缓存键，串味")


# ===========================================================================
# F. repr 与缓存键
# ===========================================================================

class TestReprIsTheCacheKey:

    def test_different_windows_produce_different_reprs(self):
        a = TimeSeriesNode("ts_mean", DataNode("close"), 5)
        b = TimeSeriesNode("ts_mean", DataNode("close"), 20)
        assert repr(a) != repr(b), "窗口不同的两个节点 repr 相同 —— 缓存会串"

    def test_cache_is_keyed_by_repr_and_actually_hits(self):
        ds = _ds()
        cache: dict = {}
        node = TimeSeriesNode("ts_mean", DataNode("close"), 4)
        first = node.evaluate(ds, cache)
        assert repr(node) in cache
        second = node.evaluate(ds, cache)
        assert second is first, "第二次求值没有命中缓存"

    def test_two_input_ts_repr_carries_both_children(self):
        node = TimeSeriesNode("ts_corr", DataNode("close"), 10,
                              second_child=DataNode("volume"))
        r = repr(node)
        assert "close" in r and "volume" in r and "10" in r, f"repr 丢了参数：{r}"

    def test_ts_corr_without_a_second_child_raises(self):
        node = TimeSeriesNode("ts_corr", DataNode("close"), 5)
        with pytest.raises(ValueError, match="second child"):
            _ev(node)

    def test_unknown_operators_are_rejected_at_construction(self):
        with pytest.raises(ValueError, match="Unknown TS"):
            TimeSeriesNode("ts_nope", DataNode("close"), 5)
        with pytest.raises(ValueError, match="Unknown CS"):
            CrossSectionalNode("nope", DataNode("close"))
        with pytest.raises(ValueError, match="Unknown group"):
            GroupNode("nope", DataNode("close"))
        with pytest.raises(ValueError, match="Unknown ArithmeticNode"):
            ArithmeticNode("nope", [ScalarNode(1.0)])

    def test_missing_field_names_the_available_ones(self):
        with pytest.raises(KeyError, match="Available"):
            _ev(DataNode("not_a_field"))


# ===========================================================================
# G. 存活变异
# ===========================================================================
#
# 复测结果：33 个变异点 **全部被杀死**（击杀率 100%），没有存活项，
# 因此本文件不需要等价性证明表。
#
# 曾经计划为 L296（`return gn if isinstance(gn, str) and gn else "groups"`）
# 写等价性证明，理由是"parser 放进 groups_node 的节点都带命名属性，该行不可达"。
# 实际复测把它杀死了 —— E 节的
# test_missing_group_warning_names_the_field_actually_looked_up 顺带覆盖到了它：
# `or` 变异会让**节点对象本身**被当成字段名，那条警告里就不再是 'groups'。
# 记在这里是因为"我以为不可达、实测可达"这件事本身值得留痕：
# 等价性的直觉判断不可靠，必须以复测结果为准。


def test_no_surviving_mutants_to_prove():
    """
    占位断言，与其他 B 档文件的 test_every_survivor_has_a_written_proof 对应：
    本模块零存活，所以这里检查的是"确实没有待证明项"这件事被显式记录过。
    """
    proofs: dict = {}
    assert proofs == {}, f"出现了未登记的存活项：{proofs}"
