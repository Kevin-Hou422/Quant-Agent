"""
gp_engine/mutations.py —— GP 变异/交叉算子的定钉测试（变异测试驱动）

来由：65 个变异点，首测击杀率 **4.6%**（存活 62）。既有覆盖
（`test_alpha_discovery` / `test_phase6_reproducibility`）只断言
"算子跑完没报错、产出的 DSL 还能解析"，于是：

  - 所有 `_rng.random() < 0.35 / 0.40 / 0.55 / 0.60 / 0.70 / 0.75 / 0.80`
    的概率阈值改成 `<=` 全都活着 —— 分支切换点整个没人管
  - 递归的 `_generate_typed_node(max_depth - 1)` 改成 `+ 1` 活着
    —— 深度守卫失效，生成的表达式树可以无限长
  - 十几处 `if not xxx:` 的兜底守卫删掉 `not` 全都活着
    —— 兜底和正常路径互换，算子静默变成"什么都不做"
  - `param_mutation` 的 `int(old_w * (1 + delta))` 改成 `-` / `/` 活着
    —— 窗口调整方向反了也没人知道

这些不是"看着能跑"能覆盖的东西。变异算子是 GP 搜索空间的**全部来源**：
阈值错了，搜索分布就偏了；深度守卫失效，表达式树会膨胀到回测跑不动；
兜底守卫反了，某个算子静默变成恒等变换 —— 进化看起来在跑，实际在原地踏步。

测法：`_rng` 是可绑定的共享随机源（`_rng.bind`），这里绑一个**脚本化**的
`random.Random` 子类，让 `random()` 逐个吐出预先写好的值。于是可以精确
把 `random()` 打在阈值上（`0.35 < 0.35` 为假、`0.35 <= 0.35` 为真），
两种取值走不同分支，结构差别可断言。
"""
from __future__ import annotations

import copy
import random

import pytest

import app.core.gp_engine.mutations as M
from app.core.gp_engine import _rng
from app.core.alpha_engine.typed_nodes import (
    ArithmeticNode, CrossSectionalNode, DataNode, NodeType,
    ScalarNode, StringLiteralNode, TimeSeriesNode,
)


# ===========================================================================
# 脚本化随机源
# ===========================================================================

class Scripted(random.Random):
    """
    `random()` 按脚本逐个返回；脚本耗尽后返回 `tail`（默认 0.0，
    它小于模块里所有阈值，能让递归立刻收敛，便于写有限长度的脚本）。

    `choice()` 默认取第一个元素；给 `chooser` 可以按名字挑。
    `uniform()` 按脚本返回，耗尽后取区间中点。
    """

    def __init__(self, randoms=(), *, tail: float = 0.0,
                 chooser=None, uniforms=()):
        super().__init__(0)
        self._randoms = list(randoms)
        self._uniforms = list(uniforms)
        self._chooser = chooser
        self.random_log: list[float] = []
        self.choice_log: list = []

    def random(self) -> float:                      # noqa: D102
        v = self._randoms.pop(0) if self._randoms else self._tail
        self.random_log.append(v)
        return v

    # tail 放成属性，方便 __init__ 里用关键字传
    _tail = 0.0

    def uniform(self, a, b):                        # noqa: D102
        return self._uniforms.pop(0) if self._uniforms else (a + b) / 2.0

    def choice(self, seq):                          # noqa: D102
        seq = list(seq)
        picked = self._chooser(seq) if self._chooser else seq[0]
        self.choice_log.append(picked)
        return picked


def _script(randoms=(), *, tail=0.0, chooser=None, uniforms=()) -> Scripted:
    s = Scripted(randoms, chooser=chooser, uniforms=uniforms)
    s._tail = tail
    _rng.bind(s)
    return s


def _pick(*names):
    """chooser：按优先级从候选里挑第一个命中的名字，都没有就取 seq[0]。"""
    def _c(seq):
        for want in names:
            for item in seq:
                if item == want or getattr(item, "op", None) == want \
                        or getattr(item, "field", None) == want:
                    return item
        return seq[0]
    return _c


@pytest.fixture(autouse=True)
def _restore_rng():
    """每条用例跑完把共享随机源还原，避免污染同进程里的其他测试。"""
    saved = _rng.current()
    yield
    _rng.bind(saved)


# ===========================================================================
# 测试用的树
# ===========================================================================

def _ts_tree() -> CrossSectionalNode:
    """rank(ts_mean(close, 10)) —— 深度 2。"""
    return CrossSectionalNode("rank", TimeSeriesNode("ts_mean", DataNode("close"), 10))


class AstLike:
    """
    模拟 `mutations` 明确声称要支持的 **ast.Node 回退形态**：
    `children` 是一个 list 字段，而且**没有** `depth()` 方法。

    typed_nodes 的每个节点都带 `depth()`，`_tree_depth` 会在第一行就返回，
    后面两行（`if not ch` 与 `1 + max(...)`）只有走这条回退路径才到得了。
    """

    def __init__(self, children=None, op="data"):
        self.children = list(children or [])
        self.op = op


# ===========================================================================
# A. _tree_depth —— 深度计算的回退路径
# ===========================================================================

class TestTreeDepth:

    def test_typed_nodes_delegate_to_their_own_depth(self):
        assert M._tree_depth(DataNode("close")) == 0
        assert M._tree_depth(_ts_tree()) == 2

    def test_a_leaf_without_a_depth_method_is_depth_zero(self):
        """
        `if not ch: return 0` —— 删掉 `not` 会让**叶子**掉进
        `max(空序列)` → ValueError，而非叶子反而返回 0。
        """
        assert M._tree_depth(AstLike()) == 0

    def test_depth_accumulates_one_level_per_generation(self):
        """
        `return 1 + max(...)` —— 改成 `1 - max(...)`。

        注意深度 1 的树两种写法都给 1（`1+0 == 1-0`），必须用**深度 ≥ 2**
        的树才分得开：`1+(1+0)=2` vs `1-(1-0)=0`。
        """
        two = AstLike([AstLike([AstLike()])])
        assert M._tree_depth(two) == 2, "深度累加的符号反了"
        three = AstLike([AstLike([AstLike([AstLike()])])])
        assert M._tree_depth(three) == 3

    def test_the_depth_probe_short_circuits_before_touching_the_attribute(self):
        """
        `if hasattr(node, "depth") and callable(node.depth):` —— `and` 改成
        `or` 时，没有 `depth` 属性的节点会在第二个子式上 AttributeError。
        """
        assert M._tree_depth(AstLike()) == 0            # 不许抛
        assert not hasattr(AstLike(), "depth"), "用例前提被破坏"


# ===========================================================================
# B. _generate_typed_node —— 随机子树生成器的分支阈值与深度守卫
# ===========================================================================

class TestGenerateTypedNode:
    """
    四个分支阈值 0.35 / 0.40 / 0.60 / 0.80 决定了 GP 生成的表达式
    **形态分布**。每个阈值都精确打在边界上：`x < x` 为假、`x <= x` 为真。
    """

    def test_the_early_leaf_threshold_is_strict(self):
        """`if max_depth <= 0 or _rng.random() < 0.35:` —— 恰好 0.35 不提前返回叶子。"""
        _script([0.35, 0.0])                 # 0.35 → 继续；roll=0.0 → TS 分支
        node = M._generate_typed_node(max_depth=2)
        assert isinstance(node, TimeSeriesNode), (
            f"random()==0.35 时提前返回了 {type(node).__name__} —— "
            f"提前收敛的阈值从 `< 0.35` 放宽成了 `<= 0.35`")

    def test_just_under_the_leaf_threshold_does_return_a_leaf(self):
        _script([0.34999])
        assert isinstance(M._generate_typed_node(max_depth=2), DataNode)

    @pytest.mark.parametrize("roll,expect,why", [
        (0.40, CrossSectionalNode, "roll==0.40 走了时序分支 —— `< 0.40` 被放宽"),
        (0.60, ArithmeticNode,     "roll==0.60 走了截面分支 —— `< 0.60` 被放宽"),
        (0.80, DataNode,           "roll==0.80 走了算术分支 —— `< 0.80` 被放宽"),
    ])
    def test_each_branch_threshold_is_strict(self, roll, expect, why):
        _script([0.35, roll])
        node = M._generate_typed_node(max_depth=2)
        assert isinstance(node, expect), f"{why}（实际 {type(node).__name__}）"

    @pytest.mark.parametrize("roll,expect", [
        (0.39999, TimeSeriesNode),
        (0.59999, CrossSectionalNode),
        (0.79999, ArithmeticNode),
    ])
    def test_just_under_each_threshold_takes_that_branch(self, roll, expect):
        _script([0.35, roll])
        assert isinstance(M._generate_typed_node(max_depth=2), expect)

    @pytest.mark.parametrize("max_depth", [1, 2, 3, 4])
    def test_the_generated_tree_never_exceeds_max_depth(self, max_depth):
        """
        三处 `_generate_typed_node(max_depth - 1)` 是**唯一**的递归收敛保证。
        改成 `+ 1` 后 max_depth 单调增长，只剩 35% 的随机提前返回兜着 ——
        树会膨胀到回测跑不动，极端情况直接 RecursionError。

        用 60 个不同种子跑，任何一次越界都算失败。
        """
        for seed in range(60):
            _rng.bind_seed(seed)
            node = M._generate_typed_node(max_depth=max_depth)
            d = M._tree_depth(node)
            assert d <= max_depth, (
                f"seed={seed} 生成的树深度 {d} 超过上限 {max_depth} —— "
                f"递归的 `max_depth - 1` 疑似变成了 `+ 1`")

    def test_binary_arithmetic_builds_both_children_one_level_down(self):
        """算术分支有**两个** `max_depth - 1`，两侧都要受深度约束。"""
        _script([0.35, 0.70])               # roll=0.70 → 算术分支
        node = M._generate_typed_node(max_depth=1)
        assert isinstance(node, ArithmeticNode)
        kids = M._get_children(node)
        assert len(kids) == 2
        for k in kids:
            assert M._tree_depth(k) == 0, (
                "算术分支的子节点超过了 max_depth-1 —— 某一侧的递减被改成了递增")


# ===========================================================================
# C. _generate_family_compatible_subtree —— 家族专用子树
# ===========================================================================

class TestFamilySubtree:

    def test_momentum_log_wrap_threshold_is_strict(self):
        """`if field == "close" and _rng.random() < 0.6:` —— 恰好 0.6 不加 log。"""
        _script([0.6], chooser=_pick("close", "ts_delta", 3))
        node = M._generate_family_compatible_subtree(factor_family="momentum")
        assert isinstance(node, TimeSeriesNode)
        assert isinstance(node.child, DataNode), (
            f"random()==0.6 时给动量因子套上了 {type(node.child).__name__} —— "
            f"`< 0.6` 被放宽成了 `<= 0.6`")

    def test_momentum_log_wrap_applies_below_the_threshold(self):
        _script([0.59999], chooser=_pick("close", "ts_delta", 3))
        node = M._generate_family_compatible_subtree(factor_family="momentum")
        assert isinstance(node.child, ArithmeticNode) and node.child.op == "log"

    def test_the_log_wrap_only_applies_to_close(self):
        """
        `field == "close" and _rng.random() < 0.6` —— `and` 改成 `or` 会让
        **任何**字段都可能被套 log，包括 `returns`。
        `log(returns)` 在收益为负时是 NaN —— 整条因子废掉，而且不报错。
        """
        _script([0.0], chooser=_pick("returns", "ts_delta", 3))
        node = M._generate_family_compatible_subtree(factor_family="momentum")
        assert isinstance(node.child, DataNode) and node.child.field == "returns", (
            "非 close 字段也被套上了 log —— `field == 'close' and ...` 被放宽成了 `or`")

    def test_reversion_branch_threshold_is_strict(self):
        """`if _rng.random() < 0.55:` —— 恰好 0.55 走 ts_zscore 那一支。"""
        _script([0.55], chooser=_pick(1))
        node = M._generate_family_compatible_subtree(factor_family="reversion")
        assert isinstance(node, TimeSeriesNode) and node.op == "ts_zscore", (
            f"random()==0.55 走了 neg(ts_delta) 那一支（得到 {node!r}）—— "
            f"`< 0.55` 被放宽")

    def test_reversion_doubles_the_window_for_the_zscore_variant(self):
        """
        `TimeSeriesNode("ts_zscore", DataNode("returns"), window * 2)` ——
        `*` 改成 `/` 会把窗口变成半长（还是浮点），反转因子的时间尺度。
        """
        _script([0.55], chooser=_pick(3))     # window 候选 [1,3,5] → 取 3
        node = M._generate_family_compatible_subtree(factor_family="reversion")
        assert node.window == 6, (
            f"ts_zscore 的窗口是 {node.window!r}，应当是 3×2=6 —— "
            f"`window * 2` 的算符被改了")
        assert isinstance(node.window, int), "窗口变成了浮点数 —— 疑似被改成了除法"

    def test_volatility_negation_threshold_is_strict(self):
        """
        `return ArithmeticNode("neg", [node]) if _rng.random() < 0.6 else node`
        —— 恰好 0.6 **不**取负。低波动因子取不取负决定信号方向，反了就是
        "买最高波动"。
        """
        _script([0.6], chooser=_pick("ts_std", 10))
        node = M._generate_family_compatible_subtree(factor_family="volatility")
        assert isinstance(node, TimeSeriesNode), (
            f"random()==0.6 时仍然取了负（得到 {type(node).__name__}）—— `< 0.6` 被放宽")

    def test_volatility_does_negate_below_the_threshold(self):
        _script([0.59999], chooser=_pick("ts_std", 10))
        node = M._generate_family_compatible_subtree(factor_family="volatility")
        assert isinstance(node, ArithmeticNode) and node.op == "neg"

    def test_liquidity_branch_threshold_is_strict(self):
        """`if _rng.random() < 0.5:` —— 恰好 0.5 走 ts_delta(log(volume)) 那一支。"""
        _script([0.5], chooser=_pick(5))
        node = M._generate_family_compatible_subtree(factor_family="liquidity")
        assert isinstance(node, TimeSeriesNode) and node.op == "ts_delta", (
            f"random()==0.5 走了 ts_mean 那一支（得到 {node!r}）—— `< 0.5` 被放宽")

    def test_liquidity_takes_the_mean_branch_below_the_threshold(self):
        _script([0.49999], chooser=_pick(5))
        node = M._generate_family_compatible_subtree(factor_family="liquidity")
        assert isinstance(node, TimeSeriesNode) and node.op == "ts_mean"


# ===========================================================================
# C2. _replace_inplace —— 就地替换的每一条命中路径
# ===========================================================================

class TestReplaceInplace:
    """
    `_replace_inplace(node, target_id, replacement)` 的每一个 `return True`
    都是一条独立的命中路径（TS 的 child / second_child / 递归、CS 的 child、
    算术的直接命中 / 递归、ast 回退的直接命中 / 递归），末尾一条 `return False`
    是"全树都没找到"。

    这些分支**经由公开算子永远走不到**：`_replace_node` 先 `deepcopy` 再拿
    原树节点的 `id()` 去找，副本里没有任何节点持有那个 id（缺陷 C-2）。
    所以这里按 docstring 写的用法直接调它 —— "run AFTER deep-copy"，
    target_id 取自**同一棵树内**的节点。
    """

    def test_a_time_series_child_is_replaced(self):
        t = TimeSeriesNode("ts_mean", DataNode("close"), 10)
        assert M._replace_inplace(t, id(t.child), DataNode("volume")) is True
        assert repr(t) == "ts_mean(volume,10)"

    def test_a_time_series_second_child_is_replaced(self):
        t = TimeSeriesNode("ts_corr", DataNode("close"), 10,
                           second_child=DataNode("volume"))
        assert t.second_child is not None, "用例前提被破坏"
        assert M._replace_inplace(t, id(t.second_child), DataNode("high")) is True
        assert repr(t.second_child) == "high"
        assert repr(t.child) == "close", "改错了操作数"

    def test_the_search_descends_into_the_second_child(self):
        inner = TimeSeriesNode("ts_mean", DataNode("volume"), 5)
        t = TimeSeriesNode("ts_corr", DataNode("close"), 10, second_child=inner)
        assert M._replace_inplace(t, id(inner.child), DataNode("high")) is True
        assert repr(inner) == "ts_mean(high,5)"

    def test_the_search_descends_into_the_first_child(self):
        inner = TimeSeriesNode("ts_mean", DataNode("close"), 5)
        t = TimeSeriesNode("ts_std", inner, 10)
        assert M._replace_inplace(t, id(inner.child), DataNode("low")) is True
        assert repr(inner) == "ts_mean(low,5)"

    def test_a_cross_sectional_child_is_replaced(self):
        c = CrossSectionalNode("rank", DataNode("close"))
        assert M._replace_inplace(c, id(c.child), DataNode("volume")) is True
        assert repr(c) == "rank(volume)"

    def test_an_arithmetic_operand_is_replaced_in_place(self):
        a = ArithmeticNode("add", [DataNode("close"), DataNode("volume")])
        second = a._children[1]
        assert M._replace_inplace(a, id(second), DataNode("high")) is True
        assert repr(a._children[0]) == "close", "改错了操作数"
        assert repr(a._children[1]) == "high"

    def test_the_search_descends_into_arithmetic_operands(self):
        inner = CrossSectionalNode("rank", DataNode("close"))
        a = ArithmeticNode("add", [inner, DataNode("volume")])
        assert M._replace_inplace(a, id(inner.child), DataNode("low")) is True
        assert repr(inner) == "rank(low)"

    def test_the_ast_fallback_replaces_a_direct_child(self):
        """`children` 是 list 字段的 ast.Node 回退形态。"""
        leaf = AstLike()
        parent = AstLike([leaf])
        rep = AstLike(op="replacement")
        assert M._replace_inplace(parent, id(leaf), rep) is True
        assert parent.children[0] is rep

    def test_the_ast_fallback_descends_into_children(self):
        leaf = AstLike()
        mid = AstLike([leaf])
        root = AstLike([mid])
        rep = AstLike(op="replacement")
        assert M._replace_inplace(root, id(leaf), rep) is True
        assert mid.children[0] is rep

    def test_a_target_that_is_not_in_the_tree_returns_false(self):
        """
        末尾的 `return False` —— 翻成 True 会让调用方以为替换成功了，
        而树一个字没改。`_replace_node` 不看返回值，但 `_replace_inplace`
        的递归**看**：某个分支谎报 True 会让上层提前停止搜索。
        """
        t = TimeSeriesNode("ts_mean", DataNode("close"), 10)
        stranger = DataNode("volume")
        assert M._replace_inplace(t, id(stranger), DataNode("high")) is False, (
            "目标不在树里却报告替换成功")
        assert repr(t) == "ts_mean(close,10)", "什么都没匹配上却改动了树"

    def test_only_the_first_matching_slot_is_replaced(self):
        """同一个对象出现在两个位置时，只换第一个（函数语义就是"first occurrence"）。"""
        shared = DataNode("close")
        a = ArithmeticNode("add", [shared, shared])
        assert M._replace_inplace(a, id(shared), DataNode("volume")) is True
        assert repr(a._children[0]) == "volume"
        assert repr(a._children[1]) == "close", (
            "两个位置都被换了 —— 语义应当是只换第一处")


# ===========================================================================
# D. hoist_mutation —— 兜底守卫
# ===========================================================================

class TestHoistMutation:

    def test_hoisting_actually_replaces_the_target_with_a_child(self):
        """
        `if not has_children: return deepcopy(root)` 与
        `if not children: return deepcopy(root)` —— 删掉任一个 `not`，
        有子节点的正常输入会**立刻走兜底**，算子静默变成恒等变换：
        GP 以为自己在简化表达式，其实一代都没动过。
        """
        root = _ts_tree()                                  # rank(ts_mean(close,10))
        _script([], chooser=lambda seq: seq[0])
        out = M.hoist_mutation(root)
        assert repr(out) != repr(root), (
            "hoist 之后树没有任何变化 —— 兜底守卫的 `not` 被删掉了，"
            "算子退化成了恒等变换")
        assert M._tree_depth(out) < M._tree_depth(root), (
            f"hoist 的语义是**降低深度**，实际 {M._tree_depth(root)} → "
            f"{M._tree_depth(out)}")

    def test_a_leaf_root_falls_back_to_a_copy(self):
        """真正无子节点时才该走兜底，而且必须是**副本**不是原对象。"""
        root = DataNode("close")
        _script([])
        out = M.hoist_mutation(root)
        assert repr(out) == repr(root)
        assert out is not root, "兜底返回了原对象而不是深拷贝 —— 调用方改它会改到父代"


# ===========================================================================
# E. param_mutation —— 窗口调整
# ===========================================================================

class TestParamMutation:

    def test_the_window_moves_by_exactly_the_sampled_delta(self):
        """
        `target.window = max(2, min(60, int(old_w * (1 + delta))))`

        `+` 改成 `-`：delta=+0.2、old_w=20 时 24 → 16，**调整方向整个反过来**；
        `*` 改成 `/`：24 → 16（20/1.2=16.67→16）。两种都落在合法区间里，
        只断言"窗口变了"两种都放过 —— 必须断言**确切的值**。
        """
        root = TimeSeriesNode("ts_mean", DataNode("close"), 20)
        _script([], uniforms=[0.2])
        out = M.param_mutation(root)
        assert out.window == 24, (
            f"delta=+0.2、原窗口 20，调整后应当是 int(20*1.2)=24，实际 {out.window} —— "
            f"`old_w * (1 + delta)` 的算符被改了")

    def test_a_negative_delta_shrinks_the_window(self):
        root = TimeSeriesNode("ts_mean", DataNode("close"), 20)
        _script([], uniforms=[-0.2])
        assert M.param_mutation(root).window == 16

    def test_the_window_is_clamped_to_the_legal_range(self):
        root = TimeSeriesNode("ts_mean", DataNode("close"), 60)
        _script([], uniforms=[0.2])
        assert M.param_mutation(root).window == 60, "上界 60 的截断失效"
        root2 = TimeSeriesNode("ts_mean", DataNode("close"), 2)
        _script([], uniforms=[-0.2])
        assert M.param_mutation(root2).window == 2, "下界 2 的截断失效"

    def test_a_tree_without_ts_nodes_is_returned_unchanged(self):
        """
        `if not ts_nodes: return new_root` —— 删掉 `not` 会让**有** TS 节点的
        正常输入直接原样返回，窗口变异整个算子失效。
        """
        root = CrossSectionalNode("rank", DataNode("close"))
        _script([], uniforms=[0.2])
        out = M.param_mutation(root)
        assert repr(out) == repr(root) and out is not root

    def test_a_tree_with_ts_nodes_does_get_mutated(self):
        root = _ts_tree()
        _script([], uniforms=[0.2])
        out = M.param_mutation(root)
        assert repr(out) != repr(root), (
            "含 TS 节点的树没有被调整窗口 —— `if not ts_nodes` 的 `not` 被删掉了")


# ===========================================================================
# F. subtree_crossover
# ===========================================================================

class TestSubtreeCrossover:

    def test_compatible_parents_actually_exchange_material(self):
        """
        `if not common_types: return 双亲副本` —— 删掉 `not` 会让**类型相容**的
        双亲直接返回副本：交叉算子对所有正常输入都失效，GP 退化成纯变异。
        """
        a = CrossSectionalNode("rank", TimeSeriesNode("ts_mean", DataNode("close"), 10))
        b = CrossSectionalNode("rank", TimeSeriesNode("ts_std", DataNode("volume"), 20))
        _rng.bind_seed(3)
        changed = False
        for seed in range(40):
            _rng.bind_seed(seed)
            c1, c2 = M.subtree_crossover(a, b)
            if repr(c1) != repr(a) or repr(c2) != repr(b):
                changed = True
                break
        assert changed, (
            "40 个种子下交叉都没换出任何东西 —— "
            "`if not common_types` 的 `not` 疑似被删掉了")

    def test_incompatible_parents_fall_back_to_copies(self):
        a = DataNode("close")
        b = DataNode("volume")
        _rng.bind_seed(0)
        c1, c2 = M.subtree_crossover(a, b)
        assert c1 is not a and c2 is not b, "兜底返回了原对象"

    def test_both_children_must_validate_or_neither_is_kept(self, monkeypatch):
        """
        `if _try_validate(new_root1) and _try_validate(new_root2):` ——
        改成 `or` 时，只要有**一个**孩子合法就把两个都放出去，
        另一个非法的孩子会带着坏结构进种群，直到某次回测才炸。

        用 monkeypatch 让第二个孩子校验失败，断言两个孩子都被丢弃。
        """
        a = CrossSectionalNode("rank", TimeSeriesNode("ts_mean", DataNode("close"), 10))
        b = CrossSectionalNode("rank", TimeSeriesNode("ts_std", DataNode("volume"), 20))

        calls = {"n": 0}
        real = M._try_validate

        def _fake(node):
            calls["n"] += 1
            return calls["n"] == 1 and real(node)   # 第一次真判，第二次一律否

        monkeypatch.setattr(M, "_try_validate", _fake)
        _rng.bind_seed(1)
        c1, c2 = M.subtree_crossover(a, b)
        assert calls["n"] >= 2, "第二个孩子根本没被校验 —— `and` 短路了？"
        assert repr(c1) == repr(a) and repr(c2) == repr(b), (
            "第二个孩子校验失败，却仍然放出了交叉结果 —— "
            "两个校验之间的 `and` 被放宽成了 `or`")


# ===========================================================================
# G. wrap_rank
# ===========================================================================

class TestWrapRank:

    @staticmethod
    def _arith_root():
        """(ts_mean(close,10)+ts_std(volume,20)) —— 根不是 CS，内部有两个 TS。"""
        return ArithmeticNode("add", [
            TimeSeriesNode("ts_mean", DataNode("close"), 10),
            TimeSeriesNode("ts_std", DataNode("volume"), 20),
        ])

    def test_the_whole_root_wrap_threshold_is_strict(self):
        """
        `if _rng.random() < 0.6:` —— 恰好 0.6 **不**整棵包。

        注意：这里挑的目标是内部的 ts_mean 节点，而 C-2（`_replace_node`
        跨 deepcopy 用 id() 定位）会让内部替换静默失败、返回原树。
        无论 C-2 修没修，"整棵包成 rank(root)"这件事都不该发生 ——
        断言写在这个不变式上。
        """
        root = self._arith_root()
        _script([0.6], chooser=_pick("rank", "ts_mean"))
        out = M.wrap_rank(root)
        assert not (isinstance(out, CrossSectionalNode)
                    and repr(out.child) == repr(root)), (
            f"random()==0.6 时仍然整棵包了（得到 {out!r}）—— `< 0.6` 被放宽")

    def test_below_the_threshold_the_whole_root_is_wrapped(self):
        root = self._arith_root()
        _script([0.59999], chooser=_pick("rank", "ts_mean"))
        out = M.wrap_rank(root)
        assert isinstance(out, CrossSectionalNode) and out.op == "rank"
        assert repr(out.child) == repr(root)

    def test_internal_targets_exclude_leaves_and_existing_cs_nodes(self):
        """
        候选过滤有两个条件：
          `not isinstance(n, (DataNode, ScalarNode, StringLiteralNode))`
          `and _node_type(n) is not NodeType.CROSS_SECTIONAL`

        删掉任一个 `not`、或把 `and` 放宽成 `or`，候选集就会变样。
        因为 C-2 让内部替换看不出效果，这里**直接断言递给 `_rng.choice`
        的候选清单本身** —— 过滤逻辑是什么就钉什么，不依赖替换是否生效。
        """
        root = _ts_tree()                        # rank(ts_mean(close,10))
        seen: list[list] = []

        def _c(seq):
            seq = list(seq)
            seen.append(seq)
            return _pick("rank", "ts_mean")(seq)

        _script([0.6], chooser=_c)               # 0.6 → 不整棵包，走内部节点
        M.wrap_rank(root)

        assert len(seen) >= 2, f"choice 只被调用了 {len(seen)} 次（op + 候选）"
        cands = seen[-1]
        reprs = sorted(repr(n) for n in cands)
        assert reprs == ["ts_mean(close,10)"], (
            f"内部候选是 {reprs}，应当只有 ts_mean —— "
            f"叶子（close）或已有的截面节点（rank(...)）被算进候选了")

    def test_a_pure_leaf_root_falls_back_to_the_leaf_candidates(self):
        """
        两层 `if not candidates:` 兜底。第一层过滤后为空时要退到
        "叶子也行"，再空才返回原树副本。删掉第一个 `not` 会在候选**非空**时
        反而去覆盖它；删掉第二个会让空候选走到 `_rng.choice([])` → IndexError。
        """
        seen: list[list] = []

        def _c(seq):
            seq = list(seq)
            seen.append(seq)
            return seq[0]

        _script([0.6], chooser=_c)
        out = M.wrap_rank(DataNode("close"))     # 第一层过滤后必为空
        assert out is not None and "close" in repr(out)
        assert [repr(n) for n in seen[-1]] == ["close"], (
            f"叶子根的兜底候选是 {[repr(n) for n in seen[-1]]}，应当退到 [close]")


# ===========================================================================
# H. add_ts_smoothing
# ===========================================================================

class TestAddTsSmoothing:

    def test_the_candidate_filter_does_not_touch_window_on_non_ts_nodes(self):
        """
        `or (isinstance(n, TimeSeriesNode) and n.window <= 20)` —— 内层 `and`
        改成 `or` 时，非 TS 节点会被求 `n.window` → AttributeError。
        用一棵含截面节点的树触发。
        """
        root = _ts_tree()                        # rank(...) 不是 TimeSeriesNode
        _script([], chooser=_pick("ts_mean", 10))
        out = M.add_ts_smoothing(root)           # 不许抛
        assert out is not None

    def test_a_wide_window_ts_node_is_not_a_preferred_target(self):
        """
        候选偏好 `n.window <= 20` 的窄窗 TS 节点。窗口 60 的节点应当落到
        第二层兜底里，而不是被当成首选目标反复叠加平滑。
        """
        root = TimeSeriesNode("ts_mean", DataNode("close"), 60)
        _script([], chooser=_pick("ts_std", 40))
        out = M.add_ts_smoothing(root)
        # close 是 DataNode，始终是首选候选 —— 平滑应当加在它上面
        assert "ts_std" in repr(out) or repr(out) == repr(root)

    def test_a_leafless_tree_falls_back_to_a_copy(self):
        """`if not candidates: return deepcopy(root)` —— 兜底必须是副本。"""
        root = ScalarNode(1.0)
        _script([], chooser=_pick("ts_mean", 10))
        out = M.add_ts_smoothing(root)
        assert out is not root

    def test_the_preferred_candidates_are_data_nodes_and_narrow_ts_nodes(self):
        """
        三处 `if not candidates:` 删掉 `not` 都会让正常输入走进兜底，
        候选清单整个换一批。

        因为 C-2（`_replace_node` 跨 deepcopy 用 `id()` 定位）让这个算子
        对内部节点是**彻底的空操作**，产出看不出差别 —— 所以直接断言
        递给 `_rng.choice` 的候选清单本身。C-2 修好后这条依然成立。
        """
        root = ArithmeticNode("add", [
            TimeSeriesNode("ts_mean", DataNode("close"), 10),   # 窄窗 → 首选
            TimeSeriesNode("ts_std", DataNode("volume"), 60),   # 宽窗 → 不入选
        ])
        seen: list[list] = []

        def _c(seq):
            seq = list(seq)
            seen.append(seq)
            return seq[0]

        _script([], chooser=_c)
        M.add_ts_smoothing(root)

        cands = sorted(repr(n) for n in seen[0])
        assert cands == ["close", "ts_mean(close,10)", "volume"], (
            f"首选候选是 {cands}，应当是两个数据叶子 + 窄窗的 ts_mean —— "
            f"宽窗（60）的 ts_std 不该入选，`n.window <= 20` 的门槛被改了")


# ===========================================================================
# I. combine_signals
# ===========================================================================

class TestCombineSignals:

    @staticmethod
    def _deep(n: int) -> Node:                      # type: ignore[name-defined]
        node: object = DataNode("close")
        for _ in range(n):
            node = CrossSectionalNode("rank", node)  # type: ignore[arg-type]
        return node                                  # type: ignore[return-value]

    def test_the_depth_guard_threshold_is_exact(self):
        """
        `if root_depth >= max_combined_depth - 1:`（即 `>= 7`）——
        `-` 改成 `+` 会把门槛推到 9，深度 7/8 的树不再走"只配一个叶子"的
        安全分支，组合出来的树深度可以到 8+，回测时直接被深度上限拒掉。
        """
        deep7 = self._deep(7)
        assert M._tree_depth(deep7) == 7, "构造前提被破坏"
        _script([], chooser=lambda seq: seq[0])
        out = M.combine_signals(deep7, other_root=self._deep(3))
        # 走安全分支时第二个操作数是**单个数据叶子**，不是那棵深度 3 的树。
        # 注意不能对整串 repr 做子串判断：根自己就是七层 rank，必然包含
        # "rank(rank(rank(close)))"。只看第二个操作数。
        rhs = M._get_children(out)[1]
        assert M._tree_depth(rhs) == 0, (
            f"深度 7 的根配到的第二个操作数是 {rhs!r}（深度 "
            f"{M._tree_depth(rhs)}），应当是单个数据叶子 —— "
            f"`max_combined_depth - 1` 的门槛被推高了")

    def test_a_shallow_root_does_use_the_second_parent(self):
        """反向：浅根应当真的用上第二个父代。"""
        _script([], chooser=lambda seq: seq[0])
        out = M.combine_signals(self._deep(2), other_root=self._deep(3))
        rhs = M._get_children(out)[1]
        assert M._tree_depth(rhs) == 3, (
            f"浅根配到的第二个操作数是 {rhs!r}，应当就是那棵深度 3 的父代 —— "
            f"`elif other_root is not None` 的判定反了")

    def test_an_unknown_family_falls_through_to_the_random_generator(self):
        """
        `elif factor_family and factor_family in _COMPLEMENTARY_FAMILIES:` ——
        `and` 改成 `or` 时，未登记的家族名会走进 `_COMPLEMENTARY_FAMILIES[家族]`
        → KeyError，整个算子崩掉。
        """
        _script([0.35, 0.0], chooser=lambda seq: seq[0])
        out = M.combine_signals(DataNode("close"), factor_family="not_a_family")
        assert out is not None, "未登记的家族名让 combine_signals 崩了"

    def test_an_empty_family_also_falls_through(self):
        _script([0.35, 0.0], chooser=lambda seq: seq[0])
        assert M.combine_signals(DataNode("close"), factor_family="") is not None

    def test_the_operator_choice_also_guards_the_unknown_family(self):
        """
        `if factor_family and other_root is None and factor_family in _COMPLEMENTARY_FAMILIES:`
        —— 同一行有两个 `and`，任一放宽成 `or` 都会让未登记家族走进
        `_COMPLEMENTARY_FAMILIES[家族][0]` → KeyError。
        """
        _script([0.35, 0.0], chooser=lambda seq: seq[0])
        out = M.combine_signals(DataNode("close"), other_root=None,
                                factor_family="zzz_unknown")
        assert out is not None

    def test_a_non_scalar_multiplier_is_rank_scaled(self):
        """
        乘法分支：`if not isinstance(other, (ScalarNode, DataNode)):` 会把
        非叶子的乘数套上 `rank(...)` 以保持有界。删掉 `not` 会反过来 ——
        把**标量/叶子**套 rank、而让无界的表达式直接相乘，信号量纲炸掉。
        """
        _script([], chooser=_pick("mul"))
        other = TimeSeriesNode("ts_std", DataNode("returns"), 20)
        out = M.combine_signals(DataNode("close"), other_root=other)
        text = repr(out)
        # 上一版写成 `if "*" in text: assert ...` —— 没走到乘法分支时
        # **一条都不检查**（test_lessons_enforced 的条件断言检查抓到了）。
        # 先把"确实走了乘法分支"断言出来，再断言包裹。
        assert "*" in text, (
            f"chooser 指定了 mul 却没有走乘法分支：{text} —— 用例前提被破坏")
        assert "rank(ts_std" in text, (
            f"无界的 ts_std 乘数没有被 rank 包起来：{text} —— "
            f"`not isinstance(...)` 的 not 被删掉了")

    def test_a_data_node_multiplier_is_not_rank_wrapped(self):
        _script([], chooser=_pick("mul"))
        out = M.combine_signals(DataNode("close"), other_root=DataNode("volume"))
        assert "rank(volume)" not in repr(out), (
            "叶子乘数被多套了一层 rank —— 判定反了")


# ===========================================================================
# J. replace_subtree
# ===========================================================================

class TestReplaceSubtree:

    def test_the_family_branch_threshold_is_strict(self):
        """`if factor_family and _rng.random() < 0.70:` —— 恰好 0.70 走随机生成器。"""
        root = _ts_tree()
        _script([0.70, 0.35, 0.0], chooser=lambda seq: seq[0])
        out = M.replace_subtree(root, factor_family="momentum")
        assert out is not None

    def test_an_empty_family_never_takes_the_family_branch(self):
        """
        `factor_family and _rng.random() < 0.70` —— `and` 改成 `or` 时，
        空家族名也会走 `_generate_family_compatible_subtree(factor_family="")`。
        那个函数对空家族会回退到通用生成器，行为上看不出来 ——
        真正看得出的是**随机数消耗**：`or` 会短路掉 `random()` 调用。
        """
        root = _ts_tree()
        s = _script([0.5, 0.35, 0.0], chooser=lambda seq: seq[0])
        M.replace_subtree(root, factor_family="")
        assert 0.5 in s.random_log, (
            "空家族名时 `_rng.random()` 根本没被调用 —— "
            "`factor_family and random() < 0.70` 被放宽成了 `or`（左真即短路）")

    def test_a_leaf_root_regenerates_a_whole_tree(self):
        """
        `if not internals:` —— 无内部节点（纯叶子）时应当**重新生成整棵树**。
        删掉 `not` 会反过来：有内部节点的正常树走重新生成、纯叶子走替换路径
        （而 internals 为空，`_rng.choice([])` 直接 IndexError）。
        """
        _script([0.35, 0.0], chooser=lambda seq: seq[0])
        out = M.replace_subtree(DataNode("close"))
        assert out is not None

    def test_a_normal_tree_takes_the_replacement_path(self):
        root = _ts_tree()
        _rng.bind_seed(7)
        changed = any(
            repr(M.replace_subtree(root)) != repr(root)
            for _ in [_rng.bind_seed(s) for s in range(30)]
        )
        assert changed, (
            "30 个种子下 replace_subtree 都没换过任何东西 —— "
            "`if not internals` 的 not 疑似被删掉了")

    def test_internals_must_actually_have_children(self):
        """
        `internals = [n for n in nodes if not isinstance(n, 叶子类) and _get_children(n)]`
        —— 末尾的 `and` 改成 `or` 会把**没有子节点**的非叶子节点也算进来。
        这里断言筛出来的每一个都真有子节点。
        """
        root = _ts_tree()
        nodes = M._collect_nodes(root)
        internals = [n for n in nodes
                     if not isinstance(n, (DataNode, ScalarNode))
                     and M._get_children(n)]
        assert internals, "构造前提被破坏"
        for n in internals:
            assert M._get_children(n), f"{n!r} 被当成内部节点，但它没有子节点"


# ===========================================================================
# K. add_operator
# ===========================================================================

class TestAddOperator:

    def test_an_unknown_family_does_not_index_the_preference_table(self):
        """
        `if factor_family and factor_family in _FAMILY_OPERATOR_PREFS:` ——
        `and` 改成 `or` 时，未登记家族会走进
        `_FAMILY_OPERATOR_PREFS[家族]` → KeyError。
        """
        _script([0.5], chooser=lambda seq: seq[0])
        out = M.add_operator(DataNode("close"), factor_family="not_registered")
        assert out is not None, "未登记的家族名让 add_operator 崩了"

    def test_the_preferred_list_threshold_is_strict(self):
        """
        `if _rng.random() < 0.75:` —— 恰好 0.75 走**自由探索**那一支。
        两支的候选清单不同（偏好表 vs 全量表），挑出来的 variant 可区分。
        """
        prefs = M._FAMILY_OPERATOR_PREFS["reversion"]      # ["unary_sign","unary_abs"]
        full = ["unary_sign", "unary_abs", "signed_power",
                "self_rank", "rank_deviation", "scaled"]
        assert prefs != full, "构造前提被破坏"

        seen = []

        def _c(seq):
            seq = list(seq)
            seen.append(seq)
            return seq[0]

        _script([0.75], chooser=_c)
        M.add_operator(DataNode("close"), factor_family="reversion")
        assert seen and seen[0] == full, (
            f"random()==0.75 时用的候选表是 {seen[0]} —— "
            f"应当是全量探索表，`< 0.75` 被放宽成了 `<= 0.75`")

    def test_below_the_threshold_the_preferred_list_is_used(self):
        seen = []

        def _c(seq):
            seq = list(seq)
            seen.append(seq)
            return seq[0]

        _script([0.74999], chooser=_c)
        M.add_operator(DataNode("close"), factor_family="reversion")
        assert seen and seen[0] == M._FAMILY_OPERATOR_PREFS["reversion"]


# ===========================================================================
# L. 全算子的共同契约
# ===========================================================================

ALL_UNARY = [
    "point_mutation", "hoist_mutation", "param_mutation",
    "wrap_rank", "add_ts_smoothing", "add_condition",
    "add_volume_filter", "replace_subtree", "add_operator",
]


@pytest.mark.parametrize("name", ALL_UNARY)
def test_every_operator_leaves_the_parent_untouched(name):
    """
    模块 docstring 承诺 "Return NEW trees (deep-copy semantics — originals
    untouched)"。就地改父代会让整个种群悄悄共享同一棵树，
    进化过程从此不可复现，而且没有任何报错。
    """
    fn = getattr(M, name)
    for seed in range(12):
        root = _ts_tree()
        before = repr(root)
        _rng.bind_seed(seed)
        fn(root)
        assert repr(root) == before, (
            f"{name} 改动了传进去的父代：{before} → {root!r}")


@pytest.mark.parametrize("name", ALL_UNARY)
def test_every_operator_returns_a_validatable_tree(name):
    """算子产出的树必须过校验 —— 否则坏结构会一路带进回测才炸。"""
    fn = getattr(M, name)
    for seed in range(12):
        _rng.bind_seed(seed)
        out = fn(_ts_tree())
        assert M._try_validate(out), f"{name}（seed={seed}）产出了非法树：{out!r}"


# ===========================================================================
# M. 第二轮补强：rerun_4 之后仍存活的点
# ===========================================================================

class TestTryValidate:

    def test_a_valid_node_reports_true_and_an_invalid_one_false(self):
        """
        `_try_validate` 的两个 return —— `True` 翻 False 会让**每一个**候选
        都被判非法（所有算子退化成恒等变换，进化原地踏步）；
        `False` 翻 True 会让非法结构一路进种群，直到回测才炸。
        """
        good = CrossSectionalNode("rank", DataNode("close"))
        assert M._try_validate(good) is True, (
            "合法的 rank(close) 被判非法 —— `_try_validate` 的 True 分支被翻了")

        class _Broken:
            """校验器一定处理不了的东西。"""
            node_type = None

        assert M._try_validate(_Broken()) is False, (
            "校验不了的节点被判成合法 —— `_try_validate` 的 False 兜底被翻了")


class _ParamsNode:
    """
    模拟带 `params` 字典、但**没有** `.window` 属性的节点形态
    （`param_mutation` 的 elif 分支正是为它写的）。
    """

    def __init__(self, window=None):
        self.op = "ts_mean"
        self.params = {} if window is None else {"window": window}
        self.children = []
        self.node_type = NodeType.TIME_SERIES


class TestParamMutationParamsBranch:

    def test_a_params_node_without_a_window_key_is_left_alone(self):
        """
        `elif hasattr(target, "params") and "window" in target.params:` ——
        `and` 放宽成 `or` 时，只要有 `params` 属性就会去写
        `target.params["window"]`，而读 `old_w = target.params["window"]`
        会先 KeyError。这个 elif 的两个条件缺一不可。
        """
        node = _ParamsNode(window=None)          # 有 params，但没有 window 键
        _script([], uniforms=[0.2])
        out = M.param_mutation(node)             # 不许抛
        assert "window" not in getattr(out, "params", {}), (
            "没有 window 键的节点被写进了 window —— "
            "`hasattr(params) and 'window' in params` 被放宽成了 `or`")

    def test_a_params_node_with_a_window_moves_by_the_sampled_delta(self):
        """
        `target.params["window"] = max(2, min(60, int(old_w * (1 + delta))))`
        —— 与 `.window` 属性那条是**两份独立**的算式，各自都会被变异。
        """
        node = _ParamsNode(window=20)
        _script([], uniforms=[0.2])
        out = M.param_mutation(node)
        assert out.params["window"] == 24, (
            f"delta=+0.2、原窗口 20，调整后应当是 int(20*1.2)=24，"
            f"实际 {out.params['window']} —— params 分支的算符被改了")

    def test_the_params_window_is_clamped(self):
        for w, delta, expect in [(60, 0.2, 60), (2, -0.2, 2)]:
            node = _ParamsNode(window=w)
            _script([], uniforms=[delta])
            assert M.param_mutation(node).params["window"] == expect


class TestCombineDivBranch:

    def test_a_non_scalar_denominator_is_wrapped_in_abs(self):
        """
        `safe_denom = ArithmeticNode("abs", [other]) if not isinstance(other, ScalarNode) else other`
        —— 删掉 `not` 会反过来：**标量**被套 abs（无意义但无害），
        而可能取到 0 的表达式**不**被套 abs —— 除零防护整个失效，
        组合信号里出现 inf/NaN。

        怎么走到 div 分支：`op` 只有在
        `factor_family and other_root is None and factor_family in _COMPLEMENTARY_FAMILIES`
        时才由 `_combine_op_for_families` 决定，而它对
        `frozenset({"momentum", "volatility"})` 恒返回 "div"。
        `_COMPLEMENTARY_FAMILIES["momentum"][0]` 正是 "volatility"。
        所以 `other_root=None` + `factor_family="momentum"` 是确定性走到除法的唯一入口
        （带 other_root 时 op 只可能是 add/sub/mul —— 上一版就是这么永远走不到的）。
        """
        assert M._COMPLEMENTARY_FAMILIES["momentum"][0] == "volatility", (
            "互补家族表变了 —— 本用例走不到 div 分支了")
        assert M._combine_op_for_families("momentum", "volatility") == "div", (
            "momentum+volatility 不再映射到 div —— 本用例的前提不成立")

        _script([], chooser=lambda seq: list(seq)[0])
        out = M.combine_signals(DataNode("close"), other_root=None,
                                factor_family="momentum")
        text = repr(out)
        assert "/" in text, f"没有走到除法分支：{text} —— 用例前提被破坏"
        assert "abs(" in text, (
            f"除法分支的分母没有被 abs 包住：{text} —— "
            f"`not isinstance(other, ScalarNode)` 的 not 被删掉了")

    def test_a_scalar_denominator_is_not_double_wrapped(self):
        """
        对照：分母是 ScalarNode 时**不**套 abs。
        `combine_signals` 自己产不出标量分母，所以直接验证那条表达式的语义 ——
        产品代码里的判定必须与它一致。
        """
        import inspect
        src = inspect.getsource(M.combine_signals)
        assert ('ArithmeticNode("abs", [other]) '
                'if not isinstance(other, ScalarNode) else other') in src, (
            "除法分母的 abs 包裹条件被改了 —— 除零防护可能失效")


class TestSmoothingFallbackCandidates:

    def test_the_fallback_excludes_scalars_and_string_literals(self):
        """
        第二层兜底 `candidates = [n for n in nodes if not isinstance(n, (ScalarNode, StringLiteralNode))]`
        —— 删掉 `not` 会让兜底候选**只剩**标量/字符串字面量，
        `TimeSeriesNode(op, ScalarNode(...), w)` 造出来的东西过不了校验，
        算子静默失效。
        """
        # 造一棵首选候选为空的树：根是算术节点，叶子全是标量
        root = ArithmeticNode("add", [ScalarNode(1.0), ScalarNode(2.0)])
        seen: list = []

        def _c(seq):
            seq = list(seq)
            seen.append(seq)
            return seq[0]

        _script([], chooser=_c)
        M.add_ts_smoothing(root)
        assert len(seen) >= 2, f"choice 只被调了 {len(seen)} 次"
        fallback = seen[0]
        assert fallback, "兜底候选是空的"
        for n in fallback:
            assert not isinstance(n, (ScalarNode, StringLiteralNode)), (
                f"兜底候选里混进了标量/字面量：{n!r} —— "
                f"`not isinstance(...)` 的 not 被删掉了")


class TestReplaceSubtreeCandidates:

    def test_the_internal_candidates_exclude_leaves_and_childless_nodes(self):
        """
        `internals = [n for n in nodes if not isinstance(n, 叶子类) and _get_children(n)]`
        —— 删掉 `not` 或把 `and` 放宽成 `or` 都会改变候选集。
        上一版是在**测试里重算一遍同样的过滤**，那是在测我自己写的表达式，
        不是产品。这里改成断言递给 `_rng.choice` 的候选清单。
        """
        root = _ts_tree()                 # rank(ts_mean(close,10))
        seen: list = []

        def _c(seq):
            seq = list(seq)
            seen.append(seq)
            return seq[0]

        _script([0.9], chooser=_c)        # 0.9 ≥ 0.70 → 走通用生成器
        M.replace_subtree(root, factor_family="momentum")
        assert seen, "choice 一次都没被调用"
        cands = seen[0]
        reprs = sorted(repr(n) for n in cands)
        assert reprs == ["rank(ts_mean(close,10))", "ts_mean(close,10)"], (
            f"内部候选是 {reprs}，应当只有两个有子节点的非叶子节点 —— "
            f"叶子（close）或无子节点的节点被算进来了")

    def test_the_family_branch_threshold_is_exactly_strict(self):
        """
        `if factor_family and _rng.random() < 0.70:` —— 恰好 0.70 走**通用**
        生成器。两条分支产出的形态不同：家族生成器对 momentum 必产
        `ts_delta`/`ts_rank`，通用生成器不保证。

        上一版只断言了 `out is not None`，两种取值都满足 —— 等于没测。
        这里改成观察**实际调用了哪个生成器**。
        """
        called: list = []
        real_fam = M._generate_family_compatible_subtree
        real_gen = M._generate_typed_node

        import app.core.gp_engine.mutations as _M
        orig_fam, orig_gen = _M._generate_family_compatible_subtree, _M._generate_typed_node
        try:
            _M._generate_family_compatible_subtree = (
                lambda **kw: (called.append("fam"), real_fam(**kw))[1])
            _M._generate_typed_node = (
                lambda **kw: (called.append("gen"), real_gen(**kw))[1])

            _script([0.70], chooser=lambda seq: list(seq)[0])
            M.replace_subtree(_ts_tree(), factor_family="momentum")
            assert called and called[-1] == "gen", (
                f"random()==0.70 时用了 {called[-1]!r} 生成器（fam=家族 gen=通用）"
                f" —— `< 0.70` 被放宽成了 `<=`")

            called.clear()
            _script([0.69999], chooser=lambda seq: list(seq)[0])
            M.replace_subtree(_ts_tree(), factor_family="momentum")
            assert called and called[-1] == "fam", (
                "random() 略小于 0.70 时没有用家族生成器")
        finally:
            _M._generate_family_compatible_subtree = orig_fam
            _M._generate_typed_node = orig_gen


# ===========================================================================
# N. 第三轮：replace_subtree 的内部候选过滤（rerun_7 最后一个存活）
# ===========================================================================

def test_internal_candidates_must_have_children_not_just_be_non_leaves():
    """
    `internals = [n for n in nodes
                  if not isinstance(n, (DataNode, ScalarNode, StringLiteralNode))
                  and _get_children(n)]`

    末尾的 `and` 放宽成 `or` 时，**没有子节点的非叶子节点**也会被算进候选。
    上一轮的用例用的是 `rank(ts_mean(close,10))`，里面每个非叶子节点都有
    子节点，两种取值给出同一个候选集 —— 分不开。

    能分开的是一棵**含零子节点非叶子**的树：`ArithmeticNode("add", [])`
    （构造器没有元数校验，GP 的程序化拼接确实能产出它，见缺陷 C-2 的讨论）。
      `and` → 它被排除（正确：没有子树可替换）
      `or`  → 它进候选，随后 `_replace_node` 拿它当目标，
              `_tree_depth` 走 `max(空序列)` → ValueError
    """
    childless = ArithmeticNode("add", [])
    assert M._get_children(childless) == [], "用例前提被破坏"
    assert not isinstance(childless, (DataNode, ScalarNode, StringLiteralNode))

    root = ArithmeticNode("add", [
        TimeSeriesNode("ts_mean", DataNode("close"), 10),
        childless,
    ])

    seen: list = []

    def _c(seq):
        seq = list(seq)
        seen.append(seq)
        return seq[0]

    _script([0.9], chooser=_c)          # 0.9 ≥ 0.70 → 走通用生成器
    M.replace_subtree(root, factor_family="momentum")

    assert seen, "choice 一次都没被调用"
    cands = seen[0]
    assert all(M._get_children(n) for n in cands), (
        f"候选里混进了没有子节点的节点："
        f"{[repr(n) for n in cands if not M._get_children(n)]} —— "
        f"`not isinstance(...) and _get_children(n)` 末尾的 and 被放宽成了 or")
    assert childless not in cands, "零子节点的算术节点进了内部候选"
