"""
mutations.py — GP 树变异与交叉算子。

四种操作：
  point_mutation     : 替换一个节点的算子（同类型）
  hoist_mutation     : 用子节点替换整棵子树（减小树深度）
  param_mutation     : 随机调整 TS 窗口参数（±20%，clamp [2, 60]）
  subtree_crossover  : 交换两棵树中类型兼容的随机子树

全部操作均保留 NodeType 约束，返回新树（不修改原树）。
"""

from __future__ import annotations

import copy
import random
from typing import List, Optional, Tuple

from ..alpha_engine.typed_nodes import (
    Node, NodeType,
    DataNode, ScalarNode,
    TimeSeriesNode, CrossSectionalNode, ArithmeticNode,
    _TS_OPS, _CS_OPS,
)
from ..alpha_engine.validator import AlphaValidator

_validator = AlphaValidator()

def _node_type(node: Node):
    """兼容两种 Node 实现：typed_nodes.NodeType 或 ast.Node 的 op 分类。"""
    nt = getattr(node, "node_type", None)
    if nt is not None:
        return nt
    op = getattr(node, "op", "data")
    if op in _TS_OPS:
        return "ts"
    if op in _CS_OPS:
        return "cs"
    return "other"


# TS 算子列表（排除双子节点算子）
_TS_LIST = sorted(_TS_OPS - {"ts_corr", "ts_cov"})
_CS_LIST = sorted(_CS_OPS - {"ind_neutralize"})

# TS 窗口候选
_TS_WINDOWS = [3, 5, 10, 20, 40, 60]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _get_children(node: Node) -> list:
    """兼容 ast.Node（children 是字段）和 typed_nodes.Node（children 是方法）。"""
    ch = node.children
    return ch if isinstance(ch, list) else ch()


def _collect_nodes(root: Node) -> List[Node]:
    """BFS 收集树中所有节点（含根）。"""
    result, queue = [], [root]
    while queue:
        n = queue.pop(0)
        result.append(n)
        queue.extend(_get_children(n))
    return result


def _replace_node(root: Node, target: Node, replacement: Node) -> Node:
    """
    深拷贝 root 并将第一个与 target 对象地址相同的节点替换为 replacement。
    若 target 是 root 本身，直接返回 replacement 的深拷贝。
    """
    if root is target:
        return copy.deepcopy(replacement)
    root_copy = copy.deepcopy(root)
    _replace_inplace(root_copy, id(target), copy.deepcopy(replacement))
    return root_copy


def _replace_inplace(node: Node, target_id: int, replacement: Node) -> bool:
    """就地替换（深拷贝后使用）。返回 True 表示已替换。"""
    if isinstance(node, TimeSeriesNode):
        if id(node.child) == target_id:
            node.child = replacement
            return True
        return _replace_inplace(node.child, target_id, replacement)
    if isinstance(node, CrossSectionalNode):
        if id(node.child) == target_id:
            node.child = replacement
            return True
        return _replace_inplace(node.child, target_id, replacement)
    if isinstance(node, ArithmeticNode):
        for i, ch in enumerate(node._children):
            if id(ch) == target_id:
                node._children[i] = replacement
                return True
        for ch in node._children:
            if _replace_inplace(ch, target_id, replacement):
                return True
    # Handle ast.Node (children is a list field)
    children_attr = getattr(node, "children", None)
    if isinstance(children_attr, list):
        for i, ch in enumerate(children_attr):
            if id(ch) == target_id:
                node.children[i] = replacement
                return True
        for ch in children_attr:
            if _replace_inplace(ch, target_id, replacement):
                return True
    return False


# ---------------------------------------------------------------------------
# 1. point_mutation
# ---------------------------------------------------------------------------

def point_mutation(root: Node) -> Node:
    """
    随机选一个 TS 或 CS 节点，替换为同类型的另一个算子。
    不改变树结构和 NodeType。
    """
    nodes = _collect_nodes(root)
    ts_nodes = [n for n in nodes
                if isinstance(n, TimeSeriesNode) or getattr(n, "op", "") in _TS_OPS]
    cs_nodes = [n for n in nodes
                if isinstance(n, CrossSectionalNode) or getattr(n, "op", "") in _CS_OPS]

    candidates: List[Tuple[str, Node]] = (
        [("ts", n) for n in ts_nodes] + [("cs", n) for n in cs_nodes]
    )
    if not candidates:
        return copy.deepcopy(root)

    kind, target = random.choice(candidates)
    new_root = copy.deepcopy(root)
    all_new = _collect_nodes(new_root)

    target_repr = repr(target)
    matched = [n for n in all_new if repr(n) == target_repr]
    if not matched:
        return new_root
    node_to_mutate = matched[0]

    if kind == "ts":
        ts_op = getattr(node_to_mutate, "op", None)
        if ts_op in _TS_OPS:
            ops = [op for op in _TS_LIST if op != ts_op]
            if ops:
                node_to_mutate.op = random.choice(ops)
    elif kind == "cs":
        cs_op = getattr(node_to_mutate, "op", None)
        if cs_op in _CS_OPS:
            ops = [op for op in _CS_LIST if op != cs_op]
            if ops:
                node_to_mutate.op = random.choice(ops)

    return new_root


# ---------------------------------------------------------------------------
# 2. hoist_mutation
# ---------------------------------------------------------------------------

def hoist_mutation(root: Node) -> Node:
    """
    随机选一个非叶节点，用其某个子节点替换整棵子树（减小深度）。
    """
    nodes = _collect_nodes(root)
    has_children = [n for n in nodes if _get_children(n)]
    if not has_children:
        return copy.deepcopy(root)

    target = random.choice(has_children)
    children = _get_children(target)
    if not children:
        return copy.deepcopy(root)

    hoisted = random.choice(children)
    new_root = _replace_node(root, target, hoisted)

    try:
        _validator.validate(new_root)
        return new_root
    except Exception:
        return copy.deepcopy(root)


# ---------------------------------------------------------------------------
# 3. param_mutation
# ---------------------------------------------------------------------------

def param_mutation(root: Node) -> Node:
    """
    随机调整一个 TS 节点的窗口参数（±20%，clamp [2, 60]）。
    """
    new_root = copy.deepcopy(root)
    nodes = _collect_nodes(new_root)
    ts_nodes = [n for n in nodes
                if isinstance(n, TimeSeriesNode) or getattr(n, "op", "") in _TS_OPS]
    if not ts_nodes:
        return new_root

    target = random.choice(ts_nodes)
    if hasattr(target, "window"):
        old_w = target.window
        delta = random.uniform(-0.2, 0.2)
        new_window = max(2, min(60, int(old_w * (1 + delta))))
        target.window = new_window
    elif hasattr(target, "params") and "window" in target.params:
        old_w = target.params["window"]
        delta = random.uniform(-0.2, 0.2)
        new_window = max(2, min(60, int(old_w * (1 + delta))))
        target.params["window"] = new_window
    return new_root


# ---------------------------------------------------------------------------
# 4. subtree_crossover
# ---------------------------------------------------------------------------

def subtree_crossover(root1: Node, root2: Node) -> Tuple[Node, Node]:
    """
    在两棵树中找类型兼容的随机子树进行交换。
    返回两棵新树；若找不到兼容点则返回原树的深拷贝。
    """
    nodes1 = _collect_nodes(root1)
    nodes2 = _collect_nodes(root2)

    def by_type(nodes):
        d = {}
        for n in nodes:
            d.setdefault(_node_type(n), []).append(n)
        return d

    g1, g2 = by_type(nodes1), by_type(nodes2)
    common_types = set(g1.keys()) & set(g2.keys())
    if not common_types:
        return copy.deepcopy(root1), copy.deepcopy(root2)

    chosen_type = random.choice(list(common_types))
    swap1 = random.choice(g1[chosen_type])
    swap2 = random.choice(g2[chosen_type])

    new_root1 = _replace_node(root1, swap1, swap2)
    new_root2 = _replace_node(root2, swap2, swap1)

    try:
        _validator.validate(new_root1)
        _validator.validate(new_root2)
        return new_root1, new_root2
    except Exception:
        return copy.deepcopy(root1), copy.deepcopy(root2)
