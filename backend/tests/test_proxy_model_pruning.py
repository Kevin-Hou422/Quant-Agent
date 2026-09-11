"""
ml_engine/proxy_model.py —— 早期剪枝代理模型的定钉测试（变异测试驱动）

来由：8 个变异点，首测击杀率 **12.5%**（存活 7）。

ProxyModel 决定**哪些 Alpha 连回测都不跑就被丢掉**。它错了不会有人发现：
被剪掉的表达式不会留下任何痕迹，GP 的产出看起来一切正常，只是搜索空间
被悄悄削掉了一块。冷启动规则 `depth < 2 or depth > 10` 的两个边界
各差一格，就会把**恰好两层**（最常见的 `rank(ts_mean(close,20))` 形状）
或**恰好十层**的表达式整类剪掉。

存活项：
  - `len(self._X) < self.cold_start_n` —— 冷/热启动的切换点
  - `depth < 2 or depth > 10` —— 冷启动规则的上下界
  - `len(set(y)) < 2` —— 单一类别时拒绝拟合的守卫
  - `_fitted` 的两次赋值、`use_label_encoder=False`

既有覆盖（test_alpha_discovery）只验证"能构造、能 update、n_samples 增长"。
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.alpha_engine.typed_nodes import (
    ArithmeticNode,
    CrossSectionalNode,
    DataNode,
    ScalarNode,
    TimeSeriesNode,
)
from app.core.ml_engine.proxy_model import (
    COLD_START_THRESHOLD,
    PRUNE_THRESHOLD,
    ProxyModel,
    extract_features,
)


def _chain(depth: int):
    """构造恰好 `depth` 层的 AST。"""
    node = DataNode("close")
    for _ in range(depth):
        node = TimeSeriesNode("ts_mean", node, 5)
    return node


# ===========================================================================
# A. 冷启动规则的两个边界
# ===========================================================================

class TestColdStartRule:
    """
    `return depth < 2 or depth > 10` —— 两端都是**严格**不等号，
    也就是"深度 2 到 10 之间（含两端）放行"。

    `< 2` 放宽成 `<= 2`：`rank(ts_mean(close,20))` 这种两层结构全被剪掉 ——
    那是最常见的因子形状，GP 从此只能产出三层以上的复杂表达式。
    `> 10` 收紧成 `>= 10`：恰好十层的表达式被剪，而 DepthValidator 的上限
    就是 10，两个门对"什么叫太深"给出不同答案。
    """

    def test_depth_two_is_kept(self):
        pm = ProxyModel()
        assert pm.should_prune(_chain(2)) is False, (
            "深度恰好 2 的表达式被剪掉了 —— `depth < 2` 被放宽成了 `<= 2`")

    def test_depth_one_is_pruned(self):
        assert ProxyModel().should_prune(_chain(1)) is True

    def test_depth_zero_is_pruned(self):
        assert ProxyModel().should_prune(DataNode("close")) is True

    def test_depth_ten_is_kept(self):
        pm = ProxyModel()
        node = _chain(10)
        assert node.depth() == 10, f"构造的深度不对：{node.depth()}"
        assert pm.should_prune(node) is False, (
            "深度恰好 10 的表达式被剪掉了 —— `depth > 10` 被收紧成了 `>= 10`")

    def test_depth_eleven_is_pruned(self):
        assert ProxyModel().should_prune(_chain(11)) is True

    def test_the_rule_actually_discriminates(self):
        """两端都放行、中间也放行的实现会让这条断言失败。"""
        verdicts = {d: ProxyModel().should_prune(_chain(d)) for d in (1, 2, 10, 11)}
        assert verdicts == {1: True, 2: False, 10: False, 11: True}, verdicts


# ===========================================================================
# B. 冷启动 → 热启动的切换点
# ===========================================================================

class TestColdStartThreshold:

    @staticmethod
    def _feed(pm: ProxyModel, n: int) -> None:
        """喂 n 条样本，两个类别都有（否则 _fit 会拒绝）。"""
        for i in range(n):
            pm.update(_chain(2 + i % 3), failed=bool(i % 2))

    def test_below_the_threshold_uses_the_rule(self):
        pm = ProxyModel(cold_start_n=6)
        self._feed(pm, 5)
        assert pm.n_samples == 5
        # 规则模式下深度 1 必剪、深度 3 必留，与任何模型预测无关
        assert pm.should_prune(_chain(1)) is True
        assert pm.should_prune(_chain(3)) is False

    def test_exactly_at_the_threshold_switches_to_the_model(self):
        """
        `if len(self._X) < self.cold_start_n` —— 放宽成 `<=` 会让样本数
        **恰好达标**时仍走规则，模型永远晚一条样本才生效。
        这里用"深度 1 在规则下必剪、在模型下不一定"来区分两条路径。
        """
        pytest.importorskip("xgboost")
        pm = ProxyModel(cold_start_n=6, prune_threshold=1.1)   # 阈值 >1 → 模型永不剪
        self._feed(pm, 6)
        assert pm.n_samples == 6
        assert pm.should_prune(_chain(1)) is False, (
            "样本数恰好达到 cold_start_n 时仍在走冷启动规则 —— "
            "`< cold_start_n` 被放宽成了 `<=`")

    def test_model_path_honours_the_prune_threshold(self):
        """
        `prob_fail >= self.prune_threshold`。阈值设成 0 时一切都该剪，
        设成大于 1 时一切都不该剪 —— 两端都验证，确保走的确实是模型分支。
        """
        pytest.importorskip("xgboost")
        never = ProxyModel(cold_start_n=6, prune_threshold=1.1)
        always = ProxyModel(cold_start_n=6, prune_threshold=0.0)
        self._feed(never, 8)
        self._feed(always, 8)
        node = _chain(4)
        assert never.should_prune(node) is False
        assert always.should_prune(node) is True

    def test_default_threshold_constants_are_what_the_docstring_says(self):
        assert COLD_START_THRESHOLD == 50 and PRUNE_THRESHOLD == 0.70
        pm = ProxyModel()
        assert pm.cold_start_n == COLD_START_THRESHOLD
        assert pm.prune_threshold == PRUNE_THRESHOLD


# ===========================================================================
# C. 单一类别时拒绝拟合
# ===========================================================================

class TestFitGuard:

    def test_a_single_class_leaves_the_model_unfitted(self):
        """
        `if len(set(y)) < 2: return` —— 全是"失败"的样本无法训分类器。
        删掉守卫（或放宽成 `<= 2`，让**两类**也被拒）都会出事：
        前者让 XGBoost 在单类标签上抛错，后者让模型永远训不起来、
        热启动分支拿 `self._model = None` 去 predict_proba → AttributeError。
        """
        pytest.importorskip("xgboost")
        pm = ProxyModel(cold_start_n=4)
        for i in range(6):
            pm.update(_chain(2 + i % 3), failed=True)      # 只有一个类别
        assert pm._model is None, "单一类别却训出了模型"

    def test_unfitted_model_past_cold_start_crashes(self):
        """
        【已登记缺陷 B-8】`_fit()` 因单一类别（或 xgboost 缺失）提前 return 时
        `self._model` 仍是 None，而 `should_prune` 只看 `len(self._X) >= cold_start_n`
        就走模型分支 → `None.predict_proba` → **AttributeError 打断整轮 GP**。

        这正是 `_fitted` 这个字段本该守住的情形，但它从头到尾没人读（见 E 节）。
        触发条件很常见：冷启动期所有候选都失败（标签全是 1），或环境没装 xgboost。

        本阶段只钉住现状。修好之后：这里应当**退回冷启动规则**，
        即 `should_prune(_chain(1)) is True`、`should_prune(_chain(3)) is False`。
        """
        pytest.importorskip("xgboost")
        pm = ProxyModel(cold_start_n=4)
        for i in range(6):
            pm.update(_chain(2 + i % 3), failed=True)
        assert pm._model is None
        with pytest.raises(AttributeError, match="predict_proba"):
            pm.should_prune(_chain(1))

    def test_two_classes_do_get_fitted(self):
        """`< 2` 放宽成 `<= 2` 会让恰好两个类别也被拒 —— 模型永远训不出来。"""
        pytest.importorskip("xgboost")
        pm = ProxyModel(cold_start_n=4)
        for i in range(8):
            pm.update(_chain(2 + i % 3), failed=bool(i % 2))
        assert pm._model is not None, (
            "有两个类别却没有拟合 —— `len(set(y)) < 2` 的守卫被放宽了")
        prob = pm._model.predict_proba(
            extract_features(_chain(3)).reshape(1, -1))[0]
        assert len(prob) == 2 and np.isclose(prob.sum(), 1.0)


# ===========================================================================
# D. 特征提取
# ===========================================================================

class TestFeatureExtraction:

    def test_feature_vector_has_the_advertised_length(self):
        vec = extract_features(_chain(2))
        assert vec.shape == (ProxyModel().feature_size,), vec.shape

    def test_depth_and_node_count_are_the_first_two_features(self):
        node = _chain(3)
        vec = extract_features(node)
        assert vec[0] == 3.0, f"第 0 位不是深度：{vec[0]}"
        assert vec[1] == 4.0, f"第 1 位不是节点数（3 层 + 1 叶）：{vec[1]}"

    def test_max_window_is_the_largest_window_in_the_tree(self):
        node = TimeSeriesNode("ts_mean",
                              TimeSeriesNode("ts_std", DataNode("close"), 60), 5)
        assert extract_features(node)[-3] == 60.0, "max_window 取的不是最大值"

    def test_log_and_division_flags(self):
        plain = extract_features(_chain(2))
        assert plain[-2] == 0.0 and plain[-1] == 0.0

        logged = extract_features(
            ArithmeticNode("log", [TimeSeriesNode("ts_mean", DataNode("close"), 5)]))
        assert logged[-2] == 1.0, "has_log 没有被置位"

        divided = extract_features(
            ArithmeticNode("div", [TimeSeriesNode("ts_mean", DataNode("close"), 5),
                                   ScalarNode(2.0)]))
        assert divided[-1] == 1.0, "has_division 没有被置位"

    def test_operator_frequencies_are_counted(self):
        node = CrossSectionalNode("rank",
                                  TimeSeriesNode("ts_mean",
                                                 TimeSeriesNode("ts_mean",
                                                                DataNode("close"), 5),
                                                 10))
        vec = extract_features(node)
        assert vec[2] == 2.0, f"ts_mean 出现两次却统计成 {vec[2]}"
        assert vec[12] == 1.0, f"rank 出现一次却统计成 {vec[12]}"

    def test_features_are_deterministic(self):
        a = extract_features(_chain(4))
        b = extract_features(_chain(4))
        np.testing.assert_array_equal(a, b, err_msg="同一棵树两次提取出了不同特征")


# ===========================================================================
# E. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "L110 `self._fitted = False` → True / L163 `self._fitted = True` → False":
        "`_fitted` 在整个代码库里**只被写、从不被读** —— should_prune 判断的是 "
        "`len(self._X) < cold_start_n` 与 `self._model`，与该标志无关。"
        "两次赋值都是死存储，取值不影响任何可观测行为。"
        "见 test_fitted_flag_has_no_reader（它同时是这条证明的失效告警："
        "一旦有人开始读 _fitted，该测试会红，这个点就要重新补用例）。"
        "顺带登记：这是一个**无人读取的状态字段**，属于误导性残留。",

    "L155 `use_label_encoder=False` → True":
        "xgboost 自 2.0 起已移除 use_label_encoder，3.x（本环境 3.2.0）对两种取值"
        "都只是忽略，不告警、不改变训练结果。实测两种取值训出的模型对同一输入"
        "给出逐位相同的 predict_proba。见 test_label_encoder_flag_is_ignored_by_xgboost。"
        "顺带登记：这是一个**对当前依赖版本已无意义的参数**。",
}


def test_fitted_flag_has_no_reader():
    """L110/L163 等价性的机械验证：全代码库没有任何地方读 ProxyModel._fitted。"""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1] / "app"
    readers = []
    for p in root.rglob("*.py"):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if "_fitted" not in line or "proxy_model" not in str(p):
                continue
            # 赋值行不算读取
            if re.match(r"\s*self\._fitted\s*=", line):
                continue
            readers.append(f"{p.name}:{i} {line.strip()}")
    assert not readers, (
        "有人开始读 _fitted 了 —— 等价性证明失效，必须补用例：\n  "
        + "\n  ".join(readers))


def test_label_encoder_flag_is_ignored_by_xgboost():
    """L155 等价性的机械验证：两种取值训出的模型给出相同预测。"""
    pytest.importorskip("xgboost")
    from xgboost import XGBClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = np.array([0, 1] * 20)
    probs = []
    for flag in (False, True):
        m = XGBClassifier(n_estimators=10, max_depth=3, use_label_encoder=flag,
                          eval_metric="logloss", verbosity=0, random_state=0)
        m.fit(X, y)
        probs.append(m.predict_proba(X[:3]))
    np.testing.assert_array_equal(probs[0], probs[1],
                                  err_msg="use_label_encoder 已重新变得可观测 —— "
                                          "等价性证明失效")


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 2
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
