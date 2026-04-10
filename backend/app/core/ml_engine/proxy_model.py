"""
proxy_model.py — AST 特征向量提取 + XGBoost 早期剪枝代理模型。

冷启动（样本 < COLD_START_THRESHOLD）：
  规则过滤：深度 < 2 或 > 10 → 直接丢弃（返回 True 表示剪枝）

热启动（样本 >= COLD_START_THRESHOLD）：
  XGBClassifier.predict_proba → 失败概率 > PRUNE_THRESHOLD → 剪枝
  每次回测结果追加训练集并增量重训。
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..alpha_engine.typed_nodes import (
    Node, TimeSeriesNode, CrossSectionalNode, ArithmeticNode,
)

COLD_START_THRESHOLD = 50
PRUNE_THRESHOLD      = 0.70


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------

_TS_OPS_ALL = [
    "ts_mean", "ts_std", "ts_max", "ts_min", "ts_rank",
    "ts_decay_linear", "ts_delta", "ts_delay", "ts_corr", "ts_cov",
]
_CS_OPS_ALL = ["rank", "zscore", "scale", "ind_neutralize"]
_FEATURE_SIZE = 2 + len(_TS_OPS_ALL) + len(_CS_OPS_ALL) + 3  # 固定长度


def extract_features(root: Node) -> np.ndarray:
    """
    将 AST 转为固定长度特征向量：
      [tree_depth, node_count,
       ts_op_freq * len(_TS_OPS_ALL),
       cs_op_freq * len(_CS_OPS_ALL),
       max_window, has_log (0/1), has_division (0/1)]
    """
    depth      = root.depth() if callable(root.depth) else root.depth
    node_count = 0
    ts_freq    = {op: 0 for op in _TS_OPS_ALL}
    cs_freq    = {op: 0 for op in _CS_OPS_ALL}
    max_window = 0
    has_log    = 0
    has_div    = 0

    queue = [root]
    while queue:
        n = queue.pop()
        node_count += 1
        op = getattr(n, "op", None)
        if op:
            if op in ts_freq:
                ts_freq[op] += 1
                w = getattr(n, "window", None)
                if w is None:
                    w = getattr(n, "params", {}).get("window", 0) if hasattr(n, "params") else 0
                max_window = max(max_window, int(w or 0))
            elif op in cs_freq:
                cs_freq[op] += 1
            elif op == "log":
                has_log = 1
            elif op == "div":
                has_div = 1
        ch = n.children if isinstance(n.children, list) else n.children()
        queue.extend(ch)

    vec = (
        [float(depth), float(node_count)]
        + [float(ts_freq[op]) for op in _TS_OPS_ALL]
        + [float(cs_freq[op]) for op in _CS_OPS_ALL]
        + [float(max_window), float(has_log), float(has_div)]
    )
    return np.array(vec, dtype=np.float32)


# ---------------------------------------------------------------------------
# ProxyModel
# ---------------------------------------------------------------------------

class ProxyModel:
    """
    早期剪枝代理模型。

    Parameters
    ----------
    prune_threshold   : 失败概率阈值（>= 此值则剪枝）
    cold_start_n      : 切换到 XGBoost 所需的最少样本数
    """

    def __init__(
        self,
        prune_threshold: float = PRUNE_THRESHOLD,
        cold_start_n:    int   = COLD_START_THRESHOLD,
    ) -> None:
        self.prune_threshold = prune_threshold
        self.cold_start_n    = cold_start_n

        self._X: List[np.ndarray] = []
        self._y: List[int]        = []   # 1 = 失败，0 = 成功
        self._model: Optional[Any] = None
        self._fitted = False

    def should_prune(self, node: Node) -> bool:
        """返回 True 表示该 Alpha 应被剪枝（跳过回测）。"""
        depth = node.depth()

        if len(self._X) < self.cold_start_n:
            return depth < 2 or depth > 10

        feat = extract_features(node).reshape(1, -1)
        prob_fail = self._model.predict_proba(feat)[0][1]
        return float(prob_fail) >= self.prune_threshold

    def update(self, node: Node, failed: bool) -> None:
        """
        向训练集追加一条样本并增量重拟合模型。

        Parameters
        ----------
        node   : Alpha AST 节点
        failed : True 表示该 Alpha 回测失败（Sharpe < 0.5）
        """
        feat = extract_features(node)
        self._X.append(feat)
        self._y.append(int(failed))

        if len(self._X) >= self.cold_start_n:
            self._fit()

    def _fit(self) -> None:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            warnings.warn("xgboost not installed; ProxyModel stays in rule-based mode.")
            return

        X = np.stack(self._X)
        y = np.array(self._y)

        if len(set(y)) < 2:
            return

        model = XGBClassifier(
            n_estimators=50,
            max_depth=4,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        self._model   = model
        self._fitted  = True

    @property
    def n_samples(self) -> int:
        return len(self._X)

    @property
    def feature_size(self) -> int:
        return _FEATURE_SIZE
