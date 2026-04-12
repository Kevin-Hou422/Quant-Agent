"""
optimization_engine — 超参优化、评估与 IS/OOS 分区

集中管理与参数搜索、模型评估、数据分区相关的组件：
  - AlphaOptimizer   : Optuna TPE 超参优化（Phase 2）
  - AlphaEvaluator   : Rolling 指标 + 过拟合评分（Phase 2）
  - DataPartitioner  : IS/OOS 严格物理分区（Phase 1）
"""

from .alpha_optimizer import AlphaOptimizer, SearchSpace, StudySummary
from .alpha_evaluator import AlphaEvaluator, AlphaEvaluatorResult, EvalMetrics
from .data_partitioner import DataPartitioner, PartitionedDataset

__all__ = [
    "AlphaOptimizer", "SearchSpace", "StudySummary",
    "AlphaEvaluator", "AlphaEvaluatorResult", "EvalMetrics",
    "DataPartitioner", "PartitionedDataset",
]
