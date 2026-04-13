"""
alpha_optimizer.py — Optuna 驱动的超参数优化引擎

严格遵循 Train → Optimize → Lock → Test 工作流：
  - AlphaOptimizer 在整个优化阶段仅访问 IS（训练集）数据
  - OOS 数据由调用方在 optimize() 返回后才解锁传入 RealisticBacktester
  - _objective 函数内部不接受、不存储任何 OOS 引用

适应度函数（业界标准）：
  Fitness = Sharpe_IS + 0.5 × mean_IC_IS - 0.1 × AnnTurnover_IS
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from ..backtest_engine.realistic_backtester import SimulationConfig

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SearchSpace — 超参数搜索空间定义
# ---------------------------------------------------------------------------

@dataclass
class SearchSpace:
    """
    Optuna 搜索空间边界。

    Parameters
    ----------
    delay_range       : (min, max) 执行延迟天数（含两端）
    decay_range       : (min, max) ts_decay_linear 窗口
    trunc_min_range   : (min, max) 截断下分位数
    trunc_max_range   : (min, max) 截断上分位数
    portfolio_modes   : 候选组合模式列表
    allow_neutralize  : 是否纳入行业中性化（需要用户预先提供 groups）
    """
    delay_range:      Tuple[int, int]         = (0, 5)
    decay_range:      Tuple[int, int]         = (0, 10)
    trunc_min_range:  Tuple[float, float]     = (0.01, 0.10)
    trunc_max_range:  Tuple[float, float]     = (0.90, 0.99)
    portfolio_modes:  Tuple[str, ...]         = ("long_short", "decile")
    allow_neutralize: bool                    = False
    neutralize_groups: Optional[np.ndarray]  = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# StudySummary — Optuna 研究摘要
# ---------------------------------------------------------------------------

@dataclass
class StudySummary:
    """优化研究的轻量摘要，可直接序列化为 dict。"""
    n_trials:        int
    best_value:      float
    best_params:     Dict[str, Any]
    trial_values:    List[float]        = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "n_trials":     self.n_trials,
            "best_value":   self.best_value,
            "best_params":  self.best_params,
            "trial_values": [v for v in self.trial_values if not np.isnan(v)],
        }


# ---------------------------------------------------------------------------
# AlphaOptimizer
# ---------------------------------------------------------------------------

class AlphaOptimizer:
    """
    基于 Optuna TPE 的 Alpha 超参数优化器。

    CRITICAL — 防泄漏设计：
    ┌─────────────────────────────────────────────────────────┐
    │  optimize() 整个执行期间，OOS 数据不得出现在任何参数中。  │
    │  _objective() 内部仅使用 self._is_dataset（深拷贝副本）。 │
    └─────────────────────────────────────────────────────────┘

    Parameters
    ----------
    dsl        : Alpha DSL 表达式
    is_dataset : IS 训练集（已由 DataPartitioner 物理分割）
    search_space : SearchSpace 实例
    n_trials   : Optuna 试验次数
    seed       : 随机种子（TPE sampler）
    timeout    : 单次 trial 超时秒数（None = 不限制）
    """

    def __init__(
        self,
        dsl:          str,
        is_dataset:   Dict[str, pd.DataFrame],
        search_space: SearchSpace = None,
        n_trials:     int  = 30,
        seed:         int  = 42,
        timeout:      Optional[float] = None,
    ) -> None:
        self._dsl         = dsl
        # 深拷贝确保外部修改不影响优化过程
        self._is_dataset  = {k: v.copy() for k, v in is_dataset.items()}
        self._space       = search_space or SearchSpace()
        self._n_trials    = n_trials
        self._seed        = seed
        self._timeout     = timeout

    # ------------------------------------------------------------------
    # 目标函数（仅 IS）
    # ------------------------------------------------------------------

    def _objective(self, trial) -> float:
        """
        Optuna 目标函数。

        严格约束：
          - 只使用 self._is_dataset（IS 分区）
          - 不接受、不访问任何 OOS 数据引用
          - 回测失败时返回 -999.0，触发 Optuna 剪枝/跳过
        """
        from app.core.alpha_engine.signal_processor import SimulationConfig
        from app.core.backtest_engine.realistic_backtester import RealisticBacktester

        # 1. 采样超参数
        delay        = trial.suggest_int("delay",    *self._space.delay_range)
        decay_window = trial.suggest_int("decay",    *self._space.decay_range)
        trunc_min    = trial.suggest_float("trunc_min", *self._space.trunc_min_range)
        trunc_max    = trial.suggest_float("trunc_max", *self._space.trunc_max_range)
        port_mode    = trial.suggest_categorical("portfolio_mode",
                                                 list(self._space.portfolio_modes))

        # 确保 trunc_min < trunc_max（非法区间直接惩罚）
        if trunc_min >= trunc_max:
            return -999.0

        # 可选：行业中性化
        neutralize_groups = None
        if self._space.allow_neutralize and self._space.neutralize_groups is not None:
            use_neutral = trial.suggest_categorical("neutralize", [True, False])
            if use_neutral:
                neutralize_groups = self._space.neutralize_groups

        # 2. 构造 SimulationConfig
        cfg = SimulationConfig(
            delay             = delay,
            decay_window      = decay_window,
            truncation_min_q  = trunc_min,
            truncation_max_q  = trunc_max,
            neutralize_groups = neutralize_groups,
            portfolio_mode    = port_mode,
        )

        # 3. IS-only 回测（绝不传入 oos_dataset）
        try:
            backtester = RealisticBacktester(config=cfg)
            result = backtester.run(
                dsl      = self._dsl,
                dataset  = self._is_dataset,
                # oos_dataset 故意省略 → None → 不触发 OOS 评估
            )
        except Exception as exc:
            logger.debug("Trial %d 失败: %s", trial.number, exc)
            return -999.0

        is_r = result.is_report

        # 4. 计算适应度（处理 NaN）
        sharpe   = is_r.sharpe_ratio   if not _isnan(is_r.sharpe_ratio)   else -5.0
        ic       = is_r.mean_ic        if not _isnan(is_r.mean_ic)        else 0.0
        turnover = is_r.ann_turnover   if not _isnan(is_r.ann_turnover)   else 5.0

        fitness = sharpe + 0.5 * ic - 0.1 * turnover
        return float(fitness)

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def optimize(self) -> Tuple["SimulationConfig", StudySummary]:
        """
        运行 Optuna 优化，返回最优超参数配置和研究摘要。

        OOS 数据在此方法执行期间完全隔离：
        调用方应仅在 optimize() 返回后，用 best_config 执行最终 IS+OOS 评估。

        Returns
        -------
        best_config : SimulationConfig，已锁定的最优超参数
        summary     : StudySummary，包含 trial 历史
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError as e:
            raise ImportError(
                "需要安装 optuna：pip install optuna>=3.0"
            ) from e

        # 静默 Optuna 日志（仅保留 WARNING 级别）
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = TPESampler(seed=self._seed)
        study   = optuna.create_study(direction="maximize", sampler=sampler)

        logger.info(
            "AlphaOptimizer 启动 | dsl='%s' | n_trials=%d | IS 数据 %d 天",
            self._dsl[:60], self._n_trials,
            len(next(iter(self._is_dataset.values()))),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(
                self._objective,
                n_trials  = self._n_trials,
                timeout   = self._timeout,
                n_jobs    = 1,          # 串行：避免 IS 数据在多线程中被并发修改
                show_progress_bar = False,
            )

        best_params = study.best_params
        trial_vals  = [t.value for t in study.trials if t.value is not None]

        logger.info(
            "优化完成 | best_fitness=%.4f | best_params=%s",
            study.best_value, best_params,
        )

        # 构造最优 SimulationConfig
        best_config = self._params_to_config(best_params)

        summary = StudySummary(
            n_trials     = len(study.trials),
            best_value   = study.best_value,
            best_params  = best_params,
            trial_values = trial_vals,
        )
        return best_config, summary

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _params_to_config(self, params: Dict[str, Any]) -> "SimulationConfig":
        """将 Optuna best_params dict 转换为 SimulationConfig。"""
        from app.core.alpha_engine.signal_processor import SimulationConfig

        neutralize_groups = None
        if params.get("neutralize") and self._space.neutralize_groups is not None:
            neutralize_groups = self._space.neutralize_groups

        return SimulationConfig(
            delay             = params.get("delay", 1),
            decay_window      = params.get("decay", 0),
            truncation_min_q  = params.get("trunc_min", 0.05),
            truncation_max_q  = params.get("trunc_max", 0.95),
            neutralize_groups = neutralize_groups,
            portfolio_mode    = params.get("portfolio_mode", "long_short"),
        )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _isnan(v) -> bool:
    """兼容 None 和 float NaN 的检查。"""
    if v is None:
        return True
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return True
