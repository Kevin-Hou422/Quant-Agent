"""
alpha_optimizer.py — Optuna 驱动的超参数优化引擎

严格遵循 Train → Optimize → Lock → Test 工作流：
  - AlphaOptimizer 在整个优化阶段仅访问 IS（训练集）数据
  - OOS 数据由调用方在 optimize() 返回后才解锁传入 RealisticBacktester
  - _objective 函数内部不接受、不存储任何 OOS 引用

适应度函数（与 GP fitness 统一，弊端 5 修复）：
  采用 IS 内部 Walk-Forward Hold-Out：
    IS_early (前 60%) = Optuna 优化集
    IS_late  (后 40%) = Optuna 内部伪 OOS 评估集

  Fitness = sharpe_late
          - 0.2 × turnover_full
          - 0.3 × |max_drawdown_full|
          - 0.5 × max(0, sharpe_full - sharpe_late)

  与 GP 适应度公式（fitness.compute_fitness）完全对齐：
    sharpe_full  → 对应 GP 的 sharpe_is
    sharpe_late  → 对应 GP 的 sharpe_oos（IS 内部伪 OOS）

  当 IS_late 不足 60 个交易日时，降级到简化公式：
    Fitness = sharpe_full - 0.2 × turnover - 0.3 × |max_drawdown|
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

    #: Minimum number of trading days required in IS_late for walk-forward.
    _MIN_LATE_DAYS = 60

    def __init__(
        self,
        dsl:            str,
        is_dataset:     Dict[str, pd.DataFrame],
        search_space:   SearchSpace = None,
        n_trials:       int   = 30,
        seed:           int   = 42,
        timeout:        Optional[float] = None,
        is_late_ratio:  float = 0.40,
    ) -> None:
        self._dsl         = dsl
        # Deep-copy: external mutations don't affect optimisation
        self._is_dataset  = {k: v.copy() for k, v in is_dataset.items()}
        self._space       = search_space or SearchSpace()
        self._n_trials    = n_trials
        self._seed        = seed
        self._timeout     = timeout

        # Walk-forward hold-out split within IS ─────────────────────────────
        # IS_early: first (1 - is_late_ratio) fraction → Optuna "train"
        # IS_late:  last  is_late_ratio fraction        → Optuna "pseudo-OOS"
        total_days = len(next(iter(self._is_dataset.values())))
        late_start = int(total_days * (1.0 - is_late_ratio))
        late_days  = total_days - late_start

        if late_days >= self._MIN_LATE_DAYS:
            self._is_late = {
                k: v.iloc[late_start:].copy()
                for k, v in self._is_dataset.items()
            }
            self._use_walkforward = True
        else:
            self._is_late         = {}
            self._use_walkforward = False
            logger.warning(
                "IS hold-out only %d days (< %d) — disabling walk-forward; "
                "falling back to simplified IS-only objective.",
                late_days, self._MIN_LATE_DAYS,
            )

    # ------------------------------------------------------------------
    # 目标函数（仅 IS）
    # ------------------------------------------------------------------

    def _objective(self, trial) -> float:
        """
        Optuna 目标函数 — 与 GP fitness 完全对齐（弊端 5 修复）。

        严格约束：
          - 仅使用 self._is_dataset 和 self._is_late（IS 分区）
          - 不接受、不访问任何 OOS 数据引用
          - 回测失败时返回 -999.0

        适应度计算流程（walk-forward 模式）：
          Step 1. 全 IS 回测  → sharpe_full, turnover, max_drawdown
          Step 2. IS_late 回测 → sharpe_late（伪 OOS）
          Step 3. compute_fitness(sharpe_full, sharpe_late, turnover, max_dd)
                  == sharpe_late - 0.2×turnover - 0.3×|max_dd| - 0.5×overfit_proxy

        降级模式（IS_late 不足）：
          compute_fitness(sharpe_full, sharpe_full, turnover, max_dd)
          去除过拟合惩罚，等价于 sharpe_full - 0.2×turnover - 0.3×|max_dd|
        """
        from app.core.alpha_engine.signal_processor import SimulationConfig
        from app.core.backtest_engine.realistic_backtester import RealisticBacktester
        from app.core.gp_engine.fitness import compute_fitness

        # 1. 采样超参数
        delay        = trial.suggest_int("delay",    *self._space.delay_range)
        decay_window = trial.suggest_int("decay",    *self._space.decay_range)
        trunc_min    = trial.suggest_float("trunc_min", *self._space.trunc_min_range)
        trunc_max    = trial.suggest_float("trunc_max", *self._space.trunc_max_range)
        port_mode    = trial.suggest_categorical("portfolio_mode",
                                                 list(self._space.portfolio_modes))

        if trunc_min >= trunc_max:
            return -999.0

        neutralize_groups = None
        if self._space.allow_neutralize and self._space.neutralize_groups is not None:
            use_neutral = trial.suggest_categorical("neutralize", [True, False])
            if use_neutral:
                neutralize_groups = self._space.neutralize_groups

        cfg = SimulationConfig(
            delay             = delay,
            decay_window      = decay_window,
            truncation_min_q  = trunc_min,
            truncation_max_q  = trunc_max,
            neutralize_groups = neutralize_groups,
            portfolio_mode    = port_mode,
        )

        # 2. 全 IS 回测（获取 turnover / drawdown / sharpe_full）
        try:
            bt_full  = RealisticBacktester(config=cfg)
            res_full = bt_full.run(self._dsl, self._is_dataset)
        except Exception as exc:
            logger.debug("Trial %d full-IS backtest failed: %s", trial.number, exc)
            return -999.0

        full_r    = res_full.is_report
        sharpe_f  = float(full_r.sharpe_ratio) if not _isnan(full_r.sharpe_ratio) else -5.0
        turnover  = float(full_r.ann_turnover)  if not _isnan(full_r.ann_turnover) else 5.0
        max_dd    = float(full_r.max_drawdown)  if not _isnan(full_r.max_drawdown) else 0.0

        # 3. Walk-forward: IS_late 回测（伪 OOS，获取 sharpe_late）
        if self._use_walkforward and self._is_late:
            try:
                bt_late  = RealisticBacktester(config=cfg)
                res_late = bt_late.run(self._dsl, self._is_late)
                late_r   = res_late.is_report
                sharpe_l = float(late_r.sharpe_ratio) if not _isnan(late_r.sharpe_ratio) else -5.0
            except Exception as exc:
                logger.debug("Trial %d IS_late backtest failed: %s", trial.number, exc)
                sharpe_l = sharpe_f  # degraded: no overfit penalty
        else:
            # Degraded mode: no walk-forward penalty
            sharpe_l = sharpe_f

        # 4. GP-aligned composite fitness
        #    compute_fitness(sharpe_is=sharpe_f, sharpe_oos=sharpe_l, ...)
        fitness = compute_fitness(
            sharpe_is    = sharpe_f,
            sharpe_oos   = sharpe_l,
            turnover     = turnover,
            max_drawdown = max_dd,
        )
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
