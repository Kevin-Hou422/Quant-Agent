"""
gp_engine.py — AlphaEvolver：DEAP GP + 多进程种群评估。

使用现有 alpha_engine.generator 生成初始种群，
用 mutations 模块实现交叉与变异，
DEAP 负责选择与进化循环调度。
"""

from __future__ import annotations

import copy
import logging
import multiprocessing
import random
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..alpha_engine.parser import Parser as _Parser
from ..alpha_engine.typed_nodes import Node
from ..alpha_engine.validator import AlphaValidator
from ..alpha_engine.dsl_executor import Executor as DSLExecutor
from .mutations import point_mutation, hoist_mutation, param_mutation, subtree_crossover
from ..ml_engine.alpha_store import AlphaResult, AlphaStore
from ..ml_engine.proxy_model import ProxyModel

logger = logging.getLogger(__name__)

_validator = AlphaValidator()
_executor  = DSLExecutor()
_parser_inst = _Parser()

# 预定义简单合法的 DSL 模板用于初始种群
_SEED_DSLS = [
    "rank(ts_mean(close,5))",
    "rank(ts_mean(close,10))",
    "rank(ts_mean(close,20))",
    "zscore(ts_delta(close,5))",
    "zscore(ts_delta(close,10))",
    "rank(ts_std(close,10))",
    "rank(ts_std(close,20))",
    "zscore(ts_mean(volume,10))",
    "zscore(ts_mean(volume,20))",
    "rank(ts_delta(close,1))",
    "zscore(ts_delta(close,20))",
    "rank(ts_rank(close,10))",
    "rank(ts_rank(volume,20))",
    "zscore(ts_std(close,5))",
    "rank(ts_mean(vwap,10))",
    "zscore(ts_delta(log(close),5))",
    "rank(ts_delta(log(close),10))",
    "zscore(ts_mean(returns,5))",
    "rank(ts_std(returns,10))",
    "zscore(ts_decay_linear(close,10))",
]


def generate_random_alpha(depth: int = 4) -> Node:
    """生成一个随机合法的 typed_nodes.Node（通过解析预定义 DSL 模板）。"""
    dsl = random.choice(_SEED_DSLS)
    return _parser_inst.parse(dsl)


# ---------------------------------------------------------------------------
# AlphaResult (GP 版，轻量，含 fitness)
# ---------------------------------------------------------------------------

@dataclass
class GPAlphaResult:
    dsl:        str
    fitness:    float   = 0.0
    sharpe:     float   = 0.0
    ic_ir:      float   = 0.0
    ann_return: float   = 0.0
    ann_turnover: float = 0.0


# ---------------------------------------------------------------------------
# Fitness 评估函数（顶层，可被 multiprocessing.Pool 序列化）
# ---------------------------------------------------------------------------

def _evaluate_individual(
    args: Tuple[str, Dict[str, np.ndarray]],
) -> GPAlphaResult:
    """
    对一条 DSL 字符串计算 fitness。
    使用 IC（信号与下期收益的截面 Spearman 相关）作为代理 Sharpe。
    不依赖 BacktestEngine（纯信号质量快速评估）。
    """
    dsl, dataset = args
    try:
        # dataset values 是 numpy arrays（为可序列化）
        # 构建带 DatetimeIndex 的 DataFrame 供 Executor 使用
        T = next(iter(dataset.values())).shape[0]
        dates = pd.bdate_range("2020-01-02", periods=T)
        df_dataset = {
            k: pd.DataFrame(v, index=dates)
            for k, v in dataset.items()
        }
        signal = _executor.run_expr(dsl, df_dataset)
        close_arr = dataset.get("close")
        if close_arr is None or signal is None:
            return GPAlphaResult(dsl=dsl, fitness=-1.0)

        close = close_arr

        # 前向收益
        fwd_ret = (close[1:] - close[:-1]) / np.where(close[:-1] == 0, np.nan, close[:-1])
        sig_arr = signal.to_numpy() if hasattr(signal, "to_numpy") else np.array(signal)

        # 截面 Rank IC
        T = min(fwd_ret.shape[0], sig_arr.shape[0] - 1)
        ics = []
        for t in range(T):
            s = sig_arr[t]
            r = fwd_ret[t]
            mask = ~(np.isnan(s) | np.isnan(r))
            if mask.sum() < 5:
                continue
            from scipy.stats import spearmanr
            rho, _ = spearmanr(s[mask], r[mask])
            if not np.isnan(rho):
                ics.append(rho)

        if not ics:
            return GPAlphaResult(dsl=dsl, fitness=-1.0)

        ic_arr = np.array(ics)
        ic_ir  = float(np.mean(ic_arr) / (np.std(ic_arr) + 1e-9))
        sharpe = ic_ir  # 用 IC-IR 作为 fitness 代理

        return GPAlphaResult(dsl=dsl, fitness=sharpe, sharpe=sharpe, ic_ir=ic_ir)
    except Exception as e:
        logger.debug("Eval failed for '%s': %s", dsl[:80], e)
        return GPAlphaResult(dsl=dsl, fitness=-1.0)


# ---------------------------------------------------------------------------
# AlphaEvolver
# ---------------------------------------------------------------------------

class AlphaEvolver:
    """
    DEAP 风格的 GP 进化引擎。

    Parameters
    ----------
    pop_size      : 种群大小
    n_gen         : 进化代数
    n_workers     : 并行进程数（1 = 单进程）
    tree_depth    : 初始树最大深度
    tourn_size    : 锦标赛选择的参赛者数
    hof_size      : Hall of Fame 保留数
    corr_penalty  : 与 HoF 中已有 Alpha 的相关系数阈值（>= 时惩罚 fitness）
    proxy         : 可选的 ProxyModel（None 则不剪枝）
    """

    def __init__(
        self,
        pop_size:    int   = 50,
        n_gen:       int   = 20,
        n_workers:   int   = 1,
        tree_depth:  int   = 4,
        tourn_size:  int   = 3,
        hof_size:    int   = 10,
        corr_penalty: float = 0.8,
        proxy:       Optional[ProxyModel] = None,
    ) -> None:
        self.pop_size     = pop_size
        self.n_gen        = n_gen
        self.n_workers    = n_workers
        self.tree_depth   = tree_depth
        self.tourn_size   = tourn_size
        self.hof_size     = hof_size
        self.corr_penalty = corr_penalty
        self.proxy        = proxy or ProxyModel()

    # ------------------------------------------------------------------

    def evolve(
        self,
        dataset: Dict[str, pd.DataFrame],
    ) -> List[GPAlphaResult]:
        """
        执行进化循环，返回 Hall of Fame 中的 Alpha 列表。

        Parameters
        ----------
        dataset : 字段 → (T, N) pd.DataFrame 的字典
        """
        # 转为 numpy dict（可序列化）
        np_dataset = {k: v.to_numpy(dtype=float) for k, v in dataset.items()}

        # 1. 初始化种群
        population = self._init_population()
        logger.info("GP 初始种群: %d 个体", len(population))

        hof: List[GPAlphaResult] = []   # Hall of Fame

        for gen in range(self.n_gen):
            # 2. 过滤（代理模型剪枝）
            filtered = [ind for ind in population
                        if not self.proxy.should_prune(ind)]
            if not filtered:
                filtered = population[:max(1, len(population) // 2)]

            # 3. 评估
            results = self._batch_evaluate(filtered, np_dataset)

            # 4. 多样性惩罚
            results = self._apply_diversity_penalty(results, hof)

            # 5. 更新 HoF
            hof = self._update_hof(hof, results)

            # 6. 选择 + 变异/交叉 → 下一代
            population = self._next_generation(filtered or population, results)

            # 7. 更新代理模型
            for r in results:
                try:
                    node = _parse_dsl(r.dsl)
                    failed = r.sharpe < 0.5
                    self.proxy.update(node, failed)
                except Exception:
                    pass

            best = max(results, key=lambda x: x.fitness) if results else None
            logger.info(
                "Gen %d/%d | pop=%d | best_IC-IR=%.4f",
                gen + 1, self.n_gen, len(population),
                best.fitness if best else 0.0,
            )

        return hof

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _init_population(self) -> List[Node]:
        """生成初始种群（随机树）。"""
        pop = []
        seen = set()
        attempts = 0
        while len(pop) < self.pop_size and attempts < self.pop_size * 10:
            attempts += 1
            node = generate_random_alpha(depth=self.tree_depth)
            key  = repr(node)
            if key in seen:
                continue
            try:
                _validator.validate(node)
                pop.append(node)
                seen.add(key)
            except Exception:
                pass
        return pop

    def _batch_evaluate(
        self,
        population: List[Node],
        np_dataset: Dict[str, np.ndarray],
    ) -> List[GPAlphaResult]:
        args = [(repr(ind), np_dataset) for ind in population]
        if self.n_workers > 1:
            with multiprocessing.Pool(self.n_workers) as pool:
                results = pool.map(_evaluate_individual, args)
        else:
            results = [_evaluate_individual(a) for a in args]
        return results

    def _apply_diversity_penalty(
        self,
        results: List[GPAlphaResult],
        hof: List[GPAlphaResult],
    ) -> List[GPAlphaResult]:
        """与 HoF 相关性 >= corr_penalty 的个体 fitness × 0.5。"""
        if not hof:
            return results
        hof_dsls = {r.dsl for r in hof}
        penalized = []
        for r in results:
            if r.dsl in hof_dsls:
                penalized.append(GPAlphaResult(
                    dsl=r.dsl, fitness=r.fitness * 0.5,
                    sharpe=r.sharpe, ic_ir=r.ic_ir,
                    ann_return=r.ann_return, ann_turnover=r.ann_turnover,
                ))
            else:
                penalized.append(r)
        return penalized

    def _update_hof(
        self,
        hof: List[GPAlphaResult],
        results: List[GPAlphaResult],
    ) -> List[GPAlphaResult]:
        combined = {r.dsl: r for r in hof + results}
        sorted_all = sorted(combined.values(), key=lambda x: x.fitness, reverse=True)
        return sorted_all[: self.hof_size]

    def _next_generation(
        self,
        population: List[Node],
        results: List[GPAlphaResult],
    ) -> List[Node]:
        """锦标赛选择 + 变异/交叉 → 下一代。"""
        if not population:
            return [generate_random_alpha(depth=self.tree_depth) for _ in range(self.pop_size)]
        dsl_to_node = {repr(ind): ind for ind in population}
        sorted_res  = sorted(results, key=lambda x: x.fitness, reverse=True)

        def tournament() -> Node:
            if not sorted_res:
                return random.choice(population)
            contestants = random.sample(sorted_res, min(self.tourn_size, len(sorted_res)))
            winner      = max(contestants, key=lambda x: x.fitness)
            return dsl_to_node.get(winner.dsl, random.choice(population))

        next_gen = []
        seen     = set()

        while len(next_gen) < self.pop_size:
            op = random.random()
            if op < 0.4 and len(population) >= 2:
                # 交叉
                p1, p2 = tournament(), tournament()
                c1, c2 = subtree_crossover(p1, p2)
                for child in (c1, c2):
                    k = repr(child)
                    if k not in seen:
                        try:
                            _validator.validate(child)
                            next_gen.append(child)
                            seen.add(k)
                        except Exception:
                            pass
            elif op < 0.7:
                # 点变异
                parent   = tournament()
                mutated  = point_mutation(parent)
                k = repr(mutated)
                if k not in seen:
                    try:
                        _validator.validate(mutated)
                        next_gen.append(mutated)
                        seen.add(k)
                    except Exception:
                        pass
            elif op < 0.85:
                # Hoist 变异
                parent  = tournament()
                mutated = hoist_mutation(parent)
                k = repr(mutated)
                if k not in seen:
                    try:
                        _validator.validate(mutated)
                        next_gen.append(mutated)
                        seen.add(k)
                    except Exception:
                        pass
            else:
                # 参数变异
                parent  = tournament()
                mutated = param_mutation(parent)
                k = repr(mutated)
                if k not in seen:
                    try:
                        _validator.validate(mutated)
                        next_gen.append(mutated)
                        seen.add(k)
                    except Exception:
                        pass

        return next_gen[: self.pop_size]


# ---------------------------------------------------------------------------
# 辅助：DSL 字符串 → Node
# ---------------------------------------------------------------------------

def _parse_dsl(dsl: str) -> Node:
    from ..alpha_engine.parser import Parser
    return Parser().parse(dsl)
