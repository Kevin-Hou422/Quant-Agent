"""
realistic_backtester.py — 集成信号处理管道的增强型回测器

在现有 BacktestEngine 基础上，接入 SimulationConfig + SignalProcessor，
支持完整的 IS/OOS 双段回测，并输出对比报告。

架构（组合模式，不继承 BacktestEngine）：
    RealisticBacktester
        ├── SignalProcessor      ← 4 步向量化信号管道
        ├── Parser + Executor   ← DSL → raw signal
        ├── PortfolioConstructor ← Long-Short / Decile
        ├── NeutralizationLayer ← 市场中性约束
        ├── BacktestEngine      ← 逐日 PnL 引擎
        └── RiskReport          ← 绩效汇总
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from app.core.alpha_engine.signal_processor import SimulationConfig, SignalProcessor
from app.core.alpha_engine.dsl_executor import Executor
from app.core.alpha_engine.parser import Parser
from app.core.alpha_engine.validator import AlphaValidator
from app.core.backtest_engine.backtest_engine import BacktestEngine, BacktestResult
from app.core.backtest_engine.portfolio_constructor import (
    DecilePortfolio,
    SignalWeightedPortfolio,
    NeutralizationLayer,
)
from app.core.backtest_engine.risk_report import RiskReport
from app.core.backtest_engine.transaction_cost import CostParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RealisticBacktestResult — IS + OOS 双段回测结果
# ---------------------------------------------------------------------------

@dataclass
class RealisticBacktestResult:
    """
    完整的 IS / OOS 双段回测结果容器。

    Attributes
    ----------
    is_report        : In-Sample 绩效报告
    oos_report       : Out-of-Sample 绩效报告（None 当未传入 oos_dataset）
    config           : 使用的 SimulationConfig
    raw_signal       : DSL 执行后、管道处理前的原始信号
    processed_signal : 经 SignalProcessor 4 步管道后的最终信号
    is_result        : BacktestResult（含净值曲线等原始序列）
    oos_result       : BacktestResult（OOS 段，可选）
    """
    is_report:        RiskReport
    oos_report:       Optional[RiskReport]
    config:           SimulationConfig
    raw_signal:       pd.DataFrame             = field(repr=False)
    processed_signal: pd.DataFrame             = field(repr=False)
    is_result:        BacktestResult           = field(repr=False)
    oos_result:       Optional[BacktestResult] = field(default=None, repr=False)

    def summary(self) -> str:
        """打印 IS / OOS 双段对比摘要。"""
        lines = [
            "",
            "╔══════════════════════════════════════════════════════╗",
            "║         RealisticBacktester — IS / OOS 对比          ║",
            "╚══════════════════════════════════════════════════════╝",
            f"  DSL 超参数: delay={self.config.delay}d  "
            f"decay={self.config.decay_window}  "
            f"trunc=[{self.config.truncation_min_q},{self.config.truncation_max_q}]  "
            f"mode={self.config.portfolio_mode}",
            "",
            "── In-Sample ──────────────────────────────────────────",
            self.is_report.summary(),
        ]
        if self.oos_report is not None:
            lines += [
                "",
                "── Out-of-Sample ──────────────────────────────────────",
                self.oos_report.summary(),
                "",
                _degradation_table(self.is_report, self.oos_report),
            ]
        else:
            lines.append("  (未提供 OOS 数据集，跳过 OOS 评估)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d: dict = {
            "config": {
                "delay":            self.config.delay,
                "decay_window":     self.config.decay_window,
                "truncation_min_q": self.config.truncation_min_q,
                "truncation_max_q": self.config.truncation_max_q,
                "portfolio_mode":   self.config.portfolio_mode,
                "top_pct":          self.config.top_pct,
            },
            "is":  self.is_report.to_dict(),
            "oos": self.oos_report.to_dict() if self.oos_report else None,
        }
        return d


def _degradation_table(is_r: RiskReport, oos_r: RiskReport) -> str:
    """生成 IS vs OOS 关键指标退化对比表。"""
    def _pct(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "  N/A  "
        return f"{v * 100:+.2f}%"

    def _f(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "  N/A  "
        return f"{v:+.4f}"

    rows = [
        ("Sharpe",        is_r.sharpe_ratio,       oos_r.sharpe_ratio,       _f),
        ("Ann.Return",    is_r.annualized_return,   oos_r.annualized_return,  _pct),
        ("MaxDrawdown",   is_r.max_drawdown,        oos_r.max_drawdown,       _pct),
        ("IC IR",         is_r.ic_ir,               oos_r.ic_ir,              _f),
        ("Ann.Turnover",  is_r.ann_turnover,        oos_r.ann_turnover,       _pct),
    ]
    header = f"  {'指标':<14} {'IS':>10}  {'OOS':>10}  {'变化':>10}"
    sep    = "  " + "-" * 48
    lines  = ["── IS vs OOS 退化对比 ───────────────────────────────", header, sep]
    for name, iv, ov, fmt in rows:
        delta = (ov - iv) if (iv is not None and ov is not None
                              and not (np.isnan(iv) or np.isnan(ov))) else None
        delta_str = fmt(delta) if delta is not None else "  N/A  "
        lines.append(f"  {name:<14} {fmt(iv):>10}  {fmt(ov):>10}  {delta_str:>10}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RealisticBacktester
# ---------------------------------------------------------------------------

class RealisticBacktester:
    """
    集成 SimulationConfig 的增强型回测器。

    与裸 BacktestEngine 的关键区别：
    - DSL 信号在组合构建前经过 SignalProcessor 4 步管道（截断→衰减→中性化→延迟）
    - 支持 Long-Short 和 Decile 两种组合模式
    - 支持 IS + OOS 双段回测，并对比退化程度

    Parameters
    ----------
    config      : SimulationConfig，策略超参数
    cost_params : CostParams，交易成本参数（默认使用标准参数）
    """

    def __init__(
        self,
        config:      SimulationConfig,
        cost_params: CostParams = None,
    ) -> None:
        self.config      = config
        self.cost_params = cost_params or CostParams()
        self._processor  = SignalProcessor(config)
        self._parser     = Parser()
        self._validator  = AlphaValidator()
        self._executor   = Executor(validate=False)   # 已在此处验证，避免重复

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def run(
        self,
        dsl:         str,
        dataset:     Dict[str, pd.DataFrame],
        *,
        oos_dataset: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> RealisticBacktestResult:
        """
        执行完整回测流程（IS 必须，OOS 可选）。

        Parameters
        ----------
        dsl         : Alpha DSL 表达式字符串
        dataset     : IS 数据集（字段 → T×N DataFrame）
        oos_dataset : OOS 数据集（可选）。None 则跳过 OOS 评估。

        Returns
        -------
        RealisticBacktestResult
        """
        # 1. 解析 + 验证 DSL（仅一次，IS 和 OOS 共用同一棵 AST）
        node = self._parser.parse(dsl)
        self._validator.validate(node)
        logger.info("RealisticBacktester | DSL='%s' | mode=%s | delay=%d",
                    dsl[:80], self.config.portfolio_mode, self.config.delay)

        # 2. IS 回测
        raw_signal, proc_signal, is_result = self._run_one_segment(
            node, dataset, label="IS"
        )

        is_report = RiskReport.from_result(
            is_result, prices=dataset.get("close")
        )

        # 3. OOS 回测（使用相同 DSL 和 config）
        oos_result: Optional[BacktestResult] = None
        oos_report: Optional[RiskReport]     = None

        if oos_dataset is not None and len(next(iter(oos_dataset.values()))) > 5:
            _, _, oos_result = self._run_one_segment(
                node, oos_dataset, label="OOS"
            )
            oos_report = RiskReport.from_result(
                oos_result, prices=oos_dataset.get("close")
            )

        return RealisticBacktestResult(
            is_report        = is_report,
            oos_report       = oos_report,
            config           = self.config,
            raw_signal       = raw_signal,
            processed_signal = proc_signal,
            is_result        = is_result,
            oos_result       = oos_result,
        )

    # ------------------------------------------------------------------
    # 内部：单段回测
    # ------------------------------------------------------------------

    def _run_one_segment(
        self,
        node:    object,    # typed_nodes.Node
        dataset: Dict[str, pd.DataFrame],
        label:   str = "",
    ):
        """
        对单个数据分区执行：信号生成 → 管道处理 → 组合构建 → BacktestEngine。

        Returns
        -------
        (raw_signal, processed_signal, BacktestResult)
        """
        # Step 2: 执行 DSL → 原始信号
        from app.core.alpha_engine.typed_nodes import Node as _Node
        raw_signal = self._executor.run(node, dataset)   # pd.DataFrame (T×N)

        # Step 3: SignalProcessor 4 步管道（截断→衰减→中性化→延迟）
        proc_signal = self._processor.process(raw_signal)

        logger.debug("[%s] 信号处理完成 | shape=%s | NaN率=%.1f%%",
                     label, proc_signal.shape,
                     proc_signal.isna().mean().mean() * 100)

        # Step 4: 构建组合权重
        weights = self._build_weights(proc_signal)

        # Step 5: 可选市场中性
        if self.config.market_neutral:
            weights = NeutralizationLayer.market_neutral(weights)

        # Step 6: BacktestEngine
        prices = dataset.get("close")
        volume = dataset.get("volume")

        if prices is None:
            raise ValueError("dataset 中缺少 'close' 字段，无法进行回测")
        if volume is None:
            # 若无成交量，用均匀虚拟量（防止 ADV 计算出错）
            volume = pd.DataFrame(
                np.ones_like(prices.to_numpy()) * 1e6,
                index=prices.index, columns=prices.columns,
            )
            logger.warning("[%s] dataset 中无 'volume' 字段，使用虚拟成交量", label)

        engine = BacktestEngine(cost_params=self.cost_params)
        result = engine.run(
            weights = weights,
            prices  = prices,
            volume  = volume,
            signal  = proc_signal,   # 用处理后信号作为 IC 基准
        )

        return raw_signal, proc_signal, result

    # ------------------------------------------------------------------
    # 内部：组合构建（Long-Short or Decile）
    # ------------------------------------------------------------------

    def _build_weights(self, signal: pd.DataFrame) -> pd.DataFrame:
        """
        根据 config.portfolio_mode 选择合适的 PortfolioConstructor。

        long_short : SignalWeightedPortfolio（权重正比于截面 z-score）
        decile     : DecilePortfolio（Top top_pct 做多，Bottom top_pct 做空）
        """
        mode = self.config.portfolio_mode
        if mode == "long_short":
            constructor = SignalWeightedPortfolio(clip_z=3.0)
        elif mode == "decile":
            constructor = DecilePortfolio(
                top_pct    = self.config.top_pct,
                bottom_pct = self.config.top_pct,
            )
        else:
            raise ValueError(f"未知 portfolio_mode: '{mode}'，应为 'long_short' 或 'decile'")

        return constructor.construct(signal)
