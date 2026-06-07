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
from typing import Dict, List, Optional, Tuple

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
# F2 + E2 — 输入数据验证
# ---------------------------------------------------------------------------

def _validate_dataset(data: Dict[str, pd.DataFrame], label: str = "dataset") -> None:
    """
    验证数据集结构、时间顺序和字段一致性。

    在 RealisticBacktester.run() 入口调用，防止：
    - 前视偏差（乱序索引）
    - 重复日期导致的计算错误
    - 缺少必要字段（close）
    - 跨字段形状不一致

    Raises
    ------
    ValueError : 任何检验失败时抛出，携带可定位问题的上下文信息。
    """
    if not data:
        raise ValueError(f"[{label}] 数据集为空 dict")

    if "close" not in data:
        present = list(data.keys())
        raise ValueError(
            f"[{label}] 缺少必要字段 'close'（当前字段: {present}）"
        )

    ref = data["close"]

    if not isinstance(ref, pd.DataFrame):
        raise ValueError(
            f"[{label}].close: 期望 pd.DataFrame，实际类型 {type(ref).__name__}"
        )

    if not isinstance(ref.index, pd.DatetimeIndex):
        raise ValueError(
            f"[{label}].close: 索引必须为 DatetimeIndex（当前: {type(ref.index).__name__}）"
        )

    if not ref.index.is_monotonic_increasing:
        bad = ref.index[ref.index != ref.index.sort_values()][:3].tolist()
        raise ValueError(
            f"[{label}].close: 时间索引未升序排列，存在前视偏差风险。"
            f" 乱序位置示例: {bad}。请在传入前调用 df.sort_index()。"
        )

    dups = ref.index[ref.index.duplicated()].unique()
    if len(dups) > 0:
        raise ValueError(
            f"[{label}].close: 发现重复日期 {dups[:5].tolist()}，"
            f" 请先去重（df.loc[~df.index.duplicated()]）。"
        )

    if ref.empty:
        raise ValueError(f"[{label}].close: DataFrame 为空（0 行）")

    # 跨字段形状一致性
    for field_name, df in data.items():
        if field_name == "close":
            continue
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"[{label}].{field_name}: 期望 pd.DataFrame，实际类型 {type(df).__name__}"
            )
        if df.shape != ref.shape:
            raise ValueError(
                f"[{label}].{field_name}: 形状 {df.shape} 与 close {ref.shape} 不一致"
            )
        if not df.index.equals(ref.index):
            raise ValueError(
                f"[{label}].{field_name}: 时间索引与 close 不一致"
            )


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
        # 1. 验证输入数据集结构（F2+E2 修复：前视偏差防护 + 结构校验）
        _validate_dataset(dataset, label="IS")
        if oos_dataset is not None:
            _validate_dataset(oos_dataset, label="OOS")

        # 2. 解析 + 验证 DSL（仅一次，IS 和 OOS 共用同一棵 AST）
        node = self._parser.parse(dsl)
        self._validator.validate(node)
        logger.info("RealisticBacktester | DSL='%s' | mode=%s | delay=%d",
                    dsl[:80], self.config.portfolio_mode, self.config.delay)

        # 3. IS 回测
        raw_signal, proc_signal, is_result = self._run_one_segment(
            node, dataset, label="IS"
        )

        is_report = RiskReport.from_result(
            is_result, prices=dataset.get("close")
        )

        # 4. OOS 回测（使用相同 DSL 和 config）
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
        raw_signal = self._executor.run(node, dataset)   # pd.DataFrame (T×N)

        # Step 3: SignalProcessor 4 步管道（截断→衰减→中性化→延迟）
        proc_signal = self._processor.process(raw_signal)

        # ── Burn-in trim ────────────────────────────────────────────────────
        # Time-series operators (ts_mean, ts_delta, …) produce all-NaN rows
        # for the first `window` days. Without trimming, those rows create
        # zero-weight positions and a flat equity-curve leading section.
        # We locate the first date where at least one asset has a valid
        # signal and restrict the entire backtest window to that date onward.
        valid_rows = proc_signal.notna().any(axis=1)
        if valid_rows.any():
            first_valid = proc_signal.index[valid_rows.to_numpy().argmax()]
            n_trimmed   = int((proc_signal.index < first_valid).sum())
            if n_trimmed > 0:
                logger.info(
                    "[%s] Trimming %d burn-in row(s) — backtest starts %s",
                    label, n_trimmed,
                    first_valid.date() if hasattr(first_valid, "date") else first_valid,
                )
                proc_signal = proc_signal.loc[first_valid:]
                dataset = {
                    k: v.loc[v.index >= first_valid] if isinstance(v, pd.DataFrame) else v
                    for k, v in dataset.items()
                }
        # ────────────────────────────────────────────────────────────────────

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

        F11 修复：当 config.max_single_weight > 0 时，对权重矩阵逐资产施加上限约束
        并重新 L1 归一化，防止极端集中风险。
        """
        import numpy as np

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

        weights = constructor.construct(signal)

        # F11: 单资产权重上限约束
        cap = self.config.max_single_weight
        if cap > 0:
            w = weights.to_numpy(dtype=float)
            # 裁剪绝对权重，方向保留
            clipped = np.sign(w) * np.minimum(np.abs(w), cap)
            # 重新 L1 归一化
            l1 = np.nansum(np.abs(clipped), axis=1, keepdims=True)
            l1 = np.where(l1 == 0, 1.0, l1)
            weights = pd.DataFrame(clipped / l1, index=weights.index, columns=weights.columns)

        return weights


# ---------------------------------------------------------------------------
# WalkForwardResult — 多折汇总（Task 2.1）
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardFoldReport:
    """单折 IS/OOS 回测结果摘要。"""
    fold_idx:     int
    is_start:     str
    is_end:       str
    oos_start:    str
    oos_end:      str
    is_days:      int
    oos_days:     int
    is_sharpe:    float
    oos_sharpe:   float
    oos_maxdd:    float
    oos_turnover: float
    oos_ic_ir:    float
    overfitting:  float   # max(0, IS_sharpe - OOS_sharpe) / |IS_sharpe|

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


@dataclass
class WalkForwardResult:
    """
    Walk-Forward 多折汇总结果。

    核心统计量：
      mean_oos_sharpe  — 各折 OOS Sharpe 均值（核心质量指标）
      std_oos_sharpe   — 各折 OOS Sharpe 标准差（稳健性指标）
      min_oos_sharpe   — 最差折 OOS Sharpe
      pct_positive     — OOS Sharpe > 0 的折数占比
      mean_overfitting — 平均过拟合程度
    """
    n_folds:          int
    fold_reports:     List[WalkForwardFoldReport]
    dsl:              str
    mean_oos_sharpe:  float
    std_oos_sharpe:   float
    min_oos_sharpe:   float
    pct_positive:     float
    mean_overfitting: float
    config:           Optional[Dict] = field(default=None)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  Walk-Forward 回测汇总 ({self.n_folds} 折)",
            "=" * 60,
            f"  DSL: {self.dsl[:70]}",
            "-" * 60,
            f"  OOS Sharpe  均值: {self.mean_oos_sharpe:+.4f}",
            f"  OOS Sharpe  标准差: {self.std_oos_sharpe:.4f}",
            f"  OOS Sharpe  最差: {self.min_oos_sharpe:+.4f}",
            f"  正收益折数比率: {self.pct_positive*100:.1f}%",
            f"  平均过拟合分: {self.mean_overfitting:.4f}",
            "-" * 60,
            "  各折明细:",
        ]
        for r in self.fold_reports:
            lines.append(
                f"  Fold {r.fold_idx+1}  IS={r.is_start[:10]}→{r.is_end[:10]}"
                f"({r.is_days}d)  OOS={r.oos_start[:10]}→{r.oos_end[:10]}"
                f"({r.oos_days}d)  OOS_Sharpe={r.oos_sharpe:+.4f}"
                f"  overfit={r.overfitting:.2f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_folds":          self.n_folds,
            "dsl":              self.dsl,
            "mean_oos_sharpe":  round(self.mean_oos_sharpe,  4),
            "std_oos_sharpe":   round(self.std_oos_sharpe,   4),
            "min_oos_sharpe":   round(self.min_oos_sharpe,   4),
            "pct_positive":     round(self.pct_positive,     4),
            "mean_overfitting": round(self.mean_overfitting, 4),
            "fold_reports":     [r.to_dict() for r in self.fold_reports],
            "config":           self.config,
        }


# ---------------------------------------------------------------------------
# WalkForwardBacktester — 多折滚动验证（Task 2.1）
# ---------------------------------------------------------------------------

class WalkForwardBacktester:
    """
    在多个滚动 IS/OOS 折上运行 RealisticBacktester 并汇总结果。

    用途：
      - 验证 Alpha DSL 在不同市场环境下的稳健性
      - 量化过拟合程度（IS vs OOS Sharpe 的系统性退化）
      - 为 GP 进化提供更可信的 OOS 评估（而非单次固定切分）

    Parameters
    ----------
    config      : SimulationConfig（信号管道参数）
    n_splits    : Walk-Forward 折数（推荐 5）
    embargo_days: IS/OOS 间隔天数（推荐 20）
    min_train_days: IS 最少天数
    cost_params : 交易成本参数
    """

    def __init__(
        self,
        config:         SimulationConfig,
        n_splits:       int   = 5,
        embargo_days:   int   = 20,
        min_train_days: int   = 120,
        cost_params:    Optional[CostParams] = None,
    ) -> None:
        self.config         = config
        self.n_splits       = n_splits
        self.embargo_days   = embargo_days
        self.min_train_days = min_train_days
        self.cost_params    = cost_params or CostParams()

    def run(
        self,
        dsl:     str,
        dataset: Dict[str, pd.DataFrame],
    ) -> WalkForwardResult:
        """
        对 DSL 执行 n_splits 折 Walk-Forward 回测。

        Parameters
        ----------
        dsl     : Alpha DSL 表达式字符串
        dataset : 完整数据集（字段 → T×N DataFrame）

        Returns
        -------
        WalkForwardResult（含折明细 + 汇总统计）
        """
        from app.core.data_engine.data_partitioner import WalkForwardPartitioner

        wf = WalkForwardPartitioner(
            n_splits       = self.n_splits,
            min_train_days = self.min_train_days,
            embargo_days   = self.embargo_days,
        )

        folds    = wf.get_folds(dataset)
        splits   = wf.split(dataset)

        if not splits:
            raise ValueError(
                f"数据集日期范围不足，无法生成 Walk-Forward 分折。"
                f" 请提供至少 {self.min_train_days + self.n_splits * 30} 个交易日的数据。"
            )

        bt       = RealisticBacktester(config=self.config, cost_params=self.cost_params)
        fold_rpts: List[WalkForwardFoldReport] = []

        for fold_meta, (is_data, oos_data) in zip(folds, splits):
            try:
                result  = bt.run(dsl, is_data, oos_dataset=oos_data)
                is_r    = result.is_report
                oos_r   = result.oos_report

                def _f(v) -> float:
                    try:
                        fv = float(v)
                        return fv if not np.isnan(fv) else 0.0
                    except Exception:
                        return 0.0

                is_s  = _f(is_r.sharpe_ratio)
                oos_s = _f(oos_r.sharpe_ratio) if oos_r else 0.0
                overfit = (
                    float(np.clip((is_s - oos_s) / abs(is_s), 0.0, 1.0))
                    if abs(is_s) > 1e-9 else 0.0
                )

                fold_rpts.append(WalkForwardFoldReport(
                    fold_idx     = fold_meta.fold_idx,
                    is_start     = str(fold_meta.is_start.date()),
                    is_end       = str(fold_meta.is_end.date()),
                    oos_start    = str(fold_meta.oos_start.date()),
                    oos_end      = str(fold_meta.oos_end.date()),
                    is_days      = fold_meta.is_days,
                    oos_days     = fold_meta.oos_days,
                    is_sharpe    = is_s,
                    oos_sharpe   = oos_s,
                    oos_maxdd    = _f(oos_r.max_drawdown if oos_r else 0.0),
                    oos_turnover = _f(is_r.ann_turnover),
                    oos_ic_ir    = _f(oos_r.ic_ir if oos_r else 0.0),
                    overfitting  = overfit,
                ))
                logger.info(
                    "WF Fold %d/%d | IS Sharpe=%.3f | OOS Sharpe=%.3f | overfit=%.2f",
                    fold_meta.fold_idx + 1, len(folds), is_s, oos_s, overfit,
                )
            except Exception as exc:
                logger.warning("WF Fold %d 失败: %s", fold_meta.fold_idx + 1, exc)

        if not fold_rpts:
            raise RuntimeError(f"所有 Walk-Forward 折均失败，DSL='{dsl}'")

        oos_sharpes = [r.oos_sharpe   for r in fold_rpts]
        overfits    = [r.overfitting  for r in fold_rpts]

        return WalkForwardResult(
            n_folds          = len(fold_rpts),
            fold_reports     = fold_rpts,
            dsl              = dsl,
            mean_oos_sharpe  = float(np.mean(oos_sharpes)),
            std_oos_sharpe   = float(np.std(oos_sharpes)),
            min_oos_sharpe   = float(np.min(oos_sharpes)),
            pct_positive     = float(np.mean([s > 0 for s in oos_sharpes])),
            mean_overfitting = float(np.mean(overfits)),
        )
