"""
validation_gate.py — 自动验证门：CANDIDATE → VALIDATED（Phase 9.3）

角色
----
把"回测好看"变成"够格进 paper"的**规则化闸门**。任一候选因子要从 CANDIDATE 升到 VALIDATED，
必须同时满足（roadmap §9.3 / OPERATIONS.md §1）：
  1. **WalkForward 全折 OOS 为正**（≥ n_splits 折，逐折 OOS Sharpe > 0，非均值为正）
  2. **Deflated Sharpe Ratio > 阈值**（默认 0.90；对多重检验/回测过拟合去膨胀）
  3. **真实数据集**（由调用方传入真实面板；本门不接受合成数据下结论——见 evaluate 备注）

设计
----
- 纯计算 + 复用现有引擎，不新造金融逻辑：
  `WalkForwardBacktester`（全折检验）、`BacktestEngine`（取日净收益）、`deflated_sharpe_from_returns`。
- **只判定、不改状态**：返回 `ValidationResult`；状态流转由调用方经生命周期状态机/审批端点执行
  （符合"状态流转不作为 LLM 工具、走人工/规则显式调用"）。
- t≥3.0 门槛与 PBO/CPCV 属 Phase R.1，本门先落 WalkForward + DSR。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WidePanel = Dict[str, pd.DataFrame]


@dataclass
class ValidationResult:
    passed:          bool
    dsl:             str
    reasons:         List[str] = field(default_factory=list)   # 未通过的原因（通过则空）
    # 指标（供审计/前端展示）
    n_folds:         int = 0
    min_oos_sharpe:  float = 0.0
    mean_oos_sharpe: float = 0.0
    pct_folds_positive: float = 0.0
    deflated_sharpe: float = 0.0
    n_trials:        int = 1

    def to_dict(self) -> dict:
        return {
            "passed": self.passed, "dsl": self.dsl, "reasons": self.reasons,
            "n_folds": self.n_folds,
            "min_oos_sharpe": round(self.min_oos_sharpe, 4),
            "mean_oos_sharpe": round(self.mean_oos_sharpe, 4),
            "pct_folds_positive": round(self.pct_folds_positive, 4),
            "deflated_sharpe": round(self.deflated_sharpe, 4),
            "n_trials": self.n_trials,
        }


class ValidationGate:
    """
    CANDIDATE → VALIDATED 自动验证门。

    Parameters
    ----------
    n_splits      : WalkForward 折数（默认 5，roadmap 要求 ≥5）。
    embargo_days  : IS/OOS 间隔离带（默认 20）。
    dsr_threshold : DSR 阈值（默认 0.90）。
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 20,
        dsr_threshold: float = 0.90,
    ) -> None:
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.dsr_threshold = dsr_threshold

    def evaluate(self, dsl: str, dataset: WidePanel, n_trials: int = 1) -> ValidationResult:
        """
        对 `dsl` 在**真实** `dataset` 上运行验证门。

        n_trials : 产生该候选所试过的策略总数（GP: Σ 各代 pop_size）。用于 DSR 去膨胀——
                   越大越严格。默认 1（最宽松）；自主发现调用方应传真实计数。

        任一环节抛错 → 判为**不通过**（fail-closed，不静默放行——见 DEV_LESSONS.md §B）。
        """
        reasons: List[str] = []
        res = ValidationResult(passed=False, dsl=dsl, n_trials=n_trials)

        # ---- 1. WalkForward 全折 OOS 为正 ----
        try:
            wf = self._walk_forward(dsl, dataset)
            res.n_folds = wf.n_folds
            res.min_oos_sharpe = float(wf.min_oos_sharpe)
            res.mean_oos_sharpe = float(wf.mean_oos_sharpe)
            res.pct_folds_positive = float(wf.pct_positive)
            if wf.n_folds < self.n_splits:
                reasons.append(f"WalkForward 折数不足（{wf.n_folds} < {self.n_splits}）")
            if wf.min_oos_sharpe <= 0.0:
                reasons.append(
                    f"存在 OOS Sharpe ≤ 0 的折（最差={wf.min_oos_sharpe:.3f}，"
                    f"正收益折比={wf.pct_positive*100:.0f}%）"
                )
        except Exception as exc:  # fail-closed
            logger.warning("[validation_gate] WalkForward 失败 → 判不通过: %s", exc)
            reasons.append(f"WalkForward 执行失败: {exc}")

        # ---- 2. Deflated Sharpe > 阈值 ----
        try:
            dsr = self._deflated_sharpe(dsl, dataset, n_trials)
            res.deflated_sharpe = float(dsr)
            if dsr <= self.dsr_threshold:
                reasons.append(f"DSR {dsr:.3f} ≤ 阈值 {self.dsr_threshold}")
        except Exception as exc:  # fail-closed
            logger.warning("[validation_gate] DSR 计算失败 → 判不通过: %s", exc)
            reasons.append(f"DSR 计算失败: {exc}")

        res.reasons = reasons
        res.passed = len(reasons) == 0
        logger.info(
            "[validation_gate] %s | passed=%s | folds=%d min_oos=%.3f DSR=%.3f | %s",
            dsl[:50], res.passed, res.n_folds, res.min_oos_sharpe, res.deflated_sharpe,
            "OK" if res.passed else "; ".join(reasons),
        )
        return res

    # ------------------------------------------------------------------
    # 内部：复用现有引擎
    # ------------------------------------------------------------------

    def _walk_forward(self, dsl: str, dataset: WidePanel):
        from app.core.backtest_engine.realistic_backtester import WalkForwardBacktester
        from app.core.alpha_engine.signal_processor import SimulationConfig

        wf_bt = WalkForwardBacktester(
            config=SimulationConfig(),
            n_splits=self.n_splits,
            embargo_days=self.embargo_days,
        )
        return wf_bt.run(dsl, dataset)

    def _deflated_sharpe(self, dsl: str, dataset: WidePanel, n_trials: int) -> float:
        from app.core.alpha_engine.dsl_executor import Executor
        from app.core.alpha_engine.signal_processor import SignalProcessor, SimulationConfig
        from app.core.backtest_engine.portfolio_constructor import SignalWeightedPortfolio
        from app.core.backtest_engine.backtest_engine import BacktestEngine
        from app.core.backtest_engine.performance_analyzer import deflated_sharpe_from_returns

        prices = dataset["close"]
        volume = dataset.get("volume")
        if volume is None:
            volume = pd.DataFrame(1e6, index=prices.index, columns=prices.columns)

        # 与每日循环同口径：Executor → SignalProcessor(delay=1) → 权重 → BacktestEngine
        cfg = SimulationConfig(delay=1, decay_window=0,
                               truncation_min_q=0.05, truncation_max_q=0.95)
        raw = Executor(validate=False).run_expr(dsl, dataset)
        proc = SignalProcessor(cfg).process(raw)
        weights = SignalWeightedPortfolio(clip_z=3.0).construct(proc)

        result = BacktestEngine().run(weights, prices, volume, proc)
        rets = pd.Series(result.net_returns).dropna()
        if len(rets) < 30 or float(np.nanstd(rets.values)) == 0.0:
            raise ValueError("净收益样本不足或方差为 0，无法计算 DSR")
        return deflated_sharpe_from_returns(rets, n_trials=n_trials)
