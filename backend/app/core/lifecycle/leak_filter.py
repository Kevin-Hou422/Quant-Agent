"""
leak_filter.py — 因子入池的**低门槛泄漏过滤**（Phase PM.7 修正错配）

设计背景（见 DEV_LESSONS §K 延伸 / 统一门控政策第 1 步）：因子级不应加**严门**（DSR>0.9/t≥3）——
那会筛掉"单独平庸但分散化极好"的因子，选出"漂亮因子"而非"好策略"。因子级只做**低门槛**：
仅滤掉**泄漏 / 明显退化的垃圾**，其余一律放进池子；**严格的统计验证上移到策略层**（StrategyGate）。

拒绝条件（低门槛）：
    - 信号执行失败 / 全 NaN / 截面零方差（常数、退化）；
    - IS 夏普高到不可信（默认 > 8，前视/泄漏的典型征兆）。
其余一律通过（admit 入池）。fail-closed：出错视为不通过。
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WidePanel = Dict[str, pd.DataFrame]


def leak_filter(dsl: str, dataset: WidePanel,
                max_plausible_sharpe: float = 8.0) -> Tuple[bool, dict]:
    """返回 (passed, detail)。低门槛：只滤泄漏/退化，不看 Sharpe 高低。"""
    from app.core.alpha_engine.dsl_executor import Executor
    from app.core.alpha_engine.signal_processor import SignalProcessor, SimulationConfig
    from app.core.backtest_engine.portfolio_constructor import SignalWeightedPortfolio
    from app.core.backtest_engine.backtest_engine import BacktestEngine

    reasons = []
    detail: dict = {"gate": "leak_filter"}
    try:
        raw = Executor(validate=False).run_expr(dsl, dataset)
        if not isinstance(raw, pd.DataFrame) or not raw.notna().any().any():
            return False, {**detail, "passed": False, "reasons": ["信号全 NaN / 执行退化"]}
        # 截面方差：常数信号（每日跨股票几乎无区分度）→ 退化
        cs_var = float(raw.var(axis=1).median())
        detail["cs_var"] = round(cs_var, 8)
        if not np.isfinite(cs_var) or cs_var < 1e-12:
            reasons.append("截面零方差（常数/退化信号）")

        # IS 夏普：过高 = 前视/泄漏征兆
        prices = dataset["close"]
        volume = dataset.get("volume")
        if volume is None:
            volume = pd.DataFrame(1e6, index=prices.index, columns=prices.columns)
        cfg = SimulationConfig(delay=1, decay_window=0, truncation_min_q=0.05, truncation_max_q=0.95)
        proc = SignalProcessor(cfg).process(raw)
        w = SignalWeightedPortfolio(clip_z=3.0).construct(proc)
        rets = pd.Series(BacktestEngine().run(w, prices, volume, proc).net_returns).dropna()
        if len(rets) >= 20:
            mu, sd = float(rets.mean()), float(rets.std(ddof=1))
            sharpe = (mu / sd) * np.sqrt(252.0) if sd > 1e-12 else 0.0
            detail["is_sharpe"] = round(float(sharpe), 3)
            if abs(sharpe) > max_plausible_sharpe:
                reasons.append(f"IS 夏普 {sharpe:.1f} 高到不可信（前视/泄漏征兆，>{max_plausible_sharpe}）")
    except Exception as exc:  # fail-closed
        logger.warning("[leak_filter] 执行失败 → 不通过: %s", exc)
        return False, {**detail, "passed": False, "reasons": [f"执行失败: {exc}"]}

    detail["passed"] = len(reasons) == 0
    detail["reasons"] = reasons
    return detail["passed"], detail
