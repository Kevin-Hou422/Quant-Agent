"""
strategy_builder.py — 从当前因子构建一份可审批的**策略配置**（Phase PM.7）

把 PM.S2 边际准入 + PM.1 合成 + PM.S1 策略门 + PM.5 风控 + PM.6 换手 的产物，收敛成**一个
`StrategyConfig`**（组合成分 + 每因子配额 + 策略门 verdict + 风控快照 + 换手/无交易带）——即
"你真正要交易、需要被审批的那份策略"。不下单（那是 run_portfolio / 执行层的事）。
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from app.db.strategy_store import StrategyConfig

WidePanel = Dict[str, pd.DataFrame]
Signals = Dict[str, pd.DataFrame]


def build_strategy_config(
    factor_signals: Signals,
    dataset: WidePanel,
    aum: float,
    method: str = "ic_weighted",
    cost_params=None,
    risk_limits=None,
    marginal_min_improve: float = 0.05,
    name: str = "",
) -> StrategyConfig:
    """
    因子集合 → 一份策略配置。流程：边际准入(PM.S2) → 合成账本(PM.1) → 策略门(PM.S1) →
    风控(PM.5) → 换手/无交易带(PM.6)。返回 `StrategyConfig`（status=proposed）。
    """
    from app.core.portfolio_manager import (
        PortfolioManager, StrategyGate, marginal_factor_selection,
        PortfolioRiskGate, RiskLimits, apply_no_trade_band, annualized_turnover,
    )

    prices = dataset["close"]
    volume = dataset.get("volume")
    if volume is None:
        volume = pd.DataFrame(1e6, index=prices.index, columns=prices.columns)

    # PM.S2 边际准入
    signals = dict(factor_signals)
    if len(signals) > 1:
        try:
            sel = marginal_factor_selection(signals, dataset, aum=aum,
                                            min_improve=marginal_min_improve)
            if sel.selected:
                signals = {k: signals[k] for k in sel.selected}
        except Exception:
            pass

    # PM.1 合成账本
    book = PortfolioManager(aum=aum, method=method, cost_params=cost_params).build_book(
        signals, prices, volume)
    weights = book.weights

    # PM.S1 策略门
    verdict = {}
    passed = False
    try:
        sv = StrategyGate(aum=aum, method=method).evaluate(signals, dataset, cost_params=cost_params)
        verdict = sv.to_dict(); passed = sv.passed
    except Exception:
        pass

    # PM.5 风控（快照，不改配置权重——配置记录的是合成后的目标）
    risk_report = {}
    limits = risk_limits or RiskLimits()
    try:
        sectors = dataset["sector"].iloc[-1] if "sector" in dataset else None
        _, rep = PortfolioRiskGate(limits).apply(weights, sectors=sectors)
        risk_report = rep.to_dict()
    except Exception:
        pass

    # PM.6 换手 / 无交易带
    band = 0.0
    try:
        from app.core.trading_context.context import TradingContext
        band = float(TradingContext(aum=aum).analyze(dataset).rebalance_band)
    except Exception:
        band = 0.0
    turnover = annualized_turnover(apply_no_trade_band(weights, band))

    return StrategyConfig(
        factors=list(signals.keys()),
        combo_weights={k: float(v) for k, v in (book.combo_weights or {}).items()},
        aum=float(aum), method=method, passed=bool(passed),
        verdict=verdict, risk_report=risk_report,
        turnover_ann=float(turnover), no_trade_band=float(band),
        name=name, status="proposed",
    )


def propose_from_paper_factors(alpha_store, dataset: WidePanel, aum: float,
                              method: str = "ic_weighted", cost_params=None,
                              name: str = "") -> Optional[StrategyConfig]:
    """
    从 AlphaStore 里当前 **PAPER/ACTIVE** 因子构建一份策略配置。无可用因子 → None。
    因子 DSL → 信号（与 run_portfolio 同口径 delay=1）→ build_strategy_config。
    """
    from app.core.alpha_engine.dsl_executor import Executor
    from app.core.alpha_engine.signal_processor import SignalProcessor, SimulationConfig
    from app.db.alpha_lifecycle import AlphaStatus, coerce_status

    cfg = SimulationConfig(delay=1, decay_window=0, truncation_min_q=0.05, truncation_max_q=0.95)
    signals: Signals = {}
    for rec in alpha_store.query(limit=500):
        try:
            if coerce_status(rec.status) in (AlphaStatus.PAPER, AlphaStatus.ACTIVE, AlphaStatus.DECAYING):
                raw = Executor(validate=False).run_expr(rec.dsl, dataset)
                signals[str(rec.id)] = SignalProcessor(cfg).process(raw)
        except Exception:
            continue
    if not signals:
        return None
    return build_strategy_config(signals, dataset, aum=aum, method=method,
                                 cost_params=cost_params, name=name)
