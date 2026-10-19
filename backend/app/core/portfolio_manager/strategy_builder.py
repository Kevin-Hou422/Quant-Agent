"""
strategy_builder.py — 从当前因子构建一份可审批的**策略配置**（Phase PM.7）

把 PM.S2 边际准入 + PM.1 合成 + PM.S1 策略门 + PM.5 风控 + PM.6 换手 的产物，收敛成**一个
`StrategyConfig`**（组合成分 + 每因子配额 + 策略门 verdict + 风控快照 + 换手/无交易带）——即
"你真正要交易、需要被审批的那份策略"。不下单（那是 run_portfolio / 执行层的事）。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

from app.db.strategy_store import StrategyConfig

logger = logging.getLogger(__name__)

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
    factor_dsl: Optional[Dict[str, str]] = None,
) -> StrategyConfig:
    """
    因子集合 → 一份策略配置。流程：边际准入(PM.S2) → 合成账本(PM.1) → 策略门(PM.S1) →
    风控(PM.5) → 换手/无交易带(PM.6)。返回 `StrategyConfig`（status=proposed）。

    factor_dsl : 各成分的表达式。进可执行规格（F07）：每日循环据此核对"交易的成分还是批准时那几条"。
                 不给 → 规格里成分表达式为空，每日循环会因对不上而拒绝按它交易。
    """
    from app.core.portfolio_manager import (
        PortfolioManager, StrategyGate, marginal_factor_selection,
        PortfolioRiskGate, RiskLimits, apply_no_trade_band, annualized_turnover,
    )

    prices = dataset["close"]
    volume = dataset.get("volume")
    if volume is None:
        volume = pd.DataFrame(1e6, index=prices.index, columns=prices.columns)

    # 成本口径：None → **grounded 真实成本**（绝不用机构默认，见 §J / strategy_gate.resolve_cost_params）
    from app.core.portfolio_manager.strategy_gate import resolve_cost_params
    cost_params = resolve_cost_params(dataset, aum, cost_params)

    # 降级留痕：任何一步失败都记进来，随 verdict 一起持久化 ——
    # 使用者必须能看出"这份配置是完整评估出来的，还是带着窟窿产出的"。
    degraded: list[str] = []

    # PM.S2 边际准入
    signals = dict(factor_signals)
    if len(signals) > 1:
        try:
            sel = marginal_factor_selection(signals, dataset, aum=aum,
                                            min_improve=marginal_min_improve)
            if sel.selected:
                signals = {k: signals[k] for k in sel.selected}
        except Exception as exc:
            # 静默吞掉会让"边际筛选失败"与"筛选后就是全选"无法区分，
            # 且结果照常产出（审计 #9）。至少要记录并在 verdict 里留痕。
            logger.warning("[strategy_builder] PM.S2 边际筛选失败（沿用全部因子）: %s", exc)
            degraded.append(f"marginal_selection_failed: {exc}")

    # PM.1 合成账本
    book = PortfolioManager(aum=aum, method=method, cost_params=cost_params).build_book(
        signals, prices, volume)
    weights = book.weights

    # 复审 2026-10-19 N05：先把要交易的那份冻结下来（合成权重 + 无交易带），门评估的就是它。
    # 此前门先按 method 滚动拟合评估，之后才确定并保存冻结权重与带 —— 门的结论说的不是存下的那份策略。
    band = 0.0
    try:
        from app.config import settings as _settings
        band = float(getattr(_settings, "pm_no_trade_band", 0.0))
        if band <= 0.0:
            from app.core.trading_context.context import TradingContext
            band = float(TradingContext(aum=aum).analyze(dataset).rebalance_band)
    except Exception as exc:
        logger.warning("[strategy_builder] PM.6 无交易带推导失败（退回 0，换手会被高估）: %s", exc)
        band = 0.0
        degraded.append(f"band_derivation_failed: {exc}")
    combo = {k: float(v) for k, v in (book.combo_weights or {}).items()}

    # PM.S1 策略门：① 冻结规格本身（与每日门同一口径）；② 产出这份规格的**流程**的滚动前推 OOS。
    # 冻结权重是在整个选择段上拟合的，①在同一段上的收益不是样本外 —— 所以两个都要过，并且如实标注口径。
    verdict = {}
    passed = False
    try:
        gate = StrategyGate(aum=aum, method=method)
        sv = gate.evaluate(signals, dataset, cost_params=cost_params,
                           fixed_combo_weights=combo or None, no_trade_band=band)
        verdict = sv.to_dict(); passed = sv.passed
        verdict["evidence_scope"] = "frozen_spec_on_selection_in_sample"
        try:
            wf = gate.evaluate(signals, dataset, cost_params=cost_params, no_trade_band=band)
            verdict["procedure_oos"] = {"passed": wf.passed, "sharpe": wf.sharpe,
                                        "deflated_sharpe": wf.deflated_sharpe, "t_stat": wf.t_stat,
                                        "reasons": list(wf.reasons),
                                        "scope": "walk_forward_refit_of_the_procedure"}
            passed = bool(passed and wf.passed)
            if not wf.passed:
                verdict.setdefault("reasons", [])
                verdict["reasons"] = list(verdict["reasons"]) + [f"流程的滚动前推 OOS 未过：{'; '.join(wf.reasons)}"]
        except Exception as exc:
            logger.error("[strategy_builder] 流程 OOS 评估失败（无证据）: %s", exc)
            verdict["procedure_oos"] = {"evaluated": False, "error": str(exc)}
            passed = False
            degraded.append(f"procedure_oos_failed: {exc}")
    except Exception as exc:
        # 策略门崩了 ≠ 策略门未过。前者是"没有证据"，后者是"证据不足"，
        # 二者混为 passed=False 会让人以为门评过了。必须显式标注。
        logger.error("[strategy_builder] PM.S1 策略门评估失败（无证据，非未过）: %s", exc)
        verdict = {"gate_error": str(exc), "evaluated": False}
        degraded.append(f"strategy_gate_failed: {exc}")

    # TR.4 第 4 步：进 PAPER 的 **A/B/C 分级**（实验模式下 B/C 也放行，但等级如实标注）。
    # 存进 verdict JSON（免 schema 迁移），前端与审批据此看清"这份策略证据有多强"。
    try:
        from app.core.lifecycle.promotion_gate import grade_paper_entry
        allowed, gdetail = grade_paper_entry(verdict)
        verdict["paper_grade"] = gdetail["grade"]
        verdict["paper_entry"] = gdetail
        if not allowed:
            verdict["paper_entry_blocked"] = True      # 非实验模式且非 A 级
    except Exception as exc:
        logger.warning("[strategy_builder] TR.4 进 PAPER 分级失败: %s", exc)
        verdict["paper_grade"] = "unknown"
        degraded.append(f"paper_grading_failed: {exc}")

    # PM.5 风控（快照，不改配置权重——配置记录的是合成后的目标）
    # F07：上限与每日循环同一来源（此前缺省用 RiskLimits() 的 dataclass 默认，与实际交易的上限不同）
    from app.core.portfolio_manager.executable_spec import build_spec, risk_limits_from_settings, spec_hash
    risk_report = {}
    limits = risk_limits or risk_limits_from_settings()
    try:
        sectors = dataset["sector"].iloc[-1] if "sector" in dataset else None
        _, rep = PortfolioRiskGate(limits).apply(weights, sectors=sectors)
        risk_report = rep.to_dict()
    except Exception as exc:
        logger.error("[strategy_builder] PM.5 风控快照失败（该配置无风险证据）: %s", exc)
        risk_report = {"risk_error": str(exc), "evaluated": False}
        degraded.append(f"risk_snapshot_failed: {exc}")

    # PM.6 换手 / 无交易带（带已在上面冻结：与每日循环同一规则，配置优先，否则由数据推导）
    turnover = annualized_turnover(apply_no_trade_band(weights, band))

    if degraded:
        verdict["degraded"] = degraded
        logger.warning("[strategy_builder] 该配置在 %d 处降级产出：%s", len(degraded), degraded)
    # F07：影响交易的全部参数收进一份规格 + 哈希；每日循环照它执行、并核对运行时没变
    spec = build_spec({k: (factor_dsl or {}).get(k) for k in signals}, combo, method, aum, band, limits)
    verdict["executable_spec"] = spec
    verdict["spec_hash"] = spec_hash(spec)
    return StrategyConfig(
        factors=list(signals.keys()),
        combo_weights=combo,
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
    from app.core.alpha_engine.signal_processor import SignalProcessor
    from app.core.portfolio_manager.executable_spec import signal_config
    from app.db.alpha_lifecycle import AlphaStatus, coerce_status

    cfg = signal_config()
    signals: Signals = {}
    dsl_of: Dict[str, str] = {}
    skipped: list = []
    for rec in alpha_store.query(limit=500):
        try:
            if coerce_status(rec.status) in (AlphaStatus.PAPER, AlphaStatus.ACTIVE, AlphaStatus.DECAYING):
                raw = Executor(validate=False).run_expr(rec.dsl, dataset)
                signals[str(rec.id)] = SignalProcessor(cfg).process(raw)
                dsl_of[str(rec.id)] = rec.dsl
        except Exception as exc:
            # 静默 continue 会让因子"从组合里消失"而无人知晓（审计 #9）。
            logger.warning("[strategy_builder] 因子 id=%s 信号生成失败，已排除: %s",
                           getattr(rec, "id", "?"), exc)
            skipped.append(getattr(rec, "id", "?"))
            continue
    if skipped:
        logger.warning("[strategy_builder] 本次构建排除了 %d 个因子：%s", len(skipped), skipped)
    if not signals:
        return None
    return build_strategy_config(signals, dataset, aum=aum, method=method,
                                 cost_params=cost_params, name=name, factor_dsl=dsl_of)
