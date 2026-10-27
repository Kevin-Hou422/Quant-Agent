"""
strategy_contract.py — 一份策略的显式三步合同（复审 2026-10-19 N05-e / §3）

复审原话："建议把规格生成提前，建立显式的三步合同：构建并冻结 spec → 验证该 spec → 以该 spec 生成
目标/证据/订单。回测和实时执行需要不同的成交适配器，但应共享同一信号、权重和风险决策过程。"

此前同一份策略在四处重新装配：提案（strategy_builder）、策略门（strategy_net_returns）、每日循环
（run_portfolio）、前向证据（_record_strategy_evidence）。各自拼信号 → 合成 → 风控 → 无交易带，
口径悄悄分叉 —— 例：门按 grounded 成本参数建账本（影响容量），每日循环用默认 CostParams()；
证据路径用第三套合成调用。现在：

    1. 冻结   contract = StrategyContract.freeze(factor_dsl, combo, method, aum, band)
              contract = StrategyContract.from_record(config)        # 已存的那份（哈希对不上 → None）
    2. 验证   contract.verify(factor_dsl_now, aum_now, settings)     # 运行时与规格的差异
              contract.evaluate(gate, signals, dataset, cost_params) # 门评估的就是这份
    3. 生成   contract.signals(dataset)                              # 规格里的表达式 + 规格里的信号参数
              contract.composite(signals, dataset)                   # 证据用的合成信号
              contract.targets(signals, dataset, cost_params, halt)  # 目标权重（门回测 / 模拟账本 / 券商）

没有批准配置的探索路径（PAPER 因子、提案的流程 OOS）用同一个 `generate_targets`，只是参数不来自规格。
成交适配器不同：门 → BacktestEngine；每日循环 → PaperBroker + 券商订单管理。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.portfolio_manager import executable_spec as _es
from app.core.portfolio_manager.executable_spec import approved_spec, build_spec, spec_hash, verify_against_runtime

logger = logging.getLogger(__name__)

WidePanel = Dict[str, pd.DataFrame]
Signals = Dict[str, pd.DataFrame]


@dataclass
class StrategyTargets:
    """一次目标生成的全部产物（门、模拟账本、券商都从这里取，不再各算各的）。"""
    weights:   pd.DataFrame            # 风控 →（熔断）→ 无交易带 → 硬约束投影：回测 / 模拟账本的目标
    gated:     pd.DataFrame            # 风控（及熔断）之后、无交易带之前：券商执行的目标
    raw:       pd.DataFrame            # 合成账本的原始权重（风控前）
    composite: pd.DataFrame
    combo_weights: Dict[str, float]
    band:      float
    port_vol_ann: Optional[float]
    risk_report: Any = None
    post_band_violations: List[str] = field(default_factory=list)
    notes:     List[str] = field(default_factory=list)


def generate_targets(signals: Signals, dataset: WidePanel, *, aum: float, method: str, limits,
                     band: float, cost_params=None, fixed_combo_weights: Optional[Dict[str, float]] = None,
                     walk_forward_folds: Optional[int] = None, vol_lookback: int = 60,
                     halt: bool = False, risk: bool = True) -> StrategyTargets:
    """
    信号 → 合成账本 → 目标波动缩放（同一个波动估计）→ 硬约束 →（回撤熔断 → 全 0）→ 无交易带 → 带后硬约束投影。
    **唯一**的目标生成路径（策略门的回测、提案的风控快照、每日循环的模拟账本与券商目标都走这里）。
    波动估计 / 风控抛错 → 照常抛出（调用方 fail-closed）；样本不足 → 本次不缩放（port_vol_ann=None）。
    `risk=False`：只要合成账本（提案拟合合成权重、`apply_risk=False` 的诊断），weights = gated = raw。
    """
    from app.core.portfolio_manager.manager import PortfolioManager
    from app.core.portfolio_manager.risk_gate import PortfolioRiskGate

    prices = dataset["close"]
    volume = dataset.get("volume")
    if volume is None:
        volume = pd.DataFrame(1e6, index=prices.index, columns=prices.columns)
    notes: List[str] = []
    from app.core.clock import paired_returns
    from app.core.portfolio_manager.manager import needs_returns
    needs_fit = needs_returns(method, len(signals), fixed=fixed_combo_weights is not None)
    book = PortfolioManager(aum=aum, method=method, cost_params=cost_params).build_book(
        signals, prices, volume, walk_forward_folds=walk_forward_folds,
        fixed_combo_weights=fixed_combo_weights,
        returns=paired_returns(dataset) if needs_fit else None)
    raw = book.weights
    if not risk:
        return StrategyTargets(weights=raw, gated=raw, raw=raw, composite=book.composite,
                               combo_weights={str(k): float(v) for k, v in (book.combo_weights or {}).items()},
                               band=float(band), port_vol_ann=None, notes=["risk_not_applied"])
    sectors = None
    if "sector" in dataset:
        try:
            sectors = dataset["sector"].iloc[-1]
        except Exception as exc:
            logger.warning("[contract] 行业标签读取失败，行业集中度约束本次不生效: %s", exc)
            notes.append(f"sector_unavailable: {exc}")
    # 逐决策日的波动估计（审计 §4.2：不拿整段样本末尾的一个标量乘到全部历史）
    vol_series = _es.estimate_port_vol_series(raw, prices, int(vol_lookback)) if limits.target_vol_ann else None
    vol = None
    if vol_series is not None and len(vol_series) and np.isfinite(vol_series.iloc[-1]):
        vol = float(vol_series.iloc[-1])
    if limits.target_vol_ann and vol is None:
        notes.append("target_vol_skipped_insufficient_sample")
    gated, report = PortfolioRiskGate(limits).apply(raw, sectors=sectors, port_vol_ann=vol_series)
    if halt:
        gated = gated * 0.0
        notes.append("drawdown_halt_flatten")
    weights = _es.band_then_reproject(gated, band, limits, sectors)
    violations = PortfolioRiskGate(limits).check(weights.tail(1), sectors=sectors)
    return StrategyTargets(
        weights=weights, gated=gated, raw=raw, composite=book.composite.reindex(columns=weights.columns),
        combo_weights={str(k): float(v) for k, v in (book.combo_weights or {}).items()},
        band=float(band), port_vol_ann=vol, risk_report=report,
        post_band_violations=list(violations or []), notes=notes,
    )


class SpecValidationRefused(ValueError):
    """已批准规格的表达式按它冻结的校验政策核对不过（或没有凭据）。"""


class StrategyContract:
    """冻结的可执行规格 + 照它验证、照它生成。规格内容决定一切；运行时只提供数据与成交适配器。"""

    def __init__(self, spec: dict, config_id: Optional[int] = None) -> None:
        self.spec = spec
        self.hash = spec_hash(spec)
        self.config_id = config_id

    # ------------------------------------------------------------------ 1. 冻结
    @classmethod
    def freeze(cls, factor_dsl: Dict[str, Optional[str]], combo_weights: Dict[str, float], method: str,
               aum: float, band: float, limits=None, settings=None) -> "StrategyContract":
        return cls(build_spec(factor_dsl, combo_weights, method, aum, band, limits, settings=settings))

    @classmethod
    def from_record(cls, record) -> Optional["StrategyContract"]:
        """已存配置的合同。没有规格 / 哈希对不上（被改过）→ None：无法证明交易的是被批准的那份。"""
        spec = approved_spec(record)
        return None if spec is None else cls(spec, getattr(record, "id", None))

    # ------------------------------------------------------------------ 规格内容
    @property
    def factors(self) -> Dict[str, Optional[str]]:
        return dict(self.spec["factors"])

    @property
    def combo_weights(self) -> Optional[Dict[str, float]]:
        return {str(k): float(v) for k, v in self.spec["combo_weights"].items()} or None

    @property
    def method(self) -> str:
        return str(self.spec["method"])

    @property
    def band(self) -> float:
        return float(self.spec["no_trade_band"])

    @property
    def aum(self) -> float:
        return float(self.spec["aum"])

    @property
    def vol_lookback(self) -> int:
        return int(self.spec.get("risk_vol_lookback", 60))

    def limits(self):
        """风控上限**取自规格**（verify 已确认与运行时一致；生成目标时不再现读 settings）。"""
        from app.core.portfolio_manager.risk_gate import RiskLimits
        return RiskLimits(**dict(self.spec["risk_limits"]))

    def signal_config(self):
        from app.core.alpha_engine.signal_processor import SimulationConfig
        return SimulationConfig(**dict(self.spec["signal"]))

    def complete(self) -> bool:
        """每个成分都有表达式（没有的话无法照规格算信号）。"""
        return bool(self.spec["factors"]) and all(v is not None for v in self.spec["factors"].values())

    # ------------------------------------------------------------------ 2. 验证
    def verify(self, factor_dsl: Dict[str, Optional[str]], aum: float, settings=None) -> List[str]:
        return verify_against_runtime(self.spec, factor_dsl, aum, settings=settings)

    def evaluate(self, gate, signals: Signals, dataset: WidePanel, cost_params=None):
        """门评估的就是这份合同（冻结合成权重 + 冻结的带）。"""
        return gate.evaluate(signals, dataset, cost_params=cost_params,
                             fixed_combo_weights=self.combo_weights, no_trade_band=self.band)

    # ------------------------------------------------------------------ 3. 生成
    def signals(self, dataset: WidePanel) -> Signals:
        """
        规格里的表达式 + 规格里的信号参数。某个成分算不出来 → 抛错（不能少一个成分照常交易）。
        表达式按**批准时冻结的校验政策**核对（2026-10-25 审计；`validation_policy.check_frozen`）：没有凭据、政策 /
        算子集对不上、按那一版不通过、安全底线不通过 → SpecValidationRefused（调用方不生成目标、不交易）。
        核对过之后执行器不再套用**现行**默认校验器 —— 已批准的规格按它批准时的那一版解释（不把 300 偷偷截成 252）。
        """
        from app.core.alpha_engine.dsl_executor import Executor
        from app.core.alpha_engine.signal_processor import SignalProcessor
        from app.core.alpha_engine.validation_policy import check_frozen
        if not self.complete():
            raise ValueError(f"规格成分缺表达式：{self.spec['factors']}")
        problems = check_frozen(self.spec)
        if problems:
            raise SpecValidationRefused("; ".join(problems))
        cfg = self.signal_config()
        return {f: SignalProcessor(cfg).process(Executor(validate=False).run_expr(dsl, dataset))
                for f, dsl in self.spec["factors"].items()}

    def composite(self, signals: Signals, dataset: WidePanel) -> pd.DataFrame:
        from app.core.clock import paired_returns
        from app.core.portfolio_manager.manager import PortfolioManager, needs_returns
        fixed = self.combo_weights
        _, _, comp = PortfolioManager(aum=self.aum, method=self.method).combined_weights(
            signals, dataset["close"], fixed,
            returns=paired_returns(dataset) if needs_returns(self.method, len(signals), fixed is not None)
            else None)
        return comp

    def targets(self, signals: Signals, dataset: WidePanel, cost_params=None,
                halt: bool = False) -> StrategyTargets:
        return generate_targets(signals, dataset, aum=self.aum, method=self.method, limits=self.limits(),
                                band=self.band, cost_params=cost_params,
                                fixed_combo_weights=self.combo_weights, vol_lookback=self.vol_lookback,
                                halt=halt)
