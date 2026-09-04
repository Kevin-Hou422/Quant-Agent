"""
daily_trading_loop.py — 每日交易循环编排（Task 7.3，2026-08-02）

对每个 PAPER / ACTIVE 因子，逐个未处理交易日执行：
  信号（Executor+SignalProcessor，delay=1）→ 目标权重（PortfolioConstructor）
  → PaperBroker.step 模拟成交（幂等持久化）
  → AlphaMonitor.update(realized IC)（滚动监控）
  → 衰减检查 → 若 ACTIVE 触发衰减则状态机 ACTIVE→DECAYING
  → 汇总日报

健壮性：
  - **逐 alpha 隔离**：单个因子任一步失败 → 该因子当日跳过 + 告警，不影响其他因子
  - **崩溃恢复 / 幂等**：每个因子从 PositionStore 的 last_pnl_date 之后续跑；
    PaperBroker.step 的 state_before + record_day 覆盖式写入保证重跑不重复记账
  - realized IC：cross-section spearman(signal_{t-1}, ret_{t-1→t})，即"昨日信号是否
    rank-预测了今日收益"——标准的实盘信号 realized IC

数据源：本模块接收一个 dataset（dict[field→(T×N) DataFrame]，含至今全部历史）。
Phase 7.1 的 daily_ingest 每日追加数据后调用本循环；测试与历史补跑直接传整段 dataset。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _as_date(v) -> _date:
    """把 str / datetime / Timestamp / date 统一成 date（Phase 11 前向起始日比较用）。"""
    if isinstance(v, _date) and not hasattr(v, "date"):
        return v
    ts = pd.Timestamp(v)
    return ts.date()


# Phase PM.4：组合账本用的保留 book id（真实 alpha 的 id 为正自增，0 不冲突）
PORTFOLIO_BOOK_ID = 0


@dataclass
class AlphaDayResult:
    alpha_id:    int
    days_processed: int = 0
    last_date:   str = ""
    equity:      float = 1.0
    decay_alert: bool = False
    transitioned: str = ""            # 若发生状态流转，记录 "old→new"
    error:       str = ""


@dataclass
class LoopReport:
    date_range:  str = ""
    n_alphas:    int = 0
    results:     List[AlphaDayResult] = field(default_factory=list)
    n_errors:    int = 0
    n_alerts:    int = 0


class DailyTradingLoop:
    def __init__(
        self,
        store=None,          # AlphaStore
        broker=None,         # PaperBroker
        monitor=None,        # AlphaMonitor
    ) -> None:
        from app.db.alpha_store import AlphaStore
        from app.core.execution.paper_broker import PaperBroker
        from app.core.monitor.alpha_monitor import AlphaMonitor
        from app.config import settings
        self.store   = store or AlphaStore()
        # 单一真实 AUM 来源：PaperBroker 的 initial_capital = settings.paper_aum。
        # 成本(含固定/最小手续费)与容量都以此计——修掉"PM 用 paper_aum、broker 用自带 1M"的不一致。
        self.broker  = broker or PaperBroker(
            store=self._shared_position_store(),
            initial_capital=float(settings.paper_aum),
        )
        self.monitor = monitor or AlphaMonitor(self.store)

    @staticmethod
    def _shared_position_store():
        from app.db.position_store import PositionStore
        return PositionStore()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self, dataset: Dict[str, pd.DataFrame]) -> LoopReport:
        """
        对全部 PAPER / ACTIVE 因子跑交易循环（每个因子从其续跑点到 dataset 末日）。
        逐 alpha 隔离——任一因子失败不影响其他。
        """
        from app.db.alpha_lifecycle import AlphaStatus, coerce_status

        prices = dataset.get("close")
        if prices is None or prices.empty:
            raise ValueError("dataset 缺少 close 或为空")

        # 取活跃因子（PAPER / ACTIVE / DECAYING）
        candidates = []
        for rec in self.store.query(limit=500):
            try:
                st = coerce_status(rec.status)
            except ValueError:
                continue
            if st in (AlphaStatus.PAPER, AlphaStatus.ACTIVE, AlphaStatus.DECAYING):
                candidates.append(rec)

        report = LoopReport(
            date_range=f"{prices.index[0].date()}→{prices.index[-1].date()}",
            n_alphas=len(candidates),
        )
        for rec in candidates:
            res = self._run_one_alpha(rec, dataset)
            report.results.append(res)
            if res.error:
                report.n_errors += 1
            if res.decay_alert:
                report.n_alerts += 1

        logger.info(
            "[daily_trading_loop] %s | %d 因子 | %d 告警 | %d 错误",
            report.date_range, report.n_alphas, report.n_alerts, report.n_errors,
        )
        return report

    # ------------------------------------------------------------------
    # Phase PM.4：组合账本（把全部 paper 因子合成一个真实 AUM 的美元账本来交易）
    # ------------------------------------------------------------------

    def run_portfolio(self, dataset: Dict[str, pd.DataFrame], aum: Optional[float] = None,
                      forward_from=None) -> dict:
        """
        把全部 PAPER/ACTIVE/DECAYING 因子经 PortfolioManager 合成**一个组合账本**
        （多因子净持仓 + 容量约束 + 真实 AUM），在保留 book id=PORTFOLIO_BOOK_ID 下交易。
        这是"真实交易员账本"侧；per-factor 的 realized IC 监控仍由 run() 负责。
        **AUM 单一来源** = self.broker.initial_capital(= settings.paper_aum)，PM 容量与 broker
        成本/容量同口径；未显式传 aum 时默认用它，杜绝口径分叉。
        """
        from app.core.alpha_engine.dsl_executor import Executor
        from app.core.alpha_engine.signal_processor import SignalProcessor, SimulationConfig
        from app.core.portfolio_manager.manager import PortfolioManager
        from app.db.alpha_lifecycle import AlphaStatus, coerce_status

        # 单一真实 AUM：默认取 broker 的 initial_capital，保证 PM 容量与 broker 成本/容量同口径
        aum = float(aum) if aum is not None else float(self.broker.initial_capital)
        prices = dataset.get("close")
        if prices is None or prices.empty:
            raise ValueError("dataset 缺少 close")
        volume = dataset.get("volume")
        if volume is None:
            volume = pd.DataFrame(1e6, index=prices.index, columns=prices.columns)

        # 组合成分 = 活跃因子
        recs = []
        for rec in self.store.query(limit=500):
            try:
                if coerce_status(rec.status) in (AlphaStatus.PAPER, AlphaStatus.ACTIVE, AlphaStatus.DECAYING):
                    recs.append(rec)
            except ValueError:
                continue

        cfg = SimulationConfig(delay=1, decay_window=0, truncation_min_q=0.05, truncation_max_q=0.95)
        signals: Dict[str, pd.DataFrame] = {}
        used_baseline = False
        if recs:
            for rec in recs:
                try:
                    raw = Executor(validate=False).run_expr(rec.dsl, dataset)
                    signals[str(rec.id)] = SignalProcessor(cfg).process(raw)
                except Exception as exc:
                    logger.warning("[portfolio] 因子 %s 信号失败，剔除组合: %s", rec.id, exc)
        else:
            # PM.S3：门控下暂无 PAPER/ACTIVE 自研因子 → 回退到经典基准策略库，
            # 保证"没有自研因子时交易环节仍能运转"（用户明确关切）。用 baseline: 前缀标记来源。
            from app.core.strategies import baseline_signals
            used_baseline = True
            for name, raw in baseline_signals(dataset).items():
                try:
                    signals[f"baseline:{name}"] = SignalProcessor(cfg).process(raw)
                except Exception as exc:
                    logger.warning("[portfolio] 基准 %s 信号失败，剔除组合: %s", name, exc)
            if signals:
                logger.info("[portfolio] 无自研 PAPER/ACTIVE 因子，回退经典基准策略库（%d 个）交易", len(signals))
        if not signals:
            return {"n_factors": 0, "days_processed": 0,
                    "reason": "no active factors" if not recs else "no valid signals",
                    "used_baseline": used_baseline}

        from app.config import settings

        # ── PM.7：若已有 **active 策略配置**，只交易它已批准的成分（配置=资金决策单位）。
        using_active_config = None
        if not used_baseline:
            try:
                import json as _json
                from app.db.strategy_store import StrategyStore
                active = StrategyStore().latest_active()
                if active is not None:
                    cfg_factors = set(_json.loads(active.factors or "[]"))
                    filtered = {k: v for k, v in signals.items() if k in cfg_factors}
                    if filtered:
                        signals = filtered
                        using_active_config = active.id
                        logger.info("[portfolio] PM.7 按 active 策略配置 #%d 交易，成分 %d 因子",
                                    active.id, len(signals))
            except Exception as exc:
                logger.warning("[portfolio] PM.7 读取 active 配置失败（忽略）: %s", exc)

        # ── PM.S2：多因子（自研）时按**边际贡献**准入（策略级），而非"全纳入"。
        #    冗余/无边际的因子被拒，单独弱但分散化好的能进（要"好策略"而非"漂亮因子"）。
        #    已按 active 配置交易时跳过（配置提出时已选定成分）。
        selection_info = None
        if (not used_baseline and using_active_config is None and len(signals) > 1
                and getattr(settings, "pm_marginal_selection", True)):
            from app.core.portfolio_manager import marginal_factor_selection
            try:
                sel = marginal_factor_selection(
                    signals, dataset, aum=aum,
                    min_improve=getattr(settings, "pm_marginal_min_improve", 0.05))
                if sel.selected:
                    signals = {k: signals[k] for k in sel.selected}
                    selection_info = sel.to_dict()
                    logger.info("[portfolio] PM.S2 边际准入：%d 候选 → 选中 %d | %s",
                                len(sel.steps), len(signals), sel.selected)
            except Exception as exc:                       # 选择失败不阻断，退回全纳入
                logger.warning("[portfolio] PM.S2 边际准入失败（退回全纳入）: %s", exc)

        res = PortfolioManager(aum=aum, method="ic_weighted").build_book(signals, prices, volume)
        weights, composite = res.weights, res.composite

        # TR.3：组合账本用**真实/推导成本**（moomoo 佣金免费 + Corwin-Schultz 数据估价差），
        # 而非硬编码机构默认。为组合账本单建一个用 grounded 成本的 broker（同一 PositionStore）。
        from app.core.execution.paper_broker import PaperBroker
        from app.core.trading_context.context import grounded_cost_params, get_broker_profile
        profile = get_broker_profile(getattr(settings, "trading_broker", "moomoo_us"))
        try:
            gp = grounded_cost_params(dataset, broker=profile, aum=aum)
        except Exception as exc:                          # 估计失败退回原成本，不阻断
            logger.warning("[portfolio] grounded 成本估计失败，用默认: %s", exc)
            gp = None

        # TR.1：交易现实摘要（估计价差/可交易池/可做空性/单边成本/无交易带）—— 供 FE-TR 展示
        tc_summary = None
        try:
            from app.core.trading_context import TradingContext
            tc_summary = TradingContext(
                aum=aum, broker=profile,
                account_type=getattr(settings, "trading_account_type", "margin"),
                allow_short=getattr(settings, "trading_allow_short", False),
            ).analyze(dataset).to_dict()
        except Exception as exc:
            logger.warning("[portfolio] TradingContext 摘要失败（不阻断）: %s", exc)
        pf_broker = PaperBroker(store=self.broker.store, cost_params=gp, initial_capital=aum) \
            if gp is not None else self.broker

        # ── PM.S1：对**组合策略**跑策略级验证门（分段 OOS + DSR 去膨胀 + 夏普 t），记录 verdict。
        #    严门加在真正交易的策略上、不加单因子。默认只记录不阻断（paper 期先收前向证据）。
        strategy_verdict = None
        if getattr(settings, "pm_strategy_gate_eval", True) and not used_baseline:
            from app.core.portfolio_manager import StrategyGate
            try:
                sv = StrategyGate(aum=aum).evaluate(signals, dataset, cost_params=gp)
                strategy_verdict = sv.to_dict()
                # TR.4 第 4 步：进 PAPER 的 A/B/C 分级（实验模式放行 B/C 但如实标注等级）
                try:
                    from app.core.lifecycle.promotion_gate import grade_paper_entry
                    _allowed, _g = grade_paper_entry(strategy_verdict)
                    strategy_verdict["paper_grade"] = _g["grade"]
                    strategy_verdict["paper_entry"] = _g
                    logger.info("[portfolio] TR.4 进 PAPER 分级=%s（allowed=%s, 实验模式=%s）",
                                _g["grade"], _allowed, _g["experiment_mode"])
                except Exception as exc:
                    logger.warning("[portfolio] TR.4 分级失败（不阻断）: %s", exc)
                logger.info("[portfolio] PM.S1 策略门：passed=%s Sharpe=%.3f DSR=%.3f t=%.2f | %s",
                            sv.passed, sv.sharpe, sv.deflated_sharpe, sv.t_stat,
                            "OK" if sv.passed else "; ".join(sv.reasons))
                if not sv.passed and getattr(settings, "pm_strategy_gate_block", False):
                    logger.warning("[portfolio] 策略门未过且 block=True → 本轮不交易")
                    return {"n_factors": len(signals), "days_processed": 0,
                            "reason": "strategy_gate_failed", "strategy_verdict": strategy_verdict,
                            "selection": selection_info, "used_baseline": used_baseline}
            except Exception as exc:
                logger.warning("[portfolio] PM.S1 策略门评估失败（不阻断）: %s", exc)

        # ── PM.5：组合级风控（敞口/集中度/目标波动）——在容量后、下单前施加到权重面板。
        from app.core.portfolio_manager import PortfolioRiskGate, RiskLimits
        limits = RiskLimits(
            max_gross=float(getattr(settings, "risk_max_gross", 1.0)),
            max_name_weight=float(getattr(settings, "risk_max_name_weight", 0.10)),
            max_sector_weight=float(getattr(settings, "risk_max_sector_weight", 0.30)),
            target_vol_ann=(float(getattr(settings, "risk_target_vol_ann", 0.0)) or None),
            max_drawdown=float(getattr(settings, "risk_max_drawdown", 0.20)),
            long_only=(not getattr(settings, "trading_allow_short", False)),
        )
        sectors = None
        if "sector" in dataset:
            try:
                sectors = dataset["sector"].iloc[-1]
            except Exception:
                sectors = None
        weights, risk_report = PortfolioRiskGate(limits).apply(weights, sectors=sectors)
        composite = composite.reindex(columns=weights.columns)
        logger.info("[portfolio] PM.5 风控：%s", risk_report.to_dict())

        # ── PM.5 回撤熔断：按组合已有净值序列判定；触发且 halt=True → 本轮清仓（全 0）de-risk。
        halt_info = None
        try:
            hist = self.broker.store.pnl_history(PORTFOLIO_BOOK_ID, limit=1000)
            if hist:
                eq_series = pd.Series([h.equity for h in hist])
                halt, dd = PortfolioRiskGate(limits).should_halt(eq_series)
                halt_info = {"halt": bool(halt), "drawdown": round(float(dd), 4)}
                if halt and getattr(settings, "risk_halt_on_drawdown", False):
                    logger.warning("[portfolio] 回撤熔断 dd=%.3f → 本轮清仓 de-risk", dd)
                    weights = weights * 0.0
        except Exception as exc:
            logger.warning("[portfolio] 回撤熔断检查失败（不阻断）: %s", exc)

        # ── PM.6：无交易带调仓（合并 TR.1 band，减换手省成本）+ 因子快慢分类 ──
        from app.core.portfolio_manager import (apply_no_trade_band, annualized_turnover,
                                                horizon_profile)
        band = float(getattr(settings, "pm_no_trade_band", 0.0))
        if band <= 0.0:                                   # 由 TradingContext 数据推导（T2）
            try:
                from app.core.trading_context.context import TradingContext
                band = float(TradingContext(aum=aum).analyze(dataset).rebalance_band)
            except Exception:
                band = 0.0
        to_before = annualized_turnover(weights)
        weights = apply_no_trade_band(weights, band)
        to_after = annualized_turnover(weights)
        horizon_info = [h.to_dict() for h in horizon_profile(
            signals, fast_threshold=float(getattr(settings, "pm_horizon_fast_thresh", 4.0)))]
        logger.info("[portfolio] PM.6 无交易带=%.4f 换手 %.2f→%.2f/年 | 快慢=%s",
                    band, to_before, to_after,
                    {h["factor"]: h["horizon"] for h in horizon_info})

        # 市场上下文（与 _run_one_alpha 同口径）
        cols = weights.columns
        prices_f = prices.reindex(columns=cols).ffill(limit=5).fillna(0.0)
        volume_f = volume.reindex(columns=cols).ffill(limit=5).fillna(0.0)
        adv_df = pf_broker._liq.compute_adv(volume_f, prices_f)            # noqa: SLF001
        vol_df = prices_f.pct_change().rolling(20, min_periods=2).std().fillna(0.02)
        ret_df = prices_f.pct_change()
        dates  = weights.index
        cal_years = max((dates[-1] - dates[0]).days / 365.25, 1 / 365.25)
        tdays = len(dates) / cal_years
        comp_arr = composite.reindex(columns=cols).to_numpy(dtype=float)

        last = pf_broker.store.last_pnl_date(PORTFOLIO_BOOK_ID)
        n_days = 0
        equity = pf_broker.store.latest_equity(PORTFOLIO_BOOK_ID)
        for t in range(len(dates)):
            d = dates[t]
            if last is not None and d.date() <= last:
                continue                                  # 幂等续跑
            pnl = pf_broker.step(
                PORTFOLIO_BOOK_ID, d,
                target_w=weights.iloc[t], prices_t=prices_f.iloc[t],
                prices_prev=prices_f.iloc[t - 1] if t > 0 else prices_f.iloc[t],
                adv_usd=adv_df.iloc[t], daily_vol=vol_df.iloc[t], tdays_per_year=tdays,
            )
            if t > 0:
                ic = _cs_spearman(comp_arr[t - 1], ret_df.iloc[t].to_numpy(dtype=float))
                if not np.isnan(ic):
                    # Phase 11：只有 forward_from 之后（摄取到新 bar 才有的日子）算真前向
                    fwd = bool(forward_from is not None and d.date() >= _as_date(forward_from))
                    self.store.record_ic(PORTFOLIO_BOOK_ID, d, float(ic),
                                         realized_return=float(pnl.net_ret),
                                         is_forward=fwd)
            equity = pnl.equity
            n_days += 1

        # ── TR.3：T3 providers（盘口/借券/账户）——仿真背 TR.1 估计、实盘背 moomoo，同接口切换。
        #    这里取账户真实记账状态（买入力/持仓），避免任何"交易当时才知道的量"被写死。
        t3_state = None
        try:
            from app.core.trading_context import get_trade_providers
            tp = get_trade_providers(
                "sim", dataset=dataset, aum=aum, broker=pf_broker, book_id=PORTFOLIO_BOOK_ID,
                account_type=getattr(settings, "trading_account_type", "margin"),
                allow_short=getattr(settings, "trading_allow_short", False))
            t3_state = tp.to_dict()
            logger.info("[portfolio] TR.3 T3 providers(%s)：买入力=$%.2f 持仓=%d 名",
                        tp.mode, tp.account.buying_power(), len(tp.account.positions()))
        except Exception as exc:
            logger.warning("[portfolio] TR.3 providers 获取失败（不阻断）: %s", exc)

        # ── PM.7 gap 修复：**策略级衰减监控**（交易的组合账本，不只因子级）。
        strategy_decay = None
        try:
            alert = self.monitor.check_decay(PORTFOLIO_BOOK_ID)
            if alert is not None:
                strategy_decay = {"reason": alert.reason,
                                  "rolling_mean_ic": round(float(alert.rolling_mean_ic), 5)}
                logger.warning("[portfolio] 策略级衰减告警：%s（组合 rolling_mean_ic=%.4f）",
                               alert.reason, alert.rolling_mean_ic)
        except Exception as exc:
            logger.warning("[portfolio] 策略衰减检查失败（不阻断）: %s", exc)

        logger.info("[portfolio] 组合账本 | AUM=%.0f | %d 因子 | %d 交易日 | 末净值=%.4f",
                    aum, len(signals), n_days, equity)
        result = {"n_factors": len(signals), "days_processed": n_days,
                "equity": equity, "aum": aum, "combo_weights": res.combo_weights,
                "used_baseline": used_baseline,
                "active_config": using_active_config,  # PM.7 交易的 active 策略配置 id
                "selection": selection_info,          # PM.S2 边际准入轨迹
                "strategy_verdict": strategy_verdict,  # PM.S1 策略门 verdict
                "risk_report": risk_report.to_dict(),  # PM.5 风控施加情况
                "drawdown": halt_info,                 # PM.5 回撤熔断状态
                "no_trade_band": round(band, 5),       # PM.6 无交易带宽
                "turnover_ann": round(to_after, 3),    # PM.6 带后年化换手
                "horizon": horizon_info,               # PM.6 因子快慢分类
                "strategy_decay": strategy_decay,      # PM.7 策略级衰减告警
                "t3": t3_state,                        # TR.3 T3 providers(模式/买入力/持仓数)
                "trading_context": tc_summary}         # TR.1 交易现实(价差/可交易/可做空/带)

        # FE-TR 前置：把本轮诊断只增不改地存下来（失败绝不影响交易）
        try:
            from app.db.diagnostics_store import DiagnosticsStore
            DiagnosticsStore().save(result)
        except Exception as exc:
            logger.warning("[portfolio] 诊断持久化失败（不阻断）: %s", exc)

        return result

    # ------------------------------------------------------------------
    # 单因子（隔离）
    # ------------------------------------------------------------------

    def _run_one_alpha(self, rec, dataset: Dict[str, pd.DataFrame]) -> AlphaDayResult:
        res = AlphaDayResult(alpha_id=rec.id)
        try:
            from app.core.alpha_engine.dsl_executor import Executor
            from app.core.alpha_engine.signal_processor import SignalProcessor, SimulationConfig
            from app.core.backtest_engine.portfolio_constructor import SignalWeightedPortfolio

            prices = dataset["close"]
            volume = dataset.get("volume")
            if volume is None:
                volume = pd.DataFrame(1e6, index=prices.index, columns=prices.columns)

            # 信号（含 delay=1）→ 权重
            cfg = SimulationConfig(delay=1, decay_window=0,
                                   truncation_min_q=0.05, truncation_max_q=0.95)
            raw_signal  = Executor(validate=False).run_expr(rec.dsl, dataset)
            proc_signal = SignalProcessor(cfg).process(raw_signal)
            weights     = SignalWeightedPortfolio(clip_z=3.0).construct(proc_signal)

            # 市场上下文（全窗口，与 PaperBroker.replay 口径一致）
            prices_f = prices.reindex(columns=weights.columns).ffill(limit=5).fillna(0.0)
            volume_f = volume.reindex(columns=weights.columns).ffill(limit=5).fillna(0.0)
            adv_df = self.broker._liq.compute_adv(volume_f, prices_f)      # noqa: SLF001
            vol_df = prices_f.pct_change().rolling(20, min_periods=2).std().fillna(0.02)
            ret_df = prices_f.pct_change()
            dates  = weights.index
            cal_years = max((dates[-1] - dates[0]).days / 365.25, 1/365.25)
            tdays = len(dates) / cal_years

            # 续跑点：last_pnl_date 之后
            last = self.broker.store.last_pnl_date(rec.id)
            raw_arr = raw_signal.reindex(columns=weights.columns).to_numpy(dtype=float)

            for t in range(len(dates)):
                d = dates[t]
                if last is not None and d.date() <= last:
                    continue                                  # 已处理（幂等续跑）
                # 模拟成交
                pnl = self.broker.step(
                    rec.id, d,
                    target_w=weights.iloc[t], prices_t=prices_f.iloc[t],
                    prices_prev=prices_f.iloc[t-1] if t > 0 else prices_f.iloc[t],
                    adv_usd=adv_df.iloc[t], daily_vol=vol_df.iloc[t],
                    tdays_per_year=tdays,
                )
                # realized IC：昨日信号 rank-预测今日收益
                if t > 0:
                    ic = _cs_spearman(raw_arr[t-1], ret_df.iloc[t].to_numpy(dtype=float))
                    if not np.isnan(ic):
                        self.monitor.update(rec.id, d, float(ic),
                                            realized_return=float(pnl.net_ret))
                res.days_processed += 1
                res.last_date = str(d.date())
                res.equity = pnl.equity

            # 衰减检查 + 状态流转（仅 ACTIVE→DECAYING）
            alert = self.monitor.check_decay(rec.id)
            if alert is not None:
                res.decay_alert = True
                from app.db.alpha_lifecycle import AlphaStatus, coerce_status
                if coerce_status(rec.status) == AlphaStatus.ACTIVE:
                    try:
                        self.store.update_status(rec.id, AlphaStatus.DECAYING.value)
                        res.transitioned = "active→decaying"
                        logger.warning("[daily_trading_loop] alpha %d 衰减 → DECAYING", rec.id)
                    except Exception as exc:
                        logger.warning("状态流转失败 alpha %d: %s", rec.id, exc)

        except Exception as exc:
            logger.exception("[daily_trading_loop] 因子 %d 当日处理失败", rec.id)
            res.error = str(exc)
        return res


def _cs_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """截面 Spearman rank 相关；有效对 < 3 返回 NaN。"""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[mask])).astype(float)
    rb = np.argsort(np.argsort(b[mask])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float(np.dot(ra, rb) / denom) if denom > 0 else float("nan")
