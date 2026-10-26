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
from datetime import date as _date, datetime
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.alpha_engine.fast_ops import average_ranks_1d
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
        gateway_factory=None,    # Phase 12：() -> BrokerGateway（默认按 settings 连 moomoo）
        execution_store=None,    # Phase 12：ExecutionStore（默认 settings.database_url）
        evidence_store=None,     # F06：EvidenceStore（信号存档 + 策略版本证据；默认 settings.database_url）
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
        self._gateway_factory = gateway_factory
        self._execution_store = execution_store
        self._evidence_store = evidence_store
        #: 本轮正式证据 / 每日记录 / 信号存档的写入失败（2026-10-25 审计 R4：不能只留日志却报"前向成功"）——
        #: run_portfolio 放进结果的 `record_failures`，每日结论与退出码据此判失败、发 critical
        self._record_failures: List[dict] = []

    def _note_failure(self, stage: str, exc) -> None:
        self._record_failures.append({"stage": stage, "error": f"{type(exc).__name__}: {exc}"})

    def _evidence(self):
        """证据库；打不开 → None（本轮所有 IC 按回放记，前向证据少一天，交易照常）。"""
        if self._evidence_store is None:
            try:
                from app.db.evidence_store import EvidenceStore
                self._evidence_store = EvidenceStore()
            except Exception as exc:
                logger.error("[daily_loop] 证据库打不开 → 本轮 IC 全部按回放记: %s", exc)
                self._note_failure("evidence_store", exc)
                return None
        return self._evidence_store

    def _forward_book(self, key: str):
        from app.db.evidence_store import ForwardBook
        return ForwardBook(self._evidence(), key, getattr(self, "_identity", None))

    def _set_identity(self, dataset: Dict[str, pd.DataFrame], data_source: Optional[str],
                      identity: Optional[dict]) -> Optional[dict]:
        """
        本轮输入身份（复审 2026-10-19 N02-c）：管线给了就用（摄取时对**交出去的那份面板**算的），
        否则就地对这份面板算。价格源只用**实际来源**（PIT 元数据）。
        2026-10-22 用户核验 C2：此前实际来源未知时退回 settings 的声称 —— 证据就按声称的来源累计前向。
        现在未知 → price_source 为空、source_verified=False：探索照常，但前向结算一律不认（见 ForwardBook）。
        """
        if identity:
            self._identity = dict(identity)
            self._identity.setdefault("source_verified", bool(self._identity.get("price_source")))
            return self._identity
        try:
            from app.core.data_engine.input_identity import identify
            self._identity = identify(dataset, price_source=data_source or "").to_dict()
            self._identity["source_verified"] = bool(data_source)
            if not data_source:
                logger.warning("[daily_loop] 本轮数据的实际价格源未知 → 证据只记回放，不计正式前向")
        except Exception as exc:
            logger.error("[daily_loop] 输入身份算不出来（证据行不带身份）: %s", exc)
            self._note_failure("input_identity", exc)
            self._identity = None
        return self._identity

    @staticmethod
    def _shared_position_store():
        from app.db.position_store import PositionStore
        return PositionStore()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self, dataset: Dict[str, pd.DataFrame], forward_from=None,
            now: Optional[datetime] = None, data_source: Optional[str] = None,
            identity: Optional[dict] = None) -> LoopReport:
        """
        对全部 PAPER / ACTIVE 因子跑交易循环（每个因子从其续跑点到 dataset 末日）。
        逐 alpha 隔离——任一因子失败不影响其他。

        now : 决策时刻（UTC）。当天的信号以它为存档时刻（F06）；不给 = 现在。
        forward_from : 只作记录。**哪天算前向**由信号存档决定（见 evidence_store）。
        """
        from app.db.alpha_lifecycle import AlphaStatus, coerce_status

        prices = dataset.get("close")
        if prices is None or prices.empty:
            raise ValueError("dataset 缺少 close 或为空")
        self._set_identity(dataset, data_source, identity)

        # 取活跃因子（PAPER / ACTIVE / DECAYING）
        candidates = []
        n_bad_status = 0
        for rec in self.store.query(limit=500):
            try:
                st = coerce_status(rec.status)
            except ValueError:
                # 状态串无法解析 → 该因子**静默退出当日交易账本**。
                # 这是实盘账本组成，不能无声。
                n_bad_status += 1
                logger.warning("[daily_loop] alpha id=%s 状态 %r 无法解析，已排除出当日账本",
                               getattr(rec, "id", "?"), getattr(rec, "status", None))
                continue
            if st in (AlphaStatus.PAPER, AlphaStatus.ACTIVE, AlphaStatus.DECAYING):
                candidates.append(rec)

        report = LoopReport(
            date_range=f"{prices.index[0].date()}→{prices.index[-1].date()}",
            n_alphas=len(candidates),
        )
        for rec in candidates:
            res = self._run_one_alpha(rec, dataset, forward_from, now=now)
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
                      forward_from=None, now: Optional[datetime] = None,
                      data_source: Optional[str] = None, identity: Optional[dict] = None) -> dict:
        """组合账本一轮（见 `_run_portfolio`）+ 这一天的前向记录（复审 2026-10-19 §7.3；任何提前返回也记）。"""
        self._record_failures = []
        result = self._run_portfolio(dataset, aum=aum, forward_from=forward_from, now=now,
                                     data_source=data_source, identity=identity)
        self._record_forward_day(result, dataset, now)
        result["record_failures"] = list(self._record_failures)
        return result

    def _record_forward_day(self, result: dict, dataset: Dict[str, pd.DataFrame],
                            now: Optional[datetime]) -> None:
        """§7.3 每日记录：只记 active 策略（交易的那一份）；失败不影响交易，但留 ERROR。"""
        try:
            import json as _json
            from datetime import timezone as _tz
            from app.db.execution_store import ExecutionStore
            from app.db.forward_record import ForwardRecordStore, build_daily_record
            from app.db.strategy_store import StrategyStore
            sid, h = result.get("active_config"), result.get("spec_hash")
            active = StrategyStore().get(int(sid)) if sid is not None else StrategyStore().latest_running()
            if active is None:
                return
            sid = active.id
            h = h or _json.loads(active.verdict or "{}").get("spec_hash", "")
            store = ForwardRecordStore()
            rec = build_daily_record(
                result, strategy_id=int(sid), spec_hash=h or "", decision_date=dataset["close"].index[-1],
                now=now or datetime.now(_tz.utc), dataset=dataset,
                paper_history=self.broker.store.pnl_history(PORTFOLIO_BOOK_ID, limit=1000),
                exec_store=self._execution_store or ExecutionStore(),
                identity=getattr(self, "_identity", None),
                forward_run=store.run_for(int(sid), h) if h else None,
                strategy_status=str(active.status),
                previous=store.previous_decision(int(sid), dataset["close"].index[-1]))
            store.record_day(rec)
            result["forward_record"] = {"formal": rec["formal"], "reason": rec["formal_reason"],
                                        "missing": sorted(rec["missing"])}
        except Exception as exc:
            logger.error("[forward] 每日记录写入失败（不影响交易）: %s", exc)
            self._note_failure("forward_record", exc)

    def _archived_at(self, key: str, day) -> Optional[str]:
        """这份组合信号的存档时刻（同日重跑时是第一次存档的时刻）。"""
        ev = self._evidence()
        if ev is None:
            return None
        try:
            row = ev.archived_before(key, day, datetime(9999, 1, 1))
        except Exception as exc:
            logger.error("[forward] 读不到 %s 的存档时刻: %s", key, exc)
            self._note_failure("signal_archive_read", exc)
            return None
        return row["recorded_at"].isoformat() if row else None

    def _run_portfolio(self, dataset: Dict[str, pd.DataFrame], aum: Optional[float] = None,
                       forward_from=None, now: Optional[datetime] = None,
                       data_source: Optional[str] = None, identity: Optional[dict] = None) -> dict:
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

        # F06：每份带规格的策略（proposed / approved / active）各自存档信号、结算前向 IC ——
        # 与今天交不交易无关，所以放在一切提前返回之前。失败不影响交易。
        self._record_strategy_evidence(dataset, now, data_source, identity)

        # 组合成分 = 活跃因子
        recs = []
        for rec in self.store.query(limit=500):
            try:
                if coerce_status(rec.status) in (AlphaStatus.PAPER, AlphaStatus.ACTIVE, AlphaStatus.DECAYING):
                    recs.append(rec)
            except ValueError:
                logger.warning("[portfolio] alpha id=%s 状态 %r 无法解析，已排除出组合成分",
                               getattr(rec, "id", "?"), getattr(rec, "status", None))
                continue

        from app.core.portfolio_manager.executable_spec import signal_config
        cfg = signal_config()
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
            reason = "no active factors" if not recs else "no valid signals"
            return {"n_factors": 0, "days_processed": 0, "reason": reason,
                    "used_baseline": used_baseline,
                    "execution": self._maintain_live(reason)}

        from app.config import settings

        # ── PM.7 / 审计 F08：**active 策略配置 = 资金决策单位**。
        #
        # 原先只有"过滤后非空"才按配置交易；配置里的因子全部退役 / 取信号失败 / 读库异常时
        # 继续交易**其他** PAPER 因子 —— 被批准的是 A，交易的是 B。现在：
        #   · 有 active 配置 → 必须**全部**成分都可用，否则本轮不调仓（只对账/熔断维护）；
        #     用它审批时冻结的 combo_weights / method，不再每天重新拟合。
        #   · 读不到配置（库异常）→ 本轮不调仓。
        #   · 没有 active 配置 → 模拟账本照常（PAPER 因子或基准库），但**券商执行不加新风险**：
        #     下真单必须有一份被人批准并激活的策略版本（live_block_reason）。
        using_active_config = None
        contract = None                      # 复审 2026-10-19 N05-e：有 active 配置 → 照它的合同生成目标
        frozen_combo = None
        combo_method = "ic_weighted"
        spec_band: Optional[float] = None
        live_block_reason = ""
        if not used_baseline:
            try:
                import json as _json
                from app.db.strategy_store import StrategyStore
                active = StrategyStore().latest_running()
            except Exception as exc:
                logger.error("[portfolio] PM.7 读取 active 策略配置失败 → 本轮不调仓: %s", exc)
                return {"n_factors": 0, "days_processed": 0,
                        "reason": "active_config_unreadable", "used_baseline": used_baseline,
                        "execution": self._maintain_live("active_config_unreadable")}
            from app.core.lifecycle.strategy_states import label, permissions
            if active is not None and not permissions(active.status)["trades_simulate"]:
                # 2026-10-22 C6：旧的 `active` 字样不代表任何权限（未验证、多半没有规格）—— 不按它交易
                why = f"策略 #{active.id} 的状态是 {label(active.status)}，不能交易；请重新提案后开始纸交易"
                logger.error("[portfolio] %s → 本轮不调仓", why)
                return {"n_factors": 0, "days_processed": 0, "reason": "state_not_tradable", "detail": why,
                        "used_baseline": used_baseline, "execution": self._maintain_live("state_not_tradable")}
            if active is not None:
                cfg_factors = [str(f) for f in _json.loads(active.factors or "[]")]
                missing = [f for f in cfg_factors if f not in signals]
                cw = {str(k): float(v) for k, v in _json.loads(active.combo_weights or "{}").items()}
                if not cfg_factors or missing or (cw and set(cfg_factors) - set(cw)):
                    why = (f"active 策略 #{active.id} 不可用：成分 {cfg_factors}，"
                           f"缺信号 {missing}，冻结权重覆盖 {sorted(cw)}")
                    logger.error("[portfolio] %s → 本轮不调仓", why)
                    return {"n_factors": 0, "days_processed": 0,
                            "reason": "active_config_unusable", "detail": why,
                            "used_baseline": used_baseline,
                            "execution": self._maintain_live("active_config_unusable")}
                # 外部审查 2026-10-18 F07：照**批准时的可执行规格**交易，并核对运行时没变。
                from app.core.portfolio_manager.strategy_contract import StrategyContract
                contract = StrategyContract.from_record(active)
                if contract is None:
                    why = (f"active 策略 #{active.id} 没有可执行规格（2026-10-18 之前批准的配置，或规格被改过）"
                           f"—— 无法证明交易的是被批准的那份，请重新提案并审批")
                    logger.error("[portfolio] %s → 本轮不调仓", why)
                    return {"n_factors": 0, "days_processed": 0,
                            "reason": "active_spec_missing", "detail": why,
                            "used_baseline": used_baseline,
                            "execution": self._maintain_live("active_spec_missing")}
                dsl_of = {str(r.id): r.dsl for r in recs}
                diff = contract.verify({f: dsl_of.get(f) for f in cfg_factors}, aum,
                                       settings=self._runtime_settings(data_source))
                if diff:
                    why = f"active 策略 #{active.id} 批准后交易参数变了：{diff}"
                    logger.error("[portfolio] %s → 本轮不调仓，需要重新提案并审批", why)
                    return {"n_factors": 0, "days_processed": 0,
                            "reason": "active_spec_mismatch", "detail": why, "spec_diff": diff,
                            "used_baseline": used_baseline,
                            "execution": self._maintain_live("active_spec_mismatch")}
                # 复审 2026-10-19 §5.4：用到基本面字段 → 这份面板的基本面质量报告必须过门，否则不调仓
                from app.core.data_engine.fundamentals.quality import strategy_quality_problem
                fq = strategy_quality_problem(contract.factors, prices.index[-1])
                if fq:
                    why = f"active 策略 #{active.id} 用到基本面字段，质量门未过：{fq}"
                    logger.error("[portfolio] %s → 本轮不调仓", why)
                    return {"n_factors": 0, "days_processed": 0,
                            "reason": "fundamentals_quality", "detail": why,
                            "used_baseline": used_baseline,
                            "execution": self._maintain_live("fundamentals_quality")}
                # 信号按合同算：规格里的表达式 + 规格里的信号参数（verify 已确认与运行时一致）
                try:
                    signals = contract.signals(dataset)
                except Exception as exc:
                    why = f"active 策略 #{active.id} 按规格算信号失败：{exc}"
                    logger.error("[portfolio] %s → 本轮不调仓", why)
                    return {"n_factors": 0, "days_processed": 0,
                            "reason": "active_config_unusable", "detail": why,
                            "used_baseline": used_baseline,
                            "execution": self._maintain_live("active_config_unusable")}
                using_active_config = active.id
                frozen_combo = contract.combo_weights
                combo_method = contract.method
                spec_band = contract.band
                logger.info("[portfolio] PM.7 按 active 策略配置 #%d 交易，成分 %d 因子，权重%s",
                            active.id, len(signals), "冻结" if frozen_combo else "按 method 拟合")
        if using_active_config is None:
            live_block_reason = "no_active_strategy"

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
        # 组合账本**始终用独立的 broker 实例**。
        #
        # 原先是 `... if gp is not None else self.broker` —— 拿不到 grounded 成本参数
        # 时直接复用 `self.broker`，而那正是**逐 alpha 影子账本**用的同一个对象。
        # 组合级净敞口上限一旦设在它身上，就会泄漏到影子账本
        # （用户 2026-09-20 明确警告的"把整体限额简单逐 alpha 套用"）。
        # 这个泄漏是给上限接线时被 TestPortfolioNetLimitIsWired 当场抓到的。
        # 复审 2026-10-19 N05-e：同一套成本 / 容量参数给模拟账本、策略门与合成账本（此前合成账本用默认
        # CostParams()，门用 grounded —— 容量绑定时门验证的权重与这里交易的不是同一组）
        cost_params = gp if gp is not None else self.broker.params
        pf_broker = PaperBroker(
            store=self.broker.store,
            cost_params=cost_params,
            initial_capital=aum,
        )

        # ── PM.S1：对**组合策略**跑策略级验证门（分段 OOS + DSR 去膨胀 + 夏普 t），记录 verdict。
        #    严门加在真正交易的策略上、不加单因子。默认只记录不阻断（paper 期先收前向证据）。
        strategy_verdict = None
        # 审计 F09：硬门（block=True）下，**评估异常 / 结果缺失 / 未通过**都必须拦住新增风险；
        # 非实验模式下分级不允许（或分级本身失败）同样拦。原先异常被 except 吞掉后继续下单，
        # `_allowed` 只写进诊断从不使用 —— TR_EXPERIMENT_MODE=false 形同虚设。
        gate_block = bool(settings.pm_strategy_gate_block)
        experiment = bool(settings.tr_experiment_mode)
        halt_reason = ""
        if (settings.pm_strategy_gate_eval or gate_block) and not used_baseline:
            from app.core.portfolio_manager import StrategyGate
            try:
                # F07：有 active 配置时评估的是**被批准的那份**（冻结权重 + 冻结的带），不是重新拟合的 ic_weighted
                gate = StrategyGate(aum=aum, method=combo_method)
                sv = (contract.evaluate(gate, signals, dataset, cost_params=cost_params) if contract is not None
                      else gate.evaluate(signals, dataset, cost_params=cost_params))
            except Exception as exc:
                logger.error("[portfolio] PM.S1 策略门评估失败（无证据）: %s", exc)
                sv = None
                strategy_verdict = {"gate_error": str(exc), "evaluated": False}
                if gate_block:
                    halt_reason = "strategy_gate_error"
            if sv is not None:
                strategy_verdict = sv.to_dict()
                # 外部审查 2026-10-18 F03：每日门看的是**整个** PIT 窗口（含冻结 Test 段与前向新数据），
                # 它是运行监控，不是"未见样本外"的资格证明 —— 结论上必须带着这个口径。
                strategy_verdict["evidence_scope"] = "full_pit_window_incl_holdout_and_forward"
                strategy_verdict["holdout_clean"] = False
                logger.info("[portfolio] PM.S1 策略门：passed=%s Sharpe=%.3f DSR=%.3f t=%.2f | %s",
                            sv.passed, sv.sharpe, sv.deflated_sharpe, sv.t_stat,
                            "OK" if sv.passed else "; ".join(sv.reasons))
                if not sv.passed and gate_block:
                    halt_reason = "strategy_gate_failed"
            # TR.4 第 4 步：进 PAPER 的 A/B/C 分级（实验模式放行 B/C 但如实标注等级）
            try:
                from app.core.lifecycle.promotion_gate import grade_paper_entry
                _allowed, _g = grade_paper_entry(strategy_verdict)
                strategy_verdict["paper_grade"] = _g["grade"]
                strategy_verdict["paper_entry"] = _g
                logger.info("[portfolio] TR.4 进 PAPER 分级=%s（allowed=%s, 实验模式=%s）",
                            _g["grade"], _allowed, _g["experiment_mode"])
                if not _allowed and not halt_reason:
                    halt_reason = "paper_grade_not_allowed"
            except Exception as exc:
                logger.error("[portfolio] TR.4 分级失败: %s", exc)
                if not experiment and not halt_reason:
                    halt_reason = "paper_grading_error"
        if halt_reason:
            logger.warning("[portfolio] %s → 本轮不调仓（只对账/熔断维护）", halt_reason)
            return {"n_factors": len(signals), "days_processed": 0,
                    "reason": halt_reason, "strategy_verdict": strategy_verdict,
                    "selection": selection_info, "used_baseline": used_baseline,
                    "execution": self._maintain_live(halt_reason)}

        # ── PM.5 / PM.6：组合级风控 → 回撤熔断 → 无交易带 → 带后硬约束投影。
        # 复审 2026-10-19 N05-e：与策略门、提案**同一个**目标生成函数（`strategy_contract.generate_targets`）；
        # 有 active 配置时参数全部来自它的合同（风控上限、波动窗口、合成权重、带），不再现读 settings。
        # 【审计 F07】带后再投影一次并独立 check（带会取消小幅减仓、重新引入超限）；
        # 【F07 R7】带后投影不再传波动估计（同一缩放系数乘两遍）—— 都在 generate_targets 里。
        from app.core.portfolio_manager import (PortfolioRiskGate, annualized_turnover,
                                                horizon_profile)
        from app.core.portfolio_manager.executable_spec import risk_limits_from_settings
        from app.core.portfolio_manager.strategy_contract import generate_targets
        limits = contract.limits() if contract is not None else risk_limits_from_settings(settings)

        # 回撤熔断：按组合已有净值序列判定；触发且 risk_halt_on_drawdown → 本轮目标清零（风控之后、带之前）
        halt_info, halt_now = None, False
        try:
            hist = self.broker.store.pnl_history(PORTFOLIO_BOOK_ID, limit=1000)
            if hist:
                eq_series = pd.Series([h.equity for h in hist])
                halt, dd = PortfolioRiskGate(limits).should_halt(eq_series)
                halt_info = {"halt": bool(halt), "drawdown": round(float(dd), 4)}
                if halt and getattr(settings, "risk_halt_on_drawdown", False):
                    logger.warning("[portfolio] 回撤熔断 dd=%.3f → 本轮清仓 de-risk", dd)
                    halt_now = True
        except Exception as exc:
            logger.warning("[portfolio] 回撤熔断检查失败（不阻断）: %s", exc)

        # F07：有 active 配置 → 用批准时冻结的带（此前每天按 settings / 数据重新推导）
        band = spec_band if spec_band is not None else float(getattr(settings, "pm_no_trade_band", 0.0))
        if band <= 0.0 and spec_band is None:             # 由 TradingContext 数据推导（T2）
            try:
                from app.core.trading_context.context import TradingContext
                band = float(TradingContext(aum=aum).analyze(dataset).rebalance_band)
            except Exception as exc:
                # band=0 意味着**每天全额调仓**，换手与成本都会显著高于预期设定
                logger.error("[portfolio] 无交易带推导失败 → 本轮按 band=0（全额调仓）: %s", exc)
                band = 0.0
        try:
            if contract is not None:
                tg = contract.targets(signals, dataset, cost_params=cost_params, halt=halt_now)
            else:
                tg = generate_targets(signals, dataset, aum=aum, method=combo_method, limits=limits, band=band,
                                      cost_params=cost_params,
                                      vol_lookback=int(getattr(settings, "risk_vol_lookback", 60)),
                                      halt=halt_now)
        except Exception as exc:
            # 此前波动估计失败时照常交易（不缩放 = 比意图更大的风险）；现在与门一样 fail-closed
            logger.error("[portfolio] 目标生成失败（合成 / 风控 / 波动估计）→ 本轮不调仓: %s", exc)
            return {"n_factors": len(signals), "days_processed": 0,
                    "reason": "target_generation_failed", "detail": str(exc),
                    "strategy_verdict": strategy_verdict, "used_baseline": used_baseline,
                    "execution": self._maintain_live("target_generation_failed")}
        # 用户决定第 18 条：疑似未知公司行动 → 隔离、不加风险、待人对账。模拟账本与券商目标同一处收缩；
        # 券商侧另由盘前风控门按实际持仓拒绝加风险单。标记读不到 → 不知道谁该隔离 → 券商只对账不下单
        ca_quarantined, ca_problem = self._corporate_action_quarantine(dataset)
        if ca_problem:
            live_block_reason = live_block_reason or ca_problem
        if ca_quarantined:
            from app.core.data_engine.corporate_actions import no_new_risk
            tg.weights = no_new_risk(tg.weights, ca_quarantined)
            tg.gated = no_new_risk(tg.gated, ca_quarantined)
        weights, composite, risk_report = tg.weights, tg.composite, tg.risk_report
        sectors = None
        if "sector" in dataset:
            try:
                sectors = dataset["sector"].iloc[-1]
            except Exception as exc:
                logger.warning("[portfolio] 行业标签读取失败，行业集中度约束本轮不生效: %s", exc)
        if limits.target_vol_ann:
            if tg.port_vol_ann is None:
                logger.warning("[portfolio] PM.5 目标波动已配置(%.3f)但可用样本不足，本轮不缩放",
                               limits.target_vol_ann)
            else:
                logger.info("[portfolio] PM.5 目标波动 %.3f / 实际估计 %.3f", limits.target_vol_ann, tg.port_vol_ann)
        logger.info("[portfolio] PM.5 风控：%s", risk_report.to_dict())

        # 组合级净敞口上限只接到**组合账本的执行层**（用户决策 2026-09-20：影子账本是测量，不授权资金）
        pf_broker.max_net = float(limits.max_net)

        to_before = annualized_turnover(tg.gated)
        gated_last = tg.gated.iloc[-1].copy()            # 风控之后、无交易带之前 → 券商执行的目标
        post_band_violations = tg.post_band_violations
        if post_band_violations:
            logger.error("[portfolio] 无交易带 + 二次风控后仍违规：%s → 券商执行不加新风险",
                         post_band_violations)
            live_block_reason = live_block_reason or "post_band_risk_violation"
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
        bars = _bars(dataset, weights.index, cols)                        # 时钟 c2：开 / 高 / 低
        label = _label_ending(bars["open"])                              # 开盘(t−1) → 开盘(t)
        dates  = weights.index
        cal_years = max((dates[-1] - dates[0]).days / 365.25, 1 / 365.25)
        tdays = len(dates) / cal_years
        comp_arr = composite.reindex(columns=cols).to_numpy(dtype=float)
        fbook = self._forward_book("portfolio")

        last = pf_broker.store.last_pnl_date(PORTFOLIO_BOOK_ID)
        n_days = 0
        equity = pf_broker.store.latest_equity(PORTFOLIO_BOOK_ID)
        for t in range(len(dates)):
            d = dates[t]
            if last is not None and d.date() <= last:
                continue                                  # 幂等续跑
            pnl = pf_broker.step(PORTFOLIO_BOOK_ID, d, **_step_args(weights, prices_f, adv_df, vol_df, bars, t),
                                 tdays_per_year=tdays)
            if t > 1:
                # 时钟 c2（C3）：在 t 结算信号日 T = t−2，标签 = 开盘(T+1) → 开盘(T+2)；
                # 前向 = 信号在开盘(T+1)（标签起点）之前就已存档；否则按当下重算的信号记回放（F06）
                arch = fbook.signal_for(dates[t - 2], dates[t - 1])
                sig = (arch["signal"].reindex(cols).to_numpy(dtype=float) if arch is not None
                       else comp_arr[t - 2])
                ic = _cs_spearman(sig, label.iloc[t].to_numpy(dtype=float))
                if not np.isnan(ic):
                    self.store.record_ic(PORTFOLIO_BOOK_ID, d, float(ic),
                                         realized_return=float(pnl.net_ret),
                                         is_forward=arch is not None)
            equity = pnl.equity
            n_days += 1
        fbook.archive(dates[-1], composite.reindex(columns=cols).iloc[-1], now)
        signal_archived_at = self._archived_at("portfolio", dates[-1])

        # ── Phase 12：执行层 —— 风控之后、无交易带之前的**最后一行**目标权重下成 moomoo 纸交易单；
        #    无交易带改在订单层按**券商实际持仓**判断（审计 F07：模拟账本重建的历史目标
        #    不等于账户里真正拿着的股数）。没有被批准的 active 策略时只对账/维护（审计 F08）。
        if live_block_reason:
            execution, t3_live = self._maintain_live(live_block_reason), None
        else:
            sector_map = ({} if sectors is None else
                          {str(k): v for k, v in sectors.items()})
            execution, t3_live = self._execute_live(
                gated_last, weights.index[-1], prices_f, adv_df,
                sectors=sector_map, no_trade_band=band, quarantined=set(ca_quarantined))

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
        if t3_live is not None:
            t3_state = t3_live              # 执行层在线时以券商实时账户为准（mode=live）

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

        # ── Phase R.2：风险归因（只读诊断，不改任何交易决策）──────────────
        #    PM.5 的 gross/net/单票上限回答不了"风险从哪来"：一篮子彼此不相关的
        #    股票和一篮子同时押在"低波动"上的股票，敞口可以一样、真实风险差一个量级。
        risk_attribution = None
        try:
            from app.core.risk_engine import fit_risk_model
            _rm = fit_risk_model(
                dataset, lookback=int(getattr(settings, "risk_attr_lookback", 252)))
            _attr = _rm.attribute(weights.iloc[-1])
            risk_attribution = _attr.to_dict()
            logger.info("[portfolio] R.2 风险归因 | 年化波动=%.2f%% | 因子占比=%.0f%% "
                        "| 行业哑变量=%s | 最大暴露=%s",
                        _attr.total_vol_ann * 100, _attr.factor_share * 100,
                        _attr.sector_included,
                        max(_attr.exposures.items(), key=lambda kv: abs(kv[1]),
                            default=("n/a", 0.0)))
        except Exception as exc:
            # 归因失败不阻断交易，但**必须留痕**：没有它，面板上"风险构成"那一块
            # 会是空的，而使用者无从判断是"没算"还是"算出来就是这样"。
            logger.warning("[portfolio] R.2 风险归因失败（不阻断交易）: %s", exc)

        logger.info("[portfolio] 组合账本 | AUM=%.0f | %d 因子 | %d 交易日 | 末净值=%.4f",
                    aum, len(signals), n_days, equity)
        result = {"n_factors": len(signals), "days_processed": n_days,
                "equity": equity, "aum": aum, "combo_weights": tg.combo_weights,
                "spec_hash": contract.hash if contract is not None else None,   # N05-e：照哪份合同交易的
                "signal_archived_at": signal_archived_at,  # §7.3 组合信号的存档时刻
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
                "risk_attribution": risk_attribution,  # R.2 风险从哪来（因子 vs 特异）
                "t3": t3_state,                        # TR.3 T3 providers(模式/买入力/持仓数)
                "execution": execution,                # Phase 12 执行层（订单/对账/风控/全平）
                # 送给券商的目标（风控后、无交易带前的决策日那一行）—— 留痕，可复核
                "execution_target": {str(k): float(v) for k, v in gated_last.items()
                                     if np.isfinite(v) and abs(v) > 0},
                "post_band_violations": post_band_violations,
                # 第 18 条：仍在隔离中的疑似公司行动 {标的: 标记日}（不加风险；基准按 0 收益留在分母）
                "ca_quarantined": sorted(ca_quarantined),
                "ca_flags": dict(ca_quarantined),
                "trading_context": tc_summary}         # TR.1 交易现实(价差/可交易/可做空/带)

        # FE-TR 前置：把本轮诊断只增不改地存下来（失败绝不影响交易）
        try:
            from app.db.diagnostics_store import DiagnosticsStore
            DiagnosticsStore().save(result)
        except Exception as exc:
            logger.warning("[portfolio] 诊断持久化失败（不阻断）: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Phase 12：执行层
    # ------------------------------------------------------------------

    def _open_live_gateway(self):
        """返回 (gateway, None) 或 (None, 不下单的原因报告)。"""
        from app.config import settings
        mode = str(getattr(settings, "execution_mode", "off"))
        if mode == "off":
            return None, {"mode": "off"}
        if mode != "moomoo_paper":
            logger.error("[portfolio] 未知的 execution_mode=%r → 本轮不下单", mode)
            return None, {"mode": mode, "blocked": "unknown_execution_mode"}
        from app.core.execution.broker_gateway import make_gateway_from_settings
        try:
            return (self._gateway_factory or make_gateway_from_settings)(), None
        except Exception as exc:
            logger.error("[portfolio] Phase 12 连不上券商 → 本轮不下单: %s", exc)
            return None, {"mode": mode, "blocked": f"gateway_unavailable: {exc}"}

    def _maintain_live(self, reason: str) -> dict:
        """不调仓的日子：只对账 + 熔断开启时续做全平（见 OrderManager.maintain）。"""
        gw, early = self._open_live_gateway()
        if gw is None:
            return early
        from app.core.execution.order_manager import OrderManager
        try:
            return OrderManager.from_settings(
                gw, store=self._execution_store, position_store=self.broker.store,
            ).maintain(reason).to_dict()
        except Exception as exc:
            logger.error("[portfolio] Phase 12 维护周期失败: %s", exc)
            return {"mode": "moomoo_paper", "blocked": f"execution_error: {exc}"}
        finally:
            gw.close()

    @staticmethod
    def _corporate_action_quarantine(dataset: Dict[str, pd.DataFrame]):
        """
        (隔离中的标的 {ticker: 标记日}, 问题)。先识别最近几根 bar 的疑似公司行动并记标记，再读出所有未结案的。
        标记库读写失败 → ({}, 原因)：调用方让券商只对账不下单（不知道谁该隔离时不能照常加风险）。
        """
        from app.core.data_engine.corporate_actions import detect_suspects
        try:
            from app.db.corporate_actions import CorporateActionStore
            store = CorporateActionStore()
            for s in detect_suspects(dataset.get("close"), dataset.get("volume")):
                store.flag(s["ticker"], s["bar_date"], kind="suspected_price_jump", ratio=s["ratio"],
                           nearest_split=s["nearest_split"])
            return store.open_tickers(), ""
        except Exception as exc:
            logger.error("[portfolio] 公司行动隔离标记读写失败 → 券商本轮只对账不下单: %s", exc)
            return {}, f"corporate_action_store_unavailable: {exc}"

    def _execute_live(self, target_last: pd.Series, decision_ts, prices_f: pd.DataFrame,
                      adv_df: pd.DataFrame, *, sectors=None, no_trade_band: float = 0.0,
                      quarantined=None):
        """
        返回 (执行报告, 实时 T3 状态或 None)。任何失败都是**不下单**，并把原因写进报告：
        执行层出错时继续下单才是危险方向（§U），而模拟账本已经照常记完了。

        target_last : 决策日（decision_ts）风控之后的目标权重（一行）。
        """
        from app.config import settings
        gw, early = self._open_live_gateway()
        if gw is None:
            return early, None
        from app.core.execution.order_manager import OrderManager
        mode = "moomoo_paper"

        t3_live = None
        try:
            try:
                rep = OrderManager.from_settings(
                    gw, store=self._execution_store, position_store=self.broker.store,
                ).run_cycle(
                    target_weights=target_last, ref_prices=prices_f.loc[decision_ts],
                    adv_usd=adv_df.loc[decision_ts], decision_date=pd.Timestamp(decision_ts).date(),
                    sectors=sectors, no_trade_band=no_trade_band, quarantined=quarantined or ())
                execution = rep.to_dict()
            except Exception as exc:
                logger.error("[portfolio] Phase 12 执行周期失败 → 本轮不下单: %s", exc)
                execution = {"mode": mode, "blocked": f"execution_error: {exc}"}
            try:
                from app.core.trading_context import get_trade_providers
                from app.core.trading_context.providers import moomoo_snapshot_fn
                tp = get_trade_providers(
                    "live", gateway=gw,
                    snapshot_fn=moomoo_snapshot_fn(settings.moomoo_host, settings.moomoo_port),
                    allow_short=bool(settings.trading_allow_short))
                t3_live = tp.to_dict()
            except Exception as exc:
                logger.warning("[portfolio] TR.3 live providers 读取失败（退回仿真 t3 展示）: %s", exc)
        finally:
            gw.close()
        return execution, t3_live

    # ------------------------------------------------------------------
    # 单因子（隔离）
    # ------------------------------------------------------------------

    #: 每份策略每次最多回看这么多根 bar 去结算（停机更久的那几天本来也不是前向）
    STRATEGY_SETTLE_LOOKBACK = 10

    @staticmethod
    def _runtime_settings(data_source: Optional[str]):
        """运行时口径：settings，但价格源换成**这份面板实际的来源**（PIT 元数据），而不是 settings 声称的。"""
        from app.config import settings
        if not data_source:
            return settings
        return settings.model_copy(update={"price_source": str(data_source)})

    def _record_strategy_evidence(self, dataset: Dict[str, pd.DataFrame],
                                  now: Optional[datetime],
                                  data_source: Optional[str] = None,
                                  identity: Optional[dict] = None) -> Dict[str, int]:
        """
        外部审查 2026-10-18 F06：**按策略版本**积累前向证据。

        每份带可执行规格的配置（proposed / approved / active）用**规格里的**成分表达式与冻结权重
        算合成信号：今天的存档；对最近几根 bar，若前一根的信号在当天收盘前已存档，就结算一条
        前向 IC，记在这份策略（id + 规格哈希）名下。于是：→ACTIVE 门看的是这份策略自己的前向记录，
        不再借 book 0 里别的策略的数；批准之前也在积累（不再"要有 60 天证据才能开始拿证据"）。
        """
        out: Dict[str, int] = {}
        self._set_identity(dataset, data_source, identity)      # 先于一切提前返回：后面的账本都用它
        ev = self._evidence()
        if ev is None:
            return out
        try:
            import json as _json
            from app.core.alpha_engine.dsl_executor import Executor
            from app.core.alpha_engine.signal_processor import SignalProcessor
            from app.core.portfolio_manager.strategy_contract import StrategyContract
            from app.db.strategy_store import StrategyStore
            prices = dataset["close"]
            label_all = _label_ending(_bars(dataset, prices.index, prices.columns)["open"])
            configs = [r for r in StrategyStore().query(limit=200)
                       if str(r.status) in ("proposed", "approved", "integration", "paper_running",
                                            "verified_active")]
        except Exception as exc:
            logger.error("[evidence] 策略版本证据本轮跳过（读不到配置）: %s", exc)
            self._note_failure("strategy_evidence", exc)
            return out
        rt = self._runtime_settings(data_source)
        for rec in configs:
            contract = StrategyContract.from_record(rec)            # N05-e：证据也按这份策略的合同生成
            if contract is None or not contract.complete():
                continue
            # 复审 2026-10-19 N02：证据路径也要核对规格 —— 运行时与规格对不上（换了价格源、风控、AUM…），
            # 算出来的就不是这份策略的信号，不能记在它的名下。隔离留痕，不计入正式前向。
            diff = contract.verify(contract.factors, float(self.broker.initial_capital), settings=rt)
            if diff:
                logger.error("[evidence] 策略 #%s 的规格与运行时不一致 → 本轮不记前向证据：%s", rec.id, diff)
                out[f"strategy:{rec.id}"] = -1
                try:
                    ev.log_revision("strategy_evidence_quarantine", f"strategy:{rec.id}",
                                    dataset["close"].index[-1], "", "; ".join(diff)[:500],
                                    "运行时与可执行规格不一致，本轮信号不存档、不结算")
                except Exception as exc:
                    logger.error("[evidence] 隔离记录写入失败: %s", exc)
                    self._note_failure(f"evidence_quarantine:{rec.id}", exc)
                continue
            try:
                composite = contract.composite(contract.signals(dataset), dataset)
                key = f"strategy:{rec.id}"
                book = self._forward_book(key)
                h = _json.loads(rec.verdict or "{}").get("spec_hash", "")
                dates = composite.index
                n = 0
                cols = composite.columns
                for t in range(max(2, len(dates) - self.STRATEGY_SETTLE_LOOKBACK), len(dates)):
                    # 时钟 c2（C3）：在 t 结算信号日 T = t−2；标签 = 开盘(T+1) → 开盘(T+2)；归档截止 = 开盘(T+1)
                    arch = book.signal_for(dates[t - 2], dates[t - 1])
                    if arch is not None and arch["spec_hash"] != h:
                        continue                          # 那份存档属于另一个规格版本
                    realized = label_all.reindex(columns=cols).iloc[t].to_numpy(dtype=float)
                    if arch is not None:                  # 前向：标签开始前就存档的那份信号
                        ic = _cs_spearman(arch["signal"].reindex(cols).to_numpy(dtype=float), realized)
                        if not np.isnan(ic) and ev.record_strategy_ic(
                                rec.id, h, dates[t], ic, dates[t - 2], arch["recorded_at"],
                                identity=getattr(self, "_identity", None),
                                signal_digest=arch.get("data_digest", ""),
                                # 2026-10-22 C6：工程联调期的前向证据单独归类，不计入验证
                                evidence_class=("integration" if str(rec.status) == "integration"
                                                else "forward")):
                            n += 1
                        continue
                    # 复审 2026-10-19 N03-b：没有及时存档（首跑 / 停机补拉 / 晚到 / 换源）→ 按当下重算的信号
                    # 记一条 **replay**：保留研究价值，不进 →ACTIVE 门（strategy_forward_ic 不读它）
                    ic = _cs_spearman(composite.iloc[t - 2].to_numpy(dtype=float), realized)
                    if not np.isnan(ic):
                        ev.record_strategy_ic(rec.id, h, dates[t], ic, dates[t - 2],
                                              now or datetime.utcnow(), identity=getattr(self, "_identity", None),
                                              evidence_class="replay")
                book.archive(dates[-1], composite.iloc[-1], now, spec_hash=h)
                out[key] = n
            except Exception as exc:
                logger.error("[evidence] 策略 #%s 的前向证据本轮失败（不影响交易）: %s", rec.id, exc)
                self._note_failure(f"strategy_evidence:{rec.id}", exc)
        return out

    def _run_one_alpha(self, rec, dataset: Dict[str, pd.DataFrame],
                       forward_from=None, now: Optional[datetime] = None) -> AlphaDayResult:
        res = AlphaDayResult(alpha_id=rec.id)
        try:
            from app.core.alpha_engine.dsl_executor import Executor
            from app.core.alpha_engine.signal_processor import SignalProcessor, SimulationConfig
            from app.core.backtest_engine.portfolio_constructor import SignalWeightedPortfolio

            prices = dataset["close"]
            volume = dataset.get("volume")
            if volume is None:
                volume = pd.DataFrame(1e6, index=prices.index, columns=prices.columns)

            # 信号 → 权重（与规格同一份信号参数；时钟 c2：T 收盘决策、T+1 执行）
            from app.core.portfolio_manager.executable_spec import signal_config
            raw_signal  = Executor(validate=False).run_expr(rec.dsl, dataset)
            proc_signal = SignalProcessor(signal_config()).process(raw_signal)
            weights     = SignalWeightedPortfolio(clip_z=3.0).construct(proc_signal)

            # 市场上下文（全窗口，与 PaperBroker.replay 口径一致）
            prices_f = prices.reindex(columns=weights.columns).ffill(limit=5).fillna(0.0)
            volume_f = volume.reindex(columns=weights.columns).ffill(limit=5).fillna(0.0)
            adv_df = self.broker._liq.compute_adv(volume_f, prices_f)      # noqa: SLF001
            vol_df = prices_f.pct_change().rolling(20, min_periods=2).std().fillna(0.02)
            bars = _bars(dataset, weights.index, weights.columns)
            label = _label_ending(bars["open"])
            dates  = weights.index
            cal_years = max((dates[-1] - dates[0]).days / 365.25, 1/365.25)
            tdays = len(dates) / cal_years

            # 续跑点：last_pnl_date 之后
            last = self.broker.store.last_pnl_date(rec.id)
            raw_arr = raw_signal.reindex(columns=weights.columns).to_numpy(dtype=float)
            fbook = self._forward_book(f"alpha:{rec.id}")

            for t in range(len(dates)):
                d = dates[t]
                if last is not None and d.date() <= last:
                    continue                                  # 已处理（幂等续跑）
                # 模拟成交（时钟 c2：今天开盘撮合昨天收盘的目标）
                pnl = self.broker.step(rec.id, d, **_step_args(weights, prices_f, adv_df, vol_df, bars, t),
                                       tdays_per_year=tdays)
                # 预测 IC（时钟 c2，与组合账本同一口径）：信号日 T = t−2 对 开盘(T+1) → 开盘(T+2)
                if t > 1:
                    arch = fbook.signal_for(dates[t - 2], dates[t - 1])
                    sig = (arch["signal"].reindex(weights.columns).to_numpy(dtype=float)
                           if arch is not None else raw_arr[t - 2])
                    ic = _cs_spearman(sig, label.iloc[t].to_numpy(dtype=float))
                    if not np.isnan(ic):
                        self.monitor.update(rec.id, d, float(ic),
                                            realized_return=float(pnl.net_ret),
                                            is_forward=arch is not None)
                res.days_processed += 1
                res.last_date = str(d.date())
                res.equity = pnl.equity
            if len(dates):
                fbook.archive(dates[-1], pd.Series(raw_arr[-1], index=weights.columns), now)

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


def _bars(dataset: Dict[str, pd.DataFrame], index, columns) -> Dict[str, pd.DataFrame]:
    """开 / 高 / 低（时钟 c2 撮合与标签用）。缺任何一项 → ValueError：不能拿收盘价凑开盘。"""
    out = {}
    for k in ("open", "high", "low"):
        df = dataset.get(k)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"时钟 c2 需要 {k}（T+1 按限价规则成交、开盘到开盘的标签）；数据集里没有")
        out[k] = df.reindex(index=index, columns=columns)
    return out


def _label_ending(open_df: pd.DataFrame) -> pd.DataFrame:
    """第 t 行 = 在 t **实现完**的标签：开盘(t−1) → 开盘(t)，属于信号日 t−2（见 app/core/clock.py）。"""
    o = open_df.astype(float)
    prev = o.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return o / prev.where(prev != 0) - 1.0


def _step_args(weights: pd.DataFrame, prices_f: pd.DataFrame, adv_df: pd.DataFrame, vol_df: pd.DataFrame,
               bars: Dict[str, pd.DataFrame], t: int) -> Dict[str, object]:
    """
    PaperBroker.step 第 t 天的参数（时钟 c2）：撮合**昨天收盘**的目标；成交量上限与波动用截至昨天的；
    开 / 高 / 低用今天的。第一天只决策、不成交（target_w=None）。
    """
    prev = t - 1 if t > 0 else t
    return dict(target_w=weights.iloc[t - 1] if t > 0 else None, prices_t=prices_f.iloc[t],
                prices_prev=prices_f.iloc[prev], adv_usd=adv_df.iloc[prev], daily_vol=vol_df.iloc[prev],
                open_t=bars["open"].iloc[t], high_t=bars["high"].iloc[t], low_t=bars["low"].iloc[t])


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """
    平均秩（并列取均值）—— Spearman 的**定义**要求的做法。

    这里曾经有一份独立实现。2026-09-20 修缺陷 C-1 时发现同一个概念在代码库里
    存在**五份**：这一份是对的，另外四份（GP 适应度 / alpha_combiner /
    evaluation_utils / alpha_workflows）用的是 `argsort(argsort(x))` 序数名次，
    会把截面恒定的零信息信号排成 [0,1,2,...] 并算出按列顺序的伪 IC。
    现已统一到 `fast_ops.average_ranks_1d`，此处转为薄封装 ——
    **同一个概念只保留一份实现**，否则修好四份仍会有第五份在犯错。
    """
    return average_ranks_1d(x)


def _cs_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """
    截面 Spearman rank 相关；有效对 < 3、或任一侧无秩差（全并列）时返回 NaN。

    "无区分度的信号得不出 IC" 是**必须**的：否则常数信号会被判为完美预测。
    """
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return float("nan")
    ra = _average_ranks(np.asarray(a[mask], dtype=float))
    rb = _average_ranks(np.asarray(b[mask], dtype=float))
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float(np.dot(ra, rb) / denom) if denom > 0 else float("nan")
