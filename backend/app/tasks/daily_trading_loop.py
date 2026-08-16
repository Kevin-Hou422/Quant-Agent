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
        self.store   = store or AlphaStore()
        self.broker  = broker or PaperBroker(store=self._shared_position_store())
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
