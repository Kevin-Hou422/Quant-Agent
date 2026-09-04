"""
daily_ingest.py — 每日数据摄取 + 健康验收门（Task 7.1，2026-08-02）

职责：收盘后拉取数据 → **健康验收门** → 通过则交给每日交易循环；不通过则
**拒绝当日调仓并告警，绝不静默降级到坏/过期数据**（验收标准 A2）。

设计：
  - 复用 `load_registry_dataset`（真实数据路径）+ `check_dataset_health`（gap/spike/
    NaN → 综合健康分）。
  - 健康门：health_score < min_score → `IngestResult(accepted=False, ...)` + WARNING 告警，
    调用方据此**跳过当日循环**。绝不因数据坏而退合成数据。
  - as_of 时间戳：记录本次摄取的观测时点（PIT 语义的前置；完整 PIT 追加存储见 Phase 8.1）。

调度：挂到 Phase 5.3 调度器（收盘后触发），成功后调用 `DailyTradingLoop.run`。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    accepted:     bool
    dataset_name: str
    as_of:        str
    health_score: float = 0.0
    n_dates:      int = 0
    n_tickers:    int = 0
    reject_reason: str = ""
    dataset:      Optional[Dict[str, pd.DataFrame]] = None
    # ── Phase 11 增量摄取 ────────────────────────────────────────────────
    mode:          str = "full"        # full(首次回填) | incremental(前向增量) | no_new_bar
    forward_from:  Optional[str] = None   # 本次新增的第一根 bar 日期 → 之后的 IC 算"真前向"
    n_new_bars:    int = 0
    calendar_note: str = ""            # 数据↔日历 交叉校验说明


class DailyIngest:
    def __init__(self, min_health: float = 0.7) -> None:
        self.min_health = min_health

    # ------------------------------------------------------------------
    # Phase 11.1 / 11.3：真增量摄取（不再整段重拉）
    # ------------------------------------------------------------------

    def ingest_incremental(self, dataset_name: str,
                           backfill_start: Optional[str] = None) -> IngestResult:
        """
        **前向增量**摄取（Phase 11 核心）：
          1. 从 PIT 读该数据集**已有的最新 bar 日期**；
          2. 空库 → 一次性**历史回填**（mode=full，这些 IC 是"回放"，非前向）；
             非空 → 只拉 `last+1 .. 今天` 的**增量**（mode=incremental）；
          3. 无新 bar → `no_new_bar`，不写库、不交易（周末/节假日/尚未收盘的自然结果）；
          4. 增量过健康门后**只追加增量**进 PIT（旧实现每天重写整段，去重键含 as_of → 1000× 膨胀）；
          5. 与交易日历**交叉校验**并记录说明（数据仍是权威，只是让不一致被看见）。

        返回的 `dataset` 是**从 PIT 读出的完整面板**（回填+历次增量），供交易循环消费；
        `forward_from` 是本次新增的第一根 bar 日期，交易循环据此把之后的 IC 标为真前向。
        """
        from app.config import settings
        from app.core.data_engine.pit_store import PITStore
        from app.core.data_engine.market_calendar import cross_check, last_trading_day

        as_of = datetime.now(timezone.utc).isoformat(timespec="seconds")
        store = PITStore(settings.pit_store_dir)
        last = store.latest_timestamp(dataset_name)

        today = pd.Timestamp.utcnow().tz_localize(None).normalize()
        target = last_trading_day(today)          # 最近应有数据的交易日

        # ---- 1) 空库 → 历史回填（回放种子） ----
        if last is None:
            start = backfill_start or settings.paper_start
            res = self.ingest(dataset_name, start, today.strftime("%Y-%m-%d"))
            res.mode = "full"
            res.forward_from = None               # 回填全部算回放，不是前向证据
            res.n_new_bars = res.n_dates
            logger.info("[daily_ingest] PIT 为空 → 历史回填 %s..%s（%d 日，计为回放）",
                        start, today.date(), res.n_dates)
            return res

        # ---- 2) 已有数据 → 只拉增量 ----
        nxt = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if last.normalize() >= target.normalize():
            note = f"PIT 最新 bar {last.date()} 已是最近交易日 {target.date()}"
            logger.info("[daily_ingest] 无新 bar（%s）→ 跳过", note)
            return IngestResult(False, dataset_name, as_of, mode="no_new_bar",
                                reject_reason="no_new_bar", calendar_note=note)

        inc = self.ingest(dataset_name, nxt, today.strftime("%Y-%m-%d"))
        if not inc.accepted:
            inc.mode = "incremental"
            return inc

        inc_close = (inc.dataset or {}).get("close")
        new_dates = [d for d in (inc_close.index if inc_close is not None else [])
                     if pd.Timestamp(d).normalize() > last.normalize()]
        ok, note = cross_check(target, has_bar=bool(new_dates))
        if not new_dates:
            logger.info("[daily_ingest] 增量窗口无新 bar → 跳过（%s）", note)
            return IngestResult(False, dataset_name, as_of, mode="no_new_bar",
                                reject_reason="no_new_bar", calendar_note=note)

        # ---- 3) 只把**增量**写进 PIT（而非整段重写） ----
        first_new = pd.Timestamp(min(new_dates)).normalize()
        increment = {f: df.loc[df.index > last] for f, df in (inc.dataset or {}).items()
                     if isinstance(df, pd.DataFrame)}
        try:
            self._append_pit(dataset_name, increment, as_of)
        except Exception as exc:
            logger.error("[daily_ingest] 增量写 PIT 失败: %s", exc)
            return IngestResult(False, dataset_name, as_of, mode="incremental",
                                reject_reason=f"pit_append_failed: {exc}")

        # ---- 4) 交易循环消费"回填+全部增量"的完整面板（从 PIT 读） ----
        panel = store.load_pit(name=dataset_name) or {}
        full = panel if panel.get("close") is not None else (inc.dataset or {})
        close = full.get("close")
        logger.info("[daily_ingest] 增量摄取：新增 %d 根 bar（%s..%s），PIT 面板 %s",
                    len(new_dates), first_new.date(), max(new_dates).date(),
                    None if close is None else close.shape)
        return IngestResult(
            True, dataset_name, as_of, health_score=inc.health_score,
            n_dates=0 if close is None else len(close),
            n_tickers=0 if close is None else close.shape[1],
            dataset=full, mode="incremental",
            forward_from=first_new.strftime("%Y-%m-%d"),
            n_new_bars=len(new_dates), calendar_note=note,
        )

    def ingest(
        self,
        dataset_name: str,
        start:        str,
        end:          str,
    ) -> IngestResult:
        """
        拉取 + 健康验收。返回 IngestResult；accepted=False 时调用方应跳过当日循环并告警。
        任何加载/校验异常都视为**拒绝**（不静默降级）。
        """
        as_of = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            from app.core.data_engine.dataset_registry import (
                load_registry_dataset, check_dataset_health,
            )
            ds = load_registry_dataset(dataset_name, start=start, end=end, health_check=False)
        except Exception as exc:
            logger.error("[daily_ingest] 数据加载失败 '%s': %s — 拒绝当日摄取", dataset_name, exc)
            return IngestResult(False, dataset_name, as_of, reject_reason=f"load_failed: {exc}")

        data = ds.data
        close = data.get("close")
        if close is None or close.empty:
            return IngestResult(False, dataset_name, as_of, reject_reason="empty_close")

        # 健康门
        score = 1.0
        try:
            report = check_dataset_health(ds, min_score=self.min_health, warn_only=True)
            if report is not None:
                score = float(report.overall_score)
        except Exception as exc:
            logger.warning("[daily_ingest] 健康检查异常，保守拒绝: %s", exc)
            return IngestResult(False, dataset_name, as_of, reject_reason=f"health_error: {exc}")

        if score < self.min_health:
            logger.warning(
                "[daily_ingest] 数据健康分 %.3f < 阈值 %.2f — **拒绝当日摄取**（不静默降级）",
                score, self.min_health,
            )
            return IngestResult(
                False, dataset_name, as_of, health_score=score,
                n_dates=close.shape[0], n_tickers=close.shape[1],
                reject_reason=f"health_below_threshold({score:.3f}<{self.min_health})",
            )

        # Task 8.1：通过验收的数据按 as_of 追加进 PIT 存储（历史只追加不修改）。
        # PIT 写入失败**不阻塞**当日交易循环（存储是审计/复现资产，非交易前置），仅告警。
        try:
            self._append_pit(dataset_name, data, as_of)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[daily_ingest] PIT 追加失败（不阻塞交易循环）: %s", exc)

        logger.info(
            "[daily_ingest] 摄取通过 '%s' | 健康=%.3f | %d 天 × %d 标的 | as_of=%s",
            dataset_name, score, close.shape[0], close.shape[1], as_of,
        )
        return IngestResult(
            True, dataset_name, as_of, health_score=score,
            n_dates=close.shape[0], n_tickers=close.shape[1], dataset=data,
        )

    @staticmethod
    def _append_pit(dataset_name: str, data: Dict[str, pd.DataFrame], as_of: str) -> None:
        """把通过验收的 wide 面板按 as_of 追加进 PIT 存储（Task 8.1）。"""
        from app.config import settings
        from app.core.data_engine.pit_store import PITStore

        store = PITStore(settings.pit_store_dir)
        store.append(data, as_of=as_of, name=dataset_name)


def run_daily_pipeline(dataset_name: str, start: str = "", end: str = "",
                       incremental: bool = True) -> Dict[str, Any]:
    """
    完整每日管线：摄取 → 健康门 → （通过则）交易循环。供调度器调用。
    数据坏 → 跳过循环 + 返回拒绝原因（A2：绝不用坏数据继续）。

    Phase 11：默认 **incremental=True** —— 前向增量（PIT 空则先历史回填，之后每天只拉新 bar；
    无新 bar 就跳过，不空转）。`incremental=False` 保留旧的整段重拉（仅供回测/排查用）。
    `forward_from` 会传给交易循环，把新 bar 之后的 IC 标为**真前向**（TR.4 →ACTIVE 门只认这些）。
    """
    if incremental:
        ing = DailyIngest().ingest_incremental(dataset_name, backfill_start=start or None)
    else:
        ing = DailyIngest().ingest(dataset_name, start, end)

    if not ing.accepted:
        if ing.mode == "no_new_bar":
            logger.info("[daily_pipeline] 无新 bar → 今日不交易（%s）", ing.calendar_note)
            return {"ingest_accepted": False, "reason": "no_new_bar",
                    "mode": ing.mode, "calendar_note": ing.calendar_note, "as_of": ing.as_of}
        logger.warning("[daily_pipeline] 摄取被拒（%s）→ 跳过当日交易循环", ing.reject_reason)
        return {"ingest_accepted": False, "reject_reason": ing.reject_reason,
                "mode": ing.mode, "as_of": ing.as_of}

    from app.tasks.daily_trading_loop import DailyTradingLoop
    loop = DailyTradingLoop()
    report = loop.run(ing.dataset)                       # per-factor：realized IC 监控 + 衰减
    # Phase PM.4：组合账本（真实 AUM 的美元账本，多因子净持仓 + 容量）
    try:
        pf = loop.run_portfolio(ing.dataset, forward_from=ing.forward_from)
    except Exception as exc:                             # 组合账本失败不拖垮 per-factor 监控
        logger.warning("[daily_pipeline] 组合账本失败（不阻断）: %s", exc)
        pf = {"n_factors": 0, "days_processed": 0, "error": str(exc)}
    return {
        "ingest_accepted": True, "as_of": ing.as_of, "health_score": ing.health_score,
        "mode": ing.mode, "forward_from": ing.forward_from, "n_new_bars": ing.n_new_bars,
        "calendar_note": ing.calendar_note,
        "n_alphas": report.n_alphas, "n_alerts": report.n_alerts, "n_errors": report.n_errors,
        "portfolio": pf,
    }
