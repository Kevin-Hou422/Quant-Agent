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


class DailyIngest:
    def __init__(self, min_health: float = 0.7) -> None:
        self.min_health = min_health

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


def run_daily_pipeline(dataset_name: str, start: str, end: str) -> Dict[str, Any]:
    """
    完整每日管线：摄取 → 健康门 → （通过则）交易循环。供调度器调用。
    数据坏 → 跳过循环 + 返回拒绝原因（A2：绝不用坏数据继续）。
    """
    ing = DailyIngest().ingest(dataset_name, start, end)
    if not ing.accepted:
        logger.warning("[daily_pipeline] 摄取被拒（%s）→ 跳过当日交易循环", ing.reject_reason)
        return {"ingest_accepted": False, "reject_reason": ing.reject_reason, "as_of": ing.as_of}

    from app.tasks.daily_trading_loop import DailyTradingLoop
    loop = DailyTradingLoop()
    report = loop.run(ing.dataset)                       # per-factor：realized IC 监控 + 衰减
    # Phase PM.4：组合账本（真实 AUM 的美元账本，多因子净持仓 + 容量）
    try:
        pf = loop.run_portfolio(ing.dataset)
    except Exception as exc:                             # 组合账本失败不拖垮 per-factor 监控
        logger.warning("[daily_pipeline] 组合账本失败（不阻断）: %s", exc)
        pf = {"n_factors": 0, "days_processed": 0, "error": str(exc)}
    return {
        "ingest_accepted": True, "as_of": ing.as_of, "health_score": ing.health_score,
        "n_alphas": report.n_alphas, "n_alerts": report.n_alerts, "n_errors": report.n_errors,
        "portfolio": pf,
    }
