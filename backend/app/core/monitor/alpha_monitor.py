"""
alpha_monitor.py — 因子在线监控与衰减检测（Task 5.1）

职责：
  - update()      : 记录某因子某日 realized IC（幂等，委托 AlphaStore.record_ic）
  - check_decay() : 滚动 IC 均值/连续负 IC 检测，触发 DecayAlert
  - get_dashboard(): 全部非终态因子的监控摘要（滚动 IC-IR、最新 IC、告警状态）

衰减规则（可配）：
  - 连续 ``consecutive_neg_limit``（默认 10）个交易日 IC < 0 → 告警
  - 或最近 ``rolling_window``（默认 20）日滚动 IC 均值 < ``mean_ic_floor``（默认 -0.01）→ 告警

设计说明：
  - 只读写 AlphaStore（SQLite），无内存状态 → 进程重启无损。
  - check_decay 只产生告警对象，不自动流转状态——状态变更是运营决策，
    由调用方（调度任务或人工 PATCH /status）显式执行，保持审计清晰。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ...db.alpha_store import AlphaStore
from ...db.alpha_lifecycle import AlphaStatus, TERMINAL_STATES, coerce_status

logger = logging.getLogger(__name__)


@dataclass
class MonitorStatus:
    """update() 返回：写入后的即时监控快照。"""
    alpha_id:        int
    date:            str
    realized_ic:     float
    n_history:       int                 # 累计 IC 记录天数
    rolling_mean_ic: float               # 最近 rolling_window 日均值
    rolling_ic_ir:   float               # 最近 rolling_window 日 IC-IR
    decay_alert:     Optional["DecayAlert"] = None


@dataclass
class DecayAlert:
    """衰减告警。"""
    alpha_id:         int
    reason:           str                # "consecutive_negative" | "rolling_mean_below_floor"
    consecutive_neg:  int
    rolling_mean_ic:  float
    suggested_action: str = "review → DECAYING or RETIRED"


@dataclass
class AlphaDashboardRow:
    """get_dashboard() 单行。"""
    alpha_id:        int
    dsl:             str
    status:          str
    sharpe:          float
    n_ic_days:       int
    latest_ic:       Optional[float]
    latest_date:     Optional[str]
    rolling_mean_ic: float
    rolling_ic_ir:   float
    consecutive_neg: int
    has_alert:       bool
    allowed_next:    List[str] = field(default_factory=list)


class AlphaMonitor:
    """
    滚动 IC/IC-IR 监控器。

    Parameters
    ----------
    store                 : AlphaStore 实例
    rolling_window        : 滚动统计窗口（交易日，默认 20）
    consecutive_neg_limit : 连续负 IC 告警阈值（默认 10）
    mean_ic_floor         : 滚动均值告警下限（默认 -0.01）
    """

    def __init__(
        self,
        store:                 AlphaStore,
        rolling_window:        int   = 20,
        consecutive_neg_limit: int   = 10,
        mean_ic_floor:         float = -0.01,
    ) -> None:
        if rolling_window < 5:
            raise ValueError(f"rolling_window 至少 5，当前={rolling_window}")
        if consecutive_neg_limit < 2:
            raise ValueError(f"consecutive_neg_limit 至少 2，当前={consecutive_neg_limit}")
        self.store                 = store
        self.rolling_window        = rolling_window
        self.consecutive_neg_limit = consecutive_neg_limit
        self.mean_ic_floor         = mean_ic_floor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        alpha_id:        int,
        date,                          # datetime.date | "YYYY-MM-DD"
        realized_ic:     float,
        realized_return: float = 0.0,
    ) -> MonitorStatus:
        """
        记录当日 realized IC 并返回即时监控快照。
        幂等：同 (alpha_id, date) 重复调用覆盖旧值，不重复记账。
        因子不存在时抛 KeyError。
        """
        if self.store.get_by_id(alpha_id) is None:
            raise KeyError(f"Alpha id={alpha_id} 不存在")

        self.store.record_ic(alpha_id, date, realized_ic, realized_return)

        ics = self._ic_values(alpha_id)
        roll = ics[-self.rolling_window:]
        status = MonitorStatus(
            alpha_id        = alpha_id,
            date            = str(date),
            realized_ic     = float(realized_ic),
            n_history       = len(ics),
            rolling_mean_ic = float(np.mean(roll)) if roll else np.nan,
            rolling_ic_ir   = self._ic_ir(roll),
            decay_alert     = self.check_decay(alpha_id, _ics=ics),
        )
        if status.decay_alert is not None:
            logger.warning(
                "[AlphaMonitor] 衰减告警 alpha_id=%d | %s | 连续负IC=%d | 滚动均值=%.4f",
                alpha_id, status.decay_alert.reason,
                status.decay_alert.consecutive_neg, status.decay_alert.rolling_mean_ic,
            )
        return status

    def check_decay(
        self,
        alpha_id: int,
        _ics: Optional[List[float]] = None,
    ) -> Optional[DecayAlert]:
        """
        衰减检测；无告警返回 None。
        记录不足 rolling_window 天时不告警（数据不足不下结论）。
        """
        ics = self._ic_values(alpha_id) if _ics is None else _ics
        if len(ics) < self.rolling_window:
            return None

        consec = self._consecutive_negative(ics)
        roll_mean = float(np.mean(ics[-self.rolling_window:]))

        if consec >= self.consecutive_neg_limit:
            return DecayAlert(
                alpha_id        = alpha_id,
                reason          = "consecutive_negative",
                consecutive_neg = consec,
                rolling_mean_ic = roll_mean,
            )
        if roll_mean < self.mean_ic_floor:
            return DecayAlert(
                alpha_id        = alpha_id,
                reason          = "rolling_mean_below_floor",
                consecutive_neg = consec,
                rolling_mean_ic = roll_mean,
            )
        return None

    def get_dashboard(self) -> List[AlphaDashboardRow]:
        """
        全部非终态因子的监控摘要（按 Sharpe 降序）。
        终态（RETIRED / SUPERSEDED）因子不出现在仪表板。
        """
        from ...db.alpha_lifecycle import allowed_next

        rows: List[AlphaDashboardRow] = []
        for rec in self.store.query(limit=500):
            try:
                st = coerce_status(rec.status)
            except ValueError:
                st = AlphaStatus.CANDIDATE          # 未知历史状态按候选处理
            if st in TERMINAL_STATES:
                continue

            hist = self.store.get_ic_history(rec.id, limit=self.rolling_window * 3)
            ics  = [h.realized_ic for h in hist]
            roll = ics[-self.rolling_window:]
            rows.append(AlphaDashboardRow(
                alpha_id        = rec.id,
                dsl             = rec.dsl,
                status          = st.value,
                sharpe          = float(rec.sharpe or 0.0),
                n_ic_days       = len(ics),
                latest_ic       = float(ics[-1]) if ics else None,
                latest_date     = str(hist[-1].date) if hist else None,
                rolling_mean_ic = float(np.mean(roll)) if roll else np.nan,
                rolling_ic_ir   = self._ic_ir(roll),
                consecutive_neg = self._consecutive_negative(ics),
                has_alert       = self.check_decay(rec.id, _ics=ics) is not None,
                allowed_next    = allowed_next(st),
            ))
        rows.sort(key=lambda r: r.sharpe, reverse=True)
        return rows

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ic_values(self, alpha_id: int) -> List[float]:
        return [h.realized_ic for h in
                self.store.get_ic_history(alpha_id, limit=self.rolling_window * 3)]

    @staticmethod
    def _consecutive_negative(ics: List[float]) -> int:
        """从末尾数起的连续负 IC 天数。"""
        n = 0
        for v in reversed(ics):
            if v < 0:
                n += 1
            else:
                break
        return n

    @staticmethod
    def _ic_ir(ics: List[float]) -> float:
        if len(ics) < 2:
            return float("nan")
        arr = np.asarray(ics, dtype=float)
        sd  = arr.std(ddof=1)
        return float(arr.mean() / sd) if sd > 1e-12 else float("nan")
