"""
cost_calibration.py — 成本模型校准回路（Task 8.2，2026-08-19）

职责（每月任务，人机协同）
--------------------------
对比 PaperBroker 的**假设成交价**（= 成交当日收盘 `fill_price`）与**实际 T+1 OHLC 区间**
（真正交易会在次日开盘附近落在 [low, high] 内），量化 close-fill 假设**遗漏的隔夜执行滑点**，
据此输出**冲击系数（impact_coef）校准建议报告**。

铁律
----
- **绝不自动改 `CostParams`**：只产出建议 + 写变更日志；由人工确认后手动更新（RESEARCH_
  OPERATING_MODEL：成本口径变更须留痕、人工把关）。
- 纯函数 `calibrate(fills, prices, cost_params)`（数据入 → 报告出），不触网、可测试。

方法
----
对每笔实际成交（|filled_weight|>0 且未被拒绝）：
  隔夜执行缺口 gap_bps = (open_{d+1} − close_d) / close_d × 1e4
  其绝对值 |gap_bps| = close-fill 假设完全忽略的执行价不确定性。
汇总 median(|gap_bps|)，与当前模型的半价差基线 `spread_bps` 比较，给出**有界的**
impact_coef 缩放建议（clip 到 [0.5×, 2.0×]，避免单月噪声导致剧烈调整）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WidePanel = Dict[str, pd.DataFrame]

# 缩放建议的边界：单次校准最多把 impact_coef 调整到 [0.5×, 2×]
_SCALE_LO, _SCALE_HI = 0.5, 2.0


@dataclass
class CalibrationReport:
    n_fills: int
    n_days: int
    period_start: str
    period_end: str
    realized_gap_bps_mean: float          # 有符号隔夜缺口均值
    realized_gap_bps_abs_median: float    # |隔夜缺口| 中位数（核心量）
    realized_gap_bps_abs_p90: float
    assumed_spread_bps: float             # 当前 CostParams.spread_bps
    current_impact_coef: float
    recommended_impact_coef: float
    recommended_scale: float
    note: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# 成本模型校准报告（Task 8.2）",
            "",
            f"- 期间：{self.period_start} → {self.period_end}",
            f"- 成交笔数：{self.n_fills} | 交易日数：{self.n_days}",
            "",
            "## 实际隔夜执行缺口（close_d → open_{d+1}）",
            f"- 有符号均值：{self.realized_gap_bps_mean:.2f} bps",
            f"- |缺口| 中位数：{self.realized_gap_bps_abs_median:.2f} bps",
            f"- |缺口| P90：{self.realized_gap_bps_abs_p90:.2f} bps",
            "",
            "## 冲击系数建议（**需人工确认后手动更新，切勿自动改**）",
            f"- 当前 spread_bps（半价差基线）：{self.assumed_spread_bps:.2f}",
            f"- 当前 impact_coef：{self.current_impact_coef:.4f}",
            f"- 建议缩放：×{self.recommended_scale:.3f}（已 clip 到 [{_SCALE_LO}, {_SCALE_HI}]）",
            f"- **建议 impact_coef：{self.recommended_impact_coef:.4f}**",
            "",
            f"> {self.note}",
        ]
        if self.warnings:
            lines += ["", "## 告警"] + [f"- {w}" for w in self.warnings]
        return "\n".join(lines)


def calibrate(
    fills: pd.DataFrame,
    prices: WidePanel,
    *,
    spread_bps: float,
    impact_coef: float,
) -> Optional[CalibrationReport]:
    """
    纯函数校准。

    Parameters
    ----------
    fills : DataFrame，至少含列 [date, ticker, filled_weight, fill_price, reject_reason]。
            date 可为 str/date/Timestamp；fill_price = 成交当日收盘。
    prices: {field: wide DataFrame(index=DatetimeIndex, columns=ticker)}，至少含 'open'。
            用于取 T+1 开盘价。
    spread_bps, impact_coef : 当前 CostParams 对应值。

    Returns
    -------
    CalibrationReport；无有效成交时返回 None。
    """
    if fills is None or len(fills) == 0 or "open" not in prices:
        logger.info("[cost_calib] 无成交或缺 open 价，跳过")
        return None

    df = fills.copy()
    df = df[(df.get("reject_reason", "").astype(str) == "") & (df["filled_weight"].abs() > 0)]
    if df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"])
    open_px = prices["open"].copy()
    open_px.index = pd.DatetimeIndex(open_px.index)
    trading_days = open_px.index.sort_values()

    warnings: List[str] = []
    gaps_bps: List[float] = []
    for _, row in df.iterrows():
        d = pd.Timestamp(row["date"])
        tkr = str(row["ticker"])
        close_d = float(row["fill_price"])
        if close_d <= 0 or tkr not in open_px.columns:
            continue
        # 次一交易日
        nxt = trading_days[trading_days > d]
        if len(nxt) == 0:
            continue
        open_next = open_px.loc[nxt[0], tkr]
        if pd.isna(open_next):
            continue
        gaps_bps.append((float(open_next) - close_d) / close_d * 1e4)

    if not gaps_bps:
        warnings.append("无可匹配 T+1 开盘价的成交，无法校准")
        logger.warning("[cost_calib] %s", warnings[-1])
        return None

    g = np.asarray(gaps_bps, dtype=float)
    abs_median = float(np.median(np.abs(g)))
    base = max(float(spread_bps), 1e-6)
    scale = float(np.clip(abs_median / base, _SCALE_LO, _SCALE_HI))
    recommended = float(impact_coef * scale)

    if abs_median > base * _SCALE_HI:
        note = ("实际隔夜缺口显著大于半价差基线 → close-fill 低估执行成本，"
                "建议上调 impact_coef（已按上界 2× 截断，可分月渐进调整）。")
    elif abs_median < base * _SCALE_LO:
        note = "实际隔夜缺口显著小于基线 → 当前成本或偏保守，可考虑下调（下界 0.5× 截断）。"
    else:
        note = "实际隔夜缺口与基线量级相当 → 当前 impact_coef 大体合理，建议维持并继续观察。"

    report = CalibrationReport(
        n_fills=len(g),
        n_days=int(df["date"].nunique()),
        period_start=str(df["date"].min().date()),
        period_end=str(df["date"].max().date()),
        realized_gap_bps_mean=float(np.mean(g)),
        realized_gap_bps_abs_median=abs_median,
        realized_gap_bps_abs_p90=float(np.percentile(np.abs(g), 90)),
        assumed_spread_bps=float(spread_bps),
        current_impact_coef=float(impact_coef),
        recommended_impact_coef=recommended,
        recommended_scale=scale,
        note=note,
        warnings=warnings,
    )
    logger.info(
        "[cost_calib] %s→%s | %d 笔 | |gap|中位=%.2fbps | impact_coef %.4f→%.4f (×%.3f) — 仅建议，人工确认",
        report.period_start, report.period_end, report.n_fills,
        abs_median, impact_coef, recommended, scale,
    )
    return report


def run_monthly_calibration(
    dataset_name: str,
    start: str,
    end: str,
    write_path: Optional[str] = None,
) -> Optional[CalibrationReport]:
    """
    每月校准任务（供调度器调用）：读区间内 PaperFill + 加载价格 → calibrate → 落报告。
    **不改 CostParams**；仅写报告 + 日志，由人工据此手动更新。
    """
    from app.config import settings
    from app.core.backtest_engine.transaction_cost import CostParams
    from app.core.data_engine.dataset_registry import load_registry_dataset
    from app.db.position_store import PositionStore

    store = PositionStore(db_url=settings.database_url)
    fill_rows = store.fills_in_range(start, end)
    if not fill_rows:
        logger.info("[cost_calib] 区间内无成交，跳过月度校准")
        return None

    fills = pd.DataFrame([{
        "date": r.date, "ticker": r.ticker,
        "filled_weight": r.filled_weight, "fill_price": r.fill_price,
        "reject_reason": r.reject_reason,
    } for r in fill_rows])

    ds = load_registry_dataset(dataset_name, start=start, end=end, health_check=False)
    params = CostParams()
    report = calibrate(
        fills, ds.data,
        spread_bps=params.spread_bps, impact_coef=params.impact_coef,
    )
    if report is None:
        return None

    if write_path:
        try:
            with open(write_path, "w", encoding="utf-8") as f:
                f.write(report.to_markdown())
            logger.info("[cost_calib] 报告已写入 %s", write_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[cost_calib] 写报告失败: %s", exc)
    return report
