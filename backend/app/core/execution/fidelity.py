"""
fidelity.py — 保真度阶梯第二级：内部模拟 vs moomoo 纸交易（Phase 12.3，纯函数）

三级保真度：内部模拟（决策日收盘价成交）→ **moomoo 纸交易**（收盘后下单、次日开盘成交）
→ （未来）moomoo 实盘。本模块量第一级与第二级之间的差。

对每一笔纸交易成交（只看调仓单；全平单在模拟账本里没有对应物）：

    总滑点   = s · (成交价 − 决策日收盘) / 决策日收盘          ← 模拟账本按收盘价成交
    隔夜缺口 = s · (成交日开盘 − 决策日收盘) / 决策日收盘      ← T+1 开盘成交模型要补的部分
    开盘执行 = s · (成交价 − 成交日开盘) / 成交日开盘          ← 价差 + 纸交易引擎的撮合

（s = 买 +1 / 卖 −1；正值 = 比模拟账本更差。总滑点 ≈ 隔夜缺口 + 开盘执行。）

不能从纸交易里得出的东西 —— 写在报告里而不是留白
-----------------------------------------------
- **永久冲击**：纸交易的成交不改变真实市场价格，永久冲击在这里恒等于不可识别。
  需要实盘成交才能估计，本报告对它只给"不可识别"及原因，不给数字。
- 冲击系数建议沿用成本校准（Task 8.2）同一套有界缩放：**只是建议，绝不自动改 CostParams**。
  且它校准的是"纸交易引擎的撮合口径"，不是真实市场冲击 —— 报告里注明。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.execution.broker_gateway import DUST_QTY
from app.tasks.cost_calibration import SCALE_BOUNDS

#: 权重口径的零值阈值（≈9.1e-13）；与 DUST_QTY 同理取 2 的幂，边界可精确测试
WEIGHT_DUST = 2.0 ** -40
#: 模拟成交价与纸交易参考价"是同一份收盘价"的相对容差（≈0.95 ppm）
REF_PRICE_TOL = 2.0 ** -20

PERMANENT_IMPACT_NOTE = (
    "纸交易成交不改变真实市场价格，永久冲击在纸交易数据里不可识别；"
    "需要实盘成交（保真度阶梯第三级）才能估计。")


def _stats(values: List[float], weights: Optional[List[float]] = None) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    v = np.asarray(values, dtype=float)
    if weights is not None and np.sum(weights) > 0:
        mean = float(np.average(v, weights=np.asarray(weights, dtype=float)))
    else:
        mean = float(np.mean(v))
    return {"n": int(len(v)), "mean": mean, "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90))}


@dataclass
class FidelityReport:
    period_start:      str
    period_end:        str
    n_live_orders:     int
    n_live_fills:      int
    n_matched_sim:     int
    n_ref_mismatch:    int
    live_fill_ratio_median: Optional[float]
    sim_fill_ratio_median:  Optional[float]
    n_live_unfilled:   int
    total_slippage_bps:  dict
    overnight_gap_bps:   dict
    at_open_exec_bps:    dict
    assumed_spread_bps:  float
    current_impact_coef: float
    recommended_impact_coef: float
    recommended_scale:   float
    permanent_impact_bps: Optional[float] = None
    permanent_impact_note: str = PERMANENT_IMPACT_NOTE
    notes:    List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        def fmt(d):
            if not d or d.get("n", 0) == 0:
                return "无样本"
            return (f"n={d['n']} · 均值(按成交额加权) {d['mean']:.2f} · 中位 {d['median']:.2f}"
                    f" · P90 {d['p90']:.2f} bps")
        lines = [
            "# 保真度报告：内部模拟 vs moomoo 纸交易（Phase 12.3）", "",
            f"- 期间（决策日）：{self.period_start} → {self.period_end}",
            f"- 纸交易调仓单 {self.n_live_orders} 张 · 成交 {self.n_live_fills} 笔 · "
            f"与模拟账本匹配 {self.n_matched_sim} 张 · 参考价不一致 {self.n_ref_mismatch} 张",
            f"- 成交率中位数：纸交易 {self.live_fill_ratio_median} · 模拟 {self.sim_fill_ratio_median} · "
            f"纸交易零成交单 {self.n_live_unfilled} 张", "",
            "## 成交价差（正 = 比模拟账本更差）",
            f"- 总滑点（成交价 vs 决策日收盘）：{fmt(self.total_slippage_bps)}",
            f"- 隔夜缺口（次日开盘 vs 决策日收盘）：{fmt(self.overnight_gap_bps)}",
            f"- 开盘执行（成交价 vs 次日开盘）：{fmt(self.at_open_exec_bps)}", "",
            "## 冲击系数建议（**仅建议，人工确认后手动改，切勿自动改**）",
            f"- 当前 spread_bps={self.assumed_spread_bps:.2f}（半价差基线 {self.assumed_spread_bps / 2:.2f}）"
            f" · impact_coef={self.current_impact_coef:.4f}",
            f"- 建议缩放 ×{self.recommended_scale:.3f} → impact_coef {self.recommended_impact_coef:.4f}",
            "- 注意：这是**纸交易引擎撮合口径**的校准，不是真实市场冲击。", "",
            f"## 永久冲击：不可识别", f"> {self.permanent_impact_note}",
        ]
        if self.notes:
            lines += ["", "## 说明"] + [f"- {n}" for n in self.notes]
        if self.warnings:
            lines += ["", "## 告警"] + [f"- {w}" for w in self.warnings]
        return "\n".join(lines)


def build_fidelity_report(
    live_orders: pd.DataFrame,
    live_fills: pd.DataFrame,
    sim_fills: pd.DataFrame,
    open_px: pd.DataFrame,
    *,
    spread_bps: float,
    impact_coef: float,
    period_start: str,
    period_end: str,
    min_fills: int = 20,
) -> FidelityReport:
    """
    live_orders : [client_id, decision_date, ticker, side, qty, dealt_qty, status, purpose, ref_price]
    live_fills  : [client_id, decision_date, fill_date, ticker, qty(有符号), price, ref_price]
    sim_fills   : [date, ticker, traded_weight, unfilled_weight, fill_price]（模拟组合账本）
    open_px     : 开盘价宽表（DatetimeIndex × ticker）
    """
    lo, hi = SCALE_BOUNDS
    warnings: List[str] = []
    notes: List[str] = []

    orders = live_orders.copy() if len(live_orders) else pd.DataFrame(
        columns=["client_id", "decision_date", "ticker", "side", "qty", "dealt_qty",
                 "status", "purpose", "ref_price"])
    orders = orders[orders["purpose"] == "rebalance"]
    rebalance_ids = set(orders["client_id"])
    fills = live_fills[live_fills["client_id"].isin(rebalance_ids)] if len(live_fills) else live_fills

    # ── 成交率 ───────────────────────────────────────────────────────────
    terminal = orders[orders["status"].isin(["FILLED", "PARTIAL", "CANCELLED", "FAILED"])]
    ratios = [float(r.dealt_qty) / float(r.qty) for r in terminal.itertuples() if float(r.qty) > 0]
    n_unfilled = int(sum(1 for r in terminal.itertuples() if float(r.dealt_qty) <= DUST_QTY))

    # ── 与模拟账本逐单匹配 ────────────────────────────────────────────────
    sim = sim_fills.copy() if len(sim_fills) else pd.DataFrame(
        columns=["date", "ticker", "traded_weight", "unfilled_weight", "fill_price"])
    sim["date"] = pd.to_datetime(sim["date"]).dt.date if len(sim) else sim["date"]
    sim_idx = {(r.date, r.ticker): r for r in sim.itertuples()}
    n_matched = n_ref_mismatch = 0
    sim_ratios: List[float] = []
    for r in orders.itertuples():
        s = sim_idx.get((pd.Timestamp(r.decision_date).date(), r.ticker))
        if s is None:
            continue
        intended = float(s.traded_weight) + float(s.unfilled_weight)
        if abs(intended) <= WEIGHT_DUST or (intended > 0) != (r.side == "BUY"):
            continue
        n_matched += 1
        sim_ratios.append(abs(float(s.traded_weight)) / abs(intended))
        ref = float(r.ref_price)
        if ref > 0 and abs(float(s.fill_price) - ref) / ref > REF_PRICE_TOL:
            n_ref_mismatch += 1
    if n_ref_mismatch:
        warnings.append(f"{n_ref_mismatch} 张单的模拟成交价与纸交易参考价不一致 —— "
                        "模拟账本与执行层用的不是同一份收盘价，比较口径不成立")

    # ── 逐笔成交的三段分解 ────────────────────────────────────────────────
    opx = open_px.copy()
    opx.index = pd.DatetimeIndex(opx.index).normalize()
    tot, gap, exe, w_tot, w_gap = [], [], [], [], []
    n_no_open = 0
    for f in fills.itertuples():
        ref, px, q = float(f.ref_price), float(f.price), float(f.qty)
        if ref <= 0 or px <= 0 or abs(q) <= DUST_QTY:
            continue
        s = 1.0 if q > 0 else -1.0
        notional = abs(q) * px
        tot.append(s * (px - ref) / ref * 1e4)
        w_tot.append(notional)
        fd = pd.Timestamp(f.fill_date).normalize()
        o = opx[f.ticker].get(fd, np.nan) if f.ticker in opx.columns else np.nan
        if not (isinstance(o, (int, float, np.floating)) and math.isfinite(float(o)) and float(o) > 0):
            n_no_open += 1
            continue
        o = float(o)
        gap.append(s * (o - ref) / ref * 1e4)
        exe.append(s * (px - o) / o * 1e4)
        w_gap.append(notional)
    if n_no_open:
        warnings.append(f"{n_no_open} 笔成交找不到成交日开盘价，未参与缺口/执行分解")

    exe_stats = _stats(exe, w_gap)
    half = max(float(spread_bps) / 2.0, 1e-6)
    if exe_stats["n"] >= min_fills:
        med = exe_stats["median"]
        scale = float(np.clip(max(med, 0.0) / half, lo, hi))
        if med <= 0:
            notes.append("开盘执行中位数 ≤ 0：纸交易撮合比开盘价还好，模型成本偏保守（按下界截断）")
    else:
        scale = 1.0
        warnings.append(f"可分解的成交只有 {exe_stats['n']} 笔（< {min_fills}），不足以校准，建议维持现值")

    return FidelityReport(
        period_start=period_start, period_end=period_end,
        n_live_orders=int(len(orders)), n_live_fills=int(len(tot)),
        n_matched_sim=n_matched, n_ref_mismatch=n_ref_mismatch,
        live_fill_ratio_median=float(np.median(ratios)) if ratios else None,
        sim_fill_ratio_median=float(np.median(sim_ratios)) if sim_ratios else None,
        n_live_unfilled=n_unfilled,
        total_slippage_bps=_stats(tot, w_tot),
        overnight_gap_bps=_stats(gap, w_gap),
        at_open_exec_bps=exe_stats,
        assumed_spread_bps=float(spread_bps), current_impact_coef=float(impact_coef),
        recommended_impact_coef=float(impact_coef) * scale, recommended_scale=scale,
        notes=notes, warnings=warnings,
    )


def fidelity_from_stores(start, end, open_px: pd.DataFrame, exec_store=None,
                         position_store=None, sim_book_id: int = 0,
                         live_book_id: int = -1) -> FidelityReport:
    """从执行账本 + 模拟组合账本取数（供端点与月度任务）。"""
    from app.core.backtest_engine.transaction_cost import CostParams
    from app.db.execution_store import ExecutionStore
    from app.db.position_store import PositionStore
    es = exec_store or ExecutionStore()
    ps = position_store or PositionStore()
    s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    orders = pd.DataFrame([{
        "client_id": r.client_id, "decision_date": r.decision_date, "ticker": r.ticker,
        "side": r.side, "qty": r.qty, "dealt_qty": r.dealt_qty, "status": r.status,
        "purpose": r.purpose, "ref_price": r.ref_price}
        for r in es.orders(since=s, limit=100000)
        if r.book_id == live_book_id and r.decision_date <= e])
    fills = pd.DataFrame([{
        "client_id": f.client_id, "decision_date": f.decision_date, "fill_date": f.fill_date,
        "ticker": f.ticker, "qty": f.qty, "price": f.price, "ref_price": f.ref_price}
        for f in es.fills(start=s, end=e, book_id=live_book_id)])
    sim = pd.DataFrame([{
        "date": r.date, "ticker": r.ticker, "traded_weight": r.traded_weight,
        "unfilled_weight": r.unfilled_weight, "fill_price": r.fill_price}
        for r in ps.fills_in_range(s, e, alpha_id=sim_book_id)])
    p = CostParams()
    return build_fidelity_report(
        orders, fills, sim, open_px, spread_bps=p.spread_bps, impact_coef=p.impact_coef,
        period_start=str(s), period_end=str(e))
