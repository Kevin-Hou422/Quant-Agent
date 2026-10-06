"""
order_builder.py — 目标权重 → 股数订单（Phase 12.1，纯函数，不触网）

口径
----
- 目标股数 = trunc(w × 账户总资产 / 参考价)，**向零取整**：整股下单永远不会买超目标。
  被取整掉的权重如实记进 `rounding`（$10k 账户买一只 $7000 的股票，目标 5% 就只能是 0 股）。
- 订单 = 目标股数 − 券商当前股数。券商持仓里有、目标里没有的名字 → 目标 0 → 卖出
  （模拟账户专供本系统；见 config.execution_mode 的说明）。
- 限价 = 参考价 × (1 ± band)，按美股最小价位取整，且**只朝保守方向取整**：
  买单向下、卖单向上，限价永远不会比配置的 band 更宽。
- 先卖后买（同时提交、次日开盘一起成交；买入力检查不计卖出所得，见 pretrade_gate）。

幂等键（client_id）
------------------
同一 (book, 决策日, 标的, 方向, 用途) 只对应一个 client_id —— 它被写进 moomoo 的
remark 字段。重跑同一决策日不会产生第二张单；崩溃后也能按 remark 在券商侧找回。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from app.core.execution.broker_gateway import DUST_QTY

PURPOSE_REBALANCE = "rebalance"
PURPOSE_FLATTEN = "flatten"

#: 本系统下的单在 remark 里的前缀 —— 对账时据此区分"我们的单"与账户里别的单
CLIENT_ID_PREFIX = "qa"


@dataclass(frozen=True)
class PlannedOrder:
    client_id:     str
    ticker:        str
    side:          str        # "BUY" | "SELL"
    qty:           int        # 正整数
    ref_price:     float      # 定股数用的参考价（决策日收盘 / 全平时为券商现价）
    limit_price:   float
    current_qty:   float      # 下单前券商持仓（有符号）
    target_qty:    float      # 目标持仓（有符号）
    target_weight: float
    purpose:       str = PURPOSE_REBALANCE

    @property
    def signed_qty(self) -> int:
        return self.qty if self.side == "BUY" else -self.qty

    @property
    def notional(self) -> float:
        return float(self.qty) * float(self.ref_price)


@dataclass
class BuildResult:
    orders:   List[PlannedOrder] = field(default_factory=list)
    skipped:  List[dict] = field(default_factory=list)        # [{ticker, reason}]
    rounding: Dict[str, float] = field(default_factory=dict)  # ticker → 实际可达权重 − 目标权重
    notes:    List[str] = field(default_factory=list)


def make_client_id(purpose: str, book_id: int, stamp: str, ticker: str, side: str) -> str:
    """
    `qaR-1-20260105-AAPL-B`。purpose 首字母 R/F，side 首字母 B/S。
    stamp：调仓用决策日 YYYYMMDD；全平用触发时刻（同一天可能全平不止一次）。
    """
    tag = "F" if purpose == PURPOSE_FLATTEN else "R"
    cid = f"{CLIENT_ID_PREFIX}{tag}{book_id}-{stamp}-{ticker}-{side[0]}"
    if len(cid.encode("utf-8")) > 64:
        raise ValueError(f"client_id 超过 64 字节：{cid!r}")
    return cid


def is_own_client_id(remark: str) -> bool:
    r = str(remark or "")
    return r.startswith(CLIENT_ID_PREFIX + "R") or r.startswith(CLIENT_ID_PREFIX + "F")


def round_to_tick(price: float, side: str) -> float:
    """
    美股最小价位：≥$1 为 0.01，<$1 为 0.0001。
    买单向下取整、卖单向上取整 —— 限价永远不会比 band 允许的更差。
    """
    if not (math.isfinite(price) and price > 0):
        raise ValueError(f"限价必须是正的有限数：{price!r}")
    tick = 0.01 if price >= 1.0 else 0.0001
    n = price / tick
    n = math.floor(n + 1e-9) if side == "BUY" else math.ceil(n - 1e-9)
    return round(n * tick, 4)


def _finite_pos(x) -> bool:
    v = pd.to_numeric(pd.Series([x], dtype=object), errors="coerce").iloc[0]
    return bool(pd.notna(v) and math.isfinite(float(v)) and float(v) > 0)


def build_rebalance_orders(
    target_weights: pd.Series,
    current_qty: Mapping[str, float],
    ref_prices: pd.Series,
    equity: float,
    decision_date: date,
    *,
    book_id: int,
    band_bps: float,
    allow_short: bool = False,
) -> BuildResult:
    if not _finite_pos(equity):
        raise ValueError(f"账户总资产必须是正的有限数：{equity!r}")
    if not (math.isfinite(band_bps) and band_bps >= 0):
        raise ValueError(f"限价带必须 ≥ 0：{band_bps!r}")
    band = band_bps * 1e-4
    stamp = f"{decision_date:%Y%m%d}"
    out = BuildResult()

    tw = target_weights.astype(float)
    universe = sorted(set(tw.index) | {tk for tk, q in current_qty.items() if abs(float(q)) > DUST_QTY})
    sells: List[PlannedOrder] = []
    buys:  List[PlannedOrder] = []
    for tk in universe:
        w = float(tw.get(tk, 0.0))
        if not math.isfinite(w):
            w = 0.0                      # 与模拟账本同口径：无信号 = 目标 0
        if w < 0 and not allow_short:
            out.notes.append(f"{tk}: 目标权重 {w:.4f} < 0 但不允许做空 → 按 0 处理")
            w = 0.0
        cur = float(current_qty.get(tk, 0.0))
        px = ref_prices.get(tk, np.nan)
        if not _finite_pos(px):
            out.skipped.append({"ticker": tk, "reason": "no_reference_price",
                                "current_qty": cur, "target_weight": w})
            continue
        px = float(px)
        target = float(math.trunc(w * equity / px))
        out.rounding[tk] = target * px / equity - w
        delta = target - cur
        if delta > 0:
            qty, side = int(math.floor(delta + 1e-9)), "BUY"
        elif delta < 0:
            qty, side = int(math.floor(-delta + 1e-9)), "SELL"
        else:
            continue
        if qty <= 0:
            # 只剩碎股差额（账户里有零碎股时会出现）：整股单下不了
            out.skipped.append({"ticker": tk, "reason": "fractional_remainder",
                                "current_qty": cur, "target_qty": target})
            continue
        limit = round_to_tick(px * (1 + band) if side == "BUY" else px * (1 - band), side)
        o = PlannedOrder(
            client_id=make_client_id(PURPOSE_REBALANCE, book_id, stamp, tk, side),
            ticker=tk, side=side, qty=qty, ref_price=px, limit_price=limit,
            current_qty=cur, target_qty=target, target_weight=w, purpose=PURPOSE_REBALANCE)
        (sells if side == "SELL" else buys).append(o)
    out.orders = sells + buys
    return out


def build_flatten_orders(
    positions: Mapping[str, "object"],
    stamp: str,
    *,
    book_id: int,
    band_bps: float,
) -> BuildResult:
    """
    一键全平：每个持仓按券商现价 × (1 ∓ band) 下反向限价单。
    `positions` 的值需有 `qty` 与 `nominal_price`（BrokerPosition）。
    """
    band = band_bps * 1e-4
    out = BuildResult()
    sells: List[PlannedOrder] = []
    buys:  List[PlannedOrder] = []
    for tk in sorted(positions):
        p = positions[tk]
        cur = float(p.qty)
        if abs(cur) <= DUST_QTY:
            continue
        px = getattr(p, "nominal_price", np.nan)
        if not _finite_pos(px):
            out.skipped.append({"ticker": tk, "reason": "no_broker_price", "current_qty": cur})
            continue
        px = float(px)
        side = "SELL" if cur > 0 else "BUY"
        qty = int(math.floor(abs(cur) + 1e-9))
        if qty <= 0:
            out.skipped.append({"ticker": tk, "reason": "fractional_remainder", "current_qty": cur})
            continue
        limit = round_to_tick(px * (1 - band) if side == "SELL" else px * (1 + band), side)
        o = PlannedOrder(
            client_id=make_client_id(PURPOSE_FLATTEN, book_id, stamp, tk, side),
            ticker=tk, side=side, qty=qty, ref_price=px, limit_price=limit,
            current_qty=cur, target_qty=0.0, target_weight=0.0, purpose=PURPOSE_FLATTEN)
        (sells if side == "SELL" else buys).append(o)
    out.orders = sells + buys
    return out
