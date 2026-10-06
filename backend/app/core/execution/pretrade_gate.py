"""
pretrade_gate.py — 下单前风控门（Phase 12.2，纯函数）

命名说明：路线图写的是 `execution/risk_gate.py`，但 `portfolio_manager/risk_gate.py`
已经存在（组合层：对**目标权重面板**削单票/行业/总敞口/目标波动）。两个同名模块正是
DEV_LESSONS §L 记录过的误删来源，所以这里叫 pretrade_gate。

与组合层风控的分工
------------------
组合层管"想持有什么"；本门管"这一张单能不能发出去"。目标权重已经过组合层风控，
所以正常情况下本门一张都不该拦 —— **它拦下的每一张都意味着上游有 bug 或数据有问题**
（价格陈旧、ADV 缺失、资金不够、持仓与账本不符…）。因此每条规则都是 fail-closed：
算不出来就拒，不是放。

规则（逐单、按订单顺序累计；先卖后买）
--------------------------------------
1. 全平熔断（kill switch）开启 → 只放行 purpose=flatten 的单
2. 日亏熔断：账户日收益 ≤ −max_daily_loss → 拒绝一切**增加风险**的单（减仓照常放行）
3. 参考价必须有效；与券商现价偏离 > max_price_deviation → 拒（数据陈旧 / 复权口径错位）
4. 只做多：卖出量不得超过持仓（卖成空头）
5. fat-finger：单笔名义额 > max_participation_pct × ADV → 拒；ADV 缺失 → 拒
   （与成本模型共用 `CostParams.max_participation_pct`，单一来源；全平单不受此限）
6. 成交后单票权重 > max_name_weight → 拒（只拦增加风险的单）
7. 成交后总敞口 > max_gross → 拒（累计计算，只拦增加风险的单）
8. 买入力：**每一张买单**（含回补空头 —— 回补同样要花现金）的累计名义额（按限价）
   > 买入力 → 拒。**不计同批卖单所得** —— 它们与买单同时在次日开盘成交，下单时那笔钱还不存在

浮点：限额比较前先按位数取整（权重 12 位、美元 6 位），只吸收运算噪声，
"恰好等于上限"放行 —— 边界因此可以被精确测试，不靠一个测不到的 epsilon 余量。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import pandas as pd

from app.core.execution.broker_gateway import DUST_QTY
from app.core.execution.order_builder import PURPOSE_FLATTEN, PlannedOrder

#: 权重比较取整位数 / 美元比较取整位数
_W_DIGITS, _USD_DIGITS = 12, 6


@dataclass(frozen=True)
class PreTradeLimits:
    max_name_weight:       float
    max_gross:             float
    max_participation_pct: float
    max_daily_loss:        float
    max_price_deviation:   float
    allow_short:           bool = False

    def __post_init__(self):
        for k in ("max_name_weight", "max_gross", "max_participation_pct",
                  "max_daily_loss", "max_price_deviation"):
            v = getattr(self, k)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"PreTradeLimits.{k} 必须是正的有限数：{v!r}")

    @classmethod
    def from_settings(cls, settings, cost_params=None) -> "PreTradeLimits":
        """
        单一来源（§J）：单票/总敞口用组合层同一组配置，ADV 参与率用成本模型同一个参数。
        """
        if cost_params is None:
            from app.core.backtest_engine.transaction_cost import CostParams
            cost_params = CostParams()
        return cls(
            max_name_weight=float(settings.risk_max_name_weight),
            max_gross=float(settings.risk_max_gross),
            max_participation_pct=float(cost_params.max_participation_pct),
            max_daily_loss=float(settings.exec_max_daily_loss),
            max_price_deviation=float(settings.exec_max_price_deviation),
            allow_short=bool(settings.trading_allow_short),
        )


@dataclass
class GateContext:
    equity:        float                     # 账户总资产（USD）
    buying_power:  float
    current_qty:   Mapping[str, float]       # 券商持仓股数（有符号）
    broker_prices: Mapping[str, float]       # 券商现价（持仓里有的名字）
    adv_usd:       Mapping[str, float]
    day_return:    Optional[float] = None    # 账户日收益；None = 没有上一交易日快照
    kill_switch:   bool = False


@dataclass
class GateDecision:
    approved: List[PlannedOrder] = field(default_factory=list)
    rejected: List[Tuple[PlannedOrder, str]] = field(default_factory=list)
    halted:   str = ""                       # 整体熔断原因（kill_switch / daily_loss）
    notes:    List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_approved": len(self.approved),
            "n_rejected": len(self.rejected),
            "halted": self.halted,
            "rejected": [{"client_id": o.client_id, "ticker": o.ticker, "side": o.side,
                          "qty": o.qty, "reason": r} for o, r in self.rejected],
            "notes": list(self.notes),
        }


def _finite(x) -> Optional[float]:
    """有限数 → float；None/'N/A'/NaN/inf → None（调用方据此 fail-closed）。"""
    v = pd.to_numeric(pd.Series([x], dtype=object), errors="coerce").iloc[0]
    return float(v) if pd.notna(v) and math.isfinite(float(v)) else None


def increases_risk(current_qty: float, signed_qty: float) -> bool:
    """成交后 |持仓| 比成交前大出零股阈值以上（含反手）= 增加风险。"""
    post = current_qty + signed_qty
    return abs(post) > abs(current_qty) + DUST_QTY or (current_qty * post < 0)


def check(orders: List[PlannedOrder], ctx: GateContext, limits: PreTradeLimits) -> GateDecision:
    dec = GateDecision()
    equity = _finite(ctx.equity)
    power = _finite(ctx.buying_power)
    if equity is None or equity <= 0:
        dec.halted = "equity_unavailable"
        dec.rejected = [(o, "equity_unavailable") for o in orders]
        return dec

    daily_halt = (ctx.day_return is not None
                  and _finite(ctx.day_return) is not None
                  and float(ctx.day_return) <= -limits.max_daily_loss)
    if ctx.kill_switch:
        dec.halted = "kill_switch_engaged"
    elif daily_halt:
        dec.halted = "daily_loss_halt"
        dec.notes.append(f"账户日收益 {float(ctx.day_return):.2%} ≤ −{limits.max_daily_loss:.2%}：只放行减仓单")
    if ctx.day_return is None:
        dec.notes.append("无上一交易日快照，日亏熔断本轮无从判定")

    qty: Dict[str, float] = {k: float(v) for k, v in ctx.current_qty.items()}
    px_book: Dict[str, float] = {}
    for tk, p in ctx.broker_prices.items():
        v = _finite(p)
        if v is not None and v > 0:
            px_book[tk] = v
    buy_spent = 0.0

    def _gross(prices: Dict[str, float]) -> Optional[float]:
        tot = 0.0
        for tk, q in qty.items():
            if abs(q) <= DUST_QTY:
                continue
            p = prices.get(tk)
            if p is None:
                return None
            tot += abs(q) * p
        return tot / equity

    for o in orders:
        cur = qty.get(o.ticker, 0.0)
        incr = increases_risk(cur, o.signed_qty)

        if ctx.kill_switch and o.purpose != PURPOSE_FLATTEN:
            dec.rejected.append((o, "kill_switch_engaged")); continue
        if daily_halt and incr:
            dec.rejected.append((o, "daily_loss_halt")); continue

        ref = _finite(o.ref_price)
        if ref is None or ref <= 0:
            dec.rejected.append((o, "no_reference_price")); continue
        bp = px_book.get(o.ticker)
        if bp is not None and abs(ref - bp) / bp > limits.max_price_deviation:
            dec.rejected.append((o, "price_mismatch")); continue

        post = cur + o.signed_qty
        if not limits.allow_short and post < -DUST_QTY:
            dec.rejected.append((o, "would_short")); continue

        if o.purpose != PURPOSE_FLATTEN:
            adv = _finite(ctx.adv_usd.get(o.ticker))
            if adv is None or adv <= 0:
                dec.rejected.append((o, "no_adv")); continue
            if (round(o.notional, _USD_DIGITS)
                    > round(limits.max_participation_pct * adv, _USD_DIGITS)):
                dec.rejected.append((o, "exceeds_adv_participation")); continue

        if incr:
            if round(abs(post) * ref / equity, _W_DIGITS) > limits.max_name_weight:
                dec.rejected.append((o, "exceeds_name_limit")); continue
            prices = dict(px_book)
            prices[o.ticker] = ref
            qty[o.ticker] = post
            g = _gross(prices)
            qty[o.ticker] = cur
            if g is None:
                dec.rejected.append((o, "gross_unpriceable")); continue
            if round(g, _W_DIGITS) > limits.max_gross:
                dec.rejected.append((o, "exceeds_gross_limit")); continue

        if o.side == "BUY":
            cost = float(o.qty) * float(o.limit_price)
            if power is None or round(buy_spent + cost, _USD_DIGITS) > round(power, _USD_DIGITS):
                dec.rejected.append((o, "insufficient_buying_power")); continue
            buy_spent += cost

        qty[o.ticker] = post
        px_book.setdefault(o.ticker, ref)
        dec.approved.append(o)
    return dec
