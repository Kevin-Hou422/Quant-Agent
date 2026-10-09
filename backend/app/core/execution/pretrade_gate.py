"""
pretrade_gate.py — 下单前风控门（Phase 12.2，纯函数）

命名说明：路线图写的是 `execution/risk_gate.py`，但 `portfolio_manager/risk_gate.py`
已经存在（组合层：对**目标权重面板**削单票/行业/总敞口/目标波动）。两个同名模块正是
DEV_LESSONS §L 记录过的误删来源，所以这里叫 pretrade_gate。

与组合层风控的分工
------------------
组合层管"想持有什么"；本门管"这一张单能不能发出去"。每条规则都是 fail-closed：
算不出来就拒，不是放。

最坏情况包络（外部审计 F05 / F06，2026-10-08）
--------------------------------------------
原先逐单累计时**假设同批卖单全部成交**、且看不见已在途的挂单：账户持有 AAA 10%、
目标换成 BBB 10%，卖 AAA 被当成已完成的减仓，买 BBB 因此获批 —— 次日若卖单没成交、
买单成交，实际敞口 20%，违反 10% 上限。现在每个标的按两条腿算：

    long_leg  = 持仓 + 在途买单剩余 + 已批新买单      （所有买单成交、卖单都不成交）
    short_leg = 持仓 − 在途卖单剩余 − 已批新卖单      （所有卖单成交、买单都不成交）
    最坏敞口  = max(|long_leg|, |short_leg|)

单票 / 总敞口 / 行业按最坏敞口算，净敞口按 Σlong_leg 与 Σshort_leg 两端算。
**卖单在确认成交之前不释放任何风险预算** —— 满仓换股因此是两步：今天卖，成交后再买。

规则（逐单、按订单顺序累计）
----------------------------
1. 全平熔断（kill switch）开启 → 只放行 purpose=flatten 的单
2. 日亏熔断：账户日收益 ≤ −max_daily_loss → 拒绝一切**扩大最坏敞口**的单
3. 参考价必须有效；与券商现价偏离 > max_price_deviation → 拒
4. 只做多：short_leg 不得为负（含在途卖单 —— 两张卖单各自不超持仓、合起来超了也算）
5. fat-finger：单笔名义额 > max_participation_pct × ADV → 拒；ADV 缺失 → 拒（全平单不受此限）
6–9. 扩大最坏敞口的单：单票 > max_name_weight、总敞口 > max_gross、
     净敞口 > max_net、行业 > max_sector_weight × max_gross → 拒
10. 买入力：在途买单（按限价）+ 已批新买单 + 本单 > 买入力 → 拒（不计任何卖单所得）

浮点：限额比较前先按位数取整（权重 12 位、美元 6 位），"恰好等于上限"放行。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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
    #: 净敞口上限（|Σw|）。None = 不在下单层检查（仅供单元测试构造最小限额）。
    max_net:               Optional[float] = None
    #: 单行业上限（占 max_gross 的比例，与组合层 RiskLimits 同口径）。None = 不检查。
    max_sector_weight:     Optional[float] = None

    def __post_init__(self):
        for k in ("max_name_weight", "max_gross", "max_participation_pct",
                  "max_daily_loss", "max_price_deviation"):
            v = getattr(self, k)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"PreTradeLimits.{k} 必须是正的有限数：{v!r}")
        for k in ("max_net", "max_sector_weight"):
            v = getattr(self, k)
            if v is not None and not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"PreTradeLimits.{k} 必须是正的有限数或 None：{v!r}")

    @classmethod
    def from_settings(cls, settings, cost_params=None) -> "PreTradeLimits":
        """
        单一来源（§J）：单票/总敞口/净敞口/行业用组合层同一组配置，ADV 参与率用成本模型同一个参数。
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
            max_net=float(settings.risk_max_net),
            max_sector_weight=float(settings.risk_max_sector_weight),
        )


@dataclass(frozen=True)
class OpenOrder:
    """券商侧仍在途的我方订单（对账后）：剩余股数与限价。"""
    ticker:        str
    side:          str        # "BUY" | "SELL"
    remaining_qty: float      # 未成交股数（≥0）
    limit_price:   float


@dataclass
class GateContext:
    equity:        float                     # 账户总资产（USD）
    buying_power:  float
    current_qty:   Mapping[str, float]       # 券商持仓股数（有符号）
    broker_prices: Mapping[str, float]       # 券商现价（持仓里有的名字）
    adv_usd:       Mapping[str, float]
    day_return:    Optional[float] = None    # 账户日收益；None = 没有上一交易日快照
    kill_switch:   bool = False
    open_orders:   Sequence[OpenOrder] = ()  # 在途挂单 —— 计入最坏情况包络（审计 F05）
    sectors:       Mapping[str, object] = field(default_factory=dict)   # 标的 → 行业


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


class _Envelope:
    """逐标的的两条腿 + 价格。所有限额都从这里算。"""

    def __init__(self, current_qty: Mapping[str, float], prices: Dict[str, float]):
        self.cur = {k: float(v) for k, v in current_qty.items()}
        self.buys: Dict[str, float] = {}
        self.sells: Dict[str, float] = {}
        self.px = prices

    def add(self, ticker: str, side: str, qty: float) -> None:
        book = self.buys if side == "BUY" else self.sells
        book[ticker] = book.get(ticker, 0.0) + float(qty)

    def long_leg(self, tk: str) -> float:
        return self.cur.get(tk, 0.0) + self.buys.get(tk, 0.0)

    def short_leg(self, tk: str) -> float:
        return self.cur.get(tk, 0.0) - self.sells.get(tk, 0.0)

    def worst(self, tk: str) -> float:
        return max(abs(self.long_leg(tk)), abs(self.short_leg(tk)))

    def names(self):
        return set(self.cur) | set(self.buys) | set(self.sells)

    def _sum(self, fn) -> Optional[float]:
        tot = 0.0
        for tk in self.names():
            v = fn(tk)
            if abs(v) <= DUST_QTY:
                continue
            p = self.px.get(tk)
            if p is None:
                return None
            tot += v * p
        return tot

    def gross(self) -> Optional[float]:
        return self._sum(self.worst)

    def net_bounds(self) -> Optional[Tuple[float, float]]:
        hi, lo = self._sum(self.long_leg), self._sum(self.short_leg)
        return None if hi is None or lo is None else (hi, lo)

    def sector(self, members) -> Optional[float]:
        tot = 0.0
        for tk in members:
            v = self.worst(tk)
            if v <= DUST_QTY:
                continue
            p = self.px.get(tk)
            if p is None:
                return None
            tot += v * p
        return tot


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

    px_book: Dict[str, float] = {}
    for tk, p in ctx.broker_prices.items():
        v = _finite(p)
        if v is not None and v > 0:
            px_book[tk] = v
    env = _Envelope(ctx.current_qty, px_book)
    buy_spent = 0.0
    for oo in ctx.open_orders:
        rem = max(_finite(oo.remaining_qty) or 0.0, 0.0)   # 剩余 0 股加进去也是 0，无需另判
        env.add(oo.ticker, oo.side, rem)
        lp = _finite(oo.limit_price)
        if oo.ticker not in env.px and lp is not None and lp > 0:
            env.px[oo.ticker] = lp
        if oo.side == "BUY":
            buy_spent += rem * (lp or 0.0)
    if ctx.open_orders:
        dec.notes.append(f"{len(ctx.open_orders)} 张在途挂单计入最坏情况包络")

    # 行业未知（缺失 / NaN）→ 该标的自成一组（仍受单票上限约束），不并进某个真实行业
    sector_of = {tk: s for tk, s in ctx.sectors.items() if pd.notna(s)}
    sector_cap = (None if limits.max_sector_weight is None
                  else limits.max_sector_weight * limits.max_gross)

    for o in orders:
        if ctx.kill_switch and o.purpose != PURPOSE_FLATTEN:
            dec.rejected.append((o, "kill_switch_engaged")); continue

        ref = _finite(o.ref_price)
        if ref is None or ref <= 0:
            dec.rejected.append((o, "no_reference_price")); continue
        bp = px_book.get(o.ticker)
        if bp is not None and abs(ref - bp) / bp > limits.max_price_deviation:
            dec.rejected.append((o, "price_mismatch")); continue

        before_worst = env.worst(o.ticker)
        before_net = env.net_bounds()
        env.add(o.ticker, o.side, o.qty)
        env.px.setdefault(o.ticker, ref)
        # 股数是整数增减、两条腿只做加减 → 比较是精确的，不需要 epsilon 余量
        widened = env.worst(o.ticker) > before_worst

        def undo(reason: str) -> None:
            env.add(o.ticker, o.side, -o.qty)
            dec.rejected.append((o, reason))

        if daily_halt and widened:
            undo("daily_loss_halt"); continue
        if not limits.allow_short and env.short_leg(o.ticker) < -DUST_QTY:
            undo("would_short"); continue

        if o.purpose != PURPOSE_FLATTEN:
            adv = _finite(ctx.adv_usd.get(o.ticker))
            if adv is None or adv <= 0:
                undo("no_adv"); continue
            if (round(o.notional, _USD_DIGITS)
                    > round(limits.max_participation_pct * adv, _USD_DIGITS)):
                undo("exceeds_adv_participation"); continue

        if widened:
            p = env.px[o.ticker]
            if round(env.worst(o.ticker) * p / equity, _W_DIGITS) > limits.max_name_weight:
                undo("exceeds_name_limit"); continue
            g = env.gross()
            if g is None:
                undo("gross_unpriceable"); continue
            if round(g / equity, _W_DIGITS) > limits.max_gross:
                undo("exceeds_gross_limit"); continue
            if sector_cap is not None:
                sec = sector_of.get(o.ticker)
                members = ([tk for tk in env.names() if sector_of.get(tk) == sec]
                           if sec is not None else [o.ticker])
                s = env.sector(members)
                if s is None:
                    undo("sector_unpriceable"); continue
                if round(s / equity, _W_DIGITS) > sector_cap:
                    undo("exceeds_sector_limit"); continue
        if limits.max_net is not None:
            nb = env.net_bounds()
            if nb is None:
                undo("net_unpriceable"); continue
            hi, lo = nb
            # 只拦**把净敞口推得更远**的单：账户已经超限时，减仓单必须放行
            worse = (before_net is None
                     or max(abs(hi), abs(lo)) > max(abs(before_net[0]), abs(before_net[1])))
            if worse and round(max(abs(hi), abs(lo)) / equity, _W_DIGITS) > limits.max_net:
                undo("exceeds_net_limit"); continue

        if o.side == "BUY":
            cost = float(o.qty) * float(o.limit_price)
            if power is None or round(buy_spent + cost, _USD_DIGITS) > round(power, _USD_DIGITS):
                undo("insufficient_buying_power"); continue
            buy_spent += cost

        dec.approved.append(o)
    return dec
