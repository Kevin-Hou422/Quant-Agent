"""
order_intent.py — 盘后冻结的订单意图：券商计划、纸账本、研究回测共同消费的同一份数量合同

来由（2026-10-25 审计，固定提交 eca36e3，原话）："券商计划按 T 日参考价和当时权益冻结整数股数；
模拟引擎将昨日目标权重按 T+1 开盘市值重新解释为开盘目标持仓"——现金 10,000、T 收盘 100、目标 50%：
券商冻结 50 股，次日开盘 90、收盘 100，冻结订单净值 1.05，模拟却按开盘重算成约 55.56 股、净值 1.0556。
要求："首轮保留当前券商的'盘后冻结订单股数'合同，将其做成研究回测、纸账本、券商计划共同消费的
不可变 OrderIntent……撮合可以决定成交量和成交价，不能无声改变原始委托数量。"

合同（时钟 c2）：
    决策    T 收盘之后：数量 = trunc(目标权重 × T 收盘切片权益 / T 收盘参考价) − 现有股数（整股，向零取整），
            限价 = 参考价 × (1 ± 限价带) 按最小价位取整（买向下、卖向上）—— 就是券商的 `build_rebalance_orders`
    资金    全部买单的"数量 × 限价"超过可用资金（切片权益 − T 收盘持仓市值；未成交卖单的现金不预支）
            → **全部买单按同一比例缩量**（`scale_buys_to_power`，规则名 BUYING_POWER_RULE 进可执行规格的执行合同）；
            缩到 0 股的那张不发（insufficient_buying_power）—— 与下单门同一函数；模拟里只对纯多头账本适用
    有效    DAY 单，有效 session = T 之后的下一个交易日
    撮合    开盘在限价内 → 开盘价；否则当日最高 / 最低触及限价 → 限价；否则不成交（limit_not_reached）。
            日线 high/low 触及只是撮合近似：不代表能排到队、不保证成交。
            成交量受截至 T 的 ADV 参与率限制（participation），整股向零取整；**成交量不超过委托量**

`whole_shares=False`（研究里不按整股的回测）：数量按同一公式但不取整，其余规则相同。

两种形态、同一套规则：`freeze_rebalance` 产出逐单的 `OrderIntent`（整股时直接包装券商的 `build_rebalance_orders`）；
两个模拟引擎逐日调用数组版 `freeze_arrays`（逐单对象版让 1000 天 × 50 只的回测慢 30 倍）。两者逐股、逐限价、
逐张资金缩量一致，由 tests/integration/test_frozen_order_intent_ledger.py 的随机对照钉住。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date
from typing import List, Mapping, Optional

import numpy as np
import pandas as pd

TIME_IN_FORCE = "DAY"
#: 股数从权重反推时的取整容差：整股账本里的持仓只经整股成交改变，反推出来离整数在这个范围内就是整数
SHARE_TOL = 1e-6
#: 可用资金不够付全部买单时怎么办。进可执行规格的 execution 段：规则一变就是另一份执行合同（规格哈希、验收证据都不继承）。
#:   reject_whole_order   （到 2026-10-30 为止）按代码顺序累计，超出的那张买单整张不发 —— 隔离评估：平均只投入 76–87% 权益
#:   scale_buys_pro_rata  全部买单按同一比例缩量、整股向下取整，再按"被舍掉的零头从大到小"逐股补回，直到再买一股就超资金
BUYING_POWER_RULE = "scale_buys_pro_rata"


@dataclass(frozen=True)
class OrderIntent:
    """一张盘后冻结的订单意图（不可变）。撮合只决定成交多少、成交价，不改这里的任何字段。"""
    client_id:     str
    ticker:        str
    side:          str                    # "BUY" | "SELL"
    qty:           float                  # 正数；整股账本里是整数
    ref_price:     float                  # 决策日收盘参考价
    limit_price:   float                  # 已按最小价位取整
    decision_date: Optional[date]
    session:       Optional[date]         # 有效交易 session（决策日之后的下一个交易日）
    book_id:       int = 0
    slice_equity:  float = 0.0            # 冻结时的切片权益
    spec_hash:     str = ""
    decision_time: Optional[str] = None
    time_in_force: str = TIME_IN_FORCE
    current_qty:   float = 0.0
    target_qty:    float = 0.0
    target_weight: float = 0.0
    purpose:       str = "rebalance"

    @property
    def signed_qty(self) -> float:
        return self.qty if self.side == "BUY" else -self.qty


@dataclass
class IntentPlan:
    intents:  List[OrderIntent] = field(default_factory=list)
    rejected: List[dict] = field(default_factory=list)     # 冻结时就不发的（资金不足）
    skipped:  List[dict] = field(default_factory=list)     # 不需要 / 不能下（无参考价、无交易带、碎股差额）
    resized:  List[dict] = field(default_factory=list)     # 资金不足缩了量的买单（原数量 → 缩后数量）


def shares_from_weights(weights: np.ndarray, equity: float, prices: np.ndarray, whole_shares: bool) -> np.ndarray:
    """权重（相对 `equity`）→ 股数。整股账本里反推值离整数 ≤ SHARE_TOL 的按整数（浮点回推的噪声）。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        sh = np.where(np.isfinite(prices) & (prices > 0), weights * equity / prices, 0.0)
    sh = np.nan_to_num(sh, nan=0.0)
    if whole_shares:
        r = np.round(sh)
        sh = np.where(np.abs(sh - r) <= SHARE_TOL * np.maximum(1.0, np.abs(r)), r, sh)
    return sh


def scale_buys_to_power(qty, limit, power: Optional[float], whole_shares: bool, names) -> np.ndarray:
    """
    可用资金 `power` 不够付全部买单（Σ 数量 × 限价）时的缩量（BUYING_POWER_RULE）。够付 → 原样返回。
        不按整股   每张 × power / 需要的钱
        整股       每张 floor(数量 × 比例)；剩下的钱按"被舍掉的零头从大到小、零头相同按代码"逐张各补一股，买不起的跳过
    比例对所有买单相同：不因为代码排在后面就整张买不到。结果不超过原数量，Σ 数量 × 限价 ≤ power（按 6 位小数比较）。
    power 缺失 / 非有限 / ≤ 0 → 全部为 0。
    """
    q = np.asarray(qty, dtype=float)
    lim = np.asarray(limit, dtype=float)
    if not len(q):
        return q.copy()
    if power is None or not math.isfinite(float(power)) or float(power) <= 0:
        return np.zeros(len(q))
    power = float(power)
    need = float(np.sum(q * lim))
    if round(need, 6) <= round(power, 6):
        return q.copy()
    exact = q * (power / need)
    if not whole_shares:
        return exact
    out = np.floor(exact + 1e-9)
    spent = float(np.sum(out * lim))
    for i in sorted(range(len(q)), key=lambda j: (-round(float(exact[j] - out[j]), 9), str(names[j]))):
        if out[i] < q[i] and round(spent + float(lim[i]), 6) <= round(power, 6):
            out[i] += 1.0
            spent += float(lim[i])
    return out


def freeze_rebalance(target_weights: pd.Series, current_qty: Mapping[str, float], ref_prices: pd.Series,
                     equity: float, decision_date, *, band_bps: float, book_id: int = 0,
                     whole_shares: bool = True, buying_power: Optional[float] = None, spec_hash: str = "",
                     decision_time: Optional[str] = None, session=None, allow_short: bool = False,
                     no_trade_band: float = 0.0, name_cap: Optional[float] = None,
                     gross_cap: Optional[float] = None) -> IntentPlan:
    """
    决策日收盘后冻结调仓意图。整股时数量与限价就是券商 `build_rebalance_orders` 给出的（同一个函数）；
    `buying_power` 给了就按下单门的规则把买单缩到买得起（缩到 0 的记进 rejected，缩了量的记进 resized）。
    权益 ≤ 0 → 不冻结任何意图（不交易）。
    """
    from app.core.execution.order_builder import build_rebalance_orders, round_to_tick
    plan = IntentPlan()
    if not (equity is not None and math.isfinite(float(equity)) and float(equity) > 0):
        return plan
    d = pd.Timestamp(decision_date).date() if decision_date is not None else None
    sess = pd.Timestamp(session).date() if session is not None else None
    common = dict(decision_date=d, session=sess, book_id=int(book_id), slice_equity=float(equity),
                  spec_hash=str(spec_hash or ""), decision_time=decision_time)
    if whole_shares:
        built = build_rebalance_orders(target_weights, current_qty, ref_prices, float(equity),
                                       d or date(1970, 1, 1), book_id=book_id, band_bps=band_bps,
                                       allow_short=allow_short, no_trade_band=no_trade_band,
                                       name_cap=name_cap, gross_cap=gross_cap)
        plan.skipped = list(built.skipped)
        raw = [OrderIntent(client_id=o.client_id, ticker=o.ticker, side=o.side, qty=float(o.qty),
                           ref_price=float(o.ref_price), limit_price=float(o.limit_price),
                           current_qty=float(o.current_qty), target_qty=float(o.target_qty),
                           target_weight=float(o.target_weight), purpose=o.purpose, **common)
               for o in built.orders]
    else:
        band = float(band_bps) * 1e-4
        raw, sells, buys = [], [], []
        names = sorted(set(target_weights.index) | {k for k, v in current_qty.items() if abs(float(v)) > 0})
        for tk in names:
            w = float(target_weights.get(tk, 0.0))
            w = w if math.isfinite(w) else 0.0
            if w < 0 and not allow_short:
                w = 0.0
            px = ref_prices.get(tk, np.nan)
            cur = float(current_qty.get(tk, 0.0))
            if not (pd.notna(px) and math.isfinite(float(px)) and float(px) > 0):
                plan.skipped.append({"ticker": tk, "reason": "no_reference_price", "current_qty": cur,
                                     "target_weight": w})
                continue
            px = float(px)
            target = w * float(equity) / px
            delta = target - cur
            if abs(delta) <= 1e-12:
                continue
            side = "BUY" if delta > 0 else "SELL"
            limit = round_to_tick(px * (1 + band) if side == "BUY" else px * (1 - band), side)
            o = OrderIntent(client_id=f"sim-{book_id}-{d}-{tk}-{side[0]}", ticker=tk, side=side, qty=abs(delta),
                            ref_price=px, limit_price=limit, current_qty=cur, target_qty=target,
                            target_weight=w, **common)
            (sells if side == "SELL" else buys).append(o)
        raw = sells + buys
    long_only = all(i.target_qty >= 0 and i.current_qty >= 0 for i in raw)
    buys = [i for i in raw if i.side == "BUY"]
    scaled = {}
    if buys and buying_power is not None and long_only:
        new = scale_buys_to_power([i.qty for i in buys], [i.limit_price for i in buys], buying_power, whole_shares,
                                  [i.ticker for i in buys])
        scaled = {i.ticker: float(n) for i, n in zip(buys, new)}
    for i in raw:
        n = scaled.get(i.ticker, i.qty) if i.side == "BUY" else i.qty
        if n <= (0.0 if whole_shares else 1e-12):
            plan.rejected.append({"ticker": i.ticker, "qty": i.qty, "reason": "insufficient_buying_power"})
            continue
        if n != i.qty:
            plan.resized.append({"ticker": i.ticker, "qty": i.qty, "resized_qty": n,
                                 "reason": "insufficient_buying_power"})
            i = replace(i, qty=n)
        plan.intents.append(i)
    return plan


def clamp_filled_shares(filled_delta: np.ndarray, intent_delta: np.ndarray, whole_shares: bool) -> np.ndarray:
    """撮合给出的成交股数：与委托同向、不超过委托量；整股账本向零取整（浮点噪声按整数）。"""
    d = np.where(np.sign(filled_delta) == np.sign(intent_delta), filled_delta, 0.0)
    d = np.sign(d) * np.minimum(np.abs(d), np.abs(intent_delta))
    if whole_shares:
        r = np.round(d)
        d = np.where(np.abs(d - r) <= SHARE_TOL * np.maximum(1.0, np.abs(r)), r, np.trunc(d))
    return d


def _tick_round(raw: np.ndarray, buy: np.ndarray) -> np.ndarray:
    """`order_builder.round_to_tick` 的数组版：≥$1 按 0.01、<$1 按 0.0001；买向下、卖向上。"""
    tick = np.where(raw >= 1.0, 0.01, 0.0001)
    n = np.where(buy, np.floor(raw / tick + 1e-9), np.ceil(raw / tick - 1e-9))
    return np.round(n * tick, 4)


def freeze_arrays(target_w: np.ndarray, cur_sh: np.ndarray, ref_px: np.ndarray, equity: float, band: float,
                  whole_shares: bool, buying_power: Optional[float], names: List[str],
                  info: Optional[dict] = None) -> tuple:
    """
    `freeze_rebalance` 的数组版（两个模拟引擎逐日调用；逐单对象版太慢）。同一套规则：数量 = trunc(w·E/ref) − 现有
    （整股时 floor），限价按最小价位取整，纯多头时买单按 `scale_buys_to_power` 缩到可用资金之内。
    只处理参考价有效的名字（调用方保证）。返回 (有符号委托股数, 限价, 被资金规则缩到 0 的布尔掩码)。
    `info` 给了就写进 info["resized"]：资金不足缩了量（没缩到 0）的布尔掩码。
    两版逐股一致由 tests/integration/test_frozen_order_intent_ledger.py 的随机对照钉住。
    """
    w = np.nan_to_num(np.asarray(target_w, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    cur = np.asarray(cur_sh, dtype=float)
    px = np.asarray(ref_px, dtype=float)
    n = len(w)
    qty = np.zeros(n)
    limit = np.full(n, np.nan)
    rejected = np.zeros(n, dtype=bool)
    if not (equity is not None and math.isfinite(float(equity)) and float(equity) > 0):
        return qty, limit, rejected
    with np.errstate(divide="ignore", invalid="ignore"):
        target = w * float(equity) / px
    if whole_shares:
        target = np.trunc(target)
        delta = target - cur
        q = np.floor(np.abs(delta) + 1e-9)
        active = (delta != 0) & (q > 0)
    else:
        delta = target - cur
        q = np.abs(delta)
        active = q > 1e-12
    buy = delta > 0
    raw = np.where(buy, px * (1 + band), px * (1 - band))
    lim = _tick_round(np.where(active, raw, 1.0), buy)
    qty = np.where(active, np.sign(delta) * q, 0.0)
    limit = np.where(active, lim, np.nan)
    if buying_power is not None and active.any() and bool(np.all((target[active] >= 0) & (cur[active] >= 0))):
        idx = np.flatnonzero(active & buy)
        new = scale_buys_to_power(q[idx], lim[idx], buying_power, whole_shares, [names[j] for j in idx])
        gone = new <= (0.0 if whole_shares else 1e-12)
        rejected[idx[gone]] = True
        if info is not None:
            info["resized"] = np.zeros(n, dtype=bool)
            info["resized"][idx[~gone & (new != q[idx])]] = True
        qty[idx] = np.where(gone, 0.0, new)
        limit = np.where(rejected, np.nan, limit)
    return qty, limit, rejected


def limit_fill_array(signed_qty: np.ndarray, limit: np.ndarray, open_: np.ndarray, high: np.ndarray,
                     low: np.ndarray) -> np.ndarray:
    """`limit_fill_price` 的数组版：没有委托 / 不成交 / 价格缺失或非正 → NaN。"""
    o, h, lo = (np.asarray(x, dtype=float) for x in (open_, high, low))
    good = (np.isfinite(limit) & (limit > 0) & np.isfinite(o) & (o > 0) & np.isfinite(h) & (h > 0)
            & np.isfinite(lo) & (lo > 0) & (signed_qty != 0))
    buy = signed_qty > 0
    px = np.where(buy, np.where(o <= limit, o, np.where(lo <= limit, limit, np.nan)),
                  np.where(o >= limit, o, np.where(h >= limit, limit, np.nan)))
    return np.where(good, px, np.nan)


@dataclass
class OpenPlan:
    """一天开盘前冻结的委托（数组形式，给两个模拟引擎用）。"""
    signed_qty: np.ndarray
    limit: np.ndarray
    rejected: List[dict] = field(default_factory=list)
    resized: List[dict] = field(default_factory=list)


def open_targets(target_w, prev_w, pre, equity, v_open, ref_px, o_eff, open_, high, low, tickers,
                 decision_day, session_day, band, whole_shares, tradable=None):
    """
    两个模拟引擎（研究回测、纸账本）开盘撮合前的一步：T 收盘冻结意图 → (调仓前权重, 意图全部成交后的权重,
    各名成交价, 冻结计划)，权重都相对开盘市值 `v_open`。`prev_w` 相对 T 收盘权益 `equity`，`pre` 是漂移到开盘的权重。
    能冻结的只有 T 收盘有正参考价、开盘有价（且 `tradable`）的名字；其余（停牌 / 退出数据集的旧仓）原样不交易。
    """
    names = [str(t) for t in tickers]
    ok = np.isfinite(ref_px) & (ref_px > 0) & np.isfinite(o_eff) & (o_eff > 0)
    if tradable is not None:
        ok &= np.asarray(tradable, dtype=bool)
    cur_sh = shares_from_weights(np.where(ok, prev_w, 0.0), equity, ref_px, whole_shares)
    invested = float(np.nansum(np.abs(prev_w))) * float(equity)
    tw = np.where(ok, np.nan_to_num(np.asarray(target_w, dtype=float), nan=0.0), 0.0)
    px_ref = np.where(ok, ref_px, 1.0)
    info: dict = {}
    qty, limit, rej = freeze_arrays(np.where(ok, tw, 0.0), np.where(ok, cur_sh, 0.0), px_ref, equity, band,
                                    whole_shares, float(equity) - invested, names, info)
    qty, limit = np.where(ok, qty, 0.0), np.where(ok, limit, np.nan)
    plan = OpenPlan(signed_qty=qty, limit=limit,
                    rejected=[{"ticker": names[i], "reason": "insufficient_buying_power"}
                              for i in np.flatnonzero(rej & ok)],
                    resized=[{"ticker": names[i], "reason": "buying_power_resized"}
                             for i in np.flatnonzero(info.get("resized", np.zeros(len(names), dtype=bool)) & ok)])
    px = limit_fill_array(qty, limit, open_, high, low)
    pre_exact, after = np.array(pre, dtype=float), np.array(pre, dtype=float)
    if v_open > 0:
        pre_exact = np.where(ok, cur_sh * o_eff / v_open, pre_exact)
        after = np.where(ok, (cur_sh + qty) * o_eff / v_open, pre_exact)
    return pre_exact, after, px, plan


def whole_share_fills(fr, pre, target_after, v_open, o_eff, whole_shares):
    """撮合（含部分成交）给出的成交 → 股数与意图同向、不超过意图量，整股账本向零取整，再换回权重（改 `fr` 原地）。"""
    if not (v_open > 0):
        return fr
    with np.errstate(divide="ignore", invalid="ignore"):
        o = np.where(o_eff > 0, o_eff, np.nan)
        filled_sh = np.nan_to_num((fr.filled_w - pre) * v_open / o, nan=0.0)
        intent_sh = np.nan_to_num((target_after - pre) * v_open / o, nan=0.0)
        dw = np.nan_to_num(clamp_filled_shares(filled_sh, intent_sh, whole_shares) * o / v_open, nan=0.0)
    fr.filled_w, fr.filled_d = pre + dw, dw
    fr.unfilled_d = (target_after - pre) - dw
    return fr
