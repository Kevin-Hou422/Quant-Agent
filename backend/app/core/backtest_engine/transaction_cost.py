"""
Transaction Cost & Liquidity Model

组件：
  CostParams          — 成本参数数据类
  SlippageModel       — 平方根冲击法则 / 简化线性模型
  LiquidityConstraint — ADV 流动性上限截断
  TransactionCostEngine — 组合计算，返回滑点 + 净成本
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 成本参数
# ---------------------------------------------------------------------------

@dataclass
class CostParams:
    """
    回测成本参数配置。

    fixed_bps              : 单边固定手续费（bps，基点）
    min_ticket_fee         : 最小票面费（USD）
    spread_bps             : 买卖价差（bps）
    impact_coef            : 市场冲击系数（平方根法则）
    adv_window             : ADV 计算窗口（交易日）
    adv_cap_pct            : **持仓容量**上限 = adv_cap_pct × ADV（USD）。
                             这是**组合构建层**的约束（"这只票最多能装多少钱"），
                             容量不足时允许把额度换到别的票上（water-filling）。
    max_participation_pct  : **单日成交量**上限 = max_participation_pct × ADV（USD）。
                             这是**执行层**的约束（"今天最多能成交多少"），
                             约束的是 prev→target 的**交易差额**，不是目标持仓；
                             买不到**不得**把额度分配给别的票。

    ⚠️ 这两个参数在 2026-09-19 之前是**同一个**（`adv_cap_pct`）：文档与实现都写
    "持仓上限"，却被 PaperBroker 拿去裁剪目标持仓并生成带 `reject_reason="adv_cap"`
    的成交记录 —— 用持仓裁剪冒充部分成交。拆开是用户决策 #3。
    slippage_model         : 'sqrt'（平方根法则）或 'linear'（简化线性）
    short_borrow_annual_bps: 做空借券年化成本（bps）。每日从空头持仓净值扣除。
                             典型值：易借券 30–100bps，难借券可达 1000bps+。
                             默认 50bps ≈ 大盘股平均借券成本。
    """
    fixed_bps:               float = 5.0
    min_ticket_fee:          float = 1.0
    spread_bps:              float = 2.0
    impact_coef:             float = 0.1
    adv_window:              int   = 20
    adv_cap_pct:             float = 0.10
    max_participation_pct:   float = 0.10
    slippage_model:          Literal["sqrt", "linear"] = "sqrt"
    short_borrow_annual_bps: float = 50.0


# ---------------------------------------------------------------------------
# 滑点模型
# ---------------------------------------------------------------------------

class SlippageModel:
    """
    计算每笔交易的滑点（单位：bps）。

    sqrt 模式（平方根冲击法则）：
        slippage_bps = spread/2 + impact * σ * sqrt(trade_usd / adv_usd)

    linear 模式（简化线性）：
        slippage_bps = 0.5 * spread + 0.1 * vol_impact_proxy
    """

    def __init__(self, params: CostParams) -> None:
        self.p = params

    def compute(
        self,
        trade_usd:  np.ndarray,   # (N,) 每个资产的交易金额（USD，取绝对值）
        adv_usd:    np.ndarray,   # (N,) 20日 ADV（USD）
        daily_vol:  np.ndarray,   # (N,) 日收益率波动率
    ) -> np.ndarray:              # (N,) 滑点 bps
        """向量化计算，零交易量位置返回 0。"""
        trade_abs = np.abs(trade_usd)
        adv_safe  = np.where(adv_usd <= 0, np.nan, adv_usd)

        if self.p.slippage_model == "sqrt":
            participation = trade_abs / adv_safe          # 参与率
            slippage = (
                self.p.spread_bps / 2.0
                + self.p.impact_coef * daily_vol * 10_000   # vol → bps scale
                * np.sqrt(np.nan_to_num(participation, nan=0.0))
            )
        else:  # linear
            slippage = (
                0.5 * self.p.spread_bps
                + 0.1 * daily_vol * 10_000
            ) * np.ones_like(trade_abs)

        # 无交易时滑点为 0
        slippage = np.where(trade_abs == 0, 0.0, slippage)
        return np.nan_to_num(slippage, nan=0.0)


# ---------------------------------------------------------------------------
# ADV 流动性约束
# ---------------------------------------------------------------------------

def project_to_capped_l1(
    w:        np.ndarray,   # (T, N) 有符号权重
    cap:      np.ndarray,   # (T, N) 每名 |权重| 上限（>=0，可为 inf）
    target:   "float | np.ndarray" = 1.0,   # 目标 L1 范数；标量或逐行 (T,) / (T,1)
    max_iter: int   = 32,
    tol:      float = 1e-12,
) -> np.ndarray:
    """
    带上限的 L1 投影（Task 6.6，water-filling）。**这是组合构建层的原语。**

    对每一行求解：|w_i| <= cap_i 且 sum_i |w_i| == target（当预算 sum_i cap_i >= target
    时可行）；若预算不足则**保持 sum |w_i| = 预算 < target**，绝不通过整体放大把
    已触顶的权重推回超限（旧 "clip→整体 L1 归一化" 的缺陷 E-N1/F-N1）。

    算法：迭代 water-filling —— 反复将自由名按比例放大以补足亏空，任何被放大到
    超过 cap 的名固定在 cap 并移出自由集，直到收敛或达 max_iter。方向（符号）保留。

    ⚠️ **不要用它来模拟成交。** 它的 water-filling 会把某只票被削掉的额度
    **再分配**给其他票 —— 在构建目标时这是对的（容量不足就换个票装），
    在执行时这是错的（券商不会因为 A 买不到就多买 B）。
    执行侧请用 `simulate_partial_fills()`。

    `target` 支持**逐行**给值：`apply_capacity` 这类按 (T,N) 批处理的调用
    必须传 `np.abs(w).sum(axis=1)`，否则 gross≠1 的输入会被整体放大到 1
    （缺陷 A-6 的形态）。
    """
    sign = np.sign(w)
    a    = np.abs(w).astype(float)
    cap  = np.abs(cap).astype(float)

    tgt = np.asarray(target, dtype=float)
    if tgt.ndim == 1:
        tgt = tgt.reshape(-1, 1)                                  # (T,) → (T,1)

    # 每行可达的最大 L1 = min(target, 预算)
    budget    = np.nansum(np.where(np.isfinite(cap), cap, a), axis=1, keepdims=True)
    row_target = np.minimum(tgt, budget)                          # (T, 1)

    a = np.minimum(a, cap)                                         # 初始截断
    for _ in range(max_iter):
        capped   = a >= cap - tol                                 # 已触顶（含 cap=inf 时永不触顶）
        capped  &= np.isfinite(cap)
        cap_mass  = np.nansum(np.where(capped, a, 0.0), axis=1, keepdims=True)
        free_mass = np.nansum(np.where(capped, 0.0, a), axis=1, keepdims=True)
        deficit   = row_target - (cap_mass + free_mass)           # >0 需放大自由名
        if np.all(np.abs(deficit) < tol):
            break
        # 自由名按比例吸收亏空；free_mass≈0 时无法再分配 → 停
        safe_free = np.where(free_mass > tol, free_mass, 1.0)     # 避免 0 除（分支被 where 丢弃）
        scale = np.where(free_mass > tol, (free_mass + deficit) / safe_free, 1.0)
        a = np.where(capped, a, a * scale)
        a = np.minimum(a, cap)                                    # 放大后可能触顶，再截断

    return sign * a


@dataclass
class FillResult:
    """一次撮合的结果。`desired_d = filled_d + unfilled_d` 恒成立（逐名）。"""
    filled_w:   np.ndarray   # 成交后的**持仓**权重
    filled_d:   np.ndarray   # 实际成交的权重变动（有符号）
    desired_d:  np.ndarray   # 想要的权重变动（有符号）
    unfilled_d: np.ndarray   # 未成交的部分（有符号）
    group_frac: np.ndarray   # 每名所属订单组的**共同可执行比例** φ ∈ [0,1]
    scaled_by:  float        # 为满足组合级约束对**全部**交易的整体缩放 μ ∈ [0,1]
    deferred:   bool         # μ == 0 且确实有交易意图（整组暂缓）
    #: 目标账本自身就违反组合约束 —— 这是**构建层**的问题，执行层补不了。
    target_violates: bool = False
    #: 连"什么都不交易"都仍然违规（昨仓已超限）—— 必须由构建层产出新目标。
    book_non_compliant: bool = False


def _largest_feasible_scale(n0: float, s: float, limit: float,
                            tol: float = 1e-9) -> "float | None":
    """
    求 `|n0 + mu*s| <= limit` 在 `mu in [0,1]` 上的**最大**可行解；无解返回 None。

    `n0 + mu*s` 对 mu 是线性的，约束区间也就是一段区间，与 [0,1] 取交后取右端点。
    独立成函数是为了能被单独测：它决定"部分成交到什么程度才不留下超限敞口"。
    """
    if abs(s) <= tol:                       # 交易不改变净敞口
        return 1.0 if abs(n0) <= limit + tol else None
    lo_raw = (-limit - n0) / s
    hi_raw = (limit - n0) / s
    lo, hi = (lo_raw, hi_raw) if s > 0 else (hi_raw, lo_raw)
    lo, hi = max(lo, 0.0), min(hi, 1.0)
    return hi if hi >= lo - tol else None


def simulate_partial_fills(
    prev_w:      np.ndarray,       # (N,) 昨仓权重
    target_w:    np.ndarray,       # (N,) 目标持仓权重
    cap_trade_w: np.ndarray,       # (N,) 本日**可成交量**上限（权重口径，>=0，可为 inf）
    *,
    max_net:     "float | None" = None,   # 组合净敞口上限（|sum w|）。None = 明确不限制
    groups:      "Sequence | None" = None,  # (N,) 订单组标签；同组必须按**原始比例**成交
    net_tol:     float = 1e-9,
) -> FillResult:
    """
    执行层撮合：**逐名按可成交量部分成交，绝不把未成交额度分配给别的标的。**

    分工（2026-09-19 定，用户决策 #1）：

      · 组合构建层：容量不足时可以 water-filling **换个票装**，产出的是**新目标**，
        并且要重新过风控。
      · 执行层（本函数）：**一个标的受限，不得擅自扩大另一个标的的目标**。
        买不到就是买不到，未成交量如实记账。

    三步：

      1. **订单组的共同可执行比例 φ**（用户决策 #2 之二）。
         对明确要求保持比例的订单组，φ 从**原始订单**算：
         `φ_g = min_i(cap_i / |desired_i|)`，全组同乘 φ_g。
         未分组的名各自独立（等价于逐名裁剪）。

         为什么不能"先逐名裁剪再整体乘 λ"：那样保持的是**裁剪后**的比例。
         `[+0.90, -0.10]` 被裁成 `[+0.01, -0.10]` 时 9:1 已经反转成 0.1:1，
         之后再怎么等比缩放都救不回来。

      2. **组合级约束 μ**。对**全部**交易（含减仓）整体缩放。

         为什么减仓不能豁免（用户决策 #2 之一）：**单标的减仓 ≠ 组合降风险**。
         `[+0.30, -0.20]`（net +0.10）平掉空头是逐名"降风险"，
         但组合净敞口升到 +0.30 —— 上一版把减仓当基准无条件放行，
         于是 `max_net=0.15` 被突破而 λ 还报 1.0。

      3. **区分"谁的问题"**。`|sum(target)| > max_net` 说明**构建层**产出的目标
         本身就违规，执行层补不了 → `target_violates`；
         连不交易都仍超限（昨仓已违规）→ `book_non_compliant`。
         两种都必须由构建层产出新目标并重新过风控，不是靠执行层少成交来掩盖。

    ⚠️ **净敞口约束不能代替对冲比例约束**：上例 `[+0.01, -0.10]` 的 `|net|=0.09`
    满足 `max_net=0.10`，但比例已经反转。要保比例就必须传 `groups`。
    """
    prev_w = np.asarray(prev_w, dtype=float)
    target_w = np.asarray(target_w, dtype=float)
    cap = np.abs(np.asarray(cap_trade_w, dtype=float))
    n = prev_w.shape[0]

    desired = target_w - prev_w

    # ---- 1) 逐组共同可执行比例 φ（未分组 = 各自独立）----
    labels = list(groups) if groups is not None else [None] * n
    assert len(labels) == n, "groups 长度必须与权重向量一致"
    phi = np.ones(n, dtype=float)
    buckets: dict = {}
    for i, g in enumerate(labels):
        buckets.setdefault(("_solo", i) if g is None else ("_grp", g), []).append(i)
    for key, idx in buckets.items():
        ratios = [cap[i] / abs(desired[i]) for i in idx if abs(desired[i]) > net_tol]
        f = min(ratios) if ratios else 1.0
        phi[idx] = min(1.0, max(0.0, f))
    f0 = phi * desired

    # ---- 2) 组合级约束 μ（作用于**全部**交易）----
    mu, target_violates, non_compliant = 1.0, False, False
    if max_net is not None:
        target_violates = abs(float(np.sum(target_w))) > max_net + net_tol
        got = _largest_feasible_scale(float(np.sum(prev_w)), float(np.sum(f0)),
                                      max_net, net_tol)
        if got is None:
            mu = 0.0
            non_compliant = abs(float(np.sum(prev_w))) > max_net + net_tol
        else:
            mu = got

    filled_d = mu * f0
    return FillResult(
        filled_w=prev_w + filled_d,
        filled_d=filled_d,
        desired_d=desired,
        unfilled_d=desired - filled_d,
        group_frac=phi,
        scaled_by=float(mu),
        deferred=bool(mu == 0.0 and np.any(np.abs(desired) > net_tol)),
        target_violates=bool(target_violates),
        book_non_compliant=bool(non_compliant),
    )


class LiquidityConstraint:
    """
    将权重矩阵中超过 ADV 上限的持仓截断，并重新归一化。

    上限规则：
        max_position_usd[t, i] = adv_cap_pct × adv_usd[t, i]
        weight_cap[t, i]        = max_position_usd[t, i] / portfolio_value
    """

    def __init__(self, params: CostParams) -> None:
        self.p = params

    def compute_adv(
        self,
        volume: pd.DataFrame,   # (T×N) 成交量（股数）
        prices: pd.DataFrame,   # (T×N) 收盘价
    ) -> pd.DataFrame:
        """计算 20日 ADV（USD），shape=(T×N)。"""
        dollar_vol = volume * prices                                   # USD 成交额
        adv = dollar_vol.rolling(
            window=self.p.adv_window,
            min_periods=max(1, self.p.adv_window // 2),
        ).mean()
        # Task 6.7（F-N2 修复）：改用前向填充。旧 `bfill()` 会用未来成交量回填早期
        # ADV（喂给流动性上限与滑点分母）→ 前视。ffill 只用历史；起始首行无历史时
        # 用当日 dollar_vol 兜底（无前视），仍无则 0（视为不可交易 → 零权重）。
        return adv.ffill().fillna(dollar_vol).fillna(0.0)

    def apply(
        self,
        weights: pd.DataFrame,  # (T×N) 目标权重
        adv_usd: pd.DataFrame,  # (T×N) ADV（USD）
        portfolio_value: float,
    ) -> pd.DataFrame:
        """
        截断超 ADV 上限的权重，并重新做 L1 归一化。
        返回调整后的权重矩阵。
        """
        w   = weights.to_numpy(dtype=float).copy()
        adv = adv_usd.to_numpy(dtype=float)

        # 上限（以权重表示）
        max_usd = adv * self.p.adv_cap_pct                      # (T, N)
        cap_w   = np.where(
            portfolio_value > 0,
            max_usd / portfolio_value,
            np.inf,
        )

        # Task 6.6：迭代投影（water-filling），保证归一化后仍不超 ADV 上限。
        # 旧实现 "clip → 整体 L1 归一化" 会把已触顶的权重推回超限（E-N1/F-N1）。
        #
        # 【缺陷 A-7，2026-09-21 修】target 原先写死 1.0。`project_to_capped_l1`
        # 的 `row_target = min(target, budget)`，而 budget 在 cap **有限**时是
        # Σcap（ADV 上限通常远大于 1），于是 row_target 恒为 1.0 —— 上游
        # gross≠1 的输入会被整体**放大**回 L1=1。本函数的职责是"削掉超 ADV 的
        # 部分"，不是"把敞口补满"；上游若已降过敞口（风控/信号弱/部分空仓），
        # 那是一个决定，这里无权抹掉。这正是 A-6 在执行层的同型错误。
        #
        # 传逐行真实 gross：容量充足时原样保敞口，容量不足时自然降到 budget。
        row_gross = np.abs(w).sum(axis=1)
        projected = project_to_capped_l1(w, cap_w, target=row_gross)

        return pd.DataFrame(projected, index=weights.index, columns=weights.columns)


# ---------------------------------------------------------------------------
# TransactionCostEngine（组合入口）
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    date:         pd.Timestamp
    ticker:       str
    direction:    str      # 'BUY' / 'SELL'
    shares:       float
    price:        float
    slippage_bps: float
    cost_usd:     float
    net_price:    float


class TransactionCostEngine:
    """
    给定 delta_weights（换手量）、价格、ADV、日波动率，
    计算每笔交易的滑点和总成本，并生成 trade_log。
    """

    def __init__(self, params: CostParams) -> None:
        self.p        = params
        self._slippage = SlippageModel(params)

    def compute(
        self,
        date:          pd.Timestamp,
        delta_w:       np.ndarray,   # (N,) 权重变化（有符号）
        prices:        np.ndarray,   # (N,) 当日收盘价
        adv_usd:       np.ndarray,   # (N,) 20日 ADV（USD）
        daily_vol:     np.ndarray,   # (N,) 日收益率波动率
        portfolio_val: float,
        tickers:       list[str],
    ) -> Tuple[np.ndarray, float, List[TradeRecord]]:
        """
        Returns
        -------
        net_cost_weight : (N,) 每个资产的成本（以权重单位，从净值扣除）
        total_cost_usd  : float 当日总成本（USD）
        records         : List[TradeRecord]
        """
        trade_usd = delta_w * portfolio_val          # (N,) 交易金额

        # 滑点
        slip_bps  = self._slippage.compute(trade_usd, adv_usd, daily_vol)

        # 固定手续费（单边）
        notional   = np.abs(trade_usd)
        fixed_cost = notional * self.p.fixed_bps * 1e-4
        slip_cost  = notional * slip_bps          * 1e-4

        # 最小票面费
        ticket_fee = np.where(
            notional > 0,
            np.maximum(fixed_cost + slip_cost, self.p.min_ticket_fee),
            0.0,
        )

        total_cost_usd = float(ticket_fee.sum())

        # 构建 trade_log 记录
        records: List[TradeRecord] = []
        for i, (dw, p, s_bps, cost) in enumerate(
            zip(delta_w, prices, slip_bps, ticket_fee)
        ):
            if abs(dw) < 1e-10:
                continue
            direction = "BUY" if dw > 0 else "SELL"
            shares    = abs(dw) * portfolio_val / p if p > 0 else 0.0
            net_price = p * (1 + (s_bps * 1e-4) * (1 if dw > 0 else -1))
            records.append(TradeRecord(
                date=date, ticker=tickers[i],
                direction=direction, shares=shares,
                price=p, slippage_bps=float(s_bps),
                cost_usd=float(cost), net_price=float(net_price),
            ))

        # 成本归一化为权重单位（从净值中扣除）
        net_cost_w = ticket_fee / max(portfolio_val, 1.0)
        return net_cost_w, total_cost_usd, records

    @staticmethod
    def records_to_df(records: List[TradeRecord]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(columns=[
                "date", "ticker", "direction", "shares",
                "price", "slippage_bps", "cost_usd", "net_price",
            ])
        return pd.DataFrame([r.__dict__ for r in records])
