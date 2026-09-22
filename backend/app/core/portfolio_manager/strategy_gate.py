"""
strategy_gate.py — 策略级门控 + 因子按边际贡献准入（Phase PM.S1 / PM.S2）

核心思想（用户："要交易策略，不要回测漂亮的因子"）
    因子的价值 = 它对**策略**的**边际贡献**（边际 IR ∝ √(1−ρ²)）。对单因子加高门槛，恰恰筛掉
    "单独平庸但与组合低相关、分散化极好"的因子——选出一堆"单独漂亮的因子"而非"好策略"，
    还常常"没有因子过门 → 无可交易"。**所以严门加在你真正交易的那个组合策略上，不加在单因子上。**

本模块两件事
    PM.S1 `StrategyGate`：把 N 个因子经 PortfolioManager 合成**一个组合策略** → 对**策略的净收益**
          跑验证门（分段 OOS 全为正 + DSR 去膨胀 + 夏普 t≥门槛）。复用现有回测/统计引擎，不新造金融逻辑。
    PM.S2 `marginal_factor_selection`：贪心前向选择——候选加入策略后若**策略 OOS（扣成本）Sharpe 提升**
          才纳入，否则不纳入。单独弱但分散化好的因子能进；一堆平庸因子的组合也能过策略门。

诚实边界
    - 合成权重（AlphaCombiner）在 IS 拟合；本门的分段 OOS/DSR 是对**已合成策略**的时序稳健性检验，
      不能替代"合成权重本身的样本外性"。发现路径的三段切割（S.1/S.2）负责后者；本门是策略级补充。
    - fail-closed：任何环节抛错 → 判不通过（不静默放行，DEV_LESSONS §B）。
LLM 不参与。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WidePanel = Dict[str, pd.DataFrame]
Signals = Dict[str, pd.DataFrame]


# ---------------------------------------------------------------------------
# 策略净收益（复用 PortfolioManager + BacktestEngine，与单因子门同一成本口径）
# ---------------------------------------------------------------------------

# 昂贵推导（Corwin-Schultz 跑全 panel）的记忆化。
# 必须缓存：`marginal_factor_selection` 是 O(n²) 次调用、PBO 又按因子循环，
# 若每次重算会把全量回归从 ~5 分钟拖到 100 分钟（实测踩过，见 DEV_LESSONS）。
_DERIVE_CACHE: Dict[tuple, object] = {}
_CACHE_MAX = 64


#: 成本推导**实际读取**的字段。缓存键必须覆盖它们全部。
#:
#: 【缺陷 N-1，2026-09-20 修】原来的指纹只取 `close` —— 而 Corwin-Schultz 价差
#: 用的是 **high/low**，市场冲击用的是 **volume**。于是 close 相同、high/low 不同的
#: 两个数据集命中同一条缓存：外部审计实测真实值 1919.83 bps 被缓存里的 37.47 bps
#: 顶替（差 51 倍），而调用方完全看不出来。
#:
#: 这里显式列字段，而不是"对整个 dataset 求指纹"：后者会把与成本无关的字段
#: （sector、returns…）也算进去，缓存命中率无谓地掉。字段增删时这张表要同步 ——
#: 由 `test_cost_cache_key_covers_every_field_the_derivation_reads` 机械核对。
_COST_INPUT_FIELDS = ("close", "high", "low", "volume")


def _panel_fingerprint(df) -> tuple:
    """单张面板的 O(N) 指纹：形状 + 索引端点 + 首末行和 + 前 4 个列名。"""
    arr = df.to_numpy()
    try:
        head = float(np.nansum(arr[0])) if len(arr) else 0.0
        tail = float(np.nansum(arr[-1])) if len(arr) else 0.0
    except Exception as exc:
        # 指纹退化只会削弱缓存区分度（不会致错），但仍记录，避免"缓存突然总不命中"无从查起
        logger.debug("[strategy_gate] 缓存指纹计算失败，退化为形状级键: %s", exc)
        head = tail = 0.0
    idx = df.index
    return (df.shape, str(idx[0]) if len(idx) else "",
            str(idx[-1]) if len(idx) else "",
            round(head, 6), round(tail, 6), tuple(map(str, df.columns[:4])))


def _cache_key(kind: str, dataset: WidePanel, aum: float) -> tuple:
    """
    内容指纹，**不用 id()**：CPython 会在对象被回收后复用地址，于是
    "同 shape + 同 aum + 恰好复用了地址" 的另一个数据集会命中旧条目 ——
    缓存里存的是**成本参数与无交易带**，串号意味着 A 数据集的成本被用到 B 的回测上。
    （实测表现为全量跑时 test_phase_pm_s 偶发失败，单独跑必过。）

    指纹覆盖 `_COST_INPUT_FIELDS` 的**每一张**面板，外加**券商档位**
    —— 后者决定佣金与最小票面费，换个券商成本就不同，原来也不在键里。
    """
    from app.config import settings

    parts: List[tuple] = []
    for name in _COST_INPUT_FIELDS:
        df = dataset.get(name)
        parts.append((name,) + (_panel_fingerprint(df) if df is not None else ("absent",)))
    broker = str(getattr(settings, "trading_broker_profile", "")
                 or getattr(settings, "broker_profile", ""))
    acct = str(getattr(settings, "trading_account_type", ""))
    return (kind, float(aum), broker, acct, tuple(parts))


def _cache_put(key: tuple, val):
    if len(_DERIVE_CACHE) > _CACHE_MAX:
        _DERIVE_CACHE.clear()
    _DERIVE_CACHE[key] = val
    return val


def resolve_band(dataset: WidePanel, aum: float) -> float:
    """无交易带：配置优先，否则由 TradingContext 推导（结果缓存）。"""
    try:
        from app.config import settings
        band = float(getattr(settings, "pm_no_trade_band", 0.0))
        if band > 0.0:
            return band
        key = _cache_key("band", dataset, aum)
        if key in _DERIVE_CACHE:
            return float(_DERIVE_CACHE[key])       # type: ignore[arg-type]
        from app.core.trading_context import TradingContext
        return float(_cache_put(
            key, float(TradingContext(aum=aum).analyze(dataset).rebalance_band)))  # type: ignore[arg-type]
    except Exception as exc:
        # 静默返回 0 会让"无交易带推导失败"与"带就是 0"无法区分，
        # 换手率会被高估且无人知晓（审计 #9 同型）。
        logger.warning("[strategy_gate] 无交易带推导失败，本轮按 0 处理: %s", exc)
        return 0.0


def resolve_cost_params(dataset: WidePanel, aum: float, cost_params=None):
    """
    成本参数解析（修 DEV_LESSONS §J 在策略层的复发）：
    `None` **绝不退回机构默认** `CostParams()`（min_ticket=$1 / fixed=5bps）——那在 $10k 散户账户上
    是每天 ~0.2% 的假性流失，会把任何策略碾成 Sharpe −15，让系统永远选不出可交易策略。
    改为按**数据 + 配置券商 + 真实 AUM** 推导（TR.3 `grounded_cost_params`，moomoo 佣金=0）。结果缓存。
    """
    if cost_params is not None:
        return cost_params
    key = _cache_key("cost", dataset, aum)
    if key in _DERIVE_CACHE:
        return _DERIVE_CACHE[key]
    try:
        from app.config import settings
        from app.core.trading_context import grounded_cost_params, get_broker_profile
        return _cache_put(key, grounded_cost_params(
            dataset,
            broker=get_broker_profile(getattr(settings, "trading_broker", "moomoo_us")),
            aum=aum,
        ))
    except Exception as exc:
        logger.warning("[strategy_gate] grounded 成本推导失败，退回引擎默认（注意：机构口径）: %s", exc)
        return None


def strategy_net_returns(factor_signals: Signals, dataset: WidePanel,
                         aum: float = 1_000_000.0, method: str = "ic_weighted",
                         cost_params=None, apply_risk: bool = True) -> Tuple[pd.Series, pd.DataFrame]:
    """
    把多因子信号经 PortfolioManager 合成组合权重，再经 BacktestEngine 得**策略净收益序列**。
    返回 (net_returns, composite_signal)。

    `apply_risk=True`（默认）：对权重施加**与实盘同一套**风控(PM.5)与无交易带(PM.6)，
    使**门评估的账本 == 实际交易的账本**（修"验证的和交易的不是同一个组合"）。
    成本参数 None → grounded（见 `resolve_cost_params`）。
    """
    from app.core.portfolio_manager.manager import PortfolioManager
    from app.core.backtest_engine.backtest_engine import BacktestEngine

    prices = dataset["close"]
    volume = dataset.get("volume")
    if volume is None:
        volume = pd.DataFrame(1e6, index=prices.index, columns=prices.columns)

    cp = resolve_cost_params(dataset, aum, cost_params)
    pm = PortfolioManager(aum=aum, method=method, cost_params=cp)
    book = pm.build_book(factor_signals, prices, volume)
    weights = book.weights

    if apply_risk:
        try:
            from app.config import settings
            from app.core.portfolio_manager.risk_gate import PortfolioRiskGate, RiskLimits
            from app.core.portfolio_manager.horizon import apply_no_trade_band

            limits = RiskLimits(
                max_gross=float(getattr(settings, "risk_max_gross", 1.0)),
                max_net=float(getattr(settings, "risk_max_net", 1.0)),
                max_name_weight=float(getattr(settings, "risk_max_name_weight", 0.10)),
                max_sector_weight=float(getattr(settings, "risk_max_sector_weight", 0.30)),
                target_vol_ann=(float(getattr(settings, "risk_target_vol_ann", 0.0)) or None),
                long_only=(not getattr(settings, "trading_allow_short", False)),
            )
            sectors = dataset["sector"].iloc[-1] if "sector" in dataset else None
            weights, _ = PortfolioRiskGate(limits).apply(weights, sectors=sectors)
            weights = apply_no_trade_band(weights, resolve_band(dataset, aum))
        except Exception as exc:
            # 【缺陷 N-3，2026-09-20 修】原先这里是
            #     logger.warning("风控/调仓对齐失败，用原始权重: %s", exc)
            # 然后**继续往下跑回测** —— 于是这段"策略净收益"对应的是**未经风控**
            # 的权重，而本函数 docstring 承诺的恰恰是
            # 「门评估的账本 == 实际交易的账本」。验证用的组合和会去交易的组合
            # 不再是同一个，门却照样给出结论。
            #
            # `apply_risk=True` 是调用方**明确要求**过风控。做不到就必须让调用方
            # 知道，而不是换一套账本继续算。三个调用方都已 fail-closed：
            # `StrategyGate.evaluate` 判不通过并写明理由，
            # `_oos` 视为 -inf，PBO 那条会让该因子落选。
            logger.error(
                "[strategy_gate] 风控/调仓对齐失败 → **拒绝**给出策略净收益"
                "（继续用原始权重会让门评估的账本与实际交易的账本不一致）: %s", exc)
            raise RuntimeError(
                f"apply_risk=True 但风控/调仓对齐失败，拒绝用未经风控的权重回测: {exc}"
            ) from exc

    engine = BacktestEngine(cost_params=cp, initial_capital=aum)
    result = engine.run(weights, prices, volume, book.composite)
    rets = pd.Series(result.net_returns).dropna()
    return rets, book.composite


def _sharpe(rets: pd.Series, tdays: float = 252.0) -> float:
    r = np.asarray(rets.dropna(), dtype=float)
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(np.mean(r) / sd * np.sqrt(tdays))


def _oos_tail_sharpe(rets: pd.Series, oos_ratio: float = 0.30) -> float:
    """策略在**时间尾段**（最后 oos_ratio）上的 Sharpe——作边际贡献的 OOS 代理。"""
    n = len(rets)
    if n < 20:
        return _sharpe(rets)
    k = max(10, int(n * oos_ratio))
    return _sharpe(rets.iloc[-k:])


# ---------------------------------------------------------------------------
# PM.S1：策略级验证门
# ---------------------------------------------------------------------------

@dataclass
class StrategyValidationResult:
    passed:             bool
    n_factors:          int
    factors:            List[str] = field(default_factory=list)
    reasons:            List[str] = field(default_factory=list)
    sharpe:             float = 0.0
    n_segments:         int = 0
    min_seg_sharpe:     float = 0.0
    pct_seg_positive:   float = 0.0
    deflated_sharpe:    float = 0.0
    t_stat:             float = 0.0
    n_trials:           int = 1
    pbo:                Optional[float] = None   # 回测过拟合概率；None=候选<2 未算
    #: PBO 的算法口径。``cpcv`` = 组合式 purged CV（块间已 purge，S.3 收尾）；
    #: ``cscv`` = Bailey 2015 原版（块直接相邻，边界泄漏会**低估** PBO）。
    #: 必须跟着数字一起返回 —— 两种口径的 0.42 不是同一个 0.42。
    pbo_method:         str = "none"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed, "n_factors": self.n_factors, "factors": self.factors,
            "reasons": self.reasons, "sharpe": round(self.sharpe, 4),
            "n_segments": self.n_segments,
            "min_seg_sharpe": round(self.min_seg_sharpe, 4),
            "pct_seg_positive": round(self.pct_seg_positive, 4),
            "deflated_sharpe": round(self.deflated_sharpe, 4),
            "t_stat": round(self.t_stat, 4), "n_trials": self.n_trials,
            "pbo": (round(self.pbo, 4) if self.pbo is not None else None),
            "pbo_method": self.pbo_method,
        }


class StrategyGate:
    """
    组合**策略**的验证门（严门加在策略，不加在单因子）。

    Parameters
    ----------
    n_segments    : 把策略净收益按时间切成的连续段数；要求**每段 Sharpe > 0**（时序稳健）。
    dsr_threshold : Deflated Sharpe 阈值（默认 0.90；用全局 trial 去膨胀）。
    min_tstat     : 夏普 t 门槛（Harvey-Liu-Zhu，默认 3.0）。
    use_global_trials : True 用**全局跨会话累计 trial 数**做 DSR 去膨胀（S.3）。
    aum / method  : 合成策略用的资金量与合成方法。
    """

    def __init__(self, n_segments: int = 5, dsr_threshold: float = 0.90,
                 min_tstat: float = 3.0, use_global_trials: bool = True,
                 aum: float = 1_000_000.0, method: str = "ic_weighted",
                 pbo_threshold: float = 0.5, pbo_n_splits: int = 8,
                 use_cpcv: Optional[bool] = None, cpcv_groups: int = 6,
                 cpcv_k: int = 2, embargo_days: Optional[int] = None) -> None:
        self.n_segments = n_segments
        self.dsr_threshold = dsr_threshold
        self.min_tstat = min_tstat
        self.use_global_trials = use_global_trials
        self.aum = aum
        self.method = method
        self.pbo_threshold = pbo_threshold   # PBO>阈值(默认0.5)判"选择流程过拟合"
        self.pbo_n_splits = pbo_n_splits
        # S.3 收尾：PBO 默认走 CPCV（块间 purge）。读不到配置时**用更严的那一侧**
        # （CPCV ≥ CSCV），与 DEV_LESSONS §U 一致 —— 兜底不许让门变松。
        self.cpcv_groups = cpcv_groups
        self.cpcv_k = cpcv_k
        if use_cpcv is None or embargo_days is None:
            try:
                from app.config import settings
                if use_cpcv is None:
                    use_cpcv = bool(settings.s_use_cpcv)
                if embargo_days is None:
                    embargo_days = int(settings.s_embargo_days)
            except Exception as exc:
                logger.error("[strategy_gate] 读不到 CPCV 配置，按更严的一侧兜底"
                             "（use_cpcv=True, embargo=20）: %s", exc)
                use_cpcv = True if use_cpcv is None else use_cpcv
                embargo_days = 20 if embargo_days is None else embargo_days
        self.use_cpcv = bool(use_cpcv)
        self.embargo_days = int(embargo_days)

    def evaluate(self, factor_signals: Signals, dataset: WidePanel,
                 cost_params=None, n_trials: Optional[int] = None) -> StrategyValidationResult:
        from app.core.backtest_engine.performance_analyzer import deflated_sharpe_from_returns

        factors = list(factor_signals)
        res = StrategyValidationResult(passed=False, n_factors=len(factors), factors=factors)
        reasons: List[str] = []

        # 全局 trial 计数（S.3）
        if n_trials is None:
            if self.use_global_trials:
                try:
                    from app.db.trial_ledger import TrialLedger
                    n_trials = max(1, TrialLedger().total())
                except Exception as exc:
                    # 【缺陷 N-2，2026-09-20 修】原先这里 `n_trials = 1` 然后继续往下跑。
                    #
                    # **这不是无害兜底**：n_trials 是 Deflated Sharpe 的多重检验校正项，
                    # 退回 1 等于宣称"只试过一个策略" → DSR 被高估 → **门变松**。
                    # 上一版的注释自己写着"门在读不到试验台账时应当更保守，而不是
                    # 更宽松"，紧接着做的却正是更宽松的那件事 —— 外部审计实测：
                    # 注入台账不可读后仍得 passed=true、reasons=[]、DSR≈0.99997。
                    #
                    # `use_global_trials=True` 是调用方**明确要求**做全局多重检验校正。
                    # 做不到就不能给结论 —— 与 N-3（风控失败）同一个道理：
                    # 缺少必要数据时 fail-closed，把"验证不完整"如实说出来。
                    logger.error(
                        "[strategy_gate] 读取全局试验台账失败 → **拒绝给出结论**"
                        "（退回 n_trials=1 会让 Deflated Sharpe 偏乐观、门变松）: %s", exc)
                    res.n_trials = 0
                    res.reasons = [
                        f"全局试验台账不可读，无法做多重检验校正 → 验证不完整，"
                        f"不予通过: {exc}"
                    ]
                    return res
            else:
                n_trials = 1
        res.n_trials = n_trials

        if not factor_signals:
            res.reasons = ["策略为空（无因子）"]
            return res

        # ---- 策略净收益 ----
        try:
            rets, _ = strategy_net_returns(factor_signals, dataset, aum=self.aum,
                                           method=self.method, cost_params=cost_params)
        except Exception as exc:  # fail-closed
            logger.warning("[strategy_gate] 策略回测失败 → 不通过: %s", exc)
            res.reasons = [f"策略回测失败: {exc}"]
            return res

        # 【缺陷 A-2，2026-09-20 修】原判据 `float(np.nanstd(rets.values)) == 0.0`
        # 用**精确相等**判浮点零。严格全零序列 `np.nanstd` 确实返回 0.0、守卫会触发；
        # 漏掉的是**非零常数**序列 —— `np.nanstd([0.001]*100) = 2.17e-19`，
        # `== 0.0` 为假，守卫放行。而回测净收益正是这种序列：多空两腿相减、
        # 成本逐日重算，浮点残渣必然非零。
        # 判据复用 `has_meaningful_variation`（缺陷 A-3 在绩效模块里建立的同一条），
        # 不在这里另写一份 —— 两份判据迟早对同一条收益序列给出相反结论。
        # 惰性导入，与本文件 `deflated_sharpe_from_returns` 同样的理由（避免循环导入）。
        from app.core.backtest_engine.performance_analyzer import (
            has_meaningful_variation)

        if len(rets) < 30 or not has_meaningful_variation(rets):
            res.reasons = [f"策略净收益样本不足（{len(rets)}）或方差为 0"]
            return res

        # ---- 1. 分段 OOS 全为正（时序稳健，非均值为正）----
        segs = np.array_split(rets.values, self.n_segments)
        seg_sharpes = [_sharpe(pd.Series(s)) for s in segs if len(s) >= 5]
        res.n_segments = len(seg_sharpes)
        res.sharpe = _sharpe(rets)
        if seg_sharpes:
            res.min_seg_sharpe = float(min(seg_sharpes))
            res.pct_seg_positive = float(np.mean([s > 0 for s in seg_sharpes]))
        if res.n_segments < self.n_segments:
            reasons.append(f"有效分段不足（{res.n_segments} < {self.n_segments}）")
        if res.min_seg_sharpe <= 0.0:
            reasons.append(f"存在 Sharpe≤0 的时间段（最差={res.min_seg_sharpe:.3f}，"
                           f"正段比={res.pct_seg_positive*100:.0f}%）")

        # ---- 2. DSR 去膨胀 + 夏普 t ----
        try:
            dsr = deflated_sharpe_from_returns(rets, n_trials=n_trials)
            mu, sd = float(np.mean(rets.values)), float(np.std(rets.values, ddof=1))
            tstat = (mu / sd) * np.sqrt(len(rets)) if sd > 1e-12 else 0.0
            res.deflated_sharpe = float(dsr)
            res.t_stat = float(tstat)
            if dsr <= self.dsr_threshold:
                reasons.append(f"策略 DSR {dsr:.3f} ≤ 阈值 {self.dsr_threshold}（n_trials={n_trials}）")
            if tstat < self.min_tstat:
                reasons.append(f"策略夏普 t={tstat:.2f} < 门槛 {self.min_tstat}")
        except Exception as exc:  # fail-closed
            logger.warning("[strategy_gate] DSR/t 失败 → 不通过: %s", exc)
            reasons.append(f"DSR/t 计算失败: {exc}")

        # ---- 3. PBO：候选各因子单因子策略收益构成矩阵 → CSCV 过拟合概率（S.3）----
        #    量化"从这些因子里挑最优"是否过拟合；单因子(<2)无法算 → 跳过、不作为门。
        if len(factor_signals) >= 2:
            try:
                cols = []
                for name, sig in factor_signals.items():
                    r, _ = strategy_net_returns({name: sig}, dataset, aum=self.aum,
                                                method="equal_weight", cost_params=cost_params)
                    cols.append(r.rename(name))
                mat = pd.concat(cols, axis=1).dropna()
                if mat.shape[0] >= 2 * self.pbo_n_splits and mat.shape[1] >= 2:
                    arr = mat.to_numpy(dtype=float)
                    pbo, method = self._pbo(arr)
                    res.pbo = pbo
                    res.pbo_method = method
                    if pbo > self.pbo_threshold:
                        reasons.append(
                            f"PBO {pbo:.2f} > {self.pbo_threshold}（{method}，选择流程过拟合）")
            except Exception as exc:
                logger.warning("[strategy_gate] PBO 计算失败（不作为门）: %s", exc)

        res.reasons = reasons
        res.passed = len(reasons) == 0
        logger.info("[strategy_gate] %d 因子 | passed=%s | Sharpe=%.3f 段正比=%.0f%% DSR=%.3f t=%.2f PBO=%s(%s) | %s",
                    len(factors), res.passed, res.sharpe, res.pct_seg_positive * 100,
                    res.deflated_sharpe, res.t_stat,
                    ("%.2f" % res.pbo) if res.pbo is not None else "NA", res.pbo_method,
                    "OK" if res.passed else "; ".join(reasons))
        return res

    def _pbo(self, returns_matrix) -> tuple:
        """
        返回 (PBO, 口径名)。CPCV 算不动时**退回 CSCV 并如实标注口径** ——
        不静默换算法：CSCV 会低估 PBO，把它当成 CPCV 的结果读，门就悄悄松了。
        """
        from app.core.backtest_engine.overfit_stats import (
            cpcv_pbo, probability_of_backtest_overfitting,
        )
        if self.use_cpcv:
            try:
                return float(cpcv_pbo(
                    returns_matrix, n_groups=self.cpcv_groups, k=self.cpcv_k,
                    embargo=self.embargo_days)), "cpcv"
            except Exception as exc:
                logger.warning(
                    "[strategy_gate] CPCV 不可用（%s）→ 退回 CSCV。"
                    "注意 CSCV 块间无 purge，PBO 会**偏低**，口径已标注。", exc)
        return float(probability_of_backtest_overfitting(
            returns_matrix, n_splits=self.pbo_n_splits)), "cscv"


# ---------------------------------------------------------------------------
# PM.S2：因子按边际贡献准入（贪心前向选择）
# ---------------------------------------------------------------------------

@dataclass
class MarginalStep:
    factor:       str
    oos_before:   float
    oos_after:    float
    improvement:  float
    admitted:     bool


@dataclass
class MarginalSelectionResult:
    selected:     List[str]
    steps:        List[MarginalStep] = field(default_factory=list)
    final_oos:    float = 0.0

    def to_dict(self) -> dict:
        return {
            "selected": self.selected, "final_oos_sharpe": round(self.final_oos, 4),
            "steps": [
                {"factor": s.factor, "oos_before": round(s.oos_before, 4),
                 "oos_after": round(s.oos_after, 4), "improvement": round(s.improvement, 4),
                 "admitted": s.admitted}
                for s in self.steps
            ],
        }


def marginal_factor_selection(candidate_signals: Signals, dataset: WidePanel,
                              aum: float = 1_000_000.0, method: str = "ic_weighted",
                              cost_params=None, min_improve: float = 0.05,
                              oos_ratio: float = 0.30,
                              seed_signals: Optional[Signals] = None) -> MarginalSelectionResult:
    """
    贪心前向选择：从 `seed_signals`（默认空）开始，反复挑一个**加入后策略 OOS-尾段 Sharpe 提升最大**
    且提升 ≥ `min_improve` 的候选纳入，直到没有候选能再提升。

    这实现"因子按对策略的边际贡献准入"：单独弱但与组合低相关的因子，只要抬升策略 OOS 就进；
    单独漂亮但与已选高度相关（边际≈0）的因子被拒。返回选中集合 + 每步决策轨迹（可审计）。
    """
    selected: Dict[str, pd.DataFrame] = dict(seed_signals or {})
    remaining: Dict[str, pd.DataFrame] = {k: v for k, v in candidate_signals.items()
                                          if k not in selected}
    steps: List[MarginalStep] = []

    def _oos(sig: Signals) -> float:
        if not sig:
            return 0.0
        try:
            rets, _ = strategy_net_returns(sig, dataset, aum=aum, method=method,
                                           cost_params=cost_params)
            return _oos_tail_sharpe(rets, oos_ratio=oos_ratio)
        except Exception as exc:
            logger.warning("[marginal] OOS 计算失败（视为 -inf）: %s", exc)
            return float("-inf")

    base_oos = _oos(selected)
    while remaining:
        best_name, best_oos, best_impr = None, base_oos, 0.0
        for name, sig in remaining.items():
            trial = dict(selected); trial[name] = sig
            cand_oos = _oos(trial)
            impr = cand_oos - base_oos
            if impr > best_impr:
                best_name, best_oos, best_impr = name, cand_oos, impr
        if best_name is None or best_impr < min_improve:
            # 记录被拒的最佳候选（若有）以便审计
            for name, sig in remaining.items():
                trial = dict(selected); trial[name] = sig
                steps.append(MarginalStep(name, base_oos, _oos(trial),
                                          _oos(trial) - base_oos, admitted=False))
            break
        selected[best_name] = remaining.pop(best_name)
        steps.append(MarginalStep(best_name, base_oos, best_oos, best_impr, admitted=True))
        base_oos = best_oos

    return MarginalSelectionResult(selected=list(selected), steps=steps, final_oos=base_oos)
