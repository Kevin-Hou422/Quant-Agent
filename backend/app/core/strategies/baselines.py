"""
baselines.py — 经典横截面基准策略库（Phase PM.S3）

作用（回答用户两个问题）
    ① "现在的门控下，如果没有因子，怎么进行后续 trading 环节？"
       → 库里这些经典策略是**确定性 DSL**，随时可 `seed_baselines()` 种成 CANDIDATE 走正常
         生命周期，或 `baseline_signals()` 直接喂 PortfolioManager 组一个真实账本来交易。
    ② "要的是具体交易策略，不是回测漂亮的因子；没因子时放经典策略作基准。"
       → 每个自研因子/策略都应与这些经典对照：跑不赢经典动量的，就没有纳入的理由（PM.S2 边际准入）。

设计纪律
    - 全部是**无拟合参数、可复现**的公式化异象（窗口取学术惯例值，非在本数据上调出来的）。
    - 只用免费 OHLCV 派生字段（close/returns/volume/high/low），与 moomoo 单一源一致，无 skew。
    - 输出是横截面信号（rank ∈ 大致 [0,1] 或 z），越大代表越该**做多**；组合侧再决定多空/权重。
    - LLM 不参与。

引用为方便追溯的经典文献，非本项目主张。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


WidePanel = Dict[str, pd.DataFrame]


@dataclass(frozen=True)
class BaselineStrategy:
    name:        str      # 稳定 slug（seed 幂等键的一部分）
    dsl:         str      # 项目 DSL 表达式（已验证可解析执行）
    category:    str      # momentum | reversal | volatility | liquidity | skew | trend
    description: str      # 一句话经济逻辑
    reference:   str      # 经典文献


# ---------------------------------------------------------------------------
# 库：8 个经过 DSL 解析/执行验证的经典异象（见 tests/test_phase_pm_s3.py）
# ---------------------------------------------------------------------------

_STRATEGIES: List[BaselineStrategy] = [
    BaselineStrategy(
        name="xs_momentum_12_1",
        dsl="rank((ts_delta(log(close),252))-(ts_delta(log(close),21)))",
        category="momentum",
        description="横截面动量 12-1：过去 12 月涨幅（剔除最近 1 月反转）越高越买。",
        reference="Jegadeesh & Titman (1993)",
    ),
    BaselineStrategy(
        name="xs_momentum_12m",
        dsl="rank(ts_delta(log(close),252))",
        category="momentum",
        description="横截面动量 12 月：过去一年累计对数收益越高越买。",
        reference="Jegadeesh & Titman (1993)",
    ),
    BaselineStrategy(
        name="short_reversal_1w",
        dsl="rank((-ts_sum(returns,5)))",
        category="reversal",
        description="短期反转：过去一周跌得越多越买（做多输家、做空赢家）。",
        reference="Jegadeesh (1990); Lehmann (1990)",
    ),
    BaselineStrategy(
        name="low_volatility",
        dsl="rank((-ts_std(returns,60)))",
        category="volatility",
        description="低波动异象：过去 3 月已实现波动越低越买。",
        reference="Ang, Hodrick, Xing & Zhang (2006)",
    ),
    BaselineStrategy(
        name="high_52w",
        dsl="rank((close/ts_max(close,252)))",
        category="momentum",
        description="52 周高点动量：现价越接近一年最高价越买（锚定不足）。",
        reference="George & Hwang (2004)",
    ),
    BaselineStrategy(
        name="ts_trend_3m",
        dsl="rank(ts_delta(log(close),60))",
        category="trend",
        description="时序趋势 3 月：近 3 月趋势越强越买（截面化的时序动量）。",
        reference="Moskowitz, Ooi & Pedersen (2012)",
    ),
    BaselineStrategy(
        name="low_turnover_liq",
        dsl="rank((-ts_mean(volume,20)))",
        category="liquidity",
        description="流动性溢价：近 20 日成交量越低（越不流动）越买，赚流动性补偿。",
        reference="Amihud (2002); Datar, Naik & Radcliffe (1998)",
    ),
    BaselineStrategy(
        name="idio_skew",
        dsl="rank((-ts_skew(returns,60)))",
        category="skew",
        description="偏度异象：近 3 月收益偏度越低（越少彩票属性）越买。",
        reference="Boyer, Mitton & Vorkink (2010)",
    ),
]

BASELINE_LIBRARY: Dict[str, BaselineStrategy] = {s.name: s for s in _STRATEGIES}


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------

def list_baselines(category: Optional[str] = None) -> List[BaselineStrategy]:
    """列出基准策略；可按 category 过滤。"""
    if category is None:
        return list(_STRATEGIES)
    return [s for s in _STRATEGIES if s.category == category]


def get_baseline(name: str) -> BaselineStrategy:
    if name not in BASELINE_LIBRARY:
        raise KeyError(f"未知基准策略 '{name}'；可选：{sorted(BASELINE_LIBRARY)}")
    return BASELINE_LIBRARY[name]


# ---------------------------------------------------------------------------
# 计算信号（可直接喂 PortfolioManager.build_book 的 factor_signals）
# ---------------------------------------------------------------------------

def baseline_signals(dataset: WidePanel,
                     names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    在给定数据集上计算基准策略信号面板 {name: signal_df(T×N)}。

    解析/执行失败的单个策略会被**跳过并记日志**（fail-safe：一个坏表达式不拖垮整批），
    返回的 dict 只含成功者。
    """
    import logging
    from app.core.alpha_engine.dsl_executor import Executor

    logger = logging.getLogger(__name__)
    picks = _STRATEGIES if names is None else [get_baseline(n) for n in names]
    execu = Executor(validate=False)
    out: Dict[str, pd.DataFrame] = {}
    for s in picks:
        try:
            sig = execu.run_expr(s.dsl, dataset)
            if isinstance(sig, pd.DataFrame) and sig.notna().any().any():
                out[s.name] = sig
            else:
                logger.warning("[baselines] '%s' 信号全空，跳过", s.name)
        except Exception as exc:  # noqa: BLE001 - fail-safe，单条坏不拖累整批
            logger.warning("[baselines] '%s' 执行失败，跳过：%s", s.name, exc)
    return out


# ---------------------------------------------------------------------------
# 种入生命周期（作为 CANDIDATE，与自研因子在同一门控下竞争 / 兜底可交易）
# ---------------------------------------------------------------------------

def seed_baselines(store, names: Optional[List[str]] = None,
                   status: str = "candidate") -> Dict[str, int]:
    """
    把基准策略作为因子写入 AlphaStore（默认 CANDIDATE），返回 {name: alpha_id}。

    **幂等**：库内已存在相同 DSL 的记录则跳过（按 DSL 去重），避免每晚重复种入。
    这样"没有自研因子时"也永远有一批经典策略在池子里，可被验证门与组合层消费。
    """
    from app.db.alpha_store import AlphaResult

    picks = _STRATEGIES if names is None else [get_baseline(n) for n in names]
    existing_dsls = {r.dsl for r in store.query(limit=100000)}
    out: Dict[str, int] = {}
    for s in picks:
        if s.dsl in existing_dsls:
            continue
        aid = store.save(AlphaResult(
            dsl=s.dsl,
            hypothesis=f"[baseline:{s.category}] {s.description}（{s.reference}）",
            status=status,
        ))
        out[s.name] = aid
    return out
