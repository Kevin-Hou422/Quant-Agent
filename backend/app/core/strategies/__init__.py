"""
strategies — 经典/基准策略库（Phase PM.S3）

不是所有价值都要靠 GP 从零挖。学术界几十年沉淀的经典横截面异象（动量/反转/低波/流动性/偏度）
既是**没有自研因子时可以直接交易的策略**，也是**衡量自研因子有没有真本事的基准**——一个自研
因子若跑不赢经典动量，就没有纳入的理由。

导出的对象都是**确定性 DSL 策略**（无 LLM、无拟合参数），可直接喂进 PortfolioManager，
或作为 CANDIDATE 种入正常生命周期，与自研因子在同一门控下竞争。
"""

from .baselines import (
    BaselineStrategy,
    BASELINE_LIBRARY,
    list_baselines,
    get_baseline,
    baseline_signals,
    seed_baselines,
)

__all__ = [
    "BaselineStrategy",
    "BASELINE_LIBRARY",
    "list_baselines",
    "get_baseline",
    "baseline_signals",
    "seed_baselines",
]
