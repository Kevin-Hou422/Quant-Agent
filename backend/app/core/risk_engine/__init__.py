"""
risk_engine — Barra-lite 风险因子模型（Phase R.2）

回答一个 PM.5 的敞口/集中度上限回答不了的问题：**这个组合的风险从哪来**。
gross/net/单票上限管的是"别把鸡蛋放一个篮子"，但一篮子彼此不相关的股票和
一篮子同时押在"低波动"上的股票，敞口可以完全一样、真实风险差一个量级。
"""
from .factor_model import (
    STYLE_FACTORS,
    RiskAttribution,
    RiskModel,
    build_exposures,
    fit_risk_model,
)

__all__ = [
    "STYLE_FACTORS",
    "RiskAttribution",
    "RiskModel",
    "build_exposures",
    "fit_risk_model",
]
