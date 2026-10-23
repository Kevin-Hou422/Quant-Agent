"""
c2_bars.py — 给"只有收盘价"的旧测试补一根**说清楚的** K 线（时钟 c2，2026-10-22 D14）。

c2 的引擎 / 模拟账本必须有开 / 高 / 低：T 收盘冻结的目标在 T+1 按 DAY 限价单撮合（参考价 = T 收盘）。
很多测试钉的是**记账公式**（换手、涨跌幅、多空腿、借券、复利、成本口径），输入只有收盘价。给它们配这根 K 线：

    开盘(t) = 收盘(t−1)          → 没有隔夜跳空；开盘恰好等于限价参考价，必然按开盘价成交
    最高 / 最低 = 开与收的较大 / 较小值

于是 c2 的账本退化为：第 t 天按 收盘(t−1) 的价格把仓位调到目标(t−1)，吃 收盘(t−1) → 收盘(t) 的涨跌 ——
"昨仓 × 今日涨跌"与旧公式相同；**差别只在调仓的成本记在执行那一天（t），不是决策那一天（t−1）**，
最后一天的决策当天不执行。断言成本 / 换手落在哪一天的用例要按这一点改写，不能把这根 K 线当成"c1 兼容层"。

c2 成交规则本身（跳空到限价外、当日触及按限价、不成交挂账）由 `test_reference_ledger_1019.py` 与
`tests/integration/test_decisions_1022_batch2.py` 覆盖，不靠这根 K 线。
"""
from __future__ import annotations

from typing import Dict

import pandas as pd


def prev_close_bars(close: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """{"open","high","low"}：开盘 = 前一日收盘（首日 = 当日收盘），高 / 低包住开与收。"""
    c = close.astype(float)
    o = c.shift(1).fillna(c)
    return {"open": o, "high": o.where(o >= c, c), "low": o.where(o <= c, c)}


def with_prev_close_bars(dataset: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """数据集里没有开 / 高 / 低时按 `prev_close_bars` 补上（已有的不动）。"""
    out = dict(dataset)
    bars = prev_close_bars(dataset["close"])
    for k, v in bars.items():
        out.setdefault(k, v)
    return out
