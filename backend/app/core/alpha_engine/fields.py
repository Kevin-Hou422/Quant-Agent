"""
fields.py — DSL 字段类别（Phase 10.1）

DSL 原来只有一类字段：每天都在变的量价。基本面进来以后有了第二、第三类，
同一个算子套在不同类别上含义完全不同，所以字段必须带类别：

    price      open/high/low/close/volume/vwap/returns        每个交易日一个新值
    valuation  market_cap, earnings_yield, book_to_price      市值每天变（估值的分子按季更新）
                                                              → 每天都有新值，可以当日频量用
    quarterly  roe, gross_margin, asset_growth, sue           **阶梯**：两次财报之间是常数
    event      days_since_earnings                            计数器：公告日归零，之后每天 +1

阶梯 / 计数器字段在短窗口里没有变化：`ts_std(roe, 20)` 几乎处处为 0，拿它做分母得到的是
inf，不是"波动率"。这类表达式由 validator.SparseFieldValidator 在解析后、计算前拒掉。

字段名与 data_engine.fundamentals.panels 一一对应（有测试守着，两边不会悄悄漂移）。
"""

from __future__ import annotations

PRICE_FIELDS = frozenset({"open", "high", "low", "close", "volume", "vwap", "returns"})
VALUATION_FIELDS = frozenset({"market_cap", "earnings_yield", "book_to_price"})
QUARTERLY_FIELDS = frozenset({"roe", "gross_margin", "asset_growth", "sue"})
EVENT_FIELDS = frozenset({"days_since_earnings"})

#: 两次更新之间取值不变（或只是机械递增）的字段
STEP_FIELDS = QUARTERLY_FIELDS | EVENT_FIELDS
#: 全部基本面字段（数据集开启基本面后才存在）
FUNDAMENTAL_FIELDS = VALUATION_FIELDS | QUARTERLY_FIELDS | EVENT_FIELDS


#: 字段 → 一句话说明（界面展示用；顺序即展示顺序）
FIELD_NOTES = {
    "market_cap":          "市值 = 不复权收盘价 × 股本（美元）",
    "earnings_yield":      "盈利收益率 E/P = 过去十二个月净利润 / 市值",
    "book_to_price":       "账面市值比 B/P = 股东权益 / 市值（权益为负时为空）",
    "roe":                 "净资产收益率 = 过去十二个月净利润 / 股东权益",
    "gross_margin":        "毛利率 = 过去十二个月毛利 / 收入",
    "asset_growth":        "总资产同比增长",
    "sue":                 "标准化盈利超预期（单季净利润同比差 / 其历史标准差）",
    "days_since_earnings": "距最近一次财报公告的交易日数",
}


def field_class(name: str) -> str:
    """'price' | 'valuation' | 'quarterly' | 'event' | 'other'（sector / groups 等辅助字段）。"""
    if name in PRICE_FIELDS:
        return "price"
    if name in VALUATION_FIELDS:
        return "valuation"
    if name in QUARTERLY_FIELDS:
        return "quarterly"
    if name in EVENT_FIELDS:
        return "event"
    return "other"
