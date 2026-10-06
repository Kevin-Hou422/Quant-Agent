"""
Execution package.

Phase 7  : paper_broker  — 内部模拟成交（决策日收盘价成交，对账回测口径）
Phase 12 : broker_gateway / order_builder / pretrade_gate / order_manager / fidelity
           — moomoo 纸交易执行层（下单、对账、风控、全平、崩溃恢复、保真度）。
           按需从子模块导入，不在包级别加载（避免无券商依赖的路径被迫导入 SDK 相关代码）。
"""

from .paper_broker import PaperBroker  # noqa: F401
