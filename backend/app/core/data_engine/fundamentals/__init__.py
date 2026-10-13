"""基本面数据（Phase 10）：EDGAR 事实 → 时点计算 → 日频稀疏字段面板。"""

from .panels import ALL_FIELDS, EVENT_FIELDS, FUNDAMENTAL_FIELDS, FundamentalsReport, build_panels

__all__ = ["ALL_FIELDS", "EVENT_FIELDS", "FUNDAMENTAL_FIELDS", "FundamentalsReport", "build_panels"]
