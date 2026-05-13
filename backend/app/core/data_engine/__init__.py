"""
Data Engine 包 — 统一导出入口

包含原有 API（向后兼容）和新增的多源生产级数据管道 API。
"""

# ============================================================
# 原有 API（保留，向后兼容）
# ============================================================
from .base import DataProvider, RawDataset
from .yahoo_provider import YahooFinanceProvider
from .dataset_loader import DatasetLoader, load_dataset, get_provider, register_provider

# ============================================================
# 新增：标准 Schema 层
# ============================================================
from .schema import (
    SchemaEnforcer,
    SchemaError,
    STANDARD_COLUMNS,
    PRICE_FIELDS,
    NUMERIC_FIELDS,
    wide_to_long,
)

# ============================================================
# 新增：多源 Provider
# ============================================================
from .alpha_vantage_provider import AlphaVantageProvider
from .local_parquet_provider import LocalParquetProvider

# ============================================================
# 新增：面板工厂 + PIT Universe 过滤
# ============================================================
from .panel_factory import PanelFactory, UniverseFilter

# ============================================================
# 新增：预处理管道
# ============================================================
from .preprocessor import (
    Preprocessor,
    CorporateActionAdjuster,
    SyntheticFieldBuilder,
    MissingValueStrategy,
)

# ============================================================
# 新增：数据健康检查
# ============================================================
from .health_report import (
    DataHealthChecker,
    HealthReport,
    GapDetector,
    SpikeDetector,
    ZeroVolumeDetector,
)

# ============================================================
# 新增：Parquet 特征库 + 分块加载
# ============================================================
from .feature_store import ParquetFeatureStore, DataChunker

# ============================================================
# 新增：顶层 DataManager 门面
# ============================================================
from .data_manager import DataManager


__all__ = [
    # 原有
    "DataProvider", "RawDataset",
    "YahooFinanceProvider",
    "DatasetLoader", "load_dataset", "get_provider", "register_provider",
    # Schema
    "SchemaEnforcer", "SchemaError",
    "STANDARD_COLUMNS", "PRICE_FIELDS", "NUMERIC_FIELDS",
    "wide_to_long",
    # Providers
    "AlphaVantageProvider",
    "LocalParquetProvider",
    # Panel
    "PanelFactory", "UniverseFilter",
    # Preprocessor
    "Preprocessor", "CorporateActionAdjuster",
    "SyntheticFieldBuilder", "MissingValueStrategy",
    # Health
    "DataHealthChecker", "HealthReport",
    "GapDetector", "SpikeDetector", "ZeroVolumeDetector",
    # Storage
    "ParquetFeatureStore", "DataChunker",
    # Manager
    "DataManager",
    # Partitioner
    "DataPartitioner",
    "PartitionedDataset",
]

# ============================================================
# 新增：数据分区器（IS/OOS 严格隔离）
# ============================================================
from .data_partitioner import DataPartitioner, PartitionedDataset

# ============================================================
# 多数据集抽象层 (PROMPT 3)
# ============================================================
from .multi_dataset import (
    Dataset,
    DatasetRegistry,
    load_dataset as load_named_dataset,
    get_registry,
    STANDARD_FIELDS,
)

# ============================================================
# 生产数据集注册表 (10 datasets, 3 providers)
# ============================================================
from .dataset_registry import (
    load_registry_dataset,
    registry_names,
    registry_spec,
    DatasetSpec,
)

# ============================================================
# 动态 Filter 系统
# ============================================================
from .dataset_filters import (
    FilterConfig,
    FilterResult,
    DatasetFilterEngine,
    apply_filters,
    validate_filter_config,
    VALID_FILTER_VALUES,
)

# ============================================================
# 数据提供商
# ============================================================
from .providers.akshare_provider import AkshareProvider
from .providers.ccxt_provider import CcxtBinanceProvider
