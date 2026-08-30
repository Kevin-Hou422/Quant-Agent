"""
Data Engine 包 — 统一导出入口

活的数据路径：providers（yahoo/moomoo/akshare/ccxt）+ schema + dataset_registry + pit_store。
旧的多源管道（data_manager/feature_store/dataset_loader/panel_factory/preprocessor/
alpha_vantage_provider）已于 2026-08-30 删除——被 dataset_registry + pit_store 完全取代。
"""

# ============================================================
# Provider 基类 + 主力 provider
# ============================================================
from .base import DataProvider, RawDataset
from .yahoo_provider import YahooFinanceProvider

# ============================================================
# 标准 Schema 层
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
# 本地 Parquet provider
# ============================================================
from .local_parquet_provider import LocalParquetProvider

# ============================================================
# 数据健康检查
# ============================================================
from .health_report import (
    DataHealthChecker,
    HealthReport,
    GapDetector,
    SpikeDetector,
    ZeroVolumeDetector,
)

# ============================================================
# 数据分区器（IS/OOS 严格隔离）
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
# 生产数据集注册表
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


__all__ = [
    "DataProvider", "RawDataset",
    "YahooFinanceProvider",
    "SchemaEnforcer", "SchemaError",
    "STANDARD_COLUMNS", "PRICE_FIELDS", "NUMERIC_FIELDS",
    "wide_to_long",
    "LocalParquetProvider",
    "DataHealthChecker", "HealthReport",
    "GapDetector", "SpikeDetector", "ZeroVolumeDetector",
    "DataPartitioner", "PartitionedDataset",
    "Dataset", "DatasetRegistry", "load_named_dataset", "get_registry", "STANDARD_FIELDS",
    "load_registry_dataset", "registry_names", "registry_spec", "DatasetSpec",
    "FilterConfig", "FilterResult", "DatasetFilterEngine",
    "apply_filters", "validate_filter_config", "VALID_FILTER_VALUES",
    "AkshareProvider", "CcxtBinanceProvider",
]
