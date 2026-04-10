"""
DataManager —— 多源金融数据管道的顶层门面。

get_panel() 执行完整的数据获取与预处理流水线：

1. 优先从 FeatureStore 读取缓存（命中则跳过网络请求）
2. 缓存未命中 → 按 provider 优先级依次尝试 fetch_panel()
3. SchemaEnforcer 强制标准格式
4. PanelFactory.reindex_to_master（补全全集主索引，退市填 NaN）
5. UniverseFilter（PIT 过滤，可选）
6. Preprocessor.run（复权 + 合成字段 + ffill）
7. DataHealthChecker.check（可选）
8. 写回 FeatureStore（可选）

返回：(panel_df: pd.DataFrame, health_report: HealthReport | None)
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .base import DataProvider
from .schema import SchemaEnforcer, STANDARD_COLUMNS
from .panel_factory import PanelFactory, UniverseFilter
from .preprocessor import Preprocessor
from .health_report import DataHealthChecker, HealthReport
from .feature_store import ParquetFeatureStore

logger = logging.getLogger(__name__)


class DataManager:
    """
    多源数据管道门面（Facade）。

    Parameters
    ----------
    providers       : DataProvider 列表，按优先级从高到低排列。
                      DataManager 依次尝试，直到某 provider 成功返回数据。
    feature_store   : ParquetFeatureStore 实例（可选）。
                      非 None 时启用读/写缓存。
    universe_filter : UniverseFilter 实例（可选）。
                      非 None 时对面板执行 PIT Universe 过滤。
    cache_to_store  : 是否在获取新数据后自动写入 feature_store（默认 True）。
    store_name      : FeatureStore 数据集名称（默认 "default"）。
    date_col        : 时间列名（默认 "timestamp"）
    ticker_col      : Ticker 列名（默认 "ticker"）
    """

    def __init__(
        self,
        providers: List[DataProvider],
        feature_store: Optional[ParquetFeatureStore] = None,
        universe_filter: Optional[UniverseFilter] = None,
        cache_to_store: bool = True,
        store_name: str = "default",
        date_col: str = "timestamp",
        ticker_col: str = "ticker",
    ) -> None:
        if not providers:
            raise ValueError("providers 列表不能为空，至少需要一个 DataProvider")

        self.providers       = providers
        self.feature_store   = feature_store
        self.universe_filter = universe_filter
        self.cache_to_store  = cache_to_store
        self.store_name      = store_name
        self.date_col        = date_col
        self.ticker_col      = ticker_col

        self._enforcer = SchemaEnforcer(allow_extra=True)
        self._factory  = PanelFactory(date_col=date_col, ticker_col=ticker_col)
        self._prep     = Preprocessor(date_col=date_col, ticker_col=ticker_col)
        self._checker  = DataHealthChecker()

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def get_panel(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
        apply_adj: bool = True,
        ffill_limit: int = 5,
        run_health_check: bool = True,
        pit_as_of: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Optional[HealthReport]]:
        """
        获取 long-format 面板数据，经过完整预处理流水线。

        Parameters
        ----------
        tickers          : 资产代码列表
        start_date       : 起始日期（"YYYY-MM-DD"）
        end_date         : 截止日期（"YYYY-MM-DD"）
        fields           : 需要的字段列表（None = 全部）
        apply_adj        : 是否复权
        ffill_limit      : 前向填充最大天数（0 = 不填充）
        run_health_check : 是否执行数据质量检测
        pit_as_of        : PIT 时间截面（None = 使用 end_date）

        Returns
        -------
        (panel: pd.DataFrame, report: HealthReport | None)
        """
        tickers = [t.upper() for t in tickers]
        logger.info(
            "DataManager.get_panel: %d tickers, %s → %s",
            len(tickers), start_date, end_date,
        )

        # Step 1: 尝试从 FeatureStore 读取缓存
        panel = self._try_cache(tickers, start_date, end_date, fields)

        # Step 2: 缓存未命中 → 从 providers 获取
        if panel is None:
            panel = self._fetch_from_providers(tickers, start_date, end_date, fields)

        if panel is None or panel.empty:
            warnings.warn(
                f"DataManager: 所有数据源均无法获取数据 "
                f"(tickers={tickers[:5]}... start={start_date} end={end_date})",
                stacklevel=2,
            )
            return pd.DataFrame(columns=STANDARD_COLUMNS), None

        # Step 3: Schema 强制
        panel = self._enforcer.enforce(panel)

        # Step 4: PanelFactory — 补全全集主索引（防止 index 不对齐）
        panel = self._factory.reindex_to_master(
            [panel], start=start_date, end=end_date
        )

        # Step 5: UniverseFilter（PIT 过滤）
        if self.universe_filter is not None:
            as_of = pit_as_of or end_date
            panel = self.universe_filter.filter(
                panel, as_of=as_of,
                date_col=self.date_col,
                ticker_col=self.ticker_col,
            )

        # Step 6: 预处理（复权 + 合成字段 + ffill）
        panel, nan_report = self._prep.run(
            panel,
            apply_adj=apply_adj,
            build_synthetic=True,
            ffill_limit=ffill_limit,
            report_nan=True,
        )

        # Step 7: 写回 FeatureStore
        if self.cache_to_store and self.feature_store is not None:
            self._write_cache(panel)

        # Step 8: 数据质量检测
        health_report: Optional[HealthReport] = None
        if run_health_check:
            health_report = self._checker.check(
                panel,
                date_col=self.date_col,
                ticker_col=self.ticker_col,
            )

        logger.info(
            "DataManager: 面板就绪 shape=%s, health=%s",
            panel.shape,
            f"{health_report.overall_score:.3f}" if health_report else "N/A",
        )
        return panel, health_report

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def get_field(
        self,
        field: str,
        tickers: List[str],
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        获取单个字段的 wide-format DataFrame（time × asset）。
        内部调用 get_panel，然后 pivot。
        """
        panel, _ = self.get_panel(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            fields=[field],
            **kwargs,
        )
        if panel.empty or field not in panel.columns:
            return pd.DataFrame()

        wide = panel.pivot_table(
            index=self.date_col,
            columns=self.ticker_col,
            values=field,
            aggfunc="last",
        )
        wide.index = pd.DatetimeIndex(wide.index)
        return wide

    def add_provider(self, provider: DataProvider, priority: int = -1) -> None:
        """
        动态添加数据源。
        priority=-1 追加到末尾（最低优先级），0 插入到最前（最高优先级）。
        """
        if priority == -1:
            self.providers.append(provider)
        else:
            self.providers.insert(priority, provider)
        logger.info("DataManager: 添加 provider %s (priority=%d)", provider, priority)

    def provider_status(self) -> List[Dict]:
        """返回各 provider 的元信息。"""
        return [p.metadata() for p in self.providers]

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _try_cache(
        self,
        tickers: List[str],
        start: str,
        end: str,
        fields: Optional[List[str]],
    ) -> Optional[pd.DataFrame]:
        """尝试从 FeatureStore 加载缓存数据。"""
        if self.feature_store is None:
            return None
        if not self.feature_store.has_data(self.store_name, tickers, start, end):
            logger.debug("FeatureStore 缓存未命中")
            return None

        logger.info("FeatureStore 缓存命中，跳过网络请求")
        return self.feature_store.load(
            name=self.store_name,
            tickers=tickers,
            start=start,
            end=end,
            fields=fields,
            date_col=self.date_col,
            ticker_col=self.ticker_col,
        )

    def _fetch_from_providers(
        self,
        tickers: List[str],
        start: str,
        end: str,
        fields: Optional[List[str]],
    ) -> Optional[pd.DataFrame]:
        """按优先级依次尝试各 provider，返回第一个成功的结果。"""
        for provider in self.providers:
            try:
                logger.info("尝试 provider: %s", provider.__class__.__name__)
                panel = provider.fetch_panel(
                    tickers=tickers, start=start, end=end, fields=fields
                )
                if panel is not None and not panel.empty:
                    logger.info(
                        "provider %s 成功返回 %d 行数据",
                        provider.__class__.__name__, len(panel),
                    )
                    return panel
            except Exception as exc:
                logger.warning(
                    "provider %s 失败: %s",
                    provider.__class__.__name__, exc,
                )
        return None

    def _write_cache(self, panel: pd.DataFrame) -> None:
        """将预处理后的面板写入 FeatureStore 缓存。"""
        if self.feature_store is None:
            return
        try:
            self.feature_store.save(
                panel=panel,
                name=self.store_name,
                date_col=self.date_col,
                ticker_col=self.ticker_col,
                overwrite=False,
            )
        except Exception as exc:
            logger.warning("写入 FeatureStore 缓存失败: %s", exc)
