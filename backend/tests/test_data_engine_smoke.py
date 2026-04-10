"""Smoke test for the data engine pipeline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tempfile
import numpy as np
import pandas as pd

from app.core.data_engine import (
    DataProvider, RawDataset, YahooFinanceProvider,
    SchemaEnforcer, SchemaError, STANDARD_COLUMNS,
    AlphaVantageProvider, LocalParquetProvider,
    PanelFactory, UniverseFilter,
    Preprocessor, CorporateActionAdjuster, SyntheticFieldBuilder, MissingValueStrategy,
    DataHealthChecker, HealthReport, GapDetector, SpikeDetector, ZeroVolumeDetector,
    ParquetFeatureStore, DataChunker,
    DataManager,
)

# ------------------------------------------------------------------
# 构造合成数据集
# ------------------------------------------------------------------
dates   = pd.date_range('2023-01-02', periods=60, freq='B')
tickers = ['AAPL', 'MSFT', 'DELIST']

rng  = np.random.default_rng(42)
rows = []
for ticker in tickers:
    for i, dt in enumerate(dates):
        if ticker == 'DELIST' and i >= 30:
            continue
        close  = 100 + rng.normal(0, 1)
        volume = 0 if (ticker == 'AAPL' and i == 5) else float(rng.integers(1_000_000, 10_000_000))
        rows.append({
            'timestamp':  dt,
            'ticker':     ticker,
            'open':       close - 0.5,
            'high':       close + 1.0,
            'low':        close - 1.0,
            'close':      close,
            'volume':     volume,
            'adj_factor': 1.2 if (ticker == 'AAPL' and i < 10) else 1.0,
        })

raw_df = pd.DataFrame(rows)
print(f'原始数据: {len(raw_df)} 行, {raw_df.ticker.nunique()} tickers')

# ------------------------------------------------------------------
# 1. SchemaEnforcer
# ------------------------------------------------------------------
enforcer  = SchemaEnforcer()
schema_df = enforcer.enforce(raw_df)
assert list(schema_df.columns[:9]) == STANDARD_COLUMNS
assert schema_df['adj_factor'].dtype == float
print(f'SchemaEnforcer OK: shape={schema_df.shape}')

# ------------------------------------------------------------------
# 2. PanelFactory
# ------------------------------------------------------------------
factory = PanelFactory()
panel   = factory.reindex_to_master([schema_df])
expected_rows = 60 * 3
assert len(panel) == expected_rows, f'预期 {expected_rows} 行，实际 {len(panel)}'
delist_late = panel[(panel['ticker'] == 'DELIST') & (panel['timestamp'] > dates[29])]
assert delist_late['close'].isna().all()
print(f'PanelFactory OK: shape={panel.shape}, DELIST NaN 校验通过')

# ------------------------------------------------------------------
# 3. UniverseFilter (PIT)
# ------------------------------------------------------------------
universe_records = []
for tk in ['AAPL', 'MSFT']:
    for dt in dates:
        universe_records.append({'date': dt, 'ticker': tk, 'active': True})
for dt in dates[:30]:
    universe_records.append({'date': dt, 'ticker': 'DELIST', 'active': True})
for dt in dates[30:]:
    universe_records.append({'date': dt, 'ticker': 'DELIST', 'active': False})

uni_df   = pd.DataFrame(universe_records)
uf       = UniverseFilter(universe_df=uni_df, strict=True)
filtered = uf.filter(panel, as_of=str(dates[-1].date()))
assert 'DELIST' not in filtered['ticker'].unique()
print(f'UniverseFilter OK: active tickers = {sorted(filtered["ticker"].unique())}')

# ------------------------------------------------------------------
# 4. Preprocessor
# ------------------------------------------------------------------
prep       = Preprocessor()
proc, rpt  = prep.run(panel, apply_adj=True, build_synthetic=True, ffill_limit=5)
assert 'vwap'        in proc.columns
assert 'returns'     in proc.columns
assert 'log_returns' in proc.columns
extra_cols = [c for c in proc.columns if c not in STANDARD_COLUMNS]
print(f'Preprocessor OK: 新字段={extra_cols}')
print(f'  NaN摘要: {rpt.get("_summary", {})}')

# ------------------------------------------------------------------
# 5. DataHealthChecker
# ------------------------------------------------------------------
checker = DataHealthChecker(spike_threshold=0.5)
report  = checker.check(proc)
assert len(report.zero_volume) >= 1, '应检测到 AAPL 零成交量'
assert 0.0 <= report.overall_score <= 1.0
print(f'HealthReport OK: {report}')
print(f'  zero_volume={len(report.zero_volume)}, score={report.overall_score}')

# ------------------------------------------------------------------
# 6. ParquetFeatureStore + DataChunker
# ------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmpdir:
    store = ParquetFeatureStore(tmpdir)
    store.save(proc, name='test_ds')

    meta = store._load_metadata('test_ds')
    assert meta is not None
    assert 'AAPL' in meta['tickers']
    print(f'FeatureStore 写入 OK: tickers={meta["tickers"]}, 日期范围={meta["start"]}~{meta["end"]}')

    loaded = store.load('test_ds', tickers=['AAPL', 'MSFT'],
                        start='2023-01-01', end='2023-12-31')
    assert not loaded.empty
    assert set(loaded['ticker'].unique()) == {'AAPL', 'MSFT'}
    print(f'FeatureStore 读取 OK: shape={loaded.shape}')

    chunker = DataChunker(store, chunk_size=2)
    chunks  = list(chunker.iter_chunks('test_ds', ['AAPL', 'MSFT', 'DELIST'],
                                       '2023-01-01', '2023-12-31'))
    assert len(chunks) >= 1
    total = sum(len(c) for c in chunks)
    print(f'DataChunker OK: {len(chunks)} 批, 合计 {total} 行')

# ------------------------------------------------------------------
# 7. DataManager 端到端（MockProvider）
# ------------------------------------------------------------------
class MockProvider(DataProvider):
    def __init__(self, data):
        self._data = data
    def available_fields(self):
        return list(STANDARD_COLUMNS)
    def fetch(self, tickers, start, end, fields=None):
        return {}
    def fetch_panel(self, tickers, start, end, fields=None):
        df = self._data[self._data['ticker'].isin(tickers)].copy()
        df = df[(df['timestamp'] >= pd.Timestamp(start)) &
                (df['timestamp'] <= pd.Timestamp(end))]
        return df

mock = MockProvider(schema_df)
dm   = DataManager(providers=[mock], cache_to_store=False)
result_panel, result_health = dm.get_panel(
    tickers=['AAPL', 'MSFT'],
    start_date='2023-01-02',
    end_date='2023-03-31',
    apply_adj=True,
    ffill_limit=3,
    run_health_check=True,
)
assert not result_panel.empty
assert result_health is not None
print(f'DataManager OK: shape={result_panel.shape}, health={result_health.overall_score:.3f}')

print()
print('=' * 40)
print('全部验证通过！')
print('=' * 40)
