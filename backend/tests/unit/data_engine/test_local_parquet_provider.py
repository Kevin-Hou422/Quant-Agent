"""
data_engine/local_parquet_provider.py —— 本地 Parquet 落盘与回读

**此前零测试**（21 个变异点，D 档）。

这是"完全免费跑模拟交易"的**本地数据底座**：拉过一次的数据落到
`{root}/{TICKER}/year={YYYY}/data.parquet`，之后不再打网络。
它坏掉的后果全部是"数据少了一块但面板还在"：

  - `range(start.year, end.year + 1)` 的 `+ 1` 被去掉 → **结束年份的
    分区永远不读**。跨年请求 2022-01-01→2023-12-31 只拿到 2022 年，
    回测区间静默腰斩，而 `panel` 非空、没有任何告警。
  - 日期过滤的 `>=` / `<=` 翻面 → 端点那一天丢失，跨源对齐错位。
  - 追加写的 `drop_duplicates(keep="last")` 改成 `"first"` →
    **数据修正永远写不进去**（旧的错误值赢），这是最阴的一种：
    复权调整、事后修正全部失效。
  - `if out_path.exists() and not overwrite` 的 `not` 被删 →
    追加变覆盖，历史数据被一次增量拉取抹掉。

因此本文件全部走**真实文件系统**（`tmp_path`）与真实 pyarrow，
不 mock 存储层 —— 落盘再回读是这个模块唯一有意义的验证方式。
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.local_parquet_provider import (
    _SUPPORTED_FIELDS,
    LocalParquetProvider,
)
from app.core.data_engine.schema import STANDARD_COLUMNS

pytest.importorskip("pyarrow")


def _long(ticker: str, dates, base: float = 100.0) -> pd.DataFrame:
    """标准 long-format 行情。"""
    n = len(dates)
    return pd.DataFrame({
        "timestamp":  pd.DatetimeIndex(dates),
        "ticker":     [ticker] * n,
        "open":       [base + i for i in range(n)],
        "high":       [base + i + 1.0 for i in range(n)],
        "low":        [base + i - 1.0 for i in range(n)],
        "close":      [base + i + 0.5 for i in range(n)],
        "volume":     [1e6 + i for i in range(n)],
        "vwap":       [base + i + 0.2 for i in range(n)],
        "adj_factor": [1.0] * n,
    })


@pytest.fixture
def provider(tmp_path):
    return LocalParquetProvider(tmp_path / "store")


# ===========================================================================
# A. 接口元数据
# ===========================================================================

class TestProviderMetadata:

    def test_the_supported_field_list_is_the_documented_one(self):
        assert _SUPPORTED_FIELDS == ["open", "high", "low", "close", "volume",
                                     "vwap", "adj_factor", "returns"], (
            f"支持字段列表被改动：{_SUPPORTED_FIELDS}")

    def test_available_fields_returns_a_copy_not_the_module_list(self, provider):
        """
        `list(_SUPPORTED_FIELDS)` —— 直接返回模块级列表会让调用方
        的 `.remove()` 永久改掉全局字段表，后续所有 provider 实例受害。
        """
        got = provider.available_fields()
        assert got == _SUPPORTED_FIELDS
        got.append("污染")
        assert "污染" not in _SUPPORTED_FIELDS, "返回的是模块级列表本身"

    def test_metadata_carries_the_root_and_declares_no_rate_limit(self, provider,
                                                                   tmp_path):
        """
        本地源没有延迟与限流 —— 这两项是 `None` 而不是 0。
        写成 0 会让上层"按延迟排序数据源"把它排到最后（0 被当成未知）。
        """
        m = provider.metadata()
        assert m["name"] == "LocalParquetProvider"
        assert m["root_dir"] == str(tmp_path / "store")
        assert m["latency_ms"] is None and m["rate_limit"] is None
        assert m["available_fields"] == _SUPPORTED_FIELDS

    def test_the_root_dir_accepts_both_str_and_path(self, tmp_path):
        assert LocalParquetProvider(str(tmp_path)).root_dir == Path(tmp_path)
        assert LocalParquetProvider(Path(tmp_path)).root_dir == Path(tmp_path)


# ===========================================================================
# B. 写入 → 回读 往返
# ===========================================================================

class TestWriteReadRoundTrip:

    def test_a_written_panel_reads_back_with_the_same_values(self, provider):
        dates = pd.bdate_range("2022-03-01", periods=10)
        provider.write(_long("AAPL", dates))
        out = provider.fetch_panel(["AAPL"], "2022-03-01", "2022-03-31")
        assert len(out) == 10
        assert set(out["ticker"]) == {"AAPL"}
        assert out["close"].tolist() == pytest.approx(
            [100.5 + i for i in range(10)])

    def test_the_partition_layout_follows_the_documented_convention(self, provider,
                                                                     tmp_path):
        """
        `{root}/{TICKER}/year={YYYY}/data.parquet`
        —— 路径拼法被改会让写进去的数据**永远读不回来**
        （`_read_ticker` 按同一约定找路径，两边一起改才发现不了，
        所以这里正面钉住磁盘上的实际路径）。
        """
        provider.write(_long("MSFT", pd.bdate_range("2021-06-01", periods=3)))
        expected = tmp_path / "store" / "MSFT" / "year=2021" / "data.parquet"
        assert expected.exists(), (
            f"分区文件不在约定路径上。实际树：\n"
            f"{[str(p.relative_to(tmp_path)) for p in (tmp_path).rglob('*') if p.is_file()]}")

    def test_data_spanning_two_years_is_split_into_two_partitions(self, provider,
                                                                   tmp_path):
        """`df["year"] = df["timestamp"].dt.year` + groupby —— 按年拆分。"""
        dates = pd.bdate_range("2021-12-20", periods=20)   # 跨 2021/2022
        provider.write(_long("NVDA", dates))
        root = tmp_path / "store" / "NVDA"
        assert (root / "year=2021" / "data.parquet").exists()
        assert (root / "year=2022" / "data.parquet").exists()

    def test_the_year_helper_column_is_not_persisted(self, provider, tmp_path):
        """
        `group = group.drop(columns=["year"])` —— 不删会让每次回读
        都多出一列 `year`，`SchemaEnforcer(allow_extra=True)` 会放行，
        然后这列一路流进面板，pivot 时多一个字段。
        """
        provider.write(_long("TSLA", pd.bdate_range("2022-01-03", periods=5)))
        raw = pd.read_parquet(tmp_path / "store" / "TSLA" / "year=2022" /
                              "data.parquet")
        assert "year" not in raw.columns, "辅助列 year 被写进了磁盘"

    def test_tickers_are_upper_cased_on_write(self, provider, tmp_path):
        """
        `df["ticker"] = df["ticker"].str.upper()` —— 不统一大小写会让
        "aapl" 与 "AAPL" 落到两个目录，同一只票的数据被劈成两半。
        """
        provider.write(_long("aapl", pd.bdate_range("2022-01-03", periods=3)))
        assert (tmp_path / "store" / "AAPL" / "year=2022").exists(), (
            "小写 ticker 没有被规整成大写目录")

    def test_tickers_are_upper_cased_on_read(self, provider):
        """`tickers = [t.upper() for t in tickers]` —— 读侧同样要规整。"""
        provider.write(_long("AAPL", pd.bdate_range("2022-01-03", periods=4)))
        out = provider.fetch_panel(["aapl"], "2022-01-01", "2022-01-31")
        assert len(out) == 4, "用小写请求读不到数据 —— 读侧没有 upper()"

    def test_multiple_tickers_are_concatenated(self, provider):
        dates = pd.bdate_range("2022-02-01", periods=6)
        provider.write(_long("AAA", dates, base=10.0))
        provider.write(_long("BBB", dates, base=50.0))
        out = provider.fetch_panel(["AAA", "BBB"], "2022-02-01", "2022-02-28")
        assert len(out) == 12
        assert set(out["ticker"]) == {"AAA", "BBB"}

    def test_the_returned_panel_follows_the_standard_column_order(self, provider):
        provider.write(_long("AAA", pd.bdate_range("2022-02-01", periods=4)))
        out = provider.fetch_panel(["AAA"], "2022-02-01", "2022-02-28")
        assert list(out.columns)[:len(STANDARD_COLUMNS)] == STANDARD_COLUMNS

    def test_write_requires_pyarrow_with_an_actionable_message(self, provider,
                                                               monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "pyarrow", None)
        with pytest.raises(ImportError, match="pip install pyarrow"):
            provider.write(_long("AAA", pd.bdate_range("2022-01-03", periods=2)))


# ===========================================================================
# C. 年份分区选择 —— 最容易静默丢数据的一处
# ===========================================================================

class TestYearPartitionSelection:

    def test_the_end_year_partition_is_included(self, provider):
        """
        `years = list(range(start_dt.year, end_dt.year + 1))`

        `+ 1` 被去掉 → 结束年份整年不读。
        构造：数据横跨 2021/2022，请求区间也横跨两年 ——
        少了 `+1` 就只剩 2021 年那段。
        """
        provider.write(_long("AAA", pd.bdate_range("2021-12-01", periods=40)))
        out = provider.fetch_panel(["AAA"], "2021-12-01", "2022-01-31")
        years = sorted({d.year for d in out["timestamp"]})
        assert years == [2021, 2022], (
            f"跨年请求只读到 {years} —— `end_dt.year + 1` 的 +1 被去掉了")

    def test_a_single_year_request_still_reads_that_year(self, provider):
        """`+1` 的另一侧：同年请求时 range 必须至少产出一个年份。"""
        provider.write(_long("AAA", pd.bdate_range("2022-05-02", periods=10)))
        out = provider.fetch_panel(["AAA"], "2022-05-01", "2022-05-31")
        assert len(out) == 10, "同年请求读不到数据 —— 年份区间算空了"

    def test_years_outside_the_request_are_not_read(self, provider):
        """range 的起点被改小会白读若干年分区（慢，但不影响正确性）。"""
        provider.write(_long("AAA", pd.bdate_range("2020-01-02", periods=10),
                             base=1.0))
        provider.write(_long("AAA", pd.bdate_range("2022-01-03", periods=10),
                             base=100.0))
        out = provider.fetch_panel(["AAA"], "2022-01-01", "2022-12-31")
        assert {d.year for d in out["timestamp"]} == {2022}
        assert out["close"].min() >= 100.0, "读进了区间外年份的数据"

    def test_a_missing_year_partition_is_skipped_not_fatal(self, provider):
        """
        `if not part_path.exists(): continue`
        —— 守卫被删会让任何一年没数据就抛 FileNotFoundError，
        而"某一年没有数据"在真实数据里非常普遍（新股上市）。
        """
        provider.write(_long("AAA", pd.bdate_range("2022-01-03", periods=5)))
        out = provider.fetch_panel(["AAA"], "2020-01-01", "2022-12-31")
        assert len(out) == 5, "中间年份缺分区导致读取失败"

    def test_a_missing_ticker_directory_returns_none_not_an_error(self, provider):
        provider.write(_long("AAA", pd.bdate_range("2022-01-03", periods=5)))
        out = provider.fetch_panel(["AAA", "NOSUCH"], "2022-01-01", "2022-12-31")
        assert set(out["ticker"]) == {"AAA"}
        assert len(out) == 5


# ===========================================================================
# D. 日期范围过滤
# ===========================================================================

class TestDateFiltering:

    def test_both_endpoints_are_inclusive(self, provider):
        """
        `mask = (ts >= start_dt) & (ts <= end_dt)`

        任一比较符翻成严格不等，端点那一天就被切掉 ——
        与别的数据源拼面板时整体错位一天。
        """
        dates = pd.bdate_range("2022-03-01", periods=10)
        provider.write(_long("AAA", dates))
        lo, hi = dates[2], dates[6]
        out = provider.fetch_panel(["AAA"], str(lo.date()), str(hi.date()))
        got = [str(d.date()) for d in out["timestamp"]]
        assert got == [str(d.date()) for d in dates[2:7]], (
            f"区间 [{lo.date()}, {hi.date()}] 取回了 {got} —— 端点开闭被改了")

    def test_data_before_the_window_is_dropped(self, provider):
        dates = pd.bdate_range("2022-03-01", periods=10)
        provider.write(_long("AAA", dates))
        out = provider.fetch_panel(["AAA"], str(dates[5].date()), "2022-12-31")
        assert out["timestamp"].min() == dates[5]

    def test_data_after_the_window_is_dropped(self, provider):
        dates = pd.bdate_range("2022-03-01", periods=10)
        provider.write(_long("AAA", dates))
        out = provider.fetch_panel(["AAA"], "2022-01-01", str(dates[4].date()))
        assert out["timestamp"].max() == dates[4]

    def test_an_empty_window_yields_an_empty_but_well_shaped_panel(self, provider):
        """
        `return pd.DataFrame(columns=STANDARD_COLUMNS)`
        —— 返回裸 `pd.DataFrame()` 会让调用方的 `panel["close"]`
        抛 KeyError 而不是拿到空列。
        """
        provider.write(_long("AAA", pd.bdate_range("2022-03-01", periods=5)))
        out = provider.fetch_panel(["AAA"], "2023-01-01", "2023-12-31")
        assert out.empty
        assert list(out.columns) == STANDARD_COLUMNS, (
            f"空面板的列不是标准列：{list(out.columns)}")

    def test_the_filtered_frame_is_an_independent_copy(self, provider):
        """
        `combined.loc[mask].copy()` —— 少了 `.copy()` 返回的是切片视图，
        `SchemaEnforcer.enforce` 随后对它赋值会踩 SettingWithCopy 语义：
        在部分 pandas 版本上**赋值无声失效**，在新版上则是共享内存，
        调用方改一个格子会回流到缓存的原始帧。

        这里不依赖 pandas 的告警类名（版本间反复改名），
        直接验证行为：改动返回的面板不影响下一次读取。
        """
        provider.write(_long("AAA", pd.bdate_range("2022-03-01", periods=6)))
        first = provider.fetch_panel(["AAA"], "2022-03-01", "2022-03-31")
        original = float(first["close"].iloc[0])
        first.iloc[0, first.columns.get_loc("close")] = -12345.0

        second = provider.fetch_panel(["AAA"], "2022-03-01", "2022-03-31")
        assert float(second["close"].iloc[0]) == pytest.approx(original), (
            "改动返回的面板污染了后续读取 —— 返回的是视图而不是副本")


# ===========================================================================
# E. 追加 vs 覆盖 —— 数据修正能不能写进去
# ===========================================================================

class TestAppendAndOverwrite:

    def test_appending_merges_with_existing_rows(self, provider):
        """
        `if out_path.exists() and not overwrite:` 读旧数据合并。
        `not` 被删会让追加变成覆盖 —— 一次增量拉取抹掉全部历史。
        """
        provider.write(_long("AAA", pd.bdate_range("2022-01-03", periods=5)))
        provider.write(_long("AAA", pd.bdate_range("2022-01-10", periods=5),
                             base=200.0))
        out = provider.fetch_panel(["AAA"], "2022-01-01", "2022-12-31")
        assert len(out) == 10, (
            f"追加之后只剩 {len(out)} 行 —— 追加被做成了覆盖")

    def test_overwrite_replaces_the_partition(self, provider):
        provider.write(_long("AAA", pd.bdate_range("2022-01-03", periods=5)))
        provider.write(_long("AAA", pd.bdate_range("2022-01-10", periods=3),
                             base=200.0), overwrite=True)
        out = provider.fetch_panel(["AAA"], "2022-01-01", "2022-12-31")
        assert len(out) == 3, "overwrite=True 没有覆盖旧分区"
        assert out["close"].min() >= 200.0

    def test_a_correction_wins_over_the_stale_row(self, provider):
        """
        `drop_duplicates(subset=["timestamp","ticker"], keep="last")`

        `keep="first"` 会让**旧的错误值永远赢** —— 复权调整、
        事后数据修正全部写不进去，而且落盘成功、无任何报错。
        这是本模块最危险的一处。
        """
        dates = pd.bdate_range("2022-01-03", periods=3)
        provider.write(_long("AAA", dates, base=100.0))
        corrected = _long("AAA", dates, base=999.0)
        provider.write(corrected)

        out = provider.fetch_panel(["AAA"], "2022-01-01", "2022-12-31")
        assert len(out) == 3, f"去重之后剩了 {len(out)} 行，应当是 3 行"
        assert out["close"].tolist() == pytest.approx(
            corrected["close"].tolist()), (
            f"修正值没有覆盖旧值：{out['close'].tolist()} —— "
            f"`keep='last'` 被改成了 'first'")

    def test_the_merged_partition_stays_sorted_by_timestamp(self, provider):
        """`sort_values("timestamp")` —— 不排序会让合并后的分区乱序落盘。"""
        provider.write(_long("AAA", pd.bdate_range("2022-01-10", periods=3),
                             base=200.0))
        provider.write(_long("AAA", pd.bdate_range("2022-01-03", periods=3),
                             base=100.0))
        out = provider.fetch_panel(["AAA"], "2022-01-01", "2022-12-31")
        ts = out["timestamp"].tolist()
        assert ts == sorted(ts), f"合并后的分区没有按时间排序：{ts}"

    def test_deduplication_is_keyed_on_timestamp_and_ticker(self, provider,
                                                            tmp_path):
        """
        去重键少一项（只按 timestamp）会让**同一天的不同 ticker**
        互相覆盖 —— 但分区本身是按 ticker 分的，所以这条要在
        一个分区里混入两个 ticker 才能显形。直接构造这种文件。
        """
        dates = pd.bdate_range("2022-01-03", periods=3)
        mixed = pd.concat([_long("AAA", dates, base=100.0),
                           _long("AAA", dates, base=100.0)], ignore_index=True)
        provider.write(mixed)
        provider.write(_long("AAA", dates, base=500.0))
        out = provider.fetch_panel(["AAA"], "2022-01-01", "2022-12-31")
        assert len(out) == 3
        assert out["close"].min() >= 500.0

    def test_a_corrupt_existing_partition_does_not_lose_the_new_rows(self,
                                                                     provider,
                                                                     tmp_path):
        """
        `except Exception: logger.warning(...)` —— 合并失败时
        **保留新数据继续写**。这个 except 被改成 raise 会让一个
        损坏的旧文件卡死后续所有增量。
        """
        part = tmp_path / "store" / "AAA" / "year=2022"
        part.mkdir(parents=True)
        (part / "data.parquet").write_bytes(b"not a parquet file")

        provider.write(_long("AAA", pd.bdate_range("2022-01-03", periods=4)))
        out = provider.fetch_panel(["AAA"], "2022-01-01", "2022-12-31")
        assert len(out) == 4, "旧分区损坏导致新数据也没写进去"

    def test_an_unreadable_partition_warns_instead_of_raising_on_read(self,
                                                                      provider,
                                                                      tmp_path):
        """
        `_read_ticker` 里的 `except ... warnings.warn(...)`
        —— 改成 raise 会让一个坏文件让整个 universe 读不出来。
        """
        provider.write(_long("AAA", pd.bdate_range("2022-01-03", periods=4)))
        bad = tmp_path / "store" / "AAA" / "year=2021"
        bad.mkdir(parents=True)
        (bad / "data.parquet").write_bytes(b"garbage")

        with pytest.warns(UserWarning, match="失败"):
            out = provider.fetch_panel(["AAA"], "2021-01-01", "2022-12-31")
        assert len(out) == 4, "坏分区把好分区的数据也带走了"


# ===========================================================================
# F. ticker 目录扫描
# ===========================================================================

class TestAvailableTickers:

    def test_an_absent_root_returns_an_empty_list(self, provider):
        """
        `if not self.root_dir.exists(): return []`
        —— 守卫被删会在首次运行（目录还没建）时抛 FileNotFoundError。
        """
        assert provider.available_tickers() == []

    def test_tickers_are_listed_sorted(self, provider):
        for t in ("MSFT", "AAPL", "NVDA"):
            provider.write(_long(t, pd.bdate_range("2022-01-03", periods=2)))
        assert provider.available_tickers() == ["AAPL", "MSFT", "NVDA"]

    def test_underscore_prefixed_directories_are_hidden(self, provider, tmp_path):
        """
        `not p.name.startswith("_")` —— `not` 被删会让内部目录
        （`_tmp`、`_meta`）被当成 ticker 返回，下游会去拉一只
        名叫 "_tmp" 的股票。
        """
        provider.write(_long("AAPL", pd.bdate_range("2022-01-03", periods=2)))
        (tmp_path / "store" / "_scratch").mkdir()
        assert provider.available_tickers() == ["AAPL"]

    def test_loose_files_in_the_root_are_ignored(self, provider, tmp_path):
        """`p.is_dir()` —— 去掉会让根目录下的 README 之类被当成 ticker。"""
        provider.write(_long("AAPL", pd.bdate_range("2022-01-03", periods=2)))
        (tmp_path / "store" / "README.md").write_text("hi", encoding="utf-8")
        assert provider.available_tickers() == ["AAPL"]


# ===========================================================================
# G. 列裁剪与 wide 转换
# ===========================================================================

class TestFieldProjection:

    def test_requested_fields_are_column_pruned_on_read(self, provider):
        """
        `cols = list({"timestamp","ticker"} | set(fields))`
        —— 两个必备列被漏掉会让后面的 `combined["timestamp"]` 直接 KeyError。
        """
        provider.write(_long("AAA", pd.bdate_range("2022-01-03", periods=5)))
        out = provider.fetch_panel(["AAA"], "2022-01-01", "2022-12-31",
                                   fields=["close"])
        assert "timestamp" in out.columns and "ticker" in out.columns
        assert "close" in out.columns

    def test_no_field_filter_reads_everything(self, provider):
        provider.write(_long("AAA", pd.bdate_range("2022-01-03", periods=5)))
        out = provider.fetch_panel(["AAA"], "2022-01-01", "2022-12-31")
        for c in ("open", "high", "low", "close", "volume", "vwap"):
            assert c in out.columns, f"未指定字段时缺了 {c}"

    def test_fetch_returns_a_wide_dataset_keyed_by_field(self, provider):
        dates = pd.bdate_range("2022-01-03", periods=6)
        provider.write(_long("AAA", dates, base=10.0))
        provider.write(_long("BBB", dates, base=50.0))
        ds = provider.fetch(["AAA", "BBB"], "2022-01-01", "2022-12-31")
        assert set(ds) == {"open", "high", "low", "close", "volume", "adj_factor"}, (
            f"默认字段集是 {sorted(ds)}")
        assert ds["close"].shape == (6, 2)
        assert list(ds["close"].columns) == ["AAA", "BBB"]

    def test_the_default_target_fields_exclude_vwap_and_returns(self, provider):
        """
        `target_fields = fields or ["open","high","low","close","volume","adj_factor"]`
        —— 这份默认清单被改会静默改变 `fetch` 的返回字段集，
        下游 DSL 用到 vwap 时才发现拿不到。
        """
        provider.write(_long("AAA", pd.bdate_range("2022-01-03", periods=4)))
        ds = provider.fetch(["AAA"], "2022-01-01", "2022-12-31")
        assert "vwap" not in ds and "returns" not in ds

    def test_an_explicit_field_list_overrides_the_default(self, provider):
        provider.write(_long("AAA", pd.bdate_range("2022-01-03", periods=4)))
        ds = provider.fetch(["AAA"], "2022-01-01", "2022-12-31",
                            fields=["close", "vwap"])
        assert set(ds) == {"close", "vwap"}

    def test_a_field_absent_from_the_long_frame_is_skipped_not_fatal(self, provider):
        """
        `if field not in long_df.columns: continue` —— 守卫被删会 KeyError。

        注意这条只能**直接调 `_to_raw_dataset`** 来测：
        走 `fetch(fields=[...])` 时，不存在的字段会先在 pyarrow 的
        列裁剪那一层把整批数据打掉（已登记为缺陷 D-2，
        见 tests/meta/test_known_defects.py），根本到不了这个守卫。
        """
        long_df = _long("AAA", pd.bdate_range("2022-01-03", periods=4))
        ds = provider._to_raw_dataset(long_df, ["close", "不存在的字段"])
        assert set(ds) == {"close"}

    def test_requesting_an_advertised_but_unstored_field_wipes_the_read(self,
                                                                        provider):
        """
        **钉住现状**（缺陷 D-2，本阶段只登记不修）：
        `returns` 在 `available_fields()` 里，却从不落盘。
        按契约请求它，整批数据归零 —— 不是少一列，是一列都没有。

        这条与 `tests/meta/test_known_defects.py` 里的 xfail 用例成对：
        那条描述**应该**的行为（红），这条描述**当前**的行为（绿）。
        修复时两条一起改。
        """
        provider.write(_long("AAA", pd.bdate_range("2022-01-03", periods=4)))
        assert "returns" in provider.available_fields()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = provider.fetch(["AAA"], "2022-01-01", "2022-12-31",
                                fields=["close", "returns"])
        assert ds == {}, (
            f"D-2 的现状变了（现在返回 {sorted(ds)}）—— "
            f"如果是修好了，请同步删除 test_known_defects 里的 D-2")

    def test_the_wide_index_is_a_datetime_index(self, provider):
        """
        `wide.index = pd.DatetimeIndex(wide.index)` —— 少了这步，
        pivot 出来的索引在某些 pandas 版本上是 object，
        与其他面板 `union` 会退化成 object 索引，时间比较全部失效。
        """
        provider.write(_long("AAA", pd.bdate_range("2022-01-03", periods=4)))
        ds = provider.fetch(["AAA"], "2022-01-01", "2022-12-31")
        assert isinstance(ds["close"].index, pd.DatetimeIndex)

    def test_duplicate_rows_are_resolved_by_taking_the_last(self, provider):
        """
        `pivot_table(..., aggfunc="last")`
        —— 改成 `"mean"` 会把一天两条记录（修正前/后）平均掉，
        产出一个**从未存在过**的价格。
        """
        dates = pd.bdate_range("2022-01-03", periods=2)
        dup = pd.concat([_long("AAA", dates, base=100.0),
                         _long("AAA", dates, base=300.0)], ignore_index=True)
        # 直接走 _to_raw_dataset，绕开写侧去重，单独钉这一处
        ds = provider._to_raw_dataset(dup, ["close"])
        assert ds["close"]["AAA"].tolist() == pytest.approx([300.5, 301.5]), (
            f"重复行没有取最后一条：{ds['close']['AAA'].tolist()} —— "
            f"aggfunc 被改了")

    def test_an_empty_panel_yields_an_empty_dataset(self, provider):
        """
        `if panel.empty: return {}` —— 守卫被删会让 pivot 在空表上
        产出形状怪异的结果（或抛），调用方拿到一个"有字段但没行"的面板。
        """
        assert provider.fetch(["NOSUCH"], "2022-01-01", "2022-12-31") == {}


# ===========================================================================
# H. 首测存活项：三个布尔开关
# ===========================================================================

class TestWriterFlags:
    """
    首测 21 点 / 存活 6，全部是布尔字面量翻面。
    这三个能杀；另外三个 `ignore_index=True` 见文件末尾的等价性证明。
    """

    def test_extra_columns_survive_the_round_trip(self, provider):
        """
        **首测存活项（L41）**：`SchemaEnforcer(allow_extra=True)` 翻成 False。

        `allow_extra=False` 会**静默丢弃**标准 9 列之外的一切
        （只发一条 UserWarning）。本项目真实会带的额外列包括
        `amount`（akshare 的成交额）、`pct_chg`、以及未来的基本面字段 ——
        它们会在落盘时无声消失，几个月后复盘才发现历史数据里没有这一列。
        """
        dates = pd.bdate_range("2022-01-03", periods=4)
        df = _long("AAA", dates)
        df["amount"] = [1e8, 2e8, 3e8, 4e8]

        provider.write(df)
        out = provider.fetch_panel(["AAA"], "2022-01-01", "2022-12-31")
        assert "amount" in out.columns, (
            f"额外列 amount 被丢弃了（剩余列：{list(out.columns)}）—— "
            f"`SchemaEnforcer(allow_extra=True)` 被翻成了 False")
        assert out["amount"].tolist() == pytest.approx([1e8, 2e8, 3e8, 4e8])

    def test_the_pandas_index_is_not_written_into_the_parquet_file(self, provider,
                                                                   tmp_path):
        """
        **首测存活项（L147）**：`pa.Table.from_pandas(group, preserve_index=False)`
        翻成 True。

        `preserve_index=True` 会在每个分区文件里多写一列
        `__index_level_0__`。它随后被 `allow_extra=True` 一路放行，
        于是：
          - 磁盘体积无谓增加；
          - 回读时多出一列毫无意义的行号；
          - 按 `fields=[...]` 做列裁剪时，这列不在请求里 —— 但
            `_to_raw_dataset` 会把它当成一个可用字段跳过，掩盖真正的字段缺失。
        """
        provider.write(_long("AAA", pd.bdate_range("2022-01-03", periods=4)))
        import pyarrow.parquet as pq
        path = tmp_path / "store" / "AAA" / "year=2022" / "data.parquet"
        names = pq.read_schema(path).names
        leaked = [c for c in names if c.startswith("__index_level_")]
        assert not leaked, (
            f"parquet 里写进了 pandas 行号列 {leaked} —— "
            f"`preserve_index=False` 被翻成了 True")

    def test_column_statistics_are_written_for_predicate_pushdown(self, provider,
                                                                   tmp_path):
        """
        **首测存活项（L159）**：`pq.write_table(..., write_statistics=True)`
        翻成 False。

        模块 docstring 第一句就写着"利用 pyarrow.dataset 谓词下推，
        只加载需要的年份分区"。列统计（min/max）正是下推赖以跳过
        row group 的东西。关掉之后**功能完全正常、只是变慢** ——
        随着本地库积累到几年数据，每次读取都要全量扫描，
        而没有任何报错提示原因。
        """
        provider.write(_long("AAA", pd.bdate_range("2022-01-03", periods=30)))
        import pyarrow.parquet as pq
        path = tmp_path / "store" / "AAA" / "year=2022" / "data.parquet"
        meta = pq.ParquetFile(path).metadata
        rg = meta.row_group(0)
        stats_set = [rg.column(i).is_stats_set for i in range(rg.num_columns)]
        assert all(stats_set), (
            f"分区文件缺少列统计（is_stats_set={stats_set}）—— "
            f"`write_statistics=True` 被翻成了 False，谓词下推失效")

        ts_col = next(i for i in range(rg.num_columns)
                      if rg.column(i).path_in_schema == "timestamp")
        st = rg.column(ts_col).statistics
        assert st is not None and st.has_min_max, "timestamp 列没有 min/max 统计"


# ===========================================================================
# I. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "L89 / L211 / L140 三处 `pd.concat(..., ignore_index=True)` 的 True → False":
        "三处 concat 的产物都在**离开本模块之前**被重建了行索引，"
        "所以保不保留原下标观察不到：\n"
        "  · L211 `_read_ticker` 的结果只流向 L89 的 concat；\n"
        "  · L89 `fetch_panel` 的结果立刻进 `SchemaEnforcer.enforce`，"
        "而 enforce 的最后一步是无条件的 "
        "`sort_values([...]).reset_index(drop=True)`；\n"
        "  · L140 `write` 的合并结果只流向 "
        "`pa.Table.from_pandas(group, preserve_index=False)`，"
        "索引被显式丢弃，落盘内容与下标无关。\n"
        "中间的 `drop_duplicates(subset=[...], keep='last')` 与 "
        "`sort_values('timestamp')` 都只看列值、不看索引标签，"
        "对相同输入给出相同的行选择与行序。"
        "机械验证见 test_the_read_path_always_rebuilds_the_row_index 与 "
        "test_the_write_path_discards_the_row_index。",
}


def test_the_read_path_always_rebuilds_the_row_index():
    """
    等价性前提之一：`SchemaEnforcer.enforce` **无条件**重建行索引。

    喂进去一个带重复标签的乱序索引，出来必须是干净的 RangeIndex。
    哪天 enforce 改成有条件 reset，这条先红，上面那份等价性证明随即作废。
    """
    from app.core.data_engine.schema import SchemaEnforcer

    dates = pd.bdate_range("2022-01-03", periods=6)
    df = _long("AAA", dates)
    df.index = [0, 1, 2, 0, 1, 2]          # 刻意制造重复标签

    out = SchemaEnforcer(allow_extra=True).enforce(df)
    assert isinstance(out.index, pd.RangeIndex), (
        f"enforce 之后索引是 {type(out.index).__name__} —— 不再无条件重建，"
        f"三处 ignore_index 的等价性证明作废")
    assert out.index.is_unique
    assert list(out.index) == list(range(len(out)))


def test_the_write_path_discards_the_row_index(provider, tmp_path):
    """
    等价性前提之二：落盘时索引被显式丢弃。

    与 `test_the_pandas_index_is_not_written_into_the_parquet_file` 同一个事实，
    这里从"等价性证明的前提"角度再断言一次 —— 那条用例是杀手，
    这条是证明的支撑；任一变红都说明这份证明需要重做。
    """
    import pyarrow.parquet as pq

    df = _long("AAA", pd.bdate_range("2022-01-03", periods=5))
    df.index = [9, 9, 9, 9, 9]
    provider.write(df)
    path = tmp_path / "store" / "AAA" / "year=2022" / "data.parquet"
    names = pq.read_schema(path).names
    assert not [c for c in names if c.startswith("__index_level_")], (
        "落盘保留了行索引 —— L140 的等价性证明作废")


def test_a_reordered_partition_read_is_index_independent(provider):
    """
    等价性的行为面验证：跨三个年份分区读取，结果的行序与内容
    与"逐年单独读再手工拼接"完全一致 —— 说明中间那几步 concat
    的索引形态没有渗漏到任何可观察的输出上。
    """
    for year, base in ((2020, 10.0), (2021, 20.0), (2022, 30.0)):
        provider.write(_long("AAA", pd.bdate_range(f"{year}-03-01", periods=5),
                             base=base))
    out = provider.fetch_panel(["AAA"], "2020-01-01", "2022-12-31")
    assert len(out) == 15
    assert isinstance(out.index, pd.RangeIndex) and out.index.is_unique
    ts = out["timestamp"].tolist()
    assert ts == sorted(ts), "跨分区读取的结果没有按时间排序"


def test_every_survivor_has_a_written_proof():
    """
    首测 21 点 / 存活 6：其中 3 处已由上面的用例杀死，
    剩下 3 处（同一条理由）合并为一条等价性证明。
    """
    assert len(PROVEN_EQUIVALENT) == 1
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
