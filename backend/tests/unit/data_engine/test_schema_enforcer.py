"""
data_engine/schema.py —— 所有数据源进入面板工厂前的**唯一**强制层

**此前零测试**（149 有效行，D 档）。

模块 docstring 写着"所有数据源的输出在进入面板工厂前，都必须经过
SchemaEnforcer 转换为统一的 long-format"。也就是说 yahoo / moomoo /
akshare / ccxt / 本地 parquet 的差异全靠它抹平。它的任何一步失效，
脏数据会直接流进回测，而回测本身不会报错 —— 只会给出一个看着正常的
Sharpe。

十步流程里每一步都是一个可变异点：
  列名规范化 → 必填列检查 → 时间戳归一 → ticker 归一 →
  补缺失列 → 数值转换 → 多余列处理 → 列序重排 → 去重 → 排序

测法与 A/B/C 一致：每一步都用**只有那一步能修复**的脏输入去驱动，
并且断言**确切结果**而不是"没报错"。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.schema import (
    NUMERIC_FIELDS, PRICE_FIELDS, STANDARD_COLUMNS, SchemaEnforcer,
    wide_to_long,
)
import app.core.data_engine.schema as S


def _raw(**over) -> pd.DataFrame:
    """一份已经干净的 long-format 输入；只覆盖要测的那一列。"""
    base = {
        "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "ticker":    ["AAPL", "AAPL"],
        "open":      [100.0, 101.0],
        "high":      [102.0, 103.0],
        "low":       [99.0, 100.0],
        "close":     [101.0, 102.0],
        "volume":    [1e6, 1.1e6],
    }
    base.update(over)
    return pd.DataFrame(base)


# ===========================================================================
# A. 必填列与空输入
# ===========================================================================

class TestRequiredColumns:

    @pytest.mark.parametrize("drop", ["timestamp", "ticker"])
    def test_a_missing_required_column_is_refused_loudly(self, drop):
        """
        `if missing: raise SchemaError(...)` —— 删掉这道闸，缺 ticker 的
        数据会一路走到 `df["ticker"].astype(str)` → KeyError，
        报错指向实现细节而不是"你的数据少了一列"。
        """
        df = _raw().drop(columns=[drop])
        with pytest.raises(S.SchemaError) as ei:
            SchemaEnforcer().enforce(df)
        assert drop in str(ei.value), (
            f"报错没有指出缺的是哪一列：{ei.value}")

    def test_the_error_lists_the_columns_actually_present(self):
        """报错要给出"当前列"，否则排查时还得再跑一遍看数据长什么样。"""
        df = _raw().drop(columns=["ticker"])
        with pytest.raises(S.SchemaError) as ei:
            SchemaEnforcer().enforce(df)
        assert "timestamp" in str(ei.value), (
            f"报错没有列出当前已有的列：{ei.value}")

    def test_a_complete_frame_is_accepted(self):
        """反向：必填列齐全时不得抛 —— 否则这道闸变成恒拒。"""
        out = SchemaEnforcer().enforce(_raw())
        assert len(out) == 2

    @pytest.mark.parametrize("empty", [None, "empty_df"])
    def test_empty_input_returns_a_typed_empty_frame(self, empty):
        """
        `if df is None or df.empty: return self._empty_frame()`

        `or` 收紧成 `and` 会让 `None` 输入走到 `df.copy()` → AttributeError。
        而且返回的空表必须**带正确 dtype** —— 下游 concat 时
        object 列和 float 列混在一起会静默产生 object 面板。
        """
        df = None if empty is None else pd.DataFrame()
        out = SchemaEnforcer().enforce(df)
        assert list(out.columns) == STANDARD_COLUMNS, (
            f"空输入返回的列不对：{list(out.columns)}")
        assert len(out) == 0
        assert str(out["close"].dtype) == "float64", (
            f"空表的 close 列 dtype 是 {out['close'].dtype} —— "
            f"下游 concat 会得到 object 面板")
        assert str(out["timestamp"].dtype).startswith("datetime64"), (
            f"空表的 timestamp dtype 是 {out['timestamp'].dtype}")


# ===========================================================================
# B. 列名与 ticker 归一
# ===========================================================================

class TestNormalisation:

    def test_column_names_are_lowercased_and_stripped(self):
        """
        `df.columns = [str(c).strip().lower() for c in df.columns]`

        不同数据源给的是 `Close` / `CLOSE` / `" close "`。
        少了 `.lower()` 或 `.strip()`，后面"补缺失标准列"那步会认为
        `close` 不存在 → **整列变成 NaN**，而原始的 `Close` 列被当成多余列。
        回测拿到全 NaN 的收盘价，结果是一堆 0 而不是报错。
        """
        df = _raw().rename(columns={"close": " CLOSE ", "volume": "Volume"})
        out = SchemaEnforcer().enforce(df)
        assert out["close"].notna().all(), (
            "大小写/空格不同的列名没有被归一 —— close 整列变成了 NaN")
        assert out["close"].tolist() == [101.0, 102.0]
        assert out["volume"].notna().all()

    def test_tickers_are_uppercased_and_stripped(self):
        """
        `df["ticker"].astype(str).str.strip().str.upper()`

        同一只票在不同源里可能是 `aapl` / `" AAPL "`。不归一的话
        它们在面板里变成**两只不同的股票**，截面排名、行业中性化全错。
        """
        df = _raw(ticker=[" aapl ", "AAPL"])
        out = SchemaEnforcer().enforce(df)
        assert set(out["ticker"]) == {"AAPL"}, (
            f"ticker 没有被归一：{sorted(set(out['ticker']))} —— "
            f"同一只票会被当成两只")

    def test_non_string_tickers_are_coerced(self):
        """`.astype(str)` —— 数字代码（如 A 股 000001）不能变成 int。"""
        df = _raw(ticker=[600519, 600519])
        out = SchemaEnforcer().enforce(df)
        # 两行的 timestamp 不同，不会被去重 —— 这里只看类型转换本身
        assert out["ticker"].tolist() == ["600519", "600519"]
        assert all(isinstance(t, str) for t in out["ticker"]), (
            "数字代码没有被转成字符串 —— A 股代码会丢掉前导 0")


class TestTimestampCoercion:

    def test_timezone_aware_timestamps_become_naive_utc(self):
        """
        `if ts.dt.tz is not None: ts.dt.tz_convert("UTC").dt.tz_localize(None)`

        删掉这个判定，带时区的时间戳会与 naive 的混在同一列 →
        pandas 在 concat / 对齐时直接抛错，或者更糟：
        两个数据源的同一天被当成不同的两天。
        """
        df = _raw(timestamp=pd.to_datetime(
            ["2024-01-02T14:30:00-05:00", "2024-01-03T14:30:00-05:00"]))
        out = SchemaEnforcer().enforce(df)
        assert out["timestamp"].dt.tz is None, "时区没有被去掉"
        # 美东 14:30 = UTC 19:30 → 归一到当天
        assert out["timestamp"].tolist() == list(
            pd.to_datetime(["2024-01-02", "2024-01-03"]))

    def test_naive_timestamps_pass_through_unchanged(self):
        """
        `is not None` 这一半：naive 的时间戳不该被再转一次
        （`tz_convert` 对 naive 序列会抛 TypeError）。
        """
        out = SchemaEnforcer().enforce(_raw())
        assert out["timestamp"].tolist() == list(
            pd.to_datetime(["2024-01-02", "2024-01-03"]))

    def test_intraday_timestamps_are_normalised_to_the_day(self):
        """
        `ts.dt.normalize()` —— 去掉时间部分。少了它，同一天的两条
        日频记录（09:30 与 16:00）不会被去重，面板里同一天出现两行。
        """
        df = _raw(timestamp=pd.to_datetime(
            ["2024-01-02 09:30", "2024-01-02 16:00"]))
        out = SchemaEnforcer().enforce(df)
        assert out["timestamp"].nunique() == 1, (
            f"同一天的两个时点没有被归一到日：{out['timestamp'].tolist()}")

    def test_unparsable_timestamps_become_nat_not_an_exception(self):
        """`errors="coerce"` —— 坏行变 NaT，而不是让整批数据加载失败。"""
        df = _raw(timestamp=["2024-01-02", "not-a-date"])
        out = SchemaEnforcer().enforce(df)
        assert out["timestamp"].isna().sum() == 1


# ===========================================================================
# C. 缺失列补齐与默认值
# ===========================================================================

class TestMissingColumns:

    def test_every_standard_column_exists_after_enforcement(self):
        """
        `for col in STANDARD_COLUMNS: if col not in df.columns: df[col] = default`

        删掉 `not` 会反过来：**已有的**列被默认值覆盖（真实收盘价变成 NaN），
        缺失的列反而不补。前者是静默的数据毁坏。
        """
        out = SchemaEnforcer().enforce(_raw())
        for col in STANDARD_COLUMNS:
            assert col in out.columns, f"标准列 {col} 没有被补齐"
        # 已有的列必须保留原值，不能被默认值盖掉
        assert out["close"].tolist() == [101.0, 102.0], (
            "已存在的 close 列被默认值覆盖了 —— `if col not in df.columns` 的 not 被删掉了")

    def test_adj_factor_defaults_to_one_not_nan(self):
        """
        `_COLUMN_DEFAULTS = {"adj_factor": 1.0, "vwap": np.nan}`

        复权因子默认必须是 **1.0**（不复权），不能是 NaN ——
        NaN 会让 `close * adj_factor` 整列变 NaN。
        """
        out = SchemaEnforcer().enforce(_raw())
        assert out["adj_factor"].tolist() == [1.0, 1.0], (
            f"adj_factor 的默认值是 {out['adj_factor'].tolist()}，应当是 1.0 —— "
            f"NaN 会让复权后的价格整列变 NaN")

    def test_vwap_defaults_to_nan_so_it_can_be_synthesised_later(self):
        """
        vwap 默认是 **NaN**（留给 SyntheticFieldBuilder 后续合成），
        不能是 0.0 —— 0 会被当成"真的成交均价是 0"，
        后续的合成步骤看到非 NaN 就跳过了。
        """
        out = SchemaEnforcer().enforce(_raw())
        assert out["vwap"].isna().all(), (
            f"vwap 的默认值是 {out['vwap'].tolist()}，应当是 NaN —— "
            f"非 NaN 会让后续的 vwap 合成被跳过")

    def test_a_supplied_vwap_is_not_overwritten(self):
        out = SchemaEnforcer().enforce(_raw(vwap=[100.5, 101.5]))
        assert out["vwap"].tolist() == [100.5, 101.5]


# ===========================================================================
# D. 数值类型强制
# ===========================================================================

class TestNumericCoercion:

    def test_string_numbers_are_coerced_to_float(self):
        """
        `pd.to_numeric(df[col], errors="coerce").astype("float64")`

        CSV / JSON 源常常给字符串。不转换的话整列是 object，
        `close.pct_change()` 会抛 TypeError，或者更糟 ——
        字符串比较让排序结果完全错乱（"9" > "10"）。
        """
        df = _raw(close=["101.0", "102.0"], volume=["1000000", "1100000"])
        out = SchemaEnforcer().enforce(df)
        assert str(out["close"].dtype) == "float64"
        assert out["close"].tolist() == [101.0, 102.0]

    def test_unparsable_numbers_become_nan_not_an_exception(self):
        """`errors="coerce"` —— 一个坏格子不该让整批数据加载失败。"""
        df = _raw(close=["101.0", "N/A"])
        out = SchemaEnforcer().enforce(df)
        assert out["close"].isna().sum() == 1
        assert out["close"].iloc[0] == 101.0

    @pytest.mark.parametrize("col", NUMERIC_FIELDS)
    def test_every_numeric_field_ends_up_float64(self, col):
        """
        `for col in NUMERIC_FIELDS:` —— 遍历的是 NUMERIC_FIELDS 而不是
        STANDARD_COLUMNS。漏掉任何一个字段，它会以 object dtype 流下去。
        """
        out = SchemaEnforcer().enforce(_raw())
        assert str(out[col].dtype) == "float64", (
            f"{col} 的 dtype 是 {out[col].dtype}，应当是 float64")

    def test_numeric_fields_cover_all_prices_plus_volume_and_adj(self):
        """
        `NUMERIC_FIELDS = PRICE_FIELDS + ["volume", "adj_factor"]`
        —— 这个拼接一旦被改（比如 `+` 变成别的），就会有字段漏掉类型转换。
        """
        assert set(NUMERIC_FIELDS) == set(PRICE_FIELDS) | {"volume", "adj_factor"}
        assert "close" in PRICE_FIELDS and "vwap" in PRICE_FIELDS
        assert "ticker" not in NUMERIC_FIELDS and "timestamp" not in NUMERIC_FIELDS


# ===========================================================================
# E. 多余列、列序、去重、排序
# ===========================================================================

class TestExtraColumnsAndOrdering:

    def test_extra_columns_are_kept_by_default(self):
        """
        `if extra_cols and not self.allow_extra:` —— 默认 `allow_extra=True`，
        额外列（比如某个源特有的 `turnover_rate`）要保留。
        `and` 放宽成 `or` 会让默认配置下也丢列。
        """
        df = _raw()
        df["turnover_rate"] = [0.01, 0.02]
        out = SchemaEnforcer().enforce(df)
        assert "turnover_rate" in out.columns, (
            "默认配置下额外列被丢掉了 —— `extra_cols and not allow_extra` 被改了")

    def test_extra_columns_are_dropped_when_disallowed(self):
        df = _raw()
        df["turnover_rate"] = [0.01, 0.02]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = SchemaEnforcer(allow_extra=False).enforce(df)
        assert "turnover_rate" not in out.columns
        assert any("多余列" in str(x.message) for x in w), (
            "丢列时没有告警 —— 静默丢数据")

    def test_no_warning_when_there_are_no_extra_columns(self):
        """
        `if extra_cols and ...` 的前半：没有多余列时不该告警。
        `or` 会让每一次 enforce 都刷一条"丢弃多余列 []"。
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SchemaEnforcer(allow_extra=False).enforce(_raw())
        assert not any("多余列" in str(x.message) for x in w), (
            "没有多余列却告警了")

    def test_standard_columns_come_first_in_the_declared_order(self):
        """
        `ordered = STANDARD_COLUMNS + [额外列]`

        列序是契约：下游有按位置取列的代码（`df.iloc[:, 2:7]` 之类）。
        `+` 两侧对调会把额外列排到前面，位置索引全错位。
        """
        df = _raw()
        df["zzz_extra"] = 1.0
        out = SchemaEnforcer().enforce(df)
        assert list(out.columns)[:len(STANDARD_COLUMNS)] == STANDARD_COLUMNS, (
            f"标准列没有排在最前：{list(out.columns)}")
        assert list(out.columns)[-1] == "zzz_extra"

    def test_duplicates_keep_the_last_row(self):
        """
        `drop_duplicates(subset=["timestamp","ticker"], keep="last")`

        `keep` 改成 "first" 会保留**旧**的那条 —— 数据源重发修正数据时，
        修正值被丢弃、错误值被留下，而且完全没有痕迹。
        """
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "ticker":    ["AAPL", "AAPL"],
            "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0],
            "close": [100.0, 999.0],     # 后一条是修正值
            "volume": [1e6, 1e6],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = SchemaEnforcer().enforce(df)
        assert len(out) == 1
        assert out["close"].iloc[0] == 999.0, (
            f"去重保留的是 {out['close'].iloc[0]}，应当是最后一条 999.0 —— "
            f"`keep='last'` 被改了，修正数据会被丢弃")

    def test_deduplication_warns_about_how_many_rows_were_removed(self):
        """
        `if len(df) < before:` —— **严格小于**。放宽成 `<=` 会让
        每次 enforce 都打出"去除重复行 0 条"的噪声告警。
        """
        df = pd.concat([_raw(), _raw()], ignore_index=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SchemaEnforcer().enforce(df)
        msgs = [str(x.message) for x in w if "重复行" in str(x.message)]
        assert msgs, "去掉了重复行却没有告警"
        assert "2" in msgs[0], f"告警没有报出去除的条数：{msgs[0]}"

    def test_no_duplicate_warning_when_there_are_none(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SchemaEnforcer().enforce(_raw())
        assert not any("重复行" in str(x.message) for x in w), (
            "没有重复行却告警了 —— `len(df) < before` 被放宽成了 `<=`")

    def test_output_is_sorted_by_timestamp_then_ticker(self):
        """
        `sort_values(["timestamp","ticker"])` —— 排序键的**顺序**是契约。
        两个键对调会让面板按 ticker 分块而不是按时间递增，
        `pivot` 之后的时间轴不再单调，所有滚动算子静默算错。
        """
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-03", "2024-01-02",
                                         "2024-01-03", "2024-01-02"]),
            "ticker":    ["MSFT", "MSFT", "AAPL", "AAPL"],
            "open": [1.0] * 4, "high": [1.0] * 4, "low": [1.0] * 4,
            "close": [1.0] * 4, "volume": [1.0] * 4,
        })
        out = SchemaEnforcer().enforce(df)
        assert out["timestamp"].is_monotonic_increasing, (
            f"输出没有按时间升序：{out['timestamp'].tolist()}")
        first_day = out[out["timestamp"] == out["timestamp"].min()]
        assert first_day["ticker"].tolist() == ["AAPL", "MSFT"], (
            f"同一天内没有按 ticker 排序：{first_day['ticker'].tolist()}")

    def test_the_index_is_reset(self):
        """`reset_index(drop=True)` —— 索引不重置会让下游 `iloc` 与 `loc` 打架。"""
        df = pd.concat([_raw(), _raw(ticker=["MSFT", "MSFT"])], ignore_index=True)
        out = SchemaEnforcer().enforce(df)
        assert out.index.tolist() == list(range(len(out)))


# ===========================================================================
# F. wide_to_long
# ===========================================================================

class TestWideToLong:

    @staticmethod
    def _wide():
        idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
        idx.name = "date"
        return {
            "close":  pd.DataFrame([[100.0, 200.0], [101.0, 201.0]],
                                   index=idx, columns=["AAPL", "MSFT"]),
            "volume": pd.DataFrame([[1e6, 2e6], [1.1e6, 2.1e6]],
                                   index=idx, columns=["AAPL", "MSFT"]),
        }

    def test_every_field_becomes_a_column(self):
        out = wide_to_long(self._wide())
        for col in ("timestamp", "ticker", "close", "volume"):
            assert col in out.columns, f"long 表缺列 {col}：{list(out.columns)}"

    def test_every_time_ticker_pair_becomes_a_row(self):
        """2 天 × 2 只票 = 4 行；少了说明某个 melt 丢了数据。"""
        out = wide_to_long(self._wide())
        assert len(out) == 4, f"2×2 的面板展平成了 {len(out)} 行"

    def test_values_survive_the_transposition(self):
        """
        展平最容易出的错是**行列错位**（把 ticker 当成时间）。
        逐格核对能抓到。
        """
        out = wide_to_long(self._wide()).set_index(["timestamp", "ticker"])
        key = (pd.Timestamp("2024-01-03"), "MSFT")
        assert out.loc[key, "close"] == 201.0, (
            f"展平后 {key} 的 close 是 {out.loc[key, 'close']}，应当是 201.0 —— "
            f"行列错位了")
        assert out.loc[key, "volume"] == 2.1e6

    def test_an_empty_input_returns_a_typed_empty_frame(self):
        """
        `if not frames: return pd.DataFrame(columns=STANDARD_COLUMNS)`
        —— 删掉 `not` 会让**有数据**时返回空表（数据静默消失），
        空输入时 `pd.concat([])` 抛 ValueError。
        """
        out = wide_to_long({})
        assert len(out) == 0
        assert list(out.columns) == STANDARD_COLUMNS

    def test_a_non_empty_input_is_not_emptied(self):
        """`not frames` 的另一侧 —— 这条是上面那条的反向不变式。"""
        assert len(wide_to_long(self._wide())) > 0, (
            "有数据的输入被展平成了空表 —— `if not frames` 的 not 被删掉了")

    def test_the_result_passes_schema_enforcement(self):
        """端到端：wide_to_long 的产物必须能直接喂给 SchemaEnforcer。"""
        out = SchemaEnforcer().enforce(wide_to_long(self._wide()))
        assert len(out) == 4
        assert list(out.columns)[:len(STANDARD_COLUMNS)] == STANDARD_COLUMNS
        assert out["close"].notna().all()


# ===========================================================================
# G. 第二轮：to_datetime 的 utc 参数
# ===========================================================================

class TestTimestampUtcFlag:
    """
    `pd.to_datetime(series, utc=False, errors="coerce")`

    `utc=False` → `True` 在**单一时区**与**naive** 输入上给出完全相同的
    最终结果（后面那两步 `tz_convert("UTC").tz_localize(None).normalize()`
    把差别抹平了）。唯一能分开的是**混合时区**输入：

      utc=False → pandas 抛 "Mixed timezones detected"（`errors="coerce"`
                  拦不住，它只处理逐元素的解析失败）
      utc=True  → 静默统一到 UTC

    也就是说：当前实现在遇到混合时区的数据源时是 **fail-loud** 的。
    这条用例把这个行为钉住 —— 它既是变异检出点，也是一条重要的现状记录：
    哪天真的接了跨区数据源（美股 + 港股同一批），这里会直接抛，
    而不是静默把两地的收盘时间对齐到错误的一天。
    """

    def test_mixed_timezone_input_fails_loudly_rather_than_silently_unifying(self):
        df = _raw(timestamp=["2024-01-02T14:30:00-05:00",
                             "2024-01-02T14:30:00+08:00"])
        with pytest.raises(Exception) as ei:
            SchemaEnforcer().enforce(df)
        assert "imezone" in str(ei.value) or "utc" in str(ei.value).lower(), (
            f"混合时区输入没有因为时区问题被拒：{ei.value} —— "
            f"`utc=False` 疑似被改成了 `utc=True`（会静默统一，"
            f"跨区数据源的日期对齐错误将无人察觉）")

    def test_single_timezone_input_still_works(self):
        """反向：单一时区（最常见的情形）必须正常通过。"""
        df = _raw(timestamp=["2024-01-02T14:30:00-05:00",
                             "2024-01-03T14:30:00-05:00"])
        out = SchemaEnforcer().enforce(df)
        assert out["timestamp"].tolist() == list(
            pd.to_datetime(["2024-01-02", "2024-01-03"]))
