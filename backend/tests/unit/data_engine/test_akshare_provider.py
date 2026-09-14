"""
data_engine/providers/akshare_provider.py —— A 股日线（前复权）

**此前零测试**（9 个变异点，D 档）。

A 股链路只有这一个入口，而且它**全是静默失败的形状**：

  - `adjust="qfq"` 被改成 `""` → 拿到未复权价，除权日凭空出现
    一根 -10% 的"暴跌"，动量因子直接被喂垃圾，没有任何报错；
  - 日期格式 `"2023-01-01" → "20230101"` 的 `replace` 被改 →
    akshare 收到它不认的格式，返回空表 → 抛 "no data returned"，
    看起来像"这批票没数据"，其实是格式错了；
  - `_COLUMN_MAP` 少一项 → 该字段在面板里变成全 NaN 列
    （`_field` 的 else 分支兜住了），**照样出结果**；
  - 重试三次里的 `if attempt < 2` 写错 → 要么最后一次也睡（白等），
    要么一次都不重试（限流时整批丢数据）；
  - `float(val) * 1e8` 的"亿"换算被改 → 市值差 1e8 倍，
    市值筛选把所有 A 股都划进 small_cap。

整条链路不碰网络：注入一个假的 `akshare` 模块。
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.providers.akshare_provider import (
    _COLUMN_MAP,
    AkshareProvider,
    _strip_suffix,
)

FIELDS = ("open", "high", "low", "close", "volume", "vwap", "returns")


def _ak_frame(days=6, start="2023-01-03", base=100.0) -> pd.DataFrame:
    """akshare 原始格式：中文列名、日期是字符串列。"""
    idx = pd.bdate_range(start, periods=days)
    return pd.DataFrame({
        "日期":   [str(d.date()) for d in idx],
        "开盘":   [base + i for i in range(days)],
        "收盘":   [base + i + 0.5 for i in range(days)],
        "最高":   [base + i + 1.0 for i in range(days)],
        "最低":   [base + i - 1.0 for i in range(days)],
        "成交量": [1e6 + i for i in range(days)],
        "成交额": [1e8 + i for i in range(days)],
        "涨跌幅": [0.1] * days,
    })


class _FakeAk:
    """可编程的假 akshare：记录调用参数、按 code 返回数据或抛异常。"""

    def __init__(self, per_code=None, fail_codes=(), fail_times=99, info=None):
        self.per_code = per_code if per_code is not None else {}
        self.fail_codes = set(fail_codes)
        self.fail_times = fail_times
        self.calls = []
        self.info_calls = []
        self._info = info or {}
        self._fail_count = {}

    def stock_zh_a_hist(self, symbol=None, period=None, start_date=None,
                        end_date=None, adjust=None):
        self.calls.append({"symbol": symbol, "period": period,
                           "start_date": start_date, "end_date": end_date,
                           "adjust": adjust})
        if symbol in self.fail_codes:
            n = self._fail_count.get(symbol, 0) + 1
            self._fail_count[symbol] = n
            if n <= self.fail_times:
                raise RuntimeError(f"akshare 限流 ({n})")
        return self.per_code.get(symbol)

    def stock_individual_info_em(self, symbol=None):
        self.info_calls.append(symbol)
        if symbol in self.fail_codes:
            raise RuntimeError("info 取不到")
        return self._info.get(symbol)


def _install(monkeypatch, fake):
    monkeypatch.setitem(sys.modules, "akshare", fake)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    return fake


# ===========================================================================
# A. 代码规整
# ===========================================================================

class TestSymbolStripping:

    @pytest.mark.parametrize("raw,bare", [
        ("600519.SH", "600519"),
        ("300750.SZ", "300750"),
        ("000001.SZ", "000001"),
        ("600000", "600000"),
    ])
    def test_the_exchange_suffix_is_removed(self, raw, bare):
        """
        `symbol.split(".")[0]` —— 下标改成 `[1]` 会把 "600519.SH"
        变成 "SH"，akshare 返回空；`[-1]` 同理。
        """
        assert _strip_suffix(raw) == bare

    def test_a_leading_zero_code_keeps_its_zeros(self):
        """A 股代码是**字符串**。任何走 int 的改写都会丢掉前导零。"""
        out = _strip_suffix("000001.SZ")
        assert out == "000001" and isinstance(out, str)

    def test_the_bare_code_is_what_reaches_akshare(self, monkeypatch):
        fake = _install(monkeypatch, _FakeAk({"600519": _ak_frame()}))
        AkshareProvider(delay_s=0.0).fetch(["600519.SH"], "2023-01-01",
                                           "2023-01-31")
        assert fake.calls[0]["symbol"] == "600519", (
            f"传给 akshare 的代码是 {fake.calls[0]['symbol']!r} —— 后缀没被剥掉")


# ===========================================================================
# B. 请求参数
# ===========================================================================

class TestRequestParameters:

    def test_dates_are_converted_to_the_compact_akshare_format(self, monkeypatch):
        """
        `start.replace("-", "")` —— akshare 只认 "20230101"。
        替换目标被改（比如换成 "/"）会让它收到 "2023-01-01"，
        返回空表，最后抛出一句"no data"，掩盖真正的原因。
        """
        fake = _install(monkeypatch, _FakeAk({"600519": _ak_frame()}))
        AkshareProvider(delay_s=0.0).fetch(["600519.SH"], "2023-01-03",
                                           "2023-02-15")
        c = fake.calls[0]
        assert c["start_date"] == "20230103", f"起始日期格式不对：{c['start_date']!r}"
        assert c["end_date"] == "20230215", f"结束日期格式不对：{c['end_date']!r}"

    def test_daily_forward_adjusted_prices_are_requested_by_default(self,
                                                                    monkeypatch):
        """
        `adjust = self.adjust`，默认 "qfq"。

        这是本模块**最贵**的一个参数：改成 ""（不复权）之后，
        除权除息日会在价格序列里留下一根巨大的跳空，
        任何动量/反转因子都会把它当成真实行情。面板形状一切正常。
        """
        fake = _install(monkeypatch, _FakeAk({"600519": _ak_frame()}))
        AkshareProvider(delay_s=0.0).fetch(["600519.SH"], "2023-01-01",
                                           "2023-01-31")
        assert fake.calls[0]["adjust"] == "qfq", (
            f"复权方式是 {fake.calls[0]['adjust']!r} —— 不是前复权，"
            f"除权日会被当成真实涨跌")
        assert fake.calls[0]["period"] == "daily"

    def test_the_adjust_mode_is_configurable(self, monkeypatch):
        fake = _install(monkeypatch, _FakeAk({"600519": _ak_frame()}))
        AkshareProvider(adjust="hfq", delay_s=0.0).fetch(
            ["600519.SH"], "2023-01-01", "2023-01-31")
        assert fake.calls[0]["adjust"] == "hfq"

    def test_the_documented_defaults(self):
        p = AkshareProvider()
        assert p.adjust == "qfq"
        assert p.delay_s == 0.3

    def test_every_requested_symbol_is_fetched(self, monkeypatch):
        fake = _install(monkeypatch, _FakeAk({"600519": _ak_frame(),
                                              "300750": _ak_frame()}))
        AkshareProvider(delay_s=0.0).fetch(["600519.SH", "300750.SZ"],
                                           "2023-01-01", "2023-01-31")
        assert [c["symbol"] for c in fake.calls] == ["600519", "300750"]

    def test_a_missing_akshare_raises_an_actionable_import_error(self, monkeypatch):
        """
        `raise ImportError("...Install with: pip install akshare") from exc`
        —— 原样冒出的 ImportError 只说 "No module named 'akshare'"，
        看不出是哪条数据链路需要它。
        """
        monkeypatch.setitem(sys.modules, "akshare", None)
        with pytest.raises(ImportError, match="pip install akshare"):
            AkshareProvider().fetch(["600519.SH"], "2023-01-01", "2023-01-31")


# ===========================================================================
# C. 重试
# ===========================================================================

class TestRetry:

    def test_a_transient_failure_is_retried_up_to_three_times(self, monkeypatch):
        """
        `for attempt in range(3)` —— 次数被改成 1 会让一次限流
        就丢掉这只票（面板里变成全 NaN 列，静默）。
        构造：前两次抛，第三次成功。
        """
        fake = _FakeAk({"600519": _ak_frame()}, fail_codes=["600519"],
                       fail_times=2)
        _install(monkeypatch, fake)
        panel = AkshareProvider(delay_s=0.0).fetch(["600519.SH"], "2023-01-01",
                                                   "2023-01-31")
        assert len(fake.calls) == 3, f"只重试到第 {len(fake.calls)} 次"
        assert panel["close"]["600519.SH"].notna().any(), (
            "第三次成功了，数据却没进面板")

    def test_persistent_failure_gives_up_after_three_attempts(self, monkeypatch):
        fake = _FakeAk({}, fail_codes=["600519"], fail_times=99)
        _install(monkeypatch, fake)
        with pytest.raises(ValueError, match="no data returned"):
            AkshareProvider(delay_s=0.0).fetch(["600519.SH"], "2023-01-01",
                                               "2023-01-31")
        assert len(fake.calls) == 3, (
            f"持续失败时请求了 {len(fake.calls)} 次，应当恰好 3 次")

    def test_a_successful_fetch_does_not_retry(self, monkeypatch):
        """`break` 被删会让成功之后仍然重复请求两次，三倍流量。"""
        fake = _FakeAk({"600519": _ak_frame()})
        _install(monkeypatch, fake)
        AkshareProvider(delay_s=0.0).fetch(["600519.SH"], "2023-01-01",
                                           "2023-01-31")
        assert len(fake.calls) == 1, f"一次成功却请求了 {len(fake.calls)} 次"

    def test_an_empty_result_also_stops_retrying(self, monkeypatch):
        """
        `if df is not None and not df.empty: ...` 之后**无条件** `break`
        —— 空表不是异常，不该重试。`break` 挪进 if 里会让空表重试三次。
        """
        fake = _FakeAk({"600519": pd.DataFrame(), "300750": _ak_frame()})
        _install(monkeypatch, fake)
        AkshareProvider(delay_s=0.0).fetch(["600519.SH", "300750.SZ"],
                                           "2023-01-01", "2023-01-31")
        n600 = len([c for c in fake.calls if c["symbol"] == "600519"])
        assert n600 == 1, f"空表被重试了 {n600} 次"

    def test_a_none_result_is_treated_as_no_data_not_a_crash(self, monkeypatch):
        """`df is not None` —— 守卫被删会让 `df.empty` 在 None 上抛。"""
        fake = _FakeAk({"600519": None, "300750": _ak_frame()})
        _install(monkeypatch, fake)
        panel = AkshareProvider(delay_s=0.0).fetch(["600519.SH", "300750.SZ"],
                                                   "2023-01-01", "2023-01-31")
        assert panel["close"]["600519.SH"].isna().all()
        assert panel["close"]["300750.SZ"].notna().any()

    def test_the_backoff_sleep_only_happens_between_attempts(self, monkeypatch):
        """
        `if attempt < 2: time.sleep(self.delay_s * 2)`

        `<` 翻成 `<=` 会让第三次（最后一次）失败之后也白等一轮；
        `* 2` 翻成 `/ 2` 则让退避变成"更快重试"，限流时雪上加霜。
        """
        slept = []
        fake = _FakeAk({}, fail_codes=["600519"], fail_times=99)
        monkeypatch.setitem(sys.modules, "akshare", fake)
        monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

        with pytest.raises(ValueError):
            AkshareProvider(delay_s=0.5).fetch(["600519.SH"], "2023-01-01",
                                               "2023-01-31")
        backoffs = [s for s in slept if s == pytest.approx(1.0)]
        assert len(backoffs) == 2, (
            f"退避睡眠发生了 {len(backoffs)} 次（每次 {0.5*2}s），应当恰好 2 次 —— "
            f"`attempt < 2` 的边界被改了")
        assert all(s >= 0.5 for s in slept)

    def test_one_failing_symbol_does_not_take_down_the_others(self, monkeypatch):
        fake = _FakeAk({"300750": _ak_frame()}, fail_codes=["600519"])
        _install(monkeypatch, fake)
        panel = AkshareProvider(delay_s=0.0).fetch(["600519.SH", "300750.SZ"],
                                                   "2023-01-01", "2023-01-31")
        assert list(panel["close"].columns) == ["600519.SH", "300750.SZ"]
        assert panel["close"]["300750.SZ"].notna().any()

    def test_no_data_at_all_raises_with_the_symbol_list(self, monkeypatch):
        """
        `if not frames: raise ValueError(...)` —— `not` 被删会在
        **正常取到数据**时反而抛，或者在全空时返回一个空面板
        （下游 `next(iter(data.values())).columns` 直接炸在别处）。
        """
        _install(monkeypatch, _FakeAk({}))
        with pytest.raises(ValueError) as ei:
            AkshareProvider(delay_s=0.0).fetch(["600519.SH"], "2023-01-01",
                                               "2023-01-31")
        assert "600519.SH" in str(ei.value), "报错里没有列出请求的代码"


# ===========================================================================
# D. 列名映射与面板
# ===========================================================================

class TestColumnMapping:

    def test_every_standard_field_has_a_chinese_source_column(self):
        """
        `_COLUMN_MAP` 里少一项 → 该字段在 `_field` 里走 else 分支，
        变成**全 NaN 列**，面板照样成形。所以必须正面钉住映射表。
        """
        assert _COLUMN_MAP == {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "涨跌幅": "pct_chg",
        }, f"akshare 列名映射被改动：{_COLUMN_MAP}"

    def test_ohlcv_lands_in_the_right_field_not_merely_non_nan(self, monkeypatch):
        """
        映射表里两个键对调（开盘↔收盘）会让面板**每一列都非空**，
        只是内容互换。所以要逐值比对原始数据。
        """
        raw = _ak_frame(days=4)
        _install(monkeypatch, _FakeAk({"600519": raw}))
        panel = AkshareProvider(delay_s=0.0).fetch(["600519.SH"], "2023-01-01",
                                                   "2023-01-31")
        col = "600519.SH"
        assert panel["open"][col].tolist() == pytest.approx(raw["开盘"].tolist())
        assert panel["close"][col].tolist() == pytest.approx(raw["收盘"].tolist())
        assert panel["high"][col].tolist() == pytest.approx(raw["最高"].tolist())
        assert panel["low"][col].tolist() == pytest.approx(raw["最低"].tolist())
        assert panel["volume"][col].tolist() == pytest.approx(raw["成交量"].tolist())

    def test_the_date_column_becomes_a_sorted_datetime_index(self, monkeypatch):
        """
        `pd.to_datetime(df["date"])` + `set_index` + `sort_index`
        —— 少了 sort 会让乱序返回原样进面板；少了 to_datetime
        会让索引是字符串，与其他数据源 union 时变成 object 索引。
        """
        raw = _ak_frame(days=5)
        raw = raw.iloc[[3, 0, 4, 1, 2]].reset_index(drop=True)   # 打乱
        _install(monkeypatch, _FakeAk({"600519": raw}))
        panel = AkshareProvider(delay_s=0.0).fetch(["600519.SH"], "2023-01-01",
                                                   "2023-01-31")
        idx = panel["close"].index
        assert isinstance(idx, pd.DatetimeIndex), f"索引类型是 {type(idx).__name__}"
        assert list(idx) == sorted(idx), "索引没有排序"

    def test_all_documented_fields_are_produced(self, monkeypatch):
        _install(monkeypatch, _FakeAk({"600519": _ak_frame()}))
        panel = AkshareProvider(delay_s=0.0).fetch(["600519.SH"], "2023-01-01",
                                                   "2023-01-31")
        assert set(panel) == set(FIELDS)

    def test_columns_follow_the_requested_symbol_order(self, monkeypatch):
        _install(monkeypatch, _FakeAk({"600519": _ak_frame(),
                                       "300750": _ak_frame()}))
        panel = AkshareProvider(delay_s=0.0).fetch(["300750.SZ", "600519.SH"],
                                                   "2023-01-01", "2023-01-31")
        assert list(panel["close"].columns) == ["300750.SZ", "600519.SH"]

    def test_the_union_index_covers_symbols_with_different_calendars(self,
                                                                     monkeypatch):
        """
        `idx.union(df.index)` —— 换成 `intersection` 会让一只票停牌的
        日子把**所有**票的那一天都删掉。
        """
        a = _ak_frame(days=4, start="2023-01-03")
        b = _ak_frame(days=4, start="2023-01-09")
        _install(monkeypatch, _FakeAk({"600519": a, "300750": b}))
        panel = AkshareProvider(delay_s=0.0).fetch(["600519.SH", "300750.SZ"],
                                                   "2023-01-01", "2023-01-31")
        assert len(panel["close"].index) == 8, (
            f"并集索引长度是 {len(panel['close'].index)} —— union 被改成了交集")

    def test_vwap_is_the_mean_of_high_low_and_close(self, monkeypatch):
        _install(monkeypatch, _FakeAk({"600519": _ak_frame()}))
        p = AkshareProvider(delay_s=0.0).fetch(["600519.SH"], "2023-01-01",
                                               "2023-01-31")
        ref = (p["high"] + p["low"] + p["close"]) / 3.0
        assert np.allclose(p["vwap"].values, ref.values, rtol=1e-12)
        assert (p["vwap"] >= p["low"]).all().all()
        assert (p["vwap"] <= p["high"]).all().all()

    def test_returns_are_log_returns_of_close(self, monkeypatch):
        _install(monkeypatch, _FakeAk({"600519": _ak_frame()}))
        p = AkshareProvider(delay_s=0.0).fetch(["600519.SH"], "2023-01-01",
                                               "2023-01-31")
        ref = np.log(p["close"] / p["close"].shift(1))
        assert np.allclose(p["returns"].values, ref.values, rtol=1e-12,
                           equal_nan=True)
        assert p["returns"].abs().max().max() < 1.0, (
            "收益率量级异常 —— `close / close.shift(1)` 的除法被改了")

    def test_a_symbol_without_data_becomes_an_all_nan_column(self, monkeypatch):
        _install(monkeypatch, _FakeAk({"600519": _ak_frame()}))
        panel = AkshareProvider(delay_s=0.0).fetch(["600519.SH", "000001.SZ"],
                                                   "2023-01-01", "2023-01-31")
        assert panel["close"]["000001.SZ"].isna().all()
        assert list(panel["close"].columns) == ["600519.SH", "000001.SZ"]

    def test_all_fields_are_float_typed(self, monkeypatch):
        _install(monkeypatch, _FakeAk({"600519": _ak_frame()}))
        panel = AkshareProvider(delay_s=0.0).fetch(["600519.SH"], "2023-01-01",
                                                   "2023-01-31")
        for name, df in panel.items():
            assert (df.dtypes == np.float64).all(), f"{name} 不是 float64"


# ===========================================================================
# E. 市值（被 dataset_filters 的 China 分支调用）
# ===========================================================================

def _info(total_cap: str) -> pd.DataFrame:
    return pd.DataFrame({"item": ["股票代码", "总市值", "流通市值"],
                         "value": ["600519", total_cap, "1000亿"]})


class TestFetchMarketCap:

    def test_the_yi_unit_is_converted_to_yuan(self, monkeypatch):
        """
        `float(val) * 1e8` —— "亿" → 元。

        `*` 翻成 `/` 会让茅台的市值变成 2.1e-6 元，
        于是 `dataset_filters` 里**每一只 A 股都是 small_cap**，
        "只做大盘股"的配置静默选中一堆微盘。
        """
        fake = _FakeAk(info={"600519": _info("21000亿")})
        _install(monkeypatch, fake)
        out = AkshareProvider.fetch_market_cap(["600519.SH"])
        assert out["600519.SH"] == pytest.approx(21000 * 1e8), (
            f"市值是 {out['600519.SH']:.3e}，应当是 2.1e12 —— 亿→元 的换算被改了")

    def test_the_yi_character_is_stripped_before_parsing(self, monkeypatch):
        """`str(...).replace("亿", "")` —— 不剥掉会让 `float()` 直接抛。"""
        fake = _FakeAk(info={"600519": _info("500亿")})
        _install(monkeypatch, fake)
        assert AkshareProvider.fetch_market_cap(["600519.SH"])["600519.SH"] == \
               pytest.approx(5e10)

    def test_the_total_market_cap_row_is_selected_not_the_float_one(self,
                                                                    monkeypatch):
        """
        `info[info["item"] == "总市值"]` —— 取成"流通市值"会系统性
        低估（A 股大量限售股），大盘股筛选的门槛因此整体下移。
        构造：两行数值差 21 倍，取错必被发现。
        """
        fake = _FakeAk(info={"600519": _info("21000亿")})
        _install(monkeypatch, fake)
        out = AkshareProvider.fetch_market_cap(["600519.SH"])
        assert out["600519.SH"] == pytest.approx(2.1e12), (
            "取到的是流通市值（1000亿）而不是总市值")

    def test_a_missing_row_leaves_the_symbol_out(self, monkeypatch):
        """
        `if not row.empty:` —— 守卫被删会让 `.iloc[0]` 在空表上抛。
        当前行为是"该代码不进结果字典"，调用方
        （`dataset_filters._fetch_market_cap` → `reindex`）会补 NaN。
        """
        no_cap = pd.DataFrame({"item": ["股票代码"], "value": ["600519"]})
        fake = _FakeAk(info={"600519": no_cap})
        _install(monkeypatch, fake)
        assert AkshareProvider.fetch_market_cap(["600519.SH"]) == {}

    def test_a_failing_symbol_degrades_to_nan_without_killing_the_batch(self,
                                                                        monkeypatch):
        fake = _FakeAk(info={"300750": _info("900亿")}, fail_codes=["600519"])
        _install(monkeypatch, fake)
        out = AkshareProvider.fetch_market_cap(["600519.SH", "300750.SZ"])
        assert np.isnan(out["600519.SH"]), "失败的代码没有降级为 NaN"
        assert out["300750.SZ"] == pytest.approx(9e10), "一个失败带走了整批"

    def test_the_bare_code_is_what_reaches_akshare(self, monkeypatch):
        fake = _FakeAk(info={"600519": _info("100亿")})
        _install(monkeypatch, fake)
        AkshareProvider.fetch_market_cap(["600519.SH"])
        assert fake.info_calls == ["600519"], (
            f"传给 akshare 的代码是 {fake.info_calls} —— 后缀没剥掉")

    def test_the_result_is_keyed_by_the_original_symbol_not_the_bare_code(self,
                                                                          monkeypatch):
        """
        `result[sym]` 而不是 `result[code]` —— 用裸代码做键会让
        调用方按 "600519.SH" 去取时全部落空，静默变成 NaN。
        """
        fake = _FakeAk(info={"600519": _info("100亿")})
        _install(monkeypatch, fake)
        out = AkshareProvider.fetch_market_cap(["600519.SH"])
        assert list(out) == ["600519.SH"], f"结果的键是 {list(out)}，不是原始代码"

    def test_a_missing_akshare_returns_an_empty_dict_rather_than_raising(self,
                                                                         monkeypatch):
        """
        这里与 `fetch` 不同：市值是**可选**增强，没装 akshare
        不该让整个筛选链路炸掉。`return {}` 被改成 raise 会让
        `dataset_filters` 的 China 分支整条失效。
        """
        monkeypatch.setitem(sys.modules, "akshare", None)
        assert AkshareProvider.fetch_market_cap(["600519.SH"]) == {}
