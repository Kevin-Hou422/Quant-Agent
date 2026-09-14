"""
data_engine/yahoo_provider.py —— Yahoo Finance 行情源

**首测击杀率 0.0%（3/3 全部存活）** —— 既有测试只 import 过它。

**它在主线上的真实地位**（查过源码，不是推测）：

`settings.price_source` 的**代码默认值**是 `"yahoo"`，但部署 `.env` 设的是
`PRICE_SOURCE=moomoo`（见 `tests/conftest.py` 里 `_hermetic_run_flags` 的注释）。
即便如此，本模块仍有三条活路径：

  1. `dataset_registry._fetch_raw` 的 moomoo 分支**只覆盖 `spec.region == "US"`**；
     非美但 `provider == "yfinance"` 的数据集仍走这里；
  2. 读 `price_source` 抛异常时的 `except` 分支显式 `use_moomoo = False` →
     **退回 yahoo**（源码注释自己写明这会破坏 TR.2 的研究/执行同源）；
  3. 任何人把 `price_source` 改回默认值时，整条美股取数立刻回到这里。

（`multi_dataset.py` 里的 `_load_via_yfinance` 也引用本模块，但那四个
`load_us_equity/china_a/crypto/etf` 在 `app/` 里**没有任何调用点** ——
只有定义与 `__init__.py` 的再导出。那条路是 import 可达而未被调用。）

所以它不是死代码，是**降级路径 + 非美路径**。三个存活点各有代价：

  - `auto_adjust=True` 翻成 False → 拿到**未复权**价格，
    每个除权除息日在收益率序列里留下一根凭空的跳空。
    动量、反转、波动因子全部被污染，而面板形状、字段、日期一切正常。
  - `progress=False` 翻成 True → 在无人值守的日循环日志里刷进度条控制字符。
  - `vwap = (h + l + c) / 3.0` 的 `+` 翻成 `-` → vwap 跑到 [low, high] 之外，
    任何以 vwap 计价的成交假设都错，而它仍然是"一条价格序列"。

整条链路不碰网络：往 `sys.modules` 注入一个可编程的假 yfinance。
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.yahoo_provider import (
    _SUPPORTED_FIELDS,
    YahooFinanceProvider,
)

IDX = pd.bdate_range("2022-01-03", periods=12)
TICKERS = ["AAPL", "MSFT"]


def _raw(tickers=TICKERS, idx=IDX, multi=True) -> pd.DataFrame:
    """
    构造 yfinance `group_by="column"` 的返回形状：MultiIndex (Field, Ticker)。

    **刻意不对称**：High = Close × 1.05、Low = Close × 0.98。
    对称的 ±2% 会让 `(H+L+C)/3` 恰好等于 Close，
    于是 vwap 公式被改成"直接取 close"完全观察不到
    （`multi_dataset` 首测就是被这个构造漏掉的）。
    """
    n = len(idx)
    base = {t: 100.0 + 10 * i + np.arange(n, dtype=float) for i, t in enumerate(tickers)}
    fields = {
        "Close": {t: v for t, v in base.items()},
        "High": {t: v * 1.05 for t, v in base.items()},
        "Low": {t: v * 0.98 for t, v in base.items()},
        "Open": {t: v * 0.995 for t, v in base.items()},
        "Volume": {t: np.full(n, 1e6 + 1000 * i) for i, t in enumerate(tickers)},
    }
    if not multi:
        return pd.DataFrame({f: v[tickers[0]] for f, v in fields.items()}, index=idx)
    cols, data = [], []
    for f, per_t in fields.items():
        for t in tickers:
            cols.append((f, t))
            data.append(per_t[t])
    return pd.DataFrame(np.column_stack(data), index=idx,
                        columns=pd.MultiIndex.from_tuples(cols))


def _install(monkeypatch, frame=None, spy=None):
    def download(**kwargs):
        if spy is not None:
            spy.update(kwargs)
        return _raw() if frame is None else frame

    monkeypatch.setitem(sys.modules, "yfinance",
                        types.SimpleNamespace(download=download))


# ===========================================================================
# A. 下载参数 —— 首测存活的两个布尔
# ===========================================================================

class TestDownloadParameters:

    def test_prices_are_split_and_dividend_adjusted_by_default(self, monkeypatch):
        """
        **首测存活项 L42**：`auto_adjust: bool = True`

        这是本模块**最贵**的一个默认值。翻成 False 之后拿到的是未复权价：
        苹果 4:1 拆股那天会出现一根 -75% 的"暴跌"，
        任何动量/反转因子都会把它当成真实行情。
        面板形状、字段、日期全部正常，健康门也未必拦得住
        （单日跳变阈值默认 50%，-75% 会被报，但 2:1 拆股的 -50% 恰好不报）。
        """
        seen = {}
        _install(monkeypatch, spy=seen)
        YahooFinanceProvider().fetch(TICKERS, "2022-01-01", "2022-02-01")
        assert seen["auto_adjust"] is True, (
            f"auto_adjust={seen['auto_adjust']} —— 拿到的是未复权价，"
            f"除权日会被当成真实涨跌")

    def test_the_progress_bar_is_off_by_default(self, monkeypatch):
        """
        **首测存活项 L43**：`progress: bool = False`

        翻成 True 会在无人值守的日循环日志里刷一堆回车控制字符，
        把真正的告警淹掉。
        """
        seen = {}
        _install(monkeypatch, spy=seen)
        YahooFinanceProvider().fetch(TICKERS, "2022-01-01", "2022-02-01")
        assert seen["progress"] is False, f"progress={seen['progress']}"

    def test_both_flags_are_configurable(self, monkeypatch):
        seen = {}
        _install(monkeypatch, spy=seen)
        YahooFinanceProvider(auto_adjust=False, progress=True).fetch(
            TICKERS, "2022-01-01", "2022-02-01")
        assert seen["auto_adjust"] is False and seen["progress"] is True

    def test_the_constructor_stores_the_flags(self):
        p = YahooFinanceProvider()
        assert p.auto_adjust is True and p.progress is False

    def test_the_window_and_grouping_are_forwarded(self, monkeypatch):
        """
        `group_by="column"` —— 改成 "ticker" 会让返回的 MultiIndex
        层级顺序反过来，`_parse` 抽出来的"字段"其实是 ticker。
        """
        seen = {}
        _install(monkeypatch, spy=seen)
        YahooFinanceProvider().fetch(TICKERS, "2020-02-02", "2021-03-03")
        assert seen["start"] == "2020-02-02" and seen["end"] == "2021-03-03"
        assert seen["group_by"] == "column"

    def test_tickers_are_upper_cased_before_the_request(self, monkeypatch):
        seen = {}
        _install(monkeypatch, spy=seen)
        YahooFinanceProvider().fetch(["aapl", "MsFt"], "2022-01-01", "2022-02-01")
        assert seen["tickers"] == ["AAPL", "MSFT"]


# ===========================================================================
# B. vwap 派生 —— 首测存活的算术
# ===========================================================================

class TestDerivedFields:

    def test_vwap_is_the_mean_of_high_low_and_close(self, monkeypatch):
        """
        **首测存活项 L162**：`raw_frames["vwap"] = (h + l + c) / 3.0`

        `+` 翻成 `-`、`/ 3.0` 翻成 `* 3.0` —— 结果仍然是"一条价格序列"，
        形状 dtype 全对，下游不会有任何抱怨，但以 vwap 计价的
        成交假设（PaperBroker 的 VWAP 撮合）全部错位。

        逐位比对参考实现，再加一条"落在 [low, high] 之间"的不变量。
        夹具刻意用 +5%/-2% 的**不对称**区间，避免 `(H+L+C)/3` 恰好等于 Close。
        """
        _install(monkeypatch)
        ds = YahooFinanceProvider().fetch(TICKERS, "2022-01-01", "2022-02-01")

        ref = (ds["high"] + ds["low"] + ds["close"]) / 3.0
        assert np.allclose(ds["vwap"].values, ref.values, rtol=1e-12), (
            "vwap 不等于 (high+low+close)/3 —— 算符被改了")
        assert not np.allclose(ds["vwap"].values, ds["close"].values), (
            "夹具太对称，vwap 恰好等于 close —— 本用例失去区分力")
        assert (ds["vwap"] >= ds["low"]).all().all(), "vwap 跌出了当日最低价"
        assert (ds["vwap"] <= ds["high"]).all().all(), "vwap 超出了当日最高价"

    def test_returns_are_log_returns_of_close(self, monkeypatch):
        """`np.log(c / c.shift(1))` —— `/` 翻成 `*` 会让收益率变成 log(P²)。"""
        _install(monkeypatch)
        ds = YahooFinanceProvider().fetch(TICKERS, "2022-01-01", "2022-02-01")
        ref = np.log(ds["close"] / ds["close"].shift(1))
        assert np.allclose(ds["returns"].values, ref.values, rtol=1e-12,
                           equal_nan=True)
        assert ds["returns"].abs().max().max() < 1.0, (
            "收益率量级异常 —— 除法方向被改了")

    def test_all_supported_fields_are_produced_by_default(self, monkeypatch):
        _install(monkeypatch)
        ds = YahooFinanceProvider().fetch(TICKERS, "2022-01-01", "2022-02-01")
        assert set(ds) == set(_SUPPORTED_FIELDS), (
            f"默认字段集是 {sorted(ds)}，应当是 {sorted(_SUPPORTED_FIELDS)}")

    def test_only_the_requested_fields_come_back(self, monkeypatch):
        """
        `for f in fields: if f in raw_frames: dataset[f] = ...`
        —— 守卫或循环被改会让调用方拿到它没要的字段（浪费内存），
        或拿不到它要的（KeyError 在别处炸）。
        """
        _install(monkeypatch)
        ds = YahooFinanceProvider().fetch(TICKERS, "2022-01-01", "2022-02-01",
                                          fields=["close", "vwap"])
        assert set(ds) == {"close", "vwap"}

    def test_asking_for_vwap_alone_still_pulls_the_inputs_it_needs(self, monkeypatch):
        """
        `if field in fields or "vwap" in fields or "returns" in fields:`
        —— 这三段 or 少一段，就会在只要 vwap 时抽不到 high/low/close，
        vwap 变成全 NaN（而不是报错）。
        """
        _install(monkeypatch)
        ds = YahooFinanceProvider().fetch(TICKERS, "2022-01-01", "2022-02-01",
                                          fields=["vwap"])
        assert set(ds) == {"vwap"}
        assert ds["vwap"].notna().all().all(), (
            "只要 vwap 时它变成了全 NaN —— 派生所需的 OHLC 没被抽取")

    def test_asking_for_returns_alone_still_works(self, monkeypatch):
        _install(monkeypatch)
        ds = YahooFinanceProvider().fetch(TICKERS, "2022-01-01", "2022-02-01",
                                          fields=["returns"])
        assert set(ds) == {"returns"}
        assert ds["returns"].iloc[1:].notna().all().all()


# ===========================================================================
# C. 解析
# ===========================================================================

class TestParsing:

    def test_column_levels_are_normalised_to_lower_and_upper(self, monkeypatch):
        """
        `(lvl0.lower(), lvl1.upper())` —— yfinance 给的是 "Close"/"aapl"。
        大小写不规整会让 `raw["close"]` 直接 KeyError（然后被兜成全 NaN）。
        """
        _install(monkeypatch)
        ds = YahooFinanceProvider().fetch(["aapl", "msft"], "2022-01-01",
                                          "2022-02-01")
        assert list(ds["close"].columns) == ["AAPL", "MSFT"]
        assert ds["close"].notna().all().all()

    def test_a_single_ticker_flat_frame_is_normalised(self, monkeypatch):
        """
        单票时 yfinance 返回**扁平列**。`else` 分支给它补一层假的
        MultiIndex —— 分支被删会让 `raw["close"]` 拿到一个 Series，
        后面 `.reindex(columns=...)` 直接 AttributeError。
        """
        _install(monkeypatch, frame=_raw(tickers=["AAPL"], multi=False))
        ds = YahooFinanceProvider().fetch(["AAPL"], "2022-01-01", "2022-02-01")
        assert list(ds["close"].columns) == ["AAPL"]
        assert ds["close"].notna().all().all()

    def test_columns_follow_the_requested_ticker_order(self, monkeypatch):
        """
        `df.reindex(columns=tickers)` —— yfinance 返回的列序不保证与
        请求一致。不 reindex 会让调用方按位置取权重时错配标的。
        """
        _install(monkeypatch)
        ds = YahooFinanceProvider().fetch(["MSFT", "AAPL"], "2022-01-01",
                                          "2022-02-01")
        assert list(ds["close"].columns) == ["MSFT", "AAPL"]

    def test_a_ticker_missing_from_the_response_becomes_a_nan_column(self,
                                                                     monkeypatch):
        """请求了但上游没给的票必须留一列 NaN，而不是从面板里消失。"""
        _install(monkeypatch, frame=_raw(tickers=["AAPL"]))
        ds = YahooFinanceProvider().fetch(["AAPL", "NOSUCH"], "2022-01-01",
                                          "2022-02-01")
        assert list(ds["close"].columns) == ["AAPL", "NOSUCH"]
        assert ds["close"]["NOSUCH"].isna().all()

    def test_a_missing_field_degrades_to_an_all_nan_panel(self, monkeypatch, caplog):
        """
        `except KeyError:` 之后返回全 NaN 面板 —— 源码注释写明
        "该字段**整段缺席**却看起来'有数据'"，所以必须同时打 ERROR 日志。
        日志是这里唯一的可见信号。
        """
        import logging

        frame = _raw()
        frame = frame.drop(columns=[c for c in frame.columns if c[0] == "Volume"])
        _install(monkeypatch, frame=frame)
        with caplog.at_level(logging.ERROR,
                             logger="app.core.data_engine.yahoo_provider"):
            ds = YahooFinanceProvider().fetch(TICKERS, "2022-01-01", "2022-02-01")
        assert ds["volume"].isna().all().all()
        assert ds["close"].notna().all().all(), "缺一个字段把别的字段也带坏了"
        assert any("volume" in r.getMessage() for r in caplog.records), (
            f"字段整段缺席却没有打 ERROR 日志：{[r.getMessage() for r in caplog.records]}")

    def test_the_index_is_a_datetime_index(self, monkeypatch):
        _install(monkeypatch)
        ds = YahooFinanceProvider().fetch(TICKERS, "2022-01-01", "2022-02-01")
        assert isinstance(ds["close"].index, pd.DatetimeIndex)

    def test_all_panels_are_float_typed(self, monkeypatch):
        _install(monkeypatch)
        ds = YahooFinanceProvider().fetch(TICKERS, "2022-01-01", "2022-02-01")
        for name, df in ds.items():
            assert (df.dtypes == np.float64).all(), f"{name} 不是 float64"


# ===========================================================================
# D. 守卫与元数据
# ===========================================================================

class TestGuardsAndMetadata:

    def test_an_empty_download_raises_with_an_actionable_message(self, monkeypatch):
        """
        `if raw.empty: raise ValueError(...)`

        `not` 被插入（或守卫被删）会让空返回一路流进 `_parse`，
        最后产出一个 0 行的面板 —— 下游把它当成"今天休市"。
        必须在这里就炸，并说清该检查什么。
        """
        _install(monkeypatch, frame=pd.DataFrame())
        with pytest.raises(ValueError) as ei:
            YahooFinanceProvider().fetch(TICKERS, "2022-01-01", "2022-02-01")
        msg = str(ei.value)
        assert "empty" in msg.lower()
        assert "tickers" in msg.lower() and "date range" in msg.lower(), (
            f"报错没有说清该检查什么：{msg}")

    def test_an_unsupported_field_is_rejected_before_the_network_call(self,
                                                                      monkeypatch):
        """
        `self.validate_fields(fields)` —— 校验必须在下载**之前**。
        放到之后会让一次拼错的字段名照样打一轮网络。
        """
        called = {}
        monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(
            download=lambda **k: called.setdefault("hit", True) or _raw()))
        with pytest.raises(ValueError):
            YahooFinanceProvider().fetch(TICKERS, "2022-01-01", "2022-02-01",
                                         fields=["close", "bid"])
        assert "hit" not in called, "字段非法却已经打了网络"

    def test_the_supported_field_list_is_the_documented_one(self):
        assert set(_SUPPORTED_FIELDS) == {"open", "high", "low", "close",
                                          "volume", "vwap", "returns"}

    def test_available_fields_returns_a_copy(self):
        got = YahooFinanceProvider().available_fields()
        got.append("污染")
        assert "污染" not in _SUPPORTED_FIELDS, "返回的是模块级列表本身"
