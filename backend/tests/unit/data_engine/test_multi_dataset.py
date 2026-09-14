"""
data_engine/multi_dataset.py —— 多市场数据集抽象与注册表

**此前零专属测试**（23 个变异点，D 档）。

四个市场（美股 / A 股 / 加密 / ETF）共用一套"对齐 + 补齐 7 字段"的
流水线。它的危险之处在于**它专门负责把残缺数据变得看起来完整**：

  - `ffill(limit=5)` 的上限被删 → 退市股以最后价格一路"活"到区间末尾，
    回测里是一条零波动的完美持仓；
  - 缺失字段补 NaN 的那一段 → 若被改成补 0，`volume=0` 会让所有
    ADV/冲击成本计算归零，"任意大的单子都没有成本"；
  - `np.log(c / c.shift(1))` 的除法方向 → 收益率变成 log(P²)；
  - 注册表缓存键 `f"{name}|{start}|{end}"` 少一段 → **换了日期区间
    却拿到上一次的缓存**，回测跑在错误的时间窗上且毫无提示。

所有加载器都不碰网络：`_load_via_yfinance` / `_load_via_ccxt` /
`_load_from_local` 三条路径分别用桩替换，只验我们自己的逻辑。
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

import app.core.data_engine.multi_dataset as MD
from app.core.data_engine.multi_dataset import (
    CHINA_A_UNIVERSE,
    CRYPTO_UNIVERSE,
    ETF_UNIVERSE,
    STANDARD_FIELDS,
    US_EQUITY_UNIVERSE,
    Dataset,
    DatasetRegistry,
    _align_and_standardize,
    _load_via_ccxt,
    get_registry,
    load_china_a,
    load_crypto,
    load_dataset,
    load_etf,
    load_us_equity,
)

IDX = pd.bdate_range("2022-01-03", periods=20)
UNI = ["AAA", "BBB", "CCC"]


def _frame(idx=IDX, cols=UNI, base=100.0) -> pd.DataFrame:
    n, m = len(idx), len(cols)
    return pd.DataFrame(base + np.arange(n * m, dtype=float).reshape(n, m),
                        index=idx, columns=cols)


def _raw(fields=("open", "high", "low", "close", "volume")) -> dict:
    out = {}
    for i, f in enumerate(fields):
        out[f] = _frame(base=100.0 + 10 * i)
    if "high" in out and "low" in out and "close" in out:
        # 刻意**不对称**：+5% / -2%。
        # 用对称的 ±2% 会让 (h+l+c)/3 恰好等于 close，
        # 于是"vwap 派生公式被改成直接取 close"这一类变异
        # 全部观察不到（首测就是这么漏掉 L167 的）。
        out["high"] = out["close"] * 1.05
        out["low"] = out["close"] * 0.98
    return out


def _full_panel() -> dict:
    d = _raw()
    return _align_and_standardize(d, UNI)


# ===========================================================================
# A. Dataset 容器
# ===========================================================================

class TestDatasetContainer:

    def test_the_seven_standard_fields_are_the_documented_ones(self):
        assert STANDARD_FIELDS == ["open", "high", "low", "close",
                                   "volume", "vwap", "returns"], (
            f"标准字段集被改动：{STANDARD_FIELDS}")

    def test_a_missing_standard_field_is_rejected_at_construction(self):
        """
        `__post_init__` 的缺字段检查 —— 被删会让一个缺 `volume` 的
        数据集一路流到回测里，直到某条 DSL 用到 volume 才 KeyError，
        那时已经跑了几分钟。
        """
        data = _full_panel()
        del data["volume"]
        with pytest.raises(ValueError) as ei:
            Dataset(name="broken", frequency="daily", universe=UNI, data=data)
        assert "volume" in str(ei.value)
        assert "broken" in str(ei.value), "报错里没有数据集名，看不出是哪一个"

    def test_every_missing_field_is_listed_not_just_the_first(self):
        """
        `missing = [f for f in STANDARD_FIELDS if f not in self.data]`
        —— 列表推导被改成"找到第一个就返回"会让人修一个撞一个。
        """
        data = _full_panel()
        del data["volume"]
        del data["vwap"]
        with pytest.raises(ValueError) as ei:
            Dataset(name="x", frequency="daily", universe=UNI, data=data)
        msg = str(ei.value)
        assert "volume" in msg and "vwap" in msg, f"只报了一个缺失字段：{msg}"

    def test_a_complete_panel_constructs_cleanly(self):
        ds = Dataset(name="ok", frequency="daily", universe=UNI,
                     data=_full_panel())
        assert ds.name == "ok" and ds.frequency == "daily"

    def test_n_assets_counts_the_universe_not_the_columns(self):
        """
        `len(self.universe)` —— 换成 `len(data["close"].columns)` 在
        universe 与实际列不一致时会掩盖"某些标的一条数据都没有"。
        """
        ds = Dataset(name="x", frequency="daily", universe=UNI + ["GHOST"],
                     data=_full_panel())
        assert ds.n_assets == 4, (
            "n_assets 取的是面板列数而不是 universe 长度 —— "
            "拿不到数据的标的会被悄悄抹掉")

    def test_n_dates_counts_the_rows_of_a_field(self):
        ds = Dataset(name="x", frequency="daily", universe=UNI,
                     data=_full_panel())
        assert ds.n_dates == len(IDX)

    def test_the_repr_reports_name_frequency_and_both_counts(self):
        ds = Dataset(name="us_equity", frequency="daily", universe=UNI,
                     data=_full_panel())
        r = repr(ds)
        assert "us_equity" in r and "daily" in r
        assert "assets=3" in r and f"dates={len(IDX)}" in r

    def test_the_data_payload_is_kept_out_of_the_repr(self):
        """一次 print 不能刷出几万行 DataFrame。"""
        ds = Dataset(name="x", frequency="daily", universe=UNI,
                     data=_full_panel())
        assert "close" not in repr(ds), "repr 里带上了整份面板"


# ===========================================================================
# B. 对齐与补齐
# ===========================================================================

class TestAlignAndStandardize:

    def test_an_empty_raw_dict_is_rejected(self):
        """
        `if not raw: raise ValueError("Empty raw dataset.")`
        —— `not` 被删会在**正常数据**上抛，或在空数据上让
        `next(iter(...))` 抛一个看不懂的 StopIteration。
        """
        with pytest.raises(ValueError, match="Empty raw dataset"):
            _align_and_standardize({}, UNI)

    def test_the_output_has_exactly_the_seven_standard_fields_in_order(self):
        """
        `return {f: aligned[f] for f in STANDARD_FIELDS}`
        —— 直接返回 `aligned` 会把上游多带的字段一起漏出去，
        并且顺序随输入漂移。
        """
        raw = _raw()
        raw["额外字段"] = _frame()
        out = _align_and_standardize(raw, UNI)
        assert list(out) == STANDARD_FIELDS, f"输出字段是 {list(out)}"

    def test_every_panel_is_reindexed_onto_the_universe(self):
        """
        `df.reindex(index=idx, columns=universe)` —— universe 里
        没有数据的标的必须变成 NaN 列，而不是消失。
        """
        out = _align_and_standardize(_raw(), UNI + ["NODATA"])
        for f, df in out.items():
            assert list(df.columns) == UNI + ["NODATA"], f"{f} 的列与 universe 不符"
        assert out["close"]["NODATA"].isna().all()

    def test_the_master_index_is_the_union_across_fields(self):
        """
        `idx.union(i)` —— 换成交集会让某个字段缺几天就把
        **所有**字段的那几天都删掉。
        """
        raw = _raw()
        raw["volume"] = raw["volume"].iloc[5:]
        out = _align_and_standardize(raw, UNI)
        assert len(out["close"].index) == len(IDX), (
            f"并集索引长度是 {len(out['close'].index)} —— union 被改成了交集")

    def test_gaps_are_forward_filled_for_at_most_five_days(self):
        """
        `ffill(limit=5)`

        上限被删会让退市股以最后价格一路填到区间末尾 ——
        回测里表现为一条零波动、零回撤的完美持仓。
        构造：某标的从第 5 天起全缺，第 5..9 天应被前推，第 10 天起必须是 NaN。
        """
        raw = _raw()
        for f in raw:
            raw[f] = raw[f].copy()
            raw[f].iloc[5:, 0] = np.nan
        out = _align_and_standardize(raw, UNI)
        col = out["close"].iloc[:, 0]
        assert col.iloc[5:10].notna().all(), f"缺口前 5 天没有被前推：{col.tolist()}"
        assert col.iloc[10:].isna().all(), (
            f"缺口第 6 天起仍在前推：{col.tolist()} —— `ffill(limit=5)` 的上限被删了")

    def test_vwap_is_derived_from_high_low_and_close_when_absent(self):
        """`(h + l + c) / 3.0` —— 逐位比对，算符翻面必红。"""
        out = _align_and_standardize(_raw(), UNI)
        ref = (out["high"] + out["low"] + out["close"]) / 3.0
        assert np.allclose(out["vwap"].values, ref.values, rtol=1e-12,
                           equal_nan=True)
        ok = out["close"].notna()
        assert (out["vwap"][ok] >= out["low"][ok]).all().all()
        assert (out["vwap"][ok] <= out["high"][ok]).all().all()

    def test_a_supplied_vwap_is_not_overwritten(self):
        """`if "vwap" not in aligned:` —— `not` 被删会用派生值盖掉真实 VWAP。"""
        raw = _raw()
        raw["vwap"] = _frame(base=7.0)
        out = _align_and_standardize(raw, UNI)
        assert out["vwap"].iloc[0, 0] == pytest.approx(7.0), (
            "上游提供的 vwap 被派生值覆盖了")

    def test_the_derived_vwap_really_differs_from_close(self):
        """
        **首测存活项（L167，删掉 not）**：
        `if h is None and l is not None and c is not None:` 会让**有完整
        OHLC 时**走进 else 分支 —— vwap 直接等于 close。

        只有当 `(h+l+c)/3 != close` 时这个差别才看得见。
        构造用了 +5% / -2% 的不对称区间，这里正面钉住这个前提。
        """
        out = _align_and_standardize(_raw(), UNI)
        assert not np.allclose(out["vwap"].values, out["close"].values), (
            "派生出的 vwap 恰好等于 close —— 构造的 high/low 太对称，"
            "本组用例已失去区分力")
        ref = (out["high"] + out["low"] + out["close"]) / 3.0
        assert np.allclose(out["vwap"].values, ref.values, rtol=1e-12)

    def test_a_half_present_ohlc_falls_back_instead_of_crashing(self):
        """
        **首测存活项（L167，and → or）**：
        `if h is not None or l is not None and c is not None:` 在
        **只有 high、没有 low** 时会变成真 → `h + None + c` 直接 TypeError。

        真实数据里"有最高价没最低价"完全可能（部分免费源只给
        Close/High）。原始的三重 `and` 会稳稳退回 close。
        """
        out = _align_and_standardize({"close": _frame(), "high": _frame(base=200.0)},
                                     UNI)
        assert np.allclose(out["vwap"].values, out["close"].values), (
            "缺 low 时 vwap 没有退回 close")

        out2 = _align_and_standardize({"close": _frame(), "low": _frame(base=50.0)},
                                      UNI)
        assert np.allclose(out2["vwap"].values, out2["close"].values), (
            "缺 high 时 vwap 没有退回 close")

    def test_vwap_falls_back_to_close_when_high_low_are_missing(self):
        """
        `base = aligned.get("close", ...)` + `base.copy()`
        —— 这条兜底被删会让只有 close 的数据源（不少免费源如此）
        在构造 Dataset 时因缺 vwap 而报错。
        """
        out = _align_and_standardize({"close": _frame()}, UNI)
        assert np.allclose(out["vwap"].values, out["close"].values,
                           equal_nan=True)

    def test_the_vwap_fallback_is_a_copy_not_an_alias(self):
        """`base.copy()` —— 少了 copy 会让改 vwap 连带改掉 close。"""
        out = _align_and_standardize({"close": _frame()}, UNI)
        out["vwap"].iloc[0, 0] = -999.0
        assert out["close"].iloc[0, 0] != -999.0, "vwap 与 close 共用了同一个对象"

    def test_returns_are_log_returns_of_close_when_absent(self):
        out = _align_and_standardize(_raw(), UNI)
        ref = np.log(out["close"] / out["close"].shift(1))
        assert np.allclose(out["returns"].values, ref.values, rtol=1e-12,
                           equal_nan=True)
        assert out["returns"].abs().max().max() < 1.0, (
            "收益率量级异常 —— `close / close.shift(1)` 的除法被改了")

    def test_a_supplied_returns_field_is_not_overwritten(self):
        raw = _raw()
        raw["returns"] = _frame(base=0.0) * 0.0 + 0.01
        out = _align_and_standardize(raw, UNI)
        assert np.allclose(out["returns"].values, 0.01, rtol=1e-12), (
            "上游提供的 returns 被派生值覆盖了")

    def test_a_missing_field_becomes_all_nan_not_all_zero(self):
        """
        `aligned[f] = pd.DataFrame(np.nan, ...)`

        填 0 会让缺失的 `volume` 变成"成交量为零"——
        下游的 ADV / 冲击成本全部归零，等于宣布"任意大的单子都没成本"。
        NaN 则是可见的缺失。
        """
        out = _align_and_standardize({"close": _frame()}, UNI)
        assert out["volume"].isna().all().all(), (
            "缺失的 volume 被填成了数值 —— 成本模型会静默失效")
        assert out["open"].isna().all().all()

    def test_the_filler_panel_matches_the_shape_of_the_real_ones(self):
        out = _align_and_standardize({"close": _frame()}, UNI)
        for f in STANDARD_FIELDS:
            assert out[f].shape == out["close"].shape, f"{f} 的形状与 close 不一致"
            assert list(out[f].index) == list(out["close"].index)
            assert list(out[f].columns) == list(out["close"].columns)

    def test_all_panels_are_float_typed(self):
        out = _align_and_standardize(_raw(), UNI)
        for f, df in out.items():
            assert (df.dtypes == np.float64).all(), f"{f} 不是 float64"


# ===========================================================================
# C. 四个加载器
# ===========================================================================

@pytest.fixture
def loaders(monkeypatch):
    """把三条取数路径全部换成桩，记录每次调用的参数。"""
    seen = {"yf": [], "ccxt": [], "local": []}

    def yf(tickers, start, end):
        seen["yf"].append({"tickers": list(tickers), "start": start, "end": end})
        return _raw()

    def ccxt(tickers, start, end, exchange_id="binance"):
        seen["ccxt"].append({"tickers": list(tickers), "start": start,
                             "end": end, "exchange_id": exchange_id})
        return _raw()

    def local(data_dir, tickers, start, end):
        seen["local"].append({"data_dir": data_dir, "tickers": list(tickers),
                              "start": start, "end": end})
        return _raw()

    monkeypatch.setattr(MD, "_load_via_yfinance", yf)
    monkeypatch.setattr(MD, "_load_via_ccxt", ccxt)
    monkeypatch.setattr(MD, "_load_from_local", local)
    return types.SimpleNamespace(seen=seen, monkeypatch=monkeypatch)


class TestLoaders:

    def test_each_loader_stamps_its_own_dataset_name(self, loaders):
        """
        四个 `name=` 字面量 —— 被改会让下游按名字分派
        （成本口径、交易日历、市值来源）全部走错分支。
        """
        assert load_us_equity().name == "us_equity"
        assert load_china_a().name == "china_a"
        assert load_crypto().name == "crypto"
        assert load_etf().name == "etf"

    def test_each_loader_uses_its_own_default_universe(self, loaders):
        assert load_us_equity().universe == US_EQUITY_UNIVERSE
        assert load_china_a().universe == CHINA_A_UNIVERSE
        assert load_crypto().universe == CRYPTO_UNIVERSE
        assert load_etf().universe == ETF_UNIVERSE

    def test_the_four_default_universes_are_disjoint(self):
        """
        四个 universe 常量被互相赋值（复制粘贴的典型事故）会让
        "加载 A 股"拿到美股列表，而一切都不报错。
        """
        names = {"us": US_EQUITY_UNIVERSE, "cn": CHINA_A_UNIVERSE,
                 "crypto": CRYPTO_UNIVERSE, "etf": ETF_UNIVERSE}
        keys = list(names)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = names[keys[i]], names[keys[j]]
                assert set(a) & set(b) == set(), (
                    f"{keys[i]} 与 {keys[j]} 的 universe 有重叠："
                    f"{sorted(set(a) & set(b))}")

    def test_the_china_universe_carries_exchange_suffixes(self):
        """A 股代码必须带 .SS/.SZ，否则 yfinance 拉不到任何数据。"""
        assert all(t.endswith((".SS", ".SZ")) for t in CHINA_A_UNIVERSE), (
            "A 股 universe 里有不带交易所后缀的代码")

    def test_the_crypto_universe_uses_usd_pairs(self):
        assert all(t.endswith("-USD") for t in CRYPTO_UNIVERSE)

    def test_explicit_tickers_override_the_default_universe(self, loaders):
        """`universe = tickers or US_EQUITY_UNIVERSE` —— `or` 的两侧。"""
        ds = load_us_equity(tickers=["X", "Y"])
        assert ds.universe == ["X", "Y"]
        assert loaders.seen["yf"][0]["tickers"] == ["X", "Y"]

    def test_an_empty_ticker_list_falls_back_to_the_default(self, loaders):
        assert load_us_equity(tickers=[]).universe == US_EQUITY_UNIVERSE

    def test_a_data_dir_routes_to_the_local_provider(self, loaders):
        """
        `_load_from_local(...) if data_dir else _load_via_yfinance(...)`
        —— 三目条件翻面会让指定了本地目录的调用照样去打网络
        （在"完全免费"的约束下这是实打实的流量与限流风险）。
        """
        load_us_equity(data_dir="/tmp/store")
        assert len(loaders.seen["local"]) == 1
        assert loaders.seen["local"][0]["data_dir"] == "/tmp/store"
        assert loaders.seen["yf"] == [], "指定了本地目录却仍然打了网络"

    def test_without_a_data_dir_the_network_path_is_used(self, loaders):
        load_us_equity()
        assert len(loaders.seen["yf"]) == 1
        assert loaders.seen["local"] == []

    def test_the_date_range_is_forwarded(self, loaders):
        load_etf(start="2019-05-05", end="2021-06-06")
        assert loaders.seen["yf"][0]["start"] == "2019-05-05"
        assert loaders.seen["yf"][0]["end"] == "2021-06-06"

    def test_the_default_window_is_the_documented_one(self, loaders):
        load_etf()
        assert loaders.seen["yf"][0]["start"] == "2018-01-01"
        assert loaders.seen["yf"][0]["end"] == "2024-01-01"

    def test_crypto_prefers_the_local_dir_over_ccxt(self, loaders):
        """三分支的优先级：data_dir > use_ccxt > yfinance。"""
        load_crypto(data_dir="/tmp/s", use_ccxt=True)
        assert len(loaders.seen["local"]) == 1
        assert loaders.seen["ccxt"] == []

    def test_crypto_uses_ccxt_only_when_asked(self, loaders):
        load_crypto(use_ccxt=True, exchange_id="okx")
        assert len(loaders.seen["ccxt"]) == 1
        assert loaders.seen["ccxt"][0]["exchange_id"] == "okx"
        assert loaders.seen["yf"] == []

    def test_crypto_defaults_to_yfinance(self, loaders):
        """
        `use_ccxt: bool = False` —— 翻成 True 会让默认路径依赖
        一个未必装了的可选依赖，且每次都打交易所接口。
        """
        load_crypto()
        assert len(loaders.seen["yf"]) == 1
        assert loaders.seen["ccxt"] == []

    def test_every_loader_produces_a_complete_seven_field_panel(self, loaders):
        for fn in (load_us_equity, load_china_a, load_crypto, load_etf):
            ds = fn(tickers=UNI)
            assert set(ds.data) == set(STANDARD_FIELDS), f"{fn.__name__} 字段不全"

    def test_every_loader_declares_daily_frequency(self, loaders):
        for fn in (load_us_equity, load_china_a, load_crypto, load_etf):
            assert fn(tickers=UNI).frequency == "daily"


# ===========================================================================
# D. ccxt 路径
# ===========================================================================

class _Ex:
    def __init__(self, opts=None, rows=None, fail=False):
        self.opts = opts or {}
        self._rows = rows or []
        self._fail = fail
        self.calls = []

    def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
        self.calls.append({"symbol": symbol, "timeframe": timeframe,
                           "since": since, "limit": limit})
        if self._fail:
            raise RuntimeError("boom")
        return list(self._rows)


def _rows(dates, base=100.0):
    out = []
    for i, d in enumerate(dates):
        ts = int(pd.Timestamp(d).timestamp() * 1000)
        out.append([ts, base + i, base + i + 1, base + i - 1, base + i + 0.5,
                    1e6 + i])
    return out


class TestCcxtPath:

    @staticmethod
    def _install(monkeypatch, ex, exchange_id="binance"):
        monkeypatch.setitem(sys.modules, "ccxt",
                            types.SimpleNamespace(**{exchange_id: lambda o: ex}))

    def test_usd_tickers_are_converted_to_usdt_pairs(self, monkeypatch):
        """
        `ticker.replace("-USD", "/USDT").replace("-USDT", "/USDT")`

        转换被改会让交易所收到 "BTC-USD" 这种它不认识的代码，
        每只票都异常 → 静默退回 yfinance，ccxt 路径形同虚设。
        """
        ex = _Ex(rows=_rows(pd.bdate_range("2022-01-03", periods=5)))
        self._install(monkeypatch, ex)
        _load_via_ccxt(["BTC-USD", "ETH-USD"], "2022-01-01", "2022-02-01")
        assert [c["symbol"] for c in ex.calls] == ["BTC/USDT", "ETH/USDT"], (
            f"代码转换结果是 {[c['symbol'] for c in ex.calls]}")

    def test_rate_limiting_is_enabled(self, monkeypatch):
        holder = {}

        def _cls(opts):
            holder["opts"] = opts
            return _Ex(rows=_rows(pd.bdate_range("2022-01-03", periods=3)))

        monkeypatch.setitem(sys.modules, "ccxt",
                            types.SimpleNamespace(binance=_cls))
        _load_via_ccxt(["BTC-USD"], "2022-01-01", "2022-02-01")
        assert holder["opts"] == {"enableRateLimit": True}

    def test_both_window_endpoints_are_inclusive(self, monkeypatch):
        """`if start_ms <= row[0] <= end_ms` —— 任一比较符翻面就丢端点。"""
        dates = pd.bdate_range("2022-01-03", periods=5)
        ex = _Ex(rows=_rows(dates))
        self._install(monkeypatch, ex)
        out = _load_via_ccxt(["BTC-USD"], str(dates[1].date()),
                             str(dates[3].date()))
        got = [str(pd.Timestamp(d).date()) for d in out["close"].index]
        assert got == [str(d.date()) for d in dates[1:4]], (
            f"区间取回了 {got} —— 端点的开闭被改了")

    def test_timestamps_are_milliseconds(self, monkeypatch):
        """`int(pd.Timestamp(start).timestamp() * 1000)` —— 秒/毫秒换算。"""
        ex = _Ex(rows=_rows(pd.bdate_range("2022-01-03", periods=3)))
        self._install(monkeypatch, ex)
        _load_via_ccxt(["BTC-USD"], "2022-01-01", "2022-02-01")
        assert ex.calls[0]["since"] == int(
            pd.Timestamp("2022-01-01").timestamp() * 1000)
        assert ex.calls[0]["since"] > 1_000_000_000_000

    def test_the_ohlcv_columns_land_in_the_right_fields(self, monkeypatch):
        """
        `row[1..5]` → open/high/low/close/volume。
        下标错位会让"最高价"其实是开盘价，OHLC 不变量全部失效。
        """
        dates = pd.bdate_range("2022-01-03", periods=3)
        ex = _Ex(rows=_rows(dates, base=100.0))
        self._install(monkeypatch, ex)
        out = _load_via_ccxt(["BTC-USD"], "2022-01-01", "2022-02-01")
        assert out["open"].iloc[0, 0] == pytest.approx(100.0)
        assert out["high"].iloc[0, 0] == pytest.approx(101.0)
        assert out["low"].iloc[0, 0] == pytest.approx(99.0)
        assert out["close"].iloc[0, 0] == pytest.approx(100.5)
        assert out["volume"].iloc[0, 0] == pytest.approx(1e6)
        assert (out["high"] >= out["low"]).all().all()

    def test_a_missing_ccxt_falls_back_to_yfinance(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ccxt", None)
        called = {}
        monkeypatch.setattr(MD, "_load_via_yfinance",
                            lambda t, s, e: called.setdefault("hit", True) or _raw())
        _load_via_ccxt(["BTC-USD"], "2022-01-01", "2022-02-01")
        assert called.get("hit") is True

    def test_an_empty_ccxt_result_falls_back_to_yfinance(self, monkeypatch):
        """`if not index_set: return _load_via_yfinance(...)` —— `not` 的方向。"""
        self._install(monkeypatch, _Ex(rows=[]))
        called = {}
        monkeypatch.setattr(MD, "_load_via_yfinance",
                            lambda t, s, e: called.setdefault("hit", True) or _raw())
        _load_via_ccxt(["BTC-USD"], "2022-01-01", "2022-02-01")
        assert called.get("hit") is True

    def test_a_successful_ccxt_result_does_not_fall_back(self, monkeypatch):
        self._install(monkeypatch, _Ex(rows=_rows(
            pd.bdate_range("2022-01-03", periods=4))))
        called = {}
        monkeypatch.setattr(MD, "_load_via_yfinance",
                            lambda t, s, e: called.setdefault("hit", True) or _raw())
        _load_via_ccxt(["BTC-USD"], "2022-01-01", "2022-02-01")
        assert "hit" not in called, (
            "ccxt 成功取到数据却仍然退回了 yfinance —— `if not index_set` 的方向反了")

    def test_a_per_ticker_failure_does_not_kill_the_batch(self, monkeypatch):
        dates = pd.bdate_range("2022-01-03", periods=4)
        good = _rows(dates)

        class _Mixed(_Ex):
            def fetch_ohlcv(self, symbol, **k):
                self.calls.append({"symbol": symbol, **k})
                if symbol.startswith("BAD"):
                    raise RuntimeError("delisted")
                return list(good)

        self._install(monkeypatch, _Mixed())
        out = _load_via_ccxt(["BAD-USD", "BTC-USD"], "2022-01-01", "2022-02-01")
        assert out["close"]["BTC-USD"].notna().any()
        assert out["close"]["BAD-USD"].isna().all()

    def test_a_pipeline_level_failure_falls_back_to_yfinance(self, monkeypatch):
        """外层 `except Exception` —— 交易所类构造失败也要能降级。"""
        monkeypatch.setitem(sys.modules, "ccxt", types.SimpleNamespace())
        called = {}
        monkeypatch.setattr(MD, "_load_via_yfinance",
                            lambda t, s, e: called.setdefault("hit", True) or _raw())
        _load_via_ccxt(["BTC-USD"], "2022-01-01", "2022-02-01")
        assert called.get("hit") is True

    def test_columns_follow_the_requested_ticker_order(self, monkeypatch):
        self._install(monkeypatch, _Ex(rows=_rows(
            pd.bdate_range("2022-01-03", periods=3))))
        out = _load_via_ccxt(["ETH-USD", "BTC-USD"], "2022-01-01", "2022-02-01")
        assert list(out["close"].columns) == ["ETH-USD", "BTC-USD"]

    def test_the_index_is_sorted(self, monkeypatch):
        """`pd.DatetimeIndex(sorted(index_set))` —— set 的迭代序不是时间序。"""
        dates = pd.bdate_range("2022-01-03", periods=8)
        self._install(monkeypatch, _Ex(rows=list(reversed(_rows(dates)))))
        out = _load_via_ccxt(["BTC-USD"], "2022-01-01", "2022-02-01")
        idx = list(out["close"].index)
        assert idx == sorted(idx), f"ccxt 面板索引没有排序：{idx}"


# ===========================================================================
# E. 注册表
# ===========================================================================

class TestDatasetRegistry:

    def test_the_four_builtin_datasets_are_registered(self):
        assert DatasetRegistry().available() == ["china_a", "crypto", "etf",
                                                 "us_equity"]

    def test_available_is_sorted(self):
        r = DatasetRegistry()
        r.register("zzz", lambda **k: None)
        r.register("aaa", lambda **k: None)
        assert r.available() == sorted(r.available())

    def test_a_custom_loader_can_be_registered_and_used(self, loaders):
        r = DatasetRegistry()
        marker = Dataset(name="mine", frequency="daily", universe=UNI,
                         data=_full_panel())
        r.register("mine", lambda **k: marker)
        assert r.load("mine") is marker

    def test_registering_does_not_leak_into_other_instances(self):
        """
        `self._loaders = dict(self._BUILTIN_LOADERS)` —— 少了 `dict(...)`
        会让所有实例共用类级字典，一次自定义注册污染全局。
        """
        a = DatasetRegistry()
        a.register("private", lambda **k: None)
        assert "private" not in DatasetRegistry().available(), (
            "自定义加载器泄漏到了新实例 —— `dict(_BUILTIN_LOADERS)` 的拷贝没了")
        assert "private" not in DatasetRegistry._BUILTIN_LOADERS

    def test_an_unknown_name_raises_and_lists_what_is_available(self):
        r = DatasetRegistry()
        with pytest.raises(KeyError) as ei:
            r.load("us_equities")
        msg = str(ei.value)
        assert "us_equities" in msg
        assert "us_equity" in msg, "报错里没有列出可用数据集，使用者无从修正"

    def test_the_cache_key_covers_name_start_and_end(self, loaders):
        """
        `cache_key = f"{name}|{start}|{end}"`

        少一段（比如只用 name）会让**换了日期区间却拿到上一次的缓存** ——
        回测跑在错误的时间窗上，没有任何提示。这是本模块最危险的一处。
        """
        r = DatasetRegistry()
        a = r.load("us_equity", start="2019-01-01", end="2020-01-01",
                   tickers=UNI)
        b = r.load("us_equity", start="2021-01-01", end="2022-01-01",
                   tickers=UNI)
        assert a is not b, (
            "换了日期区间却拿到同一个缓存对象 —— 缓存键里少了日期")
        assert len(loaders.seen["yf"]) == 2
        assert loaders.seen["yf"][0]["start"] != loaders.seen["yf"][1]["start"]

    def test_different_datasets_do_not_share_a_cache_entry(self, loaders):
        r = DatasetRegistry()
        a = r.load("us_equity", tickers=UNI)
        b = r.load("etf", tickers=UNI)
        assert a.name == "us_equity" and b.name == "etf"

    def test_the_cache_returns_the_same_object_on_a_repeat_load(self, loaders):
        r = DatasetRegistry()
        a = r.load("us_equity", tickers=UNI)
        b = r.load("us_equity", tickers=UNI)
        assert a is b
        assert len(loaders.seen["yf"]) == 1, "缓存命中却又拉了一次数据"

    def test_use_cache_false_bypasses_both_read_and_write(self, loaders):
        """
        `if use_cache and cache_key in self._cache` / `if use_cache:`
        —— 两处守卫。任一被删会让 `use_cache=False` 仍然读到旧数据，
        或把这次的结果污染进缓存。
        """
        r = DatasetRegistry()
        a = r.load("us_equity", tickers=UNI, use_cache=False)
        b = r.load("us_equity", tickers=UNI, use_cache=False)
        assert a is not b, "use_cache=False 仍然命中了缓存"
        assert len(loaders.seen["yf"]) == 2

        c = r.load("us_equity", tickers=UNI, use_cache=True)
        assert c is not a and c is not b, (
            "use_cache=False 的结果被写进了缓存")

    def test_caching_is_on_by_default(self, loaders):
        r = DatasetRegistry()
        assert r.load("us_equity", tickers=UNI) is r.load("us_equity",
                                                          tickers=UNI)

    def test_clear_cache_forces_a_reload(self, loaders):
        r = DatasetRegistry()
        a = r.load("us_equity", tickers=UNI)
        r.clear_cache()
        b = r.load("us_equity", tickers=UNI)
        assert a is not b
        assert len(loaders.seen["yf"]) == 2

    def test_extra_kwargs_reach_the_loader(self, loaders):
        r = DatasetRegistry()
        r.load("crypto", tickers=UNI, use_ccxt=True, exchange_id="okx")
        assert loaders.seen["ccxt"][0]["exchange_id"] == "okx"

    def test_the_date_range_reaches_the_loader(self, loaders):
        DatasetRegistry().load("etf", start="2020-02-02", end="2021-03-03",
                               tickers=UNI)
        assert loaders.seen["yf"][0]["start"] == "2020-02-02"
        assert loaders.seen["yf"][0]["end"] == "2021-03-03"


class TestModuleLevelApi:

    def test_get_registry_returns_the_same_shared_instance(self):
        assert get_registry() is get_registry()

    def test_load_dataset_uses_the_shared_registry(self, loaders):
        get_registry().clear_cache()
        try:
            ds = load_dataset("us_equity", start="2020-01-01",
                              end="2021-01-01", tickers=UNI)
            assert ds.name == "us_equity"
            assert get_registry()._cache, "结果没有进共享注册表的缓存"
        finally:
            get_registry().clear_cache()

    def test_load_dataset_forwards_use_cache(self, loaders):
        get_registry().clear_cache()
        try:
            a = load_dataset("etf", tickers=UNI, use_cache=False)
            b = load_dataset("etf", tickers=UNI, use_cache=False)
            assert a is not b
        finally:
            get_registry().clear_cache()


# ===========================================================================
# F. yfinance 取数路径本身
# ===========================================================================

class TestYahooLoaderWiring:
    """
    上面 `loaders` fixture 把 `_load_via_yfinance` 整个换掉了，
    所以这个函数**自己的函数体**一条用例都没覆盖到
    （首测 L201 的两个布尔翻面就是这么活下来的）。
    这里只桩掉 `YahooFinanceProvider`，验函数体本身。
    """

    @staticmethod
    def _spy(monkeypatch):
        import app.core.data_engine.yahoo_provider as YP
        seen = {}

        class _Prov:
            def __init__(self, **kwargs):
                seen["ctor"] = dict(kwargs)

            def fetch(self, tickers, start=None, end=None):
                seen["fetch"] = {"tickers": list(tickers), "start": start,
                                 "end": end}
                return {"close": _frame()}

        monkeypatch.setattr(YP, "YahooFinanceProvider", _Prov)
        return seen

    def test_prices_are_requested_adjusted_and_without_a_progress_bar(self,
                                                                      monkeypatch):
        """
        **首测存活项（L201，两处）**：
        `YahooFinanceProvider(auto_adjust=True, progress=False)`

        `auto_adjust=False` → 拿到**未复权**价格，每个除权日都会在
        收益率序列里留下一根凭空的大跌，四个市场的因子全被污染；
        `progress=True` → 在无人值守的日循环日志里刷进度条控制字符。
        """
        seen = self._spy(monkeypatch)
        MD._load_via_yfinance(["AAA"], "2020-01-01", "2021-01-01")
        assert seen["ctor"] == {"auto_adjust": True, "progress": False}, (
            f"YahooFinanceProvider 的构造参数是 {seen['ctor']} —— "
            f"复权开关或进度条开关被改了")

    def test_the_tickers_and_window_are_forwarded(self, monkeypatch):
        seen = self._spy(monkeypatch)
        MD._load_via_yfinance(["AAA", "BBB"], "2019-02-02", "2020-03-03")
        assert seen["fetch"] == {"tickers": ["AAA", "BBB"],
                                 "start": "2019-02-02", "end": "2020-03-03"}

    def test_a_fetch_failure_degrades_to_an_empty_dict(self, monkeypatch):
        """
        `except Exception: logger.warning(...); raw = {}`
        —— 兜底被删会让一次网络抖动把整个数据集加载打成异常；
        当前契约是返回空 dict，交由 `_align_and_standardize` 去抛
        一个说得清楚的 "Empty raw dataset."。
        """
        import app.core.data_engine.yahoo_provider as YP

        class _Boom:
            def __init__(self, **k):
                pass

            def fetch(self, *a, **k):
                raise RuntimeError("network down")

        monkeypatch.setattr(YP, "YahooFinanceProvider", _Boom)
        assert MD._load_via_yfinance(["AAA"], "2020-01-01", "2021-01-01") == {}

    def test_a_failed_load_surfaces_as_an_actionable_error_upstream(self,
                                                                    monkeypatch):
        import app.core.data_engine.yahoo_provider as YP

        class _Boom:
            def __init__(self, **k):
                pass

            def fetch(self, *a, **k):
                raise RuntimeError("network down")

        monkeypatch.setattr(YP, "YahooFinanceProvider", _Boom)
        with pytest.raises(ValueError, match="Empty raw dataset"):
            load_us_equity(tickers=["AAA"])


class TestLocalLoaderWiring:

    def test_the_local_provider_is_rooted_at_the_given_directory(self,
                                                                  monkeypatch):
        import app.core.data_engine.local_parquet_provider as LP
        seen = {}

        class _Prov:
            def __init__(self, root_dir=None):
                seen["root"] = root_dir

            def fetch(self, tickers, start=None, end=None):
                seen["fetch"] = {"tickers": list(tickers), "start": start,
                                 "end": end}
                return {"close": _frame()}

        monkeypatch.setattr(LP, "LocalParquetProvider", _Prov)
        MD._load_from_local("/data/store", ["AAA"], "2020-01-01", "2021-01-01")
        assert seen["root"] == "/data/store"
        assert seen["fetch"]["tickers"] == ["AAA"]

    def test_a_local_failure_degrades_to_an_empty_dict(self, monkeypatch):
        import app.core.data_engine.local_parquet_provider as LP

        class _Boom:
            def __init__(self, root_dir=None):
                pass

            def fetch(self, *a, **k):
                raise RuntimeError("磁盘坏了")

        monkeypatch.setattr(LP, "LocalParquetProvider", _Boom)
        assert MD._load_from_local("/x", ["AAA"], "2020-01-01", "2021-01-01") == {}


# ===========================================================================
# G. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "L100 `data: Dict[str, pd.DataFrame] = field(repr=False)` 的 False → True":
        "`Dataset` **自己定义了 `__repr__`**（源码 L117-121），"
        "它完全取代了 dataclass 自动生成的那一个。"
        "`field(repr=...)` 只影响自动生成的 `__repr__`，"
        "自定义实现里根本没有引用 `data`，"
        "所以这个标志位在当前类上不产生任何可观察行为 —— "
        "无论 True 还是 False，`repr(ds)` 都输出同一串。"
        "（它仍然值得保留：将来若删掉自定义 `__repr__`，"
        "这个标志位会立刻重新生效，防止 print 刷出整份面板。）"
        "机械验证见 test_the_dataclass_repr_is_overridden_by_a_custom_one。",
}


def test_the_dataclass_repr_is_overridden_by_a_custom_one():
    """
    L100 等价性的机械验证：`Dataset` 必须有自己的 `__repr__`，
    且它不引用 `data` 字段。

    哪天自定义 `__repr__` 被删掉（或改成引用 data），
    这条会红 —— 上面那份等价性证明随即作废，
    `field(repr=False)` 也重新变成一个能杀的变异点。
    """
    import ast
    import inspect

    assert "__repr__" in Dataset.__dict__, (
        "Dataset 不再自定义 __repr__ —— L100 的等价性证明作废，"
        "field(repr=False) 重新有了可观察行为")

    src = inspect.getsource(Dataset.__repr__)
    dedented = "\n".join(line[4:] if line.startswith("    ") else line
                         for line in src.splitlines())
    tree = ast.parse(dedented)
    reads = {n.attr for n in ast.walk(tree)
             if isinstance(n, ast.Attribute)
             and isinstance(n.value, ast.Name) and n.value.id == "self"}
    assert "data" not in reads, (
        f"自定义 __repr__ 引用了 data（读到的字段：{sorted(reads)}）—— "
        f"L100 的等价性证明作废")
    assert reads, "自定义 __repr__ 一个字段都没读，本验证失去意义"


def test_every_survivor_has_a_written_proof():
    """
    首测 23 点 / 存活 5：其中 4 处已由新增用例杀死
    （L167 两处 + L201 两处），剩 L100 一处为等价变异。
    """
    assert len(PROVEN_EQUIVALENT) == 1
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
