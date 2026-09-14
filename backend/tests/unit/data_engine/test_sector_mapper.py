"""
data_engine/sector_mapper.py —— 静态 GICS 行业分类映射

**首测击杀率 0.0%（5/5 全部存活）** —— 既有测试只 import 过它。

这张表喂给 `ind_neutralize` / `group_rank` / `group_zscore` 这类
**截面分组算子**。它错了不会抛，只会让"行业中性化"中性到错误的组上：

  - 未知 ticker 返回 `-1`，而 `-1` 在分组算子眼里是一个**合法的组** ——
    于是所有映射不到的票被归进同一个"第 -1 行业"，互相做中性化。
    覆盖率掉下去时没有任何信号，只有因子悄悄变差。
  - `build_sector_matrix(dynamic=True)` 的默认值被翻成 True，
    会让每次构建都对未覆盖的票发起 yfinance 网络请求 ——
    在"完全免费跑模拟"的约束下直接撞限流。
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

import app.core.data_engine.sector_mapper as SM
from app.core.data_engine.sector_mapper import (
    SECTOR_CODES,
    SECTOR_NAMES,
    build_sector_matrix,
    clear_dynamic_cache,
    coverage_report,
    get_sector_code,
    get_sector_code_dynamic,
    get_sector_name,
)

DATES = pd.bdate_range("2022-01-03", periods=5)


@pytest.fixture(autouse=True)
def _clean_cache():
    """动态查询缓存是模块级的 —— 不清会让用例之间互相污染。"""
    clear_dynamic_cache()
    yield
    clear_dynamic_cache()


# ===========================================================================
# A. 代码表
# ===========================================================================

class TestCodeTables:

    def test_the_twelve_documented_sectors_are_all_present(self):
        assert SECTOR_CODES == {
            "Information Technology": 0, "Financials": 1, "Healthcare": 2,
            "Energy": 3, "Consumer Discretionary": 4, "Consumer Staples": 5,
            "Industrials": 6, "Communication Services": 7, "Materials": 8,
            "Real Estate": 9, "Utilities": 10, "Crypto": 11,
        }, f"GICS 代码表被改动：{SECTOR_CODES}"

    def test_the_codes_are_unique(self):
        """
        两个行业撞同一个代码，会让它们在 `ind_neutralize` 里被当成一个组 ——
        跨行业的多空敞口被误判成行业内的。
        """
        codes = list(SECTOR_CODES.values())
        assert len(codes) == len(set(codes)), (
            f"代码重复：{[c for c in set(codes) if codes.count(c) > 1]}")

    def test_the_reverse_lookup_is_consistent(self):
        """`SECTOR_NAMES = {v: k for k, v in SECTOR_CODES.items()}` —— 双向一致。"""
        assert len(SECTOR_NAMES) == len(SECTOR_CODES)
        for name, code in SECTOR_CODES.items():
            assert SECTOR_NAMES[code] == name

    def test_every_statically_mapped_ticker_maps_to_a_known_sector(self):
        """
        静态表里出现一个拼错的行业名（比如 "Health Care" 多了个空格），
        那一批 ticker 会全部退回 -1 —— 而表面上它们"有映射"。
        """
        bad = {t: s for t, s in SM._STATIC_SECTOR_MAP.items()
               if s not in SECTOR_CODES}
        assert not bad, f"静态表里有不认识的行业名：{bad}"

    def test_the_yfinance_label_map_targets_known_sectors(self):
        """yfinance 的标签映射同理 —— 目标必须都在代码表里。"""
        bad = {k: v for k, v in SM._YF_SECTOR_MAP.items()
               if v not in SECTOR_CODES}
        assert not bad, f"yfinance 标签映射指向了未知行业：{bad}"


# ===========================================================================
# B. 静态查表
# ===========================================================================

class TestStaticLookup:

    @pytest.mark.parametrize("ticker,sector", [
        ("AAPL", "Information Technology"),
        ("JPM", "Financials"),
        ("JNJ", "Healthcare"),
        ("XOM", "Energy"),
        ("TSLA", "Consumer Discretionary"),
        ("KO", "Consumer Staples"),
        ("CAT", "Industrials"),
        ("GOOGL", "Communication Services"),
        ("LIN", "Materials"),
        ("PLD", "Real Estate"),
        ("NEE", "Utilities"),
        ("BTC-USD", "Crypto"),
    ])
    def test_a_representative_ticker_from_each_sector(self, ticker, sector):
        assert get_sector_code(ticker) == SECTOR_CODES[sector], (
            f"{ticker} 的行业分类不是 {sector}")
        assert get_sector_name(ticker) == sector

    def test_an_unknown_ticker_maps_to_minus_one(self):
        assert get_sector_code("不存在的代码") == -1
        assert get_sector_name("不存在的代码") == "Unknown"

    def test_the_static_lookup_never_touches_the_network(self, monkeypatch):
        """
        `get_sector_code` 的 docstring 写明"不触发网络请求"。
        它一旦退化成调用动态版本，构建一次 sector 矩阵就会打上百次网络。
        """
        monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(
            Ticker=lambda *a, **k: pytest.fail("静态查表竟然调用了 yfinance")))
        assert get_sector_code("AAPL") == 0
        assert get_sector_code("从来没见过的票") == -1

    def test_both_crypto_ticker_formats_are_covered(self):
        """
        ccxt 用 `BTC/USDT`、yfinance 用 `BTC-USD` —— 两种写法都要能查到，
        否则换个数据源行业列就整片变 -1。
        """
        assert get_sector_code("BTC/USDT") == SECTOR_CODES["Crypto"]
        assert get_sector_code("BTC-USD") == SECTOR_CODES["Crypto"]

    def test_a_shares_and_hk_codes_keep_their_suffix(self):
        assert get_sector_code("600519.SH") == SECTOR_CODES["Consumer Staples"]
        assert get_sector_code("0700.HK") == SECTOR_CODES["Communication Services"]
        assert get_sector_code("600519") == -1, (
            "不带交易所后缀的代码不该命中 —— 静态表用的是带后缀的写法")


# ===========================================================================
# C. 动态查表
# ===========================================================================

def _fake_yf(monkeypatch, sector_by_ticker, counter=None):
    class _Tk:
        def __init__(self, sym):
            self._sym = sym
            if counter is not None:
                counter.append(sym)

        @property
        def info(self):
            s = sector_by_ticker.get(self._sym)
            return {"sector": s} if s is not None else {}

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=_Tk))


class TestDynamicLookup:

    def test_the_static_map_short_circuits_before_any_network_call(self, monkeypatch):
        """
        **首测存活项 L394**：`if static is not None:`

        `not` 被删会让**已经在静态表里**的票也去打网络 ——
        一次 sector 矩阵构建从 0 次请求变成几十次，
        结果还一样（所以没人会发现），只是慢且会被限流。
        """
        calls = []
        _fake_yf(monkeypatch, {}, counter=calls)
        assert get_sector_code_dynamic("AAPL") == 0
        assert calls == [], (
            f"静态表已覆盖的票仍然发起了 {len(calls)} 次 yfinance 查询")

    def test_an_unmapped_ticker_is_looked_up_once_and_cached(self, monkeypatch):
        """
        `if ticker in _DYNAMIC_CACHE:` —— 缓存被绕过会让同一只票
        在一次构建里被反复查询（sector 矩阵里每列查一次）。
        """
        calls = []
        _fake_yf(monkeypatch, {"NEWCO": "Technology"}, counter=calls)
        assert get_sector_code_dynamic("NEWCO") == SECTOR_CODES["Information Technology"]
        assert get_sector_code_dynamic("NEWCO") == SECTOR_CODES["Information Technology"]
        assert len(calls) == 1, f"同一只票查询了 {len(calls)} 次，缓存没生效"

    def test_a_failed_lookup_is_also_cached(self, monkeypatch):
        """
        docstring 写明"无法识别的 ticker 返回 -1 并缓存，
        避免后续重复发起无效请求"。不缓存失败结果会让一批垃圾代码
        每次都重新打一遍网络。
        """
        calls = []
        _fake_yf(monkeypatch, {}, counter=calls)
        assert get_sector_code_dynamic("GARBAGE") == -1
        assert get_sector_code_dynamic("GARBAGE") == -1
        assert len(calls) == 1, f"失败结果没有被缓存（查询了 {len(calls)} 次）"

    def test_yfinance_labels_are_translated_to_gics_names(self, monkeypatch):
        """
        yfinance 给的是 "Technology" / "Financial Services" / "Consumer Cyclical"，
        与 GICS L1 名称不同。翻译表没生效的话，这些票全部退回 -1。
        """
        cases = {
            "Technology": "Information Technology",
            "Financial Services": "Financials",
            "Consumer Cyclical": "Consumer Discretionary",
            "Consumer Defensive": "Consumer Staples",
            "Basic Materials": "Materials",
        }
        for i, (yf_label, gics) in enumerate(cases.items()):
            clear_dynamic_cache()
            sym = f"SYM{i}"
            _fake_yf(monkeypatch, {sym: yf_label})
            assert get_sector_code_dynamic(sym) == SECTOR_CODES[gics], (
                f"yfinance 标签 {yf_label!r} 没有翻译成 {gics!r}")

    def test_an_unrecognised_label_falls_back_to_minus_one(self, monkeypatch):
        """
        **首测存活项 L412**：`if sector_name not in SECTOR_CODES: sector_name = None`

        `not` 被删会让**能识别的**行业名反而被丢掉（全部退回 -1），
        动态查询等于完全失效；而调用方只会看到覆盖率低，
        不会知道是翻译那一步把结果扔了。
        """
        _fake_yf(monkeypatch, {"WEIRD": "Something Nobody Defined"})
        assert get_sector_code_dynamic("WEIRD") == -1

        clear_dynamic_cache()
        _fake_yf(monkeypatch, {"GOODCO": "Healthcare"})
        assert get_sector_code_dynamic("GOODCO") == SECTOR_CODES["Healthcare"], (
            "能识别的行业名也被丢掉了 —— `not in SECTOR_CODES` 的 not 被删了")

    def test_a_yfinance_exception_degrades_to_minus_one(self, monkeypatch):
        """`except Exception: logger.debug(...)` —— 网络抖动不该掀翻整次构建。"""
        class _Boom:
            def __init__(self, sym):
                raise RuntimeError("network down")

        monkeypatch.setitem(sys.modules, "yfinance",
                            types.SimpleNamespace(Ticker=_Boom))
        assert get_sector_code_dynamic("ANY") == -1

    def test_clearing_the_cache_forces_a_fresh_lookup(self, monkeypatch):
        calls = []
        _fake_yf(monkeypatch, {"NEWCO": "Technology"}, counter=calls)
        get_sector_code_dynamic("NEWCO")
        clear_dynamic_cache()
        get_sector_code_dynamic("NEWCO")
        assert len(calls) == 2


# ===========================================================================
# D. sector 矩阵
# ===========================================================================

class TestSectorMatrix:

    def test_the_matrix_is_dates_by_tickers(self):
        tickers = ["AAPL", "JPM", "XOM"]
        m = build_sector_matrix(tickers, DATES)
        assert m.shape == (len(DATES), len(tickers))
        assert list(m.index) == list(DATES)
        assert list(m.columns) == tickers

    def test_the_codes_are_constant_along_time(self):
        """
        `np.tile(codes, (len(dates), 1))` —— 行业在时间维度上是静态的。
        广播方向搞反（tile 成 (N, T)）会让矩阵转置，
        下游按列取 ticker 时拿到的是日期。
        """
        m = build_sector_matrix(["AAPL", "JPM"], DATES)
        for col in m.columns:
            assert m[col].nunique() == 1, f"{col} 的行业代码随日期变化了"
        assert m["AAPL"].iloc[0] == 0 and m["JPM"].iloc[0] == 1

    def test_unknown_tickers_become_minus_one_columns(self):
        m = build_sector_matrix(["AAPL", "没这个票"], DATES)
        assert (m["没这个票"] == -1).all()

    def test_the_matrix_is_float_typed(self):
        """
        `dtype=float` —— 与其他字段保持一致。整型会让它在与价格面板
        做算术时触发 dtype 提升，某些算子上行为不同。
        """
        m = build_sector_matrix(["AAPL"], DATES)
        assert m.dtypes.eq(np.float64).all(), f"dtype 是 {m.dtypes.unique()}"

    def test_the_dynamic_flag_defaults_to_off(self, monkeypatch):
        """
        **首测存活项 L456**：`dynamic: bool = False`

        翻成 True 会让**每一次**构建 sector 矩阵都对未覆盖的票发起
        yfinance 请求。在"完全免费跑模拟"的约束下，这是直接撞限流 ——
        而矩阵本身看起来完全正常（只是慢，且覆盖率莫名变高）。
        """
        calls = []
        _fake_yf(monkeypatch, {"NEWCO": "Technology"}, counter=calls)
        m = build_sector_matrix(["AAPL", "NEWCO"], DATES)
        assert calls == [], (
            f"默认构建发起了 {len(calls)} 次网络查询 —— "
            f"`dynamic: bool = False` 被翻成了 True")
        assert (m["NEWCO"] == -1).all(), "默认路径不该查到未覆盖票的行业"

    def test_the_dynamic_flag_actually_enables_lookups(self, monkeypatch):
        calls = []
        _fake_yf(monkeypatch, {"NEWCO": "Technology"}, counter=calls)
        m = build_sector_matrix(["AAPL", "NEWCO"], DATES, dynamic=True)
        assert calls == ["NEWCO"], f"动态模式的查询记录不对：{calls}"
        assert (m["NEWCO"] == SECTOR_CODES["Information Technology"]).all()

    def test_an_empty_ticker_list_yields_an_empty_matrix(self):
        m = build_sector_matrix([], DATES)
        assert m.shape == (len(DATES), 0)


# ===========================================================================
# E. 覆盖率报告
# ===========================================================================

class TestCoverageReport:

    def test_mapped_and_unmapped_are_split_correctly(self):
        """
        **首测存活项 L491**：`unmapped = [t for t in tickers if t not in _STATIC_SECTOR_MAP]`

        `not` 被删会让 mapped 与 unmapped **完全对调** ——
        覆盖率报告显示"100% 未覆盖"或"100% 已覆盖"，
        两种都是彻底的误导（后者更危险：让人以为没问题）。
        """
        rep = coverage_report(["AAPL", "JPM", "没这个票", "也没有"])
        assert rep["total"] == 4
        assert rep["mapped"] == 2, f"已覆盖数是 {rep['mapped']}，应当是 2"
        assert rep["unmapped"] == 2
        assert sorted(rep["unmapped_tickers"]) == sorted(["没这个票", "也没有"]), (
            f"未覆盖清单是 {rep['unmapped_tickers']} —— mapped/unmapped 对调了")

    def test_the_two_buckets_add_up_to_the_total(self):
        rep = coverage_report(["AAPL", "JPM", "XOM", "NOPE"])
        assert rep["mapped"] + rep["unmapped"] == rep["total"]

    def test_the_distribution_counts_each_sector(self):
        """
        **首测存活项 L496**：`dist[name] = dist.get(name, 0) + 1`

        `+` 翻成 `-` 会让计数变成 -1、-2……分布图上全是负数柱；
        翻成 `*` 则恒为 0（`0 * 1 == 0`），看起来像"一个都没有"。
        用**同一行业多只票**来区分（单只票时 +1 与 -1 的绝对值一样）。
        """
        rep = coverage_report(["AAPL", "MSFT", "NVDA", "JPM"])
        dist = rep["sector_distribution"]
        assert dist["Information Technology"] == 3, (
            f"IT 行业计数是 {dist['Information Technology']}，应当是 3")
        assert dist["Financials"] == 1
        assert sum(dist.values()) == rep["mapped"], (
            "分布之和与已覆盖数对不上")

    def test_unmapped_tickers_do_not_enter_the_distribution(self):
        rep = coverage_report(["AAPL", "没这个票"])
        assert "Unknown" not in rep["sector_distribution"]
        assert sum(rep["sector_distribution"].values()) == 1

    def test_an_empty_input_reports_zeroes(self):
        rep = coverage_report([])
        assert rep["total"] == 0 and rep["mapped"] == 0 and rep["unmapped"] == 0
        assert rep["sector_distribution"] == {}

    def test_every_documented_key_is_present(self):
        rep = coverage_report(["AAPL"])
        assert set(rep) == {"total", "mapped", "unmapped",
                            "unmapped_tickers", "sector_distribution"}
