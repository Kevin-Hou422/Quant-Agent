"""
data_engine/providers/ccxt_provider.py —— Binance 加密货币日线

**此前零测试**（12 个变异点，D 档）。

这是 crypto 数据集**唯一**的入口。它坏掉的方式全部是静默的：

  - 翻页终止条件写错 → 少拿几年数据，面板照样成形，
    只是回测区间悄悄缩短，Sharpe 在一段特殊行情上被放大；
  - `since = last_ts + _MS_PER_DAY` 的 `+` 翻成 `-` → **死循环**
    或反复取同一批，30 页跑满，慢到像网络问题；
  - 区间过滤的 `<=` / `>=` 翻面 → 端点那一天被多取或少取一根，
    与其他数据源对齐时错位一天（这正是最难查的一类泄漏）；
  - `ffill(limit=3)` 的上限被改 → 停牌/断线的空洞被无限前推，
    一个死掉的币种会以最后价格"活"到区间末尾。

整条链路不碰网络：`sys.modules["ccxt"]` 注入一个可编程的假交易所，
让每一条分支都能被单独点着，并且**逐根 K 线**比对时间戳。
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.providers.ccxt_provider import (
    _MS_PER_DAY,
    CcxtBinanceProvider,
)

FIELDS = ("open", "high", "low", "close", "volume", "vwap", "returns")

DAY = 86_400_000


def _ms(date: str) -> int:
    return int(pd.Timestamp(date).timestamp() * 1000)


def _candle(ts_ms: int, base: float = 100.0) -> list:
    """[ts, open, high, low, close, volume]"""
    return [ts_ms, base, base * 1.02, base * 0.98, base * 1.01, 1_000.0]


class _FakeExchange:
    """按 since 分页返回日线；记录每一次请求的参数。"""

    def __init__(self, opts=None, candles=None, page=1000, fail_after=None):
        self.opts = opts or {}
        self._all = candles or []
        self._page = page
        self._fail_after = fail_after
        self.calls = []

    def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
        self.calls.append({"symbol": symbol, "timeframe": timeframe,
                           "since": since, "limit": limit})
        if self._fail_after is not None and len(self.calls) > self._fail_after:
            raise RuntimeError("exchange down")
        rows = [c for c in self._all if c[0] >= since]
        return rows[:self._page]


def _install_ccxt(monkeypatch, exchange=None, exchange_id="binance"):
    """把一个假的 ccxt 模块塞进 sys.modules，返回被构造出来的交易所实例。"""
    holder = {}

    def _cls(opts):
        ex = exchange if exchange is not None else _FakeExchange(opts)
        ex.opts = opts
        holder["ex"] = ex
        return ex

    mod = types.SimpleNamespace(**{exchange_id: _cls})
    monkeypatch.setitem(sys.modules, "ccxt", mod)
    return holder


# ===========================================================================
# A. 构造参数
# ===========================================================================

class TestConstruction:

    def test_the_documented_defaults(self):
        p = CcxtBinanceProvider()
        assert p.exchange_id == "binance"
        assert p.limit == 1000, "每页条数不是 Binance 的日线上限 1000"
        assert p.delay_s == 0.2

    def test_the_parameters_are_stored_verbatim(self):
        p = CcxtBinanceProvider(exchange_id="okx", limit=300, delay_s=0.0)
        assert (p.exchange_id, p.limit, p.delay_s) == ("okx", 300, 0.0)

    def test_rate_limiting_is_switched_on_for_the_exchange(self, monkeypatch):
        """
        `exchange_cls({"enableRateLimit": True})` —— 翻成 False 会让
        provider 按最快速度打 Binance，几分钟内就会吃到封 IP。
        """
        holder = _install_ccxt(monkeypatch,
                               _FakeExchange(candles=[_candle(_ms("2023-01-02"))]))
        CcxtBinanceProvider(delay_s=0.0).fetch(["BTC/USDT"], "2023-01-01", "2023-01-05")
        assert holder["ex"].opts == {"enableRateLimit": True}, (
            f"交易所构造参数是 {holder['ex'].opts} —— 限流开关被改了")

    def test_an_unknown_exchange_raises_with_the_offending_id(self, monkeypatch):
        """
        `raise ValueError(f"Unknown ccxt exchange: ...") from exc`
        —— 这个 except 被删会让调用方看到一句 `module 'ccxt' has no
        attribute 'bianance'`，看不出是自己配置拼错了。
        """
        monkeypatch.setitem(sys.modules, "ccxt", types.SimpleNamespace())
        with pytest.raises(ValueError, match="Unknown ccxt exchange: 'bianance'"):
            CcxtBinanceProvider(exchange_id="bianance").fetch(
                ["BTC/USDT"], "2023-01-01", "2023-01-05")


# ===========================================================================
# B. 时间戳换算与区间过滤
# ===========================================================================

class TestTimeWindow:

    def test_the_since_argument_is_the_start_date_in_milliseconds(self, monkeypatch):
        """
        `start_ms = int(pd.Timestamp(start).timestamp() * 1000)`

        `* 1000` 翻成 `/ 1000` 会让 since 落到 1970 年 —— Binance
        会从上市第一天开始给，翻页 30 次都到不了请求区间，
        最后返回空面板，但**没有任何报错**。
        """
        ex = _FakeExchange(candles=[_candle(_ms("2023-01-02"))])
        _install_ccxt(monkeypatch, ex)
        CcxtBinanceProvider(delay_s=0.0).fetch(["BTC/USDT"], "2023-01-01",
                                               "2023-01-10")
        assert ex.calls[0]["since"] == _ms("2023-01-01"), (
            f"首次请求的 since 是 {ex.calls[0]['since']}，"
            f"应当是 {_ms('2023-01-01')} —— 秒/毫秒换算方向反了")
        assert ex.calls[0]["since"] > 1_000_000_000_000, (
            "since 落在了毫秒量级以下 —— `* 1000` 被改成了 `/ 1000`")

    def test_the_daily_timeframe_and_page_limit_are_requested(self, monkeypatch):
        ex = _FakeExchange(candles=[_candle(_ms("2023-01-02"))])
        _install_ccxt(monkeypatch, ex)
        CcxtBinanceProvider(limit=777, delay_s=0.0).fetch(
            ["BTC/USDT"], "2023-01-01", "2023-01-10")
        assert ex.calls[0]["timeframe"] == "1d", "请求的不是日线"
        assert ex.calls[0]["limit"] == 777, "每页条数没有透传"

    def test_both_endpoints_of_the_window_are_inclusive(self, monkeypatch):
        """
        `df[df.index >= start]` 与 `df[df.index <= end]`

        任一比较符翻成严格不等，端点那一天就会被切掉 ——
        与股票面板做跨市场对齐时会整体错位一天。
        构造：恰好在 start、end 上各有一根 K 线。
        """
        days = ["2023-01-01", "2023-01-02", "2023-01-03",
                "2023-01-04", "2023-01-05"]
        ex = _FakeExchange(candles=[_candle(_ms(d), 100 + i)
                                    for i, d in enumerate(days)])
        _install_ccxt(monkeypatch, ex)
        panel = CcxtBinanceProvider(delay_s=0.0).fetch(
            ["BTC/USDT"], "2023-01-02", "2023-01-04")
        got = [str(d.date()) for d in panel["close"].index]
        assert got == ["2023-01-02", "2023-01-03", "2023-01-04"], (
            f"区间 [01-02, 01-04] 取回了 {got} —— 端点的开闭被改了")

    def test_candles_outside_the_window_are_dropped_in_the_pager_too(self,
                                                                     monkeypatch):
        """
        `[c for c in all_candles if start_ms <= c[0] <= end_ms]`
        —— 翻页器自己也做一次过滤（交易所会多给）。
        两个 `<=` 任一翻面，端点的 K 线就会在这里被丢掉，
        即使后面 DataFrame 的过滤是对的也补不回来。
        """
        p = CcxtBinanceProvider(delay_s=0.0)
        s, e = _ms("2023-01-02"), _ms("2023-01-04")
        ex = _FakeExchange(candles=[_candle(_ms(d)) for d in
                                    ("2023-01-01", "2023-01-02", "2023-01-03",
                                     "2023-01-04", "2023-01-05")])
        got = p._fetch_all_candles(ex, "BTC/USDT", s, e)
        assert [c[0] for c in got] == [s, _ms("2023-01-03"), e], (
            "翻页器的区间过滤把端点切掉了")

    def test_timestamps_are_parsed_as_utc_then_made_naive(self, monkeypatch):
        """
        `pd.to_datetime(..., unit="ms", utc=True).dt.tz_localize(None)`

        少了 `tz_localize(None)` 会让索引带上时区，与股票面板（naive）
        做 `union` 时直接抛。

        （`utc=True` 本身对整数毫秒输入不产生差别 —— 见文件末尾
        `PROVEN_EQUIVALENT` 里 L94 的等价性证明。）
        """
        ex = _FakeExchange(candles=[_candle(_ms("2023-06-15"))])
        _install_ccxt(monkeypatch, ex)
        panel = CcxtBinanceProvider(delay_s=0.0).fetch(
            ["BTC/USDT"], "2023-06-01", "2023-06-30")
        idx = panel["close"].index
        assert idx.tz is None, f"索引带着时区 {idx.tz} —— 与股票面板无法对齐"
        assert str(idx[0].date()) == "2023-06-15"

    def test_the_index_is_sorted_ascending(self, monkeypatch):
        """`sort_index()` 被删会让乱序返回的 K 线原样进面板，diff/shift 全错。"""
        days = ["2023-01-05", "2023-01-02", "2023-01-04", "2023-01-03"]
        ex = _FakeExchange(candles=[_candle(_ms(d)) for d in days])
        # 让假交易所按给定（乱）顺序返回
        ex._all = [_candle(_ms(d)) for d in days]
        ex.fetch_ohlcv = lambda symbol, timeframe=None, since=None, limit=None: (
            ex.calls.append({"symbol": symbol, "timeframe": timeframe,
                             "since": since, "limit": limit}) or ex._all)
        _install_ccxt(monkeypatch, ex)
        panel = CcxtBinanceProvider(delay_s=0.0).fetch(
            ["BTC/USDT"], "2023-01-01", "2023-01-10")
        idx = panel["close"].index
        assert list(idx) == sorted(idx), f"面板索引不是升序：{list(idx)}"


# ===========================================================================
# C. 翻页
# ===========================================================================

class TestPagination:

    def test_the_next_page_starts_one_day_after_the_last_candle(self, monkeypatch):
        """
        `since = last_ts + _MS_PER_DAY`

        `+` 翻成 `-` 会让下一页的起点**往回走一天** —— 每页都重叠、
        永远到不了区间末尾，直到 30 页跑满。既慢又少数据。
        """
        days = pd.bdate_range("2023-01-02", periods=5)
        ex = _FakeExchange(candles=[_candle(int(d.timestamp() * 1000))
                                    for d in days], page=2)
        p = CcxtBinanceProvider(limit=2, delay_s=0.0)
        p._fetch_all_candles(ex, "BTC/USDT",
                             int(days[0].timestamp() * 1000),
                             int(days[-1].timestamp() * 1000))
        sinces = [c["since"] for c in ex.calls]
        assert sinces[1] > sinces[0], (
            f"第二页的 since {sinces[1]} 不晚于第一页 {sinces[0]} —— "
            f"`last_ts + _MS_PER_DAY` 的符号被改了")
        assert sinces[1] - sinces[0] >= _MS_PER_DAY

    def test_the_day_constant_is_one_day_in_milliseconds(self):
        assert _MS_PER_DAY == 86_400_000

    def test_a_short_page_terminates_the_loop(self, monkeypatch):
        """
        `if last_ts >= end_ms or len(batch) < self.limit: break`

        `<` 翻成 `<=` 会让**满页**也被当成最后一页 —— 一次请求
        就收工，后面的数据全丢。构造：page 恰好等于 limit。
        """
        days = pd.bdate_range("2023-01-02", periods=6)
        cs = [_candle(int(d.timestamp() * 1000)) for d in days]
        ex = _FakeExchange(candles=cs, page=3)
        p = CcxtBinanceProvider(limit=3, delay_s=0.0)
        got = p._fetch_all_candles(ex, "BTC/USDT", cs[0][0], cs[-1][0])
        assert len(ex.calls) >= 2, (
            f"满页（3 条 = limit）却只请求了 {len(ex.calls)} 次 —— "
            f"`len(batch) < self.limit` 被翻成了 `<=`")
        assert len(got) == 6, f"只收集到 {len(got)} 根 K 线，应当是 6 根"

    def test_reaching_the_end_timestamp_terminates_the_loop(self, monkeypatch):
        """`last_ts >= end_ms` —— `>=` 翻成 `>` 会多打一次无用请求。"""
        days = pd.bdate_range("2023-01-02", periods=3)
        cs = [_candle(int(d.timestamp() * 1000)) for d in days]
        ex = _FakeExchange(candles=cs, page=10)
        p = CcxtBinanceProvider(limit=10, delay_s=0.0)
        p._fetch_all_candles(ex, "BTC/USDT", cs[0][0], cs[-1][0])
        assert len(ex.calls) == 1, (
            f"一页就覆盖到区间末尾，却请求了 {len(ex.calls)} 次")

    def test_an_empty_batch_terminates_the_loop(self, monkeypatch):
        """`if not batch: break` —— 守卫被删会让空返回一直循环 30 次。"""
        ex = _FakeExchange(candles=[], page=10)
        p = CcxtBinanceProvider(delay_s=0.0)
        got = p._fetch_all_candles(ex, "BTC/USDT", _ms("2023-01-01"),
                                   _ms("2023-01-10"))
        assert got == []
        assert len(ex.calls) == 1, f"空返回之后还在重试：{len(ex.calls)} 次"

    def test_the_page_loop_is_bounded(self, monkeypatch):
        """
        `for _ in range(30)` —— 上限被删（改成 while True）会在
        交易所每次都返回满页且时间戳不前进时**永久挂死**无人值守的日循环。
        构造一个恶意交易所：永远返回同一批满页数据。
        """
        stuck = [_candle(_ms("2020-01-01"))] * 5

        class _Stuck:
            def __init__(self):
                self.n = 0

            def fetch_ohlcv(self, *a, **k):
                self.n += 1
                return list(stuck)

        ex = _Stuck()
        p = CcxtBinanceProvider(limit=5, delay_s=0.0)
        p._fetch_all_candles(ex, "BTC/USDT", _ms("2020-01-01"), _ms("2030-01-01"))
        assert ex.n <= 30, f"翻页没有上限，请求了 {ex.n} 次"
        assert ex.n == 30, f"翻页上限不是 30，实测 {ex.n}"

    def test_an_exchange_error_keeps_what_was_already_collected(self, monkeypatch):
        """
        `except Exception: logger.warning(...); break`
        —— `break` 改成 `raise` 会让一次网络抖动作废整批已取数据；
        改成 `continue` 则会在持续报错时空转 30 次。
        """
        days = pd.bdate_range("2023-01-02", periods=6)
        cs = [_candle(int(d.timestamp() * 1000)) for d in days]
        ex = _FakeExchange(candles=cs, page=2, fail_after=1)
        p = CcxtBinanceProvider(limit=2, delay_s=0.0)
        got = p._fetch_all_candles(ex, "BTC/USDT", cs[0][0], cs[-1][0])
        assert len(got) == 2, (
            f"第二页报错后应当保留第一页的 2 根，实际 {len(got)} 根")
        assert len(ex.calls) == 2, "报错之后还在继续请求"


# ===========================================================================
# D. 面板组装
# ===========================================================================

def _panel_from(monkeypatch, per_symbol: dict, symbols=None, days=8):
    """per_symbol: symbol → 该 symbol 有数据的日期下标集合（None = 全部）。"""
    idx = pd.bdate_range("2023-01-02", periods=days)
    symbols = symbols or list(per_symbol)

    class _Multi:
        def __init__(self, opts=None):
            self.opts = opts or {}
            self.calls = []

        def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
            self.calls.append(symbol)
            keep = per_symbol.get(symbol)
            if keep is None:
                return []
            return [_candle(int(idx[i].timestamp() * 1000), 100.0 + i)
                    for i in keep]

    _install_ccxt(monkeypatch, _Multi())
    return CcxtBinanceProvider(delay_s=0.0).fetch(
        symbols, str(idx[0].date()), str(idx[-1].date())), idx


class TestPanelAssembly:

    def test_every_documented_field_is_produced(self, monkeypatch):
        panel, _ = _panel_from(monkeypatch, {"BTC/USDT": range(8)})
        assert set(panel) == set(FIELDS), f"字段集是 {sorted(panel)}"

    def test_columns_follow_the_requested_symbol_order(self, monkeypatch):
        """
        `for sym in symbols` —— 用 `frames` 的键序代替会让列序
        随取数成败漂移，调用方按位置取权重时错配。
        """
        panel, _ = _panel_from(monkeypatch,
                               {"ETH/USDT": range(8), "BTC/USDT": range(8)},
                               symbols=["BTC/USDT", "ETH/USDT"])
        assert list(panel["close"].columns) == ["BTC/USDT", "ETH/USDT"]

    def test_a_symbol_with_no_data_becomes_an_all_nan_column(self, monkeypatch):
        """
        `else: cols[sym] = pd.Series(np.nan, index=idx)`
        —— 这个分支被删会让取不到数据的币种**从面板里消失**，
        列数与请求数对不上，下游按 symbols 索引直接 KeyError。
        """
        panel, _ = _panel_from(monkeypatch,
                               {"BTC/USDT": range(8), "DOGE/USDT": None},
                               symbols=["BTC/USDT", "DOGE/USDT"])
        assert list(panel["close"].columns) == ["BTC/USDT", "DOGE/USDT"]
        assert panel["close"]["DOGE/USDT"].isna().all(), (
            "取不到数据的币种没有变成全 NaN 列")
        assert panel["close"]["BTC/USDT"].notna().any()

    def test_gaps_are_forward_filled_for_at_most_three_days(self, monkeypatch):
        """
        `ffill(limit=3)`

        上限被删（`ffill()`）会让一个下架币种以最后价格"活"到区间末尾，
        回测里表现为一条零波动的完美持仓；上限改小则让正常的
        周末/维护窗口留下空洞。

        构造：某币种在第 0 天有数据，之后连续缺 5 天 —— 第 1..3 天
        应当被前推，第 4、5 天必须是 NaN。
        """
        panel, idx = _panel_from(monkeypatch,
                                 {"BTC/USDT": range(8), "GAP/USDT": [0, 6, 7]},
                                 symbols=["BTC/USDT", "GAP/USDT"], days=8)
        col = panel["close"]["GAP/USDT"]
        assert col.iloc[0] == pytest.approx(101.0), "首日数据丢了"
        assert col.iloc[1:4].notna().all(), (
            f"缺口的前 3 天没有被前推：{col.tolist()}")
        assert col.iloc[4:6].isna().all(), (
            f"缺口第 4、5 天也被前推了：{col.tolist()} —— "
            f"`ffill(limit=3)` 的上限被删了")

    def test_vwap_is_the_mean_of_high_low_and_close(self, monkeypatch):
        """`(high + low + close) / 3.0` —— 逐位比对，算符翻面必红。"""
        panel, _ = _panel_from(monkeypatch, {"BTC/USDT": range(8)})
        ref = (panel["high"] + panel["low"] + panel["close"]) / 3.0
        assert np.allclose(panel["vwap"].values, ref.values, rtol=1e-12,
                           equal_nan=True)
        ok = panel["close"].notna()
        assert (panel["vwap"][ok] >= panel["low"][ok]).all().all()
        assert (panel["vwap"][ok] <= panel["high"][ok]).all().all()

    def test_returns_are_log_returns_of_close(self, monkeypatch):
        """`np.log(close / close.shift(1))` —— `/` 翻成 `*` 会让收益率变成 log(P²)。"""
        panel, _ = _panel_from(monkeypatch, {"BTC/USDT": range(8)})
        ref = np.log(panel["close"] / panel["close"].shift(1))
        assert np.allclose(panel["returns"].values, ref.values, rtol=1e-12,
                           equal_nan=True)
        # 量级检查：日收益率不可能是 log(P²) 那种 9 以上的数
        assert panel["returns"].abs().max().max() < 1.0, (
            "日收益率量级异常 —— 除法方向被改了")

    def test_ohlc_ordering_holds_on_the_assembled_panel(self, monkeypatch):
        panel, _ = _panel_from(monkeypatch, {"BTC/USDT": range(8)})
        ok = panel["close"].notna()
        assert (panel["high"][ok] >= panel["close"][ok]).all().all()
        assert (panel["low"][ok] <= panel["close"][ok]).all().all()

    def test_the_index_is_the_union_of_all_symbol_indices(self, monkeypatch):
        """
        `idx = df.index if idx is None else idx.union(df.index)`
        —— `union` 换成 `intersection` 会让任一币种停牌的日子
        从**整个面板**消失，其余币种的数据被连带丢弃。
        """
        panel, idx = _panel_from(monkeypatch,
                                 {"A/USDT": [0, 1, 2], "B/USDT": [5, 6, 7]},
                                 symbols=["A/USDT", "B/USDT"], days=8)
        assert len(panel["close"].index) == 6, (
            f"索引长度 {len(panel['close'].index)} —— union 被改成了交集")

    def test_all_fields_are_float_typed(self, monkeypatch):
        panel, _ = _panel_from(monkeypatch, {"BTC/USDT": range(8)})
        for name, df in panel.items():
            assert (df.dtypes == np.float64).all(), f"{name} 不是 float64"

    def test_a_symbol_returning_nothing_is_logged_not_fatal(self, monkeypatch):
        """`if candles: ... else: logger.warning` —— 一个空币种不能带走整批。"""
        panel, _ = _panel_from(monkeypatch,
                               {"BTC/USDT": range(8), "NOPE/USDT": None},
                               symbols=["BTC/USDT", "NOPE/USDT"])
        assert panel["close"]["BTC/USDT"].notna().any()


# ===========================================================================
# E. yfinance 兜底
# ===========================================================================

def _yf_frame(tickers, days=6, multi=True):
    idx = pd.bdate_range("2023-01-02", periods=days)
    fields = ["Open", "High", "Low", "Close", "Volume"]
    if multi:
        cols = pd.MultiIndex.from_product([fields, tickers])
        data = np.arange(len(idx) * len(cols), dtype=float).reshape(len(idx), -1) + 100
        return pd.DataFrame(data, index=idx, columns=cols)
    data = np.arange(len(idx) * len(fields), dtype=float).reshape(len(idx), -1) + 100
    return pd.DataFrame(data, index=idx, columns=fields)


class TestYfinanceFallback:

    def test_a_missing_ccxt_falls_back_instead_of_raising(self, monkeypatch):
        """
        `except ImportError: return self._yfinance_fallback(...)`
        —— 这条兜底是"完全免费跑通"的一部分：没装 ccxt 也要能出数据。
        """
        monkeypatch.setitem(sys.modules, "ccxt", None)   # import ccxt → ImportError
        called = {}

        def fake(symbols, start, end):
            called.update(symbols=symbols, start=start, end=end)
            return {"close": pd.DataFrame()}

        monkeypatch.setattr(CcxtBinanceProvider, "_yfinance_fallback",
                            staticmethod(fake))
        out = CcxtBinanceProvider().fetch(["BTC/USDT"], "2023-01-01", "2023-01-10")
        assert called["symbols"] == ["BTC/USDT"]
        assert called["start"] == "2023-01-01" and called["end"] == "2023-01-10"
        assert "close" in out

    def test_an_all_empty_ccxt_result_falls_back_too(self, monkeypatch):
        """
        `if not frames: return self._yfinance_fallback(...)`
        —— `not` 被删会在**正常拿到数据**时反而去打 yfinance，
        且把已取到的 ccxt 数据丢掉。
        """
        called = {}
        monkeypatch.setattr(CcxtBinanceProvider, "_yfinance_fallback",
                            staticmethod(lambda s, a, b: called.setdefault("hit", True)
                                         or {"close": pd.DataFrame()}))
        _panel_from(monkeypatch, {"BTC/USDT": range(8)})
        assert "hit" not in called, (
            "ccxt 明明取到了数据却走了兜底 —— `if not frames` 的 not 被删了")

        _install_ccxt(monkeypatch, _FakeExchange(candles=[]))
        CcxtBinanceProvider(delay_s=0.0).fetch(["BTC/USDT"], "2023-01-01",
                                               "2023-01-10")
        assert called.get("hit") is True, "ccxt 一条数据都没有却没有走兜底"

    def test_symbols_are_translated_to_yfinance_tickers(self, monkeypatch):
        """
        `s.replace("/USDT", "-USD").replace("/BTC", "-BTC")`

        两个替换任一被改，yfinance 会拿到 "BTC/USDT" 这种它不认识的
        代码，返回空表 —— crypto 数据集直接变空，且不报错。
        """
        seen = {}

        def fake_download(tickers=None, **k):
            seen["tickers"] = list(tickers)
            seen.update(k)
            return _yf_frame([t.upper() for t in tickers])

        monkeypatch.setitem(sys.modules, "yfinance",
                            types.SimpleNamespace(download=fake_download))
        CcxtBinanceProvider._yfinance_fallback(
            ["BTC/USDT", "ETH/BTC"], "2023-01-01", "2023-01-10")
        assert seen["tickers"] == ["BTC-USD", "ETH-BTC"], (
            f"代码转换结果是 {seen['tickers']} —— 替换规则被改了")

    def test_the_fallback_downloads_adjusted_prices_without_a_progress_bar(self,
                                                                           monkeypatch):
        seen = {}

        def fake_download(tickers=None, **k):
            seen.update(k)
            return _yf_frame([t.upper() for t in tickers])

        monkeypatch.setitem(sys.modules, "yfinance",
                            types.SimpleNamespace(download=fake_download))
        CcxtBinanceProvider._yfinance_fallback(["BTC/USDT"], "2023-01-01",
                                               "2023-01-10")
        assert seen["auto_adjust"] is True
        assert seen["progress"] is False
        assert seen["group_by"] == "column"
        assert seen["start"] == "2023-01-01" and seen["end"] == "2023-01-10"

    def test_the_panel_columns_are_mapped_back_to_ccxt_symbols(self, monkeypatch):
        """
        `ticker_map.get(c, c)` —— 映射被去掉会让面板的列名是
        "BTC-USD"，而调用方按 "BTC/USDT" 去取，得到 KeyError 或全 NaN。
        """
        monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(
            download=lambda tickers=None, **k: _yf_frame(
                [t.upper() for t in tickers])))
        out = CcxtBinanceProvider._yfinance_fallback(
            ["BTC/USDT", "ETH/USDT"], "2023-01-01", "2023-01-10")
        assert list(out["close"].columns) == ["BTC/USDT", "ETH/USDT"], (
            f"兜底面板的列名是 {list(out['close'].columns)} —— 没有映射回 ccxt 代码")

    def test_an_empty_download_returns_an_empty_dict(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(
            download=lambda **k: pd.DataFrame()))
        assert CcxtBinanceProvider._yfinance_fallback(
            ["BTC/USDT"], "2023-01-01", "2023-01-10") == {}

    def test_a_single_ticker_flat_column_frame_is_normalised(self, monkeypatch):
        """
        单币种时 yfinance 可能返回**扁平列**。
        `else:` 分支把它补成 MultiIndex —— 分支被删会让
        `raw["close"]` 取到一列 Series，后面 `.columns` 直接 AttributeError。
        """
        monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(
            download=lambda **k: _yf_frame(["BTC-USD"], multi=False)))
        out = CcxtBinanceProvider._yfinance_fallback(["BTC/USDT"], "2023-01-01",
                                                     "2023-01-10")
        assert list(out["close"].columns) == ["BTC/USDT"]
        assert out["close"].notna().all().all()

    def test_a_missing_field_becomes_an_all_nan_frame(self, monkeypatch):
        """
        `except KeyError: return DataFrame(index=..., columns=symbols)`
        —— yfinance 有时不给 Volume。这个兜底被删会让整次加载抛 KeyError。
        """
        frame = _yf_frame(["BTC-USD"])
        frame = frame.drop(columns=[c for c in frame.columns if c[0] == "Volume"])
        monkeypatch.setitem(sys.modules, "yfinance",
                            types.SimpleNamespace(download=lambda **k: frame))
        out = CcxtBinanceProvider._yfinance_fallback(["BTC/USDT"], "2023-01-01",
                                                     "2023-01-10")
        assert list(out["volume"].columns) == ["BTC/USDT"]
        assert out["volume"].isna().all().all()
        assert out["close"].notna().all().all(), "缺一个字段把其他字段也带坏了"

    def test_the_fallback_produces_the_same_field_set_as_ccxt(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(
            download=lambda tickers=None, **k: _yf_frame(
                [t.upper() for t in tickers])))
        out = CcxtBinanceProvider._yfinance_fallback(["BTC/USDT"], "2023-01-01",
                                                     "2023-01-10")
        assert set(out) == set(FIELDS), (
            f"兜底面板的字段集 {sorted(out)} 与主路径不一致 —— "
            f"切换数据源会让下游 DSL 少字段")

    def test_the_fallback_vwap_and_returns_use_the_same_formulas(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(
            download=lambda tickers=None, **k: _yf_frame(
                [t.upper() for t in tickers])))
        out = CcxtBinanceProvider._yfinance_fallback(["BTC/USDT"], "2023-01-01",
                                                     "2023-01-10")
        ref_vwap = (out["high"] + out["low"] + out["close"]) / 3.0
        assert np.allclose(out["vwap"].values, ref_vwap.values, rtol=1e-12)
        ref_ret = np.log(out["close"] / out["close"].shift(1))
        assert np.allclose(out["returns"].values, ref_ret.values, rtol=1e-12,
                           equal_nan=True)


# ===========================================================================
# N. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "L94 `pd.to_datetime(df['ts'], unit='ms', utc=True).dt.tz_localize(None)` 的 True → False":
        "这一行的输入 `df['ts']` 恒为**整数毫秒 epoch**（ccxt 的 OHLCV "
        "第 0 列规范如此，且上游 `_fetch_all_candles` 拿它与 "
        "`start_ms`/`end_ms` 做整数比较）。对整数 epoch 而言，"
        "`unit='ms'` 的换算基准**本来就是 UTC**："
        "`utc=True` 得到带 UTC 时区的时间戳，`utc=False` 得到"
        "同一时刻的 naive 时间戳，两者的墙钟读数逐位相同。"
        "紧接着的 `.dt.tz_localize(None)` 在前者上剥掉时区（保留墙钟），"
        "在后者上是空操作，于是最终的 naive 索引完全一致。"
        "（`utc` 只在输入是**带偏移量的字符串**或混合时区时才有区别，"
        "而这条路径上不可能出现。）"
        "机械验证见 test_the_utc_flag_cannot_change_an_integer_epoch_conversion。",
}


def test_the_utc_flag_cannot_change_an_integer_epoch_conversion():
    """
    L94 等价性的机械验证：对整数毫秒输入，两种 `utc` 取值
    经过同一条 `.dt.tz_localize(None)` 之后逐位相同。

    覆盖夏令时切换、跨年、负 epoch（1970 之前）等容易出岔子的时刻。
    这条一旦变红（例如 pandas 改了 naive 序列上 `tz_localize(None)`
    的语义），上面那份等价性证明立即作废。
    """
    stamps = [
        0,                                        # 1970-01-01
        -86_400_000,                              # 1969-12-31
        _ms("2023-03-12"),                        # 美国夏令时切换日
        _ms("2023-11-05"),                        # 美国冬令时切换日
        _ms("2023-12-31"), _ms("2024-01-01"),     # 跨年
        _ms("2024-02-29"),                        # 闰日
        1_700_000_000_123,                        # 带毫秒余数
    ]
    s = pd.Series(stamps, dtype="int64")

    with_utc = pd.to_datetime(s, unit="ms", utc=True).dt.tz_localize(None)
    without_utc = pd.to_datetime(s, unit="ms", utc=False).dt.tz_localize(None)

    assert list(with_utc) == list(without_utc), (
        f"整数 epoch 在两种 utc 取值下给出了不同的时间戳：\n"
        f"  utc=True  → {list(with_utc)}\n"
        f"  utc=False → {list(without_utc)}\n"
        f"L94 的等价性证明作废")
    assert with_utc.dt.tz is None and without_utc.dt.tz is None


def test_the_timestamp_column_really_is_an_integer_epoch():
    """
    等价性证明的前提：喂给 `pd.to_datetime` 的 `ts` 列确实是整数毫秒，
    而不是字符串或带时区的对象。前提没了，上面的论证也就不成立。
    """
    p = CcxtBinanceProvider(delay_s=0.0)
    cs = [_candle(_ms("2023-01-02")), _candle(_ms("2023-01-03"))]
    ex = _FakeExchange(candles=cs)
    got = p._fetch_all_candles(ex, "BTC/USDT", cs[0][0], cs[-1][0])
    assert got, "前提验证拿不到 K 线"
    for row in got:
        assert isinstance(row[0], int), (
            f"K 线时间戳不是整数而是 {type(row[0]).__name__} —— "
            f"L94 的等价性证明作废")
        assert row[0] > 1_000_000_000_000, "时间戳不在毫秒量级"


def test_every_survivor_has_a_written_proof():
    """首测 12 点 / 存活 1：唯一一处为等价变异。"""
    assert len(PROVEN_EQUIVALENT) == 1
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
