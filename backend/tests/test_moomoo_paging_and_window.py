"""
data_engine/providers/moomoo_provider.py —— 分页/节流/增量窗口的定钉测试（变异测试驱动）

来由：13 个变异点，首测击杀率 **7.7%**（存活 12）—— B 档次差。

既有的 test_phase_tr2_moomoo 只测了两件离线的事（ticker 映射、assemble_panel
结构），真正拉数据的那条路径 —— `fetch` / `fetch_latest` / `_fetch_all_klines` ——
**整条没有测试**，因为它需要 OpenD 网关。于是：

  - `while True` 的分页循环、`if not page_key: break` 的终止条件
  - `if data is not None and len(data) > 0` 的追加守卫
  - `if self.throttle_s and i < len(tickers) - 1` 的节流边界
  - `start = end - pd.Timedelta(days=max(7, n_recent * 3))` 的增量窗口
  - `if not dataset or ...empty` 的空数据拒绝

全部是盲区。这些错了的后果都很实在：分页终止条件反了会**死循环**或
**只拿第一页**（历史数据静默截断）；增量窗口的 `-` 变 `+` 会去
**拉未来日期**；空数据守卫失效会让一个全 NaN 的面板一路流进 PIT。

本文件用打桩的 quote context 把这条路径整段跑起来，不需要网关。
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.providers.moomoo_provider import MoomooProvider


RET_OK = 0
RET_ERROR = -1


def _kline(ticker: str, dates, base: float = 100.0) -> pd.DataFrame:
    n = len(dates)
    close = base + np.arange(n, dtype=float)
    return pd.DataFrame({
        "code": [f"US.{ticker}"] * n,
        "time_key": [d.strftime("%Y-%m-%d 00:00:00") for d in dates],
        "open": close - 0.5, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": np.full(n, 1_000_000.0),
    })


class _FakeCtx:
    """
    打桩的 OpenQuoteContext：按 (code) 返回预置的分页序列。

    pages[code] = [(ret, data, next_page_key), ...]
    每调用一次 request_history_kline 就吐下一项。
    """

    def __init__(self, pages: dict):
        self.pages = {k: list(v) for k, v in pages.items()}
        self.calls: list = []
        self.closed = False

    def request_history_kline(self, code, start=None, end=None, ktype=None,
                              autype=None, max_count=None, page_req_key=None):
        self.calls.append({"code": code, "start": start, "end": end,
                           "page_req_key": page_req_key, "max_count": max_count})
        seq = self.pages.get(code)
        if not seq:
            return RET_ERROR, "no more pages", None
        return seq.pop(0)

    def close(self):
        self.closed = True


def _fake_moomoo():
    """最小的 moomoo 模块替身，只提供本路径用到的常量。"""
    mod = types.ModuleType("moomoo")
    mod.RET_OK = RET_OK
    mod.KLType = types.SimpleNamespace(K_DAY="K_DAY")
    mod.AuType = types.SimpleNamespace(QFQ="qfq", HFQ="hfq", NONE="none")
    mod.OpenQuoteContext = lambda host=None, port=None: _FakeCtx({})
    return mod


@pytest.fixture
def stub_sdk(monkeypatch):
    """把 moomoo SDK 换成替身；返回一个可设置 ctx 的钩子。"""
    mod = _fake_moomoo()
    monkeypatch.setitem(sys.modules, "moomoo", mod)
    box: dict = {}

    def _install(ctx: _FakeCtx):
        box["ctx"] = ctx
        monkeypatch.setattr(MoomooProvider, "_open_quote_ctx", lambda self: ctx)
        return ctx

    return _install


@pytest.fixture
def no_sleep(monkeypatch):
    """记录 time.sleep 的调用次数与时长，而不是真睡。"""
    import app.core.data_engine.providers.moomoo_provider as mod
    slept: list = []
    monkeypatch.setattr(mod.time, "sleep", lambda s: slept.append(s))
    return slept


DATES = pd.bdate_range("2024-01-02", periods=6)


# ===========================================================================
# A. 分页循环
# ===========================================================================

class TestPaging:

    def test_all_pages_are_concatenated(self, stub_sdk, no_sleep):
        """
        `if not page_key: break` —— 删掉 `not` 会让循环在**还有下一页时**就停，
        只拿到第一页；历史数据被静默截断，回测窗口莫名变短。
        """
        p1 = _kline("AAPL", DATES[:3], 100.0)
        p2 = _kline("AAPL", DATES[3:], 200.0)
        ctx = _FakeCtx({"US.AAPL": [(RET_OK, p1, "key2"), (RET_OK, p2, None)]})
        stub_sdk(ctx)
        out = MoomooProvider(throttle_s=0.0)._fetch_all_klines(
            ctx, sys.modules["moomoo"], ["AAPL"], "2024-01-01", "2024-01-31")
        assert len(out["AAPL"]) == 6, (
            f"只取到 {len(out['AAPL'])} 行，应为两页共 6 行 —— 分页提前终止")
        assert len(ctx.calls) == 2, f"应当请求两页，实际 {len(ctx.calls)} 次"

    def test_the_page_key_is_passed_to_the_next_request(self, stub_sdk, no_sleep):
        ctx = _FakeCtx({"US.AAPL": [(RET_OK, _kline("AAPL", DATES[:3]), "key2"),
                                    (RET_OK, _kline("AAPL", DATES[3:]), None)]})
        stub_sdk(ctx)
        MoomooProvider(throttle_s=0.0)._fetch_all_klines(
            ctx, sys.modules["moomoo"], ["AAPL"], "2024-01-01", "2024-01-31")
        assert ctx.calls[0]["page_req_key"] is None, "第一页不该带 page_req_key"
        assert ctx.calls[1]["page_req_key"] == "key2", (
            f"第二页没有带上一页返回的 key：{ctx.calls[1]['page_req_key']}")

    def test_a_single_page_stops_immediately(self, stub_sdk, no_sleep):
        """page_key 为 None 就必须停 —— 否则 `while True` 变成死循环。"""
        ctx = _FakeCtx({"US.AAPL": [(RET_OK, _kline("AAPL", DATES), None)]})
        stub_sdk(ctx)
        out = MoomooProvider(throttle_s=0.0)._fetch_all_klines(
            ctx, sys.modules["moomoo"], ["AAPL"], "2024-01-01", "2024-01-31")
        assert len(ctx.calls) == 1, f"单页却请求了 {len(ctx.calls)} 次"
        assert len(out["AAPL"]) == len(DATES)

    def test_an_error_return_stops_that_ticker_without_raising(self, stub_sdk, no_sleep):
        """`if ret != RET_OK: break` —— 一个标的失败不得拖垮整批。"""
        ctx = _FakeCtx({"US.AAPL": [(RET_ERROR, "quota exceeded", None)],
                        "US.MSFT": [(RET_OK, _kline("MSFT", DATES, 300.0), None)]})
        stub_sdk(ctx)
        out = MoomooProvider(throttle_s=0.0)._fetch_all_klines(
            ctx, sys.modules["moomoo"], ["AAPL", "MSFT"], "2024-01-01", "2024-01-31")
        assert "AAPL" not in out, "失败的标的不该出现在结果里"
        assert "MSFT" in out, "一个标的失败把其余标的也拖掉了"

    def test_empty_pages_are_not_appended(self, stub_sdk, no_sleep):
        """
        `if data is not None and len(data) > 0` —— 三处变异：
        删 `not`（变成 `data is None` 时追加 → 把 None 塞进 concat）、
        `and`→`or`（data 为 None 时去算 len(None) → TypeError）、
        `> 0`→`>= 0`（空 DataFrame 也被追加 → concat 出全 NaN 列）。
        """
        empty = _kline("AAPL", DATES[:0])
        ctx = _FakeCtx({"US.AAPL": [(RET_OK, empty, "k2"),
                                    (RET_OK, None, "k3"),
                                    (RET_OK, _kline("AAPL", DATES[:2]), None)]})
        stub_sdk(ctx)
        out = MoomooProvider(throttle_s=0.0)._fetch_all_klines(
            ctx, sys.modules["moomoo"], ["AAPL"], "2024-01-01", "2024-01-31")
        assert len(out["AAPL"]) == 2, (
            f"空页/None 页被追加进了结果（{len(out['AAPL'])} 行，应为 2）")

    @pytest.mark.parametrize("payload,label", [
        (None, "None"),
        ("empty", "0 行的 DataFrame"),
    ])
    def test_a_ticker_with_no_rows_is_absent_from_the_result(self, stub_sdk,
                                                             no_sleep, payload, label):
        """
        `len(data) > 0` —— **严格大于**。放宽成 `>= 0` 时空 DataFrame 也会被
        追加，`if frames:` 随即为真，该标的以**零行**的形式进入结果。
        后续 assemble_panel 拿到一个有键无行的标的，拼出整列 NaN ——
        看起来"这只票那段时间没交易"，而事实是接口一条都没返回。

        只用 None 测不出来（`data is not None` 已经挡住了），
        必须用**空 DataFrame**；只测"多页里夹一个空页"也不行 ——
        concat 会把空页吃掉，行数不变（第一版就是这么漏的）。
        """
        data = None if payload is None else _kline("AAPL", DATES[:0])
        ctx = _FakeCtx({"US.AAPL": [(RET_OK, data, None)]})
        stub_sdk(ctx)
        out = MoomooProvider(throttle_s=0.0)._fetch_all_klines(
            ctx, sys.modules["moomoo"], ["AAPL"], "2024-01-01", "2024-01-31")
        assert out == {}, f"只返回了 {label} 的标的却进了结果：{out}"

    def test_concat_ignores_the_page_index(self, stub_sdk, no_sleep):
        """
        `pd.concat(frames, ignore_index=True)` —— 改成 False 会保留每页各自的
        0..n-1 索引，拼出来的表索引重复，后续 set_index/pivot 全乱。
        """
        ctx = _FakeCtx({"US.AAPL": [(RET_OK, _kline("AAPL", DATES[:3]), "k"),
                                    (RET_OK, _kline("AAPL", DATES[3:]), None)]})
        stub_sdk(ctx)
        out = MoomooProvider(throttle_s=0.0)._fetch_all_klines(
            ctx, sys.modules["moomoo"], ["AAPL"], "2024-01-01", "2024-01-31")
        assert list(out["AAPL"].index) == list(range(6)), (
            f"分页拼接后索引不连续：{list(out['AAPL'].index)} —— ignore_index 失效")


# ===========================================================================
# B. 节流
# ===========================================================================

class TestThrottle:

    def test_no_sleep_after_the_last_ticker(self, stub_sdk, no_sleep):
        """
        `if self.throttle_s and i < len(tickers) - 1` —— `<` 放宽成 `<=`
        会在**最后一个标的之后**也睡一次；`- 1` 改成 `+ 1` 会少睡一次。
        逐个标的都要睡时这看着无害，实际是"每批多等 0.4 秒"×几千批。
        """
        pages = {f"US.T{i}": [(RET_OK, _kline(f"T{i}", DATES), None)] for i in range(3)}
        ctx = _FakeCtx(pages)
        stub_sdk(ctx)
        MoomooProvider(throttle_s=0.4)._fetch_all_klines(
            ctx, sys.modules["moomoo"], ["T0", "T1", "T2"], "2024-01-01", "2024-01-31")
        assert no_sleep == [0.4, 0.4], (
            f"3 个标的应当睡 2 次（末个之后不睡），实际 {no_sleep}")

    def test_a_single_ticker_never_sleeps(self, stub_sdk, no_sleep):
        ctx = _FakeCtx({"US.AAPL": [(RET_OK, _kline("AAPL", DATES), None)]})
        stub_sdk(ctx)
        MoomooProvider(throttle_s=0.4)._fetch_all_klines(
            ctx, sys.modules["moomoo"], ["AAPL"], "2024-01-01", "2024-01-31")
        assert no_sleep == [], f"只有一个标的却睡了：{no_sleep}"

    def test_zero_throttle_disables_sleeping(self, stub_sdk, no_sleep):
        pages = {f"US.T{i}": [(RET_OK, _kline(f"T{i}", DATES), None)] for i in range(3)}
        ctx = _FakeCtx(pages)
        stub_sdk(ctx)
        MoomooProvider(throttle_s=0.0)._fetch_all_klines(
            ctx, sys.modules["moomoo"], ["T0", "T1", "T2"], "2024-01-01", "2024-01-31")
        assert no_sleep == [], f"throttle_s=0 却睡了：{no_sleep}"


# ===========================================================================
# C. fetch：空数据必须拒绝
# ===========================================================================

class TestFetchRejectsEmpty:

    def test_all_nan_panel_is_rejected(self, stub_sdk, no_sleep):
        """
        `if not dataset or next(iter(...)).dropna(how="all").empty: raise`
        —— 删掉 `not` 会让**有数据时**反而报错、没数据时静默通过，
        一个全 NaN 的面板会一路流进 PIT 并被当成"当天没交易"。
        """
        ctx = _FakeCtx({"US.AAPL": [(RET_ERROR, "no permission", None)]})
        stub_sdk(ctx)
        with pytest.raises(ValueError, match="空数据"):
            MoomooProvider(throttle_s=0.0).fetch(["AAPL"], "2024-01-01", "2024-01-31")

    def test_a_real_panel_is_returned(self, stub_sdk, no_sleep):
        ctx = _FakeCtx({"US.AAPL": [(RET_OK, _kline("AAPL", DATES), None)]})
        stub_sdk(ctx)
        out = MoomooProvider(throttle_s=0.0).fetch(["AAPL"], "2024-01-01", "2024-01-31")
        assert "close" in out and out["close"].shape[0] == len(DATES), (
            "正常数据被空数据守卫误杀了")

    def test_tickers_are_upcased_before_the_request(self, stub_sdk, no_sleep):
        ctx = _FakeCtx({"US.AAPL": [(RET_OK, _kline("AAPL", DATES), None)]})
        stub_sdk(ctx)
        MoomooProvider(throttle_s=0.0).fetch(["aapl"], "2024-01-01", "2024-01-31")
        assert ctx.calls[0]["code"] == "US.AAPL", (
            f"小写 ticker 没有被规范化：{ctx.calls[0]['code']}")

    def test_the_quote_context_is_closed_even_on_failure(self, stub_sdk, no_sleep):
        ctx = _FakeCtx({"US.AAPL": [(RET_ERROR, "boom", None)]})
        stub_sdk(ctx)
        with pytest.raises(ValueError):
            MoomooProvider(throttle_s=0.0).fetch(["AAPL"], "2024-01-01", "2024-01-31")
        assert ctx.closed, "拉取失败后没有关闭 quote context —— 连接泄漏"


# ===========================================================================
# D. fetch_latest 的增量窗口
# ===========================================================================

class TestIncrementWindow:

    @staticmethod
    def _capture(monkeypatch) -> dict:
        """拦下 fetch，记录它收到的 start/end。"""
        seen: dict = {}

        def _spy(self, tickers, start, end, fields=None):
            seen.update(tickers=tickers, start=start, end=end)
            return {"close": pd.DataFrame([[1.0]])}

        monkeypatch.setattr(MoomooProvider, "fetch", _spy)
        return seen

    def test_window_reaches_back_not_forward(self, monkeypatch):
        """
        `start = end - pd.Timedelta(days=...)` —— `-` 改成 `+` 会让起点
        **晚于终点**，请求一个空区间（或被网关判成非法），
        前向增量从此每天都拉不到数据，而 daily_ingest 只会记成"无新 bar"。
        """
        seen = self._capture(monkeypatch)
        MoomooProvider().fetch_latest(["AAPL"], n_recent=5)
        start, end = pd.Timestamp(seen["start"]), pd.Timestamp(seen["end"])
        assert start < end, f"增量窗口的起点不早于终点：{start} → {end}"

    def test_window_length_scales_with_n_recent(self):
        """
        `max(7, n_recent * 3)` —— `*` 改成 `/` 会让窗口在 n_recent 变大时
        反而**恒定为 7 天**，要 60 天增量时只回看一周，节假日一堵就断档。
        """
        import app.core.data_engine.providers.moomoo_provider as mod
        seen: dict = {}

        def _spy(self, tickers, start, end, fields=None):
            seen[len(seen)] = (pd.Timestamp(end) - pd.Timestamp(start)).days
            return {"close": pd.DataFrame([[1.0]])}

        orig = MoomooProvider.fetch
        try:
            MoomooProvider.fetch = _spy
            MoomooProvider().fetch_latest(["AAPL"], n_recent=5)
            MoomooProvider().fetch_latest(["AAPL"], n_recent=60)
        finally:
            MoomooProvider.fetch = orig
        assert seen[1] > seen[0], (
            f"n_recent=60 的窗口（{seen[1]} 天）没有比 n_recent=5（{seen[0]} 天）更长")
        assert seen[1] == 180, f"n_recent=60 应当回看 180 天，实际 {seen[1]}"

    def test_short_requests_still_get_a_seven_day_floor(self, monkeypatch):
        """`max(7, ...)` 的下限：n_recent=1 时仍要回看 7 天，覆盖长周末。"""
        seen = self._capture(monkeypatch)
        MoomooProvider().fetch_latest(["AAPL"], n_recent=1)
        span = (pd.Timestamp(seen["end"]) - pd.Timestamp(seen["start"])).days
        assert span == 7, f"n_recent=1 的回看窗口是 {span} 天，应为下限 7 天"
