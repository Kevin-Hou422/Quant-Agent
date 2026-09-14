"""
data_engine/base.py —— DataProvider 抽象基类

**此前零测试**（101 有效行，D 档）。

所有行情源（yahoo / moomoo / akshare / ccxt / 本地 parquet）都继承它。
基类里有两个**默认实现**，子类不覆盖就直接用：

  - `fetch_panel()`  wide → long → SchemaEnforcer，是"任何源都能给出
    标准长表"这个承诺的兜底实现
  - `metadata()`     provider 自描述，进诊断端点给人看

以及一个 `validate_fields()` 守卫 —— 它是"请求了这个源没有的字段"
唯一会 fail-loud 的地方。少了它，缺失字段会一路变成 NaN 列流进回测。

抽象基类的测试要靠**最小具体子类**来驱动，不能直接实例化。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.base import DataProvider
from app.core.data_engine.schema import STANDARD_COLUMNS


class _Stub(DataProvider):
    """最小可用子类：只实现两个抽象方法。"""

    FIELDS = ["close", "volume"]

    def __init__(self, raw=None, fields=None, delay_s: float = 0.0):
        self._raw = raw
        self._fields = fields if fields is not None else list(self.FIELDS)
        self._delay_s = delay_s
        self.fetch_calls: list = []

    def fetch(self, tickers, start, end, fields=None):
        self.fetch_calls.append(dict(tickers=tickers, start=start,
                                     end=end, fields=fields))
        if self._delay_s:
            # 延迟用例专用：Windows 上 `time.monotonic()` 的分辨率约 15.6 ms，
            # 瞬时返回的 fetch 会让实测耗时**恰好是 0.0**，
            # 于是"记录的延迟 > 0"这条断言随机变红（实测踩到过）。
            # 故意慢一点，让两侧都落在时钟分辨率之上。
            import time as _t
            _t.sleep(self._delay_s)
        if self._raw is not None:
            return self._raw
        idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
        n = len(tickers)
        # 按传入的 tickers 数量生成，避免"固定两列 vs 请求一只票"的形状不符
        close = np.array([[100.0 * (i + 1) for i in range(n)],
                          [101.0 * (i + 1) for i in range(n)]])
        vol = np.array([[1e6 * (i + 1) for i in range(n)],
                        [1.1e6 * (i + 1) for i in range(n)]])
        return {
            "close":  pd.DataFrame(close, index=idx, columns=tickers),
            "volume": pd.DataFrame(vol, index=idx, columns=tickers),
        }

    def available_fields(self):
        return list(self._fields)


# ===========================================================================
# A. 抽象性本身
# ===========================================================================

class TestAbstractness:

    def test_the_base_class_cannot_be_instantiated(self):
        """
        `@abstractmethod` 装饰的 `fetch` / `available_fields`。
        装饰器被去掉，基类就能直接实例化，调用 `fetch` 返回 None，
        随后 `if not raw` 静默返回空表 —— 一个没实现的 provider
        会表现成"这个源今天没有数据"。
        """
        with pytest.raises(TypeError) as ei:
            DataProvider()                                   # type: ignore[abstract]
        assert "abstract" in str(ei.value).lower(), (
            f"基类居然能实例化，或报错不是抽象方法相关：{ei.value}")

    @pytest.mark.parametrize("missing", ["fetch", "available_fields"])
    def test_a_subclass_missing_either_abstract_method_cannot_instantiate(
            self, missing):
        """两个抽象方法各是一个独立的契约点，少任何一个都不该放行。"""
        ns = {"fetch": lambda self, *a, **k: {},
              "available_fields": lambda self: []}
        ns.pop(missing)
        Partial = type("Partial", (DataProvider,), ns)
        with pytest.raises(TypeError):
            Partial()                                        # type: ignore[abstract]

    def test_a_complete_subclass_instantiates(self):
        """反向：两个都实现了就必须能用 —— 否则抽象标记被加到了多余的方法上。"""
        assert isinstance(_Stub(), DataProvider)


# ===========================================================================
# B. validate_fields —— 唯一 fail-loud 的字段守卫
# ===========================================================================

class TestValidateFields:

    def test_an_unknown_field_is_refused_loudly(self):
        """
        `unknown = set(fields) - supported; if unknown: raise ValueError`

        删掉这道闸，请求一个该源没有的字段（比如向 yahoo 要 `bid`）
        会静默返回缺列的面板，下游补成 NaN 列 —— 回测照跑，
        只是那个字段的因子恒为 NaN，表现为"这个因子没信号"。
        """
        with pytest.raises(ValueError) as ei:
            _Stub().validate_fields(["close", "bid"])
        assert "bid" in str(ei.value), (
            f"报错没有指出是哪个字段不支持：{ei.value}")

    def test_the_error_lists_what_is_supported(self):
        """报错要给出支持列表，否则调用方只能去翻源码。"""
        with pytest.raises(ValueError) as ei:
            _Stub().validate_fields(["bid"])
        msg = str(ei.value)
        assert "close" in msg and "volume" in msg, (
            f"报错没有列出支持的字段：{msg}")

    def test_supported_fields_pass(self):
        """
        `if unknown:` 的另一侧 —— 全部受支持时不得抛，且返回 None。

        （契约写死成"返回 None"而不是只写"不抛"：`validate_fields` 的
        类型标注就是 `-> None`，哪天它改成返回布尔，调用方的
        `if not provider.validate_fields(...)` 会静默反过来。）
        """
        assert _Stub().validate_fields(["close", "volume"]) is None

    def test_an_empty_request_passes(self):
        """空集合减任何集合仍是空集 —— 不该触发。"""
        assert _Stub().validate_fields([]) is None

    def test_a_subset_passes(self):
        assert _Stub().validate_fields(["close"]) is None

    def test_the_check_uses_set_difference_not_equality(self):
        """
        `set(fields) - supported` —— 用**差集**而不是相等判断。
        写成 `set(fields) != supported` 会让"只请求一部分字段"也被拒。
        """
        stub = _Stub(fields=["close", "volume", "vwap"])
        assert stub.validate_fields(["close"]) is None, "子集请求被拒了"
        assert stub.validate_fields(["close", "vwap"]) is None
        with pytest.raises(ValueError):
            stub.validate_fields(["close", "bid"])        # 超集必须被拒


# ===========================================================================
# C. fetch_panel 的默认实现
# ===========================================================================

class TestFetchPanelDefault:

    def test_it_delegates_to_fetch_with_the_same_arguments(self):
        """
        `raw = self.fetch(tickers=tickers, start=start, end=end, fields=fields)`
        —— 参数必须原样透传。少传 `fields` 会让子类忽略字段过滤，
        每次都拉全量（免费源有速率限制，这是会被限流的）。
        """
        stub = _Stub()
        stub.fetch_panel(["AAPL", "MSFT"], "2024-01-01", "2024-01-31",
                         fields=["close"])
        assert len(stub.fetch_calls) == 1
        call = stub.fetch_calls[0]
        assert call["tickers"] == ["AAPL", "MSFT"]
        assert call["start"] == "2024-01-01" and call["end"] == "2024-01-31"
        assert call["fields"] == ["close"], (
            f"fields 没有透传给 fetch：{call['fields']}")

    def test_an_empty_fetch_returns_an_empty_frame(self):
        """
        `if not raw: return pd.DataFrame()`

        删掉 `not` 会反过来：**有数据**时返回空表（数据静默消失），
        没数据时把 `{}` 喂进 `wide_to_long` → 走到 `pd.concat([])`。
        前者是最糟的 —— 面板为空，回测报"没有数据"，
        而实际上 provider 明明拿到了。
        """
        assert _Stub(raw={}).fetch_panel(["AAPL"], "2024-01-01", "2024-01-31").empty

    def test_a_non_empty_fetch_produces_a_standard_panel(self):
        """`not raw` 的另一侧 —— 有数据时必须真的产出长表。"""
        out = _Stub().fetch_panel(["AAPL", "MSFT"], "2024-01-01", "2024-01-31")
        assert not out.empty, (
            "provider 拿到了数据，fetch_panel 却返回空表 —— "
            "`if not raw` 的 not 被删掉了")
        assert list(out.columns)[:len(STANDARD_COLUMNS)] == STANDARD_COLUMNS, (
            f"默认实现没有过 SchemaEnforcer：{list(out.columns)}")
        assert len(out) == 4, f"2 天 × 2 只票应当是 4 行，实际 {len(out)}"

    def test_the_panel_values_survive_the_conversion(self):
        """wide → long 最容易出的错是行列错位，逐格核对。"""
        out = _Stub().fetch_panel(["AAPL", "MSFT"], "2024-01-01", "2024-01-31")
        row = out[(out["ticker"] == "MSFT") &
                  (out["timestamp"] == pd.Timestamp("2024-01-03"))]
        assert len(row) == 1
        assert row["close"].iloc[0] == pytest.approx(202.0), (
            f"MSFT 2024-01-03 的收盘价是 {row['close'].iloc[0]}，应当是 101*2=202 —— "
            f"wide→long 行列错位")

    def test_extra_fields_are_preserved_by_the_default_enforcer(self):
        """
        `SchemaEnforcer(allow_extra=True)` —— 默认**保留**额外字段。
        翻成 False 会让某个源特有的字段（如 `turnover_rate`）被静默丢掉。
        """
        idx = pd.to_datetime(["2024-01-02"])
        raw = {
            "close": pd.DataFrame([[100.0]], index=idx, columns=["AAPL"]),
            "turnover_rate": pd.DataFrame([[0.01]], index=idx, columns=["AAPL"]),
        }
        out = _Stub(raw=raw).fetch_panel(["AAPL"], "2024-01-01", "2024-01-31")
        assert "turnover_rate" in out.columns, (
            f"额外字段被丢掉了：{list(out.columns)} —— "
            f"`SchemaEnforcer(allow_extra=True)` 被翻成了 False")


# ===========================================================================
# D. 延迟记录与 metadata
# ===========================================================================

class TestLatencyAndMetadata:

    def test_latency_is_recorded_after_a_fetch(self):
        """
        `elapsed_ms = (time.monotonic() - t0) * 1000; self._last_latency_ms = ...`

        `*` 改成 `/` 会让延迟小六个数量级（毫秒变成 1e-6 秒），
        诊断端点上所有 provider 都显示"0ms"，性能问题彻底不可见。
        `-` 改成 `+` 则会给出一个巨大的绝对时间戳。
        """
        import time as _t

        # 让 fetch 明确慢 50ms —— 远超 Windows 上 `time.monotonic()`
        # 约 15.6 ms 的分辨率，两侧才都是稳定的正数。
        stub = _Stub(delay_s=0.05)
        assert getattr(stub, "_last_latency_ms", None) is None

        t0 = _t.monotonic()
        stub.fetch_panel(["AAPL"], "2024-01-01", "2024-01-31")
        measured_ms = (_t.monotonic() - t0) * 1000

        lat = stub._last_latency_ms
        assert lat is not None, "fetch_panel 之后没有记录延迟"
        assert measured_ms >= 40.0, (
            f"实测耗时只有 {measured_ms:.3f} ms —— 桩的延迟没生效，本用例失去区分力")

        # 只断言"量级对得上"抓不到 `*1000` -> `/1000`：
        # 后者给出 ~5e-8，仍然满足 `>= 0`。必须与**实测墙钟**比。
        assert lat == pytest.approx(measured_ms, rel=0.5, abs=5.0), (
            f"记录的延迟是 {lat:.6f} ms，而实测约 {measured_ms:.3f} ms —— "
            f"`(time.monotonic() - t0) * 1000` 的算符被改了"
            f"（除法会让所有 provider 在诊断页上显示 0ms，性能问题彻底不可见）")
        assert lat > 1.0, (
            f"记录的延迟是 {lat:.6f} ms —— 一次 50ms 的抓取不可能低于 1ms，"
            f"秒与毫秒的换算方向反了")

    def test_metadata_reports_the_class_name(self):
        """
        `"name": self.__class__.__name__` —— 必须是**子类**名，
        写成 `DataProvider.__name__` 会让所有 provider 在诊断里同名。
        """
        assert _Stub().metadata()["name"] == "_Stub", (
            "metadata 报的不是具体子类名")

    def test_metadata_carries_the_recorded_latency(self):
        """
        `getattr(self, "_last_latency_ms", None)` —— 没拉过数据时是 None，
        拉过之后是真实值。默认值被改成 0 会让"从没调用过"
        与"调用了但很快"无法区分。
        """
        stub = _Stub()
        assert stub.metadata()["latency_ms"] is None, (
            "还没拉过数据就报了延迟 —— getattr 的默认值不是 None")
        stub.fetch_panel(["AAPL"], "2024-01-01", "2024-01-31")
        assert stub.metadata()["latency_ms"] is not None

    def test_metadata_reflects_the_subclass_field_list(self):
        """
        `"available_fields": self.available_fields()` —— 调的是子类实现。
        返回硬编码列表会让诊断页显示错误的能力。
        """
        stub = _Stub(fields=["close", "vwap", "bid"])
        assert stub.metadata()["available_fields"] == ["close", "vwap", "bid"]

    def test_metadata_has_every_documented_key(self):
        """docstring 承诺 name / latency_ms / rate_limit / available_fields。"""
        md = _Stub().metadata()
        for k in ("name", "latency_ms", "rate_limit", "available_fields"):
            assert k in md, f"metadata 缺键 {k}：{sorted(md)}"


# ===========================================================================
# E. 小工具
# ===========================================================================

class TestUtilities:

    @pytest.mark.parametrize("tickers,expect", [
        ([], 0),
        (["AAPL"], 1),
        (["AAPL", "MSFT", "NVDA"], 3),
    ])
    def test_universe_size_counts_the_tickers(self, tickers, expect):
        assert _Stub().universe_size(tickers) == expect

    def test_repr_shows_the_class_and_its_fields(self):
        """
        `f"{self.__class__.__name__}(fields={self.available_fields()})"`
        —— repr 进日志，是排查"哪个源给了什么字段"的第一手信息。
        """
        r = repr(_Stub(fields=["close"]))
        assert r.startswith("_Stub("), f"repr 没有以子类名开头：{r}"
        assert "close" in r, f"repr 里没有字段列表：{r}"
