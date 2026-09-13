"""
data_engine/dataset_registry.py —— 取数入口的定钉测试（变异测试驱动）

来由：16 个变异点，首测击杀率 **0.0%**（存活 16）—— B 档最差，**一个都没杀死**。

这是所有研究路径的取数入口：GP、回测、策略构建、chat 全走 `load_registry_dataset`。
既有的七个引用它的测试（test_phase7_ingest / test_phase8_pit / ...）都只是**打桩掉它**
来跑别的东西，没有一条测它本身。于是这十六处全裸：

  - `if name not in _SPECS` 的三处 `not` —— 取反之后**已知数据集被拒、未知的放行**
  - `use_cache` / `with_sector` / `health_check` 三个默认 True
    —— 任一变 False：缓存失效（每次重新拉全网）、行业字段消失（行业中性静默退化）、
       **健康门整个不跑**（坏数据直接进研究管线）
  - `check_dataset_health(ds, min_score=thr, warn_only=not fc)` 的 `not`
    —— 取反之后 fail-closed 配置变成 warn-only，这正是审计 #5 修过的那个 bug
  - `use_moomoo and spec.region == "US"` 的 `and`
    —— 放宽成 `or` 会让**加密货币**数据集也去走 moomoo（券商没有这些标的）
  - `YahooFinanceProvider(auto_adjust=True, progress=False)` 两个参数
    —— auto_adjust 关掉就是**不复权价**，所有历史回测的收益率全错
  - `if report.overall_score < min_score` —— 健康分恰好等于阈值时该不该拒

本文件把 provider 全部打桩，不联网。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.core.data_engine.dataset_registry as R


DATES = pd.bdate_range("2024-01-02", periods=40)


def _raw(universe) -> dict:
    """provider 层返回的 wide 面板（已是标准字段）。"""
    n = len(universe)
    close = pd.DataFrame(
        100 + np.arange(len(DATES) * n, dtype=float).reshape(len(DATES), n),
        index=DATES, columns=list(universe))
    return {"close": close, "open": close, "high": close * 1.01,
            "low": close * 0.99, "vwap": close,
            "volume": pd.DataFrame(1e6, index=DATES, columns=list(universe)),
            "returns": np.log(close / close.shift(1))}


@pytest.fixture(autouse=True)
def clean_cache():
    R.clear_registry_cache()
    yield
    R.clear_registry_cache()


@pytest.fixture
def spy(monkeypatch):
    """把三个 provider 打桩，记录谁被调用过。"""
    calls: dict = {"yahoo": 0, "moomoo": 0, "ccxt": 0, "akshare": 0,
                   "yahoo_kwargs": None}

    def _mk(kind):
        def _f(tickers, start, end):
            calls[kind] += 1
            return _raw(tickers)
        return _f

    monkeypatch.setattr(R, "_fetch_moomoo", _mk("moomoo"))
    monkeypatch.setattr(R, "_fetch_akshare", _mk("akshare"))
    monkeypatch.setattr(R, "_fetch_ccxt", _mk("ccxt"))

    # yfinance 走真实的 _fetch_yfinance，只把 Provider 换掉 —— 这样才测得到
    # 它传给 YahooFinanceProvider 的构造参数。
    import app.core.data_engine.yahoo_provider as yp

    class _FakeYahoo:
        def __init__(self, **kw):
            calls["yahoo_kwargs"] = kw

        def fetch(self, tickers, start=None, end=None):
            calls["yahoo"] += 1
            return _raw(tickers)

    monkeypatch.setattr(yp, "YahooFinanceProvider", _FakeYahoo)

    # 行业矩阵：避免联网查 GICS
    import app.core.data_engine.sector_mapper as sm
    monkeypatch.setattr(sm, "build_sector_matrix",
                        lambda universe, dates: pd.DataFrame(
                            1, index=dates, columns=list(universe)))
    monkeypatch.setattr(sm, "coverage_report",
                        lambda universe: {"unmapped": 0, "total": len(universe),
                                          "sector_distribution": {1: len(universe)},
                                          "unmapped_tickers": []})
    return calls


# ===========================================================================
# A. 名称校验
# ===========================================================================

class TestNameValidation:

    def test_unknown_name_raises_and_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="Unknown dataset"):
            R.load_registry_dataset("no_such_dataset")

    def test_known_name_loads(self, spy):
        ds = R.load_registry_dataset("us_tech_large")
        assert ds.name == "us_tech_large" and "close" in ds.data, (
            "已知数据集被当成未知拒绝了 —— `name not in _SPECS` 疑似被取反")

    def test_shelved_name_gets_its_own_message(self):
        with pytest.raises(KeyError, match="已搁置"):
            R.load_registry_dataset("china_tech")

    def test_registry_spec_rejects_unknown_names(self):
        with pytest.raises(KeyError):
            R.registry_spec("no_such_dataset")
        assert R.registry_spec("us_tech_large").region == "US"

    def test_load_multiple_validates_every_name_before_loading(self, spy):
        """
        `unknown = [n for n in names if n not in _SPECS]` —— 取反之后
        **合法名字全被当成未知**，批量加载完全不可用；
        更糟的情况是未知名字放行，然后在线程池里才炸，部分数据集已经拉完。
        """
        with pytest.raises(KeyError, match="未知数据集"):
            R.load_multi_datasets(["us_tech_large", "no_such_dataset"])
        assert spy["yahoo"] == 0, "校验之前就已经开始拉数据了"

    def test_load_multiple_loads_all_valid_names(self, spy):
        out = R.load_multi_datasets(["us_tech_large", "us_energy"])
        assert set(out) == {"us_tech_large", "us_energy"}
        assert all("sector" in ds.data for ds in out.values()), (
            "批量加载的默认 with_sector 不再是 True")

    def test_load_multiple_caches_by_default(self, spy):
        """
        `use_cache: bool = True`（`load_multi_datasets` 的签名，与
        `load_registry_dataset` 是**两份独立的默认值**）。改成 False 会让
        批量加载每次都重新拉全网 —— 而批量加载正是 GP/多数据集回测的入口，
        一轮进化下来会把同一份数据拉上几十遍。
        """
        R.load_multi_datasets(["us_tech_large"])
        R.load_multi_datasets(["us_tech_large"])
        assert spy["yahoo"] == 1, (
            f"批量加载同一数据集拉了 {spy['yahoo']} 次 —— 默认缓存没生效")

    def test_load_multiple_honours_use_cache_false(self, spy):
        R.load_multi_datasets(["us_tech_large"])
        R.load_multi_datasets(["us_tech_large"], use_cache=False)
        assert spy["yahoo"] == 2, "批量加载的 use_cache=False 没有绕过缓存"


# ===========================================================================
# B. 三个默认开关
# ===========================================================================

class TestDefaultSwitches:

    def test_cache_is_on_by_default(self, spy):
        R.load_registry_dataset("us_tech_large")
        R.load_registry_dataset("us_tech_large")
        assert spy["yahoo"] == 1, (
            f"同一数据集拉了 {spy['yahoo']} 次 —— 默认缓存没生效，"
            f"每次研究调用都会重新拉全网")

    def test_use_cache_false_forces_a_refetch(self, spy):
        """
        `if use_cache and cache_key in _CACHE` —— `and` 放宽成 `or` 会让
        **显式要求绕过缓存**的调用照样吃到旧缓存：换了数据源、补了数据之后
        重新加载拿到的还是旧对象，而调用方以为自己拿到了新数据。
        """
        R.load_registry_dataset("us_tech_large")
        R.load_registry_dataset("us_tech_large", use_cache=False)
        assert spy["yahoo"] == 2, (
            f"use_cache=False 仍然命中了缓存（只拉了 {spy['yahoo']} 次）")

    def test_use_cache_false_also_does_not_poison_the_cache(self, spy):
        R.load_registry_dataset("us_tech_large", use_cache=False)
        R.load_registry_dataset("us_tech_large", use_cache=False)
        assert spy["yahoo"] == 2

    def test_sector_field_is_attached_by_default(self, spy):
        ds = R.load_registry_dataset("us_tech_large")
        assert "sector" in ds.data, (
            "默认没有附加 sector 字段 —— group_rank / ind_neutralize 会静默"
            "退化成全截面算子，行业中性从此不生效且无人知晓")
        assert ds.data["sector"].shape == ds.data["close"].shape

    def test_sector_can_be_turned_off(self, spy):
        ds = R.load_registry_dataset("us_tech_large", with_sector=False)
        assert "sector" not in ds.data

    def test_health_check_runs_by_default(self, monkeypatch, spy):
        """
        `health_check: bool = True` 改成 False → **健康门整个不跑**。
        缺列、断档、跳点全部放行，而 ingest 那条路是真拒的，两条路口径分叉。
        """
        seen: dict = {}
        monkeypatch.setattr(R, "check_dataset_health",
                            lambda ds, min_score=0.7, warn_only=True:
                            seen.update(min_score=min_score, warn_only=warn_only))
        R.load_registry_dataset("us_tech_large")
        assert seen, "默认没有跑健康检查"

    def test_health_check_can_be_turned_off(self, monkeypatch, spy):
        seen: dict = {}
        monkeypatch.setattr(R, "check_dataset_health",
                            lambda ds, **kw: seen.update(kw))
        R.load_registry_dataset("us_tech_large", health_check=False)
        assert not seen, "health_check=False 仍然跑了健康检查"


# ===========================================================================
# C. fail-closed 与 warn_only 的取反
# ===========================================================================

class TestFailClosedWiring:

    @pytest.mark.parametrize("fail_closed,expected_warn_only", [
        (True, False),
        (False, True),
    ])
    def test_fail_closed_is_inverted_into_warn_only(self, monkeypatch, spy,
                                                    fail_closed, expected_warn_only):
        """
        `check_dataset_health(ds, min_score=thr, warn_only=not fc)` ——
        删掉 `not` 会让 fail-closed 的配置变成 warn-only（坏数据照样放行），
        这**正是审计 #5 修过的那个 bug**，没有测试就会原样复发。
        """
        seen: dict = {}
        monkeypatch.setattr(R, "check_dataset_health",
                            lambda ds, min_score=0.7, warn_only=True:
                            seen.update(min_score=min_score, warn_only=warn_only))
        R.load_registry_dataset("us_tech_large", health_fail_closed=fail_closed)
        assert seen["warn_only"] is expected_warn_only, (
            f"health_fail_closed={fail_closed} 应当传 warn_only={expected_warn_only}，"
            f"实际 {seen['warn_only']} —— fail-closed 的取反丢了")

    def test_min_health_override_is_passed_through(self, monkeypatch, spy):
        seen: dict = {}
        monkeypatch.setattr(R, "check_dataset_health",
                            lambda ds, min_score=0.7, warn_only=True:
                            seen.update(min_score=min_score))
        R.load_registry_dataset("us_tech_large", min_health=0.42)
        assert seen["min_score"] == 0.42

    def test_fail_closed_actually_propagates_the_error(self, monkeypatch, spy):
        """端到端：fail-closed 时健康检查抛错必须冒出来，不能被吞。"""
        def _boom(ds, min_score=0.7, warn_only=True):
            raise R.DatasetHealthError("健康分过低")
        monkeypatch.setattr(R, "check_dataset_health", _boom)
        with pytest.raises(R.DatasetHealthError):
            R.load_registry_dataset("us_tech_large", health_fail_closed=True)


# ===========================================================================
# D. provider 路由
# ===========================================================================

class TestProviderRouting:

    def test_us_datasets_use_yahoo_by_default(self, spy):
        R.load_registry_dataset("us_tech_large")
        assert spy["yahoo"] == 1 and spy["moomoo"] == 0

    def test_yahoo_is_constructed_with_adjusted_prices(self, spy):
        """
        `YahooFinanceProvider(auto_adjust=True, progress=False)`。
        `auto_adjust=False` 会拿到**未复权价**：拆股当天出现几十个点的假跳空，
        所有历史回测的收益率与波动率全错，而数据"看起来很正常"。
        `progress=True` 会往 stdout 打进度条，污染日志与 CI 输出。
        """
        R.load_registry_dataset("us_tech_large")
        kw = spy["yahoo_kwargs"]
        assert kw is not None, "没有走 YahooFinanceProvider"
        assert kw.get("auto_adjust") is True, (
            f"auto_adjust={kw.get('auto_adjust')} —— 取到的是未复权价")
        assert kw.get("progress") is False, (
            f"progress={kw.get('progress')} —— 进度条会污染日志")

    def test_moomoo_is_used_only_for_us_specs(self, monkeypatch, spy):
        """
        `if use_moomoo and spec.region == "US"` —— `and` 放宽成 `or` 会让
        **加密货币**（region=Global）也去走 moomoo，而券商根本没有这些标的；
        或者反过来，在 price_source=yahoo 时也走 moomoo。
        """
        from app.config import settings
        monkeypatch.setattr(settings, "price_source", "moomoo", raising=False)

        R.load_registry_dataset("us_tech_large")
        assert spy["moomoo"] == 1 and spy["yahoo"] == 0, (
            "price_source=moomoo 时美股没有走 moomoo")

        R.load_registry_dataset("crypto_major")
        assert spy["ccxt"] == 1, (
            f"加密数据集没有走 ccxt（moomoo 调用数={spy['moomoo']}）—— "
            f"region 限制被放宽了")

    def test_yahoo_is_used_when_price_source_is_yahoo(self, monkeypatch, spy):
        from app.config import settings
        monkeypatch.setattr(settings, "price_source", "yahoo", raising=False)
        R.load_registry_dataset("us_tech_large")
        assert spy["yahoo"] == 1 and spy["moomoo"] == 0

    def test_unreadable_price_source_falls_back_to_yahoo(self, monkeypatch, spy):
        """
        读不到 price_source 时 `use_moomoo = False` —— 改成 True 会让**配置读失败**
        变成"默默改用 moomoo"，研究侧与执行侧的数据源在无人知晓的情况下分叉。
        """
        class _Boom:
            def __getattr__(self, name):
                raise RuntimeError("settings 不可用")

        import app.config
        monkeypatch.setattr(app.config, "settings", _Boom())
        # health_check=False：健康门那一步也读 settings，会先抛；这里只测取数路由。
        R.load_registry_dataset("us_tech_large", health_check=False)
        assert spy["yahoo"] == 1 and spy["moomoo"] == 0, (
            "配置读不到时改用了 moomoo —— 应当退回 yahoo")

    def test_crypto_uses_ccxt(self, spy):
        R.load_registry_dataset("crypto_major")
        assert spy["ccxt"] == 1 and spy["yahoo"] == 0


# ===========================================================================
# E. check_dataset_health 的阈值与默认
# ===========================================================================

class _Report:
    """
    健康报告替身。**通过**分支的那条 logger.info 会读 n_tickers / n_dates，
    而外层的 `except Exception` 会把 AttributeError 吞成 `return None` ——
    字段缺一个，测试看到的就是"返回了 None"，而不是真正的原因。
    """

    def __init__(self, score):
        self.overall_score = score
        self.gaps = []
        self.spikes = []
        self.n_tickers = 2
        self.n_dates = 4


class TestHealthThreshold:

    @staticmethod
    def _ds_with_score(monkeypatch, score: float):
        import app.core.data_engine.health_report as hr

        class _Checker:
            def check(self, df, date_col=None, ticker_col=None):
                return _Report(score)

        monkeypatch.setattr(hr, "DataHealthChecker", _Checker)
        return R.Dataset(name="t", frequency="daily", universe=["A", "B"],
                         data=_raw(["A", "B"]))

    def test_score_exactly_at_the_threshold_passes(self, monkeypatch):
        """
        `if report.overall_score < min_score` —— **严格小于**。
        放宽成 `<=` 会把恰好达标的数据集拒掉：阈值是"可接受的最低分"，
        踩线应当放行，否则谁也说不清 0.7 到底算不算合格。
        """
        ds = self._ds_with_score(monkeypatch, 0.70)
        report = R.check_dataset_health(ds, min_score=0.70, warn_only=False)
        assert report is not None and report.overall_score == 0.70, (
            "恰好达标的数据集没有正常返回报告 —— `<` 被放宽成了 `<=`")

    def test_score_just_below_the_threshold_raises_when_closed(self, monkeypatch):
        ds = self._ds_with_score(monkeypatch, 0.6999)
        with pytest.raises(R.DatasetHealthError, match="健康得分"):
            R.check_dataset_health(ds, min_score=0.70, warn_only=False)

    def test_warn_only_defaults_to_true(self, monkeypatch):
        """
        `warn_only: bool = True` 改成 False 会让**所有**没显式传参的调用
        变成 fail-closed —— 其中包括只想看看报告的诊断路径，一跑就抛。
        """
        ds = self._ds_with_score(monkeypatch, 0.1)
        report = R.check_dataset_health(ds, min_score=0.9)   # 不传 warn_only
        assert report is not None, "默认 warn_only 下应当返回报告而不是抛错"

    def test_empty_close_returns_none_without_raising(self, monkeypatch):
        empty = dict(_raw(["A", "B"]))
        empty["close"] = pd.DataFrame()
        ds = R.Dataset(name="t", frequency="daily", universe=["A", "B"], data=empty)
        assert R.check_dataset_health(ds, min_score=0.9, warn_only=False) is None
