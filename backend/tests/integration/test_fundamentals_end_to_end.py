"""
Phase 10 验收（路线图原文）：
    "一个基本面因子可解析 + 回测且**发布日测试无前视**；发现引擎能形成基本面假设。"

以及把基本面接进两条真实路径后必须成立的事：
  - 研究路径：load_registry_dataset 在开启时挂字段、关着时不挂、非美股永不挂、拿不到事实时报错；
  - 前向路径：价格 PIT 里不存基本面字段，喂给交易循环前现挂；挂不上时面板里没有这些字段
    （引用它们的因子报错），而不是一列 NaN。

截面需要十来只票，而真实夹具只有 AAPL / NVDA 两家。其余标的是这两家真实事实的**克隆**
（数值乘一个系数、披露日整体平移几天）—— 时点结构（披露日、重述、财年错位）是真的，
只有公司数量是造的。
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core.alpha_engine.dsl_executor import Executor
from app.core.alpha_engine.fields import FUNDAMENTAL_FIELDS
from app.core.data_engine.fundamentals import prices as fund_prices
from app.core.data_engine.fundamentals.panels import ALL_FIELDS, build_panels
from app.core.data_engine.fundamentals.service import FundamentalsUnavailable
from app.core.data_engine.fundamentals.store import FundamentalsStore
from tests.unit.data_engine import fundamentals_fixtures as fx

IDX = pd.bdate_range("2023-07-03", "2025-03-31")
TICKERS = [f"C{i:02d}" for i in range(10)]


def _clone(base: pd.DataFrame, ticker: str, i: int, shift_days: int) -> pd.DataFrame:
    """数值按 (标的, 科目) 各乘一个不同的系数 —— 否则克隆之间 ROE、毛利率全相等，截面全是并列。"""
    f = base.copy()
    f["ticker"] = ticker
    for k, concept in enumerate(sorted(f["concept"].unique())):
        if concept.startswith("shares"):
            continue
        m = f["concept"] == concept
        f.loc[m, "val"] = f.loc[m, "val"] * (0.4 + 0.15 * i) * (1.0 + 0.06 * ((i * (k + 2)) % 7))
    f["filed"] = f["filed"] + pd.Timedelta(days=shift_days)
    return f


@pytest.fixture(scope="module")
def universe():
    """10 只票：真实事实的克隆 + 合成价格。"""
    rng = np.random.default_rng(7)
    base = {"AAPL": fx.facts("AAPL"), "NVDA": fx.facts("NVDA")}
    base_ev = {"AAPL": fx.events("AAPL"), "NVDA": fx.events("NVDA")}
    facts, events = {}, {}
    for i, tk in enumerate(TICKERS):
        src = "AAPL" if i % 2 == 0 else "NVDA"
        shift = i % 3               # 披露日错开 0–2 天：同一天不会所有公司一起变
        facts[tk] = _clone(base[src], tk, i, shift_days=shift)
        ev = base_ev[src].copy()
        ev["ticker"], ev["filing_date"] = tk, ev["filing_date"] + pd.Timedelta(days=shift)
        events[tk] = ev
    ret = rng.normal(0.0004, 0.015, (len(IDX), len(TICKERS)))
    close = pd.DataFrame(150.0 * np.exp(np.cumsum(ret, axis=0)), index=IDX, columns=TICKERS)
    # 日内幅度用随机数（固定 ±1% 会让价差估计荒谬，见 DEV_LESSONS §O）
    up = np.abs(rng.normal(0, 0.006, close.shape))
    dn = np.abs(rng.normal(0, 0.006, close.shape))
    price = {"open": close * 0.999, "high": close * (1 + up), "low": close * (1 - dn), "close": close,
             "volume": pd.DataFrame(2e6, index=IDX, columns=TICKERS), "vwap": close,
             "returns": np.log(close / close.shift(1))}
    return facts, events, price


_BUILT: dict = {}


def _dataset(universe, cut=None):
    """cut：只保留该日及之前披露的事实（"当时的世界"）。按 cut 缓存，避免重复构建。"""
    if cut in _BUILT:
        return _BUILT[cut]
    facts, events, price = universe
    if cut is not None:
        ts = pd.Timestamp(cut)
        facts = {t: f[f["filed"] <= ts] for t, f in facts.items()}
        events = {t: e[e["filing_date"] <= ts] for t, e in events.items()}
    panels, _ = build_panels(IDX, TICKERS, facts, events, price["close"], {})
    _BUILT[cut] = {**price, **panels}
    return _BUILT[cut]


@pytest.fixture(scope="module")
def dataset(universe):
    return _dataset(universe)


@pytest.fixture(scope="module")
def metrics(universe):
    """全量事实的每披露日指标 —— 不验前视的测试用它省掉重复计算。"""
    from app.core.data_engine.fundamentals.panels import metric_events
    return {t: metric_events(f, since=IDX[0] - pd.Timedelta(days=400)) for t, f in universe[0].items()}


# ---------------------------------------------------------------------------
# 验收 1：基本面因子可解析 + 回测
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dsl", [
    "fund_rank(earnings_yield)",
    "fund_rank(roe) + fund_rank(gross_margin)",
    "fund_rank(trade_when(days_since_earnings < 60, sue))",
    "fund_rank(earnings_yield) + rank(ts_delta(log(close), 20))",
])
def test_fundamental_factor_parses_and_backtests(dataset, dsl):
    from app.core.alpha_engine.signal_processor import SimulationConfig
    from app.core.backtest_engine.realistic_backtester import RealisticBacktester
    res = RealisticBacktester(SimulationConfig()).run(dsl, dataset)
    sig = res.raw_signal
    assert sig.shape == (len(IDX), len(TICKERS))
    assert sig.iloc[-1].notna().sum() == len(TICKERS)
    assert np.isfinite(res.is_report.sharpe_ratio)
    assert res.is_result is not None


def test_every_seed_of_the_fundamental_families_runs_on_a_fundamentals_dataset(dataset):
    from app.core.gp_engine.gp_engine import FUNDAMENTAL_FAMILIES, get_seeds_for_family
    ex = Executor()
    for fam in FUNDAMENTAL_FAMILIES:
        for dsl in get_seeds_for_family(fam):
            out = ex.run_expr(dsl, dataset)
            assert out.iloc[-1].notna().all(), dsl
            assert out.iloc[-1].nunique() > 1, dsl


# ---------------------------------------------------------------------------
# 验收 2：发布日无前视（在最终信号上验，不只是在字段上）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dsl", [
    "fund_rank(earnings_yield)",
    "fund_rank(roe) - fund_rank(asset_growth)",
    "fund_rank(sue) + fund_rank(ts_delta(sue, 63))",
    "fund_rank(trade_when(days_since_earnings < 60, sue))",
])
@pytest.mark.parametrize("cut", ["2023-11-03", "2024-05-29", "2024-08-30"])
def test_signal_up_to_a_date_is_unchanged_by_anything_filed_after_it(universe, dataset, dsl, cut):
    """
    只用 cut 当天及之前披露的事实重建数据集，算出的信号在 cut 及之前必须与"全知"版本逐位相同。
    任何把未来披露提前用上的实现（按期末日对齐、披露当天即可见、ffill 方向反了）都过不了。
    """
    ex = Executor()
    full = ex.run_expr(dsl, dataset)
    past = ex.run_expr(dsl, _dataset(universe, cut=cut))
    pd.testing.assert_frame_equal(past.loc[:cut], full.loc[:cut], check_exact=True)


def test_the_lookahead_check_catches_alignment_by_period_end(universe):
    """
    上一条测试的判别力：把"可见日"换成**期末日**（yfinance `.quarterly_financials` 只给得出这个），
    同样的前缀检查会红 —— 2024-06 季度的数在 7 月中旬就已经出现在面板里，而它 8 月才披露。
    """
    facts, events, price = universe
    cut = pd.Timestamp("2024-07-15")

    def leaky(known_only: bool):
        out = {}
        for t in TICKERS[:2]:
            f = facts[t]
            g = f[f["filed"] <= cut] if known_only else f
            g = g[g["end"] >= "2022-01-01"].copy()
            g["filed"] = g["end"]                      # 假装期末当天就知道了
            out[t] = g
        return build_panels(IDX, TICKERS[:2], out, {}, price["close"], {})[0]

    full, past = leaky(False), leaky(True)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(past["roe"].loc[:cut], full["roe"].loc[:cut], check_exact=True)


def test_fields_are_nan_before_any_filing_is_visible(universe):
    facts, events, price = universe
    some = TICKERS[:3]
    late = {t: facts[t][facts[t]["filed"] >= "2024-01-01"] for t in some}
    px = {k: v[some] for k, v in price.items()}
    panels, _ = build_panels(IDX, some, late, {}, px["close"], {})
    for name in ("roe", "earnings_yield", "sue", "gross_margin"):
        assert panels[name].loc[:"2023-12-29"].isna().all().all(), name
    for name in ("roe", "earnings_yield", "gross_margin"):      # SUE 要 11 个季度的历史，这里凑不齐
        assert panels[name].iloc[-1].notna().all(), name
    sig = Executor().run_expr("fund_rank(roe)", {**px, **panels})
    assert sig.loc[:"2023-12-29"].isna().all().all() and sig.iloc[-1].notna().all()


# ---------------------------------------------------------------------------
# 验收 3：发现引擎能形成基本面假设
# ---------------------------------------------------------------------------

def test_observer_proposes_fundamental_directions_only_when_the_data_is_there(dataset, universe):
    from app.core.discovery.market_observer import MarketObserver
    _, _, price = universe
    with_fund = {h.family: h for h in MarketObserver().observe(dataset).hypotheses}
    price_only = [h.family for h in MarketObserver().observe(price).hypotheses]
    assert {"value", "profitability", "earnings"} <= set(with_fund)
    assert not {"value", "profitability", "earnings"} & set(price_only)
    assert len(price_only) == 6
    for fam in ("value", "profitability", "earnings"):
        assert 0.0 < with_fund[fam].score <= 1.0 and with_fund[fam].rationale


def test_earnings_direction_ranks_first_in_reporting_season(universe, metrics):
    """末日恰在财报季（多数标的刚公告）→ earnings 排到最前；淡季则排不上来。"""
    from app.core.discovery.market_observer import MarketObserver
    facts, events, price = universe

    def observe(end):
        idx = IDX[IDX <= end]
        panels, _ = build_panels(idx, TICKERS, facts, events, price["close"].loc[idx], {}, metrics)
        ds = {**{k: v.loc[idx] for k, v in price.items()}, **panels}
        frac = float((panels["days_since_earnings"].iloc[-1] <= 20).mean())
        return MarketObserver().observe(ds), frac

    busy, frac_busy = observe(pd.Timestamp("2024-11-27"))
    quiet, frac_quiet = observe(pd.Timestamp("2024-10-15"))
    assert frac_busy == 1.0 and frac_quiet == 0.0
    score = lambda obs: {h.family: h for h in obs.hypotheses}["earnings"]     # noqa: E731
    assert score(busy).score > score(quiet).score
    assert busy.top_families(1) == ["earnings"]
    assert "earnings" not in quiet.top_families(3)
    assert "100%" in score(busy).rationale and "0%" in score(quiet).rationale


def test_low_coverage_fields_do_not_produce_a_direction(universe, metrics):
    from app.core.discovery.market_observer import MIN_FUND_COVERAGE, MarketObserver
    facts, events, price = universe
    few = {t: facts[t] for t in TICKERS[:4]}                     # 4/10 < 0.5
    panels, _ = build_panels(IDX, TICKERS, few, events, price["close"], {}, metrics)
    fams = [h.family for h in MarketObserver().observe({**price, **panels}).hypotheses]
    assert MIN_FUND_COVERAGE == 0.5 and "value" not in fams and "profitability" not in fams
    half = {t: facts[t] for t in TICKERS[:5]}                    # 5/10 = 0.5 → 恰好达标
    panels, _ = build_panels(IDX, TICKERS, half, events, price["close"], {}, metrics)
    assert "value" in [h.family for h in MarketObserver().observe({**price, **panels}).hypotheses]


def test_seed_generation_uses_fundamental_templates_when_fields_exist(dataset, monkeypatch):
    from app.core.workflows import alpha_workflows as wf
    monkeypatch.setattr("app.agent.alpha_agent.AlphaAgent._generate_dsls", lambda self, h: [])
    for hyp, field in (("value", "earnings_yield"), ("profitability", "roe"), ("earnings", "sue"),
                       ("低估值", "earnings_yield"), ("财报超预期", "sue")):
        seeds = wf._generate_diverse_seeds(hyp, n_target=8, seed=3,
                                           available_fields=wf._computable_fields(dataset))
        assert any(field in s for s in seeds), (hyp, seeds)


def test_seed_generation_drops_fundamental_templates_on_a_price_only_dataset(universe, monkeypatch, caplog):
    from app.core.workflows import alpha_workflows as wf
    _, _, price = universe
    monkeypatch.setattr("app.agent.alpha_agent.AlphaAgent._generate_dsls",
                        lambda self, h: ["fund_rank(roe)", "rank(made_up_field)"])
    with caplog.at_level(logging.WARNING):
        seeds = wf._generate_diverse_seeds("value", n_target=8, seed=3,
                                           available_fields=wf._computable_fields(price))
    assert len(seeds) >= 5
    assert not any(f in s for s in seeds for f in FUNDAMENTAL_FIELDS | {"made_up_field"})
    assert fx.logged(caplog, "引用了数据集没有的字段")


def test_computable_fields_include_what_the_executor_derives():
    from app.core.workflows.alpha_workflows import _computable_fields
    assert _computable_fields({"close": 1}) == {"close", "returns"}
    assert _computable_fields({"close": 1, "high": 1, "low": 1}) == {"close", "high", "low", "returns", "vwap"}
    assert _computable_fields({"volume": 1}) == {"volume"}


def test_discovery_explores_a_fundamental_family_and_saves_a_fundamental_candidate(universe, metrics, tmp_path, monkeypatch):
    """财报季的数据集 → 观察引擎把 earnings 排第一 → GP 从基本面种子出发 → 候选因子引用基本面字段。"""
    from app.core.discovery.discovery_engine import DiscoveryEngine
    from app.db.alpha_store import AlphaStore
    facts, events, price = universe
    idx = IDX[IDX <= pd.Timestamp("2024-11-27")]
    panels, _ = build_panels(idx, TICKERS, facts, events, price["close"].loc[idx], {}, metrics)
    ds = {**{k: v.loc[idx] for k, v in price.items()}, **panels}
    monkeypatch.setattr("app.agent.alpha_agent.AlphaAgent._generate_dsls", lambda self, h: [])
    # 外部审查 2026-10-18 F03：观察引擎只看选择段，不再看数据尾部（冻结 Test 段）。这条用例要的是
    # "财报季在观察窗口里" —— 把冻结段压到末尾几天，让 11-27 前的财报季落在选择段内。
    from app.config import settings
    monkeypatch.setattr(settings, "s_test_freeze_years", 0.01)
    monkeypatch.setattr(settings, "s_embargo_days", 1)
    monkeypatch.setattr(settings, "s_min_selection_days", 100)
    store = AlphaStore(db_url=f"sqlite:///{tmp_path / 'disc.db'}")
    import app.core.workflows.alpha_workflows as wf
    seeds_seen, real_seeds = [], wf._generate_diverse_seeds
    monkeypatch.setattr(wf, "_generate_diverse_seeds",
                        lambda *a, **k: seeds_seen.extend(real_seeds(*a, **k)) or list(seeds_seen))
    rep = DiscoveryEngine(n_families=1, pop_size=8, n_generations=1, n_optuna_trials=1,
                          oos_ratio=0.3, seed=11).run(ds, store=store, save=True, auto_validate=False)
    assert rep.families == ["earnings"]
    assert rep.n_candidates == 1
    dsl = rep.candidates[0].dsl
    # GP 从基本面种子出发。赢家是否仍引用基本面字段取决于一代进化的随机性 —— 2026-10-18 起 SUE 在
    # "新公告已出、申报未到"的几天为空（F13），这组数据上的赢家就成了一个变异体；那不是这条用例要钉的。
    assert any(f in s for s in seeds_seen for f in FUNDAMENTAL_FIELDS), seeds_seen
    rec = store.get_by_id(rep.candidates[0].alpha_id)
    assert rec.status == "candidate" and rec.dsl == dsl


# ---------------------------------------------------------------------------
# 研究路径：load_registry_dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def registry(monkeypatch, tmp_path):
    """一个两只票的美股小数据集 + 预先落盘的真实事实 + 假的价格抓取。"""
    from app.config import settings
    from app.core.data_engine import dataset_registry as reg

    idx = pd.bdate_range("2024-01-02", "2024-06-28")
    close = pd.DataFrame({"AAPL": np.linspace(180, 210, len(idx)), "NVDA": np.linspace(50, 120, len(idx))}, index=idx)
    rng = np.random.default_rng(3)
    raw = {"open": close, "high": close * (1 + np.abs(rng.normal(0, 0.006, close.shape))),
           "low": close * (1 - np.abs(rng.normal(0, 0.006, close.shape))), "close": close,
           "volume": close * 0 + 1e7}
    spec = reg.DatasetSpec("fund_test_us", "yfinance", "US", "tech", ["AAPL", "NVDA"], "2024-01-02")
    monkeypatch.setitem(reg._SPECS, "fund_test_us", spec)
    monkeypatch.setitem(reg._SPECS, "fund_test_crypto",
                        reg.DatasetSpec("fund_test_crypto", "ccxt_binance", "CRYPTO", "l1", ["AAPL", "NVDA"], "2024-01-02"))
    monkeypatch.setattr(reg, "_fetch_raw", lambda spec, start, end: raw)
    monkeypatch.setattr(reg, "_CACHE", {})
    calls = []

    def unadjusted(tickers, start, end, **kw):
        calls.append((tuple(tickers), start, end))
        return close.copy(), {}

    monkeypatch.setattr(fund_prices, "yahoo_raw_and_splits", unadjusted)
    monkeypatch.setattr(settings, "price_source", "yahoo")
    monkeypatch.setattr(settings, "enable_fundamentals", True)
    monkeypatch.setattr(settings, "edgar_user_agent", "")
    monkeypatch.setattr(settings, "fundamentals_dir", str(tmp_path / "fund"))
    st = FundamentalsStore(tmp_path / "fund")
    for tk in ("AAPL", "NVDA"):
        st.save(tk, fx.facts(tk), fx.events(tk))
    return SimpleNamespace(reg=reg, settings=settings, store=st, calls=calls, close=close, tmp=tmp_path)


def _load(registry, name="fund_test_us", **kw):
    return registry.reg.load_registry_dataset(name, start="2024-01-02", end="2024-06-28",
                                              with_sector=False, health_check=False, **kw)


def test_registry_attaches_fundamental_fields_when_enabled(registry):
    ds = _load(registry)
    assert set(ALL_FIELDS) <= set(ds.data)
    assert ds.data["roe"].shape == ds.data["close"].shape
    assert ds.data["roe"]["AAPL"]["2024-02-05"] == pytest.approx(100.913 / 74.100, rel=1e-5)
    assert registry.calls == [(("AAPL", "NVDA"), "2024-01-02", "2024-06-28")]
    sig = Executor().run_expr("fund_rank(earnings_yield)", ds.data)
    assert sig.iloc[-1].notna().all()


def test_registry_leaves_the_dataset_untouched_when_disabled(registry, monkeypatch):
    monkeypatch.setattr(registry.settings, "enable_fundamentals", False)
    ds = _load(registry)
    assert not (set(ds.data) & FUNDAMENTAL_FIELDS)
    assert registry.calls == []


def test_explicit_argument_overrides_the_setting_both_ways(registry, monkeypatch):
    assert not (set(_load(registry, with_fundamentals=False).data) & FUNDAMENTAL_FIELDS)
    monkeypatch.setattr(registry.settings, "enable_fundamentals", False)
    assert set(ALL_FIELDS) <= set(_load(registry, with_fundamentals=True).data)


def test_cache_does_not_serve_a_price_only_dataset_to_a_fundamentals_request(registry):
    plain = _load(registry, with_fundamentals=False)
    rich = _load(registry, with_fundamentals=True)
    assert plain is not rich
    assert "roe" in rich.data and "roe" not in plain.data
    assert _load(registry, with_fundamentals=True) is rich            # 同一请求仍命中缓存
    assert _load(registry, with_fundamentals=False) is plain


def test_non_us_datasets_never_get_fundamentals(registry):
    ds = _load(registry, name="fund_test_crypto", with_fundamentals=True)
    assert not (set(ds.data) & FUNDAMENTAL_FIELDS)
    assert registry.calls == []


def test_enabled_with_no_facts_available_fails_the_load(registry, monkeypatch):
    monkeypatch.setattr(registry.settings, "fundamentals_dir", str(registry.tmp / "empty"))
    with pytest.raises(FundamentalsUnavailable):
        _load(registry)
    assert registry.reg._CACHE == {}                                   # 失败的加载不进缓存


# ---------------------------------------------------------------------------
# 前向路径：每日管线
# ---------------------------------------------------------------------------

def test_daily_ingest_asks_for_prices_only(registry, monkeypatch):
    """基本面是派生量，不进价格 PIT：摄取这一步必须显式关掉它。"""
    from app.tasks.daily_ingest import DailyIngest
    seen = {}
    real = registry.reg.load_registry_dataset

    def spy(name, **kw):
        seen.update(kw)
        return real(name, with_sector=False, **kw)

    monkeypatch.setattr(registry.reg, "load_registry_dataset", spy)
    res = DailyIngest(min_health=0.0).ingest("fund_test_us", "2024-01-02", "2024-06-28", append_pit=False)
    assert res.accepted and seen["with_fundamentals"] is False
    assert not (set(res.dataset) & FUNDAMENTAL_FIELDS)


def test_pipeline_attaches_fundamentals_to_the_panel_fed_to_the_trading_loop(registry):
    from app.tasks.daily_ingest import _with_fundamentals
    panel = {"close": registry.close.copy()}
    out = _with_fundamentals("fund_test_us", panel)
    assert set(ALL_FIELDS) <= set(out) and "roe" not in panel          # 不改传入的面板
    # PIT 面板是多 vintage 拼接的 → 不把它当复权价传下去（见 prices.py）
    assert registry.calls == [(("AAPL", "NVDA"), "2024-01-02", "2024-06-28")]


def test_pipeline_is_a_no_op_when_fundamentals_are_off(registry, monkeypatch):
    from app.tasks.daily_ingest import _with_fundamentals
    monkeypatch.setattr(registry.settings, "enable_fundamentals", False)
    panel = {"close": registry.close.copy()}
    assert _with_fundamentals("fund_test_us", panel) is panel
    assert _with_fundamentals("no_such_dataset", panel) is panel       # 关着时连数据集名都不查


def test_pipeline_attach_failure_leaves_the_fields_absent_and_says_so(registry, monkeypatch, caplog):
    from app.tasks.daily_ingest import _with_fundamentals
    monkeypatch.setattr(registry.settings, "fundamentals_dir", str(registry.tmp / "empty"))
    panel = {"close": registry.close.copy()}
    with caplog.at_level(logging.ERROR):
        out = _with_fundamentals("fund_test_us", panel)
    assert out is panel and not (set(out) & FUNDAMENTAL_FIELDS)
    assert fx.logged(caplog, "基本面挂载失败")
    with pytest.raises(KeyError, match="roe"):                          # 因子报错，而不是拿 NaN 继续
        Executor().run_expr("fund_rank(roe)", out)


def test_pipeline_skips_non_us_datasets(registry):
    from app.tasks.daily_ingest import _with_fundamentals
    panel = {"close": registry.close.copy()}
    assert _with_fundamentals("fund_test_crypto", panel) is panel
    assert registry.calls == []


# ---------------------------------------------------------------------------
# 前向路径：从 run_daily_pipeline 这个入口看（调度任务调的就是它）
# ---------------------------------------------------------------------------

class _CapturingLoop:
    """替身交易循环：只记下喂进来的面板长什么样。"""
    seen: list = []

    def run(self, dataset, forward_from=None, now=None, data_source=None, identity=None):
        _CapturingLoop.seen.append(("factors", set(dataset), forward_from))
        return SimpleNamespace(n_alphas=0, n_alerts=0, n_errors=0)

    def run_portfolio(self, dataset, forward_from=None, now=None, data_source=None, identity=None):
        _CapturingLoop.seen.append(("portfolio", set(dataset), forward_from))
        return {"n_factors": 0}

    def _maintain_live(self, reason):
        return {"reason": reason}


@pytest.fixture
def pipeline(registry, monkeypatch):
    from app.tasks import daily_ingest as di
    from app.tasks import daily_trading_loop as dtl
    _CapturingLoop.seen = []
    monkeypatch.setattr(dtl, "DailyTradingLoop", _CapturingLoop)
    panel = {"close": registry.close.copy(), "volume": registry.close * 0 + 1e7}
    monkeypatch.setattr(di.DailyIngest, "ingest_incremental",
                        lambda self, name, backfill_start=None, now=None: di.IngestResult(
                            True, name, "2024-06-28T21:30:00+00:00", health_score=1.0, dataset=panel,
                            mode="incremental", forward_from="2024-06-28", n_new_bars=1))
    return SimpleNamespace(di=di, panel=panel, registry=registry)


def test_daily_pipeline_feeds_fundamental_fields_to_both_loops(pipeline):
    out = pipeline.di.run_daily_pipeline("fund_test_us")
    assert out["ingest_accepted"] is True
    kinds = [k for k, _, _ in _CapturingLoop.seen]
    assert kinds == ["factors", "portfolio"]
    for _, fields, forward_from in _CapturingLoop.seen:
        assert set(ALL_FIELDS) <= fields and forward_from == "2024-06-28"
    assert set(pipeline.panel) == {"close", "volume"}                 # 摄取结果本身没被改写


def test_daily_pipeline_with_the_switch_off_feeds_prices_only(pipeline, monkeypatch):
    monkeypatch.setattr(pipeline.registry.settings, "enable_fundamentals", False)
    pipeline.di.run_daily_pipeline("fund_test_us")
    assert [fields for _, fields, _ in _CapturingLoop.seen] == [{"close", "volume"}] * 2
    assert pipeline.registry.calls == []


def test_daily_pipeline_survives_a_fundamentals_outage(pipeline, monkeypatch, caplog):
    """基本面挂不上不能拖垮每日管线：两个循环照常跑，只是面板里没有这些字段。"""
    monkeypatch.setattr(pipeline.registry.settings, "fundamentals_dir", str(pipeline.registry.tmp / "empty"))
    with caplog.at_level(logging.ERROR):
        out = pipeline.di.run_daily_pipeline("fund_test_us")
    assert out["ingest_accepted"] is True
    assert [fields for _, fields, _ in _CapturingLoop.seen] == [{"close", "volume"}] * 2
    assert fx.logged(caplog, "基本面挂载失败")
