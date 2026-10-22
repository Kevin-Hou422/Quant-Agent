"""
上线预检的"基本面"一项（Phase 10 接进前向路径之后的那块缺口）。

此前预检完全不知道基本面的存在：现役策略如果用到 `earnings_yield`，而 `.env` 里
ENABLE_FUNDAMENTALS 没开，预检照样 15 项全过 —— 到了每日管线里因子才因"字段不存在"报错，
当天不调仓。预检说"可以交给调度器"，却没查这条依赖（DEV_LESSONS：预检的措辞不得超出实际检查范围）。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from app.core.data_engine.fundamentals.store import FundamentalsStore
from app.core.execution import golive
from app.core.execution.golive import MIN_FUNDAMENTALS_READY, run_preflight
from app.db.execution_store import ExecutionStore
from app.db.position_store import PositionStore
from tests.unit.data_engine import fundamentals_fixtures as fx
from tests.unit.execution.moomoo_fake import FakeTradeContext, make_gateway

NOW = datetime(2025, 3, 4, 22, 0, tzinfo=timezone.utc)
JOBS = ["daily_backup", "daily_monitor", "daily_trading", "daily_trading_retry", "monthly_cost_calibration"]
UA = "Someone someone@example.com"


def _settings(tmp_path, **kw):
    from app.config import settings
    base = dict(
        execution_mode="moomoo_paper", enable_scheduler=True, enable_paper_trading=True,
        database_url=f"sqlite:///{(tmp_path / 'a.db').as_posix()}",
        scheduler_db_url=f"sqlite:///{(tmp_path / 's.db').as_posix()}",
        pit_store_dir=str(tmp_path / "pit"), fundamentals_dir=str(tmp_path / "fund"),
        paper_aum=1_000_000.0, moomoo_host="127.0.0.1", moomoo_port=11111,
        paper_dataset="fund_pf_us", price_source="moomoo",
        enable_fundamentals=False, edgar_user_agent="",
    )
    base.update(kw)
    return settings.model_copy(update=base)


@pytest.fixture(autouse=True)
def _dataset(monkeypatch):
    """四只票的美股数据集：AAPL / NVDA / GOOGL / BKNG（都有真实事实夹具）。"""
    from app.core.data_engine import dataset_registry as reg
    monkeypatch.setitem(reg._SPECS, "fund_pf_us",
                        reg.DatasetSpec("fund_pf_us", "yfinance", "US", "mixed",
                                        ["AAPL", "NVDA", "GOOGL", "BKNG"], "2021-01-01"))


def _strategy(tmp_path, dsls, active=True):
    from app.db.alpha_store import AlphaResult, AlphaStore
    from app.db.strategy_store import StrategyConfig, StrategyStore
    url = f"sqlite:///{(tmp_path / 'a.db').as_posix()}"
    store, ids = AlphaStore(db_url=url), []
    from tests.conftest import approved_verdict
    for dsl in dsls:
        aid = store.save(AlphaResult(dsl=dsl, status="candidate"))
        store.update_status(aid, "validated")
        store.update_status(aid, "paper")
        ids.append(str(aid))
    ss = StrategyStore(db_url=url)
    sid = ss.save(StrategyConfig(factors=ids, combo_weights={i: 1.0 for i in ids}, aum=1e6,
                                 verdict=approved_verdict(dict(zip(ids, dsls)), {i: 1.0 for i in ids},
                                                          1e6, settings=_settings(tmp_path))))
    if active:
        ss.update_status(sid, "approved")
        ss.update_status(sid, "paper_running")
    return sid


def _store(tmp_path, tickers):
    st = FundamentalsStore(tmp_path / "fund")
    for tk in tickers:
        st.save(tk, fx.facts(tk), fx.events(tk))
    return st


TODAY = pd.Timestamp("2026-09-15")        # 夹具最后一份财报之后不久


# ---------------------------------------------------------------------------
# 现役策略用到哪些基本面字段
# ---------------------------------------------------------------------------

def test_fields_used_by_the_active_strategy_are_collected_across_components(tmp_path):
    _strategy(tmp_path, ["rank(ts_delta(log(close), 252))",
                         "fund_rank(earnings_yield) + fund_rank(roe)",
                         "fund_rank(trade_when(days_since_earnings < 60, sue))"])
    assert golive._strategy_fundamental_fields(_settings(tmp_path)) == [
        "days_since_earnings", "earnings_yield", "roe", "sue"]


def test_price_only_strategy_needs_no_fundamentals(tmp_path):
    _strategy(tmp_path, ["rank(ts_delta(log(close), 252))", "rank(-ts_std(returns, 60))"])
    assert golive._strategy_fundamental_fields(_settings(tmp_path)) == []


def test_no_active_strategy_needs_no_fundamentals(tmp_path):
    _strategy(tmp_path, ["fund_rank(roe)"], active=False)
    assert golive._strategy_fundamental_fields(_settings(tmp_path)) == []


# ---------------------------------------------------------------------------
# 策略用不到基本面：只提示，不阻断
# ---------------------------------------------------------------------------

def test_off_and_not_needed_is_fine(tmp_path):
    assert golive._fundamentals_problem(_settings(tmp_path), needed=[]) == (
        True, "未开启（现役策略只用价格字段）", False)


def test_on_but_no_identity_and_not_needed_is_a_non_blocking_warning(tmp_path):
    ok, detail, blocking = golive._fundamentals_problem(_settings(tmp_path, enable_fundamentals=True), needed=[])
    assert (ok, blocking) == (False, False) and "EDGAR_USER_AGENT" in detail
    ok, detail, blocking = golive._fundamentals_problem(
        _settings(tmp_path, enable_fundamentals=True, edgar_user_agent=UA), needed=[])
    assert (ok, blocking) == (True, False)


# ---------------------------------------------------------------------------
# 策略用到基本面：每一环都是阻断项
# ---------------------------------------------------------------------------

def test_needed_but_switched_off_blocks(tmp_path):
    ok, detail, blocking = golive._fundamentals_problem(_settings(tmp_path), needed=["earnings_yield"])
    assert (ok, blocking) == (False, True)
    assert "earnings_yield" in detail and "ENABLE_FUNDAMENTALS" in detail and "不调仓" in detail


def test_needed_but_no_identity_blocks(tmp_path):
    ok, detail, blocking = golive._fundamentals_problem(
        _settings(tmp_path, enable_fundamentals=True), needed=["roe"])
    assert (ok, blocking) == (False, True) and "EDGAR_USER_AGENT" in detail


def test_needed_and_enough_fresh_local_facts_passes(tmp_path):
    _store(tmp_path, ["AAPL", "NVDA", "GOOGL", "BKNG"])
    ok, detail, blocking = golive._fundamentals_problem(
        _settings(tmp_path, enable_fundamentals=True, edgar_user_agent=UA), needed=["roe"], today=TODAY)
    assert (ok, blocking) == (True, True) and "4/4" in detail


def test_local_coverage_boundary(tmp_path):
    """4 只票里 2 只有事实 = 恰好 50% → 过；1 只 → 不过。"""
    assert MIN_FUNDAMENTALS_READY == 0.5
    s = _settings(tmp_path, enable_fundamentals=True, edgar_user_agent=UA)
    _store(tmp_path, ["AAPL"])
    ok, detail, blocking = golive._fundamentals_problem(s, needed=["roe"], today=TODAY)
    assert (ok, blocking) == (False, True) and "1/4" in detail and "同步" in detail
    _store(tmp_path, ["NVDA"])
    ok, detail, _ = golive._fundamentals_problem(s, needed=["roe"], today=TODAY)
    assert ok and "2/4" in detail


def test_stale_facts_do_not_count_as_ready(tmp_path):
    """事实都在，但最近一份财报是 200 多天前的（没在更新）→ 不算就绪。"""
    _store(tmp_path, ["AAPL", "NVDA", "GOOGL", "BKNG"])
    ok, detail, blocking = golive._fundamentals_problem(
        _settings(tmp_path, enable_fundamentals=True, edgar_user_agent=UA), needed=["roe"],
        today=pd.Timestamp("2027-06-01"))
    assert (ok, blocking) == (False, True) and "0/4" in detail and "4 只已过期" in detail


def test_the_check_reads_local_state_only(tmp_path, monkeypatch):
    import requests
    monkeypatch.setattr(requests.Session, "get", lambda *a, **k: pytest.fail("预检不该向 SEC 发请求"))
    _store(tmp_path, ["AAPL", "NVDA"])
    ok, _, _ = golive._fundamentals_problem(
        _settings(tmp_path, enable_fundamentals=True, edgar_user_agent=UA), needed=["roe"], today=TODAY)
    assert ok


# ---------------------------------------------------------------------------
# 接进 run_preflight
# ---------------------------------------------------------------------------

def _preflight(tmp_path, **settings_kw):
    ctx = FakeTradeContext(cash=1_000_000.0)
    return run_preflight(
        _settings(tmp_path, **settings_kw), gateway_factory=lambda: make_gateway(ctx),
        probe=lambda h, p: True, store=ExecutionStore(db_url=f"sqlite:///{tmp_path / 'e.db'}"),
        position_store=PositionStore(db_url=f"sqlite:///{tmp_path / 'p.db'}"), now=NOW, cwd=tmp_path,
        data_probe=lambda st, target: (True, "bar ok"), jobs=lambda: list(JOBS))


def _check(rep, name):
    return {c.name: c for c in rep.checks}[name]


def test_fundamental_strategy_with_the_switch_off_is_not_ready(tmp_path):
    """这就是那个缺口：其它每一项都是绿的，唯独策略依赖的数据没开。"""
    _strategy(tmp_path, ["fund_rank(earnings_yield)"])
    rep = _preflight(tmp_path)
    c = _check(rep, "fundamentals")
    assert not c.ok and c.blocking and "ENABLE_FUNDAMENTALS" in c.detail
    assert not rep.ready
    assert [x.name for x in rep.checks if x.blocking and not x.ok] == ["fundamentals"]
    assert rep.connected                                    # 券商这一侧没问题


def test_price_only_strategy_is_ready_without_fundamentals(tmp_path):
    _strategy(tmp_path, ["rank(ts_delta(log(close), 252))"])
    rep = _preflight(tmp_path)
    c = _check(rep, "fundamentals")
    assert c.ok and not c.blocking and rep.ready


def test_fundamental_strategy_fully_configured_is_ready(tmp_path):
    _strategy(tmp_path, ["fund_rank(earnings_yield) + rank(ts_delta(log(close), 252))"])
    _store(tmp_path, ["AAPL", "NVDA", "GOOGL"])
    rep = _preflight(tmp_path, enable_fundamentals=True, edgar_user_agent=UA)
    c = _check(rep, "fundamentals")
    assert c.ok and c.blocking and "earnings_yield" in c.detail and "3/4" in c.detail
    assert rep.ready


def test_a_crashing_fundamentals_check_fails_closed(tmp_path):
    def boom(st):
        raise RuntimeError("store unreadable")

    _strategy(tmp_path, ["rank(close)"])
    ctx = FakeTradeContext(cash=1_000_000.0)
    rep = run_preflight(
        _settings(tmp_path), gateway_factory=lambda: make_gateway(ctx), probe=lambda h, p: True,
        store=ExecutionStore(db_url=f"sqlite:///{tmp_path / 'e.db'}"),
        position_store=PositionStore(db_url=f"sqlite:///{tmp_path / 'p.db'}"), now=NOW, cwd=tmp_path,
        data_probe=lambda st, target: (True, "bar ok"), jobs=lambda: list(JOBS), fundamentals_check=boom)
    c = _check(rep, "fundamentals")
    assert not c.ok and c.blocking and "store unreadable" in c.detail and not rep.ready


def test_the_identity_itself_never_appears_in_the_report(tmp_path):
    _strategy(tmp_path, ["fund_rank(roe)"])
    _store(tmp_path, ["AAPL", "NVDA"])
    rep = _preflight(tmp_path, enable_fundamentals=True, edgar_user_agent=UA)
    dumped = repr(rep.to_dict())
    assert dumped.count("someone@example.com") == 0 and dumped.count("Someone") == 0
