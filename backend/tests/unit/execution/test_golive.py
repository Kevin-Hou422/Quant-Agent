"""
test_golive.py — 前向交易上线预检 + 活数据位置启动保险

要钉死的事：
  1. 预检的每一项都**真的会失败**（不是永远 OK 的装饰），且失败后不会继续做危险动作
     （OpenD 不在就不建连；任何情况下都不下单）。
  2. 两层语义分开（审计 F13）：`connected` 只说券商这一侧能用；`ready` 还要求配置、调度任务、
     存储可写、数据集、**已激活策略**、**数据源当下就拿得到最近已收盘交易日的 bar**。
     以前数据源必定抛错时 ready 仍是 True。
  3. 云同步目录的判断看**解析后的路径**：默认配置 `sqlite:///./alphas.db` 字符串里没有
     "onedrive"，但从 OneDrive 里的仓库启动时它就在 OneDrive。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from app.core.execution import golive
from app.core.execution.golive import (
    LAYER_CONNECT, LAYER_TRADE, assert_live_storage_safe, forward_trading_enabled, in_cloud_sync,
    live_storage_problems, run_preflight, sqlite_path, storage_not_writable,
)
from app.db.execution_store import ExecutionStore
from app.db.position_store import PositionStore
from tests.unit.execution.moomoo_fake import FakeTradeContext, make_gateway

NOW = datetime(2025, 3, 4, 22, 0, tzinfo=timezone.utc)       # 周二，收盘后
JOBS = ["daily_backup", "daily_monitor", "daily_trading", "daily_trading_retry",
        "monthly_cost_calibration"]
TRADE_CHECKS = ["execution_mode", "scheduler", "scheduler_jobs", "storage", "storage_writable",
                "calendar", "dataset", "price_source", "strategy"]
CONNECT_CHECKS = ["opend", "broker_account", "account_funds", "kill_switch", "reconcile"]


def _settings(tmp_path, **kw):
    from app.config import settings
    base = dict(
        execution_mode="moomoo_paper", enable_scheduler=True, enable_paper_trading=True,
        database_url=f"sqlite:///{(tmp_path / 'a.db').as_posix()}",
        scheduler_db_url=f"sqlite:///{(tmp_path / 's.db').as_posix()}",
        pit_store_dir=str(tmp_path / "pit"), paper_aum=1_000_000.0,
        exec_max_aum_mismatch=0.5, moomoo_host="127.0.0.1", moomoo_port=11111,
        paper_dataset="us_broad_large", price_source="moomoo",
    )
    base.update(kw)
    return settings.model_copy(update=base)


def _env(tmp_path, cash=1_000_000.0, data=(True, "bar ok"), strategy=(True, "active #1"),
         jobs=JOBS, **kw):
    ctx = FakeTradeContext(cash=cash)
    es = ExecutionStore(db_url=f"sqlite:///{tmp_path / 'e.db'}")
    ps = PositionStore(db_url=f"sqlite:///{tmp_path / 'p.db'}")
    calls = []

    def factory():
        calls.append(1)
        return make_gateway(ctx)

    s = _settings(tmp_path, **kw)
    seen = {}

    def probe_data(st, target):
        seen["data_target"] = target
        return data

    run = lambda probe=lambda h, p: True: run_preflight(               # noqa: E731
        s, gateway_factory=factory, probe=probe, store=es, position_store=ps, now=NOW,
        cwd=tmp_path, data_probe=probe_data, jobs=lambda: list(jobs),
        strategy_check=lambda st: strategy)
    run.seen = seen
    return ctx, es, ps, calls, run


def _by_name(rep):
    return {c.name: c for c in rep.checks}


# ---------------------------------------------------------------------------
# 两层预检
# ---------------------------------------------------------------------------

def test_everything_in_place_is_ready_and_places_no_order(tmp_path):
    ctx, es, _, _, run = _env(tmp_path)
    rep = run()
    assert [c.name for c in rep.checks] == TRADE_CHECKS + CONNECT_CHECKS + ["data_source"]
    assert {c.name for c in rep.checks if c.layer == LAYER_CONNECT} == set(CONNECT_CHECKS)
    assert rep.ready and rep.connected and all(c.ok for c in rep.checks)
    assert rep.gateway["acc_id"] == 2001 and rep.gateway["trd_env"] == "SIMULATE"
    assert rep.account["total_assets"] == pytest.approx(1_000_000.0)
    assert rep.reconcile["status"] == "baseline"
    assert _by_name(rep)["calendar"].detail.endswith("2025-03-04")
    assert run.seen["data_target"] == pd.Timestamp("2025-03-04")     # 探测的是最近已收盘交易日
    assert ctx.calls_of("place_order") == []
    assert ctx.closed
    d = rep.to_dict()
    assert d["ready"] is True and d["connected"] is True and len(d["checks"]) == 15


def test_connected_but_not_tradable_is_reported_as_such(tmp_path):
    """券商侧全过、策略缺失：connected=True 但 ready=False —— 两个结论不能混成一个。"""
    _, _, _, _, run = _env(tmp_path, strategy=(False, "没有 active 策略配置"))
    rep = run()
    assert rep.connected and not rep.ready
    assert not _by_name(rep)["strategy"].ok


def test_a_failing_data_source_makes_the_preflight_not_ready(tmp_path):
    """审计 F13 的原始复现：数据源必定失败时 ready 曾是 True、且从未访问数据源。"""
    _, _, _, _, run = _env(tmp_path, data=(False, "数据源最新 bar 2025-03-03 早于 2025-03-04"))
    rep = run()
    c = _by_name(rep)["data_source"]
    assert not rep.ready and rep.connected and not c.ok and "2025-03-03" in c.detail


def test_a_raising_data_probe_is_a_failure(tmp_path):
    def boom(st, target):
        raise RuntimeError("yahoo 503")

    rep = run_preflight(_settings(tmp_path, price_source="yahoo"), probe=lambda h, p: False,
                        now=NOW, cwd=tmp_path, data_probe=boom, jobs=lambda: JOBS,
                        strategy_check=lambda st: (True, ""))
    c = _by_name(rep)["data_source"]
    assert not c.ok and "yahoo 503" in c.detail


def test_a_raising_strategy_check_is_a_failure(tmp_path):
    def boom(st):
        raise RuntimeError("db locked")

    rep = run_preflight(_settings(tmp_path), probe=lambda h, p: False, now=NOW, cwd=tmp_path,
                        data_probe=lambda st, t: (True, ""), jobs=lambda: JOBS,
                        strategy_check=boom)
    c = _by_name(rep)["strategy"]
    assert not c.ok and "db locked" in c.detail


def test_opend_down_skips_broker_checks_but_reports_everything_else(tmp_path):
    _, _, _, calls, run = _env(tmp_path)
    rep = run(probe=lambda h, p: False)
    names = [c.name for c in rep.checks]
    assert names == TRADE_CHECKS + ["opend", "data_source"]
    c = _by_name(rep)
    assert not rep.ready and not rep.connected
    assert "未在监听" in c["opend"].detail
    assert calls == []                      # SDK 在网关不在时会无限阻塞 —— 绝不能去建连
    assert rep.gateway is None
    # moomoo 行情同样走 OpenD：不去探测，直接判失败
    assert not c["data_source"].ok and "OpenD" in c["data_source"].detail
    assert "data_target" not in run.seen


def test_opend_down_with_yahoo_still_probes_data(tmp_path):
    _, _, _, _, run = _env(tmp_path, price_source="yahoo")
    rep = run(probe=lambda h, p: False)
    assert _by_name(rep)["data_source"].ok and "data_target" in run.seen


def test_probe_receives_the_configured_address(tmp_path):
    seen = []
    _, _, _, _, run = _env(tmp_path, moomoo_host="10.0.0.5", moomoo_port=22222)
    run(probe=lambda h, p: seen.append((h, p)) or True)
    assert seen == [("10.0.0.5", 22222)]


def test_wrong_mode_and_disabled_scheduler_are_blocking_but_broker_is_still_checked(tmp_path):
    _, _, _, calls, run = _env(tmp_path, execution_mode="off", enable_scheduler=False)
    rep = run()
    c = _by_name(rep)
    assert not rep.ready and rep.connected
    assert not c["execution_mode"].ok and "moomoo_paper" in c["execution_mode"].detail
    assert not c["scheduler"].ok
    assert c["reconcile"].ok and calls == [1]


@pytest.mark.parametrize("sched,paper", [(True, False), (False, True)])
def test_scheduler_needs_both_switches(tmp_path, sched, paper):
    _, _, _, _, run = _env(tmp_path, enable_scheduler=sched, enable_paper_trading=paper)
    assert not _by_name(run())["scheduler"].ok


def test_missing_scheduled_jobs_are_blocking(tmp_path):
    _, _, _, _, run = _env(tmp_path, jobs=["daily_monitor", "daily_trading"])
    c = _by_name(run())["scheduler_jobs"]
    assert not c.ok and "daily_trading_retry" in c.detail


def test_scheduler_build_failure_is_blocking(tmp_path):
    def boom():
        raise RuntimeError("jobstore locked")
    rep = run_preflight(_settings(tmp_path), probe=lambda h, p: False, now=NOW, cwd=tmp_path,
                        data_probe=lambda s, t: (True, ""), jobs=boom,
                        strategy_check=lambda s: (True, ""))
    c = _by_name(rep)["scheduler_jobs"]
    assert not c.ok and "jobstore locked" in c.detail


def test_the_real_scheduler_registers_both_trading_jobs(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "enable_paper_trading", True)
    ids = golive._registered_jobs()
    assert {"daily_trading", "daily_trading_retry"} <= set(ids)


def test_non_us_dataset_is_blocking(tmp_path):
    _, _, _, _, run = _env(tmp_path, paper_dataset="crypto_major")
    c = _by_name(run())["dataset"]
    assert not c.ok and "只交易美股" in c.detail


def test_unknown_dataset_is_blocking(tmp_path):
    _, _, _, _, run = _env(tmp_path, paper_dataset="no_such_dataset")
    c = _by_name(run())["dataset"]
    assert not c.ok and "no_such_dataset" in c.detail


def test_yahoo_price_source_is_a_warning_not_a_block(tmp_path):
    _, _, _, _, run = _env(tmp_path, price_source="yahoo")
    rep = run()
    c = _by_name(rep)["price_source"]
    assert not c.ok and not c.blocking and "不同源" in c.detail
    assert rep.ready


def test_unwritable_pit_dir_is_blocking(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")                           # PIT 目录位置上是个文件 → 建不了目录
    _, _, _, _, run = _env(tmp_path, pit_store_dir=str(blocker / "pit"))
    c = _by_name(run())["storage_writable"]
    assert not c.ok and "pit_store_dir" in c.detail


def test_storage_not_writable_reports_nothing_for_good_dirs(tmp_path):
    assert storage_not_writable(_settings(tmp_path), cwd=tmp_path) == []
    assert (tmp_path / "pit").is_dir()


@pytest.mark.parametrize("cash,ok", [
    (1_500_000.0, True),       # 偏离恰好 50% —— 上限是闭区间
    (500_000.0, True),
    (1_500_001.0, False),
    (100_000.0, False),
])
def test_account_must_match_paper_aum(tmp_path, cash, ok):
    _, _, _, _, run = _env(tmp_path, cash=cash)
    c = _by_name(run())["account_funds"]
    assert c.ok is ok and c.layer == LAYER_CONNECT
    # 不通过时才给出改法；通过时不该出现误导性的建议
    assert (f"把 PAPER_AUM 设为 {cash:.0f}" in c.detail) is (not ok)


def test_empty_account_is_refused(tmp_path):
    _, _, _, _, run = _env(tmp_path, cash=0.0)
    c = _by_name(run())["account_funds"]
    assert not c.ok and "≤ 0" in c.detail


def test_unreadable_account_is_a_failure_not_a_skip(tmp_path):
    ctx, _, _, _, run = _env(tmp_path)
    ctx.fail["accinfo_query"] = "busy"
    rep = run()
    c = _by_name(rep)
    assert not rep.ready and not rep.connected
    assert not c["account_funds"].ok and "读不到账户资金" in c["account_funds"].detail
    assert not c["reconcile"].ok and "对账失败" in c["reconcile"].detail
    assert ctx.closed


def test_gateway_failure_is_reported(tmp_path):
    s = _settings(tmp_path)

    def boom():
        raise RuntimeError("login required")

    rep = run_preflight(s, gateway_factory=boom, probe=lambda h, p: True, now=NOW, cwd=tmp_path,
                        data_probe=lambda st, t: (True, ""), jobs=lambda: JOBS,
                        strategy_check=lambda st: (True, ""))
    c = _by_name(rep)["broker_account"]
    assert not rep.ready and not rep.connected and not c.ok and "login required" in c.detail
    assert [x.name for x in rep.checks][-2:] == ["broker_account", "data_source"]


def test_engaged_kill_switch_blocks(tmp_path):
    _, es, _, _, run = _env(tmp_path)
    es.set_kill_switch(True, "kevin", "drill")
    c = _by_name(run())["kill_switch"]
    assert not c.ok and "kevin" in c.detail and "drill" in c.detail


def test_reconcile_discrepancy_blocks(tmp_path):
    ctx, _, _, _, run = _env(tmp_path)
    assert run().ready                                     # 第一次：建立基准
    ctx.set_position("ZZZ", 10, 5.0)                       # 账本外的持仓出现在券商侧
    ctx.account_override = {"total_assets": 1_000_050.0}
    rep = run()
    c = _by_name(rep)["reconcile"]
    assert not rep.ready and not c.ok and "discrepancy" in c.detail


def test_calendar_failure_is_blocking_and_skips_the_data_probe(tmp_path, monkeypatch):
    import app.core.data_engine.market_calendar as mc

    def boom(now):
        raise RuntimeError("calendar gone")

    monkeypatch.setattr(mc, "last_closed_session", boom)
    _, _, _, _, run = _env(tmp_path)
    rep = run()
    c = _by_name(rep)
    assert not c["calendar"].ok and "calendar gone" in c["calendar"].detail
    assert not c["data_source"].ok and "日历" in c["data_source"].detail
    assert "data_target" not in run.seen


def test_storage_in_cloud_sync_is_blocking(tmp_path):
    synced = tmp_path / "OneDrive" / "repo"
    synced.mkdir(parents=True)
    s = _settings(tmp_path, database_url="sqlite:///./alphas.db")
    rep = run_preflight(s, probe=lambda h, p: False, now=NOW, cwd=synced,
                        data_probe=lambda st, t: (True, ""), jobs=lambda: JOBS,
                        strategy_check=lambda st: (True, ""))
    c = _by_name(rep)["storage"]
    assert not c.ok and "database_url" in c.detail


# ---------------------------------------------------------------------------
# 默认的数据源探测与策略检查（不注入时预检实际调用的实现）
# ---------------------------------------------------------------------------

def _raw(dates):
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    return {"close": pd.DataFrame({"AAPL": [100.0] * len(idx)}, index=idx)}


def test_default_data_probe_requires_the_target_bar(tmp_path, monkeypatch):
    import app.core.data_engine.dataset_registry as dr
    seen = {}

    def fetch(spec, start, end):
        seen.update(universe=list(spec.universe), start=start, end=end)
        return _raw(["2025-02-28", "2025-03-03"])

    monkeypatch.setattr(dr, "_fetch_raw", fetch)
    ok, detail = golive._probe_data(_settings(tmp_path), pd.Timestamp("2025-03-04"))
    assert not ok and "2025-03-03" in detail and "2025-03-04" in detail
    assert len(seen["universe"]) == 2 and seen["end"] == "2025-03-04"
    assert seen["start"] == "2025-02-22"                          # 目标日前 10 天

    monkeypatch.setattr(dr, "_fetch_raw", lambda spec, s, e: _raw(["2025-03-03", "2025-03-04"]))
    ok, detail = golive._probe_data(_settings(tmp_path), pd.Timestamp("2025-03-04"))
    assert ok and "2025-03-04" in detail


def test_default_data_probe_rejects_empty_data(tmp_path, monkeypatch):
    import app.core.data_engine.dataset_registry as dr
    monkeypatch.setattr(dr, "_fetch_raw", lambda spec, s, e: {"close": pd.DataFrame()})
    ok, detail = golive._probe_data(_settings(tmp_path), pd.Timestamp("2025-03-04"))
    assert not ok and "空数据" in detail


def _strategy_db(tmp_path, statuses, active=True):
    from app.db.alpha_store import AlphaResult, AlphaStore
    from app.db.strategy_store import StrategyConfig, StrategyStore
    url = f"sqlite:///{(tmp_path / 'a.db').as_posix()}"
    store, ids = AlphaStore(db_url=url), []
    for st in statuses:
        aid = store.save(AlphaResult(dsl="rank(close)", status="candidate"))
        for nxt in {"validated": ["validated"], "paper": ["validated", "paper"]}.get(st, []):
            store.update_status(aid, nxt)
        ids.append(str(aid))
    ss = StrategyStore(db_url=url)
    sid = ss.save(StrategyConfig(factors=ids, combo_weights={i: 1.0 for i in ids}, aum=1e6))
    if active:
        ss.update_status(sid, "approved")
        ss.update_status(sid, "active")
    return sid


def test_default_strategy_check_requires_an_active_config(tmp_path):
    _strategy_db(tmp_path, ["paper"], active=False)
    ok, detail = golive._active_strategy_problem(_settings(tmp_path))
    assert not ok and "没有 active 策略配置" in detail


def test_default_strategy_check_requires_tradable_components(tmp_path):
    sid = _strategy_db(tmp_path, ["paper", "validated"])
    ok, detail = golive._active_strategy_problem(_settings(tmp_path))
    assert not ok and f"#{sid}" in detail and "不在可交易状态" in detail


def test_default_strategy_check_passes_with_paper_components(tmp_path):
    sid = _strategy_db(tmp_path, ["paper", "paper"])
    ok, detail = golive._active_strategy_problem(_settings(tmp_path))
    assert ok and f"#{sid}" in detail and "2 个成分" in detail


# ---------------------------------------------------------------------------
# 路径解析与启动保险
# ---------------------------------------------------------------------------

def test_sqlite_path_resolution(tmp_path):
    assert sqlite_path("sqlite:///./a.db", tmp_path) == (tmp_path / "a.db").resolve()
    assert sqlite_path("sqlite:///a.db", tmp_path) == (tmp_path / "a.db").resolve()
    absolute = (tmp_path / "x" / "b.db").resolve()
    assert sqlite_path(f"sqlite:///{absolute.as_posix()}", Path("/elsewhere")) == absolute
    assert sqlite_path("sqlite:///:memory:", tmp_path) is None
    assert sqlite_path("sqlite:///", tmp_path) is None
    assert sqlite_path("postgresql://h/db", tmp_path) is None


@pytest.mark.parametrize("p,hit", [
    ("C:/Users/x/OneDrive/Desktop/repo/a.db", True),
    ("C:\\Users\\x\\OneDrive - Corp\\a.db", True),
    ("/home/x/Dropbox/a.db", True),
    ("/Users/x/Google Drive/a.db", True),
    ("/Users/x/GoogleDrive/a.db", True),
    ("/Users/x/Library/Mobile Documents/iCloud~x/a.db", True),
    ("C:/QuantAgentData/a.db", False),
])
def test_cloud_sync_markers(p, hit):
    assert in_cloud_sync(Path(p)) is hit


def test_default_relative_paths_land_in_onedrive_when_launched_from_there(tmp_path):
    synced = tmp_path / "OneDrive" / "Desktop" / "Quant Agent" / "backend"
    synced.mkdir(parents=True)
    from app.config import Settings
    s = Settings(_env_file=None).model_copy(update={"enable_paper_trading": True})
    probs = live_storage_problems(s, cwd=synced)
    assert [p.split(" ")[0] for p in probs] == ["database_url", "scheduler_db_url", "pit_store_dir"]
    with pytest.raises(RuntimeError, match="云同步目录"):
        assert_live_storage_safe(s, cwd=synced)


def test_local_absolute_paths_pass(tmp_path):
    synced = tmp_path / "OneDrive"
    synced.mkdir()
    s = _settings(tmp_path)
    assert live_storage_problems(s, cwd=synced) == []
    assert_live_storage_safe(s, cwd=synced)


def test_research_mode_is_not_blocked_even_in_onedrive(tmp_path):
    synced = tmp_path / "OneDrive"
    synced.mkdir()
    s = _settings(tmp_path, execution_mode="off", enable_paper_trading=False,
                  database_url="sqlite:///./alphas.db")
    assert not forward_trading_enabled(s)
    assert_live_storage_safe(s, cwd=synced)


@pytest.mark.parametrize("mode,paper,on", [
    ("off", False, False), ("moomoo_paper", False, True), ("off", True, True),
])
def test_forward_trading_enabled(tmp_path, mode, paper, on):
    s = _settings(tmp_path, execution_mode=mode, enable_paper_trading=paper)
    assert forward_trading_enabled(s) is on


def test_startup_refuses_to_serve_with_live_data_in_onedrive(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.config import settings
    from app.main import app
    synced = tmp_path / "OneDrive"
    synced.mkdir()
    monkeypatch.chdir(synced)
    monkeypatch.setattr(settings, "enable_paper_trading", True)
    monkeypatch.setattr(settings, "enable_scheduler", False)
    monkeypatch.setattr(settings, "database_url", "sqlite:///./alphas.db")
    with pytest.raises(RuntimeError, match="云同步目录"):
        with TestClient(app):
            pass
    assert not (synced / "alphas.db").exists()


def test_an_unwritable_database_dir_is_reported(tmp_path):
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    s = _settings(tmp_path, database_url=f"sqlite:///{(blocker / 'db' / 'a.db').as_posix()}")
    out = storage_not_writable(s, cwd=tmp_path)
    assert len(out) == 1 and out[0].startswith("database_url")


def test_a_non_sqlite_database_only_checks_the_pit_dir(tmp_path):
    s = _settings(tmp_path, database_url="postgresql://u@h/db")
    assert storage_not_writable(s, cwd=tmp_path) == []
    assert (tmp_path / "pit").is_dir()


def test_nested_missing_dirs_are_created_not_reported(tmp_path):
    """首次部署时 C:\\QuantAgentData\\pit_store 这类多级目录都还不存在 —— 应当建出来，不是报不可写。"""
    s = _settings(tmp_path, pit_store_dir=str(tmp_path / "x" / "y" / "pit"),
                  database_url=f"sqlite:///{(tmp_path / 'p' / 'q' / 'a.db').as_posix()}")
    assert storage_not_writable(s, cwd=tmp_path) == []
    assert (tmp_path / "x" / "y" / "pit").is_dir() and (tmp_path / "p" / "q").is_dir()


def test_the_writability_probe_leaves_no_files_behind(tmp_path):
    s = _settings(tmp_path)
    assert storage_not_writable(s, cwd=tmp_path) == []
    left = list(tmp_path.rglob(".preflight_*"))
    assert left == [], f"可写性探测在活数据目录里留下了文件：{left}"
