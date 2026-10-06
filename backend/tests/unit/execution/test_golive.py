"""
test_golive.py — 前向交易上线预检 + 活数据位置启动保险

两件事要钉死：
  1. 预检的每一项都**真的会失败**（不是永远 OK 的装饰），且失败后不会继续做危险动作
     （OpenD 不在就不建连；任何情况下都不下单）。
  2. 云同步目录的判断看**解析后的路径**：默认配置 `sqlite:///./alphas.db` 字符串里没有
     "onedrive"，但从 OneDrive 里的仓库启动时它就在 OneDrive —— 只查字符串的旧检查
     （TestLessonQ）对这个情形是瞎的。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.core.execution.golive import (
    assert_live_storage_safe, forward_trading_enabled, in_cloud_sync, live_storage_problems,
    run_preflight, sqlite_path,
)
from app.db.execution_store import ExecutionStore
from app.db.position_store import PositionStore
from tests.unit.execution.moomoo_fake import FakeTradeContext, make_gateway

NOW = datetime(2025, 3, 4, 22, 0, tzinfo=timezone.utc)       # 周二，收盘后


def _settings(tmp_path, **kw):
    from app.config import settings
    base = dict(
        execution_mode="moomoo_paper", enable_scheduler=True, enable_paper_trading=True,
        database_url=f"sqlite:///{(tmp_path / 'a.db').as_posix()}",
        scheduler_db_url=f"sqlite:///{(tmp_path / 's.db').as_posix()}",
        pit_store_dir=str(tmp_path / "pit"), paper_aum=1_000_000.0,
        exec_max_aum_mismatch=0.5, moomoo_host="127.0.0.1", moomoo_port=11111,
    )
    base.update(kw)
    return settings.model_copy(update=base)


def _env(tmp_path, cash=1_000_000.0, **kw):
    ctx = FakeTradeContext(cash=cash)
    es = ExecutionStore(db_url=f"sqlite:///{tmp_path / 'e.db'}")
    ps = PositionStore(db_url=f"sqlite:///{tmp_path / 'p.db'}")
    calls = []

    def factory():
        calls.append(1)
        return make_gateway(ctx)

    s = _settings(tmp_path, **kw)
    run = lambda probe=lambda h, p: True: run_preflight(               # noqa: E731
        s, gateway_factory=factory, probe=probe, store=es, position_store=ps, now=NOW,
        cwd=tmp_path)
    return ctx, es, ps, calls, run


def _by_name(rep):
    return {c.name: c for c in rep.checks}


# ---------------------------------------------------------------------------
# 预检
# ---------------------------------------------------------------------------

def test_everything_in_place_is_ready_and_places_no_order(tmp_path):
    ctx, es, _, _, run = _env(tmp_path)
    rep = run()
    assert [c.name for c in rep.checks] == [
        "execution_mode", "scheduler", "storage", "calendar", "opend", "broker_account",
        "account_funds", "kill_switch", "reconcile"]
    assert rep.ready and all(c.ok for c in rep.checks)
    assert rep.gateway["acc_id"] == 2001 and rep.gateway["trd_env"] == "SIMULATE"
    assert rep.account["total_assets"] == pytest.approx(1_000_000.0)
    assert rep.reconcile["status"] == "baseline"
    assert _by_name(rep)["calendar"].detail.endswith("2025-03-04")
    assert ctx.calls_of("place_order") == []
    assert ctx.closed
    d = rep.to_dict()
    assert d["ready"] is True and len(d["checks"]) == 9


def test_opend_down_stops_before_any_connection(tmp_path):
    _, _, _, calls, run = _env(tmp_path)
    rep = run(probe=lambda h, p: False)
    assert not rep.ready
    assert rep.checks[-1].name == "opend" and not rep.checks[-1].ok
    assert "未在监听" in rep.checks[-1].detail
    assert calls == []                      # SDK 在网关不在时会无限阻塞 —— 绝不能去建连
    assert rep.gateway is None


def test_probe_receives_the_configured_address(tmp_path):
    seen = []
    _, _, _, _, run = _env(tmp_path, moomoo_host="10.0.0.5", moomoo_port=22222)
    run(probe=lambda h, p: seen.append((h, p)) or True)
    assert seen == [("10.0.0.5", 22222)]


def test_wrong_mode_and_disabled_scheduler_are_blocking_but_broker_is_still_checked(tmp_path):
    _, _, _, calls, run = _env(tmp_path, execution_mode="off", enable_scheduler=False)
    rep = run()
    c = _by_name(rep)
    assert not rep.ready
    assert not c["execution_mode"].ok and "moomoo_paper" in c["execution_mode"].detail
    assert not c["scheduler"].ok
    assert c["reconcile"].ok and calls == [1]


@pytest.mark.parametrize("sched,paper", [(True, False), (False, True)])
def test_scheduler_needs_both_switches(tmp_path, sched, paper):
    _, _, _, _, run = _env(tmp_path, enable_scheduler=sched, enable_paper_trading=paper)
    assert not _by_name(run())["scheduler"].ok


@pytest.mark.parametrize("cash,ok", [
    (1_500_000.0, True),       # 偏离恰好 50% —— 上限是闭区间
    (500_000.0, True),
    (1_500_001.0, False),
    (100_000.0, False),
])
def test_account_must_match_paper_aum(tmp_path, cash, ok):
    _, _, _, _, run = _env(tmp_path, cash=cash)
    c = _by_name(run())["account_funds"]
    assert c.ok is ok
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
    assert not rep.ready
    assert not c["account_funds"].ok and "读不到账户资金" in c["account_funds"].detail
    assert not c["reconcile"].ok and "对账失败" in c["reconcile"].detail
    assert ctx.closed


def test_gateway_failure_is_reported(tmp_path):
    s = _settings(tmp_path)

    def boom():
        raise RuntimeError("login required")

    rep = run_preflight(s, gateway_factory=boom, probe=lambda h, p: True, now=NOW, cwd=tmp_path)
    c = _by_name(rep)["broker_account"]
    assert not rep.ready and not c.ok and "login required" in c.detail
    assert rep.checks[-1].name == "broker_account"


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


def test_calendar_failure_is_blocking(tmp_path, monkeypatch):
    import app.core.execution.order_manager as om

    def boom(now):
        raise RuntimeError("calendar gone")

    monkeypatch.setattr(om, "last_closed_session", boom)
    _, _, _, _, run = _env(tmp_path)
    c = _by_name(run())["calendar"]
    assert not c.ok and "calendar gone" in c.detail


def test_storage_in_cloud_sync_is_blocking(tmp_path):
    synced = tmp_path / "OneDrive" / "repo"
    synced.mkdir(parents=True)
    s = _settings(tmp_path, database_url="sqlite:///./alphas.db")
    rep = run_preflight(s, probe=lambda h, p: False, now=NOW, cwd=synced)
    c = _by_name(rep)["storage"]
    assert not c.ok and "database_url" in c.detail


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
