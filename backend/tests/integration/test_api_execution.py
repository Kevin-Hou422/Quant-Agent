"""
test_api_execution.py — Phase 12 执行层 API（状态 / 对账 / 一键全平 / 保真度）+ 启动恢复接线

全平是影响资金的人工动作：两步确认、令牌一次性、令牌绑定动作；
券商不可用时熔断状态也必须先生效（这是熔断最需要起作用的场景）。
"""

from __future__ import annotations

import threading

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.router import get_gateway_factory, get_open_price_loader
from app.core.execution.broker_gateway import BrokerError, LiveTradingRefused
from tests.unit.execution.moomoo_fake import FakeTradeContext, make_gateway


@pytest.fixture
def api(tmp_path, monkeypatch, test_client):
    from app.config import settings
    from app.main import app
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'exec.db'}")
    monkeypatch.setattr(settings, "paper_aum", 100_000.0)
    ctx = FakeTradeContext(cash=60_000.0)
    ctx.set_position("AAPL", 100, 200.0)
    state = {"factory": lambda: make_gateway(ctx)}
    app.dependency_overrides[get_gateway_factory] = lambda: (lambda: state["factory"]())
    yield test_client, ctx, state
    app.dependency_overrides.pop(get_gateway_factory, None)
    app.dependency_overrides.pop(get_open_price_loader, None)


def _engage(client, actor="kevin", reason="drill"):
    tok = client.post("/api/execution/kill_switch/arm", params={"action": "engage"}).json()["token"]
    return client.post("/api/execution/kill_switch/confirm",
                       json={"token": tok, "actor": actor, "reason": reason})


def test_status_on_a_fresh_store(api):
    client, _, _ = api
    r = client.get("/api/execution/status")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "off" and body["kill_switch"] == {"engaged": False}
    assert body["last_snapshot"] is None and body["open_orders"] == []


def test_reconcile_endpoint_returns_broker_truth(api):
    client, ctx, _ = api
    r = client.post("/api/execution/reconcile")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "baseline"
    assert body["positions"]["AAPL"]["qty"] == 100.0
    assert body["account"]["total_assets"] == pytest.approx(80_000.0)
    assert ctx.closed
    st = client.get("/api/execution/status").json()
    assert st["last_snapshot"]["status"] == "baseline"


def test_reconcile_endpoint_error_codes(api):
    client, ctx, state = api
    ctx.fail["accinfo_query"] = "busy"
    assert client.post("/api/execution/reconcile").status_code == 502
    del ctx.fail["accinfo_query"]

    def refuse():
        raise LiveTradingRefused("account 1001 is REAL")
    state["factory"] = refuse
    assert client.post("/api/execution/reconcile").status_code == 400

    def down():
        raise BrokerError("OpenD down")
    state["factory"] = down
    r = client.post("/api/execution/reconcile")
    assert r.status_code == 502 and "OpenD down" in r.json()["detail"]


def test_accept_requires_actor_and_reason(api):
    client, _, _ = api
    assert client.post("/api/execution/reconcile/accept", json={"actor": "kevin"}).status_code == 422
    r = client.post("/api/execution/reconcile/accept", json={"actor": "kevin", "reason": "checked"})
    assert r.status_code == 200 and r.json()["status"] == "accepted"
    ev = client.get("/api/execution/status").json()["events"]
    assert ev[0]["kind"] == "reconcile_accepted" and ev[0]["actor"] == "kevin"


def test_kill_switch_needs_a_valid_one_time_token(api):
    client, ctx, _ = api
    bad = client.post("/api/execution/kill_switch/confirm",
                      json={"token": "not-a-real-token", "actor": "kevin", "reason": "x"})
    assert bad.status_code == 409
    assert ctx.calls_of("place_order") == []
    assert client.post("/api/execution/kill_switch/arm",
                       params={"action": "nuke"}).status_code == 422

    tok = client.post("/api/execution/kill_switch/arm", params={"action": "engage"}).json()["token"]
    ok = client.post("/api/execution/kill_switch/confirm",
                     json={"token": tok, "actor": "kevin", "reason": "drill"})
    assert ok.status_code == 200
    body = ok.json()
    assert body["action"] == "engage" and body["state"]["engaged"] is True
    flat = [kw for kw in ctx.calls_of("place_order")]
    assert [(kw["code"], kw["trd_side"], kw["qty"], kw["price"]) for kw in flat] == [
        ("US.AAPL", "SELL", 100, 190.0)]                          # 200 × (1 − 5%)
    again = client.post("/api/execution/kill_switch/confirm",
                        json={"token": tok, "actor": "kevin", "reason": "drill"})
    assert again.status_code == 409, "令牌被用了两次"
    assert len(ctx.calls_of("place_order")) == 1


def test_expired_token_is_refused(api, monkeypatch):
    client, ctx, _ = api
    import app.api.router as r
    monkeypatch.setattr(r, "_EXEC_TOKEN_TTL_S", -1)
    tok = client.post("/api/execution/kill_switch/arm", params={"action": "engage"}).json()["token"]
    res = client.post("/api/execution/kill_switch/confirm",
                      json={"token": tok, "actor": "kevin", "reason": "x"})
    assert res.status_code == 409 and "过期" in res.json()["detail"]
    assert client.get("/api/execution/status").json()["kill_switch"] == {"engaged": False}


def test_engage_takes_effect_even_when_the_broker_is_down(api):
    client, ctx, state = api

    def down():
        # 连券商这一刻，熔断状态必须**已经**落库（连接本身可能卡住或失败）
        from app.db.execution_store import ExecutionStore
        assert ExecutionStore().kill_switch()["engaged"] is True, "熔断状态没有先于连券商落库"
        raise BrokerError("OpenD down")
    state["factory"] = down
    r = _engage(client)
    assert r.status_code == 200
    assert r.json()["state"]["engaged"] is True and "OpenD down" in r.json()["error"]
    st = client.get("/api/execution/status").json()
    assert st["kill_switch"]["engaged"] is True
    kinds = [e["kind"] for e in st["events"]]
    assert kinds[:2] == ["kill_switch_broker_unavailable", "kill_switch_engaged"]


def test_engage_with_a_reachable_broker_logs_exactly_one_engage_event(api):
    client, ctx, _ = api
    assert _engage(client).status_code == 200
    kinds = [e["kind"] for e in client.get("/api/execution/status").json()["events"]]
    assert kinds.count("kill_switch_engaged") == 1


def test_token_is_valid_up_to_and_including_its_expiry_instant(api, monkeypatch):
    """清理过期令牌与确认令牌必须用同一口径：恰在到期时刻的令牌，确认时仍有效、也不能被清理掉。"""
    client, ctx, _ = api
    import time
    import app.api.router as r
    monkeypatch.setattr(r, "_EXEC_TOKEN_TTL_S", 0)
    monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
    tok = client.post("/api/execution/kill_switch/arm", params={"action": "engage"}).json()["token"]
    client.post("/api/execution/kill_switch/arm", params={"action": "reset"})      # 触发清理
    res = client.post("/api/execution/kill_switch/confirm",
                      json={"token": tok, "actor": "kevin", "reason": "x"})
    assert res.status_code == 200 and res.json()["action"] == "engage"


def test_default_open_price_loader_reads_the_paper_dataset(monkeypatch):
    from types import SimpleNamespace
    from app.config import settings
    import app.core.data_engine.dataset_registry as reg
    seen = {}
    opens = pd.DataFrame({"AAPL": [1.0]})

    def _load(name, start, end, health_check):
        seen.update(name=name, start=start, end=end, health_check=health_check)
        return SimpleNamespace(data={"open": opens, "close": None})
    monkeypatch.setattr(reg, "load_registry_dataset", _load)
    out = get_open_price_loader()("2025-03-01", "2025-04-10")
    assert out is opens
    assert seen == {"name": settings.paper_dataset, "start": "2025-03-01", "end": "2025-04-10",
                    "health_check": False}


def test_token_is_bound_to_its_action(api):
    client, ctx, _ = api
    assert _engage(client).status_code == 200
    tok = client.post("/api/execution/kill_switch/arm", params={"action": "reset"}).json()["token"]
    r = client.post("/api/execution/kill_switch/confirm",
                    json={"token": tok, "actor": "kevin", "reason": "drill over"})
    assert r.status_code == 200 and r.json()["action"] == "reset"
    assert client.get("/api/execution/status").json()["kill_switch"]["engaged"] is False
    assert len(ctx.calls_of("place_order")) == 1, "解除熔断不应触发任何下单"


def test_fidelity_endpoint(api):
    client, _, _ = api
    from app.main import app
    opens = pd.DataFrame({"AAPL": [200.0]}, index=pd.DatetimeIndex(["2025-03-04"]))
    seen = []
    app.dependency_overrides[get_open_price_loader] = lambda: (
        lambda s, e: seen.append((s, e)) or opens)
    r = client.get("/api/execution/fidelity", params={"start": "2025-03-01", "end": "2025-03-31"})
    assert r.status_code == 200
    body = r.json()
    assert body["n_live_orders"] == 0 and body["permanent_impact_bps"] is None
    assert "保真度报告" in body["markdown"]
    assert seen == [("2025-03-01", "2025-04-10")], "开盘价要多取 10 天（月末决策日次月开盘成交）"

    assert client.get("/api/execution/fidelity",
                      params={"start": "nope", "end": "2025-03-31"}).status_code == 400
    assert client.get("/api/execution/fidelity",
                      params={"start": "2025-03-31", "end": "2025-03-01"}).status_code == 400
    assert client.get("/api/execution/fidelity",
                      params={"start": "2025-03-31", "end": "2025-03-31"}).status_code == 200

    def boom(s, e):
        raise RuntimeError("yfinance down")
    app.dependency_overrides[get_open_price_loader] = lambda: boom
    assert client.get("/api/execution/fidelity",
                      params={"start": "2025-03-01", "end": "2025-03-31"}).status_code == 502


def test_trading_status_shows_execution_mode(api):
    client, _, _ = api
    assert client.get("/api/trading/status").json()["execution_mode"] == "off"


def test_startup_recovery_runs_when_execution_is_enabled(monkeypatch):
    """§K：启动恢复写了就得真的在启动时被调用。"""
    from app.config import settings
    from app.main import app
    import app.core.execution.order_manager as om
    called = threading.Event()
    daemon = []

    def _recover(*a, **k):
        # 必须是守护线程：OpenD 卡住时它不能拖住服务退出
        daemon.append(threading.current_thread().daemon)
        called.set()
    monkeypatch.setattr(om, "recover_on_startup", _recover)
    monkeypatch.setattr(settings, "execution_mode", "moomoo_paper")
    with TestClient(app):
        assert called.wait(5.0), "执行层开启时启动恢复没有被调用"
    assert daemon == [True]

    called.clear()
    monkeypatch.setattr(settings, "execution_mode", "off")
    with TestClient(app):
        assert not called.wait(0.5), "执行层关闭时不该连券商"


def test_preflight_endpoint_reports_each_check(api, tmp_path, monkeypatch):
    from app.api.router import get_opend_probe
    from app.config import settings
    from app.main import app
    client, ctx, _ = api
    monkeypatch.setattr(settings, "execution_mode", "moomoo_paper")
    monkeypatch.setattr(settings, "enable_scheduler", True)
    monkeypatch.setattr(settings, "enable_paper_trading", True)
    monkeypatch.setattr(settings, "scheduler_db_url", f"sqlite:///{(tmp_path / 's.db').as_posix()}")
    monkeypatch.setattr(settings, "pit_store_dir", str(tmp_path / "pit"))
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{(tmp_path / 'exec.db').as_posix()}")
    import app.core.execution.golive as gl
    monkeypatch.setattr(settings, "price_source", "moomoo")
    monkeypatch.setattr(gl, "_probe_data", lambda s, t: (True, f"bar {t.date()}"))
    monkeypatch.setattr(gl, "_active_strategy_problem", lambda s: (True, "active #1"))
    app.dependency_overrides[get_opend_probe] = lambda: (lambda h, p: True)
    try:
        body = client.post("/api/execution/preflight").json()
        checks = {c["name"]: c for c in body["checks"]}
        assert body["ready"] is True and body["connected"] is True, \
            [c for c in body["checks"] if not c["ok"]]
        assert checks["scheduler_jobs"]["ok"]         # 真的按当前配置构建了调度器
        assert checks["account_funds"]["ok"]          # 60k 现金 + 20k 持仓 vs 100k：偏离 20%
        assert body["reconcile"]["status"] == "baseline"
        assert ctx.calls_of("place_order") == [] and ctx.closed

        app.dependency_overrides[get_opend_probe] = lambda: (lambda h, p: False)
        body = client.post("/api/execution/preflight").json()
        checks = {c["name"]: c for c in body["checks"]}
        assert body["ready"] is False and body["connected"] is False
        assert not checks["opend"]["ok"] and "broker_account" not in checks
    finally:
        app.dependency_overrides.pop(get_opend_probe, None)
