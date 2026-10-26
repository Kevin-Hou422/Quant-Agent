"""
运行许可：所有新增风险的入口都消费同一份、每轮重新核对的许可（2026-10-25 审计 R3）。

审计复现："将通知就绪设为 false，直接调用 daily_trading_job，仍会进入交易任务主体。"
验收："绕过包装器直接启动服务、直接调用调度任务、重启恢复、许可过期/账户变更/通知未确认/输入源变更都不得增加风险；
只读服务和安全维护仍能工作。就绪结论不能只在部署当时计算一次。"
"""

from __future__ import annotations

import ast
import json
from datetime import timedelta
from pathlib import Path

import pytest

from app.core.execution import run_permit as rp
from tests.unit.execution.test_order_manager import D1, D2, D3, _cycle, _env, _placed

ACC = 2001
BACKEND = Path(__file__).resolve().parents[2]


def _settings(**kw):
    from app.config import settings
    return settings.model_copy(update={"price_source": "moomoo", **kw})


def _guarded(tmp_path, settings=None):
    e = _env(tmp_path)
    e.settings = settings or _settings()
    e.om.permit_check = rp.settings_check(e.settings)
    return e


def _supervised(e, hours=8.0, cap=1e9, account=ACC):
    return rp.issue(e.es, mode="supervised", actor="tester", reason="联调", account=account, hours=hours,
                    max_order_notional=cap, settings=e.settings, now=e.clock())


def test_without_a_permit_no_order_is_sent_but_reconciliation_still_runs(tmp_path):
    e = _guarded(tmp_path)
    rep = _cycle(e, D1)
    assert rep.blocked.startswith("run_permit:") and "没有运行许可" in rep.blocked
    assert _placed(e) == [] and rep.reconcile["status"] == "baseline"


def test_a_supervised_permit_allows_trading_within_its_order_cap(tmp_path):
    e = _guarded(tmp_path)
    _supervised(e, cap=9_000.0)
    rep = _cycle(e, D1)
    assert rep.blocked == "" and rep.permit["allowed"] is True and rep.permit["mode"] == "supervised"
    assert sorted(t for t, *_ in _placed(e)) == ["US.BBB", "US.CCC"]          # AAA 200 × 50.25 > 9000
    assert [r["reason"] for r in rep.gate["rejected"]] == ["permit_notional_limit"]


@pytest.mark.parametrize("change,needle", [
    ("expire", "过期"), ("account", "账户变了"), ("source", "价格源变了"), ("code", "代码版本变了")])
def test_a_permit_is_rechecked_every_cycle(tmp_path, monkeypatch, change, needle):
    e = _guarded(tmp_path)
    _supervised(e, hours=6.0, account=9999 if change == "account" else ACC)
    if change == "expire":
        e.clock.t += timedelta(hours=7)
    elif change == "source":
        e.om.permit_check = rp.settings_check(_settings(price_source="yahoo"))
    elif change == "code":
        import app.core.data_engine.input_identity as ii
        monkeypatch.setattr(ii, "code_version", lambda: "0000000000ab")
    rep = _cycle(e, D2 if change == "expire" else D1)
    assert rep.blocked.startswith("run_permit:") and needle in rep.blocked, rep.blocked
    assert _placed(e) == []


def test_a_restarted_process_reads_the_same_permit_and_a_revoke_stops_it(tmp_path):
    from app.core.execution.order_manager import OrderManager
    e = _guarded(tmp_path)
    _supervised(e)
    fresh = OrderManager(e.gw, e.es, e.ps, limits=e.om.limits, paper_aum=e.om.paper_aum, limit_band_bps=50.0,
                         flatten_band_bps=500.0, clock=e.clock, permit_check=rp.settings_check(e.settings))
    assert rp.evaluate(fresh.store, account=ACC, settings=e.settings, now=e.clock()).allowed
    rp.revoke(e.es, actor="tester", reason="收工")
    e.om = fresh
    assert _cycle(e, D1).blocked.startswith("run_permit:")
    kinds = [ev.kind for ev in e.es.events(10)]
    assert "run_permit_issue" in kinds and "run_permit_revoke" in kinds


def test_flattening_under_the_kill_switch_does_not_need_a_permit(tmp_path):
    e = _guarded(tmp_path)
    _supervised(e)
    _cycle(e, D1)
    e.ctx.now = "2025-03-04 09:30:01.000"
    e.ctx.open_session({"AAA": 50.2, "BBB": 20.05, "CCC": 10.02})
    rp.revoke(e.es, actor="tester", reason="许可收回")
    n = len(_placed(e))
    e.om.engage_kill_switch("tester", "演练")                              # 开熔断即撤单 + 全平，不看许可
    assert len(_placed(e)) > n and all(side == "SELL" for _, side, *_ in _placed(e)[n:])
    rep = _cycle(e, D2, px={"AAA": 51.0, "BBB": 20.0, "CCC": 10.0})       # 之后的周期：续做全平，不调仓
    assert rep.blocked == "kill_switch_engaged" and rep.flatten is not None


# ── 无人值守 ─────────────────────────────────────────────

def _evidence(tmp_path):
    p = tmp_path / "simulate_acceptance.jsonl"
    p.write_text('{"scenario": "submit_and_fill", "step": "verdict"}\n', encoding="utf-8")
    return p


def _unattended(e, tmp_path, health=lambda s, now: [], **kw):
    kw.setdefault("code", "abc123abc123")
    kw.setdefault("spec_hash", "h" * 64)
    return rp.issue(e.es, mode="unattended", actor="tester", reason="无人值守", account=ACC, hours=48,
                    acceptance_evidence=str(_evidence(tmp_path)), settings=e.settings, now=e.clock(),
                    health=health, **kw)


@pytest.mark.parametrize("kw,needle", [
    ({"code": "abc123abc123+dirty"}, "固定的代码版本"), ({"code": "unknown"}, "固定的代码版本"),
    ({"spec_hash": ""}, "规格"), ({"health": lambda s, now: ["通知未就绪：没确认过"]}, "通知未就绪")])
def test_an_unattended_permit_is_refused_unless_every_binding_holds(tmp_path, kw, needle):
    e = _guarded(tmp_path)
    with pytest.raises(rp.PermitRefused, match=needle):
        _unattended(e, tmp_path, **kw)
    assert rp.current(e.es) is None


def test_an_unattended_permit_needs_acceptance_evidence(tmp_path):
    e = _guarded(tmp_path)
    with pytest.raises(rp.PermitRefused, match="验收证据"):
        rp.issue(e.es, mode="unattended", actor="t", reason="r", account=ACC, hours=48, settings=e.settings,
                 now=e.clock(), code="abc123abc123", spec_hash="h", health=lambda s, now: [])


def test_unattended_readiness_is_rechecked_not_computed_once(tmp_path):
    """签发时通知就绪；之后通知失败 / 验收证据被改 / 规格变了 → 下一次核对就不通过。"""
    e = _guarded(tmp_path)
    _unattended(e, tmp_path)
    base = dict(account=ACC, settings=e.settings, now=e.clock(), code="abc123abc123", spec_hash="h" * 64)
    assert rp.evaluate(e.es, health=lambda s, now: [], **base).allowed
    d = rp.evaluate(e.es, health=lambda s, now: ["通知未就绪：确认之后有 1 条通知失败"], **base)
    assert not d.allowed and "通知未就绪" in d.reasons[0]
    assert not rp.evaluate(e.es, health=lambda s, now: [], **{**base, "spec_hash": "x" * 64}).allowed
    _evidence(tmp_path).write_text("改过\n", encoding="utf-8")
    d = rp.evaluate(e.es, health=lambda s, now: [], **base)
    assert not d.allowed and any("验收证据" in r for r in d.reasons)


def test_the_real_health_check_reads_notifications_and_backups(tmp_path):
    """审计复现的反面：通知没确认过 → 无人值守许可不能签、已签的也核对不过。"""
    e = _guarded(tmp_path, _settings(alert_event_file=str(tmp_path / "alerts.jsonl"), backup_dir=str(tmp_path / "bk")))
    problems = rp._unattended_health(e.settings, e.clock())                     # noqa: SLF001
    assert any("通知未就绪" in p for p in problems) and any("备份不健康" in p for p in problems)


# ── 入口 ─────────────────────────────────────────────────

def test_the_production_constructor_always_installs_the_permit_check(tmp_path):
    from app.core.execution.order_manager import OrderManager
    e = _env(tmp_path)
    assert OrderManager.from_settings(e.gw, e.es, e.ps).permit_check is not None


def test_no_production_code_builds_an_order_manager_that_skips_the_permit():
    """生产代码里 OrderManager 只经 from_settings 构造（预检的那一处只对账、不调仓）。"""
    direct = []
    for p in (BACKEND / "app").rglob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "OrderManager":
                direct.append(f"{p.relative_to(BACKEND).as_posix()}:{n.lineno}")
    assert [d.split(":")[0] for d in direct] == ["app/core/execution/golive.py"], direct
    src = (BACKEND / "app" / "core" / "execution" / "golive.py").read_text(encoding="utf-8")
    call = src[src.index("OrderManager("):]
    assert ".reconcile(" in call[:600] and ".run_cycle(" not in call[:600]


def test_the_command_line_issues_shows_and_revokes(tmp_path, capsys):
    from app.tasks import run_permit as cli
    db = (tmp_path / "e.db").as_posix()
    assert cli.main(["issue", "--exec-db", db, "--mode", "supervised", "--account", str(ACC), "--hours", "2",
                     "--actor", "me", "--reason", "联调"]) == cli.EXIT_REFUSED            # 没给单笔上限
    assert cli.main(["issue", "--exec-db", db, "--mode", "supervised", "--account", str(ACC), "--hours", "2",
                     "--max-order-notional", "2000", "--actor", "me", "--reason", "联调"]) == cli.EXIT_OK
    capsys.readouterr()
    assert cli.main(["status", "--exec-db", db]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["permit"]["max_order_notional"] == 2000.0
    assert cli.main(["revoke", "--exec-db", db, "--actor", "me", "--reason", "收工"]) == cli.EXIT_OK


def test_an_expired_permit_after_a_long_stop_blocks_the_next_day(tmp_path):
    """停机重启：许可持久化在执行库里，重启后按此刻重新核对 —— 过期就不交易。"""
    e = _guarded(tmp_path)
    _supervised(e, hours=1.0)
    e.clock.after_close(D3)
    assert not rp.evaluate(e.es, account=ACC, settings=e.settings, now=e.clock()).allowed
