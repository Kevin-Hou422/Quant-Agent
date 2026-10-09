"""
test_forward_cli.py — `python -m app.tasks.forward preflight | run-now`

run-now 会真的下纸交易单，所以：预检有阻塞项就**一定不跑**管线，退出码非 0；
跑的必须是调度器用的同一个 `daily_trading_job`，不是另一条路径。

审计 F14：以前只要预检过了 run-now 一律返回 0 —— 执行被拦、下单异常、组合报错、摄取被拒
都被外部监控当成成功。现在退出码按**真实结果**给，每种未完成各有一条用例。
"""

from __future__ import annotations

import json

import pytest

from app.core.execution.golive import LAYER_CONNECT, LAYER_TRADE, PreflightReport
from app.tasks import forward


def _rep(*checks):
    r = PreflightReport(checked_at="2025-03-04T22:00:00+00:00")
    for name, ok, blocking, layer in checks:
        r.add(name, ok, f"{name} detail", blocking, layer)
    return r


READY = _rep(("opend", True, True, LAYER_CONNECT), ("price_source", False, False, LAYER_TRADE))
NOT_READY = _rep(("opend", True, True, LAYER_CONNECT), ("strategy", False, True, LAYER_TRADE))
NOT_CONNECTED = _rep(("opend", False, True, LAYER_CONNECT), ("strategy", True, True, LAYER_TRADE))

OK_RUN = {"ingest_accepted": True, "mode": "incremental", "forward_from": "2025-03-04",
          "n_new_bars": 1,
          "portfolio": {"execution": {"mode": "moomoo_paper", "blocked": "", "n_planned": 4,
                                      "n_submitted": 3, "n_gate_rejected": 1, "n_duplicate": 0,
                                      "n_submit_errors": 0}}}


def _body(printed):
    return json.loads(printed[printed.index("{"):printed.rindex("}") + 1])


def test_preflight_reports_both_layers(capsys):
    assert forward.preflight(run=lambda: READY) == forward.EXIT_OK
    out = capsys.readouterr().out
    assert "[OK  ] connect opend" in out and "[WARN] trade   price_source" in out
    assert "券商侧：可用" in out and "交易就绪：是" in out

    assert forward.preflight(run=lambda: NOT_READY) == forward.EXIT_NOT_READY
    out = capsys.readouterr().out
    assert "[FAIL] trade   strategy" in out and "券商侧：可用" in out and "交易就绪：否" in out

    assert forward.preflight(run=lambda: NOT_CONNECTED) == forward.EXIT_NOT_READY
    assert "券商侧：不可用" in capsys.readouterr().out


def test_run_now_refuses_without_a_clean_preflight(capsys):
    called = []
    code = forward.run_now(run=lambda: NOT_READY, job=lambda: called.append(1) or {})
    assert code == forward.EXIT_NOT_READY and called == []
    assert "拒绝执行 run-now" in capsys.readouterr().out


def test_run_now_succeeds_only_when_execution_completed(capsys):
    assert forward.run_now(run=lambda: READY, job=lambda: OK_RUN) == forward.EXIT_OK
    printed = capsys.readouterr().out
    assert _body(printed)["execution"]["n_submitted"] == 3
    assert _body(printed)["ingest_accepted"] is True and _body(printed)["forward_from"] == "2025-03-04"
    assert "执行完成：计划 4 / 已下 3 / 风控拒 1 / 重复 0" in printed


@pytest.mark.parametrize("out,needle", [
    ({"skipped": "calendar_unavailable", "error": "no calendar"}, "任务未执行：calendar_unavailable"),
    ({"ingest_accepted": False, "reject_reason": "load_failed: boom"}, "摄取被拒：load_failed: boom"),
    ({"ingest_accepted": True, "portfolio": {"error": "kaboom"}}, "组合账本失败：kaboom"),
    ({"ingest_accepted": True, "portfolio": {}}, "执行层没有运行"),
    ({"ingest_accepted": True, "portfolio": {"execution": {"mode": "off"}}}, "EXECUTION_MODE=off"),
    ({"ingest_accepted": True,
      "portfolio": {"execution": {"blocked": "gateway_unavailable: OpenD down"}}},
     "执行被拦：gateway_unavailable: OpenD down"),
    ({"ingest_accepted": True,
      "portfolio": {"execution": {"blocked": "", "n_submit_errors": 2}}}, "2 张单下单异常"),
    ({"ingest_accepted": False, "reason": "no_new_bar",
      "portfolio": {"execution": {"blocked": "stale_decision_date"}}}, "执行被拦：stale_decision_date"),
])
def test_every_unfinished_outcome_is_a_failure_exit(capsys, out, needle):
    assert forward.run_now(run=lambda: READY, job=lambda: out) == forward.EXIT_RUN_FAILED
    printed = capsys.readouterr().out
    assert "失败：" in printed and needle in printed


def test_a_planned_no_trade_day_is_success():
    code, verdict = forward.classify_run({"skipped": "not_a_trading_day", "date": "2025-03-08"})
    assert code == forward.EXIT_OK and "非美股交易日" in verdict


def test_a_no_new_bar_rerun_that_completed_execution_is_success():
    out = {"ingest_accepted": False, "reason": "no_new_bar", **{"portfolio": OK_RUN["portfolio"]}}
    assert forward.classify_run(out)[0] == forward.EXIT_OK


def test_portfolio_error_is_printed(capsys):
    forward.run_now(run=lambda: READY,
                    job=lambda: {"ingest_accepted": True, "portfolio": {"error": "kaboom"}})
    assert _body(capsys.readouterr().out)["portfolio_error"] == "kaboom"


def test_run_now_prints_reasons_readably(capsys):
    """拒单原因是中文；转义成 \\uXXXX 的话，人在终端里看不懂当天为什么没下单。"""
    out = {"portfolio": {"execution": {"blocked": "券商不可用"}}}
    forward.run_now(run=lambda: READY, job=lambda: out)
    assert '"blocked": "券商不可用"' in capsys.readouterr().out


def test_run_now_defaults_to_the_scheduler_job(monkeypatch):
    import app.tasks.scheduler as sched
    called = []
    monkeypatch.setattr(sched, "daily_trading_job", lambda: called.append(1) or OK_RUN)
    assert forward.run_now(run=lambda: READY) == forward.EXIT_OK
    assert called == [1]


def test_main_dispatches_and_rejects_unknown_commands(monkeypatch):
    monkeypatch.setattr(forward, "preflight", lambda: 7)
    monkeypatch.setattr(forward, "run_now", lambda: 9)
    assert forward.main(["preflight"]) == 7
    assert forward.main(["run-now"]) == 9
    with pytest.raises(SystemExit):
        forward.main(["trade-real-money"])
