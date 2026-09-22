"""
test_phase_s_holdout_discipline.py — Phase S.2 端到端：冻结段真的没被看

这一组用例的写法刻意反直觉：数据被**故意造成**"前半段没信号、后半段极好"。
如果门/端点偷看了冻结段，它们会表现得更好 —— 于是"表现得好"在这里恰恰是
失败信号。只断言"跑通了"的用例抓不到这种偷看。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.lifecycle.validation_gate import ValidationGate


@pytest.fixture(scope="module")
def client():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _ok(resp, expect: int = 200):
    assert resp.status_code == expect, f"期望 {expect} 实际 {resp.status_code}｜{resp.text[:500]}"
    return resp.json()


_MOMENTUM = "rank(ts_mean(returns,20))"


def _late_bloomer(T=1400, N=20, seed=0) -> dict:
    """
    前 ~65% 是纯噪声（动量因子在这段毫无优势），后 ~35% 有强横截面漂移。
    三段切割会把"好的那一段"整个冻起来 —— 门应当只看到噪声。
    """
    rng = np.random.default_rng(seed)
    q = np.linspace(-1, 1, N)
    cut = int(T * 0.65)
    ret = np.vstack([
        rng.normal(0, 0.01, (cut, N)),
        0.004 * q[None, :] + rng.normal(0, 0.006, (T - cut, N)),
    ])
    idx = pd.bdate_range("2017-01-02", periods=T)
    cols = [f"A{i:02d}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=idx, columns=cols)
    # 日内区间用随机幅度（DEV_LESSONS §O）：固定 ±1% 会让 Corwin-Schultz
    # 把价差估到 54bps，足以在成本侧改变结论。
    hi = close * (1 + rng.uniform(0.001, 0.008, close.shape))
    lo = close * (1 - rng.uniform(0.001, 0.008, close.shape))
    return {
        "open": close, "high": hi, "low": lo, "close": close,
        "vwap": close, "volume": pd.DataFrame(1e6, index=idx, columns=cols),
        "returns": np.log(close / close.shift(1)),
    }


# ===========================================================================
# A. 验证门看不到冻结段
# ===========================================================================

class TestTheGateCannotSeeTheFrozenSegment:

    def test_a_signal_that_only_works_in_the_frozen_tail_does_not_pass(self):
        ds = _late_bloomer()
        res = ValidationGate(n_splits=3, embargo_days=5,
                             dsr_threshold=0.0, min_tstat=0.0).evaluate(
            _MOMENTUM, ds, n_trials=1)
        assert res.passed is False, (
            "因子只在被冻结的那一段有效，门却判通过 —— 说明门看到了 Test 段")
        assert res.partition and res.partition.get("n_test", 0) > 0, (
            f"门没有做三段切割：partition={res.partition}")

    def test_turning_the_split_off_changes_the_verdict_inputs(self):
        """
        反向对照。没有这一条，上一条也可能是"这个因子本来就不行"造成的假通过。
        关掉三段切割后门能看到好数据，WalkForward 的数字必须因此变好。
        """
        ds = _late_bloomer()
        strict = ValidationGate(n_splits=3, embargo_days=5, dsr_threshold=0.0,
                                min_tstat=0.0, three_way=True).evaluate(_MOMENTUM, ds, n_trials=1)
        loose = ValidationGate(n_splits=3, embargo_days=5, dsr_threshold=0.0,
                               min_tstat=0.0, three_way=False).evaluate(_MOMENTUM, ds, n_trials=1)
        assert loose.mean_oos_sharpe > strict.mean_oos_sharpe, (
            f"看得见冻结段（{loose.mean_oos_sharpe:.3f}）竟不优于看不见"
            f"（{strict.mean_oos_sharpe:.3f}）—— 两条路径大概率跑的是同一份数据")
        assert loose.partition is None, "关掉三段切割后仍报告了切割口径"

    def test_the_gate_does_not_touch_the_holdout_unless_asked(self):
        res = ValidationGate(n_splits=3, embargo_days=5).evaluate(
            _MOMENTUM, _late_bloomer(), n_trials=1)
        assert res.held_out_test is None, (
            "没人要求汇报冻结段，门却算了 —— 每算一次就是消耗一次一次性预算")

    def test_reporting_on_the_holdout_is_accounted_for(self):
        ds = _late_bloomer()
        gate = ValidationGate(n_splits=3, embargo_days=5)
        first = gate.evaluate(_MOMENTUM, ds, n_trials=1,
                              dataset_key="phase_s_gate", report_test=True)
        assert first.held_out_test is not None, "要求汇报冻结段却没有结果"
        assert first.held_out_test["uses"] >= 1
        assert first.held_out_test["recorded"] is True

        second = gate.evaluate(_MOMENTUM, ds, n_trials=1,
                               dataset_key="phase_s_gate", report_test=True)
        assert second.held_out_test["uses"] > first.held_out_test["uses"], (
            "同一段 holdout 被看了第二次，计数却没涨")
        assert second.held_out_test["over_budget"] is True, (
            "第二次动用一次性 holdout 没有标 over_budget")

    def test_the_frozen_segment_is_where_the_signal_actually_lives(self):
        """
        把构造本身钉住：冻结段上这个因子**确实**很好。
        否则上面那条"门判不过"可能只是因为因子根本没用，测了个寂寞。
        """
        res = ValidationGate(n_splits=3, embargo_days=5).evaluate(
            _MOMENTUM, _late_bloomer(), n_trials=1,
            dataset_key="phase_s_contrast", report_test=True)
        assert res.held_out_test.get("test_sharpe", 0) > 0.5, (
            f"构造失败：冻结段上的夏普只有 {res.held_out_test.get('test_sharpe')}，"
            f"这组用例的前提不成立")


# ===========================================================================
# B. /api/backtest/* 全路径
# ===========================================================================

_LONG_SYNTH = {"dsl": _MOMENTUM, "dataset_name": "", "n_tickers": 10,
               "n_days": 1000, "seed": 11}


class TestEveryBacktestRouteReportsItsPartition:

    @pytest.mark.parametrize("path,payload", [
        ("/api/backtest/run", _LONG_SYNTH),
        ("/api/backtest/realistic", {**_LONG_SYNTH, "oos_ratio": 0.3}),
        ("/api/alpha/simulate", {**_LONG_SYNTH, "oos_ratio": 0.3}),
    ])
    def test_the_response_says_which_segment_the_numbers_came_from(self, client, path, payload):
        body = _ok(client.post(path, json=payload))
        part = body.get("partition")
        assert part, f"{path} 的响应里没有 partition —— 调用方无从知道这些数字是哪段算的"
        assert part.get("three_way") is True, f"{path} 没有走三段切割：{part}"
        assert part.get("n_test", 0) > 0, f"{path} 切出来的 Test 段是空的：{part}"

    def test_a_panel_too_short_to_split_says_so_instead_of_pretending(self, client):
        """
        切不动的时候**不许静默**：`three_way=False` 加一条 reason 必须出现在
        响应里。这里的坏结果不是报错，而是"看起来照常返回了一份报告，
        而那份报告其实包含了本该冻结的数据"，调用方无从分辨。
        """
        body = _ok(client.post("/api/backtest/run", json={
            "dsl": _MOMENTUM, "dataset_name": "", "n_tickers": 6,
            "n_days": 60, "seed": 3,
        }))
        part = body["partition"]
        assert part.get("three_way") is False, (
            f"60 天的面板切不出三段，却报告 three_way={part.get('three_way')}：{part}")
        assert part.get("reason"), "退回全量面板时没有给出原因"
        assert body.get("held_out_test") is None

    def test_turning_the_split_off_is_reported_as_such(self, client, monkeypatch):
        """
        `s_three_way_enabled=False` 是个**逃生口**：关掉之后所有数字都退回
        "含冻结段"的口径。它必须在响应里自报 —— 否则关了开关的人和没关的人
        拿到的 JSON 长得一模一样，而含义完全不同。

        conftest 之外的 settings 覆盖（DEV_LESSONS §H 要求逐处说明）：
        这里**故意**覆盖发布默认值，因为被测的正是"关掉之后会怎样"。
        """
        from app.config import settings
        monkeypatch.setattr(settings, "s_three_way_enabled", False, raising=False)
        body = _ok(client.post("/api/backtest/run", json={**_LONG_SYNTH, "seed": 5150}))
        part = body["partition"]
        assert part.get("three_way") is False, (
            f"开关已关掉，响应却仍报告 three_way={part.get('three_way')}")
        assert part.get("reason") == "disabled_by_config", (
            f"没说清是被开关关掉的（reason={part.get('reason')!r}）—— "
            f"与『数据太短切不动』是两回事，处置方式也不同")

    def test_walk_forward_reports_its_partition_too(self, client):
        body = _ok(client.post("/api/backtest/walk_forward", json={
            "dsl": _MOMENTUM, "dataset_name": "", "n_splits": 3,
            "embargo_days": 5, "n_tickers": 10, "n_days": 1000, "seed": 11,
        }))
        assert body.get("partition", {}).get("three_way") is True, (
            f"walk_forward 未报告三段切割：{body.get('partition')}")

    def test_the_folds_stay_inside_the_selection_segment(self, client):
        """
        WF 各折用掉的天数不得超过 selection 段 —— 超了就说明它在全量数据上滚。
        """
        body = _ok(client.post("/api/backtest/walk_forward", json={
            "dsl": _MOMENTUM, "dataset_name": "", "n_splits": 3,
            "embargo_days": 5, "n_tickers": 10, "n_days": 1000, "seed": 11,
        }))
        part = body["partition"]
        selection_days = part["n_is"] + part["embargo_days"] + part["n_val"]
        last_fold_span = max(f["is_days"] + f["oos_days"] for f in body["fold_reports"])
        assert last_fold_span <= selection_days, (
            f"某一折用了 {last_fold_span} 天，超过 selection 段的 {selection_days} 天")


class TestHoldoutReportingIsOptInAndAccounted:

    @pytest.mark.parametrize("path,payload", [
        ("/api/backtest/run", _LONG_SYNTH),
        ("/api/backtest/realistic", {**_LONG_SYNTH, "oos_ratio": 0.3}),
    ])
    def test_it_is_off_by_default(self, client, path, payload):
        """
        **每条**能汇报冻结段的路径都要单独测。只测一条的后果实测过：
        `/backtest/realistic` 的那行条件被改成 `or` 之后，它会在没人请求时
        照样动用 holdout，而 `/backtest/run` 的用例全绿 —— 这正是 §R 说的
        "只测局部"。
        """
        body = _ok(client.post(path, json=payload))
        assert body.get("held_out_test") is None, (
            f"{path}：没请求就动用了冻结段 —— 一次性预算会被常规回测悄悄烧掉")

    def test_realistic_can_report_on_the_holdout_when_asked(self, client):
        body = _ok(client.post("/api/backtest/realistic",
                               json={**_LONG_SYNTH, "seed": 777, "oos_ratio": 0.3,
                                     "report_test": True}))
        held = body.get("held_out_test")
        assert held, "report_test=True 却没有 held_out_test"
        assert held["uses"] >= 1 and held["test_key"] == body["partition"]["test_key"]

    def test_asking_for_it_returns_both_the_numbers_and_the_usage_count(self, client):
        body = _ok(client.post("/api/backtest/run",
                               json={**_LONG_SYNTH, "report_test": True}))
        held = body.get("held_out_test")
        assert held, "report_test=True 却没有 held_out_test"
        assert "report" in held, f"冻结段汇报里没有指标：{sorted(held)}"
        assert held["uses"] >= 1 and held["budget"] >= 1
        assert held["test_key"] == body["partition"]["test_key"], (
            "记账用的 test_key 与切分报告的对不上 —— 两本账对不起来")

    def test_using_it_again_is_flagged_as_over_budget(self, client):
        payload = {**_LONG_SYNTH, "seed": 4242, "report_test": True}
        first = _ok(client.post("/api/backtest/run", json=payload))["held_out_test"]
        second = _ok(client.post("/api/backtest/run", json=payload))["held_out_test"]
        assert second["uses"] > first["uses"], "重复动用同一段 holdout，计数没涨"
        assert second["over_budget"] is True, (
            "同一段 holdout 被用了两次却没标 over_budget —— "
            "『一次性』就只剩文档里的一句话")
