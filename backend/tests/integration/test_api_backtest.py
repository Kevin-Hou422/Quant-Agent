"""
test_api_backtest.py — 回测 API 端点集成测试

覆盖：/api/backtest/run, /api/backtest/walk_forward, /api/backtest/realistic,
/api/alpha/simulate, /api/alpha/optimize

断言纪律（DEV_LESSONS §S）：
  - **禁止** `assert status_code in (..., 500)` —— 那让"端点每次都崩"也算通过；
  - **禁止** `if status_code == 200:` 包住断言 —— 崩了就跳过检查等于没测；
  - 每个用例必须断言一个**具体的**期望状态码 + 无条件检查响应体。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _ok(resp, expect: int = 200):
    """断言状态码并返回 body —— 失败时把服务端返回原文带进报错，便于定位。"""
    assert resp.status_code == expect, (
        f"期望 {expect} 实际 {resp.status_code}｜body={resp.text[:600]}"
    )
    return resp.json()


# ---------------------------------------------------------------------------
# 最小合成数据载荷
# ---------------------------------------------------------------------------

MINIMAL = {
    "dsl": "rank(ts_delta(log(close), 5))",
    "dataset_name": "",
    "n_tickers": 8,
    "n_days": 60,
    "seed": 42,
}


class TestBacktestRun:

    def test_run_basic_returns_200(self, client):
        _ok(client.post("/api/backtest/run", json=MINIMAL))

    def test_run_response_has_report(self, client):
        body = _ok(client.post("/api/backtest/run", json=MINIMAL))
        assert "report" in body, f"响应缺少 report：{sorted(body)}"
        assert "sharpe_ratio" in body["report"], f"report 缺少 sharpe_ratio：{sorted(body['report'])}"

    def test_run_invalid_dsl_returns_error(self, client):
        """非法 DSL 必须被拒（4xx）或以 200+error 明确报错——**不得 5xx**。"""
        resp = client.post("/api/backtest/run", json={**MINIMAL, "dsl": "INVALID_DSL_XYZ"})
        assert resp.status_code < 500, f"非法 DSL 导致服务端崩溃 {resp.status_code}：{resp.text[:400]}"
        assert resp.status_code in (200, 400, 422), f"意外状态码 {resp.status_code}"
        # 无条件断言：200 必须带 error，4xx 必须带 detail —— 两条路都要检查到
        body = resp.json()
        key = "error" if resp.status_code == 200 else "detail"
        assert key in body, f"{resp.status_code} 响应缺少 {key} → 非法 DSL 被静默接受：{sorted(body)}"

    def test_run_missing_dsl_field_returns_error(self, client):
        """缺 dsl 字段：要么 422 校验失败，要么 200 带 error；**不得**静默用默认 DSL 跑。"""
        resp = client.post("/api/backtest/run", json={"n_tickers": 5, "n_days": 60, "dataset_name": ""})
        assert resp.status_code < 500, f"缺字段导致 5xx：{resp.text[:400]}"
        body = resp.json()
        if resp.status_code == 422:
            return                                   # Pydantic 拒绝，符合契约
        assert "error" in body, (
            f"缺 dsl 却返回了完整回测结果（静默套用默认 DSL）：{sorted(body)}｜"
            f"dsl={body.get('dsl')!r} —— 调用方会以为测的是自己的因子"
        )

    def test_run_minimal_dataset_no_crash(self, client):
        """请求模型的下限（n_tickers≥5 / n_days≥60）就是"最小"数据集，用它跑通。"""
        body = _ok(client.post("/api/backtest/run", json={**MINIMAL, "n_tickers": 5, "n_days": 60}))
        assert "report" in body

    def test_run_below_minimum_is_rejected_not_crashed(self, client):
        """低于下限必须 422 被拒 —— 而不是 500，也不是静默夹到下限后照跑。"""
        resp = client.post("/api/backtest/run", json={**MINIMAL, "n_tickers": 2, "n_days": 40})
        assert resp.status_code == 422, (
            f"低于下限的请求返回 {resp.status_code} 而非 422：{resp.text[:300]}"
        )

    def test_run_response_dsl_echoed(self, client):
        body = _ok(client.post("/api/backtest/run", json=MINIMAL))
        assert body.get("dsl") == MINIMAL["dsl"], f"回显 DSL 不符：{body.get('dsl')!r}"


class TestWalkForward:

    BASE = {
        "dsl": "rank(close)", "portfolio_mode": "long_short", "delay": 1,
        # Task 6.1：显式请求合成数据（离线测试）。省略时默认 "us_tech_large"，
        # 无网络会正确返回 502（不再静默降级），故此处显式置空。
        "dataset_name": "",
    }

    def test_walk_forward_basic(self, client):
        payload = {**self.BASE, "n_splits": 2, "embargo_days": 5,
                   "n_tickers": 8, "n_days": 120, "seed": 0}
        body = _ok(client.post("/api/backtest/walk_forward", json=payload))
        assert "fold_reports" in body, f"缺 fold_reports：{sorted(body)}"
        assert body["fold_reports"], "fold_reports 为空 —— walk-forward 实际未运行任何一折"

    def test_walk_forward_fold_count(self, client):
        payload = {**self.BASE, "n_splits": 3, "embargo_days": 0,
                   "n_tickers": 6, "n_days": 150, "seed": 1}
        body = _ok(client.post("/api/backtest/walk_forward", json=payload))
        folds = body["fold_reports"]
        assert 1 <= len(folds) <= 3, f"折数 {len(folds)} 不在 [1,3]"

    def test_walk_forward_embargo_respected(self, client):
        """
        embargo_days=10 → OOS 起始必须比 IS 结束晚**至少 10 个交易日**。
        （原用例只断言 gap>=0，那是恒真的——embargo 设成 0 也能过，等于没测。）
        """
        payload = {**self.BASE, "n_splits": 2, "embargo_days": 10,
                   "n_tickers": 6, "n_days": 200, "seed": 2}
        body = _ok(client.post("/api/backtest/walk_forward", json=payload))
        folds = body["fold_reports"]
        assert folds, "无 fold 可检查 embargo"
        import pandas as pd
        for f in folds:
            is_end    = pd.Timestamp(f["is_end"])
            oos_start = pd.Timestamp(f["oos_start"])
            # 用交易日（工作日）计数，与 embargo 的语义一致
            n_bdays = len(pd.bdate_range(is_end, oos_start)) - 1
            assert n_bdays >= 10, (
                f"embargo 未生效：fold{f.get('fold_idx')} IS 末 {is_end.date()} → "
                f"OOS 始 {oos_start.date()} 仅隔 {n_bdays} 个工作日（要求 ≥10）"
            )


class TestRealisticBacktest:

    CFG = {"delay": 1, "decay_window": 0, "truncation_min_q": 0.01,
           "truncation_max_q": 0.99, "portfolio_mode": "long_short", "top_pct": 0.2}

    def test_realistic_with_oos(self, client):
        payload = {"dsl": "rank(ts_delta(log(close), 5))", "n_tickers": 8, "n_days": 80,
                   "oos_ratio": 0.3, "seed": 0, "config": self.CFG}
        body = _ok(client.post("/api/backtest/realistic", json=payload))
        assert "is_report" in body and body["is_report"], f"缺 is_report：{sorted(body)}"
        assert body.get("oos_report"), "oos_ratio=0.3 却没有 oos_report"

    def test_realistic_no_oos_returns_none_oos_report(self, client):
        payload = {"dsl": "rank(close)", "n_tickers": 6, "n_days": 60,
                   "oos_ratio": 0.0, "seed": 0, "config": self.CFG}
        body = _ok(client.post("/api/backtest/realistic", json=payload))
        assert body.get("oos_report") is None

    def test_realistic_must_accept_a_real_dataset(self, client):
        """
        审计发现（Phase B #1）：该端点名为 realistic 却**写死合成数据**，
        没有 dataset_name 入参 —— 返回的 Sharpe/风险报告全是随机游走。
        端点必须支持指定真实数据集，且响应必须自报数据来源。
        """
        payload = {"dsl": "rank(close)", "n_tickers": 6, "n_days": 60,
                   "oos_ratio": 0.3, "seed": 0, "config": self.CFG,
                   "dataset_name": ""}          # 显式合成
        body = _ok(client.post("/api/backtest/realistic", json=payload))
        assert "data_source" in body, (
            "realistic 端点未返回 data_source —— 用户无法分辨这份所谓 realistic 的报告"
            "是真实市场还是随机游走"
        )
        assert body["data_source"].startswith("synthetic"), body["data_source"]


class TestAlphaSimulate:

    CFG = {"delay": 1, "decay_window": 0, "truncation_min_q": 0.01,
           "truncation_max_q": 0.99, "portfolio_mode": "long_short", "top_pct": 0.2}

    def test_simulate_basic(self, client):
        payload = {"dsl": "rank(ts_delta(log(close), 5))", "n_tickers": 8, "n_days": 80,
                   "oos_ratio": 0.3, "seed": 0, "config": self.CFG}
        body = _ok(client.post("/api/alpha/simulate", json=payload))
        assert "is_metrics" in body, f"缺 is_metrics：{sorted(body)}"
        assert "sharpe_ratio" in body["is_metrics"]

    def test_simulate_no_oos(self, client):
        payload = {"dsl": "rank(close)", "n_tickers": 6, "n_days": 60,
                   "oos_ratio": 0.0, "seed": 0, "config": self.CFG}
        body = _ok(client.post("/api/alpha/simulate", json=payload))
        assert body.get("oos_metrics") is None
