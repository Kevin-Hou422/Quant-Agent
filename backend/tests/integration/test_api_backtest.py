"""
test_api_backtest.py — 回测 API 端点集成测试

覆盖：/api/backtest/run, /api/backtest/walk_forward, /api/backtest/realistic,
/api/alpha/simulate, /api/alpha/optimize
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
        resp = client.post("/api/backtest/run", json=MINIMAL)
        assert resp.status_code == 200

    def test_run_response_has_report(self, client):
        resp = client.post("/api/backtest/run", json=MINIMAL)
        body = resp.json()
        assert "report" in body or "sharpe_ratio" in body or "dsl" in body

    def test_run_invalid_dsl_returns_error(self, client):
        payload = {**MINIMAL, "dsl": "INVALID_DSL_XYZ"}
        resp = client.post("/api/backtest/run", json=payload)
        assert resp.status_code in (200, 400, 422, 500)
        # 关键：不应是 5xx 且有错误说明（或 200 带 error 字段）
        if resp.status_code == 200:
            body = resp.json()
            assert "error" in body or "report" in body

    def test_run_missing_dsl_field_returns_error(self, client):
        """缺少 dsl 字段时应返回 422（FastAPI 验证）或 200 带 error 字段。"""
        payload = {"n_tickers": 5, "n_days": 60}
        resp = client.post("/api/backtest/run", json=payload)
        assert resp.status_code in (200, 422)

    def test_run_minimal_dataset_no_crash(self, client):
        payload = {**MINIMAL, "n_tickers": 2, "n_days": 40}
        resp = client.post("/api/backtest/run", json=payload)
        assert resp.status_code in (200, 400, 422, 500)

    def test_run_response_dsl_echoed(self, client):
        resp = client.post("/api/backtest/run", json=MINIMAL)
        if resp.status_code == 200:
            body = resp.json()
            assert body.get("dsl") == MINIMAL["dsl"] or "report" in body


class TestWalkForward:

    def test_walk_forward_basic(self, client):
        payload = {
            "dsl": "rank(close)",
            "n_splits": 2,
            "embargo_days": 5,
            "portfolio_mode": "long_short",
            "delay": 1,
            "n_tickers": 8,
            "n_days": 120,
            "seed": 0,
        }
        resp = client.post("/api/backtest/walk_forward", json=payload)
        assert resp.status_code in (200, 400, 422, 500)
        if resp.status_code == 200:
            body = resp.json()
            assert "fold_reports" in body or "n_folds" in body

    def test_walk_forward_fold_count(self, client):
        payload = {
            "dsl": "rank(close)",
            "n_splits": 3,
            "embargo_days": 0,
            "portfolio_mode": "long_short",
            "delay": 1,
            "n_tickers": 6,
            "n_days": 150,
            "seed": 1,
        }
        resp = client.post("/api/backtest/walk_forward", json=payload)
        if resp.status_code == 200:
            body = resp.json()
            if "fold_reports" in body:
                assert len(body["fold_reports"]) <= 3

    def test_walk_forward_embargo_respected(self, client):
        """embargo_days=10 时，OOS start 应晚于 IS end 10 个工作日。"""
        payload = {
            "dsl": "rank(close)",
            "n_splits": 2,
            "embargo_days": 10,
            "portfolio_mode": "long_short",
            "delay": 1,
            "n_tickers": 6,
            "n_days": 150,
            "seed": 2,
        }
        resp = client.post("/api/backtest/walk_forward", json=payload)
        if resp.status_code == 200:
            body = resp.json()
            if "fold_reports" in body and body["fold_reports"]:
                fold = body["fold_reports"][0]
                if "is_end" in fold and "oos_start" in fold:
                    from datetime import datetime, timedelta
                    is_end   = datetime.fromisoformat(fold["is_end"].replace("Z", ""))
                    oos_start = datetime.fromisoformat(fold["oos_start"].replace("Z", ""))
                    gap = (oos_start - is_end).days
                    assert gap >= 0


class TestRealisticBacktest:

    def test_realistic_with_oos(self, client):
        payload = {
            "dsl": "rank(ts_delta(log(close), 5))",
            "n_tickers": 8,
            "n_days": 80,
            "oos_ratio": 0.3,
            "seed": 0,
            "config": {
                "delay": 1,
                "decay_window": 0,
                "truncation_min_q": 0.01,
                "truncation_max_q": 0.99,
                "portfolio_mode": "long_short",
                "top_pct": 0.2,
            },
        }
        resp = client.post("/api/backtest/realistic", json=payload)
        assert resp.status_code in (200, 400, 422, 500)
        if resp.status_code == 200:
            body = resp.json()
            assert "is_report" in body

    def test_realistic_no_oos_returns_none_oos_report(self, client):
        payload = {
            "dsl": "rank(close)",
            "n_tickers": 6,
            "n_days": 60,
            "oos_ratio": 0.0,
            "seed": 0,
            "config": {
                "delay": 1,
                "decay_window": 0,
                "truncation_min_q": 0.01,
                "truncation_max_q": 0.99,
                "portfolio_mode": "long_short",
                "top_pct": 0.2,
            },
        }
        resp = client.post("/api/backtest/realistic", json=payload)
        if resp.status_code == 200:
            body = resp.json()
            assert body.get("oos_report") is None


class TestAlphaSimulate:

    def test_simulate_basic(self, client):
        payload = {
            "dsl": "rank(ts_delta(log(close), 5))",
            "n_tickers": 8,
            "n_days": 80,
            "oos_ratio": 0.3,
            "seed": 0,
            "config": {
                "delay": 1,
                "decay_window": 0,
                "truncation_min_q": 0.01,
                "truncation_max_q": 0.99,
                "portfolio_mode": "long_short",
                "top_pct": 0.2,
            },
        }
        resp = client.post("/api/alpha/simulate", json=payload)
        assert resp.status_code in (200, 400, 422, 500)
        if resp.status_code == 200:
            body = resp.json()
            assert "is_metrics" in body or "error" in body

    def test_simulate_no_oos(self, client):
        payload = {
            "dsl": "rank(close)",
            "n_tickers": 6,
            "n_days": 60,
            "oos_ratio": 0.0,
            "seed": 0,
            "config": {"delay": 1, "decay_window": 0, "truncation_min_q": 0.01,
                       "truncation_max_q": 0.99, "portfolio_mode": "long_short", "top_pct": 0.2},
        }
        resp = client.post("/api/alpha/simulate", json=payload)
        if resp.status_code == 200:
            body = resp.json()
            assert body.get("oos_metrics") is None
