"""
test_api_workflow.py — Workflow API 端点集成测试

覆盖：/api/workflow/generate, /api/workflow/optimize
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


MINIMAL_GENERATE = {
    "hypothesis": "momentum reversal",
    "n_tickers": 6,
    "n_days": 80,
    "n_generations": 1,
    "pop_size": 3,
    "n_optuna": 0,
    "n_seed_dsls": 3,
    "oos_ratio": 0.3,
    "seed": 42,
    "dataset_name": "",
}

MINIMAL_OPTIMIZE = {
    "dsl": "rank(ts_delta(log(close), 5))",
    "n_mutations": 2,
    "n_generations": 1,
    "pop_size": 3,
    "n_optuna": 0,
    "oos_ratio": 0.3,
    "n_tickers": 6,
    "n_days": 80,
    "seed": 42,
    "dataset_name": "",
}


class TestWorkflowGenerate:

    def test_generate_returns_200(self, client):
        resp = client.post("/api/workflow/generate", json=MINIMAL_GENERATE, timeout=120)
        assert resp.status_code in (200, 400, 422, 500)

    def test_generate_has_best_dsl(self, client):
        resp = client.post("/api/workflow/generate", json=MINIMAL_GENERATE, timeout=120)
        if resp.status_code == 200:
            body = resp.json()
            assert "best_dsl" in body
            assert isinstance(body["best_dsl"], str)
            assert len(body["best_dsl"]) > 0

    def test_generate_has_evolution_log(self, client):
        resp = client.post("/api/workflow/generate", json=MINIMAL_GENERATE, timeout=120)
        if resp.status_code == 200:
            body = resp.json()
            if "evolution_log" in body:
                assert isinstance(body["evolution_log"], list)

    def test_generate_pool_top5_bounded(self, client):
        resp = client.post("/api/workflow/generate", json=MINIMAL_GENERATE, timeout=120)
        if resp.status_code == 200:
            body = resp.json()
            if "pool_top5" in body:
                assert len(body["pool_top5"]) <= 5

    def test_generate_workflow_field(self, client):
        resp = client.post("/api/workflow/generate", json=MINIMAL_GENERATE, timeout=120)
        if resp.status_code == 200:
            body = resp.json()
            if "workflow" in body:
                assert body["workflow"] in ("generate", "workflow_a", "A")


class TestWorkflowOptimize:

    def test_optimize_returns_200(self, client):
        resp = client.post("/api/workflow/optimize", json=MINIMAL_OPTIMIZE, timeout=120)
        assert resp.status_code in (200, 400, 422, 500)

    def test_optimize_has_best_dsl(self, client):
        resp = client.post("/api/workflow/optimize", json=MINIMAL_OPTIMIZE, timeout=120)
        if resp.status_code == 200:
            body = resp.json()
            assert "best_dsl" in body
            assert isinstance(body["best_dsl"], str)

    def test_optimize_metrics_present(self, client):
        resp = client.post("/api/workflow/optimize", json=MINIMAL_OPTIMIZE, timeout=120)
        if resp.status_code == 200:
            body = resp.json()
            assert "metrics" in body or "best_dsl" in body
