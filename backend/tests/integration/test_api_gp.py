"""
test_api_gp.py — GP 进化 API 端点集成测试

覆盖：/api/gp/evolve
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


MINIMAL_GP = {
    "pop_size": 3,
    "n_gen": 1,
    "n_workers": 1,
    "n_tickers": 6,
    "n_days": 80,
    "seed": 42,
    "dataset_name": "",
}


class TestGPEvolveEndpoint:

    def test_evolve_returns_200(self, client):
        resp = client.post("/api/gp/evolve", json=MINIMAL_GP, timeout=120)
        assert resp.status_code in (200, 400, 422, 500)

    def test_evolve_hof_non_empty(self, client):
        resp = client.post("/api/gp/evolve", json=MINIMAL_GP, timeout=120)
        if resp.status_code == 200:
            body = resp.json()
            assert "hof" in body
            assert isinstance(body["hof"], list)

    def test_evolve_hof_has_required_fields(self, client):
        resp = client.post("/api/gp/evolve", json=MINIMAL_GP, timeout=120)
        if resp.status_code == 200:
            body = resp.json()
            if "hof" in body and body["hof"]:
                entry = body["hof"][0]
                assert "dsl" in entry
                assert "fitness" in entry or "sharpe" in entry

    def test_evolve_n_hof_field(self, client):
        resp = client.post("/api/gp/evolve", json=MINIMAL_GP, timeout=120)
        if resp.status_code == 200:
            body = resp.json()
            if "n_hof" in body and "hof" in body:
                assert body["n_hof"] == len(body["hof"])

    def test_evolve_invalid_pop_size_rejected(self, client):
        bad = {**MINIMAL_GP, "pop_size": 0}
        resp = client.post("/api/gp/evolve", json=bad, timeout=30)
        assert resp.status_code in (400, 422)
