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
    # 原 pop_size=3 违反 ge=5 → 端点一直返回 422，而旧断言容忍 422，
    # 于是 /api/gp/evolve 从未被集成测试真正调用过。
    "pop_size": 5,
    "n_gen": 1,
    "n_workers": 1,
    "n_tickers": 6,
    "n_days": 80,
    "seed": 42,
    "dataset_name": "",
}


def _ok(resp, expect: int = 200):
    """DEV_LESSONS §S：断言具体状态码，不容忍 5xx，不用 if 包住断言。"""
    assert resp.status_code == expect, (
        f"期望 {expect} 实际 {resp.status_code}｜body={resp.text[:600]}"
    )
    return resp.json()


class TestGPEvolveEndpoint:

    def test_evolve_returns_200(self, client):
        _ok(client.post("/api/gp/evolve", json=MINIMAL_GP, timeout=120))

    def test_evolve_hof_non_empty(self, client):
        body = _ok(client.post("/api/gp/evolve", json=MINIMAL_GP, timeout=120))
        assert isinstance(body.get("hof"), list), f"缺 hof：{sorted(body)}"
        assert body["hof"], "hof 为空 —— GP 实际没有产出任何个体"

    def test_evolve_hof_has_required_fields(self, client):
        body = _ok(client.post("/api/gp/evolve", json=MINIMAL_GP, timeout=120))
        entry = body["hof"][0]
        assert "dsl" in entry, f"hof 条目缺 dsl：{sorted(entry)}"
        assert ("fitness" in entry) or ("sharpe" in entry), f"hof 条目缺适应度：{sorted(entry)}"

    def test_evolve_n_hof_field(self, client):
        body = _ok(client.post("/api/gp/evolve", json=MINIMAL_GP, timeout=120))
        assert body["n_hof"] == len(body["hof"])

    def test_evolve_invalid_pop_size_rejected(self, client):
        bad = {**MINIMAL_GP, "pop_size": 0}
        resp = client.post("/api/gp/evolve", json=bad, timeout=30)
        assert resp.status_code in (400, 422)
