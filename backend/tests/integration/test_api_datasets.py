"""
test_api_datasets.py — 数据集 API 端点集成测试

覆盖：GET /api/datasets, GET /api/datasets/{name}/health
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from app.main import app
    with TestClient(app, client=("127.0.0.1", 50000), raise_server_exceptions=False) as c:
        yield c


def _ok(resp, expect: int = 200):
    """DEV_LESSONS §S：断言具体状态码，不容忍 5xx，不用 if 包住断言。"""
    assert resp.status_code == expect, (
        f"期望 {expect} 实际 {resp.status_code}｜body={resp.text[:600]}"
    )
    return resp.json()


class TestDatasetsList:

    def test_list_datasets_200(self, client):
        resp = client.get("/api/datasets")
        assert resp.status_code == 200

    def test_list_datasets_non_empty(self, client):
        resp = client.get("/api/datasets")
        body = resp.json()
        assert "datasets" in body
        assert isinstance(body["datasets"], list)
        assert len(body["datasets"]) > 0

    def test_list_datasets_total_field(self, client):
        resp = client.get("/api/datasets")
        body = resp.json()
        assert "total" in body
        assert body["total"] == len(body["datasets"])

    def test_list_datasets_required_fields(self, client):
        body = _ok(client.get("/api/datasets"))
        assert body.get("datasets"), "注册表为空 —— 系统没有任何可用数据集"
        for ds in body["datasets"]:
            for f in ("name", "region", "provider", "n_assets"):
                assert f in ds, f"数据集条目缺字段 {f}：{sorted(ds)}"


@pytest.fixture
def synthetic_registry(monkeypatch):
    """D4：数据集换成合成面板（不连 yfinance）；端点的其余部分（健康检查本身）是真代码。"""
    import app.core.data_engine.dataset_registry as reg
    from app.core.data_engine.multi_dataset import Dataset
    from tests.conftest import _make_dataset
    data = _make_dataset(n_days=250, n_tickers=12)
    monkeypatch.setattr(reg, "load_registry_dataset", lambda name, start=None, end=None, **kw: Dataset(
        name=name, frequency="daily", universe=list(data["close"].columns), data=data))


class TestDatasetHealth:

    def test_health_known_dataset(self, client, synthetic_registry):
        body = _ok(client.get("/api/datasets/us_tech_large/health", timeout=30))
        assert body["name"] == "us_tech_large" and body["n_tickers"] == 12 and body["n_dates"] == 250
        assert 0.0 <= body["overall_score"] <= 1.0

    def test_health_nonexistent_dataset_returns_404(self, client):
        resp = client.get("/api/datasets/NONEXISTENT_DATASET_XYZ/health", timeout=10)
        assert resp.status_code == 404, (
            f"未知数据集必须 404，实际 {resp.status_code}：{resp.text[:300]}"
        )

    @pytest.mark.live_source
    def test_health_on_the_real_source(self, client):
        """真实 yfinance：QA_LIVE_SOURCES=1 时才跑；上游拿不到就是失败（不再用 skip / 502 放行）。"""
        body = _ok(client.get("/api/datasets/us_tech_large/health", timeout=60))
        assert "overall_score" in body and body["name"] == "us_tech_large", f"健康报告缺字段：{sorted(body)}"
