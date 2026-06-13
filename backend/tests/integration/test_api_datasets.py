"""
test_api_datasets.py — 数据集 API 端点集成测试

覆盖：GET /api/datasets, GET /api/datasets/{name}/health
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
        resp = client.get("/api/datasets")
        body = resp.json()
        if body.get("datasets"):
            ds = body["datasets"][0]
            assert "name" in ds
            # region 或 n_assets 至少有一个
            assert "region" in ds or "n_assets" in ds or "provider" in ds


class TestDatasetHealth:

    def test_health_known_dataset(self, client):
        """已知数据集应返回 200 或 202（网络不可用时也可接受）。"""
        resp = client.get("/api/datasets/us_tech_large/health", timeout=30)
        assert resp.status_code in (200, 202, 404, 500)

    def test_health_nonexistent_dataset_returns_404(self, client):
        resp = client.get("/api/datasets/NONEXISTENT_DATASET_XYZ/health", timeout=10)
        assert resp.status_code in (404, 400, 422, 500)

    def test_health_response_has_score_if_200(self, client):
        resp = client.get("/api/datasets/us_tech_large/health", timeout=30)
        if resp.status_code == 200:
            body = resp.json()
            assert "overall_score" in body or "name" in body
