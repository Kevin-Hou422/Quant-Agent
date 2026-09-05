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


class TestDatasetHealth:

    def test_health_known_dataset(self, client):
        """已知数据集应返回 200 或 202（网络不可用时也可接受）。"""
        resp = client.get("/api/datasets/us_tech_large/health", timeout=30)
        # 无网络时允许 502/503（明确的上游失败），但**不得** 500（未处理异常）
        assert resp.status_code in (200, 502, 503), (
            f"健康检查返回 {resp.status_code}：{resp.text[:400]}"
        )

    def test_health_nonexistent_dataset_returns_404(self, client):
        resp = client.get("/api/datasets/NONEXISTENT_DATASET_XYZ/health", timeout=10)
        assert resp.status_code == 404, (
            f"未知数据集必须 404，实际 {resp.status_code}：{resp.text[:300]}"
        )

    def test_health_response_has_score_if_200(self, client):
        resp = client.get("/api/datasets/us_tech_large/health", timeout=30)
        if resp.status_code != 200:
            pytest.skip(f"上游数据不可用（{resp.status_code}）")
        body = resp.json()
        assert "overall_score" in body and "name" in body, f"健康报告缺字段：{sorted(body)}"
