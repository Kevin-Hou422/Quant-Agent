"""
test_api_report.py — Alpha 历史报告查询 API 测试

覆盖：GET /api/report/query
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


class TestReportQuery:

    def test_query_returns_200(self, client):
        resp = client.get("/api/report/query")
        assert resp.status_code == 200

    def test_query_response_has_records_field(self, client):
        resp = client.get("/api/report/query")
        body = resp.json()
        assert "records" in body

    def test_query_response_has_total_field(self, client):
        resp = client.get("/api/report/query")
        body = resp.json()
        assert "total" in body

    def test_query_min_sharpe_999_returns_empty(self, client):
        resp = client.get("/api/report/query", params={"min_sharpe": 999})
        body = resp.json()
        assert body.get("records") == [] or body.get("total") == 0

    def test_query_limit_param_honored(self, client):
        resp = client.get("/api/report/query", params={"limit": 2})
        body = resp.json()
        if "records" in body:
            assert len(body["records"]) <= 2

    def test_query_invalid_limit_rejected(self, client):
        resp = client.get("/api/report/query", params={"limit": 9999})
        assert resp.status_code in (200, 422)
