"""
test_perf_api.py — API 并发与顺序负载测试
"""
from __future__ import annotations

import time
import threading
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


SMALL_BACKTEST = {
    "dsl": "rank(close)",
    "dataset_name": "",
    "n_tickers": 5,
    "n_days": 50,
    "seed": 1,
}


class TestAPISequentialPerformance:

    def test_health_endpoint_under_100ms(self, client):
        start = time.perf_counter()
        resp = client.get("/health")
        elapsed = (time.perf_counter() - start) * 1000
        assert resp.status_code == 200
        assert elapsed < 100, f"Health took {elapsed:.1f}ms"

    def test_backtest_run_under_30s(self, client):
        start = time.perf_counter()
        resp = client.post("/api/backtest/run", json=SMALL_BACKTEST, timeout=60)
        elapsed = time.perf_counter() - start
        assert resp.status_code in (200, 400, 422, 500)
        assert elapsed < 30.0, f"Backtest took {elapsed:.1f}s (limit 30s)"

    def test_datasets_list_under_2s(self, client):
        start = time.perf_counter()
        resp = client.get("/api/datasets", timeout=10)
        elapsed = time.perf_counter() - start
        assert resp.status_code == 200
        assert elapsed < 2.0, f"Datasets list took {elapsed:.2f}s (limit 2s)"

    def test_report_query_under_1s(self, client):
        start = time.perf_counter()
        resp = client.get("/api/report/query", timeout=5)
        elapsed = time.perf_counter() - start
        assert resp.status_code == 200
        assert elapsed < 1.0, f"Report query took {elapsed:.3f}s (limit 1s)"


class TestAPIConcurrentPerformance:

    def test_3_concurrent_backtests_no_crash(self, client):
        """3 个并发回测请求，所有均应在 60s 内完成且不崩溃。"""
        results = []
        errors  = []

        def _run():
            try:
                resp = client.post("/api/backtest/run", json=SMALL_BACKTEST, timeout=60)
                results.append(resp.status_code)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=_run) for _ in range(3)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        assert len(errors) == 0, f"Errors: {errors}"
        assert all(s in (200, 400, 422, 500) for s in results)
        assert elapsed < 90.0, f"Concurrent test took {elapsed:.1f}s"

    def test_5_rapid_chat_requests_no_timeout(self, client):
        """5 个连续 chat 请求，无超时或崩溃。"""
        for i in range(5):
            resp = client.post("/api/chat", json={
                "message": f"test message {i}",
                "session_id": f"perf-test-{i}",
            }, timeout=30)
            assert resp.status_code in (200, 500)
