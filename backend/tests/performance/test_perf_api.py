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


#: **n_days 必须 ≥ 60**：接口的校验下限就是 60。
#:
#: 外部审计 2026-09-15 指出"性能用例可能在量一个 400/422 的耗时"。实测更糟 ——
#: 原来写的是 `n_days: 50`，该请求**每次都返回 422**，断言却只要求
#: `status_code < 500`，于是整组性能用例一路全绿，量的是**参数校验被拒**的延迟，
#: 从来没跑过一次回测。
#:
#: 规矩：**先证明业务真的做完了，再谈耗时。**
SMALL_BACKTEST = {
    "dsl": "rank(close)",
    "dataset_name": "",
    "n_tickers": 5,
    "n_days": 60,
    "seed": 1,
}


def _assert_backtest_actually_ran(resp):
    """性能断言的前置：这次请求确实完成了一次回测，而不是被校验拒掉。"""
    assert resp.status_code == 200, (
        f"回测没跑成（HTTP {resp.status_code}）—— 这次计时量的不是回测耗时："
        f"{resp.text[:300]}")
    body = resp.json()
    assert "report" in body, f"响应里没有 report：{sorted(body)}"
    assert "sharpe_ratio" in body["report"], (
        f"report 里没有 sharpe_ratio，回测未真正产出：{sorted(body['report'])}")
    return body


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
        _assert_backtest_actually_ran(resp)
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
                results.append(resp)
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
        assert len(results) == 3, f"并发 3 个请求只收到 {len(results)} 个响应"
        # 三个都必须**真的跑完回测**。旧版只要求 `< 500`，于是并发下
        # 三个 422 也算"并发通过"（见 SMALL_BACKTEST 的注释）。
        for r in results:
            _assert_backtest_actually_ran(r)
        assert elapsed < 90.0, f"Concurrent test took {elapsed:.1f}s"

    def test_5_rapid_chat_requests_no_timeout(self, client):
        """5 个连续 chat 请求，无超时或崩溃。"""
        for i in range(5):
            resp = client.post("/api/chat", json={
                "message": f"test message {i}",
                "session_id": f"perf-test-{i}",
            }, timeout=30)
            assert resp.status_code == 200, (
                f"第 {i} 个 chat 请求没成功（HTTP {resp.status_code}）—— "
                f"这次计时量的不是 chat 耗时：{resp.text[:200]}")
            assert resp.json(), f"第 {i} 个 chat 请求返回空响应体"
