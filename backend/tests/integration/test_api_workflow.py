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
    # 注意：这些下限来自 GenerateRequest 的 ge= 约束。原载荷（n_days=80/
    # n_generations=1/pop_size=3/n_seed_dsls=3）全部违约 → 端点一直返回 422，
    # 而旧断言把 422 也算通过 —— 这两个 workflow 端点从未被集成测试真正跑过。
    "hypothesis": "price momentum reverses over short horizons",
    "n_generations": 2,
    "pop_size": 8,
    "n_seed_dsls": 5,
    "n_optuna": 0,
    "oos_ratio": 0.3,
    "n_tickers": 6,
    "n_days": 120,
    "seed": 42,
    "dataset_name": "",
}

MINIMAL_OPTIMIZE = {
    "dsl": "rank(ts_delta(log(close), 5))",
    "n_mutations": 2,
    "n_generations": 2,
    "pop_size": 8,
    "n_optuna": 0,
    "oos_ratio": 0.3,
    "n_tickers": 6,
    "n_days": 120,
    "seed": 42,
    "dataset_name": "",
}


def _ok(resp, expect: int = 200):
    """断言状态码并返回 body（DEV_LESSONS §S：不得容忍 5xx、不得条件断言）。"""
    assert resp.status_code == expect, (
        f"期望 {expect} 实际 {resp.status_code}｜body={resp.text[:600]}"
    )
    return resp.json()


class TestWorkflowGenerate:

    def test_generate_returns_200(self, client):
        _ok(client.post("/api/workflow/generate", json=MINIMAL_GENERATE, timeout=120))

    def test_generate_has_best_dsl(self, client):
        body = _ok(client.post("/api/workflow/generate", json=MINIMAL_GENERATE, timeout=120))
        assert isinstance(body.get("best_dsl"), str) and body["best_dsl"],             f"best_dsl 缺失或为空：{body.get('best_dsl')!r}"

    def test_generate_has_evolution_log(self, client):
        body = _ok(client.post("/api/workflow/generate", json=MINIMAL_GENERATE, timeout=120))
        assert isinstance(body.get("evolution_log"), list), f"缺 evolution_log：{sorted(body)}"
        assert body["evolution_log"], "evolution_log 为空 —— GP 实际没有跑过任何一代"

    def test_generate_pool_top5_bounded(self, client):
        body = _ok(client.post("/api/workflow/generate", json=MINIMAL_GENERATE, timeout=120))
        assert "pool_top5" in body, f"缺 pool_top5：{sorted(body)}"
        assert len(body["pool_top5"]) <= 5

    def test_generate_workflow_field(self, client):
        body = _ok(client.post("/api/workflow/generate", json=MINIMAL_GENERATE, timeout=120))
        assert body.get("workflow") == "generation", body.get("workflow")

    def test_generate_oos_sharpe_is_not_absurd(self, client):
        """
        真实问题探针：OOS 段太短时，年化 Sharpe 会飙到荒谬值（曾观测到 15.78），
        且过拟合检测判为 "healthy"。任何 |OOS Sharpe| > 8 都不可能是真信号，
        必须要么被拒绝、要么被标注样本不足 —— 不能当作正常结果返回。
        """
        body = _ok(client.post("/api/workflow/generate", json=MINIMAL_GENERATE, timeout=120))
        m = body.get("metrics") or {}
        oos = m.get("oos_sharpe")
        if oos is None:
            pytest.skip("该响应未含 oos_sharpe")
        assert abs(float(oos)) <= 8.0, (
            f"OOS Sharpe={oos:.2f} 荒谬（|Sharpe|>8 在真实市场不存在）。"
            f"根因通常是 OOS 段样本过少后做年化：n_days={MINIMAL_GENERATE['n_days']}, "
            f"oos_ratio={MINIMAL_GENERATE['oos_ratio']}, embargo 默认 20 → OOS 仅剩十几行。"
            f"系统却把它当正常结果返回。"
        )


class TestWorkflowOptimize:

    def test_optimize_returns_200(self, client):
        _ok(client.post("/api/workflow/optimize", json=MINIMAL_OPTIMIZE, timeout=120))

    def test_optimize_has_best_dsl(self, client):
        body = _ok(client.post("/api/workflow/optimize", json=MINIMAL_OPTIMIZE, timeout=120))
        assert isinstance(body.get("best_dsl"), str) and body["best_dsl"]

    def test_optimize_metrics_present(self, client):
        body = _ok(client.post("/api/workflow/optimize", json=MINIMAL_OPTIMIZE, timeout=120))
        assert "metrics" in body, f"缺 metrics：{sorted(body)}"
        assert "is_sharpe" in body["metrics"], f"metrics 缺 is_sharpe：{sorted(body['metrics'])}"
