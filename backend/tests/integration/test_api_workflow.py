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

    def test_generate_oos_sharpe_is_absent_or_sane(self, client):
        """
        契约二选一，**不允许第三种**：
          (a) OOS 样本充足 → 返回一个不荒谬的 Sharpe（|SR|≤8）；
          (b) 样本不足     → oos_sharpe 置 None 且 insufficient_sample=True。
        绝不允许"样本只有十几天却返回 15.78 且判 healthy"。
        （原用例在 oos_sharpe 为 None 时 skip —— skip 不是断言，见 DEV_LESSONS §S。）
        """
        body = _ok(client.post("/api/workflow/generate", json=MINIMAL_GENERATE, timeout=120))
        m = body.get("metrics") or {}
        assert "oos_sharpe" in m, f"metrics 缺 oos_sharpe：{sorted(m)}"
        oos = m["oos_sharpe"]
        if oos is None:
            assert m.get("insufficient_sample") is True, (
                f"oos_sharpe 为 None 却没标 insufficient_sample —— 读者无从判断"
                f"是'没算'还是'算不出来'：{m}"
            )
            assert "n_obs_oos" in m, "未告知 OOS 实际有多少观测"
            return
        assert abs(float(oos)) <= 8.0, (
            f"OOS Sharpe={oos:.2f} 荒谬（|SR|>8 在真实市场不存在），"
            f"且未被标为样本不足。n_obs_oos={m.get('n_obs_oos')}"
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
