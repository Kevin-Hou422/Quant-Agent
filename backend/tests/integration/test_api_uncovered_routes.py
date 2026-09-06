"""
test_api_uncovered_routes.py — 补齐债务台账里"零测试覆盖"的 API 路由

来由（DEV_LESSONS §S）：属性级审计发现 6 条 /api 路由**从未被任何测试触达**。
"留白正是出错的地方" —— 同一次审计里，被容忍 500 的 walk_forward 就是从没跑通过。

覆盖：
  POST /api/agent/run
  POST /api/backtest/multi
  GET  /api/paper/{alpha_id}/pnl
  POST /api/workflow/generate/stream   (SSE)
  POST /api/workflow/optimize/stream   (SSE)

断言纪律：具体状态码 + 无条件检查响应体；不容忍 500；不用 if 包住断言。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
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
    assert resp.status_code == expect, (
        f"期望 {expect} 实际 {resp.status_code}｜body={resp.text[:600]}"
    )
    return resp.json()


def _sse_events(resp) -> list[dict]:
    """把 SSE 响应体解析成事件列表。"""
    assert resp.status_code == 200, f"SSE 建立失败 {resp.status_code}：{resp.text[:400]}"
    out = []
    for line in resp.text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            try:
                out.append(json.loads(line[5:].strip()))
            except json.JSONDecodeError:
                pass
    return out


# ---------------------------------------------------------------------------
# POST /api/agent/run
# ---------------------------------------------------------------------------

class TestAgentRun:

    PAYLOAD = {
        "hypothesis": "short-horizon price reversal",
        "dataset_name": "",          # 显式合成（离线测试，不打网络）
        "n_tickers": 8, "n_days": 120, "seed": 42,
    }

    def test_agent_run_outcome_is_unambiguous(self, client):
        """
        两种结局都合法，但必须**可分辨**：
          passed=True  → final_dsl 非空且是表达式，final_metrics 有内容；
          passed=False → 本轮无候选过 IC-IR 门（合成随机数据上是常态）。
        原先只有一个空字符串的 final_dsl，调用方无从区分"没过门"与"agent 出错"。
        """
        body = _ok(client.post("/api/agent/run", json=self.PAYLOAD, timeout=180))
        assert body.get("initial_dsls"), f"未生成任何候选：{body}"
        assert "passed" in body, f"响应缺 passed 字段：{sorted(body)}"
        if body["passed"]:
            assert body["final_dsl"] and "(" in body["final_dsl"], body["final_dsl"]
            assert body["final_metrics"], "passed=True 却没有指标"
        else:
            assert not body.get("final_dsl"), (
                f"passed=False 却有 final_dsl={body['final_dsl']!r} —— 结论自相矛盾"
            )

    def test_agent_run_declares_data_source(self, client):
        body = _ok(client.post("/api/agent/run", json=self.PAYLOAD, timeout=180))
        assert body.get("data_source") == "synthetic", body.get("data_source")

    def test_agent_run_rejects_unknown_dataset_not_500(self, client):
        """未知数据集必须 502（上游明确失败），**不得**静默降级为合成数据。"""
        resp = client.post("/api/agent/run",
                           json={**self.PAYLOAD, "dataset_name": "NO_SUCH_DATASET_XYZ"},
                           timeout=60)
        assert resp.status_code == 502, (
            f"未知数据集返回 {resp.status_code}（期望 502）：{resp.text[:300]}"
        )


# ---------------------------------------------------------------------------
# POST /api/backtest/multi
# ---------------------------------------------------------------------------

class TestBacktestMulti:

    PAYLOAD = {
        "dsl": "rank(ts_delta(log(close), 5))",
        "datasets": ["us_tech_large", "us_financials"],
        "aggregation": "mean",
        "is_split": 0.70,
        "use_synthetic": True,        # 显式合成：离线可跑
        "n_tickers": 8, "n_days": 252, "seed": 7,
    }

    def test_multi_synthetic_reports_per_dataset(self, client):
        body = _ok(client.post("/api/backtest/multi", json=self.PAYLOAD, timeout=180))
        assert body["datasets_total"] == 2, body["datasets_total"]
        assert set(body["per_dataset"]) == set(self.PAYLOAD["datasets"]), body["per_dataset"]
        assert 0.0 <= body["pass_rate"] <= 1.0

    def test_multi_declares_data_source(self, client):
        """显式合成时响应必须自报来源 —— 否则读者分不清这份聚合 Sharpe 是不是噪声。"""
        body = _ok(client.post("/api/backtest/multi", json=self.PAYLOAD, timeout=180))
        assert body.get("data_source") == "synthetic", body.get("data_source")

    def test_multi_rejects_bad_aggregation(self, client):
        resp = client.post("/api/backtest/multi",
                           json={**self.PAYLOAD, "aggregation": "median"}, timeout=30)
        assert resp.status_code == 422, f"非法 aggregation 未被拒：{resp.status_code}"

    def test_multi_unknown_dataset_is_4xx_or_502_not_500(self, client):
        """未知数据集名：要么 422 校验拒绝，要么 502 上游失败；**不得** 500。"""
        resp = client.post("/api/backtest/multi",
                           json={**self.PAYLOAD, "use_synthetic": False,
                                 "datasets": ["NO_SUCH_DATASET_XYZ"]}, timeout=60)
        assert resp.status_code in (422, 502), (
            f"未知数据集返回 {resp.status_code}（期望 422 或 502）：{resp.text[:300]}"
        )


# ---------------------------------------------------------------------------
# GET /api/paper/{alpha_id}/pnl
# ---------------------------------------------------------------------------

class TestPaperPnL:

    def test_pnl_unknown_alpha_returns_404(self, client):
        resp = client.get("/api/paper/99999999/pnl")
        assert resp.status_code == 404, f"不存在的 alpha 未返回 404：{resp.status_code}"

    def test_pnl_returns_points_and_equity(self, client):
        """有 paper 账本记录时，端点必须把逐日净值如实读回来。"""
        from app.dependencies import get_store
        from app.db.alpha_store import AlphaResult
        from app.db.position_store import PositionStore

        store = get_store()
        aid = store.save(AlphaResult(dsl="rank(close)", hypothesis="pnl-route-test",
                                     sharpe=0.0, status="candidate"))
        from app.core.execution.paper_broker import DailyPnL
        ps = PositionStore()
        eq = 1.0
        for i, r in enumerate([0.004, -0.002, 0.006]):
            eq *= (1.0 + r)
            d = f"2024-01-0{i + 2}"
            ps.record_day(alpha_id=aid, date=d, positions={"AAPL": 0.5},
                          fills=[], pnl=DailyPnL(alpha_id=aid, date=d, gross_ret=r,
                                                 net_ret=r, cost_bps=0.0, equity=eq))

        body = _ok(client.get(f"/api/paper/{aid}/pnl"))
        assert body["alpha_id"] == aid
        assert body["n_days"] == 3, f"落库 3 天却读回 {body['n_days']} 天"
        assert len(body["points"]) == 3
        assert abs(body["latest_equity"] - eq) < 1e-9, (
            f"净值读回不符：{body['latest_equity']} vs {eq}"
        )

    def test_pnl_limit_is_enforced(self, client):
        resp = client.get("/api/paper/1/pnl", params={"limit": 99999})
        assert resp.status_code == 422, f"超限 limit 未被拒：{resp.status_code}"


# ---------------------------------------------------------------------------
# SSE：/api/workflow/{generate,optimize}/stream
# ---------------------------------------------------------------------------

class TestWorkflowStreams:
    """
    §R 的同型风险：非流式版有覆盖、流式版没有，于是"只修了 POST 没修 SSE"
    这种半修复不会被发现（Phase A 的 data_source 就差点如此）。
    """

    GEN = {
        "hypothesis": "price momentum reverses over short horizons",
        "n_generations": 2, "pop_size": 8, "n_seed_dsls": 5, "n_optuna": 0,
        "oos_ratio": 0.3, "n_tickers": 6, "n_days": 120, "seed": 42,
        "dataset_name": "",
    }
    OPT = {
        "dsl": "rank(ts_delta(log(close), 5))",
        "n_mutations": 2, "n_generations": 2, "pop_size": 8, "n_optuna": 0,
        "oos_ratio": 0.3, "n_tickers": 6, "n_days": 120, "seed": 42,
        "dataset_name": "",
    }

    @pytest.mark.parametrize("path,payload", [
        ("/api/workflow/generate/stream", GEN),
        ("/api/workflow/optimize/stream", OPT),
    ])
    def test_stream_emits_done_with_best_dsl(self, client, path, payload):
        events = _sse_events(client.post(path, json=payload, timeout=300))
        kinds = [e.get("type") for e in events]
        assert "error" not in kinds, f"{path} 以 error 结束：{events[-1]}"
        done = [e for e in events if e.get("type") == "done"]
        assert done, f"{path} 未发出 done 事件，实际事件序列={kinds}"
        result = done[-1]["result"]
        assert isinstance(result.get("best_dsl"), str) and result["best_dsl"], result

    def test_generate_stream_applies_same_sample_boundary_as_post(self, client):
        """
        流式与非流式必须**同一口径**：样本不足时比率类指标一律置空。
        （只在 POST 那条路上加边界 = 只修一半，正是 §R。）
        """
        events = _sse_events(client.post("/api/workflow/generate/stream",
                                         json=self.GEN, timeout=300))
        done = [e for e in events if e.get("type") == "done"]
        assert done, "未发出 done 事件"
        m = done[-1]["result"].get("metrics") or {}
        assert "oos_sharpe" in m, f"metrics 缺 oos_sharpe：{sorted(m)}"
        oos = m["oos_sharpe"]
        if oos is None:
            assert m.get("insufficient_sample") is True, (
                f"置空却未标 insufficient_sample —— 读者分不清'没算'与'算不出来'：{m}"
            )
        else:
            assert abs(float(oos)) <= 8.0, (
                f"SSE 路径返回荒谬 OOS Sharpe={oos:.2f} 且未标样本不足 —— "
                f"边界只加在了 POST 那条路上"
            )

    def test_stream_progress_events_precede_done(self, client):
        events = _sse_events(client.post("/api/workflow/generate/stream",
                                         json=self.GEN, timeout=300))
        kinds = [e.get("type") for e in events]
        assert "done" in kinds, f"无 done 事件：{kinds}"
        assert kinds.index("done") == len(kinds) - 1, f"done 之后仍有事件：{kinds}"
        assert kinds.count("text") >= 1, f"没有任何进度事件：{kinds}"
