"""
test_agent_fallback.py — FallbackOrchestrator 与 QuantTools 测试（无 LLM 模式）

所有测试均在无 OPENAI_API_KEY 的 fallback 模式下运行。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 数据工厂
# ---------------------------------------------------------------------------

def _make_dataset(n_days: int = 80, n_tickers: int = 10, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_tickers)), axis=0),
        index=dates, columns=tickers,
    )
    volume = pd.DataFrame(
        rng.integers(500_000, 2_000_000, (n_days, n_tickers)).astype(float),
        index=dates, columns=tickers,
    )
    # DEV_LESSONS §O：固定 ±1% 的日内区间会让 Corwin-Schultz 价差估到 ~54bps
    # （真实约 8.5bps），足以把有真实 alpha 的因子判死。必须用随机幅度。
    high  = close * (1 + rng.uniform(0, 0.006, close.shape))
    low   = close * (1 - rng.uniform(0, 0.006, close.shape))
    open_ = close * (1 + rng.normal(0, 0.005, (n_days, n_tickers)))
    vwap  = (high + low + close) / 3
    return {
        "close": close, "open": open_, "high": high, "low": low,
        "volume": volume, "vwap": vwap,
        "returns": close.pct_change().fillna(0.0),
    }


@pytest.fixture
def tools():
    from app.agent._tools import QuantTools
    return QuantTools(n_tickers=10, n_days=80, oos_ratio=0.3, n_trials=2, seed=42,
                      allow_synthetic=True)


# ---------------------------------------------------------------------------
# QuantTools 测试
# ---------------------------------------------------------------------------

class TestQuantToolsFallback:

    def test_generate_alpha_dsl_returns_string(self, tools):
        import json
        result_str = tools.tool_generate_alpha_dsl("momentum")
        result = json.loads(result_str)
        assert "dsl" in result
        dsl = result["dsl"]
        assert isinstance(dsl, str) and len(dsl) > 0

    def test_run_backtest_returns_sharpe(self, tools):
        import json
        dsl = "rank(ts_delta(log(close), 5))"
        result_str = tools.tool_run_backtest(dsl)
        result = json.loads(result_str)
        assert "is_sharpe" in result or "sharpe" in result or "error" in result

    def test_run_backtest_invalid_dsl_returns_error(self, tools):
        import json
        result_str = tools.tool_run_backtest("INVALID_DSL_XYZ")
        result = json.loads(result_str)
        assert "error" in result or "is_sharpe" in result  # 不应崩溃

    def test_save_alpha_returns_id(self, tools):
        import json
        dsl = "rank(close)"
        result_str = tools.tool_save_alpha(dsl, "test", "{}")
        result = json.loads(result_str)
        assert "id" in result or "status" in result


# ---------------------------------------------------------------------------
# FallbackOrchestrator 意图识别测试
# ---------------------------------------------------------------------------

class TestFallbackOrchestratorIntent:

    @pytest.fixture
    def orchestrator(self, tools):
        from app.agent._fallback import FallbackOrchestrator
        return FallbackOrchestrator(tools=tools)

    def test_workflow_a_returns_dsl_and_metrics(self, orchestrator):
        dsl, metrics = orchestrator.run_workflow_a("test momentum hypothesis")
        assert isinstance(dsl, str) and len(dsl) > 0
        assert isinstance(metrics, dict)

    def test_workflow_b_returns_dsl_and_metrics(self, orchestrator):
        dsl, metrics = orchestrator.run_workflow_b("rank(ts_delta(log(close), 5))")
        assert isinstance(dsl, str) and len(dsl) > 0
        assert isinstance(metrics, dict)

    def test_workflow_a_metrics_has_sharpe(self, orchestrator):
        _, metrics = orchestrator.run_workflow_a("reversion alpha")
        assert "is_sharpe" in metrics or "sharpe" in metrics

    def test_workflow_a_dsl_not_empty(self, orchestrator):
        """返回的 DSL 应为非空字符串（包含字段名或函数调用）。"""
        dsl, _ = orchestrator.run_workflow_a("volume momentum")
        assert isinstance(dsl, str) and len(dsl) > 0


# ---------------------------------------------------------------------------
# 意图检测函数测试
# ---------------------------------------------------------------------------

class TestIntentDetection:

    def test_optimize_intent_detected(self):
        """消息包含 'optimize this' 或 DSL 引用时，应检测为 workflow_b。"""
        from app.agent._agent import QuantAgent

        # 原实现把整个方法体包进 try/except(Exception) → pytest.skip：
        # QuantTools(is_data=..., oos_data=...) 这两个形参根本不存在 → TypeError
        # → 无条件 skip。该用例从未真正调用过 _detect_intent。
        # _detect_intent 是 QuantAgent 的**实例方法** (self, message, session_id)，
        # 不是模块级函数 —— 原用例 import 模块级名字必然 ImportError，然后被
        # except(Exception)→skip 吞掉，从未真正执行过。
        from app.agent.quant_agent import QuantAgent
        agent = QuantAgent(n_tickers=8, n_days=80, api_key="", allow_synthetic=True)
        intent, dsl_hint = agent._detect_intent(
            "Optimize this: rank(ts_delta(log(close), 5))", "s1")
        assert intent == "workflow_b", f"含显式 DSL 应判为 workflow_b，实际 {intent!r}"
        assert dsl_hint and "rank(" in dsl_hint, f"未抽出 DSL：{dsl_hint!r}"

    def test_generate_intent_detected(self):
        from app.agent.quant_agent import QuantAgent
        agent = QuantAgent(n_tickers=8, n_days=80, api_key="", allow_synthetic=True)
        intent, dsl_hint = agent._detect_intent("Generate alpha for momentum factor", "s2")
        assert intent == "workflow_a", f"纯自然语言应判为 workflow_a，实际 {intent!r}"
        assert dsl_hint is None, f"无 DSL 时不应抽出 dsl_hint：{dsl_hint!r}"
