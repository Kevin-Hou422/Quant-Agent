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


# ===========================================================================
# 定钉测试（变异测试驱动）
# ===========================================================================
#
# `_fallback.py` 首测击杀率 **0.0%（4/4 全部存活）** —— 上面那些用例走的是
# 意图路由和 QuantTools，从没真的跑进 `run_workflow_a/b` 的内部分支。
#
# 四个存活点里最贵的是两处 `use_test_set=True`：那是 F6 加的
# **真实样本外（Test 段）最终验证**。翻成 False 之后，"真实样本外验证"
# 这一步会**重复使用 GP 已经看过的 Validate 段** —— 日志照打
# `is_true_holdout`，指标照样更新，而留出集从此再没被用过。
# 这正是整个项目最在意的那一类错误：结论看起来更强，实际上没有证据。

import json


class _ToolSpy:
    """
    QuantTools 替身：记录每一次工具调用及其参数。

    只实现 `run_workflow_a/b` 会用到的那几个方法，返回值固定，
    让控制流可预测 —— 本组用例验的是**调用了什么、传了什么**。
    """

    def __init__(self, dsl="rank(ts_delta(log(close), 5))"):
        self.calls: list = []
        self._dsl = dsl
        self._seed = 42
        # `run_workflow_b` 会直接读 tools 上的这两份数据去做 `_quick_metrics`
        ds = _make_dataset(n_days=80, n_tickers=8, seed=1)
        cut = 56
        self._is_data = {k: v.iloc[:cut] for k, v in ds.items()}
        self._oos_data = {k: v.iloc[cut:] for k, v in ds.items()}

    def _rec(self, name, **kw):
        self.calls.append((name, kw))

    def of(self, name) -> list:
        return [kw for n, kw in self.calls if n == name]

    # -- 工具 --------------------------------------------------------
    def tool_generate_alpha_dsl(self, hypothesis):
        self._rec("generate", hypothesis=hypothesis)
        return json.dumps({"dsl": self._dsl})

    def tool_run_gp_optimization(self, **kw):
        self._rec("gp", **kw)
        return json.dumps({
            "best_dsl": self._dsl,
            "metrics": {"is_sharpe": 1.2, "oos_sharpe": 0.9, "turnover": 1.0,
                        "overfitting_score": 0.1, "is_overfit": False},
        })

    def tool_run_backtest(self, dsl, config_json="{}", use_test_set=False):
        self._rec("backtest", dsl=dsl, config_json=config_json,
                  use_test_set=use_test_set)
        return json.dumps({
            "is_sharpe": 1.2, "oos_sharpe": 0.9, "turnover": 1.0,
            "overfitting_score": 0.1, "is_overfit": False,
            "is_true_holdout": bool(use_test_set),
        })

    def tool_mutate_ast(self, *a, **k):
        self._rec("mutate", args=a, kwargs=k)
        return json.dumps({"mutated_dsl": self._dsl, "mutation_type": "point"})

    def tool_save_alpha(self, *a, **k):
        self._rec("save", args=a, kwargs=k)
        return json.dumps({"saved": True, "id": 1})

    def tool_interpret_factor(self, *a, **k):
        self._rec("interpret", args=a, kwargs=k)
        return json.dumps({"factor_family": "momentum"})


def _orchestrator(spy):
    from app.agent._fallback import FallbackOrchestrator
    return FallbackOrchestrator(spy)


class TestHoldoutVerificationUsesTheTestSet:
    """
    **首测存活项 L124 / L231**：两条 workflow 收尾处的
    `tool_run_backtest(best_dsl, use_test_set=True)`。
    """

    def test_workflow_a_verifies_on_the_true_holdout(self):
        spy = _ToolSpy()
        _orchestrator(spy).run_workflow_a("动量在科技股上更强")

        bts = spy.of("backtest")
        assert bts, "Workflow A 一次回测都没跑"
        holdout = [b for b in bts if b["use_test_set"]]
        assert len(holdout) == 1, (
            f"Workflow A 的 {len(bts)} 次回测里，用真实留出集的有 "
            f"{len(holdout)} 次，应当恰好 1 次 —— "
            f"`use_test_set=True` 被翻成了 False，"
            f"「真实样本外验证」其实在重复用 GP 见过的 Validate 段")

    def test_workflow_b_verifies_on_the_true_holdout(self):
        spy = _ToolSpy()
        _orchestrator(spy).run_workflow_b("rank(ts_delta(log(close), 5))")

        bts = spy.of("backtest")
        assert bts, "Workflow B 一次回测都没跑"
        holdout = [b for b in bts if b["use_test_set"]]
        assert len(holdout) == 1, (
            f"Workflow B 用真实留出集的回测有 {len(holdout)} 次，应当恰好 1 次")

    def test_the_holdout_check_is_the_last_backtest_not_the_first(self):
        """
        留出集必须验**最终选出的那条**因子。
        如果它跑在精炼之前，验的就是一条后来被改掉的 DSL ——
        结论与实际保存的因子对不上。
        """
        spy = _ToolSpy()
        _orchestrator(spy).run_workflow_a("动量")
        bts = spy.of("backtest")
        assert bts[-1]["use_test_set"] is True, (
            f"最后一次回测不是留出集验证：{[b['use_test_set'] for b in bts]}")

    def test_the_holdout_metrics_reach_the_saved_alpha(self):
        """
        `final_metrics.update(holdout)` —— 验完不回填等于白验。
        桩会在 `use_test_set=True` 时返回 `is_true_holdout=True`，
        它必须出现在最终落库的指标里。
        """
        spy = _ToolSpy()
        best_dsl, metrics = _orchestrator(spy).run_workflow_a("动量")
        assert isinstance(best_dsl, str) and best_dsl
        assert metrics.get("is_true_holdout") is True, (
            f"留出集验证的结果没有回填进最终指标：{sorted(metrics)}")

    def test_a_failing_holdout_check_does_not_kill_the_workflow(self):
        """
        `except Exception: logger.warning(...)` —— 留出集验证失败
        （例如数据不够切三段）不该让整条 workflow 作废，
        但也不能假装验过了。
        """
        class _Boom(_ToolSpy):
            def tool_run_backtest(self, dsl, config_json="{}", use_test_set=False):
                if use_test_set:
                    raise RuntimeError("留出集不够长")
                return super().tool_run_backtest(dsl, config_json, use_test_set)

        spy = _Boom()
        best_dsl, metrics = _orchestrator(spy).run_workflow_a("动量")
        assert isinstance(best_dsl, str) and best_dsl, (
            "留出集验证失败把整条 workflow 打挂了")
        assert not metrics.get("is_true_holdout"), (
            "留出集验证抛异常了，指标里却标着 is_true_holdout")


class TestSeedGeneration:

    def test_an_empty_seed_pool_falls_back_to_the_single_dsl_tool(self, monkeypatch):
        """
        **首测存活项 L72**：`if not seed_dsls:`

        `not` 被删会让**正常产出了 12 条种子**的情况反而去调
        `tool_generate_alpha_dsl`，把种子池砍成 1 条 —— GP 失去多样性，
        而日志、返回值一切正常。
        """
        import app.core.workflows.alpha_workflows as WF

        monkeypatch.setattr(WF, "_generate_diverse_seeds",
                            lambda *a, **k: [])
        spy = _ToolSpy()
        _orchestrator(spy).run_workflow_a("动量")

        assert len(spy.of("generate")) == 1, (
            "种子池为空时没有退回 tool_generate_alpha_dsl")
        gp = spy.of("gp")
        assert gp, "没有跑 GP"

    def test_a_non_empty_seed_pool_skips_the_single_dsl_tool(self, monkeypatch):
        """`not` 被删的另一侧：有种子时不该再去要一条。"""
        import app.core.workflows.alpha_workflows as WF

        monkeypatch.setattr(
            WF, "_generate_diverse_seeds",
            lambda *a, **k: [f"rank(ts_delta(close,{n}))" for n in range(3, 15)])
        spy = _ToolSpy()
        _orchestrator(spy).run_workflow_a("动量")
        assert spy.of("generate") == [], (
            "已经有 12 条种子，却仍然调了 tool_generate_alpha_dsl —— "
            "`if not seed_dsls:` 的 not 被删了")

    def test_targeted_mutations_are_deduplicated_against_the_seed_pool(self):
        """
        **首测存活项 L179**：`if td not in seen_set:`

        `not` 被删会让**只有已经在池子里的**定向变异被再加一遍 ——
        种子池里出现重复个体，GP 的初始多样性被稀释，
        而 `len(seed_dsls)` 看起来反而更大（日志里更好看）。
        """
        user_dsl = "rank(ts_delta(log(close), 5))"
        spy = _ToolSpy(dsl=user_dsl)
        _orchestrator(spy).run_workflow_b(user_dsl)

        gp = spy.of("gp")
        assert gp, "Workflow B 没有跑 GP"
        seeds = json.loads(gp[0]["seed_dsls_json"])
        assert len(seeds) == len(set(seeds)), (
            f"种子池里有重复项：{[s for s in seeds if seeds.count(s) > 1][:3]} —— "
            f"`if td not in seen_set` 的 not 被删了")

    def test_the_seed_pool_actually_contains_the_user_dsl(self):
        """
        用户给的那条必须在种子池里（种子池会把空白规整掉，
        所以按去空白后的形式比对）。

        它不在池子里意味着 GP 从一组"改过的变体"起步，
        而基准那一条从未被评估 —— 报告里的"优化前后对比"失去基准。
        """
        user_dsl = "rank(ts_delta(log(close), 5))"
        spy = _ToolSpy(dsl=user_dsl)
        _orchestrator(spy).run_workflow_b(user_dsl)
        seeds = json.loads(spy.of("gp")[0]["seed_dsls_json"])
        squeeze = lambda s: "".join(s.split())
        assert squeeze(user_dsl) in {squeeze(s) for s in seeds}, (
            f"用户给的 DSL 没有进种子池：{seeds[:5]}")


# ===========================================================================
# 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "app/agent/_fallback.py ×1 — L179 `if td not in seen_set:` 的 not 删除":
        "`seen_set` 初值是 `set(_expand_for_optimization(user_dsl, 8))`，"
        "随后逐个加入 `_targeted_mutations(user_dsl, init_metrics)` 的产物。"
        "实测这两个函数的产物**从不相交**，且 targeted 自身也**从不含重复** —— "
        "穷举 9 条形态各异的 DSL × 81 组指标（sharpe_is/oos、turnover、"
        "overfit 各三档）共 729 种可达组合，重叠次数与内部重复次数都是 0。"
        "于是 `td not in seen_set` 恒真、`td in seen_set` 恒假，"
        "两种取值对一切可达输入给出同一个种子池。"
        "（这不是『没找到对的观察面』：`seed_dsls` 是这一步唯一的输出，"
        "而它在两种取值下逐元素相同。）"
        "机械验证见 test_targeted_mutations_never_collide_with_the_expanded_pool。",
}


def test_targeted_mutations_never_collide_with_the_expanded_pool():
    """
    L179 等价性的机械验证：穷举可达输入，确认
    `_targeted_mutations` 的产物既不与 `_expand_for_optimization` 重叠，
    自身也无重复。

    任一方将来产出重叠（比如定向变异开始复用通用变异的模板），
    这条立刻变红 —— 那时 L179 重新成为一个能杀的变异点，
    上面那份等价性证明作废。
    """
    import warnings

    from app.core.workflows.alpha_workflows import (
        _expand_for_optimization, _targeted_mutations,
    )

    dsls = [
        "rank(ts_delta(log(close), 5))", "rank(close)", "ts_mean(close,20)",
        "zscore(ts_delta(close,3))", "rank(ts_std(returns,20))",
        "rank(ts_corr(close,volume,20))", "ts_rank(close,10)",
        "scale(ts_delta(close,1))", "rank(ts_decay_linear(close,5))",
    ]
    metric_grid = [
        {"is_sharpe": si, "oos_sharpe": so, "turnover": to,
         "overfitting_score": ov}
        for si in (0.2, 1.2, 2.5)
        for so in (-0.5, 0.1, 0.9)
        for to in (0.5, 3.0, 12.0)
        for ov in (0.05, 0.4, 0.9)
    ]

    checked = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dsl in dsls:
            expanded = set(_expand_for_optimization(dsl, n_mutations=8))
            assert expanded, f"{dsl} 的通用变异池是空的 —— 本验证失去意义"
            for metrics in metric_grid:
                targeted = _targeted_mutations(dsl, metrics)
                checked += 1
                assert len(targeted) == len(set(targeted)), (
                    f"_targeted_mutations 自身产出了重复项："
                    f"{[t for t in targeted if targeted.count(t) > 1][:3]} —— "
                    f"L179 的等价性证明作废")
                overlap = set(targeted) & expanded
                assert not overlap, (
                    f"定向变异与通用变异池重叠：{sorted(overlap)[:3]} —— "
                    f"L179 的等价性证明作废")

    assert checked == len(dsls) * len(metric_grid) == 729, (
        f"只验了 {checked} 种组合 —— 网格被改小了，证明强度下降")


def test_every_survivor_has_a_written_proof():
    """
    首测 4 点 / 全部存活 → 补用例后杀死 3 处，剩 L179 一处为等价变异。
    """
    assert len(PROVEN_EQUIVALENT) == 1
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
