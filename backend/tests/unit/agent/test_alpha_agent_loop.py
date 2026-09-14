"""
agent/alpha_agent.py —— 假设 → DSL → 评估 → 精炼 的闭环

**此前零专属测试**（10 个变异点，D 档；首测报的 11 点里有 1 个是
工具把提示词散文当代码的误报，工具修好后已消失）。
（`tests/unit/workflows/test_alpha_workflows_*.py` 测的是
`core/workflows/alpha_workflows.py`，与本模块同名不同物。）

这个 agent 是**唯一一条会把因子直接写进 AlphaStore 的 LLM 路径**。
它的危险点全部落在两个数字上：

  `ic_ir >= 0.3` 与 `turnover <= 5.0`

这两道闸决定"这条因子存不存进库"。任一比较符翻面：
  - `>=` → `>`：IC-IR 恰好 0.3 的因子被永远拒绝（少存，损失有限）；
  - `<=` → `<`：换手恰好 5 的被拒；
  - `>=` → `<=`（方向反转）：**只有垃圾因子会被存进库**，
    而日志照样打"Alpha 通过"，AlphaStore 里照样多一条 candidate。

`run()` 里还有第二道同样的闸（`metrics.get("ic_ir", 0) >= 0.3`），
两道必须一致 —— 不一致会出现"精炼环节认为合格、落库环节认为不合格"
的静默丢弃。

LLM 全程用桩；`quick_ic_eval` 用桩回放指标，让每一道闸都能精确卡在边界上。
"""
from __future__ import annotations

import json
import logging
import pathlib
import types

import numpy as np
import pandas as pd
import pytest

import app.agent.alpha_agent as AA
from app.agent.alpha_agent import (
    _OP_WHITELIST,
    _SYSTEM_PROMPT,
    AlphaAgent,
    _build_llm,
    _call_llm,
    _parse_dsls_from_response,
)

IDX = pd.bdate_range("2022-01-03", periods=40)
COLS = ["AAA", "BBB", "CCC"]


def _dataset() -> dict:
    rng = np.random.default_rng(2)
    return {f: pd.DataFrame(rng.normal(100, 1, (40, 3)), index=IDX, columns=COLS)
            for f in ("close", "volume", "returns")}


class _Store:
    def __init__(self):
        self.saved = []

    def save(self, ar):
        self.saved.append(ar)
        return len(self.saved)


class _Proxy:
    def __init__(self, prune=False):
        self.prune = prune
        self.updates = []

    def should_prune(self, node):
        return self.prune

    def update(self, node, failed=False):
        self.updates.append(failed)


@pytest.fixture
def agent(monkeypatch):
    """
    构造一个不依赖 LLM、不依赖真实回测的 AlphaAgent。
    `metrics` 由 `_metrics` 列表按调用顺序回放。
    """
    state = {"metrics": [{"ic_ir": 0.5, "ann_turnover": 1.0, "sharpe": 1.2}],
             "n": 0, "evals": []}

    def fake_eval(dsl, dataset):
        state["evals"].append(dsl)
        seq = state["metrics"]
        m = seq[min(state["n"], len(seq) - 1)]
        state["n"] += 1
        return dict(m)

    monkeypatch.setattr(AA, "_quick_eval", fake_eval)
    monkeypatch.setattr(AA, "_build_llm", lambda: None)

    store, proxy = _Store(), _Proxy()
    a = AlphaAgent(store=store, proxy=proxy)
    return types.SimpleNamespace(agent=a, store=store, proxy=proxy, state=state,
                                 monkeypatch=monkeypatch)


# ===========================================================================
# A. LLM 构造与调用
# ===========================================================================

class TestLlmWiring:

    def test_no_api_key_means_no_llm(self, monkeypatch):
        """
        `if not api_key: return None` —— `not` 被删会让没有 key 时
        照样去构造 ChatOpenAI，第一次调用才炸；有 key 时反而返回 None，
        整条 LLM 路径静默失效。
        """
        monkeypatch.setenv("OPENAI_API_KEY", "")
        assert _build_llm() is None

    def test_a_missing_key_env_var_also_means_no_llm(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _build_llm() is None

    def test_a_present_key_builds_the_documented_model(self, monkeypatch):
        """
        `ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=api_key)`

        温度被改成 0 会让五条候选 DSL 变成同一条（失去多样性）；
        模型名被改会静默换成另一个价位/能力的模型。
        """
        seen = {}
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        import sys
        monkeypatch.setitem(
            sys.modules, "langchain_openai",
            types.SimpleNamespace(ChatOpenAI=lambda **k: seen.update(k) or "LLM"))
        assert _build_llm() == "LLM"
        assert seen["model"] == "gpt-4o"
        assert seen["temperature"] == 0.7, (
            f"温度是 {seen['temperature']} —— 0 会让 5 条候选退化成同一条")
        assert seen["api_key"] == "sk-test"

    def test_a_missing_langchain_openai_degrades_to_no_llm(self, monkeypatch):
        """
        `except ImportError: warnings.warn(...); return None`
        —— 兜底被删会让没装 langchain-openai 的环境连 AlphaAgent
        都构造不出来（而它本可以用内置的 5 条 fallback DSL 工作）。
        """
        import sys
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setitem(sys.modules, "langchain_openai", None)
        with pytest.warns(UserWarning, match="fallback"):
            assert _build_llm() is None

    def test_an_llm_failure_returns_an_empty_string_not_an_exception(self):
        """
        `except Exception: return ""` —— 一次网络抖动不该掀翻整轮研究。
        返回空串会让下游走进 fallback DSL，这是刻意的降级。
        """
        class _Boom:
            def invoke(self, msgs):
                raise RuntimeError("rate limited")

        assert _call_llm(_Boom(), "hi") == ""

    def test_a_successful_call_returns_the_message_content(self):
        class _Ok:
            def invoke(self, msgs):
                return types.SimpleNamespace(content="回答")

        assert _call_llm(_Ok(), "hi") == "回答"


# ===========================================================================
# B. 响应解析
# ===========================================================================

class TestResponseParsing:

    def test_a_well_formed_json_payload_is_used_directly(self):
        txt = json.dumps({"dsls": ["rank(close)", "zscore(volume)"]})
        assert _parse_dsls_from_response(txt) == ["rank(close)", "zscore(volume)"]

    def test_a_json_without_the_dsls_key_yields_an_empty_list(self):
        """`data.get("dsls", [])` —— 默认值被改会让缺键时抛或返回 None。"""
        assert _parse_dsls_from_response(json.dumps({"other": 1})) == []

    def test_a_non_json_reply_falls_back_to_regex_extraction(self):
        """
        `except json.JSONDecodeError: pass` 之后的正则兜底 ——
        LLM 经常在 JSON 外面裹一段解释文字。兜底被删会让这类回复
        全部变成"零条候选"，于是每次都走 fallback DSL，
        看起来像"LLM 没帮上忙"，实际是解析没写好。
        """
        txt = ('好的，这是我的建议：\n'
               '1. "rank(ts_mean(close,20))"\n'
               '2. "zscore(ts_delta(close,5))"\n')
        assert _parse_dsls_from_response(txt) == ["rank(ts_mean(close,20))",
                                                  "zscore(ts_delta(close,5))"]

    def test_the_regex_fallback_is_capped_at_five(self):
        """
        `matches[:5]` —— 上限被删会让一段啰嗦的回复产出几十条候选，
        每条都要跑一次回测。
        """
        txt = " ".join(f'"rank(ts_mean(close,{i}))"' for i in range(20))
        assert len(_parse_dsls_from_response(txt)) == 5

    def test_plain_text_without_any_formula_yields_nothing(self):
        assert _parse_dsls_from_response("抱歉，我无法生成。") == []


# ===========================================================================
# C. 候选生成
# ===========================================================================

class TestDslGeneration:

    def test_without_an_llm_the_five_builtin_seeds_are_used(self, agent):
        """
        无 LLM 时的 5 条内置种子 —— 它们是"完全免费跑通"的保证。
        条数或内容被改会让离线模式的搜索起点整个变样。
        """
        dsls = agent.agent._generate_dsls("动量")
        assert dsls == [
            "rank(ts_mean(close,20))",
            "zscore(ts_delta(close,5))",
            "rank(ts_std(close,10))",
            "zscore(ts_mean(volume,20))",
            "rank(ts_delta(log(close),5))",
        ], f"离线种子被改动：{dsls}"

    def test_the_builtin_seeds_all_parse_and_validate(self, agent):
        """
        种子必须是**合法 DSL** —— 否则 `_validate_and_fix` 在无 LLM 时
        直接返回 None，离线模式一条候选都跑不出来，且毫无提示。
        """
        seeds = agent.agent._generate_dsls("x")
        assert len(seeds) == 5, f"离线种子不是 5 条：{seeds}"
        for dsl in seeds:
            node = AA._parser.parse(dsl)
            assert node is not None, f"{dsl} 解析出空节点"
            assert AA._validator.validate(node) is None, (
                f"{dsl} 没有通过校验（validate 应当返回 None 且不抛）")

    def test_the_llm_reply_is_used_when_available(self, agent):
        agent.agent._llm = object()
        agent.monkeypatch.setattr(
            AA, "_call_llm",
            lambda llm, p: json.dumps({"dsls": ["rank(vwap)"]}))
        assert agent.agent._generate_dsls("动量") == ["rank(vwap)"]

    def test_an_empty_llm_reply_falls_back_to_two_seeds(self, agent):
        """
        `return dsls if dsls else [...]` —— 三目被改会让 LLM 失败时
        返回空列表，`run()` 遍历零条候选，直接报"本轮无 Alpha 通过"，
        而真正的原因是 LLM 没回话。
        """
        agent.agent._llm = object()
        agent.monkeypatch.setattr(AA, "_call_llm", lambda llm, p: "")
        assert agent.agent._generate_dsls("x") == [
            "rank(ts_delta(close,5))", "zscore(ts_mean(returns,10))"]

    def test_the_prompt_carries_the_hypothesis_and_the_operator_whitelist(self,
                                                                          agent):
        """
        白名单没进提示词，LLM 会生成解析不了的算子，
        五条候选全部在 `_validate_and_fix` 里被丢掉 —— 静默空转。
        """
        seen = {}
        agent.agent._llm = object()
        agent.monkeypatch.setattr(
            AA, "_call_llm",
            lambda llm, p: seen.setdefault("prompt", p) or "")
        agent.agent._generate_dsls("动量在科技股上更强")
        p = seen["prompt"]
        assert "动量在科技股上更强" in p, "提示词里没有用户假设"
        assert _OP_WHITELIST in p, "提示词里没有算子白名单"
        assert _SYSTEM_PROMPT in p

    def test_the_whitelist_lists_only_operators_the_parser_accepts(self):
        """
        白名单与解析器脱节是最阴的一种：LLM 严格照着白名单写，
        产出的 DSL 却解析不了。这里抽取白名单里的函数名逐个验证。
        """
        import re
        names = set(re.findall(r"(\w+)\(", _OP_WHITELIST))
        assert names, "白名单里解析不出任何算子名"
        args = {"ts_mean": "close,20", "ts_std": "close,20", "ts_delta": "close,20",
                "ts_delay": "close,20", "ts_max": "close,20", "ts_min": "close,20",
                "ts_rank": "close,20", "ts_decay_linear": "close,20",
                "rank": "close", "zscore": "close", "scale": "close",
                "log": "close", "abs": "close", "sqrt": "close", "sign": "close",
                "signed_power": "close,2", "if_else": "close,close,close"}
        for n in sorted(names):
            assert n in args, f"白名单里出现了未知算子 {n}"
            AA._parser.parse(f"{n}({args[n]})")


# ===========================================================================
# D. 校验与修复
# ===========================================================================

class TestValidateAndFix:

    def test_a_valid_dsl_passes_through_unchanged(self, agent):
        assert agent.agent._validate_and_fix("rank(close)") == "rank(close)"

    def test_an_invalid_dsl_without_an_llm_is_rejected_immediately(self, agent):
        """
        `if attempt >= 2 or self._llm is None: return None`

        `or` 翻成 `and` 会让无 LLM 时仍然循环三次（每次都走同一条
        必然失败的路径），白白慢三倍；`self._llm is None` 被删则会
        在无 LLM 时对 None 调用 `invoke`。
        """
        assert agent.agent._validate_and_fix("这不是 DSL") is None

    def test_the_llm_gets_at_most_two_repair_attempts(self, agent):
        """
        `for attempt in range(3)` + `if attempt >= 2`
        —— 第 0、1 次失败可以求助 LLM，第 2 次失败直接放弃。
        次数被改会让一条永远修不好的 DSL 把 LLM 额度耗在原地。
        """
        calls = []
        agent.agent._llm = object()
        agent.monkeypatch.setattr(
            AA, "_call_llm",
            lambda llm, p: calls.append(p) or json.dumps({"dsl": "还是不合法"}))
        assert agent.agent._validate_and_fix("不合法") is None
        assert len(calls) == 2, (
            f"向 LLM 求助了 {len(calls)} 次，应当恰好 2 次 —— "
            f"`attempt >= 2` 的边界被改了")

    def test_a_successful_repair_is_returned(self, agent):
        agent.agent._llm = object()
        agent.monkeypatch.setattr(
            AA, "_call_llm", lambda llm, p: json.dumps({"dsl": "rank(close)"}))
        assert agent.agent._validate_and_fix("坏的") == "rank(close)"

    def test_a_non_json_repair_reply_aborts_rather_than_looping(self, agent):
        """`except Exception: return None` —— 返回 None 而不是继续空转。"""
        calls = []
        agent.agent._llm = object()
        agent.monkeypatch.setattr(
            AA, "_call_llm", lambda llm, p: calls.append(1) or "不是 JSON")
        assert agent.agent._validate_and_fix("坏的") is None
        assert len(calls) == 1, "非 JSON 回复之后还在重试"

    def test_the_repair_prompt_carries_the_dsl_and_the_error(self, agent):
        seen = {}
        agent.agent._llm = object()
        agent.monkeypatch.setattr(
            AA, "_call_llm",
            lambda llm, p: seen.setdefault("p", p) or json.dumps({"dsl": "rank(close)"}))
        agent.agent._validate_and_fix("rank(close")      # 括号没闭合 → ParseError
        assert "rank(close" in seen["p"], "修复提示词里没有原始 DSL"
        assert "Error:" in seen["p"], "修复提示词里没有报错信息，LLM 无从下手"


# ===========================================================================
# E. 两道闸 —— 本模块的核心
# ===========================================================================

class TestQualityGates:

    def test_an_ic_ir_exactly_on_the_threshold_passes(self, agent):
        """
        `if ic_ir >= 0.3 and turnover <= 5.0:`

        `>=` 翻成 `>` 只改变 IC-IR **精确等于 0.3** 的那一格。
        0.3 是一个确定的双精度值，桩直接回放它，构造无误差。
        """
        agent.state["metrics"] = [{"ic_ir": 0.3, "ann_turnover": 1.0}]
        dsl, m = agent.agent._evaluate_and_refine("rank(close)", "h",
                                                  _dataset(), _log())
        assert dsl == "rank(close)", (
            "IC-IR 恰好 0.3 却没有通过 —— `>= 0.3` 被翻成了 `> 0.3`")

    def test_an_ic_ir_just_below_the_threshold_fails(self, agent):
        agent.state["metrics"] = [{"ic_ir": np.nextafter(0.3, 0.0),
                                   "ann_turnover": 1.0}]
        dsl, _ = agent.agent._evaluate_and_refine("rank(close)", "h",
                                                  _dataset(), _log())
        assert dsl == "rank(close)"          # 无 LLM → break，仍返回但不合格
        assert agent.proxy.updates == [True], "低于阈值却没有被标记为失败"

    def test_a_turnover_exactly_on_the_cap_passes(self, agent):
        """`turnover <= 5.0` —— `<=` 翻成 `<` 只改变恰好 5 的那一格。"""
        agent.state["metrics"] = [{"ic_ir": 0.9, "ann_turnover": 5.0}]
        dsl, _ = agent.agent._evaluate_and_refine("rank(close)", "h",
                                                  _dataset(), _log())
        assert dsl == "rank(close)"
        assert agent.proxy.updates == [False], (
            "换手恰好 5 却被判不合格 —— `<= 5.0` 被翻成了 `<`")

    def _refine_attempts(self, agent, metrics) -> list:
        """
        在"有 LLM"的前提下跑一次精炼，返回 LLM 收到的提示词列表。

        这是区分「闸门放行」与「闸门拦下」的**唯一**可靠观察面：
        放行时 `_evaluate_and_refine` 在第一轮就 return，根本不碰 LLM；
        拦下时才会进精炼循环去问 LLM。
        （只看返回值或 proxy 反馈是分不出来的 —— 两条路都给
        `(valid_dsl, metrics)` 和 `failed=False`，首测就是这么漏掉的。）
        """
        prompts = []
        agent.agent._llm = object()
        agent.agent._max_refine = 1
        agent.monkeypatch.setattr(
            AA, "_call_llm", lambda llm, p: prompts.append(p) or "不是 JSON")
        agent.state["metrics"] = [metrics]
        agent.agent._evaluate_and_refine("rank(close)", "h", _dataset(), _log())
        return prompts

    def test_a_turnover_above_the_cap_is_rejected(self, agent):
        """换手比 5.0 大一个 ulp 就必须被拦下 —— 拦下 = 进精炼循环问 LLM。"""
        prompts = self._refine_attempts(
            agent, {"ic_ir": 0.9, "ann_turnover": np.nextafter(5.0, np.inf)})
        assert len(prompts) == 1, (
            "换手超出上限却被放行（没有进精炼循环）—— `<= 5.0` 的边界被改了")

    def test_a_turnover_exactly_on_the_cap_is_let_through(self, agent):
        """边界另一侧：恰好 5.0 必须放行 —— 放行 = 完全不碰 LLM。"""
        prompts = self._refine_attempts(
            agent, {"ic_ir": 0.9, "ann_turnover": 5.0})
        assert prompts == [], (
            f"换手恰好 5.0 却被拦下（问了 LLM {len(prompts)} 次）—— "
            f"`<= 5.0` 被翻成了 `<`")

    def test_the_gate_is_a_conjunction_not_a_disjunction(self, agent):
        """
        `ic_ir >= 0.3 and turnover <= 5.0` 的 `and` 翻成 `or`，
        会让"IC-IR 高但换手爆表"的因子直接通过 ——
        而高换手因子扣掉成本后必然亏钱，那条 5.0 上限拦的就是它。

        `or` 与 `and` 在这组输入上给出**相同的返回值和相同的 proxy 反馈**，
        唯一的差别是走没走精炼循环。
        """
        prompts = self._refine_attempts(
            agent, {"ic_ir": 0.9, "ann_turnover": 50.0})
        assert len(prompts) == 1, (
            "IC 达标但换手 50 的因子被直接放行了 —— `and` 被改成了 `or`")
        assert "AnnTurnover=50.00" in prompts[0], (
            f"进了精炼但诊断不是换手：{prompts[0][:200]}")

    def test_both_gates_use_the_same_thresholds(self):
        """
        `run()` 与 `_evaluate_and_refine()` 各有一道 `ic_ir >= 0.3`。
        两者不一致会让"精炼认为合格、落库认为不合格"的因子被静默丢弃。
        用源码抽取两处阈值直接比对。
        """
        import inspect
        import re

        src_run = inspect.getsource(AlphaAgent.run)
        src_eval = inspect.getsource(AlphaAgent._evaluate_and_refine)
        t_run = re.findall(r'ic_ir[^\n]*?>=\s*([\d.]+)', src_run)
        t_eval = re.findall(r'ic_ir\s*>=\s*([\d.]+)', src_eval)
        assert t_run and t_eval, (
            f"抽不出阈值：run={t_run} eval={t_eval} —— 本用例需要重写")
        assert set(t_run) == set(t_eval) == {"0.3"}, (
            f"两处 IC-IR 闸值不一致：run={t_run} eval={t_eval}")


class TestMetricCoercion:

    @pytest.mark.parametrize("bad", [None, float("nan"), "不是数字", object()])
    def test_a_broken_metric_is_treated_as_zero_by_the_gate(self, agent, bad):
        """
        `_num` 的三层兜底（None / 非数 / NaN）——
        源码注释写明样本不足时 RiskReport 会把比率类指标置空，
        而后面的 `<` 比较与 `:.4f` 格式化都会 TypeError。

        观察面选在**精炼提示词**上：只有 `_num` 把坏值折成 0.0，
        后面的 `f"IC_IR={ic_ir:.4f}"` 才格式化得出来，
        并且诊断必须是"IC 太低"（0 < 0.3）而不是"换手太高"。
        任一层兜底被删，这里要么抛 TypeError、要么诊断跑偏。
        """
        prompts = []
        agent.agent._llm = object()
        agent.agent._max_refine = 1
        agent.monkeypatch.setattr(
            AA, "_call_llm", lambda llm, p: prompts.append(p) or "不是 JSON")

        agent.state["metrics"] = [{"ic_ir": bad, "ann_turnover": 1.0}]
        dsl, _ = agent.agent._evaluate_and_refine("rank(close)", "h",
                                                  _dataset(), _log())
        assert dsl == "rank(close)"
        assert prompts, "坏指标没有触发精炼 —— 它被当成了合格"
        assert "IC_IR=0.0000" in prompts[-1], (
            f"坏指标没有被折成 0.0，诊断文本是：{prompts[-1][:200]}")

    def test_a_missing_key_uses_the_default(self, agent):
        prompts = []
        agent.agent._llm = object()
        agent.agent._max_refine = 1
        agent.monkeypatch.setattr(
            AA, "_call_llm", lambda llm, p: prompts.append(p) or "不是 JSON")
        agent.state["metrics"] = [{}]
        dsl, _ = agent.agent._evaluate_and_refine("rank(close)", "h",
                                                  _dataset(), _log())
        assert dsl == "rank(close)"
        assert "IC_IR=0.0000" in prompts[-1]

    def test_the_final_proxy_update_never_escapes_on_a_broken_metric(self, agent):
        """
        收尾那句 `self._proxy.update(..., failed=(metrics.get("ic_ir", 0) < 0.3))`
        **没有**走 `_num`，坏指标会让 `<` 抛 TypeError ——
        全靠外面的 `except Exception: pass` 兜住。
        这条钉住"至少不会把异常抛给调用方"，
        同时记录下"此时代理模型拿不到反馈"这个事实。
        """
        agent.state["metrics"] = [{"ic_ir": object(), "ann_turnover": 1.0}]
        dsl, _ = agent.agent._evaluate_and_refine("rank(close)", "h",
                                                  _dataset(), _log())
        assert dsl == "rank(close)"
        assert agent.proxy.updates == [], (
            "坏指标下代理模型收到了反馈 —— 若已改为走 _num，请同步更新本用例")

    def test_a_nan_turnover_does_not_pass_the_cap_by_accident(self, agent):
        """
        NaN 被折成 0.0 → `0 <= 5` 为真。这是刻意的（保持"继续精炼"语义），
        但 IC-IR 那一侧仍然把关，所以整体不会放行。这里钉住这个组合。
        """
        agent.state["metrics"] = [{"ic_ir": 0.9, "ann_turnover": float("nan")}]
        dsl, _ = agent.agent._evaluate_and_refine("rank(close)", "h",
                                                  _dataset(), _log())
        assert dsl == "rank(close)"
        assert agent.proxy.updates == [False]


# ===========================================================================
# F. 精炼循环
# ===========================================================================

def _log():
    from app.tasks.reasoning_log import ReasoningLog
    return ReasoningLog(hypothesis="h", initial_dsls=[])


class TestRefineLoop:

    def test_a_pruned_node_short_circuits_before_evaluation(self, agent):
        """
        `if self._proxy.should_prune(node): return None, {}`
        —— `not` 被插入会让**只有**被剪枝的节点才继续评估，
        代理模型的作用完全反过来。
        """
        agent.agent._proxy = _Proxy(prune=True)
        dsl, m = agent.agent._evaluate_and_refine("rank(close)", "h",
                                                  _dataset(), _log())
        assert (dsl, m) == (None, {})
        assert agent.state["evals"] == [], "被剪枝的节点却跑了回测"

    def test_an_unfixable_dsl_short_circuits(self, agent):
        dsl, m = agent.agent._evaluate_and_refine("这不是 DSL", "h",
                                                  _dataset(), _log())
        assert (dsl, m) == (None, {})
        assert agent.state["evals"] == []

    def test_the_refine_loop_runs_at_most_max_refine_times(self, agent):
        """
        `for refine in range(self._max_refine)` —— 上限被改会让
        一条永远达不到阈值的因子把 LLM 额度耗光。
        构造：LLM 每次都给出新 DSL，但指标始终不合格。
        """
        agent.agent._llm = object()
        agent.agent._max_refine = 3
        agent.state["metrics"] = [{"ic_ir": 0.01, "ann_turnover": 1.0}]
        agent.monkeypatch.setattr(
            AA, "_call_llm",
            lambda llm, p: json.dumps({"dsl": "rank(close)", "reason": "试试"}))
        log = _log()
        agent.agent._evaluate_and_refine("rank(close)", "h", _dataset(), log)
        assert len(log.changes) == 3, (
            f"精炼了 {len(log.changes)} 轮，上限应当是 3")

    def test_each_refinement_is_recorded_in_the_reasoning_log(self, agent):
        """
        `log.add_change(old_dsl=..., new_dsl=..., reason=..., metrics=...)`
        —— 少记一项会让谱系里缺掉"从什么改成了什么、为什么"。
        """
        agent.agent._llm = object()
        agent.agent._max_refine = 1
        agent.state["metrics"] = [{"ic_ir": 0.01, "ann_turnover": 1.0}]
        agent.monkeypatch.setattr(
            AA, "_call_llm",
            lambda llm, p: json.dumps({"dsl": "zscore(close)",
                                       "reason": "加了标准化"}))
        log = _log()
        agent.agent._evaluate_and_refine("rank(close)", "h", _dataset(), log)
        assert len(log.changes) == 1
        c = log.changes[0]
        assert c.old_dsl == "rank(close)" and c.new_dsl == "zscore(close)"
        assert c.reason == "加了标准化"
        assert c.metrics, "变更记录里没有带上新指标"

    def test_the_refine_prompt_names_the_actual_problem(self, agent):
        """
        `reason = f"IC_IR=... (<0.3)" if ic_ir < 0.3 else f"AnnTurnover=... (>5)"`

        三目条件翻面会让"IC 太低"的因子收到"换手太高"的修改建议 ——
        LLM 按错误诊断去改，越改越偏，而每一步都记进了谱系。
        """
        prompts = []
        agent.agent._llm = object()
        agent.agent._max_refine = 1
        agent.monkeypatch.setattr(
            AA, "_call_llm", lambda llm, p: prompts.append(p) or "不是 JSON")

        agent.state["metrics"] = [{"ic_ir": 0.05, "ann_turnover": 1.0}]
        agent.agent._evaluate_and_refine("rank(close)", "h", _dataset(), _log())
        assert "IC_IR=0.0500" in prompts[-1], (
            f"IC 太低时给出的诊断是：{prompts[-1][:200]}")

        prompts.clear()
        agent.state["metrics"] = [{"ic_ir": 0.9, "ann_turnover": 50.0}]
        agent.state["n"] = 0
        agent.agent._evaluate_and_refine("rank(close)", "h", _dataset(), _log())
        assert "AnnTurnover=50.00" in prompts[-1], (
            f"换手太高时给出的诊断是：{prompts[-1][:200]}")

    def test_without_an_llm_the_loop_breaks_after_one_evaluation(self, agent):
        """`if self._llm is None: break` —— 离线模式不该空转 max_refine 次。"""
        agent.state["metrics"] = [{"ic_ir": 0.01, "ann_turnover": 1.0}]
        agent.agent._evaluate_and_refine("rank(close)", "h", _dataset(), _log())
        assert len(agent.state["evals"]) == 1

    def test_an_unfixable_refinement_stops_the_loop(self, agent):
        agent.agent._llm = object()
        agent.agent._max_refine = 3
        agent.state["metrics"] = [{"ic_ir": 0.01, "ann_turnover": 1.0}]
        agent.monkeypatch.setattr(
            AA, "_call_llm",
            lambda llm, p: json.dumps({"dsl": "彻底坏掉的 DSL", "reason": "r"}))
        log = _log()
        agent.agent._evaluate_and_refine("rank(close)", "h", _dataset(), log)
        assert log.changes == [], "修不好的建议也被记进了谱系"

    def test_the_proxy_is_told_whether_the_final_dsl_failed(self, agent):
        """
        `self._proxy.update(node, failed=(metrics.get("ic_ir", 0) < 0.3))`

        `<` 翻成 `>` 会让**好因子**被记为失败，代理模型此后一路剪掉
        同类结构 —— 搜索空间被悄悄削掉一大块，而且没有任何日志。
        """
        agent.state["metrics"] = [{"ic_ir": 0.01, "ann_turnover": 1.0}]
        agent.agent._evaluate_and_refine("rank(close)", "h", _dataset(), _log())
        assert agent.proxy.updates == [True], "低 IC 因子没有被标记为失败"

        agent.proxy.updates.clear()
        agent.state["metrics"] = [{"ic_ir": 0.9, "ann_turnover": 1.0}]
        agent.state["n"] = 0
        agent.agent._evaluate_and_refine("rank(close)", "h", _dataset(), _log())
        assert agent.proxy.updates == [False], "合格因子被标记成了失败"

    def test_a_failing_proxy_update_does_not_break_the_result(self, agent):
        """两处 `except Exception: pass` —— 代理模型坏掉不该丢掉已算出的因子。"""
        class _Boom(_Proxy):
            def update(self, node, failed=False):
                raise RuntimeError("代理炸了")

        agent.agent._proxy = _Boom()
        dsl, m = agent.agent._evaluate_and_refine("rank(close)", "h",
                                                  _dataset(), _log())
        assert dsl == "rank(close)" and m


# ===========================================================================
# G. run() —— 落库那一步
# ===========================================================================

class TestRunAndPersist:

    def test_a_passing_alpha_is_saved_as_a_candidate(self, agent):
        """
        `status = "candidate"`

        这是整个生命周期纪律的第一道防线：LLM 产出的因子**必须**
        以 candidate 起步，不能直接是 active。写成 "active" 会让
        一条从没验证过的因子直接进入交易候选池。
        """
        agent.state["metrics"] = [{"ic_ir": 0.9, "ann_turnover": 1.0,
                                   "sharpe": 1.5}]
        agent.agent.run("动量", _dataset())
        assert len(agent.store.saved) == 1
        ar = agent.store.saved[0]
        assert ar.status == "candidate", (
            f"落库状态是 {ar.status!r} —— LLM 产出的因子必须以 candidate 起步")

    def test_the_saved_record_carries_the_hypothesis_and_metrics(self, agent):
        agent.state["metrics"] = [{"ic_ir": 0.75, "ann_turnover": 2.5,
                                   "sharpe": 1.25}]
        agent.agent.run("动量在科技股上更强", _dataset())
        ar = agent.store.saved[0]
        assert ar.hypothesis == "动量在科技股上更强"
        assert ar.ic_ir == pytest.approx(0.75)
        assert ar.sharpe == pytest.approx(1.25)
        assert ar.ann_turnover == pytest.approx(2.5)
        assert ar.dsl

    def test_the_saved_ic_ir_is_always_a_float(self, agent):
        """
        `ic_ir = (lambda v: 0.0 if v is None else float(v))(metrics.get("ic_ir"))`

        这个 lambda 被简化成 `float(...)` 会在 IC-IR 为 None 时抛
        TypeError —— 整轮研究在最后一步作废。
        落库的字段必须是 `float`，不能是 numpy 标量或 None，
        否则 SQLAlchemy 在某些后端上会存成 BLOB。
        """
        agent.state["metrics"] = [{"ic_ir": np.float64(0.9),
                                   "ann_turnover": 1.0, "sharpe": 1.0}]
        agent.agent.run("动量", _dataset())
        ar = agent.store.saved[0]
        assert type(ar.ic_ir) is float, (
            f"落库的 ic_ir 类型是 {type(ar.ic_ir).__name__}，不是内置 float")
        assert ar.ic_ir == pytest.approx(0.9)

    def test_a_failing_alpha_is_not_saved(self, agent):
        """
        `if result_dsl and metrics.get("ic_ir", 0) >= 0.3:`
        —— 比较符翻向会让不合格的因子进库。
        """
        agent.state["metrics"] = [{"ic_ir": 0.01, "ann_turnover": 1.0}]
        log = agent.agent.run("动量", _dataset())
        assert agent.store.saved == [], "不合格的因子被存进了库"
        assert not log.final_dsl, (
            f"没有因子通过，final_dsl 却被填成了 {log.final_dsl!r}")

    def test_an_ic_ir_exactly_on_the_threshold_is_saved(self, agent):
        """落库那一道闸的边界：0.3 必须算通过（与精炼那道一致）。"""
        agent.state["metrics"] = [{"ic_ir": 0.3, "ann_turnover": 1.0}]
        agent.agent.run("动量", _dataset())
        assert len(agent.store.saved) == 1, (
            "IC-IR 恰好 0.3 没有落库 —— 两道闸的边界不一致")

    def test_the_loop_stops_at_the_first_passing_candidate(self, agent):
        """
        `break` —— 被删会让五条候选全部评估完，
        后面的还会覆盖 `log.final_dsl`，最终存进库的可能不是最好的那条。
        """
        agent.state["metrics"] = [{"ic_ir": 0.9, "ann_turnover": 1.0}]
        agent.agent.run("动量", _dataset())
        assert len(agent.store.saved) == 1, (
            f"存了 {len(agent.store.saved)} 条 —— 找到第一条合格的就该停")
        assert len(agent.state["evals"]) == 1

    def test_the_reasoning_log_records_every_initial_candidate(self, agent):
        """
        `log.initial_dsls = dsls` —— 漏掉会让谱系里缺掉"当初从哪几条起步"，
        事后无法判断是搜索起点不好还是精炼不力。
        """
        agent.state["metrics"] = [{"ic_ir": 0.01, "ann_turnover": 1.0}]
        log = agent.agent.run("动量", _dataset())
        assert len(log.initial_dsls) == 5
        assert log.initial_dsls == agent.agent._generate_dsls("动量")

    def test_the_final_metrics_are_attached_on_success(self, agent):
        agent.state["metrics"] = [{"ic_ir": 0.9, "ann_turnover": 1.0,
                                   "sharpe": 1.1}]
        log = agent.agent.run("动量", _dataset())
        assert log.final_dsl is not None
        assert log.final_metrics["ic_ir"] == pytest.approx(0.9)

    def test_the_saved_reasoning_is_the_serialised_log(self, agent):
        """
        `reasoning = log.to_json()` —— 存的是整份谱系。
        写成 `str(log)` 会让 AlphaStore 里躺着一串 repr，无法反序列化。
        """
        agent.state["metrics"] = [{"ic_ir": 0.9, "ann_turnover": 1.0}]
        agent.agent.run("动量", _dataset())
        raw = agent.store.saved[0].reasoning
        data = json.loads(raw)
        assert data["hypothesis"] == "动量"
        assert "changes" in data

    def test_the_max_refine_default_is_three(self):
        assert AlphaAgent(store=_Store(), proxy=_Proxy())._max_refine == 3


# ===========================================================================
# H. 首测存活项收口
# ===========================================================================
#
# 首测 11 点 / 击杀 45.5% / 存活 6。分两类：
#
#   1. **只有日志看得见的状态位**（L110 / L126 / L130）。
#      `passed_any` 除了决定末尾那一行日志之外不影响任何返回值 ——
#      而那一行正是无人值守时**唯一**能告诉操作者"这一轮到底有没有产出"
#      的信号。它反过来说，人会以为系统在空转（或相反，以为有产出）。
#      所以这三处必须用 caplog 钉。
#
#   2. **恰好卡在 0.3 上的两处比较**（L214 / L243）。
#      要走到它们，得让 IC-IR **精确等于 0.3** 且换手超限 ——
#      这样闸门不放行，控制流才会落到诊断文案与代理反馈那两行。


class TestRoundOutcomeLogging:

    def test_a_round_without_any_passing_alpha_says_so(self, agent, caplog):
        """
        **首测存活项（L110 / L130）**：
        `passed_any = False` 的初值、以及 `if not passed_any:` 的 not。

        这一行日志是无人值守跑批时**唯一**的产出信号。
        初值翻成 True（或 not 被删）之后，一轮颗粒无收也悄无声息，
        操作者只会看到日志里什么都没说，以为一切正常。
        """
        agent.state["metrics"] = [{"ic_ir": 0.01, "ann_turnover": 1.0}]
        with caplog.at_level(logging.INFO, logger="app.agent.alpha_agent"):
            agent.agent.run("动量假设", _dataset())
        msgs = [r.getMessage() for r in caplog.records]
        assert any("无 Alpha 通过筛选" in m for m in msgs), (
            f"一条都没通过，日志里却没说：{msgs}")
        assert any("动量假设" in m for m in msgs), "日志里没带上是哪个假设"
        assert agent.store.saved == []

    def test_a_round_with_a_passing_alpha_does_not_claim_failure(self, agent,
                                                                  caplog):
        """
        **首测存活项（L126）**：成功分支里的 `passed_any = True` 翻成 False。

        返回值、落库、final_dsl 全都正常 —— 唯一的差别是末尾
        多打一行"本轮无 Alpha 通过筛选"。跑批日志里看到这一行，
        人会去排查一个根本不存在的问题。
        """
        agent.state["metrics"] = [{"ic_ir": 0.9, "ann_turnover": 1.0,
                                   "sharpe": 1.2}]
        with caplog.at_level(logging.INFO, logger="app.agent.alpha_agent"):
            agent.agent.run("动量假设", _dataset())
        msgs = [r.getMessage() for r in caplog.records]
        assert len(agent.store.saved) == 1, "前提：这一轮确实有因子通过"
        assert not any("无 Alpha 通过筛选" in m for m in msgs), (
            f"明明有因子通过，却打了『无 Alpha 通过』：{msgs} —— "
            f"`passed_any = True` 被翻成了 False")

    def test_a_passing_alpha_is_announced_with_its_dsl_and_ic_ir(self, agent,
                                                                 caplog):
        agent.state["metrics"] = [{"ic_ir": 0.9, "ann_turnover": 1.0,
                                   "sharpe": 1.2}]
        with caplog.at_level(logging.INFO, logger="app.agent.alpha_agent"):
            agent.agent.run("动量", _dataset())
        msgs = [r.getMessage() for r in caplog.records]
        hit = [m for m in msgs if "Alpha 通过" in m and "无 Alpha" not in m]
        assert hit, f"通过的因子没有被记录：{msgs}"
        assert "0.9000" in hit[0], f"日志里没带上 IC-IR：{hit[0]}"

    def test_the_candidate_count_is_logged(self, agent, caplog):
        """`logger.info("Agent 生成 %d 条 DSL 候选", len(dsls))` —— 产出量的第一道读数。"""
        agent.state["metrics"] = [{"ic_ir": 0.01, "ann_turnover": 1.0}]
        with caplog.at_level(logging.INFO, logger="app.agent.alpha_agent"):
            agent.agent.run("动量", _dataset())
        msgs = [r.getMessage() for r in caplog.records]
        assert any("生成 5 条 DSL 候选" in m for m in msgs), (
            f"候选条数没有被记录：{msgs}")


class TestThresholdExactlyOnPointThree:
    """
    要走到 L214（诊断文案）与 L243（代理反馈）这两行，
    闸门必须**不放行**：`ic_ir >= 0.3 and turnover <= 5.0` 为假。

    于是构造 `ic_ir = 0.3`（精确）+ `turnover = 50`（超限）——
    IC 这一侧合格、换手这一侧不合格，控制流恰好落到那两行上，
    而 `0.3 < 0.3` 与 `0.3 <= 0.3` 在这里给出相反的结论。
    """

    def test_the_diagnosis_blames_turnover_not_ic_when_ic_is_exactly_at_the_bar(
            self, agent):
        """
        **首测存活项（L214）**：
        `reason = f"IC_IR=..." if ic_ir < 0.3 else f"AnnTurnover=..."`

        `<` 翻成 `<=` 会让 IC-IR 恰好达标、只是换手超限的因子
        收到"IC 太低"的诊断 —— LLM 会照着这个错误诊断去加信号强度，
        而真正该做的是降换手。越改越偏，每一步还都记进了谱系。
        """
        prompts = []
        agent.agent._llm = object()
        agent.agent._max_refine = 1
        agent.monkeypatch.setattr(
            AA, "_call_llm", lambda llm, p: prompts.append(p) or "不是 JSON")

        agent.state["metrics"] = [{"ic_ir": 0.3, "ann_turnover": 50.0}]
        agent.agent._evaluate_and_refine("rank(close)", "h", _dataset(), _log())

        assert prompts, "闸门没有拦住（换手 50 应当不放行）—— 本用例失去区分力"
        text = prompts[-1]
        assert "AnnTurnover=50.00" in text, (
            f"IC 恰好达标、换手超限，诊断却不是换手：{text[:200]} —— "
            f"`ic_ir < 0.3` 被翻成了 `<=`")
        assert "IC_IR=" not in text, f"诊断里混进了 IC 的说法：{text[:200]}"

    def test_the_diagnosis_blames_ic_when_it_is_one_ulp_below_the_bar(self, agent):
        """边界另一侧：IC-IR 比 0.3 低一个 ulp 时，诊断必须指向 IC。"""
        prompts = []
        agent.agent._llm = object()
        agent.agent._max_refine = 1
        agent.monkeypatch.setattr(
            AA, "_call_llm", lambda llm, p: prompts.append(p) or "不是 JSON")

        agent.state["metrics"] = [{"ic_ir": np.nextafter(0.3, 0.0),
                                   "ann_turnover": 50.0}]
        agent.agent._evaluate_and_refine("rank(close)", "h", _dataset(), _log())
        assert "IC_IR=" in prompts[-1], f"诊断没有指向 IC：{prompts[-1][:200]}"

    def test_an_ic_exactly_at_the_bar_is_not_reported_as_a_failure_to_the_proxy(
            self, agent):
        """
        **首测存活项（L243）**：
        `self._proxy.update(node, failed=(metrics.get("ic_ir", 0) < 0.3))`

        `<` 翻成 `>` 或 `<=` 都会让 IC-IR 恰好 0.3 的因子被记成"失败"。
        代理模型此后会一路剪掉同类结构 —— 搜索空间被悄悄削掉一块，
        而且没有任何日志说明原因。

        这里同样靠"IC 达标但换手超限"把控制流送到这一行。
        """
        agent.state["metrics"] = [{"ic_ir": 0.3, "ann_turnover": 50.0}]
        agent.agent._evaluate_and_refine("rank(close)", "h", _dataset(), _log())
        assert agent.proxy.updates == [False], (
            f"IC-IR 恰好 0.3 被反馈成失败（updates={agent.proxy.updates}）—— "
            f"`< 0.3` 的边界被改了")

    def test_an_ic_one_ulp_below_the_bar_is_reported_as_a_failure(self, agent):
        agent.state["metrics"] = [{"ic_ir": np.nextafter(0.3, 0.0),
                                   "ann_turnover": 50.0}]
        agent.agent._evaluate_and_refine("rank(close)", "h", _dataset(), _log())
        assert agent.proxy.updates == [True]

    def test_the_gate_and_the_proxy_feedback_agree_at_the_boundary(self, agent):
        """
        两处 0.3 必须同向：闸门放行的（`>= 0.3`）不能在代理那里
        被记成失败（`< 0.3`）。IC-IR 恰好 0.3 是唯一能同时触到两边的点。
        """
        agent.state["metrics"] = [{"ic_ir": 0.3, "ann_turnover": 1.0}]
        dsl, _ = agent.agent._evaluate_and_refine("rank(close)", "h",
                                                  _dataset(), _log())
        assert dsl == "rank(close)", "闸门没放行"
        assert agent.proxy.updates == [False], (
            "闸门放行了，代理那边却记成失败 —— 两处 0.3 的方向不一致")


# ===========================================================================
# I. 存活变异的等价性证明
# ===========================================================================

# 首测的第 6 个"存活"（L39）**不是等价变异，是工具误报**：
#
#   `Allowed operators and fields: {_OP_WHITELIST}`
#
# 这一行在 `_SYSTEM_PROMPT` 这个模块级 f-string 常量里，是写给 LLM 读的
# 英文散文。变异工具的 `_string_spans` 逐行 tokenize，认不出跨行字符串，
# 于是把 `and` 当成布尔运算符改成了 `or`（工具缺陷 #9，同一个缺陷在
# `_prompts.py` 上一次性报出 35 个假变异点）。
#
# **工具已修**（`_multiline_string_lines`），本模块的变异点从 11 降到 10，
# 这一"存活"随之消失 —— 复测 10/10 全杀。
#
# 下面这条不是等价性证明，而是**守住修复的前提**：一旦这段提示词被改写成
# 真正的可执行代码，工具就该重新把它算进变异点，本文件也需要补相应断言。


def test_the_whitelist_line_is_prose_inside_a_string_constant():
    """
    用 AST 确认 `_SYSTEM_PROMPT` 的右侧仍然是纯字符串字面量 ——
    也就是说，里面的 `and` / `>` / `*` 都是散文，不是代码。

    这条变红意味着两件事同时要做：
      ① 工具会重新把这些行算成变异点（分母变大）；
      ② 那些行成了真正的程序逻辑，必须为它们写行为断言。
    """
    import ast

    mod_src = pathlib.Path(AA.__file__).read_text(encoding="utf-8")
    tree = ast.parse(mod_src)

    targets = [n for n in tree.body
               if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == "_SYSTEM_PROMPT"
                       for t in n.targets)]
    assert len(targets) == 1, "alpha_agent 里找不到唯一的 _SYSTEM_PROMPT 赋值"
    value = targets[0].value
    assert isinstance(value, (ast.Constant, ast.JoinedStr)), (
        f"_SYSTEM_PROMPT 的右侧是 {type(value).__name__}，不再是字符串字面量 —— "
        f"它已经变成可执行逻辑，需要补行为断言")

    assert "Allowed operators and fields" in AA._SYSTEM_PROMPT, (
        "提示词里的白名单说明行被改写了 —— 请复核本用例的前提")


def test_the_operator_whitelist_matches_what_the_parser_accepts():
    """
    比"那行是不是散文"更有价值的一条：白名单里列的算子与字段，
    解析器必须真的都认。

    `_OP_WHITELIST` 是 `AlphaAgent` **自己**的一份清单
    （与 `_prompts.py` 那份是两张表），脱节之后 LLM 会照着它写出
    解析不了的公式，然后在 `_validate_and_fix` 里被静默丢弃。
    """
    import re

    from app.core.alpha_engine.parser import Parser
    from app.core.alpha_engine.validator import AlphaValidator

    parser, validator = Parser(), AlphaValidator()
    args = {
        "ts_mean": "close,20", "ts_std": "close,20", "ts_delta": "close,20",
        "ts_delay": "close,20", "ts_max": "close,20", "ts_min": "close,20",
        "ts_rank": "close,20", "ts_decay_linear": "close,20",
        "rank": "close", "zscore": "close", "scale": "close",
        "log": "close", "abs": "close", "sqrt": "close", "sign": "close",
        "signed_power": "close,2", "if_else": "close,close,close",
    }
    names = {m.group(1) for m in re.finditer(r"(\w+)\(", AA._OP_WHITELIST)}
    assert names, "白名单里解析不出任何算子名"

    unknown = sorted(n for n in names if n not in args)
    assert not unknown, (
        f"白名单里出现了本用例不认识的算子 {unknown} —— "
        f"请补上它的实参，或确认它是否真的存在")
    for n in sorted(names):
        validator.validate(parser.parse(f"{n}({args[n]})"))

    # 字段部分（白名单第一段，逗号分隔）
    fields = [f.strip() for f in AA._OP_WHITELIST.split("|")[0].split(",")]
    assert "close" in fields, f"白名单的字段段落解析异常：{fields}"
    for f in fields:
        parser.parse(f"rank({f})")
