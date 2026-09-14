"""
agent/_lc_agent.py —— QuantTools 接进 LangChain AgentExecutor 的那层接线

**此前零测试**（5 个变异点，D 档）。

这个模块一行金融逻辑都没有，全是**接线**。但接线错了的后果不轻，
而且全部是静默的：

  - 少注册一个工具 → LLM 永远不会调用它。对外表现只是"agent 好像
    不太会用 GP"，没有任何报错。
  - 工具参数忘了透传（比如 `factor_family`）→ GP 退回随机权重，
    结果依然返回一条 DSL，看不出偏好没生效。
  - `max_iterations` 被调大 → 一次对话可能烧掉几十次 LLM 调用；
    在"没有任何预算"的模拟阶段这是实打实的钱。
  - `MessagesPlaceholder("chat_history")` 漏掉 → 多轮记忆整条失效，
    "优化刚才那条"永远找不到上一条。
  - `chat_store` 传了却没包 `RunnableWithMessageHistory` →
    DB 持久化记忆形同虚设。

**为什么全部用假 langchain**：本机装的是 langchain 1.2.15，
`AgentExecutor` 已经不在 `langchain.agents` 里了（已登记为缺陷 D-3），
真实 import 必然失败。而本文件要验的是**我们自己的接线**，
不是 langchain 的实现 —— 所以往 `sys.modules` 注入可编程的替身，
让接线契约在任何 langchain 版本下都能被检查。
"""
from __future__ import annotations

import sys
import types

import pytest

from app.agent._lc_agent import _build_langchain_agent

#: `lc_tools` 列表的顺序 = 提示词里给 LLM 的工具顺序，
#: 模块 docstring 明确写了这个顺序（先解释、再 GP、最后保存）。
EXPECTED_TOOL_ORDER = [
    "tool_generate_alpha_dsl",
    "tool_interpret_factor",
    "tool_run_gp_optimization",
    "tool_run_backtest",
    "tool_mutate_ast",
    "tool_run_optuna",
    "tool_save_alpha",
]


# ---------------------------------------------------------------------------
# 假 langchain
# ---------------------------------------------------------------------------

class _FakeTool:
    """`@lc_tool` 的产物：保留原函数与它的名字，便于直接调用与断言。"""

    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self.description = fn.__doc__ or ""

    def __call__(self, *a, **k):
        return self.fn(*a, **k)


class _FakeExecutor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeChain:
    def __init__(self, runnable, **kwargs):
        self.runnable = runnable
        self.kwargs = kwargs


class _FakePrompt:
    def __init__(self, messages):
        self.messages = messages

    @classmethod
    def from_messages(cls, messages):
        return cls(list(messages))


class _FakePlaceholder:
    def __init__(self, name, optional=False):
        self.name = name
        self.optional = optional

    def __repr__(self):
        return f"Placeholder({self.name!r}, optional={self.optional})"


@pytest.fixture
def lc(monkeypatch):
    """
    注入一套假 langchain，并记录
    `create_tool_calling_agent` / `AgentExecutor` / `RunnableWithMessageHistory`
    收到的全部参数。
    """
    seen = {}

    def create_tool_calling_agent(llm, tools, prompt):
        seen["llm"] = llm
        seen["agent_tools"] = list(tools)
        seen["prompt"] = prompt
        return ("AGENT", llm)

    def executor(**kwargs):
        seen["executor_kwargs"] = dict(kwargs)
        return _FakeExecutor(**kwargs)

    def history_wrapper(runnable, **kwargs):
        seen["history_runnable"] = runnable
        seen["history_kwargs"] = dict(kwargs)
        return _FakeChain(runnable, **kwargs)

    monkeypatch.setitem(sys.modules, "langchain.agents", types.SimpleNamespace(
        AgentExecutor=executor,
        create_tool_calling_agent=create_tool_calling_agent))
    monkeypatch.setitem(sys.modules, "langchain.tools", types.SimpleNamespace(
        tool=_FakeTool))
    monkeypatch.setitem(sys.modules, "langchain_core.prompts",
                        types.SimpleNamespace(
                            ChatPromptTemplate=_FakePrompt,
                            MessagesPlaceholder=_FakePlaceholder))
    monkeypatch.setitem(sys.modules, "langchain_core.runnables.history",
                        types.SimpleNamespace(
                            RunnableWithMessageHistory=history_wrapper))
    return types.SimpleNamespace(seen=seen, monkeypatch=monkeypatch)


class _Tools:
    """QuantTools 替身：每个工具方法把收到的参数原样记下来。"""

    def __init__(self):
        self.calls = []

    def _rec(self, _tool_name, **kw):
        # 形参名故意用 `_tool_name`：`tool_save_alpha` 自己就有一个叫
        # `name` 的参数，写成 `name` 会撞成 "multiple values for argument"。
        self.calls.append((_tool_name, kw))
        return f"{_tool_name}:ok"

    def tool_generate_alpha_dsl(self, hypothesis):
        return self._rec("tool_generate_alpha_dsl", hypothesis=hypothesis)

    def tool_run_gp_optimization(self, **kw):
        return self._rec("tool_run_gp_optimization", **kw)

    def tool_run_backtest(self, dsl, config_json, use_test_set):
        return self._rec("tool_run_backtest", dsl=dsl, config_json=config_json,
                         use_test_set=use_test_set)

    def tool_mutate_ast(self, current_dsl, overfit_reason, mutation_target):
        return self._rec("tool_mutate_ast", current_dsl=current_dsl,
                         overfit_reason=overfit_reason,
                         mutation_target=mutation_target)

    def tool_run_optuna(self, dsl, n_trials):
        return self._rec("tool_run_optuna", dsl=dsl, n_trials=n_trials)

    def tool_save_alpha(self, name, dsl, metrics_json):
        return self._rec("tool_save_alpha", name=name, dsl=dsl,
                         metrics_json=metrics_json)

    def tool_interpret_factor(self, dsl, metrics_json):
        return self._rec("tool_interpret_factor", dsl=dsl,
                         metrics_json=metrics_json)


def _tools_by_name(seen) -> dict:
    return {t.name: t for t in seen["agent_tools"]}


# ===========================================================================
# A. 工具清单
# ===========================================================================

class TestToolRegistry:

    def test_all_seven_tools_are_registered(self, lc):
        _build_langchain_agent("LLM", _Tools())
        assert [t.name for t in lc.seen["agent_tools"]] == EXPECTED_TOOL_ORDER, (
            f"注册的工具是 {[t.name for t in lc.seen['agent_tools']]}，"
            f"应当是 {EXPECTED_TOOL_ORDER} —— 少一个 LLM 就永远不会调用它")

    def test_the_executor_gets_the_same_tool_list_as_the_agent(self, lc):
        """
        `AgentExecutor(agent=..., tools=lc_tools)` —— 两处必须是同一份。
        执行器少一个工具，LLM 会发出调用但执行器找不到，
        触发 `handle_parsing_errors` 走进兜底，看起来像"模型答非所问"。
        """
        _build_langchain_agent("LLM", _Tools())
        exec_tools = lc.seen["executor_kwargs"]["tools"]
        assert [t.name for t in exec_tools] == EXPECTED_TOOL_ORDER
        assert exec_tools == lc.seen["agent_tools"]

    def test_interpretation_is_offered_before_gp(self, lc):
        """
        模块 docstring 与系统提示词都要求"先 interpret 再 GP"。
        清单顺序是给 LLM 的第一层暗示 —— 两者位置对调会让
        `factor_family` 的抽取-回传链条在提示词层面失去支撑。
        """
        _build_langchain_agent("LLM", _Tools())
        names = [t.name for t in lc.seen["agent_tools"]]
        assert names.index("tool_interpret_factor") < \
               names.index("tool_run_gp_optimization"), (
            "tool_interpret_factor 排在了 GP 之后")

    def test_optuna_is_offered_after_gp(self, lc):
        """GP 是主优化器、Optuna 是次要的 —— 顺序体现优先级。"""
        _build_langchain_agent("LLM", _Tools())
        names = [t.name for t in lc.seen["agent_tools"]]
        assert names.index("tool_run_gp_optimization") < \
               names.index("tool_run_optuna")

    def test_every_registered_tool_has_a_docstring(self, lc):
        """
        工具的 docstring **就是**给 LLM 看的说明书。
        空描述的工具模型不知道什么时候该用，等于没注册。
        """
        _build_langchain_agent("LLM", _Tools())
        for t in lc.seen["agent_tools"]:
            assert t.description.strip(), f"{t.name} 没有描述，LLM 无从判断何时调用"
            assert len(t.description.strip()) >= 40, (
                f"{t.name} 的描述过短（{len(t.description.strip())} 字符）")

    def test_every_registered_tool_maps_to_a_real_quanttools_method(self, lc):
        """
        接线层的工具名与 `QuantTools` 的方法名必须一一对应。
        改了一边没改另一边，会在 LLM 真的调用时才 AttributeError。
        """
        from app.agent._tools import QuantTools
        _build_langchain_agent("LLM", _Tools())
        for t in lc.seen["agent_tools"]:
            assert hasattr(QuantTools, t.name), (
                f"注册了 {t.name}，但 QuantTools 上没有同名方法")


# ===========================================================================
# B. 参数透传 —— 每个工具单独调一次
# ===========================================================================

class TestArgumentForwarding:

    def test_generate_alpha_dsl_forwards_the_hypothesis(self, lc):
        tools = _Tools()
        _build_langchain_agent("LLM", tools)
        _tools_by_name(lc.seen)["tool_generate_alpha_dsl"]("动量在科技股更强")
        assert tools.calls == [("tool_generate_alpha_dsl",
                                {"hypothesis": "动量在科技股更强"})]

    def test_gp_optimization_forwards_every_named_parameter(self, lc):
        """
        八个参数逐个透传。漏掉 `factor_family` 是**最贵**的一个：
        GP 的算子权重退回默认，金融先验完全不起作用，
        而返回值一切正常 —— 这正是模块 docstring 反复强调的那条链路。
        """
        tools = _Tools()
        _build_langchain_agent("LLM", tools)
        _tools_by_name(lc.seen)["tool_run_gp_optimization"](
            seed_dsl="rank(close)", n_generations=9, pop_size=33,
            n_optuna_trials=4, factor_family="momentum",
            dataset_name="us_tech_large", dataset_start="2020-01-01",
            dataset_end="2022-01-01")
        name, kw = tools.calls[0]
        assert name == "tool_run_gp_optimization"
        assert kw == {"seed_dsl": "rank(close)", "n_generations": 9,
                      "pop_size": 33, "n_optuna_trials": 4,
                      "factor_family": "momentum",
                      "dataset_name": "us_tech_large",
                      "dataset_start": "2020-01-01",
                      "dataset_end": "2022-01-01"}, (
            f"GP 工具的参数透传不完整：{kw}")

    def test_gp_optimization_defaults_match_the_documented_ones(self, lc):
        """
        默认值直接决定一次无参调用要烧多少算力/token：
        `n_generations=4, pop_size=12, n_optuna_trials=8`。
        被改大在"没有任何预算"的阶段是实打实的成本。
        """
        tools = _Tools()
        _build_langchain_agent("LLM", tools)
        _tools_by_name(lc.seen)["tool_run_gp_optimization"]()
        _, kw = tools.calls[0]
        assert kw == {"seed_dsl": "", "n_generations": 4, "pop_size": 12,
                      "n_optuna_trials": 8, "factor_family": "",
                      "dataset_name": "", "dataset_start": "2021-01-01",
                      "dataset_end": "2024-01-01"}, (
            f"GP 工具的默认参数被改动：{kw}")

    def test_backtest_forwards_positionally_in_the_right_order(self, lc):
        """
        `tools_obj.tool_run_backtest(dsl, config_json, use_test_set)`
        —— 位置传参，任意两个对调会让 `config_json` 变成 DSL。
        用三个**类型不同**的值确保对调必被发现。
        """
        tools = _Tools()
        _build_langchain_agent("LLM", tools)
        _tools_by_name(lc.seen)["tool_run_backtest"](
            dsl="rank(vwap)", config_json='{"delay":1}', use_test_set=True)
        assert tools.calls[0][1] == {"dsl": "rank(vwap)",
                                     "config_json": '{"delay":1}',
                                     "use_test_set": True}

    def test_backtest_defaults_to_the_validate_set_not_the_holdout(self, lc):
        """
        `use_test_set: bool = False` —— 默认**不**动真实留出集。
        翻成 True 会让每次随手回测都消耗一次性的 Test 集，
        最终选出来的因子再也没有干净的样本外可验。
        """
        tools = _Tools()
        _build_langchain_agent("LLM", tools)
        _tools_by_name(lc.seen)["tool_run_backtest"](dsl="rank(close)")
        assert tools.calls[0][1]["use_test_set"] is False, (
            "回测默认动用了真实留出集 —— Test 集会被反复消耗")
        assert tools.calls[0][1]["config_json"] == "{}"

    def test_mutate_ast_forwards_the_mutation_target(self, lc):
        tools = _Tools()
        _build_langchain_agent("LLM", tools)
        _tools_by_name(lc.seen)["tool_mutate_ast"](
            "rank(close)", "turnover", "add_ts_smoothing")
        assert tools.calls[0][1] == {"current_dsl": "rank(close)",
                                     "overfit_reason": "turnover",
                                     "mutation_target": "add_ts_smoothing"}

    def test_mutate_ast_defaults_to_undirected_mutation(self, lc):
        tools = _Tools()
        _build_langchain_agent("LLM", tools)
        _tools_by_name(lc.seen)["tool_mutate_ast"]("rank(close)")
        assert tools.calls[0][1] == {"current_dsl": "rank(close)",
                                     "overfit_reason": "",
                                     "mutation_target": ""}

    def test_optuna_forwards_the_trial_count(self, lc):
        tools = _Tools()
        _build_langchain_agent("LLM", tools)
        _tools_by_name(lc.seen)["tool_run_optuna"]("rank(close)", 25)
        assert tools.calls[0][1] == {"dsl": "rank(close)", "n_trials": 25}

    def test_optuna_defaults_to_ten_trials(self, lc):
        tools = _Tools()
        _build_langchain_agent("LLM", tools)
        _tools_by_name(lc.seen)["tool_run_optuna"]("rank(close)")
        assert tools.calls[0][1]["n_trials"] == 10

    def test_save_alpha_forwards_name_dsl_and_metrics(self, lc):
        tools = _Tools()
        _build_langchain_agent("LLM", tools)
        _tools_by_name(lc.seen)["tool_save_alpha"]("我的因子", "rank(close)",
                                                   '{"oos_sharpe":1.1}')
        assert tools.calls[0][1] == {"name": "我的因子", "dsl": "rank(close)",
                                     "metrics_json": '{"oos_sharpe":1.1}'}

    def test_interpret_factor_forwards_the_metrics_json(self, lc):
        tools = _Tools()
        _build_langchain_agent("LLM", tools)
        _tools_by_name(lc.seen)["tool_interpret_factor"]("rank(close)",
                                                         '{"is_turnover":3}')
        assert tools.calls[0][1] == {"dsl": "rank(close)",
                                     "metrics_json": '{"is_turnover":3}'}

    def test_interpret_factor_works_without_metrics(self, lc):
        tools = _Tools()
        _build_langchain_agent("LLM", tools)
        _tools_by_name(lc.seen)["tool_interpret_factor"]("rank(close)")
        assert tools.calls[0][1]["metrics_json"] == "{}"

    def test_each_tool_returns_what_quanttools_returned(self, lc):
        """接线层不能吞掉或改写返回值 —— LLM 看到的必须是工具的原文。"""
        tools = _Tools()
        _build_langchain_agent("LLM", tools)
        by = _tools_by_name(lc.seen)
        assert by["tool_generate_alpha_dsl"]("h") == "tool_generate_alpha_dsl:ok"
        assert by["tool_run_optuna"]("d") == "tool_run_optuna:ok"
        assert by["tool_save_alpha"]("n", "d") == "tool_save_alpha:ok"


# ===========================================================================
# C. 提示词结构
# ===========================================================================

class TestPromptShape:

    def test_the_prompt_has_the_four_documented_slots_in_order(self, lc):
        """
        `[system, chat_history(optional), human input, agent_scratchpad]`

        顺序或组成被改的后果：
          - `chat_history` 丢了 → 多轮记忆失效（"优化刚才那条"找不到）
          - `agent_scratchpad` 丢了 → 工具调用的中间结果传不回模型，
            AgentExecutor 会在第二轮直接崩
          - system 不在第一位 → 整套金融先验被当成普通对话内容
        """
        _build_langchain_agent("LLM", _Tools())
        msgs = lc.seen["prompt"].messages
        assert len(msgs) == 4, f"提示词有 {len(msgs)} 段，应当是 4 段"

        assert isinstance(msgs[0], tuple) and msgs[0][0] == "system"
        assert isinstance(msgs[1], _FakePlaceholder) and msgs[1].name == "chat_history"
        assert isinstance(msgs[2], tuple) and msgs[2] == ("human", "{input}")
        assert isinstance(msgs[3], _FakePlaceholder) and \
               msgs[3].name == "agent_scratchpad"

    def test_the_system_slot_carries_the_project_system_prompt(self, lc):
        """
        `("system", _SYSTEM_PROMPT)` —— 换成别的字符串会让整套
        因子分类学、工作流 A/B、参数链条全部消失，而 agent 照样能聊天。
        """
        from app.agent._prompts import _SYSTEM_PROMPT
        _build_langchain_agent("LLM", _Tools())
        assert lc.seen["prompt"].messages[0][1] is _SYSTEM_PROMPT

    def test_the_chat_history_placeholder_is_optional(self, lc):
        """
        `MessagesPlaceholder("chat_history", optional=True)`
        —— `optional` 翻成 False 会让**第一轮**对话（还没有历史）
        直接抛 KeyError，agent 从一开始就不可用。
        """
        _build_langchain_agent("LLM", _Tools())
        assert lc.seen["prompt"].messages[1].optional is True, (
            "chat_history 不是 optional —— 第一轮没有历史时会抛错")

    def test_the_agent_scratchpad_placeholder_is_required(self, lc):
        """反过来，scratchpad 必须是必填 —— 它每一轮都有内容。"""
        _build_langchain_agent("LLM", _Tools())
        assert lc.seen["prompt"].messages[3].optional is False

    def test_the_llm_is_handed_to_the_agent_unchanged(self, lc):
        sentinel = object()
        _build_langchain_agent(sentinel, _Tools())
        assert lc.seen["llm"] is sentinel


# ===========================================================================
# D. 执行器配置 —— 直接决定一次对话的成本上限
# ===========================================================================

class TestExecutorConfiguration:

    def test_the_iteration_cap_is_fifteen(self, lc):
        """
        `max_iterations = 15` —— 这是**一次对话的 LLM 调用上限**。
        被调大（或删掉 → 无上限）在模拟阶段是直接烧钱；
        被调小则会让完整的 Workflow A（7 步 + 工具往返）跑不完，
        表现为"agent 每次都半途而废"。
        """
        _build_langchain_agent("LLM", _Tools())
        assert lc.seen["executor_kwargs"]["max_iterations"] == 15, (
            f"迭代上限是 {lc.seen['executor_kwargs']['max_iterations']}，应当是 15")

    def test_parsing_errors_are_handled_rather_than_raised(self, lc):
        """
        `handle_parsing_errors = True` —— 翻成 False 会让模型一次
        格式不规范的工具调用把整个请求打成 500，用户看到的是白屏。
        """
        _build_langchain_agent("LLM", _Tools())
        assert lc.seen["executor_kwargs"]["handle_parsing_errors"] is True

    def test_verbose_logging_is_off(self, lc):
        """`verbose = False` —— 翻成 True 会把完整提示词（含系统提示）刷进日志。"""
        _build_langchain_agent("LLM", _Tools())
        assert lc.seen["executor_kwargs"]["verbose"] is False

    def test_the_agent_object_is_passed_to_the_executor(self, lc):
        _build_langchain_agent("LLM", _Tools())
        assert lc.seen["executor_kwargs"]["agent"] == ("AGENT", "LLM")


# ===========================================================================
# E. 会话记忆包装
# ===========================================================================

class TestHistoryWrapping:

    def test_without_a_chat_store_the_bare_executor_is_returned(self, lc):
        """
        `if chat_store is not None:` —— `not` 被删会在没有 store 时
        也去构造 `RunnableWithMessageHistory`，
        `_make_history_factory(None)` 之后每次取历史都会炸。
        """
        out = _build_langchain_agent("LLM", _Tools())
        assert isinstance(out, _FakeExecutor)
        assert "history_kwargs" not in lc.seen

    def test_with_a_chat_store_the_executor_is_wrapped(self, lc):
        store = object()
        out = _build_langchain_agent("LLM", _Tools(), chat_store=store)
        assert isinstance(out, _FakeChain), (
            "传了 chat_store 却没有包上会话历史 —— DB 持久化记忆形同虚设")
        assert isinstance(lc.seen["history_runnable"], _FakeExecutor)

    def test_the_message_keys_match_the_prompt_placeholders(self, lc):
        """
        `input_messages_key="input"` / `history_messages_key="chat_history"`

        这两个键必须与提示词里的 `{input}` 和
        `MessagesPlaceholder("chat_history")` **同名**。
        任一被改，历史就注不进提示词 —— 记忆静默失效，
        而每一轮对话依然正常返回。
        """
        _build_langchain_agent("LLM", _Tools(), chat_store=object())
        kw = lc.seen["history_kwargs"]
        assert kw["input_messages_key"] == "input"
        assert kw["history_messages_key"] == "chat_history"

        msgs = lc.seen["prompt"].messages
        assert msgs[2][1] == "{" + kw["input_messages_key"] + "}", (
            "输入键与提示词里的占位符对不上")
        assert msgs[1].name == kw["history_messages_key"], (
            "历史键与 MessagesPlaceholder 的名字对不上")

    def test_the_history_factory_is_built_from_the_given_store(self, lc):
        """`get_session_history = _make_history_factory(chat_store)` —— 传的是同一个 store。"""
        import app.agent._lc_agent as LC
        seen = {}
        lc.monkeypatch.setattr(LC, "_make_history_factory",
                               lambda store: seen.setdefault("store", store))
        store = object()
        _build_langchain_agent("LLM", _Tools(), chat_store=store)
        assert seen["store"] is store
        assert lc.seen["history_kwargs"]["get_session_history"] is store

    def test_an_unavailable_history_wrapper_degrades_to_the_bare_executor(self, lc):
        """
        `except ImportError: logger.warning(...)` 然后落到 `return executor`
        —— 这条兜底被删会让老版本 langchain_core 上整个 agent 构建失败，
        而不是退成"能对话但没有持久记忆"。
        """
        lc.monkeypatch.setitem(sys.modules, "langchain_core.runnables.history",
                               None)
        out = _build_langchain_agent("LLM", _Tools(), chat_store=object())
        assert isinstance(out, _FakeExecutor), (
            "会话历史包装不可用时没有退回裸执行器")


# ===========================================================================
# F. 依赖缺失的报错
# ===========================================================================

class TestMissingDependency:

    def test_a_missing_langchain_raises_an_actionable_import_error(self,
                                                                   monkeypatch):
        """
        `raise ImportError("需要安装 langchain 和 langchain-openai: ...") from exc`

        原样冒出的 ImportError 只说 "cannot import name 'AgentExecutor'"，
        看不出该装什么。这条同时也是缺陷 D-3 的现状锚点：
        当前**任何**调用都会走到这里（装的是不兼容的 langchain 1.x）。
        """
        monkeypatch.setitem(sys.modules, "langchain.agents", None)
        with pytest.raises(ImportError) as ei:
            _build_langchain_agent("LLM", _Tools())
        assert "pip install langchain" in str(ei.value)
        assert ei.value.__cause__ is not None, (
            "没有用 `raise ... from exc` 串上原因，traceback 断了")

    def test_the_real_environment_currently_takes_the_failure_path(self):
        """
        **钉住现状**（缺陷 D-3，本阶段只登记不修）：
        不注入任何替身时，真实依赖下 `_build_langchain_agent` 必定抛 ImportError。

        与 `tests/meta/test_known_defects.py::TestLangChainWiringIsAlive` 成对：
        那条断言"应该能构建成功"（xfail），这条钉住"现在构建不成功"（绿）。
        修好依赖之后两条一起改。
        """
        with pytest.raises(ImportError, match="pip install langchain"):
            _build_langchain_agent("LLM", _Tools())
