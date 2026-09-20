"""
agent/_lc_agent.py —— QuantTools 接进 LangChain agent 的那层接线

**此前零测试**（5 个变异点，D 档）。

这个模块一行金融逻辑都没有，全是**接线**。但接线错了的后果不轻，
而且全部是静默的：

  - 少注册一个工具 → LLM 永远不会调用它。对外表现只是"agent 好像
    不太会用 GP"，没有任何报错。
  - 工具参数忘了透传（比如 `factor_family`）→ GP 退回随机权重，
    结果依然返回一条 DSL，看不出偏好没生效。
  - 单轮模型调用上限被调大 → 一次对话可能烧掉几十次 LLM 调用；
    在"没有任何预算"的模拟阶段这是实打实的钱。
  - 多轮记忆断掉 → "优化刚才那条"永远找不到上一条。
  - `chat_store` 传了却没接上 → DB 持久化记忆形同虚设。

后三条在 2026-09-20 迁到 `create_agent` 之后由
tests/unit/agent/test_lc_agent_migration.py 用**行为**断言守着
（真跑两轮、真打到上限），本文件专注工具清单与参数透传。

**2026-09-20：不再 mock langchain。**

本文件原先整套往 `sys.modules` 注入假 langchain，理由写在这里：
"本机装的是 1.2.15，`AgentExecutor` 已经不在 `langchain.agents` 里了
（缺陷 D-3），真实 import 必然失败"。

D-3 修好之后那个理由不成立了 —— 而"把构建器 mock 掉再宣称兼容"恰恰是
D-3 能潜伏这么久的原因：声明的依赖装出来的环境根本建不出 agent，
而这一层的测试全绿。

现在用**真的** `@lc_tool`：接线（工具清单、顺序、参数透传）由
`_build_tools` 单独构造，不需要 mock 任何 langchain 符号就能验。
被测的仍然是**我们自己的接线**，不是 langchain 的实现。
"""
from __future__ import annotations

import pytest

from app.agent._lc_agent import _build_tools

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
# QuantTools 替身（真实的 langchain，假的业务对象）
# ---------------------------------------------------------------------------

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


def _built(tools_obj=None):
    """用**真的** `@lc_tool` 构造工具清单。"""
    return _build_tools(tools_obj if tools_obj is not None else _Tools())


def _tools_by_name(tools) -> dict:
    return {t.name: t for t in tools}


# ===========================================================================
# A. 工具清单
# ===========================================================================

class TestToolRegistry:

    def test_all_seven_tools_are_registered(self):
        tools = _built()
        assert [t.name for t in tools] == EXPECTED_TOOL_ORDER, (
            f"注册的工具是 {[t.name for t in tools]}，"
            f"应当是 {EXPECTED_TOOL_ORDER} —— 少一个 LLM 就永远不会调用它")

    # `test_the_executor_gets_the_same_tool_list_as_the_agent` 已删除：
    # 旧结构里 agent 与 executor 各拿一份工具清单，两份可能不一致，所以要对照。
    # `create_agent` 只有一份 —— 那个失配根本不存在了，对照也就无从谈起。
    # "模型拿到的就是 _build_tools 的产物"由 C 节
    # test_the_tool_list_handed_to_the_model_is_the_built_one 守着
    # （断言 bind_tools 实际收到了什么，不是读源码）。

    def test_interpretation_is_offered_before_gp(self):
        """
        模块 docstring 与系统提示词都要求"先 interpret 再 GP"。
        清单顺序是给 LLM 的第一层暗示 —— 两者位置对调会让
        `factor_family` 的抽取-回传链条在提示词层面失去支撑。
        """
        names = [t.name for t in _built()]
        assert names.index("tool_interpret_factor") < \
               names.index("tool_run_gp_optimization"), (
            "tool_interpret_factor 排在了 GP 之后")

    def test_optuna_is_offered_after_gp(self):
        """GP 是主优化器、Optuna 是次要的 —— 顺序体现优先级。"""
        names = [t.name for t in _built()]
        assert names.index("tool_run_gp_optimization") < \
               names.index("tool_run_optuna")

    def test_every_registered_tool_has_a_docstring(self):
        """
        工具的 docstring **就是**给 LLM 看的说明书。
        空描述的工具模型不知道什么时候该用，等于没注册。
        """
        for t in _built():
            assert t.description.strip(), f"{t.name} 没有描述，LLM 无从判断何时调用"
            assert len(t.description.strip()) >= 40, (
                f"{t.name} 的描述过短（{len(t.description.strip())} 字符）")

    def test_every_registered_tool_maps_to_a_real_quanttools_method(self):
        """
        接线层的工具名与 `QuantTools` 的方法名必须一一对应。
        改了一边没改另一边，会在 LLM 真的调用时才 AttributeError。
        """
        from app.agent._tools import QuantTools
        for t in _built():
            assert hasattr(QuantTools, t.name), (
                f"注册了 {t.name}，但 QuantTools 上没有同名方法")


# ===========================================================================
# B. 参数透传 —— 每个工具单独调一次
# ===========================================================================

class TestArgumentForwarding:

    def test_generate_alpha_dsl_forwards_the_hypothesis(self):
        tools = _Tools()
        built = _tools_by_name(_built(tools))
        built["tool_generate_alpha_dsl"].func("动量在科技股更强")
        assert tools.calls == [("tool_generate_alpha_dsl",
                                {"hypothesis": "动量在科技股更强"})]

    def test_gp_optimization_forwards_every_named_parameter(self):
        """
        八个参数逐个透传。漏掉 `factor_family` 是**最贵**的一个：
        GP 的算子权重退回默认，金融先验完全不起作用，
        而返回值一切正常 —— 这正是模块 docstring 反复强调的那条链路。
        """
        tools = _Tools()
        built = _tools_by_name(_built(tools))
        built["tool_run_gp_optimization"].func(
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

    def test_gp_optimization_defaults_match_the_documented_ones(self):
        """
        默认值直接决定一次无参调用要烧多少算力/token：
        `n_generations=4, pop_size=12, n_optuna_trials=8`。
        被改大在"没有任何预算"的阶段是实打实的成本。
        """
        tools = _Tools()
        built = _tools_by_name(_built(tools))
        built["tool_run_gp_optimization"].func()
        _, kw = tools.calls[0]
        assert kw == {"seed_dsl": "", "n_generations": 4, "pop_size": 12,
                      "n_optuna_trials": 8, "factor_family": "",
                      "dataset_name": "", "dataset_start": "2021-01-01",
                      "dataset_end": "2024-01-01"}, (
            f"GP 工具的默认参数被改动：{kw}")

    def test_backtest_forwards_positionally_in_the_right_order(self):
        """
        `tools_obj.tool_run_backtest(dsl, config_json, use_test_set)`
        —— 位置传参，任意两个对调会让 `config_json` 变成 DSL。
        用三个**类型不同**的值确保对调必被发现。
        """
        tools = _Tools()
        built = _tools_by_name(_built(tools))
        built["tool_run_backtest"].func(
            dsl="rank(vwap)", config_json='{"delay":1}', use_test_set=True)
        assert tools.calls[0][1] == {"dsl": "rank(vwap)",
                                     "config_json": '{"delay":1}',
                                     "use_test_set": True}

    def test_backtest_defaults_to_the_validate_set_not_the_holdout(self):
        """
        `use_test_set: bool = False` —— 默认**不**动真实留出集。
        翻成 True 会让每次随手回测都消耗一次性的 Test 集，
        最终选出来的因子再也没有干净的样本外可验。
        """
        tools = _Tools()
        built = _tools_by_name(_built(tools))
        built["tool_run_backtest"].func(dsl="rank(close)")
        assert tools.calls[0][1]["use_test_set"] is False, (
            "回测默认动用了真实留出集 —— Test 集会被反复消耗")
        assert tools.calls[0][1]["config_json"] == "{}"

    def test_mutate_ast_forwards_the_mutation_target(self):
        tools = _Tools()
        built = _tools_by_name(_built(tools))
        built["tool_mutate_ast"].func(
            "rank(close)", "turnover", "add_ts_smoothing")
        assert tools.calls[0][1] == {"current_dsl": "rank(close)",
                                     "overfit_reason": "turnover",
                                     "mutation_target": "add_ts_smoothing"}

    def test_mutate_ast_defaults_to_undirected_mutation(self):
        tools = _Tools()
        built = _tools_by_name(_built(tools))
        built["tool_mutate_ast"].func("rank(close)")
        assert tools.calls[0][1] == {"current_dsl": "rank(close)",
                                     "overfit_reason": "",
                                     "mutation_target": ""}

    def test_optuna_forwards_the_trial_count(self):
        tools = _Tools()
        built = _tools_by_name(_built(tools))
        built["tool_run_optuna"].func("rank(close)", 25)
        assert tools.calls[0][1] == {"dsl": "rank(close)", "n_trials": 25}

    def test_optuna_defaults_to_ten_trials(self):
        tools = _Tools()
        built = _tools_by_name(_built(tools))
        built["tool_run_optuna"].func("rank(close)")
        assert tools.calls[0][1]["n_trials"] == 10

    def test_save_alpha_forwards_name_dsl_and_metrics(self):
        tools = _Tools()
        built = _tools_by_name(_built(tools))
        built["tool_save_alpha"].func("我的因子", "rank(close)",
                                                   '{"oos_sharpe":1.1}')
        assert tools.calls[0][1] == {"name": "我的因子", "dsl": "rank(close)",
                                     "metrics_json": '{"oos_sharpe":1.1}'}

    def test_interpret_factor_forwards_the_metrics_json(self):
        tools = _Tools()
        built = _tools_by_name(_built(tools))
        built["tool_interpret_factor"].func("rank(close)",
                                                         '{"is_turnover":3}')
        assert tools.calls[0][1] == {"dsl": "rank(close)",
                                     "metrics_json": '{"is_turnover":3}'}

    def test_interpret_factor_works_without_metrics(self):
        tools = _Tools()
        built = _tools_by_name(_built(tools))
        built["tool_interpret_factor"].func("rank(close)")
        assert tools.calls[0][1]["metrics_json"] == "{}"

    def test_each_tool_returns_what_quanttools_returned(self):
        """接线层不能吞掉或改写返回值 —— LLM 看到的必须是工具的原文。"""
        tools = _Tools()
        by = _tools_by_name(_built(tools))
        assert by["tool_generate_alpha_dsl"].func("h") == "tool_generate_alpha_dsl:ok"
        assert by["tool_run_optuna"].func("d") == "tool_run_optuna:ok"
        assert by["tool_save_alpha"].func("n", "d") == "tool_save_alpha:ok"


# ===========================================================================
# C. 组装：系统提示词与成本上限
# ===========================================================================

class TestAgentAssembly:
    """
    原先这里有四节（prompt 形状 / executor 配置 / history 包装 / 缺依赖），
    钉的全是 `ChatPromptTemplate` + `AgentExecutor(**kwargs)` 的**结构**。
    2026-09-20 迁到 `create_agent` 之后那些 kwargs 不存在了。

    **它们的意图没有消失，改成了行为断言**，去向逐条列在这里 ——
    删掉一节结构测试却不说它去哪了，等于悄悄少了一块覆盖：

      · prompt 的 system 槽携带 _SYSTEM_PROMPT
            → 本节 test_the_system_prompt_reaches_the_agent
      · chat_history 占位符（多轮记忆）
            → test_lc_agent_migration.py::TestSessionHistory（4 条，真跑两轮）
      · max_iterations == 15（成本上限）
            → 本节 test_the_model_call_budget_is_still_fifteen
              + migration::test_the_model_call_limit_ends_gracefully（真打到上限）
      · handle_parsing_errors is True
            → migration::test_a_raising_tool_becomes_an_observation_not_a_crash
      · RunnableWithMessageHistory 包装 / 不包装
            → migration::TestSessionHistory + TestTheOldContractIsPreserved
      · 缺依赖时报错可操作
            → migration::test_the_incompatibility_error_names_the_version_not_a_missing_package
    """

    def test_the_system_prompt_reaches_the_model(self):
        """
        提示词没接上，LLM 就拿不到因子分类学与 DSL 约定，产出退化成泛泛的聊天。

        **验的是模型真的收到了**，不是源码里有 `system_prompt=` 这几个字：
        用一个记录收到消息的替身跑一轮，在它看到的消息里找提示词原文。
        """
        from app.agent import _lc_agent as LC
        from app.agent._prompts import _SYSTEM_PROMPT
        from tests.unit.agent.test_lc_agent_migration import ScriptedModel

        model = ScriptedModel(n_tool_rounds=0)
        agent = LC._build_langchain_agent(model, _Tools(), None)
        agent.invoke({"input": "hi"})

        blob = chr(10).join(str(getattr(m, "content", ""))
                            for m in model.seen_messages)
        probe = _SYSTEM_PROMPT.strip().splitlines()[0][:60]
        assert probe and probe in blob, (
            f"模型收到的消息里找不到系统提示词的开头 {probe!r} —— "
            f"create_agent 没拿到 _SYSTEM_PROMPT")

    def test_the_model_call_budget_is_still_fifteen(self):
        """
        单轮对话的模型调用上限。旧 `AgentExecutor(max_iterations=15)` 的等价物。

        被改大在"没有任何预算"的模拟阶段是实打实的钱 —— 这条守的是成本，
        与 migration 里那条"到顶要优雅收尾"配对：一条管数值，一条管行为。
        """
        from app.agent import _lc_agent as LC
        assert LC.MAX_MODEL_CALLS_PER_TURN == 15, (
            f"单轮模型调用上限是 {LC.MAX_MODEL_CALLS_PER_TURN}，应当是 15")

    def test_the_tool_list_handed_to_the_model_is_the_built_one(self):
        """
        模型被绑定的工具必须就是 `_build_tools` 的产物、顺序一致 ——
        中间再过一手（过滤、重排）会让上面 A 节的全部断言失去意义。

        **验的是 `bind_tools` 实际收到了什么**，不是源码里有没有那行赋值。
        """
        from app.agent import _lc_agent as LC
        from tests.unit.agent.test_lc_agent_migration import ScriptedModel

        model = ScriptedModel(n_tool_rounds=0)
        agent = LC._build_langchain_agent(model, _Tools(), None)
        agent.invoke({"input": "hi"})

        assert model.bound_tools == EXPECTED_TOOL_ORDER, (
            f"模型拿到的工具是 {model.bound_tools}，"
            f"应当是 {EXPECTED_TOOL_ORDER}")
