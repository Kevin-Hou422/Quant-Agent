"""
_lc_agent.py 迁移到 `langchain.agents.create_agent` 的验收测试（缺陷 D-3）。

**这些用例一律使用实际安装的 LangChain 真跑一遍**，不 mock agent 构建器 ——
"把构建器 mock 掉再宣称兼容"恰恰是这条缺陷能潜伏这么久的原因：
`requirements.txt` 的 `langchain>=0.2` 没有上界、lock 锁着 1.4.0，
而旧代码要的 `AgentExecutor` 在 1.x 已经搬进未安装的 `langchain-classic`，
于是构建**必然失败**、被 `except Exception` 接住、只打一条 warning 就静默降级。
任何一个 mock 掉构建过程的用例都看不见这件事。

模型替身是**确定性**的（按脚本发工具调用），不连任何外部服务：
所以这里验证的是"离线工具调用链路通"，**不是**"真实 LLM 服务可用"。
对真实模型服务的连通性，本套件**未验证**。
"""
from __future__ import annotations

from typing import Any, List, Optional

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent import _lc_agent as LC
from app.db.chat_store import ChatStore


# ===========================================================================
# 确定性模型替身
# ===========================================================================

class ScriptedModel(BaseChatModel):
    """
    按脚本行事的模型替身：前 `n_tool_rounds` 轮各发一次工具调用，之后给最终答复。

    刻意**不**用 MagicMock —— 它必须真的走 create_agent 的图、真的被
    `bind_tools` 调用、真的让 ToolNode 执行工具，否则"跑通了"证明不了什么。
    """

    n_tool_rounds: int = 1
    tool_name: str = "tool_generate_alpha_dsl"
    tool_args: dict = {"hypothesis": "momentum"}
    calls: int = 0
    seen_tool_messages: list = []
    #: 每次 `_generate` 收到的全部消息（原样）—— 用来验系统提示词真的到了模型手里，
    #: 而不是去读 `_build_langchain_agent` 的源码文本找 `system_prompt=`。
    seen_messages: list = []
    #: `bind_tools` 拿到的工具名 —— 验"create_agent 收到的就是 _build_tools 的产物"。
    bound_tools: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted-for-tests"

    def _generate(self, messages: List[BaseMessage],
                  stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any) -> ChatResult:
        i = self.calls
        object.__setattr__(self, "calls", i + 1)
        object.__setattr__(self, "seen_tool_messages",
                           [str(m.content) for m in messages
                            if isinstance(m, ToolMessage)])
        object.__setattr__(self, "seen_messages", list(messages))
        if i < self.n_tool_rounds:
            msg = AIMessage(content="", tool_calls=[{
                "name": self.tool_name, "args": dict(self.tool_args),
                "id": f"call_{i}", "type": "tool_call"}])
        else:
            hist = [m.content for m in messages if isinstance(m, HumanMessage)]
            msg = AIMessage(content=f"FINAL|calls={i}|humans={len(hist)}")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):        # noqa: D102
        object.__setattr__(self, "bound_tools",
                           [getattr(t, "name", str(t)) for t in tools])
        return self


@pytest.fixture
def tools_obj():
    """真实的 QuantTools（合成数据，不联网）。"""
    from app.agent._tools import QuantTools
    return QuantTools(n_tickers=3, n_days=60, allow_synthetic=True)


@pytest.fixture
def store(tmp_path):
    return ChatStore(db_url=f"sqlite:///{tmp_path / 'chat.db'}")


# ===========================================================================
# 1. 用实际安装的 LangChain 真的建得出来
# ===========================================================================

class TestTheAgentActuallyBuilds:

    def test_the_installed_langchain_provides_the_new_agent_api(self):
        """
        D-3 的核心断言：**当前已安装的依赖**必须能提供 `create_agent` 与两个中间件。

        这条不 mock 任何东西 —— 它就是要回答"按 requirements 装出来的环境，
        这条链路建得起来吗"。旧代码在这里答的是"不能"，而没有任何人看见。
        """
        import langchain
        from langchain.agents import create_agent
        from langchain.agents.middleware import (
            ModelCallLimitMiddleware, wrap_tool_call)

        assert callable(create_agent), "create_agent 不可调用"
        assert callable(wrap_tool_call)
        assert ModelCallLimitMiddleware is not None
        major = int(str(langchain.__version__).split(".")[0])
        assert major >= 1, (
            f"装的是 langchain {langchain.__version__} —— 本模块要求 1.x 的 "
            f"create_agent API，0.x 提供不了")

    def test_build_returns_an_agent_not_a_fallback(self, tools_obj, store):
        agent = LC._build_langchain_agent(ScriptedModel(), tools_obj, store)
        assert agent is not None
        assert hasattr(agent, "invoke")

    def test_the_incompatibility_error_names_the_version_not_a_missing_package(self):
        """
        报错文案必须指向**大版本不兼容**。

        原文案是"需要安装 langchain 和 langchain-openai"，而 langchain 明明装着 ——
        照着文案做只会再装一遍同样的版本，真正的原因一次都不会被看见。
        用独立异常类型而不是字符串匹配来区分，文案改了也不会失效。
        """
        assert issubclass(LC.LangChainIncompatibleError, ImportError)

        # 真的把 import 打断，看**实际抛出来**的消息 —— 而不是读源码找字符串。
        import builtins
        real_import = builtins.__import__

        def _fail_on_agents(name, *a, **k):
            if name == "langchain.agents" or name.startswith("langchain.agents."):
                raise ImportError("cannot import name 'create_agent'")
            return real_import(name, *a, **k)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(builtins, "__import__", _fail_on_agents)
            with pytest.raises(LC.LangChainIncompatibleError) as ei:
                LC._build_langchain_agent(ScriptedModel(), object(), None)

        msg = str(ei.value)
        assert "pip install langchain" not in msg, (
            f"报错文案又把人引向『去装 langchain』了 —— langchain 是装着的，"
            f"照做只会再装一遍同样的版本。实际文案：{msg}")
        assert "不兼容" in msg and "1.0" in msg, (
            f"报错文案没有指出这是大版本不兼容、也没说需要哪个版本：{msg}")


# ===========================================================================
# 2. 模型发起工具调用 → 工具执行 → 最终回复
# ===========================================================================

class TestToolCallRoundTrip:

    def test_a_tool_call_actually_executes_and_feeds_back(self, tools_obj, store):
        """
        全链路：模型发 tool_call → ToolNode 真的执行了 QuantTools 的工具 →
        结果作为 ToolMessage 回到模型 → 模型给出最终回复。
        """
        model = ScriptedModel(n_tool_rounds=1)
        agent = LC._build_langchain_agent(model, tools_obj, store)
        sid = store.create_session("t").id

        out = agent.invoke({"input": "找个动量因子"},
                           config={"configurable": {"session_id": sid}})

        assert out["output"].startswith("FINAL"), out
        assert model.seen_tool_messages, (
            "模型第二轮没有看到任何 ToolMessage —— 工具根本没被执行，"
            "这条链路只是『没报错』，不是『跑通了』")

    def test_zero_tool_calls_still_produces_a_reply(self, tools_obj, store):
        """模型直接答、不调工具，也必须拿得到回复。"""
        agent = LC._build_langchain_agent(ScriptedModel(n_tool_rounds=0),
                                          tools_obj, store)
        sid = store.create_session("t").id
        out = agent.invoke({"input": "hi"},
                           config={"configurable": {"session_id": sid}})
        assert out["output"].startswith("FINAL")


# ===========================================================================
# 3. 会话历史：两轮、隔离、持久化与恢复，且不重复追加
# ===========================================================================

class TestSessionHistory:

    def test_two_turns_accumulate_exactly_two_messages_each(self, tools_obj, store):
        """
        每轮**恰好**落库 2 条（user + assistant）。

        这是迁移最容易错的地方：`create_agent` 返回的 `messages` 里**含**传进去的
        历史，整段回写就会把旧消息再写一遍，聊到第 N 轮历史指数膨胀。
        `RunnableWithMessageHistory` 当初只写新增的两条，这里必须一致。
        """
        agent = LC._build_langchain_agent(ScriptedModel(n_tool_rounds=0),
                                          tools_obj, store)
        sid = store.create_session("t").id
        cfg = {"configurable": {"session_id": sid}}

        agent.invoke({"input": "第一问"}, config=cfg)
        assert len(store.get_history(sid)) == 2

        agent.invoke({"input": "第二问"}, config=cfg)
        rows = store.get_history(sid)
        assert len(rows) == 4, (
            f"两轮之后应当是 4 条，实际 {len(rows)} 条："
            f"{[(r.role, r.content[:20]) for r in rows]}")
        assert [r.role for r in rows] == ["user", "assistant", "user", "assistant"]
        assert rows[0].content == "第一问" and rows[2].content == "第二问"

    def test_the_second_turn_sees_the_first(self, tools_obj, store):
        """历史真的被读回来喂给模型 —— 不是只存不用。"""
        model = ScriptedModel(n_tool_rounds=0)
        agent = LC._build_langchain_agent(model, tools_obj, store)
        sid = store.create_session("t").id
        cfg = {"configurable": {"session_id": sid}}

        agent.invoke({"input": "第一问"}, config=cfg)
        out = agent.invoke({"input": "第二问"}, config=cfg)
        # 替身把看到的 HumanMessage 条数写进回复
        assert out["output"].endswith("humans=2"), (
            f"第二轮模型只看到 {out['output']} —— 历史没有被读回来")

    def test_sessions_are_isolated(self, tools_obj, store):
        """A 会话读不到 B 会话的任何记录。"""
        agent = LC._build_langchain_agent(ScriptedModel(n_tool_rounds=0),
                                          tools_obj, store)
        a = store.create_session("A").id
        b = store.create_session("B").id

        agent.invoke({"input": "只属于A"},
                     config={"configurable": {"session_id": a}})
        out_b = agent.invoke({"input": "只属于B"},
                             config={"configurable": {"session_id": b}})

        assert out_b["output"].endswith("humans=1"), (
            "B 会话第一轮就看到了不止一条人类消息 —— 会话串了")
        assert all("只属于B" not in r.content for r in store.get_history(a))
        assert all("只属于A" not in r.content for r in store.get_history(b))

    def test_history_survives_a_new_agent_instance(self, tools_obj, store):
        """重建 agent（模拟进程重启）后历史仍在 —— 持久化不是进程内缓存。"""
        sid = store.create_session("t").id
        cfg = {"configurable": {"session_id": sid}}

        a1 = LC._build_langchain_agent(ScriptedModel(n_tool_rounds=0), tools_obj, store)
        a1.invoke({"input": "重启前"}, config=cfg)

        a2 = LC._build_langchain_agent(ScriptedModel(n_tool_rounds=0), tools_obj, store)
        out = a2.invoke({"input": "重启后"}, config=cfg)

        assert out["output"].endswith("humans=2")
        assert len(store.get_history(sid)) == 4


# ===========================================================================
# 4. 旧调用契约不变
# ===========================================================================

class TestTheOldContractIsPreserved:

    def test_invoke_takes_input_and_returns_output(self, tools_obj, store):
        """
        上游 `_agent._lc_chat` 读的是 `resp.get("output")`，
        `/api/chat` 与 `/api/chat/stream` 又都建在它上面。
        迁移不得改这个形状，否则"修好了 LLM 链路"会顺手打断 API。
        """
        agent = LC._build_langchain_agent(ScriptedModel(n_tool_rounds=0),
                                          tools_obj, store)
        sid = store.create_session("t").id
        out = agent.invoke({"input": "x"},
                           config={"configurable": {"session_id": sid}})
        assert isinstance(out, dict) and isinstance(out["output"], str)

    def test_history_detection_attribute_survives(self, tools_obj, store):
        """`_agent._lc_chat` 用 `hasattr(chain, "get_session_history")` 分支。"""
        with_store = LC._build_langchain_agent(ScriptedModel(), tools_obj, store)
        assert hasattr(with_store, "get_session_history")
        assert callable(with_store.get_session_history)

    def test_it_works_without_a_chat_store(self, tools_obj):
        """无 DB 时上游自带历史（`_lc_chat` 的 else 分支），不得崩。"""
        agent = LC._build_langchain_agent(ScriptedModel(n_tool_rounds=0),
                                          tools_obj, None)
        # 没有 chat_store 时不得假装有 DB 历史 —— `_agent._lc_chat` 正是靠这个
        # 属性选分支，谎报会让它走进"有历史"那条路然后拿不到 session_id。
        with pytest.raises(AttributeError):
            _ = agent.get_session_history

        out = agent.invoke({"input": "x", "chat_history": [
            HumanMessage(content="旧问"), AIMessage(content="旧答")]})
        assert out["output"].endswith("humans=2"), (
            "上游自带的 chat_history 没有被喂给模型")


# ===========================================================================
# 5. 工具异常与执行次数限制 —— 语义逐条核对，不照搬数字
# ===========================================================================

class TestFailureModesAndLimits:

    def test_a_raising_tool_becomes_an_observation_not_a_crash(self, tools_obj, store):
        """
        旧 `AgentExecutor(handle_parsing_errors=True)` 把工具异常转成观测；
        `create_agent` **默认让异常冒出 invoke**（实测），一个工具出错就把整轮
        对话打断 —— 而这些工具里跑的是 GP 进化、回测、Optuna，失败是常态。
        `wrap_tool_call` 中间件把这个语义补了回来。
        """
        model = ScriptedModel(n_tool_rounds=1)
        agent = LC._build_langchain_agent(model, tools_obj, store)
        sid = store.create_session("t").id

        def _boom(*a, **k):
            raise RuntimeError("GP 炸了")
        # 真实工具对象上打桩，让 ToolNode 真的执行到抛异常的那一步
        object.__setattr__(tools_obj, "tool_generate_alpha_dsl", _boom)

        out = agent.invoke({"input": "go"},
                           config={"configurable": {"session_id": sid}})

        assert out["output"].startswith("FINAL"), "工具异常把整轮对话打断了"
        assert any(LC._TOOL_ERROR_PREFIX in m for m in model.seen_tool_messages), (
            f"模型没看到工具失败的观测：{model.seen_tool_messages}")

    def test_the_model_call_limit_ends_gracefully(self, tools_obj, store, monkeypatch):
        """
        `max_iterations=15` 的真正对应物是 `ModelCallLimitMiddleware(run_limit=...)`，
        **不是** `recursion_limit`：前者数模型调用、到顶优雅收尾；
        后者数 LangGraph super-step、到顶抛 `GraphRecursionError`。

        这里把上限压到 2 再让模型无限要工具，断言"停下来返回"而不是抛异常。
        """
        monkeypatch.setattr(LC, "MAX_MODEL_CALLS_PER_TURN", 2)
        monkeypatch.setattr(LC, "RECURSION_LIMIT", 4 * 2 + 4)
        agent = LC._build_langchain_agent(ScriptedModel(n_tool_rounds=99),
                                          tools_obj, store)
        sid = store.create_session("t").id

        out = agent.invoke({"input": "go"},
                           config={"configurable": {"session_id": sid}})
        assert isinstance(out["output"], str), "到达上限时抛异常了，没有优雅收尾"

    def test_the_call_budget_resets_every_turn(self, tools_obj, store, monkeypatch):
        """
        必须是 `run_limit`（**单次调用**计数）而不是 `thread_limit`（跨轮累积）。

        这条**行为地**验它：把预算压到 2，连发三轮。每轮都该拿到完整预算 ——
        若用的是 thread_limit，第二轮开始就没有额度了，回复会退化成
        "Model call limits exceeded"。症状是"同一会话越聊越笨"，极难归因，
        所以不能只断言源码里没有 thread_limit 这个词。
        """
        monkeypatch.setattr(LC, "MAX_MODEL_CALLS_PER_TURN", 2)
        monkeypatch.setattr(LC, "RECURSION_LIMIT", 4 * 2 + 4)
        sid = store.create_session("t").id
        cfg = {"configurable": {"session_id": sid}}
        agent = LC._build_langchain_agent(ScriptedModel(n_tool_rounds=0),
                                          tools_obj, store)

        outs = [agent.invoke({"input": f"第{i}问"}, config=cfg)["output"]
                for i in range(1, 4)]
        for i, out in enumerate(outs, 1):
            assert out.startswith("FINAL"), (
                f"第 {i} 轮拿到的是 {out!r} —— 预算没有按轮重置，"
                f"用的多半是 thread_limit")

    def test_the_guard_is_loose_enough_for_the_limit_middleware_to_finish_first(self):
        """
        护栏必须**宽于**中间件的收口路径，否则先抛 GraphRecursionError ——
        那正是迁移要消掉的硬失败。

        实测（中间件在图里时，L = MAX_MODEL_CALLS_PER_TURN）：
          · 被 run_limit 截停     需 4L + 2
          · 模型自己在第 L 轮讲完  需 4L + 4
        取上界 4L + 4。

        **第一版写的是 2L+2** —— 那是在**没装中间件**的图上量的，
        中间件给每轮加了一个节点、开销翻倍，于是护栏比收口更早触发。
        上面那条 test_the_model_call_limit_ends_gracefully 当场把它抓了出来。
        判据与被测对象必须是同一个东西。
        """
        L = LC.MAX_MODEL_CALLS_PER_TURN
        assert LC.RECURSION_LIMIT == 4 * L + 4
        assert LC.RECURSION_LIMIT > 4 * L + 2, (
            "护栏没有严格宽于『被 run_limit 截停』所需的 4L+2")


# ===========================================================================
# 6. 降级原因必须可检测 —— 四种情形分得开
# ===========================================================================

class TestDegradationIsObservable:
    """
    【缺陷 D-3 的另一半】此前四种完全不同的情形都只表现为一条 warning +
    "/api/chat 照常返回"：没配 key（设计内）、依赖不兼容（部署缺陷）、
    LLM 初始化失败、agent 构建失败。于是 langchain 大版本搬家把整条 LLM
    研究链路打死之后，系统从外面看仍然"正常工作"。
    """

    @staticmethod
    def _agent(monkeypatch, **kw):
        from app.agent._agent import QuantAgent
        monkeypatch.setattr("app.agent._agent.QuantTools", _StubTools, raising=True)
        return QuantAgent(allow_synthetic=True, **kw)

    def test_no_api_key_is_reported_as_a_designed_fallback(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from app.agent import _agent as A
        ag = self._agent(monkeypatch, api_key="")
        assert ag.agent_mode == A.AGENT_MODE_NO_API_KEY
        assert ag.agent_degraded is True

    def test_incompatible_langchain_is_reported_as_a_deployment_defect(self, monkeypatch):
        """
        **这是 D-3 本身的形态**：依赖不兼容必须与"没配 key"分开上报，
        否则部署缺陷会被当成正常降级一直留着。
        """
        from app.agent import _agent as A

        def _boom(*a, **k):
            raise A.LangChainIncompatibleError("大版本搬家了")

        monkeypatch.setattr(A, "_build_langchain_agent", _boom)
        monkeypatch.setattr(A, "ChatOpenAI", None, raising=False)
        ag = self._agent(monkeypatch, api_key="sk-test")
        assert ag.agent_mode == A.AGENT_MODE_INCOMPATIBLE_DEPS, (
            f"依赖不兼容被归成了 {ag.agent_mode} —— 与设计内降级混在一起了")
        assert ag.agent_degraded is True

    def test_other_build_failures_get_their_own_mode(self, monkeypatch):
        from app.agent import _agent as A
        monkeypatch.setattr(A, "_build_langchain_agent",
                            lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
        ag = self._agent(monkeypatch, api_key="sk-test")
        assert ag.agent_mode == A.AGENT_MODE_AGENT_BUILD_FAILED

    def test_the_four_modes_are_distinct(self):
        """四个降级原因不能有重名 —— 重名就等于分不开。"""
        from app.agent import _agent as A
        modes = {A.AGENT_MODE_NO_API_KEY, A.AGENT_MODE_INCOMPATIBLE_DEPS,
                 A.AGENT_MODE_LLM_INIT_FAILED, A.AGENT_MODE_AGENT_BUILD_FAILED}
        assert len(modes) == 4
        assert A.AGENT_MODE_LLM not in A.AGENT_MODES_DEGRADED
        assert modes == set(A.AGENT_MODES_DEGRADED)

    def test_agent_mode_reaches_both_the_rest_and_the_stream_contract(self, monkeypatch):
        """
        两条路径都要带上。只改 POST /chat 等于只修一半 ——
        前端用的是 /chat/stream（DEV_LESSONS §R，data_source 当初就踩过）。

        流式那半**真的跑一遍** `stream_chat` 并接住 done 事件，
        而不是去源码里找 "agent_mode" 这个词：找得到不等于真的发出去了。
        """
        from app.api.chat_router import ChatResponse

        assert "agent_mode" in ChatResponse.model_fields, "REST 响应没带 agent_mode"

        # 用真的 QuantAgent（合成数据）跑完整条流式路径 —— 与
        # test_lessons_enforced.py::test_stream_done_event_carries_data_source
        # 同一个搭法：那条验的是 data_source，这条验 agent_mode，同一个道理。
        from app.agent.quant_agent import QuantAgent

        ag = QuantAgent(n_tickers=20, n_days=252, n_trials=1,
                        dataset_name="", allow_synthetic=True, api_key="")
        events: list = []
        ag.stream_chat("rank(close)", session_id="d3_stream",
                       on_event=events.append)

        done = [e for e in events if e.get("type") == "done"]
        assert done, (
            f"stream_chat 没有发出 done 事件："
            f"{[e.get('type') for e in events]}")
        payload = done[-1]["result"]
        assert "agent_mode" in payload, (
            f"流式 done 事件没带 agent_mode（只带了 {sorted(payload)}）—— "
            f"前端用的正是这条路，只改 POST /chat 等于只修一半")
        assert payload["agent_mode"] == ag.agent_mode


class _StubTools:
    """极轻的 QuantTools 替身 —— 这几条只关心模式分类，不需要真数据集。"""

    def __init__(self, *a, **kw):
        self._full_dataset = {}
        self._oos_ratio = 0.3
        self._seed = 0
        self.data_source = "synthetic"


# ===========================================================================
# 7. 护栏常数在**实际安装的版本**上自校验
# ===========================================================================

class TestTheGuardIsMeasuredNotAssumed:
    """
    `RECURSION_LIMIT` 的值不能靠"我在别的环境上量过"。

    这一点差点出事：公式最初是在系统 python 的 langchain **1.2.15** 上量的，
    而套件实际跑在 civenv 的 **1.4.0** 上 —— 两个不同的解释器、不同的版本。
    图的结构（每轮几个 super-step）完全可能随版本变，那样常数就悄悄失效了：
    症状是长对话偶发 GraphRecursionError，且只在某些版本上出现。

    所以这里**当场测**：用确定性替身把最小可行 recursion_limit 二分出来，
    再断言我们配的值不小于它。换个 langchain 版本，这条会自己重新校准。
    """

    @staticmethod
    def _min_workable_limit(tools_obj, run_limit: int, n_rounds: int,
                            hi: int = 120) -> Optional[int]:
        from langchain.agents import create_agent
        from langchain.agents.middleware import ModelCallLimitMiddleware
        from langchain.tools import tool as lc_tool

        @lc_tool
        def noop(x: str) -> str:
            """Do nothing; only forces a tool-call round."""
            return "ok"

        for lim in range(1, hi + 1):
            agent = create_agent(
                ScriptedModel(n_tool_rounds=n_rounds, tool_name="noop",
                              tool_args={"x": "1"}),
                [noop],
                middleware=[ModelCallLimitMiddleware(
                    run_limit=run_limit, exit_behavior="end")],
            )
            try:
                agent.invoke({"messages": [HumanMessage(content="go")]},
                             config={"recursion_limit": lim})
                return lim
            except Exception:
                continue
        return None

    def test_the_configured_guard_covers_both_termination_paths(self, tools_obj):
        """
        两条收口路径都要盖住：
          A) 模型一直要工具、被 run_limit 截停
          B) 模型自己在第 L 轮讲完
        护栏若小于其中任何一条，就会先抛 GraphRecursionError。

        用小 L 实测（大 L 的搜索太慢），再按实测斜率外推到生产值并断言。
        """
        import langchain

        small = 2
        cut_off = self._min_workable_limit(tools_obj, run_limit=small, n_rounds=99)
        finishes = self._min_workable_limit(tools_obj, run_limit=99, n_rounds=small)
        assert cut_off and finishes, (
            f"没量出最小 recursion_limit（langchain {langchain.__version__}）—— "
            f"替身或 API 形状变了，本条失去意义")

        # 每轮的 super-step 开销（实测斜率），再按生产 L 外推
        per_round = (finishes - 4) / small if finishes > 4 else 4
        need = max(cut_off, finishes)
        scaled = int(per_round * LC.MAX_MODEL_CALLS_PER_TURN + 4)

        assert LC.RECURSION_LIMIT >= scaled, (
            f"langchain {langchain.__version__} 上，L={LC.MAX_MODEL_CALLS_PER_TURN} "
            f"需要 recursion_limit ≈ {scaled}（L={small} 实测：截停 {cut_off}、"
            f"自然收口 {finishes}），而配置的是 {LC.RECURSION_LIMIT} —— "
            f"护栏会比 ModelCallLimitMiddleware 先触发，长对话将偶发 "
            f"GraphRecursionError。图的 super-step 结构可能随版本变过。")
        assert need <= LC.RECURSION_LIMIT


# ===========================================================================
# 8. 适配器的取值条件 —— 三个 `and` 都不能放宽
# ===========================================================================

class TestTheReplyExtractionIsStrict:
    """
    `_SessionScopedAgent` 里三处 `and` 条件，变异复核（2026-09-20）显示放宽成
    `or` 后全部存活 —— 不是等价变异，是当时没有用例走到那些形状。
    这三条按它们**真实的失败后果**补上。
    """

    class _EmptyFinalModel(ScriptedModel):
        """最后一条 AIMessage 内容为空 —— 模型调完工具却没给总结。"""

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.outputs import ChatGeneration, ChatResult
            i = self.calls
            object.__setattr__(self, "calls", i + 1)
            # 父类的 _generate 被整体覆盖了，记录也要跟着搬过来 ——
            # 否则前置断言"工具真的执行过"永远看到空列表。
            object.__setattr__(self, "seen_tool_messages",
                               [str(m.content) for m in messages
                                if isinstance(m, ToolMessage)])
            if i == 0:
                msg = AIMessage(content="", tool_calls=[{
                    "name": "tool_generate_alpha_dsl",
                    "args": {"hypothesis": "m"}, "id": "c0", "type": "tool_call"}])
            else:
                msg = AIMessage(content="")          # 空总结
            return ChatResult(generations=[ChatGeneration(message=msg)])

    def test_the_reply_is_never_raw_tool_output(self, tools_obj, store):
        """
        回复只能取 **AIMessage** 的文本，绝不能退而取 ToolMessage。

        把 `isinstance(msg, AIMessage) and ...` 放宽成 `or`，倒着找时就会先撞上
        ToolMessage —— 于是用户看到的"回复"是工具的原始输出
        （GP 的 JSON、回测指标串），而不是模型的话。
        模型调完工具却给了空总结时就会走到这条路。
        """
        model = self._EmptyFinalModel(n_tool_rounds=1)
        agent = LC._build_langchain_agent(model, tools_obj, store)
        sid = store.create_session("t").id

        out = agent.invoke({"input": "go"},
                           config={"configurable": {"session_id": sid}})

        assert model.seen_tool_messages, "构造前提：工具必须真的执行过"
        tool_text = model.seen_tool_messages[-1]
        assert out["output"] != tool_text, (
            f"回复取到了工具的原始输出：{out['output'][:80]!r}")
        assert out["output"] == "", (
            f"模型没给总结时回复应当是空串，实际 {out['output'][:80]!r}")

    def test_non_string_content_is_not_returned_as_the_reply(self, tools_obj, store):
        """
        `isinstance(msg.content, str)` 不能放宽：LangChain 的 content 可以是
        **多模态块列表**。放行之后 `{"output": [...]}` 会一路流到
        `ChatResponse.reply`（声明类型是 str），前端拿到的不是字符串。
        """
        class _ListContentModel(ScriptedModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                from langchain_core.outputs import ChatGeneration, ChatResult
                i = self.calls
                object.__setattr__(self, "calls", i + 1)
                if i == 0:
                    return ChatResult(generations=[ChatGeneration(
                        message=AIMessage(content=[{"type": "text", "text": "块"}]))])
                return ChatResult(generations=[ChatGeneration(
                    message=AIMessage(content="真正的回复"))])

        agent = LC._build_langchain_agent(_ListContentModel(n_tool_rounds=0),
                                          tools_obj, store)
        sid = store.create_session("t").id
        out = agent.invoke({"input": "go"},
                           config={"configurable": {"session_id": sid}})
        assert isinstance(out["output"], str), (
            f"回复不是字符串而是 {type(out['output']).__name__} —— "
            f"ChatResponse.reply 声明的是 str")

    def test_a_missing_session_id_does_not_touch_the_store(self, tools_obj, store):
        """
        `self._history_factory is not None and session_id` —— **两个都要**。

        放宽成 `or` 之后，带着 chat_store 但**不带 session_id** 调用时会去执行
        `self._history_factory(None)`，拿 None 当会话 id 去 `ensure_session`。
        这正是 `_agent._lc_chat` 的 else 分支形状（上游自带历史、不给 session_id）。
        正确行为：退回用 payload 里的 chat_history，一条都不写库。
        """
        before = len(store.list_sessions(limit=1000))
        agent = LC._build_langchain_agent(ScriptedModel(n_tool_rounds=0),
                                          tools_obj, store)

        out = agent.invoke({"input": "x", "chat_history": [
            HumanMessage(content="旧问"), AIMessage(content="旧答")]})

        assert out["output"].endswith("humans=2"), (
            "没有 session_id 时没有回退到 payload 里的 chat_history")
        assert len(store.list_sessions(limit=1000)) == before, (
            "缺 session_id 却往库里建了会话 —— `and` 被放宽成了 `or`")
