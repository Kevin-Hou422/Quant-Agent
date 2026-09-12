"""
agent/_agent.py —— 意图路由、上下文注入、持久化守卫的定钉测试（变异测试驱动）

来由：16 个变异点，首测击杀率 **0.0%** —— 一个都没杀死。

`QuantAgent` 是聊天入口：它决定**这条消息走 Workflow A 还是 B**、
要不要把上一条 DSL 注进上下文、以及每一步要不要落库。
既有覆盖（unit/test_agent_fallback）只验证"没有 LLM key 时能降级返回"，
所以整条路由链全裸。

存活项分三类：
  - `allow_synthetic: bool = False` —— 聊天路径的数据契约。审计修过一次：
    此前**恒用合成数据且无标识**，屏幕上的 Sharpe 在金融上毫无意义。
    默认值改成 True 等于把那个漏洞原样放回去。
  - 六处 `if self._chat_store is not None:` —— 删 `not` 后，
    **有** store 时反而不落库、没有时去调 `None.save_message`
  - 两处 `if intent == "workflow_b" and dsl_hint:`、两处 `if last_dsl and any(...)`
    —— `and` 放宽成 `or` 会让没有 DSL 的消息也走 Workflow B（拿 None 去优化），
    或让任何消息都被注入上下文
  - `if oos_s is None and is_s is not None` —— 回复文案里"样本不足"的提示条件
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest


class _FakeStore:
    """记录所有落库调用，不碰真数据库。"""

    def __init__(self):
        self.calls: List[tuple] = []
        self.history: List[Any] = []

    def ensure_session(self, sid, title=None):
        self.calls.append(("ensure", sid))

    def save_message(self, sid, role, content):
        self.calls.append(("save", sid, role, content))

    def get_history(self, sid, limit=None):
        self.calls.append(("get_history", sid))
        return self.history


class _FakeFallback:
    """打桩的 orchestrator：记录走了哪条 workflow。"""

    def __init__(self):
        self.seen: List[tuple] = []

    def run_workflow_a(self, message):
        self.seen.append(("a", message))
        return "rank(close)", {"is_sharpe": 1.0, "oos_sharpe": 0.8,
                               "overfitting_score": 0.1, "is_overfit": False}

    def run_workflow_b(self, dsl):
        self.seen.append(("b", dsl))
        return dsl, {"is_sharpe": 1.2, "oos_sharpe": 0.9,
                     "overfitting_score": 0.2, "is_overfit": False}


@pytest.fixture
def agent(monkeypatch):
    """构造一个**不连 LLM**的 QuantAgent，并把 orchestrator 换成替身。"""
    from app.agent._agent import QuantAgent
    # `allow_synthetic=True` 是**显式 opt-in**：不给数据集名就构造会被
    # QuantTools 的 fail-closed 契约直接拒绝（这条契约本身由
    # test_lessons_enforced 保证，这里不重复测）。本文件测的是路由与持久化，
    # 全程不碰数据，所以用合成数据源是安全的。
    a = QuantAgent(api_key="", chat_store=None, allow_synthetic=True)
    a._llm = None
    a._chain = None
    a._fallback = _FakeFallback()
    return a


# ===========================================================================
# A. 数据契约：聊天路径默认不得用合成数据
# ===========================================================================

class TestDataContract:

    def test_synthetic_is_off_by_default(self):
        """
        `allow_synthetic: bool = False` —— 这是外部审计修过的那个洞：
        聊天路径此前**恒用合成数据且无标识**，屏幕上的 Sharpe/OOS
        在金融上毫无意义。默认值改回 True 等于把洞原样放回去。
        """
        import inspect
        from app.agent._agent import QuantAgent
        sig = inspect.signature(QuantAgent.__init__)
        assert sig.parameters["allow_synthetic"].default is False, (
            "QuantAgent 的 allow_synthetic 默认值不再是 False —— "
            "聊天路径会默认用合成数据冒充真实回测")

    def test_dataset_window_defaults_are_pinned(self):
        import inspect
        from app.agent._agent import QuantAgent
        p = inspect.signature(QuantAgent.__init__).parameters
        assert p["dataset_name"].default == ""
        assert p["dataset_start"].default == "2021-01-01"
        assert p["dataset_end"].default == "2024-01-01"


# ===========================================================================
# B. 意图路由
# ===========================================================================

class TestIntentRouting:

    def test_an_explicit_dsl_routes_to_workflow_b(self, agent):
        intent, hint = agent._detect_intent("optimize rank(ts_delta(close,5))", "s1")
        assert intent == "workflow_b"
        assert hint and hint.startswith("rank("), hint

    def test_plain_text_routes_to_workflow_a(self, agent):
        intent, hint = agent._detect_intent("find me a momentum factor", "s1")
        assert intent == "workflow_a" and hint is None

    def test_optimise_keyword_reuses_the_last_dsl(self, agent):
        """
        `if last_dsl and any(kw in message.lower() for kw in [...])` ——
        `and` 放宽成 `or` 会让**从没跑过 DSL 的会话**也走 workflow_b，
        然后拿 `None` 去做结构优化。
        """
        agent._session_last_dsl["s1"] = "rank(close)"
        intent, hint = agent._detect_intent("请优化一下", "s1")
        assert (intent, hint) == ("workflow_b", "rank(close)")

    def test_optimise_keyword_without_history_stays_on_workflow_a(self, agent):
        intent, hint = agent._detect_intent("请优化一下", "fresh_session")
        assert intent == "workflow_a" and hint is None, (
            "没有历史 DSL 却路由到了 workflow_b —— `last_dsl and ...` 被放宽")

    def test_unrelated_message_with_history_stays_on_workflow_a(self, agent):
        agent._session_last_dsl["s1"] = "rank(close)"
        intent, _ = agent._detect_intent("今天天气怎么样", "s1")
        assert intent == "workflow_a", "无关键词的消息被路由到了 workflow_b"

    def test_fallback_needs_both_intent_and_hint_for_workflow_b(self, agent):
        """
        `if intent == "workflow_b" and dsl_hint:` —— `and` 放宽成 `or` 会让
        intent 为 workflow_b 但 hint 为 None 时也去 `run_workflow_b(None)`。
        """
        agent._fallback_chat("workflow_b", "msg", None, "s1")
        assert agent._fallback.seen[-1][0] == "a", (
            "intent=workflow_b 但没有 dsl_hint，仍然走了 workflow_b")

        agent._fallback_chat("workflow_b", "msg", "rank(close)", "s1")
        assert agent._fallback.seen[-1] == ("b", "rank(close)")


# ===========================================================================
# C. 上下文注入
# ===========================================================================

class TestContextEnrichment:

    def test_reference_words_inject_the_last_dsl(self, agent):
        agent._session_last_dsl["s1"] = "rank(close)"
        out = agent._enrich_message("把上一个改一下", "s1")
        assert "rank(close)" in out, "引用词没有触发上下文注入"

    def test_messages_without_reference_words_are_untouched(self, agent):
        """
        `if last_dsl and any(kw in message.lower() ...)` —— `and`→`or`
        会让**每一条**消息都被塞进上下文，把无关对话也拖进 DSL 语境。
        """
        agent._session_last_dsl["s1"] = "rank(close)"
        msg = "介绍一下动量因子"
        assert agent._enrich_message(msg, "s1") == msg, (
            "没有引用词的消息被注入了上下文")

    def test_without_history_nothing_is_injected(self, agent):
        msg = "把上一个改一下"
        assert agent._enrich_message(msg, "fresh") == msg

    def test_langchain_path_skips_enrichment(self, agent):
        """
        `if self._chain is not None: return message` —— 删 `not` 会让
        LangChain 路径也做一次注入，而它本来就用 MessagesPlaceholder 注上下文，
        结果是同一段上下文被塞两遍。
        """
        agent._session_last_dsl["s1"] = "rank(close)"
        agent._chain = object()
        msg = "把上一个改一下"
        assert agent._enrich_message(msg, "s1") == msg, (
            "LangChain 路径重复注入了上下文")


# ===========================================================================
# D. 持久化守卫
# ===========================================================================

class TestPersistenceGuards:

    def test_messages_are_persisted_when_a_store_is_present(self, agent):
        """
        六处 `if self._chat_store is not None:` —— 删 `not` 会让**有** store
        时反而不落库：整个会话历史静默丢失，而接口一切正常。
        """
        store = _FakeStore()
        agent._chat_store = store
        agent._fallback_chat("workflow_a", "hello", None, "s1")
        kinds = [c[0] for c in store.calls]
        assert "ensure" in kinds and "save" in kinds, (
            f"有 chat_store 却没有落库：{store.calls}")
        roles = [c[2] for c in store.calls if c[0] == "save"]
        assert "user" in roles and "assistant" in roles, (
            f"用户/助手消息没有都落库：{roles}")

    def test_no_store_means_no_crash(self, agent):
        """没有 store 时不得去调 `None.save_message`。"""
        agent._chat_store = None
        out = agent._fallback_chat("workflow_a", "hello", None, "s1")
        assert isinstance(out, dict) and "reply" in out

    def test_memory_warm_up_reads_history_only_when_a_store_exists(self, agent):
        """
        `if self._chat_store is not None:`（`_get_or_create_memory` 里那一处）
        —— 删 `not` 会让**有** store 时不再回灌历史：刷新页面继续聊，
        agent 完全不记得前面说过什么，而接口一切正常。

        上一版这条只断言了"memory 对象复用"，**根本没检查历史有没有被读进来**
        —— 名字承诺的事一件没测，那个变异点因此活着。
        """
        store = _FakeStore()

        class _Msg:
            def __init__(self, role, content):
                self.role, self.content = role, content

        store.history = [_Msg("user", "hi"), _Msg("assistant", "hello")]
        agent._chat_store = store
        mem = agent._get_or_create_memory("s1")
        assert mem is not None

        # 历史必须真的被读了一次
        assert any(c[0] == "get_history" for c in store.calls), (
            f"有 chat_store 却没有读历史：{store.calls} —— "
            f"回灌逻辑被跳过了")
        # 而且要落进 memory
        text = repr(mem.__dict__)
        assert "hi" in text and "hello" in text, (
            f"历史读到了却没有进 memory：{text[:300]}")

        assert agent._get_or_create_memory("s1") is mem, (
            "同一会话的 memory 每次都新建了 —— `session_id not in ...` 被取反")

    def test_memory_warm_up_is_skipped_without_a_store(self, agent):
        """反向：没有 store 时不得去调 `None.get_history`。"""
        agent._chat_store = None
        mem = agent._get_or_create_memory("s_no_store")
        assert mem is not None

    def test_memory_is_created_once_per_session(self, agent):
        """
        `if session_id not in self._session_memories:` —— 删 `not` 会让
        **已存在**时才新建（即永远拿不到），或每次都覆盖，多轮上下文丢失。
        """
        agent._chat_store = None
        a = agent._get_or_create_memory("s1")
        b = agent._get_or_create_memory("s1")
        c = agent._get_or_create_memory("s2")
        assert a is b and a is not c


# ===========================================================================
# E. 回复文案里的"样本不足"提示
# ===========================================================================

class TestReplyText:

    def test_insufficient_sample_note_needs_both_conditions(self, agent):
        """
        `"⚠ 样本不足…" if oos_s is None and is_s is not None else ""` ——
        `and`→`or` 会让**两个都有**的正常情况也挂上"样本不足"；
        删 `not` 会让有 OOS 时反而提示不足。这句提示是使用者判断
        "屏幕上这个 Sharpe 能不能信"的唯一线索。
        """
        class _FB:
            def run_workflow_a(self, m):
                return "rank(close)", {"is_sharpe": 1.0, "oos_sharpe": None,
                                       "overfitting_score": 0.1}

            def run_workflow_b(self, d):
                return d, {}

        agent._fallback = _FB()
        agent._chat_store = None
        out = agent._fallback_chat("workflow_a", "m", None, "s1")
        assert "样本不足" in out["reply"], (
            f"OOS 缺失却没有提示样本不足：{out['reply']}")

    def test_complete_metrics_carry_no_warning(self, agent):
        agent._chat_store = None
        out = agent._fallback_chat("workflow_a", "m", None, "s1")
        assert "样本不足" not in out["reply"], (
            f"指标齐全却提示了样本不足：{out['reply']}")

    def test_missing_backtest_data_uses_the_short_form(self, agent):
        """`) if is_s is not None else f"Generated DSL: ..."` 的另一条分支。"""
        class _FB:
            def run_workflow_a(self, m):
                return "rank(close)", {}

            def run_workflow_b(self, d):
                return d, {}

        agent._fallback = _FB()
        agent._chat_store = None
        out = agent._fallback_chat("workflow_a", "m", None, "s1")
        assert "insufficient backtest data" in out["reply"], out["reply"]

    def test_nan_metrics_render_as_na(self, agent):
        """`_fmt` 里的 `f != f` 是 NaN 判定：NaN 必须显示 N/A，不能显示 nan。"""
        class _FB:
            def run_workflow_a(self, m):
                return "rank(close)", {"is_sharpe": float("nan"),
                                       "oos_sharpe": 0.5, "overfitting_score": 0.1}

            def run_workflow_b(self, d):
                return d, {}

        agent._fallback = _FB()
        agent._chat_store = None
        reply = agent._fallback_chat("workflow_a", "m", None, "s1")["reply"]
        assert "N/A" in reply and "nan" not in reply.lower(), reply


# ===========================================================================
# F. 构造期的降级链与流式路径
# ===========================================================================

class TestConstructionAndStreaming:

    def test_the_chain_is_built_only_when_there_is_an_llm(self, monkeypatch):
        """
        `if self._llm is not None:`（构造期）—— 改成 `is None` 会在**没有 LLM**
        时去 `_build_langchain_agent(None, ...)`（异常被吞，chain 仍是 None，
        看不出来），而**有** LLM 时根本不建链 —— 整个 LangChain 路径静默失效，
        所有对话永远走 fallback。用户付了 API key 的钱，拿到的是关键词匹配。

        既有用例只测了"没 key → 没链"，那一格 `is None` 与 `is not None` 同样
        给出 chain=None，分不开。必须补**有 LLM** 那一格。
        """
        import langchain_openai
        import app.agent._agent as A

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        built: list = []

        def _fake_build(llm, tools, store):
            built.append(llm)
            return object()

        monkeypatch.setattr(A, "_build_langchain_agent", _fake_build)

        class _Llm:
            """ChatOpenAI 的替身 —— 构造期不联网。"""

            def __init__(self, **kw):
                self.kw = kw

        monkeypatch.setattr(langchain_openai, "ChatOpenAI", _Llm)

        # 没有 key → 没有 LLM → 不建链
        a = A.QuantAgent(api_key="", chat_store=None, allow_synthetic=True)
        assert a._llm is None, "没给 key 却建出了 LLM"
        assert a._chain is None and not built, "没有 LLM 却建了链"

        # 有 key → 有 LLM → **必须**建链
        b = A.QuantAgent(api_key="sk-test-not-a-real-key", chat_store=None,
                         allow_synthetic=True)
        assert isinstance(b._llm, _Llm), "给了 key 却没有建出 LLM"
        assert built == [b._llm], (
            f"有 LLM 时 _build_langchain_agent 被调用了 {len(built)} 次 —— "
            f"`if self._llm is not None:` 的判定反了，LangChain 路径静默失效")
        assert b._chain is not None

    def test_chat_uses_the_chain_when_there_is_one(self, agent, monkeypatch):
        """
        `if self._chain is not None:`（`chat()` 里那一处）—— 改成 `is None`
        会让**有链**时走 fallback、没链时去调 `self._lc_chat`（链是 None，
        里面直接崩，被外层 except 兜住 → 用户看到一句通用错误）。
        两种情况都不报错，症状只是"LLM 白配了"。
        """
        lc_calls, fb_calls = [], []
        monkeypatch.setattr(agent, "_lc_chat",
                            lambda msg, sid: lc_calls.append(msg) or
                            {"reply": "lc", "dsl": None, "metrics": None})
        monkeypatch.setattr(agent, "_fallback_chat",
                            lambda i, m, h, s: fb_calls.append(m) or
                            {"reply": "fb", "dsl": None, "metrics": None})

        agent._chain = None
        agent.chat("hello", session_id="s1")
        assert fb_calls and not lc_calls, (
            "没有链时没走 fallback —— `self._chain is not None` 的判定反了")

        agent._chain = object()
        agent.chat("hello again", session_id="s1")
        assert lc_calls, (
            "有链时没走 LangChain 路径 —— `self._chain is not None` 的判定反了")

    def test_no_api_key_means_no_llm_and_no_chain(self):
        """
        `if self._llm is not None:` —— 删掉 `not` 会让**没有 LLM 时**
        反而去构建 LangChain agent（拿 None 当 llm），构建失败被 except 吞掉，
        表面上还是降级，但每次构造都白跑一趟并打一条误导性的警告。
        """
        from app.agent._agent import QuantAgent
        a = QuantAgent(api_key="", chat_store=None, allow_synthetic=True)
        assert a._llm is None, "没有 API key 却初始化了 LLM"
        assert a._chain is None, (
            "没有 LLM 却构建了 LangChain agent —— `if self._llm is not None` 疑似被取反")
        assert a._fallback is not None, "降级 orchestrator 没有准备好"

    def test_chain_construction_failure_degrades_instead_of_crashing(self, monkeypatch):
        """LangChain 构建失败必须降级，不能让整个 agent 起不来。"""
        import app.agent._agent as mod
        monkeypatch.setattr(
            mod, "_build_langchain_agent",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no langchain")))
        a = mod.QuantAgent(api_key="", chat_store=None, allow_synthetic=True)
        a._llm = object()
        assert a._chain is None

    def test_the_tools_receive_the_data_contract(self, monkeypatch):
        """
        构造 QuantTools 时四个数据契约参数必须原样传下去 ——
        少传一个就会退回 QuantTools 自己的默认值（可能是合成数据）。
        """
        seen = {}
        import app.agent._agent as mod
        real = mod.QuantTools

        def _spy(**kw):
            seen.update(kw)
            return real(**kw)

        monkeypatch.setattr(mod, "QuantTools", _spy)
        mod.QuantAgent(api_key="", chat_store=None,
                       dataset_name="us_tech_large",
                       dataset_start="2022-01-01", dataset_end="2023-01-01",
                       allow_synthetic=False)
        assert seen.get("dataset_name") == "us_tech_large"
        assert seen.get("dataset_start") == "2022-01-01"
        assert seen.get("dataset_end") == "2023-01-01"
        assert seen.get("allow_synthetic") is False, (
            "allow_synthetic 没有传给 QuantTools —— 数据契约在这一层断了")

    def test_stream_routes_to_workflow_b_only_with_a_hint(self, agent, monkeypatch):
        """
        流式路径里的 `if intent == "workflow_b" and dsl_hint:` 是**另一份拷贝**
        （§S：同一个判断的两份实现）。`and`→`or` 会让它拿 None 去跑
        OptimizationWorkflow。这里把两个 Workflow 都打桩，记录走了哪条。
        """
        import app.core.workflows.alpha_workflows as wf_mod
        seen = []

        class _Res:
            best_dsl = "rank(close)"
            explanation = "ok"
            metrics = {}

        class _Gen:
            def __init__(self, **kw):
                pass

            def run(self, message, dataset, on_progress=None):
                seen.append("a")
                return _Res()

        class _Opt:
            def __init__(self, **kw):
                pass

            def run(self, dsl, dataset, on_progress=None):
                seen.append("b")
                return _Res()

        monkeypatch.setattr(wf_mod, "GenerationWorkflow", _Gen)
        monkeypatch.setattr(wf_mod, "OptimizationWorkflow", _Opt)
        monkeypatch.setattr(agent._tools, "dataset", {}, raising=False)

        agent.stream_chat("hello", session_id="s1", on_event=lambda e: None)
        assert seen and seen[-1] == "a", (
            f"纯文本消息在流式路径里走了 {seen[-1]} —— 应当是 Workflow A")

        agent.stream_chat("optimize rank(ts_delta(close,5))",
                          session_id="s1", on_event=lambda e: None)
        assert seen[-1] == "b", "带 DSL 的消息没有走 Workflow B"

        # 上面两次调用**分不开** `and` 与 `or`：
        #   (workflow_a, None)      → and 假、or 假
        #   (workflow_b, "rank...") → and 真、or 真
        # 真正能分开的是 **intent 为 workflow_b 但 hint 为空** 那一格：
        #   and → Workflow A（正确，没有 DSL 可优化）
        #   or  → Workflow B，拿 None 去跑结构优化
        # `_detect_intent` 产不出这一格，所以直接把它打桩造出来。
        monkeypatch.setattr(agent, "_detect_intent",
                            lambda msg, sid: ("workflow_b", None))
        agent.stream_chat("whatever", session_id="s1", on_event=lambda e: None)
        assert seen[-1] == "a", (
            f"intent=workflow_b 但 dsl_hint 为空时走了 {seen[-1]} —— "
            f"流式路径里的 `intent == 'workflow_b' and dsl_hint` 被放宽成了 `or`，"
            f"会拿 None 去跑 OptimizationWorkflow")

    def test_stream_persists_both_messages(self, agent, monkeypatch):
        """
        流式路径末尾那处 `if self._chat_store is not None:` —— 删 `not`
        会让流式聊天的记录**整段不落库**，而非流式路径的落库正常，
        症状是"用网页聊的没记录、用接口调的有记录"。
        """
        import app.core.workflows.alpha_workflows as wf_mod

        class _Res:
            best_dsl = "rank(close)"
            explanation = "done"
            metrics = {}

        class _Gen:
            def __init__(self, **kw):
                pass

            def run(self, message, dataset, on_progress=None):
                return _Res()

        monkeypatch.setattr(wf_mod, "GenerationWorkflow", _Gen)
        monkeypatch.setattr(agent._tools, "dataset", {}, raising=False)
        store = _FakeStore()
        agent._chat_store = store
        agent.stream_chat("hello", session_id="s1", on_event=lambda e: None)
        roles = [c[2] for c in store.calls if c[0] == "save"]
        assert "user" in roles and "assistant" in roles, (
            f"流式路径没有把两条消息都落库：{store.calls}")
