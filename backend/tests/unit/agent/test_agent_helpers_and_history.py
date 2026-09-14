"""
agent/_helpers.py + agent/_chat_history.py —— 两个小但要害的 agent 辅助模块

**此前均零测试**（34 + 74 有效行，D 档）。

`_helpers._extract_balanced` 是**从 LLM 自由文本里抠出 DSL 的唯一入口**。
LLM 回的是 "好的，我建议试试 rank(ts_delta(log(close), 5)) 这个因子"，
括号匹配错一位，抠出来的就是半截表达式 → 解析失败 → 整轮建议作废，
而日志里只会看到一条 ParseError，看不出是抠错了。

`_chat_history` 的模块 docstring 明写：
"All reads/writes are scoped by WHERE session_id = :sid —
 session A cannot read any records belonging to session B."
这是**会话隔离**的承诺。它一旦失效，A 用户的对话会出现在 B 用户的上下文里，
而且不会有任何报错。这条必须有测试守着。
"""
from __future__ import annotations

import pytest

from app.agent._helpers import _extract_balanced, _safe_json_loads


# ===========================================================================
# A. 从自由文本里抠平衡括号表达式
# ===========================================================================

class TestExtractBalanced:

    def test_a_simple_call_is_extracted_whole(self):
        text = "试试 rank(close) 这个"
        assert _extract_balanced(text, text.index("rank")) == "rank(close)"

    def test_nested_parens_are_matched_to_the_outermost_close(self):
        """
        `depth` 计数是这个函数的全部。少加一层或提前归零，
        抠出来的会是 `rank(ts_delta(log(close)` 这种半截表达式。
        """
        text = "建议 rank(ts_delta(log(close), 5)) 看看"
        got = _extract_balanced(text, text.index("rank"))
        assert got == "rank(ts_delta(log(close), 5))", (
            f"嵌套括号没有匹配到最外层：{got!r}")

    def test_trailing_text_after_the_expression_is_not_included(self):
        """
        `return text[start : i + 1]` —— `i + 1` 的 `+1` 少了会丢掉最后一个 `)`，
        多了会把后面的字符吃进来。
        """
        text = "rank(close) 然后加平滑"
        assert _extract_balanced(text, 0) == "rank(close)"

    def test_text_before_the_first_paren_is_kept(self):
        """
        切片从 `start` 开始而不是从 `paren_idx` —— 函数名必须留着。
        `text[paren_idx : i+1]` 会只返回 `(close)`，丢掉 `rank`。
        """
        text = "xx rank(close)"
        got = _extract_balanced(text, text.index("rank"))
        assert got.startswith("rank("), (
            f"抠出来的表达式丢了函数名：{got!r}")

    def test_a_leading_minus_is_preserved(self):
        """docstring 承诺"Handles optional leading '-'"。"""
        text = "-ts_delta(close, 5)"
        assert _extract_balanced(text, 0) == "-ts_delta(close, 5)"

    def test_no_parenthesis_returns_none(self):
        """
        `if paren_idx == -1: return None` —— `str.find` 找不到时返回 -1。
        判定被改（比如 `!= -1`）会让没有括号的文本走进循环，
        `range(-1, len(text))` 从末尾开始，行为完全不可预测。
        """
        assert _extract_balanced("没有任何括号的一句话", 0) is None

    def test_unbalanced_parens_return_none_not_a_truncated_expression(self):
        """
        括号没闭合时必须返回 None。返回半截表达式更糟 ——
        它会被当成一条合法建议送去解析，报出的错指向 DSL 语法而不是"抠取失败"。
        """
        assert _extract_balanced("rank(ts_delta(close, 5)", 0) is None

    def test_the_result_is_stripped(self):
        """`.strip()` —— 前后空白会让后续 `parser.parse` 收到带空格的串。"""
        text = "   rank(close)   "
        got = _extract_balanced(text, 0)
        assert got == got.strip() and got == "rank(close)"

    def test_extraction_starts_from_the_given_offset(self):
        """
        `start` 参数决定从哪里开始找 —— 一段文本里有多个表达式时，
        调用方靠它逐个往后抠。忽略 start 会让每次都抠到第一个。
        """
        text = "先 rank(close) 再 zscore(volume)"
        second = text.index("zscore")
        assert _extract_balanced(text, second) == "zscore(volume)"

    def test_a_close_paren_before_any_open_does_not_confuse_the_counter(self):
        """
        文本里先出现 `)` 时，`find("(")` 会跳过它 ——
        循环从第一个 `(` 开始计数，而不是从 start。
        """
        text = ") 然后 rank(close)"
        # 从 0 开始时 `find("(")` 会跳过那个孤立的 `)`，从 `rank(` 的括号
        # 起算 —— 计数到末尾都没归零，于是原样返回整段（而不是崩、
        # 也不是返回到那个孤立 `)` 为止的片段）。
        assert _extract_balanced(text, 0) == ") 然后 rank(close)", (
            "起点前有孤立右括号时的返回值变了 —— 计数起点被改了")
        # 真正的契约：从表达式本身起算时能抠到完整的那一段
        assert _extract_balanced(text, text.index("rank")) == "rank(close)"


# ===========================================================================
# B. 从 LLM 回复里抠 JSON
# ===========================================================================

class TestSafeJsonLoads:

    def test_a_bare_json_object_is_parsed(self):
        assert _safe_json_loads('{"dsl": "rank(close)"}') == {"dsl": "rank(close)"}

    def test_a_json_object_inside_markdown_fences_is_parsed(self):
        """LLM 十次有八次会包一层 ```json ... ```。"""
        raw = '```json\n{"dsl": "rank(close)"}\n```'
        assert _safe_json_loads(raw) == {"dsl": "rank(close)"}

    def test_prose_around_the_object_is_ignored(self):
        raw = '好的，这是结果：{"dsl": "rank(close)"} 希望有帮助'
        assert _safe_json_loads(raw) == {"dsl": "rank(close)"}

    def test_a_multiline_object_is_matched(self):
        """`re.DOTALL` —— 没有它，跨行的 JSON 匹配不到，一律返回 {}。"""
        raw = '{\n  "dsl": "rank(close)",\n  "explanation": "x"\n}'
        got = _safe_json_loads(raw)
        assert got.get("dsl") == "rank(close)", (
            f"跨行 JSON 没有被解析：{got} —— re.DOTALL 疑似被去掉了")

    def test_malformed_json_falls_back_to_an_empty_dict(self):
        """
        `except json.JSONDecodeError: pass` → `return {}`
        —— 返回 {} 而不是抛，让调用方走自己的兜底路径。
        """
        assert _safe_json_loads('{"dsl": unquoted}') == {}

    def test_text_without_any_object_returns_an_empty_dict(self):
        """`if m:` 的另一侧 —— 没有匹配时不能去访问 `m.group()`。"""
        assert _safe_json_loads("完全没有 JSON 的一句话") == {}

    def test_an_empty_string_is_safe(self):
        assert _safe_json_loads("") == {}


# ===========================================================================
# C. 会话隔离 —— 模块 docstring 明确承诺的安全属性
# ===========================================================================

class _FakeRow:
    def __init__(self, role, content):
        self.role, self.content = role, content


class _FakeStore:
    """按 session_id 分桶的假 store，用来验证隔离是否真的按 sid 生效。"""

    def __init__(self):
        self.buckets: dict[str, list] = {}
        self.ensured: list[str] = []

    def ensure_session(self, sid, title=None):
        self.ensured.append(sid)
        self.buckets.setdefault(sid, [])

    def save_message(self, sid, role, content):
        self.buckets.setdefault(sid, []).append(_FakeRow(role, content))

    def get_history(self, sid, limit=None):
        return list(self.buckets.get(sid, []))


class TestSessionIsolation:

    def test_a_session_only_sees_its_own_messages(self):
        """
        模块 docstring：「session A cannot read any records belonging to session B」。

        `self._store.get_history(self._session_id)` —— 传错 sid（或者
        传了别的字段）会让两个会话的历史串台，而且**完全不会报错**：
        B 用户只会觉得 agent 突然记得一些它没说过的话。
        """
        from app.agent._chat_history import SQLAlchemyChatMessageHistory

        store = _FakeStore()
        a = SQLAlchemyChatMessageHistory("sess-A", store)
        b = SQLAlchemyChatMessageHistory("sess-B", store)

        a.add_message(_msg("A 的私密内容"))
        b.add_message(_msg("B 的内容"))

        a_text = [m.content for m in a.messages]
        b_text = [m.content for m in b.messages]
        assert "A 的私密内容" in a_text and "B 的内容" not in a_text, (
            f"会话 A 读到了别的会话的内容：{a_text}")
        assert "B 的内容" in b_text and "A 的私密内容" not in b_text, (
            f"会话 B 读到了别的会话的内容：{b_text}")

    def test_construction_ensures_the_session_row_exists(self):
        """
        `self._store.ensure_session(session_id)` —— 这是外键守卫。
        少了它，第一条消息写入时会撞 FK 约束失败。
        """
        from app.agent._chat_history import SQLAlchemyChatMessageHistory

        store = _FakeStore()
        SQLAlchemyChatMessageHistory("sess-new", store)
        assert store.ensured == ["sess-new"], (
            f"构造时没有 ensure_session：{store.ensured}")

    def test_only_the_most_recent_messages_are_loaded(self):
        """
        `rows = rows[-self._max_history:]` —— token 溢出守卫。

        切片方向写反（`rows[:max]`）会**只读最早的**几条，
        agent 永远看不到刚刚说过的话；去掉切片则会在长会话里撑爆上下文。
        """
        from app.agent._chat_history import SQLAlchemyChatMessageHistory

        store = _FakeStore()
        h = SQLAlchemyChatMessageHistory("s", store, max_history=3)
        for i in range(10):
            h.add_message(_msg(f"msg-{i}"))
        loaded = [m.content for m in h.messages]
        assert len(loaded) == 3, f"max_history=3 却载入了 {len(loaded)} 条"
        assert loaded == ["msg-7", "msg-8", "msg-9"], (
            f"载入的不是**最近** 3 条：{loaded} —— 切片方向反了")

    def test_roles_round_trip_through_the_store(self):
        """
        `role = "user" if isinstance(message, HumanMessage) else "assistant"`
        —— 角色判反会让 agent 把自己说过的话当成用户说的，
        多轮对话的语义整个错乱。
        """
        pytest.importorskip("langchain_core")
        from langchain_core.messages import AIMessage, HumanMessage
        from app.agent._chat_history import SQLAlchemyChatMessageHistory

        store = _FakeStore()
        h = SQLAlchemyChatMessageHistory("s", store)
        h.add_message(HumanMessage(content="用户说的"))
        h.add_message(AIMessage(content="助手说的"))

        roles = [r.role for r in store.buckets["s"]]
        assert roles == ["user", "assistant"], (
            f"角色映射错了：{roles} —— HumanMessage/AIMessage 的判定反了")

        kinds = [type(m).__name__ for m in h.messages]
        assert kinds == ["HumanMessage", "AIMessage"], (
            f"回读时的消息类型不对：{kinds}")

    def test_long_content_is_truncated_with_a_marker(self):
        """
        `if len(content) > 2000: content = content[:2000] + "…[truncated]"`

        **严格大于 2000**，且必须留下截断标记 ——
        没有标记的话，读日志的人会以为 LLM 就回了这么多。
        """
        from app.agent._chat_history import SQLAlchemyChatMessageHistory

        store = _FakeStore()
        h = SQLAlchemyChatMessageHistory("s", store)
        h.add_message(_msg("x" * 2001))
        saved = store.buckets["s"][0].content
        assert saved.endswith("…[truncated]"), (
            f"超长内容没有截断标记：…{saved[-30:]!r}")
        assert saved.startswith("x" * 2000)

    def test_content_at_exactly_the_limit_is_not_truncated(self):
        """边界的另一侧：恰好 2000 字符不该被截。"""
        from app.agent._chat_history import SQLAlchemyChatMessageHistory

        store = _FakeStore()
        h = SQLAlchemyChatMessageHistory("s", store)
        h.add_message(_msg("y" * 2000))
        saved = store.buckets["s"][0].content
        assert saved == "y" * 2000, (
            "恰好 2000 字符被截断了 —— `len(content) > 2000` 被放宽成了 `>=`")

    def test_add_messages_persists_every_item(self):
        """`for m in messages: self.add_message(m)` —— 批量接口不能漏。"""
        from app.agent._chat_history import SQLAlchemyChatMessageHistory

        store = _FakeStore()
        h = SQLAlchemyChatMessageHistory("s", store)
        h.add_messages([_msg("a"), _msg("b"), _msg("c")])
        assert [r.content for r in store.buckets["s"]] == ["a", "b", "c"]

    def test_the_factory_binds_the_store_and_varies_the_session(self):
        """
        `_make_history_factory` 返回的闭包必须
        **固定 store、按参数变 session_id**。
        两者搞反会让所有会话共用一个 id —— 隔离彻底失效。
        """
        from app.agent._chat_history import _make_history_factory

        store = _FakeStore()
        factory = _make_history_factory(store)
        h1, h2 = factory("s1"), factory("s2")
        assert h1._session_id == "s1" and h2._session_id == "s2", (
            "工厂没有按传入的 session_id 绑定")
        assert h1._store is store and h2._store is store, (
            "工厂没有闭包住同一个 store")

    def test_clear_is_a_no_op_that_keeps_history(self):
        """
        `def clear(self): pass` —— 接口存根。**故意不删库**。
        实现成真删会让 LangChain 在某些链路上静默清空用户的历史。
        """
        from app.agent._chat_history import SQLAlchemyChatMessageHistory

        store = _FakeStore()
        h = SQLAlchemyChatMessageHistory("s", store)
        h.add_message(_msg("保留我"))
        h.clear()
        assert [r.content for r in store.buckets["s"]] == ["保留我"], (
            "clear() 真的把历史删了 —— 它应当是只保留的存根")


def _msg(text: str):
    """构造一条带 .content 的消息；没装 langchain 时退化成简单对象。"""
    try:
        from langchain_core.messages import HumanMessage
        return HumanMessage(content=text)
    except ImportError:                                        # pragma: no cover
        class _M:
            content = text
        return _M()


# ===========================================================================
# D. 第二轮：非 LangChain 消息对象
# ===========================================================================

class TestNonMessageObjects:

    def test_an_object_without_a_content_attribute_is_stringified(self):
        """
        `content = message.content if hasattr(message, "content")
                   and isinstance(message.content, str) else str(message)`

        `and` 放宽成 `or` 时，**没有 `.content` 属性**的对象会走到
        `isinstance(message.content, str)` → AttributeError。

        这条路径是真会走到的：LangChain 的某些链路会把裸字符串
        直接塞进 `add_message`，工具调用的返回也不一定带 `.content`。
        崩在这里的症状是"某些对话存不进库"，而且报错指向本模块内部。
        """
        from app.agent._chat_history import SQLAlchemyChatMessageHistory

        store = _FakeStore()
        h = SQLAlchemyChatMessageHistory("s", store)
        h.add_message("一条裸字符串消息")          # 没有 .content
        assert store.buckets["s"][0].content == "一条裸字符串消息", (
            "裸字符串消息没有被 str() 兜住")

    def test_a_non_string_content_is_stringified(self):
        """
        `isinstance(message.content, str)` 这一半：`.content` 是 list
        （多模态消息的常见形态）时，必须走 `str(message)` 而不是把
        list 直接塞进 TEXT 列。
        """
        from app.agent._chat_history import SQLAlchemyChatMessageHistory

        class _MultiModal:
            content = [{"type": "text", "text": "hi"}]

            def __str__(self):
                return "multimodal-fallback"

        store = _FakeStore()
        h = SQLAlchemyChatMessageHistory("s", store)
        h.add_message(_MultiModal())
        saved = store.buckets["s"][0].content
        assert isinstance(saved, str), f"存进去的不是字符串：{type(saved)}"
        assert saved == "multimodal-fallback"
