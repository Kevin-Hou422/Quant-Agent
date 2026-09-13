"""
POST /api/chat/stream —— SSE 流式对话端点

**为什么这个文件是新加的**

整理 tests/ 目录时删掉了 `meta/test_invariants.py` 里两条与
`meta/test_lessons_enforced.py` 逐字重复的用例，随后
`test_no_new_untested_api_route` 立刻变红，指出 `/api/chat/stream`
没有任何测试触达。

回头查才发现：这条路由此前之所以"算被覆盖"，**只是因为那条被删用例的
docstring 里写了 `/api/chat/stream` 这个字符串**——
enforcement 的判据是"路径字面量在测试源码里出现过"，注释与 docstring
同样命中。41 条 API 路由里只有这一条是这种假覆盖，但它恰好是
**前端实际消费的那一条**（DEV_LESSONS §R：只给 POST /api/chat 加东西
等于只修了一半）。

这和「自伤教训 #6」（源码子串断言，同一串出现多次就杀不掉）是同一类错误，
只不过这次出现在项目自己的 enforcement 检查里。

测什么
------
SSE 的契约不在"回复内容"，而在**帧格式与终止条件**：
  - Content-Type 必须是 text/event-stream（否则浏览器不当 SSE 解析）
  - 每帧是 `data: <json>\\n\\n`
  - 事件流必须以 `done` 或 `error` **终止**（不终止 = 前端转圈到超时）
  - 终局 `done` 必须带 `data_source`（§R：两条路径都要带，不能只修 POST）
  - 三个反缓冲头（Cache-Control / X-Accel-Buffering / Connection）——
    少任何一个，nginx 或浏览器会把整个流缓冲到结束才吐，"流式"名存实亡
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _events(resp) -> list[dict]:
    """把 SSE 响应体解析成事件列表，顺带校验帧格式。"""
    body = resp.text
    out: list[dict] = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        assert chunk.startswith("data: "), (
            f"SSE 帧不是以 `data: ` 开头：{chunk[:80]!r} —— "
            f"浏览器的 EventSource 会整帧丢弃")
        out.append(json.loads(chunk[len("data: "):]))
    return out


@pytest.fixture(scope="module")
def stream_response(client):
    """整个模块共用一次真实调用（走完整 Workflow A，约几十秒）。"""
    return client.post("/api/chat/stream",
                       json={"message": "rank(close)", "session_id": "sse_test"})


class TestStreamFraming:

    def test_the_content_type_marks_it_as_an_event_stream(self, stream_response):
        assert stream_response.status_code == 200, stream_response.text[:400]
        ctype = stream_response.headers.get("content-type", "")
        assert ctype.startswith("text/event-stream"), (
            f"Content-Type 是 {ctype!r} —— 不是 text/event-stream 时"
            f"浏览器不会按 SSE 解析，前端收不到任何增量")

    def test_the_anti_buffering_headers_are_present(self, stream_response):
        """
        少任何一个头，nginx / 浏览器都可能把整个流缓冲到结束才一次性吐出，
        "流式"就名存实亡 —— 而接口仍然 200，测不出来。
        """
        h = {k.lower(): v for k, v in stream_response.headers.items()}
        assert "no-cache" in h.get("cache-control", ""), (
            f"Cache-Control={h.get('cache-control')!r}，缺 no-cache")
        assert h.get("x-accel-buffering") == "no", (
            f"X-Accel-Buffering={h.get('x-accel-buffering')!r} —— "
            f"nginx 会缓冲整个响应")
        assert "keep-alive" in h.get("connection", "").lower(), (
            f"Connection={h.get('connection')!r}")

    def test_every_frame_is_valid_json_after_the_data_prefix(self, stream_response):
        evs = _events(stream_response)
        assert evs, "事件流是空的"
        for e in evs:
            assert isinstance(e, dict) and "type" in e, f"事件缺 type 字段：{e}"

    def test_the_stream_terminates_with_done_or_error(self, stream_response):
        """
        `if event.get("type") in ("done", "error"): break` ——
        不终止的流会让前端一直转圈到 300 秒超时。
        """
        evs = _events(stream_response)
        assert evs[-1]["type"] in ("done", "error"), (
            f"事件流以 {evs[-1]['type']!r} 结束，应当是 done 或 error —— "
            f"前端会一直等到超时。完整序列：{[e['type'] for e in evs]}")
        # 终止事件必须是**最后一个**，后面不能再有
        kinds = [e["type"] for e in evs]
        terminal = [i for i, k in enumerate(kinds) if k in ("done", "error")]
        assert terminal == [len(kinds) - 1], (
            f"终止事件出现在中间：{kinds}")


class TestStreamPayload:

    def test_the_done_event_carries_the_data_source(self, stream_response):
        """
        §R 的具体形态：前端消费的是 SSE，不是 POST /api/chat。
        只给 POST 加 `data_source` 等于只修了一半 ——
        终局事件不带来源，用户无法分辨屏幕上的 Sharpe 是不是随机数。
        """
        evs = _events(stream_response)
        if evs[-1]["type"] == "error":
            pytest.fail(f"流式对话报错，无法校验 done 载荷：{evs[-1]}")
        result = evs[-1].get("result") or {}
        assert "data_source" in result, (
            f"done 事件的 result 里没有 data_source：{sorted(result)}")
        assert result["data_source"], "data_source 是空值"

    def test_the_done_event_carries_the_result_contract(self, stream_response):
        evs = _events(stream_response)
        if evs[-1]["type"] == "error":
            pytest.skip(f"本次流式对话以 error 结束：{evs[-1]['message'][:120]}")
        result = evs[-1]["result"]
        for k in ("reply", "dsl", "metrics"):
            assert k in result, f"done 的 result 缺字段 {k}：{sorted(result)}"

    def test_progress_text_events_precede_the_terminal_event(self, stream_response):
        """
        这个端点存在的理由就是**增量进度**。只有一个 done 事件的话，
        它与非流式的 POST /api/chat 没有任何区别。
        """
        kinds = [e["type"] for e in _events(stream_response)]
        assert kinds.count("text") >= 1, (
            f"整条流里一个 text 进度事件都没有：{kinds} —— "
            f"流式端点退化成了一次性返回")
