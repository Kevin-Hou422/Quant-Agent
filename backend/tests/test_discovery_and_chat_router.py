"""
discovery/discovery_engine.py + api/chat_router.py —— 定钉测试（变异测试驱动）

来由：
  - `discovery_engine`：9 个变异点，首测击杀率 **33.3%**（存活 6）
  - `chat_router`：5 个变异点，首测击杀率 **40.0%**（存活 3）

`DiscoveryEngine` 是 Phase 9「自主发现」的编排器：观察市场 → GP → 验证门 →
写成 CANDIDATE/VALIDATED。它的三处存活项都在**门与计数**上：

  - `TrialLedger().add(pop_size * n_generations + n_optuna_trials)` 的 `*` 与 `+`
    —— 这个数直接喂给验证门的 DSR 去做多重检验膨胀修正。算少了，
       DSR 就偏松，过拟合的候选更容易通过；算多了则反过来。
  - `if auto_validate and save and aid is not None:` 的 `and`
    —— 放宽成 `or` 会在没存库（aid 为 None）时也去跑门并写状态
  - `return False, {...}` 里的两个 `False`
    —— 门**没跑起来**时必须判不通过（fail-closed）。改成 True 就是
       "门崩了 → 当作通过"，这正是 docstring 明写要避免的
  - `validated: bool = False` 的默认值

`chat_router` 的流式端点：
  - `threading.Thread(target=_run, daemon=True)` 的 daemon
    —— 改成 False 后，客户端断开时工作线程不退出，进程停不下来
  - `allow_syn = bool(getattr(settings, "chat_allow_synthetic", False))`
    —— 默认 True 等于聊天路径默认用合成数据
"""
from __future__ import annotations

import inspect

import pytest


# ===========================================================================
# A. DiscoveryEngine —— trial 计数
# ===========================================================================

class TestTrialCounting:

    def test_trial_count_is_pop_times_generations_plus_optuna(self):
        """
        `pop_size * n_generations + n_optuna_trials` —— 这个数是**多重检验的
        试验次数**，喂给验证门的 DSR 做膨胀修正。
        `*`→`/` 会让计数塌成个位数，DSR 几乎不打折，过拟合候选轻松过门；
        `+`→`-` 会让它变负，修正方向反过来。

        这里不打桩公式、而是核对源码里的表达式形状 —— 因为真正跑一轮 GP
        太慢，而这个乘加式本身就是契约。
        """
        from app.core.discovery import discovery_engine as de
        src = inspect.getsource(de.DiscoveryEngine.run)
        assert "self.pop_size * self.n_generations + self.n_optuna_trials" in src, (
            "trial 计数表达式变了 —— 它是验证门 DSR 多重检验修正的唯一输入")

    def test_default_gp_parameters_are_pinned(self):
        """
        默认参数决定每晚跑多大规模。悄悄改小会让发现产出骤降，
        而日志里看不出区别（仍然"跑完了一轮"）。
        """
        from app.core.discovery.discovery_engine import DiscoveryEngine
        p = inspect.signature(DiscoveryEngine.__init__).parameters
        assert p["n_families"].default == 2
        assert p["pop_size"].default == 12
        assert p["n_generations"].default == 4
        assert p["n_optuna_trials"].default == 5
        assert p["oos_ratio"].default == 0.30
        assert p["seed"].default == 42, "随机种子默认值变了 —— 复现性会断"

    def test_trial_count_for_the_defaults(self):
        """把默认参数下的实际数字钉住：12×4+5 = 53。"""
        from app.core.discovery.discovery_engine import DiscoveryEngine
        e = DiscoveryEngine()
        assert e.pop_size * e.n_generations + e.n_optuna_trials == 53


# ===========================================================================
# B. DiscoveryEngine —— 验证门 fail-closed
# ===========================================================================

class TestValidationGateFailClosed:

    @staticmethod
    def _engine():
        from app.core.discovery.discovery_engine import DiscoveryEngine
        return DiscoveryEngine()

    def test_a_crashing_gate_counts_as_not_passed(self, monkeypatch):
        """
        `except Exception: return False, {"gate_error": ...}` ——
        那个 `False` 改成 True 就是"**门崩了当作通过**"，
        而 docstring 明写 "fail-closed：出错视为不通过"。
        崩掉的门会把未经验证的候选直接升成 VALIDATED，进待批准队列。
        """
        import app.core.lifecycle.leak_filter as lf
        monkeypatch.setattr(lf, "leak_filter",
                            lambda dsl, ds: (_ for _ in ()).throw(RuntimeError("boom")))

        class _Store:
            def __init__(self):
                self.updated = []

            def update_status(self, aid, status):
                self.updated.append((aid, status))

        store = _Store()
        passed, detail = self._engine()._run_gate(store, 1, "rank(close)", {})
        assert passed is False, "门抛异常却判成了通过 —— fail-closed 失效"
        assert store.updated == [], "门崩了却仍然写了状态"

    def test_a_crashing_gate_reports_why(self, monkeypatch):
        """
        `return False, None` 会把"判定不通过"和"压根没跑起来"混为一谈。
        现在带诊断字段 —— 两个 `False`（`evaluated`）也必须保持。
        """
        import app.core.lifecycle.leak_filter as lf
        monkeypatch.setattr(lf, "leak_filter",
                            lambda dsl, ds: (_ for _ in ()).throw(ValueError("bad input")))
        _passed, detail = self._engine()._run_gate(None, 1, "rank(close)", {})
        assert detail is not None, "门崩了却没有给出诊断信息"
        assert "gate_error" in detail and "ValueError" in detail["gate_error"], detail
        assert detail.get("evaluated") is False, (
            "门没跑起来，evaluated 却不是 False —— 调用方会以为评估过了")

    def test_an_unreadable_config_falls_back_to_the_strict_gate(self, monkeypatch):
        """
        读不到 `factor_gate_mode` 时应当用 **strict 严门**。
        源码注释明写：原兜底退回 "leak"（较松）方向正好反了，已修。
        这里防止它被改回去。
        """
        from app.core.discovery import discovery_engine as de
        src = inspect.getsource(de.DiscoveryEngine._run_gate)
        assert 'mode = "strict"' in src, (
            "配置读失败时的兜底不再是 strict 严门 —— 方向被改回松门了")

    def test_a_passing_gate_promotes_to_validated(self, monkeypatch):
        import app.core.lifecycle.leak_filter as lf
        monkeypatch.setattr(lf, "leak_filter", lambda dsl, ds: (True, {"ok": True}))

        class _Store:
            def __init__(self):
                self.updated = []

            def update_status(self, aid, status):
                self.updated.append((aid, status))

        store = _Store()
        passed, _ = self._engine()._run_gate(store, 7, "rank(close)", {})
        assert passed is True
        assert store.updated == [(7, "validated")], (
            f"通过门却没有升到 validated：{store.updated}")

    def test_a_failing_gate_does_not_promote(self, monkeypatch):
        import app.core.lifecycle.leak_filter as lf
        monkeypatch.setattr(lf, "leak_filter", lambda dsl, ds: (False, {"ok": False}))

        class _Store:
            def __init__(self):
                self.updated = []

            def update_status(self, aid, status):
                self.updated.append((aid, status))

        store = _Store()
        passed, _ = self._engine()._run_gate(store, 7, "rank(close)", {})
        assert passed is False and store.updated == []

    def test_candidates_default_to_unvalidated(self):
        """
        `validated: bool = False` —— 改成 True 会让每个新候选**默认就是已验证**，
        即使门根本没跑。待批准队列里会出现一批没过门的东西。
        """
        from app.core.discovery.discovery_engine import DiscoveredCandidate
        c = DiscoveredCandidate(alpha_id=None, family="momentum", dsl="rank(close)")
        assert c.validated is False, "新候选默认就是 validated"
        assert c.gate is None
        assert c.to_dict()["validated"] is False

    def test_the_gate_runs_only_when_all_three_conditions_hold(self):
        """
        `if auto_validate and save and aid is not None:` —— 三个都要齐。
        `and`→`or` 会在没存库（aid 为 None）时也去 `store.update_status(None, ...)`。
        """
        from app.core.discovery import discovery_engine as de
        src = inspect.getsource(de.DiscoveryEngine.run)
        assert "auto_validate and save and aid is not None" in src, (
            "自动验证门的三重条件被改了 —— 可能在未存库时就去写状态")


# ===========================================================================
# C. chat_router —— 流式端点
# ===========================================================================

class TestChatRouterStreaming:

    def test_the_worker_thread_is_a_daemon(self):
        """
        `threading.Thread(target=_run, daemon=True)` —— 改成 False 后，
        客户端断开连接时工作线程仍在跑（一轮 GP 可能几分钟到几十分钟），
        **进程停不下来**：Ctrl-C 之后 uvicorn 挂着不退，重启部署直接卡住。
        """
        from app.api import chat_router as cr
        src = inspect.getsource(cr)
        assert "daemon=True" in src, (
            "流式端点的工作线程不再是 daemon —— 客户端断开后进程会停不下来")

    def test_synthetic_data_is_off_by_default_in_chat(self):
        """
        `allow_syn = bool(getattr(settings, "chat_allow_synthetic", False))`
        —— 这个默认 `False` 改成 True 会让聊天路径在配置缺项时
        默认用合成数据，而屏幕上的 Sharpe 不带任何标识。
        """
        from app.api import chat_router as cr
        src = inspect.getsource(cr)
        assert '"chat_allow_synthetic", False' in src, (
            "chat_allow_synthetic 的兜底默认值不再是 False")

    def test_the_stream_loop_terminates_on_done_or_error(self):
        """
        `while True:` 配 `if event.get("type") in ("done", "error"): break` ——
        `while True` 改成 `while False` 会让流一个事件都不发就结束；
        终止条件被改则会让连接永远挂着。
        """
        from app.api import chat_router as cr
        src = inspect.getsource(cr)
        assert "while True:" in src
        assert 'in ("done", "error")' in src, (
            "流式循环的终止条件变了 —— 连接可能永远不关闭")

    def test_queue_timeout_yields_a_ping_instead_of_dying(self):
        """`except _queue.Empty: yield ping` —— 长任务期间靠心跳保活。"""
        from app.api import chat_router as cr
        src = inspect.getsource(cr)
        assert '"type":"ping"' in src, "队列超时后不再发心跳，长任务会被代理断开"


# ===========================================================================
# D. chat_router —— 会话改名/删除的 404 路径
# ===========================================================================

class TestSessionMutationEndpoints:
    """
    两处 `if not ok: raise HTTPException(404)` —— 删掉 `not` 会让
    **成功时报 404、失败时返回成功**：前端删了一个不存在的会话却看到 204，
    真删掉的那个反而报"不存在"。
    """

    @staticmethod
    def _client(store):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.chat_router import chat_router, get_chat_store

        app = FastAPI()
        # chat_router 自带 prefix="/api/chat"，再传一次会变成 /api/chat/api/chat
        app.include_router(chat_router)
        app.dependency_overrides[get_chat_store] = lambda: store
        return TestClient(app)

    class _Store:
        def __init__(self, ok: bool):
            self.ok = ok
            self.calls = []

        def update_session_title(self, sid, title):
            self.calls.append(("rename", sid, title))
            return self.ok

        def delete_session(self, sid):
            self.calls.append(("delete", sid))
            return self.ok

        def get_session(self, sid):
            import datetime
            return type("S", (), {"id": sid, "title": "t",
                                  "created_at": datetime.datetime(2026, 1, 1)})()

    def test_renaming_a_missing_session_returns_404(self):
        c = self._client(self._Store(ok=False))
        r = c.patch("/api/chat/sessions/nope", json={"title": "x"})
        assert r.status_code == 404, (
            f"改名不存在的会话返回了 {r.status_code} —— `if not ok` 疑似被取反")

    def test_renaming_an_existing_session_succeeds(self):
        c = self._client(self._Store(ok=True))
        r = c.patch("/api/chat/sessions/s1", json={"title": "new"})
        assert r.status_code == 200, (
            f"改名成功的会话返回了 {r.status_code} —— 成功被当成了失败")
        assert r.json()["session_id"] == "s1"

    def test_deleting_a_missing_session_returns_404(self):
        c = self._client(self._Store(ok=False))
        assert c.delete("/api/chat/sessions/nope").status_code == 404

    def test_deleting_an_existing_session_returns_204(self):
        c = self._client(self._Store(ok=True))
        assert c.delete("/api/chat/sessions/s1").status_code == 204, (
            "删除成功却没有返回 204")

    def test_the_two_outcomes_are_distinguishable(self):
        """两条路恒同结果时这条相等断言会成立 —— 必须不等。"""
        ok = self._client(self._Store(ok=True)).delete("/api/chat/sessions/s").status_code
        bad = self._client(self._Store(ok=False)).delete("/api/chat/sessions/s").status_code
        assert ok != bad, "删除成功与失败返回了同一个状态码"
