"""
API 只在本机回环上服务（2026-10-22 用户决定 D11）。

用户原话："首轮仅本机回环，不扩远程。保留真实监听参数和关键动作检查；需要远程访问时先做认证授权，
CORS 不能代替认证。"

    · 监听地址（配置、命令行 --host、UVICORN_HOST）任何一个不是回环 → 拒绝启动；没有"显式解除保险"的开关 ——
      服务没有认证，远程访问要先做认证授权
    · 关键动作（所有改状态的请求：POST / PUT / PATCH / DELETE）只接受回环客户端；CORS 放行的来源也不例外
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestBindIsLoopbackOnly:

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "::"])
    def test_a_non_loopback_bind_is_refused_and_there_is_no_override(self, monkeypatch, host):
        import app.main as m
        monkeypatch.setattr(m.settings, "api_bind_host", host)
        monkeypatch.setenv("ALLOW_INSECURE_BIND", "true")                       # 旧逃生口：不再起作用
        with pytest.raises(RuntimeError, match="认证"):
            m._assert_safe_bind()

    @pytest.mark.parametrize("argv", [["uvicorn", "app.main:app", "--host", "0.0.0.0"],
                                      ["uvicorn", "app.main:app", "--host=0.0.0.0"]])
    def test_the_real_uvicorn_host_is_checked(self, monkeypatch, argv):
        """只看配置项不够 —— `uvicorn --host 0.0.0.0` 与配置不一致时照样会起来（外部审查 F14）。"""
        import app.main as m
        monkeypatch.setattr(m.settings, "api_bind_host", "127.0.0.1")
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(RuntimeError, match="--host='0.0.0.0'"):
            m._assert_safe_bind()

    def test_the_uvicorn_env_var_is_checked(self, monkeypatch):
        import app.main as m
        monkeypatch.setattr(m.settings, "api_bind_host", "127.0.0.1")
        monkeypatch.setattr("sys.argv", ["uvicorn", "app.main:app"])
        monkeypatch.setenv("UVICORN_HOST", "10.0.0.7")
        with pytest.raises(RuntimeError, match="UVICORN_HOST"):
            m._assert_safe_bind()

    def test_loopback_everywhere_starts(self, monkeypatch):
        import app.main as m
        monkeypatch.setattr(m.settings, "api_bind_host", "127.0.0.1")
        monkeypatch.setattr("sys.argv", ["uvicorn", "app.main:app", "--host", "127.0.0.1"])
        monkeypatch.delenv("UVICORN_HOST", raising=False)
        assert m._assert_safe_bind() is None             # 全回环：不误拒（拒绝的方向由上面几条钉住）

    def test_the_settings_no_longer_offer_an_insecure_bind_switch(self):
        from app.config import Settings
        assert "allow_insecure_bind" not in Settings.model_fields


class TestCriticalActionsNeedALoopbackClient:

    @staticmethod
    def _client(host):
        from app.main import app
        return TestClient(app, client=(host, 50000), raise_server_exceptions=False)

    @pytest.mark.parametrize("method,path", [
        ("post", "/api/strategies/1/approve"), ("post", "/api/strategies/1/verify"),
        ("patch", "/api/alphas/1/status"), ("post", "/api/execution/kill_switch/arm"),
        ("delete", "/api/chat/sessions/x"),
    ])
    def test_a_remote_client_cannot_change_state(self, fresh_db, method, path):
        with self._client("10.0.0.5") as c:
            r = getattr(c, method)(path, **({"json": {}} if method != "delete" else {}))
        assert r.status_code == 403, (path, r.status_code, r.text)
        assert "回环" in r.text

    def test_a_cors_allowed_origin_is_not_an_authentication(self, fresh_db):
        from app.config import settings
        origin = settings.cors_origins[0]
        with self._client("10.0.0.5") as c:
            r = c.post("/api/strategies/1/approve", json={}, headers={"Origin": origin})
        assert r.status_code == 403

    def test_a_loopback_client_reaches_the_endpoint(self, fresh_db):
        with self._client("127.0.0.1") as c:
            r = c.post("/api/strategies/999999/approve", json={})
        assert r.status_code != 403, r.text                          # 到了端点本身（这里是策略不存在）

    def test_reads_from_a_remote_client_are_not_the_guards_business(self, fresh_db):
        """读接口不改状态，不在关键动作之列；远程能不能连上由监听地址决定（只听回环）。"""
        with self._client("10.0.0.5") as c:
            assert c.get("/health").status_code != 403
