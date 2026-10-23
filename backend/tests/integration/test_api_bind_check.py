"""
API 启动时的监听地址检查：看真实的 uvicorn 参数与环境变量，回环才放行，显式覆盖仍可用。

来源（原测试文件的说明，保留出处）：

外部审查 2026-10-18 F14：统计强度与运营边界。

  · 审批、退役旧配置、激活新配置、写谱系分别提交 → 中途出错留下 approved 却无 active、或谱系不全；
    "批准"之后也没有独立的激活接口。
  · `_assert_safe_bind` 只看 API_BIND_HOST 配置，`uvicorn --host 0.0.0.0` 照样启动。
  · IC 的 t 检验用普通 mean/std·√n，没有考虑序列相关。
  · PBO 只看已选因子的单因子收益矩阵 —— 范围要写进结论。
  · 买入力不预支未成交卖单的现金：要记下实际资金利用率。
"""

from __future__ import annotations

import pytest


class TestBindCheck:

    @pytest.mark.parametrize("argv", [["uvicorn", "app.main:app", "--host", "0.0.0.0"],
                                      ["uvicorn", "app.main:app", "--host=0.0.0.0"]])
    def test_the_real_uvicorn_host_is_checked(self, monkeypatch, argv):
        import sys

        import app.main as m
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(m.settings, "allow_insecure_bind", False)
        with pytest.raises(RuntimeError, match="--host='0.0.0.0'"):
            m._assert_safe_bind()

    def test_the_uvicorn_env_var_is_checked(self, monkeypatch):
        import sys

        import app.main as m
        monkeypatch.setattr(sys, "argv", ["uvicorn", "app.main:app"])
        monkeypatch.setenv("UVICORN_HOST", "10.0.0.5")
        monkeypatch.setattr(m.settings, "allow_insecure_bind", False)
        with pytest.raises(RuntimeError, match="UVICORN_HOST"):
            m._assert_safe_bind()

    def test_loopback_everywhere_starts(self, monkeypatch):
        import sys

        import app.main as m
        monkeypatch.setattr(sys, "argv", ["uvicorn", "app.main:app", "--host", "127.0.0.1"])
        monkeypatch.delenv("UVICORN_HOST", raising=False)
        assert m._argv_hosts() == ["127.0.0.1"]
        assert m._assert_safe_bind() is None            # 不抛 = 允许启动

    def test_the_explicit_override_still_works(self, monkeypatch):
        import sys

        import app.main as m
        monkeypatch.setattr(sys, "argv", ["uvicorn", "app.main:app", "--host", "0.0.0.0"])
        monkeypatch.setattr(m.settings, "allow_insecure_bind", True)
        assert m._argv_hosts() == ["0.0.0.0"]
        assert m._assert_safe_bind() is None            # 显式解除保险后允许启动
