"""
策略审批流转：一组流转 + 谱系在一个事务里完成；批准后可单独激活；验证总要过门。

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


def _cfg(status="proposed"):
    from app.db.strategy_store import StrategyConfig
    from app.dependencies import get_strategy_store
    from tests.conftest import approved_verdict
    s = get_strategy_store()
    return s, s.save(StrategyConfig(factors=["1"], combo_weights={"1": 1.0}, aum=10_000, status=status,
                                    verdict=approved_verdict({"1": "rank(close)"}, {"1": 1.0}, 10_000)))


class TestOneTransaction:

    def test_a_failure_midway_changes_nothing(self, fresh_db):
        from app.db.strategy_store import IllegalStrategyTransition
        s, old = _cfg()
        s.apply_transitions([(old, "approved", "approve", "", "t"),
                             (old, "paper_running", "start_paper_running", "", "t")])
        _, new = _cfg()
        with pytest.raises(IllegalStrategyTransition):
            s.apply_transitions([(new, "approved", "approve", "", "t"),
                                 (old, "retired", "retire", "", "t"),
                                 (new, "proposed", "bogus", "", "t")])      # 第三步非法
        assert s.get(new).status == "proposed" and s.get(old).status == "paper_running"
        assert [d.decision for d in s.get_decisions(new)] == []
        assert [d.decision for d in s.get_decisions(old)] == ["approve", "start_paper_running"]

    def test_approve_and_activate_land_together(self, test_client, fresh_db):
        s, old = _cfg()
        assert test_client.post(f"/api/strategies/{old}/approve", json={"activate": True}).status_code == 200
        _, new = _cfg()
        r = test_client.post(f"/api/strategies/{new}/approve", json={"activate": True, "actor": "kevin"})
        assert r.status_code == 200 and r.json()["status"] == "paper_running"      # 2026-10-22 C6
        assert s.get(old).status == "retired"
        assert [d.decision for d in s.get_decisions(new)] == ["approve", "start_paper_running"]


class TestExplicitActivate:

    def test_an_approved_config_can_be_activated_later(self, test_client, fresh_db):
        s, sid = _cfg()
        assert test_client.post(f"/api/strategies/{sid}/approve", json={"activate": False}).json()["status"] == "approved"
        r = test_client.post(f"/api/strategies/{sid}/activate", json={"reason": "go", "actor": "kevin"})
        assert r.status_code == 200 and r.json()["status"] == "paper_running"
        act = [d for d in s.get_decisions(sid) if d.decision == "start_paper_running"]
        # 2026-10-22 C6：开始纸交易不过门，谱系里写明"未验证"
        assert act and act[0].actor == "kevin" and "未验证" in act[0].reason and "go" in act[0].reason

    def test_only_approved_configs_can_be_activated(self, test_client, fresh_db):
        _, sid = _cfg()
        r = test_client.post(f"/api/strategies/{sid}/activate", json={})
        assert r.status_code == 409 and "仅 approved / integration" in r.json()["detail"]
        assert test_client.post("/api/strategies/999999/activate", json={}).status_code == 404

    def test_verification_applies_the_gate_whatever_the_old_switch_says(self, test_client, fresh_db, monkeypatch):
        """2026-10-22 C6：门在 /verify，且 `tr_enforce_active_gate` 不管开关都不再绕过它。"""
        from app.config import settings
        for flag in (True, False):
            monkeypatch.setattr(settings, "tr_enforce_active_gate", flag)
            s, sid = _cfg()
            test_client.post(f"/api/strategies/{sid}/approve", json={"activate": True})
            r = test_client.post(f"/api/strategies/{sid}/verify", json={})
            assert r.status_code == 409 and s.get(sid).status == "paper_running"
