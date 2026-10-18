"""
外部审查 2026-10-18 F14：统计强度与运营边界。

  · 审批、退役旧配置、激活新配置、写谱系分别提交 → 中途出错留下 approved 却无 active、或谱系不全；
    "批准"之后也没有独立的激活接口。
  · `_assert_safe_bind` 只看 API_BIND_HOST 配置，`uvicorn --host 0.0.0.0` 照样启动。
  · IC 的 t 检验用普通 mean/std·√n，没有考虑序列相关。
  · PBO 只看已选因子的单因子收益矩阵 —— 范围要写进结论。
  · 买入力不预支未成交卖单的现金：要记下实际资金利用率。
"""

from __future__ import annotations

import numpy as np
import pytest


def _cfg(status="proposed"):
    from app.dependencies import get_strategy_store
    from app.db.strategy_store import StrategyConfig
    from tests.conftest import approved_verdict
    s = get_strategy_store()
    return s, s.save(StrategyConfig(factors=["1"], combo_weights={"1": 1.0}, aum=10_000, status=status,
                                    verdict=approved_verdict({"1": "rank(close)"}, {"1": 1.0}, 10_000)))


class TestOneTransaction:

    def test_a_failure_midway_changes_nothing(self, fresh_db):
        from app.db.strategy_store import IllegalStrategyTransition
        s, old = _cfg()
        s.apply_transitions([(old, "approved", "approve", "", "t"), (old, "active", "activate", "", "t")])
        _, new = _cfg()
        with pytest.raises(IllegalStrategyTransition):
            s.apply_transitions([(new, "approved", "approve", "", "t"),
                                 (old, "retired", "retire", "", "t"),
                                 (new, "proposed", "bogus", "", "t")])      # 第三步非法
        assert s.get(new).status == "proposed" and s.get(old).status == "active"
        assert [d.decision for d in s.get_decisions(new)] == []
        assert [d.decision for d in s.get_decisions(old)] == ["approve", "activate"]

    def test_approve_and_activate_land_together(self, test_client, fresh_db):
        s, old = _cfg()
        assert test_client.post(f"/api/strategies/{old}/approve", json={"activate": True}).status_code == 200
        _, new = _cfg()
        r = test_client.post(f"/api/strategies/{new}/approve", json={"activate": True, "actor": "kevin"})
        assert r.status_code == 200 and r.json()["status"] == "active"
        assert s.get(old).status == "retired"
        assert [d.decision for d in s.get_decisions(new)] == ["approve", "activate"]


class TestExplicitActivate:

    def test_an_approved_config_can_be_activated_later(self, test_client, fresh_db):
        s, sid = _cfg()
        assert test_client.post(f"/api/strategies/{sid}/approve", json={"activate": False}).json()["status"] == "approved"
        r = test_client.post(f"/api/strategies/{sid}/activate", json={"reason": "go", "actor": "kevin"})
        assert r.status_code == 200 and r.json()["status"] == "active"
        act = [d for d in s.get_decisions(sid) if d.decision == "activate"]
        assert act and act[0].actor == "kevin" and "TR.4 门" in act[0].reason

    def test_only_approved_configs_can_be_activated(self, test_client, fresh_db):
        _, sid = _cfg()
        r = test_client.post(f"/api/strategies/{sid}/activate", json={})
        assert r.status_code == 409 and "仅 approved" in r.json()["detail"]
        assert test_client.post("/api/strategies/999999/activate", json={}).status_code == 404

    def test_the_active_gate_applies_when_enforced(self, test_client, fresh_db, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "tr_enforce_active_gate", True)
        s, sid = _cfg()
        test_client.post(f"/api/strategies/{sid}/approve", json={"activate": False})
        r = test_client.post(f"/api/strategies/{sid}/activate", json={})
        assert r.status_code == 409 and s.get(sid).status == "approved"


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


class TestHacTstat:

    def test_positive_autocorrelation_shrinks_the_t_stat(self):
        from app.core.lifecycle.promotion_gate import newey_west_tstat
        rng = np.random.default_rng(0)
        e = rng.normal(0, 0.05, 400)
        x = np.empty(400)
        x[0] = e[0]
        for i in range(1, 400):
            x[i] = 0.8 * x[i - 1] + e[i]
        x = x + 0.01
        t_iid = x.mean() / x.std(ddof=1) * np.sqrt(len(x))
        assert newey_west_tstat(x) < 0.6 * t_iid

    def test_iid_series_gets_about_the_same_t(self):
        from app.core.lifecycle.promotion_gate import newey_west_tstat
        x = np.random.default_rng(1).normal(0.01, 0.05, 2000)
        t_iid = x.mean() / x.std(ddof=1) * np.sqrt(len(x))
        assert newey_west_tstat(x) == pytest.approx(t_iid, rel=0.1)

    def test_the_gate_uses_the_smaller_t(self):
        from app.core.lifecycle.promotion_gate import PromotionThresholds, check_active_promotion
        rng = np.random.default_rng(2)
        x = np.empty(120)
        x[0] = 0.0
        for i in range(1, 120):
            x[i] = 0.9 * x[i - 1] + rng.normal(0, 0.02)
        x = x + 0.02
        ok, d = check_active_promotion(list(x), PromotionThresholds(min_forward_days=60, min_ic_mean=0.0,
                                                                    min_ic_tstat=2.0))
        assert d["ic_tstat"] == min(d["ic_tstat_iid"], d["ic_tstat_hac"])
        assert d["ic_tstat_hac"] < d["ic_tstat_iid"]


def test_the_strategy_verdict_states_the_pbo_scope():
    from app.core.portfolio_manager.strategy_gate import StrategyValidationResult
    assert StrategyValidationResult(passed=False, n_factors=1).to_dict()["pbo_scope"] == \
        "selected_factor_returns_only"


def test_the_execution_report_records_capital_utilisation(tmp_path):
    from tests.unit.execution.test_order_manager import D1, _cycle, _env
    e = _env(tmp_path)
    rep = _cycle(e, D1)
    cap = rep.capital
    assert cap["slice_equity"] == 100_000.0 and cap["invested"] == 0.0
    assert cap["planned_buy"] == pytest.approx(200 * 50.25 + 400 * 20.10 + 500 * 10.05)
    assert cap["utilization_after_buys"] == pytest.approx(cap["planned_buy"] / 100_000.0, abs=1e-4)
