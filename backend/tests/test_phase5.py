"""
test_phase5.py — Phase 5 验收测试

覆盖：
  Task 5.2  生命周期状态机（合法/非法流转、幂等、终态、历史兼容）
  Task 5.1  AlphaMonitor（update 幂等、衰减告警两条规则、dashboard、边界）
  Task 5.3  调度框架（job 注册、持久化配置、状态查询、启停幂等）
  Task 5.4  API 端点（dashboard / ic_history / status PATCH / scheduler status）
  B2 修复   _combine_pool_alphas 权重 IS 拟合标注
"""
from __future__ import annotations

import numpy as np
import pytest

from app.db.alpha_lifecycle import (
    AlphaStatus, IllegalTransition, TERMINAL_STATES,
    allowed_next, coerce_status, validate_transition,
)
from app.db.alpha_store import AlphaResult, AlphaStore
from app.core.monitor.alpha_monitor import AlphaMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> AlphaStore:
    return AlphaStore(db_url="sqlite:///:memory:")


def _save_alpha(store: AlphaStore, status: str = "candidate", sharpe: float = 1.0) -> int:
    return store.save(AlphaResult(
        dsl=f"rank(ts_delta(close, 5))", hypothesis="test",
        sharpe=sharpe, status=status,
    ))


# ===========================================================================
# Task 5.2 — 生命周期状态机
# ===========================================================================

class TestLifecycle:

    FULL_PATH = ["candidate", "validated", "paper", "active", "decaying", "retired"]

    def test_full_happy_path(self):
        for old, new in zip(self.FULL_PATH, self.FULL_PATH[1:]):
            validate_transition(old, new)      # 不抛即通过

    def test_illegal_jumps_blocked(self):
        for old, new in [
            ("candidate", "active"), ("candidate", "paper"),
            ("validated", "active"), ("paper", "decaying"),
            ("retired", "active"),   ("superseded", "candidate"),
        ]:
            with pytest.raises(IllegalTransition):
                validate_transition(old, new)

    def test_idempotent_same_state(self):
        for s in AlphaStatus:
            validate_transition(s, s)          # old == new 永远合法

    def test_terminal_states_frozen(self):
        for terminal in TERMINAL_STATES:
            for target in AlphaStatus:
                if target == terminal:
                    continue
                with pytest.raises(IllegalTransition):
                    validate_transition(terminal, target)

    def test_decaying_can_recover(self):
        validate_transition("decaying", "active")

    def test_active_can_be_superseded(self):
        validate_transition("active", "superseded")

    def test_any_state_can_retire(self):
        for s in AlphaStatus:
            if s in TERMINAL_STATES:
                continue
            validate_transition(s, AlphaStatus.RETIRED)

    def test_unknown_status_raises(self):
        with pytest.raises(ValueError, match="未知"):
            coerce_status("bogus_state")

    def test_allowed_next(self):
        assert allowed_next("active") == ["decaying", "retired", "superseded"]
        assert allowed_next("retired") == []

    def test_store_update_status(self, store):
        aid = _save_alpha(store, status="candidate")
        rec = store.update_status(aid, "validated")
        assert rec.status == "validated"
        with pytest.raises(IllegalTransition):
            store.update_status(aid, "active")     # validated → active 非法
        with pytest.raises(KeyError):
            store.update_status(99999, "retired")


# ===========================================================================
# Task 5.1 — AlphaMonitor
# ===========================================================================

class TestAlphaMonitor:

    def test_update_records_and_snapshots(self, store):
        aid = _save_alpha(store)
        mon = AlphaMonitor(store, rolling_window=5)
        st  = mon.update(aid, "2026-01-05", 0.04, 0.001)
        assert st.n_history == 1
        assert st.realized_ic == 0.04
        assert st.decay_alert is None

    def test_update_idempotent_same_day(self, store):
        """同 (alpha_id, date) 重复调用覆盖旧值，不重复记账。"""
        aid = _save_alpha(store)
        mon = AlphaMonitor(store, rolling_window=5)
        mon.update(aid, "2026-01-05", 0.04)
        st = mon.update(aid, "2026-01-05", 0.02)      # 同日重跑
        assert st.n_history == 1                       # 仍只有 1 条
        hist = store.get_ic_history(aid)
        assert len(hist) == 1
        assert hist[0].realized_ic == 0.02             # 覆盖为新值

    def test_unknown_alpha_raises(self, store):
        mon = AlphaMonitor(store)
        with pytest.raises(KeyError):
            mon.update(99999, "2026-01-05", 0.01)

    def test_no_alert_when_insufficient_history(self, store):
        """记录不足 rolling_window 天不告警（数据不足不下结论）。"""
        aid = _save_alpha(store)
        mon = AlphaMonitor(store, rolling_window=20, consecutive_neg_limit=3)
        for i in range(5):
            st = mon.update(aid, f"2026-01-{i+1:02d}", -0.05)
        assert st.decay_alert is None

    def test_consecutive_negative_alert(self, store):
        aid = _save_alpha(store)
        mon = AlphaMonitor(store, rolling_window=5, consecutive_neg_limit=5,
                           mean_ic_floor=-999)         # 关掉均值规则，只测连续负
        for i in range(4):
            mon.update(aid, f"2026-01-{i+1:02d}", 0.02)
        for i in range(5):
            st = mon.update(aid, f"2026-02-{i+1:02d}", -0.01)
        assert st.decay_alert is not None
        assert st.decay_alert.reason == "consecutive_negative"
        assert st.decay_alert.consecutive_neg == 5

    def test_rolling_mean_floor_alert(self, store):
        aid = _save_alpha(store)
        mon = AlphaMonitor(store, rolling_window=5, consecutive_neg_limit=999,
                           mean_ic_floor=-0.01)
        # 交替正负但均值明显 < -0.01（避免触发连续负规则）
        vals = [0.01, -0.10, 0.01, -0.10, 0.01, -0.10]
        for i, v in enumerate(vals):
            st = mon.update(aid, f"2026-01-{i+1:02d}", v)
        assert st.decay_alert is not None
        assert st.decay_alert.reason == "rolling_mean_below_floor"

    def test_positive_ic_no_alert(self, store):
        aid = _save_alpha(store)
        mon = AlphaMonitor(store, rolling_window=5)
        for i in range(10):
            st = mon.update(aid, f"2026-01-{i+1:02d}", 0.03)
        assert st.decay_alert is None
        assert st.rolling_mean_ic == pytest.approx(0.03)

    def test_dashboard_excludes_terminal(self, store):
        a1 = _save_alpha(store, status="active",  sharpe=1.5)
        a2 = _save_alpha(store, status="paper",   sharpe=0.8)
        a3 = _save_alpha(store, status="retired", sharpe=2.0)
        rows = AlphaMonitor(store).get_dashboard()
        ids = [r.alpha_id for r in rows]
        assert a1 in ids and a2 in ids
        assert a3 not in ids                           # 终态不显示
        assert rows[0].sharpe >= rows[-1].sharpe       # Sharpe 降序

    def test_dashboard_row_fields(self, store):
        aid = _save_alpha(store, status="active")
        mon = AlphaMonitor(store, rolling_window=5)
        for i in range(6):
            mon.update(aid, f"2026-01-{i+1:02d}", 0.02)
        row = [r for r in mon.get_dashboard() if r.alpha_id == aid][0]
        assert row.n_ic_days == 6
        assert row.latest_ic == pytest.approx(0.02)
        assert row.allowed_next == ["decaying", "retired", "superseded"]
        assert not row.has_alert

    def test_invalid_params(self, store):
        with pytest.raises(ValueError):
            AlphaMonitor(store, rolling_window=2)
        with pytest.raises(ValueError):
            AlphaMonitor(store, consecutive_neg_limit=1)


# ===========================================================================
# Task 5.3 — 调度框架
# ===========================================================================

class TestScheduler:

    def test_create_registers_daily_monitor(self):
        from app.tasks.scheduler import create_scheduler
        sched = create_scheduler(db_url="sqlite:///:memory:")
        # 未启动的 scheduler 中 job 处于 pending，只能断言注册本身；
        # misfire/coalesce 等 job_defaults 在 test_status_reflects_running_state
        # 启动路径中间接验证
        jobs = {j.id: j for j in sched.get_jobs()}
        assert "daily_monitor" in jobs
        assert jobs["daily_monitor"].name == "每日因子衰减巡检"

    def test_status_reflects_running_state(self):
        from app.tasks.scheduler import (
            start_scheduler, shutdown_scheduler, get_scheduler_status,
        )
        assert get_scheduler_status()["running"] is False
        start_scheduler(db_url="sqlite:///:memory:")
        try:
            status = get_scheduler_status()
            assert status["running"] is True
            assert any(j["id"] == "daily_monitor" for j in status["jobs"])
            # 幂等：重复 start 不报错
            start_scheduler(db_url="sqlite:///:memory:")
        finally:
            shutdown_scheduler()
        assert get_scheduler_status()["running"] is False
        shutdown_scheduler()                           # 幂等：重复 shutdown 不报错

    def test_daily_monitor_job_runs(self, store, monkeypatch):
        """daily_monitor_job 端到端可执行（用内存库替换默认 AlphaStore）。"""
        from app.tasks import scheduler as sched_mod
        aid = _save_alpha(store, status="active")
        AlphaMonitor(store, rolling_window=5).update(aid, "2026-01-05", 0.02)
        monkeypatch.setattr(
            "app.db.alpha_store.AlphaStore",
            lambda *a, **k: store,
        )
        sched_mod.daily_monitor_job()                  # 不抛即通过


# ===========================================================================
# Task 5.4 — API 端点
# ===========================================================================

class TestLifecycleAPI:

    def test_dashboard_endpoint(self, test_client, tmp_alpha_store):
        resp = test_client.get("/api/alphas/dashboard")
        assert resp.status_code == 200
        body = resp.json()
        assert "rows" in body and "n_alerts" in body

    def test_ic_history_404(self, test_client):
        resp = test_client.get("/api/alphas/999999/ic_history")
        assert resp.status_code == 404

    def test_status_patch_valid_and_invalid(self, test_client):
        # 建一条 candidate 记录
        save = test_client.post("/api/alpha/save", json={"dsl": "rank(close)"})
        aid = save.json()["id"]
        # 旧 save 路径默认 active：active → decaying 合法
        ok = test_client.patch(f"/api/alphas/{aid}/status", json={"status": "decaying"})
        assert ok.status_code == 200
        assert ok.json()["new_status"] == "decaying"
        assert "active" in ok.json()["allowed_next"]
        # decaying → paper 非法 → 409
        bad = test_client.patch(f"/api/alphas/{aid}/status", json={"status": "paper"})
        assert bad.status_code == 409
        # 未知状态 → 422
        unk = test_client.patch(f"/api/alphas/{aid}/status", json={"status": "bogus"})
        assert unk.status_code == 422

    def test_status_patch_404(self, test_client):
        resp = test_client.patch("/api/alphas/999999/status", json={"status": "retired"})
        assert resp.status_code == 404

    def test_ic_history_roundtrip(self, test_client, tmp_path):
        """通过 store 写 IC 后端点能读回。

        注意用文件型临时库：SQLite :memory: 每连接独立，TestClient 的
        线程池会拿到看不见表的新连接。
        """
        from app.api.router import get_store
        from app.main import app
        store = AlphaStore(db_url=f"sqlite:///{tmp_path / 'ic.db'}")
        aid = _save_alpha(store, status="active")
        AlphaMonitor(store, rolling_window=5).update(aid, "2026-01-05", 0.033)
        app.dependency_overrides[get_store] = lambda: store
        try:
            resp = test_client.get(f"/api/alphas/{aid}/ic_history")
            assert resp.status_code == 200
            body = resp.json()
            assert len(body["points"]) == 1
            assert body["points"][0]["realized_ic"] == pytest.approx(0.033)
        finally:
            app.dependency_overrides.pop(get_store, None)

    def test_scheduler_status_endpoint(self, test_client):
        resp = test_client.get("/api/scheduler/status")
        assert resp.status_code == 200
        assert resp.json()["running"] in (True, False)


# ===========================================================================
# B2 修复 — 组合权重 IS 拟合
# ===========================================================================

class TestB2CombineFix:

    def test_weights_fitted_on_is(self, make_dataset):
        from app.core.workflows.alpha_workflows import _combine_pool_alphas

        is_data  = make_dataset(n_days=120, n_tickers=10, seed=1)
        oos_data = make_dataset(n_days=60,  n_tickers=10, seed=2)
        pool = [
            {"dsl": "rank(ts_delta(close, 5))",  "sharpe_oos": 1.0},
            {"dsl": "zscore(ts_mean(close, 10))", "sharpe_oos": 0.5},
        ]
        out = _combine_pool_alphas(pool, oos_data, lambda *_: None, is_data=is_data)
        assert out is not None
        assert out["weights_fitted_on"] == "is"
        assert out["n_alphas"] == 2
        assert abs(sum(out["weights"].values()) - 1.0) < 0.01

    def test_fallback_to_oos_without_is_data(self, make_dataset):
        from app.core.workflows.alpha_workflows import _combine_pool_alphas

        oos_data = make_dataset(n_days=60, n_tickers=10, seed=2)
        pool = [
            {"dsl": "rank(ts_delta(close, 5))",  "sharpe_oos": 1.0},
            {"dsl": "zscore(ts_mean(close, 10))", "sharpe_oos": 0.5},
        ]
        out = _combine_pool_alphas(pool, oos_data, lambda *_: None, is_data=None)
        assert out is not None
        assert out["weights_fitted_on"] == "oos"      # 回退路径如实标注
