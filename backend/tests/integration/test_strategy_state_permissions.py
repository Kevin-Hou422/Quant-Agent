"""
策略状态与权限：工程联调 / 纸交易运行 / 已验证运行 的交易、证据、正式天数权限；只有验证接口能进已验证状态。

来源（原测试文件的说明，保留出处）：

2026-10-22 用户决定 · 批 1：正确性与状态（`docs/DECISION_IMPLEMENTATION_TRACKER.md` C1 / C2 / C4 / C5 / C6）。

每条断言都对着用户的原话写；先跑、先看它们在旧代码上失败，再改。
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# C6：工程联调 / PAPER_RUNNING / VERIFIED_ACTIVE
# ---------------------------------------------------------------------------

def _saved_strategy(design_at="2000-01-03T00:00:00"):
    from app.db.strategy_store import StrategyConfig, StrategyStore
    from tests.conftest import approved_verdict
    v = approved_verdict({"1": "rank(close)"}, {"1": 1.0}, 10_000.0, design_at=design_at)
    ss = StrategyStore()
    sid = ss.save(StrategyConfig(factors=["1"], combo_weights={"1": 1.0}, aum=10_000.0, verdict=v))
    return ss, sid, v["spec_hash"]


class TestStrategyStates:

    def test_starting_to_collect_evidence_is_paper_running_not_verified(self, fresh_db, test_client):
        ss, sid, _ = _saved_strategy()
        r = test_client.post(f"/api/strategies/{sid}/approve", json={"activate": True})
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "paper_running"
        assert r.json()["verified"] is False

    def test_verification_needs_the_gate_whatever_the_settings_say(self, fresh_db, test_client, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "tr_enforce_active_gate", False)          # 关门也不行
        ss, sid, _ = _saved_strategy()
        test_client.post(f"/api/strategies/{sid}/approve", json={"activate": True})
        r = test_client.post(f"/api/strategies/{sid}/verify", json={})
        assert r.status_code == 409, "证据不足也被标成了已验证"
        assert ss.get(sid).status == "paper_running"

    def test_verification_passes_on_this_strategys_own_registered_evidence(self, fresh_db, test_client):
        from app.db.evidence_store import EvidenceStore
        ss, sid, h = _saved_strategy()
        test_client.post(f"/api/strategies/{sid}/approve", json={"activate": True})
        ev = EvidenceStore()
        for i, d in enumerate(pd.bdate_range("2024-01-02", periods=70)):
            ev.record_strategy_ic(sid, h, d, 0.08 + 0.01 * (i % 3), d - timedelta(days=1), d - timedelta(hours=20))
        r = test_client.post(f"/api/strategies/{sid}/verify", json={})
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "verified_active" and r.json()["verified"] is True

    def test_another_strategys_evidence_cannot_verify_this_one(self, fresh_db, test_client):
        from app.db.evidence_store import EvidenceStore
        ss, sid, h = _saved_strategy()
        _, other, h2 = _saved_strategy()
        test_client.post(f"/api/strategies/{sid}/approve", json={"activate": True})
        ev = EvidenceStore()
        for i, d in enumerate(pd.bdate_range("2024-01-02", periods=70)):
            ev.record_strategy_ic(other, h2, d, 0.08 + 0.01 * (i % 3), d - timedelta(days=1), d - timedelta(hours=20))
        assert test_client.post(f"/api/strategies/{sid}/verify", json={}).status_code == 409

    def test_a_legacy_active_row_is_not_verified(self):
        from app.core.lifecycle.strategy_states import permissions
        assert permissions("active")["verified"] is False
        assert permissions("paper_running") == dict(permissions("paper_running"), verified=False,
                                                   trades_simulate=True, forward_evidence=True)
        assert permissions("verified_active")["verified"] is True
        assert permissions("integration")["forward_evidence"] is False
        assert permissions("integration")["formal_days"] is False

    def test_the_daily_loop_trades_a_paper_running_strategy(self, fresh_db):
        from app.db.strategy_store import StrategyStore
        ss, sid, _ = _saved_strategy()
        ss.update_status(sid, "approved")
        ss.update_status(sid, "paper_running")
        assert StrategyStore().latest_running().id == sid

    def test_integration_days_are_never_formal(self):
        from app.db.forward_record import build_daily_record
        rec = build_daily_record({"execution": {"mode": "moomoo_paper", "reconcile": {"status": "clean"}}},
                                 strategy_id=1, spec_hash="h", decision_date="2025-03-04",
                                 now=pd.Timestamp("2025-03-04 22:30", tz="UTC"),
                                 dataset={"close": pd.DataFrame({"A": [1.0, 1.1]},
                                                                index=pd.bdate_range("2025-03-03", periods=2))},
                                 paper_history=[], forward_run=SimpleNamespace(started_at=pd.Timestamp("2025-01-01")),
                                 strategy_status="integration")
        assert rec["formal"] is False and rec["formal_reason"] == "integration"

    def test_formal_forward_cannot_start_for_a_strategy_that_is_not_running(self, fresh_db):
        from app.db.forward_record import ForwardRecordStore
        ss, sid, h = _saved_strategy()
        with pytest.raises(ValueError):
            ForwardRecordStore().start(sid, h, "kevin", "go", strategy_status="approved")
        with pytest.raises(ValueError):
            ForwardRecordStore().start(sid, h, "kevin", "go", strategy_status="integration")
