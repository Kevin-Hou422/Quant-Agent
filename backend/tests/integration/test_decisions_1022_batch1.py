"""
2026-10-22 用户决定 · 批 1：正确性与状态（`docs/DECISIONS_2026-10-22_TRACKER.md` C1 / C2 / C4 / C5 / C6）。

每条断言都对着用户的原话写；先跑、先看它们在旧代码上失败，再改。
"""

from __future__ import annotations

import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from tests.conftest import _make_dataset


# ---------------------------------------------------------------------------
# C1：提案的最终结论、评级、准入一致；子门原始结论保留
# ---------------------------------------------------------------------------

def _verdict(passed=True):
    return SimpleNamespace(passed=passed, sharpe=1.2, deflated_sharpe=.99, t_stat=3.0,
                           reasons=[] if passed else ["未过"],
                           to_dict=lambda: {"passed": passed, "sharpe": 1.2, "deflated_sharpe": .99})


def _propose(frozen=True, procedure=True, frozen_raises=False, procedure_raises=False,
             dsl="rank(close)", experiment=True, monkeypatch=None):
    from app.config import settings
    from app.core.portfolio_manager.strategy_builder import build_strategy_config
    from app.core.portfolio_manager.strategy_gate import StrategyGate
    if monkeypatch is not None:
        monkeypatch.setattr(settings, "tr_experiment_mode", experiment)
    ds = _make_dataset(80, 10)

    def ev(self, *a, **kw):
        if "fixed_combo_weights" in kw:                       # 冻结规格本身
            if frozen_raises:
                raise RuntimeError("门崩了")
            return _verdict(frozen)
        if procedure_raises:                                  # 流程的滚动前推
            raise RuntimeError("流程 OOS 崩了")
        return _verdict(procedure)
    with patch.object(StrategyGate, "evaluate", ev):
        return build_strategy_config({"1": ds["close"].rank(axis=1)}, ds, 10000, method="equal_weight",
                                     factor_dsl={"1": dsl})


class TestC1ProposalConsistency:

    def test_a_failed_procedure_oos_makes_every_view_say_not_passed(self, monkeypatch):
        cfg = _propose(frozen=True, procedure=False, monkeypatch=monkeypatch)
        v = cfg.verdict
        assert cfg.passed is False and v["passed"] is False, "cfg 与 verdict 的最终结论不一致"
        assert v["gates"]["frozen_spec"]["passed"] is True, "冻结规格门的原始结论没保留"
        assert v["gates"]["procedure_oos"]["passed"] is False
        assert v["paper_grade"] != "A", "必须的门未过却标 A"

    def test_outside_experiment_mode_a_failed_required_gate_blocks_paper(self, monkeypatch):
        cfg = _propose(frozen=True, procedure=False, experiment=False, monkeypatch=monkeypatch)
        assert cfg.verdict.get("paper_entry_blocked") is True

    @pytest.mark.parametrize("kw", [{"frozen_raises": True}, {"procedure_raises": True}])
    def test_an_unevaluated_required_gate_is_never_grade_a(self, kw, monkeypatch):
        cfg = _propose(monkeypatch=monkeypatch, **kw)
        v = cfg.verdict
        assert cfg.passed is False and v["passed"] is False
        assert v["paper_grade"] != "A"
        assert any(g.get("evaluated") is False for g in v["gates"].values()), "未评估的门没有标出来"

    def test_a_failed_fundamentals_quality_gate_is_part_of_the_final_verdict(self, monkeypatch, tmp_path):
        from app.config import settings
        monkeypatch.setattr(settings, "fundamentals_dir", str(tmp_path / "f"))
        cfg = _propose(dsl="rank(earnings_yield)", monkeypatch=monkeypatch)
        assert cfg.passed is False and cfg.verdict["passed"] is False
        assert cfg.verdict["gates"]["fundamentals_quality"]["passed"] is False
        assert cfg.verdict["paper_grade"] != "A"

    def test_all_required_gates_passing_is_grade_a(self, monkeypatch):
        cfg = _propose(monkeypatch=monkeypatch)
        assert cfg.passed is True and cfg.verdict["passed"] is True and cfg.verdict["paper_grade"] == "A"

    def test_the_api_reports_the_same_final_verdict(self, fresh_db, test_client, monkeypatch):
        from app.db.strategy_store import StrategyStore
        cfg = _propose(frozen=True, procedure=False, monkeypatch=monkeypatch)
        sid = StrategyStore().save(cfg)
        body = test_client.get(f"/api/strategies/{sid}").json()
        assert body["passed"] is False and body["verdict"]["passed"] is False


# ---------------------------------------------------------------------------
# C2：实际数据源未知 → 不退回 settings；正式证据隔离
# ---------------------------------------------------------------------------

class TestC2UnknownSource:

    def test_the_loop_identity_does_not_claim_the_settings_source(self):
        from app.tasks.daily_trading_loop import DailyTradingLoop
        loop = DailyTradingLoop.__new__(DailyTradingLoop)
        ident = loop._set_identity(_make_dataset(30, 5), None, None)
        assert ident["price_source"] == "" and ident["source_verified"] is False

    def test_the_pipeline_identity_does_not_claim_the_settings_source(self):
        from app.tasks.daily_ingest import _pipeline_identity
        ing = SimpleNamespace(identity={"price_source": "yahoo", "last_day": "2025-03-04"})
        ident = _pipeline_identity(ing, None)
        assert ident["price_source"] == "" and ident["source_verified"] is False
        assert _pipeline_identity(ing, "moomoo")["source_verified"] is True

    def test_unverified_source_days_are_replay_not_forward(self, tmp_path, monkeypatch):
        from tests.integration.test_evidence_versions_1019 import _propose as propose_strategy
        from tests.integration.test_forward_evidence_1018 import _after_close, _panel, _upto
        from app.config import settings
        from app.core.execution.paper_broker import PaperBroker
        from app.db.alpha_store import AlphaResult, AlphaStore
        from app.db.evidence_store import EvidenceStore
        from app.db.position_store import PositionStore
        from app.tasks.daily_trading_loop import DailyTradingLoop
        url = f"sqlite:///{(tmp_path / 'a.db').as_posix()}"
        monkeypatch.setattr(settings, "database_url", url)
        store, ev = AlphaStore(db_url=url), EvidenceStore(url)
        aid = store.save(AlphaResult(dsl="rank(ts_delta(close, 5))", status="candidate"))
        store.update_status(aid, "validated"); store.update_status(aid, "paper")
        sid, _ = propose_strategy(SimpleNamespace(aid=aid))

        def loop():
            return DailyTradingLoop(store=store, evidence_store=ev, broker=PaperBroker(
                store=PositionStore(f"sqlite:///{(tmp_path / 'p.db').as_posix()}"), initial_capital=10_000.0))
        ds = _panel()
        days = ds["close"].index
        for d in days[-3:]:
            loop().run_portfolio(_upto(ds, d), aum=10_000.0, now=_after_close(d))     # 来源未知
        assert ev.strategy_forward_ic(sid) == [], "来源未知的日子被记成了合格前向"
        assert ev.strategy_replay_ic(sid), "探索仍应可继续（记 replay）"


# ---------------------------------------------------------------------------
# C5：试验史 —— 进行中 / 未结案的运行一律不给正式资格
# ---------------------------------------------------------------------------

class TestC5OpenRuns:

    def test_a_fresh_running_run_already_blocks_formal_verdicts(self, fresh_db):
        from app.db.trial_ledger import TrialLedger, trial_history_problem
        TrialLedger().begin_run("gp")                        # 刚开始、远未到 6 小时
        assert trial_history_problem(), "进行中的运行在 6 小时内被当成了完整的试验史"

    def test_a_closed_history_is_complete(self, fresh_db):
        from app.db.trial_ledger import TrialLedger, trial_history_problem
        led = TrialLedger()
        rid = led.begin_run("gp")
        led.add_progress(rid, 3)
        led.finish_run(rid, "done")
        assert trial_history_problem() == ""


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


class TestC6States:

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
