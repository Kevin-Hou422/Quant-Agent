"""正式收益修订：保留决策，只发布经核对的结果版本。"""

from __future__ import annotations

import copy
import json
import sqlite3
import threading

import pytest

from app.db.forward_record import ForwardRecordStore


@pytest.fixture
def store(tmp_path):
    s = ForwardRecordStore(db_url=f"sqlite:///{(tmp_path / 'forward.db').as_posix()}")
    yield s
    s._engine.dispose()  # noqa: SLF001


def _day(day, *, sid=1, formal=True, previous=None, value=None):
    return {"strategy_id": sid, "decision_date": day, "spec_hash": "spec", "formal": formal,
            "decision": {"target_weights": {"AAA": 0.5}}, "target_weights": {"AAA": 0.5},
            "outcome": {"of_decision": previous, "counts": previous is not None},
            "reconcile_status": "clean", "net_return": {"broker": value, "paper": 0.01},
            "excess_return": {"label": "相对基准", "broker_vs_full_benchmark": None,
                              "paper_vs_full_benchmark": None, "paper_exposure_matched": None,
                              "broker_exposure_matched": None},
            "costs": {"paper_cost_bps": None, "broker_slippage_bps": None, "broker_fees": 0.0},
            "drawdown": {"broker": None, "paper": None}}


def _setup(store):
    previous = _day("2025-03-03")
    base = _day("2025-03-04", previous="2025-03-03")
    store.record_day(previous)
    store.record_day(base)
    return base


def _revise(store, base, **changes):
    revision = copy.deepcopy(base)
    revision.update(changes)
    assert store.record_day(revision) == "revision"
    return store.history(base["strategy_id"], base["decision_date"])[-1]["event_id"]


def test_correction_requires_review_and_preserves_the_original_decision(store):
    base = _setup(store)
    revision_id = _revise(store, base, formal=False, net_return={"broker": 0.01, "paper": 0.01})
    pending = store.formal_performance(1)[0]
    assert pending["performance_status"] == "pending_review" and pending["net_return"] is None
    assert pending["pending_revision_ids"] == [revision_id]
    approval_id = store.confirm_outcome_revision(1, "2025-03-04", revision_id, actor="reviewer", reason="核对券商回报")
    result = store.formal_performance(1)[0]
    assert result["net_return"] == {"broker": 0.01, "paper": 0.01}
    assert result["performance_status"] == "confirmed" and result["pending_revision_ids"] == []
    assert result["confirmation"] == {"source_event_id": revision_id, "actor": "reviewer", "reason": "核对券商回报"}
    chain = store.history(1, "2025-03-04")
    assert chain[0]["payload"]["decision"] == base["decision"]
    assert chain[-1]["event_id"] == approval_id and chain[-1]["kind"] == "outcome_confirm"
    with sqlite3.connect(store._engine.url.database) as con:  # noqa: SLF001
        raw = con.execute("SELECT payload FROM forward_day_events WHERE id = ?", (approval_id,)).fetchone()[0]
    assert "核对券商回报" in raw
    assert raw == json.dumps(json.loads(raw), ensure_ascii=False, sort_keys=True, allow_nan=False)


def test_identical_rerun_is_safe_but_later_result_change_requires_review_again(store):
    base = _setup(store)
    _revise(store, base, formal=False)
    assert store.formal_performance(1)[0]["performance_status"] == "incomplete"
    first = _revise(store, base, net_return={"broker": 0.01, "paper": 0.01})
    store.confirm_outcome_revision(1, base["decision_date"], first, actor="a", reason="核对")
    second = _revise(store, base, net_return={"broker": 0.02, "paper": 0.01})
    result = store.formal_performance(1)[0]
    assert result["performance_status"] == "pending_review" and result["pending_revision_ids"] == [second]
    assert result["net_return"] is None
    with pytest.raises(ValueError, match="更新的修订"):
        store.confirm_outcome_revision(1, base["decision_date"], first, actor="a", reason="核对")


def test_unreconciled_original_result_is_not_published(store):
    store.record_day(_day("2025-03-03"))
    record = _day("2025-03-04", previous="2025-03-03", value=0.05)
    record["reconcile_status"] = "discrepancy"
    store.record_day(record)
    result = store.formal_performance(1)[0]
    assert result["performance_status"] == "unreconciled" and result["net_return"] is None


@pytest.mark.parametrize("field,value", [("broker", None), ("broker", float("nan")),
                                         ("broker", float("inf")), ("broker", True)])
def test_nonfinite_or_missing_original_broker_return_is_not_published(store, field, value):
    store.record_day(_day("2025-03-03"))
    day = _day("2025-03-04", previous="2025-03-03", value=value)
    store.record_day(day)
    result = store.formal_performance(1)[0]
    assert result["performance_status"] == "incomplete" and result["net_return"] is None
    assert "net_return.broker" in result["performance_missing"]


@pytest.mark.parametrize("fees", [None, float("nan"), float("inf"), True])
def test_unknown_or_invalid_fees_prevent_publishing_performance(store, fees):
    store.record_day(_day("2025-03-03"))
    day = _day("2025-03-04", previous="2025-03-03", value=0.01)
    day["costs"]["broker_fees"] = fees
    store.record_day(day)
    result = store.formal_performance(1)[0]
    assert result["performance_status"] == "incomplete" and result["net_return"] is None
    assert "costs.broker_fees" in result["performance_missing"]
    assert store.days(1)[1]["net_return"]["broker"] == 0.01


def test_fee_correction_requires_confirmation_before_performance_is_published(store):
    store.record_day(_day("2025-03-03"))
    day = _day("2025-03-04", previous="2025-03-03", value=0.01)
    day["costs"]["broker_fees"] = None
    store.record_day(day)
    assert store.formal_performance(1)[0]["performance_status"] == "incomplete"
    revision = _revise(store, day, costs={**day["costs"], "broker_fees": 1.25})
    assert store.formal_performance(1)[0]["performance_status"] == "pending_review"
    store.confirm_outcome_revision(1, day["decision_date"], revision, actor="reviewer", reason="核对券商费用")
    result = store.formal_performance(1)[0]
    assert result["performance_status"] == "confirmed" and result["costs"]["broker_fees"] == 1.25
    assert result["comparison_status"] == "requires_review"


def test_legacy_confirmation_without_new_evidence_keys_remains_readable(store):
    base = _setup(store)
    revision = _revise(store, base, net_return={"broker": 0.01, "paper": 0.01})
    confirmation = store.confirm_outcome_revision(1, base["decision_date"], revision, actor="a", reason="核对")
    with sqlite3.connect(store._engine.url.database) as con:  # noqa: SLF001
        raw = con.execute("SELECT payload FROM forward_day_events WHERE id = ?", (confirmation,)).fetchone()[0]
        payload = json.loads(raw)
        for key in ("benchmark", "exposure", "return_basis"):
            payload["evidence"].pop(key, None)
        con.execute("UPDATE forward_day_events SET payload = ? WHERE id = ?", (json.dumps(payload), confirmation))
    _revise(store, base, net_return={"broker": 0.01, "paper": 0.01})
    result = store.formal_performance(1)[0]
    assert result["performance_status"] == "confirmed"
    assert result["settlement_evidence"]["return_basis"] is None


@pytest.mark.parametrize("field", ["net_return", "costs", "drawdown", "excess_return"])
def test_invalid_original_performance_sections_are_not_published(store, field):
    store.record_day(_day("2025-03-03"))
    day = _day("2025-03-04", previous="2025-03-03", value=0.01)
    day[field]["unexpected_invalid_value"] = float("inf")
    store.record_day(day)
    result = store.formal_performance(1)[0]
    assert result["performance_status"] == "incomplete" and result["net_return"] is None
    assert "invalid_performance_fields" in result["performance_missing"]


@pytest.mark.parametrize("change", [{"costs": None}, {"drawdown": {"broker": None}},
                                   {"excess_return": {"label": None, "broker_vs_full_benchmark": None,
                                                      "paper_vs_full_benchmark": None, "paper_exposure_matched": None,
                                                      "broker_exposure_matched": None}}])
def test_missing_performance_section_or_label_is_incomplete(store, change):
    store.record_day(_day("2025-03-03"))
    day = _day("2025-03-04", previous="2025-03-03", value=0.01)
    day.update(change)
    store.record_day(day)
    result = store.formal_performance(1)[0]
    assert result["performance_status"] == "incomplete" and result["net_return"] is None
    assert "invalid_performance_fields" in result["performance_missing"]


def test_decision_execution_and_outcome_evidence_are_kept_separate_across_reruns(store):
    previous = _day("2025-03-03")
    previous["execution_summary"] = {"status": "submission_unknown", "reason": "待对账"}
    store.record_day(previous)
    revision_id = _revise(store, previous, execution_summary={"status": "ok", "scope": "submission_only"})
    outcome = _day("2025-03-04", previous="2025-03-03", value=0.01)
    outcome["execution_summary"] = {"status": "blocked", "reason": "next decision"}
    outcome["fills"] = [{"decision_date": "2025-03-03", "qty": 2}]
    store.record_day(outcome)
    result = store.formal_performance(1)[0]
    assert result["decision_execution"]["initial"]["status"] == "submission_unknown"
    assert result["decision_execution"]["changed_revision_ids"] == [revision_id]
    assert result["settlement_evidence"]["fills"] == outcome["fills"]
    assert result["comparison_status"] == "requires_review"


def test_a_predecision_attempt_is_not_an_execution_revision(store):
    attempt = _day("2025-03-03")
    attempt["decision"] = {"target_weights": None}
    attempt["target_weights"] = None
    attempt["execution_summary"] = {"status": "blocked"}
    store.record_day(attempt)
    previous = _day("2025-03-03")
    previous["execution_summary"] = {"status": "ok"}
    store.record_day(previous)
    store.record_day(_day("2025-03-04", previous="2025-03-03", value=0.01))
    assert store.history(1, "2025-03-03")[1]["kind"] == "attempt"
    assert store.formal_performance(1)[0]["decision_execution"]["changed_revision_ids"] == []


@pytest.mark.parametrize("field", ["benchmark", "exposure", "return_basis"])
def test_changed_valuation_evidence_requires_review(store, field):
    base = _setup(store)
    revision = _revise(store, base, **{field: {"changed": True}})
    result = store.formal_performance(1)[0]
    assert result["performance_status"] == "pending_review" and result["pending_revision_ids"] == [revision]


def test_changed_fills_require_review_even_when_returns_are_unchanged(store):
    base = _setup(store)
    revision = _revise(store, base, fills=[{"ticker": "AAA", "qty": 3}])
    result = store.formal_performance(1)[0]
    assert result["performance_status"] == "pending_review" and result["pending_revision_ids"] == [revision]
    store.confirm_outcome_revision(1, base["decision_date"], revision, actor="a", reason="核对成交")
    assert store.formal_performance(1)[0]["performance_status"] == "incomplete"


def test_confirmation_refuses_nonfinite_outcome_metadata(store):
    previous = _day("2025-03-03")
    base = _day("2025-03-04", previous="2025-03-03")
    base["outcome"]["diagnostic"] = float("inf")
    store.record_day(previous)
    store.record_day(base)
    rev = _revise(store, base, net_return={"broker": 0.01, "paper": 0.01})
    with pytest.raises(ValueError):
        store.confirm_outcome_revision(1, base["decision_date"], rev, actor="a", reason="核对")
    assert store.formal_performance(1)[0]["performance_status"] == "pending_review"


@pytest.mark.parametrize("changes", [
    {"spec_hash": "different"},
    {"outcome": {"of_decision": "2025-03-02", "counts": True}},
    {"reconcile_status": "discrepancy"},
    {"net_return": {"broker": float("nan")}},
])
def test_untrusted_revision_cannot_be_confirmed(store, changes):
    base = _setup(store)
    rev = _revise(store, base, **changes)
    with pytest.raises(ValueError):
        store.confirm_outcome_revision(1, base["decision_date"], rev, actor="a", reason="核对")
    assert store.formal_performance(1)[0]["performance_status"] == "pending_review"


def test_confirmation_requires_actor_reason_and_a_matching_revision(store):
    base = _setup(store)
    rev = _revise(store, base, net_return={"broker": 0.01, "paper": 0.01})
    for actor, reason, event_id in (("", "核对", rev), ("a", "", rev), ("a", "核对", rev + 100)):
        with pytest.raises(ValueError):
            store.confirm_outcome_revision(1, base["decision_date"], event_id, actor=actor, reason=reason)
    assert store.formal_performance(1)[0]["performance_status"] == "pending_review"


@pytest.mark.parametrize("changes", [
    {"formal": False},
    {"spec_hash": "other-spec"},
    {"decision": {"target_weights": None}, "target_weights": None},
])
def test_formal_outcome_requires_the_actual_previous_formal_decision_and_spec(store, changes):
    previous = _day("2025-03-03")
    previous.update(changes)
    base = _day("2025-03-04", previous="2025-03-03")
    store.record_day(previous)
    store.record_day(base)
    rev = _revise(store, base, net_return={"broker": 0.01, "paper": 0.01})
    assert store.formal_performance(1) == []
    with pytest.raises(ValueError):
        store.confirm_outcome_revision(1, base["decision_date"], rev, actor="a", reason="核对")


def test_confirmation_rejects_another_strategys_revision(store):
    base = _setup(store)
    other = _day("2025-03-04", sid=2, previous="2025-03-03")
    store.record_day(other)
    foreign = _revise(store, other, net_return={"broker": 0.02, "paper": 0.01})
    with pytest.raises(ValueError, match="找不到"):
        store.confirm_outcome_revision(1, base["decision_date"], foreign, actor="a", reason="核对")


def test_confirmation_requires_a_supported_transaction_lock(store, monkeypatch):
    base = _setup(store)
    rev = _revise(store, base, net_return={"broker": 0.01, "paper": 0.01})
    monkeypatch.setattr(store._engine.dialect, "name", "other")  # noqa: SLF001
    with pytest.raises(ValueError, match="SQLite"):
        store.confirm_outcome_revision(1, base["decision_date"], rev, actor="a", reason="核对")


def test_concurrent_new_revision_cannot_be_hidden_by_confirmation(store, monkeypatch):
    from app.db import forward_record
    base = _setup(store)
    rev = _revise(store, base, net_return={"broker": 0.01, "paper": 0.01})
    real_loads = json.loads
    entered, release, writer_started, writer_done = (threading.Event() for _ in range(4))
    errors = []

    def pause_during_review(value, *args, **kwargs):
        if isinstance(value, str) and '"broker": 0.01' in value and not entered.is_set():
            entered.set()
            if not release.wait(10):
                raise TimeoutError("confirmation test timed out")
        return real_loads(value, *args, **kwargs)

    def confirm():
        try:
            store.confirm_outcome_revision(1, base["decision_date"], rev, actor="a", reason="核对")
        except Exception as exc:
            errors.append(exc)

    def write_new_revision():
        writer_started.set()
        try:
            _revise(store, base, net_return={"broker": 0.02, "paper": 0.01})
        except Exception as exc:
            errors.append(exc)
        finally:
            writer_done.set()

    monkeypatch.setattr(forward_record.json, "loads", pause_during_review)
    reviewer = threading.Thread(target=confirm)
    writer = threading.Thread(target=write_new_revision)
    reviewer.start()
    try:
        assert entered.wait(10)
        writer.start()
        assert writer_started.wait(10)
        assert not writer_done.wait(0.3)
    finally:
        release.set()
        reviewer.join(10)
        if writer.ident is not None:
            writer.join(10)
    assert not reviewer.is_alive() and not writer.is_alive() and errors == []
    result = store.formal_performance(1)[0]
    assert result["performance_status"] == "pending_review" and result["net_return"] is None
