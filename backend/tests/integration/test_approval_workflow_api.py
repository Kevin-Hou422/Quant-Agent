"""
test_approval_workflow_api.py — Phase 9.4 人工审批工作流验收

覆盖：
  - approve: VALIDATED→PAPER，写谱系；非 VALIDATED → 409；不存在 → 404
  - reject:  CANDIDATE/VALIDATED→RETIRED，带原因写谱系
  - pending: 列出 VALIDATED 待审批队列
  - decisions: 返回审批谱系（append-only）
"""

from __future__ import annotations

import pytest

from app.db.alpha_store import AlphaStore, AlphaResult


@pytest.fixture()
def store_and_client(test_client, tmp_path):
    from app.api.router import get_store
    from app.main import app
    store = AlphaStore(db_url=f"sqlite:///{tmp_path/'appr.db'}")
    app.dependency_overrides[get_store] = lambda: store
    yield store, test_client
    app.dependency_overrides.pop(get_store, None)


def _validated(store: AlphaStore, dsl="rank(close)") -> int:
    aid = store.save(AlphaResult(dsl=dsl))          # candidate
    store.update_status(aid, "validated")
    return aid


def test_approve_promotes_to_paper_and_records(store_and_client):
    store, client = store_and_client
    aid = _validated(store)
    r = client.post(f"/api/alphas/{aid}/approve", json={"reason": "looks robust"})
    assert r.status_code == 200, r.text
    assert r.json()["new_status"] == "paper"
    assert store.get_by_id(aid).status == "paper"
    decs = store.get_decisions(aid)
    assert len(decs) == 1 and decs[0].decision == "approve"
    assert decs[0].to_status == "paper" and decs[0].reason == "looks robust"


def test_approve_requires_validated(store_and_client):
    store, client = store_and_client
    aid = store.save(AlphaResult(dsl="rank(close)"))   # candidate, 未验证
    r = client.post(f"/api/alphas/{aid}/approve")
    assert r.status_code == 409


def test_approve_404(store_and_client):
    _, client = store_and_client
    assert client.post("/api/alphas/999999/approve").status_code == 404


def test_reject_retires_with_reason(store_and_client):
    store, client = store_and_client
    aid = _validated(store)
    r = client.post(f"/api/alphas/{aid}/reject", json={"reason": "crowded factor"})
    assert r.status_code == 200
    assert r.json()["new_status"] == "retired"
    assert store.get_by_id(aid).status == "retired"
    decs = store.get_decisions(aid)
    assert decs[-1].decision == "reject" and decs[-1].reason == "crowded factor"


def test_reject_candidate_allowed(store_and_client):
    store, client = store_and_client
    aid = store.save(AlphaResult(dsl="rank(close)"))   # candidate
    r = client.post(f"/api/alphas/{aid}/reject", json={"reason": "no economic logic"})
    assert r.status_code == 200
    assert store.get_by_id(aid).status == "retired"


def test_pending_lists_validated(store_and_client):
    store, client = store_and_client
    v1 = _validated(store, "rank(ts_delta(close,5))")
    _cand = store.save(AlphaResult(dsl="rank(volume)"))   # candidate, 不应出现
    r = client.get("/api/alphas/pending")
    assert r.status_code == 200
    ids = {row["alpha_id"] for row in r.json()}
    assert v1 in ids and _cand not in ids


def test_decisions_endpoint_returns_lineage(store_and_client):
    store, client = store_and_client
    aid = _validated(store)
    client.post(f"/api/alphas/{aid}/approve", json={"reason": "ok"})
    r = client.get(f"/api/alphas/{aid}/decisions")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1 and body[0]["decision"] == "approve"


# ---------------------------------------------------------------------------
# FE-10：待批准队列里标出因子用到的基本面字段与可见规则
# ---------------------------------------------------------------------------

def test_pending_marks_the_fundamental_fields_a_candidate_uses(store_and_client):
    store, client = store_and_client
    fund = _validated(store, "fund_rank(earnings_yield) + fund_rank(trade_when(days_since_earnings < 60, sue))")
    price = _validated(store, "rank(ts_delta(close,5))")
    rows = {row["alpha_id"]: row for row in client.get("/api/alphas/pending").json()}
    assert [f["name"] for f in rows[fund]["sparse_fields"]] == ["earnings_yield", "sue", "days_since_earnings"]
    assert [f["class"] for f in rows[fund]["sparse_fields"]] == ["valuation", "quarterly", "event"]
    assert all(f["note"] for f in rows[fund]["sparse_fields"])
    assert "第一个交易日" in rows[fund]["visibility_rule"] and "200" in rows[fund]["visibility_rule"]
    assert rows[price]["sparse_fields"] == [] and rows[price]["visibility_rule"] is None


def test_pending_survives_an_unparsable_expression_in_the_ledger(store_and_client, caplog):
    """库里有一条读不回来的表达式（点变异缺陷修复之前可能入库）不能让整个审批队列打不开。"""
    import logging
    store, client = store_and_client
    bad = _validated(store, "rank(ts_min(close,volume,10))")
    ok = _validated(store, "fund_rank(roe)")
    with caplog.at_level(logging.WARNING):
        r = client.get("/api/alphas/pending")
    assert r.status_code == 200
    rows = {row["alpha_id"]: row for row in r.json()}
    assert rows[bad]["sparse_fields"] == [] and rows[bad]["dsl"] == "rank(ts_min(close,volume,10))"
    assert [f["name"] for f in rows[ok]["sparse_fields"]] == ["roe"]
    assert any("解析失败" in m for m in caplog.messages)
