"""
已批准规格按批准时冻结的表达式校验政策运行（2026-10-25 审计的决定）。

原话："按批准时冻结的校验政策运行；不要每天强行改用最新研究规则，但也不同意永久、无条件跳过所有校验。……补'旧政策可兼容
继续''新提案按新政策拒绝''规格篡改/未知版本/安全不兼容拒绝'等测试，再处理那一个变异存活点"（strategy_contract L185
`Executor(validate=False)`）。"不要偷偷截断 300 为 252。"
"""

from __future__ import annotations

import pytest

from app.core.alpha_engine import validation_policy as vp
from app.core.portfolio_manager.strategy_contract import (
    SpecValidationRefused,
    StrategyContract,
)
from tests.conftest import _make_dataset

WIDE = {"max_window": 504, "max_depth": 10, "sparse_min_window": 126}
DSL_300 = "rank(ts_mean(close, 300))"


@pytest.fixture
def wide_then_tightened(monkeypatch):
    """一版更宽的旧政策（窗口上限 504）先是 CURRENT；之后"收紧"回 v1（252）。"""
    monkeypatch.setitem(vp.POLICIES, "wide", dict(WIDE))
    monkeypatch.setattr(vp, "CURRENT", "wide")

    def tighten():
        monkeypatch.setattr(vp, "CURRENT", "v1")
    return tighten


def _freeze(dsl):
    return StrategyContract.freeze({"1": dsl}, {"1": 1.0}, "ic_weighted", 10_000.0, 0.0)


def _data():
    return _make_dataset(n_days=420, n_tickers=8, seed=4)


def test_a_new_proposal_is_checked_under_the_current_policy():
    c = _freeze(DSL_300)
    v = c.spec["validation"]
    assert v["policy_version"] == "v1" and v["passed"] is False
    assert "300" in " ".join(v["results"]["1"]["errors"])
    with pytest.raises(SpecValidationRefused, match="不通过"):
        c.signals(_data())


def test_an_old_policy_keeps_working_after_the_rules_tighten(wide_then_tightened):
    """批准时按 'wide' 通过的 300 日窗口：规则收紧之后照旧按 'wide' 解释、照常生成信号（不截成 252、不报错）。"""
    c = _freeze(DSL_300)
    assert c.spec["validation"]["policy_version"] == "wide" and c.spec["validation"]["passed"] is True
    wide_then_tightened()
    sig = c.signals(_data())["1"]
    assert sig.iloc[-1].notna().sum() > 0
    assert _freeze(DSL_300).spec["validation"]["passed"] is False, "新提案按收紧后的政策拒绝"


def test_the_policy_and_operators_are_bound_to_the_spec_hash():
    a, b = _freeze("rank(ts_mean(close, 20))"), _freeze("rank(ts_mean(close, 20))")
    assert a.hash == b.hash
    b.spec["validation"]["policy_version"] = "v0"
    from app.core.portfolio_manager.executable_spec import spec_hash
    assert spec_hash(b.spec) != a.hash, "校验凭据在规格里：改了它哈希就变"


def test_a_policy_edited_after_approval_is_refused(wide_then_tightened, monkeypatch):
    c = _freeze(DSL_300)
    monkeypatch.setitem(vp.POLICIES, "wide", {**WIDE, "max_window": 600})        # 已登记的版本被改了内容
    assert any("摘要不一致" in p for p in vp.check_frozen(c.spec))


def test_an_unknown_policy_version_is_refused():
    c = _freeze("rank(ts_mean(close, 20))")
    c.spec["validation"]["policy_version"] = "v9"
    assert vp.check_frozen(c.spec) == ["规格的校验政策版本 'v9' 不认识"]


def test_a_changed_operator_set_is_refused(monkeypatch):
    c = _freeze("rank(ts_mean(close, 20))")
    monkeypatch.setattr(vp, "operators_digest", lambda: "different")
    assert any("算子集" in p for p in vp.check_frozen(c.spec))


@pytest.mark.parametrize("dsl,needle", [
    ("rank(future_close)", "future_"), ("rank(ts_mean(close, 2000))", "安全上限"), ("rank(no_such_op(close, 5))", "解析失败")])
def test_the_safety_floor_holds_under_any_policy(monkeypatch, dsl, needle):
    """安全底线与政策版本无关：哪怕某一版政策允许，读未来信息 / 超资源上限 / 不支持的算子照样拒绝。"""
    monkeypatch.setitem(vp.POLICIES, "lax", {"max_window": 5000, "max_depth": 50, "sparse_min_window": 1})
    monkeypatch.setattr(vp, "CURRENT", "lax")
    rec = vp.validation_record({"1": dsl})
    assert rec["passed"] is False and needle in " ".join(rec["results"]["1"]["errors"])


def test_a_legacy_spec_without_credentials_is_not_grandfathered():
    c = _freeze("rank(ts_mean(close, 20))")
    del c.spec["validation"]
    with pytest.raises(SpecValidationRefused, match="没有表达式校验凭据"):
        c.signals(_data())


def test_approval_refuses_a_spec_that_fails_its_frozen_policy(fresh_db):
    import json

    from fastapi import HTTPException

    from app.api.router import strategy_approve
    from app.db.strategy_store import StrategyConfig, StrategyStore
    st = StrategyStore(db_url=fresh_db)
    c = _freeze(DSL_300)
    sid = st.save(StrategyConfig(factors=["1"], combo_weights={"1": 1.0}, aum=1e4,
                                 verdict={"executable_spec": c.spec, "spec_hash": c.hash}, status="proposed"))
    from types import SimpleNamespace
    with pytest.raises(HTTPException) as e:
        strategy_approve(sid, SimpleNamespace(reason="r", actor="x", activate=False), sstore=st, store=None)
    assert e.value.status_code == 409 and "表达式校验不通过" in e.value.detail
    assert str(st.get(sid).status) == "proposed"
    assert json.loads(st.get(sid).verdict)["spec_hash"] == c.hash
