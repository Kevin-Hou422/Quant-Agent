"""
冻结 Test 边界：只能由人登记（首次调用不写库、正式结论只认登记的边界）、登记后不随数据末日滑动、新数据落在 Test 一侧、曝光按区间重叠计次且改实验名不刷新。

来源（原测试文件的说明，保留出处）：

2026-10-22 用户决定 · 证据 / 研究纪律。

D12 原话："同意先只读查询，不授权写边界。使用明确的活库路径和真正只读连接，不调用会建表或迁移的 Store 构造函数。
列出已有边界、最早曝光日期、次数、股票池和实验身份，给出日期建议后由我确认。正式入口不要继续依赖首次调用偶然冻结；
改实验名不能刷新重叠数据的曝光额度。"

---

外部审查 2026-10-18 F03 / F04 / F05：研究隔离与试验记账。

F04  Test 起点此前 = 本次数据末日 − 2 年，数据每多一天就滑动一天，test_key 又是起止日期，
     于是每次都是"第 1 次使用"（R1：两个窗口重叠 521 天，uses 都是 1）。
     现在：真实数据集第一次切分时把 Test 起点（绝对日期）存进库；holdout 按区间重叠计次。
F03  提案、工作流、自主发现的观察引擎此前都看全量数据（R2、R3）；工作流每次自动汇报 Test 且不记账。
     现在：所有选择流程只看选择段；Test 只经显式、记账的出口。
F05  试验数此前只有自主发现按配置估算一次。现在 GP / Optuna 底层按实际评估数、按 run 幂等记账；
     台账读不到 → 验证门判不通过（证据不完整）。

---

复审 2026-10-19（`review-v6/audit-review/复审报告_2026-10-19.md`）N01–N09 的回归。

每条都从复审独立脚本（`independent_checks.py`）里对应的 residual_* 场景改写：原脚本断言"问题仍在"，
这里断言修复后应有的行为。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.data_partitioner import (
    FrozenBoundaryError,
    split_from_settings,
)
from app.db.trial_ledger import BoundaryStore
from tests.conftest import _make_dataset

REAL = "real_ds_d12"


def _panel(T=1300, N=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=T)
    cols = [f"S{i:02d}" for i in range(N)]
    q = np.linspace(-1, 1, N)
    c = pd.DataFrame(100 * np.cumprod(1 + 0.0008 * q[None, :] + rng.normal(0, 0.01, (T, N)), 0), idx, cols)
    o = c.shift(1).fillna(c)
    return {"open": o, "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * 0.998, "close": c,
            "vwap": c, "volume": pd.DataFrame(1e7, idx, cols), "returns": np.log(c / c.shift(1))}


@pytest.fixture
def ds():
    return _make_dataset(n_days=1600, n_tickers=10)


def _cut(d, start=None, end=None):
    return {k: v.loc[start:end] for k, v in d.items()}


def _pin(d, key="us_x"):
    """
    2026-10-22 D12：边界只能由人登记 —— 先临时切分（不写库），再把它的 Test 起点登记下来（F04 的"之后不再滑动"
    说的是登记之后）。返回登记的日期。
    """
    start = split_from_settings(d, dataset_key=key).test_start
    BoundaryStore().register(key, start, actor="tester", reason="F04 回归：登记一个冻结边界")
    return start


# ---------------------------------------------------------------------------
# D12-a：正式入口不依赖首次调用偶然冻结
# ---------------------------------------------------------------------------

class TestNoAccidentalFreeze:

    def test_the_first_research_call_does_not_pin_a_boundary(self, fresh_db):
        from app.core.data_engine.research_context import open_research
        from app.db.trial_ledger import BoundaryStore
        ctx = open_research(_panel(), REAL, "test.first_call")
        assert BoundaryStore().get(REAL) is None, "第一次研究调用把边界写死了（偶然冻结）"
        assert ctx.boundary == "provisional" and ctx.split is not None
        assert ctx.describe()["boundary"] == "provisional"

    def test_a_formal_validation_verdict_needs_a_registered_boundary(self, fresh_db):
        from app.core.lifecycle.validation_gate import ValidationGate
        res = ValidationGate(n_splits=3, embargo_days=5, dsr_threshold=0.0, min_tstat=0.0).evaluate(
            "rank(ts_mean(returns,20))", _panel(), n_trials=1, dataset_key=REAL)
        assert res.passed is False
        assert any("边界未登记" in r for r in res.reasons), res.reasons
        assert res.research_boundary == "provisional"
        # 对照：登记之后这条理由消失（上面的不通过不是信号本身的原因）
        from app.db.trial_ledger import BoundaryStore
        BoundaryStore().register(REAL, pd.Timestamp(res.partition["test_start"]), actor="Kevin-Hou422", reason="D12")
        res2 = ValidationGate(n_splits=3, embargo_days=5, dsr_threshold=0.0, min_tstat=0.0).evaluate(
            "rank(ts_mean(returns,20))", _panel(), n_trials=1, dataset_key=REAL)
        assert res2.research_boundary == "registered"
        assert not any("边界未登记" in r for r in res2.reasons), res2.reasons

    def test_a_registered_boundary_is_used_and_is_write_once(self, fresh_db):
        from app.core.data_engine.research_context import open_research
        from app.db.trial_ledger import BoundaryAlreadyRegistered, BoundaryStore
        bs = BoundaryStore()
        bs.register(REAL, pd.Timestamp("2024-06-03"), actor="Kevin-Hou422", reason="D12 确认的日期")
        ctx = open_research(_panel(), REAL, "test.registered")
        assert ctx.boundary == "registered" and ctx.test_start == "2024-06-03"
        with pytest.raises(BoundaryAlreadyRegistered):
            bs.register(REAL, pd.Timestamp("2025-01-02"), actor="x", reason="改日期")
        assert bs.get(REAL) == pd.Timestamp("2024-06-03")

    @pytest.mark.parametrize("actor,reason", [("", "理由"), ("someone", ""), ("  ", "  ")])
    def test_registration_needs_an_actor_and_a_reason(self, fresh_db, actor, reason):
        from app.db.trial_ledger import BoundaryStore
        with pytest.raises(ValueError):
            BoundaryStore().register(REAL, pd.Timestamp("2024-06-03"), actor=actor, reason=reason)
        assert BoundaryStore().get(REAL) is None

    def test_a_legacy_auto_pinned_boundary_still_freezes_but_is_not_formal(self, fresh_db):
        """旧代码首次调用写下的边界：Test 段照样冻结（不能解冻），但不算登记过 —— 正式结论仍要人确认。"""
        from app.core.data_engine.research_context import open_research
        from app.core.lifecycle.validation_gate import ValidationGate
        from app.db.trial_ledger import BoundaryStore
        BoundaryStore().pin(REAL, pd.Timestamp("2024-06-03"), frozen_by="years")
        ctx = open_research(_panel(), REAL, "test.legacy")
        assert ctx.boundary == "unconfirmed" and ctx.test_start == "2024-06-03"
        res = ValidationGate(n_splits=3, embargo_days=5, dsr_threshold=0.0, min_tstat=0.0).evaluate(
            "rank(ts_mean(returns,20))", _panel(), n_trials=1, dataset_key=REAL)
        assert res.passed is False and any("边界未登记" in r for r in res.reasons)

    def test_a_proposal_on_an_unregistered_boundary_is_not_passed_and_not_grade_a(self, fresh_db):
        from app.core.data_engine.research_context import open_research
        from app.core.portfolio_manager.strategy_builder import build_strategy_config
        ds = _panel(T=700)
        with open_research(ds, REAL, "proposal").active() as ctx:
            sig = {"f": ctx.selection["close"].pct_change(20).rank(axis=1)}
            cfg = build_strategy_config(sig, ctx.selection, aum=10_000.0, factor_dsl={"f": "rank(close)"})
        g = cfg.verdict["gates"]["research_boundary"]
        assert g["evaluated"] is True and g["passed"] is False and g["boundary"] == "provisional"
        assert cfg.passed is False and cfg.verdict["paper_grade"] != "A"

    def test_a_proposal_outside_any_research_context_has_an_unevaluated_boundary_gate(self, fresh_db):
        from app.core.portfolio_manager.strategy_builder import build_strategy_config
        ds = _panel(T=700)
        cfg = build_strategy_config({"f": ds["close"].pct_change(20).rank(axis=1)}, ds, aum=10_000.0,
                                    factor_dsl={"f": "rank(close)"})
        assert cfg.verdict["gates"]["research_boundary"]["evaluated"] is False
        assert cfg.passed is False and cfg.verdict["paper_grade"] != "A"

    def test_a_proposal_on_a_registered_boundary_passes_the_boundary_gate(self, fresh_db):
        from app.core.data_engine.research_context import open_research
        from app.core.portfolio_manager.strategy_builder import build_strategy_config
        from app.db.trial_ledger import BoundaryStore
        ds = _panel(T=700)
        BoundaryStore().register(REAL, ds["close"].index[-200], actor="Kevin-Hou422", reason="D12")
        with open_research(ds, REAL, "proposal").active() as ctx:
            cfg = build_strategy_config({"f": ctx.selection["close"].pct_change(20).rank(axis=1)},
                                        ctx.selection, aum=10_000.0, factor_dsl={"f": "rank(close)"})
        g = cfg.verdict["gates"]["research_boundary"]
        assert g["evaluated"] is True and g["passed"] is True and g["boundary"] == "registered"


# ---------------------------------------------------------------------------
# D12-b：改实验名不能刷新重叠数据的曝光额度
# ---------------------------------------------------------------------------

def test_renaming_the_experiment_does_not_refresh_the_exposure_quota(fresh_db):
    from app.core.data_engine.data_partitioner import account_holdout_use
    from app.core.data_engine.research_context import open_research
    from app.db.trial_ledger import BoundaryStore, HoldoutLedger
    ds = _panel()
    BoundaryStore().register(REAL, ds["close"].index[-300], actor="Kevin-Hou422", reason="D12")
    ctx_a = open_research(ds, REAL, "validation_gate", experiment="exp-A")
    with ctx_a.active():
        first = account_holdout_use(ctx_a.split, REAL, purpose="validation_gate", budget=1)
    ctx_b = open_research(_panel(T=1310), REAL, "validation_gate", experiment="exp-B-renamed")
    with ctx_b.active():
        second = account_holdout_use(ctx_b.split, REAL, purpose="validation_gate", budget=1)
    assert first["uses"] == 1 and second["uses"] == 2 and second["over_budget"] is True
    rows = HoldoutLedger().usages(REAL)
    assert [r["experiment"] for r in rows] == ["exp-A", "exp-B-renamed"]
    assert all(r["universe_digest"] and r["universe_n"] == 12 for r in rows)


# ===========================================================================
# F04：冻结边界
# ===========================================================================

class TestFrozenBoundary:

    def test_the_test_start_no_longer_slides_with_the_data_end(self, ds, fresh_db):
        a = split_from_settings(ds, dataset_key="us_x")
        assert a.frozen_by == "years" and a.boundary == "provisional"
        assert BoundaryStore().get("us_x") is None, "第一次切分不能把边界写死（D12）"
        BoundaryStore().register("us_x", a.test_start, actor="tester", reason="F04 回归")
        b = split_from_settings({k: v.iloc[:-1] for k, v in ds.items()}, dataset_key="us_x")
        c = split_from_settings({k: v.iloc[:-200] for k, v in ds.items()}, dataset_key="us_x")
        assert b.frozen_by == c.frozen_by == "pinned" and b.boundary == c.boundary == "registered"
        assert a.test_start == b.test_start == c.test_start
        assert a.selection["close"].index.equals(b.selection["close"].index)
        assert BoundaryStore().get("us_x") == a.test_start

    def test_the_unpinned_contrast_does_slide(self, ds, fresh_db):
        a = split_from_settings(ds)
        b = split_from_settings({k: v.iloc[:-200] for k, v in ds.items()})
        assert a.test_start != b.test_start

    def test_new_data_lands_on_the_test_side(self, ds, fresh_db):
        first = split_from_settings({k: v.iloc[:-100] for k, v in ds.items()}, dataset_key="us_x")
        BoundaryStore().register("us_x", first.test_start, actor="tester", reason="F04 回归")
        later = split_from_settings(ds, dataset_key="us_x")
        assert later.test_start == first.test_start
        assert later.n_test == first.n_test + 100
        assert later.selection["close"].index.equals(first.selection["close"].index)

    def test_data_entirely_before_the_boundary_has_no_test(self, ds, fresh_db):
        pinned = _pin(ds)
        early = split_from_settings(_cut(ds, end=pinned - pd.Timedelta(days=200)), dataset_key="us_x")
        assert early.n_test == 0 and early.test_key == "empty"
        assert all(v.empty for v in early.test.values())

    def test_data_that_starts_inside_the_frozen_segment_is_refused_not_widened(self, ds, fresh_db):
        pinned = _pin(ds)
        with pytest.raises(FrozenBoundaryError, match="冻结"):
            split_from_settings(_cut(ds, start=pinned - pd.Timedelta(days=30)), dataset_key="us_x")
        assert not issubclass(FrozenBoundaryError, ValueError)     # 各入口的"太短→全量"兜底接不住它

    @pytest.mark.parametrize("key", [None, "", "synthetic", "unknown"])
    def test_synthetic_keys_are_never_pinned(self, ds, fresh_db, key):
        split_from_settings(ds, dataset_key=key)
        assert BoundaryStore().get(key or "") is None

    def test_a_boundary_is_written_once(self, fresh_db):
        bs = BoundaryStore()
        assert bs.pin("k", pd.Timestamp("2024-01-02")) == pd.Timestamp("2024-01-02")
        assert bs.pin("k", pd.Timestamp("2025-06-02")) == pd.Timestamp("2024-01-02")

    def test_an_unreadable_boundary_store_refuses_instead_of_sliding(self, ds, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("db down")
        monkeypatch.setattr("app.db.trial_ledger.BoundaryStore", boom)
        with pytest.raises(FrozenBoundaryError, match="db down"):
            split_from_settings(ds, dataset_key="us_x")

    def test_backtest_entrypoints_refuse_rather_than_fall_back_to_the_full_panel(self, ds, fresh_db):
        from fastapi import HTTPException

        import app.api.router as router
        pinned = _pin(ds)
        with pytest.raises(HTTPException) as e:
            router._three_way(_cut(ds, start=pinned - pd.Timedelta(days=30)), "us_x", "backtest/run")
        assert e.value.status_code == 422

    def test_validation_gate_fails_closed_on_the_frozen_segment(self, ds, fresh_db):
        from app.core.lifecycle.validation_gate import ValidationGate
        pinned = _pin(ds)
        res = ValidationGate().evaluate("rank(close)", _cut(ds, start=pinned - pd.Timedelta(days=30)),
                                        n_trials=1, dataset_key="us_x")
        assert res.passed is False and res.reasons[0].startswith("冻结边界")
        assert res.n_folds == 0                                   # 没有退回全量去跑 WalkForward


class TestHoldoutExposureFollowsTheData:

    def test_a_sliding_window_still_counts_as_the_same_exposure(self, ds, fresh_db):
        from app.core.data_engine.data_partitioner import account_holdout_use
        a = split_from_settings(ds, dataset_key="us_x")
        b = split_from_settings({k: v.iloc[:-1] for k, v in ds.items()}, dataset_key="us_x")
        assert account_holdout_use(a, "us_x")["uses"] == 1
        second = account_holdout_use(b, "us_x")
        assert (second["uses"], second["over_budget"]) == (2, True)


def test_the_earliest_exposed_test_window_can_be_looked_up(fresh_db):
    from app.db.trial_ledger import HoldoutLedger
    led = HoldoutLedger()
    assert led.earliest_exposed_start("us_x") is None
    led.record_use("us_x", "2023-05-01..2025-05-01")
    led.record_use("us_x", "2022-11-03..2024-11-01")
    led.record_use("us_x", "empty")
    assert led.earliest_exposed_start("us_x") == pd.Timestamp("2022-11-03")
