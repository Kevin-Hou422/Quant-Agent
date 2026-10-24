"""
基本面因子本轮不正式晋级：所有晋级路径都拦；验证门只给选择段诊断；预登记诊断记进试验史且不碰 Test。

来源（原测试文件的说明，保留出处）：

2026-10-22 用户决定 · 证据 / 研究纪律。

D12 原话："同意先只读查询，不授权写边界。使用明确的活库路径和真正只读连接，不调用会建表或迁移的 Store 构造函数。
列出已有边界、最早曝光日期、次数、股票池和实验身份，给出日期建议后由我确认。正式入口不要继续依赖首次调用偶然冻结；
改实验名不能刷新重叠数据的曝光额度。"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


# ---------------------------------------------------------------------------
# D1："基本面暂不正式晋级。允许选择段内预登记诊断和试验记账，不自动消耗最终 Test。"
# ---------------------------------------------------------------------------

def _fund_panel():
    ds = _panel(T=1300)
    rng = np.random.default_rng(1)
    ds["roe"] = pd.DataFrame(rng.normal(0.1, 0.05, ds["close"].shape), ds["close"].index, ds["close"].columns)
    ds["earnings_yield"] = ds["roe"] * 0.5
    return ds


class TestFundamentalsAreNotPromoted:

    @pytest.mark.parametrize("to", ["validated"])
    def test_a_fundamental_factor_cannot_be_promoted_by_any_path(self, fresh_db, to):
        from app.db.alpha_store import AlphaResult, AlphaStore
        s = AlphaStore()
        fund = s.save(AlphaResult(dsl="rank(earnings_yield)", status="candidate"))
        price = s.save(AlphaResult(dsl="rank(ts_delta(close, 5))", status="candidate"))
        with pytest.raises(ValueError, match="D1"):
            s.update_status(fund, to)
        assert s.get_by_id(fund).status == "candidate"
        assert s.update_status(price, to).status == to, "价量因子不受影响"

    def test_the_status_api_refuses_it_too(self, fresh_db, test_client):
        from app.db.alpha_store import AlphaResult, AlphaStore
        aid = AlphaStore().save(AlphaResult(dsl="rank(roe)", status="candidate"))
        r = test_client.patch(f"/api/alphas/{aid}/status", json={"status": "validated"})
        assert r.status_code == 422 and "D1" in r.text

    def test_the_validation_gate_gives_a_selection_only_diagnostic_never_a_pass(self, fresh_db):
        from app.core.lifecycle.validation_gate import ValidationGate
        from app.db.trial_ledger import BoundaryStore, HoldoutLedger
        ds = _fund_panel()
        BoundaryStore().register(REAL, ds["close"].index[-300], actor="Kevin-Hou422", reason="D12")
        res = ValidationGate(n_splits=3, embargo_days=5, dsr_threshold=0.0, min_tstat=-99).evaluate(
            "rank(roe)", ds, n_trials=1, dataset_key=REAL)
        assert res.passed is False and res.diagnostic_only is True
        assert any("D1" in r for r in res.reasons)
        assert res.n_folds >= 3, "诊断指标照样要在选择段上算出来"
        assert res.held_out_test is None and HoldoutLedger().usages(REAL) == []

    def test_pre_registered_diagnostics_are_counted_as_trials_and_never_touch_test(self, fresh_db):
        from app.core.lifecycle.fundamentals_policy import run_fundamental_diagnostics
        from app.db.trial_ledger import BoundaryStore, HoldoutLedger, TrialLedger
        ds = _fund_panel()
        BoundaryStore().register(REAL, ds["close"].index[-300], actor="Kevin-Hou422", reason="D12")
        before = TrialLedger().total()
        out = run_fundamental_diagnostics(["rank(roe)", "rank(earnings_yield)"], ds, REAL,
                                          registered_by="Kevin-Hou422", note="D1 预登记诊断")
        assert TrialLedger().total() == before + 2, "诊断没有记进试验史"
        assert out["registration"]["n"] == 2 and len(out["registration"]["digest"]) == 64
        assert all(r["diagnostic_only"] and r["promotable"] is False for r in out["results"])
        assert HoldoutLedger().usages(REAL) == [], "诊断动用了冻结 Test 段"
        runs = [r for r in TrialLedger().run_states() if r.source == "fund_diag"]
        assert len(runs) == 1 and runs[0].status == "done" and runs[0].n_recorded == 2

    def test_registration_needs_a_person_and_a_purpose(self, fresh_db):
        from app.core.lifecycle.fundamentals_policy import run_fundamental_diagnostics
        with pytest.raises(ValueError):
            run_fundamental_diagnostics(["rank(roe)"], _fund_panel(), REAL, registered_by="", note="x")

    def test_a_strategy_using_fundamentals_is_not_passed(self, fresh_db):
        from app.core.data_engine.research_context import open_research
        from app.core.portfolio_manager.strategy_builder import build_strategy_config
        ds = _panel(T=700)
        with open_research(ds, "", "proposal").active() as ctx:
            cfg = build_strategy_config({"f": ctx.selection["close"].rank(axis=1)}, ctx.selection, aum=1e4,
                                        factor_dsl={"f": "rank(earnings_yield)"})
        g = cfg.verdict["gates"]["fundamentals_policy"]
        assert g["evaluated"] is True and g["passed"] is False and "D1" in g["detail"]
        assert cfg.passed is False


def test_saving_a_fundamental_factor_directly_in_a_promoted_status_is_refused(fresh_db):
    """D1：直接以 validated / paper 入库也是一条晋级路径，同样拦住（候选状态照常入库）。"""
    from app.db.alpha_store import AlphaResult, AlphaStore
    s = AlphaStore()
    for st in ("validated", "paper", "active"):
        with pytest.raises(ValueError, match="D1"):
            s.save(AlphaResult(dsl="rank(roe)", status=st))
    assert s.save(AlphaResult(dsl="rank(roe)", status="candidate")) > 0
