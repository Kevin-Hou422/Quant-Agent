"""
外部审查 2026-10-18 F07：审批、验证、内部模拟、实际执行不是同一个策略。

R8  批准 equal_weight / AUM 1234 / band .123；每日门仍按 ic_weighted 拟合，账本用 AUM 1,000,000 / band .00134。
R7  开目标波动后风险门调用两次，同一个缩放系数乘两遍（.8 → .4 → .2），券商拿 .4、模拟账本拿 .2。

修复后的契约（`executable_spec.py`）：提案时把影响交易的全部参数收进一份规格并算哈希；每日循环
照规格执行（成分 / 冻结权重 / 方法 / 无交易带），并核对其余参数与运行时一致，否则不调仓。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.core.portfolio_manager.executable_spec import (
    build_spec, risk_limits_from_settings, spec_diff, spec_hash, verify_against_runtime,
)
from tests.unit.portfolio_manager.test_audit_portfolio_regressions import _loop


def _ok_gate(seen):
    def ev(self, signals, panel, **kw):
        seen.update(method=self.method, **kw)
        return SimpleNamespace(passed=True, sharpe=1.0, deflated_sharpe=.99, t_stat=3.0,
                               reasons=[], to_dict=lambda: {"passed": True})
    return ev


def _activate(spec=None, tamper=False, **cfg):
    from app.db.strategy_store import StrategyConfig, StrategyStore
    verdict = {}
    if spec is not None:
        verdict = {"executable_spec": spec, "spec_hash": spec_hash(spec)}
        if tamper:
            verdict["executable_spec"] = dict(spec, method="ic_weighted")
    base = dict(factors=["1"], combo_weights={"1": 1.0}, aum=1234.0, method="equal_weight",
                no_trade_band=.123, status="active", verdict=verdict)
    base.update(cfg)
    return StrategyStore().save(StrategyConfig(**base))


def _spec_for(loop, method="equal_weight", band=.123, dsl="rank(close)"):
    return build_spec({"1": dsl}, {"1": 1.0}, method, float(loop.broker.initial_capital), band)


# ---------------------------------------------------------------------------
# R8：没有规格 / 规格对不上 → 不调仓
# ---------------------------------------------------------------------------

def test_the_review_r8_config_is_no_longer_traded(tmp_path, monkeypatch):
    """审查原样的配置（没有规格）：此前照常交易，且门 / AUM / 带都与批准的不同。"""
    seen = {}
    loop, ds, calls = _loop(tmp_path, monkeypatch, gate_block=False, evaluate=_ok_gate(seen))
    _activate()
    out = loop.run_portfolio(ds)
    assert out["reason"] == "active_spec_missing" and calls == ["maintain:active_spec_missing"]
    assert seen == {}                                         # 连门都没跑：没有可评估的"被批准的那份"


def test_with_a_matching_spec_the_approved_rule_is_what_runs(tmp_path, monkeypatch):
    seen = {}
    loop, ds, calls = _loop(tmp_path, monkeypatch, gate_block=False, evaluate=_ok_gate(seen))
    sid = _activate(_spec_for(loop))
    out = loop.run_portfolio(ds)
    assert out["active_config"] == sid
    assert seen["method"] == "equal_weight"                  # 此前 ic_weighted
    assert seen["fixed_combo_weights"] == {"1": 1.0} and seen["no_trade_band"] == .123
    assert out["no_trade_band"] == .123                       # 此前 settings / 数据推导的 .00134
    assert out["aum"] == loop.broker.initial_capital
    assert "exec" in calls


@pytest.mark.parametrize("setting,value,field", [
    ("risk_max_gross", 0.5, "risk_limits.max_gross"),
    ("risk_target_vol_ann", 0.12, "risk_limits.target_vol_ann"),
    ("trading_allow_short", True, "trading.allow_short"),
    ("price_source", "moomoo", "data.price_source"),
    ("paper_dataset", "other_universe", "data.dataset"),
])
def test_changing_a_trading_setting_after_approval_stops_trading(tmp_path, monkeypatch,
                                                                 setting, value, field):
    from app.config import settings
    loop, ds, calls = _loop(tmp_path, monkeypatch, gate_block=False, evaluate=_ok_gate({}))
    _activate(_spec_for(loop))
    monkeypatch.setattr(settings, setting, value)
    out = loop.run_portfolio(ds)
    assert out["reason"] == "active_spec_mismatch" and calls == ["maintain:active_spec_mismatch"]
    assert any(d.startswith(field + ":") for d in out["spec_diff"]), out["spec_diff"]


def test_a_different_aum_stops_trading(tmp_path, monkeypatch):
    loop, ds, calls = _loop(tmp_path, monkeypatch, gate_block=False, evaluate=_ok_gate({}))
    spec = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1234.0, .123)
    _activate(spec)
    out = loop.run_portfolio(ds)
    assert out["reason"] == "active_spec_mismatch"
    assert out["spec_diff"] == [f"aum: 批准时 1234.0 → 现在 {float(loop.broker.initial_capital)!r}"]


def test_a_component_whose_expression_changed_stops_trading(tmp_path, monkeypatch):
    loop, ds, calls = _loop(tmp_path, monkeypatch, gate_block=False, evaluate=_ok_gate({}))
    _activate(_spec_for(loop, dsl="rank(ts_delta(close, 5))"))
    out = loop.run_portfolio(ds)
    assert out["reason"] == "active_spec_mismatch" and out["spec_diff"][0].startswith("factors.1:")


def test_a_tampered_spec_counts_as_missing(tmp_path, monkeypatch):
    loop, ds, calls = _loop(tmp_path, monkeypatch, gate_block=False, evaluate=_ok_gate({}))
    _activate(_spec_for(loop), tamper=True)
    assert loop.run_portfolio(ds)["reason"] == "active_spec_missing"


# ---------------------------------------------------------------------------
# R7：波动缩放只乘一次
# ---------------------------------------------------------------------------

def test_vol_targeting_is_applied_once_in_the_daily_loop(tmp_path, monkeypatch):
    from app.config import settings
    from app.core.portfolio_manager.risk_gate import PortfolioRiskGate
    loop, ds, calls = _loop(tmp_path, monkeypatch, gate_block=False, evaluate=_ok_gate({}))
    monkeypatch.setattr(settings, "risk_target_vol_ann", 0.10)
    _activate(_spec_for(loop))
    vols, real = [], PortfolioRiskGate.apply

    def spy(self, w, sectors=None, port_vol_ann=None):
        vols.append(port_vol_ann)
        return real(self, w, sectors=sectors, port_vol_ann=port_vol_ann)
    monkeypatch.setattr(PortfolioRiskGate, "apply", spy)
    loop.run_portfolio(ds)
    assert len(vols) == 2 and vols[0] is not None and vols[0] > 0
    assert vols[1] is None, "无交易带之后的第二次风控又传了波动估计 —— 缩放会叠乘"


def test_the_review_r7_numbers_once_versus_twice():
    """单独调用 apply 两次仍会叠乘（apply 本身没变）；修复在调用点：第二次只做硬约束投影。"""
    import numpy as np
    import pandas as pd
    from app.core.portfolio_manager.risk_gate import PortfolioRiskGate, RiskLimits
    g = PortfolioRiskGate(RiskLimits(max_gross=1, max_net=1, max_name_weight=1,
                                     max_sector_weight=1, target_vol_ann=.1))
    w1, _ = g.apply(pd.DataFrame({"AAA": [.4], "BBB": [.4]}), port_vol_ann=.2)
    w2, _ = g.apply(w1, port_vol_ann=None)
    assert np.isclose(w1.sum(axis=1).iloc[0], .4) and np.isclose(w2.sum(axis=1).iloc[0], .4)


# ---------------------------------------------------------------------------
# 规格本身
# ---------------------------------------------------------------------------

def test_the_proposal_stores_a_spec_built_from_production_limits(monkeypatch):
    import numpy as np
    import pandas as pd
    from app.config import settings
    from app.core.portfolio_manager.strategy_builder import build_strategy_config
    monkeypatch.setattr(settings, "risk_max_gross", 0.7)
    monkeypatch.setattr(settings, "pm_no_trade_band", 0.02)
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-03", periods=200)
    cols = [f"S{i}" for i in range(8)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0004, 0.015, (200, 8)), 0), idx, cols)
    ds = {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
          "volume": pd.DataFrame(1e6, idx, cols), "returns": close.pct_change().fillna(0.0)}
    cfg = build_strategy_config({"7": close.pct_change().rank(axis=1)}, ds, aum=10_000,
                                factor_dsl={"7": "rank(returns)"})
    spec = cfg.verdict["executable_spec"]
    assert cfg.verdict["spec_hash"] == spec_hash(spec)
    assert spec["factors"] == {"7": "rank(returns)"} and spec["aum"] == 10_000.0
    assert spec["risk_limits"]["max_gross"] == 0.7               # 此前风控快照用 RiskLimits() 默认 1.0
    assert spec["no_trade_band"] == cfg.no_trade_band == 0.02
    assert spec["combo_weights"] == {k: round(v, 10) for k, v in cfg.combo_weights.items()}
    assert verify_against_runtime(spec, {"7": "rank(returns)"}, 10_000) == []


def test_the_hash_covers_every_trading_parameter():
    base = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e4, .01)
    variants = [
        build_spec({"1": "rank(open)"}, {"1": 1.0}, "equal_weight", 1e4, .01),
        build_spec({"1": "rank(close)"}, {"1": 0.5}, "equal_weight", 1e4, .01),
        build_spec({"1": "rank(close)"}, {"1": 1.0}, "ic_weighted", 1e4, .01),
        build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 2e4, .01),
        build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e4, .02),
    ]
    hashes = {spec_hash(base)} | {spec_hash(v) for v in variants}
    assert len(hashes) == 6
    assert spec_hash(base) == spec_hash(json.loads(json.dumps(base)))   # 序列化往返不改哈希
    assert len(spec_diff(base, variants[0])) == 1


def test_the_strategy_gate_evaluates_frozen_weights_without_refitting(monkeypatch):
    from app.core.portfolio_manager import strategy_gate as sg
    from app.core.portfolio_manager.manager import PortfolioManager
    from tests.conftest import _make_dataset
    seen = {}
    real = PortfolioManager.build_book

    def spy(self, signals, prices, volume, fixed_combo_weights=None, walk_forward_folds=None):
        seen.update(fixed=fixed_combo_weights, wf=walk_forward_folds)
        return real(self, signals, prices, volume, fixed_combo_weights=fixed_combo_weights,
                    walk_forward_folds=walk_forward_folds)
    monkeypatch.setattr(PortfolioManager, "build_book", spy)
    ds = _make_dataset(n_days=120, n_tickers=8)
    sigs = {"a": ds["close"].rank(axis=1), "b": -ds["close"].pct_change(5).rank(axis=1)}
    sg.strategy_net_returns(sigs, ds, aum=1e4, fixed_combo_weights={"a": .7, "b": .3},
                            walk_forward_folds=5, no_trade_band=0.0)
    assert seen["fixed"] == {"a": .7, "b": .3}


def test_preflight_reports_a_config_without_a_spec(tmp_path, monkeypatch):
    from app.config import settings
    from app.core.execution.golive import _active_strategy_problem
    from app.db.alpha_store import AlphaResult, AlphaStore
    url = f"sqlite:///{tmp_path / 'g.db'}"
    monkeypatch.setattr(settings, "database_url", url)
    aid = AlphaStore(db_url=url).save(AlphaResult(dsl="rank(close)", status="candidate"))
    st = AlphaStore(db_url=url)
    st.update_status(aid, "validated"); st.update_status(aid, "paper")
    _activate(factors=[str(aid)], combo_weights={str(aid): 1.0})
    ok, detail = _active_strategy_problem(settings)
    assert ok is False and "没有可执行规格" in detail
    spec = build_spec({str(aid): "rank(close)"}, {str(aid): 1.0}, "equal_weight",
                      float(settings.paper_aum), .123)
    _activate(spec, factors=[str(aid)], combo_weights={str(aid): 1.0})
    ok, detail = _active_strategy_problem(settings)
    assert ok is True and spec_hash(spec)[:12] in detail
    monkeypatch.setattr(settings, "risk_max_name_weight", 0.05)
    ok, detail = _active_strategy_problem(settings)
    assert ok is False and "risk_limits.max_name_weight" in detail

