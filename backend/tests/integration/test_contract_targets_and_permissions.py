"""
策略合同的目标生成、状态权限表、预登记验证设计（D5 关键路径变异复核：
app/core/portfolio_manager/strategy_contract.py、app/core/lifecycle/strategy_states.py、
app/core/lifecycle/validation_design.py）。

复核时存活的：
    · 合同：冻结合成权重时还去要配对收益（没有开盘价就会抛错）、目标波动关着也记"样本不足"、
      回撤熔断"全 0"改成除以 0、成分缺表达式仍算完整 —— 都没人发现
    · 权限表的 9 个格子：把某个状态的权限改掉（例如联调期计入前向证据、未知状态可交易）没人发现
    · 验证设计：三条拒绝路径的 `return False` 改成 True、"登记之后"的边界（登记当天不算）、
      登记前样本的计数、带时区的登记时刻 —— 都没人发现
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core.lifecycle import strategy_states as ss
from app.core.lifecycle import validation_design as vd
from app.core.portfolio_manager.risk_gate import RiskLimits
from app.core.portfolio_manager.strategy_contract import (
    StrategyContract,
    generate_targets,
)
from tests.conftest import _make_dataset


def _world(with_open=True):
    data = _make_dataset(n_days=120, n_tickers=8, seed=5)
    if not with_open:
        data = {k: v for k, v in data.items() if k != "open"}
    rng = np.random.default_rng(11)
    close = data["close"]
    signals = {f: pd.DataFrame(rng.normal(size=close.shape), index=close.index, columns=close.columns)
               for f in ("a", "b")}
    return data, signals


LIMITS = RiskLimits(max_gross=1.0, max_net=1.0, max_name_weight=0.5, max_sector_weight=1.0, target_vol_ann=None)


class TestGenerateTargets:

    def test_frozen_weights_need_no_paired_returns(self):
        """冻结了合成权重就不拟合 —— 不需要开盘价（要了就会因为没有开盘价而抛错）。"""
        data, signals = _world(with_open=False)
        out = generate_targets(signals, data, aum=1e6, method="ic_weighted", limits=LIMITS, band=0.0,
                               fixed_combo_weights={"a": 0.5, "b": 0.5})
        assert out.combo_weights == {"a": 0.5, "b": 0.5}

    def test_fitting_without_frozen_weights_uses_the_paired_returns(self):
        data, signals = _world(with_open=True)
        out = generate_targets(signals, data, aum=1e6, method="ic_weighted", limits=LIMITS, band=0.0)
        assert set(out.combo_weights) == {"a", "b"}
        data_no_open, _ = _world(with_open=False)
        with pytest.raises(ValueError, match="open"):
            generate_targets(signals, data_no_open, aum=1e6, method="ic_weighted", limits=LIMITS, band=0.0)

    def test_no_target_vol_means_no_skipped_note(self):
        data, signals = _world()
        out = generate_targets(signals, data, aum=1e6, method="equal_weight", limits=LIMITS, band=0.0)
        assert "target_vol_skipped_insufficient_sample" not in out.notes

    def test_a_drawdown_halt_flattens_to_exact_zero(self):
        data, signals = _world()
        out = generate_targets(signals, data, aum=1e6, method="equal_weight", limits=LIMITS, band=0.0, halt=True)
        assert (out.gated.to_numpy() == 0.0).all() and (out.weights.to_numpy() == 0.0).all()
        assert "drawdown_halt_flatten" in out.notes


class TestContract:

    def _contract(self, factors):
        return StrategyContract.freeze(factors, {k: 1.0 / len(factors) for k in factors}, "ic_weighted", 1e6, 0.0)

    def test_a_component_without_an_expression_is_incomplete(self):
        assert self._contract({"a": "rank(close)", "b": "rank(volume)"}).complete() is True
        assert self._contract({"a": "rank(close)", "b": None}).complete() is False

    def test_the_composite_with_frozen_weights_needs_no_open(self):
        data, signals = _world(with_open=False)
        comp = self._contract({"a": "rank(close)", "b": "rank(volume)"}).composite(signals, data)
        assert comp.shape == data["close"].shape and comp.notna().any().any()


class TestPermissionTable:

    def test_the_full_permission_matrix(self):
        keys = ("trades_simulate", "forward_evidence", "formal_days", "verified")
        expected = {
            "integration":     (True, False, False, False),
            "paper_running":   (True, True, True, False),
            "verified_active": (True, True, True, True),
        }
        for status, row in expected.items():
            assert ss.permissions(status) == dict(zip(keys, row)), status
        for status in ("active", "approved", "proposed", "retired", "rejected", "nonsense", ""):
            assert ss.permissions(status) == dict.fromkeys(keys, False), status

    def test_permissions_are_a_copy(self):
        p = ss.permissions("integration")
        p["forward_evidence"] = True
        assert ss.permissions("integration")["forward_evidence"] is False


T_REG = datetime(2026, 10, 20, 14, 0, tzinfo=timezone.utc)


def _record(design=None, tamper=False, verdict_text=None):
    design = design or vd.register_design(now=T_REG)
    h = vd.design_hash(design) if not tamper else "0" * 64
    text = verdict_text if verdict_text is not None else json.dumps(
        {"validation_design": design, "validation_design_hash": h})
    return SimpleNamespace(id=7, verdict=text)


def _rows(days, ic=0.05):
    return [SimpleNamespace(date=d, ic=ic, evidence_class="forward") for d in days]


class TestValidationDesign:

    def test_registration_time_is_stored_as_naive_utc(self):
        cst = timezone(timedelta(hours=8))
        d = vd.register_design(now=datetime(2026, 10, 22, 9, 0, tzinfo=cst))
        assert d["registered_at"] == "2026-10-22T01:00:00"
        assert vd.register_design(now=datetime(2026, 10, 22, 9, 0))["registered_at"] == "2026-10-22T09:00:00"

    @pytest.mark.parametrize("make", [
        lambda: _record(verdict_text="{not json"),
        lambda: _record(verdict_text=json.dumps({})),
        lambda: _record(tamper=True),
        lambda: _record(design=dict(vd.register_design(now=T_REG), clock_version="c0")),
    ], ids=["unparsable", "no_design", "tampered", "other_clock"])
    def test_every_refusal_is_a_refusal(self, make):
        ok, detail = vd.check_registered_activation(make(), _rows(pd.bdate_range("2026-10-21", periods=80)))
        assert ok is False and detail["passed"] is False and detail["reasons"]

    def test_evidence_on_the_registration_day_does_not_count(self):
        days = list(pd.bdate_range("2026-10-20", periods=4))           # 登记当天 + 之后 3 天
        ok, detail = vd.check_registered_activation(_record(), _rows(days))
        assert detail["n_before_registration"] == 1
        assert detail["n_days"] == 3
