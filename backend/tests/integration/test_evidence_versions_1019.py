"""
复审 2026-10-19 余项（不依赖待决事项的部分）：

§5.1  "尚未统一时钟前，历史结果应分版本标识。"
      → 每一类存下来的结果（研究指标、因子 IC、信号存档、策略证据、规格、提案 verdict、验证门结论、
        正式前向记录）都带时钟版本；时钟一变，规格哈希跟着变（旧证据不会被新时钟的策略继承）。
N03-b "补拉、回放、晚到归档应保留研究价值，但不能升级成正式前向样本。"
      §7.1 工作包 4："将影子信号证据与有实际成交的策略收益分开记录。"
      → 策略证据分 forward / replay 两类：没有及时存档的那几天也记下来（replay，可供研究），
        →ACTIVE 门只读 forward。
N05-d "新规格的资格应来自预登记的验证设计和此后真正未见的前向数据。"
      → 提案时登记验证设计（门槛、t 统计口径、证据口径、登记时刻，带哈希）；→ACTIVE 按**登记的**
        设计判定（之后改 settings 不影响），只算登记之后的前向证据；设计缺失 / 被改 → 不通过。
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tests.integration.test_forward_evidence_1018 import _after_close, _panel, _upto, world  # noqa: F401


def _propose(world, dsl="rank(ts_delta(close, 5))"):
    from app.config import settings
    from app.db.strategy_store import StrategyConfig, StrategyStore
    from tests.conftest import approved_verdict
    f = str(world.aid)
    v = approved_verdict({f: dsl}, {f: 1.0}, 10_000.0)
    return StrategyStore(db_url=settings.database_url).save(
        StrategyConfig(factors=[f], combo_weights={f: 1.0}, aum=10_000.0, status="proposed",
                       verdict=v)), v["spec_hash"]


# ---------------------------------------------------------------------------
# §5.1 时钟版本
# ---------------------------------------------------------------------------

class TestClockVersion:

    def test_research_metrics_and_factor_ic_carry_the_clock(self, world):
        from app.core.data_engine.input_identity import CLOCK_VERSION
        rec = world.store.get_by_id(world.aid)
        assert rec.clock_version == CLOCK_VERSION
        world.store.record_ic(world.aid, "2024-05-02", 0.1, is_forward=True)
        assert world.store.get_ic_history(world.aid)[-1].clock_version == CLOCK_VERSION

    def test_signal_archives_and_strategy_evidence_carry_the_clock(self, world, monkeypatch):
        from app.config import settings
        from app.core.data_engine.input_identity import CLOCK_VERSION
        monkeypatch.setattr(settings, "database_url", world.url)
        sid, _ = _propose(world)
        ds = _panel()
        days = ds["close"].index
        for d in days[-3:]:                                         # 时钟 c2：信号日 T 在 T+2 结算
            world.loop().run_portfolio(_upto(ds, d), aum=10_000.0, data_source="yahoo", now=_after_close(d))
        rows = world.ev.strategy_forward_ic(sid)
        assert rows and {r.clock_version for r in rows} == {CLOCK_VERSION}
        from sqlalchemy import select
        from app.db.evidence_store import SignalArchive
        with world.ev._Session() as s:                                         # noqa: SLF001
            arch = list(s.scalars(select(SignalArchive)).all())
        assert arch and {a.clock_version for a in arch} == {CLOCK_VERSION}

    def test_the_spec_binds_the_clock_so_a_new_clock_is_a_new_strategy(self, monkeypatch):
        import app.core.data_engine.input_identity as ii
        from app.core.portfolio_manager.executable_spec import build_spec, spec_hash, verify_against_runtime
        a = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e4, 0.01)
        assert a["clock"] == ii.CLOCK_VERSION
        monkeypatch.setattr(ii, "CLOCK_VERSION", "v2-test")
        b = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e4, 0.01)
        assert spec_hash(a) != spec_hash(b)
        assert any("clock" in d for d in verify_against_runtime(a, {"1": "rank(close)"}, 1e4))

    def test_verdicts_and_gate_results_carry_the_clock(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from app.core.data_engine.input_identity import CLOCK_VERSION
        from app.core.lifecycle.validation_gate import ValidationGate
        from app.core.portfolio_manager.strategy_builder import build_strategy_config
        from app.core.portfolio_manager.strategy_gate import StrategyGate
        from tests.conftest import _make_dataset
        ds = _make_dataset(80, 10)

        def ev(self, *a, **kw):
            return SimpleNamespace(passed=True, sharpe=1., deflated_sharpe=.99, t_stat=3., reasons=[],
                                   to_dict=lambda: {"passed": True})
        with patch.object(StrategyGate, "evaluate", ev):
            cfg = build_strategy_config({"1": ds["close"].rank(axis=1)}, ds, 10000, method="equal_weight",
                                        factor_dsl={"1": "rank(close)"})
        assert cfg.verdict["clock_version"] == CLOCK_VERSION
        res = ValidationGate(n_splits=2).evaluate("rank(close)", ds, n_trials=1)
        assert res.to_dict()["clock_version"] == CLOCK_VERSION


# ---------------------------------------------------------------------------
# N03-b：回放保留研究价值，但不进正式前向
# ---------------------------------------------------------------------------

class TestReplayIsKeptButNotForward:

    def test_one_run_over_history_keeps_replay_rows_and_no_forward_rows(self, world, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "database_url", world.url)
        sid, _ = _propose(world)
        ds = _panel()
        world.loop().run_portfolio(ds, aum=10_000.0, data_source="yahoo", now=_after_close(ds["close"].index[-1]))
        assert world.ev.strategy_forward_ic(sid) == [], "没有及时存档的那几天被算成了前向"
        rep = world.ev.strategy_replay_ic(sid)
        assert rep, "回放那几天的 IC 没有留下（研究价值丢了）"
        assert {r.evidence_class for r in rep} == {"replay"}

    def test_day_by_day_rows_are_forward_and_the_gate_ignores_replay(self, world, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "database_url", world.url)
        sid, h = _propose(world)
        ds = _panel()
        days = ds["close"].index
        world.loop().run_portfolio(_upto(ds, days[-4]), aum=10_000.0, data_source="yahoo", now=_after_close(days[-4]))
        for d in days[-3:]:
            world.loop().run_portfolio(_upto(ds, d), aum=10_000.0, data_source="yahoo", now=_after_close(d))
        fwd = world.ev.strategy_forward_ic(sid, h)
        # 时钟 c2：days[-2] 结算 days[-4] 的信号，days[-1] 结算 days[-3] 的（都在标签起点之前存档）
        assert [pd.Timestamp(r.date) for r in fwd] == list(days[-2:])
        assert {r.evidence_class for r in fwd} == {"forward"}
        assert world.ev.strategy_replay_ic(sid), "首跑那天之前的回放行没有留下"


# ---------------------------------------------------------------------------
# N05-d：预登记的验证设计
# ---------------------------------------------------------------------------

class TestPreRegisteredDesign:

    def _proposal(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from app.core.portfolio_manager.strategy_builder import build_strategy_config
        from app.core.portfolio_manager.strategy_gate import StrategyGate
        from tests.conftest import _make_dataset
        ds = _make_dataset(80, 10)

        def ev(self, *a, **kw):
            return SimpleNamespace(passed=True, sharpe=1., deflated_sharpe=.99, t_stat=3., reasons=[],
                                   to_dict=lambda: {"passed": True})
        with patch.object(StrategyGate, "evaluate", ev):
            return build_strategy_config({"1": ds["close"].rank(axis=1)}, ds, 10000, method="equal_weight",
                                         factor_dsl={"1": "rank(close)"})

    def test_the_proposal_registers_a_design_with_a_hash(self):
        from app.core.lifecycle.validation_design import design_hash
        cfg = self._proposal()
        d = cfg.verdict["validation_design"]
        assert d["registered_at"] and d["min_forward_days"] and "min_ic_tstat" in d
        assert d["evidence"] == "strategy_evidence.forward, same spec_hash, date after registration"
        assert cfg.verdict["validation_design_hash"] == design_hash(d)

    def _rec(self, verdict):
        from types import SimpleNamespace
        return SimpleNamespace(id=7, verdict=json.dumps(verdict))

    def _ev(self, n, start="2024-01-02", ic=0.05, step=0.01):
        from types import SimpleNamespace
        days = pd.bdate_range(start, periods=n)
        return [SimpleNamespace(date=d.date(), ic=ic + step * (i % 3), evidence_class="forward")
                for i, d in enumerate(days)]

    def test_the_gate_uses_the_registered_thresholds_not_todays_settings(self, monkeypatch):
        from app.config import settings
        from app.core.lifecycle.validation_design import check_registered_activation
        cfg = self._proposal()
        v = dict(cfg.verdict)
        v["validation_design"] = dict(v["validation_design"], registered_at="2023-12-01T00:00:00")
        from app.core.lifecycle.validation_design import design_hash
        v["validation_design_hash"] = design_hash(v["validation_design"])
        need = int(v["validation_design"]["min_forward_days"])
        monkeypatch.setattr(settings, "tr_min_forward_days", 5)          # 事后把门放松
        ok, det = check_registered_activation(self._rec(v), self._ev(need - 1))
        assert ok is False and det["min_forward_days"] == need, "门用了事后放松的 settings，不是预登记的设计"

    def test_only_evidence_after_registration_counts(self):
        from app.core.lifecycle.validation_design import check_registered_activation, design_hash
        cfg = self._proposal()
        v = dict(cfg.verdict)
        need = int(v["validation_design"]["min_forward_days"])
        v["validation_design"] = dict(v["validation_design"], registered_at="2024-06-01T00:00:00")
        v["validation_design_hash"] = design_hash(v["validation_design"])
        before = self._ev(need + 10, start="2024-01-02")                  # 全在登记之前
        ok, det = check_registered_activation(self._rec(v), before)
        assert ok is False and det["n_days"] == 0, "登记之前的数据被当成了此后未见的前向数据"

    def test_a_missing_or_tampered_design_does_not_pass(self):
        from app.core.lifecycle.validation_design import check_registered_activation
        cfg = self._proposal()
        v = dict(cfg.verdict)
        v.pop("validation_design")
        ok, det = check_registered_activation(self._rec(v), self._ev(200))
        assert ok is False and any("验证设计" in r for r in det["reasons"])
        v = dict(cfg.verdict)
        v["validation_design"] = dict(v["validation_design"], min_ic_tstat=-99.0)   # 改了设计，哈希没改
        ok, det = check_registered_activation(self._rec(v), self._ev(200))
        assert ok is False and any("哈希" in r for r in det["reasons"])

    def test_the_activation_endpoint_uses_the_registered_design(self):
        """→ACTIVE 端点走预登记设计这条路（结构检查：不再直接拿 settings 阈值判）。"""
        import inspect
        import app.api.router as router
        src = inspect.getsource(router._active_gate)
        assert "check_registered_activation" in src
