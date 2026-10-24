"""
策略提案的最终结论：各必须门的原始结论保留，最终 passed / 评级 / 准入 / API 一致；门评估的就是冻结的规格。

来源（原测试文件的说明，保留出处）：

2026-10-22 用户决定 · 正确性与状态（`docs/DECISION_IMPLEMENTATION_TRACKER.md` C1 / C2 / C4 / C5 / C6）。

每条断言都对着用户的原话写；先跑、先看它们在旧代码上失败，再改。

---

复审 2026-10-19（`review-v6/audit-review/复审报告_2026-10-19.md`）N01–N09 的回归。

每条都从复审独立脚本（`independent_checks.py`）里对应的 residual_* 场景改写：原脚本断言"问题仍在"，
这里断言修复后应有的行为。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

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
    # 2026-10-22 D12：研究边界也是必须的门 —— 在合成数据的研究上下文里提案（没有可登记的边界，按 synthetic 放行）
    from app.core.data_engine.research_context import open_research
    with patch.object(StrategyGate, "evaluate", ev), open_research(ds, "", "proposal", three_way=False).active():
        return build_strategy_config({"1": ds["close"].rank(axis=1)}, ds, 10000, method="equal_weight",
                                     factor_dsl={"1": dsl})


class TestProposalVerdictConsistency:

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
# N05：提案的门评估的就是冻结规格；门与每日循环同一条风险路径
# ---------------------------------------------------------------------------

class TestProposalEvaluatesTheFrozenSpec:

    def test_the_gate_receives_the_frozen_weights_and_band(self):
        from app.core.portfolio_manager.strategy_builder import build_strategy_config
        from app.core.portfolio_manager.strategy_gate import StrategyGate
        ds = _make_dataset(80, 10)
        calls = []

        def ev(self, *a, **kw):
            calls.append(dict(kw))
            return SimpleNamespace(passed=True, sharpe=1., deflated_sharpe=.99, t_stat=3., reasons=[],
                                   to_dict=lambda: {"passed": True})
        with patch.object(StrategyGate, "evaluate", ev):
            cfg = build_strategy_config({"1": ds["close"].rank(axis=1)}, ds, 10000, method="equal_weight",
                                        factor_dsl={"1": "rank(close)"})
        frozen = calls[0]
        assert frozen["fixed_combo_weights"] == cfg.combo_weights
        assert frozen["no_trade_band"] == cfg.no_trade_band
        assert cfg.verdict["executable_spec"]["no_trade_band"] == pytest.approx(cfg.no_trade_band, abs=1e-10)
        assert "fixed_combo_weights" not in calls[1]                        # ② 流程的滚动前推
        assert cfg.verdict["evidence_scope"] == "frozen_spec_on_selection_in_sample"
        assert cfg.verdict["procedure_oos"]["scope"] == "walk_forward_refit_of_the_procedure"

    def test_failing_the_procedure_oos_fails_the_proposal(self):
        from app.core.portfolio_manager.strategy_builder import build_strategy_config
        from app.core.portfolio_manager.strategy_gate import StrategyGate
        ds = _make_dataset(80, 10)

        def ev(self, *a, **kw):
            ok = "fixed_combo_weights" in kw
            return SimpleNamespace(passed=ok, sharpe=1., deflated_sharpe=.99, t_stat=3.,
                                   reasons=[] if ok else ["DSR 不足"], to_dict=lambda: {"passed": ok})
        with patch.object(StrategyGate, "evaluate", ev):
            cfg = build_strategy_config({"1": ds["close"].rank(axis=1)}, ds, 10000,
                                        factor_dsl={"1": "rank(close)"})
        assert cfg.passed is False and any("流程的滚动前推 OOS 未过" in r for r in cfg.verdict["reasons"])

    def test_the_gate_applies_vol_targeting_and_the_post_band_projection(self, monkeypatch):
        import app.core.portfolio_manager.executable_spec as es
        from app.config import settings
        from app.core.portfolio_manager.risk_gate import PortfolioRiskGate
        from app.core.portfolio_manager.strategy_gate import strategy_net_returns
        monkeypatch.setattr(settings, "risk_target_vol_ann", 0.10)
        vols, projected = [], []
        real_apply, real_band = PortfolioRiskGate.apply, es.band_then_reproject

        def spy_apply(self, w, sectors=None, port_vol_ann=None):
            vols.append(port_vol_ann)
            return real_apply(self, w, sectors=sectors, port_vol_ann=port_vol_ann)

        def spy_band(*a, **k):
            projected.append(1)
            return real_band(*a, **k)
        monkeypatch.setattr(PortfolioRiskGate, "apply", spy_apply)
        monkeypatch.setattr(es, "band_then_reproject", spy_band)
        ds = _make_dataset(120, 8)
        strategy_net_returns({"a": ds["close"].rank(axis=1)}, ds, aum=1e4, no_trade_band=0.01)
        assert vols[0] is not None and vols[0] > 0, "门没有做和每日循环一样的目标波动缩放"
        assert projected == [1] and vols[1] is None                      # 带后硬约束投影、不再传波动
