"""
test_agent_critic.py — OverfitCritic 阈值逻辑测试

覆盖：优质 alpha 通过、过拟合失败、低 OOS Sharpe 失败、
无 OOS 数据处理、高换手率失败、高回撤失败、弱信号失败。
"""
from __future__ import annotations

import pytest

from app.agent._critic import OverfitCritic, CriticResult
from app.agent._constants import _MIN_OOS_SHARPE, _OVERFIT_THRESHOLD


class TestOverfitCriticPass:

    def test_good_alpha_passes(self):
        """优质 alpha 应通过所有检查。"""
        result = OverfitCritic.check({
            "oos_sharpe":        0.8,
            "is_sharpe":         1.0,
            "overfitting_score": 0.2,
            "is_overfit":        False,
            "is_turnover":       1.0,
            "max_drawdown":     -0.10,
        })
        assert result.passed is True
        assert result.failure_mode == ""
        assert result.severity == "ok"
        assert result.reason != ""

    def test_critic_result_iter(self):
        """CriticResult 应支持 passed, reason = critic.check(m) 解包。"""
        result = OverfitCritic.check({
            "oos_sharpe": 0.9,
            "is_sharpe": 1.1,
            "overfitting_score": 0.15,
        })
        passed, reason = result
        assert isinstance(passed, bool)
        assert isinstance(reason, str)


class TestOverfitCriticNoSignal:

    def test_oos_sharpe_near_zero_fails(self):
        """OOS Sharpe 接近 0（< 0.05）应触发 no_signal 失败。"""
        result = OverfitCritic.check({
            "oos_sharpe": 0.01,
            "is_sharpe":  1.5,
        })
        assert result.passed is False
        assert result.failure_mode == "no_signal"
        assert result.severity == "critical"

    def test_negative_oos_sharpe_fails(self):
        result = OverfitCritic.check({"oos_sharpe": -0.5, "is_sharpe": 1.0})
        assert result.passed is False


class TestOverfitCriticSevereOverfit:

    def test_is_overfit_flag_fails(self):
        """is_overfit=True 且 IS Sharpe 较高时触发 severe_overfitting。"""
        result = OverfitCritic.check({
            "oos_sharpe":        0.3,
            "is_sharpe":         2.0,
            "overfitting_score": 0.85,
            "is_overfit":        True,
        })
        assert result.passed is False
        assert result.failure_mode in ("severe_overfitting", "mild_overfitting")

    def test_high_overfit_score_fails(self):
        result = OverfitCritic.check({
            "oos_sharpe":        0.3,
            "is_sharpe":         2.0,
            "overfitting_score": 0.70,
        })
        assert result.passed is False


class TestOverfitCriticHighTurnover:

    def test_turnover_above_threshold_fails(self):
        """年化换手率 > 3× 应触发 high_turnover 失败。"""
        result = OverfitCritic.check({
            "oos_sharpe":  0.8,
            "is_sharpe":   1.0,
            "is_turnover": 4.0,
        })
        assert result.passed is False
        assert result.failure_mode == "high_turnover"


class TestOverfitCriticHighDrawdown:

    def test_drawdown_above_30pct_fails(self):
        """最大回撤 > 30% 应触发 high_drawdown 失败。"""
        result = OverfitCritic.check({
            "oos_sharpe":   0.8,
            "is_sharpe":    1.0,
            "max_drawdown": -0.35,
        })
        assert result.passed is False
        assert result.failure_mode == "high_drawdown"


class TestOverfitCriticNoOOS:

    def test_none_oos_sharpe_treated_as_zero(self):
        """oos_sharpe=None 应被当作 0.0 处理（不应崩溃）。"""
        result = OverfitCritic.check({
            "oos_sharpe": None,
            "is_sharpe":  1.0,
        })
        assert isinstance(result.passed, bool)
        assert isinstance(result.reason, str)


class TestOverfitCriticWeakSignal:

    def test_weak_oos_sharpe_below_threshold_fails(self):
        """OOS Sharpe 处于弱信号区间（0.1 < oos < 0.35）应触发 weak_signal。"""
        result = OverfitCritic.check({
            "oos_sharpe":        0.2,
            "is_sharpe":         0.4,
            "overfitting_score": 0.1,
            "is_overfit":        False,
            "is_turnover":       0.8,
            "max_drawdown":     -0.05,
        })
        assert result.passed is False
        assert result.failure_mode in ("weak_signal", "mild_overfitting")


class TestCriticResultStructure:

    def test_all_fields_present(self):
        result = OverfitCritic.check({"oos_sharpe": 0.8})
        assert hasattr(result, "passed")
        assert hasattr(result, "failure_mode")
        assert hasattr(result, "severity")
        assert hasattr(result, "recommended_mutation")
        assert hasattr(result, "reason")
        assert hasattr(result, "metrics_snapshot")

    def test_metrics_snapshot_has_oos_sharpe(self):
        result = OverfitCritic.check({"oos_sharpe": 0.75})
        assert "oos_sharpe" in result.metrics_snapshot
