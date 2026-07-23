"""
test_supplementary_fixes.py — 补充审计（SUPPLEMENTARY_AUDIT_FINDINGS，2026-07-24 核实）回归测试

逐项实测结论与对应守卫：
  S1  不属实 —— _tdays 动态推断行为正确（真实美股历≈252 / 无节假日B历≈261 / 加密≈365）
                 本文件以三组断言固化**预期行为**，防止未来"修复"它反而引入真 bug
  S2  属实已修 —— CS→CS 嵌套合法（rank(winsorize(x)) 等可解析可求值）
  S3  属实已修 —— AlphaExecutor 移出 __all__ + 实例化触发 DeprecationWarning
  S4  属实已修 —— 中文假设命中正确 family；多标签合并
  S5  属实已修 —— 量纲失稳因子（rank(x)*volume）受结构惩罚；归一化因子不受罚
  S8  属实已修 —— (N,) 一维 groups 通过 _validate_dataset（与 Task 3.4 文档一致）
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# S1 — _tdays 预期行为固化（报告不属实：无节假日 B 历的真实年频率就是 ~261）
# ===========================================================================

class TestS1TdaysIntendedBehavior:

    @staticmethod
    def _pa(idx):
        from app.core.backtest_engine.performance_analyzer import PerformanceAnalyzer
        from app.core.backtest_engine.backtest_engine import BacktestResult
        r = pd.Series(0.0008, index=idx)
        res = BacktestResult(
            (1 + r).cumprod(), r, r, pd.DataFrame(index=idx), pd.DataFrame(),
            pd.Series(0.1, index=idx), pd.DataFrame(index=idx),
            pd.Series(0.0, index=idx),
        )
        return PerformanceAnalyzer(res, rf=0.0)

    def test_us_holiday_calendar_gives_252(self):
        """真实美股日历（含节假日）→ 年化系数 ≈ 252。这是 E3 的设计目标。"""
        from pandas.tseries.holiday import USFederalHolidayCalendar
        from pandas.tseries.offsets import CustomBusinessDay
        cbd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        idx = pd.date_range("2020-01-02", periods=250, freq=cbd)
        assert abs(self._pa(idx)._tdays - 252) < 4

    def test_no_holiday_b_freq_gives_261(self):
        """freq='B'（无节假日）的真实年频率是 365.25×5/7≈261，不是 252。

        补充审计报告 S1 以此为 bug 是基准选错：该日历下 ~261 才是正确值。
        """
        idx = pd.date_range("2020-01-01", periods=252, freq="B")
        assert abs(self._pa(idx)._tdays - 365.25 * 5 / 7) < 3

    def test_crypto_daily_gives_365(self):
        idx = pd.date_range("2023-01-01", periods=365, freq="D")
        assert abs(self._pa(idx)._tdays - 365.25) < 3


# ===========================================================================
# S2 — CS→CS 嵌套（属实已修）
# ===========================================================================

class TestS2CsNesting:

    @pytest.fixture
    def parser(self):
        from app.core.alpha_engine.parser import Parser
        return Parser()

    @pytest.mark.parametrize("expr", [
        "rank(winsorize(returns, 3))",
        "ind_neutralize(rank(close))",
        "winsorize(zscore(returns), 3)",
    ])
    def test_reasonable_nesting_parses(self, parser, expr):
        node = parser.parse(expr)
        assert node is not None

    def test_nested_cs_evaluates(self, parser, make_dataset):
        """不仅可解析，还能对真实形状数据求值出 (T,N) 面板。"""
        ds  = make_dataset(n_days=60, n_tickers=8)
        from app.core.alpha_engine.dsl_executor import Executor
        out = Executor(validate=False).run(parser.parse("rank(winsorize(returns, 3))"), ds)
        assert out.shape == ds["close"].shape
        # rank 输出应在 [0,1] 分位区间
        valid = out.to_numpy()[~np.isnan(out.to_numpy())]
        assert valid.min() >= 0.0 and valid.max() <= 1.0


# ===========================================================================
# S3 — AlphaExecutor 弃用保护（属实已修）
# ===========================================================================

class TestS3LegacyExecutor:

    def test_not_in_public_all(self):
        import app.core.alpha_engine as ae
        assert "AlphaExecutor" not in ae.__all__
        assert "execute_alpha" not in ae.__all__
        assert "batch_execute" not in ae.__all__

    def test_instantiation_warns(self):
        from app.core.alpha_engine.executor import AlphaExecutor
        with pytest.warns(DeprecationWarning, match="dsl_executor"):
            AlphaExecutor()


# ===========================================================================
# S4 — 中文假设匹配（属实已修）
# ===========================================================================

class TestS4ChineseHypothesis:

    @staticmethod
    def _family_of(text: str):
        from app.core.workflows.alpha_workflows import (
            _hypothesis_templates, _HYPOTHESIS_TEMPLATES,
        )
        t = _hypothesis_templates(text)
        return t, _HYPOTHESIS_TEMPLATES

    @pytest.mark.parametrize("text,family", [
        ("动量",       "momentum"),
        ("趋势突破",   "momentum"),
        ("均值回归",   "reversion"),
        ("反转",       "reversion"),
        ("波动率",     "volatility"),
        ("成交量放大", "volume"),
        ("momentum",  "momentum"),
        ("reversion", "reversion"),
    ])
    def test_keyword_hits_family(self, text, family):
        t, all_t = self._family_of(text)
        assert set(all_t[family]).issubset(set(t)), f"'{text}' 未命中 {family}"

    def test_multi_label_merges(self):
        """'low volatility momentum' 应合并 momentum + volatility 两族种子。"""
        t, all_t = self._family_of("low volatility momentum")
        assert set(all_t["momentum"]).issubset(set(t))
        assert set(all_t["volatility"]).issubset(set(t))

    def test_unknown_falls_back_to_default(self):
        t, all_t = self._family_of("完全无关的字符串 xyz")
        assert t == all_t["default"]


# ===========================================================================
# S5 — 量纲稳定性结构惩罚（属实已修）
# ===========================================================================

class TestS5ScalePenalty:

    @staticmethod
    def _penalty(dsl: str) -> float:
        from app.core.alpha_engine.parser import Parser
        from app.core.gp_engine.fitness import scale_stability_penalty
        return scale_stability_penalty(Parser().parse(dsl))

    @pytest.mark.parametrize("dsl", [
        "rank(ts_delta(close, 5))",              # CS 归一化根
        "zscore(ts_mean(returns, 10))",
        "rank(ts_mean(close, 20)) * -1",         # 归一化 × 标量（反转模板）
        "ts_corr(close, volume, 10)",            # corr 天然有界
        "rank(close) - rank(volume)",            # 两个归一化的差
        "sign(ts_delta(close, 5))",              # ±1 输出
    ])
    def test_stable_factors_no_penalty(self, dsl):
        assert self._penalty(dsl) == 0.0, dsl

    @pytest.mark.parametrize("dsl", [
        "rank(ts_delta(log(close),10)) * volume",   # 报告 S5 的实测案例
        "close",                                    # 裸价格量纲
        "ts_mean(volume, 20)",                      # 裸成交量量纲
        "rank(close) / ts_std(close, 10)",          # 分母非标量的除法
    ])
    def test_unstable_factors_penalised(self, dsl):
        from app.core.gp_engine.fitness import SCALE_PENALTY
        assert self._penalty(dsl) == SCALE_PENALTY, dsl


# ===========================================================================
# S8 — (N,) groups 通过数据验证（属实已修）
# ===========================================================================

class TestS8AuxGroupValidation:

    def test_1d_groups_accepted(self, make_dataset):
        from app.core.backtest_engine.realistic_backtester import (
            RealisticBacktester, _validate_dataset,
        )
        from app.core.alpha_engine.signal_processor import SimulationConfig

        ds = make_dataset(n_days=80, n_tickers=6)
        ds["groups"] = np.array([0, 0, 1, 1, 2, 2])          # Task 3.4 文档格式
        _validate_dataset(ds, label="test")                   # 不抛即通过

        result = RealisticBacktester(config=SimulationConfig()).run(
            "sector_neutral(ts_delta(close, 5))", ds,
        )
        assert result.is_report is not None

    def test_wrong_length_1d_groups_rejected(self, make_dataset):
        from app.core.backtest_engine.realistic_backtester import _validate_dataset
        ds = make_dataset(n_days=80, n_tickers=6)
        ds["groups"] = np.array([0, 1, 2])                   # 长度 ≠ N
        with pytest.raises(ValueError, match="groups"):
            _validate_dataset(ds, label="test")

    def test_normal_fields_still_strict(self, make_dataset):
        """豁免只针对辅助分组字段——普通字段仍要求 (T,N) DataFrame。"""
        from app.core.backtest_engine.realistic_backtester import _validate_dataset
        ds = make_dataset(n_days=80, n_tickers=6)
        ds["volume"] = np.ones(6)                            # 非法：普通字段给 ndarray
        with pytest.raises(ValueError, match="volume"):
            _validate_dataset(ds, label="test")
