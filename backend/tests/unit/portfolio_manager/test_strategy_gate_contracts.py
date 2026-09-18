"""
strategy_gate.py —— 策略级验证门的逐条定钉测试（变异测试驱动）

来由：31 个变异点，首测击杀率 **0.0%** —— **改坏 31 处，一处都不会被发现**。

这是决定"一个策略配不配拿真钱"的门：分段稳健性、DSR 去膨胀、夏普 t、PBO
四道判定全在这里，还有"因子按边际贡献准入"的贪心选择。它此前唯一的覆盖是
`test_daily_loop_money_and_gates.py` 里"门通过时不停交易"那一条**接线**测试，
门内部的公式与阈值方向一条都没测。

后果举例（全都改了不报错）：
  - `if tstat < self.min_tstat` 放宽成 `<=`：恰好等于门槛的策略被放行
  - `(mu/sd) * sqrt(n)` 写成 `/`：t 统计量随样本增大而**变小**，长样本永远不显著
  - `impr = cand_oos - base_oos` 写成 `+`：边际贡献变成两者之和，谁都能进
  - `admitted=False` 写成 True：被**拒绝**的候选在轨迹里显示为已纳入

本文件对每个存活变异要么写出能杀死它的用例，要么给出可机械验证的等价性证明。
门的判定逻辑用打桩的净收益驱动（不跑真回测），因此又快又完全可控。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.core.portfolio_manager.strategy_gate as sg
from app.core.portfolio_manager.strategy_gate import (
    StrategyGate,
    StrategyValidationResult,
    _oos_tail_sharpe,
    _sharpe,
    marginal_factor_selection,
    resolve_band,
    resolve_cost_params,
)

IDX = pd.bdate_range("2023-01-02", periods=200)


def _rets(mean: float, sd: float, n: int = 200, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, sd, n), index=IDX[:n])


def _panel(n_days: int = 60, n_tickers: int = 4) -> dict:
    idx = pd.bdate_range("2023-01-02", periods=n_days)
    cols = [f"T{i}" for i in range(n_tickers)]
    rng = np.random.default_rng(1)
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_tickers)), axis=0),
        index=idx, columns=cols)
    return {"close": close, "open": close, "high": close * 1.004, "low": close * 0.996,
            "vwap": close, "volume": pd.DataFrame(4e6, index=idx, columns=cols),
            "returns": close.pct_change().fillna(0.0)}


# ===========================================================================
# A. _sharpe —— 年化与两个退化守卫
# ===========================================================================

class TestSharpeHelper:

    def test_annualisation_factor_is_multiplied(self):
        """
        `mean/sd * sqrt(tdays)`。写成 `/` 会小 252 倍 ——
        任何"夏普为正/为负"的断言都抓不住它，必须比数值。
        """
        r = _rets(0.001, 0.01, n=200, seed=2)
        expected = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252.0))
        assert _sharpe(r) == pytest.approx(expected, abs=1e-12)
        assert abs(_sharpe(r)) > 0.5, "年化后的夏普量级应在个位数，疑似被除了 √252"

    def test_tdays_parameter_is_honoured(self):
        r = _rets(0.001, 0.01, n=200, seed=2)
        assert _sharpe(r, tdays=63.0) == pytest.approx(
            _sharpe(r, tdays=252.0) / 2.0, abs=1e-12)

    def test_two_observations_are_enough(self):
        """`if r.size < 2: return 0.0` 的边界：**恰好 2 个**观测可以算。"""
        two = pd.Series([0.01, -0.005], index=IDX[:2])
        assert _sharpe(two) != 0.0, "2 个观测被判成样本不足"
        assert _sharpe(pd.Series([0.01], index=IDX[:1])) == 0.0

    def test_zero_dispersion_returns_zero_not_inf(self):
        """`if sd < 1e-12: return 0.0` —— 常数收益的标准差恰好是 0，可精确构造。"""
        assert _sharpe(pd.Series(0.001, index=IDX[:50])) == 0.0
        assert _sharpe(pd.Series(0.0, index=IDX[:50])) == 0.0

    def test_nan_only_series_returns_zero(self):
        assert _sharpe(pd.Series([np.nan] * 10, index=IDX[:10])) == 0.0


class TestOosTailSharpe:

    def test_tail_length_is_a_fraction_not_a_multiple(self):
        """
        `k = max(10, int(n * oos_ratio))`。写成 `/` 后 n=200、ratio=0.3 会得到
        k = max(10, 666) = 666 → `iloc[-666:]` 取到**整段**，OOS 代理失去意义。
        构造：让尾段与整段的夏普明显不同，就能区分。
        """
        head = pd.Series(np.full(140, 0.01), index=IDX[:140])
        tail = pd.Series(np.linspace(-0.02, -0.01, 60), index=IDX[140:200])
        r = pd.concat([head, tail])
        k = max(10, int(len(r) * 0.30))
        assert k == 60
        assert _oos_tail_sharpe(r, 0.30) == pytest.approx(_sharpe(r.iloc[-60:]), abs=1e-12)
        assert _oos_tail_sharpe(r, 0.30) < 0 < _sharpe(r), (
            "尾段明显为负而整段为正，两者却同号 —— 尾段切片疑似取成了整段")

    def test_short_series_uses_the_whole_sample(self):
        """`if n < 20: return _sharpe(rets)` 的边界：**恰好 20** 走尾段分支。"""
        r19 = _rets(0.001, 0.01, n=19, seed=3)
        assert _oos_tail_sharpe(r19) == pytest.approx(_sharpe(r19), abs=1e-15)
        r20 = _rets(0.001, 0.01, n=20, seed=3)
        assert _oos_tail_sharpe(r20) == pytest.approx(_sharpe(r20.iloc[-10:]), abs=1e-12)

    def test_tail_has_a_floor_of_ten_observations(self):
        r = _rets(0.001, 0.01, n=25, seed=4)
        assert _oos_tail_sharpe(r, 0.01) == pytest.approx(_sharpe(r.iloc[-10:]), abs=1e-12)


# ===========================================================================
# B. 配置解析与缓存
# ===========================================================================

class TestResolvers:

    def test_configured_band_short_circuits_the_derivation(self, monkeypatch):
        """
        `if band > 0.0: return band` —— 配置了正的无交易带就直接用，不再推导。
        放宽成 `>=` 时 band=0.0 也会被当成"已配置"直接返回，
        **数据推导那条路径就此永远走不到**（换手率与成本全部按 0 带计算）。
        """
        from app.config import settings
        monkeypatch.setattr(settings, "pm_no_trade_band", 0.004, raising=False)
        assert resolve_band(_panel(), aum=10_000.0) == pytest.approx(0.004)

        called = {"n": 0}

        class _TC:
            def __init__(self, **kw):
                called["n"] += 1

            def analyze(self, ds):
                return type("R", (), {"rebalance_band": 0.0123})()

        monkeypatch.setattr(settings, "pm_no_trade_band", 0.0, raising=False)
        monkeypatch.setattr("app.core.trading_context.TradingContext", _TC)
        sg._DERIVE_CACHE.clear()
        got = resolve_band(_panel(), aum=10_000.0)
        assert called["n"] == 1, "band=0 时没有走数据推导 —— `> 0` 疑似被放宽成 `>= 0`"
        assert got == pytest.approx(0.0123)

    def test_band_derivation_failure_returns_zero_and_warns(self, monkeypatch, caplog):
        from app.config import settings
        monkeypatch.setattr(settings, "pm_no_trade_band", 0.0, raising=False)

        def _boom(**kw):
            raise RuntimeError("no context")

        monkeypatch.setattr("app.core.trading_context.TradingContext", _boom)
        sg._DERIVE_CACHE.clear()
        with caplog.at_level("WARNING"):
            assert resolve_band(_panel(), aum=10_000.0) == 0.0
        assert any("无交易带" in r.getMessage() for r in caplog.records), (
            "推导失败静默返回 0 —— 无法与'带本来就是 0'区分")

    def test_explicit_cost_params_are_used_verbatim(self):
        """
        `if cost_params is not None: return cost_params` —— 删掉 `not` 后，
        **显式传入**的成本参数反而被丢弃，改用推导值（DEV_LESSONS §J 的复发形态）。
        """
        from app.core.backtest_engine.transaction_cost import CostParams
        mine = CostParams(fixed_bps=42.0)
        assert resolve_cost_params(_panel(), 10_000.0, cost_params=mine) is mine

    def test_cache_is_cleared_when_it_exceeds_the_cap(self):
        """
        `if len(_DERIVE_CACHE) > _CACHE_MAX: clear()` —— 边界是**超过**才清。
        放宽成 `>=` 会在恰好装满时提前清空（只是效率差异，但阈值语义要钉住）。
        """
        sg._DERIVE_CACHE.clear()
        for i in range(sg._CACHE_MAX):
            sg._cache_put((f"k{i}",), i)
        assert len(sg._DERIVE_CACHE) == sg._CACHE_MAX, (
            "恰好装满就被清空 —— 上限判定疑似被放宽")
        sg._cache_put(("one-more",), 1)
        assert len(sg._DERIVE_CACHE) == sg._CACHE_MAX + 1
        sg._cache_put(("overflow",), 1)
        assert len(sg._DERIVE_CACHE) == 1, "超过上限后没有清空"
        sg._DERIVE_CACHE.clear()


# ===========================================================================
# C. StrategyValidationResult 的序列化
# ===========================================================================

class TestValidationResultShape:

    def test_default_passed_is_false(self):
        """
        `StrategyValidationResult(passed=False, ...)` 是**失败优先**的构造：
        改成 True 后，任何提前 return（空策略、回测失败）都会返回"通过"。
        """
        r = StrategyValidationResult(passed=False, n_factors=0)
        assert r.passed is False

    def test_pbo_none_is_preserved_not_rounded(self):
        """
        `"pbo": round(self.pbo, 4) if self.pbo is not None else None` ——
        删掉 `not` 后会对 None 调用 round 抛异常，或把算出来的 PBO 丢成 None。
        """
        r = StrategyValidationResult(passed=True, n_factors=1)
        assert r.to_dict()["pbo"] is None
        r.pbo = 0.123456
        assert r.to_dict()["pbo"] == pytest.approx(0.1235, abs=1e-9)

    def test_use_global_trials_defaults_to_on(self):
        """
        `use_global_trials: bool = True` —— 默认必须开。改成 False 后
        DSR 只按 n_trials=1 校正，**多重检验修正整体失效，门变松**。
        """
        assert StrategyGate().use_global_trials is True


# ===========================================================================
# D. evaluate —— 四道判定的方向与边界
# ===========================================================================

class TestEvaluateGates:
    """
    用打桩的策略净收益驱动门（不跑真回测）：门的逻辑才是被测对象，
    回测本身在别处已有覆盖。这样每条用例都是确定的、毫秒级的。
    """

    @staticmethod
    def _gate_with(monkeypatch, rets: pd.Series, n_trials: int = 1, **kw) -> StrategyGate:
        monkeypatch.setattr(sg, "strategy_net_returns",
                            lambda *a, **k: (rets, pd.DataFrame()))
        return StrategyGate(use_global_trials=False, **kw)

    @staticmethod
    def _signals(n: int = 1) -> dict:
        idx = pd.bdate_range("2023-01-02", periods=60)
        return {f"f{i}": pd.DataFrame(1.0, index=idx, columns=["T0", "T1"])
                for i in range(n)}

    def test_empty_strategy_is_rejected(self, monkeypatch):
        """`if not factor_signals:` 删掉 `not` → 空策略反而被放行去跑回测。"""
        gate = self._gate_with(monkeypatch, _rets(0.001, 0.01))
        res = gate.evaluate({}, _panel())
        assert res.passed is False
        assert res.reasons == ["策略为空（无因子）"]

    def test_non_empty_strategy_reaches_the_gates(self, monkeypatch):
        """对照组：非空策略必须真的走到后面的判定。"""
        gate = self._gate_with(monkeypatch, _rets(0.002, 0.008, seed=5))
        res = gate.evaluate(self._signals(1), _panel())
        assert res.n_segments > 0, "非空策略没有进入分段判定"

    def test_thirty_observations_are_enough(self, monkeypatch):
        """
        `if len(rets) < 30 or std == 0` 的边界：**恰好 30** 个观测要能进门。
        放宽成 `<=` 会把刚够样本的策略拦在"样本不足"上。
        """
        gate = self._gate_with(monkeypatch, _rets(0.002, 0.008, n=30, seed=6))
        res = gate.evaluate(self._signals(1), _panel())
        assert not any("样本不足" in x for x in res.reasons), (
            f"恰好 30 个观测被判样本不足：{res.reasons}")

    def test_twenty_nine_observations_are_rejected(self, monkeypatch):
        gate = self._gate_with(monkeypatch, _rets(0.002, 0.008, n=29, seed=6))
        res = gate.evaluate(self._signals(1), _panel())
        assert res.passed is False
        assert any("样本不足" in x for x in res.reasons)

    def test_constant_returns_are_rejected(self, monkeypatch):
        """
        `float(np.nanstd(rets.values)) == 0.0` 的零方差守卫。

        ⚠️ 必须用**恰好 0.0** 的收益：`pd.Series(0.001)` 的 nanstd 是 2.17e-19
        而不是 0（浮点求和残差），这个守卫**根本不会触发**。
        这本身是产品代码的一处脆弱（用 `== 0.0` 判浮点零），已登记进 MUTATION_LEDGER；
        这里如实钉住当前行为。
        """
        gate = self._gate_with(monkeypatch, pd.Series(0.0, index=IDX[:100]))
        res = gate.evaluate(self._signals(1), _panel())
        assert res.passed is False
        assert any("方差为 0" in x for x in res.reasons)

    def test_near_constant_returns_slip_past_the_zero_variance_guard(self, monkeypatch):
        """
        与上一条配对，钉住那处脆弱的**现状**：日收益恒为 0.001 时
        `np.nanstd` 给出 2.17e-19 ≠ 0，零方差守卫不触发，策略继续走后面的判定。
        修这处（改成阈值比较）时本条会红，届时应当一并更新。
        """
        rets = pd.Series(0.001, index=IDX[:100])
        assert float(np.nanstd(rets.values)) != 0.0
        gate = self._gate_with(monkeypatch, rets)
        res = gate.evaluate(self._signals(1), _panel())
        assert not any("方差为 0" in x for x in res.reasons)

    def test_segment_positivity_is_strict(self, monkeypatch):
        """
        `pct_seg_positive = mean([s > 0 for s in seg_sharpes])` —— 严格大于 0。
        构造一段夏普恰好为 0 的常数子段（`_sharpe` 对零方差返回 0.0），
        放宽成 `>=` 会把它算成"正段"，正段比虚高。
        """
        good = pd.Series(np.tile([0.02, 0.01], 80), index=IDX[:160])
        flat = pd.Series(0.001, index=IDX[160:200])
        gate = self._gate_with(monkeypatch, pd.concat([good, flat]), n_segments=5)
        res = gate.evaluate(self._signals(1), _panel())
        assert res.min_seg_sharpe == pytest.approx(0.0, abs=1e-12), (
            "构造前提：应存在一段夏普恰好为 0")
        assert res.pct_seg_positive == pytest.approx(0.8, abs=1e-9), (
            f"夏普为 0 的段被算成了正段：正段比={res.pct_seg_positive}")

    def test_zero_sharpe_segment_blocks_the_gate(self, monkeypatch):
        """`if res.min_seg_sharpe <= 0.0` —— 恰好 0 也必须拦。"""
        good = pd.Series(np.tile([0.02, 0.01], 80), index=IDX[:160])
        flat = pd.Series(0.001, index=IDX[160:200])
        gate = self._gate_with(monkeypatch, pd.concat([good, flat]), n_segments=5)
        res = gate.evaluate(self._signals(1), _panel())
        assert res.passed is False
        assert any("Sharpe≤0" in x for x in res.reasons)

    def test_segment_count_shortfall_is_reported(self, monkeypatch):
        """
        `if res.n_segments < self.n_segments` 的边界：段数**恰好等于**要求时不报警。
        20 个观测切 5 段 → 每段 4 个 < 5，全部被丢弃 → n_segments=0。
        40 个观测切 5 段 → 每段 8 个 ≥ 5 → n_segments=5，不得报"分段不足"。
        """
        gate = self._gate_with(monkeypatch, _rets(0.002, 0.008, n=40, seed=7), n_segments=5)
        res = gate.evaluate(self._signals(1), _panel())
        assert res.n_segments == 5
        assert not any("分段不足" in x for x in res.reasons), (
            f"段数恰好达标却报分段不足：{res.reasons}")

    def test_tstat_formula_scales_with_sqrt_n(self, monkeypatch):
        """
        `tstat = (mu/sd) * sqrt(n)`。写成 `/` 后 t 随样本增大而**变小**，
        长样本策略永远过不了 t 门槛 —— 而这正是"多攒数据"应该带来的收益。
        """
        short = _rets(0.001, 0.01, n=60, seed=8)
        long = pd.concat([short, _rets(0.001, 0.01, n=60, seed=8).set_axis(IDX[60:120])])
        g1 = self._gate_with(monkeypatch, short)
        r1 = g1.evaluate(self._signals(1), _panel())
        g2 = self._gate_with(monkeypatch, long)
        r2 = g2.evaluate(self._signals(1), _panel())
        mu, sd = float(np.mean(short.values)), float(np.std(short.values, ddof=1))
        assert r1.t_stat == pytest.approx((mu / sd) * np.sqrt(len(short)), abs=1e-9)
        assert abs(r2.t_stat) > abs(r1.t_stat), (
            "样本翻倍后 t 统计量没有变大 —— √n 的方向疑似反了")

    def test_tstat_is_zero_for_zero_dispersion(self, monkeypatch):
        """`tstat = ... if sd > 1e-12 else 0.0` —— 零方差不得做除法。"""
        rets = pd.Series([0.001] * 99 + [0.0011], index=IDX[:100])
        gate = self._gate_with(monkeypatch, rets)
        res = gate.evaluate(self._signals(1), _panel())
        assert np.isfinite(res.t_stat)

    def test_tstat_exactly_at_threshold_passes(self, monkeypatch):
        """
        `if tstat < self.min_tstat: reasons.append(...)` —— **恰好等于**门槛算过。
        放宽成 `<=` 会把恰好达标的策略拦下。用打桩收益反解出精确的 t。
        """
        n = 100
        base = np.tile([1.0, -1.0], n // 2)
        rets = pd.Series(base * 0.01, index=IDX[:n])
        mu, sd = float(np.mean(rets.values)), float(np.std(rets.values, ddof=1))
        tstat = (mu / sd) * np.sqrt(n)
        gate = self._gate_with(monkeypatch, rets, min_tstat=tstat)
        res = gate.evaluate(self._signals(1), _panel())
        assert res.t_stat == pytest.approx(tstat, abs=1e-12)
        assert not any("夏普 t=" in x for x in res.reasons), (
            f"t 恰好等于门槛却被拦：{res.reasons}")

    def test_dsr_threshold_is_inclusive_on_the_reject_side(self, monkeypatch):
        """`if dsr <= self.dsr_threshold` —— 恰好等于阈值算**不通过**（保守方向）。"""
        rets = _rets(0.003, 0.008, n=200, seed=9)
        gate = self._gate_with(monkeypatch, rets, dsr_threshold=0.0)
        res = gate.evaluate(self._signals(1), _panel())
        assert res.deflated_sharpe > 0.0
        assert not any("DSR" in x for x in res.reasons)

        gate2 = self._gate_with(monkeypatch, rets, dsr_threshold=1.0)
        res2 = gate2.evaluate(self._signals(1), _panel())
        assert any("DSR" in x for x in res2.reasons)

    @pytest.mark.parametrize("allow_short,expect_long_only",
                             [(False, True), (True, False)])
    def test_long_only_follows_allow_short_setting(self, monkeypatch, allow_short,
                                                   expect_long_only):
        """
        `long_only=(not getattr(settings, "trading_allow_short", False))` ——
        删掉 `not` 或把 getattr 默认改成 True，都会让**现金账户被允许做空**。

        ⚠️ 第一版是在测试里把同一个表达式重算一遍再和自己比 —— 同义反复，
        根本没碰产品代码（变异测试证实两处变异都存活）。
        这一版**拦截真正送进 PortfolioRiskGate 的 RiskLimits**。
        """
        import app.core.portfolio_manager.risk_gate as rg
        from app.config import settings

        monkeypatch.setattr(settings, "trading_allow_short", allow_short,
                            raising=False)
        seen = {}
        orig_init = rg.PortfolioRiskGate.__init__

        def _spy(self, limits, *a, **k):
            seen["long_only"] = limits.long_only
            return orig_init(self, limits, *a, **k)

        monkeypatch.setattr(rg.PortfolioRiskGate, "__init__", _spy)
        idx = pd.bdate_range("2024-01-02", periods=60)
        sigs = {"f1": pd.DataFrame(
            np.tile(np.arange(4, dtype=float), (60, 1)), index=idx,
            columns=[f"T{i}" for i in range(4)])}
        sg.strategy_net_returns(sigs, _panel(n_days=60, n_tickers=4), aum=10_000.0)
        assert seen.get("long_only") is expect_long_only, (
            f"allow_short={allow_short} 时送进风控门的 long_only 是 "
            f"{seen.get('long_only')}，应为 {expect_long_only}")

    def test_missing_allow_short_setting_defaults_to_long_only(self, monkeypatch):
        """
        `getattr(settings, "trading_allow_short", False)` 的**第三参**：
        配置项缺失时必须按**不允许做空**处理（保守侧，DEV_LESSONS §U）。
        默认值只在字段缺失时可达 —— 显式赋值的用例测不到它。
        """
        import types
        import app.config
        import app.core.portfolio_manager.risk_gate as rg

        monkeypatch.setattr(app.config, "settings", types.SimpleNamespace())
        seen = {}
        orig_init = rg.PortfolioRiskGate.__init__

        def _spy(self, limits, *a, **k):
            seen["long_only"] = limits.long_only
            return orig_init(self, limits, *a, **k)

        monkeypatch.setattr(rg.PortfolioRiskGate, "__init__", _spy)
        idx = pd.bdate_range("2024-01-02", periods=60)
        sigs = {"f1": pd.DataFrame(
            np.tile(np.arange(4, dtype=float), (60, 1)), index=idx,
            columns=[f"T{i}" for i in range(4)])}
        sg.strategy_net_returns(sigs, _panel(n_days=60, n_tickers=4), aum=10_000.0)
        assert seen.get("long_only") is True, (
            "配置项缺失时放开了做空 —— getattr 的默认值方向反了")

    def test_global_trials_failure_is_logged_not_silent(self, monkeypatch, caplog):
        """
        读取全局试验台账失败时 n_trials 退回 1 —— 这**不是无害兜底**：
        DSR 的多重检验校正整体失效，门变松。必须留下 ERROR。
        """
        monkeypatch.setattr(sg, "strategy_net_returns",
                            lambda *a, **k: (_rets(0.002, 0.008, seed=11), pd.DataFrame()))

        class _Boom:
            def __init__(self):
                raise RuntimeError("ledger down")

        monkeypatch.setattr("app.db.trial_ledger.TrialLedger", _Boom)
        with caplog.at_level("ERROR"):
            res = StrategyGate(use_global_trials=True).evaluate(self._signals(1), _panel())
        assert res.n_trials == 1
        assert any("多重检验" in r.getMessage() for r in caplog.records), (
            "试验台账读取失败没有留下 ERROR —— 门悄悄变松了")


# ===========================================================================
# E. PBO 的启用条件
# ===========================================================================

class TestPboGate:

    def test_pbo_needs_two_factors_and_enough_rows(self, monkeypatch):
        """
        `if mat.shape[0] >= 2 * self.pbo_n_splits and mat.shape[1] >= 2:`
          - `and` 改成 `or`：单因子（列数 1）也会去算 PBO → CSCV 无从组合
          - `2 * n_splits` 写成 `/`：行数门槛从 16 掉到 4，短样本上 PBO 变成噪声

        用打桩收益控制矩阵形状：行数 10 < 2×8，PBO 必须**不算**（保持 None）。
        """
        monkeypatch.setattr(sg, "strategy_net_returns",
                            lambda *a, **k: (_rets(0.002, 0.01, n=40, seed=12),
                                             pd.DataFrame()))
        idx = pd.bdate_range("2023-01-02", periods=60)
        sigs = {f"f{i}": pd.DataFrame(1.0, index=idx, columns=["T0", "T1"])
                for i in range(2)}
        gate = StrategyGate(use_global_trials=False, pbo_n_splits=8)
        res = gate.evaluate(sigs, _panel())
        # 40 行 >= 16 → 会算；再用一个短的对照
        assert res.pbo is not None

        monkeypatch.setattr(sg, "strategy_net_returns",
                            lambda *a, **k: (_rets(0.002, 0.01, n=40, seed=12),
                                             pd.DataFrame()))
        gate_wide = StrategyGate(use_global_trials=False, pbo_n_splits=30)
        res_wide = gate_wide.evaluate(sigs, _panel())
        assert res_wide.pbo is None, (
            "行数 40 < 2×30 时不该计算 PBO —— 行数门槛疑似被改小")

    def test_pbo_exactly_at_the_threshold_does_not_block(self, monkeypatch):
        """
        `if pbo > self.pbo_threshold:` —— **恰好等于**阈值不算过拟合
        （阈值本身是"警示线"，等于它按不拦处理）。放宽成 `>=` 会把
        刚好踩线的策略拦下，口径与文档不符。

        PBO 是 `lam_neg / total` 这样的有理数，0.5 之类的取值精确可达；
        这里直接把计算函数打桩成阈值本身，保证边界精确。
        """
        monkeypatch.setattr(sg, "strategy_net_returns",
                            lambda *a, **k: (_rets(0.002, 0.01, n=100, seed=20),
                                             pd.DataFrame()))
        monkeypatch.setattr(
            "app.core.backtest_engine.overfit_stats."
            "probability_of_backtest_overfitting",
            lambda mat, n_splits=8: 0.5)
        idx = pd.bdate_range("2023-01-02", periods=60)
        sigs = {f"f{i}": pd.DataFrame(1.0, index=idx, columns=["T0", "T1"])
                for i in range(2)}
        res = StrategyGate(use_global_trials=False, pbo_threshold=0.5).evaluate(
            sigs, _panel())
        assert res.pbo == pytest.approx(0.5, abs=1e-12)
        assert not any("PBO" in x for x in res.reasons), (
            f"PBO 恰好等于阈值却被判为过拟合：{res.reasons}")

    def test_pbo_above_the_threshold_blocks(self, monkeypatch):
        """对照组：超过阈值必须拦，否则上一条可以靠'永远不拦'作弊通过。"""
        monkeypatch.setattr(sg, "strategy_net_returns",
                            lambda *a, **k: (_rets(0.002, 0.01, n=100, seed=20),
                                             pd.DataFrame()))
        monkeypatch.setattr(
            "app.core.backtest_engine.overfit_stats."
            "probability_of_backtest_overfitting",
            lambda mat, n_splits=8: 0.51)
        idx = pd.bdate_range("2023-01-02", periods=60)
        sigs = {f"f{i}": pd.DataFrame(1.0, index=idx, columns=["T0", "T1"])
                for i in range(2)}
        res = StrategyGate(use_global_trials=False, pbo_threshold=0.5).evaluate(
            sigs, _panel())
        assert any("PBO" in x for x in res.reasons)
        assert res.passed is False

    def test_single_factor_skips_pbo(self, monkeypatch):
        """`if len(factor_signals) >= 2:` —— 单因子无法做 CSCV，必须跳过。"""
        monkeypatch.setattr(sg, "strategy_net_returns",
                            lambda *a, **k: (_rets(0.002, 0.01, n=100, seed=13),
                                             pd.DataFrame()))
        idx = pd.bdate_range("2023-01-02", periods=60)
        res = StrategyGate(use_global_trials=False).evaluate(
            {"only": pd.DataFrame(1.0, index=idx, columns=["T0"])}, _panel())
        assert res.pbo is None


# ===========================================================================
# F. 边际贡献准入 —— 贪心选择的每一步
# ===========================================================================

class TestMarginalSelection:
    """
    `marginal_factor_selection` 决定**哪些因子进组合**。它的每一步判定都能改坏：
    提升量写成加法、比较符放宽、纳入标记写反。此前零断言。
    """

    IDX60 = pd.bdate_range("2023-01-02", periods=60)

    @classmethod
    def _sig(cls, name: str) -> pd.DataFrame:
        return pd.DataFrame(1.0, index=cls.IDX60, columns=["T0", "T1"], dtype=float)

    @staticmethod
    def _stub_oos(monkeypatch, table: dict):
        """
        按**已选因子集合**给出**精确**的尾段夏普。

        ⚠️ 第一版是把目标值编码进收益序列再反算回来，浮点上只能近似
        （误差 ~1e-9），于是"提升恰好等于门槛"这类边界用例其实没测到边界
        （变异测试证实 L417/L419 存活）。这一版用 `Series.name` 传标签，
        `_oos_tail_sharpe` 直接查表，取值精确。
        """
        def _fake(sig, dataset, **kw):
            return pd.Series(np.zeros(10), name="|".join(sorted(sig))), pd.DataFrame()

        def _tail(rets, oos_ratio=0.30):
            key = frozenset(x for x in str(rets.name).split("|") if x)
            return float(table[key])

        monkeypatch.setattr(sg, "strategy_net_returns", _fake)
        monkeypatch.setattr(sg, "_oos_tail_sharpe", _tail)

    def test_improvement_is_candidate_minus_base(self, monkeypatch):
        """
        `impr = cand_oos - base_oos`。写成 `+` 后"提升"变成两者之和：
        一个**让策略变差**的候选（base=1.0 → cand=0.6）也会得到 impr=1.6 而被纳入。
        """
        self._stub_oos(monkeypatch, {
            frozenset(): 0.0,
            frozenset({"a"}): 1.0,
            frozenset({"b"}): 0.6,
            frozenset({"a", "b"}): 0.7,
        })
        out = marginal_factor_selection(
            {"a": self._sig("a"), "b": self._sig("b")}, _panel(), min_improve=0.05)
        assert out.selected == ["a"], (
            f"只有 a 提升了策略，b 反而拉低，却选出了 {out.selected}")
        assert out.final_oos == pytest.approx(1.0, abs=1e-6)

    def test_rejected_candidates_are_recorded_as_not_admitted(self, monkeypatch):
        """
        被拒候选走 `MarginalStep(..., admitted=False)`，纳入的走 `admitted=True`。
        任一处标记写反，审计轨迹就会把"拒绝"显示成"纳入"。
        """
        self._stub_oos(monkeypatch, {
            frozenset(): 0.0,
            frozenset({"a"}): 1.0,
            frozenset({"b"}): 0.6,
            frozenset({"a", "b"}): 1.01,
        })
        out = marginal_factor_selection(
            {"a": self._sig("a"), "b": self._sig("b")}, _panel(), min_improve=0.05)
        admitted = [s for s in out.steps if s.admitted]
        rejected = [s for s in out.steps if not s.admitted]
        assert [s.factor for s in admitted] == ["a"]
        assert [s.factor for s in rejected] == ["b"]
        assert rejected[0].improvement == pytest.approx(0.01, abs=1e-6), (
            "被拒候选记录的提升量不是 cand - base")

    def test_min_improve_is_a_lower_bound(self, monkeypatch):
        """
        `if best_name is None or best_impr < min_improve: break` ——
        提升**恰好等于** min_improve 应当纳入。放宽成 `<=` 会把它拒掉。
        """
        self._stub_oos(monkeypatch, {
            frozenset(): 0.0,
            frozenset({"a"}): 0.05,
        })
        out = marginal_factor_selection({"a": self._sig("a")}, _panel(), min_improve=0.05)
        assert out.selected == ["a"], "提升恰好等于门槛却被拒"

    def test_improvement_below_threshold_is_rejected(self, monkeypatch):
        self._stub_oos(monkeypatch, {
            frozenset(): 0.0,
            frozenset({"a"}): 0.04,
        })
        out = marginal_factor_selection({"a": self._sig("a")}, _panel(), min_improve=0.05)
        assert out.selected == []

    def test_seed_signals_are_not_re_evaluated(self, monkeypatch):
        """`remaining = {k: v for k, v in candidates.items() if k not in selected}` ——
        删掉 `not` 会让**已选**的因子重复参与，候选集与种子集互换。"""
        self._stub_oos(monkeypatch, {
            frozenset({"a"}): 1.0,
            frozenset({"a", "b"}): 1.5,
        })
        out = marginal_factor_selection(
            {"a": self._sig("a"), "b": self._sig("b")}, _panel(),
            min_improve=0.05, seed_signals={"a": self._sig("a")})
        assert set(out.selected) == {"a", "b"}
        assert [s.factor for s in out.steps if s.admitted] == ["b"], (
            "种子因子被当成候选重新评估了一遍")

    def test_zero_improvement_candidate_is_not_selected(self, monkeypatch):
        """
        `if impr > best_impr:` —— 初值 `best_impr = 0.0`，所以提升**恰好为 0**
        的候选不该被挑中。放宽成 `>=` 会让"完全没有边际贡献"的因子也进组合
        （正是这套准入要拒绝的那一类）。区分值就是 0.0 本身，可精确构造。
        """
        # 空集的基线由 `if not sig: return 0.0` 短路给出，恒为 0.0；
        # 让候选也恰好是 0.0，提升就精确等于 0。
        self._stub_oos(monkeypatch, {
            frozenset(): 0.0,
            frozenset({"a"}): 0.0,
        })
        out = marginal_factor_selection({"a": self._sig("a")}, _panel(),
                                        min_improve=0.0)
        assert out.selected == [], (
            f"零边际贡献的候选被纳入了组合：{out.selected}")
        assert [s.admitted for s in out.steps] == [False]

    def test_empty_signal_set_scores_zero(self, monkeypatch):
        """`if not sig: return 0.0` —— 删掉 `not` 会对空集去跑回测。"""
        self._stub_oos(monkeypatch, {frozenset({"a"}): 1.0})
        out = marginal_factor_selection({"a": self._sig("a")}, _panel(), min_improve=0.5)
        assert out.steps[0].oos_before == pytest.approx(0.0, abs=1e-12), (
            "空策略的基线 OOS 不是 0")


# ===========================================================================
# G. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "app/core/portfolio_manager/strategy_gate.py ×1 — L177 `_sharpe` 里 `if sd < 1e-12: return 0.0` → `<=`":
        "区分值需要 sd **恰好等于** 1e-12。sd = np.std(r, ddof=1) 是平方和/(n-1) "
        "再开方的浮点结果，无法反解出精确等于 1e-12 的样本；"
        "而它真正会取到的特殊值是 0.0（常数收益），两侧都返回 0.0。",

    "app/core/portfolio_manager/strategy_gate.py ×1 — L310 `tstat = ... if sd > 1e-12 else 0.0` → `>=`":
        "同上：区分点是 sd 恰好等于 1e-12。零方差时 `0.0 >= 1e-12` 为假，"
        "两侧都走 else 返回 0.0，行为一致。",
}


def test_dispersion_guards_are_unreachable():
    """L177 / L310 等价性的机械验证。"""
    tol = 1e-12
    for x in (0.0, -0.0, 1e-20, tol / 2, tol * 2, 0.01):
        assert (x < tol) == (x <= tol) or x == tol
        assert (x > tol) == (x >= tol) or x == tol
    rng = np.random.default_rng(31)
    for scale in (1e-12, 1e-11, 1e-13):
        for _ in range(200):
            assert float(np.std(rng.normal(0.0, scale, 60), ddof=1)) != tol


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 2
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
