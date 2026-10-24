"""
Validate 段预热：特征只用当时已知历史、只评分 Validate 日期、空仓起步、历史不足如实少评分。

来源（原测试文件的说明，保留出处）：

2026-10-22 用户决定 · 时钟（D14 (a) + C3）与 Validate 预热（D9）。

D14 原话："T 收盘信息完成并过健康门后冻结信号与目标，T+1 按明确交易规则执行，更新时钟版本。同步修改信号、
回测、内部模拟、IC、归档、基准及调度。'计划开盘执行'不等于限价单必然按开盘价成交；预测 IC 与实际持仓收益
分别记录。旧历史保留旧版本，先隔离重算。"
C3 原话："目前虽然要求开盘前存档，评分仍是前收盘到今收盘，包含已经发生的隔夜收益。必须与 D14 一起明确标签
起止、信息截止和归档截止，不能只提前 deadline。"
D9 原话："只用当时已知历史计算特征，只评分 Validate 日期；拟合与标准化不能读取未来评分信息。覆盖 252 日因子、
嵌套窗口、跨界标签、初始持仓及历史不足。"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.portfolio_manager.executable_spec import signal_config

# ---------------------------------------------------------------------------
# D9：Validate 段预热
# ---------------------------------------------------------------------------

def _long_panel(T=800, N=12, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=T)
    cols = [f"S{i}" for i in range(N)]
    c = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.015, (T, N)), 0), idx, cols)
    o = c.shift(1).fillna(c) * (1 + rng.normal(0, 0.003, (T, N)))
    amp = np.abs(rng.normal(0, 0.006, (T, N)))
    return {"open": o, "high": np.maximum(o, c) * (1 + amp), "low": np.minimum(o, c) * (1 - amp),
            "close": c, "volume": pd.DataFrame(1e7, idx, cols), "vwap": c,
            "returns": c.pct_change().fillna(0.0)}


def _split(ds, is_end=560, oos_start=580):
    cut = lambda a, b: {k: v.iloc[a:b] for k, v in ds.items()}          # noqa: E731
    return cut(0, is_end), cut(oos_start, len(next(iter(ds.values()))))


class TestValidateWarmup:

    DSL_252 = "rank(ts_delta(log(close), 252))"
    DSL_NESTED = "rank(ts_mean(ts_delta(log(close), 20), 252))"

    @pytest.mark.parametrize("dsl", [DSL_252, DSL_NESTED])
    def test_a_long_window_factor_is_evaluable_on_validate(self, dsl):
        from app.core.backtest_engine.realistic_backtester import RealisticBacktester
        is_d, oos_d = _split(_long_panel())
        res = RealisticBacktester(config=signal_config(), min_obs=0).run(dsl, is_d, oos_dataset=oos_d)
        assert res.oos_result is not None
        pos = res.oos_result.positions
        assert pos.index[0] >= oos_d["close"].index[0], "Validate 段的结果里混进了更早的日期"
        assert float(pos.abs().sum(axis=1).gt(0).mean()) > 0.9, "252 日因子在 Validate 段仍然几乎没有持仓"

    def test_validate_scores_do_not_read_data_after_the_validate_end(self):
        from app.core.backtest_engine.realistic_backtester import RealisticBacktester
        ds = _long_panel()
        is_d, oos_d = _split(ds)
        a = RealisticBacktester(config=signal_config(), min_obs=0).run(self.DSL_252, is_d, oos_dataset=oos_d)
        # Validate 段最后 30 天的价格换掉 → 那之前的每一天结果一点都不变（没有读未来）
        cut = oos_d["close"].index[-30]
        bump = np.where(oos_d["close"].index < cut, 1.0, 1.5)
        late = {k: (v.mul(bump, axis=0) if k in ("open", "high", "low", "close", "vwap") else v)
                for k, v in oos_d.items()}
        b = RealisticBacktester(config=signal_config(), min_obs=0).run(self.DSL_252, is_d, oos_dataset=late)
        before = a.oos_result.net_returns.index < cut
        pd.testing.assert_series_equal(a.oos_result.net_returns[before], b.oos_result.net_returns[before])
        # IS 的结果不受 Validate 段价格影响（预热只单向：历史 → Validate）
        oos_bumped = {k: (v * 1.5 if k in ("open", "high", "low", "close", "vwap") else v) for k, v in oos_d.items()}
        c = RealisticBacktester(config=signal_config(), min_obs=0).run(self.DSL_252, is_d, oos_dataset=oos_bumped)
        pd.testing.assert_series_equal(a.is_result.net_returns, c.is_result.net_returns)

    def test_validate_starts_flat(self):
        """初始持仓：Validate 段从空仓开始，不继承 IS 段最后的仓位。"""
        from app.core.backtest_engine.realistic_backtester import RealisticBacktester
        is_d, oos_d = _split(_long_panel())
        res = RealisticBacktester(config=signal_config(), min_obs=0).run(self.DSL_252, is_d, oos_dataset=oos_d)
        assert float(res.oos_result.positions.iloc[0].abs().sum()) == 0.0

    def test_insufficient_history_is_reported_not_faked(self):
        """历史不足：预热不够长的那几天如实为空，并报告被评分的天数。"""
        from app.core.backtest_engine.realistic_backtester import RealisticBacktester
        ds = _long_panel(T=420)
        is_d, oos_d = _split(ds, is_end=150, oos_start=160)
        res = RealisticBacktester(config=signal_config(), min_obs=0).run(self.DSL_252, is_d, oos_dataset=oos_d)
        info = res.oos_warmup
        assert info["history_days"] == 150 + len(oos_d["close"])
        assert 0 < info["scored_days"] < len(oos_d["close"]), info
        assert pd.Timestamp(info["first_scored"]) > oos_d["close"].index[0]
