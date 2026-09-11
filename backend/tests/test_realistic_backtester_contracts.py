"""
realistic_backtester.py —— 契约与边界定钉测试（变异测试驱动）

来由：37 个变异点首测击杀率 **32.4%**（存活 25）。既有覆盖（test_phase1/2/3/4/6）
全都走"跑一遍看有没有崩"的路子，对**报告层**与**Walk-Forward 汇总层**几乎零断言：

  - `RealisticBacktestResult` 的四个 `field(repr=False)` 是防日志刷屏的，改成 True
    也没人发现（一次 repr 会把整张 T×N 面板打进日志）
  - `_degradation_table` 的 IS→OOS 变化量写成加法、`None`/NaN 判定写成 `or`，
    对比表就会给出**方向相反**或全是 N/A 的结论
  - Walk-Forward 的过拟合度 `(is_s - oos_s)/|is_s|` 写成加法后，
    OOS 越好反而"过拟合越严重"
  - 无 volume 时的虚拟成交量 `ones * 1e6` 写成 `/ 1e6` → 每天 1e-6 股的成交量，
    ADV 上限把所有持仓削光

本文件逐条处置这些盲区。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from app.core.alpha_engine.signal_processor import SimulationConfig
from app.core.backtest_engine.realistic_backtester import (
    RealisticBacktester,
    WalkForwardBacktester,
    _degradation_table,
)
from app.core.backtest_engine.risk_report import RiskReport

DSL = "rank(ts_delta(log(close), 5))"


def _dataset(n_days: int = 160, n_tickers: int = 8, seed: int = 0, with_volume: bool = True):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    cols = [f"T{i:02d}" for i in range(n_tickers)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.0003, 0.012, (n_days, n_tickers)), axis=0),
        index=idx, columns=cols)
    high = close * (1 + rng.uniform(0, 0.006, close.shape))
    low = close * (1 - rng.uniform(0, 0.006, close.shape))
    ds = {"close": close, "open": close, "high": high, "low": low,
          "vwap": (high + low + close) / 3.0,
          "returns": close.pct_change().fillna(0.0)}
    if with_volume:
        ds["volume"] = pd.DataFrame(
            rng.integers(2e6, 9e6, close.shape).astype(float), index=idx, columns=cols)
    return ds


# ===========================================================================
# A. 结果容器的 repr —— 防日志刷屏
# ===========================================================================

class TestResultRepr:
    """
    `raw_signal` / `processed_signal` / `is_result` / `oos_result` 四个字段
    都标了 `repr=False`。这不是风格问题：它们是 T×N 的面板与完整回测结果，
    一旦进入 repr，任何一次 `logger.info(result)` 或异常回溯都会把整张表打出来。
    """

    @pytest.fixture(scope="class")
    def result(self):
        return RealisticBacktester(config=SimulationConfig()).run(DSL, _dataset())

    def test_repr_excludes_the_bulky_fields(self, result):
        text = repr(result)
        for field in ("raw_signal", "processed_signal", "is_result", "oos_result"):
            assert field not in text, (
                f"{field} 出现在 repr 里 —— repr=False 疑似被改成了 True，"
                f"一次日志就会打印整张面板")

    def test_repr_stays_short(self, result):
        text = repr(result)
        assert len(text) < 2000, f"repr 长度 {len(text)}，面板疑似被打进了 repr"

    def test_repr_still_shows_the_reports(self, result):
        """对照组：该显示的仍要显示，否则上一条可以靠"repr 全空"作弊通过。"""
        assert "is_report" in repr(result)


# ===========================================================================
# B. summary / to_dict 的 OOS 分支
# ===========================================================================

class TestSummaryBranches:

    def test_summary_without_oos_does_not_crash(self):
        """
        `if self.oos_report is not None:` 删掉 `not` 后，**没有** OOS 时反而会去
        调用 `None.summary()`。这是最常见的调用方式（只跑 IS），却没有断言保护。
        """
        r = RealisticBacktester(config=SimulationConfig()).run(DSL, _dataset())
        assert r.oos_report is None
        text = r.summary()
        assert "In-Sample" in text
        assert "Out-of-Sample" not in text

    def test_summary_with_oos_includes_the_degradation_table(self):
        ds = _dataset(n_days=200, seed=1)
        oos = _dataset(n_days=80, seed=2)
        r = RealisticBacktester(config=SimulationConfig()).run(DSL, ds, oos_dataset=oos)
        assert r.oos_report is not None
        text = r.summary()
        assert "Out-of-Sample" in text
        assert "IS vs OOS 退化对比" in text

    def test_to_dict_oos_is_none_without_oos(self):
        r = RealisticBacktester(config=SimulationConfig()).run(DSL, _dataset())
        assert r.to_dict()["oos"] is None
        assert r.to_dict()["is"] is not None

    def test_oos_dataset_needs_more_than_five_rows(self):
        """
        `if oos_dataset is not None and len(...) > 5:` —— **恰好 5 行**不算够，
        6 行才跑。放宽成 `>=` 会让 5 行的 OOS 段也生成一份毫无意义的报告。
        """
        ds = _dataset(n_days=160, seed=3)
        bt = RealisticBacktester(config=SimulationConfig())
        five = {k: v.iloc[:5] for k, v in _dataset(n_days=40, seed=4).items()}
        six = {k: v.iloc[:6] for k, v in _dataset(n_days=40, seed=4).items()}
        assert bt.run(DSL, ds, oos_dataset=five).oos_report is None, (
            "只有 5 行的 OOS 段不该产出报告")
        assert bt.run(DSL, ds, oos_dataset=six).oos_report is not None, (
            "6 行的 OOS 段必须产出报告 —— 边界判错")


# ===========================================================================
# C. 退化对比表 —— 数值格式化与变化量
# ===========================================================================

class TestDegradationTable:
    """
    `_degradation_table` 是 IS→OOS 的**结论展示层**，此前零断言。
    三处独立的判定都能改坏而不被发现：`_pct`/`_f` 的 None/NaN 判定、
    delta 的两个 `not`/`and`、以及 delta 本身的减号。
    """

    @staticmethod
    def _report(**kw) -> RiskReport:
        base = dict(sharpe_ratio=1.0, annualized_return=0.20, max_drawdown=-0.10,
                    ic_ir=0.5, ann_turnover=3.0)
        base.update(kw)
        r = RiskReport.__new__(RiskReport)
        for k, v in base.items():
            object.__setattr__(r, k, v)
        return r

    def test_numeric_metrics_are_formatted_as_numbers(self):
        """
        `if v is None or (isinstance(v, float) and np.isnan(v)): return N/A`
        —— `and` 改成 `or` 后**任何 float** 都会被判成缺失，整张表变成 N/A。
        """
        table = _degradation_table(self._report(), self._report(sharpe_ratio=0.6))
        assert "N/A" not in table, f"正常数值被格式化成了 N/A：\n{table}"
        assert "+1.0000" in table and "+0.6000" in table

    def test_delta_is_oos_minus_is(self):
        """
        `delta = ov - iv`。写成 `+` 后"变化"变成两者之和：
        IS 1.0 → OOS 0.6 的**退化 -0.4** 会显示成 +1.6（看起来是大幅改善）。
        """
        table = _degradation_table(self._report(), self._report(sharpe_ratio=0.6))
        assert "-0.4000" in table, f"Sharpe 的变化量不是 OOS-IS：\n{table}"
        assert "+1.6000" not in table

    def test_missing_metric_shows_na_and_does_not_crash(self):
        """
        delta 的 `iv is not None and ov is not None and not (isnan…)`：
        删掉任一个 `not`、或把 `and` 改成 `or`，都会让 `np.isnan(None)` 抛异常
        或让缺失值参与减法。
        """
        table = _degradation_table(self._report(ic_ir=None),
                                   self._report(ic_ir=float("nan")))
        assert "N/A" in table
        assert "Sharpe" in table, "缺失一个指标不应影响其余行的输出"

    def test_nan_on_one_side_still_prints_the_other_side(self):
        table = _degradation_table(self._report(sharpe_ratio=float("nan")),
                                   self._report(sharpe_ratio=0.8))
        assert "+0.8000" in table, f"一侧 NaN 时另一侧的数值不应丢失：\n{table}"


# ===========================================================================
# D. 数据缺失时的兜底
# ===========================================================================

class TestFallbacks:

    def test_missing_volume_uses_one_million_shares(self):
        """
        `np.ones_like(prices) * 1e6`。写成 `/ 1e6` 会得到每天 1e-6 股的成交量。

        ⚠️ 只断言"持仓非零"是**不够**的：ADV 上限把权重削到极小值后仍然非零，
        净值也仍有波动（第一版就是这样，变异测试证实它存活）。
        真正的判据是**与显式 volume=1e6 的数据集逐点相同** —— 这直接钉住了那个常数。
        """
        ds_missing = _dataset(with_volume=False, seed=12)
        ds_explicit = _dataset(with_volume=False, seed=12)
        px = ds_explicit["close"]
        ds_explicit["volume"] = pd.DataFrame(
            np.ones_like(px.to_numpy()) * 1e6, index=px.index, columns=px.columns)

        bt = RealisticBacktester(config=SimulationConfig())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = bt.run(DSL, ds_missing).is_result.positions.to_numpy()
        b = bt.run(DSL, ds_explicit).is_result.positions.to_numpy()
        assert np.allclose(a, b, atol=1e-12), (
            "缺 volume 时的虚拟成交量与 1e6 不等价 —— 量纲疑似写反")
        assert np.abs(a).sum() > 0 and np.isfinite(a).all()

    def test_missing_close_raises(self):
        ds = _dataset()
        ds.pop("close")
        with pytest.raises(ValueError, match="close"):
            RealisticBacktester(config=SimulationConfig()).run(DSL, ds)

    def test_mvo_derives_returns_from_close_only_when_absent(self):
        """
        `if returns is None and "close" in dataset:` —— **两个条件都要满足**才推导。
        改成 `or` 后，即使 dataset 里已经有 `returns`，也会被 close 的 pct_change 覆盖，
        悄悄换掉收益口径（例如已做过 winsorize 的收益被原始收益顶替）。
        """
        ds = _dataset(n_days=200, seed=5)
        cfg = SimulationConfig(portfolio_mode="mvo")
        r_given = RealisticBacktester(config=cfg).run(DSL, ds)

        ds2 = {k: v for k, v in ds.items()}
        ds2["returns"] = ds["close"].pct_change() * 0.5      # 明显不同的收益口径
        r_other = RealisticBacktester(config=cfg).run(DSL, ds2)
        w1 = r_given.is_result.positions.to_numpy()
        w2 = r_other.is_result.positions.to_numpy()
        assert not np.allclose(w1, w2, atol=1e-10), (
            "换掉 dataset['returns'] 后 MVO 权重没有变化 —— returns 疑似被 close 覆盖")

    def test_burn_in_rows_are_trimmed(self):
        """
        `n_trimmed = (index < first_valid).sum()` 与 `if n_trimmed > 0`。
        时序算子的前 window 天全是 NaN，必须被裁掉，否则净值曲线前段是一条平线。
        用窗口 20 的 DSL：回测起点必须晚于数据起点。
        """
        ds = _dataset(n_days=160, seed=6)
        r = RealisticBacktester(config=SimulationConfig()).run(
            "rank(ts_mean(close, 20))", ds)
        eq = r.is_result.equity_curve
        assert eq.index[0] > ds["close"].index[0], (
            "burn-in 行没有被裁掉 —— 回测起点与数据起点相同")
        assert eq.notna().all()

    def test_no_trim_when_signal_is_valid_from_day_one(self):
        """对照组：无 burn-in 的信号不得被裁，否则上一条可能是"永远裁掉一段"。"""
        ds = _dataset(n_days=120, seed=7)
        r = RealisticBacktester(config=SimulationConfig(delay=0)).run("rank(close)", ds)
        assert r.is_result.equity_curve.index[0] == ds["close"].index[0]


# ===========================================================================
# E. Walk-Forward 汇总
# ===========================================================================

class TestWalkForwardAggregation:

    @pytest.fixture(scope="class")
    def wf_result(self):
        ds = _dataset(n_days=420, n_tickers=6, seed=9)
        return WalkForwardBacktester(
            config=SimulationConfig(), n_splits=3, min_train_days=120,
            embargo_days=5).run(DSL, ds)

    def test_insufficient_data_raises_with_a_usable_message(self):
        """`if not splits: raise` —— 删掉 `not` 会在**有**分折时反而报错。"""
        tiny = _dataset(n_days=40, n_tickers=4, seed=8)
        with pytest.raises(ValueError, match="Walk-Forward"):
            WalkForwardBacktester(config=SimulationConfig(), n_splits=3,
                                  min_train_days=120).run(DSL, tiny)

    def test_folds_are_produced_for_sufficient_data(self, wf_result):
        """对照组：数据足够时必须真的产出分折。"""
        assert wf_result.n_folds >= 2
        assert len(wf_result.fold_reports) == wf_result.n_folds

    def test_overfitting_is_is_minus_oos(self, wf_result):
        """
        `overfit = clip((is_s - oos_s) / |is_s|, 0, 1)`。写成 `+` 后，
        OOS **优于** IS（负退化）反而会得到正的过拟合度 —— 结论完全反了。

        ⚠️ 第一版是在**测试内部**把公式重算一遍再和自己比 —— 那是同义反复，
        根本没碰产品代码（变异测试证实它存活）。这一版拿**报告里真实的**
        is_sharpe / oos_sharpe 去核对报告里真实的 overfitting。
        """
        checked = 0
        for r in wf_result.fold_reports:
            expected = (float(np.clip((r.is_sharpe - r.oos_sharpe) / abs(r.is_sharpe),
                                      0.0, 1.0))
                        if abs(r.is_sharpe) > 1e-9 else 0.0)
            assert r.overfitting == pytest.approx(expected, abs=1e-12), (
                f"第 {r.fold_idx} 折：IS={r.is_sharpe:.4f} OOS={r.oos_sharpe:.4f} "
                f"→ overfitting={r.overfitting:.4f}，与 (IS-OOS)/|IS| 不符")
            # 加法版必须给出不同答案，否则本折区分不了该变异
            wrong = (float(np.clip((r.is_sharpe + r.oos_sharpe) / abs(r.is_sharpe),
                                   0.0, 1.0))
                     if abs(r.is_sharpe) > 1e-9 else 0.0)
            if wrong != pytest.approx(expected, abs=1e-9):
                checked += 1
        assert checked >= 1, (
            "没有任何一折能区分 (IS-OOS) 与 (IS+OOS) —— 本用例对该变异不敏感")

    def test_overfitting_is_reported_in_range(self, wf_result):
        for r in wf_result.fold_reports:
            assert 0.0 <= r.overfitting <= 1.0, f"过拟合度越界：{r.overfitting}"
        assert 0.0 <= wf_result.mean_overfitting <= 1.0

    def test_pct_positive_counts_strictly_positive_sharpes(self, monkeypatch):
        """
        `pct_positive = mean([s > 0 for s in oos_sharpes])` —— **严格大于 0**。
        放宽成 `>=` 会把"夏普恰好为 0"的折算成盈利折，胜率虚高。

        夏普恰好 0.0 是**可达**的：`_f()` 在 oos_report 缺失或指标为 NaN 时
        兜底成 0.0。这里把其中一折的 OOS 报告打成 None 来精确构造这个值。

        ⚠️ 第一版同样是在测试里自算自比（同义反复），对产品代码毫无约束。
        """
        ds = _dataset(n_days=420, n_tickers=6, seed=9)
        real_run = RealisticBacktester.run
        calls = {"n": 0}

        def _patched(self, dsl, dataset, *, oos_dataset=None, **kw):
            calls["n"] += 1
            res = real_run(self, dsl, dataset, oos_dataset=oos_dataset, **kw)
            if calls["n"] == 1:                       # 第一折：制造 oos_sharpe == 0.0
                object.__setattr__(res, "oos_report", None)
            return res

        monkeypatch.setattr(RealisticBacktester, "run", _patched)
        wf = WalkForwardBacktester(config=SimulationConfig(), n_splits=3,
                                   min_train_days=120, embargo_days=5).run(DSL, ds)
        oos = [r.oos_sharpe for r in wf.fold_reports]
        assert any(s == 0.0 for s in oos), "构造前提不成立：没有一折的 OOS 夏普恰好为 0"
        assert wf.pct_positive == pytest.approx(
            float(np.mean([s > 0 for s in oos])), abs=1e-12)
        assert wf.pct_positive != pytest.approx(
            float(np.mean([s >= 0 for s in oos])), abs=1e-9), (
            "严格大于与大于等于给出了相同的胜率 —— 本用例对该边界不敏感")

    def test_nan_metrics_are_coerced_to_zero(self, wf_result):
        """
        `_f(v) = fv if not np.isnan(fv) else 0.0` —— 删掉 `not` 会把**正常数值**
        统统换成 0，所有折的夏普都变成 0。
        """
        assert any(abs(r.is_sharpe) > 1e-12 for r in wf_result.fold_reports), (
            "所有折的 IS 夏普都是 0 —— NaN 兜底的分支方向疑似反了")
        for r in wf_result.fold_reports:
            assert np.isfinite(r.is_sharpe) and np.isfinite(r.oos_sharpe)

    def test_aggregate_statistics_match_the_folds(self, wf_result):
        oos = [r.oos_sharpe for r in wf_result.fold_reports]
        assert wf_result.mean_oos_sharpe == pytest.approx(float(np.mean(oos)), abs=1e-12)
        assert wf_result.std_oos_sharpe == pytest.approx(float(np.std(oos)), abs=1e-12)
        assert wf_result.min_oos_sharpe == pytest.approx(float(np.min(oos)), abs=1e-12)
        assert wf_result.pct_positive == pytest.approx(
            float(np.mean([s > 0 for s in oos])), abs=1e-12)


# ===========================================================================
# F. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "L274 `Executor(validate=False)` → True（RealisticBacktester.__init__）":
        "这一处与 daily_trading_loop 的同名开关不同：本类在 `run()` 里**已经**"
        "显式调用过 `self._validator.validate(node)`，DSL 走到 executor 时必然已通过"
        "同一套校验，再验一次结果相同（只多一次遍历）。"
        "见 test_validation_happens_before_execution。",

    "L374 `n_trimmed = int((proc_signal.index < first_valid).sum())` → `<=`":
        "n_trimmed **只用于日志文本和下一行的 >0 判定**，实际裁剪用的是 "
        "`proc_signal.loc[first_valid:]`，与这个计数无关。把 `<` 放宽成 `<=` "
        "只会让日志里多报一行，回测结果逐点不变。",

    "L375 `if n_trimmed > 0:` → `>=`":
        "n_trimmed == 0 时进入该分支所做的事（`proc_signal.loc[first_valid:]` 与"
        "同范围的 dataset 切片）都是**恒等操作**，只多打一行日志。"
        "见 test_burn_in_trim_is_idempotent_when_nothing_to_trim。",

    "L656 `if abs(is_s) > 1e-9 else 0.0` → `>=`":
        "区分值需要 |is_s| **恰好等于** 1e-9。is_s 是 `_f(is_r.sharpe_ratio)` 的输出，"
        "由整段回测的均值/标准差算出，无法反解出使其精确等于 1e-9 的输入；"
        "而它唯一容易取到的特殊值 0.0 在两侧都走 else 分支。",
}


def test_burn_in_trim_is_idempotent_when_nothing_to_trim():
    """
    L374/L375 等价性的机械验证：无 burn-in 时（n_trimmed == 0），
    进不进那个分支结果都一样 —— 因为分支里做的是恒等切片。
    """
    ds = _dataset(n_days=120, seed=7)
    r = RealisticBacktester(config=SimulationConfig(delay=0)).run("rank(close)", ds)
    sig = r.processed_signal
    valid_rows = sig.notna().any(axis=1)
    first_valid = sig.index[valid_rows.to_numpy().argmax()]
    assert int((sig.index < first_valid).sum()) == 0, "构造前提：本例无 burn-in"
    # 分支内的两个操作都是恒等的
    assert sig.loc[first_valid:].equals(sig)
    assert ds["close"].loc[first_valid:].equals(ds["close"])


def test_validation_happens_before_execution():
    """
    L274 等价性的机械验证：非法 DSL 在**执行之前**就被 `AlphaValidator` 拒掉，
    因此 executor 的 validate 开关取何值都观察不到差别。
    """
    bt = RealisticBacktester(config=SimulationConfig())
    assert bt._validator is not None
    with pytest.raises(Exception):
        bt.run("ts_mean(close, 300)", _dataset(n_days=400))   # 窗口 > 252，静态校验拒


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 4
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
