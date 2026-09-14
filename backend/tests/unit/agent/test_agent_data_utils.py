"""
agent/_data_utils.py —— QuantTools 的数据构造与切割

**此前零测试**（22 个变异点，D 档）。

这个模块干四件事，每一件坏掉都不会抛异常：

  1. `_make_synthetic_dataset` —— 合成面板。**LLM 工具链在没有真实
     数据时全靠它**。造出来的 high < low、vwap 跑到区间外、
     returns 与 close 对不上 —— 回测照跑，出来的 Sharpe 是假的。
  2. `_partition_three_way` —— IS / Validate / Test 三段切割。
     切点算错 = **样本外泄漏**，是这个项目最贵的一类错误。
  3. `load_real_dataset` —— 真实数据加载，失败必须炸得明确。
  4. `_run_backtest_core` —— 过拟合分数。除法方向、clip 边界、
     阈值比较任一处翻面，`is_overfit` 就会反过来。

因此本文件的手法是：
  - **不变量**（high ≥ close ≥ low、vwap 在 [low, high] 内）
  - **切片端点逐个钉死**，并断言三段**首尾相接、互不重叠、时间有序**
  - 过拟合分数用**参考公式逐位比对**（rtol=1e-12），而不是只看方向
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from app.agent._constants import _OVERFIT_THRESHOLD
from app.agent._data_utils import (
    _make_synthetic_dataset,
    _partition,
    _partition_three_way,
    _run_backtest_core,
    load_real_dataset,
)

FIELDS = ("close", "open", "high", "low", "volume", "vwap", "returns")


# ===========================================================================
# A. 合成面板的形状与字段
# ===========================================================================

class TestSyntheticDatasetShape:

    def test_every_documented_field_is_present(self):
        ds = _make_synthetic_dataset(n_tickers=5, n_days=30)
        assert set(ds) == set(FIELDS), (
            f"合成面板的字段集是 {sorted(ds)}，应当是 {sorted(FIELDS)} —— "
            f"少一个字段会让用到它的 DSL 直接解析失败")

    def test_all_fields_share_one_index_and_column_set(self):
        """
        任一字段的行列与 close 对不齐，下游做 `close * volume`
        就会产生 NaN 洞或笛卡尔展开，而 pandas 不会报错。
        """
        ds = _make_synthetic_dataset(n_tickers=4, n_days=25)
        ref = ds["close"]
        for name, df in ds.items():
            assert list(df.index) == list(ref.index), f"{name} 的日期索引与 close 不一致"
            assert list(df.columns) == list(ref.columns), f"{name} 的列与 close 不一致"

    def test_the_panel_has_the_requested_dimensions(self):
        ds = _make_synthetic_dataset(n_tickers=7, n_days=33)
        assert ds["close"].shape == (33, 7)

    def test_tickers_are_zero_padded_to_three_digits(self):
        """
        `f"T{i:03d}"` —— 位数被改会让列名从 T000 变成 T0，
        任何按名字对表（sector 映射、持仓续接）的地方都会错配。
        """
        ds = _make_synthetic_dataset(n_tickers=12, n_days=25)
        assert list(ds["close"].columns)[:3] == ["T000", "T001", "T002"]
        assert list(ds["close"].columns)[-1] == "T011"

    def test_dates_are_business_days_starting_from_the_documented_anchor(self):
        """
        `pd.bdate_range("2021-01-04", periods=n_days)` ——
        起点被改会让合成数据与真实数据的日期区间对不上；
        用自然日代替工作日会引入周末，破坏"每行一个交易日"的假设。
        """
        ds = _make_synthetic_dataset(n_tickers=3, n_days=10)
        idx = ds["close"].index
        assert idx[0] == pd.Timestamp("2021-01-04")
        assert (idx.dayofweek < 5).all(), "合成面板里出现了周末"
        assert list(idx) == list(pd.bdate_range("2021-01-04", periods=10))

    def test_the_default_size_is_twenty_tickers_by_one_trading_year(self):
        ds = _make_synthetic_dataset()
        assert ds["close"].shape == (252, 20)


# ===========================================================================
# B. 合成面板的金融不变量 —— 算术变异全部落在这里
# ===========================================================================

class TestSyntheticDatasetInvariants:

    def test_high_is_never_below_close_and_low_never_above(self):
        """
        `high = close * (1 + rng.uniform(0, 0.02, ...))`
        `low  = close * (1 - rng.uniform(0, 0.02, ...))`

        `1 +` 的符号翻成 `-`（或 `low` 的 `-` 翻成 `+`）会让最高价
        低于收盘价、最低价高于收盘价。回测不会报错 ——
        但任何用到 `high`/`low` 的算子（真实滑点、日内区间）都会算反。
        """
        ds = _make_synthetic_dataset(n_tickers=10, n_days=120, seed=1)
        assert (ds["high"] >= ds["close"]).all().all(), (
            "存在 high < close 的格子 —— high 的 `1 + u` 符号被改了")
        assert (ds["low"] <= ds["close"]).all().all(), (
            "存在 low > close 的格子 —— low 的 `1 - u` 符号被改了")
        assert (ds["high"] >= ds["low"]).all().all()

    def test_the_high_low_band_stays_within_two_percent_of_close(self):
        """
        `rng.uniform(0, 0.02, ...)` —— 上界被改会让日内振幅失真。
        钉住带宽，不只钉方向。
        """
        ds = _make_synthetic_dataset(n_tickers=10, n_days=120, seed=1)
        hi_ratio = (ds["high"] / ds["close"] - 1.0)
        lo_ratio = (1.0 - ds["low"] / ds["close"])
        assert hi_ratio.max().max() <= 0.02 + 1e-12, f"high 超出 2%：{hi_ratio.max().max()}"
        assert lo_ratio.max().max() <= 0.02 + 1e-12, f"low 超出 2%：{lo_ratio.max().max()}"
        # 且确实用满了区间（不是恒等于 close）
        assert hi_ratio.max().max() > 0.015
        assert lo_ratio.max().max() > 0.015

    def test_vwap_is_the_mean_of_high_low_and_close(self):
        """
        `vwap = (high + low + close) / 3`

        `+` 翻成 `-`、`/3` 翻成 `*3` —— 结果都仍然是"一个价格序列"，
        形状和 dtype 都对，下游不会有任何抱怨。
        逐位比对参考实现，再加一条"落在 [low, high] 之间"的不变量。
        """
        ds = _make_synthetic_dataset(n_tickers=6, n_days=60, seed=9)
        ref = (ds["high"] + ds["low"] + ds["close"]) / 3
        assert np.allclose(ds["vwap"].values, ref.values, rtol=1e-12, atol=0), (
            "vwap 不等于 (high+low+close)/3")
        assert (ds["vwap"] >= ds["low"]).all().all(), "vwap 跌出了当日最低价"
        assert (ds["vwap"] <= ds["high"]).all().all(), "vwap 超出了当日最高价"

    def test_open_is_the_previous_close_with_the_first_bar_self_filled(self):
        """
        `open_ = close.shift(1).fillna(close)`

        `shift(1)` 改成 `shift(-1)` 就是**前视泄漏**：开盘价等于明天的收盘价。
        这在合成数据上做出来的因子会有惊人的 Sharpe，而且完全查不出来。
        """
        ds = _make_synthetic_dataset(n_tickers=4, n_days=40, seed=3)
        o, c = ds["open"], ds["close"]
        assert np.allclose(o.iloc[1:].values, c.iloc[:-1].values, rtol=1e-12), (
            "open 不等于前一日 close —— shift 的方向或步长被改了")
        assert np.allclose(o.iloc[0].values, c.iloc[0].values), (
            "第一根 K 线的 open 没有用当日 close 兜底")
        assert o.notna().all().all(), "open 里残留了 NaN"

    def test_returns_are_the_simple_pct_change_of_close_with_a_zero_first_bar(self):
        """
        `returns = close.pct_change().fillna(0.0)`

        填充值被改成 1.0 会让每只票的第一天凭空多出 100% 收益；
        不填充则留 NaN，下游 `rolling(...).std()` 的样本数悄悄少一格。
        """
        ds = _make_synthetic_dataset(n_tickers=5, n_days=50, seed=4)
        ref = ds["close"].pct_change()
        assert np.allclose(ds["returns"].iloc[1:].values, ref.iloc[1:].values,
                           rtol=1e-12), "returns 不等于 close 的百分比变化"
        assert (ds["returns"].iloc[0] == 0.0).all(), (
            "第一根 K 线的收益率不是 0 —— fillna 的填充值被改了")

    def test_the_close_path_reproduces_the_reference_cumprod_bit_for_bit(self):
        """
        **首测存活项（L30，两处）**：`100 * np.cumprod(1 + rng.normal(...))`

        `1 +` 翻成 `-`、`100 *` 翻成 `/` 都**观察不到**，如果只断言
        "价格为正""首日在 100 附近"：
          - `1 - r` 里 r ~ N(0, 0.012)，(1-r) 依旧恒为正、依旧在 1 附近，
            对称分布下统计性质完全一样；
          - `100 / cumprod` 里 cumprod ≈ 1，所以结果也还在 100 附近。

        唯一能区分的是**同一颗种子下的具体数值**。这里重放同一个
        `default_rng(seed)` 的第一次抽样（`close` 是它的第一个消费者），
        手算出参考路径逐位比对。
        """
        n_days, n_tickers, seed = 60, 5, 42
        ds = _make_synthetic_dataset(n_tickers=n_tickers, n_days=n_days, seed=seed)

        rng = np.random.default_rng(seed)
        draws = rng.normal(0, 0.012, (n_days, n_tickers))
        ref = 100 * np.cumprod(1 + draws, axis=0)

        assert np.allclose(ds["close"].values, ref, rtol=1e-12, atol=0), (
            "收盘价路径与 `100 * cumprod(1 + r)` 不符 —— "
            "首行算符（`100 *` 或 `1 +`）被改了")

    def test_daily_returns_equal_the_drawn_shocks(self):
        """
        上一条的独立佐证，也是更直观的一条：
        `close[t] / close[t-1] == 1 + r_t`，于是 `pct_change` **精确等于**
        当初抽出来的那一组正态随机数。

        `1 -` 的变异会让每一天的收益率符号整体翻转 ——
        用合成数据做出来的"动量因子"其实在做反转，反之亦然。
        """
        n_days, n_tickers, seed = 40, 4, 7
        ds = _make_synthetic_dataset(n_tickers=n_tickers, n_days=n_days, seed=seed)
        draws = np.random.default_rng(seed).normal(0, 0.012, (n_days, n_tickers))
        got = ds["close"].pct_change().values[1:]
        assert np.allclose(got, draws[1:], rtol=1e-9, atol=1e-15), (
            "日收益率与抽样出的冲击不一致 —— `1 + r` 的符号被改了")

    def test_prices_start_near_one_hundred_and_stay_positive(self):
        """价格的量纲不变量（与上面的逐位比对互为补充，读起来更直白）。"""
        ds = _make_synthetic_dataset(n_tickers=20, n_days=252, seed=42)
        c = ds["close"]
        assert (c > 0).all().all(), "出现了非正的收盘价"
        first = c.iloc[0]
        assert (first > 90).all() and (first < 110).all(), (
            f"首日价格不在 100 附近：min={first.min()}, max={first.max()}")

    def test_daily_volatility_is_close_to_the_configured_one_point_two_percent(self):
        """`rng.normal(0, 0.012, ...)` 的标准差被改会让整个面板的风险量纲漂移。"""
        ds = _make_synthetic_dataset(n_tickers=20, n_days=252, seed=42)
        sd = ds["close"].pct_change().std().mean()
        assert sd == pytest.approx(0.012, rel=0.15), f"日波动率实测 {sd:.5f}，偏离 1.2% 过多"

    def test_volume_is_a_positive_float_in_the_documented_range(self):
        """
        `rng.integers(500_000, 5_000_000, ...).astype(float)`
        —— 不转 float 会让 `close * volume` 在某些 pandas 版本上
        变成整型溢出；下界被改成 0 会造出零成交量的票。
        """
        ds = _make_synthetic_dataset(n_tickers=10, n_days=60, seed=6)
        v = ds["volume"]
        assert v.dtypes.eq(np.float64).all(), f"volume 不是 float64：{v.dtypes.unique()}"
        assert v.values.min() >= 500_000
        assert v.values.max() < 5_000_000


class TestSyntheticDatasetDeterminism:

    def test_the_same_seed_reproduces_the_panel_bit_for_bit(self):
        """
        `np.random.default_rng(seed)` —— 种子没接上（或换成全局
        `np.random`）会让每次调用产出不同的面板，
        于是"同一条 DSL 两次回测结果不同"，谱系无法复现。
        """
        a = _make_synthetic_dataset(n_tickers=6, n_days=40, seed=123)
        b = _make_synthetic_dataset(n_tickers=6, n_days=40, seed=123)
        for f in FIELDS:
            assert a[f].equals(b[f]), f"同一种子两次生成的 {f} 不一致"

    def test_different_seeds_produce_different_panels(self):
        a = _make_synthetic_dataset(n_tickers=6, n_days=40, seed=1)
        b = _make_synthetic_dataset(n_tickers=6, n_days=40, seed=2)
        assert not a["close"].equals(b["close"]), (
            "换了种子面板却一模一样 —— seed 参数没有接到 rng 上")

    def test_the_default_seed_is_forty_two(self):
        assert _make_synthetic_dataset(n_tickers=3, n_days=20)["close"].equals(
            _make_synthetic_dataset(n_tickers=3, n_days=20, seed=42)["close"])


# ===========================================================================
# C. 三段切割 —— 泄漏防线
# ===========================================================================

def _ds(n: int) -> dict:
    idx = pd.bdate_range("2021-01-04", periods=n)
    return {"close": pd.DataFrame({"A": np.arange(n, dtype=float)}, index=idx),
            "volume": pd.DataFrame({"A": np.arange(n, dtype=float)}, index=idx)}


class TestThreeWayPartition:

    def test_the_three_slices_are_contiguous_disjoint_and_ordered(self):
        """
        三段必须**首尾相接**（不丢样本）、**互不重叠**（不泄漏）、
        **时间有序**（IS 全在 Validate 之前，Validate 全在 Test 之前）。

        `_slice(n_is, n_is + n_val)` 里任一端点的算符被改，
        这三条里至少有一条会红。
        """
        n = 300
        is_d, val_d, test_d = _partition_three_way(_ds(n))
        i, v, t = (d["close"].index for d in (is_d, val_d, test_d))

        assert len(i) + len(v) + len(t) == n, (
            f"三段合计 {len(i)+len(v)+len(t)} 条，原始 {n} 条 —— 有样本被丢弃或重复")
        assert set(i) & set(v) == set(), "IS 与 Validate 有重叠 —— 样本外泄漏"
        assert set(v) & set(t) == set(), "Validate 与 Test 有重叠 —— 样本外泄漏"
        assert set(i) & set(t) == set(), "IS 与 Test 有重叠 —— 样本外泄漏"
        assert i[-1] < v[0], "IS 的末尾晚于 Validate 的开头 —— 时间顺序反了"
        assert v[-1] < t[0], "Validate 的末尾晚于 Test 的开头 —— 时间顺序反了"

    def test_the_slice_sizes_follow_the_documented_formula(self):
        """
        `n_test = max(1, int(n * test_ratio))`
        `n_val  = max(1, int(n * oos_ratio) - n_test)`
        `n_is   = n - n_val - n_test`

        三个减号任一翻成加号，长度立刻对不上。逐个钉死。
        """
        n = 300
        is_d, val_d, test_d = _partition_three_way(_ds(n), oos_ratio=0.30,
                                                   test_ratio=0.10)
        n_test = max(1, int(n * 0.10))
        n_val = max(1, int(n * 0.30) - n_test)
        n_is = n - n_val - n_test
        assert (len(is_d["close"]), len(val_d["close"]), len(test_d["close"])) == \
               (n_is, n_val, n_test), "三段长度与公式不符"
        assert (n_is, n_val, n_test) == (210, 60, 30)

    def test_the_ratios_actually_change_the_split(self):
        n = 400
        _, v1, t1 = _partition_three_way(_ds(n), oos_ratio=0.30, test_ratio=0.10)
        _, v2, t2 = _partition_three_way(_ds(n), oos_ratio=0.50, test_ratio=0.20)
        assert len(t2["close"]) > len(t1["close"]), "test_ratio 调大后 Test 没有变长"
        assert len(v2["close"]) > len(v1["close"]), "oos_ratio 调大后 Validate 没有变长"

    def test_every_field_is_sliced_the_same_way(self):
        """
        `{field: df.iloc[start:end] for field, df in dataset.items()}`
        —— 漏掉某个字段、或字段之间端点不一致，会让
        `close` 与 `volume` 错位一天，是最隐蔽的一类泄漏。
        """
        is_d, val_d, test_d = _partition_three_way(_ds(300))
        for part in (is_d, val_d, test_d):
            assert set(part) == {"close", "volume"}, "切片之后字段数量变了"
            assert list(part["close"].index) == list(part["volume"].index), (
                "同一段里两个字段的日期索引不一致")

    def test_a_dataset_too_small_for_three_slices_raises(self):
        """
        `if n_is < 20: raise ValueError`

        守卫被删会让 IS 只剩几天 —— GP 会在 3 个样本上"搜索"出
        Sharpe 5 的因子，然后在 Test 上归零，而且没有任何提示。
        """
        with pytest.raises(ValueError, match="数据量不足以支持三段切割"):
            _partition_three_way(_ds(25))

    def test_the_guard_fires_at_n_is_below_twenty_not_at_twenty(self):
        """
        `n_is < 20` 翻成 `<=` 会让恰好 20 天的 IS 也被拒。
        构造两个规模，让 n_is 恰好落在 19 与 20 上。
        """
        def n_is_for(n: int) -> int:
            n_test = max(1, int(n * 0.10))
            n_val = max(1, int(n * 0.30) - n_test)
            return n - n_val - n_test

        n_ok = next(n for n in range(20, 80) if n_is_for(n) == 20)
        n_bad = next(n for n in range(20, 80) if n_is_for(n) == 19)
        assert n_is_for(n_ok) == 20 and n_is_for(n_bad) == 19

        is_d, _, _ = _partition_three_way(_ds(n_ok))
        assert len(is_d["close"]) == 20, (
            "n_is 恰好 20 时被拒了 —— `n_is < 20` 被翻成了 `<=`")
        with pytest.raises(ValueError):
            _partition_three_way(_ds(n_bad))

    def test_the_error_message_reports_all_four_numbers(self):
        """报错要能让人直接看出是哪一段不够，而不是只说"不够"。"""
        with pytest.raises(ValueError) as ei:
            _partition_three_way(_ds(25))
        msg = str(ei.value)
        for token in ("n=25", "n_is=", "n_val=", "n_test="):
            assert token in msg, f"报错里缺少 {token}：{msg}"

    def test_each_slice_keeps_at_least_one_row(self):
        """`max(1, ...)` 两处 —— 去掉会在小数据集上造出空的 Validate/Test 段。"""
        n = 60
        is_d, val_d, test_d = _partition_three_way(_ds(n), oos_ratio=0.05,
                                                   test_ratio=0.01)
        assert len(val_d["close"]) >= 1 and len(test_d["close"]) >= 1, (
            "极小比例下切出了空段 —— `max(1, ...)` 被删了")

    def test_the_defaults_are_thirty_and_ten_percent(self):
        a = _partition_three_way(_ds(300))
        b = _partition_three_way(_ds(300), oos_ratio=0.30, test_ratio=0.10)
        for x, y in zip(a, b):
            assert x["close"].equals(y["close"])


class TestTwoWayPartition:

    def test_partition_returns_train_then_test_in_that_order(self):
        """
        `return part.train(), part.test()`
        —— 两者互换就是把样本外拿去训练。用日期范围判定谁在前。
        """
        train, test = _partition(_ds(400), oos_ratio=0.30)
        assert train["close"].index[-1] < test["close"].index[0], (
            "返回的第一段不是时间上靠前的训练段 —— train/test 顺序反了")

    def test_the_oos_ratio_is_forwarded_to_the_partitioner(self):
        small_train, _ = _partition(_ds(400), oos_ratio=0.50)
        big_train, _ = _partition(_ds(400), oos_ratio=0.10)
        assert len(big_train["close"]) > len(small_train["close"]), (
            "oos_ratio 没有透传给 DataPartitioner")

    def test_the_partitioner_receives_the_dataset_date_range(self, monkeypatch):
        """
        `start=str(dates[0].date())` / `end=str(dates[-1].date())`
        —— 端点取错（例如都取第一天）会让切割落在空区间上。
        """
        import app.core.data_engine.data_partitioner as dp_mod
        seen = {}
        real = dp_mod.DataPartitioner

        class _Spy(real):
            def __init__(self, *a, **k):
                seen.update(k)
                super().__init__(*a, **k)

        monkeypatch.setattr(dp_mod, "DataPartitioner", _Spy)
        ds = _ds(200)
        _partition(ds, oos_ratio=0.30)
        idx = ds["close"].index
        assert seen["start"] == str(idx[0].date())
        assert seen["end"] == str(idx[-1].date())
        assert seen["oos_ratio"] == 0.30


# ===========================================================================
# D. 真实数据加载的失败路径
# ===========================================================================

class TestLoadRealDataset:

    def test_a_registry_failure_is_reraised_as_runtime_error_with_context(self,
                                                                          monkeypatch):
        """
        `raise RuntimeError(...) from exc`

        这个 except 被删会让 yfinance 的 `JSONDecodeError` 之类
        原样冒到 LLM 工具层，调用方看到的报错与"数据集加载失败"毫无关系。
        报错里必须带上数据集名与区间，否则无从排查。
        """
        import app.core.data_engine.dataset_registry as reg

        def boom(*a, **k):
            raise ValueError("upstream exploded")

        monkeypatch.setattr(reg, "load_registry_dataset", boom)
        with pytest.raises(RuntimeError) as ei:
            load_real_dataset("us_tech_large", start="2021-01-01", end="2024-01-01")
        msg = str(ei.value)
        assert "us_tech_large" in msg, f"报错里没有数据集名：{msg}"
        assert "2021-01-01" in msg and "2024-01-01" in msg, f"报错里没有区间：{msg}"
        assert "upstream exploded" in msg, f"原始异常被吞了：{msg}"
        assert isinstance(ei.value.__cause__, ValueError), (
            "没有用 `raise ... from exc` 串上原因，traceback 断了")

    def test_a_successful_load_is_partitioned_into_is_and_oos(self, monkeypatch):
        import app.core.data_engine.dataset_registry as reg
        ds = _ds(400)
        monkeypatch.setattr(reg, "load_registry_dataset",
                            lambda name, **k: types.SimpleNamespace(data=ds))
        is_d, oos_d = load_real_dataset("whatever", oos_ratio=0.30)
        assert is_d["close"].index[-1] < oos_d["close"].index[0]
        assert len(is_d["close"]) + len(oos_d["close"]) <= 400

    def test_the_cache_is_requested_and_the_arguments_are_forwarded(self, monkeypatch):
        """
        `load_registry_dataset(name, start=start, end=end, use_cache=True)`
        —— `use_cache` 翻成 False 会让每次工具调用都重新下载，
        在无预算的模拟阶段直接撞上数据源限流。
        """
        import app.core.data_engine.dataset_registry as reg
        seen = {}

        def spy(name, **k):
            seen["name"] = name
            seen.update(k)
            return types.SimpleNamespace(data=_ds(400))

        monkeypatch.setattr(reg, "load_registry_dataset", spy)
        load_real_dataset("crypto_major", start="2020-02-02", end="2023-03-03")
        assert seen == {"name": "crypto_major", "start": "2020-02-02",
                        "end": "2023-03-03", "use_cache": True}, (
            f"传给 registry 的参数不对：{seen}")

    def test_the_default_window_and_ratio_are_the_documented_ones(self, monkeypatch):
        import app.core.data_engine.dataset_registry as reg
        seen = {}
        monkeypatch.setattr(reg, "load_registry_dataset",
                            lambda name, **k: (seen.update(k),
                                               types.SimpleNamespace(data=_ds(400)))[1])
        load_real_dataset("x")
        assert seen["start"] == "2021-01-01" and seen["end"] == "2024-01-01"


# ===========================================================================
# E. 过拟合分数 —— 除法方向 / clip 边界 / 阈值比较
# ===========================================================================

class _Report:
    def __init__(self, sharpe, ann_return=0.1, turnover=1.0, ic=0.02):
        self.sharpe_ratio = sharpe
        self.annualized_return = ann_return
        self.ann_turnover = turnover
        self.mean_ic = ic


class _Result:
    def __init__(self, is_r, oos_r):
        self.is_report = is_r
        self.oos_report = oos_r

    def summary(self):
        return "SUMMARY"


def _core(is_sharpe, oos_sharpe, monkeypatch, **rep):
    """把 RealisticBacktester 换成回放指定 Sharpe 的桩。"""
    import app.core.backtest_engine.realistic_backtester as rb

    class _BT:
        def __init__(self, config=None, min_obs=None):
            _BT.last_min_obs = min_obs

        def run(self, dsl, is_data, oos_dataset=None):
            return _Result(_Report(is_sharpe, **rep),
                           _Report(oos_sharpe, **rep) if oos_sharpe is not None else None)

    monkeypatch.setattr(rb, "RealisticBacktester", _BT)
    return _run_backtest_core("rank(close)", object(), {"close": None}, {"close": None})


class TestOverfitScore:

    def test_the_degradation_formula_matches_the_reference(self, monkeypatch):
        """
        `degradation = (is_sharpe - oos_sharpe) / abs(is_sharpe)`

        分子的 `-` 翻成 `+`、除法翻成乘法 —— 结果仍然落在 [0,1]
        （被 clip 吃掉），`is_overfit` 也可能碰巧一致。
        所以逐位比对，而不是只看布尔结论。
        """
        for is_s, oos_s in ((2.0, 0.5), (1.0, 0.9), (-1.5, -0.2),
                            (-2.0, -2.5), (0.8, 1.4)):
            out = _core(is_s, oos_s, monkeypatch)
            ref = float(np.clip((is_s - oos_s) / abs(is_s), 0.0, 1.0))
            assert out["overfitting_score"] == pytest.approx(ref, rel=1e-12), (
                f"IS={is_s} OOS={oos_s} 时过拟合分数是 "
                f"{out['overfitting_score']}，参考值 {ref}")

    def test_an_oos_better_than_is_scores_zero_not_negative(self, monkeypatch):
        """`np.clip(degradation, 0.0, 1.0)` 的下界 —— 负分会让排序反过来。"""
        out = _core(0.5, 2.0, monkeypatch)
        assert out["overfitting_score"] == 0.0
        assert out["is_overfit"] is False

    def test_a_total_collapse_is_capped_at_one(self, monkeypatch):
        """上界 —— IS 2.0 / OOS -10 的退化率是 6.0，必须压到 1.0。"""
        out = _core(2.0, -10.0, monkeypatch)
        assert out["overfitting_score"] == 1.0

    def test_an_is_sharpe_exactly_on_the_epsilon_short_circuits(self, monkeypatch):
        """
        **首测存活项（L176）**：`abs(is_sharpe) > 1e-9` 翻成 `>=`
        只在 `|is_sharpe|` **精确等于** 1e-9 时才有差别。

        `is_sharpe` 直接来自回测报告，所以桩里给多少就是多少 ——
        构造 1e-9 没有任何浮点误差（下面第一条断言守着这个前提）。

        原始 `>` 为假 → 短路到 0 分；`>=` 为真 → 退化率
        (1e-9 - (-5)) / 1e-9 ≈ 5e9 → clip 成 1.0，
        于是这条因子被判成"完全过拟合"。两者天差地别。
        """
        out = _core(1e-9, -5.0, monkeypatch)
        assert out["is_sharpe"] == 1e-9, (
            f"前提失效：is_sharpe 是 {out['is_sharpe']!r}，不再精确等于 1e-9")
        assert out["overfitting_score"] == 0.0, (
            "IS Sharpe 恰好等于 1e-9 时没有短路 —— `> 1e-9` 被翻成了 `>=`")
        assert out["is_overfit"] is False

        # 稍大一点就必须真的去算
        bigger = _core(np.nextafter(1e-9, np.inf), -5.0, monkeypatch)
        assert bigger["overfitting_score"] == 1.0, (
            "略大于 1e-9 时反而短路了 —— 守卫的方向被翻了")

    def test_a_near_zero_is_sharpe_short_circuits_to_zero(self, monkeypatch):
        """
        `if oos_sharpe is not None and abs(is_sharpe) > 1e-9:`

        守卫被删会让 IS Sharpe 为 0 时除以 0 → inf/nan，
        `is_overfit` 变成 NaN 比较（恒假），过拟合检测静默失效。
        """
        out = _core(0.0, -5.0, monkeypatch)
        assert out["overfitting_score"] == 0.0
        assert np.isfinite(out["overfitting_score"])

    def test_the_epsilon_guard_uses_the_absolute_value(self, monkeypatch):
        """
        `abs(is_sharpe) > 1e-9` —— 去掉 `abs` 会让**所有负 IS Sharpe**
        都走进短路分支，过拟合分数恒为 0。
        """
        # IS 与 OOS 都为负，且 OOS **更差** —— 这才是负 Sharpe 下的退化
        out = _core(-2.0, -2.5, monkeypatch)
        ref = float(np.clip((-2.0 - (-2.5)) / 2.0, 0.0, 1.0))
        assert ref == pytest.approx(0.25), "参考值构造错了，本用例失去区分力"
        assert out["overfitting_score"] == pytest.approx(ref, rel=1e-12)
        assert out["overfitting_score"] > 0.0, (
            "负的 IS Sharpe 被当成了零（短路到 0 分）—— "
            "`abs(is_sharpe) > 1e-9` 里的 abs 没了")

    def test_a_missing_oos_report_scores_zero(self, monkeypatch):
        out = _core(2.0, None, monkeypatch)
        assert out["oos_sharpe"] is None
        assert out["overfitting_score"] == 0.0
        assert out["is_overfit"] is False

    def test_is_overfit_flips_strictly_above_the_threshold(self, monkeypatch):
        """
        `overfit_score > _OVERFIT_THRESHOLD` —— `>` 翻成 `>=`
        只改变**恰好等于 0.5** 的那一格。

        构造：IS=2.0、OOS=1.0 → 退化率 (2-1)/2 = 0.5 精确成立
        （三个数都是二进制可精确表示的）。
        """
        out = _core(2.0, 1.0, monkeypatch)
        assert out["overfitting_score"] == _OVERFIT_THRESHOLD, (
            f"前提失效：构造出的分数是 {out['overfitting_score']!r}，"
            f"不再精确等于阈值 {_OVERFIT_THRESHOLD}")
        assert out["is_overfit"] is False, (
            "过拟合分数恰好等于阈值就被判为过拟合 —— `>` 被翻成了 `>=`")

        worse = _core(2.0, 0.9, monkeypatch)
        assert worse["overfitting_score"] > _OVERFIT_THRESHOLD
        assert worse["is_overfit"] is True

    def test_the_threshold_constant_is_one_half(self):
        assert _OVERFIT_THRESHOLD == 0.50


class TestBacktestCoreOutput:

    def test_every_documented_key_is_present(self, monkeypatch):
        out = _core(1.5, 1.0, monkeypatch)
        assert set(out) == {"is_sharpe", "oos_sharpe", "is_return", "is_turnover",
                            "is_ic", "overfitting_score", "is_overfit", "summary"}

    def test_metrics_are_read_from_the_in_sample_report(self, monkeypatch):
        """
        `_f(is_r.annualized_return)` 等三处 —— 取错报告（拿 OOS 的）
        会让"样本内表现"这一栏其实是样本外的，晋级判断全乱。
        用两份取值不同的报告来区分。
        """
        import app.core.backtest_engine.realistic_backtester as rb

        class _BT:
            def __init__(self, **k):
                pass

            def run(self, dsl, is_data, oos_dataset=None):
                return _Result(_Report(1.5, ann_return=0.11, turnover=2.2, ic=0.033),
                               _Report(0.5, ann_return=0.99, turnover=9.9, ic=0.999))

        monkeypatch.setattr(rb, "RealisticBacktester", _BT)
        out = _run_backtest_core("rank(close)", object(), {}, {})
        assert out["is_sharpe"] == pytest.approx(1.5)
        assert out["oos_sharpe"] == pytest.approx(0.5)
        assert out["is_return"] == pytest.approx(0.11), "is_return 取自了 OOS 报告"
        assert out["is_turnover"] == pytest.approx(2.2), "is_turnover 取自了 OOS 报告"
        assert out["is_ic"] == pytest.approx(0.033), "is_ic 取自了 OOS 报告"

    def test_a_nan_metric_becomes_none_rather_than_nan(self, monkeypatch):
        """
        `_f` 里的 `not (isinstance(v, float) and np.isnan(v))`
        —— `not` 被删会让 NaN 原样落进结果字典，
        序列化成 JSON 时变成非法的 `NaN` 字面量，前端解析直接炸。
        """
        out = _core(1.0, 0.5, monkeypatch, ann_return=float("nan"))
        assert out["is_return"] is None, (
            f"NaN 没有被折成 None：{out['is_return']!r}")

    def test_a_none_metric_stays_none(self, monkeypatch):
        out = _core(1.0, 0.5, monkeypatch, ic=None)
        assert out["is_ic"] is None

    def test_a_missing_is_sharpe_falls_back_to_zero_not_none(self, monkeypatch):
        """
        `is_sharpe = _f(is_r.sharpe_ratio) or 0.0`
        —— `or 0.0` 被删会让 `abs(None)` 直接 TypeError。
        """
        out = _core(None, 0.5, monkeypatch)
        assert out["is_sharpe"] == 0.0

    def test_the_summary_string_comes_from_the_backtest_result(self, monkeypatch):
        assert _core(1.0, 0.5, monkeypatch)["summary"] == "SUMMARY"

    def test_the_backtester_is_constructed_with_min_obs_zero(self, monkeypatch):
        """
        `RealisticBacktester(config=cfg, min_obs=0)` —— 源码注释写明
        这里是"内部搜索排序用，只标注不置空"。
        `min_obs` 被改成默认值会让短样本的指标被整体置空，
        GP 适应度全变 0，搜索退化成随机。
        """
        import app.core.backtest_engine.realistic_backtester as rb
        seen = {}

        class _BT:
            def __init__(self, config=None, min_obs=None):
                seen["config"] = config
                seen["min_obs"] = min_obs

            def run(self, dsl, is_data, oos_dataset=None):
                return _Result(_Report(1.0), _Report(0.8))

        monkeypatch.setattr(rb, "RealisticBacktester", _BT)
        cfg = object()
        _run_backtest_core("rank(close)", cfg, {}, {})
        assert seen["min_obs"] == 0, f"min_obs 传成了 {seen['min_obs']!r}"
        assert seen["config"] is cfg, "config 没有透传"

    def test_the_oos_dataset_is_forwarded_to_the_run_call(self, monkeypatch):
        import app.core.backtest_engine.realistic_backtester as rb
        seen = {}

        class _BT:
            def __init__(self, **k):
                pass

            def run(self, dsl, is_data, oos_dataset=None):
                seen.update(dsl=dsl, is_data=is_data, oos=oos_dataset)
                return _Result(_Report(1.0), _Report(0.8))

        monkeypatch.setattr(rb, "RealisticBacktester", _BT)
        is_d, oos_d = {"close": 1}, {"close": 2}
        _run_backtest_core("rank(vwap)", None, is_d, oos_d)
        assert seen["dsl"] == "rank(vwap)"
        assert seen["is_data"] is is_d
        assert seen["oos"] is oos_d, "OOS 数据没有传给回测器 —— 样本外评估形同虚设"
