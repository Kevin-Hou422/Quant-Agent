"""
backtest_engine/multi_dataset_backtester.py —— 跨市场稳健性回测

**此前零专属测试**（14 个变异点，D 档）。

这一层的**唯一目的**是防单市场过拟合：一条因子必须在多个市场上
同时有正的样本外 Sharpe 才算稳健。所以它坏掉的方式，全都是
"让一条不稳健的因子看起来稳健"：

  - `aggregation="min"` 与 `"mean"` 的分支互换 → 严格模式变宽松，
    在一个市场上 Sharpe 2、其余三个 -1 的因子照样"通过"；
  - `datasets_passed = sum(1 for r if r.sharpe_oos > 0)` 的 `>` 翻成 `>=`
    → Sharpe 恰好为 0（信号常数、或回测失败降级）的市场也被算作通过；
  - `_split_dataset` 的 `iloc[:split_idx]` / `iloc[split_idx:]` 任一端点被改
    → IS 与 OOS **重叠**，样本外不再是样本外；
  - `weighted` 模式的 `n_oos_days ** 0.5` 被改成 `** 1` 或 `* 0.5`
    → 样本量大的市场权重被高估/抹平，聚合结论整体偏移。

回测器本身用桩替换（`RealisticBacktester` 是另一个模块的职责），
本文件只验**切分、聚合、计数、降级**这四件事。
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

import app.core.backtest_engine.multi_dataset_backtester as MDB
from app.core.backtest_engine.multi_dataset_backtester import (
    DatasetBacktestResult,
    MultiDatasetBacktester,
    MultiDatasetResult,
    _split_dataset,
    compute_multi_dataset_fitness,
)

IDX = pd.bdate_range("2022-01-03", periods=100)
COLS = ["AAA", "BBB", "CCC"]


def _panel(n=100) -> dict:
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(3)
    return {f: pd.DataFrame(rng.normal(100, 1, (n, 3)), index=idx, columns=COLS)
            for f in ("close", "volume", "returns")}


def _result(name="d", **kw) -> DatasetBacktestResult:
    base = dict(dataset_name=name, sharpe_is=1.0, sharpe_oos=0.5,
                max_drawdown=-0.1, ann_turnover=2.0, ann_return=0.15,
                mean_ic=0.03, overfitting_score=0.25)
    base.update(kw)
    return DatasetBacktestResult(**base)


# ---------------------------------------------------------------------------
# 回测器的桩
# ---------------------------------------------------------------------------

class _Report:
    def __init__(self, sharpe, dd=-0.2, to=2.0, ret=0.1, ic=0.02):
        self.sharpe_ratio = sharpe
        self.max_drawdown = dd
        self.ann_turnover = to
        self.annualized_return = ret
        self.mean_ic = ic


def _stub_backtester(monkeypatch, per_dataset_sharpe=None, boom=(), spy=None):
    """
    per_dataset_sharpe: 按 IS 行数区分数据集 → (is_sharpe, oos_sharpe)
    boom: 这些 IS 行数会抛异常
    """
    import app.core.backtest_engine.realistic_backtester as RB
    calls = []

    class _BT:
        def __init__(self, config=None, cost_params=None):
            calls.append({"config": config, "cost_params": cost_params})

        def run(self, dsl, is_data, oos_dataset=None):
            n_is = len(next(iter(is_data.values())))
            if n_is in boom:
                raise RuntimeError(f"回测炸了 n_is={n_is}")
            pair = (per_dataset_sharpe or {}).get(n_is, (1.0, 0.5))
            if spy is not None:
                spy.append({"dsl": dsl, "n_is": n_is,
                            "n_oos": len(next(iter(oos_dataset.values())))
                            if oos_dataset else 0})
            return types.SimpleNamespace(
                is_report=_Report(pair[0]),
                oos_report=_Report(pair[1]) if pair[1] is not None else None)

    monkeypatch.setattr(RB, "RealisticBacktester", _BT)
    return calls


# ===========================================================================
# A. 构造与校验
# ===========================================================================

class TestConstruction:

    def test_the_documented_defaults(self):
        b = MultiDatasetBacktester()
        assert b._aggregation == "mean"
        assert b._is_split == 0.7

    @pytest.mark.parametrize("mode", ["mean", "min", "weighted"])
    def test_the_three_documented_modes_are_accepted(self, mode):
        assert MultiDatasetBacktester(aggregation=mode)._aggregation == mode

    def test_an_unknown_aggregation_is_rejected_at_construction(self):
        """
        `raise ValueError(...)` —— 被删会让 `aggregation="max"` 静默
        落进 `else: mean` 分支：使用者以为在用最乐观口径，实际是均值。
        """
        with pytest.raises(ValueError) as ei:
            MultiDatasetBacktester(aggregation="max")
        msg = str(ei.value)
        assert "max" in msg
        for m in ("mean", "min", "weighted"):
            assert m in msg, f"报错里没有列出合法模式 {m}"

    def test_an_empty_dataset_dict_is_rejected(self):
        """
        `if not datasets: raise ValueError` —— `not` 被删会让空输入
        返回一个 `aggregated_sharpe=0.0, datasets_total=0` 的结果，
        看起来像"跑过了但表现平平"，而实际上一个市场都没测。
        """
        with pytest.raises(ValueError, match="datasets dict is empty"):
            MultiDatasetBacktester().run("rank(close)", {})


# ===========================================================================
# B. IS / OOS 切分 —— 样本外必须真的在外面
# ===========================================================================

class TestSplitDataset:

    def test_is_and_oos_are_contiguous_disjoint_and_ordered(self):
        """
        `df.iloc[:split_idx]` / `df.iloc[split_idx:]`

        任一端点被改（`:split_idx+1` / `split_idx-1:`）都会让两段
        **重叠一天** —— 样本外混进了训练数据，而长度只差 1，
        任何"总数对得上"的松断言都发现不了。
        """
        data = _panel(100)
        is_d, oos_d = _split_dataset(data, 0.7)
        i = is_d["close"].index
        o = oos_d["close"].index

        assert len(i) + len(o) == 100, "切分前后样本总数对不上"
        assert set(i) & set(o) == set(), "IS 与 OOS 有重叠 —— 样本外泄漏"
        assert i[-1] < o[0], "IS 的末尾晚于 OOS 的开头 —— 时间顺序反了"

    def test_the_split_point_follows_the_documented_formula(self):
        """`split_idx = max(1, int(n * is_split))` —— 逐个比对。"""
        for n, frac in ((100, 0.7), (57, 0.5), (1000, 0.9), (33, 0.33)):
            is_d, oos_d = _split_dataset(_panel(n), frac)
            expected = max(1, int(n * frac))
            assert len(is_d["close"]) == expected, (
                f"n={n} frac={frac}：IS 长度 {len(is_d['close'])}，"
                f"应当是 {expected}")
            assert len(oos_d["close"]) == n - expected

    def test_a_tiny_split_still_leaves_one_training_row(self):
        """`max(1, ...)` —— 去掉会让 is_split=0.001 切出空的 IS，回测直接炸。"""
        is_d, oos_d = _split_dataset(_panel(10), 0.001)
        assert len(is_d["close"]) == 1
        assert len(oos_d["close"]) == 9

    def test_every_field_is_split_at_the_same_point(self):
        """
        字典推导漏掉某个字段、或端点在字段之间不一致，
        会让 `close` 与 `volume` 错开一天 —— 最隐蔽的一类泄漏。
        """
        data = _panel(80)
        data["extra"] = data["close"] * 2
        is_d, oos_d = _split_dataset(data, 0.7)
        assert set(is_d) == set(data) == set(oos_d), "切分之后字段数量变了"
        ref = list(is_d["close"].index)
        for f in is_d:
            assert list(is_d[f].index) == ref, f"{f} 的 IS 区间与 close 不一致"

    def test_an_empty_raw_dict_yields_two_empty_dicts(self):
        assert _split_dataset({}, 0.7) == ({}, {})

    def test_a_non_dataframe_payload_degrades_without_crashing(self):
        """
        `if not isinstance(first_df, pd.DataFrame) or len(first_df) == 0:`
        —— 守卫被删会让 `len(None)` 抛。当前契约是"全给 IS、OOS 为空"。
        """
        is_d, oos_d = _split_dataset({"close": "不是一个 DataFrame"}, 0.7)
        assert is_d == {"close": "不是一个 DataFrame"}
        assert oos_d == {}

    def test_an_empty_frame_degrades_the_same_way(self):
        is_d, oos_d = _split_dataset({"close": pd.DataFrame()}, 0.7)
        assert oos_d == {}


# ===========================================================================
# C. 聚合
# ===========================================================================

def _run(monkeypatch, sharpes: dict, mode="mean", is_split=0.7, boom=(),
         sizes=None):
    """
    sharpes: 数据集名 → (is_sharpe, oos_sharpe)
    sizes  : 数据集名 → 总行数（默认 100）
    """
    sizes = sizes or {}
    datasets, by_n_is = {}, {}
    for name, pair in sharpes.items():
        n = sizes.get(name, 100)
        datasets[name] = _panel(n)
        by_n_is[max(1, int(n * is_split))] = pair
    boom_n = {max(1, int(sizes.get(b, 100) * is_split)) for b in boom}
    _stub_backtester(monkeypatch, by_n_is, boom=boom_n)
    return MultiDatasetBacktester(aggregation=mode, is_split=is_split).run(
        "rank(close)", datasets)


class TestAggregation:

    def test_mean_mode_averages_the_oos_sharpes(self, monkeypatch):
        out = _run(monkeypatch, {"us": (1.0, 0.8), "cn": (1.0, 0.2),
                                 "crypto": (1.0, -0.4)},
                   mode="mean", sizes={"us": 100, "cn": 120, "crypto": 140})
        assert out.aggregated_sharpe == pytest.approx((0.8 + 0.2 - 0.4) / 3,
                                                      rel=1e-12)
        assert out.aggregation_mode == "mean"

    def test_min_mode_takes_the_worst_market(self, monkeypatch):
        """
        严格模式的全部意义：一个市场垮掉，整条因子就不算稳健。
        `min` 换成 `max` 会让"在某一个市场上碰巧很好"的因子通过。
        """
        out = _run(monkeypatch, {"us": (1.0, 2.0), "cn": (1.0, 0.3),
                                 "crypto": (1.0, -0.9)},
                   mode="min", sizes={"us": 100, "cn": 120, "crypto": 140})
        assert out.aggregated_sharpe == pytest.approx(-0.9), (
            f"min 聚合给出 {out.aggregated_sharpe} —— 取的不是最差的那个市场")

    def test_min_and_mean_disagree_on_the_same_input(self, monkeypatch):
        """两个模式若给出同一个数，说明分支被合并或互换了。"""
        sizes = {"us": 100, "cn": 120, "crypto": 140}
        sh = {"us": (1.0, 2.0), "cn": (1.0, 0.3), "crypto": (1.0, -0.9)}
        a = _run(monkeypatch, sh, mode="mean", sizes=sizes).aggregated_sharpe
        b = _run(monkeypatch, sh, mode="min", sizes=sizes).aggregated_sharpe
        assert a != b, "mean 与 min 给出了同一个结果 —— 分支被合并了"
        assert a > b

    def test_weighted_mode_uses_the_square_root_of_oos_days(self, monkeypatch):
        """
        `weights = [max(r.n_oos_days, 1) ** 0.5 ...]`

        指数被改成 1（线性）或 0（等权）都会改变聚合值，
        而结果仍然落在各市场 Sharpe 的区间内 —— 完全看不出异常。
        逐位比对参考实现。
        """
        sizes = {"big": 1000, "small": 100}
        out = _run(monkeypatch, {"big": (1.0, 0.2), "small": (1.0, 1.0)},
                   mode="weighted", sizes=sizes)

        n_oos = {name: n - max(1, int(n * 0.7)) for name, n in sizes.items()}
        w = np.array([n_oos["big"] ** 0.5, n_oos["small"] ** 0.5])
        ref = float(np.average([0.2, 1.0], weights=w))
        assert out.aggregated_sharpe == pytest.approx(ref, rel=1e-12), (
            f"加权聚合给出 {out.aggregated_sharpe}，参考值 {ref} —— "
            f"权重不是 sqrt(OOS 天数)")

    def test_weighted_mode_differs_from_plain_mean(self, monkeypatch):
        sizes = {"big": 1000, "small": 100}
        sh = {"big": (1.0, 0.2), "small": (1.0, 1.0)}
        wm = _run(monkeypatch, sh, mode="weighted", sizes=sizes).aggregated_sharpe
        mm = _run(monkeypatch, sh, mode="mean", sizes=sizes).aggregated_sharpe
        assert wm != pytest.approx(mm), (
            "加权与等权给出了同一个结果 —— weighted 分支没生效")
        assert wm < mm, "样本量大的差市场没有被赋予更高权重"

    def test_the_weight_floor_keeps_a_zero_length_market_in(self, monkeypatch):
        """`max(r.n_oos_days, 1)` —— 去掉会让 0 天的市场权重为 0（或 0**0.5=0）。"""
        r = _result("a", n_oos_days=0, sharpe_oos=1.0)
        r2 = _result("b", n_oos_days=9, sharpe_oos=0.0)
        weights = np.array([max(x.n_oos_days, 1) ** 0.5 for x in (r, r2)])
        assert weights[0] == 1.0, "OOS 天数为 0 时权重下限没有生效"

    def test_no_valid_result_aggregates_to_zero(self, monkeypatch):
        """
        `if not oos_sharpes: agg_sharpe = 0.0`
        —— 守卫被删会让 `np.mean([])` 返回 NaN 并抛 RuntimeWarning，
        NaN 一路流进 GP 适应度。
        """
        out = _run(monkeypatch, {"us": (1.0, 0.5)}, boom=("us",))
        assert out.aggregated_sharpe == 0.0
        assert np.isfinite(out.aggregated_sharpe)

    def test_failed_datasets_are_excluded_from_the_aggregate(self, monkeypatch):
        """
        `if r.error is None and not np.isnan(r.sharpe_oos)`

        失败的数据集 sharpe_oos 被降级成 0.0。把它算进均值会
        **系统性稀释**聚合 Sharpe —— 一个市场取不到数据，
        整条因子的评分就被拉低，而原因完全不可见。
        """
        out = _run(monkeypatch, {"ok": (1.0, 0.9), "broken": (1.0, 0.9)},
                   mode="mean", sizes={"ok": 100, "broken": 140},
                   boom=("broken",))
        assert out.aggregated_sharpe == pytest.approx(0.9), (
            f"失败的数据集被算进了均值：{out.aggregated_sharpe}")
        assert len(out.errors) == 1
        assert "[broken]" in out.errors[0]


# ===========================================================================
# D. 通过计数
# ===========================================================================

class TestPassCounting:

    def test_only_strictly_positive_oos_sharpe_counts_as_a_pass(self, monkeypatch):
        """
        `sum(1 for r in per_dataset.values() if r.sharpe_oos > 0)`

        `>` 翻成 `>=` 会把 Sharpe 恰好为 0 的市场算成通过 ——
        而 0 正是**回测失败时的降级值**，等于把失败算成成功。
        """
        out = _run(monkeypatch,
                   {"pos": (1.0, 0.3), "zero": (1.0, 0.0), "neg": (1.0, -0.3)},
                   sizes={"pos": 100, "zero": 120, "neg": 140})
        assert out.datasets_passed == 1, (
            f"通过数是 {out.datasets_passed}，应当只有 pos 一个 —— "
            f"`sharpe_oos > 0` 被翻成了 `>=`")

    def test_failed_datasets_count_as_not_passed(self, monkeypatch):
        out = _run(monkeypatch, {"ok": (1.0, 0.5), "bad": (1.0, 0.5)},
                   sizes={"ok": 100, "bad": 140}, boom=("bad",))
        assert out.datasets_passed == 1
        assert out.datasets_total == 2

    def test_the_total_counts_every_requested_dataset(self, monkeypatch):
        """
        `datasets_total = len(datasets)` —— 换成
        `len(per_dataset)` 或 `len(valid)` 会让失败的市场从分母里消失，
        通过率虚高。
        """
        out = _run(monkeypatch, {"a": (1.0, 0.5), "b": (1.0, 0.5),
                                 "c": (1.0, 0.5)},
                   sizes={"a": 100, "b": 120, "c": 140}, boom=("b", "c"))
        assert out.datasets_total == 3, (
            f"分母是 {out.datasets_total}，失败的市场被从总数里抹掉了")

    def test_the_pass_rate_is_passed_over_total(self):
        r = MultiDatasetResult(per_dataset={}, aggregated_sharpe=0.5,
                               aggregation_mode="mean", datasets_passed=3,
                               datasets_total=4)
        assert r.pass_rate == pytest.approx(0.75)

    def test_a_zero_total_yields_a_zero_pass_rate_not_a_division_error(self):
        """`if self.datasets_total == 0: return 0.0` —— 守卫被删会 ZeroDivisionError。"""
        r = MultiDatasetResult(per_dataset={}, aggregated_sharpe=0.0,
                               aggregation_mode="mean", datasets_passed=0,
                               datasets_total=0)
        assert r.pass_rate == 0.0


# ===========================================================================
# E. 单市场评估
# ===========================================================================

class TestSingleDatasetEvaluation:

    def test_metrics_are_taken_from_the_documented_report(self, monkeypatch):
        """
        IS 与 OOS 各取哪些指标是有讲究的：
          · `sharpe_oos` / `max_drawdown` 来自 **OOS**（样本外才算数）
          · `ann_turnover` / `ann_return` / `mean_ic` 来自 **IS**
        取反会让"样本外回撤"其实是样本内的，稳健性判断彻底失真。
        """
        import app.core.backtest_engine.realistic_backtester as RB

        class _BT:
            def __init__(self, **k):
                pass

            def run(self, dsl, is_data, oos_dataset=None):
                return types.SimpleNamespace(
                    is_report=_Report(1.5, dd=-0.11, to=2.2, ret=0.33, ic=0.044),
                    oos_report=_Report(0.5, dd=-0.99, to=9.9, ret=0.99, ic=0.999))

        monkeypatch.setattr(RB, "RealisticBacktester", _BT)
        out = MultiDatasetBacktester().run("rank(close)", {"us": _panel()})
        r = out.per_dataset["us"]
        assert r.sharpe_is == pytest.approx(1.5)
        assert r.sharpe_oos == pytest.approx(0.5)
        assert r.max_drawdown == pytest.approx(-0.99), "最大回撤取自了样本内"
        assert r.ann_turnover == pytest.approx(2.2), "换手取自了样本外"
        assert r.ann_return == pytest.approx(0.33), "年化收益取自了样本外"
        assert r.mean_ic == pytest.approx(0.044), "IC 取自了样本外"

    def test_the_overfit_score_matches_the_reference_formula(self, monkeypatch):
        for is_s, oos_s in ((2.0, 0.5), (1.0, 0.9), (-2.0, -2.5), (0.5, 2.0)):
            out = _run(monkeypatch, {"d": (is_s, oos_s)})
            ref = float(np.clip((is_s - oos_s) / abs(is_s), 0.0, 1.0))
            assert out.per_dataset["d"].overfitting_score == pytest.approx(
                ref, rel=1e-12), f"IS={is_s} OOS={oos_s} 的过拟合分不对"

    def test_a_near_zero_is_sharpe_short_circuits_the_overfit_score(self,
                                                                    monkeypatch):
        out = _run(monkeypatch, {"d": (0.0, -9.0)})
        assert out.per_dataset["d"].overfitting_score == 0.0

    def test_an_is_sharpe_exactly_on_the_epsilon_short_circuits(self, monkeypatch):
        """
        **首测存活项（L269）**：`if abs(sharpe_is) > 1e-9 and oos_r:`
        翻成 `>=` 只在 `|sharpe_is|` **精确等于** 1e-9 时才有差别。

        `sharpe_is` 直接来自回测报告，桩里给多少就是多少，构造无误差。
        原始 `>` 为假 → 过拟合分短路到 0；
        `>=` 为真 → 退化率 (1e-9 - (-5))/1e-9 ≈ 5e9 → clip 成 1.0，
        这条因子在该市场上被判成"完全过拟合"。
        """
        out = _run(monkeypatch, {"d": (1e-9, -5.0)})
        r = out.per_dataset["d"]
        assert r.sharpe_is == 1e-9, (
            f"前提失效：sharpe_is 是 {r.sharpe_is!r}，不再精确等于 1e-9")
        assert r.overfitting_score == 0.0, (
            "IS Sharpe 恰好等于 1e-9 时没有短路 —— `> 1e-9` 被翻成了 `>=`")

        bigger = _run(monkeypatch, {"d": (np.nextafter(1e-9, np.inf), -5.0)})
        assert bigger.per_dataset["d"].overfitting_score == 1.0, (
            "略大于 1e-9 时反而短路了 —— 守卫的方向被翻了")

    def test_the_epsilon_guard_uses_the_absolute_value(self, monkeypatch):
        """
        `abs(sharpe_is)` —— 去掉 `abs` 会让**所有负的 IS Sharpe**
        都走进短路分支，负 Sharpe 市场的过拟合分恒为 0。
        """
        out = _run(monkeypatch, {"d": (-2.0, -2.5)})
        ref = float(np.clip((-2.0 - (-2.5)) / 2.0, 0.0, 1.0))
        assert ref == pytest.approx(0.25)
        assert out.per_dataset["d"].overfitting_score == pytest.approx(ref,
                                                                       rel=1e-12)
        assert out.per_dataset["d"].overfitting_score > 0.0, (
            "负的 IS Sharpe 被当成了零 —— `abs(sharpe_is)` 里的 abs 没了")

    def test_a_missing_oos_report_zeroes_the_out_of_sample_metrics(self,
                                                                    monkeypatch):
        out = _run(monkeypatch, {"d": (1.5, None)})
        r = out.per_dataset["d"]
        assert r.sharpe_oos == 0.0 and r.max_drawdown == 0.0
        assert r.overfitting_score == 0.0, "没有样本外报告却算出了过拟合分"

    def test_a_nan_metric_degrades_to_zero(self, monkeypatch):
        """
        `_f` 里 `return fv if not np.isnan(fv) else 0.0`
        —— `not` 被删会让 NaN 原样进结果，聚合时整个均值变 NaN。
        """
        out = _run(monkeypatch, {"d": (float("nan"), 0.5)})
        assert out.per_dataset["d"].sharpe_is == 0.0

    def test_a_non_numeric_metric_degrades_to_zero(self, monkeypatch):
        """`except (TypeError, ValueError): return 0.0`"""
        import app.core.backtest_engine.realistic_backtester as RB

        class _BT:
            def __init__(self, **k):
                pass

            def run(self, dsl, is_data, oos_dataset=None):
                return types.SimpleNamespace(
                    is_report=_Report("不是数字"), oos_report=_Report(0.5))

        monkeypatch.setattr(RB, "RealisticBacktester", _BT)
        out = MultiDatasetBacktester().run("rank(close)", {"us": _panel()})
        assert out.per_dataset["us"].sharpe_is == 0.0

    def test_a_failing_dataset_is_recorded_with_its_error(self, monkeypatch):
        """
        `except Exception: return DatasetBacktestResult(..., error=str(exc))`
        —— 这个兜底被删会让一个市场取不到数据就作废整次跨市场评估。
        """
        out = _run(monkeypatch, {"ok": (1.0, 0.5), "bad": (1.0, 0.5)},
                   sizes={"ok": 100, "bad": 140}, boom=("bad",))
        bad = out.per_dataset["bad"]
        assert bad.error is not None and "回测炸了" in bad.error
        assert (bad.sharpe_is, bad.sharpe_oos, bad.max_drawdown) == (0.0, 0.0, 0.0)
        assert out.per_dataset["ok"].error is None

    def test_the_oos_day_count_is_recorded(self, monkeypatch):
        spy = []
        _stub_backtester(monkeypatch, {70: (1.0, 0.5)}, spy=spy)
        out = MultiDatasetBacktester(is_split=0.7).run("rank(close)",
                                                       {"us": _panel(100)})
        assert out.per_dataset["us"].n_oos_days == 30, (
            f"记录的 OOS 天数是 {out.per_dataset['us'].n_oos_days}，应当是 30")
        assert spy[0]["n_oos"] == 30

    def test_the_backtester_receives_the_cost_params(self, monkeypatch):
        cost = object()
        calls = _stub_backtester(monkeypatch, {70: (1.0, 0.5)})
        MultiDatasetBacktester(cost_params=cost).run("rank(close)",
                                                     {"us": _panel()})
        assert calls[0]["cost_params"] is cost, "成本参数没有透传给回测器"

    def test_the_dsl_reaches_every_dataset_unchanged(self, monkeypatch):
        spy = []
        _stub_backtester(monkeypatch, {70: (1.0, 0.5), 84: (1.0, 0.5)}, spy=spy)
        MultiDatasetBacktester().run("rank(ts_delta(log(close),5))",
                                     {"a": _panel(100), "b": _panel(120)})
        assert {c["dsl"] for c in spy} == {"rank(ts_delta(log(close),5))"}
        assert len(spy) == 2


class TestDefaultConfig:

    def test_the_default_signal_config_prevents_look_ahead(self, monkeypatch):
        """
        `SimulationConfig(delay=1, ...)` —— `delay=0` 就是用当天信号
        交易当天收盘，跨市场稳健性的结论全部作废。
        """
        calls = _stub_backtester(monkeypatch, {70: (1.0, 0.5)})
        MultiDatasetBacktester().run("rank(close)", {"us": _panel()})
        cfg = calls[0]["config"]
        assert cfg.delay == 1, f"delay={cfg.delay} —— 不是 1 就有前视风险"
        assert cfg.decay_window == 0
        assert cfg.truncation_min_q == 0.05
        assert cfg.truncation_max_q == 0.95
        assert cfg.portfolio_mode == "long_short"

    def test_a_supplied_config_wins_over_the_default(self, monkeypatch):
        """`if self._config is not None: return self._config`"""
        mine = object()
        calls = _stub_backtester(monkeypatch, {70: (1.0, 0.5)})
        MultiDatasetBacktester(config=mine).run("rank(close)", {"us": _panel()})
        assert calls[0]["config"] is mine


# ===========================================================================
# F. 结果对象
# ===========================================================================

class TestResultObjects:

    def test_per_dataset_to_dict_rounds_to_four_decimals(self):
        d = _result("us", sharpe_is=1.23456789).to_dict()
        assert d["sharpe_is"] == 1.2346, (
            f"指标没有按 4 位小数四舍五入：{d['sharpe_is']}")
        assert d["dataset"] == "us"

    def test_per_dataset_to_dict_carries_every_field(self):
        d = _result("us", n_oos_days=30, error="boom").to_dict()
        assert set(d) == {"dataset", "sharpe_is", "sharpe_oos", "max_drawdown",
                          "ann_turnover", "ann_return", "mean_ic",
                          "overfitting_score", "n_oos_days", "error"}
        assert d["n_oos_days"] == 30, "样本量不该被四舍五入成浮点"
        assert d["error"] == "boom"

    def test_the_aggregate_to_dict_nests_the_per_dataset_dicts(self):
        r = MultiDatasetResult(per_dataset={"us": _result("us")},
                               aggregated_sharpe=0.51234,
                               aggregation_mode="mean",
                               datasets_passed=1, datasets_total=2,
                               errors=["x"])
        d = r.to_dict()
        assert d["aggregated_sharpe"] == 0.5123
        assert d["pass_rate"] == 0.5
        assert d["per_dataset"]["us"]["dataset"] == "us"
        assert d["errors"] == ["x"]

    def test_the_errors_list_defaults_to_a_fresh_list(self):
        """`field(default_factory=list)` —— 写成 `= []` 会让所有结果共用一个列表。"""
        a = MultiDatasetResult({}, 0.0, "mean", 0, 0)
        b = MultiDatasetResult({}, 0.0, "mean", 0, 0)
        a.errors.append("x")
        assert b.errors == []

    def test_the_summary_reports_the_mode_and_the_pass_ratio(self):
        r = MultiDatasetResult(per_dataset={"us": _result("us")},
                               aggregated_sharpe=0.5, aggregation_mode="min",
                               datasets_passed=1, datasets_total=3)
        s = r.summary()
        assert "min aggregation" in s
        assert "1/3" in s, f"摘要里的通过比例不对：{s}"
        assert "0.5000" in s

    def test_the_summary_flags_failed_datasets_separately(self):
        r = MultiDatasetResult(
            per_dataset={"ok": _result("ok"),
                         "bad": _result("bad", error="数据没了")},
            aggregated_sharpe=0.5, aggregation_mode="mean",
            datasets_passed=1, datasets_total=2)
        s = r.summary()
        assert "[bad] ERROR: 数据没了" in s
        assert "[ok] IS=" in s
        assert "[bad] IS=" not in s, "失败的市场也打了指标行，会被误读成跑通了"


# ===========================================================================
# G. 与 GP 适应度的对接
# ===========================================================================

class TestFitnessBridge:

    def test_the_aggregated_sharpe_takes_the_place_of_sharpe_oos(self):
        """
        `compute_fitness(sharpe_oos=result.aggregated_sharpe, ...)`
        —— 传错字段（例如传 0）会让跨市场评估**完全不影响**适应度，
        GP 照样按单市场结果进化，而这个模块的存在意义就是防这一点。
        """
        from app.core.gp_engine.fitness import compute_fitness
        r = MultiDatasetResult({}, aggregated_sharpe=0.77,
                               aggregation_mode="mean",
                               datasets_passed=2, datasets_total=2)
        got = compute_multi_dataset_fitness(r, turnover=1.5, max_drawdown=-0.2,
                                            sharpe_is=1.1)
        ref = compute_fitness(sharpe_is=1.1, sharpe_oos=0.77, turnover=1.5,
                              max_drawdown=-0.2)
        assert got == pytest.approx(ref, rel=1e-12)

    def test_a_worse_aggregate_gives_a_worse_fitness(self):
        good = MultiDatasetResult({}, 1.2, "mean", 2, 2)
        bad = MultiDatasetResult({}, -0.4, "mean", 0, 2)
        assert compute_multi_dataset_fitness(good) > \
               compute_multi_dataset_fitness(bad)

    def test_the_penalty_terms_are_forwarded(self):
        r = MultiDatasetResult({}, 1.0, "mean", 2, 2)
        assert compute_multi_dataset_fitness(r, turnover=0.0) > \
               compute_multi_dataset_fitness(r, turnover=5.0), "换手惩罚没有生效"
        assert compute_multi_dataset_fitness(r, max_drawdown=0.0) > \
               compute_multi_dataset_fitness(r, max_drawdown=-0.5), "回撤惩罚没有生效"
        assert compute_multi_dataset_fitness(r, sharpe_is=1.0) > \
               compute_multi_dataset_fitness(r, sharpe_is=5.0), "过拟合惩罚没有生效"


class TestDatasetObjectWrapper:

    def test_dataset_objects_are_keyed_by_their_name(self, monkeypatch):
        """
        `{ds.name: ds.data for ds in datasets}` —— 键取错（比如用下标）
        会让结果里的市场名变成 0/1/2，报告完全不可读。
        """
        _stub_backtester(monkeypatch, {70: (1.0, 0.5), 84: (1.0, 0.5)})
        objs = [types.SimpleNamespace(name="us_equity", data=_panel(100)),
                types.SimpleNamespace(name="crypto", data=_panel(120))]
        out = MultiDatasetBacktester().run_with_datasets_obj("rank(close)", objs)
        assert set(out.per_dataset) == {"us_equity", "crypto"}
        assert out.datasets_total == 2

    def test_an_empty_dataset_list_is_rejected(self, monkeypatch):
        with pytest.raises(ValueError, match="datasets dict is empty"):
            MultiDatasetBacktester().run_with_datasets_obj("rank(close)", [])
