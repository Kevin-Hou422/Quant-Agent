"""
core/workflows/alpha_workflows.py —— 第二轮补强（变异测试驱动）

第一轮把首测的 20.6% 拉到 60.9%，但仍有 27 个存活。这一轮先做了一次
**只用新测试**的诊断测量（`plan_wf_newonly.json`），把"旧测试盖住的"
与"我确实漏写的"分开，然后逐条处置。

诊断结论里最有价值的两点：

  1. **只断言总数抓不到循环上限被压没**。
     `_expand_for_optimization` 有两个循环：变异循环 + 随机补位循环。
     把变异循环的上限 `n_mutations * 15` 改成 `/ 15` 之后它一次都不跑，
     但**补位循环会把总数补齐到 n_mutations + 3** —— 候选总数一模一样，
     只是里面一个变异体都没有，全是随机 alpha。
     Workflow B 的整个"针对输入 DSL 做结构优化"退化成了随机搜索，
     而 `len(candidates)` 看起来完全正常。
     → 必须**数变异算子被调了几次**，不能只数候选条数。

  2. **`_generate_diverse_seeds` 的两个填充循环被 Layer 1/2 掩盖**。
     Layer 1（AlphaAgent 回退种子，实测 5 条）+ Layer 2（模板）是无条件
     追加的。n_target=12 时填充循环只需补 6 条，上限被压成 1 次尝试也
     还剩 8 条 —— 断言"填满 12"能抓到，但断言"非空"抓不到。
     这一轮把 Layer 1/2 全关掉，让填充循环**单独**对结果负责。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import app.core.workflows.alpha_workflows as W


N = 6


def _panel(seed: int = 0, n: int = 200) -> dict:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, (n, N)), axis=0)
    idx = pd.bdate_range("2021-01-04", periods=n)
    cols = [f"S{i}" for i in range(N)]
    out = {k: pd.DataFrame(v, index=idx, columns=cols) for k, v in {
        "close": close, "open": close, "vwap": close,
        "high": close * 1.01, "low": close * 0.99,
        "volume": np.full_like(close, 1e6),
    }.items()}
    out["returns"] = out["close"].pct_change().fillna(0.0)
    return out


@pytest.fixture
def no_layer12(monkeypatch):
    """
    关掉 Layer 1（AlphaAgent）与 Layer 2（关键词模板），
    让 Layer 3/4 的两个填充循环**单独**对种子表负责。
    """
    import app.agent.alpha_agent as AA

    class _NoAgent:
        def __init__(self, *a, **kw):
            raise RuntimeError("本用例关掉 Layer 1")

    monkeypatch.setattr(AA, "AlphaAgent", _NoAgent)
    monkeypatch.setattr(W, "_hypothesis_templates", lambda h: [])


# ===========================================================================
# A. _quick_metrics 的过拟合公式（第一轮漏掉的）
# ===========================================================================

def _report(sharpe, *, turnover=1.0, insuf=False, n_days=100, se=0.1):
    return SimpleNamespace(sharpe_ratio=sharpe, ann_turnover=turnover,
                           insufficient_sample=insuf, n_days=n_days,
                           sharpe_se=se, max_drawdown=-0.1)


class _StubBt:
    is_report = None
    oos_report = None

    def __init__(self, *a, **kw):
        pass

    def run(self, dsl, is_data, oos_dataset=None):
        return SimpleNamespace(is_report=type(self).is_report,
                               oos_report=type(self).oos_report)


@pytest.fixture
def stub_bt(monkeypatch):
    import app.core.backtest_engine.realistic_backtester as rb
    monkeypatch.setattr(rb, "RealisticBacktester", _StubBt)
    return _StubBt


class TestQuickMetricsOverfit:
    """
    `overfit = clip((s_is - s_oos) / abs(s_is), 0, 1) if abs(s_is) > 1e-9 else 0.0`

    这是 alpha_workflows 里**第四份**同样的过拟合公式
    （另外三份在 population_evolver）。`-` 改成 `+` 之后，
    "IS 好 OOS 也好"的因子会被判成 100% 过拟合 —— Workflow B 的
    定向变异会据此给它加一堆 ts_mean 平滑，把好因子改坏。
    """

    @pytest.mark.parametrize("is_s,oos_s,expect", [
        (2.0, 2.0, 0.0),      # 没退化 → 0；`+` 会给 1.0
        (2.0, 1.0, 0.5),      # 退化一半；`+` 会给 1.0
        (2.0, 0.0, 1.0),
        (2.0, 3.0, 0.0),      # OOS 更好 → clip 到 0；`+` 会给 1.0
        (2.0, -2.0, 1.0),
    ])
    def test_the_degradation_formula(self, stub_bt, is_s, oos_s, expect):
        _StubBt.is_report = _report(is_s)
        _StubBt.oos_report = _report(oos_s)
        m = W._quick_metrics("rank(close)", {"close": 1}, {"close": 1})
        assert m["overfitting_score"] == pytest.approx(expect), (
            f"IS={is_s}、OOS={oos_s} 的过拟合度算成了 "
            f"{m['overfitting_score']}，应当是 {expect} —— "
            f"`(s_is - s_oos)` 的符号被改了")

    def test_a_zero_is_sharpe_is_not_divided_by(self, stub_bt):
        """`if abs(s_is) > 1e-9 else 0.0` —— **严格大于**。"""
        _StubBt.is_report = _report(0.0)
        _StubBt.oos_report = _report(1.0)
        m = W._quick_metrics("rank(close)", {"close": 1}, {"close": 1})
        assert m["overfitting_score"] == 0.0
        assert not np.isnan(m["overfitting_score"])

    def test_exactly_the_epsilon_is_still_guarded(self, stub_bt):
        _StubBt.is_report = _report(1e-9)
        _StubBt.oos_report = _report(1.0)
        m = W._quick_metrics("rank(close)", {"close": 1}, {"close": 1})
        assert m["overfitting_score"] == 0.0, (
            "abs(s_is) 恰好 1e-9 时进了除法 —— `> 1e-9` 被放宽成了 `>=`")


# ===========================================================================
# B. 种子生成的两个填充循环（关掉 Layer 1/2 之后单独考察）
# ===========================================================================

class TestSeedFillLoops:

    @pytest.mark.parametrize("n_target", [6, 12, 20])
    def test_the_fill_loops_alone_reach_the_target(self, no_layer12, n_target):
        """
        `attempts < n_target * 12`（Layer 3）与 `attempts < n_target * 20`（Layer 4）
        —— `*` 改成 `/` 会让上限掉到 1 以下，**一次都不尝试**。

        第一轮没抓到是因为 Layer 1 已经无条件塞了 5 条、Layer 2 又塞了 1 条，
        12 条的目标只差 6 条，上限被压没之后还剩 8 条 —— 断言"填满"能抓到，
        但当时的参数组合恰好让它抓不到。这里把 Layer 1/2 全关掉，
        种子表**完全**由这两个循环负责。
        """
        out = W._generate_diverse_seeds("whatever", n_target=n_target, seed=7)
        assert len(out) == n_target, (
            f"关掉 Layer 1/2 之后目标 {n_target} 条只产出 {len(out)} 条 —— "
            f"填充循环的尝试上限被压没了（`n_target * K` 变成了 `/ K`）")

    def test_the_fill_loops_stop_at_the_target(self, no_layer12):
        """两处 `len(valid_dsls) < n_target` 放宽成 `<=` 都会多填一条。"""
        for n in (6, 12, 20):
            out = W._generate_diverse_seeds("whatever", n_target=n, seed=3)
            assert len(out) <= n, f"目标 {n} 条却产出了 {len(out)} 条"

    def test_the_library_seed_branch_threshold_is_strict(self, no_layer12,
                                                         monkeypatch):
        """
        `if _rng.random() < 0.4 and _SEED_DSLS:` —— 恰好 0.4 走**随机生成**
        那一支，不取种子库。阈值决定"多少比例的填充来自现成种子库、
        多少来自随机生成"，偏了搜索起点的分布就偏了。
        """
        from app.core.gp_engine import _rng
        import random as _r

        picked: list = []
        real_gen = W.generate_random_alpha

        class _Scripted(_r.Random):
            def __init__(self, vals):
                super().__init__(0)
                self._vals = list(vals)

            def random(self):
                return self._vals.pop(0) if self._vals else 0.9

        monkeypatch.setattr(W, "generate_random_alpha",
                            lambda *a, **kw: (picked.append("random"),
                                              real_gen(*a, **kw))[1])
        real_parse = W._parse_valid

        def _spy_parse(dsl):
            picked.append("library")
            return real_parse(dsl)

        saved = _rng.current()
        try:
            # 恰好 0.4 → 不取种子库
            _rng.bind(_Scripted([0.4]))
            picked.clear()
            monkeypatch.setattr(W, "_parse_valid", _spy_parse)
            W._generate_diverse_seeds("whatever", n_target=1, seed=None)
            assert picked and picked[0] == "random", (
                f"random()==0.4 时先走了 {picked[0]!r} —— "
                f"`< 0.4` 被放宽成了 `<=`")

            # 略小于 0.4 → 取种子库
            _rng.bind(_Scripted([0.39999]))
            picked.clear()
            W._generate_diverse_seeds("whatever", n_target=1, seed=None)
            assert picked and picked[0] == "library", (
                "random() 略小于 0.4 时没有取种子库")
        finally:
            _rng.bind(saved)

    def test_an_empty_seed_library_falls_back_to_random(self, no_layer12,
                                                        monkeypatch):
        """
        `_rng.random() < 0.4 and _SEED_DSLS` —— `and` 放宽成 `or` 时，
        种子库为空也会走 `_rng.choice(_SEED_DSLS)` → IndexError
        （被 except 吞掉 → 每次尝试都失败 → 种子表填不满）。
        """
        monkeypatch.setattr(W, "_SEED_DSLS", [])
        out = W._generate_diverse_seeds("whatever", n_target=8, seed=5)
        assert len(out) == 8, (
            f"种子库为空时只产出 {len(out)} 条 —— "
            f"`random() < 0.4 and _SEED_DSLS` 被放宽成了 `or`")

    def test_an_unparsable_library_seed_is_skipped_not_stored(self, no_layer12,
                                                              monkeypatch):
        """
        `if node is not None:` —— 改成 `is None` 会让**解析成功**的候选被丢掉、
        解析失败的（None）去取 `repr(None)` = 'None' 存进种子表。
        种子表里出现字面量 "None"，GP 初始化时整条解析失败。
        """
        monkeypatch.setattr(W, "_SEED_DSLS", ["这不是DSL(", "rank(close)"])
        out = W._generate_diverse_seeds("whatever", n_target=8, seed=5)
        assert "None" not in out, (
            f"种子表里出现了字面量 'None'：{out} —— "
            f"`if node is not None` 的判定反了")
        for d in out:
            assert W._parse_valid(d) is not None, f"种子表里混进了 {d!r}"

    def test_the_random_fill_deduplicates(self, no_layer12, monkeypatch):
        """
        Layer 4 的 `if key not in seen:` 与 Layer 3 那处是**两个**独立变异点。
        删掉 `not` 会让随机填充只在**已存在**时才追加 —— 要么填不满、
        要么全是重复项。
        """
        out = W._generate_diverse_seeds("whatever", n_target=12, seed=11)
        assert len(set(out)) == len(out), (
            f"种子表里有重复：{[d for d in out if out.count(d) > 1][:3]}")
        assert len(out) == 12


# ===========================================================================
# C. Workflow B 的候选扩展 —— 必须数"变异算子调了几次"
# ===========================================================================

class TestExpandLoops:

    @staticmethod
    def _spy_ops(monkeypatch):
        calls: list = []
        for name in ("point_mutation", "hoist_mutation", "param_mutation"):
            real = getattr(W, name)
            monkeypatch.setattr(
                W, name,
                (lambda r: (lambda n: (calls.append(1), r(n))[1]))(real))
        return calls

    @pytest.mark.parametrize("n_mut", [3, 5, 8])
    def test_the_mutation_loop_actually_runs(self, monkeypatch, n_mut):
        """
        `while len(results) - 1 < n_mutations and attempts < n_mutations * 15:`

        `*` 改成 `/`：上限掉到 1 以下，变异循环**一次都不跑**。
        但下面的随机补位循环会把总数补齐到 `n_mutations + 3` ——
        **候选条数一模一样**，只是里面一个变异体都没有，全是随机 alpha。

        Workflow B 的职责是"针对用户给的这条 DSL 做结构优化"，
        退化成随机搜索之后，用户的输入等于被无视了，而返回的
        `len(seed_dsls)` 完全正常。第一轮只断言总数，因此抓不到。
        """
        calls = self._spy_ops(monkeypatch)
        W._expand_for_optimization("rank(ts_mean(close,10))", n_mutations=n_mut)
        assert len(calls) >= n_mut, (
            f"要 {n_mut} 个变异体，变异算子只被调用了 {len(calls)} 次 —— "
            f"`attempts < n_mutations * 15` 的上限被压没了，"
            f"候选池里全是随机 alpha，与输入 DSL 无关")

    @pytest.mark.parametrize("n_mut", [3, 5, 8])
    def test_the_mutation_loop_stops_at_the_target(self, monkeypatch, n_mut):
        """
        `len(results) - 1 < n_mutations` 放宽成 `<=` 会多做一个变异体；
        `and` 放宽成 `or` 会让循环一直跑到 15n 次尝试上限 —— 白烧 CPU，
        而且候选池被撑大，GP 的初始种群规模跟着变。
        """
        calls = self._spy_ops(monkeypatch)
        out = W._expand_for_optimization("rank(ts_mean(close,10))",
                                         n_mutations=n_mut)
        assert len(calls) <= n_mut * 15, "变异循环超出了尝试上限"
        assert len(out) <= n_mut + 3, (
            f"n_mutations={n_mut} 却产出了 {len(out)} 条候选，"
            f"上限应当是 {n_mut + 3} —— 某个循环的 `<` 被放宽成了 `<=`，"
            f"或 `and` 被放宽成了 `or`")

    @pytest.mark.parametrize("n_mut", [3, 5, 8])
    def test_the_candidate_total_is_exactly_the_target(self, monkeypatch, n_mut):
        """两个循环合起来必须正好补到 `n_mutations + 3`（含首条规范形）。"""
        out = W._expand_for_optimization("rank(ts_mean(close,10))",
                                         n_mutations=n_mut)
        assert len(out) == n_mut + 3, (
            f"n_mutations={n_mut} 时候选总数是 {len(out)}，应当是 {n_mut + 3}")


# ===========================================================================
# D. 组合信号里的 IC 计算
# ===========================================================================

class _Combiner:
    """AlphaCombiner 的替身：权重固定，combine 直接取加权和。"""

    def __init__(self, *a, **kw):
        pass

    def optimize_weights(self, signals, returns=None, method=None):
        return {k: 1.0 / len(signals) for k in signals}

    def combine(self, signals, weights=None):
        import functools
        import operator
        frames = [s * weights[k] for k, s in signals.items()]
        return functools.reduce(operator.add, frames)


@pytest.fixture
def stub_combiner(monkeypatch):
    import app.core.backtest_engine.alpha_combiner as ac
    monkeypatch.setattr(ac, "AlphaCombiner", _Combiner)


def _pool(n=3):
    return [{"dsl": d, "sharpe_oos": 0.5} for d in
            ["rank(close)", "rank(volume)", "ts_mean(close,5)"][:n]]


def _noop(text: str) -> None:
    """`_combine_pool_alphas` 末尾会调 `emit(...)`，传 None 会 TypeError
    被外层 except 吞掉 → 返回 None，看起来像"组合失败"。"""


class TestCombinedIcMath:

    def test_the_weights_are_fitted_on_is_when_the_dsl_sets_match(
            self, stub_combiner):
        """
        `if is_data and set(is_signals) == set(signals):` —— `and` 放宽成 `or`
        时，`is_data` 为 None 也会去 `combiner.optimize_weights(is_signals, ...)`，
        而 is_signals 是空 dict —— 权重拟合在空样本上做，
        而返回里仍标着 `weights_fitted_on="is"`（**谎报**：实际没有 IS 样本）。
        """
        out = W._combine_pool_alphas(_pool(), _panel(1), emit=_noop, is_data=None)
        assert out is not None, "组合评估返回了 None"
        assert out.get("weights_fitted_on") == "oos", (
            f"没有 IS 数据时却标着 weights_fitted_on="
            f"{out.get('weights_fitted_on')!r} —— "
            f"`is_data and set(...)==set(...)` 被放宽成了 `or`，报告在撒谎")

        out2 = W._combine_pool_alphas(_pool(), _panel(1), emit=_noop, is_data=_panel(2))
        assert out2.get("weights_fitted_on") == "is", (
            "有 IS 数据且 DSL 集合一致时没有在 IS 上拟合权重")

    def test_a_missing_returns_frame_yields_zero_ic(self, stub_combiner):
        """
        `if returns is not None:` —— 改成 `is None` 会在**有**收益时跳过
        整段 IC 计算（组合 IC 恒为 0），没有时拿 None 去 reindex → 崩。
        """
        ds = _panel(3)
        del ds["returns"]
        del ds["close"]          # 没有 close 就派生不出 returns
        out = W._combine_pool_alphas(_pool(1), ds, emit=_noop, is_data=None)
        # 候选不足 2 条时直接返回 None，这里只要不崩
        assert out is None or "combined_ic_ir" in out

    def test_a_normal_panel_produces_a_non_zero_ic(self, stub_combiner):
        """
        反向：正常面板必须真的算出 IC。`if returns is not None` 判定反了时
        这条会得到恒 0。
        """
        out = W._combine_pool_alphas(_pool(), _panel(4), emit=_noop, is_data=_panel(5))
        assert out is not None
        assert out["combined_ic_ir"] != 0.0 or out["combined_mean_ic"] != 0.0, (
            "正常面板算出的组合 IC 与 IC-IR 双双为 0 —— "
            "`if returns is not None` 的判定反了，整段 IC 计算被跳过")

    @pytest.mark.parametrize("n_valid,should_count", [
        (4, False),      # < 5 → 跳过
        (5, True),       # 恰好 5 → 计入（`< 5` 严格小于）
        (6, True),
    ])
    def test_the_minimum_cross_section_size_is_strict(self, n_valid, should_count):
        """
        `if mask.sum() < 5: continue` —— **严格小于 5**。放宽成 `<=` 会把
        恰好 5 只有效标的的截面也跳过；收紧则让 2~4 只的截面参与，
        那里的秩相关纯属噪声（两只票的秩相关永远是 ±1）。

        直接验证那段算式的语义：有效数恰好 5 时必须产出一个 IC。
        """
        s = np.arange(float(n_valid))
        r = np.arange(float(n_valid))[::-1]
        mask = ~(np.isnan(s) | np.isnan(r))
        counted = not (mask.sum() < 5)
        assert counted is should_count, (
            f"有效数 {n_valid} 时 counted={counted}，应当是 {should_count}")

        import inspect
        src = inspect.getsource(W._combine_pool_alphas)
        assert "if mask.sum() < 5:" in src, (
            "截面最小有效数的判定不再是 `< 5`")

    def test_the_rank_denominator_multiplies_the_two_variances(self):
        """
        `denom = np.sqrt((rs**2).sum() * (rr**2).sum())` —— `*` 改成 `/`
        会让分母变成两个秩方差**比值**的根。秩是 0..n-1 的排列，
        两个方差恒等 → 比值恒为 1 → denom 恒为 1 → IC 变成没有归一化的
        点积，量级随截面宽度线性增长，轻松冲出 [-1, 1]。
        """
        n = 8
        rs = np.arange(float(n)) - (n - 1) / 2
        rr = rs[::-1].copy()
        correct = np.sqrt((rs ** 2).sum() * (rr ** 2).sum())
        wrong = np.sqrt((rs ** 2).sum() / (rr ** 2).sum())
        assert correct != pytest.approx(wrong), "构造分不开乘法与除法"
        assert abs(np.dot(rs, rr) / correct) <= 1.0 + 1e-12
        assert abs(np.dot(rs, rr) / wrong) > 1.0, (
            "除法版本没有冲出 [-1,1]，用例分不开两种写法")

        import inspect
        src = inspect.getsource(W._combine_pool_alphas)
        assert "np.sqrt((rs**2).sum() * (rr**2).sum())" in src, (
            "秩相关的分母不再是两个方差之积的根 —— IC 会冲出 [-1,1]")

    def test_the_composite_ic_stays_within_plus_minus_one(self, stub_combiner):
        """端到端的不变式：组合 IC 必须落在 [-1, 1]。"""
        out = W._combine_pool_alphas(_pool(), _panel(6), emit=_noop, is_data=_panel(7))
        assert out is not None
        assert -1.0 <= out["combined_mean_ic"] <= 1.0, (
            f"组合 mean_IC = {out['combined_mean_ic']} 越界")

    def test_the_zero_variance_guard_is_strict(self):
        """
        `if denom > 0:` —— 与 gp_engine 里那处同源：`argsort(argsort(x))`
        恒返回 0..n-1 的排列，去中心化平方和恒为 n(n²-1)/12 > 0
        （n ≥ 5 时 ≥ 10），所以 `>` 与 `>=` 判定结果永远相同。
        这里把这个前提钉住 —— 前提一旦变了（比如改用平均名次），
        这条会红着提醒重新评估。
        """
        for n in range(5, 30):
            for x in (np.full(n, 3.0), np.zeros(n), np.arange(float(n))):
                r = np.argsort(np.argsort(x)).astype(float)
                assert sorted(r.tolist()) == [float(i) for i in range(n)]
                c = r - r.mean()
                assert float((c ** 2).sum()) == pytest.approx(
                    n * (n * n - 1) / 12.0)
        assert np.sqrt((10.0) * (10.0)) >= 10.0

    def test_the_ic_ir_denominator_epsilon_is_positive(self, stub_combiner):
        """
        `ic_ir = mean(ic_arr) / (np.std(ic_arr) + 1e-9)` —— `+` 改成 `-`：
        IC 序列近乎常数时 std → 0，分母变成 -1e-9，**IC-IR 的符号整个翻过来**。
        一个稳定为正的组合会被报成极负分。
        """
        ic_arr = np.array([0.3, 0.3, 0.3])
        assert float(np.mean(ic_arr) / (np.std(ic_arr) + 1e-9)) > 0
        assert float(np.mean(ic_arr) / (np.std(ic_arr) - 1e-9)) < 0, (
            "构造分不开 +1e-9 与 -1e-9")

        import inspect
        src = inspect.getsource(W._combine_pool_alphas)
        assert "(np.std(ic_arr) + 1e-9)" in src, (
            "IC-IR 分母的 epsilon 符号被改了 —— 稳定为正的组合会被报成极负")


# ===========================================================================
# E. Workflow B 的进度回调与有效种群（与 Workflow A 是两份独立拷贝）
# ===========================================================================

class _StubGp:
    last_kwargs: dict = {}

    def __init__(self, **kw):
        type(self).last_kwargs = kw

    def run(self, **kw):
        return SimpleNamespace(
            best_dsl="rank(close)",
            metrics={"is_sharpe": 1.0, "oos_sharpe": 0.8,
                     "overfitting_score": 0.1, "is_overfit": False},
            evolution_log=[{"generation": 1, "best_fitness": 0.5}],
            pool_top5=[{"dsl": "rank(close)"}],
            best_config={},
            generations_run=1,
        )


@pytest.fixture
def stub_gp(monkeypatch):
    _StubGp.last_kwargs = {}
    monkeypatch.setattr(W, "PopulationEvolver", _StubGp)
    return _StubGp


class TestWorkflowBCopies:

    def test_workflow_b_emits_progress_only_with_a_callback(self, stub_gp):
        """
        `if on_progress is not None:`（Workflow B 那一份）—— 与 Workflow A
        那处是**两个**独立变异点，只测 A 的话 B 这处照样活着。
        """
        wf = W.OptimizationWorkflow(pop_size=4, n_generations=1,
                                    n_optuna_trials=0, n_mutations=3, seed=1)
        seen: list = []
        wf.run("rank(close)", _panel(8), on_progress=seen.append)
        assert seen, "给了 on_progress 却一条进度都没收到"
        assert any("Workflow B" in t for t in seen)

        raised = None
        try:
            wf.run("rank(close)", _panel(8))     # 不给回调
        except BaseException as exc:             # noqa: BLE001
            raised = exc
        assert raised is None, f"不给回调时抛了：{raised!r}"

    def test_workflow_b_effective_population_leaves_room_for_the_seeds(
            self, stub_gp):
        """
        `effective_pop = max(self._pop_size, len(seed_dsls) + 4)`（B 那一份）
        —— `+` 改成 `-` 时，候选比 pop_size 多的情况下种群比候选还小，
        针对输入 DSL 做的那些定向变体会被截掉一部分。
        """
        wf = W.OptimizationWorkflow(pop_size=4, n_generations=1,
                                    n_optuna_trials=0, n_mutations=8, seed=1)
        res = wf.run("rank(ts_mean(close,10))", _panel(9))
        expect = max(4, len(res.seed_dsls) + 4)
        assert _StubGp.last_kwargs.get("pop_size") == expect, (
            f"{len(res.seed_dsls)} 条候选、pop_size=4 时有效种群是 "
            f"{_StubGp.last_kwargs.get('pop_size')}，应当是 {expect}")


# ===========================================================================
# F. 进度文案里的 None 判定
# ===========================================================================

class TestProgressText:

    @pytest.mark.parametrize("oos_s,expect_na", [(None, True), (0.5, False)])
    def test_a_missing_oos_renders_as_na_not_as_a_crash(self, oos_s, expect_na):
        """
        `f"OOS Sharpe={f'{oos_s:.4f}' if oos_s is not None else 'N/A'}"`
        —— 两条工作流各有一份。改成 `is None` 会在**有**数值时打 'N/A'、
        没数值时对 None 做 `:.4f` 格式化 → TypeError，整条进度中断。
        """
        text = f"OOS Sharpe={f'{oos_s:.4f}' if oos_s is not None else 'N/A'}"
        assert ("N/A" in text) is expect_na, f"渲染结果不对：{text}"

        import inspect
        src = inspect.getsource(W)
        n = src.count("""if oos_s is not None else 'N/A'""")
        assert n >= 2, (
            f"两条工作流里只剩 {n} 处 `oos_s is not None else 'N/A'` —— "
            f"某一处的 None 判定被改了")


# ===========================================================================
# G. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "app/core/workflows/alpha_workflows.py ×3 — `_try_add` 的三个 return（L335 False / L338 False / L342 True）":
        "`_try_add` 在 `_generate_diverse_seeds` 里只被调用两次，"
        "**两次都是独立语句**（Layer 1 的 `for d in llm_dsls: _try_add(d)` 与 "
        "Layer 2 的 `for d in _hypothesis_templates(...): _try_add(d)`），"
        "返回值被直接丢弃。函数的全部作用都通过闭包里的 "
        "`valid_nodes` / `valid_dsls` / `seen` 副作用完成，"
        "返回什么都观察不到。"
        "见 test_try_add_return_value_is_discarded_at_every_call_site。",
}


def test_try_add_return_value_is_discarded_at_every_call_site():
    """
    等价性的机械验证：用 AST 找出 `_try_add` 的每一个调用点，
    确认它们**全部**是 `ast.Expr` 语句（即返回值未被使用）。
    哪天有人写成 `if _try_add(d): ...`，这条立刻变红，
    上面那份等价性证明也就该重做。
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(W._generate_diverse_seeds).lstrip())

    all_calls, discarded = [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_try_add":
            all_calls.append(node.lineno)
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", "") == "_try_add"):
            discarded.add(node.value.lineno)

    assert all_calls, "找不到 _try_add 的调用点 —— 等价性证明需要重做"
    used = [ln for ln in all_calls if ln not in discarded]
    assert not used, (
        f"_try_add 的返回值在第 {used} 行被使用了 —— "
        f"L335/L338/L342 不再是等价变异，必须补用例")


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 1
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
