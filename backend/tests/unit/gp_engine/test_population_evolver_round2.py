"""
gp_engine/population_evolver.py —— 第二轮补强（变异测试驱动）

`test_population_evolver_internals.py` 把首测的 10.4% 拉到 66.7%，
但 rerun_5 之后仍有 **14 个存活**。逐条查下来，其中几条是我上一版
**断言选错了观察面**，而不是没写用例：

  - `if gen < self._n_gen - 1:` —— 上一版断言"评估了 n_gen 次种群"。
    但 `<=` 和 `-`→`+` **都不改变评估次数**（多建的那一代会被下一轮
    评估直接覆盖），那条断言从一开始就分不开。真正能分开的是
    **`_generate_next_population` 被调了几次**：正确 n_gen-1 次。
  - `n_elite = max(1, int(elite_ratio * pop_size))` —— 上一版断言
    "带过来的原样个体 >= expect"，而变异体偶尔与父代同形也会被算进去，
    噪声盖过了信号。改成断言 `next_gen` 的**前缀**恰好是 fitness 最高的
    那几条。
  - `_evaluate_one_single` / `_evaluate_one_multi` 里各有**一份独立的**
    过拟合公式与 NaN 兜底 —— 上一版只测了 `_extract_metrics` 那一份。
  - 两个概率阈值 `roll < 0.30` / `roll < 0.65` 需要脚本化随机源才能
    精确打在边界上。
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core.gp_engine.population_evolver import EvalResult, PopulationEvolver


T, N = 120, 5


def _panel(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, (T, N)), axis=0)
    idx = pd.bdate_range("2021-01-04", periods=T)
    cols = [f"S{i}" for i in range(N)]
    out = {k: pd.DataFrame(v, index=idx, columns=cols) for k, v in {
        "close": close, "open": close, "vwap": close,
        "high": close * 1.01, "low": close * 0.99,
        "volume": np.full_like(close, 1e6),
    }.items()}
    out["returns"] = out["close"].pct_change().fillna(0.0)
    return out


def _res(dsl: str, fitness: float, node=None, **kw) -> EvalResult:
    base = dict(sharpe_is=1.0, sharpe_oos=0.8, turnover=0.5,
                max_drawdown=-0.1, overfitting_score=0.1)
    base.update(kw)
    return EvalResult(dsl=dsl, fitness=fitness, node=node, **base)


def _rep(sharpe, *, turnover=1.0, dd=-0.1):
    return SimpleNamespace(sharpe_ratio=sharpe, ann_turnover=turnover,
                           max_drawdown=dd, annualized_return=0.05, mean_ic=0.02)


class _StubBt:
    is_report = None
    oos_report = None

    def __init__(self, *a, **kw):
        pass

    def run(self, dsl, data, oos_dataset=None):
        return SimpleNamespace(is_report=type(self).is_report,
                               oos_report=type(self).oos_report)


@pytest.fixture
def stub_bt(monkeypatch):
    import app.core.backtest_engine.realistic_backtester as rb
    monkeypatch.setattr(rb, "RealisticBacktester", _StubBt)
    return _StubBt


@pytest.fixture(scope="module")
def evolver() -> PopulationEvolver:
    # fitness_mode 显式写成 "holdout"：本文件测的是**单段口径**下的过拟合公式
    # 与 NaN 兜底。发布默认已改为 purged_cv（Phase S.1），那条路径下 sharpe_oos
    # 来自 IS 内部 K 折、与 stub 的 oos_report 无关，这些断言会失去观察面。
    # purged_cv 口径另有 test_purged_cv_fitness.py 专测。
    return PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                             pop_size=8, n_generations=3, seed=42,
                             fitness_mode="holdout")


# ===========================================================================
# A. _evaluate_one_single —— 第二份过拟合公式与 NaN 兜底
# ===========================================================================

class TestEvaluateOneSingle:

    @pytest.mark.parametrize("is_s,oos_s,expect", [
        (2.0, 2.0, 0.0),      # 没退化 → 0；`+` 会给 1.0
        (2.0, 1.0, 0.5),
        (2.0, 0.0, 1.0),
        (2.0, 3.0, 0.0),      # OOS 更好 → clip 到 0；`+` 会给 1.0
    ])
    def test_the_degradation_formula(self, evolver, stub_bt, is_s, oos_s, expect):
        _StubBt.is_report = _rep(is_s)
        _StubBt.oos_report = _rep(oos_s)
        r = evolver._evaluate_one_single("rank(close)")
        assert r is not None, "替身回测器下评估返回了 None"
        assert r.overfitting_score == pytest.approx(expect), (
            f"IS={is_s}、OOS={oos_s} 的过拟合度算成了 {r.overfitting_score}，"
            f"应当是 {expect} —— `(sharpe_is - sharpe_oos)` 的符号被改了")

    def test_a_zero_is_sharpe_is_not_divided_by(self, evolver, stub_bt):
        """`if abs(sharpe_is) > 1e-9 and oos_r:` —— **严格大于**。"""
        _StubBt.is_report = _rep(0.0)
        _StubBt.oos_report = _rep(1.0)
        assert evolver._evaluate_one_single("rank(close)").overfitting_score == 0.0

    def test_exactly_the_epsilon_is_still_guarded(self, evolver, stub_bt):
        _StubBt.is_report = _rep(1e-9)
        _StubBt.oos_report = _rep(1.0)
        assert evolver._evaluate_one_single("rank(close)").overfitting_score == 0.0, (
            "abs(sharpe_is) 恰好 1e-9 时进了除法 —— `> 1e-9` 被放宽成 `>=`")

    def test_a_missing_oos_report_skips_the_formula(self, evolver, stub_bt):
        """
        `abs(sharpe_is) > 1e-9 and oos_r` —— `and` 放宽成 `or` 时，
        没有 OOS 报告也会进除法：`sharpe_oos` 是兜底的 0.0，
        于是**任何**单段回测都被判成 100% 过拟合。
        """
        _StubBt.is_report = _rep(2.0)
        _StubBt.oos_report = None
        r = evolver._evaluate_one_single("rank(close)")
        assert r.overfitting_score == 0.0, (
            f"没有 OOS 报告时过拟合度是 {r.overfitting_score} —— "
            f"`and oos_r` 被放宽成了 `or`，单段回测会被判成全退化")

    def test_a_nan_sharpe_is_coerced_to_zero(self, evolver, stub_bt):
        """`return fv if not np.isnan(fv) else 0.0`（本函数里那一份）。"""
        _StubBt.is_report = _rep(float("nan"))
        _StubBt.oos_report = _rep(1.0)
        assert evolver._evaluate_one_single("rank(close)").sharpe_is == 0.0, (
            "NaN 没有被兜成 0")

        _StubBt.is_report = _rep(1.75)
        r2 = evolver._evaluate_one_single("rank(close)")
        assert r2.sharpe_is == pytest.approx(1.75), (
            f"正常数值被改成了 {r2.sharpe_is} —— "
            f"`if not np.isnan(fv)` 的 not 被删掉了")


# ===========================================================================
# B. _evaluate_one_multi —— 第三份过拟合公式
# ===========================================================================

class TestEvaluateOneMulti:

    @staticmethod
    def _make(monkeypatch, agg_sharpe, is_sharpe):
        import app.core.backtest_engine.realistic_backtester as rb
        import app.core.backtest_engine.multi_dataset_backtester as mb

        class _Bt:
            def __init__(self, *a, **kw):
                pass

            def run(self, dsl, data, oos_dataset=None):
                return SimpleNamespace(is_report=_rep(is_sharpe))

        class _Multi:
            def __init__(self, *a, **kw):
                pass

            def run(self, dsl, datasets):
                return SimpleNamespace(
                    aggregated_sharpe=agg_sharpe,
                    per_dataset={"d": SimpleNamespace(max_drawdown=-0.2,
                                                      error=None)},
                )

        monkeypatch.setattr(rb, "RealisticBacktester", _Bt)
        monkeypatch.setattr(mb, "MultiDatasetBacktester", _Multi)
        monkeypatch.setattr(mb, "compute_multi_dataset_fitness",
                            lambda **kw: 1.0)
        return PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                                 multi_datasets={"d": _panel(3)},
                                 pop_size=4, n_generations=1, seed=1)

    @pytest.mark.parametrize("is_s,agg_s,expect", [
        (2.0, 2.0, 0.0),      # 没退化；`+` 会给 1.0
        (2.0, 1.0, 0.5),
        (2.0, 0.0, 1.0),
        (2.0, 3.0, 0.0),
    ])
    def test_the_multi_dataset_degradation_formula(self, monkeypatch,
                                                   is_s, agg_s, expect):
        ev = self._make(monkeypatch, agg_s, is_s)
        r = ev._evaluate_one_multi("rank(close)")
        assert r is not None, "多数据集评估返回了 None"
        assert r.overfitting_score == pytest.approx(expect), (
            f"IS={is_s}、聚合={agg_s} 的过拟合度算成了 {r.overfitting_score}，"
            f"应当是 {expect} —— `(sharpe_is - agg_sharpe)` 的符号被改了")

    def test_a_zero_is_sharpe_is_not_divided_by(self, monkeypatch):
        ev = self._make(monkeypatch, 1.0, 0.0)
        assert ev._evaluate_one_multi("rank(close)").overfitting_score == 0.0

    def test_exactly_the_epsilon_is_still_guarded(self, monkeypatch):
        ev = self._make(monkeypatch, 1.0, 1e-9)
        assert ev._evaluate_one_multi("rank(close)").overfitting_score == 0.0, (
            "abs(sharpe_is) 恰好 1e-9 时进了除法 —— `> 1e-9` 被放宽成 `>=`")


# ===========================================================================
# C. 种群初始化里的两个概率阈值
# ===========================================================================

class _ScriptedRandom:
    """`random()` 按脚本吐值；其余方法沿用真实 Random。"""

    def __init__(self, randoms, tail=0.99, seed=0):
        import random as _r
        object.__setattr__(self, "_base", _r.Random(seed))
        object.__setattr__(self, "_vals", list(randoms))
        object.__setattr__(self, "_tail", tail)

    def random(self):
        return self._vals.pop(0) if self._vals else self._tail

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_base"), name)


class TestInitPopulationThresholds:

    def test_the_crossover_threshold_is_strict(self, monkeypatch):
        """
        `if len(pop) >= 2 and roll < 0.30:` —— 恰好 0.30 **不**走交叉，
        落到下一档（变异）。阈值一放宽，算子分布就偏了：交叉多做、变异少做，
        而两者的探索方向完全不同。
        """
        import app.core.gp_engine.population_evolver as PE
        seen: list = []
        monkeypatch.setattr(PE, "subtree_crossover",
                            lambda a, b: (seen.append("x"), (a, b))[1])
        for name in ("point_mutation", "hoist_mutation", "param_mutation"):
            monkeypatch.setattr(PE, name,
                                lambda n: (seen.append("m"), n)[1])

        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=4, n_generations=1, seed=1)
        ev._rng = _ScriptedRandom([0.30])
        ev._init_population(None, ["rank(close)", "rank(volume)"])
        assert seen and seen[0] == "m", (
            f"roll==0.30 时首选算子是 {seen[0]!r}（x=交叉 m=变异）—— "
            f"`roll < 0.30` 被放宽成了 `<=`")

        seen.clear()
        ev2 = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                                pop_size=4, n_generations=1, seed=1)
        ev2._rng = _ScriptedRandom([0.29999])
        ev2._init_population(None, ["rank(close)", "rank(volume)"])
        assert seen and seen[0] == "x", "roll 略小于 0.30 时没有走交叉"

    def test_the_mutation_threshold_is_strict(self, monkeypatch):
        """`elif pop and roll < 0.65:` —— 恰好 0.65 **不**走变异，落到随机填充。"""
        import app.core.gp_engine.population_evolver as PE
        seen: list = []
        for name in ("point_mutation", "hoist_mutation", "param_mutation"):
            monkeypatch.setattr(PE, name, lambda n: (seen.append("m"), n)[1])

        real_random_alpha = PE.generate_random_alpha
        monkeypatch.setattr(
            PE, "generate_random_alpha",
            lambda **kw: (seen.append("r"), real_random_alpha(**kw))[1])

        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=4, n_generations=1, seed=1)
        ev._rng = _ScriptedRandom([0.65])
        ev._init_population(None, ["rank(close)"])
        assert seen and seen[0] == "r", (
            f"roll==0.65 时首选算子是 {seen[0]!r}（m=变异 r=随机）—— "
            f"`roll < 0.65` 被放宽成了 `<=`")

        seen.clear()
        ev2 = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                                pop_size=4, n_generations=1, seed=1)
        ev2._rng = _ScriptedRandom([0.64999])
        ev2._init_population(None, ["rank(close)"])
        assert seen and seen[0] == "m", "roll 略小于 0.65 时没有走变异"

    def test_family_seeds_are_deduplicated(self):
        """
        Layer 2a（家族种子）那处 `if key not in seen:` 与种子 DSL 那处是
        **两个**独立的变异点。删掉 `not` 会让家族种子一条都进不来。
        """
        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               factor_family="momentum",
                               pop_size=10, n_generations=1, seed=3)
        pop = ev._init_population(None, None)
        reprs = [repr(n) for n in pop]
        assert len(set(reprs)) == len(reprs), (
            f"家族种子进种群时出现重复："
            f"{[r for r in reprs if reprs.count(r) > 1][:3]}")
        assert len(pop) == 10

    def test_a_seed_that_duplicates_a_family_seed_enters_only_once(self):
        from app.core.gp_engine.gp_engine import get_seeds_for_family
        fam = get_seeds_for_family("momentum")
        assert fam, "momentum 家族没有种子 —— 用例前提被破坏"
        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               factor_family="momentum",
                               pop_size=10, n_generations=1, seed=3)
        pop = ev._init_population(fam[0], None)
        reprs = [repr(n) for n in pop]
        assert reprs.count(reprs[0]) == 1, (
            "显式种子与家族种子相同却进了两次 —— 去重的 `not in seen` 被删掉了")

    def test_no_warning_when_the_population_is_exactly_full(self, caplog):
        """
        `if len(pop) < self._pop_size:` 的告警 —— 放宽成 `<=` 会在**填满**时
        也打"种群未填满"。这条警告是给人看的信号，天天误报等于没有。
        """
        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=8, n_generations=1, seed=5)
        with caplog.at_level(logging.WARNING,
                             logger="app.core.gp_engine.population_evolver"):
            pop = ev._init_population("rank(close)")
        assert len(pop) == 8, "用例前提被破坏：种群没填满"
        msgs = [r.getMessage() for r in caplog.records
                if "种群未填满" in r.getMessage()]
        assert not msgs, (
            f"种群恰好填满却打出了未填满告警：{msgs} —— "
            f"`< pop_size` 被放宽成了 `<=`")


# ===========================================================================
# D. 精英数与代际边界
# ===========================================================================

_DSLS = ["rank(close)", "rank(volume)", "rank(high)", "rank(low)",
         "rank(open)", "rank(vwap)", "rank(returns)",
         "ts_mean(close,5)", "ts_mean(volume,5)", "ts_std(close,10)",
         "ts_std(volume,10)", "ts_delta(close,3)"]


class TestElitismAndGenerationBoundary:

    @pytest.mark.parametrize("pop_size,ratio,expect", [
        (12, 0.25, 3),
        (8,  0.25, 2),
        (20, 0.50, 10),
        (4,  0.25, 1),
    ])
    def test_the_elite_prefix_is_exactly_the_top_n(self, pop_size, ratio, expect):
        """
        `n_elite = max(1, int(self._elite_ratio * self._pop_size))`

        `*` 改成 `/`：0.25/12 = 0.02 → int → 0 → max(1,0) = 1。
        **不管种群多大，精英永远只有 1 个**，精英保留形同虚设。

        上一版断言"带过来的原样个体 >= expect"，而变异体偶尔与父代同形
        也会被算进去，噪声盖过信号。精英是 `next_gen` 的**前缀**且按
        fitness 降序 —— 直接断言前缀。
        """
        from app.core.alpha_engine.parser import Parser
        p = Parser()
        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=pop_size, elite_ratio=ratio,
                               n_generations=1, seed=1)
        results = [_res(d, float(len(_DSLS) - i), node=p.parse(d))
                   for i, d in enumerate(_DSLS)]
        out = ev._generate_next_population(results, {"point": 1.0})

        want = [repr(p.parse(d)) for d in _DSLS[:expect]]
        got = [repr(n) for n in out[:expect]]
        assert got == want, (
            f"pop_size={pop_size}、elite_ratio={ratio} 时精英前缀是 {got}，"
            f"应当是 fitness 最高的 {expect} 条 {want} —— "
            f"`elite_ratio * pop_size` 的算符被改了")

    @pytest.mark.parametrize("n_gen", [1, 2, 4, 5])
    def test_the_next_population_is_built_once_less_than_the_generations(
            self, monkeypatch, n_gen):
        """
        `if gen < self._n_gen - 1:` —— 上一版断言"评估了 n_gen 次种群"，
        但 `<=` 与 `-`→`+` **都不改变评估次数**（多建的那一代会被下一轮
        评估直接覆盖），那条断言分不开两种取值。

        真正能分开的是 `_generate_next_population` 的**调用次数**：
        正确 n_gen-1 次，`<=` 与 `+` 都会变成 n_gen 次 —— 最后一代白建一次，
        每次都是一整套回测的代价。
        """
        from app.core.alpha_engine.parser import Parser
        p = Parser()
        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=4, n_generations=n_gen, seed=5)
        builds: list = []
        real_build = ev._generate_next_population

        monkeypatch.setattr(ev, "_evaluate_population",
                            lambda pop: [_res("rank(close)", 1.0,
                                              node=p.parse("rank(close)"))])
        monkeypatch.setattr(ev, "_compute_signal_vec", lambda dsl: None)
        monkeypatch.setattr(ev, "_optuna_fine_tune",
                            lambda *a, **kw: ({}, {"is_sharpe": 1.0}))
        monkeypatch.setattr(ev, "_generate_next_population",
                            lambda r, w: (builds.append(1), real_build(r, w))[1])

        ev.run(seed_dsl="rank(close)", n_optuna_trials=0)
        assert len(builds) == max(0, n_gen - 1), (
            f"n_generations={n_gen} 时构建了 {len(builds)} 次下一代，"
            f"应当是 {max(0, n_gen - 1)} 次 —— "
            f"`if gen < self._n_gen - 1` 的边界被改了")


# ===========================================================================
# E. 存活变异的等价性证明
# ===========================================================================

# 本文件曾把 `L637 while len(next_gen) < pop_size ...` 的 `<` → `<=`
# 记成"等价变异"，理由是末尾的 `return next_gen[: self._pop_size]` 会把
# 多产出的那个个体截掉，返回值逐元素相同。
#
# **那个结论是错的**：输出看不出来，**调用次数看得出来** ——
# 多跑一轮就是多一次 `point_mutation` / `generate_random_alpha` 调用。
# 见 test_population_evolver_round3.py::TestLoopIterationCounts，
# 那里把 L637 与 L732 的三个点全部改成了杀死。
#
# 教训：**能杀就不要写等价证明**。"输出相同"只是没找到对的观察面，
# 不等于"观察不到"。
PROVEN_EQUIVALENT: dict = {}


def test_every_survivor_has_a_written_proof():
    # 显式写 `== 0`：**"这个模块零存活"与"忘了写声明"必须在代码里分得开**。
    # 原来只有一个空循环 —— 对空字典的全称断言恒真，等于什么都没查。
    assert len(PROVEN_EQUIVALENT) == 0
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
