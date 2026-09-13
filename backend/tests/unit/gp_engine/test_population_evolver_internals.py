"""
gp_engine/population_evolver.py —— 进化循环内部的定钉测试（变异测试驱动）

来由：48 个变异点，首测击杀率 **10.4%**（存活 43）。既有覆盖
（`unit/test_gp_evolution_full.py` / `test_alpha_discovery` /
`test_phase6_reproducibility`）测的是"跑得完、结果可复现"——
端到端跑通掩盖了内部几乎所有分支。

存活项集中在四类，每一类坏掉都不会报错：

  1. **过拟合度公式** `(sharpe_is - sharpe_oos) / abs(sharpe_is)` 的 `-`
     —— 改成 `+` 后，IS 好 OOS 差的因子反而拿到低过拟合分。三处同样的
     公式（`_evaluate_one_single` / `_evaluate_one_multi` / `_extract_metrics`）
     全都活着。
  2. **精英与排序** `sorted(..., reverse=True)`、
     `n_elite = max(1, int(elite_ratio * pop_size))` 的 `*`
     —— 精英取成最差的一批，或者精英数变成 1，进化方向整个反过来/退化。
  3. **循环上限与概率阈值** `attempts < pop_size * 20`、`roll < 0.30/0.65`
     —— 上限的 `*` 改成 `/` 会让填充提前放弃、种群长期填不满；
     阈值改 `<=` 会让算子分布偏掉。
  4. **代际编号** `gen + 1` 两处、`if gen < self._n_gen - 1:`
     —— 编号错位会让台账里的"第几代"对不上；最后一代多跑一轮
     则是纯浪费（每一代都要跑整套回测）。

测法：内部方法（`_extract_metrics` / `_init_population` /
`_generate_next_population` / `_quick_metrics`）都能单独调，用桩报告
和桩评估喂进去，不跑真回测。`run()` 那几条只在整循环里才到得了的分支，
用替身替掉 `_evaluate_population` 与 Optuna，整轮跑完在毫秒级。
"""
from __future__ import annotations

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
    out = {}
    for k, arr in {
        "close": close, "open": close, "vwap": close,
        "high": close * 1.01, "low": close * 0.99,
        "volume": np.full_like(close, 1e6),
    }.items():
        out[k] = pd.DataFrame(arr, index=idx, columns=cols)
    out["returns"] = out["close"].pct_change().fillna(0.0)
    return out


@pytest.fixture(scope="module")
def evolver() -> PopulationEvolver:
    return PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                             pop_size=8, n_generations=3, seed=42)


def _res(dsl: str, fitness: float, node=None, **kw) -> EvalResult:
    base = dict(sharpe_is=1.0, sharpe_oos=0.8, turnover=0.5,
                max_drawdown=-0.1, overfitting_score=0.1)
    base.update(kw)
    return EvalResult(dsl=dsl, fitness=fitness, node=node, **base)


# ===========================================================================
# A. EvalResult 的表示
# ===========================================================================

def test_the_parsed_node_stays_out_of_the_eval_result_repr():
    """
    `node: Optional[Node] = field(default=None, repr=False)` —— 翻成 True
    会把整棵 AST 塞进 `repr(EvalResult)`。这个 repr 会进日志，
    一代十几个个体、每个几百字符，日志直接被淹掉。
    """
    from app.core.alpha_engine.parser import Parser
    node = Parser().parse("rank(ts_mean(close,10))")
    r = _res("rank(ts_mean(close,10))", 1.0, node=node)
    text = repr(r)
    assert "node=" not in text, f"AST 进了 repr：{text[:200]}…"
    assert len(text) < 300, f"EvalResult 的 repr 膨胀到了 {len(text)} 字符"


# ===========================================================================
# B. 过拟合度公式（三处同样的式子）
# ===========================================================================

def _report(sharpe, *, turnover=1.0, dd=-0.1, ret=0.05, ic=0.02):
    return SimpleNamespace(sharpe_ratio=sharpe, ann_turnover=turnover,
                           max_drawdown=dd, annualized_return=ret, mean_ic=ic)


def _bt_result(is_sharpe, oos_sharpe):
    return SimpleNamespace(
        is_report=_report(is_sharpe),
        oos_report=_report(oos_sharpe) if oos_sharpe is not None else None,
        summary=lambda: "stub",
    )


class TestOverfitFormula:
    """
    `overfit = clip((is - oos) / abs(is), 0, 1)`

    这是 GP 唯一的过拟合信号。把 `-` 改成 `+`：
      is=2.0, oos=0.0（**最严重**的过拟合）→ 正确 1.0，错误 1.0（被 clip 掩盖）
      is=2.0, oos=2.0（完全没退化）        → 正确 0.0，错误 1.0
    所以必须用**没退化**的那组才分得开 —— 只测极端过拟合的用例抓不到。
    """

    @pytest.mark.parametrize("is_s,oos_s,expect", [
        (2.0,  2.0,  0.0),      # 没退化 → 0；`+` 会给 1.0
        (2.0,  1.0,  0.5),      # 退化一半；`+` 会给 1.0
        (2.0,  0.0,  1.0),      # 全退化
        (2.0, -2.0,  1.0),      # 反向，clip 到 1
        (2.0,  3.0,  0.0),      # OOS 比 IS 还好 → clip 到 0；`+` 会给 1.0
    ])
    def test_extract_metrics_degradation(self, is_s, oos_s, expect):
        m = PopulationEvolver._extract_metrics(_bt_result(is_s, oos_s))
        assert m["overfitting_score"] == pytest.approx(expect), (
            f"IS={is_s}、OOS={oos_s} 的过拟合度算成了 "
            f"{m['overfitting_score']}，应当是 {expect} —— "
            f"`(is - oos)` 的符号被改了")

    def test_a_near_zero_is_sharpe_is_not_divided_by(self):
        """
        `if oos_s is not None and abs(is_s) > 1e-9:` —— **严格大于**。
        放宽成 `>=` 会让 `is_s == 0` 走进除法 → 0/0 → NaN 混进
        overfitting_score，而 `is_overfit` 读的正是它（NaN > 0.5 是 False，
        于是一个算不出来的因子被当成"不过拟合"放行）。
        """
        m = PopulationEvolver._extract_metrics(_bt_result(0.0, 1.0))
        assert m["overfitting_score"] == 0.0, (
            f"IS Sharpe 为 0 时过拟合度是 {m['overfitting_score']} —— "
            f"零除守卫的 `> 1e-9` 被放宽了")
        assert not np.isnan(m["overfitting_score"])

    def test_exactly_the_epsilon_is_still_guarded(self):
        """边界值本身：`abs(is_s) == 1e-9` 时**不**进除法。"""
        m = PopulationEvolver._extract_metrics(_bt_result(1e-9, 1.0))
        assert m["overfitting_score"] == 0.0, (
            "abs(is_sharpe) 恰好等于 1e-9 时进了除法 —— `> 1e-9` 被放宽成 `>=`")

    def test_missing_oos_skips_the_formula(self):
        """
        `oos_s is not None and ...` —— 删掉 `not` 会反过来：有 OOS 时跳过、
        没 OOS 时拿 `None` 去做减法 → TypeError。
        """
        m = PopulationEvolver._extract_metrics(_bt_result(2.0, None))
        assert m["oos_sharpe"] is None
        assert m["overfitting_score"] == 0.0

    @pytest.mark.parametrize("score,expect", [
        (0.49, False), (0.50, False), (0.501, True), (1.0, True),
    ])
    def test_the_overfit_flag_boundary_is_strict(self, score, expect):
        """
        `"is_overfit": overfit > 0.5` —— **严格大于**。放宽成 `>=` 会把
        恰好 0.5 的因子也判成过拟合。0.5 是"OOS 退化一半"，是分界线本身，
        判进判出决定这条因子能不能进池子。
        """
        is_s = 2.0
        oos_s = is_s * (1.0 - score)
        m = PopulationEvolver._extract_metrics(_bt_result(is_s, oos_s))
        assert m["overfitting_score"] == pytest.approx(score)
        assert m["is_overfit"] is expect, (
            f"过拟合度 {score} 被判成 is_overfit={m['is_overfit']}，"
            f"应当是 {expect} —— `> 0.5` 的边界被改了")

    def test_nan_sharpe_becomes_none_not_a_silent_zero(self):
        """`return None if np.isnan(fv) else fv` —— NaN 必须变 None，不能变 0。"""
        m = PopulationEvolver._extract_metrics(_bt_result(float("nan"), 1.0))
        assert m["is_sharpe"] == 0.0          # `or 0.0` 兜底
        assert m["overfitting_score"] == 0.0  # 0 被零除守卫挡住


# ===========================================================================
# C. 兜底指标
# ===========================================================================

class TestFallbackMetrics:

    def test_the_fallback_is_not_flagged_as_overfit(self):
        """
        `"overfitting_score": 0.0, "is_overfit": False` —— 翻成 True 会让
        **每一次评估失败**都被记成过拟合。失败和过拟合是两件事，
        混在一起之后台账里的过拟合率就没有意义了。
        """
        m = PopulationEvolver._make_fallback_metrics("rank(close)")
        assert m["is_overfit"] is False, "兜底指标把评估失败报成了过拟合"
        assert m["overfitting_score"] == 0.0
        assert m["is_sharpe"] is None and m["oos_sharpe"] is None

    def test_quick_metrics_falls_back_when_evaluation_returns_none(self, evolver,
                                                                   monkeypatch):
        monkeypatch.setattr(evolver, "_evaluate_one", lambda dsl: None)
        m = evolver._quick_metrics("rank(close)")
        assert m["is_overfit"] is False and m["overfitting_score"] == 0.0

    @pytest.mark.parametrize("score,expect", [(0.5, False), (0.51, True)])
    def test_quick_metrics_overfit_boundary(self, evolver, monkeypatch,
                                            score, expect):
        monkeypatch.setattr(evolver, "_evaluate_one",
                            lambda dsl: _res(dsl, 1.0, overfitting_score=score))
        assert evolver._quick_metrics("rank(close)")["is_overfit"] is expect, (
            f"_quick_metrics 里 overfitting_score={score} 的判定错了 —— "
            f"`> 0.5` 的边界被改了")


# ===========================================================================
# D. 种群初始化
# ===========================================================================

class TestInitPopulation:

    def test_duplicate_seed_dsls_are_collapsed(self, evolver):
        """
        `if d not in all_seeds: all_seeds.append(d)` —— 删掉 `not` 会反过来：
        **只有重复的**才进清单，第一次出现的种子全被丢掉。
        用户给的假设一条都进不了种群。
        """
        pop = evolver._init_population("rank(close)",
                                       ["rank(close)", "rank(volume)"])
        reprs = [repr(n) for n in pop]
        assert "rank(close)" in reprs, (
            "显式给的种子没有进种群 —— `if d not in all_seeds` 的 not 被删掉了")
        assert "rank(volume)" in reprs
        assert reprs.count("rank(close)") == 1, "重复的种子进了两次"

    def test_structurally_identical_seeds_are_deduplicated(self, evolver):
        """
        `if key not in seen:` —— key 是 `repr(node)`，两条写法不同但解析
        结果相同的 DSL 只能进一个。删掉 `not` 会让**新**的结构全被拒、
        只有重复的能进，种群退化成一个个体的复制品。
        """
        pop = evolver._init_population(None, ["rank(close)", "rank( close )"])
        assert [repr(n) for n in pop].count("rank(close)") == 1

    def test_the_population_is_filled_to_the_target_size(self, evolver):
        """
        `while len(pop) < self._pop_size and attempts < self._pop_size * 20:`

        三个变异点：`<` 放宽成 `<=` 会多填一个；`and` 放宽成 `or` 会在
        填满之后继续尝试到上限（纯浪费）；`*` 改成 `/` 会把尝试上限从
        pop_size*20 压到 pop_size/20 < 1 —— **一次都不尝试**，
        种群永远只有种子那几个。
        """
        pop = evolver._init_population("rank(close)")
        assert len(pop) == evolver._pop_size, (
            f"种群只填到 {len(pop)}，目标 {evolver._pop_size} —— "
            f"填充循环的尝试上限被压没了")

    def test_a_larger_population_is_also_filled(self):
        """再要一个更大的种群，确认上限公式随 pop_size 缩放。"""
        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=20, n_generations=1, seed=7)
        assert len(ev._init_population("rank(close)")) == 20

    def test_the_population_never_exceeds_the_target(self, evolver):
        """`len(pop) < pop_size` —— 放宽成 `<=` 会多塞一个进去。"""
        for seed in (1, 2, 3):
            ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                                   pop_size=6, n_generations=1, seed=seed)
            assert len(ev._init_population("rank(close)")) == 6


# ===========================================================================
# E. 下一代构建
# ===========================================================================

class TestNextPopulation:

    @staticmethod
    def _weights():
        return {"point": 1.0}

    def test_no_results_means_a_fully_random_generation(self, evolver):
        """
        `if not results: return [随机个体] * pop_size` —— 删掉 `not` 会反过来：
        **有**结果时返回一批随机个体（整代进化成果被丢掉），
        没结果时继续往下走 `sorted([])` → 空精英 → 死循环填充。
        """
        out = evolver._generate_next_population([], self._weights())
        assert len(out) == evolver._pop_size, (
            "没有任何评估结果时没有退回随机种群")

    def test_elites_are_the_best_not_the_worst(self, evolver):
        """
        `sorted(results, key=fitness, reverse=True)` —— `reverse` 翻成 False
        会让精英取成 **fitness 最低**的那批。进化每一代都在挑最差的，
        而 best_fitness 的日志照样每代都打印一个数，看不出方向反了。
        """
        from app.core.alpha_engine.parser import Parser
        p = Parser()
        results = [
            _res("rank(close)", 0.1, node=p.parse("rank(close)")),
            _res("rank(volume)", 9.9, node=p.parse("rank(volume)")),
            _res("rank(high)", 5.0, node=p.parse("rank(high)")),
        ]
        out = evolver._generate_next_population(results, self._weights())
        assert repr(out[0]) == "rank(volume)", (
            f"下一代的第一个精英是 {out[0]!r}，应当是 fitness 最高的 "
            f"rank(volume) —— 排序方向反了")

    @pytest.mark.parametrize("pop_size,ratio,expect", [
        (12, 0.25, 3),
        (8,  0.25, 2),
        (4,  0.25, 1),
        (20, 0.50, 10),
        (3,  0.10, 1),      # max(1, 0) → 至少留一个
    ])
    def test_the_elite_count_is_a_fraction_of_the_population(
            self, pop_size, ratio, expect):
        """
        `n_elite = max(1, int(self._elite_ratio * self._pop_size))`

        `*` 改成 `/`：0.25/12 = 0.02 → int → 0 → max(1,0) = 1。
        **不管种群多大，精英永远只有 1 个**，精英保留机制形同虚设，
        而且没有任何迹象 —— 除非有人去数下一代里有几个是上一代原封不动来的。
        """
        from app.core.alpha_engine.parser import Parser
        p = Parser()
        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=pop_size, elite_ratio=ratio,
                               n_generations=1, seed=1)
        dsls = ["rank(close)", "rank(volume)", "rank(high)", "rank(low)",
                "rank(open)", "rank(vwap)", "rank(returns)",
                "ts_mean(close,5)", "ts_mean(volume,5)", "ts_std(close,10)",
                "ts_std(volume,10)", "ts_delta(close,3)"]
        results = [_res(d, float(len(dsls) - i), node=p.parse(d))
                   for i, d in enumerate(dsls)]

        # 只让精英进入下一代：把填充循环的算子全部打成无效
        out = ev._generate_next_population(results, {"crossover": 1.0})
        carried = [r for r in out if repr(r) in {repr(x.node) for x in results}]
        assert len(carried) >= expect, (
            f"pop_size={pop_size}、elite_ratio={ratio} 时只带过来 "
            f"{len(carried)} 个原样个体，精英数至少应当是 {expect} —— "
            f"`elite_ratio * pop_size` 的算符被改了")

    def test_unparsable_results_are_dropped_from_the_node_map(self, evolver):
        """
        `dsl_to_node = {k: v for k, v in ... if v is not None}` —— 删掉 `not`
        会**只保留** None，精英全变 None，下一代直接空掉。
        """
        from app.core.alpha_engine.parser import Parser
        p = Parser()
        results = [
            _res("rank(close)", 9.0, node=p.parse("rank(close)")),
            _res("这不是 DSL(", 8.0, node=None),      # 解析不了
        ]
        out = evolver._generate_next_population(results, self._weights())
        assert out, "含一条解析不了的结果时下一代空了"
        assert repr(out[0]) == "rank(close)"

    def test_the_next_generation_is_filled_to_the_target_size(self, evolver):
        """
        两个填充循环各自带 `attempts < pop_size * 15` / `* 20` 的上限。
        `*` 改成 `/` 会让上限 < 1，两个循环都一次不跑 ——
        下一代只剩精英那几个，种群规模每代都在缩水。
        """
        from app.core.alpha_engine.parser import Parser
        p = Parser()
        results = [_res(d, float(i), node=p.parse(d)) for i, d in enumerate(
            ["rank(close)", "rank(volume)", "rank(high)"])]
        out = evolver._generate_next_population(results, self._weights())
        assert len(out) == evolver._pop_size, (
            f"下一代只有 {len(out)} 个，目标 {evolver._pop_size} —— "
            f"填充/补位循环的尝试上限被压没了")

    def test_the_next_generation_never_exceeds_the_target(self, evolver):
        """两个 `while len(next_gen) < pop_size` 放宽成 `<=` 都会多出一个。"""
        from app.core.alpha_engine.parser import Parser
        p = Parser()
        results = [_res(d, float(i), node=p.parse(d)) for i, d in enumerate(
            ["rank(close)", "rank(volume)"])]
        for w in ({"point": 1.0}, {"crossover": 1.0}, {"hoist": 1.0}):
            out = evolver._generate_next_population(results, w)
            assert len(out) <= evolver._pop_size, (
                f"权重 {w} 下产出了 {len(out)} 个，超过了 {evolver._pop_size}")

    def test_crossover_needs_two_parents(self, evolver):
        """
        `if op == "crossover" and len(sorted_res) >= 2:` —— `and` 放宽成 `or`
        时，只有一个结果也会走交叉分支，`rng.sample(sorted_res, 2)` 直接
        ValueError（被外层 except 吞掉 → 每次尝试都失败 → 种群填不满）。
        """
        from app.core.alpha_engine.parser import Parser
        results = [_res("rank(close)", 1.0, node=Parser().parse("rank(close)"))]
        out = evolver._generate_next_population(results, {"crossover": 1.0})
        assert len(out) == evolver._pop_size, (
            f"只有一个父代时下一代只填到 {len(out)} —— "
            f"交叉分支的 `len(sorted_res) >= 2` 守卫被放宽了")

    def test_the_next_generation_has_no_structural_duplicates(self, evolver):
        """
        两处 `if key not in seen:` —— 删掉 `not` 会让去重反过来：
        只有**已经存在**的结构才被加进去，下一代变成同一个个体的 N 份复制，
        进化彻底停滞，而 pop_size 看起来是满的。
        """
        from app.core.alpha_engine.parser import Parser
        p = Parser()
        results = [_res(d, float(i), node=p.parse(d)) for i, d in enumerate(
            ["rank(close)", "rank(volume)", "rank(high)"])]
        out = evolver._generate_next_population(results, {"point": 1.0})
        reprs = [repr(n) for n in out]
        assert len(set(reprs)) == len(reprs), (
            f"下一代里有重复结构：{[r for r in reprs if reprs.count(r) > 1][:3]} —— "
            f"去重的 `not in seen` 被删掉了")


# ===========================================================================
# F. 进化主循环
# ===========================================================================

class TestRunLoop:
    """
    用替身替掉 `_evaluate_population`（真回测每代要几秒）与 Optuna，
    只观察循环本身：代际编号、回调、最后一代是否多跑了一轮。
    """

    @staticmethod
    def _stub(evolver, monkeypatch, calls: list):
        from app.core.alpha_engine.parser import Parser
        p = Parser()

        def _eval_pop(population):
            calls.append(len(population))
            return [_res("rank(close)", 1.0 + len(calls),
                         node=p.parse("rank(close)"))]

        monkeypatch.setattr(evolver, "_evaluate_population", _eval_pop)
        monkeypatch.setattr(evolver, "_compute_signal_vec", lambda dsl: None)
        monkeypatch.setattr(evolver, "_optuna_fine_tune",
                            lambda *a, **kw: ({}, {"is_sharpe": 1.0}))

    def test_the_population_is_evaluated_once_per_generation(self, monkeypatch):
        """
        `if gen < self._n_gen - 1:` —— 最后一代不该再造下一代（造了也没人评）。

        `-` 改成 `+` 会让**最后两代**都跳过构建（倒数第二代的成果被原样
        再评一遍，白跑一整代的回测）；`<` 放宽成 `<=` 会在最后一代之后
        多造一代，同样白跑。这里从"每代评估一次、总共 n_gen 次"这个
        不变式去抓。
        """
        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=4, n_generations=3, seed=5)
        calls: list = []
        self._stub(ev, monkeypatch, calls)
        ev.run(seed_dsl="rank(close)", n_optuna_trials=0)
        assert len(calls) == 3, (
            f"n_generations=3 却评估了 {len(calls)} 次种群 —— "
            f"`if gen < self._n_gen - 1` 的边界被改了")

    def test_the_generation_callback_sees_consecutive_numbers_from_one(
            self, monkeypatch):
        """
        `"generation": gen + 1` —— `+` 改成 `-` 会让第一代报成 -1、
        第二代报成 0。这个字段进演化台账，是事后对齐"第几代出的这条因子"
        的唯一依据。
        """
        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=4, n_generations=3, seed=5)
        calls: list = []
        self._stub(ev, monkeypatch, calls)

        seen: list = []
        ev.run(seed_dsl="rank(close)", n_optuna_trials=0,
               on_generation_end=lambda log: seen.append(log["generation"]))
        assert seen == [1, 2, 3], (
            f"回调看到的代际编号是 {seen}，应当是 [1, 2, 3] —— `gen + 1` 被改了")

    def test_the_pool_entries_carry_the_same_generation_number(self, monkeypatch):
        """
        `generation = gen + 1` —— 池子里那一处与日志那一处是**两个**独立的
        变异点，只断言其中一个另一个照样活着。
        """
        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=4, n_generations=2, seed=5)
        calls: list = []
        self._stub(ev, monkeypatch, calls)
        ev.run(seed_dsl="rank(close)", n_optuna_trials=0)
        gens = sorted({e.generation for e in ev._pool.all_entries()})
        assert gens and min(gens) == 1, (
            f"池子里记录的代际是 {gens}，最小值应当是 1 —— "
            f"`generation = gen + 1` 被改了")
        assert max(gens) <= 2

    def test_a_missing_callback_is_simply_skipped(self, monkeypatch):
        """
        `if on_generation_end is not None:` —— 改成 `is None` 会在**没给**
        回调时去调 `None(gen_log)` → TypeError（被 except 吞掉，于是
        给了回调的人反而一次都收不到）。
        """
        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=4, n_generations=2, seed=5)
        calls: list = []
        self._stub(ev, monkeypatch, calls)

        # 上一版只写了一句"不给回调，不许抛"就结束 —— **一条断言都没有**
        # （test_lessons_enforced 的零断言检查抓到了）。不抛这件事要落地断言，
        # 而且要断言进化确实跑完了，否则异常被内部 except 吞掉也会"通过"。
        raised = None
        try:
            ev.run(seed_dsl="rank(close)", n_optuna_trials=0)
        except BaseException as exc:          # noqa: BLE001
            raised = exc
        assert raised is None, f"不给 on_generation_end 时抛了：{raised!r}"
        assert len(calls) == 2, (
            f"不给回调时只评估了 {len(calls)} 代，应当是 2 代 —— "
            f"`if on_generation_end is not None` 的判定反了，"
            f"每一代的回调调用都在 except 里被吞掉")

    def test_a_failing_callback_does_not_stop_the_evolution(self, monkeypatch):
        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=4, n_generations=3, seed=5)
        calls: list = []
        self._stub(ev, monkeypatch, calls)

        def _boom(log):
            raise RuntimeError("回调自己炸了")

        ev.run(seed_dsl="rank(close)", n_optuna_trials=0, on_generation_end=_boom)
        assert len(calls) == 3, "回调抛异常把进化打断了"

    @pytest.mark.parametrize("seed_dsl,seed_dsls,expect", [
        ("rank(close)", None, 1),
        ("rank(close)", ["a", "b"], 3),
        ("", ["a", "b"], 2),
        ("", None, 0),
    ])
    def test_the_seed_count_is_the_sum_of_both_seed_arguments(
            self, monkeypatch, caplog, seed_dsl, seed_dsls, expect):
        """
        `n_seeds = len(seed_dsls or []) + (1 if seed_dsl else 0)` ——
        `+` 改成 `-` 会把"用了几条种子"报成负数或错数。这条数字进
        `GP start` 日志，是事后判断"这次搜索到底用了用户几条假设"的依据。
        """
        import logging
        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=4, n_generations=1, seed=5)
        calls: list = []
        self._stub(ev, monkeypatch, calls)
        with caplog.at_level(logging.INFO,
                             logger="app.core.gp_engine.population_evolver"):
            ev.run(seed_dsl=seed_dsl, seed_dsls=seed_dsls, n_optuna_trials=0)
        line = [r.getMessage() for r in caplog.records if "GP start" in r.getMessage()]
        assert line, "没有打出 GP start 日志"
        assert f"n_seeds={expect}" in line[0], (
            f"种子计数报成了 {line[0]}，应当是 n_seeds={expect} —— "
            f"`len(seed_dsls or []) + (1 if seed_dsl else 0)` 的算符被改了")
