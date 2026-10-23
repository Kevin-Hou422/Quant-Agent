"""
gp_engine/population_evolver.py —— 第三轮收尾（变异测试驱动）

10.4% → 66.7% → **79.2%**，剩 10 个存活。逐条查完，三类：

  1. **零除守卫的边界被 `clip` 吸收**（L490 / L554 / L843）。
     我在第二轮用 `sharpe_is=1e-9, sharpe_oos=1.0` 打边界 ——
     `>=` 那一支算出 `(1e-9 - 1.0)/1e-9 ≈ -1e9`，**被 `clip(0,1)` 压回 0.0**，
     与 `>` 那一支同解。必须让分子为**正**（`sharpe_oos=0.0`）才分得开。
     这是同一个坑在本项目里的第三次出现（前两次在 alpha_workflows）。

  2. **补位循环把填充循环的缺口补上了**（L637 / L732 的 `*` 与 `<`）。
     `_generate_next_population` 有两个循环：算子填充 + 随机补位。
     把填充循环的尝试上限压没，**补位循环会把种群补齐到 pop_size** ——
     种群规模一模一样，只是里面一个交叉/变异产物都没有，全是随机个体。
     进化退化成随机重启，而日志里的 pop 数完全正常。
     → 要让补位循环"不可用"（让它也填不进去）才能单独考察填充循环。

  3. **`and` 的短路被守卫本身掩盖**（L643）。
     `if op == "crossover" and len(sorted_res) >= 2:` 放宽成 `or` 之后，
     只有一个父代时也进交叉分支，`rng.sample(sorted_res, 2)` 抛 ValueError
     被外层 except 吞掉 —— 然后补位循环又把种群补满，外部看不出差别。
     → 数 `subtree_crossover` 被调了几次。
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core.alpha_engine.parser import Parser
from app.core.alpha_engine.typed_nodes import DataNode, TimeSeriesNode
from app.core.gp_engine.population_evolver import EvalResult, PopulationEvolver


T, N = 120, 5
_P = Parser()


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


@pytest.fixture(scope="module")
def evolver() -> PopulationEvolver:
    # fitness_mode 显式写成 "holdout"：本文件测的是**单段口径**下的过拟合公式
    # 与零除守卫。发布默认已改为 purged_cv（Phase S.1），那条路径下 sharpe_oos
    # 来自 IS 内部 K 折、与 stub 的 oos_report 无关，这些断言会失去观察面。
    # purged_cv 口径另有 test_purged_cv_fitness.py 专测。
    return PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                             pop_size=8, n_generations=3, seed=42,
                             fitness_mode="holdout")


# ===========================================================================
# A. 三处零除守卫 —— 用分子为正的参数才分得开
# ===========================================================================

def test_the_epsilon_boundary_parameters_actually_distinguish():
    """
    先把"这组参数确实分得开"钉住。`sharpe_oos=1.0` 那组两边同解
    （`clip` 把 -1e9 压回 0），`sharpe_oos=0.0` 那组才分得开。
    """
    is_s, oos_s = 1e-9, 0.0
    strict = (float(np.clip((is_s - oos_s) / abs(is_s), 0.0, 1.0))
              if abs(is_s) > 1e-9 else 0.0)
    loose = (float(np.clip((is_s - oos_s) / abs(is_s), 0.0, 1.0))
             if abs(is_s) >= 1e-9 else 0.0)
    assert (strict, loose) == (0.0, 1.0), (
        f"这组参数不再分得开 `>` 与 `>=`（{strict} vs {loose}）")

    # 对照：第二轮用的那组确实分不开
    bad_strict = (float(np.clip((1e-9 - 1.0) / 1e-9, 0.0, 1.0))
                  if abs(1e-9) > 1e-9 else 0.0)
    bad_loose = (float(np.clip((1e-9 - 1.0) / 1e-9, 0.0, 1.0))
                 if abs(1e-9) >= 1e-9 else 0.0)
    assert bad_strict == bad_loose == 0.0, (
        "第二轮那组参数居然分得开了 —— 本文件的来由说明需要更新")


class TestEpsilonGuards:

    def test_single_dataset_path(self, evolver, stub_bt):
        """`_evaluate_one_single`：`if abs(sharpe_is) > 1e-9 and oos_r:`"""
        _StubBt.is_report = _rep(1e-9)
        _StubBt.oos_report = _rep(0.0)
        r = evolver._evaluate_one_single("rank(close)")
        assert r.overfitting_score == 0.0, (
            f"abs(sharpe_is) 恰好 1e-9 时过拟合度是 {r.overfitting_score} —— "
            f"`> 1e-9` 被放宽成了 `>=`（会被算成 1.0）")

    def test_single_dataset_path_just_above(self, evolver, stub_bt):
        _StubBt.is_report = _rep(2e-9)
        _StubBt.oos_report = _rep(0.0)
        assert evolver._evaluate_one_single("rank(close)").overfitting_score \
            == pytest.approx(1.0)

    def test_extract_metrics_path(self):
        """`_extract_metrics`：`if oos_s is not None and abs(is_s) > 1e-9:`"""
        res = SimpleNamespace(is_report=_rep(1e-9), oos_report=_rep(0.0),
                              summary=lambda: "")
        m = PopulationEvolver._extract_metrics(res)
        assert m["overfitting_score"] == 0.0, (
            f"_extract_metrics 在 abs(is_s)==1e-9 时算出 "
            f"{m['overfitting_score']} —— `> 1e-9` 被放宽成了 `>=`")

    def test_extract_metrics_path_just_above(self):
        res = SimpleNamespace(is_report=_rep(2e-9), oos_report=_rep(0.0),
                              summary=lambda: "")
        m = PopulationEvolver._extract_metrics(res)
        assert m["overfitting_score"] == pytest.approx(1.0)

    def test_multi_dataset_path(self, monkeypatch):
        """`_evaluate_one_multi`：`if abs(sharpe_is) > 1e-9:`"""
        import app.core.backtest_engine.realistic_backtester as rb
        import app.core.backtest_engine.multi_dataset_backtester as mb

        class _Bt:
            def __init__(self, *a, **kw):
                pass

            def run(self, dsl, data, oos_dataset=None):
                return SimpleNamespace(is_report=_rep(1e-9))

        class _Multi:
            def __init__(self, *a, **kw):
                pass

            def run(self, dsl, datasets):
                return SimpleNamespace(
                    aggregated_sharpe=0.0,
                    per_dataset={"d": SimpleNamespace(max_drawdown=-0.2,
                                                      error=None)})

        monkeypatch.setattr(rb, "RealisticBacktester", _Bt)
        monkeypatch.setattr(mb, "MultiDatasetBacktester", _Multi)
        monkeypatch.setattr(mb, "compute_multi_dataset_fitness", lambda **kw: 1.0)

        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               multi_datasets={"d": _panel(3)},
                               pop_size=4, n_generations=1, seed=1)
        assert ev._evaluate_one_multi("rank(close)").overfitting_score == 0.0, (
            "_evaluate_one_multi 在 abs(sharpe_is)==1e-9 时进了除法")


# ===========================================================================
# B. 两个循环必须各自对结果负责
# ===========================================================================

_DSLS = ["rank(close)", "rank(volume)", "rank(high)", "rank(low)",
         "rank(open)", "rank(vwap)"]


def _results():
    return [_res(d, float(len(_DSLS) - i), node=_P.parse(d))
            for i, d in enumerate(_DSLS)]


class TestFillAndPadLoops:

    def test_the_operator_fill_loop_actually_produces_individuals(
            self, monkeypatch):
        """
        `while len(next_gen) < pop_size and attempts < pop_size * 15:`
        的 `*` 改成 `/` → 上限掉到 1 以下，算子填充循环**一次都不跑**。

        但下面的随机补位循环会把种群补齐到 pop_size —— **种群规模一模一样**，
        只是里面一个交叉/变异产物都没有，全是随机个体：
        进化退化成随机重启，而日志里的 pop 数完全正常。

        所以不能只数种群大小，要数**算子被调了几次**。
        """
        import app.core.gp_engine.population_evolver as PE
        calls: list = []
        for name in ("point_mutation", "hoist_mutation", "param_mutation"):
            real = getattr(PE, name)
            monkeypatch.setattr(
                PE, name,
                (lambda r: (lambda n: (calls.append(1), r(n))[1]))(real))

        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=12, elite_ratio=0.25,
                               n_generations=1, seed=3)
        out = ev._generate_next_population(_results(), {"point": 1.0})
        assert len(out) == 12
        assert len(calls) >= 6, (
            f"pop_size=12、精英 3 个，算子填充循环却只调用了 {len(calls)} 次 —— "
            f"`attempts < pop_size * 15` 的上限被压没了，"
            f"下一代其实是随机补位凑出来的")

    def test_the_pad_loop_is_load_bearing_when_the_fill_loop_cannot_add(
            self, monkeypatch):
        """
        `while len(next_gen) < pop_size and pad_attempts < pop_size * 20:`
        的 `*` / `<` / `and` 三个变异点。

        正常情况下填充循环就能把种群填满，补位循环根本不跑 ——
        它的边界因此完全观察不到。这里让**所有算子产出同一个节点**
        （重复项被 `not in seen` 拒掉，填充循环一个都加不进去），
        把补位循环变成唯一的填充来源，它的上限就可观察了。
        """
        import app.core.gp_engine.population_evolver as PE
        fixed = _P.parse("rank(close)")          # 与精英重复，必被去重拒掉
        for name in ("point_mutation", "hoist_mutation", "param_mutation"):
            monkeypatch.setattr(PE, name, lambda n: fixed)
        monkeypatch.setattr(PE, "subtree_crossover", lambda a, b: (fixed, fixed))

        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=10, elite_ratio=0.25,
                               n_generations=1, seed=3)
        out = ev._generate_next_population(_results(), {"point": 1.0})
        assert len(out) == 10, (
            f"算子一个都加不进去时，随机补位循环只补到了 {len(out)}/10 —— "
            f"`pad_attempts < pop_size * 20` 的上限被压没了")

    def test_the_pad_loop_does_not_overshoot(self, monkeypatch):
        """补位循环的 `<` 放宽成 `<=`、`and` 放宽成 `or` 都会多补。"""
        import app.core.gp_engine.population_evolver as PE
        fixed = _P.parse("rank(close)")
        for name in ("point_mutation", "hoist_mutation", "param_mutation"):
            monkeypatch.setattr(PE, name, lambda n: fixed)

        for pop in (6, 10, 14):
            ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                                   pop_size=pop, elite_ratio=0.25,
                                   n_generations=1, seed=3)
            out = ev._generate_next_population(_results(), {"point": 1.0})
            assert len(out) <= pop, f"pop_size={pop} 却产出了 {len(out)} 个"

    def test_crossover_is_not_attempted_with_a_single_parent(self, monkeypatch):
        """
        `if op == "crossover" and len(sorted_res) >= 2:` —— `and` 放宽成 `or`
        之后，只有一个父代时也进交叉分支。`rng.sample(sorted_res, 2)` 抛
        ValueError 被外层 except 吞掉，然后补位循环把种群补满 ——
        **外部完全看不出差别**（第二轮那条 `len(out) == pop_size` 因此没杀掉）。

        要数 `subtree_crossover` 被调了几次：`and` 是 0 次，`or` 会真的调到。
        """
        import app.core.gp_engine.population_evolver as PE
        calls: list = []
        real = PE.subtree_crossover
        monkeypatch.setattr(
            PE, "subtree_crossover",
            lambda a, b: (calls.append(1), real(a, b))[1])

        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=6, n_generations=1, seed=1)
        one = [_res("rank(close)", 1.0, node=_P.parse("rank(close)"))]
        out = ev._generate_next_population(one, {"crossover": 1.0})
        assert calls == [], (
            f"只有一个父代时仍然调用了 {len(calls)} 次交叉 —— "
            f"`len(sorted_res) >= 2` 的守卫被 `or` 短路掉了")
        assert len(out) == 6, "补位循环没有把种群补满"

    def test_crossover_is_attempted_with_two_parents(self, monkeypatch):
        """反向：有两个父代时交叉必须真的发生。"""
        import app.core.gp_engine.population_evolver as PE
        calls: list = []
        real = PE.subtree_crossover
        monkeypatch.setattr(
            PE, "subtree_crossover",
            lambda a, b: (calls.append(1), real(a, b))[1])

        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=6, n_generations=1, seed=1)
        ev._generate_next_population(_results()[:2], {"crossover": 1.0})
        assert calls, "有两个父代却一次交叉都没做"

    def test_no_warning_when_the_next_generation_is_exactly_full(self, caplog):
        """
        `if len(next_gen) < self._pop_size:` 的"未填满"告警 ——
        放宽成 `<=` 会在**正好填满**时也告警。这条 warning 是
        "下一代实际参与进化的个体数比声称的少"的唯一信号，
        天天误报就等于没有。
        """
        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=8, elite_ratio=0.25,
                               n_generations=1, seed=3)
        with caplog.at_level(logging.WARNING,
                             logger="app.core.gp_engine.population_evolver"):
            out = ev._generate_next_population(_results(), {"point": 1.0})
        assert len(out) == 8, "用例前提被破坏：下一代没填满"
        msgs = [r.getMessage() for r in caplog.records
                if "下一代种群未填满" in r.getMessage()]
        assert not msgs, (
            f"正好填满却打出了未填满告警：{msgs} —— `< pop_size` 被放宽成了 `<=`")


# ===========================================================================
# C. 两个 while 的**迭代次数** —— 截断掩盖了输出差别，但调用次数掩盖不了
# ===========================================================================

class TestLoopIterationCounts:
    """
    `_generate_next_population` 末尾是 `return next_gen[: self._pop_size]`。
    两个循环的 `<` 放宽成 `<=`、`and` 放宽成 `or` 之后多产出的个体
    **全落在切片之外**，返回值逐元素相同 —— 我第二轮据此把 L637 的
    `<` 写成了"等价变异"。

    那个结论是**错的**：输出看不出来，**调用次数看得出来**。
    多跑一轮 = 多一次 `point_mutation` / `generate_random_alpha` 调用；
    `and`→`or` 更夸张，会一直空转到 `pop_size * 20` 次尝试上限 ——
    每一代白烧几百次随机生成 + 校验。

    能杀就不要写等价证明。这一节把三个点全部改成杀死。
    """

    @staticmethod
    def _distinct_op(counter):
        """每次调用产出一个互不相同、且一定合法的节点。"""
        def _op(node):
            counter.append(1)
            return TimeSeriesNode("ts_mean", DataNode("close"),
                                  2 + len(counter))
        return _op

    def test_the_fill_loop_stops_the_moment_the_population_is_full(
            self, monkeypatch):
        """
        `while len(next_gen) < self._pop_size and attempts < self._pop_size * 15:`

        让每次算子调用都产出一个全新的合法节点 → 每轮必定加一个 →
        "调用次数"就等于"循环跑了几轮"。
        正确：pop_size − n_elite 次；`<=`：多一次。
        """
        import app.core.gp_engine.population_evolver as PE
        calls: list = []
        for name in ("point_mutation", "hoist_mutation", "param_mutation"):
            monkeypatch.setattr(PE, name, self._distinct_op(calls))

        pop_size, ratio = 10, 0.25
        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=pop_size, elite_ratio=ratio,
                               n_generations=1, seed=3)
        out = ev._generate_next_population(_results(), {"point": 1.0})

        n_elite = max(1, int(ratio * pop_size))
        assert len(out) == pop_size
        assert len(calls) == pop_size - n_elite, (
            f"精英 {n_elite} 个、目标 {pop_size} 个，算子却被调用了 "
            f"{len(calls)} 次，应当正好 {pop_size - n_elite} 次 —— "
            f"`len(next_gen) < pop_size` 被放宽成了 `<=`（多跑一轮，"
            f"多出来的个体被末尾的切片丢掉，输出看不出来）")

    def test_the_pad_loop_stops_the_moment_the_population_is_full(
            self, monkeypatch):
        """
        `while len(next_gen) < self._pop_size and pad_attempts < self._pop_size * 20:`

        让算子恒产出重复项（全被 `not in seen` 拒掉）→ 填充循环一个都加不进去
        → 补位循环成为唯一来源，它的迭代次数可观察。

        正确：正好补满就停；`<=` 多补一次；`and`→`or` 会空转到
        `pop_size * 20` 次上限。
        """
        import app.core.gp_engine.population_evolver as PE
        fixed = _P.parse("rank(close)")          # 与精英重复，必被拒
        for name in ("point_mutation", "hoist_mutation", "param_mutation"):
            monkeypatch.setattr(PE, name, lambda n: fixed)

        rand_calls: list = []
        seq = {"i": 0}

        def _rand(**kw):
            rand_calls.append(1)
            seq["i"] += 1
            return TimeSeriesNode("ts_std", DataNode("volume"), 2 + seq["i"])

        monkeypatch.setattr(PE, "generate_random_alpha", _rand)

        pop_size, ratio = 8, 0.25
        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=pop_size, elite_ratio=ratio,
                               n_generations=1, seed=3)
        out = ev._generate_next_population(_results(), {"point": 1.0})

        n_elite = max(1, int(ratio * pop_size))
        assert len(out) == pop_size
        assert len(rand_calls) == pop_size - n_elite, (
            f"精英 {n_elite} 个、目标 {pop_size} 个，随机补位却调用了 "
            f"{len(rand_calls)} 次，应当正好 {pop_size - n_elite} 次 —— "
            f"补位循环的 `< pop_size` 被放宽成了 `<=`，"
            f"或 `and` 被放宽成了 `or`（空转到 {pop_size * 20} 次上限）")

    def test_the_pad_loop_does_not_spin_after_the_population_is_full(
            self, monkeypatch):
        """
        单独把 `and` → `or` 那一格钉死：种群早已填满时，
        补位循环**一次都不该再跑**。

        用一个"填充循环就能填满"的场景（算子产出全新节点），
        断言 `generate_random_alpha` 一次都没被调用。
        `or` 会让它继续跑到 pop_size*20 次。
        """
        import app.core.gp_engine.population_evolver as PE
        calls: list = []
        for name in ("point_mutation", "hoist_mutation", "param_mutation"):
            monkeypatch.setattr(PE, name, self._distinct_op(calls))

        rand_calls: list = []
        real_rand = PE.generate_random_alpha
        monkeypatch.setattr(
            PE, "generate_random_alpha",
            lambda **kw: (rand_calls.append(1), real_rand(**kw))[1])

        ev = PopulationEvolver(is_data=_panel(1), oos_data=_panel(2),
                               pop_size=10, elite_ratio=0.25,
                               n_generations=1, seed=3)
        out = ev._generate_next_population(_results(), {"point": 1.0})
        assert len(out) == 10, "用例前提被破坏：填充循环没填满"
        assert rand_calls == [], (
            f"种群已经填满，随机补位循环却还跑了 {len(rand_calls)} 次 —— "
            f"`len(next_gen) < pop_size and pad_attempts < ...` 的 and "
            f"被放宽成了 `or`，每一代白烧几百次随机生成 + 校验")
