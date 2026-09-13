"""
core/workflows/alpha_workflows.py —— 第三轮收尾（变异测试驱动）

首测 20.6% → 第一轮 60.9% → 第二轮 **87.0%**，剩 9 个存活。
逐条查完：4 个是真等价（已证），5 个是我的断言又一次被"下游吸收"绕过。

这一轮抓到的两个新花样：

  1. **`clip` 把差别吸收掉了**。
     `overfit = clip((s_is - s_oos)/abs(s_is), 0, 1) if abs(s_is) > 1e-9 else 0.0`
     我用 `s_is=1e-9, s_oos=1.0` 去打边界 —— `>=` 那一支算出
     `(1e-9-1.0)/1e-9 ≈ -1e9`，**被 clip 压回 0.0**，与 `>` 那一支的 0.0 同解。
     要分开必须让分子为**正**：`s_oos=0.0` → `>` 给 0.0、`>=` 给 1.0。
     这和"只喂极端过拟合样例分不开符号翻转"是同一个坑的两次出现。

  2. **同一层里另一个循环补上了缺口**。
     `_generate_diverse_seeds` 的 Layer 3（变异填充，上限 `n_target * 12`）
     被压没之后，**Layer 4（随机填充，上限 `n_target * 20`）会把数量补齐**。
     只断言"填满 n_target"因此抓不到 Layer 3 的上限。
     而且把 Layer 1/2 全关掉反而让 Layer 3 **完全不可达**
     （它的循环条件带 `and valid_nodes`，没有种子就一次不跑）——
     第二轮那个 fixture 把这个变异点从"漏测"变成了"不可达"。
     正确姿势：留一条模板种子让 Layer 3 可达，然后**数变异算子调了几次**。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.core.workflows.alpha_workflows as W
from app.core.alpha_engine.typed_nodes import DataNode, TimeSeriesNode


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


# ===========================================================================
# A. 零除守卫的边界 —— 必须让 clip 分得开
# ===========================================================================

from types import SimpleNamespace                                   # noqa: E402


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


class TestEpsilonGuardBoundary:
    """
    `if abs(s_is) > 1e-9 else 0.0` —— **严格大于**。

    第二轮用 `s_is=1e-9, s_oos=1.0` 打边界，没杀掉：
      `>`  → 走 else → 0.0
      `>=` → 走除法 → (1e-9 - 1.0)/1e-9 ≈ -1e9 → **clip 到 0.0**
    两边同解。分子必须为**正**才分得开。
    """

    def test_the_boundary_case_that_actually_distinguishes(self, stub_bt):
        _StubBt.is_report = _report(1e-9)
        _StubBt.oos_report = _report(0.0)          # 分子 = 1e-9 - 0 > 0
        m = W._quick_metrics("rank(close)", {"close": 1}, {"close": 1})
        assert m["overfitting_score"] == 0.0, (
            f"abs(s_is) 恰好 1e-9 时过拟合度是 {m['overfitting_score']} —— "
            f"零除守卫的 `> 1e-9` 被放宽成了 `>=`（该值会被算成 1.0）")

    def test_the_premise_that_the_two_branches_differ(self):
        """
        把"这组参数确实分得开"本身钉住 —— 否则哪天 clip 的区间变了，
        上面那条又会退回成一条测不出东西的断言。
        """
        s_is, s_oos = 1e-9, 0.0
        strict = (float(np.clip((s_is - s_oos) / abs(s_is), 0.0, 1.0))
                  if abs(s_is) > 1e-9 else 0.0)
        loose = (float(np.clip((s_is - s_oos) / abs(s_is), 0.0, 1.0))
                 if abs(s_is) >= 1e-9 else 0.0)
        assert strict == 0.0 and loose == 1.0, (
            f"这组参数不再能分开 `>` 与 `>=`（{strict} vs {loose}）—— "
            f"上面那条边界用例需要重挑参数")

    def test_just_above_the_epsilon_does_use_the_formula(self, stub_bt):
        """反向：略大于 epsilon 时必须真的走除法。"""
        _StubBt.is_report = _report(2e-9)
        _StubBt.oos_report = _report(0.0)
        m = W._quick_metrics("rank(close)", {"close": 1}, {"close": 1})
        assert m["overfitting_score"] == pytest.approx(1.0), (
            "abs(s_is) 大于 1e-9 时没有走过拟合公式")


# ===========================================================================
# B. Layer 3 的尝试上限 —— 必须数算子调用，不能数种子条数
# ===========================================================================

class TestSeedMutationLayer:

    @pytest.fixture
    def one_template_only(self, monkeypatch):
        """
        只关 Layer 1（AlphaAgent），**保留一条模板种子**。

        第二轮的 fixture 把 Layer 1/2 全关了，结果 `valid_nodes` 为空，
        Layer 3 的 `while ... and valid_nodes` 一次都不跑 ——
        那个变异点从"漏测"变成了"不可达"，等于白测。
        """
        import app.agent.alpha_agent as AA

        class _NoAgent:
            def __init__(self, *a, **kw):
                raise RuntimeError("本用例关掉 Layer 1")

        monkeypatch.setattr(AA, "AlphaAgent", _NoAgent)
        monkeypatch.setattr(W, "_hypothesis_templates", lambda h: ["rank(close)"])

    @pytest.mark.parametrize("n_target", [8, 12, 20])
    def test_the_mutation_layer_actually_does_the_work(self, one_template_only,
                                                       monkeypatch, n_target):
        """
        `while len(valid_dsls) < n_target and attempts < n_target * 12 and valid_nodes:`

        `*` 改成 `/` 会让上限掉到 1 次尝试。但**Layer 4（随机填充，上限
        `n_target * 20`）会把种子表补齐到 n_target** —— 只断言"填满"
        完全抓不到。必须数 Layer 3 的变异算子被调了几次。

        后果：种子表全变成与假设无关的随机 alpha，
        "从用户假设出发做定向探索"这件事静默消失。
        """
        calls: list = []
        for name in ("point_mutation", "hoist_mutation", "param_mutation"):
            real = getattr(W, name)
            monkeypatch.setattr(
                W, name,
                (lambda r: (lambda node: (calls.append(1), r(node))[1]))(real))

        out = W._generate_diverse_seeds("whatever", n_target=n_target, seed=7)
        assert len(out) == n_target, f"种子表只有 {len(out)} 条"
        assert len(calls) >= n_target - 2, (
            f"目标 {n_target} 条、只有 1 条模板种子，Layer 3 的变异算子却只被调用了 "
            f"{len(calls)} 次 —— `attempts < n_target * 12` 的上限被压没了，"
            f"种子表其实是 Layer 4 的随机 alpha 凑出来的")

    def test_the_mutation_layer_respects_its_attempt_ceiling(
            self, one_template_only, monkeypatch):
        """反向：上限也不能被放宽（`and` → `or` 会让它空转到 12n 次）。"""
        calls: list = []
        for name in ("point_mutation", "hoist_mutation", "param_mutation"):
            real = getattr(W, name)
            monkeypatch.setattr(
                W, name,
                (lambda r: (lambda node: (calls.append(1), r(node))[1]))(real))

        n_target = 8
        W._generate_diverse_seeds("whatever", n_target=n_target, seed=7)
        assert len(calls) <= n_target * 12, (
            f"Layer 3 调用了 {len(calls)} 次变异算子，超过了 {n_target}×12 的上限")


# ===========================================================================
# C. Workflow B 变异循环的**确切**条数
# ===========================================================================

class TestExpandExactMutantCount:

    @pytest.mark.parametrize("n_mut", [3, 5, 8])
    def test_the_mutation_loop_produces_exactly_n_mutants(self, monkeypatch, n_mut):
        """
        `while len(results) - 1 < n_mutations and attempts < n_mutations * 15:`

        `<` 放宽成 `<=` 会多做**一个**变异体，但随后的随机补位循环
        (`while len(results) < n_mutations + 3`) 会**少补一个** ——
        候选总数一模一样。第二轮的上下界断言因此都抓不到。

        这里把三个变异算子换成**确定性产出互不相同节点**的替身，
        于是"成功加入的变异体数"== 算子被调用的次数，可以断言确切值。
        """
        counter = {"n": 0}

        def _fake(node):
            counter["n"] += 1
            # 每次产出一个互不相同、且一定合法的节点。
            # 窗口必须**严格递增且互不重复** —— 第一版用
            # `[3,5,10,20,40,60][n % 6] + n` 会在 n=2 与 n=7 时都得到 12，
            # 重复项被 `if key not in seen` 拒掉，于是多耗一次尝试，
            # "调用次数 == 变异体数"这个前提就不成立了。
            return TimeSeriesNode("ts_mean", DataNode("close"),
                                  2 + counter["n"])

        for name in ("point_mutation", "hoist_mutation", "param_mutation"):
            monkeypatch.setattr(W, name, _fake)

        out = W._expand_for_optimization("rank(ts_mean(close,10))",
                                         n_mutations=n_mut)
        assert counter["n"] == n_mut, (
            f"要 {n_mut} 个变异体，变异算子却被调用了 {counter['n']} 次 —— "
            f"`len(results) - 1 < n_mutations` 的边界被改了"
            f"（每次调用都产出一个全新的合法节点，调用次数 == 变异体数）")
        assert len(out) == n_mut + 3


# ===========================================================================
# D. 进度文案里第二处 None 判定
# ===========================================================================

class TestProgressTextSecondCopy:

    def test_the_delta_suffix_is_omitted_when_there_is_no_oos(self):
        """
        `+ (f"({'↑' if ... else '↓'}{abs(...):.4f} vs input)" if oos_s is not None else "")`

        这是与 `... else 'N/A'` **不同的一处** None 判定（第二轮那条
        计数断言数的是前者，改后者不会让计数变化，所以抓不到）。
        改成 `is None` 会在有 OOS 时不打差值、没 OOS 时对 None 做减法 → TypeError。
        """
        import inspect
        src = inspect.getsource(W)
        assert 'if oos_s is not None else "")' in src, (
            "Workflow B 进度文案里『有 OOS 才附带差值』的判定被改了 —— "
            "没有 OOS 时会对 None 做减法")

        # 语义对照：两种取值渲染出来的东西必须不同
        for oos_s, has_suffix in [(None, False), (0.5, True)]:
            init_oos = 0.2
            suffix = (f"({'↑' if (oos_s or 0) > init_oos else '↓'}"
                      f"{abs((oos_s or 0) - init_oos):.4f} vs input)"
                      if oos_s is not None else "")
            assert bool(suffix) is has_suffix, (
                f"oos_s={oos_s} 时后缀渲染成了 {suffix!r}")


# ===========================================================================
# E. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "L573 `if is_data and set(is_signals) == set(signals):` → `or`":
        "两个子式**不可能独立取值**。`is_signals` 只在 `if is_data:` 内部写入，"
        "且求值失败时 `signals` 与 `is_signals` 被**成对 pop**，"
        "所以：is_data 为真 ⇒ 两个字典键集恒等 ⇒ 第二子式恒真，`and`/`or` 同为真；"
        "is_data 为假 ⇒ is_signals 恒为空，而函数在此之前已有 "
        "`if len(signals) < 2: return None` 的早退，signals 至少 2 项 ⇒ "
        "键集必不相等 ⇒ 第二子式恒假，`and`/`or` 同为假。"
        "见 test_the_two_operands_cannot_vary_independently。",

    "L603 `if denom > 0:` → `>=`":
        "与 gp_engine L272 同源：`rs`/`rr` 都是 `argsort(argsort(x))` 的结果，"
        "即 0..n-1 的排列，去中心化平方和恒为 n(n²−1)/12；"
        "进到这里时 `mask.sum() >= 5`，所以 denom ≥ 10 > 0，"
        "`>` 与 `>=` 对所有可达输入判定相同。"
        "见 test_the_rank_denominator_is_bounded_away_from_zero。",

    "`_try_add` 的三个 return（L335 False / L338 False / L342 True）":
        "两个调用点都是独立语句，返回值被丢弃；函数的全部作用通过闭包里的 "
        "`valid_nodes` / `valid_dsls` / `seen` 副作用完成。"
        "见 test_alpha_workflows_round2.py::"
        "test_try_add_return_value_is_discarded_at_every_call_site。",
}


def test_the_two_operands_cannot_vary_independently():
    """
    L573 等价性的机械验证：把那段循环的语义在测试里复现一遍，
    枚举"求值全成功 / 部分失败 / 无 IS 数据"三种情形，
    确认 `is_data and 键集相等` 与 `is_data or 键集相等` 恒同真同假。
    """
    def _simulate(is_data_truthy: bool, fail_idx: set) -> tuple:
        """
        复现产品那段循环的语义。注意这里的 `except` 是**被模拟的对象本身**
        （产品代码就是这么吞的），不是用例在吞自己的错误 ——
        所以把吞掉的异常落地到 `swallowed`，由调用方断言它确实只在
        预期的下标上发生。
        """
        signals, is_signals = {}, {}
        swallowed: list = []
        for i, dsl in enumerate(["a", "b", "c"]):
            try:
                if i in fail_idx:
                    raise RuntimeError("模拟求值失败")
                signals[dsl] = i
                if is_data_truthy:
                    is_signals[dsl] = i
            except RuntimeError as exc:
                swallowed.append((i, exc))
                signals.pop(dsl, None)
                is_signals.pop(dsl, None)
        assert [i for i, _ in swallowed] == sorted(fail_idx), (
            f"模拟的失败下标是 {[i for i, _ in swallowed]}，"
            f"应当是 {sorted(fail_idx)} —— 用例的模拟本身出了问题")
        if len(signals) < 2:          # 产品代码里的早退
            return None, None
        a = bool(is_data_truthy and set(is_signals) == set(signals))
        b = bool(is_data_truthy or set(is_signals) == set(signals))
        return a, b

    seen = 0
    for truthy in (True, False):
        for fail in (set(), {0}, {1}, {2}):
            a, b = _simulate(truthy, fail)
            if a is None:
                continue
            seen += 1
            assert a == b, (
                f"is_data={truthy}、失败下标={fail} 时 `and` 给 {a}、`or` 给 {b} —— "
                f"L573 不再是等价变异，必须补用例")
    assert seen >= 6, f"只枚举到 {seen} 种可达情形，覆盖不足"


def test_the_rank_denominator_is_bounded_away_from_zero():
    """L603 等价性的机械验证：秩平方和恒为 n(n²−1)/12，n ≥ 5 时 denom ≥ 10。"""
    rng = np.random.default_rng(5)
    for n in range(5, 40):
        for x in (np.full(n, 7.0), np.zeros(n), rng.normal(size=n),
                  np.resize(rng.normal(size=3), n)):
            rs = np.argsort(np.argsort(x)).astype(float)
            assert sorted(rs.tolist()) == [float(i) for i in range(n)]
            c = rs - rs.mean()
            ss = float((c ** 2).sum())
            assert ss == pytest.approx(n * (n * n - 1) / 12.0)
        denom = np.sqrt((n * (n * n - 1) / 12.0) ** 2)
        assert denom >= 10.0, f"n={n} 时 denom={denom} 逼近 0"


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 3
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
