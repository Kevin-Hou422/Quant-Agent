"""
agent/_tools.py —— QuantTools 的守卫与默认值（变异测试驱动）

来由：22 个变异点，首测击杀率 **4.5%**（存活 21）。既有覆盖
（`tests/unit/test_agent_fallback.py`）只测了 LLM 不可用时的回退文案。

这个模块是 **LLM 唯一能碰到产品的地方**，它的默认值直接决定研究结论
是否可信。存活的变异点里有两个是真正危险的：

  - `allow_synthetic: bool = False` 翻成 True
    → 不给数据集名也能默默构造出 QuantTools，跑出来的 Sharpe/OOS
      全是随机游走的数字，而前端徽章照样显示得像真的
  - `use_test_set: bool = False` 翻成 True
    → 每一次普通回测都去碰 Test 段（GP 全程未见的真实样本外）。
      Test 段被反复看过之后就不再是 holdout，**过拟合检测整个失效**，
      而 `is_true_holdout` 字段还会诚实地报 True，读的人更容易被骗

其余存活项是一批 `if not X:` 的 fail-closed 守卫、`and` 短路条件、
以及 `ensure_ascii=False`（改成 True 会把中文诊断变成 \\uXXXX 转义）。

测法：全部用 `allow_synthetic=True` 的小规模合成数据集构造（会话级 fixture，
构造一次约 2 秒），只测控制流与默认值，不依赖任何回测数值。
"""
from __future__ import annotations

import json

import pytest

from app.agent._tools import QuantTools


@pytest.fixture(scope="module")
def tools() -> QuantTools:
    """小规模合成数据集；显式 allow_synthetic=True（正是被测的那道闸）。"""
    return QuantTools(n_tickers=6, n_days=260, allow_synthetic=True, n_trials=7)


# ===========================================================================
# A. 合成数据的 fail-closed 闸门
# ===========================================================================

class TestSyntheticFailClosed:

    def test_constructing_without_a_dataset_name_is_refused_by_default(self):
        """
        `allow_synthetic: bool = False` —— 默认翻成 True 之后，
        **不给数据集名**也能构造成功，整套工具静默跑在随机游走上。
        研究结论建立在噪声上而没有任何提示，这是最坏的一种失败。
        """
        with pytest.raises(RuntimeError) as ei:
            QuantTools(n_tickers=4, n_days=120)
        msg = str(ei.value)
        assert "allow_synthetic" in msg, (
            f"拒绝的理由里没提 allow_synthetic，用户不知道怎么显式开启：{msg}")
        assert "合成" in msg or "synthetic" in msg.lower()

    def test_explicitly_allowing_synthetic_does_construct(self, tools):
        assert tools.data_source == "synthetic"
        assert tools.is_synthetic is True

    def test_a_failing_real_dataset_is_not_silently_downgraded(self, monkeypatch):
        """
        `if not allow_synthetic: raise` —— 删掉 `not` 会把 fail-closed 变成
        fail-open：真实数据集加载失败时**静默**换成随机游走，
        而 `data_source` 还写着 `real:<name>`，前端徽章直接说谎。
        """
        import app.core.data_engine.dataset_registry as reg

        def _boom(*a, **kw):
            raise RuntimeError("模拟：网络不通")

        monkeypatch.setattr(reg, "load_registry_dataset", _boom)
        with pytest.raises(RuntimeError) as ei:
            QuantTools(n_tickers=4, n_days=120, dataset_name="whatever")
        assert "拒绝回退到合成数据" in str(ei.value)

    def test_a_failing_real_dataset_may_be_downgraded_when_allowed(self, monkeypatch):
        """反向：显式允许时才降级，而且来源标识必须跟着改成 fallback。"""
        import app.core.data_engine.dataset_registry as reg

        def _boom(*a, **kw):
            raise RuntimeError("模拟：网络不通")

        monkeypatch.setattr(reg, "load_registry_dataset", _boom)
        q = QuantTools(n_tickers=4, n_days=120, dataset_name="whatever",
                       allow_synthetic=True)
        assert q.data_source == "synthetic(fallback)", (
            f"降级之后来源标识仍是 {q.data_source!r} —— 展示层会继续声称是真实数据")
        assert q.is_synthetic is True

    def test_the_registry_is_loaded_with_the_cache_enabled(self, monkeypatch):
        """
        `load_registry_dataset(..., use_cache=True)` —— 翻成 False 会让每次
        构造 QuantTools 都重新拉一遍整个数据集。免费数据源有速率限制，
        这不是"慢一点"，是会被限流到拿不到数据。
        """
        import app.core.data_engine.dataset_registry as reg
        seen = {}

        class _DS:
            data = {}

        def _spy(name, start=None, end=None, **kw):
            seen.update(kw)
            raise RuntimeError("到此为止，只看调用参数")

        monkeypatch.setattr(reg, "load_registry_dataset", _spy)
        with pytest.raises(RuntimeError):
            QuantTools(n_tickers=4, n_days=120, dataset_name="x")
        assert seen.get("use_cache") is True, (
            f"load_registry_dataset 的 use_cache 实参是 {seen.get('use_cache')!r} —— "
            f"缓存被关掉了")

    def test_a_per_run_dataset_failure_is_also_fail_closed(self, monkeypatch):
        """
        `if not self._allow_synthetic: raise` —— 这是 **per-run** 那条路径
        （`tool_run_gp_optimization(dataset_name=...)`），与构造期那道闸独立。
        删掉 `not` 会让这次 GP 静默改用会话数据，而返回里报的还是指定的数据集。
        """
        import app.agent._tools as T

        def _boom(*a, **kw):
            raise RuntimeError("模拟：数据集不存在")

        monkeypatch.setattr(T, "load_real_dataset", _boom)

        strict = QuantTools(n_tickers=4, n_days=120, allow_synthetic=True)
        strict._allow_synthetic = False          # 只改这一处，模拟严格实例
        with pytest.raises(RuntimeError) as ei:
            strict.tool_run_gp_optimization(seed_dsl="rank(close)",
                                            dataset_name="does_not_exist",
                                            pop_size=4, n_generations=1)
        assert "拒绝静默改用会话数据" in str(ei.value), (
            f"per-run 数据集加载失败时的拒绝理由不对：{ei.value}")

    def test_a_per_run_downgrade_relabels_the_data_source(self, monkeypatch):
        """
        反向：显式允许合成时才降级，而且 `data_source` 必须跟着改成 fallback。
        标识不跟着走，前端徽章就会在整个会话里继续声称是真实数据。
        """
        import app.agent._tools as T

        def _boom(*a, **kw):
            raise RuntimeError("模拟：数据集不存在")

        monkeypatch.setattr(T, "load_real_dataset", _boom)
        lax = QuantTools(n_tickers=4, n_days=120, allow_synthetic=True)
        before = lax.data_source
        lax.tool_run_gp_optimization(seed_dsl="rank(close)",
                                     dataset_name="does_not_exist",
                                     pop_size=4, n_generations=1)
        assert "fallback" in lax.data_source and lax.data_source != before, (
            f"降级之后来源标识仍是 {lax.data_source!r}（原 {before!r}）—— "
            f"展示层会继续声称是真实数据")


# ===========================================================================
# B. Test 段（真实 holdout）的闸门
# ===========================================================================

class TestHoldoutGate:

    def test_a_plain_backtest_never_touches_the_test_set(self, tools):
        """
        `use_test_set: bool = False` —— 默认翻成 True 之后，**每一次**普通
        回测都会去碰 Test 段。Test 段被反复看过就不再是样本外，
        过拟合检测（IS vs OOS 的差）从此测的是同一批数据。
        """
        assert tools._test_data is not None, "用例前提被破坏：没有 Test 段"
        out = json.loads(tools.tool_run_backtest("rank(close)"))
        assert out["is_true_holdout"] is False, (
            "默认回测就用上了 Test 段 —— `use_test_set` 的默认值被翻成了 True，"
            "真实样本外从此被污染")

    def test_asking_for_the_test_set_does_use_it(self, tools):
        """
        `if use_test_set and self._test_data is not None:` —— 把 `is not None`
        改成 `is None`，显式要求 Test 段时反而拿到 Validate 段，
        而 `is_true_holdout` 报 False（至少还诚实）；反过来 `and` 放宽成 `or`
        则会在**没要求**时也用 Test 段，且报 True —— 那是直接撒谎。
        """
        out = json.loads(tools.tool_run_backtest("rank(close)", use_test_set=True))
        assert out["is_true_holdout"] is True, (
            "显式要求 Test 段却没用上 —— `self._test_data is not None` 的判定反了")

    def test_the_holdout_flag_distinguishes_the_two_calls(self, tools):
        """
        `is_true_holdout = True` / `= False` 两条赋值各自都会被翻。
        只断言其中一条时另一条照样活着 —— 必须**两条都断言，且互不相同**。
        """
        a = json.loads(tools.tool_run_backtest("rank(close)"))["is_true_holdout"]
        b = json.loads(tools.tool_run_backtest("rank(close)",
                                               use_test_set=True))["is_true_holdout"]
        assert (a, b) == (False, True), (
            f"两次调用的 is_true_holdout 是 {(a, b)}，应当是 (False, True) —— "
            f"标志位的取值被翻了")

    def test_requesting_the_test_set_without_one_falls_back_honestly(self, tools):
        """没有 Test 段时要求 holdout，必须回退到 Validate 段并如实报 False。"""
        saved = tools._test_data
        try:
            tools._test_data = None
            out = json.loads(tools.tool_run_backtest("rank(close)", use_test_set=True))
            assert out["is_true_holdout"] is False, (
                "没有 Test 段却报告 is_true_holdout=True —— 这个字段在撒谎")
        finally:
            tools._test_data = saved

    def test_a_failed_backtest_is_not_reported_as_overfit(self, tools):
        """
        回测失败时的兜底指标里 `"is_overfit": False` —— 翻成 True 会让
        **每一条解析失败的 DSL** 都被标成过拟合。GP 的淘汰逻辑读这个字段，
        于是"语法错"和"过拟合"被混成同一件事，诊断信息全乱。
        """
        out = json.loads(tools.tool_run_backtest("this_is_not_a_function(close)"))
        assert out["is_sharpe"] is None, "用例前提被破坏：这条 DSL 居然跑通了"
        assert out["is_overfit"] is False, (
            "回测失败的兜底把 is_overfit 报成了 True —— "
            "语法错会被当成过拟合")
        assert out["overfitting_score"] == 0.0


# ===========================================================================
# C. GP 入口的参数处理
# ===========================================================================

class TestGpEntryArguments:

    def test_a_non_list_seed_json_is_discarded(self, tools, monkeypatch):
        """
        `if not isinstance(seed_dsls_list, list): seed_dsls_list = None` ——
        删掉 `not` 会反过来：**合法的 list** 被丢掉、而 dict/str 被原样
        传进 PopulationEvolver，那里会以难懂的方式崩掉。
        """
        import app.core.gp_engine.population_evolver as PE
        seen = {}

        class _Stub:
            def __init__(self, **kw):
                seen.update(kw)

            def run(self, **kw):
                seen["run_kwargs"] = kw
                raise RuntimeError("到此为止，只看传进来的参数")

        monkeypatch.setattr(PE, "PopulationEvolver", _Stub)

        tools.tool_run_gp_optimization(seed_dsls_json='{"not": "a list"}',
                                    pop_size=4, n_generations=1)
        assert seen["run_kwargs"]["seed_dsls"] is None, (
            f"非 list 的 seed_dsls_json 被原样传了下去："
            f"{seen['run_kwargs']['seed_dsls']!r}")

    def test_a_list_seed_json_is_kept(self, tools, monkeypatch):
        import app.core.gp_engine.population_evolver as PE
        seen = {}

        class _Stub:
            def __init__(self, **kw):
                seen.update(kw)

            def run(self, **kw):
                seen["run_kwargs"] = kw
                raise RuntimeError("stop")

        monkeypatch.setattr(PE, "PopulationEvolver", _Stub)
        tools.tool_run_gp_optimization(seed_dsls_json='["rank(close)", "rank(volume)"]',
                                    pop_size=4, n_generations=1)
        assert seen["run_kwargs"]["seed_dsls"] == ["rank(close)", "rank(volume)"], (
            "合法的 list 种子被丢掉了 —— `if not isinstance(...)` 的 not 被删掉了")

    def test_an_explicit_factor_family_is_not_overwritten_by_detection(
            self, tools, monkeypatch):
        """
        `if not factor_family:` —— 删掉 `not` 会让**调用方显式指定的家族**
        被自动探测结果覆盖，而空家族反而不探测。
        """
        import app.core.gp_engine.population_evolver as PE
        seen = {}

        class _Stub:
            def __init__(self, **kw):
                seen.update(kw)

            def run(self, **kw):
                raise RuntimeError("stop")

        monkeypatch.setattr(PE, "PopulationEvolver", _Stub)
        tools.tool_run_gp_optimization(seed_dsl="ts_std(returns,20)",
                                    factor_family="momentum",
                                    pop_size=4, n_generations=1)
        assert seen["factor_family"] == "momentum", (
            f"显式指定的 momentum 被改写成了 {seen['factor_family']!r} —— "
            f"`if not factor_family` 的 not 被删掉了")

    def test_an_empty_factor_family_is_auto_detected(self, tools, monkeypatch):
        import app.core.gp_engine.population_evolver as PE
        seen = {}

        class _Stub:
            def __init__(self, **kw):
                seen.update(kw)

            def run(self, **kw):
                raise RuntimeError("stop")

        monkeypatch.setattr(PE, "PopulationEvolver", _Stub)
        tools.tool_run_gp_optimization(seed_dsl="ts_std(returns,20)",
                                    factor_family="",
                                    pop_size=4, n_generations=1)
        assert seen["factor_family"], (
            "空家族没有被自动探测 —— 自动识别整个失效")

    @pytest.mark.parametrize("pop,n_seeds,expect", [
        (4,  0, 4),        # 种子少 → 用 pop_size
        (4,  5, 9),        # 种子多 → 5 + 4
        (20, 5, 20),
        (2,  1, 5),        # 1 + 4 = 5 > 2
    ])
    def test_the_effective_population_leaves_room_for_the_seeds(
            self, tools, monkeypatch, pop, n_seeds, expect):
        """
        `effective_pop = max(pop_size, len(seed_dsls_list or []) + 4)`

        `+` 改成 `-` 时，种子比 pop_size 多的情况下种群反而比种子还小，
        初始化会把用户给的种子截掉一部分 —— 用户以为自己的假设进了搜索，
        其实根本没进。`*`/`/` 同理。只断言"能跑"抓不到，必须断确切的值。
        """
        import app.core.gp_engine.population_evolver as PE
        seen = {}

        class _Stub:
            def __init__(self, **kw):
                seen.update(kw)

            def run(self, **kw):
                raise RuntimeError("stop")

        monkeypatch.setattr(PE, "PopulationEvolver", _Stub)
        seeds = json.dumps([f"rank(close)+{i}" for i in range(n_seeds)]) \
            if n_seeds else ""
        tools.tool_run_gp_optimization(seed_dsl="rank(close)", seed_dsls_json=seeds,
                                    pop_size=pop, n_generations=1)
        assert seen["pop_size"] == expect, (
            f"pop_size={pop}、{n_seeds} 个种子时有效种群是 {seen['pop_size']}，"
            f"应当是 max({pop}, {n_seeds}+4)={expect}")


# ===========================================================================
# D. AST 变异入口
# ===========================================================================

class TestMutateAst:

    def test_an_unknown_mutation_target_does_not_blow_up(self, tools):
        """
        `if mutation_target and mutation_target in _ALL_OPS:` —— `and` 放宽成
        `or` 时，未登记的 target 会走进 `_ALL_OPS[target]` → KeyError。
        这个参数来自 CriticResult 的字符串字段，不是受控枚举。
        """
        out = tools.tool_mutate_ast("rank(close)", mutation_target="no_such_op")
        d = json.loads(out)
        assert "mutated_dsl" in d, f"未登记的 mutation_target 让工具崩了：{out[:200]}"

    def test_a_known_mutation_target_is_actually_used(self, tools):
        out = json.loads(tools.tool_mutate_ast("rank(ts_mean(close,10))",
                                               mutation_target="param"))
        assert out["mutation_type"] != "parse_failed"

    def test_an_unparsable_dsl_returns_the_original(self, tools):
        out = json.loads(tools.tool_mutate_ast("this is not dsl("))
        assert out["mutation_type"] == "parse_failed"
        assert out["mutated_dsl"] == "this is not dsl("


# ===========================================================================
# E. LLM 分支的短路
# ===========================================================================

class _Llm:
    """记录被调用次数的 LLM 替身。"""

    def __init__(self, text="point"):
        self.text = text
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1

        class _R:
            content = self.text
        return _R()


class TestLlmShortCircuit:

    def test_without_an_llm_the_fallback_dsl_path_is_used(self):
        """
        `if self._llm is not None:` —— 改成 `is None` 会在**没有 LLM** 时
        去调 `self._llm.invoke` → AttributeError；有 LLM 时反而走回退。
        """
        q = QuantTools(n_tickers=4, n_days=120, allow_synthetic=True)
        out = json.loads(q.tool_generate_alpha_dsl("动量"))
        assert "[Fallback]" in out["explanation"], (
            f"没有 LLM 却没走回退路径：{out}")

    def test_with_an_llm_the_llm_path_is_used(self):
        llm = _Llm(text='{"dsl": "rank(close)", "explanation": "from llm"}')
        q = QuantTools(n_tickers=4, n_days=120, allow_synthetic=True, llm=llm)
        q.tool_generate_alpha_dsl("动量")
        assert llm.calls == 1, (
            f"配了 LLM 却调用了 {llm.calls} 次 —— `self._llm is not None` 的判定反了")

    def test_an_empty_reason_does_not_reach_the_llm(self):
        """
        `if self._llm is None or not reason: return None` —— 删掉 `not` 之后，
        **空**的 reason 反而会被送进 LLM（一次没有信息量的付费调用），
        而有内容的 reason 被直接丢掉（LLM 指导整个失效）。
        """
        llm = _Llm()
        q = QuantTools(n_tickers=4, n_days=120, allow_synthetic=True, llm=llm)
        assert q._llm_guide_mutation("rank(close)", "") is None
        assert llm.calls == 0, (
            "reason 为空时仍然调用了 LLM —— `or not reason` 的 not 被删掉了")

    def test_a_non_empty_reason_does_reach_the_llm(self):
        llm = _Llm()
        q = QuantTools(n_tickers=4, n_days=120, allow_synthetic=True, llm=llm)
        assert q._llm_guide_mutation("rank(close)", "过拟合") == "point"
        assert llm.calls == 1, "有内容的 reason 没有送进 LLM —— 指导路径失效"


# ===========================================================================
# F. Optuna 试验次数
# ===========================================================================

class TestOptunaTrials:

    def test_zero_trials_means_use_the_session_default(self, tools, monkeypatch):
        """
        `n = n_trials if n_trials > 0 else self._n_trials` —— **严格大于 0**。
        放宽成 `>= 0` 之后，`n_trials=0`（"用默认值"的约定写法）会被当成
        "真的跑 0 次"，Optuna 一个 trial 都不跑就返回，
        而返回里 `n_trials: 0` 没人会当成错误。
        """
        import app.core.ml_engine.alpha_optimizer as AO
        seen = {}

        class _Stub:
            def __init__(self, **kw):
                seen["n"] = kw.get("n_trials")

            def optimize(self, *a, **kw):
                raise RuntimeError("stop")

        monkeypatch.setattr(AO, "AlphaOptimizer", _Stub)
        tools.tool_run_optuna("rank(close)", n_trials=0)
        assert seen.get("n") == tools._n_trials, (
            f"n_trials=0 时用了 {seen.get('n')} 次试验，应当回到会话默认值 "
            f"{tools._n_trials} —— `n_trials > 0` 被放宽成了 `>= 0`")

    def test_an_explicit_trial_count_wins(self, tools, monkeypatch):
        import app.core.ml_engine.alpha_optimizer as AO
        seen = {}

        class _Stub:
            def __init__(self, **kw):
                seen["n"] = kw.get("n_trials")

            def optimize(self, *a, **kw):
                raise RuntimeError("stop")

        monkeypatch.setattr(AO, "AlphaOptimizer", _Stub)
        tools.tool_run_optuna("rank(close)", n_trials=3)
        assert seen.get("n") == 3


# ===========================================================================
# G. 输出编码与可选字段
# ===========================================================================

class TestOutputEncoding:

    def test_chinese_stays_readable_in_the_json_output(self, tools):
        """
        `json.dumps(..., ensure_ascii=False)` —— 翻成 True 会把中文诊断
        变成 `\\u8fc7\\u62df\\u5408` 这样的转义串。这串东西要进日志、进
        前端展示、进 AlphaStore 的 reasoning 字段，读的人全看不懂。
        """
        out = tools.tool_interpret_factor("rank(ts_delta(close,5))")
        # 前提：输出里确实有非 ASCII 字符（破折号 —、中文说明等），
        # 否则 ensure_ascii 翻不翻都一样，这条断言就是空的。
        assert any(ord(ch) > 127 for ch in out), (
            f"输出里一个非 ASCII 字符都没有 —— 用例前提不成立，"
            f"换一条会带出非 ASCII 说明的 DSL：{out[:200]}")
        assert "\\u" not in out, (
            f"输出里出现了 \\uXXXX 转义 —— ensure_ascii 被翻成了 True："
            f"{out[:200]}")
        assert json.loads(out)["dsl"] == "rank(ts_delta(close,5))"

    def test_the_diagnosis_block_appears_only_when_there_is_one(self, tools):
        """
        `if diagnosis is not None:` —— 改成 `is None` 会让有诊断时反而不输出、
        没诊断时去读 `None.primary_issue` → AttributeError。
        """
        plain = json.loads(tools.tool_interpret_factor("rank(close)"))
        assert "diagnosis" not in plain, (
            "没有诊断信息时却输出了 diagnosis 块 —— `is not None` 的判定反了")

    def test_saved_reasoning_keeps_the_data_source_readable(self, tools):
        """
        `json.dumps({"data_source": ..., "raw_metrics": ...}, ensure_ascii=False)`
        —— 这段进 AlphaStore 的 reasoning 字段，是事后审计"这条因子当初
        跑在什么数据上"的唯一依据。转义成 \\uXXXX 会让审计的人读不了。
        """
        import inspect
        import app.agent._tools as T
        src = inspect.getsource(T.QuantTools.tool_save_alpha)
        assert "ensure_ascii=False" in src, (
            "tool_save_alpha 的 reasoning 序列化不再是 ensure_ascii=False")


# ===========================================================================
# H. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "L476 `if mutation_type_hint and mutation_type_hint in weights:` → `or`":
        "`mutation_type_hint` 来自 `_llm_guide_mutation`，它的返回值域是 "
        "`{None, 'point', 'hoist', 'param'}`（函数体里只有这三个 key 的循环）。"
        "而 `weights` 来自 `mutation_weights_from_metrics`，其键集合恒含这三个。"
        "于是：hint 为 None 时 `None and ...` 与 `None or (None in weights)` "
        "都为假（None 不是 weights 的键）；hint 非 None 时两个子式都为真。"
        "两种取值对所有可达输入判定相同。"
        "见 test_every_possible_hint_is_already_a_weight_key。",
}


def test_every_possible_hint_is_already_a_weight_key():
    """L476 等价性的机械验证：枚举 hint 的全部可能取值。"""
    import inspect
    import re
    from app.core.gp_engine.fitness import mutation_weights_from_metrics
    import app.agent._tools as T

    src = inspect.getsource(T.QuantTools._llm_guide_mutation)
    m = re.search(r'for key in \(([^)]*)\)', src)
    assert m, "找不到 _llm_guide_mutation 的候选 key 元组 —— 等价性证明需要重做"
    keys = re.findall(r'"(\w+)"', m.group(1))
    assert keys, f"候选 key 解析为空：{m.group(1)!r}"

    weights = mutation_weights_from_metrics(sharpe_oos=0.1, turnover=1.0,
                                            overfit_score=0.1)
    missing = [k for k in keys if k not in weights]
    assert not missing, (
        f"LLM 可能返回的 {missing} 不在权重表里 —— "
        f"`and` 与 `or` 会给出不同结果，L476 不再是等价变异")

    assert None not in weights, "None 成了权重表的键 —— 等价性论证的前提没了"


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 1
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
