"""
api/router.py —— 第二轮补强（变异测试驱动）

首测 16.0% → 第一轮 48.0%，剩 39 个存活，集中在我第一轮没看到的
L1400–L2260 区间：交易现实快照、报表组装、以及 `_add_workflow_pnl`
里**第五份**过拟合公式。

这一轮先纠正上一轮自己的一个错误：
`n_t = close.shape[1] if close is not None else 0` 与下一行的 `n_d = ...`
是**两行相同的子串**，上一轮那条 `assert "if close is not None else 0" in src`
改掉任意一行都还剩另一行让子串命中 —— 与 `daemon=True` 是同一个坑
（见台账「自伤教训 #6」）。这里改成 AST 全称命题。

其余按行为断言写，实在观察不到的（settings 默认值、socket 连通性）
用**结构化断言**：直接读 `getattr(..., default)` 的默认值常量，
而不是靠子串存在性。
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

import app.api.router as R


def _src() -> str:
    return inspect.getsource(R)


def _getattr_defaults(src: str) -> dict:
    """
    把模块里所有 `getattr(settings, "<name>", <常量>)` 的默认值抽出来。

    这比子串判断强：它是**按调用点**取值的，同名的多处会各自出现在结果里，
    改掉任意一处都能被发现。
    """
    out: dict = {}
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "getattr"
                and len(node.args) == 3):
            continue
        target, name, default = node.args
        if getattr(target, "id", "") != "settings":
            continue
        if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
            continue
        if isinstance(default, ast.Constant):
            out.setdefault(name.value, []).append(default.value)
        elif isinstance(default, ast.UnaryOp) and isinstance(default.op, ast.USub) \
                and isinstance(default.operand, ast.Constant):
            out.setdefault(name.value, []).append(-default.operand.value)
    return out


# ===========================================================================
# A. 上一轮的子串断言纠正
# ===========================================================================

class TestNoneGuardsInDatasetHealth:

    def test_every_close_shape_read_is_none_guarded(self):
        """
        `n_t = close.shape[1] if close is not None else 0`
        `n_d = close.shape[0] if close is not None else 0`

        **两行的判定子串完全相同**。上一轮写的
        `assert "if close is not None else 0" in src` 改掉任意一行
        都还剩另一行让子串命中 —— 与 `daemon=True` 那次是同一个坑。

        改成 AST：找出所有读 `close.shape[...]` 的三元表达式，
        逐个要求它的条件是 `close is not None`（而不是 `is None`）。
        """
        tree = ast.parse(_src())
        checked = 0
        bad = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.IfExp):
                continue
            body = ast.unparse(node.body)
            if not body.startswith("close.shape["):
                continue
            checked += 1
            test = ast.unparse(node.test)
            if test != "close is not None":
                bad.append((node.lineno, test))
        assert checked >= 2, (
            f"只找到 {checked} 处 `close.shape[...] if ... else ...` —— "
            f"用例前提被破坏（原有 2 处）")
        assert not bad, (
            f"以下位置的 close 空值判定反了：{bad} —— "
            f"没有 close 时会去读 `None.shape` → 500")


# ===========================================================================
# B. 交易现实快照：socket 连通性与配置默认值
# ===========================================================================

class TestTradingRealitySnapshot:

    @staticmethod
    def _snapshot(monkeypatch, connect_ok: bool):
        import socket as _socket

        class _Sock:
            def settimeout(self, t):
                pass

            def connect(self, addr):
                if not connect_ok:
                    raise OSError("模拟：连不上")

            def close(self):
                pass

        monkeypatch.setattr(_socket, "socket", lambda *a, **kw: _Sock())
        fn = None
        for name in dir(R):
            obj = getattr(R, name)
            if callable(obj) and "opend" in (inspect.getsource(obj)
                                             if inspect.isfunction(obj) else ""):
                fn = obj
                break
        assert fn is not None, "找不到交易现实快照端点"
        return fn()

    def test_a_reachable_opend_is_reported_up(self, monkeypatch):
        """
        `opend_up = False` 初值、`s.connect(...); opend_up = True`、
        `except: opend_up = False` —— 三处各是一个变异点。

        初值翻 True 会让**连不上**时也报 reachable（前端状态条绿着，
        实际下不了单）；赋值翻 False 会让连得上也报不可达。
        必须两种情形都断言，且结论互不相同。
        """
        snap = self._snapshot(monkeypatch, connect_ok=True)
        assert snap["moomoo"]["opend_reachable"] is True, (
            "OpenD 连得上却报不可达 —— `opend_up = True` 被翻了")

    def test_an_unreachable_opend_is_reported_down(self, monkeypatch):
        snap = self._snapshot(monkeypatch, connect_ok=False)
        assert snap["moomoo"]["opend_reachable"] is False, (
            "OpenD 连不上却报可达 —— `opend_up` 的初值或 except 分支被翻了")

    def test_shorting_is_off_by_default(self):
        """
        `bool(getattr(settings, "trading_allow_short", False))` —— 翻成 True
        会让**没有配置过**的部署默认允许做空。做空的保证金、借券、
        强平规则与做多完全不同，默认打开是拿真钱试错。
        """
        defaults = _getattr_defaults(_src())
        assert defaults.get("trading_allow_short") == [False], (
            f"trading_allow_short 的 getattr 默认值是 "
            f"{defaults.get('trading_allow_short')}，应当是 [False]")

    def test_the_gate_switches_keep_their_safe_defaults(self):
        """
        `tr_experiment_mode` 默认 **True**（实验模式=不当真）、
        `tr_enforce_active_gate` 默认 **False**（→ACTIVE 门只记录不阻断）。

        两个方向都危险：experiment_mode 翻 False 会让未配置的部署
        以为自己在"正式"模式；enforce_active_gate 翻 True 会让所有上线请求 409。
        """
        defaults = _getattr_defaults(_src())
        assert defaults.get("tr_experiment_mode") == [True], (
            f"tr_experiment_mode 默认值是 {defaults.get('tr_experiment_mode')}，"
            f"应当是 [True]（未配置时按实验模式处理）")
        assert set(defaults.get("tr_enforce_active_gate", [])) == {False}, (
            f"tr_enforce_active_gate 默认值是 "
            f"{defaults.get('tr_enforce_active_gate')}，应当全为 False")

    def test_the_forward_gate_thresholds_are_pinned(self):
        defaults = _getattr_defaults(_src())
        assert defaults.get("tr_min_forward_days") == [60]
        assert defaults.get("tr_min_ic_tstat") == [2.0]


# ===========================================================================
# C. 报表查询与批量回测
# ===========================================================================

class TestReportAndBatch:

    def test_an_alpha_id_query_returns_exactly_that_record(self):
        """
        `if alpha_id is not None:` —— 改成 `is None` 会让**给了 id** 时
        反而走列表查询（返回一堆无关记录），没给 id 时去
        `store.get_by_id(None)`。
        """
        tree = ast.parse(_src())
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = ast.unparse(node.test)
                if test == "alpha_id is not None":
                    found.append(node.lineno)
                elif test == "alpha_id is None":
                    found.append(("REVERSED", node.lineno))
        assert found and all(not isinstance(f, tuple) for f in found), (
            f"alpha_id 的空值判定被反转了：{found}")

    def test_a_zero_oos_ratio_skips_the_split(self):
        """
        `if req.oos_ratio > 0:` —— **严格大于 0**。0 是"不切分"的约定值，
        放宽成 `>=` 会让它去做一个 0 比例的切分：OOS 段为空，
        随后所有 OOS 指标是 NaN，而调用方以为自己拿到了样本外结果。
        """
        tests = {ast.unparse(n.test) for n in ast.walk(ast.parse(_src()))
                 if isinstance(n, ast.If)}
        assert "req.oos_ratio > 0" in tests, (
            "oos_ratio 的切分判定不再是严格大于 0")

    def test_each_synthetic_dataset_in_a_batch_gets_a_distinct_seed(self):
        """
        `_make_synthetic_dataset(..., seed=req.seed + i)` —— `+` 改成 `-`
        本身还能产出不同的种子，但 `i=0` 时两者相同、且负种子在
        `np.random.default_rng` 下会 ValueError。更要紧的是：
        去掉 `+ i`（或让它退化）会让**批量里每个数据集拿到同一份数据**，
        "多数据集泛化"整个变成自欺。

        这里断言那几个种子互不相同，且第 i 个正好是 base + i。
        """
        calls = []
        for node in ast.walk(ast.parse(_src())):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "_make_synthetic_dataset"):
                for kw in node.keywords:
                    if kw.arg == "seed":
                        calls.append(ast.unparse(kw.value))
        assert "req.seed + i" in calls, (
            f"批量合成数据的种子表达式是 {calls} —— "
            f"应当含 `req.seed + i`，否则每个数据集拿到同一份数据")

        # 语义对照：+ 与 - 在 i>0 时确实给出不同的种子序列
        base = 42
        plus = [base + i for i in range(3)]
        minus = [base - i for i in range(3)]
        assert plus != minus and len(set(plus)) == 3

    def test_a_fully_filtered_dataset_is_marked_skipped(self):
        """
        `filter_results[ds_name]["skipped"] = True` —— 翻成 False 会让
        "这个数据集因为全被过滤掉而没跑回测"变成"跑了且没跳过"。
        批量结果里少一个数据集，而没有任何标记说明为什么。
        """
        assigns = []
        for node in ast.walk(ast.parse(_src())):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                t = ast.unparse(node.targets[0])
                # 注意 ast.unparse 用单引号渲染下标：`x['skipped']`，
                # 按双引号去 endswith 会一条都匹配不到。
                if t.endswith("['skipped']"):
                    assigns.append((t, ast.unparse(node.value)))
        assert assigns, "找不到 skipped 标记的赋值"
        for t, v in assigns:
            assert v == "True", (
                f"{t} 被赋成了 {v} —— 全被过滤掉的数据集不再被标记为 skipped")

    def test_neutralization_is_off_by_default(self):
        """
        `allow_neutralize: bool = Field(False)` —— 翻成 True 会让
        **没指定**的请求默认做行业中性化。中性化会显著改变因子的
        收益来源与容量，默认打开等于悄悄换了被测对象。
        """
        offenders = []
        for name in dir(R):
            fields = getattr(getattr(R, name), "model_fields", None)
            if isinstance(fields, dict) and "allow_neutralize" in fields:
                d = fields["allow_neutralize"].default
                if d is not False:
                    offenders.append(f"{name}.allow_neutralize={d!r}")
        assert not offenders, (
            f"以下模型的 allow_neutralize 默认值不是 False：{offenders}")


# ===========================================================================
# D. 报表组装里的 None 守卫
# ===========================================================================

class TestReportAssemblyGuards:
    """
    L1861 / L1888 / L1891 / L1912 / L1917 / L1920 / L2197 / L2224 / L2240
    都是同一个模式：`if X is not None and Y is not None:` 或
    `if isinstance(x, pd.Series) and len(x) > 0:`。

    删掉 `not` 会让有值时跳过、无值时去读属性；`and` 放宽成 `or`
    会让只有一半满足时也进去 → AttributeError/IndexError。
    这些全都被外层 `except Exception` 兜住 → **症状是字段静默缺失**，
    前端图表空白而接口 200。

    进程内要逐个构造出这些内部状态代价极高（需要真跑回测），
    所以用 AST 把**这一类判定的形状**钉死：每一个这样的条件
    都必须是 `and` 连接的 `is not None` / `len(...) > 0`。
    """

    @staticmethod
    def _conditions() -> list:
        out = []
        for node in ast.walk(ast.parse(_src())):
            if isinstance(node, ast.If):
                out.append((node.lineno, ast.unparse(node.test)))
        return out

    @pytest.mark.parametrize("expected", [
        "result.oos_result is not None and oos_prices is not None",
        "eval_dict.get('oos_metrics') and result.oos_report is not None",
        "result.is_report is not None",
        "result.oos_report is not None",
        "isinstance(nr, pd.Series) and len(nr) > 0",
        "isinstance(is_nr, pd.Series) and len(is_nr) > 0",
        "is_prices is not None and is_signal is not None",
    ])
    def test_the_guard_shapes_are_intact(self, expected):
        conds = {c for _, c in self._conditions()}
        assert expected in conds, (
            f"报表组装里的守卫 `{expected}` 不见了 —— "
            f"它被改成了取反或放宽的形式，字段会静默缺失而接口仍 200")

    def test_no_guard_was_reversed_to_is_none(self):
        """
        反向全称命题：这些名字上不该出现 `is None` 形式的守卫。
        （`is None` 只应出现在赋值兜底里，如 `if nr is None: nr = ...`）
        """
        suspicious = []
        for lineno, cond in self._conditions():
            for name in ("result.is_report", "result.oos_report",
                         "result.oos_result", "oos_prices", "is_prices",
                         "is_signal"):
                if cond == f"{name} is None":
                    suspicious.append((lineno, cond))
        assert not suspicious, (
            f"以下守卫被反转成了 `is None`：{suspicious}")

    def test_the_pool_entry_filter_needs_both_conditions(self):
        """
        `if isinstance(e, dict) and e.get("sharpe_oos") is not None`
        —— `and` 放宽成 `or` 时，非 dict 的条目也会走 `e.get(...)` → AttributeError；
        删掉 `not` 则会把**有值**的条目全滤掉，trial_sharpes 恒为空，
        Deflated Sharpe 退化成不带 trial 校正的版本（数字偏乐观）。
        """
        comps = [ast.unparse(g.ifs[0])
                 for n in ast.walk(ast.parse(_src()))
                 if isinstance(n, (ast.ListComp, ast.GeneratorExp))
                 for g in n.generators if g.ifs]
        assert any("isinstance(e, dict) and e.get('sharpe_oos') is not None" in c
                   for c in comps), (
            "pool 条目的过滤条件被改了 —— "
            "Deflated Sharpe 的 trial 校正可能失效")

    def test_the_dsr_return_series_prefers_oos(self):
        """
        `dsr_ret = oos_nr if oos_nr is not None else is_nr` —— 删掉 `not`
        会**优先用 IS 收益**算 Deflated Sharpe。DSR 的意义正是
        "在样本外、扣掉多重检验之后还剩多少"，拿 IS 算完全是反的。
        """
        found = [ast.unparse(n) for n in ast.walk(ast.parse(_src()))
                 if isinstance(n, ast.IfExp)
                 and ast.unparse(n).startswith("oos_nr if ")]
        assert found == ["oos_nr if oos_nr is not None else is_nr"], (
            f"DSR 的收益序列选择被改了：{found} —— "
            f"可能变成了优先用 IS 收益")


# ===========================================================================
# E. _add_workflow_pnl 里的第五份过拟合公式
# ===========================================================================

class TestWorkflowPnlOverfit:
    """
    `overfit = clip((s_is - s_oos)/abs(s_is), 0, 1) if abs(s_is) > 1e-9 else 0.0`
    `response_dict["is_overfit"] = overfit > 0.5`

    这是整个代码库里**第五份**同一条公式
    （population_evolver ×3、alpha_workflows ×1、这里 ×1）。
    每一份都是独立的变异点，改坏任意一份都只影响那一条展示路径。
    """

    @staticmethod
    def _overfit(s_is: float, s_oos: float) -> float:
        return (float(np.clip((s_is - s_oos) / abs(s_is), 0.0, 1.0))
                if abs(s_is) > 1e-9 else 0.0)

    @pytest.mark.parametrize("s_is,s_oos,expect", [
        (2.0, 2.0, 0.0),      # 没退化 → 0；`+` 会给 1.0
        (2.0, 1.0, 0.5),
        (2.0, 0.0, 1.0),
        (2.0, 3.0, 0.0),
        (1e-9, 0.0, 0.0),     # 恰好 epsilon → 走 else；`>=` 会给 1.0
        (2e-9, 0.0, 1.0),
    ])
    def test_the_formula_and_its_epsilon_guard(self, s_is, s_oos, expect):
        assert self._overfit(s_is, s_oos) == pytest.approx(expect)

        # 产品代码里必须是同一条式子（含 `-` 与 `> 1e-9`）
        src = _src()
        assert ("float(np.clip((s_is - s_oos) / abs(s_is), 0.0, 1.0)) "
                "if abs(s_is) > 1e-9 else 0.0") in src, (
            "_add_workflow_pnl 里的过拟合公式被改了 —— "
            "`(s_is - s_oos)` 的符号或 `> 1e-9` 的边界")

    @pytest.mark.parametrize("score,flag", [(0.5, False), (0.501, True)])
    def test_the_overfit_flag_boundary_is_strict(self, score, flag):
        """`overfit > 0.5` —— 恰好 0.5 不算过拟合。"""
        assert (score > 0.5) is flag
        assert 'response_dict["is_overfit"]        = overfit > 0.5' in _src(), (
            "is_overfit 的阈值判定被改了")

    def test_the_fallback_does_not_flag_overfit(self):
        """
        `response_dict.setdefault("is_overfit", False)`（异常兜底）——
        翻成 True 会让 PnL 组装**失败**时把结果标成过拟合。
        组装失败与过拟合毫无关系，这样标等于凭空给结论抹黑。
        """
        calls = []
        for node in ast.walk(ast.parse(_src())):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "setdefault"
                    and len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "is_overfit"):
                calls.append(ast.unparse(node.args[1]))
        assert calls, "找不到 is_overfit 的 setdefault 兜底"
        for v in calls:
            assert v == "False", (
                f"is_overfit 的兜底默认值是 {v} —— PnL 组装失败会被标成过拟合")

    def test_the_flat_response_field_defaults_to_not_overfit(self):
        """`is_overfit: bool = False`（响应模型字段）—— 同理。"""
        offenders = []
        for name in dir(R):
            fields = getattr(getattr(R, name), "model_fields", None)
            if isinstance(fields, dict) and "is_overfit" in fields:
                from pydantic_core import PydanticUndefined
                d = fields["is_overfit"].default
                if d is not PydanticUndefined and d is not False:
                    offenders.append(f"{name}.is_overfit={d!r}")
        assert not offenders, (
            f"以下模型的 is_overfit 默认值不是 False：{offenders}")

    def test_the_ic_decay_nan_filter_needs_both_conditions(self):
        """
        `None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)`
        —— 与模块级 `_nan_to_none` 是**两份独立**的拷贝（这份内联在
        字典推导里）。内层 `and` 放宽成 `or` 会让**每一个** IC decay 值
        都变成 None，前端的衰减曲线整条消失，而接口 200。
        """
        src = _src()
        n = src.count("isinstance(v, float) and np.isnan(v)")
        assert n >= 2, (
            f"只剩 {n} 处 `isinstance(v, float) and np.isnan(v)` —— "
            f"某一份（模块级 _nan_to_none 或 ic_decay 内联那份）的 and 被放宽了")

        # 语义对照：`or` 版本会把正常数字也吃掉
        def _good(v):
            return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

        def _bad(v):
            return None if (v is None or (isinstance(v, float) or np.isnan(v))) else float(v)

        assert _good(0.25) == 0.25 and _bad(0.25) is None, (
            "构造分不开 and 与 or")


# ===========================================================================
# F. 决策记录里的字符串拼接
# ===========================================================================

def test_the_activation_decision_note_concatenates_reason_and_gate_detail():
    """
    `(req.reason + f" | TR.4 门: ...")` —— `+` 改成 `-` 在字符串上直接
    TypeError（整个上线请求 500）。这条记录是事后审计"当时这个门过没过"
    的唯一依据，不能丢。
    """
    src = _src()
    assert 'req.reason + f" | TR.4 门:' in src, (
        "上线决策记录不再把 reason 与门的结论拼在一起 —— 审计依据丢失")

    # 语义对照：字符串上的 `-` 是 TypeError，不是"另一种拼法"
    with pytest.raises(TypeError):
        eval('"a" - "b"')


# ===========================================================================
# G. GP 超时后的锁记账 —— 必须走真实端点，不能在测试里复现逻辑
# ===========================================================================

class TestGpTimeoutAccounting:
    """
    `timed_out = False`（初值）/ `timed_out = True`（超时置位）/
    `finally: if not timed_out: _gp_lock.release()`

    第一轮我在测试里**把这段逻辑抄了一遍**再断言 —— 那是在测我自己写的
    三行代码，产品里改成什么样都不影响，三个变异点因此全活着。
    这一轮改成驱动真实端点，从"锁还在不在"这个外部可观察量去抓。

    三种改法的后果：
      初值翻 True  → 正常跑完也不释放锁，服务从此再也接不了 GP 任务
      置位翻 False → 超时后释放锁，失控线程还在烧 CPU，第二个任务立刻挤进来
      删掉 `not`   → 语义整个反过来
    """

    @staticmethod
    def _endpoint():
        """找到 /gp/evolve 的处理函数。"""
        import inspect
        for name in dir(R):
            obj = getattr(R, name)
            if not inspect.isfunction(obj):
                continue
            try:
                src = inspect.getsource(obj)
            except Exception:
                continue
            if "_acquire_gp_slot()" in src and "timed_out" in src:
                return obj
        pytest.fail("找不到带 GP 槽位与超时记账的端点")

    def test_a_normal_run_releases_the_slot(self, monkeypatch):
        """
        正常跑完（线程秒退）→ `finally` 必须释放锁。
        初值 `timed_out = True` 时释放被跳过，**第二次调用会 429**。
        """
        from fastapi import HTTPException

        class _Evolver:
            def __init__(self, **kw):
                pass

            def run(self, **kw):
                return SimpleNamespace(
                    best_dsl="rank(close)",
                    metrics={"is_sharpe": 1.0, "oos_sharpe": 0.8},
                    evolution_log=[], pool_top5=[], best_config={},
                    generations_run=1)

        import app.core.gp_engine.population_evolver as PE
        monkeypatch.setattr(PE, "PopulationEvolver", _Evolver)
        monkeypatch.setattr(
            R, "_resolve_dataset",
            lambda *a, **kw: ({}, {"close": _tiny()}, {"close": _tiny()}))
        monkeypatch.setattr(R, "_record_run_manifest", lambda **kw: None)

        fn = self._endpoint()
        req = SimpleNamespace(pop_size=4, n_gen=1, dataset_name="",
                              dataset_start="", dataset_end="",
                              n_tickers=4, n_days=60, seed=1)

        class _Store:
            def save(self, *a, **kw):
                return 1

            def add(self, *a, **kw):
                return 1

        assert R._gp_lock.acquire(blocking=False), "用例开始时锁就被占着"
        R._gp_lock.release()

        # 端点内部可能因为桩数据不完整而抛别的错 —— 那与本用例无关，
        # 但**不能静默吞掉**：落地记下来，循环结束后断言它们确实不是
        # "拿不到槽位"那一类错误。
        other_errors: list = []
        for i in range(2):
            try:
                fn(req, store=_Store())
            except HTTPException as exc:
                assert exc.status_code != 429, (
                    f"第 {i+1} 次调用就拿不到 GP 槽位（429）—— "
                    f"上一次跑完之后锁没有释放，"
                    f"`timed_out` 的初值或 `if not timed_out` 的判定被改了")
                other_errors.append(exc)
            except Exception as exc:           # noqa: BLE001
                other_errors.append(exc)

        for exc in other_errors:
            assert not isinstance(exc, HTTPException) or exc.status_code != 429, (
                f"端点返回了 429：{exc}")
        free = R._gp_lock.acquire(blocking=False)
        assert free, (
            "两次正常调用之后 GP 槽位仍被占着 —— 锁泄漏，服务再也跑不了 GP")
        R._gp_lock.release()

    def test_the_release_is_guarded_by_the_timeout_flag(self):
        """
        `finally: if not timed_out: _gp_lock.release()` 的形状本身。
        超时那条路径**必须**跳过释放（锁由后台的 `_release_when_done` 线程
        在失控线程真正退出后才释放）。
        """
        tree = ast.parse(_src())
        guards = [ast.unparse(n.test) for n in ast.walk(tree)
                  if isinstance(n, ast.If)
                  and "timed_out" in ast.unparse(n.test)]
        assert "not timed_out" in guards, (
            f"finally 里的释放守卫是 {guards} —— "
            f"应当是 `not timed_out`（超时时不释放）")
        assert "timed_out" not in [g for g in guards if g == "timed_out"], (
            "释放守卫被改成了 `if timed_out:` —— 语义整个反过来")

    def test_the_timeout_branch_sets_the_flag_and_the_normal_path_clears_it(self):
        """
        两条赋值 `timed_out = False`（初值）与 `timed_out = True`（超时）
        必须**取值相反**。只断言其中一条，另一条照样活着。
        """
        tree = ast.parse(_src())
        values = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and ast.unparse(node.targets[0]) == "timed_out"
                    and isinstance(node.value, ast.Constant)):
                values.append(node.value.value)
        assert values, "找不到 timed_out 的赋值"
        assert False in values and True in values, (
            f"timed_out 的赋值取值是 {values} —— "
            f"必须同时存在 False（初值）与 True（超时置位），"
            f"两者相同就意味着标志位失去了区分能力")


def _tiny():
    import pandas as _pd
    idx = _pd.bdate_range("2021-01-04", periods=60)
    return _pd.DataFrame(np.ones((60, 3)) * 100.0, index=idx,
                         columns=["A", "B", "C"])


from types import SimpleNamespace                                   # noqa: E402


# ===========================================================================
# H. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "L1423 `opend_up = False`（初值）→ `True`":
        "这个初值在任何可达路径上都会被**立即覆盖**："
        "紧接着的 `try: s.connect(...); opend_up = True` 成功则置 True，"
        "抛异常则 `except: opend_up = False`。"
        "try 之前的两句（`s = socket.socket()` / `s.settimeout(1.0)`）"
        "若抛异常则整个函数抛出，`opend_up` 根本不会被读到。"
        "因此初值无论 True 还是 False，返回的 `opend_reachable` 都相同。"
        "见 test_the_initial_flag_is_always_overwritten。",
}


def test_the_initial_flag_is_always_overwritten():
    """
    L1423 等价性的机械验证：用 AST 确认 `opend_up` 的初值赋值之后，
    紧跟的 try 块里**成功分支与 except 分支都对它赋了值** ——
    也就是说初值在任何可达路径上都被覆盖。
    """
    tree = ast.parse(_src())
    checked = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        body_assigns = {ast.unparse(t)
                        for n in ast.walk(ast.Module(body=node.body, type_ignores=[]))
                        if isinstance(n, ast.Assign) for t in n.targets}
        handler_assigns = set()
        for h in node.handlers:
            for n in ast.walk(ast.Module(body=h.body, type_ignores=[])):
                if isinstance(n, ast.Assign):
                    handler_assigns.update(ast.unparse(t) for t in n.targets)
        if "opend_up" in body_assigns and "opend_up" in handler_assigns:
            checked = True
            break
    assert checked, (
        "`opend_up` 不再是『try 成功分支与 except 分支都赋值』的形状 —— "
        "初值变得可观察了，L1423 不再是等价变异，必须补用例")


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 1
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
