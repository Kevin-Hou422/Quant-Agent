"""
test_lessons_enforced.py — 把 DEV_LESSONS 的每一条从散文变成**可执行的强制检查**

来由：18 条教训里只有个别落成了测试，其余以散文形式躺在 md 里，强制力为零；
凡是没变成检查的，都复发了（§K 写完当天在参数层复发、§R 写完当天在 SSE 路径复发）。
本文件的规则：**每条教训都必须有一个会失败的检查**，否则那条教训等于没写。

命名约定：test_lesson_<字母>_<简述>。新增教训必须同步在此加检查，
由 test_every_lesson_has_an_enforced_check 强制（缺一条即红）。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
APP     = BACKEND / "app"
TESTS   = BACKEND / "tests"
LESSONS = BACKEND / "DEV_LESSONS.md"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _py_files(root: Path):
    for p in root.rglob("*.py"):
        if "__pycache__" not in str(p):
            yield p


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _rel(p: Path) -> str:
    return str(p.relative_to(BACKEND)).replace("\\", "/")


# ===========================================================================
# §A 测试绿 ≠ 系统能跑
#     → 断言必须能失败：禁止容忍 5xx、禁止把断言藏在 if 里
# ===========================================================================

class TestLessonA_AssertionsMustBeAbleToFail:

    def test_no_test_accepts_5xx_as_success(self):
        """
        `assert resp.status_code in (200, 400, 422, 500)` 恒真——端点每次都崩也算通过。
        实测：正是它掩盖了 /api/backtest/walk_forward 从未成功运行过。
        """
        bad = []
        for p in _py_files(TESTS):
            for node in ast.walk(ast.parse(_src(p))):
                if not (isinstance(node, ast.Compare) and len(node.ops) == 1
                        and isinstance(node.ops[0], ast.In)):
                    continue
                left = node.left
                if not (isinstance(left, ast.Attribute) and left.attr == "status_code"):
                    continue
                comp = node.comparators[0]
                if not isinstance(comp, (ast.Tuple, ast.List, ast.Set)):
                    continue
                codes = [e.value for e in comp.elts if isinstance(e, ast.Constant)]
                # 502/503 是"上游明确不可用"的**设计内**响应，允许；
                # 500 是未处理异常，绝不允许当作成功。
                if any(isinstance(c, int) and c >= 500 and c not in (502, 503) for c in codes):
                    bad.append(f"{_rel(p)}:{node.lineno} codes={codes}")
        assert not bad, (
            "以下断言把 5xx 也算通过，等于允许端点崩溃：\n  " + "\n  ".join(bad)
        )

    def test_no_assertions_hidden_behind_status_code_guard(self):
        """`if resp.status_code == 200: assert ...` —— 崩了就跳过检查。"""
        bad = []
        for p in _py_files(TESTS):
            if p.name == Path(__file__).name:
                continue
            for node in ast.walk(ast.parse(_src(p))):
                if not isinstance(node, ast.If) or node.orelse:
                    continue
                t = node.test
                has_assert = any(isinstance(n, ast.Assert) for n in ast.walk(node))
                if not has_assert:
                    continue
                names = {n.attr for n in ast.walk(t) if isinstance(n, ast.Attribute)}
                if "status_code" in names:
                    bad.append(f"{_rel(p)}:{node.lineno} if {ast.unparse(t)[:60]}")
        assert not bad, (
            "以下断言被 status_code 条件包住（请求失败即静默跳过）：\n  " + "\n  ".join(bad)
        )

    def test_no_test_body_fully_wrapped_in_except_pass(self):
        """整个测试体 try/except Exception: pass —— 任何失败都被吞。"""
        bad = []
        for p in _py_files(TESTS):
            if p.name == Path(__file__).name:
                continue
            for fn in [n for n in ast.walk(ast.parse(_src(p)))
                       if isinstance(n, ast.FunctionDef) and n.name.startswith("test")]:
                body = [b for b in fn.body
                        if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))]
                if len(body) != 1 or not isinstance(body[0], ast.Try):
                    continue
                for h in body[0].handlers:
                    tname = ast.unparse(h.type) if h.type else "bare"
                    if h.type is None or "Exception" in tname:
                        inner = ast.dump(h)
                        if "Assert" not in inner and "Raise" not in inner:
                            bad.append(f"{_rel(p)}:{fn.lineno} {fn.name} (except {tname})")
        assert not bad, (
            "以下测试整体被 except 吞掉，永远不会失败：\n  " + "\n  ".join(bad)
        )

    def test_no_tautological_assert(self):
        """`assert ... or True` / `assert True` —— 恒真断言。"""
        bad = []
        for p in _py_files(TESTS):
            if p.name == Path(__file__).name:
                continue
            for node in ast.walk(ast.parse(_src(p))):
                if not isinstance(node, ast.Assert):
                    continue
                t = node.test
                if isinstance(t, ast.Constant) and bool(t.value):
                    bad.append(f"{_rel(p)}:{node.lineno} assert {ast.unparse(t)}")
                elif isinstance(t, ast.BoolOp) and isinstance(t.op, ast.Or):
                    if any(isinstance(v, ast.Constant) and bool(v.value) for v in t.values):
                        bad.append(f"{_rel(p)}:{node.lineno} {ast.unparse(t)[:70]}")
        assert not bad, "以下断言恒真：\n  " + "\n  ".join(bad)


    #: 视为"真断言"的调用（np.testing / pytest 家族）
    _ASSERT_CALLS = {"assert_allclose", "assert_array_equal", "assert_almost_equal",
                     "assert_equal", "assert_series_equal", "assert_frame_equal",
                     "assert_index_equal", "fail", "raises", "approx"}

    @staticmethod
    def _is_fixture(fn: ast.FunctionDef) -> bool:
        return any("fixture" in ast.unparse(d) for d in fn.decorator_list)

    @staticmethod
    def _asserting_helpers(tree: ast.Module) -> set:
        """本模块内**自身含断言**的辅助函数名 —— 调用它们等同于断言（如 _ok）。"""
        out = set()
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            if fn.name.startswith("test"):
                continue
            body = ast.dump(fn)
            if any(isinstance(x, ast.Assert) for x in ast.walk(fn)) or "raises" in body:
                out.add(fn.name)
        return out

    def _test_funcs(self):
        """遍历所有测试函数，产出 (文件, 函数节点, 源码, 辅助函数名集合)。"""
        for p in _py_files(TESTS):
            if p.name == Path(__file__).name:
                continue
            src = _src(p)
            try:
                tree = ast.parse(src)
            except SyntaxError as exc:
                raise AssertionError(f"测试文件无法解析（会被静默排除出检查）：{_rel(p)}: {exc}")
            helpers = self._asserting_helpers(tree)
            for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                       and n.name.startswith("test") and not self._is_fixture(n)]:
                yield p, fn, src, helpers

    def test_no_zero_assertion_test(self):
        """
        零断言的测试 = 被测函数改成空实现也照样通过。
        实测漏网：我自己写的 test_explicit_optin_allows_exposure 就是
        "调用一下 + 注释'不抛即通过'" —— 临时扫描器抓到了，强制套件当时没这条检查。
        """
        bad = []
        for p, fn, _src_, helpers in self._test_funcs():
            has_assert = any(isinstance(n, ast.Assert) for n in ast.walk(fn))
            names = self._ASSERT_CALLS | helpers
            has_call = any(
                isinstance(n, ast.Call) and
                (getattr(n.func, "attr", None) in names
                 or getattr(n.func, "id", None) in names)
                for n in ast.walk(fn))
            has_raises = "raises" in ast.dump(fn)
            if not (has_assert or has_call or has_raises):
                bad.append(f"{_rel(p)}:{fn.lineno} {fn.name}")
        assert not bad, (
            "以下测试没有任何断言（被测实现清空也会通过）：\n  " + "\n  ".join(bad))

    def test_no_generic_conditional_assert(self):
        """
        `if <条件>: assert ...` —— 条件不成立时**一条都不检查**。
        守卫若本身就是契约（如 long_short 两侧都要有），应写成断言而非 if。
        确需守卫（样本量不足等）时，必须记录分支是否真的走到。
        """
        ALLOW_MARK = "分支"          # 注释里说明了为何保留守卫 → 放行
        bad = []
        for p, fn, src, _h in self._test_funcs():
            for node in ast.walk(fn):
                if not isinstance(node, ast.If) or node.orelse:
                    continue
                if not any(isinstance(x, ast.Assert) for x in ast.walk(node)):
                    continue
                # if 体以 return/raise 收尾 → 提前返回模式，其后代码即 else 分支，
                # 两条路都被检查到，不算"跳过断言"。
                if node.body and isinstance(node.body[-1], (ast.Return, ast.Raise)):
                    continue
                seg = ast.get_source_segment(src, node) or ""
                # 分支内自增计数器（记录是否走到）或注释说明 → 认为是有意保留
                if "+= 1" in seg or ALLOW_MARK in seg:
                    continue
                bad.append(f"{_rel(p)}:{node.lineno} if {ast.unparse(node.test)[:56]}")
        assert not bad, (
            "以下断言藏在 if 之后（条件不成立就什么都不检查）：\n  "
            + "\n  ".join(bad)
            + "\n修法：守卫即契约就写成断言；确需守卫则在分支内计数并断言计数>0。")

    def test_no_swallowed_exception_inside_test(self):
        """测试里的 except 必须 fail/raise/断言，否则被测行为出错也不会红。"""
        bad = []
        for p, fn, _src_, _h in self._test_funcs():
            for h in [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]:
                d = ast.dump(h)
                if "Assert" in d or "Raise" in d or "fail" in d or "skip" in d:
                    continue
                # 把异常**收集到变量**（供后续断言）也算处理，例如多线程里
                # errors.append(exc) + 事后 assert not errors
                if "append" in d or isinstance(h.body[0] if h.body else None, ast.Assign):
                    continue
                tname = ast.unparse(h.type) if h.type else "bare"
                bad.append(f"{_rel(p)}:{h.lineno} {fn.name} (except {tname})")
        assert not bad, (
            "以下测试内的 except 既不断言也不失败（吞掉被测行为的错误）：\n  "
            + "\n  ".join(bad))


# ===========================================================================
# §B 安全门不要 fail-open  /  §J 财务关键参数不能是隐藏默认
# ===========================================================================

class TestLessonB_GatesMustNotFailOpen:

    #: 一旦门控失败/异常就必须能真正拦住的开关，及其"启用"值
    ENFORCEMENT_FLAGS = [
        "pm_strategy_gate_block",
        "risk_halt_on_drawdown",
        "tr_enforce_active_gate",
    ]

    def test_enforcement_flags_exist_and_are_documented(self):
        """门控开关必须存在于 config —— 防止改名后拦截逻辑变成永远读不到的死分支。"""
        from app.config import Settings
        missing = [f for f in self.ENFORCEMENT_FLAGS if f not in Settings.model_fields]
        assert not missing, f"门控开关在 config 中不存在（拦截逻辑已成死分支）：{missing}"

    def test_gate_failure_paths_are_not_bare_except_pass(self):
        """
        门控/风控相关模块里，`except Exception: pass`（连日志都没有）意味着
        门算失败也照常放行且无人知晓。至少要有 logger 或重新抛出。
        """
        # 关键路径：门控 / 风控 / 实盘账本 / 数据源。这些地方的静默 except 会让
        # "门失败"变成"门通过"、"数据缺失"变成"数据正常"，且没有任何痕迹。
        watch = [
            APP / "core" / "portfolio_manager" / "strategy_builder.py",
            APP / "core" / "portfolio_manager" / "risk_gate.py",
            APP / "core" / "portfolio_manager" / "strategy_gate.py",
            APP / "core" / "portfolio_manager" / "manager.py",
            APP / "core" / "portfolio_manager" / "horizon.py",
            APP / "core" / "lifecycle" / "promotion_gate.py",
            APP / "core" / "lifecycle" / "validation_gate.py",
            APP / "core" / "trading_context" / "providers.py",
            APP / "core" / "discovery" / "discovery_engine.py",
            APP / "core" / "monitor" / "alpha_monitor.py",
            APP / "core" / "backtest_engine" / "alpha_combiner.py",
            APP / "core" / "backtest_engine" / "portfolio_constructor.py",
            APP / "tasks" / "daily_trading_loop.py",
        ]
        bad = []
        for p in watch:
            if not p.exists():
                continue
            tree = ast.parse(_src(p))
            for h in [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]:
                dumped = ast.dump(h)
                silent = ("logger" not in dumped and "logging" not in dumped
                          and "Raise" not in dumped and "warn" not in dumped.lower())
                if silent:
                    bad.append(f"{_rel(p)}:{h.lineno}")
        assert not bad, (
            "以下门控异常处理既不记录也不抛出（门失败后静默继续）：\n  " + "\n  ".join(bad)
        )


    #: 全库"既不记录也不抛出"的 except 计数上限（棘轮：只许降，不许升）。
    #: 起点 131 → 本轮降到当前值。剩余多为滚动窗口数值兜底、DB bootstrap、
    #: 可选依赖探测等**有理由的**兜底；新增一处就会顶破这个数字。
    SILENT_EXCEPT_BUDGET = 97

    def test_silent_except_count_does_not_grow(self):
        import ast as _ast
        n, unparsable = 0, []
        for p in _py_files(APP):
            try:
                tree = _ast.parse(_src(p))
            except Exception as exc:
                # 讽刺但真实：这个棘轮此前自己用 except:continue 吞掉解析失败 ——
                # 一个语法坏掉的文件就能让它少数一批，等于给计数开后门。
                unparsable.append(f"{_rel(p)}: {exc}")
                continue
            for h in [x for x in _ast.walk(tree) if isinstance(x, _ast.ExceptHandler)]:
                d = _ast.dump(h)
                if (all(k not in d for k in ("logger", "logging", "Raise", "print"))
                        and "warn" not in d.lower()):
                    n += 1
        assert not unparsable, (
            "以下 app/ 文件无法解析，未被计入静默 except 统计（棘轮会因此虚低）：\n  "
            + "\n  ".join(unparsable))
        assert n <= self.SILENT_EXCEPT_BUDGET, (
            f"静默 except 增加到 {n} 处（上限 {self.SILENT_EXCEPT_BUDGET}）。"
            f"新增的兜底必须记录或抛出 —— 否则失败会伪装成成功。"
            f"若确属良性（滚动窗口数值兜底等），下调 SILENT_EXCEPT_BUDGET 并在此说明理由。"
        )
        if n < self.SILENT_EXCEPT_BUDGET:
            pytest.fail(
                f"静默 except 已降到 {n} 处，低于记录的上限 {self.SILENT_EXCEPT_BUDGET}。"
                f"请把 SILENT_EXCEPT_BUDGET 改成 {n}（棘轮只降不升，防止回潮）。"
            )


# ===========================================================================
# §F/§I 合成数据 & held-out 纪律  —— 结果必须自报数据来源
# ===========================================================================

class TestLessonF_ProvenanceIsMandatory:

    def test_agent_refuses_synthetic_without_optin(self):
        from app.agent._tools import QuantTools
        with pytest.raises(RuntimeError):
            QuantTools(n_tickers=5, n_days=60)

    def test_every_dataset_loading_path_is_fail_closed(self):
        """
        真实数据加载失败时**不得**静默回退到会话数据/合成数据。
        实测漏网：_tools.tool_run_gp_optimization 的 per-run 覆盖仍在静默降级，
        且降级后 data_source 不变 —— 前端徽章会继续显示"真实数据"。
        """
        src = _src(APP / "agent" / "_tools.py")
        tree = ast.parse(src)
        offenders = []
        for h in [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]:
            seg = ast.get_source_segment(src, h) or ""
            # 该 except 块里把数据换成了会话/合成数据，却没有同步更新来源标识
            swaps = re.search(r"=\s*self\._(is|oos)_data|_make_synthetic_dataset", seg)
            if swaps and "_data_source" not in seg and "raise" not in seg:
                offenders.append(f"app/agent/_tools.py:{h.lineno}")
        assert not offenders, (
            "以下位置在数据加载失败后静默降级，且未更新 _data_source（用户看到的来源标识会说谎）：\n  "
            + "\n  ".join(offenders)
        )

    def test_result_carrying_endpoints_expose_data_source(self):
        """凡是返回绩效指标的响应模型，都必须带 data_source。"""
        from app.api.chat_router import ChatResponse
        assert "data_source" in ChatResponse.model_fields

    def test_stream_done_event_carries_data_source(self):
        """前端消费的是 SSE，不是 POST —— 两条路径都要带（§R 的具体形态）。"""
        from app.agent.quant_agent import QuantAgent
        agent = QuantAgent(n_tickers=20, n_days=252, n_trials=1,
                           dataset_name="", allow_synthetic=True)
        events: list[dict] = []
        agent.stream_chat("rank(close)", session_id="lesson_f",
                          on_event=lambda e: events.append(e))
        done = [e for e in events if e.get("type") == "done"]
        assert done, f"未发出 done 事件：{[e.get('type') for e in events]}"
        assert done[-1]["result"].get("data_source") == agent.data_source


# ===========================================================================
# §E 硬编码 universe 会腐坏  /  §11 提示词里的标识符必须真实存在
# ===========================================================================

class TestLessonE_IdentifiersMustExist:

    def test_prompt_dataset_names_exist_in_registry(self):
        """
        LLM 提示词/docstring 里举例的数据集名必须在注册表中真实存在，
        否则 LLM 照着调用 → 加载失败 → 走降级路径。
        实测漏网：_lc_agent 举例 us_sectors / cn_ashares，注册表里都没有。
        """
        from app.core.data_engine.dataset_registry import _SPECS
        known = set(_SPECS)
        bad = []
        # 只扫 **LLM 能读到的**提示词/工具 docstring。app 内其他模块（如
        # multi_dataset）有自己的逻辑命名空间（us_equity 等），不属于注册表键。
        prompt_surfaces = [APP / "agent" / "_lc_agent.py",
                           APP / "agent" / "_tools.py",
                           APP / "agent" / "_agent.py",
                           APP / "api" / "router.py",
                           APP / "api" / "chat_router.py"]
        for p in [f for f in prompt_surfaces if f.exists()]:
            src = _src(p)
            for m in re.finditer(r'["\']((?:us|cn|hk|crypto|eu|jp)_[a-z0-9_]{3,})["\']', src):
                name = m.group(1)
                if name in known:
                    continue
                line = src[:m.start()].count("\n") + 1
                # 只看提示词/文档语境（docstring 或注释），不误伤内部标识符
                ctx = src[max(0, m.start() - 200):m.start()]
                if any(k in ctx for k in ('"""', "'''", "e.g.", "例如", "dataset_name:")):
                    bad.append(f"{_rel(p)}:{line} → {name!r}")
        assert not bad, (
            "提示词/文档中出现了注册表里不存在的数据集名（LLM 会照着调用并失败）：\n  "
            + "\n  ".join(bad) + f"\n  注册表实有：{sorted(known)}"
        )


# ===========================================================================
# §K 完成 ≠ 接进主线 —— 模块级 + **参数级**可达性
# ===========================================================================

_ENTRY_POINTS = {"app.main", "app.api.router", "app.api.chat_router",
                 "app.tasks.scheduler", "app.config"}


def _module_graph():
    mods, is_pkg, edges = {}, {}, {}
    for p in _py_files(APP):
        rel = p.relative_to(APP.parent).with_suffix("")
        name = ".".join(rel.parts)
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
            is_pkg[name] = True
        mods[name] = p
    for name, p in mods.items():
        refs = set()
        for node in ast.walk(ast.parse(_src(p))):
            if isinstance(node, ast.Import):
                refs |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = name if is_pkg.get(name) else name.rsplit(".", 1)[0]
                    for _ in range(node.level - 1):
                        base = base.rsplit(".", 1)[0]
                    base = f"{base}.{node.module}" if node.module else base
                else:
                    base = node.module or ""
                if base.startswith("app."):
                    refs.add(base)
                    refs |= {f"{base}.{a.name}" for a in node.names}
        edges[name] = {r for r in refs if r in mods} | {
            r.rsplit(".", 1)[0] for r in refs if r.rsplit(".", 1)[0] in mods}
    return mods, is_pkg, edges


class TestLessonK_WrittenMeansWired:

    def test_no_orphan_modules(self):
        allow = {"app.core.backtest_engine.overfit_stats"}
        mods, is_pkg, edges = _module_graph()
        seen, stack = set(), [m for m in _ENTRY_POINTS if m in mods]
        while stack:
            m = stack.pop()
            if m in seen:
                continue
            seen.add(m)
            stack.extend(edges.get(m, ()) - seen)
        orphans = sorted(m for m in set(mods) - seen if not is_pkg.get(m))
        orphans = [m for m in orphans if m not in allow and not m.startswith("app.core.utils")]
        assert not orphans, "以下模块从入口不可达（写了没接线）：\n  " + "\n  ".join(orphans)

    #: (模块, 函数, 形参) —— 影响资金/风险的关键形参，必须有真实调用点传值。
    #  §K 上一版只查到模块粒度，于是 port_vol_ann 这类"参数级孤儿"从下面走过去了。
    CRITICAL_PARAMS = [
        ("app.core.portfolio_manager.risk_gate", "PortfolioRiskGate.apply", "port_vol_ann"),
    ]

    @pytest.mark.parametrize("mod,qual,param", CRITICAL_PARAMS)
    def test_critical_parameters_are_actually_passed(self, mod, qual, param):
        """
        声明了但全库没有任何调用点传值的关键形参 = 该配置在实线路径上不生效。
        实测：risk_target_vol_ann 配了也没用，因为 apply() 从没收到过 port_vol_ann。
        """
        definer = mod.replace(".", "/") + ".py"
        callers = []
        for p in _py_files(APP):
            src = _src(p)
            if _rel(p).endswith(definer):
                continue
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, ast.Call):
                    if any(k.arg == param for k in node.keywords if k.arg):
                        callers.append(f"{_rel(p)}:{node.lineno}")
        assert callers, (
            f"{qual} 的形参 `{param}` 在 app/ 下**没有任何调用点传值** —— "
            f"依赖它的配置在实线路径上永远不生效（写了等于没写）。"
        )


# ===========================================================================
# §O 合成 H/L 会让价差估计荒谬  →  测试夹具的 H/L 必须有现实幅度
# ===========================================================================

class TestLessonO_FixtureRealism:

    def test_no_fixture_uses_fixed_one_percent_high_low(self):
        """
        `high = close * 1.01` / `low = close * 0.99` 这种固定 ±1% 的日内区间
        会让 Corwin-Schultz 价差估到 54bps（真实约 8.5bps），足以把有真实 alpha
        的因子判死（曾把 OOS +2.06 打成 −2.16）。夹具必须用随机幅度。
        """
        bad = []
        for p in _py_files(TESTS):
            if p.name == Path(__file__).name:
                continue
            src = _src(p)
            for m in re.finditer(r"(high|low)\s*=\s*close\s*\*\s*(0\.99|1\.01)\b", src):
                bad.append(f"{_rel(p)}:{src[:m.start()].count(chr(10)) + 1}  {m.group(0)}")
        assert not bad, (
            "以下夹具用固定 ±1% 构造 H/L —— 价差估计会荒谬到改变结论：\n  "
            + "\n  ".join(bad)
        )


# ===========================================================================
# §H 部署 .env 不得泄漏进测试 —— 但反过来：也不得让测试永远看不到生产默认值
# ===========================================================================

class TestLessonH_TestConfigIsolation:

    def test_conftest_overrides_are_declared_and_justified(self):
        """
        conftest 覆盖 settings 是必要的（别打网络/别起调度器），但每一处覆盖都
        制造了一个"测试永远看不到生产默认值"的盲区。要求：每个覆盖点必须有注释说明。
        实测：正是这些覆盖让门控默认值（tr_experiment_mode 等）从未在测试下被验证过。
        """
        src = _src(TESTS / "conftest.py")
        lines = src.splitlines()
        undocumented = []
        for i, ln in enumerate(lines):
            m = re.match(r"\s*settings\.(\w+)\s*=", ln)
            if not m:
                continue
            ctx = "\n".join(lines[max(0, i - 6):i])
            if "#" not in ctx:
                undocumented.append(f"conftest.py:{i + 1} settings.{m.group(1)}")
        assert not undocumented, (
            "以下 settings 覆盖没有注释说明理由（每个覆盖 = 一个测试盲区）：\n  "
            + "\n  ".join(undocumented)
        )

    def test_production_gate_defaults_are_covered_by_a_test(self):
        """
        §H 的反面：必须存在**至少一个**在生产默认配置下验证门控行为的测试。
        否则"函数写对了但默认值没打开"这类 bug 结构性不可见。
        """
        marker = "production_defaults"
        found = [_rel(p) for p in _py_files(TESTS)
                 if p.name != Path(__file__).name and marker in _src(p)]
        assert found, (
            "没有任何测试在**生产默认配置**下验证门控行为。\n"
            "conftest 覆盖了 settings，于是所有测试都跑在非发布配置下 —— "
            "门控默认值是否真的会拦截，从未被验证过。\n"
            f"请新增标记 {marker!r} 的用例（见 tests/test_production_defaults.py）。"
        )


# ===========================================================================
# §Q 活的 SQLite 不得放在云同步目录
# ===========================================================================

class TestLessonB2_DataQualityGateIsConsistent:
    """
    §B 的具体形态（审计 #5）：ingest 路径**真拒**低质量数据，研究/API 路径却写死
    `warn_only=True` 只记日志 —— 同一份烂数据，走哪条路结果完全不同。
    更糟的是 `warn_only=False` 抛的 ValueError 被同函数的 `except Exception` 吞掉，
    fail-closed 开关**从来没生效过**（实测 0.665 < 0.99 时返回 None 而非抛错）。
    """

    def _broken_dataset(self):
        import numpy as np, pandas as pd
        from app.core.data_engine.dataset_registry import Dataset
        idx = pd.bdate_range("2022-01-03", periods=120)
        close = pd.DataFrame(100.0, index=idx, columns=["A", "B", "C"])
        close.iloc[30:70] = np.nan          # 大段断档
        close.iloc[80, 0] = 1e6             # 尖刺
        data = {f: close.copy() for f in
                ("close", "open", "high", "low", "volume", "vwap", "returns")}
        return Dataset(name="broken", frequency="daily",
                       universe=["A", "B", "C"], data=data)

    def test_fail_closed_actually_raises(self):
        """warn_only=False 必须真的抛错，而不是被自己的 except 吞掉。"""
        from app.core.data_engine.dataset_registry import (
            check_dataset_health, DatasetHealthError)
        with pytest.raises(DatasetHealthError):
            check_dataset_health(self._broken_dataset(), min_score=0.99, warn_only=False)

    def test_warn_only_still_returns_report(self):
        """warn_only=True 时不抛错，但必须**返回报告**（否则调用方无从判断质量）。"""
        from app.core.data_engine.dataset_registry import check_dataset_health
        rep = check_dataset_health(self._broken_dataset(), min_score=0.99, warn_only=True)
        assert rep is not None and rep.overall_score < 0.99, rep

    def test_research_path_gate_is_configurable_and_defaults_closed(self):
        """研究路径必须与 ingest 同口径：默认 fail-closed，阈值可配。"""
        from app.config import Settings
        for f in ("research_health_fail_closed", "research_min_health"):
            assert f in Settings.model_fields, f"缺配置项 {f}"
        assert Settings.model_fields["research_health_fail_closed"].default is True, (
            "研究路径默认不拦低质量数据 —— 与 ingest 路径口径不一致"
        )


class TestLessonC_ImportedDepsMustBeDeclared:
    """
    §C 的具体形态（外部审计 #6）：`market_calendar.py` 需要 pandas_market_calendars，
    但 requirements.txt 从未声明它。缺失时**静默退回** pd.bdate_range —— 节假日被
    当成交易日、拿不到 DST/半日市收盘时间，且没有任何报错。开发机恰好装了，
    换台机器就悄悄错。
    """

    #: 会**静默降级**的第三方依赖（不是硬 import，缺了不报错）→ 必须显式声明
    SILENT_FALLBACK_DEPS = {
        "pandas_market_calendars": "app/core/data_engine/market_calendar.py",
    }

    def test_silently_optional_deps_are_declared_in_requirements(self):
        req = (BACKEND / "requirements.txt")
        assert req.exists(), "requirements.txt 不存在"
        declared = _src(req).replace("-", "_").lower()
        missing = [
            f"{mod}（{where}）" for mod, where in self.SILENT_FALLBACK_DEPS.items()
            if mod.replace("-", "_").lower() not in declared
        ]
        assert not missing, (
            "以下依赖会在缺失时**静默降级**（不报错、结果悄悄变错），"
            "却未在 requirements.txt 中声明：\n  " + "\n  ".join(missing)
        )

    def test_calendar_is_actually_using_the_real_exchange_calendar(self):
        """
        不只查声明，还要查**运行时真的用上了** —— 独立日（7/4）必须不是交易日。
        退回 pd.bdate_range 时它是周中工作日，会被判为交易日。
        """
        from app.core.data_engine.market_calendar import is_trading_day
        assert is_trading_day("2024-07-03") is True, "7/3 应为交易日"
        assert is_trading_day("2024-07-04") is False, (
            "独立日被判为交易日 —— 日历库未生效，已退回工作日启发式"
        )


class TestLessonQ_LiveDbNotInCloudSync:

    def test_default_db_path_is_not_in_cloud_sync_dir(self):
        from app.config import settings
        for attr in ("database_url", "pit_store_dir"):
            val = str(getattr(settings, attr, "") or "")
            low = val.replace("\\", "/").lower()
            assert not any(k in low for k in ("onedrive", "dropbox", "google drive", "icloud")), (
                f"settings.{attr} 指向云同步目录（活库会被同步进程撕裂）：{val}"
            )


# ===========================================================================
# §R 验证不得与实现同源 —— 每条教训都必须有强制检查
# ===========================================================================

class TestLessonS_AuditUnitIsNotTheModule:
    """
    §S：审计的检索单位错了 —— 真问题活在比模块细（参数/默认值/字符串）或
    比模块粗（横切属性）的层级。本类固化"横切属性"这一维度的检查。
    """

    # ── 债务台账 ────────────────────────────────────────────────────────────
    # 本轮审计**已查出**的问题，逐条记录在案。规则：
    #   1) 只许缩短，不许加长（新增即视为回归，由 *_no_new_* 用例把关）；
    #   2) 台账非空这件事本身另有一条 xfail(strict) 红线盯着，修完必须删行；
    #   3) 每条必须写明**后果**，不许只写位置。
    #: ✅ 全部已补覆盖（test_api_uncovered_routes.py + test_phase_pm7_endpoints.py）。
    #: 台账清空后，test_debt_ledger_is_empty 的 xfail(strict) 会转为 XPASS 报错，
    #: 提醒把该 xfail 标记一并删除 —— 这正是台账机制的设计意图。
    KNOWN_UNTESTED_ROUTES: dict = {}
    UNCOVERED_BY_DESIGN = {
        "/api/datasets/{name}/refresh":   "需真实网络拉数，离线测试不覆盖",
    }
    #: 写死合成数据且无 dataset_name 入参的端点（外部审计只报了 realistic 一个，
    #: 属性级扫描发现是 4 个）。后果：返回的 Sharpe/风险报告全是随机游走。
    #: ✅ B-11 已全部修完：realistic / simulate / optimize 加了 dataset_name + data_source；
    #: backtest_multi 经复核**本就不是**"写死合成"（有显式 use_synthetic 开关，默认 False），
    #: 是上一版检查器只认 dataset_name 这一个字段名造成的误报 —— 已修正判据。
    KNOWN_SYNTHETIC_ONLY_ENDPOINTS: set = set()

    def _untested_routes(self):
        from app.main import app
        routes = {
            r.path for r in app.routes
            if getattr(r, "methods", None) and str(r.path).startswith("/api")
        }
        test_src = "\n".join(_src(p) for p in _py_files(TESTS)
                             if p.name != Path(__file__).name)
        out = []
        for path in sorted(routes):
            if path in self.UNCOVERED_BY_DESIGN:
                continue
            probe = path.split("{")[0].rstrip("/")
            if probe and probe not in test_src:
                out.append(path)
        return out

    def _synthetic_only_endpoints(self):
        router_src = _src(APP / "api" / "router.py")
        out = []
        for fn in [n for n in ast.walk(ast.parse(router_src)) if isinstance(n, ast.FunctionDef)]:
            body_src = ast.get_source_segment(router_src, fn) or ""
            if "_make_synthetic_dataset(" not in body_src:
                continue
            decs = " ".join(ast.unparse(d) for d in fn.decorator_list)
            if "router." not in decs:
                continue
            # 判据是"**有没有显式的真实/合成选择开关**"，不是某个特定字段名。
            # `dataset_name`（置空=合成）与 `use_synthetic`（True=合成）都是合法的
            # 显式 opt-in 形态；只认前者会把 backtest_multi 误判成"写死合成"。
            if not any(k in body_src for k in ("dataset_name", "use_synthetic", "req.datasets")):
                out.append(fn.name)
        return sorted(out)

    def test_no_new_untested_api_route(self):
        """
        入口清单当一等资产：新增路由必须同时有测试，或显式进台账并写明后果。
        "留白正是出错的地方" —— /api/backtest/walk_forward 正是从没被真正跑通过。
        """
        new = [r for r in self._untested_routes() if r not in self.KNOWN_UNTESTED_ROUTES]
        assert not new, (
            "以下 API 路由没有任何测试触达，且不在债务台账中：\n  " + "\n  ".join(new)
        )

    def test_no_new_synthetic_only_endpoint(self):
        """比模块细的一层：端点写死合成数据且无 dataset_name 入参。"""
        new = [f for f in self._synthetic_only_endpoints()
               if f not in self.KNOWN_SYNTHETIC_ONLY_ENDPOINTS]
        assert not new, (
            "以下端点**写死合成数据**且无 dataset_name 入参（返回指标全是随机游走）：\n  "
            + "\n  ".join(new)
        )

    def test_debt_ledger_is_empty(self):
        """
        台账红线。曾以 xfail(strict) 标记"已知未修"；现台账已清空，
        改为**必须通过**的正向断言：任何新增的未覆盖路由或写死合成端点都会让它变红。
        （若将来又要挂账，重新加回 xfail(strict) 并在上面的 KNOWN_* 里写明后果。）
        """
        assert not self._untested_routes(), self._untested_routes()
        assert not self._synthetic_only_endpoints(), self._synthetic_only_endpoints()


class TestLessonT_OperatorArgsMustReachTheirConsumer:
    """
    §T：算子的参数从 AST 传到消费方，中间只要键名对不上就会**静默退化**。
    这类 bug 用"结果均值≈0"这种弱断言测不出（退化实现恰好满足），
    必须用**判别性**断言：带参数与不带参数的结果必须不同。
    """

    @staticmethod
    def _panel(n_tickers: int = 10, n_days: int = 40):
        import numpy as np, pandas as pd
        rng = np.random.default_rng(0)
        tk = [f"T{i:02d}" for i in range(n_tickers)]
        idx = pd.bdate_range("2022-01-03", periods=n_days)
        close = pd.DataFrame(
            100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_tickers)), axis=0),
            index=idx, columns=tk)
        return {"close": close, "open": close,
                "high": close * (1 + rng.uniform(0, 0.006, close.shape)),
                "low":  close * (1 - rng.uniform(0, 0.006, close.shape)),
                "volume": close * 1000, "vwap": close,
                "returns": close.pct_change().fillna(0.0)}, tk

    def test_group_argument_actually_changes_the_result(self):
        """给 sector 与不给 sector，行业中性算子的输出**必须不同**。"""
        import numpy as np
        from app.core.alpha_engine.parser import Parser
        from app.core.alpha_engine.dsl_executor import Executor
        base, tk = self._panel()
        sec = np.array([0.0] * (len(tk) // 2) + [1.0] * (len(tk) - len(tk) // 2))
        for expr in ("ind_neutralize(close, 'sector')", "sector_neutral(close)"):
            node = Parser().parse(expr)
            with_g = Executor().run(node, {**base, "sector": sec})
            without = Executor().run(node, base)
            assert not np.allclose(with_g.values, without.values, equal_nan=True), (
                f"{expr}：给分组与不给分组输出逐位相同 —— 分组参数未被消费，"
                f"该算子实际等价于全截面版本"
            )
            # 组内均值必须归零（真正的组内中性）
            last = with_g.iloc[-1]
            for g in (0.0, 1.0):
                cols = [t for t, v in zip(tk, sec) if v == g]
                m = float(last[cols].mean())
                assert abs(m) < 1e-8, f"{expr}：行业 {g} 组内均值 {m:.3e} 未归零"

    def test_missing_group_field_raises_instead_of_fabricating(self):
        """
        缺分组字段时**不得凭空编造**（旧实现用 np.arange(N) % 10 按 i%10 分组）。
        编造输入会把"没数据"变成"有数据但全错"，且从结果上看不出来。
        """
        import pytest as _pytest
        from app.core.alpha_engine.parser import Parser
        from app.core.alpha_engine.dsl_executor import Executor
        base, _ = self._panel()
        node = Parser().parse("group_rank(close, 'sector')")
        with _pytest.raises(Exception) as ei:
            Executor().run(node, base)          # 故意不提供 sector/groups
        assert "sector" in str(ei.value) or "groups" in str(ei.value), (
            f"缺分组时的报错没有点明缺哪个字段：{ei.value}"
        )


class TestLessonU_FallbacksMustLeanConservative:
    """
    §U：兜底必须朝**保守**一侧倒。失败时自动滑向"更容易通过/更容易下单"的一侧
    比崩溃危险得多 —— 崩溃会被发现，悄悄放宽看起来一切正常。
    """

    #: (模块, 危险兜底的源码片段, 说明) —— 出现即失败
    FORBIDDEN_FALLBACKS = [
        ("app/core/portfolio_manager/manager.py", "long_only = False",
         "读不到 trading_allow_short 时不得默认允许做空"),
        ("app/core/discovery/discovery_engine.py", 'mode = "leak"',
         "读不到 factor_gate_mode 时不得默认用松门（与其 docstring 的 fail-closed 相悖）"),
    ]

    @pytest.mark.parametrize("path,snippet,why", FORBIDDEN_FALLBACKS)
    def test_no_permissive_fallback_in_except_block(self, path, snippet, why):
        import ast as _ast
        src = _src(BACKEND / path)
        tree = _ast.parse(src)
        offenders = []
        for h in [n for n in _ast.walk(tree) if isinstance(n, _ast.ExceptHandler)]:
            seg = _ast.get_source_segment(src, h) or ""
            if snippet in seg:
                offenders.append(f"{path}:{h.lineno}")
        assert not offenders, f"{why}｜出现在：{offenders}"

    def test_positions_reader_refuses_to_fake_empty_book(self):
        """
        持仓读失败绝不能返回 {} —— 空仓 = "全部权益都是可用买入力"，
        下游会按满额重新建仓（等于凭空加杠杆）。必须抛错。
        """
        from app.core.trading_context.providers import SimAccountProvider

        class _Boom:
            class store:
                @staticmethod
                def latest_positions(_):
                    raise RuntimeError("db down")

        prov = SimAccountProvider.__new__(SimAccountProvider)
        prov._broker = _Boom()
        prov._book = 0
        with pytest.raises(RuntimeError):
            prov.positions()

    def test_business_exceptions_are_not_swallowed_by_generic_handler(self):
        """
        有意抛出的业务异常必须用**专用类型**并在通用兜底前放行。
        `check_dataset_health` 曾把自己 raise 的 ValueError 又 except 掉，
        导致 fail-closed 开关从未生效。
        """
        import numpy as np, pandas as pd
        from app.core.data_engine.dataset_registry import (
            check_dataset_health, DatasetHealthError, Dataset)
        idx = pd.bdate_range("2022-01-03", periods=120)
        close = pd.DataFrame(100.0, index=idx, columns=["A", "B", "C"])
        close.iloc[30:70] = np.nan
        data = {f: close.copy() for f in
                ("close", "open", "high", "low", "volume", "vwap", "returns")}
        ds = Dataset(name="broken", frequency="daily", universe=["A", "B", "C"], data=data)
        with pytest.raises(DatasetHealthError):
            check_dataset_health(ds, min_score=0.99, warn_only=False)


class TestLessonV_NoMutationResidueInRepo:
    """
    §V：变异测试**原地改文件**，一旦提交发生在变异运行期间，被改坏的源码会进仓库。

    这不是假想 —— 实际发生了：commit 24c251a 把
    `aum = float(aum) if aum is not None else ...` 变成了 `if aum is None`
    （正是变异算子"删掉 not"），同时把临时备份 `daily_trading_loop.py.mutbak`
    （562 行）也一并提交。变异测试跑完后 finally 把工作区还原成正确版本，
    于是**仓库里是坏的、工作区是好的**，`git diff` 才把它暴露出来。
    """

    def test_no_mutbak_files_in_tree(self):
        """变异测试的临时备份绝不能留在树里，更不能进仓库。"""
        stray = [_rel(p) for p in BACKEND.rglob("*.mutbak")]
        assert not stray, (
            "发现变异测试残留备份（应已被 .gitignore 且运行结束即删）：\n  "
            + "\n  ".join(stray))

    def test_mutbak_is_gitignored(self):
        gi = BACKEND.parent / ".gitignore"
        assert gi.exists(), ".gitignore 不存在"
        assert "*.mutbak" in _src(gi), (
            "*.mutbak 未加入 .gitignore —— 变异运行期间的提交会把改坏的备份带进仓库")

    def test_known_mutation_targets_are_in_original_form(self):
        """
        对**已知被变异污染过**的关键行做形态断言。
        这些行的"正确形态"本身就是契约：写反了不会报错，只会让钱算错或门失效。
        """
        checks = [
            ("app/tasks/daily_trading_loop.py",
             "aum = float(aum) if aum is not None else",
             "aum 为 None 时才回退 broker 资金；写成 `is None` 会 float(None) 崩溃且忽略传参"),
            ("app/tasks/daily_trading_loop.py",
             "long_only=(not getattr(settings",
             "long_only 必须是 allow_short 的取反"),
            ("app/tasks/daily_trading_loop.py",
             "weights = weights * 0.0",
             "熔断清仓是乘 0，写成除 0 会产生 inf/NaN 权重"),
            ("app/tasks/daily_trading_loop.py",
             "prices_prev=prices_f.iloc[t - 1] if t > 0 else",
             "第 0 天必须取当日自身；`t >= 0` 会取末行价 → 前视"),
            ("app/core/execution/paper_broker.py",
             "pv = equity * self.initial_capital",
             "组合市值是乘法；写成除法会差 12 个数量级"),
        ]
        bad = []
        for path, snippet, why in checks:
            if snippet not in _src(BACKEND / path):
                bad.append(f"{path}: 缺少 {snippet!r} —— {why}")
        assert not bad, "关键行形态不符（疑似变异残留或被误改）：\n  " + "\n  ".join(bad)


class TestLessonR_EveryLessonIsEnforced:

    def test_every_lesson_has_an_enforced_check(self):
        """
        DEV_LESSONS 每条（## X. ...）都必须在本文件里有对应检查，
        或在 EXEMPT 中写明为什么无法自动化。散文形式的教训复发率≈100%。
        """
        EXEMPT = {
            "C": "相对导入深度——已由 test_no_orphan_modules 的导入图覆盖",
            "D": "pandas 面板坑——分散在各算子单测中，无单一可查形态",
            "G": "APScheduler 时区——需真实调度器运行，见 test_phase5 调度用例",
            "L": "删死代码方法论——是操作流程，非代码属性",
            "M": "换数据源收尾——前端 tsc 门在 CI 侧，不在 pytest 内",
            "N": "可视化绑真数据——见 test_phase_visualizer.py",
            "P": "性能回归——见 tests/performance/",
        }
        lessons = re.findall(r"^## ([A-Z])\.", _src(LESSONS), flags=re.M)
        assert lessons, "DEV_LESSONS.md 里没解析到任何教训标题"
        src = _src(Path(__file__))
        missing = []
        for L in lessons:
            if L in EXEMPT:
                continue
            if not re.search(rf"§{L}\b|Lesson{L}_", src):
                missing.append(L)
        assert not missing, (
            f"以下教训没有对应的强制检查，也没在 EXEMPT 中说明：{missing}\n"
            f"教训不落成检查就等于没写（这正是 §R 本身）。"
        )
