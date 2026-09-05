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
        watch = [
            APP / "core" / "portfolio_manager" / "strategy_builder.py",
            APP / "core" / "portfolio_manager" / "risk_gate.py",
            APP / "core" / "lifecycle" / "promotion_gate.py",
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
    KNOWN_UNTESTED_ROUTES = {
        "/api/agent/run":                 "自主 agent 入口零覆盖：跑通与否无人知晓",
        "/api/backtest/multi":            "多数据集回测零覆盖，且写死合成数据（见下）",
        "/api/paper/{alpha_id}/pnl":      "paper 账本 PnL 读取零覆盖",
        "/api/strategies/propose":        "策略提案入口零覆盖 —— 这是进审批队列的源头",
        "/api/workflow/generate/stream":  "SSE 版本零覆盖（非流式版有覆盖，§R 同型风险）",
        "/api/workflow/optimize/stream":  "同上",
    }
    UNCOVERED_BY_DESIGN = {
        "/api/datasets/{name}/refresh":   "需真实网络拉数，离线测试不覆盖",
    }
    #: 写死合成数据且无 dataset_name 入参的端点（外部审计只报了 realistic 一个，
    #: 属性级扫描发现是 4 个）。后果：返回的 Sharpe/风险报告全是随机游走。
    KNOWN_SYNTHETIC_ONLY_ENDPOINTS = {
        "backtest_realistic", "backtest_multi", "alpha_simulate", "alpha_optimize",
    }

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
            if "dataset_name" not in body_src:
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

    @pytest.mark.xfail(strict=True, reason=(
        "已知未修：6 条路由零覆盖 + 4 个端点写死合成数据。"
        "修完后请删除对应台账行；本用例转绿即提示台账已过期。"
    ))
    def test_debt_ledger_is_empty(self):
        """
        台账红线：本用例**必须**保持 xfail。一旦有人修完问题却忘了删台账行，
        它会变成 XPASS(strict) → 失败，强制台账与现实同步。
        """
        assert not self._untested_routes(), self._untested_routes()
        assert not self._synthetic_only_endpoints(), self._synthetic_only_endpoints()


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
