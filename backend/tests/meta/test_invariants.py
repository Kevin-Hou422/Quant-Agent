"""
test_invariants.py — 系统级**不变量**（不是用例）

背景（DEV_LESSONS §R）：同一会话内"只测局部"反复出现 4 次，根因是**验证用例由与实现
同一个心智模型生成**，因此结构性看不见同一盲区。用例抓实例，**不变量抓类别**。
本文件把外部审计打中的那几类问题固化成"再犯就红"的断言。
"""

from __future__ import annotations

import ast
import collections
import json
import re
import inspect
from pathlib import Path

import pytest

def _backend_root() -> Path:
    """
    向上找到含 `app/` 的目录 = backend/。

    **不要写成 `Path(__file__).resolve().parents[N]`**：层数一旦随目录重组
    变化，这里会静默指到错误的目录，`rglob("*.py")` 扫出空集合，
    而"对空集合的全称断言恒真" —— 约束静默失效且没有任何报错。
    """
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "app").is_dir():
            return parent
    raise RuntimeError(f"从 {p} 向上找不到含 app/ 的 backend 根目录")


BACKEND = _backend_root()
APP = BACKEND / "app"


# ---------------------------------------------------------------------------
# 不变量 1：生产路径不得**静默**回退到合成数据
# ---------------------------------------------------------------------------

def test_agent_refuses_synthetic_without_explicit_optin():
    """QuantTools 未指定数据集且未显式允许合成 → 必须抛错，不得默默给随机数。"""
    from app.agent._tools import QuantTools
    with pytest.raises(RuntimeError, match="合成"):
        QuantTools(n_tickers=8, n_days=60)          # 无 dataset_name、无 allow_synthetic


def test_agent_fails_closed_on_bad_dataset():
    """指定了数据集但加载失败 → 必须抛错（fail-closed），不得回退合成。"""
    from app.agent._tools import QuantTools
    with pytest.raises(RuntimeError, match="加载失败|拒绝回退"):
        QuantTools(n_tickers=8, n_days=60, dataset_name="__does_not_exist__")


def test_synthetic_requires_explicit_flag_and_is_labeled():
    """显式允许时可用合成，但**必须被标注**，供上层/前端如实展示。"""
    from app.agent._tools import QuantTools
    t = QuantTools(n_tickers=8, n_days=80, allow_synthetic=True)
    assert t.is_synthetic and t.data_source.startswith("synthetic")


def test_quant_agent_exposes_data_source():
    """聊天路径必须能报出数据来源（此前整条链路连 dataset_name 参数都没有）。"""
    from app.agent.quant_agent import QuantAgent
    sig = inspect.signature(QuantAgent.__init__)
    for p in ("dataset_name", "allow_synthetic"):
        assert p in sig.parameters, f"QuantAgent 缺少数据契约参数 {p}"
    assert "data_source" in dir(QuantAgent)


# ---------------------------------------------------------------------------
# 不变量 2：API 入口的数据契约一致（不得一个默认真实、一个默认合成）
# ---------------------------------------------------------------------------

def test_api_request_models_default_to_real_dataset():
    from app.api.router import GPEvolveRequest, BacktestRequest
    for model in (GPEvolveRequest, BacktestRequest):
        default = model.model_fields["dataset_name"].default
        assert default, f"{model.__name__}.dataset_name 默认为空 → 会静默跑合成数据"


# `test_chat_response_carries_data_source` 与
# `test_stream_done_event_carries_data_source` 原本在这里各有一份，
# 与 `meta/test_lessons_enforced.py::TestLessonF` 里的两条**逐字相同**
# （AST 指纹比对确认）。同一条约束由两个 meta 套件各查一次没有增量，
# 已删除本文件里的副本，保留 lessons_enforced 那一份 ——
# 那里是"教训固化"的正式归属地，且与同组的其他 §R 检查放在一起。


# ---------------------------------------------------------------------------
# 不变量 3：有价值的模块必须从运行入口可达（防"写了但没接线"）
# ---------------------------------------------------------------------------

_ENTRY_POINTS = {"app.main", "app.api.router", "app.api.chat_router",
                 "app.tasks.scheduler", "app.config"}


def _module_graph():
    mods, is_pkg, edges = {}, {}, {}
    for f in APP.rglob("*.py"):
        if "__pycache__" in str(f):
            continue
        rel = f.relative_to(BACKEND).as_posix()[:-3].replace("/", ".")
        pkg = rel.endswith(".__init__")
        if pkg:
            rel = rel[:-9]
        mods[rel] = f
        is_pkg[rel] = pkg
    for name, path in mods.items():
        refs = set()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                refs |= {a.name for a in node.names if a.name.startswith("app.")}
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    own = name if is_pkg.get(name) else name.rsplit(".", 1)[0]
                    parts = own.split(".")
                    base = ".".join(parts[: len(parts) - (node.level - 1)]
                                    + ([node.module] if node.module else []))
                if base.startswith("app."):
                    refs.add(base)
                    refs |= {f"{base}.{a.name}" for a in node.names}
        edges[name] = {r for r in refs if r in mods} | {
            r.rsplit(".", 1)[0] for r in refs if r.rsplit(".", 1)[0] in mods}
    return mods, is_pkg, edges


def test_no_orphan_modules_outside_allowlist():
    """
    从入口 BFS，凡不可达的非 __init__ 模块都是"写了但没接线"——要么接线，要么删。
    允许清单只放**有意保留的离线/参考实现**，且必须写明理由。
    """
    allow = {
        # R.2/R.4 的方法参考实现（roadmap 明示未接 live 路径）
        "app.core.backtest_engine.overfit_stats",   # 已接 StrategyGate，保底豁免
    }
    mods, is_pkg, edges = _module_graph()
    seen, stack = set(), [m for m in _ENTRY_POINTS if m in mods]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        stack.extend(edges.get(m, ()) - seen)
    # 包 __init__ 的可达性由其子模块代表，不单独判定
    orphans = sorted(m for m in set(mods) - seen if not is_pkg.get(m))
    orphans = [m for m in orphans if m not in allow and not m.startswith("app.core.utils")]
    assert not orphans, (
        "以下模块从运行入口不可达（写了但没接线，等于没做）：\n  " + "\n  ".join(orphans))


# ===========================================================================
# 测试进程不得对进程外产生可见副作用
# ===========================================================================
#
# 事故（2026-09-14）：`visualizer.plot()` 的 `show: bool = False` 被变异成
# True 之后，`test_backtest_plot_fidelity.py` 里三十多条"画图并检查 trace"
# 的用例**每条都真的打开了一个浏览器标签**，一次性在使用者屏幕上弹出几十个。
#
# 当时只在专门测 show 开关的那两条用例里 monkeypatch 了 `Figure.show`。
# 教训：**变异测试会把代码跑在你没预期的配置下** —— 凡是能捅到进程外的
# 东西（弹窗、发信、打开文件关联程序），必须在 conftest 级别全局堵死。
#
# 下面两条守住那个总闸本身，免得它被人当成多余的 fixture 删掉。

class TestNoOutOfProcessSideEffects:

    def test_plotly_figure_show_is_globally_stubbed(self):
        """
        `tests/conftest.py::_never_open_a_browser` 必须已经把
        `plotly.graph_objects.Figure.show` 换成记账替身。

        它被删掉之后，任何一条走到 `show=True` 的用例（包括变异测试
        造出来的配置）都会真的弹浏览器。
        """
        plotly = pytest.importorskip("plotly.graph_objects")
        show = plotly.Figure.show
        name = getattr(show, "__name__", "")
        assert name == "_recording_show", (
            f"Figure.show 现在是 {name!r} —— conftest 里的 "
            f"_never_open_a_browser 总闸没生效或被删了，"
            f"测试有可能真的弹出浏览器窗口")

    def test_calling_show_records_instead_of_opening_a_browser(self,
                                                              figure_show_calls):
        """总闸的行为面验证：调用 show 只记账，不产生任何进程外动作。"""
        plotly = pytest.importorskip("plotly.graph_objects")
        fig = plotly.Figure()
        fig.show()
        assert len(figure_show_calls) == 1, "Figure.show 的调用没有被记录下来"

    def test_the_visualizer_show_flags_default_to_false(self):
        """
        源码侧的第二道保险：两个出图入口的 `show` 默认值都必须是 False。

        这条与 `test_backtest_plot_fidelity.py` 里的行为断言互补 ——
        那边验"默认不调用 show"，这边验"签名里写的就是 False"，
        任何一侧被改都会红。
        """
        import inspect

        vis = pytest.importorskip("app.core.backtest_engine.visualizer")
        for fn_name in ("plot", "plot_decile_bar"):
            fn = getattr(vis.BacktestVisualizer, fn_name)
            default = inspect.signature(fn).parameters["show"].default
            assert default is False, (
                f"BacktestVisualizer.{fn_name} 的 show 默认值是 {default!r}，"
                f"必须是 False —— 否则每次出图都会弹浏览器")


# ===========================================================================
# 每个有变异点的模块都必须被测量过
# ===========================================================================
#
# 来由（自伤教训 #9）：D 档收尾的全量对账发现 4 个模块从未作为测量目标跑过
# （`health_report` 32 点、`sector_mapper` 5、`_fallback` 4、`yahoo_provider` 3），
# 首测全部 **0.0%** —— 既有测试只 import 过它们。
#
# 根因不是"忘了"，是**我把为人眼截断过的打印输出当成了清单**：
# 清点脚本里写了 `sorted(todo, ...)[:40]` 和 `sorted(untested, ...)[:15]`，
# 而实际有 57 / 26 个。按点数降序排在末尾的小模块就这么掉出了清单。
#
# 这里把"对账"本身变成一条测试：`app/` 下任何有变异点却不在
# `measured_modules.json` 里的模块都会让它变红。清单是机器生成的全量产物，
# 不是打印出来的摘要。

_MEASURED_JSON = Path(__file__).with_name("measured_modules.json")


def _mutation_points(src: str) -> int:
    """
    调用变异工具自己的 `build_plan` 数变异点 —— 判据必须与测量时完全一致，
    否则对账会在"我以为的点数"和"工具认的点数"之间漂移。
    """
    import sys

    tools = _backend_root() / "tools" / "mutation"
    if not tools.is_dir():
        return -1                      # 工具不在（未追踪目录）→ 交由调用方跳过
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    try:
        import mutate
    except Exception:
        return -1
    return len(mutate.build_plan(src))


def _current_plan_points():
    """按**当前**变异器集合重算 app/ 的全量变异点；工具不在时返回 None。"""
    total = 0
    app = _backend_root() / "app"
    for p in sorted(app.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        n = _mutation_points(p.read_text(encoding="utf-8"))
        if n < 0:
            return None
        total += n
    return total


class TestEveryModuleIsMeasured:

    def _manifest(self) -> dict:
        assert _MEASURED_JSON.exists(), (
            f"{_MEASURED_JSON.name} 不存在 —— 变异测量的清单是交付物的一部分，"
            f"不能删。重新生成见 MUTATION_LEDGER.md §自伤教训 #9。")
        return json.loads(_MEASURED_JSON.read_text(encoding="utf-8"))

    def test_the_manifest_is_wellformed(self):
        d = self._manifest()
        assert d["modules"], "清单里一个模块都没有"
        for mod, rec in d["modules"].items():
            assert mod.startswith("app/") and mod.endswith(".py"), f"路径异常：{mod}"
            for k in ("points", "killed", "survived", "kill_rate"):
                assert k in rec, f"{mod} 缺字段 {k}"
            assert rec["killed"] + rec["survived"] == rec["points"], (
                f"{mod} 的 killed+survived 与 points 对不上：{rec}")

    def test_no_module_with_mutants_is_missing_from_the_manifest(self):
        """
        **这条就是防第 9 条教训复发的那道闸。**

        `app/` 下任何有变异点、却不在清单里的模块 = 从没量过测试强度。
        新增模块时它会红，提示先测量再合入。
        """
        d = self._manifest()
        recorded = set(d["modules"])
        app = _backend_root() / "app"

        missing = []
        skipped = False
        for p in sorted(app.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(_backend_root()).as_posix()
            n = _mutation_points(p.read_text(encoding="utf-8"))
            if n < 0:
                skipped = True
                break
            if n > 0 and rel not in recorded:
                missing.append(f"{rel}（{n} 个变异点）")

        if skipped:
            pytest.skip("tools/mutation 不在工作区（未追踪目录），无法按同一判据对账")

        # 补齐变异器（工具缺陷 #11）之后，有模块是**新进入范围**的 ——
        # 它们确实从未测量，但这件事已经在 measurement_scope 里明示并上了棘轮。
        # 允许它们存在，但**必须逐个列名**：不许靠"反正清单里没有"蒙混过去。
        declared = set(d.get("measurement_scope", {})
                       .get("newly_in_scope_modules", []))
        undeclared = [m for m in missing if m.split("（")[0] not in declared]
        assert not undeclared, (
            "以下模块有变异点却从未测量过，且没有登记在 "
            "measurement_scope.newly_in_scope_modules 里：\n  "
            + "\n  ".join(undeclared)
            + "\n修法：跑一次 tools/mutation/runner.py 并更新 "
              "tests/meta/measured_modules.json；\n"
              "暂时测不了就把它写进 newly_in_scope_modules 并说明原因 —— "
              "**不许什么都不做**。")

    def test_newly_in_scope_modules_are_really_still_unmeasured(self):
        """
        棘轮的另一侧：`newly_in_scope_modules` 是"欠测清单"，不是豁免名单。
        某个模块一旦真的测了（进了 modules），就必须从这里划掉，
        否则这份清单会慢慢退化成一句永远够用的免责声明。
        """
        d = self._manifest()
        declared = d.get("measurement_scope", {}).get("newly_in_scope_modules", [])
        already = [m for m in declared if m in d["modules"]]
        assert not already, (
            f"这些模块已经测量并进了清单，却还挂在『欠测』名单上：{already} —— "
            f"请从 measurement_scope.newly_in_scope_modules 删掉")

    def test_the_manifest_does_not_list_modules_that_no_longer_exist(self):
        """反向：模块被删/改名之后，清单里的僵尸条目要清掉，免得掩盖真实缺口。"""
        d = self._manifest()
        root = _backend_root()
        gone = [m for m in d["modules"] if not (root / m).exists()]
        assert not gone, f"清单里有已不存在的模块：{gone}"

    def test_the_totals_match_the_per_module_entries(self):
        d = self._manifest()
        mods = d["modules"].values()
        assert d["totals"]["modules_with_mutants"] == len(d["modules"])
        assert d["totals"]["total_points"] == sum(m["points"] for m in mods)
        assert d["totals"]["total_survived"] == sum(m["survived"] for m in mods)

    def test_the_unmeasured_scope_stays_visible_and_only_shrinks(self):
        """
        **已测量的点数是在旧变异器集合下枚举的**，补齐算子后同一份源码的点数更多，
        多出来的从未测量。这条把差额钉在台面上，并做成只许减不许增的棘轮。

        为什么必须有这条：外部审计 2026-09-15 指出那套算子有系统性缺口
        （`x >= .70` / `x / w` / `a+b` 各 0 个变异点）。补齐算子之后，
        如果只把新的总点数写进清单、不记差额，交付物看起来只会**更好**，
        而"这 863 个点从没跑过"这件事就消失了 —— 这正是本项目反复踩的那种坑。
        """
        d = self._manifest()
        scope = d.get("measurement_scope")
        assert scope, (
            "measured_modules.json 的 measurement_scope 块被删了 —— "
            "『测过的范围』与『现在该测的范围』不一致，这件事必须保持可见")
        for k in ("measured_points", "measured_under", "current_plan_points",
                  "current_plan_under", "unmeasured_points", "still_blind", "owed"):
            assert scope.get(k) is not None, f"measurement_scope 缺字段 {k}"

        measured = sum(m["points"] for m in d["modules"].values())
        assert scope["measured_points"] == measured, (
            f"measurement_scope.measured_points={scope['measured_points']} 与"
            f"逐模块之和 {measured} 对不上")

        planned = _current_plan_points()
        if planned is None:
            pytest.skip("tools/mutation 不在工作区，无法按同一判据重算计划点数")
        gap = planned - measured
        assert gap == scope["unmeasured_points"], (
            f"未测量点数实际是 {gap}，清单里写的是 {scope['unmeasured_points']} —— "
            f"改了变异器却没同步这个数字，差额会在下一份报告里消失")
        ceiling = d["proof_reconciliation"]["ratchet"]["unmeasured_points_max"]
        assert gap <= ceiling, (
            f"未测量的变异点从 {ceiling} 涨到 {gap} —— 只许通过重测来减少。"
            f"若确实新增了变异器，请连同『欠一次重测』一起写进 measurement_scope.owed。")

    def test_every_recorded_test_path_still_exists(self):
        """
        清单里记的**测试出处**必须指向真实存在的文件。

        外部审计 2026-09-15 抓到的：Task 1 把 `tests/test_X.py` 挪进了
        `tests/<层>/<包>/`，而本清单记的还是旧扁平路径 —— **104 个不重复路径
        全部不存在**。点数没错，错的是出处：照着清单跑不回"当初是哪些测试
        测的它"，等于交付物里最关键的可复核性没了。

        `TestEveryModuleIsMeasured` 之前只对账**模块**，不对账**出处**，
        所以目录重组对它是透明的。这条补上另一半。
        """
        d = self._manifest()
        root = _backend_root()
        missing = sorted({t for rec in d["modules"].values()
                          for t in rec.get("tests", [])
                          if not (root / t).exists()})
        assert not missing, (
            f"清单记录的测试出处有 {len(missing)} 个已不存在：\n  "
            + "\n  ".join(missing[:20])
            + "\n修法：文件只是挪了位置就重映射路径；真的没了就置空并在 "
              "_provenance_repair 里说明，不要留下指不到的路径。")

    def test_the_provenance_repair_record_stays_visible(self):
        """
        出处是**重映射**来的，不是重测来的 —— 这个区别必须留在交付物里。
        按文件名映射保证"路径现在能跑"，不保证"当初那一版内容与现在相同"。
        """
        d = self._manifest()
        rec = d.get("_provenance_repair")
        assert rec, "_provenance_repair 块被删了 —— 出处是修补来的，这件事不能消失"
        for k in ("method", "caveat", "dropped_unmappable",
                  "modules_with_no_surviving_provenance", "modules_list"):
            assert rec.get(k) is not None, f"_provenance_repair 缺字段 {k}"
        lost = rec["modules_list"]
        assert len(lost) == rec["modules_with_no_surviving_provenance"], (
            "出处已丢失的模块清单与它自己声明的条数对不上")
        assert all(m in d["modules"] for m in lost), (
            "出处丢失清单里有不在模块清单中的条目")

    def test_the_open_reconciliation_gap_stays_visible(self):
        """
        存活项的**模块级**归属尚未逐条对账 —— 这是本阶段已知未闭合的唯一缺口。

        把它写进清单并在这里断言，是为了它不会随时间被忘掉：
        删掉那个块、或让证明条数变少，这条都会红。
        （条数变少 = 有证明被删而对应的存活项并没有被杀死。）
        """
        d = self._manifest()
        rec = d.get("proof_reconciliation")
        assert rec, (
            "measured_modules.json 里的 proof_reconciliation 块被删了 —— "
            "已知缺口必须保持可见，不能靠记忆")
        for k in ("total_survivors", "proof_entries_in_suite", "why_not",
                  "how_to_close", "ratchet"):
            assert rec.get(k), f"proof_reconciliation 缺字段 {k}"

        n = _count_proof_entries()
        floor = rec["ratchet"]["proof_entries_min"]
        assert n >= floor, (
            f"全库等价性证明从 {floor} 条降到了 {n} 条 —— "
            f"要么有证明被删（而存活项并没被杀死），"
            f"要么该模块被真正收口了：后者请把 ratchet.proof_entries_min 调低并说明。")

    def test_every_proof_key_declares_its_module_and_point_count(self):
        """
        证明键的格式：`<模块路径> ×<覆盖点数> — <原描述>`。

        **为什么必须写进键里**：外部审计 2026-09-15 指出，旧键是自由文本
        （`"L124 (weights < -tol) → <="`），不带模块路径，于是"138 个存活
        逐条对上 97 条证明"这件事只能靠模糊匹配去猜。实测猜的结果有错
        （按 import 归属会把 risk_gate 的证明记到别的模块上），
        而**判据错了比没判据更危险**。

        `×N` 是这条证明覆盖几个变异点 —— 一条论证常常同时盖住好几处
        （`fast_ops` 有一条一口气盖了 10 个 shape/keepdims 点），
        不声明就没法做数量对账。已被杀死、只作为记录保留的条目写 `×0`。
        """
        bad = []
        for key in _collect_dict_entries("PROVEN_EQUIVALENT"):
            m = re.match(r"^(app/\S+\.py) ×(\d+) — \S", key)
            if not m:
                bad.append(f"格式不符：{key[:70]}")
            elif not (_backend_root() / m.group(1)).exists():
                bad.append(f"模块不存在：{m.group(1)}")
        assert not bad, (
            "以下等价性证明的键不合规（应为 `<模块路径> ×<点数> — <描述>`）：\n  "
            + "\n  ".join(bad))

    def test_survivor_disposition_reconciles_per_module(self):
        """
        **逐模块对账**：每个模块的存活数 vs 以该模块为前缀的证明所声明的点数。

        这就是台账里挂了很久的那个"已知未闭合缺口"。它现在闭合的方式**不是**
        "全都有证明了"，而是"**差多少、差在哪，逐模块写下来并锁住**"：

          - 声明数 > 存活数 → 直接判红（证明盖住了并不存在的存活点）
          - 声明数 < 存活数 → 差额必须与清单里登记的**逐模块**数字一致
          - 差额总数只许减不许增

        当前真实状态：136 个存活，114 个有书面证明，**22 个既未被杀死也无证明**。
        这 22 个此前被"每个文件条目数 == 自己声明的存活数"这条自洽检查掩盖了 ——
        文件内自洽，跨文件汇总却对不上。
        """
        d = self._manifest()
        surv = {m: r["survived"] for m, r in d["modules"].items() if r["survived"]}
        declared = collections.Counter()
        for key in _collect_dict_entries("PROVEN_EQUIVALENT"):
            m = re.match(r"^(app/\S+\.py) ×(\d+) — ", key)
            if m:
                declared[m.group(1)] += int(m.group(2))

        over = {m: (declared[m], surv.get(m, 0))
                for m in declared if declared[m] > surv.get(m, 0)}
        assert not over, (
            f"以下模块的等价性证明声明的点数**多于**实测存活数 "
            f"（模块: 声明/存活）：{over} —— 要么证明盖住了已被杀死的点"
            f"（那条应改成 ×0 并注明），要么模块归属写错了")

        disp = d.get("survivor_disposition")
        assert disp, (
            "measured_modules.json 的 survivor_disposition 块被删了 —— "
            "『还有多少存活项没处置』必须留在交付物里")
        recorded = disp["unproven_by_module"]
        actual = {m: surv[m] - declared.get(m, 0)
                  for m in surv if surv[m] - declared.get(m, 0) > 0}
        assert actual == recorded, (
            f"逐模块未处置数与清单登记的对不上。\n"
            f"  实测：{actual}\n  清单：{recorded}\n"
            f"补了证明或杀了变异之后，请同步改小 survivor_disposition。")
        assert sum(actual.values()) <= disp["ratchet"]["unproven_max"], (
            f"未处置的存活项从 {disp['ratchet']['unproven_max']} 涨到 "
            f"{sum(actual.values())} —— 只许通过『补证明』或『写用例杀掉』来减少")

    def test_refuted_proofs_cannot_quietly_come_back(self):
        """
        **证明会是错的** —— 这是原棘轮没设想过的第三种情况。

        它只防"证明被删"，不防"证明本身站不住"。外部审计 2026-09-15 用一个
        反例推翻了 paper_broker 的两条 epsilon 证明；那两条现在登记在
        `REFUTED_EQUIVALENCE` 里，状态是**既未被杀死也无有效证明**。

        这条断言两件事：被推翻的条目只许增不许减，且不许同时出现在
        `PROVEN_EQUIVALENT` 里（那等于把推翻掉的东西又写了回去）。
        """
        d = self._manifest()
        rec = d["proof_reconciliation"]
        refuted = rec.get("refuted_by_counterexample")
        assert refuted and refuted.get("count"), (
            "proof_reconciliation.refuted_by_counterexample 被删了 —— "
            "被推翻的等价性证明必须保持可见")

        found = _collect_dict_entries("REFUTED_EQUIVALENCE")
        proven = _collect_dict_entries("PROVEN_EQUIVALENT")
        floor = rec["ratchet"].get("refuted_entries_min", 0)
        assert len(found) >= floor, (
            f"被反例推翻的条目从 {floor} 条降到 {len(found)} 条 —— "
            f"若确已重新证明或杀死，请连同反例用例一起处理再调低下限")
        back = sorted(set(found) & set(proven))
        assert not back, f"这些键既登记为已推翻又登记为已证明：{back}"


def _collect_dict_entries(var_name: str) -> list:
    """
    全库某个 dict 常量（如 PROVEN_EQUIVALENT）的所有键。

    **必须 `ast.walk` 而不是只看 `tree.body`。** 上一版只扫模块级赋值，
    于是写在**类里**的 `PROVEN_EQUIVALENT`（`test_position_store_schema.py`、
    `test_strategy_store_schema.py` 各一条）从来没被计入 —— 棘轮的下限
    因此一直比真实值小，"证明条数只许增不许减"这条守卫对它们完全无效。
    2026-09-16 自查发现，与自伤教训 #13 同型：判据没覆盖全，却被当成覆盖全了。
    """
    keys = []
    tests = _backend_root() / "tests"
    for p in tests.rglob("test_*.py"):
        if "__pycache__" in p.parts:
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        if var_name not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", None) == var_name for t in node.targets):
                if isinstance(node.value, ast.Dict):
                    keys += [k.value for k in node.value.keys
                             if isinstance(k, ast.Constant)]
            elif (isinstance(node, ast.AnnAssign)
                  and getattr(node.target, "id", None) == var_name
                  and isinstance(node.value, ast.Dict)):
                keys += [k.value for k in node.value.keys
                         if isinstance(k, ast.Constant)]
    return keys


def _count_proof_entries() -> int:
    """全库 PROVEN_EQUIVALENT 的条目总数。"""
    return len(_collect_dict_entries("PROVEN_EQUIVALENT"))
