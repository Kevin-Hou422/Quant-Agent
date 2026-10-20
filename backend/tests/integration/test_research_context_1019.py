"""
复审 2026-10-19 N09-c：CLI、chat、工作流、发现、回测、验证和提案都取得**同一个显式 ResearchContext**，
携带 dataset / universe / experiment / boundary。

此前每个入口各自切研究窗口（research_window / split_from_settings / _three_way / discovery._selection /
workflows._research_split / 提案端点 / chat 的 load_real_dataset），兜底规则互不一致 ——
chat 的 GP 指定数据集时甚至按全量 70/30 切，OOS 正好压在冻结 Test 段上。

本文件的断言：
  1. 结构：切研究窗口的函数只在 research_context.py 里被调用（不再有第二条路径）；
  2. 每个入口都经 open_research 取上下文（换成哨兵后每个入口都撞上它）；
  3. chat GP 指定真实数据集时，交给 GP 的数据不含冻结段日期；
  4. 上下文的内容：dataset / universe / experiment / boundary；配置开关关不掉已冻结的边界；
  5. 搜索运行记录里带着上下文；搜索拿到越界数据时拒绝开始（不进台账）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.conftest import _make_dataset

KEY = "audit_us"
APP = Path(__file__).resolve().parents[2] / "app"


@pytest.fixture
def pinned(fresh_db):
    """真实数据集标识 + 已冻结的边界（第一次切分时写入）。"""
    from app.core.data_engine.data_partitioner import split_from_settings
    ds = _make_dataset(1600, 10)
    frozen = split_from_settings(ds, dataset_key=KEY)
    assert frozen.test_start is not None
    return ds, frozen.test_start


class Opened(Exception):
    def __init__(self, kw):
        super().__init__(kw)
        self.kw = kw


def _sentinel(*a, **kw):
    names = ("dataset", "dataset_key", "purpose")
    kw = {**dict(zip(names, a)), **kw}
    raise Opened(kw)


# ---------------------------------------------------------------------------
# 1. 结构：只有一条路径
# ---------------------------------------------------------------------------

def test_only_the_research_context_cuts_the_research_window():
    allowed = {"research_context.py", "data_partitioner.py"}
    pat = re.compile(r"(?<![\w.])(?:\w+\.)?(split_from_settings|research_window|partition_three_way)\(")
    offenders = []
    for p in APP.rglob("*.py"):
        if p.name in allowed:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(line) and not line.lstrip().startswith(("def ", "#")):
                offenders.append(f"{p.relative_to(APP)}:{i}: {line.strip()}")
    assert not offenders, "研究窗口还有别的切法（应统一经 open_research）：\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# 2. 每个入口都经 open_research
# ---------------------------------------------------------------------------

def _patched():
    return patch("app.core.data_engine.research_context.open_research", _sentinel)


def test_cli_entry_points_open_the_context(pinned):
    import app.main as main
    ds, _ = pinned
    for fn in (main.run_gp, main.run_backtest, main.run_agent, main.run_realistic):
        with _patched(), patch.object(main, "_load_dataset", return_value=ds):
            with pytest.raises(Opened) as ei:
                fn(SimpleNamespace(pop_size=2, generations=1, optuna_trials=0, dataset=KEY, dsl="rank(close)",
                                   oos_ratio=0.3, wf_splits=0, hypothesis="x", max_iterations=1))
        assert ei.value.kw["dataset_key"] == KEY, fn.__name__


def test_chat_opens_the_context(pinned):
    from app.agent._tools import QuantTools
    ds, _ = pinned
    with _patched(), patch("app.core.data_engine.dataset_registry.load_registry_dataset",
                           return_value=SimpleNamespace(data=ds)):
        with pytest.raises(Opened) as ei:
            QuantTools(dataset_name=KEY)
    assert ei.value.kw["dataset_key"] == KEY


def test_workflows_open_the_context(pinned):
    from app.core.workflows.alpha_workflows import GenerationWorkflow, OptimizationWorkflow
    ds, _ = pinned
    with _patched():
        with pytest.raises(Opened) as a:
            GenerationWorkflow(pop_size=2, n_generations=1, n_optuna_trials=0).run("momentum", ds, dataset_key=KEY)
        with pytest.raises(Opened) as b:
            OptimizationWorkflow(pop_size=2, n_generations=1, n_optuna_trials=0).run(
                "rank(close)", ds, dataset_key=KEY)
    assert a.value.kw["dataset_key"] == b.value.kw["dataset_key"] == KEY


def test_discovery_opens_the_context(pinned):
    from app.core.discovery.discovery_engine import DiscoveryEngine
    ds, _ = pinned
    with _patched():
        with pytest.raises(Opened) as ei:
            DiscoveryEngine(n_families=1, pop_size=2, n_generations=1, n_optuna_trials=0).run(
                ds, save=False, dataset_key=KEY)
    assert ei.value.kw["dataset_key"] == KEY


def test_api_backtest_paths_open_the_context(pinned):
    import app.api.router as router
    ds, _ = pinned
    with _patched():
        with pytest.raises(Opened):
            router._three_way(ds, KEY, "backtest/run")
        with patch("app.core.data_engine.dataset_registry.load_registry_dataset",
                   return_value=SimpleNamespace(data=ds)):
            with pytest.raises(Opened):
                router._resolve_dataset(KEY, "2020-01-01", "2026-01-01", 10, 100, 0, 0.3)


def test_validation_gate_opens_the_context(pinned):
    from app.core.lifecycle.validation_gate import ValidationGate
    ds, _ = pinned
    with _patched():
        with pytest.raises(Opened) as ei:
            ValidationGate(n_splits=2).evaluate("rank(close)", ds, n_trials=1, dataset_key=KEY)
    assert ei.value.kw["dataset_key"] == KEY


def test_the_proposal_opens_the_context(pinned, monkeypatch):
    import app.api.router as router
    from app.config import settings
    ds, _ = pinned
    monkeypatch.setattr(settings, "paper_dataset", KEY)
    with _patched(), patch("app.core.data_engine.dataset_registry.load_registry_dataset",
                           return_value=SimpleNamespace(data=ds)):
        with pytest.raises(Opened) as ei:
            router.strategy_propose(store=SimpleNamespace())
    assert ei.value.kw["dataset_key"] == KEY


# ---------------------------------------------------------------------------
# 3. chat GP 指定真实数据集：此前按全量 70/30 切
# ---------------------------------------------------------------------------

def test_chat_gp_on_a_named_dataset_never_sees_frozen_dates(pinned):
    from app.agent._tools import QuantTools
    from app.core.gp_engine.population_evolver import PopulationEvolver
    ds, boundary = pinned
    seen = {}

    class Stop(Exception):
        pass

    def capture(self, *a, **kw):
        seen["is"] = int((kw["is_data"]["close"].index >= boundary).sum())
        seen["oos"] = int((kw["oos_data"]["close"].index >= boundary).sum())
        raise Stop()
    with patch("app.core.data_engine.dataset_registry.load_registry_dataset",
               return_value=SimpleNamespace(data=ds)):
        tools = QuantTools(allow_synthetic=True)
        with patch.object(PopulationEvolver, "__init__", capture):
            with pytest.raises(Stop):
                tools.tool_run_gp_optimization(seed_dsl="rank(close)", n_generations=1, pop_size=2,
                                               n_optuna_trials=0, dataset_name=KEY)
    assert seen == {"is": 0, "oos": 0}, f"chat GP 拿到了冻结段日期：{seen}"


# ---------------------------------------------------------------------------
# 4. 上下文的内容
# ---------------------------------------------------------------------------

def test_the_context_carries_dataset_universe_experiment_and_boundary(pinned):
    from app.core.data_engine.research_context import open_research
    ds, boundary = pinned
    ctx = open_research(ds, KEY, "cli.gp", experiment="exp-1")
    d = ctx.describe()
    assert d["dataset_key"] == KEY and d["experiment"] == "exp-1" and d["purpose"] == "cli.gp"
    assert d["test_start"] == str(boundary.date()) and d["pinned"] is True
    assert d["universe_n"] == 10 and len(d["universe_digest"]) == 16
    assert ctx.selection["close"].index.max() < boundary
    # 同一份数据、不同入口 → 除 purpose 外完全相同
    other = open_research(ds, KEY, "api.backtest", experiment="exp-1").describe()
    assert {k: v for k, v in other.items() if k != "purpose"} == {k: v for k, v in d.items() if k != "purpose"}


def test_a_different_universe_is_a_different_context(pinned):
    from app.core.data_engine.research_context import open_research
    ds, _ = pinned
    sub = {k: (v.iloc[:, :5] if hasattr(v, "iloc") and v.ndim == 2 else v) for k, v in ds.items()}
    a = open_research(ds, KEY, "x").describe()
    b = open_research(sub, KEY, "x").describe()
    assert a["test_start"] == b["test_start"]
    assert a["universe_digest"] != b["universe_digest"] and b["universe_n"] == 5


def test_the_config_switch_cannot_unfreeze_a_pinned_boundary(pinned, monkeypatch):
    """s_three_way_enabled=False 此前让 API / 验证门看全量（含冻结段）。已冻结的边界不能被开关关掉。"""
    from app.config import settings
    from app.core.data_engine.research_context import open_research
    ds, boundary = pinned
    monkeypatch.setattr(settings, "s_three_way_enabled", False)
    ctx = open_research(ds, KEY, "api.backtest")
    assert ctx.selection["close"].index.max() < boundary
    ctx2 = open_research(ds, KEY, "validation_gate", three_way=False)
    assert ctx2.selection["close"].index.max() < boundary


def test_synthetic_data_has_an_unpinned_context():
    from app.core.data_engine.research_context import open_research
    ctx = open_research(_make_dataset(1600, 10), None, "cli.gp")
    d = ctx.describe()
    assert d["dataset_key"] == "" and d["pinned"] is False


# ---------------------------------------------------------------------------
# 5. 运行记录带上下文；越界数据不开始搜索
# ---------------------------------------------------------------------------

def test_a_search_run_records_its_context(pinned):
    from app.core.data_engine.research_context import open_research
    from app.db.trial_ledger import TrialLedger, TrialRecorder
    ds, _ = pinned
    ctx = open_research(ds, KEY, "cli.gp", experiment="exp-2")
    with ctx.active():
        rec = TrialRecorder("gp", panels=[ctx.selection])
        rec.add(3)
        rec.finish("done")
    row = [r for r in TrialLedger().run_states("gp") if r.run_id == rec.run_id][0]
    got = json.loads(row.context)
    assert got["dataset_key"] == KEY and got["experiment"] == "exp-2"
    assert got["test_start"] == ctx.describe()["test_start"]
    assert got["universe_digest"] == ctx.describe()["universe_digest"]


def test_a_search_over_frozen_dates_does_not_start(pinned):
    from app.core.data_engine.data_partitioner import FrozenBoundaryError
    from app.core.data_engine.research_context import open_research
    from app.db.trial_ledger import TrialLedger, TrialRecorder
    ds, _ = pinned
    ctx = open_research(ds, KEY, "cli.gp")
    n_before = len(TrialLedger().run_states())
    with ctx.active(), pytest.raises(FrozenBoundaryError):
        TrialRecorder("gp", panels=[ds])
    assert len(TrialLedger().run_states()) == n_before


def test_gp_and_optuna_hand_their_panels_to_the_recorder(pinned):
    """GP / Optuna 在活动上下文下拿到越界数据 → 拒绝（证明它们把面板交给了记账器做检查）。"""
    from app.core.data_engine.data_partitioner import FrozenBoundaryError
    from app.core.data_engine.research_context import open_research
    from app.core.gp_engine.population_evolver import PopulationEvolver
    from app.core.ml_engine.alpha_optimizer import AlphaOptimizer
    ds, _ = pinned
    ctx = open_research(ds, KEY, "test")
    with ctx.active():
        with pytest.raises(FrozenBoundaryError):
            PopulationEvolver(is_data=ctx.selection, oos_data=ds, pop_size=2, n_generations=1).run(
                seed_dsl="rank(close)", n_optuna_trials=0)
        with pytest.raises(FrozenBoundaryError):
            AlphaOptimizer("rank(close)", ds, n_trials=1).optimize()
