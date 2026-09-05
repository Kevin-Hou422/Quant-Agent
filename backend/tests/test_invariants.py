"""
test_invariants.py — 系统级**不变量**（不是用例）

背景（DEV_LESSONS §R）：同一会话内"只测局部"反复出现 4 次，根因是**验证用例由与实现
同一个心智模型生成**，因此结构性看不见同一盲区。用例抓实例，**不变量抓类别**。
本文件把外部审计打中的那几类问题固化成"再犯就红"的断言。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
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


def test_chat_response_carries_data_source():
    """聊天响应必须带 data_source，否则用户无法分辨 Sharpe 是不是随机数。"""
    from app.api.chat_router import ChatResponse
    assert "data_source" in ChatResponse.model_fields


def test_stream_done_event_carries_data_source():
    """
    前端实际消费的是 SSE /api/chat/stream，不是 POST /api/chat。
    只给 POST 加 data_source = 只修了一半（DEV_LESSONS §R）——最终 done 事件必须也带。
    """
    from app.agent.quant_agent import QuantAgent
    agent = QuantAgent(n_tickers=20, n_days=252, n_trials=1,
                       dataset_name="", allow_synthetic=True)
    events: list[dict] = []
    agent.stream_chat("rank(close)", session_id="inv_stream",
                      on_event=lambda e: events.append(e))
    done = [e for e in events if e.get("type") == "done"]
    assert done, f"stream_chat 未发出 done 事件：{[e.get('type') for e in events]}"
    assert done[-1]["result"].get("data_source") == agent.data_source


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
