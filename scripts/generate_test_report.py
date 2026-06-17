#!/usr/bin/env python3
"""
generate_test_report.py — 综合测试报告生成器

读取：
  test_reports/backend_results.json   (pytest-json-report 格式)
  test_reports/frontend_results.json  (vitest --reporter=json 格式)

输出：
  test_reports/FULL_TEST_REPORT.md
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT        = Path(__file__).parent.parent
REPORT_DIR  = ROOT / "test_reports"
BACKEND_JSON  = REPORT_DIR / "backend_results.json"
FRONTEND_JSON = REPORT_DIR / "frontend_results.json"
BACKEND_STDOUT  = REPORT_DIR / "backend_stdout.txt"
FRONTEND_STDOUT = REPORT_DIR / "frontend_stdout.txt"
OUTPUT_MD   = REPORT_DIR / "FULL_TEST_REPORT.md"


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"[WARN] {path} not found — skipping")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_backend(data: dict) -> dict:
    """Parse pytest-json-report output."""
    summary = data.get("summary", {})
    tests   = data.get("tests", [])

    total   = summary.get("total",   len(tests))
    passed  = summary.get("passed",  0)
    failed  = summary.get("failed",  0)
    error   = summary.get("error",   0)
    skipped = summary.get("skipped", 0)
    duration = data.get("duration", 0)

    failures = []
    for t in tests:
        outcome = t.get("outcome", "")
        if outcome in ("failed", "error"):
            node_id = t.get("nodeid", "")
            call    = t.get("call", {}) or {}
            longrepr = call.get("longrepr", "") or t.get("longrepr", "")
            # 截取前 400 字符
            snippet  = str(longrepr)[:400].strip()
            failures.append({
                "nodeid":  node_id,
                "outcome": outcome,
                "snippet": snippet,
            })

    # 按文件分组统计
    by_file: dict[str, dict] = {}
    for t in tests:
        nid   = t.get("nodeid", "")
        fname = nid.split("::")[0] if "::" in nid else nid
        # 简化路径
        fname = fname.replace("\\", "/").replace("backend/tests/", "")
        if fname not in by_file:
            by_file[fname] = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
        outcome = t.get("outcome", "unknown")
        by_file[fname][outcome] = by_file[fname].get(outcome, 0) + 1

    return {
        "total": total, "passed": passed, "failed": failed,
        "error": error, "skipped": skipped, "duration": duration,
        "failures": failures, "by_file": by_file,
    }


def _parse_frontend(data: dict) -> dict:
    """Parse vitest --reporter=json output."""
    # vitest json format
    num_total_suites = data.get("numTotalTestSuites", 0)
    num_passed = data.get("numPassedTests", 0)
    num_failed = data.get("numFailedTests", 0)
    num_skipped = data.get("numPendingTests", 0)
    num_total = data.get("numTotalTests", 0)
    duration = data.get("testResults", [{}])[0].get("perfStats", {}).get("runtime", 0) if data.get("testResults") else 0

    failures = []
    by_file: dict[str, dict] = {}

    for suite in data.get("testResults", []):
        file_path = suite.get("testFilePath", "unknown")
        fname = file_path.replace("\\", "/").split("src/__tests__/")[-1]
        by_file.setdefault(fname, {"passed": 0, "failed": 0, "error": 0, "skipped": 0})
        for test in suite.get("testResults", []):
            status = test.get("status", "unknown")
            if status == "passed":
                by_file[fname]["passed"] += 1
            elif status == "failed":
                by_file[fname]["failed"] += 1
                failures.append({
                    "nodeid":  f"{fname} > {test.get('ancestorTitles', [''])[0]} > {test.get('title', '')}",
                    "outcome": "failed",
                    "snippet": "\n".join(test.get("failureMessages", []))[:400],
                })
            elif status == "pending":
                by_file[fname]["skipped"] += 1

    return {
        "total": num_total, "passed": num_passed, "failed": num_failed,
        "error": 0, "skipped": num_skipped, "duration": duration / 1000,
        "failures": failures, "by_file": by_file,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fix planner
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_PATTERNS: list[tuple[str, str, str]] = [
    # (匹配关键词, 根因类别, 修复建议)
    ("ImportError",         "导入错误",       "检查模块路径和依赖安装（pip install / npm install）"),
    ("ModuleNotFoundError", "模块未找到",     "确认模块已安装且 sys.path 正确"),
    ("AttributeError",      "属性不存在",     "检查类/函数签名是否已更改，更新测试断言"),
    ("ParseError",          "DSL 解析失败",   "检查测试 DSL 字符串语法是否符合当前 Parser 版本"),
    ("ValidationError",     "验证规则不符",   "检查验证阈值是否与测试用例对齐"),
    ("AssertionError",      "断言失败",       "对比期望值与实际值，检查逻辑变化"),
    ("TypeError",           "类型错误",       "检查函数参数类型约束"),
    ("KeyError",            "字段缺失",       "检查 API 响应结构是否有新增/删除字段"),
    ("TimeoutError",        "超时",           "增大 timeout 参数或使用更小的测试数据集"),
    ("ConnectionError",     "网络连接失败",   "集成测试需后端运行；或添加 skip 标记"),
    ("Cannot find module",  "模块未找到(TS)", "执行 npm install 安装缺失依赖"),
    ("is not assignable",   "TypeScript 类型", "修复 TypeScript 类型定义不匹配问题"),
    ("not a function",      "函数不存在(TS)", "检查导出函数名是否正确"),
]

def _classify_failure(snippet: str) -> tuple[str, str]:
    for keyword, category, fix in KNOWN_PATTERNS:
        if keyword.lower() in snippet.lower():
            return category, fix
    return "未分类错误", "查看完整错误栈，逐步调试"


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def _status_emoji(passed: int, failed: int, error: int) -> str:
    if failed == 0 and error == 0:
        return "✅"
    if failed + error < 5:
        return "⚠️"
    return "❌"


def _build_report(backend: dict | None, frontend: dict | None) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        "# Quant Agent 全面测试报告",
        "",
        f"> 生成时间：{now}",
        "",
    ]

    # ── 摘要总览 ────────────────────────────────────────────────────────────
    lines += ["## 一、执行摘要", ""]

    b = backend  or {"total": 0, "passed": 0, "failed": 0, "error": 0, "skipped": 0, "duration": 0}
    f = frontend or {"total": 0, "passed": 0, "failed": 0, "error": 0, "skipped": 0, "duration": 0}

    b_emoji = _status_emoji(b["passed"], b["failed"], b["error"])
    f_emoji = _status_emoji(f["passed"], f["failed"], f["error"])

    grand_total   = b["total"]   + f["total"]
    grand_passed  = b["passed"]  + f["passed"]
    grand_failed  = b["failed"]  + f["failed"]
    grand_error   = b["error"]   + f["error"]
    grand_skipped = b["skipped"] + f["skipped"]
    grand_dur     = b["duration"] + f["duration"]
    pass_rate     = (grand_passed / grand_total * 100) if grand_total > 0 else 0

    lines += [
        "| 维度 | 状态 | 总计 | 通过 | 失败 | 错误 | 跳过 | 耗时(s) |",
        "|------|------|:----:|:----:|:----:|:----:|:----:|:-------:|",
        f"| **后端 (pytest)** | {b_emoji} | {b['total']} | {b['passed']} | {b['failed']} | {b['error']} | {b['skipped']} | {b['duration']:.1f} |",
        f"| **前端 (vitest)** | {f_emoji} | {f['total']} | {f['passed']} | {f['failed']} | {f['error']} | {f['skipped']} | {f['duration']:.1f} |",
        f"| **综合** | {'✅' if grand_failed+grand_error==0 else '❌'} | **{grand_total}** | **{grand_passed}** | **{grand_failed}** | **{grand_error}** | **{grand_skipped}** | **{grand_dur:.1f}** |",
        "",
        f"**整体通过率：{pass_rate:.1f}%**",
        "",
    ]

    # ── 后端分文件统计 ───────────────────────────────────────────────────────
    lines += ["## 二、后端测试详情", ""]

    if backend is None:
        lines += ["> ⚠️ 未找到后端测试结果文件。", ""]
    else:
        if backend["by_file"]:
            lines += [
                "### 2.1 按文件统计",
                "",
                "| 文件 | 通过 | 失败 | 错误 | 跳过 |",
                "|------|:----:|:----:|:----:|:----:|",
            ]
            for fname, counts in sorted(backend["by_file"].items()):
                row_emoji = "✅" if counts["failed"] + counts.get("error", 0) == 0 else "❌"
                lines.append(
                    f"| {row_emoji} `{fname}` | {counts['passed']} | {counts['failed']} | {counts.get('error',0)} | {counts.get('skipped',0)} |"
                )
            lines.append("")

    # ── 前端分文件统计 ───────────────────────────────────────────────────────
    lines += ["## 三、前端测试详情", ""]

    if frontend is None:
        lines += ["> ⚠️ 未找到前端测试结果文件。", ""]
    else:
        if frontend["by_file"]:
            lines += [
                "### 3.1 按文件统计",
                "",
                "| 文件 | 通过 | 失败 | 跳过 |",
                "|------|:----:|:----:|:----:|",
            ]
            for fname, counts in sorted(frontend["by_file"].items()):
                row_emoji = "✅" if counts["failed"] == 0 else "❌"
                lines.append(
                    f"| {row_emoji} `{fname}` | {counts['passed']} | {counts['failed']} | {counts.get('skipped',0)} |"
                )
            lines.append("")

    # ── 失败清单 ─────────────────────────────────────────────────────────────
    all_failures = []
    if backend:
        all_failures += [("后端", f) for f in backend.get("failures", [])]
    if frontend:
        all_failures += [("前端", f) for f in frontend.get("failures", [])]

    lines += ["## 四、失败测试详情", ""]

    if not all_failures:
        lines += ["> 🎉 所有测试均通过，无失败项。", ""]
    else:
        lines += [f"共 **{len(all_failures)}** 个失败测试：", ""]
        for idx, (layer, failure) in enumerate(all_failures, 1):
            category, fix_hint = _classify_failure(failure["snippet"])
            lines += [
                f"### 4.{idx} [{layer}] `{failure['nodeid']}`",
                "",
                f"- **结果**：{failure['outcome']}",
                f"- **错误类型**：{category}",
                "",
                "**错误摘要：**",
                "```",
                failure["snippet"] or "(无详细信息)",
                "```",
                "",
            ]

    # ── 修复计划 ─────────────────────────────────────────────────────────────
    lines += ["## 五、修复计划", ""]

    if not all_failures:
        lines += ["> 无需修复。", ""]
    else:
        # 按错误类型汇总
        fix_groups: dict[str, list[str]] = {}
        for layer, failure in all_failures:
            category, fix = _classify_failure(failure["snippet"])
            key = f"**{category}** — {fix}"
            fix_groups.setdefault(key, []).append(f"[{layer}] `{failure['nodeid']}`")

        lines += [
            "| 优先级 | 错误类型 & 修复方案 | 影响测试数 |",
            "|:------:|---------------------|:----------:|",
        ]
        priority_map = {
            "导入错误": 1, "模块未找到": 1, "模块未找到(TS)": 1,
            "断言失败": 2, "属性不存在": 2, "字段缺失": 2,
            "DSL 解析失败": 3, "验证规则不符": 3,
            "超时": 4, "网络连接失败": 4,
        }
        sorted_groups = sorted(
            fix_groups.items(),
            key=lambda kv: priority_map.get(kv[0].split("**")[1] if "**" in kv[0] else "", 5),
        )
        for fix_key, test_ids in sorted_groups:
            cat_name = fix_key.split("**")[1] if "**" in fix_key else fix_key
            prio = priority_map.get(cat_name, 5)
            lines.append(f"| P{prio} | {fix_key} | {len(test_ids)} |")
        lines.append("")

        lines += ["### 详细修复步骤", ""]
        step = 1
        for fix_key, test_ids in sorted_groups:
            lines += [
                f"#### 步骤 {step}：{fix_key.replace('**', '')}",
                "",
                "受影响测试：",
            ]
            for tid in test_ids[:5]:  # 最多显示 5 个
                lines.append(f"- {tid}")
            if len(test_ids) > 5:
                lines.append(f"- ……（共 {len(test_ids)} 个）")
            lines.append("")
            step += 1

    # ── 覆盖率目标 ───────────────────────────────────────────────────────────
    lines += [
        "## 六、覆盖率目标追踪",
        "",
        "| 层级 | 目标覆盖率 | 当前状态 |",
        "|------|:----------:|:--------:|",
        "| 后端核心引擎（DSL/回测/GP） | ≥ 75% | 待 coverage 报告 |",
        "| 后端 API 端点 | ≥ 80% | 待 coverage 报告 |",
        "| 前端组件 | ≥ 60% | 待 coverage 报告 |",
        "| 前端 Store | ≥ 85% | 待 coverage 报告 |",
        "",
        "> 运行 `pytest --cov=app --cov-report=html` 和 `npm run test:coverage` 生成详细覆盖率报告。",
        "",
    ]

    # ── 新增测试文件索引 ─────────────────────────────────────────────────────
    lines += [
        "## 七、测试文件索引",
        "",
        "### 后端单元测试",
        "",
        "| 文件 | 测试维度 |",
        "|------|---------|",
        "| `unit/test_dsl_edge_cases.py` | DSL 解析边界值、异常输入、安全性 |",
        "| `unit/test_dsl_operators.py` | 全算子族执行正确性 |",
        "| `unit/test_backtest_edge_cases.py` | 回测引擎极端场景 |",
        "| `unit/test_gp_alpha_pool.py` | AlphaPool 去重、相关性、容量 |",
        "| `unit/test_gp_evolution_full.py` | GP 演化完整流程 |",
        "| `unit/test_ml_optimizer.py` | Optuna 参数优化 |",
        "| `unit/test_agent_critic.py` | OverfitCritic 阈值逻辑 |",
        "| `unit/test_agent_fallback.py` | FallbackOrchestrator 意图识别 |",
        "| `unit/test_db_alpha_store.py` | AlphaStore CRUD |",
        "| `unit/test_db_chat_store.py` | ChatStore 会话管理 |",
        "",
        "### 后端集成测试",
        "",
        "| 文件 | 端点覆盖 |",
        "|------|---------|",
        "| `integration/test_api_health.py` | GET /health |",
        "| `integration/test_api_backtest.py` | /api/backtest/* |",
        "| `integration/test_api_workflow.py` | /api/workflow/* |",
        "| `integration/test_api_gp.py` | /api/gp/evolve |",
        "| `integration/test_api_datasets.py` | /api/datasets/* |",
        "| `integration/test_api_report.py` | /api/report/query |",
        "| `integration/test_api_chat.py` | /api/chat/* |",
        "",
        "### 后端性能测试",
        "",
        "| 文件 | 测试维度 |",
        "|------|---------|",
        "| `performance/test_perf_dsl.py` | DSL 解析/执行性能基准 |",
        "| `performance/test_perf_api.py` | API 顺序与并发性能 |",
        "",
        "### 前端测试",
        "",
        "| 文件 | 测试维度 |",
        "|------|---------|",
        "| `unit/store/workspaceStore.test.ts` | Zustand store 全状态管理 |",
        "| `unit/api/client.test.ts` | Axios 客户端 Mock |",
        "| `components/analysis/OverfitBadge.test.tsx` | 过拟合徽标组件 |",
        "| `components/analysis/MetricsGrid.test.tsx` | 指标网格组件 |",
        "| `components/layout/GlobalSidebar.test.tsx` | 导航侧边栏 |",
        "| `components/chat/ChatMessage.test.tsx` | 聊天消息组件 |",
        "| `components/compiler/ConfigModal.test.tsx` | 配置模态框 |",
        "| `integration/workflow.test.tsx` | 工作流集成状态流 |",
        "",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    backend_raw  = _load_json(BACKEND_JSON)
    frontend_raw = _load_json(FRONTEND_JSON)

    backend  = _parse_backend(backend_raw)  if backend_raw  else None
    frontend = _parse_frontend(frontend_raw) if frontend_raw else None

    # 也尝试从 stdout 中解析（作为备用）
    if backend is None and BACKEND_STDOUT.exists():
        print("[INFO] 后端 JSON 报告不存在，将从 stdout 提取摘要（无详细失败信息）")
        stdout = BACKEND_STDOUT.read_text(encoding="utf-8", errors="replace")
        # 尝试提取 "X passed, Y failed" 格式
        import re
        m = re.search(r"(\d+) passed(?:, (\d+) failed)?(?:, (\d+) error)?", stdout)
        if m:
            passed  = int(m.group(1))
            failed  = int(m.group(2) or 0)
            error   = int(m.group(3) or 0)
            backend = {"total": passed+failed+error, "passed": passed, "failed": failed,
                       "error": error, "skipped": 0, "duration": 0, "failures": [], "by_file": {}}
            print(f"[INFO] 从 stdout 解析: {passed} passed, {failed} failed, {error} error")

    report_md = _build_report(backend, frontend)
    OUTPUT_MD.write_text(report_md, encoding="utf-8")
    print(f"\n[SUCCESS] 报告已生成：{OUTPUT_MD}")
    print(f"          行数：{report_md.count(chr(10))}")

    # 打印摘要到 stdout
    if backend:
        status = "PASS" if backend["failed"]+backend["error"] == 0 else "FAIL"
        print(f"\n[{status}] Backend  {backend['passed']}/{backend['total']} passed  ({backend['failed']+backend['error']} failed/error)")
    if frontend:
        status = "PASS" if frontend["failed"]+frontend["error"] == 0 else "FAIL"
        print(f"[{status}] Frontend {frontend['passed']}/{frontend['total']} passed  ({frontend['failed']+frontend['error']} failed/error)")


if __name__ == "__main__":
    main()
