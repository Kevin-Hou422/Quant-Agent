# Quant Agent 全面测试报告

> 生成时间：2026-06-17 23:48:39

## 一、执行摘要

| 维度 | 状态 | 总计 | 通过 | 失败 | 错误 | 跳过 | 耗时(s) |
|------|------|:----:|:----:|:----:|:----:|:----:|:-------:|
| **后端 (pytest)** | ✅ | 290 | 287 | 0 | 0 | 3 | 370.0 |
| **前端 (vitest)** | ✅ | 94 | 94 | 0 | 0 | 0 | 0.0 |
| **综合** | ✅ | **384** | **381** | **0** | **0** | **3** | **370.0** |

**整体通过率：99.2%**

## 二、后端测试详情

### 2.1 按文件统计

| 文件 | 通过 | 失败 | 错误 | 跳过 |
|------|:----:|:----:|:----:|:----:|
| ✅ `tests/integration/test_api_backtest.py` | 13 | 0 | 0 | 0 |
| ✅ `tests/integration/test_api_chat.py` | 16 | 0 | 0 | 0 |
| ✅ `tests/integration/test_api_datasets.py` | 7 | 0 | 0 | 0 |
| ✅ `tests/integration/test_api_gp.py` | 5 | 0 | 0 | 0 |
| ✅ `tests/integration/test_api_health.py` | 3 | 0 | 0 | 0 |
| ✅ `tests/integration/test_api_report.py` | 6 | 0 | 0 | 0 |
| ✅ `tests/integration/test_api_workflow.py` | 8 | 0 | 0 | 0 |
| ✅ `tests/performance/test_perf_api.py` | 6 | 0 | 0 | 0 |
| ✅ `tests/performance/test_perf_dsl.py` | 5 | 0 | 0 | 0 |
| ✅ `tests/test_backtest_engine.py` | 7 | 0 | 0 | 0 |
| ✅ `tests/test_dsl_engine.py` | 12 | 0 | 0 | 0 |
| ✅ `tests/test_phase1_upgrade.py` | 14 | 0 | 0 | 0 |
| ✅ `tests/test_phase2.py` | 18 | 0 | 0 | 0 |
| ✅ `tests/test_phase3.py` | 33 | 0 | 0 | 0 |
| ✅ `tests/unit/test_agent_critic.py` | 12 | 0 | 0 | 0 |
| ✅ `tests/unit/test_agent_fallback.py` | 8 | 0 | 0 | 2 |
| ✅ `tests/unit/test_backtest_edge_cases.py` | 8 | 0 | 0 | 1 |
| ✅ `tests/unit/test_db_alpha_store.py` | 13 | 0 | 0 | 0 |
| ✅ `tests/unit/test_db_chat_store.py` | 17 | 0 | 0 | 0 |
| ✅ `tests/unit/test_dsl_edge_cases.py` | 23 | 0 | 0 | 0 |
| ✅ `tests/unit/test_dsl_operators.py` | 26 | 0 | 0 | 0 |
| ✅ `tests/unit/test_gp_alpha_pool.py` | 14 | 0 | 0 | 0 |
| ✅ `tests/unit/test_gp_evolution_full.py` | 5 | 0 | 0 | 0 |
| ✅ `tests/unit/test_ml_optimizer.py` | 8 | 0 | 0 | 0 |

## 三、前端测试详情

### 3.1 按文件统计

| 文件 | 通过 | 失败 | 跳过 |
|------|:----:|:----:|:----:|
| ✅ `unknown` | 0 | 0 | 0 |

## 四、失败测试详情

> 🎉 所有测试均通过，无失败项。

## 五、修复计划

> 无需修复。

## 六、覆盖率目标追踪

| 层级 | 目标覆盖率 | 当前状态 |
|------|:----------:|:--------:|
| 后端核心引擎（DSL/回测/GP） | ≥ 75% | 待 coverage 报告 |
| 后端 API 端点 | ≥ 80% | 待 coverage 报告 |
| 前端组件 | ≥ 60% | 待 coverage 报告 |
| 前端 Store | ≥ 85% | 待 coverage 报告 |

> 运行 `pytest --cov=app --cov-report=html` 和 `npm run test:coverage` 生成详细覆盖率报告。

## 七、测试文件索引

### 后端单元测试

| 文件 | 测试维度 |
|------|---------|
| `unit/test_dsl_edge_cases.py` | DSL 解析边界值、异常输入、安全性 |
| `unit/test_dsl_operators.py` | 全算子族执行正确性 |
| `unit/test_backtest_edge_cases.py` | 回测引擎极端场景 |
| `unit/test_gp_alpha_pool.py` | AlphaPool 去重、相关性、容量 |
| `unit/test_gp_evolution_full.py` | GP 演化完整流程 |
| `unit/test_ml_optimizer.py` | Optuna 参数优化 |
| `unit/test_agent_critic.py` | OverfitCritic 阈值逻辑 |
| `unit/test_agent_fallback.py` | FallbackOrchestrator 意图识别 |
| `unit/test_db_alpha_store.py` | AlphaStore CRUD |
| `unit/test_db_chat_store.py` | ChatStore 会话管理 |

### 后端集成测试

| 文件 | 端点覆盖 |
|------|---------|
| `integration/test_api_health.py` | GET /health |
| `integration/test_api_backtest.py` | /api/backtest/* |
| `integration/test_api_workflow.py` | /api/workflow/* |
| `integration/test_api_gp.py` | /api/gp/evolve |
| `integration/test_api_datasets.py` | /api/datasets/* |
| `integration/test_api_report.py` | /api/report/query |
| `integration/test_api_chat.py` | /api/chat/* |

### 后端性能测试

| 文件 | 测试维度 |
|------|---------|
| `performance/test_perf_dsl.py` | DSL 解析/执行性能基准 |
| `performance/test_perf_api.py` | API 顺序与并发性能 |

### 前端测试

| 文件 | 测试维度 |
|------|---------|
| `unit/store/workspaceStore.test.ts` | Zustand store 全状态管理 |
| `unit/api/client.test.ts` | Axios 客户端 Mock |
| `components/analysis/OverfitBadge.test.tsx` | 过拟合徽标组件 |
| `components/analysis/MetricsGrid.test.tsx` | 指标网格组件 |
| `components/layout/GlobalSidebar.test.tsx` | 导航侧边栏 |
| `components/chat/ChatMessage.test.tsx` | 聊天消息组件 |
| `components/compiler/ConfigModal.test.tsx` | 配置模态框 |
| `integration/workflow.test.tsx` | 工作流集成状态流 |
