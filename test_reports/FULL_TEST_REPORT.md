# Quant Agent 全面测试报告

> 生成时间：2026-06-14 01:23:31

## 一、执行摘要

| 维度 | 状态 | 总计 | 通过 | 失败 | 错误 | 跳过 | 耗时(s) |
|------|------|:----:|:----:|:----:|:----:|:----:|:-------:|
| **后端 (pytest)** | ❌ | 290 | 258 | 14 | 15 | 3 | 297.3 |
| **前端 (vitest)** | ✅ | 94 | 94 | 0 | 0 | 0 | 0.0 |
| **综合** | ❌ | **384** | **352** | **14** | **15** | **3** | **297.3** |

**整体通过率：91.7%**

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
| ❌ `tests/test_dsl_engine.py` | 9 | 3 | 0 | 0 |
| ❌ `tests/test_phase1_upgrade.py` | 12 | 2 | 0 | 0 |
| ✅ `tests/test_phase2.py` | 18 | 0 | 0 | 0 |
| ❌ `tests/test_phase3.py` | 9 | 9 | 15 | 0 |
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

共 **29** 个失败测试：

### 4.1 [后端] `tests/test_dsl_engine.py::test_window_validation`

- **结果**：failed
- **错误类型**：未分类错误

**错误摘要：**
```
tests\test_dsl_engine.py:179: in test_window_validation
    validator.validate(node)
app\core\alpha_engine\validator.py:182: in validate
    all_errors.extend(v.collect(node))
                      ^^^^^^^^^^^^^^^
app\core\alpha_engine\validator.py:149: in collect
    d = node.depth()
        ^^^^^^^^^^^^
app\core\alpha_engine\typed_nodes.py:229: in depth
    if self.second_child is not None:
```

### 4.2 [后端] `tests/test_dsl_engine.py::test_lookahead_validation`

- **结果**：failed
- **错误类型**：未分类错误

**错误摘要：**
```
tests\test_dsl_engine.py:197: in test_lookahead_validation
    validator.validate(node)
app\core\alpha_engine\validator.py:182: in validate
    all_errors.extend(v.collect(node))
                      ^^^^^^^^^^^^^^^
app\core\alpha_engine\validator.py:149: in collect
    d = node.depth()
        ^^^^^^^^^^^^
app\core\alpha_engine\typed_nodes.py:229: in depth
    if self.second_child is not None:
```

### 4.3 [后端] `tests/test_dsl_engine.py::test_cs_type_constraint`

- **结果**：failed
- **错误类型**：类型错误

**错误摘要：**
```
tests\test_dsl_engine.py:227: in test_cs_type_constraint
    with pytest.raises(TypeError, match="CrossSectionalNode"):
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   Failed: DID NOT RAISE <class 'TypeError'>
```

### 4.4 [后端] `tests/test_phase1_upgrade.py::TestDataPartitioner::test_split_ratio`

- **结果**：failed
- **错误类型**：未分类错误

**错误摘要：**
```
tests\test_phase1_upgrade.py:128: in test_split_ratio
    assert total == 100
E   assert 80 == 100
```

### 4.5 [后端] `tests/test_phase1_upgrade.py::TestRealisticBacktester::test_signal_shape`

- **结果**：failed
- **错误类型**：未分类错误

**错误摘要：**
```
tests\test_phase1_upgrade.py:213: in test_signal_shape
    assert res.processed_signal.shape == ds["close"].shape
E   assert (114, 20) == (120, 20)
E     
E     At index 0 diff: 114 != 120
E     Use -v to get more diff
```

### 4.6 [后端] `tests/test_phase3.py::TestConversationMemory::test_add_and_retrieve_last_dsl`

- **结果**：failed
- **错误类型**：模块未找到

**错误摘要：**
```
tests\test_phase3.py:80: in test_add_and_retrieve_last_dsl
    from app.core.ml_engine.quant_agent import ConversationMemory
E   ModuleNotFoundError: No module named 'app.core.ml_engine.quant_agent'
```

### 4.7 [后端] `tests/test_phase3.py::TestConversationMemory::test_history_text_not_empty`

- **结果**：failed
- **错误类型**：模块未找到

**错误摘要：**
```
tests\test_phase3.py:87: in test_history_text_not_empty
    from app.core.ml_engine.quant_agent import ConversationMemory
E   ModuleNotFoundError: No module named 'app.core.ml_engine.quant_agent'
```

### 4.8 [后端] `tests/test_phase3.py::TestConversationMemory::test_max_turns_enforced`

- **结果**：failed
- **错误类型**：模块未找到

**错误摘要：**
```
tests\test_phase3.py:95: in test_max_turns_enforced
    from app.core.ml_engine.quant_agent import ConversationMemory
E   ModuleNotFoundError: No module named 'app.core.ml_engine.quant_agent'
```

### 4.9 [后端] `tests/test_phase3.py::TestConversationMemory::test_last_metrics_updated`

- **结果**：failed
- **错误类型**：模块未找到

**错误摘要：**
```
tests\test_phase3.py:103: in test_last_metrics_updated
    from app.core.ml_engine.quant_agent import ConversationMemory
E   ModuleNotFoundError: No module named 'app.core.ml_engine.quant_agent'
```

### 4.10 [后端] `tests/test_phase3.py::TestQuantTools::test_generate_alpha_dsl_fallback`

- **结果**：error
- **错误类型**：未分类错误

**错误摘要：**
```
(无详细信息)
```

### 4.11 [后端] `tests/test_phase3.py::TestQuantTools::test_generate_dsl_unknown_keyword`

- **结果**：error
- **错误类型**：未分类错误

**错误摘要：**
```
(无详细信息)
```

### 4.12 [后端] `tests/test_phase3.py::TestQuantTools::test_run_optuna_returns_best_config`

- **结果**：error
- **错误类型**：未分类错误

**错误摘要：**
```
(无详细信息)
```

### 4.13 [后端] `tests/test_phase3.py::TestQuantTools::test_run_backtest_returns_sharpe`

- **结果**：error
- **错误类型**：未分类错误

**错误摘要：**
```
(无详细信息)
```

### 4.14 [后端] `tests/test_phase3.py::TestQuantTools::test_run_backtest_with_config`

- **结果**：error
- **错误类型**：未分类错误

**错误摘要：**
```
(无详细信息)
```

### 4.15 [后端] `tests/test_phase3.py::TestQuantTools::test_save_alpha_returns_id`

- **结果**：error
- **错误类型**：未分类错误

**错误摘要：**
```
(无详细信息)
```

### 4.16 [后端] `tests/test_phase3.py::TestOverfitCritic::test_pass_good_strategy`

- **结果**：failed
- **错误类型**：模块未找到

**错误摘要：**
```
tests\test_phase3.py:164: in test_pass_good_strategy
    from app.core.ml_engine.quant_agent import OverfitCritic
E   ModuleNotFoundError: No module named 'app.core.ml_engine.quant_agent'
```

### 4.17 [后端] `tests/test_phase3.py::TestOverfitCritic::test_fail_overfit`

- **结果**：failed
- **错误类型**：模块未找到

**错误摘要：**
```
tests\test_phase3.py:170: in test_fail_overfit
    from app.core.ml_engine.quant_agent import OverfitCritic
E   ModuleNotFoundError: No module named 'app.core.ml_engine.quant_agent'
```

### 4.18 [后端] `tests/test_phase3.py::TestOverfitCritic::test_fail_low_oos_sharpe`

- **结果**：failed
- **错误类型**：模块未找到

**错误摘要：**
```
tests\test_phase3.py:177: in test_fail_low_oos_sharpe
    from app.core.ml_engine.quant_agent import OverfitCritic
E   ModuleNotFoundError: No module named 'app.core.ml_engine.quant_agent'
```

### 4.19 [后端] `tests/test_phase3.py::TestOverfitCritic::test_no_oos_always_passes_overfit_check`

- **结果**：failed
- **错误类型**：模块未找到

**错误摘要：**
```
tests\test_phase3.py:183: in test_no_oos_always_passes_overfit_check
    from app.core.ml_engine.quant_agent import OverfitCritic
E   ModuleNotFoundError: No module named 'app.core.ml_engine.quant_agent'
```

### 4.20 [后端] `tests/test_phase3.py::TestFallbackOrchestrator::test_workflow_a_returns_dsl_and_metrics`

- **结果**：error
- **错误类型**：未分类错误

**错误摘要：**
```
(无详细信息)
```

### 4.21 [后端] `tests/test_phase3.py::TestFallbackOrchestrator::test_workflow_b_returns_metrics`

- **结果**：error
- **错误类型**：未分类错误

**错误摘要：**
```
(无详细信息)
```

### 4.22 [后端] `tests/test_phase3.py::TestFallbackOrchestrator::test_workflow_a_volume_keyword`

- **结果**：error
- **错误类型**：未分类错误

**错误摘要：**
```
(无详细信息)
```

### 4.23 [后端] `tests/test_phase3.py::TestQuantAgent::test_chat_returns_reply`

- **结果**：error
- **错误类型**：未分类错误

**错误摘要：**
```
(无详细信息)
```

### 4.24 [后端] `tests/test_phase3.py::TestQuantAgent::test_chat_returns_dsl`

- **结果**：error
- **错误类型**：未分类错误

**错误摘要：**
```
(无详细信息)
```

### 4.25 [后端] `tests/test_phase3.py::TestQuantAgent::test_chat_returns_metrics`

- **结果**：error
- **错误类型**：未分类错误

**错误摘要：**
```
(无详细信息)
```

### 4.26 [后端] `tests/test_phase3.py::TestQuantAgent::test_memory_persists_across_turns`

- **结果**：error
- **错误类型**：未分类错误

**错误摘要：**
```
(无详细信息)
```

### 4.27 [后端] `tests/test_phase3.py::TestQuantAgent::test_intent_detection_workflow_b`

- **结果**：error
- **错误类型**：未分类错误

**错误摘要：**
```
(无详细信息)
```

### 4.28 [后端] `tests/test_phase3.py::TestQuantAgent::test_intent_detection_workflow_a`

- **结果**：error
- **错误类型**：未分类错误

**错误摘要：**
```
(无详细信息)
```

### 4.29 [后端] `tests/test_phase3.py::TestChatAPI::test_session_memory_across_requests`

- **结果**：failed
- **错误类型**：断言失败

**错误摘要：**
```
tests\test_phase3.py:335: in test_session_memory_across_requests
    assert sid in sessions_resp.json()["sessions"]
E   AssertionError: assert 'memory_test_session' in [{'created_at': '2026-06-13 17:20:02.558224', 'session_id': '0f04f66b-b385-40a1-b283-265b749aefaf', 'title': 'session-...6-06-13 17:19:49.368578', 'session_id': 'multiturn-728815f9-a548-4f1a-ba43-d0a1dfbd8ee6', 'title': 'New Session
```

## 五、修复计划

| 优先级 | 错误类型 & 修复方案 | 影响测试数 |
|:------:|---------------------|:----------:|
| P1 | **模块未找到** — 确认模块已安装且 sys.path 正确 | 8 |
| P2 | **断言失败** — 对比期望值与实际值，检查逻辑变化 | 1 |
| P5 | **未分类错误** — 查看完整错误栈，逐步调试 | 19 |
| P5 | **类型错误** — 检查函数参数类型约束 | 1 |

### 详细修复步骤

#### 步骤 1：模块未找到 — 确认模块已安装且 sys.path 正确

受影响测试：
- [后端] `tests/test_phase3.py::TestConversationMemory::test_add_and_retrieve_last_dsl`
- [后端] `tests/test_phase3.py::TestConversationMemory::test_history_text_not_empty`
- [后端] `tests/test_phase3.py::TestConversationMemory::test_max_turns_enforced`
- [后端] `tests/test_phase3.py::TestConversationMemory::test_last_metrics_updated`
- [后端] `tests/test_phase3.py::TestOverfitCritic::test_pass_good_strategy`
- ……（共 8 个）

#### 步骤 2：断言失败 — 对比期望值与实际值，检查逻辑变化

受影响测试：
- [后端] `tests/test_phase3.py::TestChatAPI::test_session_memory_across_requests`

#### 步骤 3：未分类错误 — 查看完整错误栈，逐步调试

受影响测试：
- [后端] `tests/test_dsl_engine.py::test_window_validation`
- [后端] `tests/test_dsl_engine.py::test_lookahead_validation`
- [后端] `tests/test_phase1_upgrade.py::TestDataPartitioner::test_split_ratio`
- [后端] `tests/test_phase1_upgrade.py::TestRealisticBacktester::test_signal_shape`
- [后端] `tests/test_phase3.py::TestQuantTools::test_generate_alpha_dsl_fallback`
- ……（共 19 个）

#### 步骤 4：类型错误 — 检查函数参数类型约束

受影响测试：
- [后端] `tests/test_dsl_engine.py::test_cs_type_constraint`

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
