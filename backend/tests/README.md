# tests/ 导航

**149 个文件 / 3394 个用例。**

## 怎么读这个目录

目录按**被测对象**分层，不按开发阶段分。
（2026-09 整理前有 33 个 `test_phaseN_*.py` 按开发阶段命名 —— 想找
「回测引擎的测试」无从下手。已全部按实际测试内容改名归位。）

```
tests/
  meta/          对套件本身的约束 —— 审计从这里开始
  unit/<包>/     单模块（或单包）测试，与 app/ 的包结构一一对应
  integration/   跨包链路 + 真实 HTTP 端点
  golden/        逐位钉死的黄金基准
  performance/   性能基线
```

## 强度是怎么保证的

覆盖率**不是**本项目采纳的强度指标。判据是**变异测试击杀率**：
机械地把产品代码改坏（`>` 改 `>=`、`and` 改 `or`、算符符号反转……），
看有没有测试变红。改坏了没人管的地方叫「存活」，那就是真实盲区。

达标标准**不是击杀率数字本身**，而是 **存活项 100% 处置**：
要么补出能杀死它的用例，要么给出**可机械验证**的等价性证明
（写在对应测试文件的 `PROVEN_EQUIVALENT` 里，并配一条会随前提变化而变红的验证用例）。

`app/` 下 111 个 `.py` 里，**88 个有变异点，已全部测量**（其余 23 个是
`__init__.py` 与纯常量模块，零变异点）。合计 **1980 个变异点**。
清单在 [`meta/measured_modules.json`](meta/measured_modules.json)，
由 `meta/test_invariants.py` 对账 —— 新增模块没测量就会红。

完整台账见 [`../MUTATION_LEDGER.md`](../MUTATION_LEDGER.md)：每一轮的数字、
每一条等价性证明、9 条工具缺陷史、以及 9 条「我自己写错过的断言」的教训。

## 每次运行都该盯的两个数

- **`N xfailed`** —— 已登记但**尚未修复**的产品缺陷数（`meta/test_known_defects.py`）。
  这不是噪声，是欠账。修好一个会变成 XPASS 并**直接判失败**，
  强制修复者同步更新登记表、台账、以及模块里「钉住现状」的那条断言。
- **`meta/test_lessons_enforced.py` 全绿** —— 说明没有新的零断言 / 重言式 /
  吞异常 / 条件断言混进来。这个文件抓过多条我自己写的弱用例，
  最近一次是 14 条（其中一条 `if complexity >= 4:` 的守卫从未成立，
  那个用例一直什么都没检查）。

## 怎么跑

```bash
cd backend
python -m pytest tests/ -q                       # 全量（空载约 12 分钟；机器有负载时可到 20 分钟）
python -m pytest tests/meta/ -q                  # 只跑套件自检
python -m pytest tests/unit/gp_engine/ -q        # 只跑某个包

# 打乱执行顺序，暴露用例间的隐式依赖（本项目**没有**装 pytest-randomly，
# 这是仓库自带的零依赖插件）
python -m pytest tests/ -q -p shuffle_check
```

## 写新测试前必读

- **能杀就不要写等价证明。**「输出相同」只是没找到对的观察面，
  不等于「观察不到」（台账自伤教训 #7）。
- **不要用源码文本断言**（`assert "daemon=True" in src`）：同一串在文件里
  出现多次就杀不掉任何东西（自伤教训 #6）。用 AST 做全称量化。
- **凡是能捅到进程外的行为**（弹浏览器、发信、打开文件关联程序）
  必须在 `conftest.py` 里全局堵死，不能只在「相关」用例里 monkeypatch ——
  变异测试会把代码跑在你没预期的配置下（自伤教训 #8，实测事故）。
- **守卫即契约就写成断言**。`if <条件>: assert ...` 在条件不成立时
  一条都不检查；确需守卫就在分支内计数并断言计数 > 0。

## `tests/golden/`

黄金基准：结果被逐位钉死，防止无意的数值漂移。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_golden_backtest.py` | 13 | 黄金基准测试（Task 6.4，E5/E-N4 核心补齐，2026-07-30） |

## `tests/integration/`

**跨包链路**：一条路径串起两个以上的包，或走真实 HTTP 端点。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_alpha_discovery.py` | 6 | 6 项 alpha_discovery 单元测试 |
| `test_api_backtest.py` | 17 | 回测 API 端点集成测试 |
| `test_api_chat.py` | 16 | Chat API 端点集成测试（无 LLM，Fallback 模式） |
| `test_api_chat_stream.py` | 7 | SSE 流式对话端点 |
| `test_api_datasets.py` | 7 | 数据集 API 端点集成测试 |
| `test_api_gp.py` | 5 | GP 进化 API 端点集成测试 |
| `test_api_health.py` | 3 | 基础健康检查端点测试 |
| `test_api_report.py` | 6 | Alpha 历史报告查询 API 测试 |
| `test_api_uncovered_routes.py` | 13 | 补齐债务台账里"零测试覆盖"的 API 路由 |
| `test_api_workflow.py` | 9 | Workflow API 端点集成测试 |
| `test_approval_workflow_api.py` | 7 | Phase 9.4 人工审批工作流验收 |
| `test_calendar_and_forward_split.py` | 11 | Phase 11：交易日历 + 回放/前向分离 |
| `test_conversational_agent_chain.py` | 29 | Phase 3 对话式 Quant Agent 测试 |
| `test_cost_model_calibration_loop.py` | 6 | Phase 8.2 成本模型校准回路验收 |
| `test_daily_ingest_health_gate.py` | 4 | Task 7.1 每日摄取健康门验收（2026-08-02） |
| `test_daily_ingest_increment.py` | 21 | 前向增量与健康门的定钉测试（变异测试驱动） |
| `test_diagnostics_and_providers.py` | 15 | 定钉测试（变异测试驱动） |
| `test_discovery_and_chat_router.py` | 19 | 定钉测试（变异测试驱动） |
| `test_fail_closed_and_sqlite_hardening.py` | 15 | Phase 6 基础设施加固与正确性修复验收（2026-07-26） |
| `test_gp_reproducibility_and_manifest.py` | 10 | Task 6.5 可复现性验收（2026-07-30） |
| `test_graded_promotion_gates.py` | 9 | Phase TR.4 分级晋级门 + 阈值配置化 + 实验模式 |
| `test_horizon_and_no_trade_band.py` | 6 | Phase PM.6 horizon 感知 验收 |
| `test_incremental_ingest_forward.py` | 4 | Phase 11.1/11.3：真前向增量摄取 |
| `test_lifecycle_monitor_scheduler_chain.py` | 31 | Phase 5 验收测试 |
| `test_multi_factor_capacity_and_capital.py` | 5 | Phase PM 第一批验收 |
| `test_nightly_discovery_orchestration.py` | 5 | Phase 9.2 自主发现编排器 + 夜间任务验收 |
| `test_optimizer_and_evaluator_chain.py` | 18 | Phase 2 高级评估 + Optuna 优化器测试 |
| `test_regime_mvo_and_backtest_chain.py` | 35 | Phase 4 验收测试 |
| `test_signal_partition_backtest_chain.py` | 14 | Phase 1 核心升级测试 |
| `test_supplementary_fixes.py` | 13 | 补充审计（SUPPLEMENTARY_AUDIT_FINDINGS，2026-07-24 核实）回归测试 |
| `test_trial_ledger_pbo_and_tstat_gate.py` | 6 | Phase S.3 验收：全局 trial 计数 + PBO + t≥3.0 门槛 |
| `test_validation_gate_api.py` | 7 | Phase 9.3 自动验证门验收 |
| `test_validation_gate_contracts.py` | 21 | 自动验证门的定钉测试 |

## `tests/meta/`

**对套件与代码库本身的约束**。不测某个功能，测的是「测试有没有资格被信任」「有没有模块写了没接线」「已登记缺陷还欠几个」「测试会不会对进程外产生副作用」。审计从这三个文件开始看。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_invariants.py` | 14 | 系统级**不变量**（不是用例） |
| `test_known_defects.py` | 30 | 每条一个 `xfail(strict=True)` 用例。 |
| `test_lessons_enforced.py` | 49 | 把 DEV_LESSONS 的每一条从散文变成**可执行的强制检查** |

## `tests/performance/`

性能基线。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_perf_api.py` | 6 | API 并发与顺序负载测试 |
| `test_perf_dsl.py` | 5 | DSL 执行性能基准测试 |

## `tests/unit/agent/`

LLM agent：意图路由、无 LLM 回退、工具守卫、批评器阈值、LangChain 接线、假设→DSL 闭环、系统提示词与代码的一致性。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_agent_critic.py` | 12 | OverfitCritic 阈值逻辑测试 |
| `test_agent_data_utils.py` | 52 | QuantTools 的数据构造与切割 |
| `test_agent_fallback.py` | 21 | FallbackOrchestrator 与 QuantTools 测试（无 LLM 模式） |
| `test_agent_helpers_and_history.py` | 28 | 两个小但要害的 agent 辅助模块 |
| `test_agent_routing.py` | 28 | 意图路由、上下文注入、持久化守卫的定钉测试（变异测试驱动） |
| `test_agent_tools_guards.py` | 31 | QuantTools 的守卫与默认值（变异测试驱动） |
| `test_alpha_agent_loop.py` | 64 | 假设 → DSL → 评估 → 精炼 的闭环 |
| `test_critic_thresholds.py` | 17 | 红队判定阈值的定钉测试（变异测试驱动） |
| `test_lc_agent_wiring.py` | 35 | QuantTools 接进 LangChain AgentExecutor 的那层接线 |
| `test_system_prompt_consistency.py` | 24 | 系统提示词与代码的一致性 |

## `tests/unit/alpha_engine/`

DSL 语言层：解析、类型、校验、算子内核、信号处理管道、因子的金融解读与诊断。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_dsl_edge_cases.py` | 23 | DSL 解析器边界值与异常输入测试 |
| `test_dsl_engine.py` | 12 | Unit tests for the Alpha DSL Engine. |
| `test_dsl_executor_pipeline.py` | 27 | 执行管线的定钉测试（变异测试驱动） |
| `test_dsl_operators.py` | 27 | 全算子族执行覆盖测试 |
| `test_dsl_parser_and_validator_bounds.py` | 37 | 元数/边界的定钉测试（变异测试驱动） |
| `test_fast_ops_kernel.py` | 63 | 算子内核的定钉测试（变异测试驱动） |
| `test_financial_diagnostics.py` | 48 | 回测指标 → 金融诊断与改进建议 |
| `test_financial_interpreter.py` | 62 | DSL → 金融语义的翻译与因子家族分类 |
| `test_leak_filter.py` | 13 | 因子入池门（**系统默认门**）的直接测试 |
| `test_signal_processor_pipeline.py` | 28 | 四步信号后处理的定钉测试（变异测试驱动） |
| `test_typed_nodes_semantics.py` | 37 | DSL AST 节点语义的定钉测试（变异测试驱动） |

## `tests/unit/api/`

API 层的守卫与默认值（并发闸、超时后的锁记账、请求模型默认值）。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_api_router_guards.py` | 24 | 并发闸、超时/锁记账、请求校验与默认值（变异测试驱动） |
| `test_api_router_round2.py` | 26 | 第二轮补强（变异测试驱动） |

## `tests/unit/backtest_engine/`

回测与绩效：撮合账务、组合构造、绩效公式、风险报告、过拟合统计、多因子合成、跨市场稳健性、出图保真度。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_alpha_combiner.py` | 71 | 多因子合成 |
| `test_backtest_edge_cases.py` | 11 | 回测引擎边界值测试 |
| `test_backtest_engine.py` | 7 | 回测引擎 7 项单元测试 |
| `test_backtest_engine_accounting.py` | 18 | 逐日记账公式的定钉测试（变异测试驱动） |
| `test_backtest_plot_fidelity.py` | 40 | 回测曲线保留且真实（研究用，非伪动画） |
| `test_multi_dataset_backtester.py` | 49 | 跨市场稳健性回测 |
| `test_overfit_stats_pbo.py` | 21 | CSCV / PBO 的定钉测试（变异测试驱动） |
| `test_performance_analyzer_formulas.py` | 70 | 逐公式定钉测试（变异测试驱动） |
| `test_portfolio_constructor_formulas.py` | 32 | 逐公式定钉测试（变异测试驱动） |
| `test_realistic_backtester_contracts.py` | 26 | 契约与边界定钉测试（变异测试驱动） |
| `test_risk_report_contracts.py` | 32 | 契约与边界定钉测试（变异测试驱动） |
| `test_run_portfolio_wiring.py` | 3 | 验证 PM.S1/S2/PM.5 已接进主线 run_portfolio（不是"库但没生效"） |
| `test_transaction_cost_formulas.py` | 23 | 逐公式定钉测试（变异测试驱动） |

## `tests/unit/data_engine/`

数据层：数据集注册与筛选、IS/OOS 切分、PIT 存储、市场日历、各行情源（Yahoo / moomoo / ccxt / akshare / 本地 parquet）、市场状态。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_akshare_provider.py` | 35 | A 股日线（前复权） |
| `test_ccxt_provider.py` | 40 | Binance 加密货币日线 |
| `test_data_partitioner_split.py` | 61 | IS/OOS 切分与 embargo 的定钉测试（变异测试驱动） |
| `test_dataset_filters.py` | 125 | 数据集选定之后的动态筛选 |
| `test_dataset_registry_loading.py` | 28 | 取数入口的定钉测试（变异测试驱动） |
| `test_health_report.py` | 62 | 数据质量健康检查 |
| `test_local_parquet_provider.py` | 51 | 本地 Parquet 落盘与回读 |
| `test_market_calendar_fail_closed.py` | 19 | fail-closed 与时区归一的定钉测试（变异测试驱动） |
| `test_moomoo_paging_and_window.py` | 17 | 分页/节流/增量窗口的定钉测试（变异测试驱动） |
| `test_moomoo_panel_assembly.py` | 6 | Phase TR.2 MoomooProvider 验收 |
| `test_multi_dataset.py` | 76 | 多市场数据集抽象与注册表 |
| `test_pit_bitemporal_queries.py` | 10 | Phase 8.1 Point-in-Time 数据存储验收 |
| `test_pit_store_storage.py` | 15 | 落盘与合并契约的定钉测试（变异测试驱动） |
| `test_provider_base.py` | 21 | DataProvider 抽象基类 |
| `test_regime_detector_labels.py` | 33 | Regime 标注边界的定钉测试（变异测试驱动） |
| `test_schema_enforcer.py` | 36 | 所有数据源进入面板工厂前的**唯一**强制层 |
| `test_sector_mapper.py` | 30 | 静态 GICS 行业分类映射 |
| `test_yahoo_provider.py` | 23 | Yahoo Finance 行情源 |

## `tests/unit/db/`

持久化层：各 store 的 schema、账务、幂等、可复现 manifest。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_alpha_store_schema.py` | 21 | schema、迁移与前向标记的定钉测试（变异测试驱动） |
| `test_chat_store_schema.py` | 13 | schema 与引擎配置的定钉测试（变异测试驱动） |
| `test_db_alpha_store.py` | 13 | AlphaStore CRUD 测试（内存 SQLite） |
| `test_db_chat_store.py` | 17 | ChatStore 会话管理测试（内存 SQLite） |
| `test_diagnostics_persistence_endpoints.py` | 3 | FE-TR 数据源验收（诊断持久化 + 两个只读端点） |
| `test_paper_broker_accounting.py` | 17 | PaperBroker 记账口径的**逐项**保护 |
| `test_paper_broker_replay_parity.py` | 8 | Phase 7 Paper Trading 验收（2026-08-02） |
| `test_position_store_schema.py` | 20 | 账本 schema 与查询契约的定钉测试（变异测试驱动） |
| `test_run_manifest_reproducibility.py` | 23 | 可复现台账的定钉测试（变异测试驱动） |
| `test_strategy_config_endpoints.py` | 5 | Phase PM.7 策略配置端点 + active 配置交易接线 |
| `test_strategy_store_and_config_build.py` | 3 | Phase PM.7 策略配置一等实体 验收（核心，端点另测） |
| `test_strategy_store_schema.py` | 18 | 策略配置台账的 schema 定钉测试（变异测试驱动） |

## `tests/unit/discovery/`

自主发现：市场观察引擎的家族排序。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_market_observer_regime_ranking.py` | 4 | Phase 9.1 市场观察引擎验收 |
| `test_market_observer_scoring.py` | 25 | 自主发现打分的定钉测试（变异测试驱动） |

## `tests/unit/entrypoint/`

进程入口与配置：启动自检（绑定地址安全闸）、CORS、生产默认值。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_main_startup_guards.py` | 20 | 启动自检、CORS、合成数据与 CLI 参数的定钉测试（变异测试驱动） |
| `test_production_defaults.py` | 11 | 在**发布配置**下验证门控是否真的会拦截 |
| `test_settings_boolean_defaults.py` | 11 | 全部布尔默认值的定钉测试（变异测试驱动） |

## `tests/unit/gp_engine/`

遗传规划：因子池、适应度、变异/交叉算子、种群进化、快速评估工具。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_alpha_pool_dedup.py` | 24 | 去重、剪枝、正交化的定钉测试（变异测试驱动） |
| `test_evaluation_utils.py` | 22 | 快速 IC-IR 评估（不跑回测引擎） |
| `test_gp_alpha_pool.py` | 14 | AlphaPool 去重、相关性过滤、容量管理测试 |
| `test_gp_engine_fitness.py` | 26 | GP 适应度公式的定钉测试（变异测试驱动） |
| `test_gp_evolution_full.py` | 5 | GP 演化完整流程测试 |
| `test_gp_fitness_weights.py` | 26 | 适应度合成、量纲稳定性、变异权重的定钉测试（变异测试驱动） |
| `test_gp_mutation_operators.py` | 75 | GP 变异/交叉算子的定钉测试（变异测试驱动） |
| `test_population_evolver_internals.py` | 29 | 进化循环内部的定钉测试（变异测试驱动） |
| `test_population_evolver_round2.py` | 16 | 第二轮补强（变异测试驱动） |
| `test_population_evolver_round3.py` | 15 | 第三轮收尾（变异测试驱动） |

## `tests/unit/lifecycle/`

生命周期门：晋级门的阈值边界。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_promotion_gate_boundaries.py` | 19 | 边界与默认值定钉测试（变异测试驱动） |

## `tests/unit/ml_engine/`

机器学习侧：Optuna 搜索、因子评估、代理剪枝模型。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_alpha_evaluator_overfit.py` | 28 | 过拟合评分与滚动指标的定钉测试（变异测试驱动） |
| `test_alpha_optimizer_search.py` | 34 | 超参数搜索的定钉测试（变异测试驱动） |
| `test_ml_optimizer.py` | 8 | Optuna 参数优化器测试 |
| `test_proxy_model_pruning.py` | 22 | 早期剪枝代理模型的定钉测试（变异测试驱动） |

## `tests/unit/monitor/`

因子监控：衰减告警。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_alpha_monitor_decay.py` | 21 | 衰减检测边界的定钉测试（变异测试驱动） |

## `tests/unit/portfolio_manager/`

组合管理：资金账本、风控门、策略门、边际准入、基准策略库、无交易带与快慢分类、策略配置装配。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_baseline_strategy_library.py` | 7 | Phase PM.S3 经典基准策略库验收 |
| `test_daily_loop_money_and_gates.py` | 6 | 实盘交易循环的「钱」与「门」逐项保护 |
| `test_horizon_band_and_classification.py` | 38 | 无交易带 + 因子快慢分类（Phase PM.6） |
| `test_portfolio_manager_book.py` | 18 | 账本视图与容量约束的定钉测试（变异测试驱动） |
| `test_portfolio_risk_gates.py` | 7 | Phase PM.5 组合级风控门 验收 |
| `test_risk_gate_formulas.py` | 30 | 风控门公式的**逐条**钉死 |
| `test_strategy_builder_assembly.py` | 39 | 把六道工序收敛成一份可审批的策略配置 |
| `test_strategy_gate_and_marginal_selection.py` | 9 | Phase PM.S1/S2 策略级门 + 边际贡献准入 验收 |
| `test_strategy_gate_contracts.py` | 44 | 策略级验证门的逐条定钉测试（变异测试驱动） |

## `tests/unit/tasks/`

定时任务：调度注册、备份、成本校准、推理谱系记录。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_backup.py` | 6 | 每日一致性快照备份 |
| `test_backup_failure_paths.py` | 17 | 备份失败路径与保留策略的定钉测试（变异测试驱动） |
| `test_cost_calibration_report.py` | 15 | 成本校准建议的定钉测试（变异测试驱动） |
| `test_reasoning_log.py` | 20 | Agent 推理过程的结构化记录 |
| `test_scheduler_registration.py` | 20 | 任务注册与生命周期的定钉测试（变异测试驱动） |

## `tests/unit/trading_context/`

交易现实：价差估计、可做空性、盘口/借券/账户 provider。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_daily_loop_survivors.py` | 26 | 把 `daily_trading_loop.py` 剩余存活变异**逐条**处理完 |
| `test_quote_borrow_account_providers.py` | 6 | Phase TR.3 T3 providers（盘口/借券/账户）验收 |
| `test_spread_estimators.py` | 18 | 价差估计量的定钉测试（变异测试驱动） |
| `test_trading_context_reality.py` | 17 | 交易现实摘要的定钉测试（变异测试驱动） |
| `test_trading_context_shortability.py` | 7 | Phase TR（TradingContext）验收 |

## `tests/unit/workflows/`

Workflow A/B 的内部 helper 与 held-out 隔离。

| 文件 | 用例 | 测什么 |
|---|---:|---|
| `test_alpha_workflows_internals.py` | 42 | Workflow A/B 内部helper 的定钉测试（变异测试驱动） |
| `test_alpha_workflows_round2.py` | 25 | 第二轮补强（变异测试驱动） |
| `test_alpha_workflows_round3.py` | 10 | 第三轮收尾（变异测试驱动） |
| `test_holdout_isolation.py` | 4 | Phase S.1+S.2 验收 |
