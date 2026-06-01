# Quant Agent Backend — 综合审计报告

**审计日期：** 2026-06-01  
**覆盖范围：** `backend/` 全部 Python 源文件（85 个 .py 文件）  
**审计方法：** 全量代码精读 + 跨模块关联分析 + 金融业务深度评估

---

## 目录

1. [工程架构总览](#1-工程架构总览)
2. [模块关联性与一致性](#2-模块关联性与一致性)
3. [系统鲁棒性](#3-系统鲁棒性)
4. [冗余代码识别](#4-冗余代码识别)
5. [架构合理性评分](#5-架构合理性评分)
6. [工作流完整性](#6-工作流完整性)
7. [金融业务深度分析](#7-金融业务深度分析)
8. [15点实现程度详评](#8-15点实现程度详评)
9. [距离可赚钱系统的差距](#9-距离可赚钱系统的差距)
10. [优先级改进建议](#10-优先级改进建议)

---

## 1. 工程架构总览

### 1.1 目录结构（精简版）

```
backend/
├── app/
│   ├── main.py                  # 双模式入口（FastAPI + CLI）
│   ├── config.py                # pydantic-settings 全局配置
│   ├── api/
│   │   ├── router.py            # REST 路由
│   │   └── chat_router.py       # 流式 SSE 聊天路由
│   ├── agent/                   # LLM Agent 层（9个文件）
│   │   ├── alpha_agent.py       # 旧版 AlphaAgent（hypothesis→DSL→评估）
│   │   ├── quant_agent.py       # 新版 LangChain Agent（含工具调用）
│   │   └── _*.py                # Agent 内部模块
│   ├── core/
│   │   ├── alpha_engine/        # DSL 解析→执行核心（12个文件）
│   │   ├── data_engine/         # 多源数据管道（15个文件 + providers/）
│   │   ├── backtest_engine/     # 回测引擎（8个文件）
│   │   ├── gp_engine/           # 遗传规划引擎（5个文件）
│   │   ├── ml_engine/           # ML 辅助（3个文件）
│   │   ├── optimization_engine/ # [冗余包] 与 ml_engine/data_engine 重叠
│   │   ├── portfolio_engine/    # [冗余包] 与 backtest_engine 完全重叠
│   │   └── workflows/
│   │       └── alpha_workflows.py  # GenerationWorkflow + OptimizationWorkflow
│   ├── db/
│   │   ├── alpha_store.py       # SQLite Alpha 持久化
│   │   └── chat_store.py
│   └── tasks/
│       └── reasoning_log.py
└── tests/
```

### 1.2 技术栈概览

| 层级 | 技术选型 |
|------|----------|
| Web 框架 | FastAPI + Uvicorn |
| LLM | LangChain + GPT-4o |
| DSL 解析 | Lark 文法（Earley 解析器） |
| GP 进化 | 自研 TypedNode AST 变异（无 DEAP） |
| 参数调优 | Optuna |
| 代理剪枝 | XGBoost ProxyModel |
| 数据处理 | Pandas + NumPy + Bottleneck |
| 数据源 | yfinance / akshare / ccxt / Alpha Vantage |
| 持久化 | SQLite（alpha_store）+ Parquet（feature_store）|

---

## 2. 模块关联性与一致性

### 2.1 主数据流向（正确路径）

```
假设（自然语言）
    ↓
AlphaAgent / GenerationWorkflow
    ↓
PopulationEvolver.run()              [GP进化主引擎]
    ├── _init_population()           [DSL→TypedNode]
    ├── _evaluate_one()
    │     ├── RealisticBacktester    [IS+OOS 回测]
    │     │     ├── SignalProcessor  [截断→衰减→中性化→延迟]
    │     │     ├── PortfolioConstructor  [Long-Short/Decile]
    │     │     └── BacktestEngine   [逐日PnL+成本]
    │     └── compute_fitness()      [多目标适应度]
    ├── AlphaPool.add()              [相关性去重]
    └── mutation_weights_from_metrics()  [自适应变异权重]
    ↓
_optuna_fine_tune()                  [参数调优（仅IS）]
    ↓
GPEvolutionResult → AlphaStore.save()
```

### 2.2 一致性问题（严重）

#### 问题 A：真实数据与优化主循环断路

`main.py` 中**所有 CLI 模式**均调用合成数据生成器：

```python
# main.py:119 — AlphaAgent 路径
dataset = _make_synthetic_dataset()      # 20只虚拟股票，120天随机游走

# main.py:140 — AlphaEvolver 路径
dataset = _make_synthetic_dataset(n_tickers=args.n_tickers, n_days=args.n_days)
```

而 `dataset_registry.py` 实现了 10 个真实数据集的完整加载链，`DataManager` 实现了 8 步数据管道（缓存→获取→Schema→预处理→健康检查），但**两者从未被主工作流调用**。真实数据管道完全孤立。

#### 问题 B：`groups` 字段使用假行业编号

`dsl_executor.py:75-78` 自动生成 groups 字段：

```python
# 将资产按序号对10取模分为10组 — 与真实行业无关
grp = np.tile(np.arange(N) % 10, (len(close), 1))
aligned["groups"] = pd.DataFrame(grp, index=close.index, columns=close.columns)
```

所有依赖 `group_rank`、`group_zscore`、`ind_neutralize` 的 DSL 实际上在做**假的行业中性化**。这是金融逻辑上的根本性错误，会使截面 Alpha 的研究结论失真。

#### 问题 C：import 路径不一致

- `realistic_backtester.py` 使用绝对导入：`from app.core.alpha_engine.signal_processor import...`
- `gp_engine.py` 使用相对导入：`from ..alpha_engine.parser import Parser`
- 两种风格混用，在不同工作目录下运行时可能触发 `ModuleNotFoundError`

#### 问题 D：`gp_engine.py` 中 fitness 完全忽略成本

`_evaluate_individual()` 中 fitness 公式基于 IC-IR 作为 Sharpe 代理，不调用 BacktestEngine，完全不计算交易成本。而 `population_evolver.py` 中使用 RealisticBacktester 计算真实 Sharpe（含成本）。两条路径的 fitness 计算方法不一致。

---

## 3. 系统鲁棒性

### 3.1 已实现的防护机制

| 机制 | 实现位置 | 质量 |
|------|----------|------|
| IS/OOS 物理隔离 | `PartitionedDataset.__slots__` | 良好 |
| 前视偏差检测 | `DatetimeIndex.is_monotonic_increasing` 验证 | 良好 |
| 重复日期检测 | `df.index.duplicated()` 检查 | 良好 |
| ADV 流动性截断 | `LiquidityConstraint.apply()` | 良好 |
| 净值归零熔断 | `if equity <= 0: ruin_date = dates[t]; break` | 良好 |
| ffill 上限 | `ffill(limit=5)` 防止停牌无限填充 | 良好 |
| 信号 Burn-in 裁剪 | 定位首个有效信号行，裁剪预热期 | 良好 |
| GP 个体去重 | `seen` set 基于 `repr(node)` 哈希 | 良好 |
| DSL 深度/类型验证 | `AlphaValidator` 多级检查 | 良好 |
| 执行结果 memoize | `per-run Cache` 字典（同子表达式只算一次）| 良好 |
| 实例级 RNG | `random.Random(seed)` 替代 `random.seed()` | 良好 |
| 数据集结构验证 | `_validate_dataset()` 入口校验 | 良好 |

### 3.2 鲁棒性缺口

| 缺口 | 风险级别 | 说明 |
|------|----------|------|
| API 无请求限速 | 高 | GP 进化耗时任务可被重复触发，无并发控制 |
| GP 进化无超时 | 高 | 大种群×多代数时无上限运行时间，可 OOM |
| AlphaStore 无并发锁 | 中 | 多请求并发写 SQLite 可能数据损坏 |
| multiprocessing.Pool 在 Windows 下不稳定 | 中 | spawn 模式下序列化限制，建议改为线程池或关闭多进程 |
| OOS 最小行数检验过宽松 | 中 | `len(oos) > 5` 检查：5 天数据不足以计算可信 Sharpe |
| Parquet FeatureStore 无文件锁 | 低 | 并发读写可能产生损坏文件 |

---

## 4. 冗余代码识别

### 4.1 模块级冗余（可直接删除）

#### 冗余组 1：`portfolio_engine/` 完全冗余

```
app/core/portfolio_engine/signal_processor.py
    ←→  app/core/alpha_engine/signal_processor.py     （实质相同）

app/core/portfolio_engine/realistic_backtester.py
    ←→  app/core/backtest_engine/realistic_backtester.py  （实质相同）

app/core/portfolio_engine/portfolio_constructor.py
    ←→  app/core/backtest_engine/portfolio_constructor.py  （实质相同）
```

`portfolio_engine/` 是旧版拷贝，主流程调用的是 `backtest_engine/` 版本，可安全删除整个包（约 3 个文件）。

#### 冗余组 2：`optimization_engine/` 完全冗余

```
app/core/optimization_engine/data_partitioner.py
    ←→  app/core/data_engine/data_partitioner.py       （实质相同）

app/core/optimization_engine/alpha_optimizer.py
    ←→  app/core/ml_engine/alpha_optimizer.py          （实质相同）

app/core/optimization_engine/alpha_evaluator.py
    ←→  app/core/ml_engine/alpha_evaluator.py          （实质相同）
```

`optimization_engine/` 可安全删除整个包（约 3 个文件）。

#### 冗余组 3：双版本 GP 引擎

```
app/core/gp_engine/gp_engine.py       # AlphaEvolver（旧版）
    ←→  app/core/gp_engine/population_evolver.py  # PopulationEvolver（新版主引擎）
```

`AlphaEvolver` 仍被 `main.py:run_gp()` 调用但已是死路；`PopulationEvolver` 是生产路径。应将 `run_gp()` 迁移后删除旧版（保留 `_SEED_DSLS` 和 `generate_random_alpha()` 函数，它们被新版共用）。

#### 冗余组 4：`alpha_engine/generator.py`

旧版随机 DSL 生成器，功能已被 `gp_engine.py` 中的 seed library + `generate_random_alpha()` 完全覆盖，可删除。

### 4.2 函数级冗余

| 冗余位置 | 说明 |
|----------|------|
| `alpha_agent._quick_eval()` ←→ `population_evolver._quick_metrics()` | 功能相同的快速 IC-IR 评估，两套实现 |
| IS/OOS 分区逻辑 | 在 3 处独立实现（`data_engine/`、`optimization_engine/`、`alpha_workflows._partition()`）|
| `operators.py`（Pandas）←→ `fast_ops.py`（NumPy） | 部分算子双重实现，调用路径不一致 |

---

## 5. 架构合理性评分

| 维度 | 评分（/10） | 说明 |
|------|-------------|------|
| 分层清晰度 | 7 | 核心六大引擎分层合理；冗余的 portfolio_engine / optimization_engine 破坏了清晰度 |
| 接口设计 | 7 | TypedNode AST + 工厂方法设计良好；`Executor.run` vs `run_expr` 命名有歧义 |
| 依赖方向 | 6 | data_engine → alpha_engine → backtest_engine 方向正确；但 agent 层直接操作 db 绕过服务层 |
| 扩展性 | 7 | DataProvider 抽象基类、DatasetRegistry 注册机制、mutation 算子表驱动均良好 |
| 可测试性 | 5 | 大量业务逻辑嵌入类内部；测试仍用合成数据；缺乏 Mock 边界 |
| 配置管理 | 8 | pydantic-settings 统一配置，设计规范 |
| **综合** | **6.7** | 核心引擎质量高，冗余和数据断路是主要减分项 |

---

## 6. 工作流完整性

### 6.1 已完整实现的工作流

```
✅ DSL 解析工作流:
   文本 → Lark → TypedNode → AlphaValidator → Executor(memoize) → pd.DataFrame

✅ GP 进化工作流（PopulationEvolver）:
   种子DSL → 种群初始化 → IS/OOS 回测评估 → 多目标适应度 →
   自适应变异（11种算子）→ AlphaPool 相关性过滤 → Optuna 参数调优

✅ 真实感回测工作流（RealisticBacktester）:
   DSL → 信号生成 → 4步管道（截断/衰减/中性化/延迟）→ Burn-in裁剪 →
   组合权重 → ADV截断 → 逐日PnL → 交易成本（sqrt冲击+ADV+借券）→
   RiskReport（IS+OOS 退化对比）

✅ 数据管道工作流（DataManager）:
   FeatureStore缓存检查 → Provider降级链 → Schema强制 →
   面板对齐 → 预处理 → 健康检查 → 写回缓存
```

### 6.2 断裂的工作流

```
❌ 真实数据端到端工作流：
   DataManager.get_panel()       ─┐
   DatasetRegistry.load()        ─┤  有完整实现，但从未被调用
   多市场数据集（10个）          ─┘
         ↓
   ??? 没有连接到 GenerationWorkflow / PopulationEvolver / main.py

❌ 因子生命周期工作流：
   AlphaStore（有 status 字段）
   → [缺] 在线 IC 监控
   → [缺] 衰减检测算法
   → [缺] 自动再训练触发
   → [缺] 淘汰规则

❌ 持续在线研究工作流：
   FastAPI 服务器（已运行）
   → [缺] 调度引擎（无定时触发）
   → [缺] 实盘数据反馈回路
   → [缺] Alpha 状态实时仪表板

❌ 多Alpha组合工作流：
   AlphaPool（积累低相关候选）
   → [缺] 联合权重优化（将多个 Alpha 合并成一个组合）
```

---

## 7. 金融业务深度分析

### 7.1 系统定位与真实价值

当前系统是**因子研究自动化平台的高质量原型**，具备以下真实工程价值：

- 完整的 DSL 因子描述语言（技术量价信号，语法严格，执行高效）
- 合理的 GP 进化架构（多目标适应度 + 多样性控制 + 代理模型早期剪枝）
- 基本可用的成本模型（sqrt 冲击法则 + ADV 约束 + 借券成本）
- 严格的 IS/OOS 物理分区防止过拟合
- 完整的多市场数据接入框架（尚未激活）

### 7.2 核心金融问题汇总

| 问题 | 严重性 | 影响范围 |
|------|--------|----------|
| 所有回测用合成随机游走数据 | 致命 | 全部结论不可信 |
| 行业中性化使用虚假行业编号 | 极高 | 截面 Alpha 信号失真 |
| 无系统性风险因子控制 | 高 | 组合暴露于 Beta/行业/规模风险 |
| 无 walk-forward 多轮验证 | 高 | 单次切分结论依赖切点位置 |
| 无 embargo/purged CV | 高 | 存在标签泄漏风险 |
| 无 Alpha 衰减监控 | 高 | 无法检测 live 信号失效 |
| GP 默认规模极小（20只股×120天）| 中 | 搜索结论统计意义不足 |

---

## 8. 15点实现程度详评

> 评分：0%=完全缺失，100%=生产级实现

---

### 点1：真实多市场数据 + Walk-Forward + Regime 验证

**实现度：28%**

**已实现：**
- `dataset_registry.py`：10 个数据集（美股 4 行业 / 中国 A 股 3 板块 / 港股 / 加密 2 类）
- `DataManager`：8 步完整数据管道（缓存→获取→Schema→预处理→健康检查）
- `multi_dataset_backtester.py`：跨数据集聚合 OOS Sharpe（mean / min 模式）
- 数据提供者：yfinance / akshare / ccxt_binance

**缺失（关键）：**
- **真实数据从未进入主优化循环**：`main.py` 全路径调用 `_make_synthetic_dataset()`
- Walk-Forward 是**单次固定切分（70/30）**，不是滚动窗口（正确：前 N 年 IS → 第 N+1 年 OOS，滚动 5+ 轮）
- 无 Regime 集成到验证流程（市场状态不影响 IS/OOS 评估策略）
- 数据可追溯性受限（DatasetSpec 默认 start=2021-01-01，历史深度不足 5 年）

---

### 点2：DSL 升级为金融结构表达语言

**实现度：20%**

**已实现：**
- 条件节点：`if_else`、`trade_when`（状态门控）
- 截面分组：`group_rank`、`group_zscore`、`ind_neutralize`（接口存在，但数据为假）
- `FinancialInterpreter`：8 类因子家族分类 + 自然语言描述
- `FinancialDiagnostics`：设计问题检测与改进建议

**缺失：**
- **行业数据为虚构**：`groups = np.arange(N) % 10`，无法做真实行业中性化
- 无资金流节点（北向资金、大单净买入、融资融券余额）
- 无事件驱动节点（财报发布、分红除权、指数调整）
- 无宏观状态输入（利率环境、信用利差、经济周期）
- 无 Beta 暴露约束节点
- DSL 表达能力限于技术量价，无法表达"行业暴露"、"因子载荷"等金融结构

---

### 点3：截面 Alpha 框架（Cross-Sectional Ranking）

**实现度：38%**

**已实现：**
- `cs_rank`、`cs_zscore`、`cs_demean` 截面算子正确实现
- IC / IC-IR 使用向量化 Spearman Rank IC（截面预测能力）
- `DecilePortfolio`：按截面分位做多空（正确的 CS alpha 框架结构）

**缺失：**
- 无正式 Alpha 组合框架（Barra IC 加权合成多因子）
- AlphaPool 存储多候选，但没有将其合并成截面得分的步骤
- 无截面协方差结构建模（因子间相关性的正式管理）
- 无多期预测能力分析（IC 随持有期的 decay 曲线）

---

### 点4：完整风险模型

**实现度：8%**

**已实现：**
- 市场中性权重（`NeutralizationLayer.market_neutral()`，权重求和为零）
- VaR (95%)、CVaR、最大回撤、Sortino Ratio
- 多空腿分解（`long_returns` / `short_returns`）
- Alpha/Beta 基准分解（需外部提供基准序列）

**缺失：**
- 无 Beta 因子风险模型（无法量化市场暴露）
- 无行业因子风险模型（Barra USE4 / CNE6 风格）
- 无规模/价值/动量/波动因子的暴露分解
- 无协方差矩阵估计（Ledoit-Wolf 收缩估计、DCC-GARCH）
- 无风险预算框架（Risk Parity / Marginal Contribution to Risk）

---

### 点5：容量/换手/滑点/冲击成本后净收益最大化

**实现度：42%**

**已实现：**
- 适应度含换手惩罚：`fitness = sharpe_oos - 0.2*turnover - 0.3*|maxDD| - 0.5*overfit`
- `TransactionCostEngine`：平方根冲击法则
- ADV 10% 上限截断
- 借券年化成本日化扣除
- `RiskReport.cost_drag_bps` 量化成本拖累

**缺失：**
- **主优化目标仍是 Sharpe，非净收益**：`gp_engine._evaluate_individual()` 用 IC-IR 代理，不调用成本引擎
- 无策略容量分析（目标 AUM 下的预期净 Sharpe 曲线）
- 无换手上限约束（优化器可接受高换手以换取高 Sharpe）
- 成本参数全局固定，未针对不同流动性分层设置

---

### 点6：因子正交化与 Alpha Pool

**实现度：32%**

**已实现：**
- `AlphaPool`：基于信号向量相关系数的多样性过滤（|ρ| < 0.90）
- GP 多样性惩罚（HoF 中已有 DSL 的个体 fitness × 0.5）
- `population_diagnostics()`：池内均值指标监控

**缺失：**
- 无正式正交化（PCA / Gram-Schmidt 在信号空间的投影）
- AlphaPool 积累的多个 Alpha **从未被合并成联合投资组合**
- 无 IC 加权合成（按预测能力动态加权各因子）
- 相关阈值 0.90 过高（α=0.70 的因子在真实组合中已可带来多样化收益）

---

### 点7：市场状态识别 + 动态切换 Alpha

**实现度：12%**

**已实现：**
- DSL 种子库含 "conditional" 家族：`trade_when(close > ts_mean(close, 60), ...)`
- `fitness.py`：因子家族偏置变异权重（momentum / reversion / volatility 等家族有不同的算子偏好）

**缺失：**
- 无显式 Regime 识别模型（HMM、k-means、Trend-Cycle 分解）
- 无 Regime 标签数据集
- 无基于 Regime 的动态 Alpha 切换逻辑
- `trade_when(close > ts_mean(close, 60))` 是简单阈值门控，不是真正的状态识别

---

### 点8：GP/LLM 基于金融语义的结构搜索

**实现度：18%**

**已实现：**
- `FinancialInterpreter`：DSL → 金融语义描述 + 因子家族分类
- `FinancialDiagnostics`：识别设计缺陷并生成改进建议
- LLM 系统 prompt 要求基于假设生成 DSL
- 因子家族引导变异（`factor_family` 偏置变异算子选择概率）

**缺失：**
- LLM 仅做**语法生成**，不做金融推理（无法推断"高利率环境下动量窗口应缩短"）
- GP 搜索是纯语法树变异，无金融语义约束
- 无 Chain-of-Thought 因子推理（经济逻辑 → 数学结构 → DSL 的演绎路径）
- LLM 无法读取历史回测结果来调整假设方向
- 无因子成因分析（为什么该 DSL 有效的经济学解释）

---

### 点9：真实执行层

**实现度：33%**

**已实现：**
- `TransactionCostEngine`：逐笔交易记录 + 平方根冲击模型
- ADV 参与率约束（`adv_cap_pct = 0.10`，10% ADV 上限）
- 买卖价差成本、固定佣金、空头借券成本
- `TradeRecord`：方向 / 股数 / 滑点 / 净价格完整记录

**缺失：**
- 所有执行为**收盘价 EOD 成交**，无日内执行模拟
- 无撮合引擎（无部分成交、无市价/限价区分）
- 无成交延迟模型（T+0 决策 → T+1 执行的时间价值损失）
- 无市场冲击持续性（临时冲击 vs 永久冲击的区分）
- 成本参数全局固定，未区分大盘股与小盘股

---

### 点10：因子生命周期管理

**实现度：5%**

**已实现：**
- `AlphaStore.status` 字段（"active" / "retired"）
- Alpha 记录含 IC-IR、Sharpe、换手、假设、时间戳

**缺失：**
- 无在线监控：上线后的 Alpha 无日常 IC 追踪
- 无衰减检测算法（滚动 IC-IR 下穿阈值触发警报）
- 无自动再训练触发器
- 无淘汰流程（从组合中移除衰减因子的规则）
- 无版本控制（同一假设下的迭代历史）
- 无部署日志（从研究 → 纸交易 → 实盘的状态流转）

---

### 点11：持续在线研究平台

**实现度：10%**

**已实现：**
- FastAPI 服务器（可 24/7 运行）
- 聊天接口（SSE 流式输出）
- Alpha 记录持久化（SQLite）
- Workflow A/B 对应 REST endpoint

**缺失：**
- 无调度引擎（Celery Beat / APScheduler）
- 无市场收盘后自动触发因子发现
- 无实盘数据接入 → 信号计算 → 持仓更新的实时管道
- 无 live P&L 反馈回路到 GP 进化
- 无 Alpha 状态实时仪表板

---

### 点12：另类数据

**实现度：0%**

所有数据提供者（yfinance / akshare / ccxt）仅提供 OHLCV + 衍生字段。

**全部缺失：**
- 新闻 / 情感分析（NLP 因子）
- 财报数据（EPS 超预期、盈利修正动量）
- 资金流数据（北向资金、ETF 申赎、融资买入）
- 宏观经济指标（PMI、CPI、利率曲线）
- 期权市场数据（隐含波动率面、Put/Call 比率）
- 链上数据（交易所流入/流出、持仓分布）

---

### 点13：元学习系统

**实现度：14%**

**已实现：**
- `ProxyModel`（XGBoost）：学习 AST 特征向量 → 预测评估失败概率（早期剪枝）
- `mutation_weights_from_metrics()`：基于种群诊断自适应调整变异算子权重
- 因子家族偏置变异策略（不同因子类型的变异算子偏好不同）

**缺失：**
- ProxyModel 学习的是**语法结构是否失败**，不是"何种市场环境下何类结构有效"
- 无跨市场 / 跨时期的迁移学习
- 无条件元学习（给定当前 Regime，推荐哪类因子）
- 冷启动需要 50 个样本（`COLD_START_THRESHOLD = 50`），在小规模 GP 中效果有限

---

### 点14：严格防数据泄露与防过拟合体系

**实现度：38%**

**已实现：**
- `PartitionedDataset.__slots__`：物理分区，IS/OOS 不共享引用
- Optuna 仅在 IS 数据上优化参数（OOS 严格隔离）
- 过拟合惩罚项：`0.5 * max(0, sharpe_is - sharpe_oos)`
- `AlphaValidator`：防 look-ahead bias 的语法规则

**缺失：**
- **无 Embargo Period**（IS/OOS 切割点无间隔窗口，事件泄漏风险）
- **无 Purged Cross-Validation**（Lopez de Prado：处理样本重叠的正确 CV）
- **无嵌套 Walk-Forward**（外层调参，内层验证）
- **无多重检验校正**（Bonferroni / BHY 调整 p 值）
- **无 Deflated Sharpe Ratio**（调整候选数量导致的 SR 膨胀）
- OOS 是**单次固定切分**（70/30），结论受切点位置影响大
- AlphaPool 中多个候选共用同一 OOS 窗口（多重假设检验问题）

---

### 点15：组合构建系统

**实现度：28%**

**已实现：**
- `SignalWeightedPortfolio`：截面 z-score 权重（`clip_z=3.0`）
- `DecilePortfolio`：Top/Bottom 分位做多空
- `NeutralizationLayer.market_neutral()`：权重求和为零的市场中性约束
- 单资产权重上限（`max_single_weight`，F11 修复）
- ADV 流动性截断（10% ADV 上限）

**缺失：**
- 无均值-方差优化（Markowitz / Black-Litterman）
- 无风险平价（各资产等风险贡献）
- **无多 Alpha 联合组合**：AlphaPool 中的候选从未被合并成联合组合（最优权重分配）
- 无因子暴露约束（如：组合 Beta < 0.1，行业偏差 < 5%）
- 无换手约束优化（从当前持仓到目标持仓的最小化换手路径）

---

## 9. 距离可赚钱系统的差距

### 9.1 综合评分矩阵

```
┌──────────────────────────────┬──────────┬──────────────────┬────────────────┐
│  维度                        │ 当前评分 │ 最低可交易门槛   │ 稳定盈利水平   │
├──────────────────────────────┼──────────┼──────────────────┼────────────────┤
│ 数据质量与覆盖度             │   28%    │      70%         │     88%        │
│ Alpha 信号质量               │   25%    │      60%         │     80%        │
│ 风险模型完整性               │    8%    │      70%         │     90%        │
│ 执行成本真实性               │   33%    │      65%         │     85%        │
│ 防过拟合严格性               │   38%    │      75%         │     92%        │
│ 组合构建完整性               │   28%    │      65%         │     85%        │
│ 运营监控系统                 │    5%    │      60%         │     80%        │
├──────────────────────────────┼──────────┼──────────────────┼────────────────┤
│ 综合评分                     │   24%    │      66%         │     86%        │
└──────────────────────────────┴──────────┴──────────────────┴────────────────┘
```

### 9.2 必须跨越的三级鸿沟

**第一级（系统性缺陷——不修复则所有结论无效）**

1. **真实数据接入**：将 `load_dataset()` / `DataManager` 接入 `GenerationWorkflow`，替换所有 `_make_synthetic_dataset()` 调用（约 10 行代码修改 + 参数传递）
2. **真实行业分组**：接入 GICS / 中证行业分类，替换 `np.arange(N) % 10` 假分组
3. **Walk-Forward 多轮验证**：将单次切分升级为滚动 5+ 轮验证

**第二级（金融可信度——有实盘意义的结论依赖这些）**

4. **基本风险模型**：至少加入市场 Beta 中性化和行业暴露跟踪
5. **多 Alpha 联合组合**：实现 AlphaPool → 联合权重优化流程
6. **Embargo + Purged CV**：IS/OOS 切分点加 20 天缓冲期

**第三级（离赚钱最近的增量改进）**

7. **Alpha 在线监控**：上线后的 IC 追踪 + 衰减预警
8. **净收益驱动优化**：将净 Sharpe 作为整个链条的优化目标
9. **策略容量估算**：在真实 ADV 下的规模上限分析

---

## 10. 优先级改进建议

### P0（2天内，代码接线工作，不需要新功能）

```python
# main.py — 接入真实数据（约10行改动）
from app.core.data_engine.dataset_registry import load_registry_dataset

def run_agent(args):
    ds = load_registry_dataset("us_tech_large", start="2020-01-01")
    dataset = ds.data          # dict[field → pd.DataFrame]
    store = AlphaStore()
    agent = AlphaAgent(store=store)
    log = agent.run(hypothesis=args.hypothesis, dataset=dataset)
    ...

# dsl_executor.py — 不再自动生成假行业分组
# 删除 _add_derived() 中 groups 相关代码，改为在调用方传入真实行业数据
```

### P1（1-2 周，清理冗余）

- 删除 `app/core/portfolio_engine/`（约 3 文件，完全冗余）
- 删除 `app/core/optimization_engine/`（约 3 文件，完全冗余）
- 将 `main.py run_gp()` 迁移到 `PopulationEvolver`
- 统一 import 风格（全部改为绝对 import 或全部改为相对 import）

### P2（1 个月，核心金融功能）

- 实现 Walk-Forward 多轮验证框架（至少 5 轮，基于真实数据）
- 接入 GICS 行业分类到 `groups` 字段
- Embargo Period（切分点后加 20 交易日缓冲）
- AlphaPool → 多 Alpha 联合权重优化（IC 加权 + 协方差调整）

### P3（2-3 个月，系统深度）

- 市场 Regime 识别模型（隐马尔可夫 / 趋势强度）
- 因子衰减监控仪表板（每日自动计算 IC）
- Purged Cross-Validation 替代固定切分
- 策略容量估算工具

---

*报告生成：2026-06-01 | 版本 v1.0 | 覆盖文件数：85个 .py 文件*
