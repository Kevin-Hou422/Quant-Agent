# 终极目标路线图：美股实盘全链路自主模拟交易

> 状态：**规划（Phase 8 已完成，Phase 9–14 + R 未实现）** · 地基 Phase 6–8 见
> `backend_retired_report/PAPER_TRADING_ROADMAP.md`（已归档）· 遵循 RESEARCH_OPERATING_MODEL.md
> 生成日期：2026-08-18

## 终极目标

agent **自主从市场观察挖掘因子 → 配置持仓 → 在美股（moomoo 纸交易）上全链路模拟真实交易**，
默认模式下**用户只批准/拒绝**，paper 阶段支持全自动无人值守；建模 paper 与真实交易的差别以
提升实验可信度。原有用户假设驱动的功能全部保留。

## Context（为什么做这件事）

当前系统（Phase 0–7 完成）已具备：GP 因子进化、真实感回测（PaperBroker 与回测逐位对账
2.2e-16）、生命周期状态机、每日交易循环、Live 仪表板、可复现台账。**但要达到终极目标，
现有代码有四个根本性缺口**（两个 Explore 代码探查确认）：

1. **发现不自主**：所有发现路径都从用户假设/DSL 起步（`chat`/`GenerationWorkflow`/
   `POST /api/workflow/generate` 都要 hypothesis 字符串）。没有"从市场观察生成假设"的引擎，
   没有定时自动发现任务。
2. **生命周期形同虚设**：晋级 CANDIDATE→…→ACTIVE 全靠手动 PATCH；唯一自动流转是
   ACTIVE→DECAYING（降级）。新因子被直接写成 `status="active"`（`router.py:1906`、
   `alpha_agent.py:121`），跳过分级。无验证门、无批准/拒绝、无红队。
3. **数据单薄**：schema 仅 7 个 OHLCV 派生字段（+ 派生 sector），无任何另类数据；所有
   provider 都是日频历史批量拉取，无实时/增量；无 PIT 数据版本存储。
4. **无真实执行**：全代码库零下单代码；PaperBroker 用**当日收盘**成交（非 T+1 开盘），
   市场冲击仅建模"我们自己交易的参与率"，无真实撮合。

**已在 Phase 8 规划（本路线复用、不重复）**：8.1 PIT 追加存储、8.2 成本校准回路、
8.3 运营规则+红队概念文档。本路线把 Phase 8 当作地基，在其上补齐 Phase 9–14。

## 当前数据与因子存储现状（本路线的改造基线）

- **数据库**：SQLite（SQLAlchemy ORM），已加固 WAL + busy_timeout（`_sqlite_utils.py`）。
  - 主库 `backend/alphas.db`（`settings.database_url = sqlite:///./alphas.db`，`config.py:23`）
    表：`alpha_records`（因子）、`alpha_ic_history`（逐日 realized IC）、`chat_sessions`、
    `chat_messages`。paper 交易的 `paper_positions/fills/daily_pnl`、`run_manifest` 也在同库。
  - 调度库 `scheduler_jobs.db`（APScheduler SQLAlchemyJobStore，`config.py:55`）。
- **因子存放/管理**：全部在 `alpha_records` 表，经 `AlphaStore`（`app/db/alpha_store.py`）
  统一读写——`save/query/export_csv/update_status`。字段：`id, dsl, hypothesis, ann_return,
  sharpe, max_drawdown, ic_ir, ann_turnover, status, reasoning(JSON)`。生命周期状态机
  （`alpha_lifecycle.py`）约束 `status` 流转。**append-only 台账**：新增不改写。
- **改造影响**：Phase 8.1 的 PIT 数据存储、Phase 9 的分级生命周期修正、Phase 10 的另类数据、
  Phase 12 的 moomoo 执行都会在此 SQLite 上扩表（生产可平滑迁 Postgres，URL 已可配）。

## 已确认的关键决策（4 个分叉）

| 分叉 | 决策 | 含义 |
|------|------|------|
| 执行落地 | **moomoo 纸交易账户**（2026-08 更新，原定 Alpaca）| 免费、美股、真实市场价撮合；**与研究数据同源(TR.2)消除 skew**；需建确定性执行工作流；LLM 不进交易回路 |
| 另类数据预算 | **先免费起步，后付费** | 先 yfinance 基本面/免费财报日历跑通 PIT+DSL 稀疏字段，验证后再付费源 |
| 实时粒度 | **日频收盘增量** | 收盘后拉当日 bar 追加进 PIT，前向积累；不做日内/tick |
| 自主挖掘首版 | **先用价量统计** | 用 regime/截面离散度/因子家族滚动表现自主生成假设，现在就能做 |

## 贯穿全程的铁律（来自 RESEARCH_OPERATING_MODEL.md，不可违反）

- **LLM 永不进交易回路**：每日信号/下单/风控是确定性代码；LLM 只在研究侧（假设生成、
  代码、报告解读、红队审查）。
- **状态流转永不作为 LLM 工具**：晋级走人工 approve/PATCH 端点；`tool_save_alpha` 是唯一
  可接受的写（只增台账，无资金后果）。Phase 6 起新工具几乎全只读。
- **另类数据必须带发布时点进 PIT**（报告期 ≠ 可见期，否则前视泄漏）。
- **多 agent 仅按触发条件引入**（目标对立/信息隔离/时间尺度不同）；红队是最早有回报的角色。
- 保留原有用户假设 Workflow A/B（加法，不替换）。

> **⚠️ 方法论警示（外部评估 2026-08-22 核实）**：当前所有回测数字**不是"偏乐观"，而是"方向不明"**——
> 幸存者偏差 + OOS 被选择性挖掘的联合效应，符号与大小都无法估计（可能把 Sharpe −0.2 显示成 1.5）。
> **在完成下面的 Phase S 前，不应对任何回测数字做任何决策解读**（含 paper 期的历史回放段）。

---

## Phase S — 数据与统计地基修正（**P0 · 阻塞项 · 最高优先级，先于除 PM 外一切**）

**为什么**：外部评估（逐行核实无误）指出三个致命问题，使目前**所有回测数字失去决策意义**。这些不是
"补强"，是"不修则一切数字无效"。工程和治理已达机构级，短板在**数据真实性 + 统计诚实性**。

- **S.1 去掉 fitness 的 OOS 选择 ✅（发现路径，2026-08-24）** —— 现状 `gp_engine/fitness.py`
  `fitness = sharpe_oos − …` **直接按 OOS 择优 → OOS 退化为第二个样本内**。已改：`GenerationWorkflow`
  用三段切割，GP 只在 **Validate** 段做适应度选择，**Test 段全程不可见**（`_partition_three_way` +
  赢家在 Test 上汇报 `test_sharpe`/`held_out_test`）。**待补**：`/api/backtest/*` 直接回测路径、
  以及把适应度换成 IS 内部 purged K-fold CV（当前 Validate 是单段内部验证，已足以去除循环论证）。
- **S.2 全路径强制真 holdout ◑（发现路径已做，2026-08-24）** —— 已给 `GenerationWorkflow` 加真
  held-out Test（GP 不可见、仅汇报）。**待补**：`/api/backtest/*`、`ValidationGate` 也走三段；test 段
  改**最近 2–3 年冻结、一次性使用**；`RunManifest` 记"该 test 段已用 N 次"超阈值告警。
- **S.3 全局多重检验计数器 ◑（2026-08-24）**—— 已建 `db/trial_ledger.py`（`TrialLedger`：跨会话持久
  累计 trial 数）；发现每轮 GP 把 `pop×gen+optuna` 累加进去；`ValidationGate` 默认用**全局累计数**做
  DSR 去膨胀（不再默认 1）+ 新增 **t≥3.0**（Harvey-Liu-Zhu，Lo 2002 t 统计量）门槛。PBO 函数
  `backtest_engine/overfit_stats.py`（CSCV，Bailey 2015）**已接进 StrategyGate（2026-08-30）**：候选各
  因子单因子策略收益构成矩阵 → PBO；`PBO>阈值(默认0.5)` 判"选择流程过拟合"，落进策略 verdict（配置
  `pbo_threshold/pbo_n_splits`）。**待补**：CPCV。（吸收原 Phase R.1。）
- **S.4 换含退市收益的价格数据源**（根治历史幸存者偏差）—— **🚫 已搁置（2026-08-25，被 TR.2 取代）**。
  原计划接付费源（Sharadar/CRSP）修历史幸存者偏差，但用户目标是**零预算的前向 paper 模拟**：
  ① 幸存者偏差只影响**历史回测**，**前向 paper 天生无此偏差且免费**（前向 paper 会自纠幸存者伪因子）；
  ② 单一权威源统一到 **moomoo（TR.2）**，退市股虽仍无，但不在"免费前向"的关键路径上。**故不做 S.4。**

**依赖/次序**：S 是**所有回测结论的前置**；与 Phase PM（组合层）并列为最高优先级——PM 补"操作"轴，
S 补"数字可信"轴。（S.4 已搁置，S.1/S.2/S.3 为本层实质内容。）

---

## Phase 8 —（✅ 已完成 2026-08-19，作地基）PIT 数据层与验证运营

**这是 Phase 9/10/11 的前置，现已就绪。** 实现：`app/core/data_engine/pit_store.py`（`PITStore`
双时点 `(field,date,as_of)` 只增不改 + `load_pit()`）、`app/tasks/cost_calibration.py`（8.2 成本
校准回路，T+1 开盘价对账，建议不自动改 `CostParams`）、`backend/OPERATIONS.md`（8.3 运营规则）。
已接入 `daily_ingest`（通过健康门的数据按 as_of 追加）。新增 16 项测试，全量回归绿。

---

## Phase 9 — 自主发现引擎 + 分级生命周期门 + 人工批准（**默认模式核心**）

**目标**：agent 用价量观察自主挖因子 → 走分级生命周期 → 自动验证门 → **默认由人批准/拒绝**。

> **进度（2026-08-22）**：Phase 9 **全部完成**（9.1/9.2/9.3/9.4 + FE-9）。新增 22 项后端测试
> （observer 4 + discovery 4 + validation_gate 7 + approval 7），前端 tsc 干净、94 项通过。

- **9.1 市场观察引擎 ✅** — 已建 `app/core/discovery/market_observer.py`（`MarketObserver`）：从 OHLCV
  算 regime（复用 `RegimeDetector`）+ 截面离散度 + 动量/反转广度 + 短期反转自相关 + 波动水平 →
  输出**按分数排序的结构化"假设方向"**（6 个有效 GP 家族，非用户文本）。确定性可复现。
- **9.2 自主发现编排器 ✅** — 已建 `app/core/discovery/discovery_engine.py`（`DiscoveryEngine`）：
  观察 → top 家族名（数据驱动）驱动 `GenerationWorkflow`（复用）→ 本轮 DSL 去重 → 赢家存为
  **CANDIDATE**；`scheduler.py` 注册 `nightly_discovery_job`（`ENABLE_DISCOVERY=true`，每晚 23:00）。
  LLM 不参与，可无 key 运行。
- **9.3 自动验证门 ✅（2026-08-20）** — 已建 `app/core/lifecycle/validation_gate.py`（`ValidationGate`：
  WalkForward 全折 OOS>0 via `min_oos_sharpe` + DSR>0.90，真实数据，**fail-closed**）；
  端点 `POST /api/alphas/{id}/validate`（通过则状态机 CANDIDATE→VALIDATED）；创建路径全部改为
  CANDIDATE 起步（`AlphaResult` 默认 + router 5 处 + alpha_agent 1 处），并同步修 test_phase5 相应用例。
  新增 `test_phase9_validation_gate.py`（7 项）。
- **9.4 人工批准/拒绝工作流（默认）✅** — 已加 `POST /api/alphas/{id}/approve`（VALIDATED→PAPER）
  + `/reject`（CANDIDATE/VALIDATED→RETIRED 带原因）+ `GET /alphas/pending`（待批准队列）+
  `GET /alphas/{id}/decisions`（审批谱系）；新增 `alpha_decisions` append-only 表；配置
  `autonomy_mode: manual|auto`（默认 manual；auto 见 Phase 13）。**FE-9**：`ApprovalQueue.tsx`
  挂到 `AlphaDashboard`，批准/拒绝按钮 + 谱系；`client.ts` 加 pending/approve/reject/decisions/validate。
- **验收 ✅**：`DiscoveryEngine.run(dataset)` 零用户文本产出 CANDIDATE；验证门端点升 VALIDATED；
  approve 升 PAPER；除批准点外全程无人值守。原 Workflow A/B 仍可用。

> **⚠️ 门控设计已演进（见 Phase PM「核心重构」，为最新设计）**：9.3/9.4 已实现的是**因子级**门/审批；
> 按"要交易策略、不要漂亮因子"的最新判断，**因子级门降级为"候选入池的低门槛滤泄漏"，严门 + 审批
> 上移到策略层**（PM.S1/S2/PM.7）。已建的 `ValidationGate`/approve 端点**复用**，只是评估/审批对象
> 从"单因子"换成"组合策略"。**统一门控政策见 Phase PM 核心重构块。**

---

## Phase PM — 组合与资金管理层（Portfolio Manager）**★ 离真实交易最大的缺口，优先于 10/11**

**为什么**：现状 paper trading 是"每个验证过的因子当成一个孤立、归一化(|w|=1)、美元中性的策略各跑各的"，
这是**研究级信号验证**，不是"一个真人 PM 在管一个真实账本"。前面的因子全链路模拟的是研究员的**思考**；
本层补的是真人的**操作**——像真人一样决定**每只股票具体投多少钱、选哪几只、每个因子配多少资本、
容量够不够、快慢因子怎么区别对待、组合风险怎么控**。这才落地终极目标里"**由因子到具体持仓配置**"。

**目标**：在"validated/paper 因子"与"paper 执行"之间插入一个独立的 **PM 层**——把 N 个 alpha 合成一个
**真实、美元计价、容量感知、风险管理的组合账本**；**审批对象从"因子"升级为"组合级持仓配置"**。

### 第一批（PM 核心：把孤立因子变成一个配资的美元账本）✅（2026-08-24）

> 已建 `app/core/portfolio_manager/manager.py`（`PortfolioManager`）：PM.1 合成+净持仓、PM.2 容量
> （AUM 越大越 binding，water-filling）、PM.3 AUM→具体美元/股数账本；PM.4 `DailyTradingLoop.run_portfolio`
> 把全部 paper 因子合成**一个组合账本**（book id=0）交易，接入 `run_daily_pipeline`；配置 `paper_aum`。
> 新增 `test_phase_pm.py`(5)。全免费、不含执行侧（留 Phase 12）。

- **PM.1 多因子合成** — 新建 `app/core/portfolio_manager/combiner.py`（**接入 live 路径**，复用现有
  `AlphaCombiner`：IC-IR 加权 / 等权 / 最小方差）：把多个 paper 因子的信号合成**一个**组合信号，
  而非各跑各的。权重 IS 拟合、标注 `weights_fitted_on`（沿用 B2 修复口径）。
- **PM.2 容量建模** — 落地 `app/core/backtest_engine/capacity.py`（= Phase R.3）：每个因子的
  Sharpe/收益随 AUM 衰减曲线（复用 `TransactionCostEngine` 的 ADV 参与率/冲击）→ 给出每因子**容量上限**。
- **PM.3 资本配置** — 新建 `app/core/portfolio_manager/allocator.py`：引入**真实 AUM** 概念，按
  风险预算 / 容量(PM.2) / 相关性分散，给每个因子分配美元资金；产出**每只股票的具体美元/股数持仓**
  （不再是归一化 |w|=1）。
- **PM.4 接入 paper 循环** — 改 `daily_trading_loop.py` / `PaperBroker`：交易的是 PM 产出的**组合美元
  账本**（单一 book + 真实 AUM），而非每因子独立归一化 book。`PositionStore` 记组合级持仓/PnL/敞口。
- **验收**：给定 AUM + 几个 paper 因子，PM 产出一份**具体美元持仓**（AAPL 多 $X、XYZ 空 $Y…）；
  单因子超容量时被封顶；组合 book 的 paper PnL 与逐因子加权一致对账。

### ★ 核心重构：策略级门控 + 边际准入 + 经典基准库（**要"交易策略"，不要"回测漂亮的因子"**）

**问题**：现状因子**独立过门**（每个因子单独 DSR>0.9/t≥3），幸存者才进 PM 合成。这是错的——因子的
价值是它对**策略**的**边际贡献**（边际 IR ∝ √(1−ρ²)），**独立高门槛恰恰筛掉最有用的低相关分散化因子**，
选出的是"单独漂亮的因子"而非"好策略"，且常常"没有因子能过 → 无可交易"。**验证单位应是策略，不是因子。**

- **PM.S1 门控搬到策略层 ✅**（2026-08-27）— 新建 `app/core/portfolio_manager/strategy_gate.py`
  的 `StrategyGate`：把 N 因子经 PortfolioManager 合成**一个组合策略** → 对**策略净收益**（复用
  BacktestEngine 同一成本引擎）跑严门——**分段 OOS 全为正 + DSR>0.90（S.3 全局 trial 去膨胀）+
  夏普 t≥3.0**，fail-closed。严门只加在真正交易的策略上，不加在单因子。`strategy_net_returns()` 复用。
- **PM.S2 因子按边际贡献准入 ✅**（2026-08-27）— `marginal_factor_selection()`：贪心前向选择，候选
  加入策略后**OOS-尾段 Sharpe 提升 ≥ min_improve** 才纳入，否则拒。**与已选高相关的冗余因子被拒
  （边际≈0）、单独弱但分散化好的因子能进**；每步决策留可审计轨迹。测试 `test_phase_pm_s.py`(7)。
  **已接线（2026-08-30）**：`run_portfolio` 组合前用 `marginal_factor_selection` 选因子（PM.S2）、
  组合后 `StrategyGate.evaluate` 出策略 verdict（PM.S1，配置 `pm_strategy_gate_eval/block`）；
  `selection/strategy_verdict` 落进返回值。测试 `test_phase_pm_wiring.py`(3)。发现→审批路径的
  策略级晋级仍属 PM.7。
- **PM.S3 经典基准策略库 ✅**（2026-08-27）— `app/core/strategies/baselines.py`：8 个经 DSL 解析/执行
  验证的经典异象——截面动量 12-1(J&T 1993)、12M、短期反转(Jegadeesh 1990)、低波(Ang 2006)、
  52 周高(George-Hwang 2004)、时序趋势(MOP 2012)、流动性(Amihud 2002)、特异偏度(Boyer 2010)。
  `baseline_signals()` 可直接喂 PortfolioManager；`seed_baselines()` 幂等种成 CANDIDATE 走正常门控。
  **交易兜底已接线**：`daily_trading_loop.run_portfolio` 在无 PAPER/ACTIVE 自研因子时**回退基准库**
  照常组账本交易（`used_baseline=True`）→ 回答"门控下没因子怎么 trade"。测试 `test_phase_pm_s3.py`(7)。
  （价值因子需基本面 → Phase 10；纯价量只能用长期反转作代理。）
- **联动 Phase 9**：Phase 9.3 的因子级验证门降级为"候选入池的低门槛滤泄漏"；晋级/审批的对象从"因子"
  升为"**策略/组合配置**"（呼应 PM.7）。**联动 Phase TR**：策略级验证用 TradingContext 的真实成本估计。
- **验收**：一个单独过不了门的因子，加入策略后能提升策略 OOS → 被纳入；经典动量 baseline 可作基准跑通。

#### ★ 统一门控政策（收敛 S.3 / Phase 9 / PM.S / TR.4 的重复与层级不一致）

**唯一的门控链**（严门只加在"你真正交易的策略"上，不加在单因子上）：

| 阶段 | 门槛 | 加在 |
|------|------|------|
| **1. 入池** | 低门槛：仅滤**泄漏/明显垃圾**（不看 Sharpe 高低） | 因子 |
| **2. 策略验证** | **严门**：WalkForward 全折 OOS>0 + **DSR>0.90（S.3 全局 trial 去膨胀）+ t≥3.0** + PBO | **组合策略**（PM.S1）|
| **3. 因子准入** | 边际贡献：加入后策略 OOS（扣真实成本 TR.3）提升才纳入 | 因子→策略（PM.S2）|
| **4. 进 PAPER** | **较松/分级**（收集前向证据，TR.4 实验模式；阈值配置化） | 策略 |
| **5. → ACTIVE（逼近真钱）** | **最严**：≥60 交易日**前向** realized IC 均值>0 且 **t>2** | 策略 |

- S.3 的 DSR/t≥3、原 Phase 9.3 的 WalkForward → 全部归到**第 2 步（策略级）**，不再加在单因子。
- TR.4 的"进 paper 松、真钱严 + 阈值配置化" = 第 4/5 步。审批对象 = **策略配置**（PM.7）。

### 第二批（PM 增强：风控 + horizon + 配置审批）

- **PM.5 组合级风控 ✅**（2026-08-30）— 已建 `app/core/portfolio_manager/risk_gate.py`
  （`PortfolioRiskGate`/`RiskLimits`）：gross/net 敞口上限、**单票/行业集中度**（NAV 比例绝对上限，
  只减不增）、**目标波动缩放**（按估计年化波动缩放整体仓位）、**回撤熔断**（`should_halt`）。
  `apply()` 产出合规账本、`check()` 只报违规（供 PM.7 审批一份配置）。测试 `test_phase_pm5_risk.py`(7)。
  **beta 中性**（顺带闭合 B6，见 Phase R.2）**留待开启做空后**——当前 long-only，net=gross，beta 对冲不适用。
  **已接线（2026-08-30）**：`run_portfolio` 在容量后、下单前用 `PortfolioRiskGate(RiskLimits(...))`
  施加到权重面板（配置 `risk_max_gross/name/sector/target_vol/max_drawdown`，long_only 随 `trading_allow_short`）；
  回撤熔断 `should_halt` 接入（配置 `risk_halt_on_drawdown`）；`risk_report/drawdown` 落进返回值。
  **实测生效**（集中持仓下单票被削）。测试 `test_phase_pm_wiring.py`。
- **PM.6 horizon 感知配置** — 按因子换手/持有期/衰减 horizon 把因子分**快/慢**两类，差异化仓位、
  成本处理与资本占用（快因子高换手→更严成本/更小容量占用；慢因子→更大配额）。数据来自 `AlphaMonitor`
  的 turnover/decay。
- **PM.7 审批对象升级** — 待批准队列从"批准因子"升级为**"批准一份组合级持仓配置"**（组合成分 + 每因子
  配额 + 目标敞口 + 风险预算）；保留因子级审批为其上游。谱系记录配置版本。
- **FE-PM** — 前端组合视图：当前组合的**具体美元持仓表**、各因子配额/容量占用、组合敞口/风险仪表、
  以及"批准这份配置"的界面（替代/叠加原因子审批）。
- **验收**：一份组合配置可被审阅+批准；风控门拦截超 gross/集中度的配置；快慢因子配额随其 turnover 区别。

### 依赖与收拢
- **吸收/前置**：Phase R.3（容量）、R.4（组合优化 Ledoit-Wolf/HRP/换手率进目标）、R.2 的 beta 闭合
  都被本层收拢进 **live 路径**（此前只是离线研究组件）。
- **下游**：Phase 12（**moomoo** 执行）消费 PM 产出的**美元账本**下单，而非单因子权重——PM 是"因子→
  真实下单"之间必需的一层。
- **复用**：`AlphaCombiner`、`TransactionCostEngine`/`LiquidityConstraint`、`portfolio_constructor`
  的 `beta_neutral`、`AlphaMonitor`（turnover/decay）、`PositionStore`。

---

## Phase TR — 交易现实引擎（Trading Reality）**★ 让"能否赚钱"的答案对散户真实**

**为什么**：财务关键参数(AUM/成本/做空/调仓/数据源)此前是**代码里的机构默认**，悄悄假设"能自由做空 +
机构级低成本 + 研究/执行同源"，会**扭曲"$10k 散户能否赚钱"的答案**（DEV_LESSONS §J）。本层让系统
**自己从真实情景 + 免费数据推导**这些，并把"只有交易当时才知道的"永不写死。

**三层纪律**：T1 现实事实（显式配置）· T2 数据推导估计（系统重算）· T3 交易当时才知道（走 provider）。

- **TR.1 TradingContext（T2 推导）✅（2026-08-25）** — 已建 `app/core/trading_context/`：
  Corwin-Schultz/Abdi-Ranaldi 从免费 H/L **估每名真实价差**；可做空性（long-only/账户/流动性启发式）；
  单边成本 = 半价差 + 券商费（`BrokerProfile`，moomoo 佣金免费）；**无交易带**（成本挂钩，**统一调仓策略：
  应驱动 PM 调仓、取代"每日全额调仓"，与 PM.6 horizon 合并**）；可交易池过滤。
  配置 `trading_account_type/allow_short/broker`（T1，显式）。新增 `test_phase_tr.py`(6)。
- **TR.2 单一权威数据源 = moomoo（消除 train/serve skew）✅**（2026-08-30）— 已建
  `app/core/data_engine/providers/moomoo_provider.py`（`MoomooProvider`）：经本地 **OpenD 网关**
  拉日 K（分页+节流+复权），产出与 Yahoo **逐字段一致**的 RawDataset；`fetch_latest` 供前向增量。
  接线：`config.price_source: yahoo|moomoo`（默认 yahoo，切 moomoo 后 **US 数据集价格改走 moomoo**，
  非美不受影响）+ `dataset_registry._fetch_moomoo`。**实测通**：真连 OpenD 拉 AAPL/MSFT 日K，价格与真实
  市场一致（含 WWDC 跳空）。测试 `test_phase_tr2_moomoo.py`(6，含真连冒烟；OpenD 未起自动 skip)。
  已核实用户账户：**美股 LV3 实时**、历史 K 线额度 100、端口 11111。**已切** `PRICE_SOURCE=moomoo`
  即全链路 discovery/validation/paper 同源。
  - **换源收尾适配（2026-08-30）**：① 股类符号归一化 `BRK-B → US.BRK.B`（实测 moomoo 用点号，
    yfinance 用横杠）；② registry 缓存键加 `price_source`（换源不返旧缓存）；③ `MoomooProvider`
    对单标的失败**跳过+记日志**，健康门兜底空列。

> **★ Universe 与额度说明（当前 $10k 阶段的显式选择，非永久上限）** — paper/discovery 固定用
> `us_broad_large`（**~95 只、覆盖 11 板块的流动大盘，≤100**），因为 moomoo 免费账户**月度历史K线额度=100 只**。
> **这是当前资金规模下的合理选择，不是架构上限**：① $10k 下实际可持仓仅 10~30 只，**约束来自 AUM 而非 universe**；
> ② 100 只流动大盘（≈标普100）足够表达/验证横截面策略，宽度对 IR 的边际在此已递减；③ 额度**每月重置、随资产增长**，
> 且**前向 paper 自积累数据、初始回填后不再依赖该额度**。
> **未来扩大资金规模时，需相应调整的模块**：
> - **universe 定义**（本 registry：换更大的可交易池，或分月/多源轮换）——直接决定宽度；
> - **数据源额度策略**（TR.2）：或研究用免费广源(yfinance)拉宽截面、moomoo 只管可交易子集+执行（代价：重新引入
>   一点 train/serve skew，需权衡）；或随入金提升 moomoo 额度；
> - **PM 容量模型**（PM.2/TradingContext）：AUM 变大后 ADV 容量约束开始 binding，须重估每名上限；
> - **成本口径**（TR.3）：更大 AUM 的市场冲击不再≈0，`grounded_cost_params` 的参与率/冲击项要重新生效；
> - **持仓名数/集中度门**（PM.5）：随可持仓名数上升放宽单票/行业集中度上限。
- **TR.3 成本落地推导值 ◑（2026-08-26）** — 已建 `grounded_cost_params(dataset, broker, aum)`：
  moomoo 佣金免费 → `fixed_bps=0/min_ticket=0`，`spread_bps`=**Corwin-Schultz 数据估计中位**（取代硬编码
  5/1/2）；`run_portfolio` 组合账本改用 grounded 成本 broker（配置 `trading_broker`）。**待补**：
  `QuoteProvider/BorrowProvider/AccountProvider` 接口（T3 实时，仿真背 TR.1 估计、实盘背 moomoo）——
  待 TR.2 moomoo 接入一并落。
- **TR.4 门分级 + 阈值配置化 ⬜** — 实现 **Phase PM「统一门控政策」的第 4/5 步**：进 PAPER 较松/分级
  （收集前向证据，"实验模式"），→ACTIVE 最严（≥60 交易日前向 realized IC t>2）；所有阈值提为配置
  （不写死）。（严门主体在策略级验证=第 2 步，见 PM.S1。）
- **依赖**：TR.3 落地后，PM 的成本/容量、PaperBroker 都吃 TradingContext 的真实估计；TR.2 是 Phase 12
  执行的前置（同源）。

---

## Phase 10 — 另类数据接入（免费起步）+ DSL 稀疏字段（依赖 Phase 8.1）

**目标**：用基本面/财报丰富发现，严格发布时点防前视，DSL 支持季频稀疏字段。
**数据源边界（与 TR.2 一致）**：**价格只来自 moomoo**（单一权威源）；基本面/财报是**另一类数据**，
可用另一免费源（yfinance `.quarterly_financials` / SEC EDGAR），**不造成价格 train/serve skew**，
但必须带发布 `as_of` 进 PIT（8.1）防前视。

- **10.1 DSL 稀疏字段支持** — 改 `typed_nodes.py`/`dsl_executor.py`：季频字段只在发布日后可见、
  日频面板 ffill、发布前 NaN（新字段类别）。这是 DSL 从"技术量价语言"升到"金融结构语言"的实质台阶。
- **10.2 基本面 provider（免费）** — yfinance `.quarterly_financials`/`.info`：E/P、盈利收益率、
  质量（ROE/毛利率），带发布 `as_of` → PIT 存储。
- **10.3 财报事件（免费日历）** — PEAD/SUE 特征、事件窗口标记。
- **10.4 新 DSL 算子** — `fund_rank`、盈利超预期等；观察引擎（9.1）+ GP 消费之。
- （付费源 FMP/Polygon 待免费验证后再上。）
- **验收**：一个基本面因子可解析+回测且**发布日测试无前视**；发现引擎能形成基本面假设。

---

## Phase 11 — 前向增量数据 + PIT 追加 + 市场日历

**目标**：从历史回放转为真正的前向 paper trading（日频收盘增量）。**价格源 = moomoo（TR.2）**，与执行同源。

- **11.1 provider 增量拉取** — **`MoomooProvider`（TR.2）**加 `fetch_latest`/增量接口 + 追加进 PIT（8.1）带 `as_of`。
- **11.2 美股日历 + 时区** — ET 收盘、节假日感知的调度（现有 job 用 UTC 固定时刻）。
- **11.3 `daily_ingest` 重构** — 真增量追加（非整段重拉）；从启动日起积累无幸存者偏差的自有数据。
- **验收**：连续 N 个真实交易日，PIT 每日增一根 bar；日循环消费增量；A1（无人值守日节奏）成立。

---

## Phase 12 — 真实执行层：**moomoo** 纸交易 + 确定性执行工作流 + 风控 + paper-vs-real 建模

**目标**：在 **moomoo**（= TR.2 的单一权威源/执行券商）纸交易上的完整**确定性**交易工作流；建模保真度差距。
**与 TR.2 同源**：moomoo 既是研究数据源也是执行券商，消除 train/serve skew。

- **12.1 moomoo 执行适配器** — 新建 `app/core/execution/moomoo_broker.py`：PM 美元账本→股数订单
  （`OrderManager`），经 **moomoo OpenAPI（OpenD 网关）**下单（**确定性代码，非 LLM**）、成交轮询、
  持仓对账（本地 vs 券商）、幂等（client order id）。复用/扩展 `PositionStore` schema。
- **12.2 风控门 + kill switch** — 新建 `app/core/execution/risk_gate.py`：单票/总敞口上限、
  日亏熔断、fat-finger（订单 vs ADV）、一键全平；违规订单被拦截。
- **12.3 paper-vs-real 保真度阶梯 + 校准（延伸 8.2）** — 对比 (a) 内部 PaperBroker 收盘成交
  vs (b) **moomoo 纸交易**真实成交 → 校准 `impact_coef`、建模 T+1 开盘成交、永久 vs 临时冲击。
  三级保真度：内部模拟 → **moomoo 纸交易** → （未来）**moomoo 实盘**。
- **12.4 崩溃恢复** — 重启时从 moomoo 持仓重建状态。
- **消费 PM 账本**：下单对象是 PM 产出的**美元账本**（策略级），而非单因子权重。
- **验收**：策略美元账本 → moomoo 纸交易订单 → 成交对账回 PositionStore；风控门拦截超限单；
  kill switch 全平；校准报告显示内部模拟 vs moomoo 纸交易的成交差。

---

## Phase 13 — 多 agent（条件触发：红队优先）+ 全自动模式

**目标**：仅在触发条件满足时引入 agent；启用全自动 paper。

- **13.1 红队审计者（最早，Paper 期）** — 独立 LLM 审查步骤（`tool_red_team_review`，只读）
  → 对每个进 PAPER 的候选出"反方报告"（泄漏/容量/拥挤度/经济逻辑）存入谱系。
  触发条件：发现产出速度 > 人工审查带宽。
- **13.2 数据质量哨兵** — Phase 10 数据源 ≥3 后引入。
- **13.3 全自动 paper 模式** — `autonomy_mode=auto`：validated→paper 由规则 + 红队通过自动决定；
  人可随时切回 manual；PAPER→ACTIVE 仍需 60 日验证 + 人工（资金决策）。
- **验收**：每个 PAPER 候选附红队反方报告；auto 模式下全流程无人值守（红队+规则为门）。

---

## Phase 14 —（长期，超出本次实现）验证期运营 → 真实资金门槛

60 日 paper 验证、首份验证报告、PAPER→ACTIVE 门槛（realized IC t-stat>2）、归因分析师
（实盘 6 个月后）。**接入真实资金 = 另立规划**（执行/风控/密钥管理），明确排除在外。

---

## Phase R — 研究可信度补强（横向层）**⚠️ 已大部分被 S/PM 吸收，此处仅留指针**

> **收拢说明（最新）**：R.1 高级验证 → **已并入 Phase S.3**（全局 trial + PBO + t≥3.0，已部分完成）；
> R.2 Barra-lite 风险模型 + beta 闭合 → **并入 PM.5**；R.3 容量曲线 → **并入 PM.2（TradingContext 也算容量）**；
> R.4 组合优化（Ledoit-Wolf/HRP/换手率进目标）→ **并入 PM.1/PM.3**。**本节不再是独立待办**，下列内容
> 为各项的方法细节参考。

**来源**：外部《Quant-Agent 覆盖度地图》评估（以 López de Prado / Barra / Almgren-Chriss 为标尺）
指出——本路线 Phase 9–14 几乎全压在"自主交易运营"轴，而对"风险模型 / 因子风险 / 容量 / 高级
验证"这条**决定因子研究可信度**的轴几乎沉默。本 Phase 补齐其中"有必要 + 高性价比"的部分。
（该评估另建议的 Golden Benchmark 与 R-N1 可复现性，**已完成**——见
`tests/golden/test_golden_backtest.py`、`gp_engine/_rng.py`，此处不再列。）

- **R.1 高级验证 → 全部并入 Phase S.3（已大部分完成）**：DSR + Harvey-Liu-Zhu **t≥3.0** 门、PBO
  函数 `backtest_engine/overfit_stats.py` **已建**；剩 CPCV + 把 PBO 接进门，见 Phase S.3 待补。此处不再展开。

- **R.2 风险因子模型（Barra-lite）— 最大结构性空白，一个接口解决四件事** ⭐
  新建 `app/core/risk_engine/`：以 **Fama-French 5 因子 + 行业哑变量**做简化版（不从零建 Barra）。
  1. **真正的风格中性化**：对 size/value/momentum/liquidity/volatility 回归取残差
     （替代现有截面 rank / 行业 demean）；**并顺带闭合 B6**——现有
     `NeutralizationLayer.beta_neutral` 对冲后再 L1 归一化，使净 beta 不严格为 0（"市场中性"
     名不副实）。该项为 PAPER_TRADING_ROADMAP 归档时移交的休眠遗留项（未接入默认 paper 路径），
     在此以"残差化 + 不破坏中性的归一化"一并修正，或在 UI 如实标注为"beta 缓和"；
  2. **因子层风险归因**：组合风险来自哪些暴露；
  3. **结构化协方差** `Σ = BΣ_fB' + D`：供 R.4 的组合优化替代样本协方差，缓解 Markowitz 病态；
  4. **alpha vs 风险因子区分**：判断信号是纯 alpha 还是风险溢价补偿。
  - **依赖**：是"风格中性化增强 + 归因（Phase 14 归因分析师 / Brinson）"的前置。
  - **复用**：`data_engine`（因子暴露构造）、现有 `signal_processor` 中性化钩子。
  - **验收**：对已知风格暴露的构造组合，风险归因能还原其暴露来源；MVO 可切换用结构化协方差。

- **R.3 容量 → 已落地为 PM.2（容量建模）+ TradingContext 容量**：AUM→容量上限已在 live 路径实现
  （water-filling），Sharpe-AUM 衰减曲线为其自然可视化（FE-R）。此处不再展开。

- **R.4 组合优化升级（中等价值，可后置）**
  - **Ledoit-Wolf 最优收缩**替代现有固定 δ=0.5 对角收缩；
  - **HRP（层次风险平价）**作为 MVO 的稳健替代（不需矩阵求逆）；
  - **换手率惩罚进优化目标**（现在仅在 fitness 里软惩罚，未进组合优化目标）。
  - **复用/扩展**：`portfolio_engine/portfolio_constructor.py`。
  - **验收**：病态协方差下 HRP 不崩；Ledoit-Wolf 收缩强度由数据估计而非写死。

- **暂不做（评估亦标"选择性"）**：Meta-labeling、FracDiff、唯一性加权/成交量时钟采样、
  Black-Litterman、Kelly、多资产/多频率/多策略族广度、深度学习/RL——与"横截面日频股票"
  定位正交，记录待未来。

---

## 前端交付路线（一等交付，与各 Phase 并行）

**原则**：每个 Phase 凡产出用户可见产物（待办队列 / 执行状态 / 报告 / 图表），都必须配套前端，
不再当附带项。前端沿用现有栈（React 19 + TS + Zustand + ECharts），复用 `api/client.ts` 类型化
fetch/SSE、`components/analysis/*` 图表、`AlphaDashboard`。**前端只读 + 人工审批动作**，不承载任何
自动交易决策（与"LLM/前端不进交易回路"一致）。

- **FE-8（补现有缺口，可选）**：成本校准报告视图——展示 `monthly_cost_calibration_job` 产出的
  impact_coef 建议与隔夜缺口分布（只读，人工据此手动改 `CostParams`）。需后端加一个只读端点
  暴露最近校准报告。
- **FE-9 审批队列 ✅（2026-08-22，配 Phase 9.4）**：`ApprovalQueue.tsx` 挂到 `AlphaDashboard`——
  待批准队列（VALIDATED 列表 + Sharpe/IC-IR）+ **批准/拒绝按钮**（`approve|reject`，拒绝填原因），
  这是"默认模式"的人机接口。**待补（后续）**：自主发现观测台（每晚候选 + 市场观察摘要 +
  被验证门刷掉的原因）、谱系时间线可视化。
- **FE-10 另类数据浏览**（配 Phase 10）：DatasetView 增加基本面/财报字段浏览 + **发布时点(as_of)**
  标注；因子详情显示其用到的稀疏字段与可见期。
- **FE-11 前向数据状态**（配 Phase 11）：每日增量摄取状态、PIT 逐日增长指标、美股日历/最新 bar
  时间（区分"已收盘入库" vs "待摄取"）。
- **FE-PM 组合与配置视图**（配 Phase PM，**重**）：当前组合的**具体美元持仓表**（每只股票多少股/多少钱）、
  各因子配额与**容量占用**、组合 gross/net 敞口与风险仪表、快/慢因子分组；以及**"批准这份持仓配置"**
  的界面（替代/叠加原因子审批）。
- **FE-12 执行监控面板**（配 Phase 12，**重**）：moomoo 纸交易的持仓/挂单/成交、**本地 vs 券商
  对账差**、风控门状态与 **kill switch（一键全平，带二次确认的人工动作）**、三级保真度对比
  （内部模拟 vs moomoo 纸交易成交）。
- **FE-13 红队报告 + 自主度开关**（配 Phase 13）：每个 PAPER 候选的"反方报告"在谱系内可查；
  **autonomy_mode 手动/全自动切换**（人工可随时切回）；数据质量哨兵告警。
- **FE-R 研究可信度图表**（配 Phase R）：PBO/CPCV 结果、**因子风险归因**（暴露分解）、
  **容量-AUM 衰减曲线**——复用 `components/analysis/*`，天然图表化。

> 每个 FE-* 的验收并入对应 Phase 的"真实启动前后端端到端"验证（沿用 Phase 7 做法）。

---

## 依赖关系（执行顺序）

```
Phase S（统计地基）★★ P0 ── 所有回测结论的前置；未完成前任何数字不可解读
  S.1 去 OOS 选择(✅) · S.2 全路径 holdout(✅) · S.3 全局 trial+PBO+t≥3(◑吸收R.1) · ~~S.4 换退市源~~(🚫搁置,被 TR.2 取代)
Phase 8（PIT 地基，✅）
Phase 9（自主发现 + 生命周期门 + 批准，✅）── 门控设计已演进为「策略级」，见 PM.S
Phase TR（交易现实：moomoo 单一源 + 真实成本/可做空 + 门分级）
  TR.1 TradingContext(✅) · TR.2 moomoo 单一权威源(✅,provider+接线+真连实测) · TR.3 成本落地(◑) · TR.4 门分级(⬜)
Phase PM（组合与资金管理层）★ ── 依赖 9；与 S 并列最高优先级
  第一批(✅) · ★核心重构:策略级门 PM.S1(✅)+边际准入 PM.S2(✅)+经典基准库 PM.S3(✅) · 第二批 PM.5(✅)/6/7(⬜) · 收拢 R.2/R.4
Phase 10（另类数据：基本面，价格仍走 moomoo）·  Phase 11（前向增量，价格源=moomoo/TR.2）
Phase 12（**moomoo** 执行，同 TR.2 源）── 依赖 PM（消费美元账本）+ TR.2 + 11
Phase 13（多 agent + 全自动）── 依赖 9 + 12
Phase 14 ── 长期验证期

Phase R ── ⚠️ 已被 S/PM 吸收（R.1→S.3、R.2→PM.5、R.3→PM.2、R.4→PM.1/3），仅留方法参考，非独立待办

前端 FE-9(✅)…FE-PM(组合/配置视图，随 PM) …FE-R
```

## 关键复用点（避免重造）

- 发现：`PopulationEvolver`、`GenerationWorkflow`（`alpha_workflows.py`）、`AlphaPool`、
  `RegimeDetector`（`data_engine/regime_detector.py`）
- 验证：`WalkForwardBacktester`、`deflated_sharpe_from_returns`（`performance_analyzer.py`）；
  Phase S.3 在此之上加 PBO/CPCV，golden 数值基线复用 `tests/golden/`
- 生命周期：`alpha_lifecycle.py` 状态机、`AlphaStore.update_status`、PATCH 端点
- 执行：`PaperBroker`/`PositionStore`、`TransactionCostEngine`（同一成本模型，禁止另立）
- 调度：`scheduler.py`（`create_scheduler` + SQLAlchemyJobStore，加新 job 即可）
- 数据：`load_registry_dataset`（`dataset_registry.py`）、`DataProvider` 抽象、`pit_store.py`
  （注：旧 `DataManager`/`feature_store.py` 多源管道已于 2026-08-30 删除，被 registry + PIT 取代）

## 验证方式（端到端）

- 每个 Phase 配专属测试套件（`test_phase9.py` … `test_phase12.py`），沿用现有模式：
  单元 + 集成 + **真实启动前后端端到端**（如 Phase 7 那样跑 uvicorn+vite 验证运行时）。
- Phase 9：离线跑 nightly_discovery_job → 断言产出 CANDIDATE（无用户文本）→ 验证门 →
  approve → PAPER，全链路。
- Phase 10：发布日前视测试（季频字段在发布前为 NaN）。
- Phase 11：连续多日增量，PIT 逐日增长，改今日不影响历史 as_of。
- Phase 12：moomoo **纸交易**下单→成交→对账；风控拦截超限单；kill switch 全平；
  校准报告。（用 moomoo OpenAPI 纸交易，不涉真实资金。）
- Phase R.2：对已知风格暴露的构造组合，风险归因还原暴露来源；MVO 可切换结构化协方差。
  （R.1 验收并入 S.3、R.3 验收并入 PM.2，此处不重列。）
- 每 Phase 完成后全量回归（当前基线 448 passed）+ 同步本报告（PAPER_TRADING_ROADMAP 已归档，不再更新）。

## 明确不在本路线范围（真实资金前提）

接入真实资金的券商实盘、机构级风控升级、合规/税务、PagerDuty 级告警——待 Phase 14 验证
报告为正后另立规划。
