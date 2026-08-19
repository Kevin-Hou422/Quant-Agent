# 终极目标路线图：美股实盘全链路自主模拟交易

> 状态：**规划（Phase 8 已完成，Phase 9–14 + R 未实现）** · 地基 Phase 6–8 见
> `backend_retired_report/PAPER_TRADING_ROADMAP.md`（已归档）· 遵循 RESEARCH_OPERATING_MODEL.md
> 生成日期：2026-08-18

## 终极目标

agent **自主从市场观察挖掘因子 → 配置持仓 → 在美股（Alpaca 纸交易）上全链路模拟真实交易**，
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
  Phase 12 的 Alpaca 执行都会在此 SQLite 上扩表（生产可平滑迁 Postgres，URL 已可配）。

## 已确认的关键决策（4 个分叉）

| 分叉 | 决策 | 含义 |
|------|------|------|
| 执行落地 | **Alpaca 纸交易账户** | 免费、美股、真实市场价撮合；需建确定性执行工作流；LLM 不进交易回路 |
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

---

## Phase 8 —（✅ 已完成 2026-08-19，作地基）PIT 数据层与验证运营

**这是 Phase 9/10/11 的前置，现已就绪。** 实现：`app/core/data_engine/pit_store.py`（`PITStore`
双时点 `(field,date,as_of)` 只增不改 + `load_pit()`）、`app/tasks/cost_calibration.py`（8.2 成本
校准回路，T+1 开盘价对账，建议不自动改 `CostParams`）、`backend/OPERATIONS.md`（8.3 运营规则）。
已接入 `daily_ingest`（通过健康门的数据按 as_of 追加）。新增 16 项测试，全量回归绿。

---

## Phase 9 — 自主发现引擎 + 分级生命周期门 + 人工批准（**默认模式核心**）

**目标**：agent 用价量观察自主挖因子 → 走分级生命周期 → 自动验证门 → **默认由人批准/拒绝**。

- **9.1 市场观察引擎** — 新建 `app/core/discovery/market_observer.py`：从 OHLCV 算市场状态
  特征（复用 `RegimeDetector`；截面收益离散度；动量/反转广度；波动状态；`AlphaPool` 各因子
  家族的滚动 IC/Sharpe）→ 输出**结构化的"假设方向"对象**（非用户文本）。
- **9.2 自主发现编排器** — 新建 `app/core/discovery/discovery_engine.py` + `scheduler.py`
  注册 `nightly_discovery_job`：观察市场 → 派生 N 个假设 → 每个跑 `GenerationWorkflow`（复用）
  → `AlphaPool` 去重 → 赢家写为 **CANDIDATE**（修掉"直接写 ACTIVE"）。
- **9.3 自动验证门** — 新建 `app/core/lifecycle/validation_gate.py`：规则化 CANDIDATE→VALIDATED
  ——WalkForward 全折为正（复用 `WalkForwardBacktester`）+ DSR>0.90（复用
  `deflated_sharpe_from_returns`）+ 真实数据 OOS。修掉 `router.py:1906`、`alpha_agent.py:121`
  等把新因子写成 active 的创建路径 → 一律 CANDIDATE 起步。
- **9.4 人工批准/拒绝工作流（默认）** — 新增 `POST /api/alphas/{id}/approve` +`/reject`
  （批准 VALIDATED→PAPER；拒绝→RETIRED 带原因）+ 谱系记录；前端 Live 仪表板加"待批准队列"。
  配置 `autonomy_mode: manual|auto`（默认 manual = 人把关；auto 见 Phase 13）。
- **验收**：nightly job 离线跑出 CANDIDATE（零用户文本）；验证门自动升 VALIDATED；approve
  升 PAPER；除批准点外全程无人值守。原 Workflow A/B 仍可用。

---

## Phase 10 — 另类数据接入（免费起步）+ DSL 稀疏字段（依赖 Phase 8.1）

**目标**：用基本面/财报丰富发现，严格发布时点防前视，DSL 支持季频稀疏字段。

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

**目标**：从历史回放转为真正的前向 paper trading（日频收盘增量）。

- **11.1 provider 增量拉取** — 加 `fetch_latest`/增量接口 + 追加进 PIT（8.1）带 `as_of`。
- **11.2 美股日历 + 时区** — ET 收盘、节假日感知的调度（现有 job 用 UTC 固定时刻）。
- **11.3 `daily_ingest` 重构** — 真增量追加（非整段重拉）；从启动日起积累无幸存者偏差的自有数据。
- **验收**：连续 N 个真实交易日，PIT 每日增一根 bar；日循环消费增量；A1（无人值守日节奏）成立。

---

## Phase 12 — 真实执行层：Alpaca 纸交易 + 确定性执行工作流 + 风控 + paper-vs-real 建模

**目标**：agent 在 Alpaca 纸交易上的完整**确定性**交易工作流；建模保真度差距。

- **12.1 Alpaca 适配器** — 新建 `app/core/execution/alpaca_broker.py`：权重→股数订单
  （`OrderManager`）、经 Alpaca REST 下单（**确定性代码，非 LLM**）、成交轮询、持仓对账
  （本地 vs 券商）、幂等（client order id）。复用/扩展 `PositionStore` schema。
- **12.2 风控门 + kill switch** — 新建 `app/core/execution/risk_gate.py`：单票/总敞口上限、
  日亏熔断、fat-finger（订单 vs ADV）、一键全平；违规订单被拦截。
- **12.3 paper-vs-real 保真度阶梯 + 校准（延伸 8.2）** — 对比 (a) 内部 PaperBroker 收盘成交
  vs (b) Alpaca 纸交易真实成交 → 校准 `impact_coef`、建模 T+1 开盘成交、永久 vs 临时冲击。
  三级保真度：内部模拟 → Alpaca 纸交易 → （未来）Alpaca 实盘。
- **12.4 崩溃恢复** — 重启时从 Alpaca 持仓重建状态。
- **验收**：PAPER 因子目标权重 → Alpaca 纸交易订单 → 成交对账回 PositionStore；风控门拦截
  超限单；kill switch 全平；校准报告显示内部模拟 vs Alpaca 纸交易的成交差。

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

## Phase R — 研究可信度补强（横向层，与 Phase 9–12 并行，不属"实盘运营"轴）

**来源**：外部《Quant-Agent 覆盖度地图》评估（以 López de Prado / Barra / Almgren-Chriss 为标尺）
指出——本路线 Phase 9–14 几乎全压在"自主交易运营"轴，而对"风险模型 / 因子风险 / 容量 / 高级
验证"这条**决定因子研究可信度**的轴几乎沉默。本 Phase 补齐其中"有必要 + 高性价比"的部分。
（该评估另建议的 Golden Benchmark 与 R-N1 可复现性，**已完成**——见
`tests/golden/test_golden_backtest.py`、`gp_engine/_rng.py`，此处不再列。）

- **R.1 高级验证工具（最便宜，直接强化项目最强的"验证"环）**
  - **PBO（回测过拟合概率，Bailey et al. 2015）** — 与已有 DSR 互补：DSR 校正单个夏普，PBO
    检验"筛选流程本身是否导致过拟合"。新建 `app/core/backtest_engine/overfit_stats.py`。
  - **Purged K-Fold CV / CPCV**（López de Prado）— 扩展现有 `WalkForwardBacktester`：训练/测试
    间 purge + embargo（embargo 已有），CPCV 生成多条回测路径替代单路径。
  - **Harvey-Liu-Zhu t≥3.0 门槛** — 把新因子显著性门槛从 t>2 提到 **t≥3.0**，作为 Phase 9.3
    验证门（CANDIDATE→VALIDATED）与 Phase 14 的可配置参数。
  - **接入点**：Phase 9.3 `validation_gate.py` 直接消费 PBO/CPCV/t 门槛。
  - **验收**：一个已知过拟合的构造因子被 PBO 判为高过拟合概率；t 门槛可配置且默认 3.0。

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

- **R.3 容量（Capacity）曲线分析（便宜、直接把 Sharpe → "能装多少钱"）**
  新建 `app/core/backtest_engine/capacity.py`：在不同假设 AUM 下重跑回测（复用
  `TransactionCostEngine` 的 ADV 参与率/冲击），输出 **Sharpe/年化收益随 AUM 衰减曲线**。
  - **验收**：给定一个因子，产出 AUM→Sharpe 衰减曲线；ADV 参与率随 AUM 单调上升。

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
- **FE-9 审批与自主发现可视化**（配 Phase 9，**最关键**）：
  - **待批准队列**：VALIDATED 候选列表 + WalkForward/DSR/t 门槛结果 + **批准/拒绝按钮**
    （调 `POST /api/alphas/{id}/approve|reject`），这是"默认模式"的人机接口。
  - **自主发现观测台**：nightly_discovery_job 每晚产出的候选、市场观察摘要（regime/离散度/
    家族滚动表现）、被验证门刷掉的原因。
  - 生命周期看板扩展：CANDIDATE→VALIDATED→PAPER 分级状态与谱系。
- **FE-10 另类数据浏览**（配 Phase 10）：DatasetView 增加基本面/财报字段浏览 + **发布时点(as_of)**
  标注；因子详情显示其用到的稀疏字段与可见期。
- **FE-11 前向数据状态**（配 Phase 11）：每日增量摄取状态、PIT 逐日增长指标、美股日历/最新 bar
  时间（区分"已收盘入库" vs "待摄取"）。
- **FE-12 执行监控面板**（配 Phase 12，**重**）：Alpaca 纸交易的持仓/挂单/成交、**本地 vs 券商
  对账差**、风控门状态与 **kill switch（一键全平，带二次确认的人工动作）**、三级保真度对比
  （内部模拟 vs Alpaca 纸交易成交）。
- **FE-13 红队报告 + 自主度开关**（配 Phase 13）：每个 PAPER 候选的"反方报告"在谱系内可查；
  **autonomy_mode 手动/全自动切换**（人工可随时切回）；数据质量哨兵告警。
- **FE-R 研究可信度图表**（配 Phase R）：PBO/CPCV 结果、**因子风险归因**（暴露分解）、
  **容量-AUM 衰减曲线**——复用 `components/analysis/*`，天然图表化。

> 每个 FE-* 的验收并入对应 Phase 的"真实启动前后端端到端"验证（沿用 Phase 7 做法）。

---

## 依赖关系（执行顺序）

```
Phase 8（PIT 地基）── 必须先做
  ├─→ Phase 9（自主发现 + 生命周期门 + 批准）── 可与 8 并行启动，9.3 依赖回测
  ├─→ Phase 10（另类数据）── 依赖 8.1 PIT
  └─→ Phase 11（前向增量）── 依赖 8.1 PIT
Phase 12（Alpaca 执行）── 依赖 9（有 PAPER 因子）+ 11（前向数据更佳）
Phase 13（多 agent + 全自动）── 依赖 9（发现产量）+ 12（执行）
Phase 14 ── 长期验证期

Phase R（研究可信度补强，横向层，与 9–12 并行）
  R.1 高级验证 ── 尽早，接入 Phase 9.3 验证门
  R.2 Barra-lite 风险模型 ── 是"风格中性化增强 + 归因（Phase 14/Brinson）"的前置
  R.3 容量曲线 ── 独立，可随时做
  R.4 组合优化升级 ── 可后置，扩展 portfolio_engine

前端 FE-9…FE-R ── 与对应 Phase 同批交付（FE-9 审批队列最关键，随 Phase 9 一起）
```

## 关键复用点（避免重造）

- 发现：`PopulationEvolver`、`GenerationWorkflow`（`alpha_workflows.py`）、`AlphaPool`、
  `RegimeDetector`（`data_engine/regime_detector.py`）
- 验证：`WalkForwardBacktester`、`deflated_sharpe_from_returns`（`performance_analyzer.py`）；
  Phase R.1 在此之上加 PBO/CPCV，golden 数值基线复用 `tests/golden/`
- 生命周期：`alpha_lifecycle.py` 状态机、`AlphaStore.update_status`、PATCH 端点
- 执行：`PaperBroker`/`PositionStore`、`TransactionCostEngine`（同一成本模型，禁止另立）
- 调度：`scheduler.py`（`create_scheduler` + SQLAlchemyJobStore，加新 job 即可）
- 数据：`DataManager.get_panel`、`DataProvider` 抽象、`feature_store.py`

## 验证方式（端到端）

- 每个 Phase 配专属测试套件（`test_phase9.py` … `test_phase12.py`），沿用现有模式：
  单元 + 集成 + **真实启动前后端端到端**（如 Phase 7 那样跑 uvicorn+vite 验证运行时）。
- Phase 9：离线跑 nightly_discovery_job → 断言产出 CANDIDATE（无用户文本）→ 验证门 →
  approve → PAPER，全链路。
- Phase 10：发布日前视测试（季频字段在发布前为 NaN）。
- Phase 11：连续多日增量，PIT 逐日增长，改今日不影响历史 as_of。
- Phase 12：Alpaca **纸交易沙盒**下单→成交→对账；风控拦截超限单；kill switch 全平；
  校准报告。（用 Alpaca paper API key，不涉真实资金。）
- Phase R.1：构造一个已知过拟合因子 → PBO 判高过拟合概率；t 门槛可配置默认 3.0。
- Phase R.2：对已知风格暴露的构造组合，风险归因还原暴露来源；MVO 可切换结构化协方差。
- Phase R.3：给定因子产出 AUM→Sharpe 衰减曲线，ADV 参与率随 AUM 单调上升。
- 每 Phase 完成后全量回归（当前基线 448 passed）+ 同步本报告（PAPER_TRADING_ROADMAP 已归档，不再更新）。

## 明确不在本路线范围（真实资金前提）

接入真实资金的券商实盘、机构级风控升级、合规/税务、PagerDuty 级告警——待 Phase 14 验证
报告为正后另立规划。
