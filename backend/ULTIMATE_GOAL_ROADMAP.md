# 终极目标路线图：美股实盘全链路自主模拟交易

> 状态：**Phase S/8/9/TR/PM/A/B 已完成（S 于 2026-09-22 关闭）；Phase 12 + FE-12 代码完成（2026-10-06，
> 真 OpenD 端到端验收待做；2026-10-08 外部审计 16 条已修复 15 条、F15 部分修复）；
> Phase 10 完成并在真实 EDGAR + moomoo 上跑通（2026-10-12；2026-10-13 补齐市值 / 风险模型 size·value /
> 价格缓存 / agent 提示词 / FE-10；尚无基本面因子过验证门）；
> 2026-10-18 外部审查 F01–F14 已处理；2026-10-19 复审判定 5 组关闭、9 组部分修复，复审的 N01–N09 已修（见「外部审查 2026-10-18 修复记录」末尾的复审一节）。**尚不能启动无人值守正式前向**：D14 时钟、D15 告警、真 OpenD 人工验收、恢复演练未完成；现役策略须重新提案审批（D13）；
> Phase 13/14 + R.4 未实现，R.2 部分完成** · 地基 Phase 6–8 见
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

## 待你确定的事项（登记于 2026-10-14；定一条划一条）

这些不是技术阻塞，是**需要你拍板**的事 —— 要么会动活数据 / 你的环境，要么是统计纪律上的取舍，要么要花
小时级的机器时间。没定之前，受它影响的开发一律不做；不受影响的照常推进。

| # | 要定的事 | 背景 | 我的建议 | 没定之前卡住什么 |
|---|---|---|---|---|
| D1 | 要不要把基本面因子送过验证门 | 过门会用掉冻结 Test 段的**一次性额度**，并在全局试验计数里记一笔（影响之后所有因子的 DSR 去膨胀）。只看 IS + Validate 的预览里，95 只大盘股上没有一个经典基本面因子有可用信号（见 Phase 10 第二批） | **暂不送。** 等宇宙更大或样本更长再看 | 基本面因子进 PAPER / 进策略 |
| D2 | 要不要重算活库里 10 个因子记录的 `ic_ir` | 2026-10-13 之前回测报告的 IC 按行号错位，入库的 `ic_ir`（及同批的 IC 类指标）是错的。**已核实**：每日监控的 IC（`alpha_ic_history`，→ACTIVE 门的证据）与 GP 适应度的 IC 不受影响，只有这一列 | **重算**（先备份 `alphas.db`，改动量很小） | 面板 / 台账上这 10 个因子的 IC 数字不可信 |
| D3 | 开发环境怎么对齐 CI | CI 装 `requirements.lock`；本机 venv 有 75/104 个包版本不同（2026-10-13 的 CI 红就是 pytest 9.0.3 vs 9.1.1）。lock 里 moomoo-api 是 10.11，而真 OpenD 联调用的是本机的 10.10 | 在 OneDrive 之外建一个**按 lock 安装的常驻环境**专门跑测试；本机 venv 先不升（升 moomoo-api 要重做一次真 OpenD 预检） | 目前靠会话临时目录里的干净环境，会话结束就没了 |
| D4 | `tests/fixtures/prices/` 留不留 | 里面是 22 个交易日的 NVDA / AAPL 真实收盘价（moomoo + yfinance），已随 a3f1512 提交。EDGAR 数据是公有领域，这两小段行情不是 | 仓库公开的话换成手写的等价数字（十分钟的事） | 无（纯合规取舍） |
| D5 | 变异点逐点复核欠账做不做、何时做 | 欠约 685 点：资金切片 6 + Phase 10 第一批 244 + 第二批 47 + 第三批 30 + 外部审查修复批 295（2026-10-18）+ 复审修复批 63（2026-10-19）。按以往速度是小时级的机器时间 | 攒到 Phase 10 收尾后一次做；或者只复核交易路径上的（资金切片 6 点 + 每日管线）。外部审查的建议：先补交易 / 证据关键路径（健康门、公司行动、策略规格、资金切片、重试认领、前向证据、恢复） | 这几批代码"每一行都有用例盯着"这句话还不能说 |
| D6 | 活环境要不要开 `ENABLE_FUNDAMENTALS` | 要在你的 `backend/.env` 里加 `ENABLE_FUNDAMENTALS=true` 和 `EDGAR_USER_AGENT=名字 邮箱`（我不能读写 `.env`）。开了之后每日管线每天向 SEC 取 95 份文件清单、向 moomoo 多取一遍不复权价（约 1 分钟） | 现役策略只用价格，**现在不用开**；等有基本面因子要进策略时再开 | 活库里的发现任务不会提基本面方向 |
| D7 | 多类别股（`BRK-B`、`V`）的股本从哪来 | `companyfacts` 不收带维度的事实，这两只没有可用股本 → 没有市值 / 估值字段。可选：(a) 保持为空；(b) 用 moomoo 快照的总股本 —— 只有"今天"的值，不是时点数据，回测里用就是前视；(c) 解析完整 XBRL 实例，工作量大 | **(a) 保持为空**，2/95 只票不值得为它引入前视风险 | 这两只票的估值字段 |
| D8 | SUE 要不要提前到财报新闻稿当天可见 | 现在等 10-Q 披露（很多公司比新闻稿晚几天到几周）。新闻稿里的数字不在 `companyfacts` 里，要另接数据源（多为付费） | 先不做；路线图本来就写了"付费源待免费验证后再上" | 财报后漂移类因子的时效 |
| D9 | Validate 段评估时要不要带预热历史 | 现在三段各自独立评估。窗口长于该段的因子在该段没有数字 —— 包括现役的 12 月动量（252 日窗口 vs 232 天的 Validate 段）。带预热 = 让 Validate 的信号用到 IS 末尾的价格（不是收益），与 embargo 的关系要想清楚 | 需要你定统计口径，我倾向于允许预热（信号用历史价格不构成泄漏，泄漏的是标签） | 长窗口因子在 Validate 段的可评估性 |
| D10 | 第一笔真实纸交易订单 | 代码与预检就绪，但**不是**"可以起调度器"：现役策略没有可执行规格（D13）、时钟未统一（D14）、告警只有日志（D15）。复审（2026-10-19）的验收清单：专用 SIMULATE 账户、最小数量、单实例，逐项保存受理 / 成交 / 撤单 / DAY 单跨 session / 超时与未知单 / 同日重跑与重启 / 提交中熔断与续平 / 数据规格报价异常 / 备份恢复的证据 | **单独、人工监督的联调验收**，不直接运行无人值守启动脚本；联调期不计入正式策略样本 | Phase 12 的真实闭环验收；正式前向观察时钟的起点 |
| D11 | 审计遗留项排不排 | F15 余项（按策略版本分账）、跨进程锁、复权价、前端竞态已在 2026-10-18 / 19 两轮修掉；剩下：API 鉴权、告警（D15）、静态宇宙、性能契约 | 分模式说：小额人工联调可以容忍鉴权 / 静态宇宙 / 性能契约暂缺（回环监听、单人使用）；**无人值守与据此晋级不可以**缺告警；远程访问前必须先做鉴权 | 无人值守运行（不阻塞人工联调） |
| D12 | 活库各数据集的**冻结 Test 起点**定在哪天（2026-10-18 登记） | 外部审查 F04 修复后，每个真实数据集第一次被切分时把 Test 起点（绝对日期）写进 `frozen_boundaries`，**只写一次，代码里没有改它的入口**。不拍板的默认行为：活库上第一次跑研究（发现任务、回测、验证门、提案）的那天，按"当时的数据末日 − 2 年"固定。此前在滑动窗口上汇报过的 Test 使用记录都还在，按区间重叠计次，会如实显示为已超预算 | 复审建议：**不要让任意一次 API 调用偶然决定边界** —— 显式登记 dataset + 绝对日期，用"此前看过的最早 Test 窗口起点"作保守边界（`HoldoutLedger.earliest_exposed_start(key)` 能查到这个日期，`BoundaryStore.pin` 写入）。旧 Test 已被看过的部分不会因为重定日期恢复清白，只能等新的前向数据。**我没有对活库做任何切分** | 活库上第一次研究调用会把它定下来（在你登记之前） |
| D13 | 现役策略要**重新提案并审批**（2026-10-18 登记） | 外部审查 F07 修复后，每日循环只按带"可执行规格"的 active 配置调仓：成分、冻结权重、合成方法、无交易带照规格执行；AUM、风控上限、信号参数、券商与做空口径、数据集与价格源、各成分表达式与运行时核对，变了就不调仓。2026-10-18 之前批准的配置没有规格 → **每日循环不会按它调仓**，上线预检第 3 项也会报红。我没有动活库 | 在前端点一次"提案"，核对规格（批准页的 `spec_hash`）后批准并激活。复审（2026-10-19）补充：在提案入口改为"先冻结规格再评估"（N05，已修）之后做；保留旧配置与旧证据，**不要给旧策略手工补一个哈希**，旧证据也不能继承给新策略。以后改 `PAPER_AUM` / 风控上限 / 价格源等交易参数，同样要重新提案 | 活库上的模拟账本与券商执行（D10 的第一笔订单之前必须先做） |
| D14 | 信号 / 成交 / IC 的时钟统一成哪一种（2026-10-18 登记） | 外部审查 F14：每日循环在 T 日收盘后用的是 T−1 日的原始信号（`delay=1` 取最后一行），订单计划 T+1 成交；因子监控算 `signal[T−1]` 对 T 日收益的 IC；内部模拟按 T 日收盘成交。三条序列的持有区间不同。可选：(a) 用 T 日信息、T+1 开盘成交（信号去掉多出的一天延迟，IC 改为 `signal[T]` 对 T+1 开盘到收盘）；(b) 保持额外一天延迟，但三处都按同一口径重写 | **(a)**，与「收盘后决策、次日成交」的实际流程一致；但它改变所有因子的信号、回测与 IC 口径，历史数字要整体重算，并与 D10 的 DAY 单跨日验收一起做 | 前向 IC、模拟账本与券商成交三者能否直接对比 |
| D15 | 告警发到哪里（2026-10-18 登记） | 外部审查 F14：现在只有日志。需要一个你会看的渠道（邮件 / 企业微信 / Telegram / 桌面通知），以及哪些事件要推（摄取拒绝、规格不一致不调仓、对账差异、熔断、证据改写尝试、调度任务失败） | 先接一个最简单的渠道，只推「当天没有调仓」和「熔断 / 对账差异」两类 | 无人值守运行（D10 之后） |

外部审查 2026-10-18 对 D1–D11 也给了建议（`audit-review/审查报告_2026-10-18.md`）。与上表不同的几处：
D1 —— "过门必耗 Test"不对：验证门默认 `report_test=False`，默认因子门是泄漏过滤；可以做选择段内诊断，暂不晋级；
D5 —— 先补交易 / 证据关键路径（健康门、公司行动、策略规格、资金切片、重试认领、前向证据、恢复），再补全部；
D9 —— 允许预热，并优先实现（预热区收益不计入 Validate）；
D10 —— 第一笔模拟单作为**人工监督的独立验收**，当前不直接起无人值守调度；联调不计入策略统计样本；
D11 —— 前向证据按策略版本分账、跨进程认领、告警提前做；价格公司行动升为阻塞项（F02，已修）。

## 贯穿全程的铁律（来自 RESEARCH_OPERATING_MODEL.md，不可违反）

- **LLM 永不进交易回路**：每日信号/下单/风控是确定性代码；LLM 只在研究侧（假设生成、
  代码、报告解读、红队审查）。
- **状态流转永不作为 LLM 工具**：晋级走人工 approve/PATCH 端点；`tool_save_alpha` 是唯一
  可接受的写（只增台账，无资金后果）。Phase 6 起新工具几乎全只读。
- **另类数据必须带发布时点进 PIT**（报告期 ≠ 可见期，否则前视泄漏）。
- **多 agent 仅按触发条件引入**（目标对立/信息隔离/时间尺度不同）；红队是最早有回报的角色。
- 保留原有用户假设 Workflow A/B（加法，不替换）。

> **⚠️ 方法论警示（外部评估 2026-08-22 核实 · 2026-09-22 随 Phase S 完成更新）**：原判断是
> 回测数字**不是"偏乐观"，而是"方向不明"**——幸存者偏差 + OOS 被选择性挖掘的联合效应，
> 符号与大小都无法估计。**Phase S 已完成，去掉了其中"被选择性挖掘"那一半**（三段切割 +
> 冻结 holdout + 使用次数台账 + purged CV + CPCV-PBO）。
> **剩下的一半仍在**：免费价格源没有退市股，历史幸存者偏差未修（S.4 已决策搁置）。
> 因此现在的口径是：**相对结论（谁比谁好、选择流程有没有过拟合）可以读；
> 绝对水平（年化多少、Sharpe 多高）仍不可当决策依据**，要等前向 paper 的 realized IC。

---

## Phase S — 数据与统计地基修正（**P0 · 阻塞项 · 最高优先级，先于除 PM 外一切**）

**为什么**：外部评估（逐行核实无误）指出三个致命问题，使目前**所有回测数字失去决策意义**。这些不是
"补强"，是"不修则一切数字无效"。工程和治理已达机构级，短板在**数据真实性 + 统计诚实性**。

- **S.1 去掉 fitness 的 OOS 选择 ✅（2026-08-24 发现路径 · 2026-09-22 收尾）** —— 原状
  `fitness = sharpe_oos − …` **直接按 OOS 择优 → OOS 退化为第二个样本内**。已改：`GenerationWorkflow`
  用三段切割，GP 只在 **Validate** 段做适应度选择，**Test 段全程不可见**。
  **收尾（2026-09-22）**：① 适应度换成 **IS 内部 purged K 折**（`data_partitioner.PurgedKFold` +
  `gp_engine/evaluation_utils.purged_cv_sharpe`，发布默认 `s_fitness_mode=purged_cv`）——
  分数不再取决于"最后那一段恰好是什么行情"，留出块两侧各 purge `s_embargo_days` 防滚动算子跨块借数；
  `EvalResult.oos_metric` 带出口径（`holdout|purged_cv`），**同名字段换了含义必须说出来**。
  ② `/api/backtest/*` 与 `/alpha/{simulate,optimize}`、`/gp/evolve`、`/agent/run`、`/alphas/{id}/retrain`
  全部改走 selection 段。
- **S.2 全路径强制真 holdout ✅（2026-09-22）** —— 三段切割升为一等组件
  `data_engine/data_partitioner.ThreeWayPartitioner`（`train | embargo | validate | embargo | test`，
  **段间 embargo 是新加的**：原 `_partition_three_way` 是纯 iloc 切片、段间无隔离）。
  - **Test 段按日历冻结**：`s_test_freeze_years=2.0`，窗口不随入参漂移（用例钉住"换个起始日
    重跑，test_key 不变"）。兑现不了时按 `s_max_test_share=0.35` **份额封顶**并标
    `frozen_by=years_capped / degraded=True` —— 不静默缩窗，也不把选择段钉死在下限上
    （钉死会让 n_days 400 与 900 切出一样大的研究样本，等于把 B-2 换个地方复发）。
  - **一次性使用台账**：`db/trial_ledger.HoldoutLedger`（append-only 表 `holdout_usages`）。
    每次真的在 Test 段上算指标就 +1，超 `s_holdout_budget=1` 返回 `over_budget=True` 并 log.error。
    **不阻断**（阻断会让人绕开台账），但 `uses/budget/over_budget` 跟着数字一起进响应体与 RunManifest。
    台账写不进去时 `recorded=False`、`uses=-1`，**不许用 1 冒充"这是第一次看"**。
  - **`ValidationGate` 走三段**：WF+DSR 只看 `split.selection`；Test 段仅在 `report_test=True` 时
    动用一次并记账。用例反向验证：把信号造成"只在冻结段有效"，门必须判不过。
  - 全路径由 `tests/meta/test_lessons_enforced.py::TestLessonK` 的新检查守着：router 里调了
    `_resolve_dataset` 却没调 `_three_way` 的端点直接判红（豁免须写明理由）。
- **S.3 全局多重检验计数器 ✅（2026-08-24 起 · 2026-09-22 收尾）**—— `db/trial_ledger.TrialLedger`
  跨会话累计 trial；`ValidationGate`/`StrategyGate` 用**全局累计数**做 DSR 去膨胀 + **t≥3.0**
  （Harvey-Liu-Zhu）。PBO 已接进 StrategyGate（2026-08-30）。
  **收尾（2026-09-22）：CPCV 已落地并成为默认** —— `overfit_stats.combinatorial_purged_splits /
  cpcv_pbo / cpcv_path_sharpes`：C(N,k) 组合、留出块两侧 purge。CSCV 的块直接相邻，日频面板上
  IS 块末尾与 OOS 块开头共享原始 bar → **PBO 被系统性低估**（选择流程看起来比实际更不过拟合）。
  `StrategyGate` 默认 `s_use_cpcv=True`，算不动时退回 CSCV 但 **`pbo_method` 如实标注**
  （两种口径的 0.42 不是同一个 0.42）。（吸收原 Phase R.1。）
- **S.4 换含退市收益的价格数据源**（根治历史幸存者偏差）—— **🚫 已搁置（2026-08-25，被 TR.2 取代）**。
  原计划接付费源（Sharadar/CRSP）修历史幸存者偏差，但用户目标是**零预算的前向 paper 模拟**：
  ① 幸存者偏差只影响**历史回测**，**前向 paper 天生无此偏差且免费**（前向 paper 会自纠幸存者伪因子）；
  ② 单一权威源统一到 **moomoo（TR.2）**，退市股虽仍无，但不在"免费前向"的关键路径上。**故不做 S.4。**

**依赖/次序**：S 是**所有回测结论的前置**；与 Phase PM（组合层）并列为最高优先级——PM 补"操作"轴，
S 补"数字可信"轴。（S.4 已搁置，S.1/S.2/S.3 为本层实质内容，**均已完成 2026-09-22**。）

### Phase S 完成后，哪些结论可以读、哪些仍然不可以（2026-09-22）

| 问题 | 状态 | 说明 |
|------|------|------|
| 按 OOS 择优的循环论证 | ✅ 已消除 | 选择只在 selection 段发生；Test 段被冻结且使用次数有账可查 |
| 段间泄漏（滚动算子跨切点） | ✅ 已消除 | 三段之间、CV 折之间、CPCV 组合之间都做了 purge/embargo |
| 多重检验膨胀 | ✅ 已校正 | 全局 trial 台账 + DSR + t≥3.0 + **CPCV** 口径的 PBO |
| 单段 holdout 的段位偏倚 | ✅ 已降低 | 适应度改 purged K 折，每个样本都当过一次留出 |
| **历史幸存者偏差** | 🚫 仍在（S.4 已决策搁置） | 免费价格源没有退市股。**历史回测的绝对水平仍不可当结论**；解法是前向 paper（天然无此偏差） |
| **前向证据不足** | ⬜ 未开始 | 判定口径已就绪：信号先存档、结果后结算，按策略 id + 规格哈希记账（2026-10-18 外部审查 F06）；但真实前向样本仍为 0（见 OPERATIONS M5） |

一句话：**"选出来的东西是不是被数据挖出来的"这个问题已经能回答了；"这个收益率数字有多大"
仍然不能当真**——后者要等前向 paper 跑满 60 个交易日。

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
- **PM.6 horizon 感知配置 ✅**（2026-08-31）— 已建 `app/core/portfolio_manager/horizon.py`：
  ①**无交易带调仓 `apply_no_trade_band`（合并 TR.1 band，减换手省成本）已接进 `run_portfolio`**
  （band 由 TradingContext 数据推导或配置 `pm_no_trade_band`）；②因子按年化换手分 fast/slow
  （`horizon_profile`，阈值 `pm_horizon_fast_thresh`）落进返回值 `horizon/no_trade_band/turnover_ann`。
  测试 `test_phase_pm6_horizon.py`(6+接线)。**实测生效**（小漂移被无交易带吸收、换手下降）。
- **PM.7 审批对象升级 ✅**（2026-08-31）— 策略配置升为**一等实体**：新建 `db/strategy_store.py`
  （`StrategyStore`/`StrategyConfig` + `strategy_configs`/`strategy_decisions` 表，状态机
  proposed→approved→active→retired/rejected）、`portfolio_manager/strategy_builder.py`
  （`build_strategy_config`：边际准入+合成+策略门+风控+换手 → 一份配置；`propose_from_paper_factors`）。
  端点 `POST /strategies/propose`、`GET /strategies/pending`、`/{id}`、`POST /{id}/approve(activate)`、
  `/{id}/reject`。**接线**：`run_portfolio` 有 active 配置时**只交易该配置成分**（`active_config` 落返回值）。
  **修错配**：① 因子入池门降级为**低门槛泄漏过滤** `lifecycle/leak_filter.py`（严门在策略层，配置
  `factor_gate_mode=leak|strict`）；② 新增**策略级衰减监控**（组合账本 `check_decay`，`strategy_decay` 落返回值）。
  测试 `test_phase_pm7_strategy.py`(3)+`test_phase_pm7_endpoints.py`(3)。
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
  5/1/2）；`run_portfolio` 组合账本改用 grounded 成本 broker（配置 `trading_broker`）。
  **T3 接口已补齐 ✅（2026-09-02）**：`trading_context/providers.py`——`QuoteProvider`(盘口价差/中间价)、
  `BorrowProvider`(可做空/借券费)、`AccountProvider`(买入力/现金/持仓) 抽象 + **Sim 实现**（背 TR.1
  Corwin-Schultz 估计 + 纸账户真实记账）+ 工厂 `get_trade_providers(mode)`。**`mode="live"` 显式抛错**
  （绝不用估计冒充实时——T3 纪律），moomoo 实时实现待 Phase 12。已接进 `run_portfolio`（`t3` 落返回值）。
  测试 `test_phase_tr3_providers.py`(6)。
- **TR.4 门分级 + 阈值配置化 ✅**（2026-09-02）— 已建 `lifecycle/promotion_gate.py`：
  **第 4 步 进 PAPER** `grade_paper_entry` **A/B/C 分级**（A=严门全过、B=未过但 Sharpe>0、C=不合格），
  **实验模式**（`tr_experiment_mode`，默认开）放行 B/C 以收前向证据但**如实标注等级**；
  **第 5 步 →ACTIVE** `check_active_promotion` **最严**（≥`tr_min_forward_days`=60 交易日、
  realized IC 均值>0 且 **t>`tr_min_ic_tstat`**=2）。**阈值全配置化**（`tr_*`，不写死）。
  **已接线（两步都接，2026-09-02 补齐）**：第 4 步分级接进 `build_strategy_config` 与 `run_portfolio`
  （`paper_grade`/`paper_entry` 落进 strategy verdict，前端策略卡显示等级徽章）——
  *注：首次交付时第 4 步只写了库未接线，违反 DEV_LESSONS §K，已补*；
  第 5 步 `POST /strategies/{id}/approve?activate` 跑 →ACTIVE 门并把结论写进审批谱系；
  是否阻断由 `tr_enforce_active_gate` 控制（**默认 False**——因 ic_history 尚未分离回放/前向，
  见 Phase 11；分离完成且逼近真钱时应设 True）。测试 `test_phase_tr4_promotion.py`(7)。
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

### 实施记录（2026-10-12）—— 代码完成，真实数据跑通

**数据源改成了 SEC EDGAR，没有用上面写的 yfinance `.quarterly_financials`。** 原因是唯一不能错的
那件事：yfinance 只给财报**期末日**，不给披露日，用它就得猜一个滞后；EDGAR `companyfacts` 的每条事实
带 `filed`（实际披露日），同一期间被重述时多出一条 `filed` 更晚的记录，"某天已披露的最新版本"可以
逐条还原。集成测试里有一条专门演示：把可见日换成期末日，同样的前视检查会红。

| 子项 | 落地 | 位置 |
|---|---|---|
| 10.2 基本面 | 7 个字段：`earnings_yield` `book_to_price` `roe` `gross_margin` `asset_growth` `sue` `days_since_earnings`；事实只追加落盘在 `<PIT_STORE_DIR>/fundamentals`，面板由 (事实, 价格) 现算，**不写进价格 PIT** | `data_engine/fundamentals/` |
| 10.1 稀疏字段 | 字段带类别（估值 / 季频阶梯 / 事件计数器）；披露日之后**第一个交易日**起可见，ffill，200 天无新披露置空；`SparseFieldValidator` 拒绝阶梯字段上的短窗口离散度运算 | `alpha_engine/fields.py`、`validator.py` |
| 10.3 财报事件 | SUE（季节性随机游走，净利润口径，Q4 = 全年 − 前三季）；`days_since_earnings` 取 8-K item 2.02 的申报日 | `fundamentals/pit.py`、`panels.py` |
| 10.4 算子与发现 | `fund_rank`（带 50% 覆盖率门槛的截面排名）；GP 新增 value / profitability / earnings 三族种子；观察引擎在字段覆盖率 ≥ 50% 时给出这三个方向；数据集没有的字段，种子在进 GP 前就被丢掉 | `fast_ops.py`、`gp_engine.py`、`market_observer.py`、`alpha_workflows.py` |
| 接线 | `ENABLE_FUNDAMENTALS`（默认关）+ `EDGAR_USER_AGENT`（无默认值，SEC 要求声明身份）；研究路径 `load_registry_dataset` 与每日前向管线都挂字段；挂不上时字段**缺席**（引用它的因子报错），不是一列 NaN | `config.py`、`dataset_registry.py`、`daily_ingest.py` |

**验收逐条**：基本面因子可解析 + 回测 ✅；发布日无前视 ✅（三种验法：具体日期、删掉未来事实后过去逐位不变、
重述只影响其披露日之后；并有反例证明检查有判别力）；发现引擎形成基本面假设 ✅（财报季数据上 earnings
排第一，GP 产出引用基本面字段的候选）。

**真实运行（us_broad_large 95 只，真 EDGAR + 真 moomoo 不复权价）**：

- 末日覆盖率：E/P 96.8%、B/P 89.5%、ROE 91.6%、毛利率 57.9%、资产增长 98.9%、SUE 98.9%、财报日 100%。
- **独立核对**：与 moomoo 行情快照自带的市盈率(TTM) / 市净率逐只比 —— B/P 可比的 81 只**全部**在 ±15%
  内（比值中位数 1.000）；E/P 可比的 88 只里 84 只在 ±15% 内（中位数 1.011），没有一只差到 2 倍。
  偏离的 4 只：FCX 1.67、MDLZ 0.74、ABT 1.17、D 1.17（未追查，推测是归母口径 / 非经常项差异）。
- 识别出 12 只票的 13 次拆股（含 GE 的 8 并 1），拆股前后市值连续。
- 首次同步 95 只约 9 分钟（下载 + 逐披露日计算）。之后每天只取各家的文件清单（很小），有新的年报 / 季报
  才重下那一家的 companyfacts（5–8 MB）；逐披露日的指标按披露日增量缓存。不复权价自 2026-10-13 起
  落盘缓存（见下方"第二批"）。

**第一次真实运行查出、单测（当时只有 AAPL / NVDA 两家夹具）看不到的问题，均已修并配了真实数据的回归用例**：

1. 毛利率覆盖率只有 29.5% —— 多数公司不报 `GrossProfit`，只报营业成本 → 用 收入 − 营业成本 兜底（→ 57.9%）。
2. BKNG 的净利润只在 `NetIncomeLossAvailableToCommonStockholdersBasic` 下 → 加为第三顺位科目。
3. 净资产为负的公司（MCD、SBUX、PM、BKNG …）B/P 为负数混进截面 → 置空。
4. XOM 2026-07 重组换了 CIK，历史在旧主体名下 → 登记继任关系并合并（只登记核实过的这一条）。
5. Citi 2026 年的季报 SEC 接口没收进来 → 过期保护已把它置空，现在加载报告里点名。
6. 落盘的是解析后的事实，解析口径变了缓存不会失效 → 加解析版本号。
7. yfinance 的 `Close`（`auto_adjust=False`）其实**已按拆股调整**，moomoo 的不复权才是真不复权 → 两个源分开处理。
8. EDGAR 的 `acceptanceDateTime` 口径因申报人而异（AAPL 按 UTC 读会与申报日矛盾，BKNG 又是准的）→ 事件日只用申报日。

### 第二批（2026-10-13）—— 把第一批留下的口子收掉

| 项 | 做了什么 | 位置 |
|---|---|---|
| 市值字段 | 第 8 个字段 `market_cap`（不复权价 × 拆股调整后的股本），估值被判离谱的日子一并置空；新增 `size` 家族（种子 + 关键词，观察引擎不主动提 —— 大盘股宇宙里这个方向很弱） | `fundamentals/panels.py`、`gp_engine.py` |
| 风险模型 size / value | 数据集带 `market_cap` / `book_to_price` 且估计窗口里 ≥90% 的交易日有 ≥50% 的股票有值 → 这两个风格进截面回归；否则缺席，并在归因结果里写明原因（`styles_missing_reason`）。R.2 留下的"`styles_missing: [size, value]`"不再是写死的 | `risk_engine/factor_model.py` |
| 不复权价缓存 | 不复权价是历史事实，只追加落盘；已取过的区间不再向数据源要，只取尾巴（重叠一天，拆股靠相邻两天的比值识别）。真实加载 152.7 秒 → 62.8 秒（剩下的是复权价本身） | `fundamentals/store.py`、`service.py` |
| agent 提示词 | 可用字段按数据集**实际有的**给；LLM 给出的表达式若引用了数据集没有的字段（编的，或没开基本面却用了基本面字段）不放行，降级到关键词映射 | `agent/_constants.py`、`_tools.py` |
| FE-10 | `GET /api/datasets/{name}/fundamentals`（只读本地事实，不发任何请求）+ 数据集页的 Fundamentals 面板：字段类别、覆盖率、每只票的最近披露日（= 从哪天之后可见）、过期标记、算不出来的字段 | `api/router.py`、`frontend/.../FundamentalsPanel.tsx` |

**基本面基准因子的真实预览**（us_broad_large 95 只；只算 IS 2021-01~2023-09 与 Validate 2023-10~2024-09，
**没碰冻结的 Test 段，没过验证门，没写任何库**；多空、含成本）：

| 因子 | IS Sharpe | IS IC | Val Sharpe | Val IC |
|---|---:|---:|---:|---:|
| E/P | −0.33 | −0.003 | −0.80 | −0.008 |
| B/P | −0.18 | −0.009 | −1.06 | −0.015 |
| ROE | −0.31 | 0.006 | −0.16 | 0.007 |
| 毛利率（覆盖 60%） | 0.17 | 0.005 | 0.95 | 0.006 |
| −资产增长 | 0.15 | 0.003 | −0.85 | −0.012 |
| SUE | −0.79 | 0.004 | −0.15 | 0.018 |
| −log 市值 | 0.18 | 0.002 | −0.30 | −0.005 |
| [对照] 12 月动量（现役策略的因子） | −0.54 | 0.015 | —（窗口长于该段） | — |

读法：**这个宇宙、这段时间里，没有一个经典基本面因子显示出可用的信号**。价值是负的（2021–2024 是成长股行情），
SUE 的 IC 为正但扣成本后是负的，毛利率最好也只是 IC 0.005。95 只大盘股 + 不到四年的样本，这个结论本身的把握也有限。
所以**没有把任何基本面因子送去过验证门**：那会用掉冻结 Test 段的一次性额度、并在试验计数里记一笔，值不值得由你定。

**这一批顺带查出并修掉的两个既有缺陷（与基本面无关）**：

1. **回测报告的 IC 按行号错位。** 信号有预热窗口时回测结果比价格面板短，`rolling_rank_ic_from_prices` /
   `ic_decay_curve` / `decile_analysis` 却按行号配对，信号配到的是它自己形成窗口之内某一天的收益。
   纯随机游走上（真实 IC = 0）5 日反转报 −0.417、12 月动量报 +0.055；真实数据上 5 日反转报 −0.35
   （预览里就是这个不可能的数字露的馅）。改为按日期对齐后分别是 0.00 / 0.00 与 +0.009。
   盈亏（Sharpe）不受影响；受影响的是 IC / ICIR / 分档收益，它们会落库（因子记录的 `ic_ir`）并显示在面板上
   —— **此前入库的因子，这几个数是错的**（见下方未做项）。
2. **点变异会产出解析不了的 DSL。** `ts_corr(close, volume, 10)` 的算子被换成单输入算子而第二输入还留在节点上，
   `repr` 出 `ts_min(close,volume,10)`；种子生成会把它存进种子表（40 个随机种子 × 2 个假设里出现过 63 种）。
   现在带第二输入的节点只在双输入算子之间互换；全部变异算子的产物都验证过能往返解析。

**没做 / 已知局限**：

- **没有任何基本面因子过验证门**，更没有进策略。字段是对的 ≠ 因子有效（见上面的预览）。
- **此前入库的因子记录里的 `ic_ir` 没有重算**（活库里 10 个因子）：重算要动 `C:\QuantAgentData\alphas.db`，没有擅自做。
- `BRK-B`、`V` 没有估值字段：多类别股按类别报股本，`companyfacts` 不收带维度的事实。
- `sue` 要等 10-Q 披露才可见（很多公司比财报新闻稿晚几天到几周）；盘前发布的财报，`days_since_earnings` 晚记一天。
- Validate 段单独评估、不带预热历史：窗口长于该段的因子（如 12 月动量在 232 天的 Validate 段上）没有数字。
- 继任主体只登记了 XOM；别的公司重组换 CIK 时会表现为"只有最近几个季度的数据"
  （2026-10-14 起加载报告会把这种票点名，见第三批）。
- 每日前向管线仍然每天向 moomoo 取一次不复权价的尾巴（每只票一次请求，约 1 分钟）。
- 变异点未做逐点复核（见 `MUTATION_LEDGER.md`）。

### 第三批（2026-10-14）—— 不依赖待决事项的部分

待你拍板的事集中登记在文首「待你确定的事项」（D1–D11）。下面这些与它们无关，已做：

| 项 | 做了什么 | 位置 |
|---|---|---|
| 上线预检认识基本面 | 新增第 16 项 `fundamentals`。此前现役策略若用到 `earnings_yield` 而开关没开，预检照样全绿，到了每日管线里因子才因"字段不存在"报错、当天不调仓。现在：策略用到基本面字段时，开关没开 / 没配 `EDGAR_USER_AGENT` / 本地未过期事实不足一半宇宙 → **阻断**；策略用不到时只提示。只读配置和本地事实，不向 SEC 发请求 | `execution/golive.py` |
| 历史异常短的票被点名 | 事实的最早披露日比面板起点晚一年以上（新上市，或像 XOM 那样重组换了申报主体）→ 记入加载报告的 `short_history` 并告警。末日覆盖率看不出这件事（XOM 当时就是覆盖率正常、前五年整列为空） | `fundamentals/panels.py`、`service.py` |
| 每日管线入口的测试 | 从 `run_daily_pipeline`（调度任务调的入口）验证：开启时两个交易循环拿到的面板都带基本面字段；关着只有价格；基本面挂不上时管线照常跑完 | `tests/integration/test_fundamentals_e2e.py` |
| FE-10 后半 | 待批准队列里，候选因子用到基本面字段时列出字段名并附可见规则（`/api/alphas/pending` 多了 `sparse_fields` / `visibility_rule`，由解析树得出，不是找子串）。库里有解析不了的表达式时队列照常打开 | `api/router.py`、`alpha_engine/fields.py`、`frontend/.../ApprovalQueue.tsx` |

另外核实了一件事（写进了 D2）：IC 错位只在回测报告那条路上。每日监控的 IC（`alpha_ic_history`，→ACTIVE 门的证据）
与 GP 适应度的 IC 在纯随机游走上都是 0，没有同样的问题。

---

## 外部审查 2026-10-18 修复记录（按模块）

报告：`audit-review/审查报告_2026-10-18.md`（F01–F14，复现 R1–R13）。每条都先用审查的复现脚本确认在 HEAD 上成立，
再修；回归用例里有一部分就是把复现改写成"修复后应有的行为"。变异点未逐点复核（见 `MUTATION_LEDGER.md` 第七批）。

| 模块 | 发现 | 修复 | 位置 |
|---|---|---|---|
| 数据 | F01 健康门把缺失删掉再评分（R10：95% 缺失、成交量全零得 1.0 分） | 按 日期 × 股票池 完整网格检查（不依赖 `stack()`），OHLCV 都送进去；综合分之外加**硬性条件**（末日覆盖率 ≥ 90%、整天缺失的交易日对照交易所日历、末日零成交量 ≤ 10%、末日 OHLC 合法）；检查器返回空 → 拒绝（此前按满分放行） | `health_report.py`、`dataset_registry.py`、`daily_ingest.py`、`/api/datasets/{name}/health` |
| 数据 | F02 不同复权 vintage 直接拼接（R4：拼出 −50% 的假收益） | 增量摄取比对重叠那一根的复权收盘价（相对差 > 1e-6）与 PIT 元数据里的价格源；任一变化 → **整段重取**作为新 vintage 追加，旧 vintage 原样保留。PIT 追加加跨进程文件锁 | `daily_ingest.py`、`pit_store.py`、`db/process_lock.py` |
| 研究隔离 | F04 冻结 Test 随数据末日滑动（R1：重叠 521 天，两次都是"第 1 次使用"） | 每个真实数据集第一次切分时把 Test 起点**绝对日期**写进 `frozen_boundaries`，之后一律按它切；holdout 按**区间重叠**计次。有冻结边界却切不出选择段 → `FrozenBoundaryError`（故意不是 ValueError，各入口"太短退回全量"的兜底接不住它）→ 回测接口 422、验证门判不通过 | `data_partitioner.py`、`db/trial_ledger.py`；初始日期见 D12 |
| 研究隔离 | F03 提案 / 工作流 / 发现观察引擎看全量（R2、R3） | 策略提案只拿选择段；Workflow A 默认**不看 Test**（`report_test=True` 才看，且先记账），组合也只在 Validate 上评估；Workflow B 此前对全量两段切、GP 按最后 30% 择优（正是冻结段），改为先切三段；自主发现的观察只看选择段；agent 工具先切研究窗口，`use_test_set` 记账；每日策略门的结论标 `evidence_scope=full_pit_window…`、`holdout_clean=false` | `router.py`、`alpha_workflows.py`、`discovery_engine.py`、`agent/_tools.py`、`daily_trading_loop.py` |
| 策略规格 | F07 审批、验证、模拟、执行不是同一个策略（R8：批准 equal_weight / AUM 1234 / band .123，每日门按 ic_weighted 拟合、账本用 AUM 1e6 / band .00134）；R7 波动缩放乘两遍 | 新增 `ExecutableStrategySpec`（`executable_spec.py`）：提案时把成分表达式、冻结权重、方法、AUM、无交易带、风控上限、信号参数、券商 / 做空、数据集 / 价格源收进一份规格并算哈希，存进 verdict。每日循环照规格执行、核对其余参数，规格缺失 / 被改过 / 对不上 → 不调仓（见 D13）；每日策略门评估冻结权重 + 冻结带；风控上限只在一处装配（此前提案的风控快照用 dataclass 默认）；无交易带之后的第二次风控只做硬约束投影，不再传波动估计；上线预检核对规格 | `executable_spec.py`、`strategy_builder.py`、`strategy_gate.py`、`daily_trading_loop.py`、`golive.py` |
| 前向证据 | F06 前向 IC 可被回放改写（R12）；停机补拉的 bar 全标前向；重跑时 `forward_from=None` 丢标；组合 IC 全在 book 0、→ACTIVE 门借别的策略的数 | 新增 `evidence_store`：每次运行把当天信号**先存档**（只写一次）；某天的 IC 只有用"前一根 bar、且在当天收盘前就存好"的那份信号结算出来才算前向 —— 补拉的、重跑的、离线补算的自然只算回放，而重跑不再丢标。前向行写入后不可改，改写尝试记 `evidence_revisions`。每份带规格的策略（proposed / approved / active）按自己的规格存档、结算，记在 `strategy_evidence`（id + 规格哈希 + 信号日期 + 决策时刻）；→ACTIVE 门只读这份策略自己的记录，批准之前也在积累 | `evidence_store.py`、`alpha_store.py`、`daily_trading_loop.py`、`daily_ingest.py`、`router.py` |
| 执行 | F08 重试意图可被两个会话同时认领（R11）；进程内锁挡不住调度器 / run-now / API / 恢复任务 | 认领改为**带条件的 UPDATE**（WHERE 状态可重试，看受影响行数）或靠 client_id 唯一键的 INSERT，同一时刻只有一个赢家；OrderManager 的对账 / 接受差异 / 调仓 / 维护 / 熔断都在**账户级跨进程文件锁**里做（锁文件在执行库同目录 `.locks/`），拿不到 → BrokerError（本轮不动账户）。开熔断先落状态再拿锁，账户忙也不会丢 | `execution_store.py`、`order_manager.py`、`db/process_lock.py` |
| 执行 | F09 新开仓没有独立报价校验（R5：空账户第一笔 BUY 拿自己的参考价给自己估值） | 网关新增 `quotes()`（moomoo 行情快照：最新价，缺则买卖中间价，带报价时刻）；下单门对所有待交易标的核对：持仓里没有价的票没有报价 → `no_independent_quote`；报价超过 `EXEC_MAX_QUOTE_AGE_DAYS`（默认 4 个自然日）→ `stale_quote`；与参考价偏差超过上限 → `quote_mismatch`。取报价失败 → 只放行减仓和已有持仓的单；全平从不被报价拦住。测试替身默认**没有**报价 | `broker_gateway.py`、`pretrade_gate.py`、`order_manager.py`、`config.py` |
| 账务 / 成本 | F10 两套引擎都直接拿昨日权重与今日目标做差（R6：50/50、A 涨 10%，回到 50/50 的调仓成本记成 0） | 新增共用原语 `drift_weights`：先把昨仓按今日价格漂移成调仓前权重（w·(1+r)/(1+Σw·r)），再撮合到目标；成交差额、换手、成本都从漂移后算。PaperBroker 与回测引擎仍逐日一致，并用现金 + 股数的手算账本对过 | `transaction_cost.py`、`paper_broker.py`、`backtest_engine.py` |
| 账务 / 成本 | F11 成本缓存键只看形状 / 端点 / 首末行（R13：改中间的 high 键不变，价差 12.85 → 1584 bps）；券商键读错了字段 | 键 = 成本输入面板的**全量内容哈希** + 推导真正读取的 `trading_broker` / 账户类型 / 能否做空 | `strategy_gate.py` |
| 账务 / 成本 | F12 min_variance 用信号截面均值估协方差（R9：a、−a、a 各 1/3） | 改为**因子组合收益**（截面去均值、按 Σ|w| 归一的美元中性组合）的协方差，向对角收缩 20%；没有收益序列 → 显式等权并告警 | `alpha_combiner.py` |
| 基本面 | F13 SUE 与 8-K 公告错配；股本兜底口径不可见；拆股启发式；过期按最近一次披露统一算 | 新公告已出、对应申报未到的那几天 SUE 置空（真实数据上每只票每年约 6–8 天），字段说明改为"申报后可得的标准化盈利变化"（D8 的建议）；各字段按自己那条事实的期末日过期（`MAX_PERIOD_AGE_DAYS`）；加载报告给出每只票的股本来源；拆股要求不复权价同日反向跳变（否则当数据问题），1.15–1.5 倍的跳变只告警不当拆股。`METRICS_VERSION` 2 → 3，基本面指标缓存会重算一次 | `fundamentals/panels.py`、`prices.py`、`fields.py`、`agent/_constants.py` |
| 运营 | F14 审批 / 退役 / 激活 / 谱系分别提交；没有独立激活接口；绑定检查只看配置；IC t 没考虑序列相关；PBO 范围不写明；前端按钮没有逐项锁、取消拒绝原因仍会拒绝；CI 的 `tsc --noEmit` 什么都没查；资金利用率不可见；文档口径不一 | `StrategyStore.apply_transitions` 一个事务完成一组流转 + 谱系（任一步非法整组回滚）；新增 `POST /strategies/{id}/activate`（approved → active，同一个 →ACTIVE 门）；启动自检同时检查命令行 `--host` 与 `UVICORN_HOST`；→ACTIVE 的 t 取普通 t 与 Newey-West t 中较小者；策略门结论带 `pbo_scope`；`purged_cv` 的说明写明是 IS 内分块稳健性；前端逐卡 busy 锁、取消即不拒绝、approved 配置可单独启用、卡片显示规格哈希；CI 改 `tsc -b`；执行报告记录资金利用率；OPERATIONS 新增 §7 运行边界（T−1 信号时钟、熔断日频、告警只有日志），README 同步。**未做**：信号时钟统一（D14）、告警渠道（D15）、盘中熔断巡检、备份恢复演练 | `strategy_store.py`、`router.py`、`main.py`、`promotion_gate.py`、`strategy_gate.py`、`order_manager.py`、前端、`ci.yml`、`OPERATIONS.md`、`README.md` |
| 研究隔离 | F05 只有自主发现按配置估算试验数 | GP 按实际评估过的结构数、Optuna 按实际跑完的 trial 数在**底层**记账（`trial_runs` 表，按 run 幂等；原子 UPDATE）；发现编排不再自己估。台账读不到 → 验证门判不通过（此前退回 1 继续）；`/alphas/{id}/validate` 的 `n_trials` 默认改为全局累计（此前默认 1） | `population_evolver.py`、`alpha_optimizer.py`、`validation_gate.py`、`router.py` |


### 复审 2026-10-19（`review-v6/audit-review/复审报告_2026-10-19.md`）

复审对首轮 14 组的判定：**关闭** F02、F04、F08、F11、F12；**部分修复** 其余 9 组。新发现 N01–N09 全部有复审独立脚本的复现，
本轮逐条修复，回归在 `tests/integration/test_review_1019.py`（由复审脚本的 residual_* 场景改写成断言修复后行为）。

| # | 问题 | 修复 | 位置 |
|---|---|---|---|
| N01 | 一批订单发送途中开熔断，余下的单照发 | 每张调仓单发出前读最新的持久化熔断状态；开了就停下余下的单、转去撤单 + 全平；撤单等终态之后、建单之前再读一次 | `order_manager.py` |
| N02 | 规格与运行时不符时仍为该规格累计前向证据；无新 bar 时换源不检查 | 证据路径也跑 `verify_against_runtime`，对不上 → 隔离（记 `strategy_evidence_quarantine`）不存档不结算；核对用 PIT 元数据里的**实际**价格源；无新 bar 时换源也整段重建 | `daily_trading_loop.py`、`daily_ingest.py` |
| N03 | 结果日收盘前 1 分钟存档的信号也算前向 | 存档截止 = 被评分那一天的**开盘**（`session_open_utc`）。隔夜跳空仍在被评分窗口里，标签口径归 D14 | `evidence_store.py`、`market_calendar.py` |
| N04 | 健康门检查截断前的面板，实际落盘 / 使用的是截断后的 | 检查截断后、真正被使用的那份；另加持仓标的检查（券商账本里持有的票末日必须有收盘价，不能被宇宙平均覆盖率抵消） | `daily_ingest.py` |
| N05 | 提案的门评估的不是随后保存的冻结规格；门与日循环风险路径不同构 | 提案先冻结权重与带、再让门评估**这份规格**；另跑一次流程的滚动前推 OOS（`procedure_oos`），两个都过才算过，并标 `evidence_scope=frozen_spec_on_selection_in_sample`；波动估计与带后硬约束投影抽成共用函数，门与日循环都调它 | `strategy_builder.py`、`strategy_gate.py`、`executable_spec.py`、`daily_trading_loop.py` |
| N06 | 报价无时间仍可买；已有持仓无独立报价可加仓 | 加风险的单（新开、加仓、反手）必须有有限正价 + 报价时刻 + 不在未来 + 不早于最近一个已收盘交易日 + 与参考价对得上；减仓 / 平仓不需要报价（保守通道） | `pretrade_gate.py` |
| N07 | 试验写失败 / 进程被杀后门仍按残缺历史给结论 | 搜索开始前登记运行（登记不进去就不搜）、逐代 / 逐 trial 记账、结束置 done / aborted；停在 running 超过 6 小时 = 试验史不完整 → 验证门与策略门不给正式结论，直到 `resolve_run` 补记并写明理由；去掉了一次性的 `record_trials` | `trial_ledger.py`、`population_evolver.py`、`alpha_optimizer.py`、两个门 |
| N08 | 漂移修好后费用仍乘昨日权益 | 成交额 / 成本 / ADV 上限都按**调仓前市值**换算美元，当天净收益按期初权益归一；回测把撮合并进逐日净值循环 | `paper_broker.py`、`backtest_engine.py` |
| N09 | CLI GP 绕过冻结边界；工作流接口的 PnL 切分压在冻结段上 | 新增共用 `research_window()`；CLI 的 GP / 回测 / agent / realistic 与 `_resolve_dataset` 都先限定研究窗口 | `data_partitioner.py`、`main.py`、`router.py` |

复审其余意见的处理：路线图文首状态、D10、D11、D12、D13 按复审口径改写（不再追加互相矛盾的脚注）；README 的统计表述收敛。
**仍未做**：D14 信号 / 成交 / IC 时钟统一、D15 告警渠道、盘中熔断巡检、备份恢复演练、真 OpenD 人工验收（D10）、
规格哈希覆盖有效股票池 / 数据 vintage / 代码版本、账户锁按券商账户 id 建锁域、基本面股本 fallback 与 SUE 的期间配对（价量首轮关闭基本面可隔离）。
---

## Phase 11 — 前向增量数据 + PIT 追加 + 市场日历

**目标**：从历史回放转为真正的前向 paper trading（日频收盘增量）。**价格源 = moomoo（TR.2）**，与执行同源。

- **11.1 provider 增量拉取 ✅**（2026-09-03）— `MoomooProvider.fetch_latest` 已有；新增
  `PITStore.latest_timestamp(name)`（只读末年分区取最新 bar 日期，不加载全量）。
- **11.2 美股日历 + 时区 ✅**（2026-09-03）— 新建 `data_engine/market_calendar.py`（依赖
  `pandas_market_calendars`）：`is_trading_day` / `last_/next_trading_day` /
  **`session_close_utc`（DST 感知：夏令时 20:00 UTC、冬令时 21:00 UTC、半日市 18:00 UTC）** /
  `minutes_after_close_utc` / **`cross_check`（数据↔日历双向校验，不静默）**。
  **双权威**：数据是权威（有没有新 bar 以 provider 为准），日历用于提前调度与交叉校验告警。
  调度器 `daily_trading_job` **非交易日直接跳过**，不空转。
  *（顺带修：原写死 21:00 UTC 只在冬令时才是收盘后，夏令时/半日市都会错。）*
- **11.3 `daily_ingest` 重构 ✅**（2026-09-03）— `DailyIngest.ingest_incremental()`：
  PIT 空 → 一次性历史回填（`mode=full`，计为**回放**）；非空 → 只拉 `last+1..今天`
  （`mode=incremental`）；无新 bar → `no_new_bar` 不写库不交易。**只追加增量**进 PIT。
  `run_daily_pipeline(..., incremental=True)` 为默认。
  *量化收益*：旧实现每天整段重写、去重键含 `as_of` 故不去重 → **单次 665,000 行、一年 1.68 亿行**；
  真增量为 **665 行/日**，**膨胀 1000× 已消除**。
- **★ 回放/前向分离 ✅**（2026-09-03，本 Phase 关键补充）— `alpha_ic_history` 新增 **`is_forward`** 列
  （SQLite `ADD COLUMN` 安全幂等迁移，旧库自动补列）；`record_ic(..., is_forward=)` **只升不降**
  （回放重跑不会抹掉已积累的前向证据）；`get_forward_ic()` 只取前向样本；
  `run_portfolio(..., forward_from=)` 按摄取到的首根新 bar 标记。
  **TR.4 的 →ACTIVE 门改为只吃前向样本** —— 这解除了它此前"只能仅记录不拦"的根本限制。
  > **2026-10-18 起被外部审查 F06 取代**：按 `forward_from` 标记会把停机后补拉的 bar 全标成前向、
  > 重跑时又会丢标；"只升不降"保护了标签、没保护数值。现在前向 = 结算用的是结果发生前就存档的信号
  > （`evidence_store`），前向行写入后不可改，→ACTIVE 门读策略版本自己的记录。见文首修复记录。
- 测试：`test_phase11_forward.py`(8) + `test_phase11_incremental.py`(4)。
- **验收（待真实运行）**：连续 N 个真实交易日，PIT 每日增一根 bar；日循环消费增量；A1 成立。
  **前提：OpenD 需长期在线**（每个交易日收盘后拉）。

---

## Phase A — 外部审计整改（**数据契约**）✅（2026-09-05）

**背景**：一份外部审计指出"聊天路径恒跑合成数据且屏幕上无任何标识"。逐条核对代码后
**7/8 属实**（且比审计说的更糟：`QuantAgent.__init__` 当时连 `dataset_name` 形参都没有，
所以调用方**根本无法**传真实数据集），1 条不实（A股/港股 registry 仍在）。整改如下：

- **A.1 fail-closed 数据契约 ✅** — `QuantTools.__init__(..., allow_synthetic=False)`：
  未指定 `dataset_name` 或真实数据加载失败时 **直接 raise**，绝不静默用随机游走冒充真实
  数据。要合成必须显式 opt-in。`QuantAgent` 透传 `dataset_name/start/end/allow_synthetic`。
- **A.2 生产默认走真实数据 ✅** — `chat_router._get_agent` 默认 `settings.default_dataset`；
  新增配置 `chat_dataset` / `chat_allow_synthetic`。`GPEvolveRequest.dataset_name` 默认
  由 `""`（=合成）改为 `"us_tech_large"`。
- **A.3 数据来源全链路披露 ✅** — `data_source`（`"real:<name>"` / `"synthetic"`）经
  **POST `/api/chat`** 与 **SSE `/api/chat/stream` 的 `done` 事件**双路返回；前端
  `ChatMessage` 在指标 chips 前渲染徽章：真实=绿色、合成=琥珀色 **"⚠ 合成数据（指标无效）"**。
  *（差点只修 POST 一条路——前端实际消费的是 SSE，见 DEV_LESSONS §R。）*
- **A.4 台账不再被合成数字污染 ✅** — `tool_save_alpha` 在合成数据下**把指标清零**、
  hypothesis 打 `[synthetic]` 前缀、`reasoning` 存 `{data_source, raw_metrics}` 留痕。
  *（历史残留：用户库中 521 条非零 Sharpe（98 条 >1，25 条 >2，最大 3.44）全部来自
  开发期合成数据，已全部 retire。门控本就从 DSL 在真实数据上重算，但**人**看的是库里
  存的数字——这才是真实危害。）*
- **A.5 不变量测试 ✅** — 新建 `tests/test_invariants.py`(8)：拒绝无 opt-in 的合成、
  加载失败必须 fail-closed、合成必须带标识、API 模型默认真实数据集、两条聊天路径都带
  `data_source`、**从入口 BFS 无孤儿模块**（防"写了没接线"）。
- 顺带修：`alpha_workflows.py` 用 `m.get('is_sharpe', 0):.4f` 格式化 —— 键存在但值为
  `None` 时 `TypeError`，导致 GP 全部评估失败时聊天流以 500 结束而非返回消息。
- 回归：后端 **585 passed / 4 skipped / 0 error**；前端 `tsc -b` 干净 + vitest 94 passed。
- **未整改（已知且已在路线内，非本轮范围）**：fitness=OOS Sharpe 的选择性挖掘、OOS 分段
  不连续、执行层仅 PaperBroker（Phase 12）、`vwap≈(H+L+C)/3`、universe 写死无幸存者偏差修正。

---

## Phase B — 全量测试审计 + DEV_LESSONS 强制化（2026-09-06）

**背景**：外部审计连续两轮命中真问题，而本仓库"562 passed"一直全绿。逐条追因后发现
**不是不够仔细，是三个机制性盲区**（详见 DEV_LESSONS §S）。本 Phase 对 64 个测试文件、
589 个用例做 AST 级审计，把"不会失败的测试"全部收紧，并把教训编排成强制检查。

### B.1 审计手法（先量化，再动手）
AST 扫描全部 `tests/`，按"这个断言可能失败吗"分类，查出 **90 处反模式**：
`accepts_5xx` 16 · `conditional_assert` 30 · `except_swallow` 13 · `no_assert` 12(净) ·
`body_wrapped_in_try` 2 · `assert_true` 1 · 其余为 4xx 宽容。

### B.2 收紧断言后暴露的**真实问题**（此前全部被绿灯掩盖）

| # | 问题 | 后果 | 状态 |
|---|------|------|------|
| B-1 | `/api/backtest/walk_forward` 合成路径**结构性不可用**：端点写死 `n_days=120`，而 `WalkForwardBacktester.min_train_days` 也是 120。实测请求模型允许的 **`n_splits∈[2,10]` × `embargo∈[0,60]` 全 12 种组合均 `ValueError` → 500** | 不是"某些参数下失败"，是**永远不可能成功**；旧断言容忍 500 故一直"通过" | ✅ 已修 |
| B-2 | 同端点**静默忽略** `n_tickers/n_days/seed`（`WalkForwardRequest` 无这些字段，`_resolve_dataset` 里写死 20/120/42） | 调用方以为自己控制了数据规模，实际没有 | ✅ 已修 |
| B-3 | `test_api_workflow.py` 全部 8 个用例载荷违反 `ge=` 约束 → 端点**从未被调用过**（一直 422 而断言容忍 422） | `/api/workflow/generate` 与 `/optimize` 零集成覆盖 | ✅ 已修（载荷合规，8→9 用例真跑） |
| B-4 | `/api/gp/evolve` 同型：`pop_size=3 < ge=5` → 一直 422 | GP 端点零集成覆盖 | ✅ 已修 |
| B-5 | `DataPartitioner` 默认 `embargo_days=20` **全部从 OOS 段扣除**，且无最小样本保护：n_days=60→OOS **0 行**（下游 500）；n_days=80→OOS **4 行** | 4 行数据算出的年化 Sharpe 被当正常结果返回；或直接 500 | ✅ 已修 |
| B-6 | 承 B-5：`/api/workflow/generate` 在 120 天合成数据上报出 **OOS Sharpe = 15.78**，且过拟合检测判为 `healthy (0.0000)` | 金融上不可能的数字被当作正常结论展示 | ✅ 已修 |
| B-7 | **`ind_neutralize` 行业中性从未生效**：parser 把分组存进 `params["groups_node"]`，`CrossSectionalNode._compute` 读的是 `params["groups"]` → 恒 `None` → 退化成 `cs_zscore`。给不给 `sector` 输出**逐位相同** | 任何用 `ind_neutralize` 的因子都不是行业中性的。兄弟算子 `sector_neutral` 正常 | ✅ 已修 |
| B-8 | 承 B-7：缺 `sector` 字段时静默退化，**无告警** | 使用者以为做了行业中性 | ✅ 已修 |
| B-9 | `tests/unit/test_backtest_edge_cases.py` 的 `_run_backtest` 少传必填参数（`run(weights,prices,volume,signal)` 写成 `run(signal,close,volume)`），**每次 TypeError**，被 `except: pass` 吃掉 | 整个边界值测试文件长期空跑 | ✅ 已修（修正后又查出 2 个真问题） |
| B-10 | `POST /api/backtest/run` **缺 `dsl` 字段仍返回 200 + 完整报告**（静默套用默认 DSL） | 调用方以为测的是自己的因子 | ✅ 已修 |
| B-11 | 写死合成数据且无 `dataset_name` 入参的端点，属性级扫描查出 **4 个**（外部审计只报了 1 个）：`backtest_realistic` / `backtest_multi` / `alpha_simulate` / `alpha_optimize` | 返回的 Sharpe/风险报告全是随机游走 | ✅ 已修 3/4（realistic/simulate/optimize）；`backtest_multi` 仍在台账 |
| B-12 | **6 条 API 路由零测试覆盖**：`/api/agent/run`、`/api/backtest/multi`、`/api/paper/{id}/pnl`、`/api/strategies/propose`、`/api/workflow/{generate,optimize}/stream` | `/strategies/propose` 是进审批队列的源头 | ✅ 已修 `/strategies/propose`；其余 5 条在台账 |
| B-13 | `test_agent_fallback.py` 的两个意图识别用例 import 了不存在的模块级 `_detect_intent`（实为实例方法）→ `except(Exception)` → **无条件 skip** | 意图识别从未被测过 | ✅ 已修 |

### B.3 已完成的整改
- **审计 #4（目标波动不生效）✅** — `daily_trading_loop` 现按已实现组合收益估计年化波动
  （权重 t-1 · 收益 t，避免同日自指），样本 <20 日则不缩放并告警；新增 `risk_vol_lookback`。
  `PortfolioRiskGate.apply(..., port_vol_ann=)` 至此在实线路径上真正接通。
- **审计 #9（静默吞异常）✅** — `strategy_builder` / `risk_gate` / `promotion_gate` 的
  9 处裸 `except: pass` 全部改为记录 + 留痕；新增 **`verdict["degraded"]`** 字段：
  任何一步降级产出都写进策略配置并持久化，审批者能看出"这份配置是完整评估的还是带窟窿的"。
  策略门崩溃与策略门未过现在**明确区分**（`{"gate_error":..., "evaluated": False}`）。
- **审计 #11（提示词数据集名不实）✅** — `_lc_agent` 的 `us_sectors`/`cn_ashares` 改为注册表实有名。
- **上轮 Phase A 的漏网第三处 ✅** — `tool_run_gp_optimization` 的 per-run 数据集覆盖此前
  加载失败时静默改用会话数据**且 `data_source` 不变**（前端徽章会说谎）。现改为 fail-closed；
  显式允许合成时，来源标识同步降级为 `...+fallback(gp:X失败)`。

### B.4 DEV_LESSONS 强制化（本 Phase 的主交付）
新建 **`tests/test_lessons_enforced.py`**（20 passed + 1 xfail），把散文教训变成会失败的检查：

| 教训 | 强制检查 |
|------|----------|
| §A | 禁 `assert status_code in (...,500)`（502/503 属设计内上游失败，放行）；禁 `if status_code:` 包住断言；禁整体 `except: pass`；禁 `assert True` / `assert x or True` |
| §B | 门控开关必须存在于 config（防改名成死分支）；门控模块的 `except` 必须记录或抛出 |
| §E | LLM 可见的提示词里出现的数据集名必须在注册表中真实存在 |
| §F/§I | 无 opt-in 不得用合成数据；数据加载失败切源时**必须同步更新来源标识**；POST 与 SSE **两条**聊天路径都要带 `data_source` |
| §H | conftest 每处 `settings` 覆盖必须有注释说明（每个覆盖 = 一个测试盲区）；**必须存在跑在发布配置下的测试** |
| §K | 模块级可达性 **+ 参数级可达性**（关键形参必须有真实调用点传值 —— 上一版只查到模块粒度，`port_vol_ann` 正是从那底下走过去的） |
| §O | 禁止夹具用固定 ±1% 构造 H/L |
| §Q | 活库路径不得落在云同步目录 |
| §S | 新增路由必须有测试或进债务台账；端点不得新增"写死合成数据"形态 |
| §R | **元规则**：扫 DEV_LESSONS 标题，每条都要有检查或在 `EXEMPT` 写明为何无法自动化 —— 防止本文件重新退化成散文 |

新建 **`tests/test_production_defaults.py`**（7 passed，标记 `production_defaults`）：
不覆盖 settings，只读 `Settings` 类默认值（用户实际拿到的配置），断言"配置→行为"链是通的。
门控默认值写成**显式快照表**，任何变动都必须有人改这张表 → 在 review 里被看见。

**债务台账机制**：已知未修项记录在 `KNOWN_*` 集合里，配 `no_new_*` 用例防新增，
外加一条 `xfail(strict=True)` 的 `test_debt_ledger_is_empty` —— 修完却忘删台账行会 XPASS 报错，
强制台账与现实同步。

### B.5 第三批整改（2026-09-06，收尾）✅

- **`backtest_multi` ✅** — 复核后**推翻上一版定性**：它有显式 `use_synthetic: bool = False`
  开关、默认走真实数据，**不是**"写死合成"。是检查器只认 `dataset_name` 一个字段名造成的
  误报，判据已改为"有无显式的真实/合成选择开关"。真实缺陷只有两处，已修：
  响应新增 `data_source`；数据集加载失败由 **500 改 502**（与 `_resolve_dataset` 同口径）。
- **5 条零覆盖路由全部补齐 ✅** — 新建 `tests/integration/test_api_uncovered_routes.py`(14)：
  `/api/agent/run`、`/api/backtest/multi`、`/api/paper/{id}/pnl`、
  `/api/workflow/{generate,optimize}/stream`。其中 SSE 两条专门断言
  **流式与非流式同一样本充足性口径**（防"只修 POST 不修 SSE"的 §R 复发）。
- **`/api/agent/run` 结论歧义 ✅** — `final_dsl=""` 此前既可能是"所有候选没过 IC-IR 门"
  也可能是"agent 出错返回空 log"，调用方无从分辨。新增 `passed: bool` + `data_source`。
- **SSE workflow 静默不落库 ✅** — 存 alpha 的 `except Exception: pass` 改为记录错误并在
  响应里带 `persist_error`：跑完却没入库不能悄无声息。
- **债务台账已清空** — `KNOWN_UNTESTED_ROUTES` / `KNOWN_SYNTHETIC_ONLY_ENDPOINTS` 均为空，
  `test_debt_ledger_is_empty` 由 `xfail(strict)` 转为**必须通过**的正向断言；
  新增未覆盖路由或新增"写死合成"端点都会让它变红。

### B.6 第四批：审计 #5 / #9 / #10 整改（2026-09-07）✅

**#5 研究路径健康门 ✅** —— 修的过程中发现比审计所述**更严重**的问题：
`check_dataset_health` 里 `warn_only=False` 抛的 `ValueError` 被同函数末尾的
`except Exception` 当成"检查器故障"吞掉并返回 `None` —— **fail-closed 开关从未生效过**
（实测健康分 0.665 < 阈值 0.99 时返回 None 而非抛错）。
整改：新增 `DatasetHealthError` 并 `except DatasetHealthError: raise` 放行
（判定结果 ≠ 检查器故障）；`load_registry_dataset` 与 ingest 口径统一，
新增 `research_health_fail_closed`（默认 **True**）+ `research_min_health`（0.7）。
⚠️ 该默认值会让此前被静默放行的低质量数据直接报错，逃生口 `RESEARCH_HEALTH_FAIL_CLOSED=false`。

**#10 零认证 ✅（按"本机自用"形态）** —— 正确的防线不是加密码，而是保证服务
**不会在无人察觉时被绑到对外地址**：CORS 由 `["*"]` 收紧为 localhost 白名单
（`*` + `allow_credentials=True` 是危险组合）；新增启动自检 `_assert_safe_bind()`，
绑非回环地址即**拒绝启动**，除非显式 `ALLOW_INSECURE_BIND=true`。
前端走 vite proxy（dev server 转发），不触发 CORS，不受影响。

**#9 静默吞异常 131 → 97**（棘轮锁定，只许降不许升）。按危害而非数量整改：

| 类别 | 具体问题 | 后果 |
|------|----------|------|
| **兜底方向反了** | `manager.py` 读不到 `trading_allow_short` → `long_only=False`（**允许做空**） | 现金账户会构造出无法成交的空头腿 |
| | `discovery_engine.py` 读不到 `factor_gate_mode` → `"leak"`（**松门**），与自身 docstring "fail-closed：出错视为不通过"矛盾 | 门失败时反而更宽松 |
| **门变松** | `validation_gate.py` 读不到试验台账 → `n_trials=1` | DSR 未做多重检验校正，**偏乐观** |
| **空仓假象** | `trading_context/providers.positions()` 读失败 → 返回 `{}` | 下游把全部权益当可用买入力，**按满额重新建仓** ≈ 凭空加杠杆。已改为抛错 |
| **账本静默缺员** | `daily_trading_loop` 状态无法解析 → `continue`（×2） | 因子悄悄退出当日交易账本／组合成分 |
| **口径静默改变** | `AlphaCombiner` 协方差奇异 → 退回**等权**；`PortfolioConstructor` 求解失败 → 沿用基准权重 | 调用方以为拿到最小方差/优化权重 |
| **数据静默缺席** | `yahoo_provider` 缺字段 → 全 NaN 面板；`PITStore` 分区名畸形 → 跳过；`dataset_filters` 市值取不到 → NaN 参与筛选 | universe 与数据完整性悄悄改变 |
| **同源被破坏** | `dataset_registry` 读不到 `price_source` → 退回 yahoo | 研究用 yahoo、执行用 moomoo → 重新引入 train/serve skew（TR.2 的全部意义） |
| 其余 | `horizon` 因子丢失、`evaluation_utils` 惩罚哨兵、`alpha_monitor` 状态重归类、`performance_analyzer` 周期分解为空 | 结论不完整而不自知 |

剩余 97 处已逐条分类：滚动窗口数值兜底 13、DB bootstrap 7、数值/类型转换 18、
可选依赖探测 6、GP 变异算子内部 9，其余为影响面较小的 agent/chat 管线。
新增 `test_silent_except_count_does_not_grow` 棘轮：**降了必须同步下调预算**（防回潮）。

**补审此前声明的留白 ✅** —— `tests/golden/`（1e-12 金标准 + 专门验证"金标准真能抓到 1bp
变化"的敏感性测试）、`test_phase6_reproducibility.py`（判别性断言 + 反例）质量良好；
`test_supplementary_fixes.py` 两条弱用例（断言仅 `is not None`）已收紧为
**repr round-trip 结构保真**与**组内均值必须归零**。

### B.7 仍未处理（需你单独决策，非技术阻塞）

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

> **进度（2026-10-06）**：12.1–12.4 + FE-12 **代码完成，离线验收全绿；真 OpenD 端到端未做**（需要本机
> 起 OpenD + 纸交易账户）。默认 `execution_mode="off"`，不开不连券商。
>
> 实际落地与原规划的差异：
> - 文件名：`moomoo_broker.py` 拆成 `broker_gateway.py`（SDK 边界，**写死 `trd_env=SIMULATE`**，遇真实账户
>   直接拒绝）+ `order_builder.py`（账本→股数/限价）+ `order_manager.py`（对账/下单/熔断/恢复）；
>   `risk_gate.py` 叫 `pretrade_gate.py`（避免与 PM 已有的策略级门混名）。执行台账独立成
>   `app/db/execution_store.py`（订单/成交/快照/事件/状态五张表），对账后的持仓仍写回 `PositionStore`（book −1）。
> - 幂等：先落**写前意图**再下单，client id 写进 moomoo `remark`（≤64 字节）；下单调用超时**当天不重试**
>   （等券商侧出现或次日确认未提交），防重复挂单。
> - 12.3：月度任务 `run_monthly_fidelity` 产出保真度报告（实际成交 vs 决策价 / 次日开盘 / 内部模拟），
>   成交 < 20 笔不校准；永久冲击目前**不可识别**，报告里明说。
> - **实测发现**：OpenD 不在时 moomoo SDK 建连会**无限阻塞**并留下非守护线程 → 建连前 2 秒端口探测。
>   同样的问题在行情侧 `MoomooProvider`（`PRICE_SOURCE=moomoo`）也存在 —— 已在上线工具批次修复。
>
> **只能在真 OpenD 上验证的两件事**：纸交易环境是否支持历史订单查询（不支持时对账会保守地挂起
> "待确认"而非乐观判定）；收盘后下的单次日是否出现在当日订单列表。
>
> **上线工具（2026-10-06 第二批）**：`golive.py` 上线预检（`python -m app.tasks.forward preflight` /
> `POST /api/execution/preflight`，9 项逐项检查，会对账一次、不下单）+ `run-now`（预检全过才跑调度器的
> 同一个 `daily_trading_job`）；**启动保险**：前向交易开启时活库/调度库/PIT **解析后**落在云同步目录 → 拒绝
> 启动（默认 `./alphas.db` 从 OneDrive 里的仓库启动就在 OneDrive，原 TestLessonQ 只查字符串看不见）；
> `MoomooProvider` 建连前端口探测（上面那条"本 Phase 未修"已修）；`.env.forward.example` +
> `scripts/start_forward.ps1`。**真 OpenD 端到端仍待做**：需要人在 OpenD 窗口登录。
>
> **外部审计（2026-10-07）→ NO-GO，16 条，全部复现属实；修复批（2026-10-08）**：
>
> | 编号 | 问题 | 处置 |
> |---|---|---|
> | F01 | Yahoo `end` 排他 → 收盘后永远拿不到当天 bar，执行层天天判陈旧 | provider 契约统一为闭区间，Yahoo 内部 +1 天；摄取目标日 = 最近已收盘交易日（与执行层同一定义 `market_calendar.last_closed_session`），丢弃未收盘半根 |
> | F02 | 无新 bar / 摄取被拒直接 return，不对账不续做全平 | no_new_bar 时用 PIT 面板重跑组合 + 执行（幂等）；摄取被拒 / 组合异常只做维护；新增 22:45 UTC 补跑 |
> | F03 | 下单后落库前崩溃 + 券商延迟可见 → 同日重下 | 下单调用**之前**落 `submit_attempted_at`；同一 remark 出现在两张券商单上 → 阻断 |
> | F04 | 撤单受理即下新单 / 全平 | 轮询到终态再重新对账后建单；全平按撤单之后的真实持仓；超时不下单 |
> | F05 | 风控看不见在途单、假设同批卖单全成交 | 最坏情况包络：卖单确认成交前不释放预算；在途买单占买入力；日亏熔断撤在途加仓单 |
> | F06 | max_net 只声明不施加 | 组合层按占优一侧投影；下单层加净敞口与行业上限；限额必须为有限正数 |
> | F07 | 无交易带撤销风控 | 带后再施加并独立 check；券商侧按**实际持仓**判断带宽，超限时减仓不被吞 |
> | F08 | active 配置不可用时改交易其他因子；基准库无审批直接下单 | 券商只交易已激活策略；配置不可用 → 只维护；使用冻结的 combo_weights / method |
> | F09 | 硬门异常放行；非实验模式分级不用 | 异常 / 未过 / 分级不允许都只维护 |
> | F10 | 全样本拟合组合权重再切段称 OOS | 策略门改滚动前推拟合（第 k 段只用之前数据）；实盘用冻结权重 |
> | F11 | 增量窗口首日 returns 为 NaN | 增量窗口重叠上一根 bar，只写入新增部分 |
> | F12 | PIT 原地覆盖 | 临时文件 → 读回校验 → `os.replace`；首次回填写失败即拒绝 |
> | F13 | 预检 ready 名不副实 | 分 `connected` / `ready` 两层；新增调度任务、存储可写、数据集、已激活策略、**真实取数**检查 |
> | F14 | run-now 失败也返回 0 | 按真实结果给退出码（执行被拦 / 下单异常 / 组合报错 / 摄取被拒 → 3） |
> | F15 | ACTIVE 门借用他人前向证据；先改状态再 409 | 门先于状态变更；只计本策略创建之后的 IC；激活新策略退役旧的。**部分修复**：组合账本仍是一个，未按策略版本分账 |
> | F16 | 因子级 IC 不标前向；PBO 异常即豁免 | 透传 is_forward；PBO 失败判不通过，样本不足显式标注 |
>
> 同批修正：`.env.forward.example` 按 TR.2 改为 `PRICE_SOURCE=moomoo` + `us_broad_large`（上一版照抄了
> 代码默认值）。测试：16 条审计复现全部改写为正确行为回归（`test_audit_order_regressions.py`、
> `test_audit_portfolio_regressions.py` 等），新增**多日端到端** `integration/test_forward_daily_e2e.py`
> （yfinance 契约替身：end 排他 + 数据延迟发布；券商悲观替身：异步撤单 + 新单延迟可见）。
> **仍未做**：真 OpenD 上的完整交易日闭环；按策略版本分账（F15 余项）；审计"设计与运营问题"
> 7 条（复权口径、鉴权、跨进程锁、外部告警、静态 universe、前端竞态、性能合同）未在本批处理。
>
> **首次接上真 OpenD（2026-10-11）**：
> - 预检 15 项在真实纸账户上全过（模拟账户自动选定、未碰实盘账户；纸账户 $1,000,000、买入力 2×）。
> - **两件只能在真环境确认的事，已确认一件**：纸交易**支持历史订单查询**（对账不会卡在"待确认"）；
>   moomoo 日 K 的区间为闭区间、收盘后能取到当日 bar。另一件（收盘后下的单次日是否出现在
>   当日订单列表）要等第一笔真实订单之后才能确认。
> - **资金切片**：纸账户 $1M、`PAPER_AUM` $10k —— 原先"账户资产必须 ≈ PAPER_AUM"的检查让这种
>   配置无法交易。改为只动用 `PAPER_AUM`：切片权益 = `PAPER_AUM` + 账户自基准以来的盈亏，股数 /
>   风控 / 日亏熔断 / 净值全部按切片算，切片不加杠杆；`exec_max_aum_mismatch` 删除。
> - **第一份策略**：一轮自主发现（3 家族，小规模 GP）产出 2 个候选，经济含义站不住（股价 / 成交量的
>   绝对水平），未采用。改用经典基准库走正式流程：8 个经典因子入 PAPER（其中 7 个只跑影子账本作
>   前向对照），边际准入只留下 12 个月动量；**策略门未过**（t=1.80、5 段 3 正、DSR=0.145），分级 B，
>   用户批准作为**实验策略**激活（策略 #1），目的是跑通链路并收集前向证据，不是已验证的策略。
> - **仍未做**：第一笔真实纸交易订单与次日成交对账（完整交易日闭环）。
>
> 测试：`tests/unit/execution/*`、`test_execution_store_schema.py`、`test_live_providers.py`、
> `test_monthly_fidelity.py`、`integration/test_api_execution.py`、`integration/test_phase12_execution_wiring.py`；
> 新增变异点逐点复核 441 点（430 杀 / 5 等价 / 6 导入期即崩），见 `MUTATION_LEDGER.md`。

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

  #### 进度（2026-09-22）：**2 和 3 已完成并接线；1 和 4 未做**

  已建 `app/core/risk_engine/factor_model.py`：
  - **风格暴露** `build_exposures`：动量(12-1，跳过近月)、反转、波动、流动性(Amihud)、
    市场 beta，逐日截面 z-score + ±3σ 截断；行业哑变量取自数据集的 `sector` 字段。
  - **因子收益**：逐日截面回归 `r_t = B_t·f_t + u_t`（`np.linalg.lstsq`，自由度不足的
    截面跳过；有效天数 < `min_obs` **直接抛错**——10 天估出来的协方差与 400 天的长得
    一模一样，只是错的）。
  - **结构化协方差** `RiskModel.covariance()`：`Σ = BΣ_fBᵀ + D`（= 第 3 项）。
  - **风险归因** `RiskModel.attribute()`（= 第 2 项）：方差拆成因子/特异，因子部分按
    **Euler 分解**摊到各因子（含交叉项，合计恒等于因子方差；实现成"各因子单独方差
    相加"会漏掉因子间相关性、合计对不上）。
  - **接线**：`run_portfolio` → `result["risk_attribution"]` → `DiagnosticsStore` →
    `GET /api/portfolio/diagnostics` → 前端交易现实面板。**只读，不改任何交易决策。**
  - 测试 `tests/unit/risk_engine/test_factor_model.py`(23) + 接线 1 条。核心是三条
    **恒等式**（逐位核对，不是近似）：`factor_var + specific_var == total_var`、
    `Σ by_factor == factor_var`、`w'Σw == total_var`。风险归因的危险在于**它永远
    给得出一个数字** —— 样本太短的协方差、漏掉行业哑变量的分解、丢了交叉项的逐因子
    贡献，产出的 JSON 和正确结果长得一模一样。

  **⚠️ 本轮的诚实边界（已写进返回值，不只写在文档里）**：
  - **size 与 value 缺席**：需要股数/账面价值等基本面数据，Phase 10 之前拿不到。
    （**2026-10-13 更新**：数据集开启基本面且覆盖率够时，这两个风格已进模型；`styles_missing`
    改为按实际情况给出并附原因。未开启基本面的数据集仍如下所述。）
    每份归因都带 `styles_missing: ["size","value"]` —— 否则"因子风险占比 66%"会被
    读成"覆盖了主要风格"。所以当前的分解应读作"在这 5 个价量风格 + 行业上的分解"。
  - **第 1 项（风格中性化替换）未做**：它会**改变信号**（残差化替代 rank/行业 demean），
    属于另一次变更，需要单独验证对因子表现的影响。
  - **第 4 项（alpha vs 风险溢价区分）未做**：依赖第 1 项的残差。
  - **B6（beta 中性数学不闭合）仍未闭合**：按 PM.5 的既有决策，留待开启做空之后 ——
    当前 long-only，net=gross，beta 对冲不适用。
  - 无 `sector` 字段时行业哑变量缺席，**行业共同风险会被算进"特异"里**（低估因子风险）。
    这时会记 warning，`sector_included=False` 也随结果返回。

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
- **FE-10 另类数据浏览 ✅**（2026-10-13 / 10-14，配 Phase 10）：DatasetView 增加基本面/财报字段浏览 +
  **发布时点**标注（`FundamentalsPanel`，数据来自 `/api/datasets/{name}/fundamentals`）；
  因子用到的稀疏字段与可见规则显示在**待批准队列**里（人做决定的地方；仓库里没有独立的因子详情页）。
- **FE-11 前向数据状态**（配 Phase 11）：每日增量摄取状态、PIT 逐日增长指标、美股日历/最新 bar
  时间（区分"已收盘入库" vs "待摄取"）。
- **FE-PM 组合与配置视图 ✅**（2026-08-31，配 Phase PM）：新建 `frontend/src/components/portfolio/PortfolioView.tsx`
  （侧栏 Portf 入口）——**策略配置控制台**：成分+每因子配额条、策略门 verdict(Sharpe/DSR/t/PBO)、风控
  (单票削/行业缩/gross缩/vol缩放)、换手/无交易带、状态徽章、审批谱系;**"提出新策略配置""批准/批准并启用/拒绝"**
  按钮,消费 `/strategies/{propose,pending,list,{id},approve,reject}`。前端构建绿 + 94 测试过。
  **补充（2026-09-02）**：策略卡已显示 TR.4 的**进 PAPER A/B/C 等级徽章**。

- **FE-TR 交易现实面板 ✅**（2026-09-02，配 Phase TR；原前端路线漏了 TR，本轮补齐）
  **已实现**：`components/portfolio/TradingRealityPanel.tsx`，挂在 Portf 视图的 **"交易现实" 标签页**
  （与"策略配置"并列）。**前置也已补**：新建 `db/diagnostics_store.py`（`portfolio_diagnostics` 表，
  每轮 run_portfolio 诊断只增不改地持久化）+ 端点 `GET /api/portfolio/diagnostics`、`GET /api/trading/status`；
  `run_portfolio` 新增 `trading_context` 摘要（TR.1 价差/可交易/可做空/带）并落库。
  面板四块：①数据源(价格源/同源告警/OpenD 连通/券商/账户·做空/AUM) ②成本与可交易性(中位价差·单边成本·
  可交易·可做空·无交易带+notes) ③**T3 账户(SIM 估计 vs LIVE 实时醒目区分)** ④门分级(A/B/C 等级·实验模式·
  →ACTIVE 是否真拦·阈值·因子入池门) + **最近运行表**。测试 `test_phase_fetr_endpoints.py`(3)；
  前端 tsc/build 绿 + 94 测试过。
  *（下方为原规划，保留备查）*

- ~~**FE-TR 交易现实面板（原规划）**~~（配 Phase TR）
  **为什么要**：TR 决定"这笔交易在散户 $10k 下到底什么成本、能不能做空、买入力多少、证据够不够"，
  这些**直接决定能否赚钱**，却只存在于日志里，用户看不见 → 无法核对、无法发现问题。
  **⚠️ 前置（必须先做，否则前端无源）**：`run_portfolio` 的运行诊断（`t3 / risk_report / horizon /
  no_trade_band / turnover_ann / strategy_verdict(含 paper_grade) / strategy_decay`）目前**只进日志与
  调度器返回值，没有任何 API 端点**。需先补：①把每次 run 的诊断持久化（新表或复用 run_manifest），
  ②开只读端点 `GET /api/portfolio/diagnostics`（最近 N 次）。
  **面板四块**：
  1. **数据源状态（TR.2）**：OpenD 连通、行情权限（美股 LV3）、**历史K线额度用量 x/100**、
     当前 `price_source`（yahoo/moomoo 醒目标识——研究/执行是否同源）。
  2. **交易现实（TR.1）**：每名**估计价差**（Corwin-Schultz）分布、**可交易池**（哪些被价格/流动性
     过滤及原因）、**可做空性**（long-only 明确标注）、单边成本估计、**无交易带**。
  3. **T3 账户（TR.3）**：**provider 模式 sim/live 必须醒目区分**（防止把估计当实时——T3 纪律的
     可视化兜底）、买入力、现金、持仓数。
  4. **门分级（TR.4）**：当前策略 **A/B/C 等级**、**→ACTIVE 门状态**（已积累前向天数/需要天数、
     realized IC t/阈值）、**实验模式**与 `tr_enforce_active_gate` 开关状态（让人一眼看出"门是否在真拦"）。
  **入口**：Portf 视图加 "交易现实" 标签页（与策略配置并列），避免再加一个顶层导航。
  **验收**：真启动前后端，面板显示的价差/成本/买入力与后端返回一致；`sim/live` 标识正确；
  额度用量与 moomoo 实际一致。

- **FE-12 执行监控面板**（配 Phase 12，**重**）：moomoo 纸交易的持仓/挂单/成交、**本地 vs 券商
  对账差**、风控门状态与 **kill switch（一键全平，带二次确认的人工动作）**、三级保真度对比
  （内部模拟 vs moomoo 纸交易成交）。
  **✅（2026-10-06）**：`ExecutionPanel.tsx`（Portf 视图「执行监控」标签页）；全平/解除走 arm→输入确认词→confirm
  两步，接受对账差必须填理由。真启动端到端随 Phase 12 一起待做。
- **FE-13 红队报告 + 自主度开关**（配 Phase 13）：每个 PAPER 候选的"反方报告"在谱系内可查；
  **autonomy_mode 手动/全自动切换**（人工可随时切回）；数据质量哨兵告警。
- **FE-R 研究可信度图表**（配 Phase R）：PBO/CPCV 结果、**因子风险归因**（暴露分解）、
  **容量-AUM 衰减曲线**——复用 `components/analysis/*`，天然图表化。
  **后端数据已就绪（2026-09-22）**：`GET /api/portfolio/diagnostics` 的每条记录已含
  `risk_attribution`（因子/特异方差、逐因子贡献、组合净暴露、`styles_missing`）与
  `strategy_verdict.pbo/pbo_method`。前端只差把它画出来 —— 画的时候**必须把
  `styles_missing` 一起显示**，否则"因子风险占比"会被读成"覆盖了主要风格"。

> 每个 FE-* 的验收并入对应 Phase 的"真实启动前后端端到端"验证（沿用 Phase 7 做法）。

---

## 依赖关系（执行顺序）

```
Phase S（统计地基）★★ P0 ── **已完成 2026-09-22**；相对结论可读，绝对水平仍需前向证据
  S.1 去 OOS 选择+purged CV 适应度(✅) · S.2 全路径三段+冻结 holdout+使用台账(✅)
  S.3 全局 trial+t≥3+**CPCV**-PBO(✅,吸收R.1) · ~~S.4 换退市源~~(🚫搁置,被 TR.2 取代)
Phase 8（PIT 地基，✅）
Phase 9（自主发现 + 生命周期门 + 批准，✅）── 门控设计已演进为「策略级」，见 PM.S
Phase TR（交易现实：moomoo 单一源 + 真实成本/可做空 + 门分级）
  TR.1 TradingContext(✅) · TR.2 moomoo 单一权威源(✅) · TR.3 成本落地+T3 providers(✅) · TR.4 门分级+阈值配置化(✅) ── **Phase TR 全部完成**
Phase PM（组合与资金管理层）★ ── 依赖 9；与 S 并列最高优先级
  第一批(✅) · ★核心重构:策略级门 PM.S1(✅)+边际准入 PM.S2(✅)+经典基准库 PM.S3(✅) · 第二批 PM.5/6/7(✅) · FE-PM(✅) · 收拢 R.2/R.4
Phase 10（另类数据：基本面，价格仍走 moomoo）·  Phase 11（前向增量，价格源=moomoo/TR.2）
Phase 12（**moomoo** 执行，同 TR.2 源）── 依赖 PM（消费美元账本）+ TR.2 + 11
  12.1 网关+OrderManager(✅) · 12.2 风控门+熔断(✅) · 12.3 保真度报告(✅) · 12.4 启动恢复(✅) ── 真 OpenD 端到端(⬜)
Phase 13（多 agent + 全自动）── 依赖 9 + 12
Phase 14 ── 长期验证期

Phase R ── R.1→S.3(✅) · R.3→PM.2(✅) · R.4→PM.1/3(⬜ Ledoit-Wolf/HRP/换手进目标仍未做)
  R.2 风险引擎 ── **不再是"被 PM.5 吸收"**：PM.5 做的是敞口/集中度上限，回答不了
  "风险从哪来"。已建 app/core/risk_engine（2026-09-22）：归因 + 结构化协方差(✅，已接进
  run_portfolio 诊断)；风格中性化替换与 alpha/风险溢价区分(⬜)；B6 beta 闭合仍待做空开启

前端 FE-9(✅) · FE-PM(✅) · FE-TR 交易现实面板(✅,含诊断持久化+两端点) · FE-12 执行监控(✅) · FE-8/10/11/13/R(⬜)
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
