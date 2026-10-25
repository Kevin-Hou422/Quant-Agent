# OPERATIONS.md — 验证期运营规则

> 状态：制度文档（Task 8.3）· 适用阶段：Phase 8 起系统进入每日自动 paper trading 的验证期
> 依据：backend_retired_report/PAPER_TRADING_ROADMAP.md §五（已归档）、RESEARCH_OPERATING_MODEL.md §4
> 生成日期：2026-08-19

本文件规定**因子从发现到 paper 交易再到"激活"全过程的人工与制度门槛**。它约束的是流程与
决策纪律，不是代码——代码层的状态机在 `app/db/alpha_lifecycle.py`，晋级走人工 PATCH/审批端点。

**核心边界：本系统在验证期不接入真实资金。** 见 §6。

---

## 1. 进入 PAPER 前的硬门槛（CANDIDATE → VALIDATED → PAPER）

任一因子在被放入 paper 交易前，**必须同时满足**：

1. **WalkForward ≥ 5 折全正** — 每一折的 OOS 表现为正（不是平均为正，是逐折为正）。
   工具：`WalkForwardBacktester`（`performance_analyzer.py`），端点 `/api/alphas/{id}/walk_forward`。
2. **Deflated Sharpe Ratio > 0.90** — 对多重检验/回测过拟合去膨胀后仍显著。
   工具：`deflated_sharpe_from_returns`（Bailey & López de Prado 2014）。
3. **真实数据集回测** — 用注册的真实市场数据集（US/crypto），**禁止**用合成数据下结论。
4. **红队反方报告**（见 §2）已附于该候选的谱系。

> **已落地（Phase S，2026-09-22）**：显著性门槛已是 **t≥3.0**（Harvey-Liu-Zhu），
> DSR 用**全局累计 trial 数**去膨胀，过拟合检验用 **CPCV**（组合式 purged CV）算 PBO。
> 严门加在**策略**层（`StrategyGate`），因子层只做泄漏过滤 —— 见 roadmap 的统一门控政策。

### 1.1 冻结样本外（holdout）的使用纪律

全部研究路径（CLI、chat、工作流、发现、`/api/backtest/*`、`/alpha/*`、`/gp/evolve`、验证门、提案）都在**三段切割**下运行：
最近一段（默认 2 年）被冻结，选择流程一律看不到它。切法只有一种（`research_context.open_research`，复审 2026-10-19 N09-c）；
每次 GP / Optuna 运行的记录（`trial_run_state.context`）都写着它用的数据集、股票池、实验名与冻结边界。

- **`S_THREE_WAY_ENABLED=false` 关不掉已冻结的边界**：数据集已经固定过 Test 起点时，这个开关被忽略（会记 ERROR）；
  它只对还没有边界的数据集生效。冻结过的数据不会因为关开关而重新变成可看的。
- 实验名 `RESEARCH_EXPERIMENT`（缺省 `default`）只是**记录**：边界仍按数据集固定，要不要按数据集 + 股票池 + 实验分别登记是待决事项 D12。

- **动用冻结段必须是一次显式动作**（`report_test=true`），默认不碰。
- 每次动用都被记进 `holdout_usages` 台账；**超过 1 次**会在响应里带 `over_budget=true`。
- 看到 `over_budget=true` 时，**那段数据已经不是样本外了**：它的结论按"已挖掘"解读，
  或换一段真正没看过的数据。系统不阻断（阻断只会让人绕开台账），但这件事必须进复盘。
- 响应里 `partition.degraded=true` 表示冻结窗口没兑现（数据太短，按份额切的），
  那次的样本外强度更弱 —— 别拿它和真冻结窗口的结论平级比较。

晋级操作：人工经 `PATCH /api/alphas/{id}/status` 将 VALIDATED → PAPER，**这是默认的人工
决策点**（capital/风险暴露的把关）。系统不自动前向晋级。

---

## 2. 红队审计（RESEARCH_OPERATING_MODEL §4.2 落地）

**每个进入 PAPER 的候选**必须附一份**"反方报告"**，从对立视角尝试证伪该因子：

- **泄漏**：是否存在前视/未来函数/发布时点错位（尤其未来接入基本面/季频数据后）。
- **容量**：ADV 参与率、可承载 AUM；在现实规模下 alpha 是否被成本吃掉。
- **拥挤度**：是否是已知的、被广泛交易的公共因子（动量/反转/规模等）的换皮。
- **经济逻辑**：是否有可陈述的经济机制，还是纯数据挖掘产物。

**实现形式**：可用**独立 prompt 的一次性 LLM 审查步骤**产出反方报告存入谱系，**不必是常驻
agent**。该步骤只读、不改状态、不进交易回路（符合"LLM 不进交易回路 + 状态流转不作为 LLM
工具"）。达到"候选产出速度 > 人工审查带宽"时，再考虑将其升级为常驻红队 agent（见 roadmap
Phase 13.1）。

---

## 3. PAPER → ACTIVE 门槛

累计 **≥ 60 个交易日的前向证据**后，若满足：

- realized IC 均值 **> 0**，且
- realized IC 的 **t-stat > 2** —— 取普通 t 与 Newey-West（HAC）t 中**较小**的那个（日度 IC 有序列相关，
  普通 t 会把相关的样本当成独立的；外部审查 2026-10-18 F14），

方可由**人工**晋级。60 个交易日是"运转起来了"的里程碑，不是盈利优势的证明 —— 还要看净收益、
基准超额、回撤、成交率。

**什么算前向证据**（外部审查 2026-10-18 F06，复审 2026-10-19 N03，2026-10-22 C3 / D14）：信号日 T 的标签是
开盘(T+1) → 开盘(T+2)；只有在**标签起点 开盘(T+1) 之前**就存档的那份信号结算出来才算前向 —— 存档时标签里一笔收益
都还没发生（时钟 c2，`app/core/clock.py`）；规格与运行时（含 PIT 的实际价格源）对不上的那几天隔离、不计入（N02）（`signal_archive` → `strategy_evidence` / `alpha_ic_history.is_forward`）。
停机后补拉的、重跑的、离线补算的都只算回放。前向行写入后不可改，改写尝试记 `evidence_revisions`。

**策略状态与权限**（2026-10-22）：`integration`（工程联调：人工监督下在模拟账户交易，证据与正式天数都不计）→
`paper_running`（纸交易、收集前向证据，**未验证**；开始不需要证据）→ `verified_active`（只能经预登记的验证门，
没有开关可以绕过）。旧字样 `active` 不代表任何权限，每日循环不按它交易。

**按策略版本记账**：申请验证（paper_running → verified_active）时只读**这份配置（id + 规格哈希）自己**的前向记录；
每日循环从配置被提出那天起就按它的规格存档信号、次日结算，所以批准之前也在积累。
因子级：`alpha_ic_history`（`/api/alphas/{id}/ic_history`）；衰减监控由 `AlphaMonitor` +
`daily_monitor_job` 自动巡检（唯一自动流转是 ACTIVE → DECAYING 降级，前向晋级永远人工）。

---

## 4. 复盘与维护节奏

**每周人工复盘 checklist：**
- [ ] 每日摄取健康门是否有拒绝日？拒绝原因是否已排查（限流/字段变更）？
- [ ] 各 PAPER/ACTIVE 因子的 realized IC 滚动均值、连续负 IC 天数（衰减预警）。
- [ ] paper 净值曲线与 `PositionStore` 记录是否自洽；有无崩溃补跑遗留。
- [ ] 待审批队列（VALIDATED 候选 + 红队报告）是否积压。
- [ ] 本周有没有 `over_budget=true` 的 holdout 使用？哪一段、为什么要再看一次（§1.1）。

**每月：**
- [ ] 全量回归（当前约 5,550 条、约 40 分钟）在独立测试环境里跑（D3）：
      `py -3.12 tools/env/isolated_test_env.py stage` 后 `… run -- -q`（`C:\QuantAgentEnv\lock-test`，按 `requirements.lock` 装，
      源码镜像不含 `.env`，子进程只带白名单环境变量，`OPENBLAS_NUM_THREADS=1`）；别与别的测试并发 ——
      本机内存紧时会出现 OpenBLAS 分配失败的假红。测试不许连外网（`tests/network_guard.py`）；真实数据源用例
      （`live_source`）另跑：`QA_LIVE_SOURCES=1`，SEC 那条还要 `EDGAR_USER_AGENT`。+ 依赖升级窗口
      （在此窗口内做 pip/npm 升级并跑全量；升级要新建环境，不改 `lock-test`）。
- [ ] 查 `evidence_revisions`：有没有被拒绝的证据改写尝试？每一条都要能说清楚为什么同一天会算出两个数。
- [ ] 运行**成本模型校准报告**（Task 8.2，`monthly_cost_calibration_job` 自动产出
      `cost_calibration_YYYYMM.md`）；**人工确认**后方可手动更新 `CostParams`——
      **绝不自动改**，变更须留痕（写入变更日志）。
- [ ] 复核 PIT 存储是否逐日增长（Task 8.1，无人值守摄取的健康度指标之一）。

---

## 5. 数据与可复现

- 每日摄取通过健康门的数据按 `(field, date, as_of)` 追加进 PIT 存储（`pit_store/`，Task 8.1），
  历史分区**只追加不修改**——从启动日起积累无幸存者偏差的自有数据。
- 任何验证/回放固定 `as_of` 以复现历史视角；`RunManifest`（data_sha256 + git_commit + seed +
  config）记录每次关键运行，`verify_dataset` 在回放前校验数据一致。

---

## 6. 明确边界：不接入真实资金

**本系统不连接任何真实资金账户。** 两条账本：

- 内部模拟（`PaperBroker`，与回测同一成交规则：T+1 DAY 限价单，开盘在限价内按开盘价、否则当日触及按限价、否则不成交；复用回测成本模型）；
- Phase 12 起，`EXECUTION_MODE=moomoo_paper` 时向 moomoo **SIMULATE** 纸交易账户下单。
  代码层拒绝任何非 SIMULATE 环境（`LiveTradingRefused`），这不是一个配置项。

若未来决定接入真实资金，**必须另立独立规划**，至少覆盖：实盘硬风控、API 密钥安全管理、
FastAPI 鉴权、告警升级、合规与税务。这些**不在**当前范围内。

---

## 7. 运行边界（外部审查 2026-10-18 后如实写明）

- **信号与成交的时钟**（D14 (a)，时钟版本 c2，唯一来源 `app/core/clock.py`）：信息截止 = T 日收盘（数据过健康门之后）；
  每日循环在 T 收盘后冻结信号与目标（信号不再额外延迟一天）；券商订单与内部模拟 / 回测都是 T+1 的 DAY 限价单
  （参考价 T 收盘、限价带 `EXEC_LIMIT_BAND_BPS`），模拟按同一规则成交：开盘在限价内 → 开盘价；否则当日触及限价 → 限价；
  否则不成交 —— "计划开盘执行"不当成必按开盘价成交。成交量上限用截至 T 的 ADV。每日收盘估值：隔夜收益算旧仓、
  盘中收益算新仓。预测标签 = 开盘(T+1) → 开盘(T+2)，在 T+2 结算；归档截止 = 开盘(T+1)。预测 IC 与模拟账本 / 券商的
  实际收益分开记。研究侧（回测 IC、IC 衰减、合成权重拟合、工作流的组合 IC）用同一个标签。
- **熔断**：收盘后的两个任务做日亏检查与熔断后的续做全平；另有**盘中巡检**（`app/core/execution/intraday_watchdog.py`，
  缺省关闭，`INTRADAY_WATCHDOG_ENABLED=true` 开启，`INTRADAY_WATCHDOG_INTERVAL_S` 缺省 60 秒）：只在交易所日历的常规交易时段
  （夏令时由日历处理）连券商；切片相对上一个可信收盘快照的跌幅达到 `EXEC_MAX_DAILY_LOSS`（与下单风控门同一个阈值）→
  拉起同一个熔断开关并撤单 / 全平。与调仓 / 对账共用账户锁，但不等待：拿不到锁本拍跳过；熔断已开（不论谁开的）只续做全平，
  有在途全平单时不再下。连续连不上券商（`INTRADAY_WATCHDOG_UNREACHABLE_ALERT_AFTER`，缺省 3 次）、快照长时间不变
  （`INTRADAY_WATCHDOG_STALE_AFTER_S`，缺省 600 秒，此时不据此做任何决定）、日历不可用 → 严重告警；恢复后发恢复通知。
  上线预检 `intraday_watchdog` 是阻断项：没开就不能无人值守。巡检**还没在真 OpenD 上跑过**（D10 人工验收）。
  开熔断（kill switch）会立即落库并尽量撤单 / 全平；账户正被另一个进程操作时撤单 / 全平留给下一周期。
- **告警**（D15）：桌面通知（Windows 通知）+ 持久化事件文件（缺省在主库同目录的 `alerts/events.jsonl`，`ALERT_EVENT_FILE` 可改）。
  每条事件有分类、严重度、去重（`ALERT_DEDUP_MINUTES`，缺省 60 分钟）、重试（`ALERT_RETRY_ATTEMPTS`，缺省 3 次；调度器每 30 分钟补发失败的）
  和送达状态（file_only 只落盘 / suppressed 去重 / handed_to_os 已交给系统 / failed）。每日循环的结果分类发出：正常无订单（info）与
  异常（任务没执行、摄取被拒、没有新 bar、组合账本失败、执行层没运行或被拦、下单出错）分开，非交易日单独一类；
  另发：带后风控仍违规、隔离的疑似公司行动、调度任务失败 / 错过、熔断开 / 关、盘中巡检。
  **文件落盘 ≠ 通知成功，交给系统 ≠ 人看见**：`python -m app.tasks.alerts test` 发一条测试通知，人在屏幕上看见后
  `python -m app.tasks.alerts confirm <事件 id> --actor <你>`；`status` 看是否就绪，`retry` 手动补发。上线预检 `alert_channel` 是阻断项：
  没有 `ALERT_CONFIRM_VALID_DAYS`（缺省 14 天）内的现场确认、或确认之后有通知失败 → 不能称无人值守就绪。
  没有外部渠道（邮件等）：人不在这台机器前就看不到。
- **API 只接受本机回环**（D11）：绑非回环地址（配置、`--host`、`UVICORN_HOST`）一律拒绝启动，没有开关（原来的 `ALLOW_INSECURE_BIND` 已删除）；
  改状态的请求（POST / PUT / PATCH / DELETE）来自非回环客户端 → 403。服务没有认证：需要远程访问时先做认证授权，CORS 不能代替认证。
- **备份与恢复**：每日快照任务 `daily_backup`（主库 + 调度库用 `VACUUM INTO` 并做完整性检查，PIT 打 zip，保留 `BACKUP_KEEP_N` 份）。
  恢复只落到隔离目录：`python -m app.tasks.restore <快照目录> <新的空目录>`（目标与活库 / 活 PIT 路径重叠或目录非空 → 拒绝；恢复出的库做完整性检查）；
  `verify_restore` 逐表比对行数、逐文件比对 PIT。合成数据演练：`python tools/ops/recovery_drill.py --work <新的空目录>`。
  **合成恢复通过不等于活库恢复已验证**：活库的恢复要拿活库的真实快照在隔离目录里做并比对差异，换回活库须人工确认（D2）。
- **策略规格**：每日循环只按带可执行规格、处于运行状态（integration / paper_running / verified_active）的配置调仓；改了 `PAPER_AUM`、风控上限、价格源等
  交易参数，或现役配置是 2026-10-18 之前批准的，都要重新提案并审批（D13）。
- **冻结 Test 段**（2026-10-22 D12）：真实数据集的 Test 起点只能由人**登记**（只写一次，带操作人与理由），
  不再由第一次碰巧发生的调用写死。没登记之前研究按"临时切分"跑、不写库，验证门与策略提案的正式结论一律拒绝
  （"研究边界未登记"）。holdout 按数据集 + 区间重叠计次，实验名、股票池只记录、不进计次（改实验名不会刷新额度）。
  研究工作流默认不看 Test。登记步骤：
  1. 只读查看：`python tools/research/research_boundary_report.py --db <活库路径> --assume <数据集>=<数据起点> …`
     （sqlite3 只读、不加锁、不建表；库旁边有未合并的 -wal 时拒绝，先停掉用库的进程）；
  2. 从报告的建议里选定日期（clean = Test 里没有任何被研究看过的数据，conventional = 按惯例冻结两年但其中日期已曝光）；
  3. 先备份活库（D2），再 `python tools/research/register_research_boundary.py --db <活库> --dataset <名> --test-start <日期>
     --actor <你> --reason "<理由>"` 预演，确认后加 `--confirm` 才写。
- **跨进程互斥**：PIT 追加、券商账户操作都有操作系统级文件锁；同一账户同一时刻只有一个进程在动。
  账户锁按**券商账户**（券商 / 证券公司 / 环境 / 账户 id）建在整台机器共用的目录 `ACCOUNT_LOCK_DIR`
  （缺省 `%PROGRAMDATA%\QuantAgent`；复审 2026-10-19 §5.3）—— 从不同目录跑两套部署也互斥。拿不到账户 id 就不交易；
  执行库第一次用时绑定账户（`bound_account`），之后连到别的账户（例如 `MOOMOO_TRD_ACC_ID=0` 自动取到了另一个模拟账户）就拒绝 ——
  要换账户请用新的执行库。
- **加风险订单的报价时段**（复审 N06-d）：报价必须在最近一个已收盘交易日收盘前 30 分钟内、当天盘后，或之后交易日的
  盘前 / 盘中（按交易所时区）；更早的盘中报价、当天盘前、非交易日时间戳都拒绝。周末 / 假日手动跑时加风险的单会被拦，属正常。
- **正式前向的天数**（复审 §7.3）：每日循环每轮给运行中的策略写一条 `forward_days` 记录；同一决策日的决策不可替换，重跑 / 更正追加进修订链（`forward_day_events`）。**只有人启动正式前向之后**
  （`ForwardRecordStore.start(策略 id, 规格哈希, 操作人, 理由)`，同一策略 + 规格只能启动一次，记下代码版本、数据身份、环境版本），
  并且这一轮在决策日收盘到下一交易日开盘之间运行、执行层在线，那一天才算正式；联调、补跑、回放都不算，也不能覆盖已经是正式的那天。
  启动放在重新提案并批准（D13）之后；启动时必须同时冻结基准成分（`benchmark_universe`，D16）。
- **基准**（D16，规则见 `app/core/benchmark.py`，版本号写进每天的记录）：等权宇宙，成分在正式前向启动时冻结、
  每日收盘再平衡、窗口收盘(T−1) → 收盘(T)（与账本 / 券商收盘估值同一窗口）、复权收盘价；缺价成分与被隔离的
  公司行动成分按 0 收益**留在分母**并列出名单，不删后再平均。"相对满仓基准的差额"不是 alpha —— 记录里同时有
  按净敞口折算的差额。SPY 第二基准后排。
- **未知公司行动**（2026-10-22 第 18 条）：每轮看最近几根 bar，单日 |对数收益| ≥ log(1.4) 的票记隔离标记
  （`corporate_action_flags`），一直有效到人对账结案。隔离期间模拟账本与券商只许减仓、不许加风险（券商侧由盘前风控门
  按实际持仓拒绝加风险单）；券商股数与账本对不上、比例像拆股时差异里会标出比例。标记库读不到 → 券商本轮只对账不下单。
  查看 / 结案：`python tools/ops/ca_flags.py --db <活库> list`；核对券商股数 / 现金与数据源复权之后
  `... resolve <票> --actor <你> --resolution "<结论>"`。完整公司行动账本后排。
- **目标波动**（N05-c）：首轮关闭 —— 活 `.env` 里 `RISK_TARGET_VOL_ANN=0`（由你手工改）。上线预检有一项
  `target_vol_off`：运行时没关，或运行中策略的规格里还开着（批准于关闭之前）→ 红。改 `.env` 不会悄悄改变已批准的策略：
  规格里的风控上限与运行时对不上，每日循环本来就不按它调仓 —— 关闭后按新配置重新提案（新规格里 `target_vol_ann` 为空）。
- **基本面**（D1）：本轮不正式晋级。用到基本面字段的因子不能流转到 validated / paper / active（所有路径都经
  `AlphaStore.update_status`），验证门对它们只给选择段诊断；需要诊断时用
  `app.core.lifecycle.fundamentals_policy.run_fundamental_diagnostics`（先预登记：登记人、目的、表达式摘要与条数，
  条数先记进试验史；全程不碰冻结 Test 段）。多类别股的股本保持 NaN（D7）。
- **验证（→verified_active）按预登记的验证设计判**（复审 N05-d）：提案时把门槛（`TR_MIN_FORWARD_DAYS` / `TR_MIN_IC_MEAN` /
  `TR_MIN_IC_TSTAT`）、t 的口径、证据口径、时钟口径和登记时刻冻结进 verdict（带哈希）。之后改这些配置**不影响**已提出的策略；
  只算登记日之后、同一规格的 forward 证据。2026-10-20 之前提出的策略没有设计 → 门不过（本来就要重新提案，D13）。
- **回放证据单列**（复审 N03-b）：没有及时存档的那几天记为 replay（研究用），不计入前向天数。
- **基本面质量门**（复审 §5.4）：用到基本面字段的策略，这份面板的基本面质量报告必须过门（各字段末日覆盖率 ≥ 80%、
  估值字段用估计股本的标的 ≤ 10%），否则提案不通过、每日循环不调仓（`fundamentals_quality`）。首轮价量策略不受影响（D6）。
- **时钟版本**（复审 §5.1）：所有存下的结果带 `clock_version`；2026-10-22 起为 c2（之前的行是 c1，NULL 也读作 c1）。
  规格哈希含时钟版本，c1 时期的规格、证据一律不被继承；因子监控与 IC 历史统计只读当前时钟，c1 的行保留、带标注展示，
  重算另走 D2 的版本化 dry-run。
- **一份策略只有一套目标**（复审 N05-e）：提案、策略门、每日循环的模拟账本与券商目标、前向证据都按同一份合同
  （`strategy_contract`）生成。目标生成里的波动估计或风控出错 → 本轮不调仓（此前会不缩放照常交易）。
- **活库位置**：活库放在非云同步的绝对路径（`C:\QuantAgentData`）；OneDrive 目录下的 SQLite 会被同步进程锁住。
