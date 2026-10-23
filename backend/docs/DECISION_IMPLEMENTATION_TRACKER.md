# 2026-10-22 用户决定的逐项追踪

来源：用户 2026-10-22 的指示（核验发现的 6 个问题 + 对 21 条待定事项的逐项决定 + 交付要求）。
**状态只以本表为准**；不凭测试总数或两套引擎互相一致宣称修复。

总约束（原话）：不立即启动正式前向；"同意技术方案"不等于允许修改活库或下单；保留已修好的熔断、健康门、报价、会计和研究隔离回归。

状态：✅ 做完并验证（附证据） · 🟡 部分 · ⬜ 未开始 · ⛔ 等用户 / 现场 · 🧪 需现场验证

推进顺序：正确性与状态 → 时钟 / 预热 → 证据 / 研究纪律 → 告警巡检 / 恢复 → 环境与关键路径复核 → 活库预览 → 人工验收。

## 批 1：正确性与状态

| # | 要求（原话摘录） | 状态 | 证据 / 说明 |
|---|---|---|---|
| C1 | 提案结果矛盾：统一最终 passed、评级、准入、API 和前端展示，保留各子门原始结论；必须的门失败或未评估不能标 A | ✅ | 三道必须的门（冻结规格 / 流程滚动前推 / 基本面质量）各自的原始结论进 `verdict.gates`；`verdict.passed` = `cfg.passed` = 都评估了且都过了；评级在最终结论之后做，失败或未评估的门不可能是 A、有未评估的门也不是 B（`grade_paper_entry`）；API 同值；前端按最终结论 + 逐门显示（`StrategyCard`）。先失败：cfg=False / verdict=True、非实验模式未拦。顺带修了一个我这次改动引入、被旧用例抓到的 bug（循环变量 `name` 覆盖了配置名）。`tests/integration/test_strategy_proposal_verdict.py`（TestProposalVerdictConsistency）；前端 `StrategyCard.test.tsx` 4 条（**改完才写，没看到先失败**） |
| C2 | 实际数据源未知 / PIT 元数据读不到时不得退回 settings；正式证据来源必须可验证，未知 / 缺失 / 不匹配 → 隔离；探索可继续但不得伪装成合格前向 | ✅ | 每日循环身份与管线身份（`_pipeline_identity`）的价格源只取 PIT 实际来源，未知 → 空 + `source_verified=False`，声称的来源另存 `claimed_price_source`；`ForwardBook.signal_for` 只认本轮已验证、存档时有来源且两者一致，否则不当前向（策略证据记 replay、组合 IC 不标前向）。先失败：身份里是 settings 的 yahoo、来源未知的 2 天被记成合格前向。**交易本身仍按规格核对**（来源未知时只隔离证据，不额外阻断交易 —— 要不要连交易也拦，需要你定）。约 12 条按旧口径（不传来源）写的用例补上真实来源 yahoo，断言未放松。`tests/integration/test_evidence_input_identity.py`（TestUnknownPriceSource） |
| C3 | N03：评分仍是前收盘到今收盘、含已发生的隔夜收益 —— 与 D14 一起明确标签起止、信息截止、归档截止 | ✅ | 三条边界写在唯一来源 `app/core/clock.py`（`describe()` / `label_window()` / `archive_deadline()`）：信息截止 = T 收盘；标签 = 开盘(T+1) → 开盘(T+2)；归档截止 = 开盘(T+1) = 标签起点，存档时标签里没有任何已发生的收益。每日循环在 T+2 结算信号日 T（组合、逐因子、策略证据三处同一口径），`signal_for(T, T+1)` 截止就是开盘(T+1)。先失败：clock 模块不存在、前向 IC 仍按收盘到收盘结算。`test_trading_clock.py`：TestClockDefinition 4 条、`test_forward_ic_is_scored_on_the_open_to_open_label`（存档信号对开盘到开盘标签，独立 scipy 复算）。旧用例按 c2 改写时，有 7 条"只隔一天"的前向 / 隔离用例在 c2 下会**空转**（无论如何都不会有前向），已改成隔两天，保持判别力 |
| C4a | 正式日报不能被另一条正式日报覆盖：决策和信号不可变；后续成交、费用、更正追加事件 / 完整修订链 | ✅ | 一天的决策（第一条带目标的记录）写进 `forward_days` 后不替换；同日重跑 / 补跑 / 更正整份追加进 `forward_day_events`（revision；决策之前没有目标的尝试 = attempt）；`history()` 看完整链。`tests/integration/test_formal_forward_daily_record.py`；§7.3 两条旧用例改按修订链断言（比原来更严） |
| C4b | 启动资格比较完整 UTC 时间戳，不只比日期；启动前已发生的收益不计入正式成绩 | ✅ | 决策时刻必须晚于启动时刻（`decided_before_formal_start`）；当天实现的收益属于前一个决策（`outcome.of_decision`），只有前一个决策正式才进 `formal_performance()`；启动要求策略处于允许正式天数的状态（以库里状态为准）。`tests/integration/test_formal_forward_daily_record.py` |
| C5 | 试验史中断识别不能有 6 小时宽限：running / 被杀 / 结束标记写失败 → 正式资格等结案、保守计数或明确 incomplete；超时只用于发现僵尸任务 | ✅ | `TrialLedger.open_runs()` = running（不论多久）+ incomplete；`trial_history_problem` 有任何未结案运行就不给正式结论，原因里标进行中 / 疑似僵尸 / 有漏记；结案靠运行结束或 `resolve_run`。先失败。`tests/integration/test_trial_history_completeness.py`（TestOpenRuns）。**影响**：另一个进程正在跑搜索时，验证门 / 策略门不给正式结论（按你的要求偏保守） |
| C6 | 分清"开始收集证据"与"已通过验证"：明确工程联调、PAPER_RUNNING、VERIFIED_ACTIVE 的状态和权限；不能靠关门、借旧策略 IC、把 active 字样当验证通过来自举 | ✅ | `app/core/lifecycle/strategy_states.py` 是权限表的单一来源：integration（交易，证据与正式天数都不计）/ paper_running（交易、计证据、可正式，未验证）/ verified_active（只能经 `POST /strategies/{sid}/verify` 过预登记验证门，**没有开关**；`tr_enforce_active_gate` 废弃）；旧字样 `active` 无任何权限，每日循环不按它交易（哪怕带规格）。`approve?activate` / `/activate` → paper_running，谱系写明未验证；新增 `/integration`、`/verify`；联调期策略证据记 `integration` 类、不计。前端按状态给按钮。先失败。`tests/integration/test_strategy_state_permissions.py`（TestStrategyStates）；约 22 条按旧状态 / 旧开关写的用例逐条按新状态改写，意图保留 |

## 批 2：时钟 / 预热

| # | 要求 | 状态 | 证据 |
|---|---|---|---|
| D14 | 选 (a)：T 收盘信息完成并过健康门后冻结信号与目标，T+1 按明确交易规则执行，换时钟版本；信号、回测、内部模拟、IC、归档、基准、调度同步改；计划开盘执行 ≠ 必按开盘价成交；预测 IC 与实际持仓收益分开记；旧历史保留旧版本、先隔离重算 | 🟡 | **做完**：时钟版本 c2（`CLOCK_VERSION`，进规格哈希 → c1 的规格 / 证据不被继承）。**信号**：`SIGNAL_DELAY=0`，所有 `delay=1` 处改取时钟。**回测 + 内部模拟**：两套引擎都按 T+1 DAY 限价单撮合（参考价 T 收盘、限价带 `exec_limit_band_bps`；开盘在限价内 → 开盘价，否则当日触及 → 限价，否则不成交并按 `limit_not_reached` 挂账；ADV / 波动截至 T）；隔夜收益算旧仓、盘中算新仓；没有开 / 高 / 低就拒绝运行。独立份额 + 现金账本对照 1e-9（`test_engines_match_share_cash_ledger.py`，五条），回测与模拟账本在真实跳空 K 线下逐位一致（`test_paper_broker_replay_parity.py`）。**IC**：每日循环、研究侧回测 IC / IC 衰减 / 分档、评估器、合成器拟合（`clock.paired_returns`）、工作流组合 IC 都用开盘到开盘标签。**归档**：见 C3。**调度**：交易循环 21:30 UTC 与重试 22:45 UTC 在冬夏令时都落在 T 收盘与开盘(T+1) 之间（新用例，5 个日期）。**分开记**：预测 IC 与账本 / 券商收益是两条序列（新用例：限价没成交 → 账本 0，标签照常）。**旧历史**：因子 IC 按时钟隔离，监控与滚动统计只读当前时钟，c1 行保留、面板分两条线显示（`icClock.test.ts`）。顺带修了一个被新用例抓到的缺陷：**完全没成交**的订单不留成交记录（c2 下这是常态）。**未做**：基准（等权宇宙）按 D16 在批 3 统一定义 —— 现在每日记录的基准仍是数据集等权收盘到收盘，与账本的收盘估值同口径，但股票池时点 / 缺价 / 公司行动规则未固定；c1 历史的重算走 D2（批 6，版本化 dry-run）；`gp_engine._evaluate_individual` 仍是收盘到收盘（生产代码不调用，只有测试用）；真 OpenD 下的限价成交与模拟规则的对照要在 D10 人工验收里看 |
| D9 | 预热：只用当时已知历史算特征，只评分 Validate 日期；拟合 / 标准化不读未来评分信息；覆盖 252 日因子、嵌套窗口、跨界标签、初始持仓、历史不足 | ✅ | `RealisticBacktester.run`：OOS 特征在"IS 中早于 OOS 的行 + OOS"拼成的历史上算，只取 OOS 日期评分，账本从 OOS 第一天空仓开始；`oos_warmup` 报 history_days / scored_days / first_scored。逐项：252 日因子与嵌套窗口在 Validate 有值（先失败）；Validate 末尾 30 天的价格改了，之前每天结果不变、IS 结果不受 OOS 影响（**这条在旧代码上也通过，是守卫不是先失败**）；初始空仓；历史不足如实少评分（先失败）；跨界标签：信号 T 的标签用到开盘(T+2)，c2 引擎第一天只决策不成交，OOS 段内不读段外价格；**拟合**：滚动前推合成的第 k 段只用 start 收盘时已实现的标签 —— 用例把 start 之后的标签整片换掉，第 2 段逐位不变（信号截断与收益截断两道守卫，同时去掉才会红，已验证）。需要拟合却没给配对收益 → 报错，不拿收盘到收盘凑。**口径说明**：IS 与 OOS 之间的隔离带不在两段数据里，按行计算的时序窗口会跨过这段空档（行数口径，已写进 `_warmup_history` 说明） |

## 批 3：证据 / 研究纪律

| # | 要求 | 状态 | 证据 |
|---|---|---|---|
| D12 | 只读查询活库（明确路径、真正只读连接、不调用会建表 / 迁移的 Store 构造函数）：已有边界、最早曝光日期、次数、股票池、实验身份，给出日期建议；正式入口不再依赖首次调用偶然冻结；改实验名不能刷新重叠数据的曝光额度 | 🟡（代码 ✅；日期 ⛔ 等你确认） | **只读报告**：`tools/research/research_boundary_report.py` 只用标准库 sqlite3、URI `mode=ro&immutable=1`（不加锁、不建 -wal/-shm、不建表），库旁有未合并 -wal 时拒绝，读前读后比对哈希 / 修改时间 / 目录；不 import 任何 Store（AST 检查）。2026-10-23 对活库 `C:\QuantAgentData\alphas.db` 跑过：文件前后不变；没有 `frozen_boundaries`、`trial_run_state` 表，`holdout_usages` 为空；2026-10-11 有研究活动（10 条因子、累计试验 258）而当时没有边界、库里没记数据集与窗口 → 按给定的数据起点**推定**此前数据都已曝光，给出 clean（2026-10-12）/ conventional（2024-10-10，已曝光）两种建议，数据起点取配置缺省值（实际以活 .env 为准）。**正式入口**：研究入口不再写边界（临时切分，`boundary=provisional`）；验证门与策略提案（新增必须的门 `research_boundary`）只认人登记的边界，旧代码偶然写下的算 unconfirmed 也不认；登记 = `BoundaryStore.register`（只写一次、要操作人与理由；CLI `tools/research/register_research_boundary.py` 缺省预演，`--confirm` 才写）。**曝光额度**：holdout 使用记下实验名与股票池，但计次只按数据集 + 区间重叠，改实验名后同一窗口算第 2 次。**顺带修了一个被旧用例抓到的缺口**：聊天工具只在边界"已持久化"时才走记账出口，临时切分下 `use_test_set` 会落到一条不记账的旧路径（还标 is_true_holdout）—— 现在真实数据集不论边界状态都按切分的 Test 段记账。直接以晋级状态入库（`AlphaStore.save`）也按 D1 拦。先失败。`tests/integration/test_frozen_test_boundary.py`、`tests/integration/test_research_boundary_tools.py`（共 17 条）（含报告对活库形状的旧结构库、预演不改库） |
| D16 | 等权宇宙基准：预先固定股票池时点、再平衡、收益窗口、缺价、公司行动处理；不能事后删缺价股票再平均；低仓位对满仓基准的差额不称 alpha | ✅ | 规则写在 `app/core/benchmark.py`（版本 b1，写进每天记录）：成分在 `ForwardRecordStore.start(benchmark_universe=...)` 时冻结（启动必须给，不给拒绝；启动前的记录标 universe_frozen=False）；每日再平衡；收盘(T−1) → 收盘(T)；缺价 / 隔离成分按 0 收益留在分母并列名单（此前是对有值的票求平均 = 事后删缺价）；差额字段改名为 `*_vs_full_benchmark` 并附说明「不是 alpha」，另给 `*_exposure_matched`（收益 − 净敞口 × 基准）；SPY 记为后排。先失败。`tests/integration/test_equal_weight_benchmark.py`（7 条）。**迁移**：`forward_runs` 补一列 `benchmark`（可空，只加不改） |
| D1 | 基本面不正式晋级；允许选择段内预登记诊断和试验记账，不自动消耗最终 Test | ✅ | `app/core/lifecycle/fundamentals_policy.py`：用到基本面字段的表达式不能流转到 validated / paper / active —— 拦在 `AlphaStore.update_status`（API、验证门、发现、审批都经过它；API 返回 422）；验证门对它们照算选择段指标、标 `diagnostic_only`、永不通过；策略提案多一道必须的门 `fundamentals_policy`；`run_fundamental_diagnostics` 先预登记（登记人、目的、表达式摘要与条数）并把条数记进试验史，再在选择段逐条跑门，`report_test=False`、不记 holdout 使用。先失败。`tests/integration/test_fundamentals_promotion_policy.py`（7 条，含直接以晋级状态入库） |
| D7 | 多类别股保持 NaN，不回填、不变成零市值 / 便宜估值 | ✅ 现状即是（补了守卫） | 没有股本的公司（用真实 AAPL 事实去掉股本概念构造）市值 / E/P / B/P 全为 NaN、`fund_rank` 里保持空缺；基本面路径里没有任何快照股本来源。三条在旧代码上就通过（守卫）；变异验证：把缺失股本填成「看起来合理」的 5e9 → 两条红（填成 1 会被估值合理性护栏挡回 NaN）。风险模型对缺失暴露 `fillna(0)` 是 z 分数空间的「平均」，不是零市值，不在 D7 范围。`tests/integration/test_multi_class_share_counts.py`（3 条） |
| D8 | 不扩展新闻稿级 SUE；保留 filing-based 名称与期间配对；文档里"必须付费"改为尚未核实 | ✅ | 代码不变（filing-based 名称与配对见 2026-10-19 F13）。路线图 D8 行问题栏的"多为付费"改为"是否需要付费尚未核实"；README 未决清单改为已决（付费与否未核实） |
| N05-c | 首轮关闭目标波动：用户手工改 .env；我提供预检和配置说明；决定须体现在新规格中，不能偷偷改变已批准策略 | ✅（.env 由你改） | 上线预检新增 `target_vol_off`（阻断）：`RISK_TARGET_VOL_ANN` 不为 0 → 红；运行中策略的规格里目标波动还开着 → 红并要求重新提案。规格本来就记 `risk_limits.target_vol_ann`（关闭时为空），关闭后旧规格与运行时对不上、每日循环不按它调仓（这条在旧代码上就通过，守卫）。配置说明写进 OPERATIONS §7。先失败（三条预检）。`tests/integration/test_target_vol_preflight.py`（4 条） |
| F02′ | 完整公司行动账本后排；但未知公司行动必须隔离、停止新增风险并对账（价量策略同样依赖） | ✅（最小层；完整账本后排） | **识别**：每轮看最近 5 根 bar，单日 \|对数收益\| ≥ log(1.4) → 疑似未知公司行动，标出最近的拆股比例（新:旧）。**隔离**：`corporate_action_flags` 表，人结案前一直有效（`tools/ops/ca_flags.py list / resolve`，结案要操作人与对账结论）。**不加风险**：模拟账本与券商目标从标记日起不超过标记前一行（同向收缩可以，加仓 / 反手不行）；盘前风控门对隔离名单拒绝加风险单（减仓放行）；标记库读不到 → 券商只对账不下单。**对账**：券商股数与账本对不上且比例像拆股 → 差异里标 `suspected_corporate_action`。每日记录的异常里列出隔离名单，基准按 0 收益留在分母。先失败。变异验证：收缩没接到券商目标 → 端到端用例红。`tests/integration/test_corporate_action_quarantine.py`（8 条）。**未覆盖**：现金侧（分红到账）没有自动核对；影子账本（逐因子）不做隔离（测量用，不动资金）；单看价格分不清真暴跌与未复权拆股（宁可误报） |

## 批 4：告警巡检 / 恢复

| # | 要求 | 状态 | 证据 |
|---|---|---|---|
| D15 | 桌面通知 + 持久化事件文件：分类、严重度、去重、重试、送达状态；区分正常无订单与异常跳过；文件落盘 ≠ 通知成功；不可达时不得称无人值守就绪；桌面通知须现场确认可见 | ⬜ | |
| W17 | 盘中巡检：交易所日历 + DST 调度，可配置 60 秒，沿用已批准风险阈值；共用账户锁和熔断状态；覆盖锁竞争、断线、陈旧快照、恢复，避免并发重复平仓；无人值守前验收 | ⬜ | |
| R21 | 隔离恢复演练（合成恢复通过 ≠ 活库恢复已验证） | ⬜ | |
| T21 | 文档核对"盘中熔断"等说法；直接修正原结论，不追加矛盾脚注 | ⬜ | |
| D11 | API 首轮仅本机回环；保留监听参数与关键动作检查；远程前先认证授权，CORS 不代替认证 | ⬜（核对现状） | |

## 批 5：环境与关键路径复核

| # | 要求 | 状态 | 证据 |
|---|---|---|---|
| D3 | `C:\QuantAgentEnv`（已有则新建子目录）按 lock 建独立测试环境；不升级交易环境、不改系统默认 Python、不复制活 .env | ⬜ | |
| D4 | 仓库已公开：真实行情夹具换等价合成数据，保留测试合同；真实源测试单独管理；不重写 Git 历史 | ⬜ | |
| D5 | 关键路径变异复核分批：健康 / 身份、时钟、规格与结论、资金切片、认领、熔断、证据、研究计数、恢复；报告用时、存活 mutant、未覆盖项 | ⬜ | |
| D6 | 活环境基本面继续关闭；保留研究与质量门测试 | ✅ 现状即是 | 未改 .env；质量门测试在 `tests/unit/data_engine/test_fundamentals_quality_gate.py` |

## 批 6：活库预览（只读 / 副本）

| # | 要求 | 状态 | 证据 |
|---|---|---|---|
| D2 | 先备份、验证恢复、版本化 dry-run；展示受影响记录、旧新值、算法 / 数据 / 时钟版本、排序影响；保留旧值与曝光史；写活库另行确认 | ⬜ | |
| D13 | 重提案：先修正确性、D14 和资格状态；先隔离预演，再由用户批准活环境操作；不补 hash、不继承旧时钟证据、不先激活旧时钟再重做 | ⬜ | |

## 批 7：人工验收

| # | 要求 | 状态 | 证据 |
|---|---|---|---|
| D10 | 专用 SIMULATE 账户：先准备脚本和证据模板（提交成交、DAY 跨 session、部分成交撤单、未知订单、重启去重、熔断续平、输入异常、恢复）；连接、订单、时间由用户安排 | ⬜ | |
| G20 | 正式启动：现在不启动；届时给固定版本的 Go/No-Go 报告，由用户启动；不补记正式天数 | ⛔ | |

## 交付要求（每批）

固定 commit（**由用户提交** —— 我这边建分支被拒，不做 git 写操作）、行为变化、测试命令与环境、失败到通过的证据、未覆盖项、迁移与回滚方案、本表更新。
