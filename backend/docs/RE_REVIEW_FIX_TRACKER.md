# 复审 2026-10-19 逐项追踪表

来源：`review-v6/audit-review/复审报告_2026-10-19.md`。**这张表是"做完没有"的唯一依据**：
报告里每一条"修复："要求、§5 风险、§6 决策建议、§7 工作包都单列一行，验收标准尽量直接引用报告原话。

状态：✅ 做完且有验证 · 🟡 部分（写明缺什么）· ⛔ 需要你拍板 · 🧪 需要现场（真 OpenD / 活库）· ⬜ 未开始

验证一栏写的是**证据**：审查方脚本 `independent_checks.py` 的结果、回归用例名、或"无"。

## §4 剩余缺陷 N01–N09（按报告"修复："逐句拆开）

| # | 报告要求（原话摘录） | 状态 | 验证 / 缺口 |
|---|---|---|---|
| N01-a | 每张非 flatten 订单发送前读取最新熔断状态 | ✅ | 审查脚本 residual 断言失败；`tests/integration/test_kill_switch_mid_batch.py::TestKillSwitchMidBatch` |
| N01-b | 发现 engaged 立即停止剩余调仓并进入撤销加风险单和对账流程 | ✅ | 同上（转 `_ensure_flatten`：撤全部在途 + 全平） |
| N01-c | 测试覆盖"第一笔成功后熔断"和"等待撤单期间熔断" | ✅ | 两条用例都在 |
| N02-a | 证据路径消费同一已核对的 spec，不匹配时隔离并停止正式累计 | ✅ | 审查脚本断言失败；`TestEvidenceIdentity` |
| N02-b | 输入带实际 provider | ✅ | 用 PIT 元数据的价格源 |
| N02-c | 输入带价格口径、vintage / 摘要、有效股票池、代码版本 | ✅ | 新增 `input_identity.py`（数据摘要 / 末日 / 有效股票池摘要与个数 / 实际价格源 / PIT vintage / git 版本）；信号存档与策略证据行都记下；`TestInputIdentity`。旧库的两张表做了加列迁移 |
| N02-d | 不同数据源的历史与前向证据不能借相同策略哈希混用 | ✅ | 存档时的价格源与本轮不同 → 不当前向结算（账本层面拦，不依赖规格核对）；`test_a_signal_archived_under_another_price_source_does_not_settle_as_forward` |
| N02-e | 无新 bar 时也要在消费 PIT 前核对真实来源 | ✅ | `test_a_source_switch_without_a_new_bar_rebuilds_the_pit` |
| N03-a | 明确预测标签与执行收益的起点（close-to-close / T+1 open-to-close / open-to-next-open 分别规定存档截止） | ⛔ | 依赖 D14。现在只把截止提前到被评分日开盘，隔夜跳空仍在评分窗口里 |
| N03-b | 补拉、回放、晚到归档保留研究价值但不升级为正式前向 | ✅ | 策略证据分 `evidence_class` = forward / replay：没有及时存档的那几天（首跑、停机补拉、晚到、换源）按重算信号记 replay（`strategy_replay_ic`），→ACTIVE 只读 forward（`strategy_forward_ic`；旧行 NULL 读作 forward —— 那时只记 forward）。先失败：此前这些天直接跳过，研究价值丢失。§7.1 工作包 4 "影子信号证据与有实际成交的策略收益分开"：信号 IC 在 `strategy_evidence`，实际成交收益在 §7.3 的 `forward_days`。`tests/integration/test_forward_evidence.py` |
| N04-a | 先截断到目标 session，再检查并持久化同一对象 | ✅ | 审查脚本断言失败；`TestHealthOnTheConsumedPanel` |
| N04-b | 健康报告的末日、股票池和摘要与消费面板一致 | ✅ | 摄取结果带交给交易循环那份面板的输入身份，并核对它与健康门检查的那份同末日、同股票池，不一致 → 拒绝（`consumed_panel_mismatch`） |
| N04-c | "未收盘完整日掩盖已收盘缺失日"的回归 | ✅ | 同上 |
| N04-d | 对已持仓名字另设不能被平均覆盖率抵消的检查 | ✅ | `test_a_held_name_without_a_close_rejects_the_day` |
| N05-a | 先形成候选 spec，再让 gate 消费 spec | ✅ | 审查脚本断言失败；`TestProposalEvaluatesTheFrozenSpec` |
| N05-b | 共享波动估计与带后投影逻辑 | ✅ | `test_the_gate_applies_vol_targeting_and_the_post_band_projection` |
| N05-c | 首轮关闭目标波动 | ⛔ | 是配置决定（`RISK_TARGET_VOL_ANN=0`），不是代码 |
| N05-d | 新规格资格来自预登记的验证设计与此后真正未见的前向数据 | ✅ | `app/core/lifecycle/validation_design.py`：提案时登记设计（最少前向天数 / IC 均值 / t 阈值 / t 口径 / 证据口径 / 时钟 / 登记时刻，带哈希）；→ACTIVE（`_active_gate` → `check_registered_activation`）按**登记的**门槛判（之后改 settings 不影响），只算登记日之后、同规格的 forward 证据；设计缺失（2026-10-20 前提出的）/ 被改 / 时钟变了 → 不通过。用例 helper `approved_verdict` 随之带设计（真实提案都会登记）。先失败。`tests/integration/test_preregistered_validation_design.py` |
| N05-e（§3） | 显式三步合同：构建并冻结 spec → 验证该 spec → 以该 spec 生成目标 / 证据 / 订单 | ✅ | `app/core/portfolio_manager/strategy_contract.py`：`StrategyContract`（freeze / from_record → verify / evaluate → signals / composite / targets）+ 唯一的目标生成函数 `generate_targets`（合成账本 → 目标波动 → 硬约束 → 熔断 → 带 → 带后投影）。提案先冻结合同、门评估它、风控快照与换手按它生成、存下的就是它（`evaluated_spec_hash`）；策略门回测、每日循环的模拟账本与券商目标、前向证据的合成信号都走它（有 active 配置时风控上限 / 波动窗口取自规格，不再现读 settings）。**查出并修了两个真分叉**：① 每日循环的合成账本用默认 `CostParams()`、门用 grounded 成本参数（影响容量 → 容量绑定时门验证的权重≠交易的权重）；② `risk_vol_lookback` 改变目标却不在规格里（已加入规格 v2）。顺带：每日循环波动估计 / 风控抛错此前照常交易（不缩放 = 比意图更大的风险），现在 fail-closed。`tests/integration/test_strategy_contract.py` 7 条（结构检查：合成 / 风控 / 带后投影只在一处装配）。**先失败的证据只有静态的**：补丁先于用例运行打上了，结构检查在旧代码上应列出 builder:75/147、gate:180/197、loop:349/493（据当时 grep），没有实跑旧代码。门里的 PBO 为每个单因子各建等权策略，是反事实对照、不是这份策略的目标 —— 用例显式排除并断言被排除的只有单因子调用 |
| N06-a | 加风险订单要求有限正报价、明确时间 / 时区、可接受年龄 | ✅ | 审查脚本断言失败；`TestQuoteContract` |
| N06-b | 缺失、未来或不可信时间拒绝 | ✅ | 同上 |
| N06-c | 缺独立报价只可走保守减仓 / 紧急平仓 | ✅ | `test_a_reduction_needs_no_quote` |
| N06-d | 允许的交易 session；按常规盘 / 盘前 / 收盘后定义报价合同 | ✅ | 收盘前 30 分钟内 / 盘后 / 下一交易日盘前或盘中可用；更早的盘中报价、当天盘前、非交易日时间戳拒绝；`TestQuoteSessions`（6 条，先写、先看它失败） |
| N07-a | 开始前持久化 run，分批追加，异常终止保留未完成标记 | ✅ | `TestIncompleteTrialHistory` |
| N07-b | 统计恢复前门阻断正式晋级或标为探索 | ✅ | 同上（两个门都判不通过） |
| N07-c | 用审查方脚本验证 | ✅ | 审查脚本那一项因 `record_trials` 被删而 ImportError，改写成等价场景 `test_a_lost_trial_write_blocks_the_gate_right_away`。**这次验证查出了一个真缺口**：此前漏记的运行要等 6 小时过期才被门看见，写入暂时失败、恢复后马上评估仍会通过；现在结束时标 `incomplete`，立即生效 |
| N08-a | 明确昨日权益 / 交易前市值 / 交易后净值三个量 | ✅ | 审查脚本断言失败；`test_the_rebalance_fee_uses_the_pre_trade_value` |
| N08-b | 独立股数 + 现金参考账本覆盖涨跌、现金、空头、费用、部分成交 | ✅ | `tests/unit/backtest_engine/test_engines_match_share_cash_ledger.py`：股数 + 现金 oracle（4 票、空头、借券、D 每日只能部分成交、中途换仓），PaperBroker 与 BacktestEngine 都对到 rel 1e-9；另有区分力前提断言。**先失败**：两套引擎每天偏约 费用率 × 收益率（成交后权重没按成交后净值重新归一，两套一起偏，互相对账看不出）。修：持仓 = 成交权重 × 调仓前市值 / 成交后净值（paper_broker / backtest_engine）。口径：成交记录 `filled_weight` 仍是成交那一刻（相对调仓前市值），持仓表是相对成交后净值 —— 见 position_store 注释。连带改了 12 条按旧口径写的断言（test_paper_broker_accounting 11 条、test_rebalance_accounting 费用 4.95→5.00，后者是我上一轮写错、恰好把这个偏差钉成了期望）；epsilon 守卫组改用零成本券商以保持逐位构造 |
| N09-a | CLI 不再绕过冻结边界 | ✅ | 审查脚本断言失败；`test_cli_gp_never_hands_frozen_dates_to_gp` |
| N09-b | 审查其余旧 DataPartitioner 调用，不能只改这一行 | ✅ | **上一轮标 ✅ 是错的**：当时只查了 `DataPartitioner(` 的直接调用，漏了 chat `tool_run_gp_optimization(dataset_name=…)` 经 `load_real_dataset` 对全量 70/30 切 —— 先写的验收测试实测 GP 的 OOS 含 474 个、IS 含 28 个冻结日期。这次改按"所有加载真实数据集的地方"（`load_registry_dataset` / `load_real_dataset` / `load_pit` 全部调用点）逐个查：研究用途的都改走 N09-c 的上下文；`/regime`、`/datasets/{name}/health`、成本校准、日更摄取是行情 / 运营用途，不是选择流程，未改。`test_chat_gp_on_a_named_dataset_never_sees_frozen_dates` |
| N09-c | CLI / chat / 工作流 / 发现 / 回测 / 验证 / 提案取得同一显式 ResearchContext（dataset / universe / experiment / boundary） | ✅ | `app/core/data_engine/research_context.py`：`open_research(dataset, key, purpose, experiment)` 是研究窗口**唯一**的切法（`research_window` 已删）；上下文带 dataset_key / universe（选择段标的哈希 + 个数）/ experiment（缺省 `settings.research_experiment`）/ purpose / boundary。入口：CLI 4 个、chat（会话 + GP 指定数据集 + Optuna）、工作流 A/B、发现、API（`_resolve_dataset` / `_three_way` / agent / gp / optimize）、验证门、提案。`with ctx.active()` 下的 GP / Optuna 运行把上下文写进 `trial_run_state.context`，拿到冻结段日期 → 不登记、不开始。顺带统一了兜底：`s_three_way_enabled=False` 或验证门 `three_way=False` **关不掉已冻结的边界**（此前 API 与验证门会看全量）；只有"数据太短、又没有边界"才看全量；其它异常不再退回全量（此前 API `_three_way` 与验证门任何异常都退回全量）。`tests/integration/test_research_context.py` 16 条，先写、先看失败（结构检查列出 8 处各自切的地方）。**未做（D12，待定）**：边界仍按 dataset_key 持久化，不按 dataset+universe+experiment 分别登记 —— 现在只是每次运行都带着这三项、可分开看 |

## §2 首轮 F 组的剩余范围（不在 N01–N09 里的）

| 组 | 剩余 | 状态 |
|---|---|---|
| F02 | 不是完整的公司行动账本 | ⬜ |
| F04 | 冻结不能清洗已看过的数据（D12 登记） | ⛔ |
| F08 | 锁依赖共享存储路径，不同目录的两个部署不共锁 | ✅ | 见 §5.3 |
| F12 | 协方差有效性、估计误差需研究评估 | ✅ 评估已做 | `docs/MIN_VARIANCE_STUDY.md` + `tools/research/min_variance_study.py`（蒙特卡洛，真协方差已知，三种设计）。结论：① 原最少样本 max(2n, 20) 太低 —— 等波动时 8 因子 20 天，比等权差 10% 以上的概率 86% → 改为 max(15n, 60)，该概率 ≤ 4%（先写用例、先失败）；② 协方差非平稳时多给数据也没有可靠优势 → 不能假定 min_variance 更好，只能由策略门的流程滚动前推 OOS 判断，保持非缺省。没覆盖：厚尾 / 波动聚集、> 8 个因子、收缩强度的选择、真实因子收益上的检验 |
| F13 | 股本 fallback 是估计；小比例送股只告警不更正；8-K 与 SUE 按时间先后而非期间配对 | ✅ | ① SUE 按盈利期间配对（`pair_announcements`：只用公告日之前已知的财季日历推出公告报告的季；只有更新的季才置空 SUE；推不出 → 按时间，偏保守）；② 股本按自己那条事实的期末日过期（`sh_end`，`METRICS_VERSION` 4）；③ 估计股本（稀释加权平均）进报告 `estimated_shares`；④ 1.15–1.5 倍复权因子跳变（moomoo 路径）持久化 + 估值字段**异常隔离**到口径日晚于跳变日的股本出现为止。先失败（真缺陷复现：同季更正 8-K 把 SUE 置空 2024-05-16→07-24）。**真实数据冒烟**（scratchpad 里 Phase 10 取下的 95 只真实 EDGAR 事实，只读不联网）：面板期内公告推出的财季 vs 之后实际申报的那一季 —— 1988 条一致；26 条不一致中 25 条是"申报之后的 2.02 公告"（更正 / 展望 / 重述，配到已申报那季正是要的行为），1 条（SPGI 2025-02-11）是本地事实缺 FY2024 10-K 净利润造成的对账假象，配对本身正确；12 条推不出。1175 个标的日的 SUE 不再被同季公告误置空。估计股本：ACN / CMCSA / MA / META / NKE / SPG / UPS。疑似送股的隔离需要 moomoo 的复权价，离线没验（单元用例覆盖）。`tests/unit/data_engine/test_fundamentals_sue_visibility.py`、`test_fundamentals_share_counts.py`、`test_fundamentals_split_detection.py` |
| F14 | D14 时钟、D15 通知、持续熔断维护、恢复演练、现场验收 | 🟡 / 🧪 | 代码已做：D14（c2）；D15 桌面通知 + 事件文件、盘中巡检（持续熔断维护）、隔离恢复与合成演练（见 `DECISION_IMPLEMENTATION_TRACKER.md`「告警巡检 / 恢复」）。待现场：通知现场确认可见、巡检与订单在真 OpenD 上验收（D10）、活库恢复（D2） |

## §5 其他风险

| # | 内容 | 状态 |
|---|---|---|
| 5.1 | D14 未统一前，历史结果应分版本标识 | ✅ | 唯一来源 `input_identity.CLOCK_VERSION`（当时为 c1；2026-10-22 D14 起为 c2，口径写在 `CLOCK_VERSIONS`）。带时钟的：研究指标（`alpha_records`）、因子 IC（`alpha_ic_history`）、信号存档、策略证据、规格（`clock` 键 → 时钟一变规格哈希就变，旧证据不被继承）、提案 verdict、验证门结论、§7.3 每日记录、预登记设计。旧行 NULL = c1（那之前只有过这一种）。先失败。`tests/integration/test_trading_clock.py` |
| 5.2 | 规格哈希覆盖有效股票池 / 数据 vintage / 代码版本 / 执行账户 / 报价规则 / 提交时段 | ✅ | 规格 v2 加了 universe（登记标的列表哈希）、execution（账户、限价带、价格偏差、日亏、报价年龄与收盘窗口）、submission（调度时刻、时区、DAY、盘外成交）；调度时刻与订单有效期改成具名常量，规格读的就是代码实际用的值。**数据 vintage 不进规格**（每天都变，进了就天天不一致），记在证据行；**代码版本**记在提案 verdict 的 `approved_code_version` 与每条证据行，不当作不一致（否则每次部署都要重新审批）；`TestSpecCoverage`（9 条，先看它失败） |
| 5.3 | 锁按账户 id 建锁域；首轮专用账户、单部署、固定路径并**校验账户身份** | ✅ | 锁名 = 券商 / 证券公司 / 环境 / 解析出的账户 id，放在整台机器共用的目录（`ACCOUNT_LOCK_DIR`，默认 %PROGRAMDATA%\QuantAgent）；拿不到账户 id 不交易；执行库第一次使用时绑定账户，之后连到别的账户 → 拒绝（`moomoo_trd_acc_id=0` 自动取到另一个模拟账户的情形）；`TestAccountLockDomain` |
| 5.4 | 基本面来源 / 期间质量门 | ✅ | `fundamentals/quality.py`：策略表达式用到基本面字段时，这份面板末日的质量报告（`attach_fundamentals` 每次落盘）必须过门 —— 各字段末日覆盖率 ≥ `FUND_MIN_COVERAGE`（0.8）、估值字段用估计股本的标的占比 ≤ `FUND_MAX_ESTIMATED_SHARES`（0.1）；报告缺失 / 不是这份面板的 → 不过。提案不过 → `passed=False` + 原因；每日循环 active 策略不过 → 不调仓（`fundamentals_quality`）。按真实冒烟的数，现在的 gross_margin（58%）会被拦。`tests/unit/data_engine/test_fundamentals_quality_gate.py` |
| 5.5 | 告警、恢复演练、盘中熔断说明 | 🟡 / 🧪 | 说明已在 OPERATIONS §7 原处改写（盘中巡检、告警、回环、备份恢复）；现场确认与活库恢复未做 |
| 5.6 | 统计表述收敛（README） | ✅ |

## §6 决策建议

| D | 报告建议 | 状态 |
|---|---|---|
| D1 | 选择段内预登记诊断，暂不晋级 | 无需动作 |
| D2 | 备份后版本化重算活库旧 ic_ir | ⛔（要动活库） |
| D3 | OneDrive 外按 lock 建长期测试环境 | ⛔（你的机器环境） |
| D4 | 公开仓库改合成夹具 | ⛔ |
| D5 | 先补 N01–N09 与恢复关键路径的变异复核 | ⛔（小时级，按约定先问） |
| D6 / D7 | 保持关 / 保留 NaN | 现状即如此 |
| D8 | 不做新闻稿级 SUE，先把名称 / 期间 / 配对写清 | ✅ 名称已改；期间配对见 F13 ①，配对结果进报告 `sue_pairing` |
| D9 | 允许预热，**优先实现** | ⛔ |
| D10 | 人工监督的独立验收 | 🧪 |
| D11 | 改成分模式说明 | ✅ |
| D12 | 显式登记 dataset + 日期，保守取最早曝光起点 | ⛔（已加只读查询 `earliest_exposed_start`） |
| D13 | N05 修好后重新提案，不手工补哈希 | ⛔（条件已具备） |
| D14 | T 收盘冻结、T+1 按规则执行，回测 / IC 同步重算并版本化 | ⛔ |
| D15 | 落地一个通知渠道，区分正常 no-op 与异常跳过 | ⛔ |
| — | 路线图删掉"全部修复"、改原表而非追加脚注 | ✅ |

## §7 走到正式前向还要做的

| 项 | 状态 |
|---|---|
| 7.1 六个工作包 | 部分（见上面各行） |
| 7.2 真 OpenD 的 8 个验收场景 | 🧪 |
| 7.3 正式前向每日记录：信号 / 目标生成与存档时间、预期与实际 session、目标股数、实际成交、未成交原因、现金、成本、净收益、基准超额、回撤、gross / net、资金利用率、对账状态 | ✅ 记录机制；**启动是人的事（D13 之后）** | 逐项核对：此前这些散在诊断 JSON、执行库订单 / 成交 / 快照、组合账本、信号存档五处，没有"这一天"的记录，也没有预期 session、基准超额、环境版本、"正式 / 联调 / 回放"的区分。新增 `app/db/forward_record.py`：`ForwardRecordStore.start`（启动记录：唯一启动时间、规格哈希、代码版本、数据身份、环境版本；同一策略 + 规格只能启动一次）；每日循环每轮写一条 `ForwardDay`（字段逐项对应报告原话，见 `REQUIRED_FIELDS`；算不出的值为 None 并在 `missing` 写原因）。`formal` 只在 有启动记录 + 决策日不早于启动 + 在决策日收盘到下一交易日开盘之间运行 + 执行层在线未阻断 时为真；事后补跑不能覆盖已经是正式的那一天。`tests/integration/test_formal_forward_daily_record.py` 4 条（多日端到端，假券商真实成交）——验收用例先于每日循环的挂接写好，挂接前 4 条都失败（`b9xyi3snd` 那次运行）。**已知缺口**：券商逐笔佣金接口没给（记 None + 原因）；基准缺省等权宇宙，选哪个是 D16；**没有启动任何正式前向**（我不做） |
| §8 审查方首轮关键套件里出现过 1 个无 traceback 的 F | 🟡 无法定位 + 查出一个我们自己的真 flake | `targeted-tests.log` 只有进度点：约 650 条中第 257 条 F，无命令、无用例名、无 traceback；与 476 条关键套件不是同一组 → 从现有产物**无法定位**。排查测试里依赖真实时钟的用例时查出：`test_phase12_execution_wiring` 的报价时间取真实当前时刻，周末 / 假日跑会被 N06-d 的 session 合同拒为 `quote_outside_session`（脚本实测周六拒、周一过）—— 这是我这一轮引入的，**不是**审查方那个 F（他们跑在 N06-d 之前）。已改为固定决策时钟（最近收盘后 90 分钟）。其余真实时钟用例（月度校准 / 保真度按 date.today 取上月）只在跨午夜瞬间可能不一致，未改 |
