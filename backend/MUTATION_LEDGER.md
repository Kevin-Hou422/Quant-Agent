# 变异测试台账

**这份文档只保留三样东西**：判据、踩过的坑、以及指向机器可读产物的指针。
所有数字、清单、缺陷登记都已经落在**可执行**的地方 —— 散文里的数字会过期，
测试里的不会。

| 你想知道的 | 去哪看 | 谁保证它不过期 |
|---|---|---|
| 每个模块的点数 / 击杀数 / 存活数 | `tests/meta/measured_modules.json` | `tests/meta/test_invariants.py::TestEveryModuleIsMeasured` —— `app/` 下有变异点却不在清单里的模块直接判红 |
| 还欠哪些产品缺陷没修 | `tests/meta/test_known_defects.py` | 每条一个 `xfail(strict=True)`，每次运行的 `N xfailed` 就是欠账数；修好变 XPASS → 判失败，强制同步 |
| 某个存活变异为什么是等价的 | 各测试文件的 `PROVEN_EQUIVALENT` | `test_lessons_enforced.py::TestLessonX` —— 说明少于 40 字、点名了不存在的验证用例、缺少存活数声明，都判红 |
| 怎么自己复跑一遍 | `tools/mutation/README.md` | `TestLessonW` 检查工具仍在隔离副本里跑、变异器没被静默丢弃 |

## 完成标准（用户定义，不得自行降低）

> 「你说结束就代表你认为所有的测试都已经足够严格，并且新排查出的问题都已经
> 完整修好，项目所有模块都已全部覆盖完整的测试，接下来要交付给别人严格检查。」

拆成可判定的三条：

1. **覆盖**：`app/` 下每个有变异点的模块都被测量过 —— 已达成，见上表第一行。
2. **强度**：每一个存活变异，要么被新测试杀死，要么有**书面且可机械验证**的
   等价性证明。"可机械验证"指证明本身也是一条可执行断言
   （例如 `(limit + tol) - limit != tol` 逐个验证 epsilon 守卫的区分值在浮点上
   不可构造），而不是在注释里写一句"我认为它们等价"。
3. **可见**：已知坏掉的东西必须出现在每次运行的汇总行里，而不是躺在 md 里。

## 判定标准：为什么不看击杀率数字本身

`N passed` 只说明"以本套件**能察觉**的方式没有变坏"，说不了本套件能察觉多少。
行覆盖率只说明代码**被执行**过，说不了执行时**有没有被断言**。

击杀率单看同样会误导：`risk_gate` **76.2%** 已达标（剩余 10 处全部证明为等价变异），
而它曾经口头报过的 **100%** 是工具缺陷造成的假数 —— 那批数字**已作废**。
判据是「存活项 100% 处置」，不是百分比。

---

## 结果总览

| | |
|---|---|
| `app/` 下 `.py` | 111 |
| 其中**有变异点** | **88**（其余 23 个是 `__init__.py` 与纯常量模块） |
| 变异点合计 | **1980** |
| 存活合计 | **138**（分布在 44 个模块） |
| 零存活模块 | **44** |

**从未测量**的模块数：**0**。这一条由 `TestEveryModuleIsMeasured` 持续对账 ——
它是本项目吃过亏的地方（见自伤教训 #9），所以做成了测试而不是一句承诺。

### 仍有存活项的 44 个模块

"首测 → 复测"一栏只有一个数字的，表示该模块首测即达标，未做补强。

| 模块（相对 `app/`） | 变异点 | 首测 → 复测 | 存活 |
|---|---:|---|---:|
| `core/alpha_engine/fast_ops.py` | 145 | 6.9% → **87.6%** | 18 |
| `core/portfolio_manager/risk_gate.py` | 42 | **76.2%** | 10 |
| `tasks/daily_trading_loop.py` | 63 | **84.1%** | 10 |
| `core/backtest_engine/transaction_cost.py` | 41 | **78.0%** | 9 |
| `core/data_engine/data_partitioner.py` | 54 | 18.5% → **83.3%** | 9 |
| `core/backtest_engine/portfolio_constructor.py` | 37 | 32.4% → **86.5%** | 5 |
| `core/data_engine/pit_store.py` | 24 | 70.8% → **79.2%** | 5 |
| `core/execution/paper_broker.py` | 26 | **80.8%** | 5 |
| `core/workflows/alpha_workflows.py` | 69 | 20.3% → **92.8%** | 5 |
| `core/backtest_engine/performance_analyzer.py` | 92 | 30.4% → **95.7%** | 4 |
| `core/backtest_engine/realistic_backtester.py` | 37 | 32.4% → **89.2%** | 4 |
| `core/gp_engine/gp_engine.py` | 15 | 0.0% → **73.3%** | 4 |
| `core/trading_context/spread.py` | 21 | 47.6% → **81.0%** | 4 |
| `core/data_engine/local_parquet_provider.py` | 21 | 71.4% → **85.7%** | 3 |
| `core/ml_engine/proxy_model.py` | 8 | 12.5% → **62.5%** | 3 |
| `core/alpha_engine/signal_processor.py` | 18 | 44.4% → **88.9%** | 2 |
| `core/backtest_engine/overfit_stats.py` | 10 | 20.0% → **80.0%** | 2 |
| `core/backtest_engine/risk_report.py` | 32 | 3.1% → **93.8%** | 2 |
| `core/data_engine/market_calendar.py` | 22 | 77.3% → **90.9%** | 2 |
| `core/gp_engine/evaluation_utils.py` | 10 | 0.0% → **80.0%** | 2 |
| `core/lifecycle/leak_filter.py` | 9 | **77.8%** | 2 |
| `core/lifecycle/validation_gate.py` | 8 | 25.0% → **75.0%** | 2 |
| `core/portfolio_manager/manager.py` | 14 | 35.7% → **85.7%** | 2 |
| `core/portfolio_manager/strategy_gate.py` | 31 | 0.0% → **93.5%** | 2 |
| `core/strategies/baselines.py` | 3 | **33.3%** | 2 |
| `db/run_manifest.py` | 14 | 14.3% → **85.7%** | 2 |
| `agent/_fallback.py` | 4 | 0.0% → **75.0%** | 1 |
| `agent/_tools.py` | 22 | 4.5% → **95.5%** | 1 |
| `api/router.py` | 75 | 16.0% → **98.7%** | 1 |
| `core/alpha_engine/parser.py` | 13 | 15.4% → **92.3%** | 1 |
| `core/backtest_engine/alpha_combiner.py` | 25 | 64.0% → **96.0%** | 1 |
| `core/data_engine/dataset_filters.py` | 54 | 77.8% → **98.1%** | 1 |
| `core/data_engine/multi_dataset.py` | 23 | 78.3% → **95.7%** | 1 |
| `core/data_engine/providers/ccxt_provider.py` | 12 | **91.7%** | 1 |
| `core/data_engine/regime_detector.py` | 17 | 52.9% → **94.1%** | 1 |
| `core/gp_engine/alpha_pool.py` | 22 | 59.1% → **95.5%** | 1 |
| `core/lifecycle/promotion_gate.py` | 13 | **92.3%** | 1 |
| `core/ml_engine/alpha_optimizer.py` | 24 | 4.2% → **95.8%** | 1 |
| `core/trading_context/context.py` | 11 | 45.5% → **90.9%** | 1 |
| `db/alpha_store.py` | 23 | 26.1% → **95.7%** | 1 |
| `db/diagnostics_store.py` | 7 | 28.6% → **85.7%** | 1 |
| `db/position_store.py` | 25 | 24.0% → **96.0%** | 1 |
| `db/strategy_store.py` | 13 | 38.5% → **92.3%** | 1 |
| `db/trial_ledger.py` | 8 | 50.0% → **87.5%** | 1 |

### 零存活的 44 个模块

`agent/_agent.py`、`agent/_chat_history.py`、`agent/_critic.py`、`agent/_data_utils.py`、`agent/_helpers.py`、`agent/_lc_agent.py`、`agent/_memory.py`、`agent/alpha_agent.py`、`api/chat_router.py`、`config.py`、`core/alpha_engine/dsl_executor.py`、`core/alpha_engine/financial_diagnostics.py`、`core/alpha_engine/financial_interpreter.py`、`core/alpha_engine/typed_nodes.py`、`core/alpha_engine/validator.py`、`core/backtest_engine/backtest_engine.py`、`core/backtest_engine/multi_dataset_backtester.py`、`core/backtest_engine/visualizer.py`、`core/data_engine/base.py`、`core/data_engine/dataset_registry.py`、`core/data_engine/health_report.py`、`core/data_engine/providers/akshare_provider.py`、`core/data_engine/providers/moomoo_provider.py`、`core/data_engine/schema.py`、`core/data_engine/sector_mapper.py`、`core/data_engine/yahoo_provider.py`、`core/discovery/discovery_engine.py`、`core/discovery/market_observer.py`、`core/gp_engine/fitness.py`、`core/gp_engine/mutations.py`、`core/gp_engine/population_evolver.py`、`core/ml_engine/alpha_evaluator.py`、`core/monitor/alpha_monitor.py`、`core/portfolio_manager/horizon.py`、`core/portfolio_manager/strategy_builder.py`、`core/trading_context/providers.py`、`db/alpha_lifecycle.py`、`db/chat_store.py`、`main.py`、`tasks/backup.py`、`tasks/cost_calibration.py`、`tasks/daily_ingest.py`、`tasks/reasoning_log.py`、`tasks/scheduler.py`

### 已知未闭合的缺口（只有这一个）

138 个存活项的**模块级逐条归属**还没做。已经建立的是：

- 88 个模块全部测量过（机器对账）
- 44 个写了证明的文件，条目数与各自声明的存活数**逐个相等**
- 每条证明说明 ≥40 字、点名的验证用例真实存在（机器检查）

做不到的是把 138 个存活**逐条**对上 97 条证明 —— `PROVEN_EQUIVALENT` 的键是
自由文本（如 `"L124 (weights < -tol) → <="`），不带模块路径；按 import 归属会错判
（`risk_gate` 的 10 条证明在它自己的文件里，而那个文件并不直接 import 该模块）。

**闭合办法**：把每条键改成 `<模块路径>:L<行号> <变异描述>` 的固定格式，
再加一条 meta 测试逐条对账。约 97 条键需要改写。

这个缺口连同棘轮（证明条数只许增不许减）写在
`measured_modules.json` 的 `proof_reconciliation` 块里，
由 `TestEveryModuleIsMeasured::test_the_open_reconciliation_gap_stays_visible` 守着 ——
删掉那个块或让证明条数变少都会判红。

---

## 工具缺陷史（每一条都曾让整批数字作废）

度量工具的缺陷**不报错，只静默缩小分母** —— 这是最危险的一类，
因为数字看起来完全合理。九条都留在这里，是为了下次"数字看起来合理"时
不要重新相信它。

| # | 缺陷 | 后果 |
|---|---|---|
| 1 | 变异后没有 `ast.parse` 校验 | 语法错让测试立刻红，被**误记为"杀死"** |
| 2 | 替换行丢了行尾换行 | 与下一行粘连，同样制造假"杀死" |
| 3 | 原地改主工作区，不建隔离副本 | 变异残留被提交进仓库（commit `24c251a` 真实发生过） |
| 4 | 每行只取第一个变异器 | 比较符排在算术符前面，乘除法**永远轮不到** |
| 5 | `re.sub(pat, rep, m.group(0))` | `m.group(0)` 不含 lookaround 消耗的字符，**正向后顾断言**在孤立片段上必然失配 → `*` `+` `-` 三个算术变异器**从未生效过**且不报错。`transaction_cost` 129 个候选行只有 22 行被变异过 |
| 6 | 不屏蔽行尾注释 | 注释里的 `>` `<` 被变异，记出假存活项 |
| 7 | `subprocess.run(timeout=)` 只杀直接子进程 | pytest 的孙进程握着管道，`communicate()` 继续阻塞 → 超时上限形同虚设（`alpha_workflows` 一轮跑了 11.6 小时）。改用 `taskkill /F /T` 杀进程树 |
| 8 | 改完测试没有重测 | 台账里的存活列表过期，照着它补用例等于在补已经修好的洞 |
| 9 | `_string_spans` **逐行** tokenize | 模块级三引号字符串的中间各行单独 tokenize 不是合法 Python，走进 `except` 后返回空区间 → **整段散文被当成代码变异**。`_prompts.py` 因此报出 35 个假变异点，`alpha_agent.py` 混进 1 个 |

修 #9 之后必须验证分母没被误伤 —— 逐模块对比修复前后的点数：
**109 个模块不变**，只有那两个含散文常量的变了（`_prompts` 35→0，`alpha_agent` 11→10）。
其余各档的分母一个没动。

#3/#5/#6 由 `TestLessonW_KillRateLedgerIsMaintained` 持续强制。

---

## 自伤教训（我自己写错的断言）

| # | 症状 | 根因 | 现在由谁强制 |
|---|---|---|---|
| 1 | 断言写成恒真形状（`assert x in (200,400,500)`、`assert ... or True`） | 只想让它绿 | `TestLessonA::test_no_tautological_assert` |
| 2 | 整个测试体被 `try/except: pass` 包住 | 不想处理异常 | `TestLessonA::test_no_test_body_fully_wrapped_in_except_pass` |
| 3 | 零断言测试（只调用不检查） | 以为"不抛就是对" | `TestLessonA::test_no_zero_assertion_test` |
| 4 | `if <条件>: assert ...` —— 条件不成立就一条都不检查 | 守卫与契约混淆 | `TestLessonA::test_no_generic_conditional_assert` |
| 5 | 模块写了没接线 | 写完就以为完了 | `TestLessonK` / `test_invariants.py::test_no_orphan_modules_outside_allowlist` |
| 6 | **源码文本断言用子串判断** —— `assert "daemon=True" in src`，而 `router.py` 里 `threading.Thread(...)` 有三处、其中两处一模一样，改掉任意一处断言仍为真 | 把"文本存在"当成"约束成立" | `TestLessonY`（棘轮：存量 65 处，只许降不许升；新增必须改用 AST 全称量化） |
| 7 | **能杀却写了等价证明** —— `alpha_pool.py:204` 只看了 `top_k` 与 `_seen_dsls`，漏了 `all_entries()` 的插入序；`population_evolver.py:637` 的输出被 `[:pop_size]` 截断，换个观察面（数算子调用次数）立刻能杀 | "输出相同"只是没找到对的观察面，不等于"观察不到" | `TestLessonX`：每条证明必须配一条真实存在的可执行验证用例 |
| 8 | **变异测试会把代码跑在你没预期的配置下** —— `visualizer.plot()` 的 `show: bool = False` 被变异成 `True`，三十多条"画图再检查 trace"的用例**每条都真的打开了一个浏览器标签**，一次性在使用者屏幕上弹出几十个 | 隐含前提"这个参数默认是 False，所以别的用例不会触发它"——而这个前提正是变异要破坏的 | `conftest.py::_never_open_a_browser`（session 级总闸）+ `test_invariants.py::TestNoOutOfProcessSideEffects`（三条：总闸在不在、调用是否只记账、两个出图入口的默认值是否还是 False） |
| 9 | **把为人眼截断过的打印输出当成清单** —— 清点脚本写了 `sorted(todo,…)[:40]` 和 `[:15]`，实际有 57 / 26 个。按点数降序排在末尾的小模块就这么掉出了清单，4 个模块因此从未测量（首测全部 **0.0%**） | 脚本把全量写进了 json，我用的却是终端里滚出来的摘要 | `test_invariants.py::TestEveryModuleIsMeasured` |

**#8 的验证不是推测**：把当初闯祸的那一个变异原样重跑了一遍 ——
`visualizer.py L56 show: bool = False → True` 结论 **[OK] 变异被杀死**，
且没有弹出任何标签页。这个变异从"存活"变成了"被杀死"。

**#9 还有一个附带教训**：第一版审计脚本用 `grep` 全文匹配模块名，
命中的是变异记录里 `code` 字段的**代码文本** —— `paper_broker` / `_fallback`
被误判成"已测量"，而它们当时一个测过一个没测过。
**判据错了比没判据更危险**：它会给出一个看起来令人安心的答案。

---

## 存活项为什么会存活：六种花样

首测击杀率低，从来不是"忘了写测试"，而是**观察面选错了**。六种实测到的形态：

### 1. 参数组合恰好让两种取值同解
`>` 与 `>=` 只在**恰好等于阈值**那一格不同。测试用的参数离边界太远
（如用 `oos=0.1` 去测 `< 0.2`），两种取值给出同一个结论。
**破法**：把指标**精确**构造到阈值上 —— `adv20 == 100_000_000.0` 而不是"差不多一亿"；
够不着时用 `np.nextafter` 取相邻浮点。

### 2. 下游把差别吸收掉了
`clip` 把 -1e9 和 -3 都压成 0；`np.nansum` 把 NaN 当 0。
**破法**：在被吸收**之前**取观察点，或逐位比对参考实现。

### 3. 前置层把被测层掩盖了
复杂度测试喂的是 stub 节点，直接调 `node_count()`，`_count_nodes` 整个被绕过。
**破法**：确认调用链真的经过被测函数（加一次计数）。

### 4. 尺度不变性把整条比值吸收掉
`_ic_ir` 算的是 `mean/std`。每天的 IC 被同一个常数缩放时比值原封不动 ——
而测试面板**每天的有效截面宽度都一样**，于是 `denom` 的 `*` 改成 `/`
只是整体乘了个常数。
**破法**：让每天的有效标的数不同（NaN 模式轮换），缩放因子随 t 变化 ——
实测 IC-IR 从 1e9 掉到 2.9。

### 5. 夹具的构造自带对称性
`high = close * 1.02`、`low = close * 0.98` 看起来是"整洁的测试数据"，
但它让 `(h + l + c) / 3` **恰好等于 close** —— "把 vwap 派生公式改成直接取 close"
完全观察不到。
**破法**：刻意不对称（+5% / −2%）。

### 6. 只有日志看得见的状态位
`passed_any` 不影响任何返回值，只决定末尾那一行日志 ——
而那行是无人值守跑批时**唯一**能告诉操作者"这一轮有没有产出"的信号。
**破法**：用 `caplog` 钉住日志。**日志是产品的一部分**，不是调试残留。

---

## 已登记产品缺陷（25 条，只登记不修）

编号、一句话描述、以及断言"应有行为"的 `xfail(strict=True)` 用例，
全部在 `tests/meta/test_known_defects.py`。**那里是权威**，
本表只给索引，由 `test_defect_registry_matches_the_ledger` 机械核对两边一致。

| 编号 | 一句话 |
|---|---|
| B-1 | `fast_ops.bn_ts_rank` bottleneck 分支值域是 [-1/w, 1/w]，非 docstring 承诺的 [0,1] |
| B-2 | `fast_ops.ts_corr` cov(ddof=0)/std(ddof=1) 不配套 → 系统性偏低 (w-1)/w |
| B-3 | `fast_ops.cs_rank` 并列处理是序数名次，非声称的平均名次 |
| B-4 | `fast_ops.ts_entropy(n_bins=1)` 静默返回 -0.0，而非 NaN/报错 |
| B-5 | `fast_ops.bn_ts_max/min` NaN 策略在两条分支之间不一致 |
| B-6 | `fast_ops.cs_rank` 含 NaN 的截面上值域越出 [0,1] |
| B-7 | `fast_ops` 七个滚动算子在面板短于窗口时抛 ValueError，而非返回 NaN |
| B-8 | `proxy_model._fit()` 放弃后仍走模型分支 → AttributeError |
| B-9 | `proxy_model` 的 `use_label_encoder=False` 对 xgboost 3.x 已无意义（仅整洁问题） |
| B-10 | `fast_ops` 向量化分支被 `except Exception` 完全兜住（结构问题） |
| B-11 | `data_partitioner` 的「OOS 为空」守卫不可达（结构问题） |
| B-12 | `chat_store` 的 ORDER BY 无第二排序键，同一 tick 内顺序反了 |
| A-1 | `ingest_incremental` 把增量写进 PIT 两次 |
| A-2 | `strategy_gate` 用 `np.nanstd(...) == 0.0` 判零方差，浮点零判不出来，守卫从不触发 |
| A-3 | `PerformanceAnalyzer` 对近零波动无防护：全常数收益算出年化 Sharpe ≈ 3e16 |
| A-4 | `PerformanceAnalyzer` 遇非日期索引先在 `_tdays` 抛 AttributeError，且报错指向内部实现 |
| A-5 | `MVOPortfolio` 注释写「剔除的资产保留基准权重」，实现是整行替换 → 拿到 0 |
| A-6 | `PaperBroker.step` 写死 `target=1.0`，把目标总敞口强行放大到 L1=1 |
| C-1 | GP 适应度的截面秩用 `argsort(argsort(x))`，不处理并列 → 零信息信号被按**列顺序**摊开，IC 成了伪相关 |
| C-2 | `mutations._replace_node` 先 deepcopy 再按 `id(target)` 找节点 → 除非 target 是 root，替换**永远静默失败**；`add_ts_smoothing` 在多数情况下是彻底的空操作 |
| D-1 | `financial_interpreter` 只认 `neg` 节点：`-x` 判 reversion，语义相同的 `(0-x)` 判 momentum |
| D-2 | `LocalParquetProvider` 宣称支持 `returns`，按此请求会让**整批数据返回空**（连 close 都没有） |
| D-3 | `langchain>=0.2` 无上界，装上的 1.x 已移除 `AgentExecutor` → LLM 链路整条**静默降级**，只打一条 warning |
| D-4 | 系统提示词把 `rank(neg(...))` 当作 4 个因子家族的标准模板，而解析器不认 `neg(x)` |
| D-5 | 提示词写 `corr > 0.9`，`AlphaPool` 实际默认 `0.70` 且用 `>=` |

### 为什么要有 `test_known_defects.py`

此前这些缺陷只被"钉住当前错误行为"的断言覆盖（断言错的值，注释写"修好后改成对的值"）。
钉住现状有价值 —— 任何一处算术被改坏仍会被抓到 —— 但它有个致命副作用：
**已知坏掉的东西在每次运行里完全不可见**，二十几个缺陷躺着，套件照样报全绿。

`xfail(strict=True)` 补上另一半：断言**应有行为**。于是每次运行的汇总行显示
`N xfailed`，欠账摆在台面上；谁修好了，用例变 XPASS → strict 判失败 →
强制他同步更新三处（xfail 标记、模块里钉住现状的断言、本台账索引）。

两套断言互补：钉住现状的那条提供**检出能力**，xfail 那条提供**可见性与修复告警**。

---

## 目录整理（2026-09-13）时炸出的三个真问题

整理 `tests/` 时顺带发现的，不是变异测试查出来的 —— 记在这里是因为
它们都属于"测试自己骗自己"这一类：

1. **`/api/chat/stream` 零测试**，却因为某个文件的 docstring 里提了一句
   就被算成"已覆盖"。已补 `test_api_chat_stream.py`。
2. **共享 session 级 DB 造成顺序依赖**：`StrategyStore()` 默认走
   `settings.database_url`，一个泄漏的 `status="active"` 配置让
   `run_portfolio` 跳过边际选择。已用 autouse fixture 隔离。
3. **`importlib.reload(app.main)` 污染整个会话**：它新建一个 `FastAPI()` 实例，
   而 session 级 `test_client` 抓着旧的，后续 `dependency_overrides` 全部失效 ——
   表现为一批不相关的 404/409。已把双重导入检查挪进子进程。

配套新增零依赖插件 `shuffle_check.py`：打乱收集顺序暴露用例间的隐式依赖
（本项目**没有**装 pytest-randomly，之前 `-p no:randomly` 是在禁用一个不存在的插件）。

```bash
python -m pytest tests/ -q -p shuffle_check
```

---

## 怎么复跑

```bash
cd backend
python tools/mutation/make_plan.py                    # 生成计划：88 模块 / 1980 点
python tools/mutation/runner.py plan_full.json --state progress_full.json
python tools/mutation/runner.py plan_full.json --state progress_full.json --status
```

`progress_full.json` 是可续跑的中间态，已在 `.gitignore` 里。
结论请更新 `tests/meta/measured_modules.json` —— 那份才是交付物。

早期每轮测量各写一个 `plan_*.json` + `progress_*.json`，攒下约 100 个文件。
它们对复核没有价值（复核者要的是"照同一套规则重跑"，不是当时分了几批），
已全部删除，改用 `make_plan.py` 重新生成。
