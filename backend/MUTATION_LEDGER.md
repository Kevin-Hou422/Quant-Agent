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
   等价性证明。"可机械验证"指证明本身也是一条可执行断言，而不是在注释里写一句
   "我认为它们等价"。

   > **这一条曾经举错了例子。** 原文举的是 `(limit + tol) - limit != tol`
   > ——"逐个验证 epsilon 守卫的区分值在浮点上不可构造"。外部审计 2026-09-15
   > 给出反例推翻了它：`project_to_capped_l1([[1e-12, 1-1e-12]])` 逐位保留
   > 1e-12，于是 `> 1e-12` 与 `>= 1e-12` 结论不同。那条断言证的是
   > **"tol 不能由一次加法还原"**，而到达 `filled[i]` 的值根本不必来自加法。
   >
   > 教训：**可执行 ≠ 证对了**。一条跑得起来的断言，证的可能是一个更弱的命题，
   > 然后被当成结论用。所以第 2 条现在还要求：证明必须说清楚
   > **"被测的值可能从哪里来"**，并对每条来源给出反驳尝试；
   > 站不住的证明移入 `REFUTED_EQUIVALENCE`，由
   > `test_invariants.py::test_refuted_proofs_cannot_quietly_come_back` 看着，
   > 不许再写回去。

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
| 其中**有变异点** | **89**（旧算子集下是 88；`_sqlite_utils.py` 是算子补齐后新进来的） |
| **已测量**的变异点 | **1981**（1980 个在 MUTATORS v1 下枚举，+1 个补测） |
| 存活合计 | **136**（分布在 43 个模块） |
| 零存活模块 | **44** |
| **当前算子集下应测的点数** | **2843**（MUTATORS v2） |
| → **从未测量**的点数 | **862** |

> **1980 与 2843 的差额必须一直看得见。** 外部审计 2026-09-15 用四个微型探针
> 证明旧算子集有系统性缺口：`x >= .70`、`x / w`、`a+b` **各 0 个变异点**，
> `a + b + c` 只变第一个加号。补齐算子（工具缺陷 #11）后同一份源码枚举出
> 2843 个点 —— 多出来的 863 个从没跑过（其中 1 个已补测，余 862）。
>
> 如果只把新总数写进清单、不记差额，交付物看起来只会**更好**，而"这 863 个点
> 没测过"就消失了。所以差额落在 `measured_modules.json` 的 `measurement_scope`
> 块里，由 `test_the_unmeasured_scope_stays_visible_and_only_shrinks` 做成
> **只许减不许增**的棘轮。**欠一次对 2843 个点的重测，本轮未做。**

**从未测量**的**模块**数：**0**。这一条由
`TestEveryModuleIsMeasured` 持续对账 —— 它是本项目吃过亏的地方
（见自伤教训 #9），所以做成了测试而不是一句承诺。
但注意它对账的是**模块**，不是**点数**，也不是**出处** ——
后两者分别由 `test_the_unmeasured_scope_stays_visible_and_only_shrinks` 与
`test_every_recorded_test_path_still_exists` 补上，两条都是被审计打脸之后才加的。

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
| `core/execution/paper_broker.py` | 26 | 80.8% → **88.5%** | 3 |
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

### 已知未闭合的缺口

外部审计 2026-09-15 之后，这里有 **四** 个，不是原来写的"只有这一个"。
（第一个已从"没对账"变成"对完账、缺口已量化并锁住"。）

**（一）22 个存活变异既未被杀死，也没有书面证明。**

这一条以前写的是"逐条归属还没做"。**归属已经做完了**（2026-09-16）：
97 条证明的键全部改写成 `<模块路径> ×<覆盖点数> — <描述>`，
由 `test_every_proof_key_declares_its_module_and_point_count` 强制格式、
`test_survivor_disposition_reconciles_per_module` 逐模块对账。

对完账才看清真正的问题不是"对不上"，而是**账本身是缺的**：

| 模块 | 存活 | 有证明 | **未处置** |
|---|---:|---:|---:|
| `tasks/daily_trading_loop.py` | 10 | 0 | **10** |
| `core/data_engine/data_partitioner.py` | 9 | 5 | **4** |
| `core/alpha_engine/fast_ops.py` | 18 | 15 | **3** |
| `core/backtest_engine/transaction_cost.py` | 9 | 6 | **3** |
| `core/strategies/baselines.py` | 2 | 0 | **2** |
| 合计 | 136 | 114 | **22** |

`daily_trading_loop` 与 `baselines` 是**一条证明都没有**。

**此前为什么没发现**：旧的自洽检查是"每个证明文件的条目数 == 它自己声明的存活数"
—— **文件内自洽，跨文件汇总从未对过账**。44 个文件各自都"对得上"，
加起来却少了 22 个。这是自伤教训 #13 的又一个实例：判据只覆盖了局部。

顺带修掉的另一个判据盲区：`_count_proof_entries()` 只扫模块级 `tree.body`，
**写在类里的 `PROVEN_EQUIVALENT` 它看不见**（`position_store`、`strategy_store`
各一条）。棘轮下限因此一直比真实值小，那两条从来不受保护。

**闭合办法**：逐模块补用例杀死（首选），或写可机械验证的等价性证明并附反驳尝试。
棘轮 `survivor_disposition.ratchet.unproven_max = 22`，只许减不许增。

**（零）依赖安装入口已定死（原来是模糊的）。**

外部审计 2026-09-15 指出仓库有**两个**安装入口且互不一致：CI 装
`requirements.txt`，仓库里还躺着一份 `requirements.lock`。实测两边都有洞 ——
txt 漏了 `scikit-learn`（CI 连红三次的根因），而 2026-07-30 那版 lock
**漏了 `pandas_market_calendars`**，照它安装会让 `market_calendar.py`
静默退回 `pd.bdate_range`（外部审计 #6 的原始现场）。**两个入口各自复现了
同一类缺陷，而没有任何检查会发现这件事。**

现在的分工写进了两份文件的头部，并由测试强制：

| 入口 | 谁用 | 作用 |
|---|---|---|
| `requirements.lock` | CI **阻塞**任务 | 逐位钉死 → "这批结果是用哪套版本跑的"可复现 |
| `requirements.txt` | CI **非阻塞**任务 | 装当日最新 → **依赖漂移要被看见，而不是被 lock 掩盖** |

- lock 已用跑通完整套件的那个干净环境重新生成（101 个包，含头部的生成步骤）
- `TestLessonC::test_every_declared_requirement_appears_in_the_lock_file`
  逐条核对"txt 里声明的每个包都在 lock 里"，漏一个判红
- CI 不再排除 `tests/performance`：那组此前长期在量一个 422 的延迟
  （`n_days=50` 违反接口的 `ge=60`），改对之后才真的在测东西

**（二）862 个变异点从未测量。** 见上面「结果总览」。补齐算子后应测 2843 点，
已测 1981 点。棘轮：`test_the_unmeasured_scope_stays_visible_and_only_shrinks`。
**欠一次重测。**

**重测的实测代价**（不是估计：各包测试目录的真实耗时 × 各自的变异点数）：

| 方案 | 机器时间 | 说明 |
|---|---:|---|
| 全量 2843 点，按当前选路（包目录 + `tests/meta`） | **206 小时** | 最严格，`tests/meta` 单次就要 ~210s，乘在每个点上 |
| 全量 2843 点，不含 `tests/meta` | **40 小时** | 放弃"变异把某个已登记缺陷修好 → xfail 变 XPASS → 判杀"这一类击杀 |
| 仅新增的 862 点，不含 `tests/meta` | **12 小时** | 已测的 1981 点仍停留在旧判定器口径（击杀率是上界） |

三个方案的严格性不同，**选哪个是取舍不是优化**，需要人来定。
本轮没有启动任何一个 —— 跑起来会占住机器数小时到数天。

命令（可续跑，关机重启后再跑同一条即可接上；全程在隔离副本里，不动工作区）：

```bash
cd backend
python tools/mutation/make_plan.py                 # 89 模块 / 2843 点
python tools/mutation/runner.py plan_full.json --state progress_full.json
python tools/mutation/runner.py plan_full.json --state progress_full.json --status   # 只看进度
```


> 唯一补上的那个是 `app/db/_sqlite_utils.py`：它在旧算子集下显示"无变异点"
> （`!=` 当时没有对应变异器），实际是**一条测试都没有**。补测首轮 1 点 / 0 杀死 /
> **0.0%**，补 `tests/unit/db/test_sqlite_hardening.py` 后用 `verify_mutant.py`
> 单点复核确认被杀。
>
> **这是"算子缺口"造成的漏测的活样本**：不是某个模块测得差，是它**根本不在视野里**，
> 而清单上它看起来和"无需测量的纯常量模块"没有区别。

**（三）测量出处是"重映射"来的，不是重跑来的。** 审计发现清单里 104 个不重复的
测试路径在 Task 1 目录重组后已不存在。已按文件名唯一匹配修回（96 条重映射、
59 条无法映射置空，3 个模块因此完全没有出处记录）。这保证"路径现在能跑"，
**不保证"当初那一版文件的内容与现在相同"**。消除这个不确定性同样只能靠重测。
棘轮：`test_every_recorded_test_path_still_exists` + `test_the_provenance_repair_record_stays_visible`。

**（四）击杀率是上界，不是测量值。** 工具缺陷 #10 修复前，超时 / 收集错误 /
导入失败 / 任何无关的偶发失败都被记成"杀死"。**已测的 1980 个点全部是在那个
判定器下跑出来的**，所以 138 这个存活数是下界、各模块击杀率是上界。
修复后的判定器会把这些归入 `inconclusive` 并从分母剔除，但**旧数字没有重跑**。

---

## 工具缺陷史（每一条都曾让整批数字作废）

度量工具的缺陷**不报错，只静默缩小分母** —— 这是最危险的一类，
因为数字看起来完全合理。十二条都留在这里，是为了下次"数字看起来合理"时
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
| 10 | `run_tests` 把**退出码非 0 一律当成"杀死"**，超时也算，且 `stdout/stderr=DEVNULL` 丢掉全部证据 | 收集错误、导入失败（exit 2/4）、一个测试都没收集到（exit 5、**分母为空**）、超时、以及 `-x` 之下任何无关的偶发失败，统统被记成"断言抓到了"。**偏置方向永远朝着数字更好看。** 已改为按 pytest 退出码分类：只有 exit 1（有测试失败）算杀死，其余归入 `inconclusive` 并从分母剔除；基线不绿直接 `SystemExit`，不再让"根本没测"伪装成"击杀率 0" |
| 11 | 变异器集合有系统性缺口，且每行每算子**只取第一处**匹配 | 外部审计的四个探针：`x >= .70` / `x / w` / `a+b` 各 **0 个**变异点，`a + b + c` 只变第一个加号。我自查又补两个：`x <= 5`、`a == b` 同样是 0。已补 9 个算子并改为枚举全部匹配位置 → 1980 → **2843** 点。**仍在盲区**：数值常量、对象身份/deepcopy（C-2 那一类）、调用实参、控制流结构 |
| 12 | `make_sandbox()` 只复制 `backend/`，仓库根的 `.gitignore` 不在沙箱里 | `tests/meta` 里有检查仓库根 `.gitignore` 的用例，于是**只要测试路径包含 `tests/meta`，沙箱里的基线必然是红的** —— 而 `make_plan.py` 给每个模块都加了 `tests/meta`。旧判定器遇到这种情况只打一句「基线就是红的」就 return，模块**静默没测**，从外面看不出与「跑过了」的区别。**这一条是缺陷 #10 的修复（基线不绿就 SystemExit）当场抓出来的** |

**#10/#11 合起来的结论**：`B-1` 的除法归一化错误与 `C-2` 的对象身份错误
都是**读代码**发现的，不是变异测试发现的。"所选变异全部被处理"
与"检出能力已证明"从来不是同一个命题 —— 台账此前把它们当成了一回事。

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
| 10 | **把"本地全套绿"当成依赖完整的证据** —— CI 的 backend job 自 2026-09-10 起连红三次；本地用同一条命令、在只含已提交文件的干净 clone 上（因而也没有 `.env`）跑了 46 分钟，**3755 passed / 0 failed**。真正的差别是开发机多装了 scikit-learn 1.8.0：`XGBClassifier` 在**构造时**才要求它，而全库没有一行 `import sklearn`，`requirements.txt` 也就一直没声明 | 拿一个"恰好装全了"的环境去验证"依赖声明是否完整"，等于拿被测对象当判据 | `TestLessonC::test_construction_time_deps_are_declared_in_requirements` + `test_xgboost_sklearn_api_is_constructible_not_merely_importable`（后者与环境无关：任何缺 scikit-learn 的机器上都会红） |

| 11 | **期望值是照着实现算出来的** —— `SHARPE_T = 8.32356013267212  # SHARPE * sqrt(120) / sqrt(1+0.5*SHARPE**2)`，注释里那行公式**就是被测代码本身**。产品把年化 Sharpe 配日频样本数，t 被放大约 √TDAYS 倍；同频口径是 0.5660，报告按 1.96 判定时"不显著"被显示成"✓显著" | 判据与被测对象同源。这种断言能检出"公式被改动"，永远检不出"公式本来就错" | 登记为 N-4；`TestSharpeTStatFrequency` 用**同频日 Sharpe** + scipy 单样本 t **双口径交叉验证**后断言应有值，原常量加注"钉住当前错误实现" |
| 12 | **xfail 的失败原因不受约束** —— A-1 在 `inspect.getsource(di.ingest_incremental)` 处抛 AttributeError（那是 `DailyIngest` 的方法，模块上没有这个属性），C-1 在 `_evaluate_individual()` 签名变更处抛 TypeError。两条都在**碰到目标行为之前**就"失败"了，汇总行里的 `xfailed` 计数一直很好看 | `strict=True` 只保证"意外通过要报错"，对"因为别的原因失败"一无所知。把一个 bit（失败/没失败）当成了"登记的原因仍然成立" | `_xfail(defect_id, raises=...)` 默认限定 `AssertionError`；缺陷本身就是抛异常的显式传类型。前置条件必须拆成**不带 xfail** 的独立用例（D-4 已拆） |
| 13 | **教训被代码化成"不许变得更糟"，而不是"把现有的查一遍"** —— `TestLessonY` 的源码文本断言棘轮基线定在 65，于是 A-2/A-5 这两条**正是该教训的实例**被永久豁免；`TestEveryModuleIsMeasured` 只对账模块不对账出处，所以目录重组把 104 个出处路径变成死链它完全看不见 | 棘轮只管增量。写棘轮的时候我知道存量有问题，但把"先止血"当成了"已解决" | A-2/A-5 已改成行为断言并把基线降到 63；出处对账补 `test_every_recorded_test_path_still_exists`；点数对账补 `test_the_unmeasured_scope_stays_visible_and_only_shrinks` |

**#11/#12/#13 是同一件事的三个侧面：判据的独立性从来没有被检验过。**
期望值来自实现（#11）、失败的一个 bit 被当成原因成立（#12）、
守卫只拦新增不查存量（#13）—— 加上工具缺陷 #10（仪器的异常路径偏向好消息）
和 #11（算子集合被当成缺陷空间的代理），五条的共同形状是：
**我验证了"我能想到的那个命题"，没验证"我需要的那个命题"。**

**#10 的复现是对照实验**：同一个 HEAD clone，换本地 venv → `3755 passed`；
换只按 `requirements.txt` 装的干净 venv → `4 failed, 3751 passed`，
失败的恰好只有 `test_proxy_model_pruning.py` 那四条，报错一律是
`ImportError: sklearn needs to be installed in order to use this module`。
补上声明后同一环境复跑 **22 passed**。
附带结论：`requirements.txt` 全是 `>=` 无上界，CI 每次装的是当日最新
（干净装拿到的是 plotly 7.0 / langchain 1.4 / numpy 2.5.3，均高于开发机）——
这一点已作为 **D-3** 登记在案，本轮未改。

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

## 已登记产品缺陷（**0 条未修行为缺陷** + 3 条技术债 + 1 条前端）

**分三类数，不混成一个数**（外部审计 2026-09-15 的判定）：

| 类别 | 条数 | 含义 |
|---|---:|---|
| 行为缺陷（**未修**） | **0** | 有复现、有"应有行为"的 `xfail(strict=True)` 断言 |
| 技术债 | 3 | B-9 / B-10 / B-11 —— 只有结构证据，**没有"产品结果是错的"的复现** |
| 前端 | 1 | N-6 —— 已在当前代码上复现，但后端套件里没有可执行断言 |

混成一个数会让它读起来比实际严重，也会稀释真正该优先修的那几条。
由 `test_the_outstanding_defect_count_is_visible` 三类分别锁住。

**修复进度（第 3 阶段，2026-09-18 起）**：登记时 28 条行为缺陷 → **全部修复完毕（0 条）**。
已修的在下表里以 ~~删除线~~ 标出并写明改法与后果，**不从表里删掉** ——
下一份报告要能看出"修了什么、当初错在哪"。

剩下的 2 条都已获授权，是工程活不是决策，**不是都能直接开工**：

| 性质 | 编号 | 卡在哪 |
|---|---|---|
| 待定契约（需你裁决） | B-4 / A-5 / D-5 | 分别是：单箱熵返回 0 还是 NaN；MVO 剔除资产拿基准权重还是 0；提示词的 `corr > 0.9` 与代码的 `abs(corr) >= 0.9`（绝对值 + 边界都不一致）|
| 设计决策 | A-6 后半 / A-7 | A-6 要统一回测引擎与执行层语义，**会改变历史回测收益**，按你的决定 #4 需要前后对比与差异解释；A-7 是构建层三处写死的 `target=1.0`，每处的上游口径要单独确认 |


编号、一句话描述、以及断言"应有行为"的 `xfail(strict=True)` 用例，
全部在 `tests/meta/test_known_defects.py`。**那里是权威**，
本表只给索引，由 `test_defect_registry_matches_the_ledger` 机械核对两边一致。

| 编号 | 一句话 |
|---|---|
| ~~B-1~~ | **已修（2026-09-20）**：`ts_rank` 统一为**平均秩 /(count-1)**，值域真的是 [0,1]。此前两条分支算的**不是同一个量**：bottleneck 是 `bn.move_rank(...)/window`（实测 move_rank 值域 [-1,1]，再除 w → [-1/w, 1/w]，一半为负且随窗口缩放），numpy 是 `le/count`（取不到 0）。后果不只是数不好看 —— `ts_rank(close,20) > 0.8` 在装了 bottleneck 的环境里**恒为假**（上界才 0.05），于是装没装 bottleneck 决定了信号存不存在，GP 进化出的表达式依赖运行环境。映射 `(raw+1)/2` 不是凑出来的：`bn.move_rank` 的定义 `(#less-#greater)/(count-1)` 恒等于 `2r/(count-1)-1`（r 为 0-based 平均秩），代数相等，已用 scipy.stats.rankdata 在 4 个窗口 × 3 条路径上逐位校验 |
| ~~B-2~~ | **已修（2026-09-20）**：分子改成 `np.sum(dx*dy)/(window-1)`，与分母的 ddof=1 配套。偏差**随窗口变化**（完全线性相关的两条序列 w=5 只给 0.8、w=20 给 0.95），所以相关性阈值在短窗口上更难触发，短窗口的配对信号被系统性压制。`ts_cov` 一直是对的，两者本该同口径。修复后 `ts_corr` 并入三路径逐位对账 |
| ~~B-3~~ | **已修（2026-09-20）**：`cs_rank` 改为按并列区间取中点的平均秩（全向量化，无 Python 循环）。这是 **C-1 的截面版本** —— 同一个 `argsort(argsort(x))` 错误，时序一份、截面一份。并列值按**列顺序**强行分先后，而列序是面板的存储顺序、不是市场事实：换个标的顺序，同一只票的因子值就变了。新增 test_cs_rank_does_not_depend_on_column_order 把这条直接钉住（打乱列顺序，结果必须只是同样被打乱）|
| ~~B-4~~ | **已修（2026-09-21，用户裁定契约）**：`n_bins < 2` 当场抛 ValueError。依据是 **n_bins 走不到 GP/DSL** —— `FAST_TS_OPS` 按 `fn(x, window)` 派发、n_bins 恒为默认 10，fast_ops.py 之外全库零引用，所以 `n_bins=1` 只可能是开发者写错参数，该当场报错而不是返回一个看不出错的数。**顺带修掉登记表列的第二点（负零）**：`h = -np.sum(probs*log(probs))` 在常数窗口（某只票当天没动）上得到 **-0.0**，与 n_bins 无关、合法参数照样触发；负零会传播（`1/-0.0 = -inf`）、报告里显示成 "-0.0" 像故障。这一点是补的反向对照用例 test_a_constant_window_has_zero_entropy 抓出来的 |
| ~~B-5~~ | **已修（2026-09-20）**：numpy 两条分支从 `np.nanmax/np.nanmin` 改成 `np.max/np.min`（NaN 自然传播），与 bottleneck 的 `min_count=window` 一致。旧行为不止是两边对不上：numpy 那边的极值是**用更少的观测**算出来的 —— 停牌期间 ts_max 系统性偏低、ts_min 偏高，而调用方无从知道这一行的样本数不足。`TestPathsAgree` 新增**含 NaN**的对账用例：无 NaN 的输入上 nanmax 与 max 恒等，旧用例根本看不见这个缺陷 |
| ~~B-6~~ | **已修（2026-09-20）**：NaN 不再被填成 `-inf` 参与排序（`np.argsort` 本就把 NaN 排到末尾，不占有效名次）。旧实现让 NaN 占掉低位名次、分母却只按有效个数算，于是有效资产的名次从 n_nan/(n_valid-1) 起跳、最高越过 1。危害是 `rank(x) > 0.9` 命中多少取决于**当天停牌几只票**而不是信号。新增按缺失个数参数化的值域不变性用例（缺 0/1/2/3 只票，有效资产都必须铺满 [0,1]）|
| ~~B-7~~ | **已修（2026-09-20）**：`_too_short` 守卫提到 `if _HAS_BN` 分支**之前**，7 个 bn_* 包装器共用一份。已逐个实测确认 7 个全部受影响（不止 move_mean）。面板一短，整条 DSL 表达式求值**崩掉**而不是给 NaN —— walk-forward 第一折、次新股子集、小 universe 切片都会踩到 |
| ~~B-8~~ | **已修（2026-09-20）**：`should_prune` 不再只看样本数就走模型分支 —— `_fit()` 放弃时 `_model` 是 None，原来会 `None.predict_proba` 打断整轮 GP。**样本够 ≠ 模型就绪**。连带效果：`_fitted` 从死存储变成真守卫，原「`_fitted` 全库无人读 → 等价」那条证明**失效并撤销**（它自带的失效告警 test_fitted_flag_has_no_reader 如期响了），对应 2 个存活点已记入 survivor_disposition |
| B-9 | `proxy_model` 的 `use_label_encoder=False` 对 xgboost 3.x 已无意义（仅整洁问题） |
| B-10 | `fast_ops` 向量化分支被 `except Exception` 完全兜住（结构问题） |
| B-11 | `data_partitioner` 的「OOS 为空」守卫不可达（结构问题） |
| ~~B-12~~ | **已修（2026-09-20）**：三处排序都补了稳定的第二键。注意 `ChatSession.id` 是 **UUID 字符串**，按它排是随机序而非插入序 —— 另加了单调递增的 `seq` 列（同事务内取 max+1；`autoincrement` 只对整型主键生效）。`ChatMessage.id` 本就是自增整数，直接用 |
| ~~A-1~~ | **已修（2026-09-20）**：`ingest()` 新增 `append_pit` 参数，增量路径传 `False`。原先 `ingest()` 内部先把**整个抓取窗口**写进 PIT，`ingest_incremental` 随后又按 `> last` 过滤后写第二次 —— 同一批 bar 两个 vintage，而且两次写的窗口还不同（内层可能含一根重叠旧 bar）。集成用例的基线已从 3 个 vintage 改到 2 个、增量日从 2 改到 1 |
| ~~A-2~~ | **已修（2026-09-20）**：零方差判据统一到 `performance_analyzer.has_meaningful_variation`（相对量级：`sd > 1e-12 × mean|r|`）。原判据 `float(np.nanstd(...)) == 0.0` 用**精确相等**判浮点零 —— 严格全零序列确实返回 0.0、守卫会触发，漏掉的是**只在浮点噪声级别变动**的序列（回测净收益正是这种：多空两腿相减、成本逐日重算，残渣必然非零）。**这与 A-3 本是同一个问题的两处独立实现**（`sharpe_tstat` 里一度还有第三份 `sd <= 1e-15`）；判据分头维护迟早对同一条收益序列给出相反结论，所以提成模块级共享函数，两边薄封装。另修一处：该函数返回 `np.bool_` 而非标注承诺的 `bool`（调用方用 np.nextafter 取边界时暴露）|
| ~~A-3~~ | **已修（2026-09-20）**：`vol > 0` 判据换成**相对**量级（`sd > 1e-12 × 收益自身量级`）。全常数收益的 `std(ddof=1)` 是 6.5e-19 的浮点残渣，旧判据成立 → 年化 Sharpe 3e16、t 15.5，报告显示「高度显著」。相对判据保证真实低波动（日波动 1e-6）不被误杀，且 `sharpe_tstat` 共用同一判据不分叉 |
| ~~A-4~~ | **已修（2026-09-20）**：`_tdays` 提前判 `DatetimeIndex` 并抛带说明的 `TypeError`，不再让 `(idx[-1]-idx[0]).days` 抛指向内部实现的 `AttributeError`。`max_drawdown` 里那条按序号相减的 else 分支**可达性未变**（仍不可达），对应的等价性证明措辞已同步 |
| ~~A-5~~ | **已修（2026-09-21，用户裁定「数据资格先于优化」）**：**两个选项都不对** —— 我当初问的是「保留基准权重 vs 置 0」，而真正的问题是**那一行根本不该是一个数**。返回值现在有三种含义：有限数值=有效目标、全零=**主动清仓**、全 NaN=**本日无有效目标**。把 NaN 当 0 会在数据缺失的日子直接把仓位卖光。原实现有**五条**静默回退路径（returns=None / 预热期 / valid<3 / 求解失败 / 优化结果退化），每条都产出一行看起来完全正常的权重；其中「优化结果退化」那条**连 warning 都没有**，只按登记文字改根本碰不到它。旧实现还自相矛盾：部分资产数据不合格→那些资产清零，不合格到 valid<3→**全部保留**基准权重，缺得越多持仓反而越满。`returns=None` 的 SignalWeighted 回退改为**显式开关**（默认报错，打开后每次构造记 WARNING）。下游 `RealisticBacktester._resolve_no_target_rows` 显式解析 NaN：有过有效目标→沿用（持仓不动，减仓交执行层按实际成交处理）、从未建仓→0（陈述事实而非清仓动作），并在 `_build_weights` 末尾断言无 NaN 漏出。旧注释不作为正确性依据 |
| ~~A-6~~ | **已全部修复**（执行层 2026-09-18/19/20，**引擎统一 2026-09-21**）。执行层曾同时犯两个错：① `target=1.0` 写死，目标总敞口 0.27–0.30 被放大到 0.90–1.00（3.33×，成交名义额 1.54×）；② 用 water-filling 裁剪**目标持仓**冒充成交，A 被削掉后亏空摊给 B（[0.9,-0.1] → [0.01,-0.99]，10% 的对冲腿变成 99% 的方向性空头）。**后半（回测引擎）**：`BacktestEngine` 一直走 `LiquidityConstraint` 的整矩阵 water-filling —— 口径也错（裁的是**持仓**，而流动性约束的是**当天成交量**：昨天已持有、今天目标不变的名字根本不需要交易，却照样被削）。现改为逐日 `simulate_partial_fills`，与 PaperBroker 同一份原语、同一个 cap 公式，**路径依赖**（今天成不了的量留到以后接着成交，不消失也不摊给别人）。**前后对比见 `docs/A6_ENGINE_UNIFICATION.md`**：上限不绑定时**逐位不变**（两档场景），绑定时 gross 均值 0.114→0.552、换手 3.71→12.88、总收益 −0.05%→−2.88%。方向是"变差"很重要 —— 旧实现在流动性差的标的上**同时低估敞口和成本**，让本该"慢慢建仓、付不少成本"的策略看起来"几乎没持仓、也几乎没亏"，是**对自己有利的偏差**。原 xfail 用例 test_the_two_engines_agree_when_liquidity_binds 已转正（限流绑定数据上 1e-9 对账）|
| ~~A-7~~ | **已修（2026-09-21）**：三处都改传逐行真实 gross `np.abs(w).sum(axis=1)`。语义核准：`row_target = min(target, budget)`，而 budget 在 cap **有限**时是 Σcap（ADV / 单票上限通常远大于 1），于是 row_target 恒为 1.0 —— 上游 gross≠1 会被整体**放大**回 L1=1，把每个降敞口的决定（波动率目标、风控缩减、部分空仓）抹掉。（cap 为 inf 时 budget 取 `a` 本身，反而不会放大 —— 所以缺陷只在有限 cap 上成立，而三处用的全是有限 cap。）**今天多数情况下是 no-op**：上游全是 SignalWeighted、gross 本就≈1，713 条测试里只有 A-7 自己那条 xfail 变了。修的不是「现在算错」，而是「硬编码了对上游的假设且不校验」。`apply_capacity` 尤其自相矛盾：docstring 写「容量不足时 gross<1」却传 1.0；改完之后两条路径同时对了。补两条反向对照（容量不足仍要降 gross、上限不绑定时权重一点都不该变）。**顺带修掉一条空真用例**：`test_max_single_weight_enforced_end_to_end` 只断言 `max|w| <= cap`，账本被清空时 `0 <= 0.15` 照样成立 —— 变异复核把它抓了出来，已补「加上限后总敞口不得塌陷」的第三段 |
| ~~C-1~~ | **已修（2026-09-20）**：截面秩改用**平均秩**（并列取均值）。`argsort(argsort(x))` 是序数名次，把截面恒定的零信息信号排成 [0,1,2,...]，算出的 IC 是「ticker 在面板里的位置 vs 未来收益」的伪相关，列序一换 fitness 就变。**同一个错误复制了四份**（gp_engine / alpha_combiner / evaluation_utils / alpha_workflows）；而 `daily_trading_loop` 里早有一份正确实现且写明了理由，却没被复用。现统一到 `fast_ops.average_ranks_1d`，那份改为薄封装 —— 同一概念只留一份实现 |
| ~~C-2~~ | **已修（2026-09-20）**：`_replace_node` 改为**先在原树里定位路径、再沿路径在副本上替换**。原实现先 deepcopy 再按 `id(target)` 找，副本里没有任何节点持有那个 id，于是除根节点外替换**永远静默失败** —— hoist / wrap_rank / add_ts_smoothing / replace_subtree / subtree_crossover 五个算子只能在根上动手，GP 看着在变异实际原地踏步 |
| ~~D-1~~ | **已修（2026-09-20）**：取负识别从只认 `op == "neg"` 扩展到「`neg(x)` 或 `sub(0, x)`」。一元负号 `-x` 解析成 `neg(x)`，语义相同的 `(0-x)` 解析成 `sub(ScalarNode(0), x)` —— 同一因子换个等价写法就换家族。新增反向对照：`(1-x)` 不是取负，仍判 momentum（防判据放宽过头）|
| ~~D-2~~ | **已修（2026-09-20）**：`returns` 是**派生**字段、从不落盘，已从 `available_fields()` 去掉；请求不可用字段改为**当场报 ValueError**，不再读到一半被 except 吞掉、最后交回空 dict。新增机械对账：宣称的每个字段都必须在 `STANDARD_COLUMNS` 里（判据取写入路径的事实来源，不另抄清单）|
| ~~D-3~~ | **已修（2026-09-20，单独提交）**：迁移到 langchain 1.x 的 `create_agent` API（用户决定）。原状比登记文字更糟 —— `requirements.txt` 无上界，而 **`requirements.lock` 锁的是 1.4.0**，也就是说按声明的依赖装出来的环境，LLM 链路**必然**建不起来，然后被 `except Exception` 吞成一条 warning。CI 全绿，`/api/chat` 照常返回，研究链路整条死掉。迁移中**两处语义不自动等价**，已逐个实测补回：① 工具异常在旧 `AgentExecutor(handle_parsing_errors=True)` 下转成观测喂回模型，`create_agent` 默认**直接冒出 invoke** → 用 `wrap_tool_call` 中间件补回；② `max_iterations=15` 到顶停下返回，`recursion_limit` 到顶**抛 GraphRecursionError** → 真正的对应物是 `ModelCallLimitMiddleware(run_limit=15, exit_behavior="end")`，`recursion_limit` 只作护栏（实测装中间件后需 `4L+4`=64，**不是**没装时的 `2L+2`；第一版照搬了后者，被自己的用例当场抓住）。**未引入 langgraph checkpointer** —— ChatStore 已是会话历史唯一真相，再挂一个就是同一概念两份实现。降级原因做成可检测状态 `AGENT_MODE_*`（四态，按**异常类型** `LangChainIncompatibleError` 而非文案匹配分类），REST 与流式 done **两条都带**。`requirements.txt` 改 `>=1.0,<2.0` 并机械核对 lock 满足。新增 46 条用例（25 迁移验收 + 21 接线），全部用**实际安装的** LangChain，零 mock |
| ~~D-4~~ | **已修（2026-09-20）**：4 条家族模板从 `rank(neg(X))` 改为 `rank(-X)`，并在 AVAILABLE OPERATORS 下方写明取负只能用一元负号。实测提示词里 6 条 `DSL pattern:` 现在**全部可解析**，`neg(` 归零。**修法不是把 `neg` 加进算子清单** —— 解析器根本没有这个函数，加进去只会让提示词与解析器一起错；`-x` 才是解析器接受、且系统自己序列化时也输出的形式（`str(parse("rank(-ts_delta(close,5))"))` 往返稳定，已实测）。原后果：LLM 照模板产出的公式一律解析失败 → `_validate_and_fix` 白烧两次修复调用后放弃 → 六个家族里四个走模板路径时产出为零，对外只表现为「agent 老是生成非法公式」。两侧用例都改成**全称断言**（每条模板都要解析得了、每个用到的函数都要被声明），不再只防 `neg` 一个；另加反向护栏禁止写回 `neg(` |
| ~~D-5~~ | **已修（2026-09-21，用户裁定改文档）**：代码 `abs(corr) >= threshold` 不动，改提示词两处（`_prompts.py` / `_lc_agent.py` 工具 docstring）写明**绝对值**与 `>=`。负相关的 alpha 携带的是同一份信息、只是符号相反，同样该被拒 —— 代码行为在金融上是对的，文档该跟着实现走。（阈值那一半的指控此前已被外部审计推翻：生产链路 `PopulationEvolver` 显式传 0.90，类默认 0.70 不是链路行为。）|
| ~~D-6~~ | **已修（2026-09-20）**：`_fit()` 的 `try` 现在同时包住 import 与`XGBClassifier(...)` 构造。缺 scikit-learn 时是**构造**抛 ImportError，原来越过守卫直接冒泡 → GP 进化崩溃，而非 warning 承诺的退回 rule-based |
| ~~N-1~~ | **已修（2026-09-20）**：成本推导的缓存键原来只指纹 `close`，而价差算的是 high/low、冲击用的是 volume —— close 相同、high/low 不同的数据集命中同一条缓存（审计实测真实 1919.83 bps 被 37.47 bps 顶替，差 51 倍）。现在覆盖 `_COST_INPUT_FIELDS`（close/high/low/volume）全部面板 + 券商档位 + 账户类型。守卫两条：AST 从 `trading_context` 抽出**实际读取**的字段与清单对账（不抄一份同源清单）；行为上验证改 high/low 后不再命中旧条目，且同数据集仍然命中（缓存没退化成永不命中）|
| ~~N-2~~ | **已修（2026-09-20）**：全局试验台账读不到时不再退回 `n_trials=1` 然后照常给结论 —— 那会让 DSR 少做多重检验校正、**门变松**（审计实测 passed=true / DSR≈0.99997）。改为 fail-closed：拒绝给出结论并写明「验证不完整」。旧测试断言的正是 `n_trials == 1`，**它保护的就是那个错误行为**，已改为断言结论 |
| ~~N-3~~ | **已修（2026-09-20）**：`strategy_net_returns` 在 `apply_risk=True` 下风控/调仓对齐失败时，不再打一条 warning 就用未经风控的原始权重继续回测 —— 改为抛错。三个调用方均已 fail-closed（门判不通过并写明理由 / OOS 视为 -inf）。顺带补上了此处缺失的 `max_net` |
| ~~N-4~~ | **已修（2026-09-20）**：`sharpe_tstat` 改用**日频** Sharpe 配日频观测数，SR 与 T 同频。实测 8.3236 → 0.5660，与独立的单样本 t（0.5664）相差 3.8e-4；按 1.96 的判定从「✓显著」翻成「✗不显著」。修的是**频率口径**不是显著性阈值。钉住旧值的常量已作废，新断言用 `scipy.stats.ttest_1samp` 作**独立同频基准** |
| ~~N-5~~ | **已修（2026-09-20）**：`PaperBroker.step` 改用**昨仓与新目标的并集**，不再用 `target_w.index` 截断旧持仓。实测 A/B 各半仓、次日目标只留 B 且 A 涨 10% → gross_ret 从 0 变回 **+5%**，且 A 产生真实平仓成交。另处理「旧持仓今日无行情」：不估值也不交易（`reindex` 给 NaN 而 `nansum` 当 0 = 悄悄丢掉），并留 WARNING |
| N-6 | （前端，本轮不做）`useQuantWorkspace.switchSession` 在 await 后无条件 `setMessages`，迟到响应覆盖当前会话内容 |

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
python tools/mutation/make_plan.py                    # 生成计划：89 模块 / 2843 点
python tools/mutation/runner.py plan_full.json --state progress_full.json
python tools/mutation/runner.py plan_full.json --state progress_full.json --status
```

`progress_full.json` 是可续跑的中间态，已在 `.gitignore` 里。
结论请更新 `tests/meta/measured_modules.json` —— 那份才是交付物。

早期每轮测量各写一个 `plan_*.json` + `progress_*.json`，攒下约 100 个文件。
它们对复核没有价值（复核者要的是"照同一套规则重跑"，不是当时分了几批），
已全部删除，改用 `make_plan.py` 重新生成。
