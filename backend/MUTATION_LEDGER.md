# 变异测试台账（断言强度的唯一可信记录）

> **为什么需要这张表**：我在会话中多次口头报出击杀率，其中多个数字后来被证明是
> 工具缺陷造成的**虚高**（语法错被误记为"杀死"）。口头数字不可信，一律以本表为准。
>
> **测量前提（缺一则数字作废）**：
> 1. 变异必须在 **隔离副本** 中进行（`mutate.py` 复制 backend/ 到临时目录），主工作区零改动
> 2. 变异后必须仍能 `ast.parse`，否则不计入分子分母
> 3. 替换行必须保留行尾换行
> 4. 隔离副本本身是并发安全的；**不安全的只有共享 state 文件**（见前提 7）
> 5. **变异器必须真的生效** —— 见下方"工具缺陷史"，两次虚高都源于此
> 6. **每行每个变异点各算一个**，不是每行只取第一个
> 7. **并发运行必须各用各的 `--state` 文件**（缺陷史 #7）
>
> **判读补充**：测试超时（默认 1800s）计为**被杀死**。理由是"跑不完"同样是
> 一种可观测的回归；但这类杀死比断言失败慢两个数量级，看到某个模块卡住很久，
> 多半就是撞上了这种变异（strategy_gate 的一处 PBO 相关变异实测卡满 30 分钟）。
>
> 工具：`backend/tools/mutation/`（在仓库里，不在临时目录 —— 关机重启可续跑，
> 交付复核时对方能原样复跑；见该目录的 README）

## 工具缺陷史（每一条都曾让整批数字作废）

| # | 缺陷 | 后果 | 发现方式 |
|---|------|------|----------|
| 1 | 替换行丢了行尾 `\n`，与下一行粘连 | 语法错 → 误记"杀死" | 手查存活列表 |
| 2 | `-> float` 注解被 `>` 变异器改成 `->=` | 同上 | 同上 |
| 3 | 原地改源码 + 并行运行 | **变异被提交进仓库**（24c251a） | `git diff` |
| 4 | 只认 `tokenize.STRING`，漏 Python 3.12 的 `FSTRING_*` | f-string 里的 `>` 被当代码 → 假存活 | 逐条核对存活项 |
| 5 | `re.sub(pat, rep, m.group(0))` —— **正向后顾断言在孤立片段上必然失配** | `*` `+` `-` 三个算术变异器**从未生效过**，`transaction_cost` 129 个候选行只有 22 个被变异 | 存活列表清一色是比较符，一个乘除都没有 |
| 6 | 每行只做第一个匹配的变异器 | 比较符排在算术符前，`abs(dw) * val / p if p > 0` 这类行永远只测 `>` | 与 #5 一并发现 |
| 7 | 两个并发批次共用同一个 `--state` 文件 | 各自持一份内存快照整份覆盖写，**先完成那批的 12 个模块结果被后一批清空** | 进度表突然从 11/24 掉回 4/24 |

**#5 + #6 修复后，变异面翻倍**：`transaction_cost` 22 → 42 个变异点（其中 19 个算术），
`risk_gate` 22 → 42（其中 13 个算术）。**先前所有击杀率——包括 risk_gate 的 54.5%——
都是在不到一半的变异面上算出来的，一律作废重测。**

## 状态图例

- ✅ **已用隔离版工具测量** —— 数字可信
- ⚠️ **作废** —— 用有缺陷的工具测过，数字虚高，必须重测
- ⬜ **从未测量**

---

## 已测量（隔离版工具）—— 唯一可信的数字

**全部数字均为"每行每个变异器各算一个变异点"口径（旧口径分母只有一半，不可比）。**

| 模块 | 击杀率 | 变异点 | 存活 | 存活项处置 | 作废过的数字 |
|------|--------|--------|------|------------|--------------|
| `lifecycle/promotion_gate.py` | **92.3%** | 13 | 1 | ✅ 7 处写用例杀死，1 处证明等价 | ~~92.9~~ ~~27.3~~ ~~38.5~~ |
| `backtest_engine/transaction_cost.py` | **85.4%** | 41 | 6 | ✅ 10 处写用例杀死，6 处证明等价 | ~~100~~ ~~41.2~~ ~~57.1~~ |
| `lifecycle/leak_filter.py` | **77.8%** | 9 | 2 | ✅ 3 处写用例杀死，2 处证明等价 | ~~100~~ ~~37.5~~ ~~44.4~~ |
| `portfolio_manager/risk_gate.py` | **76.2%** | 42 | 10 | ✅ 10 处全部证明为等价变异 | ~~96.7~~ ~~16.1~~ ~~54.5~~ |
| `execution/paper_broker.py` | **80.8%** | 26 | 5 | ✅ 4 处写用例杀死，5 处证明等价 | ~~73.7~~ ~~65.4~~ |
| `tasks/daily_trading_loop.py` | **84.1%** | 63 | 10 | ✅ 12 处写用例杀死，10 处证明等价 | ~~18.6~~ ~~66.0~~ ~~65.1~~ |

**这 6 个模块已达标**（存活项 100% 已处置）。连同下方 A 档的 24 个，
共 **30 个模块**完成收口。每一条新增用例都对应一个"改坏了原本没人发现"的
具体位置，不是补覆盖率。

### 单点验证工具（`tools/mutation/verify_mutant.py`）

补一条用例之后要确认它**确实**杀得死目标变异，但重跑整模块太贵
（daily_trading_loop 一轮 **4890 秒**）。该工具在隔离副本里只施加**一个**指定变异，
只跑相关测试，给出"基线绿 / 变异红 = 已杀死"的结论，秒级到分钟级。

用它确认了 `daily_trading_loop L520`（`_run_one_alpha` 的第 0 天昨收）已被杀死 ——
该用例是在那一轮全量重测**开跑之后**才补的，所以那轮结果里它仍显示存活，
最终击杀率因此是 **53/63 = 84.1%**，而非输出里的 82.5%。

> 规矩：凡是用单点验证补上的结论，必须像这样在台账里写清"哪一轮的数字被这样修正过"，
> 否则台账里会出现一个跟任何一次运行输出都对不上的数字。

### paper_broker 收口详情（65.4% → 80.8%）

存活的 9 处里有 4 处是真盲区：

- `if self.initial_capital > 0` 改 `>=`：资金为 0 且 ADV 为 0 时 `0/0 = nan`
  → 投影全 NaN → **持仓整片消失**。只测"资金为 0"不够，必须**被除数也取到 0**
  （这与 transaction_cost L187 是同一个坑，我在那里判错过一次）。
- `abs(delta) < 1e-12 **and** abs(filled) < 1e-12` 改 `or`：
  "持仓没变但仍持有"的名字会被整条丢出 fills，审计看不到自己还拿着什么。
- `abs(filled) < abs(tgt) **- 1e-9**` 改 `+`：**足额成交**的订单被误标成
  "被 ADV 上限拒绝"，审计结论完全反了。
- `cost_bps = (cost + borrow) * 1e4` 改 `/`：只改量纲（差 1e8 倍），
  符号与相对大小都不变，"成本为正""净收益低于毛收益"这类断言全都抓不住。
  用恒等式 `cost_bps == (gross_ret - net_ret) * 1e4` 钉死。

剩余 5 处：3 个 epsilon 守卫 + `_as_date` 里一行**对所有实际输入形态都不可达**的
防御代码（str/datetime/Timestamp 在更早分支返回，date 没有 `.date` 属性）。

**每一个口头报过的数字都被推翻了，无一例外。**

### risk_gate：变异面翻倍后反而更高（76.2%）

分母从 22 涨到 42，新增的 20 个全是算术变异（`*→/`、`+→-`、`-→+`）——
**这 20 个全部被杀死**，所以击杀率从 54.5% 升到 76.2%，存活项一个没变，
仍是那 10 个 epsilon 守卫的比较符，仍是同一份等价性证明。

这说明两件事：
1. 之前为 risk_gate 写的 30 条用例是**真的在钉公式**，不是只钉边界；
2. 但 54.5% 那个数当时**没有资格被宣布"已达标"**——它的分母漏掉了整整一半，
   我在不知道乘除有没有被测的情况下说了"这个模块收口了"。

### 判定标准（比击杀率本身更重要）

单看击杀率会误导：**54.5% 的 risk_gate 已达标，而 100% 的口头数字是假的**。
一个模块算"达标"必须满足：

> **每一个存活变异，要么被新测试杀死，要么有书面且可机械验证的等价性证明。**

risk_gate 的 10 处存活全部属于后者，且证明本身写成了可执行断言
（`test_risk_gate_formulas.py::test_epsilon_guarded_boundaries_are_provably_unreachable`
逐个验证浮点上 `(limit + tol) - limit != tol`，即区分 `>` 与 `>=` 的那个唯一取值不可构造）。
另有 `test_every_survivor_has_a_written_proof` 强制每条证明不得敷衍——它抓到过我自己写的
"同 L117"（7 个字），已补全。

---

## risk_gate 收口详情（第一个达标模块）

- 击杀率 **16.1% → 54.5%**，用例 12 → **30**
- 过程中修掉测量工具的 f-string 缺陷：`f"...{x} > {y}"` 里的 `>` 是消息文本不是代码。
  Python 3.12 把 f-string 拆成 `FSTRING_START/MIDDLE/END`，旧版只认 `tokenize.STRING`
  → 4 处误报存活。
- **`L181 g_now = float(np.abs(a).sum())` 三次没咬住**，值得记：
  1. 用 `long_only=True`（权重全正），加不加 `abs` 结果一样；
  2. 方向猜反——以为 target 变小输出变小，实测 `project_to_capped_l1` 在
     target < Σcap 时**反而放大**（gross 7.8）；
  3. 断言最终 gross **仍然无效**——紧随其后的 gross 上限把 7.8 又缩回 4.0，
     **下游兜底吸收了错误**（与 `vol_scalar` 被 `np.clip(0,3)` 盖住是同一个模式）。

  真正的差别在**逐名权重**：正确 `[2, -2, 0, 0]`，变异后 `[1.03, -1.03, -0.97, -0.97]`
  ——两个 0.1 的小仓被放大成接近满仓的空头，而**聚合量完全看不出来**。

---

## A 档（直接算钱/下单/账本）—— 24 模块 / 504 变异点

工具与进度都在仓库里，关机重启原地续跑：

```
backend/tools/mutation/
  mutate.py          # 隔离副本 + 逐点落盘（--state）
  runner.py          # 按 plan 跑整档；--status 看进度
  rerun.py           # 补完用例后重测指定模块
  verify_mutant.py   # 单点复核，不必重跑整模块
  plan_tier_a.json   # 模块 → 覆盖测试集
  progress*.json     # 逐变异点的判定结果
```

### 首测击杀率（补用例之前）

| 模块 | 首测 | 变异点 | 存活 |
|------|------|--------|------|
| `portfolio_manager/strategy_gate.py` | **0.0%** | 31 | 31 |
| `backtest_engine/risk_report.py` | **3.1%** | 32 | 31 |
| `db/position_store.py` | 24.0% | 25 | 19 |
| `db/alpha_store.py` | 26.1% | 23 | 17 |
| `backtest_engine/performance_analyzer.py` | 30.4% | 92 | 64 |
| `tasks/daily_ingest.py` | 31.8% | 22 | 15 |
| `backtest_engine/portfolio_constructor.py` | 32.4% | 37 | 25 |
| `backtest_engine/realistic_backtester.py` | 32.4% | 37 | 25 |
| `trading_context/spread.py` | 47.6% | 21 | 11 |
| `tasks/cost_calibration.py` | 50.0% | 16 | 8 |
| `db/chat_store.py` | 58.8% | 17 | 7 |
| `data_engine/pit_store.py` | 70.8% | 24 | 7 |
| `data_engine/market_calendar.py` | 77.3% | 22 | 5 |

**`strategy_gate` 是 0.0%** —— 31 个变异点全部存活，即"决定一个策略配不配拿真钱"
的那道门，改坏任何一处都没有测试会红。`risk_report` 3.1% 紧随其后，
而它是所有对外结论的载体（策略门读它的 Sharpe/DSR、晋级门读它的
`insufficient_sample`、前端与台账读它的 `to_dict()`）。

### 补用例之后（复测）

A 档 24 个模块各自补了一份**定钉测试**，命名统一为
`tests/test_<模块>_<主题>.py`（formulas / contracts / schema / …）。
每个文件开头写明"该模块首测多少、存活哪几类、为什么这些存活项危险"。

| 模块 | 首测 → 复测 | 剩余存活 | 处置 |
|------|-------------|----------|------|
| `backtest_engine/performance_analyzer.py` | 30.4% → **95.7%** | 4 | ✅ 全部证明等价 |
| `backtest_engine/risk_report.py` | 3.1% → **93.8%** | 2 | ✅ 全部证明等价 |
| `backtest_engine/realistic_backtester.py` | 32.4% → **89.2%** | 4 | ✅ 全部证明等价 |
| `backtest_engine/portfolio_constructor.py` | 32.4% → **86.5%** | 5 | ✅ 全部证明等价 |
| `backtest_engine/backtest_engine.py` | 20.0% → **100%** | 0 | ✅ |
| `tasks/cost_calibration.py` | 50.0% → **100%** | 0 | ✅ |
| `trading_context/providers.py` | 50.0% → **100%** | 0 | ✅ |
| `db/chat_store.py` | 58.8% → **100%** | 0 | ✅ |
| `db/position_store.py` | 24.0% → **96.0%** | 1 | ✅ 证明等价 |
| `db/alpha_store.py` | 26.1% → **95.7%** | 1 | ✅ 证明等价 |
| `db/strategy_store.py` | 38.5% → **92.3%** | 1 | ✅ 证明等价 |
| `data_engine/market_calendar.py` | 77.3% → **90.9%** | 2 | ✅ 全部证明等价 |
| `tasks/daily_ingest.py` | 31.8% → **100%** | 0 | ✅（最后 1 处由单点验证确认杀死）|
| `trading_context/context.py` | 45.5% → **90.9%** | 1 | ✅ 证明等价 |
| `db/trial_ledger.py` | 50.0% → **87.5%** | 1 | ✅ 证明等价 |
| `portfolio_manager/manager.py` | 35.7% → **85.7%** | 2 | ✅ 全部证明等价 |
| `db/diagnostics_store.py` | 28.6% → **85.7%** | 1 | ✅ 证明等价 |
| `trading_context/spread.py` | 47.6% → **81.0%** | 4 | ✅ 全部证明等价 |
| `data_engine/pit_store.py` | 70.8% → **79.2%** | 5 | ✅ 全部证明等价 |
| `db/run_manifest.py` | 14.3% → **85.7%** | 2 | ✅ 全部证明等价 |
| `lifecycle/validation_gate.py` | 25.0% → **75.0%** | 2 | ✅ 全部证明等价 |
| `portfolio_manager/strategy_gate.py` | 0.0% → **93.5%** | 2 | ✅ 全部证明等价 |
| `backtest_engine/overfit_stats.py` | 20.0% → **80.0%** | 2 | ✅ 全部证明等价 |
| `db/alpha_lifecycle.py` | **100%** | 0 | ✅ |

> 击杀率本身不是达标标准 —— `pit_store` 复测后**降到** 79.2% 却已达标
> （分母没变，是先前被误杀的几个布尔参数在补测后暴露为真正的等价变异）；
> 标准始终是"**每个存活项要么被杀死、要么有可机械验证的等价性证明**"。

### A 档查出的问题（登记，本阶段不修）

- **PIT 被双写**：`ingest_incremental` 的注释写着"只把**增量**写进 PIT（而非整段重写）"，
  但它调用的 `ingest()` **内部已经把取到的整段写过一遍**（`_append_pit(dataset_name, data, as_of)`），
  随后外层又追加一次增量。一次增量摄取因此产生两个 vintage；
  只因 `as_of` 精度是**秒**、两次写入通常落在同一秒而被幂等去重掩盖。
  实测注入递增时钟后，8 个交易日每天都出现 2 个 vintage。

  > 顺带修掉一处**测试自身的不确定性**：`test_pit_only_receives_the_increment`
  > 原先断言"每天最多 1 个 vintage"，这依赖两次写落在同一秒 —— 单文件跑绿、
  > 与其他文件并跑就红。现改为注入确定时钟后**逐日**钉住 vintage 数
  > （回填日 1、增量日 2）。`df.index > last` → `>=` 会让重叠那天也变成 2，
  > 仍被杀死（`verify_mutant.py app/tasks/daily_ingest.py 114` 复核通过）。
  > 修掉双写之后，这条断言要同步改成"增量日也是 1"。


- **`np.nanstd(rets) == 0.0` 判不出零方差**：`pd.Series(0.001)` 的 nanstd 是
  2.17e-19 而非 0，策略门的"方差为 0"守卫**根本不会触发**（浮点零用 `==` 判）。
- **`PerformanceAnalyzer` 对近零波动没有防护**：全常数收益序列的 std 是 1.06e-17，
  `vol > 0` 成立 → 年化 Sharpe ≈ 3e16、t ≈ 15.5，"高度显著"。
- **`PerformanceAnalyzer` 不接受非日期索引**：`max_drawdown` 里有一条按序号相减的
  else 分支，但 `__init__` 的 `_tdays` 先做 `(idx[-1]-idx[0]).days`，整数索引在那里
  就抛 AttributeError —— 该分支**不可达**，且报错信息指向内部实现而非"索引类型不对"。
- **`MVOPortfolio` 注释与实现不符**：注释写"剔除 NaN 过多的资产（保留其基准权重）"，
  实现是 `w_out[t] = row / l1` **整行替换**，被剔除的资产拿到的是 0 而不是基准权重。

---

## promotion_gate 收口详情（38.5% → 92.3%）

新增 `tests/test_promotion_gate_boundaries.py`（19 条）。存活的 8 处分三类：

**一、默认值从来没人测过。** `experiment_mode` 的类默认与读配置失败时的兜底
**都是 `True`（放行）**，把它们改成 `False` 全套测试照样绿——既有用例每次都
显式传 `experiment_mode=`，从没人问过"不传时是什么"。这个值决定
**策略门没过的因子能不能进 paper**，不是无关紧要的细节。

**二、判级/判天数的边界可以精确构造，不是等价变异。**
`sharpe > 0` 的区分值是 `0.0`，`n < min_forward_days` 的区分值是阈值本身。
后者改成 `<=` 会让**恰好攒够 60 天**的策略被多卡一天，理由还是"观测不足"。

**三、失败路径的 detail 没人看。** 观测不足这条早退路径里
`detail.update({"passed": False, ...})` 改成 `True` 也全绿——既有用例只断言返回值，
而调用方（谱系记录、前端）读的正是 detail。已加"两条路径上 detail['passed']
必须恒等于返回值"的契约用例。

另外补上了 `and` → `or` 的清洗测试：`[v for v in ic if v is not None and isfinite(v)]`
改成 `or` 后 NaN/inf 会混进统计，把均值和 t 污染成 NaN。

---

## leak_filter 收口详情（44.4% → 77.8%）

**`Executor(validate=False)` 里的 `False` 是有意设计，此前无用例保护。**
本门要靠"实测夏普高到不可信"抓泄漏，而不是靠静态规则先毙掉表达式。
改成 `True` 后，窗口 > 252 或嵌套过深的表达式会走进 `except` 被记成"执行失败"
——**理由完全指错方向**，正常的长窗口因子被误伤。已用 `rank(ts_mean(close,300))` 钉死。

**年化系数 `* np.sqrt(252)` 改成 `/` 没人发现**，因为符号和相对大小都不变，
只有绝对值缩小 252 倍——任何"夏普为正/为负"的断言都抓不住它。
已用复刻管线算出未取整的精确夏普来比数值。

同一份复刻还让 `abs(sharpe) > max_plausible_sharpe` 的**等号侧**可测：
把阈值设成恰好等于实测夏普，正确实现放行、变异实现拦截。这类边界
**不属于**"浮点上造不出区分值"的等价变异——阈值是调用方传的参数，可以精确构造。

---

### risk_gate 存活项里的高危公式（改坏无人发现）

```
sec_cap = lim.max_sector_weight * lim.max_gross    * → /   行业上限
cap     = lim.max_name_weight   * lim.max_gross    * → /   单票上限
g_now   = float(np.abs(a).sum())                   删 abs  gross 敞口
a       = np.minimum(a, cap)                       min→max 容量约束反向
a       = np.where(a < 0.0, 0.0, a)                < → <=  long-only 去负
```

---

## transaction_cost 收口详情（第二个达标模块）

**57.1% → 85.4%**，新增 `tests/test_transaction_cost_formulas.py`（23 条用例）。
41 个变异点杀死 35，存活 6 条全部书面证明为等价变异并写成可执行断言。

### 查出来的真问题：`slippage_model="linear"` 是一片测试真空

`grep -rn slippage_model app/` 全库只有两处——默认值和 `== "sqrt"` 判断，
**没有任何代码把它设成 linear，测试里也一次没出现过**。于是这三行：

```python
slippage = (
    0.5 * self.p.spread_bps        # ← 改成 0.5 / spread 全绿
    + 0.1 * daily_vol * 10_000     # ← 改成 0.1 / daily_vol 全绿
) * np.ones_like(trade_abs)
```

前两行怎么改都没人发现。这不是"等价变异"，是**分支级的零断言**。
已补 5 条用例钉死它的契约，其中一条专门锁住"那个 `0.1` 是写死常数、
**不是** `impact_coef`"——否则日后有人"顺手统一一下"就悄悄改了成本模型。

### 我判错的一条等价性（值得记）

`L187 portfolio_value > 0 → >=`：我一度判它等价，理由是 pv==0 且 max_usd>0 时
两个分支都得 `inf`。**判错了**——`max_usd == 0` 时变异分支是 `0/0 = nan`，
整片权重被污染成 NaN。

> 教训：判等价前要把**被除数也取遍边界**，只遍历除数是不够的。

### 另一类真问题：投影的"需要缩小"分支从没被测过

既有用例（test_phase6 + golden）只有两种输入：L1 恰好 == target、预算不足。
**没有一条是 L1 > target 且预算充足 → 必须按比例缩小**。
于是 `np.all(np.abs(deficit) < tol)` 里的 `abs` 删掉也全绿——删掉后
`deficit < 0`（L1 超标，需缩小）会被误判成"已收敛"直接 break，
投影不再缩放，L1 范数原样超标返回。已补用例。

## 从未测量（≥20 行的应用模块）

按"错了会不会花钱/改变结论"排序，**这是待办的真实规模**：

### 一档：直接影响交易决策或金额
- `core/backtest_engine/backtest_engine.py`
- `core/backtest_engine/portfolio_constructor.py`
- `core/backtest_engine/alpha_combiner.py`
- `core/backtest_engine/performance_analyzer.py`
- `core/backtest_engine/risk_report.py`
- `core/backtest_engine/realistic_backtester.py`
- `core/portfolio_manager/manager.py`
- `core/portfolio_manager/horizon.py`
- `core/trading_context/context.py`、`providers.py`
- `core/lifecycle/validation_gate.py`
- `db/position_store.py`、`db/alpha_store.py`、`db/strategy_store.py`
- `tasks/daily_ingest.py`

### 二档：影响 universe / 数据正确性
- `core/data_engine/dataset_filters.py`（307 行，覆盖率仅 25.4%）
- `core/data_engine/dataset_registry.py`
- `core/data_engine/pit_store.py`
- `core/data_engine/market_calendar.py`
- `core/alpha_engine/fast_ops.py`（475 行，覆盖率 53.3%）
- `core/alpha_engine/signal_processor.py`、`dsl_executor.py`、`typed_nodes.py`

### 三档：搜索与 agent 管线
- `core/gp_engine/*`（population_evolver / mutations / alpha_pool / gp_engine）
- `core/workflows/alpha_workflows.py`
- `agent/*`
- `api/router.py`、`api/chat_router.py`

---

## 行覆盖率现状（2026-09-08 全量）

整体 **75.0%**，未覆盖 **2993 行**。低于 70% 的模块 20 个，其中低于 40% 的 9 个：

```
13.8% ccxt_provider        16.7% _lc_agent            19.0% akshare_provider
20.6% local_parquet        21.7% main.py              24.4% _chat_history
25.4% dataset_filters      32.8% schema               35.3% multi_dataset
```

---

---

## daily_trading_loop 处置详情（65.1% → 84.1%）

**这是真正下单的那条路径，此前是全项目最差的模块**（首次可信测量 18.6%）。
最终 63 个变异点杀死 53，存活 10 —— 全部为已机械验证的等价变异：
7 个 `getattr(settings, X, 默认值)` 的默认值 + L411/L525 的 IC 守卫 + L565 的循环边界。

### 同一个 bug 的两份拷贝（§S 的又一例）

第 0 天取末行价当昨收的前视，在 `run_portfolio`（L408）和 `_run_one_alpha`（L520）
**各有一份**。我上一轮只修了前者，重测时 L520 照样存活。
按"模块"审计会以为修完了 —— 审计单位必须是**每一个变异点**，不是文件。


这是**真正下单的那条路径**，也是三份既有测试文件（41 条用例）覆盖的模块。
22 处存活里有 **6 条是"用例在场却杀不死"** —— 断言写了，但写得没法失败：

| 变异 | 既有用例 | 为什么杀不死 |
|------|----------|--------------|
| L408 第 0 天取末行价当昨收（前视） | `test_portfolio_first_day_has_no_lookahead` | 全新账本第 0 天 `prev_w` 全 0，`Σ(prev_w × price_chg)` 恒为 0，昨收取哪天都一样。**修法：先种一条早于数据起点的持仓**，正确实现仍为 0，变异实现毛收益 −11.4% |
| L361 熔断清仓 `* 0.0` → `/ 0.0` | `test_drawdown_halt_flattens_book_only_when_enabled` | 除零得到 inf 与 NaN：inf 被 ADV 上限截回有限值、NaN 被 `abs(w) > 1e-12` 过滤，落账持仓照样"看起来是空仓"。**修法：拦截送进 broker 的目标权重**，断言逐项恰好为 0 且有限 |
| L210/L215 PM.7 配置过滤 | `test_active_strategy_config_filters_components` | 断言写成 `active_config == sid **or** n_factors == 1` —— 过滤没发生时，PM.S2 边际准入也可能自己把成分收敛到 1 个，右半边照样成立。**修法：关掉边际准入隔离变量，同时断言配置 id 与成分集合** |
| L230 边际准入的启用条件 | 两条 `test_marginal_selection_*` | 两条**都是否定断言**（"该关的时候关掉了"），把整个分支永久关掉也全绿。**缺的是肯定断言** |
| L271 组合账本的 broker 选择 | `test_portfolio_broker_uses_grounded_capital` | 断言 `out["aum"] == 传入值`，而 `aum` 是局部变量直接写进返回值，**跟用哪个 broker 无关**。修法：把 grounded 成本参数换成 `fixed_bps=500`，看落账 cost_bps（0.8 → 180bps） |
| L441 策略级衰减告警 | `test_decay_alert_flag_follows_the_monitor` | 它测的是 `run()` 的**因子级** `decay_alert`，与 `run_portfolio` 的**策略级** `strategy_decay` 是两条独立分支——DEV_LESSONS §S「审计单位错了」的又一例 |

### 判错过的一条等价性（自己推翻自己）

上一轮我把 `Executor(validate=False) → True` 写进了 `EQUIVALENT_MUTANTS`，
理由是"打开验证只会让非法 DSL 更早报错，入库因子必然合法"。**这是错的**：
`WindowValidator`（窗口 > 252）和 `DepthValidator`（嵌套 > 10）会把
**能正常执行、也没有泄漏**的因子一并拒掉。在 `run_portfolio` 里的后果是
该因子被剔出组合 → 无因子可用 → **退回基准库，账本悄悄换了策略**。
已用 `rank(ts_mean(close, 300))` 写成用例（两条路径各一条）。

> 与 leak_filter 那条 `Executor(validate=False)` 是同一个设计决定、同一个坑，
> 一个模块判对了、另一个模块判错了 —— 说明"等价性证明"必须逐模块重新做，
> 不能跨模块套用结论。

### 三组仍然等价的存活（已机械验证）

- **7 处 `getattr(settings, X, 默认值)` 的默认值**：只在配置项缺失时可达。
  不用文字说明，`test_settings_fields_backing_getattr_defaults_all_exist`
  逐个字段核 `Settings.model_fields`，谁改了字段名它立刻红 ——
  这很重要，因为其中几个默认值是**放行方向**的（`block=False`、`halt=False`），
  字段一旦改名就会静默把门关掉。
- **L411/L525 `if t > 0`（IC 记录）**：`pct_change()` 第 0 行恒为 NaN，
  第 0 天即使进了分支也拿不到可记录的 IC。
- **L565 `while i < len(x)`**：多空转一轮，空切片赋值合法无副作用。
  用例把两个循环边界都跑一遍逐元素比对，不靠嘴说。

### 一条只能用"警告升级为错误"来杀的变异

`L587 float(np.dot(ra, rb) / denom) if denom > 0 else nan` 改成 `>=` 后，
denom == 0 走进 `0.0 / 0.0` —— **结果同样是 NaN**，所以既有的
`test_spearman_returns_nan_when_denominator_is_zero` 用返回值断言永远杀不死它。
差别只在变异版每算一次常数信号就抛一个 RuntimeWarning，而交易循环每天都调它。
`warnings.simplefilter("error", RuntimeWarning)` 才能区分。

---

## 测试查出的**产品问题**（本阶段只登记，不修）

按用户定的顺序：所有测试达标之后才开始修。登记在此以免遗忘。

### P1（严重）`PaperBroker.step` 把目标总敞口强行放大到 L1 = 1.0

```python
filled = project_to_capped_l1(tgt[np.newaxis, :], cap_w[np.newaxis, :], target=1.0)
```

`target=1.0` 写死，不看传进来的 `tgt` 实际总敞口。最小复现：

```
target_w = [0.25, -0.25]   (gross 0.5)
→ 落账持仓 = {'A': 0.5, 'B': -0.5}   (gross 1.0)
```

**上游任何降敞口的决定都在这一步被抹掉**——波动率目标缩放、`max_gross` 上限、
PM.6 无交易带，全部失效。而 `risk_gate` 那 42 个变异点的工作正是在保证
这些上限被正确执行；上限算对了，到了 broker 又被归一化掉。

同一行还有第二个后果：当某只票被 ADV 上限削掉时，water-filling 会把亏空
**摊到其余名字上以凑满 L1=1**。实测：

```
target = [0.9, -0.1]，A 的 ADV 只够 0.01
→ 落账 = {'A': 0.01, 'B': -0.99}
```

想要 10% 的空头变成了 99%。"买不到 A 就把钱全砸进 B"不是任何风控愿意接受的行为。

> 两条都由 `tests/test_paper_broker_accounting.py::test_adv_capped_order_is_flagged`
> 附近的探查发现，当前用例**如实钉住了现状**（不是钉住"应该的样子"），
> 修复时这些断言需要一并改，改的时候要能说清新旧口径差异。

---

## 完成标准（用户定义，不得自行降低）

本阶段只有同时满足以下四条才算完成：

1. **所有测试都足够严格** —— 关键模块有隔离版变异测量，存活项要么杀死、要么书面证明为等价变异
2. **新排查出的问题全部完整修好** —— 不是减量，不是分类，是修完
3. **项目所有模块都覆盖完整测试**
4. **可以交付给外部做严格检查** —— 任何一项"未知"都不算达标

在此之前不得宣布阶段结束，也不得转去修被发现的问题。

---

## B 档（数据正确性 / 因子计算与评估）—— 16 模块 / 450 变异点

**队列**：`tools/mutation/plan_tier_b.json`，进度文件 `tools/mutation/progress_b.json`
（与 A 档分开 —— 共用一个会互相整体覆盖，见工具缺陷 #7）。

### 为什么是这 16 个，以及被排除的 6 个

按"错了会不会改变结论"排完序之后，B 档取的是**数据正确性 + 因子计算/评估**
这一层。收录标准只有一条：**该模块有既有测试**。

| 模块 | 变异点 | 覆盖测试 |
|---|---:|---|
| `core/alpha_engine/fast_ops.py` | 145 | test_dsl_operators / test_dsl_edge_cases / test_dsl_engine |
| `core/data_engine/data_partitioner.py` | 54 | test_phase1_upgrade / test_phase2 |
| `core/alpha_engine/typed_nodes.py` | 33 | test_dsl_engine / test_dsl_operators / test_dsl_edge_cases |
| `core/alpha_engine/dsl_executor.py` | 25 | test_dsl_operators / test_dsl_edge_cases / test_dsl_engine / test_leak_filter |
| `core/ml_engine/alpha_optimizer.py` | 24 | unit/test_ml_optimizer |
| `core/ml_engine/alpha_evaluator.py` | 22 | test_phase2 |
| `core/discovery/market_observer.py` | 22 | test_phase9_market_observer |
| `core/alpha_engine/signal_processor.py` | 18 | test_phase1_upgrade / test_leak_filter |
| `config.py` | 17 | test_production_defaults / test_invariants |
| `core/data_engine/regime_detector.py` | 17 | test_phase4 / test_phase6 |
| `core/data_engine/dataset_registry.py` | 16 | test_phase7_ingest / test_phase8_pit / test_daily_ingest_increment |
| `core/data_engine/providers/moomoo_provider.py` | 13 | test_phase_tr2_moomoo |
| `core/alpha_engine/parser.py` | 13 | test_dsl_edge_cases / test_dsl_operators / test_dsl_engine |
| `core/monitor/alpha_monitor.py` | 12 | test_phase5 / test_phase_pm |
| `core/alpha_engine/validator.py` | 11 | test_alpha_discovery / test_dsl_engine / test_dsl_edge_cases |
| `core/ml_engine/proxy_model.py` | 8 | test_alpha_discovery |

**被排除的 6 个（零测试引用，属于下一阶段"给未覆盖模块补测试"，不是 B 档）**：
`backtest_engine/alpha_combiner.py`(25)、`data_engine/multi_dataset.py`(23)、
`data_engine/schema.py`(11)、`portfolio_manager/strategy_builder.py`(8)、
`portfolio_manager/horizon.py`(7)、`data_engine/base.py`(5) —— 共 79 点。
对零测试的模块做变异测量没有意义（击杀率必然是 0），先补测试再测量。

> 引用判定用的是**精确匹配**（`from app.x.y import` / `import app.x.y` /
> `"app.x.y..."` 形式的 patch 目标），不是模块名裸串匹配 —— 裸串会把
> `tests` 里任何提到 `horizon` 字样的文件都算成覆盖，得出 4 个假覆盖。

### B 档在测量之前就已确认的产品问题（读代码 + 直接复现，非变异结果）

这四条都在 `fast_ops.py`，即**全系统每一个因子值都要流过的算子内核**。

1. **`ts_rank` 的量纲与 docstring 不符，且两条分支互相矛盾（严重）**
   docstring 写 "Rolling rank (percentile in [0,1])"。`_HAS_BN=True`（生产环境
   实测已安装 bottleneck 1.6.0）时走 `bn.move_rank(...) / window`，而
   `bn.move_rank` 的值域是 **[-1, 1]**，除以 window 之后变成 **[-1/w, 1/w]**。
   numpy 回退分支算的是 `le/count` ∈ (0, 1]。实测单调上升序列 window=5：
   bottleneck 分支 **0.2**，numpy 分支 **1.0**。
   后果：① 任何 `ts_rank(x, w) > 0.8` 形式的表达式**永远为假**（上限只有 1/w）；
   ② 与其他算子做算术时量纲差一个 window 倍 ——
   `ts_rank(close,20) + ts_zscore(volume,10)` 里 rank 项（±0.05）被 zscore 项
   （±3）完全淹没，等价于只用了后者。
   当前种子表达式都把 ts_rank 包在 `rank(...)` 里（单调变换，暂不受影响），
   所以这是**潜伏**问题，但 GP 变异随时可以生成受影响的表达式。

2. **`ts_corr` 系统性偏低 (w-1)/w（严重）**
   `cov_xy = np.mean(dx*dy, axis=1)` 用 ddof=0，而 `std_x/std_y` 用 ddof=1，
   两者不配套。实测完全线性相关的两条序列、window=20，返回 **0.950000**
   而不是 1.0，偏差因子恰好 `(w-1)/w = 0.95`。
   后果：`ts_corr` 永远够不到 ±1；window 越小偏得越多（w=5 时只有 0.8）。
   任何拿 `ts_corr` 与阈值比较、或把它当相关系数解读的地方都被压缩了。

3. **`cs_rank` 的 docstring 声称 "ties resolved by average rank"，实现是序数名次**
   `argsort(argsort(x))` 给的是序数。实测 `[1, 1, 2, 3]` → `[0, 0.333, 0.667, 1]`，
   两个并列的 1.0 拿到了不同名次，且**谁在前谁排低**取决于列顺序 ——
   同一截面换个标的顺序会得到不同因子值。

4. **`ts_entropy(n_bins=1)` 返回 -0.0 而不是 NaN/报错**
   `log_nbins = np.log(n_bins) if n_bins > 1 else 1.0` 把退化情形静默地
   归一化成 0，调用方看不出参数给错了。

> 按用户定的顺序，本阶段**只登记不修**。

### B 档在补用例过程中新查出的产品问题（登记，本阶段不修）

编号接上面四条。**每一条都有一条测试把当前的错误行为原样钉住**，
断言消息里写明"修好之后这条要改成什么"，修复时不会漏改。

5. **`ts_max` / `ts_min` 的 NaN 策略两条分支不一致（严重）**
   模块 docstring 承诺 "All rolling operators enforce strict NaN policy:
   fewer than `window` valid observations → NaN output"。
   bottleneck 分支遵守（`min_count=window`），numpy 分支用 `np.nanmax` /
   `np.nanmin` **直接忽略 NaN**。实测 `[1,2,NaN,4,5,6]` window=3：
   bottleneck `[nan,nan,nan,nan,nan,6]`，numpy `[nan,nan,2,4,5,6]`。
   后果：没装 bottleneck 的环境里，缺了一根 bar 的标的照样吐出 ts_max 值 ——
   停牌/次新股的窗口被当成完整窗口用。
   钉在 `test_fast_ops_kernel.py::test_ts_max_min_nan_policy_differs_between_paths`。

6. **`cs_rank` 在含 NaN 的截面上值域越出 [0,1]（严重）**
   NaN 资产被 `-inf` 填充后**参与了 argsort**、占掉名次 0，而分母只按有效
   个数算（`valid_count - 1`）。实测 `[10, NaN, 30, 40]` → `[0.5, nan, 1.0, 1.5]`。
   后果：`rank(x) > 0.9` 这类阈值条件会因为当天缺了几只票而莫名多命中；
   缺失越多偏得越狠，而缺失率本身是随时间变化的 —— 因子含有一个
   **与数据完整度相关的伪信号**。
   钉在 `test_fast_ops_kernel.py::test_cs_rank_range_overflows_one_when_the_row_has_nan`。

7. **面板行数短于窗口时 bottleneck 分支抛异常而不是返回 NaN（严重）**
   `if T < window: return 全 NaN` 这个守卫**只存在于 numpy 分支**；
   bottleneck 分支直接把 window 传给 `bn.move_*`，抛
   `ValueError: Moving window (=5) must between 1 and 3, inclusive`。
   七个算子全部如此：ts_mean / ts_std / ts_var / ts_sum / ts_max / ts_min / ts_rank。
   后果：walk-forward 第一折、次新股子集、小 universe 切片都会让整条 DSL
   表达式求值崩掉，而 docstring 承诺的是"返回 NaN"。
   钉在 `test_fast_ops_kernel.py::test_bottleneck_rolling_ops_raise_when_the_panel_is_shorter_than_window`。

8. **`ProxyModel` 在 `_fit()` 放弃之后仍走模型分支 → AttributeError（严重）**
   `_fit()` 遇到单一类别（或 xgboost 未安装）会提前 return，`self._model`
   保持 None；但 `should_prune` 只判断 `len(self._X) >= cold_start_n`，
   随即调用 `self._model.predict_proba` → `'NoneType' object has no attribute`。
   触发条件很常见：冷启动期所有候选都失败（标签全是 1）。
   讽刺的是 `self._fitted` 这个字段正是为此存在的 —— 它从头到尾**没有任何读取方**，
   两次赋值都是死存储（变异测试因此把它们判为等价变异，证明见测试文件）。
   钉在 `test_proxy_model_pruning.py::test_unfitted_model_past_cold_start_crashes`。

9. **`use_label_encoder=False` 对当前依赖版本已无意义（轻）**
   xgboost 自 2.0 起移除该参数，本环境 3.2.0 对两种取值都只是忽略。
   留着它会让人以为有效果。机械验证见
   `test_proxy_model_pruning.py::test_label_encoder_flag_is_ignored_by_xgboost`。

10. **`fast_ops` 的向量化分支被 `except Exception` 完全兜住（中）**
    每个滚动算子都是「try: 向量化实现 / except Exception: 纯循环兜底」。
    兜底本身是对的（`as_strided` 在异常内存布局下确实可能失败），
    问题是 `except Exception` 太宽：**向量化实现写错到抛异常的程度时，
    外部一点差别都观察不到**，只是慢了。变异测试因此把十处（四个 shape 算术、
    六个 `keepdims=True`）判成等价变异 —— 它们不是测试写不到，是这段代码的
    结构决定了写不到。
    建议（本阶段不改）：收窄成 `except (ValueError, TypeError)` 并在兜底时
    `logger.warning` 一次，让"向量化路径失效"成为可观测事件。
    证明与失效告警见 `test_fast_ops_kernel.py::test_vectorised_paths_are_wrapped_in_a_rescuing_except`。

11. **`data_partitioner` 的"OOS 为空"守卫不可达（轻）**
    `if oos_ratio > 0 and (self._oos_start is None or oos_count < 1): raise`
    以及 `else: self._oos_start = None` 这两处，在当前算术下**永远不会触发**：
    前置守卫已保证 `usable >= 2`，而 `is_count` 被夹在 `[1, usable-1]`，
    于是 `oos_count >= 1` 且 `is_count + embargo <= total - 1 < total` 恒成立。
    对全部 (总天数 10..400) × (embargo 0..59) × 9 档 ratio 的组合穷举验证，
    零反例。
    后果不是当下出错，而是**给人以"我们检查过空 OOS"的假印象**：
    哪天上面的切分算术被改动，真正的空 OOS 会绕过它静默通过。
    证明与失效告警见
    `test_data_partitioner_split.py::test_the_oos_guards_are_unreachable_over_the_whole_parameter_space`。

### B 档首测击杀率（补用例之前）

整体 **19.1%**（450 个变异点，存活 364）—— 也就是说，在这一层里
**改坏 10 处只有 2 处会被现有测试发现**，而当时这些模块的测试全是绿的。

| 模块 | 变异点 | 存活 | 首测击杀率 |
|---|---:|---:|---:|
| `core/data_engine/dataset_registry.py` | 16 | 16 | **0.0%** |
| `core/ml_engine/alpha_optimizer.py` | 24 | 23 | **4.2%** |
| `core/alpha_engine/fast_ops.py` | 145 | 135 | **6.9%** |
| `core/data_engine/providers/moomoo_provider.py` | 13 | 12 | **7.7%** |
| `core/ml_engine/proxy_model.py` | 8 | 7 | **12.5%** |
| `core/alpha_engine/parser.py` | 13 | 11 | **15.4%** |
| `core/data_engine/data_partitioner.py` | 54 | 44 | **18.5%** |
| `core/discovery/market_observer.py` | 22 | 17 | **22.7%** |
| `core/ml_engine/alpha_evaluator.py` | 22 | 17 | **22.7%** |
| `core/alpha_engine/validator.py` | 11 | 8 | **27.3%** |
| `core/alpha_engine/typed_nodes.py` | 33 | 23 | **30.3%** |
| `config.py` | 17 | 11 | **35.3%** |
| `core/alpha_engine/dsl_executor.py` | 25 | 16 | **36.0%** |
| `core/alpha_engine/signal_processor.py` | 18 | 10 | **44.4%** |
| `core/monitor/alpha_monitor.py` | 12 | 6 | **50.0%** |
| `core/data_engine/regime_detector.py` | 17 | 8 | **52.9%** |

**0.0% 的那一个值得单说**：`dataset_registry` 有 7 个测试文件"引用"它，
16 个变异点却**一个都杀不死** —— 因为那 7 个文件全都是把它 `monkeypatch` 掉
去测别的东西。"有多少测试提到这个模块"和"这些测试能发现这个模块的问题"
是两件毫不相干的事，这是本档最直观的一个例证。

### B 档补用例之后（复测）

整体 **19.1% → 92.2%**（450 点，存活从 364 降到 **35**）。
剩下的 35 个存活项**全部**有书面且可机械验证的等价性证明（见下表最后一列指向的测试文件）。

| 模块 | 变异点 | 首测 | 复测 | 剩余存活 | 取数于 |
|---|---:|---:|---:|---:|---|
| `core/alpha_engine/fast_ops.py` | 145 | 6.9% | **87.6%** | 18 | `progress_b_rerun_7.json` |
| `core/data_engine/data_partitioner.py` | 54 | 18.5% | **83.3%** | 9 | `progress_b_rerun_8b.json` |
| `core/alpha_engine/typed_nodes.py` | 33 | 30.3% | **100.0%** | 0 | `progress_b_rerun_9.json` |
| `core/alpha_engine/dsl_executor.py` | 25 | 36.0% | **100.0%** | 0 | `progress_b_rerun_9.json` |
| `core/ml_engine/alpha_optimizer.py` | 24 | 4.2% | **95.8%** | 1 | `progress_b_rerun_7.json` |
| `core/discovery/market_observer.py` | 22 | 22.7% | **100.0%** | 0 | `progress_b_rerun_3.json` |
| `core/ml_engine/alpha_evaluator.py` | 22 | 22.7% | **100.0%** | 0 | `progress_b_rerun_7.json` |
| `core/alpha_engine/signal_processor.py` | 18 | 44.4% | **88.9%** | 2 | `progress_b_rerun_5.json` |
| `config.py` | 17 | 35.3% | **100.0%** | 0 | `progress_b_rerun_4.json` |
| `core/data_engine/regime_detector.py` | 17 | 52.9% | **94.1%** | 1 | `progress_b_rerun_5.json` |
| `core/data_engine/dataset_registry.py` | 16 | 0.0% | **100.0%** | 0 | `progress_b_rerun_9.json` |
| `core/alpha_engine/parser.py` | 13 | 15.4% | **92.3%** | 1 | `progress_b_rerun_dsl2.json` |
| `core/data_engine/providers/moomoo_provider.py` | 13 | 7.7% | **100.0%** | 0 | `progress_b_rerun_3.json` |
| `core/monitor/alpha_monitor.py` | 12 | 50.0% | **100.0%** | 0 | `progress_b_rerun_5.json` |
| `core/alpha_engine/validator.py` | 11 | 27.3% | **100.0%** | 0 | `progress_b_rerun_dsl2.json` |
| `core/ml_engine/proxy_model.py` | 8 | 12.5% | **62.5%** | 3 | `progress_b_rerun_2.json` |

> "取数于"这一列是**可复跑的凭据**：每个数字都来自那个进度文件里那一次运行，
> 而那次运行用的测试集写在对应的 `plan_tier_b_rerun_*.json` 里。
> 复测分了九批，因为每补一批用例就要拿"新用例 + 原有覆盖"重测一次 ——
> 只跑新文件会漏掉旧文件覆盖到的点，只跑旧文件就等于重做首测。

**`proxy_model` 的 62.5% 是本档最低，但它达标**：8 个点里 3 个存活，
三个都是不可观测的赋值（`_fitted` 这个字段**全代码库没有任何读取方**，
`use_label_encoder` 在 xgboost 3.x 里已被忽略）。击杀率本身从来不是标准 ——
"每个存活项要么被杀死、要么有可机械验证的等价性证明"才是。

### B 档新增的 15 个测试文件

共 **6400 余行 / 581 条用例**（`pytest --collect-only` 实数）。每个文件的头部写明了：该模块首测击杀率是多少、
存活了哪些、为什么那些存活项危险、既有测试为什么没抓到。

| 测试文件 | 覆盖模块 | 条数 |
|---|---|---:|
| `test_fast_ops_kernel.py` | fast_ops（三条执行路径交叉验证） | 151 |
| `test_data_partitioner_split.py` | data_partitioner + WalkForwardPartitioner + `_slice_dataset` | 61 |
| `test_typed_nodes_semantics.py` | typed_nodes | 39 |
| `test_alpha_optimizer_search.py` | alpha_optimizer | 36 |
| `test_dsl_parser_and_validator_bounds.py` | parser + validator | 37 |
| `test_signal_processor_pipeline.py` | signal_processor | 36 |
| `test_regime_detector_labels.py` | regime_detector | 35 |
| `test_alpha_evaluator_overfit.py` | alpha_evaluator | 30 |
| `test_dataset_registry_loading.py` | dataset_registry | 29 |
| `test_dsl_executor_pipeline.py` | dsl_executor | 27 |
| `test_market_observer_scoring.py` | market_observer | 25 |
| `test_proxy_model_pruning.py` | proxy_model | 22 |
| `test_alpha_monitor_decay.py` | alpha_monitor | 21 |
| `test_moomoo_paging_and_window.py` | moomoo_provider | 18 |
| `test_settings_boolean_defaults.py` | config（全部 16 个 bool 默认值的快照） | 14 |

### B 档过程中我自己踩的坑（与 A 档不重复的那几条）

1. **"上界钉住了"不等于"这一行钉住了"**
   `np.clip(x, mu - k * sd, mu + k * sd)` 一行里有**两个** `*`，变异器只改第一个，
   于是只有**下界**变了。我只断言了 `max()`，那个变异连续两轮存活。
   同一个错在 `dsl_executor._postprocess` 里犯了第二次。
   → 一行里出现多次同一运算符时，每一处都要有各自的断言。

2. **断言的字符串恰好也出现在别处**
   `ind_neutralize` 缺分组时的警告是"没有分组字段 %r（也没有 'groups'）"。
   我断言 `"'groups' in msg"，而消息后半句本来就有这个词 —— 变异版照样命中。
   → 断言"某个错误值**没有**出现"往往比断言"正确值出现了"更有区分力。

3. **等价性的直觉判断必须由机械验证兜底，而且它真的会推翻你**
   我给 `ts_corr` 的 `denom > 1e-12` 写了"边界值不可构造"的证明，
   机械验证当场证伪：`sqrt(1e-12)² == 1e-12` 在浮点上是精确的，
   `[-c, 0, c]` 的样本标准差恰好等于 `|c|`，边界一构就中。
   同样地，`typed_nodes` 的 L296、`fast_ops` 的 L37 我都判成了"不可达"，
   复测把它们都杀死了。
   → **写了证明不代表成立**；把证明写成可执行断言，让它有机会当场打脸。

4. **归一化会把绝对值抹掉，断言要挑不受归一化影响的量**
   `market_observer` 的六个族分最后除以 `max(raw)`。我一开始比较两个独立面板的
   归一化分数，结果 regime 跟着变了，动量分抬高了归一化基准，
   liquidity 的分反而更低 —— 断言错在构造，不在实现。
   → 改成"同一次观察内部两个族的比值"，或构造出只有一个变量在动的两个面板。

5. **一个模块里可能有两套独立的算术**
   `data_partitioner` 的 54 个点里，`DataPartitioner` 只占一半，
   另一半在 `WalkForwardPartitioner` 和共用的 `_slice_dataset` 里。
   先只覆盖前者时复测停在 37%。
   → 补用例之前先按**函数/类**把存活项分组，别按文件。

### 剩余待办（本节取代前面「从未测量（≥20 行的应用模块）」那份清单）

A+B 完成后，已测量模块 **46 个 / 1148 变异点**。按同样的口径重算，剩余：

**C 档候选 —— 有既有测试，可以直接测量：17 模块 / 476 点**

| 模块 | 点数 | | 模块 | 点数 |
|---|---:|---|---|---:|
| `api/router.py` | 75 | | `tasks/scheduler.py` | 20 |
| `core/workflows/alpha_workflows.py` | 69 | | `core/gp_engine/fitness.py` | 20 |
| `core/gp_engine/mutations.py` | 65 | | `agent/_critic.py` | 18 |
| `core/gp_engine/population_evolver.py` | 48 | | `agent/_agent.py` | 16 |
| `core/data_engine/health_report.py` | 32 | | `core/gp_engine/gp_engine.py` | 15 |
| `core/gp_engine/alpha_pool.py` | 22 | | `tasks/backup.py` | 14 |
| `agent/_tools.py` | 22 | | `core/discovery/discovery_engine.py` | 9 |
| `main.py` | 21 | | `core/data_engine/sector_mapper.py` | 5 |
| | | | `api/chat_router.py` | 5 |

**零测试引用 —— 必须先补测试再测量：19 模块 / 376 点**

`data_engine/dataset_filters.py`(54)、`alpha_engine/financial_interpreter.py`(50)、
`alpha_engine/financial_diagnostics.py`(35)、`agent/_prompts.py`(35)、
`backtest_engine/alpha_combiner.py`(25)、`data_engine/multi_dataset.py`(23)、
`agent/_data_utils.py`(22)、`data_engine/local_parquet_provider.py`(21)、
`backtest_engine/visualizer.py`(19)、`backtest_engine/multi_dataset_backtester.py`(14)、
`data_engine/providers/ccxt_provider.py`(12)、`data_engine/schema.py`(11)、
`agent/alpha_agent.py`(11)、`gp_engine/evaluation_utils.py`(10)、
`data_engine/providers/akshare_provider.py`(9)、`portfolio_manager/strategy_builder.py`(8)、
`portfolio_manager/horizon.py`(7)、`data_engine/base.py`(5)、`agent/_lc_agent.py`(5)

> 前面那份清单写于 A 档开始之前，现在已经过期（里面列的 `fast_ops`、
> `dataset_filters` 之外的绝大多数都已测量）。按"不改已结束的报告"的规矩，
> 那一节原样保留，以本节为准。

### B 档收尾时由全量回归暴露的一条（登记，本阶段不修）

12. **`chat_store` 的排序没有第二排序键，同一时钟 tick 内的记录顺序反了（中）**
    `list_sessions()` 是 `ORDER BY created_at DESC`，`get_history()` 是 ASC，
    两处都只有一个排序键。而 `created_at = datetime.utcnow()` 在 **Windows 上
    的分辨率约 15.6 ms**（实测连续两次调用返回完全相同的值），
    连着创建的记录很容易落在同一个 tick 上。

    实测：把时钟冻住让两条会话的 `created_at` 完全相同，
    `list_sessions()` 稳定返回 **(A, B)** —— 即**最老的排在最前**，
    与"最新在前"的契约恰好相反。

    发现过程值得记：这条是**全量回归自己抓出来的**，
    `tests/unit/test_db_chat_store.py::test_list_sessions_sorted_by_created_desc`
    在单跑时一直绿、在 1802 条的全量套件里红了 ——
    因为单跑时两次创建之间的 DB flush 恰好跨过一个 tick，机器忙的时候跨不过去。
    同一文件的 `test_get_history_returns_messages_in_order` 有同样的隐患，
    只是这一轮没轮到它。

    **测试侧已修**（属于本阶段范围）：新增 `tick` fixture 注入
    "每次读取前进一秒"的确定时钟，两条用例都改成钉住完整顺序，
    并额外断言"构造的时间戳确实互不相同"，防止将来又退化成测并列裁决。
    连跑 5 次稳定通过。
    **产品侧未修**：排序需要补一个第二排序键（如自增主键或 id），本阶段只登记。

### 既有测试到底还值多少：仅新 / 仅旧 / 新+旧 三方对照

起因是一个很直接的质疑：**那些首测击杀率极低的旧测试，留着是不是只在充覆盖率？**
这个不该靠判断，能测。队列 `plan_b_newonly.json`，进度 `progress_b_newonly.json`：
**只用新写的定钉文件、不带任何既有测试**，与「新+旧」对比。

| 模块 | 仅旧 | 仅新 | 新+旧 | 旧测试的边际贡献 |
|---|---:|---:|---:|---|
| `core/monitor/alpha_monitor.py` | 50.0% | 58.3% | 100.0% | **+41.7 个百分点** |
| `core/data_engine/providers/moomoo_provider.py` | 7.7% | 92.3% | 100.0% | +7.7 |
| `core/discovery/market_observer.py` | 22.7% | 95.5% | 100.0% | +4.5 |
| `core/alpha_engine/validator.py` | 27.3% | 100.0% | 100.0% | 0 |
| `core/alpha_engine/parser.py` | 15.4% | 92.3% | 92.3% | 0 |
| `core/data_engine/dataset_registry.py` | 0.0% | 100.0% | 100.0% | 0 |
| `config.py` | 35.3% | 100.0% | 100.0% | 0 |

**结论与我动手前的判断相反，记下来**：

1. **新文件不是旧文件的替代品，是补集。** 我写新用例时是盯着**存活项**写的 ——
   已经被旧测试杀死的那些根本没重复写。省力，但代价是新文件单独跑覆盖不全：
   `alpha_monitor` 的新文件单独只有 58.3%，是旧测试把它顶到 100% 的。
   按"低击杀率=没用"删掉旧测试，这个模块会掉回 58.3%，丢 5 个变异点。

2. **"零贡献"是对这个模块而言，不能据此删文件。**
   那 4 个之所以出现在模块的引用列表里，是因为它们把该模块 `monkeypatch` 掉
   去测别的东西 —— 它们真正保护的是别的模块。

3. **真正该堵的不是这些文件，是读数方式。**
   留着它们的成本只有运行时间；危害在于 "1892 passed" 会被读成
   "1892 件事被保护着"。`dataset_registry` 首测 0.0% 之前，
   任何人看测试列表都会以为它被 7 个文件覆盖着。
   → 覆盖的唯一口径应当是**该模块的变异击杀率**，不是"有几个测试提到它"。
   （是否把这条写成 `test_lessons_enforced.py` 里的强制检查，待定，需用户拍板。）

---

## 工具缺陷史续：#8 —— 改完测试没有重测，台账里的存活列表会过期

**发现方式**：B 档收尾后做交付前自检，写了一个机械核对 ——
把每个模块**最新一次测量**里的存活变异点，逐个拿行号去所有测试文件的
`PROVEN_EQUIVALENT` 表里找。89 个存活点里 **10 个找不到对应行号**，全在 A 档。

**根因**：A 档收尾阶段我改过若干测试（`daily_ingest` 的去 flake 修复、
`run_manifest` 改成打桩 `subprocess.run`、`strategy_gate` 补用例……），
但**没有在改完之后重新测量那些模块**。台账里记的还是改动之前那一轮的存活列表。

这比"数字不准"更糟：它让"每个存活项都有书面证明"这句话**看起来**成立
（表里那几条证明确实在），实际上台账列的存活项与当前代码对不上号 ——
既可能漏掉真盲区，也可能给已经被杀死的点写证明。

**实测确认是过期而非真缺证明**：按当前测试状态重测 `strategy_gate`，
存活从 7 个降到 2 个（L158×2 / L336 / L417 / L419 全部已被杀死），
只剩 L177、L310 两个 epsilon 守卫 —— 而这两个本来就在证明表里。

**规矩（补进本文件的测量前置条件）**：
> **任何一次测试改动之后，该模块必须重新做整模块测量。**
> 单点 `verify_mutant.py` 只能用来快速确认"这一个杀死了"，
> **不能**用它的结论去更新台账里的击杀率 —— 台账的每个数字必须来自
> 一次完整的整模块运行，并在结果表里注明取自哪个 `progress*.json`。

**这个核对本身应当固化**：上面那段"逐个存活点核对行号"的检查现在只是我
手跑的一次性脚本。它应该进 `test_lessons_enforced.py`，否则下一次还是靠人声称。
（属于改动测试体系，待用户确认后再加。）

---

## 已登记缺陷编号表（与 `tests/test_known_defects.py` 一一对应）

上面各节的产品问题此前只有序号没有编号，测试侧引用起来对不上。
这里给出**正式编号**，`test_known_defects.py::test_defect_registry_matches_the_ledger`
会机械核对两边一致。

| 编号 | 模块 | 一句话 | 行为断言 |
|---|---|---|---|
| **B-1** | `fast_ops.bn_ts_rank` | bottleneck 分支值域是 [-1/w, 1/w]，非 docstring 承诺的 [0,1] | 有 |
| **B-2** | `fast_ops.ts_corr` | cov(ddof=0)/std(ddof=1) 不配套 → 系统性偏低 (w-1)/w | 有 |
| **B-3** | `fast_ops.cs_rank` | 并列处理是序数名次，非 docstring 声称的平均名次 | 有 |
| **B-4** | `fast_ops.ts_entropy` | `n_bins=1` 静默返回 -0.0，而非 NaN/报错 | 有 |
| **B-5** | `fast_ops.bn_ts_max/min` | NaN 策略在 bottleneck 与 numpy 分支之间不一致 | 有 |
| **B-6** | `fast_ops.cs_rank` | 含 NaN 的截面上值域越出 [0,1] | 有 |
| **B-7** | `fast_ops` 七个滚动算子 | 面板短于窗口时抛 ValueError，而非返回 NaN | 有 |
| **B-8** | `ml_engine.proxy_model` | `_fit()` 放弃后仍走模型分支 → AttributeError | 有 |
| **B-9** | `ml_engine.proxy_model` | `use_label_encoder=False` 对 xgboost 3.x 已无意义 | 无（整洁问题） |
| **B-10** | `fast_ops` | 向量化分支被 `except Exception` 完全兜住 | 无（结构问题） |
| **B-11** | `data_partitioner` | "OOS 为空"守卫不可达 | 无（结构问题） |
| **B-12** | `db.chat_store` | ORDER BY 无第二排序键，同一 tick 内顺序反了 | 有 |
| **A-1** | `tasks.daily_ingest` | 增量被写进 PIT 两次 | 有（见 test_daily_ingest_increment） |

### 为什么要有 `test_known_defects.py`

此前这些缺陷只被"钉住当前错误行为"的断言覆盖（断言错的值，注释写"修好后改成对的值"）。
钉住现状有价值 —— 任何一处算术被改坏仍会被抓到 —— 但它有个致命副作用：
**已知坏掉的东西在每次运行里完全不可见**，十几个缺陷躺着，套件照样报全绿。

`test_known_defects.py` 用 `xfail(strict=True)` 补上另一半：断言**应有行为**。
于是每次运行的汇总行会显示 `N xfailed`，已知坏掉的数量摆在台面上；
谁修好了缺陷，用例变 XPASS → strict 判失败 → 强制他同步更新三处
（xfail 标记、模块里钉住现状的断言、本台账）。

两套断言互补：钉住现状的那条提供**检出能力**，xfail 那条提供**可见性与修复告警**。

---

## Tier C 收尾：把上一轮误判为"等价"的三处翻案

Tier C rerun 2 剩 12 个存活点。逐个坐下来查之后，**有三处是我自己判错了**——
把可杀的漏测写成了"等价变异证明"。这一节记录翻案过程，因为它比结论更重要：
**等价性证明必须枚举所有对外可观察面，漏一个面就是假证明。**

### 翻案 1：`alpha_pool.py:204` `if len(self._entries) > self._max_size:`

原证明说：`len == max_size` 时切片取回全部、`removed` 为空、`_seen_dsls` 不变、
`top_k` 自己排序，所以观察不到差别。

**漏了 `all_entries()`**——它返回 `list(self._entries)` 的**插入顺序**，
而 `>=` 分支会就地 `sort(key=fitness, reverse=True)` 把它重排。
`get_orthogonal_signals()` 的 `enumerate(valid)` 同样吃这个顺序。

改为直接杀：`test_pool_order_is_insertion_order_until_it_actually_overflows`
按插入顺序钉死，并验证真正溢出时淘汰的是 fitness 最低那条。

### 翻案 2：`alpha_pool.py:174` `1.0 / (s[:k] + 1e-9)` 的 `+`

原先没有证明，只是存活。关键观察：**n 条信号去中心化后秩最多 n-1**，
最小奇异值 ≈ 1e-17，于是 `1/(1e-17 - 1e-9)` 与 `1/(1e-17 + 1e-9)`
差一个**符号**（±1e9）。松断言（`allclose` 默认 rtol=1e-5）两种都放过。

改为跟公式的参考实现逐位比，`rtol=1e-12, atol=0`，并把"最小奇异值确实退化"
这个前提一并钉住 —— 前提没了，符号论证也就不成立，测试会红着提醒。

### 翻案 3：`fitness.py:133` `return all(...) if children else False`

原证明说 parser 产不出零子节点的 `ArithmeticNode`，所以分支不可达。
**理由站不住**：`ArithmeticNode.__init__` 没有元数校验，`ArithmeticNode("add", [])`
直接就能造，而 GP 的变异/交叉是程序化拼节点、不走 parser。
改为直接断言 `_is_scale_stable(ArithmeticNode("add", [])) is False`。

### 仍然成立的等价性证明（新增两条机械验证）

| 位置 | 结论 | 机械验证 |
|---|---|---|
| `alpha_pool.py:170` `mean(axis=0, keepdims=True)` → `False` | 等价：`mu` 只用于 `mat_clean - mu`，numpy 按尾轴对齐，(n,T)−(T,) 与 (n,T)−(1,T) 逐位相同 | `test_keepdims_makes_no_difference_to_the_broadcast`，四组 (n,T) 形状 `array_equal` |
| `alpha_pool.py:172` `svd(full_matrices=False)` → `True` | 等价：`Vt` 只以 `Vt[:k]` 被读，k ≤ min(n,T)，两种形式的前 min(n,T) 行是同一组右奇异向量 | `test_full_matrices_does_not_change_the_rows_that_are_read`；**注意** n<T 时两种形式走不同 LAPACK 驱动，实测有 ≤ 3.4e-16 的舍入抖动（非语义差别），断言用 `≤1e-14` 的上界而非逐位相等 |
| `gp_engine.py:272` `if denom > 0:` → `>=` | 等价：`argsort(argsort(x))` 恒返回 0..n-1 的**排列**，去中心化平方和恒为 n(n²−1)/12；`n_valid ≥ 5` ⇒ denom ≥ 10 | `test_the_rank_denominator_can_never_be_zero`，n=5..39 × 4 类输入（全并列/全零/连续/大量重复） |
| `gp_engine.py:254` `(close[1:] - close[:-1])` → `+` | 等价：`(P₁+P₀)/P₀ = 2 + r` 是 r 的严格单调增变换，而 fwd_ret **只**进 Spearman 秩相关 | 已有 `test_sum_form_is_a_monotone_transform_of_the_return`，另加 `fwd_ret` 使用点计数的失效告警 |
| `gp_engine.py:258` `sig_arr.shape[0] - 1` → `+ 1` | 等价：signal 行数与 close 恒等（都是 T），两种取值下 `min(...)` 都取 T−1 | 已有 `test_signal_and_close_always_have_the_same_row_count` |
| `gp_engine.py:171` `mapped and mapped in ...` → `or` | 等价：`_ALIAS` 的每个目标值都是 `_SEED_DSLS_BY_FAMILY` 的键，未知家族时 mapped 是空串、两式同假 | 已有 `test_every_alias_target_exists_in_the_seed_table` |

### 顺带查出的自身问题：`test_gp_engine_fitness.py` 有整段重复

`TestRankIc` 里有 57 行被整段复制粘贴了两遍，四个用例名重名 ——
Python 只保留后定义的那份，**前一份从未被执行过**。已去重。
这也说明：用例数（"25 个测试"）同样不是强度指标，重名会静默吞掉用例。

### 新登记缺陷 C-1：GP 适应度的截面秩不处理并列

翻案 3 的副产品。`rs = np.argsort(np.argsort(s[mask]))` 是**序数**名次。
当日信号对所有票同值时，它给出的是 0,1,2,… —— 也就是**列在面板里的位置**。

后果：`(close/close)` 这种截面恒定、零信息的信号，mean_IC 不是 0，
而是"ticker 排列顺序 vs 未来收益秩"的相关系数。**把面板的列顺序打乱，
同一个信号的 fitness 会变** —— GP 的分数依赖数据加载时的列序，而列序不是市场事实。

正确做法是并列取平均名次（`scipy.stats.rankdata` 或手写 tie-average），
此时常数信号去中心化后全 0 → denom = 0 → 该截面被 `if denom > 0:` 正确跳过
（那条守卫本来就是为这个场景写的，只是现在永远进不去）。

| 编号 | 模块 | 一句话 | 行为断言 |
|---|---|---|---|
| **C-1** | `gp_engine._evaluate_individual` | 截面秩用 argsort(argsort) 不处理并列，零信息信号拿到由列序决定的非零 IC | 有（`TestGpFitnessRankTies`，strict xfail） |

已登记缺陷总数由 18 升至 **19**，`test_the_outstanding_defect_count_is_visible`
的数字同步改为 19。

---

## 自伤教训 #6：源码文本断言用**子串**判断，同一串在文件里出现多次就杀不掉

写 `router.py` 的 `daemon=True` 用例时用了：

```python
assert "threading.Thread(target=_run, daemon=True)" in src
```

`daemon=True` 这件事在进程内**观察不到**（差别只在解释器退出时显现），
所以源码断言是合理的。但 router.py 里有 **3 处** `daemon=True`，其中
**两处一模一样**（L431 与 L2471 都是 `threading.Thread(target=_run, daemon=True)`）。
把 L431 那处改成 `False`，L2471 那处仍让子串命中 —— 断言照过。
在隔离副本上实测：三处逐个改，**一处都杀不掉**。

这和工具缺陷 #3（"一行上有多个 `*`，变异器只改第一个"）是同一类错误的两面：
**一个位置的性质，不能用全文件的存在性去断言。**

改法是走 AST，断言的是**全称命题**而不是存在性：

```python
for node in ast.walk(ast.parse(inspect.getsource(R))):
    if isinstance(node, ast.Call) and 是 threading.Thread(...):
        assert kw["daemon"] 是常量 True, f"第 {node.lineno} 行不是守护线程"
```

改完之后三处逐个变异，**三处全部被杀**。附带好处：以后新加线程忘了写
`daemon` 同样会被这条抓到 —— 存在性断言给不了这个保证。

**规则**：源码文本断言只在"进程内确实观察不到"时才用；一旦要用，
先确认那个串在文件里**唯一**，不唯一就走 AST 写成全称命题。
本轮据此改写了 `test_every_thread_the_router_spawns_is_a_daemon`。

### 同一轮里已用行为断言替换掉的源码断言

`main.py:422` 的 `getattr(args, "walk_forward", False)` 起初也想用
`inspect.getsource` 断言，后来改成**真跑 `run_realistic()`、看它选了哪个
回测器**（把两个回测器换成只记账的替身）。行为断言在这里是可行的，
就不该退回文本断言。

---

## C 档方法论：三条"覆盖了行，却没覆盖能区分真假的那一格"

C 档补强过程中，同一类错误反复出现，值得单独立一节。它们的共同点是
**用例确实执行到了那一行，但选的观察面看不出两种取值的差别**——
覆盖率报告全绿，变异却活着。

### 1. 参数组合恰好让两种取值同解

`_agent.py` 的 `if intent == "workflow_b" and dsl_hint:`，既有用例喂的是

| intent | dsl_hint | `and` | `or` |
|---|---|---|---|
| workflow_a | None | 假 | 假 |
| workflow_b | "rank(close)" | 真 | 真 |

**两格都同解**。真正能分开的是 `(workflow_b, None)` 那一格，而
`_detect_intent` 产不出来 —— 必须打桩造。

同类：`population_evolver` 的过拟合公式 `(is - oos)/abs(is)`。
只喂"极端过拟合"（is=2.0, oos=0.0）时 `+` 与 `-` 都被 `clip(0,1)` 压成 1.0，
**必须喂"完全没退化"（oos == is）** 才分得开：正确 0.0，错误 1.0。
四份同样的公式（`_evaluate_one_single` / `_evaluate_one_multi` /
`_extract_metrics` / `alpha_workflows._quick_metrics`）全栽在这一点上。

### 2. 下游把差别吸收掉了

`alpha_workflows._expand_for_optimization` 有两个循环：变异循环 + 随机补位循环。
把变异循环的尝试上限 `n_mutations * 15` 改成 `/ 15`，它**一次都不跑**，
但补位循环会把总数补齐到 `n_mutations + 3` —— **候选总数一模一样**，
只是里面一个变异体都没有，全是随机 alpha。

后果不小：Workflow B 的职责是"针对用户给的这条 DSL 做结构优化"，
这样就退化成了纯随机搜索，用户的输入被无视，而 `len(candidates)` 完全正常。

抓法：**数变异算子被调用了几次**，不要数候选条数。

同类：`population_evolver._generate_next_population` 末尾的
`return next_gen[: self._pop_size]` 把"多填一个"截断掉了 ——
那个 `<` → `<=` 是真等价（已证），但它说明**凡是产出经过截断/归一化的地方，
计数类断言都要往上游挪一层**。

### 3. 前置层把被测层掩盖了

`_generate_diverse_seeds` 的 Layer 1（AlphaAgent 回退种子，实测 5 条）
与 Layer 2（关键词模板）是**无条件追加**的，只有 Layer 3/4 看 `n_target`。
n_target=12 时填充循环只需补 6 条，把它的尝试上限压成 1 次，
结果还剩 8 条 —— 断言"填满 12"抓得到，但换个参数就抓不到了。

抓法：把前置层**全部关掉**，让被测层单独对结果负责。

### 这一轮用来定位问题的手段：新测试单跑

上面这些不是靠读代码想出来的，是靠一次**只用新测试**的测量
（`plan_wf_newonly.json`，基线 3 秒 vs 原选择的 16 秒 + 大量死循环变异）
把"旧测试盖住的"与"我确实漏写的"分开之后逐条查出来的。

三路对照（仅旧 / 仅新 / 新+旧）此前只在 B 档做过一次，用来衡量旧测试的
边际贡献；这一轮发现它还有第二个用途：**定位自己的盲区**。
只跑新测试时存活、而新+旧时被杀的点，说明是旧测试在兜底；
两种情况下都存活的，才是真正需要动手的。

## 工具改进：单点超时与进程树

`alpha_workflows` 首测用 1800s 超时跑了 11.6 小时才判完 12 个变异点 ——
平均 58 分钟/点，**比超时上限还长**。原因是
`subprocess.run(timeout=...)` 超时后只 kill 直接子进程，pytest 派生的孙进程
还握着 stdout 管道，`communicate()` 继续阻塞到孙进程自己退出，超时上限形同虚设。

改法（`mutate.py`）：
- 改用 `Popen` + `wait(timeout=...)`，stdout/stderr 直接丢弃（不再用管道）
- 超时后 Windows 走 `taskkill /F /T /PID`、POSIX 走 `killpg`，**连孙进程一起杀**
- 默认超时从 1800s 降到 300s，`runner.py` 加 `--timeout` 透传

选超时值的规矩：**取该测试选择基线耗时的 10 倍以上**。实测基线
`_tools` 70s / `population_evolver` 35s / `router` 54s，故用 600s。
超时判为"击杀"是对的（变异让测试跑不完本身就是被检出），
但上限太紧会把"慢而正确"的变异误判成击杀，制造假强度。

---

# C 档结果（15 个模块 / 439 个变异点）

| 模块 | 首测 | 终测 | 存活 |
|---|---|---|---|
| `agent/_agent.py` | 0.0% | **100%** | 0 |
| `agent/_critic.py` | 50.0% | **100%** | 0 |
| `api/chat_router.py` | 40.0% | **100%** | 0 |
| `core/discovery/discovery_engine.py` | 33.3% | **100%** | 0 |
| `core/gp_engine/fitness.py` | 35.0% | **100%** | 0 |
| `core/gp_engine/mutations.py` | 4.6% | **100%** | 0 |
| `core/gp_engine/population_evolver.py` | 10.4% | **100%** | 0 |
| `main.py` | 9.5% | **100%** | 0 |
| `tasks/backup.py` | 57.1% | **100%** | 0 |
| `tasks/scheduler.py` | 5.0% | **100%** | 0 |
| `api/router.py` | 16.0% | 98.7% | 1 |
| `agent/_tools.py` | 4.5% | 95.5% | 1 |
| `core/gp_engine/alpha_pool.py` | 59.1% | 95.5% | 1 |
| `core/workflows/alpha_workflows.py` | 20.3% | 92.8% | 5 |
| `core/gp_engine/gp_engine.py` | 0.0% | 73.3% | 4 |
| **合计** | — | **97.3%（427/439）** | **12** |

**12 个存活项全部有机械可验证的等价性证明**（达标标准是"存活项 100% 处置"，
不是击杀率数字本身）：

| 位置 | 证明要点 | 验证用例 |
|---|---|---|
| `gp_engine` L171 | `_ALIAS` 的每个目标值都是 `_SEED_DSLS_BY_FAMILY` 的键 | `test_every_alias_target_exists_in_the_seed_table` |
| `gp_engine` L254 | `(P₁+P₀)/P₀ = 2+r`，是 r 的严格单调增变换，而 fwd_ret 只进 Spearman 秩相关 | `test_sum_form_is_a_monotone_transform_of_the_return` |
| `gp_engine` L258 | signal 行数与 close 恒等，两种取值下 `min(...)` 都取 T−1 | `test_signal_and_close_always_have_the_same_row_count` |
| `gp_engine` L272 / `alpha_workflows` L603 | `argsort(argsort(x))` 恒为 0..n−1 的排列 ⇒ 平方和恒为 n(n²−1)/12，n≥5 时 denom ≥ 10 | `test_the_rank_denominator_can_never_be_zero` |
| `alpha_workflows` L335/338/342 | `_try_add` 两个调用点都是 `ast.Expr` 语句，返回值被丢弃 | `test_try_add_return_value_is_discarded_at_every_call_site` |
| `alpha_workflows` L573 | `is_data` 与"键集相等"两个子式不可能独立取值（失败时成对 pop + `len<2` 早退） | `test_the_two_operands_cannot_vary_independently` |
| `router` L1423 | `opend_up` 初值在 try 成功分支与 except 分支都被覆盖 | `test_the_initial_flag_is_always_overwritten` |
| `_tools` L476 | LLM 返回值域 `{None,'point','hoist','param'}` ⊆ 权重表键集 | `test_every_possible_hint_is_already_a_weight_key` |
| `alpha_pool` L170 | numpy 广播按尾轴对齐，`(n,T)−(T,)` 与 `(n,T)−(1,T)` 逐位相同 | `test_keepdims_makes_no_difference_to_the_broadcast` |

## 自伤教训 #7：**能杀就不要写等价证明**

第二轮我把 `population_evolver` 的
`while len(next_gen) < self._pop_size and attempts < ...` 的 `<` → `<=`
判成了等价变异，理由写得很像样：末尾 `return next_gen[: self._pop_size]`
会把多产出的那个个体截掉，**返回值逐元素相同**（还配了一条机械验证用例）。

第三轮才发现这个结论是错的：

> 输出看不出来，**调用次数看得出来**。多跑一轮就是多一次
> `point_mutation` / `generate_random_alpha` 调用。

同一行的 `and` → `or` 更明显：种群填满之后还会空转到 `pop_size * 20` 次
尝试上限 —— 每一代白烧几百次随机生成 + 校验。这根本不是"观察不到"，
是我**没找对观察面**。三个点最后全部改成杀死。

**规则**：写等价性证明之前，先穷举可观察面 ——
返回值、副作用、**调用次数**、日志、耗时、异常。
只有当**每一个**都证明不受影响时才能写等价；
"我试的那个观察面没差别"不是等价。

**待办（交付审计前）**：A/B 档已写的 35 + 若干条等价证明，
应按这条标准复查一遍，重点是那些理由为"下游把差别吸收了"
（截断、clip、归一化、兜底重算）的条目 —— 它们最可能只是观察面没选对。

## C 档补强的测试文件

| 文件 | 用例数 | 针对模块 |
|---|---|---|
| `test_gp_mutation_operators.py` | 98 | `mutations.py` |
| `test_alpha_workflows_internals.py` + `_round2` + `_round3` | 62+40+14 | `alpha_workflows.py` |
| `test_population_evolver_internals.py` + `_round2` + `_round3` | 44+28+15 | `population_evolver.py` |
| `test_api_router_guards.py` + `test_api_router_round2.py` | 35+38 | `router.py` |
| `test_agent_tools_guards.py` | 34 | `_tools.py` |
| （A/B 档已有）`test_agent_routing.py` 等 | — | 其余模块 |

全量回归：**2523 passed / 1 skipped / 20 xfailed**（24 分 30 秒）。
其中 20 个 xfailed = 已登记但未修复的产品缺陷，每次运行都摆在汇总行上。

---

# tests/ 目录整理（2026-09-13）

## 结论先行：废料几乎没有，但整理过程炸出三个真问题

用**四条可机械确认**的判据扫完 2550 个用例（不是靠感觉挑）：

| 判据 | 命中 |
|---|---|
| 空壳文件（整文件不 import 任何 app 模块） | **0** |
| 无条件 skip（永不执行） | **0**（扫出的 10 条全是 `pytest.importorskip`，依赖守卫，是扫描器误报） |
| 重言式（断言恒真，常量折叠后判定） | **1** |
| 逐字重复（AST 指纹相同） | **2 条用例** |

**真正删掉的只有那 2 条**逐字重复（`test_invariants.py` 与 `test_lessons_enforced.py`
各有一份完全相同的 data_source 检查，保留后者）。其余一律保留 ——
原则是"只有确认完全没用才删，模糊不确定就留"。

那 1 条重言式是
`assert ("" if "" is not None else current_git_commit()) == ""` ——
`"" is not None` 恒真，整条等价于 `assert "" == ""`，
**把产品的表达式在测试里抄了一遍而没有碰产品**。
它有真实意图（空串是显式取值，不该触发自动探测），所以**修而不删**：
改成真存一条 `git_commit=""` 的记录，并把自动探测打成"一旦被调用就失败"的地雷。

> 第一版重言式扫描器只查"断言里一个名字都没有"，漏掉了这条（它里面有
> `current_git_commit` 这个名字，只是那个调用**永远执行不到**）。
> 加了常量折叠（`X is not None` 且 X 为字面量 → 折叠）之后才抓到。

## 发现 1：`/api/chat/stream` 零测试，靠一句 docstring "被覆盖"

删掉那 2 条重复用例后，`test_no_new_untested_api_route` 立刻变红。

回查发现：这条路由此前之所以算"已覆盖"，**只因为被删那条用例的 docstring 里
写了 `/api/chat/stream` 这个字符串** —— enforcement 的判据是
"路径字面量在测试源码里出现过"，注释与 docstring 同样命中。
41 条 API 路由里只有这一条是假覆盖，而它恰好是**前端唯一消费的那条**。

这与「自伤教训 #6」（源码子串断言）是同一类错误，这次出现在
**项目自己的 enforcement 检查里**。两处都修：

1. 新增 `integration/test_api_chat_stream.py`（7 条）——
   测 SSE 的真契约：帧格式 `data: <json>\n\n`、必须以 `done`/`error` **终止**、
   终局带 `data_source`、三个反缓冲头、至少有一个增量 `text` 事件
2. 判据加强：先剥掉 docstring 与注释再匹配（`_executable_text`）

## 发现 2：共享 session 级 DB 造成的顺序依赖

`StrategyStore()` 不传 `db_url` 时回落到 `settings.database_url`，
而 conftest 的 `_hermetic_run_flags` 把它指向一个 **session 级共享临时库**。
`unit/db/` 里的策略端点用例往里存了 `status="active"` 的配置，
后面 `run_portfolio` 读 `latest_active()` 拿到它 →
`using_active_config` 非 None → **边际准入分支被整个跳过** → `selection` 为 None。

旧的扁平目录下执行顺序恰好让它没暴露；重组后 `unit/db/` 排到
`unit/trading_context/` 前面，`test_marginal_selection_runs_when_enabled` 立刻变红。

修法：给该文件加 autouse fixture 把 `latest_active` 默认打成 None，
使整个文件与执行顺序无关；需要"有 active 配置"的那条用例在体内自行覆盖。

## 发现 3（最隐蔽）：`importlib.reload(app.main)` 污染整个会话

`test_main_startup_guards.py` 里验"重复导入不会重复插 sys.path"时
调了 `importlib.reload(app.main)`。而 `app/main.py` 顶层有 `app = FastAPI(...)`，
reload 会**重新执行模块、造出一个全新的 FastAPI 实例**绑到 `app.main.app`。

后果链：

1. 任一测试用 conftest 的 **session 级 `test_client`** → 它包住**当时**的 app 对象
2. reload → `app.main.app` 变成**新**实例
3. 之后任何 `from app.main import app; app.dependency_overrides[...] = ...`
   改的是新实例 → **覆盖到不了 client** → 接口打到真实依赖上

症状是毫不相干的 **404 / 409**，完全看不出跟 reload 有关。

洗牌顺序（seed=20260913）下的时间线严丝合缝：
最早建 client 在 **#10**、reload 在 **#2149**、失败用例在 **#2350**。
三步顺序复现：修复前 `1 failed`，修复后 `3 passed`。

修法：把"重复导入"那半移到**子进程**。既消除污染，也更忠实 ——
要验的本来就是"全新解释器里导入两次不会重复插 `sys.path`"。

> 注：前两次复现尝试失败（都通过），因为漏了第 1 步"先建 session client"。
> 当时如实说了"假设未被证实"，补齐顺序后才确认。
> **没复现出来之前不要把猜测当成因**。

## 新增常备工具：`shuffle_check.py`

本仓**没有** pytest-randomly —— 顺序依赖此前**没有任何机制在防**
（我一度误以为有，`-p no:randomly` 关的是个不存在的插件）。
新增一个零依赖的收集期洗牌插件：

```bash
python -m pytest tests/ -q -p shuffle_check                   # 默认种子
SHUFFLE_SEED=123 python -m pytest tests/ -q -p shuffle_check  # 换种子
SHUFFLE_DUMP=order.txt python -m pytest tests/ --co -p shuffle_check   # 导出顺序
```

失败后用同种子 + `SHUFFLE_DUMP` 导出执行顺序，即可二分定位是哪条前置用例弄脏了状态。
上面两个顺序依赖就是这么找出来的。

**交付前必须过这一关**：固定顺序全绿 **不等于** 套件可信。

## 目录结构

114 个文件 `git mv`（保留历史），33 个 `test_phaseN_*.py` 按实际测试内容改名
（`test_phase7_paper.py` → `test_paper_broker_replay_parity.py` 之类）。

```
tests/
  README.md          导航索引：每个文件测什么、强度怎么保证、该盯哪两个数
  meta/          3   对套件本身的约束 —— 审计入口
  unit/<包>/    88   17 个子目录，与 app/ 的包结构一一对应
  integration/  33   跨包链路 + 真实 HTTP 端点
  golden/        1
  performance/   2
```

**移动前必须先做的一步**：把 3 处 `Path(__file__).resolve().parents[1]`
这类**硬编码层数**改成"向上找含 `app/` 的目录"。否则文件下沉一层之后
它会静默指到 `tests/` 而不是 `backend/`，`rglob("*.py")` 扫出空集合 ——
而**对空集合的全称断言恒真**，一批约束会静默变成空转且没有任何报错。

## 整理后的口径

| 指标 | 值 |
|---|---|
| 文件 / 用例 | 127 / 2550 |
| 固定顺序全量 | 2529 passed · 1 skipped · 20 xfailed · **0 failed**（11分48秒） |
| **洗牌顺序全量** | 2529 passed · 1 skipped · 20 xfailed · **0 failed**（18分02秒） |
| 全量耗时 | 24 分钟 → **11分48秒**（重组后 fixture 局部性变好） |

`20 xfailed` = 已登记但尚未修复的产品缺陷，每次运行都摆在汇总行上。
