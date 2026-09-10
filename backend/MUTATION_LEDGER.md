# 变异测试台账（断言强度的唯一可信记录）

> **为什么需要这张表**：我在会话中多次口头报出击杀率，其中多个数字后来被证明是
> 工具缺陷造成的**虚高**（语法错被误记为"杀死"）。口头数字不可信，一律以本表为准。
>
> **测量前提（缺一则数字作废）**：
> 1. 变异必须在 **隔离副本** 中进行（`mutate.py` 复制 backend/ 到临时目录），主工作区零改动
> 2. 变异后必须仍能 `ast.parse`，否则不计入分子分母
> 3. 替换行必须保留行尾换行
> 4. 不并行运行（原地改文件的工具不是并行安全的）
> 5. **变异器必须真的生效** —— 见下方"工具缺陷史"，两次虚高都源于此
> 6. **每行每个变异器各算一个变异点**，不是每行只取第一个
>
> 工具：`scratchpad/mutate.py`（隔离版，2026-09-09 起）

## 工具缺陷史（每一条都曾让整批数字作废）

| # | 缺陷 | 后果 | 发现方式 |
|---|------|------|----------|
| 1 | 替换行丢了行尾 `\n`，与下一行粘连 | 语法错 → 误记"杀死" | 手查存活列表 |
| 2 | `-> float` 注解被 `>` 变异器改成 `->=` | 同上 | 同上 |
| 3 | 原地改源码 + 并行运行 | **变异被提交进仓库**（24c251a） | `git diff` |
| 4 | 只认 `tokenize.STRING`，漏 Python 3.12 的 `FSTRING_*` | f-string 里的 `>` 被当代码 → 假存活 | 逐条核对存活项 |
| 5 | `re.sub(pat, rep, m.group(0))` —— **正向后顾断言在孤立片段上必然失配** | `*` `+` `-` 三个算术变异器**从未生效过**，`transaction_cost` 129 个候选行只有 22 个被变异 | 存活列表清一色是比较符，一个乘除都没有 |
| 6 | 每行只做第一个匹配的变异器 | 比较符排在算术符前，`abs(dw) * val / p if p > 0` 这类行永远只测 `>` | 与 #5 一并发现 |

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
| `tasks/daily_trading_loop.py` | 65.1% → 重测中 | 63 | 22 | 🔄 11 处已写用例，3 组已证明等价 | ~~18.6~~ ~~66.0~~ |

**五个模块已达标**（存活项 100% 已处置）。新增用例合计 **71 条**，
每一条都对应一个"改坏了原本没人发现"的具体位置，不是补覆盖率。

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

## daily_trading_loop 处置详情（65.1% → 重测中）

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
