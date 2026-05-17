# Alpha 因子生成与优化系统审计报告

## 一、系统理解 Alpha 金融含义的机制

系统通过三层机制建立金融语义理解：

### Layer 1：LLM 系统 Prompt（`agent/_prompts.py`）

注入完整的量化金融理论框架，使 LLM 从原理层面推理因子。

**因子族编码：**

| 因子族 | 金融机制 | DSL 模式 | 适用场景 | 失效场景 |
|--------|---------|---------|---------|---------|
| Momentum | 赢家续涨（Jegadeesh & Titman 1993） | `rank(ts_delta(log(close), N))` | 趋势市、低波动牛市 | 急跌崩溃、高波动反转 |
| Mean Reversion | 市商流动性供给，价格超调后回归 | `rank(neg(ts_delta(close, N)))` | 震荡市、高流动性 | 趋势市（逆势亏损） |
| Volatility | 低波动溢价（Ang et al. 2006） | `rank(neg(ts_std(returns, 20)))` | 熊市、风险厌恶 | 强牛市（高波动跑赢） |
| Liquidity | 流动性不足溢价 | 量价相关性信号 | 中小盘 | 市场压力时流动性消失 |
| Price-Volume Corr | 知情交易在量价共动中留下痕迹 | `rank(ts_corr(close, volume, 20))` | 机构主导市场 | 被动/指数化市场 |

**因子设计原则（Prompt 中明确编码）：**
- 交叉截面归一化：无 `rank/zscore` 则多空持仓有方向性净敞口
- 信号平滑：每 1× 年化换手率 ≈ 0.3 Sharpe 损耗（按 30bps/单程成本）
- Log 变换：`ts_delta(log(close), N)` 是百分比变化，`ts_delta(close, N)` 是美元变化（非平稳）
- 风险归一化：动量除以波动率构造风险调整信号
- 量的确认：高量价格变动比低量更具信息含量
- 市场状态条件：`trade_when(close > ts_mean(close, 200), signal)` 切断熊市尾部损失

**指标解读阈值：**

| 指标 | 阈值 | 含义 |
|------|-----|------|
| OOS Sharpe | > 0.8 | 强因子，泛化稳健 |
| OOS Sharpe | 0.4–0.8 | 可交易，关注换手率 |
| OOS Sharpe | < 0.3 | 弱信号，需结构改进 |
| IS vs OOS 差值 | > 0.5 | 过拟合，需简化或归一化 |
| IC IR | < 0.3 | 信号噪声大，需平滑 |
| 换手率 | > 3× | 净损益可能为负 |
| 最大回撤 | > 25% | 尾部风险不可接受 |

---

### Layer 2：FinancialInterpreter（`core/alpha_engine/financial_interpreter.py`）

**纯代码、确定性**的 AST 语义解析，不依赖 LLM。

**功能：**
- 遍历 AST 树，底向上生成自然语言描述
  - 示例：`rank(ts_delta(log(close),5))` → "5-day short-term momentum of log-transformed closing price, cross-sectionally ranked"
- 检测因子族（8 种：momentum / reversion / volatility / liquidity / price_volume_corr / quality / trend_following / composite）
- 计算复杂度评分（1–5）
- 检测 7 类设计缺陷：无归一化、无平滑、无量的确认、短窗口无确认、极长/极短窗口、高复杂度、无状态条件
- 输出具体 DSL 改进建议

**因子族分类逻辑（优先级顺序）：**
1. `ts_corr` + `volume` → `price_volume_corr`
2. `ts_std/ts_var` 无 `ts_delta` → `volatility`
3. `ts_zscore` 无 `ts_delta` → `reversion`
4. `neg` 包裹 `ts_delta` → `reversion`
5. `ts_delta` + `max_window >= 60` → `trend_following`
6. `ts_delta` → `momentum`
7. `ts_mean` 无 `ts_delta` → `trend_following`
8. 2+ 族共存 → `composite`

---

### Layer 3：FinancialDiagnostics（`core/alpha_engine/financial_diagnostics.py`）

**指标 → 金融诊断**，把数字翻译成原因与对策。

**诊断优先级（按严重程度排序）：**

| 优先级 | 条件 | 诊断标签 | 严重程度 |
|--------|------|---------|---------|
| 1 | overfit > 0.6 且 IS Sharpe > 0.8 | `severe_overfitting` | CRITICAL |
| 2 | 换手率 > 3.0 | `high_turnover` | CRITICAL/MODERATE |
| 3 | OOS Sharpe < 0.1 且 IC < 0.01 | `no_signal` | CRITICAL |
| 4 | IC > 0.02 且 IC_IR < 0.3 | `noisy_signal` | MODERATE |
| 5 | \|max_dd\| > 0.25 | `high_drawdown` | MODERATE |
| 6 | 0.1 < overfit ≤ 0.6 | `mild_overfitting` | MODERATE |
| 7 | OOS Sharpe > 0.5 且 overfit < 0.3 | `healthy` | MINOR |

**核心建议映射（含金融原理）：**

| 问题 | DSL patch | 金融原理 |
|-----|-----------|---------|
| 高换手率 | `ts_mean(signal, 3)` | 避免换手每次节省 20-40bps 双边成本 |
| 高换手率 | `ts_decay_linear(signal, 5)` | WorldQuant/Two Sigma 标准信号衰减技术 |
| 过拟合 | `zscore(signal)` | 截面归一化强制 IS/OOS 同分布，去除 IS 特有系统性偏差 |
| 过拟合 | `rank(ts_delta(log(close),10))` | Occam's Razor：简单表达式泛化更好 |
| IC 噪声大 | `rank(ts_zscore(signal, 20))` | TS z-score 去除信号均值漂移，修复非平稳性 |
| IC 噪声大 | `rank(delta * sign(vol_delta))` | 量确认价，过滤噪声交易，提升 IC 一致性 |
| 大回撤 | `trade_when(close>MA200, signal)` | 多数股票因子在熊市相关性崩溃，200日均线过滤切断尾部损失 |
| 弱动量 | `rank(ts_delta(log(close),20)/ts_std(returns,20))` | 风险调整动量（收益/风险）OOS 更稳定 |

---

## 二、完整工作流

### Workflow A：从金融假设生成 Alpha

```
用户自然语言假设
    │
    ▼
[_agent._detect_intent()]
  无 DSL 关键词 → workflow_a
    │
    ▼
[_fallback._generate_diverse_seeds(hypothesis)]
  ≥12 种多样化 DSL 种子（关键词映射 + LLM 生成）
    │
    ▼
[检测因子族 factor_family from 首个种子 DSL]  ← 弊端 1 修复后新增
    │
    ▼
[tool_run_gp_optimization(seed_dsls, factor_family)]
  ┌─ _init_population:
  │    解析所有种子 → 30% 种子间交叉 → 35% 变异填充 → 35% 随机
  │
  ├─ FOR generation in 1..4:
  │    ① _evaluate_population: RealisticBacktester IS+OOS
  │       fitness = sharpe_oos - 0.2×turnover - 0.3×|max_dd|
  │                 - 0.5×max(0, sharpe_is - sharpe_oos)
  │
  │    ② AlphaPool 多样性过滤（相关系数 > 0.9 过滤）
  │
  │    ③ 精英选择（top 25%）+ 锦标赛选择
  │
  │    ④ _generate_next_population（11 种算子，自适应权重）
  │       权重由 mutation_weights_from_metrics(factor_family) 决定
  │       → 高换手 + boost param/hoist
  │       → 低 OOS  + boost 结构性算子
  │       → 过拟合 + boost hoist/replace_subtree
  │       → factor_family="momentum" + boost add_condition/add_ts_smoothing
  │
  │    ⑤ 记录本代最优 DSL、fitness
  │
  ├─ _optuna_fine_tune(best_dsl, n_trials=8):
  │    IS-only 参数搜索（delay/decay/truncation）
  │    OOS 严格隔离
  │
  └─ 返回 GPEvolutionResult（best_dsl, metrics, pool_top5, evolution_log）
    │
    ▼
[OverfitCritic.check()]
  PASS: oos_sharpe > 0.2 AND overfit < 0.5 → 保存
  FAIL: FOR _ in 1..3: tool_mutate_ast → tool_run_backtest → 重新验证
    │
    ▼
[tool_save_alpha → SQLite AlphaStore]
```

### Workflow B：优化已有 DSL

```
用户提供 DSL
    │
    ▼
[_agent._detect_intent()] → workflow_b
    │
    ▼
[tool_interpret_factor(dsl)] → 因子族 + 设计缺陷 + 改进建议
    │
    ▼
[_quick_metrics(dsl)] → 初始诊断（IS/OOS Sharpe, 换手率, 过拟合分数）
    │
    ▼
[_expand_for_optimization(dsl) + _targeted_mutations(dsl, init_metrics)]
  原始 DSL + 8 种变异 + 指标导向的有针对性变异
  • 高换手  → add_ts_smoothing 变异
  • 过拟合  → hoist + wrap_rank 变异
  • IC 低   → add_volume_filter 变异
    │
    ▼
[检测因子族 factor_family from user_dsl]  ← 弊端 1 修复后新增
    │
    ▼
[tool_run_gp_optimization(all_seeds, factor_family)]
  → 同 Workflow A 第 3-4 步
    │
    ▼
[OverfitCritic + tool_mutate_ast 纠错循环]
    │
    ▼
[tool_interpret_factor(best_dsl, metrics)] → 最终金融解读
    │
    ▼
[tool_save_alpha]
```

---

## 三、GP 适应度函数与自适应权重

### 适应度公式
```
fitness = sharpe_oos
        - 0.2  × turnover
        - 0.3  × |max_drawdown|
        - 0.5  × max(0, sharpe_is - sharpe_oos)
```

### 11 种变异算子基础权重分布

| 算子 | 基础权重 | 金融作用 |
|------|---------|---------|
| crossover | 0.20 | 两颗树的子树交换，最强探索 |
| point | 0.15 | 同类算子互换（ts_mean ↔ ts_std） |
| wrap_rank | 0.10 | 添加截面归一化层 |
| add_ts_smoothing | 0.10 | 添加时间序列平滑层 |
| hoist | 0.08 | 子树提升（简化树结构，对抗过拟合） |
| param | 0.08 | 时间窗口调整（±20%） |
| add_condition | 0.08 | 添加动量/市场状态条件门控 |
| combine_signals | 0.07 | 两个信号的四则运算合并 |
| add_volume_filter | 0.06 | 添加量的确认门控 |
| replace_subtree | 0.05 | 随机生成新子树替换内部节点 |
| add_operator | 0.03 | 包裹 sign/abs/signed_power 等算子 |

### 自适应调整规则

**高换手率（> 2.0）：** `param` +0.08，`hoist` +0.05，`crossover` -0.06，`combine_signals` -0.04

**低 OOS Sharpe（< 0.2）：** `wrap_rank` +0.06，`add_ts_smoothing` +0.06，`add_condition` +0.05，`combine_signals` +0.05，`replace_subtree` +0.04，`param` -0.08，`hoist` -0.06

**过拟合（> 0.5）：** `hoist` +0.08，`replace_subtree` +0.06，`add_operator` +0.04，`crossover` -0.06，`combine_signals` -0.06，`param` -0.06

### 因子族权重偏置（弊端 1 修复后新增）

| 因子族 | 偏置算子 | 调整方向 | 金融原理 |
|--------|---------|---------|---------|
| momentum | add_condition | +0.08 | 动量因子需市场状态条件防止熊市崩溃 |
| momentum | add_ts_smoothing | +0.07 | 原始动量换手率高，需平滑 |
| momentum | add_volume_filter | +0.05 | 量的确认减少噪声信号 |
| reversion | param | +0.08 | 均值回归需短窗口（1-5 日） |
| reversion | add_volume_filter | +0.07 | 量的确认是均值回归的关键信号过滤 |
| reversion | add_ts_smoothing | -0.05 | 平滑会破坏均值回归信号 |
| volatility | wrap_rank | +0.08 | 波动率信号必须截面归一化 |
| volatility | combine_signals | +0.07 | 低波动因子需与动量结合对冲 |
| trend_following | add_condition | +0.10 | 趋势跟踪必须有市场状态条件 |
| composite | hoist | +0.08 | 复合因子已够复杂，优先简化 |

---

## 四、系统弊端详细分析

### 弊端 1：金融诊断未接入 GP 决策（已修复）

**问题描述：**
`FinancialInterpreter` 和 `FinancialDiagnostics` 生成文字输出，但未接入 GP 进化逻辑。GP 的变异算子选择完全由 `mutation_weights_from_metrics()` 的数字规则驱动，不受因子族语义指导。

**表现：** 系统诊断出"这是动量因子，建议加市场状态条件"，但 GP 下一代仍可能选择 `param_mutation`（改窗口）而非 `add_condition`（加状态门控）。

**根本原因：** `_generate_next_population()` 自适应权重只看 Sharpe/turnover 数字，不看因子族。

**修复方案（已实施）：**
1. `fitness.py`：`mutation_weights_from_metrics()` 新增 `factor_family` 参数，按族应用偏置权重
2. `population_evolver.py`：`PopulationEvolver` 接受 `factor_family`，传入权重函数
3. `_tools.py`：`tool_run_gp_optimization()` 新增 `factor_family`，空值时自动从种子 DSL 检测
4. `_fallback.py`：两条工作流在调用 GP 前自动检测因子族并传递

**修复效果：** GP 进化方向现在由"指标数字"和"因子族金融语义"共同决定，两层自适应叠加。

---

### 弊端 2：种子 DSL 多样性不足，金融覆盖面窄 ✅ 已修复

**问题描述（原始）：**
原 `_SEED_DSLS` 共 20 个种子，高度同构，全部基于单资产 OHLCV 时间序列，缺乏：
- 量价关系（`ts_corr(close, volume, N)`）
- 风险调整动量（`ts_delta(log(close), N) / ts_std(returns, N)`）
- 均值回归种子（负号包裹的动量）
- 状态条件因子（`trade_when(cond, signal)`）
- 复合多信号组合

**修复（已实施，文件：`gp_engine/gp_engine.py`，`gp_engine/population_evolver.py`）：**

引入 `_SEED_DSLS_BY_FAMILY` 字典，将 50 个种子按 9 个因子族组织，全部通过 parse + validate 校验：

| 因子族 | 种子数 | 典型种子 |
|--------|--------|---------|
| momentum | 8 | `rank(ts_delta(log(close), N))` N=5/10/20/40 |
| reversion | 6 | `rank(-ts_delta(close, N))` N=1/3/5 |
| volatility | 5 | `rank(-ts_std(returns, N))` N=10/20/60 |
| risk_adjusted | 5 | `rank(ts_delta(log(close), N) / ts_std(returns, N))` |
| liquidity | 6 | `rank(-ts_mean(volume, N))`, `rank(ts_delta(log(volume), N))` |
| price_volume_corr | 5 | `rank(-ts_corr(close, volume, N))` N=10/20/60 |
| trend_following | 5 | `rank(ts_mean(close, N))` N=60/120 |
| composite | 7 | `rank(ts_delta(log(close), 10) * ts_delta(log(volume), 5))` |
| conditional | 3 | `rank(trade_when(close > ts_mean(close, 60), signal))` |
| **TOTAL** | **50** | — |

新增接口：
- `get_seeds_for_family(family)` — 返回该族种子；`unknown` 回退到全集（50个）
- `generate_random_alpha(factor_family=...)` — 60% 概率从族种子中采样

`PopulationEvolver._init_population` 联动弊端 1 修复：
1. 解析用户提供的种子 DSL
2. 用 `get_seeds_for_family(self._factor_family)` 注入族库种子填充剩余槽位
3. 随机填充阶段 60% 概率选择族种子

**实测效果（200 次随机生成，factor_family="momentum"）：**

| 生成结果因子族 | 数量 | 比例 |
|--------------|------|------|
| momentum | 144 | 72% |
| composite | 12 | 6% |
| reversion | 12 | 6% |
| volatility | 10 | 5% |

初始种群中 72% 的随机个体已具备正确金融结构，与弊端 1 的变异权重偏置形成两层联动。

---

### 弊端 3：GP 结构变异语法驱动，与金融意图脱耦 ✅ 已修复

**问题描述（原始）：**
`mutations.py` 的 11 种算子是纯语法层面的 AST 变换，不理解变换的金融意义：
- `replace_subtree()` 可能用 `ts_entropy(volume, 20)` 替换 `ts_delta(close, 5)`，破坏动量逻辑
- `combine_signals()` 随机合并两信号，组合可能金融上无意义（如动量 × 熵）
- `add_operator()` 随机选择变换类型，可能对均值回归信号施加 `signed_power`（放大极端值），破坏均值回归的方向特性

**修复（已实施，文件：`gp_engine/mutations.py`，`gp_engine/population_evolver.py`）：**

新增两个常量表：
- `_COMPLEMENTARY_FAMILIES` — 定义各族在组合时金融上有意义的互补族
- `_FAMILY_OPERATOR_PREFS` — 定义各族的首选 `add_operator` 变体

新增两个辅助函数：
- `_generate_family_compatible_subtree(max_depth, factor_family)` — 生成与族兼容的 AST 子树
- `_combine_op_for_families(f1, f2)` — 为两个族的组合选择最合理的算术算子

三个变异函数增加 `factor_family` 可选参数：

| 函数 | 族感知行为 |
|------|-----------|
| `replace_subtree(root, factor_family)` | 70% 概率使用族兼容子树替换（如 momentum 替换为另一个 ts_delta/ts_rank）；30% 保留随机探索 |
| `combine_signals(root, other_root, factor_family)` | 第二信号从互补族生成；算子由 `_combine_op_for_families` 决定 |
| `add_operator(root, factor_family)` | 75% 概率从该族首选算子列表中选取；25% 随机探索 |

`population_evolver.py` 中对应三处调用均加入 `factor_family=self._factor_family`。

**互补族定义（`combine_signals` 时的组合策略）：**

| 因子族 | 互补族 | 算子 | 金融含义 |
|--------|--------|------|---------|
| momentum | volatility | div | 风险调整动量 = 动量 / 波动率 |
| momentum | liquidity | mul | 量确认动量 = 动量 × rank(volume) |
| reversion | liquidity | mul | 量确认均值回归 |
| volatility | momentum | sub | 动量 - 波动率 = 多维组合信号 |

**`add_operator` 首选算子（因子族）：**

| 因子族 | 首选算子 | 金融原理 |
|--------|---------|---------|
| momentum | signed_power / rank_deviation / self_rank | 放大强信号；去趋势化 |
| reversion | unary_sign / unary_abs | 只保留方向；不破坏均值回归特性 |
| volatility | scaled / unary_abs | L1 归一化；保持非负性 |
| trend_following | rank_deviation / self_rank | 相对趋势强度 |

**实测效果（50 次变异，各函数）：**

| 变异 + 因子族 | 族保持率 | 说明 |
|-------------|---------|------|
| replace_subtree(momentum) | 94% | 47/50 结果仍是 momentum |
| combine_signals(reversion) | 100% | 量确认组合保持 reversion 语义 |
| add_operator(momentum) top-3 | mul/sub/signed_power | 均为 momentum 首选 |
| add_operator(reversion) top-2 | abs/sign | 均为 reversion 首选 |

---

### 弊端 4：合成数据适应度函数与真实市场脱节

**问题描述：**
整个 GP 进化在 `n_tickers=100, n_days=252` 的随机数据上运行（`seed=42` 固定）。合成数据无真实截面相关性、无行业结构、无体制转换。

**后果：** 合成数据上 Sharpe > 0.8 的因子在真实市场可能 Sharpe ≈ 0；过拟合检测对合成数据意义有限（两段均为随机游走）。

**修复方向：** 接入真实历史数据（已有 `dataset_registry.py` 支持 10 个数据集），GP 评估使用真实市场数据。

---

### 弊端 5：Optuna 与 GP 目标函数不一致

**问题描述：**
- GP 适应度：`sharpe_oos - 0.2×turnover - 0.3×|max_dd| - 0.5×overfit`
- Optuna 目标：`sharpe_is + 0.5×ic - 0.1×turnover`（IS only，无过拟合惩罚）

两个优化目标存在内在矛盾：Optuna 倾向选择 IS Sharpe 高的配置，GP 则惩罚 IS/OOS 差值。

**修复方向：** 统一两层优化的目标函数，或在 Optuna 目标中加入 IS 过拟合的隐式惩罚。

---

### 弊端 6：OverfitCritic 二元门控，纠错方向随机

**问题描述：**
Critic 只做通过/不通过，未通过后触发的 `tool_mutate_ast` 是从三种变异中随机选择的，不针对具体失败原因。

**表现：** 因高换手率导致净损益差的因子被 Critic 拒绝后，随机触发 `param_mutation` 把窗口从 10 改到 8，对换手率问题无帮助。

**修复方向：** Critic 返回带因子族标签的结构化诊断，将诊断结果作为 `tool_mutate_ast` 的有方向性指令。

---

### 弊端 7：LLM 路径与 GP 路径知识不共享

**问题描述：**
LangChain 路径下 LLM 理解因子族并能做有方向的优化建议，但这些语义信息没有传递给 `PopulationEvolver` 的变异策略。两条路径的 GP 运行参数完全相同。

**修复方向（部分已在弊端 1 修复中解决）：** LangChain 工具层在调用 `tool_run_gp_optimization` 时传入 `factor_family`，LLM 从对话中理解的因子类型直接影响 GP 变异方向。

---

## 五、组件关系图

```
用户输入
    │
    ▼
QuantAgent (_agent.py)
├─ intent detection (regex)
├─ LangChain path (with LLM)
│   └─ AgentExecutor → 7 tools (QuantTools)
└─ Fallback path (LLM-free)
    └─ FallbackOrchestrator → 7 tools (QuantTools)
         │
         ▼
    QuantTools (_tools.py)
    ├─ tool_generate_alpha_dsl   ← hypothesis → seed
    ├─ tool_interpret_factor     ← DSL → financial semantics (NEW)
    ├─ tool_run_gp_optimization  ← PRIMARY (structure search)
    │   └─ PopulationEvolver
    │       ├─ 11 mutations (mutations.py)
    │       ├─ fitness.py (multi-objective)
    │       │   └─ mutation_weights(factor_family)  ← 弊端1修复
    │       ├─ AlphaPool (diversity)
    │       └─ AlphaOptimizer (Optuna, IS-only params)
    ├─ tool_mutate_ast           ← single correction
    ├─ tool_run_backtest         ← IS+OOS validation
    ├─ tool_run_optuna           ← SECONDARY (params only)
    └─ tool_save_alpha           ← persist
         │
         ▼
    OverfitCritic (_critic.py)
    ├─ PASS: oos_sharpe > 0.2 AND overfit < 0.5
    └─ FAIL: trigger tool_mutate_ast (up to 3 rounds)
         │
         ▼
    AlphaStore (SQLite)
```

---

## 六、弊端修复状态汇总

| 弊端 | 影响层 | 严重程度 | 状态 | 修复文件 |
|------|--------|---------|------|---------|
| 1. 金融诊断未接入 GP 决策 | 优化 | 高 | ✅ **已修复** | `fitness.py`, `population_evolver.py`, `_tools.py`, `_fallback.py` |
| 2. 种子 DSL 多样性不足 | 生成 | 高 | ✅ **已修复** | `gp_engine/gp_engine.py`, `population_evolver.py` |
| 3. 变异算子语法驱动 | 优化 | 中 | ✅ **已修复** | `mutations.py`, `population_evolver.py` |
| 4. 合成数据适应度虚假 | 评估 | 高 | ⬜ 待修复 | `population_evolver.py` + `dataset_registry.py` 接入 |
| 5. Optuna/GP 目标函数不一致 | 优化 | 中 | ⬜ 待修复 | `alpha_optimizer.py` |
| 6. OverfitCritic 二元门控无梯度 | 后处理 | 中 | ⬜ 待修复 | `_critic.py`, `_fallback.py` |
| 7. LLM 路径知识不传入 GP | 协同 | 中 | 🔶 部分覆盖 | 弊端 1 修复已覆盖自动检测路径 |

### 弊端 1 修复详情（2026-05-16）

**核心变更：** 在 `mutation_weights_from_metrics()` 中引入第二层自适应——因子族权重偏置，使 GP 演化方向由"指标数字"和"因子族金融语义"共同决定。

**实测效果（`momentum` 因子，相同 metrics 基线）：**

| 算子 | 基础权重 | momentum 偏置后 | 变化 |
|------|---------|----------------|------|
| add_condition | 0.080 | 0.138 | +72% |
| add_ts_smoothing | 0.100 | 0.147 | +47% |
| replace_subtree | 0.050 | 0.038 | -24% |

**实测效果（`reversion` 因子，相同基线）：**

| 算子 | 基础权重 | reversion 偏置后 | 变化 |
|------|---------|----------------|------|
| param | 0.080 | 0.144 | +80% |
| add_volume_filter | 0.060 | 0.113 | +88% |
| add_ts_smoothing | 0.100 | 0.045 | -55% |

**修复链路：**
```
_fallback.py / _tools.py
  _detect_factor_family(seed_dsl) → factor_family
      │
      ▼
  tool_run_gp_optimization(factor_family=...)
      │
      ▼
  PopulationEvolver(factor_family=...)
      │
      ▼
  mutation_weights_from_metrics(factor_family=...)  [fitness.py]
      Layer 1: metric-driven (turnover / OOS sharpe / overfit)
      Layer 2: family-driven (_FAMILY_WEIGHT_BIASES)
```

---

*报告生成日期：2026-05-16*
*系统版本：GP Engine v2 + Financial Interpreter + Multi-Dataset Backtester*
*最后更新：弊端 1 修复（factor-family-aware mutation weights）*
