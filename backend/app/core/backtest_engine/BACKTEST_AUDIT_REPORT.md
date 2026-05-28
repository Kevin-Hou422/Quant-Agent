# 回测引擎审计报告

> 审计范围：`app/core/backtest_engine/`（全部文件）、`app/core/alpha_engine/signal_processor.py`、`app/agent/_data_utils.py`
> 审计日期：2026-05-28

---

## 一、运行机制总览

### 1.1 端到端执行路径

```
用户 DSL
    │
    ▼
RealisticBacktester.run()
    ├─ DSL Parser → AST 执行 → raw signal (T×N DataFrame)
    ├─ SignalProcessor.process()
    │   ├─ Step A: Truncation  （行级 Winsorization，截断 5%–95% 分位）
    │   ├─ Step B: Decay       （ts_decay_linear 指数加权平滑）
    │   ├─ Step C: Neutralization（行业中性化，减去组内均值）
    │   └─ Step D: Delay       （信号整体 shift(delay=1)，模拟 T+1 执行）
    │
    ├─ Burn-in Trimming        （截断 time-series 算子预热期的全 NaN 行）
    │
    ├─ PortfolioConstructor
    │   ├─ DecilePortfolio    （前/后 10% 等权做多/做空）
    │   └─ SignalWeightedPortfolio（Z-score 归一化后 L1 归一化）
    │
    ├─ MarketNeutralizer       （行减均值 → 强制 Σw_t ≈ 0）
    │
    ├─ BacktestEngine.run()    （逐日 PnL 循环）
    │   ├─ ADV 流动性约束     （单仓 ≤ ADV 10%，超限裁切后重归一化）
    │   ├─ TransactionCostEngine（固定佣金 + 买卖价差 + √参与率冲击）
    │   └─ 逐日净值更新        equity_t = equity_{t-1} × (1 + net_ret_t)
    │
    └─ PerformanceAnalyzer → RiskReport
        ├─ 收益指标：annualized_return, vol, sharpe, calmar
        ├─ 风险指标：max_drawdown, sortino, VaR/CVaR
        └─ Alpha 质量：mean_ic, ic_ir, ann_turnover, decile_returns
```

### 1.2 IS/OOS 分割

- **方法**：按时间顺序固定比例切割（默认 70% IS / 30% OOS）
- **代码**：`DataPartitioner` → `_partition_dataset()` → `.train()` / `.test()`
- **多数据集验证**：`MultiDatasetBacktester` 在多个市场数据集上独立运行，聚合 OOS Sharpe（mean 或 min 模式）

### 1.3 交易成本建模

| 成本分量 | 模型 | 默认参数 |
|---------|------|---------|
| 固定佣金 | 单边 `fixed_bps = 5.0bps` | 5bps/单边 |
| 买卖价差 | 半价差 `spread_bps/2 = 1bps` | 1bps |
| 市场冲击 | 平方根模型 `σ × √(trade/ADV)` | impact_coef = 0.1 |
| 最小票面费 | `min_ticket_fee = $1/笔` | $1 |
| 仓位上限 | ADV 10% 日均成交额 | adv_cap_pct = 0.10 |

### 1.4 性能指标体系

| 指标类别 | 已实现指标 |
|---------|---------|
| 收益 | 年化收益率、年化波动率、Sharpe、Calmar |
| 风险 | 最大回撤（含起止日期与持续天数）、Sortino、VaR(95%)、CVaR(95%) |
| Alpha 质量 | 平均 IC、IC IR、年化换手率、成本拖累(bps)、十分组收益 |
| 时间序列 | 滚动 Sharpe（60日）、滚动 IC（20日）、回撤序列 |

---

## 二、回测合理性与准确性评估

### 2.1 整体评分

| 维度 | 评分（满分5） | 说明 |
|------|------------|------|
| IS/OOS 隔离 | 3.5 | 时间切割正确，但存在隐性泄露风险（见 §3） |
| 成本模型真实性 | 3.0 | 平方根冲击模型合理，但参数脱离现实 |
| 信号处理合规性 | 4.0 | 4步流水线符合行业标准 |
| 指标完整性 | 4.0 | 覆盖主要风险指标，缺少统计显著性检验 |
| 多市场稳健性 | 3.5 | MultiDatasetBacktester 架构完整，但聚合逻辑过于简化 |

### 2.2 已正确实现的部分

- **T+1 执行延迟**：`SignalProcessor` 末步 `df.shift(config.delay)` 强制模拟下一日执行，防止当日价格偷跑
- **平方根冲击模型**：`slippage = spread/2 + impact_coef × σ × √(trade_usd/adv_usd)` 符合 Almgren-Chriss 微观结构模型
- **L1 归一化**：组合权重强制 `|w|.sum() = 1`，确保多空账本平衡，不引入隐性杠杆
- **Winsorization**：行级截断（默认 5%–95%）有效减少极端信号对组合的扭曲
- **分母安全保护**：所有除法操作均有零分母防护，不会产生 Inf
- **组合预热截断**：自动识别并跳过 time-series 算子的 NaN 热身期，避免虚假平坦净值

---

## 三、异常防御机制

### 3.1 现有防御措施

| 防御点 | 实现位置 | 机制 |
|-------|---------|------|
| 全 NaN 行 | `SignalWeightedPortfolio:113` | `all_nan` 检测 → 置 0，不参与计算 |
| 零分母（Std/L1）| `signal_processor.py:121,129` | `np.where(==0, nan/1.0)` |
| 零价格除法 | `backtest_engine.py:167` | `np.where(price==0, nan)` |
| 分位数稳定 | `TruncationStep:137` | 仅对有限值行计算分位数 |
| 数据集错误隔离 | `multi_dataset_backtester.py:273` | `try/except` 捕获单数据集错误，不影响整体 |
| IC 样本不足 | `performance_analyzer.py:154` | 每日有效资产 < 5 则跳过 |
| 类型强制转换 | `multi_dataset_backtester.py:241` | 所有指标转 float，NaN → 0.0 |
| OOS 最小行数 | `realistic_backtester.py:216` | OOS ≤ 5 行则跳过 OOS 回测 |

### 3.2 防御机制的不足

1. **负净值无熔断**：`equity = equity * (1 + net_ret)` 无下限，允许净值无限穿零，不符合真实账户行为
2. **无输入验证层**：数据字段名、形状、时间顺序均未验证；接收错误格式数据后可能静默产生错误结果
3. **前向填充无上限**：`prices.ffill()` 无最大天数限制，停牌超过数月的股票会持续使用最后价格，产生虚假平滑回报
4. **异常捕获粒度过粗**：多数 `except Exception` 块仅记录 warning 并返回默认值，掩盖了真正的数据或逻辑错误

---

## 四、金融业务层面不足

### 4.1 前视偏差（Lookahead Bias）风险

**问题**：IS/OOS 分割依赖数据 DataFrame 的时间索引已升序排列，但无任何运行时验证。

```python
# _data_utils.py — 无序索引下 DataPartitioner 行为未定义
self._is_data, self._oos_data = _partition(ds, oos_ratio)
```

**隐患**：若外部数据源以乱序返回（如 akshare 在某些接口下），会导致 OOS 数据混入 IS，产生前视偏差却无任何警告。

**修复方向**：在 `_partition()` 前强制 `df.sort_index()` 并验证索引单调递增。

---

### 4.2 无风险利率默认为零

**问题**：`PerformanceAnalyzer.__init__(rf=0.0)` 默认无风险利率为零。

```python
sharpe = (annualized_return - self.rf * TRADING_DAYS) / vol
```

**影响**：当前利率环境（美国联邦基金利率 ~5%）下，实际 Sharpe 比报告值低约 `5% / vol`。一个年化收益 8%、波动率 15% 的策略：
- 当前系统报告 Sharpe：8/15 ≈ 0.53
- 真实 Sharpe（扣除 5% 无风险）：3/15 ≈ 0.20

**修复方向**：接入市场无风险利率（T-Bill 收益率或 SOFR），按回测时间段自动匹配历史无风险利率序列。

---

### 4.3 幸存者偏差

**问题**：Universe 在构建时固定，无股票进出处理。

**隐患**：
- 回测期间退市或破产的股票自动从数据中消失，不产生损失记录
- 若使用当前成分股回测历史，会纳入事后知晓"幸存下来"的股票
- 特别影响中小盘策略（退市率高）和 CN A 股策略（退市/ST 比例高）

**修复方向**：
1. 使用历史点位成分股数据（point-in-time universe）
2. 对退市股票产生 -100% 最终收益记录后退出组合

---

### 4.4 短卖成本未建模

**问题**：做空头寸（负权重）只承担与做多相同的交易成本，缺少借券成本建模。

**隐患**：
- 难借券股票的年化借贷成本可达 10%–50%+（如高空头兴趣的小盘股）
- 导致做空密集策略的实际回报被严重高估
- 对均值回归等高换手空头策略影响尤为显著

**修复方向**：引入 `short_borrow_rate` 参数（按股票或按策略的年化成本），做空持仓每日扣除 `|w_short| × borrow_rate / 252`。

---

### 4.5 IC 计算方法存在设计错误

**问题**：默认 IC 方法 `rolling_rank_ic()` 使用**持仓权重**作为未来收益代理：

```python
# performance_analyzer.py — 权重不等于收益
fwd_w = weights.shift(-1).fillna(0.0)
rho, _ = spearmanr(signal[t], fwd_w[t])
```

**问题本质**：权重 `w_t` 是当期信号的非线性变换（Z-score + L1 归一化），用 `w_{t+1}` 代理 `ret_{t+1}` 是错误的，导致 IC 被系统性低估或与持仓构建逻辑混淆。

**正确方法**：`rolling_rank_ic_from_prices()` 已正确实现 `spearman(signal_t, fwd_price_ret_{t+1})`，但不是默认调用路径。

**修复方向**：将 `RiskReport.from_result()` 中的 IC 计算切换至 `rolling_rank_ic_from_prices()`，或明确在报告中标注 IC 计算方法。

---

### 4.6 换手率计算被低估

**问题**：年化换手率 = `daily_mean_turnover × 252`，而日均换手率是**单边换手的历史均值**。

**隐患**：
- 初始建仓日（第一日）的换手率接近 100%（从零到全仓），被纳入均值会稀释后续真实换手率
- 对于低频调仓策略（如月度），用 252 倍化将产生过高的年化换手率估计
- 更严谨的方式是按实际调仓频率计算（月换手率 × 12，日换手率 × 252）

---

### 4.7 无基准对比分析

**问题**：所有性能指标均为绝对指标，无相对于基准的 alpha/beta 分解。

**影响**：
- 一个 Sharpe 为 0.8 的策略可能 90% 来自市场 beta 暴露，alpha 极低
- 无法区分"聪明的选股因子"与"带杠杆的指数基金"
- 在大牛市回测期间，几乎任何多头偏向策略都会呈现高 Sharpe

**修复方向**：引入 `BenchmarkComparator`，计算相对 benchmark 的：
- 超额收益（Alpha）
- 市场 Beta
- 跟踪误差（Tracking Error）
- 信息比率（Information Ratio = Alpha / TE）

---

### 4.8 无统计显著性检验

**问题**：Sharpe、IC 等指标仅为点估计，无置信区间。

**影响**：在 2 年 OOS 数据（约 500 个交易日）上，Sharpe = 0.5 的策略 95% 置信区间约为 [0.1, 0.9]——可能不显著异于零。IC = 0.03 在 500 个观测值下的 t 统计量约为 0.03 × √500 ≈ 0.67，远未达到统计显著。

**修复方向**：
```
Sharpe 标准误 ≈ √(1 + 0.5 × Sharpe²) / √T     （Lo 2002 公式）
IC t 统计量  = mean_IC / (std_IC / √T)
```
在 `RiskReport` 中加入 `sharpe_tstat`、`ic_tstat`、`sharpe_ci_95`。

---

### 4.9 多重比较问题（GP 路径）

**问题**：GP 框架对同一 OOS 数据集测试 12+ 个种子×4 代×每代 12 个个体，共计约 500+ 次 OOS 评估。

**影响**：即使每次评估的 OOS 显著性阈值为 5%，多重比较后真实误报率远高于 5%（Bonferroni 矫正后阈值应为 0.05/500 ≈ 0.0001）。最终报告的"OOS Sharpe"实质上已部分拟合 OOS 集合，应称为"validated Sharpe"而非真正 OOS。

**修复方向**：引入 **Paper Trading Set**（继 IS 和 OOS 之后的第三段时间）作为最终验证，GP 搜索过程中完全不接触该段数据。

---

### 4.10 日历效应与特殊事件未处理

**问题**：回测引擎对以下情形无特殊处理：
- 除权除息日（价格跳空不代表真实损益）
- 财报窗口期（信息不对称最强的时段）
- 节假日/提前收盘（流动性异常日）
- 股票分拆/合并（价格序列不连续）
- A 股涨跌停（无法按目标权重执行）

---

### 4.11 组合集中度无约束

**问题**：除 ADV 上限外，无单一持仓、行业、因子暴露的约束。

**隐患**：Z-score 归一化 + L1 归一化仍可能产生极端集中的组合（如信号方差很小时，少数股票获得接近 ±100% 的权重），导致极高的个股特异性风险。

---

## 五、软件工程层面不足

### 5.1 BacktestEngine 日循环未向量化

**问题**：`BacktestEngine.run()` 的核心日循环为 Python `for t in range(T)` 循环（约 209 行），每日操作 NumPy 标量。

```python
for t in range(1, T):
    target_w = adj_weights[t]
    delta_w = target_w - prev_w          # 均为 NumPy 向量，但在 Python 层循环
    turnover_t = np.abs(delta_w).sum() / 2.0
    ...
    equity = equity * (1 + net_ret)      # 标量更新
```

**影响**：1000 天 × 50 只股票的回测在 Python 层有约 1000 次循环调用开销；完全向量化后可快 10–50 倍。

**修复方向**：将 PnL 计算重写为矩阵运算：
```python
# 向量化版本思路
price_chg = (prices[1:] / prices[:-1]) - 1          # (T-1, N)
gross_ret = (weights[:-1] * price_chg).sum(axis=1)  # (T-1,)
equity_curve = initial_capital * (1 + gross_ret).cumprod()
```

---

### 5.2 输入数据无验证层

**问题**：`RealisticBacktester`、`BacktestEngine`、`MultiDatasetBacktester` 均直接消费传入数据，无结构验证。

**缺失的验证**：
```python
# 应验证但未验证：
assert isinstance(data, dict), ...
assert "close" in data, "close field required"
assert data["close"].index.is_monotonic_increasing, "index must be sorted"
assert not data["close"].index.duplicated().any(), "duplicate dates found"
assert data["close"].shape == data["volume"].shape, "shape mismatch"
```

**影响**：格式错误的数据会在随机位置产生 NaN 或 KeyError，调试极其困难。

---

### 5.3 TRADING_DAYS 硬编码为 252

**问题**：`performance_analyzer.py` 中 `TRADING_DAYS = 252` 为模块级常量，全局使用。

**影响**：
- 国内 A 股实际交易日约 244 天/年（更多假日）
- 加密货币 365 天全年无休
- 固定 252 会使 A 股年化收益高估约 3%，加密货币策略 Sharpe 被低估约 17%

**修复方向**：通过数据的实际交易日历动态计算 `trading_days_per_year = actual_trading_days / years`。

---

### 5.4 QuantTools 实例非线程安全

**问题**：`QuantTools.__init__` 在实例化时生成合成数据集并存储为实例变量：

```python
self._is_data, self._oos_data = _partition(ds, oos_ratio)
```

**影响**：若多个 API 请求共享同一 `QuantTools` 实例（FastAPI 路由的常见模式），并发调用 `tool_run_gp_optimization` 时会产生竞态条件（不同请求可能修改共享状态）。

**修复方向**：确保每个请求创建独立的 `QuantTools` 实例，或改为无状态设计（每次调用时接收数据集参数）。

---

### 5.5 IC 计算模块过度重复

**问题**：`PerformanceAnalyzer` 实现了两种 IC 计算方法（`rolling_rank_ic` 和 `rolling_rank_ic_from_prices`），逻辑高度重合，且 `RiskReport.from_result()` 中根据数据可用性在两者之间切换的逻辑不透明。

```python
# risk_report.py:92 — 切换逻辑不明确
ic = pa.rolling_rank_ic_from_prices(prices, fwd_window=1)
if ic is None or ic.empty:
    ic = pa.rolling_rank_ic()   # 降级到近似方法，但不记录日志
```

**修复方向**：统一为一个 IC 计算方法，记录所用方法到报告元数据。

---

### 5.6 Visualizer 与核心引擎耦合

**问题**：`visualizer.py` 在 `backtest_engine/` 目录中与核心计算逻辑混放，且直接依赖 `RiskReport` 内部结构。

**影响**：引入 `plotly` 等前端依赖污染了后端核心计算包，在无 UI 的 worker/batch 环境中增加无用依赖。

**修复方向**：将 `visualizer.py` 移至 `app/api/` 或 `app/visualization/` 层。

---

### 5.7 缺少核心路径单元测试

**问题**：根据代码结构，回测引擎核心路径缺少单元测试覆盖：
- `BacktestEngine` 的 PnL 计算逻辑（无参考值对比）
- `PerformanceAnalyzer` 的 Sharpe/max_drawdown 计算（无已知结果验证）
- `TransactionCostEngine` 的成本模型精度
- IS/OOS 分割的时间顺序不变性

**风险**：无测试意味着重构时无法安全验证指标计算的正确性。

---

### 5.8 错误信息不携带上下文

**问题**：多处异常处理只记录异常类型，不携带足以定位问题的上下文：

```python
# multi_dataset_backtester.py:280
logger.warning("MultiDataset eval failed for dataset '%s': %s", name, exc)
# 没有记录：DSL 内容、数据时间范围、数据形状
```

**修复方向**：在 `logger.warning` 中附加 DSL 前 80 字符、数据集日期范围、数据形状等上下文信息。

---

## 六、其他类型不足/缺陷

### 6.1 无压力测试场景

**问题**：无针对特定历史压力期（2020 COVID 崩盘、2008 金融危机、2015 A 股熔断）的专项分析。整体 Sharpe 可能由少数极端期主导，但现有报告无法揭示这一点。

**修复方向**：引入 `StressTestAnalyzer`，提取回测期内最大月度回撤、最大连续亏损天数的子区间分析。

---

### 6.2 Alpha 衰减分析缺失

**问题**：IC 仅计算 `t+1` 前向预测能力。对于持仓超过一日的策略，`t+5`、`t+10`、`t+20` 的 IC 衰减曲线是判断持仓期合理性的关键指标，但完全缺失。

**修复方向**：在 `PerformanceAnalyzer` 中加入 `ic_decay_curve(horizons=[1,5,10,20,60])`，输出多时间跨度的 Spearman IC 序列。

---

### 6.3 容量分析缺失

**问题**：无法回答"该策略最多能承载多少 AUM"的问题。当前 ADV 约束仅影响单次持仓大小，但未汇总计算整体策略容量上限。

**修复方向**：计算 `strategy_capacity = min_stock_adv_daily × adv_cap_pct × turnover / 252`，输出至 `RiskReport`。

---

### 6.4 多空分离报告缺失

**问题**：`RiskReport` 仅报告组合整体 PnL，不区分多头腿和空头腿的贡献。

**影响**：无法判断 alpha 来源是否偏向多头（可能只是 beta 暴露）或空头（受益于借券成本低估）。

**修复方向**：将 `BacktestResult` 扩展为包含 `long_pnl`、`short_pnl`、`long_sharpe`、`short_sharpe`。

---

### 6.5 无跨市场标准化

**问题**：`MultiDatasetBacktester` 直接对不同市场（美股、A 股、加密货币）的 OOS Sharpe 取均值，未考虑：
- 各市场历史波动率差异（A 股波动率约为美股 1.5–2 倍）
- 各市场的样本长度差异（数据量少的市场权重应降低）
- 不同货币计价（未做汇率影响调整）

---

### 6.6 回报分布非正态性未检测

**问题**：Sharpe 比率基于正态分布假设。加密货币等资产的日收益偏度和峰度极高，Sharpe 会严重高估风险调整后收益。

**修复方向**：在 `RiskReport` 中加入 `skewness`、`excess_kurtosis` 及 Omega Ratio（不依赖正态假设的风险调整收益指标）。

---

## 七、不足汇总表

| 编号 | 分类 | 问题 | 严重程度 | 修复难度 |
|------|------|------|---------|---------|
| F1 | 金融 | 无风险利率默认为零，Sharpe 高估 | 高 | 低 |
| F2 | 金融 | 前视偏差风险：数据排序未验证 | 高 | 低 |
| F3 | 金融 | 幸存者偏差：无退市/破产处理 | 高 | 高 |
| F4 | 金融 | IC 默认使用权重代理收益（方法错误） | 高 | 低 |
| F5 | 金融 | 短卖借券成本未建模 | 中 | 中 |
| F6 | 金融 | 多重比较导致 OOS 失去意义 | 高 | 高 |
| F7 | 金融 | 无统计显著性检验（t 统计量 / 置信区间） | 中 | 低 |
| F8 | 金融 | 无基准 alpha/beta 分解 | 中 | 中 |
| F9 | 金融 | 换手率年化计算被初始建仓日稀释 | 低 | 低 |
| F10 | 金融 | 无日历/特殊事件处理（除权、停牌、涨跌停） | 中 | 高 |
| F11 | 金融 | 无组合集中度约束 | 中 | 低 |
| E1 | 软件工程 | BacktestEngine 日循环未向量化（性能瓶颈） | 中 | 中 |
| E2 | 软件工程 | 无输入数据验证层 | 高 | 低 |
| E3 | 软件工程 | 交易日数 252 硬编码，跨市场不适用 | 中 | 低 |
| E4 | 软件工程 | QuantTools 非线程安全 | 高 | 中 |
| E5 | 软件工程 | 缺少核心路径单元测试 | 高 | 高 |
| E6 | 软件工程 | 负净值无熔断机制 | 中 | 低 |
| E7 | 软件工程 | 前向填充无最大天数限制 | 中 | 低 |
| E8 | 软件工程 | 错误日志上下文不足，调试困难 | 低 | 低 |
| E9 | 软件工程 | Visualizer 混入计算引擎，污染依赖树 | 低 | 低 |
| O1 | 其他 | Alpha 衰减曲线缺失（仅 t+1 IC） | 中 | 低 |
| O2 | 其他 | 无压力测试子区间分析 | 中 | 中 |
| O3 | 其他 | 无策略容量估算 | 低 | 中 |
| O4 | 其他 | 多空腿贡献未分离 | 中 | 低 |
| O5 | 其他 | 跨市场 Sharpe 聚合未加权/标准化 | 中 | 低 |
| O6 | 其他 | 收益分布非正态性未检测（无偏度/峰度/Omega） | 低 | 低 |

---

## 八、优先修复建议

### 第一优先级（高严重程度 + 低修复难度）

1. **F2 — 数据时间顺序验证**：在 `_partition()` 前加一行 `assert df.index.is_monotonic_increasing`
2. **F1 — 无风险利率接入**：为 `PerformanceAnalyzer` 增加 `rf_series` 参数，支持历史国债利率输入
3. **F4 — IC 计算方法修正**：`RiskReport.from_result()` 优先使用 `rolling_rank_ic_from_prices()`，并在报告中标注方法名
4. **E2 — 输入验证**：封装 `_validate_dataset(data: dict)` 函数，在 `RealisticBacktester.run()` 入口调用
5. **E6 — 负净值熔断**：在日循环中加 `if equity <= 0: break`，记录 `ruin_date`

### 第二优先级（中严重程度 or 中修复难度）

6. **F7 — Sharpe t 统计量**：将 `sharpe_tstat = sharpe * sqrt(T) / sqrt(1 + 0.5 * sharpe²)` 加入 `RiskReport`
7. **F5 — 短卖借券成本**：在 `CostParams` 加 `short_borrow_annual_bps: float = 50`，做空持仓每日扣除
8. **E3 — 动态交易日计算**：从实际数据索引计算 `trading_days_per_year`
9. **O1 — IC 衰减曲线**：`PerformanceAnalyzer.ic_decay_curve(horizons=[1,5,10,20,60])` 新增方法
10. **O4 — 多空腿分离**：`BacktestEngine` 中分别记录 `long_gross_ret` 和 `short_gross_ret`

### 第三优先级（架构改进）

11. **E1 — PnL 计算向量化**：重写 `BacktestEngine` 日循环为矩阵运算
12. **E4 — QuantTools 线程安全**：检查 API 层实例化模式，确保 per-request 实例隔离
13. **F6 — Paper Trading Set**：引入第三段完全隔离数据用于最终验证
14. **F3 — 幸存者偏差修复**：接入历史退市数据，构建 point-in-time universe
15. **E5 — 单元测试**：为 `BacktestEngine`、`PerformanceAnalyzer`、`TransactionCostEngine` 各加 ≥5 个测试用例
