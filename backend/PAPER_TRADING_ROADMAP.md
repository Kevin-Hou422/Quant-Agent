# Quant Agent — 从路线图完结到「可测试程度」完整规划

**制定日期：** 2026-07-17
**最后更新：** 2026-07-17 v1.1 —— Phase 5 已完成（DEV_ROADMAP v5.0 关闭），本规划从 Phase 6 起算
**前置状态：** DEV_ROADMAP Phase 0–5 **全部完成并验收**（342 后端 + 94 前端测试全绿）
**目标：** 系统达到 **Paper Trading Ready** —— 本地全自动运行、每日模拟交易、
工程与金融层面经得起验证（不含真金白银下单）

---

## 〇、「可测试程度」的验收定义

系统满足以下全部条件即视为达标：

| # | 验收标准 | 验证方式 |
|---|---------|---------|
| A1 | 每个交易日收盘后自动完成：拉数据 → 数据验收 → 信号计算 → 目标权重 → 模拟成交 → 持仓/PnL 入库，全程无人工干预 | 连续 10 个交易日无一天漏跑（日志佐证）|
| A2 | 任一环节失败时**跳过当日调仓并告警**，绝不静默降级到合成数据或过期数据 | 断网/坏数据注入测试 |
| A3 | 每个 PAPER 状态的 Alpha 有逐日 realized IC 记录与滚动 IC-IR 曲线 | 监控面板可视 + 数据库可查 |
| A4 | 任意一条历史记录（回测或模拟成交）可用「数据哈希 + 代码版本 + 种子 + 配置」完整重放，误差为零 | 重放脚本对比 |
| A5 | 回测引擎输出与手工计算的黄金基准精确一致 | golden-master 测试 |
| A6 | 进程重启后从数据库恢复持仓与调度状态，不重复、不遗漏 | kill -9 后重启验证 |
| A7 | 全部测试在 CI 中自动运行且通过（后端 + 前端 + lint + type check）| GitHub Actions 绿灯 |

---

## 一、阶段总览

```
Phase 5  ── 在线化与生命周期                          ✅ 已完成（2026-07-17）
Phase 6  ── 基础设施加固（工程严谨性 + 体检修复）      ~11 天
Phase 7  ── PaperBroker 与模拟交易循环（核心新增）    ~10 天
Phase 8  ── PIT 数据层与验证运营（持续运行）           ~5 天启动 + 长期
─────────────────────────────────────────────────────
剩余专注开发约 26 个工作日（5-6 周）
之后进入 ≥ 3 个月的 paper trading 验证期（系统自动运行，人工每周复盘）
```

依赖关系：Phase 6 的 6.1/6.2 应**最先做**（消除静默损坏源）；
Phase 7 依赖已完成的调度器与生命周期状态机；Phase 8 依赖 Phase 7。

---

## 二、Phase 5 — 在线化与生命周期 ✅ 已完成（2026-07-17）

> 全部验收标准达成，实现细节见 DEV_ROADMAP v5.0 Phase 5 章节。

| Task | 内容 | 验收结果 |
|------|------|---------|
| 5.2 | 生命周期状态机 | ✅ 10 项测试：全路径/非法跳转/幂等/终态冻结/历史兼容全部通过 |
| 5.1 | AlphaMonitor + `alpha_ic_history` 表 | ✅ 10 项测试：`update()` 幂等（UNIQUE(alpha_id,date) 覆盖式写入）、双衰减规则、数据不足不告警、dashboard 排除终态 |
| 5.3 | APScheduler 调度框架 | ✅ SQLAlchemyJobStore 持久化 + misfire_grace_time=3600 + 显式时区；启停幂等；daily_monitor_job 端到端可执行 |
| 5.4 | API 端点扩展（6 个）| ✅ 6 项集成测试；PATCH status 非法流转 409、未知状态 422 |
| B2  | 组合权重 IS 拟合修复（本规划风险表建议项）| ✅ `weights_fitted_on` 标注 + 回退路径测试 |

**验收记录**：test_phase5.py 31 项全绿；全量回归 342 passed + 3 skipped 零失败；
前端 94 项 + tsc 零错误；前端 FE-5.1/5.2/5.3（Live 视图）同步交付。

---

## 三、Phase 6 — 基础设施加固（~11 天，含 2026-07-20 体检新增的 6.6/6.7 正确性修复）

### 3.0 Phase 6 前彻底体检 — 新发现漏洞明细（2026-07-20）

> 对全代码库批判性精读，专找**前述所有报告（AUDIT/BACKTEST_AUDIT/DEV_ROADMAP/
> FRONTEND_AUDIT）均未记录**的问题。均已落为下方 Task。严重度：🔴高 🟡中 🟢低/潜在。

**金融正确性**

| ID | 严重 | 问题 | 后果 | 去向 |
|----|------|------|------|------|
| **F-N1** | 🔴 | **ADV 10% 参与率上限被 L1 重归一化撤销**（`transaction_cost.LiquidityConstraint.apply`：clip 到上限后除以 L1，把权重整体放大回 sum=1，capped 名被推回超限）| 实际持仓可超 ADV 限制 → 市场冲击与策略容量被系统性低估 | 6.6 |
| **F-N4** | 🟡潜在 | **RegimeDetector 全样本分位数前视**（`vol_cut = vol.quantile(0.80)` 用含未来的整段样本定高波动阈值）| 当前仅驱动展示徽章（无害）；`regime_to_alpha_weights` 一旦接入配置即成真实前视 | 6.7 |
| **F-N2** | 🟢 | **`compute_adv` 用 `bfill()` 回填早期 ADV** → 前 ~10 天流动性上限与滑点分母用未来成交量 | 早期小幅前视 | 6.7 |
| **F-N3** | 🟢 | **ADV 上限与冲击以 `initial_capital` 计**，非当日 equity | 净值漂移后参与率/冲击/容量口径偏差 | 6.7 |
| **F-N5** | 🟢 | **报告 IC 口径**：`result.signal = processed_signal`（延迟后），`mean_ic/ic_ir` 测的是延迟信号预测力，非标准因子 1 日 IC | 可辩护（=策略 IC），但不可与外部因子 IC 直接比较且未注明 | 6.7 |

**工程健壮性**

| ID | 严重 | 问题 | 后果 | 去向 |
|----|------|------|------|------|
| **E-N1** | 🔴 | **"clip→L1 renorm"撤销硬约束是系统性模式**（除 F-N1 外，`_build_weights` F11 的 `max_single_weight` 单票上限同样失效）| 集中度约束名存实亡 | 6.6 |
| **E-N2** | 🟡 | **SQLite 无 WAL/`busy_timeout` + 调度器为独立并发写者**（`daily_monitor_job` 自建 `AlphaStore` 与 API 线程池并发写同一文件）| Phase 5 调度器启用后可抛 `database is locked` | 6.2 |
| **E-N3** | 🟡 | **调度器与 API 的 DB 路径来源不一致**（`AlphaStore()` 默认 `sqlite:///alphas.db` vs `settings.database_url` = `sqlite:///./alphas.db`）| CWD 不同则读写不同物理库，监控与服务数据分叉 | 6.2 |
| **E-N4** | 🟡 | **测试以存在性为主**（约 60 处"非 None"断言 vs 12 处数值近似，**零手工黄金**）| 342 绿灯证明"不崩溃"而非"算得对"，掩盖 F-N1/E-N1 等数值缺陷 | 6.4/6.6 |

**体检确认正确、勿再重复怀疑的关键点**：执行延迟（SignalProcessor `shift(delay)` 是唯一
执行滞后，引擎 `prev_w` 是无前视的自然记账，delay=1 恰为 1 日滞后，**无双重延迟**）；
Walk-Forward/embargo（2026-06-09 修复后 OOS 不互叠、末折零浪费）；Deflated Sharpe 方差公式
（Lo/Mertens 高阶矩修正正确）；MVO 协方差窗口无前视；AlphaCombiner B1/B2 修复确已生效。

---

### Task 6.1 🔴 消除静默降级（B5 修复）— 1 天

**问题**：`_resolve_dataset()` 真实数据加载失败时静默回退合成数据，研究结论可能建立在噪声上。

**方案**：
- 新增 `DataSourceInfo` 字段贯穿所有响应：`{"source": "registry|synthetic", "loaded_at": ..., "fallback_reason": ...}`
- API 层：请求 `dataset_name` 非空但加载失败 → **默认返回 502**（新增 `allow_synthetic_fallback: bool = False` 请求参数供测试显式开启）
- 前端：响应含 `source: "synthetic"` 时 Console 显示醒目警告 + MetricsGrid 顶部黄条

**验收**：断网状态下请求真实数据集 → 502 + 明确错误信息；显式开启 fallback → 响应带 synthetic 标记且前端显示警告。

### Task 6.2 🔴 迁出 OneDrive + SQLite 加固 — 1 天（原 0.5 天 + 并发加固）

- 项目迁移到非同步目录（如 `C:\quant-agent\`）；OneDrive 仅存代码（git remote 更佳）
- `alphas.db` 开 WAL 模式（`PRAGMA journal_mode=WAL`）；每日调度任务内置 `VACUUM INTO` 备份（保留 7 份滚动）
- `.gitignore` 排除 `*.db` / parquet 缓存
- **并发写加固（体检新发现 E-N2/E-N3，见 §3.0 明细）**：
  - `AlphaStore` connect_args 增加 `busy_timeout`（如 5000ms）+ WAL —— 一旦 Phase 5 调度器
    启用，调度线程（`daily_monitor_job` 自建 AlphaStore）与 API 线程池会并发写同一 SQLite，
    无 WAL/busy_timeout 时会抛 `database is locked`
  - **统一 DB 路径来源**：`daily_monitor_job` 现调用无参 `AlphaStore()`（默认 `sqlite:///alphas.db`），
    与 API 的 `settings.database_url`（`sqlite:///./alphas.db`）不一定解析到同一文件 →
    调度器可能读写与 API 不同的库。改为全部经 `settings.database_url` / 依赖注入

**验收**：数据库文件不在任何云同步路径下；备份文件每日生成；调度器启用后并发写
压力测试（scheduler job + 并发 API 写）无 `database is locked`；调度器与 API 命中同一物理库。

### Task 6.3 🟡 依赖锁定 + CI 流水线 — 2 天

- `requirements.txt` 全部 pin 精确版本 + 生成 `requirements.lock`（pip-tools）
- 前端 `package-lock.json` 入库（若未入库）
- GitHub Actions：`pytest`（排除 performance 与两个 legacy broken 文件）+ `ruff check` + `mypy app/core`（渐进式，先 core）+ `vitest run` + `tsc --noEmit`
- 清理两个收集即报错的 legacy 测试文件（`test_alpha_discovery.py`、`test_data_engine_smoke.py`：修复 import 或删除）

**验收**：push 触发 CI 全绿；本地 `pip install -r requirements.lock` 可复现环境。

### Task 6.4 🔴 黄金基准测试（E5 核心补齐）— 3 天

手工构造微型数据集（5 天 × 3 只股票，价格为手算友好的整数），人工推导：
- 逐日权重（SignalWeighted 与 Decile 两种模式）
- 逐日毛收益 / 换手率 / 各成本分量（佣金、价差、冲击、借券）/ 净值曲线
- 最终 Sharpe、最大回撤、年化换手

写入 `tests/golden/test_golden_backtest.py`，断言到 `1e-10` 精度。同样为
`TransactionCostEngine.compute()` 和 `PerformanceAnalyzer` 各写 ≥ 3 个已知答案用例。

**必须覆盖的黄金用例（体检新发现 E-N4，见 §3.0 明细）**：当前 342 项后端测试中
约 60 处仅断言"非 None/存在性"、仅 12 处做数值近似比对、**零手工黄金**——即测试
证明的是"不崩溃"而非"算得对"，对 PnL/成本/Sharpe 的正确性给了虚假信心。黄金用例
必须包含：
- 成本三分量（佣金/价差/√冲击）在已知 participation 下的精确 bps
- **约束硬化验证**（配合 Task 6.6）：构造超 ADV 上限 / 超 max_single_weight 的信号，
  断言最终权重**不超过**上限（当前会因 L1 重归一化被推回超限）
- Deflated Sharpe 在已知 (mean, std, skew, kurt, n_trials) 下的解析值

**验收**：黄金测试通过；故意在引擎中注入 1bp 偏差时测试必然失败（灵敏度自检）。

### Task 6.5 🟡 可复现性清单 — 1.5 天（+0.5 天谱系扩展）

- 新增 `RunManifest`：每次回测/GP/模拟成交记录 `{data_sha256, git_commit, seed, config_json, timestamp}` 入库
- `scripts/replay.py`：给定 manifest id 重放并对比关键指标
- **审计谱系**（RESEARCH_OPERATING_MODEL §2 落地）：GP 每代每个个体的完整评估
  结果持久化到 `gp_lineage` 表（关联 manifest id），**不推 UI**；因子失效归因时
  可完整回放"它如何被选出"

**验收**：A4 标准——任选一条历史记录重放，指标误差为零；任选一个入池因子
可查询其完整进化谱系。

### Task 6.6 🔴 组合约束硬化 — 1 天（体检新发现 E-N1/F-N1）

**问题（系统性）**：多处采用"clip 到上限 → 再 L1 归一化"的模式，而 L1 归一化会
把权重整体放大回 sum=1，**撤销刚施加的硬上限**。受影响：
- `transaction_cost.LiquidityConstraint.apply()` —— ADV 10% 参与率上限被撤销 → 实际
  持仓可超 ADV 限制 → 市场冲击与容量被低估（金融正确性 F-N1）
- `realistic_backtester._build_weights()` F11 —— `max_single_weight` 单票上限同样失效
- MVO / beta_neutral 的 L1 归一化不违反硬上限（无逐名上限），但同属该模式，需一并审查

**方案**：将"clip→renorm"改为**迭代投影**（clip → renorm → 再 clip 直至收敛，或
水位线 water-filling 算法），使归一化后仍满足上限；上限不可满足时（如所有名都触顶）
显式告警而非静默放大。

**验收**：黄金用例（Task 6.4）断言归一化后无一权重超上限；随机压力测试 1000 次无违例。

### Task 6.7 🟡 成本与 Regime 的前视/口径修复 — 1 天（体检新发现 F-N2/F-N3/F-N4/F-N5）

- **F-N4 Regime 全样本分位数前视**（潜在高危）：`RegimeDetector` 的
  `vol_cut = vol.quantile(0.80)` 用**整个样本**（含未来）的波动率分布定阈值 → 每日
  regime 标签依赖未来信息。当前仅用于展示徽章（无害），但 **`regime_to_alpha_weights`
  一旦接入回测/实盘配置即成真实前视**。改为扩展窗口分位数（只用 t 及之前）；在接入
  任何配置逻辑前**必须**修复。
- **F-N2 ADV bfill 前视**：`compute_adv()` 对前 ~adv_window/2 天用 `bfill()` 回填 →
  早期 ADV（喂给流动性上限与滑点分母）使用未来成交量。改为 `ffill` 或该期不建仓。
- **F-N3 ADV/冲击以 initial_capital 计**：净值显著漂移后真实参与率与建模值背离，
  容量/冲击被错估。改用当日 equity（需在日循环内计算，接受与向量化的取舍）。
- **F-N5 IC 口径**：报告的 `mean_ic`/`ic_ir` 基于**延迟后**信号（`result.signal =
  processed_signal`），测的是"策略 IC"（延迟信号→前向收益），非标准因子 1 日 IC。
  可辩护但需在 RiskReport 文档/字段名注明口径，避免与外部因子 IC 直接比较。

**验收**：Regime 用扩展窗口分位数后，历史某日标签不随追加未来数据而改变；ADV 早期
无 bfill；IC 口径在报告中标注。

---

## 四、Phase 7 — PaperBroker 与模拟交易循环（~10 天）

> 系统从「研究平台」变为「每日自动运行的模拟交易系统」的核心阶段。
> 市场选择：**首选加密货币**（ccxt 已是依赖、24/7、无退市/行业分类问题），
> 美股次之（yfinance 日线，注意时区与交易日历）。A 股仅 paper，不规划实盘。

### Task 7.1 每日数据摄取任务 — 2 天

**新建：** `app/tasks/daily_ingest.py`（挂到 Phase 5.3 的调度器）

- 收盘后触发（按市场配置：crypto UTC 00:05 / 美股 ET 16:30 + 缓冲）
- 增量拉取当日 OHLCV → `check_dataset_health()` 验收门（gap/spike/NaN 超阈值 → 拒绝入库 + 告警）
- 通过验收的数据**追加写入 PIT 存储**（见 8.1），带 `as_of` 时间戳

**验收**：连续 5 日自动摄取成功；注入坏数据（人工改缓存）时当日被拒绝且告警可见。

### Task 7.2 PaperBroker + PositionStore — 4 天

**新建：** `app/core/execution/paper_broker.py`、`app/db/position_store.py`

```python
class PaperBroker:
    """T 日收盘信号 → T+1 日模拟成交（与回测 delay=1 语义严格一致）"""
    def submit_target_weights(self, alpha_id, date, weights: pd.Series) -> list[PaperFill]
    def get_positions(self, alpha_id) -> pd.Series          # 当前持仓权重
    def mark_to_market(self, alpha_id, date, prices) -> DailyPnL
```

- 成交价：T+1 开盘价（可配收盘价）；滑点/成本复用 `TransactionCostEngine`（**同一参数，禁止另立一套**）
- `PaperFill` 记录：目标权重、成交权重、成交价、成本分解、拒绝原因（如 ADV 超限截断）
- SQLite 表：`paper_positions`、`paper_fills`、`paper_daily_pnl`；全部写操作幂等（`(alpha_id, date)` 唯一键，重跑不重复记账）

**验收**：用历史数据回放 20 个交易日，PaperBroker 逐日累计 PnL 与
RealisticBacktester 同区间回测结果一致（容差 < 1bp/日，差异来源仅限成交价假设，需注释说明）。

### Task 7.3 每日交易循环编排 — 2 天

**新建：** `app/tasks/daily_trading_loop.py`

```
数据摄取(7.1) → 验收门 → 对每个 PAPER/ACTIVE Alpha：
  信号计算(复用 Executor+SignalProcessor) → 目标权重(复用 PortfolioConstructor)
  → PaperBroker 提交 → AlphaMonitor.update(realized IC)(5.1)
  → 衰减检查 → 状态机流转(5.2) → 日报写入
任一步失败 → 该 Alpha 当日跳过 + 告警；循环级失败 → 全局告警
```

- 崩溃恢复：循环开始前对比「上次成功日期」，补跑缺失日（数据允许时）或标记 gap
- 告警通道：本地日志 + Windows toast / 邮件（可配，最简 SMTP）

**验收**：A1/A2/A6 标准；人为 kill 进程后重启，次日循环正确补齐且无重复记账。

### Task 7.4 前端 Paper Trading 仪表板 — 2 天

- GlobalSidebar 新增 "Live" 入口：PAPER/ACTIVE Alpha 卡片（当前持仓、昨日 PnL、累计净值、滚动 IC 迷你图）
- 衰减告警红色标记；数据摄取状态（最近成功时间 + 健康分）
- 复用 Phase 5.4 端点 + 新增 `GET /api/paper/{alpha_id}/pnl`

**验收**：仪表板数据与数据库一致；无 mock 数据。

---

## 五、Phase 8 — PIT 数据层与验证运营（~5 天启动 + 长期）

### Task 8.1 Point-in-Time 数据存储 — 3 天

- FeatureStore 扩展：每日摄取的数据按 `(field, date, as_of)` 追加，**历史分区只追加不修改**
- 从启动日起自然积累无幸存者偏差的自有数据（时间越久价值越大——这是 F3 的长期解法）
- 提供 `load_pit(field, start, end, as_of)` 接口；回测默认 `as_of=None`（最新），验证时可固定 as_of 复现历史视角

**验收**：修改今日数据不影响昨日 as_of 查询结果；A4 重放用 PIT 数据。

### Task 8.2 成本模型校准回路 — 1 天

- 每月任务：对比 PaperFill 假设成交价 vs 实际 T+1 OHLC 区间，输出冲击系数校准建议报告
- 人工确认后更新 `CostParams`（不自动改，写入变更日志）

### Task 8.3 验证期运营规则（制度，非代码）— 1 天文档

写入 `OPERATIONS.md`：
- 任何因子进入 PAPER 前必须：WalkForward ≥ 5 折全正 + DSR > 0.90 + 真实数据集回测
- **红队审计**（RESEARCH_OPERATING_MODEL §4.2 落地）：每个进入 PAPER 的候选
  附一份"反方报告"（泄漏/容量/拥挤度/经济逻辑质疑）存入谱系；可用独立 prompt
  的 LLM 审查步骤实现，不必是常驻 agent
- PAPER → ACTIVE 门槛：**≥ 60 个交易日** realized IC 均值 > 0 且 t-stat > 2
- 每周人工复盘 checklist；每月全量回归 + 依赖升级窗口
- 明确「本系统不接入真实资金」的边界；若未来接入，需另立执行/风控/密钥管理规划（不在本报告范围）

---

## 六、里程碑时间线（以 2026-07-20 开工估算）

| 里程碑 | 内容 | 预计完成（2026-07-20 开工估算）|
|--------|------|---------|
| ~~M0~~ | ~~Phase 5 全部 + 验收~~ | ✅ 已完成（2026-07-17，提前达成）|
| M1 | Phase 6.1 + 6.2（静默降级修复 + 迁出 OneDrive）| 第 1 周内 |
| M2 | Phase 6 剩余（CI + 黄金测试 + 可复现性）| 第 2 周 |
| M3 | Phase 7 全部 + 20 日历史回放对账通过 | 第 4 周 |
| M4 | Phase 8 启动，**系统进入每日自动 paper trading** | 第 5 周 |
| M5 | 首批因子完成 60 交易日 PAPER 验证，出具首份验证报告 | M4 + 3 个月 |

---

## 七、风险与已知未解决项

| 风险 | 影响 | 缓解 |
|------|------|------|
| yfinance/akshare 免费源限流或字段变更 | 每日摄取失败 | 验收门保证不产生坏数据；准备 ccxt（crypto）为主市场降低依赖 |
| 幸存者偏差（F3）在历史回测中仍存在 | 历史回测收益虚高 | PIT 存储从启动日起消除**增量**偏差；历史段结论仅作参考，决策以 paper 期 realized IC 为准 |
| ~~B2（组合权重 OOS 拟合）~~ | ~~combined_metrics 乐观偏差~~ | ✅ 已随 Phase 5 修复（2026-07-17）：权重 IS 拟合 + `weights_fitted_on` 标注 |
| beta_neutral 数学不闭合（B6）| "市场中性"名不副实 | Phase 7 前修复或在 UI 中如实标注为"beta 缓和"（~1 天）|
| **ADV/单票上限被 L1 重归一化撤销（E-N1/F-N1）** | 硬约束失效 → 市场冲击与容量低估、集中度超限 | Task 6.6 迭代投影修复（Phase 6，paper 前必修）|
| **Regime 全样本分位数前视（F-N4）** | 接入配置即成真实前视 | 当前仅展示无害；Task 6.7 改扩展窗口分位数，接入配置前必修 |
| ADV bfill / initial_capital 计量（F-N2/F-N3）| 早期前视 + 容量口径偏差 | Task 6.7 修复（低幅度，Phase 6）|
| 测试以存在性为主、零黄金（E-N4）| 342 绿灯给"算得对"虚假信心 | Task 6.4 黄金基准 + 6.6 约束硬化断言 |
| 单机 Windows 环境（无冗余）| 停电/重启漏跑 | A6 崩溃恢复 + 补跑机制；paper 阶段可接受 |
| LLM 依赖（OpenAI key）| Agent 功能不可用 | 每日交易循环**不依赖 LLM**（纯确定性管道）；LLM 仅用于研究侧 |

---

## 八、明确不在本规划范围内（真实交易的额外前提）

以下属于「真正交易」的差距，待 M6 验证报告为正后另行规划：
券商/交易所下单接入（ccxt create_order / IBKR）、订单状态机与对账、
实盘硬风控（日亏熔断 / fat-finger / kill switch）、API 密钥安全管理、
FastAPI 鉴权、告警升级（PagerDuty 级）、合规与税务。

---

*规划版本 v1.2 | 2026-07-20 | Phase 6 前彻底体检：新增 Task 6.6（约束硬化）/6.7（前视口径修复），扩充 6.2/6.4，风险表新增 4 项 | 剩余 Phase 6-8 约 26 个工作日*
