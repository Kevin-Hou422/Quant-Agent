# Quant Agent Backend — 优化开发路线图

**基于：** `AUDIT_REPORT.md`（2026-06-01 审计）  
**制定日期：** 2026-06-01  
**最后更新：** 2026-06-07 v4（Phase 2 全部任务完成）  
**目标：** 从当前综合评分，分阶段提升至可交易水平（≥ 66%）

---

## 完成状态总览（2026-06-07 核查）

| Phase | 目标评分 | 实际状态 | 说明 |
|-------|---------|---------|------|
| Phase 0 | 38% | ✅ **已完成** | 三个 P0 任务全部实现 |
| Phase 1 | 44% | ✅ **已完成** | 全部 5 个任务完成（含 AlphaEvolver 删除 + API 并发保护）|
| Phase 2 | 54% | ✅ **已完成** | 全部 4 个任务完成（WF框架 + Embargo + 并发加载 + 健康检查）|
| Phase 3 | 64% | ❌ 未开始 | |
| Phase 4 | 74% | ❌ 未开始 | |
| Phase 5 | 82% | ❌ 未开始 | |

---

## 总体规划

```
Phase 0 ── 接线与激活          ✅ 已完成
Phase 1 ── 清理与一致性         ✅ 已完成
Phase 2 ── 数据与验证升级        ✅ 已完成
Phase 3 ── 金融核心修复          ❌ 待开始
Phase 4 ── 风控与组合深化        ❌ 待开始
Phase 5 ── 在线化与生命周期       ❌ 待开始
```

---

## Phase 0：接线与激活 ✅ 已完成

> 审计确认三个 P0 任务均已在代码库中实现。

---

### Task 0.1 ✅ 将真实数据接入主工作流

**完成状态：已完成**

**已实现内容：**
- `app/main.py` 新增 `_load_dataset()` 函数，当 `--use-synthetic` 未设置时调用 `load_registry_dataset()`
- CLI 参数已添加：`--dataset`、`--start`、`--end`、`--use-synthetic`
- `app/config.py` 已添加：`default_dataset="us_tech_large"`、`default_start="2020-01-01"`、`default_end="2024-01-01"`
- `app/api/router.py` 中 `_resolve_dataset()` 函数同时支持 API 路径的真实数据加载
- 所有 5 个 CLI 模式（agent/gp/backtest/report/realistic）均接入真实数据

---

### Task 0.2 ✅ 修复假行业分组（`groups` 字段）

**完成状态：已完成**

**已实现内容：**
- `app/core/alpha_engine/dsl_executor.py` 中 `_add_derived()` 已删除 `np.arange(N) % 10` 假分组
- 修改为：`if "groups" not in aligned and "sector" in aligned: aligned["groups"] = aligned["sector"]`
- 新建 `app/core/data_engine/sector_mapper.py`，实现 GICS L1 静态映射 + 动态查询
- `load_registry_dataset()` 默认附加真实 sector 字段（`with_sector=True`）
- 无 sector 时 group_rank 等算子退化为全截面操作（正确行为）

---

### Task 0.3 ✅ 修复 gp_engine.py 中 fitness 不调用成本引擎的问题

**完成状态：已完成**

**已实现内容：**
- `app/core/gp_engine/fitness.py` 实现统一适应度公式：
  `fitness = sharpe_oos - 0.2*turnover - 0.3*|maxDD| - 0.5*max(0, sharpe_is - sharpe_oos)`
- `PopulationEvolver._evaluate_one_single()` 使用 RealisticBacktester 计算真实 Sharpe（含成本）
- 多数据集适应度聚合：`compute_multi_dataset_fitness()` 已实现
- 自适应变异权重：`mutation_weights_from_metrics()` 已实现

---

## Phase 1：清理与一致性 ⚠️ 部分完成

---

### Task 1.1 ✅ 删除两个冗余包

**完成状态：已完成（2026-06-07）**

**已完成内容：**
- `optimization_engine/` 中的模块已成为 `ml_engine/` 的干净 re-export（无重复逻辑）
- `portfolio_engine/` 的独立逻辑已归并到 `backtest_engine/`
- `AlphaEvolver` 类（约 235 行）已从 `gp_engine/gp_engine.py` 删除
- 随之清理了 6 个不再使用的 import（`multiprocessing`、`List`/`Optional`、mutations、`ProxyModel`、`AlphaStore`）
- `gp_engine/__init__.py` 中移除 `AlphaEvolver` 导出，保留 `GPAlphaResult`、`generate_random_alpha`

---

### Task 1.2 ✅ 统一 import 风格

**完成状态：已完成**

- 全量代码遵循 PEP 8 分组风格（stdlib → third-party → local）
- 统一使用 `from __future__ import annotations`
- 无通配符 import（除有 noqa 注释的 re-export 文件外）
- 类型注解使用 Python 3.9+ 现代风格（`dict[str, float]` 而非 `Dict`）

---

### Task 1.3 ✅ 退役旧版 AlphaEvolver，保留共用组件

**完成状态：已完成（2026-06-07）**

**已完成内容：**
- `AlphaEvolver` 类已从 `gp_engine.py` 中删除（见 Task 1.1）
- `gp_engine.py` 文档字符串已更新，明确标注退役日期并指向 `PopulationEvolver`
- `__init__.py` 已移除 `AlphaEvolver` 导出
- 共用组件（`_SEED_DSLS`、`generate_random_alpha()`、`GPAlphaResult`）保留完好
- `router.py` 文档字符串更新：`/api/gp/evolve` 说明改为"PopulationEvolver GP 进化"

---

### Task 1.4 ✅ 合并重复的快速评估函数

**完成状态：已完成**

- `optimization_engine/alpha_evaluator.py` 已成为 `ml_engine/alpha_evaluator.py` 的干净 re-export
- 所有 IC-IR 快速评估逻辑集中在 `population_evolver._quick_metrics()` 一处

---

### Task 1.5 ✅ 加入 API 并发保护与超时限制

**完成状态：已完成（2026-06-07）**

**已实现内容：**
- `router.py` 顶部添加 `_gp_lock = Lock()`（模块级单例）
- `/api/gp/evolve` 端点使用 `_gp_lock.acquire(blocking=False)`：
  - 并发请求立即返回 `HTTP 429`（"GP 任务正在运行，请稍后重试"）
  - 超过 300 秒通过 `threading.Thread.join(timeout=300)` 判活，返回 `HTTP 408`
  - `finally` 块确保锁无论如何都会释放
- 实现方式：同步端点 + daemon 线程 + join 超时，无需将端点改为 async
- 前端已配套 Axios 拦截器处理 429/408 错误消息

---

## Phase 2：数据与验证升级 ✅ 已完成（2026-06-07）

---

### Task 2.1 ✅ Walk-Forward 多轮验证框架

**完成状态：已完成**

**已实现内容：**
- `data_partitioner.py` 新增 `WalkForwardPartitioner`：扩展窗口策略，`n_splits`/`embargo_days`/`min_train_days` 可配
- `WalkForwardFold` dataclass 描述单折元信息（is_start/end、oos_start/end、embargo_days）
- `realistic_backtester.py` 新增 `WalkForwardBacktester`：在所有折上运行 `RealisticBacktester` 并聚合
- `WalkForwardResult` dataclass：含各折明细 + `mean/std/min_oos_sharpe`、`pct_positive`、`mean_overfitting`
- `router.py` 新增 `POST /api/backtest/walk_forward` 端点（`WalkForwardRequest`/`WalkForwardResponse`）
- CLI `--walk-forward` / `--wf-splits` 参数，`run_realistic()` 分支执行 `WalkForwardBacktester`

---

### Task 2.2 ✅ Embargo Period

**完成状态：已完成**

**已实现内容：**
- `DataPartitioner.__init__()` 新增 `embargo_days: int = 20` 参数
- `partition()` 中 IS 末日到 OOS 首日之间跳过 `embargo_days` 个工作日（不参与训练也不参与验证）
- `PartitionedDataset.__slots__` 新增 `embargo_days` 只读字段
- `summary()` 输出中显示 Embargo 天数
- `main.py run_gp()` 和 `run_realistic()` 均传入 `embargo_days`（默认 20）
- CLI `--embargo-days` 参数新增（默认 20）

---

### Task 2.3 ✅ 多市场并发加载

**完成状态：已完成**

**已实现内容：**
- `dataset_registry.py` 新增 `load_multi_datasets(names, start, end, max_workers=4)`
- 使用 `concurrent.futures.ThreadPoolExecutor` 并发请求
- 加载前校验所有名称合法，失败时抛出 `RuntimeError` 携带详情
- 命中缓存的数据集跳过网络请求，整体耗时约等于最慢那个数据集

---

### Task 2.4 ✅ 数据健康检查集成

**完成状态：已完成**

**已实现内容：**
- `dataset_registry.py` 新增 `check_dataset_health(ds, min_score, warn_only)` 独立函数
- wide-format → long-format 转换后传给 `DataHealthChecker`
- `load_registry_dataset()` 新增 `health_check: bool = True` 参数；加载后自动调用，warn_only 模式不阻断加载
- 健康得分 < 0.7 时 WARNING 日志，包含 gaps/spikes 数量

---

## Phase 3：金融核心修复（6-8 周）

---

### Task 3.1 🔴 接入真实行业分类数据（已部分实现）

**现状更新：** `sector_mapper.py` 已新建并集成到 `load_registry_dataset()`，提供 GICS L1 静态映射。

**待完成：**
- 为 akshare（A股）和 ccxt（加密）提供行业分类（当前 sector=-1 的 ticker 需要覆盖）
- 增加 yfinance 动态查询路径（`info["sector"]` 作为静态映射的补充）

**估时：** 2 天（增量）

---

### Task 3.2 🔴 实现多 Alpha 联合组合构建

**问题：** AlphaPool 积累了多个低相关候选 Alpha，但没有将其合并成联合组合的机制。

**涉及文件（新建）：**
- `app/core/portfolio_engine/alpha_combiner.py`

```python
class AlphaCombiner:
    """支持 ic_weighted / equal_weight / pca / min_corr_opt 合并方法"""
    
    def combine(self, signals: dict[str, pd.DataFrame], method="ic_weighted") -> pd.DataFrame:
        """返回合并后的联合信号矩阵 (T×N)"""
        ...
    
    def optimize_weights(self, signals, returns, method="ic_weighted") -> dict[str, float]:
        """在 IS 数据上估计最优合并权重"""
        ...
```

**估时：** 5 天

---

### Task 3.3 🟡 市场 Beta 中性化与系统因子暴露跟踪

**涉及文件：**
- `app/core/backtest_engine/portfolio_constructor.py` — 新增 Beta 中性化
- `app/core/backtest_engine/risk_report.py` — 扩展因子暴露报告

**估时：** 3 天

---

### Task 3.4 🟡 DSL 扩展：真实行业暴露节点 + 动量衰减节点

**涉及文件：**
- `app/core/alpha_engine/operators.py` — 新增 `cs_sector_neutral`、`ts_momentum_decay`
- `app/core/alpha_engine/parser.py` — 扩展文法

**估时：** 3 天

---

### Task 3.5 🟢 AlphaPool 相关阈值调整与正交化

**涉及文件：**
- `app/core/gp_engine/alpha_pool.py` — 阈值从 0.90 降到 0.70，新增 PCA 正交化

**估时：** 2 天

---

## Phase 4：风控与组合深化（6-8 周）

---

### Task 4.1 🟡 简单市场 Regime 识别

**涉及文件（新建）：**
- `app/core/data_engine/regime_detector.py`

```python
class RegimeDetector:
    """基于趋势强度指标或 HMM 2-state 的市场状态识别"""
    Regime = Literal["bull", "bear", "sideways", "high_vol"]
    
    def fit(self, market_returns: pd.Series, method="trend") -> "RegimeDetector": ...
    def predict(self, dates: pd.DatetimeIndex) -> pd.Series: ...
    def regime_to_alpha_weights(self, regime, pool_top5) -> dict[str, float]: ...
```

**估时：** 5 天

---

### Task 4.2 🟡 均值-方差组合优化替代等权

**涉及文件：**
- `app/core/backtest_engine/portfolio_constructor.py` — 新增 `MVOPortfolio`

**估时：** 5 天

---

### Task 4.3 🟡 Deflated Sharpe Ratio 与多重检验校正

**涉及文件：**
- `app/core/backtest_engine/performance_analyzer.py` — 新增 `deflated_sharpe_ratio()`
- `app/core/backtest_engine/risk_report.py` — 新增 `deflated_sharpe_ratio` 字段

**估时：** 2 天

---

## Phase 5：在线化与生命周期（8-12 周）

---

### Task 5.1 🟡 因子在线监控与衰减检测

**涉及文件（新建）：**
- `app/core/monitor/alpha_monitor.py`
- `app/db/alpha_store.py` — 新增 `alpha_ic_history` 表

```python
class AlphaMonitor:
    """滚动 IC/IC-IR 监控，连续负 IC → 告警/再训练/停用"""
    
    def update(self, alpha_id, date, realized_ic, realized_return) -> MonitorStatus: ...
    def check_decay(self, alpha_id) -> DecayAlert | None: ...
    def get_dashboard(self) -> pd.DataFrame: ...
```

**估时：** 5 天

---

### Task 5.2 🟡 因子生命周期状态机

**涉及文件：**
- `app/db/alpha_store.py`、`app/db/alpha_lifecycle.py`（新建）

```python
class AlphaStatus(Enum):
    CANDIDATE → VALIDATED → PAPER → ACTIVE → DECAYING → RETIRED | SUPERSEDED
```

**估时：** 3 天

---

### Task 5.3 🟢 APScheduler 定时任务框架

**涉及文件（新建）：**
- `app/tasks/scheduler.py`
- 集成到 `app/main.py` FastAPI startup

**估时：** 3 天

---

### Task 5.4 🟢 API 端点扩展：因子管理仪表板

**新增端点：**
```
GET  /api/datasets                    ✅ 已实现（2026-06-07）
GET  /api/alphas/dashboard
GET  /api/alphas/{id}/ic_history
POST /api/alphas/{id}/retrain
PATCH /api/alphas/{id}/status
GET  /api/alphas/{id}/walk_forward
```

**估时：** 3 天（剩余 5 个端点）

---

## 附录 A：任务依赖图（更新版）

```
Phase 0 ── ✅ 全部完成
  0.1 真实数据接入
  0.2 修复假行业分组       ←── 0.1
  0.3 修复 fitness 一致性

Phase 1 ── ⚠️ 部分完成
  1.1 删除冗余代码        ⚠️ AlphaEvolver 待删
  1.2 统一 import 风格    ✅
  1.3 退役 AlphaEvolver   ⚠️ 功能已退役，代码待清理
  1.4 合并重复评估函数     ✅
  1.5 API 并发保护        ❌ 待实现

Phase 2 ──  ← 0.1（需真实数据）
  2.1 Walk-Forward 框架
  2.2 Embargo Period      ← 2.1
  2.3 多市场并发加载      ← 0.1
  2.4 数据健康检查集成    ← 0.1

Phase 3 ── ← 2.1
  3.1 行业分类数据（部分已实现）
  3.2 多 Alpha 联合组合   ← 2.1
  3.3 Beta 中性化         ← 3.1
  3.4 DSL 新节点          ← 3.1
  3.5 AlphaPool 正交化    ← 3.2

Phase 4 ── ← 3.2
  4.1 Regime 识别         ← 3.2
  4.2 MVO 组合优化        ← 3.3
  4.3 Deflated Sharpe     ← 2.1

Phase 5 ── ← 4.1
  5.1 因子在线监控        ← 4.1
  5.2 生命周期状态机      ← 5.1
  5.3 APScheduler 调度    ← 5.1 + 5.2
  5.4 API 端点扩展        ← 5.2（/datasets 已完成）
```

---

## 附录 B：各阶段完成后的预期指标改善（更新版）

| 完成阶段 | 综合评分 | 实际/预期 |
|----------|----------|---------|
| 初始状态 | 24% | 2026-06-01 审计基线 |
| Phase 0 | 38% | ✅ **已达成**（数据质量 28%→65%，信号可信度提升）|
| Phase 1 | 44% | ✅ **已达成**（全部 5 个任务完成，可维护性提升，API 并发保护上线）|
| Phase 2 | 54% | 防过拟合（38%→65%）、数据质量（65%→78%）|
| Phase 3 | 64% | Alpha 信号质量（25%→55%）、组合完整性（28%→55%）|
| Phase 4 | 74% | 风险模型（8%→65%）、组合完整性（55%→75%）|
| Phase 5 | 82% | 运营系统（5%→70%）、因子生命周期（5%→72%）|

---

## 附录 C：剩余任务估时汇总

> Phase 0 和大部分 Phase 1 已完成，以下仅列出剩余工作。

| Task | 描述 | 估时 | 难度 | 状态 |
|------|------|------|------|------|
| ~~1.1~~ | ~~删除 AlphaEvolver 类~~ | ~~0.5 天~~ | ~~低~~ | ✅ 已完成 |
| ~~1.3~~ | ~~添加弃用注释/移除导出~~ | ~~0.5 天~~ | ~~低~~ | ✅ 已完成 |
| ~~1.5~~ | ~~API 并发保护~~ | ~~1 天~~ | ~~低~~ | ✅ 已完成 |
| 2.1 | Walk-Forward 框架 | 4 天 | 中 | ❌ |
| 2.2 | Embargo Period | 1 天 | 低 | ❌ |
| 2.3 | 多市场并发加载 | 2 天 | 低 | ❌ |
| 2.4 | 数据健康检查集成 | 1 天 | 低 | ❌ |
| 3.1 (残) | A股/加密行业分类补全 | 2 天 | 中 | ⚠️ 部分 |
| 3.2 | 多 Alpha 联合组合 | 5 天 | 高 | ❌ |
| 3.3 | Beta 中性化 | 3 天 | 中 | ❌ |
| 3.4 | DSL 新节点扩展 | 3 天 | 中 | ❌ |
| 3.5 | AlphaPool 正交化 | 2 天 | 中 | ❌ |
| 4.1 | Regime 识别 | 5 天 | 高 | ❌ |
| 4.2 | MVO 组合优化 | 5 天 | 高 | ❌ |
| 4.3 | Deflated Sharpe | 2 天 | 中 | ❌ |
| 5.1 | 因子在线监控 | 5 天 | 高 | ❌ |
| 5.2 | 生命周期状态机 | 3 天 | 中 | ❌ |
| 5.3 | APScheduler 调度 | 3 天 | 中 | ❌ |
| 5.4 (残) | API 端点扩展（5个）| 3 天 | 低 | ⚠️ 部分 |
| **剩余总计** | | **~51 天** | | |

---

*路线图版本 v2.0 | 2026-06-07 | Phase 0 完成确认 | Phase 1 状态核查*
