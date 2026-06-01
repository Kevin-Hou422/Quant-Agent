# Quant Agent Backend — 优化开发路线图

**基于：** `AUDIT_REPORT.md`（2026-06-01 审计）  
**制定日期：** 2026-06-01  
**目标：** 从当前 24% 综合评分，分阶段提升至可交易水平（≥ 66%）

---

## 总体规划

```
Phase 0 ── 接线与激活          （1-3 天）   当前得分 24% → 目标 38%
Phase 1 ── 清理与一致性         （1-2 周）   目标 44%
Phase 2 ── 数据与验证升级        （3-4 周）   目标 54%
Phase 3 ── 金融核心修复          （6-8 周）   目标 64%
Phase 4 ── 风控与组合深化        （6-8 周）   目标 74%
Phase 5 ── 在线化与生命周期       （8-12 周）  目标 82%
```

每个阶段包含若干**独立可并行**的小任务，同类修改归为同一任务。  
优先级标注：🔴 阻塞性（不修复则后续工作无意义）| 🟡 重要 | 🟢 增量改进

---

## Phase 0：接线与激活（1-3 天）

> 最高 ROI 阶段。不需要新功能，只需把已存在的正确代码连接起来。  
> 完成后：合成数据 → 真实数据，假行业 → 真实分组，系统基本可信。

---

### Task 0.1 🔴 将真实数据接入主工作流

**问题：** `main.py` 所有 CLI 模式（agent / gp / backtest / realistic）均调用 `_make_synthetic_dataset()`，而 `dataset_registry.py`（10个真实数据集）从未被调用。

**涉及文件：**
- `app/main.py` — 修改 5 个 `run_*()` 函数
- `app/config.py` — 新增数据集相关配置项

**改动内容：**

```python
# config.py — 新增
default_dataset: str = "us_tech_large"
default_start: str = "2020-01-01"
default_end: str = "2024-01-01"

# main.py — 替换 _make_synthetic_dataset() 调用
from app.core.data_engine.dataset_registry import load_registry_dataset

def _load_dataset(args) -> dict:
    """统一数据加载入口，替代 _make_synthetic_dataset()"""
    dataset_name = getattr(args, "dataset", settings.default_dataset)
    start = getattr(args, "start", settings.default_start)
    end   = getattr(args, "end",   settings.default_end)
    ds = load_registry_dataset(dataset_name, start=start, end=end)
    return ds.data

# 所有 run_agent / run_gp / run_backtest / run_realistic 函数
# 将 dataset = _make_synthetic_dataset(...) 替换为 dataset = _load_dataset(args)
```

**CLI 参数新增：**
```
--dataset     us_tech_large | us_financials | china_tech | crypto_major ...
--start       YYYY-MM-DD
--end         YYYY-MM-DD
```

**验收标准：** `python app/main.py --mode backtest --dataset us_tech_large --dsl "rank(ts_delta(log(close),5))"` 能正常输出真实数据的回测结果。

---

### Task 0.2 🔴 修复假行业分组（`groups` 字段）

**问题：** `dsl_executor.py:75-78` 中 `groups = np.arange(N) % 10`，所有依赖 `group_rank` / `group_zscore` / `ind_neutralize` 的 DSL 在做假的行业中性化，金融含义错误。

**涉及文件：**
- `app/core/alpha_engine/dsl_executor.py` — 删除假 groups 自动生成
- `app/core/data_engine/dataset_registry.py` — 新增行业映射
- `app/core/data_engine/multi_dataset.py` — 新增 `sector` 标准字段

**改动内容：**

```python
# dataset_registry.py — 为每个 DatasetSpec 新增行业映射
SECTOR_MAP_US_TECH = {
    "AAPL": "hardware",  "MSFT": "software", "NVDA": "semiconductor",
    "GOOGL": "internet", "META": "internet",  "AMZN": "ecommerce",
    "AMD": "semiconductor", "AVGO": "semiconductor", ...
}

# multi_dataset.py — _align_and_standardize() 接受可选 sector_map
# 生成 "sector" 字段：pd.DataFrame(sector codes, index=dates, columns=tickers)

# dsl_executor.py — _add_derived() 修改
def _add_derived(aligned):
    # 删除以下代码块：
    # if "groups" not in aligned and "close" in aligned:
    #     grp = np.tile(np.arange(N) % 10, ...)
    #     aligned["groups"] = pd.DataFrame(grp, ...)
    
    # 改为：只有当外部传入了真实 sector 字段时才设置 groups
    if "sector" in aligned and "groups" not in aligned:
        aligned["groups"] = aligned["sector"]
    # 没有 sector 时，group_rank 等算子退化为全截面（等同于 cs_rank）
    return aligned
```

**验收标准：** `group_rank(close, 'sector')` 在 us_tech_large 数据集上能按真实行业（hardware/software/semiconductor 等）分组排序。

---

### Task 0.3 🔴 修复 gp_engine.py 中 fitness 不调用成本引擎的问题

**问题：** `gp_engine._evaluate_individual()` 用 IC-IR 代理 Sharpe，完全不计算交易成本；而 `population_evolver._evaluate_one()` 使用 RealisticBacktester（含成本）。两条路径的评估口径不一致，`run_gp()` 模式得出的 HoF 无可信成本信息。

**涉及文件：**
- `app/core/gp_engine/gp_engine.py` — 替换 `_evaluate_individual()` 实现
- `app/main.py` — `run_gp()` 改用 `PopulationEvolver`

**改动内容：**

```python
# main.py run_gp() — 替换 AlphaEvolver 为 PopulationEvolver
def run_gp(args):
    from app.core.data_engine.dataset_registry import load_registry_dataset
    from app.core.data_engine.data_partitioner import DataPartitioner
    from app.core.gp_engine.population_evolver import PopulationEvolver

    ds = load_registry_dataset(args.dataset, start=args.start, end=args.end)
    dataset = ds.data
    
    # IS/OOS 分区
    dates = next(iter(dataset.values())).index
    dp = DataPartitioner(str(dates[0].date()), str(dates[-1].date()), oos_ratio=0.3)
    parts = dp.partition(dataset)
    
    evolver = PopulationEvolver(
        is_data=parts.train(), oos_data=parts.test(),
        pop_size=args.pop_size, n_generations=args.generations,
    )
    result = evolver.run(n_optuna_trials=5)
    # 输出 result.best_dsl, result.metrics, result.pool_top5
```

**验收标准：** `run_gp()` 输出的 fitness 与 `run_realistic()` 的 Sharpe 口径一致（均含成本）。

---

## Phase 1：清理与一致性（1-2 周）

> 删除冗余包，统一代码风格，修复工程一致性问题。  
> 不新增功能，只减少维护负担和潜在的歧义。

---

### Task 1.1 🟡 删除两个冗余包

**问题：** `portfolio_engine/` 和 `optimization_engine/` 各含 3 个与其他模块重复的文件，增加维护成本。

**涉及文件（删除）：**
```
app/core/portfolio_engine/signal_processor.py       → 由 alpha_engine/signal_processor.py 覆盖
app/core/portfolio_engine/realistic_backtester.py   → 由 backtest_engine/realistic_backtester.py 覆盖
app/core/portfolio_engine/portfolio_constructor.py  → 由 backtest_engine/portfolio_constructor.py 覆盖
app/core/portfolio_engine/__init__.py

app/core/optimization_engine/data_partitioner.py   → 由 data_engine/data_partitioner.py 覆盖
app/core/optimization_engine/alpha_optimizer.py    → 由 ml_engine/alpha_optimizer.py 覆盖
app/core/optimization_engine/alpha_evaluator.py    → 由 ml_engine/alpha_evaluator.py 覆盖
app/core/optimization_engine/__init__.py
```

**操作步骤：**
1. `grep -r "portfolio_engine\|optimization_engine" backend/app --include="*.py"` 确认没有外部引用
2. 删除对应文件夹
3. 如有测试引用，将测试 import 路径更新为正确模块

**验收标准：** 删除后 `pytest` 全量通过，无 ImportError。

---

### Task 1.2 🟡 统一 import 风格

**问题：** 部分文件用绝对 import（`from app.core...`），部分用相对 import（`from ..alpha_engine...`），在不同工作目录下运行时行为不一致。

**涉及文件（主要）：**
- `app/core/backtest_engine/realistic_backtester.py` — 使用绝对 import，改为相对
- `app/core/gp_engine/population_evolver.py` — 使用绝对 import，改为相对
- `app/core/workflows/alpha_workflows.py` — 使用绝对 import，改为相对

**规则（确定后统一执行）：**
```python
# 规则：core/ 内部模块之间使用相对 import
# 规则：agent/ api/ tasks/ 使用绝对 import（跨顶层包）

# 修改示例（population_evolver.py）
# 改前：
from app.core.gp_engine.mutations import hoist_mutation, ...
from app.core.alpha_engine.parser import Parser
# 改后：
from .mutations import hoist_mutation, ...
from ..alpha_engine.parser import Parser
```

**验收标准：** 从 `backend/` 目录和 `backend/app/` 目录分别运行测试，均通过。

---

### Task 1.3 🟡 退役旧版 AlphaEvolver，保留共用组件

**问题：** `gp_engine.py` 中 `AlphaEvolver` 是旧版 GP 引擎，与新版 `PopulationEvolver` 并行维护成本高。两者 fitness 定义不同会导致混淆。

**涉及文件：**
- `app/core/gp_engine/gp_engine.py` — 删除 `AlphaEvolver` 类和 `_evaluate_individual()` 函数
- 保留：`_SEED_DSLS_BY_FAMILY`、`_SEED_DSLS`、`generate_random_alpha()`、`get_seeds_for_family()`、`GPAlphaResult`（这些被 `population_evolver.py` 共用）

**操作：**
```python
# gp_engine.py — 删除以下内容
# class AlphaEvolver: ...（约 235行）
# def _evaluate_individual(...): ...（约 80行）

# 文件顶部添加弃用说明
"""
gp_engine.py — 种子库与随机Alpha生成工具

AlphaEvolver 已于 2026-06-01 退役，请使用：
    from app.core.gp_engine.population_evolver import PopulationEvolver

本文件保留：
  - _SEED_DSLS_BY_FAMILY / _SEED_DSLS / get_seeds_for_family()  — 种子库
  - generate_random_alpha()                                       — 随机生成
  - GPAlphaResult                                                 — 结果数据类
"""
```

**验收标准：** `from app.core.gp_engine.gp_engine import generate_random_alpha` 仍可用；`AlphaEvolver` 引用抛出 `AttributeError`。

---

### Task 1.4 🟡 合并重复的快速评估函数

**问题：** `alpha_agent._quick_eval()` 和 `population_evolver._quick_metrics()` 实现了相同的 IC-IR 快速评估逻辑，两套代码同步维护。

**涉及文件：**
- `app/core/gp_engine/population_evolver.py` — 将 `_quick_metrics()` 提取为模块级函数
- `app/agent/alpha_agent.py` — `_quick_eval()` 改为调用公共函数

**操作：**
```python
# app/core/gp_engine/evaluation_utils.py（新建，约 40行）
def quick_ic_eval(dsl: str, dataset: dict) -> dict:
    """快速 IC-IR 评估，不调用 BacktestEngine，用于 GP 快速筛选。"""
    ...

# alpha_agent.py 中
from app.core.gp_engine.evaluation_utils import quick_ic_eval
# 删除 _quick_eval() 函数，改为调用 quick_ic_eval()
```

**验收标准：** `_quick_eval` 和 `_quick_metrics` 的相同输入产生相同输出。

---

### Task 1.5 🟢 加入 API 并发保护与超时限制

**问题：** GP 进化是耗时任务（每代 × 每个体 × IS+OOS 回测），API 无任何并发控制，可被重复触发导致 OOM。

**涉及文件：**
- `app/api/router.py` — 新增运行锁和超时
- `app/config.py` — 新增超时配置项

**改动内容：**
```python
# config.py
gp_max_concurrent: int = 1        # 最多同时运行1个GP任务
gp_timeout_seconds: int = 300     # 单次GP最大运行时间

# router.py
import asyncio
from threading import Lock

_gp_lock = Lock()

@router.post("/gp/run")
async def run_gp_endpoint(...):
    if not _gp_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="GP任务正在运行，请稍后重试")
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(run_gp_sync, ...),
            timeout=settings.gp_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="GP任务超时")
    finally:
        _gp_lock.release()
    return result
```

**验收标准：** 并发两次 POST `/gp/run` 时，第二次立即返回 429；单次运行超过 300s 时返回 408。

---

## Phase 2：数据与验证升级（3-4 周）

> 将验证体系从"单次固定切分"升级为"多轮滚动验证"，解决选择性偏差和标签泄漏问题。

---

### Task 2.1 🔴 实现 Walk-Forward 多轮验证框架

**问题：** 当前 IS/OOS 是单次固定切分（70/30），结论严重依赖切点位置，不具统计意义。

**涉及文件：**
- `app/core/data_engine/data_partitioner.py` — 新增 `WalkForwardPartitioner` 类
- `app/core/backtest_engine/realistic_backtester.py` — 支持多轮回测并聚合

**新增类设计：**
```python
# data_partitioner.py
class WalkForwardPartitioner:
    """
    滚动 Walk-Forward 分区器。
    
    参数
    ----
    n_splits    : 轮次数（推荐 5-10）
    train_ratio : 每轮 IS 占比（如 0.7）
    embargo_days: IS/OOS 切点间隔天数（推荐 20，防标签泄漏）
    
    示例（5轮，2年历史）
    --------------------
    轮1: IS=[2020-01, 2021-06]  embargo  OOS=[2021-08, 2021-12]
    轮2: IS=[2020-01, 2021-12]  embargo  OOS=[2022-02, 2022-06]
    轮3: IS=[2020-01, 2022-06]  embargo  OOS=[2022-08, 2022-12]
    轮4: IS=[2020-01, 2022-12]  embargo  OOS=[2023-02, 2023-06]
    轮5: IS=[2020-01, 2023-06]  embargo  OOS=[2023-08, 2023-12]
    """
    def __init__(self, n_splits=5, train_ratio=0.7, embargo_days=20): ...
    
    def split(self, dataset: dict) -> list[tuple[dict, dict]]:
        """返回 [(is_data, oos_data), ...] 共 n_splits 轮"""
        ...

class WalkForwardResult:
    """聚合多轮 Walk-Forward 结果"""
    per_fold: list[RiskReport]
    mean_sharpe_oos: float
    std_sharpe_oos: float
    min_sharpe_oos: float
    pct_positive_folds: float        # 正收益轮次占比
    ic_stability: float              # IC 序列的稳定性（std/mean）
    
    def summary(self) -> str: ...
```

**在 PopulationEvolver 中的集成：**
```python
# population_evolver.py — 新增 walk_forward 评估模式
class PopulationEvolver:
    def __init__(self, ..., walk_forward: bool = False, wf_n_splits: int = 5): ...
    
    def _evaluate_one(self, dsl, node):
        if self._walk_forward:
            return self._evaluate_walk_forward(dsl, node)
        return self._evaluate_one_single(dsl, node)   # 原路径保留
    
    def _evaluate_walk_forward(self, dsl, node):
        """多轮 IS+OOS 评估，fitness = mean(OOS Sharpe) - std(OOS Sharpe)"""
        ...
```

**验收标准：** 对同一 DSL 运行 5 轮 WF，能输出每轮 OOS Sharpe 和聚合统计；对比单次切分结论与 WF 均值，差距在可解释范围内。

---

### Task 2.2 🔴 加入 Embargo Period

**问题：** IS/OOS 切割点无间隔窗口，同一资产的相邻日期同时出现在 IS 末和 OOS 初，存在标签泄漏风险（尤其当信号有自相关时）。

**涉及文件：**
- `app/core/data_engine/data_partitioner.py` — `DataPartitioner` 新增 `embargo_days` 参数

**改动：**
```python
class DataPartitioner:
    def __init__(
        self,
        start: str,
        end: str,
        oos_ratio: float = 0.3,
        embargo_days: int = 20,   # 新增：切点后跳过 N 个交易日
    ): ...
    
    def partition(self, dataset: dict) -> PartitionedDataset:
        # 计算切分点
        split_idx = int(len(dates) * (1 - self.oos_ratio))
        split_date = dates[split_idx]
        
        # Embargo: OOS 从 split_date + embargo_days 开始
        oos_start_idx = split_idx + self.embargo_days
        oos_start_date = dates[min(oos_start_idx, len(dates)-1)]
        
        is_data  = {k: v.loc[v.index < split_date] for k, v in dataset.items()}
        oos_data = {k: v.loc[v.index >= oos_start_date] for k, v in dataset.items()}
        # IS 末 ~ OOS 初之间的 embargo_days 行数据被丢弃（不参与训练也不参与验证）
```

**验收标准：** IS 末日期 + 20 交易日 < OOS 首日期。两段数据无重叠且有间隔。

---

### Task 2.3 🟡 多市场并发加载与缓存优化

**问题：** `load_registry_dataset()` 每次调用都是单线程串行加载，10 个数据集全部加载需要数分钟（yfinance 网络请求）。

**涉及文件：**
- `app/core/data_engine/dataset_registry.py` — 新增并发批量加载
- `app/core/data_engine/feature_store.py` — 确保 Parquet 缓存正确命中

**改动：**
```python
# dataset_registry.py
import concurrent.futures

def load_multi_datasets(
    names: list[str],
    start: str,
    end: str,
    max_workers: int = 4,
) -> dict[str, Dataset]:
    """并发加载多个数据集，命中缓存则跳过网络请求。"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            name: ex.submit(load_registry_dataset, name, start, end)
            for name in names
        }
    return {name: fut.result() for name, fut in futures.items()}
```

**Parquet 缓存命名规范：**
```python
# feature_store.py — 确保 cache key 包含数据集名+日期范围
cache_path = f"{store_root}/{dataset_name}/{start}_{end}.parquet"
```

**验收标准：** 首次加载 5 个数据集后，第二次加载全部命中 Parquet 缓存，速度 < 5 秒。

---

### Task 2.4 🟢 数据健康检查集成到主工作流

**问题：** `DataManager` 中的 `DataHealthChecker` 已实现但不在 GP 主流程中使用，真实数据的质量问题（NaN 率过高、价格跳变）可能悄悄污染回测结果。

**涉及文件：**
- `app/core/data_engine/dataset_registry.py` — 在 `load_registry_dataset()` 后运行健康检查
- `app/core/gp_engine/population_evolver.py` — 在 `__init__` 中验证输入数据集健康

**改动：**
```python
# dataset_registry.py
from .health_report import DataHealthChecker

def load_registry_dataset(name, start, end, health_check=True):
    ...
    ds = Dataset(...)
    
    if health_check:
        checker = DataHealthChecker()
        # 将 wide-format 的 close DataFrame 转为 long-format 做检查
        report = checker.check(ds.data["close"].stack().reset_index(), ...)
        if report.overall_score < 0.7:
            logger.warning(
                "数据集 '%s' 健康得分 %.2f < 0.7，存在较多缺失/异常值",
                name, report.overall_score,
            )
    return ds
```

**验收标准：** 加载数据集时控制台输出健康得分；低于 0.7 时打印具体问题（哪只票 NaN 率最高）。

---

## Phase 3：金融核心修复（6-8 周）

> 解决系统最根本的金融逻辑缺陷：行业风险暴露、截面Alpha框架、多Alpha组合。

---

### Task 3.1 🔴 接入真实行业分类数据

**问题：** `groups` 字段使用假行业编号，所有行业中性化操作产生虚假结果。

**涉及文件（新建）：**
- `app/core/data_engine/sector_mapper.py` — 行业映射数据与查询接口

**设计：**
```python
# sector_mapper.py
class SectorMapper:
    """
    资产 → 行业代码映射。
    
    支持来源：
      1. 内置静态映射（GICS L1 for US stocks, 中证行业 for A股）
      2. yfinance 动态查询（info["sector"]）
      3. akshare 行业分类接口
    """
    
    # 内置 GICS L1 行业代码（数字，便于 DSL 中作为整数使用）
    _GICS_L1 = {
        "Information Technology": 0,
        "Financials": 1,
        "Healthcare": 2,
        "Energy": 3,
        "Consumer Discretionary": 4,
        "Consumer Staples": 5,
        "Industrials": 6,
        "Communication Services": 7,
        "Materials": 8,
        "Real Estate": 9,
        "Utilities": 10,
    }
    
    def get_sector_dataframe(
        self,
        tickers: list[str],
        dates: pd.DatetimeIndex,
        source: str = "static",
    ) -> pd.DataFrame:
        """
        返回 sector 矩阵 (T × N)，值为行业代码整数。
        所有行日期相同（行业分类不随日期变化，为静态矩阵）。
        """
        ...

# 在 dataset_registry._fetch_raw() 中集成
def _fetch_raw(spec, start, end):
    raw = ...  # 原有 OHLCV 获取
    mapper = SectorMapper()
    sector_df = mapper.get_sector_dataframe(spec.universe, dates=raw_dates)
    raw["sector"] = sector_df    # 加入标准字段
    return raw
```

**验收标准：** `ds.data["sector"]` 是整数矩阵；`group_rank(close, 'sector')` 在 us_tech_large 上返回按 GICS 行业内分位排名的信号。

---

### Task 3.2 🔴 实现多 Alpha 联合组合构建

**问题：** AlphaPool 积累了多个低相关候选 Alpha，但没有将其合并成联合组合的机制。这是 Alpha 研究流程最重要的缺失步骤之一。

**涉及文件（新建）：**
- `app/core/portfolio_engine/alpha_combiner.py` — 多 Alpha 合并器

**设计：**
```python
# alpha_combiner.py
class AlphaCombiner:
    """
    将 AlphaPool 中的多个候选 Alpha 合并成一个联合信号。
    
    支持合并方法：
      ic_weighted  : 以各 Alpha 的 IC-IR 作为权重的加权平均
      equal_weight : 等权平均（基线）
      pca          : 主成分提取（去除公共噪声）
      min_corr_opt : 最小化信号矩阵方差的权重优化
    """
    
    def combine(
        self,
        signals: dict[str, pd.DataFrame],     # {dsl: signal_matrix (T×N)}
        weights: dict[str, float] | None = None,  # None → ic_weighted
        method: str = "ic_weighted",
    ) -> pd.DataFrame:
        """返回合并后的联合信号矩阵 (T×N)"""
        ...
    
    def optimize_weights(
        self,
        signals: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        method: str = "ic_weighted",
    ) -> dict[str, float]:
        """在 IS 数据上估计最优合并权重"""
        ...

# 在 alpha_workflows.py GenerationWorkflow 中集成
class GenerationWorkflow:
    def run(self, hypothesis, dataset, ...):
        ...
        # 现有：返回最优单 Alpha
        # 新增：从 pool_top5 计算联合信号并评估
        if len(gp_result.pool_top5) >= 2:
            combiner = AlphaCombiner()
            joint_signal = combiner.combine(
                {e["dsl"]: self._compute_signal(e["dsl"], oos_data)
                 for e in gp_result.pool_top5[:5]}
            )
            joint_report = self._backtest_signal(joint_signal, oos_data)
            result.joint_signal_report = joint_report
```

**验收标准：** `WorkflowResult` 包含 `joint_signal_report`，其 OOS Sharpe 高于任意单 Alpha 的 OOS Sharpe（预期效果）。

---

### Task 3.3 🟡 市场 Beta 中性化与系统因子暴露跟踪

**问题：** `NeutralizationLayer.market_neutral()` 只约束权重求和为零，不能消除市场 Beta 暴露（持多空各 50% 的组合仍可有显著 Beta）。

**涉及文件：**
- `app/core/backtest_engine/portfolio_constructor.py` — 新增 Beta 中性化
- `app/core/backtest_engine/risk_report.py` — 扩展因子暴露报告

**新增功能：**
```python
# portfolio_constructor.py
class NeutralizationLayer:
    
    @staticmethod
    def market_neutral(weights: pd.DataFrame) -> pd.DataFrame:
        """现有：权重求和为零"""
        ...
    
    @staticmethod
    def beta_neutral(
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        window: int = 60,
    ) -> pd.DataFrame:
        """
        新增：Beta 中性化约束。
        计算各资产滚动 Beta，调整权重使组合 Beta ≈ 0。
        
        portfolio_beta = sum(w_i * beta_i) ≈ 0
        """
        # 滚动 OLS Beta 估计
        betas = returns.rolling(window).corr(market_returns).fillna(1.0)
        # 调整权重消除 Beta 暴露（凸二次规划或近似线性调整）
        ...
    
    @staticmethod
    def sector_neutral(
        weights: pd.DataFrame,
        sector_map: pd.DataFrame,      # (T×N) 行业代码矩阵
    ) -> pd.DataFrame:
        """行业中性化：约束每个行业的净敞口 ≈ 0"""
        ...

# risk_report.py — 新增因子暴露报告字段
@dataclass
class RiskReport:
    ...
    market_beta: float = np.nan          # 组合市场 Beta
    sector_exposures: dict = None        # {sector: net_exposure}
    factor_t_stats: dict = None          # 各因子暴露的 t 统计量
```

**验收标准：** Beta 中性化后组合 Beta < 0.05（之前可能 > 0.3）；`RiskReport.market_beta` 字段有值。

---

### Task 3.4 🟡 DSL 扩展：真实行业暴露节点 + 动量衰减节点

**问题：** DSL 当前只能表达技术量价信号，缺少金融结构性表达能力。

**涉及文件：**
- `app/core/alpha_engine/operators.py` — 新增算子实现
- `app/core/alpha_engine/typed_nodes.py` — 新增节点类型或扩展现有节点参数
- `app/core/alpha_engine/parser.py` — 扩展文法

**新增算子（第一批，不超过 5 个）：**

```python
# operators.py

def cs_sector_neutral(
    x: pd.DataFrame,
    sector: pd.DataFrame,     # (T×N) 行业代码矩阵
) -> pd.DataFrame:
    """
    行业内标准化后减去行业均值：真正的行业中性化。
    等价于 Barra 式行业哑变量回归残差（线性近似）。
    """
    result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for date in x.index:
        row = x.loc[date]
        sec_row = sector.loc[date] if date in sector.index else pd.Series(0, index=x.columns)
        for sec_code in sec_row.unique():
            mask = sec_row == sec_code
            cols = mask[mask].index.intersection(x.columns)
            if len(cols) > 1:
                result.loc[date, cols] = row[cols] - row[cols].mean()
    return result


def ts_momentum_decay(
    x: pd.DataFrame,
    window: int = 12,
    skip: int = 1,
) -> pd.DataFrame:
    """
    跳过最近 skip 个月的动量（skip=1 是学术标准，避免短期反转污染）。
    momentum = ts_mean(x, window, start=skip+1)（用 skip 到 window+skip 的平均）
    """
    return x.shift(skip).rolling(window, min_periods=max(1, window//2)).mean()
```

**验收标准：** `sector_neutral(close, 'sector')` 在 DSL 中可以解析和执行；`ts_momentum_decay(close, 12, 1)` 产生符合学术标准的动量信号。

---

### Task 3.5 🟢 AlphaPool 相关阈值调整与正交化

**问题：** 当前相关阈值 0.90 过高，导致 AlphaPool 接受了大量高度相关的因子；无正式正交化。

**涉及文件：**
- `app/core/gp_engine/alpha_pool.py` — 新增 PCA 正交化 + 调整默认阈值

**改动：**
```python
class AlphaPool:
    def __init__(
        self,
        max_size: int = 200,
        corr_threshold: float = 0.70,    # 从 0.90 降到 0.70
        orthogonalize: bool = False,     # 新增：是否做 PCA 正交化
    ): ...
    
    def get_orthogonal_signals(self) -> dict[str, np.ndarray]:
        """
        返回正交化后的信号矩阵（Gram-Schmidt / PCA）。
        用于 AlphaCombiner 时比原始信号相关性更低。
        """
        if len(self._entries) < 2:
            return {e.dsl: e.signal_vec for e in self._entries}
        
        mat = np.stack([e.signal_vec for e in self._entries if e.signal_vec is not None])
        # PCA 白化
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(len(self._entries), 10), whiten=True)
        orthogonalized = pca.fit_transform(mat.T).T  # (n_alphas, T)
        return {
            e.dsl: orthogonalized[i]
            for i, e in enumerate(self._entries)
            if e.signal_vec is not None
        }
```

**验收标准：** 相关阈值调整为 0.70 后，AlphaPool 中任意两个信号的相关系数 < 0.70。

---

## Phase 4：风控与组合深化（6-8 周）

> 建立正式风险模型，升级组合构建为均值-方差优化，加入 Regime 感知。

---

### Task 4.1 🟡 简单市场 Regime 识别

**问题：** 系统无 Regime 识别，动量/均值回归信号不随市场状态动态切换。

**涉及文件（新建）：**
- `app/core/data_engine/regime_detector.py`

**设计：**
```python
class RegimeDetector:
    """
    基于价格动态的市场状态识别。
    
    方法 1（基线）：趋势强度指标
      - 上升趋势：close > SMA200 且 SMA50 > SMA200
      - 下降趋势：close < SMA200 且 SMA50 < SMA200  
      - 震荡：|close - SMA200| / SMA200 < threshold
    
    方法 2（进阶）：HMM 2-state
      - 隐状态：低波动（牛市）/ 高波动（熊市/危机）
    """
    
    Regime = Literal["bull", "bear", "sideways", "high_vol"]
    
    def fit(self, market_returns: pd.Series, method: str = "trend") -> "RegimeDetector":
        ...
    
    def predict(self, dates: pd.DatetimeIndex) -> pd.Series:
        """返回每个日期的 Regime 标签"""
        ...
    
    def regime_to_alpha_weights(
        self,
        regime: str,
        pool_top5: list[dict],
    ) -> dict[str, float]:
        """
        根据当前 Regime 对 AlphaPool 中各因子分配权重。
        
        bull:       momentum 权重高（0.6），reversion 低（0.1）
        bear:       reversion 高（0.5），volatility 高（0.3）
        sideways:   reversion 高（0.4），volatility 中（0.3）
        high_vol:   均等权重，但整体降低仓位
        """
        ...
```

**与 GenerationWorkflow 的集成：**
```python
# alpha_workflows.py
class GenerationWorkflow:
    def run(self, hypothesis, dataset, ...):
        ...
        # 训练 Regime 检测器
        market_ret = dataset["returns"].mean(axis=1)
        regime_detector = RegimeDetector().fit(market_ret)
        current_regime = regime_detector.predict(oos_dates).iloc[-1]
        
        # 根据 Regime 选择合适的 Alpha 组合权重
        alpha_weights = regime_detector.regime_to_alpha_weights(
            current_regime, gp_result.pool_top5
        )
        result.regime_alpha_weights = alpha_weights
        result.current_regime = current_regime
```

**验收标准：** 对 2018-2024 年 US 市场数据，Regime 检测能识别出 2020Q1（bear）、2021（bull）、2022（bear）等主要市场状态变化。

---

### Task 4.2 🟡 均值-方差组合优化替代等权

**问题：** `DecilePortfolio` 和 `SignalWeightedPortfolio` 没有考虑资产间协方差，权重分配次优。

**涉及文件：**
- `app/core/backtest_engine/portfolio_constructor.py` — 新增 `MVOPortfolio`

**设计：**
```python
class MVOPortfolio:
    """
    均值-方差优化组合构建器（Markowitz 框架）。
    
    目标函数：max(signal' × w - λ × w' × Σ × w)
    约束：
      sum(|w|) = 1（杠杆约束）
      w'×1 = 0（市场中性）
      |w_i| ≤ max_weight（单资产上限）
      ||w - w_prev||_1 ≤ max_turnover（换手约束）
    """
    
    def __init__(
        self,
        risk_aversion: float = 1.0,
        max_weight: float = 0.10,
        max_turnover: float = 0.30,
        cov_window: int = 60,
        cov_method: str = "ledoit_wolf",   # 或 "sample"
    ): ...
    
    def construct(
        self,
        signal: pd.DataFrame,              # 预期收益代理
        prev_weights: pd.DataFrame = None, # 前一期权重（换手约束用）
    ) -> pd.DataFrame:
        """
        使用 scipy.optimize.minimize 或 cvxpy 求解 QP。
        """
        ...
```

**验收标准：** `MVOPortfolio` 生成的组合在相同 Sharpe 下，年化波动率比 `SignalWeightedPortfolio` 低 15-25%（理论预期）。

---

### Task 4.3 🟡 Deflated Sharpe Ratio 与多重检验校正

**问题：** GP 在 IS 上测试了数十个候选，OOS 的 Sharpe 并非真实预测能力，需要调整。

**涉及文件：**
- `app/core/backtest_engine/performance_analyzer.py` — 新增 DSR 计算
- `app/core/backtest_engine/risk_report.py` — 新增 DSR 字段

**新增计算：**
```python
# performance_analyzer.py
def deflated_sharpe_ratio(
    sharpe_ratio: float,
    n_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    n_obs: int = 252,
) -> float:
    """
    Lopez de Prado (2018) Deflated Sharpe Ratio。
    
    DSR = Φ(SR_hat - SR_0) / sqrt(1/T * (1 - skew*SR + (kurt-1)/4 * SR^2))
    
    其中 SR_0 = E[max(SR_1, ..., SR_n)] = 修正后的期望最大 Sharpe
    """
    import scipy.stats as stats
    
    # 期望最大 Sharpe（Bonferroni 近似）
    sr_0 = ((1 - np.euler_gamma) * stats.norm.ppf(1 - 1/n_trials) + 
            np.euler_gamma * stats.norm.ppf(1 - 1/(n_trials * np.e)))
    
    sigma_sr = np.sqrt((1 - skewness * sharpe_ratio + 
                        (kurtosis - 1) / 4 * sharpe_ratio**2) / (n_obs - 1))
    
    dsr = float(stats.norm.cdf((sharpe_ratio - sr_0) / sigma_sr))
    return dsr

# risk_report.py — 新增字段
@dataclass
class RiskReport:
    ...
    deflated_sharpe_ratio: float = np.nan     # DSR，> 0.95 才算显著
    n_trials_adjusted: int = 0                # 调整 DSR 时使用的试验次数
```

**验收标准：** 当 GP 测试了 50 个候选时，DSR 比原始 Sharpe t-stat 更保守；Sharpe=1.5 在 50 次试验下，DSR 约为 0.78（不显著）。

---

## Phase 5：在线化与生命周期（8-12 周）

> 将批处理研究系统升级为持续在线平台，实现因子从发现到退役的完整生命周期管理。

---

### Task 5.1 🟡 因子在线监控与衰减检测

**问题：** AlphaStore 只是数据库，无主动监控机制；上线后的 Alpha 没有 IC 追踪。

**涉及文件（新建）：**
- `app/core/monitor/alpha_monitor.py` — 因子监控引擎
- `app/db/alpha_store.py` — 新增 IC 历史表

**设计：**
```python
# alpha_monitor.py
class AlphaMonitor:
    """
    监控活跃 Alpha 的实时表现，检测衰减并触发再训练。
    
    指标跟踪：
      - 滚动 IC（20天窗口）
      - 滚动 IC-IR（60天窗口）
      - 连续 N 天 IC < 0 计数
    
    衰减判定规则（可配置）：
      - 滚动 IC-IR < 0.3（原始阈值的一半）持续 20 天 → 警告
      - 滚动 IC-IR < 0.1 持续 10 天 → 触发再训练
      - 净值回撤 > 30% → 立即停用
    """
    
    def update(
        self,
        alpha_id: int,
        date: pd.Timestamp,
        realized_ic: float,
        realized_return: float,
    ) -> MonitorStatus:
        """更新一条 Alpha 的当日表现记录"""
        ...
    
    def check_decay(self, alpha_id: int) -> DecayAlert | None:
        """检查是否触发衰减条件，返回 None 或 DecayAlert"""
        ...
    
    def get_dashboard(self) -> pd.DataFrame:
        """返回所有活跃 Alpha 的实时状态摘要"""
        ...

@dataclass
class DecayAlert:
    alpha_id: int
    dsl: str
    alert_type: Literal["warning", "retrain", "deactivate"]
    reason: str
    rolling_ic_ir: float
    consecutive_negative_days: int
```

**数据库扩展：**
```sql
-- alpha_store.db 新增表
CREATE TABLE alpha_ic_history (
    id          INTEGER PRIMARY KEY,
    alpha_id    INTEGER REFERENCES alpha_results(id),
    date        DATE NOT NULL,
    ic          REAL,
    ret         REAL,
    rolling_ic_ir REAL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_alpha_ic_date ON alpha_ic_history(alpha_id, date);
```

**验收标准：** 模拟 30 天 IC 历史后，对 IC 急剧下降的 Alpha 能正确触发 `DecayAlert(alert_type="retrain")`。

---

### Task 5.2 🟡 因子生命周期状态机

**问题：** AlphaStore 的 `status` 字段是字符串，没有状态流转规则。Alpha 从"研究"到"实盘"的过程没有版本控制。

**涉及文件：**
- `app/db/alpha_store.py` — 实现状态机
- `app/db/` — 新增 `alpha_lifecycle.py`

**设计：**
```python
# alpha_lifecycle.py
from enum import Enum

class AlphaStatus(Enum):
    CANDIDATE   = "candidate"    # GP 产出，尚未通过 WF 验证
    VALIDATED   = "validated"    # 通过 walk-forward 多轮验证
    PAPER       = "paper"        # 纸交易跟踪中
    ACTIVE      = "active"       # 实盘运行中
    DECAYING    = "decaying"     # 监控到衰减信号
    RETIRED     = "retired"      # 已退役
    SUPERSEDED  = "superseded"   # 被更新版本替代

class AlphaLifecycleManager:
    """
    管理 Alpha 的状态流转。
    
    合法流转:
      CANDIDATE → VALIDATED（通过 WF 验证后）
      VALIDATED → PAPER（手动或自动触发纸交易）
      PAPER → ACTIVE（纸交易达标后）
      ACTIVE → DECAYING（监控触发）
      DECAYING → RETIRED（进一步恶化）或 → ACTIVE（经再训练恢复）
      任意 → SUPERSEDED（被新版本替代）
    """
    
    def transition(
        self,
        alpha_id: int,
        target_status: AlphaStatus,
        reason: str,
        triggered_by: str = "system",
    ) -> bool:
        """执行状态流转，记录历史日志"""
        ...
    
    def get_active_alphas(self) -> list[AlphaResult]:
        """返回当前 ACTIVE 状态的所有 Alpha"""
        ...
```

**验收标准：** 非法状态流转（如 `CANDIDATE → ACTIVE`）抛出 `ValueError`；状态流转历史在 `alpha_transitions` 表中有记录。

---

### Task 5.3 🟢 APScheduler 定时任务框架

**问题：** 系统是纯批处理，无定时触发机制。因子监控、数据更新、再训练均需手动触发。

**涉及文件（新建）：**
- `app/tasks/scheduler.py` — 任务调度器
- `app/main.py` — FastAPI startup 时启动调度器

**设计：**
```python
# tasks/scheduler.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

def create_scheduler(app) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()
    
    # 每个交易日 21:00 更新数据并刷新 IC 监控
    scheduler.add_job(
        daily_data_update,
        CronTrigger(hour=21, minute=0, day_of_week="mon-fri"),
        id="daily_data_update",
        replace_existing=True,
    )
    
    # 每周一 22:00 对 DECAYING Alpha 触发再训练
    scheduler.add_job(
        weekly_retrain_check,
        CronTrigger(hour=22, minute=0, day_of_week="mon"),
        id="weekly_retrain",
        replace_existing=True,
    )
    
    return scheduler

async def daily_data_update():
    """更新数据缓存 → 计算当日 IC → 写入监控历史 → 检查衰减"""
    ...

async def weekly_retrain_check():
    """取出所有 DECAYING Alpha → 触发 OptimizationWorkflow → 更新状态"""
    ...
```

**验收标准：** FastAPI 启动后 `GET /health` 返回包含调度器状态；调度器在测试时能手动触发 `daily_data_update`。

---

### Task 5.4 🟢 API 端点扩展：因子管理仪表板

**问题：** 当前 API 缺少因子生命周期管理端点，前端无法展示 Alpha 状态全貌。

**涉及文件：**
- `app/api/router.py` — 新增端点

**新增端点：**
```python
# router.py

# 获取所有 Alpha 的实时状态摘要（仪表板数据）
GET /alphas/dashboard
→ {
    "active_count": 3,
    "decaying_count": 1,
    "candidates": [...],
    "last_updated": "2026-06-01T21:00:00"
  }

# 获取单个 Alpha 的 IC 历史
GET /alphas/{alpha_id}/ic_history?days=60
→ {"dates": [...], "ic": [...], "rolling_ic_ir": [...]}

# 触发指定 Alpha 的再训练
POST /alphas/{alpha_id}/retrain
→ {"task_id": "...", "status": "queued"}

# 手动变更 Alpha 状态
PATCH /alphas/{alpha_id}/status
Body: {"status": "retired", "reason": "人工决策"}
→ {"alpha_id": 1, "old_status": "decaying", "new_status": "retired"}

# 获取 Walk-Forward 验证结果
GET /alphas/{alpha_id}/walk_forward
→ {"n_folds": 5, "mean_sharpe": 0.87, "per_fold": [...]}
```

**验收标准：** `GET /alphas/dashboard` 在有 3 个 active Alpha 时返回正确数量；IC 历史端点数据与 `alpha_ic_history` 表内容一致。

---

## 附录 A：任务依赖图

```
Phase 0 ──────────────────────────────────────────────
  0.1 真实数据接入
  0.2 修复假行业分组       ←── 0.1（需要真实 ticker 列表）
  0.3 修复 fitness 一致性

Phase 1 ──────────────────────────────────────────────
  1.1 删除冗余包           ←── 0.3（先统一 fitness，再清包）
  1.2 统一 import 风格
  1.3 退役旧 AlphaEvolver  ←── 1.1
  1.4 合并重复评估函数      ←── 1.1
  1.5 API 并发保护

Phase 2 ──────────────────────────────────────────────
  2.1 Walk-Forward 框架    ←── 0.1（需要真实数据才有意义）
  2.2 Embargo Period       ←── 2.1（集成进 WF）
  2.3 多市场并发加载        ←── 0.1
  2.4 数据健康检查集成      ←── 0.1

Phase 3 ──────────────────────────────────────────────
  3.1 行业分类数据          ←── 0.2（接口已定义）
  3.2 多 Alpha 联合组合     ←── 2.1（需要 WF 验证的 OOS 信号）
  3.3 Beta 中性化          ←── 3.1（行业数据先行）
  3.4 DSL 新节点           ←── 3.1
  3.5 AlphaPool 正交化     ←── 3.2

Phase 4 ──────────────────────────────────────────────
  4.1 Regime 识别          ←── 3.2（与 Alpha 组合集成）
  4.2 MVO 组合优化         ←── 3.3（需要 Beta 估计）
  4.3 Deflated Sharpe      ←── 2.1（需要 WF 试验次数）

Phase 5 ──────────────────────────────────────────────
  5.1 因子在线监控         ←── 4.1（Regime 驱动监控策略）
  5.2 生命周期状态机       ←── 5.1
  5.3 APScheduler 调度    ←── 5.1 + 5.2
  5.4 API 端点扩展        ←── 5.2
```

---

## 附录 B：各阶段完成后的预期指标改善

| 完成阶段 | 综合评分 | 最大改善项 |
|----------|----------|-----------|
| 初始状态 | 24% | — |
| Phase 0 | 38% | 数据质量（28%→65%）、信号可信度 |
| Phase 1 | 44% | 可维护性提升，无得分项改变 |
| Phase 2 | 54% | 防过拟合（38%→65%）、数据质量（65%→78%）|
| Phase 3 | 64% | Alpha 信号质量（25%→55%）、组合完整性（28%→55%）|
| Phase 4 | 74% | 风险模型（8%→65%）、组合完整性（55%→75%）|
| Phase 5 | 82% | 运营系统（5%→70%）、因子生命周期（5%→72%）|

---

## 附录 C：各任务估时汇总

| Task | 描述 | 估时 | 难度 |
|------|------|------|------|
| 0.1 | 真实数据接入主流程 | 0.5 天 | 低 |
| 0.2 | 修复假行业分组 | 1 天 | 低 |
| 0.3 | 修复 fitness 一致性 | 1 天 | 低 |
| 1.1 | 删除冗余包 | 0.5 天 | 低 |
| 1.2 | 统一 import 风格 | 1 天 | 低 |
| 1.3 | 退役旧 AlphaEvolver | 0.5 天 | 低 |
| 1.4 | 合并重复评估函数 | 1 天 | 低 |
| 1.5 | API 并发保护 | 1 天 | 低 |
| 2.1 | Walk-Forward 框架 | 4 天 | 中 |
| 2.2 | Embargo Period | 1 天 | 低 |
| 2.3 | 多市场并发加载 | 2 天 | 低 |
| 2.4 | 数据健康检查集成 | 1 天 | 低 |
| 3.1 | 行业分类数据接入 | 3 天 | 中 |
| 3.2 | 多 Alpha 联合组合 | 5 天 | 高 |
| 3.3 | Beta 中性化 | 3 天 | 中 |
| 3.4 | DSL 新节点扩展 | 3 天 | 中 |
| 3.5 | AlphaPool 正交化 | 2 天 | 中 |
| 4.1 | Regime 识别 | 5 天 | 高 |
| 4.2 | MVO 组合优化 | 5 天 | 高 |
| 4.3 | Deflated Sharpe | 2 天 | 中 |
| 5.1 | 因子在线监控 | 5 天 | 高 |
| 5.2 | 生命周期状态机 | 3 天 | 中 |
| 5.3 | APScheduler 调度 | 3 天 | 中 |
| 5.4 | API 端点扩展 | 3 天 | 低 |
| **总计** | | **~56 天** | |

---

*路线图版本 v1.0 | 2026-06-01 | 基于 AUDIT_REPORT.md*
