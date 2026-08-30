# DEV_LESSONS — 项目专属经验库

> 用途：记录**修好某个 bug 后**能沉淀出的、对后续开发有用的具体经验。开发/调试相关模块时值得先扫一眼。
> 约定：每条 = 症状 → 结论 → 防护。越具体越好，不为凑字数。新经验**追加**到对应主题下，注明日期。

---

## A. 测试绿 ≠ 系统能跑（最重要）

**2026-08-20** — 一次真实启动前后端的 E2E 验证，在 448 项测试全绿的情况下，仍查出 3 个真实 bug。
原因：测试要么 mock 数据、要么用合成/缓存数据，**结构性地绕开了真实运行路径**（真实 provider
拉取、真实面板的列名、真实退市股票）。

- **结论**：里程碑收尾前，**必须真实跑一遍前后端 + 真实数据**，不能只看测试全绿。
- **防护**：凡涉及"真实数据加载 / 实时 / 摄取 / 成交"的改动，收尾时手动跑一次真实 E2E
  （启动 uvicorn + vite，打真实端点，肉眼核对金融数字是否自洽）。

## B. 安全门不要 fail-open（静默放行）

**2026-08-20** — `check_dataset_health` 内部 `except Exception: return None`，而调用方把 `None`
当"满分通过"。结果健康门崩溃后**一直静默放行**，"拒绝坏数据"（A2）形同虚设，且测试照样绿。

- **结论**：安全/质量门一旦异常，**宁可 fail-closed（拒绝）也不要 fail-open（放行）**；
  宽泛 `except` 里返回"中性/通过值"= 把保护悄悄关掉。
- **防护**：安全门的异常至少 `logger.error`；**写一个断言"坏数据真的被拒"的测试**，而不只是
  "好数据通过"。

## C. 相对导入深度 & 真实路径只在运行时暴露

**2026-08-20** — `dataset_registry.py` 里 `_fetch_yfinance/akshare/ccxt` 导入写错
（`..yahoo_provider` 应为 `.yahoo_provider`；akshare/ccxt 少了 `.providers.`），**所有真实数据
加载都 500**，但因为测试不走真实拉取，没人发现。

- **结论**：`.` vs `..`、子包前缀（`.providers.`）这类相对导入错误，**只在真正执行那条路径时才炸**。
- **防护**：改动"按 provider 分发"的懒加载导入后，直接跑一次真实加载（或 `python -c` 导入那几个函数）。

## D. pandas 面板处理的两个坑

**2026-08-20**
- **列名别按假设的 `level_0/level_1` 重命名**：`df.stack().reset_index()` 的列名取决于索引名
  （yfinance 的行索引叫 `Date`，不是 `level_0`）→ 重命名漏改 → 下游 `KeyError`。
  **按位置赋值**：`long_df.columns = ["timestamp","ticker","close"]`。
- **`stack(dropna=...)` 在 pandas 2.2+ 已废弃/报错**：保留 NA 用 `stack(future_stack=True)`
  （本项目 `pit_store.py`、健康检查处各踩过一次）。

## E. 硬编码 ticker универс会随时间腐坏

**2026-08-20** — SQ(→XYZ)、ABC(→COR)、DFS/PXD(被收购) 等退市/更名后，yfinance 返回**全 NaN 列**，
静默污染所有截面算子（rank/zscore/neutralize）。

- **结论**：硬编码股票池会因并购/更名而出现死列；全 NaN 列不报错但破坏截面计算。
- **防护**：定期扫描各数据集的死列（`close.isna().mean() > 0.95`）；健康门（B 修好后）也能兜住。
  注意：删死列会引入幸存者偏差——但它们本就无数据、删除只是止损，历史幸存者偏差是已知的 F3 遗留。

## F. "机器精度"对账多半是合成数据下的数字

**2026-08-20** — PaperBroker↔回测对账在合成数据下 2.2e-16，但在**真实数据**下是 5.8e-5
（因为真实数据里 ADV 流动性约束会**真的 binding**，两条代码路径的处理有微小差异）。5.8e-5 累计
/501 天在容差内、金融上可忽略，但要知道它不是 2.2e-16。

- **结论**：号称"机器精度"的对账，往往是约束不 binding 的合成数据下才成立。
- **防护**：容差断言要在**约束真的生效的真实数据**上验证；报告数字时标清是合成还是真实口径。

## G. APScheduler 的 CronTrigger 默认用本地时区，不是 scheduler 的时区

**2026-08-22** — `BackgroundScheduler(timezone="UTC")` + `CronTrigger(hour=21)`，实际 next_run 却是
`21:00+08:00`（本地）。因为 **`CronTrigger` 在构造时就把本地时区烘进去了**，不会继承 scheduler
的 timezone。后果：本该"美股收盘后（21:00 UTC）"跑的任务，在本地 21:00（=13:00 UTC=美股早上）就跑了，
当天收盘数据还不存在。**只有真跑起来看 next_run 才会发现**（单测只查 job 是否注册，不查时区）。

- **结论**：APScheduler 的 `CronTrigger`/`IntervalTrigger` **必须显式传 `timezone=`**，别指望 scheduler
  的 timezone 生效。
- **防护**：每个 trigger 显式 `CronTrigger(..., timezone=timezone)`；启动后断言/核对 `next_run` 的
  时区偏移是否为预期（+00:00）。

## H. 部署 .env 不得泄漏进测试

**2026-08-22** — 为真实运行在 `backend/.env` 里设了 `ENABLE_SCHEDULER/PAPER_TRADING/DISCOVERY=true`。
测试也从 `backend/` 跑 → 读到同一 `.env` → **session 级 `TestClient` 的 lifespan 启动了真实调度器**
（起后台线程 + 单例 `_scheduler` 全程 running），导致 `test_status_reflects_running_state` 的前置
`running is False` 失败。全量跑挂、单独跑过（因为单独跑没有别的 test_client 触发 lifespan）。

- **结论**：测试结果**必须与部署配置无关**；部署用的运行开关（enable_*）会经共享 `.env` 泄漏进测试。
- **防护**：conftest 加 **autouse session fixture 强制 `settings.enable_* = False`**（已加
  `_hermetic_run_flags`）；更广义地，开发/测试与运行实例**别共用同一工作目录/.env**（本仓库已踩过多次
  cwd 共用坑：scheduler_jobs.db 竞争、.env 泄漏）。

## I. 把 held-out 集喂进选择目标 = 它就不再是样本外了

**2026-08-24**（外部评估指出，Phase S.1 修）— GP 适应度是
`fitness = sharpe_oos − …`，即**直接按 OOS Sharpe 择优**。跑 100+ 候选全按 OOS 排序 →
**OOS 段被彻底挖掘，退化为第二个样本内**；后面的 WalkForward/DSR 在被挖过的同一段上"验证"，
沦为**循环论证**。看起来很严，实际零信息。而且 README 声称"immutable 三段切割"，代码里发现/验证
路径其实是两段——**文档-代码不符，且不符方向是高估**。

- **结论**：**任何进入选择/优化目标的数据段，都不再是样本外。** 真 held-out 必须对**整个搜索过程
  不可见**，只在最后汇报一次。
- **防护**：三段切割 IS/Validate/Test——GP 只在 **Validate** 上选择，**Test 全程不可见**（已加
  `_partition_three_way` 到发现路径 + 赢家在 Test 上汇报 `test_sharpe`）；更进一步，持久化**跨会话
  累计 trial 数**喂给 DSR/PBO（Phase S.3），否则多重检验膨胀被系统性低估。

## J. 财务关键参数不能是代码里的隐藏默认

**2026-08-25**（用户发现）— PM 用 `paper_aum`、`PaperBroker` 用自带 `initial_capital=1M`，**两个"资金规模"
各说各话**；且用户从没设过 AUM，系统悄悄按 $1M 算持仓。更广地一查：**成本模型**（`min_ticket=$1`、
`fixed_bps=5`、机构口径）、**策略结构**（永远美元中性**多空**、假设可自由做空）、**调仓频率**（隐含每日
全额）——全是代码默认，悄悄假设了"能自由做空 + 机构级低成本"。**这些直接决定"能不能赚钱"的答案**：
同一因子在 $10k 散户口径 vs $1M 机构口径下，结论可能相反。

- **结论**：当系统的目的就是"**验证在特定资金/成本下能否赚钱**"时，**资金规模、成本口径、可交易性约束
  （能否做空）、调仓频率**这些**直接决定答案的输入，绝不能是代码里的隐藏默认**——必须①显式（用户设）、
  ②由真实情景推导（有分析、非拍脑袋）、③**单一来源**（不重复、不分叉）。
- **防护**：把每个财务/行为参数分三层区别对待——
  - **T1 现实事实**（AUM、账户类型、allow_short、券商费率）→ **显式配置**（用户声明，非隐藏默认）。
  - **T2 数据推导的估计**（价差、冲击、可做空性、调仓带）→ **系统自己从免费数据每次重算**
    （`TradingContext`：Corwin-Schultz 估价差等），明确是"估计"，不冻结成常量。
  - **T3 只有交易当时才知道的**（真实盘口价差、借券 locate/费、成交价、买入力、停牌）→
    **绝不写字面值**，一律走 provider 接口；仿真背 T2 估计、实盘背券商实时 API，**同接口切换**。
  - AUM 只走 `paper_aum` 一条线贯穿 PM 与 broker（已修）；**研究与执行必须用同一个权威数据源**
    （否则 train/serve skew，验证白做）。任何"机构默认"进入"散户小资金"目标前，先问"它在目标情景下还成立吗"。

## K. 完成 ≠ 接进主线（本轮最重要的复盘）

**2026-08-30**（用户一针见血）— 建了 PM.S1/S2（策略门/边际准入）、PM.5（风控门），每个都带单测且全绿，
但**都没接进 `run_portfolio` 主线**——是"带单测的库"，不是"生效的功能"。用户说"很多已完成的部分和没做一样"。

- **结论**：一个功能"完成"的标准 = **在运行时主线上被真正调用 + 端到端验证过**，不是"单测绿"。
  孤立的带单测模块 = 没做。
- **防护**：每做一个模块，**同一批次就接进主线 + 写一个"主线确实调用了它"的集成测试**
  （如 `test_phase_pm_wiring.py` 断言 `run_portfolio` 返回里有该门的产物 `selection/strategy_verdict/risk_report`）；
  里程碑收尾用 **AST 可达性从运行入口扫一遍**，"有价值但不可达"的模块要么接线要么删；再跑真 E2E 串起所有存活组件（§A）。

## L. 删死代码的方法论（可达性 / 同名 / 实测）

**2026-08-30** — 清 ~2740 行死代码时踩了三个坑：① 静态可达性若不建模**包 `__init__` 执行 + re-export**，
会把"经 __init__ 被加载"的模块误判为孤儿；② 死壳与活模块**同名**（`portfolio_constructor` 活在
`backtest_engine`、死壳在 `portfolio_engine`），**按名字删会误杀活的**（验证脚本就误删过）；③ 按**符号名**
grep 会把不同模块的同名函数混为一谈（`generate_random_alpha` 活在 `gp_engine`、死在 `generator`）。

- **结论**："删了无影响"不能靠单一 grep 断言。
- **防护**：删前**四步确认**——① git 历史查来历+为何存在；② 确认有同功能替代且在主线；③ 追**实际 import
  来源**（非符号名）证明活代码不用它；④ 前端依赖（前端只经 API，查端点 handler 不碰它）。最后**实测**：
  备份 → 真删+清 __init__ re-export → `import app.main` + 全量测试 → 还原。**删除按精确路径，绝不按 basename。**

## M. 换数据源的收尾 + 前端"能跑"≠"类型门过"

**2026-08-30**
- **换数据源不只是换 fetch**：符号格式会不同（yfinance `BRK-B` vs moomoo `US.BRK.B`，实测确认）、
  缓存键要含 `price_source`（否则换源返旧缓存）、新部署配置（`PRICE_SOURCE=moomoo`）要在 conftest
  强制回退（否则测试打真网关 + 烧行情额度，再犯 §H）。
- **前端"能构建"≠"类型门过"**：`vite build` 不做类型检查照样产出可运行包；`npm run build` 的 `tsc -b`
  是独立严格门，且会把**测试文件**的类型错误（vitest 全局 `afterEach`、未用 import）带进生产构建门。
  **防护**：生产 tsconfig 排除 `src/__tests__`（测试交 vitest 跑）、`vite.config` 用 `defineConfig from 'vitest/config'`；
  区分"app 能跑"和"类型门过"。

## N. 可视化必须有"绑真数据"的测试（防伪动画）

**2026-08-30** — 回测曲线 `visualizer` 画的是 `report.equity_curve` 真实数据，但原先**没有测试保证这点**——
将来有人改成随机/装饰动画不会被发现。已加 `test_phase_visualizer` 断言"图里净值曲线**逐点 == 真实回测
equity_curve**"。

- **结论**：任何曲线/仪表盘，都要有一个**把图里的数据绑回真实来源**的测试，否则它可能悄悄变成装饰动画。
