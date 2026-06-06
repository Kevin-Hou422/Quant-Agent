# Quant Agent Frontend — 综合审计报告

**审计日期：** 2026-06-07  
**覆盖范围：** `frontend/src/` 全部 TypeScript/TSX 源文件（17 个文件）  
**参考文件：** `backend/AUDIT_REPORT.md`（2026-06-01）| `backend/DEV_ROADMAP.md`（2026-06-01）  
**审计方法：** 全量代码精读 + 前后端接口对照 + 与后端路线图交叉分析

---

## 目录

1. [工程架构总览](#1-工程架构总览)
2. [界面现状描述](#2-界面现状描述)
3. [前后端一致性分析](#3-前后端一致性分析)
4. [功能完整性评估](#4-功能完整性评估)
5. [代码质量问题](#5-代码质量问题)
6. [与后端路线图的对齐差距](#6-与后端路线图的对齐差距)
7. [优先级改进建议](#7-优先级改进建议)
8. [各阶段前端任务清单](#8-各阶段前端任务清单)

---

## 1. 工程架构总览

### 1.1 目录结构

```
frontend/
├── index.html
├── vite.config.ts              # Vite 开发服务器，/api 代理到 http://127.0.0.1:8000
├── package.json                # React 19.2.4 / TS 6.0 / Vite 8.0 / Tailwind 4.2
├── src/
│   ├── main.tsx                # React 应用入口
│   ├── App.tsx                 # 根组件（渲染 WorkspaceLayout）
│   ├── types/index.ts          # 全局类型定义（10 个接口）
│   ├── api/client.ts           # HTTP / SSE API 客户端（axios + fetch）
│   ├── store/workspaceStore.ts # Zustand 全局状态（单一 store，18 个状态字段）
│   ├── hooks/useQuantWorkspace.ts  # 业务逻辑 hook（6 个核心操作）
│   └── components/
│       ├── layout/
│       │   ├── WorkspaceLayout.tsx   # 四列布局主框架
│       │   ├── GlobalSidebar.tsx     # 左侧 64px 图标工具栏
│       │   ├── SessionHistoryPanel.tsx # Chat 模式下的会话列表（192px）
│       │   ├── LeftLedgerPane.tsx    # Compiler 模式下的 Alpha 台账（240px）
│       │   └── RightPane.tsx         # 分析面板（360px，固定宽度）
│       ├── chat/
│       │   ├── ChatView.tsx          # 聊天界面（消息列表 + 输入框）
│       │   ├── ChatMessage.tsx       # 单条消息渲染
│       │   └── ThoughtBlock.tsx      # 可折叠推理块
│       ├── compiler/
│       │   ├── CompilerView.tsx      # Monaco 编辑器 + 多标签页
│       │   ├── ConsoleOutput.tsx     # 实时日志控制台
│       │   └── ConfigModal.tsx       # 回测参数配置弹窗
│       ├── analysis/
│       │   ├── PnLChart.tsx          # ECharts IS/OOS 累计收益曲线
│       │   ├── MetricsGrid.tsx       # IS/OOS 指标对比表格
│       │   └── OverfitBadge.tsx      # 过拟合评分徽章
│       └── ErrorBoundary.tsx         # React 错误边界
```

### 1.2 技术栈

| 层级 | 技术选型 |
|------|----------|
| UI 框架 | React 19.2.4 + TypeScript 6.0 |
| 构建工具 | Vite 8.0（HMR，/api 反向代理） |
| 样式 | Tailwind CSS 4.2（暗色调，slate/emerald 主题） |
| 状态管理 | Zustand 5.0（单一扁平 Store） |
| 代码编辑器 | Monaco Editor + 自定义 QuantDSL 语言注册 |
| 图表 | ECharts（echarts-for-react） |
| HTTP | Axios（REST）+ 原生 Fetch（SSE） |
| 布局 | react-resizable-panels（Monaco ↔ Console 纵向可拖拽） |
| 图标 | Lucide React |

---

## 2. 界面现状描述

### 2.1 整体布局

界面采用四列固定 + 弹性中间列的布局，风格接近 IDE（深色主题，slate-950 背景）：

```
┌────────┬────────────────┬────────────────────────────────┬──────────────┐
│        │                │                                │              │
│  Icon  │  CHAT 模式:     │  CHAT 模式: ChatView           │  仅 COMPILER │
│ Toolbar│  Session List  │  ─────────────────────────     │  模式显示:   │
│ 64px   │  192px         │  消息列表（流式打字机效果）       │  RightPane   │
│        │                │  + 输入框                       │  360px       │
│        │  COMPILER 模式: │  ─────────────────────────     │              │
│        │  Alpha Ledger  │  COMPILER 模式: CompilerView    │  PnLChart    │
│        │  240px         │  ─────────────────────────     │  MetricsGrid │
│        │  (可关闭)      │  Monaco 编辑器（多标签）         │  OverfitBadge│
│        │                │  工具栏（Run/AI Optimize/Config）│              │
│        │                │  ─── 可拖拽分割线 ───            │              │
│        │                │  ConsoleOutput（日志控制台）     │              │
└────────┴────────────────┴────────────────────────────────┴──────────────┘
```

### 2.2 CHAT 模式界面

| 区域 | 描述 | 现状 |
|------|------|------|
| SessionHistoryPanel | 侧边栏会话列表，支持新建/重命名/删除 | 已实现，功能完整 |
| ChatView | 消息列表 + 发送输入框，流式打字机效果 | 已实现，体验流畅 |
| ThinkingIndicator | 处理中动画（彩色跳动点 + 状态文字）| 已实现，区分 optimizing/backtesting |
| 空状态引导 | 显示示例提示语，引导用户首次交互 | 已实现 |

### 2.3 COMPILER 模式界面

| 区域 | 描述 | 现状 |
|------|------|------|
| 多标签页 Tab Bar | 多 Alpha DSL 并行编辑，修改状态指示（●）| 已实现 |
| Monaco Editor | 自定义 QuantDSL 语言，语法高亮（绿/蓝/金）+ IntelliSense 补全 | 已实现，质量高 |
| 工具栏 | Run Backtest（绿）/ AI Optimize（紫）/ Config（齿轮）| 已实现 |
| ConfigModal | 支持 delay/decay_window/truncation/portfolio_mode/top_pct 配置 | 已实现 |
| ConsoleOutput | 实时日志，13 种颜色分类（ERROR/WARN/OK/GP/Optuna/Backtest 等）| 已实现，细节丰富 |
| Alpha Ledger | 历史 Alpha 列表（Sharpe/IC-IR/日期），点击在新标签页打开 | 已实现 |

### 2.4 分析面板（RightPane）

| 组件 | 描述 | 现状 |
|------|------|------|
| PnLChart | ECharts 面积折线图，IS（绿）/ OOS（蓝）双系列，Tooltip | 已实现，UI 精良 |
| MetricsGrid | IS/OOS 六项指标对比表（Return/Sharpe/Drawdown/IC/IC-IR/Turnover）| 已实现 |
| OverfitBadge | 过拟合评分徽章（高分→红色，低分→绿色）| 已实现 |
| 空状态 | 未运行时显示提示和图标占位 | 已实现 |

---

## 3. 前后端一致性分析

### 3.1 严重断路：前端硬编码合成数据参数

所有前端发出的 workflow 和 simulate 请求均硬编码合成数据参数：

```typescript
// api/client.ts:17-67 — 所有 4 个核心 API 函数
apiSimulate(dsl, config, nTickers = 20, nDays = 252, ...)
apiWorkflowGenerate(hypothesis, nDays = 252)  // 内含 n_tickers: 20
apiWorkflowOptimize(dsl, nDays = 252)         // 内含 n_tickers: 20
streamWorkflowOptimize(dsl, ...)              // 内含 n_tickers: 20, n_days: 252
```

后端 `AUDIT_REPORT.md` 第 9.2 节指出，P0 任务是将真实数据接入主工作流（替换 `_make_synthetic_dataset()`）。**即使后端 Phase 0 全部完成，只要前端不修改 API 调用参数，用户依然无法使用真实数据集**。这是前后端之间最高优先级的断路问题。

**缺失 UI：** 前端没有任何数据集选择器（`us_tech_large` / `china_tech` / `crypto_major` 等），也没有日期范围选择器。

### 3.2 Workflow A（假设→生成）从未在 UI 中暴露

`api/client.ts` 中定义了 `apiWorkflowGenerate` 和 `streamWorkflowGenerate`，但 `useQuantWorkspace.ts` 中**没有调用这两个函数的路径**。

- `sendChat()` 调用 `streamChat`（聊天路由），由后端 `quant_agent.py` 内部决定是否触发 WorkflowGenerate
- 用户无法直接触发 `POST /workflow/generate`（从假设出发的 GP 进化）
- `streamWorkflowGenerate` 是死代码

### 3.3 回测进度是假动画

`useQuantWorkspace.ts:22-29` 中 `runBacktest()` 使用 `startProgressStream` 以固定 750ms 间隔播放预写好的 7 条日志，而非真实 SSE 进度：

```typescript
const BACKTEST_STEPS = [
  '[System] Parsing DSL expression...',
  '[System] Compiling AST to signal tree...',
  // ...硬编码的 7 条，与实际执行状态无关
]
```

`runOptimize()` 正确使用了 SSE（`streamWorkflowOptimize`），但 `runBacktest()` 使用的 `apiSimulate` 是普通 REST 接口，前端用假动画掩盖了等待延迟。用户看到的进度与实际执行无关。

### 3.4 PnL 图表 X 轴使用假日期

`PnLChart.tsx:58-64` 生成 x 轴日期的方式是从今天往前倒数，而非使用后端返回的实际回测日期：

```typescript
const today = new Date()
const xData = Array.from({ length: total }, (_, i) => {
  const d = new Date(today)
  d.setDate(d.getDate() - (total - 1 - i))  // 从今天往前计算，与实际回测区间无关
  return d.toISOString().slice(5, 10)
})
```

后端的 `SimResult` 中有 `split_date` 字段，但前端未使用它来对齐真实日期坐标轴。

### 3.5 IS/OOS 指标字段映射不完整

`useQuantWorkspace.ts:426-444` 在 GP 优化完成后手工映射指标字段，但仅映射了 3 个字段（sharpe/return/IC），而 MetricsGrid 显示 6 个指标：

```typescript
// runOptimize() 中的映射（不完整）
is_metrics: {
  sharpe_ratio:      m?.is_sharpe      ?? null,
  annualized_return: m?.is_return      ?? null,
  ic_ir:             m?.is_ic          ?? null,
  // 缺失: max_drawdown, ann_turnover, mean_ic
}
oos_metrics: m?.oos_sharpe != null ? { sharpe_ratio: m.oos_sharpe } : null,
// OOS 仅有 Sharpe，其余 5 项 MetricsGrid 均显示 "—"
```

---

## 4. 功能完整性评估

### 4.1 已完整实现

| 功能 | 评分 | 说明 |
|------|------|------|
| 双视图布局切换（Chat/Compiler）| ✅ | 流畅，状态不丢失 |
| Monaco DSL 语法高亮 + IntelliSense | ✅ | QuantDSL 自定义语言，补全完整 |
| 多标签页 DSL 编辑 | ✅ | 修改状态指示、从台账打开均正确 |
| 聊天 SSE 流式打字机 | ✅ | 事件队列防乱序，体验良好 |
| GP 优化 SSE 进度流 | ✅ | 每代进度实时推送到 Console 和 Chat |
| IS/OOS PnL 图表 | ✅ | 双系列、区域填充、Tooltip 完整 |
| IS/OOS 指标对比表 | ✅ | Sharpe 退化高亮（红/橙色警告）|
| 回测结果自动保存到台账 | ✅ | 每次 runBacktest 完成后自动调用 apiSaveAlpha |
| 会话持久化（localStorage）| ✅ | 刷新后恢复最近会话 |
| 乐观更新（重命名/删除会话）| ✅ | 先更新 UI，失败时回滚 |
| 过拟合徽章 | ✅ | 评分连续渐变，颜色直观 |
| 错误分类日志（13 种颜色）| ✅ | [ERROR]/[WARN]/[GP]/[Optuna] 等区分清晰 |
| Console 行操作（View/Rerun/Copy）| ✅ | Hover 时显示快捷操作 |
| 离线降级（网络异常时创建内存会话）| ✅ | try/catch 全链路兜底 |

### 4.2 已实现但有缺陷

| 功能 | 评分 | 问题 |
|------|------|------|
| 回测进度反馈 | ⚠️ | 假动画，与实际执行无关 |
| PnL 图表时间轴 | ⚠️ | 使用今天日期倒推，非真实回测区间 |
| GP 优化后指标展示 | ⚠️ | OOS 仅 Sharpe，其余字段为 `—` |
| IC Decay 展示 | ⚠️ | 仅当 `ic_decay` 非空时显示，后端当前返回空 `{}` 导致区块不渲染 |
| Alpha 台账状态点 | ⚠️ | `status` 字段固定为 `"active"`，未反映后端实际 status 值 |
| 请求取消 | ⚠️ | `streamChat`/`streamWorkflowOptimize` 接受 `signal` 参数但调用方未传入 AbortController |

### 4.3 完全缺失

| 功能 | 优先级 | 说明 |
|------|--------|------|
| 数据集选择器 | P0 | 无 UI 让用户选择 us_tech_large / china_tech 等数据集 |
| 日期范围选择 | P0 | 回测起止日期完全不可配置 |
| Workflow A 直接触发 | P0 | 假设→GP 生成路径未暴露到 UI |
| Walk-Forward 结果展示 | P2 | 后端 Phase 2 完成后需要多折叠图表 |
| 多 Alpha 对比视图 | P3 | 台账目前只能逐一打开，无并排对比 |
| Regime 市场状态指示器 | P4 | 后端 Phase 4 完成后需前端展示 |
| Alpha 生命周期仪表板 | P5 | 后端 Phase 5 新增 /alphas/dashboard 端点后需对应 UI |
| IC 衰减监控折线图 | P5 | 需新增时序图表组件 |
| 真实数据健康检查反馈 | P2 | 后端加入健康检查后前端需展示数据质量评分 |
| 请求并发控制（429 提示）| P1 | 后端 Task 1.5 加入限流后，前端需处理 429/408 响应 |

---

## 5. 代码质量问题

### 5.1 useEffect 缺失依赖数组

```typescript
// WorkspaceLayout.tsx:36
useEffect(() => { initSessions() }, [])  // eslint-disable-line react-hooks/exhaustive-deps
```

`initSessions` 是 `useQuantWorkspace()` 的返回值，理应在 deps 中声明。当前用 eslint-disable 绕过是工程债务，正确做法是将 `initSessions` 提取为 `useCallback` 并使其引用稳定。

### 5.2 状态管理中的冗余

`workspaceStore.ts` 同时维护了 `editorDsl`（字符串）和 `editorTabs[i].dsl`（数组项内的字符串），两者需手工同步（`setEditorDsl`、`setActiveTab` 均更新 `editorDsl`）。当标签页很多时，这种双重源容易产生不一致——特别是 `openAlphaInNewTab` 中设置了 `activeView: 'COMPILER'` 但没有确保 `editorDsl` 同步到新 tab 内容的情况。

事实上代码是正确的，但这种双重源的存在增加了未来修改时引入 bug 的风险，建议在注释中明确说明 `editorDsl` 是 `activeTab.dsl` 的镜像。

### 5.3 打字机队列轮询

`useQuantWorkspace.ts:265-270`（chat）和 `:394-398`（optimize）中均使用 `setInterval` 轮询 `textQueue` 清空状态来等待打字结束：

```typescript
await new Promise<void>((resolve) => {
  const check = setInterval(() => {
    if (!typing && textQueue.length === 0) { clearInterval(check); resolve() }
  }, 50)
})
```

这是一种 busy-wait 模式，50ms 间隔轮询浪费了少量 CPU。更好的方式是使用 `Promise` + `resolve` 回调在打字真正结束时直接通知，或者使用 `EventEmitter`/`BroadcastChannel`。影响较小，但代码复杂度值得优化。

### 5.4 API 层无请求去重

用户快速多次点击 `Run Backtest` 或 `AI Optimize` 时，按钮虽然有 `disabled={isRunning}` 保护，但 `isRunning` 只检查 `status` 字段。如果用户在 `status` 切换前双击（竞态窗口），会发出多个请求。建议在 hook 层加入 `in-flight ref` 守卫。

### 5.5 Console 日志无自动滚动到底部

`ConsoleOutput.tsx` 中日志列表是 `overflow-y-auto` 但没有自动滚动逻辑（`useRef` + `scrollIntoView`）。当新日志追加时，用户需要手动滚动到底部查看最新内容，体验差于 Chat 视图。

### 5.6 `streamWorkflowGenerate` 是死代码

`api/client.ts:186-218` 中定义的 `streamWorkflowGenerate` 函数未被任何组件或 hook 引用，属于死代码，应删除或补上调用路径。

---

## 6. 与后端路线图的对齐差距

下表对应 `backend/DEV_ROADMAP.md` 的 5 个 Phase，列出前端对应的工作量：

| Phase | 后端任务 | 前端当前状态 | 需要的前端工作 |
|-------|----------|------------|--------------|
| **Phase 0** | 接入真实数据，修复假行业分组 | ❌ 前端所有 API 调用硬编码合成数据参数 | **新增数据集选择器 + 日期范围选择器；修改全部 API 调用参数** |
| **Phase 0** | 修复 GP fitness 一致性 | ✅ 前端无感知，后端内部变化 | 无需改动 |
| **Phase 1** | API 并发保护（429/408 响应）| ❌ 前端未处理 429/408 | 新增 429 toast 提示（"请稍后重试"）和 408 超时提示 |
| **Phase 2** | Walk-Forward 多轮验证 | ❌ 前端无 WF 结果展示 | **新增多折图表组件（每折 OOS Sharpe + 均值/std 汇总）** |
| **Phase 2** | 数据健康检查 | ❌ 前端无反馈 | 在数据集选择后展示健康评分 badge |
| **Phase 3** | 多 Alpha 联合组合 | ❌ 台账无对比功能 | **新增多选 checkbox + 组合信号图表** |
| **Phase 3** | Beta 中性化 + 行业暴露 | ❌ MetricsGrid 无行业/Beta 行 | 新增 `market_beta` 和 `sector_exposures` 展示区 |
| **Phase 4** | Regime 识别 | ❌ 无相关 UI | 新增 Regime 指示器（Bull/Bear/Sideways 标签） |
| **Phase 4** | Deflated Sharpe Ratio | ❌ MetricsGrid 无 DSR 行 | 在 Metrics 表增加 DSR 行，并标注"调整后" |
| **Phase 5** | 因子生命周期仪表板 | ❌ 台账仅列表，无生命周期视图 | **新增专用 Alpha 管理页面**（生命周期状态机可视化）|
| **Phase 5** | IC 衰减监控 | ❌ 无时序监控图 | 新增 `IC History` 折线图组件 |
| **Phase 5** | APScheduler 定时任务 | ❌ 无任务调度 UI | 新增任务状态展示（daily_data_update 上次运行时间等）|

**最关键差距**：前端数据集参数硬编码是 P0 阻塞级问题。即使后端 Phase 0 全部完成，若前端不同步修改，整个真实数据链路依然无法被用户使用。

---

## 7. 优先级改进建议

### P0（1-3 天，与后端 Phase 0 同步）

#### 7.1 新增数据集选择器

在 `ConfigModal.tsx` 中增加数据集配置项：

```typescript
// types/index.ts — 扩展 SimulationConfig
interface SimulationConfig {
  // 现有字段...
  dataset:    string   // 新增，默认 "us_tech_large"
  start_date: string   // 新增，默认 "2020-01-01"
  end_date:   string   // 新增，默认 "2024-01-01"
}

// ConfigModal.tsx — 新增数据集下拉框
const DATASETS = [
  { value: 'us_tech_large',   label: 'US Tech (Large Cap)' },
  { value: 'us_financials',   label: 'US Financials' },
  { value: 'china_tech',      label: 'China Tech (CSI)' },
  { value: 'china_value',     label: 'China Value' },
  { value: 'hong_kong_large', label: 'Hong Kong Large Cap' },
  { value: 'crypto_major',    label: 'Crypto Major' },
]
```

#### 7.2 修改全部 API 调用，传入 dataset/start/end

```typescript
// api/client.ts — apiSimulate 调用替换
export const apiSimulate = (dsl: string, config: SimulationConfig) =>
  http.post<SimResult>('/alpha/simulate', {
    dsl,
    config,
    dataset:    config.dataset    ?? 'us_tech_large',
    start_date: config.start_date ?? '2020-01-01',
    end_date:   config.end_date   ?? '2024-01-01',
    oos_ratio:  0.3,
    // 移除 n_tickers 和 n_days（合成数据参数）
  })

// streamWorkflowOptimize 同理
```

#### 7.3 Workflow A 入口：在 Chat 输入框增加"Generate Alpha"模式切换

将 `streamWorkflowGenerate` 接入 `sendChat` 路径，或在 CompilerView 工具栏增加"Generate from Hypothesis"按钮（触发 Workflow A 而非 Workflow B）。

---

### P1（1-2 周，与后端 Phase 1 同步）

#### 7.4 为 Backtest 添加真实 SSE 进度

后端可以扩展 `/alpha/simulate` 为 SSE 端点（或新增 `/alpha/simulate/stream`），前端对应修改 `runBacktest` 使用 `streamWorkflowOptimize` 相同的模式替代假动画。

**短期过渡方案**（不改后端）：将假动画的步骤字符串替换为基于实际 `apiSimulate` 响应时间的动态估算，或去掉假动画改为单条 "Running…" 状态。

#### 7.5 处理 429/408 响应

```typescript
// api/client.ts — http.interceptors.response.use
http.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 429)
      return Promise.reject(new Error('GP 任务正在运行，请稍后重试'))
    if (err.response?.status === 408)
      return Promise.reject(new Error('任务超时，请缩小种群规模后重试'))
    return Promise.reject(err)
  }
)
```

#### 7.6 修复 Console 自动滚动

```typescript
// ConsoleOutput.tsx — 添加底部 ref
const bottomRef = useRef<HTMLDivElement>(null)
useEffect(() => {
  bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
}, [consoleLogs])
```

#### 7.7 修复 PnL 图表 X 轴

使用后端返回的 `split_date` 来构建真实日期轴，而非从今天倒推。

```typescript
// PnLChart.tsx — 使用 split_date 对齐坐标轴
const splitDate = simulationResult.split_date ? new Date(simulationResult.split_date) : null
// 从 splitDate 向前推 isRaw.length 天，向后推 oosRaw.length 天
```

---

### P2（3-4 周，与后端 Phase 2 同步）

#### 7.8 Walk-Forward 结果展示组件

新建 `WalkForwardChart.tsx`：
- 展示每折 OOS Sharpe 的柱状图
- 展示均值 ± 标准差
- 标注正收益折数占比（`pct_positive_folds`）

当 `SimResult` 中包含 `walk_forward_result` 字段时，在 `RightPane` 中追加此组件。

#### 7.9 数据集元信息展示

在 RightPane 顶部增加当前数据集信息 Banner：
```
数据集: US Tech Large | 日期: 2020-01-01 ~ 2024-01-01 | 股票数: 50 | 健康评分: 0.92
```

---

### P3（6-8 周，与后端 Phase 3 同步）

#### 7.10 多 Alpha 对比视图

在 Alpha Ledger 中增加多选 checkbox，选中多个 Alpha 后触发"Compare"按钮，弹出对比面板：
- 各 Alpha 的 PnL 曲线叠加
- 相关性热力图（信号间 pairwise correlation）

#### 7.11 Beta / 行业暴露展示

在 MetricsGrid 下方增加风险暴露区块：
```
市场 Beta: 0.03 ✓  |  行业偏差最大: IT +3.2%
```

---

### P4（6-8 周，与后端 Phase 4 同步）

#### 7.12 Regime 状态徽章

在 ChatView 头部或 RightPane 新增 Regime 指示器：
```
当前 Regime: [BULL ↑]  |  建议：偏向动量因子
```

#### 7.13 Deflated Sharpe 在 MetricsGrid 中展示

MetricsGrid 增加一行 DSR，并在旁边附注当前 GP 试验次数（n_trials_run）：

```
Deflated SR | 0.63 | —  ← 标注"调整了 50 次试验"
```

---

### P5（8-12 周，与后端 Phase 5 同步）

#### 7.14 Alpha 生命周期管理页面

在 GlobalSidebar 增加第三个视图入口（"Dashboard"图标），显示：
- 活跃 Alpha 卡片（含实时滚动 IC-IR）
- 衰减告警（DecayAlert 红色卡片）
- 状态流转按钮（Validate / Paper / Activate / Retire）

#### 7.15 IC 历史监控图

在 Alpha 详情页（点击台账条目后展开）显示：
- 30/60 天滚动 IC 折线图
- IC-IR 滚动趋势
- 衰减阈值参考线（0.3 虚线）

---

## 8. 各阶段前端任务清单

对应 `backend/DEV_ROADMAP.md` 附录 C，增补前端任务：

| Task | 描述 | 估时 | 优先级 | 依赖 |
|------|------|------|--------|------|
| FE-0.1 | 新增数据集选择器 + ConfigModal 扩展 | 1 天 | 🔴 | BE Task 0.1 完成后联调 |
| FE-0.2 | 修改全部 API 调用移除合成数据参数 | 0.5 天 | 🔴 | FE-0.1 |
| FE-0.3 | 接入 Workflow A（hypothesis → generate）| 1 天 | 🔴 | — |
| FE-1.1 | 处理 429/408 响应 + toast 提示 | 0.5 天 | 🟡 | BE Task 1.5 |
| FE-1.2 | Console 自动滚动到底部 | 0.5 天 | 🟡 | — |
| FE-1.3 | 修复 PnL 图表 X 轴（使用 split_date）| 1 天 | 🟡 | — |
| FE-1.4 | 修复 GP 优化后 OOS 指标映射完整性 | 0.5 天 | 🟡 | — |
| FE-1.5 | 请求 AbortController（支持取消）| 1 天 | 🟢 | — |
| FE-2.1 | Walk-Forward 结果图表组件 | 2 天 | 🟡 | BE Task 2.1 |
| FE-2.2 | 数据集元信息 Banner | 1 天 | 🟢 | BE Task 0.1 |
| FE-3.1 | 多 Alpha 对比视图（台账多选 + 叠加图）| 3 天 | 🟡 | BE Task 3.2 |
| FE-3.2 | Beta / 行业暴露展示 | 2 天 | 🟡 | BE Task 3.3 |
| FE-4.1 | Regime 状态徽章 | 1 天 | 🟢 | BE Task 4.1 |
| FE-4.2 | Deflated Sharpe 在 MetricsGrid | 1 天 | 🟢 | BE Task 4.3 |
| FE-5.1 | Alpha 生命周期仪表板页面 | 5 天 | 🟡 | BE Task 5.4 |
| FE-5.2 | IC 历史监控折线图 | 2 天 | 🟡 | BE Task 5.1 |
| FE-5.3 | 任务调度状态展示 | 1 天 | 🟢 | BE Task 5.3 |
| **总计** | | **~23 天** | | |

---

## 附录：前端架构合理性评分

| 维度 | 评分（/10）| 说明 |
|------|------------|------|
| 视觉设计 | 9 | 深色 IDE 主题统一，配色层次清晰，组件细节精良 |
| 组件拆分 | 8 | layout/chat/compiler/analysis 四层合理；可继续将 hooks 细分 |
| 状态管理 | 7 | Zustand 单 Store 对当前规模合适；双源 editorDsl 是隐患 |
| API 层设计 | 6 | 接口封装清晰；但硬编码合成数据参数是严重问题 |
| 用户体验 | 7 | 流式打字机、状态色彩、空状态引导均好；假进度动画是减分项 |
| 可扩展性 | 6 | 目前为单页应用，无路由；Phase 5 的多页面需求需引入 react-router |
| 类型安全 | 7 | 全量 TypeScript；部分 `as any` 类型断言需整理 |
| 与后端对齐 | 4 | 数据集参数断路、Workflow A 死代码、假日期轴是主要失分项 |
| **综合** | **6.8** | 视觉质量高，但与后端的数据流对齐是当前最大短板 |

---

*报告生成：2026-06-07 | 版本 v1.0 | 覆盖文件数：17 个 .tsx/.ts 文件*
