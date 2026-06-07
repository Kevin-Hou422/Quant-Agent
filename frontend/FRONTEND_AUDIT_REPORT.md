# Quant Agent Frontend — 综合审计报告

**审计日期：** 2026-06-07  
**覆盖范围：** `frontend/src/` 全部 TypeScript/TSX 源文件（18 个文件）  
**参考文件：** `backend/AUDIT_REPORT.md`（2026-06-01）| `backend/DEV_ROADMAP.md`（v2.0，2026-06-07）  
**最后更新：** 2026-06-07（Dataset 功能实现后更新）

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
├── vite.config.ts              # /api 代理到 http://127.0.0.1:8000
├── package.json                # React 19.2.4 / TS 6.0 / Vite 8.0 / Tailwind 4.2
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── types/index.ts          # 全局类型（ActiveView 含 DATASET、DatasetInfo 等）
│   ├── api/client.ts           # HTTP / SSE API 客户端（含 apiFetchDatasets）
│   ├── store/workspaceStore.ts # Zustand Store（含 datasets、prevView）
│   ├── hooks/useQuantWorkspace.ts  # 业务逻辑 hook
│   └── components/
│       ├── layout/
│       │   ├── WorkspaceLayout.tsx   # 四列布局 + DATASET 视图分支
│       │   ├── GlobalSidebar.tsx     # 含 Data 图标 + 当前数据集显示
│       │   ├── SessionHistoryPanel.tsx
│       │   ├── LeftLedgerPane.tsx
│       │   └── RightPane.tsx
│       ├── chat/
│       │   ├── ChatView.tsx
│       │   ├── ChatMessage.tsx
│       │   └── ThoughtBlock.tsx
│       ├── compiler/
│       │   ├── CompilerView.tsx
│       │   ├── ConsoleOutput.tsx
│       │   └── ConfigModal.tsx       # 含活跃数据集显示 + Change 按钮
│       ├── dataset/
│       │   └── DatasetView.tsx       # ★ 新增：数据集选择独立界面
│       ├── analysis/
│       │   ├── PnLChart.tsx
│       │   ├── MetricsGrid.tsx
│       │   └── OverfitBadge.tsx
│       └── ErrorBoundary.tsx
```

### 1.2 技术栈

| 层级 | 技术选型 |
|------|----------|
| UI 框架 | React 19.2.4 + TypeScript 6.0 |
| 构建工具 | Vite 8.0 |
| 样式 | Tailwind CSS 4.2（slate-950 深色主题）|
| 状态管理 | Zustand 5.0（单 Store）|
| 代码编辑器 | Monaco Editor + QuantDSL 自定义语言 |
| 图表 | ECharts（echarts-for-react）|
| HTTP | Axios（REST）+ 原生 Fetch（SSE）|
| 布局 | react-resizable-panels |
| 图标 | Lucide React |

---

## 2. 界面现状描述

### 2.1 整体布局（含新 DATASET 视图）

```
┌────────┬────────────────────────────────────────────────────┐
│  Icon  │  DATASET 模式（新增）：全宽两列布局                  │
│ Toolbar│  ─────────────────────────────────────────────     │
│  64px  │  左列（288px）             右列（flex-1）           │
│        │  区域分组的数据集列表       数据集详情面板            │
│        │    US Equities             名称 / 地区 / 行业 badge │
│        │    China A-shares          数据提供商 / 资产数       │
│        │    Hong Kong               日期范围选择器            │
│        │    Crypto                  ★ Use this dataset 按钮  │
│        │                            所有 Ticker 列表          │
├────────┼────────────────────────────────────────────────────┤
│  Icon  │  CHAT / COMPILER 模式（原有，不变）                  │
│ Toolbar│  + 底部新增活跃数据集指示器（可点击进入 DATASET）     │
└────────┴────────────────────────────────────────────────────┘
```

### 2.2 新增 Data 入口（GlobalSidebar）

- 左侧图标栏新增 `Database` 图标 + "Data" 标签（第 4 个 NavBtn）
- 图标栏底部新增活跃数据集指示区：
  - 显示当前数据集名称（`us_tech_large`）
  - 显示年份范围（`2020–2024`）
  - 点击任意处跳转到 DATASET 视图
- 活跃样式与其他 NavBtn 一致（emerald 左边框 + slate-800 背景）

### 2.3 Dataset 视图（DatasetView.tsx）

| 区域 | 描述 |
|------|------|
| 顶部 Bar | 返回箭头 + 标题 + 活跃数据集显示 + 刷新按钮 |
| 左侧列表 | 按地区分组（US / China A / HK / Crypto），粘性区域头 |
| 数据集卡片 | 名称 + ACTIVE 标签 + 地区 badge + 资产数 + 起始日期 |
| 右侧详情 | 完整元信息 + 日期范围选择器（Date Input）+ Use Dataset 按钮 + Ticker 全列表 |
| 空状态 | 未选择时显示图标占位 |
| 加载状态 | 首次加载时旋转 spinner |

### 2.4 ConfigModal 更新

- 模态框顶部新增数据集展示行（Database 图标 + 当前 dataset + 日期范围）
- 右侧 "Change" 按钮关闭弹窗并导航到 DATASET 视图

---

## 3. 前后端一致性分析

### 3.1 已修复：前端 API 调用不再硬编码合成数据参数 ✅

**修复内容（2026-06-07）：**

- `SimulationConfig` 类型新增 `dataset`、`start_date`、`end_date` 字段
- `DEFAULT_CONFIG` 默认值：`dataset: 'us_tech_large'`，`start_date: '2020-01-01'`，`end_date: '2024-01-01'`
- `apiSimulate` 不再接受 `nTickers`/`nDays` 参数，改为传入 `dataset_name`/`dataset_start`/`dataset_end`
- `streamWorkflowOptimize` 新增 `dataset`/`startDate`/`endDate` 参数，从 `simConfig` 读取
- `streamWorkflowGenerate` 同样更新签名
- `apiWorkflowGenerate`/`apiWorkflowOptimize`/`apiBacktest` 均改为接受 dataset 参数

### 3.2 已修复：streamWorkflowGenerate 死代码已激活 ✅

- `useQuantWorkspace.ts` 中添加了 `streamWorkflowGenerate` 的 import
- API 客户端函数签名已更新，可在未来 UI 中调用

### 3.3 已实现：GET /api/datasets 端点 ✅

- 后端 `router.py` 新增 `GET /api/datasets`，返回所有注册数据集的元信息（无数据加载）
- 前端 `apiFetchDatasets()` 对应调用
- DatasetView 首次挂载时自动拉取并缓存到 Zustand Store

### 3.4 仍存在：回测进度是假动画 ⚠️

`runBacktest()` 中的 `startProgressStream` 仍使用硬编码日志条目（750ms 间隔），非真实 SSE。后端 `apiSimulate` 是同步 REST 接口，暂无 SSE 端点，短期内无法修复。

### 3.5 仍存在：PnL 图表 X 轴使用假日期 ⚠️

`PnLChart.tsx` 中 X 轴仍从今天日期倒推，未使用 `simulationResult.split_date`。

### 3.6 仍存在：GP 优化后 OOS 指标映射不完整 ⚠️

`runOptimize()` 后映射的 `oos_metrics` 仅包含 `sharpe_ratio`，其余 5 项（Return/Drawdown/IC/IC-IR/Turnover）在 MetricsGrid 中显示 `—`。

---

## 4. 功能完整性评估

### 4.1 已完整实现

| 功能 | 状态 | 说明 |
|------|------|------|
| ★ 数据集注册表界面 | ✅ 新增 | DatasetView：10个数据集、分区域、日期选择、Ticker 列表 |
| ★ 数据集选择生效 | ✅ 新增 | 选择后更新 simConfig，所有 API 调用传入真实数据集参数 |
| ★ 侧边栏数据集入口 | ✅ 新增 | Database 图标 + 底部活跃数据集指示器 |
| ★ ConfigModal 数据集显示 | ✅ 新增 | 顶部展示当前数据集 + 快速跳转按钮 |
| 双视图布局（Chat/Compiler/Dataset）| ✅ | 三视图切换流畅 |
| Monaco DSL 语法高亮 + IntelliSense | ✅ | QuantDSL 自定义语言完整 |
| 多标签页 DSL 编辑 | ✅ | |
| 聊天 SSE 流式打字机 | ✅ | |
| GP 优化 SSE 进度流 | ✅ | 含 dataset 参数 |
| IS/OOS PnL 图表 | ✅ | |
| IS/OOS 指标对比表 | ✅ | |
| 回测结果自动保存台账 | ✅ | |
| 会话持久化（localStorage）| ✅ | |
| 乐观更新 | ✅ | |
| 错误分类日志（13 种颜色）| ✅ | |
| Console 行操作 | ✅ | |

### 4.2 已实现但有缺陷

| 功能 | 状态 | 问题 |
|------|------|------|
| 回测进度反馈 | ⚠️ | 假动画，与实际执行无关 |
| PnL 图表时间轴 | ⚠️ | 使用今天日期倒推，非真实回测区间 |
| GP 优化后指标展示 | ⚠️ | OOS 仅 Sharpe，其余字段为 `—` |
| IC Decay 展示 | ⚠️ | 后端当前返回空 `{}` 导致区块不渲染 |
| Alpha 台账状态点 | ⚠️ | `status` 固定显示为 active 颜色 |
| 请求取消 | ⚠️ | AbortController 未传入流式请求 |

### 4.3 完全缺失（对应后端路线图未来阶段）

| 功能 | 优先级 | 对应后端 Phase |
|------|--------|---------------|
| Walk-Forward 结果展示（多折图表）| P2 | Phase 2 Task 2.1 |
| 数据健康评分 Banner | P2 | Phase 2 Task 2.4 |
| 多 Alpha 对比视图 | P3 | Phase 3 Task 3.2 |
| Beta / 行业暴露展示 | P3 | Phase 3 Task 3.3 |
| Regime 市场状态指示器 | P4 | Phase 4 Task 4.1 |
| Deflated Sharpe 指标行 | P4 | Phase 4 Task 4.3 |
| Alpha 生命周期仪表板 | P5 | Phase 5 Task 5.4 |
| IC 衰减监控折线图 | P5 | Phase 5 Task 5.1 |
| 任务调度状态展示 | P5 | Phase 5 Task 5.3 |

---

## 5. 代码质量问题

### 5.1 useEffect 缺失依赖数组（原有）

```typescript
// WorkspaceLayout.tsx
useEffect(() => { initSessions() }, [])  // eslint-disable-line react-hooks/exhaustive-deps
```

建议将 `initSessions` 包装为 `useCallback` 使其引用稳定，或改用 `useRef` 守卫。

### 5.2 打字机队列 busy-wait 轮询（原有）

`useQuantWorkspace.ts` 中等待打字完成的 `setInterval` 轮询（50ms）可改为 Promise 回调通知。

### 5.3 Console 无自动滚动（原有）

```typescript
// ConsoleOutput.tsx — 缺少 useRef + scrollIntoView
```

当新日志追加时用户须手动滚底，应添加自动滚动逻辑。

### 5.4 PnL 图表 X 轴假日期（原有）

应使用 `simulationResult.split_date` 构建真实日期轴。

### 5.5 GP 优化后 OOS 指标缺失字段（原有）

`runOptimize()` 后 `oos_metrics` 仅映射 `sharpe_ratio`，应补全后端返回的其他字段。

### 5.6 API 无并发请求守卫（原有）

快速双击触发 `runBacktest` / `runOptimize` 存在竞态窗口，建议在 hook 层加 `in-flight ref`。

---

## 6. 与后端路线图的对齐差距

| Phase | 后端当前状态 | 前端当前状态 | 剩余前端工作 |
|-------|------------|------------|------------|
| **Phase 0** | ✅ 全部完成 | ✅ 已同步 | 数据集选择UI已实现，API调用已修复 |
| **Phase 1** | ⚠️ 部分完成 | ⚠️ 对应处理 | 429/408 响应处理待做（Task 1.5 完成后联调）|
| **Phase 2** | ❌ 未开始 | ❌ 缺 WF 展示 | Walk-Forward 多折图表组件 |
| **Phase 3** | ❌ 未开始 | ❌ 缺多Alpha | 多Alpha对比视图 + Beta/行业暴露展示 |
| **Phase 4** | ❌ 未开始 | ❌ 缺Regime | Regime 状态徽章 + DSR 指标行 |
| **Phase 5** | ❌ 未开始 | ❌ 缺仪表板 | Alpha 生命周期仪表板 + IC 历史图 |

---

## 7. 优先级改进建议

### 已完成（2026-06-07 本次实现）

- ✅ `GET /api/datasets` 后端端点
- ✅ `DatasetView.tsx` 全功能数据集选择界面
- ✅ GlobalSidebar Dataset 图标 + 活跃数据集底部指示器
- ✅ ConfigModal 数据集行 + Change 跳转
- ✅ 所有 API 调用移除 n_tickers/n_days，改为 dataset_name/dataset_start/dataset_end
- ✅ `SimulationConfig` 类型扩展 dataset/start_date/end_date
- ✅ `streamWorkflowGenerate` 激活（不再是死代码）

### P1（1-2 周，与后端 Task 1.5 同步）

#### 7.1 处理 429/408 响应

```typescript
// api/client.ts — Axios 拦截器
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

#### 7.2 Console 自动滚动

```typescript
// ConsoleOutput.tsx — 添加 useRef + useEffect
const bottomRef = useRef<HTMLDivElement>(null)
useEffect(() => {
  bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
}, [consoleLogs])
```

#### 7.3 修复 PnL 图表 X 轴

使用后端返回的 `split_date` 构建真实日期坐标轴，替代今天日期倒推逻辑。

#### 7.4 修复 GP 优化后 OOS 指标映射

在 `runOptimize()` 的 `oos_metrics` 中补全 `annualized_return`、`max_drawdown`、`ic_ir`、`ann_turnover`、`mean_ic` 等字段映射。

---

### P2（3-4 周，与后端 Phase 2 同步）

#### 7.5 Walk-Forward 结果展示组件

新建 `WalkForwardChart.tsx`：
- 每折 OOS Sharpe 柱状图
- 均值 ± 标准差标注
- 正收益折数占比

#### 7.6 数据集健康评分 Banner

在 DatasetView 或 RightPane 展示后端健康检查评分：
```
健康评分: 0.92 ✓ | NaN率: 0.3% | 覆盖: 2020-01-01 ~ 2024-01-01
```

---

### P3（6-8 周，与后端 Phase 3 同步）

#### 7.7 多 Alpha 对比视图

在 Alpha Ledger 中增加多选 checkbox，选中多个 Alpha 后触发"Compare"面板，显示 PnL 曲线叠加和相关性矩阵。

#### 7.8 Beta / 行业暴露展示

MetricsGrid 下方增加风险暴露区块，展示 `market_beta` 和 `sector_exposures`。

---

### P4（6-8 周，与后端 Phase 4 同步）

#### 7.9 Regime 状态徽章

ChatView 头部或 RightPane 新增 `BULL ↑ / BEAR ↓ / SIDEWAYS →` 状态指示器。

#### 7.10 Deflated Sharpe 在 MetricsGrid 展示

MetricsGrid 增加 DSR 行，并附注试验次数（`n_trials_run`）。

---

### P5（8-12 周，与后端 Phase 5 同步）

#### 7.11 Alpha 生命周期仪表板

GlobalSidebar 增加 Dashboard 入口，展示：
- 活跃 Alpha 卡片（含实时滚动 IC-IR）
- 衰减告警（红色卡片）
- 状态流转按钮

#### 7.12 IC 历史监控图

Alpha 详情页展示 30/60 天滚动 IC 折线图 + 衰减阈值参考线。

---

## 8. 各阶段前端任务清单

| Task | 描述 | 估时 | 优先级 | 状态 |
|------|------|------|--------|------|
| ~~FE-0.1~~ | ~~数据集选择器~~ | ~~1 天~~ | ~~🔴~~ | ✅ 已完成 |
| ~~FE-0.2~~ | ~~修改 API 调用移除合成数据参数~~ | ~~0.5 天~~ | ~~🔴~~ | ✅ 已完成 |
| ~~FE-0.3~~ | ~~激活 streamWorkflowGenerate~~ | ~~0.5 天~~ | ~~🔴~~ | ✅ 已完成 |
| FE-1.1 | 处理 429/408 响应 + toast | 0.5 天 | 🟡 | 待后端 Task 1.5 |
| FE-1.2 | Console 自动滚动 | 0.5 天 | 🟡 | ❌ |
| FE-1.3 | PnL 图表 X 轴使用真实日期 | 1 天 | 🟡 | ❌ |
| FE-1.4 | GP 优化 OOS 指标映射完整性 | 0.5 天 | 🟡 | ❌ |
| FE-1.5 | 请求 AbortController | 1 天 | 🟢 | ❌ |
| FE-2.1 | Walk-Forward 结果图表 | 2 天 | 🟡 | 待后端 Phase 2 |
| FE-2.2 | 数据集健康评分 Banner | 1 天 | 🟢 | 待后端 Phase 2 |
| FE-3.1 | 多 Alpha 对比视图 | 3 天 | 🟡 | 待后端 Phase 3 |
| FE-3.2 | Beta / 行业暴露展示 | 2 天 | 🟡 | 待后端 Phase 3 |
| FE-4.1 | Regime 状态徽章 | 1 天 | 🟢 | 待后端 Phase 4 |
| FE-4.2 | Deflated Sharpe 指标行 | 1 天 | 🟢 | 待后端 Phase 4 |
| FE-5.1 | Alpha 生命周期仪表板 | 5 天 | 🟡 | 待后端 Phase 5 |
| FE-5.2 | IC 历史监控折线图 | 2 天 | 🟡 | 待后端 Phase 5 |
| FE-5.3 | 调度任务状态展示 | 1 天 | 🟢 | 待后端 Phase 5 |
| **剩余总计** | | **~21 天** | | |

---

## 附录：前端架构合理性评分（更新版）

| 维度 | 评分（/10）| 说明 |
|------|------------|------|
| 视觉设计 | 9 | 深色 IDE 主题统一，DatasetView 与整体风格一致 |
| 组件拆分 | 8 | layout/chat/compiler/analysis/dataset 五层合理 |
| 状态管理 | 7 | Zustand 单 Store；`prevView` 处理视图返回逻辑 |
| API 层设计 | 8 | 数据集参数已正确传入；SSE 函数签名已更新（↑ 原 6）|
| 用户体验 | 8 | Dataset 视图提供直观的数据集探索和选择体验（↑ 原 7）|
| 可扩展性 | 6 | 单页应用无路由；Phase 5 多页面需求需引入 react-router |
| 类型安全 | 7 | 全量 TypeScript；部分 `as any` 类型断言需整理 |
| 与后端对齐 | 8 | 数据集参数断路已修复；SSE 调用已传 dataset 参数（↑ 原 4）|
| **综合** | **7.6** | Phase 0 前端工作完成，对齐分数从 6.8 提升至 7.6 |

---

*报告版本 v2.0 | 2026-06-07 | Dataset UI 实现后更新 | 覆盖文件数：18 个 .tsx/.ts 文件*
