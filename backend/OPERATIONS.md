# OPERATIONS.md — 验证期运营规则

> 状态：制度文档（Task 8.3）· 适用阶段：Phase 8 起系统进入每日自动 paper trading 的验证期
> 依据：PAPER_TRADING_ROADMAP.md §五、RESEARCH_OPERATING_MODEL.md §4
> 生成日期：2026-08-19

本文件规定**因子从发现到 paper 交易再到"激活"全过程的人工与制度门槛**。它约束的是流程与
决策纪律，不是代码——代码层的状态机在 `app/db/alpha_lifecycle.py`，晋级走人工 PATCH/审批端点。

**核心边界：本系统在验证期不接入真实资金。** 见 §6。

---

## 1. 进入 PAPER 前的硬门槛（CANDIDATE → VALIDATED → PAPER）

任一因子在被放入 paper 交易前，**必须同时满足**：

1. **WalkForward ≥ 5 折全正** — 每一折的 OOS 表现为正（不是平均为正，是逐折为正）。
   工具：`WalkForwardBacktester`（`performance_analyzer.py`），端点 `/api/alphas/{id}/walk_forward`。
2. **Deflated Sharpe Ratio > 0.90** — 对多重检验/回测过拟合去膨胀后仍显著。
   工具：`deflated_sharpe_from_returns`（Bailey & López de Prado 2014）。
3. **真实数据集回测** — 用注册的真实市场数据集（US/crypto），**禁止**用合成数据下结论。
4. **红队反方报告**（见 §2）已附于该候选的谱系。

> 未来（Phase R.1）可将新因子显著性门槛由 t>2 提升至 **t≥3.0**（Harvey-Liu-Zhu），
> 并引入 PBO/CPCV 作为附加过拟合检验。届时更新本节。

晋级操作：人工经 `PATCH /api/alphas/{id}/status` 将 VALIDATED → PAPER，**这是默认的人工
决策点**（capital/风险暴露的把关）。系统不自动前向晋级。

---

## 2. 红队审计（RESEARCH_OPERATING_MODEL §4.2 落地）

**每个进入 PAPER 的候选**必须附一份**"反方报告"**，从对立视角尝试证伪该因子：

- **泄漏**：是否存在前视/未来函数/发布时点错位（尤其未来接入基本面/季频数据后）。
- **容量**：ADV 参与率、可承载 AUM；在现实规模下 alpha 是否被成本吃掉。
- **拥挤度**：是否是已知的、被广泛交易的公共因子（动量/反转/规模等）的换皮。
- **经济逻辑**：是否有可陈述的经济机制，还是纯数据挖掘产物。

**实现形式**：可用**独立 prompt 的一次性 LLM 审查步骤**产出反方报告存入谱系，**不必是常驻
agent**。该步骤只读、不改状态、不进交易回路（符合"LLM 不进交易回路 + 状态流转不作为 LLM
工具"）。达到"候选产出速度 > 人工审查带宽"时，再考虑将其升级为常驻红队 agent（见 roadmap
Phase 13.1）。

---

## 3. PAPER → ACTIVE 门槛

因子在 paper 交易中累计 **≥ 60 个交易日**后，若满足：

- realized IC 均值 **> 0**，且
- realized IC 的 **t-stat > 2**（对逐日 realized IC 序列做单样本 t 检验），

方可由**人工**经状态端点晋级 PAPER → ACTIVE。数据来源：`alpha_ic_history` 表 /
`/api/alphas/{id}/ic_history`；衰减监控由 `AlphaMonitor` + `daily_monitor_job` 自动巡检
（唯一自动流转是 ACTIVE → DECAYING 降级，前向晋级永远人工）。

---

## 4. 复盘与维护节奏

**每周人工复盘 checklist：**
- [ ] 每日摄取健康门是否有拒绝日？拒绝原因是否已排查（限流/字段变更）？
- [ ] 各 PAPER/ACTIVE 因子的 realized IC 滚动均值、连续负 IC 天数（衰减预警）。
- [ ] paper 净值曲线与 `PositionStore` 记录是否自洽；有无崩溃补跑遗留。
- [ ] 待审批队列（VALIDATED 候选 + 红队报告）是否积压。

**每月：**
- [ ] 全量回归（当前基线 `432 passed`）+ 依赖升级窗口（在此窗口内做 pip/npm 升级并跑全量）。
- [ ] 运行**成本模型校准报告**（Task 8.2，`monthly_cost_calibration_job` 自动产出
      `cost_calibration_YYYYMM.md`）；**人工确认**后方可手动更新 `CostParams`——
      **绝不自动改**，变更须留痕（写入变更日志）。
- [ ] 复核 PIT 存储是否逐日增长（Task 8.1，无人值守摄取的健康度指标之一）。

---

## 5. 数据与可复现

- 每日摄取通过健康门的数据按 `(field, date, as_of)` 追加进 PIT 存储（`pit_store/`，Task 8.1），
  历史分区**只追加不修改**——从启动日起积累无幸存者偏差的自有数据。
- 任何验证/回放固定 `as_of` 以复现历史视角；`RunManifest`（data_sha256 + git_commit + seed +
  config）记录每次关键运行，`verify_dataset` 在回放前校验数据一致。

---

## 6. 明确边界：不接入真实资金

**验证期本系统不连接任何真实资金账户，不向任何券商/交易所发送真实订单。**
paper 交易全部为内部模拟成交（`PaperBroker`，收盘价成交，复用回测成本模型）。

若未来决定接入真实资金（或先接 Alpaca 纸交易做撮合保真，见 roadmap Phase 12），**必须另立
独立规划**，至少覆盖：券商下单接入与订单状态机、实盘硬风控（日亏熔断 / fat-finger /
kill switch）、API 密钥安全管理、FastAPI 鉴权、告警升级、合规与税务。这些**不在**验证期范围内。
