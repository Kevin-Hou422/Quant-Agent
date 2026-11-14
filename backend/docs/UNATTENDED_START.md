# 从现在到无人值守前向：步骤与命令

依据：

- 2026-10-25 审计（固定提交 eca36e3）的工作包 B / C / D。
- 复审（固定提交 21cd90b）的三个放行边界与 2026-10-30 用户决定。

规则：

- 每一步写明谁做、用什么命令、产出什么证据。任何一步没通过就停在那里，不跳步。
- `go_no_go.py` 分别给出四个结论：人工联调 / 正式未验证纸交易 / 无人值守 / verified_active。前三个只有"No-Go"或"可提交你做最后决定"；第四个只说满足 / 不满足。**工具永远不给 Go**，启动由你决定。

## 1. 代码（你）

当前代码核对（2026-11-14）：盘中巡检记录的存在和新鲜度、运行中代码身份、正式收益读取修订链仍需修复并验证；固定提交与现场证据齐备后再执行各级放行判断。

1. 提交本批改动。许可绑定固定提交，工作区不能有未提交的改动。
2. 隔离全量回归：`py -3.12 backend/tools/env/isolated_test_env.py stage`，再 `... run -- -q`。

## 2. 停写快照（你停写，我核对）

见 `docs/STOP_WRITE_SNAPSHOT.md`：停服务和调度 → 只读副本 → 产品备份。产品备份的清单要 `cross_item_consistent: true` → 真实快照的恢复演练。

## 3. 研究边界（我出报告，你定、你登记）

```
python tools\research\research_boundary_report.py --db C:\QuantAgentData\alphas.db ^
    --assume us_broad_large=2021-01-04 --data-end <数据末日> ^
    --candidate <候选> ... --artifacts <各预演目录> --out <报告>
python tools\research\register_research_boundary.py --db C:\QuantAgentData\alphas.db ^
    --dataset us_broad_large --test-start next-session --actor <你> --reason "未来窗口"        # 先预演
... --confirm                                                                                # 再写库（先做第 2 步）
```

- 未来窗口用 `next-session`：在**写库那一刻**按交易所日历算"登记日之后的下一个交易日"，不要照抄报告里的日期。
- 冻结日期不能清洗已经看过的数据。

## 4. IC 字段写回预演（我在副本上做，你确认后才写活库）

```
python tools\research\factor_metrics_dry_run.py writeback-plan --copy <快照>\restored\alphas.db --out <新的空目录> ^
    [--reevaluation <dry-run 报告>]
```

- **纠错**和**重评估**是两份分开的变更集，修订记录分别是 `metric_fix` 和 `metric_reeval`。
- 重评估只有在价格源是 moomoo、且边界已登记时才可执行。
- 预演会在副本上执行写回、核对、逻辑回滚、整库回滚，并输出写回活库的步骤。

## 5. 重提案（OpenD 开着，正式价格源 moomoo）

1. 在副本上预演：`python tools\research\reproposal_rehearsal.py --copy <副本>`，`PRICE_SOURCE=moomoo`，并且边界已登记。
2. 看过结果后，你核对并批准活环境中的具体规格；integration 按现场联调许可与策略状态安排。#9 保持影子观察。
3. 完成第 6 步现场验收、撤单清仓与基线确认后，由你安排进入 paper_running（未验证）及正式前向启动。
4. 两份候选此前的 B 级和门失败结果按原结论保留。

## 6. 现场验收（你操作，脚本记证据）

见 `docs/OPEND_FIELD_ACCEPTANCE.md`：验收专用执行库 + 人工联调许可 → 三个交易日完成八个场景 → 额外核对 R1 / R6 / R8、通知、断线 → 收尾清仓、撤许可、重记基线。

## 7. 运行保障（你）

1. 备份：`python -m app.tasks.backup` 退出码 0，并生成 `last_backup.json`。
2. 通知：`python -m app.tasks.alerts test`，看见后 `confirm`。以下任何一种情况，都要修好后重新做一次 test + confirm 才算就绪：
   - 有告警没写进事件文件（执行库里会记 `alert_channel_fault`）。
   - 有告警结果未知。
   - 事件文件有损坏行。

## 8. 无人值守许可与 Go/No-Go（你签，工具核对）

```
python -m app.tasks.run_permit issue --mode unattended --account <专用账户> --hours 72 ^
    --acceptance-evidence <simulate_acceptance.jsonl> --actor <你> --reason "<…>" ^
    [--waive-uncovered partial_fill_then_cancel="当天没出现部分成交"]
python tools\ops\go_no_go.py --db C:\QuantAgentData\alphas.db --dataset us_broad_large ^
    --exec-db <正式执行库> --acceptance-exec-db <验收执行库> --evidence <同一份 simulate_acceptance.jsonl> ^
    --backup-dir <备份目录> --alert-events <事件文件> --settings-json <去掉密钥的配置导出> --out go_no_go.json
```

- **签发时**，许可对验收证据逐场景重算，并要求证据与许可一致：
  - 账户（每一行都是这个账户，且是 SIMULATE）。
  - 固定代码版本。
  - 运行中策略规格里的执行合同（execution / submission）。
- **签发时**，以下情况拒签：失败、未覆盖（除非写明理由豁免）、文件损坏。
- **签发后**，许可绑定这份文件的 sha256。Go/No-Go 只认这一份文件；内容相同的另一份文件也不算。
- **每一轮下单前**，许可都会重新核对以下各项，任何一项失效就停止新增风险并告警：
  - 代码、规格、账户、价格源。
  - 验收证据：哈希没变，并且按此刻的代码 / 规格重算仍然通过。
  - 通知就绪（含告警通道故障）。
  - 备份健康。
  - 盘中巡检的最近一拍：失明、有持仓行情缺失、时间在未来、时区不可判定时停止新增风险；记录存在与新鲜度校验待修复。
- 许可最长 7 天，到期要重新核验、续签。核对单里的人工确认项由你填写；全部满足后，由你决定是否启动。

## 停止与回滚

| 要做的事 | 命令 |
|---|---|
| 立刻停止新增风险（保留对账、撤单、全平） | `python -m app.tasks.run_permit revoke --actor <你> --reason "<…>"` |
| 全平并停止调仓 | API `POST /api/execution/kill_switch/arm` → `/confirm`（两步）；验收执行库上用验收脚本 `kill-switch engage` |
| 停调度 | 关后端窗口；`Disable-ScheduledTask -TaskName QuantAgentForward` |
| 回到写回 / 换库之前 | 按 `docs/STOP_WRITE_SNAPSHOT.md` 用写回前的快照整库恢复，由你确认 |
