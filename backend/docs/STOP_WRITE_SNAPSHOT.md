# 停写快照与恢复演练：可复跑清单

依据：复审（固定提交 21cd90b）"五、准备停写快照和恢复演练"。目的：

- 在一个**验证过的停写窗口**里取得主库、调度库和 PIT 的同一时间点副本。
- 恢复到隔离目录，逐项核对。
- 这份副本供 IC 写回预演、研究边界登记前的核对、Go/No-Go 使用。

本清单不写活库，不删活库旁的 `-wal` / `-shm`。换回活库、写回活库要你另行确认。

## 0. 涉及什么

**主库（一个文件）**

- 路径：`DATABASE_URL`，通常是 `C:\QuantAgentData\alphas.db`。
- 内容：因子、策略配置与运行槽、执行库（订单认领、在途、成交、熔断）、运行许可、告警通道故障记录、盘中巡检最近一拍、正式前向证据、冻结边界。
- 这些都在同一个 SQLite 文件里。

**调度库**

- 路径：`SCHEDULER_DB_URL`。
- 内容是任务定义，可以重建，但会一并备份。

**PIT**

- 路径：`PIT_STORE_DIR`，通常是 `C:\QuantAgentData\pit_store`。
- 前向数据买不回来，这是最不可再生的部分。

**不在产品备份里的文件**（停写窗口内要你手工复制，见第 6 节）

- 告警事件文件：`ALERT_EVENT_FILE`，或主库目录下的 `alerts\events.jsonl`。
- 验收证据：`<验收执行库目录>\simulate_acceptance.jsonl`。
- `<BACKUP_DIR>\last_backup.json`。

**必需配置**

- 工具不读 `.env`。你导出一份去掉密钥的 JSON，给 `go_no_go.py --settings-json` 用。
- 至少包括：`price_source`、`execution_mode`、`paper_dataset`、`paper_aum`、`moomoo_trd_acc_id`、`moomoo_security_firm`、`exec_limit_band_bps`、`exec_flatten_band_bps`、`exec_max_daily_loss`、`exec_max_price_deviation`、`exec_max_quote_age_days`、`scheduler_timezone`、`risk_target_vol_ann`（首轮为 0）、`intraday_watchdog_enabled`、`intraday_watchdog_stale_after_s`、`backup_dir`、`backup_keep_n`，以及 `s_test_freeze_years` / `s_val_ratio` / `s_embargo_days`。

## 1. 停写（你做）

1. 停后端，调度器跑在后端进程里：关掉 `scripts\start_forward.ps1` 起的 uvicorn 窗口。
2. 如果注册过"登录即启动"，先停用：
   ```
   Disable-ScheduledTask -TaskName QuantAgentForward
   ```
   恢复时用 `Enable-ScheduledTask`。
3. 确认没有别的进程在用库：
   ```
   Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Select-Object ProcessId, CommandLine
   ```
   - 列表里不能有 `uvicorn app.main:app`、`app.tasks.`、`tools\ops\simulate_acceptance.py`、`tools\research\`。
   - OpenD 和前端（node / vite）不写这些文件，可以开着。
4. 确认没有剩余的写入：
   ```
   Get-ChildItem C:\QuantAgentData | Select-Object Name, Length, LastWriteTime
   Get-FileHash C:\QuantAgentData\alphas.db
   ```
   - `alphas.db-wal` 必须是 0 字节或不存在。非 0 说明有未合并的提交：先把用库的进程正常关掉，不要删文件。
   - 隔 60 秒再跑一次 `Get-FileHash`，两次哈希要相同。

## 2. 取快照（窗口两端的验证都在工具里）

先取只读副本，再跑产品备份。顺序不能反，因为产品备份会对活库建普通连接。

### a. 只读副本

```
cd backend
python tools\research\factor_metrics_dry_run.py snapshot --db C:\QuantAgentData\alphas.db --out <新的空目录>
```

- 工作方式：活库不建连接，在验证过的停写窗口里逐字节复制。
- 窗口两端会比对主库、`-wal`、`-shm`、`-journal` 的哈希、大小和修改时间。任何一项变了，副本作废并报错。
- 看输出：`live_unchanged: true`、`restore.ok: true`、`restore.content_mismatches: []`、`restore.invariant_mismatches: []`。
- 快照时刻：`<目录>\preview.json` 的 `window.window_opened_at` / `window_closed_at`。

### b. 产品备份（主库 + 调度库 + PIT 同一时间点）

```
python -m app.tasks.backup
```

- 退出码要是 0。
- 打开 `<BACKUP_DIR>\snap-…\manifest.json`，确认：
  - `cross_item_consistent: true`：从取第一项到取完最后一项，各项的已提交内容都没变。
  - `quiet_window.changed_during_backup: []`。
- 如果是 `false`，说明窗口里有写入，`changed_during_backup` 会写出是哪一项。回到第 1 步。
- 注意：备份对活库建的是普通连接（`VACUUM INTO`）。最后一个连接关闭时，SQLite 可能自己合并并移除 `-wal` / `-shm`。这是 SQLite 的正常关闭，不是手动删除。

## 3. 恢复演练（不碰活库，我可以做）

```
python tools\ops\recovery_drill.py --from-snapshot <BACKUP_DIR>\snap-… --work <新的空目录>
```

通过条件：

- `ok: true`，`cross_item_consistent: true`。
- `databases.alphas.db` 下：`content_mismatches: []`、`invariant_mismatches: []`、`invariant_violations: []`。
- `pit_mismatches: []`。

`facts_after_restore` 里逐项看：

| 事实 | 看什么 |
|---|---|
| `running_strategies` | 至多一份；是你预期的那一份 |
| `orders` / `orders_in_flight` | 认领的 client_id 与状态；在途单（PENDING_SUBMIT / SUBMITTED）和停写前一致 |
| `fills` / `fill_totals_differ` | 成交增量；合计与订单成交量不同的要能说明原因 |
| `kill_switch` | 熔断状态原文。恢复后不手动清除 |
| `exec_state_keys` | `run_permit`、`alert_channel_fault`（有就要处理）、`intraday_watchdog` |
| `formal_days` / `evidence` / `signal_archive` | 正式前向天数、证据行数、信号存档最后一天 |
| `frozen_boundaries` | 登记过的研究边界 |

## 4. 恢复之后、上线之前（只有真要换回活库时）

1. 换回活库要你确认，本清单不提供命令。
2. 换回后先跑：
   ```
   python -m app.tasks.forward preflight
   ```
   对账结果要干净。
3. 对账规则：
   - 券商端有、执行库没有的单（未知订单），对账报差异并阻断调仓。你核对之后用 accept 接受。
   - 在途认领按 client_id 去重，同一决策日重跑不会重复下单。现场的 restart_dedup / crash_recovery 场景验证的就是这条路径。
4. 熔断以恢复库为准。许可如果已过期，重新签。

## 5. 结束停写

- 只有在你决定恢复运行时才做：`Enable-ScheduledTask -TaskName QuantAgentForward`，或重新运行 `scripts\start_forward.ps1`。
- 结束前再确认一次第 1 节第 4 步：结束之前没有写入，快照就代表停写前的最后状态。

## 6. 一致性：能证明的与不能证明的

| 范围 | 怎么证明 | 结论 |
|---|---|---|
| 主库内所有表（含执行、许可、熔断、证据、边界） | 同一个读事务（`VACUUM INTO`），或停写窗口逐字节副本 | 能证明 |
| 主库 + 调度库 + PIT | 清单 `cross_item_consistent`：备份全过程各项已提交内容不变 | 停写窗口里能证明。调度在跑时如实为 false |
| 只读副本（2a）与产品备份（2b）是同一时间点 | 2a 的 `copy_sha256` 等于 2b 开始时 `quiet_window.before.main_db.sha256` | 两者都在同一停写窗口里时能证明，你可以对照这两个哈希 |
| 告警事件文件、验收证据、`last_backup.json` | **不在产品备份里** | **待处理**：停写窗口里手工复制，并记下 `Get-FileHash`。无人值守许可绑定的是验收证据文件的 sha256，复制件要与许可里的一致 |
| 券商端状态 | 只能在恢复后对账时核对 | 现场 |
