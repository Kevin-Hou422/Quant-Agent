# OpenD 现场验收：操作单

依据：复审（固定提交 21cd90b）"六、准备 OpenD 人工验收"。

- 每个场景的逐步命令、判定和结果栏见 `docs/SIMULATE_ACCEPTANCE.md`。本单讲顺序、前提、额外核对和收尾。
- 当天没出现的事件（例如部分成交）记"未覆盖"。不用替身或历史记录冒充现场通过。
- 无人值守许可只认**经过判定的**证据文件：八个场景逐个重算，账户、代码版本、执行合同要一致。
- 未覆盖的场景只能由你在签无人值守许可时写明理由豁免（`--waive-uncovered 场景=理由`）。失败的场景不能豁免。

## 1. 状态机与顺序（联调证据不进正式前向天数）

| 阶段 | 策略状态 | 执行库 | 许可 | 计入正式前向天数 |
|---|---|---|---|---|
| A 现场验收（八场景） | 不涉及策略 | **验收专用执行库**（`--exec-db`） | 验收执行库里的**人工联调许可**（supervised，≤ 24 小时、单笔上限） | 否：订单与证据都在验收库里 |
| B 工程联调（可选） | `approved → integration` | 正式执行库 | 正式执行库里的 supervised 许可，每天续签 | 否：integration 的权限就是不计证据、不计正式天数 |
| C 正式但未验证的纸交易 | `approved / integration → paper_running`（你批准） | 正式执行库 | supervised 每天续签（人工值守）；或满足条件后签 unattended | 是：从进入 paper_running 起 |
| D 无人值守 | 仍是 `paper_running` | 正式执行库 | unattended（≤ 7 天，到期重新核验、续签） | 是 |
| E 已验证 | `paper_running → verified_active`（只能经验证门） | 正式执行库 | 同上 | 是 |

要点：

- A 与 C 用的是**同一个专用 SIMULATE 账户**、**不同的执行库**。A 收尾时必须把验收留下的持仓清掉、在途单撤掉（第 6 节）。否则 C 第一次对账会把这些持仓当成基准，混进正式切片。
- B 可以跳过：approved 直接 paper_running。B 的订单是真实的 SIMULATE 订单，但不计证据。
- integration 转 paper_running 时，持仓是同一策略、同一规格的，延续下去。

## 2. 开始前（你填写）

- [ ] 固定代码提交 SHA：`________`。工作区干净，`git status` 无输出。
- [ ] OpenD 版本：`________`（OpenD 窗口"关于"）。
- [ ] SDK 版本：`________`。在后端 venv 里运行：`python -c "import moomoo; print(moomoo.__version__)"`。
- [ ] 部署路径：`________`。Windows 运行用户：`________`。
- [ ] 操作人：`________`。
- [ ] 专用 SIMULATE 账户 id：`________`。必须写明，不用 `MOOMOO_TRD_ACC_ID=0`。
- [ ] 账户基线：现金 `________`、持仓 `________`（最好为空）、近期出入金 `________`。没有不明现金流。
- [ ] 验收执行库路径：`________`。不能是正式执行库，Go/No-Go 会核对两者不同。
- [ ] 验收切片：`PAPER_AUM=________`（例如 10000）。单笔上限 `--max-notional 2000`、`--max-qty 5`。
- [ ] 本次进程的环境变量（只在当前 PowerShell 窗口设，不改 `.env`）：
  `$env:EXECUTION_MODE="moomoo_paper"; $env:MOOMOO_TRD_ACC_ID="<id>"; $env:PAPER_AUM="10000"; $env:PRICE_SOURCE="moomoo"`
- [ ] 调度器没有在跑，后端停着。账户锁整机共用，脚本拿不到锁会报错，不会抢。
- [ ] 券商侧预检通过：`python -m app.tasks.forward preflight`。
- [ ] 签人工联调许可：
  ```
  python -m app.tasks.run_permit issue --mode supervised --exec-db <验收执行库> --account <id> --hours 8 --max-order-notional 2000 --actor <你> --reason "现场验收"
  ```
  每天开工前重签，最长 24 小时。

公共参数：`python tools\ops\simulate_acceptance.py --account <id> --exec-db <验收执行库> --actor <你>`。

## 3. 三个交易日的安排

每个场景用**不同的票**，避免互相干扰。证据统一写在验收执行库同目录的 `simulate_acceptance.jsonl`。

| 时点 | 做什么 |
|---|---|
| T 收盘后 | `submit_and_fill`、`day_order_across_sessions`、`restart_dedup`（两次 cycle）、`crash_recovery`（含新进程 reconcile 与再 cycle）、`input_anomaly`、`unknown_order`（先 reconcile，再 external-order） |
| T+1 盘中 | `day_order_across_sessions` 盘中 snapshot。`partial_fill_then_cancel`：挑流动性一般的票，限价贴盘口下单，部分成交后 cancel。额外核对 R6（第 4 节） |
| T+1 收盘后 | 各场景 snapshot / reconcile。`unknown_order` 成交后 reconcile 再 cycle，看是否被拦。`kill_switch_flatten`：有持仓后 engage |
| T+2 收盘后 | `kill_switch_flatten` maintain，直到清空，然后 reset |
| 每个场景做完 | `verify --scenario <场景>`，结论抄进 `SIMULATE_ACCEPTANCE.md` 的结果栏 |

失败判定以 `verify` 的逐项结论为准。脚本判"过"只说明这次记录满足条件，你还要在券商端核对成交回报。

## 4. 额外核对（现场才能证明）

| 编号 | 核对 | 怎么做 | 证据 |
|---|---|---|---|
| R1 | 跳空日：券商成交股数 = 冻结委托股数 | 挑 T+1 有明显跳空的票（`submit_and_fill` 或另一张 cycle）。比较 cycle 记录 `new_orders[].qty`、券商订单 `qty`、成交后 `dealt_qty`：委托股数在 T 收盘冻结，跳空不改股数；成交量 ≤ 委托量 | 证据文件对应行 + 券商成交截图 |
| R6 | 盘中行情时间逐拍前进 | T+1 盘中隔 1 分钟跑 3 次：`python -c "from app.core.execution.broker_gateway import make_gateway_from_settings as m; g=m(); print({k: (q.price, q.asof) for k, q in g.quotes(['US.<持仓票>', 'US.<另一只>']).items()}); g.close()"`。每只持仓的 `asof` 都要前进，价格不变而时间在走也算新鲜 | 三次输出的 asof |
| R8 | 盘后提交的 DAY 单在实际账户上 T+1 有效 | 就是 `day_order_across_sessions`：次一时段内观察到在途，或成交时间落在时段内；T 日已结束的不算 | verify 结论 |
| 通知 | 实际启动方式、同一 Windows 用户、锁屏、重启后都看得到 | 用 `scripts\start_forward.ps1` 起后端后跑 `python -m app.tasks.alerts test`，看见后 `confirm <事件 id> --actor <你>`。锁屏：`Start-Sleep 60; python -m app.tasks.alerts test` 后立刻 Win+L，解锁后在通知中心找到它。重启后再做一次 | 事件 id；`alerts status` 显示就绪 |
| 断线 / 重连 | 盘中关掉 OpenD | 巡检连续 N 拍后发 critical"连不上券商"；执行库 `intraday_watchdog` 记 `unreachable`，无人值守许可此时不新增风险。重开 OpenD 后下一拍发 info"恢复"，记录回到 fresh | `alerts status` + `run_permit status` |
| 重启 / 认领 / 恢复 | — | 由 `restart_dedup`、`crash_recovery` 两个场景覆盖（同一决策日重跑不重复下单；崩溃后新进程对账认领到委托号） | verify 结论 |

## 5. 不能当作现场通过的东西

- 替身券商上的用例（`tests/integration/test_simulate_acceptance.py`）只证明脚本本身。
- 历史订单列表里查得到的旧单。
- 别的原因拦下的调仓：没有许可、日历不可用，都不能当作"外部单被识别"或"异常输入被拦"。
- 部分成交、跳空当天没出现：记"未覆盖"。签无人值守许可时你可以写理由豁免"未覆盖"，Go/No-Go 会单列为"可接受的限制"。

## 6. 收尾（转入正式前向之前必须做）

- [ ] 八个场景的 `verify` 结论都抄进结果栏。`simulate_acceptance.jsonl` 另存一份，并记下 `Get-FileHash`。无人值守许可绑定的就是这个文件和这个哈希，之后不要改它。
- [ ] 验收执行库里没有在途单：`reconcile` 之后 `orders_in_flight` 为空。
- [ ] 熔断已 reset，在验收执行库里。
- [ ] 账户清仓、撤掉所有挂单，然后重新记一次账户基线（现金、持仓应为空）。这是正式执行库第一次对账的起点。
- [ ] 撤销验收许可：`python -m app.tasks.run_permit revoke --exec-db <验收执行库> --actor <你> --reason "验收结束"`。
- [ ] 跑一次 Go/No-Go，看前两个结论：
  ```
  python tools\ops\go_no_go.py --db <主库> --dataset us_broad_large --exec-db <正式执行库> --acceptance-exec-db <验收执行库> --evidence <simulate_acceptance.jsonl> --backup-dir <备份目录> --alert-events <事件文件> --settings-json <配置导出>
  ```
