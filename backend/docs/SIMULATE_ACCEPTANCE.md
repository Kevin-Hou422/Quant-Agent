# 专用 SIMULATE 账户验收：步骤与证据模板

来源：2026-10-22 用户决定 D10 ——"同意人工监督、专用 SIMULATE 账户验收。先准备脚本和证据模板，连接账户、订单和时间由我安排。覆盖提交成交、DAY 跨 session、部分成交撤单、未知订单、重启去重、熔断续平、输入异常和恢复。"

脚本：`tools/ops/simulate_acceptance.py`（`plan` 打印本文同一份清单）。脚本在替身券商上的行为由 `tests/integration/test_simulate_acceptance.py` 钉住；**真券商上的行为只能在这里由人验收**。

## 开始前

- [ ] 专用 SIMULATE 账户 id：`________`（脚本要求写明 `--account`，不接受 `MOOMOO_TRD_ACC_ID=0` 自动取第一个）
- [ ] 验收专用执行库：`________`（`--exec-db`，例如 `C:\QuantAgentData\acceptance\execution.db`；不要用正式前向的执行库 —— 验收不计入策略证据）
- [ ] 调度器没有在跑（账户锁是整台机器共用的，同一账户同一时刻只有一个进程能动；脚本拿不到锁会报错，不会抢）
- [ ] 进程环境变量（不改 `.env`）：`EXECUTION_MODE=moomoo_paper`、`MOOMOO_TRD_ACC_ID=<专用账户>`、`PAPER_AUM=<给验收的切片，例如 10000>`
- [ ] `python -m app.tasks.forward preflight` 的券商侧各项通过
- [ ] 证据文件位置：执行库同目录的 `simulate_acceptance.jsonl`（每一步一行；`verify` 的结论也写进去）

下面命令都省略了公共参数：`python tools/ops/simulate_acceptance.py --account <id> --exec-db <路径> --actor <你>`。
所有下单 / 撤单 / 动熔断的命令不带 `--confirm` 时只打印要做什么。单笔上限缺省 5 股、2000 美元（`--max-qty`、`--max-notional`）。

## 场景

每个场景做完跑 `verify --scenario <场景>`，把输出的逐项结论抄进"结果"一栏；不通过的写清楚哪一项、券商端看到的是什么。

### submit_and_fill — 提交与成交

| 步骤 | 命令 / 动作 | 要看什么 |
|---|---|---|
| 1 | T 收盘后 `snapshot --scenario submit_and_fill --note 开始前` | 账户、持仓、订单列表 |
| 2 | `cycle --scenario submit_and_fill --ticker <票> --weight 0.01 --confirm` | 系统下了 DAY 限价单，委托号 |
| 3 | T+1 收盘后 `snapshot`、`reconcile` | 券商端 FILLED_ALL；对账 clean |

判定：系统下了单；券商侧全部成交；执行库的成交量与券商一致；成交后对账干净。

结果：日期 ____ 操作人 ____ 委托号 ____ verify ____ 券商端截图 ____ 备注 ____

### day_order_across_sessions — DAY 单跨 session

| 步骤 | 命令 / 动作 | 要看什么 |
|---|---|---|
| 1 | T 收盘后 `cycle --scenario day_order_across_sessions --ticker <票> --weight 0.01 --confirm` | 下单时刻在 T 收盘之后 |
| 2 | T+1 开盘后、收盘前 `snapshot --note 次日盘中` | 这张单**仍在途**（SUBMITTED 等非终态），或已成交且券商成交时间在 T+1 开盘之后 |
| 3 | T+1 收盘后 `snapshot --note 次日收盘后` | 这张单已结束（成交或 DAY 过期），没有挂到 T+2 |

判定：下单时间在 T 收盘之后；次一交易时段开始之前这张单没有结束；次一交易时段里这张单可执行（时段内观察到在途，或券商成交时间落在时段内）；次一交易日收盘后这张单已结束（成交或过期）。

2026-10-25 审计 R8：只在订单列表里**查得到**这张单不算 —— T 日已 CANCELLED_ALL 的单次日照样查得到。判定用证据记录的时刻与券商的订单更新时间（美东）：次一时段开始前任何一条记录里它已是终态 / 已有成交 → 不过；开盘即成交、没来得及看到在途的单，以券商成交时间落在次一时段内为证。若盘后提交的 DAY 单在实际账户上做不到 T+1 有效，要改提交时点和规格后重新验收，不改判定。

结果：日期 ____ 操作人 ____ 委托号 ____ 次日是否可见 ____ 最终状态 ____ verify ____ 备注 ____

### partial_fill_then_cancel — 部分成交后撤单

| 步骤 | 命令 / 动作 | 要看什么 |
|---|---|---|
| 1 | 挑流动性一般的票、限价贴近盘口：`cycle … --confirm` 或 `external-order … --confirm` | 委托号 |
| 2 | 出现部分成交后 `cancel --scenario partial_fill_then_cancel --broker-order-id <号> --confirm` | 券商端 CANCELLED_PART |
| 3 | `reconcile --scenario partial_fill_then_cancel` | 执行库成交量 = 券商成交量；对账无差异 |

判定：出现部分成交后撤单的订单；执行库记下的成交量等于券商的（本系统的单）；撤单后对账没有差异。
"出现部分成交后撤单"只认本场景下的单（`cycle` / `external-order` 提交的），且要有针对它的 `cancel` 步骤、撤单时已部分成交 —— 历史列表里的旧单不算。当天没出现部分成交就记"未覆盖"，不改预期。

结果：日期 ____ 操作人 ____ 委托号 ____ 成交 / 委托 ____ verify ____ 备注 ____

### unknown_order — 未知订单

| 步骤 | 命令 / 动作 | 要看什么 |
|---|---|---|
| 1 | `reconcile --scenario unknown_order`（先有对账基准） | |
| 2 | `external-order --scenario unknown_order --ticker <票> --side BUY --qty 1 --limit <价> --confirm`（备注不是本系统格式；也可以直接在 moomoo 客户端里下） | 委托号 |
| 3 | 成交后 `reconcile`，再 `cycle … --confirm` | 对账报差异；调仓被拦、没有新下单 |

判定：外部单没有被当成本系统的单；外部单成交后对账报差异并阻断调仓。核对券商端之后用 `accept --scenario unknown_order --reason "…" --confirm` 接受差异（作用于验收执行库；API 的 `/execution/reconcile/accept` 用的是缺省执行库，验收时不要用它），再 `cycle` 确认恢复调仓。
被拦的调仓只认对账差异（`reconcile_*`）；因没有运行许可等别的原因被拦，证明不了外部单被识别（审计 R8）。

结果：日期 ____ 操作人 ____ 外部委托号 ____ 对账状态 ____ verify ____ 备注 ____

### restart_dedup — 重启后不重复下单

| 步骤 | 命令 / 动作 | 要看什么 |
|---|---|---|
| 1 | `cycle --scenario restart_dedup --ticker <票> --weight 0.01 --confirm` | 委托号 |
| 2 | 同一决策日再跑一次同样的命令（脚本每次都是新进程 = 重启） | 第二次没有新下单 |
| 3 | `snapshot` | 券商端每个备注只有一张单 |

判定：同一决策日跑了两次；第二次是另一个进程；第二次没有新下单；券商侧每个委托号只有一张单。（每条证据记了进程号。）

结果：日期 ____ 操作人 ____ verify ____ 备注 ____

### kill_switch_flatten — 熔断后续做全平

| 步骤 | 命令 / 动作 | 要看什么 |
|---|---|---|
| 1 | 先有持仓（做完 submit_and_fill） | |
| 2 | `kill-switch engage --scenario kill_switch_flatten --reason "验收：熔断续平" --confirm` | 撤单 + 全平单 |
| 3 | 之后每个交易日收盘后 `maintain --scenario kill_switch_flatten --confirm`，直到清仓；`snapshot` | 没有买单；持仓清空 |
| 4 | `kill-switch reset --scenario kill_switch_flatten --reason "验收结束" --confirm` | |

判定：熔断已开；开熔断时确有持仓；熔断开着时没有加仓的买单；持仓最终清空。（开熔断时没有持仓，"清空"证明不了全平。）

结果：日期 ____ 操作人 ____ 清仓用了几个交易日 ____ verify ____ 备注 ____

### input_anomaly — 输入异常被拦

| 步骤 | 命令 / 动作 | 要看什么 |
|---|---|---|
| 1 | `cycle --scenario input_anomaly --ticker <票> --weight 0.01 --ref-skew 0.3 --confirm`（决策参考价比独立报价偏 30%） | 风控门拒绝（价格偏离），没有下单 |

判定：异常输入下没有下单，且被风控门或预检拦下。另可人工核对：数据源 / 规格对不上时每日循环不调仓（见每日记录的 quarantine / spec 字段）。
只认参考价与独立报价 / 持仓价对不上的拒单（`quote_mismatch` / `price_mismatch`）；被别的原因拦下不算（审计 R8）。

结果：日期 ____ 操作人 ____ 拒绝原因 ____ verify ____ 备注 ____

### crash_recovery — 下单途中崩溃后恢复

| 步骤 | 命令 / 动作 | 要看什么 |
|---|---|---|
| 1 | `cycle --scenario crash_recovery --ticker <票> --weight 0.01 --crash-after-submit --confirm`（第一张单被券商接受后进程立即退出，退出码 70） | 券商端有这张单 |
| 2 | 新进程 `reconcile --scenario crash_recovery` | 没有悬而未决的单；执行库记下了它的委托号 |
| 3 | 同一决策日再 `cycle … --confirm`；`snapshot` | 没有重复下单 |

判定：崩溃前已有单被券商接受；恢复后对账没有悬而未决的单（在另一个进程里）；恢复后没有重复下单。

结果：日期 ____ 操作人 ____ 委托号 ____ verify ____ 备注 ____

## 收尾

- [ ] 八个场景的 `verify` 结论都抄进上面的结果栏，证据文件 `simulate_acceptance.jsonl` 另存一份
- [ ] 验收期间的订单不计入任何策略的证据（用的是验收专用执行库）
- [ ] 验收结论交给 Go/No-Go 报告；正式启动由你做
