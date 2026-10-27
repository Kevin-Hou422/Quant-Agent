# 从现在到无人值守前向：步骤与命令

依据：2026-10-25 审计（固定提交 eca36e3）的工作包 B / C / D 与放行三档。每一步写明谁做、用什么命令、产出什么证据。
任何一步没通过就停在那里，不跳步。正式启动由你决定；`go_no_go.py` 只给"No-Go"或"可提交你做最后决定"。

## 1. 代码（你）

1. 提交本批改动。无人值守许可绑定固定提交，工作区不能有未提交的改动。
2. 隔离全量回归：`py -3.12 backend/tools/env/isolated_test_env.py stage`，再 `... run -- -q`。

## 2. 研究边界（我出报告，你定、你登记）

```
python tools/research/research_boundary_report.py --db C:\QuantAgentData\alphas.db ^
    --assume us_broad_large=<数据起点> --data-end <数据末日> ^
    --candidate <候选起点> ... --artifacts C:\QuantAgentEnv\live-preview --out exposure_report.json
```

读曝光汇总：库里记录到的只是下界，看不到的来源写在"不确定项"里。证明不了没看过，就选未来窗口。选定后：

```
python tools/research/register_research_boundary.py --db C:\QuantAgentData\alphas.db ^
    --dataset us_broad_large --test-start <日期> --actor <你> --reason "<为什么>"          # 先预演
... --confirm                                                                         # 再写库（先备份）
```

## 3. 活库快照与预演（你停写，我在副本上做）

1. 停掉服务和调度，活库旁的 `-wal` 必须是 0 字节。不要删 `-wal` / `-shm`。
2. `python tools/research/factor_metrics_dry_run.py snapshot --db C:\QuantAgentData\alphas.db --out <新的空目录>`
   - 停写窗口内有任何写入，副本就作废并报错。
3. `... dry-run --copy <out>/restored/alphas.db --dataset us_broad_large`
   - 产出逐行的旧值 / 新值（ic_mean 与 ic_ir 分列），每条的修订理由分开列。
   - 写回活库要你另行确认。

## 4. 重提案（OpenD 开着，正式价格源 moomoo）

1. `PRICE_SOURCE=moomoo` 下先在副本上做预演：`python tools/research/reproposal_rehearsal.py --copy <副本>`。
2. 看过结果后，你在活环境批准 #4，进入 paper_running（未验证）。
3. 规格里冻结了表达式校验政策（v3）。之前批准的旧规格没有凭据，运行时会被拒绝。

## 5. 现场验收（你操作，脚本记证据）

1. 签人工联调许可：`python -m app.tasks.run_permit issue --mode supervised --exec-db <验收执行库> --account <专用账户> --hours 8 --max-order-notional 2000 --actor <你> --reason "现场验收"`
2. 按 `docs/SIMULATE_ACCEPTANCE.md` 做八个场景，每个跑 `verify`。当天没出现的就记"未覆盖"，不改预期。
3. 另外核对：盘中巡检的行情时间逐拍前进、跳空日的股数一致、桌面通知（锁屏、重启后都能看到）。

## 6. 运行保障（你）

1. `python -m app.tasks.backup` 退出码 0，并生成 `last_backup.json`。
2. `python -m app.tasks.alerts test`，看见后 `confirm`。

## 7. 无人值守许可与 Go/No-Go（你签，工具核对）

```
python -m app.tasks.run_permit issue --mode unattended --account <专用账户> --hours 72 ^
    --acceptance-evidence <simulate_acceptance.jsonl> --actor <你> --reason "<…>"
python tools/ops/go_no_go.py --db C:\QuantAgentData\alphas.db --dataset us_broad_large ^
    --exec-db <执行库> --evidence <simulate_acceptance.jsonl> --backup-dir <备份目录> ^
    --alert-events <事件文件> --settings-json <去掉密钥的配置导出> --out go_no_go.json
```

- 许可最长 7 天，到期要续签。
- 每一轮下单前，系统都会重新核对代码、规格、账户、价格源、通知、备份和验收证据；任何一项失效就停止新增风险并告警。
- 核对单里的人工确认项由你填写。全部满足后，由你决定是否启动。
