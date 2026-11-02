"""
simulate_acceptance.py — 专用 SIMULATE 账户的人工监督验收：动作脚本 + 证据记录（2026-10-22 用户决定 D10）

用户原话："同意人工监督、专用 SIMULATE 账户验收。先准备脚本和证据模板，连接账户、订单和时间由我安排。
覆盖提交成交、DAY 跨 session、部分成交撤单、未知订单、重启去重、熔断续平、输入异常和恢复。"

每个场景怎么做、看什么、什么算过，见 `docs/SIMULATE_ACCEPTANCE.md`（证据模板）；`plan` 打印同一份清单。

    python tools/ops/simulate_acceptance.py --account <acc_id> --exec-db <验收专用执行库> plan
    … snapshot   --scenario <场景> [--note …]
    … cycle      --scenario <场景> --ticker AAPL --weight 0.01 [--ref-skew 0.3] [--crash-after-submit] --confirm
    … reconcile  --scenario <场景>
    … maintain   --scenario <场景> --confirm
    … kill-switch engage|reset --scenario <场景> --actor <你> --reason "…" --confirm
    … external-order --scenario unknown_order --ticker AAPL --side BUY --qty 1 --limit 150 --confirm
    … cancel     --scenario <场景> --broker-order-id <id> --confirm
    … accept     --scenario <场景> --reason "…" --confirm        （人工接受对账差异：以券商当前状态为新基准）
    … verify     --scenario <场景>

安全约束（不满足就拒绝）：
    · 网关必须是 SIMULATE（真钱环境在网关层就拒绝，`LiveTradingRefused`），且解析出的账户 id 必须等于 `--account`
      —— 专用账户要写明，不能靠 `MOOMOO_TRD_ACC_ID=0` 自动取第一个
    · 执行库要单独给（`--exec-db`）：验收不混进正式前向的执行库，也就不计入策略证据
    · 任何会下单 / 撤单 / 动熔断的动作都要 `--confirm`；不带就只打印要做什么
    · 单笔上限：`--max-qty`（缺省 5 股）、`--max-notional`（缺省 2000 美元）；目标权重 × 切片权益也受这个上限
    · 下单（cycle 与 external-order）要验收执行库里有有效的**人工联调许可**（`python -m app.tasks.run_permit issue
      --mode supervised --exec-db <验收执行库> …`；审计 R3）—— 没有、过期、账户 / 代码 / 价格源变了都不下单
证据：每一步追加一行到 `<exec-db 所在目录>/simulate_acceptance.jsonl`（时间、场景、步骤、账户、代码版本、执行合同、
券商与执行库的状态），`verify` 只读这份文件判定，结果也追加进去。判定"过"只说明这一次的记录满足条件，不代替你看券商端的成交回报。
判定逻辑在 `app/core/execution/acceptance_evidence.py`：无人值守许可的签发 / 每轮核对、Go/No-Go 用的是同一份判定。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.execution.acceptance_evidence import (  # noqa: E402,F401  判定挪进产品代码：许可与 Go/No-Go 共用
    ANOMALY_REASONS,
    CHECKS,
    SCENARIOS,
    _check_day_order,
    _check_input_anomaly,
    _check_kill_switch,
    _check_partial_cancel,
    _check_restart_dedup,
    _check_unknown_order,
    contract_of_settings,
    read_evidence,
)

EVIDENCE_NAME = "simulate_acceptance.jsonl"
EXTERNAL_PREFIX = "acceptance-external-"

class AcceptanceRefused(RuntimeError):
    """不满足验收的安全约束。"""


def _hard_exit() -> None:
    os._exit(70)


class AcceptanceSession:
    def __init__(self, gateway, store, *, account: int, evidence_file: Path, actor: str = "",
                 now: Optional[Callable[[], datetime]] = None, max_qty: int = 5, max_notional: float = 2000.0,
                 om_factory: Optional[Callable] = None, crash: Callable[[], None] = _hard_exit,
                 pid: Optional[int] = None, settings=None) -> None:
        self.gw, self.store = gateway, store
        #: 每条证据记下此刻生效的执行合同（规格的 execution / submission）：无人值守许可要求它与运行中策略的规格一致
        self.contract = contract_of_settings(settings)
        #: 写进每条证据：重启 / 崩溃恢复的场景要证明后一步真的发生在另一个进程里
        self.pid = os.getpid() if pid is None else int(pid)
        self.evidence_file = Path(evidence_file)
        self.actor = actor
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.max_qty, self.max_notional = int(max_qty), float(max_notional)
        self._om_factory = om_factory
        self._crash = crash
        desc = gateway.describe()
        if str(desc.get("trd_env")) != "SIMULATE":
            raise AcceptanceRefused(f"网关环境是 {desc.get('trd_env')}，不是 SIMULATE")
        try:
            gateway.account()                     # 解析账户 id（moomoo_trd_acc_id=0 时在这里才确定）
        except Exception as exc:
            raise AcceptanceRefused(f"连不上券商账户：{exc}") from exc
        self.env = gateway.describe()
        if int(self.env.get("acc_id") or 0) != int(account):
            raise AcceptanceRefused(f"连到的账户是 {self.env.get('acc_id')}，不是写明的专用账户 {account}")

    # ------------------------------------------------------------------ 证据
    def _record(self, scenario: str, step: str, data: dict) -> dict:
        from app.core.data_engine.input_identity import code_version
        if scenario not in SCENARIOS:
            raise AcceptanceRefused(f"未知场景 {scenario}；可选：{sorted(SCENARIOS)}")
        row = {"at": self.now().isoformat(), "scenario": scenario, "step": step, "actor": self.actor,
               "pid": self.pid, "account": self.env, "code_version": code_version(), "contract": self.contract,
               "data": data}
        self.evidence_file.parent.mkdir(parents=True, exist_ok=True)
        with self.evidence_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return row

    def _state(self) -> dict:
        from app.core.execution.order_manager import market_today
        acct = self.gw.account()
        pos = self.gw.positions()
        q = self.gw.orders(market_today(self.now()) - timedelta(days=7))
        return {
            "account": {"total_assets": acct.total_assets, "cash": getattr(acct, "cash", None)},
            "positions": {tk: {"qty": p.qty, "nominal_price": p.nominal_price} for tk, p in pos.items()},
            "history_ok": q.history_ok,
            "broker_orders": [{"broker_order_id": o.broker_order_id, "client_id": o.client_id, "ticker": o.ticker,
                               "side": o.side, "qty": o.qty, "price": o.price, "status": o.status,
                               "dealt_qty": o.dealt_qty, "create_time": o.create_time,
                               "updated_time": o.updated_time} for o in q.orders],
            "store_orders": [self._order_row(r) for r in self.store.orders()],
            "kill_switch": self.store.kill_switch(),
        }

    @staticmethod
    def _order_row(r) -> dict:
        return {"client_id": r.client_id, "decision_date": str(r.decision_date), "ticker": r.ticker, "side": r.side,
                "qty": r.qty, "status": r.status, "broker_order_id": r.broker_order_id,
                "broker_status": r.broker_status, "dealt_qty": r.dealt_qty}

    def _om(self):
        if self._om_factory is not None:
            return self._om_factory(self.gw, self.store, self.now)
        from app.core.execution.order_manager import OrderManager
        return OrderManager.from_settings(self.gw, self.store, clock=self.now)

    def _need(self, confirm: bool, what: str) -> bool:
        if not confirm:
            print(f"（预演，没有执行）{what} —— 确认无误后加 --confirm")
        return confirm

    # ------------------------------------------------------------------ 动作
    def snapshot(self, scenario: str, note: str = "") -> dict:
        return self._record(scenario, "snapshot", {"note": note, **self._state()})

    def reconcile(self, scenario: str) -> dict:
        rep = self._om().reconcile()
        return self._record(scenario, "reconcile", {"report": rep.to_dict(), **self._state()})

    def cycle(self, scenario: str, ticker: str, weight: float, *, confirm: bool = False, ref_skew: float = 0.0,
              crash_after_submit: bool = False) -> Optional[dict]:
        import pandas as pd

        from app.core.execution.order_manager import last_closed_session
        om = self._om()
        acct = self.gw.account()
        equity = om.slice_equity(acct.total_assets) or om.paper_aum
        if abs(weight) * equity > self.max_notional:
            raise AcceptanceRefused(f"目标 {weight:.2%} × 切片权益 {equity:,.0f} = {abs(weight) * equity:,.0f} 美元，"
                                    f"超过单笔上限 {self.max_notional:,.0f}")
        quote = self.gw.quotes([ticker]).get(ticker)
        if quote is None:
            raise AcceptanceRefused(f"{ticker} 取不到独立报价")
        ref = float(quote.price) * (1.0 + float(ref_skew))
        decision = last_closed_session(self.now())
        what = f"按 {decision} 决策、目标 {ticker} {weight:.2%}、参考价 {ref:.2f}（报价 {quote.price:.2f}）跑一次调仓周期"
        if not self._need(confirm, what):
            return None
        before = {r.client_id for r in self.store.orders()}
        crashed = {"after": None}
        if crash_after_submit:
            real_place = self.gw.place

            def place_then_crash(req):
                oid = real_place(req)
                crashed["after"] = {"client_id": req.client_id, "broker_order_id": oid}
                self._record(scenario, "crash", {"decision_date": str(decision), "submitted": [crashed["after"]]})
                self._crash()
                return oid
            self.gw.place = place_then_crash
        rep = om.run_cycle(pd.Series({ticker: float(weight)}), pd.Series({ticker: ref}),
                           pd.Series({ticker: 1e12}), decision)
        new = [self._order_row(r) for r in self.store.orders() if r.client_id not in before]
        return self._record(scenario, "cycle", {
            "decision_date": str(decision), "ticker": ticker, "weight": weight, "ref_price": ref,
            "quote": float(quote.price), "ref_skew": ref_skew, "report": rep.to_dict(),
            "new_orders": new, "submitted": [o for o in new if o["broker_order_id"]], **self._state()})

    def maintain(self, scenario: str, *, confirm: bool = False) -> Optional[dict]:
        if not self._need(confirm, "对账，熔断开着时续做全平"):
            return None
        before = {r.client_id for r in self.store.orders()}
        rep = self._om().maintain("simulate_acceptance")
        new = [self._order_row(r) for r in self.store.orders() if r.client_id not in before]
        return self._record(scenario, "maintain", {"report": rep.to_dict(), "new_orders": new,
                                                    "submitted": [o for o in new if o["broker_order_id"]],
                                                    **self._state()})

    def kill_switch(self, scenario: str, engage: bool, reason: str, *, confirm: bool = False) -> Optional[dict]:
        if not self.actor:
            raise AcceptanceRefused("动熔断要写明 --actor")
        if not self._need(confirm, f"{'开' if engage else '关'}熔断：{reason}"):
            return None
        om = self._om()
        out = (om.engage_kill_switch(self.actor, reason) if engage else om.reset_kill_switch(self.actor, reason))
        return self._record(scenario, "kill_switch_engage" if engage else "kill_switch_reset",
                            {"result": out, **self._state()})

    def external_order(self, scenario: str, ticker: str, side: str, qty: int, limit: float, *,
                       confirm: bool = False) -> Optional[dict]:
        from app.core.execution.broker_gateway import OrderRequest
        if not (0 < int(qty) <= self.max_qty):
            raise AcceptanceRefused(f"数量 {qty} 超出 1–{self.max_qty}")
        if int(qty) * float(limit) > self.max_notional:
            raise AcceptanceRefused(f"金额 {int(qty) * float(limit):,.0f} 超过单笔上限 {self.max_notional:,.0f}")
        req = OrderRequest(client_id=EXTERNAL_PREFIX + uuid.uuid4().hex[:12], ticker=ticker, side=side.upper(),
                           qty=int(qty), limit_price=float(limit))
        if not self._need(confirm, f"直接向券商下一张不是本系统格式的单：{req}"):
            return None
        # 2026-10-25 审计 R3：绕过 OrderManager 直接下单的入口也要有效的运行许可（验收执行库里的那份）
        from app.core.execution.run_permit import evaluate
        dec = evaluate(self.store, account=int(self.env.get("acc_id") or 0), now=self.now())
        if not dec.allowed:
            raise AcceptanceRefused("运行许可不通过：" + "; ".join(dec.reasons))
        oid = self.gw.place(req)
        return self._record(scenario, "external_order", {"request": req.__dict__, "broker_order_id": oid,
                                                          **self._state()})

    def accept(self, scenario: str, reason: str, *, confirm: bool = False) -> Optional[dict]:
        """人工接受对账差异（作用于验收专用执行库；API 的同名接口用的是缺省执行库）。"""
        if not self.actor or not str(reason).strip():
            raise AcceptanceRefused("接受对账差异要写明 --actor 与 --reason")
        if not self._need(confirm, f"以券商当前状态为新的可信基准：{reason}"):
            return None
        rep = self._om().accept_discrepancies(self.actor, reason)
        return self._record(scenario, "accept", {"report": rep.to_dict(), "reason": reason, **self._state()})

    def cancel(self, scenario: str, broker_order_id: str, *, confirm: bool = False) -> Optional[dict]:
        if not self._need(confirm, f"撤单 {broker_order_id}"):
            return None
        self.gw.cancel(str(broker_order_id))
        return self._record(scenario, "cancel", {"broker_order_id": str(broker_order_id), **self._state()})

    def verify(self, scenario: str) -> dict:
        rows = [r for r in read_evidence(self.evidence_file) if r["scenario"] == scenario and r["step"] != "verdict"]
        checks = CHECKS[scenario](rows)
        ok = bool(checks) and all(c["ok"] for c in checks)
        return self._record(scenario, "verdict", {"passed": ok, "checks": checks, "n_records": len(rows)})


def print_plan() -> None:
    for key, s in SCENARIOS.items():
        print(f"\n[{key}] {s['title']}")
        for st in s["steps"]:
            print(f"  - {st}")
        print("  判定：" + "；".join(s["pass"]))


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="专用 SIMULATE 账户的人工监督验收")
    ap.add_argument("--account", type=int, required=True, help="专用 SIMULATE 账户 id（必须写明）")
    ap.add_argument("--exec-db", required=True, help="验收专用的执行库路径（不混进正式前向的执行库）")
    ap.add_argument("--actor", default="")
    ap.add_argument("--max-qty", type=int, default=5)
    ap.add_argument("--max-notional", type=float, default=2000.0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    for name in ("snapshot", "reconcile", "maintain", "verify", "cycle", "kill-switch", "external-order", "cancel",
                 "accept"):
        p = sub.add_parser(name)
        p.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
        p.add_argument("--confirm", action="store_true")
        if name == "snapshot":
            p.add_argument("--note", default="")
        if name == "cycle":
            p.add_argument("--ticker", required=True)
            p.add_argument("--weight", type=float, required=True)
            p.add_argument("--ref-skew", type=float, default=0.0)
            p.add_argument("--crash-after-submit", action="store_true")
        if name == "kill-switch":
            p.add_argument("action", choices=["engage", "reset"])
        if name in ("kill-switch", "accept"):
            p.add_argument("--reason", required=True)
        if name == "external-order":
            p.add_argument("--ticker", required=True)
            p.add_argument("--side", required=True, choices=["BUY", "SELL"])
            p.add_argument("--qty", type=int, required=True)
            p.add_argument("--limit", type=float, required=True)
        if name == "cancel":
            p.add_argument("--broker-order-id", required=True)
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.cmd == "plan":
        print_plan()
        return 0
    from app.core.execution.broker_gateway import make_gateway_from_settings
    from app.db.execution_store import ExecutionStore
    db = Path(a.exec_db).expanduser().resolve()
    store = ExecutionStore(db_url=f"sqlite:///{db.as_posix()}")
    gw = make_gateway_from_settings()
    try:
        s = AcceptanceSession(gw, store, account=a.account, evidence_file=db.parent / EVIDENCE_NAME, actor=a.actor,
                              max_qty=a.max_qty, max_notional=a.max_notional)
        out = {
            "snapshot": lambda: s.snapshot(a.scenario, a.note),
            "reconcile": lambda: s.reconcile(a.scenario),
            "maintain": lambda: s.maintain(a.scenario, confirm=a.confirm),
            "verify": lambda: s.verify(a.scenario),
            "cycle": lambda: s.cycle(a.scenario, a.ticker, a.weight, confirm=a.confirm, ref_skew=a.ref_skew,
                                     crash_after_submit=a.crash_after_submit),
            "kill-switch": lambda: s.kill_switch(a.scenario, a.action == "engage", a.reason, confirm=a.confirm),
            "external-order": lambda: s.external_order(a.scenario, a.ticker, a.side, a.qty, a.limit, confirm=a.confirm),
            "cancel": lambda: s.cancel(a.scenario, a.broker_order_id, confirm=a.confirm),
            "accept": lambda: s.accept(a.scenario, a.reason, confirm=a.confirm),
        }[a.cmd]()
    except AcceptanceRefused as exc:
        print(f"拒绝：{exc}")
        return 2
    finally:
        close = getattr(gw, "close", None)
        if close:
            close()
        store._engine.dispose()                           # noqa: SLF001
    if out is not None:
        print(json.dumps({k: out[k] for k in ("at", "scenario", "step")} | {"data": out["data"] if a.cmd == "verify"
                                                                            else "（已记入证据文件）"},
                         ensure_ascii=False, indent=1, default=str))
        if a.cmd == "verify":
            return 0 if out["data"]["passed"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
