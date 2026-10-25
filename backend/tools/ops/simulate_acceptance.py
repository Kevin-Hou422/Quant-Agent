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
证据：每一步追加一行到 `<exec-db 所在目录>/simulate_acceptance.jsonl`（时间、场景、步骤、账户、代码版本、券商与执行库的状态），
`verify` 只读这份文件判定，结果也追加进去。判定"过"只说明这一次的记录满足条件，不代替你看券商端的成交回报。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

EVIDENCE_NAME = "simulate_acceptance.jsonl"
EXTERNAL_PREFIX = "acceptance-external-"

SCENARIOS: Dict[str, dict] = {
    "submit_and_fill": {
        "title": "提交与成交",
        "steps": ["收盘后 snapshot", "cycle --weight 0.01 --confirm（T 收盘后下 T+1 的 DAY 限价单）",
                  "T+1 开盘后 / 收盘后 snapshot", "T+1 收盘后 reconcile"],
        "pass": ["系统下了单", "券商侧全部成交", "执行库的成交量与券商一致", "成交后对账干净"],
    },
    "day_order_across_sessions": {
        "title": "DAY 单跨 session",
        "steps": ["T 收盘后 cycle --confirm", "T+1 开盘后 snapshot（看这张单在不在订单列表里）",
                  "T+1 收盘后 snapshot（看它是否已结束）"],
        "pass": ["下单时间在 T 收盘之后", "次一交易日的订单列表里有这张单", "次一交易日收盘后这张单已结束（成交或过期）"],
    },
    "partial_fill_then_cancel": {
        "title": "部分成交后撤单",
        "steps": ["挑一只流动性一般的票、限价贴近盘口下单（cycle 或 external-order）", "出现部分成交后 cancel",
                  "snapshot", "reconcile"],
        "pass": ["出现部分成交后撤单的订单", "执行库记下的成交量等于券商的", "撤单后对账没有差异"],
    },
    "unknown_order": {
        "title": "未知订单（不是本系统下的单）",
        "steps": ["external-order --confirm（备注不是本系统格式）", "snapshot", "reconcile",
                  "成交后再 reconcile；再 cycle --confirm 看是否被拦"],
        "pass": ["外部单没有被当成本系统的单", "外部单成交后对账报差异并阻断调仓"],
    },
    "restart_dedup": {
        "title": "重启后不重复下单",
        "steps": ["cycle --confirm", "关掉进程（本工具每次就是一个新进程）", "同一决策日再 cycle --confirm", "snapshot"],
        "pass": ["同一决策日跑了两次", "第二次没有新下单", "券商侧每个委托号只有一张单"],
    },
    "kill_switch_flatten": {
        "title": "熔断后续做全平",
        "steps": ["先有持仓（submit_and_fill 之后）", "kill-switch engage --confirm", "maintain --confirm（之后每个交易日一次）",
                  "持仓清空后 snapshot", "kill-switch reset --confirm"],
        "pass": ["熔断已开", "熔断开着时没有加仓的买单", "持仓最终清空"],
    },
    "input_anomaly": {
        "title": "输入异常被拦",
        "steps": ["cycle --ref-skew 0.3 --confirm（决策参考价比独立报价偏 30%）"],
        "pass": ["异常输入下没有下单，且被风控门或预检拦下"],
    },
    "crash_recovery": {
        "title": "下单途中崩溃后恢复",
        "steps": ["cycle --crash-after-submit --confirm（第一张单被券商接受后进程立即退出）", "新进程 reconcile",
                  "同一决策日再 cycle --confirm", "snapshot"],
        "pass": ["崩溃前已有单被券商接受", "恢复后对账没有悬而未决的单", "恢复后没有重复下单"],
    },
}


class AcceptanceRefused(RuntimeError):
    """不满足验收的安全约束。"""


def _hard_exit() -> None:
    os._exit(70)


class AcceptanceSession:
    def __init__(self, gateway, store, *, account: int, evidence_file: Path, actor: str = "",
                 now: Optional[Callable[[], datetime]] = None, max_qty: int = 5, max_notional: float = 2000.0,
                 om_factory: Optional[Callable] = None, crash: Callable[[], None] = _hard_exit) -> None:
        self.gw, self.store = gateway, store
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
               "account": self.env, "code_version": code_version(), "data": data}
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


def read_evidence(path: Path) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------- 判定（只读证据）
OPEN = "open"


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _steps(rows, *names) -> List[dict]:
    return [r for r in rows if r["step"] in names]


def _broker(row: dict, oid: str) -> Optional[dict]:
    return next((o for o in row["data"].get("broker_orders", []) if o["broker_order_id"] == oid), None)


def _store(row: dict, cid: str) -> Optional[dict]:
    return next((o for o in row["data"].get("store_orders", []) if o["client_id"] == cid), None)


def _is_open(status: str) -> bool:
    from app.core.execution.broker_gateway import is_open_status
    return is_open_status(status)


def _after(rows, row) -> List[dict]:
    """证据文件里排在 row 之后的记录（按写入顺序，不按时间戳 —— 同一时刻的两步也分得清先后）。"""
    return rows[next(i for i, r in enumerate(rows) if r is row) + 1:]


def _check_submit_and_fill(rows) -> List[dict]:
    cyc = [r for r in _steps(rows, "cycle") if r["data"]["submitted"]]
    if not cyc:
        return [_check("系统下了单", False, "没有一条下了单的 cycle 记录")]
    first = cyc[0]
    subs = first["data"]["submitted"]
    later = _steps(_after(rows, first), "snapshot", "reconcile")
    filled = next((r for r in later
                   if all((_broker(r, s["broker_order_id"]) or {}).get("status") == "FILLED_ALL" for s in subs)), None)
    out = [_check("系统下了单", True, f"{len(subs)} 张：{[s['broker_order_id'] for s in subs]}"),
           _check("券商侧全部成交", filled is not None, "" if filled else "之后的记录里没看到全部 FILLED_ALL")]
    if filled is None:
        return out
    # 执行库靠对账同步券商的成交 —— 一致性看成交之后的那次对账（对账记录里的状态是对账做完之后拍的）
    rec = _steps(([filled] if filled["step"] == "reconcile" else []) + _after(rows, filled), "reconcile")
    if not rec:
        out.append(_check("执行库的成交量与券商一致", False, "成交之后没有对账记录"))
        return out
    r0 = rec[0]
    mism = [s["client_id"] for s in subs
            if abs(float((_store(r0, s["client_id"]) or {}).get("dealt_qty") or 0)
                   - float((_broker(r0, s["broker_order_id"]) or {}).get("dealt_qty") or 0)) > 1e-9]
    out.append(_check("执行库的成交量与券商一致", not mism, f"不一致：{mism}" if mism else ""))
    clean = any(r["data"]["report"]["status"] in ("clean", "accepted") for r in rec)
    out.append(_check("成交后对账干净", clean, "" if clean else "成交之后没有一次干净的对账"))
    return out


def _check_day_order(rows) -> List[dict]:
    import pandas as pd

    from app.core.data_engine.market_calendar import (
        session_close_utc,
        session_open_utc,
        trading_days,
    )
    cyc = [r for r in _steps(rows, "cycle") if r["data"]["submitted"]]
    if not cyc:
        return [_check("下单时间在 T 收盘之后", False, "没有下了单的 cycle 记录")]
    c = cyc[0]
    d = pd.Timestamp(c["data"]["decision_date"])
    close = pd.Timestamp(session_close_utc(d))
    nxt = next(x for x in trading_days(d + pd.Timedelta(days=1), d + pd.Timedelta(days=20)) if x > d)
    nxt_open, nxt_close = pd.Timestamp(session_open_utc(nxt)), pd.Timestamp(session_close_utc(nxt))
    at = lambda r: pd.Timestamp(r["at"]).tz_convert("UTC")                      # noqa: E731
    oid = c["data"]["submitted"][0]["broker_order_id"]
    later = _steps(_after(rows, c), "snapshot", "reconcile")
    in_next = [r for r in later if nxt_open <= at(r)]
    seen = next((r for r in in_next if _broker(r, oid) is not None), None)
    ended = [r for r in later if at(r) >= nxt_close and _broker(r, oid) is not None
             and not _is_open(_broker(r, oid)["status"])]
    return [
        _check("下单时间在 T 收盘之后", at(c) >= close, f"下单 {at(c).isoformat()}，收盘 {close.isoformat()}"),
        _check("次一交易日的订单列表里有这张单", seen is not None,
               f"委托号 {oid}，次一交易日 {nxt.date()}" + ("" if seen else "：开盘之后的记录里都查不到（这正是要确认的问题）")),
        _check("次一交易日收盘后这张单已结束（成交或过期）", bool(ended),
               f"状态 {_broker(ended[0], oid)['status']}" if ended else "次一交易日收盘之后还没有看到它进入终态"),
    ]



def _check_partial_cancel(rows) -> List[dict]:
    hit = None
    for r in _steps(rows, "snapshot", "reconcile", "cancel"):
        for o in r["data"].get("broker_orders", []):
            if 0 < float(o["dealt_qty"]) < float(o["qty"]) and o["status"] == "CANCELLED_PART":
                hit = (r, o)
                break
        if hit:
            break
    if hit is None:
        return [_check("出现部分成交后撤单的订单", False, "记录里没有 0 < 成交 < 委托 且 CANCELLED_PART 的单")]
    r, o = hit
    out = [_check("出现部分成交后撤单的订单", True, f"{o['broker_order_id']}：{o['dealt_qty']}/{o['qty']}")]
    rec = _steps(([r] if r["step"] == "reconcile" else []) + _after(rows, r), "reconcile")
    if rec:
        st = next((s for s in rec[0]["data"]["store_orders"] if s["broker_order_id"] == o["broker_order_id"]), None)
        if st is not None:                               # 本系统的单：对账之后执行库要对得上；外部单只看对账
            out.append(_check("执行库记下的成交量等于券商的", abs(float(st["dealt_qty"]) - float(o["dealt_qty"])) < 1e-9,
                              f"执行库 {st['dealt_qty']} / 券商 {o['dealt_qty']}"))
    ok = bool(rec) and rec[-1]["data"]["report"]["status"] in ("clean", "accepted", "baseline")
    out.append(_check("撤单后对账没有差异", ok, rec[-1]["data"]["report"]["status"] if rec else "撤单后没有对账记录"))
    return out


def _check_unknown_order(rows) -> List[dict]:
    ext = _steps(rows, "external_order")
    if not ext:
        return [_check("外部单没有被当成本系统的单", False, "没有 external_order 记录")]
    oid = ext[0]["data"]["broker_order_id"]
    later = _after(rows, ext[0])
    adopted = [r for r in later if any(s["broker_order_id"] == oid for s in r["data"].get("store_orders", []))]
    out = [_check("外部单没有被当成本系统的单", not adopted, "被执行库收录了" if adopted else f"委托号 {oid}")]
    filled_at = next((r for r in later if float((_broker(r, oid) or {}).get("dealt_qty") or 0) > 0), None)
    if filled_at is None:
        out.append(_check("外部单成交后对账报差异并阻断调仓", False, "外部单还没成交 —— 成交之后再 reconcile / cycle"))
        return out
    after_fill = _after(rows, filled_at) + [filled_at]
    blocking = [r for r in _steps(after_fill, "reconcile") if r["data"]["report"]["status"] in ("discrepancy", "unresolved")]
    blocked_cycle = [r for r in _steps(after_fill, "cycle")
                     if r["data"]["report"].get("blocked") and not r["data"]["submitted"]]
    out.append(_check("外部单成交后对账报差异并阻断调仓", bool(blocking) and bool(blocked_cycle),
                      f"差异对账 {len(blocking)} 次、被拦的调仓 {len(blocked_cycle)} 次"))
    return out


def _check_restart_dedup(rows) -> List[dict]:
    cyc = _steps(rows, "cycle")
    by_day: Dict[str, List[dict]] = {}
    for r in cyc:
        by_day.setdefault(r["data"]["decision_date"], []).append(r)
    day = next((d for d, rs in by_day.items() if len(rs) >= 2 and rs[0]["data"]["submitted"]), None)
    if day is None:
        return [_check("同一决策日跑了两次", False, "没有同一决策日、第一次下了单的两次 cycle")]
    second = by_day[day][1]
    last = rows[-1]["data"].get("broker_orders", [])
    cids = [o["client_id"] for o in last if o["client_id"]]
    dup = sorted({c for c in cids if cids.count(c) > 1})
    return [_check("同一决策日跑了两次", True, day),
            _check("第二次没有新下单", not second["data"]["submitted"],
                   f"第二次新下 {len(second['data']['submitted'])} 张"),
            _check("券商侧每个委托号只有一张单", not dup, f"重复：{dup}" if dup else f"共 {len(set(cids))} 个备注")]


def _check_kill_switch(rows) -> List[dict]:
    eng = _steps(rows, "kill_switch_engage")
    if not eng:
        return [_check("熔断已开", False, "没有 kill_switch_engage 记录")]
    e = eng[0]
    after = _after(rows, e)
    reset_at = next((r["at"] for r in after if r["step"] == "kill_switch_reset"), None)
    during = [r for r in after if reset_at is None or r["at"] < reset_at]
    buys = [o for r in _steps(during, "cycle", "maintain") for o in r["data"]["submitted"] if o["side"] == "BUY"]
    flat = any(not {k: v for k, v in r["data"].get("positions", {}).items() if float(v["qty"])} for r in during)
    return [_check("熔断已开", bool(e["data"]["kill_switch"].get("engaged")), str(e["data"]["kill_switch"])),
            _check("熔断开着时没有加仓的买单", not buys, f"买单：{buys}" if buys else ""),
            _check("持仓最终清空", flat, "" if flat else "熔断期间的记录里持仓一直不为空")]


def _check_input_anomaly(rows) -> List[dict]:
    cyc = [r for r in _steps(rows, "cycle") if abs(float(r["data"].get("ref_skew") or 0)) > 0]
    if not cyc:
        return [_check("异常输入下没有下单，且被风控门或预检拦下", False, "没有带 --ref-skew 的 cycle 记录")]
    r = cyc[0]["data"]
    rep = r["report"]
    stopped = (rep.get("n_gate_rejected", 0) > 0) or bool(rep.get("blocked"))
    return [_check("异常输入下没有下单，且被风控门或预检拦下", not r["submitted"] and stopped,
                   f"新下 {len(r['submitted'])} 张；风控拒 {rep.get('n_gate_rejected')}；拦截 {rep.get('blocked')!r}")]


def _check_crash_recovery(rows) -> List[dict]:
    crash = _steps(rows, "crash")
    if not crash:
        return [_check("崩溃前已有单被券商接受", False, "没有 crash 记录")]
    c = crash[0]
    sub = c["data"]["submitted"][0]
    after = _after(rows, c)
    rec = _steps(after, "reconcile")
    rec_ok = bool(rec) and not rec[0]["data"]["report"]["unresolved"] \
        and (_store(rec[0], sub["client_id"]) or {}).get("broker_order_id") == sub["broker_order_id"]
    again = [r for r in _steps(after, "cycle") if r["data"]["decision_date"] == c["data"]["decision_date"]]
    cids = [o["client_id"] for o in (after[-1] if after else c)["data"].get("broker_orders", []) if o["client_id"]]
    dup = sorted({x for x in cids if cids.count(x) > 1})
    return [_check("崩溃前已有单被券商接受", bool(sub.get("broker_order_id")), str(sub)),
            _check("恢复后对账没有悬而未决的单", rec_ok,
                   (str(rec[0]["data"]["report"]["unresolved"]) if rec else "崩溃之后没有对账记录")),
            _check("恢复后没有重复下单", bool(again) and not dup and all(
                o["client_id"] != sub["client_id"] for r in again for o in r["data"]["submitted"]),
                f"同日再跑 {len(again)} 次；重复备注 {dup}")]


CHECKS = {
    "submit_and_fill": _check_submit_and_fill,
    "day_order_across_sessions": _check_day_order,
    "partial_fill_then_cancel": _check_partial_cancel,
    "unknown_order": _check_unknown_order,
    "restart_dedup": _check_restart_dedup,
    "kill_switch_flatten": _check_kill_switch,
    "input_anomaly": _check_input_anomaly,
    "crash_recovery": _check_crash_recovery,
}


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
