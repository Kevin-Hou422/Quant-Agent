"""
acceptance_evidence.py — 现场验收证据的**唯一**判定：验收脚本的 verify、无人值守许可的签发与每轮核对、Go/No-Go 都用它

复审（固定提交 21cd90b）原话："当前代码看起来只要求验收文件存在并记录哈希；Go/No-Go 分别读取许可和所提供的验收文件，
还需要证明两者是同一份有效证据。"要求："验收结果应由统一的验证逻辑生成，检查所有必需场景的真实结果；验证证据与当前固定
代码、专用账户、运行模式、验收范围和相关规格一致；许可绑定经过验证的证据文件及哈希；Go/No-Go 检查的是同一份文件；失败、
未覆盖、版本或账户不一致、文件变更时，不能自动认定无人值守就绪。"

`assess` 只读证据文件，逐场景**重算**判定（不信文件里记的结论），再与最后一次 verify 记下的结论对照：

    passed        重算通过，且最后一次 verify 记下的也是通过
    failed        重算不通过；或记下的结论与重算不一致
    not_covered   这个场景没有记录、前提事件没发生（没下单、当天没出现部分成交……），或还没跑 verify
    waived        not_covered 且签发许可的人写明了理由豁免（失败的场景不能豁免）

整份证据还要满足（任何一条不满足 → problems，整体不通过）：
    · 文件存在、每行是完整的 JSON 记录（损坏 / 截断 / 缺字段都算）
    · 每一行：SIMULATE、账户 = 许可的账户、代码版本 = 许可绑定的固定版本（不能 +dirty / unknown）、
      执行合同 = 运行中策略规格里的 execution / submission（`contract_of_spec`）
    · 没有不认识的场景

执行合同不含 execution.account_id：那是配置里的值（可能是 0 = 自动取），账户由每行实际连上的 acc_id 单独核对。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

#: 八个场景：做什么、什么算过（`docs/SIMULATE_ACCEPTANCE.md` 是同一份清单的证据模板）
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
        "pass": ["下单时间在 T 收盘之后", "次一交易时段开始之前这张单没有结束",
                 "次一交易时段里这张单可执行（时段内观察到在途，或券商成交时间落在时段内）",
                 "次一交易日收盘后这张单已结束（成交或过期）"],
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
        "pass": ["同一决策日跑了两次", "第二次是另一个进程", "第二次没有新下单", "券商侧每个委托号只有一张单"],
    },
    "kill_switch_flatten": {
        "title": "熔断后续做全平",
        "steps": ["先有持仓（submit_and_fill 之后）", "kill-switch engage --confirm", "maintain --confirm（之后每个交易日一次）",
                  "持仓清空后 snapshot", "kill-switch reset --confirm"],
        "pass": ["熔断已开", "开熔断时确有持仓", "熔断开着时没有加仓的买单", "持仓最终清空"],
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
        "pass": ["崩溃前已有单被券商接受", "恢复后对账没有悬而未决的单（在另一个进程里）", "恢复后没有重复下单"],
    },
}


def read_evidence(path: Path) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------- 判定（只读证据）
OPEN = "open"


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _premise(name: str, detail: str) -> dict:
    """场景的前提事件没发生（没下单、没出现部分成交……）：判不过，且归为"未覆盖"而不是"失败"。"""
    return {"name": name, "ok": False, "detail": detail, "premise": True}


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
        return [_premise("系统下了单", "没有一条下了单的 cycle 记录")]
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


#: 券商订单的创建 / 更新时间是交易所本地时间（moomoo 美股：美东）
BROKER_TZ = "America/New_York"


def _broker_time(s):
    """券商时间串 → UTC；空 / 解析不了 → None（证明不了时序就不能算过）。"""
    import pandas as pd
    try:
        t = pd.Timestamp(str(s or "").strip())
    except (ValueError, TypeError) as exc:
        logger.warning("[acceptance] 券商时间 %r 解析不了（这条时序证明不了）: %s", s, exc)
        return None
    if pd.isna(t):
        return None
    return (t.tz_localize(BROKER_TZ) if t.tzinfo is None else t).tz_convert("UTC")


def _check_day_order(rows) -> List[dict]:
    """
    2026-10-25 审计 R8："订单在 T 日已 CANCELLED_ALL，次日开盘及收盘快照中只是仍能查到这条历史订单，脚本三项检查
    却全部通过。"查询结果里出现了委托号只说明历史可见。现在要求用时序证明这张单在次一交易时段**可执行**：
        · 次一时段开始前的任何记录里它都没进入终态（T 日已取消 / 过期 / 失败 → 不过）
        · 次一时段内（开盘 ≤ 记录时刻 < 收盘）观察到它仍在途；或者券商记的成交（更新时间）落在次一时段内、
          且开盘前没有任何成交 —— 开盘即成交、没来得及看到在途的单也能证明
        · 次一交易日收盘后它已结束（成交或过期）
    """
    import pandas as pd

    from app.core.data_engine.market_calendar import (
        session_close_utc,
        session_open_utc,
        trading_days,
    )
    cyc = [r for r in _steps(rows, "cycle") if r["data"]["submitted"]]
    if not cyc:
        return [_premise("下单时间在 T 收盘之后", "没有下了单的 cycle 记录")]
    c = cyc[0]
    d = pd.Timestamp(c["data"]["decision_date"])
    close = pd.Timestamp(session_close_utc(d))
    nxt = next(x for x in trading_days(d, d + pd.Timedelta(days=20)) if x > d)
    nxt_open, nxt_close = pd.Timestamp(session_open_utc(nxt)), pd.Timestamp(session_close_utc(nxt))
    at = lambda r: pd.Timestamp(r["at"]).tz_convert("UTC")                      # noqa: E731
    oid = c["data"]["submitted"][0]["broker_order_id"]
    views = [(r, _broker(r, oid)) for r in [c] + _steps(_after(rows, c), "snapshot", "reconcile")]
    views = [(r, o) for r, o in views if o is not None]

    def ended_before_open(r, o):
        if _is_open(o["status"]):
            return False
        upd = _broker_time(o.get("updated_time"))
        return at(r) < nxt_open or (upd is not None and upd < nxt_open)
    early_end = next(((r, o) for r, o in views if ended_before_open(r, o)), None)
    filled_early = next(((r, o) for r, o in views if at(r) < nxt_open and float(o.get("dealt_qty") or 0) > 0), None)
    open_in_session = next((r for r, o in views if nxt_open <= at(r) < nxt_close and _is_open(o["status"])), None)
    fill_in_session = next((o for r, o in views if float(o.get("dealt_qty") or 0) > 0
                            and (lambda u: u is not None and nxt_open <= u < nxt_close)(_broker_time(o.get("updated_time")))),
                           None)
    executable = early_end is None and filled_early is None and (open_in_session is not None or fill_in_session is not None)
    if early_end is not None:
        why = (f"{early_end[1]['status']}（记录 {at(early_end[0]).isoformat()}，券商更新 {early_end[1].get('updated_time')}）"
               f"在次一时段 {nxt_open.isoformat()} 开始之前就结束了 —— 只是历史可见，不是跨日有效")
    elif filled_early is not None:
        why = f"开盘前的记录里已有成交 {filled_early[1].get('dealt_qty')}：不是次一时段的成交"
    elif open_in_session is not None:
        why = f"次一时段内 {at(open_in_session).isoformat()} 观察到在途"
    elif fill_in_session is not None:
        why = f"券商成交时间 {fill_in_session.get('updated_time')}（美东）落在次一时段内"
    else:
        why = "次一时段内既没观察到在途，也没有时间落在时段内的成交"
    ended = [r for r, o in views if at(r) >= nxt_close and not _is_open(o["status"])]
    return [
        _check("下单时间在 T 收盘之后", at(c) >= close, f"下单 {at(c).isoformat()}，收盘 {close.isoformat()}"),
        _check("次一交易时段开始之前这张单没有结束", early_end is None and filled_early is None,
               f"委托号 {oid}，次一交易日 {nxt.date()}；{why}"),
        _check("次一交易时段里这张单可执行（时段内观察到在途，或券商成交时间落在时段内）", executable, why),
        _check("次一交易日收盘后这张单已结束（成交或过期）", bool(ended),
               f"状态 {_broker(ended[0], oid)['status']}" if ended else "次一交易日收盘之后还没有看到它进入终态"),
    ]


def _scenario_orders(rows) -> set:
    """这个场景里**自己下的**单（cycle / maintain 提交的、external_order 下的）—— 历史列表里别的单不算。"""
    ids = {o["broker_order_id"] for r in _steps(rows, "cycle", "maintain") for o in r["data"].get("submitted", [])}
    ids |= {r["data"]["broker_order_id"] for r in _steps(rows, "external_order")}
    return {i for i in ids if i}


def _check_partial_cancel(rows) -> List[dict]:
    """
    部分成交、撤单要真的在这个场景里发生（审计 R8：不能让历史列表里的一张旧单、或后来一次干净对账顶替）：
    单是本场景下的；有针对它的 cancel 步骤；撤单那一刻之前（含）它已部分成交；之后它是 CANCELLED_PART。
    """
    mine = _scenario_orders(rows)
    hit = None
    for cr in _steps(rows, "cancel"):
        oid = cr["data"]["broker_order_id"]
        if oid not in mine:
            continue
        upto = rows[:next(i for i, r in enumerate(rows) if r is cr) + 1]
        dealt_before = any(0 < float((_broker(r, oid) or {}).get("dealt_qty") or 0) for r in upto)
        o = next((_broker(r, oid) for r in [cr] + _after(rows, cr)
                  if (_broker(r, oid) or {}).get("status") == "CANCELLED_PART"), None)
        if dealt_before and o is not None and 0 < float(o["dealt_qty"]) < float(o["qty"]):
            hit = (cr, o)
            break
    if hit is None:
        return [_premise("出现部分成交后撤单的订单",
                       "本场景下的单里没有『撤单前已部分成交、撤单后 CANCELLED_PART』的 —— 当天没出现就记未覆盖")]
    r, o = hit
    out = [_check("出现部分成交后撤单的订单", True, f"{o['broker_order_id']}：{o['dealt_qty']}/{o['qty']}")]
    rec = _steps(_after(rows, r), "reconcile")          # r 是 cancel 那一步：对账只看它之后的
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
        return [_premise("外部单没有被当成本系统的单", "没有 external_order 记录")]
    oid = ext[0]["data"]["broker_order_id"]
    later = _after(rows, ext[0])
    adopted = [r for r in later if any(s["broker_order_id"] == oid for s in r["data"].get("store_orders", []))]
    out = [_check("外部单没有被当成本系统的单", not adopted, "被执行库收录了" if adopted else f"委托号 {oid}")]
    filled_at = next((r for r in later if float((_broker(r, oid) or {}).get("dealt_qty") or 0) > 0), None)
    if filled_at is None:
        out.append(_premise("外部单成交后对账报差异并阻断调仓", "外部单还没成交 —— 成交之后再 reconcile / cycle"))
        return out
    after_fill = _after(rows, filled_at) + [filled_at]
    blocking = [r for r in _steps(after_fill, "reconcile") if r["data"]["report"]["status"] in ("discrepancy", "unresolved")]
    # 只认"因对账差异被拦"（审计 R8：别的原因 —— 没有运行许可、日历不可用 —— 拦下的调仓证明不了外部单被识别）
    blocked_cycle = [r for r in _steps(after_fill, "cycle")
                     if str(r["data"]["report"].get("blocked") or "").startswith("reconcile_")
                     and not r["data"]["submitted"]]
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
        return [_premise("同一决策日跑了两次", "没有同一决策日、第一次下了单的两次 cycle")]
    first, second = by_day[day][0], by_day[day][1]
    last = rows[-1]["data"].get("broker_orders", [])
    cids = [o["client_id"] for o in last if o["client_id"]]
    dup = sorted({c for c in cids if cids.count(c) > 1})
    return [_check("同一决策日跑了两次", True, day),
            _check("第二次是另一个进程", first.get("pid") is not None and first.get("pid") != second.get("pid"),
                   f"进程 {first.get('pid')} → {second.get('pid')}"),
            _check("第二次没有新下单", not second["data"]["submitted"],
                   f"第二次新下 {len(second['data']['submitted'])} 张"),
            _check("券商侧每个委托号只有一张单", not dup, f"重复：{dup}" if dup else f"共 {len(set(cids))} 个备注")]


def _check_kill_switch(rows) -> List[dict]:
    eng = _steps(rows, "kill_switch_engage")
    if not eng:
        return [_premise("熔断已开", "没有 kill_switch_engage 记录")]
    e = eng[0]
    held = {k: v for k, v in e["data"].get("positions", {}).items() if float(v["qty"])}
    if not held:
        return [_check("熔断已开", bool(e["data"]["kill_switch"].get("engaged")), str(e["data"]["kill_switch"])),
                _premise("开熔断时确有持仓", "开熔断时没有持仓 —— 『持仓最终清空』证明不了全平（先做 submit_and_fill）")]
    after = _after(rows, e)
    reset_at = next((r["at"] for r in after if r["step"] == "kill_switch_reset"), None)
    during = [r for r in after if reset_at is None or r["at"] < reset_at]
    buys = [o for r in _steps(during, "cycle", "maintain") for o in r["data"]["submitted"] if o["side"] == "BUY"]
    flat = any(not {k: v for k, v in r["data"].get("positions", {}).items() if float(v["qty"])} for r in during)
    return [_check("熔断已开", bool(e["data"]["kill_switch"].get("engaged")), str(e["data"]["kill_switch"])),
            _check("开熔断时确有持仓", True, str({k: v["qty"] for k, v in held.items()})),
            _check("熔断开着时没有加仓的买单", not buys, f"买单：{buys}" if buys else ""),
            _check("持仓最终清空", flat, "" if flat else "熔断期间的记录里持仓一直不为空")]


#: input_anomaly 场景认可的拒单原因：决策参考价与独立报价 / 券商持仓价对不上
ANOMALY_REASONS = frozenset({"quote_mismatch", "price_mismatch"})


def _check_input_anomaly(rows) -> List[dict]:
    cyc = [r for r in _steps(rows, "cycle") if abs(float(r["data"].get("ref_skew") or 0)) > 0]
    if not cyc:
        return [_premise("异常输入下没有下单，且被风控门或预检拦下", "没有带 --ref-skew 的 cycle 记录")]
    r = cyc[0]["data"]
    rep = r["report"]
    # 只认参考价与独立报价对不上这一类原因（审计 R8：被别的原因拦下 —— 没有运行许可、对账差异 —— 证明不了输入异常被识别）
    reasons = [x.get("reason") for x in ((rep.get("gate") or {}).get("rejected") or [])]
    stopped = bool(reasons) and all(x in ANOMALY_REASONS for x in reasons)
    return [_check("异常输入下没有下单，且被风控门或预检拦下", not r["submitted"] and stopped,
                   f"新下 {len(r['submitted'])} 张；拒单原因 {reasons}；拦截 {rep.get('blocked')!r}")]


def _check_crash_recovery(rows) -> List[dict]:
    crash = _steps(rows, "crash")
    if not crash:
        return [_premise("崩溃前已有单被券商接受", "没有 crash 记录")]
    c = crash[0]
    sub = c["data"]["submitted"][0]
    after = _after(rows, c)
    rec = _steps(after, "reconcile")
    rec_ok = bool(rec) and not rec[0]["data"]["report"]["unresolved"] \
        and (_store(rec[0], sub["client_id"]) or {}).get("broker_order_id") == sub["broker_order_id"] \
        and rec[0].get("pid") is not None and rec[0].get("pid") != c.get("pid")
    again = [r for r in _steps(after, "cycle") if r["data"]["decision_date"] == c["data"]["decision_date"]]
    cids = [o["client_id"] for o in (after[-1] if after else c)["data"].get("broker_orders", []) if o["client_id"]]
    dup = sorted({x for x in cids if cids.count(x) > 1})
    return [_check("崩溃前已有单被券商接受", bool(sub.get("broker_order_id")), str(sub)),
            _check("恢复后对账没有悬而未决的单（在另一个进程里）", rec_ok,
                   (f"未决 {rec[0]['data']['report']['unresolved']}；进程 {c.get('pid')} → {rec[0].get('pid')}"
                    if rec else "崩溃之后没有对账记录")),
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


# ---------------------------------------------------------------------- 整份证据的判定
#: 无人值守要求的验收范围：全部场景
REQUIRED = tuple(SCENARIOS)
PASSED, FAILED, NOT_COVERED, WAIVED = "passed", "failed", "not_covered", "waived"
_ROW_KEYS = ("at", "scenario", "step", "account", "code_version", "data")


def contract_digest(execution: dict, submission: dict) -> str:
    ex = {k: v for k, v in (execution or {}).items() if k != "account_id"}
    blob = json.dumps({"execution": ex, "submission": submission or {}}, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def contract_of_settings(settings=None) -> str:
    """验收时实际生效的执行合同（与规格里的 execution / submission 同一装配）。"""
    from app.core.portfolio_manager.executable_spec import _execution_section, _submission_section
    if settings is None:
        from app.config import settings
    return contract_digest(_execution_section(settings), _submission_section(settings))


def contract_of_spec(spec: Optional[dict]) -> Optional[str]:
    if not isinstance(spec, dict) or "execution" not in spec or "submission" not in spec:
        return None
    return contract_digest(spec["execution"], spec["submission"])


def file_sha256(path) -> Optional[str]:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def _load(path) -> tuple:
    p = Path(path)
    if not p.is_file():
        return [], [f"证据文件不存在：{p}"]
    try:
        text = p.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        logger.warning("[acceptance] 证据文件 %s 不是 UTF-8: %s", p, exc)
        return [], [f"证据文件不是 UTF-8 文本：{exc}"]
    rows, problems = [], []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError as exc:
            logger.warning("[acceptance] 证据文件 %s 第 %d 行损坏: %s", p, i, exc)
            problems.append(f"第 {i} 行不是完整的 JSON（损坏或截断）")
            continue
        if not isinstance(r, dict) or any(k not in r for k in _ROW_KEYS) \
                or not isinstance(r["account"], dict) or not isinstance(r["data"], dict):
            problems.append(f"第 {i} 行缺字段（要有 {', '.join(_ROW_KEYS)}）")
            continue
        rows.append(r)
    return rows, problems


def _acc(r) -> Optional[int]:
    v = r["account"].get("acc_id")
    return int(v) if isinstance(v, int) or (isinstance(v, str) and v.strip().isdigit()) else None


def _judge(scenario: str, rows: List[dict]) -> dict:
    mine = [r for r in rows if r["scenario"] == scenario and r["step"] != "verdict"]
    recorded = [r for r in rows if r["scenario"] == scenario and r["step"] == "verdict"]
    if not mine:
        return {"status": NOT_COVERED, "detail": "没有这个场景的记录", "checks": []}
    try:
        checks = CHECKS[scenario](mine)
    except Exception as exc:                            # 记录内容不全：如实判不过，不吞掉
        logger.warning("[acceptance] %s 判定时出错: %r", scenario, exc)
        return {"status": FAILED, "detail": f"判定时出错（记录内容不全？）：{exc!r}", "checks": []}
    fails = [c for c in checks if not c["ok"]]
    if any(c.get("premise") for c in fails):
        return {"status": NOT_COVERED, "detail": "前提没发生：" + "；".join(c["detail"] for c in fails if c.get("premise")),
                "checks": checks}
    if fails or not checks:
        return {"status": FAILED, "detail": "未过：" + "；".join(c["name"] for c in fails), "checks": checks}
    if not recorded:
        return {"status": NOT_COVERED, "detail": "记录齐了但还没跑 verify", "checks": checks}
    last = recorded[-1]["data"]
    if not last.get("passed"):
        return {"status": FAILED, "detail": "最后一次 verify 记下的是不通过（与重算不一致）", "checks": checks}
    return {"status": PASSED, "detail": f"{len(mine)} 条记录；最后一次 verify {recorded[-1]['at']}", "checks": checks}


def assess(path, *, account: int, code: str, contract: Optional[str], required=REQUIRED,
           waived: Optional[Dict[str, str]] = None) -> dict:
    """整份证据能不能支撑无人值守（见模块说明）。只读文件。"""
    waived = {str(k): str(v) for k, v in (waived or {}).items()}
    rows, problems = _load(path)
    sha = file_sha256(path)
    if sha is not None and not rows and not problems:
        problems.append("证据文件是空的")
    if not code or code == "unknown" or code.endswith("+dirty"):
        problems.append(f"代码版本 {code!r} 不是固定提交")
    envs = sorted({str(r["account"].get("trd_env")) for r in rows if r["account"].get("trd_env") != "SIMULATE"})
    if envs:
        problems.append(f"有记录不是在 SIMULATE 环境下做的：{envs}")
    accs = sorted({str(r["account"].get("acc_id")) for r in rows if _acc(r) != int(account)})
    if accs:
        problems.append(f"有记录来自别的账户：{accs}（许可的账户是 {account}）")
    codes = sorted({str(r["code_version"]) for r in rows if r["code_version"] != code})
    if codes:
        problems.append(f"有记录来自别的代码版本：{codes}（要求 {code}）")
    if contract is None:
        problems.append("没有运行中策略的执行合同（规格的 execution / submission）—— 核对不了验收与规格是否一致")
    else:
        cs = sorted({str(r.get("contract")) for r in rows if r.get("contract") != contract})
        if cs:
            problems.append(f"有记录的执行合同与运行中策略的规格不一致：{cs}（规格 {contract}；"
                            "None = 旧格式证据没记合同）")
    unknown = sorted({str(r["scenario"]) for r in rows if r["scenario"] not in SCENARIOS})
    if unknown:
        problems.append(f"不认识的场景：{unknown}")
    out_of_scope = sorted(set(waived) - set(required))
    if out_of_scope:
        problems.append(f"豁免了不在验收范围里的场景：{out_of_scope}")
    scenarios = {}
    for s in required:
        j = _judge(s, rows)
        if s in waived:
            if j["status"] == NOT_COVERED and waived[s].strip():
                j = {**j, "status": WAIVED, "detail": f"未覆盖，已豁免：{waived[s]}（{j['detail']}）"}
            elif j["status"] == FAILED:
                problems.append(f"{s} 是失败，不是未覆盖 —— 失败的场景不能豁免")
            elif j["status"] == NOT_COVERED:
                problems.append(f"豁免 {s} 要写明理由")
        scenarios[s] = j
    ok = not problems and all(v["status"] in (PASSED, WAIVED) for v in scenarios.values())
    return {"path": str(Path(path).resolve()), "sha256": sha, "account": int(account), "code_version": code,
            "contract": contract, "required": list(required), "waived": waived, "n_rows": len(rows),
            "scenarios": scenarios, "problems": problems, "ok": ok}


def reasons(assessment: dict) -> List[str]:
    """不通过的原因，一行一条（许可拒签 / 运行时核对 / Go/No-Go 共用的措辞）。"""
    out = list(assessment["problems"])
    out += [f"{s}：{v['status']}（{v['detail']}）" for s, v in assessment["scenarios"].items()
            if v["status"] not in (PASSED, WAIVED)]
    return out
