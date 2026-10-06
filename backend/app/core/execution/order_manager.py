"""
order_manager.py — 确定性执行工作流（Phase 12.1 / 12.2 / 12.4）

PM 美元账本的最后一行目标权重 → moomoo 纸交易订单 → 成交对账回 PositionStore。
全程确定性代码，LLM 不参与。

一个交易日的执行周期（`run_cycle`）
-----------------------------------
1. 对账（= 崩溃恢复，同一条代码路径）：读券商订单与持仓，把新增成交落账，
   核对"上一可信快照 + 我方订单新增成交"是否等于券商持仓。
2. 全平熔断已开启 → 只继续全平，不调仓。
3. 对账不干净（持仓对不上 / 有找不到的在途单 / 读数不稳定）→ **不交易**，等人确认。
4. 数据新鲜度：决策日必须是"最近一个已收盘的交易日"，且当前时刻在其收盘之后。
5. 账户总资产与 paper_aum 偏离过大 → 不交易（配置与现实不符）。
6. 撤掉更早决策日仍在途的我方调仓单。
7. 目标权重 → 股数订单 → 幂等去重 → 下单前风控门 → 先写意图、再下单。

收盘后下单、DAY 有效、次日开盘成交 —— 与模拟账本"决策日收盘价成交"的差，
正是 12.3 保真度报告要量的东西。

并发：同一进程内用一把锁串行化（调度器与人工 API 可能同时触发）。多进程部署不在本期范围。
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

import pandas as pd

from app.core.execution.broker_gateway import (
    DUST_QTY, AccountSnapshot, BrokerError, BrokerGateway, BrokerPosition, OrderQuery,
    OrderRequest, is_open_status,
)
from app.core.execution.order_builder import (
    PURPOSE_FLATTEN, PURPOSE_REBALANCE, build_flatten_orders, build_rebalance_orders,
    is_own_client_id,
)
from app.core.execution.pretrade_gate import GateContext, PreTradeLimits, check
from app.db.execution_store import (
    ST_PENDING, ST_SUBMITTED, ExecutionStore,
)

logger = logging.getLogger(__name__)

#: moomoo 纸交易账本在 PositionStore 里的 book id。
#: 0 = 内部模拟组合账本（PORTFOLIO_BOOK_ID），正数 = 各 alpha 影子账本，负数不会冲突。
LIVE_BOOK_ID = -1

_LOCK = threading.RLock()
_CID_RE = re.compile(r"^qa([RF])(-?\d+)-(\d{8})(\d{6})?-(.+)-([BS])$")
_STABLE_READ_ATTEMPTS = 3


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_client_id(cid: str) -> Optional[dict]:
    m = _CID_RE.match(str(cid or ""))
    if not m:
        return None
    tag, book, ymd, _hms, ticker, side = m.groups()
    return {"purpose": PURPOSE_FLATTEN if tag == "F" else PURPOSE_REBALANCE,
            "book_id": int(book), "decision_date": datetime.strptime(ymd, "%Y%m%d").date(),
            "ticker": ticker, "side": "BUY" if side == "B" else "SELL"}


def _date_of(ts: str, default: date) -> date:
    """券商时间串（交易所本地时间 'YYYY-MM-DD HH:MM:SS.fff'）的日期部分；解析不了用 default。"""
    t = pd.to_datetime(str(ts)[:10], errors="coerce", format="%Y-%m-%d")
    return default if pd.isna(t) else t.date()


# ---------------------------------------------------------------------------
# 交易日历：决策日是否"最近一个已收盘的交易日"
# ---------------------------------------------------------------------------

def market_today(now: datetime) -> date:
    """美东日期（交易日按交易所所在地计）。"""
    return pd.Timestamp(now).tz_convert("America/New_York").date()


def last_closed_session(now: datetime) -> date:
    """收盘时刻 ≤ now 的最近一个交易日。日历不可用时抛 CalendarUnavailable（fail-closed）。"""
    from app.core.data_engine.market_calendar import last_trading_day, session_close_utc
    d = last_trading_day(market_today(now)).date()
    close = session_close_utc(d)
    if close is None or close > now:
        d = last_trading_day(d - timedelta(days=1)).date()
    return d


def freshness_problem(decision_date: date, now: datetime) -> str:
    """返回空串 = 可以按这个决策日下单；否则返回拒绝原因。"""
    from app.core.data_engine.market_calendar import session_close_utc
    close = session_close_utc(decision_date)
    if close is None:
        return "not_a_trading_day"
    if now < close:
        return "session_not_closed"
    if decision_date != last_closed_session(now):
        return "stale_decision_date"
    return ""


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

@dataclass
class ReconcileReport:
    market_date:   str
    status:        str                       # baseline|clean|discrepancy|unresolved|unstable|accepted
    account:       dict
    positions:     Dict[str, dict]
    history_ok:    bool                      # 无默认值："订单查全了"必须由查询结果给出
    discrepancies: List[dict] = field(default_factory=list)
    unresolved:    List[dict] = field(default_factory=list)
    #: 下单调用报错、当天在券商侧还查不到的单：不重试、不阻断，次日再判定
    awaiting_confirmation: List[dict] = field(default_factory=list)
    fills:         List[dict] = field(default_factory=list)
    adopted:       List[str] = field(default_factory=list)
    day_return:    Optional[float] = None
    # 供本周期后续步骤复用（不进 to_dict）
    account_obj:   Optional[AccountSnapshot] = None
    position_objs: Dict[str, BrokerPosition] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return self.status in ("discrepancy", "unresolved", "unstable")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("account_obj", None)
        d.pop("position_objs", None)
        return d


@dataclass
class ExecutionReport:
    mode:            str = "moomoo_paper"
    decision_date:   str = ""
    blocked:         str = ""
    gateway:         Optional[dict] = None
    reconcile:       Optional[dict] = None
    n_planned:       int = 0
    n_duplicate:     int = 0
    n_gate_rejected: int = 0
    n_submitted:     int = 0
    n_submit_errors: int = 0
    orders:          List[dict] = field(default_factory=list)
    gate:            Optional[dict] = None
    build:           Optional[dict] = None
    cancelled_stale: List[str] = field(default_factory=list)
    flatten:         Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------

class OrderManager:
    def __init__(
        self,
        gateway: BrokerGateway,
        store: Optional[ExecutionStore] = None,
        position_store=None,
        *,
        limits: PreTradeLimits,
        paper_aum: float,
        limit_band_bps: float,
        flatten_band_bps: float,
        max_aum_mismatch: float,
        book_id: int = LIVE_BOOK_ID,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if not (math.isfinite(paper_aum) and paper_aum > 0):
            raise ValueError(f"paper_aum 必须为正：{paper_aum!r}")
        self.gw = gateway
        self.store = store or ExecutionStore()
        if position_store is None:
            from app.db.position_store import PositionStore
            position_store = PositionStore()
        self.pstore = position_store
        self.limits = limits
        self.paper_aum = float(paper_aum)
        self.limit_band_bps = float(limit_band_bps)
        self.flatten_band_bps = float(flatten_band_bps)
        self.max_aum_mismatch = float(max_aum_mismatch)
        self.book_id = int(book_id)
        self._clock = clock or _utcnow

    @classmethod
    def from_settings(cls, gateway: BrokerGateway, store: Optional[ExecutionStore] = None,
                      position_store=None, clock=None) -> "OrderManager":
        from app.config import settings
        return cls(
            gateway, store, position_store,
            limits=PreTradeLimits.from_settings(settings),
            paper_aum=float(settings.paper_aum),
            limit_band_bps=float(settings.exec_limit_band_bps),
            flatten_band_bps=float(settings.exec_flatten_band_bps),
            max_aum_mismatch=float(settings.exec_max_aum_mismatch),
            clock=clock,
        )

    # ------------------------------------------------------------------
    # 对账 / 恢复
    # ------------------------------------------------------------------

    def reconcile(self) -> ReconcileReport:
        with _LOCK:
            return self._reconcile(accept=False)

    def accept_discrepancies(self, actor: str, reason: str) -> ReconcileReport:
        """
        人工确认：以券商当前状态为准重新建立可信基准。
        找不到的在途单按"未到达券商"/"券商侧已不存在"结案。审计日志里记下是谁、为什么。
        """
        if not actor or not reason:
            raise ValueError("接受对账差异必须写明操作人与原因")
        with _LOCK:
            rep = self._reconcile(accept=True)
            self.store.add_event("reconcile_accepted", actor, reason,
                                 {"discrepancies": rep.discrepancies, "unresolved": rep.unresolved})
            return rep

    def _stable_read(self, since: date):
        """订单 → 持仓 → 账户 → 订单；两次订单的累计成交一致才算读到一致的状态。"""
        sig = lambda q: sorted((o.broker_order_id, o.dealt_qty) for o in q.orders)  # noqa: E731
        for _ in range(_STABLE_READ_ATTEMPTS):
            q1 = self.gw.orders(since)
            pos = self.gw.positions()
            acct = self.gw.account()
            q2 = self.gw.orders(since)
            if sig(q1) == sig(q2):
                return q2, pos, acct, True
        logger.warning("[exec] 连续 %d 次读到的成交在变化（盘中成交中），本轮对账不作判定",
                       _STABLE_READ_ATTEMPTS)
        return q2, pos, acct, False

    def _query_since(self, today: date) -> date:
        ds = [r.decision_date for r in self.store.open_orders(self.book_id)]
        snap = self.store.latest_snapshot()
        if snap is not None:
            ds.append(snap.market_date)
        start = min(ds) if ds else today
        return start - timedelta(days=7)

    def _reconcile(self, accept: bool) -> ReconcileReport:
        from app.core.data_engine.market_calendar import last_trading_day
        now = self._clock()
        market_date = last_trading_day(market_today(now)).date()
        q, pos, acct, stable = self._stable_read(self._query_since(market_date))
        if accept and not stable:
            # 盘中成交进行中时把券商状态立成基准，基准本身就和成交记录对不上
            raise RuntimeError("券商读数不稳定（盘中成交进行中），不能作为新基准 —— 请稍后再接受")

        own = {o.client_id: o for o in q.orders if is_own_client_id(o.client_id)}
        unresolved: List[dict] = []
        awaiting: List[dict] = []

        # 1) 本地在途单：在券商侧找得到 → 交给第 2 步落账；找不到 → 判定
        for row in self.store.open_orders(self.book_id):
            if row.client_id in own:
                continue
            if accept:
                if row.status == ST_PENDING:
                    self.store.mark_not_submitted(row.client_id, "人工接受：券商侧找不到")
                else:
                    self.store.apply_broker_state(row.client_id, _missing_order(row), market_date)
                continue
            if row.status == ST_PENDING:
                attempted = bool(row.last_err_msg)
                if not attempted and q.history_ok:
                    # 写了意图、但下单调用从未发出（崩在下单之前）→ 券商不可能收到
                    self.store.mark_not_submitted(row.client_id, "券商侧不存在该 remark，确认未提交")
                    continue
                if attempted and row.decision_date >= market_date:
                    # 下单调用报错（如超时）的当天：券商可能已收单只是还没出现在列表里。
                    # 判成"未提交"会触发同日重试 → 可能挂出两张同 remark 的单。
                    # 保持待确认：不重试（PENDING 不可重试）、也不阻断其他交易。
                    awaiting.append({"client_id": row.client_id, "ticker": row.ticker,
                                     "error": row.last_err_msg})
                    continue
                if attempted and q.history_ok:
                    # 之后的交易日、历史订单里仍查不到 → 确认券商没收到
                    self.store.mark_not_submitted(
                        row.client_id, f"下单报错且之后的交易日券商侧仍不存在：{row.last_err_msg}")
                    continue
            unresolved.append({"client_id": row.client_id, "status": row.status,
                               "ticker": row.ticker, "history_ok": q.history_ok})

        # 2) 我方订单的成交增量落账（本地没有的 → 补登）
        fills, adopted = [], []
        for cid, bo in own.items():
            if self.store.get(cid) is None:
                meta = parse_client_id(cid) or {}
                self.store.adopt_broker_order(
                    bo, meta.get("book_id", self.book_id),
                    meta.get("decision_date", market_date),
                    meta.get("purpose", PURPOSE_REBALANCE), ref_price=0.0)
                adopted.append(cid)
                logger.warning("[exec] 券商侧有我方订单 %s 但本地无记录 → 已补登", cid)
            inc = self.store.apply_broker_state(cid, bo, _date_of(bo.updated_time, market_date))
            if inc is not None:
                fills.append({"client_id": cid, "ticker": inc.ticker, "qty": inc.qty,
                              "price": inc.price, "ref_price": inc.ref_price,
                              "fill_date": str(inc.fill_date)})

        # 3) 账本推算 vs 券商持仓
        broker_qty = {tk: p.qty for tk, p in pos.items()}
        trusted = self.store.latest_snapshot(trusted_only=True)
        discrepancies: List[dict] = []
        if trusted is None:
            status = "baseline"
        else:
            expected = self._expected_positions(trusted)
            for tk in sorted(set(expected) | set(broker_qty)):
                e, b = expected.get(tk, 0.0), broker_qty.get(tk, 0.0)
                if abs(e - b) > DUST_QTY:
                    discrepancies.append({"ticker": tk, "expected": e, "broker": b})
            status = "discrepancy" if discrepancies else "clean"
        if unresolved:
            status = "unresolved"
        if not stable:
            status = "unstable"
        if accept:
            status = "accepted"

        prev = self.store.trusted_before(market_date)
        day_ret = (acct.total_assets / prev.total_assets - 1.0
                   if prev is not None and prev.total_assets > 0 else None)

        positions = {tk: {"qty": p.qty, "price": p.nominal_price, "market_val": p.market_val}
                     for tk, p in pos.items()}
        self.store.add_snapshot(
            market_date=market_date, account=acct, positions=positions,
            discrepancies=discrepancies + unresolved, status=status,
            note="" if q.history_ok else f"history_unavailable: {q.history_error}")
        if (self.store.get_state("baseline_total_assets") is None
                and status in ("baseline", "clean", "accepted") and acct.total_assets > 0):
            # 净值基线 = 第一次在可信对账里看到的**正**资产（空账户之后才入金也能建立基线；
            # 记成 0 的话之后所有净值都是 x/0）。一旦建立就不再改 —— 接受差异也不重置。
            self.store.set_state("baseline_total_assets", repr(float(acct.total_assets)))

        if status in ("baseline", "clean", "accepted"):
            self._record_book(market_date, acct, pos)
        if status in ("discrepancy", "unresolved", "unstable"):
            logger.error("[exec] 对账状态=%s → 本轮不下单 | 差异=%s | 未解决=%s",
                         status, discrepancies, unresolved)

        return ReconcileReport(
            market_date=str(market_date), status=status,
            account={"total_assets": acct.total_assets, "cash": acct.cash,
                     "power": acct.power, "market_val": acct.market_val,
                     "currency": acct.currency},
            positions=positions, discrepancies=discrepancies, unresolved=unresolved,
            awaiting_confirmation=awaiting,
            fills=fills, adopted=adopted, day_return=day_ret, history_ok=q.history_ok,
            account_obj=acct, position_objs=pos)

    def _expected_positions(self, trusted) -> Dict[str, float]:
        """
        可信快照的持仓 + 该快照之后入账的我方成交增量（含成交撤销的负增量）。
        同一轮对账"先入账成交、后写快照"，且读数稳定性已确认，所以快照之前入账的成交
        都已体现在快照持仓里 —— 按快照 id 切分不会重复也不会遗漏。
        """
        expected = {tk: float(d["qty"]) for tk, d in json.loads(trusted.positions_json).items()}
        for f in self.store.fills_after_snapshot(trusted.id, self.book_id):
            expected[f.ticker] = expected.get(f.ticker, 0.0) + float(f.qty)
        return expected                     # 零股噪声由调用方按 DUST_QTY 比较时吸收

    def _record_book(self, market_date: date, acct: AccountSnapshot,
                     pos: Dict[str, BrokerPosition]) -> None:
        """把券商真相写进 PositionStore 的 LIVE_BOOK_ID 账本（与模拟账本同表、可并排比较）。"""
        from app.db.position_store import DailyPnL
        ta = float(acct.total_assets)
        base = float(self.store.get_state("baseline_total_assets") or ta)
        equity = ta / base if base > 0 else 0.0       # 账户资产为 0：净值记 0，而不是 x/0
        prev_eq, _ = self.pstore.state_before(self.book_id, market_date)
        net_ret = equity / prev_eq - 1.0 if prev_eq > 0 else 0.0
        weights = {tk: p.market_val / ta for tk, p in pos.items() if ta > 0}

        agg: Dict[str, dict] = {}
        for f in self.store.fills(book_id=self.book_id):
            if f.fill_date != market_date:
                continue
            a = agg.setdefault(f.ticker, {"qty": 0.0, "notional": 0.0, "shortfall": 0.0})
            a["qty"] += f.qty
            a["notional"] += f.qty * f.price
            if f.ref_price > 0:                       # 补登单没有决策日参考价，不计入执行缺口
                a["shortfall"] += f.qty * (f.price - f.ref_price)
        prev_usd = prev_eq * base
        shortfall = sum(a["shortfall"] for a in agg.values())
        cost_bps = shortfall / prev_usd * 1e4 if prev_usd > 0 else 0.0
        fills = [{
            "ticker": tk, "target_weight": 0.0, "filled_weight": weights.get(tk, 0.0),
            "fill_price": (a["notional"] / a["qty"]) if abs(a["qty"]) > DUST_QTY else 0.0,
            "cost_usd": a["shortfall"], "reject_reason": "",
            "traded_weight": a["notional"] / ta if ta > 0 else 0.0, "unfilled_weight": 0.0,
        } for tk, a in sorted(agg.items())]
        self.pstore.record_day(
            self.book_id, market_date, weights, fills,
            DailyPnL(alpha_id=self.book_id, date=str(market_date),
                     gross_ret=net_ret + cost_bps * 1e-4, net_ret=net_ret,
                     cost_bps=cost_bps, equity=equity))

    # ------------------------------------------------------------------
    # 调仓
    # ------------------------------------------------------------------

    def run_cycle(self, target_weights: pd.Series, ref_prices: pd.Series,
                  adv_usd: pd.Series, decision_date: date) -> ExecutionReport:
        with _LOCK:
            return self._run_cycle(target_weights, ref_prices, adv_usd, decision_date)

    def maintain(self, reason: str) -> ExecutionReport:
        """
        不调仓的日子（没有可交易信号、策略门硬拦截）也要做的事：对账（成交照样会发生、
        账本照样要更新），以及熔断开启时**续做全平** —— 全平不能依赖当天有没有策略信号。
        """
        from app.core.data_engine.market_calendar import CalendarUnavailable
        with _LOCK:
            rep = ExecutionReport(blocked=f"maintenance_only: {reason}")
            rep.gateway = self.gw.describe()
            try:
                rec = self._reconcile(accept=False)
            except (BrokerError, CalendarUnavailable) as exc:
                logger.error("[exec] 维护对账失败: %s", exc)
                rep.blocked = f"reconcile_failed: {exc}"
                return rep
            rep.reconcile = rec.to_dict()
            if self.store.kill_switch().get("engaged"):
                rep.flatten = self._ensure_flatten(rec)
            return rep

    def _run_cycle(self, target_weights, ref_prices, adv_usd, decision_date) -> ExecutionReport:
        from app.core.data_engine.market_calendar import CalendarUnavailable
        rep = ExecutionReport(decision_date=str(decision_date))
        rep.gateway = self.gw.describe()
        try:
            rec = self._reconcile(accept=False)
        except (BrokerError, CalendarUnavailable) as exc:
            logger.error("[exec] 对账失败 → 本轮不下单: %s", exc)
            rep.blocked = f"reconcile_failed: {exc}"
            return rep
        rep.reconcile = rec.to_dict()

        if self.store.kill_switch().get("engaged"):
            rep.flatten = self._ensure_flatten(rec)
            rep.blocked = "kill_switch_engaged"
            return rep
        if rec.blocking:
            rep.blocked = f"reconcile_{rec.status}"
            return rep

        try:
            why = freshness_problem(decision_date, self._clock())
        except CalendarUnavailable as exc:
            logger.error("[exec] 交易日历不可用 → 本轮不下单: %s", exc)
            why = "calendar_unavailable"
        if why:
            logger.warning("[exec] 决策日 %s 不可下单：%s", decision_date, why)
            rep.blocked = why
            return rep

        acct = rec.account_obj
        ratio = acct.total_assets / self.paper_aum
        if abs(ratio - 1.0) > self.max_aum_mismatch:
            logger.error("[exec] 账户总资产 %.2f 与 paper_aum %.2f 偏离 %.0f%% → 不交易",
                         acct.total_assets, self.paper_aum, (ratio - 1) * 100)
            rep.blocked = "aum_mismatch"
            return rep

        # 撤掉与本次调仓冲突的在途单：更早决策日的调仓单，以及**任何**残留的全平单
        # （熔断解除后，没成交完的全平卖单若不撤，会和新的调仓买单对冲打架）。
        for row in self.store.open_orders(self.book_id):
            if (row.status == ST_SUBMITTED and row.broker_order_id
                    and (row.purpose == PURPOSE_FLATTEN or row.decision_date < decision_date)):
                try:
                    self.gw.cancel(row.broker_order_id)
                    rep.cancelled_stale.append(row.client_id)
                except BrokerError as exc:
                    logger.error("[exec] 撤销过期单 %s 失败 → 本轮不下新单: %s", row.client_id, exc)
                    rep.blocked = "stale_cancel_failed"
                    return rep

        pos = rec.position_objs
        build = build_rebalance_orders(
            target_weights, {tk: p.qty for tk, p in pos.items()}, ref_prices,
            acct.total_assets, decision_date, book_id=self.book_id,
            band_bps=self.limit_band_bps, allow_short=self.limits.allow_short)
        rep.build = {"skipped": build.skipped, "notes": build.notes,
                     "max_rounding_drift": max((abs(v) for v in build.rounding.values()), default=0.0)}
        fresh = [o for o in build.orders if not self.store.is_final_for_day(o.client_id)]
        rep.n_planned = len(build.orders)
        rep.n_duplicate = len(build.orders) - len(fresh)

        dec = check(fresh, GateContext(
            equity=acct.total_assets, buying_power=acct.power,
            current_qty={tk: p.qty for tk, p in pos.items()},
            broker_prices={tk: p.nominal_price for tk, p in pos.items()},
            adv_usd=adv_usd, day_return=rec.day_return, kill_switch=False), self.limits)
        rep.gate = dec.to_dict()
        for o, reason in dec.rejected:
            self.store.record_gate_rejection(o, self.book_id, decision_date, reason)
        rep.n_gate_rejected = len(dec.rejected)
        if dec.rejected:
            logger.warning("[exec] 下单前风控拦下 %d 张：%s", len(dec.rejected),
                           [(o.ticker, r) for o, r in dec.rejected])

        for o in dec.approved:
            rep.orders.append(self._submit(o, decision_date, rep))
        logger.info("[exec] %s | 计划 %d | 重复 %d | 风控拒 %d | 已下 %d | 下单异常 %d",
                    decision_date, rep.n_planned, rep.n_duplicate, rep.n_gate_rejected,
                    rep.n_submitted, rep.n_submit_errors)
        return rep

    def _submit(self, o, decision_date, rep: ExecutionReport) -> dict:
        info = {"client_id": o.client_id, "ticker": o.ticker, "side": o.side, "qty": o.qty,
                "limit_price": o.limit_price, "ref_price": o.ref_price, "purpose": o.purpose}
        if not self.store.add_intent(o, self.book_id, decision_date):
            rep.n_duplicate += 1
            info["result"] = "duplicate"
            return info
        try:
            oid = self.gw.place(OrderRequest(client_id=o.client_id, ticker=o.ticker, side=o.side,
                                             qty=o.qty, limit_price=o.limit_price))
        except (BrokerError, ValueError) as exc:
            # 保持 PENDING_SUBMIT：调用失败不等于券商没收到（超时），下次对账按 remark 确认
            self.store.note_submit_error(o.client_id, str(exc))
            rep.n_submit_errors += 1
            logger.error("[exec] 下单 %s 异常（保持待确认）: %s", o.client_id, exc)
            info["result"] = f"submit_error: {exc}"
            return info
        self.store.mark_submitted(o.client_id, oid)
        rep.n_submitted += 1
        info["result"] = "submitted"
        info["broker_order_id"] = oid
        return info

    # ------------------------------------------------------------------
    # 一键全平（kill switch）
    # ------------------------------------------------------------------

    def engage_kill_switch(self, actor: str, reason: str, persist_state: bool = True) -> dict:
        """
        开启全平熔断：状态**先**落库（之后任何调仓周期都只会继续全平），再撤单、再全平。
        券商调用失败时熔断状态保持开启 —— 下一个周期会继续尝试全平。
        `persist_state=False`：调用方已在连券商**之前**落好了状态（API 端点就是这么做的，
        因为连券商本身可能失败），这里不再重复写状态与事件。
        """
        if not actor or not reason:
            raise ValueError("全平必须写明操作人与原因")
        with _LOCK:
            if persist_state:
                st = self.store.set_kill_switch(True, actor, reason)
                self.store.add_event("kill_switch_engaged", actor, reason)
            else:
                st = self.store.kill_switch()
                if not st.get("engaged"):
                    raise RuntimeError("persist_state=False 但熔断状态并未开启 —— 拒绝在未落库时全平")
            out = {"state": st}
            try:
                out["cancelled"] = self._cancel_all_open()
                out["flatten"] = self._place_flatten(self.gw.positions())
            except BrokerError as exc:
                logger.error("[exec] 全平执行中券商调用失败（熔断保持开启，下周期续做）: %s", exc)
                out["error"] = str(exc)
            return out

    def reset_kill_switch(self, actor: str, reason: str) -> dict:
        if not actor or not reason:
            raise ValueError("解除全平熔断必须写明操作人与原因")
        with _LOCK:
            st = self.store.set_kill_switch(False, actor, reason)
            self.store.add_event("kill_switch_reset", actor, reason)
            return {"state": st}

    def _cancel_all_open(self) -> List[str]:
        """撤销账户里**所有**在途单（模拟账户专供本系统，全平时不区分来源）。"""
        q: OrderQuery = self.gw.orders(market_today(self._clock()) - timedelta(days=7))
        done = []
        for o in q.orders:
            if is_open_status(o.status):
                self.gw.cancel(o.broker_order_id)
                done.append(o.broker_order_id)
        return done

    def _place_flatten(self, positions: Dict[str, BrokerPosition]) -> dict:
        now = self._clock()
        stamp = now.strftime("%Y%m%d%H%M%S")
        build = build_flatten_orders(positions, stamp, book_id=self.book_id,
                                     band_bps=self.flatten_band_bps)
        acct = self.gw.account()
        dec = check(build.orders, GateContext(
            equity=acct.total_assets, buying_power=acct.power,
            current_qty={tk: p.qty for tk, p in positions.items()},
            broker_prices={tk: p.nominal_price for tk, p in positions.items()},
            adv_usd={}, kill_switch=True), self.limits)
        rep = ExecutionReport(decision_date=str(market_today(now)))
        decision = market_today(now)
        for o, reason in dec.rejected:
            self.store.record_gate_rejection(o, self.book_id, decision, reason)
        orders = [self._submit(o, decision, rep) for o in dec.approved]
        self.store.add_event("flatten_submitted", "system", "",
                             {"n": rep.n_submitted, "errors": rep.n_submit_errors,
                              "skipped": build.skipped,
                              "rejected": [(o.ticker, r) for o, r in dec.rejected]})
        return {"orders": orders, "skipped": build.skipped, "gate": dec.to_dict(),
                "n_submitted": rep.n_submitted, "n_submit_errors": rep.n_submit_errors}

    def _ensure_flatten(self, rec: ReconcileReport) -> dict:
        """熔断开启时的每周期动作：还有持仓、且没有在途全平单 → 再下一轮全平单。"""
        pending = [r for r in self.store.open_orders(self.book_id) if r.purpose == PURPOSE_FLATTEN]
        if pending:
            return {"status": "waiting", "open_flatten_orders": [r.client_id for r in pending]}
        if not rec.position_objs:
            return {"status": "flat"}
        try:
            self._cancel_all_open()
            return {"status": "resubmitted", **self._place_flatten(rec.position_objs)}
        except BrokerError as exc:
            logger.error("[exec] 续做全平失败: %s", exc)
            return {"status": "error", "error": str(exc)}


def _missing_order(row):
    """人工接受时，把券商侧找不到的已提交单按"已撤、零成交"结案（成交量保持本地已知值）。"""
    from app.core.execution.broker_gateway import BrokerOrder
    return BrokerOrder(
        broker_order_id=row.broker_order_id or "", client_id=row.client_id, ticker=row.ticker,
        side=row.side, qty=float(row.qty), price=float(row.limit_price),
        status="CANCELLED_ALL" if not row.dealt_qty else "CANCELLED_PART",
        dealt_qty=float(row.dealt_qty or 0.0), dealt_avg_price=float(row.dealt_avg_price or 0.0),
        create_time="", updated_time="", last_err_msg="accepted_missing_at_broker")


# ---------------------------------------------------------------------------
# 只读状态（不连券商）与启动恢复
# ---------------------------------------------------------------------------

def execution_status(store: Optional[ExecutionStore] = None) -> dict:
    from app.config import settings
    store = store or ExecutionStore()
    snap = store.latest_snapshot()
    return {
        "mode": settings.execution_mode,
        "kill_switch": store.kill_switch(),
        "last_snapshot": None if snap is None else {
            "taken_at": snap.taken_at.isoformat(timespec="seconds"),
            "market_date": str(snap.market_date), "status": snap.status,
            "total_assets": snap.total_assets, "cash": snap.cash, "power": snap.power,
            "positions": json.loads(snap.positions_json),
            "discrepancies": json.loads(snap.discrepancies_json), "note": snap.note},
        "open_orders": [_order_dict(r) for r in store.open_orders()],
        "recent_orders": [_order_dict(r) for r in store.orders(limit=50)],
        "recent_fills": [{"client_id": f.client_id, "ticker": f.ticker, "side": f.side,
                          "qty": f.qty, "price": f.price, "ref_price": f.ref_price,
                          "decision_date": str(f.decision_date), "fill_date": str(f.fill_date)}
                         for f in store.fills()[-50:]],
        "events": [{"at": e.at.isoformat(timespec="seconds"), "kind": e.kind,
                    "actor": e.actor, "reason": e.reason} for e in store.events(20)],
    }


def _order_dict(r) -> dict:
    return {"client_id": r.client_id, "decision_date": str(r.decision_date),
            "purpose": r.purpose, "ticker": r.ticker, "side": r.side, "qty": r.qty,
            "limit_price": r.limit_price, "ref_price": r.ref_price, "status": r.status,
            "broker_status": r.broker_status, "broker_order_id": r.broker_order_id,
            "dealt_qty": r.dealt_qty, "dealt_avg_price": r.dealt_avg_price,
            "reject_reason": r.reject_reason, "last_err_msg": r.last_err_msg}


def recover_on_startup(gateway_factory: Optional[Callable[[], BrokerGateway]] = None) -> dict:
    """
    Phase 12.4：服务启动时对账一次 —— 补记停机期间的成交、找回崩溃前写了意图的订单、
    用券商持仓重建 LIVE_BOOK_ID 账本。失败只记录，不阻止服务启动（之后的交易周期
    会先对账，对不上就不交易）。
    """
    from app.core.execution.broker_gateway import make_gateway_from_settings
    store = ExecutionStore()
    try:
        gw = (gateway_factory or make_gateway_from_settings)()
    except BrokerError as exc:
        logger.error("[exec] 启动恢复：连不上券商（交易周期会继续 fail-closed）: %s", exc)
        store.add_event("startup_recovery_failed", "system", str(exc))
        return {"ok": False, "error": str(exc)}
    try:
        rep = OrderManager.from_settings(gw, store=store).reconcile()
        store.add_event("startup_recovery", "system", rep.status,
                        {"fills": len(rep.fills), "adopted": rep.adopted,
                         "unresolved": rep.unresolved, "discrepancies": rep.discrepancies})
        logger.info("[exec] 启动恢复完成：状态=%s 补记成交=%d 补登订单=%d",
                    rep.status, len(rep.fills), len(rep.adopted))
        return {"ok": True, "report": rep.to_dict()}
    except Exception as exc:
        logger.error("[exec] 启动恢复失败: %s", exc)
        store.add_event("startup_recovery_failed", "system", str(exc))
        return {"ok": False, "error": str(exc)}
    finally:
        gw.close()
