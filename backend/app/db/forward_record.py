"""
forward_record.py — 正式前向的启动记录与每日记录（复审 2026-10-19 §7.3）

复审原话：
  "完成前两阶段、重新提案并批准后，记录唯一启动时间、规格哈希、代码版本、数据身份和环境版本。
   联调和历史回放不得补记成正式天数。"
  "每天至少记录：信号/目标生成与存档时间、预期与实际 session、目标权重/股数、实际成交、未成交原因、
   持仓现金、成本、净收益、基准超额、回撤、gross/net、资金利用率、对账状态及异常。"

此前这些散在五处（诊断 JSON、执行库订单 / 成交 / 快照、组合账本、信号存档），没有一条"这一天"的记录，
也分不出哪天算正式前向。现在：

  · `ForwardRecordStore.start(strategy_id, spec_hash, actor, reason, ...)` —— **人**启动正式前向
    （D13：重新提案并批准之后）。同一策略 + 规格只能启动一次（唯一启动时间）。
  · 每日循环每轮写一条 `ForwardDay`（策略 × 决策日，同日重跑覆盖）：`build_daily_record` 组装字段；
    `formal` 只有在 ① 有启动记录、② 决策日不早于启动、③ 本轮在决策日收盘后到下一交易日开盘前运行
    （不是事后补跑 / 回放）、④ 执行层真的在线并且没被阻断 时为真，否则写明原因。
  · 某项算不出来 → 值为 None，并在 `missing` 里写明为什么（不把"没算"写成 0）。

基准：缺省是**数据集等权宇宙**的收盘到收盘收益（`ew_universe`）。用哪个基准是待定事项 D16。
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import Column, Date, DateTime, Integer, String, Text, UniqueConstraint, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

BENCHMARK = "ew_universe"
ENV_PACKAGES = ("numpy", "pandas", "scipy", "sqlalchemy", "fastapi", "moomoo-api", "pydantic")


class _Base(DeclarativeBase):
    pass


class ForwardRun(_Base):
    """正式前向的启动记录（每份策略 + 规格一条）。"""
    __tablename__ = "forward_runs"
    id:           int      = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id:  int      = Column(Integer, nullable=False)
    spec_hash:    str      = Column(String(64), nullable=False)
    started_at:   datetime = Column(DateTime, nullable=False)
    actor:        str      = Column(String(64), nullable=False)
    reason:       str      = Column(Text, nullable=False)
    code_version: str      = Column(String(40), nullable=False)
    data_identity: str     = Column(Text, nullable=False, default="{}")
    environment:  str      = Column(Text, nullable=False, default="{}")
    __table_args__ = (UniqueConstraint("strategy_id", "spec_hash", name="uq_forward_run"),)


class ForwardDay(_Base):
    """某策略某决策日的记录（同日重跑覆盖；payload 是 build_daily_record 的完整结果）。"""
    __tablename__ = "forward_days"
    id:            int      = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id:   int      = Column(Integer, nullable=False, index=True)
    spec_hash:     str      = Column(String(64), nullable=False, default="")
    decision_date: object   = Column(Date, nullable=False, index=True)
    formal:        int      = Column(Integer, nullable=False, default=0)
    formal_reason: str      = Column(String(128), nullable=False, default="")
    recorded_at:   datetime = Column(DateTime, nullable=False)
    payload:       str      = Column(Text, nullable=False, default="{}")
    __table_args__ = (UniqueConstraint("strategy_id", "decision_date", name="uq_forward_day"),)


def environment_version() -> Dict[str, Any]:
    from importlib import metadata
    pkgs = {}
    for name in ENV_PACKAGES:
        try:
            pkgs[name] = metadata.version(name)
        except Exception as exc:
            logger.warning("[forward] 取不到 %s 的版本（启动记录里记为 None）: %s", name, exc)
            pkgs[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": pkgs}


class ForwardRecordStore:
    def __init__(self, db_url: Optional[str] = None) -> None:
        if db_url is None:
            db_url = os.getenv("DATABASE_URL", "")
            if not db_url:
                from app.config import settings
                db_url = settings.database_url
        connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        self._engine = create_engine(db_url, connect_args=connect_args, echo=False)
        from ._sqlite_utils import harden_sqlite_engine
        harden_sqlite_engine(self._engine)
        _Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    # ------------------------------------------------------------------ 启动（人做）
    def start(self, strategy_id: int, spec_hash: str, actor: str, reason: str,
              data_identity: Optional[dict] = None, started_at: Optional[datetime] = None) -> ForwardRun:
        """启动正式前向。同一策略 + 规格第二次启动 → ValueError（启动时间唯一，不能重置后"重新开始"）。"""
        if not actor or not reason:
            raise ValueError("启动正式前向必须写明操作人与原因")
        if not spec_hash:
            raise ValueError("没有规格哈希的策略不能启动正式前向")
        from app.core.data_engine.input_identity import code_version
        row = ForwardRun(strategy_id=int(strategy_id), spec_hash=str(spec_hash),
                         started_at=_naive_utc(started_at or datetime.now(timezone.utc)),
                         actor=str(actor)[:64], reason=str(reason), code_version=code_version(),
                         data_identity=json.dumps(data_identity or {}, ensure_ascii=False, default=str),
                         environment=json.dumps(environment_version(), ensure_ascii=False))
        with self._Session() as s:
            s.add(row)
            try:
                s.commit()
            except IntegrityError as exc:
                s.rollback()
                raise ValueError(f"策略 #{strategy_id}（规格 {spec_hash[:12]}）已经启动过正式前向") from exc
            return row

    def run_for(self, strategy_id: int, spec_hash: str) -> Optional[ForwardRun]:
        with self._Session() as s:
            return s.scalars(select(ForwardRun).where(ForwardRun.strategy_id == int(strategy_id),
                                                      ForwardRun.spec_hash == str(spec_hash))).first()

    # ------------------------------------------------------------------ 每日
    def record_day(self, record: Dict[str, Any]) -> None:
        day = pd.Timestamp(record["decision_date"]).date()
        sid = int(record["strategy_id"])
        with self._Session() as s:
            row = s.scalars(select(ForwardDay).where(ForwardDay.strategy_id == sid,
                                                     ForwardDay.decision_date == day)).first()
            if row is None:
                row = ForwardDay(strategy_id=sid, decision_date=day)
                s.add(row)
            elif row.formal and not record.get("formal"):
                # 事后补跑 / 回放不能把已经是正式的那一天改写成别的（"不得补记"的另一面：也不得覆盖）
                logger.warning("[forward] 策略 #%s %s 已有正式记录；本轮（%s）不覆盖", sid, day,
                               record.get("formal_reason"))
                return
            row.spec_hash = str(record.get("spec_hash") or "")
            row.formal = int(bool(record.get("formal")))
            row.formal_reason = str(record.get("formal_reason") or "")[:128]
            row.recorded_at = _naive_utc(datetime.now(timezone.utc))
            row.payload = json.dumps(record, ensure_ascii=False, default=str, sort_keys=True)
            s.commit()

    def days(self, strategy_id: Optional[int] = None, formal_only: bool = False) -> List[Dict[str, Any]]:
        with self._Session() as s:
            q = select(ForwardDay).order_by(ForwardDay.decision_date)
            if strategy_id is not None:
                q = q.where(ForwardDay.strategy_id == int(strategy_id))
            if formal_only:
                q = q.where(ForwardDay.formal == 1)
            return [json.loads(r.payload) for r in s.scalars(q).all()]


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------

#: 每条每日记录必须有的键（复审 §7.3 的原话逐项对应；值可以是 None，但要在 missing 里说明）
REQUIRED_FIELDS = (
    "signal_archived_at", "targets_generated_at",            # 信号 / 目标生成与存档时间
    "expected_session", "actual_sessions",                   # 预期与实际 session
    "target_weights", "target_shares",                       # 目标权重 / 股数
    "fills",                                                 # 实际成交
    "unfilled",                                              # 未成交原因
    "cash", "positions",                                     # 持仓现金
    "costs",                                                 # 成本
    "net_return",                                            # 净收益
    "benchmark", "excess_return",                            # 基准超额
    "drawdown",                                              # 回撤
    "exposure",                                              # gross / net
    "capital_utilization",                                   # 资金利用率
    "reconcile_status", "anomalies",                         # 对账状态及异常
)


def build_daily_record(result: Dict[str, Any], *, strategy_id: int, spec_hash: str, decision_date,
                       now: datetime, dataset: Dict[str, pd.DataFrame], paper_history: List[Any],
                       exec_store=None, identity: Optional[dict] = None,
                       forward_run: Optional[ForwardRun] = None) -> Dict[str, Any]:
    """把一轮 run_portfolio 的结果 + 执行库 + 组合账本组装成 §7.3 的每日记录。"""
    from app.core.data_engine.input_identity import code_version
    from app.core.data_engine.market_calendar import session_close_utc, session_open_utc, trading_days

    missing: Dict[str, str] = {}
    day = pd.Timestamp(decision_date).normalize()
    ex = result.get("execution") or {}
    rec = (ex.get("reconcile") or {}) if isinstance(ex, dict) else {}
    now_n = _naive_utc(now)

    out: Dict[str, Any] = {"decision_date": str(day.date()), "strategy_id": int(strategy_id),
                           "spec_hash": spec_hash, "code_version": code_version(), "data_identity": identity}

    # ---- 信号 / 目标生成与存档时间
    out["targets_generated_at"] = now_n.isoformat()
    out["signal_archived_at"] = result.get("signal_archived_at")
    if out["signal_archived_at"] is None:
        missing["signal_archived_at"] = "本轮没有存档组合信号（提前返回或存档失败）"

    # ---- 预期与实际 session：盘后决策 → 下一交易日的常规盘
    nxt = [d for d in trading_days(day + timedelta(days=1), day + timedelta(days=10)) if d > day]
    out["expected_session"] = str(pd.Timestamp(nxt[0]).date()) if nxt else None
    if not nxt:
        missing["expected_session"] = "日历给不出下一交易日"
    fills = [dict(f) for f in (rec.get("fills") or [])]           # 本轮对账回来的成交（qty 有符号）
    for f in fills:
        if exec_store is not None and f.get("client_id") and "decision_date" not in f:
            try:
                o = exec_store.get(f["client_id"])
                f["decision_date"] = str(o.decision_date) if o is not None else None
            except Exception as exc:
                logger.warning("[forward] 成交 %s 的决策日读不到: %s", f.get("client_id"), exc)
                missing["fills.decision_date"] = f"执行库读不到订单：{exc}"
    out["fills"] = fills
    out["actual_sessions"] = sorted({str(f.get("fill_date")) for f in fills if f.get("fill_date")})

    # ---- 目标权重 / 股数
    out["target_weights"] = result.get("execution_target")
    orders = [o for o in (ex.get("orders") or [])] if isinstance(ex, dict) else []
    out["target_shares"] = {str(o.get("ticker")): (int(o.get("qty", 0)) * (1 if str(o.get("side", "")).upper()
                                                                           .startswith("B") else -1))
                            for o in orders if o.get("ticker")}
    if not orders:
        missing["target_shares"] = f"本轮没有订单计划（{ex.get('blocked') or result.get('reason') or '无'}）"

    # ---- 未成交原因：本轮被拒 / 被带跳过 / 发送失败；之前决策日的单到终态仍没成交完
    unfilled: List[Dict[str, Any]] = []
    for r in ((ex.get("gate") or {}).get("rejected") or []) if isinstance(ex, dict) else []:
        unfilled.append({"ticker": r.get("ticker"), "decision_date": str(day.date()),
                         "qty": r.get("qty"), "reason": f"gate:{r.get('reason')}"})
    for r in ((ex.get("build") or {}).get("skipped") or []) if isinstance(ex, dict) else []:
        unfilled.append({"ticker": r.get("ticker"), "decision_date": str(day.date()),
                         "qty": r.get("qty"), "reason": f"build:{r.get('reason')}"})
    for o in orders:
        if str(o.get("result", "")).startswith(("submit_error", "skipped")):
            unfilled.append({"ticker": o.get("ticker"), "decision_date": str(day.date()),
                             "qty": o.get("qty"), "reason": f"order:{o.get('result')}"})
    if exec_store is not None:
        try:
            for o in exec_store.orders(since=(day - timedelta(days=10)).date()):
                if o.decision_date >= day.date():
                    continue
                if o.status in ("CANCELLED", "FAILED", "PARTIAL") and float(o.dealt_qty or 0) < float(o.qty):
                    unfilled.append({"ticker": o.ticker, "decision_date": str(o.decision_date),
                                     "qty": int(o.qty) - int(float(o.dealt_qty or 0)),
                                     "reason": f"broker:{o.status}:{o.reject_reason or o.broker_status}"})
        except Exception as exc:
            logger.warning("[forward] 之前决策日的订单读不到（未成交原因不完整）: %s", exc)
            missing["unfilled"] = f"执行库读不到之前的订单：{exc}"
    out["unfilled"] = unfilled

    # ---- 持仓现金
    acct = rec.get("account") or {}
    out["cash"] = _f(acct.get("cash"))
    out["total_assets"] = _f(acct.get("total_assets"))
    out["positions"] = rec.get("positions")
    if not acct:
        missing["cash"] = missing["positions"] = f"本轮没有券商账户快照（{ex.get('blocked') or ex.get('mode') or '执行层未运行'}）"

    # ---- 组合账本（模拟）当天
    paper = None
    for h in paper_history or []:
        if pd.Timestamp(h.date).normalize() == day:
            paper = h
    eq = [float(h.equity) for h in paper_history or []]

    # ---- 成本：模拟账本的成本 + 券商成交相对参考价的滑点（券商佣金接口没给 → 显式缺失）
    slip_num = slip_den = 0.0
    for f in fills:
        px, ref, q = _f(f.get("price")), _f(f.get("ref_price")), _f(f.get("qty"))
        if px and ref and q:
            slip_num += (px - ref) * q          # 买（q>0）贵了 / 卖（q<0）便宜了 都是正成本
            slip_den += abs(q) * ref
    out["costs"] = {"paper_cost_bps": _f(getattr(paper, "cost_bps", None)),
                    "broker_slippage_bps": (slip_num / slip_den * 1e4) if slip_den > 0 else None,
                    "broker_fees": None}
    missing["costs.broker_fees"] = "券商成交回报里没有逐笔佣金 / 费用字段"
    if paper is None:
        missing["costs.paper_cost_bps"] = "组合账本没有这一天"

    # ---- 净收益 / 基准超额
    broker_ret = _f(rec.get("day_return"))
    paper_ret = _f(getattr(paper, "net_ret", None))
    out["net_return"] = {"broker": broker_ret, "paper": paper_ret}
    bench = None
    try:
        close = dataset["close"]
        if day in close.index:
            r = close.pct_change().loc[day]
            bench = float(r.mean()) if r.notna().any() else None
    except Exception as exc:
        logger.warning("[forward] 基准收益算不出: %s", exc)
        missing["benchmark"] = f"基准收益算不出：{exc}"
    out["benchmark"] = {"name": BENCHMARK, "return": bench}
    out["excess_return"] = {"broker": (broker_ret - bench) if broker_ret is not None and bench is not None else None,
                            "paper": (paper_ret - bench) if paper_ret is not None and bench is not None else None}
    if broker_ret is None:
        missing["net_return.broker"] = "券商侧没有可比的前一可信快照（首日 / 对账未通过）"

    # ---- 回撤
    broker_dd = None
    if exec_store is not None:
        try:
            tas = [float(s.total_assets) for s in reversed(exec_store.snapshots(limit=1000))
                   if s.status in ("clean", "baseline", "accepted")]
            if tas:
                broker_dd = tas[-1] / max(tas) - 1.0
        except Exception as exc:
            logger.warning("[forward] 券商快照读不到（券商回撤记为 None）: %s", exc)
            missing["drawdown.broker"] = f"读不到券商快照：{exc}"
    out["drawdown"] = {"paper": (eq[-1] / max(eq) - 1.0) if eq else None, "broker": broker_dd}

    # ---- gross / net：目标（决策日）与实际（券商持仓市值 / 总资产）
    tw = result.get("execution_target") or {}
    ta = out["total_assets"]
    mv = [float((p or {}).get("market_val", 0.0)) for p in (out["positions"] or {}).values()]
    out["exposure"] = {
        "target_gross": float(sum(abs(v) for v in tw.values())) if tw else None,
        "target_net": float(sum(tw.values())) if tw else None,
        "actual_gross": (sum(abs(v) for v in mv) / ta) if ta else None,
        "actual_net": (sum(mv) / ta) if ta else None,
    }

    # ---- 资金利用率 / 对账 / 异常
    out["capital_utilization"] = ex.get("capital") if isinstance(ex, dict) else None
    if out["capital_utilization"] is None:
        missing["capital_utilization"] = "执行层未运行到资金分配"
    out["reconcile_status"] = rec.get("status")
    anomalies = []
    for k in ("blocked", "quote_error"):
        if isinstance(ex, dict) and ex.get(k):
            anomalies.append({"kind": k, "detail": ex.get(k)})
    if result.get("reason"):
        anomalies.append({"kind": "portfolio_reason", "detail": result.get("reason")})
    for k in ("discrepancies", "unresolved", "awaiting_confirmation"):
        for item in rec.get(k) or []:
            anomalies.append({"kind": k, "detail": item})
    for v in result.get("post_band_violations") or []:
        anomalies.append({"kind": "post_band_violation", "detail": v})
    out["anomalies"] = anomalies

    # ---- 正式与否
    formal, why = True, "formal"
    close_t, open_next = session_close_utc(day), (session_open_utc(nxt[0]) if nxt else None)
    if forward_run is None:
        formal, why = False, "no_formal_start"
    elif day.date() < pd.Timestamp(forward_run.started_at).date():
        formal, why = False, "before_formal_start"
    elif close_t is None or open_next is None or not (_naive_utc(close_t) <= now_n < _naive_utc(open_next)):
        formal, why = False, "not_run_in_decision_window"           # 事后补跑 / 回放 / 盘中
    elif not isinstance(ex, dict) or ex.get("blocked") or not ex.get("mode") or not rec:
        formal, why = False, "execution_not_live"
    elif not spec_hash:
        formal, why = False, "no_spec_hash"
    out["formal"], out["formal_reason"] = formal, why
    out["missing"] = missing
    return out


def _f(v) -> Optional[float]:
    """有限数 → float；None / 非数 / NaN / inf → None（"没有"不当成 0）。"""
    if v is None or isinstance(v, bool) or not np.isscalar(v):
        return None
    x = pd.to_numeric(v, errors="coerce")
    return float(x) if pd.notna(x) and np.isfinite(x) else None


def _naive_utc(t) -> datetime:
    t = pd.Timestamp(t)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.to_pydatetime()
