"""
service.py — 把基本面字段挂到一个价格面板上（Phase 10.2 的装配层）

    sync()                从 EDGAR 拉事实 → FundamentalsStore（按 ticker，日内不重复拉）
    attach_fundamentals() 读 store + 取不复权价 → build_panels → 写进 data 字典

失败语义（`ENABLE_FUNDAMENTALS=true` 是显式开启，配置错了必须响）
----------------------------------------------------------------
- 没配 `EDGAR_USER_AGENT` 且本地 store 里一只票都没有        → 抛 FundamentalsUnavailable
- EDGAR 暂时不可用但 store 里有数据                          → 用 store（数据只会**滞后**，不会前视），告警
- 个别 ticker 没有事实（ETF、外国发行人、刚上市）             → 该列 NaN，记入报告
- 取不到不复权价                                             → 估值字段**不挂**（引用它的表达式报错），其余照常
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from .edgar import (
    FACT_COLUMNS, PARSER_VERSION, EdgarClient, EdgarError, latest_report_date, parse_company_facts,
    parse_earnings_events,
)
from .panels import FundamentalsReport, build_panels
from .store import FundamentalsStore

logger = logging.getLogger(__name__)


class FundamentalsUnavailable(RuntimeError):
    """开启了基本面但一条事实都拿不到 —— 配置问题，不能当成"没有数据"悄悄过去。"""


@dataclass
class SyncReport:
    synced:    List[str] = field(default_factory=list)      # 重新下载并解析了 companyfacts
    unchanged: List[str] = field(default_factory=list)      # 查了文件清单，没有新的年报 / 季报
    cached:    List[str] = field(default_factory=list)      # max_age 内同步过，一个请求都没发
    failed:    Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"synced": len(self.synced), "unchanged": len(self.unchanged),
                "cached": len(self.cached), "failed": dict(self.failed)}


def sync(tickers: List[str], client: EdgarClient, store: FundamentalsStore, *,
         max_age_h: float = 20.0, now: Optional[float] = None) -> SyncReport:
    """
    逐只同步并落盘。单只失败不影响其余。

    每只票分三档，开销从小到大：
      cached     `max_age_h` 内同步过 → 不发请求；
      unchanged  取 submissions（文件清单，很小），最近一份年报 / 季报的申报日不晚于本地事实的
                 最新披露日 → 只更新财报事件，不下 companyfacts（每家 5–8 MB，财报一季才一份）；
      synced     有新财报、或本地没有、或落盘时的解析口径（PARSER_VERSION）与现在不同 → 重下重解析。
    """
    rep = SyncReport()
    t_now = time.time() if now is None else float(now)
    for tk in tickers:
        at = store.synced_at(tk)
        if (at is not None and t_now - at <= max_age_h * 3600.0
                and store.parser_version(tk) == PARSER_VERSION):
            rep.cached.append(tk)
            continue
        try:
            subs = client.submissions(tk)
            events = parse_earnings_events(tk, subs)
            have = store.load(tk) if store.parser_version(tk) == PARSER_VERSION else None
            newest = latest_report_date(subs)
            if (have is not None and newest is not None and not have[0].empty
                    and pd.Timestamp(have[0]["filed"].max()) >= newest):
                store.save(tk, have[0], events, now=t_now)
                rep.unchanged.append(tk)
                continue
            parts = [parse_company_facts(tk, client.company_facts(tk))]
            parts += [parse_company_facts(tk, old) for old in client.predecessor_facts(tk)]
            parts = [p for p in parts if not p.empty]
            facts = (pd.concat(parts, ignore_index=True) if parts
                     else pd.DataFrame(columns=FACT_COLUMNS))
        except EdgarError as exc:
            logger.debug("[fundamentals] %s 同步失败: %s", tk, exc)
            rep.failed[tk] = str(exc)
            continue
        if facts.empty:
            rep.failed[tk] = "EDGAR 有这家公司，但没有 10-K/10-Q 里的 us-gaap 事实（外国发行人 / 基金？）"
            continue
        store.save(tk, facts, events, now=t_now)
        rep.synced.append(tk)
    if rep.failed:
        logger.warning("[fundamentals] %d 只票同步失败（这些列将为 NaN 或沿用旧数据）: %s",
                       len(rep.failed), dict(list(rep.failed.items())[:8]))
    return rep


def cached_metrics(store: FundamentalsStore, ticker: str, facts: pd.DataFrame) -> pd.DataFrame:
    """
    `metric_events(facts)` 的增量缓存。某披露日 F 的指标只取决于 `filed <= F` 的事实，
    所以已算过的行不会变，新财报出来只需补算新披露日。

    缓存失效条件（任一不符 → 全量重算）：算法版本变了；"截至缓存末日"的事实条数变了
    （事实只追加，条数变 = 有旧披露日的事实被补进来）。
    """
    from .panels import METRICS_VERSION, metric_events
    got = store.load_metrics(ticker)
    if got is not None:
        old, stamp = got
        upto = pd.Timestamp(stamp.get("upto")) if stamp.get("upto") else None
        same = (stamp.get("version") == METRICS_VERSION and upto is not None
                and int((facts["filed"] <= upto).sum()) == stamp.get("n_facts_upto"))
        if same:
            if not (facts["filed"] > upto).any():
                return old
            fresh = metric_events(facts, since=upto + pd.Timedelta(days=1))
            out = pd.concat([old, fresh], ignore_index=True)
            _stamp_and_save(store, ticker, out, facts)
            return out
    out = metric_events(facts)
    _stamp_and_save(store, ticker, out, facts)
    return out


def _stamp_and_save(store: FundamentalsStore, ticker: str, metrics: pd.DataFrame,
                    facts: pd.DataFrame) -> None:
    from .panels import METRICS_VERSION
    if metrics.empty:
        return
    upto = pd.Timestamp(metrics["filed"].max())
    store.save_metrics(ticker, metrics, {"version": METRICS_VERSION, "upto": upto.isoformat(),
                                         "n_facts_upto": int((facts["filed"] <= upto).sum())})


def fundamentals_dir(settings) -> Path:
    explicit = str(getattr(settings, "fundamentals_dir", "") or "").strip()
    return Path(explicit) if explicit else Path(str(settings.pit_store_dir)) / "fundamentals"


def fetch_raw_and_splits(tickers: List[str], start: str, end: str, settings,
                         adj_close: Optional[pd.DataFrame] = None):
    """按 `price_source` 取不复权收盘价 + 拆股记录（两个源口径不同，见 prices.py）。"""
    from . import prices
    if str(getattr(settings, "price_source", "yahoo")) == "moomoo":
        return prices.moomoo_raw_and_splits(
            tickers, start, end, host=getattr(settings, "moomoo_host", "127.0.0.1"),
            port=int(getattr(settings, "moomoo_port", 11111)), adj_close=adj_close)
    return prices.yahoo_raw_and_splits(tickers, start, end)


def _today() -> pd.Timestamp:
    """"今天"的唯一出处 —— 测试据此把日历钉住，否则"是否过期"会随真实日期漂移。"""
    return pd.Timestamp.today()


def store_status(store: FundamentalsStore, universe: List[str],
                 today: Optional[pd.Timestamp] = None) -> dict:
    """
    本地已有的基本面数据的状态 —— **只读磁盘，不发任何请求**（给界面用）。

    每只票：最近披露日（= 这批数字从哪天之后可见）、对应的财报期末日、是否过期、
    按最近一份财报哪些字段算得出来。字段覆盖率 = 算得出来且未过期的票 / 全部票。
    没有逐披露日指标缓存的票（还没被加载过一次）只报披露日，字段列为未知。
    """
    from .panels import ALL_FIELDS, MAX_STALE_DAYS
    today = _today() if today is None else pd.Timestamp(today)
    if today.tzinfo is not None:                 # 调用方给的是带时区的时刻（预检的 now）→ 取 UTC 日期
        today = today.tz_convert(None)
    today = today.normalize()
    rows, have_count = [], {f: 0 for f in ALL_FIELDS}
    for tk in universe:
        got = store.load(tk)
        if got is None or got[0].empty:
            rows.append({"ticker": tk, "has_facts": False, "first_filed": None, "last_filed": None,
                         "last_period_end": None, "stale": False, "fields": None, "synced_at": None})
            continue
        facts, events = got
        last_filed = pd.Timestamp(facts["filed"].max())
        stale = (today - last_filed).days > MAX_STALE_DAYS
        fields = None
        cached = store.load_metrics(tk)
        if cached is not None and not cached[0].empty:
            m = cached[0].sort_values("filed").iloc[-1]
            ok = lambda k: pd.notna(m[k])                                    # noqa: E731
            pos = lambda k: pd.notna(m[k]) and float(m[k]) > 0               # noqa: E731
            fields = {
                "market_cap": ok("shares"),
                "earnings_yield": ok("shares") and ok("ni_ttm"),
                "book_to_price": ok("shares") and pos("equity"),
                "roe": ok("ni_ttm") and pos("equity"),
                "gross_margin": ok("gp_ttm") and pos("rev_ttm"),
                "asset_growth": ok("assets") and pos("assets_prev"),
                "sue": ok("sue"),
                "days_since_earnings": not events.empty,
            }
            fields = {k: bool(v) for k, v in fields.items()}
            if not stale:
                for k, v in fields.items():
                    have_count[k] += int(v)
        at = store.synced_at(tk)
        rows.append({
            "ticker": tk, "has_facts": True,
            "last_filed": str(last_filed.date()),
            # 封面页股本的日期是"截至某日"，晚于财报期末 → 不算在内
            "last_period_end": str(pd.Timestamp(
                facts.loc[(facts["filed"] == facts["filed"].max()) & (facts["concept"] != "shares_cover"),
                          "end"].max()).date()),
            "first_filed": str(pd.Timestamp(facts["filed"].min()).date()),
            "stale": bool(stale), "fields": fields,
            "synced_at": None if at is None else pd.Timestamp(at, unit="s", tz="UTC").isoformat(timespec="seconds"),
        })
    n = max(len(universe), 1)
    return {
        "n_tickers": len(universe),
        "n_with_facts": sum(1 for r in rows if r["has_facts"]),
        "n_stale": sum(1 for r in rows if r["stale"]),
        "coverage": {f: round(c / n, 4) for f, c in have_count.items()},
        "tickers": rows,
        "max_stale_days": MAX_STALE_DAYS,
    }


def cached_raw_and_splits(store: FundamentalsStore, tickers: List[str], index: pd.DatetimeIndex,
                          settings, adj_close: Optional[pd.DataFrame] = None,
                          fetch: Optional[Callable] = None):
    """
    不复权收盘价 + 拆股记录，优先读本地缓存，只向数据源要没取过的部分。

    每只票按缓存里记的已取区间 [起, 止] 分三种：
      区间已覆盖 [index 首日, index 末日]      → 不发请求；
      只差尾巴（末日更晚）                     → 从"止"那天起取增量（重叠一天：拆股靠相邻两天的
                                                 不复权/复权比值识别，少了前一天就看不出来）；
      没取过、或首日更早                       → 整段取。
    数据源没返回的票（代码不对 / 没权限）不记区间，下次还会再试。
    `index` 的首末日来自价格面板，都是真实有 bar 的交易日（周末加载不会因此多取一次）。
    """
    fetch = fetch or (lambda tk, s, e, adj: fetch_raw_and_splits(tk, s, e, settings, adj))
    source = str(getattr(settings, "price_source", "yahoo") or "yahoo")
    start, end = pd.Timestamp(index[0]).normalize(), pd.Timestamp(index[-1]).normalize()
    got = store.load_raw_prices(source)
    coverage = got[2] if got else {}
    full, tail = [], {}
    for tk in tickers:
        c = coverage.get(tk)
        if c is None or start < pd.Timestamp(c[0]):
            full.append(tk)
        elif end > pd.Timestamp(c[1]):
            tail.setdefault(pd.Timestamp(c[1]), []).append(tk)
    jobs = ([(full, start)] if full else []) + [(tks, since) for since, tks in sorted(tail.items())]
    for tks, since in jobs:
        raw, splits = fetch(tks, since.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), adj_close)
        raw = raw.copy()
        raw.index = pd.DatetimeIndex(raw.index).normalize()
        have = [tk for tk in tks if tk in raw.columns and raw[tk].notna().any()]
        if not have:
            continue
        # 区间的末端记**实际拿到的最后一根**，不是请求的末日：末日那根还没出来时（盘中 / 数据源延迟），
        # 下次加载会再取一次尾巴，而不是把那天永远记成取过了。
        new_cov = {tk: [min(since, pd.Timestamp(coverage[tk][0])).strftime("%Y-%m-%d")
                        if tk in coverage else since.strftime("%Y-%m-%d"),
                        pd.Timestamp(raw[tk].last_valid_index()).strftime("%Y-%m-%d")] for tk in have}
        store.save_raw_prices(source, raw[have], {tk: s for tk, s in (splits or {}).items() if tk in have},
                              new_cov)
    if jobs:
        got = store.load_raw_prices(source)
        logger.info("[fundamentals] 不复权价：整段取 %d 只，取增量 %d 只，其余 %d 只读缓存",
                    len(full), sum(len(v) for v in tail.values()),
                    len(tickers) - len(full) - sum(len(v) for v in tail.values()))
    cols = [tk for tk in tickers if got is not None and tk in got[0].columns]
    if not cols:
        raise ValueError("数据源没有返回任何不复权价")
    raw_all, splits_all, _ = got
    return raw_all.reindex(index=index, columns=cols), {tk: s for tk, s in splits_all.items() if tk in cols}


def attach_fundamentals(
    data: Dict[str, pd.DataFrame],
    universe: List[str],
    *,
    settings=None,
    adj_close: Optional[pd.DataFrame] = None,
    client: Optional[EdgarClient] = None,
    store: Optional[FundamentalsStore] = None,
    price_fetch: Optional[Callable] = None,
) -> FundamentalsReport:
    """
    就地给 `data` 加上基本面字段，返回覆盖率报告。

    adj_close : 与 `data["close"]` **同一次抓取**得到的复权价（moomoo 路径用它识别拆股，省一次请求）。
                `data` 来自 PIT 增量拼接时传 None —— 那种面板是多个 vintage 拼的，不能用来识别拆股。
    """
    if settings is None:
        from app.config import settings as _s
        settings = _s
    close = data.get("close")
    if close is None or close.empty:
        raise FundamentalsUnavailable("面板里没有 close，无法确定交易日序列")
    index = pd.DatetimeIndex(close.index)
    store = store or FundamentalsStore(fundamentals_dir(settings))

    ua = str(getattr(settings, "edgar_user_agent", "") or "").strip()
    if client is None and ua:
        client = EdgarClient(ua, cache_dir=store.root / "_raw",
                             max_age_h=float(getattr(settings, "fundamentals_max_age_h", 20.0)))
    if client is not None:
        sync(universe, client, store,
             max_age_h=float(getattr(settings, "fundamentals_max_age_h", 20.0)))
    else:
        logger.warning("[fundamentals] 未配置 EDGAR_USER_AGENT —— 只用本地已有的事实，不会更新")

    facts, events = store.load_many(universe)
    if not facts:
        raise FundamentalsUnavailable(
            "ENABLE_FUNDAMENTALS 已开启，但一只票的事实都没有。"
            + ("EDGAR 全部请求失败，见上方日志。" if client is not None else
               "请配置 EDGAR_USER_AGENT（形如 \"Your Name you@example.com\"，SEC 要求声明身份）。"))

    raw = splits = None
    try:
        if price_fetch is not None:
            raw, splits = price_fetch()
        else:
            raw, splits = cached_raw_and_splits(store, list(universe), index, settings, adj_close)
    except Exception as exc:
        logger.error("[fundamentals] 取不复权收盘价失败 → 估值字段（earnings_yield / book_to_price）"
                     "本次**不提供**；其余基本面字段照常: %s", exc)

    metrics = {tk: cached_metrics(store, tk, fx) for tk, fx in facts.items()}
    panels, rep = build_panels(index, list(universe), facts, events, raw, splits, metrics)
    for name, df in panels.items():
        data[name] = df.reindex(index=close.index, columns=close.columns)
    logger.info("[fundamentals] 已挂字段 %s | 末日覆盖率 %s | 无事实 %d 只 | 拆股 %d 只",
                sorted(panels), {k: round(v, 2) for k, v in rep.coverage.items()},
                len(rep.no_facts), len(rep.splits))
    if rep.short_history:
        logger.warning("[fundamentals] %d 只票的事实从面板起点一年多以后才开始（新上市，或重组换了申报主体 —— "
                       "后者需要在 edgar.PREDECESSOR_CIKS 登记旧主体）: %s",
                       len(rep.short_history), rep.short_history)
    if rep.stale:
        logger.warning("[fundamentals] %d 只票的最近披露已过期（> 200 天没有新财报进来，末日字段为 NaN）—— "
                       "上游缺数据或已停止申报: %s", len(rep.stale), rep.stale)
    return rep
