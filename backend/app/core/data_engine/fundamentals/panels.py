"""
panels.py — 基本面事实 → 日频稀疏字段面板（Phase 10.1 / 10.2 / 10.3）

可见性规则（唯一的一条，所有字段共用）
--------------------------------------
一条在 F 日披露的事实，从 **F 之后的第一个交易日**起可见。`filed` 只有日期没有时刻，
而 10-Q / 10-K 大多在收盘后提交 —— 当天收盘价里不含这条信息，所以不能让它影响 F 日的信号。
之后按日 ffill；超过 MAX_STALE_DAYS 没有新披露 → NaN（退市 / 停止申报的公司不能永远
挂着最后一份财报的数字）。

字段
----
    market_cap          不复权收盘价 × 股本（美元）
    earnings_yield      净利润(TTM) / 市值                    —— E/P
    book_to_price       股东权益 / 市值（权益 ≤ 0 → NaN）       —— B/P
    roe                 净利润(TTM) / 股东权益（权益 ≤ 0 → NaN）
    gross_margin        毛利(TTM) / 收入(TTM)；没直接报毛利的用 收入 − 营业成本
    asset_growth        总资产 / 一年前总资产 − 1
    sue                 标准化盈利超预期（见 pit.sue）
    days_since_earnings 距最近一次财报公告（8-K 2.02）的交易日数

市值与拆股
----------
市值 = **不复权**收盘价 × 股本。股本来自财报，口径是披露时的；拆股日到下一份财报之间，
股价已是拆后的、已知股本还是拆前的（NVDA 2024-06-10 一拆十，下一份 10-Q 在 08-28）——
不调整的话这两个半月市值低估 10 倍。拆股记录由 prices.py 给出，
对"口径日"之后发生的拆股做累乘调整。净利润、权益是美元金额，不受拆股影响。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import pit

logger = logging.getLogger(__name__)

FUNDAMENTAL_FIELDS = ("market_cap", "earnings_yield", "book_to_price", "roe", "gross_margin",
                      "asset_growth", "sue")
EVENT_FIELDS = ("days_since_earnings",)
ALL_FIELDS = FUNDAMENTAL_FIELDS + EVENT_FIELDS
#: 需要市值（= 不复权价 × 股本）的字段
VALUATION_FIELDS = ("market_cap", "earnings_yield", "book_to_price")

#: 最近一次披露超过这么多天 → 字段置 NaN（季报间隔约 91 天，容许错过一期）
MAX_STALE_DAYS = 200
#: 每个披露日回看多久的事实（按期末日）。SUE 最吃历史：8 个同比差 + 1 年基期 ≈ 3.3 年
METRIC_LOOKBACK = pd.Timedelta(days=int(365 * 4.5))
METRIC_COLUMNS = ["filed", "ni_ttm", "rev_ttm", "gp_ttm", "equity", "assets", "assets_prev",
                  "shares", "shares_basis", "sue"]
#: metric_events 的算法版本。改了计算口径就 +1 —— service 层的缓存据此失效。
METRICS_VERSION = 2
#: 估值合理性护栏：超出即视为股本口径错误（多类别股等），置 NaN 并记入报告
MAX_ABS_EARNINGS_YIELD, MAX_BOOK_TO_PRICE = 2.0, 20.0


@dataclass
class FundamentalsReport:
    coverage:        Dict[str, float] = field(default_factory=dict)    # 字段 → 末日非空占比
    no_facts:        List[str] = field(default_factory=list)
    #: 有事实、但最近一次披露距面板末日超过 MAX_STALE_DAYS（上游缺数据 / 已停止申报）→ 末日为 NaN
    stale:           Dict[str, str] = field(default_factory=dict)      # ticker → 最近披露日
    splits:          Dict[str, Dict[str, float]] = field(default_factory=dict)
    sanity_rejected: Dict[str, int] = field(default_factory=dict)      # ticker → 被置 NaN 的天数

    def to_dict(self) -> dict:
        return {"coverage": {k: round(v, 4) for k, v in self.coverage.items()},
                "no_facts": list(self.no_facts), "stale": dict(self.stale), "splits": self.splits,
                "sanity_rejected": dict(self.sanity_rejected)}


def _split_factor(splits: Optional[pd.Series], basis: np.ndarray,
                  dates: pd.DatetimeIndex) -> np.ndarray:
    """口径日 basis[i]（不含）到 dates[i]（含）之间拆股乘数的累乘。"""
    if splits is None or len(splits) == 0:
        return np.ones(len(dates))
    splits = splits.sort_index()
    sd = splits.index.values
    cum = np.concatenate([[1.0], np.cumprod(splits.to_numpy(dtype=float))])
    upto = cum[np.searchsorted(sd, dates.values, side="right")]
    base = cum[np.searchsorted(sd, basis, side="right")]
    return upto / base


def metric_events(facts: pd.DataFrame, since: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    每个披露日算一次"当天已知的全部"：TTM 净利润 / 收入 / 毛利、权益、总资产及其一年前值、
    股本（及其口径日）、SUE。只用 `filed <= F` 的事实 —— 时点语义的唯一入口。
    """
    rows = []
    # 只带计算用得到的列，且每个披露日只看期末日在 LOOKBACK 内的事实：
    # SUE 要 8 个季度的同比差（= 3 年多一点的单季数），其余指标更短。
    slim = facts[["concept", "start", "end", "filed", "val", "accn"]].astype({"concept": object,
                                                                             "accn": object})
    for f in pit.filing_dates(facts):
        if since is not None and f < since:
            continue
        vis = slim[(slim["filed"] <= f) & (slim["end"] > f - METRIC_LOOKBACK)]
        ni, rev, gp = pit.ttm(vis, "net_income"), pit.ttm(vis, "revenue"), pit.ttm(vis, "gross_profit")
        if gp is None and rev is not None:
            # 没直接报毛利 → 收入 − 营业成本（两者必须是同一个期末日的 TTM）
            cost = pit.ttm(vis, "cost_of_revenue")
            if cost is not None and cost[1] == rev[1]:
                gp = (rev[0] - cost[0], rev[1])
        eq, at = pit.latest_instant(vis, "equity"), pit.latest_instant(vis, "assets")
        shares, basis = _shares(vis)
        s = pit.sue(vis)
        rows.append({
            "filed": f,
            "ni_ttm": np.nan if ni is None else ni[0],
            "rev_ttm": np.nan if rev is None else rev[0],
            "gp_ttm": np.nan if gp is None else gp[0],
            "equity": np.nan if eq is None else float(eq["val"]),
            "assets": np.nan if at is None else float(at["val"]),
            "assets_prev": np.nan if at is None else _or_nan(
                pit.instant_near(vis, "assets", at["end"] - pit.YEAR)),
            "shares": shares, "shares_basis": basis,
            "sue": np.nan if s is None else s[0],
        })
    out = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    for c in ("filed", "shares_basis"):
        out[c] = pd.to_datetime(out[c])
    return out


def _or_nan(v: Optional[float]) -> float:
    return np.nan if v is None else float(v)


def _shares(vis: pd.DataFrame):
    """
    (股本, 口径日)。候选里取**期末日最新**的；并列时 封面页 > 资产负债表 > 稀释加权平均。
    口径日：封面页股本是"截至某日"的实数 → 用它自己的日期；另外两种在后续文件里会被按
    披露时的拆股口径重述 → 用披露日。
    """
    cands = []
    cover = pit.latest_instant(vis, "shares_cover")
    if cover is not None:
        cands.append((cover["end"], 2, float(cover["val"]), cover["end"]))
    bs = pit.latest_instant(vis, "shares_bs")
    if bs is not None:
        cands.append((bs["end"], 1, float(bs["val"]), bs["filed"]))
    dil = pit.flows(vis, "shares_diluted")
    dil = dil[(dil["days"] >= pit.QUARTER_MIN) & (dil["days"] <= pit.QUARTER_MAX)]
    if not dil.empty:
        d = dil.sort_values(["end", "filed"]).iloc[-1]
        cands.append((d["end"], 0, float(d["val"]), d["filed"]))
    if not cands:
        return np.nan, pd.NaT
    _, _, val, basis = max(cands, key=lambda c: (c[0], c[1]))
    return (val, basis) if val > 0 else (np.nan, pd.NaT)


def _step_frame(events: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """披露事件 → 日频阶梯：F 之后第一个交易日起生效，ffill，过期置 NaN。"""
    if events.empty:
        return pd.DataFrame(index=index, columns=events.columns.drop("filed"), dtype=float)
    ev = events.sort_values("filed")
    pos = np.searchsorted(index.values, ev["filed"].values, side="right")   # 严格晚于披露日
    keep = pos < len(index)
    ev = ev[keep].assign(_eff=index[pos[keep]]).drop_duplicates("_eff", keep="last")
    out = ev.set_index("_eff").reindex(index, method="ffill")
    stale = (index.to_series().values - out["filed"].values) > np.timedelta64(MAX_STALE_DAYS, "D")
    out.loc[stale | out["filed"].isna().values, :] = np.nan
    return out.drop(columns=["filed"])


def earnings_days(events: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """
    距最近一次财报公告的**交易日**数（生效日 = 0）。

    生效日 = 申报日**之后**的第一个交易日，与基本面事实同一条可见性规则。财报多在盘后发布
    （申报日当天收盘价里没有这条信息）；盘前发布的会因此晚记一天 —— 宁可晚一天，
    也不依赖 EDGAR 那个不可信的受理时间戳去分辨盘前盘后（见 edgar.py）。
    """
    out = pd.Series(np.nan, index=index)
    if events.empty or len(index) == 0:
        return out
    days = pd.to_datetime(events["filing_date"]).dropna()
    # 面板起点之前的公告：不知道中间隔了几个交易日，数不出来 → 不用（起点后第一次公告前为 NaN）。
    # 留 4 天余量：起点前最后一个交易日（隔周末 / 一个假日）的公告，生效日正是面板第一天。
    days = days[days >= index[0] - pd.Timedelta(days=4)].values
    pos = np.searchsorted(index.values, days, side="right")
    pos = np.unique(pos[pos < len(index)])
    if len(pos) == 0:
        return out
    last = np.full(len(index), -1)
    last[pos] = pos
    last = np.maximum.accumulate(last)
    out[:] = np.where(last >= 0, np.arange(len(index)) - last, np.nan)
    return out


def build_panels(
    index: pd.DatetimeIndex,
    tickers: List[str],
    facts: Dict[str, pd.DataFrame],
    events: Dict[str, pd.DataFrame],
    raw_close: Optional[pd.DataFrame] = None,
    splits: Optional[Dict[str, pd.Series]] = None,
    metrics: Optional[Dict[str, pd.DataFrame]] = None,
) -> tuple:
    """
    返回 ({字段: DataFrame(index × tickers)}, FundamentalsReport)。

    raw_close : **不复权**收盘价（见 prices.py）。None → 市值及估值两项（market_cap /
                earnings_yield / book_to_price）**不出现在返回里** —— 没有当时的成交价就算不出市值，
                字段缺席（引用它的表达式会报错）好过一列用复权价凑出来的数。
    splits    : ticker → Series(拆股日 → 股本乘数)。
    metrics   : ticker → 已算好的 `metric_events` 结果（service 层按披露日增量缓存）；缺的现算。
    """
    index = pd.DatetimeIndex(index)
    fields = ALL_FIELDS if raw_close is not None else tuple(
        f for f in ALL_FIELDS if f not in VALUATION_FIELDS)
    panels = {f: pd.DataFrame(np.nan, index=index, columns=tickers) for f in fields}
    rep = FundamentalsReport()
    splits = splits or {}
    since = index[0] - pd.Timedelta(days=MAX_STALE_DAYS + 30)
    for tk in tickers:
        fx = facts.get(tk)
        ev = events.get(tk)
        if ev is not None and not ev.empty:
            panels["days_since_earnings"][tk] = earnings_days(ev, index).values
        if fx is None or fx.empty:
            rep.no_facts.append(tk)
            continue
        mev = metrics[tk] if metrics and tk in metrics else metric_events(fx, since=since)
        m = _step_frame(mev, index)
        last_filed = pd.Timestamp(fx["filed"].max())
        if len(index) and (index[-1] - last_filed).days > MAX_STALE_DAYS:
            rep.stale[tk] = str(last_filed.date())
        eq = m["equity"].to_numpy(dtype=float)
        ni = m["ni_ttm"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            panels["roe"][tk] = np.where(eq > 0, ni / eq, np.nan)
            rev = m["rev_ttm"].to_numpy(dtype=float)
            panels["gross_margin"][tk] = np.where(rev > 0, m["gp_ttm"].to_numpy(dtype=float) / rev, np.nan)
            prev = m["assets_prev"].to_numpy(dtype=float)
            panels["asset_growth"][tk] = np.where(prev > 0, m["assets"].to_numpy(dtype=float) / prev - 1.0,
                                                  np.nan)
        panels["sue"][tk] = m["sue"].to_numpy(dtype=float)

        if raw_close is None or tk not in raw_close:
            continue
        raw = raw_close[tk].reindex(index)
        sp = splits.get(tk)
        if sp is not None and len(sp):
            rep.splits[tk] = {str(pd.Timestamp(d).date()): float(r) for d, r in sp.items()}
        basis = pd.to_datetime(m["shares_basis"]).values
        factor = _split_factor(sp, basis, index)
        mcap = raw.to_numpy(dtype=float) * m["shares"].to_numpy(dtype=float) * factor
        with np.errstate(divide="ignore", invalid="ignore"):
            ey = np.where(mcap > 0, ni / mcap, np.nan)
            # 净资产 ≤ 0（回购把账面权益买成负的：MCD、SBUX、PM …）时 B/P 没有"便宜 / 贵"的含义
            bp = np.where((mcap > 0) & (eq > 0), eq / mcap, np.nan)
        bad = (np.abs(ey) > MAX_ABS_EARNINGS_YIELD) | (bp > MAX_BOOK_TO_PRICE)
        if bad.any():
            rep.sanity_rejected[tk] = int(bad.sum())
            logger.warning("[fundamentals] %s 有 %d 天的估值超出合理范围（疑似股本口径问题，"
                           "如多类别股）→ 这些天的市值、E/P、B/P 置 NaN", tk, int(bad.sum()))
            ey, bp = np.where(bad, np.nan, ey), np.where(bad, np.nan, bp)
        panels["market_cap"][tk] = np.where((mcap > 0) & ~bad, mcap, np.nan)
        panels["earnings_yield"][tk] = ey
        panels["book_to_price"][tk] = bp
    for f, df in panels.items():
        rep.coverage[f] = float(df.iloc[-1].notna().mean()) if len(df) else 0.0
    return panels, rep
