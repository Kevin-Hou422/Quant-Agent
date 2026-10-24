"""
prices.py — 算市值用的**不复权**收盘价 + 拆股记录（Phase 10.2）

为什么不能直接用数据集里的 close
--------------------------------
数据集的 close 是复权价（moomoo 前复权 / yahoo auto_adjust），历史价被分红、拆股整体缩放过。
市值 = 当天真实成交价 × 当天真实股本，两者都必须是当时的口径。

两个价格源的"不复权"不是一回事（2026-10-12 用 NVDA 2024-06-10 一拆十实测）：

    moomoo  AuType.NONE          真·不复权：一拆十前一天的收盘就是当天成交价，拆股日起是拆后价
    yahoo   auto_adjust=False    Close **已按拆股调整**（一拆十前一天给的是成交价的 1/10），只是没按分红调；
                                 拆股记录在 `Stock Splits` 列（06-10 = 10.0）

所以 yahoo 的 Close 要乘回"之后发生的拆股"才是当时的成交价；moomoo 没有拆股列，
拆股从 不复权/前复权 比值的跳变里读（06-07 → 06-10 比值从 10.0285 掉到 1.00285，正好 10 倍）。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: 不复权/复权 比值单日变化超过这个倍数才算拆股（分红只造成 1–2% 的台阶）
SPLIT_MIN_RATIO = 1.5
#: 比值跳变在 [这个, SPLIT_MIN_RATIO) 之间：可能是小比例送股，也可能是特别分红 —— 不当拆股，但告警
SPLIT_AMBIGUOUS_RATIO = 1.15
#: 拆股当天，不复权价必须反向跳了差不多同样的倍数（容许 ±30% 的当日行情）。只有复权因子跳、
#: 不复权价没动 → 是数据问题（两份不同 vintage 的复权价拼在一起），不是拆股（外部审查 2026-10-18 F13）
SPLIT_PRICE_TOL = 1.30

Splits = Dict[str, pd.Series]          # ticker → Series(拆股日 → 股本乘数；一拆十 = 10.0)


def _factor_jumps(raw_close: pd.Series, adj_close: pd.Series) -> pd.Series:
    f = (raw_close / adj_close).replace([np.inf, -np.inf], np.nan).dropna()
    return (f.shift(1) / f).dropna()


def detect_ambiguous(raw_close: pd.Series, adj_close: pd.Series) -> pd.Series:
    """
    跳变日 → 倍数：复权因子 1.15–1.5 倍的跳变（可能是小比例送股，也可能是特别分红，分不清）。
    不当拆股，但调用方要把估值字段**异常隔离**（复审 2026-10-19 F13：此前只告警，市值照样偏）。
    """
    r = _factor_jumps(raw_close, adj_close)
    small = r[((r >= SPLIT_AMBIGUOUS_RATIO) & (r < SPLIT_MIN_RATIO))
              | ((r <= 1.0 / SPLIT_AMBIGUOUS_RATIO) & (r > 1.0 / SPLIT_MIN_RATIO))]
    return small.round(4)


def detect_splits(raw_close: pd.Series, adj_close: pd.Series) -> pd.Series:
    """
    拆股日 → 股本乘数（一拆十 = 10.0，十并一 = 0.1）。
    复权因子 f = raw/adj；拆股当天 f 跳变，乘数 = 前一日 f / 当日 f。
    """
    f = (raw_close / adj_close).replace([np.inf, -np.inf], np.nan).dropna()
    r = _factor_jumps(raw_close, adj_close)
    big = r[(r >= SPLIT_MIN_RATIO) | (r <= 1.0 / SPLIT_MIN_RATIO)]
    small = detect_ambiguous(raw_close, adj_close)
    if len(small):
        logger.warning("[fundamentals] %s 复权因子有 %d 次 %.2f–%.2f 倍的跳变（%s）—— 可能是小比例送股或特别分红，"
                       "**没有**当作拆股；估值字段在新股本到来前异常隔离（见 panels.build_panels）",
                       raw_close.name, len(small), SPLIT_AMBIGUOUS_RATIO, SPLIT_MIN_RATIO,
                       [str(pd.Timestamp(d).date()) for d in small.index])
    raw = raw_close.reindex(f.index)
    jump = (raw / raw.shift(1)).reindex(big.index)
    ok = (jump * big).between(1.0 / SPLIT_PRICE_TOL, SPLIT_PRICE_TOL)
    if (~ok).any():
        logger.warning("[fundamentals] %s 有 %d 次复权因子跳变没有伴随不复权价的反向跳变（%s）—— 当作数据问题，不当拆股",
                       raw_close.name, int((~ok).sum()), [str(pd.Timestamp(d).date()) for d in big.index[~ok]])
    return big[ok].round(4)


def unsplit(split_adjusted_close: pd.Series, splits: pd.Series) -> pd.Series:
    """把"已按拆股调整"的收盘价还原成当时的成交价：乘上该日**之后**发生的全部拆股乘数。"""
    if splits is None or len(splits) == 0:
        return split_adjusted_close
    s = splits.sort_index()
    after = np.concatenate([np.cumprod(s.to_numpy(dtype=float)[::-1])[::-1], [1.0]])
    # 日期 d 之后（不含 d 当天：拆股日的收盘价已经是拆后的）的拆股数 = 从 searchsorted(right) 开始
    pos = np.searchsorted(s.index.values, split_adjusted_close.index.values, side="right")
    return split_adjusted_close * after[pos]


def moomoo_raw_and_splits(tickers: List[str], start: str, end: str, *, host: str, port: int,
                          adj_close: Optional[pd.DataFrame] = None,
                          provider_factory=None, ambiguous_out: Optional[Splits] = None) -> Tuple[pd.DataFrame, Splits]:
    """
    adj_close : 同一次加载里已经拿到的前复权 close（单一 vintage）；None → 自己再取一次。
                **不要传 PIT 增量拼出来的面板** —— 那是多个 vintage 的复权价拼接，拆股日前后
                的比值不跳变，拆股会漏检。
    ambiguous_out : 给了就把疑似小比例送股（detect_ambiguous）按 ticker 填进去（yahoo 路径的拆股列本身含送股）。
    """
    if provider_factory is None:
        from ..providers.moomoo_provider import MoomooProvider
        provider_factory = lambda autype: MoomooProvider(host=host, port=port, autype=autype)  # noqa: E731
    raw = provider_factory(None).fetch(tickers, start=start, end=end, fields=["close"])["close"]
    if adj_close is None:
        adj_close = provider_factory("qfq").fetch(tickers, start=start, end=end, fields=["close"])["close"]
    splits: Splits = {}
    for tk in raw.columns:
        if tk in adj_close.columns:
            adj = adj_close[tk].reindex(raw.index)
            s = detect_splits(raw[tk], adj)
            if not s.empty:
                splits[tk] = s
            if ambiguous_out is not None:
                a = detect_ambiguous(raw[tk], adj)
                if not a.empty:
                    ambiguous_out[tk] = a
    return raw, splits


def yahoo_raw_and_splits(tickers: List[str], start: str, end: str, *,
                         download=None) -> Tuple[pd.DataFrame, Splits]:
    if download is None:
        import yfinance as yf
        download = yf.download
    tickers = [t.upper() for t in tickers]
    end_exclusive = (pd.Timestamp(end).normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = download(tickers=tickers, start=start, end=end_exclusive, auto_adjust=False,
                  actions=True, progress=False, group_by="column")
    if df is None or df.empty:
        raise ValueError("yfinance 返回空数据（不复权价）")
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples([(c, tickers[0]) for c in df.columns])
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    close = df["Close"].reindex(columns=tickers).astype(float)
    if "Stock Splits" not in df.columns.get_level_values(0):
        raise ValueError("yfinance 响应里没有 Stock Splits 列 —— 无法还原不复权价")
    sp = df["Stock Splits"].reindex(columns=tickers).astype(float)
    raw = pd.DataFrame(index=close.index, columns=tickers, dtype=float)
    splits: Splits = {}
    for tk in tickers:
        s = sp[tk]
        s = s[(s > 0) & (s != 1.0)]
        if not s.empty:
            splits[tk] = s
        raw[tk] = unsplit(close[tk], s)
    return raw, splits
