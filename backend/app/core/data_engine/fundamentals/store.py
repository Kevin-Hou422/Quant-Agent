"""
store.py — 基本面事实的本地存储（Phase 10.2，接 8.1 的 PIT 目录）

存的是**事实**（每条带 `filed` 披露日）和财报公告事件，不是算好的日频面板 ——
面板是 (事实, 价格) 的确定性函数，每次加载现算；事实才是需要保全的时点资产。

只追加：重新同步时与已存内容按 (concept, start, end, filed, accn) 合并，旧记录不会被删。
EDGAR 的 companyfacts 是当前快照，万一上游撤掉某条历史事实，本地的"当时看到过什么"还在。
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from .edgar import EVENT_COLUMNS, FACT_COLUMNS, PARSER_VERSION, normalize_ticker

logger = logging.getLogger(__name__)

_FACT_KEY = ["concept", "start", "end", "filed", "accn"]
_EVENT_KEY = ["accn"]


def _atomic_parquet(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    df.to_parquet(tmp, index=index)
    tmp.replace(path)


class FundamentalsStore:
    """root/<TICKER>/facts.parquet · events.parquet · meta.json"""

    def __init__(self, root) -> None:
        self.root = Path(root)

    def _dir(self, ticker: str) -> Path:
        return self.root / normalize_ticker(ticker)

    def save(self, ticker: str, facts: pd.DataFrame, events: pd.DataFrame,
             now: Optional[float] = None) -> None:
        d = self._dir(ticker)
        old = self.load(ticker)
        if old is not None:
            facts = (pd.concat([old[0], facts], ignore_index=True)
                       .drop_duplicates(subset=_FACT_KEY, keep="last"))
            events = (pd.concat([old[1], events], ignore_index=True)
                        .drop_duplicates(subset=_EVENT_KEY, keep="last"))
        _atomic_parquet(facts[FACT_COLUMNS].reset_index(drop=True), d / "facts.parquet")
        _atomic_parquet(events[EVENT_COLUMNS].reset_index(drop=True), d / "events.parquet")
        meta = {"ticker": normalize_ticker(ticker), "synced_at": float(time.time() if now is None else now),
                "n_facts": int(len(facts)), "n_events": int(len(events)),
                "parser_version": PARSER_VERSION}
        tmp = d / "meta.json.tmp"
        tmp.write_text(json.dumps(meta), encoding="utf-8")
        tmp.replace(d / "meta.json")

    def load(self, ticker: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        d = self._dir(ticker)
        fp, ep = d / "facts.parquet", d / "events.parquet"
        if not fp.exists() or not ep.exists():
            return None
        return pd.read_parquet(fp), pd.read_parquet(ep)

    def _meta(self, ticker: str) -> Optional[dict]:
        p = self._dir(ticker) / "meta.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def synced_at(self, ticker: str) -> Optional[float]:
        m = self._meta(ticker)
        return None if m is None else float(m["synced_at"])

    def parser_version(self, ticker: str) -> Optional[int]:
        m = self._meta(ticker)
        return None if m is None else m.get("parser_version")

    # -- 每披露日指标的缓存（面板计算最慢的一步；纯函数，可随时删掉重算）------------

    def load_metrics(self, ticker: str) -> Optional[Tuple[pd.DataFrame, dict]]:
        d = self._dir(ticker)
        mp, jp = d / "metrics.parquet", d / "metrics.json"
        if not mp.exists() or not jp.exists():
            return None
        return pd.read_parquet(mp), json.loads(jp.read_text(encoding="utf-8"))

    def save_metrics(self, ticker: str, metrics: pd.DataFrame, stamp: dict) -> None:
        d = self._dir(ticker)
        _atomic_parquet(metrics.reset_index(drop=True), d / "metrics.parquet")
        tmp = d / "metrics.json.tmp"
        tmp.write_text(json.dumps(stamp), encoding="utf-8")
        tmp.replace(d / "metrics.json")

    # -- 不复权收盘价的缓存 ------------------------------------------------------
    # 不复权价是历史事实，过去的值不会再变（这正是它与复权价的区别），所以可以只追加：
    # 已取过的区间不再向数据源要。`coverage` 记的是**请求过并拿到数据的区间**，不是有值的
    # 区间 —— 上市晚的票区间开头本来就没有价格，按"有没有值"判断会每次都重取。

    def _price_dir(self, source: str) -> Path:
        return self.root / "_prices" / str(source)

    def load_raw_prices(self, source: str) -> Optional[Tuple[pd.DataFrame, Dict[str, pd.Series], dict]]:
        """(不复权收盘价宽表, {ticker: 拆股序列}, {ticker: [起, 止]})；没有缓存 → None。"""
        d = self._price_dir(source)
        fp, jp = d / "raw_close.parquet", d / "meta.json"
        if not fp.exists() or not jp.exists():
            return None
        meta = json.loads(jp.read_text(encoding="utf-8"))
        raw = pd.read_parquet(fp)
        raw.index = pd.DatetimeIndex(raw.index)
        splits = {tk: pd.Series(list(v.values()), index=pd.DatetimeIndex(list(v)), dtype=float)
                  for tk, v in meta.get("splits", {}).items()}
        return raw, splits, dict(meta.get("coverage", {}))

    def load_ambiguous(self, source: str) -> Dict[str, pd.Series]:
        """{ticker: Series(跳变日 → 倍数)}：疑似小比例送股（复审 2026-10-19 F13，估值字段要异常隔离）。"""
        jp = self._price_dir(source) / "meta.json"
        if not jp.exists():
            return {}
        meta = json.loads(jp.read_text(encoding="utf-8"))
        return {tk: pd.Series(list(v.values()), index=pd.DatetimeIndex(list(v)), dtype=float)
                for tk, v in meta.get("ambiguous", {}).items()}

    def save_report(self, report, asof) -> None:
        """最近一次挂基本面时的质量报告（§5.4 质量门读它；asof = 那份面板的末日）。"""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / "_quality.json.tmp"
        tmp.write_text(json.dumps({"asof": str(pd.Timestamp(asof).date()), "report": report.to_dict()},
                                  ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(self.root / "_quality.json")

    def load_report(self):
        """(FundamentalsReport, 面板末日)；没有 → None。"""
        from .panels import FundamentalsReport
        p = self.root / "_quality.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        return FundamentalsReport.from_dict(d["report"]), pd.Timestamp(d["asof"])

    def save_raw_prices(self, source: str, raw: pd.DataFrame, splits: Dict[str, pd.Series],
                        coverage: dict, ambiguous: Optional[Dict[str, pd.Series]] = None) -> None:
        """与已有缓存合并后落盘：已有的值保留，只补新日期 / 新标的；拆股与疑似送股按日期并集。"""
        d = self._price_dir(source)
        old_amb = self.load_ambiguous(source)
        old = self.load_raw_prices(source)
        merged_splits = {tk: s.copy() for tk, s in (old[1] if old else {}).items()}
        merged_cov = dict(old[2]) if old else {}
        if old is not None:
            raw = old[0].combine_first(raw)
        for tk, s in (splits or {}).items():
            if s is None or len(s) == 0:
                continue
            prev = merged_splits.get(tk)
            merged_splits[tk] = s if prev is None else pd.concat([prev, s[~s.index.isin(prev.index)]])
        merged_cov.update(coverage)
        merged_amb = {tk: s.copy() for tk, s in old_amb.items()}
        for tk, s in (ambiguous or {}).items():
            if s is None or len(s) == 0:
                continue
            prev = merged_amb.get(tk)
            merged_amb[tk] = s if prev is None else pd.concat([prev, s[~s.index.isin(prev.index)]])
        _atomic_parquet(raw.sort_index(), d / "raw_close.parquet", index=True)
        meta = {"coverage": merged_cov,
                "splits": {tk: {str(pd.Timestamp(k).date()): float(v) for k, v in s.sort_index().items()}
                           for tk, s in merged_splits.items()},
                "ambiguous": {tk: {str(pd.Timestamp(k).date()): float(v) for k, v in s.sort_index().items()}
                              for tk, s in merged_amb.items()}}
        tmp = d / "meta.json.tmp"
        tmp.write_text(json.dumps(meta), encoding="utf-8")
        tmp.replace(d / "meta.json")

    def load_many(self, tickers) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        facts: Dict[str, pd.DataFrame] = {}
        events: Dict[str, pd.DataFrame] = {}
        for tk in tickers:
            got = self.load(tk)
            if got is not None:
                facts[tk], events[tk] = got
        return facts, events
