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


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    df.to_parquet(tmp, index=False)
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

    def load_many(self, tickers) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        facts: Dict[str, pd.DataFrame] = {}
        events: Dict[str, pd.DataFrame] = {}
        for tk in tickers:
            got = self.load(tk)
            if got is not None:
                facts[tk], events[tk] = got
        return facts, events
