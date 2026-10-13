"""
基本面测试的共用夹具装载（Phase 10）。

`tests/fixtures/edgar/*.json` 是 2026-10-12 从 SEC EDGAR 取回的**真实响应**，只做了裁剪
（留 edgar.CONCEPTS 用到的科目、近几年披露的事实），结构与字段原样保留。
四家各有用处：AAPL / NVDA（常规 + 一拆十）、GOOGL（不报毛利、多类别股没有封面页股本）、
BKNG（净利润在第三个科目下、净资产为负、2026 年一拆二十五）。
`tests/fixtures/prices/*.csv` 是同一天从 moomoo OpenD / yfinance 取回的 NVDA 一拆十前后的真实价格。

替身只替到 HTTP 这一层（`FakeSession`）：EdgarClient 的缓存、限速、错误处理全是真代码。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from app.core.data_engine.fundamentals import edgar

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
EDGAR_DIR = FIXTURES / "edgar"
PRICE_DIR = FIXTURES / "prices"
CIK = {r["ticker"]: int(r["cik_str"]) for r in
       json.loads((EDGAR_DIR / "company_tickers.json").read_text(encoding="utf-8")).values()}
UA = "Unit Test tester@example.com"


def payload(kind: str, ticker: str) -> dict:
    return json.loads((EDGAR_DIR / f"{kind}_{ticker}.json").read_text(encoding="utf-8"))


def facts(ticker: str) -> pd.DataFrame:
    return edgar.parse_company_facts(ticker, payload("companyfacts", ticker))


def events(ticker: str) -> pd.DataFrame:
    return edgar.parse_earnings_events(ticker, payload("submissions", ticker))


def moomoo_prices() -> pd.DataFrame:
    """列 = (raw|qfq, ticker)。"""
    return pd.read_csv(PRICE_DIR / "moomoo_nvda_split.csv", header=[0, 1], index_col=0,
                       parse_dates=True)


def yahoo_download_frame() -> pd.DataFrame:
    """yf.download(auto_adjust=False, actions=True, group_by='column') 的真实返回。"""
    return pd.read_csv(PRICE_DIR / "yahoo_nvda_split.csv", header=[0, 1], index_col=0,
                       parse_dates=True)


class _Resp:
    def __init__(self, status: int, body: dict) -> None:
        self.status_code = status
        self._body = body

    def json(self) -> dict:
        return self._body


class FakeSession:
    """
    按真实 URL 形状应答的 HTTP 替身。默认悲观：没登记的 URL 一律 404（EDGAR 对没有
    XBRL 事实的 CIK 就是这么回的），没带 User-Agent 的请求 403（SEC 的真实行为）。
    """

    def __init__(self, tickers=("AAPL", "NVDA")) -> None:
        self.calls: List[str] = []
        self.headers_seen: List[dict] = []
        self.routes: Dict[str, dict] = {
            edgar.TICKERS_URL: json.loads((EDGAR_DIR / "company_tickers.json").read_text(encoding="utf-8"))}
        for tk in tickers:
            self.routes[edgar.FACTS_URL.format(cik=CIK[tk])] = payload("companyfacts", tk)
            self.routes[edgar.SUBMISSIONS_URL.format(cik=CIK[tk])] = payload("submissions", tk)
        self.fail_with: Exception | None = None

    def get(self, url, headers=None, timeout=None):
        self.calls.append(url)
        self.headers_seen.append(dict(headers or {}))
        if self.fail_with is not None:
            raise self.fail_with
        if "@" not in str((headers or {}).get("User-Agent", "")):
            return _Resp(403, {})
        if url not in self.routes:
            return _Resp(404, {})
        return _Resp(200, self.routes[url])


def logged(caplog, needle: str) -> bool:
    """某条日志里出现了 needle（逐条看 message，不拼成一大段文本再找）。"""
    return any(needle in message for message in caplog.messages)


def client(session=None, cache_dir=None, **kw) -> edgar.EdgarClient:
    return edgar.EdgarClient(UA, cache_dir=cache_dir, session=session or FakeSession(),
                             sleep=lambda s: None, **kw)
