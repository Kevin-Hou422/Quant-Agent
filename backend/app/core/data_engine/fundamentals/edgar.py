"""
edgar.py — SEC EDGAR 客户端 + XBRL 事实解析（Phase 10.2）

为什么是 EDGAR 而不是 yfinance `.quarterly_financials`
------------------------------------------------------
基本面数据要进回测，唯一不能错的是**"这个数字哪天开始能被人知道"**。yfinance 只给财报
**期末日**，不给披露日 —— 用它就得猜一个滞后，猜错就是前视。EDGAR 的每一条事实都带
`filed`（实际披露日），同一期间被后续文件**重述**时会多出一条 `filed` 更晚的记录，
所以"某一天已披露的最新版本"可以逐条还原，不需要任何假设。

真实响应里的几个坑（都从真数据里读出来的，测试夹具是裁剪后的真实响应）
----------------------------------------------------------------------
- `fy` / `fp` 是**这份文件**的财年财期，不是这条事实的期间：AAPL 有 3 个月窗口的收入被标成
  `fp="FY"`。期间只能看 `start` / `end`。
- 同一 (start, end) 会出现在多份文件里（比较期数字、重述）。按 `filed` 取版本。
- 收入科目名各家不同（`Revenues` / `RevenueFromContractWithCustomerExcludingAssessedTax` …）。
- 事实也来自 DEF 14A 等非财报文件（`fy` 为 None）—— 只收 10-K / 10-Q 及其修正件。
- submissions 里的 `acceptanceDateTime` 标着 `Z`，但**口径因申报人而异，不能当 UTC 用**。
  AAPL 2024-08-01（盘后发财报那天）的两份文件：
      8-K   filingDate 2024-08-01   acceptanceDateTime 2024-08-02T00:30:26.000Z
      10-Q  filingDate 2024-08-02   acceptanceDateTime 2024-08-02T02:03:34.000Z
  按 UTC 读，8-K 是美东 08-01 20:30 受理 —— 晚于 EDGAR 17:30 的当日截止，申报日却仍是 08-01，
  自相矛盾；减 8 小时（冬令时的样本是减 10 小时）才对得上。NVDA 同样。而 BKNG 2026-02-18 的
  8-K 是 `21:02:41Z`，按 UTC 读正好是美东 16:02 当日 —— 又是准确的。同一个字段两种口径，
  所以**事件日只用 `filingDate`**，时间戳原样留档（`accepted_raw`），不参与任何判断。

SEC 的访问规定：请求头必须声明身份（名字 + 联系邮箱），且每秒不超过 10 次请求。
本模块**不内置任何身份**：`EDGAR_USER_AGENT` 没配就拒绝发请求。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

#: 只收这些表格里的事实（年报 / 季报及其修正件）
FORMS = frozenset({"10-K", "10-Q", "10-K/A", "10-Q/A"})

#: 概念 → (分类标准, 单位, 科目名按优先级)。同一 (期间, 披露日) 有多个科目时取靠前的。
CONCEPTS: Dict[str, tuple] = {
    # 第三个科目：有的公司（BKNG 2016 年后）只报"归属于普通股股东的净利润"
    "net_income":     ("us-gaap", "USD", ("NetIncomeLoss", "ProfitLoss",
                                          "NetIncomeLossAvailableToCommonStockholdersBasic")),
    "revenue":        ("us-gaap", "USD", ("Revenues",
                                          "RevenueFromContractWithCustomerExcludingAssessedTax",
                                          "RevenueFromContractWithCustomerIncludingAssessedTax",
                                          "SalesRevenueNet")),
    "gross_profit":   ("us-gaap", "USD", ("GrossProfit",)),
    # 多数公司不报 GrossProfit，只报营业成本（us_broad_large 95 只里直接报毛利的不到三成）
    "cost_of_revenue": ("us-gaap", "USD", ("CostOfRevenue", "CostOfGoodsAndServicesSold",
                                           "CostOfGoodsSold")),
    "equity":         ("us-gaap", "USD", ("StockholdersEquity",
                                          "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")),
    "assets":         ("us-gaap", "USD", ("Assets",)),
    "shares_cover":   ("dei", "shares", ("EntityCommonStockSharesOutstanding",)),
    "shares_bs":      ("us-gaap", "shares", ("CommonStockSharesOutstanding",)),
    "shares_diluted": ("us-gaap", "shares", ("WeightedAverageNumberOfDilutedSharesOutstanding",)),
}

FACT_COLUMNS = ["ticker", "concept", "tag", "start", "end", "filed", "val", "form", "accn"]
EVENT_COLUMNS = ["ticker", "filing_date", "accepted_raw", "form", "items", "accn"]

#: SEC 限速 10 次/秒；留足余量
MIN_INTERVAL_S = 0.15

#: 解析口径的版本。改了 CONCEPTS / FORMS / 解析规则就 +1 —— 已落盘的事实是按旧口径解析的，
#: 同步时发现版本不符会重新解析（不等缓存过期）。
PARSER_VERSION = 2

#: 继任主体。公司重组换壳后 ticker 指向新 CIK，历史财报留在旧 CIK 名下，SEC 的接口不给这层关系。
#: 只登记**核实过**的：XOM —— 2026-07 起由 ExxonMobil Holdings Corp（CIK 2115436）申报，
#: 此前的全部财报在 Exxon Mobil Corp（CIK 34088）名下。
PREDECESSOR_CIKS: Dict[str, tuple] = {
    "XOM": (34088,),
}


class EdgarError(RuntimeError):
    """EDGAR 访问或解析失败。调用方据此**不提供**基本面字段，而不是当成"没有数据"。"""


def normalize_ticker(ticker: str) -> str:
    """SEC 的 ticker 用连字符（BRK-B），数据集里可能是点号（BRK.B）。"""
    return str(ticker).strip().upper().replace(".", "-")


class EdgarClient:
    """
    Parameters
    ----------
    user_agent  : SEC 要求的身份声明，形如 "Name email@example.com"。空或不含 @ → 拒绝构造。
    cache_dir   : 小响应（ticker 表、submissions）的磁盘缓存目录；None = 不缓存。
                  companyfacts 每家 5–8 MB，不缓存原文 —— 要存的是解析后的事实（见 store.py）。
    max_age_h   : 缓存有效期（小时）。财报按季披露，日内重复拉没有意义。
    session     : 测试注入点，需有 `.get(url, headers=..., timeout=...)`。
    """

    def __init__(self, user_agent: str, cache_dir: Optional[Path] = None, *,
                 max_age_h: float = 20.0, session=None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.time) -> None:
        ua = str(user_agent or "").strip()
        if "@" not in ua:
            raise EdgarError(
                "EDGAR_USER_AGENT 未配置或不含联系邮箱。SEC 要求每个请求声明身份"
                "（形如 \"Your Name you@example.com\"），本系统不会替你编一个。")
        self._headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
        self._cache = Path(cache_dir) if cache_dir is not None else None
        self._max_age_s = float(max_age_h) * 3600.0
        if session is None:
            import requests
            session = requests.Session()
        self._session = session
        self._sleep, self._clock = sleep, clock
        self._last_request = 0.0
        self._ciks: Optional[Dict[str, int]] = None

    # -- HTTP + 缓存 -----------------------------------------------------------

    def _get_json(self, url: str, cache_name: Optional[str]) -> dict:
        path = None if (self._cache is None or cache_name is None) else self._cache / cache_name
        if path is not None and path.exists():
            if self._clock() - path.stat().st_mtime <= self._max_age_s:
                return json.loads(path.read_text(encoding="utf-8"))
        wait = MIN_INTERVAL_S - (self._clock() - self._last_request)
        if wait > 0:
            self._sleep(wait)
        try:
            resp = self._session.get(url, headers=self._headers, timeout=30)
        except Exception as exc:
            raise EdgarError(f"EDGAR 请求失败 {url}: {exc}") from exc
        finally:
            self._last_request = self._clock()
        if resp.status_code != 200:
            raise EdgarError(f"EDGAR 返回 {resp.status_code}：{url}")
        payload = resp.json()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            tmp.replace(path)
        return payload

    # -- 接口 ------------------------------------------------------------------

    def cik_for(self, ticker: str) -> int:
        if self._ciks is None:
            raw = self._get_json(TICKERS_URL, "company_tickers.json")
            self._ciks = {normalize_ticker(r["ticker"]): int(r["cik_str"]) for r in raw.values()}
        tk = normalize_ticker(ticker)
        if tk not in self._ciks:
            raise EdgarError(f"SEC 的 ticker 表里没有 {tk}")
        return self._ciks[tk]

    def company_facts(self, ticker: str) -> dict:
        return self._get_json(FACTS_URL.format(cik=self.cik_for(ticker)), None)

    def predecessor_facts(self, ticker: str) -> List[dict]:
        """该 ticker 登记过的前任主体的 companyfacts（没有登记 → 空列表）。"""
        return [self._get_json(FACTS_URL.format(cik=c), None)
                for c in PREDECESSOR_CIKS.get(normalize_ticker(ticker), ())]

    def submissions(self, ticker: str) -> dict:
        cik = self.cik_for(ticker)
        return self._get_json(SUBMISSIONS_URL.format(cik=cik), f"submissions_{cik:010d}.json")


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def parse_company_facts(ticker: str, payload: dict) -> pd.DataFrame:
    """
    companyfacts JSON → 长表（FACT_COLUMNS）。只保留 10-K/10-Q 系文件里的事实。
    同一 (concept, start, end, filed) 有多个科目时保留优先级靠前的那个。
    """
    facts = (payload or {}).get("facts") or {}
    rows: List[dict] = []
    for concept, (taxonomy, unit, tags) in CONCEPTS.items():
        for prio, tag in enumerate(tags):
            for x in ((facts.get(taxonomy) or {}).get(tag) or {}).get("units", {}).get(unit, []):
                if x.get("form") not in FORMS or x.get("val") is None:
                    continue
                if not x.get("end") or not x.get("filed"):
                    continue
                rows.append({"ticker": ticker, "concept": concept, "tag": tag, "_prio": prio,
                             "start": x.get("start"), "end": x["end"], "filed": x["filed"],
                             "val": float(x["val"]), "form": x["form"], "accn": x.get("accn", "")})
    if not rows:
        return pd.DataFrame(columns=FACT_COLUMNS)
    df = pd.DataFrame(rows)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["end", "filed"])
    df = (df.sort_values(["concept", "end", "filed", "_prio", "accn"])
            .drop_duplicates(subset=["concept", "start", "end", "filed"], keep="first"))
    return df[FACT_COLUMNS].reset_index(drop=True)


def latest_report_date(payload: dict) -> Optional[pd.Timestamp]:
    """submissions 里最近一份年报 / 季报（含修正件）的申报日；一份都没有 → None。"""
    rec = ((payload or {}).get("filings") or {}).get("recent") or {}
    dates = [d for form, d in zip(rec.get("form") or [], rec.get("filingDate") or [])
             if form in FORMS and d]
    return pd.Timestamp(max(dates)) if dates else None


def parse_earnings_events(ticker: str, payload: dict) -> pd.DataFrame:
    """
    submissions JSON → 财报公告事件（8-K 且 items 含 2.02 "Results of Operations"）。
    事件日 = `filing_date`（EDGAR 申报日）；`accepted_raw` 只是原样留档，见模块 docstring。
    """
    rec = ((payload or {}).get("filings") or {}).get("recent") or {}
    forms = rec.get("form") or []
    rows = []
    for i, form in enumerate(forms):
        items = str((rec.get("items") or [""] * len(forms))[i] or "")
        if form != "8-K" or "2.02" not in [s.strip() for s in items.split(",")]:
            continue
        rows.append({"ticker": ticker,
                     "filing_date": (rec.get("filingDate") or [None] * len(forms))[i],
                     "accepted_raw": str((rec.get("acceptanceDateTime") or [""] * len(forms))[i] or ""),
                     "form": form, "items": items,
                     "accn": (rec.get("accessionNumber") or [""] * len(forms))[i]})
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.DataFrame(rows)
    df["filing_date"] = pd.to_datetime(df["filing_date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["filing_date"]).sort_values(["filing_date", "accn"])
    return df[EVENT_COLUMNS].reset_index(drop=True)
