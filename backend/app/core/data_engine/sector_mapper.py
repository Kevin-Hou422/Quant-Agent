"""
sector_mapper.py — 静态 GICS 行业分类映射

为 dataset_registry.py 中的所有 universe 提供 GICS L1 整数行业代码，
生成 (T × N) 的 sector DataFrame，供 dsl_executor 的 group_rank / group_zscore /
ind_neutralize 等截面分组算子使用。

GICS L1 代码映射：
  0  Information Technology
  1  Financials
  2  Healthcare
  3  Energy
  4  Consumer Discretionary
  5  Consumer Staples
  6  Industrials
  7  Communication Services
  8  Materials
  9  Real Estate
  10 Utilities
  11 Crypto / Digital Assets（非标准，加密市场专用）
 -1  未知 / 未分类
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Module-level cache for dynamically looked-up sector names (avoids repeated API calls)
_DYNAMIC_CACHE: Dict[str, Optional[str]] = {}

# ---------------------------------------------------------------------------
# GICS L1 名称 → 整数代码
# ---------------------------------------------------------------------------

SECTOR_CODES: Dict[str, int] = {
    "Information Technology": 0,
    "Financials":             1,
    "Healthcare":             2,
    "Energy":                 3,
    "Consumer Discretionary": 4,
    "Consumer Staples":       5,
    "Industrials":            6,
    "Communication Services": 7,
    "Materials":              8,
    "Real Estate":            9,
    "Utilities":              10,
    "Crypto":                 11,
}

SECTOR_NAMES: Dict[int, str] = {v: k for k, v in SECTOR_CODES.items()}

# ---------------------------------------------------------------------------
# 静态 ticker → GICS L1 名称映射
# 覆盖 dataset_registry.py 中所有 universe 的全量 ticker
# ---------------------------------------------------------------------------

_STATIC_SECTOR_MAP: Dict[str, str] = {

    # ── us_tech_large ────────────────────────────────────────────────────
    # Mega-cap platform / hardware
    "AAPL":  "Information Technology",
    "MSFT":  "Information Technology",
    "NVDA":  "Information Technology",
    "GOOGL": "Communication Services",
    "META":  "Communication Services",
    "AMZN":  "Consumer Discretionary",
    # Semiconductors
    "AMD":   "Information Technology",
    "AVGO":  "Information Technology",
    "QCOM":  "Information Technology",
    "INTC":  "Information Technology",
    "TXN":   "Information Technology",
    "MRVL":  "Information Technology",
    "KLAC":  "Information Technology",
    "AMAT":  "Information Technology",
    # Enterprise software
    "CRM":   "Information Technology",
    "ORCL":  "Information Technology",
    "ADBE":  "Information Technology",
    "SAP":   "Information Technology",
    "NOW":   "Information Technology",
    "WDAY":  "Information Technology",
    # Cybersecurity
    "PANW":  "Information Technology",
    "CRWD":  "Information Technology",
    "ZS":    "Information Technology",
    "FTNT":  "Information Technology",
    "S":     "Information Technology",
    "OKTA":  "Information Technology",
    # Cloud / data infra
    "DDOG":  "Information Technology",
    "NET":   "Information Technology",
    "SNOW":  "Information Technology",
    "MDB":   "Information Technology",
    "TEAM":  "Information Technology",
    "PLTR":  "Information Technology",
    "ESTC":  "Information Technology",
    # Fintech / payments infra
    "PYPL":  "Financials",
    "SQ":    "Financials",
    "FI":    "Financials",
    "FSLR":  "Information Technology",   # Solar hardware

    # ── us_financials ────────────────────────────────────────────────────
    # Banks
    "JPM":  "Financials",
    "BAC":  "Financials",
    "WFC":  "Financials",
    "C":    "Financials",
    "USB":  "Financials",
    "TFC":  "Financials",
    "PNC":  "Financials",
    "KEY":  "Financials",
    # Investment banks / brokers
    "GS":   "Financials",
    "MS":   "Financials",
    "SCHW": "Financials",
    "RJF":  "Financials",
    # Asset management
    "BLK":  "Financials",
    "BEN":  "Financials",
    "IVZ":  "Financials",
    # Insurance
    "MET":  "Financials",
    "PRU":  "Financials",
    "AIG":  "Financials",
    "AFL":  "Financials",
    "ALL":  "Financials",
    "TRV":  "Financials",
    # Payments / networks
    "V":    "Financials",
    "MA":   "Financials",
    "AXP":  "Financials",
    "DFS":  "Financials",
    "COF":  "Financials",
    # Exchanges / data
    "CME":  "Financials",
    "ICE":  "Financials",
    "SPGI": "Financials",
    "MCO":  "Financials",
    "CBOE": "Financials",

    # ── us_healthcare ────────────────────────────────────────────────────
    # Managed care
    "UNH":  "Healthcare",
    "CVS":  "Healthcare",
    "CI":   "Healthcare",
    "HUM":  "Healthcare",
    "CNC":  "Healthcare",
    "MOH":  "Healthcare",
    # Large pharma
    "JNJ":  "Healthcare",
    "LLY":  "Healthcare",
    "ABBV": "Healthcare",
    "MRK":  "Healthcare",
    "PFE":  "Healthcare",
    "BMY":  "Healthcare",
    "AZN":  "Healthcare",
    # Biotech
    "AMGN": "Healthcare",
    "GILD": "Healthcare",
    "BIIB": "Healthcare",
    "REGN": "Healthcare",
    "VRTX": "Healthcare",
    "MRNA": "Healthcare",
    # Med devices / diagnostics
    "ISRG": "Healthcare",
    "MDT":  "Healthcare",
    "ABT":  "Healthcare",
    "TMO":  "Healthcare",
    "DHR":  "Healthcare",
    "BSX":  "Healthcare",
    "SYK":  "Healthcare",
    "BAX":  "Healthcare",
    # Pharma services / distributors
    "MCK":  "Healthcare",
    "ABC":  "Healthcare",
    "CAH":  "Healthcare",

    # ── us_energy ────────────────────────────────────────────────────────
    # Integrated / upstream
    "XOM":  "Energy",
    "CVX":  "Energy",
    "COP":  "Energy",
    "EOG":  "Energy",
    "PXD":  "Energy",
    "OXY":  "Energy",
    "DVN":  "Energy",
    "FANG": "Energy",
    # Midstream / pipelines
    "KMI":  "Energy",
    "WMB":  "Energy",
    "ET":   "Energy",
    "TRGP": "Energy",
    # Refiners
    "MPC":  "Energy",
    "VLO":  "Energy",
    "PSX":  "Energy",
    "DK":   "Energy",
    # Services
    "SLB":  "Energy",
    "HAL":  "Energy",
    "BKR":  "Energy",
    "NOV":  "Energy",
    # LNG
    "LNG":  "Energy",
    "CQP":  "Energy",

    # ── china_tech (akshare, A-share codes) ──────────────────────────────
    "300750.SZ": "Information Technology",   # CATL（EV/batteries）
    "002594.SZ": "Consumer Discretionary",   # BYD
    "688981.SH": "Information Technology",   # SMIC
    "603986.SH": "Information Technology",   # GigaDevice
    "688111.SH": "Information Technology",   # Beijing Kingsoft
    "688256.SH": "Information Technology",   # Cambricon
    "688036.SH": "Information Technology",   # Transsion
    "002415.SZ": "Information Technology",   # Hikvision
    "300308.SZ": "Information Technology",   # Zhongji Innolight
    "300274.SZ": "Information Technology",   # Sungrow Power
    "688047.SH": "Information Technology",   # China Yida (AI)
    "300760.SZ": "Healthcare",               # Mindray Medical
    "000725.SZ": "Information Technology",   # BOE Technology
    "002916.SZ": "Information Technology",   # Shengyi Technology
    "000063.SZ": "Communication Services",   # ZTE Corp
    "601138.SH": "Information Technology",   # Foxconn Industrial
    "603515.SH": "Information Technology",   # Oupu Lighting
    "688561.SH": "Industrials",              # Commercial Vehicle Group

    # ── china_consumer ───────────────────────────────────────────────────
    "600519.SH": "Consumer Staples",         # Kweichow Moutai
    "000858.SZ": "Consumer Staples",         # Wuliangye
    "600809.SH": "Consumer Staples",         # Shanxi Fenjiu
    "603288.SH": "Consumer Staples",         # Haitian Flavoring
    "000568.SZ": "Consumer Staples",         # Luzhou Laojiao
    "600887.SH": "Consumer Staples",         # Yili
    "002714.SZ": "Consumer Staples",         # Muyuan Foodstuff
    "000876.SZ": "Consumer Staples",         # New Hope Liuhe
    "600298.SH": "Consumer Staples",         # Angel Yeast
    "600690.SH": "Consumer Discretionary",   # Haier Smart Home
    "000333.SZ": "Consumer Discretionary",   # Midea Group
    "600900.SH": "Utilities",                # China Yangtze Power
    "002304.SZ": "Consumer Staples",         # Yanghe Brewery
    "603369.SH": "Consumer Staples",         # Jiuguijiu

    # ── china_state_owned ────────────────────────────────────────────────
    "601398.SH": "Financials",               # ICBC
    "601939.SH": "Financials",               # CCB
    "601988.SH": "Financials",               # Bank of China
    "601288.SH": "Financials",               # ABC
    "601328.SH": "Financials",               # Bank of Communications
    "600036.SH": "Financials",               # China Merchants Bank
    "601166.SH": "Financials",               # Industrial Bank
    "600000.SH": "Financials",               # SPD Bank
    "601857.SH": "Energy",                   # PetroChina
    "600028.SH": "Energy",                   # Sinopec
    "601985.SH": "Utilities",                # China Nuclear Power
    "600050.SH": "Communication Services",   # China Unicom
    "601728.SH": "Communication Services",   # China Telecom
    "601390.SH": "Industrials",              # China Railway Group
    "601800.SH": "Industrials",              # China Communications Construction

    # ── hk_china_tech ────────────────────────────────────────────────────
    "0700.HK":  "Communication Services",    # Tencent
    "9988.HK":  "Consumer Discretionary",    # Alibaba
    "9618.HK":  "Consumer Discretionary",    # JD.com
    "9888.HK":  "Communication Services",    # Baidu
    "3690.HK":  "Consumer Discretionary",    # Meituan
    "1810.HK":  "Information Technology",    # Xiaomi
    "1177.HK":  "Healthcare",               # Sino Biopharm
    "6969.HK":  "Consumer Staples",         # Smoore International
    "0981.HK":  "Information Technology",    # SMIC (HK)
    "6185.HK":  "Information Technology",    # CALB Group
    "0272.HK":  "Financials",               # Shriram Transport
    "2382.HK":  "Information Technology",    # Sunny Optical
    "0175.HK":  "Consumer Discretionary",    # Geely Auto
    "2015.HK":  "Consumer Discretionary",    # Li Auto
    "9868.HK":  "Consumer Discretionary",    # Xpeng

    # ── crypto_major / crypto_alt ────────────────────────────────────────
    # All crypto assets → "Crypto" (non-standard GICS)
    "BTC/USDT":   "Crypto",
    "ETH/USDT":   "Crypto",
    "BNB/USDT":   "Crypto",
    "SOL/USDT":   "Crypto",
    "XRP/USDT":   "Crypto",
    "ADA/USDT":   "Crypto",
    "AVAX/USDT":  "Crypto",
    "DOT/USDT":   "Crypto",
    "MATIC/USDT": "Crypto",
    "LINK/USDT":  "Crypto",
    "ATOM/USDT":  "Crypto",
    "LTC/USDT":   "Crypto",
    "ARB/USDT":   "Crypto",
    "OP/USDT":    "Crypto",
    "APT/USDT":   "Crypto",
    "SUI/USDT":   "Crypto",
    "SEI/USDT":   "Crypto",
    "TIA/USDT":   "Crypto",
    "RNDR/USDT":  "Crypto",
    "FET/USDT":   "Crypto",
    "OCEAN/USDT": "Crypto",
    "WLD/USDT":   "Crypto",
    "UNI/USDT":   "Crypto",
    "AAVE/USDT":  "Crypto",
    "CRV/USDT":   "Crypto",
    "IMX/USDT":   "Crypto",
    "BLUR/USDT":  "Crypto",
    # yfinance 格式的加密货币 ticker
    "BTC-USD":    "Crypto",
    "ETH-USD":    "Crypto",
    "BNB-USD":    "Crypto",
    "XRP-USD":    "Crypto",
    "SOL-USD":    "Crypto",
    "ADA-USD":    "Crypto",
    "DOGE-USD":   "Crypto",
    "AVAX-USD":   "Crypto",
    "DOT-USD":    "Crypto",
    "MATIC-USD":  "Crypto",
    "LINK-USD":   "Crypto",
    "UNI-USD":    "Crypto",
    "ATOM-USD":   "Crypto",
    "LTC-USD":    "Crypto",
    "BCH-USD":    "Crypto",
}


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def get_sector_code(ticker: str) -> int:
    """
    返回 ticker 的整数 GICS 行业代码（静态映射）。
    未知 ticker 返回 -1；不触发网络请求。
    """
    sector_name = _STATIC_SECTOR_MAP.get(ticker)
    if sector_name is None:
        return -1
    return SECTOR_CODES.get(sector_name, -1)


def get_sector_code_dynamic(ticker: str) -> int:
    """
    返回 ticker 的整数 GICS 行业代码。

    查找顺序：
      1. 静态映射（_STATIC_SECTOR_MAP）— 无网络，最快
      2. 模块级缓存（_DYNAMIC_CACHE）— 本次进程内已查询过
      3. yfinance Ticker.info["sector"] — 单次网络请求，结果写入缓存

    对于无法识别的 ticker（yfinance 也无 sector 信息）返回 -1 并缓存，
    避免后续重复发起无效请求。
    """
    # 1. Static map (fast path)
    static = _STATIC_SECTOR_MAP.get(ticker)
    if static is not None:
        return SECTOR_CODES.get(static, -1)

    # 2. Dynamic cache
    if ticker in _DYNAMIC_CACHE:
        cached = _DYNAMIC_CACHE[ticker]
        return SECTOR_CODES.get(cached, -1) if cached else -1

    # 3. yfinance lookup (network call, timeout-protected)
    sector_name: Optional[str] = None
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        raw = info.get("sector", None)
        if raw:
            # yfinance returns "Technology", "Financial Services", etc.
            # Map common yfinance sector names to our GICS L1 names
            sector_name = _YF_SECTOR_MAP.get(raw, raw)
            if sector_name not in SECTOR_CODES:
                sector_name = None
    except Exception as exc:
        logger.debug("yfinance sector lookup failed for '%s': %s", ticker, exc)

    _DYNAMIC_CACHE[ticker] = sector_name
    if sector_name:
        logger.debug("Dynamic sector: '%s' → '%s'", ticker, sector_name)
    else:
        logger.debug("No sector found for '%s', using -1", ticker)

    return SECTOR_CODES.get(sector_name, -1) if sector_name else -1


# Mapping from yfinance sector labels to GICS L1 names used in SECTOR_CODES
_YF_SECTOR_MAP: Dict[str, str] = {
    "Technology":             "Information Technology",
    "Financial Services":     "Financials",
    "Financial":              "Financials",
    "Healthcare":             "Healthcare",
    "Energy":                 "Energy",
    "Consumer Cyclical":      "Consumer Discretionary",
    "Consumer Defensive":     "Consumer Staples",
    "Industrials":            "Industrials",
    "Communication Services": "Communication Services",
    "Basic Materials":        "Materials",
    "Real Estate":            "Real Estate",
    "Utilities":              "Utilities",
}


def get_sector_name(ticker: str) -> str:
    """返回 ticker 的 GICS L1 行业名称，未知时返回 'Unknown'。"""
    return _STATIC_SECTOR_MAP.get(ticker, "Unknown")


def clear_dynamic_cache() -> None:
    """清除 yfinance 动态查询缓存（测试 / 强制刷新时使用）。"""
    _DYNAMIC_CACHE.clear()


def build_sector_matrix(
    tickers: List[str],
    dates: pd.DatetimeIndex,
    dynamic: bool = False,
) -> pd.DataFrame:
    """
    构建 (T × N) 行业代码矩阵。

    行为日期，列为 ticker，值为整数 GICS L1 代码。
    未知 ticker 的代码为 -1（静态映射未覆盖时）。

    Parameters
    ----------
    tickers : 资产代码列表
    dates   : 时间索引（DatetimeIndex）
    dynamic : False（默认）= 仅使用静态映射；
              True = 对静态未覆盖的 ticker 额外尝试 yfinance 动态查询

    Returns
    -------
    pd.DataFrame (T × N)，dtype=float（与其他字段保持一致）
    """
    getter = get_sector_code_dynamic if dynamic else get_sector_code
    codes  = np.array([getter(t) for t in tickers], dtype=float)
    # 行业代码在时间维度上为静态（不随日期变化），广播为 (T, N)
    data = np.tile(codes, (len(dates), 1))
    return pd.DataFrame(data, index=dates, columns=tickers)


def coverage_report(tickers: List[str]) -> dict:
    """
    返回 tickers 列表中的行业覆盖情况报告。

    Returns
    -------
    dict with keys: total, mapped, unmapped, sector_distribution
    """
    mapped   = [t for t in tickers if t in _STATIC_SECTOR_MAP]
    unmapped = [t for t in tickers if t not in _STATIC_SECTOR_MAP]

    dist: dict = {}
    for t in mapped:
        name = _STATIC_SECTOR_MAP[t]
        dist[name] = dist.get(name, 0) + 1

    return {
        "total":               len(tickers),
        "mapped":              len(mapped),
        "unmapped":            len(unmapped),
        "unmapped_tickers":    unmapped,
        "sector_distribution": dist,
    }
