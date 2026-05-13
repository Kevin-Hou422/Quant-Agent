"""
dataset_registry.py — Production dataset registry with 10 market datasets.

10 datasets across 4 regions, 10 industries:
  US Equities   : us_tech_large, us_financials, us_healthcare, us_energy
  China A-shares: china_tech, china_consumer, china_state_owned
  Hong Kong     : hk_china_tech
  Crypto        : crypto_major, crypto_alt

Each dataset maps to a provider (yfinance / akshare / ccxt_binance) and
contains a curated universe with more tickers than the minimum spec.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

from .multi_dataset import Dataset, _align_and_standardize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    name:      str
    provider:  str          # "yfinance" | "akshare" | "ccxt_binance"
    region:    str
    industry:  str
    universe:  List[str]
    start:     str = "2021-01-01"


# ---------------------------------------------------------------------------
# Universes  (expanded beyond the minimum spec)
# ---------------------------------------------------------------------------

_SPECS: Dict[str, DatasetSpec] = {

    # ── US Technology ────────────────────────────────────────────────────
    "us_tech_large": DatasetSpec(
        name="us_tech_large", provider="yfinance",
        region="US", industry="Technology",
        universe=[
            # Mega-cap platform / hardware
            "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN",
            # Semiconductors
            "AMD", "AVGO", "QCOM", "INTC", "TXN", "MRVL", "KLAC", "AMAT",
            # Enterprise software
            "CRM", "ORCL", "ADBE", "SAP", "NOW", "WDAY",
            # Cybersecurity
            "PANW", "CRWD", "ZS", "FTNT", "S", "OKTA",
            # Cloud / data infra
            "DDOG", "NET", "SNOW", "MDB", "TEAM", "PLTR", "ESTC",
            # Fintech / payments infra
            "PYPL", "SQ", "FI", "FSLR",
        ],
    ),

    # ── US Financials ────────────────────────────────────────────────────
    "us_financials": DatasetSpec(
        name="us_financials", provider="yfinance",
        region="US", industry="Financials",
        universe=[
            # Banks
            "JPM", "BAC", "WFC", "C", "USB", "TFC", "PNC", "KEY",
            # Investment banks / brokers
            "GS", "MS", "SCHW", "RJF",
            # Asset management
            "BLK", "BEN", "IVZ",
            # Insurance
            "MET", "PRU", "AIG", "AFL", "ALL", "TRV",
            # Payments / networks
            "V", "MA", "AXP", "DFS", "COF",
            # Exchanges / data
            "CME", "ICE", "SPGI", "MCO", "CBOE",
        ],
    ),

    # ── US Healthcare ────────────────────────────────────────────────────
    "us_healthcare": DatasetSpec(
        name="us_healthcare", provider="yfinance",
        region="US", industry="Healthcare",
        universe=[
            # Managed care / insurance
            "UNH", "CVS", "CI", "HUM", "CNC", "MOH",
            # Large pharma
            "JNJ", "LLY", "ABBV", "MRK", "PFE", "BMY", "AZN",
            # Biotech
            "AMGN", "GILD", "BIIB", "REGN", "VRTX", "MRNA",
            # Med devices / diagnostics
            "ISRG", "MDT", "ABT", "TMO", "DHR", "BSX", "SYK", "BAX",
            # Pharma services / distributors
            "MCK", "ABC", "CAH",
        ],
    ),

    # ── US Energy ────────────────────────────────────────────────────────
    "us_energy": DatasetSpec(
        name="us_energy", provider="yfinance",
        region="US", industry="Energy",
        universe=[
            # Integrated / upstream
            "XOM", "CVX", "COP", "EOG", "PXD", "OXY", "DVN", "FANG",
            # Midstream / pipelines
            "KMI", "WMB", "ET", "TRGP",
            # Refiners
            "MPC", "VLO", "PSX", "DK",
            # Services
            "SLB", "HAL", "BKR", "NOV",
            # LNG / utilities-adjacent
            "LNG", "CQP",
        ],
    ),

    # ── China A Technology ───────────────────────────────────────────────
    "china_tech": DatasetSpec(
        name="china_tech", provider="akshare",
        region="China", industry="Technology",
        universe=[
            # EV / batteries
            "300750.SZ",   # CATL
            "002594.SZ",   # BYD
            # Semiconductors
            "688981.SH",   # SMIC
            "603986.SH",   # GigaDevice
            "688111.SH",   # Beijing Kingsoft
            "688256.SH",   # Cambricon
            "688036.SH",   # Transsion (chip fabless)
            # Security / software
            "002415.SZ",   # Hikvision
            "300308.SZ",   # Zhongji Innolight
            "300274.SZ",   # Sungrow Power
            # AI / data
            "688047.SH",   # China Yida (AI)
            "300760.SZ",   # Mindray Medical (tech-heavy)
            # Display / components
            "000725.SZ",   # BOE Technology
            "002916.SZ",   # Shengyi Technology
            # 5G / telecom equipment
            "000063.SZ",   # ZTE Corp
            "601138.SH",   # Foxconn Industrial
            # Cloud / SaaS
            "603515.SH",   # Oupu Lighting → keep for diversity
            "688561.SH",   # Commercial Vehicle Group
        ],
    ),

    # ── China A Consumer ────────────────────────────────────────────────
    "china_consumer": DatasetSpec(
        name="china_consumer", provider="akshare",
        region="China", industry="Consumer",
        universe=[
            # Baijiu (spirits)
            "600519.SH",   # Kweichow Moutai
            "000858.SZ",   # Wuliangye
            "600809.SH",   # Shanxi Fenjiu
            "603288.SH",   # Haitian Flavoring
            "000568.SZ",   # Luzhou Laojiao
            # Dairy / food
            "600887.SH",   # Yili
            "002714.SZ",   # Muyuan Foodstuff
            "000876.SZ",   # New Hope Liuhe
            "600298.SH",   # Angel Yeast
            # Household / appliances
            "600690.SH",   # Haier Smart Home
            "000333.SZ",   # Midea Group
            "002415.SZ",   # Hikvision (also consumer)
            # Retail / e-commerce adjacent
            "600900.SH",   # China Yangtze Power
            "002304.SZ",   # Yanghe Brewery
            "603369.SH",   # Jiuguijiu
        ],
    ),

    # ── China State-Owned ────────────────────────────────────────────────
    "china_state_owned": DatasetSpec(
        name="china_state_owned", provider="akshare",
        region="China", industry="StateOwned",
        universe=[
            # Big-4 banks + BOCOM
            "601398.SH",   # ICBC
            "601939.SH",   # CCB
            "601988.SH",   # Bank of China
            "601288.SH",   # ABC
            "601328.SH",   # Bank of Communications
            # Joint-stock banks
            "600036.SH",   # China Merchants Bank
            "601166.SH",   # Industrial Bank
            "600000.SH",   # SPD Bank
            # Energy SOE
            "601857.SH",   # PetroChina
            "600028.SH",   # Sinopec
            "600900.SH",   # China Yangtze Power
            "601985.SH",   # China Nuclear Power
            # Telecom SOE
            "600050.SH",   # China Unicom
            "601728.SH",   # China Telecom
            # Infrastructure SOE
            "601390.SH",   # China Railway Group
            "601800.SH",   # China Communications Construction
        ],
    ),

    # ── Hong Kong China Tech ─────────────────────────────────────────────
    "hk_china_tech": DatasetSpec(
        name="hk_china_tech", provider="yfinance",
        region="HongKong", industry="ChinaTech",
        universe=[
            # Platform / internet
            "0700.HK",    # Tencent
            "9988.HK",    # Alibaba
            "9618.HK",    # JD.com
            "9888.HK",    # Baidu
            "3690.HK",    # Meituan
            "1810.HK",    # Xiaomi
            # Biotech / healthcare
            "1177.HK",    # Sino Biopharm
            "6969.HK",    # Smoore International
            # Fintech
            "0981.HK",    # SMIC (HK)
            "6185.HK",    # CALB Group
            # Gaming
            "0272.HK",    # Shriram Transport (added for diversity)
            "2382.HK",    # Sunny Optical
            # EV / auto
            "0175.HK",    # Geely Auto
            "2015.HK",    # Li Auto
            "9868.HK",    # Xpeng
        ],
    ),

    # ── Crypto Major ─────────────────────────────────────────────────────
    "crypto_major": DatasetSpec(
        name="crypto_major", provider="ccxt_binance",
        region="Global", industry="CryptoMajor",
        universe=[
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
            "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOT/USDT",
            "MATIC/USDT", "LINK/USDT", "ATOM/USDT", "LTC/USDT",
        ],
    ),

    # ── Crypto Alt / L2 ──────────────────────────────────────────────────
    "crypto_alt": DatasetSpec(
        name="crypto_alt", provider="ccxt_binance",
        region="Global", industry="CryptoAlt",
        universe=[
            # L2 / rollups
            "ARB/USDT", "OP/USDT", "MATIC/USDT",
            # New L1
            "APT/USDT", "SUI/USDT", "SEI/USDT", "TIA/USDT",
            # AI / DePIN
            "RNDR/USDT", "FET/USDT", "OCEAN/USDT", "WLD/USDT",
            # DeFi
            "UNI/USDT", "AAVE/USDT", "CRV/USDT",
            # Gaming / NFT infra
            "IMX/USDT", "BLUR/USDT",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_registry_dataset(
    name:      str,
    start:     Optional[str] = None,
    end:       Optional[str] = None,
    use_cache: bool = True,
) -> Dataset:
    """
    Load a named dataset from the production registry.

    Parameters
    ----------
    name      : Dataset name, one of REGISTRY_NAMES
    start/end : Override date range (ISO "YYYY-MM-DD"). Defaults to spec start.
    use_cache : Return cached Dataset if already loaded for same (name,start,end)

    Returns
    -------
    Dataset with exactly 7 standard fields (open,high,low,close,volume,vwap,returns)
    """
    if name not in _SPECS:
        raise KeyError(
            f"Unknown dataset '{name}'. "
            f"Available: {sorted(_SPECS.keys())}"
        )

    spec = _SPECS[name]
    start_dt = start or spec.start
    end_dt   = end   or _default_end()

    cache_key = f"{name}|{start_dt}|{end_dt}"
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    logger.info(
        "Loading registry dataset '%s' (%s, %s) [%s → %s] — %d tickers",
        name, spec.region, spec.industry,
        start_dt, end_dt, len(spec.universe),
    )

    raw = _fetch_raw(spec, start_dt, end_dt)
    data = _align_and_standardize(raw, spec.universe)

    ds = Dataset(
        name      = name,
        frequency = "daily",
        universe  = spec.universe,
        data      = data,
    )

    if use_cache:
        _CACHE[cache_key] = ds

    return ds


def clear_registry_cache() -> None:
    _CACHE.clear()


def registry_names() -> List[str]:
    return sorted(_SPECS.keys())


def registry_spec(name: str) -> DatasetSpec:
    if name not in _SPECS:
        raise KeyError(f"Unknown dataset '{name}'")
    return _SPECS[name]


# ---------------------------------------------------------------------------
# Internal fetch dispatch
# ---------------------------------------------------------------------------

_CACHE: Dict[str, Dataset] = {}


def _default_end() -> str:
    return pd.Timestamp.today().strftime("%Y-%m-%d")


def _fetch_raw(spec: DatasetSpec, start: str, end: str) -> Dict:
    if spec.provider == "yfinance":
        return _fetch_yfinance(spec.universe, start, end)
    elif spec.provider == "akshare":
        return _fetch_akshare(spec.universe, start, end)
    elif spec.provider == "ccxt_binance":
        return _fetch_ccxt(spec.universe, start, end)
    else:
        raise ValueError(f"Unknown provider: '{spec.provider}'")


def _fetch_yfinance(tickers: List[str], start: str, end: str) -> Dict:
    from ..yahoo_provider import YahooFinanceProvider
    return YahooFinanceProvider(auto_adjust=True, progress=False).fetch(
        tickers, start=start, end=end
    )


def _fetch_akshare(symbols: List[str], start: str, end: str) -> Dict:
    from .akshare_provider import AkshareProvider
    return AkshareProvider(adjust="qfq").fetch(symbols, start=start, end=end)


def _fetch_ccxt(symbols: List[str], start: str, end: str) -> Dict:
    from .ccxt_provider import CcxtBinanceProvider
    return CcxtBinanceProvider().fetch(symbols, start=start, end=end)
