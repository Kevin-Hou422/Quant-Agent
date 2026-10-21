"""
input_identity.py — 一次决策 / 结算到底用了哪份数据、哪份代码（复审 2026-10-19 N02-c / N04-b）

此前证据只记了策略 id + 规格哈希：两次运行用的是不是同一份数据、同一个股票池、同一个价格源、
同一版代码，事后都说不清。复审原话："输入需要带实际 provider、价格口径、vintage/摘要、有效股票池和代码版本；
不能只相信 settings 声称的来源。"

    data_digest      价格面板（OHLCV）的内容哈希 —— 任何一个数变了都不同
    last_day         面板末日
    universe_digest  末日有收盘价的标的集合的哈希（有效股票池），universe_n 是个数
    price_source     实际来源（PIT 元数据），不是 settings 的声称
    vintage          这份数据的 PIT as_of
    code_version     git HEAD（工作区有改动时带 "+dirty"）
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

PRICE_FIELDS = ("open", "high", "low", "close", "volume")

#: 信号 / 成交 / IC 的**时钟口径版本**（复审 2026-10-19 §5.1："尚未统一时钟前，历史结果应分版本标识"）。
#: 每一类存下来的结果都带它；进可执行规格 —— 时钟一变就是另一个策略版本，旧证据不被继承。
#: D14 统一时钟时改这里（并在 CLOCK_VERSIONS 里写明新口径）。
#: 2026-10-20 之前存的行没有这一列（NULL）：那时只有过 c1 这一种口径，读作 c1。
CLOCK_VERSION = "c1"
CLOCK_VERSIONS = {
    "c1": ("T 日收盘后运行：信号 delay=1（用 T−1 的原始信号）；内部模拟按 T 日收盘成交；"
           "券商订单 T 日收盘后提交、计划 T+1 成交；因子 / 策略 IC = signal[T−1] 对 T 日 close-to-close 收益"),
}


def clock_version() -> str:
    """当前时钟口径（调用时读模块常量，便于测试换口径）。"""
    return CLOCK_VERSION


@dataclass(frozen=True)
class InputIdentity:
    data_digest:     str
    last_day:        str
    universe_digest: str
    universe_n:      int
    price_source:    str
    vintage:         str
    code_version:    str
    clock_version:   str = CLOCK_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def code_version() -> str:
    """git HEAD 的短哈希；工作区有未提交改动 → 后缀 +dirty；不是 git 环境 → "unknown"。"""
    root = Path(__file__).resolve().parents[3]
    try:
        head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=root, capture_output=True,
                              text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as exc:
        logger.warning("[identity] 取不到代码版本: %s", exc)
        return "unknown"
    return (head + ("+dirty" if dirty else "")) if head else "unknown"


def _sha(parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


def identify(panel: Dict[str, pd.DataFrame], price_source: Optional[str] = None,
             vintage: Optional[str] = None) -> InputIdentity:
    close = panel.get("close")
    if close is None or close.empty:
        raise ValueError("面板没有收盘价，无法标识输入")
    parts = []
    for f in PRICE_FIELDS:
        df = panel.get(f)
        if isinstance(df, pd.DataFrame):
            parts.append(f.encode())
            parts.append(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
            parts.append("\x1f".join(map(str, df.columns)).encode("utf-8"))
    last = close.iloc[-1]
    universe = sorted(str(t) for t in last.index[last.notna()])
    return InputIdentity(
        data_digest=_sha(parts),
        last_day=str(pd.Timestamp(close.index[-1]).date()),
        universe_digest=_sha(["\x1f".join(universe).encode("utf-8")])[:16],
        universe_n=len(universe),
        price_source=str(price_source or ""),
        vintage=str(vintage or ""),
        code_version=code_version(),
        clock_version=clock_version(),
    )
