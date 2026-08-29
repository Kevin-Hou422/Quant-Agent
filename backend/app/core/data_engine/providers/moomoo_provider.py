"""
moomoo_provider.py — moomoo OpenAPI DataProvider（Phase TR.2：单一权威数据源）

为什么
------
把**研究(discovery/validation) 与 执行(paper/实盘) 统一到同一个数据源 = moomoo**，消除
train/serve skew（DEV_LESSONS §J）。moomoo 美股实时免费（LV1/LV3 推广）、日 K 复权、经本地
OpenD 网关读取；纸交易用真实市场数据、无需入金。

结构对齐
--------
`fetch()` 返回与 `YahooFinanceProvider` **逐字段一致**的 RawDataset：
    {open, high, low, close, volume, vwap, returns} ，每个 = DataFrame(DatetimeIndex × ticker)。
下游（alpha/backtest/paper 引擎）无需改动即可从 yahoo 切到 moomoo。

依赖运行时
----------
- 需要本地 **OpenD 网关**已启动并登录（默认 127.0.0.1:11111），且账户有对应市场行情权限。
- 需要 `pip install moomoo-api`。二者缺一 → 抛清晰错误（不静默）。

纪律
----
- ticker 用无市场前缀的裸符号（'AAPL'）对外；内部映射为 moomoo 的 'US.AAPL'。
- 历史 K 线有**额度**（账户「历史K线 x/100」）与请求频率限制 → 逐标的间加节流；分页取全。
- 仅数据来源，不含任何金融逻辑；LLM 不参与。
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..base import DataProvider, RawDataset

logger = logging.getLogger(__name__)

_SUPPORTED_FIELDS = ["open", "high", "low", "close", "volume", "vwap", "returns"]
_DEFAULT_MARKET = "US"


def _to_moomoo_code(ticker: str, market: str = _DEFAULT_MARKET) -> str:
    """'AAPL' → 'US.AAPL'；已含市场前缀（含 '.'）则原样。"""
    t = ticker.strip().upper()
    return t if "." in t else f"{market}.{t}"


def _from_moomoo_code(code: str) -> str:
    """'US.AAPL' → 'AAPL'。"""
    return code.split(".", 1)[1] if "." in code else code


def assemble_panel(kline_by_ticker: Dict[str, pd.DataFrame],
                   tickers: List[str], fields: List[str]) -> RawDataset:
    """
    纯函数：把 {ticker: 单标的日K DataFrame(含 time_key/open/high/low/close/volume)} 拼成
    与 YahooFinanceProvider 一致的 RawDataset（DatetimeIndex × ticker）。可离线测试。
    """
    tickers = [t.upper() for t in tickers]
    per_field: Dict[str, Dict[str, pd.Series]] = {f: {} for f in ("open", "high", "low", "close", "volume")}

    for tk in tickers:
        df = kline_by_ticker.get(tk)
        if df is None or len(df) == 0:
            continue
        s = df.copy()
        idx = pd.to_datetime(s["time_key"]).dt.normalize()
        s = s.set_index(idx)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        for f in ("open", "high", "low", "close", "volume"):
            if f in s.columns:
                per_field[f][tk] = pd.to_numeric(s[f], errors="coerce")

    # 统一时间轴（所有标的日期并集）
    all_dates = sorted(set().union(*[set(ser.index) for d in per_field.values()
                                     for ser in d.values()])) if any(per_field.values()) else []
    date_index = pd.DatetimeIndex(all_dates)

    def _wide(field: str) -> pd.DataFrame:
        cols = {tk: per_field[field].get(tk) for tk in tickers}
        df = pd.DataFrame(cols, index=date_index).reindex(columns=tickers)
        return df.astype(float)

    frames: Dict[str, pd.DataFrame] = {f: _wide(f) for f in ("open", "high", "low", "close", "volume")}

    if "vwap" in fields:
        frames["vwap"] = (frames["high"] + frames["low"] + frames["close"]) / 3.0
    if "returns" in fields:
        c = frames["close"]
        frames["returns"] = np.log(c / c.shift(1))

    return {f: frames[f] for f in fields if f in frames}


class MoomooProvider(DataProvider):
    """
    Parameters
    ----------
    host / port : OpenD 网关地址（默认 127.0.0.1:11111）。
    market      : 默认市场前缀（'US'）。
    autype      : 复权方式（'qfq' 前复权 | 'hfq' 后复权 | None 不复权）；默认前复权，与 yahoo auto_adjust 近似。
    throttle_s  : 逐标的请求间隔秒（尊重历史K线频率限制）。
    max_count   : 单次分页最大条数。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        market: str = _DEFAULT_MARKET,
        autype: Optional[str] = "qfq",
        throttle_s: float = 0.4,
        max_count: int = 1000,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.market = market
        self.autype = autype
        self.throttle_s = float(throttle_s)
        self.max_count = int(max_count)

    def available_fields(self) -> List[str]:
        return list(_SUPPORTED_FIELDS)

    # ------------------------------------------------------------------
    # SDK 句柄（懒加载 + 清晰错误）
    # ------------------------------------------------------------------

    def _sdk(self):
        try:
            import moomoo  # noqa: F401
        except Exception as exc:  # pragma: no cover - 环境相关
            raise RuntimeError(
                "未安装 moomoo-api（pip install moomoo-api），无法使用 MoomooProvider。"
            ) from exc
        return moomoo

    def _open_quote_ctx(self):
        moomoo = self._sdk()
        try:
            return moomoo.OpenQuoteContext(host=self.host, port=self.port)
        except Exception as exc:  # pragma: no cover - 需网关
            raise RuntimeError(
                f"无法连接 OpenD 网关 {self.host}:{self.port}——请确认 OpenD 已启动并登录。原始错误: {exc}"
            ) from exc

    def _autype_enum(self):
        moomoo = self._sdk()
        return {
            "qfq": moomoo.AuType.QFQ,
            "hfq": moomoo.AuType.HFQ,
            None:  moomoo.AuType.NONE,
        }.get(self.autype, moomoo.AuType.QFQ)

    # ------------------------------------------------------------------
    # DataProvider 接口
    # ------------------------------------------------------------------

    def fetch(
        self,
        tickers: List[str],
        start: str,
        end: str,
        fields: Optional[List[str]] = None,
    ) -> RawDataset:
        moomoo = self._sdk()
        if fields is None:
            fields = list(_SUPPORTED_FIELDS)
        else:
            self.validate_fields(fields)

        tickers = [t.upper() for t in tickers]
        ctx = self._open_quote_ctx()
        try:
            kline_by_ticker = self._fetch_all_klines(ctx, moomoo, tickers, start, end)
        finally:
            try:
                ctx.close()
            except Exception:  # pragma: no cover
                pass

        dataset = assemble_panel(kline_by_ticker, tickers, fields)
        if not dataset or next(iter(dataset.values())).dropna(how="all").empty:
            raise ValueError(
                f"moomoo 返回空数据（tickers={tickers[:5]}…, {start}→{end}）；"
                "请确认标的正确、日期范围内有交易日、且账户有该市场历史K线权限/额度。"
            )
        logger.info("[moomoo] 拉取 %d 标的 [%s→%s]，字段=%s，shape=%s",
                    len(tickers), start, end, list(dataset.keys()),
                    next(iter(dataset.values())).shape)
        return dataset

    def fetch_latest(self, tickers: List[str], n_recent: int = 5) -> RawDataset:
        """
        取最近 n_recent 个交易日的日K（Phase 11 前向增量用）。日期窗口向前多留缓冲以覆盖节假日。
        """
        end = pd.Timestamp.utcnow().normalize()
        start = end - pd.Timedelta(days=max(7, n_recent * 3))
        return self.fetch(tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    # ------------------------------------------------------------------
    # 内部：逐标的 + 分页拉全
    # ------------------------------------------------------------------

    def _fetch_all_klines(self, ctx, moomoo, tickers: List[str],
                          start: str, end: str) -> Dict[str, pd.DataFrame]:
        RET_OK = moomoo.RET_OK
        ktype = moomoo.KLType.K_DAY
        autype = self._autype_enum()
        out: Dict[str, pd.DataFrame] = {}

        for i, tk in enumerate(tickers):
            code = _to_moomoo_code(tk, self.market)
            frames: List[pd.DataFrame] = []
            page_key = None
            while True:
                ret, data, page_key = ctx.request_history_kline(
                    code, start=start, end=end, ktype=ktype, autype=autype,
                    max_count=self.max_count, page_req_key=page_key,
                )
                if ret != RET_OK:
                    logger.warning("[moomoo] %s 历史K线失败，跳过: %s", code, data)
                    break
                if data is not None and len(data) > 0:
                    frames.append(data)
                if not page_key:
                    break
            if frames:
                out[tk] = pd.concat(frames, ignore_index=True)
            if self.throttle_s and i < len(tickers) - 1:
                time.sleep(self.throttle_s)
        return out

    def metadata(self) -> Dict:
        md = super().metadata()
        md.update({"source": "moomoo_opend", "host": self.host, "port": self.port,
                   "autype": self.autype})
        return md
