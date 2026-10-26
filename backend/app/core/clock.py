"""
clock.py — 信号 / 成交 / 标签 / 归档的时钟（2026-10-22 用户决定 D14 (a)；时钟版本 c2）

用户原话："T 收盘信息完成并过健康门后冻结信号与目标，T+1 按明确交易规则执行，更新时钟版本。同步修改信号、
回测、内部模拟、IC、归档、基准及调度。'计划开盘执行'不等于限价单必然按开盘价成交；预测 IC 与实际持仓收益
分别记录。"
C3（同批）："不能只提前 deadline —— 明确标签起止、信息截止和归档截止。"

c1（旧，2026-10-22 之前）：T 收盘后用 T−1 的原始信号（delay=1）；内部模拟按 T 收盘成交；IC = signal[T−1] 对
T 日收盘到收盘；券商订单 T 收盘后提交、计划 T+1 成交 —— 三条序列的持有区间各不相同。

c2（现在）：

    信息截止   T 日收盘（数据过健康门之后）
    决策       T 收盘后冻结信号与目标（信号不再多延迟一天：delay = 0）
    执行       T+1 常规盘，DAY 限价单；参考价 = T 收盘，限价带 = exec_limit_band_bps（与券商下单同一参数）
    数量       委托股数在 T 收盘按 T 收盘净值与参考价冻结（app/core/execution/order_intent.py，券商 / 纸账本 / 回测同一份）；
               撮合只决定成交多少、成交价，不按 T+1 开盘市值重新解释目标
    模拟成交   买：开盘 ≤ 限价 → 按开盘价；否则当日最低 ≤ 限价 → 按限价；否则不成交（卖对称）。
               成交量上限按截至 T 的 ADV（开盘时当天的量还不知道）
    估值       每个交易日收盘：隔夜（昨收→今开）算旧仓，盘中（今开→今收）算新仓
    标签       信号 T 的预测标签 = 开盘(T+1) → 开盘(T+2)：两次执行之间的持有期
    归档截止   开盘(T+1) —— 正好是标签起点，标签里没有任何在存档时已经发生的收益
    实际收益   模拟账本 / 券商按真实成交记账，与预测 IC 分开记

本模块是唯一来源：引擎、每日循环的 IC 结算、研究侧的前向收益、归档截止都从这里取。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

CLOCK = "c2"

#: 信号处理的执行延迟（天）：c2 下决策用到 T 收盘为止的信息，执行的那一天由引擎负责（T+1），信号本身不再延迟
SIGNAL_DELAY = 0


def describe() -> Dict[str, str]:
    """时钟的每一条边界（写进规格说明、文档与 Go/No-Go 报告）。"""
    return {
        "version": CLOCK,
        "information_cutoff": "T 日收盘（数据过健康门之后）",
        "decision": "T 收盘后冻结信号与目标；信号 delay = 0",
        "execution": "T+1 常规盘，DAY 限价单，参考价 = T 收盘，限价带 = exec_limit_band_bps",
        "fill_rule": "买：开盘 ≤ 限价 → 开盘价；否则最低 ≤ 限价 → 限价；否则不成交（卖对称）；成交量上限用截至 T 的 ADV",
        "valuation": "每日收盘估值：隔夜收益算旧仓，盘中收益算新仓",
        "label": "信号 T 的预测标签 = 开盘(T+1) → 开盘(T+2)",
        "archive_deadline": "开盘(T+1)，即标签起点",
        "realized": "模拟账本 / 券商按真实成交记账，与预测 IC 分开记录",
    }


# ---------------------------------------------------------------------------
# 标签与归档截止
# ---------------------------------------------------------------------------

def _next_sessions(day: pd.Timestamp, n: int):
    from app.core.data_engine.market_calendar import trading_days
    d = pd.Timestamp(day).normalize()
    days = [x for x in trading_days(d + pd.Timedelta(days=1), d + pd.Timedelta(days=20)) if x > d]
    if len(days) < n:
        raise ValueError(f"日历给不出 {d.date()} 之后的 {n} 个交易日")
    return days[:n]


def label_window(signal_day) -> Dict[str, object]:
    """信号日 T 的标签窗口：start = 开盘(T+1)，end = 开盘(T+2)（UTC）。"""
    from app.core.data_engine.market_calendar import session_open_utc
    t1, t2 = _next_sessions(pd.Timestamp(signal_day), 2)
    return {"start": session_open_utc(t1), "end": session_open_utc(t2), "start_day": t1, "end_day": t2}


def archive_deadline(signal_day):
    """信号日 T 的存档截止 = 标签起点 = 开盘(T+1)。"""
    return label_window(signal_day)["start"]


def label_returns(panel: Dict[str, pd.DataFrame], horizon: int = 1) -> pd.DataFrame:
    """
    按信号日对齐的预测标签：开盘(T+1) → 开盘(T+1+horizon)。末尾还没发生的为 NaN。
    没有开盘价 → ValueError（不用收盘价凑：那会把已经发生的隔夜收益算进标签）。
    """
    o = panel.get("open") if isinstance(panel, dict) else None
    if not isinstance(o, pd.DataFrame) or o.empty:
        raise ValueError("c2 标签需要 open（开盘价）；面板里没有")
    o = o.astype(float)
    start = o.shift(-1)
    end = o.shift(-1 - int(horizon))
    with np.errstate(divide="ignore", invalid="ignore"):
        return end / start.where(start != 0) - 1.0


def paired_returns(panel: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    给"信号 t 配 returns[t+1]"约定的消费方（AlphaCombiner 的 IC 加权 / 合成器拟合、工作流的组合 IC）用的收益矩阵：
    returns[t] = 开盘(t) → 开盘(t+1)，于是 returns[t+1] = 开盘(t+1) → 开盘(t+2) = 信号 t 的标签。
    没有开盘价 → ValueError（不拿收盘价凑）。
    """
    o = panel.get("open") if isinstance(panel, dict) else None
    if not isinstance(o, pd.DataFrame) or o.empty:
        raise ValueError("c2 的配对收益需要 open（开盘价）；面板里没有")
    o = o.astype(float)
    nxt = o.shift(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return nxt / o.where(o != 0) - 1.0


# ---------------------------------------------------------------------------
# 模拟成交规则
# ---------------------------------------------------------------------------

def limit_fill_prices(side: np.ndarray, ref: np.ndarray, open_: np.ndarray, high: np.ndarray,
                      low: np.ndarray, band: float) -> np.ndarray:
    """
    每名的成交价（不成交 → NaN）。side > 0 买、< 0 卖、0 不交易（NaN）。
    买：限价 = ref·(1+band)；开盘 ≤ 限价 → 开盘价；否则最低 ≤ 限价 → 限价；否则不成交。卖对称。
    任何一个价格缺失 → 不成交（不能假设成交）。
    """
    side = np.asarray(side, dtype=float)
    ref, o, h, lo = (np.asarray(x, dtype=float) for x in (ref, open_, high, low))
    out = np.full(side.shape, np.nan)
    ok = np.isfinite(ref) & np.isfinite(o) & np.isfinite(h) & np.isfinite(lo) & (ref > 0) & (o > 0)
    buy = ok & (side > 0)
    sell = ok & (side < 0)
    lim_b = ref * (1.0 + band)
    lim_s = ref * (1.0 - band)
    out = np.where(buy & (o <= lim_b), o, out)
    out = np.where(buy & (o > lim_b) & (lo <= lim_b), lim_b, out)
    out = np.where(sell & (o >= lim_s), o, out)
    out = np.where(sell & (o < lim_s) & (h >= lim_s), lim_s, out)
    return out


def default_band() -> float:
    """缺省限价带（小数）：settings.exec_limit_band_bps（与券商下单同一参数）。"""
    try:
        from app.config import settings
        return float(getattr(settings, "exec_limit_band_bps", 50.0)) * 1e-4
    except Exception as exc:                                  # pragma: no cover - 配置坏了才会到这
        import logging
        logging.getLogger(__name__).error("[clock] 读不到 exec_limit_band_bps，按 50bp: %s", exc)
        return 50.0 * 1e-4


def require_ohlc(ohlc: Optional[Dict[str, pd.DataFrame]], index, columns) -> Dict[str, np.ndarray]:
    """引擎用：取 open / high / low 对齐成数组；缺任何一项 → ValueError（c2 没有它们无法按规则成交）。"""
    if not ohlc:
        raise ValueError("时钟 c2 的引擎需要 open / high / low（T+1 按限价规则成交）；没有传入 ohlc")
    out = {}
    for k in ("open", "high", "low"):
        df = ohlc.get(k)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"时钟 c2 的引擎需要 {k}；ohlc 里没有")
        out[k] = df.reindex(index=index, columns=columns).to_numpy(dtype=float)
    return out
