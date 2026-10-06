"""
broker_gateway.py — 券商网关（Phase 12.1）

为什么要一层抽象
----------------
执行逻辑（下单、对账、熔断、全平）是**确定性代码**，必须能在没有券商的地方被
完整测试；而 moomoo 的交易上下文只在本地 OpenD 网关启动并登录后才可用。
所以执行层只依赖 `BrokerGateway` 这个接口，`MoomooGateway` 是它的唯一生产实现。

moomoo SDK 的三个危险默认值（实测 moomoo-api 10.10.7008 的签名）
------------------------------------------------------------------
- `place_order / order_list_query / position_list_query / …` 的 `trd_env` 默认 **'REAL'**
  —— 漏传一次就是真钱下单。本模块**每一次**交易调用都显式传 `TrdEnv.SIMULATE`，
  并在构造时拒绝任何非 SIMULATE 的环境（`LiveTradingRefused`）。实盘不是配置项。
- `OpenSecTradeContext(filter_trdmarket=…)` 默认 **'HK'** —— 美股账户会查不到。显式传 US。
- `accinfo_query(currency=…)` 默认 **'HKD'** —— 资产会按港币报。显式传 USD。

失败语义（DEV_LESSONS §B / §U）
------------------------------
任何 `ret != RET_OK` 都抛 `BrokerError`，**绝不**返回空持仓 / 空订单列表代替。
空持仓在这里是一个具体且危险的断言："我什么都没持有" → 下游按满额重新建仓。
"""

from __future__ import annotations

import logging
import math
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from app.core.data_engine.providers.moomoo_provider import _from_moomoo_code, _to_moomoo_code

logger = logging.getLogger(__name__)

#: 唯一允许的交易环境。改这一行之前先读模块 docstring。
TRD_ENV = "SIMULATE"

#: remark 字段的上限（SDK 在 place_order 里按 UTF-8 字节数校验）
REMARK_MAX_BYTES = 64

#: 执行层统一的"零股"阈值：|股数| ≤ 它就当作没有持仓 / 没有成交。
#: 取 2 的幂（≈9.3e-10）而不是 1e-9：整数股 ± 它在浮点里**精确可表示**，
#: 边界上的行为因此可以被精确测试（1e-9 构造不出"恰好相等"的输入）。
#: 网关、订单生成、风控门、对账、执行账本全部用这一个值，不各写各的 epsilon。
DUST_QTY = 2.0 ** -30

#: 建连前的端口探测超时（秒）。
OPEND_PROBE_TIMEOUT_S = 2.0


def opend_reachable(host: str, port: int, timeout: float = OPEND_PROBE_TIMEOUT_S) -> bool:
    """
    OpenD 端口是否在监听。**必须在构造任何 SDK 上下文之前调用**：实测（moomoo-api 10.10）
    OpenD 不在时 `OpenSecTradeContext(...)` / `OpenQuoteContext(...)` 不会抛错，而是每 8 秒
    重连一次、**永不返回**，并起一个非守护线程 —— 调用线程被永久占住，进程也退不出去。
    """
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError as exc:
        logger.warning("[moomoo] OpenD %s:%s 未在监听：%s", host, port, exc)
        return False


class BrokerError(RuntimeError):
    """券商调用失败（连接失败或 ret != RET_OK）。调用方据此**不交易**，而不是当成"没有数据"。"""


class LiveTradingRefused(ValueError):
    """要求在实盘环境下单。Phase 12 只做纸交易，接真钱另立规划。"""


# ---------------------------------------------------------------------------
# 订单状态分类（moomoo OrderStatus 的字符串值）
# ---------------------------------------------------------------------------

#: 已经走完生命周期的状态。之后 dealt_qty 不会再变（FILL_CANCELLED 除外，它会**减少**）。
TERMINAL_STATUSES = frozenset({
    "FILLED_ALL", "CANCELLED_PART", "CANCELLED_ALL", "FAILED", "SUBMIT_FAILED",
    "DISABLED", "DELETED", "TIMEOUT", "FILL_CANCELLED",
})


def is_open_status(status: str) -> bool:
    """
    订单是否仍在途。**不在终态集合里的一律算在途** —— 包括 'N/A' 和 SDK 将来新增的
    未知状态。把未知状态当成"已结束"会让对账以为这笔单不会再成交，而它可能还会。
    """
    return str(status) not in TERMINAL_STATUSES


# ---------------------------------------------------------------------------
# 网关数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BrokerPosition:
    ticker:        str
    qty:           float      # 有符号：多头为正、空头为负
    can_sell_qty:  float
    nominal_price: float      # 券商现价
    market_val:    float      # USD


@dataclass(frozen=True)
class AccountSnapshot:
    total_assets: float
    cash:         float
    power:        float       # 买入力
    market_val:   float
    currency:     str


@dataclass(frozen=True)
class OrderRequest:
    client_id:   str          # 幂等键，写进 moomoo 的 remark
    ticker:      str          # 裸符号（'AAPL' / 'BRK-B'）
    side:        str          # "BUY" | "SELL"
    qty:         int
    limit_price: float


@dataclass(frozen=True)
class BrokerOrder:
    broker_order_id: str
    client_id:       str      # = remark；不是本系统下的单则为空或别的字符串
    ticker:          str
    side:            str
    qty:             float
    price:           float
    status:          str
    dealt_qty:       float
    dealt_avg_price: float
    create_time:     str
    updated_time:    str
    last_err_msg:    str


@dataclass
class OrderQuery:
    """
    订单查询结果。`history_ok=False` 表示**只拿到了当日订单**：
    对账时在当日列表里找不到的在途单不能被推断为"已结束"，必须当成未解决。
    `history_ok` 没有默认值 —— "查全了"是需要证明的断言，不能靠默认值宣称。
    """
    history_ok:    bool
    orders:        List[BrokerOrder] = field(default_factory=list)
    history_error: str = ""


class BrokerGateway(ABC):
    """执行层看到的券商。所有方法失败时抛 BrokerError。"""

    @abstractmethod
    def account(self) -> AccountSnapshot: ...

    @abstractmethod
    def positions(self) -> Dict[str, BrokerPosition]: ...

    @abstractmethod
    def place(self, req: OrderRequest) -> str:
        """下单，返回券商订单号。"""

    @abstractmethod
    def orders(self, since: date) -> OrderQuery: ...

    @abstractmethod
    def cancel(self, broker_order_id: str) -> None: ...

    @abstractmethod
    def describe(self) -> dict:
        """环境/账户描述（状态端点回显用）。"""

    def close(self) -> None:          # pragma: no cover - 默认无资源
        return None


# ---------------------------------------------------------------------------
# 数值解析：SDK 对缺失字段返回 'N/A' 字符串
# ---------------------------------------------------------------------------

def _to_float(v: Any) -> Optional[float]:
    """'N/A'/None/NaN/inf → None；其余转 float。不靠 try/except 吞错。"""
    x = pd.to_numeric(pd.Series([v], dtype=object), errors="coerce").iloc[0]
    return float(x) if pd.notna(x) and math.isfinite(float(x)) else None


def _num(v: Any, what: str) -> float:
    """必须是有限数。'N/A'/None/NaN 一律视为券商没给出该值 → BrokerError。"""
    x = _to_float(v)
    if x is None:
        raise BrokerError(f"券商返回的 {what} 不是有限数值：{v!r}")
    return x


def _num_or(v: Any, default: float) -> float:
    """
    非关键字段：缺失时取 default。只用于"缺了也不会让下单更冒险"的量 ——
    market_val（可由 qty×价 推出）、可卖数量、限价（市价单为 N/A）、零成交时的成交均价。
    数量/成交量这类决定对账结果的字段一律用 `_num`（缺失即报错）。
    """
    x = _to_float(v)
    return default if x is None else x


def _check_remark(client_id: str) -> None:
    if not client_id:
        raise ValueError("client_id 不能为空 —— 它是幂等键，没有它就无法判断是否重复下单")
    if len(client_id.encode("utf-8")) > REMARK_MAX_BYTES:
        raise ValueError(f"client_id 超过 {REMARK_MAX_BYTES} 字节：{client_id!r}")


# ---------------------------------------------------------------------------
# moomoo 实现
# ---------------------------------------------------------------------------

def _import_sdk():
    try:
        import moomoo  # noqa: F401
    except Exception as exc:
        raise BrokerError("未安装 moomoo-api（pip install moomoo-api），无法连接券商") from exc
    return moomoo


class MoomooGateway(BrokerGateway):
    """
    moomoo OpenD 交易上下文（美股 · 纸交易）。

    Parameters
    ----------
    security_firm   : 券商主体（'FUTUINC' = moomoo 美国）。必须是 SDK SecurityFirm 的成员。
    acc_id          : 0 = 取该市场第一个可用的**模拟**账户；非 0 时必须是一个模拟账户。
    trd_env         : 只接受 'SIMULATE'。传别的直接拒绝。
    context_factory : 测试注入点；返回一个与 OpenSecTradeContext 同签名的对象。
    sdk             : 测试注入点；默认 import moomoo。
    preflight       : 用默认建连路径时先探测 OpenD 端口（见 opend_reachable）。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        security_firm: str = "FUTUINC",
        acc_id: int = 0,
        trd_env: str = TRD_ENV,
        context_factory: Optional[Callable[[], Any]] = None,
        sdk: Any = None,
        preflight: bool = True,
    ) -> None:
        if str(trd_env).upper() != TRD_ENV:
            raise LiveTradingRefused(
                f"拒绝 trd_env={trd_env!r}：Phase 12 只允许纸交易（SIMULATE）。"
                f"接真实资金需要另立规划（密钥管理/实盘风控/人工签核），不是改一个参数。")
        self._m = sdk if sdk is not None else _import_sdk()
        if security_firm not in {k for k in vars(self._m.SecurityFirm) if k.isupper()}:
            raise ValueError(f"未知的 security_firm={security_firm!r}")
        self.host, self.port = host, int(port)
        self.security_firm = security_firm
        self.preflight = bool(preflight)
        self._ctx = (context_factory or self._default_context)()
        self._acc_id = self._resolve_account(int(acc_id))

    # -- 连接 -----------------------------------------------------------------

    def _default_context(self):
        m = self._m
        if self.preflight and not opend_reachable(self.host, self.port):
            raise BrokerError(
                f"OpenD {self.host}:{self.port} 未在监听 —— 不建连（SDK 在网关不在时会无限阻塞重连）。"
                f"请先启动 OpenD 并登录。")
        try:
            return m.OpenSecTradeContext(
                filter_trdmarket=m.TrdMarket.US, host=self.host, port=self.port,
                security_firm=getattr(m.SecurityFirm, self.security_firm))
        except Exception as exc:
            raise BrokerError(
                f"无法连接 OpenD 交易上下文 {self.host}:{self.port}（请确认 OpenD 已启动并登录）: {exc}"
            ) from exc

    def _ok(self, ret, data, what: str):
        if ret != self._m.RET_OK:
            raise BrokerError(f"{what} 失败：{data}")
        if data is None:
            raise BrokerError(f"{what} 返回空结果")
        return data

    def _resolve_account(self, acc_id: int) -> int:
        """
        明确选定一个**模拟**账户的 acc_id，之后每次调用都显式传它。
        不依赖 SDK 的 acc_index=0 顺序 —— 那个顺序里实盘账户和模拟账户是混在一起的。
        """
        df = self._ok(*self._ctx.get_acc_list(), what="get_acc_list")
        if acc_id:
            row = df[df["acc_id"] == acc_id]
            if row.empty:
                raise BrokerError(f"账户 {acc_id} 不在该市场的账户列表中")
            env = str(row.iloc[0]["trd_env"])
            if env != TRD_ENV:
                raise LiveTradingRefused(f"账户 {acc_id} 的环境是 {env}，不是模拟账户")
            return acc_id
        sims = df[(df["trd_env"] == TRD_ENV)
                  & df["trdmarket_auth"].apply(lambda a: "US" in list(a or []))
                  & (df["acc_status"].astype(str) != "DISABLED")]
        if sims.empty:
            raise BrokerError("没有可用的美股模拟账户（检查 security_firm / OpenD 登录账号）")
        stock = sims[sims["sim_acc_type"].astype(str).isin(["STOCK", "STOCK_AND_OPTION"])]
        chosen = int((stock if not stock.empty else sims).iloc[0]["acc_id"])
        logger.info("[moomoo] 选定模拟账户 acc_id=%s", chosen)
        return chosen

    def _env_kwargs(self) -> dict:
        return {"trd_env": self._m.TrdEnv.SIMULATE, "acc_id": self._acc_id, "acc_index": 0}

    # -- 查询 -----------------------------------------------------------------

    def account(self) -> AccountSnapshot:
        df = self._ok(*self._ctx.accinfo_query(
            refresh_cache=True, currency=self._m.Currency.USD, **self._env_kwargs()),
            what="accinfo_query")
        if len(df) == 0:
            raise BrokerError("accinfo_query 没有返回任何账户行")
        r = df.iloc[0]
        return AccountSnapshot(
            total_assets=_num(r["total_assets"], "total_assets"),
            cash=_num(r["cash"], "cash"),
            power=_num(r["power"], "power"),
            market_val=_num_or(r["market_val"], 0.0),
            currency=str(r.get("currency", "USD")),
        )

    def positions(self) -> Dict[str, BrokerPosition]:
        df = self._ok(*self._ctx.position_list_query(
            refresh_cache=True, currency=self._m.Currency.USD, **self._env_kwargs()),
            what="position_list_query")
        out: Dict[str, BrokerPosition] = {}
        for _, r in df.iterrows():
            mag = abs(_num(r["qty"], f"{r['code']} qty"))
            if mag <= DUST_QTY:
                continue                      # 当日已平的仓位行（qty=0）仍会被返回
            sign = -1.0 if str(r.get("position_side", "LONG")) == "SHORT" else 1.0
            tk = _from_moomoo_code(str(r["code"]))
            px = _num(r["nominal_price"], f"{tk} nominal_price")
            prev = out.get(tk)
            qty = sign * mag + (prev.qty if prev else 0.0)
            out[tk] = BrokerPosition(
                ticker=tk, qty=qty,
                can_sell_qty=_num_or(r["can_sell_qty"], 0.0) + (prev.can_sell_qty if prev else 0.0),
                nominal_price=px,
                market_val=_num_or(r["market_val"], qty * px),
            )
        return out

    def orders(self, since: date) -> OrderQuery:
        """
        当日订单（必须成功）+ 历史订单（尽力而为）。按 order_id 合并，取更新时间**严格更晚**
        的一行；时间相同保留先读到的当日行（refresh_cache=True，比历史接口新鲜）。
        历史接口失败时如实标记 `history_ok=False`，不假装拿全了。
        """
        today = self._ok(*self._ctx.order_list_query(
            order_id="", status_filter_list=[], code="", start="", end="",
            refresh_cache=True, **self._env_kwargs()), what="order_list_query")
        rows = [today]
        hist_ok, hist_err = True, ""
        try:
            ret, hist = self._ctx.history_order_list_query(
                status_filter_list=[], code="", start=f"{since:%Y-%m-%d} 00:00:00", end="",
                **self._env_kwargs())
            if ret != self._m.RET_OK:
                raise BrokerError(str(hist))
            rows.append(hist)
        except BrokerError as exc:
            hist_ok, hist_err = False, str(exc)
            logger.warning("[moomoo] 历史订单查询失败，只用当日订单对账（未找到的在途单将视为未解决）: %s", exc)

        merged: Dict[str, BrokerOrder] = {}
        for df in rows:
            for _, r in df.iterrows():
                o = self._to_order(r)
                cur = merged.get(o.broker_order_id)
                if cur is None or str(o.updated_time) > str(cur.updated_time):
                    merged[o.broker_order_id] = o
        return OrderQuery(history_ok=hist_ok, orders=list(merged.values()), history_error=hist_err)

    @staticmethod
    def _to_order(r) -> BrokerOrder:
        side = str(r["trd_side"])
        oid = str(r["order_id"])
        dealt = _num(r["dealt_qty"], f"订单 {oid} dealt_qty")
        avg = (_num(r["dealt_avg_price"], f"订单 {oid} dealt_avg_price") if dealt > 1e-9
               else _num_or(r["dealt_avg_price"], 0.0))
        return BrokerOrder(
            broker_order_id=oid,
            client_id=str(r["remark"] or ""),
            ticker=_from_moomoo_code(str(r["code"])),
            side="SELL" if side in ("SELL", "SELL_SHORT") else "BUY",
            qty=_num(r["qty"], f"订单 {oid} qty"),
            price=_num_or(r["price"], 0.0),
            status=str(r["order_status"]),
            dealt_qty=dealt,
            dealt_avg_price=avg,
            create_time=str(r["create_time"]),
            updated_time=str(r["updated_time"]),
            last_err_msg=str(r["last_err_msg"] or ""),
        )

    # -- 动作 -----------------------------------------------------------------

    def place(self, req: OrderRequest) -> str:
        _check_remark(req.client_id)
        if req.side not in ("BUY", "SELL"):
            raise ValueError(f"不支持的方向 {req.side!r}（Phase 12 只做多：BUY/SELL）")
        if int(req.qty) <= 0:
            raise ValueError(f"下单数量必须为正整数：{req.qty!r}")
        m = self._m
        df = self._ok(*self._ctx.place_order(
            price=float(req.limit_price), qty=int(req.qty), code=_to_moomoo_code(req.ticker),
            trd_side=m.TrdSide.BUY if req.side == "BUY" else m.TrdSide.SELL,
            order_type=m.OrderType.NORMAL, remark=req.client_id,
            time_in_force=m.TimeInForce.DAY, fill_outside_rth=False,
            **self._env_kwargs()), what=f"place_order {req.client_id}")
        if len(df) == 0 or "order_id" not in df.columns:
            raise BrokerError(f"place_order {req.client_id} 未返回订单号")
        return str(df["order_id"].iloc[0])

    def cancel(self, broker_order_id: str) -> None:
        self._ok(*self._ctx.modify_order(
            modify_order_op=self._m.ModifyOrderOp.CANCEL, order_id=str(broker_order_id),
            qty=0, price=0, **self._env_kwargs()), what=f"cancel {broker_order_id}")

    def describe(self) -> dict:
        return {"broker": "moomoo", "trd_env": TRD_ENV, "acc_id": self._acc_id,
                "security_firm": self.security_firm, "host": self.host, "port": self.port}

    def close(self) -> None:
        try:
            self._ctx.close()
        except Exception as exc:          # pragma: no cover - 关闭失败不影响已完成的动作
            logger.warning("[moomoo] 关闭交易上下文失败: %s", exc)


def make_gateway_from_settings() -> BrokerGateway:
    """生产入口：按 settings 构造 moomoo 纸交易网关。"""
    from app.config import settings
    return MoomooGateway(
        host=settings.moomoo_host, port=settings.moomoo_port,
        security_firm=settings.moomoo_security_firm,
        acc_id=int(settings.moomoo_trd_acc_id),
    )
