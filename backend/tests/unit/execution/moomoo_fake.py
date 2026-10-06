"""
moomoo_fake.py — moomoo OpenSecTradeContext 的测试替身 + 一个最小撮合市场

替身与真实 SDK 之间的两条契约由 test_broker_gateway 机械核对（不是"看起来像"）：
  1. 每次调用记录下来的关键字参数，都必须能 bind 到**已安装 SDK** 对应方法的签名
     —— 网关里拼错一个参数名（`trdenv=`），测试就红，而不是等到连上 OpenD 才炸（§C）。
  2. 返回 DataFrame 的列 = SDK 源码里该方法的 `col_list`，**用 AST 从已安装的 SDK
     源码读出**，不是手抄 —— 网关读了真实返回里不存在的列，测试时就 KeyError。

撮合：收盘后挂的 DAY 限价单在 `open_session(开盘价)` 时按开盘价成交（买单 开盘价 ≤ 限价、
卖单 开盘价 ≥ 限价），可指定部分成交比例；`expire_day()` 把仍在途的单撤掉。
"""

from __future__ import annotations

import ast
import inspect
import itertools
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import moomoo

from app.core.data_engine.providers.moomoo_provider import _from_moomoo_code, _to_moomoo_code

RET_OK, RET_ERROR = moomoo.RET_OK, moomoo.RET_ERROR
_OPEN = {"SUBMITTED", "FILLED_PART", "WAITING_SUBMIT", "SUBMITTING"}


@lru_cache(maxsize=None)
def sdk_col_list(method: str) -> tuple:
    """已安装 moomoo SDK 源码里 `method` 的 col_list（AST 读取）。"""
    path = Path(inspect.getfile(moomoo.OpenSecTradeContext))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == method:
                for node in ast.walk(fn):
                    if (isinstance(node, ast.Assign)
                            and any(getattr(t, "id", None) == "col_list" for t in node.targets)):
                        return tuple(e.value for e in node.value.elts)
    raise LookupError(f"SDK 源码里找不到 {method} 的 col_list")


def default_accounts() -> List[dict]:
    # 实盘账户故意排在前面：网关不能按顺序取第一个
    return [
        {"acc_id": 1001, "trd_env": "REAL", "acc_type": "MARGIN", "security_firm": "FUTUINC",
         "sim_acc_type": "N/A", "trdmarket_auth": ["US"], "acc_status": "ACTIVE"},
        {"acc_id": 2001, "trd_env": "SIMULATE", "acc_type": "MARGIN", "security_firm": "N/A",
         "sim_acc_type": "STOCK", "trdmarket_auth": ["US"], "acc_status": "ACTIVE"},
    ]


class FakeTradeContext:
    def __init__(self, *, accounts: Optional[List[dict]] = None, cash: float = 100_000.0,
                 history_supported: bool = True) -> None:
        self.calls: List[tuple] = []
        self.accounts = accounts if accounts is not None else default_accounts()
        self.cash = float(cash)
        self.pos: Dict[str, dict] = {}            # code -> {qty, nominal_price}
        self.orders: Dict[str, dict] = {}         # order_id -> row
        self._ids = itertools.count(90001)
        self.fail: Dict[str, str] = {}            # method -> 错误信息（持续生效直到删除）
        self.history_supported = history_supported
        self.now = "2025-03-03 16:30:00.000"      # 交易所本地时间
        self.on_order_read: Optional[Callable[["FakeTradeContext"], None]] = None
        #: 置为错误信息时：place_order **照收这张单**，但给调用方返回错误（模拟超时）
        self.accept_then_error: Optional[str] = None
        #: 额外追加到持仓查询结果的原始行（模拟同一代码多行、字段为 'N/A' 等）
        self.extra_position_rows: List[dict] = []
        #: 历史订单接口返回前对行做变换（模拟历史接口给出陈旧副本）
        self.history_transform: Optional[Callable[[List[dict]], List[dict]]] = None
        self.closed = False

    # -- 工具 ------------------------------------------------------------------

    def _rec(self, name: str, kw: dict) -> None:
        self.calls.append((name, dict(kw)))

    def _df(self, method: str, rows: List[dict]) -> pd.DataFrame:
        cols = list(sdk_col_list(method))
        return pd.DataFrame([{c: r.get(c, "N/A") for c in cols} for r in rows], columns=cols)

    def calls_of(self, name: str) -> List[dict]:
        return [kw for n, kw in self.calls if n == name]

    # -- SDK 方法（名称与参数名必须与真实 SDK 一致）----------------------------

    def get_acc_list(self):
        self._rec("get_acc_list", {})
        if "get_acc_list" in self.fail:
            return RET_ERROR, self.fail["get_acc_list"]
        return RET_OK, self._df("get_acc_list", self.accounts)

    def accinfo_query(self, **kw):
        self._rec("accinfo_query", kw)
        if "accinfo_query" in self.fail:
            return RET_ERROR, self.fail["accinfo_query"]
        mv = sum(p["qty"] * p["nominal_price"] for p in self.pos.values())
        row = {"power": self.cash, "total_assets": self.cash + mv, "cash": self.cash,
               "market_val": mv, "currency": "USD"}
        if getattr(self, "account_override", None):
            row.update(self.account_override)
        return RET_OK, self._df("accinfo_query", [row])

    def position_list_query(self, **kw):
        self._rec("position_list_query", kw)
        if "position_list_query" in self.fail:
            return RET_ERROR, self.fail["position_list_query"]
        rows = [{"code": c, "qty": abs(p["qty"]), "can_sell_qty": max(p["qty"], 0.0),
                 "nominal_price": p["nominal_price"], "market_val": p["qty"] * p["nominal_price"],
                 "position_side": "SHORT" if p["qty"] < 0 else "LONG", "currency": "USD"}
                for c, p in self.pos.items()]
        return RET_OK, self._df("position_list_query", rows + list(self.extra_position_rows))

    def place_order(self, price, qty, code, trd_side, **kw):
        self._rec("place_order", {"price": price, "qty": qty, "code": code,
                                  "trd_side": trd_side, **kw})
        if "place_order" in self.fail:
            return RET_ERROR, self.fail["place_order"]
        oid = str(next(self._ids))
        self.orders[oid] = {
            "code": code, "trd_side": trd_side, "order_type": kw.get("order_type"),
            "order_status": "SUBMITTED", "order_id": oid, "qty": float(qty), "price": float(price),
            "create_time": self.now, "updated_time": self.now, "dealt_qty": 0.0,
            "dealt_avg_price": 0.0, "last_err_msg": "", "remark": kw.get("remark") or "",
            "time_in_force": kw.get("time_in_force"), "currency": "USD"}
        if self.accept_then_error:
            return RET_ERROR, self.accept_then_error
        return RET_OK, self._df("place_order", [self.orders[oid]])

    def order_list_query(self, **kw):
        self._rec("order_list_query", kw)
        if "order_list_query" in self.fail:
            return RET_ERROR, self.fail["order_list_query"]
        out = self._df("order_list_query", list(self.orders.values()))
        if self.on_order_read is not None:
            self.on_order_read(self)
        return RET_OK, out

    def history_order_list_query(self, **kw):
        self._rec("history_order_list_query", kw)
        if not self.history_supported:
            return RET_ERROR, "simulate trading does not support history orders"
        rows = [dict(o) for o in self.orders.values()]
        if self.history_transform is not None:
            rows = self.history_transform(rows)
        return RET_OK, self._df("history_order_list_query", rows)

    def modify_order(self, modify_order_op, order_id, qty, price, **kw):
        self._rec("modify_order", {"modify_order_op": modify_order_op, "order_id": order_id,
                                   "qty": qty, "price": price, **kw})
        if "modify_order" in self.fail:
            return RET_ERROR, self.fail["modify_order"]
        o = self.orders.get(str(order_id))
        if o is None or o["order_status"] not in _OPEN:
            return RET_ERROR, f"order {order_id} not cancellable"
        o["order_status"] = "CANCELLED_PART" if o["dealt_qty"] > 0 else "CANCELLED_ALL"
        o["updated_time"] = self.now
        return RET_OK, self._df("modify_order", [{"trd_env": kw.get("trd_env"), "order_id": order_id}])

    def close(self):
        self.closed = True

    # -- 撮合市场 ------------------------------------------------------------------

    def set_position(self, ticker: str, qty: float, price: float) -> None:
        self.pos[_to_moomoo_code(ticker)] = {"qty": float(qty), "nominal_price": float(price)}

    def mark(self, prices: Dict[str, float]) -> None:
        for tk, px in prices.items():
            c = _to_moomoo_code(tk)
            if c in self.pos:
                self.pos[c]["nominal_price"] = float(px)

    def _fill(self, o: dict, px: float, qty: float) -> None:
        if qty <= 0:
            return
        old_q, old_p = o["dealt_qty"], o["dealt_avg_price"]
        new_q = old_q + qty
        o["dealt_avg_price"] = (old_p * old_q + px * qty) / new_q
        o["dealt_qty"] = new_q
        o["order_status"] = "FILLED_ALL" if new_q >= o["qty"] - 1e-9 else "FILLED_PART"
        o["updated_time"] = self.now
        sign = 1.0 if o["trd_side"] == "BUY" else -1.0
        p = self.pos.setdefault(o["code"], {"qty": 0.0, "nominal_price": px})
        p["qty"] += sign * qty
        self.cash -= sign * qty * px
        if abs(p["qty"]) < 1e-9:
            del self.pos[o["code"]]

    def open_session(self, opens: Dict[str, float], ratio: float = 1.0) -> None:
        """按开盘价撮合所有在途单（可成交才成交），ratio = 本次成交占剩余量的比例。"""
        for o in list(self.orders.values()):
            if o["order_status"] not in _OPEN:
                continue
            tk = _from_moomoo_code(o["code"])
            if tk not in opens:
                continue
            px = float(opens[tk])
            ok = px <= o["price"] if o["trd_side"] == "BUY" else px >= o["price"]
            if ok:
                remaining = o["qty"] - o["dealt_qty"]
                self._fill(o, px, float(int(remaining * ratio + 1e-9)))

    def expire_day(self) -> None:
        for o in self.orders.values():
            if o["order_status"] in _OPEN:
                o["order_status"] = "CANCELLED_PART" if o["dealt_qty"] > 0 else "CANCELLED_ALL"
                o["updated_time"] = self.now


def make_gateway(ctx: FakeTradeContext, **kw):
    from app.core.execution.broker_gateway import MoomooGateway
    return MoomooGateway(sdk=moomoo, context_factory=lambda: ctx, **kw)
