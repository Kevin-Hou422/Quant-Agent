"""
test_broker_gateway.py — Phase 12.1 券商网关（MoomooGateway）

重点不是"能调通"，而是这几件**写错了不报错、只会让钱出事**的事：
  · 永远不在实盘环境下单（SDK 的 trd_env 默认是 'REAL'）
  · 账户按模拟账户挑选，不按列表顺序
  · 每一次交易调用都显式带 SIMULATE + 选定 acc_id + USD
  · 失败抛 BrokerError，绝不返回空持仓/空订单冒充"什么都没有"
  · 调用参数名与返回列名与**已安装 SDK** 一致（替身契约，见 moomoo_fake）
"""

from __future__ import annotations

import inspect
from datetime import date
from types import SimpleNamespace

import moomoo
import pytest

from app.core.execution.broker_gateway import (
    TERMINAL_STATUSES, BrokerError, LiveTradingRefused, MoomooGateway, OrderRequest,
    is_open_status,
)
from tests.unit.execution.moomoo_fake import FakeTradeContext, make_gateway, sdk_col_list


# ---------------------------------------------------------------------------
# 实盘拒绝 / 账户选择
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env", ["REAL", "real", "", "PAPER"])
def test_any_non_simulate_env_is_refused_before_connecting(env):
    connected = []
    with pytest.raises(LiveTradingRefused):
        MoomooGateway(trd_env=env, sdk=moomoo,
                      context_factory=lambda: connected.append(1) or FakeTradeContext())
    assert connected == [], "拒绝实盘必须发生在建连之前"


def test_account_resolution_skips_the_real_account_listed_first():
    ctx = FakeTradeContext()
    gw = make_gateway(ctx)
    assert gw.describe()["acc_id"] == 2001
    assert gw.describe()["trd_env"] == "SIMULATE"


def test_explicit_real_account_id_is_refused():
    with pytest.raises(LiveTradingRefused, match="1001"):
        make_gateway(FakeTradeContext(), acc_id=1001)


def test_unknown_account_id_and_missing_simulate_account_are_errors():
    with pytest.raises(BrokerError, match="4242"):
        make_gateway(FakeTradeContext(), acc_id=4242)
    only_real = [a for a in FakeTradeContext().accounts if a["trd_env"] == "REAL"]
    with pytest.raises(BrokerError, match="模拟账户"):
        make_gateway(FakeTradeContext(accounts=only_real))


def test_disabled_and_non_us_simulate_accounts_are_not_chosen():
    accts = [
        {"acc_id": 1, "trd_env": "SIMULATE", "trdmarket_auth": ["HK"], "acc_status": "ACTIVE",
         "sim_acc_type": "STOCK"},
        {"acc_id": 2, "trd_env": "SIMULATE", "trdmarket_auth": ["US"], "acc_status": "DISABLED",
         "sim_acc_type": "STOCK"},
        {"acc_id": 3, "trd_env": "SIMULATE", "trdmarket_auth": ["US"], "acc_status": "ACTIVE",
         "sim_acc_type": "OPTION"},
        {"acc_id": 4, "trd_env": "SIMULATE", "trdmarket_auth": ["US"], "acc_status": "ACTIVE",
         "sim_acc_type": "STOCK"},
    ]
    assert make_gateway(FakeTradeContext(accounts=accts)).describe()["acc_id"] == 4


def test_unknown_security_firm_is_rejected():
    with pytest.raises(ValueError, match="security_firm"):
        make_gateway(FakeTradeContext(), security_firm="NOT_A_FIRM")


def _sdk_with(ctx_factory):
    sdk = SimpleNamespace(**{k: getattr(moomoo, k) for k in dir(moomoo) if not k.startswith("_")})
    sdk.OpenSecTradeContext = ctx_factory
    return sdk


@pytest.fixture
def no_real_sdk_context(monkeypatch):
    """真实 SDK 的构造在 OpenD 不在时会永久阻塞 —— 任何用例都不许真的走到它。"""
    def _forbidden(**kw):
        raise AssertionError("用到了真实 SDK 的 OpenSecTradeContext，而不是注入的 sdk")
    monkeypatch.setattr(moomoo, "OpenSecTradeContext", _forbidden)


def test_default_context_is_us_market_with_explicit_firm(monkeypatch, no_real_sdk_context):
    """不注入 context_factory 时，建连参数必须显式给 US 与券商主体（SDK 默认是 HK / N/A）。"""
    import app.core.execution.broker_gateway as bg
    monkeypatch.setattr(bg, "opend_reachable", lambda host, port, timeout=2.0: True)
    seen = {}

    def _ctx(**kw):
        seen.update(kw)
        return FakeTradeContext()

    MoomooGateway(sdk=_sdk_with(_ctx), host="10.0.0.5", port=12345, security_firm="FUTUINC")
    assert seen == {"filter_trdmarket": "US", "host": "10.0.0.5", "port": 12345,
                    "security_firm": "FUTUINC"}


def test_unreachable_opend_fails_fast_without_constructing(monkeypatch, no_real_sdk_context):
    """
    实测：OpenD 不在时 SDK 构造会每 8 秒重连、永不返回，还留下非守护线程让进程退不出去。
    所以端口探测失败必须**在构造之前**就抛 BrokerError。
    """
    import app.core.execution.broker_gateway as bg
    probed, built = [], []
    monkeypatch.setattr(bg, "opend_reachable",
                        lambda host, port, timeout=2.0: probed.append((host, port)) or False)
    with pytest.raises(BrokerError, match="未在监听"):
        MoomooGateway(sdk=_sdk_with(lambda **kw: built.append(kw)), host="127.0.0.1", port=11111)
    assert probed == [("127.0.0.1", 11111)] and built == []


def test_preflight_can_be_disabled_explicitly(monkeypatch, no_real_sdk_context):
    import app.core.execution.broker_gateway as bg
    monkeypatch.setattr(bg, "opend_reachable",
                        lambda *a, **k: pytest.fail("preflight=False 时不应探测"))
    built = []
    MoomooGateway(sdk=_sdk_with(lambda **kw: built.append(kw) or FakeTradeContext()),
                  preflight=False)
    assert len(built) == 1


def test_opend_reachable_detects_listening_and_closed_ports():
    import socket
    from app.core.execution.broker_gateway import opend_reachable
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert opend_reachable("127.0.0.1", port, timeout=1.0) is True
    finally:
        srv.close()
    assert opend_reachable("127.0.0.1", port, timeout=1.0) is False


def test_connection_failure_is_a_broker_error(monkeypatch):
    import app.core.execution.broker_gateway as bg
    monkeypatch.setattr(bg, "opend_reachable", lambda *a, **k: True)

    def _boom(**kw):
        raise ConnectionRefusedError("OpenD down")
    with pytest.raises(BrokerError, match="OpenD"):
        MoomooGateway(sdk=_sdk_with(_boom))


# ---------------------------------------------------------------------------
# 每次交易调用的环境参数 + 替身契约
# ---------------------------------------------------------------------------

def _exercise(gw: MoomooGateway, ctx: FakeTradeContext) -> None:
    ctx.set_position("AAPL", 10, 190.0)
    gw.account()
    gw.positions()
    oid = gw.place(OrderRequest("qaR-1-20250303-AAPL-B", "AAPL", "BUY", 5, 191.0))
    gw.orders(date(2025, 3, 1))
    gw.cancel(oid)


TRADING_CALLS = ("accinfo_query", "position_list_query", "place_order", "order_list_query",
                 "history_order_list_query", "modify_order")


def test_every_trading_call_is_simulate_with_the_resolved_account():
    ctx = FakeTradeContext()
    gw = make_gateway(ctx)
    _exercise(gw, ctx)
    seen = set()
    for name, kw in ctx.calls:
        if name not in TRADING_CALLS:
            continue
        seen.add(name)
        assert kw.get("trd_env") == "SIMULATE", f"{name} 没有显式传 SIMULATE：{kw}"
        assert kw.get("acc_id") == 2001, f"{name} 没有显式传选定的模拟账户：{kw}"
    assert seen == set(TRADING_CALLS), f"有交易调用没被走到：{set(TRADING_CALLS) - seen}"
    for name in ("accinfo_query", "position_list_query"):
        assert all(kw["currency"] == "USD" for kw in ctx.calls_of(name)), f"{name} 未按 USD 计价"
    # 对账读的是券商**服务端**状态，不是 OpenD 本地缓存（缓存可能落后于成交）
    for name in ("accinfo_query", "position_list_query", "order_list_query"):
        assert ctx.calls_of(name) and all(kw.get("refresh_cache") is True
                                          for kw in ctx.calls_of(name)), f"{name} 读了缓存"


def test_recorded_kwargs_bind_to_the_installed_sdk_signatures():
    """网关传的每个参数名都必须是真实 SDK 方法的形参（拼错 → 这里红，而不是连上 OpenD 才炸）。"""
    ctx = FakeTradeContext()
    _exercise(make_gateway(ctx), ctx)
    bad, checked = [], set()
    for name, kw in ctx.calls:
        sig = inspect.signature(getattr(moomoo.OpenSecTradeContext, name))
        try:
            sig.bind(None, **kw)
        except TypeError as exc:
            bad.append(f"{name}({sorted(kw)}): {exc}")
        checked.add(name)
    assert not bad, "网关传了 SDK 不认识的参数：\n" + "\n".join(bad)
    assert set(TRADING_CALLS) | {"get_acc_list"} <= checked, "有调用没被核对到（空集合上恒真）"


@pytest.mark.parametrize("method", ["get_acc_list", "accinfo_query", "position_list_query",
                                    "order_list_query", "history_order_list_query", "place_order"])
def test_fake_returns_exactly_the_sdk_columns(method):
    """替身的列来自 SDK 源码（AST），这里再确认替身确实返回了这些列 —— 两边不会悄悄分叉。"""
    ctx = FakeTradeContext()
    gw = make_gateway(ctx)
    _exercise(gw, ctx)
    fn = getattr(ctx, method)
    if method == "place_order":
        ret, df = fn(price=1.0, qty=1, code="US.AAPL", trd_side="BUY",
                     trd_env="SIMULATE", acc_id=2001, acc_index=0, remark="x")
    else:
        ret, df = fn() if method == "get_acc_list" else fn(trd_env="SIMULATE", acc_id=2001)
    assert ret == moomoo.RET_OK
    assert tuple(df.columns) == sdk_col_list(method)
    assert len(sdk_col_list(method)) >= 5, "AST 没解析到列名（空集合上的契约恒真）"


def test_place_order_parameters():
    ctx = FakeTradeContext()
    gw = make_gateway(ctx)
    oid = gw.place(OrderRequest("qaR-1-20250303-BRK-B-S", "BRK-B", "SELL", 3, 410.5))
    kw = ctx.calls_of("place_order")[-1]
    assert kw["code"] == "US.BRK.B"
    assert kw["trd_side"] == "SELL" and kw["qty"] == 3 and kw["price"] == 410.5
    assert kw["order_type"] == "NORMAL" and kw["time_in_force"] == "DAY"
    assert kw["fill_outside_rth"] is False
    assert kw["remark"] == "qaR-1-20250303-BRK-B-S"
    assert oid in ctx.orders


@pytest.mark.parametrize("req,match", [
    (OrderRequest("x" * 65, "AAPL", "BUY", 1, 1.0), "64"),
    (OrderRequest("", "AAPL", "BUY", 1, 1.0), "幂等键"),
    (OrderRequest("qaR-1-1-A-B", "AAPL", "BUY", 0, 1.0), "正整数"),
    (OrderRequest("qaR-1-1-A-B", "AAPL", "SELL_SHORT", 1, 1.0), "方向"),
])
def test_invalid_orders_never_reach_the_sdk(req, match):
    ctx = FakeTradeContext()
    gw = make_gateway(ctx)
    with pytest.raises(ValueError, match=match):
        gw.place(req)
    assert ctx.calls_of("place_order") == []


# ---------------------------------------------------------------------------
# 失败语义：抛错，不返回空结果
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,call", [
    ("accinfo_query", lambda gw: gw.account()),
    ("position_list_query", lambda gw: gw.positions()),
    ("order_list_query", lambda gw: gw.orders(date(2025, 3, 1))),
    ("place_order", lambda gw: gw.place(OrderRequest("qaR-1-1-AAPL-B", "AAPL", "BUY", 1, 1.0))),
    ("modify_order", lambda gw: gw.cancel("123")),
])
def test_sdk_errors_raise_broker_error(method, call):
    ctx = FakeTradeContext()
    gw = make_gateway(ctx)
    ctx.fail[method] = "network timeout"
    with pytest.raises(BrokerError, match="network timeout"):
        call(gw)


def test_missing_account_numbers_are_errors_not_zeros():
    ctx = FakeTradeContext()
    gw = make_gateway(ctx)
    ctx.account_override = {"total_assets": "N/A"}
    with pytest.raises(BrokerError, match="total_assets"):
        gw.account()
    ctx.account_override = {"power": float("nan")}
    with pytest.raises(BrokerError, match="power"):
        gw.account()
    ctx.account_override = {"cash": float("inf")}
    with pytest.raises(BrokerError, match="cash"):
        gw.account()


def test_remark_of_exactly_64_bytes_reaches_the_sdk():
    ctx = FakeTradeContext()
    gw = make_gateway(ctx)
    cid = "q" * 64
    gw.place(OrderRequest(cid, "AAPL", "BUY", 1, 1.0))
    assert ctx.calls_of("place_order")[-1]["remark"] == cid


def test_order_query_must_state_whether_history_was_complete():
    from app.core.execution.broker_gateway import OrderQuery
    with pytest.raises(TypeError):
        OrderQuery()                                  # "查全了"不能靠默认值宣称


def test_dust_position_rows_are_dropped():
    from app.core.execution.broker_gateway import DUST_QTY
    ctx = FakeTradeContext()
    ctx.set_position("AAA", DUST_QTY, 10.0)
    ctx.set_position("BBB", 2 * DUST_QTY, 10.0)
    assert set(make_gateway(ctx).positions()) == {"BBB"}


def test_duplicate_position_rows_are_merged_and_missing_values_derived():
    ctx = FakeTradeContext()
    ctx.set_position("AAPL", 10, 200.0)
    ctx.extra_position_rows = [
        {"code": "US.AAPL", "qty": 5, "can_sell_qty": 3, "nominal_price": 200.0,
         "market_val": 1000.0, "position_side": "LONG"},
        {"code": "US.MSFT", "qty": 4, "can_sell_qty": 4, "nominal_price": 300.0,
         "market_val": "N/A", "position_side": "LONG"},
    ]
    pos = make_gateway(ctx).positions()
    assert pos["AAPL"].qty == 15 and pos["AAPL"].can_sell_qty == 13
    assert pos["MSFT"].market_val == pytest.approx(1200.0), "缺市值时应由 股数×现价 推出"


def test_same_timestamp_keeps_the_fresh_today_row():
    """当日接口（refresh_cache）与历史接口给出同一订单、同一更新时间时，以当日行为准。"""
    ctx = FakeTradeContext()
    gw = make_gateway(ctx)
    gw.place(OrderRequest("qaR-1-20250303-AAA-B", "AAA", "BUY", 10, 50.0))
    o = next(iter(ctx.orders.values()))
    o["dealt_qty"], o["dealt_avg_price"], o["order_status"] = 10.0, 50.0, "FILLED_ALL"

    def _stale(rows):
        for r in rows:
            r.update(dealt_qty=0.0, dealt_avg_price=0.0, order_status="SUBMITTED")
        return rows
    ctx.history_transform = _stale
    q = gw.orders(date(2025, 3, 1))
    assert len(q.orders) == 1 and q.orders[0].dealt_qty == 10.0


def test_positions_parsing_signs_codes_and_zero_rows():
    ctx = FakeTradeContext()
    gw = make_gateway(ctx)
    ctx.set_position("AAPL", 10, 190.0)
    ctx.set_position("BRK-B", -4, 400.0)
    ctx.set_position("MSFT", 0, 300.0)            # 当日已平：SDK 仍会返回 qty=0 的行
    pos = gw.positions()
    assert set(pos) == {"AAPL", "BRK-B"}
    assert pos["AAPL"].qty == 10 and pos["AAPL"].market_val == pytest.approx(1900.0)
    assert pos["BRK-B"].qty == -4, "空头必须是负数，否则会被当成多头去卖"


def test_order_with_unparseable_dealt_qty_is_an_error():
    ctx = FakeTradeContext()
    gw = make_gateway(ctx)
    gw.place(OrderRequest("qaR-1-20250303-AAPL-B", "AAPL", "BUY", 5, 191.0))
    next(iter(ctx.orders.values()))["dealt_qty"] = "N/A"
    with pytest.raises(BrokerError, match="dealt_qty"):
        gw.orders(date(2025, 3, 1))


def test_orders_merge_prefers_latest_and_reports_missing_history():
    ctx = FakeTradeContext()
    gw = make_gateway(ctx)
    gw.place(OrderRequest("qaR-1-20250303-AAPL-B", "AAPL", "BUY", 5, 191.0))
    q = gw.orders(date(2025, 3, 1))
    assert q.history_ok and len(q.orders) == 1, "同一订单出现在当日与历史两张表里，必须合并成一行"
    assert q.orders[0].client_id == "qaR-1-20250303-AAPL-B" and q.orders[0].ticker == "AAPL"

    ctx.history_supported = False
    q2 = gw.orders(date(2025, 3, 1))
    assert q2.history_ok is False and "history" in q2.history_error
    assert len(q2.orders) == 1, "历史接口失败时当日订单仍要返回"


def test_history_query_start_uses_since_date():
    ctx = FakeTradeContext()
    make_gateway(ctx).orders(date(2025, 2, 24))
    assert ctx.calls_of("history_order_list_query")[-1]["start"] == "2025-02-24 00:00:00"


def test_status_classification_treats_unknown_as_open():
    for s in TERMINAL_STATUSES:
        assert is_open_status(s) is False
    for s in ("SUBMITTED", "FILLED_PART", "N/A", "SOME_NEW_STATUS"):
        assert is_open_status(s) is True, f"{s} 被当成已结束 —— 之后的成交会被漏记"


def test_terminal_status_set_matches_sdk_enum():
    """终态集合里的每个名字都必须是 SDK 真有的状态（拼错 = 那个状态永远被当成在途）。"""
    sdk = {k for k in vars(moomoo.OrderStatus) if k.isupper()}
    assert TERMINAL_STATUSES <= sdk, TERMINAL_STATUSES - sdk
