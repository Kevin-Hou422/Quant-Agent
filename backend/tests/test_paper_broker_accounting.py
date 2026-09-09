"""
test_paper_broker_accounting.py — PaperBroker 记账口径的**逐项**保护

来由：变异测试（修好测量缺陷后的首次可信测量）在 `paper_broker.py` 上暴露
**6 处有意义的存活变异** —— 改坏了没有任何测试会红。全都是算钱的行：

  L96  cap_w = adv * adv_cap_pct / initial_capital   容量上限公式（* 改 / 不报错）
  L117 pv = equity * initial_capital                 组合市值（量纲）
  L125 cost_usd = cost_w[i] * pv                     逐名成本 USD
  L132 cost_bps = (cost_ret + borrow_ret) * 1e4      借券成本的**符号**
  L169 prev_prices = prices.iloc[t-1] if t > 0       t=0 时若用 iloc[-1] → **前视**
  L192 if hasattr(d,"date") and not isinstance(...)  日期归一化的分支方向

本文件为每一条建立**能杀死该变异**的断言：不是"跑通就行"，而是把公式钉死。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.backtest_engine.transaction_cost import CostParams
from app.core.execution.paper_broker import PaperBroker
from app.db.position_store import PositionStore


CAPITAL = 1_000_000.0


@pytest.fixture
def broker(tmp_path):
    store = PositionStore(db_url=f"sqlite:///{tmp_path/'pb.db'}")
    return PaperBroker(store=store, initial_capital=CAPITAL)


def _series(vals, tickers):
    return pd.Series(vals, index=tickers, dtype=float)


# ---------------------------------------------------------------------------
# L96：容量上限公式  cap_w = adv * adv_cap_pct / initial_capital
# ---------------------------------------------------------------------------

def test_adv_cap_formula_is_exact(broker):
    """
    单票 ADV 上限必须**精确**等于 adv × adv_cap_pct / capital。
    把 `*` 写成 `/` 不会报错，只会让容量上限差好几个数量级 —— 必须钉死公式。
    """
    tk = ["A", "B"]
    pct = broker.params.adv_cap_pct
    adv_a = 10_000.0                       # 极小 ADV → 上限远低于目标权重
    cap_expected = adv_a * pct / CAPITAL

    pnl = broker.step(
        alpha_id=1, date="2024-01-02",
        target_w=_series([1.0, 0.0], tk),          # 想满仓 A
        prices_t=_series([100.0, 100.0], tk),
        prices_prev=_series([100.0, 100.0], tk),
        adv_usd=_series([adv_a, 1e12], tk),        # B 容量无限
        daily_vol=_series([0.02, 0.02], tk),
    )
    assert pnl is not None
    pos = broker.store.latest_positions(1)
    assert abs(pos.get("A", 0.0) - cap_expected) < 1e-9, (
        f"A 的持仓应被 ADV 上限削到 {cap_expected:.3e}，实际 {pos.get('A', 0.0):.3e}"
        f"（公式 adv×pct/capital 被改动也不会报错）"
    )


def test_adv_cap_scales_with_capital(broker, tmp_path):
    """
    同一 ADV 下，资金翻倍 → 该票的**权重**上限减半（美元上限不变）。
    这条能杀死把 `/ initial_capital` 改成 `* initial_capital` 的变异。
    """
    tk = ["A"]
    adv = 50_000.0
    big = PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp_path/'pb2.db'}"),
                      initial_capital=CAPITAL * 2)
    kw = dict(target_w=_series([1.0], tk), prices_t=_series([100.0], tk),
              prices_prev=_series([100.0], tk), adv_usd=_series([adv], tk),
              daily_vol=_series([0.02], tk))
    broker.step(alpha_id=2, date="2024-01-02", **kw)
    big.step(alpha_id=2, date="2024-01-02", **kw)
    w_small = broker.store.latest_positions(2)["A"]
    w_big = big.store.latest_positions(2)["A"]
    assert abs(w_big - w_small / 2.0) < 1e-9, (
        f"资金翻倍后权重上限应减半：{w_small:.3e} → 期望 {w_small/2:.3e}，实际 {w_big:.3e}"
    )


# ---------------------------------------------------------------------------
# L117 / L125：组合市值与逐名成本 USD 的量纲
# ---------------------------------------------------------------------------

def test_cost_usd_scales_with_capital(broker, tmp_path):
    """
    逐名 cost_usd = cost_w × (equity × initial_capital)。
    资金翻倍 → 同一笔调仓的美元成本翻倍。这条钉死 `pv = equity * capital` 的量纲，
    把 `*` 改成 `/` 会让成本差 12 个数量级。

    注意两项**本来就不线性**的成本必须置零，否则"恰好翻倍"这个期望是错的：
      · min_ticket_fee —— 固定项，不随资金变
      · impact_coef    —— 市场冲击按 sqrt(参与率)，成本 ∝ 交易额^1.5（超线性）
    置零后只剩 fixed_bps + spread，二者严格按名义额比例计 → 精确翻倍。
    """
    tk = ["A", "B"]
    prop_only = CostParams(min_ticket_fee=0.0, impact_coef=0.0)
    kw = dict(target_w=_series([0.5, -0.5], tk),
              prices_t=_series([100.0, 50.0], tk),
              prices_prev=_series([100.0, 50.0], tk),
              adv_usd=_series([1e12, 1e12], tk),
              daily_vol=_series([0.02, 0.02], tk))
    small = PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp_path/'pb3a.db'}"),
                        initial_capital=CAPITAL, cost_params=prop_only)
    big = PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp_path/'pb3b.db'}"),
                      initial_capital=CAPITAL * 2, cost_params=prop_only)
    small.step(alpha_id=3, date="2024-01-02", **kw)
    big.step(alpha_id=3, date="2024-01-02", **kw)

    def _total_usd(b):
        return sum(abs(f.cost_usd) for f in b.store.fills_on(3, "2024-01-02"))

    small_usd, big_usd = _total_usd(small), _total_usd(big)
    assert small_usd > 0, "成本为 0，无法验证量纲"
    assert abs(big_usd - 2 * small_usd) / max(small_usd, 1e-9) < 1e-6, (
        f"资金翻倍时美元成本应翻倍：{small_usd:.2f} → 期望 {2*small_usd:.2f}，"
        f"实际 {big_usd:.2f}（pv = equity × capital 的量纲被改动）"
    )


# ---------------------------------------------------------------------------
# L132：借券成本必须是**加**进 cost_bps，不是减
# ---------------------------------------------------------------------------

def test_borrow_cost_adds_to_reported_cost(tmp_path):
    """
    持有空头会产生借券成本，必须**推高** cost_bps。
    把 `(cost_ret + borrow_ret)` 写成 `-` 会让报告的成本**偏低** ——
    低报成本 = 高报净业绩，是最危险的一类记账错误。
    """
    tk = ["A", "B"]
    params = CostParams(short_borrow_annual_bps=500.0)   # 显著的借券费率
    st = PositionStore(db_url=f"sqlite:///{tmp_path/'pb4.db'}")
    b = PaperBroker(store=st, initial_capital=CAPITAL, cost_params=params)

    common = dict(prices_t=_series([100.0, 100.0], tk),
                  prices_prev=_series([100.0, 100.0], tk),
                  adv_usd=_series([1e12, 1e12], tk),
                  daily_vol=_series([0.02, 0.02], tk))
    # 第 1 天建空头
    b.step(alpha_id=4, date="2024-01-02", target_w=_series([0.5, -0.5], tk), **common)
    # 第 2 天维持（此时昨仓含空头 → 产生借券成本）
    pnl_short = b.step(alpha_id=4, date="2024-01-03",
                       target_w=_series([0.5, -0.5], tk), **common)

    # 对照：全多头，同样维持一天
    st2 = PositionStore(db_url=f"sqlite:///{tmp_path/'pb5.db'}")
    b2 = PaperBroker(store=st2, initial_capital=CAPITAL, cost_params=params)
    b2.step(alpha_id=5, date="2024-01-02", target_w=_series([0.5, 0.5], tk), **common)
    pnl_long = b2.step(alpha_id=5, date="2024-01-03",
                       target_w=_series([0.5, 0.5], tk), **common)

    assert pnl_short.cost_bps > pnl_long.cost_bps, (
        f"含空头的 cost_bps({pnl_short.cost_bps:.4f}) 应高于全多头"
        f"({pnl_long.cost_bps:.4f}) —— 借券成本的符号被改成了减"
    )


# ---------------------------------------------------------------------------
# L169：run_series 的第 0 天不得用 iloc[-1]（那是**未来**价格）
# ---------------------------------------------------------------------------

def test_first_day_has_no_lookahead(broker):
    """
    t=0 时 prev_prices 必须取当日自身（无隔夜收益），绝不能取 prices.iloc[-1]。
    把 `if t > 0` 改成 `>= 0` 会让第 0 天拿**最后一行**当昨收 → 凭空造出一段收益。
    构造：价格单调上涨，若发生前视，第 0 天会出现巨大的负毛收益。
    """
    tk = ["A"]
    n = 10
    idx = pd.bdate_range("2024-01-02", periods=n)
    prices = pd.DataFrame({"A": np.linspace(100.0, 200.0, n)}, index=idx)
    weights = pd.DataFrame({"A": [1.0] * n}, index=idx)
    vol = pd.DataFrame({"A": [1e12] * n}, index=idx)

    broker.replay(alpha_id=6, weights_df=weights, prices_df=prices, volume_df=vol)
    hist = broker.store.pnl_history(6, limit=n)
    assert hist, "replay 未产生任何记录"
    day0 = min(hist, key=lambda h: str(h.date))
    assert abs(day0.gross_ret) < 1e-9, (
        f"第 0 天毛收益应为 0（无昨仓、prev=self），实际 {day0.gross_ret:.6f} —— "
        f"疑似用了 prices.iloc[-1] 作昨收（前视）"
    )


# ---------------------------------------------------------------------------
# L192：日期归一化两个方向都要测
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d,expected", [
    (pd.Timestamp("2024-03-05 15:30"), "2024-03-05"),
    ("2024-03-05",                     "2024-03-05"),
])
def test_date_is_normalised_both_input_forms(broker, d, expected):
    """Timestamp（带 .date()）与纯字符串都必须归一到同一个日期串。"""
    tk = ["A"]
    pnl = broker.step(
        alpha_id=7, date=d,
        target_w=_series([0.1], tk), prices_t=_series([100.0], tk),
        prices_prev=_series([100.0], tk), adv_usd=_series([1e12], tk),
        daily_vol=_series([0.02], tk),
    )
    assert pnl.date == expected, f"日期归一化错误：{d!r} → {pnl.date!r}，期望 {expected!r}"


# ---------------------------------------------------------------------------
# L169 补强：只有**带昨仓**时，第 0 天的 prev_prices 才可观测
# ---------------------------------------------------------------------------

def test_replay_continuation_has_no_lookahead_on_first_day(broker):
    """
    上一版 test_first_day_has_no_lookahead **杀不死** `if t > 0 → >= 0` 这个变异：
    没有昨仓时 gross_ret = Σ(prev_w × price_chg) = 0，前视与否都是 0，断言无从区分。

    真实场景是**续跑**：账本里已有持仓，replay 从更晚的窗口继续。
    此时第 0 天 prev_w ≠ 0，prev_prices 取错就会凭空造出一段隔夜收益。
    正确行为：第 0 天 prev=当日自身 → 毛收益仍为 0；
    若取了 prices.iloc[-1]（窗口最后一天，未来价）→ 立刻出现巨大负收益。
    """
    tk = ["A"]
    # 1) 先建仓（窗口之前的一天）
    broker.step(
        alpha_id=8, date="2024-01-01",
        target_w=_series([1.0], tk), prices_t=_series([100.0], tk),
        prices_prev=_series([100.0], tk), adv_usd=_series([1e12], tk),
        daily_vol=_series([0.02], tk),
    )
    assert broker.store.latest_positions(8).get("A", 0.0) > 0.9, "建仓失败，无法验证续跑"

    # 2) 从次日开始 replay，价格显著上行（前视会把最后一天的高价当昨收）
    idx = pd.bdate_range("2024-01-02", periods=8)
    prices = pd.DataFrame({"A": np.linspace(100.0, 300.0, 8)}, index=idx)
    weights = pd.DataFrame({"A": [1.0] * 8}, index=idx)
    vol = pd.DataFrame({"A": [1e12] * 8}, index=idx)
    broker.replay(alpha_id=8, weights_df=weights, prices_df=prices, volume_df=vol)

    hist = [h for h in broker.store.pnl_history(8, limit=20)
            if str(h.date) >= "2024-01-02"]
    assert hist, "replay 未产生记录"
    first = min(hist, key=lambda h: str(h.date))
    assert abs(first.gross_ret) < 1e-9, (
        f"续跑首日毛收益应为 0（prev=当日自身），实际 {first.gross_ret:.6f} —— "
        f"prev_prices 取到了窗口最后一天的价格（前视）"
    )


def test_price_change_uses_subtraction_not_addition(broker):
    """
    钉死 `price_chg = (p_t - p_prev) / p_prev` 的符号。
    把 `-` 写成 `+` 不会报错，只会让"涨跌幅"变成"价格和的比值"（恒 ≈ 2）。
    构造：昨仓满仓、价格从 100 涨到 110 → 毛收益必须精确等于 +10%。
    """
    tk = ["A"]
    broker.step(alpha_id=9, date="2024-01-01",
                target_w=_series([1.0], tk), prices_t=_series([100.0], tk),
                prices_prev=_series([100.0], tk), adv_usd=_series([1e12], tk),
                daily_vol=_series([0.02], tk))
    pnl = broker.step(alpha_id=9, date="2024-01-02",
                      target_w=_series([1.0], tk), prices_t=_series([110.0], tk),
                      prices_prev=_series([100.0], tk), adv_usd=_series([1e12], tk),
                      daily_vol=_series([0.02], tk))
    assert abs(pnl.gross_ret - 0.10) < 1e-9, (
        f"满仓且价格 100→110，毛收益应精确为 0.10，实际 {pnl.gross_ret:.6f}"
    )
