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


# ===========================================================================
# 第二轮变异测量（工具修好算术变异器之后）：26 个变异点存活 9 个。
#
# 上一轮只有 22 个点，且 `*` `+` `-` 三个算术变异器当时**静默失效**，
# 所以文件头注释里写的"L96 容量上限公式""L117 组合市值"其实
# **从来没有被真正验证过** —— 只是注释这么写。这一轮才是真的。
# ===========================================================================

def test_zero_initial_capital_falls_back_to_no_cap_not_nan(tmp_path):
    """
    `if self.initial_capital > 0: cap_w = adv * pct / initial_capital`。
    改成 `>=` 后 initial_capital==0 会走进除零：
      - adv > 0  → inf（与正确分支的 np.inf 同值，看不出来）
      - adv == 0 → **nan**（0/0）→ 投影全 nan → `abs(nan) > 1e-12` 为假
                   → **持仓整片消失**。
    所以必须 initial_capital==0 **且** adv==0 才能区分，只测前者不够。
    """
    from app.db.position_store import PositionStore
    b = PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp_path/'zero.db'}"),
                    initial_capital=0.0)
    tk = ["A", "B"]
    b.step(alpha_id=31, date="2024-01-02",
           target_w=_series([0.6, -0.4], tk), prices_t=_series([100.0, 50.0], tk),
           prices_prev=_series([100.0, 50.0], tk), adv_usd=_series([0.0, 0.0], tk),
           daily_vol=_series([0.02, 0.02], tk))
    pos = b.store.latest_positions(31)
    assert pos, "initial_capital==0 且 adv==0 时持仓整片消失（cap 被算成 NaN）"
    assert pos["A"] == pytest.approx(0.6, abs=1e-12)
    assert pos["B"] == pytest.approx(-0.4, abs=1e-12)


def test_unchanged_position_still_appears_in_fills(broker):
    """
    `if abs(delta[i]) < 1e-12 and abs(filled[i]) < 1e-12: continue`
    —— **两个条件同时成立**才跳过（既没交易、也没持仓）。
    改成 `or` 后，"持仓不变但仍有仓位"的名字会被整条丢出 fills，
    审计时看不到自己还拿着什么。

    构造：连续两天下同一个目标权重 → 第二天 delta≈0 而 filled≠0。
    """
    tk = ["A", "B"]
    for d in ("2024-02-01", "2024-02-02"):
        broker.step(alpha_id=32, date=d,
                    target_w=_series([0.6, -0.4], tk), prices_t=_series([100.0, 50.0], tk),
                    prices_prev=_series([100.0, 50.0], tk),
                    adv_usd=_series([1e12, 1e12], tk), daily_vol=_series([0.02, 0.02], tk))
    fills = {f.ticker: f for f in broker.store.fills_on(32, "2024-02-02")}
    assert set(fills) == {"A", "B"}, (
        f"持仓未变的名字从 fills 里消失了：{sorted(fills)}")
    assert fills["A"].filled_weight == pytest.approx(0.6, abs=1e-12)
    assert fills["B"].filled_weight == pytest.approx(-0.4, abs=1e-12)


def test_fully_filled_order_is_not_flagged_as_adv_capped(broker):
    """
    `reject = "adv_cap" if abs(filled) < abs(tgt) - 1e-9 else ""`
    —— 那个 `- 1e-9` 是容差；写成 `+ 1e-9` 会让**足额成交**的订单
    （filled == tgt）被误标成"被 ADV 上限拒绝"，审计结论完全反了。
    """
    tk = ["A", "B"]
    broker.step(alpha_id=33, date="2024-02-05",
                target_w=_series([0.6, -0.4], tk), prices_t=_series([100.0, 50.0], tk),
                prices_prev=_series([100.0, 50.0], tk),
                adv_usd=_series([1e12, 1e12], tk), daily_vol=_series([0.02, 0.02], tk))
    fills = {f.ticker: f for f in broker.store.fills_on(33, "2024-02-05")}
    assert fills["A"].reject_reason == "", "足额成交却被标成 adv_cap"
    assert fills["B"].reject_reason == ""


def test_participation_capped_order_is_flagged_and_booked_as_unfilled(broker):
    """
    对照组：真被**单日成交量**上限削掉时必须标出来，否则上一条可能只是"永远不标"。

    2026-09-19 的口径变更（用户决策 #2/#3）：

      · `reject_reason` 从 `adv_cap` 改成 `participation` —— 约束的是**成交量**，
        不是持仓容量，旧名字把两个概念混在一起
      · 未成交的部分必须**如实记账**：`traded + unfilled == desired`，
        而不是只留一个成交后的持仓数字（那看不出这笔单成交了多少）
    """
    tk = ["A", "B"]
    broker.step(alpha_id=34, date="2024-02-06",
                target_w=_series([0.9, -0.1], tk), prices_t=_series([100.0, 50.0], tk),
                prices_prev=_series([100.0, 50.0], tk),
                adv_usd=_series([1e5, 1e12], tk), daily_vol=_series([0.02, 0.02], tk))
    fills = {f.ticker: f for f in broker.store.fills_on(34, "2024-02-06")}
    a = fills["A"]
    assert a.reject_reason == "participation"
    assert abs(a.filled_weight) < abs(a.target_weight)
    # 未成交量如实记账：想要 0.9（昨仓 0）→ 成交 + 未成交 == 0.9
    assert a.traded_weight + a.unfilled_weight == pytest.approx(0.9, abs=1e-12), (
        f"成交 {a.traded_weight} + 未成交 {a.unfilled_weight} 对不上意图 0.9")
    assert abs(a.unfilled_weight) > 1e-9, "被限流却没有记下未成交量"


def test_a_constrained_name_does_not_enlarge_the_others(broker):
    """
    **用户决策 #1 的核心断言**：执行层不得因为一个标的受限，就擅自扩大另一个标的。

    A 想要 90% 但当日成交量上限只允许 1%，B 想要 -10%。
    旧实现用 water-filling 把 89% 的亏空摊给 B → 落账 [0.01, -0.99]，
    10% 的对冲腿变成 99% 的方向性空头。
    """
    tk = ["A", "B"]
    cap_pct = broker.params.max_participation_pct
    adv_a = 0.01 * CAPITAL / cap_pct              # A 当日最多成交 1% 权重
    broker.step(alpha_id=36, date="2024-02-07",
                target_w=_series([0.9, -0.1], tk),
                prices_t=_series([100.0, 100.0], tk),
                prices_prev=_series([100.0, 100.0], tk),
                adv_usd=_series([adv_a, 1e15], tk),
                daily_vol=_series([0.02, 0.02], tk))
    pos = broker.store.latest_positions(36)
    assert pos.get("A", 0.0) == pytest.approx(0.01, abs=1e-9), (
        f"A 应被成交量上限削到 1%，实际 {pos.get('A', 0.0)}")
    assert pos.get("B", 0.0) == pytest.approx(-0.10, abs=1e-9), (
        f"B 的目标是 -10%，落账 {pos.get('B', 0.0):.4f} —— "
        f"A 的未成交额度被摊到了 B 头上")


def test_a_reduction_that_cannot_fill_is_booked_at_its_true_state(broker):
    """
    **用户决策 #2 的后半**：已有持仓的减仓/平仓失败，要按真实未成交状态记账。

    第 1 天建 20% 多头（成交量充足）；第 2 天想清零，但当日成交量只够 5%。
    落账必须是"还剩 15%"，且未成交量记为 -0.15 —— 不能假装已清仓。
    """
    tk = ["A"]
    cap_pct = broker.params.max_participation_pct
    broker.step(alpha_id=37, date="2024-02-08",
                target_w=_series([0.20], tk), prices_t=_series([100.0], tk),
                prices_prev=_series([100.0], tk), adv_usd=_series([1e15], tk),
                daily_vol=_series([0.02], tk))
    assert broker.store.latest_positions(37)["A"] == pytest.approx(0.20, abs=1e-12)

    adv_small = 0.05 * CAPITAL / cap_pct          # 当日最多成交 5% 权重
    broker.step(alpha_id=37, date="2024-02-09",
                target_w=_series([0.0], tk), prices_t=_series([100.0], tk),
                prices_prev=_series([100.0], tk), adv_usd=_series([adv_small], tk),
                daily_vol=_series([0.02], tk))
    pos = broker.store.latest_positions(37)
    assert pos.get("A", 0.0) == pytest.approx(0.15, abs=1e-9), (
        f"只成交了 5%，应还剩 15%，实际 {pos.get('A', 0.0)} —— 平仓失败被当成了已平")
    f = {x.ticker: x for x in broker.store.fills_on(37, "2024-02-09")}["A"]
    assert f.traded_weight == pytest.approx(-0.05, abs=1e-9)
    assert f.unfilled_weight == pytest.approx(-0.15, abs=1e-9), (
        f"未成交的减仓量记成了 {f.unfilled_weight}，应为 -0.15")


def test_cost_bps_is_consistent_with_gross_minus_net(broker):
    """
    `cost_bps = (cost_ret + borrow_ret) * 1e4`，而 `net = gross - cost - borrow`。
    两者必须自洽：cost_bps == (gross_ret - net_ret) * 1e4。
    把 `* 1e4` 写成 `/ 1e4` 只改量纲（差 1e8 倍），符号与相对大小都不变，
    任何"成本为正""净收益低于毛收益"的断言都抓不住它。
    """
    tk = ["A", "B"]
    broker.step(alpha_id=35, date="2024-03-01",
                target_w=_series([0.6, -0.4], tk), prices_t=_series([100.0, 50.0], tk),
                prices_prev=_series([100.0, 50.0], tk),
                adv_usd=_series([1e12, 1e12], tk), daily_vol=_series([0.02, 0.02], tk))
    pnl = broker.step(alpha_id=35, date="2024-03-04",
                      target_w=_series([0.6, -0.4], tk), prices_t=_series([101.0, 50.0], tk),
                      prices_prev=_series([100.0, 50.0], tk),
                      adv_usd=_series([1e12, 1e12], tk), daily_vol=_series([0.02, 0.02], tk))
    assert pnl.cost_bps == pytest.approx((pnl.gross_ret - pnl.net_ret) * 1e4, rel=1e-6)
    assert pnl.cost_bps > 0.0, "持有空头仓位，借券成本不可能为零"


def test_borrow_cost_is_charged_on_shorts_only(broker):
    """
    `borrow_ret = sum(max(-prev_w, 0)) * daily_borrow` 只对**昨仓空头**计费。
    同等条件下多空组合的成本必须严格高于纯多头，否则借券那一项形同虚设。
    """
    tk = ["A", "B"]
    common = dict(prices_t=_series([100.0, 50.0], tk),
                  prices_prev=_series([100.0, 50.0], tk),
                  adv_usd=_series([1e12, 1e12], tk),
                  daily_vol=_series([0.02, 0.02], tk))
    for aid, w in ((36, [0.6, -0.4]), (37, [0.6, 0.4])):
        for d in ("2024-03-01", "2024-03-04"):
            broker.step(alpha_id=aid, date=d, target_w=_series(w, tk), **common)
    pick = lambda aid: [p.cost_bps for p in broker.store.pnl_history(aid, limit=9)
                        if str(p.date) == "2024-03-04"][0]
    s, l = pick(36), pick(37)
    assert s > l, f"多空组合的成本没有高于纯多头（借券费没生效）：short={s} long={l}"


# ---------------------------------------------------------------------------
# 存活变异的等价性证明
# ---------------------------------------------------------------------------

#: **已被反例推翻的"等价性证明"** —— 留在这里是为了不让它们被重新写回去。
#:
#: 外部审计 2026-09-15 给出的反例：`project_to_capped_l1` 对
#: `[1e-12, 1-1e-12]` 的输出**逐位保留**了 1e-12（L1 已等于 target、上限不绑定时
#: 投影是恒等映射）。于是 filled[i] 恰好等于 1e-12 是**可构造的**，
#: 原证明"投影输出无法精确落在该值上"不成立。
#:
#: 我当初的"机械验证"断言的是 `(base + tol) - base != tol`，
#: 也就是"tol 不能由一次加法还原" —— 而到达 filled[i] 的值根本不必来自加法。
#: **证明了一个更弱的命题，然后当成结论用了。**
#: 这两个变异点现在的状态是：既未被杀死，也无有效证明。
REFUTED_EQUIVALENCE = {
    "L119 `abs(delta[i]) < 1e-12` -> `<=`":
        "反例：prev_w=0、filled=1e-12 时 delta 恰为 1e-12，`<` 判假而 `<=` 判真。"
        "见 test_the_epsilon_guards_are_reachable_so_the_old_proof_is_void。",

    "L127 `abs(filled[i]) > 1e-12` -> `>=`":
        "反例：project_to_capped_l1([[1e-12, 1-1e-12]], cap=inf) 的第一项逐位等于 "
        "1e-12，`>` 判假而 `>=` 判真 —— 前者不记入持仓，后者记入。"
        "见 test_the_epsilon_guards_are_reachable_so_the_old_proof_is_void。",
}

PROVEN_EQUIVALENT = {
    "app/core/execution/paper_broker.py ×1 — L121 `abs(filled) < abs(tgt) - 1e-9` -> `<=`":
        "区分值需要 |filled| 恰好等于 |tgt| - 1e-9。filled 是投影输出，"
        "无法构造成与 tgt 相差恰好 1e-9 的值；该容差存在的目的就是让"
        "「足额成交」的判定对末位浮点误差不敏感。"
        "（注意：L119/L127 的同型证明已被反例推翻，见 REFUTED_EQUIVALENCE；"
        "这一条与它们的区别是区分值要求的是**两个量之差**恰好等于容差，"
        "不是某个量本身恰好等于容差，投影的恒等路径给不出这种构造。）",

    "app/core/execution/paper_broker.py ×1 — L192 `hasattr(d, 'date') and not isinstance(d, date)` -> 删掉 not":
        "该分支对系统实际产生的**每一种**日期形态都不可达：str 与 datetime/Timestamp "
        "在前两个 if 就已返回；datetime.date 没有 `.date` 属性（hasattr 为假）。"
        "条件恒为假，改不改 not 都返回 d。见 "
        "test_date_normalisation_branch_is_unreachable_for_real_inputs。",

    "app/core/execution/paper_broker.py ×1 — L192 `hasattr(...) and not isinstance(...)` -> `or`":
        "同上：对四种实际输入形态条件恒为假。date 对象走 hasattr 假 + not isinstance 假，"
        "or 之后仍是假；其余三种在更早的分支已返回，根本走不到这一行。",
}


def test_date_normalisation_branch_is_unreachable_for_real_inputs():
    """
    L192 等价性的机械验证：枚举系统实际会传进 `_as_date` 的四种形态，
    逐一确认它们要么在更早的分支返回、要么让 L192 的条件为假。
    """
    from datetime import date as _date, datetime
    from app.core.execution.paper_broker import _as_date

    samples = ["2024-03-05", datetime(2024, 3, 5, 15, 30),
               pd.Timestamp("2024-03-05 15:30"), _date(2024, 3, 5)]
    # 前两个 if 会拦下 str 与 datetime（Timestamp 是 datetime 子类）；
    # 只有剩下的形态才真的走到 L192。把"谁走到了"数出来再断言，
    # 而不是写 `if 走到了: assert ...` —— 那样一旦没人走到就什么都没查（§A）。
    reached = []
    for d in samples:
        assert _as_date(d) == _date(2024, 3, 5)
        if not isinstance(d, (str, datetime)):
            reached.append(d)
    assert reached, "没有任何样本走到 L192，本用例证明不了它不可达"
    for d in reached:
        # 能走到 L192 的形态里，date 没有 `.date` 属性 → 条件恒为假
        assert not hasattr(d, "date"), f"{type(d).__name__} 竟然带 .date，L192 可达"


class TestGrossExposureFollowsTheTarget:
    """
    缺陷 **A-6 的前半**（2026-09-18 修）：`project_to_capped_l1(..., target=1.0)`
    的字面量写死，不看 `tgt` 实际要多少总敞口。

    上游每一个**降敞口**的决定都会在这一步被抹掉：
    `risk_target_vol_ann` 把 gross 缩到 0.5、`max_gross` 拦到 0.6、
    无交易带压掉换手 —— 落到券商全部被拉回 1.0，**实际下的单是意图的两倍**。

    这一组断言的是"上游要多少就下多少"。ADV 上限不绑定时（本组都给了极大 ADV），
    投影是恒等映射，填单权重必须**逐位**等于目标。
    """

    @staticmethod
    def _fill(broker, weights, alpha_id):
        tk = [f"T{i}" for i in range(len(weights))]
        broker.step(
            alpha_id=alpha_id, date="2024-01-02",
            target_w=_series(weights, tk),
            prices_t=_series([100.0] * len(tk), tk),
            prices_prev=_series([100.0] * len(tk), tk),
            adv_usd=_series([1e15] * len(tk), tk),      # 上限不绑定
            daily_vol=_series([0.02] * len(tk), tk),
        )
        return broker.store.latest_positions(alpha_id)

    @pytest.mark.parametrize("weights,gross", [
        ([0.25, 0.25], 0.5),        # 波动率目标把敞口缩了一半
        ([0.3, -0.3], 0.6),         # 多空各 30%
        ([0.1, 0.05, -0.05], 0.2),  # 明显低敞口
    ])
    def test_the_filled_gross_equals_the_requested_gross(self, broker, weights, gross):
        pos = self._fill(broker, weights, alpha_id=hash(tuple(weights)) % 10000)
        got = sum(abs(v) for v in pos.values())
        assert got == pytest.approx(gross, abs=1e-12), (
            f"目标总敞口 {gross}，落账 {got:.6f} —— "
            f"写死 target=1.0 会把它拉回 1.0，上游的降敞口决定被抹掉")

    def test_each_name_is_filled_exactly_as_requested(self, broker):
        """不只总量对，**逐名**都要对 —— 总量对而分布被改写同样是错的。"""
        w = [0.25, 0.25]
        pos = self._fill(broker, w, alpha_id=4242)
        assert [pos["T0"], pos["T1"]] == pytest.approx(w, abs=1e-12), (
            f"逐名填单与目标不符：{pos}")

    def test_a_unit_gross_target_is_unchanged(self, broker):
        """
        回归保护：上游给 L1=1 时新旧行为必须**逐位相同** ——
        `test_paper_broker_replay_parity.py::test_replay_matches_backtest_engine`
        的 1e-9 对账依赖这一点。
        """
        pos = self._fill(broker, [0.6, -0.4], alpha_id=4243)
        assert sum(abs(v) for v in pos.values()) == pytest.approx(1.0, abs=1e-12)


class TestSubEpsilonPositionsAreADeliberatePolicy:
    """
    L119 / L127 的两个 epsilon 守卫**可达**（见下面那条反例用例），所以它们不是
    等价变异，必须由断言杀死。

    这里钉住的是**当前的处置政策**，并且这个政策是有意的：

      - 权重恰好 1e-12 的名字 **不进持仓**（L127 用 `>`）——
        100 万美元资金上折合 1e-6 美元，记进持仓只会在账本里留噪声
      - 但它 **要留一条成交记录**（L119 的 `and` 短路）——
        因为确实发生了一次（极小的）目标变动，审计链路不该凭空少一行

    两条合起来的意思是"小到不计入仓位，但不假装没发生过"。
    改成 `>=` 或 `<=` 会分别翻转这两个决定，下面两条断言各杀一个。

    构造：`project_to_capped_l1` 在 L1 已等于 target、ADV 上限不绑定时是恒等映射，
    于是目标权重里的 1e-12 **逐位**原样出现在 `filled` 里。
    """

    @staticmethod
    def _run(broker):
        tk = ["A", "B"]
        broker.step(
            alpha_id=77, date="2024-01-02",
            target_w=_series([1e-12, 1.0 - 1e-12], tk),
            prices_t=_series([100.0, 100.0], tk),
            prices_prev=_series([100.0, 100.0], tk),
            adv_usd=_series([1e15, 1e15], tk),          # 上限远大于目标 → 不绑定
            daily_vol=_series([0.02, 0.02], tk),
        )
        return (broker.store.latest_positions(77),
                broker.store.fills_on(77, "2024-01-02"))

    def test_a_position_of_exactly_one_picoweight_is_not_recorded(self, broker):
        """L127 `abs(filled[i]) > 1e-12` —— 改成 `>=` 会把这个名字记进持仓。"""
        pos, _ = self._run(broker)
        assert "A" not in pos, (
            f"权重恰好 1e-12 的名字进了持仓：{ {k: repr(v) for k, v in pos.items()} } —— "
            f"L127 的守卫被放宽成 `>=` 了（100 万美元上折合 1e-6 美元的噪声仓位）")
        assert pos.get("B") == pytest.approx(1.0 - 1e-12, abs=1e-15), (
            "另一腿的权重被改动了，说明构造没有落在预期的那一格上")

    def test_that_same_picoweight_still_produces_a_fill_record(self, broker):
        """
        L119 `abs(delta[i]) < 1e-12 and abs(filled[i]) < 1e-12: continue`
        —— 首日 prev_w=0，delta 恰好等于 1e-12，`<` 判假使 `and` 短路，成交记录保留。
        改成 `<=` 后两侧都成立 → `continue` → A 的成交记录消失。
        """
        _, fills = self._run(broker)
        names = sorted(f.ticker for f in fills)
        assert "A" in names, (
            f"权重 1e-12 的调仓没有留下成交记录（本日成交：{names}）—— "
            f"L119 的守卫被放宽成 `<=` 了：仓位不记、成交也不记，"
            f"这次目标变动在账本里彻底消失")

    #: `nextafter(1e-12, +inf)` —— 比阈值**大一个 ulp**，所以 L127 会把它存进持仓。
    #: 这是能让 `prev_w` 落在"非 0 且 |v| 只比 1e-12 大一点点"的唯一办法。
    _PREV = float(np.nextafter(1e-12, np.inf))
    #: 选它是因为 `_FILL - _PREV` 在浮点上**恰好**等于 -1e-12（穷举搜出来的，不是估的），
    #: 同时 `|_FILL| < 1e-12` 成立 —— 两者必须同时满足才能区分 L119 的第一个比较符。
    _FILL = 2.0194839173657902e-28

    def test_a_pico_trade_off_a_pico_position_still_leaves_a_fill_record(self, broker):
        """
        L119 第一个比较符 `abs(delta[i]) < 1e-12` → `<=`。

        **为什么需要这么刁钻的构造**：`and` 会短路。第一天用 1e-12 建仓时
        `delta == filled`，两个条件永远同真同假，改任何一个都被另一个挡住 ——
        我第一版就是这么写的，`verify_mutant` 当场判 **[X] 变异存活**。

        真正能区分的那一格要求：`|delta|` **恰好** 1e-12，而 `|filled|` 严格小于它。
        由 `delta = filled - prev_w` 且 `prev_w` 只能取 0 或 `|v| > 1e-12`
        （L127 决定的），prev_w=0 时 delta≡filled 必然同真同假，
        所以只剩"prev_w 比阈值大一个 ulp"这一条路。穷举浮点找到了
        `prev = nextafter(1e-12, ∞)`、`filled = 2.0194839173657902e-28`
        这一对，`filled - prev` 逐位等于 `-1e-12`。

        改成 `<=` 之后两个条件同时成立 → `continue` → 这笔调仓在成交记录里消失。
        """
        tk = ["A", "B"]
        common = dict(prices_t=_series([100.0, 100.0], tk),
                      prices_prev=_series([100.0, 100.0], tk),
                      adv_usd=_series([1e15, 1e15], tk),
                      daily_vol=_series([0.02, 0.02], tk))
        broker.step(alpha_id=78, date="2024-01-02",
                    target_w=_series([self._PREV, 1.0 - self._PREV], tk), **common)
        prev = broker.store.latest_positions(78).get("A")
        assert prev == self._PREV, (
            f"昨仓没有逐位保留 {self._PREV!r}（实际 {prev!r}）—— "
            f"构造前提变了，本用例区分不了 L119")
        assert (self._FILL - self._PREV) == -1e-12, (
            "delta 不再逐位等于 -1e-12 —— 请重新穷举区分值，不要让这条用例空转")

        broker.step(alpha_id=78, date="2024-01-03",
                    target_w=_series([self._FILL, 1.0 - self._FILL], tk), **common)
        names = sorted(f.ticker for f in broker.store.fills_on(78, "2024-01-03"))
        assert "A" in names, (
            f"|delta| 恰好 1e-12、|filled| 小于 1e-12 的调仓没有留下成交记录"
            f"（本日成交：{names}）—— L119 的第一个守卫被放宽成 `<=` 了")

    def test_a_sub_ulp_trim_down_to_exactly_one_picoweight_still_records_a_fill(self, broker):
        """
        L119 **第二个**比较符 `abs(filled[i]) < 1e-12` → `<=`。

        和上一条是**互补**的一格，必须分开写：上一条要 `|delta|` 恰好 1e-12、
        `|filled|` 更小；这一条要反过来 —— `|filled|` **恰好** 1e-12，
        而 `|delta|` 严格小于它。

        构造：昨仓 = `nextafter(1e-12, ∞)`（比阈值大一个 ulp，所以存得下），
        今日目标 = 1e-12 整。于是 `delta` 只有一个 ulp，`filled` 正好压在阈值上。

          - 原式：`|delta| < 1e-12` 真、`|filled| < 1e-12` **假** → 不 continue → 有成交记录
          - 变异：两者皆真 → continue → 这笔"削掉一个 ulp"的调仓凭空消失

        `verify_mutant` 对 L119 的两个比较符各跑一次，两条都要判 [OK] 才算处置完。
        """
        tk = ["A", "B"]
        common = dict(prices_t=_series([100.0, 100.0], tk),
                      prices_prev=_series([100.0, 100.0], tk),
                      adv_usd=_series([1e15, 1e15], tk),
                      daily_vol=_series([0.02, 0.02], tk))
        broker.step(alpha_id=79, date="2024-01-02",
                    target_w=_series([self._PREV, 1.0 - self._PREV], tk), **common)
        prev = broker.store.latest_positions(79).get("A")
        assert prev == self._PREV, f"昨仓没有逐位保留 {self._PREV!r}（实际 {prev!r}）"

        delta = 1e-12 - self._PREV
        assert 0 < abs(delta) < 1e-12, (
            f"delta={delta!r} 不在 (0, 1e-12) 内 —— 这一格构造不出来，本用例会空转")

        broker.step(alpha_id=79, date="2024-01-03",
                    target_w=_series([1e-12, 1.0 - 1e-12], tk), **common)
        names = sorted(f.ticker for f in broker.store.fills_on(79, "2024-01-03"))
        assert "A" in names, (
            f"|filled| 恰好压在 1e-12 上、|delta| 只有一个 ulp 的调仓没有成交记录"
            f"（本日成交：{names}）—— L119 的第二个守卫被放宽成 `<=` 了")

    def test_the_two_guards_disagree_on_purpose(self, broker):
        """
        把上面两条的关系钉死：**同一个名字**、**同一天**，
        成交记录里有、持仓里没有。任何一侧被改宽都会破坏这个组合。
        """
        pos, fills = self._run(broker)
        assert ("A" in [f.ticker for f in fills]) and ("A" not in pos), (
            f"『记成交但不记仓位』这个组合被破坏了："
            f"成交={sorted(f.ticker for f in fills)} 持仓={sorted(pos)}")


def test_the_epsilon_guards_are_reachable_so_the_old_proof_is_void():
    """
    外部审计 2026-09-15 的反例，原样固化下来。

    旧证明写的是"投影输出无法精确落在 1e-12 上"，配的"机械验证"是
    `(base + tol) - base != tol` —— 那只证明了**tol 不能由一次加法还原**。
    可是到达 `filled[i]` 的值不必来自加法：`project_to_capped_l1` 在
    L1 已等于 target、上限不绑定时就是恒等映射，输入里的 1e-12 原样出来。

    这条用例现在断言的是**反例仍然成立**（守卫可达），
    所以谁也不能再把 L119/L127 写回 PROVEN_EQUIVALENT。
    """
    import numpy as np

    from app.core.backtest_engine.transaction_cost import project_to_capped_l1

    w = np.array([[1e-12, 1.0 - 1e-12]])
    cap = np.array([[np.inf, np.inf]])
    filled = np.asarray(project_to_capped_l1(w, cap), dtype=float)[0]

    assert filled[0] == 1e-12, (
        f"投影没有原样保留 1e-12（得到 {filled[0]!r}）—— 反例的构造前提变了，"
        f"请重新确认 L119/L127 的可达性，不要默认它们又变回等价")

    # L127：记不记入持仓
    assert bool(abs(filled[0]) > 1e-12) is False, "`>` 不再排除它"
    assert bool(abs(filled[0]) >= 1e-12) is True, "`>=` 不再纳入它"

    # L119：prev_w = 0 时 delta 恰为 1e-12
    delta = filled[0] - 0.0
    assert bool(abs(delta) < 1e-12) is False, "`<` 不再判它为『没交易』之外"
    assert bool(abs(delta) <= 1e-12) is True, "`<=` 不再把它归进容差带"


def test_every_survivor_has_a_written_proof():
    """
    存活项要么被用例杀死，要么有书面证明，要么**明确登记为未解决**；
    不许有第四种状态（"看起来有证明，其实证明是错的"就是第四种）。
    """
    assert len(PROVEN_EQUIVALENT) == 3, (
        f"证明条目数变成 {len(PROVEN_EQUIVALENT)}（原 5，其中 L119/L127 "
        f"两条已被反例推翻，移入 REFUTED_EQUIVALENCE）")
    assert len(REFUTED_EQUIVALENCE) == 2, "被推翻的条目不许悄悄消失"
    overlap = set(PROVEN_EQUIVALENT) & set(REFUTED_EQUIVALENCE)
    assert not overlap, f"同一个变异点既算已证明又算已推翻：{overlap}"
    for key, why in list(PROVEN_EQUIVALENT.items()) + list(REFUTED_EQUIVALENCE.items()):
        assert len(why) >= 40, f"{key} 的说明过于敷衍：{why!r}"
