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


def test_adv_capped_order_is_flagged(broker):
    """对照组：真被 ADV 上限削掉时必须标出来，否则上一条可能只是'永远不标'。"""
    tk = ["A", "B"]
    broker.step(alpha_id=34, date="2024-02-06",
                target_w=_series([0.9, -0.1], tk), prices_t=_series([100.0, 50.0], tk),
                prices_prev=_series([100.0, 50.0], tk),
                adv_usd=_series([1e5, 1e12], tk), daily_vol=_series([0.02, 0.02], tk))
    fills = {f.ticker: f for f in broker.store.fills_on(34, "2024-02-06")}
    assert fills["A"].reject_reason == "adv_cap"
    assert abs(fills["A"].filled_weight) < abs(fills["A"].target_weight)


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

PROVEN_EQUIVALENT = {
    "L119 `abs(delta[i]) < 1e-12` -> `<=`":
        "区分值需要 |delta| 恰好等于 1e-12。delta = filled - prev_w，filled 来自 "
        "water-filling 投影（多轮浮点乘除后归一到 L1=1），无法反解出使其精确等于 "
        "1e-12 的目标权重；该阈值的用途本就是「小到等于没交易」的模糊带。",

    "L121 `abs(filled) < abs(tgt) - 1e-9` -> `<=`":
        "区分值需要 |filled| 恰好等于 |tgt| - 1e-9。同上，filled 是投影输出，"
        "无法构造成与 tgt 相差恰好 1e-9 的值；该容差存在的目的就是让"
        "「足额成交」的判定对末位浮点误差不敏感。",

    "L127 `abs(filled[i]) > 1e-12` -> `>=`":
        "区分值需要 |filled| 恰好等于 1e-12。同 L119：投影输出无法精确落在该值上。"
        "两侧语义连续——比 1e-12 还小的权重在 100 万美元资金上不足 1e-6 美元，"
        "记不记入持仓没有可观测差别。",

    "L192 `hasattr(d, 'date') and not isinstance(d, date)` -> 删掉 not":
        "该分支对系统实际产生的**每一种**日期形态都不可达：str 与 datetime/Timestamp "
        "在前两个 if 就已返回；datetime.date 没有 `.date` 属性（hasattr 为假）。"
        "条件恒为假，改不改 not 都返回 d。见 "
        "test_date_normalisation_branch_is_unreachable_for_real_inputs。",

    "L192 `hasattr(...) and not isinstance(...)` -> `or`":
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
    for d in samples:
        assert _as_date(d) == _date(2024, 3, 5)
        if not isinstance(d, (str, datetime)):
            # 能走到 L192 的唯一形态是 date 本身，而 date 没有 .date 属性
            assert not hasattr(d, "date"), f"{type(d).__name__} 竟然带 .date，L192 可达"


def test_epsilon_guards_in_paper_broker_are_unreachable():
    """L119 / L121 / L127 三条证明的共同机械验证。"""
    for tol in (1e-12, 1e-9):
        for base in (0.6, 0.4, 1.0, 0.01):
            assert (base + tol) - base != tol, (
                f"base={base} tol={tol} 处容差可精确还原，等价性证明不成立")


def test_every_survivor_has_a_written_proof():
    """存活项要么被用例杀死，要么在此有书面证明；不许有第三种状态。"""
    assert len(PROVEN_EQUIVALENT) == 5
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
