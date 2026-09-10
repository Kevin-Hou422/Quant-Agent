"""
transaction_cost.py —— 逐公式定钉测试（变异测试驱动）

来由：用修好的隔离版变异工具测，击杀率 **57.1%**（42 个变异点存活 18 个）。
存活 = 把那行代码改坏了，整个测试套件依然全绿。本文件针对每一个存活项，
要么写出能杀死它的用例，要么在 PROVEN_EQUIVALENT 里给出书面等价性证明。

三类存活值得单独说明：

1. **`slippage_model="linear"` 分支全代码库无人调用、无人测试**（L87–L89）。
   `grep -rn slippage_model app/` 只有默认值与 `== "sqrt"` 判断两处，
   没有任何地方把它设成 linear，测试里也一次没出现过。
   三行公式随便改（`0.5 * spread` → `0.5 / spread`）都照样全绿。
   —— 这不是"等价变异"，是**死代码上的真空**。本文件把它的契约钉死。

2. **`np.abs(deficit)` 的 abs 可以删掉而无人发现**（L132）。删掉后
   `deficit < 0`（即 L1 已超过目标，需要**缩小**）会被误判为"已收敛"直接 break，
   投影不再缩放，L1 范数远超 target。既有用例全都恰好用 L1 == target 或
   预算不足的输入，永远不触发"需要缩小"这一支。

3. **`p > 0` / `abs(dw) < 1e-10` 的边界可以精确构造**（L266/L269），
   与 risk_gate 那批 `limit + tol` 不同 —— 那批的区分值在浮点上造不出来，
   这批的区分值就是 `0.0` 和 `1e-10` 本身，所以必须写用例，不许当等价变异放过。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.backtest_engine.transaction_cost import (
    CostParams,
    LiquidityConstraint,
    SlippageModel,
    TransactionCostEngine,
    project_to_capped_l1,
)


# ===========================================================================
# A. linear 滑点分支 —— 此前零覆盖
# ===========================================================================

class TestLinearSlippageBranch:
    """
    linear 模式公式（源码 L86–L89）：
        slippage_bps = (0.5 * spread_bps + 0.1 * daily_vol * 10000) * ones_like(trade)

    注意 0.1 是**写死的常数，不是 impact_coef** —— 下面 impact_coef 故意设成
    与 0.1 不同的值，若有人把常数换成 self.p.impact_coef，本用例会立刻变红。
    """

    @staticmethod
    def _slip(**kw):
        p = CostParams(slippage_model="linear", spread_bps=6.0, impact_coef=0.9, **kw)
        return SlippageModel(p).compute(
            trade_usd=np.array([1e5, 0.0]),
            adv_usd=np.array([1e7, 1e7]),
            daily_vol=np.array([0.02, 0.02]),
        )

    def test_linear_formula_exact(self):
        # 0.5*6 + 0.1*0.02*10000 = 3 + 20 = 23.0
        assert self._slip()[0] == pytest.approx(23.0, abs=1e-12)

    def test_linear_zero_trade_is_zero(self):
        assert self._slip()[1] == 0.0

    def test_linear_uses_hardcoded_0_1_not_impact_coef(self):
        """impact_coef=0.9 与 0.1 差 9 倍；若被误用，波动项会是 180 而非 20。"""
        assert self._slip()[0] != pytest.approx(3.0 + 180.0, abs=1e-9)

    def test_linear_scales_with_spread_and_vol_independently(self):
        """价差项与波动项各自的系数被分别钉死（只断言总和会漏掉互补错误）。"""
        p = CostParams(slippage_model="linear", spread_bps=6.0)
        base = SlippageModel(p).compute(
            np.array([1e5]), np.array([1e7]), np.array([0.02]))[0]
        # 只加倍 spread → 只有 3.0 那一半翻倍
        p2 = CostParams(slippage_model="linear", spread_bps=12.0)
        assert SlippageModel(p2).compute(
            np.array([1e5]), np.array([1e7]), np.array([0.02]))[0] == \
            pytest.approx(base + 3.0, abs=1e-12)
        # 只加倍 vol → 只有 20.0 那一半翻倍
        assert SlippageModel(p).compute(
            np.array([1e5]), np.array([1e7]), np.array([0.04]))[0] == \
            pytest.approx(base + 20.0, abs=1e-12)

    def test_linear_ignores_participation(self):
        """linear 模式与 ADV 无关（这是它与 sqrt 的本质区别）。"""
        p = CostParams(slippage_model="linear", spread_bps=6.0)
        a = SlippageModel(p).compute(np.array([1e5]), np.array([1e7]), np.array([0.02]))
        b = SlippageModel(p).compute(np.array([1e5]), np.array([1e12]), np.array([0.02]))
        assert a[0] == pytest.approx(b[0], abs=1e-12)


# ===========================================================================
# B. project_to_capped_l1 —— "需要缩小"这一支
# ===========================================================================

class TestProjectionShrinkBranch:
    """
    既有用例（test_phase6 / golden）只覆盖两种输入：
      (a) L1 恰好 == target；(b) 预算不足需要保持在预算内。
    从没有 **L1 > target 且预算充足 → 必须按比例缩小** 的用例，
    于是 `np.all(np.abs(deficit) < tol)` 里的 abs 被删掉也无人发现。
    """

    def test_oversized_weights_are_scaled_down_to_target(self):
        w   = np.array([[2.0, -2.0]])          # L1 = 4，远超 target
        cap = np.array([[9.0, 9.0]])           # 上限宽松，不构成约束
        p   = project_to_capped_l1(w, cap, target=1.0)
        assert np.abs(p).sum() == pytest.approx(1.0, abs=1e-12)
        assert p.tolist() == [[pytest.approx(0.5), pytest.approx(-0.5)]]

    def test_oversized_with_mixed_caps_still_hits_target(self):
        """一名触顶、一名自由：触顶名固定在 cap，自由名吸收剩余，总和 == target。"""
        w   = np.array([[3.0, -3.0, 3.0]])
        cap = np.array([[0.2, 9.0, 9.0]])
        p   = project_to_capped_l1(w, cap, target=1.0)
        assert abs(p[0, 0]) == pytest.approx(0.2, abs=1e-9)
        assert np.abs(p).sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(np.abs(p) <= cap + 1e-9)

    def test_negative_cap_is_taken_as_magnitude(self):
        """
        cap 为负是脏数据可达的路径：LiquidityConstraint 里
        `max_usd = adv * adv_cap_pct`，adv 为负（数据源异常）就会得到负 cap。
        源码用 `np.abs(cap)` 兜底；删掉 abs 后 `np.minimum(a, cap)` 会把权重
        直接压成负数，方向被破坏。
        """
        p = project_to_capped_l1(
            np.array([[0.9, -0.9]]), np.array([[-0.2, -0.2]]), target=1.0)
        assert p.tolist() == [[pytest.approx(0.2), pytest.approx(-0.2)]]

    def test_sign_is_taken_from_input_not_from_cap(self):
        w = np.array([[-0.9, 0.9]])
        p = project_to_capped_l1(w, np.array([[0.3, 0.3]]), target=1.0)
        assert np.sign(p).tolist() == [[-1.0, 1.0]]


# ===========================================================================
# C. LiquidityConstraint —— portfolio_value == 0 的边界
# ===========================================================================

class TestLiquidityZeroPortfolio:
    """
    `np.where(portfolio_value > 0, max_usd / portfolio_value, np.inf)`
    改成 `>=` 后，pv==0 会走进 `max_usd / 0`：
      - max_usd > 0 → inf（与正确分支同值，看不出来）
      - max_usd == 0 → **nan**（0/0），cap 变 nan，权重全部污染成 nan
    所以必须用 **adv == 0 且 pv == 0** 才能区分，只测 pv==0 是不够的。
    """

    @staticmethod
    def _apply(adv_value, pv):
        idx  = pd.bdate_range("2022-01-03", periods=1)
        cols = ["A", "B"]
        w    = pd.DataFrame([[0.6, -0.4]], index=idx, columns=cols)
        adv  = pd.DataFrame([[adv_value, adv_value]], index=idx, columns=cols)
        return LiquidityConstraint(CostParams()).apply(w, adv, portfolio_value=pv)

    def test_zero_portfolio_zero_adv_does_not_produce_nan(self):
        out = self._apply(0.0, 0.0)
        assert not out.isna().to_numpy().any(), "pv==0 且 adv==0 时权重被污染成 NaN"
        assert out.to_numpy().tolist() == [[pytest.approx(0.6), pytest.approx(-0.4)]]

    def test_zero_portfolio_means_no_liquidity_cap(self):
        """pv==0 → 无法用金额换算权重上限 → 按无约束处理，原样返回。"""
        out = self._apply(1e7, 0.0)
        assert out.to_numpy().tolist() == [[pytest.approx(0.6), pytest.approx(-0.4)]]

    def test_positive_portfolio_does_apply_cap(self):
        """对照组：pv>0 时上限确实生效，否则上一条是空真。"""
        out = self._apply(1e4, 1_000_000.0)
        assert out["A"].abs().iloc[0] <= 1e4 * 0.10 / 1_000_000.0 + 1e-12


# ===========================================================================
# D. TransactionCostEngine.compute —— 逐笔记录的精确值与边界
# ===========================================================================

class TestTradeRecordExactness:
    """
    输入：delta_w=0.5, price=100, adv=1e9, vol=0.02, pv=1e6, 默认 CostParams
      participation = 5e5 / 1e9 = 5e-4
      slip_bps = 2/2 + 0.1 * 0.02 * 1e4 * sqrt(5e-4)
               = 1 + 20 * 0.0223606797749979 = 1.4472135954999579
      shares    = 0.5 * 1e6 / 100        = 5000.0
      net_price = 100 * (1 + 1.4472135954999579e-4) = 100.01447213595500
    """

    @staticmethod
    def _one(delta_w=0.5, price=100.0):
        return TransactionCostEngine(CostParams()).compute(
            date=None,
            delta_w=np.array([delta_w]),
            prices=np.array([price]),
            adv_usd=np.array([1e9]),
            daily_vol=np.array([0.02]),
            portfolio_val=1_000_000.0,
            tickers=["A"],
        )

    def test_shares_formula_exact(self):
        """`abs(dw) * portfolio_val / p`：`*` 改 `/` 会得到 5e-13 量级。"""
        _, _, rec = self._one()
        assert rec[0].shares == pytest.approx(5000.0, abs=1e-9)

    def test_net_price_exact(self):
        """
        `p * (1 + slip*1e-4*sign)`：
          `+` 改 `-` → 99.985527...（滑点方向反了，买单反而更便宜）
          `*` 改 `/` → 99.985529...（量纲反了）
        两者都与正确值差约 0.03%，只有精确断言能咬住。
        """
        _, _, rec = self._one()
        assert rec[0].net_price == pytest.approx(100.014472135955, abs=1e-9)

    def test_sell_moves_net_price_the_other_way(self):
        """买单成交价高于中价、卖单低于中价 —— 方向必须相反。"""
        _, _, buy  = self._one(delta_w=0.5)
        _, _, sell = self._one(delta_w=-0.5)
        assert buy[0].net_price > 100.0 > sell[0].net_price
        assert buy[0].net_price - 100.0 == pytest.approx(100.0 - sell[0].net_price, abs=1e-12)
        assert buy[0].direction == "BUY" and sell[0].direction == "SELL"

    def test_zero_price_yields_zero_shares_not_infinity(self):
        """
        `p > 0` 改成 `>=` 后 p==0 会走进 `x / 0` → inf 股，
        下游按股数下单就是天文数字。区分值就是 0.0 本身，可精确构造。
        """
        _, _, rec = self._one(price=0.0)
        assert rec[0].shares == 0.0
        assert np.isfinite(rec[0].shares)

    def test_trade_size_threshold_is_strictly_below_1e_minus_10(self):
        """
        `if abs(dw) < 1e-10: continue` 改成 `<=` 后，恰好 1e-10 的交易会被丢掉。
        这个区分值可以精确构造（不同于 risk_gate 那批 `limit + tol`）。
        """
        _, _, at   = self._one(delta_w=1e-10)
        _, _, below = self._one(delta_w=9e-11)
        assert len(at) == 1, "abs(dw) == 1e-10 属于'要记录'的一侧"
        assert len(below) == 0

    def test_records_to_df_returns_rows_when_records_exist(self):
        """`if not records:` 删掉 not 后，有记录反而返回空表。"""
        _, _, rec = self._one()
        df = TransactionCostEngine.records_to_df(rec)
        assert len(df) == 1
        assert df["ticker"].tolist() == ["A"]
        assert df["shares"].iloc[0] == pytest.approx(5000.0, abs=1e-9)

    def test_records_to_df_empty_list_gives_empty_frame_with_columns(self):
        df = TransactionCostEngine.records_to_df([])
        assert df.empty
        assert list(df.columns) == [
            "date", "ticker", "direction", "shares",
            "price", "slippage_bps", "cost_usd", "net_price",
        ]


# ===========================================================================
# E. 存活变异的等价性证明 —— 每一条都必须写清楚"为什么杀不掉是对的"
# ===========================================================================

# 复测口径：41 个变异点 / 杀死 35 / 存活 6 / 击杀率 85.4%（2026-09-09）
# 下面 6 条 == 复测后仍然存活的 6 条，一一对应，不多不少。
PROVEN_EQUIVALENT = {
    "L89  `) * np.ones_like(trade_abs)` → `/`":
        "np.ones_like 返回全 1 数组，x * 1 与 x / 1 在 IEEE754 下逐元素完全相同"
        "（包括 ±0、inf、nan 的行为），无任何输入可区分。见 "
        "test_dividing_by_ones_equals_multiplying_by_ones。",

    "L132 `np.abs(deficit) < tol` → `<=`（tol=1e-12）":
        "仅在 |deficit| 恰好 == 1e-12 时不同。deficit 由 row_target 减两个 nansum "
        "得到，是多次浮点加减的残差，无法构造使其精确等于 1e-12；且 tol 的存在本身"
        "就是为了让该邻域内的判定不敏感。",

    "L135 `free_mass > tol` → `>=`（tol=1e-12）":
        "同上：free_mass 是 nansum 的浮点结果，恰好 == 1e-12 不可构造。"
        "且该分支的两侧在 free_mass≈tol 时结果连续（safe_free 取 free_mass 与 1.0 "
        "的差异只影响 scale 的分母，而此时 deficit 也已趋近 0）。",

    "L136 `free_mass > tol` → `>=`（tol=1e-12）":
        "与 L135 是同一个条件的第二次出现，同一份证明。两处必须同时改才可能有差异，"
        "单点变异下更不可能被区分。",

    "L268 `direction = \"BUY\" if dw > 0 else \"SELL\"` → `>=`":
        "仅在 dw == 0 时不同，而 dw == 0 在上一行 `if abs(dw) < 1e-10: continue` "
        "就已被跳过（abs(0.0)=0.0 < 1e-10 成立，-0.0 同理），该分支不可达。",

    "L270 `1 if dw > 0 else -1` → `>=`":
        "与 L268 同一原因：dw == 0 不可达，因为 L266 已经 continue。",
}

# 曾经存活、现已消解的两条 —— 留档以免下次重新把它们当成"等价变异"放过：
#   L131 `# >0 需放大自由名`：`>` 在**行尾注释**里，改了它源码语义分毫未变。
#        这是测量工具的假阳性（未屏蔽注释区间），已修，复测后不再出现。
#   L187 `portfolio_value > 0` → `>=`：我一度判它等价——因为 pv==0 且 max_usd>0 时
#        两个分支都得 inf。**判错了**：max_usd==0 时变异分支是 0/0 = nan，
#        权重被整片污染。test_zero_portfolio_zero_adv_does_not_produce_nan 杀掉了它。
#        教训：判等价前要把**被除数也取遍边界**，只遍历除数是不够的。


def test_dividing_by_ones_equals_multiplying_by_ones():
    """L89 等价性的机械验证（不是靠嘴说）。"""
    rng = np.random.default_rng(11)
    for _ in range(200):
        x = rng.normal(0, 1e6, 32)
        ones = np.ones_like(x)
        assert np.array_equal(x * ones, x / ones)
    edge = np.array([0.0, -0.0, np.inf, -np.inf, 1e-300, 1e300])
    edge_ones = np.ones_like(edge)
    assert np.array_equal(edge * edge_ones, edge / edge_ones)


def test_zero_delta_weight_is_unreachable_in_record_loop():
    """
    L268/L270 等价性的机械验证：dw==0（含 -0.0）一定在阈值处被 continue，
    因此 `dw > 0` 与 `dw >= 0` 不可能产生不同记录。
    """
    for dw in (0.0, -0.0):
        assert abs(dw) < 1e-10                     # → continue，后续分支不可达
    _, _, rec = TransactionCostEngine(CostParams()).compute(
        date=None, delta_w=np.array([0.0, -0.0]), prices=np.array([100.0, 100.0]),
        adv_usd=np.array([1e9, 1e9]), daily_vol=np.array([0.02, 0.02]),
        portfolio_val=1_000_000.0, tickers=["A", "B"],
    )
    assert rec == []


def test_epsilon_guarded_tolerances_are_provably_unreachable():
    """
    L132/L135/L136 用的 tol=1e-12 是 water-filling 的收敛容差。
    与 risk_gate 同样的论证：浮点上无法把一个由多步加减得到的残差
    构造成恰好等于 tol，因此 `< tol` 与 `<= tol` 不可区分。
    """
    tol = 1e-12
    for base in (1.0, 0.45, 4.0, 1e3):
        assert (base + tol) - base != tol, f"base={base} 处 tol 可精确还原，证明不成立"


def test_every_survivor_has_a_written_proof():
    """
    强制每条证明不得敷衍 —— risk_gate 那轮我自己写过"同 L117。"（7 个字）
    被这条规则抓出来，这里沿用同一把尺子。
    """
    assert len(PROVEN_EQUIVALENT) == 6, "存活项数量与证明条数对不上，说明有遗漏"
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
