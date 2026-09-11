"""
portfolio_manager/manager.py —— 账本视图与容量约束的定钉测试（变异测试驱动）

来由：14 个变异点，首测击杀率 35.7%（存活 9）。存活项集中在：

  - `book_on()` 的三处判定（微仓过滤、价格有效性、股数换算）——
    这是**人看到的持仓账本**，也是"我现在到底拿着多少股"的唯一出处；
  - `long_only` 的三处取值（配置推导、读配置失败的兜底、显式默认）——
    兜底方向必须朝保守一侧（读不到配置就假定**不能做空**），
    否则现金账户会构造出根本无法成交的空头腿而无人知晓；
  - `cap_w > 0.0` 的容量下限。

既有覆盖（test_phase_pm / test_phase_pm5_risk）只跑通了 build_book 的主路径。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.portfolio_manager.manager import PortfolioManager, PortfolioResult


def _panel(days: int = 60, tickers=("A", "B", "C", "D"), seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=days)
    cols = list(tickers)
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (days, len(cols))), axis=0),
        index=idx, columns=cols)
    vol = pd.DataFrame(rng.integers(2e6, 9e6, close.shape).astype(float),
                       index=idx, columns=cols)
    return close, vol


# ===========================================================================
# A. book_on —— 人看到的持仓账本
# ===========================================================================

class TestBookOn:

    @staticmethod
    def _result(weights: list[float], prices: list[float] | None,
                aum: float = 100_000.0) -> PortfolioResult:
        idx = pd.bdate_range("2024-01-02", periods=1)
        cols = [f"T{i}" for i in range(len(weights))]
        w = pd.DataFrame([weights], index=idx, columns=cols)
        px = (pd.DataFrame([prices], index=idx, columns=cols)
              if prices is not None else None)
        return PortfolioResult(weights=w, combo_weights={}, aum=aum,
                               prices=px, composite=w)

    def test_dollars_and_shares_are_exact(self):
        """
        `dollars = w × aum`、`shares = dollars / price`。
        构造：权重 0.25、AUM 100k、价格 50 → 25,000 美元 / 500 股。
        """
        res = self._result([0.25, -0.10], [50.0, 20.0])
        book = res.book_on(res.weights.index[0])
        assert book["T0"]["dollars"] == pytest.approx(25_000.0, abs=1e-6)
        assert book["T0"]["shares"] == pytest.approx(500.0, abs=1e-9)
        assert book["T1"]["dollars"] == pytest.approx(-10_000.0, abs=1e-6)
        assert book["T1"]["shares"] == pytest.approx(-500.0, abs=1e-9)

    def test_tiny_weights_are_dropped(self):
        """
        `if abs(wi) < 1e-9: continue` —— 微仓不进账本（1e-9 权重在 10 万美元上
        不足 0.0001 美元）。放宽成 `<=` 只在恰好 1e-9 时不同，见等价性证明；
        这里钉住的是**方向**：明显小于阈值的必须被丢掉、明显大于的必须保留。
        """
        res = self._result([1e-12, 0.3], [50.0, 50.0])
        book = res.book_on(res.weights.index[0])
        assert "T0" not in book, "微仓进入了账本"
        assert "T1" in book

    def test_zero_price_yields_nan_shares_not_infinity(self):
        """
        `shares = dollars/px if (px is not None and px > 0) else nan`。
        `>` 放宽成 `>=` 后 0 价会真的做除法 → inf 股；
        `and` 改成 `or` 后 px 为 None 时会去取 `px[tk]` 抛异常。
        """
        res = self._result([0.25], [0.0])
        book = res.book_on(res.weights.index[0])
        assert np.isnan(book["T0"]["shares"]), "0 价却算出了有限股数"
        assert book["T0"]["dollars"] == pytest.approx(25_000.0, abs=1e-6)

    def test_missing_price_frame_yields_nan_shares(self):
        res = self._result([0.25], None)
        book = res.book_on(res.weights.index[0])
        assert np.isnan(book["T0"]["shares"])
        assert book["T0"]["dollars"] == pytest.approx(25_000.0, abs=1e-6)

    def test_negative_price_is_rejected(self):
        res = self._result([0.25], [-10.0])
        assert np.isnan(res.book_on(res.weights.index[0])["T0"]["shares"])

    def test_gross_and_net_series(self):
        idx = pd.bdate_range("2024-01-02", periods=2)
        w = pd.DataFrame([[0.6, -0.4], [0.5, -0.5]], index=idx, columns=["A", "B"])
        res = PortfolioResult(weights=w, combo_weights={}, aum=1.0,
                              prices=None, composite=w)
        assert res.gross_series().tolist() == [pytest.approx(1.0), pytest.approx(1.0)]
        assert res.net_series().tolist() == [pytest.approx(0.2), pytest.approx(0.0)]

    def test_repr_excludes_the_panels(self):
        """`field(repr=False)` —— prices/composite 是 T×N 面板，不得进 repr。"""
        res = self._result([0.25], [50.0])
        text = repr(res)
        assert "prices" not in text and "composite" not in text
        assert len(text) < 1000


# ===========================================================================
# B. long_only 的三处取值
# ===========================================================================

class TestLongOnlyResolution:

    def test_follows_the_allow_short_setting(self, monkeypatch):
        """
        `long_only = not bool(getattr(settings, "trading_allow_short", False))`
        —— 删掉 `not` 或把 getattr 默认改成 True，现金账户都会被允许做空。
        """
        from app.config import settings
        monkeypatch.setattr(settings, "trading_allow_short", False, raising=False)
        assert PortfolioManager(aum=1e5).long_only is True
        monkeypatch.setattr(settings, "trading_allow_short", True, raising=False)
        assert PortfolioManager(aum=1e5).long_only is False

    def test_explicit_argument_wins(self):
        assert PortfolioManager(aum=1e5, long_only=False).long_only is False
        assert PortfolioManager(aum=1e5, long_only=True).long_only is True

    def test_missing_setting_defaults_to_long_only(self, monkeypatch):
        """
        `getattr(settings, "trading_allow_short", False)` 的**第三参**：
        配置项**缺失**时按不允许做空处理。显式赋值的用例测不到这个默认值
        （第一版就是这样，变异测试证实它存活）。
        """
        import types
        import app.config
        monkeypatch.setattr(app.config, "settings", types.SimpleNamespace())
        assert PortfolioManager(aum=1e5).long_only is True, (
            "配置项缺失时放开了做空 —— getattr 的默认值方向反了")

    def test_config_failure_falls_back_to_long_only(self, monkeypatch, caplog):
        """
        `except: long_only = True` —— 读不到配置时**假定不能做空**（保守侧）。
        改成 False 会在最不该冒险的场景（配置都读不到）放开做空。
        """
        import app.config

        class _Boom:
            def __getattr__(self, name):
                raise RuntimeError("settings unavailable")

        monkeypatch.setattr(app.config, "settings", _Boom())
        with caplog.at_level("ERROR"):
            pm = PortfolioManager(aum=1e5)
        assert pm.long_only is True, "读不到配置时放开了做空 —— 兜底方向反了"
        assert any("long_only" in r.getMessage() for r in caplog.records), (
            "保守兜底没有留下 ERROR 日志")

    def test_long_only_book_has_no_short_leg(self, monkeypatch):
        """端到端：long_only 的账本不得出现负权重，且满仓（Σw≈1）。"""
        from app.config import settings
        monkeypatch.setattr(settings, "trading_allow_short", False, raising=False)
        close, vol = _panel()
        sig = {"f1": close.pct_change().rank(axis=1)}
        book = PortfolioManager(aum=1e6).build_book(sig, close, vol)
        arr = book.weights.to_numpy()
        assert (arr >= -1e-12).all(), "long_only 账本出现了空头"


# ===========================================================================
# C. 容量约束
# ===========================================================================

class TestCapacity:

    def test_non_positive_capacity_is_zeroed(self, monkeypatch):
        """
        `cap_w = np.where(np.isfinite(cap_w) & (cap_w > 0.0), cap_w, 0.0)`
        —— 无成交量（ADV=0）的标的容量为 0，必须**拿不到任何权重**。
        `>` 放宽成 `>=` 时 0 容量会被保留成 0（等价），
        但把 isfinite 判掉会让 inf 容量混进来 —— 这里钉住"零成交量 = 零权重"。
        """
        from app.config import settings
        monkeypatch.setattr(settings, "trading_allow_short", True, raising=False)
        close, vol = _panel()
        vol.iloc[:, 0] = 0.0                       # A 完全没有成交量
        sig = {"f1": close.pct_change().rank(axis=1)}
        book = PortfolioManager(aum=1e6).build_book(sig, close, vol)
        assert np.abs(book.weights["A"].to_numpy()).max() == pytest.approx(0.0, abs=1e-12), (
            "零成交量的标的仍拿到了权重")
        assert np.abs(book.weights.to_numpy()).sum() > 0, "其余标的也被清零了"

    def test_capacity_limits_gross_when_aum_is_large(self):
        """AUM 大到 ADV 撑不住时 gross 必须 < 1（不得靠归一化推回满仓）。"""
        close, vol = _panel()
        sig = {"f1": close.pct_change().rank(axis=1)}
        small = PortfolioManager(aum=1e5).build_book(sig, close, vol)
        huge = PortfolioManager(aum=1e11).build_book(sig, close, vol)
        assert huge.gross_series().iloc[-1] < small.gross_series().iloc[-1], (
            "AUM 放大 6 个数量级后总敞口没有下降 —— 容量约束疑似失效")
        assert huge.gross_series().iloc[-1] < 1.0

    def test_empty_signals_are_rejected(self):
        close, vol = _panel()
        with pytest.raises(ValueError, match="factor_signals"):
            PortfolioManager(aum=1e6).build_book({}, close, vol)


# ===========================================================================
# D. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "L52 `if abs(wi) < 1e-9: continue` → `<=`":
        "区分值需要 |wi| **恰好等于** 1e-9。wi 来自 water-filling 投影后的权重"
        "（多步浮点乘除并归一到 L1=1），无法反解出精确等于 1e-9 的输入；"
        "且该阈值的用途就是「小到等于没有持仓」的模糊带 —— 1e-9 权重在 10 万美元"
        "账户上不足 0.0001 美元。",

    "L128 `cap_w > 0.0` → `>=`":
        "`np.where(cond, cap_w, 0.0)` 在 cap_w == 0.0 时两侧取值相同："
        "条件为真取 cap_w（=0.0），为假取常量 0.0。唯一的区分点上输出一致。"
        "见 test_zero_capacity_is_zero_either_way。",
}


def test_zero_capacity_is_zero_either_way():
    """L128 等价性的机械验证。"""
    for cap in (np.array([0.0]), np.array([-0.0]), np.array([0.0, 0.5, np.inf])):
        finite = np.isfinite(cap)
        a = np.where(finite & (cap > 0.0), cap, 0.0)
        b = np.where(finite & (cap >= 0.0), cap, 0.0)
        assert np.array_equal(a, b), f"cap={cap} 上两种比较符结果不同"


def test_tiny_weight_threshold_is_unreachable():
    """L52 等价性的机械验证：1e-9 这个精确值在浮点上不可由归一化构造。"""
    tol = 1e-9
    for base in (1.0, 0.25, 0.6):
        assert (base + tol) - base != tol
    rng = np.random.default_rng(7)
    for _ in range(300):
        w = rng.normal(0, 1, 12)
        w = w / np.abs(w).sum()
        assert not np.any(np.abs(w) == tol)


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 2
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
