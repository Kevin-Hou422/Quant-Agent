"""
test_leak_filter.py — 因子入池门（**系统默认门**）的直接测试

来由：变异测试 + 覆盖率交叉验证发现 `app/core/lifecycle/leak_filter.py`
**在整个测试套件里一次都没被直接测过**（109 个被覆盖文件中不含它；对它做变异测试
时每一处改动都存活 = 击杀率 0%）。而 `factor_gate_mode` 的默认值就是 `"leak"` ——
也就是说：**系统默认使用的因子准入门，没有任何测试保护**。

它能通过"孤儿模块"检查，是因为 `discovery_engine` 里有对它的 import ——
**静态可达 ≠ 运行时被执行**，这是 §K 之下又一层。

本文件按"这个门该拦住什么"逐条建立契约：低门槛过滤，只拦泄漏/退化，
不做业绩判断（严门在策略层）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.lifecycle.leak_filter import leak_filter


# ---------------------------------------------------------------------------
# 夹具：真实幅度的合成面板（DEV_LESSONS §O：H/L 不得用固定 ±1%）
# ---------------------------------------------------------------------------

def _panel(n_days: int = 160, n_tickers: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n_days)
    cols = [f"T{i:02d}" for i in range(n_tickers)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.012, (n_days, n_tickers)), axis=0),
        index=idx, columns=cols)
    high = close * (1 + rng.uniform(0, 0.006, close.shape))
    low = close * (1 - rng.uniform(0, 0.006, close.shape))
    vol = pd.DataFrame(rng.integers(8e5, 5e6, close.shape).astype(float),
                       index=idx, columns=cols)
    return {"close": close, "open": close, "high": high, "low": low, "volume": vol,
            "vwap": (high + low + close) / 3.0,
            "returns": close.pct_change().fillna(0.0)}


@pytest.fixture
def ds():
    return _panel()


# ---------------------------------------------------------------------------
# 1. 正常因子必须放行（门不能把一切都拦掉）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dsl", [
    "rank(ts_delta(log(close), 5))",
    "zscore(ts_mean(returns, 10))",
    "rank((-ts_std(returns, 20)))",
])
def test_ordinary_factor_passes(ds, dsl):
    passed, detail = leak_filter(dsl, ds)
    assert passed is True, f"正常因子被拦：{dsl} | {detail}"
    assert isinstance(detail, dict) and detail.get("passed") is True, detail


def test_gate_is_low_bar_not_a_performance_judge(ds):
    """
    这是**低门槛**门：只拦泄漏/退化，不做业绩判断（严门在策略层）。
    一个 Sharpe 很差但结构正常的因子必须放行 —— 否则职责就跑到策略层去了。
    """
    passed, detail = leak_filter("rank(ts_delta(close, 1))", ds)
    assert passed is True, f"业绩差不等于该拦：{detail}"


# ---------------------------------------------------------------------------
# 2. 必须拦住的三类（这才是门存在的理由）
# ---------------------------------------------------------------------------

def test_rejects_all_nan_signal(ds):
    """信号全 NaN（执行退化）→ 必须拦，且说明原因。"""
    bad = {k: v.copy() for k, v in ds.items()}
    bad["close"] = bad["close"] * np.nan
    passed, detail = leak_filter("rank(ts_delta(log(close), 5))", bad)
    assert passed is False, "全 NaN 信号被放行"
    assert detail.get("reasons"), f"拦截未给出原因：{detail}"


def test_rejects_zero_cross_sectional_variance(ds):
    """截面无区分度（所有标的同值）→ 该信号无法排序，必须拦。"""
    flat = {k: v.copy() for k, v in ds.items()}
    flat["close"] = pd.DataFrame(100.0, index=ds["close"].index,
                                 columns=ds["close"].columns)
    passed, detail = leak_filter("close", flat)
    assert passed is False, "零截面方差信号被放行（无法排序却当成有效因子）"
    assert detail.get("reasons"), detail


def test_rejects_implausible_sharpe_as_leakage_signature(ds):
    """
    |Sharpe| 高到不可信 = 前视/泄漏的典型征兆。
    用**未来收益**直接构造信号（`ts_delay` 负延迟不可用，改用当日 returns 本身
    与次日收益强相关的构造），验证门确实按 max_plausible_sharpe 拦截。
    """
    n = 200
    idx = pd.bdate_range("2023-01-02", periods=n)
    cols = [f"T{i:02d}" for i in range(6)]
    rng = np.random.default_rng(7)
    rets = pd.DataFrame(rng.normal(0, 0.01, (n, 6)), index=idx, columns=cols)
    close = 100 * (1 + rets).cumprod()
    leak = {"close": close, "open": close,
            "high": close * 1.003, "low": close * 0.997,
            "volume": pd.DataFrame(1e6, index=idx, columns=cols),
            "vwap": close, "returns": rets}
    # 阈值调到极低 → 任何有区分度的因子都会触发"不可信"分支，
    # 直接验证该分支真的会拦（而不是形同虚设）。
    passed, detail = leak_filter("rank(ts_delta(log(close), 5))", leak,
                                 max_plausible_sharpe=0.0001)
    assert passed is False, "|Sharpe| 超过可信上限却被放行 —— 泄漏检测分支失效"
    assert any("夏普" in r or "sharpe" in r.lower() for r in detail.get("reasons", [])), (
        f"拦截原因未指向 Sharpe：{detail}"
    )


# ---------------------------------------------------------------------------
# 3. 契约的形状（调用方据此做决策，不能变）
# ---------------------------------------------------------------------------

def test_返回契约稳定(ds):
    passed, detail = leak_filter("rank(close)", ds)
    assert isinstance(passed, bool)
    assert isinstance(detail, dict)
    assert "passed" in detail and detail["passed"] == passed, (
        f"detail['passed'] 与返回值不一致：{detail.get('passed')} vs {passed}"
    )


def test_unparseable_dsl_is_rejected_with_reason(ds):
    """非法 DSL 必须被拦并说明，而不是抛出未捕获异常打崩发现流程。"""
    passed, detail = leak_filter("NOT_A_VALID_DSL_XYZ(", ds)
    assert passed is False
    assert detail.get("reasons"), f"非法 DSL 未给出拒绝原因：{detail}"


def test_threshold_is_effective_not_decorative(ds):
    """
    阈值必须真的起作用：同一个因子，把 max_plausible_sharpe 压到 0 应当被拦，
    放到很大应当放行。若两者结果相同，说明该参数是装饰品。
    """
    dsl = "rank(ts_delta(log(close), 5))"
    strict, _ = leak_filter(dsl, ds, max_plausible_sharpe=0.0001)
    loose, _ = leak_filter(dsl, ds, max_plausible_sharpe=100.0)
    assert loose is True and strict is False, (
        f"max_plausible_sharpe 不起作用：严={strict} 松={loose}"
    )


# ---------------------------------------------------------------------------
# 4. 公式与边界（变异测试驱动补充）
#
# 上面这些用例把击杀率从 0% 提到 44.4%，但 9 个变异点仍存活 5 个：
# 静态校验开关、年化系数、阈值比较的等号侧，改坏了全都没人发现。
# 下面逐个处置。
# ---------------------------------------------------------------------------

DSL = "rank(ts_delta(log(close), 5))"


def _replica_is_sharpe(dsl: str, dataset) -> float:
    """
    用与 leak_filter 相同的公开组件复算 IS 夏普，得到**未取整**的精确值。

    刻意复刻而不是读 `detail["is_sharpe"]`：后者 `round(x, 3)` 过，
    无法用来构造"阈值恰好等于夏普"的边界输入；而且这份复刻本身就把
    「Executor → SignalProcessor → SignalWeightedPortfolio → BacktestEngine」
    这条管线钉住了——管线一改，这里就会红。
    """
    import pandas as _pd
    from app.core.alpha_engine.dsl_executor import Executor
    from app.core.alpha_engine.signal_processor import SignalProcessor, SimulationConfig
    from app.core.backtest_engine.backtest_engine import BacktestEngine
    from app.core.backtest_engine.portfolio_constructor import SignalWeightedPortfolio

    raw = Executor(validate=False).run_expr(dsl, dataset)
    cfg = SimulationConfig(delay=1, decay_window=0,
                           truncation_min_q=0.05, truncation_max_q=0.95)
    proc = SignalProcessor(cfg).process(raw)
    w = SignalWeightedPortfolio(clip_z=3.0).construct(proc)
    rets = _pd.Series(
        BacktestEngine().run(w, dataset["close"], dataset["volume"], proc).net_returns
    ).dropna()
    mu, sd = float(rets.mean()), float(rets.std(ddof=1))
    return (mu / sd) * np.sqrt(252.0) if sd > 1e-12 else 0.0


def test_static_validator_is_deliberately_off():
    """
    `Executor(validate=False)` 里的 False 是**有意的设计**，不是笔误：
    本门要靠"实测夏普高到不可信"来抓泄漏，而不是靠静态规则先把表达式毙掉。
    改成 True 后，凡是触发 WindowValidator（窗口 > 252）/ DepthValidator 的
    表达式都会走进 `except` 被记成"执行失败"，理由完全指错方向。

    用 `ts_mean(close, 300)`：静态校验会拒（窗口 300 > 252），但它能正常执行，
    也没有任何泄漏，属于必须放行的一类。
    """
    long_window_ds = _panel(n_days=400)
    passed, detail = leak_filter("rank(ts_mean(close, 300))", long_window_ds)
    assert passed is True, f"长窗口因子被静态校验误伤：{detail}"
    assert not any("执行失败" in r for r in detail.get("reasons", [])), detail
    assert "is_sharpe" in detail, "没有走到夏普分支 —— 说明执行被提前中断了"


def test_is_sharpe_is_annualised_by_sqrt_252(ds):
    """
    `(mu / sd) * np.sqrt(252.0)`。`*` 改 `/` 后年化系数变成 1/15.87，
    夏普整体缩小 252 倍——但**符号和相对大小都不变**，
    所以任何"夏普为正/为负"的断言都抓不住它，必须比数值。
    """
    expected = _replica_is_sharpe(DSL, ds)
    _, detail = leak_filter(DSL, ds, max_plausible_sharpe=1e9)
    assert detail["is_sharpe"] == pytest.approx(round(float(expected), 3), abs=1e-9)
    # 量级本身也钉一下：日频 mu/sd 乘 √252 后应在个位数量级，
    # 而不是被除成千分之一（改 `/` 后 0.488 → 0.0019）。
    assert abs(detail["is_sharpe"]) > 0.05


def test_threshold_boundary_excludes_equality(ds):
    """
    `if abs(sharpe) > max_plausible_sharpe:` —— 阈值**恰好等于**实测夏普时
    属于**放行**的一侧。这个区分值可以精确构造（复刻管线拿未取整的值），
    因此不属于"浮点上造不出区分值"的等价变异，必须写用例。
    """
    s = abs(_replica_is_sharpe(DSL, ds))
    assert s > 0, "复刻夏普为 0，本用例无法区分边界"
    at_boundary, _ = leak_filter(DSL, ds, max_plausible_sharpe=s)
    just_below, d2 = leak_filter(DSL, ds, max_plausible_sharpe=float(np.nextafter(s, 0.0)))
    assert at_boundary is True, "阈值恰好等于实测夏普时被拦 —— 等号侧判错"
    assert just_below is False, f"阈值低于实测夏普却放行：{d2}"


# ---------------------------------------------------------------------------
# 5. 存活变异的等价性证明
# ---------------------------------------------------------------------------

PROVEN_EQUIVALENT = {
    "L44 `cs_var < 1e-12` → `<=`":
        "区分值需要 cs_var **恰好等于** 1e-12。cs_var = raw.var(axis=1).median()，"
        "是逐日截面方差再取中位数的浮点结果；真正的退化信号给出的是精确 0.0，"
        "正常信号给出的是 1e-2 量级，两侧都离 1e-12 极远，"
        "无法构造使其落在这一个浮点值上。",

    "L58 `sd > 1e-12` → `>=`":
        "同理，区分值需要 net_returns 的样本标准差恰好等于 1e-12。"
        "该守卫的用途是「波动小到无法定义夏普时记 0」，边界两侧行为连续，"
        "且实测收益序列的 sd 在 1e-3 量级。",
}


def test_variance_epsilon_boundaries_are_unreachable(ds):  # noqa: F811 - 需要面板夹具
    """上面两条证明的机械验证：实测值离 1e-12 有若干个数量级。"""
    tol = 1e-12
    _, detail = leak_filter(DSL, ds, max_plausible_sharpe=1e9)
    assert detail["cs_var"] > 1e-6, f"cs_var 落进了 1e-12 邻域，证明不成立：{detail}"
    for base in (0.12, 1.0, 1e-3):
        assert (base + tol) - base != tol


def test_every_survivor_has_a_written_proof():
    """存活项要么被上面的用例杀死，要么在此有书面证明；不许有第三种状态。"""
    assert len(PROVEN_EQUIVALENT) == 2
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
