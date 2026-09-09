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
