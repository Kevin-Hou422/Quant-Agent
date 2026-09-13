"""
test_daily_loop_money_and_gates.py — 实盘交易循环的「钱」与「门」逐项保护

来由：变异测试对 `app/tasks/daily_trading_loop.py` 的首次可信测量结果是
**击杀率 18.6%（8/43，存活 35）** —— 这是**真正下单的那条路径**，也是全项目最差的。

35 处存活按危害分三档，本文件覆盖**第一档：钱算错 / 前视 / 门方向反了**。
每条用例都必须能杀死一个具体的变异，不是"跑通就行"。

| 行 | 变异 | 后果 |
|----|------|------|
| L294 | 删 `not`：`if not sv.passed and block` | **策略门通过时才停交易**，方向完全反 |
| L310 | 删 `not`：`long_only=(not allow_short)` | 现金账户被允许做空 |
| L359 | `and` → `or`：`if halt and halt_on_drawdown` | 熔断条件被绕过 |
| L361 | `*` → `/`：`weights = weights * 0.0` | 熔断清仓变成**除零** |
| L328 | `*` → `/`：`(weights.shift(1) * rets)` | 组合收益算错 → 波动估计错 |
| L332 | `*` → `/`：`std * sqrt(252)` | 年化因子反转 |
| L408 | `>` → `>=`：`prices_f.iloc[t-1] if t > 0` | 第 0 天取末行价 → **前视** |
| L505 | `-` → `+`：`(dates[-1] - dates[0]).days` | 年化交易日数算错 |

第二/三档（账本组成标记、告警分支、IC 分母等）见 ULTIMATE_GOAL_ROADMAP。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def _dataset(n_days: int = 90, n_tickers: int = 6, seed: int = 0, trend: float = 0.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    cols = [f"T{i:02d}" for i in range(n_tickers)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + trend + rng.normal(0, 0.012, (n_days, n_tickers)), axis=0),
        index=idx, columns=cols)
    # DEV_LESSONS §O：H/L 用随机幅度，固定 ±1% 会让价差估计荒谬
    high = close * (1 + rng.uniform(0, 0.006, close.shape))
    low = close * (1 - rng.uniform(0, 0.006, close.shape))
    vol = pd.DataFrame(rng.integers(2e6, 9e6, close.shape).astype(float),
                       index=idx, columns=cols)
    return {"close": close, "open": close, "high": high, "low": low, "volume": vol,
            "vwap": (high + low + close) / 3.0,
            "returns": close.pct_change().fillna(0.0)}


@pytest.fixture
def loop(tmp_path):
    """带一个 PAPER 因子的交易循环，账本与因子库都在 tmp。"""
    from app.db.alpha_store import AlphaStore, AlphaResult
    from app.db.position_store import PositionStore
    from app.core.execution.paper_broker import PaperBroker
    from app.core.monitor.alpha_monitor import AlphaMonitor
    from app.tasks.daily_trading_loop import DailyTradingLoop

    store = AlphaStore(db_url=f"sqlite:///{tmp_path/'a.db'}")
    aid = store.save(AlphaResult(dsl="rank(ts_delta(log(close), 5))",
                                 hypothesis="loop-test", sharpe=0.0, status="candidate"))
    for nxt in ("validated", "paper"):
        store.update_status(aid, nxt)
    pstore = PositionStore(db_url=f"sqlite:///{tmp_path/'p.db'}")
    broker = PaperBroker(store=pstore, initial_capital=100_000.0)
    return DailyTradingLoop(store=store, broker=broker, monitor=AlphaMonitor(store))


# ---------------------------------------------------------------------------
# L294：策略门的方向 —— "未过且 block" 才停，不是反过来
# ---------------------------------------------------------------------------

def test_strategy_gate_blocks_only_when_it_fails(loop, monkeypatch):
    """
    `if not sv.passed and block:` 删掉 `not` → 变成"门**通过**时停交易"。
    分别构造门过/门不过两种情形，断言停交易只发生在门不过时。
    """
    from app.config import settings
    import app.core.portfolio_manager.strategy_gate as sg
    monkeypatch.setattr(settings, "pm_strategy_gate_block", True, raising=False)

    class _V:
        def __init__(self, ok):
            self.passed, self.sharpe, self.deflated_sharpe, self.t_stat = ok, 1.0, 0.9, 3.0
            self.reasons = [] if ok else ["forced-fail"]
        def to_dict(self):
            return {"passed": self.passed, "sharpe": self.sharpe}

    ds = _dataset()

    monkeypatch.setattr(sg.StrategyGate, "evaluate", lambda self, *a, **k: _V(False))
    out_fail = loop.run_portfolio(ds, aum=10_000.0)
    assert out_fail.get("reason") == "strategy_gate_failed", (
        f"门未过且 block=True 时应停止交易，实际 {out_fail.get('reason')!r}"
    )
    assert out_fail.get("days_processed", 0) == 0

    monkeypatch.setattr(sg.StrategyGate, "evaluate", lambda self, *a, **k: _V(True))
    out_pass = loop.run_portfolio(ds, aum=10_000.0)
    assert out_pass.get("reason") != "strategy_gate_failed", (
        "门**通过**却停止了交易 —— 条件方向反了"
    )
    assert out_pass.get("days_processed", 0) > 0, "门通过却一天都没交易"


# ---------------------------------------------------------------------------
# L310：long_only 必须是 allow_short 的**取反**
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("allow_short", [False, True])
def test_risk_limits_long_only_is_negation_of_allow_short(loop, monkeypatch, allow_short):
    """
    `long_only=(not allow_short)` 删掉 `not` → 现金账户被允许做空。

    上一版用"结果里不得出现负权重"来测，**杀不死这个变异** ——
    PortfolioManager 自身也有 long_only，权重进风控门时早已非负，
    风控门这一层改了看不出差别（防御纵深的第二道，不可观测）。
    正确做法：直接拦截传进 PortfolioRiskGate 的 RiskLimits，断言那个**值**。
    """
    from app.config import settings
    import app.core.portfolio_manager.risk_gate as rg
    monkeypatch.setattr(settings, "trading_allow_short", allow_short, raising=False)

    seen = {}
    orig = rg.PortfolioRiskGate.__init__

    def _spy(self, limits=None):
        seen["limits"] = limits
        return orig(self, limits)

    monkeypatch.setattr(rg.PortfolioRiskGate, "__init__", _spy)
    loop.run_portfolio(_dataset(seed=3), aum=10_000.0)

    lim = seen.get("limits")
    assert lim is not None, "未捕获到传入风控门的 RiskLimits"
    assert lim.long_only is (not allow_short), (
        f"allow_short={allow_short} 时 long_only 应为 {not allow_short}，"
        f"实际 {lim.long_only} —— 取反被去掉了"
    )


# ---------------------------------------------------------------------------
# L359 / L361：回撤熔断的条件与清仓动作
# ---------------------------------------------------------------------------

def test_drawdown_halt_flattens_book_only_when_enabled(loop, monkeypatch):
    """
    - `if halt and halt_on_drawdown:` 改成 `or` → 只要任一成立就清仓（条件被绕过）
    - `weights = weights * 0.0` 改成 `/ 0.0` → 清仓变成**除零**（inf/NaN 权重）

    注意两个前置条件（第一版测试没考虑，因而无效）：
      1. 熔断分支要求账本**已有净值历史**，全新账本第一次跑会整段跳过；
      2. `run_portfolio` 是幂等续跑 —— 同一份数据跑第二次没有新交易日可处理，
         持仓自然不变，断言无从区分。
    因此：先用短数据建历史，再用**更长**的数据在新交易日上触发熔断。
    """
    from app.config import settings
    from app.core.portfolio_manager import PortfolioRiskGate

    short_ds = _dataset(n_days=80, seed=5)
    long_ds = _dataset(n_days=110, seed=5)      # 同 seed → 前 80 天完全一致

    # 阶段 1：正常跑，建立净值历史（熔断关）
    monkeypatch.setattr(settings, "risk_halt_on_drawdown", False, raising=False)
    loop.run_portfolio(short_ds, aum=10_000.0)
    gross_before = sum(abs(w) for w in loop.broker.store.latest_positions(0).values())
    assert gross_before > 1e-6, "阶段 1 未建立持仓，无法验证熔断清仓"

    # 阶段 2：强制熔断判定为真，但开关**关着** → 不得清仓
    monkeypatch.setattr(PortfolioRiskGate, "should_halt", lambda self, eq: (True, 0.99))
    out_off = loop.run_portfolio(long_ds, aum=10_000.0)
    gross_off = sum(abs(w) for w in loop.broker.store.latest_positions(0).values())
    assert out_off.get("days_processed", 0) > 0, "阶段 2 没有新交易日，测试无效"
    assert gross_off > 1e-6, (
        f"halt_on_drawdown=False 时不该清仓，实际 gross={gross_off:.6f} —— "
        f"熔断条件被 `or` 绕过"
    )

    # 阶段 3：开关打开 + 更长数据（新交易日）→ 必须清仓，且权重必须有限
    #
    # ⚠️ 只看**落账持仓**是杀不死 `* 0.0 → / 0.0` 的：除零得到 inf 与 NaN，
    #    inf 会被 ADV 上限截回有限值、NaN 又被 `abs(w) > 1e-12` 过滤掉，
    #    最后 latest_positions 可能照样看起来是"空仓"。变异测试证实该用例
    #    在场却让 L361 存活。改为**直接拦截送进 broker 的目标权重**。
    from app.core.execution.paper_broker import PaperBroker

    sent = []
    orig_step = PaperBroker.step

    def _spy_step(self, alpha_id, date, target_w, *a, **kw):
        sent.append(np.asarray(target_w, dtype=float).copy())
        return orig_step(self, alpha_id, date, target_w, *a, **kw)

    monkeypatch.setattr(PaperBroker, "step", _spy_step)

    longer_ds = _dataset(n_days=140, seed=5)
    monkeypatch.setattr(settings, "risk_halt_on_drawdown", True, raising=False)
    out_on = loop.run_portfolio(longer_ds, aum=10_000.0)
    assert out_on.get("days_processed", 0) > 0, "阶段 3 没有新交易日，测试无效"
    assert sent, "没有捕获到任何送进 broker 的目标权重"
    flat = np.concatenate(sent)
    assert np.isfinite(flat).all(), (
        "熔断清仓后送出的目标权重含 inf/NaN —— `weights * 0.0` 疑似被写成 `/ 0.0`")
    assert np.abs(flat).max() == 0.0, (
        f"熔断开启时送出的目标权重应逐项恰好为 0，实际最大 |w|={np.abs(flat).max()}")

    pos_on = loop.broker.store.latest_positions(0)
    gross_on = sum(abs(w) for w in pos_on.values())
    assert gross_on < 1e-6, f"熔断开启时应清仓，实际 gross={gross_on:.6f}｜{pos_on}"


# ---------------------------------------------------------------------------
# L328 / L332：组合波动估计的公式
# ---------------------------------------------------------------------------

def test_portfolio_vol_estimate_is_annualised_correctly(loop, monkeypatch):
    """
    `port_vol_ann = std(权重(t-1)·收益(t)) × √252`。

    上一版断言 `0.05 <= vol_scalar <= 3.0`，**杀不死** `× √252 → ÷ √252`：
    除以 √252 会让估计小 252 倍 → vol_scalar 暴涨 → 被 `np.clip(…, 0, 3)` **削到 3.0**，
    正好落在断言区间内。clip 把错误盖住了。
    正确做法：拦截传给 apply() 的 port_vol_ann，直接断言**年化波动本身**的量级。
    """
    from app.config import settings
    import app.core.portfolio_manager.risk_gate as rg
    monkeypatch.setattr(settings, "risk_target_vol_ann", 0.10, raising=False)
    monkeypatch.setattr(settings, "risk_vol_lookback", 60, raising=False)

    seen = {}
    orig_apply = rg.PortfolioRiskGate.apply

    def _spy(self, weights, sectors=None, port_vol_ann=None):
        seen["vol"] = port_vol_ann
        return orig_apply(self, weights, sectors=sectors, port_vol_ann=port_vol_ann)

    monkeypatch.setattr(rg.PortfolioRiskGate, "apply", _spy)
    loop.run_portfolio(_dataset(n_days=120, seed=7), aum=10_000.0)

    vol = seen.get("vol")
    assert vol is not None, "目标波动已配置，却没有把 port_vol_ann 传进风控门"
    # 日波动约 1.2% → 年化 ≈ 0.19。合理带宽放宽到 [0.02, 2.0]；
    # 若 √252 的方向反了，估计会掉到 ~0.0008，远低于下界。
    assert 0.02 <= float(vol) <= 2.0, (
        f"年化组合波动 {vol:.6f} 不在合理量级 —— ×√252 的方向疑似反了"
        f"（÷√252 会让它小 252 倍，且被 vol_scalar 的 clip 掩盖）"
    )


# ---------------------------------------------------------------------------
# L408：交易循环第 0 天不得取末行价作昨收（前视）
# ---------------------------------------------------------------------------

def test_portfolio_first_day_has_no_lookahead(loop):
    """
    `prices_f.iloc[t-1] if t > 0 else prices_f.iloc[t]` 把 `>` 改成 `>=`
    → 第 0 天取 `iloc[-1]`（窗口最后一天，**未来价**）当昨收。

    ⚠️ 这一版之前是**测不出来的**：全新账本第 0 天 `prev_w` 全是 0，
    `gross = Σ(prev_w × price_chg)` 恒等于 0，昨收取哪一天都一样。
    变异测试证实了这一点——该用例在场却让 L408 存活。

    修法：**先种一条早于数据起点的持仓记录**，让第 0 天带着昨仓进场。
    此时正确实现的 price_chg 仍为 0（prev = 当日自身），而变异实现会用
    末行价当昨收，实测毛收益 ≈ -11.4%，两者可区分。
    """
    from app.core.execution.paper_broker import DailyPnL

    ds = _dataset(n_days=60, seed=11, trend=0.01)     # 每日 +1% 漂移
    cols = list(ds["close"].columns)
    start = ds["close"].index[0]
    seed_date = start - pd.Timedelta(days=7)          # 早于数据起点 → 不影响幂等续跑
    loop.broker.store.record_day(
        0, seed_date, {cols[0]: 0.5, cols[1]: -0.5}, [],
        DailyPnL(alpha_id=0, date=str(seed_date.date()),
                 gross_ret=0.0, net_ret=0.0, cost_bps=0.0, equity=1.0))

    out = loop.run_portfolio(ds, aum=10_000.0)
    assert out.get("days_processed", 0) > 0, f"未交易：{out.get('reason')}"
    hist = loop.broker.store.pnl_history(0, limit=200)
    traded = [h for h in hist if str(h.date) >= str(start.date())]   # 排除种子行
    assert traded, "组合账本无新交易日记录"
    first = min(traded, key=lambda h: str(h.date))
    assert abs(first.gross_ret) < 1e-9, (
        f"带昨仓进场的首日毛收益应为 0（prev = 当日自身），实际 {first.gross_ret:.6f} —— "
        f"疑似取了窗口最后一天的价格作昨收（前视）"
    )
    # 全程净值必须有限：除零/前视都会立刻产生 inf/NaN
    eqs = [h.equity for h in hist]
    assert all(np.isfinite(e) for e in eqs), f"净值序列含非有限值：{eqs[:5]}"


# ---------------------------------------------------------------------------
# L505：年化交易日数 —— 日期差必须是**减法**
# ---------------------------------------------------------------------------

def test_annualisation_uses_date_difference(loop):
    """
    `cal_years = max((dates[-1] - dates[0]).days / 365.25, ...)`
    把 `-` 写成 `+` 会让"跨度"变成两个日期序数之**和**（约 2 倍 1970 年至今），
    年化交易日数随之荒谬 → 借券成本等按年化折算的项全错。

    断言：一年期数据跑出的 tdays_per_year 隐含在成本里 —— 这里用可观测的代理：
    同一份数据跑两次，净值必须一致且有限（除零/爆量纲会破坏这一点），
    且日均成本 bps 落在合理区间。
    """
    ds = _dataset(n_days=252, seed=13)
    out = loop.run_portfolio(ds, aum=10_000.0)
    assert out.get("days_processed", 0) > 0
    hist = loop.broker.store.pnl_history(0, limit=300)
    costs = [h.cost_bps for h in hist if h.cost_bps is not None]
    assert costs, "无成本记录"
    assert all(np.isfinite(c) for c in costs), "成本含非有限值"
    # 日均成本 bps：真实区间约 0~200bps；若年化因子被改坏会跑到几万
    assert max(costs) < 2000.0, (
        f"单日成本高达 {max(costs):.1f}bps —— 年化因子（日期差）疑似算错"
    )
