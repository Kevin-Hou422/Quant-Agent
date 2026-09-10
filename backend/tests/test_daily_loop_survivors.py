"""
test_daily_loop_survivors.py — 把 `daily_trading_loop.py` 剩余存活变异**逐条**处理完

上一批（test_daily_loop_money_and_gates.py）只覆盖了"第一档"8 处，把另外 27 处
列进待办就收工了 —— 那仍然是做一半。本文件把**全部剩余项**走完：
每一处要么补断言杀死，要么在 EQUIVALENT_MUTANTS 里**给出证明**，不留待办。

覆盖的纯函数（一次钉死 4 处变异）：
  _as_date       L37   `isinstance(v,_date) and not hasattr(v,"date")`
  _cs_spearman   L556  `mask.sum() < 3`
                 L561  `sqrt((ra**2).sum() * (rb**2).sum())`
                 L562  `if denom > 0`

覆盖的流程分支：账本来源标记、active 配置过滤、边际选择启用条件、
IC 记录条件、幂等续跑、衰减告警方向、broker 选择。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.tasks.daily_trading_loop import (
    DailyTradingLoop, PORTFOLIO_BOOK_ID, _as_date, _average_ranks, _cs_spearman,
)


# ===========================================================================
# 已证明的**等价变异** —— 不是漏测，是改了也不可能被观测到
# ===========================================================================

# 口径：63 个变异点 / 杀死 41 / 存活 22 / 击杀率 65.1%（2026-09-09 重测，
# 工具修好算术变异器与注释屏蔽之后。此前的 66.0% 与 18.6% 均作废，
# 它们的分母只有 43 且 `*` `+` `-` 三个变异器从未生效——见 MUTATION_LEDGER 工具缺陷史。）
#
# 上一版这里写着 "L185/L494 Executor(validate=False) → True 等价"，**判错了**：
# 打开静态校验不只是"非法 DSL 更早报错"，`WindowValidator`（窗口 > 252）和
# `DepthValidator`（嵌套 > 10）会把**能正常执行、也没有泄漏**的因子一并拒掉，
# 在 run_portfolio 里表现为该因子被剔出组合、退回基准库。已改为写用例杀死。

# 下面这些是 `getattr(settings, "X", 默认值)` 的**默认值**变异。它们只在配置项
# 缺失时可达；只要字段在 Settings 里声明过，默认值就是死代码。
# 不用文字断言这一点 —— test_settings_fields_backing_getattr_defaults_all_exist
# 逐个字段机械验证，谁改了字段名它立刻红。
GETATTR_DEFAULT_FIELDS = [
    "pm_marginal_selection",      # L231 True  -> False
    "trading_allow_short",        # L266/L310/L430 False -> True
    "pm_strategy_gate_eval",      # L276 True  -> False
    "pm_strategy_gate_block",     # L294 False -> True
    "risk_halt_on_drawdown",      # L359 False -> True
]

EQUIVALENT_MUTANTS = {
    "L231/L266/L276/L294/L310/L359/L430  getattr(settings, X, 默认值) 的默认值":
        "第三参仅在配置项**缺失**时生效。GETATTR_DEFAULT_FIELDS 里的 5 个字段"
        "都在 Settings 中声明，默认值分支不可达 —— 由 "
        "test_settings_fields_backing_getattr_defaults_all_exist 机械验证，"
        "字段被改名/删除时该用例立刻变红。",

    "L411/L525  `if t > 0:`（IC 记录）→ `>= 0`":
        "第 0 天即使进入分支也拿不到 IC：`ret_df = prices.pct_change()` 的**第 0 行恒为 NaN**，"
        "`_cs_spearman` 的 mask 全假 → 有效对 0 < 3 → 返回 NaN → "
        "被 `if not np.isnan(ic)` 拦下，不会写库。两侧可观测行为完全一致。"
        "见 test_ic_at_index_zero_is_nan_regardless_of_the_guard。",

    "L565  `while i < len(x):`（_average_ranks 主循环）→ `<=`":
        "多出的那一轮里 i == len(x)：内层 `while j+1 < len(x)` 立即为假，"
        "`ranks[order[len:len+1]]` 是**空切片**赋值（合法且无副作用），"
        "随后 i = len+1 退出。输出数组逐元素相同。"
        "见 test_average_ranks_loop_bound_mutation_is_equivalent。",
}


def test_equivalent_mutants_are_documented():
    """等价变异必须**逐条写明理由**，否则就是拿'等价'当借口跳过。"""
    assert len(EQUIVALENT_MUTANTS) == 3
    for k, why in EQUIVALENT_MUTANTS.items():
        assert len(why) > 30, f"{k} 的等价性说明太薄弱：{why}"


def test_settings_fields_backing_getattr_defaults_all_exist():
    """
    把"默认值不可达"这句话变成**检查**。
    任何一个字段被改名或删除，`getattr` 就会悄悄改用第三参 —— 而这些第三参
    有几个是**放行方向**的（block=False、halt=False），静默生效等于门被关掉。
    """
    from app.config import Settings, settings

    declared = set(getattr(Settings, "model_fields", None) or Settings.__fields__)
    missing = [f for f in GETATTR_DEFAULT_FIELDS if f not in declared]
    assert not missing, (
        f"以下字段不在 Settings 声明里，daily_trading_loop 的 getattr 默认值会悄悄生效：{missing}")
    for f in GETATTR_DEFAULT_FIELDS:
        assert hasattr(settings, f), f"settings 实例上没有 {f}"


# ===========================================================================
# 纯函数：_as_date（L37）
# ===========================================================================

def test_as_date_returns_plain_date_for_timestamp():
    """
    L37 `isinstance(v, _date) and not hasattr(v, "date")` 删掉 not：
    `pd.Timestamp` 同时满足 isinstance(date) 与 hasattr('date')，
    变异后会**原样返回 Timestamp**（而非 date），下游日期比较随之出错。
    """
    from datetime import date as _date
    out = _as_date(pd.Timestamp("2024-03-05 15:30"))
    assert type(out) is _date, f"应返回纯 date，实际 {type(out).__name__}"
    assert out == _date(2024, 3, 5)


@pytest.mark.parametrize("v", ["2024-03-05", pd.Timestamp("2024-03-05")])
def test_as_date_normalises_all_input_forms(v):
    from datetime import date as _date
    assert _as_date(v) == _date(2024, 3, 5)


# ===========================================================================
# 纯函数：_cs_spearman（L556 / L561 / L562）
# ===========================================================================

def test_spearman_exact_value_pins_the_denominator():
    """
    L561 `sqrt((ra**2).sum() * (rb**2).sum())` 把 `*` 改成 `/`：
    相关系数的**归一化分母**错掉，结果不再落在 [-1,1]。
    用完全同序/完全反序两个精确值钉死。
    """
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(_cs_spearman(a, a) - 1.0) < 1e-12, "完全同序应为 +1"
    assert abs(_cs_spearman(a, a[::-1]) + 1.0) < 1e-12, "完全反序应为 -1"


def test_spearman_requires_at_least_three_valid_pairs():
    """L556 `mask.sum() < 3` → `<=`：恰好 3 对时必须**能算**，不能返回 NaN。"""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 1.0, 2.0])
    assert not np.isnan(_cs_spearman(a, b)), "恰好 3 对有效值时不应返回 NaN"
    two = np.array([1.0, 2.0, np.nan])
    assert np.isnan(_cs_spearman(two, two)), "只有 2 对有效值时应返回 NaN"


def test_spearman_returns_nan_when_denominator_is_zero():
    """L562 `if denom > 0` → `>=`：分母为 0 时若不拦，会变成 0/0。"""
    const = np.array([5.0, 5.0, 5.0, 5.0])
    out = _cs_spearman(const, np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.isnan(out), f"常数序列无秩差，应返回 NaN，实际 {out}"


# ===========================================================================
# 流程分支
# ===========================================================================

def _dataset(n_days=90, n_tickers=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    cols = [f"T{i:02d}" for i in range(n_tickers)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.012, (n_days, n_tickers)), axis=0),
        index=idx, columns=cols)
    high = close * (1 + rng.uniform(0, 0.006, close.shape))
    low = close * (1 - rng.uniform(0, 0.006, close.shape))
    vol = pd.DataFrame(rng.integers(2e6, 9e6, close.shape).astype(float),
                       index=idx, columns=cols)
    return {"close": close, "open": close, "high": high, "low": low, "volume": vol,
            "vwap": (high + low + close) / 3.0,
            "returns": close.pct_change().fillna(0.0)}


def _make_loop(tmp_path, dsls=("rank(ts_delta(log(close), 5))",), tag="x"):
    from app.db.alpha_store import AlphaStore, AlphaResult
    from app.db.position_store import PositionStore
    from app.core.execution.paper_broker import PaperBroker
    from app.core.monitor.alpha_monitor import AlphaMonitor

    store = AlphaStore(db_url=f"sqlite:///{tmp_path/f'a_{tag}.db'}")
    ids = []
    for d in dsls:
        aid = store.save(AlphaResult(dsl=d, hypothesis="surv", sharpe=0.0, status="candidate"))
        for nxt in ("validated", "paper"):
            store.update_status(aid, nxt)
        ids.append(aid)
    pstore = PositionStore(db_url=f"sqlite:///{tmp_path/f'p_{tag}.db'}")
    broker = PaperBroker(store=pstore, initial_capital=100_000.0)
    return DailyTradingLoop(store=store, broker=broker, monitor=AlphaMonitor(store)), ids


# ---- L193 / L203 / L210：账本来源标记与"无因子"的理由串 ----------------------

def test_baseline_fallback_is_flagged(tmp_path):
    """
    无 PAPER/ACTIVE 因子时回退经典基准库，`used_baseline` 必须为 True（L193），
    且该情形下不得进入 active-配置过滤分支（L210 的 `not`）。
    """
    loop, _ = _make_loop(tmp_path, dsls=(), tag="base")
    out = loop.run_portfolio(_dataset(), aum=10_000.0)
    assert out.get("used_baseline") is True, (
        f"无自研因子时应标记 used_baseline=True，实际 {out.get('used_baseline')}")
    assert out.get("days_processed", 0) > 0, "基准回退后仍应能交易"


def test_no_factor_reason_string_is_correct(tmp_path, monkeypatch):
    """L203：无 recs 时理由必须是 'no active factors'，不是 'no valid signals'。"""
    import app.core.strategies as strat
    monkeypatch.setattr(strat, "baseline_signals", lambda ds: {})   # 连基准也没有
    loop, _ = _make_loop(tmp_path, dsls=(), tag="noreason")
    out = loop.run_portfolio(_dataset(), aum=10_000.0)
    assert out.get("reason") == "no active factors", (
        f"无任何因子时理由应为 'no active factors'，实际 {out.get('reason')!r}")


# ---- L215：active 策略配置必须真的被用来过滤成分 ----------------------------

def test_active_strategy_config_filters_components(tmp_path, monkeypatch):
    """
    PM.7 过滤链上一次挂着三个变异，这一版三个都要杀：
      L210 `if not used_baseline:`   删 not → 非基准路径反而跳过过滤
      L215 `if active is not None:`  删 not → 有 active 配置时反而不过滤
      L230 `not used_baseline and …` 见 test_marginal_selection_runs_when_enabled

    ⚠️ 上一版断言是 `active_config == sid **or** n_factors == 1`，
    这个 `or` 让它测不出东西：即使过滤根本没发生，PM.S2 边际准入也可能
    自己把成分收敛到 1 个，右半边照样成立。变异测试证实三个变异全部存活。
    改为：关掉边际准入以隔离变量，再**同时**断言配置 id 与成分集合。
    """
    from app.config import settings
    from app.db.strategy_store import StrategyStore, StrategyConfig

    monkeypatch.setattr(settings, "pm_marginal_selection", False, raising=False)
    loop, ids = _make_loop(
        tmp_path, dsls=("rank(ts_delta(log(close), 5))", "zscore(ts_mean(returns, 10))"),
        tag="active")
    keep = str(ids[0])

    sstore = StrategyStore(db_url=f"sqlite:///{tmp_path/'s.db'}")
    sid = sstore.save(StrategyConfig(factors=[keep], combo_weights={keep: 1.0},
                                     aum=10_000.0, passed=True, status="proposed",
                                     name="active-filter-test"))
    sstore.update_status(sid, "approved")
    sstore.update_status(sid, "active")
    # 直接替换 latest_active，避免依赖 StrategyStore 的默认 db 路径
    monkeypatch.setattr(StrategyStore, "latest_active", lambda self: sstore.get(sid))

    out = loop.run_portfolio(_dataset(), aum=10_000.0)
    assert out.get("active_config") == sid, (
        f"有 active 配置时必须记录它的 id，实际 active_config={out.get('active_config')!r} —— "
        f"过滤分支没有执行")
    assert out.get("n_factors") == 1, (
        f"应只交易配置里的 1 个成分，实际 n_factors={out.get('n_factors')}")
    assert set(out.get("combo_weights") or {}) == {keep}, (
        f"交易的成分与 active 配置不一致：{sorted((out.get('combo_weights') or {}))} != [{keep}]")


def test_baseline_book_skips_active_config_and_strategy_gate(tmp_path, monkeypatch):
    """
    L210 的**另一侧**，以及 L276 的 `and` → `or`：

      `if not used_baseline:`                       ← 基准账本不走 active 配置过滤
      `if pm_strategy_gate_eval and not used_baseline:` ← 基准账本不跑策略门

    把 `and` 改成 `or` 后，基准库回退时也会去跑策略门并写出 verdict ——
    而策略门评的是"我们自研策略配得上真钱吗"，给经典基准打分毫无意义，
    还会把一个不该存在的 verdict 塞进日报。
    """
    from app.db.strategy_store import StrategyStore, StrategyConfig

    sstore = StrategyStore(db_url=f"sqlite:///{tmp_path/'sb.db'}")
    sid = sstore.save(StrategyConfig(factors=["999"], combo_weights={"999": 1.0},
                                     aum=10_000.0, passed=True, status="proposed",
                                     name="should-be-ignored"))
    sstore.update_status(sid, "approved")
    sstore.update_status(sid, "active")
    monkeypatch.setattr(StrategyStore, "latest_active", lambda self: sstore.get(sid))

    loop, _ = _make_loop(tmp_path, dsls=(), tag="basegate")     # 无自研因子 → 基准回退
    out = loop.run_portfolio(_dataset(), aum=10_000.0)

    assert out.get("used_baseline") is True, "前置条件不成立：没有走基准回退"
    assert out.get("days_processed", 0) > 0, "基准回退后应当仍能交易"
    assert out.get("active_config") is None, (
        f"基准账本不该套用 active 策略配置，实际 active_config={out.get('active_config')!r}")
    assert out.get("strategy_verdict") is None, (
        "基准账本不该跑策略门 —— `and` 疑似被写成 `or`，"
        f"实际 verdict={out.get('strategy_verdict')!r}")


# ---- L230 / L231：边际选择的启用条件 ---------------------------------------

def test_marginal_selection_respects_its_switch(tmp_path, monkeypatch):
    """
    `len(signals) > 1 and pm_marginal_selection` ——
    L231 把 `and` 改成 `or` → 开关关掉也会跑边际选择。
    """
    from app.config import settings
    loop, _ = _make_loop(
        tmp_path, dsls=("rank(ts_delta(log(close), 5))", "zscore(ts_mean(returns, 10))"),
        tag="marg")
    monkeypatch.setattr(settings, "pm_marginal_selection", False, raising=False)
    out = loop.run_portfolio(_dataset(), aum=10_000.0)
    assert out.get("selection") in (None, {}), (
        f"开关关闭时不应执行边际选择，实际 selection={out.get('selection')!r}")


def test_marginal_selection_runs_when_enabled(tmp_path, monkeypatch):
    """
    L230 `if (not used_baseline and using_active_config is None and len(signals) > 1 …)`
    删掉 `not` → 自研多因子时**跳过**边际准入（全纳入），只有基准回退才跑。

    上面两条用例都是"该关的时候关掉了"，全是**否定**断言，
    因此把整个分支永久关掉也照样绿。缺的正是这条**肯定**断言。
    """
    from app.config import settings
    monkeypatch.setattr(settings, "pm_marginal_selection", True, raising=False)
    loop, _ = _make_loop(
        tmp_path, dsls=("rank(ts_delta(log(close), 5))", "zscore(ts_mean(returns, 10))"),
        tag="margon")
    out = loop.run_portfolio(_dataset(), aum=10_000.0)
    sel = out.get("selection")
    assert sel, ("开关打开且有 2 个自研因子时必须执行边际准入并记录轨迹，"
                 f"实际 selection={sel!r}")
    assert sel.get("selected"), f"边际准入没有选出任何成分：{sel}"


def test_marginal_selection_needs_more_than_one_factor(tmp_path, monkeypatch):
    """L230 `len(signals) > 1` → `>= 1`：单因子时不该跑边际选择。"""
    from app.config import settings
    monkeypatch.setattr(settings, "pm_marginal_selection", True, raising=False)
    loop, _ = _make_loop(tmp_path, dsls=("rank(ts_delta(log(close), 5))",), tag="one")
    out = loop.run_portfolio(_dataset(), aum=10_000.0)
    assert out.get("selection") in (None, {}), (
        f"只有 1 个因子时不应执行边际选择，实际 {out.get('selection')!r}")


# ---- L333：波动估计的有效性判定方向 ----------------------------------------

def test_valid_vol_estimate_is_used_not_discarded(tmp_path, monkeypatch):
    """
    `if not np.isfinite(port_vol_ann) or port_vol_ann <= 1e-9: port_vol_ann = None`
    删掉 `not` → **有效**的估计反而被丢弃 → 目标波动永不生效。
    """
    from app.config import settings
    import app.core.portfolio_manager.risk_gate as rg
    monkeypatch.setattr(settings, "risk_target_vol_ann", 0.10, raising=False)
    monkeypatch.setattr(settings, "risk_vol_lookback", 60, raising=False)
    loop, _ = _make_loop(tmp_path, tag="vol")

    seen = {}
    orig = rg.PortfolioRiskGate.apply

    def _spy(self, weights, sectors=None, port_vol_ann=None):
        seen["vol"] = port_vol_ann
        return orig(self, weights, sectors=sectors, port_vol_ann=port_vol_ann)

    monkeypatch.setattr(rg.PortfolioRiskGate, "apply", _spy)
    loop.run_portfolio(_dataset(n_days=120, seed=7), aum=10_000.0)
    assert seen.get("vol") is not None, (
        "样本充足且估计有效，port_vol_ann 却是 None —— 有效性判定的方向反了")


# ---- L411 / L413 / L525 / L527：IC 记录的条件 -------------------------------

def test_first_trading_day_has_no_ic(tmp_path):
    """
    - L411/L525 `if t > 0` → `>= 0`：第 0 天会用 comp_arr[-1]（**未来**信号）算 IC
    - L413/L527 `if not np.isnan(ic)` → 删 not：只记录 NaN → 一条都记不下

    第一版断言 `len(ics) == days - 1`，**是错的**：信号有 burn-in 期
    （ts_delta 窗口 + delay），那几天 IC 本就是 NaN 不入库，条数必然更少。
    这个变异的要害是"**首个交易日不该有 IC**"（它没有昨日信号），直接断言这一点。
    """
    loop, _ = _make_loop(tmp_path, tag="ic")
    out = loop.run_portfolio(_dataset(n_days=40, seed=9), aum=10_000.0)
    assert out.get("days_processed", 0) > 1, f"交易日不足：{out.get('days_processed')}"

    ics = loop.store.get_ic_history(PORTFOLIO_BOOK_ID)
    assert ics, "一条 realized IC 都没记录 —— `if not np.isnan(ic)` 的方向疑似反了"
    assert all(not np.isnan(h.realized_ic) for h in ics), "记录了 NaN IC"

    pnls = loop.broker.store.pnl_history(PORTFOLIO_BOOK_ID, limit=500)
    first_trade_day = min(str(h.date) for h in pnls)
    first_ic_day = min(str(h.date) for h in ics)
    assert first_ic_day > first_trade_day, (
        f"首个交易日 {first_trade_day} 也记了 IC（{first_ic_day}）—— "
        f"它没有昨日信号，只能是用了未来信号（`t >= 0`）")


# ---- L512：幂等续跑 ---------------------------------------------------------

def test_replay_is_idempotent(tmp_path):
    """
    `if last is not None and d.date() <= last: continue` 删掉 not →
    幂等判据失效，同一天会被**重复记账**。
    """
    loop, _ = _make_loop(tmp_path, tag="idem")
    ds = _dataset(n_days=50, seed=4)
    r1 = loop.run(ds)
    n1 = r1.results[0].days_processed if r1.results else 0
    assert n1 > 0, "首轮未处理任何交易日"
    r2 = loop.run(ds)
    n2 = r2.results[0].days_processed if r2.results else 0
    assert n2 == 0, f"同一份数据第二次跑应处理 0 天（幂等），实际 {n2} 天"


# ---- L441 / L536 / L53 / L537：衰减告警的方向与默认值 -------------------------

@pytest.mark.parametrize("has_alert", [False, True])
def test_decay_alert_flag_follows_the_monitor(tmp_path, monkeypatch, has_alert):
    """
    - L53  `decay_alert: bool = False` → True：所有结果默认带告警
    - L441/L536 `if alert is not None` → 删 not：有/无告警的分支反转
    - L537 `res.decay_alert = True` → False：真有衰减也不标记

    第一版用"短随机序列不会触发衰减"来测，**假设是错的** ——
    `check_decay` 在连续负 IC 达阈值时本来就会触发，随机数据上完全可能发生。
    改为**直接控制** monitor 的返回值，正反两面都断言。
    """
    from app.core.monitor.alpha_monitor import AlphaMonitor, DecayAlert

    alert = DecayAlert(alpha_id=1, reason="consecutive_negative",
                       consecutive_neg=5, rolling_mean_ic=-0.03) if has_alert else None
    monkeypatch.setattr(AlphaMonitor, "check_decay", lambda self, aid, _ics=None: alert)

    loop, _ = _make_loop(tmp_path, tag=f"decay{int(has_alert)}")
    rep = loop.run(_dataset(n_days=40, seed=6))
    assert rep.results, "无结果"
    got = [r.decay_alert for r in rep.results]
    assert all(g is has_alert for g in got), (
        f"monitor 返回 {'告警' if has_alert else 'None'}，decay_alert 却是 {got} —— "
        f"默认值或告警分支方向有误")
    assert rep.n_alerts == (len(rep.results) if has_alert else 0), (
        f"告警计数 {rep.n_alerts} 与实际不符")


# ---- L271：pf_broker 的选择 -------------------------------------------------

def test_portfolio_broker_uses_grounded_cost_params(tmp_path, monkeypatch):
    """
    `pf_broker = PaperBroker(..., cost_params=gp, initial_capital=aum) if gp is not None
                 else self.broker`
    删掉 not → **拿到了** grounded 成本参数时反而用回默认 broker，
    于是 TR.3 那套"按 moomoo 免佣 + Corwin-Schultz 实测价差"的成本全部落空，
    交易成本悄悄退回硬编码的机构默认值。

    ⚠️ 上一版断言 `out["aum"] == 传入的 aum`，**测不出这个分支** ——
    `aum` 是函数里的局部变量，直接原样写进返回值，跟用哪个 broker 无关。
    变异测试证实该用例在场而 L271 存活。

    改法：把 grounded 成本参数换成一个**极端可辨识**的值（fixed_bps=500），
    再看落账的 cost_bps。用了 grounded → 成本量级 ~180bps；
    退回默认 broker → 仍是 ~0.8bps，相差 200 倍。
    """
    import app.core.trading_context.context as ctx
    from app.core.backtest_engine.transaction_cost import CostParams

    aum = 10_000.0
    ds = _dataset(n_days=60, seed=8)

    # 基线：默认 grounded 成本
    base_loop, _ = _make_loop(tmp_path, tag="brk_base")
    base_loop.run_portfolio(ds, aum=aum)
    base = float(np.mean([h.cost_bps for h in
                          base_loop.broker.store.pnl_history(PORTFOLIO_BOOK_ID, limit=300)]))
    assert base < 50.0, f"基线成本已经很高（{base:.2f}bps），无法与放大后的区分"

    # 换成极端 grounded 成本参数
    monkeypatch.setattr(ctx, "grounded_cost_params",
                        lambda *a, **k: CostParams(fixed_bps=500.0, min_ticket_fee=0.0,
                                                   spread_bps=0.0, impact_coef=0.0))
    loop, _ = _make_loop(tmp_path, tag="brk_gp")
    out = loop.run_portfolio(ds, aum=aum)
    assert out.get("days_processed", 0) > 0
    got = float(np.mean([h.cost_bps for h in
                         loop.broker.store.pnl_history(PORTFOLIO_BOOK_ID, limit=300)]))
    assert got > 100.0, (
        f"grounded 成本参数没有生效：平均 cost_bps={got:.2f}（基线 {base:.2f}）—— "
        f"组合账本疑似退回了默认 broker")
    assert abs(float(out.get("aum", aum)) - aum) < 1e-6


def test_strategy_decay_alert_is_reported_in_the_portfolio_result(tmp_path, monkeypatch):
    """
    L441 `if alert is not None:` 删掉 not → **有**告警时反而不写进结果
    （`alert.reason` 在 None 上抛 AttributeError，被外层 except 吞掉，
    日报里 `strategy_decay` 永远是 None）。

    既有的 `test_decay_alert_flag_follows_the_monitor` 走的是 `run()` 的
    **因子级** decay_alert，与这里 `run_portfolio` 的**策略级** strategy_decay
    是两条独立的分支 —— 这就是 DEV_LESSONS §S 说的"审计单位错了"。
    """
    from app.core.monitor.alpha_monitor import AlphaMonitor, DecayAlert

    alert = DecayAlert(alpha_id=PORTFOLIO_BOOK_ID, reason="consecutive_negative",
                       consecutive_neg=5, rolling_mean_ic=-0.031)
    monkeypatch.setattr(AlphaMonitor, "check_decay", lambda self, aid, _ics=None: alert)
    loop, _ = _make_loop(tmp_path, tag="sdecay")
    out = loop.run_portfolio(_dataset(n_days=50, seed=2), aum=10_000.0)
    sd = out.get("strategy_decay")
    assert sd, "monitor 报了策略级衰减，日报里却没有 strategy_decay"
    assert sd.get("reason") == "consecutive_negative"
    assert sd.get("rolling_mean_ic") == pytest.approx(-0.031, abs=1e-9)

    # 反向：没有告警时不得凭空造一个
    monkeypatch.setattr(AlphaMonitor, "check_decay", lambda self, aid, _ics=None: None)
    loop2, _ = _make_loop(tmp_path, tag="sdecay0")
    out2 = loop2.run_portfolio(_dataset(n_days=50, seed=2), aum=10_000.0)
    assert out2.get("strategy_decay") is None


# ---- L185 / L494：静态校验是**故意关掉**的 -----------------------------------

def test_long_window_factor_is_not_dropped_by_static_validation(tmp_path):
    """
    `Executor(validate=False)` 里的 False 是有意的：进了 PAPER 的因子不该再被
    静态规则二次裁决。改成 True 后 `WindowValidator`（窗口 > 252）会抛
    ValidationError，在 run_portfolio 里被 `except` 吞掉 → 该因子被剔出组合
    → 无因子可用 → 退回基准库，**账本悄悄换了策略**。

    `ts_mean(close, 300)` 正是这种因子：静态校验拒，执行完全正常，也无泄漏。
    """
    loop, _ = _make_loop(tmp_path, dsls=("rank(ts_mean(close, 300))",), tag="win")
    out = loop.run_portfolio(_dataset(n_days=400, seed=3), aum=10_000.0)
    assert out.get("used_baseline") is False, (
        "长窗口因子被静态校验剔除，账本退回了基准库 —— validate 疑似被打开")
    assert out.get("n_factors") == 1, f"因子未参与组合：n_factors={out.get('n_factors')}"
    assert out.get("days_processed", 0) > 0


def test_per_factor_loop_first_day_has_no_lookahead(tmp_path):
    """
    L520 `prices_f.iloc[t-1] if t > 0 else prices_f.iloc[t]` —— 与 run_portfolio 的
    L408 **是同一个坑的第二份拷贝**，在 `_run_one_alpha`（逐因子隔离路径）里。

    我上一轮只修了 run_portfolio 那条，重测时 L520 照样存活 ——
    这正是 DEV_LESSONS §S「审计单位错了」：同一个 bug 在两个函数里各有一份，
    按"模块"审计会以为修完了。

    构造同 L408：先种一条早于数据起点的持仓，让第 0 天带着昨仓进场。
    正确实现 prev = 当日自身 → 毛收益恒为 0；变异实现取末行价 → 显著非零。
    """
    from app.core.execution.paper_broker import DailyPnL

    loop, ids = _make_loop(tmp_path, tag="la1")
    aid = ids[0]
    ds = _dataset(n_days=60, seed=11)
    cols = list(ds["close"].columns)
    start = ds["close"].index[0]
    seed_date = start - pd.Timedelta(days=7)
    loop.broker.store.record_day(
        aid, seed_date, {cols[0]: 0.5, cols[1]: -0.5}, [],
        DailyPnL(alpha_id=aid, date=str(seed_date.date()),
                 gross_ret=0.0, net_ret=0.0, cost_bps=0.0, equity=1.0))

    rep = loop.run(ds)
    assert rep.results and rep.results[0].error == "", (
        f"逐因子循环报错：{rep.results[0].error if rep.results else '无结果'}")
    assert rep.results[0].days_processed > 0, "没有处理任何交易日"

    hist = loop.broker.store.pnl_history(aid, limit=300)
    traded = [h for h in hist if str(h.date) >= str(start.date())]
    assert traded, "无新交易日记录"
    first = min(traded, key=lambda h: str(h.date))
    assert abs(first.gross_ret) < 1e-9, (
        f"带昨仓进场的首日毛收益应为 0，实际 {first.gross_ret:.6f} —— "
        f"疑似取了窗口最后一天的价格作昨收（前视）")


def test_long_window_factor_runs_in_per_factor_loop(tmp_path):
    """L494：`_run_one_alpha` 里同一个开关，走的是另一条路径（逐因子隔离）。"""
    loop, ids = _make_loop(tmp_path, dsls=("rank(ts_mean(close, 300))",), tag="win1")
    rep = loop.run(_dataset(n_days=400, seed=3))
    assert rep.results, "没有产生逐因子结果"
    res = rep.results[0]
    assert res.error == "", f"长窗口因子在逐因子循环里报错：{res.error}"
    assert res.days_processed > 0, "长窗口因子一天都没处理"


# ---- L587：秩方差为 0 时不得触发除零 ----------------------------------------

def test_spearman_zero_denominator_does_not_divide():
    """
    `float(np.dot(ra, rb) / denom) if denom > 0 else nan` 改成 `>=` 后，
    denom == 0 会走进 `0.0 / 0.0` —— 结果同样是 NaN，
    **所以用返回值断言永远杀不死它**（既有 test_spearman_returns_nan_… 就是这样）。

    差别只在：变异版每算一次常数信号就抛一个
    `RuntimeWarning: invalid value encountered in scalar divide`。
    交易循环里每天都会调它，日志会被这个警告淹没。
    把警告升级成错误即可区分。
    """
    import warnings
    const = np.array([5.0, 5.0, 5.0, 5.0])
    other = np.array([1.0, 2.0, 3.0, 4.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = _cs_spearman(const, other)
    assert np.isnan(out)
    # 两侧都为常数时同样不得除零
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert np.isnan(_cs_spearman(const, const.copy()))


# ---- 等价性证明的机械验证 ---------------------------------------------------

def test_ic_at_index_zero_is_nan_regardless_of_the_guard():
    """
    L411/L525 等价性：`ret_df = prices.pct_change()` 第 0 行恒为 NaN，
    因此第 0 天即使进了分支也拿不到可记录的 IC。
    """
    ds = _dataset(n_days=30, seed=5)
    ret0 = ds["close"].pct_change().iloc[0]
    assert ret0.isna().all(), "pct_change 第 0 行竟然不是全 NaN，等价性证明不成立"
    sig = np.arange(len(ret0), dtype=float)
    assert np.isnan(_cs_spearman(sig, ret0.to_numpy(dtype=float)))


def test_average_ranks_loop_bound_mutation_is_equivalent():
    """
    L565 等价性：把 `while i < len(x)` 改成 `<=` 只是多空转一轮
    （空切片赋值合法且无副作用）。这里把两个版本都跑一遍逐元素比对，
    而不是靠嘴说"应该一样"。
    """
    def _ranks(x, inclusive):
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(len(x), dtype=float)
        i = 0
        while (i <= len(x)) if inclusive else (i < len(x)):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            ranks[order[i:j + 1]] = 0.5 * (i + j)
            i = j + 1
        return ranks

    rng = np.random.default_rng(17)
    cases = [np.array([]), np.array([1.0]), np.array([1.0, 1.0, 1.0]),
             np.array([3.0, 1.0, 2.0, 1.0]), np.array([5.0, 4.0, 3.0, 2.0, 1.0])]
    cases += [rng.choice([0.0, 1.0, 2.0], size=n) for n in (2, 7, 20)]
    for x in cases:
        a, b = _ranks(x, False), _ranks(x, True)
        assert np.array_equal(a, b), f"输入 {x} 上两个循环边界给出不同结果：{a} vs {b}"
        if len(x):
            assert np.array_equal(a, _average_ranks(x)), "复刻实现与生产实现不一致"
