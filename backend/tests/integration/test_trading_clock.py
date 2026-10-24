"""
信号 / 成交 / 标签 / 归档时钟：边界定义、规格里的时钟版本、开盘到开盘的标签与 IC、旧时钟历史隔离、调度时间窗、结果带时钟版本。

来源（原测试文件的说明，保留出处）：

2026-10-22 用户决定 · 时钟（D14 (a) + C3）与 Validate 预热（D9）。

D14 原话："T 收盘信息完成并过健康门后冻结信号与目标，T+1 按明确交易规则执行，更新时钟版本。同步修改信号、
回测、内部模拟、IC、归档、基准及调度。'计划开盘执行'不等于限价单必然按开盘价成交；预测 IC 与实际持仓收益
分别记录。旧历史保留旧版本，先隔离重算。"
C3 原话："目前虽然要求开盘前存档，评分仍是前收盘到今收盘，包含已经发生的隔夜收益。必须与 D14 一起明确标签
起止、信息截止和归档截止，不能只提前 deadline。"
D9 原话："只用当时已知历史计算特征，只评分 Validate 日期；拟合与标准化不能读取未来评分信息。覆盖 252 日因子、
嵌套窗口、跨界标签、初始持仓及历史不足。"

---

复审 2026-10-19 余项（不依赖待决事项的部分）：

§5.1  "尚未统一时钟前，历史结果应分版本标识。"
      → 每一类存下来的结果（研究指标、因子 IC、信号存档、策略证据、规格、提案 verdict、验证门结论、
        正式前向记录）都带时钟版本；时钟一变，规格哈希跟着变（旧证据不会被新时钟的策略继承）。
N03-b "补拉、回放、晚到归档应保留研究价值，但不能升级成正式前向样本。"
      §7.1 工作包 4："将影子信号证据与有实际成交的策略收益分开记录。"
      → 策略证据分 forward / replay 两类：没有及时存档的那几天也记下来（replay，可供研究），
        →ACTIVE 门只读 forward。
N05-d "新规格的资格应来自预登记的验证设计和此后真正未见的前向数据。"
      → 提案时登记验证设计（门槛、t 统计口径、证据口径、登记时刻，带哈希）；→ACTIVE 按**登记的**
        设计判定（之后改 settings 不影响），只算登记之后的前向证据；设计缺失 / 被改 → 不通过。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.market_calendar import session_close_utc, session_open_utc
from tests.integration.test_forward_evidence import _after_close, _panel, _upto, world  # noqa: F401

# ---------------------------------------------------------------------------
# 每日循环：T 的决策用到 T 的数据，T+1 才成交；IC 用开盘到开盘的标签
# ---------------------------------------------------------------------------

def _evidence_world(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from app.config import settings
    from app.core.execution.paper_broker import PaperBroker
    from app.db.alpha_store import AlphaResult, AlphaStore
    from app.db.evidence_store import EvidenceStore
    from app.db.position_store import PositionStore
    from app.tasks.daily_trading_loop import DailyTradingLoop
    url = f"sqlite:///{(tmp_path / 'a.db').as_posix()}"
    monkeypatch.setattr(settings, "database_url", url)
    store, ev = AlphaStore(db_url=url), EvidenceStore(url)
    aid = store.save(AlphaResult(dsl="rank(ts_delta(close, 5))", status="candidate"))
    store.update_status(aid, "validated"); store.update_status(aid, "paper")

    def loop():
        return DailyTradingLoop(store=store, evidence_store=ev, broker=PaperBroker(
            store=PositionStore(f"sqlite:///{(tmp_path / 'p.db').as_posix()}"), initial_capital=10_000.0))
    return SimpleNamespace(store=store, ev=ev, aid=aid, loop=loop, url=url)


# ---------------------------------------------------------------------------
# D9：Validate 段预热
# ---------------------------------------------------------------------------

def _long_panel(T=800, N=12, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=T)
    cols = [f"S{i}" for i in range(N)]
    c = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.015, (T, N)), 0), idx, cols)
    o = c.shift(1).fillna(c) * (1 + rng.normal(0, 0.003, (T, N)))
    amp = np.abs(rng.normal(0, 0.006, (T, N)))
    return {"open": o, "high": np.maximum(o, c) * (1 + amp), "low": np.minimum(o, c) * (1 - amp),
            "close": c, "volume": pd.DataFrame(1e7, idx, cols), "vwap": c,
            "returns": c.pct_change().fillna(0.0)}


def _propose(world, dsl="rank(ts_delta(close, 5))"):
    from app.config import settings
    from app.db.strategy_store import StrategyConfig, StrategyStore
    from tests.conftest import approved_verdict
    f = str(world.aid)
    v = approved_verdict({f: dsl}, {f: 1.0}, 10_000.0)
    return StrategyStore(db_url=settings.database_url).save(
        StrategyConfig(factors=[f], combo_weights={f: 1.0}, aum=10_000.0, status="proposed",
                       verdict=v)), v["spec_hash"]


# ---------------------------------------------------------------------------
# 时钟定义（单一来源）
# ---------------------------------------------------------------------------

class TestClockDefinition:

    def test_the_clock_states_every_boundary(self):
        from app.core import clock
        from app.core.data_engine.input_identity import CLOCK_VERSION, CLOCK_VERSIONS
        assert CLOCK_VERSION == "c2" and "c1" in CLOCK_VERSIONS and "c2" in CLOCK_VERSIONS
        d = clock.describe()
        for k in ("information_cutoff", "decision", "execution", "fill_rule", "label", "archive_deadline",
                  "valuation"):
            assert d.get(k), f"时钟没有写明 {k}"

    def test_the_label_starts_after_the_archive_deadline(self):
        """C3：信号 T 的标签 = 开盘(T+1) → 开盘(T+2)；归档截止 = 开盘(T+1) —— 标签里没有任何已发生的收益。"""
        from app.core import clock
        T = pd.Timestamp("2025-03-04")
        win = clock.label_window(T)
        assert win["start"] == session_open_utc(pd.Timestamp("2025-03-05"))
        assert win["end"] == session_open_utc(pd.Timestamp("2025-03-06"))
        assert clock.archive_deadline(T) == win["start"]
        assert clock.archive_deadline(T) > session_close_utc(T)

    def test_label_returns_are_open_to_open_aligned_to_the_signal_date(self):
        from app.core import clock
        idx = pd.bdate_range("2025-03-03", periods=5)
        o = pd.DataFrame({"A": [10., 11., 12.1, 13.31, 14.641]}, index=idx)
        c = pd.DataFrame({"A": [99., 99., 99., 99., 99.]}, index=idx)       # 收盘价不该出现在标签里
        lab = clock.label_returns({"open": o, "close": c})
        assert lab["A"].iloc[0] == pytest.approx(12.1 / 11 - 1)            # 信号 3-03 → 开 3-04 到开 3-05
        assert np.isnan(lab["A"].iloc[-1]) and np.isnan(lab["A"].iloc[-2])  # 还没发生的不给

    def test_label_returns_require_open_prices(self):
        from app.core import clock
        idx = pd.bdate_range("2025-03-03", periods=5)
        with pytest.raises(ValueError, match="open"):
            clock.label_returns({"close": pd.DataFrame({"A": 1.0}, index=idx)})


# ---------------------------------------------------------------------------
# 规格：时钟与信号参数进哈希
# ---------------------------------------------------------------------------

def test_the_spec_carries_the_current_clock_and_no_extra_signal_delay():
    from app.core.portfolio_manager.executable_spec import build_spec
    s = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e4, 0.01)
    assert s["clock"] == "c2" and s["signal"]["delay"] == 0


def test_an_old_clock_spec_no_longer_matches_the_runtime():
    """旧时钟的规格不能被新时钟悄悄执行（重新提案，不继承旧时钟证据）。"""
    from app.core.portfolio_manager.executable_spec import (
        build_spec,
        verify_against_runtime,
    )
    s = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e4, 0.01)
    old = dict(s, clock="c1", signal=dict(s["signal"], delay=1))
    diff = verify_against_runtime(old, {"1": "rank(close)"}, 1e4)
    assert any("clock" in d for d in diff) and any("signal" in d for d in diff)


def test_forward_ic_is_scored_on_the_open_to_open_label(tmp_path, monkeypatch):
    """C3：前向 IC = 存档信号 T 对 开盘(T+1)→开盘(T+2)；改动 T 收盘→T+1 开盘的隔夜价格不影响它。"""
    w = _evidence_world(tmp_path, monkeypatch)
    sid, _ = _propose(w)
    ds = _panel(T=70)
    days = ds["close"].index
    for d in days[-4:]:
        w.loop().run_portfolio(_upto(ds, d), aum=10_000.0, data_source="yahoo", now=_after_close(d))
    rows = w.ev.strategy_forward_ic(sid)
    assert rows, "没有前向证据"
    r = rows[-1]
    T = pd.Timestamp(r.signal_asof)
    i = list(days).index(T)
    assert pd.Timestamp(r.date) == days[i + 2], "标签结束日应为 T+2 的开盘"
    # 独立重算：存档的信号（库里取）× 开盘到开盘
    arch = w.ev.archived_before("strategy:%d" % sid, T, pd.Timestamp("2100-01-01").to_pydatetime())
    lab = ds["open"].iloc[i + 2] / ds["open"].iloc[i + 1] - 1
    from scipy.stats import spearmanr
    expect = spearmanr(arch["signal"].reindex(lab.index), lab).correlation
    assert r.ic == pytest.approx(expect, abs=1e-9)


# ---------------------------------------------------------------------------
# 旧时钟的历史隔离
# ---------------------------------------------------------------------------

def test_factor_monitoring_reads_only_current_clock_ic(tmp_path):
    import app.core.data_engine.input_identity as ii
    from app.db.alpha_store import AlphaResult, AlphaStore
    url = f"sqlite:///{(tmp_path / 'a.db').as_posix()}"
    s = AlphaStore(db_url=url)
    aid = s.save(AlphaResult(dsl="rank(close)", status="candidate"))
    orig = ii.CLOCK_VERSION
    try:
        ii.CLOCK_VERSION = "c1"
        s.record_ic(aid, "2025-01-02", 0.5, is_forward=True)
    finally:
        ii.CLOCK_VERSION = orig
    s.record_ic(aid, "2025-01-03", 0.1, is_forward=True)
    assert [float(r.realized_ic) for r in s.get_forward_ic(aid)] == [0.1], "旧时钟的前向 IC 混进了当前口径"
    assert len(s.get_ic_history(aid, all_clocks=True)) == 2, "旧版本要保留，只是隔离"
    # 面板接口：两个时钟的点都给（带版本），滚动统计只用当前时钟，并告诉前端当前是哪个
    from app.api.router import alpha_ic_history
    resp = alpha_ic_history(aid, limit=250, store=s)
    assert [(p.clock_version, p.realized_ic) for p in resp.points] == [("c1", 0.5), ("c2", 0.1)]
    assert resp.rolling_mean_ic == pytest.approx(0.1) and resp.current_clock == "c2"


def test_paired_returns_put_the_open_to_open_label_at_row_t_plus_1():
    from app.core.clock import label_returns, paired_returns
    ds = _long_panel(T=40)
    pr, lab = paired_returns(ds), label_returns(ds)
    pd.testing.assert_frame_equal(pr.shift(-1).iloc[:-2], lab.iloc[:-2])


@pytest.mark.parametrize("day", ["2025-01-15", "2025-07-15", "2025-03-07", "2025-03-10", "2025-11-03"])
def test_the_scheduled_trading_runs_fall_between_the_close_and_the_archive_deadline(day):
    """调度（D14）：交易循环与重试都在 T 收盘之后、开盘(T+1)（归档截止）之前 —— 冬令时 / 夏令时 / 切换前后都成立。"""
    from datetime import datetime, timezone

    from app.core import clock
    from app.tasks.scheduler import DAILY_TRADING_AT, DAILY_TRADING_RETRY_AT
    T = pd.Timestamp(day)
    close, deadline = session_close_utc(T), clock.archive_deadline(T)
    for h, m in (DAILY_TRADING_AT, DAILY_TRADING_RETRY_AT):
        run = datetime(T.year, T.month, T.day, h, m, tzinfo=timezone.utc)
        assert close < run < deadline, (day, (h, m), close, deadline)


# ---------------------------------------------------------------------------
# §5.1 时钟版本
# ---------------------------------------------------------------------------

class TestClockVersion:

    def test_research_metrics_and_factor_ic_carry_the_clock(self, world):
        from app.core.data_engine.input_identity import CLOCK_VERSION
        rec = world.store.get_by_id(world.aid)
        assert rec.clock_version == CLOCK_VERSION
        world.store.record_ic(world.aid, "2024-05-02", 0.1, is_forward=True)
        assert world.store.get_ic_history(world.aid)[-1].clock_version == CLOCK_VERSION

    def test_signal_archives_and_strategy_evidence_carry_the_clock(self, world, monkeypatch):
        from app.config import settings
        from app.core.data_engine.input_identity import CLOCK_VERSION
        monkeypatch.setattr(settings, "database_url", world.url)
        sid, _ = _propose(world)
        ds = _panel()
        days = ds["close"].index
        for d in days[-3:]:                                         # 时钟 c2：信号日 T 在 T+2 结算
            world.loop().run_portfolio(_upto(ds, d), aum=10_000.0, data_source="yahoo", now=_after_close(d))
        rows = world.ev.strategy_forward_ic(sid)
        assert rows and {r.clock_version for r in rows} == {CLOCK_VERSION}
        from sqlalchemy import select

        from app.db.evidence_store import SignalArchive
        with world.ev._Session() as s:                                         # noqa: SLF001
            arch = list(s.scalars(select(SignalArchive)).all())
        assert arch and {a.clock_version for a in arch} == {CLOCK_VERSION}

    def test_the_spec_binds_the_clock_so_a_new_clock_is_a_new_strategy(self, monkeypatch):
        import app.core.data_engine.input_identity as ii
        from app.core.portfolio_manager.executable_spec import (
            build_spec,
            spec_hash,
            verify_against_runtime,
        )
        a = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e4, 0.01)
        assert a["clock"] == ii.CLOCK_VERSION
        monkeypatch.setattr(ii, "CLOCK_VERSION", "v2-test")
        b = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e4, 0.01)
        assert spec_hash(a) != spec_hash(b)
        assert any("clock" in d for d in verify_against_runtime(a, {"1": "rank(close)"}, 1e4))

    def test_verdicts_and_gate_results_carry_the_clock(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from app.core.data_engine.input_identity import CLOCK_VERSION
        from app.core.lifecycle.validation_gate import ValidationGate
        from app.core.portfolio_manager.strategy_builder import build_strategy_config
        from app.core.portfolio_manager.strategy_gate import StrategyGate
        from tests.conftest import _make_dataset
        ds = _make_dataset(80, 10)

        def ev(self, *a, **kw):
            return SimpleNamespace(passed=True, sharpe=1., deflated_sharpe=.99, t_stat=3., reasons=[],
                                   to_dict=lambda: {"passed": True})
        with patch.object(StrategyGate, "evaluate", ev):
            cfg = build_strategy_config({"1": ds["close"].rank(axis=1)}, ds, 10000, method="equal_weight",
                                        factor_dsl={"1": "rank(close)"})
        assert cfg.verdict["clock_version"] == CLOCK_VERSION
        res = ValidationGate(n_splits=2).evaluate("rank(close)", ds, n_trials=1)
        assert res.to_dict()["clock_version"] == CLOCK_VERSION
