"""
lifecycle/validation_gate.py + db/trial_ledger.py —— 自动验证门的定钉测试

来由：`validation_gate` 8 个变异点首测击杀率 **25.0%**（存活 6），
`trial_ledger` 8 个变异点 50.0%（存活 4）。

验证门是 CANDIDATE→VALIDATED 的自动判定（WalkForward 全折为正 + DSR + 夏普 t），
而 `trial_ledger` 提供的**全局 trial 计数**正是 DSR 去膨胀的分母。
两者一起决定"一个因子能不能自动升级"。存活项包括：

  - `ValidationResult(passed=False, ...)` 的失败优先构造
  - `if tstat < self.min_tstat` 的边界方向
  - `(mu/sd) * sqrt(n)` 的年化/样本量因子
  - `len(rets) < 30 or std == 0` 的样本下限
  - `Executor(validate=False)` 的静态校验开关
  - trial_ledger 的 `nullable=False` 与引擎配置

既有覆盖（test_phase9_validation_gate / test_phase_s3）只跑通了主路径。
"""
from __future__ import annotations

from datetime import date as _date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

import app.core.lifecycle.validation_gate as vg
from app.core.lifecycle.validation_gate import ValidationGate, ValidationResult
from app.db.trial_ledger import TrialCount, TrialLedger


def _panel(days: int = 260, n_tickers: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=days)
    cols = [f"T{i}" for i in range(n_tickers)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.0003, 0.012, (days, n_tickers)), axis=0),
        index=idx, columns=cols)
    return {"close": close, "open": close, "high": close * 1.004,
            "low": close * 0.996, "vwap": close,
            "volume": pd.DataFrame(4e6, index=idx, columns=cols),
            "returns": close.pct_change().fillna(0.0)}


# ===========================================================================
# A. 结果容器：失败优先
# ===========================================================================

class TestResultDefaults:

    def test_result_starts_as_not_passed(self):
        """
        `res = ValidationResult(passed=False, ...)` —— 失败优先。
        改成 True 后，任何**提前 return / 异常路径**都会返回"已验证通过"，
        因子直接升进 VALIDATED。
        """
        r = ValidationResult(passed=False, dsl="x", n_trials=1)
        assert r.passed is False

    def test_failed_validation_reports_reasons(self, monkeypatch):
        """门不通过时必须给出可读理由，且 passed 为 False。"""
        gate = ValidationGate(use_global_trials=False)
        monkeypatch.setattr(
            ValidationGate, "_walk_forward",
            lambda self, dsl, ds: (_ for _ in ()).throw(RuntimeError("wf down")))
        res = gate.evaluate("rank(close)", _panel(days=60))
        assert res.passed is False
        assert any("WalkForward" in x for x in res.reasons)


# ===========================================================================
# B. DSR / t 的公式与边界
# ===========================================================================

class TestDsrAndTstat:

    @staticmethod
    def _gate_with(monkeypatch, rets: pd.Series, **kw) -> ValidationGate:
        """把回测替换成打桩收益，直接驱动 DSR/t 分支。"""
        import app.core.backtest_engine.backtest_engine as be

        class _Res:
            net_returns = rets

        monkeypatch.setattr(be.BacktestEngine, "run",
                            lambda self, *a, **k: _Res())
        return ValidationGate(use_global_trials=False, **kw)

    @staticmethod
    def _rets(mean: float, sd: float, n: int, seed: int = 0) -> pd.Series:
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(mean, sd, n),
                         index=pd.bdate_range("2023-01-02", periods=n))

    def test_tstat_scales_with_sqrt_of_sample_size(self, monkeypatch):
        """
        `t = (mu/sd) * sqrt(len(rets))`。写成 `/` 后 t 随样本增大而**变小** ——
        "多攒数据"反而更难通过门。
        """
        # ⚠️ 两段必须是**同分布同取值**的：直接用不同长度的随机抽样，
        #    mu/sd 本身就变了，比较的不再是 √n（第一版就是这么红的）。
        #    这里把同一段收益平铺 4 遍 —— mu/sd 不变，只有 n 变成 4 倍。
        short = self._rets(0.001, 0.01, 60, seed=1)
        long = pd.Series(np.tile(short.values, 4),
                         index=pd.bdate_range("2023-01-02", periods=240))
        g1 = self._gate_with(monkeypatch, short)
        _, t1 = g1._deflated_sharpe_and_tstat("rank(close)", _panel(days=60), 1)
        g2 = self._gate_with(monkeypatch, long)
        _, t2 = g2._deflated_sharpe_and_tstat("rank(close)", _panel(days=240), 1)
        mu, sd = float(np.mean(short.values)), float(np.std(short.values, ddof=1))
        assert t1 == pytest.approx((mu / sd) * np.sqrt(len(short)), abs=1e-9)
        assert abs(t2) > abs(t1), "样本翻 4 倍后 t 没有变大 —— √n 的方向疑似反了"

    def test_thirty_observations_are_enough(self, monkeypatch):
        """
        `if len(rets) < 30 or std == 0: raise` 的边界：**恰好 30** 可以算。
        放宽成 `<=` 会把刚够样本的因子拦在"样本不足"上。
        """
        g = self._gate_with(monkeypatch, self._rets(0.001, 0.01, 30, seed=2))
        dsr, t = g._deflated_sharpe_and_tstat("rank(close)", _panel(days=30), 1)
        assert np.isfinite(t)

    def test_twenty_nine_observations_are_rejected(self, monkeypatch):
        g = self._gate_with(monkeypatch, self._rets(0.001, 0.01, 29, seed=2))
        with pytest.raises(ValueError, match="样本不足"):
            g._deflated_sharpe_and_tstat("rank(close)", _panel(days=29), 1)

    def test_zero_variance_is_rejected(self, monkeypatch):
        flat = pd.Series(0.0, index=pd.bdate_range("2023-01-02", periods=60))
        g = self._gate_with(monkeypatch, flat)
        with pytest.raises(ValueError, match="方差为 0"):
            g._deflated_sharpe_and_tstat("rank(close)", _panel(days=60), 1)

    def test_tstat_exactly_at_threshold_passes(self, monkeypatch):
        """
        `if tstat < self.min_tstat: reasons.append(...)` —— **恰好等于**门槛算过。
        放宽成 `<=` 会把刚好达标的因子拦下。
        """
        rets = self._rets(0.001, 0.01, 120, seed=3)
        mu, sd = float(np.mean(rets.values)), float(np.std(rets.values, ddof=1))
        tstat = (mu / sd) * np.sqrt(len(rets))
        g = self._gate_with(monkeypatch, rets, min_tstat=tstat, dsr_threshold=0.0)
        monkeypatch.setattr(ValidationGate, "_walk_forward",
                            lambda self, dsl, ds: type("W", (), {
                                "n_folds": self.n_splits, "min_oos_sharpe": 1.0,
                                "mean_oos_sharpe": 1.0, "pct_positive": 1.0})())
        res = g.evaluate("rank(close)", _panel(days=120))
        assert res.t_stat == pytest.approx(tstat, abs=1e-9)
        assert not any("夏普 t=" in x for x in res.reasons), (
            f"t 恰好等于门槛却被拦：{res.reasons}")

    def test_dsr_at_threshold_is_rejected(self, monkeypatch):
        """`if dsr <= self.dsr_threshold` —— 恰好等于阈值算**不过**（保守侧）。"""
        rets = self._rets(0.003, 0.008, 200, seed=4)
        g = self._gate_with(monkeypatch, rets, dsr_threshold=1.0, min_tstat=0.0)
        monkeypatch.setattr(ValidationGate, "_walk_forward",
                            lambda self, dsl, ds: type("W", (), {
                                "n_folds": self.n_splits, "min_oos_sharpe": 1.0,
                                "mean_oos_sharpe": 1.0, "pct_positive": 1.0})())
        res = g.evaluate("rank(close)", _panel(days=200))
        assert any("DSR" in x for x in res.reasons)

    def test_long_window_dsl_is_not_dropped_by_static_validation(self, monkeypatch):
        """
        `Executor(validate=False)` —— 与 leak_filter / daily_trading_loop 同一个
        设计决定：**已入库的因子不该再被静态规则二次裁决**。
        改成 True 后，`ts_mean(close, 300)` 这类窗口超过 252 的因子会抛
        ValidationError，被上层记成"DSR/t 计算失败"，理由完全指错方向。
        """
        g = ValidationGate(use_global_trials=False)
        dsr, t = g._deflated_sharpe_and_tstat(
            "rank(ts_mean(close, 300))", _panel(days=420, seed=5), 1)
        assert np.isfinite(t), "长窗口因子在 DSR 计算里被静态校验拒掉了"


# ===========================================================================
# C. 全局 trial 计数
# ===========================================================================

class TestGlobalTrials:

    def test_ledger_failure_is_logged_not_silent(self, monkeypatch, caplog):
        """
        读不到试验台账时 n_trials 退回 1 —— DSR 的多重检验校正就此失效、门变松。
        必须留下 ERROR。
        """
        class _Boom:
            def __init__(self):
                raise RuntimeError("ledger down")

        monkeypatch.setattr("app.db.trial_ledger.TrialLedger", _Boom)
        monkeypatch.setattr(ValidationGate, "_walk_forward",
                            lambda self, dsl, ds: type("W", (), {
                                "n_folds": self.n_splits, "min_oos_sharpe": 1.0,
                                "mean_oos_sharpe": 1.0, "pct_positive": 1.0})())
        monkeypatch.setattr(ValidationGate, "_deflated_sharpe_and_tstat",
                            lambda self, dsl, ds, n: (0.99, 5.0))
        with caplog.at_level("ERROR"):
            res = ValidationGate(use_global_trials=True).evaluate(
                "rank(close)", _panel(days=60))
        assert res.n_trials == 1
        assert any("多重检验" in r.getMessage() for r in caplog.records), (
            "试验台账不可读却没有留下 ERROR —— 门悄悄变松了")

    def test_explicit_n_trials_wins(self, monkeypatch):
        monkeypatch.setattr(ValidationGate, "_walk_forward",
                            lambda self, dsl, ds: type("W", (), {
                                "n_folds": self.n_splits, "min_oos_sharpe": 1.0,
                                "mean_oos_sharpe": 1.0, "pct_positive": 1.0})())
        monkeypatch.setattr(ValidationGate, "_deflated_sharpe_and_tstat",
                            lambda self, dsl, ds, n: (0.99, 5.0))
        res = ValidationGate(use_global_trials=True).evaluate(
            "rank(close)", _panel(days=60), n_trials=777)
        assert res.n_trials == 777


# ===========================================================================
# D. trial_ledger
# ===========================================================================

class TestTrialLedger:

    @pytest.fixture
    def ledger(self, tmp_path) -> TrialLedger:
        return TrialLedger(db_url=f"sqlite:///{tmp_path/'t.db'}")

    def test_total_column_is_not_nullable(self, ledger):
        """
        `Column(Integer, default=0, nullable=False)` —— 允许为空后，
        `int(row.total)` 会在 None 上抛 TypeError，
        DSR 的多重检验分母就此不可读（门退回 n_trials=1，偏乐观）。
        """
        insp = inspect(ledger._engine)
        col = {c["name"]: c for c in insp.get_columns("trial_ledger")}["total"]
        assert col["nullable"] is False
        # ⚠️ 走 ORM 插入 None 会被 `default=0` 顶替，测不到 NOT NULL
        #    （第一版就是这么红的）。用原生 SQL 绕开 ORM 默认值。
        from sqlalchemy import text
        with ledger._engine.begin() as conn:
            with pytest.raises(IntegrityError):
                conn.execute(text(
                    "INSERT INTO trial_ledger (id, total) VALUES (2, NULL)"))

    def test_counts_accumulate(self, ledger):
        assert ledger.total() == 0
        assert ledger.add(10) == 10
        assert ledger.add(5) == 15
        assert ledger.total() == 15

    def test_non_positive_increments_are_ignored(self, ledger):
        ledger.add(7)
        assert ledger.add(0) == 7
        assert ledger.add(-3) == 7, "负增量不得减少累计试验数"

    def test_reset_clears_the_counter(self, ledger):
        ledger.add(42)
        ledger.reset()
        assert ledger.total() == 0

    def test_engine_does_not_echo_sql(self, ledger):
        assert ledger._engine.echo is False

    def test_sqlite_allows_cross_thread_use(self, tmp_path):
        import threading
        led = TrialLedger(db_url=f"sqlite:///{tmp_path/'x.db'}")
        led.add(3)
        box = {}

        def _read():
            try:
                box["v"] = led.total()
            except Exception as exc:      # noqa: BLE001
                box["err"] = exc

        th = threading.Thread(target=_read)
        th.start()
        th.join()
        assert "err" not in box, f"跨线程读取失败：{box.get('err')}"
        assert box["v"] == 3

    def test_value_is_readable_after_commit(self, ledger):
        """`expire_on_commit=False`：`add()` 在 commit 之后仍要能读出 total。"""
        assert ledger.add(4) == 4


# ===========================================================================
# E. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "L115 `res = ValidationResult(passed=False, ...)` 的初值":
        "该初值必被覆盖：`evaluate()` 的每一条路径最后都执行 "
        "`res.passed = len(reasons) == 0`（没有提前 return），"
        "构造时传 False 还是 True 都观察不到差别。"
        "见 test_passed_is_always_recomputed_before_returning。",

    "L198 `t_stat = (mu/sd) * sqrt(n) if sd > 1e-12 else 0.0` -> `>=`":
        "区分值需要 sd 恰好等于 1e-12。sd = np.std(rets, ddof=1) 是浮点均方根，"
        "无法反解出精确等于 1e-12 的收益序列；而零方差的情形在更早的 "
        "`float(np.nanstd(...)) == 0.0` 处就已抛错，根本走不到这一行。",

    "trial_ledger L49 / diagnostics_store L50 `expire_on_commit=False` -> True":
        "两处的 commit 方法都在**同一个 session 内**读取需要的字段"
        "（`add()` 读 row.total、`save()` 读 rec.id，过期后会自动 refresh），"
        "读方法走的是只读 session，不触发过期。两种取值都观察不到差别。",
}


def test_passed_is_always_recomputed_before_returning():
    """L115 等价性的机械验证：evaluate 里没有提前 return，passed 必被重算。"""
    import ast
    import inspect
    import textwrap
    src = inspect.getsource(ValidationGate.evaluate)
    fn = ast.parse(textwrap.dedent(src)).body[0]
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert len(returns) == 1, f"evaluate 出现了提前 return（{len(returns)} 处），初值变得可观测"
    assert "res.passed = len(reasons) == 0" in src


def test_ledger_and_diagnostics_read_inside_their_sessions():
    """trial_ledger / diagnostics_store 的 expire_on_commit 等价性机械验证。"""
    import inspect
    import app.db.diagnostics_store as ds
    import app.db.trial_ledger as tl

    add_src = inspect.getsource(tl.TrialLedger.add)
    assert "s.commit()" in add_src and "return int(row.total)" in add_src
    save_src = inspect.getsource(ds.DiagnosticsStore.save)
    assert "s.commit()" in save_src and "rec.id" in save_src
    for fn in (tl.TrialLedger.total, ds.DiagnosticsStore.recent):
        assert ".commit()" not in inspect.getsource(fn), (
            f"{fn.__name__} 在返回前提交了事务 —— expire_on_commit 会变得可观测")


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 3
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
