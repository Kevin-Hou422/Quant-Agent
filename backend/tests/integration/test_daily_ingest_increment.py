"""
daily_ingest.py —— 前向增量与健康门的定钉测试（变异测试驱动）

来由：22 个变异点，首测击杀率 **31.8%**（存活 15）。

这是 Phase 11 的核心：它决定**哪一天起的 IC 算"真前向"**。
`forward_from` 之后的 IC 会被标为前向证据，直接喂给 TR.4 的 →ACTIVE 门。
于是这里的每一处边界都与"会不会把历史回放当成前向战绩"直接相关：

  - `if pd.Timestamp(d).normalize() > last.normalize()` 放宽成 `>=`
    → **最后一根旧 bar 被当成新 bar**，forward_from 提前一天，回放冒充前向
  - `nxt = (last + 1天)` 写成 `-` → 增量窗口起点回退，重复拉取旧数据
  - `IngestResult(False, ...)` 的 False 改成 True → **被拒绝的摄取被当成接受**，
    坏数据照样进入交易循环
  - `if score < self.min_health` 放宽成 `<=` → 恰好达标的数据被拒

既有覆盖（test_phase7_ingest / test_phase11_*）只验证了"能跑通"与少数几条主路径。
本文件用打桩的数据源驱动每一条分支。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.tasks.daily_ingest as di
from app.tasks.daily_ingest import DailyIngest, IngestResult


def _panel(days: int, start: str = "2024-01-02", n_tickers: int = 3) -> dict:
    idx = pd.bdate_range(start, periods=days)
    cols = [f"T{i}" for i in range(n_tickers)]
    close = pd.DataFrame(
        100 + np.arange(days * n_tickers, dtype=float).reshape(days, n_tickers),
        index=idx, columns=cols)
    return {"close": close, "open": close, "high": close * 1.002,
            "low": close * 0.998, "vwap": close,
            "volume": pd.DataFrame(1e6, index=idx, columns=cols)}


class _FakeDS:
    def __init__(self, data: dict):
        self.data = data


@pytest.fixture
def ingest(tmp_path, monkeypatch):
    """把 PIT 目录指到 tmp，并把注册表加载/健康检查打桩。"""
    from app.config import settings
    monkeypatch.setattr(settings, "pit_store_dir", str(tmp_path / "pit"), raising=False)
    monkeypatch.setattr(settings, "paper_start", "2024-01-02", raising=False)
    return DailyIngest()


def _stub_loader(monkeypatch, panel: dict, score: float = 1.0):
    """打桩 `load_registry_dataset` 与 `check_dataset_health`。"""
    import app.core.data_engine.dataset_registry as reg

    def _load(name, start=None, end=None, health_check=False):
        close = panel["close"]
        s = pd.Timestamp(start) if start else close.index[0]
        e = pd.Timestamp(end) if end else close.index[-1]
        sub = {k: v.loc[(v.index >= s) & (v.index <= e)] for k, v in panel.items()}
        return _FakeDS(sub)

    monkeypatch.setattr(reg, "load_registry_dataset", _load)
    monkeypatch.setattr(reg, "check_dataset_health",
                        lambda ds, min_score=0.0, warn_only=True:
                        type("R", (), {"overall_score": score})())


def _freeze_today(monkeypatch, day: str):
    """把"今天"和"最近交易日"都钉到指定日期，让增量窗口可预期。"""
    ts = pd.Timestamp(day)
    monkeypatch.setattr(di.pd.Timestamp, "utcnow", staticmethod(
        lambda: ts.tz_localize("UTC")))
    import app.core.data_engine.market_calendar as mc
    monkeypatch.setattr(mc, "last_trading_day", lambda *a, **k: ts.normalize())


# ===========================================================================
# A. 空库 → 历史回填（全部计为回放）
# ===========================================================================

class TestBackfill:

    def test_empty_store_backfills_and_marks_nothing_as_forward(self, ingest,
                                                                monkeypatch):
        """
        回填的 `forward_from` 必须是 None —— 这些 IC 是回放，不是前向证据。
        它一旦被填上，首次运行就会把整段历史记成前向战绩。
        """
        panel = _panel(10)
        _stub_loader(monkeypatch, panel)
        _freeze_today(monkeypatch, "2024-01-15")
        res = ingest.ingest_incremental("px")
        assert res.accepted is True
        assert res.mode == "full"
        assert res.forward_from is None, "历史回填被标成了前向起点"
        assert res.n_new_bars == res.n_dates > 0


# ===========================================================================
# B. 增量窗口与"新 bar"的判定
# ===========================================================================

class TestIncrementWindow:

    @staticmethod
    def _inject_clock(monkeypatch):
        """
        注入"每次调用前进一分钟"的时钟。

        `as_of = datetime.now(timezone.utc).isoformat(timespec="seconds")` 的
        分辨率是**秒**：同一个测试里的多次写入通常落在同一秒，主键
        (timestamp,ticker,as_of) 完全相同 → 被幂等去重吃掉，多写的行看不见；
        而一旦跨秒又会看得见 —— vintage 计数因此是**时钟相关**的。
        本注入把它变成确定值。
        """
        import datetime as _dt
        import app.tasks.daily_ingest as mod

        clock = {"n": 0}
        real_dt = _dt.datetime

        class _Clock(real_dt):
            @classmethod
            def now(cls, tz=None):
                clock["n"] += 1
                return real_dt(2024, 6, 1, 12, clock["n"], 0,
                               tzinfo=tz or _dt.timezone.utc)

        monkeypatch.setattr(mod, "datetime", _Clock)

    def _seed(self, ingest, monkeypatch, seeded_days: int = 5):
        panel = _panel(seeded_days)
        _stub_loader(monkeypatch, panel)
        _freeze_today(monkeypatch, str(panel["close"].index[-1].date()))
        first = ingest.ingest_incremental("px")
        assert first.accepted and first.mode == "full"
        return panel

    def test_no_new_bar_when_pit_is_already_current(self, ingest, monkeypatch):
        """
        `if last.normalize() >= target.normalize(): no_new_bar`
        —— 已经跟上最近交易日就不该再拉，也**不得**把结果标成 accepted。
        """
        self._seed(ingest, monkeypatch)
        res = ingest.ingest_incremental("px")
        assert res.accepted is False, "无新 bar 却返回 accepted=True"
        assert res.mode == "no_new_bar"
        assert res.reject_reason == "no_new_bar"

    def test_only_strictly_newer_bars_count_as_new(self, ingest, monkeypatch):
        """
        `if pd.Timestamp(d).normalize() > last.normalize()` —— **严格大于**。
        放宽成 `>=` 会把 PIT 里已有的最后一根 bar 也算成新 bar，
        `forward_from` 因此提前一天，那一天的 IC 被错误标成前向证据。

        构造：PIT 已有 5 天；新数据覆盖第 5~8 天（第 5 天是重叠的旧 bar）。
        """
        self._seed(ingest, monkeypatch, seeded_days=5)
        full = _panel(8)
        _stub_loader(monkeypatch, full)
        _freeze_today(monkeypatch, str(full["close"].index[-1].date()))
        res = ingest.ingest_incremental("px")
        assert res.accepted is True and res.mode == "incremental"
        assert res.n_new_bars == 3, f"新 bar 数应为 3，实际 {res.n_new_bars}"
        assert res.forward_from == str(full["close"].index[5].date()), (
            f"前向起点应是第 6 根 bar，实际 {res.forward_from} —— "
            f"重叠的旧 bar 疑似被算成了新 bar")

    def test_increment_window_starts_the_day_after_the_last_bar(self, ingest,
                                                               monkeypatch):
        """
        `nxt = (last + pd.Timedelta(days=1))`。写成 `-` 会让增量窗口
        从**最后一根 bar 的前一天**开始，重复拉取已有数据。
        通过记录 loader 收到的 start 参数来验证。
        """
        self._seed(ingest, monkeypatch, seeded_days=5)
        seen = {}
        full = _panel(8)
        import app.core.data_engine.dataset_registry as reg
        real_close = full["close"]

        def _load(name, start=None, end=None, health_check=False):
            seen["start"] = start
            s, e = pd.Timestamp(start), pd.Timestamp(end)
            return _FakeDS({k: v.loc[(v.index >= s) & (v.index <= e)]
                            for k, v in full.items()})

        monkeypatch.setattr(reg, "load_registry_dataset", _load)
        monkeypatch.setattr(reg, "check_dataset_health",
                            lambda ds, min_score=0.0, warn_only=True:
                            type("R", (), {"overall_score": 1.0})())
        _freeze_today(monkeypatch, str(real_close.index[-1].date()))
        ingest.ingest_incremental("px")
        expected = (real_close.index[4] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        assert seen["start"] == expected, (
            f"增量窗口起点应为最后一根 bar 的次日 {expected}，实际 {seen['start']}")

    def _stub_sloppy_provider(self, monkeypatch, panel: dict):
        """
        打桩一个**忽略 start 参数**的数据源 —— 真实 provider 常会把区间向前对齐，
        把 PIT 里已有的最后一根 bar 一并返回。模块正是靠
        `> last.normalize()` 与 `df.index > last` 两道过滤来防这件事，
        而"守规矩的 stub"永远触发不了它们（第一版就是这样）。
        """
        import app.core.data_engine.dataset_registry as reg
        monkeypatch.setattr(reg, "load_registry_dataset",
                            lambda name, start=None, end=None, health_check=False:
                            _FakeDS(panel))
        monkeypatch.setattr(reg, "check_dataset_health",
                            lambda ds, min_score=0.0, warn_only=True:
                            type("R", (), {"overall_score": 1.0})())

    def test_overlapping_bar_from_a_sloppy_provider_is_not_new(self, ingest,
                                                               monkeypatch):
        """
        `if pd.Timestamp(d).normalize() > last.normalize()` —— **严格大于**。
        数据源把整段历史（含已有的 5 根 bar）原样返回时，
        `>=` 会把最后一根旧 bar 也算成新 bar，forward_from 提前一天。
        """
        self._seed(ingest, monkeypatch, seeded_days=5)
        full = _panel(8)
        self._stub_sloppy_provider(monkeypatch, full)
        _freeze_today(monkeypatch, str(full["close"].index[-1].date()))
        res = ingest.ingest_incremental("px")
        assert res.accepted is True
        assert res.n_new_bars == 3, (
            f"数据源返回了 8 根 bar（含 5 根旧的），新 bar 应只有 3 根，"
            f"实际 {res.n_new_bars}")
        assert res.forward_from == str(full["close"].index[5].date())

        # 这里只验"新 bar 计数"；PIT 只收增量那一条另见
        # test_increment_written_to_pit_excludes_the_overlapping_bar。

    def test_provider_with_no_new_bars_is_rejected(self, ingest, monkeypatch):
        """
        `if not new_dates: return IngestResult(False, ...)` —— 数据源只返回旧 bar 时，
        这条路径的 False 一旦被改成 True，**没有新数据的一天也会被当成摄取成功**，
        交易循环照常跑并把回放 IC 当成前向。
        """
        panel = self._seed(ingest, monkeypatch, seeded_days=5)
        self._stub_sloppy_provider(monkeypatch, panel)     # 只有那 5 根旧 bar
        _freeze_today(monkeypatch,
                      str((panel["close"].index[-1] + pd.Timedelta(days=3)).date()))
        res = ingest.ingest_incremental("px")
        assert res.accepted is False, "数据源没给新 bar，却标成了摄取成功"
        assert res.mode == "no_new_bar"
        assert res.reject_reason == "no_new_bar"

    def test_pit_append_failure_is_rejected(self, ingest, monkeypatch):
        """
        写 PIT 失败这条路径的 `IngestResult(False, ...)`：改成 True 后，
        **数据没落库**却报告摄取成功，下一天的增量窗口也会算错。
        """
        self._seed(ingest, monkeypatch, seeded_days=5)
        full = _panel(8)
        _stub_loader(monkeypatch, full)
        _freeze_today(monkeypatch, str(full["close"].index[-1].date()))

        def _boom(*a, **k):
            raise RuntimeError("disk full")

        monkeypatch.setattr(type(ingest), "_append_pit", _boom)
        res = ingest.ingest_incremental("px")
        assert res.accepted is False, "写 PIT 失败却报告摄取成功"
        assert res.reject_reason.startswith("pit_append_failed")

    def test_registry_is_loaded_without_its_own_health_gate(self, ingest, monkeypatch):
        """
        `load_registry_dataset(..., health_check=False)` —— 健康门由本模块**自己**
        按分数下（拒绝原因才会是"分数不足"）。改成 True 后加载器会自行抛异常，
        所有低分数据的拒绝原因都变成"load_failed"，运维看到的诊断完全不同。
        """
        import app.core.data_engine.dataset_registry as reg
        seen = {}
        panel = _panel(5)

        def _load(name, start=None, end=None, health_check=False):
            seen["health_check"] = health_check
            return _FakeDS(panel)

        monkeypatch.setattr(reg, "load_registry_dataset", _load)
        monkeypatch.setattr(reg, "check_dataset_health",
                            lambda ds, min_score=0.0, warn_only=True:
                            type("R", (), {"overall_score": 1.0})())
        ingest.ingest("px", "2024-01-02", "2024-01-10")
        assert seen["health_check"] is False, (
            "注册表加载时打开了自带的健康门 —— 拒绝原因会从'分数不足'变成'加载失败'")

    def test_increment_written_to_pit_excludes_the_overlapping_bar(self, ingest,
                                                                   monkeypatch):
        """
        `increment = {f: df.loc[df.index > last] ...}` —— 写进 PIT 的也必须**严格大于**。
        放宽成 `>=` 会把 PIT 里已有的最后一根 bar 再写一份新 vintage。

        ⚠️ 这一处**不能**靠"同一天只有一个 vintage"来断言：`as_of` 的分辨率是
        **秒**（`isoformat(timespec="seconds")`），同一个测试里两次写入落在同一秒，
        主键 (timestamp,ticker,as_of) 完全相同 → 被幂等去重吃掉，多写的那一行看不见。
        这正是我第一版漏掉它的原因。
        这里注入一个**每次调用都前进一分钟**的时钟，让两次写入的 vintage 必然不同，
        多出来的那一行才暴露得出来。
        """
        self._inject_clock(monkeypatch)
        self._seed(ingest, monkeypatch, seeded_days=5)
        full = _panel(8)
        self._stub_sloppy_provider(monkeypatch, full)
        _freeze_today(monkeypatch, str(full["close"].index[-1].date()))
        ingest.ingest_incremental("px")

        from app.config import settings
        from app.core.data_engine.pit_store import PITStore
        store = PITStore(settings.pit_store_dir)
        part = next((store.store_dir / "px").glob("year=*")) / "data.parquet"
        df = pd.read_parquet(part)
        per_day = df.groupby("timestamp")["as_of"].nunique()
        last_seeded = full["close"].index[4]

        # 基线下共有 3 个 vintage：①回填时 `ingest()` 内部写的 5 天；
        # ②增量取数时 `ingest()` 内部写的整段 8 天；③增量追加写的 3 天。
        # （②这条"取数即整段写 PIT"是登记在案的产品问题——注释说"只写增量"，
        #   实际上游已经整段写过一遍了。见 MUTATION_LEDGER。）
        assert df["as_of"].nunique() == 3, (
            f"构造前提不成立：应有 3 个 vintage，实际 {df['as_of'].nunique()}")
        assert int(per_day.loc[last_seeded]) == 2, (
            f"PIT 里最后一根旧 bar（{last_seeded.date()}）出现了 "
            f"{int(per_day.loc[last_seeded])} 个 vintage —— "
            f"它被增量追加又写了一遍，说明筛选不是**严格大于**")
        assert per_day.max() == 2, f"有交易日的 vintage 数超出预期：\n{per_day}"

    def test_pit_only_receives_the_increment(self, ingest, monkeypatch):
        """
        `increment = {f: df.loc[df.index > last] ...}` —— 只追加**新** bar：
        重叠的那根旧 bar 不得再次进 PIT。`>` 放宽成 `>=` 会把它再写一次。

        钉住的是**逐日 vintage 数**，而不是"每天最多 1 个 vintage"：
        产品当前对增量日确实写了两次（`ingest()` 内部先写整段增量窗口，
        `ingest_incremental()` 随后再 `_append_pit(increment)` 写一次 ——
        已登记在 MUTATION_LEDGER "A 档查出的问题"，本阶段不修）。
        两次写的 `as_of` 只有跨秒才不同，所以原先"max()==1"的断言是
        **时钟相关**的：单跑绿、并跑红。这里注入确定时钟，把两次写固定成
        两个 vintage，于是：
          - 回填的 5 天 → 恰好 1 个 vintage（只写过一次）
          - 增量的 3 天 → 恰好 2 个 vintage（双写，不多不少）
        `>` → `>=` 会让第 5 天也出现在增量里 → 它的 vintage 数变 2 → 被杀。
        """
        self._inject_clock(monkeypatch)
        panel = self._seed(ingest, monkeypatch, seeded_days=5)
        seeded = {d.normalize() for d in panel["close"].index}
        full = _panel(8)
        _stub_loader(monkeypatch, full)
        _freeze_today(monkeypatch, str(full["close"].index[-1].date()))
        ingest.ingest_incremental("px")

        from app.config import settings
        from app.core.data_engine.pit_store import PITStore
        store = PITStore(settings.pit_store_dir)
        part = next((store.store_dir / "px").glob("year=*")) / "data.parquet"
        df = pd.read_parquet(part)
        per_day = df.groupby("timestamp")["as_of"].nunique()
        assert len(per_day) == 8, f"PIT 天数应为 8，实际 {len(per_day)}"
        for ts, n in per_day.items():
            ts = pd.Timestamp(ts).normalize()
            if ts in seeded:
                assert n == 1, (
                    f"回填日 {ts.date()} 出现了 {n} 个 vintage —— "
                    f"重叠的旧 bar 被增量重写了")
            else:
                assert n == 2, (
                    f"增量日 {ts.date()} 的 vintage 数是 {n}，预期 2（已登记的双写）")


# ===========================================================================
# C. 健康门与拒绝路径
# ===========================================================================

class TestHealthGate:

    def test_rejection_results_are_not_accepted(self, ingest, monkeypatch):
        """
        三处 `IngestResult(False, ...)`（no_new_bar / empty_close / health_error）
        的首参一旦被改成 True，**被拒绝的摄取会被当成接受**，坏数据直接进交易循环。
        """
        import app.core.data_engine.dataset_registry as reg
        monkeypatch.setattr(reg, "load_registry_dataset",
                            lambda *a, **k: _FakeDS({"close": pd.DataFrame()}))
        res = ingest.ingest("px", "2024-01-02", "2024-01-10")
        assert res.accepted is False, "空 close 的摄取被标成 accepted"
        assert res.reject_reason == "empty_close"

    def test_health_check_exception_is_a_rejection(self, ingest, monkeypatch):
        import app.core.data_engine.dataset_registry as reg
        panel = _panel(5)
        monkeypatch.setattr(reg, "load_registry_dataset",
                            lambda *a, **k: _FakeDS(panel))

        def _boom(ds, min_score=0.0, warn_only=True):
            raise RuntimeError("health subsystem down")

        monkeypatch.setattr(reg, "check_dataset_health", _boom)
        res = ingest.ingest("px", "2024-01-02", "2024-01-10")
        assert res.accepted is False
        assert res.reject_reason.startswith("health_error"), (
            "健康检查异常没有被当成拒绝 —— 保守方向反了")

    def test_load_failure_is_a_rejection(self, ingest, monkeypatch):
        import app.core.data_engine.dataset_registry as reg

        def _boom(*a, **k):
            raise RuntimeError("provider offline")

        monkeypatch.setattr(reg, "load_registry_dataset", _boom)
        res = ingest.ingest("px", "2024-01-02", "2024-01-10")
        assert res.accepted is False
        assert res.reject_reason.startswith("load_failed")

    def test_score_exactly_at_threshold_is_accepted(self, ingest, monkeypatch):
        """
        `if score < self.min_health: 拒绝` 的边界：**恰好等于**阈值算通过。
        放宽成 `<=` 会把刚好达标的数据拒掉。
        """
        panel = _panel(5)
        _stub_loader(monkeypatch, panel, score=ingest.min_health)
        res = ingest.ingest("px", "2024-01-02", "2024-01-10")
        assert res.accepted is True, (
            f"健康分恰好等于阈值 {ingest.min_health} 却被拒：{res.reject_reason}")

    def test_score_below_threshold_is_rejected(self, ingest, monkeypatch):
        panel = _panel(5)
        _stub_loader(monkeypatch, panel, score=ingest.min_health - 0.01)
        res = ingest.ingest("px", "2024-01-02", "2024-01-10")
        assert res.accepted is False
        assert "health" in (res.reject_reason or "").lower() or res.health_score < ingest.min_health

    def test_health_check_runs_in_warn_only_mode(self, ingest, monkeypatch):
        """
        `check_dataset_health(..., warn_only=True)` —— 门由本模块按分数自己下，
        不靠被调方抛异常。改成 False 后健康检查会**自己抛**，
        于是所有低分数据都走到 `health_error` 分支，拒绝原因从"分数不足"
        变成"检查异常"，运维看到的诊断完全不同。
        """
        import app.core.data_engine.dataset_registry as reg
        seen = {}
        panel = _panel(5)
        monkeypatch.setattr(reg, "load_registry_dataset",
                            lambda *a, **k: _FakeDS(panel))

        def _check(ds, min_score=0.0, warn_only=True):
            seen["warn_only"] = warn_only
            return type("R", (), {"overall_score": 0.2})()

        monkeypatch.setattr(reg, "check_dataset_health", _check)
        res = ingest.ingest("px", "2024-01-02", "2024-01-10")
        assert seen["warn_only"] is True, "健康检查未以 warn_only 模式调用"
        assert res.accepted is False
        assert not (res.reject_reason or "").startswith("health_error"), (
            "低分数据的拒绝原因变成了'检查异常'")


# ===========================================================================
# D. 接受路径返回的面板
# ===========================================================================

class TestAcceptedPanel:

    def test_accepted_increment_reports_accepted_true(self, ingest, monkeypatch):
        """
        `IngestResult(True, dataset_name, ...)` 的首参改成 False 后，
        **成功的增量摄取会被当成失败**，交易循环整天不跑。
        """
        panel = _panel(5)
        _stub_loader(monkeypatch, panel)
        _freeze_today(monkeypatch, str(panel["close"].index[-1].date()))
        ingest.ingest_incremental("px")

        full = _panel(8)
        _stub_loader(monkeypatch, full)
        _freeze_today(monkeypatch, str(full["close"].index[-1].date()))
        res = ingest.ingest_incremental("px")
        assert res.accepted is True, "成功的增量摄取被标成了失败"
        assert res.dataset is not None and "close" in res.dataset

    def test_returned_panel_covers_backfill_plus_increments(self, ingest, monkeypatch):
        """
        `full = panel if panel.get("close") is not None else (inc.dataset or {})`
        —— 交易循环消费的必须是**PIT 里的完整面板**（回填 + 历次增量），
        删掉 `not` 会退化成只给本次增量的几根 bar。
        """
        panel = _panel(5)
        _stub_loader(monkeypatch, panel)
        _freeze_today(monkeypatch, str(panel["close"].index[-1].date()))
        ingest.ingest_incremental("px")

        full = _panel(8)
        _stub_loader(monkeypatch, full)
        _freeze_today(monkeypatch, str(full["close"].index[-1].date()))
        res = ingest.ingest_incremental("px")
        assert res.n_dates >= 8, (
            f"返回面板只有 {res.n_dates} 天 —— 应为回填 5 天 + 增量 3 天的完整面板")
        assert len(res.dataset["close"]) == res.n_dates


# ===========================================================================
# E. 日管线的返回契约
# ===========================================================================

class TestDailyPipelineContract:
    """
    `run_daily_pipeline` 的返回 dict 是**调度器与运维看到的东西**：
    `ingest_accepted` 决定"今天到底交没交易"。三处取值必须各自正确。
    """

    def _stub_loop(self, monkeypatch):
        import app.tasks.daily_trading_loop as dtl

        class _Report:
            n_alphas = n_alerts = n_errors = 0

        monkeypatch.setattr(dtl.DailyTradingLoop, "__init__",
                            lambda self, *a, **k: None)
        monkeypatch.setattr(dtl.DailyTradingLoop, "run",
                            lambda self, ds: _Report())
        monkeypatch.setattr(dtl.DailyTradingLoop, "run_portfolio",
                            lambda self, ds, **k: {"n_factors": 1, "days_processed": 1})

    def test_successful_run_reports_accepted_true(self, ingest, monkeypatch):
        """
        `return {"ingest_accepted": True, ...}` —— 改成 False 后，
        **交易明明跑了**，日报却说今天没摄取成功，运维据此会去人工补跑。
        """
        import app.tasks.daily_ingest as mod
        panel = _panel(10)
        _stub_loader(monkeypatch, panel)
        _freeze_today(monkeypatch, "2024-01-20")
        self._stub_loop(monkeypatch)
        out = mod.run_daily_pipeline("px", incremental=True)
        assert out["ingest_accepted"] is True, "成功的一轮被报成了摄取失败"
        assert out["mode"] == "full"
        assert "portfolio" in out and out["n_alphas"] == 0

    def test_no_new_bar_reports_accepted_false(self, ingest, monkeypatch):
        import app.tasks.daily_ingest as mod
        panel = _panel(5)
        _stub_loader(monkeypatch, panel)
        _freeze_today(monkeypatch, str(panel["close"].index[-1].date()))
        self._stub_loop(monkeypatch)
        mod.run_daily_pipeline("px", incremental=True)      # 先回填
        out = mod.run_daily_pipeline("px", incremental=True)
        assert out["ingest_accepted"] is False
        assert out["reason"] == "no_new_bar"

    def test_rejected_ingest_reports_accepted_false(self, ingest, monkeypatch):
        import app.core.data_engine.dataset_registry as reg
        import app.tasks.daily_ingest as mod
        monkeypatch.setattr(reg, "load_registry_dataset",
                            lambda *a, **k: _FakeDS({"close": pd.DataFrame()}))
        _freeze_today(monkeypatch, "2024-01-20")
        self._stub_loop(monkeypatch)
        out = mod.run_daily_pipeline("px", incremental=True)
        assert out["ingest_accepted"] is False
        assert out.get("reject_reason")
