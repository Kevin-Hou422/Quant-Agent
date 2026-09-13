"""
tasks/scheduler.py —— 任务注册与生命周期的定钉测试（变异测试驱动）

来由：20 个变异点，首测击杀率 **5.0%**（存活 19）。

scheduler 决定**每天有没有任务跑、什么时候跑、跑几次**。它错了不会有人
立刻发现：任务只是不跑（或多跑），而"没跑"在日志里和"没到时间"长得一样。

存活项：
  - 四处 `replace_existing=True` —— 改成 False 后，jobstore 里的**旧定义**
    会顶掉代码里的新定义：改了 cron 时间、改了函数，重启后仍按旧的跑
  - `"coalesce": True` —— 改成 False 后，停机期间错过的 N 次会在重启时
    **一次性补跑 N 遍**，同一天的交易循环跑多次
  - `if getattr(settings, "enable_backup", True)` 的默认 True
    —— 配置里没写这一项时备份任务静默不注册（前向数据丢了买不回来）
  - `start/shutdown/status` 三处 `_scheduler is not None and _scheduler.running`
    —— `and`→`or` / 删 `not` 会让幂等性失效：重复 start 会起第二个调度器，
    两个调度器同时跑同一批任务
  - `run_daily_pipeline(..., incremental=True)` 的 True
    —— 改成 False 会退回"整段重拉"，前向增量的语义没了
  - `load_registry_dataset(..., health_check=False)` 的 False
    —— 发现任务被健康门拦下，夜间发现静默停摆
  - `last_prev = first_this - timedelta(days=1)` 的 `-`
    —— 月度成本校准会去算**下个月**的区间

既有覆盖（test_phase5 / test_phase9_discovery / test_phase8_cost_calib /
test_phase11_incremental / test_invariants）只验证了"任务能注册上、能调用"。
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

import app.tasks.scheduler as sched_mod
from app.tasks.scheduler import (
    create_scheduler,
    get_scheduler_status,
    shutdown_scheduler,
    start_scheduler,
)


def _safe_shutdown(s) -> None:
    """create_scheduler 返回的调度器**尚未 start**，直接 shutdown 会抛
    SchedulerNotRunningError。这里只做清理，不该让清理动作干扰断言。"""
    try:
        s.shutdown(wait=False)
    except Exception:                      # noqa: BLE001
        pass


@pytest.fixture
def jobs_db(tmp_path):
    return f"sqlite:///{tmp_path / 'sched.db'}"


@pytest.fixture(autouse=True)
def _clean_singleton():
    """每条用例前后都把模块级单例清干净，避免互相污染。"""
    sched_mod._scheduler = None
    yield
    try:
        shutdown_scheduler()
    except Exception:                      # noqa: BLE001
        sched_mod._scheduler = None
    sched_mod._scheduler = None


@pytest.fixture
def all_jobs_on(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "enable_paper_trading", True, raising=False)
    monkeypatch.setattr(settings, "enable_discovery", True, raising=False)
    monkeypatch.setattr(settings, "enable_backup", True, raising=False)


# ===========================================================================
# A. 任务注册
# ===========================================================================

class TestJobRegistration:

    def test_monitor_job_is_always_registered(self, jobs_db):
        s = create_scheduler(db_url=jobs_db)
        try:
            assert {j.id for j in s.get_jobs()} >= {"daily_monitor"}
        finally:
            _safe_shutdown(s)

    def test_optional_jobs_follow_their_switches(self, jobs_db, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "enable_paper_trading", False, raising=False)
        monkeypatch.setattr(settings, "enable_discovery", False, raising=False)
        monkeypatch.setattr(settings, "enable_backup", False, raising=False)
        s = create_scheduler(db_url=jobs_db)
        try:
            ids = {j.id for j in s.get_jobs()}
        finally:
            _safe_shutdown(s)
        assert "daily_trading" not in ids and "nightly_discovery" not in ids
        assert "daily_backup" not in ids

    def test_all_five_jobs_register_when_enabled(self, jobs_db, all_jobs_on):
        s = create_scheduler(db_url=jobs_db)
        try:
            ids = {j.id for j in s.get_jobs()}
        finally:
            _safe_shutdown(s)
        assert ids == {"daily_monitor", "daily_trading", "monthly_cost_calibration",
                       "nightly_discovery", "daily_backup"}, ids

    def test_backup_defaults_to_on_when_the_setting_is_absent(self, jobs_db, monkeypatch):
        """
        `if getattr(settings, "enable_backup", True):` —— 这个**默认 True**
        改成 False 之后，任何没有显式写 `enable_backup` 的配置都会静默
        不注册备份任务。前向 PIT 数据不可再生，丢了买不回来。
        """
        from app.config import settings
        monkeypatch.delattr(settings, "enable_backup", raising=False)
        s = create_scheduler(db_url=jobs_db)
        try:
            ids = {j.id for j in s.get_jobs()}
        finally:
            _safe_shutdown(s)
        assert "daily_backup" in ids, (
            "配置里没有 enable_backup 时备份任务没有注册 —— 默认值不再是 True")

    def test_every_job_replaces_the_stored_definition(self, jobs_db, all_jobs_on):
        """
        五处 `replace_existing=True`。改成 False 之后，jobstore 里已有同 id 的
        任务时 `add_job` 会抛 ConflictingIdError —— 也就是**每次重启都起不来**；
        或（取决于版本）保留旧定义：改了 cron 时间、换了函数，线上仍按旧的跑。

        关键是任务必须**真的落进 jobstore**：`create_scheduler` 返回的调度器
        尚未 start，SQLAlchemyJobStore 此时还没写盘，第二次建当然不冲突
        （上一版就是这么让这五个变异全活下来的）。所以这里先 start 一次让它落盘。
        """
        first = create_scheduler(db_url=jobs_db)
        first.start()
        try:
            assert len(first.get_jobs()) == 5
        finally:
            first.shutdown(wait=False)

        # 同一个 jobstore 再建一次 —— 模拟重启
        second = create_scheduler(db_url=jobs_db)
        second.start()
        try:
            ids = {j.id for j in second.get_jobs()}
        finally:
            second.shutdown(wait=False)
        assert ids == {"daily_monitor", "daily_trading", "monthly_cost_calibration",
                       "nightly_discovery", "daily_backup"}, (
            f"复用同一个 jobstore 重建后任务不全：{ids} —— "
            f"replace_existing 疑似被改成了 False")

    def test_a_changed_schedule_overwrites_the_stored_one(self, jobs_db, all_jobs_on,
                                                          monkeypatch):
        """
        `replace_existing=True` 的**真正意义**：代码里改了时间，重启后以代码为准。
        构造：第一次用 UTC 建库，第二次换个时区重建，触发器必须跟着变。
        """
        first = create_scheduler(db_url=jobs_db, timezone="UTC")
        first.start()
        try:
            before = {j.id: str(getattr(j.trigger, "timezone", "")) for j in first.get_jobs()}
        finally:
            first.shutdown(wait=False)

        second = create_scheduler(db_url=jobs_db, timezone="America/New_York")
        second.start()
        try:
            after = {j.id: str(getattr(j.trigger, "timezone", "")) for j in second.get_jobs()}
        finally:
            second.shutdown(wait=False)
        assert before != after, (
            "改了时区重建，jobstore 里的触发器没有被覆盖 —— "
            "线上会继续按旧定义跑，而代码里看着已经改了")

    def test_missed_runs_are_coalesced_into_one(self, jobs_db):
        """
        `"coalesce": True` —— 改成 False 后，停机期间错过的每一次触发都会
        在重启时各补一遍：同一天的"每日交易循环"可能连跑好几次，
        每一次都会真的去下单。
        """
        s = create_scheduler(db_url=jobs_db)
        try:
            assert s._job_defaults.get("coalesce") is True, (
                f"coalesce={s._job_defaults.get('coalesce')} —— "
                f"错过的触发会被逐次补跑")
            assert s._job_defaults.get("max_instances") == 1, "同一任务可以并发了"
            assert s._job_defaults.get("misfire_grace_time") == 3600
        finally:
            _safe_shutdown(s)

    def test_cron_triggers_use_the_scheduler_timezone(self, jobs_db, all_jobs_on):
        """
        每个 CronTrigger 都显式传了 timezone。漏传会让它用**构造时的本地时区**，
        在非 UTC 机器上整体偏移几小时 —— 收盘后的任务跑到了盘中。
        """
        s = create_scheduler(db_url=jobs_db, timezone="UTC")
        try:
            for j in s.get_jobs():
                # 注意用 .timezone 属性，不是 str(trigger)：
                # CronTrigger 的 __str__ 只打字段（cron[hour='21', minute='0']），
                # 时区只在 __repr__ / .timezone 里，拿 str 去断言永远失败。
                assert "UTC" in str(getattr(j.trigger, "timezone", "")), (
                    f"任务 {j.id} 的触发器时区不是 UTC：{j.trigger!r}")
        finally:
            _safe_shutdown(s)

    def test_job_hours_match_the_documented_order(self, jobs_db, all_jobs_on):
        """巡检 21:00 → 交易 21:30 → 发现 23:00 → 备份 23:45，顺序不能乱。"""
        s = create_scheduler(db_url=jobs_db)
        try:
            trig = {j.id: str(j.trigger) for j in s.get_jobs()}
        finally:
            _safe_shutdown(s)
        assert "hour='21'" in trig["daily_monitor"]
        assert "hour='21'" in trig["daily_trading"] and "minute='30'" in trig["daily_trading"]
        assert "hour='23'" in trig["nightly_discovery"]
        assert "hour='23'" in trig["daily_backup"] and "minute='45'" in trig["daily_backup"]


# ===========================================================================
# B. 单例的幂等性
# ===========================================================================

class TestSingletonLifecycle:

    def test_start_is_idempotent(self, jobs_db):
        """
        `if _scheduler is not None and _scheduler.running: return _scheduler`
        —— `and`→`or` 会让 `_scheduler is None` 时也直接 return None；
        删 `not` 会让**已在运行**时再起一个，两个调度器同时跑同一批任务，
        每天的交易循环执行两遍。
        """
        a = start_scheduler(db_url=jobs_db)
        b = start_scheduler(db_url=jobs_db)
        assert a is b, "重复 start 返回了不同的调度器实例 —— 幂等性失效"
        assert a.running

    def test_shutdown_is_idempotent_and_clears_the_singleton(self, jobs_db):
        start_scheduler(db_url=jobs_db)
        shutdown_scheduler()
        assert sched_mod._scheduler is None
        shutdown_scheduler()          # 第二次不得抛

    def test_shutdown_does_not_wait_for_running_jobs(self, jobs_db, monkeypatch):
        """
        `_scheduler.shutdown(wait=False)` —— 改成 True 会让进程退出时
        **阻塞等待**正在跑的任务（可能是一整轮回测），服务停不下来。
        """
        seen = {}
        s = start_scheduler(db_url=jobs_db)
        monkeypatch.setattr(s, "shutdown", lambda **kw: seen.update(kw))
        shutdown_scheduler()
        assert seen.get("wait") is False, (
            f"shutdown 传的是 wait={seen.get('wait')} —— 停机会被运行中的任务阻塞")

    def test_status_before_start_reports_not_running(self):
        st = get_scheduler_status()
        assert st == {"running": False, "jobs": []}

    def test_status_after_start_lists_the_jobs(self, jobs_db):
        """
        `if _scheduler is None or not _scheduler.running: return {...False...}`
        —— 删 `not` 会让**正在运行**时反而报 not running，
        前端的调度面板从此永远显示"未运行"。
        """
        start_scheduler(db_url=jobs_db)
        st = get_scheduler_status()
        assert st["running"] is True, "已启动却报 running=False"
        assert any(j["id"] == "daily_monitor" for j in st["jobs"]), st["jobs"]
        assert all({"id", "name", "next_run", "trigger"} <= set(j) for j in st["jobs"])

    def test_status_after_shutdown_reports_not_running(self, jobs_db):
        start_scheduler(db_url=jobs_db)
        shutdown_scheduler()
        assert get_scheduler_status()["running"] is False


# ===========================================================================
# C. 任务体里的关键参数
# ===========================================================================

class TestJobBodies:

    def test_daily_trading_uses_incremental_ingest(self, monkeypatch):
        """
        `run_daily_pipeline(..., incremental=True)` —— 改成 False 会退回
        "整段重拉"：前向增量的语义消失，每天把整段历史重新灌一遍 PIT，
        而 `forward_from` 再也标不出"哪天起是真前向"。
        """
        seen = {}
        import app.tasks.daily_ingest as di
        import app.core.data_engine.market_calendar as mc
        monkeypatch.setattr(mc, "is_trading_day", lambda d: True)
        monkeypatch.setattr(di, "run_daily_pipeline",
                            lambda *a, **kw: (seen.update(args=a, kw=kw),
                                              {"ingest_accepted": True})[1])
        sched_mod.daily_trading_job()
        assert seen["kw"].get("incremental") is True, (
            f"每日交易任务传的是 incremental={seen['kw'].get('incremental')}")

    def test_daily_trading_skips_on_non_trading_days(self, monkeypatch):
        called = {"n": 0}
        import app.tasks.daily_ingest as di
        import app.core.data_engine.market_calendar as mc
        monkeypatch.setattr(mc, "is_trading_day", lambda d: False)
        monkeypatch.setattr(di, "run_daily_pipeline",
                            lambda *a, **kw: called.update(n=called["n"] + 1))
        sched_mod.daily_trading_job()
        assert called["n"] == 0, "非交易日仍然跑了交易循环"

    def test_daily_trading_is_fail_closed_when_the_calendar_is_unavailable(self, monkeypatch):
        """日历判不出来 → **不交易**，而不是退回工作日启发式。"""
        from app.core.data_engine.market_calendar import CalendarUnavailable
        called = {"n": 0}
        import app.tasks.daily_ingest as di

        def _boom(_d):
            raise CalendarUnavailable("no calendar")

        import app.core.data_engine.market_calendar as mc
        monkeypatch.setattr(mc, "is_trading_day", _boom)
        monkeypatch.setattr(di, "run_daily_pipeline",
                            lambda *a, **kw: called.update(n=called["n"] + 1))
        sched_mod.daily_trading_job()
        assert called["n"] == 0, "日历不可用时仍然跑了交易循环 —— fail-closed 失效"

    def test_nightly_discovery_bypasses_the_health_gate(self, monkeypatch):
        """
        `load_registry_dataset(..., health_check=False)` —— 改成 True 会让
        夜间发现任务被健康门拦下（发现用的数据集与交易用的不是同一份，
        健康标准也不同），自主发现从此静默停摆。
        """
        seen = {}
        import app.core.data_engine.dataset_registry as reg
        import app.core.discovery.discovery_engine as de

        monkeypatch.setattr(reg, "load_registry_dataset",
                            lambda name, **kw: (seen.update(kw),
                                                type("D", (), {"data": {}})())[1])
        run_kw = {}

        class _Engine:
            def run(self, data, save=False, **kw):
                run_kw["save"] = save
                return type("R", (), {"regime": "x", "families": [],
                                      "n_candidates": 0})()

        monkeypatch.setattr(de, "DiscoveryEngine", _Engine)
        sched_mod.nightly_discovery_job()
        assert seen.get("health_check") is False, (
            f"发现任务传的是 health_check={seen.get('health_check')}")
        assert run_kw.get("save") is True, (
            f"夜间发现任务传的是 save={run_kw.get('save')} —— "
            f"跑完一轮 GP 却不落库，产出全部丢弃")

    def test_monthly_calibration_targets_the_previous_month(self, monkeypatch):
        """
        `last_prev = first_this - timedelta(days=1)` —— `-` 改成 `+` 会让
        "上月最后一天"变成"本月 2 号"，校准区间跑到**未来**，
        每月的成本校准从此永远拿不到成交数据（或拿到错的）。
        """
        seen = {}
        import app.tasks.cost_calibration as cc
        monkeypatch.setattr(cc, "run_monthly_calibration",
                            lambda ds, start, end, write_path=None:
                            (seen.update(start=start, end=end), None)[1])
        sched_mod.monthly_cost_calibration_job()

        today = date.today()
        first_this = today.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        assert seen["end"] == last_prev.isoformat(), (
            f"校准区间的终点是 {seen['end']}，应为上月最后一天 {last_prev}")
        assert seen["start"] == last_prev.replace(day=1).isoformat()
        assert seen["end"] < today.isoformat(), "校准区间跑到了今天或未来"
