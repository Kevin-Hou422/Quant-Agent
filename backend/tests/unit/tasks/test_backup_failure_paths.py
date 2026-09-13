"""
tasks/backup.py —— 备份失败路径与保留策略的定钉测试（变异测试驱动）

来由：14 个变异点，首测击杀率 **57.1%**（存活 6）—— 存活的全是**失败路径**。

备份这件事的特点是：**成功时没人看，失败时才要紧**，而失败恰恰是最少被测的。
`ok=False` 被改成 True 之后，一次什么都没备到的运行会报"成功"，
日志、返回值、调度器统统看不出异常 —— 等到真要恢复时才发现没有快照。
PIT 是前向积累的数据，丢了买不回来。

存活项：
  - 两处 `res.ok = False`（主库快照失败、PIT 快照失败）
    —— 改成 True 就是"失败报成功"
  - `if sch_db and sch_db.exists():` 的 `and`
    —— 放宽成 `or` 会在 URL 为空时去 `None.exists()` 直接崩
  - `if pit.exists() and any(pit.rglob("*.parquet")):` 的 `and`
    —— 放宽成 `or` 会去打包一个不存在的目录
  - `dest_base.parent.mkdir(parents=True, exist_ok=True)` 的 exist_ok
    —— 改成 False 时第二次备份直接抛 FileExistsError
  - `[d for d in ... if d.is_dir() and d.name.startswith("snap-")]` 的 `and`
    —— 放宽成 `or` 会把**任意文件**也当成快照目录去 rmtree

既有覆盖（test_backup）测的是"能备份、能清理"，全是成功路径。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import app.tasks.backup as B


def _make_sqlite(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE t (a INTEGER)")
    con.execute("INSERT INTO t VALUES (1)")
    con.commit()
    con.close()
    return p


@pytest.fixture
def env(tmp_path, monkeypatch):
    """一套完整可备份的环境：主库 + 调度库 + 非空 PIT。"""
    from app.config import settings
    main_db = _make_sqlite(tmp_path / "live" / "main.db")
    sch_db = _make_sqlite(tmp_path / "live" / "sched.db")
    pit = tmp_path / "pit_store" / "px" / "year=2024"
    pit.mkdir(parents=True)
    (pit / "data.parquet").write_bytes(b"not-a-real-parquet")

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{main_db}", raising=False)
    monkeypatch.setattr(settings, "scheduler_db_url", f"sqlite:///{sch_db}", raising=False)
    monkeypatch.setattr(settings, "pit_store_dir",
                        str(tmp_path / "pit_store"), raising=False)
    monkeypatch.setattr(settings, "backup_dir", str(tmp_path / "backups"), raising=False)
    monkeypatch.setattr(settings, "backup_keep_n", 14, raising=False)
    return {"root": tmp_path, "main_db": main_db, "sch_db": sch_db,
            "pit": tmp_path / "pit_store", "backups": tmp_path / "backups"}


# ===========================================================================
# A. 失败必须报失败
# ===========================================================================

class TestFailuresAreReported:

    def test_a_healthy_run_reports_ok(self, env):
        """先证明 ok=True 这条路走得通，否则下面的断言可能恒成立。"""
        res = B.run_daily_backup()
        assert res.ok is True, f"完整环境下备份失败：{res.errors}"
        assert res.bytes_written > 0 and res.items

    def test_a_failed_main_db_snapshot_marks_the_run_failed(self, env, monkeypatch):
        """
        `except Exception: res.ok = False; res.errors.append(...)` ——
        那个 `False` 改成 True 就是**失败报成功**：主库（因子/审批/成交的
        审计资产）一次都没备上，而返回值说 ok。
        """
        def _boom(src, dest):
            raise RuntimeError("disk full")

        monkeypatch.setattr(B, "snapshot_sqlite", _boom)
        res = B.run_daily_backup()
        assert res.ok is False, "主库快照抛异常，备份仍然报成功"
        assert any("main_db" in e for e in res.errors), res.errors

    def test_a_missing_main_db_marks_the_run_failed(self, env, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "database_url", "sqlite:///nope.db", raising=False)
        res = B.run_daily_backup()
        assert res.ok is False, "主库不存在却报成功"
        assert any("main_db" in e for e in res.errors)

    def test_a_failed_pit_snapshot_marks_the_run_failed(self, env, monkeypatch):
        """PIT 是**最不可再生**的部分，它失败必须让整次备份判失败。"""
        def _boom(src, dest):
            raise RuntimeError("archive error")

        monkeypatch.setattr(B, "snapshot_tree", _boom)
        res = B.run_daily_backup()
        assert res.ok is False, "PIT 快照抛异常，备份仍然报成功"
        assert any("pit" in e for e in res.errors), res.errors

    def test_a_failed_scheduler_db_is_recorded_but_not_fatal(self, env, monkeypatch):
        """
        调度库可重建，失败只记 errors、不翻 ok —— 这个**不对称**是有意的，
        改成一致会让可重建资产的故障拖垮整次备份。
        """
        real = B.snapshot_sqlite

        def _selective(src, dest):
            if "sched" in Path(src).name:
                raise RuntimeError("locked")
            return real(src, dest)

        monkeypatch.setattr(B, "snapshot_sqlite", _selective)
        res = B.run_daily_backup()
        assert res.ok is True, "调度库失败不该让整次备份判失败"
        assert any("scheduler_db" in e for e in res.errors), (
            f"调度库失败没有记进 errors：{res.errors}")

    def test_ok_distinguishes_the_two_outcomes(self, env, monkeypatch):
        """ok 恒 True 或恒 False 时这条相等断言会成立 —— 必须不相等。"""
        good = B.run_daily_backup().ok
        monkeypatch.setattr(B, "snapshot_sqlite",
                            lambda src, dest: (_ for _ in ()).throw(RuntimeError("x")))
        bad = B.run_daily_backup().ok
        assert good != bad, "成功与失败给出了同一个 ok 值"


# ===========================================================================
# B. 可选项的存在性守卫
# ===========================================================================

class TestOptionalSources:

    def test_absent_scheduler_db_is_skipped_not_crashed(self, env, monkeypatch):
        """
        `if sch_db and sch_db.exists():` —— `and` 放宽成 `or` 会在
        URL 解析出 None 时去调 `None.exists()`，整个备份任务直接崩，
        而它是每天 23:45 自动跑的。
        """
        from app.config import settings
        monkeypatch.setattr(settings, "scheduler_db_url", "", raising=False)
        res = B.run_daily_backup()
        assert res.ok is True, f"没有调度库时备份崩了：{res.errors}"
        assert not any("scheduler_db" in e for e in res.errors)

    def test_empty_pit_is_skipped_without_error(self, env, monkeypatch):
        """
        `if pit.exists() and any(pit.rglob("*.parquet")):` —— `and` 放宽成 `or`
        会去打包一个空目录（甚至不存在的目录），产出一个没内容的 zip
        并报成功 —— 恢复时才发现 PIT 是空的。
        """
        from app.config import settings
        empty = env["root"] / "empty_pit"
        empty.mkdir()
        monkeypatch.setattr(settings, "pit_store_dir", str(empty), raising=False)
        res = B.run_daily_backup()
        assert res.ok is True
        assert "pit_store.zip" not in res.items, (
            "PIT 里没有 parquet 却仍然打了包")

    def test_nonexistent_pit_dir_is_skipped(self, env, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "pit_store_dir",
                            str(env["root"] / "no_such_dir"), raising=False)
        res = B.run_daily_backup()
        assert res.ok is True and "pit_store.zip" not in res.items

    def test_snapshot_tree_creates_missing_intermediate_dirs(self, env, tmp_path):
        """
        `dest_base.parent.mkdir(parents=True, exist_ok=True)` 有**两个**开关：

          - `parents=True` → 中间目录缺失时一并创建。改成 False 后，
            备份目录形如 `<backup_dir>/snap-2026…/pit_store` 而 `<backup_dir>`
            尚不存在时（首次备份、或换了 backup_dir）直接抛 FileNotFoundError。
          - `exist_ok=True` → 同一父目录第二次写不抛。改成 False 后
            **第二天**的备份就失败。

        变异器改的是行内**第一个** True，即 `parents` —— 只测"同一父目录写两次"
        碰不到它（上一版就是这么让它活下来的）。这里两种都构造。
        """
        src = env["pit"]
        # ① 中间目录 deep/ 与 deeper/ 都不存在 → 靠 parents=True 创建
        base = tmp_path / "deep" / "deeper" / "a"
        assert not base.parent.exists()
        B.snapshot_tree(src, base)
        assert base.parent.exists(), "中间目录没有被创建 —— parents 疑似为 False"
        # ② 同一父目录再写一次 → 靠 exist_ok=True
        B.snapshot_tree(src, tmp_path / "deep" / "deeper" / "b")


# ===========================================================================
# C. 快照保留策略
# ===========================================================================

class TestPruning:

    def test_only_snapshot_dirs_are_considered(self, tmp_path):
        """
        `if d.is_dir() and d.name.startswith("snap-")` —— `and` 放宽成 `or`
        会把**任意文件**和**任意目录**都当成快照，随后 `shutil.rmtree` 掉 ——
        备份目录如果指向云盘同步目录（文档里就是这么建议的），
        那会删掉用户放在那儿的别的东西。
        """
        root = tmp_path / "bk"
        root.mkdir()
        for i in range(4):
            (root / f"snap-2024010{i}").mkdir()
        (root / "important.txt").write_text("keep me", encoding="utf-8")
        (root / "not-a-snapshot").mkdir()

        pruned = B.prune_snapshots(root, keep=2)
        assert (root / "important.txt").exists(), "非快照文件被删了"
        assert (root / "not-a-snapshot").exists(), "名字不以 snap- 开头的目录被删了"
        assert len(pruned) == 2, f"应当删掉 2 个最旧的快照，实际 {pruned}"

    def test_newest_snapshots_are_kept(self, tmp_path):
        root = tmp_path / "bk"
        root.mkdir()
        names = [f"snap-2024010{i}" for i in range(5)]
        for n in names:
            (root / n).mkdir()
        B.prune_snapshots(root, keep=2)
        left = sorted(d.name for d in root.iterdir())
        assert left == names[-2:], f"保留的不是最新的两个：{left}"

    def test_keep_zero_or_negative_is_a_noop(self, tmp_path):
        """`if keep <= 0 ... return []` —— 配置成 0 时不该把快照全删光。"""
        root = tmp_path / "bk"
        root.mkdir()
        (root / "snap-20240101").mkdir()
        assert B.prune_snapshots(root, keep=0) == []
        assert (root / "snap-20240101").exists(), "keep=0 时快照被删光了"

    def test_missing_root_is_a_noop(self, tmp_path):
        assert B.prune_snapshots(tmp_path / "nope", keep=3) == []

    def test_exactly_keep_many_snapshots_are_all_kept(self, tmp_path):
        """`dirs[keep:]` 的切片边界：恰好 keep 个时一个都不该删。"""
        root = tmp_path / "bk"
        root.mkdir()
        for i in range(3):
            (root / f"snap-2024010{i}").mkdir()
        assert B.prune_snapshots(root, keep=3) == []
        assert len(list(root.iterdir())) == 3


# ===========================================================================
# D. sqlite 快照的完整性校验
# ===========================================================================

class TestSqliteSnapshot:

    def test_snapshot_is_integrity_checked(self, tmp_path, monkeypatch):
        """
        `if res != "ok": raise` —— 快照做完要跑一次 `PRAGMA integrity_check`。
        这条守卫失效会让一个**损坏的**快照被当成有效备份留下，
        而损坏只有在真要恢复时才暴露。
        """
        src = _make_sqlite(tmp_path / "src.db")
        n = B.snapshot_sqlite(src, tmp_path / "out" / "copy.db")
        assert n > 0 and (tmp_path / "out" / "copy.db").exists()

    def test_a_corrupt_snapshot_raises(self, tmp_path, monkeypatch):
        src = _make_sqlite(tmp_path / "src.db")

        class _FakeCur:
            def fetchone(self):
                return ("malformed database",)

        class _FakeCon:
            def execute(self, *a, **k):
                return _FakeCur()

            def close(self):
                pass

        real_connect = sqlite3.connect

        def _connect(path, *a, **k):
            if "copy.db" in str(path):
                return _FakeCon()
            return real_connect(path, *a, **k)

        monkeypatch.setattr(B.sqlite3, "connect", _connect)
        with pytest.raises(RuntimeError, match="完整性校验失败"):
            B.snapshot_sqlite(src, tmp_path / "out" / "copy.db")
