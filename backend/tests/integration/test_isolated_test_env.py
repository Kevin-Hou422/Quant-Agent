"""
独立的 lock 测试环境（2026-10-22 用户决定 D3）。

用户原话："同意创建 OneDrive 外的独立 lock 测试环境。可以使用 C:\\QuantAgentEnv；已有目录则保留并新建子目录。
不升级现有交易环境，不修改系统默认 Python，不复制活 .env。"

    · 根目录不能在云同步目录 / 仓库里；根目录已有内容 → 保留，只新建子目录；子目录已存在 → 拒绝（不覆盖）
    · 基础解释器必须是 CI 的 3.12；按 requirements.lock 装完后 pip freeze 逐条对 lock，有漂移就判失败
    · 测试从仓库的 backend/ 跑会读到活 .env —— 所以源码也镜像一份，`.env*` 一律不拷（`*.example` 模板除外）
    · 子进程只带白名单环境变量：系统里设的 DATABASE_URL / EXECUTION_MODE 一类配置不进测试
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.env import isolated_test_env as ite

LOCK = "alembic==1.20.0\nAPScheduler==3.11.3\nmoomoo-api==10.10.11\npandas_market_calendars==5.1.1\n"


class FakeRunner:
    """记下每条命令；`-m venv` 真建出解释器文件，`pip freeze` 回 lock 的内容。"""

    def __init__(self, version="3.12.10", freeze=None, check_rc=0):
        self.cmds, self.version, self.freeze, self.check_rc = [], version, freeze, check_rc

    def __call__(self, cmd, **kw):
        self.cmds.append([str(c) for c in cmd])
        out, rc = "", 0
        if "-c" in cmd:
            out = self.version
        elif "venv" in cmd:
            py = Path(cmd[-1]) / "Scripts" / "python.exe"
            py.parent.mkdir(parents=True)
            py.write_text("")
        elif "freeze" in cmd:
            out = self.freeze if self.freeze is not None else LOCK.replace("APScheduler", "apscheduler")
        elif "check" in cmd:
            rc = self.check_rc
        return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr="")


def _lock(tmp_path):
    p = tmp_path / "requirements.lock"
    p.write_text("# 注释行\n" + LOCK, encoding="utf-8")
    return p


class TestWhereItGoes:

    @pytest.mark.parametrize("root", [r"C:\Users\Administrator\OneDrive\QuantAgentEnv",
                                      r"C:\Users\x\OneDrive - Contoso\env"])
    def test_a_cloud_synced_root_is_refused(self, root):
        with pytest.raises(ite.EnvRefused, match="云同步"):
            ite.check_root(Path(root), repo=Path(r"C:\repo"), environ={"OneDrive": r"C:\Users\Administrator\OneDrive"})

    def test_a_root_inside_the_repository_is_refused(self, tmp_path):
        with pytest.raises(ite.EnvRefused, match="仓库"):
            ite.check_root(tmp_path / "repo" / "env", repo=tmp_path / "repo", environ={})

    def test_an_existing_root_is_kept_and_only_a_new_subdirectory_is_made(self, tmp_path):
        root = tmp_path / "QuantAgentEnv"
        root.mkdir()
        (root / "someone_elses.txt").write_text("keep")
        run = FakeRunner()
        ite.create(root, "lock-test", python="py312", lock=_lock(tmp_path), repo=tmp_path / "repo",
                   runner=run, environ={})
        assert (root / "someone_elses.txt").read_text() == "keep"
        assert (root / "lock-test" / "venv" / "Scripts" / "python.exe").exists()

    def test_an_existing_subdirectory_is_not_overwritten(self, tmp_path):
        target = tmp_path / "env" / "lock-test"
        target.mkdir(parents=True)
        (target / "x").write_text("x")
        with pytest.raises(ite.EnvRefused, match="已存在"):
            ite.create(tmp_path / "env", "lock-test", python="py312", lock=_lock(tmp_path),
                       repo=tmp_path / "repo", runner=FakeRunner(), environ={})
        assert (target / "x").read_text() == "x"


class TestWhatGoesIn:

    def test_the_base_interpreter_must_be_the_ci_version(self, tmp_path):
        with pytest.raises(ite.EnvRefused, match="3.12"):
            ite.create(tmp_path / "env", "lock-test", python="py313", lock=_lock(tmp_path),
                       repo=tmp_path / "repo", runner=FakeRunner(version="3.13.1"), environ={})
        assert not (tmp_path / "env" / "lock-test").exists()

    def test_it_installs_the_lock_and_records_what_was_installed(self, tmp_path):
        run = FakeRunner()
        lock = _lock(tmp_path)
        man = ite.create(tmp_path / "env", "lock-test", python="py312", lock=lock, repo=tmp_path / "repo",
                         runner=run, environ={})
        flat = [" ".join(c) for c in run.cmds]
        assert any("-m venv" in c for c in flat)
        assert any("pip install" in c and "-r" in c and str(lock) in c for c in flat)
        assert any("pip check" in c for c in flat)
        assert all("--upgrade" not in c and "-U " not in c for c in flat), "不升级任何东西"
        assert man["python_version"] == "3.12.10" and man["lock_sha256"] == ite.sha256(lock)
        assert man["drift"] == [] and man["ok"] is True
        assert (tmp_path / "env" / "lock-test" / "environment.json").exists()

    def test_drift_from_the_lock_fails_the_environment(self, tmp_path):
        run = FakeRunner(freeze="alembic==1.20.0\nAPScheduler==3.11.4\nmoomoo-api==10.10.11\n")
        man = ite.create(tmp_path / "env", "lock-test", python="py312", lock=_lock(tmp_path),
                         repo=tmp_path / "repo", runner=run, environ={})
        assert man["ok"] is False
        assert "APScheduler: lock 3.11.3, 装上的是 3.11.4" in man["drift"]
        assert "pandas_market_calendars: lock 5.1.1, 没装上" in man["drift"]

    def test_a_failed_pip_check_fails_the_environment(self, tmp_path):
        man = ite.create(tmp_path / "env", "lock-test", python="py312", lock=_lock(tmp_path),
                         repo=tmp_path / "repo", runner=FakeRunner(check_rc=1), environ={})
        assert man["ok"] is False and man["pip_check_ok"] is False

    def test_lock_names_compare_like_pip_does(self):
        assert ite.lock_drift("Foo_Bar==1.0\n", "foo-bar==1.0\n") == []


class TestSourceMirror:

    def test_the_live_env_file_is_never_copied(self, tmp_path):
        src = tmp_path / "backend"
        (src / "app").mkdir(parents=True)
        names = ["app/config.py", ".env", ".env.local", ".env.example", ".env.forward.example"]
        for f in names:
            (src / f).write_text("x")
        dest = tmp_path / "mirror"
        out = ite.stage(src, dest, files=names)
        assert (dest / "app" / "config.py").exists() and (dest / ".env.example").exists()
        assert (dest / ".env.forward.example").exists()
        assert not (dest / ".env").exists() and not (dest / ".env.local").exists()
        assert sorted(out["refused"]) == [".env", ".env.local"]

    def test_files_dropped_from_the_tree_are_removed_from_the_mirror(self, tmp_path):
        src = tmp_path / "backend"
        src.mkdir()
        (src / "a.py").write_text("a")
        (src / "b.py").write_text("b")
        dest = tmp_path / "mirror"
        ite.stage(src, dest, files=["a.py", "b.py"])
        (dest / "made_by_tests.log").write_text("x")              # 不是镜像拷的：不动
        out = ite.stage(src, dest, files=["a.py"])
        assert not (dest / "b.py").exists() and out["removed"] == ["b.py"]
        assert (dest / "made_by_tests.log").exists()

    def test_the_tree_listing_follows_gitignore(self, tmp_path):
        """在临时目录里建一个小仓库（镜像本身不是 git 仓库，不能拿真实仓库来测）。"""
        repo = tmp_path / "repo"
        for f in ("backend/app/config.py", "backend/.env", "backend/.env.forward.example", "venv/x.py",
                  "frontend/node_modules/m.js", "README.md"):
            (repo / f).parent.mkdir(parents=True, exist_ok=True)
            (repo / f).write_text("x")
        (repo / ".gitignore").write_text("venv/\n.env\nnode_modules/\n")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        files = ite.tree_files(repo)
        assert files == [".gitignore", "README.md", "backend/.env.forward.example", "backend/app/config.py"]


class TestChildEnvironment:

    def test_only_whitelisted_variables_reach_the_tests(self, tmp_path):
        venv = tmp_path / "venv"
        base = {"SYSTEMROOT": r"C:\Windows", "TEMP": r"C:\t", "PATH": r"C:\Windows",
                "DATABASE_URL": "sqlite:///C:/QuantAgentData/alphas.db", "EXECUTION_MODE": "moomoo_paper",
                "EDGAR_USER_AGENT": "someone", "VIRTUAL_ENV": r"C:\repo\venv", "OPENAI_API_KEY": "k"}
        env = ite.child_env(base, venv)
        assert env["SYSTEMROOT"] == r"C:\Windows" and env["TEMP"] == r"C:\t"
        for k in ("DATABASE_URL", "EXECUTION_MODE", "EDGAR_USER_AGENT", "OPENAI_API_KEY"):
            assert k not in env
        assert env["VIRTUAL_ENV"] == str(venv) and env["PATH"].startswith(str(venv / "Scripts"))
        assert env["OPENBLAS_NUM_THREADS"] == "1"


class TestRunTimingIsComparable:
    """
    一次全量回归报了 4.5 小时（同一套用例接着电源只要 23 分钟）：系统睡眠记录显示笔记本中途待机 3 个多小时，
    其余时间在用电池。`run` 要自己量出待机、报出电源，并请求系统别因空闲待机。
    """

    @staticmethod
    def _clock(times):
        it = iter(times)
        return lambda: next(it)

    def test_time_the_machine_spent_suspended_is_measured_and_load_jitter_is_not(self):
        # 每 5 秒一拍：正常 → 晚到 20 秒（负载）→ 中间隔了 3 小时 27 分（待机）→ 正常
        w = ite.SuspendWatch(interval=5.0, slack=30.0, clock=self._clock([0, 5, 30, 35, 35 + 12_420, 12_460]))
        for _ in range(5):
            w.tick()
        assert w.suspended == 12_415.0 and [g["seconds"] for g in w.gaps] == [12_415]

    def test_a_gap_exactly_at_the_slack_is_not_a_suspend(self):
        w = ite.SuspendWatch(interval=5.0, slack=30.0, clock=self._clock([0, 35, 70.5]))
        w.tick()
        assert w.suspended == 0.0
        w.tick()
        assert w.suspended == 30.5

    def test_the_report_says_when_the_duration_cannot_be_compared(self):
        ac, batt = {"ac": True, "battery_pct": 90, "saver": False}, {"ac": False, "battery_pct": 40, "saver": True}
        assert ite.run_report(1383.0, 0.0, ac, ac) == ["[env] 用时 1383 秒（期间机器没有待机）"]
        slept = ite.run_report(16197.0, 12415.0, ac, ac)
        assert "待机 12415 秒" in slept[0] and "实际在跑 3782 秒" in slept[0] and "不能当作测试变慢的证据" in slept[1]
        on_battery = ite.run_report(4400.0, 0.0, ac, batt)
        assert len(on_battery) == 2 and "用电池" in on_battery[1] and "40%" in on_battery[1] and "节电模式" in on_battery[1]
        assert "读不到" in ite.run_report(10.0, 0.0, {"ac": None}, {"ac": None})[1]

    def test_a_run_asks_the_system_to_stay_awake_and_records_itself(self, tmp_path, capsys):
        import json
        calls = []
        runner = FakeRunner()
        code = ite.run_pytest(["py", "-m", "pytest", "-q"], tmp_path, {"PATH": "x"}, tmp_path / ite.RUN_HISTORY, ["-q"],
                              runner=runner, watch=ite.SuspendWatch(interval=0.01),
                              power=lambda: {"ac": False, "battery_pct": 55, "saver": False},
                              awake=lambda on: calls.append(on) or True)
        assert code == 0 and calls == [True, False], "先请求不待机，结束后（含出错）撤销"
        assert "用电池" in capsys.readouterr().out
        (row,) = [json.loads(x) for x in (tmp_path / ite.RUN_HISTORY).read_text(encoding="utf-8").splitlines()]
        assert row["args"] == ["-q"] and row["returncode"] == 0 and row["power_start"]["ac"] is False
        assert row["suspended_s"] == 0.0 and row["keep_awake_requested"] is True

    def test_the_stay_awake_request_is_withdrawn_even_if_the_run_blows_up(self, tmp_path):
        calls = []

        def boom(*a, **k):
            raise OSError("解释器不见了")
        with pytest.raises(OSError):
            ite.run_pytest(["py"], tmp_path, {}, tmp_path / ite.RUN_HISTORY, [], runner=boom,
                           watch=ite.SuspendWatch(interval=0.01), power=lambda: {"ac": True},
                           awake=lambda on: calls.append(on) or True)
        assert calls == [True, False]

    def test_power_status_never_guesses(self):
        st = ite.power_status()
        assert set(st) == {"ac", "battery_pct", "saver"} and st["ac"] in (True, False, None)
