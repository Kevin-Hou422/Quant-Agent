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
