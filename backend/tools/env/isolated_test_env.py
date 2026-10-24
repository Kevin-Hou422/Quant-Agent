"""
isolated_test_env.py — OneDrive 之外按 requirements.lock 建的独立测试环境（2026-10-22 用户决定 D3）

用户原话："同意创建 OneDrive 外的独立 lock 测试环境。可以使用 C:\\QuantAgentEnv；已有目录则保留并新建子目录。
不升级现有交易环境，不修改系统默认 Python，不复制活 .env。"

    py -3.12 tools/env/isolated_test_env.py create --root C:\\QuantAgentEnv --name lock-test
    py -3.12 tools/env/isolated_test_env.py stage  --root C:\\QuantAgentEnv --name lock-test
    py -3.12 tools/env/isolated_test_env.py run    --root C:\\QuantAgentEnv --name lock-test -- -q

create  在 <root>/<name> 下建 venv（基础解释器必须是 CI 的 3.12；不碰系统默认 Python、不碰仓库里的交易 venv），
        `pip install -r backend/requirements.lock`（不加 --upgrade），装完 pip freeze 逐条对 lock、pip check，
        结果写进 <root>/<name>/environment.json。根目录已有内容保留；<name> 已存在 → 拒绝。
stage   把仓库工作树（git 跟踪的 + 没被忽略的新文件）镜像到 <root>/<name>/repo。
        为什么要镜像：测试从仓库的 backend/ 跑会读到那里的活 .env（Settings 的 env_file=".env"）。
        `.env*` 一律不拷（`*.example` 模板除外）；工作树里删掉的文件也从镜像里删掉。
run     在镜像的 backend/ 里用这个 venv 跑 pytest；子进程只带白名单环境变量（系统里设的配置变量不进测试）。

只用标准库：用任何 3.12 解释器都能跑，不依赖要被安装的包。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional

REQUIRED_PYTHON = "3.12"
STAGED_LIST = ".staged_files.json"
CLOUD_MARKERS = ("onedrive", "dropbox", "google drive", "icloud")
CHILD_ENV_KEYS = ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "USERPROFILE",
                  "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES",
                  "PROGRAMFILES(X86)", "COMMONPROGRAMFILES", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
                  "OS", "COMPUTERNAME", "USERNAME")

Runner = Callable[..., subprocess.CompletedProcess]


class EnvRefused(RuntimeError):
    """目标位置 / 解释器不符合 D3 的约束。"""


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inside(child: Path, parent: Path) -> bool:
    c, p = os.path.normcase(str(child)), os.path.normcase(str(parent))
    return c == p or c.startswith(p.rstrip("\\/") + os.sep)


def check_root(root: Path, repo: Path, environ: Mapping[str, str]) -> None:
    root = Path(os.path.abspath(root))
    synced = [Path(v) for k, v in environ.items() if k.lower().startswith("onedrive") and v]
    if any(m in str(root).lower() for m in CLOUD_MARKERS) or any(_inside(root, s) for s in synced):
        raise EnvRefused(f"{root} 在云同步目录里 —— 独立测试环境要放在 OneDrive 之外（例如 C:\\QuantAgentEnv）")
    if _inside(root, Path(os.path.abspath(repo))):
        raise EnvRefused(f"{root} 在仓库 {repo} 里 —— 独立测试环境不能和仓库 / 交易环境混在一起")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins(text: str) -> Dict[str, tuple]:
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" in line:
            name, ver = line.split("==", 1)
            out[_norm(name.strip())] = (name.strip(), ver.strip())
        else:                                                   # `name @ file://...` 一类：不是版本号
            name = re.split(r"[ @]", line, 1)[0]
            out[_norm(name)] = (name, None)
    return out


def lock_drift(lock_text: str, freeze_text: str) -> List[str]:
    """lock 里的每一条都必须原样装上；返回差异（空 = 一致）。"""
    want, got = _pins(lock_text), _pins(freeze_text)
    drift = []
    for key, (name, ver) in want.items():
        if key not in got:
            drift.append(f"{name}: lock {ver}, 没装上")
        elif got[key][1] != ver:
            drift.append(f"{name}: lock {ver}, 装上的是 {got[key][1]}")
    return drift


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" or (venv / "Scripts").exists() else "bin/python")


def create(root: Path, name: str, python: str, lock: Path, repo: Path, runner: Runner = subprocess.run,
           environ: Optional[Mapping[str, str]] = None) -> dict:
    environ = os.environ if environ is None else environ
    check_root(root, repo, environ)
    target = Path(root) / name
    if target.exists():
        raise EnvRefused(f"{target} 已存在 —— 不覆盖，换一个 --name")
    run = lambda cmd: runner([str(c) for c in cmd], capture_output=True, text=True)   # noqa: E731
    ver = run([python, "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"]).stdout.strip()
    if not ver.startswith(REQUIRED_PYTHON + "."):
        raise EnvRefused(f"基础解释器 {python} 是 {ver or '未知版本'}，不是 CI 的 {REQUIRED_PYTHON}")
    target.mkdir(parents=True)
    venv = target / "venv"
    mk = run([python, "-m", "venv", venv])
    if mk.returncode != 0:
        raise EnvRefused(f"建 venv 失败：{mk.stderr.strip()}")
    vpy = _venv_python(venv)
    inst = run([vpy, "-m", "pip", "install", "--no-input", "--disable-pip-version-check", "-r", lock])
    freeze = run([vpy, "-m", "pip", "freeze"]).stdout
    check = run([vpy, "-m", "pip", "check"])
    head = run(["git", "-C", repo, "rev-parse", "HEAD"]).stdout.strip()
    drift = lock_drift(Path(lock).read_text(encoding="utf-8"), freeze)
    man = {
        "ok": inst.returncode == 0 and check.returncode == 0 and not drift,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_python": str(python), "python_version": ver, "venv": str(venv),
        "lock": str(lock), "lock_sha256": sha256(lock), "repo_head_at_create": head,
        "pip_install_ok": inst.returncode == 0, "pip_install_tail": (inst.stderr or inst.stdout or "")[-2000:],
        "pip_check_ok": check.returncode == 0, "pip_check_output": (check.stdout or "").strip(),
        "drift": drift, "freeze": freeze.splitlines(),
        "note": "不含活 .env；测试源码用 stage 镜像（.env* 不拷），run 只带白名单环境变量",
    }
    (target / "environment.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


def tree_files(repo: Path) -> List[str]:
    """仓库工作树里的文件：git 跟踪的 + 没被 .gitignore 忽略的新文件（只读 git 命令）。"""
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=str(repo),
                         capture_output=True, check=True).stdout.decode("utf-8")
    return sorted({f for f in out.split("\0") if f and (Path(repo) / f).is_file()})


def _refused(rel: str) -> bool:
    name = Path(rel).name
    return name.startswith(".env") and not name.endswith(".example")


def stage(src: Path, dest: Path, files: Iterable[str]) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    listing = dest / STAGED_LIST
    previous = set(json.loads(listing.read_text(encoding="utf-8"))) if listing.exists() else set()
    copied, refused = [], []
    for rel in files:
        if _refused(rel):
            refused.append(rel)
            continue
        s, d = Path(src) / rel, dest / rel
        st = s.stat()
        if not (d.exists() and d.stat().st_size == st.st_size and d.stat().st_mtime_ns == st.st_mtime_ns):
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
        copied.append(rel)
    removed = sorted(previous - set(copied))
    for rel in removed:
        (dest / rel).unlink(missing_ok=True)
    listing.write_text(json.dumps(sorted(copied), ensure_ascii=False), encoding="utf-8")
    return {"copied": len(copied), "refused": refused, "removed": removed}


def child_env(base: Mapping[str, str], venv: Path) -> Dict[str, str]:
    keep = {k.upper() for k in CHILD_ENV_KEYS}
    env = {k: v for k, v in base.items() if k.upper() in keep}
    other_venv = base.get("VIRTUAL_ENV", "")
    path = [p for p in base.get("PATH", "").split(os.pathsep)
            if p and not (other_venv and _inside(Path(p), Path(other_venv)))]
    env["PATH"] = os.pathsep.join([str(Path(venv) / "Scripts"), str(Path(venv) / "bin")] + path)
    env["VIRTUAL_ENV"] = str(venv)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def main(argv: Optional[List[str]] = None) -> int:
    repo = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="OneDrive 之外按 requirements.lock 建的独立测试环境")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("create", "stage", "run"):
        p = sub.add_parser(c)
        p.add_argument("--root", default=r"C:\QuantAgentEnv")
        p.add_argument("--name", default="lock-test")
        if c == "create":
            p.add_argument("--python", default=None, help="3.12 解释器路径；缺省用 `py -3.12` 找")
        if c == "run":
            p.add_argument("pytest_args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    target = Path(a.root) / a.name
    try:
        if a.cmd == "create":
            py = a.python or subprocess.run(["py", f"-{REQUIRED_PYTHON}", "-c", "import sys; print(sys.executable)"],
                                            capture_output=True, text=True).stdout.strip()
            man = create(Path(a.root), a.name, py, repo / "backend" / "requirements.lock", repo)
            print(json.dumps({k: v for k, v in man.items() if k != "freeze"}, ensure_ascii=False, indent=1))
            return 0 if man["ok"] else 1
        check_root(Path(a.root), repo, os.environ)
        if not (target / "environment.json").exists():
            raise EnvRefused(f"{target} 不是用 create 建的环境")
        if a.cmd == "stage":
            print(json.dumps(stage(repo, target / "repo", tree_files(repo)), ensure_ascii=False))
            return 0
        args = [x for x in a.pytest_args if x != "--"]
        venv = target / "venv"
        return subprocess.run([str(_venv_python(venv)), "-m", "pytest", *args], cwd=str(target / "repo" / "backend"),
                              env=child_env(os.environ, venv)).returncode
    except EnvRefused as exc:
        print(f"拒绝：{exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
