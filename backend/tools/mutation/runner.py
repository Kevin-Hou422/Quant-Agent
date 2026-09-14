"""
按队列跑变异测量，**逐个变异点落盘**，关机重启后原地续跑。

用法（在 backend/ 下）:
  python tools/mutation/make_plan.py                      # 先生成计划
  python tools/mutation/runner.py plan_full.json --state progress_full.json
  python tools/mutation/runner.py plan_full.json --state progress_full.json --status

**每个队列必须给 --state 指定各自的进度文件**，两个队列共用一个会互相整体覆盖
（工具缺陷 #7 就是这么丢掉 12 个模块结果的）。

进度文件是**中间态**，已在 .gitignore 里，不要提交；
结论请更新 tests/meta/measured_modules.json。

进度默认写在 tools/mutation/progress.json：
  - 每个模块记 tests / planned / 每个变异点的 killed|survived / 汇总击杀率
  - mutate.py 每判定一个点就落盘一次，断电只丢当前那一个点

判读与处置规矩（见 MUTATION_LEDGER）：
  存活项必须逐条处置——要么补出能杀死它的用例，要么给出**可机械验证**的
  等价性证明。只有"存活项 100% 处置完毕"才算该模块达标，击杀率本身不是标准。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[1]

# 每一档用**各自**的进度文件。共用一个会互相整体覆盖——A 档踩过：
# 两个批次同时写 progress_final.json，后写的把先写的 12 个模块结果抹掉，
# 进度从 11/24 掉回 4/24（工具缺陷 #7）。
PROGRESS = HERE / "progress.json"


def load_progress() -> dict:
    if PROGRESS.exists():
        try:
            return json.loads(PROGRESS.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def show_status(queue) -> None:
    st = load_progress()
    done = pending = 0
    print(f"{'状态':<6}{'击杀率':>8}{'点数':>7}{'存活':>6}  模块")
    for item in queue:
        m = item["module"]
        s = st.get(m)
        if s and s.get("total") and s.get("total") >= s.get("planned", 0) > 0:
            done += 1
            print(f"{'已测量':<6}{s['kill_rate']:>7.1f}%{s['total']:>7}{s['survived']:>6}  {m}")
        elif s and s.get("mutants"):
            pending += 1
            print(f"{'进行中':<6}{'':>8}{len(s['mutants']):>7}{'':>6}  {m}"
                  f"（已判定 {len(s['mutants'])}/{s.get('planned', '?')}）")
        else:
            pending += 1
            print(f"{'未开始':<6}{'':>8}{'':>7}{'':>6}  {m}")
    print(f"\n已测量 {done} / 待办 {pending}")


def main() -> None:
    global PROGRESS
    argv = sys.argv[1:]
    if "--state" in argv:
        i = argv.index("--state")
        PROGRESS = Path(argv[i + 1])
        if not PROGRESS.is_absolute():
            PROGRESS = HERE / PROGRESS
        argv = argv[:i] + argv[i + 2:]
    # 单点超时（秒）。透传给 mutate.py；不给则用 mutate.py 的默认值。
    # 队列里每个模块的测试选择耗时不同，超时要按**该选择的基线**的 10 倍以上给，
    # 否则慢而正确的变异会被误判成"击杀"（假强度）。
    timeout_args: list[str] = []
    if "--timeout" in argv:
        i = argv.index("--timeout")
        timeout_args = ["--timeout", argv[i + 1]]
        argv = argv[:i] + argv[i + 2:]
    plan_path = HERE / argv[0] if argv and not argv[0].startswith("--") \
        else HERE / "plan_tier_a.json"
    queue = json.loads(plan_path.read_text(encoding="utf-8"))["queue"]
    sys.argv = [sys.argv[0], *argv]          # --status 检测沿用下面的 in 判断

    if "--status" in sys.argv:
        show_status(queue)
        return

    for item in queue:
        module, tests = item["module"], item["tests"]
        st = load_progress().get(module, {})
        if st.get("total") and st.get("total") >= st.get("planned", 0) > 0:
            print(f"[skip] {module} 已测量完（{st['kill_rate']}%）")
            continue
        print(f"\n{'=' * 72}\n[run ] {module}", flush=True)
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, str(HERE / "mutate.py"), module, *tests,
             "--state", str(PROGRESS), *timeout_args],
            cwd=BACKEND, text=True, errors="replace")
        print(f"[done] {module} 用时 {time.time() - t0:.0f}s (exit {r.returncode})",
              flush=True)

    print(f"\n{'=' * 72}\n队列跑完。")
    show_status(queue)


if __name__ == "__main__":
    main()
