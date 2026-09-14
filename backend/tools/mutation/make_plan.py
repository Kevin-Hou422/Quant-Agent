"""
生成变异测试计划 `plan_full.json`（覆盖 app/ 下全部有变异点的模块）。

为什么要有这个脚本
------------------
早期每一轮测量都手写一个 `plan_*.json`，加上各自的 `progress_*.json`，
最后攒下 ~100 个中间态文件。它们对**复核**没有价值：复核者要的是
"照着同一套规则重跑一遍"，不是我当时分了几批、每批叫什么名字。

所以中间态全部删除，改成这份**可重新生成**的规则化计划：

    模块 app/<pkg>/<sub>/<mod>.py  →  测试 tests/unit/<sub>/ + tests/meta/
                                       （若 manifest 里记了专属测试文件，用它）

用法（在 backend/ 下）::

    python tools/mutation/make_plan.py                 # 全量
    python tools/mutation/make_plan.py app/core/gp_engine   # 只要某个前缀

随后::

    python tools/mutation/runner.py plan_full.json --state progress_full.json

`progress_full.json` 是可续跑的中间态，**不要提交** —— 它已在 .gitignore 里。
测量结论请更新 `tests/meta/measured_modules.json`（那份才是交付物）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[1]

sys.path.insert(0, str(HERE))
import mutate  # noqa: E402

MANIFEST = BACKEND / "tests" / "meta" / "measured_modules.json"


def _test_targets(rel: str, recorded: list[str]) -> list[str]:
    """
    模块 → 要跑哪些测试。

    规则：**同名包的单元测试目录 + meta**，再**并上** manifest 里记录的
    专属测试文件。

    为什么是并集而不是"优先用记录的那一组"：manifest 里的 `tests` 是
    当初某一**批次**实际用的测试集（比如复测只跑了一个文件），
    拿它当全量会把分母之外的检出能力漏掉 —— 复核者会得到一个偏低的击杀率，
    然后以为测试比实际更弱。取并集只会多跑，不会少算。
    """
    parts = Path(rel).parts                      # app/core/gp_engine/fitness.py
    pkg = parts[-2] if len(parts) >= 2 else ""
    targets = []
    if (BACKEND / "tests" / "unit" / pkg).is_dir():
        targets.append(f"tests/unit/{pkg}")
    else:
        targets.append("tests/unit")
    targets.append("tests/meta")
    targets += [t for t in recorded
                if t.endswith(".py") and (BACKEND / t).is_file()]
    # 已被目录覆盖的具体文件就不用再单列
    dirs = tuple(t for t in targets if not t.endswith(".py"))
    targets = [t for t in targets
               if not t.endswith(".py") or not t.startswith(dirs)]
    return sorted(set(targets))


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "app"
    recorded = {}
    if MANIFEST.exists():
        recorded = {
            k: v.get("tests") or []
            for k, v in json.loads(
                MANIFEST.read_text(encoding="utf-8"))["modules"].items()
        }

    queue = []
    for p in sorted((BACKEND / "app").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(BACKEND).as_posix()
        if not rel.startswith(prefix):
            continue
        n = len(mutate.build_plan(p.read_text(encoding="utf-8")))
        if n == 0:
            continue
        queue.append({"module": rel, "points": n,
                      "tests": _test_targets(rel, recorded.get(rel, []))})

    out = {
        "_doc": ("覆盖 app/ 下全部有变异点的模块。由 tools/mutation/make_plan.py "
                 "生成 —— 模块清单变了就重新生成，不要手工维护。"),
        "_generated_by": "python tools/mutation/make_plan.py",
        "totals": {"modules": len(queue),
                   "points": sum(q["points"] for q in queue)},
        "queue": sorted(queue, key=lambda q: -q["points"]),
    }
    dest = HERE / "plan_full.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"写入 {dest.relative_to(BACKEND)}："
          f"{out['totals']['modules']} 模块 / {out['totals']['points']} 变异点")


if __name__ == "__main__":
    main()
