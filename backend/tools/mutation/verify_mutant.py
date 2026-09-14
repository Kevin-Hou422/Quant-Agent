"""
单点变异验证 —— 补一条用例之后，确认它**确实**杀得死那一个变异，
而不必重跑整个模块（daily_trading_loop 一轮实测 4890 秒）。

与 mutate.py 同样在隔离副本里做，主工作区零改动。

用法:
  python verify_mutant.py <模块相对 backend 的路径> <行号> <要替换的片段> <替换成> <测试路径...>

例:
  python verify_mutant.py app/tasks/daily_trading_loop.py 520 "t > 0" "t >= 0" \
      tests/test_daily_loop_survivors.py

判读:
  基线绿 + 变异红  → 该变异被杀死 ✅
  基线绿 + 变异绿  → 仍是盲区    ❌
  基线红           → 测量无效（先修测试）

注意：用它得到的结论若与某一轮全量测量的输出不一致（例如用例是在那一轮开跑
之后才补的），必须在 MUTATION_LEDGER 里写明"哪一轮的数字被这样修正过"，
否则台账里会出现一个跟任何一次运行输出都对不上的数字。
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mutate import make_sandbox, run_tests  # noqa: E402


def main() -> None:
    rel, lineno, old, new = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    tests = sys.argv[5:]

    sandbox = make_sandbox()
    target = sandbox / rel
    assert target.exists(), f"副本里找不到 {rel}"
    original = target.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    line = lines[lineno - 1]
    assert old in line, f"L{lineno} 里没有片段 {old!r}\n实际内容: {line!r}"

    mutated = line.replace(old, new, 1)
    src = "".join(lines[:lineno - 1] + [mutated] + lines[lineno:])
    ast.parse(src)                       # 语法错会被误记成"杀死"

    print(f"目标 : {rel} L{lineno}")
    print(f"原句 : {line.strip()}")
    print(f"变异 : {mutated.strip()}")
    try:
        print("基线 ...", flush=True)
        if not run_tests(sandbox, tests):
            print("!! 基线就是红的，本次验证无效")
            return
        print("基线绿。施加变异 ...", flush=True)
        target.write_text(src, encoding="utf-8")
        survived = run_tests(sandbox, tests)
    finally:
        shutil.rmtree(sandbox.parent, ignore_errors=True)

    print("\n结论: " + ("[X] 变异存活 —— 仍是测试盲区" if survived
                        else "[OK] 变异被杀死"))


if __name__ == "__main__":
    main()
