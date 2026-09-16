"""
变异测试（隔离版 + 可续跑）—— 回答"断言够不够强"的唯一机械方法。

**关键设计一：绝不修改主工作区。**
每次运行先把 backend/ 复制到临时目录，全部变异都发生在副本里。
主工作区因此始终可提交、可跑测试、可并行做别的事。

来由（DEV_LESSONS §V）：上一版原地改源码再还原，结果
  1) 一次提交恰好发生在变异运行期间 → **变异被提交进仓库**（commit 24c251a）
     + 临时备份 .mutbak（562 行）也被带进去；
  2) 两个变异任务并行 → 互相污染，表现为"基线莫名其妙是红的"。
规则写了却没执行，这一版把它做成机制。

**关键设计二：逐个变异点落盘，断电重启可续跑。**
`--state <file>` 把每个变异点的结论即时写进 JSON；重启后已判定的直接跳过。
一个模块动辄几十分钟到一个多小时（daily_trading_loop 实测 4890 秒），
没有续跑能力就等于每次关机都从头再来。

判读：
  被杀死(killed) = 有测试抓到 → 该处逻辑被断言保护
  存活(survived) = 没测试抓到 → **改坏了也没人知道**（真实盲区）

四个必须的保险（缺一则数字不可信，全部踩过坑，见 MUTATION_LEDGER「工具缺陷史」）：
  1. 变异后必须仍能 ast.parse —— 语法错会让测试立刻红，被误记为"杀死"
  2. 替换行必须保留行尾换行 —— 否则与下一行粘连，同样制造假"杀死"
  3. 变异器必须真的生效 —— 不要用 re.sub 去改 m.group(0)，正向后顾断言会失配
  4. 每行每个变异器各算一个变异点 —— 只取第一个会让算术变异永远轮不到

用法:
  python mutate.py <目标模块相对 backend 的路径> <测试路径...> [--state <进度文件>]
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# tools/mutation/mutate.py → parents[2] == backend/
# 不写死绝对路径：换机器、交付给别人复核时都要能直接跑。
REAL_BACKEND = Path(__file__).resolve().parents[2]

# 复制时跳过的目录/后缀（体积大且与测试无关）
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", "backups", "pit_store",
             "node_modules", ".git"}
SKIP_SUFFIX = {".db", ".db-wal", ".db-shm", ".mutbak", ".parquet"}

MUTATORS = [
    # 逻辑
    (r"(?<![<>=!])<(?![=<])",           "<=",    "< -> <=        边界放宽"),
    (r"(?<![<>=!\-])>(?![=>])",         ">=",    "> -> >=        边界放宽"),
    (r"\bnot\s+",                       "",      "删掉 not       条件取反"),
    (r"\band\b",                        "or",    "and -> or      条件放宽"),
    (r"\bTrue\b",                       "False", "True -> False"),
    (r"\bFalse\b",                      "True",  "False -> True"),
    # 算术：记账/成本代码里符号错一个方向，钱就算反了
    (r"(?<=[\w\)\]])\s-\s(?=[\w\(])",   " + ",   "- -> +         符号反转"),
    (r"(?<=[\w\)\]])\s\+\s(?=[\w\(])",  " - ",   "+ -> -         符号反转"),
    (r"(?<=[\w\)\]])\s\*\s(?=[\w\(])",  " / ",   "* -> /         量纲反转"),
    (r"np\.maximum\(",                  "np.minimum(", "maximum -> minimum"),
    (r"np\.minimum\(",                  "np.maximum(", "minimum -> maximum"),
    (r"\.cumprod\(",                    ".cumsum(",    "cumprod -> cumsum"),
    (r"np\.abs\(",                      "(",           "删掉 np.abs    丢符号"),
    # ---- 工具缺陷 #11（外部审计 2026-09-15）补上的九个 ----
    # 审计用四个微型探针指出：`x >= .70`、`x / w`、`a+b` 都是 **0 个变异点**，
    # `a + b + c` 只变第一个加号。也就是说"所选变异全部被处理"与"检出能力已证明"
    # 从来不是同一个命题 —— B-1 的除法归一化错误、C-2 的对象身份错误，
    # 本来就不在这套算子能表达的范围里。
    # 下面补的是**能表达**的那部分；对象身份、数值常量仍在盲区（见 README）。
    (r"(?<![<>=!])>=(?!=)",             ">",     ">= -> >        边界收紧"),
    (r"(?<![<>=!])<=(?!=)",             "<",     "<= -> <        边界收紧"),
    (r"(?<![<>=!])==(?!=)",             "!=",    "== -> !=       相等取反"),
    (r"(?<![<>=!])!=(?!=)",             "==",    "!= -> ==       相等取反"),
    (r"(?<=[\w\)\]])\s/\s(?=[\w\(])",   " * ",   "/ -> *         量纲反转"),
    # 无空格写法：`a+b` / `w/total` 这类此前一个变异点都没有
    (r"(?<=[\w\)\]])\+(?=[\w\(])",      "-",     "+ -> -（无空格）"),
    (r"(?<=[\w\)\]])-(?=[\w\(])",       "+",     "- -> +（无空格）"),
    (r"(?<=[\w\)\]])\*(?=[\w\(])",      "/",     "* -> /（无空格）"),
    (r"(?<=[\w\)\]])/(?=[\w\(])",       "*",     "/ -> *（无空格）"),
]


def _ignore(dirpath, names):
    out = set()
    for n in names:
        if n in SKIP_DIRS or any(n.endswith(s) for s in SKIP_SUFFIX):
            out.add(n)
    return out


#: 仓库根目录下、测试会读到的文件。**必须一并复制进沙箱。**
#:
#: 工具缺陷 #12：沙箱只复制 `backend/`，而 `tests/meta` 里有检查仓库根
#: `.gitignore` 的用例（`TestLessonV::test_mutbak_is_gitignored`）。
#: 于是只要测试路径包含 `tests/meta`，**基线在沙箱里必然是红的** ——
#: 而 `make_plan.py` 给每个模块都加了 `tests/meta`。
#: 旧判定器遇到这种情况只打一句"基线就是红的"就 return，模块**静默没测**，
#: 外面看不出与"跑过了"的区别。这条是新的基线守卫（缺陷 #10 的修复）抓出来的。
REPO_ROOT_FILES = (".gitignore",)


def make_sandbox() -> Path:
    """把 backend/ 复制到临时目录；返回副本根路径。"""
    tmp = Path(tempfile.mkdtemp(prefix="mut_sandbox_"))
    dst = tmp / "backend"
    shutil.copytree(REAL_BACKEND, dst, ignore=_ignore)
    repo_root = REAL_BACKEND.parent
    for name in REPO_ROOT_FILES:
        src = repo_root / name
        if src.is_file():
            shutil.copy2(src, tmp / name)
    return dst


def candidate_lines(src: str):
    """挑出值得变异的行（跳过注释/docstring/import/日志/签名）。"""
    out = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    doc = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) \
                and isinstance(n.value.value, str):
            doc.update(range(n.lineno, (n.end_lineno or n.lineno) + 1))
    # 多行 logger(...) 调用的**续行**也要跳过：里面的 < > 多半在格式串里
    logger_span = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = getattr(f, "attr", "") if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in {"debug", "info", "warning", "error", "exception", "critical"}:
                logger_span.update(range(n.lineno, (n.end_lineno or n.lineno) + 1))
    for i, line in enumerate(src.splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#") or i in doc or i in logger_span:
            continue
        if s.startswith(("import ", "from ", "logger", "logging",
                         "def ", "async def ", "class ", "@")):
            continue
        if "logger." in s or '"""' in s or "'''" in s or "->" in s:
            continue
        out.append((i, line))
    return out


def _string_spans(line):
    """
    返回该行中**非代码区间**：字符串字面量（含 f-string）**与行尾注释**。
    变异落在其中只改文本，无信息量 —— 必须排除，否则会记出假的"存活"。

    注释这条是真被坑过：`deficit = row_target - (...)   # >0 需放大自由名`
    的 `>` 在注释里，变异器把它改成 `>=`，源码语义分毫未变，却记成一个"存活"。

    注意 Python 3.12：f-string 被拆成 FSTRING_START / FSTRING_MIDDLE / FSTRING_END，
    只认 tokenize.STRING 会漏掉 f"...{x} > {y}" 里的比较符号（上一版就是这么漏的）。
    """
    import io
    import tokenize
    spans = []
    FS_START = getattr(tokenize, "FSTRING_START", None)
    FS_MID = getattr(tokenize, "FSTRING_MIDDLE", None)
    FS_END = getattr(tokenize, "FSTRING_END", None)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(line).readline))
    except Exception:
        qs = [i for i in (line.find(chr(34)), line.find(chr(39))) if i >= 0]
        return [(min(qs), len(line))] if qs else []
    fs_open = None
    for tok in toks:
        if tok.type == tokenize.STRING:
            spans.append((tok.start[1], tok.end[1]))
        elif tok.type == tokenize.COMMENT:
            spans.append((tok.start[1], len(line)))
        elif FS_START is not None and tok.type == FS_START:
            fs_open = tok.start[1]
        elif FS_END is not None and tok.type == FS_END and fs_open is not None:
            spans.append((fs_open, tok.end[1]))   # 整个 f-string 视为字符串
            fs_open = None
        elif FS_MID is not None and tok.type == FS_MID:
            spans.append((tok.start[1], tok.end[1]))
    if fs_open is not None:
        spans.append((fs_open, len(line)))        # f-string 跨行未闭合
    return spans


def _multiline_string_lines(src: str) -> dict:
    """
    返回 {行号: [(col_start, col_end), ...]}，覆盖**跨行字符串字面量**
    在每一行上占据的区间。

    【工具缺陷 #9】`_string_spans` 是**逐行**调用 tokenize 的。
    一个模块级三引号字符串（典型例子：`app/agent/_prompts.py` 整个文件
    就是一个 `_SYSTEM_PROMPT = <三引号串>`）的中间各行单独拿去 tokenize
    根本不是合法 Python，走进 except 之后又找不到引号，于是返回空区间 ——
    **整段散文被当成代码变异了一遍**。

    实测后果：`_prompts.py` 报出 35 个"变异点"，全部是提示词正文里的
    not / and / > / *；`alpha_agent.py` 的 _SYSTEM_PROMPT 里也混进一个。
    它们改的是给 LLM 读的英文，不是程序行为，却会以"存活"的身份把击杀率
    拉低，诱使人去为散文写源码文本断言（自伤教训 #6 明令禁止的那种）。

    这里改成对**整份源码**做一次 tokenize，把每个跨行字符串在各行上的
    占位登记下来，供 build_plan 与逐行的 _string_spans 合并使用。
    """
    import io
    import tokenize

    out: dict = {}
    FS_START = getattr(tokenize, "FSTRING_START", None)
    FS_END = getattr(tokenize, "FSTRING_END", None)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except Exception:
        return out

    BIG = 10 ** 6

    def _mark(start, end):
        (r0, c0), (r1, c1) = start, end
        if r1 == r0:
            out.setdefault(r0, []).append((c0, c1))
            return
        out.setdefault(r0, []).append((c0, BIG))
        for r in range(r0 + 1, r1):
            out.setdefault(r, []).append((0, BIG))
        out.setdefault(r1, []).append((0, c1))

    fs_open = None
    for tok in toks:
        if tok.type == tokenize.STRING and tok.end[0] != tok.start[0]:
            _mark(tok.start, tok.end)
        elif FS_START is not None and tok.type == FS_START:
            fs_open = tok.start
        elif FS_END is not None and tok.type == FS_END and fs_open is not None:
            if tok.end[0] != fs_open[0]:
                _mark(fs_open, tok.end)
            fs_open = None
    return out


def _in_string(pos, spans):
    return any(a <= pos < b for a, b in spans)


def build_plan(src: str):
    """
    枚举全部变异点：**每行每个变异器各算一个**，不是"每行只取第一个"。
    旧版每行只做第一个匹配的变异器，而 MUTATORS 里比较符排在算术符前面，
    于是 `shares = abs(dw) * val / p if p > 0` 这种行永远只测 `>`，乘法从不被测。
    """
    plan = []
    # 工具缺陷 #9：跨行字符串（模块级三引号常量）的中间各行，逐行 tokenize
    # 认不出来 —— 先对整份源码做一次，把它们的占位登记下来。
    ml = _multiline_string_lines(src)
    for lineno, line in candidate_lines(src):
        spans = _string_spans(line) + ml.get(lineno, [])
        seen = set()
        for pat, rep, desc in MUTATORS:
            # 工具缺陷 #11（外部审计 2026-09-15）：旧版对每个变异器**只取本行
            # 第一处匹配**（找到就 break）。于是 `return a + b + c` 只有第一个
            # 加号被测过，第二个改坏了不会有任何变异点覆盖它。
            # 现在枚举全部出现位置，各算一个变异点。
            for m in re.finditer(pat, line):
                if _in_string(m.start(), spans):
                    continue
                # 直接按位置替换整段匹配文本。
                # 【不要改回 re.sub(pat, rep, m.group(0))】：m.group(0) 不含
                # lookaround 消耗的字符，把它单独送进 re.sub，**正向**后顾断言
                # (?<=[\w\)\]]) 前面没有字符必然失配 → 原样返回 → mutated == line
                # → 被静默记成"无可变异"。三个算术变异器（* + -）因此从未生效过。
                mutated = line[:m.start()] + rep + line[m.end():]
                if mutated == line or mutated in seen:
                    continue
                seen.add(mutated)
                plan.append((lineno, line, desc, mutated, m.start()))
    return plan


#: 每个变异点的 pytest 超时（秒）。默认值见 main() 里的 --timeout。
#:
#: 【为什么不能用大默认值】alpha_workflows 那一轮用 1800s 跑了 11.6 小时才判完
#: 12 个变异点 —— 平均 58 分钟/点，**比超时上限还长**。原因是
#: `subprocess.run(timeout=...)` 在超时后只 kill 直接子进程，pytest 派生的
#: 孙进程还握着 stdout 管道，`communicate()` 于是继续阻塞到孙进程自己退出。
#: 超时上限形同虚设。下面改成显式杀进程树。
#:
#: 超时判为"击杀"是对的：变异让测试跑不完（死循环/无限重试）本身就是被检出。
#: 但上限必须远高于基线才不会误判慢而正确的变异 —— 取基线的 10 倍以上。
DEFAULT_TIMEOUT = 300


def _kill_tree(proc: subprocess.Popen) -> None:
    """连孙进程一起杀。Windows 上用 taskkill /T，POSIX 上杀进程组。"""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=60)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


#: pytest 的退出码语义（`_pytest.config.ExitCode`）。
#: **只有 1（有测试失败）才算「断言抓到了」。**
_VERDICT_BY_EXITCODE = {
    0: "survived",        # 全过 → 改坏了没人发现
    1: "killed",          # 有测试失败 → 断言抓到了
    2: "interrupted",     # 收集期中断 / KeyboardInterrupt
    3: "internal_error",  # pytest 内部错误
    4: "usage_error",     # 命令行用法错（测试路径不存在等）
    5: "no_tests",        # 一个测试都没收集到 —— 分母是空的
}


def run_tests(sandbox: Path, test_paths, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """
    在副本里跑测试，返回 `{"verdict": ..., "exit_code": ..., "output": ...}`。

    工具缺陷 #10（外部审计 2026-09-15 指出）
    ----------------------------------------
    旧版是 `return proc.wait(timeout) == 0`：**退出码只要非 0 就当成「杀死」**，
    超时也 `return False`（同样算杀死），而且 `stdout/stderr=DEVNULL` 把证据全丢了。
    于是这些都会被记成「断言抓到了」：

      - 收集错误 / 导入失败（exit 2、4）——变异让模块 import 不了
      - 一个测试都没收集到（exit 5）——测试路径写错时分母为空
      - 超时（环境卡住、死循环）
      - 与本变异无关的偶发失败（还叠加了 `-x`：第一个失败就停）

    **它们的共同点是都让击杀率变好看。** 度量工具的偏置方向永远朝着
    「结果更漂亮」，这是本项目第 10 次踩同一类坑（见 #1、#2、#5、#9）。

    现在按退出码分类，非 1 的一律**不算杀死**，并保留输出尾部供复核。
    """
    popen_kw = {}
    if os.name == "nt":
        popen_kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kw["start_new_session"] = True

    proc = subprocess.Popen(
        [sys.executable, "-m", "pytest", *test_paths, "-x", "-q",
         "-p", "no:randomly", "--no-header", "-W", "ignore"],
        cwd=sandbox, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", **popen_kw,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, _ = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            out = ""
        return {"verdict": "timeout", "exit_code": None,
                "output": (out or "")[-2000:]}

    return {"verdict": _VERDICT_BY_EXITCODE.get(code, "unknown"),
            "exit_code": code, "output": (out or "")[-2000:]}


def _load_state(path: Path | None) -> dict:
    if path and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(path: Path | None, state: dict) -> None:
    """
    整份 state 覆盖写。

    ⚠️ **同一个 state 文件不能被两个并发进程共用**：各自持有一份内存快照，
    最后写入者会把对方已完成的模块整段抹掉。实际发生过——两批并行重测共用
    progress_final.json，先完成那批的 12 个模块结果被后一批清空了。
    并发跑请给每批一个**独立的 state 文件**（隔离副本本身是并发安全的，
    不安全的只有这个共享状态文件）。
    """
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    argv = sys.argv[1:]
    state_path = None
    if "--state" in argv:
        i = argv.index("--state")
        state_path = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    timeout = DEFAULT_TIMEOUT
    if "--timeout" in argv:
        i = argv.index("--timeout")
        timeout = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    rel_target, tests = argv[0], argv[1:]

    print(f"目标: {rel_target}")
    print(f"测试: {' '.join(tests)}")
    print(f"单点超时: {timeout}s")
    state = _load_state(state_path)
    verdicts: dict = state.setdefault(rel_target, {}).setdefault("mutants", {})
    if verdicts:
        print(f"续跑：已判定 {len(verdicts)} 个变异点，跳过之", flush=True)

    print("建立隔离副本（主工作区不会被修改）...", flush=True)
    sandbox = make_sandbox()
    target = sandbox / rel_target
    assert target.exists(), f"副本里找不到 {rel_target}"

    original = target.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    plan = build_plan(original)
    print(f"副本: {sandbox}\n变异点: {len(plan)}")
    print("先确认基线为绿 ...", flush=True)
    base = run_tests(sandbox, tests, timeout)
    if base["verdict"] != "survived":
        # 基线必须是 exit 0。旧版只判 `!= 0` 就打印一句提示后 return，
        # 调用方（runner.py）看不出区别 —— 收集错误、路径写错（exit 4/5）
        # 与"测试真的红了"混成一件事，而后两者意味着**整轮测量的分母是空的**。
        print(f"!! 基线不是绿的：verdict={base['verdict']} "
              f"exit={base['exit_code']}，无法做变异测试")
        print(base["output"][-1200:])
        shutil.rmtree(sandbox.parent, ignore_errors=True)
        raise SystemExit(
            f"基线失败（{base['verdict']}）—— 拒绝在不可信的基线上测量。"
            f"这不是『本模块击杀率为 0』，是**根本没测**。")

    state[rel_target]["tests"] = tests
    state[rel_target]["planned"] = len(plan)
    invalid = 0
    t0 = time.time()
    try:
        for lineno, line, desc, mutated, col in plan:
            # key 里必须带列号：同一行同一个变异器可能有多处匹配（工具缺陷 #11
            # 修复后），不带列号会让第二处覆盖第一处的结论。
            key = f"L{lineno}|c{col}|{desc.split()[0]}"
            if key in verdicts:
                continue
            if not mutated.endswith("\n"):
                mutated += "\n"
            cand_src = "".join(lines[:lineno - 1] + [mutated] + lines[lineno:])
            try:
                ast.parse(cand_src)
            except SyntaxError:
                invalid += 1
                continue
            target.write_text(cand_src, encoding="utf-8")
            r = run_tests(sandbox, tests, timeout)
            verdicts[key] = {"verdict": r["verdict"], "exit_code": r["exit_code"],
                             "desc": desc, "code": line.strip()[:96]}
            if r["verdict"] not in ("survived", "killed"):
                # 超时 / 收集错误 / 没收集到测试 —— **既不算杀死也不算存活**。
                # 旧版把它们全算成"杀死"，击杀率因此系统性偏高（工具缺陷 #10）。
                verdicts[key]["output_tail"] = r["output"][-600:]
            _save_state(state_path, state)          # 逐个落盘 → 断电可续
            if r["verdict"] == "survived":
                print(f"  存活 L{lineno:<5d} {desc}  |  {line.strip()[:56]}", flush=True)
            elif r["verdict"] != "killed":
                print(f"  ?? L{lineno:<5d} {desc}  |  {r['verdict']} "
                      f"(exit={r['exit_code']})", flush=True)
            target.write_text(original, encoding="utf-8")     # 立即还原副本
    finally:
        shutil.rmtree(sandbox.parent, ignore_errors=True)      # 副本用完即毁

    survived = [(k, v) for k, v in verdicts.items() if v["verdict"] == "survived"]
    killed = len(verdicts) - len(survived)
    total = len(verdicts)
    rate = (killed / total * 100) if total else 0.0
    state[rel_target].update({"killed": killed, "survived": len(survived),
                              "total": total, "kill_rate": round(rate, 1)})
    _save_state(state_path, state)

    print("\n" + "=" * 70)
    print(f"有效变异 {total} | 杀死 {killed} | **存活 {len(survived)}** | 击杀率 {rate:.1f}%")
    print(f"（语法非法 {invalid} · 本次用时 {time.time()-t0:.0f}s）")
    if survived:
        print("\n存活变异 = 测试盲区：")
        for k, v in survived:
            print(f"  {k:<12s} {v['desc']}")
            print(f"          {v['code']}")


if __name__ == "__main__":
    main()
