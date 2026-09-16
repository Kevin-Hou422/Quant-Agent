# 变异测试工具（断言强度的度量与复核）

放进仓库而不是临时目录，是因为一件事：**交付要能被原样复跑**。
"这些击杀率是怎么量出来的"必须由对方自己跑一遍来确认，而不是听我口述 ——
本项目已有 6 个口头数字被工具缺陷推翻（见下面的缺陷史）。

## 为什么需要它

`N passed` 只说明"以本套件**能察觉**的方式没有变坏"，说不了本套件能察觉多少。
行覆盖率只说明代码**被执行**过，说不了执行时**有没有被断言**。
只有变异击杀率直接度量检出能力：改坏一处，测试会不会红。

本项目实测：多个模块首测击杀率在 0%–44% 之间 —— 即**改坏 10 处只有 0–4 处
会被发现**，而当时这些模块的测试全是绿的。

## 用法

```bash
cd backend

# 1) 生成计划（模块清单变了就重新生成，不要手工维护）
python tools/mutation/make_plan.py                    # 全量：89 模块 / 2843 点
python tools/mutation/make_plan.py app/core/gp_engine # 只要某个前缀

# 2) 跑（可续跑：关机重启后再跑一次同样的命令即可接上）
python tools/mutation/runner.py plan_full.json --state progress_full.json

# 3) 只看进度，不跑
python tools/mutation/runner.py plan_full.json --state progress_full.json --status

# 4) 补完用例后单点复核，不必重跑整模块
python tools/mutation/verify_mutant.py app/core/xxx.py 123 "a > b" "a >= b" \
    tests/unit/xxx/test_xxx.py
```

`progress_full.json` 是可续跑的**中间态**，已在 `.gitignore` 里，不要提交。
测量结论请更新 `tests/meta/measured_modules.json` —— 那份才是交付物，
并由 `tests/meta/test_invariants.py::TestEveryModuleIsMeasured` 对账：
`app/` 下任何有变异点却不在那份清单里的模块都会让测试变红。

## 这套工具**看不见**什么（先读这一节）

变异器是一组有限的正则规则，**不是对任意产品缺陷的枚举**。
外部审计 2026-09-15 用四个微型探针戳破过一次：

| 探针 | 修复前 | 修复后 |
|---|---:|---:|
| `return x >= .70` | **0** 个变异点 | 1 |
| `return x / w` | **0** | 1 |
| `return a+b`（无空格） | **0** | 1 |
| `return a + b + c` | 1（只第一个加号） | 2 |
| `return x <= 5`（自查补充） | **0** | 1 |
| `return a == b`（自查补充） | **0** | 1 |

补齐后 `app/` 的全量点数从 1980 涨到 **2843**。**多出来的 863 个从未测量**，
差额记在 `tests/meta/measured_modules.json` 的 `measurement_scope` 块里。

**仍在盲区**（下列缺陷这套工具**不可能**发现，别把它的击杀率当成覆盖证明）：

- 数值常量本身（阈值 0.9 写成 0.7、窗口 20 写成 60）
- 对象身份与 deepcopy 语义 —— 已登记的 **C-2** 就是这一类
- 调用实参的增删、顺序、关键字名
- 控制流结构（提前 return、循环边界、异常捕获范围）—— **D-6** 就是这一类
- 频率/量纲这种"公式整体错了"的问题 —— **N-4** 就是这一类

已登记的 32 条缺陷里，`B-1`（除法归一化）、`C-2`、`N-4` 都是**读代码**发现的。
**"所选变异全部被处理" ≠ "检出能力已证明"。**

## 达标标准（不是击杀率本身）

> **每一个存活变异，要么被新测试杀死，要么有书面且可机械验证的等价性证明。**

击杀率单看会误导：`risk_gate` 76.2% 已达标（剩余 10 处全部证明为等价变异），
而曾经口头报过的 100% 是工具缺陷造成的假数。

"可机械验证"指证明本身也是一条可执行断言，而不是在注释里写一句"我认为它们等价"。

**但可执行 ≠ 证对了。** 本文档原来举的例子就是反面教材：
`(limit + tol) - limit != tol` 曾被当作"epsilon 守卫的区分值在浮点上不可构造"的
证明，外部审计 2026-09-15 一个反例就推翻了它 ——
`project_to_capped_l1([[1e-12, 1-1e-12]])` 逐位保留 1e-12，`>` 与 `>=` 结论不同。
那条断言证的是**"tol 不能由一次加法还原"**，而到达被测值的路径根本不必是加法。

所以证明还必须写清**"被测的值可能从哪里来"**并逐条尝试反驳；
站不住的移入 `REFUTED_EQUIVALENCE`，由
`test_invariants.py::test_refuted_proofs_cannot_quietly_come_back` 盯着，不许写回去。

等价性证明写在对应测试文件的 `PROVEN_EQUIVALENT` 字典里，
并由 `tests/meta/test_lessons_enforced.py::TestLessonX_EquivalenceProofsAreMechanical`
强制：说明少于 40 字、点名了不存在的验证用例、或缺少存活数声明，都会红。

## 工具缺陷史（每一条都曾让整批数字作废）

度量工具的缺陷**不报错，只静默缩小分母** —— 这是最危险的一类，
因为数字看起来完全合理。十二条都留在这里，是为了下次"数字看起来合理"时
不要重新相信它。

| # | 缺陷 | 后果 |
|---|---|---|
| 1 | 变异后没有 `ast.parse` 校验 | 语法错让测试立刻红，被**误记为"杀死"** |
| 2 | 替换行丢了行尾换行 | 与下一行粘连，同样制造假"杀死" |
| 3 | 原地改主工作区，不建隔离副本 | 变异残留被提交进仓库（commit 24c251a 真实发生过） |
| 4 | 每行只取第一个变异器 | 比较符排在算术符前面，乘除法**永远轮不到** |
| 5 | `re.sub(pat, rep, m.group(0))` | `m.group(0)` 不含 lookaround 消耗的字符，**正向后顾断言**在孤立片段上必然失配 → `*` `+` `-` 三个算术变异器**从未生效过**且不报错。`transaction_cost` 129 个候选行只有 22 行被变异过 |
| 6 | 不屏蔽行尾注释 | 注释里的 `>` `<` 被变异，记出假存活项 |
| 7 | `subprocess.run(timeout=)` 只杀直接子进程 | pytest 的孙进程握着管道，`communicate()` 继续阻塞 → 超时上限形同虚设（`alpha_workflows` 一轮跑了 11.6 小时）。改用 `taskkill /F /T` 杀进程树 |
| 8 | 改完测试没有重测 | 台账里的存活列表过期，照着它补用例等于在补已经修好的洞 |
| 9 | `_string_spans` **逐行** tokenize | 模块级三引号字符串的中间各行单独 tokenize 不是合法 Python，走进 `except` 后返回空区间 → **整段散文被当成代码变异**。`_prompts.py` 因此报出 35 个假变异点。改为对整份源码 tokenize 一次（`_multiline_string_lines`） |
| 10 | `return proc.wait(timeout) == 0` —— **退出码非 0 一律算"杀死"**，超时也算，输出还丢进 DEVNULL | 收集错误（exit 2/4）、一个测试都没收集到（exit 5，**分母为空**）、超时、以及 `-x` 之下任何无关的偶发失败，全被记成"断言抓到了"。**偏置方向永远朝着数字更好看。** 已改为按 pytest 退出码分类：只有 exit 1 算杀死，其余进 `inconclusive` 并从分母剔除；基线不绿直接 `SystemExit` |
| 11 | 算子集合有系统性缺口，且每行每算子**只取第一处**匹配 | 见上面「这套工具看不见什么」。已补 9 个算子 + 枚举全部匹配位置 |
| 12 | `make_sandbox()` 只复制 `backend/`，仓库根的 `.gitignore` 不在沙箱里 | `tests/meta` 里有检查仓库根 `.gitignore` 的用例，于是**只要测试路径包含 `tests/meta`，沙箱里的基线必然是红的** —— 而 `make_plan.py` 给每个模块都加了 `tests/meta`。旧判定器遇到这种情况只打一句「基线就是红的」就 return，模块**静默没测**，从外面看不出与「跑过了」的区别。**这一条是缺陷 #10 的修复（基线不绿就 SystemExit）当场抓出来的** |

## 四个必须的保险（对应 #1/#2/#4/#5）

1. 变异后必须仍能 `ast.parse`
2. 替换行必须保留行尾换行
3. 每行每个变异器各算一个变异点
4. 不要用 `re.sub` 去改 `m.group(0)`

后两条由 `tests/meta/test_lessons_enforced.py::TestLessonW_KillRateLedgerIsMaintained`
持续强制；隔离副本（#3）也在那里检查。

## 中间态文件去哪了

早期每轮测量各写一个 `plan_*.json` + `progress_*.json`，攒下约 100 个文件。
它们对复核没有价值 —— 复核者要的是"照同一套规则重跑一遍"，
不是当时分了几批、每批叫什么名字。已全部删除，用 `make_plan.py` 重新生成。

唯一保留下来的结论在 `tests/meta/measured_modules.json`（每模块的点数、
击杀数、存活数、来源），它是被测试对账的交付物。
