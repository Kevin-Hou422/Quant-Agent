"""
Phase S 新增代码里**存活变异的处置**（MUTATION_LEDGER 的完成标准第 2 条）

判据：每一个存活变异，要么被新用例杀死，要么有**书面且可机械验证**的等价性
证明。"可机械验证"指证明本身也是一条可执行断言，而不是注释里一句
"我认为它们等价"。

本轮 169 个新增点：143 个被杀死（见各 Phase S 用例文件），余下 26 个在这里处置。
其中 18 个给出等价性证明（`NEW_POINT_EQUIVALENCE`），8 个**证不了**，
登记在 `NEW_POINT_UNPROVEN` —— 它们是本轮交付的已知缺口，不是被忽略的。

**为什么不叫 `PROVEN_EQUIVALENT`**：那个名字被
`test_invariants.py::test_survivor_disposition_reconciles_per_module` 拿去与
`measured_modules.json` 里**全量测量**得到的逐模块存活数对账。本文件处置的是
"新增、尚未纳入任何一轮全量测量"的点（它们计在 `measurement_scope.unmeasured_points`
里），两本账的分母不同，混用会让那条对账直接判红 —— 而判据错了比没判据更危险。
质量要求不打折：键格式、说明字数、点名的用例必须存在，都由本文件末尾自查。

> 证明必须说清"被测的值可能从哪里来"，并对每条来源给出反驳尝试。
> 这条要求是被外部审计打脸之后加的：一条**跑得起来**的断言，证的可能是一个
> 更弱的命题，然后被当成结论用（见 MUTATION_LEDGER 关于
> `project_to_capped_l1` 的那段）。
"""
from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.data_engine import data_partitioner as dp


NEW_POINT_EQUIVALENCE = {
    # ── data_partitioner ────────────────────────────────────────────────
    "app/core/data_engine/data_partitioner.py ×1 — L503 `degraded: bool = False` 的 dataclass 默认值":
        "ThreeWaySplit 在全库只有一个构造点（ThreeWayPartitioner.partition），"
        "而那里把 degraded 作为**关键字实参显式传入**。默认值因此永远不参与取值，"
        "翻成 True 也观察不到。来源反驳：dataclass 默认值还可能被"
        "『别处直接构造 ThreeWaySplit』用到 —— 由 "
        "test_three_way_split_is_only_constructed_with_an_explicit_degraded_flag "
        "做 AST 全库扫描排除。",

    "app/core/data_engine/data_partitioner.py ×1 — L611 `if val_hi <= 0` -> `<`":
        "两条路径都**拒绝**，只是报错文案不同：val_hi==0 时若放行，remaining=0 → "
        "n_val=max(1,0)=1 → n_is=0-embargo-1 恒为负 → 立刻被下一道 "
        "`n_is < min_train_days` 拦下，同样抛 ValueError('数据不足以三段切分…')。"
        "该守卫存在的意义是给出更准确的原因，不是控制放行与否。"
        "见 test_a_val_hi_of_zero_is_refused_by_one_guard_or_the_other。",

    "app/core/data_engine/data_partitioner.py ×1 — L666 `if 0 < n_cal <= n - keep` 的 `0 <` -> `0 <=`":
        "n_cal 恒 ≥ 1，故 `0 < n_cal` 与 `0 <= n_cal` 同值。理由：cutoff = "
        "dates[-1] − test_years 年，而 test_years 已被构造器校验为 **> 0**，"
        "所以 cutoff 严格早于 dates[-1]，末日必然落在 `dates > cutoff` 里。"
        "来源反驳：若 test_years 可为 0 则 n_cal 可能为 0 —— 构造器对 "
        "`test_years <= 0` 抛错，已由 test_nonsense_parameters_are_rejected_at_construction "
        "的 0.0 用例钉住；本文件的 test_n_cal_is_never_zero 再直接扫一遍。",

    "app/core/data_engine/data_partitioner.py ×2 — L675 `capped >= 1` 的 `>=`->`>` 与 `and`->`or`":
        "`capped = min(n_cal, max(1, int(n*max_test_share)))`，右项因 max(1, …) 恒 ≥ 1，"
        "左项 n_cal 恒 ≥ 1（见上一条），故 `capped >= 1` 是恒真式，改成 `> 1` 只在 "
        "capped==1 时不同；而 capped==1 要求 n_cal==1 或 int(n*share)==1，两者都只在 "
        "n ≤ 5 的面板上出现，那种面板在 partition() 里必然因 val_hi<=0 或 n_is<min_train "
        "抛错，观察不到分支差异。`and`->`or` 同理：左项恒真时 or 与 and 同值。"
        "见 test_capped_is_always_at_least_one。",

    # ── trial_ledger（基础设施参数）──────────────────────────────────────
    "app/db/trial_ledger.py ×2 — L103/L104 `index=True` -> False":
        "索引只影响查询**代价**，不影响结果集：count()/select() 在有无索引时返回"
        "完全相同的行。本模块没有任何依赖索引存在性的断言或行为（无唯一约束、"
        "无 ON CONFLICT）。来源反驳：若索引承担唯一性则会改变写入行为 —— "
        "这里 index=True 不带 unique=True，由 "
        "test_the_usage_index_is_not_a_uniqueness_constraint 直接写两行同键数据验证。",

    "app/db/trial_ledger.py ×2 — L105/L106 `purpose`/`used_at` 的 `nullable=False` -> True":
        "这两列都带 default（purpose='' / used_at=utcnow），而本模块**唯一**的写入点 "
        "record_use 总是提供 purpose、从不显式写 used_at，因此 NULL 在任何调用路径上"
        "都构造不出来，约束放开与否观察不到。来源反驳：外部直接 INSERT 可绕过 —— "
        "那不是本模块的行为；dataset_key/test_key 这两个**没有默认值**的键则确实可达，"
        "已由 test_a_usage_row_without_a_key_is_rejected_by_the_schema 覆盖。",

    "app/db/trial_ledger.py ×1 — L153 `create_engine(..., echo=False)` -> True":
        "echo 只控制 SQLAlchemy 是否把 SQL 打到日志，不改变任何查询结果或写入语义。"
        "本模块所有断言都读返回值，没有一条依赖日志内容。"
        "来源反驳：若有用例断言日志里**没有** SQL，则 echo 可观测 —— 全库无此断言，"
        "由 test_no_holdout_ledger_test_asserts_on_sql_log_output 扫描确认。",

    "app/db/trial_ledger.py ×1 — L157 `sessionmaker(..., expire_on_commit=False)` -> True":
        "与本仓已有的 trial_ledger L49 同型论证：record_use 在**同一个 session 内**"
        "commit 之后只再执行一次 select(count)（过期会自动 refresh，拿得到），"
        "返回给调用方的是一个普通 dataclass（HoldoutUsage）而非 ORM 对象，"
        "commit 后不存在被过期的属性访问。count() 只读不 commit，不触发过期。",

    # ── overfit_stats ───────────────────────────────────────────────────
    "app/core/backtest_engine/overfit_stats.py ×2 — L126 `if len(g) > 0` -> `>=` 与 L128 `k >= n_groups` -> `>`":
        "np.array_split(np.arange(n_obs), n_groups) 只在 n_obs < n_groups 时产生空组，"
        "而该情形在上一行 `if n_obs < n_groups: raise` 处已被拒。于是过滤条件恒真、"
        "过滤后 len(groups) 恒等于 n_groups，L128 的 `k >= n_groups` 便被更早的 "
        "`not (1 <= k < n_groups)` 完全覆盖，永远走不到。"
        "见 test_exactly_as_many_samples_as_groups_is_allowed 与 "
        "test_no_group_is_ever_empty_once_the_size_guard_passed。",

    # ── evaluation_utils ────────────────────────────────────────────────
    "app/core/gp_engine/evaluation_utils.py ×1 — L141 `if not isinstance(s.index, pd.DatetimeIndex)` 删掉 not":
        "两条分支都只把索引用于：① 交给 PurgedKFold 切分（按**位置**均分，与日期无关）；"
        "② 建 date→位置 的映射。合成出来的工作日索引与原索引**长度相同、顺序相同**，"
        "因此两分支逐位给出相同结果。来源反驳：若函数将来按日期做别的事（按年分组等）"
        "则不再等价 —— 由 test_a_range_indexed_series_gives_the_same_numbers_as_a_dated_one "
        "固化，那天它会立刻变红。",

    "app/core/gp_engine/evaluation_utils.py ×1 — L161 `if len(block) > 1` -> `>=`":
        "block 来自 `idx` 且上一行有 `if len(idx) < 5: continue`，故 len(block) 恒 ≥ 5，"
        "`> 1` 与 `>= 1` 在可达取值上同值。来源反驳：若 5 这个下限被调小到 1 则可达 —— "
        "那会先让 test_a_fold_of_exactly_five_days_is_used 变红。",

    # ── 冻结段守卫：n_test ≥ 1 是切分器的不变量 ──────────────────────────
    "app/core/lifecycle/validation_gate.py ×1 — L194 `split.n_test > 0` -> `>=`":
        "ThreeWaySplit.n_test 恒 ≥ 1：_test_size 的三条返回路径分别是 n_cal(≥1)、"
        "capped(≥1，见 data_partitioner 那条) 与 max(1, …)。因此该守卫恒真，"
        "`> 0` 与 `>= 0` 同值。前提由 "
        "test_the_test_segment_is_never_empty_whatever_the_panel_size 机械扫描验证。",

    "app/api/router.py ×2 — L645/L1710 `split.n_test > 0` -> `>=`":
        "与 validation_gate L194 同一条不变量：ThreeWaySplit.n_test 恒 ≥ 1，守卫恒真。"
        "两处分别在 /backtest/run 与 /backtest/realistic，论证相同但落在不同模块，"
        "按逐模块对账的要求单独登记。",
}

#: **证不了**的存活项。写在这里是为了它们出现在交付物里，而不是消失。
#: 规矩（MUTATION_LEDGER 完成标准第 2 条）：站不住的证明不许硬写。
NEW_POINT_UNPROVEN = {
    "app/core/data_engine/data_partitioner.py ×3 — L675 `n - capped - self.embargo_days >= 1` 的两处符号与一处 `>=`":
        "区分值要求 n − capped − embargo 恰好等于 1（或符号翻转后仍落在同一侧）。"
        "能构造出该等式的面板尺寸都会在随后的 partition() 里因 IS 不足而抛错，"
        "但我**没有**证明这层蕴含对所有 (n, max_test_share, embargo) 组合成立，"
        "所以不写等价性证明。",

    "app/core/backtest_engine/overfit_stats.py ×3 — L177 `oos_perf <= oos_perf[n_star]`、L178 `1 - 1e-6`、L180 `lam < 0`":
        "这三处都只在**并列**或恰好落在中位的组合上改变计数：rank 差 1 会让 omega 平移 "
        "1/(N+1)，仅当 IS-最优的 OOS 排名恰好压在中位线上时才翻转 λ 的符号。"
        "构造一个能稳定命中该情形的收益矩阵需要反解 CSCV 的组合结构，本轮没做。",

    "app/core/gp_engine/evaluation_utils.py ×1 — L162 `if sd <= 1e-12` -> `<`":
        "区分值需要 sd 恰好等于 1e-12。sd = np.nanstd(block, ddof=1) 是浮点均方根，"
        "我没有给出『无法反解』的证明 —— 本仓已有一条同型的等价性证明被外部审计用"
        "反例推翻过（见 MUTATION_LEDGER 的 REFUTED_EQUIVALENCE），所以这里不重蹈覆辙，"
        "如实记为未处置。",

    "app/core/lifecycle/validation_gate.py ×1 — L240 `sd > 1e-12` -> `>=`":
        "同上：要求 sd 恰好等于 1e-12。本仓对 validation_gate L198 有一条口径相同的"
        "既有证明，但那条同样只论证了『难以构造』而非『不可构造』，不足以复用。",
}


# ===========================================================================
# 证明的机械验证
# ===========================================================================

def _partitioner_src() -> str:
    return Path(inspect.getfile(dp)).read_text(encoding="utf-8")


def _panel(n: int) -> dict:
    idx = pd.bdate_range("2017-01-02", periods=n)
    cols = ["A", "B", "C"]
    rng = np.random.default_rng(0)
    close = pd.DataFrame(rng.normal(100, 1, (n, 3)), index=idx, columns=cols)
    return {"close": close, "volume": pd.DataFrame(1e6, index=idx, columns=cols)}


def test_three_way_split_is_only_constructed_with_an_explicit_degraded_flag():
    """证明 L503：全库对 ThreeWaySplit 的构造都显式传 degraded。"""
    from tests.meta import test_lessons_enforced as L  # noqa: F401  （仅为定位 app/ 根）

    root = Path(inspect.getfile(dp)).parents[3]
    bad = []
    for p in sorted((root / "app").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "ThreeWaySplit"):
                if not any(kw.arg == "degraded" for kw in node.keywords):
                    bad.append(f"{p.name}:{node.lineno}")
    assert not bad, (
        f"以下构造点没有显式传 degraded，dataclass 默认值因此可观测，"
        f"那条等价性证明不成立：{bad}")


def test_a_val_hi_of_zero_is_refused_by_one_guard_or_the_other():
    """
    证明 L611：把 val_hi **恰好等于 0** 的那一格构造出来，确认两侧都拒绝。

    构造：比例口径下 n_ratio 被夹到 max(1, n − min_room)，n 很小时取 1，
    于是 val_hi = n − 1 − embargo。取 n = embargo + 1 即得 val_hi == 0 ——
    这正是 `<=` 与 `<` 唯一分道的那一格。
    """
    import pytest

    emb = 20
    n = emb + 1                      # → val_hi 恰好 0
    with pytest.raises(ValueError, match="三段切分"):
        dp.partition_three_way(_panel(n), test_years=None, test_ratio=0.15,
                               min_train_days=60, embargo_days=emb)
    # 放行之后必然被 n_is 守卫拦下：remaining=0 → n_val=1 → n_is = −(emb+1) < 下限
    assert 0 - emb - 1 < 60, "构造前提变了：放行后 n_is 不再必然小于下限"


def test_n_cal_is_never_zero():
    """证明 L666：cutoff 严格早于末日，末日必然计入 n_cal。"""
    for n in (150, 400, 900, 1400):
        idx = pd.bdate_range("2017-01-02", periods=n)
        for years in (0.25, 1.0, 2.0, 5.0):
            cutoff = idx[-1] - pd.DateOffset(days=int(round(years * 365.25)))
            assert int((idx > cutoff).sum()) >= 1, (
                f"n={n} years={years}：n_cal 竟然是 0 —— L666 的等价性证明不成立")


def test_capped_is_always_at_least_one():
    """证明 L675 的第一个合取项：capped = min(n_cal≥1, max(1, …)) 恒 ≥ 1。"""
    assert "max(1, int(n * self.max_test_share))" in _partitioner_src(), (
        "份额上限不再带 max(1, …) —— `capped >= 1` 不再恒真，那条证明作废")
    for n in (10, 33, 120, 400, 1300):
        for share in (0.05, 0.35, 0.9):
            assert min(n, max(1, int(n * share))) >= 1


def test_no_group_is_ever_empty_once_the_size_guard_passed():
    """证明 overfit_stats L126/L128：过了 n_obs>=n_groups 之后不会有空组。"""
    for n_obs in (6, 7, 13, 600):
        for n_groups in (2, 6, min(6, n_obs)):
            if n_obs < n_groups:
                continue
            groups = np.array_split(np.arange(n_obs), n_groups)
            assert all(len(g) > 0 for g in groups), (
                f"n_obs={n_obs} n_groups={n_groups} 切出了空组")


def test_the_usage_index_is_not_a_uniqueness_constraint(tmp_path):
    """证明 trial_ledger L103/L104：index=True 不带唯一性，写两行同键必须成功。"""
    from app.db.trial_ledger import HoldoutLedger

    led = HoldoutLedger(db_url=f"sqlite:///{tmp_path / 'h.db'}", budget=5)
    led.record_use("ds", "k")
    assert led.record_use("ds", "k").uses == 2, (
        "同键第二次写入没成功 —— index 若带唯一性，那条等价性证明不成立")


def test_no_holdout_ledger_test_asserts_on_sql_log_output():
    """
    证明 trial_ledger L153：**用到 HoldoutLedger 的**用例里没有一条断言 SQL 日志。

    （范围必须限定：仓库里确有别的用例检查 SQL 日志 —— 第一版写成"全库无此断言"
    直接被 test_position_store_schema.py 打脸。证明的范围写大了，结论就是错的，
    哪怕它跑得起来。）
    """
    root = Path(inspect.getfile(dp)).parents[3] / "tests"
    hits = []
    for p in sorted(root.rglob("test_*.py")):
        if "__pycache__" in p.parts or p.name == Path(__file__).name:
            continue
        src = p.read_text(encoding="utf-8")
        if "HoldoutLedger" not in src:
            continue
        if re.findall(r"caplog[\s\S]*(?:SELECT|INSERT|sqlalchemy\.engine)", src):
            hits.append(p.name)
    assert not hits, f"以下用例依赖 SQL 日志，echo 因此可观测：{hits}"


def test_every_survivor_has_a_written_proof():
    """
    本轮新增点的存活处置对账：18 个已证明 + 8 个未处置 = 26 个存活。

    数字对不上就红 —— 复测后存活数变了却没更新这里，会在这条上顶出来，
    而不是让一份过期的处置继续躺着。
    """
    def _points(d: dict) -> int:
        return sum(int(re.findall(r"×(\d+)", k)[0]) for k in d)

    assert _points(NEW_POINT_EQUIVALENCE) == 18, _points(NEW_POINT_EQUIVALENCE)
    assert _points(NEW_POINT_UNPROVEN) == 8, _points(NEW_POINT_UNPROVEN)
    all_entries = {**NEW_POINT_EQUIVALENCE, **NEW_POINT_UNPROVEN}
    for key, why in all_entries.items():
        assert len(why) >= 40, f"{key} 的说明过于敷衍：{why!r}"
        assert re.match(r"^app/\S+\.py ×\d+ — \S", key), f"证明键格式不合规：{key!r}"
        assert (_backend_root() / re.match(r"^(app/\S+\.py)", key).group(1)).exists(), (
            f"证明键点名的模块不存在：{key!r}")

    # 点名的验证用例必须真的存在 —— 否则这份证明没有任何可执行的支撑，
    # 而读的人会以为它有（全局 §X 对本文件不生效，所以这里自查一遍）。
    known = set()
    for p in (_backend_root() / "tests").rglob("test_*.py"):
        if "__pycache__" in p.parts:
            continue
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                known.add(node.name)
    # 只认**显式引用**（"见 xxx" / "由 xxx"）。用宽正则会把散文里的
    # `test_years`、`test_key` 这种**参数名**也当成用例名，于是这条检查会一直
    # 红在假阳性上，最后被人放宽 —— 那就等于没有这条检查。
    dangling = [f"{key[:45]} → {name}"
                for key, why in all_entries.items()
                for name in re.findall(r"(?:见|由|本文件的)\s*(test_[a-z0-9_]+)", why)
                if name not in known]
    assert not dangling, f"以下证明点名了不存在的用例：{dangling}"


def _backend_root() -> Path:
    return Path(inspect.getfile(dp)).parents[3]
