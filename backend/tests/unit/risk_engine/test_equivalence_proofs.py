"""
risk_engine 新增点里**存活变异的处置**（Phase R.2）

40 个新增变异点：33 个被杀死（见 test_factor_model.py），余下 7 个在这里处置 ——
6 个给出等价性证明，1 个**证不了**、如实登记。

命名与 Phase S 同：故意**不叫** `PROVEN_EQUIVALENT`。那个名字被
`test_invariants.py::test_survivor_disposition_reconciles_per_module` 拿去与
`measured_modules.json` 里**全量测量**得到的逐模块存活数对账；本模块从未进过
任何一轮全量测量，它的点计在 `measurement_scope.unmeasured_points` 里，
两本账分母不同，混用会让那条对账判红。质量要求不打折，本文件末尾自查。

> 证明要说清"被测的值可能从哪里来"，并对每条来源给出反驳尝试。
"""
from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.risk_engine import factor_model as fm


NEW_POINT_EQUIVALENCE = {
    "app/core/risk_engine/factor_model.py ×1 — L79 `np.divide(..., where=cnt > 0)` -> `>=`":
        "两侧同值：分母是 `np.maximum(cnt, 1)`，cnt==0 时算出 0/1 = 0，"
        "而 `out=np.zeros(...)` 在 where 为假时填的也正是 0。"
        "来源反驳：若分母不再带 maximum(…,1)，cnt==0 会变成 0/0=nan，两侧就不同了 —— "
        "由 test_every_divide_guard_has_a_clamped_denominator 盯住那个前提。",

    "app/core/risk_engine/factor_model.py ×1 — L82 `np.divide(..., where=cnt > 1)` -> `>=`":
        "cnt==1 时：唯一有效样本的离差 v-mu 恒为 0，故分子 (dev**2).sum() 为 0，"
        "无论分母取 max(cnt-1,1)=1 还是被 where 屏蔽，var 都是 0、sd 都是 0。"
        "而该行之后 `usable` 要求 cnt >= 2，这一行本就会被整行清零。"
        "见 test_a_single_valid_observation_yields_zero_exposure。",

    "app/core/risk_engine/factor_model.py ×1 — L134 `px.where(px > 0)` -> `>=`":
        "价格恰好为 0 时，`>` 给 NaN、`>=` 保留 0 再由 np.log 给 -inf。两者都是"
        "**非有限值**，而下游 `_zscore_panel` 用 `np.isfinite` 判有效性，对 NaN 与 "
        "-inf 一视同仁地剔除，逐位产出相同暴露。可观测的差别只有 numpy 的一条 "
        "RuntimeWarning。来源反驳：负价格在两侧都被剔除，只有『恰好 0』这一格不同 —— "
        "由 test_a_zero_price_does_not_leak_negative_infinity 实际注入 0 价验证输出全有限。",

    "app/core/risk_engine/factor_model.py ×2 — L193/L230 `sector_included: bool = False` 的 dataclass 默认值":
        "RiskAttribution 与 RiskModel 在全库的每个构造点都把 sector_included 作为"
        "**关键字实参显式传入**（attribute() 传 self.sector_included，fit_risk_model "
        "传 sec is not None），默认值永远不参与取值。"
        "来源反驳：别处可能直接构造这两个 dataclass 而不传该字段 —— "
        "由 test_sector_flag_is_always_passed_explicitly 做 AST 全库扫描排除。",

    "app/core/risk_engine/factor_model.py ×1 — L306 `if lookback > 0` -> `>=`":
        "lookback==0 时两条分支给出**同一个列表**：Python 切片 `x[-0:]` 等价于 "
        "`x[0:]`，即整段索引，与 else 分支的 list(prices.index) 逐位相同。"
        "负的 lookback 在两侧都走 else。见 test_a_zero_lookback_means_the_whole_history。",
}

#: **证不了**的存活项 —— 写在这里是为了它出现在交付物里，而不是消失。
NEW_POINT_UNPROVEN = {
    "app/core/risk_engine/factor_model.py ×1 — L85 `(sd > 1e-12)` -> `>=`":
        "区分值需要 sd 恰好等于 1e-12。sd 是逐行离差平方和开方得到的浮点数，"
        "我没有给出『无法反解出这样一行输入』的证明。本仓已有一条同型的等价性"
        "证明被外部审计用反例推翻过（见 MUTATION_LEDGER 的 REFUTED_EQUIVALENCE），"
        "所以这里不硬写，如实记为未处置。",
}


# ===========================================================================
# 证明的机械验证
# ===========================================================================

def _src() -> str:
    return Path(inspect.getfile(fm)).read_text(encoding="utf-8")


def _backend_root() -> Path:
    return Path(inspect.getfile(fm)).parents[3]


def test_every_divide_guard_has_a_clamped_denominator():
    """
    L79/L82 证明的前提：**每一处** np.divide 的分母都被 maximum(…, 1) 兜住，
    且都带 where= 守卫。分母一旦不再夹住，cnt==0/1 会出现 0/0=nan，
    那两条等价性证明立刻作废。

    写成 AST 全称量化而不是 `"np.maximum(cnt, 1)" in src`（§Y）：子串断言
    在同一串出现多次时杀不掉任何变异 —— 改坏第二处、第一处还在，照样绿。
    """
    tree = ast.parse(_src())
    divides = [n for n in ast.walk(tree)
               if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and n.func.attr == "divide"]
    assert divides, "factor_model 里已经没有 np.divide 了 —— 这两条证明该重写"
    for call in divides:
        assert any(kw.arg == "where" for kw in call.keywords), (
            f"L{call.lineno} 的 np.divide 没有 where= 守卫")
        denom = call.args[1] if len(call.args) >= 2 else None
        assert (isinstance(denom, ast.Call)
                and isinstance(denom.func, ast.Attribute)
                and denom.func.attr == "maximum"), (
            f"L{call.lineno} 的 np.divide 分母不是 np.maximum(…) —— "
            f"cnt==0/1 时会出现 0/0=nan，L79/L82 的等价性证明作废")


def test_a_single_valid_observation_yields_zero_exposure():
    """L82 证明的行为面：整行只有 1 个有效值时，暴露必须是 0（而不是 nan）。"""
    df = pd.DataFrame([[1.0, np.nan, np.nan]], columns=list("ABC"))
    out = fm._zscore_panel(df)                       # noqa: SLF001 —— 故意测内部不变量
    assert (out.to_numpy() == 0.0).all(), f"单有效值的行没有被清零：{out.to_numpy()}"
    assert np.isfinite(out.to_numpy()).all()


def test_an_all_nan_row_yields_zero_exposure():
    """同族边界：整行全 NaN 也必须是 0，且不产生任何 numpy 告警。"""
    df = pd.DataFrame([[np.nan, np.nan]], columns=list("AB"))
    with np.errstate(all="raise"):
        out = fm._zscore_panel(df)                   # noqa: SLF001
    assert (out.to_numpy() == 0.0).all()


def test_sector_flag_is_always_passed_explicitly():
    """L193/L230 证明：全库构造这两个 dataclass 时都显式传 sector_included。"""
    bad = []
    for p in sorted((_backend_root() / "app").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) in ("RiskAttribution", "RiskModel")):
                if not any(kw.arg == "sector_included" for kw in node.keywords):
                    bad.append(f"{p.name}:{node.lineno}")
    assert not bad, (
        f"以下构造点没有显式传 sector_included，dataclass 默认值因此可观测，"
        f"那条等价性证明不成立：{bad}")


def test_a_zero_lookback_means_the_whole_history():
    """L306 证明：`x[-0:]` 就是整段 —— 两条分支给出同一个列表。"""
    idx = pd.bdate_range("2021-01-04", periods=7)
    assert list(idx[-0:]) == list(idx), "Python 切片语义变了，L306 的等价性证明作废"


def test_every_survivor_has_a_written_disposition():
    """
    本模块新增点的存活处置对账：6 个已证明 + 1 个未处置 = 7 个存活。
    数字对不上就红 —— 复测后存活数变了却没更新这里，会在这条上顶出来。
    """
    def _points(d: dict) -> int:
        return sum(int(re.findall(r"×(\d+)", k)[0]) for k in d)

    assert _points(NEW_POINT_EQUIVALENCE) == 6, _points(NEW_POINT_EQUIVALENCE)
    assert _points(NEW_POINT_UNPROVEN) == 1, _points(NEW_POINT_UNPROVEN)

    all_entries = {**NEW_POINT_EQUIVALENCE, **NEW_POINT_UNPROVEN}
    known = set()
    for p in (_backend_root() / "tests").rglob("test_*.py"):
        if "__pycache__" in p.parts:
            continue
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                known.add(node.name)

    for key, why in all_entries.items():
        assert len(why) >= 40, f"{key} 的说明过于敷衍：{why!r}"
        assert re.match(r"^app/\S+\.py ×\d+ — \S", key), f"键格式不合规：{key!r}"
        # 只认显式引用（"见 xxx" / "由 xxx"）；宽正则会把散文里的参数名当成用例名
        for name in re.findall(r"(?:见|由)\s*(test_[a-z0-9_]+)", why):
            assert name in known, f"{key[:40]} 点名了不存在的用例 {name}"
