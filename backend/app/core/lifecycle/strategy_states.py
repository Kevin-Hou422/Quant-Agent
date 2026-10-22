"""
strategy_states.py — 策略配置的运行状态与权限（2026-10-22 用户核验 C6）

用户原话："分清'开始收集证据'和'已通过验证'。当前 tr_enforce_active_gate=False 时，证据不足也能激活；
打开强制门后，新提案又需要先积累前向。请明确工程联调、PAPER_RUNNING、VERIFIED_ACTIVE 的状态和权限。
不能通过关闭门、借旧策略 IC 或把 active 字样当作验证通过来解决自举问题。"

自举问题的解法：**开始收集证据不需要证据，被称为"已验证"必须有证据**。两件事是两个状态：

    proposed → approved ─┬→ integration（工程联调）─→ paper_running
                         └→ paper_running（纸交易、收集前向证据）─→ verified_active（过了预登记的验证设计）
    任何运行状态 → retired；proposed → rejected；verified_active → paper_running（降级，重新收证据）

| 状态             | 在模拟账户交易 | 前向证据计入 | 可成为正式前向天数 | 算"已验证" |
|------------------|----------------|--------------|--------------------|------------|
| integration      | 是（人工监督） | 否           | 否                 | 否         |
| paper_running    | 是             | 是           | 是（人启动后）     | 否         |
| verified_active  | 是             | 是           | 是                 | 是         |
| active（旧行）   | 否（没有规格）  | 否           | 否                 | 否         |

· paper_running → verified_active 只能经验证门（预登记设计 + 这份规格自己登记之后的 forward 证据），**没有开关可以绕过**；
· 真钱交易在代码层禁止（与本表无关，任何状态都不解锁）；
· `active` 是 2026-10-22 之前的旧状态字样，读到时一律当作**未验证、不可交易**（它们本来就没有可执行规格，须重新提案）。
"""

from __future__ import annotations

from typing import Dict

RUNNING_STATES = ("integration", "paper_running", "verified_active")
LEGACY_ACTIVE = "active"

_PERMISSIONS: Dict[str, Dict[str, bool]] = {
    "integration":     {"trades_simulate": True,  "forward_evidence": False, "formal_days": False, "verified": False},
    "paper_running":   {"trades_simulate": True,  "forward_evidence": True,  "formal_days": True,  "verified": False},
    "verified_active": {"trades_simulate": True,  "forward_evidence": True,  "formal_days": True,  "verified": True},
}
_NONE = {"trades_simulate": False, "forward_evidence": False, "formal_days": False, "verified": False}

LABELS = {
    "proposed": "已提案", "approved": "已批准（未运行）", "integration": "工程联调（不计证据）",
    "paper_running": "纸交易·收集证据（未验证）", "verified_active": "已验证", "retired": "已退役",
    "rejected": "已拒绝", LEGACY_ACTIVE: "旧版 active（未验证、无规格，须重新提案）",
}


def permissions(status: str) -> Dict[str, bool]:
    """某状态的权限（未知 / 旧 active / 非运行状态 → 全否）。"""
    return dict(_PERMISSIONS.get(str(status), _NONE))


def label(status: str) -> str:
    return LABELS.get(str(status), str(status))
