"""
golive.py — 前向交易上线预检（Phase 12 收尾）

回答一个问题：**此刻把系统交给调度器，明天收盘后它能不能真的在 moomoo 纸交易账户上
下单、成交、对账？** 每一项都是一个独立检查，失败的给出原因与修法；任何一项阻塞性
检查不过 → `ready=False`，`python -m app.tasks.forward run-now` 拒绝执行。

另有一条**启动保险**（`assert_live_storage_safe`）：前向交易开启时，活库 / 调度库 /
PIT 目录若**解析后**落在云同步目录里就拒绝启动。只看配置字符串是不够的 —— 默认值
`sqlite:///./alphas.db` 本身不含 "onedrive"，但从仓库目录启动时它解析到的正是 OneDrive。
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

#: 云同步客户端的目录特征（小写比较）。WAL 模式的 SQLite 三件套被分别异步上传会撕裂。
CLOUD_SYNC_MARKERS = ("onedrive", "dropbox", "google drive", "googledrive", "icloud")

_SQLITE_PREFIX = "sqlite:///"


# ---------------------------------------------------------------------------
# 存储位置
# ---------------------------------------------------------------------------

def sqlite_path(url: str, cwd: Path) -> Optional[Path]:
    """SQLite URL 解析成绝对路径；非 SQLite / 内存库返回 None。相对路径按 cwd 解析（与 SQLAlchemy 一致）。"""
    if not str(url).startswith(_SQLITE_PREFIX):
        return None
    raw = str(url)[len(_SQLITE_PREFIX):]
    if raw in ("", ":memory:"):
        return None
    return _absolute(raw, cwd)


def _absolute(raw: str, cwd: Path) -> Path:
    p = Path(raw)
    return (p if p.is_absolute() else Path(cwd) / p).resolve()


def in_cloud_sync(path: Path) -> bool:
    low = str(path).replace("\\", "/").lower()
    return any(m in low for m in CLOUD_SYNC_MARKERS)


def live_storage_problems(settings, cwd: Optional[Path] = None) -> List[str]:
    """返回落在云同步目录里的活数据位置（空列表 = 安全）。"""
    cwd = Path(cwd or os.getcwd())
    out: List[str] = []
    for attr in ("database_url", "scheduler_db_url"):
        p = sqlite_path(str(getattr(settings, attr)), cwd)
        if p is not None and in_cloud_sync(p):
            out.append(f"{attr} → {p}")
    pit = _absolute(str(settings.pit_store_dir), cwd)
    if in_cloud_sync(pit):
        out.append(f"pit_store_dir → {pit}")
    return out


def forward_trading_enabled(settings) -> bool:
    """会产生不可再生的前向数据（真实下单 / 前向 PIT / 每日账本）的配置。"""
    return str(settings.execution_mode) != "off" or bool(settings.enable_paper_trading)


def assert_live_storage_safe(settings, cwd: Optional[Path] = None) -> None:
    """前向交易开启且活数据落在云同步目录 → 拒绝启动。研究/开发模式不拦。"""
    if not forward_trading_enabled(settings):
        return
    problems = live_storage_problems(settings, cwd)
    if problems:
        raise RuntimeError(
            "拒绝启动前向交易：以下活数据位于云同步目录（同步进程会撕裂 SQLite WAL 三件套，"
            "前向数据丢了买不回来）：" + "；".join(problems) +
            "。请在 .env 里把 DATABASE_URL / SCHEDULER_DB_URL / PIT_STORE_DIR 设为本地非同步盘的"
            "绝对路径（见 backend/.env.forward.example）。")


# ---------------------------------------------------------------------------
# 预检
# ---------------------------------------------------------------------------

@dataclass
class Check:
    name:     str
    ok:       bool
    detail:   str
    blocking: bool


@dataclass
class PreflightReport:
    checked_at: str
    checks:     List[Check] = field(default_factory=list)
    gateway:    Optional[dict] = None
    account:    Optional[dict] = None
    reconcile:  Optional[dict] = None

    @property
    def ready(self) -> bool:
        return all(c.ok for c in self.checks if c.blocking)

    def add(self, name: str, ok: bool, detail: str, blocking: bool = True) -> bool:
        self.checks.append(Check(name, bool(ok), detail, blocking))
        return bool(ok)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ready"] = self.ready
        return d


def run_preflight(
    settings=None,
    *,
    gateway_factory: Optional[Callable] = None,
    probe: Optional[Callable[[str, int], bool]] = None,
    store=None,
    position_store=None,
    now: Optional[datetime] = None,
    cwd: Optional[Path] = None,
) -> PreflightReport:
    """
    逐项检查。连券商的检查会**真的对账一次**（与 `/execution/reconcile` 同一条路径：
    补记成交、首次运行时建立可信基准快照），不会下单。
    """
    from app.core.execution.broker_gateway import make_gateway_from_settings, opend_reachable
    if settings is None:
        from app.config import settings as _s
        settings = _s
    now = now or datetime.now(timezone.utc)
    probe = probe or opend_reachable
    rep = PreflightReport(checked_at=now.isoformat(timespec="seconds"))

    mode = str(settings.execution_mode)
    rep.add("execution_mode", mode == "moomoo_paper",
            f"EXECUTION_MODE={mode}" + ("" if mode == "moomoo_paper"
                                        else "（需设为 moomoo_paper 才会下单）"))
    sched_ok = bool(settings.enable_scheduler) and bool(settings.enable_paper_trading)
    rep.add("scheduler", sched_ok,
            f"ENABLE_SCHEDULER={bool(settings.enable_scheduler)} "
            f"ENABLE_PAPER_TRADING={bool(settings.enable_paper_trading)}"
            + ("" if sched_ok else "（两者都为 true，每日管线才会自动跑）"))
    problems = live_storage_problems(settings, cwd)
    rep.add("storage", not problems,
            "活库 / 调度库 / PIT 均不在云同步目录" if not problems else "；".join(problems))

    try:
        from app.core.execution.order_manager import last_closed_session
        rep.add("calendar", True, f"最近一个已收盘交易日 {last_closed_session(now)}")
    except Exception as exc:
        logger.warning("[golive] 交易日历不可用: %s", exc)
        rep.add("calendar", False, f"交易日历不可用（调度器会 fail-closed 不交易）：{exc}")

    host, port = str(settings.moomoo_host), int(settings.moomoo_port)
    reachable = bool(probe(host, port))
    rep.add("opend", reachable, f"OpenD {host}:{port} 在监听" if reachable else
            f"OpenD {host}:{port} 未在监听 —— 启动 moomoo OpenD 并登录")
    if not reachable:
        return rep

    try:
        gw = (gateway_factory or make_gateway_from_settings)()
    except Exception as exc:
        logger.warning("[golive] 连不上纸交易账户: %s", exc)
        rep.add("broker_account", False, f"连不上纸交易账户：{exc}")
        return rep
    try:
        rep.gateway = gw.describe()
        rep.add("broker_account", True, f"模拟账户 acc_id={rep.gateway.get('acc_id')}")
        _check_funds(rep, gw, settings)
        _check_kill_switch_and_reconcile(rep, gw, settings, store, position_store)
    finally:
        gw.close()
    return rep


def _check_funds(rep: PreflightReport, gw, settings) -> None:
    try:
        acct = gw.account()
    except Exception as exc:
        logger.warning("[golive] 读不到账户资金: %s", exc)
        rep.add("account_funds", False, f"读不到账户资金：{exc}")
        return
    rep.account = asdict(acct)
    aum = float(settings.paper_aum)
    tol = float(settings.exec_max_aum_mismatch)
    if not acct.total_assets > 0:
        rep.add("account_funds", False, f"账户总资产 {acct.total_assets} ≤ 0")
        return
    gap = abs(acct.total_assets / aum - 1.0)
    rep.add("account_funds", gap <= tol,
            f"账户总资产 ${acct.total_assets:,.2f} vs PAPER_AUM ${aum:,.2f}（偏离 {gap:.1%}，"
            f"上限 {tol:.0%}）" + ("" if gap <= tol else
                                 f" —— 把 PAPER_AUM 设为 {acct.total_assets:.0f}"))


def _check_kill_switch_and_reconcile(rep: PreflightReport, gw, settings, store,
                                     position_store) -> None:
    from app.core.execution.order_manager import OrderManager
    from app.core.execution.pretrade_gate import PreTradeLimits
    from app.db.execution_store import ExecutionStore
    store = store or ExecutionStore()
    ks = store.kill_switch()
    rep.add("kill_switch", not ks.get("engaged"),
            "未开启" if not ks.get("engaged") else
            f"已开启（{ks.get('actor')}：{ks.get('reason')}）—— 解除前只会继续全平")
    try:
        rec = OrderManager(
            gw, store, position_store,
            limits=PreTradeLimits.from_settings(settings),
            paper_aum=float(settings.paper_aum),
            limit_band_bps=float(settings.exec_limit_band_bps),
            flatten_band_bps=float(settings.exec_flatten_band_bps),
            max_aum_mismatch=float(settings.exec_max_aum_mismatch),
        ).reconcile()
    except Exception as exc:
        logger.warning("[golive] 对账失败: %s", exc)
        rep.add("reconcile", False, f"对账失败：{exc}")
        return
    rep.reconcile = rec.to_dict()
    rep.add("reconcile", not rec.blocking,
            f"对账状态 {rec.status}" + ("" if not rec.blocking else
                                       " —— 在「执行监控」面板核对后接受，或排查差异"))
