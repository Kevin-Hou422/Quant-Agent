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

#: 检查分两层（审计 F13：以前一个 ready 同时冒充"连得上"与"明天会交易"）：
#:   connect —— OpenD / 模拟账户 / 资金 / 熔断 / 对账：券商这一侧能不能用
#:   trade   —— 配置 / 调度 / 存储 / 日历 / 数据集 / 策略 / 数据源：明天收盘后会不会真的下单
LAYER_CONNECT, LAYER_TRADE = "connect", "trade"


@dataclass
class Check:
    name:     str
    ok:       bool
    detail:   str
    blocking: bool
    layer:    str


@dataclass
class PreflightReport:
    checked_at: str
    checks:     List[Check] = field(default_factory=list)
    gateway:    Optional[dict] = None
    account:    Optional[dict] = None
    reconcile:  Optional[dict] = None

    @property
    def connected(self) -> bool:
        """券商这一侧可用（不代表会交易）。"""
        return all(c.ok for c in self.checks if c.blocking and c.layer == LAYER_CONNECT)

    @property
    def ready(self) -> bool:
        """两层都过 = 可以交给调度器。"""
        return all(c.ok for c in self.checks if c.blocking)

    def add(self, name: str, ok: bool, detail: str, blocking: bool = True,
            layer: str = LAYER_TRADE) -> bool:
        self.checks.append(Check(name, bool(ok), detail, blocking, layer))
        return bool(ok)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ready"] = self.ready
        d["connected"] = self.connected
        return d


def _registered_jobs() -> List[str]:
    """按当前配置**实际构建**一次调度器（内存 jobstore，不启动），返回注册了哪些任务。"""
    from app.tasks.scheduler import create_scheduler
    return sorted(j.id for j in create_scheduler(db_url="sqlite://").get_jobs())


def _probe_data(settings, target) -> tuple:
    """
    向真实数据源取交易宇宙前 2 只、截至最近已收盘交易日的日线，确认**目标日那根 bar 拿得到**。
    这一项是 F01 那类问题（数据源区间语义 / 更新延迟）唯一能在上线前被看见的地方。
    """
    import dataclasses
    import pandas as pd
    from app.core.data_engine.dataset_registry import _fetch_raw, registry_spec
    spec = registry_spec(str(settings.paper_dataset))
    sub = dataclasses.replace(spec, universe=list(spec.universe[:2]))
    start = (pd.Timestamp(target) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    raw = _fetch_raw(sub, start, pd.Timestamp(target).strftime("%Y-%m-%d"))
    close = raw.get("close")
    if close is None or close.dropna(how="all").empty:
        return False, f"数据源对 {sub.universe} 返回空数据"
    last = pd.Timestamp(close.dropna(how="all").index.max()).normalize()
    if last < pd.Timestamp(target).normalize():
        return False, (f"数据源最新 bar {last.date()} 早于最近已收盘交易日 {pd.Timestamp(target).date()}"
                       f"（收盘后数据未更新，或区间语义不对）")
    return True, f"数据源已有 {pd.Timestamp(target).date()} 的 bar（抽查 {sub.universe}）"


def _active_strategy_problem(settings) -> tuple:
    """(ok, detail)：有一份 active 策略配置，且其成分都处于 PAPER/ACTIVE/DECAYING。"""
    import json
    from app.db.alpha_lifecycle import AlphaStatus, coerce_status
    from app.db.alpha_store import AlphaStore
    from app.db.strategy_store import StrategyStore
    active = StrategyStore(db_url=settings.database_url).latest_active()
    if active is None:
        return False, ("没有 active 策略配置 —— 券商执行只对账、不下单（基准库与未批准组合"
                       "只进模拟账本）。先提出并批准+激活一份策略配置")
    factors = [str(f) for f in json.loads(active.factors or "[]")]
    store = AlphaStore(db_url=settings.database_url)
    bad = []
    for f in factors:
        rec = store.get_by_id(int(f)) if f.isdigit() else None
        try:
            st = coerce_status(rec.status) if rec is not None else None
        except ValueError:
            logger.warning("[golive] 因子 %s 状态 %r 无法解析 → 按不可交易处理", f, rec.status)
            st = None
        if st not in (AlphaStatus.PAPER, AlphaStatus.ACTIVE, AlphaStatus.DECAYING):
            bad.append(f)
    if not factors or bad:
        return False, f"active 策略 #{active.id} 的成分 {factors} 中 {bad} 不在可交易状态"
    return True, f"active 策略 #{active.id}（{len(factors)} 个成分，method={active.method}）"


def run_preflight(
    settings=None,
    *,
    gateway_factory: Optional[Callable] = None,
    probe: Optional[Callable[[str, int], bool]] = None,
    store=None,
    position_store=None,
    now: Optional[datetime] = None,
    cwd: Optional[Path] = None,
    data_probe: Optional[Callable] = None,
    jobs: Optional[Callable[[], List[str]]] = None,
    strategy_check: Optional[Callable] = None,
) -> PreflightReport:
    """
    逐项检查，两层（见 LAYER_*）。连券商的检查会**真的对账一次**（与 `/execution/reconcile`
    同一条路径：补记成交、首次运行时建立可信基准快照），不会下单。数据源检查会真的取数。
    OpenD 不在时券商侧检查记为失败并跳过，其余检查照常做完 —— 一次看全所有问题。
    """
    from app.core.execution.broker_gateway import make_gateway_from_settings, opend_reachable
    if settings is None:
        from app.config import settings as _s
        settings = _s
    now = now or datetime.now(timezone.utc)
    probe = probe or opend_reachable
    rep = PreflightReport(checked_at=now.isoformat(timespec="seconds"))

    # ---- 交易层：配置 / 调度 / 存储 / 日历 / 数据集 / 策略 ----
    mode = str(settings.execution_mode)
    rep.add("execution_mode", mode == "moomoo_paper",
            f"EXECUTION_MODE={mode}" + ("" if mode == "moomoo_paper"
                                        else "（需设为 moomoo_paper 才会下单）"))
    sched_ok = bool(settings.enable_scheduler) and bool(settings.enable_paper_trading)
    rep.add("scheduler", sched_ok,
            f"ENABLE_SCHEDULER={bool(settings.enable_scheduler)} "
            f"ENABLE_PAPER_TRADING={bool(settings.enable_paper_trading)}"
            + ("" if sched_ok else "（两者都为 true，每日管线才会自动跑）"))
    try:
        ids = (jobs or _registered_jobs)()
        need = {"daily_trading", "daily_trading_retry"}
        rep.add("scheduler_jobs", need <= set(ids),
                f"按当前配置构建的调度器任务：{ids}" + ("" if need <= set(ids)
                                                   else f"（缺 {sorted(need - set(ids))}）"))
    except Exception as exc:
        logger.warning("[golive] 构建调度器失败: %s", exc)
        rep.add("scheduler_jobs", False, f"按当前配置构建调度器失败：{exc}")
    problems = live_storage_problems(settings, cwd)
    rep.add("storage", not problems,
            "活库 / 调度库 / PIT 均不在云同步目录" if not problems else "；".join(problems))
    unwritable = storage_not_writable(settings, cwd)
    rep.add("storage_writable", not unwritable,
            "活库目录 / PIT 目录可写" if not unwritable else "；".join(unwritable))

    target = None
    try:
        from app.core.data_engine.market_calendar import last_closed_session
        target = last_closed_session(now)
        rep.add("calendar", True, f"最近一个已收盘交易日 {target.date()}")
    except Exception as exc:
        logger.warning("[golive] 交易日历不可用: %s", exc)
        rep.add("calendar", False, f"交易日历不可用（调度器会 fail-closed 不交易）：{exc}")

    try:
        from app.core.data_engine.dataset_registry import registry_spec
        spec = registry_spec(str(settings.paper_dataset))
        rep.add("dataset", spec.region == "US",
                f"PAPER_DATASET={spec.name}（{spec.region}，{len(spec.universe)} 只）"
                + ("" if spec.region == "US" else " —— 执行层只交易美股"))
    except Exception as exc:
        logger.warning("[golive] 数据集无效: %s", exc)
        rep.add("dataset", False, f"PAPER_DATASET={settings.paper_dataset} 无效：{exc}")
    src = str(getattr(settings, "price_source", "yahoo"))
    rep.add("price_source", src == "moomoo",
            f"PRICE_SOURCE={src}" + ("（研究与执行同源，TR.2）" if src == "moomoo" else
                                     " —— 研究用 yahoo、执行用 moomoo，不同源（TR.2 的决策是 moomoo）"),
            blocking=False)
    try:
        ok, detail = (strategy_check or _active_strategy_problem)(settings)
    except Exception as exc:
        logger.warning("[golive] 读取策略配置失败: %s", exc)
        ok, detail = False, f"读取策略配置失败：{exc}"
    rep.add("strategy", ok, detail)

    # ---- 券商层：OpenD / 账户 / 资金 / 熔断 / 对账 ----
    host, port = str(settings.moomoo_host), int(settings.moomoo_port)
    reachable = bool(probe(host, port))
    rep.add("opend", reachable, f"OpenD {host}:{port} 在监听" if reachable else
            f"OpenD {host}:{port} 未在监听 —— 启动 moomoo OpenD 并登录", layer=LAYER_CONNECT)
    if reachable:
        try:
            gw = (gateway_factory or make_gateway_from_settings)()
        except Exception as exc:
            logger.warning("[golive] 连不上纸交易账户: %s", exc)
            rep.add("broker_account", False, f"连不上纸交易账户：{exc}", layer=LAYER_CONNECT)
            gw = None
        if gw is not None:
            try:
                rep.gateway = gw.describe()
                rep.add("broker_account", True, f"模拟账户 acc_id={rep.gateway.get('acc_id')}",
                        layer=LAYER_CONNECT)
                _check_funds(rep, gw, settings)
                _check_kill_switch_and_reconcile(rep, gw, settings, store, position_store)
            finally:
                gw.close()

    # ---- 数据源：真的取一次（moomoo 行情需要 OpenD）----
    if target is None:
        rep.add("data_source", False, "交易日历不可用，无法确定目标日")
    elif src == "moomoo" and not reachable:
        rep.add("data_source", False, "PRICE_SOURCE=moomoo 但 OpenD 不在 —— 拿不到行情")
    else:
        try:
            ok, detail = (data_probe or _probe_data)(settings, target)
        except Exception as exc:
            logger.warning("[golive] 数据源探测失败: %s", exc)
            ok, detail = False, f"数据源取数失败：{exc}"
        rep.add("data_source", ok, detail)
    return rep


def storage_not_writable(settings, cwd: Optional[Path] = None) -> List[str]:
    """活库所在目录与 PIT 目录能否创建并写入（启动后才发现写不了，当天的前向数据就丢了）。"""
    import tempfile
    cwd = Path(cwd or os.getcwd())
    dirs = []
    p = sqlite_path(str(settings.database_url), cwd)
    if p is not None:
        dirs.append(("database_url", p.parent))
    dirs.append(("pit_store_dir", _absolute(str(settings.pit_store_dir), cwd)))
    out = []
    for name, d in dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=d, prefix=".preflight_", delete=True):
                pass
        except OSError as exc:
            logger.warning("[golive] %s 目录不可写 %s: %s", name, d, exc)
            out.append(f"{name} 目录 {d} 不可写：{exc}")
    return out


def _check_funds(rep: PreflightReport, gw, settings) -> None:
    try:
        acct = gw.account()
    except Exception as exc:
        logger.warning("[golive] 读不到账户资金: %s", exc)
        rep.add("account_funds", False, f"读不到账户资金：{exc}", layer=LAYER_CONNECT)
        return
    rep.account = asdict(acct)
    aum = float(settings.paper_aum)
    tol = float(settings.exec_max_aum_mismatch)
    if not acct.total_assets > 0:
        rep.add("account_funds", False, f"账户总资产 {acct.total_assets} ≤ 0", layer=LAYER_CONNECT)
        return
    gap = abs(acct.total_assets / aum - 1.0)
    rep.add("account_funds", gap <= tol,
            f"账户总资产 ${acct.total_assets:,.2f} vs PAPER_AUM ${aum:,.2f}（偏离 {gap:.1%}，"
            f"上限 {tol:.0%}）" + ("" if gap <= tol else
                                 f" —— 把 PAPER_AUM 设为 {acct.total_assets:.0f}"), layer=LAYER_CONNECT)


def _check_kill_switch_and_reconcile(rep: PreflightReport, gw, settings, store,
                                     position_store) -> None:
    from app.core.execution.order_manager import OrderManager
    from app.core.execution.pretrade_gate import PreTradeLimits
    from app.db.execution_store import ExecutionStore
    store = store or ExecutionStore(db_url=str(settings.database_url))
    ks = store.kill_switch()
    rep.add("kill_switch", not ks.get("engaged"),
            "未开启" if not ks.get("engaged") else
            f"已开启（{ks.get('actor')}：{ks.get('reason')}）—— 解除前只会继续全平", layer=LAYER_CONNECT)
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
        rep.add("reconcile", False, f"对账失败：{exc}", layer=LAYER_CONNECT)
        return
    rep.reconcile = rec.to_dict()
    rep.add("reconcile", not rec.blocking,
            f"对账状态 {rec.status}" + ("" if not rec.blocking else
                                       " —— 在「执行监控」面板核对后接受，或排查差异"), layer=LAYER_CONNECT)
