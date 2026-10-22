"""
daily_ingest.py — 每日数据摄取 + 健康验收门（Task 7.1，2026-08-02）

职责：收盘后拉取数据 → **健康验收门** → 通过则交给每日交易循环；不通过则
**拒绝当日调仓并告警，绝不静默降级到坏/过期数据**（验收标准 A2）。

设计：
  - 复用 `load_registry_dataset`（真实数据路径）+ `check_dataset_health`（gap/spike/
    NaN → 综合健康分）。
  - 健康门：health_score < min_score → `IngestResult(accepted=False, ...)` + WARNING 告警，
    调用方据此**跳过当日循环**。绝不因数据坏而退合成数据。
  - as_of 时间戳：记录本次摄取的观测时点（PIT 语义的前置；完整 PIT 追加存储见 Phase 8.1）。

调度：挂到 Phase 5.3 调度器（收盘后触发），成功后调用 `DailyTradingLoop.run`。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    accepted:     bool
    dataset_name: str
    as_of:        str
    health_score: float = 0.0
    n_dates:      int = 0
    n_tickers:    int = 0
    reject_reason: str = ""
    dataset:      Optional[Dict[str, pd.DataFrame]] = None
    # ── Phase 11 增量摄取 ────────────────────────────────────────────────
    mode:          str = "full"        # full(首次回填) | incremental(前向增量) | no_new_bar
    forward_from:  Optional[str] = None   # 本次新增的第一根 bar 日期 → 之后的 IC 算"真前向"
    n_new_bars:    int = 0
    calendar_note: str = ""            # 数据↔日历 交叉校验说明
    #: 交给交易循环的那份面板的输入身份（复审 2026-10-19 N02-c / N04-b；见 input_identity.py），
    #: 外加 checked_last_day / checked_universe_digest = 健康门检查的那份的末日与股票池
    identity:      Optional[Dict[str, Any]] = None


class DailyIngest:
    def __init__(self, min_health: float = 0.7) -> None:
        self.min_health = min_health

    # ------------------------------------------------------------------
    # Phase 11.1 / 11.3：真增量摄取（不再整段重拉）
    # ------------------------------------------------------------------

    def _rebuild_for_source(self, store, dataset_name: str, end: str, as_of: str,
                            target: pd.Timestamp, reason: str) -> "IngestResult":
        """整段按新价格源重取，作为新 vintage 追加（旧 vintage 保留），并更新 PIT 元数据。"""
        from app.config import settings
        logger.warning("[daily_ingest] %s → 整段重取历史，按新源作为新 vintage 追加", reason)
        start_full = store.earliest_timestamp(dataset_name) or pd.Timestamp(settings.paper_start)
        full = self.ingest(dataset_name, start_full.strftime("%Y-%m-%d"), end,
                           append_pit=False, as_of=as_of, upto=target)
        if not full.accepted:
            return IngestResult(False, dataset_name, as_of, mode="rebased",
                                reject_reason=f"rebase_failed: {full.reject_reason}")
        try:
            self._append_pit(dataset_name, {f: df for f, df in (full.dataset or {}).items()
                                            if isinstance(df, pd.DataFrame)}, as_of)
            meta = store.read_meta(dataset_name)
            meta.update(price_source=str(getattr(settings, "price_source", "yahoo")),
                        last_rebase=as_of, last_rebase_reason=reason)
            store.write_meta(meta, name=dataset_name)
        except Exception as exc:
            logger.error("[daily_ingest] 换源重建写 PIT 失败: %s", exc)
            return IngestResult(False, dataset_name, as_of, mode="rebased",
                                reject_reason=f"pit_append_failed: {exc}")
        full.mode, full.forward_from, full.n_new_bars = "rebased", None, 0
        full.calendar_note = reason
        return full

    def ingest_incremental(self, dataset_name: str,
                           backfill_start: Optional[str] = None,
                           now: Optional[datetime] = None) -> IngestResult:
        """
        **前向增量**摄取（Phase 11 核心）：
          1. 从 PIT 读该数据集**已有的最新 bar 日期**；
          2. 目标日 = **最近一个已收盘的交易日**（交易所日历 + 收盘时刻，与执行层同一定义）；
             晚于目标日的 bar（盘中未收盘的半根）一律丢弃；
          3. 空库 → 一次性**历史回填**到目标日（mode=full，这些 IC 是"回放"，非前向）；
             非空 → 拉 `last .. 目标日` 的**增量**（mode=incremental），**故意重叠 last 那一根**：
             provider 在窗口内用 close.shift(1) 派生 returns，不重叠的话每根新 bar 的
             returns 都是 NaN（外部审计 F11）；
          4. 无新 bar → `no_new_bar`，不写库（周末/节假日/尚未收盘的自然结果）；
          5. 增量过健康门后**只追加 > last 的部分**进 PIT；
          6. 与交易日历**交叉校验**并记录说明（数据仍是权威，只是让不一致被看见）。

        返回的 `dataset` 是**从 PIT 读出的完整面板**（回填+历次增量），供交易循环消费；
        `forward_from` 是本次新增的第一根 bar 日期，交易循环据此把之后的 IC 标为真前向。
        """
        from app.config import settings
        from app.core.data_engine.pit_store import PITStore
        from app.core.data_engine.market_calendar import cross_check, last_closed_session

        now = now or datetime.now(timezone.utc)
        as_of = now.isoformat(timespec="seconds")
        store = PITStore(settings.pit_store_dir)
        last = store.latest_timestamp(dataset_name)
        target = last_closed_session(now)          # 此刻应已有日 bar 的最近交易日
        end = target.strftime("%Y-%m-%d")

        # ---- 1) 空库 → 历史回填（回放种子） ----
        if last is None:
            start = backfill_start or settings.paper_start
            res = self.ingest(dataset_name, start, end, as_of=as_of, upto=target,
                              pit_required=True)
            if res.accepted:
                store.write_meta({"price_source": str(getattr(settings, "price_source", "yahoo"))},
                                 name=dataset_name)
            res.mode = "full"
            res.forward_from = None               # 回填全部算回放，不是前向证据
            res.n_new_bars = res.n_dates
            logger.info("[daily_ingest] PIT 为空 → 历史回填 %s..%s（%d 日，计为回放）",
                        start, end, res.n_dates)
            return res

        # ---- 2) 已有数据 → 只拉增量 ----
        if last.normalize() >= target.normalize():
            # 复审 2026-10-19 N02：PIT 已是最新、当天换了价格源 → 此前直接"无新 bar"返回，
            # 交易循环接着用旧源的 PIT。换源检查必须在消费 PIT 之前做，不只在有新 bar 时做。
            old_src = (store.read_meta(dataset_name) or {}).get("price_source")
            src_now = str(getattr(settings, "price_source", "yahoo"))
            if old_src and old_src != src_now:
                return self._rebuild_for_source(store, dataset_name, end, as_of, target,
                                                f"价格源从 {old_src} 换成了 {src_now}（无新 bar）")
            note = f"PIT 最新 bar {last.date()} 已是最近已收盘交易日 {target.date()}"
            logger.info("[daily_ingest] 无新 bar（%s）→ 跳过", note)
            return IngestResult(False, dataset_name, as_of, mode="no_new_bar",
                                reject_reason="no_new_bar", calendar_note=note)

        # `append_pit=False`：本方法在第 3 步自己按 `> last` 过滤后写 PIT（缺陷 A-1）。
        inc = self.ingest(dataset_name, last.strftime("%Y-%m-%d"), end,
                          append_pit=False, as_of=as_of, upto=target)
        if not inc.accepted:
            inc.mode = "incremental"
            return inc

        inc_close = (inc.dataset or {}).get("close")
        new_dates = [d for d in (inc_close.index if inc_close is not None else [])
                     if pd.Timestamp(d).normalize() > last.normalize()]
        ok, note = cross_check(target, has_bar=bool(new_dates))
        if not new_dates:
            logger.info("[daily_ingest] 增量窗口无新 bar → 跳过（%s）", note)
            return IngestResult(False, dataset_name, as_of, mode="no_new_bar",
                                reject_reason="no_new_bar", calendar_note=note)

        # ---- 3) 只把**增量**写进 PIT（而非整段重写） ----
        #     例外：数据源重算了历史复权价（分红 / 拆股），或者价格源换了 —— 这时新 bar 与库里
        #     的旧 bar 不是同一个尺度，直接拼接会在接缝处造出一个假收益（外部审查 2026-10-18 F02：
        #     旧 100、新 50，拼出 −50%，而数据源自己算的收益是 0）。改为整段重取，作为新 vintage 追加：
        #     旧 vintage 原样保留（双时点历史不丢），最新视图是同一个尺度。
        first_new = pd.Timestamp(min(new_dates)).normalize()
        source = str(getattr(settings, "price_source", "yahoo"))
        rebase_reason = _rebase_reason(store, dataset_name, last, inc.dataset or {}, source)
        mode = "incremental"
        if rebase_reason:
            logger.warning("[daily_ingest] %s → 整段重取历史，按新尺度作为新 vintage 追加", rebase_reason)
            start_full = store.earliest_timestamp(dataset_name) or pd.Timestamp(settings.paper_start)
            full_res = self.ingest(dataset_name, start_full.strftime("%Y-%m-%d"), end,
                                   append_pit=False, as_of=as_of, upto=target)
            if not full_res.accepted:
                return IngestResult(False, dataset_name, as_of, mode="incremental",
                                    reject_reason=f"rebase_failed: {full_res.reject_reason}")
            to_write = {f: df for f, df in (full_res.dataset or {}).items() if isinstance(df, pd.DataFrame)}
            mode = "incremental_rebased"
        else:
            to_write = {f: df.loc[df.index > last] for f, df in (inc.dataset or {}).items()
                        if isinstance(df, pd.DataFrame)}
        try:
            self._append_pit(dataset_name, to_write, as_of)
        except Exception as exc:
            logger.error("[daily_ingest] 增量写 PIT 失败: %s", exc)
            return IngestResult(False, dataset_name, as_of, mode="incremental",
                                reject_reason=f"pit_append_failed: {exc}")
        try:
            meta = store.read_meta(dataset_name)
            meta["price_source"] = source
            if rebase_reason:
                meta.update(last_rebase=as_of, last_rebase_reason=rebase_reason)
            store.write_meta(meta, name=dataset_name)
        except Exception as exc:
            # 数据已经落盘；元数据写不进去只影响下次的"换源"判断 —— 记错误，不拒绝今天的摄取
            logger.error("[daily_ingest] PIT 元数据写入失败（下次换源检查可能失灵）: %s", exc)

        # ---- 4) 交易循环消费"回填+全部增量"的完整面板（从 PIT 读） ----
        panel = store.load_pit(name=dataset_name) or {}
        full = panel if panel.get("close") is not None else (inc.dataset or {})
        # 复审 2026-10-19 N04-b：交出去的面板必须和健康门检查的那份**同一个末日、同一个股票池**
        # （PIT 读回来少了最后一天 / 少了票，交易就会用一份没被检查过的面板）
        ident = _identity_of(full, inc.dataset or {}, as_of)
        if (ident is None or ident["last_day"] != ident["checked_last_day"]
                or ident["universe_digest"] != ident["checked_universe_digest"]):
            why = ("算不出输入身份" if ident is None else
                   f"交易面板末日 {ident['last_day']} / 股票池 {ident['universe_n']} 只，"
                   f"健康门检查的是 {ident['checked_last_day']} / {ident['checked_universe_n']} 只")
            logger.error("[daily_ingest] %s → 拒绝", why)
            return IngestResult(False, dataset_name, as_of, mode=mode,
                                reject_reason=f"consumed_panel_mismatch: {why}")
        close = full.get("close")
        logger.info("[daily_ingest] 增量摄取：新增 %d 根 bar（%s..%s），PIT 面板 %s",
                    len(new_dates), first_new.date(), max(new_dates).date(),
                    None if close is None else close.shape)
        return IngestResult(
            True, dataset_name, as_of, health_score=inc.health_score,
            n_dates=0 if close is None else len(close),
            n_tickers=0 if close is None else close.shape[1],
            dataset=full, mode=mode,
            forward_from=first_new.strftime("%Y-%m-%d"),
            n_new_bars=len(new_dates), calendar_note=note, identity=ident,
        )

    def ingest(
        self,
        dataset_name: str,
        start:        str,
        end:          str,
        append_pit:   bool = True,
        as_of:        Optional[str] = None,
        upto:         Optional[pd.Timestamp] = None,
        pit_required: bool = False,
    ) -> IngestResult:
        """
        拉取 + 健康验收。返回 IngestResult；accepted=False 时调用方应跳过当日循环并告警。
        任何加载/校验异常都视为**拒绝**（不静默降级）。

        upto         : 丢弃晚于该日的 bar（尚未收盘的盘中半根不得进入 PIT / 决策）。
        pit_required : PIT 写入失败时**拒绝**（首次回填：没落盘的数据不能成为前向记录的起点）。
        """
        as_of = as_of or datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            from app.core.data_engine.dataset_registry import (
                load_registry_dataset, check_dataset_health,
            )
            # 基本面字段是 (事实, 价格) 的派生量，**不进价格 PIT**（Phase 10）：
            # 这里只取价格，喂给交易循环前由 _with_fundamentals 现挂。
            ds = load_registry_dataset(dataset_name, start=start, end=end, health_check=False,
                                       with_fundamentals=False)
        except Exception as exc:
            logger.error("[daily_ingest] 数据加载失败 '%s': %s — 拒绝当日摄取", dataset_name, exc)
            return IngestResult(False, dataset_name, as_of, reject_reason=f"load_failed: {exc}")

        data = ds.data
        if upto is not None and data.get("close") is not None:     # 没有 close 的面板下面会被拒
            cut = pd.Timestamp(upto).normalize()
            late = [d for d in data["close"].index if pd.Timestamp(d).normalize() > cut]
            if late:
                logger.warning("[daily_ingest] 丢弃 %d 根晚于最近已收盘交易日 %s 的 bar（未收盘）: %s",
                               len(late), cut.date(), [str(pd.Timestamp(d).date()) for d in late])
            data = {f: (df.loc[[pd.Timestamp(i).normalize() <= cut for i in df.index]]
                        if isinstance(df, pd.DataFrame) else df) for f, df in data.items()}
        close = data.get("close")
        if close is None or close.empty:
            return IngestResult(False, dataset_name, as_of, reject_reason="empty_close")

        # 健康门 —— 检查的必须是**截断之后**、真正会被落盘和使用的那份面板（复审 2026-10-19 N04）。
        # 此前检查的是截断前的 `ds`：一根完整的未收盘 bar 能把前一个已收盘日只有 10% 覆盖的问题盖住。
        import copy
        checked = copy.copy(ds)
        checked.data = data
        try:
            report = check_dataset_health(checked, min_score=self.min_health, warn_only=True)
        except Exception as exc:
            logger.warning("[daily_ingest] 健康检查异常，保守拒绝: %s", exc)
            return IngestResult(False, dataset_name, as_of, reject_reason=f"health_error: {exc}")
        if report is None:
            # 检查器内部出错时返回 None。此前这里按满分 1.0 放行 —— 检查器坏了，门就开着
            logger.warning("[daily_ingest] 健康检查没有给出结果 → 保守拒绝当日摄取")
            return IngestResult(False, dataset_name, as_of, reject_reason="health_unavailable")
        score = float(report.overall_score)
        held_gap = _held_without_close(data)
        if held_gap:
            # 持仓标的的报价不能被宇宙的平均覆盖率抵消：缺一只也可能让那只票被当成 0 或被平掉
            logger.warning("[daily_ingest] %s → 拒绝当日摄取", held_gap)
            return IngestResult(False, dataset_name, as_of, health_score=score,
                                n_dates=close.shape[0], n_tickers=close.shape[1],
                                reject_reason=f"health_hard_failure: {held_gap}")
        if report.hard_failures:
            logger.warning("[daily_ingest] 数据硬性条件未满足 → **拒绝当日摄取**: %s", report.hard_failures)
            return IngestResult(
                False, dataset_name, as_of, health_score=score,
                n_dates=close.shape[0], n_tickers=close.shape[1],
                reject_reason="health_hard_failure: " + "；".join(report.hard_failures))

        if score < self.min_health:
            logger.warning(
                "[daily_ingest] 数据健康分 %.3f < 阈值 %.2f — **拒绝当日摄取**（不静默降级）",
                score, self.min_health,
            )
            return IngestResult(
                False, dataset_name, as_of, health_score=score,
                n_dates=close.shape[0], n_tickers=close.shape[1],
                reject_reason=f"health_below_threshold({score:.3f}<{self.min_health})",
            )

        # Task 8.1：通过验收的数据按 as_of 追加进 PIT 存储（历史只追加不修改）。
        # PIT 写入失败**不阻塞**当日交易循环（存储是审计/复现资产，非交易前置），仅告警。
        #
        # 【缺陷 A-1，2026-09-20 修】`append_pit=False` 是给 `ingest_incremental`
        # 用的：那条路径**自己**会按 `> last` 过滤后再写一次，
        # 这里若也写，同一批 bar 就进了 PIT 两次（两个 vintage）。
        # 而且两次写的窗口还不完全相同 —— 这里写的是整个抓取窗口
        # （provider 可能返回一根重叠的旧 bar），外层写的才是严格的新增部分。
        if append_pit:
            try:
                self._append_pit(dataset_name, data, as_of)
            except Exception as exc:  # noqa: BLE001
                if pit_required:
                    logger.error("[daily_ingest] PIT 写入失败 → 拒绝（未落盘的数据不能作为前向起点）: %s",
                                 exc)
                    return IngestResult(False, dataset_name, as_of, health_score=score,
                                        reject_reason=f"pit_append_failed: {exc}")
                logger.warning("[daily_ingest] PIT 追加失败（不阻塞交易循环）: %s", exc)

        logger.info(
            "[daily_ingest] 摄取通过 '%s' | 健康=%.3f | %d 天 × %d 标的 | as_of=%s",
            dataset_name, score, close.shape[0], close.shape[1], as_of,
        )
        return IngestResult(
            True, dataset_name, as_of, health_score=score,
            n_dates=close.shape[0], n_tickers=close.shape[1], dataset=data,
            identity=_identity_of(data, data, as_of),
        )

    @staticmethod
    def _append_pit(dataset_name: str, data: Dict[str, pd.DataFrame], as_of: str) -> None:
        """把通过验收的 wide 面板按 as_of 追加进 PIT 存储（Task 8.1）。"""
        from app.config import settings
        from app.core.data_engine.pit_store import PITStore

        store = PITStore(settings.pit_store_dir)
        store.append(data, as_of=as_of, name=dataset_name)


#: 重叠那一根 bar 的收盘价，库里的与这次取到的相对差超过这个值 → 认为数据源重算了历史复权价。
#: 同一 vintage 重取应当逐位相同；分红带来的前复权调整通常 ≥ 1e-4。
REBASE_REL_TOL = 1e-6


def _rebase_reason(store, dataset_name: str, last: pd.Timestamp, inc: Dict[str, pd.DataFrame],
                   source: str) -> str:
    """需要整段重取的原因（空串 = 不需要）。只看重叠的那一根 bar（`last`）与 PIT 元数据。"""
    meta = store.read_meta(dataset_name)
    old_src = meta.get("price_source")
    if old_src and old_src != source:
        return f"价格源从 {old_src} 换成了 {source}"
    new_close = inc.get("close")
    if new_close is None or new_close.empty:
        return ""
    day = pd.Timestamp(last).normalize()
    rows = [d for d in new_close.index if pd.Timestamp(d).normalize() == day]
    if not rows:
        return ""
    old = store.load_pit(fields=["close"], start=str(day.date()), end=str(day.date()), name=dataset_name)
    old_close = (old or {}).get("close")
    if old_close is None or old_close.empty:
        return ""
    a = old_close.iloc[-1]
    b = new_close.loc[rows[-1]].reindex(a.index)
    both = a.notna() & b.notna() & (a != 0)
    rel = ((b[both] / a[both]) - 1.0).abs()
    moved = rel[rel > REBASE_REL_TOL]
    if moved.empty:
        return ""
    tk = moved.idxmax()
    return (f"{len(moved)} 只票在 {day.date()} 的复权收盘价与库中不一致"
            f"（最大 {tk}: {float(a[tk]):.4f} → {float(b[tk]):.4f}），数据源重算了历史复权")


def _identity_of(consumed: Dict[str, pd.DataFrame], checked: Dict[str, pd.DataFrame],
                 vintage: Optional[str]) -> Optional[Dict[str, Any]]:
    """交出去的面板的输入身份 + 健康门检查的那份的末日 / 股票池（两者应一致）。"""
    from app.config import settings
    from app.core.data_engine.input_identity import identify
    try:
        src = str(getattr(settings, "price_source", ""))
        out = identify(consumed, price_source=src, vintage=vintage).to_dict()
        chk = identify(checked, price_source=src, vintage=vintage)
    except Exception as exc:
        logger.error("[daily_ingest] 输入身份算不出来: %s", exc)
        return None
    out.update(checked_last_day=chk.last_day, checked_universe_digest=chk.universe_digest,
               checked_universe_n=chk.universe_n, checked_data_digest=chk.data_digest)
    return out


def _held_without_close(data: Dict[str, pd.DataFrame]) -> str:
    """券商账本里持有的标的，在最后一个交易日没有收盘价 → 说明原因；都有 → 空串。读不到持仓 → 拒绝。"""
    close = data.get("close")
    if close is None or close.empty:
        return ""
    try:
        from app.core.execution.order_manager import LIVE_BOOK_ID
        from app.db.position_store import PositionStore
        held = [tk for tk, w in (PositionStore().latest_positions(LIVE_BOOK_ID) or {}).items()
                if abs(float(w)) > 1e-12]
    except Exception as exc:
        logger.error("[daily_ingest] 读不到券商账本持仓 → 拒绝当日摄取: %s", exc)
        return f"读不到券商账本持仓，无法核对持仓标的的报价（{exc}）"
    last = close.iloc[-1]
    missing = sorted(tk for tk in held if tk not in close.columns or pd.isna(last.get(tk)))
    if not missing:
        return ""
    return (f"持仓标的 {missing[:10]}{' …' if len(missing) > 10 else ''} 在最后一个交易日 "
            f"{pd.Timestamp(close.index[-1]).date()} 没有收盘价")


def _execution_on() -> bool:
    from app.config import settings
    return str(getattr(settings, "execution_mode", "off")) != "off"


def _with_fundamentals(dataset_name: str, panel: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    开启基本面时，给喂进交易循环的面板挂上基本面字段（Phase 10）。

    PIT 面板是历次增量拼出来的（多个 vintage 的复权价），不能用来识别拆股 —— 所以不传
    adj_close，让 service 自己取一份单一 vintage 的。挂载失败 → 返回原面板并报错：
    引用基本面字段的因子会因"字段不存在"而失败（不调仓），不会拿缺的数据继续算。
    """
    from app.config import settings
    if not getattr(settings, "enable_fundamentals", False):
        return panel
    try:
        from app.core.data_engine.dataset_registry import _fundamentals_wanted, registry_spec
        spec = registry_spec(dataset_name)
        if not _fundamentals_wanted(spec, None):
            return panel
        from app.core.data_engine.fundamentals.service import attach_fundamentals
        out = dict(panel)
        attach_fundamentals(out, spec.universe)
        return out
    except Exception as exc:
        logger.error("[daily_pipeline] 基本面挂载失败 → 本轮面板**不含**基本面字段，"
                     "引用它们的因子将报错: %s", exc)
        return panel


def _pipeline_identity(ing, src: Optional[str]) -> Dict[str, Any]:
    """
    交给交易循环的输入身份。价格源**只**取 PIT 元数据里的实际来源（2026-10-22 用户核验 C2：此前读不到时
    退回摄取时按 settings 声称的来源）。读不到 → 空串 + source_verified=False；声称的来源另存一列供排查。
    """
    ident = dict(getattr(ing, "identity", None) or {})
    ident["claimed_price_source"] = ident.get("price_source", "")
    ident["price_source"] = str(src or "")
    ident["source_verified"] = bool(src)
    return ident


def _pit_source(dataset_name: str) -> Optional[str]:
    """PIT 元数据里记的**实际**价格源（复审 2026-10-19 N02：证据与规格核对用它，不用 settings 的声称）。"""
    try:
        from app.config import settings
        from app.core.data_engine.pit_store import PITStore
        return (PITStore(settings.pit_store_dir).read_meta(dataset_name) or {}).get("price_source")
    except Exception as exc:
        logger.error("[daily_pipeline] 读不到 PIT 元数据里的价格源: %s", exc)
        return None


def _run_portfolio_safely(loop, dataset, forward_from, now=None, data_source=None,
                          identity=None) -> Dict[str, Any]:
    """组合账本失败不拖垮 per-factor 监控；但执行层仍要对账/续做全平（失败时不加新风险）。"""
    try:
        return loop.run_portfolio(dataset, forward_from=forward_from, now=now, data_source=data_source,
                                  identity=identity)
    except Exception as exc:
        logger.error("[daily_pipeline] 组合账本失败 → 本轮不调仓，只做对账/熔断维护: %s", exc)
        pf = {"n_factors": 0, "days_processed": 0, "error": str(exc)}
        if _execution_on():
            pf["execution"] = loop._maintain_live(f"portfolio_error: {exc}")
        return pf


def run_daily_pipeline(dataset_name: str, start: str = "", end: str = "",
                       incremental: bool = True,
                       now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    完整每日管线：摄取 → 健康门 → （通过则）交易循环。供调度器调用。
    数据坏 → 跳过循环 + 返回拒绝原因（A2：绝不用坏数据继续）。

    Phase 11：默认 **incremental=True** —— 前向增量（PIT 空则先历史回填，之后每天只拉新 bar；
    无新 bar 就跳过，不空转）。`incremental=False` 保留旧的整段重拉（仅供回测/排查用）。
    `forward_from` 会传给交易循环，把新 bar 之后的 IC 标为**真前向**（TR.4 →ACTIVE 门只认这些）。
    """
    if incremental:
        ing = DailyIngest().ingest_incremental(dataset_name, backfill_start=start or None, now=now)
    else:
        ing = DailyIngest().ingest(dataset_name, start, end)

    from app.tasks.daily_trading_loop import DailyTradingLoop
    if not ing.accepted:
        # 【外部审计 F02】以前这两个分支直接 return：数据游标顶替了交易完成游标。
        # 摄取成功后若券商当时不可用 / 进程崩溃，同日重跑会因"无新 bar"跳过执行，
        # 熔断后的续做全平、停机期间的成交补记也都不会发生。
        if ing.mode == "no_new_bar":
            out = {"ingest_accepted": False, "reason": "no_new_bar",
                   "mode": ing.mode, "calendar_note": ing.calendar_note, "as_of": ing.as_of}
            if _execution_on():
                # 数据已是最新 → 用 PIT 面板重跑组合 + 执行。模拟账本按 last_pnl_date 幂等续跑、
                # 执行层按 client id 去重，重跑安全；决策日不新鲜时执行层自己会拒（只对账/续做全平）。
                from app.config import settings
                from app.core.data_engine.pit_store import PITStore
                panel = PITStore(settings.pit_store_dir).load_pit(name=dataset_name) or {}
                if panel.get("close") is not None and not panel["close"].empty:
                    panel = _with_fundamentals(dataset_name, panel)
                    out["portfolio"] = _run_portfolio_safely(DailyTradingLoop(), panel, None, now,
                                                             _pit_source(dataset_name))
                else:
                    out["portfolio"] = {"execution": DailyTradingLoop()._maintain_live("no_data")}
            logger.info("[daily_pipeline] 无新 bar（%s）", ing.calendar_note)
            return out
        logger.warning("[daily_pipeline] 摄取被拒（%s）→ 不调仓，只做对账/熔断维护", ing.reject_reason)
        out = {"ingest_accepted": False, "reject_reason": ing.reject_reason,
               "mode": ing.mode, "as_of": ing.as_of}
        if _execution_on():
            out["portfolio"] = {"execution": DailyTradingLoop()._maintain_live(
                f"ingest_rejected: {ing.reject_reason}")}
        return out

    loop = DailyTradingLoop()
    panel = _with_fundamentals(dataset_name, ing.dataset)
    # `now` = 决策时刻：当天的信号以它为存档时刻，明天据此判断结算是不是前向（F06）
    src = _pit_source(dataset_name)
    ident = _pipeline_identity(ing, src)
    report = loop.run(panel, forward_from=ing.forward_from, now=now,     # per-factor：IC 监控 + 衰减
                      data_source=src, identity=ident or None)
    # Phase PM.4：组合账本（真实 AUM 的美元账本，多因子净持仓 + 容量）
    pf = _run_portfolio_safely(loop, panel, ing.forward_from, now, src, ident or None)
    return {
        "ingest_accepted": True, "as_of": ing.as_of, "health_score": ing.health_score,
        "mode": ing.mode, "forward_from": ing.forward_from, "n_new_bars": ing.n_new_bars,
        "calendar_note": ing.calendar_note,
        "n_alphas": report.n_alphas, "n_alerts": report.n_alerts, "n_errors": report.n_errors,
        "portfolio": pf,
    }
