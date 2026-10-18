"""
外部审查 2026-10-18 F02：多次抓取的前复权价被直接拼进 PIT。

增量摄取故意重叠上一根 bar，但只把 `date > last` 的部分写进 PIT。数据源因分红 / 拆股重算了
历史复权价时，新 bar 是新尺度、库里的旧 bar 是旧尺度，接缝处就是一个假收益：
审查的 R4 里旧 100、新 50，拼出 −50%，而数据源自己算的收益是 0。

修复后的契约：重叠那一根的复权价与库里不一致（或价格源换了）→ 整段重取，作为新 vintage 追加。
最新视图是同一个尺度；旧 vintage 原样保留，按旧的 as_of 查询仍能看到当时看到的数。
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import app.core.data_engine.dataset_registry as reg
from app.core.data_engine.pit_store import PITStore
from app.tasks.daily_ingest import REBASE_REL_TOL, DailyIngest

DAYS = pd.bdate_range("2024-03-04", "2024-03-29")
SPLIT_DAY = pd.Timestamp("2024-03-25")              # 一拆二：当天起成交价减半
COLS = ["AAA", "BBB"]


def _close_utc(day) -> datetime:
    """该交易日收盘后 2 小时（UTC），让 last_closed_session = 这一天。"""
    return datetime(day.year, day.month, day.day, 22, 0, tzinfo=timezone.utc)


class Provider:
    """
    前复权数据源替身。`AAA` 在 SPLIT_DAY 一拆二；拆股一旦发生（请求的末日 ≥ SPLIT_DAY），
    整段历史都按拆后尺度给出 —— 前复权的真实行为。`BBB` 没有公司行动。
    `nudge`：给某天的 AAA 收盘价乘一个系数（模拟分红带来的微小重算）。
    """

    def __init__(self):
        self.calls = []
        self.nudge = 1.0
        self.fail_full_fetch = False

    def panel(self, start, end):
        end = pd.Timestamp(end)
        split_known = end >= SPLIT_DAY
        idx = [d for d in DAYS if pd.Timestamp(start) <= d <= end]
        aaa = [50.0 if (split_known or d >= SPLIT_DAY) else 100.0 for d in idx]
        close = pd.DataFrame({"AAA": aaa, "BBB": [20.0 + 0.1 * DAYS.get_loc(d) for d in idx]},
                             index=pd.DatetimeIndex(idx))
        close["AAA"] *= self.nudge
        rets = np.log(close / close.shift(1))
        return {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
                "volume": pd.DataFrame(1e6, index=close.index, columns=COLS), "returns": rets}

    def load(self, name, start=None, end=None, health_check=False, with_fundamentals=None):
        self.calls.append((start, end))
        if self.fail_full_fetch and pd.Timestamp(start) <= DAYS[0]:
            raise RuntimeError("provider down")
        return type("DS", (), {"data": self.panel(start, end)})()


@pytest.fixture
def world(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "pit_store_dir", str(tmp_path / "pit"))
    monkeypatch.setattr(settings, "paper_start", str(DAYS[0].date()))
    monkeypatch.setattr(settings, "price_source", "moomoo")
    prov = Provider()
    monkeypatch.setattr(reg, "load_registry_dataset", prov.load)
    monkeypatch.setattr(reg, "check_dataset_health",
                        lambda ds, min_score=0.7, warn_only=True:
                        type("R", (), {"overall_score": 1.0, "hard_failures": []})())
    return prov, PITStore(tmp_path / "pit"), settings


def _run(day):
    return DailyIngest().ingest_incremental("px", now=_close_utc(day))


def test_a_split_triggers_a_full_rebase_and_the_seam_carries_no_fake_return(world):
    prov, pit, _ = world
    first = _run(pd.Timestamp("2024-03-22"))                 # 拆股前：回填，AAA 全是 100
    assert first.mode == "full" and first.dataset["close"]["AAA"].iloc[-1] == 100.0
    res = _run(SPLIT_DAY)                                     # 拆股当天：重叠的 03-22 变成 50
    assert res.accepted and res.mode == "incremental_rebased"
    assert res.forward_from == str(SPLIT_DAY.date()) and res.n_new_bars == 1
    aaa = res.dataset["close"]["AAA"]
    assert (aaa == 50.0).all()                                # 最新视图：同一个尺度
    assert aaa.pct_change().loc[SPLIT_DAY] == 0.0             # 此前是 −50%
    assert prov.calls[-1][0] == str(DAYS[0].date())           # 整段重取从库里最早的一天开始
    assert "复权收盘价与库中不一致" in pit.read_meta("px")["last_rebase_reason"]


def test_history_as_seen_before_the_rebase_is_still_available(world):
    _, pit, _ = world
    _run(pd.Timestamp("2024-03-22"))
    _run(SPLIT_DAY)
    vintages = pit.available_as_of("px")
    assert len(vintages) == 2
    before = pit.load_pit(name="px", as_of=vintages[0])["close"]["AAA"]
    assert (before == 100.0).all() and before.index[-1] == pd.Timestamp("2024-03-22")


def test_no_corporate_action_means_a_plain_increment(world):
    prov, pit, _ = world
    _run(pd.Timestamp("2024-03-19"))
    res = _run(pd.Timestamp("2024-03-20"))
    assert res.mode == "incremental" and res.n_new_bars == 1
    assert prov.calls[-1][0] == "2024-03-19"                  # 只取了重叠一天 + 新的一天
    long = pit._load_long("px")
    assert long.groupby(["timestamp", "ticker"]).size().max() == 1   # 没有整段重写
    assert "last_rebase_reason" not in pit.read_meta("px")


@pytest.mark.parametrize("nudge,rebased", [(1 + REBASE_REL_TOL, False), (1 + 3 * REBASE_REL_TOL, True)])
def test_rebase_tolerance_boundary(world, nudge, rebased):
    prov, _, _ = world
    _run(pd.Timestamp("2024-03-19"))
    prov.nudge = nudge
    res = _run(pd.Timestamp("2024-03-20"))
    assert (res.mode == "incremental_rebased") is rebased


def test_a_small_dividend_style_readjustment_also_rebases(world):
    prov, _, _ = world
    _run(pd.Timestamp("2024-03-19"))
    prov.nudge = 0.997                                        # 0.3% 的分红调整
    res = _run(pd.Timestamp("2024-03-20"))
    assert res.mode == "incremental_rebased"
    assert res.dataset["close"]["AAA"].pct_change().dropna().abs().max() < 1e-12


def test_switching_the_price_source_rebases(world):
    prov, pit, settings = world
    _run(pd.Timestamp("2024-03-19"))
    assert pit.read_meta("px")["price_source"] == "moomoo"
    settings.price_source = "yahoo"
    res = _run(pd.Timestamp("2024-03-20"))
    assert res.mode == "incremental_rebased"
    meta = pit.read_meta("px")
    assert meta["price_source"] == "yahoo" and "moomoo 换成了 yahoo" in meta["last_rebase_reason"]


def test_a_failed_rebase_fetch_rejects_the_day_and_leaves_the_pit_alone(world):
    prov, pit, _ = world
    _run(pd.Timestamp("2024-03-22"))
    before = len(pit._load_long("px"))
    prov.fail_full_fetch = True
    res = _run(SPLIT_DAY)
    assert res.accepted is False and res.reject_reason.startswith("rebase_failed")
    assert len(pit._load_long("px")) == before


def test_a_legacy_pit_without_metadata_is_still_checked_for_scale(world):
    _, pit, _ = world
    _run(pd.Timestamp("2024-03-22"))
    (pit.store_dir / "px" / "meta.json").unlink()
    res = _run(SPLIT_DAY)
    assert res.mode == "incremental_rebased"
    assert pit.read_meta("px")["price_source"] == "moomoo"
