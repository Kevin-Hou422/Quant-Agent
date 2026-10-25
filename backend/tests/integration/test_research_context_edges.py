"""
研究上下文的边界与分支（D5 关键路径变异复核：app/core/data_engine/research_context.py）。

复核时 33 个点存活 20 个：冻结段检查恰好在 Test 起点那一天、三段切割被关掉时"真实数据集有没有边界"的分支、
登记过的边界带上 frozen_by / pinned、选择段起止日期、上下文身份串的规范写法、数据类的 compare / repr 设置、
读不到三段切割开关时偏严 —— 都没人钉。
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.core.data_engine import research_context as rc
from tests.conftest import _make_dataset

KEY = "audit_us"


@pytest.fixture
def pinned(fresh_db):
    from app.core.data_engine.data_partitioner import split_from_settings
    from app.db.trial_ledger import BoundaryStore
    ds = _make_dataset(1600, 10)
    frozen = split_from_settings(ds, dataset_key=KEY)
    BoundaryStore().register(KEY, frozen.test_start, actor="tester", reason="研究上下文边界用例")
    return ds, frozen.test_start


class TestFrozenSegmentCheck:

    def test_a_panel_ending_exactly_on_the_test_start_is_refused_and_counted(self, pinned):
        ds, start = pinned
        ctx = rc.open_research(ds, KEY, "cli.gp")
        upto = {k: v.loc[:start] for k, v in ds.items()}
        assert upto["close"].index.max() == start
        with pytest.raises(rc.FrozenBoundaryError, match=r" 1 个日期"):
            ctx.check_panels([upto])
        ctx.check_panels([{k: v.loc[:start - pd.Timedelta(days=1)] for k, v in ds.items()}])


class TestThreeWaySwitch:

    def test_off_on_a_real_dataset_without_a_boundary_uses_everything(self, fresh_db):
        ds = _make_dataset(1600, 10)
        ctx = rc.open_research(ds, "fresh_key", "cli.gp", three_way=False)
        assert ctx.note == "three_way_disabled" and ctx.test_start is None
        assert ctx.selection_end == str(ds["close"].index.max().date())

    def test_off_on_a_dataset_with_a_boundary_is_ignored(self, pinned):
        ds, start = pinned
        ctx = rc.open_research(ds, KEY, "cli.gp", three_way=False)
        assert ctx.note.startswith("three_way_disabled_ignored") and ctx.test_start == str(start.date())

    def test_an_unreadable_switch_means_cut_in_three(self, monkeypatch):
        import app.config

        class _Broken:
            def __getattr__(self, name):
                raise RuntimeError("配置坏了")
        monkeypatch.setattr(app.config, "settings", _Broken())
        assert rc._three_way_enabled() is True


class TestContextContents:

    def test_a_registered_boundary_is_pinned_and_says_who_froze_it(self, pinned):
        ds, start = pinned
        ctx = rc.open_research(ds, KEY, "cli.gp")
        assert ctx.pinned is True and ctx.boundary == "registered" and ctx.frozen_by == "pinned"
        assert ctx.selection_start == str(ds["close"].index.min().date())
        assert pd.Timestamp(ctx.selection_end) < start

    def test_a_provisional_split_is_not_pinned(self, fresh_db):
        ctx = rc.open_research(_make_dataset(1600, 10), "unregistered_key", "cli.gp")
        assert ctx.boundary == "provisional" and ctx.pinned is False and ctx.test_start is not None

    def test_an_empty_selection_has_no_dates(self):
        ctx = rc._build({}, "synthetic", "e", "p", None, pinned=False, note="")
        assert (ctx.selection_start, ctx.selection_end, ctx.frozen_by, ctx.pinned) == ("", "", "", False)

    def test_the_identity_string_is_canonical_readable_json(self, pinned):
        ds, _ = pinned
        ctx = rc.open_research(ds, KEY, "cli.gp", experiment="动量复核")
        assert ctx.to_json() == json.dumps(ctx.describe(), ensure_ascii=False, sort_keys=True)
        assert "动量复核" in ctx.to_json()

    def test_contexts_compare_by_description_not_by_their_dataframes(self, pinned):
        ds, _ = pinned
        a = rc.open_research(ds, KEY, "cli.gp")
        b = rc.open_research({k: v.copy() for k, v in ds.items()}, KEY, "cli.gp")
        assert a == b
        r = repr(a)
        assert "selection=" not in r and "split=" not in r and len(r) < 2000


def test_without_a_split_nothing_is_pinned():
    """L143 / L151 等价的机械验证：没有切分时 test_start 是 None，pinned 由 `pinned and ts is not None and …` 算出，传 True 也是 False。"""
    for pinned in (False, True):
        ctx = rc._build(_make_dataset(50, 3), "audit_us", "e", "p", None, pinned=pinned, note="")
        assert ctx.pinned is False and ctx.test_start is None


def test_a_pinned_boundary_with_too_little_data_raises_instead_of_falling_back(pinned):
    """L152 等价的机械验证：note 非空只发生在"关掉三段切割、但已有冻结边界"时；那时数据不够切 → FrozenBoundaryError，
    走不到拼接 note 的 ValueError 分支（那个分支只在没有边界时出现，而那时 note 是空串）。"""
    ds, _ = pinned
    with pytest.raises(rc.FrozenBoundaryError):
        rc.open_research({k: v.iloc[:30] for k, v in ds.items()}, KEY, "x", three_way=False)


PROVEN_EQUIVALENT = {
    "app/core/data_engine/research_context.py ×1 — L152 `note + \"; \"` -> `note - \"; \"`（数据太短时的说明拼接）":
        "这个分支只在 split_from_settings 抛 ValueError（没有冻结边界、数据不够切）时走到，那时 note 是空串，"
        "三元表达式取 else 分支，`+` 还是 `-` 都不执行；note 非空意味着已有冻结边界，数据不够时抛的是 FrozenBoundaryError。"
        "见 test_a_pinned_boundary_with_too_little_data_raises_instead_of_falling_back。",
    "app/core/data_engine/research_context.py ×2 — L143 / L151 `_build(..., pinned=False, ...)` -> `pinned=True`（不切三段 / 数据太短）":
        "这两处都传 split=None：`_build` 里 ts = None，pinned = bool(pinned and ts is not None and …)，"
        "传进来的 pinned 取什么结果都是 False。见 test_without_a_split_nothing_is_pinned。",
}


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 2
