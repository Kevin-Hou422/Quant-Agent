"""
test_holdout_ledger.py — Phase S.2 冻结 Test 段的使用次数台账

为什么这本账必须存在：`一次性使用` 写在文档里就等于没写。同一段 holdout 跑到
第 20 次时，选出来的东西早就是按它挑的了，而屏幕上的数字和第 1 次长得一模一样。
这些用例钉住的是"次数真的被记下来、超了真的说出来、记不上也不许装作第一次"。
"""
from __future__ import annotations

import pytest

from app.db.trial_ledger import HoldoutLedger


@pytest.fixture
def ledger(tmp_path):
    return HoldoutLedger(db_url=f"sqlite:///{tmp_path / 'holdout.db'}", budget=1)


class TestCounting:

    def test_the_first_use_reports_one_and_is_within_budget(self, ledger):
        u = ledger.record_use("us_tech_large", "2022-01-03..2024-01-02", "backtest/run")
        assert u.uses == 1, f"第一次使用记成了 {u.uses} 次"
        assert u.over_budget is False
        assert u.recorded is True

    def test_the_second_use_is_over_budget(self, ledger):
        key = "2022-01-03..2024-01-02"
        ledger.record_use("us_tech_large", key)
        u = ledger.record_use("us_tech_large", key)
        assert u.uses == 2, f"第二次使用记成了 {u.uses} 次"
        assert u.over_budget is True, (
            "预算=1 时第二次使用没有标 over_budget —— 『一次性』就只剩一句口号")

    def test_a_larger_budget_moves_the_line(self, tmp_path):
        led = HoldoutLedger(db_url=f"sqlite:///{tmp_path / 'h.db'}", budget=3)
        key = "k"
        uses = [led.record_use("ds", key).over_budget for _ in range(4)]
        assert uses == [False, False, False, True], f"预算=3 的越界点不对：{uses}"

    def test_different_test_windows_are_counted_separately(self, ledger):
        ledger.record_use("ds", "2022-01-03..2024-01-02")
        u = ledger.record_use("ds", "2021-01-04..2023-01-03")
        assert u.uses == 1, (
            f"换了一个 Test 窗口却继承了上一段的计数（{u.uses}）—— "
            f"每个窗口必须各记各的")

    def test_different_datasets_are_counted_separately(self, ledger):
        ledger.record_use("us_tech_large", "k")
        u = ledger.record_use("us_broad_large", "k")
        assert u.uses == 1, f"不同数据集的同名窗口被并到一起数了（{u.uses}）"

    def test_count_matches_what_record_use_reported(self, ledger):
        for _ in range(3):
            ledger.record_use("ds", "k")
        assert ledger.count("ds", "k") == 3

    def test_counts_survive_a_new_connection(self, tmp_path):
        """跨会话累计 —— 重开一个 ledger 必须看见之前的次数，否则重启就能"洗白"。"""
        url = f"sqlite:///{tmp_path / 'h.db'}"
        HoldoutLedger(db_url=url).record_use("ds", "k")
        assert HoldoutLedger(db_url=url).count("ds", "k") == 1


class TestItRefusesNonsense:

    def test_a_budget_below_one_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="budget"):
            HoldoutLedger(db_url=f"sqlite:///{tmp_path / 'h.db'}", budget=0)

    @pytest.mark.parametrize("missing", ["dataset_key", "test_key"])
    def test_a_usage_row_without_a_key_is_rejected_by_the_schema(self, ledger, missing):
        """
        `dataset_key` / `test_key` 都是 `nullable=False`。允许 NULL 的后果是
        台账里出现一条**无法归属**的使用记录：次数确实 +1 了，却不知道加在
        哪一段 holdout 上 —— 那条线依然形同虚设。

        两个键都要测：只测一个的话，另一个被放开时这条照样绿。
        """
        import sqlalchemy

        from app.db.trial_ledger import HoldoutUse

        kwargs = {"dataset_key": "ds", "test_key": "k", "purpose": ""}
        kwargs[missing] = None
        with pytest.raises((sqlalchemy.exc.IntegrityError,
                            sqlalchemy.exc.StatementError)):
            with ledger._Session() as s:      # noqa: SLF001 —— 故意绕过 record_use 测 schema
                s.add(HoldoutUse(**kwargs))
                s.commit()


class TestItWorksAcrossThreads:

    def test_the_ledger_can_be_used_from_another_thread(self, ledger):
        """
        SQLite 默认 `check_same_thread=True`：换线程用同一个连接会直接抛错。
        本台账会被调度器/线程池里的 run_portfolio、以及 FastAPI 的线程池
        执行的同步端点调用 —— 这个开关配错，线上表现是"记账偶发崩溃"，
        而单线程测试永远绿。
        """
        import threading

        box: dict = {}

        def _work():
            try:
                box["usage"] = ledger.record_use("ds", "k", "from-thread")
            except Exception as exc:          # 把异常带回主线程，否则会被静默吞掉
                box["error"] = exc

        t = threading.Thread(target=_work)
        t.start()
        t.join(timeout=30)
        assert "error" not in box, f"换线程调用台账抛了错：{box.get('error')!r}"
        assert box["usage"].uses == 1


class TestAccountingFailureIsVisible:

    def test_an_unwritable_ledger_still_returns_a_result_but_marks_it_unrecorded(
            self, monkeypatch):
        """
        台账写不进去时，结论照常返回（否则大家会干脆绕开台账），
        但 `recorded=False` 必须跟着返回值一起出去 —— 不许让一个失真的
        "uses" 冒充真实次数。
        """
        from app.core.data_engine import data_partitioner as dp

        class _Boom:
            def __init__(self, *a, **kw):
                raise RuntimeError("台账不可写")

        monkeypatch.setattr("app.db.trial_ledger.HoldoutLedger", _Boom)

        split = dp.partition_three_way(_mini_panel())
        payload = dp.account_holdout_use(split, dataset_key="ds", purpose="t")
        assert payload["recorded"] is False, "台账写失败却仍自称已记账"
        assert payload["uses"] == -1, (
            f"记不上账时 uses 应是明确的哨兵值，实际 {payload['uses']} —— "
            f"返回 1 会被读成『这是第一次看这段数据』")
        assert payload["test_key"] == split.test_key
        assert payload["budget"] == 1, (
            f"记账失败时 budget 报成了 {payload['budget']!r} —— None 会让下游的"
            f"『uses > budget』比较直接抛错或恒假，那条线就没了")
        assert payload["over_budget"] is False


def _mini_panel():
    import numpy as np
    import pandas as pd
    idx = pd.bdate_range("2018-01-01", periods=1200)
    cols = ["A", "B"]
    rng = np.random.default_rng(0)
    close = pd.DataFrame(rng.normal(100, 1, (1200, 2)), index=idx, columns=cols)
    return {"close": close, "volume": pd.DataFrame(1e6, index=idx, columns=cols)}
