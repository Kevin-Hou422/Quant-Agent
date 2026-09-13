"""
db/diagnostics_store.py + trading_context/providers.py —— 定钉测试（变异测试驱动）

来由：`diagnostics_store` 7 个变异点首测击杀率 28.6%（存活 5），
`providers` 4 个变异点 50.0%（存活 2）。

诊断台账是**每日交易的事后可审计记录**（前端与复盘都读它），
providers 则是"账户里现在有多少可用买入力、哪些票能做空"的唯一出处 ——
后者算错会直接影响下一轮建仓规模。

存活项：
  - `run_at` 的 `index=True`（按时间倒序取最近 N 轮是唯一的查询模式）
  - `json.dumps(..., ensure_ascii=False, default=str)` 的两个取值
  - 引擎的 `check_same_thread` / `echo` / `expire_on_commit`
  - `self._shortable.get(ticker, False)` 的**缺省不可做空**
  - `cash = max(0, 1 - invested) * equity` 的减号
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import inspect

from app.db.diagnostics_store import DiagnosticsStore


# ===========================================================================
# A. 诊断台账
# ===========================================================================

class TestDiagnosticsStore:

    @pytest.fixture
    def store(self, tmp_path) -> DiagnosticsStore:
        return DiagnosticsStore(db_url=f"sqlite:///{tmp_path/'d.db'}")

    def test_run_at_is_indexed(self, store):
        """`index=True`：`recent()` 按 run_at 倒序取，是这张表唯一的查询模式。"""
        indexed = {c for ix in inspect(store._engine).get_indexes("portfolio_diagnostics")
                   for c in ix["column_names"]}
        assert "run_at" in indexed

    def test_recent_returns_newest_first(self, store):
        for i in range(5):
            store.save({"n": i})
        got = store.recent(limit=3)
        assert len(got) == 3
        assert [g["n"] for g in got] == [4, 3, 2], f"顺序不是新→旧：{got}"

    def test_non_ascii_payload_is_stored_readably(self, store):
        """
        `ensure_ascii=False` —— 改成 True 会把中文转义成 `\\uXXXX`，
        诊断台账变成不可读的转义串（它是给人复盘用的）。
        """
        rid = store.save({"结论": "本轮未交易", "原因": "无新 bar"})
        with store._Session() as s:
            from app.db.diagnostics_store import PortfolioDiagnostic
            raw = s.get(PortfolioDiagnostic, rid).payload
        assert "本轮未交易" in raw, f"中文被转义了：{raw}"
        assert json.loads(raw)["原因"] == "无新 bar"

    def test_non_serialisable_values_do_not_break_the_save(self, store):
        """
        `default=str` —— 交易循环的诊断里混着 Timestamp / ndarray。
        去掉它会抛 TypeError，**诊断写不进去**，而这一轮到底发生了什么就此没有记录。
        """
        payload = {"as_of": pd.Timestamp("2024-03-05"),
                   "weights": np.arange(3),
                   "nested": {"t": pd.Timestamp("2024-03-06")}}
        rid = store.save(payload)
        got = store.recent(limit=1)[0]
        assert rid > 0
        assert "2024-03-05" in str(got["as_of"])
        assert isinstance(got["weights"], str)

    def test_engine_does_not_echo_sql(self, store):
        assert store._engine.echo is False

    def test_sqlite_allows_cross_thread_use(self, tmp_path):
        """调度线程写诊断、API 线程读诊断。"""
        import threading
        st = DiagnosticsStore(db_url=f"sqlite:///{tmp_path/'t.db'}")
        st.save({"n": 1})
        box = {}

        def _read():
            try:
                box["v"] = st.recent(limit=1)
            except Exception as exc:      # noqa: BLE001
                box["err"] = exc

        th = threading.Thread(target=_read)
        th.start()
        th.join()
        assert "err" not in box, f"跨线程读取失败：{box.get('err')}"
        assert box["v"][0]["n"] == 1

    def test_save_returns_the_row_id_after_commit(self, store):
        """`expire_on_commit=False`：save 在 commit 之后读 `rec.id`。"""
        rid = store.save({"a": 1})
        assert isinstance(rid, int) and rid > 0

    def test_corrupt_payload_does_not_break_recent(self, store):
        """坏 JSON 不得让整段历史读不出来。"""
        from app.db.diagnostics_store import PortfolioDiagnostic
        store.save({"ok": True})
        with store._Session() as s:
            s.add(PortfolioDiagnostic(payload="{not json"))
            s.commit()
        got = store.recent(limit=10)
        assert any(g.get("ok") is True for g in got)


# ===========================================================================
# B. 仿真 providers
# ===========================================================================

def _panel(days: int = 30, tickers=("AAA", "BBB"), price: float = 50.0,
           adv: float = 1e8) -> dict:
    idx = pd.bdate_range("2024-01-02", periods=days)
    cols = list(tickers)
    close = pd.DataFrame(price, index=idx, columns=cols)
    volume = pd.DataFrame(adv / price, index=idx, columns=cols)
    return {"close": close, "open": close, "high": close * 1.003,
            "low": close * 0.997, "vwap": close, "volume": volume}


class TestSimBorrowProvider:

    def test_unknown_ticker_is_not_shortable(self):
        """
        `self._shortable.get(ticker, False)` —— **未知标的默认不可做空**。
        默认值改成 True 会让任何拼错的/新上市的代码都被当成可做空，
        而"能不能借到券"恰恰是最不该乐观假设的一件事（DEV_LESSONS §U）。
        """
        from app.core.trading_context.providers import SimBorrowProvider
        p = SimBorrowProvider(_panel(), aum=10_000.0, account_type="margin",
                              allow_short=True)
        assert p.is_shortable("NEVER_HEARD_OF_IT") is False
        assert p.borrow_fee_bps("AAA") == 0.0

    def test_long_only_account_can_short_nothing(self):
        from app.core.trading_context.providers import SimBorrowProvider
        p = SimBorrowProvider(_panel(), aum=10_000.0, account_type="cash",
                              allow_short=False)
        assert p.is_shortable("AAA") is False


class TestSimAccountProvider:

    @staticmethod
    def _broker(tmp_path, capital: float = 100_000.0):
        from app.core.execution.paper_broker import PaperBroker
        from app.db.position_store import PositionStore
        return PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp_path/'p.db'}"),
                           initial_capital=capital)

    def test_cash_is_the_uninvested_share_of_equity(self, tmp_path):
        """
        `cash = max(0, 1 - invested) * equity`。把 `-` 写成 `+` 会得到
        **(1 + 已投比例) × 权益** —— 满仓时现金反而是权益的两倍，
        下一轮据此加仓等于凭空加杠杆。
        """
        from app.core.execution.paper_broker import DailyPnL
        from app.core.trading_context.providers import SimAccountProvider

        broker = self._broker(tmp_path)
        broker.store.record_day(
            0, "2024-01-02", {"AAA": 0.6, "BBB": -0.1}, [],
            DailyPnL(alpha_id=0, date="2024-01-02", gross_ret=0.0, net_ret=0.0,
                     cost_bps=0.0, equity=1.0))
        acct = SimAccountProvider(broker, book_id=0)
        assert acct.buying_power() == pytest.approx(100_000.0, abs=1e-6)
        # 已投权重绝对值之和 = 0.7 → 现金 = 0.3 × 100k
        assert acct.cash() == pytest.approx(30_000.0, abs=1e-6), (
            "现金不是未投入部分 —— 1 - invested 的减号疑似被写成加号")

    def test_cash_is_never_negative(self, tmp_path):
        """已投超过 100%（杠杆）时现金应被 max(0, …) 截到 0，而不是负数。"""
        from app.core.execution.paper_broker import DailyPnL
        from app.core.trading_context.providers import SimAccountProvider

        broker = self._broker(tmp_path)
        broker.store.record_day(
            0, "2024-01-02", {"AAA": 0.9, "BBB": -0.8}, [],
            DailyPnL(alpha_id=0, date="2024-01-02", gross_ret=0.0, net_ret=0.0,
                     cost_bps=0.0, equity=1.0))
        acct = SimAccountProvider(broker, book_id=0)
        assert acct.cash() == pytest.approx(0.0, abs=1e-9)

    def test_equity_tracks_the_normalised_curve(self, tmp_path):
        from app.core.execution.paper_broker import DailyPnL
        from app.core.trading_context.providers import SimAccountProvider

        broker = self._broker(tmp_path, capital=50_000.0)
        broker.store.record_day(
            0, "2024-01-02", {"AAA": 0.5}, [],
            DailyPnL(alpha_id=0, date="2024-01-02", gross_ret=0.0, net_ret=0.0,
                     cost_bps=0.0, equity=1.2))
        acct = SimAccountProvider(broker, book_id=0)
        assert acct.buying_power() == pytest.approx(60_000.0, abs=1e-6)
        assert acct.cash() == pytest.approx(0.5 * 60_000.0, abs=1e-6)

    def test_positions_are_read_from_the_book(self, tmp_path):
        from app.core.execution.paper_broker import DailyPnL
        from app.core.trading_context.providers import SimAccountProvider

        broker = self._broker(tmp_path)
        broker.store.record_day(
            0, "2024-01-02", {"AAA": 0.4, "BBB": -0.2}, [],
            DailyPnL(alpha_id=0, date="2024-01-02", gross_ret=0.0, net_ret=0.0,
                     cost_bps=0.0, equity=1.0))
        pos = SimAccountProvider(broker, book_id=0).positions()
        assert pos == {"AAA": pytest.approx(0.4), "BBB": pytest.approx(-0.2)}

    def test_position_read_failure_raises_instead_of_returning_empty(self, tmp_path,
                                                                    monkeypatch):
        """
        读持仓失败**绝不能返回 {}** —— 空字典等于"我什么都没持有"，
        下游会把全部权益当成可用买入力按满额重新建仓（凭空加杠杆）。
        """
        from app.core.trading_context.providers import SimAccountProvider
        from app.db.position_store import PositionStore

        broker = self._broker(tmp_path)
        monkeypatch.setattr(
            PositionStore, "latest_positions",
            lambda self, aid: (_ for _ in ()).throw(RuntimeError("db down")))
        acct = SimAccountProvider(broker, book_id=0)
        with pytest.raises(Exception):
            acct.positions()
