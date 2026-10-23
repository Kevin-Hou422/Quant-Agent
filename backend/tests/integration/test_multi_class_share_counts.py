"""
多类别股缺股本：市值与估值字段保持 NaN，不回填、不变成零市值或便宜估值。

来源（原测试文件的说明，保留出处）：

2026-10-22 用户决定 · 批 3：证据 / 研究纪律。

D12 原话："同意先只读查询，不授权写边界。使用明确的活库路径和真正只读连接，不调用会建表或迁移的 Store 构造函数。
列出已有边界、最早曝光日期、次数、股票池和实验身份，给出日期建议后由我确认。正式入口不要继续依赖首次调用偶然冻结；
改实验名不能刷新重叠数据的曝光额度。"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# D7："保持 NaN。不拿今天快照回填历史，不把缺失变成零市值或便宜估值。"
# ---------------------------------------------------------------------------

class TestMultiClassSharesStayNaN:

    @staticmethod
    def _panels():
        from app.core.data_engine.fundamentals.panels import build_panels
        from tests.unit.data_engine import fundamentals_fixtures as fx
        aapl = fx.facts("AAPL")
        no_shares = aapl[~aapl["concept"].str.startswith("shares")].assign(ticker="BRK-B")   # 像 BRK-B / V：没有可用股本
        idx = pd.bdate_range("2023-03-01", "2023-06-30")
        raw = pd.DataFrame({"AAPL": 160.0, "BRK-B": 300.0}, index=idx)
        p, rep = build_panels(idx, ["AAPL", "BRK-B"], {"AAPL": aapl, "BRK-B": no_shares}, {}, raw)
        return p, rep

    def test_valuation_fields_are_nan_not_zero_or_cheap(self):
        p, _ = self._panels()
        for f in ("market_cap", "earnings_yield", "book_to_price"):
            assert p[f]["BRK-B"].isna().all(), f"{f}：没有股本却算出了值（零市值 / 便宜估值）"
            assert p[f]["AAPL"].notna().any(), f"前提：{f} 对正常公司有值"
        assert p["roe"]["BRK-B"].notna().any(), "不需要股本的字段不该受影响"

    def test_a_cross_sectional_rank_keeps_the_gap_as_a_gap(self):
        from app.core.alpha_engine.fast_ops import cs_fund_rank
        p, _ = self._panels()
        ey = p["earnings_yield"][["AAPL", "BRK-B"]].to_numpy(dtype=float)
        r = cs_fund_rank(np.column_stack([ey, ey[:, :1] * 0.9, ey[:, :1] * 1.1, ey[:, :1] * 1.2]),
                         min_coverage=0.5)
        assert np.isnan(r[:, 1]).all(), "缺失的估值在截面排名里被当成了某个名次"

    def test_no_snapshot_share_source_exists_in_the_fundamentals_path(self):
        """不拿"今天"的券商快照股本回填历史：基本面路径里没有任何快照股本的来源。"""
        import re
        root = Path(__file__).resolve().parents[2] / "app" / "core" / "data_engine" / "fundamentals"
        hits = [f.name for f in root.glob("*.py")
                if re.search(r"issued_shares|total_shares|outstanding_shares|market_snapshot",
                             f.read_text(encoding="utf-8"))]
        assert hits == [], hits
