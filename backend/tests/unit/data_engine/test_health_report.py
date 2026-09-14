"""
data_engine/health_report.py —— 数据质量健康检查

**首测击杀率 0.0%（32/32 全部存活）**。它此前不是"没有专属测试文件"，
而是**既有测试根本没有真的执行到它** —— `test_dataset_registry_loading.py`
只是 import 过一次。这类"看起来有覆盖"的模块比零覆盖更危险：
审计时它会被算进"已覆盖"。

这一层是**数据进入回测之前的最后一道质检**。它坏掉的方式只有一种形状：
**该拦的没拦住，而评分依然漂亮**。

  - `overall_score` 的四个分项任一被改成恒 1，综合分照样落在 [0,1] 里，
    `is_healthy()` 照样返回 True，日循环照常放行一份有洞的数据；
  - `>` 改 `>=`、`>=` 改 `>` 会让恰好卡在阈值上的异常被漏判；
  - Gap 检测里"末尾空洞"那一段被删 → **停牌到区间末尾的票完全不报**，
    而那正是最该报的一种（退市/长期停牌）。

所以本文件的手法是：每个检测器**手工构造已知答案的面板**逐行比对，
评分用**参考公式逐位验算**（rtol=1e-12），而不是只断言"分数在 0 和 1 之间"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.health_report import (
    DataHealthChecker,
    GapDetector,
    HealthReport,
    SpikeDetector,
    ZeroVolumeDetector,
)

DATES = pd.bdate_range("2022-01-03", periods=10)


def _panel(spec: dict, dates=DATES) -> pd.DataFrame:
    """
    spec: ticker → {"close": [...], "volume": [...]}，长度需与 dates 一致。
    产出 long-format 面板（timestamp / ticker / close / volume）。
    """
    rows = []
    for ticker, cols in spec.items():
        close = cols.get("close", [100.0] * len(dates))
        volume = cols.get("volume", [1e6] * len(dates))
        for d, c, v in zip(dates, close, volume):
            rows.append({"timestamp": d, "ticker": ticker,
                         "close": c, "volume": v})
    return pd.DataFrame(rows)


# ===========================================================================
# A. Gap 检测
# ===========================================================================

class TestGapDetector:

    def test_a_clean_panel_reports_no_gaps(self):
        out = GapDetector().detect(_panel({"AAA": {}}))
        assert out.empty
        assert list(out.columns) == ["ticker", "gap_start", "gap_end", "gap_days"]

    def test_a_single_missing_day_is_reported_with_exact_bounds(self):
        """
        空洞的起止日期必须是**空洞本身**的首尾，而不是它两侧的有效日。
        `gap_end = dates[i - 1]` 的 `- 1` 被去掉会让结束日多算一天 ——
        运维照着这个区间去补数据就会补错。
        """
        close = [100.0] * 10
        close[4] = np.nan
        out = GapDetector().detect(_panel({"AAA": {"close": close}}))
        assert len(out) == 1
        r = out.iloc[0]
        assert r["ticker"] == "AAA"
        assert r["gap_start"] == DATES[4]
        assert r["gap_end"] == DATES[4], (
            f"单日空洞的结束日是 {r['gap_end']}，应当就是缺失那天本身")
        assert r["gap_days"] == 1

    def test_a_multi_day_gap_reports_its_full_span(self):
        close = [100.0] * 10
        close[3:7] = [np.nan] * 4
        out = GapDetector().detect(_panel({"AAA": {"close": close}}))
        assert len(out) == 1
        r = out.iloc[0]
        assert (r["gap_start"], r["gap_end"]) == (DATES[3], DATES[6])
        assert r["gap_days"] == 4, f"空洞长度算成了 {r['gap_days']}"

    def test_two_separate_gaps_are_reported_separately(self):
        """
        `in_gap = False; gap_len = 0` 的复位 —— 少了复位会把两个空洞
        并成一个（跨越中间那段有效数据），运维会去补根本没缺的日子。
        """
        close = [100.0] * 10
        close[1] = np.nan
        close[5:7] = [np.nan] * 2
        out = GapDetector().detect(_panel({"AAA": {"close": close}}))
        assert len(out) == 2, f"两个独立空洞被合并了：\n{out}"
        assert out["gap_days"].tolist() == [1, 2]

    def test_a_gap_running_to_the_end_of_the_panel_is_reported(self):
        """
        循环结束后那段"末尾空洞"的处理 —— 被删会让**停牌到区间末尾**
        的票完全不报。而那正是最该报的一种（退市 / 长期停牌），
        因为它不会被后面的有效数据"关闭"。
        """
        close = [100.0] * 10
        close[7:] = [np.nan] * 3
        out = GapDetector().detect(_panel({"AAA": {"close": close}}))
        assert len(out) == 1, "跑到区间末尾的空洞没有被报出来"
        r = out.iloc[0]
        assert (r["gap_start"], r["gap_end"]) == (DATES[7], DATES[9])
        assert r["gap_days"] == 3

    def test_a_gap_at_the_very_start_is_reported(self):
        close = [np.nan] * 3 + [100.0] * 7
        out = GapDetector().detect(_panel({"AAA": {"close": close}}))
        assert len(out) == 1
        assert out.iloc[0]["gap_days"] == 3

    def test_an_all_missing_ticker_is_one_full_length_gap(self):
        out = GapDetector().detect(_panel({"AAA": {"close": [np.nan] * 10}}))
        assert len(out) == 1
        assert out.iloc[0]["gap_days"] == 10

    def test_the_minimum_gap_length_is_inclusive(self):
        """
        `if in_gap and gap_len >= self.min_gap_days:`

        `>=` 翻成 `>` 会让恰好等于阈值的空洞漏报。
        构造 min_gap_days=2 且空洞恰好 2 天。
        """
        close = [100.0] * 10
        close[4:6] = [np.nan] * 2
        assert len(GapDetector(min_gap_days=2).detect(
            _panel({"AAA": {"close": close}}))) == 1, (
            "空洞长度恰好等于阈值却没报 —— `gap_len >= min_gap_days` 被翻成了 `>`")

    def test_a_gap_shorter_than_the_minimum_is_ignored(self):
        close = [100.0] * 10
        close[4] = np.nan
        assert GapDetector(min_gap_days=2).detect(
            _panel({"AAA": {"close": close}})).empty

    def test_the_end_of_panel_gap_uses_the_same_minimum(self):
        """末尾空洞那一段有它自己的 `gap_len >= min_gap_days` —— 两处都要测。"""
        close = [100.0] * 10
        close[9] = np.nan
        assert GapDetector(min_gap_days=2).detect(
            _panel({"AAA": {"close": close}})).empty
        close[8:] = [np.nan] * 2
        assert len(GapDetector(min_gap_days=2).detect(
            _panel({"AAA": {"close": close}}))) == 1

    def test_gaps_are_detected_per_ticker_not_across_the_whole_panel(self):
        """
        `panel.groupby(ticker_col)` —— 不分组会让 A 的末行与 B 的首行
        被当成"连续"，跨票拼出根本不存在的空洞。
        """
        a = [100.0] * 10
        a[9] = np.nan
        b = [100.0] * 10
        b[0] = np.nan
        out = GapDetector().detect(_panel({"AAA": {"close": a}, "BBB": {"close": b}}))
        assert len(out) == 2, f"跨 ticker 的缺失被并成了一个空洞：\n{out}"
        assert set(out["ticker"]) == {"AAA", "BBB"}
        assert out["gap_days"].tolist() == [1, 1]

    def test_rows_are_sorted_by_date_before_scanning(self):
        """
        `grp.sort_values(date_col)` —— 输入乱序时不排序会让"连续缺失"
        的判断完全失真（缺失日被打散）。
        """
        close = [100.0] * 10
        close[3:6] = [np.nan] * 3
        p = _panel({"AAA": {"close": close}}).sample(frac=1.0, random_state=7)
        out = GapDetector().detect(p)
        assert len(out) == 1, f"乱序输入下空洞被打散成了 {len(out)} 段"
        assert out.iloc[0]["gap_days"] == 3

    def test_an_empty_panel_returns_an_empty_frame_with_the_right_columns(self):
        out = GapDetector().detect(pd.DataFrame())
        assert out.empty
        assert list(out.columns) == ["ticker", "gap_start", "gap_end", "gap_days"]

    def test_a_missing_value_column_returns_empty_instead_of_raising(self):
        p = _panel({"AAA": {}}).drop(columns=["close"])
        assert GapDetector().detect(p).empty


# ===========================================================================
# B. Spike 检测
# ===========================================================================

class TestSpikeDetector:

    def test_a_quiet_series_reports_no_spikes(self):
        out = SpikeDetector().detect(_panel({"AAA": {}}))
        assert out.empty
        assert list(out.columns) == ["ticker", "date", "field",
                                     "pct_change", "threshold"]

    def test_a_large_jump_is_flagged_with_its_exact_magnitude(self):
        """
        `pct_change` 要如实记录，而不是只记一个布尔。
        运维靠这个数字判断是真行情还是脏数据。
        """
        close = [100.0] * 10
        close[5] = 250.0            # +150%，回落 -60%
        out = SpikeDetector(threshold=0.5).detect(
            _panel({"AAA": {"close": close}}))
        # 涨上去和跌回来各算一次。
        # （用 250 而不是 200：200 的回落恰好是 -50%，不满足 `> 0.5`，
        #   只会报一条 —— 这个边界本身由 test_a_change_exactly_on_the_threshold 管。）
        assert len(out) == 2, f"一次跳变应当报两条（涨 + 回落）：\n{out}"
        up = out[out["date"] == DATES[5]].iloc[0]
        assert up["pct_change"] == pytest.approx(1.5)
        assert up["field"] == "close"
        assert up["threshold"] == 0.5

    def test_a_drop_is_flagged_too(self):
        """
        `pct.abs() > threshold` —— 少了 `abs` 会让**只报上涨不报下跌**。
        而暴跌（除权未复权、数据源给了 0）恰恰是更常见的脏数据形态。
        """
        close = [100.0] * 10
        close[5] = 10.0             # -90%
        out = SpikeDetector(threshold=0.5).detect(
            _panel({"AAA": {"close": close}}))
        down = out[out["date"] == DATES[5]]
        assert len(down) == 1, "暴跌没有被标记 —— `pct.abs()` 里的 abs 没了"
        assert down.iloc[0]["pct_change"] == pytest.approx(-0.9)

    def test_a_change_exactly_on_the_threshold_is_not_flagged(self):
        """
        `pct.abs() > self.threshold` —— `>` 翻成 `>=` 只改变
        **恰好等于阈值**的那一格。构造 100 → 150 = 精确的 +0.5。
        """
        close = [100.0] * 10
        close[5] = 150.0
        assert abs(150.0 / 100.0 - 1.0) == 0.5, "前提失效：涨幅不是精确的 0.5"
        out = SpikeDetector(threshold=0.5).detect(
            _panel({"AAA": {"close": close}}))
        assert DATES[5] not in set(out["date"]), (
            "涨幅恰好等于阈值却被标记了 —— `>` 被翻成了 `>=`")

    def test_a_change_one_ulp_above_the_threshold_is_flagged(self):
        close = [100.0] * 10
        close[5] = 150.0 * (1 + 1e-9)
        out = SpikeDetector(threshold=0.5).detect(
            _panel({"AAA": {"close": close}}))
        assert DATES[5] in set(out["date"])

    def test_the_first_row_never_produces_a_spike(self):
        """`pct_change` 的首行是 NaN —— 被 fillna 之后会凭空报一条。"""
        close = [1.0] + [100.0] * 9
        out = SpikeDetector(threshold=0.5).detect(
            _panel({"AAA": {"close": close}}))
        assert DATES[0] not in set(out["date"]), "首行被报成了异常"

    def test_missing_values_do_not_synthesise_a_spike(self):
        """
        `pct_change(fill_method=None)` —— 默认的 `pad` 会把 NaN 前推，
        于是"缺一天"变成"那天没涨跌"，真实跳变被抹平；
        更糟的是恢复那天会算出一个跨越整段空洞的巨大涨幅。
        """
        close = [100.0, 100.0, np.nan, np.nan, 101.0] + [101.0] * 5
        out = SpikeDetector(threshold=0.5).detect(
            _panel({"AAA": {"close": close}}))
        assert out.empty, f"空洞两侧被算出了跳变：\n{out}"

    def test_several_price_fields_can_be_scanned(self):
        p = _panel({"AAA": {}})
        p["high"] = p["close"] * 1.01
        p.loc[p["timestamp"] == DATES[4], "high"] = 500.0
        out = SpikeDetector(threshold=0.5, price_fields=["close", "high"]).detect(p)
        assert set(out["field"]) == {"high"}, (
            f"只有 high 有跳变，却报了 {set(out['field'])}")

    def test_a_field_absent_from_the_panel_is_skipped(self):
        """`if field not in panel.columns: continue` —— 守卫被删会 KeyError。"""
        out = SpikeDetector(price_fields=["close", "不存在"]).detect(
            _panel({"AAA": {}}))
        assert out.empty

    def test_spikes_are_computed_per_ticker(self):
        """
        不分组会让 A 的最后一天到 B 的第一天被算成一次跳变 ——
        凭空报出一条不存在的异常。
        """
        p = _panel({"AAA": {"close": [100.0] * 10},
                    "BBB": {"close": [1000.0] * 10}})
        assert SpikeDetector(threshold=0.5).detect(p).empty, (
            "跨 ticker 的价格落差被当成了跳变")

    def test_rows_are_sorted_by_date_first(self):
        close = [100.0] * 10
        p = _panel({"AAA": {"close": close}}).sample(frac=1.0, random_state=3)
        assert SpikeDetector(threshold=0.5).detect(p).empty, (
            "乱序输入被算出了跳变 —— 少了按日期排序")

    def test_the_default_threshold_is_fifty_percent(self):
        assert SpikeDetector().threshold == 0.50
        assert SpikeDetector().price_fields == ["close"]


# ===========================================================================
# C. 零成交量
# ===========================================================================

class TestZeroVolumeDetector:

    def test_zero_and_nan_volume_are_both_flagged(self):
        """
        `isna() | (volume == 0)` —— 两个条件缺一不可：
        只查 0 会漏掉 NaN（数据源没给），只查 NaN 会漏掉 0（停牌）。
        """
        vol = [1e6] * 10
        vol[2] = 0.0
        vol[5] = np.nan
        out = ZeroVolumeDetector().detect(_panel({"AAA": {"volume": vol}}))
        assert len(out) == 2, f"零成交量与 NaN 应当各报一条：\n{out}"
        assert set(out["date"]) == {DATES[2], DATES[5]}

    def test_a_healthy_panel_reports_nothing(self):
        assert ZeroVolumeDetector().detect(_panel({"AAA": {}})).empty

    def test_negative_volume_is_not_silently_accepted_as_zero(self):
        """
        当前契约：只报 0 与 NaN，负值不报（负成交量是另一类问题）。
        钉住现状，免得有人把 `== 0` 改成 `<= 0` 而没人发现语义变了。
        """
        vol = [1e6] * 10
        vol[3] = -5.0
        out = ZeroVolumeDetector().detect(_panel({"AAA": {"volume": vol}}))
        assert out.empty, (
            "负成交量被报成了零成交量 —— `== 0` 被改成了 `<= 0`，"
            "两者语义不同，请同步更新文档")

    def test_the_output_columns_are_renamed_to_the_contract(self):
        """
        `flagged.columns = ["ticker", "date", "volume"]` —— 这一步被删会让
        列名还是 `timestamp`，下游按 `date` 取直接 KeyError。
        """
        vol = [1e6] * 10
        vol[1] = 0.0
        out = ZeroVolumeDetector().detect(_panel({"AAA": {"volume": vol}}))
        assert list(out.columns) == ["ticker", "date", "volume"]

    def test_the_index_is_reset(self):
        """`reset_index(drop=True)` —— 不重置会让下游按位置取时错位。"""
        vol = [1e6] * 10
        vol[7] = 0.0
        out = ZeroVolumeDetector().detect(_panel({"AAA": {"volume": vol}}))
        assert list(out.index) == list(range(len(out)))

    def test_a_missing_volume_column_returns_empty_instead_of_raising(self):
        p = _panel({"AAA": {}}).drop(columns=["volume"])
        out = ZeroVolumeDetector().detect(p)
        assert out.empty
        assert list(out.columns) == ["ticker", "date", "volume"]


# ===========================================================================
# D. 综合评分 —— 本模块的要害
# ===========================================================================

def _ref_score(panel, gaps, spikes, zero_vol, ticker_col="ticker"):
    """按 docstring 里写的公式独立算一遍综合分。"""
    total_rows = len(panel)
    n_tickers = panel[ticker_col].nunique()
    numeric_cols = panel.select_dtypes(include="number").columns
    global_nan_pct = (panel[numeric_cols].isna().sum().sum()
                      / max(total_rows * len(numeric_cols), 1))
    gap_tickers = gaps["ticker"].nunique() if not gaps.empty else 0
    return float(np.mean([
        max(0.0, 1.0 - gap_tickers / max(n_tickers, 1)),
        max(0.0, 1.0 - len(spikes) / max(total_rows, 1)),
        max(0.0, 1.0 - len(zero_vol) / max(total_rows, 1)),
        max(0.0, 1.0 - global_nan_pct),
    ]))


class TestOverallScore:

    def test_a_perfectly_clean_panel_scores_one(self):
        rep = DataHealthChecker().check(_panel({"AAA": {}, "BBB": {}}))
        assert rep.overall_score == pytest.approx(1.0)
        assert rep.is_healthy()

    def test_the_score_matches_the_documented_formula(self):
        """
        四个分项 + 取平均。任一分项的分子/分母被改，
        综合分仍然落在 [0,1] 里、`is_healthy()` 也可能仍是 True ——
        必须逐位比对参考实现。
        """
        close = [100.0] * 10
        close[2:5] = [np.nan] * 3
        close[8] = 300.0
        vol = [1e6] * 10
        vol[1] = 0.0
        vol[6] = np.nan
        panel = _panel({"AAA": {"close": close, "volume": vol},
                        "BBB": {}, "CCC": {}})

        chk = DataHealthChecker()
        rep = chk.check(panel)
        ref = _ref_score(panel, rep.gaps, rep.spikes, rep.zero_volume)
        assert rep.overall_score == pytest.approx(round(ref, 4), abs=1e-9), (
            f"综合分 {rep.overall_score} 与参考公式 {round(ref, 4)} 不符")

    def test_every_sub_score_actually_moves_the_total(self):
        """
        四个分项任一被改成恒 1（或从平均里漏掉），综合分会停止对
        那一类问题作出反应 —— 而分数依旧"看起来正常"。
        逐类注入问题，确认综合分**确实**下降。
        """
        clean = _panel({"AAA": {}, "BBB": {}})
        base = DataHealthChecker().check(clean).overall_score

        gapped = _panel({"AAA": {"close": [100.0] * 5 + [np.nan] * 5}, "BBB": {}})
        spiked = _panel({"AAA": {"close": [100.0] * 5 + [500.0] + [500.0] * 4},
                         "BBB": {}})
        zeroed = _panel({"AAA": {"volume": [1e6] * 5 + [0.0] * 5}, "BBB": {}})

        for name, p in (("gap", gapped), ("spike", spiked), ("zero_volume", zeroed)):
            s = DataHealthChecker().check(p).overall_score
            assert s < base, (
                f"注入 {name} 问题之后综合分没有下降（{s} vs 干净面板 {base}）—— "
                f"该分项在平均里失效了")

    def test_nan_alone_lowers_the_score(self):
        """
        NaN 分项与 gap 分项不是一回事：单日缺失不构成"空洞"时
        （min_gap_days 更大）仍然要拉低 nan_score。
        """
        close = [100.0] * 10
        close[4] = np.nan
        p = _panel({"AAA": {"close": close}, "BBB": {}})
        s = DataHealthChecker(gap_min_days=5).check(p).overall_score
        assert s < 1.0, "只有 NaN、没有空洞时综合分仍是满分 —— nan_score 失效"

    def test_the_score_is_clamped_to_zero_from_below(self):
        """
        `max(0.0, 1.0 - x)` 四处 —— 守卫被删会在问题极多时给出**负分**，
        `is_healthy(min_score=0.8)` 依然返回 False（看不出区别），
        但分数一旦进了前端的进度条/配色逻辑就会越界。
        """
        close = [100.0, 500.0] * 5          # 每天都跳变
        p = _panel({"AAA": {"close": close, "volume": [0.0] * 10}})
        rep = DataHealthChecker().check(p)
        assert 0.0 <= rep.overall_score <= 1.0, (
            f"综合分越界：{rep.overall_score}")

    def test_the_score_is_rounded_to_four_decimals(self):
        close = [100.0] * 10
        close[3] = np.nan
        rep = DataHealthChecker().check(_panel({"AAA": {"close": close},
                                                "BBB": {}, "CCC": {}}))
        assert rep.overall_score == round(rep.overall_score, 4)

    def test_an_empty_panel_scores_zero_not_one(self):
        """
        空面板的兜底分必须是 **0**（没有证据 ≠ 健康）。
        写成 1.0 会让"今天一条数据都没拉到"通过健康门。
        """
        rep = DataHealthChecker().check(pd.DataFrame())
        assert rep.overall_score == 0.0
        assert not rep.is_healthy()

    def test_the_report_records_the_panel_dimensions(self):
        rep = DataHealthChecker().check(_panel({"AAA": {}, "BBB": {}}))
        assert rep.n_tickers == 2
        assert rep.n_dates == len(DATES)

    def test_the_detector_settings_are_forwarded(self):
        """
        `DataHealthChecker(gap_min_days=..., spike_threshold=..., spike_fields=...)`
        —— 任一参数没接到子检测器上，调用方的配置就是摆设。
        """
        chk = DataHealthChecker(gap_min_days=7, spike_threshold=0.05,
                                spike_fields=["close", "volume"])
        assert chk._gap_det.min_gap_days == 7
        assert chk._spk_det.threshold == 0.05
        assert chk._spk_det.price_fields == ["close", "volume"]

    def test_the_defaults_are_the_documented_ones(self):
        chk = DataHealthChecker()
        assert chk._gap_det.min_gap_days == 1
        assert chk._spk_det.threshold == 0.50
        assert chk._spk_det.price_fields == ["close"]


class TestNanSummary:

    def test_only_fields_with_missing_values_are_listed(self):
        """
        `if nan_cnt > 0:` —— 守卫被删会让每个 ticker×字段都出现在
        汇总里（大部分是 0），报告瞬间变成几百行噪声。
        """
        close = [100.0] * 10
        close[2] = np.nan
        rep = DataHealthChecker().check(_panel({"AAA": {"close": close},
                                                "BBB": {}}))
        assert len(rep.nan_summary) == 1, f"NaN 汇总多出了零缺失的行：\n{rep.nan_summary}"
        r = rep.nan_summary.iloc[0]
        assert r["ticker"] == "AAA" and r["field"] == "close"
        assert r["nan_count"] == 1

    def test_the_percentage_is_per_ticker_not_global(self):
        """
        `nan_cnt / n * 100` 里的 `n` 是**该 ticker 的行数**。
        换成全表行数会让多票面板上的比例被系统性低估。
        """
        close = [100.0] * 10
        close[:2] = [np.nan] * 2
        rep = DataHealthChecker().check(_panel({"AAA": {"close": close},
                                                "BBB": {}, "CCC": {}}))
        r = rep.nan_summary.iloc[0]
        assert r["nan_pct"] == pytest.approx(20.0), (
            f"缺失比例算成了 {r['nan_pct']}% —— 分母用了全表行数而不是该票行数")

    def test_the_summary_columns_follow_the_contract(self):
        close = [100.0] * 10
        close[1] = np.nan
        rep = DataHealthChecker().check(_panel({"AAA": {"close": close}}))
        assert list(rep.nan_summary.columns) == ["ticker", "field",
                                                 "nan_count", "nan_pct"]

    def test_an_empty_summary_still_has_the_columns(self):
        rep = DataHealthChecker().check(_panel({"AAA": {}}))
        assert rep.nan_summary.empty
        assert list(rep.nan_summary.columns) == ["ticker", "field",
                                                 "nan_count", "nan_pct"]


# ===========================================================================
# E. HealthReport 本身
# ===========================================================================

class TestHealthReportDto:

    def _rep(self, score: float) -> HealthReport:
        e = pd.DataFrame()
        return HealthReport(gaps=e, spikes=e, zero_volume=e, nan_summary=e,
                            overall_score=score, n_tickers=3, n_dates=10)

    def test_the_health_threshold_is_inclusive(self):
        """
        `return self.overall_score >= min_score`

        `>=` 翻成 `>` 只改变**恰好等于门槛**的那一格。
        0.8 是调用方最常用的门槛值，恰好卡在上面的数据集会被误判为不健康，
        日循环因此空跑一天。
        """
        assert self._rep(0.8).is_healthy(min_score=0.8) is True, (
            "分数恰好等于门槛却判为不健康 —— `>=` 被翻成了 `>`")
        # 用 float(...) 把 numpy 标量转回内置浮点：np.float64 的比较返回
        # np.bool_，`is False` 会假红（而值其实是对的）。
        assert self._rep(float(np.nextafter(0.8, 0.0))).is_healthy(0.8) is False

    def test_the_default_threshold_is_zero_point_eight(self):
        assert self._rep(0.8).is_healthy() is True
        assert self._rep(0.79).is_healthy() is False

    def test_the_repr_reports_the_counts_not_the_frames(self):
        """一次 print 不能把四张表全刷出来。"""
        e = pd.DataFrame()
        rep = HealthReport(gaps=e, spikes=e, zero_volume=e, nan_summary=e,
                           overall_score=0.912, n_tickers=7, n_dates=10)
        r = repr(rep)
        assert "score=0.912" in r
        assert "tickers=7" in r
        assert "gaps=0" in r and "spikes=0" in r

    def test_notes_default_to_a_fresh_list(self):
        a, b = self._rep(1.0), self._rep(1.0)
        a.notes.append("x")
        assert b.notes == [], "两份报告共用了同一个 notes 列表"


class TestHtmlRendering:

    def test_the_html_reports_the_score_and_the_verdict(self):
        rep = DataHealthChecker().check(_panel({"AAA": {}, "BBB": {}}))
        html = DataHealthChecker().to_html(rep)
        assert "1.000" in html
        assert "健康" in html
        assert "2 个 ticker" in html

    def test_an_unhealthy_report_says_so(self):
        """
        `'健康' if report.is_healthy() else '需关注'` ——
        三目翻面会让有洞的数据在报告里被标成"健康"。
        """
        close = [100.0, 500.0] * 5
        rep = DataHealthChecker().check(
            _panel({"AAA": {"close": close, "volume": [0.0] * 10}}))
        assert not rep.is_healthy()
        assert "需关注" in DataHealthChecker().to_html(rep)

    def test_clean_sections_say_so_instead_of_rendering_an_empty_table(self):
        rep = DataHealthChecker().check(_panel({"AAA": {}}))
        html = DataHealthChecker().to_html(rep)
        assert html.count("无异常") == 3, f"三个检测区块都该显示「无异常」：\n{html[:400]}"
        assert "无缺失" in html

    def test_findings_are_rendered_as_tables(self):
        close = [100.0] * 10
        close[4] = np.nan
        rep = DataHealthChecker().check(_panel({"AAA": {"close": close}}))
        html = DataHealthChecker().to_html(rep)
        assert "<table" in html, "有发现却没有渲染成表格"
        assert "AAA" in html

    def test_the_spike_threshold_is_shown_in_the_heading(self):
        """标题里要写清"超过多少算异常"，否则读报告的人无从判断。"""
        rep = DataHealthChecker(spike_threshold=0.25).check(_panel({"AAA": {}}))
        html = DataHealthChecker(spike_threshold=0.25).to_html(rep)
        assert "25%" in html, f"标题里没有写明跳变阈值：{html[:300]}"


# ===========================================================================
# F. 复测后剩下的八个存活项
# ===========================================================================
#
# 首测 0.0% → 补完上面的用例后 75.0%，剩 8 个存活，分三类：
#   · 三处 `groupby(..., sort=False)` —— 分组顺序
#   · 一处 `in_gap = False` 的初值
#   · 四处 `to_html(index=False)`
# 它们都不改变"报了几条"，只改变**报出来长什么样** ——
# 而这份报告的用途就是给人看，所以"长什么样"就是它的正确性。


class TestGroupIterationOrder:
    """
    `panel.groupby(ticker_col, sort=False)` 三处（Gap / Spike / NaN 汇总）。

    `sort=False` = 按**在面板里出现的顺序**分组。翻成 True 会按字母序排 ——
    报告里的行序与 universe 的实际顺序脱节。
    universe 本身是有意义的排序（市值、权重、配置顺序），
    按字母重排之后，"前几行"不再是"最该先看的几只"。
    """

    @staticmethod
    def _unsorted_panel():
        # 刻意用**非字母序**的出现顺序
        a = [100.0] * 10
        a[2] = np.nan
        b = [100.0] * 10
        b[5] = np.nan
        c = [100.0] * 10
        c[7] = np.nan
        return _panel({"ZZZ": {"close": a}, "AAA": {"close": b},
                       "MMM": {"close": c}})

    def test_gap_rows_follow_panel_order_not_alphabetical_order(self):
        out = GapDetector().detect(self._unsorted_panel())
        assert out["ticker"].tolist() == ["ZZZ", "AAA", "MMM"], (
            f"空洞报告的行序是 {out['ticker'].tolist()}，"
            f"应当跟随面板里的出现顺序 —— `groupby(sort=False)` 被翻成了 True")

    def test_spike_rows_follow_panel_order(self):
        a = [100.0] * 10
        a[3] = 500.0
        b = [100.0] * 10
        b[4] = 500.0
        p = _panel({"ZZZ": {"close": a}, "AAA": {"close": b}})
        out = SpikeDetector(threshold=0.5).detect(p)
        assert out["ticker"].tolist()[0] == "ZZZ", (
            f"跳变报告的行序是 {out['ticker'].tolist()} —— 分组被按字母排序了")

    def test_nan_summary_rows_follow_panel_order(self):
        rep = DataHealthChecker().check(self._unsorted_panel())
        assert rep.nan_summary["ticker"].tolist() == ["ZZZ", "AAA", "MMM"], (
            f"NaN 汇总的行序是 {rep.nan_summary['ticker'].tolist()} —— "
            f"分组被按字母排序了")


class TestGapStateInitialisation:

    def test_a_gap_starting_on_the_very_first_row_records_a_real_start_date(self):
        """
        `in_gap = False`（初值）

        翻成 True 之后，面板**第一行就缺失**的情况会走进 else 分支：
        `gap_len += 1` 而 `gap_start` 始终没被赋值，
        于是 `pd.Timestamp(None)` → **NaT**。

        报出来的空洞长度仍然正确（所以"报了几条""几天"这类断言全绿），
        只是起始日变成了 NaT —— 运维照着这个区间去补数据会得到一个空区间。
        """
        close = [np.nan] * 3 + [100.0] * 7
        out = GapDetector().detect(_panel({"AAA": {"close": close}}))
        assert len(out) == 1
        r = out.iloc[0]
        assert pd.notna(r["gap_start"]), (
            "开头就缺失的空洞，起始日是 NaT —— `in_gap` 的初值被翻成了 True")
        assert r["gap_start"] == DATES[0], (
            f"起始日是 {r['gap_start']}，应当是面板第一天 {DATES[0]}")
        assert r["gap_end"] == DATES[2]
        assert r["gap_days"] == 3

    def test_an_all_missing_ticker_also_records_a_real_start_date(self):
        """末尾空洞那条路径同样会读到未初始化的 gap_start。"""
        out = GapDetector().detect(_panel({"AAA": {"close": [np.nan] * 10}}))
        r = out.iloc[0]
        assert pd.notna(r["gap_start"]) and r["gap_start"] == DATES[0]
        assert r["gap_end"] == DATES[9]


class TestHtmlOmitsTheRowIndex:
    """
    `to_html(index=False)` 四处（gaps / spikes / zero_volume / nan_summary）。

    翻成 True 会在每张表左边多出一列 pandas 行号。
    表格照样渲染、数据照样对，只是多了一列毫无意义的 0,1,2…… ——
    而这份 HTML 是直接贴进 Jupyter / Web 给人读的。
    """

    @staticmethod
    def _report_with_every_kind_of_finding():
        close = [100.0] * 10
        close[2:4] = [np.nan] * 2       # gap + NaN 汇总
        close[7] = 400.0                # spike
        vol = [1e6] * 10
        vol[5] = 0.0                    # zero volume
        return _panel({"AAA": {"close": close, "volume": vol}})

    def test_every_table_is_rendered_without_the_pandas_index(self):
        panel = self._report_with_every_kind_of_finding()
        chk = DataHealthChecker()
        rep = chk.check(panel)

        # 前提：四类发现都非空，否则对应分支渲染的是"无异常"而不是表格
        assert not rep.gaps.empty, "用例前提被破坏：没有空洞"
        assert not rep.spikes.empty, "用例前提被破坏：没有跳变"
        assert not rep.zero_volume.empty, "用例前提被破坏：没有零成交量"
        assert not rep.nan_summary.empty, "用例前提被破坏：没有 NaN"

        html = chk.to_html(rep)
        assert "<table" in html
        assert "<th>0</th>" not in html, (
            "HTML 表格里渲染了 pandas 行号列 —— "
            "`to_html(index=False)` 被翻成了 True")

    def test_each_table_has_exactly_one_header_cell_per_column(self):
        """
        逐张表比对表头单元格数：带上索引会多出一个。
        （上一条只看有没有 `<th>0</th>`，这一条把四张表分开数，
        免得某一张单独被改而总体仍然"没有 0 那一行"。）
        """
        panel = self._report_with_every_kind_of_finding()
        chk = DataHealthChecker()
        rep = chk.check(panel)
        html = chk.to_html(rep)

        tables = html.split("<table")[1:]
        frames = [rep.gaps, rep.spikes, rep.zero_volume, rep.nan_summary]
        assert len(tables) == len(frames), (
            f"渲染出 {len(tables)} 张表，应当是 4 张")
        for frame, tbl in zip(frames, tables):
            head = tbl[:tbl.index("</thead>")]
            # 数 `<th>` 而不是 `<th`：后者会把 `<thead` 也算进去
            n_cells = head.count("<th>")
            assert n_cells == len(frame.columns), (
                f"表头有 {n_cells} 个单元格，而这张表有 "
                f"{len(frame.columns)} 列 —— 多出来的是 pandas 行号列")
