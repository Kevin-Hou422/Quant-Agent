"""
目标波动关闭的上线预检：运行时没关或运行中规格还开着 → 红；新规格记下关闭，旧规格与运行时对不上。

来源（原测试文件的说明，保留出处）：

2026-10-22 用户决定 · 批 3：证据 / 研究纪律。

D12 原话："同意先只读查询，不授权写边界。使用明确的活库路径和真正只读连接，不调用会建表或迁移的 Store 构造函数。
列出已有边界、最早曝光日期、次数、股票池和实验身份，给出日期建议后由我确认。正式入口不要继续依赖首次调用偶然冻结；
改实验名不能刷新重叠数据的曝光额度。"
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# N05-c："首轮关闭目标波动。由我手工调整活 .env；你提供预检和配置说明，开发不必因此停下。
#         该决定须体现在新规格中，不能偷偷改变已批准策略。"
# ---------------------------------------------------------------------------

class TestTargetVolOff:

    @staticmethod
    def _pre(tmp_path, **kw):
        from datetime import datetime, timezone

        from app.core.execution.golive import run_preflight
        from tests.unit.execution.test_golive import JOBS, _settings
        rep = run_preflight(_settings(tmp_path, **kw), probe=lambda h, p: False,
                            now=datetime(2025, 3, 4, 22, 0, tzinfo=timezone.utc), cwd=tmp_path,
                            data_probe=lambda st, t: (True, ""), jobs=lambda: JOBS,
                            strategy_check=lambda st: (True, ""), fundamentals_check=lambda st: (True, "", True))
        return {c.name: c for c in rep.checks}["target_vol_off"]

    def test_the_preflight_is_red_while_target_vol_is_still_on(self, tmp_path):
        c = self._pre(tmp_path, risk_target_vol_ann=0.12)
        assert c.ok is False and c.blocking is True and "RISK_TARGET_VOL_ANN" in c.detail

    def test_the_preflight_is_green_once_it_is_off_and_no_running_spec_says_otherwise(self, tmp_path):
        assert self._pre(tmp_path, risk_target_vol_ann=0.0).ok is True

    def test_a_running_strategy_whose_spec_still_targets_vol_must_be_re_proposed(self, tmp_path):
        """改了 .env 不会悄悄改变已批准策略：规格里还开着目标波动 → 预检红，要求重新提案。"""
        from app.core.portfolio_manager.executable_spec import build_spec, spec_hash
        from app.db.strategy_store import StrategyConfig, StrategyStore
        from tests.unit.execution.test_golive import _settings
        s_on = _settings(tmp_path, risk_target_vol_ann=0.12)
        spec = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e6, 0.0, settings=s_on)
        assert spec["risk_limits"]["target_vol_ann"] == pytest.approx(0.12)
        ss = StrategyStore(db_url=s_on.database_url)
        sid = ss.save(StrategyConfig(factors=["1"], combo_weights={"1": 1.0}, aum=1e6, status="proposed",
                                     verdict={"executable_spec": spec, "spec_hash": spec_hash(spec)}))
        ss.update_status(sid, "approved"); ss.update_status(sid, "paper_running")
        c = self._pre(tmp_path, risk_target_vol_ann=0.0)
        assert c.ok is False and "重新提案" in c.detail and f"#{sid}" in c.detail

    def test_the_new_spec_records_the_decision_and_old_specs_stop_matching(self, tmp_path):
        from app.core.portfolio_manager.executable_spec import (
            build_spec,
            verify_against_runtime,
        )
        from tests.unit.execution.test_golive import _settings
        s_on, s_off = _settings(tmp_path, risk_target_vol_ann=0.12), _settings(tmp_path, risk_target_vol_ann=0.0)
        old = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e6, 0.0, settings=s_on)
        new = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e6, 0.0, settings=s_off)
        assert new["risk_limits"]["target_vol_ann"] is None, "新规格里看不出目标波动已关闭"
        diff = verify_against_runtime(old, {"1": "rank(close)"}, 1e6, settings=s_off)
        assert any("target_vol" in d for d in diff), f"旧规格在关闭目标波动后仍被当成一致：{diff}"
