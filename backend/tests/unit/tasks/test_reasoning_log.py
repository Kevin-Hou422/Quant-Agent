"""
tasks/reasoning_log.py —— Agent 推理过程的结构化记录

**此前零测试**（75 有效行，D 档）。

这份日志序列化成 JSON 写进 `AlphaStore.reasoning` 列，是事后回答
"这条因子当初是怎么一步步改出来的"的**唯一**依据。
它坏掉不会有任何运行时症状 —— 只是若干个月后想复盘时发现
步号乱了、中文变成了转义串、或者反序列化直接抛。

三处要害：
  - `step=len(self.changes) + 1` —— 步号，错位会让谱系对不上
  - `to_json(..., ensure_ascii=False)` —— 中文理由不能变成 \\uXXXX
  - `from_json` 的 round-trip —— 存进去能读回来，且 changes 是
    `ChangeEntry` 对象而不是裸 dict
"""
from __future__ import annotations

import json

import pytest

from app.tasks.reasoning_log import ChangeEntry, ReasoningLog


def _log(**over) -> ReasoningLog:
    base = dict(hypothesis="动量在科技股上更强",
                initial_dsls=["rank(ts_delta(close,5))", "rank(close)"])
    base.update(over)
    return ReasoningLog(**base)


# ===========================================================================
# A. 步号
# ===========================================================================

class TestStepNumbering:

    def test_steps_start_at_one_and_increment(self):
        """
        `step=len(self.changes) + 1`

        `+` 改成 `-` 会让第一步变成 -1、第二步 0；改成 `*` 让所有步都是 0。
        步号是谱系里定位"第几次改动"的键，错位之后
        `changes[i].step` 与列表下标对不上，复盘时读到的是别人的理由。
        """
        log = _log()
        for i in range(4):
            log.add_change(f"old{i}", f"new{i}", f"理由{i}")
        assert [c.step for c in log.changes] == [1, 2, 3, 4], (
            f"步号序列是 {[c.step for c in log.changes]}，应当是 1..4 —— "
            f"`len(self.changes) + 1` 的算符被改了")

    def test_the_first_change_is_step_one_not_zero(self):
        """人读的编号从 1 开始 —— 0 会让"第 0 步"这种说法出现在报告里。"""
        log = _log()
        log.add_change("a", "b", "r")
        assert log.changes[0].step == 1

    def test_each_change_records_both_sides_of_the_edit(self):
        """
        old_dsl / new_dsl 必须**都**记下来。只记新的话，
        复盘时无法还原"从什么改成了什么"。
        """
        log = _log()
        log.add_change("rank(close)", "rank(ts_mean(close,5))", "降噪")
        c = log.changes[0]
        assert c.old_dsl == "rank(close)"
        assert c.new_dsl == "rank(ts_mean(close,5))"
        assert c.reason == "降噪"

    def test_metrics_default_to_an_empty_dict_not_none(self):
        """
        `metrics=metrics or {}` —— 传 None 时要变成 `{}`。
        留 None 会让 `asdict` 出来是 null，反序列化后
        `entry.metrics.get(...)` 直接 AttributeError。
        """
        log = _log()
        log.add_change("a", "b", "r", metrics=None)
        assert log.changes[0].metrics == {}, (
            f"metrics 是 {log.changes[0].metrics!r}，应当是空 dict")

    def test_supplied_metrics_are_kept(self):
        """`or {}` 的另一侧：传了真实指标就不能被空 dict 覆盖。"""
        log = _log()
        log.add_change("a", "b", "r", metrics={"is_sharpe": 1.23})
        assert log.changes[0].metrics == {"is_sharpe": 1.23}

    def test_an_empty_metrics_dict_stays_empty(self):
        """`{} or {}` → `{}`，行为上无差别，但确认不会变成 None。"""
        log = _log()
        log.add_change("a", "b", "r", metrics={})
        assert log.changes[0].metrics == {}


# ===========================================================================
# B. 序列化
# ===========================================================================

class TestSerialisation:

    def test_chinese_reasons_stay_readable_in_the_json(self):
        """
        `json.dumps(data, ensure_ascii=False, indent=2)`

        `ensure_ascii` 翻成 True 会把"动量在科技股上更强"变成
        `\\u52a8\\u91cf...`。这串东西要存进数据库、要在复盘时被人读，
        转义之后完全不可读，而且没有任何报错。
        """
        log = _log()
        log.add_change("a", "b", "换手太高，加了 3 日平滑")
        raw = log.to_json()
        assert "\\u" not in raw, (
            f"JSON 里出现了 \\uXXXX 转义 —— ensure_ascii 被翻成了 True：{raw[:200]}")
        assert "换手太高" in raw
        assert "动量在科技股上更强" in raw

    def test_the_json_is_indented_for_human_reading(self):
        """
        `indent=2` —— 这列是给人读的。压成一行之后，
        一份十几步的推理日志是一条几千字符的长串。
        """
        raw = _log().to_json()
        assert "\n" in raw, "JSON 被压成了一行 —— indent 参数被去掉了"

    def test_every_field_survives_a_round_trip(self):
        """
        `from_json` → `to_json` 的往返。任何字段丢失都意味着
        那部分谱系永久消失（数据库里存的就是这个串）。
        """
        log = _log(final_dsl="rank(ts_mean(close,5))",
                   final_metrics={"is_sharpe": 1.5, "oos_sharpe": 0.9})
        log.add_change("rank(close)", "rank(ts_mean(close,5))", "降噪",
                       metrics={"turnover": 2.1})

        back = ReasoningLog.from_json(log.to_json())
        assert back.hypothesis == log.hypothesis
        assert back.initial_dsls == log.initial_dsls
        assert back.final_dsl == log.final_dsl
        assert back.final_metrics == log.final_metrics
        assert back.created_at == log.created_at

    def test_changes_come_back_as_objects_not_dicts(self):
        """
        `changes = [ChangeEntry(**c) for c in data.pop("changes", [])]`

        少了这一步，`back.changes[0]` 是裸 dict，
        调用方写的 `c.step` 会 AttributeError ——
        而这个错只在有人真去复盘时才暴露。
        """
        log = _log()
        log.add_change("a", "b", "理由", metrics={"x": 1.0})
        back = ReasoningLog.from_json(log.to_json())

        assert back.changes, "往返之后 changes 空了"
        c = back.changes[0]
        assert isinstance(c, ChangeEntry), (
            f"changes 回来的是 {type(c).__name__}，应当是 ChangeEntry —— "
            f"复盘代码里的 `c.step` 会 AttributeError")
        assert c.step == 1 and c.old_dsl == "a" and c.new_dsl == "b"
        assert c.reason == "理由" and c.metrics == {"x": 1.0}

    def test_changes_are_popped_before_constructing_the_log(self):
        """
        `data.pop("changes", [])` —— 必须 **pop**（而不是 get）。
        用 get 的话 `cls(**data)` 会收到一个 `changes=[dict,...]`，
        把裸 dict 直接塞进 changes 字段，上面那条断言就会红。
        """
        log = _log()
        log.add_change("a", "b", "r")
        back = ReasoningLog.from_json(log.to_json())
        assert all(isinstance(c, ChangeEntry) for c in back.changes)

    def test_a_log_without_changes_round_trips(self):
        """
        `data.pop("changes", [])` 的默认值 —— 没有 changes 键时给空列表。
        默认值被改成 None 会让 `for c in None` 直接抛。
        """
        back = ReasoningLog.from_json(_log().to_json())
        assert back.changes == []

    def test_the_json_is_valid_and_carries_the_documented_keys(self):
        log = _log(final_dsl="rank(close)")
        log.add_change("a", "b", "r")
        data = json.loads(log.to_json())
        for k in ("hypothesis", "initial_dsls", "changes", "final_dsl",
                  "final_metrics", "created_at"):
            assert k in data, f"序列化结果缺键 {k}：{sorted(data)}"
        for k in ("step", "old_dsl", "new_dsl", "reason", "metrics", "timestamp"):
            assert k in data["changes"][0], (
                f"变更记录缺键 {k}：{sorted(data['changes'][0])}")


# ===========================================================================
# C. 摘要
# ===========================================================================

class TestSummary:

    def test_the_summary_reports_the_actual_counts(self):
        """
        `f"Initial DSLs: {len(self.initial_dsls)}"` /
        `f"Changes    : {len(self.changes)}"`
        —— 数字必须是真实长度。写死或算错会让摘要与明细对不上。
        """
        log = _log(initial_dsls=["a", "b", "c"])
        for i in range(2):
            log.add_change(f"o{i}", f"n{i}", "r")
        s = log.summary()
        assert "Initial DSLs: 3" in s, f"初始 DSL 计数不对：{s}"
        assert "Changes    : 2" in s, f"变更计数不对：{s}"

    def test_the_summary_carries_the_hypothesis_and_final_dsl(self):
        log = _log(final_dsl="rank(ts_mean(close,5))")
        s = log.summary()
        assert "动量在科技股上更强" in s
        assert "rank(ts_mean(close,5))" in s

    def test_metrics_are_formatted_to_four_decimals(self):
        """
        `f"  {k}: {v:.4f}"` —— 位数被改会让摘要里的 Sharpe
        要么精度不足（1.2 看不出与 1.23 的差别），要么刷一长串小数。
        """
        log = _log(final_metrics={"is_sharpe": 1.23456})
        assert "1.2346" in log.summary(), (
            f"指标没有按 4 位小数格式化：{log.summary()}")

    def test_the_metrics_block_is_omitted_when_empty(self):
        """
        `if self.final_metrics:` —— 删掉守卫会在没有指标时
        打出一个空的指标区块（或者 `for k, v in {}` 什么都不打，
        但守卫本身是契约的一部分）。
        """
        s = _log().summary()
        assert s.count("\n") == 3, (
            f"没有指标时摘要多出了行：\n{s}")


# ===========================================================================
# D. 时间戳
# ===========================================================================

class TestTimestamps:

    def test_each_change_gets_its_own_timestamp(self):
        """
        `timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())`
        —— 必须是 `default_factory`。写成 `default=datetime.utcnow().isoformat()`
        会让**所有**记录共用导入时刻的那一个字符串，
        整份谱系的时间信息全部作废。
        """
        import time
        log = _log()
        log.add_change("a", "b", "r")
        time.sleep(0.002)
        log.add_change("b", "c", "r")
        stamps = [c.timestamp for c in log.changes]
        assert len(set(stamps)) == len(stamps), (
            f"两条变更拿到了同一个时间戳：{stamps} —— "
            f"`default_factory` 疑似被写成了 `default`")

    def test_the_log_records_its_creation_time(self):
        log = _log()
        assert log.created_at, "created_at 为空"
        # ISO 格式，能被解析回来
        from datetime import datetime
        datetime.fromisoformat(log.created_at)

    def test_two_logs_created_apart_have_different_stamps(self):
        import time
        a = _log()
        time.sleep(0.002)
        b = _log()
        assert a.created_at != b.created_at, (
            "两份先后创建的日志拿到了同一个 created_at —— "
            "`default_factory` 疑似被写成了 `default`")
