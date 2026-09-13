"""
config.py —— 全部布尔默认值的定钉测试（变异测试驱动）

来由：17 个变异点，首测击杀率 **35.3%**（存活 11）—— 存活的**全部是布尔默认值**。

这些默认值就是"用户装好、什么都不改"时系统的实际行为。它们被改掉不会让任何
功能测试变红：函数都还对，只是**开关反了**。已有的
`test_production_defaults.py::EXPECTED_GATE_DEFAULTS` 已经用快照的方式钉住了
五个门控开关，但 Settings 里还有十一个布尔字段不在那张表里，于是：

  - `enable_scheduler: bool = False` → True：装好就**自动开始定时交易**
  - `enable_paper_trading: bool = False` → True：同上，未经确认就开始下模拟单
  - `trading_allow_short: bool = False` → True：long-only 的前提被推翻，
    可以裸空，而当前成本模型与券商权限都没按做空建模
  - `research_health_fail_closed: bool = True` → False：数据健康检查从
    **拒绝**变成**放行**，坏数据直接进研究管线
  - `calendar_allow_heuristic: bool = False` → True：交易日历失败时用猜的，
    fail-closed 的整条设计被架空
  - `enable_backup: bool = True` → False：备份静默关闭，出事没有回滚点
  - `debug: bool = False` → True：调试模式上线
  - `pm_marginal_selection` / `pm_strategy_gate_eval` → False：
    组合构建换算法、策略门不再评估
  - `enable_discovery: bool = False` → True：夜间自动发现任务被打开
  - `case_sensitive=False` → True：**环境变量名的大小写规则变了**，
    用户 .env 里写 `ENABLE_SCHEDULER=` 的那一行会被静默忽略，
    配置看着改了其实没生效

本文件用"全量快照"的方式钉：**Settings 里每一个 bool 字段都必须在表里**，
新增字段忘了登记会直接红。
"""
from __future__ import annotations

import pytest

from app.config import Settings


#: 发布默认值的完整快照（bool 字段一个不落）。
#: 故意写死：任何一处变动都必须由人显式改这张表，从而在 review 里被看见。
EXPECTED_BOOL_DEFAULTS = {
    # ── 运行时开关：装好就跑的东西，默认必须是关的 ──────────────────
    "enable_scheduler":            False,   # 不自动起定时任务
    "enable_paper_trading":        False,   # 不自动开始模拟交易
    "enable_discovery":            False,   # 不自动起夜间发现任务
    "debug":                       False,   # 不在调试模式下发布
    "enable_backup":               True,    # 备份默认开着

    # ── 安全 ────────────────────────────────────────────────────────
    "allow_insecure_bind":         False,   # 不对外暴露未鉴权服务
    "chat_allow_synthetic":        False,   # 不拿合成数据冒充真实数据

    # ── 交易约束 ────────────────────────────────────────────────────
    "trading_allow_short":         False,   # 暂 long-only

    # ── 数据契约：fail-closed 一侧默认必须是"严" ───────────────────
    "research_health_fail_closed": True,    # 健康检查不过 → 拒绝
    "calendar_allow_heuristic":    False,   # 日历拿不到 → 报错，不猜

    # ── 组合构建 ────────────────────────────────────────────────────
    "pm_marginal_selection":       True,    # PM.S2 按边际贡献贪心选
    "pm_strategy_gate_eval":       True,    # PM.S1 每次评估并记录 verdict

    # ── 门控（与 test_production_defaults.EXPECTED_GATE_DEFAULTS 重叠，
    #     两处都留：那边测"门控语义"，这里测"bool 字段一个不落"）────
    "pm_strategy_gate_block":      False,
    "tr_experiment_mode":          True,
    "tr_enforce_active_gate":      False,
    "risk_halt_on_drawdown":       False,
}


@pytest.fixture
def defaults() -> dict:
    """
    Settings 的**类默认值** —— 即用户装好拿到的配置。
    不能直接实例化：那会读 .env 与环境变量，测的就不是默认值了。
    """
    return {k: v.default for k, v in Settings.model_fields.items()}


def _bool_fields() -> dict:
    return {k: v.default for k, v in Settings.model_fields.items()
            if v.annotation is bool}


class TestBooleanDefaults:

    def test_every_boolean_field_is_covered_by_the_snapshot(self):
        """新增 bool 配置却忘了登记 → 这里红。快照的价值全靠这条维持。"""
        actual = set(_bool_fields())
        expected = set(EXPECTED_BOOL_DEFAULTS)
        missing = actual - expected
        stale = expected - actual
        assert not missing, (
            f"以下 bool 配置没有登记进 EXPECTED_BOOL_DEFAULTS：{sorted(missing)}\n"
            f"新增开关必须显式记录它的发布默认值。")
        assert not stale, (
            f"以下字段已不存在或不再是 bool，请从快照里删掉：{sorted(stale)}")

    def test_all_boolean_defaults_match_the_snapshot(self):
        drift = {k: (v, EXPECTED_BOOL_DEFAULTS[k])
                 for k, v in _bool_fields().items()
                 if v is not EXPECTED_BOOL_DEFAULTS[k]}
        assert not drift, (
            "布尔默认值与快照不符（实际, 期望）：\n  "
            + "\n  ".join(f"{k}: {a!r} vs {b!r}" for k, (a, b) in drift.items())
            + "\n若这是有意变更，请同步改本文件并在 roadmap 里说明原因。")

    @pytest.mark.parametrize("field", [
        "enable_scheduler", "enable_paper_trading", "enable_discovery", "debug",
    ])
    def test_nothing_starts_running_on_a_fresh_install(self, field):
        """
        这四个各自能让"装好就自动跑起来"。分开写是为了**红的时候一眼看出是哪个** ——
        合在一张表里时，失败信息要在一堆字段里找。
        """
        assert _bool_fields()[field] is False, (
            f"{field} 默认开启 —— 用户装好什么都没配就会自动开始跑")

    @pytest.mark.parametrize("field", [
        "research_health_fail_closed",
    ])
    def test_fail_closed_contracts_default_to_strict(self, field):
        assert _bool_fields()[field] is True, (
            f"{field} 默认变成了放行 —— fail-closed 的设计被架空")

    def test_calendar_never_guesses_by_default(self):
        assert _bool_fields()["calendar_allow_heuristic"] is False, (
            "交易日历默认允许启发式猜测 —— 拿不到日历时会编一个出来")

    def test_short_selling_is_off_until_explicitly_enabled(self):
        assert _bool_fields()["trading_allow_short"] is False, (
            "默认允许做空 —— 当前成本模型与券商权限都没按做空建模")

    def test_backup_is_on_by_default(self):
        assert _bool_fields()["enable_backup"] is True, (
            "备份默认关闭 —— 出事没有回滚点")


class TestEnvLoadingContract:

    def test_env_var_names_are_case_insensitive(self):
        """
        `case_sensitive=False` 改成 True 之后，pydantic-settings 只认**与字段名
        完全同形**的环境变量。用户 .env 里习惯写的 `ENABLE_SCHEDULER=true`
        会被静默忽略 —— 配置看着改了，其实一行都没生效，且没有任何报错。
        """
        assert Settings.model_config.get("case_sensitive") is False, (
            "环境变量名变成大小写敏感 —— .env 里的大写写法会被静默忽略")

    def test_dotenv_is_the_configured_source(self):
        cfg = Settings.model_config
        assert cfg.get("env_file") == ".env"
        assert cfg.get("env_file_encoding") == "utf-8", (
            "env 文件编码不是 utf-8 —— 中文注释/取值会读乱")

    def test_unknown_env_keys_are_ignored_not_fatal(self):
        assert Settings.model_config.get("extra") == "ignore", (
            "未知配置项从忽略变成报错 —— 用户 .env 里多一行就起不来")

    def test_uppercase_env_var_actually_overrides(self, monkeypatch):
        """契约的行为侧验证：大写环境变量必须真的能覆盖默认值。"""
        monkeypatch.setenv("ENABLE_SCHEDULER", "true")
        assert Settings(_env_file=None).enable_scheduler is True, (
            "大写环境变量没有覆盖默认值 —— case_sensitive 的取值有问题")
