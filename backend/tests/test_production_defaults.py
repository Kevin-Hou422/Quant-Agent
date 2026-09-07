"""
test_production_defaults.py — 在**发布配置**下验证门控是否真的会拦截

为什么单独一个文件：conftest 的 `_hermetic_run_flags` 覆盖了 settings（别打网络、
别起调度器），代价是**没有任何测试跑在实际发布的配置下**。于是"函数写对了但默认
值没打开"这一整类 bug 在测试体系里结构性不可见 —— 门控函数单测全绿，线上却一个
都不拦。本文件不覆盖门控开关，只读 Settings 的**类默认值**（即用户装好就跑的值），
断言"配置 → 行为"这条链是通的。

标记：production_defaults
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fresh_settings():
    """未经 conftest 覆盖的 Settings 类默认值 —— 即用户实际拿到的配置。"""
    from app.config import Settings
    return Settings.model_construct(**{
        k: v.default for k, v in Settings.model_fields.items()
        if v.default is not None
    })


# ---------------------------------------------------------------------------
# 1. 门控开关的默认状态必须被明确记录（改了就得有人知道）
# ---------------------------------------------------------------------------

#: 发布默认值快照。**故意写死**：任何一处变动都必须由人显式修改此表，
#: 从而在 review 里被看见 —— 而不是悄悄从"不拦"变成"拦"或反过来。
EXPECTED_GATE_DEFAULTS = {
    "pm_strategy_gate_block":  False,   # 策略门未过 → 仍交易（paper 期收前向证据）
    "risk_halt_on_drawdown":   False,   # 回撤熔断 → 仅告警，不清仓
    "tr_enforce_active_gate":  False,   # →ACTIVE 门未过 → 仍可激活
    "tr_experiment_mode":      True,    # 实验模式：B/C 级也放进 paper
    "risk_target_vol_ann":     0.0,     # 0 = 关闭 vol targeting
}


class TestGateDefaultsAreExplicit:

    def test_gate_defaults_match_documented_snapshot(self, fresh_settings):
        drift = {
            k: (getattr(fresh_settings, k), v)
            for k, v in EXPECTED_GATE_DEFAULTS.items()
            if getattr(fresh_settings, k) != v
        }
        assert not drift, (
            "门控默认值与本文件记录的快照不符（实际, 期望）：\n  "
            + "\n  ".join(f"{k}: {a!r} vs {b!r}" for k, (a, b) in drift.items())
            + "\n若这是有意变更，请同步更新 EXPECTED_GATE_DEFAULTS 并在 roadmap 说明。"
        )

    def test_all_hard_gates_are_off_by_default_is_a_conscious_choice(self, fresh_settings):
        """
        当前发布配置下，**没有任何一个硬门是开的**。这在 paper 取证期是刻意选择，
        但必须显式暴露出来 —— 否则用户会以为 "passed/failed/risk halt" 是真拦截。
        本用例不判对错，只保证这个事实不会悄无声息。
        """
        hard_gates = ["pm_strategy_gate_block", "risk_halt_on_drawdown",
                      "tr_enforce_active_gate"]
        enabled = [g for g in hard_gates if getattr(fresh_settings, g)]
        assert not enabled or len(enabled) == len(hard_gates), (
            f"硬门处于**部分开启**状态：{enabled} —— 半开的门最危险"
            f"（一部分结论被拦，另一部分照常放行，使用者无法判断哪个是哪个）"
        )


# ---------------------------------------------------------------------------
# 2. 配置 → 行为：开关打开时必须真的拦住（否则开关是装饰品）
# ---------------------------------------------------------------------------

def _panel(n_days=120, n_tickers=8, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n_days)
    cols = [f"T{i:02d}" for i in range(n_tickers)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.012, (n_days, n_tickers)), axis=0),
        index=idx, columns=cols)
    return close, idx, cols


class TestSwitchesActuallyBite:

    def test_active_gate_blocks_when_enabled(self):
        """tr_enforce_active_gate=True 且前向样本不足 → 必须拒绝激活。"""
        from app.core.lifecycle.promotion_gate import check_active_promotion
        ok, detail = check_active_promotion([])          # 零前向样本
        assert not ok, f"零前向 IC 样本却通过 →ACTIVE 门：{detail}"
        assert detail.get("reasons"), "门未过却没有给出理由"

    def test_drawdown_halt_detects_breach(self):
        """回撤熔断判定本身必须有效（与 halt 开关是否打开无关）。"""
        from app.core.portfolio_manager import PortfolioRiskGate, RiskLimits
        eq = pd.Series([1.0, 1.1, 1.2, 0.8, 0.75])        # 峰值 1.2 → 0.75，回撤 37.5%
        halt, dd = PortfolioRiskGate(RiskLimits(max_drawdown=0.20)).should_halt(eq)
        assert halt, f"37.5% 回撤未触发熔断判定（dd={dd}）"
        assert dd > 0.20

    def test_vol_targeting_actually_scales_when_wired(self):
        """
        目标波动是**配置 → 行为**链的典型：只有 apply() 收到 port_vol_ann 才会缩放。
        本用例直接传参，证明能力存在；实线路径是否传值由
        test_lessons_enforced.py::test_critical_parameters_are_actually_passed 把关。
        """
        from app.core.portfolio_manager import PortfolioRiskGate, RiskLimits
        _, idx, cols = _panel(n_days=10, n_tickers=4)
        w = pd.DataFrame(0.25, index=idx, columns=cols)
        lim = RiskLimits(target_vol_ann=0.10, max_gross=1.0, max_name_weight=1.0,
                         max_sector_weight=1.0)
        _, rep_off = PortfolioRiskGate(lim).apply(w)                        # 不传实际波动
        _, rep_on  = PortfolioRiskGate(lim).apply(w, port_vol_ann=0.20)     # 实际 20%
        assert rep_off.vol_scalar == 1.0, "未提供波动估计时不应缩放"
        assert abs(rep_on.vol_scalar - 0.5) < 1e-9, (
            f"目标 10% / 实际 20% 应缩放到 0.5，实际 {rep_on.vol_scalar}"
        )


# ---------------------------------------------------------------------------
# 3. 发布配置下的数据契约
# ---------------------------------------------------------------------------

class TestProductionDataContract:

    def test_chat_defaults_to_real_dataset_not_synthetic(self, fresh_settings):
        """发布配置下聊天必须走真实数据集（合成需显式 opt-in）。"""
        assert fresh_settings.chat_allow_synthetic is False, (
            "发布默认允许合成数据 —— 用户屏幕上的 Sharpe 会是随机噪声且无从分辨"
        )
        assert fresh_settings.default_dataset, "发布配置没有默认数据集名"

    def test_live_db_not_in_cloud_sync_by_default(self, fresh_settings):
        for attr in ("database_url", "pit_store_dir"):
            val = str(getattr(fresh_settings, attr, "") or "").replace("\\", "/").lower()
            assert not any(k in val for k in ("onedrive", "dropbox", "icloud")), (
                f"发布默认把活库放在云同步目录：{attr}={val}"
            )


# ---------------------------------------------------------------------------
# 审计 #10：零认证服务的暴露面（本机自用形态）
# ---------------------------------------------------------------------------

class TestNoAuthServiceIsNotExposed:
    """
    本服务**没有任何认证**：策略审批/拒绝/状态变更/删会话/跑 GP 全部裸奔。
    当前形态是"只在本机跑"，因此正确的防线不是加密码，而是
    **保证它不会在无人察觉的情况下被绑到对外地址**。
    """

    def test_bind_host_defaults_to_loopback(self, fresh_settings):
        assert fresh_settings.api_bind_host in ("127.0.0.1", "localhost", "::1"), (
            f"默认绑定 {fresh_settings.api_bind_host!r} 不是回环地址 —— "
            f"零认证服务会对整个网络开放"
        )
        assert fresh_settings.allow_insecure_bind is False

    def test_cors_is_not_wildcard_by_default(self, fresh_settings):
        assert "*" not in fresh_settings.cors_origins, (
            "CORS 默认允许任意来源，且 allow_credentials=True —— 危险组合"
        )
        assert fresh_settings.cors_origins, "CORS 白名单为空会让前端无法访问"

    def test_startup_refuses_non_loopback_bind(self, monkeypatch):
        """绑非回环地址时必须**拒绝启动**，除非显式解除保险。"""
        from app.config import settings as live
        import app.main as m
        monkeypatch.setattr(live, "api_bind_host", "0.0.0.0", raising=False)
        monkeypatch.setattr(live, "allow_insecure_bind", False, raising=False)
        with pytest.raises(RuntimeError, match="零认证|回环"):
            m._assert_safe_bind()

    def test_explicit_optin_allows_exposure(self, monkeypatch):
        """显式 allow_insecure_bind=true 才放行（保留逃生口，但必须是有意为之）。"""
        from app.config import settings as live
        import app.main as m
        monkeypatch.setattr(live, "api_bind_host", "0.0.0.0", raising=False)
        monkeypatch.setattr(live, "allow_insecure_bind", True, raising=False)
        m._assert_safe_bind()          # 不抛即通过
