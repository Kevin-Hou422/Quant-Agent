"""
config.py — Quant Agent 全局配置

通过 pydantic-settings 读取环境变量（或 .env 文件）。
所有组件通过 `from app.config import settings` 引用唯一实例。
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── 数据库 ────────────────────────────────────────────────────────────
    # ⚠️ 活库**绝不能放在云同步目录**（OneDrive/Dropbox…）：WAL 模式下
    # .db/.db-wal/.db-shm 三件套必须互相一致，云盘会分别异步上传 → 可能同步出损坏副本，
    # 上传时占用文件还会引发 database is locked。活数据放本地非同步盘，
    # 备份用 backup.py 的**一致性快照**（静态文件，放云盘才安全）。
    database_url: str = "sqlite:///./alphas.db"

    # ── 备份（每日一致性快照）──────────────────────────────────────────────
    # backup_dir 建议指向**云同步目录**（快照是静态文件，同步安全，等于异地副本）。
    enable_backup:  bool = True
    backup_dir:     str  = "backups"   # 建议设为 OneDrive 下的某个目录
    backup_keep_n:  int  = 14          # 保留最近 N 份快照

    # ── LLM ──────────────────────────────────────────────────────────────
    openai_api_key: str = Field("", validation_alias="OPENAI_API_KEY")

    # ── FastAPI ───────────────────────────────────────────────────────────
    app_title: str = "Quant Agent API"
    app_version: str = "0.1.0"
    debug: bool = False
    cors_origins: list[str] = ["*"]

    # ── 数据集默认参数 ────────────────────────────────────────────────────
    # 默认使用真实市场数据集（dataset_registry.py 中的注册名称）
    # 可通过 CLI --dataset 或环境变量 DEFAULT_DATASET 覆盖
    default_dataset: str = "us_tech_large"
    default_start:   str = "2020-01-01"
    default_end:     str = "2024-01-01"

    # ── 回测默认参数 ──────────────────────────────────────────────────────
    # n_tickers / n_days 仅在合成数据回退模式下使用（已弃用）
    default_n_tickers: int = 20
    default_n_days: int = 120
    initial_capital: float = 1_000_000.0

    # ── GP 默认参数 ───────────────────────────────────────────────────────
    default_pop_size: int = 20
    default_n_gen: int = 5
    default_n_workers: int = 1

    # ── 调度器（Task 5.3）─────────────────────────────────────────────────
    # 默认关闭：测试与 CLI 模式不启动；生产服务器设 ENABLE_SCHEDULER=true
    enable_scheduler: bool = False
    scheduler_db_url: str = "sqlite:///scheduler_jobs.db"
    scheduler_timezone: str = "UTC"

    # ── Paper Trading 每日管线（Task 7.1/7.3）─────────────────────────────
    # 默认关闭：设 ENABLE_PAPER_TRADING=true 后，调度器会注册每日摄取+交易循环任务
    enable_paper_trading: bool = False
    paper_dataset: str = "us_tech_large"
    paper_start:   str = "2020-01-01"
    paper_end:     str = "2024-01-01"
    # Phase PM.4：组合账本的真实资金量（容量约束依赖它）
    paper_aum:     float = 1_000_000.0

    # ── 交易现实（Phase TR，显式声明的现实事实 T1，非隐藏默认）────────────
    trading_account_type: str  = "margin"     # cash | margin
    trading_allow_short:  bool = False        # 暂 long-only（即使 margin 也先不做空）
    trading_broker:       str  = "moomoo_us"  # 决定佣金/规费档（moomoo 美股佣金免费）

    # ── 组合层门控与风控（Phase PM 接线，2026-08-30）─────────────────────
    # 以下均为「行为/风险偏好」选择（T1，用户显式设定）；默认值是**合理起点、非自然定律**，
    # 请按你的实际风险偏好确认/修改。全部配置化，不在代码里硬编码。
    pm_marginal_selection:   bool  = True    # PM.S2：多因子时按边际贡献贪心选，而非全纳入
    pm_marginal_min_improve: float = 0.05    # 边际 OOS-Sharpe 提升阈值（低于则不纳入）
    pm_strategy_gate_eval:   bool  = True    # PM.S1：每次组合评估策略级验证门并记录 verdict
    pm_strategy_gate_block:  bool  = False   # 策略门不过是否停交易（默认否：paper 期先收集前向证据）
    pm_no_trade_band:        float = 0.0     # PM.6：无交易带宽（0=由 TradingContext 数据推导；>0=显式覆盖）
    pm_horizon_fast_thresh:  float = 4.0     # 年化换手 > 此值 → fast 因子（快慢分类阈值）
    # PM.7 修正错配：因子入池门。leak=低门槛泄漏过滤（默认，严门在策略层）；strict=旧因子级严门。
    factor_gate_mode:        str   = "leak"  # leak | strict

    # ── Phase TR.4：分级晋级门（阈值全配置化，不写死）─────────────────────
    tr_experiment_mode:       bool  = True   # 实验模式：策略门未过也可进 paper（分级标注）收前向证据
    tr_paper_min_sharpe:      float = -99.0  # 非实验模式下进 paper 的 Sharpe 下限
    tr_min_forward_days:      int   = 60     # →ACTIVE 需要的前向观测交易日
    tr_min_ic_tstat:          float = 2.0    # →ACTIVE 的 realized IC t 门槛
    tr_min_ic_mean:           float = 0.0    # →ACTIVE 的 realized IC 均值下限
    # 是否**阻断**未过 →ACTIVE 门的激活。
    # Phase 11 已完成回放/前向分离（is_forward），该门现在**只吃真前向样本**，技术上已可启用。
    # 仍默认 False：此刻前向样本为 0（尚未开始真实前向积累）；等累计满
    # tr_min_forward_days 个交易日、准备逼近真钱时，把它设为 True 让门真正拦。
    tr_enforce_active_gate:   bool  = False
    # 组合风控（PM.5）—— 风险偏好，务必按你的意愿设定
    risk_max_gross:          float = 1.0     # 总敞口上限（Σ|w|）
    risk_max_name_weight:    float = 0.10    # 单票 ≤ 10% NAV
    risk_max_sector_weight:  float = 0.30    # 单行业 ≤ 30% NAV
    risk_target_vol_ann:     float = 0.0     # 目标年化波动（0=关闭 vol targeting）
    risk_max_drawdown:       float = 0.20    # 回撤熔断阈值
    risk_halt_on_drawdown:   bool  = False   # 熔断触发是否停交易（默认否：先记录告警）

    # ── 价格数据源（Phase TR.2，单一权威源）───────────────────────────────
    # yahoo = 默认（yfinance）；moomoo = 经本地 OpenD 网关（研究/执行同源，消除 skew）。
    # 设 moomoo 后，US 数据集价格改从 MoomooProvider 取；非美（akshare/ccxt）不受影响。
    price_source: str = "yahoo"               # yahoo | moomoo
    moomoo_host:  str = "127.0.0.1"           # OpenD 网关地址
    moomoo_port:  int = 11111                 # OpenD API 端口

    # ── Point-in-Time 数据存储（Task 8.1）────────────────────────────────
    # 每日摄取通过健康门的数据按 (field, date, as_of) 追加进此目录，历史只追加不修改
    pit_store_dir: str = "pit_store"

    # ── 自主度（Phase 9.4）────────────────────────────────────────────────
    # manual = 默认模式：VALIDATED→PAPER 由人工 approve/reject 把关（用户只批准或拒绝）
    # auto   = 全自动：VALIDATED→PAPER 由规则/红队自动决定（Phase 13.3 才真正启用）
    autonomy_mode: str = "manual"

    # ── 自主发现（Phase 9.2）──────────────────────────────────────────────
    # 默认关闭：设 ENABLE_DISCOVERY=true 后，调度器注册每晚的自主因子发现任务
    enable_discovery: bool = False
    discovery_dataset: str = "us_broad_large"
    discovery_start:   str = "2021-01-01"
    discovery_end:     str = "2024-01-01"


settings = Settings()
