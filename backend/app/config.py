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
    database_url: str = "sqlite:///./alphas.db"

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
