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


settings = Settings()
