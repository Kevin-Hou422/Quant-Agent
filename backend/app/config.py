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

    # ── 回测默认参数 ──────────────────────────────────────────────────────
    default_n_tickers: int = 20
    default_n_days: int = 120
    initial_capital: float = 1_000_000.0

    # ── GP 默认参数 ───────────────────────────────────────────────────────
    default_pop_size: int = 20
    default_n_gen: int = 5
    default_n_workers: int = 1


settings = Settings()
