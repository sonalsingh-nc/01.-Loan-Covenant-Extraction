"""Central configuration for the covenant extraction pipeline.

Values are loaded from environment variables (and a local .env file, if
present) using pydantic-settings. See .env.example for the full list of
supported settings.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LM Studio (OpenAI-compatible local server)
    lm_studio_base_url: str = "http://localhost:1234/v1"
    llm_model_name: str = "qwen2.5-32b-instruct"

    # Claude (Anthropic API) -- alternative backend to LM Studio, for
    # comparing extraction quality/cost across models (see evals/).
    # Credentials resolve from the environment (ANTHROPIC_API_KEY etc.);
    # not read from settings here.
    claude_model_name: str = "claude-haiku-4-5"

    # Embedding model
    embed_model_name: str = "BAAI/bge-m3"
    embed_model_cache_dir: Path = Path("./models")

    # Retrieval
    max_candidates: int = 5


settings = Settings()
