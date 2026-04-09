"""Configuration management for simil using Pydantic Settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings for simil.

    All fields can be overridden via environment variables prefixed with
    ``SIMIL_`` or via a ``.env`` file in the current working directory.
    """

    model_config = SettingsConfigDict(
        env_prefix="SIMIL_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Library location
    library_path: Path | None = None
    library_name: str = "default"

    # Embedder / backend
    embedder: str = "mfcc"
    index_backend: str = "numpy"

    # Audio clip parameters
    max_clip_duration: float = 180.0
    url_clip_duration: float = 60.0
    clip_start_pct: float = 0.10

    # Search defaults
    workers: int = 4
    top_k: int = 10
    min_score: float = 0.0

    # Indexing behaviour
    checkpoint_every: int = 100

    # Runtime
    warmup_on_startup: bool = True
    log_queries: bool = False

    # Optional API keys
    spotify_client_id: str | None = None
    spotify_client_secret: str | None = None

    @computed_field  # type: ignore[misc]
    @property
    def index_dir(self) -> Path:
        """Computed index directory: ``~/.simil/libraries/{library_name}/``."""
        return Path.home() / ".simil" / "libraries" / self.library_name
