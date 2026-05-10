"""Application settings and path helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime settings for the PaperDesk backend."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "PaperDesk"
    app_version: str = "0.1.0"
    llm_provider: str = "openai"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    openalex_api_key: str | None = None
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-m3"
    sqlite_path: str = "./data/paperdesk.db"
    chroma_path: str = "./data/chroma"
    workspace_dir: str = "./workspace"
    upload_dir: str = "./workspace/uploads"
    report_dir: str = "./workspace/reports"
    vectorstore_dir: str = "./workspace/vectorstore"
    openalex_base_url: str = "https://api.openalex.org"
    arxiv_base_url: str = "http://export.arxiv.org/api/query"
    cors_origins: str = Field(default="*")

    @field_validator("llm_base_url", "llm_api_key", "openalex_api_key", mode="before")
    @classmethod
    def empty_string_as_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    def resolve_path(self, value: str) -> Path:
        """Resolve configured paths relative to the backend root."""
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        return (BACKEND_ROOT / candidate).resolve()

    @property
    def sqlite_file(self) -> Path:
        return self.resolve_path(self.sqlite_path)

    @property
    def chroma_storage_path(self) -> Path:
        return self.resolve_path(self.chroma_path)

    @property
    def workspace_path(self) -> Path:
        return self.resolve_path(self.workspace_dir)

    @property
    def upload_path(self) -> Path:
        return self.resolve_path(self.upload_dir)

    @property
    def report_path(self) -> Path:
        return self.resolve_path(self.report_dir)

    @property
    def vectorstore_path(self) -> Path:
        return self.resolve_path(self.vectorstore_dir)

    def get_cors_origins_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    def ensure_directories(self) -> None:
        """Create required runtime directories."""
        for directory in (
            self.sqlite_file.parent,
            self.chroma_storage_path,
            self.workspace_path,
            self.upload_path,
            self.report_path,
            self.vectorstore_path,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings."""
    settings = Settings()
    settings.ensure_directories()
    return settings
