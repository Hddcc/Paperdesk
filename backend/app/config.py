"""Application settings and path helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

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
    max_context_tokens: int | None = None
    response_reserve_tokens: int = 16000
    compact_warn_ratio: float = 0.72
    compact_force_ratio: float = 0.90
    recent_turns_min: int = 8
    max_evidence_items: int = 8
    max_evidence_chars_per_item: int = 420
    openalex_api_key: str | None = None
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-m3"
    embedding_warmup_on_start: bool = True
    embedding_cache_dir: str | None = None
    embedding_hf_endpoint: str | None = None
    embedding_local_files_only: bool = False
    sqlite_path: str = "./data/paperdesk.db"
    milvus_uri: str | None = None
    milvus_token: str | None = None
    milvus_database: str = "default"
    milvus_collection: str = "paperdesk_local_library"
    milvus_lite_path: str = "./data/milvus/paperdesk_milvus.db"
    milvus_auto_start: bool = True
    milvus_start_timeout_seconds: int = 90
    milvus_runtime_dir: str = "./runtime/milvus"
    milvus_container_name: str = "paperdesk-milvus"
    milvus_image: str = "milvusdb/milvus:v2.6.14"
    docker_desktop_path: str = "C:/Program Files/Docker/Docker/Docker Desktop.exe"
    workspace_dir: str = "./workspace"
    upload_dir: str = "./workspace/uploads"
    file_asset_dir: str = "./workspace/files"
    file_asset_max_upload_bytes: int = 5 * 1024 * 1024
    report_dir: str = "./workspace/reports"
    runtime_context_dir: str | None = "./runtime/context"
    claude_dir: str | None = None
    openalex_base_url: str = "https://api.openalex.org"
    arxiv_base_url: str = "http://export.arxiv.org/api/query"
    cors_origins: str = Field(default="*")
    enable_research_task_agent: bool = False
    enable_research_from_knowledge: bool = False
    enable_experimental_mcp: bool = False
    enable_mcp_in_knowledge: bool = False
    enable_subagent_execution: bool = False
    enable_auto_reflection: bool = False
    enable_skill_context_prompt_injection: bool = True
    enable_skill_context_paper_qa_lightweight_only: bool = True

    @field_validator(
        "llm_base_url",
        "llm_api_key",
        "openalex_api_key",
        "embedding_cache_dir",
        "embedding_hf_endpoint",
        "milvus_uri",
        "milvus_token",
        "runtime_context_dir",
        "claude_dir",
        mode="before",
    )
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
    def workspace_path(self) -> Path:
        return self.resolve_path(self.workspace_dir)

    @property
    def upload_path(self) -> Path:
        return self.resolve_path(self.upload_dir)

    @property
    def report_path(self) -> Path:
        return self.resolve_path(self.report_dir)

    @property
    def file_asset_path(self) -> Path:
        return self.resolve_path(self.file_asset_dir)

    @property
    def claude_path(self) -> Path:
        """Backward-compatible alias for the runtime context directory."""
        return self.runtime_context_path

    @property
    def runtime_context_path(self) -> Path:
        return self.resolve_path(self.claude_dir or self.runtime_context_dir or "./runtime/context")

    @property
    def project_root(self) -> Path:
        return BACKEND_ROOT.parent

    @property
    def milvus_lite_file(self) -> Path:
        return self.resolve_path(self.milvus_lite_path)

    @property
    def effective_milvus_uri(self) -> str:
        if self.milvus_uri:
            return self.milvus_uri
        if sys.platform == "win32":
            return "http://127.0.0.1:19530"
        return str(self.milvus_lite_file)

    @property
    def uses_embedded_milvus(self) -> bool:
        normalized = self.effective_milvus_uri.lower()
        return not normalized.startswith(("http://", "https://", "tcp://", "grpc://", "unix://"))

    @property
    def uses_local_managed_milvus(self) -> bool:
        normalized = self.effective_milvus_uri.lower()
        return normalized.startswith(("http://127.0.0.1", "http://localhost", "https://127.0.0.1", "https://localhost"))

    @property
    def milvus_runtime_path(self) -> Path:
        return self.resolve_path(self.milvus_runtime_dir)

    @property
    def docker_desktop_executable(self) -> Path:
        return Path(self.docker_desktop_path)

    @property
    def effective_max_context_tokens(self) -> int:
        if self.max_context_tokens is not None:
            return max(self.max_context_tokens, 4000)

        model_name = self.llm_model.casefold()
        if any(marker in model_name for marker in ("gpt-4.1", "gpt-4o", "o4-", "o3-")):
            return 128000
        if any(marker in model_name for marker in ("qwen", "deepseek", "glm")):
            return 64000
        return 64000

    def get_cors_origins_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def effective_llm_api_key(self) -> str | None:
        return self.llm_api_key

    @property
    def effective_llm_base_url(self) -> str | None:
        if self.llm_base_url:
            return self.llm_base_url
        return self._provider_default_base_url(self.llm_provider)

    @property
    def effective_llm_model(self) -> str:
        return self.llm_model

    @staticmethod
    def _provider_default_base_url(provider: str) -> str | None:
        return {
            "deepseek": "https://api.deepseek.com",
            "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "glm": "https://open.bigmodel.cn/api/paas/v4",
            "zhipu": "https://open.bigmodel.cn/api/paas/v4",
        }.get(provider.casefold())

    def ensure_directories(self) -> None:
        """Create required runtime directories."""
        directories = [
            self.sqlite_file.parent,
            self.milvus_lite_file.parent,
            self.milvus_runtime_path,
            self.workspace_path,
            self.upload_path,
            self.file_asset_path,
            self.report_path,
            self.runtime_context_path,
            self.runtime_context_path / "runtime",
            self.runtime_context_path / "sessions",
        ]
        if self.embedding_cache_path is not None:
            directories.append(self.embedding_cache_path)

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def embedding_cache_path(self) -> Path | None:
        if self.embedding_cache_dir is None:
            return None
        return self.resolve_path(self.embedding_cache_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings."""
    settings = Settings()
    settings.ensure_directories()
    return settings
