from __future__ import annotations

import sys

from fastapi.testclient import TestClient

from app.api.main import create_app, get_embedding_service, get_vectorstore
from app.config import Settings, get_settings


def test_settings_read_env_file_and_prepare_runtime_paths(sandbox_dir) -> None:
    env_file = sandbox_dir / ".env"
    env_file.write_text(
        "\n".join(
            [
                "APP_NAME=EnvDesk",
                "APP_VERSION=0.2.0",
                "LLM_PROVIDER=openai",
                "LLM_BASE_URL=   ",
                "LLM_API_KEY=   ",
                "LLM_MODEL=gpt-test",
                "OPENALEX_API_KEY=openalex-secret",
                "EMBEDDING_PROVIDER=local",
                "EMBEDDING_MODEL=test-embedding",
                "EMBEDDING_WARMUP_ON_START=false",
                "EMBEDDING_CACHE_DIR=./runtime/huggingface",
                "EMBEDDING_HF_ENDPOINT=https://hf-mirror.example",
                "EMBEDDING_LOCAL_FILES_ONLY=true",
                "SQLITE_PATH=./runtime/paperdesk.db",
                "MILVUS_URI=http://milvus.example:19530",
                "MILVUS_TOKEN=milvus-token",
                "MILVUS_DATABASE=paperdesk_runtime",
                "MILVUS_COLLECTION=paperdesk_collection",
                "WORKSPACE_DIR=./runtime/workspace",
                "UPLOAD_DIR=./runtime/workspace/uploads",
                "REPORT_DIR=./runtime/workspace/reports",
                "RUNTIME_CONTEXT_DIR=./runtime/context",
                "OPENALEX_BASE_URL=https://openalex.example/api",
                "ARXIV_BASE_URL=https://arxiv.example/api",
                "CORS_ORIGINS=http://localhost:5173, http://127.0.0.1:5173",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)
    settings.ensure_directories()

    assert settings.app_name == "EnvDesk"
    assert settings.app_version == "0.2.0"
    assert settings.llm_provider == "openai"
    assert settings.llm_base_url is None
    assert settings.llm_api_key is None
    assert settings.llm_model == "gpt-test"
    assert settings.openalex_api_key == "openalex-secret"
    assert settings.embedding_provider == "local"
    assert settings.embedding_model == "test-embedding"
    assert settings.embedding_warmup_on_start is False
    assert settings.embedding_cache_path == settings.resolve_path("./runtime/huggingface")
    assert settings.embedding_hf_endpoint == "https://hf-mirror.example"
    assert settings.embedding_local_files_only is True
    assert settings.milvus_uri == "http://milvus.example:19530"
    assert settings.effective_milvus_uri == "http://milvus.example:19530"
    assert settings.uses_embedded_milvus is False
    assert settings.milvus_token == "milvus-token"
    assert settings.milvus_database == "paperdesk_runtime"
    assert settings.milvus_collection == "paperdesk_collection"
    assert settings.openalex_base_url == "https://openalex.example/api"
    assert settings.arxiv_base_url == "https://arxiv.example/api"
    assert settings.get_cors_origins_list() == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    assert settings.sqlite_file == (settings.resolve_path("./runtime/paperdesk.db"))
    assert settings.workspace_path == settings.resolve_path("./runtime/workspace")
    assert settings.upload_path == settings.resolve_path("./runtime/workspace/uploads")
    assert settings.report_path == settings.resolve_path("./runtime/workspace/reports")
    assert settings.runtime_context_path == settings.resolve_path("./runtime/context")

    assert settings.sqlite_file.parent.exists()
    assert settings.workspace_path.exists()
    assert settings.upload_path.exists()
    assert settings.report_path.exists()
    assert settings.runtime_context_path.exists()


def test_create_app_with_blank_api_key_and_milvus_vectorstore(
    sandbox_dir,
    monkeypatch,
) -> None:
    data_dir = sandbox_dir / "data"
    workspace_dir = sandbox_dir / "workspace"
    runtime_context_dir = sandbox_dir / "runtime" / "context"

    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("OPENALEX_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_WARMUP_ON_START", "false")
    monkeypatch.setenv("SQLITE_PATH", str(data_dir / "paperdesk.db"))
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace_dir))
    monkeypatch.setenv("UPLOAD_DIR", str(workspace_dir / "uploads"))
    monkeypatch.setenv("REPORT_DIR", str(workspace_dir / "reports"))
    monkeypatch.setenv("RUNTIME_CONTEXT_DIR", str(runtime_context_dir))
    monkeypatch.setenv("MILVUS_URI", "http://fake-milvus:19530")
    monkeypatch.setenv("MILVUS_DATABASE", "paperdesk_test")
    monkeypatch.setenv("MILVUS_COLLECTION", "paperdesk_collection")
    monkeypatch.setenv("MILVUS_AUTO_START", "false")

    get_settings.cache_clear()
    get_embedding_service.cache_clear()
    get_vectorstore.cache_clear()

    settings = get_settings()
    assert settings.llm_api_key is None
    assert settings.openalex_api_key is None
    assert settings.runtime_context_path == runtime_context_dir
    assert settings.milvus_uri == "http://fake-milvus:19530"
    assert settings.effective_milvus_uri == "http://fake-milvus:19530"

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["vectorstore_status"] in {"starting", "ready", "failed"}
    assert payload["vectorstore_uri"] == "http://fake-milvus:19530"
    assert payload["embedding_status"] == "disabled"
    assert payload["embedding_model"] == settings.embedding_model
    assert get_vectorstore().uri == "http://fake-milvus:19530"


def test_settings_default_to_embedded_milvus_lite_file(sandbox_dir) -> None:
    env_file = sandbox_dir / ".env"
    env_file.write_text("SQLITE_PATH=./runtime/paperdesk.db\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)
    settings.ensure_directories()

    assert settings.milvus_uri is None
    if sys.platform == "win32":
        assert settings.uses_embedded_milvus is False
        assert settings.effective_milvus_uri == "http://127.0.0.1:19530"
    else:
        assert settings.uses_embedded_milvus is True
        assert settings.effective_milvus_uri.endswith("data\\milvus\\paperdesk_milvus.db") or settings.effective_milvus_uri.endswith("data/milvus/paperdesk_milvus.db")
    assert settings.milvus_lite_file.parent.exists()
