from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.api.main import create_app, get_vectorstore
from app.config import Settings, get_settings


def test_settings_read_env_file_and_prepare_runtime_paths(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
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
                "SQLITE_PATH=./runtime/paperdesk.db",
                "CHROMA_PATH=./runtime/chroma",
                "WORKSPACE_DIR=./runtime/workspace",
                "UPLOAD_DIR=./runtime/workspace/uploads",
                "REPORT_DIR=./runtime/workspace/reports",
                "VECTORSTORE_DIR=./runtime/workspace/vectorstore",
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
    assert settings.openalex_base_url == "https://openalex.example/api"
    assert settings.arxiv_base_url == "https://arxiv.example/api"
    assert settings.get_cors_origins_list() == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    assert settings.sqlite_file == (settings.resolve_path("./runtime/paperdesk.db"))
    assert settings.chroma_storage_path == settings.resolve_path("./runtime/chroma")
    assert settings.workspace_path == settings.resolve_path("./runtime/workspace")
    assert settings.upload_path == settings.resolve_path("./runtime/workspace/uploads")
    assert settings.report_path == settings.resolve_path("./runtime/workspace/reports")
    assert settings.vectorstore_path == settings.resolve_path("./runtime/workspace/vectorstore")

    assert settings.sqlite_file.parent.exists()
    assert settings.chroma_storage_path.exists()
    assert settings.workspace_path.exists()
    assert settings.upload_path.exists()
    assert settings.report_path.exists()
    assert settings.vectorstore_path.exists()


def test_create_app_with_blank_api_key_and_reserved_chroma_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    workspace_dir = tmp_path / "workspace"
    chroma_dir = tmp_path / "chroma"
    vector_dir = workspace_dir / "vectorstore"

    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("OPENALEX_API_KEY", "")
    monkeypatch.setenv("SQLITE_PATH", str(data_dir / "paperdesk.db"))
    monkeypatch.setenv("CHROMA_PATH", str(chroma_dir))
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace_dir))
    monkeypatch.setenv("UPLOAD_DIR", str(workspace_dir / "uploads"))
    monkeypatch.setenv("REPORT_DIR", str(workspace_dir / "reports"))
    monkeypatch.setenv("VECTORSTORE_DIR", str(vector_dir))

    get_settings.cache_clear()
    get_vectorstore.cache_clear()

    settings = get_settings()
    assert settings.llm_api_key is None
    assert settings.openalex_api_key is None
    assert settings.chroma_storage_path == chroma_dir
    assert settings.vectorstore_path == vector_dir

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert get_vectorstore().base_path == vector_dir
