from __future__ import annotations

import os
from pathlib import Path
import shutil
from uuid import uuid4

import chromadb
import pytest
from fastapi.testclient import TestClient

TEST_SANDBOX_ROOT = Path(__file__).resolve().parent.parent / "workspace" / "test-sandboxes"
TEST_SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(TEST_SANDBOX_ROOT))


@pytest.fixture()
def sandbox_dir() -> Path:
    sandbox = TEST_SANDBOX_ROOT / f"sandbox-{uuid4()}"
    sandbox.mkdir(parents=True, exist_ok=True)
    try:
        yield sandbox
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


@pytest.fixture(autouse=True)
def patch_chroma_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.vectorstores.chroma_store.chromadb.PersistentClient",
        lambda path: chromadb.EphemeralClient(),
    )


@pytest.fixture()
def client(sandbox_dir: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    data_dir = sandbox_dir / "data"
    workspace_dir = sandbox_dir / "workspace"
    upload_dir = workspace_dir / "uploads"
    report_dir = workspace_dir / "reports"
    vector_dir = workspace_dir / "vectorstore"

    monkeypatch.setenv("SQLITE_PATH", str(data_dir / "paperdesk.db"))
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace_dir))
    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
    monkeypatch.setenv("REPORT_DIR", str(report_dir))
    monkeypatch.setenv("VECTORSTORE_DIR", str(vector_dir))

    from app.api.main import (
        create_app,
        get_document_library_service,
        get_embedding_service,
        get_export_service,
        get_paper_search_service,
        get_repository,
        get_research_orchestrator,
        get_vectorstore,
    )
    from app.config import get_settings

    get_settings.cache_clear()
    get_repository.cache_clear()
    get_embedding_service.cache_clear()
    get_vectorstore.cache_clear()
    get_paper_search_service.cache_clear()
    get_document_library_service.cache_clear()
    get_export_service.cache_clear()
    get_research_orchestrator.cache_clear()

    app = create_app()
    return TestClient(app)
