from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    data_dir = tmp_path / "data"
    workspace_dir = tmp_path / "workspace"
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
        get_export_service,
        get_repository,
        get_research_orchestrator,
        get_vectorstore,
    )
    from app.config import get_settings

    get_settings.cache_clear()
    get_repository.cache_clear()
    get_vectorstore.cache_clear()
    get_document_library_service.cache_clear()
    get_export_service.cache_clear()
    get_research_orchestrator.cache_clear()

    app = create_app()
    return TestClient(app)

