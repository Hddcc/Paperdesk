from __future__ import annotations

import math
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

TEST_SANDBOX_ROOT = Path(__file__).resolve().parent.parent / "workspace" / "test-sandboxes"
TEST_SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(TEST_SANDBOX_ROOT))


class FakeMilvusClient:
    def __init__(self) -> None:
        self.collections: dict[str, list[dict]] = {}

    def has_collection(self, *, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, *, collection_name: str, **kwargs) -> None:
        _ = kwargs
        self.collections.setdefault(collection_name, [])

    def load_collection(self, *, collection_name: str) -> None:
        self.collections.setdefault(collection_name, [])

    def insert(self, *, collection_name: str, data: list[dict]) -> None:
        bucket = self.collections.setdefault(collection_name, [])
        bucket.extend([dict(item) for item in data])

    def delete(self, *, collection_name: str, filter: str) -> None:
        bucket = self.collections.setdefault(collection_name, [])
        self.collections[collection_name] = [
            item for item in bucket if not self._matches_filter(item, filter)
        ]

    def search(
        self,
        *,
        collection_name: str,
        data: list[list[float]],
        limit: int,
        filter: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[list[dict]]:
        query_vector = data[0]
        candidates = [
            item
            for item in self.collections.get(collection_name, [])
            if self._matches_filter(item, filter)
        ]
        ranked = sorted(
            candidates,
            key=lambda item: self._distance(query_vector, item.get("embedding") or []),
        )
        results: list[dict] = []
        for item in ranked[:limit]:
            entity = dict(item)
            if output_fields:
                entity = {field: item.get(field) for field in output_fields}
            results.append(
                {
                    "id": item.get("chunk_id"),
                    "distance": self._distance(query_vector, item.get("embedding") or []),
                    "entity": entity,
                }
            )
        return [results]

    @staticmethod
    def _distance(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 1.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 1.0
        cosine = numerator / (left_norm * right_norm)
        return 1 - cosine

    @classmethod
    def _matches_filter(cls, item: dict, filter_expr: str | None) -> bool:
        if not filter_expr:
            return True

        in_match = re.fullmatch(r'(\w+)\s+in\s+\[(.*)\]', filter_expr.strip())
        if in_match:
            field_name = in_match.group(1)
            raw_values = in_match.group(2).strip()
            values = cls._parse_values(raw_values)
            return str(item.get(field_name) or "") in values

        equals_match = re.fullmatch(r'(\w+)\s*==\s*"([^"]*)"', filter_expr.strip())
        if equals_match:
            field_name = equals_match.group(1)
            value = equals_match.group(2)
            return str(item.get(field_name) or "") == value

        return True

    @staticmethod
    def _parse_values(raw_values: str) -> set[str]:
        values: set[str] = set()
        for entry in re.findall(r'"((?:\\.|[^"])*)"', raw_values):
            values.add(entry.replace('\\"', '"').replace("\\\\", "\\"))
        return values


@pytest.fixture()
def sandbox_dir() -> Path:
    sandbox = TEST_SANDBOX_ROOT / f"sandbox-{uuid4()}"
    sandbox.mkdir(parents=True, exist_ok=True)
    try:
        yield sandbox
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


@pytest.fixture(autouse=True)
def patch_milvus_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.vectorstores.milvus_store.MilvusVectorStore._create_client",
        lambda self: FakeMilvusClient(),
    )


@pytest.fixture()
def client(sandbox_dir: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    data_dir = sandbox_dir / "data"
    workspace_dir = sandbox_dir / "workspace"
    upload_dir = workspace_dir / "uploads"
    report_dir = workspace_dir / "reports"
    vector_dir = workspace_dir / "vectorstore"
    claude_dir = sandbox_dir / ".claude"

    monkeypatch.setenv("SQLITE_PATH", str(data_dir / "paperdesk.db"))
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace_dir))
    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
    monkeypatch.setenv("REPORT_DIR", str(report_dir))
    monkeypatch.setenv("VECTORSTORE_DIR", str(vector_dir))
    monkeypatch.setenv("CLAUDE_DIR", str(claude_dir))
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("OPENALEX_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_WARMUP_ON_START", "false")
    monkeypatch.setenv("MILVUS_URI", "http://fake-milvus:19530")
    monkeypatch.setenv("MILVUS_TOKEN", "")
    monkeypatch.setenv("MILVUS_DATABASE", "paperdesk_test")
    monkeypatch.setenv("MILVUS_COLLECTION", "paperdesk_test_collection")
    monkeypatch.setenv("MILVUS_AUTO_START", "false")

    from app.api.main import (
        get_chat_memory_service,
        get_chat_service,
        create_app,
        get_document_library_service,
        get_embedding_service,
        get_export_service,
        get_context_assembler,
        get_context_budget_service,
        get_context_compaction_service,
        get_context_file_store,
        get_knowledge_ingestion_service,
        get_milvus_bootstrap_service,
        get_paper_analysis_agent,
        get_paper_search_service,
        get_paper_selection_agent,
        get_query_translation_service,
        get_rag_service,
        get_report_writer,
        get_repository,
        get_research_orchestrator,
        get_research_workspace_service,
        get_vectorstore,
    )
    from app.config import get_settings

    get_settings.cache_clear()
    get_repository.cache_clear()
    get_embedding_service.cache_clear()
    get_milvus_bootstrap_service.cache_clear()
    get_vectorstore.cache_clear()
    get_paper_search_service.cache_clear()
    get_paper_selection_agent.cache_clear()
    get_paper_analysis_agent.cache_clear()
    get_query_translation_service.cache_clear()
    get_context_file_store.cache_clear()
    get_context_budget_service.cache_clear()
    get_context_compaction_service.cache_clear()
    get_context_assembler.cache_clear()
    get_chat_memory_service.cache_clear()
    get_chat_service.cache_clear()
    get_document_library_service.cache_clear()
    get_knowledge_ingestion_service.cache_clear()
    get_export_service.cache_clear()
    get_research_workspace_service.cache_clear()
    get_report_writer.cache_clear()
    get_rag_service.cache_clear()
    get_research_orchestrator.cache_clear()

    app = create_app()
    return TestClient(app)
