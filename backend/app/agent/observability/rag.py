"""RAG evidence trace helpers for the PaperDesk Agent lifecycle."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.models import AgentLifecycleStage, RuntimeRequest


class AgentRagTraceService:
    """Attach compact RAG evidence and quality metadata to lifecycle traces."""

    def build_payload(
        self,
        *,
        evidence: Iterable[dict[str, Any]],
        quality: dict[str, Any] | None = None,
        selected_document_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        evidence_items = list(evidence)
        return {
            "evidence_count": len(evidence_items),
            "selected_document_ids": list(selected_document_ids or []),
            "citations": [
                item.get("citation") or item.get("citation_label")
                for item in evidence_items
                if item.get("citation") or item.get("citation_label")
            ],
            "document_ids": sorted(
                {
                    str(item.get("document_id"))
                    for item in evidence_items
                    if item.get("document_id")
                }
            ),
            "quality": quality or {},
            "warnings": list((quality or {}).get("warnings", [])) if isinstance(quality, dict) else [],
        }

    def record(
        self,
        request: RuntimeRequest,
        *,
        evidence: Iterable[dict[str, Any]],
        quality: dict[str, Any] | None = None,
    ) -> RuntimeRequest:
        request.add_trace(
            AgentLifecycleStage.RAG,
            "RAG evidence metadata attached",
            self.build_payload(
                evidence=evidence,
                quality=quality,
                selected_document_ids=request.context.selected_document_ids,
            ),
        )
        return request
