"""Runtime response finalization for the PaperDesk Agent lifecycle."""

from __future__ import annotations

from typing import Any

from app.models import AgentLifecycleStage, RuntimeRequest, RuntimeResult


class AgentRuntimeResponseRecorder:
    """Attach final response metadata to a runtime result and trace."""

    def complete(
        self,
        *,
        request: RuntimeRequest,
        result: RuntimeResult,
        response_text: str,
        action_status: str | None = None,
        retrieval_status: str | None = None,
        citations: list[str] | None = None,
        used_document_ids: list[str] | None = None,
        evidence_items: list[Any] | None = None,
        error_reason: str | None = None,
    ) -> RuntimeResult:
        citations = list(citations or [])
        used_document_ids = list(used_document_ids or [])
        evidence_items = list(evidence_items or [])
        status = "failed" if error_reason else "completed"
        payload = {
            "route": request.route.route.value,
            "runtime": result.runtime.value,
            "orchestration_pattern": request.route.orchestration_pattern.value,
            "response_owner": result.runtime.value,
            "response_status": status,
            "action_status": action_status,
            "retrieval_status": retrieval_status,
            "target_scope": request.route.target_scope,
            "active_skill": self._active_skill_payload(request),
            "citation_count": len(citations),
            "used_document_count": len(used_document_ids),
            "evidence_count": len(evidence_items),
            "error_reason": error_reason,
        }
        result.status = status
        result.response_text = response_text
        result.error = error_reason
        result.data.update(payload)
        result.metrics.update(
            {
                "response_char_count": len(response_text),
                "citation_count": len(citations),
                "used_document_count": len(used_document_ids),
                "evidence_count": len(evidence_items),
                "error_reason": error_reason,
            }
        )
        result.trace.append(
            request_trace_event(
                stage=AgentLifecycleStage.RESPONSE,
                message="runtime response finalized",
                payload=payload,
            )
        )
        return result

    @staticmethod
    def _active_skill_payload(request: RuntimeRequest) -> dict[str, Any] | None:
        if request.active_skill is None:
            return None
        return {
            "skill_id": request.active_skill.skill_id,
            "name": request.active_skill.name,
            "source": request.active_skill.source,
            "confidence": request.active_skill.confidence,
            "allowed_tool_count": len(request.active_skill.allowed_tool_ids),
        }


def request_trace_event(*, stage: AgentLifecycleStage, message: str, payload: dict[str, Any]):
    """Create trace events without leaking pydantic construction into callers."""

    from app.models import AgentLifecycleTraceEvent

    return AgentLifecycleTraceEvent(stage=stage, message=message, payload=payload)
