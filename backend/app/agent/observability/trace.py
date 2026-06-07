"""Trace helpers for the PaperDesk Agent lifecycle."""

from __future__ import annotations

from app.models import AgentLifecycleStage, AgentLifecycleTraceEvent, RuntimeRequest, RuntimeResult


class AgentTraceRecorder:
    """Append lifecycle trace events to requests and results."""

    def record_request(
        self,
        request: RuntimeRequest,
        stage: AgentLifecycleStage,
        message: str,
        payload: dict | None = None,
    ) -> RuntimeRequest:
        request.add_trace(stage, message, payload or {})
        return request

    def record_result(
        self,
        result: RuntimeResult,
        stage: AgentLifecycleStage,
        message: str,
        payload: dict | None = None,
    ) -> RuntimeResult:
        result.trace.append(
            AgentLifecycleTraceEvent(
                stage=stage,
                message=message,
                payload=payload or {},
            )
        )
        return result
