"""Chat trace adapter for the runtime-first Agent Core path."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.agent.skills import SkillContextBuilder, SkillSelector, SkillRegistry
from app.agent.tools import ToolRegistry
from app.models import ResearchRunStatus, TraceEventType
from app.runtime.message_bus import MessageBus


class AgentCoreTraceAdapter:
    """Provide chat trace and registry access without legacy route selection."""

    def __init__(
        self,
        *,
        research_repository: Any,
        runtime_repository: Any,
        tool_registry: ToolRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        self.research_repository = research_repository
        self.message_bus = MessageBus(runtime_repository)
        self.tool_registry = tool_registry or ToolRegistry()
        self.skill_registry = skill_registry or SkillRegistry()
        self.skill_selector = SkillSelector()
        self.skill_context_builder = SkillContextBuilder(self.skill_registry)

    def available_tools(self):
        return self.tool_registry.list_default_candidates(scope="knowledge")

    def available_skills(self):
        return self.skill_registry.list_enabled()

    def select_skills_for_trace(self, payload: Any):
        return self.skill_selector.select(
            prompt=payload.user_prompt,
            command=payload.runtime_context.get("command"),
            intent_hint=payload.runtime_context.get("intent_hint"),
            selected_document_count=len(payload.selected_document_ids),
            attachments=payload.attachments,
            available_skills=payload.available_skills,
            task_type=payload.runtime_context.get("task_type"),
            route=payload.runtime_context.get("route"),
        )

    def _begin_trace(self, payload: Any) -> str:
        trace_id = f"chat-core-{uuid4().hex}"
        self.research_repository.create_run(trace_id, f"Chat Agent Core: {payload.user_prompt[:80]}")
        self.research_repository.update_run_status(trace_id, ResearchRunStatus.RUNNING_TASK)
        self.append_trace(
            trace_id,
            status="agent_core_started",
            message="Agent Core chat lifecycle started.",
            payload={
                "session_id": payload.session_id,
                "message_id": payload.message_id,
                "selected_document_count": len(payload.selected_document_ids),
                "attachment_count": len(payload.attachments),
            },
        )
        return trace_id

    def append_trace(
        self,
        trace_id: str,
        *,
        status: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.message_bus.append_trace(
            run_id=trace_id,
            task_id=None,
            trace_type=TraceEventType.CONTROL,
            status=status,
            message=message,
            payload=payload or {},
        )

    def finish_trace(
        self,
        trace_id: str,
        *,
        status: ResearchRunStatus = ResearchRunStatus.COMPLETED,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.message_bus.append_trace(
            run_id=trace_id,
            task_id=None,
            trace_type=TraceEventType.MERGE,
            status="agent_core_finished",
            message="Agent Core chat lifecycle finished.",
            payload=payload or {},
        )
        self.research_repository.update_run_status(trace_id, status)
