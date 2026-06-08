"""Compatibility adapters for legacy chat-side orchestration.

This module keeps the old AgentOrchestrator vocabulary behind the Agent runtime
boundary while ChatService is migrated toward the runtime-first lifecycle.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from app.models import (
    AgentModeDecision,
    AgentOrchestratorInput,
    AgentRunMode,
    ChatAttachment,
    ChatMessage,
    ChatMessageRequest,
    ChatSession,
    KnowledgeIntent,
    KnowledgeRiskLevel,
    KnowledgeRoute,
    MemorySnapshot,
    PaperDeskRoute,
)


class AgentModeCompatibilityAdapter:
    """Build legacy mode decisions from runtime-first chat inputs."""

    def __init__(
        self,
        *,
        agent_orchestrator: Any | None,
        knowledge_agent_provider: Any,
    ) -> None:
        self.agent_orchestrator = agent_orchestrator
        self.knowledge_agent_provider = knowledge_agent_provider

    def build_fast_path_decision(
        self,
        *,
        session: ChatSession,
        user_message: ChatMessage,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        memory_snapshot: MemorySnapshot,
        route: KnowledgeRoute | PaperDeskRoute | str,
        intent: KnowledgeIntent,
        reason: str,
        requires_tools: bool,
        target_runtime: str,
    ) -> AgentModeDecision | None:
        if self.agent_orchestrator is None:
            return None
        payload = self._payload(
            session=session,
            user_message=user_message,
            request=request,
            attachments=attachments,
            selected_document_ids=selected_document_ids,
            memory_snapshot=memory_snapshot,
            runtime_context={
                "fast_path": True,
                "session_title": session.title,
                "entrypoint": "knowledge",
                "command": request.command,
                "intent_hint": request.intent_hint,
            },
        )
        begin_trace = getattr(self.agent_orchestrator, "_begin_trace", None)
        trace_id = begin_trace(payload) if callable(begin_trace) else f"chat-fast-{uuid4().hex}"
        skill_selection = self.agent_orchestrator.select_skills_for_trace(payload)
        skill_context_builder = getattr(self.agent_orchestrator, "skill_context_builder", None)
        skill_context_summary = (
            skill_context_builder.build(skill_selection)
            if skill_context_builder is not None and hasattr(skill_context_builder, "build")
            else None
        )
        legacy_route = self._legacy_route(route)
        legacy_mode = (
            AgentRunMode.DIRECT
            if target_runtime == "DeterministicReadRuntime"
            else self._legacy_mode_for_route(route)
        )
        decision = AgentModeDecision(
            mode=legacy_mode,
            route=legacy_route,
            intent=intent,
            reason=reason,
            confidence=0.93,
            target_runtime=target_runtime,
            requires_tools=requires_tools,
            requires_rag=False,
            requires_confirmation=False,
            risk_level=KnowledgeRiskLevel.LOW if requires_tools else KnowledgeRiskLevel.NONE,
            required_capabilities=self._fast_path_capabilities(
                requires_tools=requires_tools,
                target_runtime=target_runtime,
            ),
            trace_id=trace_id,
            fallback_used=False,
        )
        self.agent_orchestrator.append_trace(
            trace_id,
            status="agent_mode_selected",
            message=f"Agent mode selected by local fast path: {decision.mode.value}.",
            payload={
                "mode": decision.mode.value,
                "route": decision.route.value,
                "intent": decision.intent.value,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "target_runtime": decision.target_runtime,
                "requires_tools": decision.requires_tools,
                "requires_rag": decision.requires_rag,
                "requires_confirmation": decision.requires_confirmation,
                "risk_level": decision.risk_level.value,
                "target_objects": [],
                "fallback_used": decision.fallback_used,
                "required_capabilities": decision.required_capabilities,
                "decision_source": "local_fast_path",
                "guardrail_candidate": None,
                "rule_candidate": None,
                "fallback_candidate": None,
                "llm_candidate": None,
                "available_tool_ids": [tool.tool_id for tool in payload.available_tools[:30]],
                "available_skill_ids": [skill.skill_id for skill in payload.available_skills[:20]],
                "primary_skill_id": (
                    skill_selection.primary_skill.skill_id
                    if skill_selection.primary_skill is not None
                    else None
                ),
                "used_skill_ids": [skill.skill_id for skill in skill_selection.used_skills],
                "used_skills": [skill.model_dump(mode="json") for skill in skill_selection.used_skills],
                "skill_context_summary": (
                    skill_context_summary.model_dump(mode="json") if skill_context_summary is not None else None
                ),
                "has_conversation_referents": False,
                "memory_hit_count": len(memory_snapshot.items),
            },
        )
        return decision

    def select_mode(
        self,
        *,
        session: ChatSession,
        user_message: ChatMessage,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        memory_snapshot: MemorySnapshot,
        has_pending_action_reader: Callable[[str], bool],
    ) -> AgentModeDecision | None:
        if self.agent_orchestrator is None:
            return None
        conversation_referents: dict = {}
        has_pending_action = False
        if self.knowledge_agent_provider.available:
            try:
                conversation_referents = self.knowledge_agent_provider.conversation_referents(session.id)
            except Exception:
                conversation_referents = {}
            try:
                has_pending_action = self.knowledge_agent_provider.has_pending_action(session.id)
            except Exception:
                has_pending_action = False
        if not has_pending_action:
            try:
                has_pending_action = has_pending_action_reader(session.id)
            except Exception:
                has_pending_action = False
        payload = self._payload(
            session=session,
            user_message=user_message,
            request=request,
            attachments=attachments,
            selected_document_ids=selected_document_ids,
            memory_snapshot=memory_snapshot,
            conversation_referents=conversation_referents,
            runtime_context={
                "has_pending_action": has_pending_action,
                "session_title": session.title,
                "entrypoint": "knowledge",
                "command": request.command,
                "intent_hint": request.intent_hint,
            },
        )
        try:
            return self.agent_orchestrator.select_mode(payload)
        except Exception:
            return None

    def _payload(
        self,
        *,
        session: ChatSession,
        user_message: ChatMessage,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        memory_snapshot: MemorySnapshot,
        runtime_context: dict[str, Any],
        conversation_referents: dict[str, Any] | None = None,
    ) -> AgentOrchestratorInput:
        return AgentOrchestratorInput(
            session_id=session.id,
            message_id=user_message.id,
            user_prompt=request.content,
            selected_document_ids=selected_document_ids,
            attachments=attachments,
            conversation_referents=conversation_referents or {},
            memory_snapshot=memory_snapshot,
            available_tools=self.agent_orchestrator.available_tools(),
            available_skills=self.agent_orchestrator.available_skills(),
            runtime_context=runtime_context,
        )

    @staticmethod
    def _fast_path_capabilities(*, requires_tools: bool, target_runtime: str) -> list[str]:
        if not requires_tools:
            return []
        if target_runtime == "WorkspaceFileWriteNewRuntime":
            return ["deterministic_write_new"]
        if target_runtime == "WorkspaceFileOverwriteRuntime":
            return ["workspace_file_overwrite_confirmation"]
        if target_runtime == "WorkspaceCommandBoundaryRuntime":
            return ["workspace_command_execution_blocked"]
        return ["deterministic_read"]

    @staticmethod
    def _legacy_route(route: KnowledgeRoute | PaperDeskRoute | str) -> KnowledgeRoute:
        if isinstance(route, KnowledgeRoute):
            return route
        if isinstance(route, PaperDeskRoute):
            if route == PaperDeskRoute.DIRECT_CHAT:
                return KnowledgeRoute.DIRECT_ANSWER
            if route == PaperDeskRoute.EXPERIMENTAL_RESEARCH:
                return KnowledgeRoute.OPTIONAL_PLANNER
            if route in {PaperDeskRoute.WRITE_PENDING, PaperDeskRoute.WRITE_CONFIRMED}:
                return KnowledgeRoute.CONFIRMED_WRITE
            return KnowledgeRoute.TOOL_ACTION
        normalized = str(route).strip().casefold()
        if normalized in {"direct", "direct_answer", "direct_chat"}:
            return KnowledgeRoute.DIRECT_ANSWER
        if normalized in {"confirmed_write", "write_confirmed"}:
            return KnowledgeRoute.CONFIRMED_WRITE
        if normalized in {"planner", "optional_planner", "experimental_research"}:
            return KnowledgeRoute.OPTIONAL_PLANNER
        if normalized in {"reflection", "optional_reflection"}:
            return KnowledgeRoute.OPTIONAL_REFLECTION
        return KnowledgeRoute.TOOL_ACTION

    @staticmethod
    def _legacy_mode_for_route(route: KnowledgeRoute | PaperDeskRoute | str) -> AgentRunMode:
        if isinstance(route, KnowledgeRoute):
            if route == KnowledgeRoute.DIRECT_ANSWER:
                return AgentRunMode.DIRECT
            if route == KnowledgeRoute.OPTIONAL_PLANNER:
                return AgentRunMode.PLANNER
            if route == KnowledgeRoute.OPTIONAL_REFLECTION:
                return AgentRunMode.REFLECTION
            return AgentRunMode.REACT
        if isinstance(route, PaperDeskRoute):
            if route == PaperDeskRoute.DIRECT_CHAT:
                return AgentRunMode.DIRECT
            if route == PaperDeskRoute.EXPERIMENTAL_RESEARCH:
                return AgentRunMode.PLANNER
            return AgentRunMode.REACT
        normalized = str(route).strip().casefold()
        if normalized in {"direct", "direct_answer", "direct_chat", "paper_rag"}:
            return AgentRunMode.DIRECT
        if normalized in {"planner", "optional_planner", "experimental_research"}:
            return AgentRunMode.PLANNER
        if normalized in {"reflection", "optional_reflection"}:
            return AgentRunMode.REFLECTION
        return AgentRunMode.REACT

    @staticmethod
    def is_direct(decision: Any | None) -> bool:
        return bool(decision is not None and getattr(getattr(decision, "mode", None), "value", None) == AgentRunMode.DIRECT.value)

    @staticmethod
    def is_reviewable_tool_or_plan(decision: Any | None) -> bool:
        mode_value = getattr(getattr(decision, "mode", None), "value", None)
        return mode_value in {AgentRunMode.REACT.value, AgentRunMode.PLANNER.value}


class AgentModeExecutionAdapter:
    """Select legacy-compatible runtime executors behind the Agent runtime layer."""

    def __init__(
        self,
        *,
        agent_orchestrator: Any | None,
        tool_action_runtime_executor: Any,
        write_runtime_executor: Any,
        report_runtime_executor: Any,
        experimental_runtime_executor: Any,
    ) -> None:
        self.agent_orchestrator = agent_orchestrator
        self.tool_action_runtime_executor = tool_action_runtime_executor
        self.write_runtime_executor = write_runtime_executor
        self.report_runtime_executor = report_runtime_executor
        self.experimental_runtime_executor = experimental_runtime_executor

    def execute(
        self,
        *,
        decision: AgentModeDecision | None,
        route: PaperDeskRoute | None = None,
        execute_agent_mode: Callable[..., Any],
        is_report_runtime_request: Callable[[str], bool],
        has_pending_action: Callable[[str], bool],
        has_active_write_intent: Callable[[str], bool],
        **kwargs: Any,
    ) -> Any:
        request = kwargs["request"]
        session = kwargs["session"]
        if route == PaperDeskRoute.REPORT_ACTION:
            self._append_runtime_executor_trace(decision, "ReportRuntimeExecutor")
            return self.report_runtime_executor.run_agent_mode(
                execute_agent_mode=execute_agent_mode,
                decision=decision,
                **kwargs,
            )
        if route == PaperDeskRoute.EXPERIMENTAL_RESEARCH:
            self._append_runtime_executor_trace(decision, "ExperimentalRuntimeExecutor")
            return self.experimental_runtime_executor.run_agent_mode(
                execute_agent_mode=execute_agent_mode,
                decision=decision,
                **kwargs,
            )
        if route in {PaperDeskRoute.WRITE_PENDING, PaperDeskRoute.WRITE_CONFIRMED}:
            self._append_runtime_executor_trace(decision, "WriteRuntimeExecutor")
            return self.write_runtime_executor.run_pending_write(
                execute_agent_mode=execute_agent_mode,
                decision=decision,
                **kwargs,
            )
        if route in {PaperDeskRoute.LIBRARY_READ, PaperDeskRoute.TOOL_ACTION}:
            self._append_runtime_executor_trace(decision, "ToolActionRuntimeExecutor")
            return self.tool_action_runtime_executor.run_agent_mode(
                execute_agent_mode=execute_agent_mode,
                decision=decision,
                **kwargs,
            )
        if is_report_runtime_request(request.content):
            self._append_runtime_executor_trace(decision, "ReportRuntimeExecutor")
            return self.report_runtime_executor.run_agent_mode(
                execute_agent_mode=execute_agent_mode,
                decision=decision,
                **kwargs,
            )
        if decision is not None and decision.mode in {AgentRunMode.PLANNER, AgentRunMode.REFLECTION}:
            self._append_runtime_executor_trace(decision, "ExperimentalRuntimeExecutor")
            return self.experimental_runtime_executor.run_agent_mode(
                execute_agent_mode=execute_agent_mode,
                decision=decision,
                **kwargs,
            )
        if has_pending_action(session.id) or has_active_write_intent(request.content):
            self._append_runtime_executor_trace(decision, "WriteRuntimeExecutor")
            return self.write_runtime_executor.run_pending_write(
                execute_agent_mode=execute_agent_mode,
                decision=decision,
                **kwargs,
            )
        self._append_runtime_executor_trace(decision, "ToolActionRuntimeExecutor")
        return self.tool_action_runtime_executor.run_agent_mode(
            execute_agent_mode=execute_agent_mode,
            decision=decision,
            **kwargs,
        )

    def execute_legacy_mode(
        self,
        *,
        decision: Any | None,
        session: ChatSession,
        request: ChatMessageRequest,
        attachments: list[ChatAttachment],
        selected_document_ids: list[str],
        history: list[ChatMessage],
        has_pending_action: Callable[[str], bool],
        run_knowledge_agent: Callable[..., Any],
        knowledge_planner_runtime: Any | None,
        reflection_runtime: Any | None,
        append_trace: Callable[[str, str, str, dict[str, Any]], None],
    ) -> Any:
        if has_pending_action(session.id):
            return run_knowledge_agent(
                session=session,
                request=request,
                attachments=attachments,
                selected_document_ids=selected_document_ids,
                trace_id=getattr(decision, "trace_id", None),
            )
        if decision is None:
            return run_knowledge_agent(
                session=session,
                request=request,
                attachments=attachments,
                selected_document_ids=selected_document_ids,
            )
        mode_value = getattr(getattr(decision, "mode", None), "value", None)
        if mode_value == AgentRunMode.DIRECT.value:
            return None
        try:
            if mode_value == AgentRunMode.REACT.value:
                return run_knowledge_agent(
                    session=session,
                    request=request,
                    attachments=attachments,
                    selected_document_ids=selected_document_ids,
                    trace_id=decision.trace_id,
                )
            if mode_value == AgentRunMode.PLANNER.value and knowledge_planner_runtime is not None:
                return knowledge_planner_runtime.handle(
                    session=session,
                    request=request,
                    attachments=attachments,
                    selected_document_ids=selected_document_ids,
                    decision=decision,
                )
            if mode_value == AgentRunMode.REFLECTION.value and reflection_runtime is not None:
                return reflection_runtime.handle(
                    session=session,
                    request=request,
                    attachments=attachments,
                    selected_document_ids=selected_document_ids,
                    history=history,
                    decision=decision,
                )
        except Exception as exc:
            append_trace(
                decision.trace_id,
                "agent_mode_execution_failed",
                "Agent mode execution failed.",
                {"mode": mode_value, "error": str(exc)},
            )
        return None

    def _append_runtime_executor_trace(self, decision: AgentModeDecision | None, executor_name: str) -> None:
        if decision is None or self.agent_orchestrator is None:
            return
        self.agent_orchestrator.append_trace(
            decision.trace_id,
            status="route_runtime_executor_selected",
            message="Route runtime executor selected for legacy-compatible execution.",
            payload={
                "executor": executor_name,
                "mode": decision.mode.value,
                "route": decision.route.value,
                "target_runtime": decision.target_runtime,
            },
        )
