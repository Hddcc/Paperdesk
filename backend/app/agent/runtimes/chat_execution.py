"""Runtime-owned answer execution wrappers used by ChatService."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


AnswerGenerator = Callable[..., tuple[str, Any]]


@dataclass(slots=True)
class RuntimeAnswerResult:
    """Answer text plus context state emitted by a route runtime executor."""

    content: str
    context_state: Any


class DirectChatRuntimeExecutor:
    """Single-turn direct chat response owner."""

    def run(self, *, generate_answer: AnswerGenerator, **kwargs: Any) -> RuntimeAnswerResult:
        content, context_state = generate_answer(**kwargs)
        return RuntimeAnswerResult(content=content, context_state=context_state)


class PaperRagRuntimeExecutor:
    """Retrieve-then-synthesize paper RAG response owner."""

    def run(self, *, generate_answer: AnswerGenerator, **kwargs: Any) -> RuntimeAnswerResult:
        content, context_state = generate_answer(**kwargs)
        return RuntimeAnswerResult(content=content, context_state=context_state)


class ToolActionRuntimeExecutor:
    """Bounded tool-action execution owner for library reads and ReAct calls."""

    def run_deterministic_read(self, *, deterministic_read: Callable[..., Any], **kwargs: Any) -> Any:
        return deterministic_read(**kwargs)

    def run_agent_mode(self, *, execute_agent_mode: Callable[..., Any], **kwargs: Any) -> Any:
        return execute_agent_mode(**kwargs)


class WriteRuntimeExecutor:
    """Preview-confirm-execute-verify execution owner for write routes."""

    def run_pending_write(self, *, execute_agent_mode: Callable[..., Any], **kwargs: Any) -> Any:
        return execute_agent_mode(**kwargs)

    def create_pending(self, *, create_pending: Callable[..., Any], **kwargs: Any) -> Any:
        return create_pending(**kwargs)


class ReportRuntimeExecutor:
    """Report service-workflow owner for report-producing Agent calls."""

    def run_agent_mode(self, *, execute_agent_mode: Callable[..., Any], **kwargs: Any) -> Any:
        return execute_agent_mode(**kwargs)


class WorkspaceRuntimeExecutor:
    """Workspace service-workflow owner for file reads and writes."""

    def boundary_message(self, *, message_builder: Callable[[], str]) -> str:
        return message_builder()

    def handle_pending_response(self, *, handler: Callable[..., Any], **kwargs: Any) -> Any:
        return handler(**kwargs)

    def create_file(self, *, create_file: Callable[..., Any], **kwargs: Any) -> Any:
        return create_file(**kwargs)

    def read_context(self, *, read_context: Callable[..., Any], **kwargs: Any) -> Any:
        return read_context(**kwargs)


class ExperimentalRuntimeExecutor:
    """Explicit experimental execution owner for planner and reflection routes."""

    def run_agent_mode(self, *, execute_agent_mode: Callable[..., Any], **kwargs: Any) -> Any:
        return execute_agent_mode(**kwargs)
