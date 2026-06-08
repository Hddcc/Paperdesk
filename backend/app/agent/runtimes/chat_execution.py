"""Runtime-owned chat and paper execution paths used by Agent Core."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


AnswerGenerator = Callable[..., tuple[str, Any]]


@dataclass(slots=True)
class RuntimeAnswerResult:
    """Answer text plus context state emitted by a route runtime executor."""

    content: str
    context_state: Any
    evidence_items: list[Any] | None = None
    citations: list[str] | None = None
    retrieval_status: str | None = None
    warning: str | None = None
    diagnostic: dict[str, Any] | None = None


@dataclass(slots=True)
class RuntimeLLMDiagnostic:
    """Provider call metadata recorded by runtime-owned LLM execution."""

    status: str
    error_type: str | None = None
    error: str | None = None
    model: str | None = None
    base_url_configured: bool = False
    api_key_configured: bool = False
    stream: bool = False

    def as_trace_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "error_type": self.error_type,
            "error": self.error,
            "model": self.model,
            "base_url_configured": self.base_url_configured,
            "api_key_configured": self.api_key_configured,
            "stream": self.stream,
        }


class RuntimeLLMClient:
    """Small LLM client facade owned by chat runtimes."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None,
        timeout: float = 30.0,
        openai_factory: Any | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.openai_factory = openai_factory or OpenAI
        self.last_diagnostic = RuntimeLLMDiagnostic(
            status="not_called",
            model=self.model,
            base_url_configured=bool(self.base_url),
            api_key_configured=bool(self.api_key),
        )

    def call(
        self,
        messages: list[dict],
        *,
        has_images: bool,
        delta_sink: Callable[[str], None] | None = None,
    ) -> str | None:
        if not self.api_key:
            self.last_diagnostic = RuntimeLLMDiagnostic(
                status="not_configured",
                model=self.model,
                base_url_configured=bool(self.base_url),
                api_key_configured=False,
                stream=delta_sink is not None,
            )
            return None
        try:
            client = self.openai_factory(
                api_key=self.api_key,
                base_url=self.base_url or None,
                timeout=self.timeout,
            )
            if delta_sink is not None:
                stream = client.chat.completions.create(
                    model=self.model,
                    temperature=0.3,
                    messages=messages,
                    stream=True,
                )
                parts: list[str] = []
                for item in stream:
                    choices = getattr(item, "choices", None)
                    if not choices:
                        continue
                    delta = getattr(choices[0], "delta", None)
                    content = delta.get("content") if isinstance(delta, dict) else getattr(delta, "content", None)
                    text = self._content_to_text(content)
                    if not text:
                        continue
                    parts.append(text)
                    delta_sink(text)
                result = "".join(parts).strip() if parts else None
                self.last_diagnostic = RuntimeLLMDiagnostic(
                    status="success" if result else "empty_response",
                    model=self.model,
                    base_url_configured=bool(self.base_url),
                    api_key_configured=True,
                    stream=True,
                )
                return result
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                messages=messages,
            )
        except Exception as exc:
            self.last_diagnostic = RuntimeLLMDiagnostic(
                status="error",
                error_type=type(exc).__name__,
                error=str(exc)[:500],
                model=self.model,
                base_url_configured=bool(self.base_url),
                api_key_configured=True,
                stream=delta_sink is not None,
            )
            if has_images:
                raise RuntimeError("当前模型未开启图片理解或图片请求失败，请检查视觉模型配置。") from exc
            return None

        choices = getattr(response, "choices", None)
        if not choices:
            self.last_diagnostic = RuntimeLLMDiagnostic(
                status="empty_response",
                model=self.model,
                base_url_configured=bool(self.base_url),
                api_key_configured=True,
                stream=False,
            )
            return None
        message = getattr(choices[0], "message", None)
        if message is None:
            self.last_diagnostic = RuntimeLLMDiagnostic(
                status="empty_response",
                model=self.model,
                base_url_configured=bool(self.base_url),
                api_key_configured=True,
                stream=False,
            )
            return None
        result = self._content_to_text(getattr(message, "content", None), strip=True)
        self.last_diagnostic = RuntimeLLMDiagnostic(
            status="success" if result else "empty_response",
            model=self.model,
            base_url_configured=bool(self.base_url),
            api_key_configured=True,
            stream=False,
        )
        return result

    @staticmethod
    def _content_to_text(content: Any, *, strip: bool = False) -> str | None:
        if isinstance(content, str):
            return content.strip() if strip else content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                if isinstance(text, str):
                    cleaned = text.strip() if strip else text
                    if cleaned:
                        parts.append(cleaned)
            return "\n".join(parts).strip() if parts else None
        return None


class ChatAnswerRuntime:
    """Shared prompt assembly and LLM invocation for chat-style runtimes."""

    def __init__(
        self,
        *,
        context_assembler: Any,
        llm_client: RuntimeLLMClient,
        knowledge_context_provider: Callable[..., list[str]],
        template_answer_builder: Callable[..., str],
    ) -> None:
        self.context_assembler = context_assembler
        self.llm_client = llm_client
        self.knowledge_context_provider = knowledge_context_provider
        self.template_answer_builder = template_answer_builder

    @property
    def last_diagnostic(self) -> RuntimeLLMDiagnostic:
        return self.llm_client.last_diagnostic

    def synthesize(
        self,
        *,
        session: Any,
        history: list[Any],
        current_request: Any,
        attachments: list[Any],
        memory_snapshot: Any,
        evidence_items: list[Any],
        session_file_context_lines: list[str] | None = None,
        delta_sink: Callable[[str], None] | None = None,
    ) -> RuntimeAnswerResult:
        prompt_messages, context_state = self.context_assembler.assemble(
            session=session,
            history=history,
            current_request=current_request,
            attachments=attachments,
            evidence_items=evidence_items,
            knowledge_context_lines=self.knowledge_context_provider(
                current_request=current_request,
                attachments=attachments,
                evidence_items=evidence_items,
            ),
            session_file_context_lines=session_file_context_lines or [],
        )
        try:
            response = self.llm_client.call(
                prompt_messages,
                has_images=any(item.kind == "image" for item in attachments),
                delta_sink=delta_sink,
            )
        except RuntimeError as exc:
            return RuntimeAnswerResult(
                content=str(exc),
                context_state=context_state,
                diagnostic=self.last_diagnostic.as_trace_payload(),
            )
        answer = response or self.template_answer_builder(
            content=current_request.content,
            attachments=attachments,
            evidence_items=evidence_items,
            model_available=bool(self.llm_client.api_key),
        )
        return RuntimeAnswerResult(
            content=answer,
            context_state=context_state,
            diagnostic=self.last_diagnostic.as_trace_payload(),
        )


class DirectChatRuntimeExecutor:
    """Single-turn direct chat response owner."""

    def __init__(self, answer_runtime: ChatAnswerRuntime | None = None) -> None:
        self.answer_runtime = answer_runtime

    def run(self, *, generate_answer: AnswerGenerator | None = None, **kwargs: Any) -> RuntimeAnswerResult:
        if self.answer_runtime is not None:
            kwargs.pop("evidence_items", None)
            return self.answer_runtime.synthesize(evidence_items=[], **kwargs)
        if generate_answer is None:
            raise ValueError("DirectChatRuntimeExecutor requires an answer runtime or generator")
        content, context_state = generate_answer(**kwargs)
        return RuntimeAnswerResult(content=content, context_state=context_state)


class PaperRagRuntimeExecutor:
    """Retrieve-then-synthesize paper RAG response owner."""

    def __init__(
        self,
        *,
        answer_runtime: ChatAnswerRuntime | None = None,
        rag_service: Any | None = None,
        citation_collector: Callable[[list[Any]], list[str]] | None = None,
    ) -> None:
        self.answer_runtime = answer_runtime
        self.rag_service = rag_service
        self.citation_collector = citation_collector

    def retrieve(
        self,
        *,
        question: str,
        documents: list[Any],
        top_k: int = 4,
    ) -> tuple[list[Any], str, str | None]:
        if not documents:
            return [], "skipped", None
        if self.rag_service is None:
            return [], "degraded", self.selected_document_retrieval_failed_message()
        try:
            return self.rag_service.retrieve_evidence(
                question=question,
                documents=documents,
                top_k=top_k,
            ), "ready", None
        except Exception:
            return [], "degraded", self.selected_document_retrieval_failed_message()

    def run(self, *, generate_answer: AnswerGenerator | None = None, **kwargs: Any) -> RuntimeAnswerResult:
        if self.answer_runtime is not None:
            result = self.answer_runtime.synthesize(**kwargs)
            evidence_items = list(kwargs.get("evidence_items") or [])
            result.evidence_items = evidence_items
            if self.citation_collector is not None:
                result.citations = self.citation_collector(evidence_items)
            return result
        if generate_answer is None:
            raise ValueError("PaperRagRuntimeExecutor requires an answer runtime or generator")
        content, context_state = generate_answer(**kwargs)
        return RuntimeAnswerResult(content=content, context_state=context_state)

    @staticmethod
    def selected_document_retrieval_failed_message() -> str:
        return (
            "论文正文向量检索服务当前不可用，无法完成基于所选论文正文的分析。"
            "请先启动 Milvus 向量服务并确认论文状态为 ready，然后重新发送问题。"
            "本轮不会改用普通聊天或仅凭论文元数据生成分析结论。"
        )


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
