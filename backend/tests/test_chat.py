from datetime import datetime, timezone
from io import BytesIO
import time

import fitz

from app.services.embedding_service import EmbeddingService
from app.services.rag_service import RagService
from app.services.rag_service import RetrievalResult
from app.models import EvidenceQuality
from app.models import EvidenceItem
from app.models import AgentRunMode
from app.models import AgentModeDecision
from app.models import KnowledgeIntent
from app.models import KnowledgeRiskLevel
from app.models import KnowledgeRoute
from app.api.main import get_chat_service, get_knowledge_agent_runtime, get_repository
from app.models import ChunkRecord
from app.models import LibraryDocument
from app.runtime.knowledge_agent_runtime import KnowledgeAgentResult, KnowledgeAgentRuntime, _ReactAction
from app.runtime.agent_orchestrator import AgentOrchestrator, _ModeCandidate


def _install_fake_openai(monkeypatch, module_path: str, *, content: str | None = None, error: Exception | None = None):
    class FakeMessage:
        def __init__(self, value: str | None) -> None:
            self.content = value

    class FakeChoice:
        def __init__(self, value: str | None) -> None:
            self.message = FakeMessage(value)

    class FakeResponse:
        def __init__(self, value: str | None) -> None:
            self.choices = [FakeChoice(value)]

    class FakeCompletions:
        def create(self, **kwargs):
            _ = kwargs
            if error is not None:
                raise error
            return FakeResponse(content)

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            _ = kwargs
            self.chat = FakeChat()

    monkeypatch.setattr(module_path, FakeOpenAI)


def _trace_payloads(trace_id: str, status: str) -> list[dict]:
    return [
        trace.payload
        for trace in get_repository().runtime.list_traces(trace_id)
        if trace.status == status
    ]


def _build_pdf_bytes(text: str, *, title: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(
        fitz.Rect(72, 72, 520, 760),
        text,
        fontsize=11,
    )
    document.set_metadata({"title": title})
    return document.tobytes()


def _wait_for_document_status(client, document_id: str, expected_status: str, *, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get("/api/documents")
        assert response.status_code == 200
        for item in response.json():
            if item["id"] == document_id and item["status"] == expected_status:
                return item
        time.sleep(0.05)
    raise AssertionError(f"Document {document_id} did not reach status {expected_status}")


def _upload_document(client, monkeypatch):
    monkeypatch.setattr(
        EmbeddingService,
        "embed_texts",
        lambda self, texts: [[float(index + 1), 0.8, 0.2] for index, _ in enumerate(texts)],
    )
    monkeypatch.setattr(
        EmbeddingService,
        "embed_query",
        lambda self, query: [1.0, 0.8, 0.2],
    )
    response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "chat-rag.pdf",
                BytesIO(
                    _build_pdf_bytes(
                        "This paper explains retrieval grounding, attribution, and evaluation evidence. " * 8,
                        title="Chat RAG",
                    )
                ),
                "application/pdf",
            )
        },
    )
    assert response.status_code == 200
    return _wait_for_document_status(client, response.json()["id"], "ready")


def _create_ready_document(document_id: str, name: str, *, title: str | None = None) -> LibraryDocument:
    now = datetime.now(timezone.utc)
    document = LibraryDocument(
        id=document_id,
        filename=f"{document_id}_{name}",
        display_name=name,
        title=title or name.removesuffix(".pdf"),
        file_path=f"D:/virtual/{document_id}_{name}",
        sha256=(document_id.replace("-", "") + "0" * 64)[:64],
        page_count=10,
        status="ready",
        parser_status="indexed",
        indexed_at=now,
        version=1,
        created_at=now,
        uploaded_at=now,
    )
    get_repository().library.create_document(document)
    return document


def _add_abstract_chunk(document: LibraryDocument, abstract: str) -> None:
    get_repository().chunk.replace_document_chunks(
        document.id,
        [
            ChunkRecord(
                id=f"{document.id}-chunk-1",
                document_id=document.id,
                page_number=1,
                chunk_index=0,
                section="Abstract",
                title=document.title,
                text=abstract,
                content=abstract,
                token_estimate=len(abstract.split()),
            )
        ],
    )


def _add_metadata_chunk(document: LibraryDocument, *, venue: str | None = None, published: str | None = None, year: str | None = None) -> None:
    metadata = {}
    if venue is not None:
        metadata["journal"] = venue
    if published is not None:
        metadata["publication_date"] = published
    if year is not None:
        metadata["year"] = year
    get_repository().chunk.replace_document_chunks(
        document.id,
        [
            ChunkRecord(
                id=f"{document.id}-metadata-1",
                document_id=document.id,
                page_number=1,
                chunk_index=0,
                section="Metadata",
                title=document.title,
                text=f"Journal: {venue or ''}\nPublication Date: {published or ''}\nYear: {year or ''}",
                content=f"Journal: {venue or ''}\nPublication Date: {published or ''}\nYear: {year or ''}",
                token_estimate=24,
                metadata=metadata,
            )
        ],
    )


def test_chat_session_round_trip_and_memory_snapshot(client):
    create_response = client.post("/api/chat/sessions", json={"title": "新对话"})
    assert create_response.status_code == 200
    session = create_response.json()

    send_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "请用中文回答，并保留引用来源。RAG 评估通常关注什么？",
            "attachments": [],
            "selected_document_ids": [],
        },
    )
    assert send_response.status_code == 200
    payload = send_response.json()
    assert payload["assistant_message"]["content"]
    assistant = payload["assistant_message"]
    assert assistant["agent_trace_id"]
    assert assistant["action_status"] == "direct_completed"
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(trace.status == "agent_mode_selected" and trace.payload["mode"] == "DIRECT" for trace in traces)
    assert all(trace.status != "react_action_planned" for trace in traces)
    assert payload["memory_snapshot"]["items"]
    assert payload["context_state"]["budget_tokens"] > 0
    assert payload["context_state"]["stage"] == "normal"

    memory_response = client.get(f"/api/chat/sessions/{session['id']}/memory")
    assert memory_response.status_code == 200
    memory_items = memory_response.json()["items"]
    assert any("中文" in item["summary"] for item in memory_items)

    context_response = client.get(f"/api/chat/sessions/{session['id']}/context")
    assert context_response.status_code == 200
    assert context_response.json()["sources"]

    detail_response = client.get(f"/api/chat/sessions/{session['id']}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert len(detail["messages"]) == 2
    assert detail["context_state"]["budget_tokens"] > 0


def test_delete_chat_session_removes_it_from_list(client):
    create_response = client.post("/api/chat/sessions", json={"title": "待删除对话"})
    assert create_response.status_code == 200
    session = create_response.json()

    delete_response = client.delete(f"/api/chat/sessions/{session['id']}")
    assert delete_response.status_code == 200
    assert delete_response.json()["id"] == session["id"]

    list_response = client.get("/api/chat/sessions")
    assert list_response.status_code == 200
    assert all(item["id"] != session["id"] for item in list_response.json())

    detail_response = client.get(f"/api/chat/sessions/{session['id']}")
    assert detail_response.status_code == 404


def test_chat_uses_selected_library_documents(client, monkeypatch):
    document = _upload_document(client, monkeypatch)
    _install_fake_openai(
        monkeypatch,
        "app.runtime.knowledge_agent_runtime.OpenAI",
        content="The selected paper explains retrieval grounding, attribution, and evaluation evidence.",
    )
    get_knowledge_agent_runtime().api_key = "selected-doc-key"
    create_response = client.post("/api/chat/sessions", json={"title": "RAG 文档对话"})
    assert create_response.status_code == 200
    session_id = create_response.json()["id"]

    send_response = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={
            "content": "这篇论文写了什么？",
            "attachments": [],
            "selected_document_ids": [document["id"]],
        },
    )
    assert send_response.status_code == 200
    assistant = send_response.json()["assistant_message"]
    assert assistant["retrieval_status"] == "ready"
    assert assistant["citations"]
    assert assistant["used_document_ids"] == [document["id"]]


def test_general_multi_question_bundle_stays_direct_when_llm_suggests_react(client, monkeypatch):
    def fake_llm_candidate(self, payload):
        _ = self, payload
        return _ModeCandidate(
            mode=AgentRunMode.REACT,
            reason="LLM incorrectly routed ordinary bundled questions to tools.",
            confidence=0.95,
            target_runtime="KnowledgeAgentRuntime",
            source="llm",
        )

    monkeypatch.setattr(AgentOrchestrator, "_llm_candidate", fake_llm_candidate)
    session = client.post("/api/chat/sessions", json={"title": "多问题"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": (
                "东汉末年分为哪些国家？宋朝第二任皇帝是谁？宋朝和清朝哪个国土面积大？"
                "明朝是朱元璋在什么条件下建立的？锦衣卫是他设置的吗？"
                "清朝第一任皇帝是谁？他怎么推翻明朝的？"
            ),
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "direct_completed"
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(trace.status == "agent_mode_selected" and trace.payload["mode"] == "DIRECT" for trace in traces)
    assert all(trace.status != "react_action_planned" for trace in traces)


def test_direct_answer_success_uses_llm_output_for_general_question(client, monkeypatch):
    answer = (
        "\u79e6\u59cb\u7687\u7edf\u4e00\u516d\u56fd\u7684\u987a\u5e8f\u662f"
        "\u97e9\u3001\u8d75\u3001\u9b4f\u3001\u695a\u3001\u71d5\u3001\u9f50\uff1b"
        "\u6700\u96be\u901a\u5e38\u8ba4\u4e3a\u662f\u695a\u56fd\uff1b"
        "\u79e6\u59cb\u7687\u4e0b\u4e00\u4efb\u662f\u79e6\u4e8c\u4e16\u80e1\u4ea5\u3002"
    )
    _install_fake_openai(monkeypatch, "app.services.chat_service.OpenAI", content=answer)
    service = get_chat_service()
    service.api_key = "direct-success-key"
    service.base_url = "http://fake-llm"
    session = client.post("/api/chat/sessions", json={"title": "direct success"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": (
                "\u79e6\u59cb\u7687\u7edf\u4e00\u516d\u56fd\u7684\u987a\u5e8f\u662f\uff1f"
                "\u6700\u96be\u7684\u662f\u54ea\u4e2a\u56fd\u5bb6\uff1f"
                "\u79e6\u59cb\u7687\u4e0b\u4e00\u4efb\u662f\u8c01\uff1f"
            ),
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "direct_completed"
    assert "\u97e9\u3001\u8d75\u3001\u9b4f\u3001\u695a\u3001\u71d5\u3001\u9f50" in assistant["content"]
    assert "\u6ca1\u6709\u9644\u52a0\u77e5\u8bc6\u5e93\u8bc1\u636e" not in assistant["content"]
    payloads = _trace_payloads(assistant["agent_trace_id"], "direct_llm_call_finished")
    assert payloads
    assert payloads[-1]["status"] == "success"


def test_general_chat_fast_path_skips_router_and_uses_direct_llm(client, monkeypatch):
    answer = "Paris, France."
    _install_fake_openai(monkeypatch, "app.services.chat_service.OpenAI", content=answer)
    service = get_chat_service()
    service.api_key = "direct-fast-path-key"
    service.base_url = "http://fake-llm"

    def fail_select_mode(self, payload):
        _ = self, payload
        raise AssertionError("router should be skipped for pure general chat")

    monkeypatch.setattr(AgentOrchestrator, "select_mode", fail_select_mode)
    session = client.post("/api/chat/sessions", json={"title": "general fast path"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "What is the capital of France? Answer with the city and country.",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["content"] == answer
    assert assistant["action_status"] == "direct_completed"
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    selected = next(trace for trace in traces if trace.status == "agent_mode_selected")
    assert selected.payload["decision_source"] == "local_fast_path"
    assert selected.payload["mode"] == "DIRECT"
    assert any(trace.status == "direct_llm_call_finished" for trace in traces)
    assert all(trace.status != "react_action_planned" for trace in traces)


def test_fast_path_private_classifiers_cover_chinese_library_reads(client):
    service = get_chat_service()

    assert service._is_library_count_read("\u6211\u672c\u5730\u8bba\u6587\u5e93\u91cc\u6709\u51e0\u7bc7\u8bba\u6587\uff1f")
    assert service._has_library_or_paper_reference("\u6211\u672c\u5730\u8bba\u6587\u5e93\u91cc\u6709\u51e0\u7bc7\u8bba\u6587\uff1f")
    assert not service._can_skip_router_for_general_chat(
        content="\u6211\u672c\u5730\u8bba\u6587\u5e93\u91cc\u6709\u51e0\u7bc7\u8bba\u6587\uff1f",
        selected_document_ids=[],
        attachments=[],
    )


def test_selected_document_paper_question_does_not_use_general_chat_fast_path(client, monkeypatch):
    document = _create_ready_document("doc-fast-path-selected", "SelectedFastPath.pdf")
    called = {"router": False}

    def fake_select_mode(self, payload):
        called["router"] = True
        trace_id = self._begin_trace(payload)
        return AgentModeDecision(
            mode=AgentRunMode.REACT,
            route=KnowledgeRoute.TOOL_ACTION,
            intent=KnowledgeIntent.PAPER_QA,
            reason="selected paper question requires knowledge runtime",
            confidence=0.9,
            target_runtime="KnowledgeAgentRuntime",
            requires_tools=True,
            requires_rag=True,
            risk_level=KnowledgeRiskLevel.LOW,
            trace_id=trace_id,
        )

    def fake_execute_agent_mode(self, **kwargs):
        return KnowledgeAgentResult(
            content="grounded selected-paper answer",
            action_status="completed",
            retrieval_status="ready",
            used_document_ids=[document.id],
            agent_trace_id=kwargs["decision"].trace_id,
        )

    monkeypatch.setattr(AgentOrchestrator, "select_mode", fake_select_mode)
    monkeypatch.setattr(type(get_chat_service()), "_execute_agent_mode", fake_execute_agent_mode)
    session = client.post("/api/chat/sessions", json={"title": "selected not direct"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "\u8fd9\u7bc7\u8bba\u6587\u8bb2\u4ec0\u4e48\uff1f",
            "attachments": [],
            "selected_document_ids": [document.id],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert called["router"] is True
    assert assistant["action_status"] == "completed"
    assert assistant["content"] == "grounded selected-paper answer"


def test_library_grounded_question_does_not_use_general_chat_fast_path(client, monkeypatch):
    called = {"router": False}

    def fake_select_mode(self, payload):
        called["router"] = True
        trace_id = self._begin_trace(payload)
        return AgentModeDecision(
            mode=AgentRunMode.REACT,
            route=KnowledgeRoute.TOOL_ACTION,
            intent=KnowledgeIntent.PAPER_QA,
            reason="library-grounded question requires router",
            confidence=0.9,
            target_runtime="KnowledgeAgentRuntime",
            requires_tools=True,
            requires_rag=True,
            risk_level=KnowledgeRiskLevel.LOW,
            trace_id=trace_id,
        )

    def fake_execute_agent_mode(self, **kwargs):
        return KnowledgeAgentResult(
            content="library-grounded answer",
            action_status="completed",
            retrieval_status="ready",
            agent_trace_id=kwargs["decision"].trace_id,
        )

    monkeypatch.setattr(AgentOrchestrator, "select_mode", fake_select_mode)
    monkeypatch.setattr(type(get_chat_service()), "_execute_agent_mode", fake_execute_agent_mode)
    session = client.post("/api/chat/sessions", json={"title": "library grounded"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "\u6839\u636e\u6211\u7684\u8bba\u6587\u5e93\uff0c\u4f4e\u5149\u7167\u589e\u5f3a\u65b9\u5411\u6709\u54ea\u4e9b\u65b9\u6cd5\uff1f",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assert called["router"] is True
    assert response.json()["assistant_message"]["content"] == "library-grounded answer"


def test_metadata_read_fast_path_uses_template_without_react(client, monkeypatch):
    document = _create_ready_document("doc-fast-path-meta", "MetaFastPath.pdf", title="Meta Fast Path Paper")
    get_repository().chunk.replace_document_chunks(
        document.id,
        [
            ChunkRecord(
                id="chunk-fast-path-meta-1",
                document_id=document.id,
                page_number=1,
                chunk_index=0,
                text="metadata chunk",
                metadata={"authors": ["Alice Zhang", "Bob Li"]},
            )
        ],
    )

    def fail_select_mode(self, payload):
        _ = self, payload
        raise AssertionError("router should be skipped for pure metadata read")

    monkeypatch.setattr(AgentOrchestrator, "select_mode", fail_select_mode)
    session = client.post("/api/chat/sessions", json={"title": "metadata fast path"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "\u8fd9\u7bc7\u8bba\u6587\u7684\u6807\u9898\u3001\u4f5c\u8005\u548c\u9875\u6570\u662f\u591a\u5c11\uff1f",
            "attachments": [],
            "selected_document_ids": [document.id],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert "Meta Fast Path Paper" in assistant["content"]
    assert "\u4f5c\u8005\uff1aAlice Zhang\u3001Bob Li" in assistant["content"]
    assert "\u9875\u6570\uff1a10" in assistant["content"]
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(trace.status == "agent_mode_selected" and trace.payload["decision_source"] == "local_fast_path" for trace in traces)
    assert any(
        trace.status == "tool_call_log"
        and trace.payload["tool_name"] == "library.explorer.document_metadata"
        and trace.payload["io_type"] == "read"
        and trace.payload["write_type"] == "none"
        for trace in traces
    )
    assert any(
        trace.status == "deterministic_observation"
        and trace.payload["tool"] == "library.explorer.document_metadata"
        for trace in traces
    )
    assert all(trace.status != "react_action_planned" for trace in traces)


def test_category_stats_fast_path_uses_template_without_react(client, monkeypatch):
    doc_a = _create_ready_document("doc-fast-path-tag-a", "TagA.pdf", title="Tagged A")
    doc_b = _create_ready_document("doc-fast-path-tag-b", "TagB.pdf", title="Tagged B")
    tag = get_repository().category.create_category("\u7efc\u8ff0", "#0f5fb8")
    get_repository().category.replace_document_categories(doc_a.id, [tag.id])
    get_repository().category.replace_document_categories(doc_b.id, [tag.id])

    def fail_select_mode(self, payload):
        _ = self, payload
        raise AssertionError("router should be skipped for pure category stats read")

    monkeypatch.setattr(AgentOrchestrator, "select_mode", fail_select_mode)
    session = client.post("/api/chat/sessions", json={"title": "category fast path"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "\u5f53\u524d\u6709\u54ea\u4e9b\u6807\u7b7e\u5206\u7c7b\uff1f\u8bf7\u7ed9\u51fa\u6bcf\u4e2a\u6807\u7b7e\u6570\u91cf\u3002",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert "\u7efc\u8ff0" in assistant["content"]
    assert "2 \u7bc7" in assistant["content"]
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(trace.status == "agent_mode_selected" and trace.payload["decision_source"] == "local_fast_path" for trace in traces)
    assert any(
        trace.status == "tool_call_log"
        and trace.payload["tool_name"] == "library.explorer.category_stats"
        and trace.payload["io_type"] == "read"
        and trace.payload["write_type"] == "none"
        for trace in traces
    )
    assert any(
        trace.status == "deterministic_observation"
        and trace.payload["tool"] == "library.explorer.category_stats"
        for trace in traces
    )
    assert all(trace.status != "react_action_planned" for trace in traces)


def test_category_membership_fast_path_records_find_documents_trace(client, monkeypatch):
    document = _create_ready_document("doc-fast-path-baseline", "BaselinePaper.pdf", title="Baseline Paper")
    tag = get_repository().category.create_category("baseline", "#047c71")
    get_repository().category.replace_document_categories(document.id, [tag.id])

    def fail_select_mode(self, payload):
        _ = self, payload
        raise AssertionError("router should be skipped for pure category membership read")

    monkeypatch.setattr(AgentOrchestrator, "select_mode", fail_select_mode)
    session = client.post("/api/chat/sessions", json={"title": "category membership fast path"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "baseline \u6807\u7b7e\u4e0b\u6709\u54ea\u4e9b\u8bba\u6587\uff1f",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert "BaselinePaper.pdf" in assistant["content"]
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(
        trace.status == "tool_call_log"
        and trace.payload["tool_name"] == "library.explorer.find_documents"
        and trace.payload["io_type"] == "read"
        and trace.payload["write_type"] == "none"
        for trace in traces
    )
    assert all(trace.status != "react_action_planned" for trace in traces)


def test_category_read_then_explain_is_not_deterministic_fast_path(client, monkeypatch):
    document = _create_ready_document("doc-mixed-baseline", "MixedBaseline.pdf", title="Mixed Baseline")
    tag = get_repository().category.create_category("baseline", "#047c71")
    get_repository().category.replace_document_categories(document.id, [tag.id])
    called = {"router": False}

    def fake_select_mode(self, payload):
        called["router"] = True
        trace_id = self._begin_trace(payload)
        return AgentModeDecision(
            mode=AgentRunMode.REACT,
            route=KnowledgeRoute.TOOL_ACTION,
            intent=KnowledgeIntent.PAPER_QA,
            reason="mixed category read and explanation must use agent chain",
            confidence=0.9,
            target_runtime="KnowledgeAgentRuntime",
            requires_tools=True,
            risk_level=KnowledgeRiskLevel.LOW,
            trace_id=trace_id,
        )

    def fake_execute_agent_mode(self, **kwargs):
        return KnowledgeAgentResult(
            content="category list plus baseline explanation",
            action_status="completed",
            retrieval_status="skipped",
            used_document_ids=[document.id],
            agent_trace_id=kwargs["decision"].trace_id,
        )

    monkeypatch.setattr(AgentOrchestrator, "select_mode", fake_select_mode)
    monkeypatch.setattr(type(get_chat_service()), "_execute_agent_mode", fake_execute_agent_mode)
    session = client.post("/api/chat/sessions", json={"title": "mixed category explain"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "\u5217\u51fa baseline \u6807\u7b7e\u4e0b\u8bba\u6587\u6807\u9898\uff0c\u7136\u540e\u89e3\u91ca baseline \u662f\u4ec0\u4e48\u610f\u601d\u3002",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert called["router"] is True
    assert assistant["content"] == "category list plus baseline explanation"
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert all(trace.status != "deterministic_observation" for trace in traces)


def test_metadata_read_then_explain_is_not_deterministic_fast_path(client, monkeypatch):
    document = _create_ready_document("doc-mixed-meta", "MixedMeta.pdf", title="Mixed Meta")
    called = {"router": False}

    def fake_select_mode(self, payload):
        called["router"] = True
        trace_id = self._begin_trace(payload)
        return AgentModeDecision(
            mode=AgentRunMode.REACT,
            route=KnowledgeRoute.TOOL_ACTION,
            intent=KnowledgeIntent.PAPER_QA,
            reason="mixed metadata read and explanation must use agent chain",
            confidence=0.9,
            target_runtime="KnowledgeAgentRuntime",
            requires_tools=True,
            risk_level=KnowledgeRiskLevel.LOW,
            trace_id=trace_id,
        )

    def fake_execute_agent_mode(self, **kwargs):
        return KnowledgeAgentResult(
            content="page count plus super-resolution explanation",
            action_status="completed",
            retrieval_status="skipped",
            used_document_ids=[document.id],
            agent_trace_id=kwargs["decision"].trace_id,
        )

    monkeypatch.setattr(AgentOrchestrator, "select_mode", fake_select_mode)
    monkeypatch.setattr(type(get_chat_service()), "_execute_agent_mode", fake_execute_agent_mode)
    session = client.post("/api/chat/sessions", json={"title": "mixed metadata explain"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "\u5148\u7ed9\u51fa\u6240\u9009\u8bba\u6587\u9875\u6570\uff0c\u518d\u89e3\u91ca\u4ec0\u4e48\u662f super-resolution\u3002",
            "attachments": [],
            "selected_document_ids": [document.id],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert called["router"] is True
    assert assistant["content"] == "page count plus super-resolution explanation"
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert all(trace.status != "deterministic_observation" for trace in traces)


def test_library_count_then_general_abstract_is_not_deterministic_fast_path(client, monkeypatch):
    _create_ready_document("doc-mixed-count", "MixedCount.pdf")
    called = {"router": False}

    def fake_select_mode(self, payload):
        called["router"] = True
        trace_id = self._begin_trace(payload)
        return AgentModeDecision(
            mode=AgentRunMode.REACT,
            route=KnowledgeRoute.TOOL_ACTION,
            intent=KnowledgeIntent.PAPER_QA,
            reason="mixed library count and abstract explanation must use agent chain",
            confidence=0.9,
            target_runtime="KnowledgeAgentRuntime",
            requires_tools=True,
            risk_level=KnowledgeRiskLevel.LOW,
            trace_id=trace_id,
        )

    def fake_execute_agent_mode(self, **kwargs):
        return KnowledgeAgentResult(
            content="library count plus abstract explanation",
            action_status="completed",
            retrieval_status="skipped",
            agent_trace_id=kwargs["decision"].trace_id,
        )

    monkeypatch.setattr(AgentOrchestrator, "select_mode", fake_select_mode)
    monkeypatch.setattr(type(get_chat_service()), "_execute_agent_mode", fake_execute_agent_mode)
    session = client.post("/api/chat/sessions", json={"title": "mixed count abstract"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "\u5148\u56de\u7b54\uff1a\u8bba\u6587\u5e93\u91cc\u6709\u51e0\u7bc7\u8bba\u6587\uff1f\u518d\u8865\u5145\u4e00\u53e5\uff1a\u4ec0\u4e48\u662f\u8bba\u6587\u6458\u8981\uff1f",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert called["router"] is True
    assert assistant["content"] == "library count plus abstract explanation"
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert all(trace.status != "deterministic_observation" for trace in traces)


def test_report_request_with_metadata_words_is_not_deterministic_fast_path(client, monkeypatch):
    document = _create_ready_document("doc-report-meta-guard", "ReportMeta.pdf", title="Report Meta")
    called = {"router": False}

    def fake_select_mode(self, payload):
        called["router"] = True
        trace_id = self._begin_trace(payload)
        return AgentModeDecision(
            mode=AgentRunMode.REACT,
            route=KnowledgeRoute.TOOL_ACTION,
            intent=KnowledgeIntent.PAPER_QA,
            reason="report request must stay on agent/report chain",
            confidence=0.9,
            target_runtime="KnowledgeAgentRuntime",
            requires_tools=True,
            requires_rag=True,
            risk_level=KnowledgeRiskLevel.LOW,
            trace_id=trace_id,
        )

    def fake_execute_agent_mode(self, **kwargs):
        return KnowledgeAgentResult(
            content="report chain answer",
            action_status="completed",
            retrieval_status="ready",
            used_document_ids=[document.id],
            agent_trace_id=kwargs["decision"].trace_id,
        )

    monkeypatch.setattr(AgentOrchestrator, "select_mode", fake_select_mode)
    monkeypatch.setattr(type(get_chat_service()), "_execute_agent_mode", fake_execute_agent_mode)
    session = client.post("/api/chat/sessions", json={"title": "report metadata guard"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "\u4e3a\u6240\u9009\u8bba\u6587\u5199\u4e00\u6bb5\u7b80\u77ed\u62a5\u544a\uff0c\u5305\u542b\u6807\u9898\u3001\u6838\u5fc3\u4efb\u52a1\u548c\u4e00\u53e5\u8bc1\u636e\u4f9d\u636e\u3002",
            "attachments": [],
            "selected_document_ids": [document.id],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert called["router"] is True
    assert assistant["content"] == "report chain answer"
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert all(trace.status != "deterministic_observation" for trace in traces)


def test_read_then_write_request_is_not_collapsed_to_read_only_fast_path(client, monkeypatch):
    _create_ready_document("doc-fast-path-untagged-a", "UntaggedA.pdf", title="Untagged A")
    called = {"router": False}

    def fake_select_mode(self, payload):
        called["router"] = True
        trace_id = self._begin_trace(payload)
        return AgentModeDecision(
            mode=AgentRunMode.REACT,
            route=KnowledgeRoute.CONFIRMED_WRITE,
            intent=KnowledgeIntent.TAG_WRITE,
            reason="read-then-write must use the knowledge write path",
            confidence=0.9,
            target_runtime="KnowledgeAgentRuntime",
            requires_tools=True,
            requires_confirmation=True,
            risk_level=KnowledgeRiskLevel.HIGH,
            trace_id=trace_id,
        )

    def fake_execute_agent_mode(self, **kwargs):
        return KnowledgeAgentResult(
            content="read result and write preview required",
            action_status="confirmation_required",
            retrieval_status="skipped",
            agent_trace_id=kwargs["decision"].trace_id,
        )

    monkeypatch.setattr(AgentOrchestrator, "select_mode", fake_select_mode)
    monkeypatch.setattr(type(get_chat_service()), "_execute_agent_mode", fake_execute_agent_mode)
    session = client.post("/api/chat/sessions", json={"title": "read then write guard"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "\u6211\u6709\u51e0\u7bc7\u4e0d\u5e26\u6807\u7b7e\u7684\u8bba\u6587\uff1f\u5e2e\u6211\u628a\u8fd9\u4e9b\u8bba\u6587\u90fd\u52a0\u4e0a\u7efc\u8ff0\u6807\u7b7e\u3002",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert called["router"] is True
    assert assistant["action_status"] == "confirmation_required"
    assert assistant["content"] == "read result and write preview required"
    assert all(category.name != "\u7efc\u8ff0" for category in get_repository().category.list_categories())


def test_read_only_safety_request_does_not_enter_write_tool(client):
    get_repository().category.create_category("\u7a7a\u6807\u7b7e", "#0f5fb8")
    session = client.post("/api/chat/sessions", json={"title": "readonly safety"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "\u8bf7\u544a\u8bc9\u6211\u5f53\u524d\u7a7a\u6807\u7b7e\u5206\u7c7b\u6709\u54ea\u4e9b\uff0c\u4f46\u4e0d\u8981\u5220\u9664\u4efb\u4f55\u4e1c\u897f\u3002",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed"
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert all(
        not (
            trace.status == "react_observation"
            and str(trace.payload.get("tool", "")).startswith("library.operator.")
        )
        for trace in traces
    )


def test_write_request_is_not_general_chat_fast_path(client, monkeypatch):
    called = {"router": False}

    def fake_select_mode(self, payload):
        called["router"] = True
        trace_id = self._begin_trace(payload)
        return AgentModeDecision(
            mode=AgentRunMode.REACT,
            route=KnowledgeRoute.CONFIRMED_WRITE,
            intent=KnowledgeIntent.TAG_WRITE,
            reason="write request must use knowledge runtime",
            confidence=0.9,
            target_runtime="KnowledgeAgentRuntime",
            requires_tools=True,
            requires_confirmation=True,
            risk_level=KnowledgeRiskLevel.HIGH,
            trace_id=trace_id,
        )

    def fake_execute_agent_mode(self, **kwargs):
        return KnowledgeAgentResult(
            content="write preview required",
            action_status="confirmation_required",
            retrieval_status="skipped",
            agent_trace_id=kwargs["decision"].trace_id,
        )

    monkeypatch.setattr(AgentOrchestrator, "select_mode", fake_select_mode)
    monkeypatch.setattr(type(get_chat_service()), "_execute_agent_mode", fake_execute_agent_mode)
    session = client.post("/api/chat/sessions", json={"title": "write guard"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "\u7ed9\u6240\u6709\u8bba\u6587\u52a0\u4e0a\u7efc\u8ff0\u6807\u7b7e\u3002",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert called["router"] is True
    assert assistant["action_status"] == "confirmation_required"
    assert assistant["content"] == "write preview required"


def test_direct_answer_general_history_question_does_not_claim_library_limit(client, monkeypatch):
    answer = "秦始皇统一六国的顺序是韩、赵、魏、楚、燕、齐；秦始皇之后继位的是秦二世胡亥。"
    captured: dict = {}

    class FakeMessage:
        def __init__(self, value: str) -> None:
            self.content = value

    class FakeChoice:
        def __init__(self, value: str) -> None:
            self.message = FakeMessage(value)

    class FakeResponse:
        def __init__(self, value: str) -> None:
            self.choices = [FakeChoice(value)]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse(answer)

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            _ = kwargs
            self.chat = FakeChat()

    monkeypatch.setattr("app.services.chat_service.OpenAI", FakeOpenAI)
    service = get_chat_service()
    service.api_key = "direct-natural-key"
    service.base_url = "http://fake-llm"
    session = client.post("/api/chat/sessions", json={"title": "natural direct"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "秦始皇统一六国的顺序是？最难的是哪个国家？秦始皇下一任是谁？",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "direct_completed"
    assert "韩、赵、魏、楚、燕、齐" in assistant["content"]
    assert "知识库无法" not in assistant["content"]
    assert "无法直接回答" not in assistant["content"]
    assert "根据我的知识库" not in assistant["content"]
    system_prompt = captured["messages"][0]["content"]
    assert "PaperDesk 知识库运行态摘要" not in system_prompt
    assert "通用问答助手" in system_prompt


def test_direct_answer_llm_failure_is_observable(client, monkeypatch):
    _install_fake_openai(
        monkeypatch,
        "app.services.chat_service.OpenAI",
        error=RuntimeError("fake provider exploded"),
    )
    service = get_chat_service()
    service.api_key = "direct-failure-key"
    service.base_url = "http://fake-llm"
    session = client.post("/api/chat/sessions", json={"title": "direct failure"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "who was the next ruler after Qin Shi Huang?",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "direct_completed"
    assert "model call" in assistant["content"].lower() or "\u6a21\u578b\u8c03\u7528" in assistant["content"]
    payloads = _trace_payloads(assistant["agent_trace_id"], "direct_llm_call_finished")
    assert payloads
    assert payloads[-1]["status"] == "error"
    assert payloads[-1]["error_type"] == "RuntimeError"
    assert "fake provider exploded" in payloads[-1]["error"]


def test_selected_document_multi_question_drafts_answer_after_retrieval(client, monkeypatch):
    document = _create_ready_document("doc-compound-rag", "CompoundRag.pdf", title="Compound RAG Paper")
    _install_fake_openai(
        monkeypatch,
        "app.runtime.knowledge_agent_runtime.OpenAI",
        content="Compound RAG Paper studies retrieval grounding, answer synthesis, evaluation evidence, experiments, conclusions, limits, and improvements.",
    )
    get_knowledge_agent_runtime().api_key = "compound-draft-key"
    session = client.post("/api/chat/sessions", json={"title": "论文多问题"}).json()

    def fake_retrieve_evidence(self, **kwargs):
        _ = self, kwargs
        return [
            EvidenceItem(
                id="compound-evidence-1",
                source_type="local_document",
                source_id=document.id,
                title=document.title,
                snippet="This paper studies retrieval grounding, answer synthesis, and evaluation evidence.",
                citation_label="CompoundRag.pdf p.1",
                document_id=document.id,
                page_number=1,
                score=0.91,
            )
        ]

    monkeypatch.setattr(RagService, "retrieve_evidence", fake_retrieve_evidence)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": (
                "这篇论文解决什么问题？方法是什么？用了哪些证据？实验怎么做？"
                "结论是什么？局限是什么？还能怎么改进？"
            ),
            "attachments": [],
            "selected_document_ids": [document.id],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert assistant["content"] != "已检索到 1 条证据。"
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    observed_tools = [
        trace.payload.get("tool")
        for trace in traces
        if trace.status == "react_observation"
    ]
    assert "evidence.retriever.search" in observed_tools
    assert "report.drafter.write" in observed_tools


def test_selected_documents_innovation_question_drafts_answer_after_retrieval(client, monkeypatch):
    doc_a = _create_ready_document("doc-innovation-a", "BlindDiff.pdf", title="BlindDiff")
    doc_b = _create_ready_document("doc-innovation-b", "PAMI_LUT.pdf", title="PAMI LUT")
    doc_c = _create_ready_document("doc-innovation-c", "Deform-Mamba.pdf", title="Deform-Mamba")
    _install_fake_openai(
        monkeypatch,
        "app.runtime.knowledge_agent_runtime.OpenAI",
        content="BlindDiff, PAMI LUT, and Deform-Mamba each contribute distinct image restoration techniques grounded in the retrieved evidence.",
    )
    get_knowledge_agent_runtime().api_key = "innovation-draft-key"
    session = client.post("/api/chat/sessions", json={"title": "论文创新点"}).json()

    def fake_retrieve_evidence(self, **kwargs):
        _ = self
        return [
            EvidenceItem(
                id=f"innovation-{document.id}",
                source_type="local_document",
                source_id=document.id,
                title=document.title,
                snippet=f"{document.title} proposes a distinct technical contribution for image restoration.",
                citation_label=f"{document.filename} p.1",
                document_id=document.id,
                page_number=1,
                score=0.9,
            )
            for document in kwargs["documents"]
        ]

    monkeypatch.setattr(RagService, "retrieve_evidence", fake_retrieve_evidence)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "三篇论文各自的创新点分别是什么",
            "attachments": [],
            "selected_document_ids": [doc_a.id, doc_b.id, doc_c.id],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert assistant["content"] != "已检索到 3 条证据。"
    assert set(assistant["used_document_ids"]) == {doc_a.id, doc_b.id, doc_c.id}
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    observed_tools = [
        trace.payload.get("tool")
        for trace in traces
        if trace.status == "react_observation"
    ]
    assert "evidence.retriever.search" in observed_tools
    assert "report.drafter.write" in observed_tools


def test_selected_documents_review_success_uses_evidence_and_llm_draft(client, monkeypatch):
    doc_a = _create_ready_document("doc-review-success-a", "ReviewA.pdf", title="Review A")
    doc_b = _create_ready_document("doc-review-success-b", "ReviewB.pdf", title="Review B")
    session = client.post("/api/chat/sessions", json={"title": "review success"}).json()
    draft = (
        "The review synthesizes the adaptive retrieval method, the cross-document contribution, "
        "and the ablation experiment evidence from both selected papers."
    )
    _install_fake_openai(monkeypatch, "app.runtime.knowledge_agent_runtime.OpenAI", content=draft)
    runtime = get_knowledge_agent_runtime()
    runtime.api_key = "draft-success-key"
    runtime.base_url = "http://fake-llm"

    def fake_next_action(self, **kwargs):
        observations = kwargs["observations"]
        if not observations:
            return _ReactAction(
                "evidence.retriever.search",
                {"question": kwargs["content"], "document_ids": kwargs["selected_document_ids"]},
                "retrieve selected paper evidence",
            )
        return _ReactAction(
            "report.drafter.write",
            {"question": kwargs["content"], "document_ids": kwargs["selected_document_ids"]},
            "draft grounded review",
        )

    def fake_retrieve_evidence(self, **kwargs):
        documents = kwargs["documents"]
        return [
            EvidenceItem(
                id=f"review-success-{document.id}",
                source_type="local_document",
                source_id=document.id,
                title=document.title,
                snippet=(
                    f"{document.title} describes an adaptive retrieval method, "
                    "a concrete contribution, and controlled experiment results."
                ),
                citation_label=f"{document.filename} p.2",
                document_id=document.id,
                page_number=2,
                score=0.92,
                metadata={"section": "Introduction"},
            )
            for document in documents
        ]

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action", fake_next_action)
    monkeypatch.setattr(RagService, "retrieve_evidence", fake_retrieve_evidence)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "write a 1000 word review report based on these selected papers",
            "attachments": [],
            "selected_document_ids": [doc_a.id, doc_b.id],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert "adaptive retrieval method" in assistant["content"]
    assert "cross-document contribution" in assistant["content"]
    assert "ablation experiment" in assistant["content"]
    assert "##" not in assistant["content"] or "involved papers" not in assistant["content"].lower()
    payloads = _trace_payloads(assistant["agent_trace_id"], "react_observation")
    drafter_payload = next(
        payload["payload"]
        for payload in payloads
        if payload.get("tool") == "report.drafter.write"
    )
    assert drafter_payload["llm_draft_success"] is True
    assert drafter_payload["fallback_used"] is False
    assert drafter_payload["evidence_count"] == 2
    assert drafter_payload["used_document_count"] == 2


def test_selected_documents_review_with_evidence_marks_drafter_failure_degraded(client, monkeypatch):
    document = _create_ready_document("doc-review-draft-fails", "DraftFails.pdf", title="Draft Fails")
    session = client.post("/api/chat/sessions", json={"title": "review draft failure"}).json()
    _install_fake_openai(
        monkeypatch,
        "app.runtime.knowledge_agent_runtime.OpenAI",
        error=RuntimeError("draft provider unavailable"),
    )
    runtime = get_knowledge_agent_runtime()
    runtime.api_key = "draft-failure-key"
    runtime.base_url = "http://fake-llm"

    def fake_next_action(self, **kwargs):
        observations = kwargs["observations"]
        if not observations:
            return _ReactAction(
                "evidence.retriever.search",
                {"question": kwargs["content"], "document_ids": kwargs["selected_document_ids"]},
                "retrieve selected paper evidence",
            )
        return _ReactAction(
            "report.drafter.write",
            {"question": kwargs["content"], "document_ids": kwargs["selected_document_ids"]},
            "draft grounded review",
        )

    def fake_retrieve_evidence(self, **kwargs):
        _ = self, kwargs
        return [
            EvidenceItem(
                id="draft-failure-evidence",
                source_type="local_document",
                source_id=document.id,
                title=document.title,
                snippet="The paper contains real method and experiment evidence for a review.",
                citation_label="DraftFails.pdf p.4",
                document_id=document.id,
                page_number=4,
                score=0.9,
                metadata={"section": "Method"},
            )
        ]

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action", fake_next_action)
    monkeypatch.setattr(RagService, "retrieve_evidence", fake_retrieve_evidence)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "write a review report based on the selected paper",
            "attachments": [],
            "selected_document_ids": [document.id],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "degraded"
    assert assistant["retrieval_status"] == "degraded"
    payloads = _trace_payloads(assistant["agent_trace_id"], "react_observation")
    drafter_payload = next(
        payload["payload"]
        for payload in payloads
        if payload.get("tool") == "report.drafter.write"
    )
    assert drafter_payload["llm_draft_success"] is False
    assert drafter_payload["fallback_used"] is True
    assert "RuntimeError" in drafter_payload["drafting_error"]
    assert "draft provider unavailable" in drafter_payload["drafting_error"]


def test_knowledge_default_does_not_persist_subagent_tasks_for_selected_document_summary(client, monkeypatch):
    document = _create_ready_document("doc-no-subagent-default", "NoSubagentDefault.pdf", title="No Subagent Default")
    session = client.post("/api/chat/sessions", json={"title": "Subagent 闅旂"}).json()

    def fake_retrieve_evidence(self, **kwargs):
        _ = self, kwargs
        return [
            EvidenceItem(
                id="no-subagent-evidence",
                source_type="local_document",
                source_id=document.id,
                title=document.title,
                snippet="The paper provides grounded evidence for the requested summary.",
                citation_label="NoSubagentDefault.pdf p.1",
                document_id=document.id,
                page_number=1,
                score=0.9,
            )
        ]

    monkeypatch.setattr(RagService, "retrieve_evidence", fake_retrieve_evidence)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "鎬荤粨杩欑瘒璁烘枃",
            "attachments": [],
            "selected_document_ids": [document.id],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert get_repository().runtime.list_tasks(assistant["agent_trace_id"]) == []
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(
        trace.status == "knowledge_internal_step_started"
        and trace.payload.get("experimental_feature") == "subagent"
        and trace.payload.get("config_flag") == "ENABLE_SUBAGENT_EXECUTION=false"
        for trace in traces
    )


def test_selected_documents_metadata_question_uses_metadata_not_report_template(client):
    documents = [
        _create_ready_document(f"doc-meta-{index}", f"Meta{index}.pdf", title=f"Meta Paper {index}")
        for index in range(1, 6)
    ]
    _add_metadata_chunk(documents[0], venue="Journal of Vision", published="2024-05-01", year="2024")
    _add_metadata_chunk(documents[1], venue="CVPR", published="2023", year="2023")
    _add_metadata_chunk(documents[2], venue="Pattern Recognition", published="2022-11", year="2022")
    _add_metadata_chunk(documents[3], venue="TNNLS", published=None, year="2021")
    session = client.post("/api/chat/sessions", json={"title": "元数据字段"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "这五篇文章分别出自什么期刊？什么时间？",
            "attachments": [],
            "selected_document_ids": [document.id for document in documents],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    content = assistant["content"]
    assert assistant["action_status"] == "completed"
    assert "Journal of Vision" in content
    assert "CVPR" in content
    assert "Pattern Recognition" in content
    assert "TNNLS" in content
    assert "缺失" in content
    assert "摘要" not in content
    assert "初步总结" not in content
    assert "结论边界" not in content
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    observed_tools = [
        trace.payload.get("tool")
        for trace in traces
        if trace.status == "react_observation"
    ]
    assert "library.explorer.document_metadata" in observed_tools
    assert "report.drafter.write" not in observed_tools


def test_react_final_answer_synthesis_repairs_retrieval_status_only_answer(client, monkeypatch):
    doc_a = _create_ready_document("doc-status-only-a", "StatusOnlyA.pdf", title="Status Only A")
    doc_b = _create_ready_document("doc-status-only-b", "StatusOnlyB.pdf", title="Status Only B")
    _install_fake_openai(
        monkeypatch,
        "app.runtime.knowledge_agent_runtime.OpenAI",
        content="Status Only A and Status Only B both introduce an evidence-grounded contribution for restoration.",
    )
    get_knowledge_agent_runtime().api_key = "status-repair-key"
    session = client.post("/api/chat/sessions", json={"title": "检索后合成"}).json()

    def fake_next_action(self, **kwargs):
        observations = kwargs["observations"]
        if not observations:
            return _ReactAction(
                "evidence.retriever.search",
                {"question": kwargs["content"], "document_ids": kwargs["selected_document_ids"]},
                "retrieve evidence",
            )
        return _ReactAction("final.answer", {"content": "已检索到 2 条证据。"}, "premature status")

    def fake_retrieve_evidence(self, **kwargs):
        _ = self
        return [
            EvidenceItem(
                id=f"status-only-{document.id}",
                source_type="local_document",
                source_id=document.id,
                title=document.title,
                snippet=f"{document.title} introduces an evidence-grounded contribution for restoration.",
                citation_label=f"{document.filename} p.2",
                document_id=document.id,
                page_number=2,
                score=0.88,
            )
            for document in kwargs["documents"]
        ]

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action", fake_next_action)
    monkeypatch.setattr(RagService, "retrieve_evidence", fake_retrieve_evidence)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "这两篇论文各自说明了什么贡献",
            "attachments": [],
            "selected_document_ids": [doc_a.id, doc_b.id],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["content"] != "已检索到 2 条证据。"
    assert "Status Only A" in assistant["content"]
    assert "Status Only B" in assistant["content"]
    assert "evidence-grounded contribution" in assistant["content"]
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(trace.status == "final_answer_synthesis_started" for trace in traces)
    finished = [trace for trace in traces if trace.status == "final_answer_synthesis_finished"]
    assert finished
    assert finished[-1].payload["synthesis_used_evidence_count"] == 2


def test_chat_service_boundary_repairs_status_only_agent_result(client, monkeypatch):
    document = _create_ready_document("doc-service-synthesis", "ServiceSynthesis.pdf", title="Service Synthesis")
    session = client.post("/api/chat/sessions", json={"title": "服务边界合成"}).json()
    runtime = get_knowledge_agent_runtime()

    def fake_execute_agent_mode(self, **kwargs):
        _ = self, kwargs
        if not get_repository().research.get_run("chat-service-synthesis-trace"):
            get_repository().research.create_run("chat-service-synthesis-trace", "Chat service synthesis test")
        evidence = EvidenceItem(
            id="service-synthesis-evidence",
            source_type="local_document",
            source_id=document.id,
            title=document.title,
            snippet="Service-layer fallback evidence describes a concrete contribution.",
            citation_label="ServiceSynthesis.pdf p.3",
            document_id=document.id,
            page_number=3,
            score=0.9,
        )
        return KnowledgeAgentResult(
            content="retrieval ready",
            retrieval_status="ready",
            citations=[evidence.citation_label],
            used_document_ids=[document.id],
            evidence_items=[evidence],
            action_status="completed",
            agent_trace_id="chat-service-synthesis-trace",
        )

    monkeypatch.setattr(type(get_chat_service()), "_execute_agent_mode", fake_execute_agent_mode)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "这篇论文的贡献是什么",
            "attachments": [],
            "selected_document_ids": [document.id],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["content"] != "retrieval ready"
    assert "Service-layer fallback evidence" in assistant["content"]
    assert runtime.is_status_only_answer("已检索到 9 条证据。")


def test_selected_library_documents_do_not_route_direct_when_llm_suggests_direct(client, monkeypatch):
    document = _create_ready_document("doc-selected-direct-guard", "SelectedDirectGuard.pdf")
    _install_fake_openai(
        monkeypatch,
        "app.runtime.knowledge_agent_runtime.OpenAI",
        content="The selected paper discusses grounded paper analysis from local PDF chunks.",
    )
    get_knowledge_agent_runtime().api_key = "selected-route-key"
    session = client.post("/api/chat/sessions", json={"title": "选中文档路由"}).json()

    def fake_llm_candidate(self, payload):
        _ = self, payload
        return _ModeCandidate(
            mode=AgentRunMode.DIRECT,
            reason="LLM tried to answer selected-paper analysis as plain chat.",
            confidence=0.98,
            target_runtime="DirectChatRuntime",
            source="llm",
        )

    def fake_retrieve_evidence(self, **kwargs):
        _ = self, kwargs
        return [
            EvidenceItem(
                id="selected-direct-evidence-1",
                source_type="local_document",
                source_id=document.id,
                title=document.title,
                snippet="The selected paper discusses grounded paper analysis from local PDF chunks.",
                citation_label="SelectedDirectGuard.pdf p.1",
                document_id=document.id,
                page_number=1,
                score=0.9,
            )
        ]

    monkeypatch.setattr(AgentOrchestrator, "_llm_candidate", fake_llm_candidate)
    monkeypatch.setattr(RagService, "retrieve_evidence", fake_retrieve_evidence)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "一段话总结这篇论文",
            "attachments": [],
            "selected_document_ids": [document.id],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed"
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(trace.status == "agent_mode_selected" and trace.payload["mode"] == "REACT" for trace in traces)
    assert any(
        trace.status == "react_observation"
        and trace.payload.get("tool") == "evidence.retriever.search"
        for trace in traces
    )


def test_selected_documents_without_evidence_do_not_generate_metadata_only_summary(client, monkeypatch):
    document_a = _create_ready_document("doc-no-evidence-a", "NoEvidenceA.pdf", title="No Evidence A")
    document_b = _create_ready_document("doc-no-evidence-b", "NoEvidenceB.pdf", title="No Evidence B")
    session = client.post("/api/chat/sessions", json={"title": "无证据综述"}).json()

    monkeypatch.setattr(RagService, "retrieve_evidence", lambda self, **kwargs: [])

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "根据这两篇文章写一份1000字的综述报告",
            "attachments": [],
            "selected_document_ids": [document_a.id, document_b.id],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "needs_clarification"
    assert assistant["retrieval_status"] == "skipped"
    assert "没有检索到可引用的论文正文证据" in assistant["content"]
    assert "不能只根据文件名、标题或元数据生成综述报告" in assistant["content"]
    assert "# 根据这两篇文章写一份1000字的综述报告" not in assistant["content"]
    assert "## 涉及论文" not in assistant["content"]


def test_rag_uses_keyword_chunks_when_vector_store_is_unavailable(client):
    _ = client
    document = _create_ready_document("doc-keyword-vector-down", "KeywordVectorDown.pdf")
    _add_abstract_chunk(
        document,
        "Adaptive intervals for lookup tables improve low-light image enhancement with grounded evidence.",
    )

    class FailingVectorStore:
        def upsert_document(self, document):
            _ = document

        def add_chunks(self, chunks):
            _ = chunks

        def query_evidence(self, query, documents, top_k):
            _ = query, documents, top_k
            raise RuntimeError("milvus unavailable")

        def delete_document(self, document_id):
            _ = document_id

    service = RagService(
        library_repository=get_repository().library,
        chunk_repository=get_repository().chunk,
        vectorstore=FailingVectorStore(),
    )

    result = service.retrieve_evidence_with_quality(
        question="adaptive intervals lookup tables evidence",
        documents=[document],
        top_k=3,
    )

    assert result.evidence_items
    assert result.retrieval_strategy == "keyword_only_vector_unavailable"
    assert "vector_unavailable" in result.evidence_quality.warnings
    assert result.evidence_items[0].strategy == "keyword"


def test_rag_summary_rerank_prefers_body_sections_over_references(client):
    _ = client
    document = _create_ready_document("doc-rerank-sections", "RerankSections.pdf", title="Rerank Sections")
    reference_item = EvidenceItem(
        id="rerank-reference",
        source_type="local_document",
        source_id=document.id,
        title=document.title,
        snippet="REFERENCES [1] Prior work. Acknowledgement: funding support.",
        citation_label="RerankSections.pdf p.12",
        document_id=document.id,
        page_number=12,
        score=0.5,
        metadata={"section": "References"},
    )
    body_item = EvidenceItem(
        id="rerank-body",
        source_type="local_document",
        source_id=document.id,
        title=document.title,
        snippet=(
            "Introduction and Method: the paper proposes a contribution and reports "
            "experiment results with evaluation evidence."
        ),
        citation_label="RerankSections.pdf p.2",
        document_id=document.id,
        page_number=2,
        score=0.5,
        metadata={"section": "Method"},
    )

    ranked = RagService._rerank(
        [reference_item, body_item],
        documents=[document],
        question="write a review report for this paper",
    )

    assert ranked[0].id == "rerank-body"
    assert ranked[0].rerank_score > ranked[1].rerank_score


def test_chat_agent_answers_library_count_from_sqlite(client):
    _create_ready_document("doc-a12", "A12AAAAA.pdf")
    _create_ready_document("doc-b45", "B45BBB.pdf")
    session = client.post("/api/chat/sessions", json={"title": "论文库统计"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "我本地论文库里有几篇论文？",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert "共有 2 篇论文" in assistant["content"]
    assert assistant["action_status"] == "completed", assistant["content"]
    assert assistant["agent_trace_id"]
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    mode_trace = next(trace for trace in traces if trace.status == "agent_mode_selected")
    assert mode_trace.payload["mode"] == "DIRECT"
    assert mode_trace.payload["route"] == "ToolAction"
    assert mode_trace.payload["intent"] in {"paper_qa", "tag_query"}
    assert mode_trace.payload["requires_tools"] is True
    assert mode_trace.payload["decision_source"] == "local_fast_path"
    assert all(trace.status != "react_action_planned" for trace in traces)
    assert all(trace.status != "reflection_result_created" for trace in traces)


def test_chat_agent_reads_document_categories_as_tags(client):
    document = _create_ready_document("doc-a12", "A12AAAAA.pdf", title="A12 Adaptive Paper")
    category = get_repository().category.create_category("低光增强", "#0f5fb8")
    get_repository().category.replace_document_categories(document.id, [category.id])
    session = client.post("/api/chat/sessions", json={"title": "标签查询"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "A12什么的那篇论文有什么标签？",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert "低光增强" in assistant["content"]
    assert assistant["used_document_ids"] == [document.id]


def test_chat_agent_reads_all_document_category_links_instead_of_runtime_summary(client, monkeypatch):
    documents = [
        _create_ready_document("doc-all-tags-1", "WV-LUT.pdf"),
        _create_ready_document("doc-all-tags-2", "AdaInt.pdf"),
        _create_ready_document("doc-all-tags-3", "Zero-DCE.pdf"),
        _create_ready_document("doc-all-tags-4", "Medical-Mamba.pdf"),
        _create_ready_document("doc-all-tags-5", "Deform-Mamba.pdf"),
        _create_ready_document("doc-all-tags-6", "MambaIR.pdf"),
        _create_ready_document("doc-all-tags-7", "PAMI_LUT.pdf"),
        _create_ready_document("doc-all-tags-8", "Chao.pdf"),
    ]
    chinese = get_repository().category.create_category("中文", "#0f5fb8")
    french = get_repository().category.create_category("法语", "#047c71")
    hx23 = get_repository().category.create_category("hx23", "#6957d8")
    backend = get_repository().category.create_category("后端", "#b76a00")
    get_repository().category.replace_document_categories(documents[0].id, [backend.id])
    get_repository().category.replace_document_categories(documents[1].id, [hx23.id])
    get_repository().category.replace_document_categories(documents[2].id, [hx23.id])
    get_repository().category.replace_document_categories(documents[3].id, [hx23.id])
    get_repository().category.replace_document_categories(documents[4].id, [chinese.id, french.id])
    get_repository().category.replace_document_categories(documents[5].id, [french.id])
    get_repository().category.replace_document_categories(documents[6].id, [chinese.id])
    get_repository().category.replace_document_categories(documents[7].id, [chinese.id, french.id])
    session = client.post("/api/chat/sessions", json={"title": "每篇标签"}).json()
    runtime = get_knowledge_agent_runtime()
    monkeypatch.setattr(runtime, "api_key", "test-key")

    def fake_llm_action(self, **kwargs):
        _ = self, kwargs
        return _ReactAction("final.answer", {"content": "根据运行态摘要，法语和后端暂未关联到论文。"}, "bad final")

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", fake_llm_action)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "每篇文章对应的标签是什么？",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    content = assistant["content"]
    assert assistant["action_status"] == "completed"
    assert "运行态摘要" not in content
    assert "暂未关联" not in content
    assert "WV-LUT.pdf：后端" in content
    assert "Deform-Mamba.pdf：中文、法语" in content
    assert "MambaIR.pdf：法语" in content
    assert "Chao.pdf：中文、法语" in content
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(
        trace.status == "react_observation"
        and trace.payload.get("tool") == "library.explorer.document_categories"
        and len(trace.payload.get("payload", {}).get("documents", [])) == 8
        for trace in traces
    )


def test_chat_agent_resolves_existing_tag_entity_before_paper_candidates_for_report(client, monkeypatch):
    hx23 = get_repository().category.create_category("hx23", "#6957d8")
    docs = [
        _create_ready_document("doc-hx23-report-a", "HX23A.pdf", title="HX23 Method A"),
        _create_ready_document("doc-hx23-report-b", "HX23B.pdf", title="HX23 Method B"),
        _create_ready_document("doc-hx23-report-c", "HX23C.pdf", title="HX23 Method C"),
    ]
    other = _create_ready_document("doc-hx23-other", "Other.pdf", title="Other Paper")
    for document in docs:
        get_repository().category.replace_document_categories(document.id, [hx23.id])
        _add_abstract_chunk(document, f"{document.title} proposes a distinct contribution for tagged paper analysis.")
    _add_abstract_chunk(other, "This unrelated paper must not be selected by the hx23 tag request.")
    session = client.post("/api/chat/sessions", json={"title": "标签报告"}).json()
    monkeypatch.setattr(
        RagService,
        "retrieve_evidence_with_quality",
        lambda self, **kwargs: RetrievalResult(
            evidence_items=[],
            evidence_quality=EvidenceQuality(
                coverage_score=0.0,
                diversity_score=0.0,
                citation_score=0.0,
                relevance_score=0.0,
                warnings=["insufficient_evidence"],
            ),
            cache_hit=False,
        ),
    )

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "分析我的 hx23 标签的所有论文，写一份分析报告",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    content = assistant["content"]
    assert assistant["action_status"] == "degraded"
    assert "没有在论文库中唯一定位" not in content
    assert "已识别到「hx23」是标签/分类" in content
    assert "3 篇" in content
    assert "HX23A.pdf" in content
    assert "HX23B.pdf" in content
    assert "HX23C.pdf" in content
    assert "Other.pdf" not in content
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    find_trace = next(
        trace
        for trace in traces
        if trace.status == "react_observation"
        and trace.payload.get("tool") == "library.explorer.find_documents"
    )
    assert find_trace.payload["payload"]["category_lookup"]["matched_names"] == ["hx23"]
    assert set(find_trace.payload["payload"]["document_ids"]) == {document.id for document in docs}


def test_chat_agent_lists_existing_tag_documents_without_fuzzy_pdf_candidates(client):
    hx23 = get_repository().category.create_category("hx23", "#6957d8")
    docs = [
        _create_ready_document("doc-hx23-list-a", "ListA.pdf", title="List Paper A"),
        _create_ready_document("doc-hx23-list-b", "ListB.pdf", title="List Paper B"),
    ]
    other = _create_ready_document("doc-hx23-list-other", "ListOther.pdf", title="Other Paper")
    for document in docs:
        get_repository().category.replace_document_categories(document.id, [hx23.id])
    session = client.post("/api/chat/sessions", json={"title": "标签论文列表"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "hx23 标签下有哪些论文？",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    content = assistant["content"]
    assert "已识别到「hx23」是标签/分类，下面共有 2 篇" in content
    assert "ListA.pdf" in content
    assert "ListB.pdf" in content
    assert "ListOther.pdf" not in content
    assert "候选论文" not in content
    assert set(assistant["used_document_ids"]) == {document.id for document in docs}


def test_chat_agent_counts_existing_tag_documents(client):
    hx23 = get_repository().category.create_category("hx23", "#6957d8")
    docs = [
        _create_ready_document("doc-hx23-count-a", "CountA.pdf"),
        _create_ready_document("doc-hx23-count-b", "CountB.pdf"),
        _create_ready_document("doc-hx23-count-c", "CountC.pdf"),
    ]
    for document in docs:
        get_repository().category.replace_document_categories(document.id, [hx23.id])
    session = client.post("/api/chat/sessions", json={"title": "标签论文数量"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "统计 hx23 标签下有几篇论文",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    content = response.json()["assistant_message"]["content"]
    assert "已识别到「hx23」是标签/分类，下面共有 3 篇" in content
    assert "关联论文如下" not in content


def test_chat_agent_reports_missing_tag_entity_instead_of_pdf_candidates(client):
    get_repository().category.create_category("hx23", "#6957d8")
    _create_ready_document("doc-missing-tag-candidate", "abc999-method.pdf", title="abc999 Similar Paper")
    session = client.post("/api/chat/sessions", json={"title": "不存在标签"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "分析不存在的标签 abc999 下的论文",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "needs_clarification"
    assert "没有找到标签/分类「abc999」" in assistant["content"]
    assert "没有在论文库中唯一定位" not in assistant["content"]
    assert "abc999-method.pdf" not in assistant["content"]


def test_chat_agent_compares_two_existing_tag_collections(client, monkeypatch):
    tag_a = get_repository().category.create_category("A组", "#0f5fb8")
    tag_b = get_repository().category.create_category("B组", "#047c71")
    doc_a = _create_ready_document("doc-tag-a-compare", "TagA.pdf", title="Tag A Paper")
    doc_b = _create_ready_document("doc-tag-b-compare", "TagB.pdf", title="Tag B Paper")
    get_repository().category.replace_document_categories(doc_a.id, [tag_a.id])
    get_repository().category.replace_document_categories(doc_b.id, [tag_b.id])
    session = client.post("/api/chat/sessions", json={"title": "双标签对比"}).json()
    monkeypatch.setattr(
        RagService,
        "retrieve_evidence_with_quality",
        lambda self, **kwargs: RetrievalResult(
            evidence_items=[],
            evidence_quality=EvidenceQuality(
                coverage_score=0.0,
                diversity_score=0.0,
                citation_score=0.0,
                relevance_score=0.0,
                warnings=["insufficient_evidence"],
            ),
            cache_hit=False,
        ),
    )

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "对 A组 标签和 B组 标签下的论文做对比",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    content = response.json()["assistant_message"]["content"]
    assert "已识别到「A组、B组」是标签/分类" in content
    assert "TagA.pdf" in content
    assert "TagB.pdf" in content


def test_chat_agent_answers_tagged_count_and_lists_each_tagged_document(client):
    tagged_a = _create_ready_document("doc-tagged-list-a", "TaggedListA.pdf")
    tagged_b = _create_ready_document("doc-tagged-list-b", "TaggedListB.pdf")
    untagged = _create_ready_document("doc-tagged-list-c", "UntaggedListC.pdf")
    chinese = get_repository().category.create_category("中文", "#0f5fb8")
    backend = get_repository().category.create_category("后端", "#b76a00")
    get_repository().category.replace_document_categories(tagged_a.id, [chinese.id])
    get_repository().category.replace_document_categories(tagged_b.id, [chinese.id, backend.id])
    session = client.post("/api/chat/sessions", json={"title": "带标签明细"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "我有几篇带标签的文章？分别是什么",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    content = assistant["content"]
    assert assistant["action_status"] == "completed"
    assert "当前带分类/标签的论文有 2 篇" in content
    assert "TaggedListA.pdf：中文" in content
    assert "TaggedListB.pdf：中文、后端" in content
    assert "UntaggedListC.pdf" not in content
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(
        trace.status == "react_observation"
        and trace.payload.get("tool") == "library.explorer.category_stats"
        for trace in traces
    )
    document_category_trace = next(
        trace
        for trace in traces
        if trace.status == "react_observation"
        and trace.payload.get("tool") == "library.explorer.document_categories"
    )
    assert set(document_category_trace.payload["payload"]["document_ids"]) == {tagged_a.id, tagged_b.id}


def test_chat_agent_blocks_half_answer_when_compound_tag_query_has_unmet_obligation(client, monkeypatch):
    tagged_a = _create_ready_document("doc-compound-obligation-a", "CompoundTaggedA.pdf")
    tagged_b = _create_ready_document("doc-compound-obligation-b", "CompoundTaggedB.pdf")
    _create_ready_document("doc-compound-obligation-c", "CompoundUntagged.pdf")
    chinese = get_repository().category.create_category("中文", "#0f5fb8")
    backend = get_repository().category.create_category("后端", "#b76a00")
    get_repository().category.replace_document_categories(tagged_a.id, [chinese.id])
    get_repository().category.replace_document_categories(tagged_b.id, [backend.id])
    session = client.post("/api/chat/sessions", json={"title": "复合标签查询"}).json()
    runtime = get_knowledge_agent_runtime()
    monkeypatch.setattr(runtime, "api_key", "test-key")
    actions = iter(
        [
            _ReactAction("library.explorer.category_stats", {}, "先读取标签数量。"),
            _ReactAction("final.answer", {"content": "当前带分类/标签的论文有 2 篇。"}, "half answer"),
        ]
    )

    def fake_next_action_with_llm(self, **kwargs):
        _ = self, kwargs
        return next(actions)

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", fake_next_action_with_llm)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "我有几篇带标签的文章？分别是什么",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    content = assistant["content"]
    assert assistant["action_status"] == "completed"
    assert "当前带分类/标签的论文有 2 篇" in content
    assert "CompoundTaggedA.pdf：中文" in content
    assert "CompoundTaggedB.pdf：后端" in content
    assert "CompoundUntagged.pdf" not in content


def test_chat_agent_answers_multi_part_tag_stats_detail_extreme_and_repeat(client, monkeypatch):
    backend_doc = _create_ready_document("doc-multi-backend", "MultiBackend.pdf")
    hx_doc_a = _create_ready_document("doc-multi-hx-a", "MultiHxA.pdf")
    hx_doc_b = _create_ready_document("doc-multi-hx-b", "MultiHxB.pdf")
    untagged = _create_ready_document("doc-multi-untagged", "MultiUntagged.pdf")
    backend = get_repository().category.create_category("后端", "#b76a00")
    hx33 = get_repository().category.create_category("hx33", "#6957d8")
    get_repository().category.replace_document_categories(backend_doc.id, [backend.id])
    get_repository().category.replace_document_categories(hx_doc_a.id, [hx33.id])
    get_repository().category.replace_document_categories(hx_doc_b.id, [hx33.id])
    session = client.post("/api/chat/sessions", json={"title": "复合四问"}).json()
    runtime = get_knowledge_agent_runtime()
    monkeypatch.setattr(runtime, "api_key", "test-key")
    actions = iter(
        [
            _ReactAction("library.explorer.category_stats", {}, "读取标签统计。"),
            _ReactAction("library.explorer.document_categories", {}, "读取逐篇标签明细。"),
            _ReactAction("final.answer", {"content": "已完成。"}, "final"),
        ]
    )

    def fake_next_action_with_llm(self, **kwargs):
        _ = self, kwargs
        return next(actions)

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", fake_next_action_with_llm)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "我有几篇带标签的文章？分别是什么？我的哪种标签底下的论文最多？把这个标签重复输出5次",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    content = assistant["content"]
    assert assistant["action_status"] == "completed"
    assert "当前带分类/标签的论文有 3 篇" in content
    assert "MultiBackend.pdf：后端" in content
    assert "MultiHxA.pdf：hx33" in content
    assert "MultiHxB.pdf：hx33" in content
    assert "MultiUntagged.pdf" not in content
    assert "论文数量最多的标签/分类是：hx33（2 篇）" in content
    assert content.count("hx33") >= 8


def test_chat_agent_plans_compound_read_then_defers_destructive_category_delete(client, monkeypatch):
    chinese_docs = [
        _create_ready_document(f"doc-compound-delete-cn-{index}", f"CompoundDeleteChinese{index}.pdf")
        for index in range(4)
    ]
    backend_doc = _create_ready_document("doc-compound-delete-backend", "CompoundDeleteBackend.pdf")
    hx_docs = [
        _create_ready_document(f"doc-compound-delete-hx-{index}", f"CompoundDeleteHx{index}.pdf")
        for index in range(3)
    ]
    chinese = get_repository().category.create_category("中文", "#0f5fb8")
    backend = get_repository().category.create_category("后端", "#b76a00")
    hx33 = get_repository().category.create_category("hx33", "#6957d8")
    for document in chinese_docs:
        get_repository().category.replace_document_categories(document.id, [chinese.id])
    get_repository().category.replace_document_categories(backend_doc.id, [backend.id])
    for document in hx_docs:
        get_repository().category.replace_document_categories(document.id, [hx33.id])
    session = client.post("/api/chat/sessions", json={"title": "复合删除确认"}).json()
    runtime = get_knowledge_agent_runtime()
    monkeypatch.setattr(runtime, "api_key", "test-key")
    actions = iter(
        [
            _ReactAction("library.explorer.category_stats", {}, "读取标签统计。"),
            _ReactAction("library.explorer.document_categories", {}, "读取逐篇标签明细。"),
            _ReactAction("final.answer", {"content": "已完成安全部分。"}, "final"),
        ]
    )

    def fake_next_action_with_llm(self, **kwargs):
        _ = self, kwargs
        return next(actions)

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", fake_next_action_with_llm)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": (
                "我有几篇带标签的文章？分别是什么？我的哪种标签底下的论文最多？"
                "把这个标签重复输出5次，然后把这个标签的分类删掉，最后输出一个“好的喵”"
            ),
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    content = assistant["content"]
    assert assistant["action_status"] == "confirmation_required"
    assert payload["library_mutated"] is False
    assert "当前带分类/标签的论文有 8 篇" in content
    assert "论文数量最多的标签/分类是：中文（4 篇）" in content
    assert content.count("中文") >= 6
    assert "删除部分会删除「中文」" in content
    assert "好的喵" in content
    assert get_knowledge_agent_runtime().has_pending_action(session["id"]) is True
    assert get_repository().category.get_category(chinese.id) is not None

    confirm_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "确认删除",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert confirm_response.status_code == 200
    confirm_payload = confirm_response.json()
    confirm_assistant = confirm_payload["assistant_message"]
    assert confirm_assistant["action_status"] == "completed"
    assert confirm_payload["library_mutated"] is True
    assert get_repository().category.get_category(chinese.id) is None
    assert "已删除「中文」" in confirm_assistant["content"]
    assert "后端（1 篇）" in confirm_assistant["content"]
    assert "hx33（3 篇）" in confirm_assistant["content"]
    assert "中文（8 篇）" not in confirm_assistant["content"]
    assert confirm_assistant["content"].strip().endswith("好的喵")
    for document in chinese_docs:
        assert get_repository().get_document(document.id).categories == []


def test_chat_agent_creates_category_directly(client):
    session = client.post("/api/chat/sessions", json={"title": "新建分类"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "帮我新建一个分类：低光增强",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert "低光增强" in assistant["content"]
    assert assistant["action_status"] == "completed", assistant["content"]
    assert any(category.name == "低光增强" for category in get_repository().category.list_categories())


def test_chat_agent_answers_combined_library_and_tag_stats(client):
    doc_a = _create_ready_document("doc-a12", "A12AAAAA.pdf")
    doc_b = _create_ready_document("doc-b45", "B45BBB.pdf")
    _create_ready_document("doc-c90", "C90CCC.pdf")
    low_light = get_repository().category.create_category("低光增强", "#0f5fb8")
    mamba = get_repository().category.create_category("Mamba", "#047c71")
    get_repository().category.replace_document_categories(doc_a.id, [low_light.id])
    get_repository().category.replace_document_categories(doc_b.id, [mamba.id])
    session = client.post("/api/chat/sessions", json={"title": "标签统计"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "我的论文库里有几篇论文？有几篇有标签？我有几类标签？",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert "共有 3 篇论文" in assistant["content"]
    assert "有标签的论文 2 篇" in assistant["content"]
    assert "没有标签的论文 1 篇" in assistant["content"]
    assert "共有 2 类标签" in assistant["content"]
    assert "低光增强" in assistant["content"]
    assert "Mamba" in assistant["content"]
    assert assistant["action_status"] == "completed", assistant


def test_chat_routes_user_correction_to_reflection_runtime(client):
    session = client.post("/api/chat/sessions", json={"title": "反思修正"}).json()
    first_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "请简要解释一下 RAG。",
            "attachments": [],
            "selected_document_ids": [],
        },
    )
    assert first_response.status_code == 200

    second_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "你刚才回答不对，重新检查一下",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert second_response.status_code == 200
    assistant = second_response.json()["assistant_message"]
    assert assistant["agent_trace_id"]
    assert "重新" in assistant["content"]
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(trace.status == "agent_mode_selected" and trace.payload["mode"] == "REFLECTION" for trace in traces)
    assert any(trace.status == "reflection_started" for trace in traces)
    assert any(trace.status == "reflection_result_created" for trace in traces)


def test_library_count_fast_path_does_not_trigger_low_score_reflection(client, monkeypatch):
    calls = {"count": 0}

    def fake_run_react(self, **kwargs):
        _ = self
        calls["count"] += 1
        trace_id = kwargs.get("trace_id")
        if calls["count"] == 1:
            return KnowledgeAgentResult(
                content="我猜测论文库里有很多论文。",
                retrieval_status="skipped",
                action_status="completed",
                agent_trace_id=trace_id,
            )
        return KnowledgeAgentResult(
            content="补充检索后的回答：当前论文库统计已按工具结果重新核对。",
            retrieval_status="ready",
            used_document_ids=["doc-low"],
            action_status="completed",
            agent_trace_id=trace_id,
        )

    monkeypatch.setattr(KnowledgeAgentRuntime, "run_react", fake_run_react)
    session = client.post("/api/chat/sessions", json={"title": "低分反思"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "我的论文库里有几篇论文？",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert calls["count"] == 0
    assert "共有" in assistant["content"]
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(
        trace.status == "agent_mode_selected" and trace.payload.get("decision_source") == "local_fast_path"
        for trace in traces
    )
    assert all(trace.status != "react_action_planned" for trace in traces)
    assert all(trace.status != "reflection_improvement_started" for trace in traces)
    assert all(trace.status != "reflection_result_created" for trace in traces)
    assert not any("反思经验" in item["summary"] for item in payload["memory_snapshot"]["items"])


def test_chat_agent_creates_tag_and_assigns_only_untagged_documents(client):
    tagged = _create_ready_document("doc-tagged", "Tagged.pdf")
    untagged_a = _create_ready_document("doc-a12", "A12AAAAA.pdf")
    untagged_b = _create_ready_document("doc-b45", "B45BBB.pdf")
    existing = get_repository().category.create_category("已有标签", "#6957d8")
    get_repository().category.replace_document_categories(tagged.id, [existing.id])
    session = client.post("/api/chat/sessions", json={"title": "复合标签命令"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "帮我新增一个“hdcc”标签，并且把目前所有没有标签的论文都加上这个标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert "hdcc" in assistant["content"]
    assert "2 篇" in assistant["content"]

    categories = get_repository().category.list_categories()
    category_names = [category.name for category in categories]
    assert "hdcc" in category_names
    assert all("没有标签" not in name for name in category_names)
    assert all("并且" not in name for name in category_names)

    refreshed_tagged = get_repository().get_document(tagged.id)
    refreshed_a = get_repository().get_document(untagged_a.id)
    refreshed_b = get_repository().get_document(untagged_b.id)
    assert refreshed_tagged is not None
    assert refreshed_a is not None
    assert refreshed_b is not None
    assert [category.name for category in refreshed_tagged.categories] == ["已有标签"]
    assert [category.name for category in refreshed_a.categories] == ["hdcc"]
    assert [category.name for category in refreshed_b.categories] == ["hdcc"]


def test_chat_agent_assigns_untagged_documents_with_da_shang_phrase(client):
    untagged_a = _create_ready_document("doc-good-a", "GoodA.pdf")
    untagged_b = _create_ready_document("doc-good-b", "GoodB.pdf")
    tagged = _create_ready_document("doc-already", "Tagged.pdf")
    existing = get_repository().category.create_category("已有标签", "#6957d8")
    get_repository().category.replace_document_categories(tagged.id, [existing.id])
    session = client.post("/api/chat/sessions", json={"title": "打上标签"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "帮我把论文库里没有标签的论文都打上一个“好论文”标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert "好论文" in assistant["content"]
    assert [category.name for category in get_repository().get_document(untagged_a.id).categories] == ["好论文"]
    assert [category.name for category in get_repository().get_document(untagged_b.id).categories] == ["好论文"]
    assert [category.name for category in get_repository().get_document(tagged.id).categories] == ["已有标签"]


def test_chat_agent_reports_first_successful_assignment_when_llm_repeats_noop(client, monkeypatch):
    tagged = _create_ready_document("doc-hx-tagged", "HxTagged.pdf")
    untagged_a = _create_ready_document("doc-hx-a", "HxA.pdf")
    untagged_b = _create_ready_document("doc-hx-b", "HxB.pdf")
    untagged_c = _create_ready_document("doc-hx-c", "HxC.pdf")
    existing = get_repository().category.create_category("已有标签", "#6957d8")
    get_repository().category.replace_document_categories(tagged.id, [existing.id])
    session = client.post("/api/chat/sessions", json={"title": "重复工具调用"}).json()
    actions = iter(
        [
            _ReactAction(
                "library.operator.assign_category",
                {"category_name": "hx23", "scope": "untagged"},
                "给无标签论文加 hx23。",
            ),
            _ReactAction(
                "library.operator.assign_category",
                {"category_name": "hx23", "scope": "untagged"},
                "重复尝试同一写操作。",
            ),
        ]
    )

    def fake_next_action_with_llm(self, **kwargs):
        _ = self, kwargs
        return next(actions)

    runtime = get_knowledge_agent_runtime()
    monkeypatch.setattr(runtime, "api_key", "test-key")
    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", fake_next_action_with_llm)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把我没有标签的几篇文章加一个 hx23 标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert payload["library_mutated"] is True
    assert "hx23" in assistant["content"]
    assert "3 篇" in assistant["content"]
    assert "0 篇" not in assistant["content"]
    for document in (untagged_a, untagged_b, untagged_c):
        assert [category.name for category in get_repository().get_document(document.id).categories] == ["hx23"]
    assert [category.name for category in get_repository().get_document(tagged.id).categories] == ["已有标签"]


def test_chat_agent_treats_delete_words_inside_tag_name_as_non_destructive_assignment(client):
    untagged_a = _create_ready_document("doc-delete-tag-a", "DeleteTagA.pdf")
    untagged_b = _create_ready_document("doc-delete-tag-b", "DeleteTagB.pdf")
    session = client.post("/api/chat/sessions", json={"title": "标签名包含删除"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把无标签论文都加个“删除论文”标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert "确认删除" not in assistant["content"]
    assert any(category.name == "删除论文" for category in get_repository().category.list_categories())
    assert [category.name for category in get_repository().get_document(untagged_a.id).categories] == ["删除论文"]
    assert [category.name for category in get_repository().get_document(untagged_b.id).categories] == ["删除论文"]


def test_chat_agent_assigns_untagged_and_renames_category_without_delete_confirmation(client):
    untagged_a = _create_ready_document("doc-untagged-a", "UntaggedA.pdf")
    untagged_b = _create_ready_document("doc-untagged-b", "UntaggedB.pdf")
    chinese_doc_a = _create_ready_document("doc-chinese-a", "ChineseA.pdf")
    chinese_doc_b = _create_ready_document("doc-chinese-b", "ChineseB.pdf")
    chinese = get_repository().category.create_category("中文", "#0f5fb8")
    get_repository().category.replace_document_categories(chinese_doc_a.id, [chinese.id])
    get_repository().category.replace_document_categories(chinese_doc_b.id, [chinese.id])
    session = client.post("/api/chat/sessions", json={"title": "标签替换"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把所有没有标签的论文新增一个“删除论文”标签，然后把所有“中文”标签删除，换成“chinese”标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert payload["library_mutated"] is True
    assert "确认删除" not in assistant["content"]
    category_names = [category.name for category in get_repository().category.list_categories()]
    assert "删除论文" in category_names
    assert "chinese" in category_names
    assert "中文" not in category_names
    assert [category.name for category in get_repository().get_document(untagged_a.id).categories] == ["删除论文"]
    assert [category.name for category in get_repository().get_document(untagged_b.id).categories] == ["删除论文"]
    assert [category.name for category in get_repository().get_document(chinese_doc_a.id).categories] == ["chinese"]
    assert [category.name for category in get_repository().get_document(chinese_doc_b.id).categories] == ["chinese"]


def test_chat_agent_replaces_named_tag_through_tool_runtime_when_llm_routes_direct(client, monkeypatch):
    hx_doc_a = _create_ready_document("doc-replace-hx-a", "ReplaceHxA.pdf")
    hx_doc_b = _create_ready_document("doc-replace-hx-b", "ReplaceHxB.pdf")
    other = _create_ready_document("doc-replace-other", "ReplaceOther.pdf")
    hx23 = get_repository().category.create_category("hx23", "#6957d8")
    french = get_repository().category.create_category("法语", "#047c71")
    get_repository().category.replace_document_categories(hx_doc_a.id, [hx23.id])
    get_repository().category.replace_document_categories(hx_doc_b.id, [hx23.id, french.id])
    get_repository().category.replace_document_categories(other.id, [french.id])
    session = client.post("/api/chat/sessions", json={"title": "标签换名"}).json()
    runtime = get_knowledge_agent_runtime()
    monkeypatch.setattr(runtime, "api_key", "test-openai-compatible-key")

    def fake_llm_candidate(self, payload):
        _ = self, payload
        return _ModeCandidate(
            mode=AgentRunMode.DIRECT,
            reason="LLM incorrectly treated tag replacement as plain chat.",
            confidence=0.95,
            target_runtime="DirectChatRuntime",
            source="llm",
        )

    def fake_llm_action(self, **kwargs):
        _ = self, kwargs
        return _ReactAction(
            "library.operator.rename_category",
            {"source_category_name": "hx23", "target_category_name": "更新"},
            "LLM selected rename_category for replacement.",
        )

    monkeypatch.setattr(AgentOrchestrator, "_llm_candidate", fake_llm_candidate)
    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", fake_llm_action)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把里面带有hx23标签的论文，标签都换成“更新”",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "completed", assistant["content"]
    assert payload["library_mutated"] is True
    assert "hx23" in assistant["content"]
    assert "更新" in assistant["content"]
    assert "2 篇" in assistant["content"]
    category_names = [category.name for category in get_repository().category.list_categories()]
    assert "更新" in category_names
    assert "hx23" not in category_names
    assert [category.name for category in get_repository().get_document(hx_doc_a.id).categories] == ["更新"]
    assert [category.name for category in get_repository().get_document(hx_doc_b.id).categories] == ["法语", "更新"]
    assert [category.name for category in get_repository().get_document(other.id).categories] == ["法语"]


def test_chat_agent_replaces_named_tag_with_alternate_wording(client, monkeypatch):
    source_doc = _create_ready_document("doc-replace-topic-a", "ReplaceTopicA.pdf")
    untouched_doc = _create_ready_document("doc-replace-topic-b", "ReplaceTopicB.pdf")
    source = get_repository().category.create_category("旧主题", "#6957d8")
    other = get_repository().category.create_category("其它", "#047c71")
    get_repository().category.replace_document_categories(source_doc.id, [source.id])
    get_repository().category.replace_document_categories(untouched_doc.id, [other.id])
    session = client.post("/api/chat/sessions", json={"title": "标签替换同义词"}).json()
    runtime = get_knowledge_agent_runtime()
    monkeypatch.setattr(runtime, "api_key", "test-openai-compatible-key")

    def fake_llm_action(self, **kwargs):
        _ = self, kwargs
        return _ReactAction(
            "library.operator.rename_category",
            {"source_category_name": "旧主题", "target_category_name": "新主题"},
            "LLM selected rename_category for replacement.",
        )

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", fake_llm_action)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "请把带有旧主题标签的论文，标签都替换成「新主题」",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "completed", assistant["content"]
    assert payload["library_mutated"] is True
    category_names = [category.name for category in get_repository().category.list_categories()]
    assert "新主题" in category_names
    assert "旧主题" not in category_names
    assert [category.name for category in get_repository().get_document(source_doc.id).categories] == ["新主题"]
    assert [category.name for category in get_repository().get_document(untouched_doc.id).categories] == ["其它"]


def test_chat_agent_clears_all_categories_requires_strong_confirmation(client):
    doc_a = _create_ready_document("doc-clear-a", "ClearA.pdf")
    doc_b = _create_ready_document("doc-clear-b", "ClearB.pdf")
    category = get_repository().category.create_category("中文", "#0f5fb8")
    get_repository().category.replace_document_categories(doc_a.id, [category.id])
    get_repository().category.replace_document_categories(doc_b.id, [category.id])
    session = client.post("/api/chat/sessions", json={"title": "清空分类"}).json()

    clear_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "清空所有论文的标签和分类",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert clear_response.status_code == 200
    clear_payload = clear_response.json()
    clear_assistant = clear_payload["assistant_message"]
    assert clear_assistant["action_status"] == "confirmation_required"
    assert clear_payload["library_mutated"] is False
    assert "确认清空所有标签" in clear_assistant["content"]
    assert [category.name for category in get_repository().get_document(doc_a.id).categories] == ["中文"]
    assert [category.name for category in get_repository().get_document(doc_b.id).categories] == ["中文"]

    weak_confirm = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "确认",
            "attachments": [],
            "selected_document_ids": [],
        },
    )
    assert weak_confirm.status_code == 200
    weak_assistant = weak_confirm.json()["assistant_message"]
    assert weak_assistant["action_status"] == "confirmation_required"
    assert [category.name for category in get_repository().get_document(doc_a.id).categories] == ["中文"]
    assert [category.name for category in get_repository().get_document(doc_b.id).categories] == ["中文"]

    confirm_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "确认清空所有标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )
    assert confirm_response.status_code == 200
    clear_payload = confirm_response.json()
    clear_assistant = clear_payload["assistant_message"]
    assert clear_assistant["action_status"] == "completed"
    assert clear_payload["library_mutated"] is True
    assert "已按确认范围清空 2 篇论文的分类/标签关系" in clear_assistant["content"]
    assert get_repository().get_document(doc_a.id).categories == []
    assert get_repository().get_document(doc_b.id).categories == []

    count_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "我有几篇带标签的文章？",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert count_response.status_code == 200
    count_assistant = count_response.json()["assistant_message"]
    assert count_assistant["action_status"] == "completed"
    assert count_assistant["content"] == "当前带分类/标签的论文有 0 篇。"
    # Context may be used for referent resolution, but final user-facing answers should not expose process wording.
    assert "根据刚刚" not in count_assistant["content"]
    assert "根据上下文" not in count_assistant["content"]
    assert "运行态摘要" not in count_assistant["content"]
    assert "对话记录" not in count_assistant["content"]


def test_chat_agent_clears_categories_for_documents_with_named_tag(client, monkeypatch):
    french_a = _create_ready_document("doc-french-a", "FrenchA.pdf")
    french_b = _create_ready_document("doc-french-b", "FrenchB.pdf")
    other = _create_ready_document("doc-english", "English.pdf")
    french = get_repository().category.create_category("法语", "#0f5fb8")
    english = get_repository().category.create_category("英语", "#047c71")
    get_repository().category.replace_document_categories(french_a.id, [french.id])
    get_repository().category.replace_document_categories(french_b.id, [french.id, english.id])
    get_repository().category.replace_document_categories(other.id, [english.id])
    session = client.post("/api/chat/sessions", json={"title": "按标签清空"}).json()

    def fake_next_action_with_llm(self, **kwargs):
        _ = self, kwargs
        return _ReactAction(
            "library.operator.clear_categories",
            {"category_name": "法语"},
            "清空带有法语标签论文的标签。",
        )

    runtime = get_knowledge_agent_runtime()
    monkeypatch.setattr(runtime, "api_key", "test-key")
    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", fake_next_action_with_llm)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "帮我把所有文章中带有“法语”这个标签的论文的标签清空",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "confirmation_required"
    assert payload["library_mutated"] is False
    assert "确认移除法语标签" in assistant["content"]
    assert [category.name for category in get_repository().get_document(french_a.id).categories] == ["法语"]
    assert [category.name for category in get_repository().get_document(french_b.id).categories] == ["法语", "英语"]
    assert [category.name for category in get_repository().get_document(other.id).categories] == ["英语"]

    confirm_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "确认移除法语标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )
    assert confirm_response.status_code == 200
    payload = confirm_response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert payload["library_mutated"] is True
    assert "已从 2 篇论文中移除标签/分类「法语」" in assistant["content"]
    assert get_repository().get_document(french_a.id).categories == []
    assert [category.name for category in get_repository().get_document(french_b.id).categories] == ["英语"]
    assert [category.name for category in get_repository().get_document(other.id).categories] == ["英语"]


def test_chat_agent_rejects_empty_clear_category_tool_args(client, monkeypatch):
    category = get_repository().category.create_category("空参数保护", "#0f5fb8")
    document = _create_ready_document("doc-empty-clear-guard", "EmptyClearGuard.pdf")
    get_repository().category.replace_document_categories(document.id, [category.id])
    session = client.post("/api/chat/sessions", json={"title": "空参数保护"}).json()

    def fake_next_action_with_llm(self, **kwargs):
        _ = self, kwargs
        return _ReactAction(
            "library.operator.clear_categories",
            {},
            "Bad LLM action with empty destructive args.",
        )

    runtime = get_knowledge_agent_runtime()
    monkeypatch.setattr(runtime, "api_key", "test-key")
    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", fake_next_action_with_llm)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "清空标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "validation_failed"
    assert payload["library_mutated"] is False
    assert [item.name for item in get_repository().get_document(document.id).categories] == ["空参数保护"]


def test_chat_agent_blocks_all_scope_when_user_named_single_tag(client, monkeypatch):
    target = get_repository().category.create_category("范围保护", "#0f5fb8")
    other_tag = get_repository().category.create_category("其它标签", "#047c71")
    target_doc = _create_ready_document("doc-scope-guard-target", "ScopeTarget.pdf")
    other_doc = _create_ready_document("doc-scope-guard-other", "ScopeOther.pdf")
    get_repository().category.replace_document_categories(target_doc.id, [target.id, other_tag.id])
    get_repository().category.replace_document_categories(other_doc.id, [other_tag.id])
    session = client.post("/api/chat/sessions", json={"title": "范围保护"}).json()

    def fake_next_action_with_llm(self, **kwargs):
        _ = self, kwargs
        return _ReactAction(
            "library.operator.clear_categories",
            {"operation": "clear_all_categories", "scope": "all"},
            "Bad LLM action widened a single-tag request to all tags.",
        )

    runtime = get_knowledge_agent_runtime()
    monkeypatch.setattr(runtime, "api_key", "test-key")
    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", fake_next_action_with_llm)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "清空范围保护标签下论文的标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "validation_failed"
    assert payload["library_mutated"] is False
    assert [item.name for item in get_repository().get_document(target_doc.id).categories] == ["范围保护", "其它标签"]
    assert [item.name for item in get_repository().get_document(other_doc.id).categories] == ["其它标签"]


def test_chat_agent_compound_rename_then_preview_clear_other_tagged_documents(client):
    source = get_repository().category.create_category("A复合", "#0f5fb8")
    target_existing = get_repository().category.create_category("保留标签", "#047c71")
    other = get_repository().category.create_category("其它标签", "#6957d8")
    source_doc_a = _create_ready_document("doc-compound-source-a", "CompoundSourceA.pdf")
    source_doc_b = _create_ready_document("doc-compound-source-b", "CompoundSourceB.pdf")
    other_doc = _create_ready_document("doc-compound-other", "CompoundOther.pdf")
    untagged_doc = _create_ready_document("doc-compound-untagged", "CompoundUntagged.pdf")
    get_repository().category.replace_document_categories(source_doc_a.id, [source.id])
    get_repository().category.replace_document_categories(source_doc_b.id, [source.id, target_existing.id])
    get_repository().category.replace_document_categories(other_doc.id, [other.id])
    session = client.post("/api/chat/sessions", json={"title": "复合写操作"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把所有 A复合 标签换成 0516，然后把其他带有标签的论文的标签都清空",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "confirmation_required"
    assert payload["library_mutated"] is True
    assert "已完成第 1 步" in assistant["content"]
    assert "A复合" in assistant["content"]
    assert "0516" in assistant["content"]
    assert "确认清空其他有标签论文的标签" in assistant["content"]
    assert [item.name for item in get_repository().get_document(source_doc_a.id).categories] == ["0516"]
    assert [item.name for item in get_repository().get_document(source_doc_b.id).categories] == ["保留标签", "0516"]
    assert [item.name for item in get_repository().get_document(other_doc.id).categories] == ["其它标签"]
    assert get_repository().get_document(untagged_doc.id).categories == []

    confirm = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "确认清空其他有标签论文的标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert confirm.status_code == 200
    confirm_payload = confirm.json()
    confirm_assistant = confirm_payload["assistant_message"]
    assert confirm_assistant["action_status"] == "completed"
    assert confirm_payload["library_mutated"] is True
    assert [item.name for item in get_repository().get_document(source_doc_a.id).categories] == ["0516"]
    assert [item.name for item in get_repository().get_document(source_doc_b.id).categories] == ["保留标签", "0516"]
    assert get_repository().get_document(other_doc.id).categories == []
    assert get_repository().get_document(untagged_doc.id).categories == []


def test_chat_agent_compound_rename_then_counts_target_tag(client):
    source = get_repository().category.create_category("A统计", "#0f5fb8")
    existing_target = get_repository().category.create_category("B统计", "#047c71")
    source_doc_a = _create_ready_document("doc-compound-count-source-a", "CompoundCountSourceA.pdf")
    source_doc_b = _create_ready_document("doc-compound-count-source-b", "CompoundCountSourceB.pdf")
    existing_target_doc = _create_ready_document("doc-compound-count-target", "CompoundCountTarget.pdf")
    get_repository().category.replace_document_categories(source_doc_a.id, [source.id])
    get_repository().category.replace_document_categories(source_doc_b.id, [source.id])
    get_repository().category.replace_document_categories(existing_target_doc.id, [existing_target.id])
    session = client.post("/api/chat/sessions", json={"title": "复合重命名统计"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把 A统计 标签换成 B统计，然后统计 B统计 标签下有几篇",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert payload["library_mutated"] is True
    assert "已将标签/分类「A统计」合并到已有标签「B统计」" in assistant["content"]
    assert "当前「B统计」标签下有 3 篇论文" in assistant["content"]
    assert [item.name for item in get_repository().get_document(source_doc_a.id).categories] == ["B统计"]
    assert [item.name for item in get_repository().get_document(source_doc_b.id).categories] == ["B统计"]
    assert [item.name for item in get_repository().get_document(existing_target_doc.id).categories] == ["B统计"]


def test_chat_agent_compound_rename_then_assigns_untagged_without_target_pollution(client):
    source = get_repository().category.create_category("A隔离", "#0f5fb8")
    source_doc = _create_ready_document("doc-isolation-source", "IsolationSource.pdf")
    untagged_doc = _create_ready_document("doc-isolation-untagged", "IsolationUntagged.pdf")
    get_repository().category.replace_document_categories(source_doc.id, [source.id])
    session = client.post("/api/chat/sessions", json={"title": "复合步骤参数隔离"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把 A隔离 标签换成 B隔离，然后把没有标签的论文加上 C隔离 标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "completed", assistant["content"]
    assert payload["library_mutated"] is True
    assert [item.name for item in get_repository().get_document(source_doc.id).categories] == ["B隔离"]
    assert [item.name for item in get_repository().get_document(untagged_doc.id).categories] == ["C隔离"]
    assert not any(category.name == "A隔离" for category in get_repository().category.list_categories())
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    completion = next(trace for trace in traces if trace.status == "plan_completion_checked")
    steps = completion.payload["hard_gate"]["plan"]["steps"]
    assert len(steps) >= 2
    assert any(step["resolved_tool_name"] == "library.operator.rename_category" for step in steps)
    assert any(
        step["resolved_tool_name"] == "library.operator.assign_category"
        and step["resolved_tool_args"].get("category_names") == ["C隔离"]
        for step in steps
    )


def test_chat_agent_blocks_undercovered_llm_single_step_plan_for_compound_write(client, monkeypatch):
    untagged_a = _create_ready_document("doc-undercovered-a", "UndercoveredA.pdf")
    untagged_b = _create_ready_document("doc-undercovered-b", "UndercoveredB.pdf")
    session = client.post("/api/chat/sessions", json={"title": "计划覆盖检查"}).json()
    runtime = get_knowledge_agent_runtime()
    monkeypatch.setattr(runtime, "api_key", "test-key")
    actions = iter(
        [
            _ReactAction(
                "library.operator.assign_category",
                {"category_name": "Alpha", "scope": "untagged"},
                "LLM only planned the first write step.",
                action_plan=[
                    {
                        "tool": "library.operator.assign_category",
                        "arguments": {"category_name": "Alpha", "scope": "untagged"},
                    }
                ],
                confidence=0.95,
            ),
            _ReactAction("final.answer", {"content": "全部完成。"}, "premature final", confidence=0.95),
        ]
    )

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action", lambda self, **kwargs: next(actions))
    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: _ModeCandidate(
            mode=AgentRunMode.REACT,
            reason="LLM recognized a compound write.",
            confidence=0.95,
            target_runtime="KnowledgeAgentRuntime",
            source="llm",
            needs_tool=True,
            risk_level="write",
        ),
    )

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把没有标签的论文加上 Alpha 标签，然后把没有标签的论文加上 Beta 标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "validation_failed"
    assert "没有整体完成" in assistant["content"]
    assert [item.name for item in get_repository().get_document(untagged_a.id).categories] == ["Alpha"]
    assert [item.name for item in get_repository().get_document(untagged_b.id).categories] == ["Alpha"]
    assert not any(category.name == "Beta" for category in get_repository().category.list_categories())
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    completion = next(trace for trace in traces if trace.status == "plan_completion_checked")
    assert completion.payload["hard_gate"]["uncovered_subgoals"] is True


def test_chat_agent_rejects_clear_tags_but_keep_categories_conflict(client):
    doc_a = _create_ready_document("doc-conflict-a", "ConflictA.pdf")
    category = get_repository().category.create_category("中文", "#0f5fb8")
    get_repository().category.replace_document_categories(doc_a.id, [category.id])
    session = client.post("/api/chat/sessions", json={"title": "标签分类冲突"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把他们的标签都清楚，但保留对应的分类",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "needs_clarification"
    assert payload["library_mutated"] is False
    assert "标签和分类是同一字段" in assistant["content"]
    assert [category.name for category in get_repository().get_document(doc_a.id).categories] == ["中文"]


def test_chat_agent_blocks_llm_fake_completed_write_without_tool_observation(client, monkeypatch):
    target_category = get_repository().category.create_category("中文", "#0f5fb8")
    untagged = _create_ready_document("doc-fake-write", "FakeWrite.pdf")
    other = _create_ready_document("doc-fake-write-tagged", "TaggedFakeWrite.pdf")
    get_repository().category.replace_document_categories(other.id, [target_category.id])
    session = client.post("/api/chat/sessions", json={"title": "拦截假完成"}).json()
    runtime = get_knowledge_agent_runtime()
    monkeypatch.setattr(runtime, "api_key", "fake-key")

    def fake_llm_action(self, **_kwargs):
        return _ReactAction("final.answer", {"content": "所有操作已完成！"}, "fake final")

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", fake_llm_action)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把那篇论文改成“happy”标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "degraded"
    assert payload["library_mutated"] is False
    assert "没有用规则分支冒充模型理解" in assistant["content"]
    refreshed = get_repository().get_document(untagged.id)
    assert refreshed is not None
    assert refreshed.categories == []


def test_chat_agent_repairs_invalid_llm_tool_plan_for_safe_untagged_assignment(client, monkeypatch):
    untagged_a = _create_ready_document("doc-invalid-plan-a", "InvalidPlanA.pdf")
    untagged_b = _create_ready_document("doc-invalid-plan-b", "InvalidPlanB.pdf")
    tagged = _create_ready_document("doc-invalid-plan-tagged", "TaggedInvalidPlan.pdf")
    existing = get_repository().category.create_category("已有标签", "#6957d8")
    get_repository().category.replace_document_categories(tagged.id, [existing.id])
    session = client.post("/api/chat/sessions", json={"title": "LLM非JSON工具计划修复"}).json()
    runtime = get_knowledge_agent_runtime()
    monkeypatch.setattr(runtime, "api_key", "test-openai-compatible-key")

    calls = {"primary": 0, "forced": 0}

    def fake_llm_action(self, **_kwargs):
        calls["primary"] += 1
        return None

    def fake_forced_action(self, **_kwargs):
        calls["forced"] += 1
        return _ReactAction(
            "library.operator.assign_category",
            {"category_name": "happy", "scope": "untagged"},
            "LLM corrected its tool plan to a write tool.",
        )

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", fake_llm_action)
    monkeypatch.setattr(KnowledgeAgentRuntime, "_force_llm_write_tool_action", fake_forced_action)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把没有标签的文章都加一个'happy'标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "completed", assistant["content"]
    assert payload["library_mutated"] is True
    assert calls["primary"] >= 1
    assert calls["forced"] >= 1
    assert "happy" in assistant["content"]
    assert "2 篇" in assistant["content"]
    assert [category.name for category in get_repository().get_document(untagged_a.id).categories] == ["happy"]
    assert [category.name for category in get_repository().get_document(untagged_b.id).categories] == ["happy"]
    assert [category.name for category in get_repository().get_document(tagged.id).categories] == ["已有标签"]


def test_chat_agent_forces_library_write_to_react_when_llm_routes_direct(client, monkeypatch):
    untagged = _create_ready_document("doc-route-direct", "RouteDirect.pdf")
    session = client.post("/api/chat/sessions", json={"title": "路由误判修复"}).json()
    runtime = get_knowledge_agent_runtime()
    monkeypatch.setattr(runtime, "api_key", "test-openai-compatible-key")

    def fake_llm_candidate(self, payload):
        _ = self, payload
        return _ModeCandidate(
            mode=AgentRunMode.DIRECT,
            reason="LLM incorrectly treated the request as plain chat.",
            confidence=0.95,
            target_runtime="DirectChatRuntime",
            source="llm",
        )

    def fake_llm_action(self, **_kwargs):
        return _ReactAction(
            "library.operator.assign_category",
            {"category_name": "Happt", "scope": "untagged"},
            "LLM selected the tag assignment tool.",
        )

    monkeypatch.setattr(AgentOrchestrator, "_llm_candidate", fake_llm_candidate)
    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", fake_llm_action)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把我没标签的那几篇新增一个“Happt”标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "completed", assistant["content"]
    assert payload["library_mutated"] is True
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(trace.status == "agent_mode_selected" and trace.payload["mode"] == "REACT" for trace in traces)
    assert [category.name for category in get_repository().get_document(untagged.id).categories] == ["Happt"]


def test_chat_agent_executes_configured_llm_tool_plan_without_rule_fallback(client, monkeypatch):
    untagged_a = _create_ready_document("doc-llm-plan-a", "LlmPlanA.pdf")
    untagged_b = _create_ready_document("doc-llm-plan-b", "LlmPlanB.pdf")
    session = client.post("/api/chat/sessions", json={"title": "LLM工具计划"}).json()
    runtime = get_knowledge_agent_runtime()
    monkeypatch.setattr(runtime, "api_key", "test-openai-compatible-key")
    actions = iter(
        [
            _ReactAction(
                "library.operator.create_category",
                {"category_name": "删除论文"},
                "LLM识别用户是在创建标签。",
            ),
            _ReactAction(
                "library.operator.assign_category",
                {"category_name": "删除论文", "scope": "untagged"},
                "LLM计划把无标签论文分配到该标签。",
            ),
            _ReactAction("final.answer", {"content": "已按模型工具计划完成标签创建与分配。"}, "完成。"),
        ]
    )
    fallback_called = {"value": False}

    def fake_llm_action(self, **kwargs):
        _ = self, kwargs
        return next(actions)

    def fake_fallback(self, **kwargs):
        _ = self, kwargs
        fallback_called["value"] = True
        return _ReactAction("final.answer", {"content": "fallback"}, "fallback")

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", fake_llm_action)
    monkeypatch.setattr(KnowledgeAgentRuntime, "_fallback_next_react_action", fake_fallback)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把无标签论文都加个“删除论文”标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert fallback_called["value"] is False
    assert [category.name for category in get_repository().get_document(untagged_a.id).categories] == ["删除论文"]
    assert [category.name for category in get_repository().get_document(untagged_b.id).categories] == ["删除论文"]


def test_chat_agent_hides_llm_process_text_after_verified_library_write(client, monkeypatch):
    untagged_a = _create_ready_document("doc-happy-a", "HappyA.pdf")
    untagged_b = _create_ready_document("doc-happy-b", "HappyB.pdf")
    session = client.post("/api/chat/sessions", json={"title": "隐藏工具过程"}).json()
    runtime = get_knowledge_agent_runtime()
    monkeypatch.setattr(runtime, "api_key", "test-openai-compatible-key")
    actions = iter(
        [
            _ReactAction(
                "library.operator.assign_category",
                {"category_name": "happy", "scope": "untagged"},
                "LLM plans a write tool.",
            ),
            _ReactAction(
                "final.answer",
                {
                    "content": (
                        "根据运行态摘要，我将使用 `library.explorer.list_papers` 工具。\n"
                        "正在执行添加标签操作...\n"
                        "已成功为以下论文添加 happy 标签。"
                    )
                },
                "verbose final",
            ),
        ]
    )

    def fake_llm_action(self, **kwargs):
        _ = self, kwargs
        return next(actions)

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", fake_llm_action)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把没有标签的论文都加一个“happy”标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert payload["library_mutated"] is True
    assert "library." not in assistant["content"]
    assert "工具" not in assistant["content"]
    assert "运行态摘要" not in assistant["content"]
    assert "正在执行" not in assistant["content"]
    assert "2 篇" in assistant["content"]
    assert [category.name for category in get_repository().get_document(untagged_a.id).categories] == ["happy"]
    assert [category.name for category in get_repository().get_document(untagged_b.id).categories] == ["happy"]


def test_intent_scope_assigns_label_only_to_current_selection(client):
    selected_a = _create_ready_document("doc-scope-current-a", "ScopeCurrentA.pdf")
    selected_b = _create_ready_document("doc-scope-current-b", "ScopeCurrentB.pdf")
    untouched = _create_ready_document("doc-scope-current-other", "ScopeCurrentOther.pdf")
    session = client.post("/api/chat/sessions", json={"title": "当前选择打标签"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "给这些论文打上“超分”标签",
            "attachments": [],
            "selected_document_ids": [selected_a.id, selected_b.id],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed", assistant["content"]
    assert [item.name for item in get_repository().get_document(selected_a.id).categories] == ["超分"]
    assert [item.name for item in get_repository().get_document(selected_b.id).categories] == ["超分"]
    assert get_repository().get_document(untouched.id).categories == []
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assign_payload = next(
        trace.payload["payload"]
        for trace in traces
        if trace.status == "react_observation" and trace.payload.get("tool") == "library.operator.assign_category"
    )
    assert set(assign_payload["document_ids"]) == {selected_a.id, selected_b.id}
    assert assign_payload["_resolved_action"]["scope_type"] == "current_selection"


def test_intent_scope_assigns_label_to_recent_selected_documents(client, monkeypatch):
    doc_a = _create_ready_document("doc-scope-recent-a", "ScopeRecentA.pdf")
    doc_b = _create_ready_document("doc-scope-recent-b", "ScopeRecentB.pdf")
    untouched = _create_ready_document("doc-scope-recent-other", "ScopeRecentOther.pdf")
    session = client.post("/api/chat/sessions", json={"title": "最近选择打标签"}).json()
    runtime = get_knowledge_agent_runtime()
    runtime._write_react_state(
        session["id"],
        {
            "last_document_set": {
                "label": "刚才分析的 2 篇论文",
                "document_ids": [doc_a.id, doc_b.id],
                "source_tool": "report.drafter.write",
                "count": 2,
            },
            "last_user_goal": "根据这几篇论文写个简短总结",
        },
    )

    second = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把刚刚这几篇论文打上“超分”标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert second.status_code == 200
    assistant = second.json()["assistant_message"]
    assert assistant["action_status"] == "completed", assistant["content"]
    assert [item.name for item in get_repository().get_document(doc_a.id).categories] == ["超分"]
    assert [item.name for item in get_repository().get_document(doc_b.id).categories] == ["超分"]
    assert get_repository().get_document(untouched.id).categories == []
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assign_payload = next(
        trace.payload["payload"]
        for trace in traces
        if trace.status == "react_observation" and trace.payload.get("tool") == "library.operator.assign_category"
    )
    assert set(assign_payload["document_ids"]) == {doc_a.id, doc_b.id}
    assert assign_payload["_resolved_action"]["scope_type"] == "recent_selection"


def test_intent_scope_changes_label_on_recent_single_document_only(client, monkeypatch):
    recent = _create_ready_document("doc-scope-single-change", "ScopeSingleChange.pdf")
    earlier_a = _create_ready_document("doc-scope-single-earlier-a", "ScopeEarlierA.pdf")
    earlier_b = _create_ready_document("doc-scope-single-earlier-b", "ScopeEarlierB.pdf")
    untouched = _create_ready_document("doc-scope-single-change-other", "ScopeSingleOther.pdf")
    old = get_repository().category.create_category("超分", "#6957d8")
    get_repository().category.replace_document_categories(recent.id, [old.id])
    get_repository().category.replace_document_categories(untouched.id, [old.id])
    session = client.post("/api/chat/sessions", json={"title": "最近单篇改标签"}).json()
    runtime = get_knowledge_agent_runtime()
    runtime._write_react_state(
        session["id"],
        {
            "last_document_set": {
                "label": "刚才分析的 1 篇论文",
                "document_ids": [recent.id],
                "source_tool": "report.drafter.write",
                "count": 1,
            },
            "last_single_document": {
                "label": "刚才分析的 1 篇论文",
                "document_id": recent.id,
                "document_ids": [recent.id],
                "source_tool": "report.drafter.write",
                "count": 1,
            },
            "last_multi_document_set": {
                "label": "更早分析的 2 篇论文",
                "document_ids": [earlier_a.id, earlier_b.id],
                "source_tool": "report.drafter.write",
                "count": 2,
            },
            "last_user_goal": "介绍这篇论文",
        },
    )
    monkeypatch.setattr(runtime, "api_key", "test-key")

    def fake_llm_action(self, **kwargs):
        _ = self, kwargs
        return _ReactAction(
            "library.operator.rename_category",
            {"source_category_name": "超分", "target_category_name": "英文超分"},
            "LLM incorrectly selected global rename.",
        )

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", fake_llm_action)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"content": "把这篇论文的标签改成“英文超分”", "attachments": [], "selected_document_ids": []},
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed", assistant["content"]
    assert [item.name for item in get_repository().get_document(recent.id).categories] == ["超分", "英文超分"]
    assert [item.name for item in get_repository().get_document(untouched.id).categories] == ["超分"]
    assert get_repository().get_document(earlier_a.id).categories == []
    assert get_repository().get_document(earlier_b.id).categories == []
    assert any(category.name == "超分" for category in get_repository().category.list_categories())
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert not any(
        trace.status == "react_observation"
        and trace.payload.get("tool") == "library.operator.rename_category"
        and trace.payload.get("status") == "completed"
        for trace in traces
    )
    assign_payload = next(
        trace.payload["payload"]
        for trace in traces
        if trace.status == "react_observation" and trace.payload.get("tool") == "library.operator.assign_category"
    )
    assert assign_payload["document_ids"] == [recent.id]
    assert assign_payload["_resolved_action"]["target_type"] == "paper_label_relation"
    assert assign_payload["_resolved_action"]["scope_type"] == "recent_selection"


def test_intent_scope_adds_label_to_recent_single_document_only(client):
    recent = _create_ready_document("doc-scope-single-add", "ScopeSingleAdd.pdf")
    untouched = _create_ready_document("doc-scope-single-add-other", "ScopeSingleAddOther.pdf")
    session = client.post("/api/chat/sessions", json={"title": "最近单篇加标签"}).json()
    get_knowledge_agent_runtime()._write_react_state(
        session["id"],
        {
            "last_single_document": {
                "label": "刚才分析的 1 篇论文",
                "document_id": recent.id,
                "document_ids": [recent.id],
                "source_tool": "report.drafter.write",
                "count": 1,
            },
            "last_document_set": {
                "label": "刚才分析的 1 篇论文",
                "document_ids": [recent.id],
                "source_tool": "report.drafter.write",
                "count": 1,
            },
        },
    )

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"content": "给这篇论文打上英文超分标签", "attachments": [], "selected_document_ids": []},
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed", assistant["content"]
    assert [item.name for item in get_repository().get_document(recent.id).categories] == ["英文超分"]
    assert get_repository().get_document(untouched.id).categories == []


def test_intent_scope_removes_label_from_recent_single_document_only(client):
    recent = _create_ready_document("doc-scope-single-remove", "ScopeSingleRemove.pdf")
    untouched = _create_ready_document("doc-scope-single-remove-other", "ScopeSingleRemoveOther.pdf")
    target = get_repository().category.create_category("英文超分", "#6957d8")
    other = get_repository().category.create_category("其它", "#047c71")
    get_repository().category.replace_document_categories(recent.id, [target.id, other.id])
    get_repository().category.replace_document_categories(untouched.id, [target.id, other.id])
    session = client.post("/api/chat/sessions", json={"title": "最近单篇删标签"}).json()
    get_knowledge_agent_runtime()._write_react_state(
        session["id"],
        {
            "last_single_document": {
                "label": "刚才分析的 1 篇论文",
                "document_id": recent.id,
                "document_ids": [recent.id],
                "source_tool": "report.drafter.write",
                "count": 1,
            },
            "last_document_set": {
                "label": "刚才分析的 1 篇论文",
                "document_ids": [recent.id],
                "source_tool": "report.drafter.write",
                "count": 1,
            },
        },
    )

    preview = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"content": "把这篇论文的英文超分标签删掉", "attachments": [], "selected_document_ids": []},
    )

    assert preview.status_code == 200
    assistant = preview.json()["assistant_message"]
    assert assistant["action_status"] == "confirmation_required", assistant["content"]
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    preview_payload = next(
        trace.payload
        for trace in traces
        if trace.status == "write_preview_created" and trace.payload.get("tool") == "library.operator.clear_categories"
    )
    assert preview_payload["affected_count"] == 1
    assert {item["document_id"] for item in preview_payload["targets"]} == {recent.id}

    confirm = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"content": "确认移除英文超分标签", "attachments": [], "selected_document_ids": []},
    )
    assert confirm.status_code == 200
    assert confirm.json()["assistant_message"]["action_status"] == "completed"
    assert [item.name for item in get_repository().get_document(recent.id).categories] == ["其它"]
    assert [item.name for item in get_repository().get_document(untouched.id).categories] == ["英文超分", "其它"]


def test_intent_scope_remove_label_from_recent_documents_only(client):
    doc_a = _create_ready_document("doc-scope-remove-a", "ScopeRemoveA.pdf")
    doc_b = _create_ready_document("doc-scope-remove-b", "ScopeRemoveB.pdf")
    untouched = _create_ready_document("doc-scope-remove-other", "ScopeRemoveOther.pdf")
    category = get_repository().category.create_category("超分", "#6957d8")
    other = get_repository().category.create_category("其它", "#047c71")
    for document in (doc_a, doc_b, untouched):
        get_repository().category.replace_document_categories(document.id, [category.id, other.id])
    session = client.post("/api/chat/sessions", json={"title": "最近选择删标签"}).json()

    first = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "给这些论文打上“临时”标签",
            "attachments": [],
            "selected_document_ids": [doc_a.id, doc_b.id],
        },
    )
    assert first.status_code == 200

    preview = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把刚刚这几篇论文的超分标签删掉",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert preview.status_code == 200
    payload = preview.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "confirmation_required", assistant["content"]
    assert payload["library_mutated"] is False
    assert "确认移除超分标签" in assistant["content"]
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    preview_payload = next(
        trace.payload
        for trace in traces
        if trace.status == "write_preview_created" and trace.payload.get("tool") == "library.operator.clear_categories"
    )
    assert preview_payload["affected_count"] == 2
    assert {item["document_id"] for item in preview_payload["targets"]} == {doc_a.id, doc_b.id}

    confirm = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"content": "确认移除超分标签", "attachments": [], "selected_document_ids": []},
    )
    assert confirm.status_code == 200
    assert confirm.json()["assistant_message"]["action_status"] == "completed"
    assert [item.name for item in get_repository().get_document(doc_a.id).categories] == ["其它", "临时"]
    assert [item.name for item in get_repository().get_document(doc_b.id).categories] == ["其它", "临时"]
    assert [item.name for item in get_repository().get_document(untouched.id).categories] == ["超分", "其它"]


def test_intent_scope_ambiguous_recent_reference_does_not_default_to_all_library(client):
    doc_a = _create_ready_document("doc-scope-ambiguous-a", "ScopeAmbiguousA.pdf")
    doc_b = _create_ready_document("doc-scope-ambiguous-b", "ScopeAmbiguousB.pdf")
    session = client.post("/api/chat/sessions", json={"title": "无上下文模糊指代"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把刚刚这几篇论文打上“超分”标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] in {"validation_failed", "needs_clarification", "degraded"}
    assert payload["library_mutated"] is False
    assert get_repository().get_document(doc_a.id).categories == []
    assert get_repository().get_document(doc_b.id).categories == []


def test_intent_scope_ambiguous_single_reference_does_not_search_or_mutate_all_library(client):
    doc_a = _create_ready_document("doc-scope-ambiguous-single-a", "ScopeAmbiguousSingleA.pdf")
    doc_b = _create_ready_document("doc-scope-ambiguous-single-b", "ScopeAmbiguousSingleB.pdf")
    session = client.post("/api/chat/sessions", json={"title": "无上下文单篇指代"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"content": "把这篇论文标签改成英文超分", "attachments": [], "selected_document_ids": []},
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] in {"validation_failed", "needs_clarification", "degraded"}
    assert payload["library_mutated"] is False
    assert get_repository().get_document(doc_a.id).categories == []
    assert get_repository().get_document(doc_b.id).categories == []
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert not any(
        trace.status == "react_observation"
        and trace.payload.get("tool") == "library.explorer.find_documents"
        for trace in traces
    )


def test_intent_scope_explicit_all_library_batch_assign_requires_confirmation_or_refusal(client):
    doc_a = _create_ready_document("doc-scope-all-a", "ScopeAllA.pdf")
    doc_b = _create_ready_document("doc-scope-all-b", "ScopeAllB.pdf")
    session = client.post("/api/chat/sessions", json={"title": "明确全库打标签"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把所有论文都打上“超分”标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] in {"validation_failed", "confirmation_required"}
    assert payload["library_mutated"] is False
    assert get_repository().get_document(doc_a.id).categories == []
    assert get_repository().get_document(doc_b.id).categories == []
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(
        trace.status == "react_observation"
        and trace.payload.get("tool") == "library.operator.assign_category"
        and trace.payload.get("payload", {}).get("guardrail") in {"write_scope_validation", None}
        for trace in traces
    )


def test_intent_scope_delete_empty_labels_is_entity_level_not_relation_clear(client):
    empty = get_repository().category.create_category("空标签", "#6957d8")
    used = get_repository().category.create_category("已用", "#047c71")
    document = _create_ready_document("doc-scope-delete-empty", "ScopeDeleteEmpty.pdf")
    get_repository().category.replace_document_categories(document.id, [used.id])
    session = client.post("/api/chat/sessions", json={"title": "删除空标签分类"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"content": "删除空标签分类", "attachments": [], "selected_document_ids": []},
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "confirmation_required"
    assert payload["library_mutated"] is False
    assert [item.name for item in get_repository().get_document(document.id).categories] == ["已用"]
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(
        trace.status == "write_preview_created"
        and trace.payload.get("tool") == "library.operator.delete_unused_categories"
        and trace.payload.get("operation_level") == "entity-level"
        for trace in traces
    )
    assert not any(
        trace.status == "write_preview_created"
        and trace.payload.get("tool") == "library.operator.clear_categories"
        for trace in traces
    )
    assert empty.name in assistant["content"]


def test_intent_scope_explicit_label_entity_rename_still_uses_rename_category(client):
    document = _create_ready_document("doc-scope-entity-rename", "ScopeEntityRename.pdf")
    old = get_repository().category.create_category("A标签", "#6957d8")
    get_repository().category.replace_document_categories(document.id, [old.id])
    session = client.post("/api/chat/sessions", json={"title": "实体标签重命名"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"content": "把标签 A标签 重命名为 B标签", "attachments": [], "selected_document_ids": []},
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed", assistant["content"]
    assert [item.name for item in get_repository().get_document(document.id).categories] == ["B标签"]
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(
        trace.status == "react_observation"
        and trace.payload.get("tool") == "library.operator.rename_category"
        and trace.payload.get("status") == "completed"
        for trace in traces
    )


def test_intent_scope_selected_paper_question_goes_through_evidence_chain(client, monkeypatch):
    document = _create_ready_document("doc-scope-paper-qa", "ScopePaperQA.pdf", title="Scope Paper QA")
    session = client.post("/api/chat/sessions", json={"title": "这篇论文讲什么"}).json()
    _install_fake_openai(
        monkeypatch,
        "app.runtime.knowledge_agent_runtime.OpenAI",
        content="这篇论文围绕正文证据中的方法、贡献和实验结果展开。",
    )
    runtime = get_knowledge_agent_runtime()
    runtime.api_key = "paper-qa-scope-key"

    def fake_retrieve_evidence(self, **kwargs):
        documents = kwargs["documents"]
        return [
            EvidenceItem(
                id=f"paper-qa-{documents[0].id}",
                source_type="local_document",
                source_id=documents[0].id,
                title=documents[0].title,
                snippet="The paper proposes a method, explains contributions, and reports experiment results.",
                citation_label=f"{documents[0].filename} p.1",
                document_id=documents[0].id,
                page_number=1,
                score=0.9,
            )
        ]

    monkeypatch.setattr(RagService, "retrieve_evidence", fake_retrieve_evidence)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"content": "这篇论文讲什么", "attachments": [], "selected_document_ids": [document.id]},
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed"
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    tools = [trace.payload.get("tool") for trace in traces if trace.status == "react_observation"]
    assert "evidence.retriever.search" in tools
    assert assistant["used_document_ids"] == [document.id]


def test_chat_agent_assigns_multiple_tags_to_all_untagged_documents(client):
    existing = get_repository().category.create_category("已有标签", "#6957d8")
    tagged = _create_ready_document("doc-multitag-tagged", "AlreadyTagged.pdf")
    untagged_a = _create_ready_document("doc-multitag-a", "UntaggedA.pdf")
    untagged_b = _create_ready_document("doc-multitag-b", "UntaggedB.pdf")
    get_repository().category.replace_document_categories(tagged.id, [existing.id])
    session = client.post("/api/chat/sessions", json={"title": "多标签批量补全"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把没有标签的都加上一个 cscd 标签和一个 hddcc 标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "completed", assistant["content"]
    assert payload["library_mutated"] is True
    assert "cscd" in assistant["content"]
    assert "hddcc" in assistant["content"]
    assert "2 篇" in assistant["content"]
    assert "当前仍无标签论文 0 篇" in assistant["content"]
    assert [category.name for category in get_repository().get_document(tagged.id).categories] == ["已有标签"]
    assert [category.name for category in get_repository().get_document(untagged_a.id).categories] == ["cscd", "hddcc"]
    assert [category.name for category in get_repository().get_document(untagged_b.id).categories] == ["cscd", "hddcc"]
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assign_observation = next(
        trace
        for trace in traces
        if trace.status == "react_observation"
        and trace.payload.get("tool") == "library.operator.assign_category"
    )
    assert assign_observation.payload["payload"]["category_names"] == ["cscd", "hddcc"]
    assert assign_observation.payload["payload"]["verified_state"]["untagged_document_count"] == 0
    assert any(
        trace.status == "library_write_verified"
        and trace.payload.get("verified_state", {}).get("untagged_document_count") == 0
        for trace in traces
    )


def test_chat_agent_uses_context_for_remaining_untagged_assignment(client, monkeypatch):
    tagged_category = get_repository().category.create_category("中文", "#0f5fb8")
    tagged_documents = [
        _create_ready_document(f"doc-tagged-{index}", f"Tagged{index}.pdf")
        for index in range(4)
    ]
    untagged_documents = [
        _create_ready_document(f"doc-remaining-{index}", f"Remaining{index}.pdf")
        for index in range(3)
    ]
    for document in tagged_documents:
        get_repository().category.replace_document_categories(document.id, [tagged_category.id])
    session = client.post("/api/chat/sessions", json={"title": "上下文另外几篇"}).json()

    first_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "我有几篇带标签的文章？",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert first_response.status_code == 200
    assert first_response.json()["assistant_message"]["content"] == "当前带分类/标签的论文有 4 篇。"

    runtime = get_knowledge_agent_runtime()
    monkeypatch.setattr(runtime, "api_key", "test-openai-compatible-key")

    def fake_llm_action(self, **_kwargs):
        return _ReactAction("final.answer", {"content": "好的，我现在为那3篇无标签的文章添加 happy 标签。"}, "fake final")

    def fake_forced_action(self, **_kwargs):
        return _ReactAction(
            "library.operator.assign_category",
            {"category_name": "happy", "scope": "untagged"},
            "LLM corrected its premature final answer to a write tool.",
        )

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", fake_llm_action)
    monkeypatch.setattr(KnowledgeAgentRuntime, "_force_llm_write_tool_action", fake_forced_action)

    second_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把另外三篇都加一个“happy”标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert second_response.status_code == 200
    payload = second_response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "completed", assistant["content"]
    assert payload["library_mutated"] is True
    assert "根据运行态摘要" not in assistant["content"]
    assert "工具" not in assistant["content"]
    assert "3 篇" in assistant["content"]
    for document in untagged_documents:
        assert [category.name for category in get_repository().get_document(document.id).categories] == ["happy"]
    for document in tagged_documents:
        assert [category.name for category in get_repository().get_document(document.id).categories] == ["中文"]


def test_chat_agent_compares_only_documents_with_named_category(client, monkeypatch):
    chinese = get_repository().category.create_category("中文", "#0f5fb8")
    selected_a = _create_ready_document("doc-cn-a", "ChineseA.pdf", title="中文论文 A")
    selected_b = _create_ready_document("doc-cn-b", "ChineseB.pdf", title="中文论文 B")
    other = _create_ready_document("doc-other", "Other.pdf", title="其它论文")
    get_repository().category.replace_document_categories(selected_a.id, [chinese.id])
    get_repository().category.replace_document_categories(selected_b.id, [chinese.id])
    session = client.post("/api/chat/sessions", json={"title": "中文标签对比"}).json()
    calls: list[list[str]] = []

    def fake_retrieve(self, **kwargs):
        calls.append([document.id for document in kwargs["documents"]])
        return []

    monkeypatch.setattr(RagService, "retrieve_evidence", fake_retrieve)

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "为我所有带着“中文”标签的论文写一篇对比，对比他们的区别",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "degraded"
    assert len(calls) == 1
    assert set(calls[0]) == {selected_a.id, selected_b.id}
    assert other.id not in calls[0]
    assert "没有在论文库中唯一定位" not in assistant["content"]
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    find_observation = next(
        trace
        for trace in traces
        if trace.status == "react_observation"
        and trace.payload.get("tool") == "library.explorer.find_documents"
    )
    assert find_observation.payload["payload"]["category_name"] == "中文"
    assert set(find_observation.payload["payload"]["document_ids"]) == {selected_a.id, selected_b.id}


def test_chat_agent_writes_report_from_all_chinese_category_documents_with_real_abstracts(client, monkeypatch):
    chinese = get_repository().category.create_category("chinese", "#0f5fb8")
    selected_a = _create_ready_document("doc-chinese-report-a", "ChineseReportA.pdf", title="Chinese Report A")
    selected_b = _create_ready_document("doc-chinese-report-b", "ChineseReportB.pdf", title="Chinese Report B")
    selected_c = _create_ready_document("doc-chinese-report-c", "ChineseReportC.pdf", title="Chinese Report C")
    other = _create_ready_document("doc-not-chinese-report", "OtherReport.pdf", title="Other Report")
    for document in (selected_a, selected_b, selected_c):
        get_repository().category.replace_document_categories(document.id, [chinese.id])
    _add_abstract_chunk(
        selected_a,
        "Abstract This first paper studies adaptive lookup tables for image enhancement with reliable training evidence.",
    )
    _add_abstract_chunk(
        selected_b,
        "Abstract This second paper proposes zero-reference curve estimation for low-light image enhancement.",
    )
    _add_abstract_chunk(
        selected_c,
        "Abstract This third paper builds a mobile image enhancement network with efficient dual stream design.",
    )
    _add_abstract_chunk(other, "Abstract This unrelated paper should not be used for the chinese tag report.")
    _install_fake_openai(
        monkeypatch,
        "app.runtime.knowledge_agent_runtime.OpenAI",
        content=(
            "论文一：\n论文名称：ChineseReportA.pdf\n论文完整摘要：Abstract This first paper studies adaptive lookup tables for image enhancement with reliable training evidence.\n论文页数：10页\n"
            "论文二：\n论文名称：ChineseReportB.pdf\n论文完整摘要：Abstract This second paper proposes zero-reference curve estimation for low-light image enhancement.\n论文页数：10页\n"
            "论文三：\n论文名称：ChineseReportC.pdf\n论文完整摘要：Abstract This third paper builds a mobile image enhancement network with efficient dual stream design.\n论文页数：10页\n"
            "三篇文章都关注图像增强，但方法路线分别强调 lookup tables、zero-reference curve estimation 和 dual stream design。"
        ),
    )
    get_knowledge_agent_runtime().api_key = "chinese-report-key"
    session = client.post("/api/chat/sessions", json={"title": "chinese报告"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": (
                "用我所有“chinese”标签的论文，来写一篇报告，报告要求格式如下：\n"
                "论文一：\n论文名称：xxx\n论文完整摘要：xxxxx（这里是论文真实摘要）\n论文页数：xx页\n\n"
                "最后用一段不分点的文字对三篇文章的相同点和不同点做一个总结"
            ),
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "degraded"
    assert set(assistant["used_document_ids"]) == {selected_a.id, selected_b.id, selected_c.id}
    assert other.id not in assistant["used_document_ids"]
    content = assistant["content"]
    assert "论文一：" in content
    assert "论文二：" in content
    assert "论文三：" in content
    assert "论文名称：" in content
    assert "论文完整摘要：" in content
    assert "论文页数：10页" in content
    assert "adaptive lookup tables" in content
    assert "zero-reference curve estimation" in content
    assert "dual stream design" in content
    assert "unrelated paper" not in content


def test_chat_agent_uses_previous_react_observation_for_followup_assignment(client):
    documents = [
        _create_ready_document("doc-a12", "A12AAAAA.pdf"),
        _create_ready_document("doc-b45", "B45BBB.pdf"),
        _create_ready_document("doc-c90", "C90CCC.pdf"),
        _create_ready_document("doc-d20", "D20DDD.pdf"),
    ]
    session = client.post("/api/chat/sessions", json={"title": "跨轮指代"}).json()

    first_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "我的论文库里有几篇论文？",
            "attachments": [],
            "selected_document_ids": [],
        },
    )
    assert first_response.status_code == 200
    assert "共有 4 篇论文" in first_response.json()["assistant_message"]["content"]

    second_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "把这四篇都设置成'法律'标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert second_response.status_code == 200
    assistant = second_response.json()["assistant_message"]
    assert assistant["action_status"] == "completed", assistant
    assert "法律" in assistant["content"]
    assert "4 篇" in assistant["content"]
    assert any(category.name == "法律" for category in get_repository().category.list_categories())
    for document in documents:
        refreshed = get_repository().get_document(document.id)
        assert refreshed is not None
        assert [category.name for category in refreshed.categories] == ["法律"]


def test_chat_agent_executes_llm_planned_complex_tool_chain(client, monkeypatch):
    law = get_repository().category.create_category("法律", "#0f5fb8")
    law_doc = _create_ready_document("doc-law", "LawPaper.pdf", title="Legal Reasoning Paper")
    history_a = _create_ready_document("doc-history-a", "HistoryA.pdf", title="History Method Paper")
    history_b = _create_ready_document("doc-history-b", "HistoryB.pdf", title="Historical Archive Paper")
    get_repository().category.replace_document_categories(law_doc.id, [law.id])
    session = client.post("/api/chat/sessions", json={"title": "复杂工具链"}).json()
    actions = iter(
        [
            _ReactAction("tool.registry.list", {}, "查看可用工具。"),
            _ReactAction("library.explorer.category_stats", {}, "读取标签和无标签论文。"),
            _ReactAction(
                "library.operator.assign_category",
                {"category_name": "历史", "scope": "untagged"},
                "把无标签论文补上历史标签。",
            ),
            _ReactAction(
                "evidence.retriever.search_by_category",
                {"question": "按标签分别总结全部论文"},
                "按标签分组做RAG检索。",
            ),
            _ReactAction(
                "report.drafter.write_by_category",
                {"question": "把所有论文按标签分别写一份总结", "target_chars": 600},
                "根据RAG观察写每个标签的总结。",
            ),
        ]
    )

    def fake_next_action(self, **kwargs):
        _ = self, kwargs
        return next(actions)

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action", fake_next_action)
    monkeypatch.setattr(
        RagService,
        "retrieve_evidence_with_quality",
        lambda self, **kwargs: RetrievalResult(
            evidence_items=[],
            evidence_quality=EvidenceQuality(
                coverage_score=0.0,
                diversity_score=0.0,
                citation_score=0.0,
                relevance_score=0.0,
                warnings=["insufficient_evidence"],
            ),
            cache_hit=False,
        ),
    )

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "我的论文库里有几篇论文，哪些有标签？分别是什么标签？帮我把没有标签的那几篇加上“历史”标签，然后把我所有的论文按标签分别写一份总结，每份600字左右",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed"
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    mode_trace = next(trace for trace in traces if trace.status == "agent_mode_selected")
    assert mode_trace.payload["route"] == "ConfirmedWrite"
    assert mode_trace.payload["requires_confirmation"] is False
    assert all(trace.status != "planner_plan_created" for trace in traces)
    assert all(trace.status != "reflection_result_created" for trace in traces)
    assert "法律" in assistant["content"]
    assert "历史" in assistant["content"]
    assert "600" in assistant["content"]
    assert "该标签下论文存在，但正文证据不足" in assistant["content"]
    grouped_observation = next(
        trace
        for trace in traces
        if trace.status == "react_observation"
        and trace.payload.get("tool") == "evidence.retriever.search_by_category"
    )
    category_groups = grouped_observation.payload["payload"]["category_groups"]
    standard_observation = grouped_observation.payload["observation"]
    assert standard_observation["tool_name"] == "evidence.retriever.search_by_category"
    assert standard_observation["operation_level"] == "query-level"
    assert standard_observation["io_type"] == "read"
    assert standard_observation["counts"]["evidence_items"] == 0
    assert category_groups
    assert all("evidence_quality" in group for group in category_groups)
    assert all(group["evidence_boundary"] == "该标签下论文存在，但正文证据不足。" for group in category_groups)
    refreshed_history_a = get_repository().get_document(history_a.id)
    refreshed_history_b = get_repository().get_document(history_b.id)
    refreshed_law = get_repository().get_document(law_doc.id)
    assert refreshed_history_a is not None
    assert refreshed_history_b is not None
    assert refreshed_law is not None
    assert [category.name for category in refreshed_history_a.categories] == ["历史"]
    assert [category.name for category in refreshed_history_b.categories] == ["历史"]
    assert [category.name for category in refreshed_law.categories] == ["法律"]


def test_chat_tool_registry_returns_structured_tool_metadata(client):
    _ = client
    runtime = get_knowledge_agent_runtime()
    registry_observation = runtime._finalize_tool_observation(
        runtime._tool_registry_list("run-tool-metadata", None, "", {}, [])
    )

    tools = registry_observation.payload["tools"]
    delete_tool = next(tool for tool in tools if tool["name"] == "library.operator.delete_unused_categories")
    assert delete_tool["operation_level"] == "entity-level"
    assert delete_tool["io_type"] == "write"
    assert delete_tool["write_type"] == "delete"
    assert delete_tool["requires_confirmation"] is True
    assert delete_tool["requires_post_read_verification"] is True
    assert all("operation_level" in tool and "io_type" in tool for tool in tools)
    assert all(tool["scope"] != "experimental" for tool in tools)
    assert registry_observation.observation["tool_name"] == "tool.registry.list"
    assert registry_observation.observation["success"] is True


def test_chat_agent_requires_confirmation_before_destructive_actions(client):
    category = get_repository().category.create_category("待删除分类", "#b42318")
    document = _create_ready_document("doc-delete-category-linked", "DeleteCategoryLinked.pdf")
    get_repository().category.replace_document_categories(document.id, [category.id])
    session = client.post("/api/chat/sessions", json={"title": "删除保护"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "帮我删除分类待删除分类",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "confirmation_required"
    assert get_repository().category.get_category(category.id) is not None
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(trace.status == "agent_mode_selected" and trace.payload["mode"] == "REACT" for trace in traces)
    assert all(trace.status != "reflection_improvement_started" for trace in traces)

    confirm_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "确认删除",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert confirm_response.status_code == 200
    confirm_payload = confirm_response.json()
    confirm_assistant = confirm_response.json()["assistant_message"]
    assert confirm_assistant["action_status"] == "completed"
    assert confirm_payload["library_mutated"] is True
    assert get_repository().category.get_category(category.id) is None
    assert get_repository().get_document(document.id).categories == []


def test_chat_agent_previews_unused_category_entity_cleanup_without_clearing_documents(client):
    empty_a = get_repository().category.create_category("空标签A", "#b42318")
    empty_b = get_repository().category.create_category("空标签B", "#b76a00")
    non_empty_c = get_repository().category.create_category("非空C", "#047c71")
    non_empty_d = get_repository().category.create_category("非空D", "#0f5fb8")
    doc_c = _create_ready_document("doc-unused-cleanup-c", "UnusedCleanupC.pdf")
    doc_d = _create_ready_document("doc-unused-cleanup-d", "UnusedCleanupD.pdf")
    get_repository().category.replace_document_categories(doc_c.id, [non_empty_c.id])
    get_repository().category.replace_document_categories(doc_d.id, [non_empty_d.id])
    session = client.post("/api/chat/sessions", json={"title": "清理空标签"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "帮我把空的标签分类都删掉",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "confirmation_required"
    assert "空标签A" in assistant["content"]
    assert "空标签B" in assistant["content"]
    assert "论文的标签关系" in assistant["content"]
    assert get_repository().category.get_category(empty_a.id) is not None
    assert get_repository().category.get_category(empty_b.id) is not None
    assert [category.name for category in get_repository().get_document(doc_c.id).categories] == ["非空C"]
    assert [category.name for category in get_repository().get_document(doc_d.id).categories] == ["非空D"]
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    preview = next(trace for trace in traces if trace.status == "write_preview_created")
    assert preview.payload["tool"] == "library.operator.delete_unused_categories"
    assert preview.payload["operation_level"] == "entity-level"
    assert preview.payload["write_type"] == "delete"
    assert preview.payload["target_type"] == "category"
    assert preview.payload["target_count"] == 2
    assert preview.payload["will_delete_entities"] is True
    assert preview.payload["will_modify_relations"] is False
    assert preview.payload["requires_confirmation"] is True
    assert {target["name"] for target in preview.payload["targets"]} == {"空标签A", "空标签B"}
    assert all(
        not (
            trace.status == "react_observation"
            and trace.payload.get("tool") == "library.operator.clear_categories"
        )
        for trace in traces
    )
    react_preview = next(
        trace
        for trace in traces
        if trace.status == "react_observation"
        and trace.payload.get("tool") == "library.operator.delete_unused_categories"
    )
    observation = react_preview.payload["observation"]
    assert observation["tool_name"] == "library.operator.delete_unused_categories"
    assert observation["success"] is False
    assert observation["operation_level"] == "entity-level"
    assert observation["io_type"] == "write"
    assert observation["write_type"] == "delete"
    assert observation["requires_confirmation"] is True
    assert observation["error"]["code"] == "CONFIRMATION_REQUIRED"


def test_chat_agent_confirms_unused_category_entity_cleanup_only_deletes_empty_entities(client):
    empty_a = get_repository().category.create_category("空标签确认A", "#b42318")
    empty_b = get_repository().category.create_category("空标签确认B", "#b76a00")
    non_empty_c = get_repository().category.create_category("非空确认C", "#047c71")
    non_empty_d = get_repository().category.create_category("非空确认D", "#0f5fb8")
    doc_c = _create_ready_document("doc-unused-confirm-c", "UnusedConfirmC.pdf")
    doc_d = _create_ready_document("doc-unused-confirm-d", "UnusedConfirmD.pdf")
    get_repository().category.replace_document_categories(doc_c.id, [non_empty_c.id])
    get_repository().category.replace_document_categories(doc_d.id, [non_empty_d.id])
    before_doc_c = [category.name for category in get_repository().get_document(doc_c.id).categories]
    before_doc_d = [category.name for category in get_repository().get_document(doc_d.id).categories]
    session = client.post("/api/chat/sessions", json={"title": "确认清理空标签"}).json()

    preview_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "帮我把空的标签分类都删掉，就是那些没有任何一篇论文带的标签",
            "attachments": [],
            "selected_document_ids": [],
        },
    )
    assert preview_response.status_code == 200
    assert preview_response.json()["assistant_message"]["action_status"] == "confirmation_required"

    confirm_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "确认删除空标签分类",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert confirm_response.status_code == 200
    payload = confirm_response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert payload["library_mutated"] is True
    assert get_repository().category.get_category(empty_a.id) is None
    assert get_repository().category.get_category(empty_b.id) is None
    assert get_repository().category.get_category(non_empty_c.id) is not None
    assert get_repository().category.get_category(non_empty_d.id) is not None
    assert [category.name for category in get_repository().get_document(doc_c.id).categories] == before_doc_c
    assert [category.name for category in get_repository().get_document(doc_d.id).categories] == before_doc_d
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    executed = next(trace for trace in traces if trace.status == "pending_write_executed")
    assert executed.payload["operation_level"] == "entity-level"
    assert executed.payload["library_mutated"] is True
    write_log = next(
        trace
        for trace in traces
        if trace.status == "pending_write_executed"
    )
    assert write_log.payload["operation_level"] == "entity-level"


def test_chat_agent_unused_category_cleanup_noops_when_none_empty(client):
    non_empty = get_repository().category.create_category("非空无清理", "#047c71")
    doc = _create_ready_document("doc-unused-noop", "UnusedNoop.pdf")
    get_repository().category.replace_document_categories(doc.id, [non_empty.id])
    session = client.post("/api/chat/sessions", json={"title": "没有空标签"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "帮我把空的标签分类都删掉",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "completed"
    assert payload["library_mutated"] is False
    assert "没有 count=0" in assistant["content"]
    assert "要清空分类/标签的论文" not in assistant["content"]
    assert get_repository().category.get_category(non_empty.id) is not None
    assert [category.name for category in get_repository().get_document(doc.id).categories] == ["非空无清理"]


def test_chat_agent_clarifies_ambiguous_delete_papers_under_tag(client):
    category = get_repository().category.create_category("模糊标签", "#0f5fb8")
    document = _create_ready_document("doc-ambiguous-delete-tag", "AmbiguousDeleteTag.pdf")
    get_repository().category.replace_document_categories(document.id, [category.id])
    session = client.post("/api/chat/sessions", json={"title": "模糊删除"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "删除这个标签下的论文",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] in {"needs_clarification", "completed"}
    assert payload["library_mutated"] is False
    assert "删除论文实体" in assistant["content"]
    assert "标签/分类之间的关系" in assistant["content"]
    assert get_repository().category.get_category(category.id) is not None
    assert [item.name for item in get_repository().get_document(document.id).categories] == ["模糊标签"]


def test_chat_agent_does_not_report_category_delete_success_without_verification(client, monkeypatch):
    category = get_repository().category.create_category("待验证分类", "#b42318")
    document = _create_ready_document("doc-delete-category-unverified", "DeleteCategoryUnverified.pdf")
    get_repository().category.replace_document_categories(document.id, [category.id])
    session = client.post("/api/chat/sessions", json={"title": "删除验证"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "删除待验证分类这个分类",
            "attachments": [],
            "selected_document_ids": [],
        },
    )
    assert response.status_code == 200
    assert get_knowledge_agent_runtime().has_pending_action(session["id"]) is True

    def fake_delete_category(category_id: str):
        _ = category_id
        return category

    monkeypatch.setattr(get_repository().category, "delete_category", fake_delete_category)

    confirm_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "确认删除",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert confirm_response.status_code == 200
    payload = confirm_response.json()
    assistant = payload["assistant_message"]
    assert assistant["action_status"] == "failed"
    assert payload["library_mutated"] is False
    assert "没有能验证" in assistant["content"]
    assert get_repository().category.get_category(category.id) is not None
    assert [item.name for item in get_repository().get_document(document.id).categories] == ["待验证分类"]


def test_chat_assistant_message_can_be_saved_as_report(client):
    session = client.post("/api/chat/sessions", json={"title": "保存报告"}).json()
    send_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "请用中文给我一个简短说明",
            "attachments": [],
            "selected_document_ids": [],
        },
    )
    assert send_response.status_code == 200
    assistant = send_response.json()["assistant_message"]

    save_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages/{assistant['id']}/report"
    )

    assert save_response.status_code == 200
    report = save_response.json()
    assert report["markdown"] == assistant["content"]
    list_response = client.get("/api/reports")
    assert any(item["id"] == report["id"] for item in list_response.json())
    detail_response = client.get(f"/api/chat/sessions/{session['id']}")
    saved_message = next(item for item in detail_response.json()["messages"] if item["id"] == assistant["id"])
    assert saved_message["saved_report_id"] == report["id"]
    assert report["lifecycle_status"] == "saved_report"
    assert report["source"] == "knowledge_answer"
    assert report["source_message_id"] == assistant["id"]
    assert report["paper_ids"] == assistant["used_document_ids"]
    assert report["updated_at"]

    alias_response = client.post(
        "/api/reports/from-message",
        json={"session_id": session["id"], "message_id": assistant["id"]},
    )
    assert alias_response.status_code == 200
    assert alias_response.json()["id"] == report["id"]


def test_chat_answer_does_not_auto_create_saved_report(client):
    session = client.post("/api/chat/sessions", json={"title": "普通回答不保存"}).json()
    send_response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "总结一下 RAG 的常见评估方式",
            "attachments": [],
            "selected_document_ids": [],
        },
    )
    assert send_response.status_code == 200
    assistant = send_response.json()["assistant_message"]
    assert assistant["saved_report_id"] is None

    list_response = client.get("/api/reports")
    assert list_response.status_code == 200
    assert list_response.json() == []


def test_direct_chat_without_llm_returns_configuration_boundary_not_placeholder(client):
    session = client.post("/api/chat/sessions", json={"title": "通用问题"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "秦始皇统一六国的顺序是？最难的是哪个国家？秦始皇下一任是谁？",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "direct_completed"
    assert "当前没有可用的 LLM 配置" in assistant["content"]
    assert "我先根据你这轮的问题" not in assistant["content"]
    assert "我会按普通聊天方式先回答" not in assistant["content"]


def test_chat_memory_only_records_low_risk_preferences(client):
    session = client.post("/api/chat/sessions", json={"title": "偏好收敛"}).json()
    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "以后不要把这次工具误判写进长期记忆，但默认中文，Markdown，先总结再展开，并列出处。",
            "attachments": [],
            "selected_document_ids": [],
        },
    )
    assert response.status_code == 200

    memory_response = client.get(f"/api/chat/sessions/{session['id']}/memory")
    assert memory_response.status_code == 200
    summaries = [item["summary"] for item in memory_response.json()["items"]]
    assert "默认使用中文回答。" in summaries
    assert "回答时优先使用 Markdown 格式。" in summaries
    assert "回答时优先先总结再展开。" in summaries
    assert "回答时优先给出引用或出处。" in summaries
    assert all("工具误判" not in summary for summary in summaries)


def test_chat_reports_selected_document_retrieval_failure_without_fake_fallback(client, monkeypatch):
    document = _upload_document(client, monkeypatch)
    create_response = client.post("/api/chat/sessions", json={"title": "降级测试"})
    assert create_response.status_code == 200
    session_id = create_response.json()["id"]

    monkeypatch.setattr(
        RagService,
        "retrieve_evidence",
        lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("milvus unavailable")),
    )

    send_response = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={
            "content": "请结合论文回答这个问题。",
            "attachments": [],
            "selected_document_ids": [document["id"]],
        },
    )
    assert send_response.status_code == 200
    assistant = send_response.json()["assistant_message"]
    assert assistant["retrieval_status"] == "degraded"
    assert assistant["action_status"] == "failed"
    assert assistant["warning"] in (None, "")
    assert "无法完成基于所选论文正文的分析" in assistant["content"]
    assert "不会改用普通聊天" in assistant["content"]
    assert "milvus unavailable" not in assistant["content"]
    assert "RuntimeError" not in assistant["content"]


def test_chat_local_pdf_attachment_does_not_trigger_library_ingestion(client):
    create_response = client.post("/api/chat/sessions", json={"title": "本地 PDF 附件"})
    assert create_response.status_code == 200
    session_id = create_response.json()["id"]

    send_response = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={
            "content": "请基于我刚上传的 PDF 给我一个概览。",
            "attachments": [
                {
                    "id": "local-pdf-1",
                    "kind": "uploaded_pdf",
                    "display_name": "draft-paper.pdf",
                    "mime_type": "application/pdf",
                    "status": "ready",
                    "metadata": {
                        "filename": "draft-paper.pdf",
                        "size": 1024,
                    },
                }
            ],
            "selected_document_ids": [],
        },
    )
    assert send_response.status_code == 200
    payload = send_response.json()
    user_message = payload["user_message"]
    assistant = payload["assistant_message"]

    assert user_message["attachments"][0]["display_name"] == "draft-paper.pdf"
    assert user_message["attachments"][0]["document_id"] is None
    assert assistant["retrieval_status"] == "skipped"
    assert assistant["warning"] in (None, "")
    assert assistant["used_document_ids"] == []
    assert any("draft-paper.pdf" in item["summary"] for item in payload["memory_snapshot"]["items"])


def test_knowledge_research_task_request_redirects_when_upgrade_disabled(client, monkeypatch):
    monkeypatch.setenv("ENABLE_RESEARCH_FROM_KNOWLEDGE", "false")
    session = client.post("/api/chat/sessions", json={"title": "Research redirect"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "请按研究任务执行，分步骤完成研究计划",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "research_task_redirect"
    assert "不会直接启动实验性的 Research Task Agent Loop" in assistant["content"]


def test_chat_context_compacts_long_history_into_runtime_context_files(client):
    from app.api.main import get_context_file_store
    from app.config import get_settings

    settings = get_settings()
    settings.max_context_tokens = 700
    settings.response_reserve_tokens = 120
    settings.compact_warn_ratio = 0.45
    settings.compact_force_ratio = 0.6
    settings.recent_turns_min = 1
    settings.max_evidence_chars_per_item = 80

    create_response = client.post("/api/chat/sessions", json={"title": "长上下文压缩测试"})
    assert create_response.status_code == 200
    session_id = create_response.json()["id"]

    for index in range(5):
        send_response = client.post(
            f"/api/chat/sessions/{session_id}/messages",
            json={
                "content": (
                    f"第 {index + 1} 轮，请继续记住这个需求：以后默认中文、要保留引用，"
                    + "并围绕 RAG 证据压缩、历史摘要与上下文窗口控制展开说明。" * 12
                ),
                "attachments": [],
                "selected_document_ids": [],
            },
        )
        assert send_response.status_code == 200

    context_response = client.get(f"/api/chat/sessions/{session_id}/context")
    assert context_response.status_code == 200
    context_state = context_response.json()
    assert context_state["stage"] in {"history_compacted", "truncated"}

    file_store = get_context_file_store()
    compact_dir = file_store.get_session_dir(session_id) / "compact"
    compact_files = list(compact_dir.glob("compact-*.md"))
    assert compact_files

    session_summary = (file_store.get_session_dir(session_id) / "session.md").read_text(encoding="utf-8")
    assert "历史长对话已压缩为 compact 摘要" in session_summary


def test_chat_message_stream_emits_processing_delta_and_done(client):
    session = client.post("/api/chat/sessions", json={"title": "stream"}).json()

    with client.stream(
        "POST",
        f"/api/chat/sessions/{session['id']}/messages/stream",
        json={
            "content": "stream this answer",
            "attachments": [],
            "selected_document_ids": [],
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "event: status" in body
    assert "event: assistant_delta" in body
    assert "event: done" in body

    detail = client.get(f"/api/chat/sessions/{session['id']}").json()
    assert len(detail["messages"]) == 2
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][1]["role"] == "assistant"
    assert detail["messages"][1]["content"]


def test_chat_service_llm_call_forwards_streaming_deltas(monkeypatch):
    class FakeDelta:
        def __init__(self, content: str) -> None:
            self.content = content

    class FakeChoice:
        def __init__(self, content: str) -> None:
            self.delta = FakeDelta(content)

    class FakeChunk:
        def __init__(self, content: str) -> None:
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            return iter([FakeChunk("逐"), FakeChunk("字"), FakeChunk("输出")])

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            _ = kwargs
            self.chat = FakeChat()

    monkeypatch.setattr("app.services.chat_service.OpenAI", FakeOpenAI)
    service = get_chat_service()
    service.api_key = "stream-test-key"
    deltas: list[str] = []

    response = service._call_llm(
        [{"role": "user", "content": "stream"}],
        has_images=False,
        delta_sink=deltas.append,
    )

    assert response == "逐字输出"
    assert deltas == ["逐", "字", "输出"]
