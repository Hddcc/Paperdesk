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
from app.api.main import get_chat_service, get_knowledge_agent_runtime, get_repository
from app.models import ChunkRecord
from app.models import LibraryDocument
from app.runtime.knowledge_agent_runtime import KnowledgeAgentResult, KnowledgeAgentRuntime, _ReactAction
from app.runtime.agent_orchestrator import AgentOrchestrator, _ModeCandidate


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


def test_selected_document_multi_question_drafts_answer_after_retrieval(client, monkeypatch):
    document = _create_ready_document("doc-compound-rag", "CompoundRag.pdf", title="Compound RAG Paper")
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


def test_selected_library_documents_do_not_route_direct_when_llm_suggests_direct(client, monkeypatch):
    document = _create_ready_document("doc-selected-direct-guard", "SelectedDirectGuard.pdf")
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
    assert any(trace.status == "agent_mode_selected" and trace.payload["mode"] == "REACT" for trace in traces)
    assert any(trace.status == "react_action_planned" for trace in traces)
    assert any(trace.status == "reflection_result_created" for trace in traces)


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


def test_low_score_reflection_runs_one_improvement_and_records_lesson(client, monkeypatch):
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
    assert calls["count"] == 2
    assert "自检" not in assistant["content"]
    assert "补充检索后的回答" in assistant["content"]
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert sum(1 for trace in traces if trace.status == "reflection_improvement_started") == 1
    assert any(trace.status == "reflection_improvement_finished" for trace in traces)
    assert any(trace.status == "reflection_result_created" for trace in traces)
    assert any("反思经验" in item["summary"] for item in payload["memory_snapshot"]["items"])


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


def test_chat_agent_clears_all_categories_and_reports_zero_without_process_wording(client):
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
    assert clear_assistant["action_status"] == "completed"
    assert clear_payload["library_mutated"] is True
    assert "已清空 2 篇论文的分类/标签关系" in clear_assistant["content"]
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
    assert assistant["action_status"] == "completed"
    assert payload["library_mutated"] is True
    assert "已清空 2 篇论文的分类/标签关系" in assistant["content"]
    assert get_repository().get_document(french_a.id).categories == []
    assert get_repository().get_document(french_b.id).categories == []
    assert [category.name for category in get_repository().get_document(other.id).categories] == ["英语"]


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
    assert assistant["action_status"] == "completed"
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


def test_chat_agent_writes_report_from_all_chinese_category_documents_with_real_abstracts(client):
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
    assert assistant["action_status"] == "completed"
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
    assert any(trace.status == "agent_mode_selected" and trace.payload["mode"] == "PLANNER" for trace in traces)
    assert any(trace.status == "planner_plan_created" for trace in traces)
    assert any(trace.status == "reflection_result_created" for trace in traces)
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
