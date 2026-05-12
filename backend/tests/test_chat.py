from io import BytesIO
import time

import fitz

from app.services.embedding_service import EmbeddingService
from app.services.rag_service import RagService


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


def test_chat_degrades_gracefully_when_retrieval_breaks(client, monkeypatch):
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
    assert "知识库检索暂不可用" in assistant["warning"]


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


def test_chat_context_compacts_long_history_into_claude_session_files(client):
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
