from datetime import datetime, timezone

from app.api.main import get_knowledge_agent_runtime, get_repository
from app.models import AgentRunMode, LibraryDocument
from app.runtime.agent_orchestrator import AgentOrchestrator, _ModeCandidate
from app.runtime.knowledge_agent_runtime import KnowledgeAgentRuntime, _ReactAction


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


def test_llm_tool_plan_resolves_tag_collection_before_report(client, monkeypatch):
    get_knowledge_agent_runtime().api_key = "test-key"
    tag = get_repository().category.create_category("plan-tag", "#6957d8")
    docs = [
        _create_ready_document("doc-plan-tag-a", "PlanA.pdf", title="Plan Paper A"),
        _create_ready_document("doc-plan-tag-b", "PlanB.pdf", title="Plan Paper B"),
    ]
    other = _create_ready_document("doc-plan-tag-other", "PlanOther.pdf", title="Other Paper")
    for document in docs:
        get_repository().category.replace_document_categories(document.id, [tag.id])
    session = client.post("/api/chat/sessions", json={"title": "LLM tool plan tag report"}).json()

    def planned_action(self, *, session, content, selected_document_ids, attachments, observations):
        if not observations:
            return _ReactAction(
                tool="library.explorer.find_documents",
                arguments={"category_names": ["plan-tag"], "expected": "many"},
                rationale="Resolve the tag entity before analysis.",
                task_intent={
                    "task_type": "collection_analysis",
                    "entities": [{"text": "plan-tag", "type": "tag", "role": "filter", "confidence": 0.99}],
                    "requested_output": "analysis_report",
                },
                action_plan=[
                    {"tool": "library.explorer.find_documents", "arguments": {"category_names": ["plan-tag"]}},
                    {"tool": "report.drafter.write", "arguments": {}},
                ],
                confidence=0.95,
            )
        return _ReactAction(
            tool="final.answer",
            arguments={"content": "Use observations to answer."},
            rationale="Finish from repository observations.",
            task_intent={"task_type": "collection_analysis"},
            confidence=0.95,
        )

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", planned_action)
    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: _ModeCandidate(
            mode=AgentRunMode.REACT,
            reason="LLM recognized a labeled collection analysis.",
            confidence=0.95,
            target_runtime="KnowledgeAgentRuntime",
            source="llm",
            task_type="collection_analysis",
            needs_tool=True,
        ),
    )

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "analyze all papers under the plan-tag tag and write a report",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    content = assistant["content"]
    assert "plan-tag" in content
    assert "PlanA.pdf" in content
    assert "PlanB.pdf" in content
    assert "PlanOther.pdf" not in content
    assert other.id not in assistant["used_document_ids"]
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(
        trace.status == "react_action_planned"
        and trace.payload.get("planning_source") == "llm"
        and trace.payload.get("task_intent", {}).get("task_type") == "collection_analysis"
        for trace in traces
    )


def test_llm_tool_plan_renames_tag_and_preserves_links(client, monkeypatch):
    get_knowledge_agent_runtime().api_key = "test-key"
    source = get_repository().category.create_category("old-plan-tag", "#6957d8")
    docs = [
        _create_ready_document("doc-rename-plan-a", "RenameA.pdf"),
        _create_ready_document("doc-rename-plan-b", "RenameB.pdf"),
    ]
    for document in docs:
        get_repository().category.replace_document_categories(document.id, [source.id])
    session = client.post("/api/chat/sessions", json={"title": "LLM tool plan rename"}).json()

    def planned_action(self, *, session, content, selected_document_ids, attachments, observations):
        if not observations:
            return _ReactAction(
                tool="library.operator.rename_category",
                arguments={"source_category_name": "old-plan-tag", "target_category_name": "new-plan-tag"},
                rationale="Rename the tag while preserving document links.",
                task_intent={
                    "task_type": "tag_rename",
                    "entities": [
                        {"text": "old-plan-tag", "type": "tag", "role": "source", "confidence": 0.99},
                        {"text": "new-plan-tag", "type": "tag", "role": "target", "confidence": 0.99},
                    ],
                    "needs_verification": True,
                },
                action_plan=[
                    {
                        "tool": "library.operator.rename_category",
                        "arguments": {"source_category_name": "old-plan-tag", "target_category_name": "new-plan-tag"},
                        "requires_verification_after": True,
                    }
                ],
                confidence=0.96,
            )
        return _ReactAction(
            tool="final.answer",
            arguments={"content": "Use observations to answer."},
            rationale="Finish after verified write observation.",
            task_intent={"task_type": "tag_rename"},
            confidence=0.95,
        )

    monkeypatch.setattr(KnowledgeAgentRuntime, "_next_react_action_with_llm", planned_action)
    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: _ModeCandidate(
            mode=AgentRunMode.REACT,
            reason="LLM recognized a tag rename write.",
            confidence=0.95,
            target_runtime="KnowledgeAgentRuntime",
            source="llm",
            task_type="tag_rename",
            needs_tool=True,
            risk_level="write",
        ),
    )

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "content": "make old-plan-tag become new-plan-tag",
            "attachments": [],
            "selected_document_ids": [],
        },
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert assistant["action_status"] == "completed"
    categories = get_repository().category.list_categories()
    assert not any(category.name == "old-plan-tag" for category in categories)
    target = next(category for category in categories if category.name == "new-plan-tag")
    for document in docs:
        linked = get_repository().category.list_document_categories(document.id)
        assert [category.id for category in linked] == [target.id]
    traces = get_repository().runtime.list_traces(assistant["agent_trace_id"])
    assert any(trace.status == "library_write_verified" for trace in traces)
