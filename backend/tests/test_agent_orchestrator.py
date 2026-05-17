from app.models import AgentOrchestratorInput, AgentRunMode, MemorySnapshot
from app.repositories import SQLiteRepository
from app.runtime import AgentOrchestrator, SkillRegistry, ToolRegistry


def _orchestrator(sandbox_dir):
    repository = SQLiteRepository(sandbox_dir / "orchestrator.db")
    return AgentOrchestrator(
        research_repository=repository.research,
        runtime_repository=repository.runtime,
        model="test-model",
        api_key=None,
        base_url=None,
    )


def _payload(prompt: str, *, selected_document_ids: list[str] | None = None, has_pending_action: bool = False):
    return AgentOrchestratorInput(
        session_id="session-1",
        message_id="message-1",
        user_prompt=prompt,
        selected_document_ids=selected_document_ids or [],
        memory_snapshot=MemorySnapshot(),
        available_tools=ToolRegistry().list_enabled(),
        available_skills=SkillRegistry().list_enabled(),
        runtime_context={"has_pending_action": has_pending_action},
    )


def test_orchestrator_routes_plain_chat_to_direct(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(_payload("你好，帮我解释一下 RAG 是什么"))

    assert decision.mode == AgentRunMode.DIRECT
    assert decision.trace_id.startswith("chat-orch-")
    assert decision.target_runtime == "DirectChatRuntime"


def test_orchestrator_routes_library_question_to_react(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(_payload("我的论文库里有几篇论文？"))

    assert decision.mode == AgentRunMode.REACT
    assert decision.target_runtime == "KnowledgeAgentRuntime"


def test_orchestrator_keeps_library_read_questions_in_tool_runtime_when_llm_routes_direct(sandbox_dir, monkeypatch):
    orchestrator = _orchestrator(sandbox_dir)
    orchestrator.api_key = "provider-secret"

    from app.runtime.agent_orchestrator import _ModeCandidate

    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: _ModeCandidate(
            mode=AgentRunMode.DIRECT,
            reason="LLM incorrectly treated the tag mapping question as plain chat.",
            confidence=0.95,
            target_runtime="DirectChatRuntime",
        ),
    )

    decision = orchestrator.select_mode(_payload("每篇文章对应的标签是什么？"))

    assert decision.mode == AgentRunMode.REACT
    assert decision.target_runtime == "KnowledgeAgentRuntime"
    assert "library state questions must enter" in decision.reason


def test_orchestrator_routes_multistage_task_to_planner(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(
        _payload("先查哪些论文没有标签，再补上历史标签，然后按标签分别写总结")
    )

    assert decision.mode == AgentRunMode.PLANNER
    assert decision.target_runtime == "KnowledgePlannerRuntime"


def test_orchestrator_routes_user_correction_to_reflection(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(_payload("你刚才回答不对，重新检查一下"))

    assert decision.mode == AgentRunMode.REFLECTION
    assert decision.target_runtime == "ReflectionRuntime"


def test_orchestrator_does_not_route_general_go_explanation_to_reflection(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(_payload("不对这个 Go 代码继续解释一下"))

    assert decision.mode == AgentRunMode.DIRECT
    assert decision.target_runtime == "DirectChatRuntime"


def test_orchestrator_keeps_destructive_actions_on_confirmation_path(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(_payload("帮我删除分类法律"))

    assert decision.mode == AgentRunMode.REACT
    assert decision.initial_context["permission_policy"] == "confirmation_required"


def test_orchestrator_treats_tag_replacement_as_verified_library_write(sandbox_dir, monkeypatch):
    orchestrator = _orchestrator(sandbox_dir)
    orchestrator.api_key = "provider-secret"

    from app.runtime.agent_orchestrator import _ModeCandidate

    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: _ModeCandidate(
            mode=AgentRunMode.DIRECT,
            reason="LLM incorrectly treated tag replacement as plain chat.",
            confidence=0.95,
            target_runtime="DirectChatRuntime",
        ),
    )

    decision = orchestrator.select_mode(_payload("把里面带有hx23标签的论文，标签都换成“更新”"))

    assert decision.mode == AgentRunMode.REACT
    assert decision.target_runtime == "KnowledgeAgentRuntime"
    assert "library write requests must enter" in decision.reason


def test_orchestrator_treats_replacement_synonyms_as_verified_library_write(sandbox_dir, monkeypatch):
    orchestrator = _orchestrator(sandbox_dir)
    orchestrator.api_key = "provider-secret"

    from app.runtime.agent_orchestrator import _ModeCandidate

    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: _ModeCandidate(
            mode=AgentRunMode.DIRECT,
            reason="LLM incorrectly treated tag replacement synonym as plain chat.",
            confidence=0.95,
            target_runtime="DirectChatRuntime",
        ),
    )

    decision = orchestrator.select_mode(_payload("把旧主题标签替换成新主题"))

    assert decision.mode == AgentRunMode.REACT
    assert decision.target_runtime == "KnowledgeAgentRuntime"


def test_orchestrator_falls_back_to_rule_candidate_when_llm_candidate_is_absent(sandbox_dir):
    decision = _orchestrator(sandbox_dir).select_mode(_payload("我的论文库里有几篇论文？"))

    assert decision.mode == AgentRunMode.REACT
    assert decision.fallback_used is False


def test_orchestrator_uses_high_confidence_llm_intention_over_rule_candidate(sandbox_dir, monkeypatch):
    orchestrator = _orchestrator(sandbox_dir)
    orchestrator.api_key = "provider-secret"

    from app.runtime.agent_orchestrator import _ModeCandidate

    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: _ModeCandidate(
            mode=AgentRunMode.PLANNER,
            reason="用户要先筛选标签再写对比。",
            confidence=0.92,
            target_runtime="KnowledgePlannerRuntime",
            required_capabilities=["knowledge_tools", "structured_plan"],
        ),
    )

    decision = orchestrator.select_mode(_payload("为我所有带着中文标签的论文写一篇对比，对比他们的区别"))

    assert decision.mode == AgentRunMode.PLANNER
    assert decision.target_runtime == "KnowledgePlannerRuntime"
    assert decision.reason.startswith("LLM intention decision")


def test_orchestrator_uses_low_confidence_llm_fallback(sandbox_dir, monkeypatch):
    orchestrator = _orchestrator(sandbox_dir)
    orchestrator.api_key = "provider-secret"

    from app.runtime.agent_orchestrator import _ModeCandidate

    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: _ModeCandidate(
            mode=AgentRunMode.PLANNER,
            reason="Low confidence semantic guess.",
            confidence=0.2,
            target_runtime="KnowledgePlannerRuntime",
            source="llm",
        ),
    )

    decision = orchestrator.select_mode(_payload("你好，帮我解释一下 RAG 是什么"))

    assert decision.mode == AgentRunMode.DIRECT
    assert decision.fallback_used is True


def test_orchestrator_keeps_selected_documents_grounded_when_llm_routes_direct(sandbox_dir, monkeypatch):
    orchestrator = _orchestrator(sandbox_dir)
    orchestrator.api_key = "provider-secret"

    from app.runtime.agent_orchestrator import _ModeCandidate

    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: _ModeCandidate(
            mode=AgentRunMode.DIRECT,
            reason="LLM guessed this was plain chat.",
            confidence=0.95,
            target_runtime="DirectChatRuntime",
            source="llm",
        ),
    )

    decision = orchestrator.select_mode(_payload("总结这篇论文", selected_document_ids=["doc-1"]))

    assert decision.mode == AgentRunMode.REACT
    assert decision.target_runtime == "KnowledgeAgentRuntime"
    assert "document observations" in decision.reason


def test_orchestrator_preserves_llm_task_plan_in_initial_context(sandbox_dir, monkeypatch):
    orchestrator = _orchestrator(sandbox_dir)
    orchestrator.api_key = "provider-secret"

    monkeypatch.setattr(
        AgentOrchestrator,
        "_llm_candidate",
        lambda self, payload: self._llm_candidate_from_payload(
            {
                "mode": "PLANNER",
                "reason": "User wants a multi-step tag operation.",
                "confidence": 0.93,
                "risk_level": "write",
                "task_type": "tag_rename",
                "user_intent": "Rename a tag and keep document links visible under the new name.",
                "entities": [
                    {"text": "A", "type": "tag", "role": "source", "confidence": 0.9},
                    {"text": "B", "type": "tag", "role": "target", "confidence": 0.9},
                ],
                "needs_tool": True,
                "needs_plan": True,
                "needs_verification": True,
                "requested_output": "operation_result",
                "required_capabilities": ["label_lookup", "label_update", "write_verification"],
                "action_plan": [
                    {
                        "tool": "library.operator.rename_category",
                        "purpose": "Rename or merge the source tag into the target tag.",
                        "arguments": {"source_category_name": "A", "target_category_name": "B"},
                        "requires_verification_after": True,
                    }
                ],
            }
        ),
    )

    decision = orchestrator.select_mode(_payload("rename tag A to B"))

    assert decision.mode == AgentRunMode.PLANNER
    assert decision.initial_context["user_intent"].startswith("Rename a tag")
    assert decision.initial_context["needs_verification"] is True
    assert decision.initial_context["entities"][0]["text"] == "A"
    assert decision.initial_context["action_plan"][0]["tool"] == "library.operator.rename_category"
