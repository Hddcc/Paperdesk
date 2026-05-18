from app.runtime.knowledge_agent_runtime import KnowledgeAgentRuntime, _ReactObservation


def test_react_observation_payload_falls_back_to_legacy_payload():
    observation = _ReactObservation(
        tool="library.explorer.find_documents",
        status="completed",
        summary="Found documents.",
        payload={"document_ids": ["doc-1"], "legacy_only": True},
    )

    assert KnowledgeAgentRuntime._react_observation_payload(observation) == {
        "document_ids": ["doc-1"],
        "legacy_only": True,
    }


def test_react_observation_payload_merges_standard_data_over_payload():
    observation = _ReactObservation(
        tool="library.explorer.find_documents",
        status="completed",
        summary="Found documents.",
        payload={"document_ids": ["old-doc"], "legacy_only": True},
        observation={"data": {"document_ids": ["new-doc"], "standard_only": 1}},
    )

    assert KnowledgeAgentRuntime._react_observation_payload(observation) == {
        "document_ids": ["new-doc"],
        "legacy_only": True,
        "standard_only": 1,
    }


def test_react_observation_evidence_prefers_standard_evidence():
    observation = _ReactObservation(
        tool="evidence.retriever.search",
        status="completed",
        summary="Retrieved evidence.",
        payload={"evidence_items": [{"id": "legacy-evidence"}]},
        observation={"evidence": [{"id": "standard-evidence"}]},
    )

    assert KnowledgeAgentRuntime._react_observation_evidence(observation) == [{"id": "standard-evidence"}]


def test_react_observation_evidence_falls_back_to_payload_items():
    observation = _ReactObservation(
        tool="evidence.retriever.search",
        status="completed",
        summary="Retrieved evidence.",
        payload={"evidence_items": [{"id": "legacy-evidence"}]},
    )

    assert KnowledgeAgentRuntime._react_observation_evidence(observation) == [{"id": "legacy-evidence"}]


def test_react_observation_verification_keeps_legacy_payload_fields():
    observation = _ReactObservation(
        tool="library.operator.clear_categories",
        status="completed",
        summary="Cleared categories.",
        payload={
            "verification": {"method": "payload_check"},
            "verified_state": {"tagged_document_count": 0},
            "verification_error": "secondary check failed",
            "rollback_error": "rollback not attempted",
        },
        observation={"verification": {"performed": True, "success": False}},
    )

    assert KnowledgeAgentRuntime._react_observation_verification(observation) == {
        "performed": True,
        "success": False,
        "verified_state": {"tagged_document_count": 0},
        "verification_error": "secondary check failed",
        "rollback_error": "rollback not attempted",
    }
