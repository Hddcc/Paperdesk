from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.models import ChatMessage
from app.services.context_compaction_service import ContextCompactionService
from app.services.context_file_store import ContextFileStore


def test_context_compaction_uses_llm_summary_when_api_key_available(
    sandbox_dir: Path,
    monkeypatch,
) -> None:
    settings = Settings(
        runtime_context_dir=str(sandbox_dir / "runtime" / "context"),
        llm_api_key="test-key",
        llm_model="paperdesk-test-model",
        embedding_warmup_on_start=False,
        milvus_uri="http://fake-milvus:19530",
        milvus_auto_start=False,
    )
    settings.recent_turns_min = 1
    file_store = ContextFileStore(settings)
    captured: dict = {}

    class FakeMessage:
        content = "- 用户希望默认使用中文回答\n- 已决定后续回答需要保留论文引用"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured["client_kwargs"] = kwargs
            self.chat = FakeChat()

    monkeypatch.setattr("app.services.context_compaction_service.OpenAI", FakeOpenAI)

    service = ContextCompactionService(
        settings,
        file_store,
        model="paperdesk-test-model",
        api_key="test-key",
        base_url="https://llm.example/v1",
    )
    history = [
        ChatMessage(
            id=f"message-{index}",
            session_id="session-llm-compact",
            role="user" if index % 2 == 0 else "assistant",
            content=f"第 {index} 条旧对话，用户偏好中文回答并要求保留引用。",
        )
        for index in range(6)
    ]

    compacted_ids, summary_lines, filename = service.compact_history(
        "session-llm-compact",
        history=history,
        already_compacted_ids=set(),
    )

    assert filename == "compact-001.md"
    assert summary_lines == ["用户希望默认使用中文回答", "已决定后续回答需要保留论文引用"]
    assert compacted_ids == {"message-0", "message-1"}
    assert captured["model"] == "paperdesk-test-model"
    assert captured["temperature"] == 0.1
    assert captured["client_kwargs"]["base_url"] == "https://llm.example/v1"

    compact_path = (
        file_store.get_session_dir("session-llm-compact")
        / "compact"
        / "compact-001.md"
    )
    compact_text = compact_path.read_text(encoding="utf-8")
    assert "用户希望默认使用中文回答" in compact_text
    assert "已决定后续回答需要保留论文引用" in compact_text
