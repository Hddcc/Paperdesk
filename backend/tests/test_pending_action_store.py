from app.config import Settings
from app.runtime.pending_action_store import PendingActionStore
from app.services.context_file_store import ContextFileStore


def _store(sandbox_dir):
    settings = Settings(runtime_context_dir=str(sandbox_dir / "runtime" / "context"))
    return PendingActionStore(ContextFileStore(settings))


def test_pending_action_store_writes_and_reads_equivalent_payload(sandbox_dir):
    store = _store(sandbox_dir)
    payload = {"type": "tool_action", "confirmation_phrase": "确认删除", "nested": {"count": 2}}

    store.write("session-1", payload)

    assert store.read("session-1") == payload


def test_pending_action_store_clear_removes_pending_file(sandbox_dir):
    store = _store(sandbox_dir)
    store.write("session-1", {"type": "tool_action"})

    store.clear("session-1")

    assert store.read("session-1") is None


def test_pending_action_store_read_missing_file_returns_none(sandbox_dir):
    store = _store(sandbox_dir)

    assert store.read("missing-session") is None


def test_pending_action_store_path_keeps_legacy_filename(sandbox_dir):
    store = _store(sandbox_dir)

    assert store.path_for("session-1").name == "pending_knowledge_action.json"
