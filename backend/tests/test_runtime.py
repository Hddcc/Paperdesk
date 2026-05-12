from datetime import datetime, timezone

from app.models import (
    AgentTask,
    AgentTaskStatus,
    SubagentProfile,
    TaskArtifactRef,
    TaskExecutionTrace,
    TaskNotification,
    ToolPolicy,
    TraceEventType,
)
from app.repositories import SQLiteRepository
from app.runtime import TaskRegistry


def test_runtime_repository_and_registry_support_continue_flow(sandbox_dir):
    repository = SQLiteRepository(sandbox_dir / "runtime.db")
    repository.research.create_run("run-1", "runtime topic")

    task = AgentTask(
        id="task-1",
        run_id="run-1",
        parent_task_id="todo-1",
        profile=SubagentProfile.EXPLORE,
        goal="Inspect online evidence",
        context_bundle={"topic": "runtime topic"},
        done_criteria="Return a concise evidence summary.",
        tool_policy=ToolPolicy(read_only=True, network_allowed=True),
        artifact_dir="scratch/t1",
    )

    stored = repository.runtime.create_task(task)
    assert stored.status == AgentTaskStatus.CREATED

    registry = TaskRegistry(repository.runtime)
    continued = registry.continue_task(task.id)
    assert continued is not None
    assert continued.id == task.id

    notification = TaskNotification(
        task_id=task.id,
        agent_profile=SubagentProfile.EXPLORE,
        status=AgentTaskStatus.COMPLETED,
        summary="Collected evidence.",
        result_payload={"paper_records": [{"title": "Paper"}]},
        token_usage={"result_items": 1},
        artifact_refs=[
            TaskArtifactRef(
                name="papers.json",
                path=str(sandbox_dir / "papers.json"),
                kind="json",
                description="Collected paper candidates",
            )
        ],
        created_at=datetime.now(timezone.utc),
    )
    repository.runtime.record_notification("run-1", notification)
    repository.runtime.save_artifacts("run-1", task.id, notification.artifact_refs)
    repository.runtime.append_trace(
        TaskExecutionTrace(
            run_id="run-1",
            task_id=task.id,
            trace_type=TraceEventType.CONTROL,
            status="continue",
            message="Continuing the existing task context.",
            payload={"continued": True},
            created_at=datetime.now(timezone.utc),
        )
    )

    notifications = repository.runtime.list_notifications("run-1", task_id=task.id)
    artifacts = repository.runtime.list_artifacts("run-1", task_id=task.id)
    traces = repository.runtime.list_traces("run-1", task_id=task.id)

    assert len(notifications) == 1
    assert notifications[0].summary == "Collected evidence."
    assert "<task-notification>" in notifications[0].to_xml_block()
    assert len(artifacts) == 1
    assert artifacts[0].name == "papers.json"
    assert len(traces) == 1
    assert traces[0].status == "continue"
