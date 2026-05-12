"""Coordinator-side decision helpers for the main agent runtime."""

from __future__ import annotations

from uuid import uuid4

from app.models import (
    AgentTask,
    CoordinatorDecision,
    ResearchRequest,
    SubagentProfile,
    TaskNotification,
    TodoTask,
    ToolPolicy,
)


class MainAgentRuntime:
    """Apply simple Claude Code-style delegation rules for PaperDesk."""

    def decide(self, request: ResearchRequest) -> CoordinatorDecision:
        compact_topic = "".join(request.topic.split())
        if len(compact_topic) <= 8 and not request.notes:
            return CoordinatorDecision(
                action="direct_execute",
                reason="Topic is small enough for a single-task direct path.",
                spawn_subagents=False,
            )
        return CoordinatorDecision(
            action="plan_and_spawn",
            reason="Research topic benefits from read-only explore workers.",
            spawn_subagents=True,
            profile=SubagentProfile.EXPLORE,
        )

    def build_explore_task(
        self,
        *,
        run_id: str,
        parent_task: TodoTask,
        channel: str,
        context_bundle: dict,
        done_criteria: str,
    ) -> AgentTask:
        return AgentTask(
            id=str(uuid4()),
            run_id=run_id,
            parent_task_id=parent_task.id,
            profile=SubagentProfile.EXPLORE,
            goal=f"{parent_task.title} [{channel}]",
            context_bundle={
                "channel": channel,
                "todo_task": parent_task.model_dump(mode="json"),
                **context_bundle,
            },
            done_criteria=done_criteria,
            tool_policy=ToolPolicy(
                read_only=True,
                network_allowed=(channel == "online"),
                workspace_write=False,
                db_write=False,
            ),
            artifact_dir=f"scratch/{parent_task.id}/{channel}",
        )

    def build_verify_task(
        self,
        *,
        run_id: str,
        parent_task: TodoTask,
        notifications: list[TaskNotification],
    ) -> AgentTask:
        return AgentTask(
            id=str(uuid4()),
            run_id=run_id,
            parent_task_id=parent_task.id,
            profile=SubagentProfile.VERIFY,
            goal=f"Verify evidence consistency for {parent_task.title}",
            context_bundle={
                "todo_task": parent_task.model_dump(mode="json"),
                "notifications": [notification.model_dump(mode="json") for notification in notifications],
            },
            done_criteria="Flag whether the gathered evidence is empty, conflicting, or safe to merge.",
            tool_policy=ToolPolicy(
                read_only=True,
                network_allowed=False,
                workspace_write=False,
                db_write=False,
            ),
            artifact_dir=f"scratch/{parent_task.id}/verify",
        )

    @staticmethod
    def should_verify(notifications: list[TaskNotification]) -> bool:
        if not notifications:
            return True
        return any(not notification.result_payload for notification in notifications)

    @staticmethod
    def format_notifications_xml(notifications: list[TaskNotification]) -> str:
        return "\n".join(notification.to_xml_block() for notification in notifications)
