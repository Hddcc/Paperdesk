"""Service-layer lifecycle dispatcher used by ChatService."""

from __future__ import annotations

from typing import Any

from app.models import RuntimeRequest, RuntimeResult
from .dispatcher import RuntimeDispatcher, default_runtime_dispatcher


class AgentRuntimeDispatchService:
    """Dispatch lifecycle requests to route-specific adapter runtimes."""

    def __init__(self, dispatcher: RuntimeDispatcher | None = None) -> None:
        self.dispatcher = dispatcher or default_runtime_dispatcher()

    def dispatch(self, request: RuntimeRequest) -> RuntimeResult:
        return self.dispatcher.dispatch(request)
