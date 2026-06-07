"""Compatibility wrapper for Agent Core runtime dispatcher."""

from app.agent.runtimes.dispatcher import (
    DEFAULT_ROUTE_RUNTIME_MAP,
    RuntimeCallable,
    RuntimeDispatcher,
    RuntimeHandler,
    default_runtime_dispatcher,
)

__all__ = [
    "DEFAULT_ROUTE_RUNTIME_MAP",
    "RuntimeCallable",
    "RuntimeDispatcher",
    "RuntimeHandler",
    "default_runtime_dispatcher",
]
