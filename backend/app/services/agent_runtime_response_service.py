"""Compatibility wrapper for Agent Core runtime response recording."""

from app.agent.observability.response import AgentRuntimeResponseRecorder, request_trace_event

__all__ = ["AgentRuntimeResponseRecorder", "request_trace_event"]
