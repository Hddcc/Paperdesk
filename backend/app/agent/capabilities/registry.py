"""Capability registry used by Agent lifecycle and documentation surfaces."""

from __future__ import annotations

from collections.abc import Iterable

from app.models import CapabilityDeclaration, CapabilityResolution


class CapabilityRegistry:
    """In-memory registry for extension capability declarations."""

    def __init__(self, declarations: Iterable[CapabilityDeclaration] | None = None) -> None:
        self._capabilities: dict[str, CapabilityDeclaration] = {}
        for declaration in declarations or []:
            self.register(declaration)

    def register(self, declaration: CapabilityDeclaration) -> None:
        self._capabilities[declaration.capability_id] = declaration

    def get(self, capability_id: str) -> CapabilityDeclaration | None:
        capability = self._capabilities.get(capability_id)
        if capability is None or not capability.enabled:
            return None
        return capability

    def get_declared(self, capability_id: str) -> CapabilityDeclaration | None:
        return self._capabilities.get(capability_id)

    def list_all(self) -> list[CapabilityDeclaration]:
        return sorted(self._capabilities.values(), key=lambda item: item.capability_id)

    def list_enabled(self) -> list[CapabilityDeclaration]:
        return [item for item in self.list_all() if item.enabled]

    def resolve(self, capability_id: str, *, fallback: str = "chat") -> CapabilityResolution:
        capability = self.get(capability_id)
        if capability is not None:
            return CapabilityResolution(
                capability_id=capability.capability_id,
                declaration=capability,
                enabled=True,
                reason="capability resolved",
            )
        fallback_capability = self.get(fallback)
        if fallback_capability is not None:
            return CapabilityResolution(
                capability_id=fallback_capability.capability_id,
                declaration=fallback_capability,
                enabled=True,
                reason=f"capability {capability_id!r} unavailable; using {fallback!r}",
            )
        return CapabilityResolution(
            capability_id=capability_id,
            declaration=None,
            enabled=False,
            reason="capability is not registered or not enabled",
        )
