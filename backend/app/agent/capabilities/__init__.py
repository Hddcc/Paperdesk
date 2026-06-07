"""Capability registry for PaperDesk's extension-ready Agent Core."""

from .defaults import (
    default_capability_registry,
    drawio_capability_declaration,
    paper_capability_declaration,
    token_usage_capability_declaration,
)
from .registry import CapabilityRegistry

__all__ = [
    "CapabilityRegistry",
    "default_capability_registry",
    "drawio_capability_declaration",
    "paper_capability_declaration",
    "token_usage_capability_declaration",
]
