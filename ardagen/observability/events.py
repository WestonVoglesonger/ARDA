"""
Typed observability event definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class ObservabilityEventType(str, Enum):
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    TOOL_INVOKED = "tool_invoked"
    CUSTOM = "custom"


@dataclass
class ObservabilityEvent:
    """Base event structure captured by the observability manager."""

    event_type: ObservabilityEventType
    stage: str
    payload: Dict[str, Any] = field(default_factory=dict)


__all__ = ["ObservabilityEventType", "ObservabilityEvent"]
