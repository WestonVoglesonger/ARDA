"""
Structured observability manager for ARDA stages.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from .events import ObservabilityEvent, ObservabilityEventType


class ObservabilityManager:
    """Collects orchestration events and forwards them to tracing emitters."""

    def __init__(
        self,
        trace_emitter: Optional[Callable[[str, str, str, str], Any]] = None,
    ) -> None:
        self._trace_emitter = trace_emitter
        self._events: List[ObservabilityEvent] = []

    @property
    def events(self) -> List[ObservabilityEvent]:
        return list(self._events)

    def stage_started(self, stage: str, attempt: int) -> None:
        self._record(ObservabilityEventType.STAGE_STARTED, stage, {"attempt": attempt})

    def stage_completed(self, stage: str, result: Any) -> None:
        payload = {"result": self._maybe_serialize(result)}
        self._record(ObservabilityEventType.STAGE_COMPLETED, stage, payload)

    def stage_failed(self, stage: str, error: str) -> None:
        self._record(ObservabilityEventType.STAGE_FAILED, stage, {"error": error})

    def tool_invoked(self, stage: str, tool_name: str, metadata: Dict[str, Any]) -> None:
        payload = {"tool": tool_name, **metadata}
        self._record(ObservabilityEventType.TOOL_INVOKED, stage, payload)

    def custom_event(self, stage: str, payload: Dict[str, Any]) -> None:
        self._record(ObservabilityEventType.CUSTOM, stage, payload)

    def _record(self, event_type: ObservabilityEventType, stage: str, payload: Dict[str, Any]) -> None:
        event = ObservabilityEvent(event_type=event_type, stage=stage, payload=payload)
        self._events.append(event)
        if self._trace_emitter:
            try:
                self._trace_emitter(
                    stage,
                    stage,
                    event_type.value,
                    json.dumps(payload, default=self._default_json),
                )
            except Exception:
                # Observability should never break pipeline execution; swallow errors.
                pass

    @staticmethod
    def _maybe_serialize(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        return value

    @staticmethod
    def _default_json(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        return str(value)
