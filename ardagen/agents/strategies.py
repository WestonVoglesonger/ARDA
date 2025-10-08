"""
Agent strategy protocols for ARDA stages.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from ..domain import FeedbackDecision


class StageAgentStrategy(Protocol):
    """Callable capable of producing a stage output."""

    async def __call__(self, context: Mapping[str, Any]) -> Any:
        ...


class FeedbackAgentStrategy(Protocol):
    """Callable capable of producing a feedback decision."""

    async def __call__(self, context: Mapping[str, Any]) -> FeedbackDecision:
        ...
