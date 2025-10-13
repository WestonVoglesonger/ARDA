"""
Agent runner abstractions for orchestrated pipeline execution.

`PipelineAgentRunner` defines the contract used by the simplified pipeline to
obtain stage artefacts and feedback decisions.  `DefaultAgentRunner` provides a
deterministic implementation that keeps the CLI functional without external
agents, while custom runners can integrate real LLM- or tool-backed flows.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol

from ..domain import FeedbackDecision
from ..agents import AgentRegistry, create_default_registry


class PipelineAgentRunner(Protocol):
    """Interface for executing pipeline stages and feedback decisions."""

    async def run_stage(self, stage: str, context: Mapping[str, Any]) -> Any:
        """Return the artefact for a specific pipeline stage."""

    async def run_feedback(self, context: Mapping[str, Any]) -> FeedbackDecision:
        """Return a feedback decision based on current pipeline state."""


class DefaultAgentRunner:
    """
    Deterministic agent runner that consults the agent registry.

    Callers can inject their own registry with fully-fledged agents or rely on
    the default deterministic implementations shipped for demos and testing.
    """

    def __init__(self, registry: Optional[AgentRegistry] = None) -> None:
        self._registry = registry or create_default_registry()

    async def run_stage(self, stage: str, context: Mapping[str, Any]) -> Any:
        # Check if this is a sub-stage (phase within verification)
        phase = context.get("phase")
        if phase:
            # Route to specialized agent (test_generation or simulation)
            handler = self._registry.get_stage_agent(phase)
        else:
            # Route to main stage agent
            handler = self._registry.get_stage_agent(stage)
        
        return await handler(context)

    async def run_feedback(self, context: Mapping[str, Any]) -> FeedbackDecision:
        try:
            handler = self._registry.get_feedback_agent()
        except KeyError:
            return FeedbackDecision(action="continue")
        return await handler(context)
