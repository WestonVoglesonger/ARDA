"""
Agent registry responsible for mapping stages to concrete strategies.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from ..domain import FeedbackDecision
from .strategies import FeedbackAgentStrategy, StageAgentStrategy


StageCallable = Callable[[Mapping[str, Any]], Awaitable[Any] | Any]
FeedbackCallable = Callable[[Mapping[str, Any]], Awaitable[FeedbackDecision] | FeedbackDecision]


@dataclass
class AgentDefinition:
    """Metadata describing a registered agent handler."""

    name: str
    stage: str
    handler: StageAgentStrategy
    description: Optional[str] = None


def _ensure_coroutine(func: StageCallable | FeedbackCallable):
    if asyncio.iscoroutinefunction(func):
        return func

    async def wrapper(context: Mapping[str, Any]):
        return func(context)

    return wrapper


class AgentRegistry:
    """Registry of stage and feedback agent handlers."""

    def __init__(self) -> None:
        self._stage_agents: Dict[str, AgentDefinition] = {}
        self._feedback_agent: Optional[FeedbackAgentStrategy] = None

    def register_stage_agent(
        self,
        stage: str,
        handler: StageCallable,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        coroutine_handler = _ensure_coroutine(handler)
        definition = AgentDefinition(
            name=name or stage,
            stage=stage,
            handler=coroutine_handler,
            description=description,
        )
        self._stage_agents[stage] = definition

    def register_feedback_agent(
        self,
        handler: FeedbackCallable,
        *,
        name: str = "feedback",
    ) -> None:
        self._feedback_agent = _ensure_coroutine(handler)  # type: ignore[assignment]

    def get_stage_agent(self, stage: str) -> StageAgentStrategy:
        if stage not in self._stage_agents:
            raise KeyError(f"No agent registered for stage '{stage}'.")
        return self._stage_agents[stage].handler

    def get_feedback_agent(self) -> FeedbackAgentStrategy:
        if self._feedback_agent is None:
            raise KeyError("No feedback agent registered.")
        return self._feedback_agent


def create_default_registry() -> AgentRegistry:
    """Create a registry populated with deterministic placeholder agents."""

    from ..domain import (
        EvaluateResults,
        FeedbackDecision,
        LintResults,
        MicroArchConfig,
        QuantConfig,
        RTLConfig,
        SpecContract,
        SynthResults,
        VerifyResults,
    )
    from ..tools import lint, simulation, synthesis, reporting

    registry = AgentRegistry()

    # Spec ---------------------------------------------------------------------
    def spec_agent(context: Mapping[str, Any]) -> SpecContract:
        return lint.build_spec_contract(context)

    registry.register_stage_agent("spec", spec_agent, description="Generate hardware specification")

    # Quant --------------------------------------------------------------------
    def quant_agent(context: Mapping[str, Any]) -> QuantConfig:
        return lint.build_quant_config(context)

    registry.register_stage_agent("quant", quant_agent, description="Quantize design parameters")

    # Micro-architecture -------------------------------------------------------
    def microarch_agent(context: Mapping[str, Any]) -> MicroArchConfig:
        return lint.build_microarch_config(context)

    registry.register_stage_agent("microarch", microarch_agent, description="Synthesize micro-architecture directives")

    # RTL ----------------------------------------------------------------------
    def rtl_agent(context: Mapping[str, Any]) -> RTLConfig:
        return lint.build_rtl_config(context)

    registry.register_stage_agent("rtl", rtl_agent, description="Generate RTL artifacts")

    # Static checks ------------------------------------------------------------
    def static_checks_agent(context: Mapping[str, Any]) -> LintResults:
        return lint.run_static_checks(context)

    registry.register_stage_agent("static_checks", static_checks_agent, description="Run lint and style analysis")

    # Verification -------------------------------------------------------------
    def verification_agent(context: Mapping[str, Any]) -> VerifyResults:
        return simulation.run_verification(context)

    registry.register_stage_agent("verification", verification_agent, description="Execute simulation-based verification")

    # Synthesis ----------------------------------------------------------------
    def synth_agent(context: Mapping[str, Any]) -> SynthResults:
        return synthesis.run_synthesis(context)

    registry.register_stage_agent("synth", synth_agent, description="Launch synthesis backend")

    # Evaluation ---------------------------------------------------------------
    def evaluate_agent(context: Mapping[str, Any]) -> EvaluateResults:
        return reporting.build_evaluation_summary(context)

    registry.register_stage_agent("evaluate", evaluate_agent, description="Aggregate reports into scorecard")

    # Feedback -----------------------------------------------------------------
    def feedback_agent(context: Mapping[str, Any]) -> FeedbackDecision:
        return FeedbackDecision(action="continue")

    registry.register_feedback_agent(feedback_agent)

    return registry
