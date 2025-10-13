"""
Pipeline wrapper that drives the orchestrator with an agent strategy.

This implementation provides the `Pipeline` class for running ARDA pipelines with
the CLI and tests, while delegating execution to the modular orchestrator
stages. It supports feedback-driven retries, collects observability events, and
falls back to deterministic placeholder agents when no external strategy is
wired in.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Type

from pydantic import BaseModel

from .core import PipelineOrchestrator, StageExecutionError
from .core.stages import (
    EvaluateStage,
    MicroArchStage,
    ArchitectureStage,
    QuantStage,
    RTLStage,
    VerificationStage,
    SpecStage,
    Stage,
    SynthStage,
)
from .core.strategies import AgentStrategy
from .domain import FeedbackDecision
from .model_config import (
    DEFAULT_FPGA_DEVICE,
    DEFAULT_FPGA_FAMILY,
    DEFAULT_SYNTHESIS_BACKEND,
)
from .observability.manager import ObservabilityManager
from .runtime import PipelineAgentRunner, DefaultAgentRunner
from .workspace import Workspace, workspace_manager


@dataclass(frozen=True)
class StageOrder:
    """Ordered stage metadata for driving restarts."""

    names: Tuple[str, ...]
    index: Dict[str, int]


class Pipeline:
    """
    High-level pipeline runner that coordinates orchestrator stages with feedback.
    """

    _stage_builders: Tuple[Type[Stage], ...] = (
        SpecStage,
        QuantStage,
        MicroArchStage,
        ArchitectureStage,
        RTLStage,
        # StaticChecksStage removed - now part of VerificationStage Phase 1
        VerificationStage,  # Now includes lint + test gen + simulation
        SynthStage,
        EvaluateStage,
    )
    _stage_names: Tuple[str, ...] = tuple(builder().name for builder in _stage_builders)
    _stage_order = StageOrder(names=_stage_names, index={name: idx for idx, name in enumerate(_stage_names)})
    _feedback_stages: frozenset[str] = frozenset(
        {"spec", "quant", "microarch", "architecture", "rtl", "verification", "synth", "evaluate"}
    )
    # Removed "static_checks" - now integrated into verification

    def __init__(
        self,
        *,
        synthesis_backend: str = DEFAULT_SYNTHESIS_BACKEND,
        fpga_family: Optional[str] = None,
        fpga_device: Optional[str] = None,
        observability: Optional[ObservabilityManager] = None,
        agent_runner: Optional[PipelineAgentRunner] = None,
    ) -> None:
        self.synthesis_backend = synthesis_backend
        self.fpga_family = fpga_family or DEFAULT_FPGA_FAMILY
        self.fpga_device = fpga_device or DEFAULT_FPGA_DEVICE
        self.observability = observability or ObservabilityManager()
        self._agent_runner = agent_runner or DefaultAgentRunner()
        self._strategy = self._build_strategy()

        self.results: Dict[str, BaseModel] = {}
        self.stage_attempts: Dict[str, int] = defaultdict(int)
        self.feedback_history: list[FeedbackDecision] = []
        self.workspace_token: Optional[str] = None
        self._bundle: Optional[str] = None

    async def run(self, algorithm_bundle: str) -> Dict[str, Any]:
        """
        Execute the pipeline for the provided algorithm bundle.
        """
        self._reset_state()
        self._bundle = algorithm_bundle

        try:
            self.workspace_token = workspace_manager.ingest_bundle(algorithm_bundle)
        except Exception as exc:  # pragma: no cover - defensive against malformed bundles
            return {
                "success": False,
                "error": f"Failed to ingest bundle: {exc}",
                "details": None,
            }

        run_inputs = {
            "bundle": algorithm_bundle,
            "workspace_token": self.workspace_token,
            "synthesis_backend": self.synthesis_backend,
            "fpga_family": self.fpga_family,
            "fpga_device": self.fpga_device,
        }

        start_index = 0

        while True:
            stages = self._build_stages()[start_index:]
            if not stages:
                break

            initial_results = self._collect_initial_results(start_index)
            orchestrator = PipelineOrchestrator(stages=stages, strategy=self._strategy)
            restart_requested = False

            try:
                async for stage_name, stage_output in orchestrator.run_iter(
                    run_inputs=self._add_feedback_to_run_inputs(run_inputs),
                    initial_results=initial_results,
                ):
                    self.results[stage_name] = stage_output
                    self.observability.stage_completed(stage_name, stage_output)

                    directive = await self._apply_feedback(
                        stage_name,
                        run_inputs,
                        attempt=self.stage_attempts[stage_name],
                    )

                    if directive == "abort":
                        return {
                            "success": False,
                            "error": "Pipeline aborted by feedback agent",
                            "workspace_token": self.workspace_token,
                            "results": self.results,
                            "stage_attempts": dict(self.stage_attempts),
                            "feedback": self.feedback_history,
                        }

                    if isinstance(directive, tuple) and directive[0] == "jump":
                        target_index = directive[1]
                        self._discard_results_from(target_index)
                        start_index = target_index
                        restart_requested = True
                        break
                else:
                    start_index = len(self._stage_builders)

            except StageExecutionError as exc:
                self.observability.stage_failed(exc.stage, str(exc.error))
                directive = await self._apply_feedback(
                    exc.stage,
                    run_inputs,
                    attempt=self.stage_attempts[exc.stage],
                    error=str(exc.error),
                )

                if directive == "abort":
                    return {
                        "success": False,
                        "error": "Pipeline aborted by feedback agent",
                        "workspace_token": self.workspace_token,
                        "results": self.results,
                        "stage_attempts": dict(self.stage_attempts),
                        "feedback": self.feedback_history,
                    }

                if isinstance(directive, tuple) and directive[0] == "jump":
                    target_index = directive[1]
                    self._discard_results_from(target_index)
                    start_index = target_index
                    restart_requested = True
                else:
                    return {
                        "success": False,
                        "error": str(exc.error),
                        "workspace_token": self.workspace_token,
                        "results": self.results,
                        "stage_attempts": dict(self.stage_attempts),
                        "feedback": self.feedback_history,
                    }

            except Exception as exc:
                return {
                    "success": False,
                    "error": str(exc),
                    "workspace_token": self.workspace_token,
                    "results": self.results,
                    "stage_attempts": dict(self.stage_attempts),
                    "feedback": self.feedback_history,
                }

            if restart_requested:
                continue
            if start_index >= len(self._stage_builders):
                break

        # Store state before resetting
        workspace_token = self.workspace_token
        results = dict(self.results)
        stage_attempts = dict(self.stage_attempts)
        feedback_history = list(self.feedback_history)
        observability_events = list(self.observability.events)
        
        # Reset state for next run
        self._reset_state()

        return {
            "success": True,
            "workspace_token": workspace_token,
            "results": results,
            "stage_attempts": stage_attempts,
            "feedback": feedback_history,
            "observability": observability_events,
        }

    # --- Internal helpers ---------------------------------------------------------

    def _reset_state(self) -> None:
        self.results.clear()
        self.stage_attempts.clear()
        self.feedback_history.clear()
        self.workspace_token = None
        self._strategy = self._build_strategy()

    def _build_strategy(self) -> AgentStrategy:
        pipeline = self

        class _Strategy(AgentStrategy):
            async def run(
                self,
                stage: Stage,
                stage_inputs: Dict[str, Any],
                run_inputs: Mapping[str, Any],
            ) -> Any:
                attempt = pipeline._register_stage_start(stage.name)
                pipeline.observability.stage_started(stage.name, attempt)
                context = pipeline._build_stage_context(stage, stage_inputs, run_inputs, attempt)
                try:
                    output = await pipeline._run_agent_with_context(stage.name, context)
                except Exception as exc:  # pragma: no cover - surfaced to caller
                    pipeline.observability.stage_failed(stage.name, str(exc))
                    raise RuntimeError(f"Stage '{stage.name}' failed during agent execution: {exc}") from exc
                return output

        return _Strategy()

    def _build_stages(self) -> Tuple[Stage, ...]:
        return tuple(builder() for builder in self._stage_builders)

    def _collect_initial_results(self, start_index: int) -> Dict[str, BaseModel]:
        return {
            name: self.results[name]
            for name in self._stage_order.names[:start_index]
            if name in self.results
        }

    def _discard_results_from(self, start_index: int) -> None:
        for name in self._stage_order.names[start_index:]:
            self.results.pop(name, None)

    def _register_stage_start(self, stage_name: str) -> int:
        self.stage_attempts[stage_name] += 1
        return self.stage_attempts[stage_name]

    async def _apply_feedback(
        self,
        completed_stage: str,
        run_inputs: Mapping[str, Any],
        attempt: int,
        error: Optional[str] = None,
    ) -> Any:
        if completed_stage not in self._feedback_stages:
            return "continue"

        # Check confidence level if stage completed successfully
        if error is None:
            confidence = self._get_stage_confidence(completed_stage)
            if confidence is not None and confidence >= 80.0:
                # High confidence - skip feedback
                return "continue"

        decision = await self._request_feedback(completed_stage, run_inputs, attempt, error)
        if decision is None:
            return "continue"

        return self._interpret_feedback(decision, completed_stage)

    async def _request_feedback(
        self,
        stage: str,
        run_inputs: Mapping[str, Any],
        attempt: int,
        error: Optional[str],
    ) -> Optional[FeedbackDecision]:
        feedback_context = self._build_feedback_context(stage, run_inputs, attempt, error)
        feedback_attempt = self.stage_attempts.get("feedback", 0) + 1
        self.stage_attempts["feedback"] = feedback_attempt
        self.observability.stage_started("feedback", feedback_attempt)
        try:
            decision_raw = await self._run_agent_with_context("feedback", feedback_context)
        except Exception as exc:
            self.observability.stage_failed("feedback", str(exc))
            return None

        try:
            decision = self._coerce_feedback(decision_raw)
        except Exception as exc:
            self.observability.stage_failed("feedback", f"Invalid feedback payload: {exc}")
            return None

        self.feedback_history.append(decision)
        self.observability.stage_completed("feedback", decision)
        return decision

    def _interpret_feedback(self, decision: FeedbackDecision, stage: str) -> Any:
        action = decision.action
        if action == "continue":
            return "continue"
        if action == "abort":
            return "abort"

        if action.startswith("retry"):
            target = decision.target_stage or action.replace("retry_", "", 1)
            if not target:
                target = stage

            # Handle verification phase retries
            verification_phases = {"lint", "test_generation", "simulation"}
            if target in verification_phases:
                retry_phase = target
                target = "verification"
                # Store the specific phase to retry in the decision
                decision.target_stage = target
                decision.guidance = f"retry_from_phase:{retry_phase}"
        elif action == "tune_microarch":
            target = decision.target_stage or "microarch"
        else:
            return "continue"

        if target not in self._stage_order.index:
            return "continue"

        return ("jump", self._stage_order.index[target])

    def _add_feedback_to_run_inputs(self, run_inputs: Mapping[str, Any]) -> Dict[str, Any]:
        """Add feedback information to run_inputs for stages that need it."""
        inputs = dict(run_inputs)
        if self.feedback_history:
            inputs["last_feedback"] = self.feedback_history[-1].model_dump()
        return inputs

    def _build_stage_context(
        self,
        stage: Stage,
        stage_inputs: Mapping[str, Any],
        run_inputs: Mapping[str, Any],
        attempt: int,
    ) -> Dict[str, Any]:
        workspace = self._get_workspace()
        context = {
            "stage": stage.name,
            "attempt": attempt,
            "stage_inputs": self._serialize_results(stage_inputs),
            "prior_results": self._serialize_results(self.results),
            "run_inputs": dict(run_inputs),
            "workspace_token": self.workspace_token,
            "workspace_files": workspace.list_files() if workspace else [],
            "observability": self.observability,
        }

        # Include last feedback decision if this is a retry
        if self.feedback_history:
            context["last_feedback"] = self.feedback_history[-1].model_dump()

        return context

    def _build_feedback_context(
        self,
        completed_stage: str,
        run_inputs: Mapping[str, Any],
        attempt: int,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        context = {
            "stage": completed_stage,
            "attempt": attempt,
            "results": self._serialize_results(self.results),
            "run_inputs": dict(run_inputs),
            "workspace_token": self.workspace_token,
            "observability": self.observability,
        }
        if error is not None:
            context["error"] = str(error)
        return context

    @staticmethod
    def _serialize_results(results: Mapping[str, Any]) -> Dict[str, Any]:
        serialised: Dict[str, Any] = {}
        for name, value in results.items():
            if isinstance(value, BaseModel):
                serialised[name] = value.model_dump()
                # Debug: Check quantized_coefficients specifically
                if name == "quant" and "quantized_coefficients" in serialised[name]:
                    coeffs = serialised[name]["quantized_coefficients"]
                    print(f"DEBUG: Serializing quant stage - coeffs type: {type(coeffs)}, length: {len(coeffs) if coeffs else 'None'}")
            else:
                serialised[name] = value
        return serialised

    @staticmethod
    def _coerce_feedback(raw_decision: Any) -> FeedbackDecision:
        if isinstance(raw_decision, FeedbackDecision):
            return raw_decision
        if isinstance(raw_decision, BaseModel):
            return FeedbackDecision(**raw_decision.model_dump())
        if isinstance(raw_decision, dict):
            return FeedbackDecision(**raw_decision)
        return FeedbackDecision(action="continue")

    async def _run_agent_with_context(
        self,
        agent_name: str,
        context: Mapping[str, Any],
    ) -> Any:
        if agent_name == "feedback":
            return await self._agent_runner.run_feedback(context)
        return await self._agent_runner.run_stage(agent_name, context)

    def _get_workspace(self) -> Optional[Workspace]:
        if not self.workspace_token:
            return None
        return workspace_manager.get_workspace(self.workspace_token)

    def _get_stage_confidence(self, stage_name: str) -> Optional[float]:
        """Extract confidence level from stage result."""
        if stage_name not in self.results:
            return None
        
        result = self.results[stage_name]
        if hasattr(result, 'confidence'):
            return result.confidence
        elif isinstance(result, dict) and 'confidence' in result:
            return result['confidence']
        
        return None


__all__ = ["Pipeline"]
