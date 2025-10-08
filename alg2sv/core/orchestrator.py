"""
Pipeline orchestrator that coordinates stage execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Dict, Iterable, List, Mapping, Optional, Tuple

from .stages import Stage, StageContext
from .strategies import AgentStrategy


class StageExecutionError(RuntimeError):
    """Exception raised when a stage fails during orchestrator execution."""

    def __init__(self, stage: str, error: Exception):
        super().__init__(f"Stage '{stage}' failed: {error}")
        self.stage = stage
        self.error = error


@dataclass
class PipelineRunResult:
    """Aggregate results from orchestrator execution."""

    results: Dict[str, object]
    stages: List[str]

    def get(self, stage_name: str):
        return self.results.get(stage_name)


class PipelineOrchestrator:
    """Simple orchestrator that executes stages sequentially using a strategy."""

    def __init__(self, stages: Iterable[Stage], strategy: AgentStrategy):
        self._stages: List[Stage] = list(stages)
        self._strategy = strategy

    async def run(
        self,
        run_inputs: Optional[Mapping[str, object]] = None,
        initial_results: Optional[Mapping[str, object]] = None,
    ) -> PipelineRunResult:
        run_inputs = run_inputs or {}
        context = StageContext(run_inputs=run_inputs, results=dict(initial_results or {}))
        executed: List[str] = []

        for stage in self._stages:
            if stage.name in context.results:
                continue

            missing = [dep for dep in stage.dependencies if dep not in context.results]
            if missing:
                raise RuntimeError(
                    f"Stage '{stage.name}' missing dependencies: {', '.join(missing)}"
                )

            try:
                output = await stage.run(context, self._strategy)
            except StageExecutionError:
                raise
            except Exception as exc:
                raise StageExecutionError(stage.name, exc) from exc
            context.results[stage.name] = output
            executed.append(stage.name)

        return PipelineRunResult(results=context.results, stages=executed)

    async def run_iter(
        self,
        run_inputs: Optional[Mapping[str, object]] = None,
        initial_results: Optional[Mapping[str, object]] = None,
    ) -> AsyncIterator[Tuple[str, object]]:
        """
        Execute stages sequentially, yielding after each stage completes.

        Yields:
            Tuple[str, object]: Stage name and its result model.
        """
        run_inputs = run_inputs or {}
        context = StageContext(run_inputs=run_inputs, results=dict(initial_results or {}))

        for stage in self._stages:
            if stage.name in context.results:
                continue

            missing = [dep for dep in stage.dependencies if dep not in context.results]
            if missing:
                raise RuntimeError(
                    f"Stage '{stage.name}' missing dependencies: {', '.join(missing)}"
                )

            try:
                output = await stage.run(context, self._strategy)
            except StageExecutionError:
                raise
            except Exception as exc:
                raise StageExecutionError(stage.name, exc) from exc
            context.results[stage.name] = output
            yield stage.name, output
