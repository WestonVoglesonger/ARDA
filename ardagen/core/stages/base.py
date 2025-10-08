"""
Base stage definitions for the ARDA orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple, Type, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..strategies import AgentStrategy


@dataclass
class StageContext:
    """Runtime context handed to each stage execution."""

    run_inputs: Mapping[str, Any]
    results: Dict[str, BaseModel]


class Stage:
    """Base class for pipeline stages executed by the orchestrator."""

    name: str = "stage"
    dependencies: Tuple[str, ...] = ()
    output_model: Type[BaseModel] = BaseModel

    def gather_inputs(self, context: StageContext) -> Dict[str, Any]:
        """Collect inputs required for stage execution."""
        inputs: Dict[str, Any] = {}
        for dep in self.dependencies:
            if dep not in context.results:
                raise KeyError(f"Dependency '{dep}' missing for stage '{self.name}'")
            inputs[dep] = context.results[dep]
        return inputs

    async def run(self, context: StageContext, strategy: "AgentStrategy") -> BaseModel:
        """Execute the stage using the provided strategy."""
        stage_inputs = self.gather_inputs(context)
        raw_output = await strategy.run(self, stage_inputs, context.run_inputs)
        output = self._coerce_output(raw_output)
        self.validate_output(output, context)
        return output

    def _coerce_output(self, raw_output: Any) -> BaseModel:
        """Convert raw strategy output into the expected Pydantic model."""
        if isinstance(raw_output, self.output_model):
            return raw_output
        if isinstance(raw_output, dict):
            return self.output_model(**raw_output)
        raise TypeError(
            f"Stage '{self.name}' expected {self.output_model.__name__} or dict, "
            f"got {type(raw_output)}"
        )

    def describe(self) -> str:
        """Human-readable description of the stage."""
        return self.__class__.__doc__ or self.name

    def validate_output(self, output: BaseModel, context: StageContext) -> None:
        """Hook for subclasses to enforce quality gates."""
        return None
