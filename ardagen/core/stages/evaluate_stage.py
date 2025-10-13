"""
Evaluation stage for orchestrator summary.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import Stage, StageContext
from ...domain import (
    EvaluateResults,
    SpecContract,
    QuantConfig,
    MicroArchConfig,
    RTLConfig,
    LintResults,
    VerifyResults,
    SynthResults,
)


class EvaluateStage(Stage):
    """Aggregate metrics across stages for final scoring."""

    name = "evaluate"
    dependencies = ("spec", "quant", "microarch", "rtl", "verification", "synth")
    output_model = EvaluateResults
    # Note: static_checks removed - lint results now in verification.lint_results

    def gather_inputs(self, context: StageContext) -> Dict[str, Any]:
        inputs = super().gather_inputs(context)
        dependency_types = {
            "spec": SpecContract,
            "quant": QuantConfig,
            "microarch": MicroArchConfig,
            "rtl": RTLConfig,
            "verification": VerifyResults,
            "synth": SynthResults,
        }
        for name, expected in dependency_types.items():
            value = inputs.get(name)
            if not isinstance(value, expected):
                raise TypeError(f"EvaluateStage requires {expected.__name__} from '{name}'")
        return inputs
