"""
Synthesis stage enforcing timing gates.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import Stage, StageContext
from ...domain import SynthResults, RTLConfig, LintResults, VerifyResults


class SynthStage(Stage):
    """Run synthesis and ensure timing goals are met."""

    name = "synth"
    dependencies = ("rtl", "static_checks", "verification")
    output_model = SynthResults

    def gather_inputs(self, context: StageContext) -> Dict[str, Any]:
        inputs = super().gather_inputs(context)
        if not isinstance(inputs["rtl"], RTLConfig):
            raise TypeError("SynthStage requires RTLConfig from 'rtl' dependency.")
        if not isinstance(inputs["static_checks"], LintResults):
            raise TypeError("SynthStage requires LintResults from 'static_checks' dependency.")
        if not isinstance(inputs["verification"], VerifyResults):
            raise TypeError("SynthStage requires VerifyResults from 'verification' dependency.")
        return inputs

    def validate_output(self, output: SynthResults, context: StageContext) -> None:
        if not output.timing_met:
            raise ValueError(
                "Synthesis gate failed: timing not met (fmax={}MHz)".format(
                    output.fmax_mhz
                )
            )
