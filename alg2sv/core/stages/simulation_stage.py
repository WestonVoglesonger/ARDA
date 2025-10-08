"""
Verification stage enforcing functional gates.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import Stage, StageContext
from ...domain import RTLConfig, VerifyResults


class VerificationStage(Stage):
    """Run RTL verification flows and ensure no regressions are present."""

    name = "verification"
    dependencies = ("rtl",)
    output_model = VerifyResults

    def gather_inputs(self, context: StageContext) -> Dict[str, Any]:
        inputs = super().gather_inputs(context)
        if not isinstance(inputs["rtl"], RTLConfig):
            raise TypeError("VerificationStage requires RTLConfig from 'rtl' dependency.")
        return inputs

    def validate_output(self, output: VerifyResults, context: StageContext) -> None:
        if not output.all_passed or output.tests_passed < output.tests_total:
            raise ValueError("Verification gate failed: functional mismatches detected")
