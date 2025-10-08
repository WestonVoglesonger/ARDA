"""
Static checks stage enforcing code quality gates.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import Stage, StageContext
from ...domain import RTLConfig, LintResults


class StaticChecksStage(Stage):
    """Run lint analysis on generated RTL and enforce cleanliness."""

    name = "static_checks"
    dependencies = ("rtl",)
    output_model = LintResults

    def gather_inputs(self, context: StageContext) -> Dict[str, Any]:
        inputs = super().gather_inputs(context)
        if not isinstance(inputs["rtl"], RTLConfig):
            raise TypeError("StaticChecksStage requires RTLConfig from 'rtl' dependency.")
        return inputs

    def validate_output(self, output: LintResults, context: StageContext) -> None:
        if not output.lint_clean or output.critical_issues > 0:
            raise ValueError("Static checks gate failed: critical issues detected in RTL output")
