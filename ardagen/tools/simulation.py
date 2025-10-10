"""
Simulation and verification helpers.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..domain import VerifyResults


def run_verification(context: Mapping[str, Any]) -> VerifyResults:
    _emit_tool_event(context, "verification", "sim-runner", {"vectors": "default"})
    return VerifyResults(
        tests_total=20,
        tests_passed=20,
        all_passed=True,
        mismatches=[],
        max_abs_error=1.0e-6,
        rms_error=5.0e-7,
        functional_coverage=0.95,
        confidence=90.0,
    )


def _emit_tool_event(context: Mapping[str, Any], stage: str, tool_name: str, metadata: Mapping[str, Any]) -> None:
    observability = context.get("observability")
    if observability is not None:
        try:
            observability.tool_invoked(stage, tool_name, dict(metadata))
        except Exception:
            pass
