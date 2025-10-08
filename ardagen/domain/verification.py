"""
Verification-related result models for ARDA.
"""

from typing import Any, Dict, List
from pydantic import BaseModel


class VerifyResults(BaseModel):
    """Verification results."""

    tests_total: int
    tests_passed: int
    all_passed: bool
    mismatches: List[Dict[str, Any]]
    max_abs_error: float
    rms_error: float
    functional_coverage: float


class LintResults(BaseModel):
    """Linting results for SystemVerilog code."""

    syntax_errors: int
    style_warnings: int
    lint_violations: int
    critical_issues: int
    issues_list: List[Dict[str, Any]]
    overall_score: float  # 0-100 quality score
    lint_clean: bool


class SimulateResults(BaseModel):
    """RTL simulation results."""

    test_passed: int
    test_failed: int
    test_total: int
    coverage_percent: float
    timing_violations: int
    simulation_errors: List[str]
    waveform_available: bool
    simulation_time_ms: float
