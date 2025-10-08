"""
Reporting helpers for aggregating stage outputs.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..domain import EvaluateResults


def build_evaluation_summary(context: Mapping[str, Any]) -> EvaluateResults:
    results = context.get("prior_results", {})
    synth = results.get("synth", {})
    verification = results.get("verification", {})

    _emit_tool_event(context, "evaluate", "report-builder", {"sources": list(results.keys())})

    timing_met = synth.get("timing_met", True)
    all_passed = verification.get("all_passed", True)

    performance_score = 95.0 if timing_met else 70.0
    quality_score = 94.0 if all_passed else 65.0

    return EvaluateResults(
        overall_score=(performance_score + quality_score) / 2,
        performance_score=performance_score,
        resource_score=90.0,
        quality_score=quality_score,
        correctness_score=quality_score,
        recommendations=[
            "Review synthesized timing if targets tighten.",
            "Capture additional verification vectors for broader coverage.",
        ],
        bottlenecks=[],
        optimization_opportunities=[],
    )


def _emit_tool_event(context: Mapping[str, Any], stage: str, tool_name: str, metadata: Mapping[str, Any]) -> None:
    observability = context.get("observability")
    if observability is not None:
        try:
            observability.tool_invoked(stage, tool_name, dict(metadata))
        except Exception:
            pass
