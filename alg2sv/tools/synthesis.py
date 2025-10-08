"""
Synthesis adapter producing placeholder reports.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..domain import SynthResults


def run_synthesis(context: Mapping[str, Any]) -> SynthResults:
    run_inputs = context.get("run_inputs", {})
    backend = run_inputs.get("synthesis_backend", "auto")

    _emit_tool_event(context, "synth", "synth-backend", {"backend": backend})

    return SynthResults(
        fmax_mhz=220.0,
        timing_met=True,
        lut_usage=4200,
        ff_usage=8100,
        dsp_usage=28,
        bram_usage=12,
        total_power_mw=48.0,
        slack_ns=0.65,
        reports_path=f"{backend}_reports",
    )


def _emit_tool_event(context: Mapping[str, Any], stage: str, tool_name: str, metadata: Mapping[str, Any]) -> None:
    observability = context.get("observability")
    if observability is not None:
        try:
            observability.tool_invoked(stage, tool_name, dict(metadata))
        except Exception:
            pass
