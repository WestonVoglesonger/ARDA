"""
Feedback decision models for ARDA.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class FeedbackDecision(BaseModel):
    """Decision produced by the feedback agent after reviewing stage outputs."""

    action: Literal[
        "continue",
        "retry_spec",
        "retry_quant",
        "retry_microarch",
        "retry_architecture",
        "retry_rtl",
        "retry_verify",
        "retry_lint",
        "retry_test_generation",
        "retry_simulation",
        "retry_synth",
        "retry_evaluate",
        "tune_microarch",
        "abort",
    ] = Field(
        description=(
            "Requested pipeline action. Use retry_<stage> to rerun a specific stage, "
            "tune_microarch to revisit micro-architecture design, or abort to stop the pipeline."
        )
    )
    target_stage: Optional[str] = Field(
        default=None,
        description="Specific stage this decision applies to (e.g., 'synth').",
    )
    guidance: Optional[str] = Field(
        default=None,
        description="Additional instructions or context for the targeted stage retry or adjustment.",
    )
    notes: Optional[List[str]] = Field(
        default=None,
        description="Optional structured notes for logging or downstream analysis.",
    )
