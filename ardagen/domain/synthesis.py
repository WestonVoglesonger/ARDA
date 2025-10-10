"""
Synthesis result models for ARDA.
"""

from typing import Optional
from pydantic import BaseModel, Field


class SynthResults(BaseModel):
    """Synthesis results."""

    fmax_mhz: float
    timing_met: bool
    lut_usage: int
    ff_usage: int
    dsp_usage: int
    bram_usage: int
    total_power_mw: Optional[float] = None
    slack_ns: float
    reports_path: str
    confidence: float = Field(default=90.0, ge=0, le=100, description="Confidence level (0-100%)")
