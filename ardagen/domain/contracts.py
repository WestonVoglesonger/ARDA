"""
Hardware contract specification models used across the ARDA pipeline.
"""

from typing import Any, Dict
from pydantic import BaseModel, Field


class SpecContract(BaseModel):
    """Hardware contract specification."""

    name: str
    description: str
    clock_mhz_target: float
    throughput_samples_per_cycle: int
    input_format: Dict[str, Any] = Field(description="width and fractional_bits")
    output_format: Dict[str, Any] = Field(description="width and fractional_bits")
    resource_budget: Dict[str, Any] = Field(description="lut, ff, dsp, bram budgets")
    verification_config: Dict[str, Any] = Field(description="test parameters")
    confidence: float = Field(default=90.0, ge=0, le=100, description="Confidence level (0-100%)")

    class Config:
        extra = "allow"  # Allow additional fields for extensibility
