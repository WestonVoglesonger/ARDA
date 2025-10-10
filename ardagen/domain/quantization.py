"""
Quantization configuration models for ARDA fixed-point stages.
"""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class QuantConfig(BaseModel):
    """Fixed-point quantization configuration."""

    fixed_point_config: Dict[str, Any]
    error_metrics: Dict[str, Any]
    quantized_coefficients: List[float]
    fxp_model_path: str
    confidence: float = Field(default=90.0, ge=0, le=100, description="Confidence level (0-100%)")

    class Config:
        extra = "allow"
