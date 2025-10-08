"""
Quantization configuration models for ARDA fixed-point stages.
"""

from typing import Any, Dict, List
from pydantic import BaseModel


class QuantConfig(BaseModel):
    """Fixed-point quantization configuration."""

    fixed_point_config: Dict[str, Any]
    error_metrics: Dict[str, Any]
    quantized_coefficients: List[float]
    fxp_model_path: str

    class Config:
        extra = "allow"
