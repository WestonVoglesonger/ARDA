"""
Micro-architecture configuration models for ARDA.
"""

from typing import Any, Dict
from pydantic import BaseModel, Field


class MicroArchConfig(BaseModel):
    """Micro-architecture configuration."""

    pipeline_depth: int
    unroll_factor: int
    memory_config: Dict[str, Any]
    dsp_usage_estimate: int
    estimated_latency_cycles: int
    handshake_protocol: str
    confidence: float = Field(default=85.0, ge=0, le=100, description="Confidence level (0-100%)")

    class Config:
        extra = "allow"
