"""
Micro-architecture configuration models for ARDA.
"""

from typing import Any, Dict
from pydantic import BaseModel


class MicroArchConfig(BaseModel):
    """Micro-architecture configuration."""

    pipeline_depth: int
    unroll_factor: int
    memory_config: Dict[str, Any]
    dsp_usage_estimate: int
    estimated_latency_cycles: int
    handshake_protocol: str

    class Config:
        extra = "allow"
