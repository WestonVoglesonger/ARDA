"""
Evaluation result models for ARDA.
"""

from typing import List
from pydantic import BaseModel, Field


class EvaluateResults(BaseModel):
    """Comprehensive evaluation of the design."""

    overall_score: float  # 0-100
    performance_score: float  # timing, throughput, latency
    resource_score: float  # efficiency vs requirements
    quality_score: float  # code quality, verification completeness
    correctness_score: float  # functional accuracy
    recommendations: List[str]
    bottlenecks: List[str]
    optimization_opportunities: List[str]
    confidence: float = Field(default=85.0, ge=0, le=100, description="Confidence level (0-100%)")
