"""
Domain models for ARDA pipeline stages.
This module re-exports commonly used dataclasses/Pydantic models to provide
stable import paths while the architecture refactor progresses.
"""

from .contracts import SpecContract
from .quantization import QuantConfig
from .architecture import MicroArchConfig, ArchitectureConfig, ModuleSpec
from .rtl_artifacts import RTLConfig
from .verification import (
    VerifyResults,
    LintResults,
    SimulateResults,
)
from .synthesis import SynthResults
from .evaluation import EvaluateResults
from .feedback import FeedbackDecision

__all__ = [
    "SpecContract",
    "QuantConfig",
    "MicroArchConfig",
    "ArchitectureConfig",
    "ModuleSpec",
    "RTLConfig",
    "VerifyResults",
    "LintResults",
    "SimulateResults",
    "SynthResults",
    "EvaluateResults",
    "FeedbackDecision",
]
