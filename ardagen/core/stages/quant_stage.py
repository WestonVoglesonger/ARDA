"""
Quantization stage implementation for the ARDA orchestrator.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import Stage, StageContext
from ...domain import QuantConfig, SpecContract


class QuantStage(Stage):
    """Convert floating point contract into fixed-point configuration."""

    name = "quant"
    dependencies = ("spec",)
    output_model = QuantConfig

    def gather_inputs(self, context: StageContext) -> Dict[str, Any]:
        inputs = super().gather_inputs(context)
        spec = inputs.get("spec")
        if not isinstance(spec, SpecContract):
            raise TypeError("QuantStage requires SpecContract from 'spec' dependency.")
        return inputs
