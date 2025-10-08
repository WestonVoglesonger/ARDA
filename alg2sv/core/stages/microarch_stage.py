"""
Micro-architecture stage implementation for the ARDA orchestrator.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import Stage, StageContext
from ...domain import MicroArchConfig, QuantConfig, SpecContract


class MicroArchStage(Stage):
    """Design the micro-architecture informed by spec and quantization."""

    name = "microarch"
    dependencies = ("spec", "quant")
    output_model = MicroArchConfig

    def gather_inputs(self, context: StageContext) -> Dict[str, Any]:
        inputs = super().gather_inputs(context)
        spec = inputs.get("spec")
        quant = inputs.get("quant")
        if not isinstance(spec, SpecContract):
            raise TypeError("MicroArchStage requires SpecContract from 'spec' dependency.")
        if not isinstance(quant, QuantConfig):
            raise TypeError("MicroArchStage requires QuantConfig from 'quant' dependency.")
        return inputs
