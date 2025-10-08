"""
RTL generation stage for the ARDA orchestrator.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import Stage, StageContext
from ...domain import RTLConfig, MicroArchConfig, QuantConfig, SpecContract


class RTLStage(Stage):
    """Generate SystemVerilog implementation based on upstream design decisions."""

    name = "rtl"
    dependencies = ("spec", "quant", "microarch")
    output_model = RTLConfig

    def gather_inputs(self, context: StageContext) -> Dict[str, Any]:
        inputs = super().gather_inputs(context)
        if not isinstance(inputs["spec"], SpecContract):
            raise TypeError("RTLStage requires SpecContract from 'spec' dependency.")
        if not isinstance(inputs["quant"], QuantConfig):
            raise TypeError("RTLStage requires QuantConfig from 'quant' dependency.")
        if not isinstance(inputs["microarch"], MicroArchConfig):
            raise TypeError("RTLStage requires MicroArchConfig from 'microarch' dependency.")
        return inputs
