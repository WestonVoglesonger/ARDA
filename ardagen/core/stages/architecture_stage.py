"""
Architecture planning stage for RTL generation.
"""

from __future__ import annotations
from typing import Any, Dict, TYPE_CHECKING
from pydantic import BaseModel

from .base import Stage, StageContext
from ...domain import ArchitectureConfig, MicroArchConfig, QuantConfig, SpecContract

if TYPE_CHECKING:
    from ...core.strategies import AgentStrategy


class ArchitectureStage(Stage):
    """
    Design RTL architecture: module decomposition, hierarchy, interfaces.
    
    Consumes: spec, quant, microarch
    Produces: ArchitectureConfig with module specifications
    """
    
    name = "architecture"
    dependencies = ("spec", "quant", "microarch")
    output_model = ArchitectureConfig
    
    def gather_inputs(self, context: StageContext) -> Dict[str, Any]:
        """Gather and validate inputs from upstream stages."""
        inputs = super().gather_inputs(context)
        
        # Validate input types
        if not isinstance(inputs["spec"], SpecContract):
            raise TypeError("ArchitectureStage requires SpecContract from 'spec'")
        if not isinstance(inputs["quant"], QuantConfig):
            raise TypeError("ArchitectureStage requires QuantConfig from 'quant'")
        if not isinstance(inputs["microarch"], MicroArchConfig):
            raise TypeError("ArchitectureStage requires MicroArchConfig from 'microarch'")
        
        return inputs

