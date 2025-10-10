"""
Architecture configuration domain models for ARDA.

Contains both micro-architecture (MicroArchConfig) and architectural decomposition (ArchitectureConfig) models.
"""

from pydantic import BaseModel, Field, validator
from typing import Any, List, Dict, Optional


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


class ModuleSpec(BaseModel):
    """Specification for a single RTL module."""
    
    name: str = Field(description="Module name (e.g., 'conv2d_pe', 'fir_mac_pipeline')")
    purpose: str = Field(description="One-sentence description of module purpose")
    file_name: str = Field(description="File name (e.g., 'conv2d_pe.sv')")
    estimated_lines: int = Field(ge=20, le=300, description="Estimated lines of code")
    
    # Module interface
    inputs: List[Dict[str, str]] = Field(description="Input ports: [{name, width, description}]")
    outputs: List[Dict[str, str]] = Field(description="Output ports")
    parameters: Optional[List[Dict[str, str]]] = Field(default=None)
    
    # Dependencies
    instantiates: List[str] = Field(default=[], description="List of sub-modules this module instantiates")
    
    class Config:
        extra = "allow"


class ArchitectureConfig(BaseModel):
    """Complete architecture specification."""
    
    # High-level architecture
    architecture_type: str = Field(description="Architecture pattern (e.g., 'pipelined_fir', 'systolic_array', 'butterfly_network')")
    decomposition_rationale: str = Field(description="Why this decomposition was chosen")
    
    # Module specifications
    modules: List[ModuleSpec] = Field(min_items=3, max_items=15)
    
    # Hierarchy
    top_module: str = Field(description="Name of top-level module")
    hierarchy_diagram: str = Field(description="ASCII art or text description of module hierarchy")
    
    # File organization
    parameters_file: str = Field(default="params.svh", description="Parameters package file")
    
    # Design decisions
    pipeline_stages: int = Field(ge=1, le=20)
    parallelism_factor: int = Field(ge=1, le=64, description="Degree of parallel processing")
    memory_architecture: str = Field(description="Memory organization (e.g., 'distributed_regs', 'bram_buffers', 'fifo_chain')")
    
    # Metadata
    confidence: float = Field(ge=0, le=100)
    research_sources: List[str] = Field(default=[], description="URLs or references consulted")
    
    @validator('modules')
    def validate_hierarchy(cls, v, values):
        """Ensure top module exists and hierarchy is valid."""
        if 'top_module' in values:
            top = values['top_module']
            module_names = [m.name for m in v]
            if top not in module_names:
                raise ValueError(f"Top module '{top}' not found in modules list")
        return v
    
    @validator('modules')
    def validate_no_cycles(cls, v):
        """Check for circular dependencies in module instantiation."""
        # Build dependency graph and check for cycles
        graph = {m.name: set(m.instantiates) for m in v}
        
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False
        
        visited = set()
        for module in v:
            if module.name not in visited:
                if has_cycle(module.name, visited, set()):
                    raise ValueError(f"Circular dependency detected in module hierarchy")
        
        return v

    class Config:
        extra = "allow"
